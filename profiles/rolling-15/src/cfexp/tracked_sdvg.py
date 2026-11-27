"""New SDVG launches: server-side W&B and an isolated bounded profiling mode.

The existing campaign and numerical backends are called unchanged. CUDA traces
and synchronization are enabled only in the separate profiling campaign, never
in the main timing cohort. Only the controller owns a W&B run.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ENV = 'CFEXP_SDVG_TRACKING'
PROFILE = 'CFEXP_SDVG_BOUNDED_PROFILE'


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, allow_nan=False, sort_keys=True, indent=2)
        f.write('\n')



def materialize_sdk_log_links(directory):
    """Preserve the SDK core log as bytes before the unsealed run is indexed.

    W&B 0.19.11 respects symlink=False for latest-run/user logs but its core
    backend still creates logs/debug-core.log. Do not weaken the artifact
    walker, follow arbitrary links, or repair a sealed historical run.
    """
    import shutil
    from .stage_timing import _unsealed_directory
    directory = Path(directory)
    _unsealed_directory(directory)
    for link in directory.rglob('*'):
        if not link.is_symlink():
            continue
        target = link.resolve(strict=True)
        if (link.name != 'debug-core.log' or link.parent.name != 'logs'
                or not target.is_file() or not target.name.startswith('core-debug-')
                or target.suffix != '.log' or target.stat().st_uid != os.getuid()):
            raise ValueError('unexpected SDK symlink; preserve run for inspection')
        temporary = link.with_name(link.name+'.materialized')
        with temporary.open('xb') as output, target.open('rb') as source:
            shutil.copyfileobj(source, output)
        os.replace(temporary, link)


def tracking_mode(env):
    mode = env.get('CFEXP_WANDB_MODE', 'auto')
    if mode not in ('auto', 'offline', 'online'):
        raise ValueError('W&B mode must be auto, offline or online; disabled is not supported')
    if mode == 'auto':
        return 'online' if env.get('WANDB_API_KEY') else 'offline'
    return mode


def process_factory(argv, **kwargs):
    import subprocess
    if os.environ.get(ENV) != '1' or list(argv[1:3]) != ['-m', 'cfexp.correction_worker']:
        raise ValueError('tracked SDVG requires its explicit standard-worker entry')
    command = list(argv)
    command[2] = 'cfexp.tracked_sdvg_worker'
    return subprocess.Popen(command, **kwargs)


def launch_draft(**kwargs):
    from .native_sdvg import launch_conditioned_draft_worker
    kwargs['process_factory'] = process_factory
    return launch_conditioned_draft_worker(**kwargs)


def launch_target(**kwargs):
    from .native_sdvg import launch_conditioned_target_worker
    kwargs['process_factory'] = process_factory
    return launch_conditioned_target_worker(**kwargs)


def observed_worker_class(base):
    """All timing/memory is attributed to the owning worker, never summed as
    if two allocator peaks were a simultaneous physical-device peak.
    """
    class Observed(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            from .stage_timing import _unsealed_directory
            self._tracking_root = _unsealed_directory(self.run_dir)
            self._tracking_forward_calls = 0
            self._tracking_profiled = False
            self._tracking_sequence = 0
            self._tracking_diagnostic = os.environ.get(PROFILE, '0') == '1'
            import torch
            if not torch.cuda.is_available():
                raise ValueError('GPU telemetry requires actual CUDA; no fabricated CPU memory')
            self._tracking_torch = torch
            torch.cuda.reset_peak_memory_stats()
            self._tracking_path = self._tracking_root/'provenance'/'tracking'/f'{self.role}-{os.getpid()}.jsonl'
            self._tracking_path.parent.mkdir(parents=True, exist_ok=True)
            # Fail if another invocation already used this writer identity.
            self._tracking_path.open('x').close()

        def handle(self, request):
            torch = self._tracking_torch
            forward = request.get('action') in ('generate', 'generate_persistent')
            selected = (self._tracking_diagnostic and forward
                and self._tracking_forward_calls >= 2 and not self._tracking_profiled)
            profiler = None
            if selected:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                    profile_memory=True, with_stack=False)
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            if selected:
                start_event.record()
            with profiler if profiler is not None else nullcontext():
                value = super().handle(request)
            if selected:
                end_event.record()
                end_event.synchronize()
            elapsed = time.perf_counter()-before
            self._tracking_sequence += 1
            row = dict(schema='sdvg_worker_observation_v1', role=self.role, pid=os.getpid(),
                rank=0, sequence=self._tracking_sequence, action=request.get('action'),
                request_id=request.get('request_id'), request_wall_seconds=elapsed,
                chunk_index=request.get('chunk_index'),
                warmup_forward_calls_before=self._tracking_forward_calls,
                bounded_profile_sample=selected,
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                memory_scope='this worker allocator; not a sum or physical-device simultaneous peak',
                wall_scope='worker request incl host/history/compute; not compute-only FPS')
            if selected:
                from .kernel_profile import summarize_events
                kernels = summarize_events(profiler.events())
                if not kernels:
                    raise ValueError('bounded CUDA profiler captured no CUDA kernels')
                trace = self._tracking_root/'tracking'/'profiler'/f'{self.role}-{os.getpid()}.trace.json'
                trace.parent.mkdir(parents=True, exist_ok=True)
                if trace.exists():
                    raise ValueError('refusing to overwrite profiler trace')
                profiler.export_chrome_trace(str(trace))
                row.update(cuda_stream_elapsed_seconds=start_event.elapsed_time(end_event)/1000,
                    cuda_kernel_us_sum=sum(k['cuda_us'] for k in kernels if k['kind']=='kernel'),
                    cuda_transfer_us_sum=sum(k['cuda_us'] for k in kernels if k['kind']=='transfer'),
                    trace_path=trace.relative_to(self._tracking_root).as_posix(),
                    trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    profile_scope='third-or-later forward after two warmup calls; instrumented, not main E2E timing')
                # These samples are continuation blocks: 3 latent frames map
                # to 12 comparison RGB frames. Stream elapsed includes gaps,
                # so this is a scoped throughput, not a kernel-busy FPS claim.
                row['comparison_frame_equivalent_count'] = 12
                row['stream_window_frame_equivalents_per_second'] = 12/row['cuda_stream_elapsed_seconds']
                self._tracking_profiled = True
            with self._tracking_path.open('a') as f:
                f.write(json.dumps(row, allow_nan=False, sort_keys=True)+'\n')
            if forward:
                self._tracking_forward_calls += 1
            return value
    return Observed


def log_result(run, root, result):
    raw = root/result['raw_rows']['path']
    if hashlib.sha256(raw.read_bytes()).hexdigest() != result['raw_rows']['sha256']:
        raise ValueError('W&B metric rows differ from campaign output')
    rows = json.loads(raw.read_text())['rows']
    for i, row in enumerate(rows):
        metrics = {f'quality/{k}': v for k, v in row['quality_scores'].items()}
        metrics.update(prompt_index=i, generation_seed=row['seed'],
            **{'quality/six_dimension_mean_100': sum(row['quality_scores'].values())*100/6,
               'timing/generation_seconds': row['generation_wall_seconds']})
        if 'video_ready_wall_seconds' in row:
            metrics['timing/video_ready_seconds'] = row['video_ready_wall_seconds']
            metrics['throughput/video_ready_fps'] = (result['duration']['frame_count'])/row['video_ready_wall_seconds']
        run.log(metrics)
    for path in sorted((root/'provenance'/'tracking').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line); prefix = f'worker/{row["role"]}/{row["action"]}'
            run.log({f'{prefix}/{k}': row[k] for k in ('request_wall_seconds',
                'peak_allocated_bytes','peak_reserved_bytes')})
            if row['bounded_profile_sample']:
                run.log({f'profile/{row["role"]}/{k}': row[k] for k in
                    ('cuda_stream_elapsed_seconds', 'cuda_kernel_us_sum', 'cuda_transfer_us_sum',
                     'warmup_forward_calls_before', 'peak_allocated_bytes', 'peak_reserved_bytes',
                     'stream_window_frame_equivalents_per_second')})
    run.log({f'timing/{key}':result[key] for key in
             ('draft_model_load_seconds','target_model_load_seconds','vbench_wall_seconds')
             if key in result})
    return rows


def require_profile_samples(root, *, config, result):
    samples = []
    for path in (root/'provenance'/'tracking').glob('*.jsonl'):
        samples.extend(json.loads(line) for line in path.read_text().splitlines()
                       if json.loads(line)['bounded_profile_sample'])
    roles = [r['role'] for r in samples]
    if roles.count('draft') != 1 or roles.count('target') not in (0,1) or len(samples) != len(set(roles)):
        raise ValueError('bounded profile needs one Draft sample and at most one Target sample')
    write_new(root/'provenance'/'tracking'/'profile-summary.json',dict(
        schema='sdvg_bounded_profile_summary_v1', executed=True, samples=samples,
        config=config, hardware=result['hardware'],
        coverage_complete_for_this_campaign=set(roles)=={'draft','target'},
        missing_post_warmup_roles=sorted({'draft','target'}-set(roles)),
        sparse_target_note=('Target was not called after its two warmup generation requests. '
            'No Target latency is imputed; a separately verified identical Target operator '
            'profile is required before main dispatch.' if 'target' not in roles else None),
        device_count=1, memory_scope='per-process allocator peaks; physical simultaneous peak not measured',
        purpose='separate diagnostic; never substitute these instrumented latencies into main speedup'))


def main(argv=None):
    from . import main_table_sdvg_campaign as campaign
    from .stage_timing import _unsealed_directory
    from .metadata import collect_git_metadata, find_git_root
    if os.environ.get(ENV) != '1' or os.environ.get(PROFILE, '0') not in ('0','1'):
        raise ValueError('tracked SDVG requires explicit valid tracking/profile flags')
    args = campaign._build_parser().parse_args(argv)
    diagnostic = os.environ.get(PROFILE, '0') == '1'
    if diagnostic != (args.smoke_prompt_id is not None):
        raise ValueError('bounded profile must use a separate one-prompt campaign; main must use full shard')
    root = _unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']))
    repo = find_git_root(Path.cwd())
    mode = tracking_mode(os.environ)
    config = dict(method='sdvg', phase='bounded_profile' if diagnostic else 'main',
        git_sha=collect_git_metadata(repo)['sha'], model_pair=args.draft_model_id,
        generation_seed=int(os.environ['CFEXP_GENERATION_SEED']),
        source_population_sha256='8a2f89312402c2688cb00074194d748e264dd47e5a2f2e6fe010ca2afb0d9d94',
        prompt_shard=int(os.environ['CFEXP_PAPER_PROMPT_SHARD_INDEX']),
        nominal_seconds=args.duration_seconds, physical_device_count=1, rank=0,
        batch_size=1, output_resolution=[480,832], output_fps=24,
        chunk_count=args.duration_seconds*2, root_target=True, threshold=-0.7,
        draft_history='independent proposals', target_history='accepted outputs',
        warmup_forward_calls_per_worker=2, native_kv=True,
        optimizer='not applicable: inference', loss='not applicable: inference',
        learning_rate='not applicable: inference', gradient_norm='not applicable: inference',
        backward='not applicable: inference', distributed_communication='not applicable: one device',
        precision='model-specific; frozen artifact adapter_config is authoritative',
        qualitative_comparison='fixed original prompt/seed and local video/provenance paths; media not auto-uploaded')
    for label, path in [('draft',args.draft_artifact_spec),('target',args.target_artifact_spec),
                        ('sdvg',args.sdvg_artifact_spec)]:
        value = json.loads(path.read_text())
        config[label+'_artifact_spec_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        config[label+'_adapter_config'] = value.get('adapter_config',{})
    import wandb
    directory = root/'tracking'/'wandb';directory.mkdir(parents=True,exist_ok=True)
    kwargs = dict(project='collaborative-forcing-reproduction', name=root.name,
        config=config, dir=str(directory), save_code=False,
        settings=wandb.Settings(init_timeout=25, disable_git=True, disable_code=True, symlink=False,
                                console='off', _disable_stats=True))
    fallback = False
    try:
        run = wandb.init(mode=mode, **kwargs)
    except Exception:
        if mode != 'online':
            raise
        # No credentials or error text is copied to provenance.
        fallback = True;mode='offline';run=wandb.init(mode='offline',**kwargs)
    write_new(root/'provenance'/'tracking'/'session.json',dict(schema='wandb_session_v1',
        mode=mode, online_fallback_to_offline=fallback, run_id=run.id,
        online_session_url=run.url if mode=='online' else None,
        synced_evidence_verified=False, directory='tracking/wandb', config=config))
    ok = False
    try:
        options = dict(vars(args))
        for name in ('draft_artifact_spec','target_artifact_spec','sdvg_artifact_spec'):
            options[name+'_path'] = options.pop(name)
        result = campaign.run_campaign(**options, draft_launcher=launch_draft,
                                       target_launcher=launch_target)
        if diagnostic:
            require_profile_samples(root, config=config, result=result)
        log_result(run, root, result)
        ok = True
        return 0
    finally:
        run.finish(exit_code=0 if ok else 1)
        materialize_sdk_log_links(directory)
        write_new(root/'provenance'/'tracking'/'finished.json',dict(schema='wandb_finish_v1',
            mode=mode, execution_succeeded=ok, synced_evidence_verified=False,
            note='Offline files remain on server; online remote history not independently queried.'))


if __name__ == '__main__':
    raise SystemExit(main())
