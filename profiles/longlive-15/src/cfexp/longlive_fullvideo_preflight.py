"""Paired complete-video LongLive FuseStep preflight, separate from main scores.

The original rank-burst campaign, noise, renderer and official scorer are used
unchanged. Two fixed cases are run in both orders on one physical H200. First
run separate, explicitly instrumented profiles; never mix those timings with
the unprofiled paired observations or interpret this small cohort as paper128.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from . import longlive_fusion_runtime as fusion
from .tracked_sdvg import (write_new, tracking_mode, materialize_sdk_log_links,
                           require_profile_samples, log_result)

NORMAL_SOURCE = 'c8e4d345bb747765ef72fb8490e3d26a46be4a7f'
NORMAL_ENV = 'CFEXP_LONGLIVE_NORMAL_GATE_RUN'
CALIBRATION_SOURCE = '3d27906622e66883a765e5672e186e984bb3968a'
CELL_ORDER = (('plain-profile', False, True), ('fused-profile', True, True),
              ('plain-first', False, False), ('fused-second', True, False),
              ('fused-first', True, False), ('plain-second', False, False))


def require(value, message):
    if not value:
        raise ValueError(message)


@contextmanager
def environment(values):
    """Restore controller flags even when a worker or scorer fails."""
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_options(env=None):
    env = os.environ if env is None else env
    for name in ('CFEXP_NATIVE_SINGLE_GPU',
                 'CFEXP_ACCEPTED_VIDEO_STREAM', 'CFEXP_RENDER_TARGET_PRIORITY',
                 'CFEXP_PAPER_REPRODUCTION', 'CFEXP_SDVG_TRACKING', 'CFEXP_STAGE_TIMING'):
        require(env.get(name) == '1', 'complete-video preflight requires '+name)
    require(env.get('CFEXP_NATIVE_VIDEO_STREAM_MODEL') == 'longlive'
            and env.get('CFEXP_GENERATION_SEED') == '20260911'
            and env.get('CFEXP_PAPER_PROMPT_SHARD_COUNT') == '64'
            and env.get('CFEXP_PAPER_PROMPT_SHARD_INDEX') == '0',
            'preflight must retain the two predetermined original cases and seed')
    require(env.get('CFEXP_SF_SINGLE_GPU', '0') == '0'
            and env.get('CFEXP_KREA_TIMESTEP_POLICY', 'main_comparison_v1') == 'main_comparison_v1',
            'preflight cannot change model ownership or Target sampling')
    require('CFEXP_TORCH_PROFILE_CHUNK' not in env, 'cannot nest the old chunk profiler')
    # This switch installs the Self-Forcing adapter. LongLive's owned native
    # executor already reuses KV and explicitly forbids that foreign adapter.
    for name in ('CFEXP_DRAFT_KV_REUSE','CFEXP_SF_TARGET_TWO_STEP',
                 'CFEXP_SF_TARGET_BURST_INTERIOR_ONE_STEP','CFEXP_SF_TARGET_BURST_EXIT_TWO_STEP',
                 'CFEXP_SF_TARGET_LOW_NOISE_REFINEMENT','CFEXP_SF_TARGET_EARLY_NATIVE_BURSTS',
                 'CFEXP_PAPER_C10_CALIBRATION','CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SEAL'):
        require(env.get(name,'0') == '0', 'preflight cannot install a foreign generation mode: '+name)
    require(bool(env.get(NORMAL_ENV)), 'original successful ordinary-worker gate is required')
    fusion.validate_flags(env)


def admit_normal_gate(repo, run):
    """Revalidate original sealed proof and exact sources without writing to it."""
    from .main_table_merge import _load_frozen_index
    from .longlive_fusion_gate import validate_report
    from .longlive_fusion_gate_oracle import cold_probe_trace
    frozen, marker = _load_frozen_index(Path(run))
    require(marker.get('status') == 'SUCCEEDED' and marker.get('exit_code') == 0
            and marker.get('git_sha') == NORMAL_SOURCE, 'normal-worker gate seal differs')
    frozen.verify_all()
    prefix = 'attempts/'+marker['attempt_id']+'/'
    def load(name):
        return json.loads(frozen.file(prefix+name, label=name).read_text())
    manifest = json.loads(frozen.file('resolved-manifest.json', label='manifest').read_text())['manifest']
    require(manifest['metadata']['name'] == 'longlive-fusion-normal-worker-profile-single-h200-gate-ntu-v1'
            and manifest['spec']['resources']['gpus'] == 1, 'wrong ordinary-worker gate workload')
    for item in manifest['spec']['inputs']:
        if item['uri'].startswith('repository://'):
            path = Path(repo)/item['target']
            require(path.is_file() and not path.is_symlink() and fusion.sha(path) == item['sha256'],
                    'normal-worker gate source changed: '+item['target'])
    driver = load('longlive-fusion-normal-worker-gate.json')
    require(driver.get('passed') is True and driver.get('physical_gpu_count') == 1
            and driver.get('normal_factory_installed') is True
            and driver.get('normal_rpc_probe_executed') is True, 'ordinary-worker gate incomplete')
    proof = driver['gate']
    files = proof['input_files']
    require(len(files) == 19 and files[:12] == [dict(path=x['path'], sha256=x['sha256'])
            for x in driver['target_history']], 'original Target/noise bindings changed')
    for item in files:
        require(fusion.sha(frozen.file(prefix+'live/'+item['path'], label='gate input')) == item['sha256'],
                'normal-worker input payload changed')
    validate_report(proof, files=files)
    profile = load('provenance/tracking/profile-summary.json')
    from .joint_gate_profile import validate_forward_records
    from .longlive_fusion_gate_oracle import expected_observed_forward_trace
    records=[]
    for path in (Path(run)/'attempts'/marker['attempt_id']/'live/provenance/tracking').glob('draft-forwards-*.jsonl'):
        relative=path.relative_to(Path(run)).as_posix()
        records.extend(json.loads(line) for line in frozen.file(relative,label='actual forward records').read_text().splitlines())
    samples=validate_forward_records(records,expected_observed_forward_trace())
    require(profile.get('executed_actual_cuda') is True
            and profile.get('original_full_oracle_passed') is True
            and profile.get('native_forward_count') == 416
            and profile.get('normal_rpc_forward_count') == 6
            and profile.get('total_draft_physical_forwards') == 422
            and profile.get('samples') == samples
            and load('live/provenance/tracking/normal-probe-forwards.json') == cold_probe_trace()
            and load('provenance/tracking/finished.json').get('execution_succeeded') is True,
            'normal-worker actual execution/profile evidence incomplete')
    return dict(run_id=marker['run_id'], source=NORMAL_SOURCE,
        artifact_index_sha256=marker['artifact_index_sha256'], ordinary_worker_gate_verified=True,
        total_draft_physical_forwards=422, no_full_video_speed_claim=True)


def process_factory(argv, **kwargs):
    command = list(argv)
    require(command[1:3] == ['-m', 'cfexp.correction_worker']
            and os.environ.get('CFEXP_SDVG_TRACKING') == '1', 'wrong telemetry worker entry')
    role = command[command.index('--role')+1]
    require(role in ('draft', 'target'), 'unknown worker role')
    index = command.index('--backend-factory')+1
    expected = ('cfexp.longlive_policy_worker:build_backend' if role == 'draft'
                else 'cfexp.correction_runtime:build_target_backend')
    require(command[index] == expected, 'preflight changed the native model factory')
    if os.environ.get(fusion.ENV) == '1':
        command = fusion.worker_argv(command)
    else:
        require(os.environ.get(fusion.ENV) == '0', 'explicit plain/fused cell required')
    command[2] = 'cfexp.tracked_sdvg_worker'
    return subprocess.Popen(command, **kwargs)


def launch_runtime(**kwargs):
    from .accepted_stream_worker import process_factory as ordinary
    from .correction_runtime import launch_correction_runtime
    require(kwargs.pop('process_factory') is ordinary
            and kwargs.get('draft_model_id') == 'longlive_1.3b', 'wrong coupled native runtime')
    return launch_correction_runtime(process_factory=process_factory, **kwargs)


def bound(root, binding):
    path = Path(root)/binding['path']
    require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve())
            and path.is_file() and fusion.sha(path) == binding['sha256'], 'artifact bytes changed')
    return path


def snapshot(root, result, *, fused):
    """Compare numerical/route identities, not allocator/session IDs or times."""
    raw = json.loads(bound(root, result['raw_rows']).read_text())['rows']
    ids = result['population']['selected_prompt_ids']
    require(len(ids) == len(set(ids)) == len(raw) == 2
            and {x['prompt_id'] for x in raw} == set(ids), 'preflight case coverage differs')
    checkpoints = {pid: {} for pid in ids}
    fused_chunks = 0; physical_calls = 0
    for path in (Path(root)/'state/rollouts').glob('*/chunk-*.json'):
        row = json.loads(path.read_text())
        pid = row['history']['prompt_id']
        generated = row['generated']; chunk = generated['segment']['chunk_index']
        require(pid in checkpoints and chunk not in checkpoints[pid], 'foreign/duplicate accepted checkpoint')
        segment = generated['segment']
        bound(root, dict(path=segment['latent_path'], sha256=segment['latent_sha256']))
        noise = generated['context']['noise']
        from .correction_state import InitialNoiseDescriptor
        InitialNoiseDescriptor.from_dict(noise)
        bound(root, dict(path=f'noise/{pid}/seed-20260911/chunk-{chunk:02d}.safetensors',
                         sha256=noise['tensor_sha256']))
        diagnostics = generated['diagnostics']
        if generated['context']['role'] == 'draft':
            evidence = diagnostics.get('longlive_joint_execution')
            require((evidence is not None) is fused, 'requested native execution was not installed')
            trace = diagnostics['forward_trace']
            require(len([x for x in trace if x['timestep']>0]) == 4,
                    'native Draft sampling steps changed')
            require(isinstance(diagnostics.get('rank_evidence'),dict)
                    and isinstance(diagnostics.get('kv_witness'),dict), 'Draft numerical witnesses missing')
            physical_calls += len(trace)
            if fused:
                fused_chunks += evidence['fused_history_chunks']
        checkpoints[pid][chunk] = dict(latent_sha256=segment['latent_sha256'],
            role=generated['context']['role'], noise=noise,
            rank_evidence=diagnostics.get('rank_evidence'),
            accepted_clean_witness=diagnostics.get('accepted_clean_witness'),
            kv_witness=diagnostics.get('kv_witness'))
    cases = {}
    for row in raw:
        pid = row['prompt_id']
        require(row['seed'] == 20260911 and set(checkpoints[pid]) == set(range(60)),
                'complete thirty-second accepted trajectory is required')
        provenance = json.loads((Path(root)/'provenance/ours'/f'{pid}.json').read_text())
        stream = json.loads(bound(root, provenance['accepted_stream']).read_text())
        bound(root, stream['video'])
        require(stream.get('decoder_and_encoder_joined') is True
                and stream.get('gpu_decode_deferred_until_finish') is True
                and stream['pixels']['shape'] == [717,3,480,832], 'wrong completed native video horizon')
        cases[pid] = dict(chunks=[checkpoints[pid][i] for i in range(60)],
            selected_target_chunks=provenance['accepted_target_chunks'],
            decisions=provenance['native_cache_audit']['decisions'],
            raw_rgb_sha256=stream['pixels']['raw_sha256'], mp4_sha256=stream['video']['sha256'])
    require(not fused or fused_chunks > 0, 'no actual Target history/first-Draft fusion observed')
    return dict(population=result['population'], calibration=result['calibration_receipt'],
                hardware=result['hardware'], cases=cases,
                execution=dict(fused_history_chunks=fused_chunks, observed_draft_forward_calls=physical_calls))


def compare(left, right):
    for key in ('population', 'calibration', 'hardware'):
        require(left[key] == right[key], 'paired '+key+' differs')
    require(left['cases'].keys() == right['cases'].keys(), 'paired cases differ')
    mp4 = {}
    for pid, original in left['cases'].items():
        candidate = right['cases'][pid]
        for key in ('chunks', 'selected_target_chunks', 'decisions', 'raw_rgb_sha256'):
            require(original[key] == candidate[key], 'complete-video parity failed: '+pid+' '+key)
        mp4[pid] = original['mp4_sha256'] == candidate['mp4_sha256']
    return dict(exact_accepted_latents_noise_rank_routing_and_raw_rgb=True,
                mp4_bytes_equal_by_prompt=mp4, prompt_count=len(mp4), chunk_count_per_prompt=60)


def execute(args, root, run):
    from .main_table_ours_campaign import run_campaign
    cells = {}; reference = None; comparisons = []
    for label, fused, profiled in CELL_ORDER:
        directory = root/'cells'/label
        directory.mkdir(parents=True, exist_ok=False)
        with environment({'CFEXP_RUN_DIR':str(directory), fusion.ENV:'1' if fused else '0',
                          'CFEXP_SDVG_BOUNDED_PROFILE':'1' if profiled else '0'}):
            result = run_campaign(draft_artifact_spec_path=args.draft_artifact_spec,
                draft_model_id='longlive_1.3b', target_artifact_spec_path=args.target_artifact_spec,
                boundary_config_path=args.boundary_config, vbench_checkout=args.vbench_checkout,
                vbench_python=args.vbench_python, vbench_asset_manifest=args.vbench_asset_manifest,
                vbench_asset_spec=args.vbench_asset_spec, duration_seconds=30,
                baseline_population_v2=True, selection_strategy='rank_burst',
                burst_calibration_revision=CALIBRATION_SOURCE, sealed_calibration_ranks=True,
                record_video_ready_timing=True, overlap_native_vae=True, runtime_launcher=launch_runtime)
        if profiled:
            require_profile_samples(directory, config=dict(cell=label), result=result)
            profile = json.loads((directory/'provenance/tracking/profile-summary.json').read_text())
            require(profile['coverage_complete_for_this_campaign'] is True, 'both owners need actual CUDA profiles')
            from .krea_long_session_gate import validate_operator_profile
            for sample in profile['samples']:
                validate_operator_profile(sample, role=sample['role'], action='generate', root=directory)
        current = snapshot(directory, result, fused=fused)
        if reference is None:
            reference = current
        else:
            comparisons.append(dict(cell=label, **compare(reference, current)))
        from types import SimpleNamespace
        tagged = SimpleNamespace(log=lambda metrics: run.log({**metrics,'cell':label,'profiled':profiled}))
        rows = log_result(tagged, directory, result)
        cells[label] = dict(profiled=profiled, fused=fused,
            video_ready_seconds_by_prompt={r['prompt_id']:r['video_ready_wall_seconds'] for r in rows},
            hardware=result['hardware'], raw_rows=result['raw_rows'])
        cells[label]['execution'] = current['execution']
    paired = []
    for plain, fused in (('plain-first','fused-second'),('plain-second','fused-first')):
        for pid, seconds in cells[plain]['video_ready_seconds_by_prompt'].items():
            other = cells[fused]['video_ready_seconds_by_prompt'][pid]
            require(all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in (seconds,other)),
                    'invalid paired video-ready latency')
            paired.append(dict(plain_cell=plain, fused_cell=fused, prompt_id=pid,
                plain_video_ready_seconds=seconds, fused_video_ready_seconds=other,
                saved_seconds=seconds-other, plain_over_fused_ratio=seconds/other))
    return dict(schema='longlive_complete_video_fusion_preflight_v1', passed=True,
        not_paper128_main=True, original_population_shard='first and sixty-fifth original cases',
        comparisons=comparisons, cells=cells, paired_unprofiled_observations=paired,
        cannot_substitute_for_matched_target_speedup=True,
        no_claim_about_quality_advantage_or_population_speedup=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','boundary-config','vbench-checkout',
                 'vbench-python','vbench-asset-manifest','vbench-asset-spec'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(argv); validate_options()
    from .metadata import find_git_root, collect_git_metadata
    from .stage_timing import _unsealed_directory
    from .stage_timing_probe import _hardware
    from .accepted_stream import stream_devices
    from .runtime_helpers import shutdown_owned_runtime_helpers
    root = _unsealed_directory(Path(os.environ['CFEXP_RUN_DIR'])); repo = find_git_root(Path.cwd())
    hardware = _hardware((stream_devices()[0],))
    require(len(hardware['gpus']) == 1 and hardware['gpus'][0]['name'] == 'NVIDIA H200', 'one actual H200 required')
    admission = dict(ancestor=fusion.admit_gate(repo,os.environ[fusion.GATE_ENV]),
                     ordinary_worker=admit_normal_gate(repo,os.environ[NORMAL_ENV]))
    config = dict(phase='complete_video_engineering_preflight_not_main',
        git_sha=collect_git_metadata(repo)['sha'], hardware=hardware, admission=admission,
        physical_device_count=1, rank=0, batch_size=1, resolution=[480,832], fps=24,
        precision='BF16', nominal_seconds=30, frame_count=717, prompt_count=2,
        generation_seed=20260911, target_root=True, target_budget_after_root=4,
        observation_horizon=8, burst_length=4, native_local12_sink3_cleanB3=True,
        cell_order=[x[0] for x in CELL_ORDER], actual_profile_before_timed_cells=True,
        telemetry_utility='shared SDVG-named observation only; no SDVG algorithm',
        training_loss='not applicable: inference', learning_rate='not applicable: inference',
        gradient_norm='not applicable: inference', backward='not applicable: inference',
        optimizer='not applicable: inference', distributed_communication='none: one GPU')
    for label,path in (('draft',args.draft_artifact_spec),('target',args.target_artifact_spec)):
        config[label+'_artifact_spec_sha256'] = fusion.sha(path)
    import wandb
    mode = tracking_mode(os.environ); directory=root/'tracking/wandb';directory.mkdir(parents=True)
    options = dict(project='collaborative-forcing-reproduction',name=root.name,config=config,
        dir=str(directory),save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,
        disable_code=True,symlink=False,console='off',_disable_stats=True))
    fallback=False
    try: run=wandb.init(mode=mode,**options)
    except Exception:
        if mode!='online': raise
        mode='offline';fallback=True;run=wandb.init(mode=mode,**options)
    write_new(root/'provenance/tracking/session.json',dict(mode=mode,run_id=run.id,
        online_fallback_to_offline=fallback,synced_evidence_verified=False,config=config))
    ok=False
    try:
        result=execute(args,root,run);write_new(root/'result.json',result);ok=True;return 0
    finally:
        try: shutdown_owned_runtime_helpers(label='longlive-fullvideo-preflight')
        finally:
            run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
            write_new(root/'provenance/tracking/finished.json',dict(mode=mode,
                execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__': raise SystemExit(main())
