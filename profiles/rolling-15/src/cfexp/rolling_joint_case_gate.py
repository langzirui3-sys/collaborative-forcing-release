"""Single-H200 trained Rolling joint case oracle with real profiling and W&B.

Read the original sealed two-owner case; do not regenerate its Target outputs.
Three identical oracle repetitions warm the physical calls before a bounded
profile. This is correctness/diagnostic evidence, never full-video speedup.
"""
import hashlib
import json
import os
from pathlib import Path

from .rolling_joint_case_oracle import run_case
from .rolling_joint_handoff import ENV, require
from .tracked_sdvg import write_new, tracking_mode, materialize_sdk_log_links

RUN_ENV = 'CFEXP_ROLLING_JOINT_ORIGINAL_CASE_RUN'
BUDGET_ENV = 'CFEXP_ROLLING_JOINT_CASE_TARGET_BLOCKS'
ORIGINAL_RUN = None


def admit(run, repo):
    from .rolling_transaction_proof import verify
    from .main_table_merge import _load_frozen_index
    from .correction_state import LatentHistory, InitialNoiseDescriptor
    from .rolling_forcing_noise import verify_schedule
    from .correction_runtime import NOISE_NAMESPACE
    require(str(run) == ORIGINAL_RUN, 'the original sealed mixed case is required')
    proof = verify(run, repository_root=repo)
    frozen, seal = _load_frozen_index(Path(run)); prefix = 'attempts/'+seal['attempt_id']+'/'
    attempt = Path(run)/prefix
    result = json.loads(frozen.file(prefix+'result.json', label='original mixed case').read_text())
    models = json.loads(frozen.file(prefix+'provenance/models.json', label='original model/noise').read_text())
    history = LatentHistory.from_dict(result['accepted_history']); history.verify_files(attempt)
    receipt = models['noise_schedule']
    binding = dict(run_dir=attempt, relative_path=receipt['path'], sha256=receipt['sha256'],
        prompt_id=history.prompt_id, prompt=result['prompt'], trajectory_seed=result['seed'],
        namespace=NOISE_NAMESPACE, chunk_count=16)
    payload, _ = verify_schedule(**binding)
    require(len(history.segments) == 16 and history.next_chunk_index == 16,
            'original accepted trajectory is incomplete')
    for segment, row in zip(history.segments, payload['chunks']):
        require(segment.initial_noise_descriptor_sha256 ==
                InitialNoiseDescriptor.from_dict(row['noise']).descriptor_sha256,
                'original accepted segment and full-noise schedule differ')
    return proof, history, binding


def load_inputs(history, binding, *, torch, load, device):
    from .rolling_forcing_noise import load_schedule
    tensors = []
    for segment in history.segments:
        segment.verify_file(binding['run_dir'])
        raw = (binding['run_dir']/segment.latent_path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == segment.latent_sha256,
                'original latent changed before decoding')
        values = load(raw); value = values.get('latent')
        require(set(values) == {'latent'} and isinstance(value, torch.Tensor)
                and tuple(value.shape) == (1,3,16,60,104) and value.dtype == torch.bfloat16
                and bool(torch.isfinite(value).all()), 'invalid original accepted latent tensor')
        tensors.append(value)
    noise, seed = load_schedule(torch=torch, load_file=load, device=device, **binding)
    require(seed == 20260911, 'original noise seed changed')
    return torch.cat(tensors, dim=1).to(device=device), noise


def main():
    from .main_table_baseline_campaign import _run_dir, _model_root, _gpu_compute_pids
    from .metadata import find_git_root, collect_git_metadata
    from .stage_timing_probe import _hardware
    from .rolling_forcing_backend import RollingNativeBackend
    from .rolling_joint_case_profile import observe, validate
    from .stage_timing import timing_session, stage
    require(os.environ.get(ENV) == '1' and os.environ.get('CFEXP_STAGE_TIMING') == '1'
            and os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS','0') == '0'
            and os.environ.get('CFEXP_ROLLING_HOST_WINDOW','0') == '0',
            'explicit original-metadata profiled diagnostic required')
    budget = os.environ.get(BUDGET_ENV, '4')
    require(budget in ('2', '4'), 'only frozen B2/B4 gate shapes are allowed')
    width = int(budget)
    resume = 8+width
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    require(len(visible) == 1 and visible[0].strip(), 'one actual visible H200 required')
    root = _run_dir(); repo = find_git_root(Path.cwd()); hardware = _hardware(visible)
    require(len(hardware['gpus']) == 1 and hardware['gpus'][0]['name'] == 'NVIDIA H200',
            'single physical H200 correctness gate required')
    uuid = hardware['gpus'][0]['uuid']
    require(not _gpu_compute_pids(uuid), 'diagnostic GPU already occupied')
    proof, original, binding = admit(os.environ[RUN_ENV], repo)
    config = dict(method=f'Rolling native joint B{width} case correctness gate',
        source=collect_git_metadata(repo)['sha'], original_case=proof, hardware=hardware,
        physical_device_count=1, rank=0, batch_size=1, resolution=[480,832], precision='BF16',
        prompt='a car stuck in traffic during rush hour', seed=20260911,
        repetitions=3, matching_forward_warmup=2, latent_blocks=16,
        retained_prefix_blocks=8, accepted_target_blocks=list(range(8,resume)),
        following_draft_blocks=list(range(resume,16)), target_suffix_blocks=width,
        original_resumed_latents_required=width == 4, original_native_tensor_metadata=True,
        no_quality_or_end_to_end_speed_claim=True, target_loaded_in_this_run=False,
        target_outputs='original sealed history, independently verified',
        backward='not applicable: inference', optimizer='not applicable: inference',
        loss='not applicable: inference', learning_rate='not applicable: inference',
        gradient_norm='not applicable: inference', decoding='not applicable: latent/cache gate',
        distributed_communication='none: one physical GPU')
    import wandb
    mode = tracking_mode(os.environ); directory = root/'tracking/wandb'; directory.mkdir(parents=True)
    options = dict(project='collaborative-forcing-reproduction', name=root.name, dir=str(directory),
        config=config, save_code=False, settings=wandb.Settings(init_timeout=25, disable_git=True,
            disable_code=True, symlink=False, console='off', _disable_stats=True))
    fallback = False
    try: run = wandb.init(mode=mode, **options)
    except Exception:
        if mode != 'online': raise
        mode='offline'; fallback=True; run=wandb.init(mode=mode, **options)
    write_new(root/'provenance/tracking/session.json', dict(mode=mode, run_id=run.id,
        online_fallback_to_offline=fallback, synced_evidence_verified=False, config=config))
    backend = None; ok = False; checks = 0
    def ownership():
        nonlocal checks
        actual = _gpu_compute_pids(uuid)
        write_new(root/('provenance/gpu-ownership-%04d.json'%checks), dict(
            gpu_uuid=uuid, allowed_pid=os.getpid(), actual_pids=sorted(actual),
            exclusive=actual <= {os.getpid()})); checks += 1
        require(actual <= {os.getpid()}, 'unrelated GPU process appeared during the gate')
    try:
        with timing_session(root, 'rolling-joint-case', dict(physical_device_count=1)) as timing:
            with stage('native_model_loading'):
                backend = RollingNativeBackend(run_dir=root, repository_root=repo, model_root=_model_root())
            t = backend.torch
            require(backend.audit['checkpoint_sha256'] ==
                '1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98'
                and backend.audit['strict_state_load'] is True, 'trained checkpoint load differs')
            timing.attach_cuda(t, backend.device)
            from safetensors.torch import load
            with stage('original_history_and_noise_loading', cuda=True):
                history, noise = load_inputs(original, binding, torch=t, load=load, device=backend.device)
            write_new(root/'provenance/models.json', dict(native_load=backend.audit,
                rolling=backend.records, original_case=proof, hardware=hardware))
            reports = []
            with t.no_grad(), observe(backend, root, target_suffix_blocks=width) as forwards:
                for repetition in range(3):
                    ownership()
                    with stage('complete_case_oracle', detail={'repetition':repetition}, cuda=True):
                        report = run_case(pipeline=backend.pipeline, torch=t, history=history,
                            noise=noise, prompt=config['prompt'], seed=config['seed'], ownership=ownership,
                            target_suffix_blocks=width)
                    require(report['passed'] is True and report['executed_device_type'] == 'cuda',
                            'actual trained CUDA oracle did not pass')
                    write_new(root/('parity/repetition-%s.json'%repetition), report); reports.append(report)
            samples = validate(forwards, target_suffix_blocks=width)
            for sample in samples:
                path = root/sample['trace_path']
                require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()
                        == sample['trace_sha256'], 'actual CUDA trace changed')
                run.log({'profile/'+sample['kind']+'/'+key:sample[key] for key in (
                    'wall_seconds','cuda_stream_seconds','cuda_kernel_us_sum',
                    'cuda_transfer_us_sum','peak_allocated_bytes','peak_reserved_bytes',
                    'noisy_latent_frames_per_cuda_stream_second')})
            write_new(root/'provenance/tracking/profile-summary.json', dict(
                executed_actual_cuda=True, samples=samples, physical_forward_count=len(forwards),
                full_cache_oracle_hooks_included=True, no_operation_ratio_speedup_claim=True))
            write_new(root/'result.json', dict(schema='rolling_joint_trained_case_gate_v1', passed=True,
                source=config['source'], original_case=proof, hardware=hardware,
                target_suffix_blocks=width, original_resumed_latents_required=width == 4,
                repetitions=reports, physical_forward_count=len(forwards),
                gpu_ownership_checks=checks, actual_post_warmup_cuda_profiles=True,
                original_target_outputs_not_regenerated=True,
                no_quality_or_end_to_end_speed_claim=True))
            run.log({'correctness/complete_cuda_oracle_passed':1,'correctness/repetitions':3})
        ok = True; return 0
    finally:
        try:
            if backend is not None: backend.close()
        finally:
            run.finish(exit_code=0 if ok else 1); materialize_sdk_log_links(directory)
            write_new(root/'provenance/tracking/finished.json', dict(mode=mode,
                execution_succeeded=ok, synced_evidence_verified=False))


if __name__ == '__main__': raise SystemExit(main())
