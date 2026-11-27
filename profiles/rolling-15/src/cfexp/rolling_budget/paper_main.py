"""Single-H200 Rolling Forcing main comparison on the original paper population."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path

from ..longlive_fullvideo_preflight import environment
from ..native_single_gpu import TIMING_SCOPE
from ..paper_population import generation_seed, load
from ..rolling_budget.admission import admit_original_pair, digest, require
from ..rolling_budget.trial import RUN_ROOT_ENV, process_factory as tracked_process, scoped_auditor
from ..rolling_budget.worker import build_joint_backend
from ..rolling_joint_handoff import ENV as JOINT_ENV
from ..tracked_sdvg import (log_result, materialize_sdk_log_links,
                          require_profile_samples, tracking_mode, write_new)

METHODS = ('ours', 'target', 'draft')
FACTORY = 'cfexp.rolling_budget.paper_main:build_backend'
PROFILE_ENV = 'CFEXP_ROLLING_PAPER_PROFILE_PLAN'
CALIBRATION_INDEX = 'a917d7e58c7180ce072ac8d0d31bbba167089bd80c2ab49f30d37baa7cbe17ae'
CALIBRATION_SOURCE = '3d27906622e66883a765e5672e186e984bb3968a'
CALIBRATION_RECEIPT = 'dba6c86745d7bcb72150c441b6b6ddfdbb9adfe71fc0a998bf6e4dfe46d82d26'
CALIBRATION_RANKS = '64de7e01ee6d32a0e271d3ce4ced54a3aa66519864181a313659d1599a7eaa84'


def validate_options(method, seconds, phase, env=None):
    env = os.environ if env is None else env
    require(method in METHODS and seconds in (15, 30)
            and phase in ('profile', 'main'), 'unknown Rolling comparison cell')
    require(all(env.get(key) == '1' for key in (
        'CFEXP_NATIVE_SINGLE_GPU', 'CFEXP_PAPER_REPRODUCTION',
        'CFEXP_ACCEPTED_VIDEO_STREAM', 'CFEXP_RENDER_TARGET_PRIORITY',
        'CFEXP_VIDEO_ONLY_DECODE', 'CFEXP_SDVG_TRACKING', 'CFEXP_STAGE_TIMING',
    )) and env.get('CFEXP_VIDEO_ENCODING') == 'serial_abr_v1'
       and env.get('CFEXP_NATIVE_VIDEO_STREAM_MODEL') == 'rolling'
       and env.get('CFEXP_KREA_TIMESTEP_POLICY') == 'main_comparison_v1',
       'Rolling comparison requires the native single-H200 renderer and Target schedule')
    require(all(env.get(key, '0') == '0' for key in (
        'CFEXP_DRAFT_KV_REUSE', 'CFEXP_ROLLING_HOST_KV_CURSORS', 'CFEXP_ROLLING_HOST_WINDOW',
        'CFEXP_ROLLING_CPU_HISTORY_IDENTITY', 'CFEXP_ROLLING_COMMITTED_PREFIX',
        'CFEXP_ROLLING_PREFIX_TRANSACTIONS', 'CFEXP_PAPER_C10_CALIBRATION',
        'CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SEAL',
    )) and 'CFEXP_TORCH_PROFILE_CHUNK' not in env,
       'unexpected cache mode or legacy profiler')
    require(env.get(JOINT_ENV) == ('1' if method == 'ours' else '0')
            and env.get('CFEXP_SDVG_BOUNDED_PROFILE') == str(int(phase == 'profile')),
            'only Ours uses joint history execution and profiling has its own phase')
    seed = generation_seed(env)
    shard = env.get('CFEXP_PAPER_PROMPT_SHARD_INDEX')
    count = env.get('CFEXP_PAPER_PROMPT_SHARD_COUNT')
    if phase == 'profile':
        require(method == 'ours' and shard == '0' and count == '128' and seed == 20260911,
                'profile uses the first predetermined original case and seed')
    else:
        require(shard in ('0', '1', '2', '3') and count == '4'
                and bool(env.get(PROFILE_ENV)) and bool(env.get(PROFILE_ENV+'_SHA256')), 'main requires one paper shard and its profile plan')
    if method == 'ours':
        require(env.get('CFEXP_INSIGHT2_REQUIRE_CALIBRATION_RECEIPT') == '1'
                and env.get('CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SHA256') == CALIBRATION_RECEIPT,
                'Ours must use the original independent Rolling C10 receipt')


def build_backend(raw_config):
    count = os.environ.get('CFEXP_ROLLING_PAPER_CHUNK_COUNT')
    require(count in ('30', '60'), 'Rolling worker requires the declared main horizon')
    return build_joint_backend(raw_config, blocks=4, chunk_count=int(count))


def launch_runtime(*, chunk_count, process_factory: object, **kwargs):
    from ..accepted_stream_worker import process_factory as ordinary
    from ..correction_runtime import launch_correction_runtime

    require(process_factory is ordinary and chunk_count in (30, 60)
            and kwargs.get('draft_model_id') == 'rolling_forcing_1.3b'
            and kwargs.get('allow_shared_device') is True
            and len(set(kwargs['devices'])) == 1, 'wrong Rolling horizon or device')
    return launch_correction_runtime(
        process_factory=partial(tracked_process,
            draft_factory=FACTORY if os.environ[JOINT_ENV] == '1' else None),
        chunk_count=chunk_count, **kwargs)


def profile_manifest(seconds):
    return f'experiments/paper128-rolling-profile-{seconds}s-ntu-v1.manifest.json'


def check_profile(root, result, hardware):
    from ..krea_long_session_gate import validate_operator_profile

    summary = json.loads((root/'provenance/tracking/profile-summary.json').read_text())
    samples = summary['samples']
    require(summary['coverage_complete_for_this_campaign'] is True
            and len(samples) == 2 and {x['role'] for x in samples} == {'draft', 'target'}
            and summary['hardware'] == result['hardware'], 'both actual worker profiles are required')
    expected = hardware['gpus'][0]
    observed = result['hardware']['gpus']
    require(len(observed) == 1 and all(observed[0][key] == expected[key]
            for key in ('name', 'memory_total', 'driver_version')),
            'profile hardware class differs')
    for sample in samples:
        validate_operator_profile(sample, role=sample['role'], action='generate', root=root)
    return summary


def admit_profile(path, plan_sha, *, seconds, source, hardware, first_prompt_id):
    from ..main_table_merge import _load_frozen_index

    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and digest(path) == plan_sha, 'Rolling profile plan changed')
    plan = json.loads(path.read_text())
    tasks = [task for task in plan['tasks'] if task['manifest'] == profile_manifest(seconds)]
    require(plan.get('schema') == 'cfexp_erank_queue_v1' and plan.get('ops') == str(path.parent)
            and len(tasks) == 1 and tasks[0]['commit'] == source,
            'profile plan does not bind one case at the current source')
    state_path = path.parent/'state.json'
    require(not state_path.is_symlink(), 'profile queue state cannot be a symlink')
    state = json.loads(state_path.read_text())
    row = state['tasks'].get(tasks[0]['id'], {})
    require(state['plan_sha256'] == plan_sha and row.get('status') == 'succeeded',
            'required profile has not completed')
    run_dir = Path(row['run_dir'])
    require(run_dir.is_absolute() and run_dir.resolve(strict=True) == run_dir,
            'profile queue must point to the original run')
    frozen, marker = _load_frozen_index(run_dir)
    require(marker['status'] == 'SUCCEEDED' and marker['exit_code'] == 0
            and marker['git_sha'] == source, 'profile seal differs')
    prefix = 'attempts/'+marker['attempt_id']+'/'
    result = json.loads(frozen.file(prefix+'result.json', label='profile result').read_text())
    complete = json.loads(frozen.file(prefix+'provenance/rolling-paper-completion.json',
                                      label='profile completion').read_text())
    frozen.file(prefix+'provenance/tracking/profile-summary.json', label='profile samples')
    finish = json.loads(frozen.file(prefix+'provenance/tracking/finished.json',
                                    label='profile tracking finish').read_text())
    require(complete['phase'] == 'profile' and complete['method'] == 'ours'
            and complete['seconds'] == seconds and complete['row_count'] == 1
            and result['population']['selected_prompt_ids'] == [first_prompt_id]
            and result['population']['seed'] == 20260911
            and finish['execution_succeeded'] is True, 'wrong profiled workload')
    verify_calibration(result)
    attempt = run_dir/prefix
    check_profile(attempt, result, hardware)
    return dict(run_id=marker['run_id'], index=marker['artifact_index_sha256'],
                task_id=tasks[0]['id'], source=source, seconds=seconds)


def verify_calibration(result):
    proof = result['calibration_receipt']
    require(proof['mode'] == 'sealed_rank_distribution'
            and proof['code_revision'] == CALIBRATION_SOURCE
            and proof['receipt_sha256'] == CALIBRATION_RECEIPT
            and proof['source_artifact_index_sha256'] == CALIBRATION_INDEX
            and proof['rank_count'] == 1190 and proof['rank_sha256'] == CALIBRATION_RANKS,
            'resolved calibration does not match the original Rolling C10')


def execute(args):
    from .. import main_table_ours_campaign as campaign, native_video_ready_target as reference

    common = dict(draft_artifact_spec=args.draft_artifact_spec,
                  target_artifact_spec=args.target_artifact_spec,
                  vbench_checkout=args.vbench_checkout, vbench_python=args.vbench_python,
                  vbench_asset_manifest=args.vbench_asset_manifest,
                  vbench_asset_spec=args.vbench_asset_spec,
                  duration_seconds=args.duration_seconds, overlap_native_vae=True)
    if args.method != 'ours':
        original = reference.launch_correction_runtime
        reference.launch_correction_runtime = launch_runtime
        try:
            return reference.run(**common, native_draft_model_id='rolling_forcing_1.3b',
                                 pure_draft=args.method == 'draft')
        finally:
            reference.launch_correction_runtime = original
    common['draft_artifact_spec_path'] = common.pop('draft_artifact_spec')
    common['target_artifact_spec_path'] = common.pop('target_artifact_spec')
    chunk_count = args.duration_seconds*2
    with environment({'CFEXP_ROLLING_PAPER_CHUNK_COUNT': str(chunk_count)}), \
            scoped_auditor(4, chunk_count=chunk_count):
        return campaign.run_campaign(**common, draft_model_id='rolling_forcing_1.3b',
            boundary_config_path=args.boundary_config, baseline_population_v2=True,
            selection_strategy='rank_burst', burst_calibration_revision=CALIBRATION_SOURCE,
            sealed_calibration_ranks=True, record_video_ready_timing=True,
            runtime_launcher=launch_runtime)


def main(argv=None):
    from ..accepted_stream import stream_devices
    from ..metadata import collect_git_metadata, find_git_root
    from ..runtime_helpers import shutdown_owned_runtime_helpers
    from ..stage_timing import _unsealed_directory
    from ..stage_timing_probe import _hardware

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec', 'target-artifact-spec', 'boundary-config',
                 'vbench-checkout', 'vbench-python', 'vbench-asset-manifest', 'vbench-asset-spec'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase', choices=('profile', 'main'), required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--duration-seconds', type=int, choices=(15, 30), required=True)
    args = parser.parse_args(argv)
    seconds = args.duration_seconds
    validate_options(args.method, seconds, args.phase)
    root = _unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']))
    repo = find_git_root(Path.cwd())
    hardware = _hardware(stream_devices()[:1])
    require(len(hardware['gpus']) == 1 and hardware['gpus'][0]['name'] == 'NVIDIA H200',
            'a physical H200 is required')
    source = collect_git_metadata(repo)['sha']
    prompts, population = load(repo, model_pair='krea_rolling_forcing')
    proof = admit_original_pair(repo, os.environ[RUN_ROOT_ENV]) if args.method == 'ours' else None
    profile = None
    if args.phase == 'main':
        with environment({'CFEXP_PAPER_PROMPT_SHARD_INDEX': '0'}):
            first_prompt = load(repo, model_pair='krea_rolling_forcing')[1]['selected_prompt_ids'][0]
        profile = admit_profile(os.environ[PROFILE_ENV], os.environ[PROFILE_ENV+'_SHA256'],
            seconds=seconds, source=source, hardware=hardware, first_prompt_id=first_prompt)

    config = dict(method=args.method, phase=args.phase, source=source, population=population,
        hardware=hardware, prompts=prompts, actual_gpu_count=1, rank=0, batch_size=1,
        precision='BF16', resolution=[832, 480], fps=24, nominal_seconds=seconds,
        frame_count=seconds*24-3, timing_scope=TIMING_SCOPE,
        backward='not applicable: inference', optimizer='not applicable: inference',
        loss='not applicable: inference', learning_rate='not applicable: inference',
        gradient_norm='not applicable: inference', distributed_communication='none',
        target_root=args.method != 'draft', observation_horizon=8 if proof else None,
        post_root_target_blocks=4 if proof else None, profile=profile,
        draft_artifact_spec_sha256=digest(args.draft_artifact_spec),
        target_artifact_spec_sha256=digest(args.target_artifact_spec))
    directory = root/'tracking/wandb'
    directory.mkdir(parents=True)
    import wandb
    options = dict(project='collaborative-forcing-reproduction', name=root.name, config=config,
        dir=str(directory), save_code=False, settings=wandb.Settings(init_timeout=25,
            disable_git=True, disable_code=True, symlink=False, console='off', _disable_stats=True))
    mode = tracking_mode(os.environ)
    fallback = False
    try:
        run = wandb.init(mode=mode, **options)
    except Exception:
        if mode != 'online':
            raise
        mode, fallback = 'offline', True
        run = wandb.init(mode=mode, **options)
    write_new(root/'provenance/tracking/session.json', dict(mode=mode, run_id=run.id, config=config,
        online_fallback_to_offline=fallback, synced_evidence_verified=False))
    ok = False
    try:
        result = execute(args)
        ours = args.method == 'ours'
        if ours:
            verify_calibration(result)
        actual = result['population'] if ours else result['population_binding']
        expected_rows = 1 if args.phase == 'profile' else 32
        require(actual == population and result['hardware'] == hardware
                and result['timing_scope'] == TIMING_SCOPE and result['row_count'] == expected_rows
                and result['accepted_video_schedule'] == 'decode_after_generation',
                'executed workload differs from the declared hardware, timing or population')
        if args.phase == 'profile':
            require_profile_samples(root, config=config, result=result)
            check_profile(root, result, hardware)
        rows = log_result(run, root, result) if ours else log_reference(run, root, result, config)
        require(len(rows) == expected_rows
                and {row['prompt_id'] for row in rows} == set(population['selected_prompt_ids'])
                and all(row['seed'] == population['seed'] for row in rows), 'prompt/seed coverage differs')
        write_new(root/'provenance/rolling-paper-completion.json',
            dict(phase=args.phase, method=args.method, seconds=seconds, row_count=len(rows),
                 hardware=hardware, population=population, profile=profile,
                 joint_kernel_gate_indexes={str(k): v['index'] for k, v in proof['gates'].items()}
                    if proof else None))
        ok = True
        return 0
    finally:
        try:
            shutdown_owned_runtime_helpers(label='rolling-paper-main')
        finally:
            run.finish(exit_code=0 if ok else 1)
            materialize_sdk_log_links(directory)
            write_new(root/'provenance/tracking/finished.json',
                dict(mode=mode, execution_succeeded=ok, synced_evidence_verified=False))


def log_reference(run, root, result, config):
    rows = result['rows']
    for row in rows:
        run.log({**{'quality/'+k: v for k, v in row['quality_scores'].items()},
            'generation_seed': row['seed'], 'prompt_id': row['prompt_id'],
            'timing/video_ready_seconds': row['video_ready_wall_seconds'],
            'timing/generation_seconds': row['generation_wall_seconds'],
            'throughput/video_ready_fps': config['frame_count']/row['video_ready_wall_seconds']})
    for path in (root/'provenance/tracking').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            require(record['bounded_profile_sample'] is False, 'main timing includes a profiler')
            prefix = 'worker/'+record['role']+'/'+record['action']
            run.log({prefix+'/'+k: record[k] for k in (
                'request_wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')})
    return rows


if __name__ == '__main__':
    raise SystemExit(main())
