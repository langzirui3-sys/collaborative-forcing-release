"""Instrumented independent-C10 native-four-step SF development, not a main result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .native_c10_development import ARMS, validate_options, verify_target_gate
from .krea_timestep_adapter import target_four_step_schedule, target_timestep_policy
from .tracked_sdvg import (write_new, tracking_mode, materialize_sdk_log_links,
                           require_profile_samples)


def process_factory(argv, **kwargs):
    """Only telemetry is shared with SDVG; no SDVG scoring or routing is used."""
    command = list(argv)
    if (command[1:3] != ['-m', 'cfexp.correction_worker']
            or os.environ.get('CFEXP_SDVG_TRACKING') != '1'):
        raise ValueError('native C10 requires its explicit standard-worker telemetry entry')
    role = command[command.index('--role')+1]
    if role not in ('draft', 'target'):
        raise ValueError('unexpected native C10 worker role')
    index = command.index('--backend-factory')+1
    expected = 'cfexp.correction_runtime:build_'+role+'_backend'
    if command[index] != expected:
        raise ValueError('native C10 backend identity differs')
    if role == 'draft':
        command[index] = 'cfexp.sf_joint_handoff_adapter:build_backend'
    elif target_timestep_policy()=='upstream_release_four_v1':
        command[index]='cfexp.krea_timestep_adapter:build_backend'
    command[2] = 'cfexp.tracked_sdvg_worker'
    return subprocess.Popen(command, **kwargs)


def launch_runtime(**kwargs):
    from .sf_joint_handoff_adapter import process_factory as ordinary
    from .correction_runtime import launch_correction_runtime
    if kwargs.pop('process_factory') is not ordinary:
        raise ValueError('native C10 must retain the original joint SF handoff')
    return launch_correction_runtime(process_factory=process_factory, **kwargs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec', 'target-artifact-spec', 'boundary-config',
                 'vbench-checkout', 'vbench-python', 'vbench-asset-manifest', 'vbench-asset-spec'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--calibration-code-revision', required=True)
    return p


def verify_native_updates(worker_root, result, budget):
    rows=result['rows']
    indexed={row['prompt_id']:row for row in rows}
    if (len(rows)!=10 or len(indexed)!=10
            or set(indexed)!=set(result['population_binding']['selected_prompt_ids'])
            or result.get('population')!='independent_C10_development_not_held_out'):
        raise ValueError('incomplete independent C10 output population')
    seen={prompt:{} for prompt in indexed}
    paths=list((worker_root/'state/rollouts').glob('*/chunk-*.json'))
    if len(paths)!=600:raise ValueError('C10 must retain all sixty accepted checkpoints per prompt')
    for path in paths:
        data=json.loads(path.read_text());prompt=data['history']['prompt_id']
        generated=data['generated'];chunk=generated['segment']['chunk_index']
        if prompt not in seen or chunk in seen[prompt] or not 0<=chunk<60:
            raise ValueError('duplicate or foreign C10 accepted checkpoint')
        seen[prompt][chunk]=generated
    for prompt,row in indexed.items():
        accepted=set(row['trajectory']['accepted_chunks'])|{0}
        if (len(accepted)!=budget+1 or row['trajectory']['counters']['physical_calls']
                !=dict(draft=59-budget,target=budget+1)):
            raise ValueError('actual C10 Draft/Target calls differ from budget')
        for chunk,generated in seen[prompt].items():
            expected='target' if chunk in accepted else 'draft'
            if generated['context']['role']!=expected:
                raise ValueError('accepted C10 output producer differs from selected route')
            if generated['segment']['latent_sha256']!=row['trajectory']['committed'][chunk]['latent_sha256']:
                raise ValueError('C10 output differs from committed trajectory')
            if expected=='target' and generated['diagnostics'].get('denoising_schedule')!=list(target_four_step_schedule()):
                raise ValueError('actual C10 Target block did not use four native updates')
    return dict(prompt_count=10,accepted_checkpoints=600,target_blocks=10*(budget+1),
        target_denoising_updates=40*(budget+1),all_target_schedules_verified=True,
        all_committed_output_hashes_verified=True)


def main(argv=None):
    args = parser().parse_args(argv)
    validate_options(cohort='main24_burst', arm=args.arm, seconds=30,
                     sealed_ranks=True, smoke=None)
    from .metadata import find_git_root, collect_git_metadata
    from .stage_timing import _unsealed_directory
    from .runtime_helpers import shutdown_owned_runtime_helpers
    from . import erank_acceptance_pilot as campaign
    root = _unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']))
    repo = find_git_root(Path.cwd())
    gate = verify_target_gate(os.environ.get('CFEXP_TARGET_NATIVE60_GATE_RUN'), repo,
        plan_sha256=os.environ.get('CFEXP_TARGET_NATIVE60_GATE_PLAN_SHA256'))
    write_new(root/'provenance/native60-target-admission.json', gate)
    budget = int(args.arm.rsplit('_k', 1)[1])
    config = dict(method='collaborative_forcing', phase='independent_C10_native4_development',
        held_out=False, development_only=True, may_replace_paper128_results=False,
        git_sha=collect_git_metadata(repo)['sha'], nominal_seconds=30, chunk_count=60,
        physical_device_count=1, batch_size=1, resolution=[480, 832], precision='BF16',
        seed=20260911, prompt_count=10, arm=args.arm, target_budget_after_root=budget,
        target_denoising_schedule=list(target_four_step_schedule()),
        timestep_policy=target_timestep_policy(),
        expected_target_forward_updates=4*(budget+1), target_burst_length=4,
        observation_horizon=8, split_target_bursts=True, draft_native_kv=True,
        joint_draft_handoff=True, selector='unchanged frozen C10 eRank; no predictor',
        target_gate=gate, warmup_forward_calls_per_worker=2,
        profiler_scope='one third-or-later request per worker; diagnostic timings, not main speedup',
        training='not applicable: inference', optimizer='not applicable: inference',
        loss='not applicable: inference', learning_rate='not applicable: inference',
        gradient_norm='not applicable: inference', backward='not applicable: inference',
        distributed_communication='none: one physical H200',
        qualitative_comparison='all ten original C10 prompt texts, fixed seed and local MP4 hashes; no auto-upload',
        telemetry_schema_note='shared SDVG-named observation utility only; no SDVG algorithm is executed')
    for label, path in (('draft',args.draft_artifact_spec),('target',args.target_artifact_spec)):
        config[label+'_artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        config[label+'_artifact_definition'] = json.loads(path.read_text())
    import wandb
    mode = tracking_mode(os.environ)
    directory = root/'tracking/wandb'; directory.mkdir(parents=True)
    kwargs = dict(project='collaborative-forcing-reproduction', name=root.name,
        config=config, dir=str(directory), save_code=False,
        settings=wandb.Settings(init_timeout=25, disable_git=True, disable_code=True,
                                symlink=False, console='off', _disable_stats=True))
    fallback = False
    try:
        run = wandb.init(mode=mode, **kwargs)
    except Exception:
        if mode != 'online': raise
        fallback=True; mode='offline'; run=wandb.init(mode=mode, **kwargs)
    write_new(root/'provenance/tracking/session.json', dict(schema='wandb_session_v1',
        mode=mode, run_id=run.id, online_fallback_to_offline=fallback,
        online_session_url=run.url if mode=='online' else None,
        synced_evidence_verified=False, config=config))
    ok = False
    try:
        before=time.perf_counter()
        result = campaign.run_probe(draft_artifact_spec_path=args.draft_artifact_spec,
            target_artifact_spec_path=args.target_artifact_spec,
            boundary_config_path=args.boundary_config, vbench_checkout=args.vbench_checkout,
            vbench_python=args.vbench_python, vbench_asset_manifest=args.vbench_asset_manifest,
            vbench_asset_spec=args.vbench_asset_spec, arm=args.arm, cohort='main24_burst',
            calibration_code_revision=args.calibration_code_revision, duration_seconds=30,
            sealed_calibration_ranks=True, development_c10=True, runtime_launcher=launch_runtime)
        worker_root = root/'cells'/args.arm
        execution=verify_native_updates(worker_root,result,budget)
        write_new(root/'provenance/native-c10-update-audit.json',execution)
        for index, row in enumerate(result['rows']):
            scores=row['quality_scores']
            run.log(dict(prompt_index=index, **{'quality/'+k:v for k,v in scores.items()},
                         **{'quality/six_dimension_mean_100':sum(scores.values())*100/6,
                            'timing/diagnostic_video_ready_seconds':row['video_ready_wall_seconds'],
                            'timing/generation_seconds':row['generation_wall_seconds'],
                            'timing/post_generation_decode_encode_seconds':row['decode_encode_and_prepare_wall_seconds'],
                            'timing/model_load_seconds':row['model_load_wall_seconds_per_arm']}))
        run.log({'timing/vbench_seconds':result['vbench_wall_seconds']})
        require_profile_samples(worker_root, config=config, result=result)
        # A separate verified Target profile cannot stand in for these actual workers.
        samples=json.loads((worker_root/'provenance/tracking/profile-summary.json').read_text())['samples']
        if {x['role'] for x in samples} != {'draft','target'}:
            raise ValueError('native C10 requires actual post-warmup profiles for BOTH model owners')
        from .krea_long_session_gate import validate_operator_profile
        for sample in samples:
            validate_operator_profile(sample,role=sample['role'],action='generate',root=worker_root)
            run.log({'profile/'+sample['role']+'/'+k:sample[k] for k in
                     ('request_wall_seconds','cuda_stream_elapsed_seconds','cuda_kernel_us_sum',
                      'cuda_transfer_us_sum','peak_allocated_bytes','peak_reserved_bytes',
                      'stream_window_frame_equivalents_per_second')})
        write_new(root/'provenance/native-c10-completion.json',dict(development_only=True,
            row_count=len(result['rows']), target_gate=gate, wall_seconds=time.perf_counter()-before,
            no_paper_quality_or_speed_claim=True,native_target_update_audit=execution))
        ok=True; return 0
    finally:
        try:
            shutdown_owned_runtime_helpers(label='native-c10-controller')
        finally:
            run.finish(exit_code=0 if ok else 1)
            materialize_sdk_log_links(directory)
            write_new(root/'provenance/tracking/finished.json',dict(schema='wandb_finish_v1',
                mode=mode,execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':raise SystemExit(main())
