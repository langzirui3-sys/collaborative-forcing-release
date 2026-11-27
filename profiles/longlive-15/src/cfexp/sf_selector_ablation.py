"""First-twelve routing quality ablation with common one-H200 execution.

The historical pilot does not admit fused fixed/random controls. This separate
entry point uses its original trajectory primitives, with the same SF joint
backend for every selector. It never changes that pilot's admission rules.
This is the named small routing ablation, not a paper128 main-table result.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from . import erank_acceptance_pilot as pilot
from .accepted_stream import stream_devices, SINGLE_GPU_TIMING_SCOPE
from .accepted_stream_client import StreamDecoderProxy
from .branch_gate_probe import calibration_control_binding, collect_trajectory
from .duration_sweep import duration_profile
from .insight2_campaign import Insight2Campaign, PromptCase, validate_prompt_partition
from .krea_timestep_adapter import (target_timestep_policy, target_four_step_schedule,
                                    COMPARISON_TIMESTEP_POLICY)
from .main_table_merge import _load_frozen_index
from .native_c10_development import verify_target_gate
from .rank_burst import materialize_rank_bursts, validate_burst_schedule
from .sf_joint_handoff import verify_rollout
from .tracked_native_c10 import launch_runtime
from .tracked_sdvg import (write_new, tracking_mode, materialize_sdk_log_links,
                           require_profile_samples)

SELECTORS = ('erank', 'random', 'fixed')
CALIBRATION_REVISION = '5307f1b6c4cb4f88fdc9d879adb98a8d0808ff1e'
PROFILE_ENV = 'CFEXP_SELECTOR_PROFILE_RUN'
SCHEMA = 'sf_first12_common_execution_selector_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_options(selector, phase, env=None):
    env = os.environ if env is None else env
    require(selector in SELECTORS and phase in ('profile', 'ablation'), 'unknown selector or phase')
    for name in ('CFEXP_SF_SINGLE_GPU', 'CFEXP_SF_JOINT_HANDOFF', 'CFEXP_DRAFT_KV_REUSE',
                 'CFEXP_ACCEPTED_VIDEO_STREAM', 'CFEXP_RENDER_TARGET_PRIORITY',
                 'CFEXP_SDVG_TRACKING', 'CFEXP_STAGE_TIMING'):
        require(env.get(name) == '1', 'selector ablation requires common single-GPU KV/fusion/tracking: '+name)
    require(env.get('CFEXP_SDVG_BOUNDED_PROFILE') == ('1' if phase == 'profile' else '0'),
            'bounded CUDA profiling must be confined to the preceding profile phase')
    for name in ('CFEXP_PAPER_REPRODUCTION', 'CFEXP_PAPER_C10_CALIBRATION',
                 'CFEXP_SF_TARGET_TWO_STEP', 'CFEXP_SF_TARGET_BURST_INTERIOR_ONE_STEP',
                 'CFEXP_SF_TARGET_BURST_EXIT_TWO_STEP', 'CFEXP_SF_TARGET_LOW_NOISE_REFINEMENT',
                 'CFEXP_SF_TARGET_EARLY_NATIVE_BURSTS', 'CFEXP_ACCEPTED_TARGET_PREFILL',
                 'CFEXP_BOUNDARY_EARLY_REJECT', 'CFEXP_NATIVE_SINGLE_GPU'):
        require(env.get(name, '0') == '0', 'selector ablation forbids a different execution recipe: '+name)
    for name in ('CFEXP_GENERATION_SEED', 'CFEXP_PAPER_PROMPT_SHARD_COUNT',
                 'CFEXP_PAPER_PROMPT_SHARD_INDEX', 'CFEXP_TORCH_PROFILE_CHUNK'):
        require(name not in env, 'selector ablation forbids a population/profiler override: '+name)
    require(target_timestep_policy(env) == COMPARISON_TIMESTEP_POLICY,
            'routing controls must use the same original comparison Target schedule')
    if phase == 'ablation':
        require(bool(env.get(PROFILE_ENV)), 'full selector ablation requires its own completed CUDA profile')


def select_prompts(repo, protocol, phase):
    selected, binding = pilot.load_baseline_population(repo)
    require(len(selected) == 24, 'selector ablation requires the frozen original twenty-four population')
    cases = tuple(PromptCase(p['id'], p['text'], 'evaluation') for p in selected[:12])
    calibration = tuple(PromptCase(p['prompt_id'], p['prompt'], 'calibration')
                        for p in protocol.calibration_prompts)
    validate_prompt_partition(calibration, cases)
    require(len(calibration) == 10 and len(cases) == 12, 'frozen first12/C10 population differs')
    chosen = cases[:2] if phase == 'profile' else cases
    return chosen, calibration, dict(original_population=binding, fixed_first12=[p.prompt_id for p in cases],
        selected_prompt_ids=[p.prompt_id for p in chosen], selection='first_two_bounded_profile' if phase == 'profile'
        else 'complete_fixed_first12', not_paper128_main=True)


def arms_for(selector):
    return (pilot.RANDOM_P8_B4_ARMS if selector == 'random' else
            (pilot.P8_TRANSFER_ARM,) if selector == 'erank' else ('fixed_deadline8_k4',))


def materialize(campaign, *, selector, arm, prompt, calibrator, on_commit):
    require(arm in arms_for(selector), 'selector arm identity differs')
    if selector == 'erank':
        return materialize_rank_bursts(campaign, prompt=prompt, seed=20260911, budget=4,
            calibrator=calibrator, burst_length=4, observation_horizon=8, on_commit=on_commit)
    positions = pilot.block_positions(arm, chunk_count=30, prompt_id=prompt.prompt_id)
    result = campaign._materialize_fixed_positions(prompt=prompt, seed=20260911,
        corrected_chunks=positions, bypass_draft_on_target_chunks=True, on_commit=on_commit)
    require(tuple(result[2]) == positions, 'frozen fixed/random positions changed')
    return result


def prepared_evidence(root, directory, prepared):
    """Retain the native first-chunk frame count, without padding or retiming."""
    video = pilot.bound_file(root, directory/prepared.video_path)
    decode = pilot.bound_file(root, directory/prepared.decode_evidence_path)
    profile = duration_profile(15)
    require(video['sha256'] == prepared.video_sha256 and video['nbytes'] == prepared.video_nbytes
        and decode['sha256'] == prepared.decode_evidence_sha256
        and prepared.frame_count == profile.frame_count and prepared.fps == 24,
        'video/evidence/native horizon differs')
    from .video_encoding_policy import verify_video
    return video, decode, verify_video(root/video['path'])


def admit_profile(repo, run_path, selector, revision, calibration, population, hardware):
    run = Path(run_path)
    queue = None
    if run.is_file():
        digest = os.environ.get('CFEXP_SELECTOR_PROFILE_PLAN_SHA256')
        require(run.is_absolute() and not run.is_symlink() and isinstance(digest,str)
                and len(digest)==64 and hashlib.sha256(run.read_bytes()).hexdigest()==digest,
                'selector profile queue plan hash differs')
        plan = json.loads(run.read_text())
        manifest = f'experiments/sf-selector-common-{selector}-profile-15s-ntu-v1.manifest.json'
        tasks = [t for t in plan.get('tasks',[]) if t.get('manifest')==manifest]
        require(plan.get('schema')=='cfexp_erank_queue_v1' and plan.get('ops')==str(run.parent)
                and len(tasks)==1 and tasks[0].get('commit')==revision,
                'selector profile queue source or unique task differs')
        state_path = run.parent/'state.json'
        require(not state_path.is_symlink(), 'unsafe selector profile state')
        state = json.loads(state_path.read_text()); row = state.get('tasks',{}).get(tasks[0]['id'],{})
        require(state.get('plan_sha256')==digest and row.get('status')=='succeeded'
                and isinstance(row.get('run_dir'),str) and Path(row['run_dir']).is_absolute(),
                'selector profile queue has no original successful run')
        queue = dict(plan=str(run),sha256=digest,task_id=tasks[0]['id'])
        run = Path(row['run_dir'])
    frozen, marker = _load_frozen_index(run)
    require(marker.get('git_sha') == revision and marker.get('status') == 'SUCCEEDED'
            and marker.get('exit_code') == 0, 'profile source or successful seal differs')
    frozen.verify_all()
    prefix = 'attempts/'+marker['attempt_id']+'/'
    result = json.loads(frozen.file(prefix+'result.json', label='selector profile result').read_text())
    require(result.get('schema') == SCHEMA and result.get('phase') == 'profile'
            and result.get('selector') == selector and result.get('arms') == list(arms_for(selector))
            and result.get('calibration_receipt') == calibration
            and result.get('population_binding', {}).get('fixed_first12') == population['fixed_first12']
            and result['population_binding']['selected_prompt_ids'] == population['fixed_first12'][:2]
            and result.get('row_count') == 2*len(arms_for(selector))
            and result.get('physical_device_count') == 1
            and result.get('hardware', {}).get('gpus') == hardware['gpus'],
            'profile does not cover this exact selector/population/calibration')
    finish = json.loads(frozen.file(prefix+'provenance/tracking/finished.json', label='tracking finish').read_text())
    require(finish.get('execution_succeeded') is True, 'profile tracking did not finish')
    for arm in arms_for(selector):
        path = frozen.file(prefix+'cells/'+arm+'/provenance/tracking/profile-summary.json', label='CUDA profile')
        evidence = json.loads(path.read_text())
        require(evidence.get('executed') is True and evidence.get('coverage_complete_for_this_campaign') is True
                and {r['role'] for r in evidence.get('samples', [])} == {'draft', 'target'},
                'both actual post-warmup model profiles are required')
        from .krea_long_session_gate import validate_operator_profile
        for row in evidence['samples']:
            validate_operator_profile(row, role=row['role'], action='generate',
                                      root=run/'attempts'/marker['attempt_id']/'cells'/arm)
    return dict(run_id=marker['run_id'], artifact_index_sha256=marker['artifact_index_sha256'],
                source=revision, selector=selector, actual_cuda_profile_verified=True, queue_binding=queue)


def execute(args, root, repo, run):
    """Load once per routing repeat; score only after all native workers close."""
    phase, selector = args.phase, args.selector
    config = pilot.load_run_config(args.boundary_config, repo_root=repo)
    protocol = pilot.load_protocol(config.protocol_path)
    prompts, calibration, population = select_prompts(repo, protocol, phase)
    devices = stream_devices()
    hardware = pilot._hardware((devices[0],))
    require(len(hardware['gpus']) == 1 and hardware['gpus'][0]['name'] == 'NVIDIA H200',
            'selector replication requires one actual H200')
    model_root = pilot._model_root()
    _, dv, dm = pilot.verify_draft_artifact_spec(args.draft_artifact_spec, model_root=model_root, repo_root=repo)
    _, tv, tm = pilot.verify_target_artifact_spec(args.target_artifact_spec, model_root=model_root, repo_root=repo)
    require((dm, tm) == ('cfexp.self_forcing_frame_adapter', 'cfexp.krea_chunk_backend'), 'model pair differs')
    ds = pilot._sha256_file(args.draft_artifact_spec)[0]
    ts = pilot.target_artifact_spec_sha256(args.target_artifact_spec)
    binding = pilot._calibration_receipt_binding(repo_root=repo, config=config, protocol=protocol,
        protocol_sha256=pilot._sha256_file(config.protocol_path)[0], draft_spec_sha256=ds,
        target_spec_sha256=ts, draft_verified_artifacts=dv, target_verified_artifacts=tv,
        phase_seeds=(20260911,))
    calibrator, cdf = pilot._read_sealed_calibration(calibration_control_binding(binding, CALIBRATION_REVISION))
    gate = verify_target_gate(os.environ.get('CFEXP_TARGET_NATIVE60_GATE_RUN'), repo,
        plan_sha256=os.environ.get('CFEXP_TARGET_NATIVE60_GATE_PLAN_SHA256'))
    revision = pilot.collect_git_metadata(repo)['sha']
    profile = (admit_profile(repo, os.environ[PROFILE_ENV], selector, revision, cdf, population, hardware)
               if phase == 'ablation' else None)
    write_new(root/'provenance/selector-admission.json', dict(target_gate=gate, profile=profile,
        calibration_receipt=cdf, population_binding=population, hardware=hardware))
    run.config.update(dict(population_binding=population, calibration_receipt=cdf, target_gate=gate,
                           admitted_profile=profile, hardware=hardware))
    cells, requests = [], []
    for arm in arms_for(selector):
        directory = root/'cells'/arm
        directory.mkdir(parents=True, exist_ok=False)
        pilot._assert_idle_pair(*devices)
        from .sf_joint_handoff_adapter import process_factory
        started = time.perf_counter()
        runtime = launch_runtime(run_dir=directory, repo_root=repo,
            draft_artifact_spec_path=args.draft_artifact_spec, draft_artifact_spec_sha256=ds,
            draft_verified_artifacts=dv, target_artifact_spec_path=args.target_artifact_spec,
            target_artifact_spec_sha256=ts, target_verified_artifacts=tv, devices=devices,
            write_full_videos=True, chunk_count=30, allow_shared_device=True,
            conditioning_prompts=[dict(prompt_id=p.prompt_id, prompt=p.prompt) for p in prompts],
            shared_precomputed_conditioning=True, process_factory=process_factory)
        try:
            renderer = StreamDecoderProxy(runtime.draft, runtime.target, directory, chunk_count=30)
            load_seconds = time.perf_counter()-started
            runtime.evaluator.client = renderer
            runtime.evaluator.full_quality_metric_id = 'vbench_custom_six_dimension_macro_mean'
            campaign = Insight2Campaign(run_dir=directory, protocol_id='sf-first12-selector-common-execution-v1',
                seeds=(20260911,), chunk_count=30, calibration_prompts=calibration, evaluation_prompts=prompts,
                draft=runtime.draft, target=runtime.target, noise_provider=runtime.noise_provider,
                evaluator=runtime.evaluator)
            for prompt in prompts:
                pilot._assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                    target_device=devices[1], phase='selector:'+arm+':'+prompt.prompt_id)
                event = hashlib.sha256((arm+':'+prompt.prompt_id).encode()).hexdigest()
                before = time.perf_counter()
                with campaign._target_rpc_scope(event):
                    key, route, selected, history = materialize(campaign, selector=selector, arm=arm,
                        prompt=prompt, calibrator=calibrator, on_commit=renderer.append)
                generation = time.perf_counter()-before
                prepared = campaign._prepare_rollout(key=key, route_id=route, corrected_chunks=selected,
                    proposed_target_chunks=selected, history=history).prepared_quality
                ready = time.perf_counter()-before
                trajectory = collect_trajectory(campaign, key=key, history=history, accepted=selected, proposed=selected)
                require(len(selected) == 4 and trajectory['counters']['physical_calls'] == dict(draft=25,target=5),
                        'selector changed the shared root-plus-four Target/Draft budget')
                cache = pilot.verify_native_cache(campaign, key, trajectory)
                cache['joint_handoff'] = verify_rollout(campaign, key, selected=selected)
                if selector == 'erank':
                    validate_burst_schedule(trajectory, budget=4, burst_length=4, chunk_count=30, observation_horizon=8)
                for index in (0, *selected):
                    checkpoint = json.loads(campaign._checkpoint_path('rollouts',key,index).read_text())
                    require(checkpoint['generated']['diagnostics']['denoising_schedule'] == list(target_four_step_schedule()),
                            'one selected Target block did not retain all four native updates')
                video, decode, encoding = prepared_evidence(root, directory, prepared)
                mid = ('selector-'+arm+'-'+prompt.prompt_id).replace('_','-')
                cell = dict(arm=arm, selector=selector, prompt_id=prompt.prompt_id, prompt=prompt.prompt,
                    seed=20260911, measurement_id=mid, proposal_budget=4, trajectory=trajectory,
                    native_cache_audit=cache, generation_wall_seconds=generation, video_ready_wall_seconds=ready,
                    decode_encode_and_prepare_wall_seconds=ready-generation, model_load_wall_seconds=load_seconds,
                    timing_scope=SINGLE_GPU_TIMING_SCOPE, video=video, decode_evidence=decode,
                    encoding_policy=encoding, accepted_stream=renderer.last_evidence,
                    latency_is_diagnostic_not_a_main_speedup=True)
                write_new(directory/'provenance'/('selector-'+prompt.prompt_id+'.json'), cell)
                cells.append(cell)
                requests.append(pilot.VBenchLongVideoRequest(video_path=root/video['path'],
                    prompt=prompt.prompt, measurement_id=mid))
        finally:
            runtime.close()
        from .stage_timing_analysis import analyze_records
        stages = [json.loads(line) for p in (directory/'provenance/stage-timing').glob('*.jsonl')
                  for line in p.read_text().splitlines()]
        require(bool(stages), 'actual worker stage timing is missing')
        write_new(directory/'provenance/tracking/stage-analysis.json', analyze_records(stages))
        if phase == 'profile':
            require_profile_samples(directory, config=dict(selector=selector,arm=arm), result=dict(hardware=hardware))
            evidence = json.loads((directory/'provenance/tracking/profile-summary.json').read_text())
            require(evidence['coverage_complete_for_this_campaign'] is True, 'both models must be profiled after warm-up')
            from .krea_long_session_gate import validate_operator_profile
            for sample in evidence['samples']:
                validate_operator_profile(sample, role=sample['role'], action='generate', root=directory)
                run.log({'profile/'+arm+'/'+sample['role']+'/'+k:sample[k] for k in
                    ('request_wall_seconds','cuda_stream_elapsed_seconds','cuda_kernel_us_sum',
                     'cuda_transfer_us_sum','peak_allocated_bytes','peak_reserved_bytes',
                     'stream_window_frame_equivalents_per_second')})
    pilot._assert_idle_pair(*devices)
    scorer = pilot.OfficialVBenchScorer(run_dir=root, checkout=args.vbench_checkout,
        python_executable=args.vbench_python, asset_manifest=args.vbench_asset_manifest,
        asset_spec=args.vbench_asset_spec, visible_device=devices[0])
    before = time.perf_counter()
    measurements = scorer.score_videos(requests)
    scoring = time.perf_counter()-before
    require(set(measurements) == {x.measurement_id for x in requests}, 'incomplete official scoring coverage')
    rows = []
    for cell in cells:
        measurement = measurements[cell['measurement_id']]
        require(pilot.bound_file(root,root/cell['video']['path']) == cell['video'], 'scored video changed')
        row = dict(cell, quality_scores=dict(measurement.dimension_scores),
                   quality_evidence=pilot.bound_file(root,root/measurement.evidence_path))
        require(set(row['quality_scores']) == set(pilot.QUALITY_DIMENSIONS) and all(
            type(x) in (float,int) and math.isfinite(x) and 0 <= x <= 1 for x in row['quality_scores'].values()),
            'invalid six-dimension quality')
        rows.append(row)
        run.log({'quality/six_dimension_mean_100':100*sum(row['quality_scores'].values())/6,
                 'timing/diagnostic_video_ready_seconds':row['video_ready_wall_seconds'],
                 **{'quality/'+k:v for k,v in row['quality_scores'].items()}})
    values = {arm:{p.prompt_id:next(r for r in rows if r['arm']==arm and r['prompt_id']==p.prompt_id)
                              for p in prompts} for arm in arms_for(selector)}
    # Random's three routing draws are averaged WITHIN prompt first. They are
    # never treated as extra independent prompts or three generation seeds.
    per_prompt = {p.prompt_id:{d:sum(values[a][p.prompt_id]['quality_scores'][d]
                                   for a in arms_for(selector))/len(arms_for(selector))
                              for d in pilot.QUALITY_DIMENSIONS} for p in prompts}
    result = dict(schema=SCHEMA, phase=phase, selector=selector, status='succeeded',
        formal_ours_result=False, not_paper128_main=True, physical_device_count=1, hardware=hardware,
        arms=list(arms_for(selector)), row_count=len(rows), rows=rows, generation_seed=20260911,
        routing_repeat_count=len(arms_for(selector)), population_binding=population, calibration_receipt=cdf,
        calibration_code_revision=CALIBRATION_REVISION, source=revision, target_gate=gate, admitted_profile=profile,
        budget_after_root=4, target_burst_length=4, observation_horizon=8,
        duration=dict(nominal_seconds=15,chunk_count=30,frame_count=duration_profile(15).frame_count,fps=24),
        execution='same SF StateBridge/native KV/FuseStep factory for every selector',
        target_schedule=list(target_four_step_schedule()), per_prompt_quality=per_prompt,
        quality_mean_100=100*sum(sum(v.values())/6 for v in per_prompt.values())/len(per_prompt),
        vbench_wall_seconds=scoring, no_speedup_claim=True, automatic_winner_selection=False)
    write_new(root/'result.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','boundary-config',
                 'vbench-checkout','vbench-python','vbench-asset-manifest','vbench-asset-spec'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--selector', choices=SELECTORS, required=True)
    p.add_argument('--phase', choices=('profile','ablation'), required=True)
    args = p.parse_args(argv)
    validate_options(args.selector,args.phase)
    from .metadata import find_git_root
    from .stage_timing import _unsealed_directory
    from .runtime_helpers import shutdown_owned_runtime_helpers
    root = _unsealed_directory(pilot._run_dir()); repo = find_git_root(Path.cwd())
    config = dict(method='SF first12 routing ablation',phase=args.phase,selector=args.selector,
        resolution=[480,832],fps=24,precision='BF16',batch_size=1,physical_device_count=1,rank=0,
        nominal_seconds=15,frame_count=duration_profile(15).frame_count,chunk_count=30,
        generation_seed=20260911,target_root=True,target_budget_after_root=4,target_burst_length=4,
        observation_horizon=8,erank_layer=15,rank_visual_tokens=256,rank_denoise_step=0,
        actual_cuda_profile_phase=args.phase=='profile',backward='not applicable: inference',
        optimizer='not applicable: inference',loss='not applicable: inference',
        learning_rate='not applicable: inference',gradient_norm='not applicable: inference',
        distributed_communication='not applicable: single physical GPU',
        qualitative_comparison='original prompt/seed/video bindings retained on server; no automatic media upload',
        checkpoint_definitions={str(x):json.loads(x.read_text()) for x in
                                (args.draft_artifact_spec,args.target_artifact_spec)})
    import wandb
    mode = tracking_mode(os.environ); directory=root/'tracking/wandb'; directory.mkdir(parents=True)
    kwargs=dict(project='collaborative-forcing-reproduction',name=root.name,config=config,dir=str(directory),
        save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,disable_code=True,
                                             symlink=False,console='off',_disable_stats=True))
    fallback=False
    try:run=wandb.init(mode=mode,**kwargs)
    except Exception:
        if mode!='online':raise
        fallback=True;mode='offline';run=wandb.init(mode=mode,**kwargs)
    write_new(root/'provenance/tracking/session.json',dict(schema='wandb_session_v1',mode=mode,
        run_id=run.id,online_fallback_to_offline=fallback,synced_evidence_verified=False,config=config))
    ok=False
    try:
        execute(args,root,repo,run);ok=True;return 0
    finally:
        try:shutdown_owned_runtime_helpers(label='sf-selector-common-execution')
        finally:
            run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
            write_new(root/'provenance/tracking/finished.json',dict(schema='wandb_finish_v1',mode=mode,
                execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':raise SystemExit(main())
