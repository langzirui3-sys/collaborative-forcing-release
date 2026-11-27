"""Fixed-position LongLive development screens and paired cohort controls.

Controls and late-handoff policies use separate worker lifetimes to prevent
mixing ordinary and experimental Draft cache owners.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

from .correction_state import LatentHistory
from .duration_sweep import duration_profile
from .insight2_campaign import ChunkGenerationRequest, PromptCase
from .longlive_late_handoff import POLICIES, require
from .longlive_motion_continuation import (
    CONTROL_ARM, GUIDANCE_ARMS, TARGET_POSITIONS, receipt, reference_chunk,
)

PROMPT_IDS = ('018_temporal_flickering_vbl0004', '042_aesthetic_quality_vbl0719')
SEED = 20260911
CHUNKS = 30
CONTROLS = ('target', 'draft_root')
CANDIDATES = tuple(p + '_' + k for p in POLICIES for k in ('k4', 'all'))
ARMS = CONTROLS + CANDIDATES
FULL_TARGET_POSITIONS = {
    'opening_k4': (1, 2, 3, 4),
    'paired_k4': (1, 2, 15, 16),
    'spaced_k4': (1, 8, 15, 22),
}
SPACING_ARMS = CONTROLS + tuple(FULL_TARGET_POSITIONS)
COHORT_PROMPT_COUNT = 24
LATE_SPREAD_ARM = 'd3_t625_spread_k8'
LATE_SPREAD_POSITIONS = (1, 8, 15, 22, 29, 36, 43, 50)
LATE_SPREAD_SECONDS = 30
FULL_SPREAD_ARM = 'full_target_spread_k8'
MOTION_ARMS = (CONTROL_ARM, *GUIDANCE_ARMS)
THIRTY_SECOND_ARMS = (FULL_SPREAD_ARM, *MOTION_ARMS)
SPACING_COHORT_ARMS = (*FULL_TARGET_POSITIONS, FULL_SPREAD_ARM, *GUIDANCE_ARMS)


def full_target_positions(arm):
    if arm in MOTION_ARMS:
        return TARGET_POSITIONS
    return LATE_SPREAD_POSITIONS if arm == FULL_SPREAD_ARM else FULL_TARGET_POSITIONS.get(arm, ())


def policy(arm):
    require(arm in CANDIDATES, 'unknown late screening candidate')
    name, budget = arm.rsplit('_', 1)
    return name, 4 if budget == 'k4' else CHUNKS - 1


def materialize(runtime, prompt, arm, renderer, *, check=lambda: None, seconds=15):
    from .longlive_late_client import propose, target, commit, generated, continue_draft
    from .krea_chunk_backend import MAIN_TABLE_FOUR_STEP_SCHEDULE, validate_native_correction_state
    require(arm in ARMS or arm in SPACING_COHORT_ARMS or arm in MOTION_ARMS or arm == LATE_SPREAD_ARM,
            'unknown screening arm')
    if arm in CONTROLS:
        allowed_seconds = (15, LATE_SPREAD_SECONDS)
    else:
        allowed_seconds = (LATE_SPREAD_SECONDS,) if arm in (LATE_SPREAD_ARM, *THIRTY_SECOND_ARMS) else (15,)
    require(seconds in allowed_seconds, 'screening arm and horizon differ')
    chunk_count = duration_profile(seconds).chunk_count
    if arm == 'target':
        from .main_table_full_target import generate_full_target_trajectory
        generation = generate_full_target_trajectory(prompt_id=prompt.prompt_id,
            prompt=prompt.prompt, target=runtime.target, noise_provider=runtime.noise_provider,
            chunk_count=chunk_count, duration_seconds=seconds, on_commit=renderer.append,
            safety_check=lambda phase, n: check())
        return generation.history, dict(arm=arm, selected_chunks=[],
            full_target_native_evidence=dict(generation.result),
            chunks=[dict(chunk_index=n, action='full_target',
                initial_noise_sha256=x.context.noise.tensor_sha256)
                for n, x in enumerate(generation.generated)])
    identity = hashlib.sha256(f'll-late-screen:{arm}:{prompt.prompt_id}'.encode()).hexdigest()
    history = LatentHistory(identity, prompt.prompt_id, SEED)
    name, budget = policy(arm) if arm in CANDIDATES else (None, 0)
    late_positions = tuple(range(1, budget + 1))
    if arm == LATE_SPREAD_ARM:
        name, late_positions = 'd3_t625', LATE_SPREAD_POSITIONS
    full_positions = full_target_positions(arm)
    target_session = None
    last_draft = None
    rows, selected = [], []
    for n in range(chunk_count):
        check()
        noise = runtime.noise_provider.materialize(prompt_id=prompt.prompt_id,
            trajectory_seed=SEED, chunk_index=n)
        request = ChunkGenerationRequest(prompt=prompt, history=history, noise=noise,
            output_latent_path=f'latents/{arm}/{prompt.prompt_id}/{n:03d}.safetensors',
            capture_hidden=False)
        row = dict(chunk_index=n, initial_noise_sha256=noise.descriptor.tensor_sha256)
        if n == 0 or n in full_positions:
            if n:
                renderer.before_generate('target')
            item = runtime.target.generate(replace(request,
                target_logical_call_id=hashlib.sha256(
                    (identity + (f':{n}' if n else ':root')).encode()).hexdigest(),
                target_call_phase='unscoped'))
            native = item.diagnostics['native_correction_state']
            validate_native_correction_state(native, chunk_index=n)
            if n:
                require(native['native_session_id'] == target_session and not native['session_recreated'],
                    'full Target spacing recreated its native correction session')
                selected.append(n)
            target_session = native['native_session_id']
            row.update(action='full_target', native_state=native)
            if arm in THIRTY_SECOND_ARMS:
                schedule = item.diagnostics['denoising_schedule']
                require(schedule == list(MAIN_TABLE_FOUR_STEP_SCHEDULE),
                        'fixed placement executed a different Target schedule')
                row['target_denoising_schedule'] = schedule
        elif n in late_positions:
            renderer.before_generate('draft')
            noises = [runtime.noise_provider.materialize(
                prompt_id=prompt.prompt_id + f'-late-renoise-{j}',
                trajectory_seed=SEED, chunk_index=n).tensor_path
                for j in range(POLICIES[name][0])]
            proposal = propose(runtime.draft, request, session_id=identity, policy=name,
                renoise_paths=noises, output_path=f'handoff/{arm}/{prompt.prompt_id}/{n:03d}.safetensors',
                verify_parity=False)
            renderer.before_generate('target')
            prediction = target(runtime.target, request, proposal, verify_parity=False)
            renderer.before_generate('draft')
            committed = commit(runtime.draft, session_id=identity, proposal=proposal,
                target_receipt=prediction, verify_parity=False)
            require('proposal_oracle' not in proposal['result']
                and 'target_oracle' not in prediction['result']
                and 'commit_oracle' not in committed['result'],
                'timed screening executed diagnostic oracle work')
            item = generated(runtime.target, request, prediction['result'])
            row.update(action='late_target', proposal=proposal, target=prediction, commit=committed)
            selected.append(n)
        else:
            renderer.before_generate('draft')
            item = (runtime.draft.generate(request) if name is None
                else continue_draft(runtime.draft, request, session_id=identity))
            reuse = item.diagnostics['kv_reuse']
            missing = (n if last_draft is None else n - last_draft - 1) if name is None else 0
            require(reuse['replayed_chunks'] == missing and reuse['reused_chunks'] == n - missing,
                'screening did not reuse Draft history outside the missing accepted Target suffix')
            last_draft = n
            row.update(action='draft', reuse=reuse)
            ref = reference_chunk(n) if arm in GUIDANCE_ARMS else None
            guidance = item.diagnostics.get('motion_continuation')
            if ref is None:
                require(guidance is None, 'guidance ran outside its declared continuation window')
            else:
                require(history.segments[ref].producer_role == 'target'
                    and guidance == dict(receipt(n, ref, arm=arm),
                        reference_latent_sha256=history.segments[ref].latent_sha256),
                    'guided Draft did not bind the accepted Target reference')
                row['motion_continuation'] = guidance
        item.segment.verify_context(history, item.context)
        item.segment.verify_file(runtime.draft.run_dir)
        history = history.append(item.segment)
        renderer.append(history)
        rows.append(row)
    require(selected == list(full_positions or late_positions), 'fixed intervention positions drifted')
    return history, dict(arm=arm, selected_chunks=selected, chunks=rows,
        synthetic_control_is_erank_router=False,
        changed_sampling_policy=bool(late_positions) or arm in GUIDANCE_ARMS)


def check_pairing(cells, *, arms=ARMS, prompt_ids=PROMPT_IDS, chunk_count=CHUNKS):
    expected = {(a, p) for a in arms for p in prompt_ids}
    require(len(cells) == len(expected) and {(x['arm'], x['prompt_id']) for x in cells} == expected,
        'incomplete or duplicate late screening population')
    for prompt in prompt_ids:
        rows = [x for x in cells if x['prompt_id'] == prompt]
        require(len({x['root_latent_sha256'] for x in rows}) == 1
            and all(x['initial_noise_sha256'] == rows[0]['initial_noise_sha256'] for x in rows)
            and len(rows[0]['initial_noise_sha256']) == chunk_count,
            'screening changed its full Target root or per-chunk input noises')


def summarize(cells, *, arms=ARMS, prompt_ids=PROMPT_IDS, chunk_count=CHUNKS):
    from .main_table_v1 import QUALITY_DIMENSIONS
    check_pairing(cells, arms=arms, prompt_ids=prompt_ids, chunk_count=chunk_count)
    for x in cells:
        require(set(x['quality_scores']) == set(QUALITY_DIMENSIONS)
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                    for v in x['quality_scores'].values())
            and type(x['video_ready_wall_seconds']) in (int, float)
            and math.isfinite(x['video_ready_wall_seconds'])
            and type(x['generation_wall_seconds']) in (int, float)
            and math.isfinite(x['generation_wall_seconds'])
            and 0 < x['generation_wall_seconds'] <= x['video_ready_wall_seconds'],
            'invalid screening quality or completed-ready timing')
    result = {}
    for arm in arms:
        rows = [x for x in cells if x['arm'] == arm]
        result[arm] = dict(quality_mean_100=math.fsum(
            math.fsum(x['quality_scores'].values()) / 6 for x in rows) / len(rows) * 100,
            dimensions_100={d: math.fsum(x['quality_scores'][d] for x in rows) / len(rows) * 100
                for d in QUALITY_DIMENSIONS},
            ready_mean_seconds=math.fsum(x['video_ready_wall_seconds'] for x in rows) / len(rows))
    for value in result.values():
        value['speedup_ratio_of_mean_ready_times'] = result['target']['ready_mean_seconds'] / value['ready_mean_seconds']
        value['quality_delta_vs_target'] = value['quality_mean_100'] - result['target']['quality_mean_100']
        value['dynamic_degree_delta_vs_target'] = (value['dimensions_100']['dynamic_degree']
            - result['target']['dimensions_100']['dynamic_degree'])
    return result


def trial_population(prompts, *, full_target_spacing, spacing_cohort_arm, late_spread_cohort=False):
    if late_spread_cohort:
        require(len(prompts) == len({p['id'] for p in prompts}) == COHORT_PROMPT_COUNT,
            'late spread cohort requires the complete frozen population')
        return prompts, CONTROLS + (LATE_SPREAD_ARM,)
    if spacing_cohort_arm is not None:
        require(spacing_cohort_arm in SPACING_COHORT_ARMS, 'unknown cohort placement')
        require(len(prompts) == len({p['id'] for p in prompts}) == COHORT_PROMPT_COUNT,
            'spacing cohort requires the complete frozen population')
        arms = (CONTROL_ARM, spacing_cohort_arm) if spacing_cohort_arm in GUIDANCE_ARMS else (spacing_cohort_arm,)
        return prompts, CONTROLS + arms
    selected = [p for p in prompts if p['id'] in PROMPT_IDS]
    require(tuple(p['id'] for p in selected) == PROMPT_IDS, 'screening prompt order drifted')
    return selected, SPACING_ARMS if full_target_spacing else ARMS


def run(*, gate_run, vbench_checkout, vbench_python, vbench_asset_manifest, vbench_asset_spec,
        full_target_spacing=False, spacing_cohort_arm=None, late_spread_cohort=False):
    from .accepted_stream_client import StreamDecoderProxy
    from .accepted_stream_worker import process_factory as stream_process_factory
    from .baseline_population import load_baseline_population, model_pair_for_draft
    from .correction_artifacts import verify_target_artifact_spec
    from .correction_runtime import launch_correction_runtime
    from .erank_acceptance_pilot import bound_file, QUALITY_DIMENSIONS
    from .longlive_late_quality_gate import verify_gate
    from .longlive_late_worker import process_factory
    from .main_table_baseline_campaign import (_run_dir, _model_root, _visible_devices,
        _sha256_file, _write_new_json, verify_artifact_spec)
    from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
    from .metadata import find_git_root, collect_git_metadata
    from .native_ready_report import codec_options
    from .native_vae_component import campaign_model, completed_render_binding, OVERLAP_READY_SCOPE
    from .stage_timing_probe import _hardware
    from .vbench_short import OfficialVBenchScorer
    from .vbench_long import VBenchLongVideoRequest
    root, model_root, repo = _run_dir(), _model_root(), find_git_root(Path.cwd())
    spacing = full_target_spacing or spacing_cohort_arm is not None
    require((gate_run is None) == spacing and not (full_target_spacing and spacing_cohort_arm is not None)
        and not (spacing and late_spread_cohort),
        'choose one gated late handoff or full-Target placement trial')
    full_cohort = spacing_cohort_arm is not None or late_spread_cohort
    seconds = LATE_SPREAD_SECONDS if late_spread_cohort or spacing_cohort_arm in THIRTY_SECOND_ARMS else 15
    chunk_count = duration_profile(seconds).chunk_count
    gate = None if spacing else verify_gate(gate_run, repo)
    require(campaign_model('longlive_1.3b', overlap=True) == 'longlive',
        'screening requires the coupled native LongLive renderer')
    prompts, population = load_baseline_population(repo, model_pair=model_pair_for_draft('longlive_1.3b'))
    prompts, all_arms = trial_population(prompts, full_target_spacing=full_target_spacing,
        spacing_cohort_arm=spacing_cohort_arm, late_spread_cohort=late_spread_cohort)
    prompt_ids = tuple(p['id'] for p in prompts)
    ds, ts = (repo / 'experiments' / n for n in
        ('erank3-longlive-policy-v1.artifacts.json', 'insight2-krea14b.artifacts.json'))
    _, dv, _ = verify_artifact_spec(ds, method_id='draft_only', model_root=model_root, repository_root=repo)
    _, tv, _ = verify_target_artifact_spec(ts, model_root=model_root, repo_root=repo)
    devices = _visible_devices(); _assert_idle_pair(*devices)
    hardware = _hardware(devices)
    cells, requests, paired = [], [], {}
    if spacing_cohort_arm in GUIDANCE_ARMS:
        from .longlive_motion_continuation import process_factory as stream_process_factory
    phases = (('controls', all_arms),) if spacing else (
        ('controls', CONTROLS), ('late', all_arms[len(CONTROLS):]))
    for phase, arms in phases:
        phase_root = root / phase; phase_root.mkdir()
        _assert_idle_pair(*devices)
        runtime = launch_correction_runtime(run_dir=phase_root, repo_root=repo,
            draft_artifact_spec_path=ds, draft_artifact_spec_sha256=_sha256_file(ds)[0],
            draft_model_id='longlive_1.3b', draft_verified_artifacts=dv,
            target_artifact_spec_path=ts, target_artifact_spec_sha256=_sha256_file(ts)[0],
            target_verified_artifacts=tv, devices=devices, chunk_count=chunk_count,
            write_full_videos=True,
            process_factory=stream_process_factory if spacing else process_factory,
            conditioning_prompts=[{'prompt_id':p['id'], 'prompt':p['text']} for p in prompts],
            shared_precomputed_conditioning=True)
        try:
            runtime.evaluator.full_quality_metric_id = 'vbench_custom_six_dimension_macro_mean'
            renderers = {role: StreamDecoderProxy(runtime.draft, worker, phase_root, chunk_count=chunk_count)
                for role, worker in (('draft', runtime.draft), ('target', runtime.target))}
            for arm in arms:
                renderer = renderers['draft' if arm == 'target' else 'target']
                runtime.evaluator.client = renderer
                for prompt in prompts:
                    check = lambda: _assert_worker_pair_owns_gpus(runtime=runtime,
                        draft_device=devices[0], target_device=devices[1], phase='late Target screening')
                    started = time.perf_counter()
                    history, trajectory = materialize(runtime,
                        PromptCase(prompt['id'], prompt['text'], 'evaluation'), arm, renderer,
                        check=check, seconds=seconds)
                    generation = time.perf_counter() - started
                    mid = 'll-late-' + arm + '-' + prompt['id']
                    prepared = runtime.evaluator.prepare_full(history=history, measurement_id=mid)
                    ready = time.perf_counter() - started
                    identity = (history.segments[0].latent_sha256,
                        [x['initial_noise_sha256'] for x in trajectory['chunks']])
                    require(paired.setdefault(prompt['id'], identity) == identity,
                        'fresh screening arms changed Target root or initial noises')
                    path = root / 'provenance' / 'longlive-late-quality' / (mid + '.json')
                    _write_new_json(path, dict(trajectory, input_root=phase, history=history.to_dict()))
                    video = bound_file(root, phase_root / prepared.video_path)
                    cell = dict(arm=arm, prompt_id=prompt['id'], prompt=prompt['text'], seed=SEED,
                        phase_base=phase, generation_wall_seconds=generation, video_ready_wall_seconds=ready,
                        timing_scope=OVERLAP_READY_SCOPE, root_latent_sha256=identity[0],
                        initial_noise_sha256=identity[1], video=video, native_trajectory=bound_file(root, path),
                        accepted_stream=completed_render_binding(renderer, history, prepared))
                    cells.append(cell)
                    requests.append(VBenchLongVideoRequest(video_path=root / video['path'],
                        prompt=prompt['text'], measurement_id=mid))
                    _write_new_json(root / 'provenance' / 'longlive-late-quality' / (mid + '-video-ready.json'), cell)
                    print('LONGLIVE_LATE_VIDEO_READY', json.dumps(cell), flush=True)
        finally:
            runtime.close()
    check_pairing(cells, arms=all_arms, prompt_ids=prompt_ids, chunk_count=chunk_count)
    common_codec = None
    if full_cohort:
        for cell in cells:
            codec = codec_options(root / cell['video']['path'])
            require(common_codec is None or codec == common_codec,
                'paired cohort changed actual scored MP4 encoder options')
            common_codec = codec
    _assert_idle_pair(*devices)
    scorer = OfficialVBenchScorer(run_dir=root, checkout=vbench_checkout, python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest, asset_spec=vbench_asset_spec, visible_device=devices[1])
    measurements = scorer.score_videos(requests)
    require(set(measurements) == {r.measurement_id for r in requests}, 'incomplete screening score coverage')
    for cell, request in zip(cells, requests):
        measurement = measurements[request.measurement_id]
        scores = dict(measurement.dimension_scores)
        require(set(scores) == set(QUALITY_DIMENSIONS), 'screening score dimensions drifted')
        cell.update(quality_scores=scores, quality_evidence=bound_file(root, root / measurement.evidence_path))
    if late_spread_cohort:
        schema = 'longlive_late_spread_cohort_v1'
    elif spacing_cohort_arm is not None:
        schema = 'longlive_full_target_spacing_cohort_v1'
    elif spacing:
        schema = 'longlive_full_target_spacing_trial_v1'
    else:
        schema = 'longlive_late_development_trial_v1'
    result = dict(schema=schema, status='succeeded',
        formal_main_table_result=spacing_cohort_arm in FULL_TARGET_POSITIONS,
        held_out=False, synthetic_control_is_erank_router=False,
        prompt_selection=('main24_method_development_not_held_out' if full_cohort
            else 'failure018_control042'), gate=gate, git=collect_git_metadata(repo),
        hardware=hardware, population=population, arms=list(all_arms), rows=cells,
        comparison_summary=summarize(cells, arms=all_arms, prompt_ids=prompt_ids, chunk_count=chunk_count),
        target_reference_backend='persistent_native_session',
        timings_are_descriptive_development_only=True,
        noise_policy=('shared_initial_noises' if spacing
            else 'shared_initial_noises_and_explicit_policy_stage_renoise_bank'),
        model_order_comparison=False)
    if spacing:
        result['full_target_positions'] = {a: full_target_positions(a) for a in all_arms[len(CONTROLS):]}
    if full_cohort:
        result.update(fixed_placement_cohort=True, prompt_ids=list(prompt_ids),
            common_codec_options=common_codec)
    if late_spread_cohort or spacing_cohort_arm in THIRTY_SECOND_ARMS:
        result.update(nominal_seconds=seconds, chunk_count=chunk_count)
    if late_spread_cohort:
        result['late_target_positions'] = {LATE_SPREAD_ARM: list(LATE_SPREAD_POSITIONS)}
    _write_new_json(root / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--gate-run', type=Path)
    mode.add_argument('--full-target-spacing', action='store_true')
    mode.add_argument('--full-target-spacing-cohort', dest='spacing_cohort_arm', choices=SPACING_COHORT_ARMS)
    parser.add_argument('--late-spread-cohort', action='store_true')
    for name in ('vbench-checkout', 'vbench-python', 'vbench-asset-manifest', 'vbench-asset-spec'):
        parser.add_argument('--' + name, type=Path, required=True)
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        print(json.dumps(run(**vars(parser.parse_args()))))
    finally:
        shutdown_owned_runtime_helpers(label='longlive-late-quality')


if __name__ == '__main__':
    main()
