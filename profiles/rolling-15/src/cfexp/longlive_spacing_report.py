"""Verify original sealed LongLive spacing results without running either model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from . import native_ready_report as audit
from .baseline_population import load_baseline_population
from .correction_state import LatentHistory
from .duration_sweep import duration_profile
from .krea_chunk_backend import MAIN_TABLE_FOUR_STEP_SCHEDULE, validate_native_correction_state
from .longlive_late_quality import (
    CHUNKS, FULL_TARGET_POSITIONS, LATE_SPREAD_ARM, LATE_SPREAD_POSITIONS,
    LATE_SPREAD_SECONDS, SEED, THIRTY_SECOND_ARMS, check_pairing, full_target_positions,
    summarize, trial_population,
)
from .longlive_motion_continuation import (
    AMPLITUDE_ARM, AMPLITUDE_OPTIONS, CONTROL_ARM, FLOOR_ARM, FLOOR_OPTIONS, GAIN_BY_LAG,
    GUIDANCE_ARMS, TARGET_POSITIONS, receipt, reference_chunk,
)
from .native_reference_report import official_scores
from .video_only_decode import SCHEMA as DECODE_SCHEMA


def trial_contract(result, manifest, repo):
    argv = manifest['spec']['command']['argv']
    env = manifest['spec']['command']['env']
    options = manifest['spec']['parameters']
    cohort = '--full-target-spacing-cohort' in argv
    audit._require('cfexp.longlive_late_quality' in argv
        and '--gate-run' not in argv and cohort != ('--full-target-spacing' in argv),
        'manifest does not select a LongLive spacing trial')
    renderer = dict(CFEXP_NATIVE_VIDEO_STREAM_MODEL='longlive', CFEXP_ACCEPTED_VIDEO_STREAM='1',
        CFEXP_RENDER_TARGET_PRIORITY='1', CFEXP_VIDEO_ONLY_DECODE='1',
        CFEXP_VIDEO_ENCODING='serial_abr_v1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    audit._require(all(env.get(k) == v for k, v in renderer.items()),
        'spacing manifest does not bind the native video-ready renderer')
    arm = argv[argv.index('--full-target-spacing-cohort') + 1] if cohort else None
    population, binding = load_baseline_population(repo, model_pair='krea_longlive')
    prompts, arms = trial_population(population, full_target_spacing=not cohort,
        spacing_cohort_arm=arm)
    positions = {a: list(full_target_positions(a)) for a in arms if full_target_positions(a)}
    chunk_count = duration_profile(LATE_SPREAD_SECONDS).chunk_count if arm in THIRTY_SECOND_ARMS else CHUNKS
    budget = len(full_target_positions(arm)) if cohort else 4
    main_table = arm in FULL_TARGET_POSITIONS
    schema = 'longlive_full_target_spacing_cohort_v1' if cohort else 'longlive_full_target_spacing_trial_v1'
    scope = 'main24_method_development_not_held_out' if cohort else 'failure018_control042'
    ids = [p['id'] for p in prompts]
    audit._require(result['schema'] == schema and result['status'] == 'succeeded'
        and result['population'] == binding and result['prompt_selection'] == scope
        and result['formal_main_table_result'] is main_table
        and result['synthetic_control_is_erank_router'] is False and result['held_out'] is False
        and result['target_reference_backend'] == 'persistent_native_session'
        and result['noise_policy'] == 'shared_initial_noises' and result['gate'] is None
        and result['arms'] == options['arms'] == list(arms)
        and result['full_target_positions'] == options['full_target_positions'] == positions
        and options['prompt_ids'] == ids and options['seed'] == SEED and options['chunk_count'] == chunk_count
        and options['formal_main_table_result'] is main_table and options['held_out'] is False
        and options['erank_routing'] is False and options['target_budget_after_root'] == budget,
        'spacing result or population differs from the frozen invocation')
    if cohort:
        audit._require(result['fixed_placement_cohort'] is True and result['prompt_ids'] == ids,
            'spacing cohort lacks its frozen prompt identities')
    if arm in THIRTY_SECOND_ARMS:
        audit._require(result['nominal_seconds'] == options['nominal_seconds'] == LATE_SPREAD_SECONDS
            and result['chunk_count'] == chunk_count, '30-second placement duration differs')
    if arm in GUIDANCE_ARMS:
        audit._require(options['motion_continuation_gain_by_lag'] == list(GAIN_BY_LAG),
            'guided Draft gain schedule differs from the manifest')
    if arm == FLOOR_ARM:
        audit._require(options['motion_floor'] == FLOOR_OPTIONS,
            'guided Draft floor differs from the manifest')
    if arm == AMPLITUDE_ARM:
        audit._require(options['motion_amplitude'] == AMPLITUDE_OPTIONS,
            'guided Draft amplitude differs from the manifest')
    return prompts, arms


def verify_route(cell, trajectory, history, count):
    arm = cell['arm']
    chunks = trajectory['chunks']
    selected = full_target_positions(arm)
    audit._require(trajectory['arm'] == arm and trajectory['input_root'] == 'controls'
        and trajectory['selected_chunks'] == list(selected)
        and history.prompt_id == cell['prompt_id'] and history.trajectory_seed == SEED
        and history.next_chunk_index == len(history.segments) == len(chunks) == count
        and history.segments[0].latent_sha256 == cell['root_latent_sha256']
        and [x['initial_noise_sha256'] for x in chunks] == cell['initial_noise_sha256']
        and [x['chunk_index'] for x in chunks] == list(range(count)),
        'trajectory does not bind the original spacing route and noise')
    if arm == 'target':
        target = trajectory['full_target_native_evidence']
        audit._require(target['target_session_policy'] == 'persistent_native_session'
            and target['target_session_count'] == 1 and target['history_reconstruction_calls'] == 0
            and target['draft_model_calls'] == 0 and target['target_model_calls'] == count
            and target['history_manifest_sha256'] == history.manifest_sha256
            and target['history_head_sha256'] == history.head_sha256
            and [x['latent_sha256'] for x in target['timing']['per_chunk']]
                == [x.latent_sha256 for x in history.segments],
            'full Target reference did not preserve one native session')
    session = None
    last_draft = None
    for n, (step, segment) in enumerate(zip(chunks, history.segments)):
        use_target = arm == 'target' or n == 0 or n in selected
        if use_target:
            audit._require(step['action'] == 'full_target' and segment.producer_role == 'target',
                'spacing trajectory changed a declared Target block')
            if arm != 'target':
                native = step['native_state']
                validate_native_correction_state(native, chunk_index=n)
                if arm in THIRTY_SECOND_ARMS:
                    audit._require(step['target_denoising_schedule'] == list(MAIN_TABLE_FOUR_STEP_SCHEDULE),
                        'full spread did not execute four Target updates')
                if n:
                    audit._require(native['native_session_id'] == session and not native['session_recreated'],
                        'fixed placement recreated its native Target correction session')
                session = native['native_session_id']
        else:
            missing = n if last_draft is None else n - last_draft - 1
            reuse = step['reuse']
            audit._require(step['action'] == 'draft' and segment.producer_role == 'draft'
                and reuse['replayed_chunks'] == missing and reuse['reused_chunks'] == n - missing,
                'Draft cache did not reuse the accepted prefix outside the missing Target suffix')
            last_draft = n
            ref = reference_chunk(n) if arm in GUIDANCE_ARMS else None
            actual = step.get('motion_continuation')
            if ref is None:
                audit._require(actual is None, 'motion continuation ran outside the declared window')
            else:
                anchor = history.segments[ref]
                audit._require(anchor.producer_role == 'target'
                    and actual == dict(receipt(n, ref, arm=arm), reference_latent_sha256=anchor.latent_sha256),
                    'motion continuation changed its accepted Target reference or native forward counts')


def late_trial_contract(result, manifest, repo):
    command = manifest['spec']['command']
    argv, env = command['argv'], command['env']
    options = manifest['spec']['parameters']
    renderer = dict(CFEXP_NATIVE_VIDEO_STREAM_MODEL='longlive', CFEXP_ACCEPTED_VIDEO_STREAM='1',
        CFEXP_RENDER_TARGET_PRIORITY='1', CFEXP_VIDEO_ONLY_DECODE='1',
        CFEXP_VIDEO_ENCODING='serial_abr_v1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    audit._require('cfexp.longlive_late_quality' in argv and '--late-spread-cohort' in argv
        and '--gate-run' in argv and all(env.get(k) == v for k, v in renderer.items()),
        'manifest does not select the gated native late-spread trial')
    prompts, binding = load_baseline_population(repo, model_pair='krea_longlive')
    prompts, arms = trial_population(prompts, full_target_spacing=False,
        spacing_cohort_arm=None, late_spread_cohort=True)
    ids = [p['id'] for p in prompts]
    positions = {LATE_SPREAD_ARM: list(LATE_SPREAD_POSITIONS)}
    profile = duration_profile(LATE_SPREAD_SECONDS)
    approval = json.loads((repo/'experiments/longlive-late-quality-gate-v1.json').read_text())
    gate = result['gate']
    audit._require(result['schema'] == 'longlive_late_spread_cohort_v1'
        and result['status'] == 'succeeded' and result['population'] == binding
        and result['prompt_selection'] == 'main24_method_development_not_held_out'
        and result['fixed_placement_cohort'] is True and result['formal_main_table_result'] is False
        and result['synthetic_control_is_erank_router'] is False and result['held_out'] is False
        and result['target_reference_backend'] == 'persistent_native_session'
        and result['noise_policy'] == 'shared_initial_noises_and_explicit_policy_stage_renoise_bank'
        and result['arms'] == options['arms'] == list(arms)
        and result['late_target_positions'] == options['late_target_positions'] == positions
        and result['prompt_ids'] == options['prompt_ids'] == ids and options['seed'] == SEED
        and result['nominal_seconds'] == options['nominal_seconds'] == LATE_SPREAD_SECONDS
        and result['chunk_count'] == options['chunk_count'] == profile.chunk_count
        and options['formal_main_table_result'] is False and options['held_out'] is False
        and options['erank_routing'] is False and options['non_root_target_updates'] == len(LATE_SPREAD_POSITIONS)
        and gate['report_sha256'] == approval['publication_pending_seal']['report_sha256'],
        'late-spread result differs from the frozen invocation or CUDA gate')
    return prompts, arms


def verify_late_route(cell, trajectory, history, count, phase):
    from .longlive_late_handoff import validate_trace
    from .longlive_late_worker import validate_proposal

    chunks = trajectory['chunks']
    audit._require(trajectory['arm'] == cell['arm'] == LATE_SPREAD_ARM
        and trajectory['input_root'] == cell['phase_base'] == 'late'
        and trajectory['selected_chunks'] == list(LATE_SPREAD_POSITIONS)
        and trajectory['changed_sampling_policy'] is True
        and history.prompt_id == cell['prompt_id'] and history.trajectory_seed == SEED
        and len(chunks) == history.next_chunk_index == count
        and history.segments[0].latent_sha256 == cell['root_latent_sha256']
        and [x['initial_noise_sha256'] for x in chunks] == cell['initial_noise_sha256']
        and [x['chunk_index'] for x in chunks] == list(range(count)),
        'late-spread trajectory changed its prompt, positions or initial noise')
    root = chunks[0]
    native = root['native_state']
    validate_native_correction_state(native, chunk_index=0)
    audit._require(root['action'] == 'full_target' and history.segments[0].producer_role == 'target',
        'late-spread trajectory lost its native Target root')
    for n, (step, segment) in enumerate(zip(chunks[1:], history.segments[1:], strict=True), 1):
        if n not in LATE_SPREAD_POSITIONS:
            reuse = step['reuse']
            audit._require(step['action'] == 'draft' and segment.producer_role == 'draft'
                and reuse['replayed_chunks'] == 0 and reuse['reused_chunks'] == n,
                'late-spread Draft replayed an accepted prefix')
            continue
        proposal, target, commit = (step[k] for k in ('proposal', 'target', 'commit'))
        validate_proposal(proposal)
        b, prediction, accepted = proposal['input_binding'], target['result'], commit['result']
        audit._require(step['action'] == 'late_target' and segment.producer_role == 'target'
            and b['prompt'] == cell['prompt'] and b['policy'] == 'd3_t625' and b['chunk_index'] == n
            and b['hashes'] == [s.latent_sha256 for s in history.segments[:n]] + [step['initial_noise_sha256']]
            and target['proposal'] == proposal and prediction['native_session_id'] == native['native_session_id']
            and prediction['denoising_schedule'] == [625.] and prediction['context_frames'] == 3
            and prediction['accepted_history_frames'] == n*3 and prediction['completed_four_step_target_chunk'] is False
            and prediction['forward_trace'] == [dict(timestep=t,current_start=4680,
                batch_timesteps_equal=True) for t in (0.,625.)]
            and prediction['latent']['sha256'] == commit['accepted_latent_sha256'] == segment.latent_sha256
            and accepted['draft_clean_forwards'] == 1 and accepted['history_replay_chunks'] == 0
            and accepted['committed_kv_width'] == 3 and accepted['committed_chunks'] == n+1
            and accepted['branch_snapshot_bytes'] == 0
            and 'proposal_oracle' not in proposal['result'] and 'target_oracle' not in prediction
            and 'commit_oracle' not in accepted, 'late-spread native handoff receipts differ')
        validate_trace(accepted['forward_trace'], [0.], start=n*4680)
        for path, sha in zip(b['renoise_paths'], b['renoise_sha256'], strict=True):
            audit.referenced_file(phase, dict(path=path, sha256=sha), 'late-spread re-noising input')


def verify_cell(src, cell, prompt, profile, short_batches):
    arm = cell['arm']
    mid = 'll-late-' + arm + '-' + cell['prompt_id']
    phase_base = 'late' if arm == LATE_SPREAD_ARM else 'controls'
    audit._require(cell['seed'] == SEED and cell['phase_base'] == phase_base
        and cell['prompt'] == prompt['text'] and cell['timing_scope'] == audit.OVERLAP_READY_SCOPE,
        'spacing score row changed prompt, seed, or timing scope')
    evidence = audit.read_json(src, 'provenance/longlive-late-quality/' + mid + '-video-ready.json')
    audit._require(evidence == {k: v for k, v in cell.items() if k not in ('quality_scores', 'quality_evidence')},
        'spacing score row differs from its sealed video-ready evidence')
    trajectory = audit._bound(src.frozen, src.prefix, cell['native_trajectory'], 'spacing trajectory')
    history = LatentHistory.from_dict(trajectory['history'])
    phase = SimpleNamespace(frozen=src.frozen, prefix=src.prefix+phase_base+'/', run=src.run)
    if arm == LATE_SPREAD_ARM:
        verify_late_route(cell, trajectory, history, profile.chunk_count, phase)
    else:
        verify_route(cell, trajectory, history, profile.chunk_count)
    for segment in history.segments:
        audit.referenced_file(phase, dict(path=segment.latent_path, sha256=segment.latent_sha256,
            nbytes=segment.latent_nbytes), 'accepted spacing latent')
    seconds = profile.nominal_seconds
    video, bound, codec = audit.quality(src, cell, cell['quality_evidence'], prompt, mid, seconds)
    audit._require(video == src.run / src.prefix / cell['video']['path']
        and all(bound[k] == cell['video'][k] for k in ('sha256', 'nbytes')),
        'spacing score used a different video')
    official_scores(src, cell, prompt, mid, seconds, video, short_batches=short_batches)
    audit.verify_geometry(video, profile)
    receipt = audit._bound(src.frozen, phase.prefix, cell['accepted_stream'], 'spacing renderer')
    audit._require(all(receipt['video'][k] == bound[k] for k in ('sha256', 'nbytes')),
        'spacing render and scored video bytes differ')
    audit.verify_accepted_renderer(phase, cell['accepted_stream'], history,
        dict(schema=DECODE_SCHEMA, pixels=receipt['pixels'], video=receipt['video']),
        model='longlive', role='draft' if arm == 'target' else 'target', profile=profile)
    return codec, history


def report(run, *, repo, commit, manifest):
    audit._require(not run.is_symlink(), 'spacing run is a symlink')
    frozen, marker = audit._load_frozen_index(run)
    run = frozen.run_dir
    audit._require(marker['git_sha'] == commit and marker['run_id'] == run.name,
        'spacing seal has a different generation revision or run identity')
    envelope = audit._load_json(frozen.file('resolved-manifest.json', label='manifest'), label='manifest')
    audit._require(envelope['manifest'] == audit.source_manifest(repo, commit, manifest)
        and all(envelope[k] == marker[k] for k in ('run_id', 'attempt_id', 'spec_hash')),
        'spacing invocation differs from the published source')
    terminal = audit._load_json(frozen.file('run.json', label='terminal run'), label='terminal run')
    audit._require(terminal['execution_status'] == 'SUCCEEDED' and terminal['exit_code'] == 0
        and terminal['run_id'] == run.name and terminal['latest_attempt_id'] == marker['attempt_id'],
        'spacing terminal identity or execution differs')
    src = SimpleNamespace(frozen=frozen, prefix='attempts/'+marker['attempt_id']+'/', run=run)
    result = audit.read_json(src, 'result.json')
    audit._require(result['git']['sha'] == commit, 'spacing result changed its generation revision')
    gpus = result['hardware']['gpus']
    audit._require(len(gpus) == 2 and len({x['uuid'] for x in gpus}) == 2
        and all(x['uuid'].startswith('GPU-') for x in gpus),
        'spacing result has no distinct physical GPU pair')
    late_spread = '--late-spread-cohort' in envelope['manifest']['spec']['command']['argv']
    if late_spread:
        prompts, arms = late_trial_contract(result, envelope['manifest'], repo)
    else:
        prompts, arms = trial_contract(result, envelope['manifest'], repo)
    long_horizon = late_spread or any(arm in THIRTY_SECOND_ARMS for arm in arms)
    profile = duration_profile(LATE_SPREAD_SECONDS if long_horizon else 15)
    by_id = {p['id']: p for p in prompts}
    rows = result['rows']
    check_pairing(rows, arms=arms, prompt_ids=tuple(by_id), chunk_count=profile.chunk_count)
    codec = None
    short_batches = {}
    common_noise = {}
    model_artifacts = {}
    motion_prefixes = {}
    for cell in rows:
        prompt = by_id[cell['prompt_id']]
        options, history = verify_cell(src, cell, prompt, profile, short_batches)
        if any(a in GUIDANCE_ARMS for a in arms) and cell['arm'] in (CONTROL_ARM, *GUIDANCE_ARMS):
            initial = tuple(x.latent_sha256 for x in history.segments[:TARGET_POSITIONS[1]+1])
            audit._require(motion_prefixes.setdefault(prompt['id'], initial) == initial,
                'the native Target prefix changed before the first guided Draft block')
        audit._require(codec is None or codec == options, 'spacing arms used different encoder options')
        codec = options
        noise = [s.initial_noise_descriptor_sha256 for s in history.segments]
        audit._require(common_noise.setdefault(prompt['id'], noise) == noise,
            'spacing arms used different initial noise descriptors')
        for segment in history.segments:
            identity = (segment.model_id, segment.model_artifact_sha256)
            audit._require(model_artifacts.setdefault(segment.producer_role, identity) == identity,
                'spacing arms changed a model checkpoint')
    if result.get('fixed_placement_cohort') is True:
        audit._require(codec == result['common_codec_options'], 'cohort encoder evidence differs')
    summary = summarize(rows, arms=arms, prompt_ids=tuple(by_id), chunk_count=profile.chunk_count)
    declared = result['comparison_summary']
    # The initial two-prompt revision recorded dynamic degree without its delta.
    checks = {a: ({k: v for k, v in s.items() if k != 'dynamic_degree_delta_vs_target'}
        if not result['formal_main_table_result'] and 'dynamic_degree_delta_vs_target' not in declared[a]
        else s) for a, s in summary.items()}
    audit._require(checks == declared, 'spacing comparison summary differs from its scored rows')
    return dict(schema='longlive_spacing_verified_report_v1', status='verified_original_local_seal',
        generation_revision=commit, run_id=marker['run_id'], attempt_id=marker['attempt_id'],
        artifact_index_sha256=marker['artifact_index_sha256'],
        marker_sha256=audit._sha256_file(run/'_SUCCESS.json')[0],
        publication_verified=False, prompt_selection=result['prompt_selection'],
        formal_main_table_result=result['formal_main_table_result'],
        full_target_positions=result.get('full_target_positions', {}), comparison_summary=summary,
        **({'nominal_seconds':profile.nominal_seconds} if long_horizon else {}),
        **({'late_target_positions':result['late_target_positions']} if late_spread else {}),
        rows=[{k: x[k] for k in ('arm', 'prompt_id', 'prompt', 'quality_scores',
            'video_ready_wall_seconds', 'video')} for x in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'repo'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('commit', 'manifest'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(report(**vars(parser.parse_args())), sort_keys=True))


if __name__ == '__main__':
    main()
