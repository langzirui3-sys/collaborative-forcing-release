"""Read-only, fail-closed native LongLive/Rolling video-ready comparison.

Only consumes successfully published formal queue tasks. Indexed artifacts are
verified against their original seals. Excluded local latents/decode sidecars
must match hashes transitively bound by those sealed records. No seal is changed.
The Target's legacy population model-pair label is retained as a warning while
its actual prompt texts and native decoder identity are independently checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from types import SimpleNamespace

from .baseline_population import load_baseline_population, model_pair_for_draft
from .baseline_shard_report import _bound, _published, source_manifest
from .correction_state import LatentHistory
from .duration_sweep import duration_profile
from .insight2_campaign import PendingSequentialRollout
from .main_table_merge import MainTableMergeError, _load_frozen_index, _load_json, _require, _sha256_file
from .main_table_v1 import QUALITY_DIMENSIONS
from .longlive_motion import MOTION_ENERGY_FLOOR, latent_motion_energy
from .native_vae_component import OVERLAP_READY_SCOPE, validate_identity
from .native_video_ready_target import TIMING_SCOPE
from .rank_burst import (
    NATIVE_BURST_WORD,
    SPLIT_BURST_WORD,
    valid_split_burst_count,
    validate_burst_schedule,
)

MODELS = {'longlive': 'longlive_1.3b', 'rolling': 'rolling_forcing_1.3b'}


def scheduling_revision(candidate_revision, target, repo):
    """Prove that a Target retry only changed PBS duration estimates."""
    target_revision = target.result['code_revision']
    if candidate_revision == target_revision:
        return {}
    for revision in (candidate_revision, target_revision):
        _require(re.fullmatch(r'[0-9a-f]{40}', revision) is not None,
                 'full generation revision required')
    env = dict(PATH='/usr/bin:/bin', HOME=os.path.expanduser('~'),
               GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_SYSTEM='/dev/null',
               GIT_CONFIG_NOSYSTEM='1', GIT_NO_REPLACE_OBJECTS='1')

    def git(*args):
        try:
            return subprocess.check_output(
                ['git', '--no-replace-objects', '-C', str(repo), *args],
                env=env, stderr=subprocess.PIPE, text=True).strip()
        except subprocess.CalledProcessError as error:
            raise MainTableMergeError('generation revisions cannot be verified') from error

    _require(git('rev-list', '--parents', '-n', '1', target_revision).split()
             == [target_revision, candidate_revision],
             'generation revisions are not a direct scheduling-only retry')
    changes = git('diff-tree', '-r', '--no-commit-id', '--no-renames', '--raw',
                  candidate_revision, target_revision).splitlines()
    budgets = {}
    for change in changes:
        fields, path = change.split('\t', 1)
        mode_before, mode_after, _, _, status = fields.split()
        _require(status == 'M' and mode_before == ':100644' and mode_after == '100644'
                 and re.fullmatch(r'experiments/native-stream-ready-[a-z0-9-]+\.manifest\.json', path),
                 'retry changed files other than native experiment manifests')
        old = json.loads(git('show', candidate_revision+':'+path))
        new = json.loads(git('show', target_revision+':'+path))
        old_minutes = old['spec']['resources'].pop('expected_duration_minutes')
        new_minutes = new['spec']['resources'].pop('expected_duration_minutes')
        _require(old == new and type(old_minutes) is int and type(new_minutes) is int
                 and 0 < new_minutes < old_minutes,
                 'retry changed generation configuration or invalid PBS duration')
        budgets[path] = {'original_minutes': old_minutes, 'retry_minutes': new_minutes}
    _require(getattr(target, 'task', {}).get('manifest') in budgets,
             'Target retry manifest is absent from the verified scheduling diff')
    return {'target_generation_code_revision': target_revision,
            'verified_scheduling_changes': budgets}


def positive(value, name):
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
             'invalid '+name)
    return value


def codec_options(path):
    # x264 writes its actual options into the MP4 SEI. Do not infer encoder
    # equality from equal manifest environment variables.
    raw = path.read_bytes()
    start = raw.find(b'x264 - core ')
    _require(start >= 0, 'scored MP4 lacks x264 option evidence')
    data = raw[start:start+4000].split(b'\x00', 1)[0]
    _require(b' options: ' in data, 'x264 option section missing')
    options = {}
    for token in data.split(b' options: ', 1)[1].decode('ascii').split():
        if '=' in token:
            key, value = token.split('=', 1)
            _require(key not in options, 'duplicate x264 option')
            options[key] = value
    _require(all(k in options for k in ('threads', 'lookahead_threads', 'sliced_threads',
                                       'rc', 'bitrate', 'bframes', 'keyint')),
             'incomplete x264 option evidence')
    return options


def source(queue, task_id, repo, commit, *, timing_scope=TIMING_SCOPE):
    plan = _load_json(queue, label='queue plan')
    state = _load_json(Path(plan['ops'])/'state.json', label='queue state')
    _require(plan.get('schema') == 'cfexp_erank_queue_v1' and
             state.get('plan_sha256') == _sha256_file(queue)[0], 'queue plan/state binding differs')
    tasks = [t for t in plan['tasks'] if t['id'] == task_id]
    _require(len(tasks) == 1 and tasks[0]['commit'] == commit, 'formal task/source identity differs')
    task = tasks[0]
    row = state['tasks'][task_id]
    _require(row['status'] == 'succeeded', 'formal task is not successfully published')
    run = Path(row['run_dir'])
    _require(not run.is_symlink() and run.resolve().parent == (Path(plan['root'])/'runs').resolve(),
             'run escaped queue root')
    frozen, marker = _load_frozen_index(run)
    frozen.verify_all()
    _published(row, run, marker)
    _require(marker['git_sha'] == commit, 'sealed generation commit differs')
    terminal = _load_json(frozen.file('run.json', label='terminal run'), label='terminal run')
    _require(terminal.get('execution_status') == 'SUCCEEDED' and terminal.get('exit_code') == 0
             and terminal.get('latest_attempt_id') == marker['attempt_id']
             and terminal.get('run_id') == marker['run_id'], 'terminal run identity differs')
    envelope = _load_json(frozen.file('resolved-manifest.json', label='manifest'), label='manifest')
    _require(all(envelope.get(k) == marker[k] for k in ('run_id', 'attempt_id', 'spec_hash'))
             and envelope['manifest'] == source_manifest(repo, commit, task['manifest']),
             'sealed invocation differs from published manifest')
    prefix = 'attempts/'+marker['attempt_id']+'/'
    result = _load_json(frozen.file(prefix+'result.json', label='result'), label='result')
    _require(result.get('status') == 'succeeded' and result.get('code_revision') == commit
             and result.get('timing_scope') == timing_scope, 'result revision/status/timing differs')
    return SimpleNamespace(frozen=frozen, marker=marker, prefix=prefix, result=result,
                           manifest=envelope['manifest'], task=task, run=run)


def read_json(src, relative):
    return _load_json(src.frozen.file(src.prefix+relative, label=relative), label=relative)


def referenced_file(src, binding, label):
    """Verify a regular local blob from a hash in a sealed parent record.

    The old publication policies intentionally omit raw latents and decoded
    tensors. A sealed history/ready record still binds their content hashes.
    This never adds an entry to the original index or trusts an unbound file.
    """
    relative = src.prefix+binding['path']
    posix = PurePosixPath(relative)
    digest = binding.get('sha256')
    size = binding.get('nbytes', binding.get('bytes'))
    _require(relative == str(posix) and not posix.is_absolute() and
             all(part not in ('', '.', '..') for part in posix.parts) and
             isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None and
             (size is None or (type(size) is int and size > 0)), 'unsafe/unbound '+label)
    if relative in src.frozen.entries:
        path = src.frozen.file(relative, label=label)
    else:
        path = src.frozen.run_dir
        for part in posix.parts:
            path = path/part
            _require(not path.is_symlink(), 'symlink in '+label)
        _require(path.is_file() and path.resolve().is_relative_to(src.frozen.run_dir),
                 'missing or escaping '+label)
    actual_digest, actual_size = _sha256_file(path)
    _require(actual_digest == digest and (size is None or actual_size == size),
             'transitive hash/size differs for '+label)
    if relative not in src.frozen.entries:
        if not hasattr(src, 'transitive_references'):
            src.transitive_references = {}
        src.transitive_references[relative] = dict(path=relative, sha256=digest, nbytes=actual_size)
    return path


def quality(src, row, proof, prompt, mid, seconds):
    scores = row['quality_scores']
    _require(set(scores) == set(QUALITY_DIMENSIONS) and all(type(v) in (int, float) and
             math.isfinite(v) and 0 <= v <= 1 for v in scores.values()), 'invalid quality dimensions')
    side = _bound(src.frozen, src.prefix, proof, 'quality evidence')
    _require(side.get('measurement_id') == mid and side.get('dimension_scores') == scores and
             abs(side.get('macro_mean', -1)-math.fsum(scores.values())/6) < 1e-10,
             'quality aggregate/measurement binding differs')
    if seconds == 60:
        config = _bound(src.frozen, src.prefix, side['batch_config'], 'VBench-Long config')
        outcome = _bound(src.frozen, src.prefix, side['batch_attempt'], 'VBench-Long attempt')
        _bound(src.frozen, src.prefix, side['official_result'], 'VBench-Long result')
        matches = [x for x in config['inputs'] if x['measurement_id'] == mid]
        _require(len(matches) == 1 and matches[0]['prompt'] == prompt['text'] and
                 matches[0]['source_video'] == side['source_video'], 'long evaluator input differs')
        video = side['source_video']
    else:
        config = _bound(src.frozen, src.prefix, side['bindings']['config'], 'VBench config')
        outcome = _bound(src.frozen, src.prefix, side['bindings']['exit'], 'VBench exit')
        matches = [x for x in config['inputs'] if x['id'] == mid]
        _require(len(matches) == 1 and matches[0]['prompt'] == prompt['text'], 'short evaluator input differs')
        item = matches[0]
        video = dict(path=item['source'], sha256=item['sha256'], nbytes=item['nbytes'])
    _require(outcome.get('returncode') == 0, 'official evaluator failed')
    try:
        relative = Path(video['path']).relative_to(src.run).as_posix()
    except ValueError as error:
        raise ValueError('evaluator video escaped sealed run') from error
    path = src.frozen.file(relative, label='scored MP4')
    _require(_sha256_file(path) == (video['sha256'], video['nbytes']), 'scored MP4 binding differs')
    return path, video, codec_options(path)


def ours_history(src, prompt_id, count, provenance, *, burst_length=4, observation_horizon=8,
                 observation_offset=0, opening_target_burst=False, split_target_burst=False,
                 split_burst_count=2, final_window_at_tail=False, motion_trigger=False):
    matches = []
    for name in src.frozen.entries:
        if name.startswith(src.prefix+'measurements/rollouts/') and name.endswith('.pending.json'):
            item = read_json(src, name[len(src.prefix):])
            if item['history']['prompt_id'] == prompt_id:
                matches.append(PendingSequentialRollout.from_dict(item))
    _require(len(matches) == 1, 'sealed pending rollout is missing or ambiguous')
    pending = matches[0]
    history = pending.history
    _require(history.next_chunk_index == count and history.trajectory_seed == 20260911 and
             history.head_sha256 == provenance['final_history_hash'] and
             pending.corrected_chunks == tuple(provenance['accepted_target_chunks']), 'accepted history differs')
    selected = provenance['accepted_target_chunks']
    audit = provenance['native_cache_audit']
    model = src.result['draft_model_id']
    targets = {0, *selected}
    budget = split_burst_count * burst_length if split_target_burst else burst_length
    _require(audit['schema'] == f'native_model_e4_p{observation_horizon}_k{budget}_cache_audit_v1' and
             audit.get('observation_offset', 0) == observation_offset and
             audit.get('opening_target_burst', False) is opening_target_burst and
             audit.get('split_target_burst', False) is split_target_burst and
             audit.get('split_burst_count', 2) == split_burst_count and
             audit.get('final_window_at_tail', False) is final_window_at_tail and
             audit.get('motion_trigger', False) is motion_trigger and
             audit['draft_model_id'] == model and audit['target_chunks'] == sorted(targets) and
             isinstance(audit['target_session_id'], str) and bool(audit['target_session_id']),
             'native producer cache witness identity differs')
    validate_burst_schedule(dict(decisions=audit['decisions'], accepted_chunks=selected,
        proposed_chunks=selected), budget=budget, burst_length=burst_length,
        chunk_count=count, observation_horizon=observation_horizon, observation_offset=observation_offset,
        opening_target_burst=opening_target_burst, split_target_burst=split_target_burst,
        final_window_at_tail=final_window_at_tail, motion_trigger=motion_trigger)
    expected = [i for i in range(count) if i not in targets]
    _require([s['chunk_index'] for s in audit['draft_states']] == expected,
             'native Draft cache witness coverage differs')
    previous = None
    for state, index in zip(audit['draft_states'], expected):
        missing = index if previous is None else index-previous-1
        if model == 'longlive_1.3b':
            _require(type(state['replayed_chunks']) is int and state['replayed_chunks'] == missing and
                     type(state['reused_chunks']) is int and state['reused_chunks'] == index-missing,
                     'LongLive sealed cache witness replay/reuse differs')
        elif model == 'rolling_forcing_1.3b':
            reused = previous is not None and missing == 0
            _require(state['state_policy'] == 'single_active_branch_clean_history_restart_v1' and
                     state['reused_rolling_state'] is reused and
                     type(state['replayed_history_blocks']) is int and
                     state['replayed_history_blocks'] == (0 if reused else index) and
                     state['discarded_future_state'] is (previous is not None and not reused),
                     'Rolling sealed cache witness continuation/restart differs')
        else:
            raise ValueError('unsupported native cache witness')
        previous = index
    for index, segment in enumerate(history.segments):
        _require(segment.producer_role == ('target' if index in targets else 'draft'),
                 'accepted history producer differs from native schedule')
        referenced_file(src, dict(path=segment.latent_path, sha256=segment.latent_sha256,
                                  nbytes=segment.latent_nbytes), 'accepted latent')
        if motion_trigger and index:
            energy = audit['decisions'][index - 1]['decision'].get('motion_energy')
            if energy is not None:
                _require(energy == latent_motion_energy(src.run / src.prefix, segment),
                         'motion energy differs from the sealed Draft latent')
    return history, audit


def verify_geometry(path, profile):
    import av
    with av.open(str(path)) as container:
        streams = container.streams.video
        _require(len(streams) == 1 and streams[0].average_rate == 24 and
                 streams[0].width == 832 and streams[0].height == 480, 'video geometry/FPS differs')
        _require(sum(1 for _ in container.decode(streams[0])) == profile.frame_count,
                 'decoded video horizon differs')


def verify_accepted_renderer(src, binding, history, decode, *, model, role, profile):
    from .video_only_decode import SCHEMA, validate_pixels

    relative = 'provenance/accepted-stream/' + history.manifest_sha256 + '.json'
    _require(binding.get('path') == relative, 'accepted renderer receipt belongs to another history')
    receipt = _bound(src.frozen, src.prefix, binding, 'accepted renderer receipt')
    decoder = receipt['decoder']
    if model == 'self-forcing':
        from .streaming_vae_probe import VAE_SHA, WEIGHTS_SHA
        _require(decoder.get('native_vae_sha256') == VAE_SHA
                 and decoder.get('checkpoint_sha256') == WEIGHTS_SHA
                 and decoder.get('separate_causal_decoder_state') is True
                 and 'coupled_native_identity' not in decoder, 'wrong native Self-Forcing decoder')
    else:
        validate_identity(decoder['coupled_native_identity'], model)
    validate_pixels(receipt['pixels'], profile.frame_count)
    _require(decode['schema'] == SCHEMA and decode['pixels'] == receipt['pixels']
             and receipt['decoder_and_encoder_joined'] is True
             and receipt['role_aware_priority'] is True
             and receipt['decoder_worker_role'] == role
             and receipt['history_manifest_sha256'] == history.manifest_sha256
             and receipt['history_head_sha256'] == history.head_sha256
             and receipt['latent_chunk_count'] == len(receipt['chunks']) == profile.chunk_count,
             'render receipt does not bind the complete decoded video')
    _require(all(receipt['video'][key] == decode['video'][key]
                 for key in ('sha256', 'nbytes', 'fps', 'frames')), 'scored and rendered MP4 differ')
    for index, (chunk, segment) in enumerate(zip(receipt['chunks'], history.segments)):
        _require(chunk['chunk_index'] == segment.chunk_index == index
                 and chunk['segment_sha256'] == segment.segment_sha256
                 and chunk['latent_sha256'] == segment.latent_sha256
                 and chunk['frame_start'] == max(0, index * 12 - 3)
                 and chunk['frame_count'] == (9 if index == 0 else 12),
                 'accepted renderer frame or latent sequence differs')


def compare(ours, target, *, repo, model, seconds, overlap=False,
            burst_length=4, observation_horizon=8, opening_target_burst=False,
            split_target_burst=False, split_burst_count=2, final_window_at_tail=False,
            motion_trigger=False):
    _require(model in MODELS and type(seconds) is int and seconds in (15, 30, 60),
             'unsupported native condition')
    _require(type(burst_length) is int and burst_length in (1, 2, 3, 4)
             and (burst_length == 4 or (overlap and (split_target_burst
                  or (burst_length == 2 and seconds == 15)))),
             'native burst audit requires a declared width and overlap control')
    _require(type(observation_horizon) is int and
             (observation_horizon == 8 or (observation_horizon == 2 and burst_length == 4
                                           and model == 'longlive' and seconds == 30 and overlap)),
             'early burst audit requires the 30-second LongLive four-block overlap control')
    _require(type(opening_target_burst) is bool and
             (not opening_target_burst or (model == 'longlive' and overlap
                                           and burst_length == 4 and observation_horizon == 8)),
             'opening burst audit requires the LongLive four-block overlap path')
    _require(type(split_target_burst) is bool and
             (not split_target_burst or (model == 'longlive' and overlap and burst_length in (1, 2, 3)
                                         and observation_horizon == 8 and not opening_target_burst)),
             'split burst audit requires the LongLive overlap path')
    _require(valid_split_burst_count(split_burst_count, burst_length=burst_length,
                                    seconds=seconds, split=split_target_burst),
             'unsupported audited Target burst count for the declared width and horizon')
    _require(type(final_window_at_tail) is bool and
             (not final_window_at_tail or (split_target_burst and split_burst_count == 3)),
             'tail-aligned observation requires the 30-second LongLive K6 route')
    _require(type(motion_trigger) is bool and
             (not motion_trigger or (split_target_burst and split_burst_count == 3 and not final_window_at_tail)),
             'motion trigger requires the original 30-second LongLive K6 route')
    budget = split_burst_count * burst_length if split_target_burst else burst_length
    pair = model_pair_for_draft(MODELS[model])
    prompts, population = load_baseline_population(repo, model_pair=pair)
    profile = duration_profile(seconds)
    a, b = ours.result, target.result
    timing_scope = OVERLAP_READY_SCOPE if overlap else TIMING_SCOPE
    strategy = f'committed_rank_burst{burst_length}_probe{observation_horizon}_k{budget}'
    if opening_target_burst:
        strategy += '_opening'
    if split_target_burst:
        strategy += '_split'
    if final_window_at_tail:
        strategy += '_tail'
    if motion_trigger:
        strategy += '_motion'
    _require(a.get('code_revision') == b.get('code_revision'), 'generation revisions differ')
    _require(a.get('schema') == 'erank_baseline24_result_v1' and a.get('method_id') == 'ours'
             and a.get('run_kind') == 'formal_campaign'
             and a.get('strategy') == strategy
             and a.get('draft_model_id') == MODELS[model] and a.get('seed_count') == 1
             and a.get('native_cache_audit_count') == a.get('prompt_count') == a.get('row_count') == 24,
             'Ours is not the declared native formal campaign')
    target_schema = 'native_overlap_video_ready_target_v1' if overlap else 'native_serial_video_ready_target_v1'
    _require(b.get('schema') == target_schema
             and b.get('population') == 'main24_method_development_not_held_out'
             and b.get('row_count') == 24, 'Target is not the formal native reference')
    expected_benchmark = 'VBench-Long' if seconds == 60 else 'VBench'
    argv = ours.manifest['spec']['command']['argv']
    manifest_burst_length = (int(argv[argv.index('--native-burst-length')+1])
                             if '--native-burst-length' in argv else 4)
    _require(manifest_burst_length == burst_length, 'native burst length differs from the invocation')
    horizon_option = '--native-burst-observation-horizon'
    manifest_horizon = int(argv[argv.index(horizon_option)+1]) if horizon_option in argv else 8
    _require(manifest_horizon == observation_horizon, 'native burst observation horizon differs from the invocation')
    _require(('--native-opening-target-burst' in argv) is opening_target_burst,
             'opening burst manifest differs from the invocation')
    _require(('--native-split-target-burst' in argv) is split_target_burst,
             'split burst manifest differs from the invocation')
    count_option = '--native-split-burst-count'
    manifest_count = int(argv[argv.index(count_option)+1]) if count_option in argv else 2
    _require(manifest_count == split_burst_count, 'split burst window count differs from the invocation')
    _require(('--native-final-window-at-tail' in argv) is final_window_at_tail,
             'tail observation window differs from the invocation')
    _require(('--native-motion-trigger' in argv) is motion_trigger,
             'motion trigger differs from the invocation')
    if overlap:
        required = {'CFEXP_NATIVE_VIDEO_STREAM_MODEL': model, 'CFEXP_ACCEPTED_VIDEO_STREAM': '1',
                    'CFEXP_RENDER_TARGET_PRIORITY': '1', 'CFEXP_VIDEO_ONLY_DECODE': '1',
                    'CFEXP_VIDEO_ENCODING': 'serial_abr_v1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
        for src in (ours, target):
            command = src.manifest['spec']['command']
            _require('--overlap-native-vae' in command['argv'] and src.result['accepted_stream_count'] == 24
                     and all(command['env'].get(key) == value for key, value in required.items()),
                     'campaign renderer and thread configuration differ')
    calibration_revision = argv[argv.index('--burst-calibration-revision')+1]
    ours_duration = dict(profile.metadata(), experiment_id=f'native-model-e4-p{observation_horizon}-k{budget}-v1',
        benchmark=expected_benchmark, ours_proposal_budget=budget,
        budget_rule=f'one_target_root_and_one_{NATIVE_BURST_WORD[burst_length]}_chunk_burst',
        calibration_policy='explicit_sealed_model_specific_C10',
        calibration_code_revision=calibration_revision, observation_horizon=observation_horizon)
    if opening_target_burst:
        ours_duration.update(opening_target_burst=True)
        ours_duration['experiment_id'] += '-opening'
    if split_target_burst:
        burst_word = SPLIT_BURST_WORD[split_burst_count]
        width = NATIVE_BURST_WORD[burst_length]
        ours_duration.update(split_target_burst=True,
                             budget_rule=f'one_target_root_and_{burst_word}_{width}_chunk_bursts')
        if split_burst_count != 2:
            ours_duration['split_burst_count'] = split_burst_count
        ours_duration['experiment_id'] += '-split'
    if final_window_at_tail:
        ours_duration['final_window_at_tail'] = True
        ours_duration['experiment_id'] += '-tail'
    if motion_trigger:
        ours_duration.update(motion_trigger=True, motion_energy_floor=MOTION_ENERGY_FLOOR)
        ours_duration['experiment_id'] += '-motion'
    for result, expected_duration in ((a, ours_duration), (b, profile.metadata())):
        _require(result['benchmark'] == expected_benchmark and result['duration'] == expected_duration
                 and result['timing_scope'] == timing_scope, 'horizon/benchmark/timing mismatch')
    _require(a['population'] == population, 'native Ours population differs')
    # Older Target code selected the common text section using the SF label.
    # Never relabel a sealed result. Verify every actual prompt independently.
    target_pop = b['population_binding']
    _require(all(target_pop[k] == population[k] for k in ('protocol', 'sha256', 'seed')),
             'Target population source/seed differs')
    _require(not overlap or target_pop == population, 'overlap Target population model label differs')
    preflight = read_json(ours, 'provenance/preflight.json')
    _require(len(b['hardware']['gpus']) == 2 and preflight['git']['sha'] == a['code_revision'] and preflight['gpu_uuids'] ==
             dict(zip(('draft','target'), (x['uuid'] for x in b['hardware']['gpus']))),
             'physical GPU pair/role differs')
    arows = _bound(ours.frozen, ours.prefix, a['raw_rows'], 'Ours raw rows')['rows']
    brows = b['rows']
    ids = [p['id'] for p in prompts]
    _require(len(arows) == len(brows) == len(a['vbench_measurements']) == 24 and
             [x['prompt_id'] for x in arows] == [x['prompt_id'] for x in brows] == ids,
             'formal population coverage/order differs')
    pairs, common_codec = [], None
    for prompt, x, y, proof in zip(prompts, arows, brows, a['vbench_measurements']):
        _require(x['seed'] == y['seed'] == 20260911 and y['prompt'] == prompt['text']
                 and x['method_id'] == 'ours' and x['duration'] == ours_duration
                 and y['native_decoder_model_id'] == MODELS[model], 'row input/backend differs')
        for row in (x, y):
            _require(row['timing_scope'] == timing_scope and
                     positive(row['video_ready_wall_seconds'], 'ready time') >=
                     positive(row['generation_wall_seconds'], 'generation time'), 'incomplete ready timing')
        qx = quality(ours, x, proof, prompt, 'mt-ours-'+prompt['id'], seconds)
        qy = quality(target, y, y['quality_evidence'], prompt, 'stream-target-'+prompt['id'], seconds)
        _require(qx[2] == qy[2] and (common_codec is None or common_codec == qx[2]),
                 'actual scored encoders differ')
        common_codec = qx[2]
        verify_geometry(qx[0], profile); verify_geometry(qy[0], profile)
        _require(qy[1]['sha256'] == y['video']['sha256'] and qy[1]['nbytes'] == y['video']['nbytes'],
                 'Target row and scored video differ')
        provenance = read_json(ours, 'provenance/ours/'+prompt['id']+'.json')
        _require(provenance['prompt_id'] == prompt['id'] and provenance['seed'] == 20260911
                 and provenance['method_id'] == 'ours' and provenance['strategy'] == a['strategy']
                 and provenance['budget'] == budget and provenance['accepted_target_chunks'] ==
                 provenance['proposed_target_chunks'] and provenance['timing'] ==
                 {k:x[k] for k in ('generation_wall_seconds', 'video_ready_wall_seconds', 'timing_scope')},
                 'native intervention or timing provenance differs')
        history, audit = ours_history(
            ours, prompt['id'], profile.chunk_count, provenance, burst_length=burst_length,
            observation_horizon=observation_horizon, opening_target_burst=opening_target_burst,
            split_target_burst=split_target_burst, split_burst_count=split_burst_count,
            final_window_at_tail=final_window_at_tail, motion_trigger=motion_trigger)
        # Deferred campaign scoring uses immutable pending-rollout manifests,
        # rather than the evaluator's optional direct-scoring alias.
        aliases = []
        for name in ours.frozen.entries:
            if name.startswith(ours.prefix+'measurements/rollouts/') and name.endswith('.pending.json'):
                item = read_json(ours, name[len(ours.prefix):])
                if item.get('prepared_quality', {}).get('history_head_sha256') == history.head_sha256:
                    aliases.append(item)
        _require(len(aliases) == 1, 'accepted-history video binding is missing or ambiguous')
        pending = PendingSequentialRollout.from_dict(aliases[0])
        prepared = pending.prepared_quality
        _require(pending.history == history and pending.corrected_chunks == tuple(provenance['accepted_target_chunks'])
                 and prepared.video_sha256 == qx[1]['sha256'] and prepared.video_nbytes == qx[1]['nbytes']
                 and (ours.run/ours.prefix/prepared.video_path).resolve() == qx[0],
                 'accepted history, scored video or evaluator alias differs')
        finalized = read_json(ours, 'measurements/rollouts/'+pending.route_key+'.json')
        _require(finalized['final_history_hash'] == history.head_sha256
                 and finalized['target_call_count'] == budget + 1
                 and finalized['budgeted_target_call_count'] == budget + 1 and
                 abs(finalized['full_video_quality']-math.fsum(x['quality_scores'].values())/6) < 1e-10,
                 'finalized rollout quality or accepted budget differs')
        decode_path = referenced_file(ours, dict(path=prepared.decode_evidence_path,
            sha256=prepared.decode_evidence_sha256), 'accepted decode')
        accepted_decode = _load_json(decode_path, label='accepted decode')
        _require(accepted_decode['history_manifest_sha256'] == history.manifest_sha256 and
                 accepted_decode['expected_frames'] == profile.frame_count and
                 accepted_decode['latent_chunk_count'] == profile.chunk_count and
                 accepted_decode['video'] == dict(path=prepared.video_path, sha256=prepared.video_sha256,
                     nbytes=prepared.video_nbytes, fps=24, frames=profile.frame_count),
                 'accepted decode history/video differs')
        native = y['native_generation']
        _require(native['target_model_calls'] == profile.chunk_count and native['draft_model_calls'] == 0
                 and native['target_session_count'] == 1 and native['history_reconstruction_calls'] == 0
                 and native['target_session_policy'] == 'persistent_native_session'
                 and native['chunk_count'] == profile.chunk_count and native['rgb_frame_count'] == profile.frame_count
                 and native['interleaved_decode'] is overlap and native['accept_reject_gate'] is False,
                 'reference is not persistent pure Target')
        native_file = read_json(target, 'provenance/target-native/'+prompt['id']+'.json')
        th = LatentHistory.from_dict(native_file['history'])
        _require({k:v for k,v in native_file.items() if k != 'history'} == native and
                 th.head_sha256 == native['history_head_sha256'] and th.next_chunk_index == profile.chunk_count
                 and th.manifest_sha256 == native['history_manifest_sha256']
                 and th.prompt_id == native['prompt_id'] == prompt['id']
                 and th.trajectory_seed == native['seed'] == 20260911
                 and len(native['timing']['per_chunk']) == profile.chunk_count
                 and all(s.producer_role == 'target' for s in th.segments) and
                 history.segments[0].latent_sha256 == th.segments[0].latent_sha256 == y['root_latent_sha256'],
                 'Target trajectory or shared root differs')
        for i, (s, chunk) in enumerate(zip(th.segments, native['timing']['per_chunk'])):
            _require(s.latent_sha256 == chunk['latent_sha256'] and s.chunk_index == chunk['chunk_index'] == i,
                     'persistent Target chunk evidence differs')
            referenced_file(target, dict(path=s.latent_path, sha256=s.latent_sha256,
                                        nbytes=s.latent_nbytes), 'Target latent')
            _require(s.initial_noise_descriptor_sha256 == history.segments[i].initial_noise_descriptor_sha256,
                     'Target latent or per-chunk initial noise differs')
        decode = _load_json(referenced_file(target, y['decode_evidence'], 'native decode'), label='native decode')
        _require(decode['history_manifest_sha256'] == th.manifest_sha256 and
                 decode['expected_frames'] == profile.frame_count and decode['latent_chunk_count'] == profile.chunk_count
                 and decode['video'] == dict(y['video'], fps=24, frames=profile.frame_count),
                 'Target decode does not bind its history/video')
        if overlap:
            verify_accepted_renderer(ours, provenance['accepted_stream'], history, accepted_decode,
                                     model=model, role='target', profile=profile)
            verify_accepted_renderer(target, y['accepted_stream'], th, decode,
                                     model=model, role='draft', profile=profile)
        pairs.append(dict(prompt_id=prompt['id'], ours_quality_100=math.fsum(x['quality_scores'].values())/6*100,
            target_quality_100=math.fsum(y['quality_scores'].values())/6*100,
            ours_ready_seconds=x['video_ready_wall_seconds'], target_ready_seconds=y['video_ready_wall_seconds'],
            ours_dimensions=x['quality_scores'], target_dimensions=y['quality_scores'],
            target_chunks=audit['target_chunks'], ours_video_sha256=qx[1]['sha256'],
            target_video_sha256=qy[1]['sha256']))
    mean = lambda name:math.fsum(x[name] for x in pairs)/24
    sources = [dict(run_id=s.marker['run_id'], attempt_id=s.marker['attempt_id'],
                    artifact_index_sha256=s.marker['artifact_index_sha256'],
                    destination=s.marker['artifact_publication']['destination']) for s in (ours,target)]
    return dict(schema='native_ready_matched_report_v2', status='verified', model_pair=pair,
        generation_code_revision=a['code_revision'],
        nominal_seconds=seconds, benchmark=expected_benchmark, prompt_count=24, seed=20260911,
        non_root_target_burst_length=burst_length, observation_horizon=observation_horizon,
        timing_scope=timing_scope, quality_definition='unweighted_six_dimension_mean_percent_not_official_total',
        ours_quality_mean_100=mean('ours_quality_100'), target_quality_mean_100=mean('target_quality_100'),
        ours_video_ready_mean_seconds=mean('ours_ready_seconds'),
        target_video_ready_mean_seconds=mean('target_ready_seconds'),
        video_ready_speedup_ratio_of_means=mean('target_ready_seconds')/mean('ours_ready_seconds'),
        physical_gpu_uuids=preflight['gpu_uuids'], actual_x264_options=common_codec, pairs=pairs, sources=sources,
        target_population_label=target_pop['model_pair'], verified_actual_population=population,
        native_cache_evidence_scope='sealed_runtime_checked_producer_witness_revalidated_for_schedule_and_native_prefix_counts',
        checkpoint_files_reverified=False, raw_rgb_parity_claim=False,
        transitively_hash_bound_local_files=[dict(run_id=s.marker['run_id'],
            files=list(getattr(s, 'transitive_references', {}).values())) for s in (ours,target)],
        caveats=['Development inputs, one seed; quality is six shared dimensions, not the official full total.',
                 ('Accepted stream receipts verify completed native rendering for both arms.' if overlap else
                  'Target legacy population label retained; actual texts, seed, native decoder, root and input source independently verified.'),
                 'Original policy excludes checkpoints/raw latents/decode sidecars from publication. Required local latent/decode blobs match hashes in sealed parent records; independent raw checkpoint/RGB parity is not claimed.',
                 'Warm-model/cached-text wall-time ratio; not a total GPU-time or throughput claim.'],
        **({'opening_target_burst': True} if opening_target_burst else {}),
        **({'split_burst_count': split_burst_count} if split_burst_count != 2 else {}),
        **({'final_window_at_tail': True} if final_window_at_tail else {}),
        **({'motion_trigger': True, 'motion_energy_floor': MOTION_ENERGY_FLOOR} if motion_trigger else {}),
        **({'split_target_burst': True, 'non_root_target_budget': budget} if split_target_burst else {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--seconds', type=int, choices=(15,30,60), required=True)
    parser.add_argument('--overlap-native-vae', action='store_true')
    parser.add_argument('--native-burst-length', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--native-burst-observation-horizon', type=int, choices=(2, 8), default=8)
    parser.add_argument('--native-opening-target-burst', action='store_true')
    parser.add_argument('--native-split-target-burst', action='store_true')
    parser.add_argument('--native-split-burst-count', type=int, choices=(2, 3, 4, 8), default=2)
    parser.add_argument('--native-final-window-at-tail', action='store_true')
    parser.add_argument('--native-motion-trigger', action='store_true')
    args = parser.parse_args()
    prefix = ('native-stream-ready-' if args.overlap_native_vae else 'native-ready-')+args.model+'-'
    timing_scope = OVERLAP_READY_SCOPE if args.overlap_native_vae else TIMING_SCOPE
    label = f'{args.seconds}s-k2-24p' if args.native_burst_length == 2 else f'{args.seconds}s-24p'
    if args.native_burst_observation_horizon == 2:
        label = f'{args.seconds}s-p2-24p'
    target_label = f'{args.seconds}s-24p'
    if args.native_opening_target_burst:
        label = target_label = f'{args.seconds}s-opening-24p'
    if args.native_split_target_burst:
        label = f'{args.seconds}s-split-24p'
        if args.native_burst_length == 3:
            label = target_label = f'{args.seconds}s-split-k6-b3-24p'
        elif args.native_split_burst_count == 3:
            label = f'{args.seconds}s-split-k6-24p'
            if args.native_final_window_at_tail:
                label = f'{args.seconds}s-split-k6-tail-24p'
            if args.native_motion_trigger:
                label = f'{args.seconds}s-split-k6-motion-24p'
        elif args.native_split_burst_count in (4, 8):
            label = f'{args.seconds}s-split-k8-b{args.native_burst_length}-24p'
    ours = source(args.queue, prefix+'ours-'+label, args.repo, args.commit, timing_scope=timing_scope)
    target = source(args.queue, prefix+'target-'+target_label, args.repo, args.commit, timing_scope=timing_scope)
    print(json.dumps(compare(ours, target, repo=args.repo, model=args.model, seconds=args.seconds,
                             overlap=args.overlap_native_vae,
                             burst_length=args.native_burst_length,
                             observation_horizon=args.native_burst_observation_horizon,
                             opening_target_burst=args.native_opening_target_burst,
                             split_target_burst=args.native_split_target_burst,
                             split_burst_count=args.native_split_burst_count,
                             final_window_at_tail=args.native_final_window_at_tail,
                             motion_trigger=args.native_motion_trigger), sort_keys=True))


if __name__ == '__main__':
    main()
