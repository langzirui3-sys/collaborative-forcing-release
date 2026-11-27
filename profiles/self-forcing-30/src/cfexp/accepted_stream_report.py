"""Fail-closed comparison of sealed accepted-stream P8 and genuine pure Target.

Reports exact paired ready latency and all six quality dimensions. A stage
speedup, incomplete population, unmatched codec, stale native state or an old
offline timing scope cannot satisfy this comparison. No winner is selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .accepted_stream import TIMING_SCOPE
from .baseline_population import load_baseline_population
from .correction_state import LatentHistory
from .duration_sweep import duration_profile
from .main_table_merge import _load_frozen_index
from .main_table_baseline_campaign import _run_dir, _write_new_json
from .main_table_v1 import QUALITY_DIMENSIONS
from .metadata import find_git_root
from .krea_chunk_backend import validate_native_correction_state
from . import sf_target_two_step
from .erank_acceptance_pilot import (P8_TRANSFER_ARM, LONG_TARGET_OPENING_ARMS, SHORT_TARGET_OPENING_ARM,
                                    SHORT_POSITIVE_OPENING_ARM, WIDE_TARGET_BURST_ARM,
                                    SHORT_REDUCED_BUDGET_ARM, SHORT_REDUCED_OPENING_ARM, SHORT_TARGET_OPENING_ARMS,
                                    burst_length_for_arm, observation_horizon_for_arm, split_transfer_options)
from .rank_burst import validate_burst_schedule
from .streaming_vae_probe import VAE_SHA, WEIGHTS_SHA
from .video_encoding_policy import SERIAL, verify_video
from .video_only_decode import validate_pixels


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_stream(row, *, target, chunk_count=30, candidate_arm=P8_TRANSFER_ARM):
    frame_count = 12 * chunk_count - 3
    stream = row['accepted_stream']
    require(stream['latent_chunk_count'] == chunk_count and stream['decoder_and_encoder_joined'] is True,
            'stream has not drained every accepted chunk')
    require(stream['decoder_worker_role'] == ('draft' if target else 'target')
            and type(stream['decoder_pid']) is int and stream['decoder_pid'] > 0,
            'renderer was not owned by the intended existing worker')
    decoder = stream['decoder']
    require(decoder.get('native_vae_sha256') == VAE_SHA
            and decoder.get('checkpoint_sha256') == WEIGHTS_SHA
            and decoder.get('separate_causal_decoder_state') is True,
            'renderer differs from the frozen separate native SF decoder')
    validate_pixels(stream['pixels'], frame_count)
    video = stream['video']
    require(video['encoding_policy'] == SERIAL and video['fps'] == 24 and video['frames'] == frame_count
            and video['input_rgb_tchw_sha256'] == stream['pixels']['raw_sha256']
            and (video['sha256'],video['nbytes']) == (row['video']['sha256'],row['video']['nbytes']),
            'rendered frames or MP4 do not bind the scored video')
    chunks = stream['chunks']
    require(len(chunks) == chunk_count, 'incomplete streamed chunk evidence')
    for i, chunk in enumerate(chunks):
        require(chunk['chunk_index'] == i and chunk['frame_count'] == (9 if i == 0 else 12)
                and chunk['frame_start'] == (0 if i == 0 else i*12-3),
                'streamed frames are missing, duplicated or out of order')
    if target:
        native = row['native_generation']
        require(native['target_model_calls'] == chunk_count and native['draft_model_calls'] == 0
                and native['chunk_count'] == chunk_count and native['rgb_frame_count'] == frame_count
                and native['target_session_policy'] == 'persistent_native_session'
                and native['target_session_count'] == 1 and native['history_reconstruction_calls'] == 0
                and native['interleaved_decode'] is True,
                'reference is not genuine persistent full-Target generation')
        require(native['history_head_sha256'] == stream['history_head_sha256']
                and [x['latent_sha256'] for x in native['timing']['per_chunk']]
                    == [x['latent_sha256'] for x in chunks],
                'Target rendering is bound to another trajectory')
        require(row['root_latent_sha256'] == chunks[0]['latent_sha256'], 'Target root differs')
    else:
        seconds = chunk_count // 2
        split_options = split_transfer_options(candidate_arm, seconds)
        opening = split_options.get('opening_target_burst', False)
        require(candidate_arm == P8_TRANSFER_ARM or split_options, 'wrong streamed burst arm for horizon')
        budget = int(candidate_arm.rsplit('_k', 1)[1])
        burst_length = burst_length_for_arm(candidate_arm)
        draft_count = chunk_count - 1 - budget
        trace = row['trajectory']
        validate_burst_schedule(trace, budget=budget, burst_length=burst_length, chunk_count=chunk_count,
                                observation_horizon=observation_horizon_for_arm(candidate_arm),
                                **split_options)
        reuse = row['native_cache_audit']['burst_reuse']
        require(trace['counters']['physical_calls'] == {'draft': draft_count, 'target': budget + 1}
                and row['proposal_budget'] == budget
                and row['native_cache_audit']['draft_reused_chunks'] > 0
                and reuse['suffix_replay_bound_verified'] is True
                and reuse['maximum_replayed_chunks'] <= burst_length + int(opening)
                and (not opening or reuse.get('initial_replayed_chunks') == burst_length + 1)
                and reuse['draft_calls'] == draft_count
                and reuse['burst_length'] == burst_length,
                'P8 budget or native incremental KV reuse differs')
        states=row['native_cache_audit']['target_native_states']
        require({int(k) for k in states} == {0,*trace['accepted_chunks']},
                'P8 omitted native Target correction cache evidence')
        for index,state in states.items():
            validate_native_correction_state(state,chunk_index=int(index))
        require(len({s['native_session_id'] for s in states.values()}) == 1
                and all(not s['session_recreated'] for i,s in states.items() if int(i)),
                'Target correction session was recreated')
        require(trace['final_history_head_sha256'] == stream['history_head_sha256']
                and [x['latent_sha256'] for x in trace['committed']]
                    == [x['latent_sha256'] for x in chunks],
                'P8 rendering is bound to an unaccepted branch')


def compare(candidate, target, population, *, seconds=None, population_binding=None):
    """Pure comparison; never mutates/relabels sealed source evidence."""
    profile = duration_profile(seconds) if seconds is not None else None
    count = 24 if profile else 12
    chunk_count = profile.chunk_count if profile else 30
    require(len(population) == len({p['id'] for p in population}) == count,
            'expected the complete frozen population')
    require(candidate['code_revision'] == target['code_revision']
            and candidate['hardware'] == target['hardware']
            and not candidate['hardware'].get('unverified_test_fixture')
            and len(candidate['hardware']['gpus']) == 2,
            'execution code or exact two-GPU hardware is not matched')
    require(candidate['timing_scope'] == target['timing_scope'] == TIMING_SCOPE,
            'offline and streaming ready timing scopes cannot be mixed')
    require(('sealed_native_rope_audit' in candidate)==('sealed_native_rope_audit' in target),
            'native RoPE execution-gate scope differs')
    require(('sealed_native_full_attention_audit' in candidate)==('sealed_native_full_attention_audit' in target),
            'full attention execution-gate scope differs')
    require(('sealed_native_index_constants_audit' in candidate) ==
            ('sealed_native_index_constants_audit' in target),
            'native witness index execution-gate scope differs')
    arms = candidate['arms']
    arm = arms[0] if isinstance(arms, list) and len(arms) == 1 else None
    split_options = split_transfer_options(arm, seconds)
    require((arm == P8_TRANSFER_ARM or split_options)
            and candidate['cohort'] == ('main24_burst' if profile else 'main12_burst_15s')
            and target['schema'] == 'accepted_stream_target_reference_v1',
            'wrong method or reference schema')
    if profile:
        benchmark = 'VBench-Long' if seconds == 60 else 'VBench'
        candidate_duration = dict(profile.metadata(),
            experiment_id=f'committed-rank-burst-main24-{seconds}s-v1',
            calibration_policy='explicit_frozen_C10_control',
            ours_proposal_budget={arm: int(arm.rsplit('_k', 1)[1])},
            budget_rule='explicit_per_arm_budget_after_target_root')
        candidate_duration.update(split_options)
        require(candidate['duration'] == candidate_duration and target['duration'] == profile.metadata()
                and population_binding is not None
                and all(x['population_binding'] == population_binding
                        and x['population'] == 'main24_method_development_not_held_out'
                        and x['benchmark'] == benchmark and x['row_count'] == count
                        for x in (candidate, target)),
                'transfer population, duration, or benchmark differs')
    indexed = []
    for data in (candidate,target):
        rows = {r['prompt_id']:r for r in data['rows']}
        require(len(rows) == len(data['rows']) == count and set(rows) == {p['id'] for p in population},
                'partial, duplicate or selected prompt coverage')
        indexed.append(rows)
    paired = []
    for prompt in population:
        a,b = (rows[prompt['id']] for rows in indexed)
        for row, is_target in ((a,False),(b,True)):
            require(row['prompt'] == prompt['text'] and row['seed'] == 20260911,
                    'prompt text or generation seed differs')
            require(row['timing_scope'] == TIMING_SCOPE, 'row timing scope differs')
            gen, ready = row['generation_wall_seconds'],row['video_ready_wall_seconds']
            require(all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in (gen,ready))
                    and ready >= gen, 'invalid ready timing or final drain excluded')
            scores = row['quality_scores']
            require(set(scores) == set(QUALITY_DIMENSIONS)
                    and all(type(x) in (int,float) and math.isfinite(x) and 0 <= x <= 1 for x in scores.values()),
                    'quality does not contain all six finite dimensions')
            validate_stream(row, target=is_target, chunk_count=chunk_count, candidate_arm=arm)
        require(a['accepted_stream']['chunks'][0]['latent_sha256'] == b['root_latent_sha256'],
                'shared Target root differs between compared inputs')
        paired.append({'prompt_id':prompt['id'],
            'candidate_ready_seconds':a['video_ready_wall_seconds'],
            'target_ready_seconds':b['video_ready_wall_seconds'],
            'ready_speedup':b['video_ready_wall_seconds']/a['video_ready_wall_seconds'],
            'candidate_quality_100':math.fsum(a['quality_scores'].values())/6*100,
            'target_quality_100':math.fsum(b['quality_scores'].values())/6*100})
    ctime = math.fsum(r['candidate_ready_seconds'] for r in paired) / count
    ttime = math.fsum(r['target_ready_seconds'] for r in paired) / count
    return {'schema':'accepted_stream_matched_comparison_v1',
        'population':('main24_method_development_not_held_out' if profile
                      else 'frozen_first12_method_development_not_held_out'),
        **({'nominal_seconds': seconds, 'prompt_count': count} if profile else {}),
        'quality_definition':'six_dimension_equal_weight_mean_not_official_full_VBench_total',
        'timing_scope':TIMING_SCOPE, 'encoding_policy':SERIAL,
        **{name: candidate[name] for name in ('target_nonroot_denoising_schedule',
            'target_burst_denoising_schedules', 'target_early_native_bursts') if name in candidate},
        'quality_macro_mean_100':math.fsum(r['candidate_quality_100'] for r in paired)/count,
        'target_quality_macro_mean_100':math.fsum(r['target_quality_100'] for r in paired)/count,
        'video_ready_mean_seconds':ctime, 'target_video_ready_mean_seconds':ttime,
        'video_ready_speedup_ratio_of_means':ttime/ctime, 'speed_at_least_2x':ttime/ctime >= 2,
        'quality_preservation_requires_previous_reference':True,
        'automatic_goal_completion':False, 'publication_ready_method_claim':False,
        'native_rope_execution_audits':{'candidate':candidate.get('sealed_native_rope_audit'),
                                       'target':target.get('sealed_native_rope_audit')},
        'native_full_attention_execution_audits':{'candidate':candidate.get('sealed_native_full_attention_audit'),
                                                'target':target.get('sealed_native_full_attention_audit')},
        'dimensions':{d:{'candidate_mean_100':math.fsum(indexed[0][p['id']]['quality_scores'][d] for p in population)/count*100,
                         'target_mean_100':math.fsum(indexed[1][p['id']]['quality_scores'][d] for p in population)/count*100}
                      for d in QUALITY_DIMENSIONS}, 'paired':paired}


def transfer_noise_bindings(frozen, prefix, rows, *, target, chunk_count):
    """Bind every initial noise descriptor to the actual rendered history."""
    if target:
        paths = [prefix + f"provenance/target-native/{row['prompt_id']}.json" for row in rows]
    else:
        suffix = f'/chunk-{chunk_count - 1:02d}.json'
        paths = [name for name in frozen.entries if name.startswith(prefix)
                 and '/state/rollouts/' in name and name.endswith(suffix)]
    indexed = {row['prompt_id']: row for row in rows}
    require(len(paths) == len(indexed) == len(rows), 'transfer history coverage differs')
    bindings = {}
    for path in paths:
        record = json.loads(frozen.file(path, label='final transfer history').read_text())
        history = LatentHistory.from_dict(record['history'])
        pid = history.prompt_id
        require(pid in indexed and pid not in bindings, 'duplicate or unknown transfer history')
        row = indexed[pid]
        stream = row['accepted_stream']
        selected = set(range(chunk_count)) if target else {0, *row['trajectory']['accepted_chunks']}
        require(history.trajectory_seed == row['seed'] and history.next_chunk_index == chunk_count
                and history.manifest_sha256 == stream['history_manifest_sha256']
                and history.head_sha256 == stream['history_head_sha256']
                and [s.segment_sha256 for s in history.segments]
                    == [x['segment_sha256'] for x in stream['chunks']]
                and {s.chunk_index for s in history.segments if s.producer_role == 'target'} == selected,
                'transfer history does not bind the rendered stream and selected producers')
        bindings[pid] = tuple(segment.initial_noise_descriptor_sha256 for segment in history.segments)
    return bindings


def audit_reduced_target_steps(frozen, prefix, rows, *, arm, burst_schedules, early_native_bursts):
    """Bind every measured Target forward to its accepted chunk and burst position."""
    accepted = {}
    for row in rows:
        chunks = row['accepted_stream']['chunks']
        selected = row['trajectory']['accepted_chunks']
        expected_schedules = {}
        if burst_schedules is not None:
            width = len(burst_schedules)
            budget = int(arm.rsplit('_k', 1)[1])
            require(width == burst_length_for_arm(arm) and len(selected) == budget,
                    'Target burst width or budget differs from the invocation')
            for start in range(0, len(selected), width):
                burst = selected[start:start+width]
                require(burst == list(range(burst[0], burst[0]+width))
                        and (start == 0 or burst[0] > selected[start-1]+1),
                        'Target burst is interrupted or overlaps another burst')
                schedules = sf_target_two_step.schedules_for_burst(
                    burst_schedules, burst_index=start // width, early_native_bursts=early_native_bursts,
                )
                expected_schedules.update(zip(burst, schedules, strict=True))
        for index in (0, *selected):
            key = (row['prompt_id'], index)
            require(key not in accepted, 'duplicate accepted Target chunk')
            accepted[key] = chunks[index], expected_schedules.get(index, sf_target_two_step.SCHEDULE)
    diagnostic_prefix = prefix + f'cells/{arm}/provenance/target-rpc-diagnostics/'
    for name in frozen.entries:
        if not name.startswith(diagnostic_prefix) or not name.endswith('.json'):
            continue
        record = json.loads(frozen.file(name, label='two-step Target diagnostic').read_text())
        index = record['chunk_index']
        binding = accepted.pop((record['prompt_id'], index), None)
        require(binding is not None, 'unexpected Target diagnostic')
        chunk, expected_schedule = binding
        require(all(record[k] == chunk[k] for k in ('latent_sha256', 'segment_sha256')),
                'two-step Target diagnostic does not bind an accepted chunk')
        backend = record['backend_diagnostics']
        trace = sf_target_two_step.validate_correction_trace(
            backend['forward_trace'], backend['denoising_schedule'],
            chunk_index=index, expected_nonroot_schedule=expected_schedule,
        )
        require(trace == backend['denoising_forward_trace'],
                'recorded two-step Target forwards differ from the validated trace')
    require(not accepted, 'missing two-step Target diagnostics')


def load_run(root, *, target, seconds=None, repo=None, manifest=None):
    frozen,marker = _load_frozen_index(Path(root))
    frozen.verify_all()
    prefix = 'attempts/'+marker['attempt_id']+'/'
    data = json.loads(frozen.file(prefix+'result.json',label='result').read_text())
    require(data['schema'] == ('accepted_stream_target_reference_v1' if target else 'erank_acceptance_pilot_v1'),
            'unexpected streamed source result schema')
    require(data['code_revision'] == marker['git_sha'] and data['status'] == 'succeeded',
            'result is not bound to its sealed successful source')
    resolved = json.loads(frozen.file('resolved-manifest.json',label='manifest').read_text())
    chunk_count = 30
    require(manifest is None or seconds is not None, 'custom manifest requires a formal duration')
    if seconds is not None:
        from .baseline_shard_report import source_manifest
        profile = duration_profile(seconds)
        chunk_count = profile.chunk_count
        role = 'target' if target else 'p8'
        manifest = manifest or f'experiments/sf-stream-transfer-{role}-{seconds}s-24p-ntu-v1.manifest.json'
        require(all(resolved.get(k) == marker[k] for k in ('run_id', 'attempt_id', 'spec_hash'))
                and resolved['manifest'] == source_manifest(repo, marker['git_sha'], manifest),
                'sealed transfer invocation differs from its published source')
        terminal = json.loads(frozen.file('run.json', label='terminal run').read_text())
        require(terminal['execution_status'] == 'SUCCEEDED' and terminal['exit_code'] == 0
                and terminal['run_id'] == marker['run_id']
                and terminal['latest_attempt_id'] == marker['attempt_id'],
                'transfer run terminal identity differs')
    command = resolved['manifest']['spec']['command']
    env = command['env']
    budget = 4
    if not target and data['arms'] != [P8_TRANSFER_ARM]:
        arm, = data['arms']
        require(split_transfer_options(arm, seconds) and arm in command['argv'],
                'split transfer arm differs from the invocation')
        budget = int(arm.rsplit('_k', 1)[1])
    target_schedule = sf_target_two_step.schedule_metadata(env)
    schedule_keys = ('target_nonroot_denoising_schedule', 'target_burst_denoising_schedules',
                     'target_early_native_bursts')
    parameters = resolved['manifest']['spec'].get('parameters', {})
    require(all(data.get(key) == target_schedule.get(key) == parameters.get(key)
                for key in schedule_keys), 'Target schedule evidence differs from the frozen invocation')
    if target_schedule:
        burst_schedules = sf_target_two_step.burst_schedules(env)
        schedule_width = len(burst_schedules) if burst_schedules is not None else 4
        long_opening = sf_target_two_step.long_opening(env)
        short_schedule = (seconds == 30 and not long_opening
                          and (arm, budget) in ((sf_target_two_step.ARM, 36),
                                                (SHORT_TARGET_OPENING_ARM, 36),
                                                (SHORT_POSITIVE_OPENING_ARM, 36),
                                                (WIDE_TARGET_BURST_ARM, 40),
                                                (SHORT_REDUCED_BUDGET_ARM, 28),
                                                (SHORT_REDUCED_OPENING_ARM, 28))
                          and (not target_schedule.get('target_early_native_bursts')
                               or arm == sf_target_two_step.ARM
                               or (arm in SHORT_TARGET_OPENING_ARMS
                                   and target_schedule["target_early_native_bursts"] == 2)))
        long_schedule = seconds == 60 and long_opening and arm in LONG_TARGET_OPENING_ARMS
        require(not target and (short_schedule or long_schedule)
                and schedule_width == burst_length_for_arm(arm),
                'Target step allocation differs from the frozen invocation')
        audit_reduced_target_steps(
            frozen, prefix, data['rows'], arm=arm, burst_schedules=burst_schedules,
            early_native_bursts=target_schedule.get('target_early_native_bursts', 0),
        )
    deferred = '--defer-video-delivery' in command['argv']
    schedule = 'after_generation' if deferred else 'interleaved'
    require(not (target and deferred) and data.get('accepted_video_schedule', 'interleaved') == schedule,
            'accepted video delivery schedule differs from the invocation')
    require(env.get('CFEXP_ACCEPTED_VIDEO_STREAM') == env.get('CFEXP_DRAFT_KV_REUSE') == '1'
            and env.get('CFEXP_VIDEO_ENCODING') == SERIAL,
            'resolved native reuse/stream/codec flags differ')
    def bound(binding, label):
        relative=prefix+binding['path'];path=frozen.file(relative,label=label)
        size,sha=frozen.entries[relative]
        require((size,sha) == (binding['nbytes'],binding['sha256']),label+' binding differs')
        return path
    common_codec = None
    for row in data['rows']:
        require(row.get('accepted_video_schedule', 'interleaved') == schedule,
                'accepted video delivery row differs from the invocation')
        video=bound(row['video'],'video');verify_video(video,SERIAL)
        decode=json.loads(bound(row['decode_evidence'],'decode evidence').read_text())
        require(decode['pixels'] == row['accepted_stream']['pixels']
                and decode['history_head_sha256'] == row['accepted_stream']['history_head_sha256'],
                'prepared pixels or accepted history differs')
        if seconds is not None:
            from .native_ready_report import quality
            src = SimpleNamespace(frozen=frozen, prefix=prefix, run=frozen.run_dir)
            mid = 'stream-target-' + row['prompt_id'] if target else row['measurement_id']
            scored, _, codec = quality(src, row, row['quality_evidence'],
                                       {'text': row['prompt']}, mid, seconds)
            require(scored == video and (common_codec is None or common_codec == codec),
                    'transfer scoring input or actual codec differs')
            common_codec = codec
        else:
            quality_record = json.loads(bound(row['quality_evidence'], 'quality evidence').read_text())
            require(quality_record['schema'] == 'vbench_short_measurement_v1'
                    and quality_record['aggregation'] == 'six_dimension_arithmetic_mean'
                    and quality_record['dimension_scores'] == row['quality_scores'],
                    'scored dimensions differ from bound official measurement')
            config = json.loads(bound(quality_record['bindings']['config'], 'quality config').read_text())
            inputs = [x for x in config['inputs'] if x['id'] == quality_record['measurement_id']]
            require(len(inputs) == 1 and inputs[0]['prompt'] == row['prompt']
                    and (inputs[0]['sha256'], inputs[0]['nbytes']) == (row['video']['sha256'], row['video']['nbytes']),
                    'quality scorer used another video or prompt')
    if seconds is not None:
        data['audited_codec_options'] = common_codec
        data['audited_noise_bindings'] = transfer_noise_bindings(
            frozen, prefix, data['rows'], target=target, chunk_count=chunk_count)
    if env.get('CFEXP_NATIVE_ROPE_COEFFICIENT_REUSE')=='1':
        from .rope_worker import audit_sealed_execution
        data['sealed_native_rope_audit']=audit_sealed_execution(frozen,marker,data,target=target,chunk_count=chunk_count,budget=budget)
    if env.get('CFEXP_NATIVE_FULL_LENGTH_ATTENTION')=='1':
        from .full_attention_worker import audit_sealed_execution
        data['sealed_native_full_attention_audit']=audit_sealed_execution(frozen,marker,data,target=target,chunk_count=chunk_count,budget=budget)
    if env.get('CFEXP_ACCEPTED_TARGET_PREFILL')=='1':
        from .accepted_prefill import burst_length as prefill_burst_length
        from .accepted_prefill_audit import audit_sealed_execution
        width = prefill_burst_length(env)
        require(width == (4 if target else burst_length_for_arm(data['arms'][0])),
                'accepted prefill width differs from the invocation')
        data['sealed_accepted_target_prefill_audit'] = audit_sealed_execution(
            frozen, marker, data, target=target, chunk_count=chunk_count, budget=budget,
            burst_length=width, opening_target_burst=(
                not target and split_transfer_options(data['arms'][0], seconds).get('opening_target_burst', False)
            ))
    if env.get('CFEXP_KV_WITNESS_INDEX_REUSE') == '1':
        require(env.get('CFEXP_RANK_INDEX_REUSE') == ('0' if target else '1'),
                'native rank index execution scope differs')
        from .native_index_audit import audit_sealed_execution
        data['sealed_native_index_constants_audit'] = audit_sealed_execution(
            frozen, marker, data, target=target, chunk_count=chunk_count, budget=budget)
    return data, marker


def previous_quality(candidate, previous, previous_pixels):
    """Compare exact generated content separately from lossy MP4 metric noise."""
    require(previous.get('arms') == ['rank_burst4_p8_k4']
            and previous.get('cohort') == 'main12_burst_15s', 'wrong previous P8 method')
    old={r['prompt_id']:r for r in previous['rows']}
    require(len(old) == len(previous['rows']) == 12
            and set(old) == {r['prompt_id'] for r in candidate['rows']},
            'previous reference does not cover the same complete first12 population')
    rows=[]
    for current in candidate['rows']:
        pid=current['prompt_id'];prior=old[pid]
        require(current['prompt'] == prior['prompt'] and current['seed'] == prior['seed'],
                'previous prompt or seed differs')
        scores=prior['quality_scores']
        require(set(scores) == set(QUALITY_DIMENSIONS)
                and all(type(x) in (int,float) and math.isfinite(x) and 0 <= x <= 1 for x in scores.values()),
                'previous quality dimensions incomplete')
        pixels=previous_pixels[pid];validate_pixels(pixels,357)
        rows.append({'prompt_id':pid,
            'accepted_latents_bitwise_equal': [x['latent_sha256'] for x in current['trajectory']['committed']]
                == [x['latent_sha256'] for x in prior['trajectory']['committed']],
            'generated_pixels_bitwise_equal':current['accepted_stream']['pixels'] == pixels,
            'target_positions_equal':current['trajectory']['accepted_chunks'] == prior['trajectory']['accepted_chunks'],
            'rank_observations_equal':[d['rank'] for d in current['trajectory']['decisions']]
                == [d['rank'] for d in prior['trajectory']['decisions']],
            'previous_quality_100':math.fsum(scores.values())/6*100,
            'current_quality_100':math.fsum(current['quality_scores'].values())/6*100})
    return {'previous_quality_macro_mean_100':math.fsum(r['previous_quality_100'] for r in rows)/12,
        'current_quality_macro_mean_100':math.fsum(r['current_quality_100'] for r in rows)/12,
        'accepted_latents_bitwise_preserved':all(r['accepted_latents_bitwise_equal'] for r in rows),
        'generated_pixels_bitwise_preserved':all(r['generated_pixels_bitwise_equal'] for r in rows),
        'target_positions_preserved':all(r['target_positions_equal'] for r in rows),
        'rank_observations_preserved':all(r['rank_observations_equal'] for r in rows),
        'lossy_encoding_scores_are_reported_separately':True,'paired':rows}


def previous_draft_witnesses(candidate_root,previous_root):
    """Compare every real Draft witness and forward trace, outside ready time."""
    collections=[]
    for root in (candidate_root,previous_root):
        frozen,marker=_load_frozen_index(Path(root));frozen.verify_all()
        prefix='attempts/'+marker['attempt_id']+'/';rows={}
        for name in frozen.entries:
            if not name.startswith(prefix) or '/state/rollouts/' not in name or not name.endswith('.json'):
                continue
            from .insight2_run import _load_audited_checkpoint
            checkpoint,history,parsed_context,segment,_=_load_audited_checkpoint(
                frozen.file(name,label='Draft witness checkpoint'),label='Draft witness checkpoint')
            generated=checkpoint['generated'];context=generated['context']
            if context['role']!='draft':continue
            key=(parsed_context.noise.prompt_id,parsed_context.next_chunk_index);diagnostics=generated['diagnostics']
            descriptor=diagnostics['kv_witness'];sha=diagnostics['kv_witness_sha256']
            require(hashlib.sha256(json.dumps(descriptor,ensure_ascii=False,allow_nan=False,
                sort_keys=True,separators=(',',':')).encode()).hexdigest()==sha
                and context['kv']['cache_sha256']==sha,'Draft witness descriptor/digest binding differs')
            require(key not in rows,'duplicate real Draft witness')
            trace=diagnostics['forward_trace']
            resolved=json.loads(frozen.file('resolved-manifest.json',label='manifest').read_text())
            if resolved['manifest']['spec']['command']['env'].get('CFEXP_ACCEPTED_TARGET_PREFILL')=='1':
                from .accepted_prefill_audit import canonical_trace
                cell=name.split('/state/rollouts/',1)[0]
                result=json.loads(frozen.file(prefix+'result.json',label='result').read_text())
                prompt=next(r['prompt'] for r in result['rows'] if r['prompt_id']==key[0])
                def read(binding):
                    relative=cell+'/'+binding['path']
                    require(frozen.entries[relative]==(binding['nbytes'],binding['sha256']),
                            'prefill parity receipt differs from the index')
                    return json.loads(frozen.file(relative,label='prefill parity receipt').read_text())
                trace,_=canonical_trace(diagnostics,history=history.prefix_before(parsed_context.next_chunk_index),
                                        prompt=prompt,read_receipt=read)
            rows[key]=(sha,descriptor,trace)
        prompts={p for p,_ in rows}
        require(len(rows)==300 and len(prompts)==12
            and all(sum(p==q for p,_ in rows)==25 for q in prompts),'incomplete first12 Draft witness coverage')
        collections.append(rows)
    current,prior=collections
    require(current.keys()==prior.keys(),'Draft witness positions or prompt population differ')
    differences=[{'prompt_id':p,'chunk_index':i,'witness_equal':current[p,i][:2]==prior[p,i][:2],
                  'forward_trace_equal':current[p,i][2]==prior[p,i][2]}
                 for p,i in sorted(current) if current[p,i]!=prior[p,i]]
    return {'schema':'real_first12_draft_witness_parity_v1','draft_calls_compared':300,
        'descriptor_and_digest_equal':all(x['witness_equal'] for x in differences),
        'forward_traces_equal':all(x['forward_trace_equal'] for x in differences),'differences':differences}


def main():
    from .stage_timing import _unsealed_directory
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run',type=Path,required=True)
    parser.add_argument('--target-run',type=Path,required=True)
    parser.add_argument('--previous-candidate-run',type=Path)
    parser.add_argument('--candidate-manifest', help='repository-relative manifest for an alternate frozen transfer arm')
    parser.add_argument('--duration-seconds', type=int, choices=(15, 30, 60),
                        help='verify the complete 24-prompt transfer cohort')
    args=parser.parse_args()
    output_root=_unsealed_directory(_run_dir())
    repo = find_git_root(Path.cwd())
    population, binding = load_baseline_population(repo)
    candidate, cm = load_run(args.candidate_run, target=False, seconds=args.duration_seconds, repo=repo,
                             manifest=args.candidate_manifest)
    target, tm = load_run(args.target_run, target=True, seconds=args.duration_seconds, repo=repo)
    if args.duration_seconds is None:
        report = compare(candidate, target, population[:12])
    else:
        require(args.previous_candidate_run is None, 'prior execution parity is only defined for first12')
        require(candidate['audited_codec_options'] is not None
                and candidate['audited_codec_options'] == target['audited_codec_options'],
                'transfer MP4 codec options differ between the two arms')
        require(candidate['audited_noise_bindings'] == target['audited_noise_bindings'],
                'transfer initial noise descriptors differ between the two arms')
        report = compare(candidate, target, population, seconds=args.duration_seconds,
                         population_binding=binding)
        report['codec_options'] = candidate['audited_codec_options']
        report['initial_noise_descriptors_matched'] = True
    report['sources']={'candidate':cm,'target':tm}
    if args.previous_candidate_run is not None:
        from .erank_acceptance_report import load_run as load_previous
        prior,pm=load_previous(args.previous_candidate_run,target=False)
        frozen,_=_load_frozen_index(args.previous_candidate_run);frozen.verify_all()
        prefix='attempts/'+pm['attempt_id']+'/'
        pixels={r['prompt_id']:json.loads(frozen.file(prefix+r['decode_evidence']['path'],
            label='previous native pixels').read_text())['pixels'] for r in prior['rows']}
        report['quality_preservation']=previous_quality(candidate,prior,pixels)
        report['sources']['previous_candidate']=pm
    _write_new_json(output_root/'result.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
