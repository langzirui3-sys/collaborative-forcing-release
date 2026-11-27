"""Compare sealed acceptance and native Target runs on identical GPU pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .baseline_population import load_baseline_population
from .erank_acceptance_pilot import PREFIX_ARMS, EARLY_ARMS, BURST_ARMS, BLOCK_ARMS, RANDOM_P8_B4_ARMS, DIRECT_ARMS, MAIN24_ARMS, READY_TIMING_SCOPE, block_positions, burst_length_for_arm, observation_horizon_for_arm
from .main_table_baseline_campaign import _run_dir, _write_new_json
from .main_table_merge import _load_frozen_index
from .main_table_v1 import QUALITY_DIMENSIONS
from .metadata import find_git_root


def population_encoding(data: dict) -> str:
    from .video_encoding_policy import LEGACY, validate_policy
    values = {validate_policy(row.get('encoding_policy', LEGACY)) for row in data['rows']}
    if len(values) != 1:
        raise ValueError('encoding policy must be uniform across the complete population')
    return values.pop()


def compare(candidate: dict, target: dict, prompt_ids: tuple[str, ...]) -> dict:
    encoding = population_encoding(candidate)
    if encoding != population_encoding(target):
        raise ValueError('candidate and Target encoding policies differ')
    if (candidate['code_revision'] != target['code_revision']
            or candidate['timing_scope'] != READY_TIMING_SCOPE
            or target['timing_scope'] != READY_TIMING_SCOPE
            or candidate['hardware'] != target['hardware']
            or candidate['hardware'].get('unverified_test_fixture')):
        raise ValueError('code, timing scope or exact GPU hardware is not matched')
    arm, = candidate['arms']
    prefix = arm in PREFIX_ARMS and candidate.get('cohort') == 'main12_prefix_15s'
    early = arm in EARLY_ARMS and candidate.get('cohort') == 'main12_early_15s'
    burst = arm in BURST_ARMS and candidate.get('cohort') == 'main12_burst_15s'
    direct = arm in DIRECT_ARMS and candidate.get('cohort') == 'main12_direct_15s'
    block = arm in (*BLOCK_ARMS, *RANDOM_P8_B4_ARMS) and candidate.get('cohort') == 'main12_block_15s'
    if not (prefix or early or direct or block or burst) and (arm not in MAIN24_ARMS or candidate.get('cohort') != 'main24_15s'):
        raise ValueError('wrong acceptance cohort or arm')
    if (prefix or early or direct or block or burst) and len(prompt_ids) != 12:
        raise ValueError('direct diagnostic requires the fixed twelve prompts')
    paired = []
    indexed = []
    for data in (candidate, target):
        rows = {r['prompt_id']:r for r in data['rows']}
        if len(rows) != len(data['rows']) or set(rows) != set(prompt_ids):
                raise ValueError('comparison does not cover exactly the requested frozen prompts')
        indexed.append(rows)
    for pid in prompt_ids:
        a, b = (d[pid] for d in indexed)
        if prefix or early or direct or block or burst:
            budget = int(arm.rsplit('_k',1)[1])
            trajectory = a['trajectory']
            if (a.get('proposal_budget') != budget
                    or len(trajectory['proposed_chunks']) != budget
                    or trajectory['accepted_chunks'] != trajectory['proposed_chunks']
                    or trajectory['counters']['physical_calls'] != ({'draft':29,'target':1,'target_first_update':budget} if prefix else {'draft':29-budget if block or burst else 29, 'target':budget+1})
                    or a['native_cache_audit']['draft_reused_chunks'] <= 0):
                raise ValueError('direct replacement budget, call count or native KV evidence differs')
            if prefix:
                expected={'schema':'target_prefix_route_audit_v1','probe_count':29,'prefix_count':budget,
                    'full_target_chunks':1,'history_replay_after_root':0,'branch_snapshot_bytes':0,
                    'diagnostic_oracle_forwards':0,'validated':True}
                c=trajectory['counters']
                if (not candidate.get('prefix_gate') or trajectory.get('prefix_execution_audit')!=expected
                        or a['native_cache_audit'].get('prefix_execution')!=expected
                        or c.get('target_denoising_forwards')!=4+budget
                        or c.get('draft_three_step_suffixes')!=budget or c.get('replayed_chunks')!=1):
                    raise ValueError('prefix computation or gate evidence is incomplete')
            if early:
                expected = {'schema':'early_route_audit_v1', 'probe_count':29,
                    'stopped_probe_count':budget, 'completed_draft_count':29-budget,
                    'current_slot_clean_updates':budget, 'history_replay_after_root':0,
                    'branch_snapshot_bytes':0, 'validated':True}
                counters = trajectory['counters']
                if (trajectory.get('early_execution_audit') != expected
                        or a['native_cache_audit'].get('early_execution') != expected
                        or counters.get('replayed_chunks') != 1
                        or counters.get('stopped_draft_probes') != budget
                        or counters.get('completed_draft_generations') != 29-budget
                        or counters.get('current_slot_clean_updates') != budget):
                    raise ValueError('early stop, clean update or incremental history evidence differs')
            if burst:
                from .rank_burst import validate_burst_schedule
                validate_burst_schedule(trajectory, budget=budget, burst_length=burst_length_for_arm(arm),
                    observation_horizon=observation_horizon_for_arm(arm))
                reuse = a['native_cache_audit'].get('burst_reuse', {})
                if (reuse.get('suffix_replay_bound_verified') is not True
                        or reuse.get('burst_length') != burst_length_for_arm(arm)
                        or reuse.get('maximum_replayed_chunks', 30) > burst_length_for_arm(arm)):
                    raise ValueError('burst native suffix reuse is not verified')
            if block:
                positions = list(block_positions(arm, prompt_id=pid))
                if trajectory['accepted_chunks'] != positions:
                    raise ValueError('block-control intervention positions differ from frozen plan')
                if arm == 'target_root_only_k0' and trajectory['counters'].get('replayed_chunks') != 1:
                    raise ValueError('root-only Draft rebuilt history after the initial root replay')
        if a['prompt'] != b['prompt'] or a['seed'] != b['seed'] or a['seed'] != 20260911:
            raise ValueError('prompt or seed mismatch')
        for row in (a,b):
            for key in ('generation_wall_seconds','video_ready_wall_seconds'):
                v = row[key]
                if type(v) not in (int,float) or not math.isfinite(v) or v <= 0:
                    raise ValueError('invalid timing')
            if row['video_ready_wall_seconds'] < row['generation_wall_seconds']:
                raise ValueError('video-ready timing excludes generation')
            scores = row['quality_scores']
            if set(scores) != set(QUALITY_DIMENSIONS) or any(type(v) not in (int,float)
                    or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
                raise ValueError('incomplete or invalid six-dimension quality')
        paired.append({'prompt_id':pid, 'prompt':a['prompt'],
            'candidate_quality':sum(a['quality_scores'].values())/6*100,
            'target_quality':sum(b['quality_scores'].values())/6*100,
            'video_ready_speedup':b['video_ready_wall_seconds']/a['video_ready_wall_seconds'],
            'shared_target_root_bytes':a['trajectory']['committed'][0]['latent_sha256'] == b['root_latent_sha256'],
            'accepted':len(a['trajectory']['accepted_chunks']),
            'proposed':len(a['trajectory']['proposed_chunks'])})
    quality = sum(r['candidate_quality'] for r in paired)/len(paired)
    latency = sum(r['video_ready_wall_seconds'] for r in candidate['rows'])/len(paired)
    target_latency = sum(r['video_ready_wall_seconds'] for r in target['rows'])/len(paired)
    speedup = target_latency/latency
    dimensions = {d: {
        'candidate_mean_100': sum(r['quality_scores'][d] for r in candidate['rows'])/len(paired)*100,
        'target_mean_100': sum(r['quality_scores'][d] for r in target['rows'])/len(paired)*100,
        'contribution_to_overall_gap_points': sum(
            indexed[0][pid]['quality_scores'][d]-indexed[1][pid]['quality_scores'][d]
            for pid in prompt_ids)/len(paired)*100/6,
    } for d in QUALITY_DIMENSIONS}
    return {'schema':'erank_acceptance_matched_comparison_v1','arm':arm,'encoding_policy':encoding,
        'population':('main24_fixed_first12_target_prefix_diagnostic_not_held_out' if prefix else
                      'main24_fixed_first12_early_current_diagnostic_not_held_out' if early else
                      'main24_fixed_first12_rank_burst_diagnostic_not_held_out' if burst else
                      'main24_fixed_first12_block_control_not_held_out' if block else
                      'main24_fixed_first12_causal_direct_diagnostic_not_held_out' if direct
                      else 'main24_method_development_not_held_out'), 'prompt_count':len(paired),
        'quality_dimensions':dimensions,
        'quality_macro_mean_100':quality, 'target_quality_macro_mean_100':sum(r['target_quality'] for r in paired)/len(paired),
        'video_ready_mean_seconds':latency, 'target_video_ready_mean_seconds':target_latency,
        'video_ready_speedup_ratio_of_means':speedup, 'timing_scope':READY_TIMING_SCOPE,
        'criteria':{'quality_at_least':82.45,'target_speedup_at_least':1.5},
        'meets_numeric_development_thresholds':quality >= 82.45 and speedup >= 1.5,
        'publication_ready_method_claim':False, 'paired':paired}


def load_run(root: Path, *, target: bool, supplement_root: Path | None = None) -> tuple[dict,dict]:
    frozen, marker = _load_frozen_index(root)
    prefix = 'attempts/'+marker['attempt_id']+'/'
    def read(relative):
        return json.loads(frozen.file(prefix+relative,label=relative).read_text())
    data = read('provenance/acceptance-target-reference.json' if target else 'result.json')
    if data['code_revision'] != marker['git_sha']:
        raise ValueError('result code revision differs from sealed run')
    from .video_encoding_policy import LEGACY, ENV, validate_policy, verify_video
    resolved = json.loads(frozen.file('resolved-manifest.json', label='resolved manifest').read_text())
    encoding = validate_policy(resolved['manifest']['spec']['command'].get('env', {}).get(ENV, LEGACY))
    if population_encoding(data) != encoding:
        raise ValueError('result encoding policy differs from resolved manifest')
    for row in data['rows']:
        for field in (('video','native_evidence') if target else ('video','decode_evidence','quality_evidence')):
            binding = row[field]
            if field == 'decode_evidence' and prefix+binding['path'] not in frozen.entries and supplement_root is not None:
                from .acceptance_evidence_supplement import supplementary_decode
                source_result = frozen.file(prefix+'result.json',label='source result')
                supplementary_decode(supplement_root, marker, hashlib.sha256(source_result.read_bytes()).hexdigest(),
                                     row['prompt_id'], binding)
                continue
            path = frozen.file(prefix+binding['path'],label=field)
            indexed_size, indexed_hash = frozen.entries[prefix+binding['path']]
            if field == 'video':
                verify_video(path, encoding)
            if (indexed_size,indexed_hash) != (binding['nbytes'],binding['sha256']):
                raise ValueError('row binding differs from sealed artifact index')
    return data, marker


def compare_fixed_prefix(candidate: dict, target: dict, frozen_ids: tuple[str, ...],
                         count: int = 24) -> dict:
    """Validate the complete cohort before selecting a predeclared prefix.

    Selection follows the frozen manifest order, never completion order, score,
    or filename order. Input runs and their sealed artifacts remain unchanged.
    """
    if len(frozen_ids) != 24 or len(set(frozen_ids)) != 24 or count not in (12, 24):
        raise ValueError('expected 24 unique frozen prompts and a 12 or 24 prompt report')
    full = compare(candidate, target, frozen_ids)
    selected = frozen_ids[:count]
    if count == 24:
        result = full
    else:
        def subset(data):
            indexed = {r['prompt_id']: r for r in data['rows']}
            return {**data, 'rows': [indexed[pid] for pid in selected]}
        result = compare(subset(candidate), subset(target), selected)
        result['population'] = 'main24_fixed_first12_method_development_not_held_out'
    result['selection'] = {'policy': 'frozen_population_prefix',
                           'source_prompt_count': 24, 'prompt_ids': list(selected)}
    return result


def compare_direct_prefix(candidate: dict, target: dict, frozen_ids: tuple[str, ...]) -> dict:
    """Compare a twelve-prompt diagnostic to its complete, matched Target run."""
    if (len(frozen_ids) != 24 or len(set(frozen_ids)) != 24
            or candidate.get('cohort') not in {'main12_direct_15s', 'main12_block_15s', 'main12_burst_15s', 'main12_early_15s', 'main12_prefix_15s'}):
        raise ValueError('expected direct diagnostic and frozen 24-prompt source population')
    if population_encoding(candidate) != population_encoding(target):
        raise ValueError('candidate and full Target encoding policies differ')
    target_rows = {row['prompt_id']:row for row in target['rows']}
    if len(target_rows) != len(target['rows']) or set(target_rows) != set(frozen_ids):
        raise ValueError('Target reference must cover the entire frozen population')
    # Validate even the unselected Target rows; a prefix must not hide a failed score.
    for row in target['rows']:
        scores = row['quality_scores']
        times = [row[k] for k in ('generation_wall_seconds','video_ready_wall_seconds')]
        if (row['seed'] != 20260911 or set(scores) != set(QUALITY_DIMENSIONS)
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1
                       for v in scores.values())
                or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in times)
                or times[1] < times[0]):
            raise ValueError('invalid full Target reference metrics')
    selected = frozen_ids[:12]
    result = compare(candidate, {**target,'rows':[target_rows[pid] for pid in selected]}, selected)
    result['selection'] = {'policy':'frozen_population_prefix','source_prompt_count':24,
                           'prompt_ids':list(selected)}
    result['calibration_code_revision'] = candidate['calibration_code_revision']
    result['execution_code_revision'] = candidate['code_revision']
    result['diagnostic_changes'] = (
        ['Target_first_update_only', 'Draft_three_step_suffix', 'explicit_renoise', 'native_incremental_KV']
        if candidate['cohort'] == 'main12_prefix_15s' else
        ['same_current_chunk_erank_rule', 'stop_at_first_forward_layer14',
         'continue_same_forward_if_draft', 'one_clean_forward_after_target', 'no_post_root_history_replay']
        if candidate['cohort'] == 'main12_early_15s' else
        ['committed_draft_rank_signal', 'future_target_bursts', 'no_discarded_draft',
         'bounded_missing_suffix_replay', 'explicit_target_block_budget']
        if candidate['cohort'] == 'main12_burst_15s' else
        ['fixed_target_positions', 'skip_draft_on_target_chunks', 'no_rank_routing',
         'unconditional_target_replacement', 'explicit_replacement_budget']
        if candidate['cohort'] == 'main12_block_15s' else
        ['current_chunk_causal_rank_signal', 'no_successor_probe',
         'unconditional_target_replacement', 'explicit_replacement_budget'])
    return result


def compare_random_p8_prefix(candidate: dict, target: dict, frozen_ids: tuple[str, ...]) -> dict:
    """Verify three routing repeats and use the twelve prompts as the analysis units."""
    repeat_count = len(RANDOM_P8_B4_ARMS)
    if candidate.get('arms') != list(RANDOM_P8_B4_ARMS) or len(candidate['rows']) != 12 * repeat_count:
        raise ValueError('Random P8 requires exactly three complete routing repeats')
    reports = {}
    for arm in RANDOM_P8_B4_ARMS:
        rows = [row for row in candidate['rows'] if row['arm'] == arm]
        reports[arm] = compare_direct_prefix({**candidate, 'arms': [arm], 'rows': rows},
                                             target, frozen_ids)
    reference = reports[RANDOM_P8_B4_ARMS[0]]
    all_pairs = [result['paired'] for result in reports.values()]
    candidate_times = {(row['arm'], row['prompt_id']): row['video_ready_wall_seconds']
                       for row in candidate['rows']}
    target_times = {row['prompt_id']: row['video_ready_wall_seconds'] for row in target['rows']}
    paired = []
    for members in zip(*all_pairs, strict=True):
        first = members[0]
        if not all(row['prompt_id'] == first['prompt_id'] and row['prompt'] == first['prompt']
                   and row['shared_target_root_bytes'] for row in members):
            raise ValueError('Random P8 changed the prompt, seed, or accepted Target root')
        prompt_id = first['prompt_id']
        paired.append(dict(prompt_id=prompt_id, prompt=first['prompt'],
            random_quality_100=math.fsum(row['candidate_quality'] for row in members)/repeat_count,
            target_quality_100=first['target_quality'],
            random_ready_seconds=math.fsum(candidate_times[arm, prompt_id] for arm in RANDOM_P8_B4_ARMS)/repeat_count,
            target_ready_seconds=target_times[prompt_id]))
    quality = math.fsum(row['random_quality_100'] for row in paired)/len(paired)
    ready = math.fsum(row['random_ready_seconds'] for row in paired)/len(paired)
    return dict(schema='sf_random_p8_b4_matched_report_v1', status='verified',
        prompt_count=len(paired), routing_repeat_count=len(reports), non_root_target_blocks=4,
        observation_horizon=8, quality_macro_mean_100=quality, video_ready_mean_seconds=ready,
        target_quality_macro_mean_100=reference['target_quality_macro_mean_100'],
        target_video_ready_mean_seconds=reference['target_video_ready_mean_seconds'],
        video_ready_speedup_ratio_of_means=reference['target_video_ready_mean_seconds']/ready,
        selection=reference['selection'], timing_scope=reference['timing_scope'],
        paired=paired, repeats=reports)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--decode-supplement',type=Path,
                   help='Separate sealed supplement for legacy unindexed decode JSON; original seals remain unchanged')
    p.add_argument('--prompt-count',type=int,choices=(12,24),default=24,
                   help='Report a fixed prefix of the frozen population; validate both full sealed runs first')
    a=p.parse_args(argv)
    rows,_=load_baseline_population(find_git_root(Path.cwd()))
    candidate,cm=load_run(a.candidate,target=False,supplement_root=a.decode_supplement)
    target,tm=load_run(a.target,target=True)
    ids=tuple(r['id'] for r in rows)
    if candidate.get('arms') == list(RANDOM_P8_B4_ARMS):
        if a.prompt_count != 12:
            p.error('Random P8 requires --prompt-count 12')
        result=compare_random_p8_prefix(candidate,target,ids)
    elif candidate.get('cohort') in {'main12_direct_15s', 'main12_block_15s', 'main12_burst_15s', 'main12_early_15s', 'main12_prefix_15s'}:
        if a.prompt_count != 12:
            p.error('direct diagnostic requires --prompt-count 12')
        result=compare_direct_prefix(candidate,target,ids)
    else:
        result=compare_fixed_prefix(candidate,target,ids,a.prompt_count)
    result['sources']={'candidate':cm,'target':tm}
    if a.decode_supplement is not None:
        _, sm = _load_frozen_index(a.decode_supplement)
        result['sources']['decode_supplement'] = sm
    _write_new_json(_run_dir()/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired','sources')},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
