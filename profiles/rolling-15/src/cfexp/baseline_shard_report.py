"""Merge four published native Draft shards; never infer video-ready speedup."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from .baseline_population import load_baseline_population, select_prompt_shard
from .duration_sweep import duration_profile
from .main_table_merge import _load_frozen_index, _load_json, _require, _sha256_file
from .main_table_v1 import QUALITY_DIMENSIONS
from .manifest import normalize_manifest

PAIRS = {'self-forcing': 'krea_self_forcing', 'longlive': 'krea_longlive'}


def source_manifest(repo, commit, relative):
    _require(re.fullmatch(r'[0-9a-f]{40}', commit) is not None, 'full source SHA required')
    _require(re.fullmatch(r'experiments/[a-z0-9-]+\.manifest\.json', relative) is not None,
             'unsafe source manifest')
    env = dict(PATH='/usr/bin:/bin', HOME=os.path.expanduser('~'),
               GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_SYSTEM='/dev/null',
               GIT_CONFIG_NOSYSTEM='1', GIT_NO_REPLACE_OBJECTS='1')
    return normalize_manifest(json.loads(subprocess.check_output(
        ['git', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null',
         '-C', str(repo), 'show', commit+':'+relative], env=env)))


def _bound(frozen, prefix, item, label):
    path = frozen.file(prefix+item['path'], label=label)
    size = item.get('nbytes', item.get('bytes'))
    _require(_sha256_file(path) == (item['sha256'], size), label+' binding mismatch')
    return _load_json(path, label=label)


def _published(row, run, marker):
    _require(row['status'] == 'succeeded' and row.get('success') == marker,
             'queue has not confirmed the exact sealed publication')
    if row.get('exit_code') == 2:
        repair = row.get('publication_reconciliation', {})
        _require(repair.get('succeeded') is True and repair.get('returncode') == 0 and
                 repair.get('marker_sha256') == _sha256_file(run/'_SUCCESS.json')[0] and
                 repair.get('artifact_index_sha256') == marker['artifact_index_sha256'],
                 'publication reconciliation does not bind the original seal')
        report = Path(repair['report_path'])
        _require(not report.is_symlink() and _sha256_file(report)[0] == repair['report_sha256'],
                 'publication reconciliation report changed')
        outcome = _load_json(report, label='publication reconciliation')
        successes = outcome.get('successes', [])
        _require(not outcome.get('failures') and not outcome.get('skipped') and len(successes) == 1 and
                 successes[0].get('run_id') == marker['run_id'] and
                 successes[0].get('marker') == '_SUCCESS.json' and
                 successes[0].get('destination') == marker['artifact_publication']['destination'],
                 'publication reconciliation destination mismatch')
    else:
        _require(row.get('exit_code') == 0, 'publication completion lacks a valid wrapper exit')


def analyze_queue(*, queue: Path, repo: Path, model: str, seconds: int, commit: str):
    _require(model in PAIRS and type(seconds) is int and seconds in (15, 30, 60),
             'unsupported native baseline condition')
    _require(not queue.is_symlink(), 'queue plan may not be a symlink')
    plan = _load_json(queue, label='queue plan')
    state = _load_json(Path(plan['ops'])/'state.json', label='queue state')
    _require(plan['schema'] == 'cfexp_erank_queue_v1' and
             state['plan_sha256'] == _sha256_file(queue)[0], 'queue plan/state binding changed')
    version = '1' if model == 'self-forcing' else '2'
    pattern = re.compile(r'experiments/baseline24-'+model+r'-draft-'+str(seconds)+
                         r's-shard([0-3])-ntu-v'+version+r'\.manifest\.json')
    selected = [(int(m[1]), t) for t in plan['tasks'] if (m := pattern.fullmatch(t['manifest']))]
    _require(len(selected) == 4 and sorted(i for i, _ in selected) == list(range(4)),
             'exactly four distinct formal native Draft shards are required')
    _require(all(t['commit'] == commit for _, t in selected), 'shard revisions differ')
    unfinished = {t['id']: state['tasks'][t['id']]['status'] for _, t in selected
                  if state['tasks'][t['id']]['status'] != 'succeeded'}
    if unfinished:
        return {'schema': 'native_draft_shard_quality_report_v1', 'status': 'not_ready',
                'unfinished': unfinished, 'speedup_verified': False}
    prompts, population = load_baseline_population(repo, model_pair=PAIRS[model])
    profile = duration_profile(seconds)
    benchmark = 'VBench-Long' if seconds == 60 else 'VBench'
    combined, sources = {}, []
    for index, task in sorted(selected):
        row = state['tasks'][task['id']]
        requested_run = Path(row['run_dir'])
        _require(not requested_run.is_symlink(), 'source run may not be a symlink')
        run = requested_run.resolve(strict=True)
        _require(run.parent.resolve() == (Path(plan['root'])/'runs').resolve(), 'source escaped queue run root')
        frozen, marker = _load_frozen_index(run)
        frozen.verify_all()
        _published(row, run, marker)
        _require(marker['git_sha'] == commit, 'sealed code revision mismatch')
        prefix = 'attempts/'+marker['attempt_id']+'/'
        terminal = _load_json(frozen.file('run.json', label='run'), label='run')
        _require(terminal.get('execution_status') == 'SUCCEEDED' and terminal.get('exit_code') == 0 and
                 terminal.get('latest_attempt_id') == marker['attempt_id'] and
                 terminal.get('run_id') == marker['run_id'], 'run terminal identity mismatch')
        envelope = _load_json(frozen.file('resolved-manifest.json', label='manifest'), label='manifest')
        _require(all(envelope.get(k) == marker[k] for k in ('run_id', 'attempt_id', 'spec_hash')) and
                 envelope['manifest'] == source_manifest(repo, commit, task['manifest']),
                 'sealed invocation differs from the published source manifest')
        protocol = 'experiments/'+population['protocol']
        _require(any(x.get('target') == protocol and x.get('sha256') == population['sha256']
                     for x in envelope['manifest']['spec']['inputs']),
                 'population working copy differs from the published source binding')
        result = _load_json(frozen.file(prefix+'result.json', label='result'), label='result')
        expected_pop = dict(population)
        if model == 'longlive':
            expected_pop.update(shard_index=index, shard_count=4, full_population_count=24)
        _require(result.get('status') == 'succeeded' and result.get('method_id') == 'draft_only' and
                 result.get('run_kind') == 'formal_prompt_shard' and result.get('prompt_count') == 6 and
                 result.get('model_pair') == PAIRS[model] and result.get('benchmark') == benchmark and
                 result.get('population') == expected_pop, 'shard population/model/benchmark mismatch')
        if model == 'self-forcing':
            _require(result.get('schema') == 'self_forcing_baseline_campaign_result_v1' and
                     result.get('nominal_duration_seconds') == seconds and result.get('seed') == 20260911 and
                     result.get('shard_index') == index and result.get('shard_count') == 4,
                     'native Self-Forcing shard identity mismatch')
            evidence = result['quality_evidence']
        else:
            _require(result.get('schema') == 'baseline_population_v2_method_result' and
                     result.get('row_count') == 6 and result.get('seed_count') == 1 and
                     result['duration']['nominal_seconds'] == seconds and
                     result['duration']['frame_count'] == profile.frame_count and
                     result['duration']['fps'] == 24, 'native LongLive shard identity mismatch')
            evidence = result['vbench_measurements']
        rows = _bound(frozen, prefix, result['raw_rows'], 'raw rows')['rows']
        shard = select_prompt_shard(prompts, index, 4)
        ids = [p['id'] for p in shard]
        _require(len(rows) == len(evidence) == 6 and [r['prompt_id'] for r in rows] == ids and
                 [e['prompt_id'] for e in evidence] == ids, 'shard coverage/order mismatch')
        for item, proof, prompt in zip(rows, evidence, shard):
            flag = ('official_quality_verified' if model == 'self-forcing' else
                    'official_vbench_long_verified' if seconds == 60 else 'official_vbench_verified')
            _require(item.get(flag) is True and item.get('model_pair') == PAIRS[model] and
                     item.get('method_id') == 'draft_only' and item.get('seed') == 20260911,
                     'row model, seed or official quality provenance mismatch')
            scores = item['quality_scores']
            _require(set(scores) == set(QUALITY_DIMENSIONS) and all(type(x) in (int, float) and
                     math.isfinite(x) and 0 <= x <= 1 for x in scores.values()), 'invalid quality scores')
            sidecar = _bound(frozen, prefix, proof, 'quality evidence')
            mid = ('self-forcing-' if model == 'self-forcing' else 'mt-')+'draft_only-'+prompt['id']
            _require(sidecar.get('measurement_id') == mid and sidecar.get('dimension_scores') == scores,
                     'row quality differs from its evaluator sidecar')
            _require(abs(sidecar.get('macro_mean', -1)-math.fsum(scores.values())/6) < 1e-10,
                     'quality sidecar aggregate differs from its dimensions')
            if seconds == 60:
                config = _bound(frozen, prefix, sidecar['batch_config'], 'VBench-Long config')
                attempt = _bound(frozen, prefix, sidecar['batch_attempt'], 'VBench-Long attempt')
                _bound(frozen, prefix, sidecar['official_result'], 'VBench-Long result')
                _require(attempt.get('returncode') == 0, 'official VBench-Long did not succeed')
                inputs = [v for v in config['inputs'] if v['measurement_id'] == mid]
                _require(len(inputs) == 1 and inputs[0]['prompt'] == prompt['text'] and
                         inputs[0]['source_video'] == sidecar['source_video'],
                         'VBench-Long scored different video/prompt')
                video = sidecar['source_video']
            else:
                config = _bound(frozen, prefix, sidecar['bindings']['config'], 'VBench config')
                outcome = _bound(frozen, prefix, sidecar['bindings']['exit'], 'VBench exit')
                _require(outcome.get('returncode') == 0, 'official VBench did not succeed')
                inputs = [v for v in config['inputs'] if v['id'] == mid]
                _require(len(inputs) == 1 and inputs[0]['prompt'] == prompt['text'],
                         'VBench scored a different prompt')
                video = dict(path=inputs[0]['source'], sha256=inputs[0]['sha256'], nbytes=inputs[0]['nbytes'])
            source = Path(video['path'])
            _require(source.is_absolute(), 'evaluator video source must be absolute')
            try:
                relative = source.relative_to(run).as_posix()
            except ValueError as error:
                raise ValueError('evaluator video escaped its sealed run') from error
            actual = frozen.file(relative, label='scored video')
            _require(_sha256_file(actual) == (video['sha256'], video['nbytes']),
                     'scored video identity changed')
            if model == 'self-forcing':
                _require(item.get('nominal_duration_seconds') == seconds and item.get('fps') == 24 and
                         item.get('decoded_frame_count') == profile.frame_count and
                         prefix+item['video']['path'] == relative and
                         item['video']['sha256'] == video['sha256'], 'native Self-Forcing video/geometry mismatch')
            else:
                _require(item.get('duration') == result['duration'], 'native LongLive row horizon mismatch')
            _require(prompt['id'] not in combined, 'duplicate prompt across shards')
            combined[prompt['id']] = scores
        sources.append({'task': task['id'], 'job': task['job'], 'gpus': task['gpus'],
                        'run_id': marker['run_id'], 'attempt_id': marker['attempt_id'],
                        'artifact_index_sha256': marker['artifact_index_sha256'],
                        'destination': marker['artifact_publication']['destination']})
    _require(set(combined) == {p['id'] for p in prompts}, 'merged population incomplete')
    means = {d: math.fsum(combined[p['id']][d] for p in prompts)/24*100 for d in QUALITY_DIMENSIONS}
    return {'schema': 'native_draft_shard_quality_report_v1', 'status': 'verified',
            'model_pair': PAIRS[model], 'nominal_seconds': seconds, 'benchmark': benchmark,
            'method_id': 'draft_only', 'prompt_count': 24, 'seed': 20260911,
            'quality_definition': 'unweighted_six_dimension_mean_percent_not_official_total',
            'quality_dimension_means_100': means, 'quality_mean_100': math.fsum(means.values())/6,
            'population': population, 'generation_commit': commit, 'sources': sources,
            'speedup_verified': False, 'video_ready_latency_verified': False,
            'timing_note': 'Legacy shard timing cannot establish a matched full video-ready speedup.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--model', choices=tuple(PAIRS), required=True)
    parser.add_argument('--seconds', type=int, choices=(15, 30, 60), required=True)
    parser.add_argument('--commit', required=True)
    print(json.dumps(analyze_queue(**vars(parser.parse_args())), sort_keys=True))


if __name__ == '__main__':
    main()
