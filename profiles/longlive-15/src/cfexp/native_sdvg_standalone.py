"""Audit sealed native SDVG outputs and, when supplied, their paired Target."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import native_ready_report as audit
from .baseline_population import load_baseline_population, model_pair_for_draft
from .correction_state import LatentHistory
from .duration_sweep import duration_profile
from .main_table_sdvg import THRESHOLD, SCHEMA
from .native_random_standalone import summarize
from .native_reference_report import MODELS, official_scores, reference_rows, timing_scope
from .native_sdvg import audit_native_draft_history


def header(src, repo, *, model, seconds):
    audit._require(model in MODELS and type(seconds) is int and seconds in (15, 30, 60),
                   'unsupported standalone SDVG condition')
    profile = duration_profile(seconds)
    prompts, population = load_baseline_population(repo,
        model_pair=model_pair_for_draft(MODELS[model]))
    benchmark = 'VBench-Long' if seconds == 60 else 'VBench'
    duration = {k: v for k, v in profile.metadata().items()
                if not k.startswith(('ours_', 'calibration_')) and k != 'budget_rule'}
    duration.update(experiment_id='main-table-population-v2', benchmark=benchmark)
    result = src.result
    audit._require(result.get('schema') == 'baseline_population_v2_method_result'
        and result.get('method_id') == 'sdvg' and result.get('run_kind') == 'formal_campaign'
        and result.get('native_model') == model and result.get('population') == population
        and result.get('prompt_count') == result.get('row_count') == 24
        and result.get('seed_count') == 1 and result.get('duration') == duration
        and result.get('benchmark') == benchmark
        and result.get('draft_history_policy') == 'always_append_draft_candidate'
        and result.get('selected_history_policy') == 'append_accepted_draft_else_target_regeneration',
        'source is not the frozen formal native SDVG cohort')
    command = src.manifest['spec']['command']
    required = {'CFEXP_ACCEPTED_VIDEO_STREAM': '1',
        'CFEXP_RENDER_TARGET_PRIORITY': '1', 'CFEXP_VIDEO_ONLY_DECODE': '1',
                'CFEXP_VIDEO_ENCODING': 'serial_abr_v1'}
    if model == 'self-forcing':
        audit._require('CFEXP_NATIVE_VIDEO_STREAM_MODEL' not in command['env'],
                       'Self-Forcing SDVG selected another decoder')
        required.update(CFEXP_DRAFT_KV_REUSE='1', CFEXP_DRAFT_KV_FORK='0',
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    else:
        required.update(CFEXP_NATIVE_VIDEO_STREAM_MODEL=model, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    argv = command['argv']
    audit._require(result.get('timing_scope') == timing_scope(model)
        and 'cfexp.main_table_sdvg_campaign' in argv
        and '--record-video-ready-timing' in argv and '--overlap-native-vae' in argv
        and all(command['env'].get(k) == v for k, v in required.items()),
        'SDVG renderer, invocation or timing differs')
    spec_path = (repo / argv[argv.index('--sdvg-artifact-spec') + 1]).resolve(strict=True)
    audit._require(spec_path.is_relative_to(repo.resolve(strict=True)), 'SDVG spec escaped repository')
    relative = spec_path.relative_to(repo.resolve()).as_posix()
    digest = audit._sha256_file(spec_path)[0]
    audit._require(any(x['target'] == relative and x['sha256'] == digest
                      for x in src.manifest['spec']['inputs']), 'SDVG spec is not source-bound')
    spec = json.loads(spec_path.read_text())
    routing = spec['adapter_config']['routing']
    audit._require(spec.get('status') == 'frozen' and spec.get('method_id') == 'sdvg'
        and spec.get('adapter_module') == 'cfexp.sdvg_adapter'
        and routing.get('threshold') == THRESHOLD and routing.get('force_reject_chunk_indices') == [0]
        and routing.get('draft_history_policy') == 'always_append_draft_candidate'
        and routing.get('target_history_policy') == 'append_accepted_draft_else_target_regeneration'
        and routing.get('rejection_noise_policy') == 'target_regenerates_from_same_initial_noise'
        and spec['adapter_config']['benchmark_adaptations']['chunk_count'] == profile.chunk_count,
        'SDVG routing artifact differs')
    preflight = audit.read_json(src, 'provenance/preflight.json')
    audit._require(preflight.get('git', {}).get('sha') == result['code_revision']
        and preflight.get('sdvg_artifact_spec_sha256') == digest
        and preflight.get('draft_model_id') == MODELS[model]
        and preflight.get('threshold') == THRESHOLD and preflight.get('seed') == 20260911
        and preflight.get('prompt_ids') == [p['id'] for p in prompts], 'SDVG preflight differs')
    return profile, prompts, duration


def histories(src, provenance, row, *, model, profile):
    pid = row['prompt_id']
    draft = LatentHistory.from_dict(audit.read_json(src, f'provenance/sdvg/{pid}-draft-history.json'))
    selected = LatentHistory.from_dict(audit.read_json(src, f'provenance/sdvg/{pid}-selected-history.json'))
    generation = provenance['generation']
    audit._require(generation.get('schema') == SCHEMA and generation.get('method_id') == 'sdvg'
        and generation.get('draft_model_id') == MODELS[model]
        and generation.get('prompt_id') == pid and generation.get('seed') == 20260911
        and generation.get('chunk_count') == profile.chunk_count
        and generation.get('threshold') == THRESHOLD
        and generation['draft_history_manifest_sha256'] == draft.manifest_sha256
        and generation['selected_history_manifest_sha256'] == selected.manifest_sha256,
        'SDVG generation/history identity differs')
    for history in (draft, selected):
        audit._require(history.prompt_id == pid and history.trajectory_seed == 20260911
            and history.next_chunk_index == profile.chunk_count, 'SDVG history coverage differs')
        for segment in history.segments:
            audit.referenced_file(src, dict(path=segment.latent_path, sha256=segment.latent_sha256,
                                            nbytes=segment.latent_nbytes), 'SDVG latent')
    proof = audit_native_draft_history(generation)
    audit._require(row['native_cache_audit'] == provenance['native_cache_audit'] == proof,
                   'SDVG cache audit differs from per-chunk evidence')
    targets = []
    for i, (route, d, s) in enumerate(zip(generation['routes'], draft.segments, selected.segments)):
        reward = route['block_reward']
        audit._require(type(reward) in (int, float) and math.isfinite(reward), 'invalid SDVG reward')
        accepted = i > 0 and reward >= THRESHOLD
        audit._require(route['threshold'] == THRESHOLD and route['forced_reject'] is (i == 0)
            and route['route'] == ('draft' if accepted else 'target')
            and route['reward']['block_reward'] == reward
            and route['reward']['threshold'] == THRESHOLD
            and route['reward']['accepted_by_threshold'] is (reward >= THRESHOLD)
            and route['reward']['forced_reject'] is (i == 0)
            and d.producer_role == 'draft' and s.producer_role == route['route']
            and route['draft_latent_path'] == d.latent_path
            and route['selected_latent_path'] == s.latent_path
            and route['selected_latent_sha256'] == s.latent_sha256
            and d.initial_noise_descriptor_sha256 == s.initial_noise_descriptor_sha256
            and (not accepted or d.latent_sha256 == s.latent_sha256),
            'SDVG reward, selected latent or same-noise decision differs')
        if not accepted:
            targets.append(i)
    audit._require(generation['target_regeneration_count'] == len(targets)
        and generation['draft_accept_count'] == profile.chunk_count - len(targets),
        'SDVG target count differs')
    return selected, targets


def pair_row(sdvg, history, target):
    row = target['row']
    audit._require((sdvg['prompt_id'], sdvg['prompt']) == (row['prompt_id'], row['prompt']),
                   'SDVG/Target prompt order or text differs')
    sdvg_segments = history.segments
    target_segments = target['history'].segments
    audit._require(sdvg_segments[0].latent_sha256 == target_segments[0].latent_sha256,
                   'SDVG/Target root latents differ')
    audit._require([s.initial_noise_descriptor_sha256 for s in sdvg_segments] ==
                   [s.initial_noise_descriptor_sha256 for s in target_segments],
                   'SDVG/Target initial noises differ')
    target_identity = (target_segments[0].model_id, target_segments[0].model_artifact_sha256)
    audit._require(all((s.model_id, s.model_artifact_sha256) == target_identity
                       for s in (*target_segments, *sdvg_segments) if s.producer_role == 'target'),
                   'SDVG/Target model or checkpoint differs')
    return dict(sdvg, target_quality_scores=row['quality_scores'],
                target_ready_seconds=row['video_ready_wall_seconds'],
                target_video_sha256=row['video']['sha256'])


def report(src, *, repo, model, seconds, target=None):
    profile, prompts, duration = header(src, repo, model=model, seconds=seconds)
    ready_scope = timing_scope(model)
    if target is not None:
        scheduling = audit.scheduling_revision(src.result['code_revision'], target, repo)
        if model == 'self-forcing':
            audit._require(src.manifest['spec']['command']['env'] == target.manifest['spec']['command']['env'],
                           'Self-Forcing SDVG/Target execution flags differ')
        sdvg_gpus = [gpu['uuid'] for gpu in src.result['hardware']['gpus']]
        target_gpus = [gpu['uuid'] for gpu in target.result['hardware']['gpus']]
        audit._require(len(sdvg_gpus) == len(set(sdvg_gpus)) == 2 and all(sdvg_gpus)
                       and sdvg_gpus == target_gpus, 'SDVG/Target physical GPU pair or role differs')
        target_rows = reference_rows(target, repo=repo, model=model, seconds=seconds, role='target')
        audit._require(len(target_rows) == len(prompts), 'SDVG/Target cohort coverage differs')
    rows = audit._bound(src.frozen, src.prefix, src.result['raw_rows'], 'SDVG raw rows')['rows']
    proofs = src.result['vbench_measurements']
    audit._require(len(rows) == len(proofs) == 24
        and [r['prompt_id'] for r in rows] == [p['id'] for p in prompts], 'SDVG prompt coverage differs')
    checked, common_codec, short_batches = [], None, {}
    metric = 'official_vbench_long' if seconds == 60 else 'official_vbench'
    for index, (prompt, row, evidence) in enumerate(zip(prompts, rows, proofs)):
        provenance = audit.read_json(src, 'provenance/sdvg/' + prompt['id'] + '.json')
        audit._require(row['method_id'] == provenance['method_id'] == 'sdvg'
            and provenance['schema'] == 'main_table_sdvg_prompt_result_v1'
            and provenance['prompt_id'] == prompt['id']
            and row['seed'] == provenance['seed'] == 20260911 and row['duration'] == duration
            and row[metric + '_verified'] is True and provenance[metric]['verified'] is True
            and provenance[metric]['dimension_scores'] == row['quality_scores']
            and provenance[metric]['evidence'] == evidence
            and row['timing_scope'] == provenance['timing_scope'] == ready_scope
            and row['video_ready_wall_seconds'] == provenance['video_ready_wall_seconds']
            and audit.positive(row['video_ready_wall_seconds'], 'ready') >=
                audit.positive(row['generation_wall_seconds'], 'generation')
            and row['generation_wall_seconds'] == provenance['generation']['timing']['generation_wall_seconds'],
            'SDVG row/scoring/timing identity differs')
        history, targets = histories(src, provenance, row, model=model, profile=profile)
        mid = 'mt-sdvg-' + prompt['id']
        video = audit.quality(src, row, evidence, prompt, mid, seconds)
        official_scores(src, dict(row, quality_evidence=evidence), prompt, mid, seconds,
                        video[0], short_batches=short_batches)
        audit.verify_geometry(video[0], profile)
        generation = provenance['generation']
        declared = generation['video']
        audit._require(video[0] == src.run / src.prefix / declared['path']
            and all(video[1][k] == declared[k] for k in ('sha256', 'nbytes'))
            and declared['fps'] == 24 and declared['frames'] == profile.frame_count
            and generation['frames'] is not None
            and row['accepted_stream_evidence'] == provenance['accepted_stream_evidence'],
            'SDVG scored video differs from final accepted history')
        # The sealed rollout retains pixels/video directly, rather than a separate decode sidecar.
        from .video_only_decode import SCHEMA as DECODE_SCHEMA
        audit.verify_accepted_renderer(src, row['accepted_stream_evidence'], history,
            dict(schema=DECODE_SCHEMA, pixels=generation['frames'], video=declared),
            model=model, role='target', profile=profile)
        audit._require(common_codec is None or common_codec == video[2], 'SDVG encoders differ')
        common_codec = video[2]
        cell = dict(prompt_id=prompt['id'], prompt=prompt['text'],
            quality_scores=row['quality_scores'], ready_seconds=row['video_ready_wall_seconds'],
            generation_seconds=row['generation_wall_seconds'], target_chunks=targets,
            video_sha256=video[1]['sha256'])
        if target is not None:
            matched = target_rows[index]
            audit._require(video[2] == matched['codec'], 'SDVG/Target actual scored encoders differ')
            cell = pair_row(cell, history, matched)
        checked.append(cell)
    result = dict(schema='native_sdvg_standalone_report_v1', status='verified', method_id='sdvg',
        model=model, nominal_seconds=seconds, prompt_count=24, seed=20260911,
        generation_code_revision=src.result['code_revision'], timing_scope=ready_scope,
        **summarize(checked, [p['id'] for p in prompts]), rows=checked,
        common_codec_options=common_codec,
        mean_target_chunks=math.fsum(len(x['target_chunks']) for x in checked) / 24,
        source=dict(run_id=src.marker['run_id'], attempt_id=src.marker['attempt_id'],
            artifact_index_sha256=src.marker['artifact_index_sha256'],
            destination=src.marker['artifact_publication']['destination'],
            transitive_local_references=sorted(getattr(src, 'transitive_references', {}).values(),
                                                key=lambda v: v['path'])),
        caveats=['24 development prompts, one seed, six-dimension mean rather than official full total.',
            'No matched Target ratio or quality delta is established.',
            'Independent Draft proposal history and selected Target history are verified separately.',
            'Fixed scored MP4s are audited; binary-motion robustness is not established.'])

    if target is not None:
        target_quality = math.fsum(math.fsum(row['target_quality_scores'].values()) / 6 * 100
                                   for row in checked) / len(checked)
        target_ready = math.fsum(row['target_ready_seconds'] for row in checked) / len(checked)
        result.update(schema='native_sdvg_matched_report_v1', matched_target_comparison=True, **scheduling,
            target_quality_mean_100=target_quality, target_mean_ready_seconds=target_ready,
            speedup_ratio_of_mean_ready_times=target_ready / result['mean_ready_seconds'],
            quality_delta_vs_target=result['quality_mean_100'] - target_quality,
            target_source=dict(run_id=target.marker['run_id'], attempt_id=target.marker['attempt_id'],
                artifact_index_sha256=target.marker['artifact_index_sha256'],
                destination=target.marker['artifact_publication']['destination'],
                transitive_local_references=sorted(getattr(target, 'transitive_references', {}).values(),
                                                    key=lambda value: value['path'])))
        result['caveats'].remove('No matched Target ratio or quality delta is established.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('queue', 'repo'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--target-queue', type=Path)
    parser.add_argument('--target-task')
    parser.add_argument('--target-commit')
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--seconds', type=int, choices=(15, 30, 60), required=True)
    args = parser.parse_args()
    src = audit.source(args.queue, args.task, args.repo, args.commit,
                       timing_scope=timing_scope(args.model))
    if (args.target_queue or args.target_commit) and not args.target_task:
        parser.error('Target queue or revision requires --target-task')
    target = None
    if args.target_task:
        target = audit.source(args.target_queue or args.queue, args.target_task, args.repo,
                              args.target_commit or args.commit, timing_scope=timing_scope(args.model))
    print(json.dumps(report(src, repo=args.repo, model=args.model, seconds=args.seconds,
                            target=target), sort_keys=True))


if __name__ == '__main__':
    main()
