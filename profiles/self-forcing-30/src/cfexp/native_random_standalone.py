"""Audit one native Random cohort without manufacturing a Target comparison.

The matched reader remains mandatory for a speedup or a quality delta. This
reader verifies the candidate's own scored videos, cache and completed delivery
so its absolute quality and ready time can be reported independently.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import native_ready_report as audit
from .baseline_population import load_baseline_population, model_pair_for_draft
from .correction_policy import random_positions
from .duration_sweep import duration_profile
from .main_table_v1 import QUALITY_DIMENSIONS
from .native_random_report import fixed_history, protocol_id
from .native_reference_report import official_scores


def header(src, repo, *, model, seconds):
    audit._require(model in audit.MODELS and type(seconds) is int and seconds in (15, 30, 60),
                   'unsupported standalone Random condition')
    profile = duration_profile(seconds)
    prompts, population = load_baseline_population(
        repo, model_pair=model_pair_for_draft(audit.MODELS[model]))
    duration = dict(profile.metadata(), experiment_id='erank-baseline24-v1',
                    benchmark='VBench-Long' if seconds == 60 else 'VBench')
    for name in ('calibration_chunk_count', 'calibration_policy'):
        duration.pop(name, None)
    duration['target_replacement_budget'] = duration.pop('ours_proposal_budget')
    result = src.result
    audit._require(result.get('schema') == 'erank_baseline24_result_v1'
        and result.get('method_id') == 'random' and result.get('strategy') == 'native_random_direct'
        and result.get('run_kind') == 'formal_campaign'
        and result.get('draft_model_id') == audit.MODELS[model]
        and result.get('population') == population and result.get('seed_count') == 1
        and result.get('random_repeat') == 0 and result.get('prompt_count') == 24
        and result.get('row_count') == result.get('native_cache_audit_count') == 24
        and result.get('accepted_stream_count') == 24 and result.get('duration') == duration,
        'standalone source is not the frozen formal native Random cohort')
    command = src.manifest['spec']['command']
    required = {'CFEXP_NATIVE_VIDEO_STREAM_MODEL': model, 'CFEXP_ACCEPTED_VIDEO_STREAM': '1',
        'CFEXP_RENDER_TARGET_PRIORITY': '1', 'CFEXP_VIDEO_ONLY_DECODE': '1',
        'CFEXP_VIDEO_ENCODING': 'serial_abr_v1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    audit._require(result.get('timing_scope') == audit.OVERLAP_READY_SCOPE
        and result.get('benchmark') == duration['benchmark']
        and '--overlap-native-vae' in command['argv']
        and all(command['env'].get(k) == v for k, v in required.items()),
        'standalone native renderer or timing scope differs')
    return profile, prompts, duration


def completed_ready(src, row, provenance, pending, video, *, model, profile):
    audit._require(row['timing_scope'] == audit.OVERLAP_READY_SCOPE
        and audit.positive(row['video_ready_wall_seconds'], 'ready') >=
            audit.positive(row['generation_wall_seconds'], 'generation')
        and provenance['timing'] == {k: row[k] for k in
            ('generation_wall_seconds', 'video_ready_wall_seconds', 'timing_scope')},
        'standalone timing identity differs')
    prepared, history = pending.prepared_quality, pending.history
    audit._require(src.run / src.prefix / prepared.video_path == video[0]
        and prepared.video_sha256 == video[1]['sha256']
        and prepared.video_nbytes == video[1]['nbytes'], 'standalone accepted/scored video differs')
    finalized = audit.read_json(src, 'measurements/rollouts/' + pending.route_key + '.json')
    audit._require(finalized['final_history_hash'] == history.head_sha256
        and finalized['target_call_count'] == finalized['budgeted_target_call_count'] == profile.proposal_budget + 1
        and abs(finalized['full_video_quality'] - math.fsum(row['quality_scores'].values()) / 6) < 1e-10,
        'standalone finalized call budget or quality differs')
    binding = dict(path=prepared.decode_evidence_path, sha256=prepared.decode_evidence_sha256)
    decode = audit._load_json(audit.referenced_file(src, binding, 'native decode'), label='native decode')
    audit._require(decode['history_manifest_sha256'] == history.manifest_sha256
        and decode['expected_frames'] == profile.frame_count
        and decode['latent_chunk_count'] == profile.chunk_count
        and decode['video'] == dict(path=prepared.video_path, sha256=prepared.video_sha256,
            nbytes=prepared.video_nbytes, fps=24, frames=profile.frame_count),
        'standalone native decode binding differs')
    audit.verify_accepted_renderer(src, provenance['accepted_stream'], history, decode,
                                   model=model, role='target', profile=profile)


def summarize(rows, prompt_ids):
    audit._require(len(rows) == len(prompt_ids) == 24 and len(set(prompt_ids)) == 24
        and [row['prompt_id'] for row in rows] == list(prompt_ids),
        'standalone Random coverage or order differs')
    for row in rows:
        scores = row['quality_scores']
        audit._require(set(scores) == set(QUALITY_DIMENSIONS)
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                    for v in scores.values()), 'invalid standalone quality dimensions')
        audit._require(audit.positive(row['ready_seconds'], 'ready') >=
                       audit.positive(row['generation_seconds'], 'generation'),
                       'standalone ready time precedes generation')
    return dict(quality_mean_100=math.fsum(math.fsum(r['quality_scores'].values()) / 6
                                         for r in rows) / 24 * 100,
        dimension_means_100={d: math.fsum(r['quality_scores'][d] for r in rows) / 24 * 100
                             for d in QUALITY_DIMENSIONS},
        mean_ready_seconds=math.fsum(r['ready_seconds'] for r in rows) / 24,
        matched_target_comparison=False, speedup_ratio_of_mean_ready_times=None,
        quality_delta_vs_target=None)


def report(src, *, repo, model, seconds):
    profile, prompts, duration = header(src, repo, model=model, seconds=seconds)
    preflight = audit.read_json(src, 'provenance/preflight.json')
    roles = preflight['gpu_uuids']
    audit._require(preflight['git']['sha'] == src.result['code_revision']
        and set(roles) == {'draft', 'target'} and len(set(roles.values())) == 2
        and all(isinstance(v, str) and v for v in roles.values()),
        'standalone source or physical GPU identity differs')
    rows = audit._bound(src.frozen, src.prefix, src.result['raw_rows'], 'Random raw rows')['rows']
    proofs = src.result['vbench_measurements']
    audit._require(len(rows) == len(proofs) == 24
        and [r['prompt_id'] for r in rows] == [p['id'] for p in prompts],
        'standalone raw/scoring prompt coverage differs')
    protocol = protocol_id(src, repo)
    checked, common_codec = [], None
    for prompt, row, proof in zip(prompts, rows, proofs):
        audit._require(row['seed'] == 20260911 and row['method_id'] == 'random'
            and row['duration'] == duration, 'standalone row inputs differ')
        provenance = audit.read_json(src, 'provenance/random/' + prompt['id'] + '.json')
        audit._require(provenance['method_id'] == 'random'
            and provenance['strategy'] == 'native_random_direct'
            and provenance['budget'] == profile.proposal_budget
            and provenance['prompt_id'] == prompt['id'] and provenance['seed'] == 20260911,
            'standalone Random provenance differs')
        positions = random_positions(protocol_id=protocol, prompt_id=prompt['id'],
            generation_seed=20260911, budget=profile.proposal_budget, repeat_id=0,
            chunk_count=profile.chunk_count)
        pending = fixed_history(src, prompt['id'], profile.chunk_count, provenance, positions)
        mid = 'mt-random-' + prompt['id']
        video = audit.quality(src, row, proof, prompt, mid, seconds)
        official_scores(src, dict(row, quality_evidence=proof), prompt, mid, seconds, video[0])
        audit.verify_geometry(video[0], profile)
        audit._require(common_codec is None or common_codec == video[2],
                       'standalone scored encoders differ')
        common_codec = video[2]
        completed_ready(src, row, provenance, pending, video, model=model, profile=profile)
        checked.append(dict(prompt_id=prompt['id'], prompt=prompt['text'],
            quality_scores=row['quality_scores'], ready_seconds=row['video_ready_wall_seconds'],
            generation_seconds=row['generation_wall_seconds'],
            target_chunks=[0, *positions], video_sha256=video[1]['sha256']))
    return dict(schema='native_random_standalone_report_v1', status='verified',
        method_id='random', model=model, nominal_seconds=seconds, prompt_count=24, seed=20260911,
        non_root_target_budget=profile.proposal_budget,
        generation_code_revision=src.result['code_revision'], timing_scope=audit.OVERLAP_READY_SCOPE,
        **summarize(checked, [p['id'] for p in prompts]), common_codec_options=common_codec,
        physical_gpu_roles=roles, rows=checked,
        source=dict(run_id=src.marker['run_id'], attempt_id=src.marker['attempt_id'],
            artifact_index_sha256=src.marker['artifact_index_sha256'],
            destination=src.marker['artifact_publication']['destination'],
            transitive_local_references=sorted(getattr(src, 'transitive_references', {}).values(),
                                                key=lambda v: v['path'])),
        caveats=['24 development prompts and one seed; six-dimension mean, not official full total.',
            'No matched Target is consumed; neither a speedup nor a Target quality delta is established.',
            'Original seals, accepted latents, native cache and completed-render evidence are retained.',
            'This audits fixed scored MP4s, not encoder repeatability or binary-motion robustness.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--model', choices=tuple(audit.MODELS), required=True)
    parser.add_argument('--seconds', type=int, choices=(15, 30, 60), required=True)
    args = parser.parse_args()
    src = audit.source(args.queue, args.task, args.repo, args.commit,
                       timing_scope=audit.OVERLAP_READY_SCOPE)
    print(json.dumps(report(src, repo=args.repo, model=args.model, seconds=args.seconds), sort_keys=True))


if __name__ == '__main__':
    main()
