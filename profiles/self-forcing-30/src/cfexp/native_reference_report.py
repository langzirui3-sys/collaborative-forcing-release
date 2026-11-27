"""Audit pure native Draft against its same-pair video-ready Target reference.

Consumes successful formal runs only. Pure Draft has its own root; equal noise
descriptors are required, equal root latents are deliberately not required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import native_ready_report as audit
from .accepted_stream import TIMING_SCOPE as SF_TIMING_SCOPE
from .baseline_population import load_baseline_population, model_pair_for_draft
from .correction_state import LatentHistory
from .duration_sweep import duration_profile
from .main_table_v1 import QUALITY_DIMENSIONS
from .native_video_ready_draft import SELF_FORCING, validate_cache

MODELS = {**audit.MODELS, 'self-forcing': SELF_FORCING}


def timing_scope(model):
    return SF_TIMING_SCOPE if model == 'self-forcing' else audit.OVERLAP_READY_SCOPE


def trajectory(src, row, *, role, model, profile):
    native = row['native_generation']
    stored = audit.read_json(src, f'provenance/{role}-native/{row["prompt_id"]}.json')
    history = LatentHistory.from_dict(stored['history'])
    audit._require({k: v for k, v in stored.items() if k != 'history'} == native
        and history.prompt_id == native['prompt_id'] == row['prompt_id']
        and history.trajectory_seed == native['seed'] == row['seed'] == 20260911
        and history.next_chunk_index == native['chunk_count'] == profile.chunk_count
        and history.head_sha256 == native['history_head_sha256']
        and history.manifest_sha256 == native['history_manifest_sha256']
        and (role == 'random' or all(s.producer_role == role for s in history.segments))
        and history.segments[0].latent_sha256 == row['root_latent_sha256'],
        'native reference trajectory identity differs')
    if role == 'random':
        from .native_video_ready_random import target_positions
        positions = target_positions(row['prompt_id'], profile)
        audit._require(native['schema'] == 'native_sf_random_trajectory_v1'
            and native['draft_model_id'] == SELF_FORCING
            and native['initialization'] == 'target_root_then_fixed_random_positions'
            and native['routing_repeat'] == 0 and native['target_chunks'] == list(positions)
            and native['target_model_calls'] == len(positions)
            and native['draft_model_calls'] == profile.chunk_count - len(positions),
            'native Random schedule or budget differs')
        return _verify_random_history(src, history, native['native_cache_states'], set(positions))
    if role == 'draft':
        audit._require(native['schema'] == 'native_pure_draft_trajectory_v1'
            and native['draft_model_id'] == MODELS[model]
            and native['initialization'] == 'pure_native_draft_no_target_root'
            and type(native['target_model_calls']) is int and native['target_model_calls'] == 0
            and native['draft_model_calls'] == profile.chunk_count
            and native['history_reconstruction_calls'] == 0,
            'reference is not pure native Draft')
        chunks = native['native_cache_states']
    else:
        audit._require(native['target_model_calls'] == profile.chunk_count
            and native['draft_model_calls'] == 0 and native['target_session_count'] == 1
            and native['history_reconstruction_calls'] == 0
            and native['target_session_policy'] == 'persistent_native_session'
            and native['rgb_frame_count'] == profile.frame_count
            and native['interleaved_decode'] is True and native['accept_reject_gate'] is False,
            'reference is not persistent pure Target')
        chunks = native['timing']['per_chunk']
    audit._require(len(chunks) == profile.chunk_count, 'native reference chunk evidence is incomplete')
    for i, (segment, chunk) in enumerate(zip(history.segments, chunks)):
        audit._require(segment.chunk_index == chunk['chunk_index'] == i
            and segment.latent_sha256 == chunk['latent_sha256'], 'native reference chunk binding differs')
        audit.referenced_file(src, dict(path=segment.latent_path, sha256=segment.latent_sha256,
                                       nbytes=segment.latent_nbytes), 'reference latent')
        if role == 'draft':
            audit._require(chunk['initial_noise_descriptor_sha256'] == segment.initial_noise_descriptor_sha256,
                           'pure Draft noise binding differs')
            key = 'rolling_state' if model == 'rolling' else 'kv_reuse'
            validate_cache({key: chunk['cache']}, model=MODELS[model], index=i)
    return history


def _verify_random_history(src, history, chunks, positions):
    from .krea_chunk_backend import validate_native_correction_state
    from .native_video_ready_random import verify_draft_suffix

    cached_through = 0
    session = None
    audit._require(len(chunks) == history.next_chunk_index, 'incomplete native Random history')
    for index, (segment, chunk) in enumerate(zip(history.segments, chunks)):
        role = 'target' if index in positions else 'draft'
        audit._require(segment.chunk_index == chunk['chunk_index'] == index
            and segment.producer_role == chunk['role'] == role
            and segment.latent_sha256 == chunk['latent_sha256']
            and segment.initial_noise_descriptor_sha256 == chunk['initial_noise_descriptor_sha256'],
            'native Random committed producer or chunk binding differs')
        audit.referenced_file(src, dict(path=segment.latent_path, sha256=segment.latent_sha256,
            nbytes=segment.latent_nbytes), 'Random latent')
        state = chunk['cache']
        if role == 'target':
            validate_native_correction_state(state, chunk_index=index)
            if index == 0:
                session = state['native_session_id']
            audit._require(session == state['native_session_id'] and (not index or not state['session_recreated']),
                           'native Random changed the Target correction session')
        else:
            verify_draft_suffix(state, index=index, target_chunks=positions,
                                cached_through=cached_through)
            cached_through = index + 1
    return history


def official_scores(src, row, prompt, mid, seconds, video, *, short_batches=None):
    """Reparse retained official details, not merely the summary sidecar."""
    proof = row['quality_evidence']
    path = src.frozen.file(src.prefix + proof['path'], label='quality evidence')
    if seconds == 60:
        from .vbench_long import validate_vbench_long_batch_measurement
        value = validate_vbench_long_batch_measurement(path,
            run_dir=src.run / src.prefix, measurement_id=mid, prompt=prompt['text'], video_path=video)
        scores = dict(value.dimension_scores)
    else:
        from .vbench_short import validate_short_result
        side = audit._bound(src.frozen, src.prefix, proof, 'quality evidence')
        config_binding = side['bindings']['config']
        config = audit._bound(src.frozen, src.prefix, config_binding, 'VBench config')
        result_binding = side['bindings']['result']
        audit._bound(src.frozen, src.prefix, result_binding, 'official short result')
        batch = None
        if short_batches is not None:
            key = (src.run, src.prefix, config_binding['path'], config_binding['sha256'],
                result_binding['path'], result_binding['sha256'])
            batch = short_batches.get(key)
        if batch is None:
            root = (src.run / src.prefix / config_binding['path']).parent
            staged = {}
            for item in config['inputs']:
                source = Path(item['source'])
                alias = root / 'videos' / (item['id'] + '.mp4')
                audit._require(source.is_relative_to(src.run / src.prefix)
                    and alias.is_file() and not alias.is_symlink() and alias.samefile(source)
                    and audit._sha256_file(alias) == (item['sha256'], item['nbytes']),
                    'retained short evaluator alias differs')
                staged[item['id']] = alias
            batch = validate_short_result(src.run / src.prefix / result_binding['path'], staged)
            if short_batches is not None:
                short_batches[key] = batch
        scores = batch[mid]
    audit._require(scores == row['quality_scores'], 'row scores differ from official per-video details')


def reference_rows(src, *, repo, model, seconds, role):
    audit._require(role in ('draft', 'target', 'random') and model in MODELS
        and (role != 'random' or model == 'self-forcing')
        and type(seconds) is int and seconds in (15, 30, 60), 'unsupported reference audit')
    profile = duration_profile(seconds)
    scope = timing_scope(model)
    prompts, population = load_baseline_population(repo, model_pair=model_pair_for_draft(MODELS[model]))
    result = src.result
    audit._require(result.get('schema') == f'native_overlap_video_ready_{role}_v1'
        and result.get('population') == 'main24_method_development_not_held_out'
        and result.get('population_binding') == population
        and result.get('row_count') == result.get('accepted_stream_count') == 24
        and result.get('duration') == profile.metadata()
        and result.get('timing_scope') == scope
        and result.get('benchmark') == ('VBench-Long' if seconds == 60 else 'VBench'),
        'native reference population/horizon/timing differs')
    if role == 'draft':
        audit._require(result.get('method_id') == 'draft_only'
            and result.get('draft_model_id') == MODELS[model]
            and result.get('initialization') == 'pure_native_draft_no_target_root'
            and result.get('target_model_calls_per_video') == 0
            and result.get('draft_model_calls_per_video') == profile.chunk_count,
            'result does not declare pure native Draft')
    command = src.manifest['spec']['command']
    expected_module = 'cfexp.native_video_ready_' + ('target' if role == 'random' else role)
    audit._require(('--random-control' in command['argv']) == (role == 'random'),
                   'native reference Random flag differs')
    env = {'CFEXP_ACCEPTED_VIDEO_STREAM': '1',
           'CFEXP_RENDER_TARGET_PRIORITY': '1', 'CFEXP_VIDEO_ONLY_DECODE': '1',
           'CFEXP_VIDEO_ENCODING': 'serial_abr_v1'}
    if model == 'self-forcing':
        env.update(CFEXP_DRAFT_KV_REUSE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        audit._require('CFEXP_NATIVE_VIDEO_STREAM_MODEL' not in command['env'],
                       'Self-Forcing reference selected another native decoder')
    else:
        env.update(CFEXP_NATIVE_VIDEO_STREAM_MODEL=model, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    audit._require(expected_module in command['argv'] and '--overlap-native-vae' in command['argv']
        and all(command['env'].get(k) == v for k, v in env.items()), 'reference invocation differs')
    rows = result['rows']
    audit._require(len(rows) == 24 and [r['prompt_id'] for r in rows] == [p['id'] for p in prompts],
                   'reference prompt coverage/order differs')
    output, short_batches = [], {}
    for row, prompt in zip(rows, prompts):
        audit._require(row['prompt'] == prompt['text'] and row['seed'] == 20260911
            and row['arm'] == role and row['native_decoder_model_id'] == MODELS[model]
            and row['timing_scope'] == scope
            and audit.positive(row['video_ready_wall_seconds'], 'ready') >=
                audit.positive(row['generation_wall_seconds'], 'generation'), 'reference row input/timing differs')
        ready = audit._bound(src.frozen, src.prefix, row['video_ready_evidence'], 'video-ready evidence')
        audit._require(ready.get('schema') == f'native_{role}_video_ready_record_v1'
            and all(ready.get(k) == v for k, v in row.items()
                    if k not in ('video_ready_evidence', 'quality_scores', 'quality_evidence')),
            'scored row differs from pre-scoring video-ready evidence')
        context = audit._bound(src.frozen, src.prefix, ready['generation_context'], 'generation context')
        expected = dict(schema=f'native_{role}_generation_context_v1', hardware=result['hardware'],
            code_revision=result['code_revision'], population=result['population'],
            population_binding=population, benchmark=result['benchmark'], duration=profile.metadata(),
            timing_scope=scope, native_decoder_model_id=MODELS[model])
        audit._require(context == expected, 'generation context differs')
        mid = 'stream-' + role + '-' + prompt['id']
        video, binding, codec = audit.quality(src, row, row['quality_evidence'], prompt, mid, seconds)
        audit._require(binding['sha256'] == row['video']['sha256']
            and binding['nbytes'] == row['video']['nbytes']
            and video == src.run / src.prefix / row['video']['path'], 'reference row/scored MP4 differs')
        official_scores(src, row, prompt, mid, seconds, video, short_batches=short_batches)
        audit.verify_geometry(video, profile)
        history = trajectory(src, row, role=role, model=model, profile=profile)
        decode = audit._load_json(audit.referenced_file(src, row['decode_evidence'], 'native decode'), label='native decode')
        audit._require(decode['history_manifest_sha256'] == history.manifest_sha256
            and decode['expected_frames'] == profile.frame_count
            and decode['latent_chunk_count'] == profile.chunk_count
            and decode['video'] == dict(row['video'], fps=24, frames=profile.frame_count),
            'native reference decode binding differs')
        audit.verify_accepted_renderer(src, row['accepted_stream'], history, decode,
            model=model, role='draft' if role == 'target' else 'target', profile=profile)
        output.append(dict(row=row, history=history, codec=codec))
    return output


def compare(draft, target, *, repo, model, seconds):
    a, b = draft.result, target.result
    scheduling = audit.scheduling_revision(a['code_revision'], target, repo)
    if model == 'self-forcing':
        commands = [src.manifest['spec']['command'] for src in (draft, target)]
        argv = list(commands[0]['argv'])
        argv[argv.index('cfexp.native_video_ready_draft')] = 'cfexp.native_video_ready_target'
        audit._require(commands[1] == {**commands[0], 'argv': argv},
                       'Self-Forcing reference pair changed flags or inputs')
    ga, gb = a['hardware']['gpus'], b['hardware']['gpus']
    audit._require(len(ga) == len(gb) == 2
        and all(x.get('uuid') and x['uuid'] == y.get('uuid') for x, y in zip(ga, gb))
        and ga[0]['uuid'] != ga[1]['uuid'], 'reference physical GPU pair/role differs')
    aa = reference_rows(draft, repo=repo, model=model, seconds=seconds, role='draft')
    bb = reference_rows(target, repo=repo, model=model, seconds=seconds, role='target')
    audit._require(len(aa) == len(bb) == 24, 'reference paired coverage differs')
    pairs, common_codec = [], None
    for x, y in zip(aa, bb):
        audit._require(x['codec'] == y['codec'] and (common_codec is None or common_codec == x['codec']),
                       'actual scored encoders differ')
        common_codec = x['codec']
        audit._require([s.initial_noise_descriptor_sha256 for s in x['history'].segments] ==
                       [s.initial_noise_descriptor_sha256 for s in y['history'].segments],
                       'reference per-chunk initial noises differ')
        u, v = x['row'], y['row']
        audit._require(u['prompt_id'] == v['prompt_id'], 'reference paired prompt order differs')
        pairs.append(dict(prompt_id=u['prompt_id'],
            draft_quality_100=math.fsum(u['quality_scores'].values()) / 6 * 100,
            target_quality_100=math.fsum(v['quality_scores'].values()) / 6 * 100,
            draft_dimensions=u['quality_scores'], target_dimensions=v['quality_scores'],
            draft_ready_seconds=u['video_ready_wall_seconds'], target_ready_seconds=v['video_ready_wall_seconds'],
            draft_video_sha256=u['video']['sha256'], target_video_sha256=v['video']['sha256']))
    mean = lambda key: math.fsum(p[key] for p in pairs) / 24
    return dict(schema='native_pure_draft_matched_report_v1', status='verified', model=model,
        nominal_seconds=seconds, prompt_count=24, seed=20260911,
        generation_code_revision=a['code_revision'], timing_scope=timing_scope(model),
        **scheduling,
        draft_quality_mean_100=mean('draft_quality_100'), target_quality_mean_100=mean('target_quality_100'),
        draft_dimensions_mean_100={d: math.fsum(p['draft_dimensions'][d] for p in pairs) / 24 * 100 for d in QUALITY_DIMENSIONS},
        target_dimensions_mean_100={d: math.fsum(p['target_dimensions'][d] for p in pairs) / 24 * 100 for d in QUALITY_DIMENSIONS},
        draft_mean_ready_seconds=mean('draft_ready_seconds'), target_mean_ready_seconds=mean('target_ready_seconds'),
        speedup_ratio_of_mean_ready_times=mean('target_ready_seconds') / mean('draft_ready_seconds'),
        common_codec_options=common_codec, pairs=pairs,
        sources=[dict(run_id=s.marker['run_id'], attempt_id=s.marker['attempt_id'],
            artifact_index_sha256=s.marker['artifact_index_sha256'],
            destination=s.marker['artifact_publication']['destination'],
            transitive_local_references=sorted(getattr(s, 'transitive_references', {}).values(), key=lambda v: v['path']))
            for s in (draft, target)],
        caveats=['24 development prompts, one seed; six-dimension mean, not official full total.',
            'Pure Draft generates its own root. Equal per-chunk input noise, not equal root latents.',
            'Target worker stays resident for independent native VAE only in Draft arm; model loading and text encoding are outside timing.',
            'Timing includes native adapter observation/witness overhead; not the fastest native CLI or total GPU-time claim.',
            'Audit requires retained local latent/decode/evaluator aliases bound by original sealed records.'])


def compare_random(candidate, target, *, repo, seconds):
    """Audit same-revision Self-Forcing Random and its pure Target control."""
    command = candidate.manifest['spec']['command']
    reference = target.manifest['spec']['command']
    a, b = candidate.result, target.result
    audit._require(command == {**reference, 'argv': reference['argv'] + ['--random-control']}
        and a['code_revision'] == b['code_revision'], 'native Random reference invocation differs')
    gpus = [[x['uuid'] for x in result['hardware']['gpus']] for result in (a, b)]
    audit._require(gpus[0] == gpus[1] and len(set(gpus[0])) == 2,
                   'native Random physical GPU roles differ')
    aa, bb = (reference_rows(src, repo=repo, model='self-forcing', seconds=seconds, role=role)
              for src, role in ((candidate, 'random'), (target, 'target')))
    pairs, codec = [], None
    for x, y in zip(aa, bb, strict=True):
        u, v = x['row'], y['row']
        audit._require(u['prompt_id'] == v['prompt_id'] and x['codec'] == y['codec']
            and (codec is None or x['codec'] == codec)
            and u['root_latent_sha256'] == v['root_latent_sha256']
            and [s.initial_noise_descriptor_sha256 for s in x['history'].segments] ==
                [s.initial_noise_descriptor_sha256 for s in y['history'].segments],
            'native Random root, noise, prompt or actual codec differs')
        codec = x['codec']
        pairs.append(dict(prompt_id=u['prompt_id'],
            random_quality=math.fsum(u['quality_scores'].values())*100/6,
            target_quality=math.fsum(v['quality_scores'].values())*100/6,
            random_ready_seconds=u['video_ready_wall_seconds'],
            target_ready_seconds=v['video_ready_wall_seconds'],
            random_dimensions=u['quality_scores'], target_dimensions=v['quality_scores']))
    def mean(key):
        return math.fsum(row[key] for row in pairs) / len(pairs)
    return dict(schema='native_sf_random_matched_report_v1', status='verified',
        nominal_seconds=seconds, prompt_count=len(pairs), generation_code_revision=a['code_revision'],
        timing_scope=SF_TIMING_SCOPE, seed=20260911, routing_repeat=0,
        common_codec_options=codec, physical_gpu_uuids=gpus[0],
        random_quality_mean_100=mean('random_quality'), target_quality_mean_100=mean('target_quality'),
        random_mean_ready_seconds=mean('random_ready_seconds'), target_mean_ready_seconds=mean('target_ready_seconds'),
        speedup_ratio_of_mean_ready_times=mean('target_ready_seconds')/mean('random_ready_seconds'),
        pairs=pairs, sources=[{k: src.marker[k] for k in ('run_id', 'attempt_id', 'artifact_index_sha256')}
                             for src in (candidate, target)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--draft-task', required=True)
    parser.add_argument('--target-task', required=True)
    parser.add_argument('--target-queue', type=Path)
    parser.add_argument('--target-commit')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--seconds', type=int, choices=(15, 30, 60), required=True)
    args = parser.parse_args()
    draft = audit.source(args.queue, args.draft_task, args.repo, args.commit,
                         timing_scope=timing_scope(args.model))
    target = audit.source(args.target_queue or args.queue, args.target_task, args.repo,
                          args.target_commit or args.commit, timing_scope=timing_scope(args.model))
    print(json.dumps(compare(draft, target, repo=args.repo,
                             model=args.model, seconds=args.seconds), sort_keys=True))


if __name__ == '__main__':
    main()
