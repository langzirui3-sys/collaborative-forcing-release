"""Audit original128 x three seeds on one H200; no legacy24 relabeling.

Input is a JSON list of formal run directories. Successful local seals are
accepted independently of dataset publication. All files retain their original
seals. Missing, duplicate or mismatched cases prevent the paired aggregate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import native_ready_report as audit
from . import native_reference_report as legacy
from . import paper_population as population
from .accepted_stream import SINGLE_GPU_TIMING_SCOPE
from .baseline_population import model_pair_for_draft
from .baseline_shard_report import source_manifest
from .correction_state import LatentHistory
from .duration_sweep import duration_profile
from .main_table_merge import _load_frozen_index, _load_json, _require
from .main_table_v1 import QUALITY_DIMENSIONS
from .native_single_gpu import TIMING_SCOPE as NATIVE_SCOPE
from .native_video_ready_draft import validate_cache

MATRIX = 'experiments/paper128-single-gpu-reference-matrix-v1.protocol.json'
MATRIX_SHA = '0176e1cba3c97a478003f5f0f5d248abf5de9e004f3ffe3afc0079fa532238aa'
MODELS = dict(legacy.MODELS)


def scope(model):
    return SINGLE_GPU_TIMING_SCOPE if model == 'self-forcing' else NATIVE_SCOPE


def matrix(repo):
    path = Path(repo)/MATRIX
    _require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == MATRIX_SHA,
             'paper reference matrix differs from its frozen definition')
    return json.loads(path.read_text())


def source(run, *, repo, model, seconds):
    requested = Path(run)
    _require(not requested.is_symlink(), 'reference run is a symlink')
    run = requested.resolve(strict=True)
    frozen, marker = _load_frozen_index(run)
    terminal = _load_json(frozen.file('run.json', label='terminal run'), label='terminal run')
    _require(terminal.get('execution_status') == 'SUCCEEDED' and terminal.get('exit_code') == 0
        and terminal.get('latest_attempt_id') == marker['attempt_id']
        and terminal.get('run_id') == marker['run_id'] == run.name,
        'reference producer has not sealed successful execution')
    envelope = _load_json(frozen.file('resolved-manifest.json', label='resolved manifest'), label='manifest')
    invocation = envelope['manifest']
    relative = 'experiments/'+invocation['metadata']['name']+'.manifest.json'
    selected = [x for x in matrix(repo)['runs'] if x['manifest'] == relative]
    _require(len(selected) == 1, 'run is outside the frozen reference matrix')
    definition = selected[0]
    _require(definition['model_pair'] == model_pair_for_draft(MODELS[model])
        and definition['nominal_seconds'] == seconds
        and all(envelope.get(k) == marker[k] for k in ('run_id', 'attempt_id', 'spec_hash'))
        and invocation == source_manifest(repo, marker['git_sha'], relative),
        'reference invocation/source/model/horizon differs')
    prefix = 'attempts/'+marker['attempt_id']+'/'
    result = _load_json(frozen.file(prefix+'result.json', label='result'), label='result')
    _require(result.get('status') == 'succeeded' and result.get('code_revision') == marker['git_sha'],
             'reference result revision or execution status differs')
    src = SimpleNamespace(run=run, prefix=prefix, frozen=frozen, marker=marker,
                          result=result, manifest=invocation, definition=definition)
    return src


def selection(src, *, repo, model, seconds):
    definition, result = src.definition, src.result
    command = src.manifest['spec']['command']
    env, argv = command['env'], command['argv']
    _require(src.manifest['spec']['resources']['gpus'] == 1
        and env.get(population.ENV) == '1'
        and env.get(population.SEED_ENV) == str(definition['generation_seed'])
        and env.get('CFEXP_PAPER_PROMPT_SHARD_COUNT') == '4'
        and env.get('CFEXP_PAPER_PROMPT_SHARD_INDEX') == str(definition['logical_matched_gpu_slot'])
        and '--smoke-prompt-id' not in argv
        and argv[argv.index('--duration-seconds')+1] == str(seconds),
        'reference is not its full one-card paper shard')
    # This read-only command is serialized. Load the existing source validator
    # under the sealed invocation environment, never the observer's settings.
    with patch.dict(os.environ, env, clear=True):
        prompts, binding = population.load(repo, model_pair=model_pair_for_draft(MODELS[model]))
    _require([p['id'] for p in prompts] == definition['prompt_ids']
        and len(prompts) == 32 and result.get('population_binding') == binding,
        'reference paper population or actual generation seed differs')
    _require(result.get('population') == 'paper128_three_seed_replication'
        and result.get('duration') == duration_profile(seconds).metadata()
        and result.get('timing_scope') == scope(model)
        and result.get('benchmark') == 'VBench'
        and result.get('row_count') == 32 and len(result.get('rows', [])) == 32
        and result.get('accepted_video_schedule') == 'decode_after_generation',
        'reference population, timing or deferred native rendering differs')
    gpus = result['hardware']['gpus']
    _require(len(gpus) == 1 and gpus[0].get('uuid', '').startswith('GPU-')
        and 'H200' in gpus[0].get('name', ''), 'reference is not an actual one-H200 allocation')
    role = 'draft' if definition['method_id'] == 'draft_only' else 'target'
    _require(result['schema'] == (f'accepted_stream_{role}_reference_v1'
        if model == 'self-forcing' else f'native_overlap_video_ready_{role}_v1'),
        'reference producer schema differs')
    if model == 'self-forcing':
        _require('cfexp.accepted_stream_target' in argv
            and ('--pure-draft' in argv) == (role == 'draft')
            and env.get('CFEXP_SF_SINGLE_GPU') == env.get('CFEXP_DRAFT_KV_REUSE') == '1'
            and 'CFEXP_NATIVE_VIDEO_STREAM_MODEL' not in env,
            'Self-Forcing reference runtime differs')
    else:
        _require('cfexp.native_video_ready_'+role in argv and '--overlap-native-vae' in argv
            and env.get('CFEXP_NATIVE_SINGLE_GPU') == '1'
            and env.get('CFEXP_NATIVE_VIDEO_STREAM_MODEL') == model
            and result.get('accepted_stream_count') == 32,
            'native reference runtime differs')
    _require(env.get('CFEXP_ACCEPTED_VIDEO_STREAM') == env.get('CFEXP_RENDER_TARGET_PRIORITY') == '1'
        and env.get('CFEXP_VIDEO_ONLY_DECODE') == '1'
        and env.get('CFEXP_VIDEO_ENCODING') == 'serial_abr_v1', 'reference renderer/codec policy differs')
    return prompts, binding, role


def trajectory(src, row, *, model, role, seed, profile):
    stored = audit.read_json(src, f'provenance/{role}-native/{row["prompt_id"]}.json')
    native = row['native_generation']
    history = LatentHistory.from_dict(stored['history'])
    _require({k:v for k,v in stored.items() if k != 'history'} == native
        and native['prompt_id'] == history.prompt_id == row['prompt_id']
        and history.trajectory_seed == native['seed'] == row['seed'] == seed
        and history.next_chunk_index == native['chunk_count'] == profile.chunk_count
        and history.head_sha256 == native['history_head_sha256']
        and history.manifest_sha256 == native['history_manifest_sha256']
        and history.segments[0].latent_sha256 == row['root_latent_sha256']
        and all(x.producer_role == role for x in history.segments),
        'reference history, actual seed or producer differs')
    if role == 'draft':
        _require(native['schema'] == 'native_pure_draft_trajectory_v1'
            and native['draft_model_id'] == MODELS[model]
            and native['initialization'] == 'pure_native_draft_no_target_root'
            and native['target_model_calls'] == native['history_reconstruction_calls'] == 0
            and native['draft_model_calls'] == profile.chunk_count, 'not pure native Draft')
        chunks = native['native_cache_states']
    else:
        _require(native['schema'] == 'main_table_full_target_rollout_v1'
            and native['method_id'] == 'full_target'
            and native['trajectory_id'] == history.trajectory_id
            and native['rgb_frame_count'] == profile.frame_count
            and (native['fps'],native['height'],native['width']) == (24,480,832)
            and native['target_model_id'] == 'krea-realtime-video-14b'
            and native['target_model_calls'] == profile.chunk_count
            and native['draft_model_calls'] == native['history_reconstruction_calls'] == 0
            and native['target_session_count'] == 1
            and native['target_session_policy'] == 'persistent_native_session'
            and native['interleaved_decode'] is True and native['accept_reject_gate'] is False,
            'not persistent pure Target')
        chunks = native['timing']['per_chunk']
    _require(len(chunks) == profile.chunk_count, 'reference cache/chunk evidence is incomplete')
    for index, (segment, chunk) in enumerate(zip(history.segments, chunks)):
        _require(segment.chunk_index == chunk['chunk_index'] == index
            and segment.latent_sha256 == chunk['latent_sha256'], 'reference chunk binding differs')
        audit.referenced_file(src, dict(path=segment.latent_path, sha256=segment.latent_sha256,
                                       nbytes=segment.latent_nbytes), 'paper reference latent')
        if role == 'draft':
            _require(chunk['initial_noise_descriptor_sha256'] == segment.initial_noise_descriptor_sha256,
                     'reference Draft noise differs')
            key = 'rolling_state' if model == 'rolling' else 'kv_reuse'
            validate_cache({key:chunk['cache']}, model=MODELS[model], index=index)
    return history


def verify_renderer(src, binding, history, decode, *, model, role, profile):
    # SF's reference producer stores the complete receipt inline as well as in
    # the indexed provenance file. Native reference producers store a file
    # binding instead. Verify both SF copies before using the shared validator.
    if model == 'self-forcing':
        relative = 'provenance/accepted-stream/' + history.manifest_sha256 + '.json'
        path = src.frozen.file(src.prefix + relative, label='SF accepted renderer receipt')
        stored = _load_json(path, label='SF accepted renderer receipt')
        _require(binding == stored, 'inline SF accepted renderer receipt differs from indexed provenance')
        digest, size = audit._sha256_file(path)
        binding = dict(path=relative, sha256=digest, nbytes=size)
    audit.verify_accepted_renderer(src, binding, history, decode,
        model=model, role=role, profile=profile)


def verified_rows(src, *, repo, model, seconds):
    prompts, binding, role = selection(src, repo=repo, model=model, seconds=seconds)
    profile = duration_profile(seconds)
    output, batches = [], {}
    for row, prompt in zip(src.result['rows'], prompts):
        _require(row['prompt_id'] == prompt['id'] and row['prompt'] == prompt['text']
            and row['seed'] == binding['seed'] and row['arm'] == role
            and row['timing_scope'] == scope(model)
            and row.get('accepted_video_schedule') == 'decode_after_generation'
            and audit.positive(row['video_ready_wall_seconds'], 'ready') >=
                audit.positive(row['generation_wall_seconds'], 'generation'), 'reference row input/timing differs')
        if model != 'self-forcing':
            _require(row['native_decoder_model_id'] == MODELS[model], 'native decoder model differs')
            ready = audit._bound(src.frozen, src.prefix, row['video_ready_evidence'], 'video-ready evidence')
            _require(ready.get('schema') == f'native_{role}_video_ready_record_v1'
                and all(ready.get(k) == v for k,v in row.items()
                    if k not in ('quality_scores','quality_evidence','video_ready_evidence')),
                'reference differs from completed pre-scoring evidence')
            context = audit._bound(src.frozen, src.prefix, ready['generation_context'], 'generation context')
            _require(context == dict(schema=f'native_{role}_generation_context_v1',
                hardware=src.result['hardware'], code_revision=src.result['code_revision'],
                population=src.result['population'], population_binding=binding,
                benchmark='VBench', duration=profile.metadata(), timing_scope=scope(model),
                native_decoder_model_id=MODELS[model]), 'reference generation context differs')
        mid = 'stream-'+role+'-'+prompt['id']
        video, actual, codec = audit.quality(src, row, row['quality_evidence'], prompt, mid, seconds)
        _require(actual['sha256'] == row['video']['sha256'] and actual['nbytes'] == row['video']['nbytes']
            and video == src.run/src.prefix/row['video']['path'], 'scored and timed MP4 differ')
        legacy.official_scores(src,row,prompt,mid,seconds,video,short_batches=batches)
        audit.verify_geometry(video,profile)
        history = trajectory(src,row,model=model,role=role,seed=binding['seed'],profile=profile)
        decode = _load_json(audit.referenced_file(src,row['decode_evidence'],'native decode'),label='native decode')
        _require(decode['history_manifest_sha256'] == history.manifest_sha256
            and decode['expected_frames'] == profile.frame_count
            and decode['latent_chunk_count'] == profile.chunk_count
            and decode['video'] == dict(row['video'],fps=24,frames=profile.frame_count),
            'reference decode binding differs')
        verify_renderer(src,row['accepted_stream'],history,decode,
            model=model,role='draft' if role == 'target' else 'target',profile=profile)
        output.append(dict(row=row, history=history, codec=codec))
    return output


def compare(sources, *, repo, model, seconds):
    _require(model in MODELS and type(seconds) is int and seconds in (15,30), 'unsupported paper reference')
    expected = [r for r in matrix(repo)['runs'] if r['model_pair'] == model_pair_for_draft(MODELS[model])
                and r['nominal_seconds'] == seconds]
    _require(len(sources) == len(expected) == 24
        and len({s.run for s in sources}) == 24
        and {s.definition['id'] for s in sources} == {r['id'] for r in expected},
        'paper reference needs exactly all24 unique source shards')
    cases = {'draft':{}, 'target':{}}
    for src in sources:
        role = 'draft' if src.definition['method_id'] == 'draft_only' else 'target'
        for item in verified_rows(src,repo=repo,model=model,seconds=seconds):
            row = item['row'];key = row['prompt_id'],row['seed']
            _require(key not in cases[role], 'duplicate paper prompt/seed')
            cases[role][key] = item,src
    protocol = json.loads((Path(repo)/population.PROTOCOL).read_text())
    keys = [(p['id'],seed) for p in protocol['prompts'] for seed in population.SEEDS]
    _require(len(keys) == 384 and set(cases['draft']) == set(cases['target']) == set(keys),
             'paper reference384-case coverage differs')
    pairs = []
    for key in keys:
        a,sa = cases['draft'][key];b,sb = cases['target'][key]
        _require(sa.marker['git_sha'] == sb.marker['git_sha']
            and sa.result['hardware'] == sb.result['hardware']
            and a['codec'] == b['codec']
            and [x.initial_noise_descriptor_sha256 for x in a['history'].segments] ==
                [x.initial_noise_descriptor_sha256 for x in b['history'].segments],
            'paired paper source, physical GPU, codec or noise differs')
        pairs.append(dict(prompt_id=key[0],seed=key[1],
            draft_quality_100=math.fsum(a['row']['quality_scores'].values())*100/6,
            target_quality_100=math.fsum(b['row']['quality_scores'].values())*100/6,
            draft_ready_seconds=a['row']['video_ready_wall_seconds'],
            target_ready_seconds=b['row']['video_ready_wall_seconds'],
            gpu_uuid=sa.result['hardware']['gpus'][0]['uuid']))
    mean = lambda field:math.fsum(r[field] for r in pairs)/384
    return dict(schema='paper128_single_h200_reference_report_v1',status='verified',
        model=model,nominal_seconds=seconds,prompt_count=128,seeds=list(population.SEEDS),
        case_count=384,actual_gpus_per_video=1,timing_scope=scope(model),
        draft_quality_mean_100=mean('draft_quality_100'),target_quality_mean_100=mean('target_quality_100'),
        draft_mean_ready_seconds=mean('draft_ready_seconds'),target_mean_ready_seconds=mean('target_ready_seconds'),
        speedup_ratio_of_mean_ready_times=mean('target_ready_seconds')/mean('draft_ready_seconds'),
        pairs=pairs,sources=[{k:s.marker[k] for k in ('run_id','attempt_id','git_sha','artifact_index_sha256')}
                            for s in sources],dataset_publication_verified=False,
        caveats=['Six-dimension macro mean, not the official full VBench total.',
            'Local successful seals are verified; dataset upload is a separate status.',
            'Pure Draft generates its own root; per-chunk initial noise is matched.',
            'SF stores timing in the sealed scored result; native models also retain pre-scoring ready records.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--model',choices=tuple(MODELS),required=True)
    parser.add_argument('--seconds',type=int,choices=(15,30),required=True)
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text())
    _require(isinstance(runs,list) and len(runs) == 24 and all(isinstance(x,str) for x in runs),
             'input must list exactly24 formal run directories')
    sources = [source(x,repo=args.repo,model=args.model,seconds=args.seconds) for x in runs]
    print(json.dumps(compare(sources,repo=args.repo,model=args.model,seconds=args.seconds),allow_nan=False))


if __name__ == '__main__':
    main()
