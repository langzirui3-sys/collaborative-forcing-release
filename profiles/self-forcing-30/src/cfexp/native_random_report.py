"""Read-only matched native Random / Target audit for the main table."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

from . import native_ready_report as audit
from .baseline_population import load_baseline_population, model_pair_for_draft
from .baseline_shard_report import source_manifest
from .duration_sweep import duration_profile
from .insight2_campaign import PendingSequentialRollout
from .main_table_v1 import QUALITY_DIMENSIONS
from .native_reference_report import official_scores, trajectory


def fixed_history(src, prompt_id, count, provenance, positions):
    audit._require(isinstance(positions,(list,tuple)) and list(positions) == sorted(set(positions))
        and all(type(i) is int and 1 <= i < count for i in positions), 'invalid fixed Random positions')
    matches = []
    for name in src.frozen.entries:
        if name.startswith(src.prefix+'measurements/rollouts/') and name.endswith('.pending.json'):
            item = audit.read_json(src, name[len(src.prefix):])
            if item['history']['prompt_id'] == prompt_id:
                matches.append(PendingSequentialRollout.from_dict(item))
    audit._require(len(matches) == 1, 'Random pending history is missing or ambiguous')
    pending = matches[0]
    history = pending.history
    selected = list(positions)
    audit._require(history.next_chunk_index == count and history.trajectory_seed == 20260911
        and history.head_sha256 == provenance['final_history_hash']
        and pending.corrected_chunks == tuple(selected)
        and provenance['accepted_target_chunks'] == provenance['proposed_target_chunks'] == selected,
        'Random accepted positions/history differ')
    targets = {0, *selected}
    proof = provenance['native_cache_audit']
    model = src.result['draft_model_id']
    audit._require(proof['schema'] == 'native_fixed_route_cache_audit_v1'
        and proof['draft_model_id'] == model and proof['target_chunks'] == sorted(targets)
        and proof['decisions'] == [] and isinstance(proof['target_session_id'], str)
        and bool(proof['target_session_id'])
        and not any(proof.get(k, False) for k in ('opening_target_burst','split_target_burst','longlive_prefill')),
        'Random native cache witness identity differs')
    expected = [i for i in range(count) if i not in targets]
    audit._require([s['chunk_index'] for s in proof['draft_states']] == expected,
                   'Random native cache witness coverage differs')
    previous = None
    for state, i in zip(proof['draft_states'], expected):
        missing = i if previous is None else i-previous-1
        if model == 'longlive_1.3b':
            audit._require(type(state['replayed_chunks']) is int and state['replayed_chunks'] == missing
                and type(state['reused_chunks']) is int and state['reused_chunks'] == i-missing,
                'Random LongLive replay/reuse differs')
        elif model == 'rolling_forcing_1.3b':
            reused = previous is not None and missing == 0
            audit._require(state['state_policy'] == 'single_active_branch_clean_history_restart_v1'
                and state['reused_rolling_state'] is reused
                and type(state['replayed_history_blocks']) is int
                and state['replayed_history_blocks'] == (0 if reused else i)
                and state['discarded_future_state'] is (previous is not None and not reused),
                'Random Rolling continuation/restart differs')
        else:
            raise ValueError('unsupported native Random model')
        previous = i
    for i, segment in enumerate(history.segments):
        audit._require(segment.producer_role == ('target' if i in targets else 'draft'),
                       'Random accepted producer differs')
        audit.referenced_file(src, dict(path=segment.latent_path, sha256=segment.latent_sha256,
                                        nbytes=segment.latent_nbytes), 'Random accepted latent')
    return pending


def protocol_id(src, repo):
    from .insight2_run import load_run_config
    from .insight2_protocol import load_protocol
    repo = repo.resolve(strict=True)
    command = src.manifest['spec']['command']
    argv = command['argv']
    audit._require(argv[argv.index('--selection-strategy')+1] == 'random'
        and argv[argv.index('--random-repeat')+1] == '0'
        and '--native-model-burst' not in argv, 'Random invocation changed selection/repeat')
    config_path = (repo / argv[argv.index('--boundary-config')+1]).resolve(strict=True)
    audit._require(config_path.is_relative_to(repo), 'Random configuration escaped the repository')
    config = load_run_config(config_path, repo_root=repo)
    for path in (config_path, config.protocol_path):
        relative = path.relative_to(repo).as_posix()
        audit._require(any(x['target'] == relative and x['sha256'] == audit._sha256_file(path)[0]
            for x in src.manifest['spec']['inputs']), 'Random configuration is not source-bound')
    return load_protocol(config.protocol_path).raw['protocol_id']


ROLLING_PLACEMENT_SOURCE = 'afd5e35507c1bf4556b0d41d129c1f35a33e8365'
ROLLING_PLACEMENT_RANDOM_MANIFEST = 'experiments/rolling-p8-k2-placement-random-15s-24p-ntu-v1.manifest.json'


def _scheduling_revision(candidate, target, repo):
    candidate_revision = candidate.result['code_revision']
    target_revision = target.result['code_revision']
    manifest_path = getattr(candidate, 'task', {}).get('manifest')
    if (candidate_revision == target_revision or target_revision != ROLLING_PLACEMENT_SOURCE
            or manifest_path != ROLLING_PLACEMENT_RANDOM_MANIFEST
            or candidate.result.get('strategy') != 'native_random_p8_k2'):
        return audit.scheduling_revision(candidate_revision, target, repo)

    audit._require(re.fullmatch(r'[0-9a-f]{40}', candidate_revision) is not None,
                   'full Random recovery revision required')
    env = dict(PATH='/usr/bin:/bin', HOME=os.path.expanduser('~'),
               GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_SYSTEM='/dev/null',
               GIT_CONFIG_NOSYSTEM='1', GIT_NO_REPLACE_OBJECTS='1')

    def git(*args):
        return subprocess.check_output(['git', '--no-replace-objects', '-C', str(repo), *args],
                                       env=env, stderr=subprocess.PIPE, text=True).strip()

    audit._require(git('rev-list', '--parents', '-n', '1', candidate_revision).split()
                   == [candidate_revision, target_revision],
                   'Random retry is not a direct child of its matched Target revision')
    changes = git('diff-tree', '-r', '--no-commit-id', '--no-renames', '--raw',
                  target_revision, candidate_revision).splitlines()
    audit._require(len(changes) == 1, 'Random retry must change only its own manifest')
    fields, path = changes[0].split('\t', 1)
    mode_before, mode_after, _, _, status = fields.split()
    audit._require(path == manifest_path and status == 'M'
                   and mode_before == ':100644' and mode_after == '100644',
                   'Random retry must change only its own manifest')
    original = source_manifest(repo, target_revision, manifest_path)
    retry = source_manifest(repo, candidate_revision, manifest_path)
    old_minutes = original['spec']['resources']['expected_duration_minutes']
    new_minutes = retry['spec']['resources']['expected_duration_minutes']
    expected = {**original, 'spec': {**original['spec'], 'resources': {
        **original['spec']['resources'], 'expected_duration_minutes': new_minutes}}}
    audit._require(type(new_minutes) is int and 0 < new_minutes < old_minutes
                   and candidate.manifest == retry == expected,
                   'Random retry changed generation configuration or invalid PBS duration')
    return {'target_generation_code_revision': target_revision,
            'verified_candidate_scheduling_changes': {
                manifest_path: {'original_minutes': old_minutes, 'retry_minutes': new_minutes}}}


def compare(candidate, target, *, repo, model, seconds):
    from .correction_policy import random_p8_k2_positions, random_positions

    audit._require(model in audit.MODELS and type(seconds) is int and seconds in (15,30,60),
                   'unsupported native Random condition')
    profile = duration_profile(seconds)
    prompts, population = load_baseline_population(repo, model_pair=model_pair_for_draft(audit.MODELS[model]))
    a, b = candidate.result, target.result
    scheduling = _scheduling_revision(candidate, target, repo)
    p8_k2 = a.get('strategy') == 'native_random_p8_k2'
    strategy = 'native_random_p8_k2' if p8_k2 else 'native_random_direct'
    audit._require(not p8_k2 or (model == 'rolling' and seconds == 15),
                   'Random P8/K2 requires 15-second Rolling Forcing')
    random_duration = dict(profile.metadata(), experiment_id='erank-baseline24-v1',
                           benchmark='VBench-Long' if seconds == 60 else 'VBench')
    for name in ('calibration_chunk_count','calibration_policy'):
        random_duration.pop(name, None)
    random_duration['target_replacement_budget'] = random_duration.pop('ours_proposal_budget')
    if p8_k2:
        random_duration['random_burst_observation_horizon'] = 8
    budget = profile.proposal_budget
    audit._require(a.get('schema') == 'erank_baseline24_result_v1' and a.get('method_id') == 'random'
        and a.get('strategy') == strategy and a.get('run_kind') == 'formal_campaign'
        and a.get('draft_model_id') == audit.MODELS[model] and a.get('population') == population
        and a.get('seed_count') == 1 and a.get('random_repeat') == 0
        and a.get('prompt_count') == a.get('row_count') == a.get('native_cache_audit_count') == 24
        and a.get('accepted_stream_count') == 24 and a.get('duration') == random_duration,
        'candidate is not the frozen formal native Random cohort')
    audit._require(b.get('schema') == 'native_overlap_video_ready_target_v1'
        and b.get('population') == 'main24_method_development_not_held_out'
        and b.get('population_binding') == population and b.get('row_count') == b.get('accepted_stream_count') == 24
        and b.get('duration') == profile.metadata(), 'Target reference population/horizon differs')
    required = {'CFEXP_NATIVE_VIDEO_STREAM_MODEL':model, 'CFEXP_ACCEPTED_VIDEO_STREAM':'1',
        'CFEXP_RENDER_TARGET_PRIORITY':'1','CFEXP_VIDEO_ONLY_DECODE':'1','CFEXP_VIDEO_ENCODING':'serial_abr_v1',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    random_argv = candidate.manifest['spec']['command']['argv']
    audit._require(random_argv.count('--native-random-p8-k2') == int(p8_k2),
                   'Random schedule differs from the published invocation')
    for src in (candidate,target):
        command = src.manifest['spec']['command']
        audit._require(src.result['timing_scope'] == audit.OVERLAP_READY_SCOPE
            and src.result['benchmark'] == random_duration['benchmark']
            and '--overlap-native-vae' in command['argv']
            and all(command['env'].get(k) == v for k,v in required.items()), 'native Random ready/renderer scope differs')
    preflight = audit.read_json(candidate, 'provenance/preflight.json')
    audit._require(len(b['hardware']['gpus']) == 2 and preflight['git']['sha'] == a['code_revision']
        and preflight['gpu_uuids'] == dict(zip(('draft','target'), (x['uuid'] for x in b['hardware']['gpus'])))
        and len(set(preflight['gpu_uuids'].values())) == 2, 'Random/Target physical GPU roles differ')
    protocol = protocol_id(candidate, repo)
    rows = audit._bound(candidate.frozen, candidate.prefix, a['raw_rows'], 'Random raw rows')['rows']
    audit._require(len(rows) == len(b['rows']) == len(a['vbench_measurements']) == 24
        and [x['prompt_id'] for x in rows] == [x['prompt_id'] for x in b['rows']] == [p['id'] for p in prompts],
        'Random/Target prompt coverage/order differs')
    pairs, common_codec = [], None
    for prompt, x, y, proof in zip(prompts, rows, b['rows'], a['vbench_measurements']):
        audit._require(x['seed'] == y['seed'] == 20260911 and x['method_id'] == 'random'
            and x['duration'] == random_duration and y['prompt'] == prompt['text']
            and y['native_decoder_model_id'] == audit.MODELS[model], 'Random row input/backend differs')
        for row in (x,y):
            audit._require(row['timing_scope'] == audit.OVERLAP_READY_SCOPE
                and audit.positive(row['video_ready_wall_seconds'],'ready') >=
                    audit.positive(row['generation_wall_seconds'],'generation'), 'Random row timing differs')
        provenance = audit.read_json(candidate, 'provenance/random/'+prompt['id']+'.json')
        audit._require(provenance['method_id'] == 'random' and provenance['strategy'] == strategy
            and provenance['budget'] == budget and provenance['prompt_id'] == prompt['id']
            and provenance['seed'] == 20260911 and provenance['timing'] ==
                {k:x[k] for k in ('generation_wall_seconds','video_ready_wall_seconds','timing_scope')},
            'Random provenance timing/identity differs')
        random_input = dict(protocol_id=protocol, prompt_id=prompt['id'], generation_seed=20260911, repeat_id=0)
        if p8_k2:
            positions = random_p8_k2_positions(**random_input)
        else:
            positions = random_positions(**random_input, budget=budget, chunk_count=profile.chunk_count)
        pending = fixed_history(candidate,prompt['id'],profile.chunk_count,provenance,positions)
        history = pending.history
        qx = audit.quality(candidate,x,proof,prompt,'mt-random-'+prompt['id'],seconds)
        qy = audit.quality(target,y,y['quality_evidence'],prompt,'stream-target-'+prompt['id'],seconds)
        for src, row, mid, q in ((candidate,dict(x,quality_evidence=proof),'mt-random-'+prompt['id'],qx),
                                 (target,y,'stream-target-'+prompt['id'],qy)):
            official_scores(src,row,prompt,mid,seconds,q[0]);audit.verify_geometry(q[0],profile)
        audit._require(qx[2] == qy[2] and (common_codec is None or common_codec == qx[2]), 'scored encoders differ')
        common_codec = qx[2]
        prepared = pending.prepared_quality
        audit._require(prepared.video_sha256 == qx[1]['sha256'] and prepared.video_nbytes == qx[1]['nbytes']
            and candidate.run/candidate.prefix/prepared.video_path == qx[0], 'Random accepted/scored video differs')
        finalized = audit.read_json(candidate,'measurements/rollouts/'+pending.route_key+'.json')
        audit._require(finalized['final_history_hash'] == history.head_sha256
            and finalized['target_call_count'] == finalized['budgeted_target_call_count'] == budget+1
            and abs(finalized['full_video_quality']-math.fsum(x['quality_scores'].values())/6)<1e-10,
            'Random finalized call budget/quality differs')
        th = trajectory(target,y,role='target',model=model,profile=profile)
        audit._require(history.segments[0].latent_sha256 == th.segments[0].latent_sha256
            and [s.initial_noise_descriptor_sha256 for s in history.segments] ==
                [s.initial_noise_descriptor_sha256 for s in th.segments], 'Random/Target root or initial noises differ')
        db = dict(path=prepared.decode_evidence_path,sha256=prepared.decode_evidence_sha256)
        for src, h, binding, role, video, receipt in ((candidate,history,db,'target',dict(path=prepared.video_path,
                sha256=prepared.video_sha256,nbytes=prepared.video_nbytes),provenance['accepted_stream']),
                (target,th,y['decode_evidence'],'draft',y['video'],y['accepted_stream'])):
            decode = audit._load_json(audit.referenced_file(src,binding,'native decode'),label='native decode')
            audit._require(decode['history_manifest_sha256'] == h.manifest_sha256
                and decode['expected_frames'] == profile.frame_count and decode['latent_chunk_count'] == profile.chunk_count
                and decode['video'] == dict(video,fps=24,frames=profile.frame_count), 'native decode binding differs')
            audit.verify_accepted_renderer(src,receipt,h,decode,model=model,role=role,profile=profile)
        pairs.append(dict(prompt_id=prompt['id'],random_quality_100=math.fsum(x['quality_scores'].values())/6*100,
            target_quality_100=math.fsum(y['quality_scores'].values())/6*100,
            random_dimensions=x['quality_scores'],target_dimensions=y['quality_scores'],
            random_ready_seconds=x['video_ready_wall_seconds'],target_ready_seconds=y['video_ready_wall_seconds'],
            random_target_chunks=[0,*positions],random_video_sha256=qx[1]['sha256'],target_video_sha256=qy[1]['sha256']))
    mean = lambda key:math.fsum(p[key] for p in pairs)/24
    return dict(schema='native_random_matched_report_v1',status='verified',method_id='random',model=model,
        nominal_seconds=seconds,prompt_count=24,seed=20260911,non_root_target_budget=budget,
        generation_code_revision=a['code_revision'],timing_scope=audit.OVERLAP_READY_SCOPE,
        **({'random_burst_observation_horizon': 8} if p8_k2 else {}),
        **scheduling,
        random_quality_mean_100=mean('random_quality_100'),target_quality_mean_100=mean('target_quality_100'),
        random_mean_ready_seconds=mean('random_ready_seconds'),target_mean_ready_seconds=mean('target_ready_seconds'),
        speedup_ratio_of_mean_ready_times=mean('target_ready_seconds')/mean('random_ready_seconds'),
        dimension_means_100={arm:{d:math.fsum(p[arm+'_dimensions'][d] for p in pairs)/24*100
                                  for d in QUALITY_DIMENSIONS} for arm in ('random','target')},
        common_codec_options=common_codec,pairs=pairs,
        sources=[dict(run_id=s.marker['run_id'],attempt_id=s.marker['attempt_id'],
            artifact_index_sha256=s.marker['artifact_index_sha256'],destination=s.marker['artifact_publication']['destination'],
            transitive_local_references=sorted(getattr(s,'transitive_references',{}).values(),key=lambda v:v['path']))
            for s in (candidate,target)],
        caveats=['24 development prompts and one seed; six-dimension mean, not official full total.',
            ('Random schedules one two-chunk Target burst after one of the first eight Draft observations.'
             if p8_k2 else 'Random uses 2/3/6 non-root Target blocks at 15/30/60 seconds.'),
            'Native cache witnesses bind accepted history; Rolling clean restarts after external Target history remain part of its native algorithm.',
            'Original local latent/decode/evaluator aliases are required; source seals are never extended.',
            'Fixed scored MP4s and official per-video details are audited; this does not prove encoder or binary-motion-score repeatability.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--random-task',required=True);parser.add_argument('--target-task',required=True)
    parser.add_argument('--target-queue',type=Path);parser.add_argument('--target-commit')
    parser.add_argument('--commit',required=True);parser.add_argument('--model',choices=tuple(audit.MODELS),required=True)
    parser.add_argument('--seconds',type=int,choices=(15,30,60),required=True)
    args = parser.parse_args()
    candidate = audit.source(args.queue, args.random_task, args.repo, args.commit,
                             timing_scope=audit.OVERLAP_READY_SCOPE)
    target = audit.source(args.target_queue or args.queue, args.target_task, args.repo,
                          args.target_commit or args.commit, timing_scope=audit.OVERLAP_READY_SCOPE)
    print(json.dumps(compare(candidate, target, repo=args.repo, model=args.model,
                             seconds=args.seconds), sort_keys=True))


if __name__=='__main__':main()
