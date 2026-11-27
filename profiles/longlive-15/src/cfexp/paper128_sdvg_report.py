"""Audit complete single-H200 SDVG replications, never substitute a profile.

Read original seals, independent proposal/selected histories, every frame's
reward, official scored MP4s, real worker telemetry and original Target pairs.
All outputs belong outside the source tree. This module never generates media.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

from . import native_ready_report as audit
from . import native_sdvg_standalone as legacy
from . import paper128_reference_report as reference
from . import paper_population as population
from .baseline_population import model_pair_for_draft
from .baseline_shard_report import source_manifest
from .correction_state import LatentHistory
from .main_table_merge import _load_frozen_index, _load_json, _require, _sha256_file
from .native_sdvg import audit_native_draft_history
from .sdvg_profile_coverage import read_profile, coverage
from .video_only_decode import SCHEMA as DECODE_SCHEMA

MATRIX='experiments/paper128-sdvg-v1.protocol.json'
MATRIX_SHA='0a1fdf62532199eefa9daa9aee83f7f77ce2325089a06c5185f5ce0d120208d1'
# These are the only changes between the pinned main and reference producers.
# Diagnostic/reader files are not on these non-joint SDVG/Target execution paths.
# Telemetry wrappers are disclosed and their overhead stays in SDVG timings.
SOURCE_DIFFERENCES={
    'src/cfexp/longlive_joint_gate.py','src/cfexp/longlive_joint_handoff.py',
    'src/cfexp/longlive_joint_oracle.py','src/cfexp/longlive_joint_trace.py',
    'src/cfexp/paper128_reference_report.py','src/cfexp/sdvg_profile_coverage.py',
    'src/cfexp/sf_handoff_continuity.py','src/cfexp/sf_handoff_legacy_executor.py',
    'src/cfexp/tracked_sdvg.py','src/cfexp/tracked_sdvg_worker.py',
    'pyproject.toml','uv.lock'}
MAIN_REVISION='a91506acdc538a05ca70be864e25704211f61400'
TARGET_REVISION='6dc91cb78c8681d58e3727e8aef9979e066b1328'


def matrix(repo):
    p=Path(repo)/MATRIX
    _require(not p.is_symlink() and _sha256_file(p)[0]==MATRIX_SHA,'frozen SDVG matrix differs')
    return _load_json(p,label='SDVG matrix')


def source(run, *, repo, model, seconds):
    _require(model in reference.MODELS and type(seconds) is int and seconds in (15,30),
             'unsupported main SDVG model/horizon')
    requested=Path(run);_require(not requested.is_symlink(),'SDVG source is a symlink')
    run=requested.resolve(strict=True);frozen,marker=_load_frozen_index(run)
    terminal=_load_json(frozen.file('run.json',label='terminal SDVG'),label='terminal SDVG')
    _require(terminal.get('execution_status')=='SUCCEEDED' and terminal.get('exit_code')==0
        and terminal.get('latest_attempt_id')==marker['attempt_id']
        and terminal.get('run_id')==marker['run_id']==run.name
        and marker['git_sha']==MAIN_REVISION,'SDVG source has not sealed the pinned successful main execution')
    envelope=_load_json(frozen.file('resolved-manifest.json',label='SDVG manifest'),label='SDVG manifest')
    invocation=envelope['manifest'];relative='experiments/'+invocation['metadata']['name']+'.manifest.json'
    selected=[d for d in matrix(repo)['runs'] if d['manifest']==relative]
    _require(len(selected)==1 and selected[0]['nominal_seconds']==seconds
        and selected[0]['model_pair']==model_pair_for_draft(reference.MODELS[model])
        and all(envelope.get(k)==marker[k] for k in ('run_id','attempt_id','spec_hash'))
        and invocation==source_manifest(repo,marker['git_sha'],relative),
        'SDVG source/invocation/population differs')
    prefix='attempts/'+marker['attempt_id']+'/'
    result=_load_json(frozen.file(prefix+'result.json',label='SDVG result'),label='SDVG result')
    _require(result.get('status')=='succeeded' and result.get('code_revision')==marker['git_sha'],
             'SDVG status/revision differs')
    return SimpleNamespace(run=run,prefix=prefix,frozen=frozen,marker=marker,result=result,
                           manifest=invocation,definition=selected[0])


def selection(src, *, repo, model, seconds):
    d=src.definition;spec=src.manifest['spec'];argv=spec['command']['argv'];env=spec['command']['env']
    _require(spec['resources']['gpus']==1 and env.get(population.ENV)=='1'
        and env.get(population.SEED_ENV)==str(d['generation_seed'])
        and env.get('CFEXP_PAPER_PROMPT_SHARD_COUNT')=='4'
        and env.get('CFEXP_PAPER_PROMPT_SHARD_INDEX')==str(d['logical_matched_gpu_slot'])
        and env.get('CFEXP_SDVG_TRACKING')=='1' and env.get('CFEXP_SDVG_BOUNDED_PROFILE')=='0'
        and env.get('CFEXP_STAGE_TIMING')=='0' and argv[2]=='cfexp.tracked_sdvg'
        and '--smoke-prompt-id' not in argv and '--record-video-ready-timing' in argv
        and '--overlap-native-vae' in argv and '--baseline-population-v2' in argv
        and argv[argv.index('--duration-seconds')+1]==str(seconds)
        and argv[argv.index('--draft-model-id')+1]==reference.MODELS[model],
        'SDVG main is not its full unprofiled one-card shard')
    required=dict(CFEXP_ACCEPTED_VIDEO_STREAM='1',CFEXP_RENDER_TARGET_PRIORITY='1',
                  CFEXP_VIDEO_ONLY_DECODE='1',CFEXP_VIDEO_ENCODING='serial_abr_v1')
    if model=='self-forcing':
        required.update(CFEXP_SF_SINGLE_GPU='1',CFEXP_DRAFT_KV_REUSE='1',CFEXP_DRAFT_KV_FORK='0')
        _require('CFEXP_NATIVE_VIDEO_STREAM_MODEL' not in env,'SF selected another decoder')
    else:required.update(CFEXP_NATIVE_SINGLE_GPU='1',CFEXP_NATIVE_VIDEO_STREAM_MODEL=model)
    _require(all(env.get(k)==v for k,v in required.items())
        and not any(v=='1' for k,v in env.items() if 'JOINT' in k or 'DIAGNOSTIC' in k),
        'SDVG native KV/renderer/execution flags differ')
    with patch.dict(os.environ,env,clear=True):
        prompts,binding=population.load(repo,model_pair=model_pair_for_draft(reference.MODELS[model]))
    profile=reference.duration_profile(seconds)
    duration={k:v for k,v in profile.metadata().items() if not k.startswith(('ours_','calibration_')) and k!='budget_rule'}
    duration.update(experiment_id='main-table-population-v2',benchmark='VBench')
    result=src.result
    _require([p['id'] for p in prompts]==d['prompt_ids'] and len(prompts)==32
        and result.get('schema')=='baseline_population_v2_method_result'
        and result.get('population')==binding and result.get('run_kind')=='formal_campaign'
        and result.get('method_id')=='sdvg' and result.get('native_model')==model
        and result.get('prompt_count')==result.get('row_count')==32 and result.get('seed_count')==1
        and result.get('duration')==duration and result.get('benchmark')=='VBench'
        and result.get('timing_scope')==reference.scope(model)
        and result.get('draft_history_policy')=='always_append_draft_candidate'
        and result.get('selected_history_policy')=='append_accepted_draft_else_target_regeneration',
        'SDVG full population/policy/timing differs')
    gpus=result['hardware']['gpus']
    _require(len(gpus)==1 and gpus[0].get('name')=='NVIDIA H200'
        and gpus[0].get('uuid','').startswith('GPU-') and not result['hardware'].get('unverified_test_fixture'),
        'SDVG is not an actual single H200')
    path=argv[argv.index('--sdvg-artifact-spec')+1]
    digest=_sha256_file(Path(repo)/path)[0];artifact=_load_json(Path(repo)/path,label='SDVG artifact')
    routing=artifact['adapter_config']['routing']
    _require(any(i['target']==path and i['sha256']==digest for i in spec['inputs'])
        and artifact.get('status')=='frozen' and artifact.get('method_id')=='sdvg'
        and routing.get('threshold')==legacy.THRESHOLD and routing.get('force_reject_chunk_indices')==[0]
        and routing.get('draft_history_policy')=='always_append_draft_candidate'
        and routing.get('target_history_policy')=='append_accepted_draft_else_target_regeneration'
        and routing.get('rejection_noise_policy')=='target_regenerates_from_same_initial_noise'
        and artifact['adapter_config']['benchmark_adaptations']['chunk_count']==profile.chunk_count,
        'SDVG threshold/noise/history artifact differs')
    pre=audit.read_json(src,'provenance/preflight.json')
    _require(pre.get('git',{}).get('sha')==src.marker['git_sha']
        and pre.get('sdvg_artifact_spec_sha256')==digest and pre.get('seed')==binding['seed']
        and pre.get('draft_model_id')==reference.MODELS[model]
        and pre.get('prompt_ids')==d['prompt_ids'] and pre.get('threshold')==legacy.THRESHOLD,
        'SDVG original preflight differs')
    return prompts,binding,duration


def frame_rewards(generation):
    for i,route in enumerate(generation['routes']):
        reward=route['reward'];values=reward.get('frame_rewards',[]);count=9 if i==0 else 12
        _require(reward.get('chunk_index')==i and reward.get('rgb_frame_count')==len(values)==count
            and all(type(x) in (float,int) and math.isfinite(x) for x in values)
            and min(values)==reward.get('block_reward')==route['block_reward'],
            'SDVG reward is not the minimum over every decoded RGB frame')


def histories(src, provenance, row, *, model, profile, seed):
    # The historical one-seed reader is itself a frozen manifest input. Keep
    # its bytes unchanged and apply its independent-history checks here with
    # the actual case seed, never rewrite/relabel the old evidence.
    pid=row['prompt_id']
    draft=LatentHistory.from_dict(audit.read_json(src,f'provenance/sdvg/{pid}-draft-history.json'))
    selected=LatentHistory.from_dict(audit.read_json(src,f'provenance/sdvg/{pid}-selected-history.json'))
    generation=provenance['generation']
    _require(generation.get('schema')==legacy.SCHEMA and generation.get('method_id')=='sdvg'
        and generation.get('draft_model_id')==reference.MODELS[model]
        and generation.get('prompt_id')==pid and generation.get('seed')==seed
        and generation.get('chunk_count')==profile.chunk_count and generation.get('threshold')==legacy.THRESHOLD
        and generation['draft_history_manifest_sha256']==draft.manifest_sha256
        and generation['selected_history_manifest_sha256']==selected.manifest_sha256,
        'SDVG actual seed or independent history identity differs')
    for history in (draft,selected):
        _require(history.prompt_id==pid and history.trajectory_seed==seed
            and history.next_chunk_index==profile.chunk_count,'SDVG history coverage differs')
        for segment in history.segments:
            audit.referenced_file(src,dict(path=segment.latent_path,sha256=segment.latent_sha256,
                                          nbytes=segment.latent_nbytes),'SDVG original latent')
    proof=audit_native_draft_history(generation)
    _require(row['native_cache_audit']==provenance['native_cache_audit']==proof,
             'SDVG cache audit differs from actual per-chunk evidence')
    targets=[]
    for i,(route,d,s) in enumerate(zip(generation['routes'],draft.segments,selected.segments)):
        reward=route['block_reward']
        _require(type(reward) in (int,float) and math.isfinite(reward),'invalid SDVG reward')
        accepted=i>0 and reward>=legacy.THRESHOLD
        _require(route['threshold']==legacy.THRESHOLD and route['forced_reject'] is (i==0)
            and route['route']==('draft' if accepted else 'target')
            and route['reward']['block_reward']==reward and route['reward']['threshold']==legacy.THRESHOLD
            and route['reward']['accepted_by_threshold'] is (reward>=legacy.THRESHOLD)
            and route['reward']['forced_reject'] is (i==0)
            and d.producer_role=='draft' and s.producer_role==route['route']
            and route['draft_latent_path']==d.latent_path and route['selected_latent_path']==s.latent_path
            and route['selected_latent_sha256']==s.latent_sha256
            and d.initial_noise_descriptor_sha256==s.initial_noise_descriptor_sha256
            and (not accepted or d.latent_sha256==s.latent_sha256),
            'SDVG threshold, independent candidate, selected latent or same-noise decision differs')
        if not accepted:targets.append(i)
    _require(generation['target_regeneration_count']==len(targets)
        and generation['draft_accept_count']==profile.chunk_count-len(targets),'SDVG target count differs')
    return selected,targets


def pre_scoring(src, provenance, *, pid, seed, duration):
    for stage in ('generation','decoded'):
        saved=audit.read_json(src,f'provenance/sdvg/{pid}-{stage}.json')
        expected=provenance['generation']
        if stage=='generation':expected={k:v for k,v in expected.items() if k not in ('frames','video')}
        _require(saved.get('schema')==f'sdvg_{stage}_checkpoint_v1'
            and saved.get('quality_evaluated') is False and saved.get('method_id')=='sdvg'
            and saved.get('prompt_id')==pid and saved.get('seed')==seed
            and saved.get('duration')==duration and saved.get('generation')==expected,
            'SDVG original pre-scoring generation/decode differs')


def sdk_observation(src, session):
    """Read a server-only SDK file without pretending it was originally sealed.

    A915 deliberately publishes provenance, not the private SDK directory. The
    indexed session and worker records remain mandatory; a new observer binds
    the currently present SDK bytes separately and never updates the old index.
    """
    sid=session.get('run_id')
    _require(isinstance(sid,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',sid) is not None
        and session.get('directory')=='tracking/wandb','unsafe W&B session identity/directory')
    root=src.run/src.prefix/'tracking/wandb'
    current=src.run
    for part in root.relative_to(src.run).parts:
        current=current/part
        _require(not current.is_symlink() and current.is_dir(),'W&B SDK directory unavailable or symlinked')
    candidates=list(root.rglob('run-'+sid+'.wandb'))
    _require(len(candidates)==1,'matching W&B SDK file missing or ambiguous')
    path=candidates[0];relative=path.relative_to(src.run).as_posix();current=src.run
    for part in path.relative_to(src.run).parts:
        current=current/part
        _require(not current.is_symlink(),'W&B SDK path traverses a symlink')
    before=path.stat()
    _require(stat.S_ISREG(before.st_mode) and before.st_size>0,'W&B SDK is not a nonempty regular file')
    originally_indexed=relative in src.frozen.entries
    if originally_indexed:src.frozen.file(relative,label='original indexed W&B SDK binary')
    sha,size=_sha256_file(path);after=path.stat()
    _require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
             and size==after.st_size,'W&B SDK changed during observation')
    return dict(path=relative,sha256=sha,nbytes=size,observed_server_file_verified=True,
        original_artifact_index_bound=originally_indexed,
        evidence_scope=('original_sealed_artifact' if originally_indexed else 'current_server_observation'),
        retroactive_seal_claim=False,synced_evidence_verified=False)


def tracking(src, *, binding, model, seconds, target_calls):
    session=audit.read_json(src,'provenance/tracking/session.json')
    finish=audit.read_json(src,'provenance/tracking/finished.json');c=session['config']
    _require(session.get('schema')=='wandb_session_v1' and session.get('mode') in ('online','offline')
        and session.get('synced_evidence_verified') is False
        and finish.get('schema')=='wandb_finish_v1' and finish.get('mode')==session['mode']
        and finish.get('execution_succeeded') is True and finish.get('synced_evidence_verified') is False
        and c.get('phase')=='main' and c.get('git_sha')==src.marker['git_sha']
        and c.get('generation_seed')==binding['seed'] and c.get('nominal_seconds')==seconds
        and c.get('prompt_shard')==binding['shard_index'] and c.get('physical_device_count')==1
        and c.get('model_pair')==reference.MODELS[model] and c.get('native_kv') is True
        and c.get('batch_size')==1 and c.get('output_resolution')==[480,832]
        and c.get('threshold')==legacy.THRESHOLD,'actual W&B session/finish/configuration differs')
    logs=[]
    for relative in src.frozen.entries:
        if relative.startswith(src.prefix+'provenance/tracking/') and relative.endswith('.jsonl'):
            logs.extend(json.loads(l) for l in src.frozen.file(relative,label='actual worker telemetry').read_text().splitlines())
    _require(logs,'original indexed worker records missing')
    sdk=sdk_observation(src,session)
    for role,expected,action in [('draft',seconds*2*32,'generate'),('target',target_calls,'generate_persistent')]:
        records=[x for x in logs if x['role']==role]
        forwards=[x for x in records if x['action']==action]
        _require(len({x['pid'] for x in records})==1 and len(forwards)==expected
            and len({x['sequence'] for x in records})==len(records)
            and all(x.get('bounded_profile_sample') is False and x.get('rank')==0
                and type(x.get('peak_allocated_bytes')) is int and x['peak_allocated_bytes']>0
                and type(x.get('peak_reserved_bytes')) is int
                and x['peak_reserved_bytes']>=x['peak_allocated_bytes'] for x in records),
            'SDVG worker sample counts/profile mode/real allocator evidence differs')
    return dict(mode=session['mode'],wandb_run_id=session['run_id'],synced_evidence_verified=False,
                sdk_file=sdk,original_indexed_worker_records_verified=True,worker_observations=len(logs))


def verified_rows(src, *, repo, model, seconds):
    prompts,binding,duration=selection(src,repo=repo,model=model,seconds=seconds)
    raw=audit._bound(src.frozen,src.prefix,src.result['raw_rows'],'SDVG raw rows')['rows']
    proofs=src.result['vbench_measurements']
    _require(len(raw)==len(proofs)==32 and [r['prompt_id'] for r in raw]==src.definition['prompt_ids'],
             'SDVG raw/scoring population differs')
    checked=[];batches={};profile=reference.duration_profile(seconds);target_calls=0
    for prompt,row,evidence in zip(prompts,raw,proofs):
        pid=prompt['id'];pre=audit.read_json(src,'provenance/sdvg/'+pid+'.json')
        _require(row['method_id']==pre['method_id']=='sdvg' and pre['prompt_id']==pid
            and row['seed']==pre['seed']==binding['seed'] and row['duration']==duration
            and row.get('official_vbench_verified') is True and pre['official_vbench']['verified'] is True
            and pre['official_vbench']['dimension_scores']==row['quality_scores']
            and pre['official_vbench']['evidence']==evidence
            and row['timing_scope']==pre['timing_scope']==reference.scope(model)
            and row['video_ready_wall_seconds']==pre['video_ready_wall_seconds']
            and audit.positive(row['video_ready_wall_seconds'],'ready')>=audit.positive(row['generation_wall_seconds'],'generation')
            and row['generation_wall_seconds']==pre['generation']['timing']['generation_wall_seconds'],
            'SDVG scoring/input/ready timing differs')
        pre_scoring(src,pre,pid=pid,seed=binding['seed'],duration=duration)
        history,targets=histories(src,pre,row,model=model,profile=profile,seed=binding['seed'])
        frame_rewards(pre['generation']);target_calls+=len(targets)
        mid='mt-sdvg-'+pid;video,actual,codec=audit.quality(src,row,evidence,prompt,mid,seconds)
        reference.legacy.official_scores(src,dict(row,quality_evidence=evidence),prompt,mid,seconds,
                                        video,short_batches=batches)
        audit.verify_geometry(video,profile);g=pre['generation'];declared=g['video']
        _require(video==src.run/src.prefix/declared['path']
            and all(actual[k]==declared[k] for k in ('sha256','nbytes'))
            and declared['fps']==24 and declared['frames']==profile.frame_count and g['frames'] is not None
            and row['accepted_stream_evidence']==pre['accepted_stream_evidence'],
            'SDVG timed/scored MP4 or selected renderer differs')
        audit.verify_accepted_renderer(src,row['accepted_stream_evidence'],history,
            dict(schema=DECODE_SCHEMA,pixels=g['frames'],video=declared),model=model,role='target',profile=profile)
        checked.append(dict(row=dict(row,prompt=prompt['text']),history=history,codec=codec,target_chunks=targets))
    src.tracking=tracking(src,binding=binding,model=model,seconds=seconds,target_calls=target_calls)
    return checked


def source_compatibility(repo, candidate, target):
    _require(candidate==MAIN_REVISION and target==TARGET_REVISION,'SDVG/Target source pair is not the frozen replication')
    env=dict(PATH='/usr/bin:/bin',HOME=os.path.expanduser('~'),GIT_CONFIG_NOSYSTEM='1',
             GIT_CONFIG_GLOBAL='/dev/null',GIT_NO_REPLACE_OBJECTS='1')
    def git(*args):return subprocess.check_output(['git','--no-replace-objects','-c','core.hooksPath=/dev/null',
        '-C',str(repo),*args],env=env,text=True)
    changed=git('diff','--name-only',candidate,target,'--','src/cfexp','pyproject.toml','uv.lock').splitlines()
    _require(set(changed)<=SOURCE_DIFFERENCES,'common production operator or renderer source differs')
    before=tomllib.loads(git('show',target+':pyproject.toml'));after=tomllib.loads(git('show',candidate+':pyproject.toml'))
    dependencies=lambda x:x['project']['dependencies']
    _require(set(dependencies(after))-set(dependencies(before))=={'wandb==0.19.11'}
        and set(dependencies(before))<=set(dependencies(after)),'runtime dependency change exceeds W&B instrumentation')
    old=tomllib.loads(git('show',target+':uv.lock'));new=tomllib.loads(git('show',candidate+':uv.lock'))
    # Root metadata records the added dependency; installed existing packages
    # must retain their exact prior versions, sources and wheel bindings.
    extra_mac_wheels=verify_locked_runtime(old,new)
    return dict(candidate_revision=candidate,target_revision=target,source_differences=changed,
        common_generation_backend_and_renderer_unchanged=True,additional_instrumentation='wandb==0.19.11',
        added_macos_wheel_bindings=extra_mac_wheels,
        instrumentation_overhead_included_in_sdvg_latency=True,identical_whole_source_claim=False)


def verify_locked_runtime(old, new):
    additions={}
    for package in old['package']:
        if package['name']=='collaborative-forcing':continue
        peers=[p for p in new['package'] if (p['name'],p.get('version'),p.get('source'))==
               (package['name'],package.get('version'),package.get('source'))]
        _require(len(peers)==1,'an existing locked package version/source disappeared')
        peer=peers[0]
        _require({k:v for k,v in package.items() if k!='wheels'}==
                 {k:v for k,v in peer.items() if k!='wheels'}
                 and all(w in peer.get('wheels',[]) for w in package.get('wheels',[])),
                 'an existing locked runtime package or wheel changed')
        extra=[w for w in peer.get('wheels',[]) if w not in package.get('wheels',[])]
        _require(all('macosx_' in w['url'] for w in extra),
                 'new non-macOS wheel binding in an existing runtime package')
        if extra:additions[package['name']]=len(extra)
    return additions


def compare(sources, targets, *, repo, model, seconds, profile_coverage):
    definitions=[d for d in matrix(repo)['runs'] if d['model_pair']==model_pair_for_draft(reference.MODELS[model])
                 and d['nominal_seconds']==seconds]
    _require(len(sources)==len(targets)==12 and len({s.run for s in sources+targets})==24
        and {s.definition['id'] for s in sources}=={d['id'] for d in definitions}
        and {s.definition['manifest'] for s in targets}=={d['matched_target_manifest'] for d in definitions}
        and all(s.definition['method_id']=='target_only' for s in targets),'need all twelve SDVG and paired pure Target shards')
    _require(profile_coverage['status']=='verified_operator_coverage'
        and profile_coverage['profile_count']==6
        and all(p['git_sha']==MAIN_REVISION for p in profile_coverage['profiles']),
        'six actual same-source operator preflight profiles are required')
    compatibility=source_compatibility(repo,sources[0].marker['git_sha'],targets[0].marker['git_sha'])
    methods={};controls={}
    for src in sources:
        _require(src.marker['git_sha']==MAIN_REVISION,'mixed SDVG source revisions')
        for x in verified_rows(src,repo=repo,model=model,seconds=seconds):
            k=x['row']['prompt_id'],x['row']['seed'];_require(k not in methods,'duplicate SDVG case');methods[k]=x,src
    for src in targets:
        _require(src.marker['git_sha']==TARGET_REVISION,'mixed Target source revisions')
        for x in reference.verified_rows(src,repo=repo,model=model,seconds=seconds):
            k=x['row']['prompt_id'],x['row']['seed'];_require(k not in controls,'duplicate Target case');controls[k]=x,src
    keys=[(p['id'],seed) for p in _load_json(Path(repo)/population.PROTOCOL,label='population')['prompts'] for seed in population.SEEDS]
    _require(len(keys)==384 and set(methods)==set(controls)==set(keys),'full384-case pairing differs')
    pairs=[]
    for key in keys:
        a,sa=methods[key];b,sb=controls[key]
        _require(sa.result['hardware']==sb.result['hardware'] and a['codec']==b['codec'],
                 'SDVG/Target physical GPU or actual encoder differs')
        paired=legacy.pair_row(a['row'],a['history'],b)
        pairs.append(dict(prompt_id=key[0],seed=key[1],gpu_uuid=sa.result['hardware']['gpus'][0]['uuid'],
            sdvg_quality_100=math.fsum(paired['quality_scores'].values())*100/6,
            target_quality_100=math.fsum(paired['target_quality_scores'].values())*100/6,
            sdvg_ready_seconds=paired['video_ready_wall_seconds'],target_ready_seconds=paired['target_ready_seconds'],
            target_blocks=len(a['target_chunks'])))
    mean=lambda k:math.fsum(p[k] for p in pairs)/384
    return dict(schema='paper128_sdvg_single_h200_matched_report_v1',status='verified',model=model,
        nominal_seconds=seconds,prompt_count=128,seeds=list(population.SEEDS),case_count=384,actual_gpus_per_video=1,
        sdvg_quality_mean_100=mean('sdvg_quality_100'),target_quality_mean_100=mean('target_quality_100'),
        sdvg_mean_ready_seconds=mean('sdvg_ready_seconds'),target_mean_ready_seconds=mean('target_ready_seconds'),
        speedup_ratio_of_mean_ready_times=mean('target_ready_seconds')/mean('sdvg_ready_seconds'),
        mean_target_blocks_including_root=mean('target_blocks'),pairs=pairs,source_compatibility=compatibility,
        tracking=[dict(run_id=s.marker['run_id'],**s.tracking) for s in sources],
        profile_source_bindings=[{k:p[k] for k in ('run_id','git_sha','artifact_index_sha256')} for p in profile_coverage['profiles']],
        sources=[{k:s.marker[k] for k in ('run_id','attempt_id','git_sha','artifact_index_sha256')} for s in sources+targets],
        dataset_publication_verified=False,replication_not_recovered_original_submission_record=True,
        caveats=['Six-dimension macro mean, not official full VBench total.',
            'ImageReward SDVG adaptation uses every decoded frame minimum and fixed threshold -0.7.',
            'Independent native Draft proposal history and selected Target history remain separate.',
            'Original indexed W&B session/worker records are verified. Server-only SDK bytes receive a separate current observation hash, not a retroactive seal. Remote synchronization is not established.',
            'Main timing includes telemetry overhead; separate CUDA profile samples are never substituted.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('ours','targets','repo','profile-runs'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--model',choices=tuple(reference.MODELS),required=True)
    p.add_argument('--seconds',type=int,choices=(15,30),required=True);a=p.parse_args()
    profiles=coverage([read_profile(r,repo=a.repo) for r in json.loads(a.profile_runs.read_text())])
    methods=[source(r,repo=a.repo,model=a.model,seconds=a.seconds) for r in json.loads(a.ours.read_text())]
    targets=[reference.source(r,repo=a.repo,model=a.model,seconds=a.seconds) for r in json.loads(a.targets.read_text())]
    print(json.dumps(compare(methods,targets,repo=a.repo,model=a.model,seconds=a.seconds,profile_coverage=profiles),allow_nan=False))


if __name__=='__main__':main()
