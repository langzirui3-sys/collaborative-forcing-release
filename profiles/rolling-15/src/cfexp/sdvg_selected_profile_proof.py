"""Verify the original selected-VAE integration profiles before full dispatch.

This consumes the original sealed runs, not an unsealed observer's success flag.
Missing trajectory-specific Target samples remain missing; common operator
coverage is established from independently measured, identical Target inputs.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from . import native_ready_report as audit
from . import paper128_sdvg_report as previous
from . import paper_population as population
from .baseline_shard_report import source_manifest
from .main_table_merge import _load_frozen_index, _load_json, _sha256_file, _require
from .sdvg_profile_coverage import validate_sample
from .sdvg_accepted_vae import POLICY
from .sdvg_vae_transaction import SCHEMA, sha, FIELDS
from .tracked_sdvg_accepted import validate_native_gate

PROTOCOL='experiments/paper128-sdvg-selected-profiles-v1.protocol.json'
PROTOCOL_SHA='b9ba3919904c1903d06f3430c3429f95a08a7a3d972f3a48e669c0d75e614436'
PROFILE_REVISION='6542fe13706375a69e57df71a02b2c2df96d9bc3'
MODELS={'self-forcing':'self_forcing_1.3b','rolling':'rolling_forcing_1.3b','longlive':'longlive_1.3b'}


def target_identity(artifact, *, artifact_sha256, precision, native_kv):
    """Use the Target artifact schema, which differs from Draft adapters."""
    config=artifact.get('backend_config',{})
    generation=config.get('generation',{})
    _require(artifact.get('schema')=='insight2_target_artifacts_v1'
        and artifact.get('status')=='frozen' and artifact.get('backend_module')=='cfexp.krea_chunk_backend'
        and config.get('schema')=='krea_chunk_backend_config_v1'
        and generation.get('model_id')=='krea-realtime-video-14b'
        and generation.get('denoising_steps')==4 and generation.get('kv_cache_num_frames')==3
        and generation.get('latent_shape_per_chunk')==[1,3,16,60,104]
        and generation.get('dtype')=='bfloat16', 'Target native backend configuration differs')
    return dict(artifact_sha256=artifact_sha256,backend_module=artifact['backend_module'],
        backend_config=config,precision=precision,native_kv=native_kv)


def receipts(generation, draft, selected):
    """Bind every actual cache commit to the reward decision and selected latent."""
    count=generation['chunk_count'];commits=generation.get('scoring_vae_commits')
    _require(generation.get('scoring_vae_policy')==POLICY
        and generation.get('selected_vae_commit_in_generation_timing') is True
        and generation.get('scoring_and_output_renderers_separate') is True
        and type(commits) is list and len(commits)==len(generation['routes'])==count
        and len(draft.segments)==len(selected.segments)==count,
        'selected scoring-VAE coverage or timed implementation differs')
    previous.frame_rewards(generation)
    sid=commits[0].get('session_id');_require(isinstance(sid,str) and len(sid)==64,'missing scoring-VAE session')
    head=sha(dict(schema=SCHEMA,session_id=sid,accepted=[]));tokens=set();targets=[]
    for i,(c,d,s,r) in enumerate(zip(commits,draft.segments,selected.segments,generation['routes'])):
        accepted=i>0 and r['block_reward']>=-.7
        after=sha(dict(schema=SCHEMA,previous=head,chunk_index=i,selected_sha256=s.latent_sha256))
        pending=r['reward'].get('vae_transaction',{})
        _require(c.get('schema')==SCHEMA and c.get('session_id')==sid
            and c.get('chunk_index')==i and c.get('committed_blocks')==i+1
            and c.get('prompt_id')==generation['prompt_id']
            and c.get('candidate_sha256')==d.latent_sha256==pending.get('candidate_sha256')
            and c.get('pending_token')==pending.get('pending_token')
            and isinstance(c.get('pending_token'),str) and len(c['pending_token'])==64
            and c['pending_token'] not in tokens
            and c.get('accepted_prefix_head_before')==head
            and c.get('accepted_prefix_head_after')==after
            and c.get('accepted_draft') is accepted and c.get('forced_reject') is (i==0)
            and c.get('threshold')==-.7 and c.get('block_reward')==r['block_reward']
            and c.get('selected_role')==s.producer_role==('draft' if accepted else 'target')
            and c.get('selected_sha256')==s.latent_sha256
            and c.get('selected_latent_path')==s.latent_path
            and c.get('diffusion_kv_untouched') is True
            and c.get('selected_decode_calls')==int(not accepted)
            and c.get('operation')==('keep_scored_draft_cache' if accepted else
                                    'restore_prefix_then_decode_selected_target')
            and c.get('snapshot_accounting',{}).get('fields')==list(FIELDS),
            'selected scoring-VAE receipt/decision/hash/history chain differs')
        _require(type(c.get('worker_commit_wall_seconds')) in (int,float)
            and math.isfinite(c['worker_commit_wall_seconds']) and c['worker_commit_wall_seconds']>=0,
            'selected commit timing is invalid')
        tokens.add(c['pending_token']);head=after
        if not accepted:targets.append(i)
    _require(generation['target_regeneration_count']==len(targets)
        and generation['draft_accept_count']==count-len(targets),'selected route count differs')
    return dict(committed_blocks=count,selected_prefix_head_sha256=head,target_blocks=len(targets),
        worker_commit_seconds=math.fsum(c['worker_commit_wall_seconds'] for c in commits))


def worker_evidence(src, *, count, target_count, samples):
    rows=[]
    for relative in src.frozen.entries:
        if relative.startswith(src.prefix+'provenance/tracking/') and relative.endswith('.jsonl'):
            rows.extend(json.loads(x) for x in src.frozen.file(relative,label='worker telemetry').read_text().splitlines())
    _require(rows,'selected-VAE original worker records missing')
    for role,expected in [('draft',{'generate':count,'score_sdvg_candidate':count,
                                  'commit_sdvg_selected_vae':count,'finish_sdvg_trajectory':1}),
                          ('target',{'generate_persistent':target_count})]:
        group=[x for x in rows if x.get('role')==role]
        _require(group and len({x.get('pid') for x in group})==1
            and len({x.get('sequence') for x in group})==len(group),
            'worker identity or sequence differs')
        for action,n in expected.items():
            _require(sum(x.get('action')==action for x in group)==n,'worker action coverage differs: '+action)
    for row in rows:
        _require(row.get('rank')==0 and type(row.get('peak_allocated_bytes')) is int
            and row['peak_allocated_bytes']>0 and type(row.get('peak_reserved_bytes')) is int
            and row['peak_reserved_bytes']>=row['peak_allocated_bytes'],
            'worker allocator evidence missing or inconsistent')
    actual=[r for r in rows if r.get('bounded_profile_sample')]
    _require(sorted(json.dumps(x,sort_keys=True) for x in samples)==
             sorted(json.dumps(x,sort_keys=True) for x in actual),'profile samples differ from actual worker log')
    return len(rows)


def read_profile(root, definition, *, repo, native_gate):
    name=definition['run_id'];relative=PurePosixPath(name)
    _require(isinstance(name,str) and len(relative.parts)==1 and relative.name==name
        and name not in ('.','..') and not relative.is_absolute(), 'unsafe original profile run identifier')
    run=root/definition['run_id']
    _require(run.is_dir() and not any(p.is_symlink() for p in (run,*run.parents)),
             'profile source is missing or symlinked')
    _require(_sha256_file(run/'artifact-index.json')[0]==definition['artifact_index_sha256'],
             'original selected profile index differs')
    frozen,marker=_load_frozen_index(run);frozen.verify_all()
    _require(marker['git_sha']==PROFILE_REVISION and marker['run_id']==run.name
        and marker['attempt_id']==definition['attempt_id'],'selected profile source identity differs')
    terminal=_load_json(frozen.file('run.json',label='profile terminal'),label='profile terminal')
    _require(terminal.get('execution_status')=='SUCCEEDED' and terminal.get('exit_code')==0
        and terminal.get('latest_attempt_id')==marker['attempt_id'],'profile execution failed')
    envelope=_load_json(frozen.file('resolved-manifest.json',label='profile invocation'),label='profile invocation')
    manifest=envelope['manifest'];relative='experiments/'+manifest['metadata']['name']+'.manifest.json'
    _require(manifest==source_manifest(repo,PROFILE_REVISION,relative)
        and all(envelope.get(k)==marker[k] for k in ('run_id','attempt_id','spec_hash')),
        'selected profile invocation differs from immutable source')
    spec=manifest['spec'];env=spec['command']['env'];argv=spec['command']['argv']
    _require(spec['resources']['gpus']==1 and argv[2]=='cfexp.tracked_sdvg_accepted'
        and '--smoke-prompt-id' in argv and env.get('CFEXP_SDVG_ACCEPTED_VAE')=='1'
        and env.get('CFEXP_SDVG_TRACKING')==env.get('CFEXP_SDVG_BOUNDED_PROFILE')==env.get('CFEXP_STAGE_TIMING')=='1',
        'profile was not the selected-VAE bounded integration entry')
    src=SimpleNamespace(run=run,prefix='attempts/'+marker['attempt_id']+'/',frozen=frozen,marker=marker)
    load=lambda path:audit.read_json(src,path)
    result=load('result.json');session=load('provenance/tracking/session.json')
    finish=load('provenance/tracking/finished.json');profile=load('provenance/tracking/profile-summary.json')
    c=session['config'];model=definition['model'];seconds=definition['seconds']
    _require(result.get('status')=='succeeded' and result.get('code_revision')==PROFILE_REVISION
        and result.get('native_model')==model and result.get('prompt_count')==result.get('row_count')==1
        and result.get('timing_scope')==previous.reference.scope(model)
        and c.get('git_sha')==PROFILE_REVISION and c.get('phase')=='bounded_reward_rpc_integration_not_main'
        and c.get('scoring_vae_policy')==POLICY and c.get('native_vae_gate')==native_gate
        and c.get('physical_device_count')==c.get('batch_size')==1 and c.get('rank')==0
        and c.get('nominal_seconds')==seconds and c.get('resolution')==[480,832]
        and c.get('generation_seed')==20260911 and c.get('threshold')==-.7
        and c.get('native_kv') is True and c.get('unchanged_sampler') is True
        and c.get('warmup_generator_forwards')==2
        and profile.get('config')==c and profile.get('hardware')==result.get('hardware')
        and profile.get('executed') is True and profile.get('device_count')==1,
        'profile workload, source, prerequisite or timing differs')
    gpus=result['hardware'].get('gpus',[])
    _require(len(gpus)==1 and gpus[0].get('name')=='NVIDIA H200'
        and gpus[0].get('uuid','').startswith('GPU-'),'profile is not one physical H200')
    _require(session.get('mode') in ('online','offline') and finish.get('mode')==session['mode']
        and finish.get('execution_succeeded') is True
        and session.get('synced_evidence_verified') is False and finish.get('synced_evidence_verified') is False,
        'actual W&B session did not finish or sync claim is unsupported')
    specs={}
    bindings={x['target']:x['sha256'] for x in spec['inputs']}
    for label,arg in [('draft','draft'),('target','target'),('reward','sdvg')]:
        path=argv[argv.index('--'+arg+'-artifact-spec')+1];digest=bindings[path]
        _require(c.get(label+'_artifact_spec_sha256')==digest
            and _sha256_file(Path(repo)/path)[0]==digest,'profile artifact identity differs')
        specs[label]=_load_json(Path(repo)/path,label='profile artifact')
    raw=audit._bound(frozen,src.prefix,result['raw_rows'],'profile raw rows')['rows']
    _require(len(raw)==1 and raw[0]['prompt_id']==c['prompt_id']=='001_subject_consistency_vbl0268'
        and raw[0]['seed']==20260911,
             'profile fixed prompt differs')
    row=raw[0];pid=row['prompt_id'];prov=load('provenance/sdvg/'+pid+'.json')
    generation=load('provenance/sdvg/'+pid+'-generation.json')['generation']
    _require(prov['generation']['scoring_vae_commits']==generation['scoring_vae_commits'],
             'pre-scoring and final selected cache receipts differ')
    selected,targets=previous.histories(src,prov,row,model=model,
        profile=previous.reference.duration_profile(seconds),seed=20260911)
    from .correction_state import LatentHistory
    draft=LatentHistory.from_dict(load('provenance/sdvg/'+pid+'-draft-history.json'))
    receipt=receipts(generation,draft,selected)
    prompts=_load_json(Path(repo)/population.PROTOCOL,label='original population')['prompts']
    prompt=next(p for p in prompts if p['id']==pid)
    audit.quality(src,row,result['vbench_measurements'][0],prompt,'mt-sdvg-'+pid,seconds)
    samples=profile['samples'];roles=[x.get('role') for x in samples]
    _require(roles.count('draft')==1 and roles.count('target') in (0,1) and len(roles)==len(set(roles))
        and profile.get('coverage_complete_for_this_campaign')==('target' in roles)
        and profile.get('missing_post_warmup_roles')==([] if 'target' in roles else ['target']),
        'missing trajectory-specific profile coverage was altered')
    for sample in samples:
        validate_sample(sample);path=PurePosixPath(sample['trace_path'])
        _require(not path.is_absolute() and '..' not in path.parts and str(path).startswith('tracking/profiler/'),
                 'unsafe server profiler path')
        trace=run/src.prefix/str(path)
        _require(not any(p.is_symlink() for p in (trace,*trace.parents))
            and _sha256_file(trace)[0]==sample['trace_sha256'],'server profiler trace differs from bound hash')
    observations=worker_evidence(src,count=seconds*2,target_count=len(targets),samples=samples)
    sdk=previous.sdk_observation(src,dict(session,directory='tracking/wandb'))
    return dict(model=model,seconds=seconds,run_id=marker['run_id'],git_sha=PROFILE_REVISION,
        artifact_index_sha256=definition['artifact_index_sha256'],samples=samples,
        missing_post_warmup_roles=profile['missing_post_warmup_roles'],worker_observations=observations,
        selected_cache_receipts=receipt,hardware=result['hardware'],wandb_mode=session['mode'],
        wandb_run_id=session['run_id'],wandb_sdk_observation=sdk,wandb_synced_evidence_verified=False,
        target_identity=target_identity(specs['target'],artifact_sha256=c['target_artifact_spec_sha256'],
            precision=c['precision'],native_kv=c['native_kv']))


def coverage(profiles):
    expected={(m,s) for m in MODELS for s in (15,30)}
    _require(len(profiles)==6 and {(p['model'],p['seconds']) for p in profiles}==expected,
             'all six selected-history integration workloads are required')
    identity=profiles[0]['target_identity'];common=[]
    for p in profiles:
        _require(p['git_sha']==PROFILE_REVISION and p['target_identity']==identity,
                 'Target operator source/artifact/precision/KV differs across profiles')
        for sample in p['samples']:
            validate_sample(sample)
            if sample['role']=='target':common.append(dict(source_run_id=p['run_id'],sample=sample))
    _require(common,'no actual post-warmup Target operator profile')
    return dict(schema='sdvg_selected_profile_admission_v1',status='verified_operator_and_cache_coverage',
        profiles=profiles,profile_count=6,scoring_vae_policy=POLICY,
        common_target_operator_profiles=common,no_imputed_trajectory_target_latencies=True,
        not_main_quality_or_speedup_result=True,
        memory_scope='per-process allocator peaks; not simultaneous physical-device peaks')


def validate_profiles(*,repo,runs_root):
    repo=Path(repo).resolve(strict=True);root=Path(runs_root)
    _require(root.is_absolute() and root.is_dir() and not any(p.is_symlink() for p in (root,*root.parents)),
             'profile runs root must be canonical and nonsymlinked')
    _require(not (repo/PROTOCOL).is_symlink() and _sha256_file(repo/PROTOCOL)[0]==PROTOCOL_SHA,
             'original selected profile prerequisite definitions changed')
    protocol=_load_json(repo/PROTOCOL,label='selected profile definitions')
    _require(protocol.get('schema')=='sdvg_selected_profile_prerequisites_v1'
        and protocol.get('profile_source_commit')==PROFILE_REVISION
        and protocol.get('evaluation_source_sha256')==population.SOURCE_SHA256
        and _sha256_file(repo/population.SOURCE_FILE)[0]==population.SOURCE_SHA256,
        'profile prerequisites or original evaluation source differ')
    gate=protocol['native_gate'];native=validate_native_gate(root/gate['run_id'],gate['artifact_index_sha256'])
    result=coverage([read_profile(root,d,repo=repo,native_gate=native) for d in protocol['profiles']])
    result.update(prerequisite_protocol_sha256=_sha256_file(repo/PROTOCOL)[0],native_vae_gate=native)
    return result
