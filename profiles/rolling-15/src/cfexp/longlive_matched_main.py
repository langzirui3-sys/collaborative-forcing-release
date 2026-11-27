"""Tracked paper128 LongLive Ours/Target/Draft entries after full-video admission.

This module does not change any model, sampler, route or original driver. Each
invocation owns one frozen seed/shard and one physical H200. Matched comparison
is performed only after independently sealed runs prove identical GPU, cases,
source and timing scope; executing this entry alone is not a speedup claim.
"""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path

from . import longlive_fullvideo_preflight as preflight
from . import longlive_fusion_runtime as fusion
from .tracked_sdvg import write_new, tracking_mode, materialize_sdk_log_links, log_result

PREFLIGHT_SOURCE = '047594b47db4443c42d4f8d1ba093fe6907a8cf1'
PREFLIGHT_MANIFEST = 'experiments/longlive-complete-video-fusion-preflight-single-h200-ntu-v1.manifest.json'
PREFLIGHT_ENV = 'CFEXP_LONGLIVE_FULLVIDEO_PREFLIGHT_RUN'
METHODS = ('ours','target','draft')


def require(value,message):
    if not value:
        raise ValueError(message)


def validate_options(method,seconds,env=None):
    from .paper_population import generation_seed
    env=os.environ if env is None else env
    require(method in METHODS and seconds in (15,30), 'unsupported main model/duration')
    require(env.get(fusion.ENV) == ('1' if method=='ours' else '0'),
            'only Ours may install native history/first-step fusion')
    require(env.get('CFEXP_SDVG_BOUNDED_PROFILE')=='0', 'main latency cannot contain profiler hooks')
    require(env.get('CFEXP_PAPER_PROMPT_SHARD_COUNT')=='4'
            and env.get('CFEXP_PAPER_PROMPT_SHARD_INDEX') in ('0','1','2','3'),
            'main requires one of the four complete original population shards')
    require(bool(env.get(PREFLIGHT_ENV)), 'main requires original sealed complete-video proof')
    generation_seed(env)
    # Reuse its unchanged model/execution isolation checks. Only the population
    # phase and plain/fused opt-in differ; actual main values are checked above.
    check={**env, fusion.ENV:'1','CFEXP_GENERATION_SEED':'20260911',
           'CFEXP_PAPER_PROMPT_SHARD_COUNT':'64','CFEXP_PAPER_PROMPT_SHARD_INDEX':'0'}
    preflight.validate_options(check)


def resolve_preflight(path,plan_sha=None):
    path=Path(path)
    require(path.is_absolute() and not path.is_symlink(), 'unsafe preflight location')
    if path.is_dir():
        require(plan_sha is None, 'a direct run cannot inherit a queue binding')
        return path,None
    require(path.is_file() and isinstance(plan_sha,str) and len(plan_sha)==64
            and fusion.sha(path)==plan_sha, 'preflight queue hash differs')
    plan=json.loads(path.read_text())
    tasks=[x for x in plan.get('tasks',[]) if x.get('manifest')==PREFLIGHT_MANIFEST]
    require(plan.get('schema')=='cfexp_erank_queue_v1' and plan.get('ops')==str(path.parent)
            and len(tasks)==1 and tasks[0].get('commit')==PREFLIGHT_SOURCE,
            'preflight queue does not bind one original source/workload')
    state_path=path.parent/'state.json'
    require(not state_path.is_symlink(), 'unsafe preflight queue state')
    state=json.loads(state_path.read_text());row=state.get('tasks',{}).get(tasks[0]['id'],{})
    require(state.get('plan_sha256')==plan_sha and row.get('status')=='succeeded'
            and isinstance(row.get('run_dir'),str) and Path(row['run_dir']).is_absolute(),
            'preflight original successful run is not available')
    return Path(row['run_dir']),dict(plan=str(path),sha256=plan_sha,task_id=tasks[0]['id'])


def admit_preflight(repo,path,hardware,plan_sha=None):
    from .main_table_merge import _load_frozen_index
    run,queue=resolve_preflight(path,plan_sha)
    frozen,marker=_load_frozen_index(run)
    require(marker.get('status')=='SUCCEEDED' and marker.get('exit_code')==0
            and marker.get('git_sha')==PREFLIGHT_SOURCE, 'complete-video source/seal differs')
    frozen.verify_all();prefix='attempts/'+marker['attempt_id']+'/'
    manifest=json.loads(frozen.file('resolved-manifest.json',label='preflight manifest').read_text())['manifest']
    require(manifest['metadata']['name']==Path(PREFLIGHT_MANIFEST).name.removesuffix('.manifest.json')
            and manifest['spec']['resources']['gpus']==1, 'wrong full-video preflight')
    for item in manifest['spec']['inputs']:
        if item['uri'].startswith('repository://'):
            original=Path(repo)/item['target']
            require(original.is_file() and not original.is_symlink()
                    and fusion.sha(original)==item['sha256'],
                    'full-video admitted source changed: '+item['target'])
    report=json.loads(frozen.file(prefix+'result.json',label='full-video result').read_text())
    require(report.get('schema')=='longlive_complete_video_fusion_preflight_v1'
            and report.get('passed') is True and report.get('not_paper128_main') is True
            and list(report.get('cells',{}))==sorted(x[0] for x in preflight.CELL_ORDER)
            and len(report.get('comparisons',[]))==5
            and len(report.get('paired_unprofiled_observations',[]))==4,
            'full-video proof lacks complete profile and both-order coverage')
    attempt=run/'attempts'/marker['attempt_id']; reference=None
    for label,fused,profiled in preflight.CELL_ORDER:
        cell=attempt/'cells'/label
        result=json.loads(frozen.file(prefix+'cells/'+label+'/result.json',label=label).read_text())
        current=preflight.snapshot(cell,result,fused=fused)
        require(current['population']['shard_count']==64
                and current['population']['shard_index']==0
                and current['population']['seed']==20260911
                and len(current['hardware']['gpus'])==1, 'preflight case/seed/device identity differs')
        gpu=current['hardware']['gpus'][0]; actual=hardware['gpus'][0]
        require(all(gpu[k]==actual[k] for k in ('name','memory_total','driver_version')),
                'main hardware class differs from actual bounded preflight')
        if reference is None: reference=current
        else: preflight.compare(reference,current)
        if profiled:
            profile=json.loads(frozen.file(prefix+'cells/'+label+'/provenance/tracking/profile-summary.json',
                label='actual bounded profile').read_text())
            require(profile.get('coverage_complete_for_this_campaign') is True
                    and {x['role'] for x in profile['samples']}=={'draft','target'}, 'missing actual owner profile')
            from .krea_long_session_gate import validate_operator_profile
            for sample in profile['samples']:
                validate_operator_profile(sample,role=sample['role'],action='generate',root=cell)
    finish=json.loads(frozen.file(prefix+'provenance/tracking/finished.json',label='tracking finish').read_text())
    require(finish.get('execution_succeeded') is True, 'full-video tracking did not finish')
    return dict(run_id=marker['run_id'],source=PREFLIGHT_SOURCE,
        artifact_index_sha256=marker['artifact_index_sha256'],queue_binding=queue,
        complete_video_exactness_and_actual_profiles_verified=True,
        representative_hardware_class=True,same_physical_gpu_as_preflight=reference['hardware']==hardware,
        no_population_quality_or_matched_target_speedup_claim=True)


@contextmanager
def tracked_reference_runtime(module):
    """Isolated controller process; restore the original callable on every exit."""
    original=module.launch_correction_runtime
    module.launch_correction_runtime=preflight.launch_runtime
    try: yield
    finally: module.launch_correction_runtime=original


def execute(args):
    common=dict(draft_artifact_spec=args.draft_artifact_spec,target_artifact_spec=args.target_artifact_spec,
        vbench_checkout=args.vbench_checkout,vbench_python=args.vbench_python,
        vbench_asset_manifest=args.vbench_asset_manifest,vbench_asset_spec=args.vbench_asset_spec,
        duration_seconds=args.duration_seconds,overlap_native_vae=True)
    if args.method=='ours':
        from .main_table_ours_campaign import run_campaign
        common['draft_artifact_spec_path']=common.pop('draft_artifact_spec')
        common['target_artifact_spec_path']=common.pop('target_artifact_spec')
        return run_campaign(**common,draft_model_id='longlive_1.3b',
            boundary_config_path=args.boundary_config,baseline_population_v2=True,
            selection_strategy='rank_burst',burst_calibration_revision=preflight.CALIBRATION_SOURCE,
            sealed_calibration_ranks=True,record_video_ready_timing=True,
            runtime_launcher=preflight.launch_runtime)
    from . import native_video_ready_target as reference
    with tracked_reference_runtime(reference):
        return reference.run(**common,native_draft_model_id='longlive_1.3b',pure_draft=args.method=='draft')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','boundary-config','vbench-checkout',
                 'vbench-python','vbench-asset-manifest','vbench-asset-spec'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--duration-seconds',type=int,choices=(15,30),required=True)
    args=p.parse_args(argv);validate_options(args.method,args.duration_seconds)
    from .stage_timing import _unsealed_directory
    from .metadata import find_git_root,collect_git_metadata
    from .stage_timing_probe import _hardware
    from .accepted_stream import stream_devices
    from .runtime_helpers import shutdown_owned_runtime_helpers
    from .paper_population import load
    root=_unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']));repo=find_git_root(Path.cwd())
    hardware=_hardware((stream_devices()[0],))
    require(len(hardware['gpus'])==1 and hardware['gpus'][0]['name']=='NVIDIA H200','one actual H200 required')
    admission=admit_preflight(repo,os.environ[PREFLIGHT_ENV],hardware,
        os.environ.get('CFEXP_LONGLIVE_FULLVIDEO_PREFLIGHT_PLAN_SHA256'))
    normal=preflight.admit_normal_gate(repo,os.environ[preflight.NORMAL_ENV])
    ancestor=fusion.admit_gate(repo,os.environ[fusion.GATE_ENV])
    prompts,population=load(repo,model_pair='krea_longlive')
    config=dict(method=args.method,phase='paper128_single_seed_shard',
        git_sha=collect_git_metadata(repo)['sha'],hardware=hardware,population_binding=population,
        admission=admission,ordinary_worker_gate=normal,ancestor_gate=ancestor,
        physical_device_count=1,rank=0,batch_size=1,resolution=[480,832],fps=24,precision='BF16',
        nominal_seconds=args.duration_seconds,frame_count=args.duration_seconds*24-3,
        target_root=args.method!='draft',later_target_budget=4 if args.method=='ours' else None,
        actual_profiler_in_main=False,stage_costs_enabled=True,native_local12_sink3_cleanB3=True,
        warmup_scope='matched operator preflight; model/text initialization excluded by unchanged driver',
        qualitative_comparison=prompts,training_loss='not applicable: inference',
        learning_rate='not applicable: inference',gradient_norm='not applicable: inference',
        backward='not applicable: inference',optimizer='not applicable: inference',
        distributed_communication='none: one physical GPU')
    for label,path in (('draft',args.draft_artifact_spec),('target',args.target_artifact_spec)):
        config[label+'_artifact_spec_sha256']=fusion.sha(path)
    import wandb
    mode=tracking_mode(os.environ);directory=root/'tracking/wandb';directory.mkdir(parents=True)
    options=dict(project='collaborative-forcing-reproduction',name=root.name,dir=str(directory),config=config,
        save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,disable_code=True,
        symlink=False,console='off',_disable_stats=True))
    fallback=False
    try:run=wandb.init(mode=mode,**options)
    except Exception:
        if mode!='online':raise
        mode='offline';fallback=True;run=wandb.init(mode=mode,**options)
    write_new(root/'provenance/tracking/session.json',dict(mode=mode,run_id=run.id,
        online_fallback_to_offline=fallback,synced_evidence_verified=False,config=config))
    ok=False
    try:
        result=execute(args)
        from .native_single_gpu import TIMING_SCOPE
        actual_population=result['population'] if args.method=='ours' else result['population_binding']
        require(actual_population==population and result['hardware']==hardware
                and result['timing_scope']==TIMING_SCOPE and result['row_count']==32
                and result['accepted_video_schedule']=='decode_after_generation',
                'actual main execution/population/timing differs from admission')
        if args.method=='ours':rows=log_result(run,root,result)
        else:
            rows=result['rows']
            for row in rows:
                run.log({'quality/'+k:v for k,v in row['quality_scores'].items()})
                run.log({'generation_seed':row['seed'],'prompt_id':row['prompt_id'],
                    'timing/video_ready_seconds':row['video_ready_wall_seconds'],
                    'timing/generation_seconds':row['generation_wall_seconds'],
                    'throughput/video_ready_fps':config['frame_count']/row['video_ready_wall_seconds']})
            for path in (root/'provenance/tracking').glob('*.jsonl'):
                for line in path.read_text().splitlines():
                    record=json.loads(line);prefix='worker/'+record['role']+'/'+record['action']
                    require(record['bounded_profile_sample'] is False,'unexpected actual profiler in main')
                    run.log({prefix+'/'+k:record[k] for k in ('request_wall_seconds',
                        'peak_allocated_bytes','peak_reserved_bytes')})
        require({r['prompt_id'] for r in rows}==set(population['selected_prompt_ids'])
                and len(rows)==32 and all(r['seed']==population['seed'] for r in rows),
                'actual main output does not cover its exact frozen seed/shard')
        write_new(root/'provenance/main-admission-completion.json',dict(method=args.method,
            row_count=result['row_count'],admission=admission,hardware=hardware,population=population,
            no_aggregate_or_speedup_claim=True))
        ok=True;return 0
    finally:
        try:shutdown_owned_runtime_helpers(label='longlive-matched-main')
        finally:
            run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
            write_new(root/'provenance/tracking/finished.json',dict(mode=mode,
                execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':raise SystemExit(main())
