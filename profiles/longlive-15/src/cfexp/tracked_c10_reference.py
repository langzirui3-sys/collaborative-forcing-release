"""Pure native controls on the original C10 development cases, never paper128."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from . import accepted_stream_target as reference
from .native_c10_development import reference_population, verify_target_gate
from .krea_timestep_adapter import target_four_step_schedule, target_timestep_policy
from .krea_long_session_gate import validate_operator_profile
from .tracked_sdvg import write_new, tracking_mode, materialize_sdk_log_links


def launch_runtime(**kwargs):
    from .accepted_stream_worker import process_factory
    from .correction_runtime import launch_correction_runtime
    if kwargs.pop('process_factory') is not process_factory:
        raise ValueError('C10 reference must retain its native accepted-stream worker')
    from .krea_timestep_adapter import process_factory as schedule_factory
    return launch_correction_runtime(process_factory=schedule_factory, **kwargs)


def verify_reference(root, result, *, pure_draft):
    role='draft' if pure_draft else 'target'
    rows=result['rows']
    if (result.get('population')!='independent_C10_development_not_held_out'
            or result.get('development_only') is not True or result.get('held_out') is not False
            or result.get('may_replace_paper128_results') is not False
            or len(rows)!=10 or len({x['prompt_id'] for x in rows})!=10
            or {x['seed'] for x in rows}!={20260911}
            or {x['prompt_id'] for x in rows}!=set(result['population_binding']['selected_prompt_ids'])):
        raise ValueError('incomplete or mislabeled original C10 pure reference')
    for row in rows:
        if row['arm']!=role:
            raise ValueError('reference row producer differs')
        native=row['native_generation']
        if (native['chunk_count']!=60 or native['draft_model_calls']!=(60 if pure_draft else 0)
                or native['target_model_calls']!=(0 if pure_draft else 60)):
            raise ValueError('reference executed the wrong model call budget')
        path=Path(root)/'provenance'/(role+'-native')/(row['prompt_id']+'.json')
        stored=json.loads(path.read_text())
        if {k:v for k,v in stored.items() if k!='history'}!=native:
            raise ValueError('reference native provenance differs from scored row')
        segments=stored['history']['segments']
        if ([s['chunk_index'] for s in segments]!=list(range(60))
                or any(s['producer_role']!=role for s in segments)):
            raise ValueError('reference history is incomplete or contains another model')
        if pure_draft:
            states=native['native_cache_states']
            if len(states)!=60:
                raise ValueError('reference is missing Draft native KV evidence')
            for i,(state,segment) in enumerate(zip(states,segments)):
                cache=state['cache']
                if (state['chunk_index']!=i or state['latent_sha256']!=segment['latent_sha256']
                        or cache.get('replayed_chunks')!=0 or cache.get('reused_chunks')!=i
                        or cache.get('history_kv_widths')!=[3]*i):
                    raise ValueError('pure Draft lost its native three-frame cache history')
        else:
            chunks=native['development_actual_chunks']
            if (len(chunks)!=60 or native['target_session_count']!=1
                    or native['history_reconstruction_calls']!=0):
                raise ValueError('pure Target lost its native session')
            for i,(chunk,segment) in enumerate(zip(chunks,segments)):
                diag=chunk['diagnostics']
                if (chunk['chunk_index']!=i or chunk['producer_role']!='target'
                        or chunk['latent_sha256']!=segment['latent_sha256']
                        or chunk['initial_noise_descriptor_sha256']!=segment['initial_noise_descriptor_sha256']
                        or diag.get('native_session_block_index')!=i
                        or diag.get('denoising_schedule')!=list(target_four_step_schedule())):
                    raise ValueError('actual pure Target sampler differs from declared policy')
    return dict(prompt_count=10,accepted_blocks=600,model_role=role,
                draft_calls_per_video=60 if pure_draft else 0,
                target_calls_per_video=0 if pure_draft else 60,
                target_updates_per_video=0 if pure_draft else 240,
                all_native_history_and_target_schedule_checks_passed=True)


def profile_reference(root, *, role, config, hardware):
    records=[]
    for path in (Path(root)/'provenance/tracking').glob('*.jsonl'):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    forwards=[x for x in records if x['action'] in ('generate','generate_persistent')]
    action='generate' if role=='draft' else 'generate_persistent'
    if (len(forwards)!=600 or any(x['role']!=role or x['action']!=action for x in forwards)
            or [x.get('chunk_index') for x in forwards]!=list(range(60))*10
            or [x.get('warmup_forward_calls_before') for x in forwards]!=list(range(600))
            or len({x.get('request_id') for x in forwards})!=600
            or len({x.get('pid') for x in forwards})!=1):
        raise ValueError('reference worker coverage differs from the executed pure model')
    samples=[x for x in records if x.get('bounded_profile_sample')]
    if len(samples)!=1:
        raise ValueError('exactly one real executing-model post-warmup profile is required')
    validate_operator_profile(samples[0],role=role,action=action,root=root)
    write_new(Path(root)/'provenance/tracking/profile-summary.json',dict(
        schema='c10_pure_reference_profile_v1',samples=samples,config=config,hardware=hardware,
        executed_generation_role=role,non_generation_roles=['target' if role=='draft' else 'draft'],
        memory_scope='per-process allocator; simultaneous physical peak not measured',
        not_main_video_ready_latency=True))
    return samples[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','boundary-config','vbench-checkout',
                 'vbench-python','vbench-asset-manifest','vbench-asset-spec'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pure-draft',action='store_true')
    args=parser.parse_args(argv)
    from .metadata import find_git_root, collect_git_metadata
    from .stage_timing import _unsealed_directory
    from .runtime_helpers import shutdown_owned_runtime_helpers
    repo=find_git_root(Path.cwd());root=_unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']))
    prompts,binding=reference_population(repo,args.boundary_config,seconds=30,smoke=None)
    gate=verify_target_gate(os.environ.get('CFEXP_TARGET_NATIVE60_GATE_RUN'),repo,
        plan_sha256=os.environ.get('CFEXP_TARGET_NATIVE60_GATE_PLAN_SHA256'))
    write_new(root/'provenance/native60-target-admission.json',gate)
    role='draft' if args.pure_draft else 'target'
    config=dict(method='pure_'+role,phase='independent_C10_reference_development',
        git_sha=collect_git_metadata(repo)['sha'],development_only=True,held_out=False,target_gate=gate,
        may_replace_paper128_results=False,prompt_count=10,seed=20260911,population=binding,
        physical_device_count=1,batch_size=1,resolution=[480,832],precision='BF16',
        nominal_seconds=30,chunk_count=60,native_kv=True,executed_generation_role=role,
        target_history_policy='one session; native three-frame clean-context rebase disclosed',
        timestep_policy=target_timestep_policy(),target_schedule=list(target_four_step_schedule()),
        warmup_calls=2,profile_request='third real generating-model request',
        timing='instrumented diagnostic only; no main-table speedup',
        compute_fps='not measured separately; report sampled CUDA stream window and kernel durations',
        training='not applicable: inference',backward='not applicable: inference',
        optimizer='not applicable: inference',loss='not applicable: inference',
        learning_rate='not applicable: inference',gradient_norm='not applicable: inference',
        distributed_communication='none; one physical H200',
        qualitative='all ten original prompt texts, fixed seed and MP4 hashes; no automatic media upload')
    for label,path in (('draft',args.draft_artifact_spec),('target',args.target_artifact_spec)):
        config[label+'_artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        config[label+'_artifact_definition']=json.loads(path.read_text())
    import wandb
    mode=tracking_mode(os.environ);directory=root/'tracking/wandb';directory.mkdir(parents=True)
    kwargs=dict(project='collaborative-forcing-reproduction',name=root.name,config=config,
        dir=str(directory),save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,
            disable_code=True,symlink=False,console='off',_disable_stats=True))
    fallback=False
    try:run=wandb.init(mode=mode,**kwargs)
    except Exception:
        if mode!='online':raise
        mode='offline';fallback=True;run=wandb.init(mode=mode,**kwargs)
    write_new(root/'provenance/tracking/session.json',dict(mode=mode,run_id=run.id,config=config,
        online_fallback_to_offline=fallback,synced_evidence_verified=False,
        online_session_url=run.url if mode=='online' else None))
    ok=False
    try:
        result=reference.run(**vars(args),duration_seconds=30,development_c10=True,
                             runtime_launcher=launch_runtime)
        audit=verify_reference(root,result,pure_draft=args.pure_draft)
        sample=profile_reference(root,role=role,config=config,hardware=result['hardware'])
        for index,row in enumerate(result['rows']):
            run.log(dict(prompt_index=index,generation_seed=row['seed'],
                **{'quality/'+k:v for k,v in row['quality_scores'].items()},
                **{'quality/six_dimension_mean_100':100*sum(row['quality_scores'].values())/6,
                   'timing/diagnostic_video_ready_seconds':row['video_ready_wall_seconds'],
                   'timing/generation_seconds':row['generation_wall_seconds'],
                   'throughput/diagnostic_generation_loop_fps':717/row['generation_wall_seconds'],
                   'throughput/diagnostic_video_ready_fps':717/row['video_ready_wall_seconds'],
                   'timing/model_load_and_renderer_warmup_seconds':row['model_load_and_decoder_warmup_wall_seconds'],
                   'timing/decode_encode_tail_seconds':row['decode_encode_and_prepare_wall_seconds']}))
        run.log({'profile/'+k:sample[k] for k in ('request_wall_seconds','cuda_stream_elapsed_seconds',
            'cuda_kernel_us_sum','cuda_transfer_us_sum','peak_allocated_bytes','peak_reserved_bytes',
            'stream_window_frame_equivalents_per_second')})
        write_new(root/'provenance/c10-reference-completion.json',dict(
            execution=audit,development_only=True,may_replace_paper128_results=False))
        ok=True;return 0
    finally:
        try:shutdown_owned_runtime_helpers(label='c10-reference-controller')
        finally:
            run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
            write_new(root/'provenance/tracking/finished.json',dict(
                mode=mode,execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':
    raise SystemExit(main())
