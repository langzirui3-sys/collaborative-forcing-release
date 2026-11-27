"""Full30s Target execution parity, not quality or a main timing result."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .correction_state import LatentHistory
from .insight2_campaign import ChunkGenerationRequest,PromptCase
from .krea_chunk_backend import validate_native_correction_state
from .krea_timestep_adapter import target_four_step_schedule, target_timestep_policy
from .krea_session_gate import PROMPT,SEED
from .tracked_sdvg import write_new,tracking_mode,materialize_sdk_log_links

BLOCKS=60


def validate_operator_profile(sample, *, role, action, root):
    """Check measured fields and trace while preserving the real request action."""
    root=Path(root)
    if (sample.get('bounded_profile_sample') is not True or sample.get('action')!=action
            or sample.get('role')!=role or sample.get('rank')!=0
            or type(sample.get('warmup_forward_calls_before')) is not int
            or sample['warmup_forward_calls_before']<2):
        raise ValueError('sample was not the declared post-warmup operator')
    for key in ('request_wall_seconds','cuda_stream_elapsed_seconds','cuda_kernel_us_sum',
                'peak_allocated_bytes','peak_reserved_bytes'):
        x=sample.get(key)
        if not isinstance(x,(float,int)) or isinstance(x,bool) or not math.isfinite(x) or x<=0:
            raise ValueError('missing or invalid measured sample: '+key)
    if sample['peak_reserved_bytes']<sample['peak_allocated_bytes']:
        raise ValueError('allocator peaks are inconsistent')
    relative=Path(sample.get('trace_path',''))
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:2]!=('tracking','profiler'):
        raise ValueError('unsafe profiler trace path')
    trace=root/relative
    if (any(p.is_symlink() for p in (trace,*trace.parents)) or not trace.is_file()
            or hashlib.sha256(trace.read_bytes()).hexdigest()!=sample.get('trace_sha256')):
        raise ValueError('profiler trace differs from measured sample')


def validate_profiles(records, *, pids, root):
    """Validate each real operator without relabelling its recorded action.

    The SDVG profile reader intentionally accepts persistent Target requests
    only. This diagnostic also measures the existing correction RPC, whose
    actual action is ``generate`` and must remain recorded as such.
    """
    root=Path(root)
    if set(pids)!={'native','corrector'} or len(set(pids.values()))!=2:
        raise ValueError('two independent sequential Target worker identities required')
    samples=[r for r in records if r.get('bounded_profile_sample')]
    if len(samples)!=2 or {s['pid'] for s in samples}!=set(pids.values()):
        raise ValueError('both real paths require an independent post-warmup CUDA sample')
    for label,pid in pids.items():
        action='generate_persistent' if label=='native' else 'generate'
        group=[r for r in records if r.get('pid')==pid]
        forwards=[r for r in group if r.get('action') in ('generate','generate_persistent')]
        if (len(forwards)!=BLOCKS or any(r.get('action')!=action or r.get('role')!='target'
                or r.get('rank')!=0 for r in forwards)
                or [r.get('chunk_index') for r in forwards]!=list(range(BLOCKS))):
            raise ValueError('actual worker forward coverage/operator differs')
        sample=next(s for s in samples if s['pid']==pid)
        validate_operator_profile(sample,role='target',action=action,root=root)
    return samples


def require_environment(env=None):
    env=os.environ if env is None else env
    target_timestep_policy(env)
    if any(env.get(k)!='1' for k in ('CFEXP_SDVG_TRACKING','CFEXP_SDVG_BOUNDED_PROFILE',
                                    'CFEXP_STAGE_TIMING','CFEXP_ACCEPTED_VIDEO_STREAM')):
        raise ValueError('explicit stream worker opt-in, real profiling and W&B required')


def compare_paths(*,root,case,provider,launch):
    """Use the SAME canonical accepted prefix/noise at every comparison.

    Corrector mismatches never contaminate the next canonical condition. Both
    workers are sequential owners of one physical GPU, not concurrent replicas.
    """
    root=Path(root);scope=hashlib.sha256(b'krea-native60-accepted-prefix-parity-v1').hexdigest()
    prefixes=[];expected=[];pairs=[];calls=[];client=None;identities={}
    history=LatentHistory('native60-canonical',case.prompt_id,SEED)
    def generate(client,history,label,persistent):
        noise_started=time.perf_counter()
        noise=provider.materialize(prompt_id=case.prompt_id,trajectory_seed=SEED,
                                   chunk_index=history.next_chunk_index)
        noise_seconds=time.perf_counter()-noise_started
        req=ChunkGenerationRequest(prompt=case,history=history,noise=noise,
            output_latent_path='latents/'+label+'.safetensors',capture_hidden=False,
            target_logical_call_id=hashlib.sha256(label.encode()).hexdigest(),
            target_call_scope_id=scope,target_call_phase='route_materialization')
        before=time.perf_counter()
        out=(client.generate_persistent if persistent else client.generate)(req)
        wall=time.perf_counter()-before
        if out.diagnostics.get('denoising_schedule')!=list(target_four_step_schedule()):
            raise ValueError('native four-step Target schedule differs')
        diag=out.diagnostics;i=history.next_chunk_index
        if persistent:
            identity=diag.get('native_session_id')
            if (diag.get('native_session_block_index')!=i or diag.get('session_recreated') is not False
                    or diag.get('history_reconstructed') is not False
                    or diag.get('explicit_initial_noise_verified') is not True
                    or diag.get('vae_decode_performed') is not False):
                raise ValueError('pure Target native KV/cursor/noise contract differs')
        else:
            state=diag.get('native_correction_state');validate_native_correction_state(state,chunk_index=i)
            identity=state['native_session_id']
            if i and state['session_recreated']:raise ValueError('corrector recreated its native session')
        if not isinstance(identity,str) or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('native session identity missing')
        if identities.setdefault(persistent,identity)!=identity:raise ValueError('native session owner changed')
        row=dict(label=label,chunk_index=history.next_chunk_index,persistent=persistent,
            accepted_history_head_sha256=history.head_sha256,
            initial_noise_descriptor_sha256=noise.descriptor.descriptor_sha256,
            segment=asdict(out.segment),diagnostics=out.diagnostics,
            parent_request_wall_seconds=wall,parent_noise_materialization_seconds=noise_seconds,
            wall_scope='diagnostic RPC including host/history; not main video-ready timing')
        write_new(root/('parity/'+label+'.json'),row);calls.append(row)
        return out
    try:
        client=launch('native')
        client.start_trajectory(prompt_id=case.prompt_id,prompt=case.prompt,seed=SEED,chunk_count=BLOCKS)
        for i in range(BLOCKS):
            prefixes.append(history)
            item=generate(client,history,f'native-{i:03d}',True)
            expected.append(item);history=history.append(item.segment)
        client.close();client=None
        write_new(root/'parity/native-history.json',history.to_dict())
        client=launch('corrector')
        for i,prefix in enumerate(prefixes):
            actual=generate(client,prefix,f'corrector-{i:03d}',False)
            a,b=calls[i],calls[BLOCKS+i]
            if (a['accepted_history_head_sha256']!=b['accepted_history_head_sha256']
                    or a['initial_noise_descriptor_sha256']!=b['initial_noise_descriptor_sha256']):
                raise ValueError('comparison prefix/noise differs')
            pairs.append(dict(chunk_index=i,latent_exact=actual.segment.latent_sha256==expected[i].segment.latent_sha256,
                native_latent_sha256=expected[i].segment.latent_sha256,
                corrector_latent_sha256=actual.segment.latent_sha256,
                accepted_history_head_sha256=prefix.head_sha256,
                initial_noise_descriptor_sha256=a['initial_noise_descriptor_sha256']))
    finally:
        if client is not None:client.close()
    if len(pairs)!=BLOCKS or len(calls)!=2*BLOCKS:raise ValueError('incomplete full30s parity')
    mismatched=[x['chunk_index'] for x in pairs if not x['latent_exact']]
    return dict(schema='krea_long_session_parity_v1',passed=not mismatched,chunk_count=BLOCKS,
        timestep_policy=target_timestep_policy(),denoising_schedule=list(target_four_step_schedule()),
        prompt=case.prompt,seed=SEED,canonical_prefix_comparisons=pairs,requests=calls,
        first_mismatch_chunk=mismatched[0] if mismatched else None,mismatched_chunks=mismatched,
        no_quality_or_video_ready_speedup_claim=True,
        claim_scope='full30s native versus accepted-history corrector, same canonical conditions; no Draft routing')


def main():
    require_environment()
    from .main_table_baseline_campaign import _run_dir,_model_root,_gpu_uuid,_gpu_compute_pids
    from .metadata import find_git_root,collect_git_metadata
    from .correction_artifacts import verify_target_artifact_spec
    from .correction_runtime import DeterministicNoiseProvider,launch_target_generation_worker
    from .krea_timestep_adapter import process_factory
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]:raise ValueError('one physical GPU required')
    root=_run_dir();repo=find_git_root(Path.cwd());models=_model_root();uuid=_gpu_uuid(devices[0])
    spec=repo/'experiments/insight2-krea14b.artifacts.json'
    _,artifacts,backend=verify_target_artifact_spec(spec,model_root=models,repo_root=repo)
    if backend!='cfexp.krea_chunk_backend':raise ValueError('unexpected Target backend')
    digest=hashlib.sha256(spec.read_bytes()).hexdigest()
    config=dict(schema='krea_long_session_parity_config_v1',git_sha=collect_git_metadata(repo)['sha'],
        target_artifact_spec_sha256=digest,physical_device_count=1,gpu_uuid=uuid,batch_size=1,
        resolution=[480,832],precision='BF16',chunk_count=BLOCKS,native_denoising_schedule=list(target_four_step_schedule()),timestep_policy=target_timestep_policy(),
        prompt=PROMPT,seed=SEED,native_kv=True,canonical_prefix_conditioning=True,
        corrector='existing bounded accepted-history correction session; replay disclosed in each native state record',
        sequential_model_owners=True,warmup_forward_calls_per_worker=2,
        profile_scope='third forward of EACH real path after two warmup calls; not E2E or quality',
        decoding='not executed: latent parity only',backward='not applicable: inference',
        optimizer='not applicable: inference',distributed_communication='none: single physical GPU')
    import wandb
    mode=tracking_mode(os.environ);directory=root/'tracking/wandb';directory.mkdir(parents=True)
    kwargs=dict(project='collaborative-forcing-reproduction',name=root.name,dir=str(directory),config=config,
        save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,disable_code=True,
                                              symlink=False,console='off',_disable_stats=True))
    fallback=False
    try:run=wandb.init(mode=mode,**kwargs)
    except Exception:
        if mode!='online':raise
        mode='offline';fallback=True;run=wandb.init(mode=mode,**kwargs)
    write_new(root/'provenance/tracking/session.json',dict(schema='wandb_session_v1',mode=mode,
        run_id=run.id,config=config,directory='tracking/wandb',synced_evidence_verified=False,
        online_fallback_to_offline=fallback,online_session_url=run.url if mode=='online' else None))
    pids={};load_seconds={}
    def launch(label):
        if _gpu_compute_pids(uuid):raise ValueError('assigned GPU was not released before path '+label)
        before=time.perf_counter()
        client=launch_target_generation_worker(run_dir=root,repo_root=repo,target_artifact_spec_path=spec,
            target_artifact_spec_sha256=digest,target_verified_artifacts=artifacts,visible_device=devices[0],
            chunk_count=BLOCKS,process_factory=process_factory)
        load_seconds[label]=time.perf_counter()-before
        pids[label]=client.process.pid
        return client
    ok=False
    try:
        result=compare_paths(root=root,case=PromptCase('native60-target-parity',PROMPT,'evaluation'),
            provider=DeterministicNoiseProvider(run_dir=root),launch=launch)
        records=[]
        for p in (root/'provenance/tracking').glob('*.jsonl'):
            records.extend(json.loads(x) for x in p.read_text().splitlines())
        samples=validate_profiles(records,pids=pids,root=root)
        for label,pid in pids.items():
            sample=next(s for s in samples if s['pid']==pid)
            run.log({f'{label}/model_load_seconds':load_seconds[label],
                f'{label}/cuda_kernel_us_sum':sample['cuda_kernel_us_sum'],
                f'{label}/cuda_transfer_us_sum':sample['cuda_transfer_us_sum'],
                f'{label}/request_wall_seconds':sample['request_wall_seconds'],
                f'{label}/cuda_stream_seconds':sample['cuda_stream_elapsed_seconds'],
                f'{label}/peak_allocated_bytes':sample['peak_allocated_bytes'],
                f'{label}/peak_reserved_bytes':sample['peak_reserved_bytes'],
                f'{label}/frame_equivalents_per_second':sample['stream_window_frame_equivalents_per_second']})
        for row in result['requests']:
            label='native' if row['persistent'] else 'corrector'
            run.log({f'{label}/chunk_index':row['chunk_index'],
                f'{label}/parent_rpc_wall_seconds':row['parent_request_wall_seconds'],
                f'{label}/parent_noise_materialization_seconds':row['parent_noise_materialization_seconds']})
        result.update(status='succeeded' if result['passed'] else 'mismatch',hardware=config,
            worker_pids=pids,model_load_seconds=load_seconds,profile_samples=samples,worker_observations=len(records),
            wandb_mode=mode,wandb_run_id=run.id,wandb_synced_evidence_verified=False)
        write_new(root/'result.json',result);run.log({'parity/mismatched_chunks':len(result['mismatched_chunks'])})
        ok=result['passed'];return 0 if ok else 1
    finally:
        run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
        write_new(root/'provenance/tracking/finished.json',dict(schema='wandb_finish_v1',mode=mode,
            execution_succeeded=ok,synced_evidence_verified=False))

if __name__=='__main__':raise SystemExit(main())
