"""One-H200 native scoring-decoder rollback parity gate, not SDVG quality.

Uses sealed latent fixtures and a fixed diagnostic route. It does not execute
ImageReward or diffusion generation, and cannot establish SDVG speed or quality.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

from .sdvg_vae_transaction import (AcceptedVaeTransaction,FIELDS,clone_state,
    native_owner_binding,require,sha,state_values)

ENV='CFEXP_SDVG_ACCEPTED_VAE_GATE'
DECISIONS=(False,True,False,False,True,True,False,True,False,False,True,True)
SOURCE_SCHEMA='sdvg_accepted_vae_gate_source_v1'


def equal_state(torch,left,right):
    """Exact tensors, nested containers and native scalar cursor metadata."""
    if isinstance(left,torch.Tensor):
        return (isinstance(right,torch.Tensor) and left.dtype==right.dtype
            and left.device==right.device and left.shape==right.shape and torch.equal(left,right))
    if type(left) is not type(right):return False
    if type(left) is dict:
        return left.keys()==right.keys() and all(equal_state(torch,left[k],right[k]) for k in left)
    if type(left) in (list,tuple):
        return len(left)==len(right) and all(equal_state(torch,x,y) for x,y in zip(left,right))
    return left==right


def exercise(torch,models,scale,fixtures,*,session_id,clock=time.perf_counter):
    """Independent accepted-only decoder is the oracle, never a cache copy."""
    from .streaming_vae_probe import to_pixels
    owner,oracle=models
    for model in models:model.clear_cache()
    tx=AcceptedVaeTransaction(owner=owner,torch=torch,session_id=session_id)
    rows=[]
    def measure(function):
        torch.cuda.synchronize();started=clock();value=function();torch.cuda.synchronize()
        return value,clock()-started
    for index,(candidate,selected,bindings) in enumerate(fixtures):
        accept=DECISIONS[index]
        rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone()
        (proposal,pending),probe_seconds=measure(lambda:tx.probe(chunk_index=index,
            candidate_sha256=bindings['draft'],decode_and_score=lambda:owner.cached_decode(candidate,scale)))
        (replacement,receipt),commit_seconds=measure(lambda:tx.commit(
            pending_token=pending['pending_token'],accepted=accept,
            selected_sha256=bindings['draft' if accept else 'selected'],
            decode_selected=None if accept else lambda:owner.cached_decode(selected,scale)))
        expected,oracle_seconds=measure(lambda:oracle.cached_decode(candidate if accept else selected,scale))
        actual=proposal if accept else replacement
        require(actual.dtype==expected.dtype and actual.shape==expected.shape and torch.equal(actual,expected),
                'accepted-history decoder float RGB differs at block '+str(index))
        pixels=to_pixels(torch,actual)
        require(torch.equal(pixels,to_pixels(torch,expected)),
                'accepted-history decoder quantized RGB differs at block '+str(index))
        require(equal_state(torch,state_values(owner),state_values(oracle)),
                'complete native decoder cache differs at block '+str(index))
        require(torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state()),
                'decoder transaction changed RNG state')
        rows.append(dict(chunk_index=index,accepted_draft=accept,float_rgb_exact=True,pixels_exact=True,
            all_decoder_fields_exact=list(FIELDS),rgb_frames=int(pixels.shape[0]),
            probe_including_candidate_decode_seconds=probe_seconds,
            commit_including_restore_and_selected_decode_seconds=commit_seconds,
            independent_oracle_decode_seconds=oracle_seconds,receipt=receipt,
            rgb_u8_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),rng_unchanged=True))
        del actual,proposal,replacement,expected,pixels,pending
    tx.finish(expected_chunks=len(DECISIONS))
    return rows


def load_fixtures(config_path,*,torch):
    from .main_table_merge import _load_frozen_index
    from .main_table_baseline_campaign import _sha256_file
    from .correction_state import LatentHistory
    from safetensors.torch import load_file
    require(config_path.is_file() and not config_path.is_symlink(),'invalid VAE gate source config')
    config=json.loads(config_path.read_text())
    require(set(config)=={'schema','source_run','artifact_index_sha256','prompt_id'}
        and config['schema']==SOURCE_SCHEMA,'invalid VAE gate source schema')
    root=Path(config['source_run'])
    require(root.is_absolute() and root.resolve(strict=True)==root and not root.is_symlink(),
            'VAE gate requires a canonical sealed source directory')
    require(_sha256_file(root/'artifact-index.json')[0]==config['artifact_index_sha256'],
            'VAE gate fixture index changed')
    index,marker=_load_frozen_index(root)
    prefix='attempts/'+marker['attempt_id']+'/'
    prompt=config['prompt_id'];require('/' not in prompt and '\\' not in prompt and prompt,'invalid fixture prompt id')
    histories={}
    for label in ('draft','selected'):
        path=index.file(prefix+'provenance/sdvg/'+prompt+'-'+label+'-history.json',label='sealed VAE fixture history')
        history=LatentHistory.from_dict(json.loads(path.read_text()));history.verify_files(root/prefix)
        require(history.prompt_id==prompt and history.trajectory_seed==20260911
            and history.next_chunk_index==30,'VAE gate fixture cohort differs')
        histories[label]=history
    fixtures=[]
    for i in range(len(DECISIONS)):
        row=[];bindings={}
        for label in ('draft','selected'):
            segment=histories[label].segments[i]
            values=load_file(str(root/prefix/segment.latent_path),device='cpu')
            latent=values.get('latent')
            require(set(values)=={'latent'} and latent is not None and latent.dtype==torch.bfloat16
                and tuple(latent.shape)==(1,3,16,60,104) and bool(torch.isfinite(latent).all()),
                'sealed VAE fixture tensor violates native geometry')
            row.append(latent.permute(0,2,1,3,4).contiguous());bindings[label]=segment.latent_sha256
        fixtures.append((*row,bindings))
    require(fixtures[0][2]['draft']!=fixtures[0][2]['selected'],
            'root rejection must exercise an actual different Target latent')
    return fixtures,dict(config=config,config_sha256=_sha256_file(config_path)[0],marker=marker,
        history_manifest_sha256={k:v.manifest_sha256 for k,v in histories.items()},
        fixed_route=list(DECISIONS),reward_routing_executed=False)


def run_gate(root,repo,model_root,*,source_config,wandb_run):
    from .native_vae_component import components
    from .main_table_baseline_campaign import _gpu_uuid,_gpu_compute_pids
    from .stage_timing_probe import _hardware
    from .tracked_sdvg import write_new
    from .kernel_profile import summarize_events,toolchain
    import torch
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    require(len(devices)==1 and devices[0].isdigit(),'VAE gate requires one explicit physical GPU')
    hardware=_hardware(tuple(devices));require(len(hardware['gpus'])==1
        and hardware['gpus'][0]['name']=='NVIDIA H200','VAE gate requires one actual H200')
    def exclusive():
        require(_gpu_compute_pids(_gpu_uuid(devices[0]))<={os.getpid()},
                'VAE gate GPU acquired an unrelated compute process')
    exclusive();require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one CUDA device required')
    stages={};started=time.perf_counter()
    fixtures,source=load_fixtures(source_config,torch=torch)
    stages['sealed_fixture_read_seconds']=time.perf_counter()-started
    source_bytes,constants,weights,identity=components('longlive',repository=repo,model_root=model_root)
    path=root/'tmp'/'native-vae.py';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:handle.write(source_bytes)
    spec=importlib.util.spec_from_file_location('cfexp_sdvg_accepted_vae_native',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    started=time.perf_counter()
    models=[module._video_vae(pretrained_path=str(weights),z_dim=16).eval().requires_grad_(False)
            .to(dtype=torch.bfloat16,device='cuda') for _ in range(2)]
    torch.cuda.synchronize();stages['two_native_vae_load_seconds']=time.perf_counter()-started
    bindings=[native_owner_binding(model) for model in models]
    scale=[torch.tensor(constants['mean'],device='cuda',dtype=torch.bfloat16),
           1/torch.tensor(constants['std'],device='cuda',dtype=torch.bfloat16)]
    started=time.perf_counter()
    fixtures=[(a.to('cuda'),b.to('cuda'),c) for a,b,c in fixtures]
    torch.cuda.synchronize();stages['fixture_host_to_device_seconds']=time.perf_counter()-started
    require(torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(),
            'native CUDA profiler is unavailable')
    samples=[];trace=root/'tracking'/'profiler'/'native-vae-transaction.trace.json'
    trace.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        # Two complete same-shaped mixed-route warmups, never zero-sized stand-ins.
        for i in range(2):
            exclusive();exercise(torch,models,scale,fixtures,session_id=sha(['warmup',i,source]))
        for sample in range(2):
            exclusive();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            if sample==0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as profiler:
                    rows=exercise(torch,models,scale,fixtures,session_id=sha(['sample',sample,source]))
                profiler.export_chrome_trace(str(trace));kernels=summarize_events(profiler.events())
                require(kernels and sum(k['cuda_us'] for k in kernels if k['kind']=='kernel')>0,
                        'native VAE profile captured no CUDA kernels')
            else:rows=exercise(torch,models,scale,fixtures,session_id=sha(['sample',sample,source]))
            elapsed=time.perf_counter()-started
            costs=[r['probe_including_candidate_decode_seconds']+
                   r['commit_including_restore_and_selected_decode_seconds'] for r in rows]
            record=dict(sample=sample,instrumented=sample==0,warmup_full_trajectories=2,
                wall_seconds_including_oracle_and_parity=elapsed,
                transaction_stage_seconds_sum=sum(costs),transaction_chunk_latency_median_seconds=statistics.median(costs),
                transaction_rgb_frames_per_second=sum(r['rgb_frames'] for r in rows)/sum(costs),
                throughput_scope='synchronized decoder transaction stages only; excludes oracle/parity, not video-ready FPS',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                memory_scope='single process with two native decoders, fixtures, snapshots, output/oracle tensors',rows=rows)
            samples.append(record)
            wandb_run.log({'correctness/all_selected_rgb_cache_exact':1,'sample':sample,
                'profile/transaction_chunk_latency_median_seconds':record['transaction_chunk_latency_median_seconds'],
                'profile/transaction_rgb_frames_per_second':record['transaction_rgb_frames_per_second'],
                'profile/peak_allocated_bytes':record['peak_allocated_bytes'],
                'profile/peak_reserved_bytes':record['peak_reserved_bytes']})
    report=dict(schema='sdvg_accepted_vae_native_gate_v1',passed=True,hardware=hardware,
        native_component=identity,owner_bindings=bindings,fixture_source=source,
        physical_device_count=1,batch_size=1,resolution=[480,832],precision='bfloat16',
        stage_costs=stages,samples=samples,toolchain=toolchain(torch),kernels=kernels,
        trace=dict(path=trace.relative_to(root).as_posix(),sha256=hashlib.sha256(trace.read_bytes()).hexdigest()),
        quality_measured=False,formal_speedup_claim=False,reward_model_executed=False,
        diffusion_generation_executed=False,production_sdvg_enabled=False)
    write_new(root/'sdvg-accepted-vae-native-gate.json',report)
    return report


def main():
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root,collect_git_metadata
    from .tracked_sdvg import tracking_mode,materialize_sdk_log_links,write_new
    from .stage_timing import _unsealed_directory
    from .streaming_vae_probe import VAE_SHA,WEIGHTS_SHA
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-config',type=Path,required=True)
    args=parser.parse_args();require(os.environ.get(ENV)=='1','native VAE transaction gate requires explicit opt-in')
    root=_unsealed_directory(_run_dir());repo=find_git_root(Path.cwd())
    config=dict(method='SDVG accepted-history VAE cache transaction diagnostic',phase='bounded_profile_not_main',
        git_sha=collect_git_metadata(repo)['sha'],physical_device_count=1,batch_size=1,resolution=[480,832],
        precision='bfloat16',fixed_accept_draft=list(DECISIONS),warmup_trajectories=2,
        no_quality_or_speedup_claim=True,reward_model_executed=False,diffusion_generation_executed=False,
        backward='not applicable: inference',optimizer='not applicable: inference',
        encoding='not applicable: float/uint8 parity only',distributed_communication='not applicable: one device',
        checkpoint_sha256=WEIGHTS_SHA,native_vae_source_sha256=VAE_SHA,
        source_config_sha256=hashlib.sha256(args.source_config.read_bytes()).hexdigest())
    import wandb
    mode=tracking_mode(os.environ);directory=root/'tracking'/'wandb';directory.mkdir(parents=True,exist_ok=True)
    kwargs=dict(project='collaborative-forcing-reproduction',name=root.name,dir=str(directory),config=config,
        save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,disable_code=True,symlink=False,
                                             console='off',_disable_stats=True))
    fallback=False
    try:run=wandb.init(mode=mode,**kwargs)
    except Exception:
        if mode!='online':raise
        fallback=True;mode='offline';run=wandb.init(mode=mode,**kwargs)
    write_new(root/'provenance'/'tracking'/'session.json',dict(schema='wandb_session_v1',mode=mode,
        run_id=run.id,config=config,directory='tracking/wandb',online_fallback_to_offline=fallback,
        synced_evidence_verified=False))
    ok=False
    try:
        run_gate(root,repo,_model_root(),source_config=args.source_config,wandb_run=run);ok=True;return 0
    finally:
        run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
        write_new(root/'provenance'/'tracking'/'finished.json',dict(schema='wandb_finish_v1',mode=mode,
            execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':raise SystemExit(main())
