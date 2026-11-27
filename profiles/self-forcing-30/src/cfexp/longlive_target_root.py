"""One-prompt Teacher-first ablation; retain original5s/10s correction budgets.

All source runs remain immutable. Three new arms isolate Target's first chunk
from subsequent Target bursts, using the same native actual noise as the
sealed tea case. Static text is retained; camera motion is not a quality label.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import time

from .cross_model_delay import SEED,load_plan,require,write_new
from .incremental_draft import latent_digest

ROOT_ARMS={
    'target_root_only':(0,),
    'target_root_plus_5s':(0,10,11,12),
    'target_root_plus_10s':(0,20,21,22),
}


def validate_plan(path,expected):
    from .cross_model_delay import sha
    require(sha(path)==expected,'root protocol digest drift')
    plan=json.loads(path.read_text())
    require(plan['schema']=='longlive_target_root_v1' and plan['seed']==SEED
            and plan['prompt_index']==0 and plan['nominal_seconds']==60,'root trial identity drift')
    require(plan['target_chunks']=={k:list(v) for k,v in ROOT_ARMS.items()},'root budget drift')
    return plan


def run(protocol,protocol_sha):
    from safetensors.torch import save_file,load_file
    from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices,_gpu_uuid,_gpu_compute_pids
    from .metadata import find_git_root
    from .main_table_merge import _load_frozen_index
    from .cross_model_longlive import launch_target,build_longlive,observe_native,save_arm,target_request
    output=_run_dir();repository=find_git_root(Path.cwd());model_root=_model_root()
    require(not output.is_relative_to(repository),'output inside checkout')
    plan=validate_plan(protocol,protocol_sha)
    _,item=load_plan(repository/plan['base_protocol'],plan['base_protocol_sha256'],'longlive_krea',0)
    source=Path(plan['source_run']);frozen,seal=_load_frozen_index(source)
    require(seal['artifact_index_sha256']==plan['source_index_sha256']
            and seal['git_sha']=='b60fe73858bb339c0afc1240125192e0a09c7ce0','original tea seal drift')
    base=source/'attempts'/seal['attempt_id']
    def bound(relative):return frozen.file((base/relative).relative_to(source).as_posix(),label='teacher-root baseline')
    old=json.loads(bound('result.json').read_text())
    native_trace=json.loads(bound('arms/draft_only/trace.json').read_text())
    from .lingbot_reset_probe import load_trace_bound_latent
    target_trace=json.loads(bound('arms/target_only/trace.json').read_text())
    import torch
    first_target=load_trace_bound_latent(base,'target_only',0,target_trace['chunks'][0],torch,longlive=True)
    require(old['pair']=='longlive_krea' and old['prompt_index']==0 and old['prompt']==item,'original prompt drift')
    devices=_visible_devices();require(len(devices)==2 and len(set(devices))==2,'two GPUs required')
    uuids=[_gpu_uuid(d) for d in devices]
    require(all(not _gpu_compute_pids(u) for u in uuids),'GPU occupied')
    def ownership(pid):
        expected=[{os.getpid()},{pid} if pid is not None else {os.getpid()}]
        require(all(_gpu_compute_pids(u)<=pids for u,pids in zip(uuids,expected,strict=True)),'foreign GPU process')
    write_new(output/'preflight.json',{'protocol_sha256':protocol_sha,'prompt':item,
        'source_index_sha256':plan['source_index_sha256'],'gpu_uuids':uuids,
        'local_latent_binding':'BF16 tensor content digest in indexed trace; source safetensors not indexed',
        'rank_routes_generation':False,'first_target_compared_to_original_pure_target':True})
    initialization=time.perf_counter();worker,target_records=launch_target(output,repository,model_root,devices[1])
    prior=os.environ.get('CUDA_VISIBLE_DEVICES');backend=None
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=devices[0]
        backend,records=build_longlive(output,repository,model_root);t=backend.torch
        initialization=time.perf_counter()-initialization
        t.manual_seed(SEED);t.cuda.manual_seed_all(SEED)
        noise=t.randn((1,360,16,60,104),device=backend.device,dtype=t.bfloat16)
        noise_hash=latent_digest(noise,t)
        require(noise_hash==old['initial_noise_sha256'],'original initial noise differs')
        write_new(output/'provenance.json',{'draft_artifacts':records,'target_artifacts':target_records,
            'native_load':backend.native_load,'initial_noise_sha256':noise_hash})
        (output/'inputs').mkdir()
        for i in range(120):save_file({'noise':noise[:,i*3:(i+1)*3].cpu().contiguous()},str(output/'inputs'/f'{i:03d}.safetensors'))
        with t.inference_mode():
            v,z=backend.pipeline.inference(noise=noise[:,:3],text_prompts=[item['prompt']],return_latents=True,profile=False,low_memory=False)
        del v,z
        backend.pipeline.vae.model.clear_cache();backend.pipeline.kv_cache1=None;backend.pipeline.crossattn_cache=None
        warm=output/'warmup';warm.mkdir()
        target_request(worker,output,warm,item['prompt'],0,output/'inputs/000.safetensors',persistent=False)
        results=[];baseline_chunks=None;rank_rows=[]
        for arm,targets in ROOT_ARMS.items():
            t.manual_seed(SEED);t.cuda.manual_seed_all(SEED)
            check=t.randn(noise.shape,device=backend.device,dtype=t.bfloat16)
            require(latent_digest(check,t)==noise_hash,'initial RNG drift');del check
            ownership(worker.process.pid)
            data=observe_native(backend,worker,arm=arm,prompt=item['prompt'],output=output,noise=noise,
                reference=None,ownership=lambda:ownership(worker.process.pid),target_chunks=targets)
            directory,frames,rows,stages=data
            require(t.equal(load_file(str(directory/'000.safetensors'))['latent'],first_target),
                    'Teacher root differs from sealed native Target root')
            trace=json.loads((directory/'trace.json').read_text())
            require(trace['renoise']==native_trace['renoise'],'actual stochastic inputs differ from original')
            if baseline_chunks is None:baseline_chunks=rows
            else:
                cutoff=targets[1]
                require([x['latent_sha256'] for x in rows[:cutoff]]==[
                    x['latent_sha256'] for x in baseline_chunks[:cutoff]],'Teacher-first pre-burst prefix differs')
            result=save_arm(backend,directory,frames,rows,stages)
            result['warm_pipeline_plus_encode_seconds']=result['pipeline_with_monitoring_wall_seconds']+result['encode_seconds']
            results.append(result)
            for row in rows:
                rank=row['erank_observation'];rank_rows.append([arm,row['chunk'],*row['video_interval_seconds'],
                    rank['producer'],rank['rank']])
            require(latent_digest(noise,t)==noise_hash,'initial noise mutated')
            del frames;gc.collect();t.cuda.empty_cache()
        with (output/'erank.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['arm','chunk','start_seconds','end_seconds','producer','hidden_erank']);w.writerows(rank_rows)
        require(len(rank_rows)==360,'all new native chunk ranks required')
        write_new(output/'result.json',{'schema':'longlive_target_root_result_v1','prompt':item,
            'seed':SEED,'protocol_sha256':protocol_sha,'arms':results,'initial_noise_sha256':noise_hash,
            'initialization_seconds':initialization,'original_comparison_result':old,
            'all_roots_equal_original_target':True,'all_actual_noises_equal_original':True,
            'all_shared_pre_burst_prefixes_exact':True,'visual_status':'pending independent review',
            'note':'A still-frame prompt; camera motion alone is not quality or a recovery endpoint.'})
    finally:
        if backend is not None:backend.close()
        worker.close()
        if prior is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=prior


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True);a=p.parse_args()
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:run(a.protocol,a.protocol_sha256)
    finally:shutdown_owned_runtime_helpers(label='longlive-target-root')


if __name__=='__main__':main()
