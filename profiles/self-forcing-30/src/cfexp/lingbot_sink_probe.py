"""Causal attention-history ablation after an exact 50-chunk native prefix.

The two ablations retain equal numbers of attention tokens. Neither modifies
cache writes, positions, input images, text, or sampling noise. This is a
single-case diagnostic, not evidence of general quality or KV numerical error.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path

from .cross_model_delay import load_plan, require, sha, write_new

SOURCE_RUN='cross-model-delay-lingbot-world-2-prompt-03-60s-ntu-v1-20261027T175546Z-f1cffce2-6f001157-d838fa36'
SOURCE_INDEX='0b1ab20f94ed8ea087e7d693c5ccf4e92ee58382c30ad674efae481fec8ebb02'
VARIANTS=('control','drop_initial_sink','drop_middle_history')
START=50


def kept_frames(variant,chunk):
    require(variant in VARIANTS and type(chunk) is int and 0<=chunk<60,'invalid ablation coordinate')
    if chunk<START or variant=='control':return tuple(range(18))
    if variant=='drop_initial_sink':return tuple(range(6,18))
    return tuple(range(6))+tuple(range(12,18))


def select_history(k,v,variant,chunk,frame,torch):
    require(k.shape[1]==v.shape[1]==18*frame and frame>0,'native attention geometry drift')
    keep=kept_frames(variant,chunk)
    if keep==tuple(range(18)):return k,v
    indices=torch.cat([torch.arange(i*frame,(i+1)*frame,device=k.device) for i in keep])
    return k.index_select(1,indices),v.index_select(1,indices)


def run(protocol,protocol_sha):
    from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices,_gpu_uuid,_gpu_compute_pids
    from .metadata import find_git_root
    from .main_table_merge import _load_frozen_index
    from .stage_timing import _unsealed_directory
    from . import cross_model_lingbot as native
    output=_run_dir();_unsealed_directory(output);repository=find_git_root(Path.cwd())
    require(not output.is_relative_to(repository),'output inside repository')
    plan,item=load_plan(protocol,protocol_sha,'lingbot_world_2',3)
    source=output.parents[2]/SOURCE_RUN
    frozen,seal=_load_frozen_index(source)
    require(seal['artifact_index_sha256']==SOURCE_INDEX,'source index drift')
    base=source/'attempts'/seal['attempt_id']
    path=(base/'arms/draft_only/trace.json').relative_to(source).as_posix()
    original=json.loads(frozen.file(path,label='sink source trace').read_text())
    require(len(original['chunks'])==60,'incomplete source trace')
    devices=_visible_devices();require(len(devices)==2 and len(set(devices))==2,'two distinct leased GPUs required')
    uuids=[_gpu_uuid(d) for d in devices]
    def ownership():
        require(all(_gpu_compute_pids(u)<={os.getpid()} for u in uuids),'foreign GPU process')
    require(all(not _gpu_compute_pids(u) for u in uuids),'GPU occupied')
    write_new(output/'preflight.json',{'source_run':str(source),'source_index_sha256':SOURCE_INDEX,
        'protocol_sha256':protocol_sha,'prompt':item,'gpu_uuids':uuids,
        'intervention_start_chunk':START,'variants':VARIANTS,
        'claim_limit':'single-case attention-history intervention; not proof of KV numerical error'})
    torch,pipes,prepared,receipt=native.build_pair(output,repository,_model_root())
    action=prepared/item['lingbot_example']
    write_new(output/'input-binding.json',{'image_sha256':sha(action/'image.jpg'),'input_receipt':receipt})
    from PIL import Image
    for pipe in pipes.values():
        with Image.open(action/'image.jpg') as image,torch.cuda.device(pipe.device):
            pipe.prewarm(image.convert('RGB'),max_area=480*832,frame_num=957,chunk_size=4)
    module=importlib.import_module(type(pipes['1.3b'].model).__module__)
    original_attention=module.attention;OriginalRouter=native.NativeRouter
    traces={};reports=[]
    try:
        for variant in VARIANTS:
            ownership();directory=output/variant;directory.mkdir()
            calls=[];cursor={'chunk':None,'frame_tokens':None}
            def attention(q,k,v,*args,**kw):
                chunk=cursor['chunk'];frame=cursor['frame_tokens']
                require(chunk is not None and frame is not None,'attention without native cursor')
                if chunk>=START:
                    require(k.shape[1]==v.shape[1]==18*frame,'native attention geometry drift')
                    keep=kept_frames(variant,chunk)
                    k,v=select_history(k,v,variant,chunk,frame,torch)
                    calls.append({'chunk':chunk,'retained_local_frames':keep,'query_tokens':q.shape[1],
                                  'key_tokens':k.shape[1]})
                return original_attention(q,k,v,*args,**kw)
            class Router(OriginalRouter):
                def forward(self,*args,**kw):
                    cursor['chunk']=kw['current_start']//(4*kw['frame_seqlen'])
                    cursor['frame_tokens']=kw['frame_seqlen']
                    return super().forward(*args,**kw)
            module.attention=attention;native.NativeRouter=Router
            try:
                result=native.generate_arm(pipes=pipes,torch=torch,item=item,action=action,output=directory,
                    arm='draft_only',reference=None,ownership=ownership)
            finally:module.attention=original_attention;native.NativeRouter=OriginalRouter
            trace=json.loads((directory/'arms/draft_only/trace.json').read_text())
            require(trace['renoise']==original['renoise'] and trace['initial_chunk_noise_sha256']==original['initial_chunk_noise_sha256'], 'actual noise differs from sealed source')
            # Conditioning hashes include max_attention_size, unchanged in all arms.
            require([x['conditioning_sha256'] for x in trace['chunks']]==[x['conditioning_sha256'] for x in original['chunks']], 'conditioning drift')
            cutoff=60 if variant=='control' else START
            require([x['latent_sha256'] for x in trace['chunks'][:cutoff]]==[x['latent_sha256'] for x in original['chunks'][:cutoff]], 'native prefix reproduction failed')
            layers=pipes['1.3b'].model.config.num_layers
            require(len(calls)==(60-START)*5*layers,'attention intervention coverage drift')
            write_new(directory/'attention-policy.json',{'variant':variant,'calls':calls,
                'native_cache_writes_unchanged':True,'exact_source_prefix_chunks':cutoff})
            traces[variant]=trace;reports.append({'variant':variant,'result':result,'exact_source_prefix_chunks':cutoff})
    finally:module.attention=original_attention;native.NativeRouter=OriginalRouter
    write_new(output/'result.json',{'schema':'lingbot_sink_probe_v1','arms':reports,
        'all_noise_and_prefix_checks_passed':True,'visual_review':'pending',
        'conclusion':'No causal outcome assumed before visual and latent comparison'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True);a=p.parse_args()
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:run(a.protocol,a.protocol_sha256)
    finally:shutdown_owned_runtime_helpers(label='lingbot-sink-probe')

if __name__=='__main__':main()
