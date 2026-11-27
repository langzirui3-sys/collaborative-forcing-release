"""Formal one-prompt, four-arm entry for the six-prompt cross-model study."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time

from .cross_model_delay import ARMS, PAIRS, load_plan, require, sha, write_new


def run(protocol, protocol_sha256, pair, prompt_index):
    from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices,_gpu_uuid,_gpu_compute_pids
    from .metadata import find_git_root,collect_git_metadata
    output=_run_dir();repository=find_git_root(Path.cwd());model_root=_model_root()
    require(not output.is_relative_to(repository),'output inside Git checkout')
    require(not any((p/n).exists() for p in (output,*output.parents) for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')),'sealed output')
    plan,item=load_plan(protocol,protocol_sha256,pair,prompt_index)
    devices=_visible_devices()
    require(len(devices)==2 and len(set(devices))==2,'two distinct reserved GPUs required')
    uuids=[_gpu_uuid(d) for d in devices]
    require(all(not _gpu_compute_pids(u) for u in uuids),'GPU already occupied')
    def ownership(target_pid):
        expected=[{os.getpid()},{target_pid} if target_pid is not None else {os.getpid()}]
        require(all(_gpu_compute_pids(u)<=pids for u,pids in zip(uuids,expected,strict=True)), 'foreign GPU process invalidates experiment')
    write_new(output/'preflight.json',{'schema':'cross_model_delay_preflight_v1','pair':pair,
        'prompt_index':prompt_index,'prompt':item,'protocol_sha256':protocol_sha256,'gpu_uuids':uuids,
        'git':collect_git_metadata(repository),'rank_routes_generation':False})
    started=time.perf_counter()
    if pair=='longlive_krea':
        from .cross_model_longlive import run as execute
    else:
        from .cross_model_lingbot import run as execute
    result=execute(plan,item,output,repository,model_root,ownership,devices)
    rows={x['arm']:x for x in result['arms']}
    require(set(rows)==set(ARMS),'missing or duplicate arms')
    for row in rows.values():
        row['warm_pipeline_plus_encode_seconds']=row.get('pipeline_with_monitoring_wall_seconds',row.get('pipeline_wall_seconds'))+row['encode_seconds']
    reference=rows['target_only']['warm_pipeline_plus_encode_seconds']
    for row in rows.values():row['speedup_vs_native_target']=reference/row['warm_pipeline_plus_encode_seconds']
    from PIL import Image,ImageDraw
    for page in range(6):
        panel=Image.new('RGB',(4160,4*264),'white');draw=ImageDraw.Draw(panel)
        for j,arm in enumerate(ARMS):
            for column in range(10):
                second=page*10+column
                with Image.open(output/'arms'/arm/f'second-{second:02d}.png') as image:panel.paste(image,(column*416,j*264+24))
                draw.text((column*416+4,j*264+4),f'{arm} | {second}s',fill='black')
        panel.save(output/f'comparison-seconds-{page*10:02d}-{page*10+9:02d}.png')
    complete={'schema':'cross_model_delay_result_v1','status':'generation_complete_quality_pending',
        'pair':pair,'prompt_index':prompt_index,'prompt':item,'seed':plan['seed'],
        'protocol_sha256':protocol_sha256,**result,'arms':[rows[a] for a in ARMS],
        'experiment_wall_seconds':time.perf_counter()-started,
        'success_criterion':'delay visual collapse versus native Draft AND reduce warm pipeline time versus native Target',
        'conclusion':'pending rank-hidden visual review; no positive outcome assumed'}
    write_new(output/'result.json',complete)
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--pair',choices=PAIRS,required=True);p.add_argument('--prompt-index',type=int,required=True)
    a=p.parse_args()
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:run(a.protocol,a.protocol_sha256,a.pair,a.prompt_index)
    finally:shutdown_owned_runtime_helpers(label='cross-model-delay')

if __name__=='__main__':main()
