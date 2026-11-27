"""Three-arm complex-scene diagnostic; no quality or collapse outcome assumed."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from .cross_model_delay import SEED, geometry, require, sha, write_new

ARMS=('draft_only','target_only','target_root_only')
PAIRS=('longlive_krea','lingbot_world_2')


def validate_plan(path,expected):
    require(sha(path)==expected,'complex protocol digest drift')
    p=json.loads(path.read_text())
    require(p['schema']=='complex_scene_diagnostic_v1' and p['status']=='frozen','unfrozen complex plan')
    require(p['model_pairs']==list(PAIRS) and p['arms']==list(ARMS),'complex matrix drift')
    require(p['seed']==SEED and p['nominal_seconds']==60,'complex geometry drift')
    rows=p['prompts']
    require(len(rows)==6 and len({r['prompt_id'] for r in rows})==6,'six unique scenes required')
    require([r['category'] for r in rows]==['multi_person','dance','running','vehicle_occlusion','object_interaction','continuous_camera'],'scene category drift')
    require(all(isinstance(r['prompt'],str) and r['prompt'].strip() for r in rows),'empty prompt')
    return p


def bind_source(source,index,item):
    from .main_table_merge import _load_frozen_index
    frozen,seal=_load_frozen_index(source)
    require(seal['artifact_index_sha256']==index,'source index drift')
    base=source/'attempts'/seal['attempt_id']
    def bound(relative):return frozen.file((base/relative).relative_to(source).as_posix(),label='complex scene input')
    result=json.loads(bound('result.json').read_text())
    require(result['schema']=='complex_scene_diagnostic_result_v1' and result['pair']=='longlive_krea' and result['prompt']==item,'source scene mismatch')
    return bound('arms/target_only/first-frame.png'),{'source_run':str(source),'source_index_sha256':index,'source_attempt':seal['attempt_id']}


def bind_conditioning(path, image, source_index, item):
    """Freeze a visually reviewed image-compatible prompt before generation."""
    record=json.loads(path.read_text())
    require(record['schema']=='complex_scene_conditioning_v1','conditioning schema mismatch')
    require(record['prompt_id']==item['prompt_id'] and record['source_prompt']==item['prompt'],'conditioning source prompt drift')
    require(record['source_index_sha256']==source_index and record['image_sha256']==sha(image),'conditioning image binding drift')
    require(record['review_status']=='image_reviewed' and record['source_image_observation'].strip(),'unreviewed source image')
    require(isinstance(record['conditioning_prompt'],str) and record['conditioning_prompt'].strip(),'empty conditioning prompt')
    require(record['comparison_scope']=='within_lingbot_pair_only','unsupported cross-pair comparison')
    return {**item,'prompt':record['conditioning_prompt']},record


def prepare_action(prepared,output,image,*,moving_camera=False,tracking_camera=False):
    """Hold the official example00 first camera pose: no invented key actions."""
    from .lingbot_inputs import read_npy,resize_npy
    require(not (moving_camera and tracking_camera),'ambiguous camera motion')
    action=output/'scene-input';action.mkdir()
    shutil.copyfile(image,action/'image.png')
    from PIL import Image
    with Image.open(image) as im:im.convert('RGB').save(action/'image.jpg',quality=100,subsampling=0)
    for name in ('poses.npy','intrinsics.npy'):
        original=(prepared/'examples/00'/name).read_bytes()
        # First truncate to a single calibrated pose, then hold it for957 frames.
        first=resize_npy(original,1,hold_last=True)
        (action/name).write_bytes(resize_npy(first,957,hold_last=True))
        require(read_npy((action/name).read_bytes())[0]['shape'][0]==957,'camera horizon drift')
    if moving_camera or tracking_camera:
        import numpy as np
        poses=np.load(action/'poses.npy',allow_pickle=False)
        require(poses.shape==(957,4,4),'camera pose layout drift')
        initial=poses[0].copy()
        for i,angle in enumerate(np.linspace(0,.35,957)):
            c,sn=float(np.cos(angle)),float(np.sin(angle))
            rotation=np.array([[c,0,sn,0],[0,1,0,0],[-sn,0,c,0],[0,0,0,1]],dtype=poses.dtype)
            if tracking_camera:
                rotation=np.eye(4,dtype=poses.dtype)
                rotation[0,3]=3.0*i/(len(poses)-1)
            poses[i]=initial@rotation
        np.save(action/'poses.npy',poses,allow_pickle=False)
    write_new(action/'binding.json',{'source_image_sha256':sha(image),'jpeg_sha256':sha(action/'image.jpg'),
        'camera_policy':'initial_pose_local_x_0_to_3_units_957_frames' if tracking_camera else 'initial_pose_local_yaw_0_to_0.35_radians_957_frames' if moving_camera else 'official_example00_first_pose_and_intrinsics_held_957_frames',
        'wasd_action':None,'claim_limit':'image-conditioned within-pair diagnostic; synthetic camera path is not ground-truth scene geometry'})
    return action


def run(protocol,expected,pair,prompt_index,source=None,source_index=None,scene_binding=None):
    from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices,_gpu_uuid,_gpu_compute_pids
    from .metadata import find_git_root
    from .stage_timing import _unsealed_directory
    plan=validate_plan(protocol,expected)
    require(pair in PAIRS and type(prompt_index) is int and 0<=prompt_index<6,'invalid diagnostic coordinate')
    require((pair=='lingbot_world_2')==(source is not None and source_index is not None),'LingBot requires a sealed matching LongLive source')
    require((pair=='lingbot_world_2')==(scene_binding is not None),'LingBot requires reviewed image conditioning')
    item=plan['prompts'][prompt_index]
    output=_run_dir();_unsealed_directory(output);repository=find_git_root(Path.cwd())
    require(not output.is_relative_to(repository),'output inside repository')
    devices=_visible_devices();require(len(devices)==2 and len(set(devices))==2,'two distinct GPUs required')
    uuids=[_gpu_uuid(d) for d in devices]
    require(all(not _gpu_compute_pids(u) for u in uuids),'GPU occupied')
    def ownership(pid):
        allowed=[{os.getpid()},{pid} if pid else {os.getpid()}]
        require(all(_gpu_compute_pids(u)<=a for u,a in zip(uuids,allowed,strict=True)),'foreign GPU process')
    write_new(output/'preflight.json',{'protocol_sha256':expected,'pair':pair,'prompt':item,'gpu_uuids':uuids,'rank_routes_generation':False})
    if pair=='longlive_krea':
        from .cross_model_longlive import run as execute
        schedules={'draft_only':(),'target_root_only':(0,)}
        result=execute(plan,item,output,repository,_model_root(),ownership,devices,arms=ARMS,target_schedules=schedules)
    else:
        from . import cross_model_lingbot as native
        image,binding=bind_source(source,source_index,item)
        conditioned_item,conditioning=bind_conditioning(scene_binding,image,source_index,item)
        write_new(output/'conditioning-review.json',conditioning)
        t,pipes,prepared,receipt=native.build_pair(output,repository,_model_root())
        action=prepare_action(prepared,output,image,moving_camera=item['category']=='continuous_camera',
            tracking_camera=item['category']=='running' and plan.get('camera_policy_version')==2)
        write_new(output/'input-binding.json',{**binding,'materialization':receipt,'image_sha256':sha(image)})
        from PIL import Image
        for pipe in pipes.values():
            with Image.open(action/'image.jpg') as im,t.cuda.device(pipe.device):pipe.prewarm(im.convert('RGB'),max_area=480*832,frame_num=957,chunk_size=4)
        results=[]
        for arm in ARMS:
            targets=() if arm=='draft_only' else tuple(range(60)) if arm=='target_only' else (0,)
            results.append(native.generate_arm(pipes=pipes,torch=t,item=conditioned_item,action=action,output=output,arm=arm,
                reference=output/'arms/draft_only' if arm=='target_root_only' else None,ownership=lambda:ownership(None),target_chunks=targets))
        result={'arms':results,'conditioning_prompt':conditioned_item['prompt'],'conditioning_binding_sha256':sha(scene_binding),'comparison_scope':'within_lingbot_pair_only'}
    traces={a:json.loads((output/'arms'/a/'trace.json').read_text()) for a in ARMS}
    root=traces['target_root_only']['chunks'];target=traces['target_only']['chunks']
    require(root[0]['latent_sha256']==target[0]['latent_sha256'],'Target root differs from pure Target')
    g=geometry(pair)
    ranks=[]
    for a in ARMS:
        tr=traces[a]
        require(len(tr['chunks'])==g['chunk_count'],'chunk horizon mismatch')
        if pair=='lingbot_world_2' or a!='target_only':
            require(tr['renoise']==traces['draft_only']['renoise'],'actual stochastic input differs')
        for r in tr['chunks']:
            e=r['erank_observation'];ranks.append([a,r['chunk'],*r['video_interval_seconds'],e['producer'],e['rank']])
    with (output/'erank.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','chunk','start_seconds','end_seconds','producer','hidden_erank']);w.writerows(ranks)
    for row in result['arms']:row['warm_pipeline_plus_encode_seconds']=row.get('pipeline_with_monitoring_wall_seconds',row.get('pipeline_wall_seconds'))+row['encode_seconds']
    write_new(output/'result.json',{'schema':'complex_scene_diagnostic_result_v1','pair':pair,'prompt':item,
        'seed':SEED,'protocol_sha256':expected,**result,'visual_review':'pending; no collapse or benefit assumed',
        'review_categories':['severe_artifact','motion_stall','identity_drift','interaction_error','prompt_deviation']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--pair',choices=PAIRS,required=True);p.add_argument('--prompt-index',type=int,required=True)
    p.add_argument('--source-run',type=Path);p.add_argument('--source-index');p.add_argument('--scene-binding',type=Path)
    a=p.parse_args()
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:run(a.protocol,a.protocol_sha256,a.pair,a.prompt_index,a.source_run,a.source_index,a.scene_binding)
    finally:shutdown_owned_runtime_helpers(label='complex-scene-diagnostic')

if __name__=='__main__':main()
