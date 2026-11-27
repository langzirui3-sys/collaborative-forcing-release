"""Frozen LingBot native baselines: six-prompt generation and official quality.

Launch with run_remote.sh only. Two GPUs are reserved: one generation worker,
then the other for the unchanged offline VBench / VBench-Long evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .baseline_population import load_baseline_population,select_prompt_shard
from .lingbot_geometry import require_horizon
from .lingbot_smoke import digest,write_new
from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices,_gpu_uuid,_gpu_compute_pids
from .metadata import find_git_root
from .vbench_long import OfficialVBenchLongScorer,VBenchLongVideoRequest,VBENCH_LONG_DIMENSIONS
from .vbench_short import OfficialVBenchScorer


def validate_generation(result, directory, prompts, model_size, duration):
    if result.get('status')!='succeeded':raise ValueError('generation did not succeed')
    rows=result.get('rows') if result.get('schema')=='lingbot_native_batch_result_v1' else [result]
    if not isinstance(rows,list) or [r.get('prompt_id') for r in rows]!=[p['id'] for p in prompts]:
        raise ValueError('generation coverage/order differs from frozen shard')
    contract=require_horizon(duration,duration*16+1)
    for row,prompt in zip(rows,prompts):
        if (row.get('model_size')!=model_size or row.get('seed')!=20260911
                or row.get('model_pair')!='lingbot_world_2'
                or row.get('method_id')!=('draft_only' if model_size=='1.3b' else 'target_only')
                or row.get('prompt_sha256')!=hashlib.sha256(prompt['text'].encode()).hexdigest()):
            raise ValueError('native prompt/model identity mismatch')
        latency=row.get('timing',{}).get('generation_wall_seconds')
        if type(latency) not in (int,float) or not math.isfinite(latency) or latency<=0:
            raise ValueError('invalid native generation timing')
        pipeline_latency=row['timing'].get('pipeline_wall_seconds')
        if type(pipeline_latency) not in (int,float) or not math.isfinite(pipeline_latency) or pipeline_latency<latency:
            raise ValueError('invalid native pipeline timing')
        geometry=row['timing'].get('temporal_contract',{})
        if geometry!=require_horizon(duration,contract['requested_rgb_frames']):
            raise ValueError('native input horizon differs')
        video=row.get('video',{});relative=Path(video.get('path',''))
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe video path')
        path=directory/relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('video absent or escaped generation directory')
        if (digest(path)!=video.get('sha256') or path.stat().st_size!=video.get('bytes')
                or video.get('decoded_frame_count')!=contract['requested_rgb_frames'] or video.get('fps')!=16):
            raise ValueError('native video identity/geometry mismatch')
        if row.get('input_receipt_sha256')!=digest(directory/'prepared-inputs/input-receipt.json'):
            raise ValueError('prepared input receipt mismatch')
        load=row.get('native_load',{})
        if load.get('strict') is not True or load.get('missing_keys')!=[] or load.get('unexpected_keys')!=[]:
            raise ValueError('native weights were not loaded completely')
    return rows


def run(*,model_size,duration,shard_index,shard_count,vbench_checkout,vbench_python,vbench_asset_manifest,vbench_asset_spec,smoke_prompt_id=None):
    if model_size not in ('1.3b','14b') or duration not in (15,30,60):raise ValueError('unsupported condition')
    run_dir=_run_dir();_model_root();repo=find_git_root(Path.cwd())
    if run_dir.is_relative_to(repo):raise ValueError('output inside repository')
    if any((p/n).exists() for p in (run_dir,*run_dir.parents) for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')):
        raise ValueError('sealed run')
    generation_device,quality_device=_visible_devices()
    uuids=[_gpu_uuid(d) for d in (generation_device,quality_device)]
    if len(set(uuids))!=2 or any(_gpu_compute_pids(u) for u in uuids):raise ValueError('reserved GPUs not idle/distinct')
    prompts,population=load_baseline_population(repo)
    population=dict(population,model_pair='lingbot_world_2')
    if smoke_prompt_id:
        if shard_count!=1 or shard_index!=0:raise ValueError('cannot shard smoke')
        prompts=[p for p in prompts if p['id']==smoke_prompt_id]
        if len(prompts)!=1:raise ValueError('unknown smoke prompt')
    else:
        prompts=select_prompt_shard(prompts,shard_index,shard_count)
    generation_dir=run_dir/'generation';generation_dir.mkdir(exist_ok=False)
    env=dict(os.environ,CFEXP_RUN_DIR=str(generation_dir),CUDA_VISIBLE_DEVICES=generation_device)
    command=[sys.executable,'-m','cfexp.lingbot_smoke','--model-size',model_size,'--duration',str(duration)]
    for p in prompts:command+=['--prompt-id',p['id']]
    log=run_dir/'logs/generation.log';log.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with log.open('x') as handle:
        subprocess.run(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,check=True)
    generation_elapsed=time.perf_counter()-started
    native=json.loads((generation_dir/'result.json').read_text())
    generated=validate_generation(native,generation_dir,prompts,model_size,duration)
    if _gpu_compute_pids(uuids[1]):raise ValueError('quality GPU became busy')
    benchmark='VBench' if duration in (15,30) else 'VBench-Long'
    scorer=(OfficialVBenchScorer if benchmark=='VBench' else OfficialVBenchLongScorer)(
        run_dir=run_dir,checkout=vbench_checkout,python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest,asset_spec=vbench_asset_spec,visible_device=quality_device)
    method='draft_only' if model_size=='1.3b' else 'target_only'
    requests=[VBenchLongVideoRequest(video_path=generation_dir/r['video']['path'],prompt=p['text'],measurement_id=f'lingbot-{method}-{p["id"]}') for r,p in zip(generated,prompts)]
    started=time.perf_counter();measurements=scorer.score_videos(requests);score_elapsed=time.perf_counter()-started
    if set(measurements)!={r.measurement_id for r in requests}:raise ValueError('quality coverage incomplete')
    rows=[];evidence=[]
    for generated_row,prompt,request in zip(generated,prompts,requests):
        measure=measurements[request.measurement_id];scores=dict(measure.dimension_scores)
        if tuple(scores)!=VBENCH_LONG_DIMENSIONS or any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
            raise ValueError('invalid official dimension scores')
        path=run_dir/measure.evidence_path
        if not path.resolve().is_relative_to(run_dir) or path.is_symlink():raise ValueError('quality evidence escaped run')
        evidence.append({'prompt_id':prompt['id'],'path':path.relative_to(run_dir).as_posix(),'sha256':digest(path),'bytes':path.stat().st_size})
        rows.append({'model_pair':'lingbot_world_2','method_id':method,'prompt_id':prompt['id'],'seed':20260911,
            'nominal_duration_seconds':duration,'benchmark':benchmark,'fps':16,
            'decoded_frame_count':generated_row['video']['decoded_frame_count'],
            'generation_wall_seconds':generated_row['timing']['generation_wall_seconds'],
            'pipeline_wall_seconds':generated_row['timing']['pipeline_wall_seconds'],
            'quality_scores':scores,'official_quality_verified':True,
            'input_receipt_sha256':generated_row['input_receipt_sha256'],
            'video':dict(generated_row['video'],path='generation/'+generated_row['video']['path'])})
    raw=run_dir/'raw/lingbot-baseline.json';write_new(raw,{'schema':'lingbot_baseline_rows_v1','rows':rows})
    result={'schema':'lingbot_baseline_campaign_result_v1','status':'succeeded','model_pair':'lingbot_world_2',
        'method_id':method,'nominal_duration_seconds':duration,'benchmark':benchmark,
        'run_kind':'single_prompt_smoke' if smoke_prompt_id else 'formal_prompt_shard',
        'population':population,'prompt_ids':[p['id'] for p in prompts],'prompt_count':len(prompts),
        'shard_index':shard_index,'shard_count':shard_count,'seed':20260911,
        'generation_process_wall_seconds':generation_elapsed,'quality_wall_seconds':score_elapsed,
        'raw_rows':{'path':raw.relative_to(run_dir).as_posix(),'sha256':digest(raw),'bytes':raw.stat().st_size},
        'quality_evidence':evidence,'comparison_scope':'image/action-conditioned within LingBot; six shared quality dimensions, not official total'}
    write_new(run_dir/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-size',required=True,choices=('1.3b','14b'))
    parser.add_argument('--duration',required=True,type=int,choices=(15,30,60))
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--smoke-prompt-id')
    for option in ('vbench-checkout','vbench-python','vbench-asset-manifest','vbench-asset-spec'):
        parser.add_argument('--'+option,type=Path,required=True)
    args=parser.parse_args();run(**vars(args))
