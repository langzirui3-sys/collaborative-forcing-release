"""Native LingBot generation worker with source/weight binding and decoded proof.

Enter through scripts/run_remote.sh at a full published commit. This verifies
runtime feasibility or produces a prompt batch for lingbot_campaign. Generation
alone is not a quality score or a completed table row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .lingbot_geometry import SOURCE_REVISION, validate_decoded_video
from .lingbot_inputs import prepare_inputs
from .lingbot_native import generate_native, native_config

LARGE_REPO = 'robbyant/lingbot-world-v2-14b-causal-fast'
SMALL_REPO = 'robbyant/lingbot-world-v2-1.3b-causal-fast'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def verified_file(root, record):
    relative=Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe frozen asset path')
    path=root/relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('asset outside model root or absent')
    if digest(path)!=record['sha256'] or ('size' in record and path.stat().st_size!=record['size']):
        raise ValueError('frozen asset digest/size mismatch: '+str(relative))
    return path


def materialize(repository, model_root, run_dir, model_size, duration):
    from .longlive_adapter import _safe_extract_tar
    from .self_forcing_frame_adapter import _safe_extract_wheel
    native_config(model_size)
    records=json.loads((repository/'experiments/four-pair-model-downloads-v1.json').read_text())['files']
    # The small model shares the verified large-model T5/VAE/tokenizer files.
    selected=[r for r in records if (r['repository']==(LARGE_REPO if model_size=='14b' else SMALL_REPO))
              or (r['repository']==LARGE_REPO and (r['filename'].startswith('google/') or r['filename'] in ('Wan2.1_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth')))]
    if not selected:raise ValueError('no frozen model assets')
    source_record=json.loads((repository/'experiments/lingbot-source-download-v1.json').read_text())['files'][0]
    source_archive=verified_file(model_root,source_record)
    wheel_record=next(r for r in json.loads((repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json').read_text())['artifacts'] if r['name']=='flash_attn_wheel')
    wheel=verified_file(model_root,wheel_record)
    bound=[(r,verified_file(model_root,r)) for r in selected]
    scratch=run_dir/'tmp/lingbot-native'
    _safe_extract_tar(source_archive,scratch/'source')
    source=scratch/'source'/('lingbot-world-v2-'+SOURCE_REVISION)
    _safe_extract_wheel(wheel,scratch/'site')
    paths={'checkpoint':scratch/'checkpoint','shared':scratch/'shared'}
    for record,actual in bound:
        shared=record['repository']==LARGE_REPO and model_size=='1.3b'
        destination=paths['shared' if shared else 'checkpoint']/record['filename']
        destination.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(actual,destination)
        except OSError:shutil.copy2(actual,destination)
        if digest(destination)!=record['sha256']:raise ValueError('materialized asset changed')
    policy=json.loads((repository/'experiments/lingbot-input-extension-v1.policy.json').read_text())
    protocol=repository/policy['source_protocol']
    if digest(protocol)!=policy['source_protocol_sha256']:raise ValueError('input policy binding drift')
    prepared=run_dir/'prepared-inputs'
    receipt=prepare_inputs(source,protocol,prepared,duration)
    provenance={'schema':'lingbot_native_materialization_v1','source_revision':SOURCE_REVISION,
                'model_size':model_size,'files':selected,'source_archive':source_record,
                'runtime_wheel':wheel_record,'input_policy_sha256':digest(repository/'experiments/lingbot-input-extension-v1.policy.json'),
                'input_receipt_sha256':digest(prepared/'input-receipt.json')}
    write_new(run_dir/'provenance/materialization.json',provenance)
    return source,scratch/'site',paths,prepared,receipt


def strict_dit_loader(module, loads):
    """Use official architecture and state reader, reject partial weight loading."""
    def load(model_cls, checkpoint_dir, subfolder, config, torch_dtype, extra=None):
        directory=module._resolve_dit_dir(checkpoint_dir,subfolder)
        # Both frozen uploads provide a shard index. Do not use from_pretrained's
        # permissive missing-key initialization or fetch any remote artifacts.
        state=module._load_safetensors_state_dict(directory)
        model=model_cls(**module._dit_kwargs_from_config(config,extra))
        result=model.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError('incomplete native DiT weight loading')
        loads.append({'tensor_count':len(state),'parameter_count':sum(p.numel() for p in model.parameters()),
                      'missing_keys':[],'unexpected_keys':[],'strict':True})
        del state
        return model.to(dtype=torch_dtype)
    return load


def run(model_size, duration, prompt_id):
    from .baseline_population import load_baseline_population
    from .main_table_baseline_campaign import _run_dir,_model_root,_gpu_uuid,_gpu_compute_pids
    from .metadata import find_git_root
    run_dir=_run_dir();repository=find_git_root(Path.cwd());model_root=_model_root()
    if run_dir.is_relative_to(repository):raise ValueError('run output inside Git checkout')
    if any((p/n).exists() for p in (run_dir,*run_dir.parents) for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')):
        raise ValueError('run already sealed')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]:raise ValueError('native smoke requires exactly one reserved GPU')
    uuid=_gpu_uuid(devices[0])
    if _gpu_compute_pids(uuid):raise ValueError('reserved GPU has existing compute processes')
    prompts,population=load_baseline_population(repository)
    population=dict(population,model_pair='lingbot_world_2')
    prompt_ids=[prompt_id] if isinstance(prompt_id,str) else list(prompt_id)
    if not prompt_ids or len(set(prompt_ids))!=len(prompt_ids):raise ValueError('empty or duplicate prompts')
    selected=[next((p for p in prompts if p['id']==pid),None) for pid in prompt_ids]
    if any(p is None for p in selected):raise ValueError('prompt outside frozen population')
    source,site,paths,prepared,receipt=materialize(repository,model_root,run_dir,model_size,duration)
    for name in ('wan','flash_attn'):
        if name in sys.modules:raise ValueError('native dependency already imported from unknown source')
    sys.path[:0]=[str(site),str(source)]
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import flash_attn
    import wan.image2video as module
    from wan.configs import WAN_CONFIGS
    from PIL import Image
    if flash_attn.__version__!='2.8.3':raise ValueError('FlashAttention runtime version drift')
    if not Path(module.__file__).resolve().is_relative_to(source):raise ValueError('wrong native source import')
    loads=[];original=module.load_dit_model
    module.load_dit_model=strict_dit_loader(module,loads)
    started=time.perf_counter()
    try:
        pipeline=module.WanI2VCausal(config=WAN_CONFIGS['i2v-A14B' if model_size=='14b' else 'i2v-1.3B'],
            checkpoint_dir=str(paths['checkpoint']),assets_dir=str(paths['shared'] if model_size=='1.3b' else paths['checkpoint']),
            device_id=0,rank=0,t5_fsdp=False,dit_fsdp=False,use_sp=False,t5_cpu=False,
            convert_model_dtype=True,local_attn_size=18,sink_size=6,infer_mode='causal_fast')
    finally:module.load_dit_model=original
    if len(loads)!=1:raise ValueError('native DiT loader was not used exactly once')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    rows=[]
    for prompt in selected:
        prompt_id=prompt['id']
        mapping=next(m for m in receipt['mappings'] if m['prompt_id']==prompt_id)
        action=prepared/mapping['example_directory']
        if _gpu_compute_pids(uuid)-{os.getpid()}:raise ValueError('foreign GPU process before generation')
        torch.cuda.reset_peak_memory_stats()
        with Image.open(action/'image.jpg') as image:
            video,timing=generate_native(pipeline=pipeline,prompt=prompt['text'],image=image.convert('RGB'),
                action_path=action,duration_seconds=duration,model_size=model_size,torch=torch)
        peak={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
        from wan.utils.utils import save_video
        path=run_dir/'videos'/f'{model_size}-{duration}s-{prompt_id}.mp4';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('video destination exists')
        save_video(tensor=video[None],save_file=str(path),fps=16,nrow=1,normalize=True,value_range=(-1,1))
        import av
        with av.open(str(path)) as container:
            stream=container.streams.video[0];fps=stream.average_rate
            if fps is None or fps.denominator!=1:raise ValueError('unexpected encoded FPS')
            decoded=sum(1 for _ in container.decode(stream))
        validate_decoded_video(timing['temporal_contract'],frame_count=decoded,fps=int(fps))
        if _gpu_compute_pids(uuid)-{os.getpid()}:raise ValueError('foreign GPU process invalidated timing')
        result={'schema':'lingbot_native_smoke_result_v1','status':'succeeded','run_kind':'native_prompt_generation' if len(selected)>1 else 'single_prompt_smoke',
                'model_pair':'lingbot_world_2','method_id':'draft_only' if model_size=='1.3b' else 'target_only',
                'prompt_id':prompt_id,'prompt_sha256':hashlib.sha256(prompt['text'].encode()).hexdigest(),
                'population':population,'seed':20260911,'model_size':model_size,'native_load':loads[0],
                'load_seconds':load_seconds,'timing':timing,'peak_memory':peak,'gpu_uuid':uuid,
                'quality_evaluated':False,'input_receipt_sha256':digest(prepared/'input-receipt.json'),
                'video':{'path':path.relative_to(run_dir).as_posix(),'sha256':digest(path),'bytes':path.stat().st_size,
                         'decoded_frame_count':decoded,'fps':int(fps)}}
        rows.append(result)
        write_new(run_dir/'provenance'/f'{prompt_id}.json',result)
        del video
    if len(rows)>1:
        result={'schema':'lingbot_native_batch_result_v1','status':'succeeded','rows':rows,
                'prompt_count':len(rows),'model_size':model_size,'load_seconds':load_seconds,
                'quality_evaluated':False,'seed':20260911,'population':population}
    write_new(run_dir/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-size',choices=('1.3b','14b'),required=True)
    parser.add_argument('--duration',type=int,choices=(15,30,60),required=True)
    parser.add_argument('--prompt-id',required=True,action='append')
    args=parser.parse_args();run(args.model_size,args.duration,args.prompt_id)
