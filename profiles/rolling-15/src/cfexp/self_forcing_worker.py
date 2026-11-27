"""Hash-bound native pure-Draft batch worker; formal entry is run_remote.sh."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import sys
import time

from .baseline_population import load_baseline_population
from .erank_frame_campaign import verify_artifact_spec
from .lingbot_smoke import digest, write_new
from .main_table_baseline_campaign import _run_dir, _model_root
from .metadata import find_git_root
from .self_forcing_frame_adapter import _load_runtime_config, _materialize_source
from .self_forcing_native import generate_native, load_native_weights, validate_config

SPEC = 'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'


def select_prompts(repository, ids):
    population, _ = load_baseline_population(repository)
    lookup = {p['id']: p for p in population}
    if not ids or len(set(ids)) != len(ids) or any(i not in lookup for i in ids):
        raise ValueError('unknown or duplicate frozen prompt IDs')
    return [lookup[i] for i in ids]


def run(*, duration, prompt_id):
    if type(duration) is not int or duration not in (15, 30, 60):
        raise ValueError('invalid native duration')
    run_dir = _run_dir(); repository = find_git_root(Path.cwd()); model_root = _model_root()
    if run_dir.is_relative_to(repository): raise ValueError('output inside repository')
    if any((p/n).exists() for p in (run_dir,*run_dir.parents)
           for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')):
        raise ValueError('sealed run')
    prompts = select_prompts(repository, prompt_id)
    spec_path = repository/SPEC
    _, verified, _ = verify_artifact_spec(spec_path, model_root=model_root, repo_root=repository)
    artifacts = {v['name']: Path(v['resolved_path']) for v in verified}
    config = _load_runtime_config({'artifact_spec_path': str(spec_path),
        'artifact_spec_sha256': digest(spec_path)}, artifacts)
    source, site = _materialize_source(run_dir=run_dir, worker_index=0,
        artifacts=artifacts, config=config, scratch_suffix='native-baseline')
    write_new(run_dir/'provenance/inputs.json', {'artifact_spec_sha256': digest(spec_path),
        'artifacts': verified, 'official_source_commit': config['official_source_commit']})
    old = Path.cwd(); sys.path[:0] = [str(site), str(source)]; os.chdir(source)
    try:
        import torch
        from omegaconf import OmegaConf
        from torchvision.io import write_video
        from .longlive_adapter import _verify_flash_attention
        _verify_flash_attention(site)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('native worker requires exactly one visible CUDA GPU')
        device = torch.device('cuda:0'); generation = config['generation']
        merged = OmegaConf.merge(OmegaConf.load(str(source/'configs/default_config.yaml')),
                                 OmegaConf.load(str(artifacts[generation['config_artifact']])))
        resolved = OmegaConf.to_container(merged, resolve=True); validate_config(resolved)
        pipeline_type = importlib.import_module('pipeline.causal_inference').CausalInferencePipeline
        start = time.perf_counter()
        pipeline = pipeline_type(merged, device=device)
        loaded = load_native_weights(pipeline=pipeline,
            checkpoint=artifacts[generation['checkpoint_artifact']],
            state_key=generation['checkpoint_state_key'], torch=torch, config=resolved)
        pipeline.to(dtype=torch.bfloat16)
        for module in (pipeline.generator, pipeline.text_encoder, pipeline.vae):
            module.to(device=device)
        pipeline.eval().requires_grad_(False)
        if tuple(float(x) for x in pipeline.denoising_step_list.tolist()) != (1000.0, 937.5, 833.3333129882812, 625.0):
            raise ValueError("native warped denoising schedule drifted")
        torch.cuda.synchronize(device); load_seconds = time.perf_counter()-start
        rows=[]
        for prompt in prompts:
            from .native_kv_audit import NativeKVAudit
            audit=NativeKVAudit();real_forward=pipeline.generator.forward
            def audited_forward(*args,**kwargs):
                audit.before(kwargs)
                return real_forward(*args,**kwargs)
            pipeline.generator.forward=audited_forward
            try:
                video, timing = generate_native(pipeline=pipeline, prompt=prompt['text'],
                    duration_seconds=duration, torch=torch, device=device)
                cache_audit=audit.finish(duration*2)
            finally:
                pipeline.generator.forward=real_forward
            path=run_dir/'videos'/('draft_only-'+prompt['id']+'.mp4')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists(): raise ValueError('video output already exists')
            from .video_encoding_policy import writer_options,verify_video
            encode_started=time.perf_counter()
            pixels=video[0].permute(0,2,3,1).mul(255).round().clamp(0,255).to(device='cpu',dtype=torch.uint8).contiguous()
            write_video(str(path),pixels,fps=24,**(writer_options() or {'video_codec':'h264'}))
            encode_seconds=time.perf_counter()-encode_started
            timing=dict(timing,video_encode_and_transfer_seconds=encode_seconds,
                video_ready_wall_seconds=timing['generation_wall_seconds']+timing['decode_seconds']+encode_seconds)
            encoding_policy=verify_video(path)
            del pixels,video
            pipeline.vae.model.clear_cache()
            import av
            with av.open(str(path)) as container:
                stream=container.streams.video[0]
                fps=float(stream.average_rate)
                geometry=set(); frames=0
                for frame in container.decode(video=0):
                    frames+=1; geometry.add((frame.height,frame.width))
            if fps!=24 or frames!=duration*24-3 or geometry!={(480,832)}:
                raise ValueError('decoded MP4 does not match requested horizon')
            row={'model_pair':'krea_self_forcing','method_id':'draft_only',
                'prompt_id':prompt['id'],'prompt_sha256':hashlib.sha256(prompt['text'].encode()).hexdigest(),
                'seed':20260911,'nominal_duration_seconds':duration,'timing':timing,
                'encoding_policy':encoding_policy,
                'native_cache_audit':cache_audit,
                'native_load':loaded,'video':{'path':path.relative_to(run_dir).as_posix(),
                'sha256':digest(path),'bytes':path.stat().st_size,'decoded_frame_count':frames,'fps':fps}}
            write_new(run_dir/'provenance'/(prompt['id']+'-generated.json'),row)
            rows.append(row)
        result={'schema':'self_forcing_native_batch_result_v1','status':'succeeded',
                'quality_evaluated':False,'model_load_seconds':load_seconds,'rows':rows}
        write_new(run_dir/'result.json',result)
        return result
    finally:
        os.chdir(old)
        for p in (str(source),str(site)):
            if p in sys.path: sys.path.remove(p)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=int,required=True,choices=(15,30,60))
    parser.add_argument('--prompt-id',action='append',required=True)
    run(**vars(parser.parse_args()))
