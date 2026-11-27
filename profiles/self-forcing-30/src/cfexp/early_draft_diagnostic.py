"""GPU correctness gate for layer-14 early exit; NOT a video quality benchmark.

The replacement latent is explicitly synthetic. This isolates cache mechanics
before spending a Target generation or claiming a collaborative speedup.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

from .runtime_helpers import shutdown_owned_runtime_helpers

PROMPT = "A red toy car drives along a winding road, passing trees as the camera follows it."
REPLACE_AT = (1, 6, 7, 10)


def compare(a, b, torch):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("comparison geometry differs")
    d = a.float()-b.float()
    return {"exact": bool(torch.equal(a,b)), "max_abs": float(d.abs().max().item()),
            "relative_l2": float((d.norm()/b.float().norm().clamp_min(1e-30)).item())}


def clean_reference(backend, history, widths):
    """Independent native replay: never executes a rejected Draft proposal."""
    from .self_forcing_frame_adapter import StepZeroCapture, LAYER_INDICES
    t,p = backend.torch,backend.pipeline
    p._initialize_kv_cache(batch_size=1,dtype=t.bfloat16,device=backend.device)
    p._initialize_crossattn_cache(batch_size=1,dtype=t.bfloat16,device=backend.device)
    backend._active_capture = StepZeroCapture(layers=LAYER_INDICES,target_timestep=1000,
        sample_hidden=backend._sample_hidden,rank_sample=backend._rank_sample)
    try:
        with t.inference_mode():
            cond=p.text_encoder(text_prompts=[PROMPT])
            for i,width in enumerate(widths):
                p.generator(noisy_image_or_video=history[:,i*3:i*3+3].to(backend.device),
                    conditional_dict=cond,timestep=t.zeros((1,width),device=backend.device,dtype=t.int64),
                    kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                    current_start=i*3*p.frame_seq_length)
        return {"self":p.kv_cache1,"cross":p.crossattn_cache}
    finally:
        backend._active_capture=None
        p.kv_cache1=p.crossattn_cache=None


def run(output, repository, model_root):
    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend,_materialize_source
    from .main_table_baseline_campaign import _gpu_uuid,_gpu_compute_pids
    spec,verified,module=verify_artifact_spec(
        repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json',
        model_root=model_root,repo_root=repository)
    if module!='cfexp.self_forcing_frame_adapter' or os.environ.get('CFEXP_DRAFT_KV_REUSE')!='1':
        raise ValueError('native Self-Forcing reuse is required')
    artifacts={r['name']:Path(r['resolved_path']) for r in verified}
    source,site=_materialize_source(run_dir=output,worker_index=0,artifacts=artifacts,
        config=spec['adapter_config'],scratch_suffix='early-draft-parity')
    device=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not device or ',' in device:raise ValueError('exactly one GPU required')
    uuid=_gpu_uuid(device)
    if _gpu_compute_pids(uuid):raise ValueError('GPU is occupied')
    b=OfficialSelfForcingBackend(source_root=source,python_site=site,artifacts=artifacts,
        config=spec['adapter_config'],device_name='cuda:0',chunk_count=16)
    report={"schema":"early_draft_native_gate_v1","formal_quality_result":False,
        "prompt":PROMPT,"seed":20260911,"gpu_uuid":uuid,
        "replacement":"synthetic fixed latent; no Target quality or speed claim",
        "layer_index":14,"replace_at":list(REPLACE_AT),"rows":[],"passed":False}
    handles=[];block_calls=[]
    try:
        t=b.torch
        for index,block in enumerate(b.pipeline.generator.model.blocks):
            def count(_m,_a,index=index):block_calls.append(index)
            handles.append(block.register_forward_pre_hook(count))
        history=None;widths=()
        for i in range(16):
            if not _gpu_compute_pids(uuid)<={os.getpid()}:raise ValueError('unrelated GPU workload appeared')
            noise=t.randn((1,3,16,60,104),generator=t.Generator(device='cpu').manual_seed(20260911+i),dtype=t.bfloat16)
            kwargs=dict(prompt=PROMPT,history_latents=history,initial_noise=noise,
                internal_seed=20260911+i,chunk_index=i,capture_hidden=False,history_kv_widths=widths)
            # Rebuild is an explicitly labelled oracle, never the optimized arm.
            b._incremental_draft.cache.clear()
            full=b.generate_chunk(**kwargs)
            block_calls.clear();t.cuda.synchronize();start=time.perf_counter()
            early=b.generate_chunk(**kwargs,early_should_replace=lambda measurement: i in REPLACE_AT)
            t.cuda.synchronize();elapsed=time.perf_counter()-start
            row={"chunk_index":i,"early_request_seconds":elapsed,
                 "early_block_calls":len(block_calls),"kv_reuse":early['early_execution'], "native_reuse":early['kv_reuse']}
            if i in REPLACE_AT:
                if not early.get('stopped') or 'latent' in early or len(block_calls)!=15:
                    raise ValueError('Draft did not stop at layer 14')
                row['rank_equal']=early['probe']['effective_rank']==full['rank']
                external=t.full_like(noise,0.125)
                block_calls.clear();t.cuda.synchronize();start=time.perf_counter()
                row['commit']=b.commit_early_target(prompt=PROMPT,history_latents=history,
                    history_kv_widths=widths,latent=external)
                t.cuda.synchronize();row['commit_seconds']=time.perf_counter()-start
                row['commit_block_calls']=len(block_calls)
                if len(block_calls)!=len(b.pipeline.generator.model.blocks):
                    raise ValueError('external commit was not one complete clean forward')
                accepted=external;width=1
            else:
                row['latent']=compare(early['latent'],full['latent'],t)
                row['rank_equal']=early['rank']==full['rank']
                accepted=early['latent'];width=3
            history=accepted if history is None else t.cat((history,accepted),dim=1)
            widths=widths+(width,)
            if i in REPLACE_AT:
                reference=clean_reference(b,history,widths)
                differences=[]
                for layer,(actual,expected) in enumerate(zip(b._early_draft.payload['self'],reference['self'])):
                    for key in ('k','v','global_end_index','local_end_index'):
                        c=compare(actual[key],expected[key],t)
                        if not c['exact']:differences.append({'layer':layer,'field':key,**c})
                row['cache_differences']=differences
                del reference
            report['rows'].append(row)
            print('EARLY_GATE',i,json.dumps(row,sort_keys=True),flush=True)
            if not row['rank_equal'] or not row.get('latent',{'exact':True})['exact'] or row.get('cache_differences'):
                raise ValueError('native parity failed; do not deploy as an optimization')
        report['passed']=True
        return report
    finally:
        with (output/'early-gate.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
        for handle in handles:handle.remove()
        if hasattr(b,'_early_draft'):b._early_draft.payload=None
        b.close()


def main():
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root
    try:run(_run_dir(),find_git_root(Path.cwd()),_model_root())
    finally:shutdown_owned_runtime_helpers(label='early-draft-gate')

if __name__=='__main__':main()
