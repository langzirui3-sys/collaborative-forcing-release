"""Synthetic GPU audit parity/timing gate; not a model quality benchmark."""
from __future__ import annotations
import json
import os
from pathlib import Path
import statistics
import time


def main():
    import torch
    from .self_forcing_frame_adapter import sampled_kv_cache_witness
    root = Path(os.environ['CFEXP_RUN_DIR']).resolve(strict=True)
    if (root/'result.json').exists():
        raise ValueError('fresh diagnostic output required')
    if not torch.cuda.is_available():
        raise ValueError('GPU timing gate requires CUDA')
    device = 'cuda:0'
    torch.manual_seed(20260911)
    layers, cross = [], []
    for i in range(24):
        k = torch.randn((1,4096,8,128),device=device,dtype=torch.bfloat16)
        v = torch.randn_like(k)
        layers.append({'k':k,'v':v,'global_end_index':torch.tensor(12000,device=device),
            'local_end_index':torch.tensor(4096,device=device)})
        cross.append({'k':k[:,:512].clone(),'v':v[:,:512].clone(),'is_init':True})
    originals = [x[k].clone() for x in layers+cross for k in ('k','v')]
    reference = sampled_kv_cache_witness(layers,cross,batched=False)
    assert reference == sampled_kv_cache_witness(layers,cross,batched=True)
    for _ in range(3):
        sampled_kv_cache_witness(layers,cross,batched=False)
        sampled_kv_cache_witness(layers,cross,batched=True)
    timings = {False:[],True:[]}
    for repeat in range(20):
        for selected in ([False,True] if repeat%2==0 else [True,False]):
            torch.cuda.synchronize();started=time.perf_counter()
            value=sampled_kv_cache_witness(layers,cross,batched=selected)
            torch.cuda.synchronize();timings[selected].append(time.perf_counter()-started)
            assert value == reference
    current=[x[k] for x in layers+cross for k in ('k','v')]
    assert all(torch.equal(a,b) for a,b in zip(originals,current))
    properties=torch.cuda.get_device_properties(0)
    result={'schema':'batched_kv_witness_gpu_gate_v1','scope':'synthetic audit only; no video speedup claim',
        'torch_version':torch.__version__,'device':properties.name,
        'self_layers':24,'cross_layers':24,'source_dtype':'bfloat16','repeat_count':20,
        'all_descriptors_and_hashes_equal':True,'all_cache_tensors_bitwise_unchanged':True,
        'serial_seconds':timings[False],'batched_seconds':timings[True],
        'median_serial_seconds':statistics.median(timings[False]),
        'median_batched_seconds':statistics.median(timings[True]),
        'ratio_of_medians':statistics.median(timings[False])/statistics.median(timings[True]),
        'witness_sha256':reference[0]}
    with (root/'result.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_seconds') or not isinstance(v,list)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
