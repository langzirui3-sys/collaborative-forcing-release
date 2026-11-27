"""Synthetic call-local witness index gate; no video speedup claim."""
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
    if (root/'result.json').exists() or not torch.cuda.is_available():
        raise ValueError('fresh output and CUDA required')
    torch.manual_seed(20260911)
    results = []
    for used, cross_initialized in ((0, False), (768, True), (4096, True), (12000, True)):
        layers, cross = [], []
        for i in range(30):
            k = torch.randn((1,4096,12,128), device='cuda:0', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            layers.append({'k': k, 'v': v, 'global_end_index': used,
                'local_end_index': min(used,4096)})
            cross.append({'k': k[:,:512].clone(), 'v': v[:,:512].clone(),
                'is_init': cross_initialized})
        before = [x[n].clone() for x in layers+cross for n in ('k','v')]
        reference = sampled_kv_cache_witness(layers, cross, batched=True, reuse_indices=False)
        assert reference == sampled_kv_cache_witness(layers, cross, batched=True, reuse_indices=True)
        for _ in range(3):
            for selected in (False, True):
                sampled_kv_cache_witness(layers, cross, batched=True, reuse_indices=selected)
        timings = {False: [], True: []}
        for repeat in range(30):
            for selected in ((False,True) if repeat%2 == 0 else (True,False)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = sampled_kv_cache_witness(layers, cross, batched=True, reuse_indices=selected)
                torch.cuda.synchronize()
                timings[selected].append(time.perf_counter()-start)
                assert value == reference
        assert all(torch.equal(a,x[n]) for a,(x,n) in zip(before,
            [(x,n) for x in layers+cross for n in ('k','v')]))
        # Fresh invocation must observe changed samples, not cached values.
        if used:
            layers[0]['k'].reshape(-1)[0] += 1
            changed = sampled_kv_cache_witness(layers,cross,batched=True,reuse_indices=True)
            assert changed != reference
            assert changed == sampled_kv_cache_witness(layers,cross,batched=True,reuse_indices=False)
        baseline, optimized = (statistics.median(timings[x]) for x in (False,True))
        results.append({'used_tokens':used,'cross_initialized':cross_initialized,
            'median_baseline_seconds':baseline,'median_reuse_seconds':optimized,
            'ratio_of_medians':baseline/optimized,'seconds_saved':baseline-optimized,
            'baseline_seconds':timings[False],'reuse_seconds':timings[True],
            'all_descriptors_and_hashes_equal':True,'source_tensors_unchanged_before_mutation':True,
            'fresh_mutation_observed':bool(used),'witness_sha256':reference[0]})
        del before,layers,cross,k,v
        torch.cuda.empty_cache()
    result = {'schema':'call_local_witness_indices_cuda_gate_v1',
        'scope':'synthetic audit overhead only; no generated model quality or speedup claim',
        'torch_version':torch.__version__,'device':torch.cuda.get_device_name(0),
        'rows':results,'native_generation_optimization_enabled':False}
    with (root/'result.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k not in
        ('baseline_seconds','reuse_seconds')} for r in results]}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
