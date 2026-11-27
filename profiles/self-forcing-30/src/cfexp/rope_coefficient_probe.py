"""Pinned native RoPE CUDA parity/timing diagnostic, not video generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from .rope_coefficients import (ARCHIVE_SHA, MODULE_SHA, FUNCTION_SHA,
                                RopeCoefficientCache, load_native)


def tensor_sha(tensor, torch):
    value=tensor.detach().to('cpu').contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def main(argv=None):
    import torch
    from .stage_timing import _unsealed_directory
    root=_unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']).resolve(strict=True))
    model_root=Path(os.environ['CFEXP_MODEL_ROOT']).resolve(strict=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h800-archive", action="store_true")
    args = parser.parse_args(argv)
    archive_sha = ARCHIVE_SHA
    if args.h800_archive:
        from . import h800_native_probe_archive
        archive_sha = h800_native_probe_archive.ARCHIVE_SHA
        native = h800_native_probe_archive.load_function(
            model_root, module_path="wan/modules/causal_model.py", module_sha=MODULE_SHA,
            name="causal_rope_apply", function_sha=FUNCTION_SHA, scope={"torch": torch})
    else:
        archive = model_root / 'source/Self-Forcing-33593df3e81fa3ec10239271dd2c100facac6de1.tar.gz'
        native = load_native(torch, archive)
    if torch.cuda.device_count()!=2:
        raise ValueError('probe requires exactly two reserved visible CUDA devices')
    results=[]
    for device in (0,1):
        with torch.cuda.device(device):
            torch.manual_seed(20260911)
            freqs=torch.polar(torch.ones((1024,64),dtype=torch.float64),
                torch.randn((1024,64),dtype=torch.float64)).to('cuda:'+str(device))
            cache=RopeCoefficientCache(native,torch)
            for f in (1,3):
                grid=torch.tensor([[f,30,52]],dtype=torch.int64)
                x=torch.randn((1,f*30*52,12,128),device="cuda:"+str(device),dtype=torch.bfloat16)
                x_sha=tensor_sha(x,torch);freq_sha=tensor_sha(freqs,torch)
                parity=[]
                for position in (0,3,12,45,87):
                    a=native(x,grid,freqs,position);b=cache(x,grid,freqs,position)
                    equal=torch.equal(a,b)
                    if not equal:raise ValueError('RoPE output is not bitwise equal')
                    parity.append({'start_frame':position,'output_sha256':tensor_sha(a,torch),
                                   'bitwise_equal':equal})
                # Version changes must invalidate the last constant on CUDA.
                freqs.mul_(complex(0.999,0.001))
                if not torch.equal(native(x,grid,freqs,87),cache(x,grid,freqs,87)):
                    raise ValueError('source mutation did not invalidate coefficient')
                mutated_sha=tensor_sha(freqs,torch)
                for _ in range(10):
                    native(x,grid,freqs,87);cache(x,grid,freqs,87)
                torch.cuda.synchronize(device)
                timings={'native':[],'cached':[]}
                for repeat in range(20):
                    order=('native','cached') if repeat%2==0 else ('cached','native')
                    for label in order:
                        operation=native if label=='native' else cache
                        torch.cuda.synchronize(device);started=time.perf_counter()
                        for _ in range(30):operation(x,grid,freqs,87)
                        torch.cuda.synchronize(device)
                        timings[label].append((time.perf_counter()-started)/30)
                if tensor_sha(x,torch)!=x_sha or tensor_sha(freqs,torch)!=mutated_sha:
                    raise ValueError('probe changed activation or source frequencies')
                evidence=cache.evidence()
                if evidence['hits']<=0 or evidence['peak_coefficient_bytes']>evidence['bound_bytes']:
                    raise ValueError('coefficient reuse or bound evidence missing')
                results.append({'device_index':device,'device_name':torch.cuda.get_device_name(device),
                    'shape':list(x.shape),'dtype':'bfloat16','frequency_dtype':'complex128',
                    'grid':grid.tolist(),'initial_frequency_sha256':freq_sha,
                    'mutated_frequency_sha256':mutated_sha,'source_mutation_checked':True,
                    'input_sha256':x_sha,'inputs_unchanged':True,'parity':parity,
                    'cache':evidence,'timing_seconds_per_call':timings,
                    'median_native_seconds':statistics.median(timings['native']),
                    'median_cached_seconds':statistics.median(timings['cached']),
                    'kernel_speedup_ratio_of_medians':statistics.median(timings['native'])/
                        statistics.median(timings['cached'])})
                del x,a,b
            cache.clear()
            del cache,freqs
            torch.cuda.empty_cache()
    result={'schema':'native_rope_coefficient_cuda_probe_v1',
        'scope':'synthetic native-kernel parity and hot-position cost only',
        'video_quality_evaluated':False,'end_to_end_speedup_claim':False,
        'generation_worker_integration':False,'torch_version':torch.__version__,
        'archive_sha256':archive_sha,'module_sha256':MODULE_SHA,'function_sha256':FUNCTION_SHA,
        'repeat_count':20,'calls_per_timing_sample':30,'rows':results}
    with (root/'result.json').open('x') as file:json.dump(result,file,indent=2)
    for row in results:
        print(json.dumps({k:row[k] for k in ('device_index','grid','median_native_seconds',
            'median_cached_seconds','kernel_speedup_ratio_of_medians','cache')}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
