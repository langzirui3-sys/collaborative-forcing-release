"""Compare the entire native attention wrapper, not only its CUDA kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from .full_length_attention import (ATTENTION_SHA,FLASH_FUNCTION_SHA,
                                    FullLengthAttention,load_native)
from .rope_coefficients import ARCHIVE_SHA

SHAPES=((1560,1560),(4680,4680),(4680,9360),(4680,32760),(1560,512),(4680,512))


def tensor_sha(x,torch):
    return hashlib.sha256(x.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def measure(operation,torch,device,*,calls=5):
    torch.cuda.synchronize(device);started=time.perf_counter()
    for _ in range(calls):operation()
    torch.cuda.synchronize(device)
    result=(time.perf_counter()-started)/calls
    if not math.isfinite(result) or result<=0:raise ValueError('invalid wrapper timing')
    return result


def main(argv=None):
    from .self_forcing_frame_adapter import _safe_extract_wheel
    from .main_table_baseline_campaign import (_sha256_file,_visible_devices,
        _gpu_compute_pids,_gpu_uuid,_write_new_json)
    from .main_table_ours_campaign import _assert_idle_pair
    from .stage_timing import _unsealed_directory
    from . import full_length_attention as helper
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-artifact-spec',type=Path,required=True)
    parser.add_argument("--h800-archive", action="store_true")
    args=parser.parse_args(argv)
    root=_unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']).resolve(strict=True))
    models=Path(os.environ['CFEXP_MODEL_ROOT']).resolve(strict=True)
    spec=json.loads(args.draft_artifact_spec.read_text())
    artifact=next(x for x in spec['artifacts'] if x['name']=='flash_attn_wheel')
    if artifact['root']!='model' or artifact['sha256']!='ca5a5614530bad3bcaf936917d53a6d60c9057d1218030efe3ca588cb4f4ebb6':
        raise ValueError('unexpected native FA2 wheel')
    wheel=(models/artifact['path']).resolve(strict=True)
    if not wheel.is_relative_to(models) or _sha256_file(wheel)[0]!=artifact['sha256']:
        raise ValueError('native FA2 wheel differs')
    site=root/'tmp/native-fa2-wheel';_safe_extract_wheel(wheel,site);sys.path.insert(0,str(site))
    devices=_visible_devices();_assert_idle_pair(*devices)
    import torch
    import flash_attn
    if torch.cuda.device_count()!=2 or str(torch.__version__)!='2.5.1+cu121' or flash_attn.__version__!='2.8.3':
        raise ValueError('native attention probe runtime differs')
    archive_sha = ARCHIVE_SHA
    if args.h800_archive:
        import warnings
        from . import h800_native_probe_archive
        archive_sha = h800_native_probe_archive.ARCHIVE_SHA
        native = h800_native_probe_archive.load_function(
            models, module_path="wan/modules/attention.py", module_sha=ATTENTION_SHA,
            name="flash_attention", function_sha=FLASH_FUNCTION_SHA,
            scope={"torch": torch, "flash_attn": flash_attn, "warnings": warnings,
                   "FLASH_ATTN_2_AVAILABLE": True, "FLASH_ATTN_3_AVAILABLE": False})
    else:
        archive = models / 'source/Self-Forcing-33593df3e81fa3ec10239271dd2c100facac6de1.tar.gz'
        native = load_native(torch, flash_attn, archive)
    results=[]
    for device in (0,1):
        with torch.cuda.device(device),torch.inference_mode():
            if torch.cuda.get_device_capability(device)!=(9,0):raise ValueError('probe requires Hopper')
            candidate=FullLengthAttention(native,flash_attn.flash_attn_func,torch)
            for ordinal,(nq,nk) in enumerate(SHAPES):
                if any(not _gpu_compute_pids(_gpu_uuid(d))<={os.getpid()} for d in devices):
                    raise ValueError('selected GPU acquired unrelated compute')
                torch.manual_seed(20260911+ordinal)
                q=torch.randn((1,nq,12,128),device='cuda',dtype=torch.bfloat16)
                # A selected view of a larger cache must retain its native strides.
                k=torch.randn((1,nk+8,12,128),device='cuda',dtype=torch.bfloat16)[:,4:4+nk]
                v=torch.randn((1,nk+8,12,128),device='cuda',dtype=torch.bfloat16)[:,4:4+nk]
                inputs=[tensor_sha(x,torch) for x in (q,k,v)]
                a=native(q,k,v);b=candidate(q,k,v)
                if not torch.equal(a,b):raise ValueError('full native/dense attention differs in output bits')
                output_sha=tensor_sha(a,torch)
                for _ in range(5):native(q,k,v);candidate(q,k,v)
                times={'native':[],'dense':[]}
                before=candidate.evidence()
                for repeat in range(20):
                    for key in (('native','dense') if repeat%2==0 else ('dense','native')):
                        operation=native if key=='native' else candidate
                        times[key].append(measure(lambda:operation(q,k,v),torch,device))
                if [tensor_sha(x,torch) for x in (q,k,v)]!=inputs:
                    raise ValueError('attention probe mutated native inputs')
                after=candidate.evidence()
                delta={k:after[k]-before[k] for k in before}
                if delta!={'calls':100,'dense_calls':100,'fallbacks':0}:
                    raise ValueError('native length-setup bypass did not actually execute')
                row={'device_index':device,'device_name':torch.cuda.get_device_name(device),
                    'shape':[1,nq,nk,12,128],'dtype':'bfloat16',
                    'input_sha256':inputs,'input_strides':[list(x.stride()) for x in (q,k,v)],
                    'inputs_unchanged':True,'bitwise_equal':True,'output_sha256':output_sha,
                    'timing_seconds_per_call':times,'repeat_count':20,'calls_per_sample':5,
                    'median_native_seconds':statistics.median(times['native']),
                    'median_dense_seconds':statistics.median(times['dense']),
                    'wrapper_speedup_ratio_of_medians':statistics.median(times['native'])/statistics.median(times['dense']),
                    'execution_counts':delta}
                results.append(row);print(json.dumps({k:row[k] for k in
                    ('device_index','shape','median_native_seconds','median_dense_seconds','wrapper_speedup_ratio_of_medians')}),flush=True)
                del q,k,v,a,b
            del candidate
            torch.cuda.empty_cache()
    if any(not _gpu_compute_pids(_gpu_uuid(d))<={os.getpid()} for d in devices):
        raise ValueError('selected GPU acquired unrelated compute')
    result={'schema':'native_full_attention_wrapper_probe_v1','rows':results,
        'scope':'unmasked single-sample native wrapper including CPU lengths and GPU prefix sums',
        'torch_version':str(torch.__version__),'flash_attn_version':flash_attn.__version__,
        'native_archive_sha256':archive_sha,'native_module_sha256':ATTENTION_SHA,
        'native_function_sha256':FLASH_FUNCTION_SHA,'helper_sha256':hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
        'generation_worker_integration':False,'video_quality_evaluated':False,'end_to_end_speedup_claim':False}
    _write_new_json(root/'result.json',result)
    return 0


if __name__=='__main__':raise SystemExit(main())
