"""Isolated exact-attention kernel diagnostic; never a video speed claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys


SHAPES = (
    ('draft_root', 4680, 4680, 12, 128),
    ('draft_context_6', 4680, 9360, 12, 128),
    ('draft_context_21', 4680, 32760, 12, 128),
    ('draft_text', 4680, 512, 12, 128),
    ('target_context_6', 4680, 9360, 40, 128),
    ('target_text', 4680, 512, 40, 128),
)


def dense_cudnn(torch, q, k, v, *, contiguous_inputs=False, backend='cudnn'):
    """Full, noncausal, unpadded BF16 attention; reject every other contract."""
    if (any(x.ndim != 4 or x.dtype != torch.bfloat16 for x in (q, k, v))
            or q.shape[0] != 1 or k.shape != v.shape
            or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]
            or any(x.device != q.device for x in (k, v))
            or any(x.stride(-1) != 1 for x in (q, k, v))):
        raise ValueError('expected matching unpadded [1,S,H,D] BF16 tensors')
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if backend not in ('cudnn', 'torch_flash'):
        raise ValueError('unsupported explicit SDPA backend')
    inputs = [x.transpose(1, 2) for x in (q, k, v)]
    if contiguous_inputs:
        inputs = [x.contiguous() for x in inputs]
    # Explicit selection forbids a silent fallback to another SDPA backend.
    selected = SDPBackend.CUDNN_ATTENTION if backend == 'cudnn' else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(selected):
        value = torch.nn.functional.scaled_dot_product_attention(
            *inputs,
            dropout_p=0.0, is_causal=False)
    return value.transpose(1, 2).contiguous()


def error_metrics(torch, actual, expected):
    if (actual.shape != expected.shape or not bool(torch.isfinite(actual).all())
            or not bool(torch.isfinite(expected).all())):
        raise ValueError('attention produced invalid output')
    difference = actual.float() - expected.float()
    rms = float(expected.float().square().mean().sqrt().item())
    return {'max_abs': float(difference.abs().max().item()),
            'relative_rmse': float(difference.square().mean().sqrt().item()) / max(rms, 1e-12),
            'bitwise_equal': bool(torch.equal(actual, expected))}


def benchmark(torch, operation, *, warmup=3, repeats=9):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repeats):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        output = operation()
        end.record()
        end.synchronize()
        durations.append(float(begin.elapsed_time(end)))
    if any(not math.isfinite(x) or x <= 0 for x in durations):
        raise ValueError('invalid CUDA event timing')
    return {'cuda_ms': durations, 'median_cuda_ms': statistics.median(durations)}, output


def run(spec_path):
    from .self_forcing_frame_adapter import _safe_extract_wheel
    from .stage_timing import _unsealed_directory
    from .metadata import find_git_root, collect_git_metadata
    from .main_table_baseline_campaign import (_sha256_file, _write_new_json,
        _visible_devices, _gpu_uuid, _gpu_compute_pids)
    from .main_table_ours_campaign import _assert_idle_pair
    # No secret or external runtime output enters the checkout.
    run_dir = _unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']))
    repository = find_git_root(Path.cwd())
    spec = json.loads(spec_path.read_text())
    row = next(x for x in spec['artifacts'] if x['name'] == 'flash_attn_wheel')
    if row['root'] != 'model' or row['sha256'] != 'ca5a5614530bad3bcaf936917d53a6d60c9057d1218030efe3ca588cb4f4ebb6':
        raise ValueError('unexpected FA2 artifact binding')
    model_root = Path(os.environ['CFEXP_MODEL_ROOT']).resolve(strict=True)
    wheel = (model_root / row['path']).resolve(strict=True)
    if not wheel.is_relative_to(model_root) or _sha256_file(wheel)[0] != row['sha256']:
        raise ValueError('FA2 wheel hash mismatch')
    site = run_dir / 'tmp' / 'attention-kernel-wheel'
    _safe_extract_wheel(wheel, site)
    sys.path.insert(0, str(site))
    devices = _visible_devices()
    _assert_idle_pair(*devices)
    import torch
    import flash_attn
    if str(torch.__version__) != '2.5.1+cu121' or flash_attn.__version__ != '2.8.3':
        raise ValueError('diagnostic runtime differs from frozen model runtime')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ValueError('diagnostic requires exactly two visible CUDA devices')
    result = {'schema': 'attention_kernel_probe_v1', 'formal_video_speedup': False,
              'quality_measured': False, 'native_kv_changed': False,
              'shape_scope': 'unmasked full attention on already selected native cache slice',
              'torch': str(torch.__version__), 'cudnn': torch.backends.cudnn.version(),
              'flash_attn': flash_attn.__version__, 'git': collect_git_metadata(repository),
              'source_spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
              'rows': []}
    for ordinal, (name, nq, nk, heads, dim) in enumerate(SHAPES):
        device = 0 if name.startswith('draft') else 1
        with torch.cuda.device(device), torch.inference_mode():
            if any(not _gpu_compute_pids(_gpu_uuid(d)) <= {os.getpid()} for d in devices):
                raise ValueError('selected GPU acquired an unrelated compute process')
            if torch.cuda.get_device_capability() != (9, 0):
                raise ValueError('this diagnostic targets the measured Hopper workload')
            torch.manual_seed(20261104 + ordinal)
            q = torch.randn((1, nq, heads, dim), device='cuda', dtype=torch.bfloat16)
            k = torch.randn((1, nk, heads, dim), device='cuda', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            originals = (q.clone(), k.clone(), v.clone())
            cq = torch.tensor([0, nq], device='cuda', dtype=torch.int32)
            ck = torch.tensor([0, nk], device='cuda', dtype=torch.int32)
            native = lambda: flash_attn.flash_attn_varlen_func(q.flatten(0, 1),
                k.flatten(0, 1), v.flatten(0, 1), cq, ck, nq, nk, dropout_p=0.0,
                causal=False).unsqueeze(0)
            fa2_dense = lambda: flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
            entry = {'name': name, 'shape': [nq, nk, heads, dim], 'gpu': device,
                     'gpu_name': torch.cuda.get_device_name(), 'implementations': {},
                     'native_model_baseline': 'fa2_varlen' if name.startswith('draft') else 'fa2_dense'}
            native_time, reference = benchmark(torch, native)
            entry['implementations']['fa2_varlen'] = native_time
            for implementation, operation in [('fa2_dense', fa2_dense),
                    ('cudnn', lambda: dense_cudnn(torch, q, k, v)),
                    ('cudnn_contiguous', lambda: dense_cudnn(torch, q, k, v, contiguous_inputs=True)),
                    ('torch_flash', lambda: dense_cudnn(torch, q, k, v, backend='torch_flash')),
                    ('torch_flash_contiguous', lambda: dense_cudnn(torch, q, k, v,
                        contiguous_inputs=True, backend='torch_flash'))]:
                try:
                    timing, output = benchmark(torch, operation)
                    metrics = error_metrics(torch, output, reference)
                    entry['implementations'][implementation] = {
                        **timing, 'status': 'measured', 'output_error_vs_fa2_varlen': metrics,
                        'kernel_speedup_vs_varlen': native_time['median_cuda_ms'] / timing['median_cuda_ms']}
                except RuntimeError as error:
                    # Unsupported backends are evidence, not permission to fall back.
                    if implementation == 'fa2_dense':
                        raise
                    entry['implementations'][implementation] = {'status': 'unsupported', 'reason': str(error)[:2048]}
            for key in ('cudnn', 'cudnn_contiguous', 'torch_flash', 'torch_flash_contiguous'):
                candidate = entry['implementations'][key]
                if candidate['status'] == 'measured':
                    baseline = entry['implementations'][entry['native_model_baseline']]
                    candidate['kernel_speedup_vs_model_baseline'] = baseline['median_cuda_ms'] / candidate['median_cuda_ms']
            if not all(torch.equal(x, original) for x, original in zip((q, k, v), originals)):
                raise ValueError('attention implementation mutated its inputs')
            if any(not _gpu_compute_pids(_gpu_uuid(d)) <= {os.getpid()} for d in devices):
                raise ValueError('selected GPU acquired an unrelated compute process')
            entry['inputs_unchanged'] = True
            result['rows'].append(entry)
            _write_new_json(run_dir / 'provenance' / ('attention-kernel-' + name + '.json'), entry)
            print(json.dumps(entry), flush=True)
            del q, k, v, originals, reference, output
    _write_new_json(run_dir / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-artifact-spec', type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.draft_artifact_spec)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
