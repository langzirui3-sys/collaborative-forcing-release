"""One-request CUDA kernel diagnostics; disabled in normal generation."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

ENV = 'CFEXP_TORCH_PROFILE_CHUNK'


def selected_chunk():
    raw = os.environ.get(ENV)
    if raw is None:
        return None
    if not raw.isascii() or not raw.isdecimal() or str(int(raw)) != raw or not 1 <= int(raw) < 120:
        raise ValueError(f'{ENV} must be a canonical integer from 1 to 119')
    return int(raw)


def summarize_events(events):
    """Aggregate actual CUDA events, not inclusive CPU operator device totals."""
    kernels = {}
    for event in events:
        if str(event.device_type) != 'DeviceType.CUDA':
            continue
        name = str(event.name)
        duration = float(event.device_time_total)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('invalid CUDA event duration')
        kind = 'transfer' if 'memcpy' in name.lower() else 'memset' if 'memset' in name.lower() else 'kernel'
        row = kernels.setdefault(name, {'name': name, 'kind': kind, 'count': 0, 'cuda_us': 0.0})
        row['count'] += 1
        row['cuda_us'] += duration
    return sorted(kernels.values(), key=lambda row: (-row['cuda_us'], row['name']))


def toolchain(torch):
    nvcc = shutil.which('nvcc')
    version = None
    if nvcc is not None:
        try:
            process = subprocess.run([nvcc, '--version'], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
            version = {'returncode': process.returncode, 'output': process.stdout[-4096:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            version = {'error': type(error).__name__}
    return {'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda,
            'device_capability': list(torch.cuda.get_device_capability()),
            'nvcc_path': nvcc, 'nvcc_version': version}


@contextmanager
def profile_request(backend, *, run_dir, metadata):
    chunk = selected_chunk()
    action, role = metadata.get('action'), metadata.get('role')
    if chunk is None or action not in ('generate', 'generate_persistent') or metadata.get('chunk_index') != chunk:
        yield
        return
    if role not in ('draft', 'target'):
        raise ValueError('invalid kernel-profile role')
    key = (role, action, chunk)
    seen = getattr(backend, '_cfexp_kernel_profiles', set())
    if key in seen:
        yield
        return
    from .stage_timing import _unsealed_directory
    from .correction_worker import (_output_path, _temporary_sibling, _sha256_file,
        _StagedOutput, _publish_staged, _discard_staged)
    root = _unsealed_directory(Path(run_dir))
    torch = getattr(backend, 'torch', None)
    if (torch is None or not torch.cuda.is_available()
            or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities()):
        raise ValueError('CUDA kernel profiling is unavailable; no CPU-only fallback')
    prefix = f'provenance/kernel-profile/{role}-{action}-chunk-{chunk:02d}'
    outputs = [_output_path(root, prefix+suffix, label='kernel diagnostic output')
               for suffix in ('.json', '.trace.json')]
    environment = toolchain(torch)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
            profile_memory=False, with_stack=False) as profiler:
        yield
    kernels = summarize_events(profiler.events())
    if not kernels:
        raise ValueError('CUDA profiler captured no kernels')
    operators = []
    for event in profiler.key_averages():
        if str(event.device_type) == 'DeviceType.CPU':
            operators.append({'name': str(event.key), 'count': int(event.count),
                'self_cpu_us': float(event.self_cpu_time_total),
                'self_cuda_us': float(event.self_device_time_total)})
    report = {'schema': 'native_kernel_profile_v1', 'role': role, 'action': action,
        'chunk_index': chunk, 'request_id': metadata.get('request_id'),
        'prompt_sha256': metadata.get('prompt_sha256'), 'toolchain': environment,
        'cuda_device_activity_us_sum': sum(row['cuda_us'] for row in kernels),
        'cuda_kernel_us_sum': sum(row['cuda_us'] for row in kernels if row['kind'] == 'kernel'),
        'kernels': kernels, 'operators': sorted(operators, key=lambda row: -row['self_cuda_us']),
        'scope': 'instrumented request, not video-ready latency or a speedup claim; kernel sums do not represent wall time or account for overlap'}
    staged = []
    temporaries = [_temporary_sibling(path) for _, path in outputs]
    try:
        temporaries[0].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
        profiler.export_chrome_trace(str(temporaries[1]))
        for (relative, target), temporary in zip(outputs, temporaries):
            digest, size = _sha256_file(temporary)
            staged.append(_StagedOutput(relative_path=relative, target=target,
                temporary=temporary, metadata={'sha256': digest, 'nbytes': size}))
        _publish_staged(staged)
    except BaseException:
        _discard_staged(staged)
        for path in temporaries:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    backend._cfexp_kernel_profiles = set(seen) | {key}
