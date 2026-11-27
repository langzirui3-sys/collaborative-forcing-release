"""Formal CUDA comparison of native rolling inference and resumable stream.

This gate tests pure Draft latent equality only. It does not authorize Target
intervention or claim any benchmark score. Eight blocks include warmup, steady
windows, and the three-window drain.
"""
from __future__ import annotations

import os
from pathlib import Path

from .incremental_draft import latent_digest
from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
)
from .metadata import find_git_root
from .rolling_forcing_backend import RollingNativeBackend
from .rolling_forcing_stream import RollingForcingStream

PROMPT = 'a motorcycle accelerating to gain speed'
SEED = 20260911
BLOCKS = 8


def compare(*, backend, output, ownership, observe=False):
    t, p = backend.torch, backend.pipeline
    ownership()
    t.manual_seed(SEED)
    t.cuda.manual_seed_all(SEED)
    noise = t.randn((1, BLOCKS*3, 16, 60, 104), device=backend.device, dtype=t.bfloat16)
    cpu_rng, cuda_rng = t.get_rng_state(), t.cuda.get_rng_state_all()
    original_decode = p.vae.decode_to_pixel
    # VAE has no bearing on latent parity; exclude its memory/time from gate.
    p.vae.decode_to_pixel = lambda latent, **kw: t.zeros((1,1,3,1,1), device=latent.device)
    try:
        with t.inference_mode():
            _, expected = p.inference_rolling_forcing(
                noise=noise.clone(), text_prompts=[PROMPT], return_latents=True, profile=False)
    finally:
        p.vae.decode_to_pixel = original_decode
    expected = expected.detach().cpu().contiguous()
    final_cpu, final_cuda = t.get_rng_state(), t.cuda.get_rng_state_all()
    native_cursors = [{name: int(layer[name].item())
                       for name in ('global_end_index', 'local_end_index')}
                      for layer in p.kv_cache_clean]
    p.kv_cache_clean = None
    p.crossattn_cache = None
    t.cuda.empty_cache()
    t.set_rng_state(cpu_rng)
    t.cuda.set_rng_state_all(cuda_rng)
    stream = None
    observer = None
    rows = []
    try:
        if observe:
            from .rolling_forcing_observation import RollingStepZeroObserver
            observer = RollingStepZeroObserver(pipeline=p, torch=t)
        with t.inference_mode():
            stream = RollingForcingStream(pipeline=p, torch=t, noise=noise, prompt=PROMPT)
            for index in range(BLOCKS):
                ownership()
                actual_index, actual = stream.next_block()
                actual = actual.detach().cpu().contiguous()
                reference = expected[:,index*3:(index+1)*3]
                equal = actual_index == index and bool(t.equal(reference, actual))
                row = {'chunk':index, 'equal':equal,
                       'max_abs_difference':float((actual.float()-reference.float()).abs().max()),
                       'native_sha256':latent_digest(reference,t), 'stream_sha256':latent_digest(actual,t)}
                if observer is not None:
                    observation = observer.take(index)
                    observation.pop('hidden')
                    row['step_zero_observation'] = observation
                rows.append(row)
                _write_new_json(output/f'parity/chunk-{index:03d}.json', row)
                if not equal:
                    _write_new_json(output/'result.json', {'status':'failed',
                        'first_divergent_chunk':index, 'chunks':rows})
                    raise ValueError(f'native Rolling latent divergence at block {index}')
            # The stream owns its RNG; compare its final state to the native
            # sequence, not the ambient state restored by fork_rng.
            if not t.equal(stream.rng_state, final_cuda[0]) or not t.equal(t.get_rng_state(), final_cpu):
                raise ValueError('native Rolling RNG consumption differs')
            try:
                stream.next_block()
            except StopIteration:
                pass
            else:
                raise ValueError('stream returned an extra block after drain')
            if observer is not None and observer.pending:
                raise ValueError('rolling observations remain after drain')
        ownership()
        result = {'schema':'rolling_native_parity_v1','status':'succeeded',
                  'prompt':PROMPT, 'seed':SEED,'blocks':BLOCKS,'chunks':rows,
                  'rng_equal':True,'native_load':backend.audit,
                  'claim_scope':'pure Draft native latent parity only; Target branching untested'}
        if stream.host_kv_cursors:
            from .rolling_host_cursors import evidence
            proof = evidence(p.kv_cache_clean, torch=t)
            if proof['layers'] != native_cursors:
                raise ValueError('native Rolling final host KV cursors differ')
            result['host_cursor_execution'] = proof
            result['native_final_cursors_equal'] = True
        if observe:
            result['observation_policy'] = 'layer14_block_local_step_zero_native_window_v1'
            result['claim_scope'] = ('instrumented pure Draft latent/RNG parity and block-local '
                                     'eRank capture only; Target branching and quality untested')
        _write_new_json(output/'result.json', result)
        return result
    finally:
        if observer is not None:
            observer.close()
        if stream is not None:
            stream.close()


def run(*, observe=False):
    output = _run_dir()
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('Rolling parity requires one assigned GPU')
    uuid = _gpu_uuid(devices[0])
    if _gpu_compute_pids(uuid):
        raise ValueError('assigned GPU is occupied')
    def ownership():
        if not _gpu_compute_pids(uuid) <= {os.getpid()}:
            raise ValueError('assigned GPU acquired an unrelated process')
    backend = None
    try:
        backend = RollingNativeBackend(run_dir=output,
            repository_root=find_git_root(Path.cwd()), model_root=_model_root())
        _write_new_json(output/'provenance/native.json', {
            'verified_artifacts':backend.records,'native_load':backend.audit})
        return compare(backend=backend, output=output, ownership=ownership, observe=observe)
    finally:
        if backend is not None:
            backend.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe-step-zero', action='store_true')
    args = parser.parse_args()
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        run(observe=args.observe_step_zero)
        return 0
    finally:
        shutdown_owned_runtime_helpers(label='rolling-native-parity')


if __name__ == '__main__':
    raise SystemExit(main())
