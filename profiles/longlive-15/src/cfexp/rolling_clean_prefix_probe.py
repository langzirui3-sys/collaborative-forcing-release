"""Diagnostic only: prove whether a completed native Rolling prefix is reusable.

Compares every populated clean K/V element with independent B3 accepted-history
replay. Restores the original cache objects and proves that the diagnostic did
not change following latents, caches or RNG. No optimized handoff is enabled.
"""
from __future__ import annotations

import os
from pathlib import Path

from .incremental_draft import latent_digest
from .rolling_forcing_stream import RollingForcingError, RollingForcingStream

PROMPT = 'a car stuck in traffic during rush hour'
SEED = 20260911
BLOCKS = 16
FOLLOW_THROUGH = 13
PREFIX_MARKS = (1, 4, 8, 12)
ENV = 'CFEXP_ROLLING_CLEAN_PREFIX_DIAGNOSTIC'


def _require(value, message):
    if not value:
        raise RollingForcingError(message)


def _tensors(caches):
    return [value for group in caches for layer in group
            for value in layer.values() if hasattr(value, 'data_ptr')]


def compare_caches(actual, expected, *, torch, layers=30):
    """Compare complete populated storage, not sampled witnesses or free slots."""
    _require(type(layers) is int and layers > 0, 'invalid native layer count')
    a, ax = actual
    b, bx = expected
    _require(len(a) == len(b) == len(ax) == len(bx) == layers,
             'native cache layer coverage differs')
    rows = []
    for index, (u, v, ux, vx) in enumerate(zip(a, b, ax, bx)):
        _require(set(u) == set(v) == {'k', 'v', 'global_end_index', 'local_end_index'}
                 and set(ux) == set(vx) == {'k', 'v', 'is_init'},
                 'native cache fields differ')
        cursors = {}
        for name in ('global_end_index', 'local_end_index'):
            _require(u[name].dtype == v[name].dtype == torch.int64
                     and tuple(u[name].shape) == tuple(v[name].shape) == (1,)
                     and torch.equal(u[name], v[name]), 'native cache cursors differ')
            cursors[name] = int(u[name].item())
        end = cursors['local_end_index']
        _require(u['k'].ndim == v['k'].ndim == 4
                 and u['k'].shape == u['v'].shape and v['k'].shape == v['v'].shape
                 and 0 < end <= u['k'].shape[1]
                 and cursors['global_end_index'] >= end,
                 'invalid populated native cache interval')
        count = 0
        for name in ('k', 'v'):
            x, y = u[name], v[name]
            _require(x.ndim == y.ndim == 4 and x.shape == y.shape
                     and x.dtype == y.dtype and x.device == y.device,
                     'native K/V geometry differs')
            _require(torch.equal(x[:, :end], y[:, :end]),
                     f'clean prefix {name} differs at native layer {index}')
            count += x[:, :end].numel()
        _require(type(ux['is_init']) is bool and ux['is_init'] is vx['is_init'] is True,
                 'native cross-attention conditioning is not initialized')
        for name in ('k', 'v'):
            x, y = ux[name], vx[name]
            _require(x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
                     and torch.equal(x, y), 'native cross-attention K/V differs')
            count += x.numel()
        rows.append(dict(layer=index, **cursors, compared_elements=count))
    return dict(schema='rolling_full_populated_cache_equality_v1', equal=True,
                layers=rows, unused_storage_compared=False)


def replay_probe(*, pipeline, torch, history, condition, ownership, layers=30):
    """Fresh oracle uses separate storage; restore live object identity even on error."""
    t, p = torch, pipeline
    _require(history.ndim == 5 and history.shape[0] == 1
             and history.shape[1] >= 3 and history.shape[1] % 3 == 0,
             'probe requires complete accepted B3 history')
    original = (p.kv_cache_clean, p.crossattn_cache)
    tensors = _tensors(original)
    versions = [(value, value.data_ptr(), value._version) for value in tensors]
    blocks = history.shape[1] // 3
    devices = [history.device.index if history.device.index is not None
               else t.cuda.current_device()] if history.device.type == 'cuda' else []
    try:
        p.kv_cache_clean = p.crossattn_cache = None
        with t.random.fork_rng(devices=devices), t.no_grad():
            p._initialize_kv_cache(batch_size=1, dtype=history.dtype, device=history.device)
            p._initialize_crossattn_cache(batch_size=1, dtype=history.dtype, device=history.device)
            oracle = (p.kv_cache_clean, p.crossattn_cache)
            _require(not ({x.data_ptr() for x in tensors} &
                          {x.data_ptr() for x in _tensors(oracle)}),
                     'fresh prefix oracle aliases live cache storage')
            for index in range(blocks):
                ownership()
                p.generator(noisy_image_or_video=history[:, index*3:(index+1)*3],
                    conditional_dict=condition,
                    timestep=t.zeros((1, 3), device=history.device, dtype=t.float32),
                    kv_cache=p.kv_cache_clean, crossattn_cache=p.crossattn_cache,
                    current_start=index*3*p.frame_seq_length, updating_cache=True)
            comparison = compare_caches(original, oracle, torch=t, layers=layers)
        return dict(history_blocks=blocks, clean_b3_forwards=blocks, comparison=comparison,
                    history_sha256=latent_digest(history, t))
    finally:
        p.kv_cache_clean, p.crossattn_cache = original
        _require(all(value.data_ptr() == pointer and value._version == version
                     for value, pointer, version in versions),
                 'diagnostic mutated original native cache storage')


def compare(*, backend, output, ownership, hardware):
    """Two complete identical trajectories; only the first contains oracle probes."""
    from .main_table_baseline_campaign import _write_new_json
    t, p = backend.torch, backend.pipeline
    _require(p.num_transformer_blocks == 30 and p.frame_seq_length == 1560,
             'unexpected trained native Rolling geometry')
    ownership()
    t.manual_seed(SEED)
    t.cuda.manual_seed_all(SEED)
    noise = t.randn((1, BLOCKS*3, 16, 60, 104), device=backend.device, dtype=t.bfloat16)
    cpu_rng, gpu_rng = t.get_rng_state().clone(), t.cuda.get_rng_state().clone()
    first = second = None
    latents, probes, continuations = [], [], []
    try:
        with t.no_grad():
            first = RollingForcingStream(pipeline=p, torch=t, noise=noise, prompt=PROMPT)
            for index in range(FOLLOW_THROUGH):
                ownership()
                actual_index, value = first.next_block()
                _require(actual_index == index, 'probed native position differs')
                latents.append(value.detach().cpu().contiguous())
                if index+1 in PREFIX_MARKS:
                    history = t.cat(latents, dim=1).to(device=backend.device)
                    proof = replay_probe(pipeline=p, torch=t, history=history,
                        condition=first.condition, ownership=ownership)
                    _write_new_json(output/f'probe/prefix-{index+1:02d}.json', proof)
                    probes.append(proof)
            original_caches = (p.kv_cache_clean, p.crossattn_cache)
            first_rng = first.rng_state.clone()
            _require(t.equal(t.get_rng_state(), cpu_rng)
                     and t.equal(t.cuda.get_rng_state(), gpu_rng),
                     'probed stream changed ambient RNG')
            # Keep the live first branch untouched while the independent second
            # trajectory regenerates the same prefix and its following blocks.
            p.kv_cache_clean = p.crossattn_cache = None
            second = RollingForcingStream(pipeline=p, torch=t, noise=noise, prompt=PROMPT)
            for index, expected in enumerate(latents):
                ownership()
                actual_index, actual = second.next_block()
                _require(actual_index == index
                         and t.equal(expected, actual.detach().cpu()),
                         f'probe changed subsequent native latent at block {index}')
                continuations.append(dict(chunk=index,
                    probed_sha256=latent_digest(expected, t),
                    unprobed_sha256=latent_digest(actual, t), equal=True))
            final = compare_caches(original_caches,
                (p.kv_cache_clean, p.crossattn_cache), torch=t)
            _require(t.equal(first_rng, second.rng_state)
                     and t.equal(t.get_rng_state(), cpu_rng)
                     and t.equal(t.cuda.get_rng_state(), gpu_rng),
                     'probed and unprobed native RNG streams differ')
        result = dict(schema='rolling_clean_prefix_diagnostic_v1', status='succeeded',
            prompt=PROMPT, seed=SEED, noise_blocks=BLOCKS,
            hardware=hardware,
            generated_blocks=FOLLOW_THROUGH, prefix_marks=list(PREFIX_MARKS),
            probes=probes, continuation=continuations, final_cache_equality=final,
            rng_equal=True, native_load=backend.audit,
            claim_scope='pure Draft clean-prefix equality only; Target replacement, rollback, fusion, quality and speed untested')
        _write_new_json(output/'result.json', result)
        return result
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()


def main():
    import socket
    from .main_table_baseline_campaign import (
        _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
    )
    from .metadata import find_git_root
    from .rolling_forcing_backend import RollingNativeBackend
    from .runtime_helpers import shutdown_owned_runtime_helpers
    _require(os.environ.get(ENV) == '1', 'clean-prefix probe is diagnostic opt-in only')
    _require(os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS', '0') == '0'
             and os.environ.get('CFEXP_ROLLING_HOST_WINDOW', '0') == '0',
             'clean-prefix gate requires unchanged native cursor/window execution')
    output = _run_dir()
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    _require(len(devices) == 1 and bool(devices[0].strip()), 'one assigned GPU required')
    uuid = _gpu_uuid(devices[0])
    _require(not _gpu_compute_pids(uuid), 'assigned GPU is occupied')
    def ownership():
        _require(_gpu_compute_pids(uuid) <= {os.getpid()},
                 'assigned GPU acquired an unrelated process')
    backend = None
    try:
        backend = RollingNativeBackend(run_dir=output,
            repository_root=find_git_root(Path.cwd()), model_root=_model_root())
        _write_new_json(output/'provenance/native.json', dict(
            verified_artifacts=backend.records, native_load=backend.audit))
        return 0 if compare(backend=backend, output=output, ownership=ownership,
            hardware=dict(hostname=socket.gethostname(), gpu_uuid=uuid,
                          visible_gpu_count=1, assigned_gpu=devices[0])) else 1
    finally:
        if backend is not None:
            backend.close()
        shutdown_owned_runtime_helpers(label='rolling-clean-prefix-probe')


if __name__ == '__main__':
    raise SystemExit(main())
