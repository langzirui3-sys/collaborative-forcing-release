"""Opt-in placement of LongLive 1.0's scalar cache metadata.

The pinned LongLive model reads int64[1] cursors with item() and writes them
with fill_(). K/V storage and LongLive's deferred updates are untouched. This
module never implements attention, cache rolling, prefix replay or sampling.
"""
from __future__ import annotations

import os

ENV = 'CFEXP_LONGLIVE_HOST_CURSORS'
FIELDS = ('global_end_index', 'local_end_index')
LAYERS = 30


def enabled():
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(ENV+' must be 0 or 1')
    return value == '1'


def _layers(cache, torch):
    if not isinstance(cache, (list, tuple)) or len(cache) != LAYERS:
        raise ValueError('LongLive host cursors require its thirty native layers')
    for layer in cache:
        if not isinstance(layer, dict):
            raise ValueError('invalid LongLive native cache layer')
        k, v = layer.get('k'), layer.get('v')
        if (not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor)
                or k.ndim != 4 or k.shape != v.shape or k.device != v.device
                or k.dtype != v.dtype):
            raise ValueError('invalid native LongLive K/V storage')
        for name in FIELDS:
            x = layer.get(name)
            if (not isinstance(x, torch.Tensor) or x.dtype != torch.int64
                    or tuple(x.shape) != (1,)):
                raise ValueError('LongLive cursor must be int64[1]')
    return cache


def relocate(cache, *, torch):
    # Complete validation/copies before committing any metadata replacement.
    layers = _layers(cache, torch)
    replacements = []
    for layer in layers:
        for name in FIELDS:
            host = layer[name].detach().to(device='cpu', copy=True)
            if host.item() < 0:
                raise ValueError('negative LongLive cursor')
            replacements.append((layer, name, host))
    identities = [(id(x['k']), id(x['v'])) for x in layers]
    for layer, name, host in replacements:
        layer[name] = host
    if identities != [(id(x['k']), id(x['v'])) for x in layers]:
        raise ValueError('LongLive K/V identity changed')
    return len(replacements)


def validate_host(cache, *, torch):
    for layer in _layers(cache, torch):
        for name in FIELDS:
            x = layer[name]
            if x.device.type != 'cpu' or x.item() < 0:
                raise ValueError('native LongLive cursor is not nonnegative host metadata')


class CheckedGenerator:
    """Session-local facade; the native module and all its hooks stay intact."""
    def __init__(self, generator, policy):
        self.native, self.policy = generator, policy

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, *args, **kwargs):
        validate_host(kwargs.get('kv_cache'), torch=self.policy.torch)
        self.policy.checked_calls += 1
        return self.native(*args, **kwargs)


class HostCursors:
    def __init__(self, *, torch):
        self.enabled = enabled()
        self.torch = torch
        self.allocations = self.relocated = self.checked_calls = 0

    def initialize(self, cache):
        if self.enabled:
            self.relocated += relocate(cache, torch=self.torch)
            self.allocations += 1

    def snapshot(self):
        return self.allocations, self.relocated, self.checked_calls

    def receipt(self, before):
        a, r, c = before
        result = dict(schema='longlive_host_cursor_execution_v1', enabled=True,
            native_layers=LAYERS, scalar_storage='cpu_int64_singleton',
            allocations_relocated=self.allocations-a,
            cursor_tensors_relocated=self.relocated-r,
            native_generator_calls_checked=self.checked_calls-c,
            kv_storage_and_native_deferred_updates_unchanged=True)
        validate_evidence(result)
        return result


def validate_evidence(proof):
    if (not isinstance(proof, dict)
            or proof.get('schema') != 'longlive_host_cursor_execution_v1'
            or proof.get('enabled') is not True or proof.get('native_layers') != LAYERS
            or proof.get('scalar_storage') != 'cpu_int64_singleton'
            or proof.get('kv_storage_and_native_deferred_updates_unchanged') is not True
            or any(type(proof.get(k)) is not int or proof[k] < 0 for k in
                   ('allocations_relocated','cursor_tensors_relocated','native_generator_calls_checked'))
            or proof['cursor_tensors_relocated'] != LAYERS*2*proof['allocations_relocated']
            or proof['native_generator_calls_checked'] < 5):
        raise ValueError('missing or invalid LongLive host cursor execution evidence')


def audit_checkpoints(campaign, key, chunk_count):
    """Require proof on each committed Draft, not just an enabled environment."""
    import hashlib
    import json
    rows = []
    for index in range(chunk_count):
        path = campaign._checkpoint_path('rollouts', key, index)
        if path.is_symlink() or not path.is_file():
            raise ValueError('LongLive cursor checkpoint is missing or a symlink')
        relative = path.resolve().relative_to(campaign.run_dir.resolve()).as_posix()
        raw = path.read_bytes()
        generated = json.loads(raw)['generated']
        if (generated['context']['role'] not in ('draft','target')
                or generated['context']['next_chunk_index'] != index):
            raise ValueError('LongLive cursor receipt has the wrong role or chunk position')
        if generated['context']['role'] == 'draft':
            proof = generated.get('diagnostics', {}).get('longlive_host_cursor_execution')
            validate_evidence(proof)
            rows.append(dict(chunk_index=index, execution=proof,
                checkpoint=dict(path=relative,sha256=hashlib.sha256(raw).hexdigest(),nbytes=len(raw))))
    if len(rows) != chunk_count-5:
        raise ValueError('LongLive fixed-route Draft receipt coverage differs')
    return dict(schema='longlive_host_cursor_campaign_audit_v1', draft_chunks=len(rows), rows=rows)
