"""Native Rolling B3 undo record; no Target K/V or speculative future is reused.

Only a potentially evicted first working block is copied. On rejection, shift
the retained working interval back and restore that block and native cursors.
Non-populated storage is not part of the accepted prefix. Production callers
remain disabled until the trained Target handoff gate is bound and verified.
"""
from __future__ import annotations

from .incremental_draft import latent_digest
from .rolling_forcing_stream import RollingForcingError


def require(ok, message):
    if not ok:
        raise RollingForcingError(message)


class AcceptedPrefix:
    """One-shot transfer of this pipeline's own accepted clean cache ownership."""
    def __init__(self, *, pipeline, torch, history, prompt, condition):
        self.pipeline = pipeline
        self.caches = (pipeline.kv_cache_clean, pipeline.crossattn_cache)
        self.history_digest = latent_digest(history, torch)
        self.prompt, self.condition = prompt, condition
        self.used = False

    def consume(self, *, pipeline, torch, history, prompt):
        require(not self.used and pipeline is self.pipeline and prompt == self.prompt
                and history is not None
                and latent_digest(history, torch) == self.history_digest,
                'accepted prefix capability belongs to another history or owner')
        require(pipeline.kv_cache_clean is self.caches[0]
                and pipeline.crossattn_cache is self.caches[1],
                'accepted prefix cache owner changed')
        expected = history.shape[1]*pipeline.frame_seq_length
        for layer in self.caches[0]:
            require(int(layer['global_end_index'].item()) == expected
                    and int(layer['local_end_index'].item()) == min(expected, layer['k'].shape[1]),
                    'accepted prefix native cursor differs')
        self.used = True
        return self.condition


class PrefixTransaction:
    def __init__(self, *, pipeline, torch, prefix_blocks, history_digest):
        self.pipeline, self.torch = pipeline, torch
        self.prefix_blocks, self.history_digest = prefix_blocks, history_digest
        self.block_tokens = 3*pipeline.frame_seq_length
        self.caches = (pipeline.kv_cache_clean, pipeline.crossattn_cache)
        require(type(prefix_blocks) is int and prefix_blocks >= 0
                and isinstance(self.caches[0], list) and bool(self.caches[0])
                and len(self.caches[0]) == len(self.caches[1]), 'invalid native transaction prefix')
        self.rows = []
        self.saved_bytes = 0
        self.consumed = False
        for layer, cross in zip(*self.caches):
            k, v = layer['k'], layer['v']
            g, e = (int(layer[name].item()) for name in ('global_end_index', 'local_end_index'))
            capacity = k.shape[1]
            require(k.ndim == 4 and k.shape == v.shape and k.dtype == v.dtype
                    and k.device == v.device and capacity % self.block_tokens == 0
                    and capacity >= 2*self.block_tokens
                    and g == prefix_blocks*self.block_tokens and e == min(g, capacity),
                    'transaction requires an exact native accepted prefix')
            require(type(cross['is_init']) is bool and (not prefix_blocks or cross['is_init']),
                    'transaction conditioning state is invalid')
            lost = max(0, e+self.block_tokens-capacity)
            require(lost in (0, self.block_tokens), 'non-B3 native eviction')
            saved = {name: layer[name][:, self.block_tokens:self.block_tokens+lost].clone()
                     for name in ('k', 'v')} if lost else {}
            self.saved_bytes += sum(x.numel()*x.element_size() for x in saved.values())
            self.rows.append(dict(global_end=g, local_end=e, capacity=capacity, lost=lost,
                pointers=(k.data_ptr(), v.data_ptr()), saved=saved,
                cross_init=cross['is_init'],
                cross_versions={name:(cross[name].data_ptr(), cross[name]._version)
                                for name in ('k', 'v')}))

    def matches(self, history):
        if self.consumed or history is None or history.shape[1] < self.prefix_blocks*3:
            return False
        if self.prefix_blocks == 0:
            return self.history_digest is None
        return latent_digest(history[:, :self.prefix_blocks*3], self.torch) == self.history_digest

    def rollback(self):
        require(not self.consumed, 'native prefix transaction was already consumed')
        p, n = self.pipeline, self.block_tokens
        require(p.kv_cache_clean is self.caches[0] and p.crossattn_cache is self.caches[1],
                'native prefix transaction cache owner changed')
        # Validate every layer before changing any cache storage.
        for layer, cross, row in zip(*self.caches, self.rows):
            require((layer['k'].data_ptr(), layer['v'].data_ptr()) == row['pointers']
                    and int(layer['global_end_index'].item()) == row['global_end']+n
                    and int(layer['local_end_index'].item()) == min(row['local_end']+n, row['capacity']),
                    'native transaction is not immediately after one candidate')
            if row['cross_init']:
                require(cross['is_init'] is True and all(
                    (cross[name].data_ptr(), cross[name]._version) == witness
                    for name, witness in row['cross_versions'].items()),
                    'native conditioning changed during a candidate')
        with self.torch.no_grad():
            for layer, cross, row in zip(*self.caches, self.rows):
                lost, e = row['lost'], row['local_end']
                if lost:
                    for name in ('k', 'v'):
                        layer[name][:, n+lost:e] = layer[name][:, n:e-lost].clone()
                        layer[name][:, n:n+lost] = row['saved'][name]
                layer['global_end_index'].fill_(row['global_end'])
                layer['local_end_index'].fill_(e)
                cross['is_init'] = row['cross_init']
        self.consumed = True
        return dict(schema='rolling_native_prefix_rollback_v1', prefix_blocks=self.prefix_blocks,
                    copied_undo_bytes=self.saved_bytes,
                    evicted_tokens_per_layer=[row['lost'] for row in self.rows],
                    unused_slots_restored=False)
