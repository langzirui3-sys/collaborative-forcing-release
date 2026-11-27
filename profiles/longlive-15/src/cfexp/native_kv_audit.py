"""Observe the pure Draft loop without changing its native cache updates."""
from __future__ import annotations


class NativeKVAudit:
    def __init__(self):
        self.identity = None
        self.positions = []
        self.calls = 0

    def before(self, kwargs):
        cache = kwargs.get('kv_cache')
        position = kwargs.get('current_start')
        if not isinstance(cache, list) or not cache or type(position) is not int:
            raise ValueError('native forward omitted KV cache or position')
        identity = (id(cache), tuple((id(layer), id(layer['k']), id(layer['v']),
            layer['k'].data_ptr(), layer['v'].data_ptr()) for layer in cache))
        if self.identity is None:
            self.identity = identity
        elif identity != self.identity:
            raise ValueError('native Draft replaced its persistent KV storage')
        if not self.positions or position != self.positions[-1]:
            if position != len(self.positions) * 3 * 1560:
                raise ValueError('native KV cursor skipped or replayed history')
            if any(int(layer['global_end_index'].item()) != position for layer in cache):
                raise ValueError('native KV prefix does not match the next chunk')
            self.positions.append(position)
        self.calls += 1

    def finish(self, chunk_count):
        if len(self.positions) != chunk_count or self.calls != chunk_count * 5:
            raise ValueError('native KV audit has incomplete chunk/forward coverage')
        return {'schema':'pure_draft_native_kv_audit_v1', 'chunk_count':chunk_count,
            'generator_calls':self.calls, 'persistent_kv_storage_verified':True,
            'prefix_cursor_verified':True, 'history_reconstruction_calls':0,
            'chunk_start_token_positions':self.positions}
