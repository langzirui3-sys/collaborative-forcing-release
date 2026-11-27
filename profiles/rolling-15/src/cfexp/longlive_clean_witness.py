"""Sample the accepted clean state inside a native LongLive mixed forward.

Each layer exposes its temporary cache immediately before processing the noisy
Draft block. Only the canonical 64-value samples are retained. Reading the
ordinary cache after the complete forward would instead witness the new noisy
slot, because LongLive applies its deferred updates at the model boundary.
"""
import hashlib
import json

from .longlive_joint_handoff import require
from .self_forcing_frame_adapter import sampled_kv_cache_witness


class CleanHistoryWitness:
    def __init__(self, *, pipeline, torch, expected_start):
        require(type(expected_start) is int and expected_start > 0
                and expected_start % (3*pipeline.frame_seq_length) == 0,
                'clean witness requires the logical next Draft position')
        self.pipeline, self.torch = pipeline, torch
        self.expected_start = expected_start
        self.layers = []
        self.finished = False

    def observe(self, index, k, v, global_end, local_end):
        t = self.torch
        capacity = 12*self.pipeline.frame_seq_length
        require(not self.finished and type(index) is int and index == len(self.layers)
                and index < 30 and type(global_end) is int and type(local_end) is int
                and global_end == self.expected_start and local_end == min(global_end, capacity)
                and isinstance(k, t.Tensor) and isinstance(v, t.Tensor)
                and k.ndim == v.ndim == 4 and k.shape == v.shape
                and k.shape[0] == 1 and k.shape[1] == capacity
                and k.device == v.device and k.dtype == v.dtype == t.bfloat16,
                'LongLive clean observation order, position, precision or geometry differs')
        # Force immediate samples before the next noisy group mutates temp K/V.
        # No full model tensor, temporary view or device sample survives here.
        _, witness = sampled_kv_cache_witness([dict(k=k, v=v,
            global_end_index=global_end, local_end_index=local_end)],
            batched=False, reuse_indices=False)
        record = witness['layers'][0]
        record['layer'] = index
        self.layers.append(record)

    def finish(self):
        require(not self.finished and len(self.layers) == 30,
                'clean witness requires exactly thirty observed layers and one finish')
        p = self.pipeline
        require(len(p.kv_cache1) == len(p.crossattn_cache) == 30
                and all(layer.get('is_init') is True for layer in p.crossattn_cache),
                'clean witness requires all initialized native cross-attention layers')
        first = p.kv_cache1[0]
        # The zero-length dummy prevents sampling the now-updated self cache.
        # Cross KV was already initialized by the retained accepted prefix.
        _, descriptor = sampled_kv_cache_witness([dict(k=first['k'], v=first['v'],
            global_end_index=0, local_end_index=0)], p.crossattn_cache,
            batched=False, reuse_indices=False)
        descriptor['layers'] = self.layers
        digest = hashlib.sha256(json.dumps(descriptor, ensure_ascii=False,
            allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        self.finished = True
        return digest, descriptor
