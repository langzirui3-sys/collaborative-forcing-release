"""Bounded exact RoPE coefficient reuse; diagnostic only, no worker patch."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tarfile

ARCHIVE_SHA = '2ec791e1224511b4b62971a0c236a918d960bdba38676957cfc090b1f85954bf'
MODULE_SHA = 'ae2644311ad212977260549e4b6034618647b82e2ee47a244bdaac50d8ff0778'
FUNCTION_SHA = 'c6f076f439ddd2f76438662c0a83232afb819bd7a95d8becf6b89f7595714e90'


def load_native(torch, archive_path: Path):
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('native SF archive differs')
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive.getmembers()
                   if m.name.endswith('/wan/modules/causal_model.py')]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError('native RoPE module missing or ambiguous')
        source = archive.extractfile(members[0]).read()
    if hashlib.sha256(source).hexdigest() != MODULE_SHA:
        raise ValueError('native RoPE module differs')
    text = source.decode()
    nodes = [n for n in ast.parse(text).body
             if isinstance(n, ast.FunctionDef) and n.name == 'causal_rope_apply']
    if len(nodes) != 1:
        raise ValueError('native RoPE function missing or ambiguous')
    function = ast.get_source_segment(text, nodes[0])
    if hashlib.sha256(function.encode()).hexdigest() != FUNCTION_SHA:
        raise ValueError('native RoPE function differs')
    scope = {'torch':torch}
    exec(compile(function, '<pinned-native-causal-rope>', 'exec'), scope)
    return scope['causal_rope_apply']


class RopeCoefficientCache:
    """One coefficient tensor, keyed by source identity/version and position.

    Unversioned inference tensors fall back to the original function. Holding
    the source object prevents Python/storage identity reuse from hitting stale
    coefficients. No input activation, attention KV or history is retained.
    """
    def __init__(self, native, torch, *, max_bytes=8*1024**2):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive coefficient memory bound required')
        self.native, self.torch, self.max_bytes = native, torch, max_bytes
        self._source = self._key = self._coefficient = None
        self.calls = self.hits = self.misses = self.fallbacks = self.peak_bytes = 0

    def clear(self):
        self._source = self._key = self._coefficient = None

    def evidence(self):
        size = (0 if self._coefficient is None else
                self._coefficient.numel()*self._coefficient.element_size())
        return {'calls':self.calls, 'hits':self.hits, 'misses':self.misses,
                'fallbacks':self.fallbacks, 'resident_coefficient_bytes':size,
                'peak_coefficient_bytes':self.peak_bytes, 'bound_bytes':self.max_bytes,
                'coefficient_count':int(self._coefficient is not None)}

    def __call__(self, x, grid_sizes, freqs, start_frame=0):
        t = self.torch
        self.calls += 1
        try:
            version = freqs._version
        except RuntimeError:
            version = None
        if (version is None or freqs.requires_grad or freqs.dtype != t.complex128
                or grid_sizes.device.type != 'cpu' or tuple(grid_sizes.shape) != (1,3)
                or x.ndim != 4 or x.shape[0] != 1 or x.shape[3] % 2
                or type(start_frame) is not int or start_frame < 0):
            self.fallbacks += 1
            self.clear()
            return self.native(x, grid_sizes, freqs, start_frame)
        f,h,w = grid_sizes.tolist()[0]
        n,c = x.size(2), x.size(3)//2
        seq_len = f*h*w
        size = seq_len*c*freqs.element_size()
        if (min(f,h,w) <= 0 or seq_len != x.size(1) or freqs.ndim != 2
                or freqs.size(1) != c or max(start_frame+f,h,w) > freqs.size(0)
                or size > self.max_bytes):
            self.fallbacks += 1
            self.clear()
            return self.native(x, grid_sizes, freqs, start_frame)
        key = (version, freqs.data_ptr(), tuple(freqs.shape), tuple(freqs.stride()),
               str(freqs.device), f,h,w,c,start_frame)
        if self._source is freqs and self._key == key:
            self.hits += 1
            coefficient = self._coefficient
        else:
            self.misses += 1
            self.clear()  # release the previous constant before allocating another
            split = freqs.split([c-2*(c//3), c//3, c//3],dim=1)
            coefficient = t.cat([
                split[0][start_frame:start_frame+f].view(f,1,1,-1).expand(f,h,w,-1),
                split[1][:h].view(1,h,1,-1).expand(f,h,w,-1),
                split[2][:w].view(1,1,w,-1).expand(f,h,w,-1)
            ],dim=-1).reshape(seq_len,1,-1)
            self._source, self._key, self._coefficient = freqs,key,coefficient
            self.peak_bytes = max(self.peak_bytes,size)
        # Same conversion, reshape and complex multiply as the original. For
        # exactly one unpadded sample, empty-tail cat and batch stack are copies.
        value = t.view_as_complex(x[0].to(t.float64).reshape(seq_len,n,-1,2))
        value = t.view_as_real(value*coefficient).flatten(2)
        return value.unsqueeze(0).type_as(x)
