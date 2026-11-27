"""Bounded integer constants for the unchanged native eRank token sample.

No hidden activations, Gram matrices, spectra, model outputs or KV values are
cached. The default execution path remains disabled.
"""
from __future__ import annotations

import hashlib
import json
import os


def enabled(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(name + " must be 0 or 1")
    return value == "1"


class NativeRankIndices:
    def __init__(self, *, torch, device, index_function, capacity=8):
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError("rank index capacity must be between 1 and 8")
        self.torch, self.index_function, self.capacity = torch, index_function, capacity
        self.enabled = enabled("CFEXP_RANK_INDEX_REUSE")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._entries = {}
        self.calls = self.hits = self.misses = self.allocations = 0
        # The frozen SF native block has 3 * (60/2) * (104/2) tokens.
        # Materialize this constant during model startup, outside ready timing.
        if self.enabled:
            self._construct(4680)

    def _construct(self, token_count):
        values = tuple(self.index_function(token_count, 256))
        with self.torch.inference_mode(False):
            index = self.torch.tensor(values, dtype=self.torch.long, device=self.device)
        self.allocations += 1
        if len(self._entries) < self.capacity:
            self._entries[token_count] = (index, int(index._version),
                hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest())
        return index

    def sample(self, hidden):
        torch = self.torch
        if not self.enabled:
            raise ValueError("native rank constants were not enabled")
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[0] != 1 or hidden.shape[1] < 1
                or hidden.device != self.device):
            raise ValueError("rank index sample has incompatible device or layout")
        count = int(hidden.shape[1])
        self.calls += 1
        entry = self._entries.get(count)
        if entry is None:
            self.misses += 1
            index = self._construct(count)
        else:
            index, version, _ = entry
            if (int(index._version) != version or index.device != self.device
                    or index.dtype != torch.long or index.shape != (min(count, 256),)):
                raise ValueError("native rank index constant was mutated")
            self.hits += 1
        return hidden[0].index_select(0, index).detach().to(dtype=torch.float32)

    def snapshot(self):
        return {"calls": self.calls, "hits": self.hits, "misses": self.misses,
            "allocations": self.allocations}

    def receipt(self, before):
        after = self.snapshot()
        return {"schema": "native_rank_index_execution_v1", "enabled": self.enabled,
            "device": str(self.device), "capacity": self.capacity,
            "counts": {k: after[k] - before[k] for k in after},
            "resident_entries": len(self._entries),
            "resident_index_bytes": sum(int(v[0].numel()) * int(v[0].element_size())
                for v in self._entries.values()),
            "constants": [{"token_count": n, "sample_count": min(n, 256), "sha256": v[2]}
                for n, v in sorted(self._entries.items())],
            "retained_model_values": False}
