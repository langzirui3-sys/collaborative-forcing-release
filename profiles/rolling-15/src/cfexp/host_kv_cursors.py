"""Opt-in host placement for native scalar KV cursors, never K/V tensors.

The pinned native attention reads these counters with .item() and updates them
with .fill_(). Keeping counters on the host avoids reading CUDA scalars after
each layer. This does not change the native rolling-window algorithm.
"""
from __future__ import annotations

import os
from typing import Any

ENV = "CFEXP_HOST_KV_CURSORS"
FIELDS = ("global_end_index", "local_end_index")


def enabled() -> bool:
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def relocate(cache: Any, *, torch: Any) -> int:
    """Validate the complete cache before replacing only its scalar metadata."""
    if not isinstance(cache, (list, tuple)) or not cache:
        raise ValueError("native KV cache must contain layers")
    replacements = []
    for layer in cache:
        if not isinstance(layer, dict):
            raise ValueError("native KV layer must be a dictionary")
        for name in FIELDS:
            value = layer.get(name)
            # Krea's pinned native implementation already stores Python ints.
            # Preserve that representation; its attention uses integer arithmetic
            # directly rather than Tensor.item()/fill_().
            if type(value) is int:
                if value < 0:
                    raise ValueError("native KV cursor must be nonnegative")
                continue
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                    or tuple(value.shape) != (1,)):
                raise ValueError("native KV cursor must be a one-element int64 tensor")
            host = value.detach().to(device="cpu", copy=True)
            if host.item() < 0:
                raise ValueError("native KV cursor must be nonnegative")
            replacements.append((layer, name, host))
    for layer, name, host in replacements:
        layer[name] = host
    return len(replacements)


def install(pipeline: Any, *, torch: Any) -> dict[str, Any]:
    """Wrap each native allocation, including later prompt/session resets.

    Existing live caches are forbidden. The default does not wrap or alter
    anything. Fail closed on unsupported native interfaces; never fall back.
    """
    selected = enabled()
    evidence = {"schema": "host_native_kv_cursors_v1", "enabled": selected,
                "native_python_integer_cursors_preserved": True,
                "kv_tensor_placement_unchanged": True,
                "native_attention_algorithm_unchanged": True}
    if not selected:
        return evidence
    if getattr(pipeline, "_cfexp_host_kv_cursors_installed", False):
        raise ValueError("native cursor optimization installed twice")
    if getattr(pipeline, "kv_cache1", None) is not None:
        raise ValueError("install cursor placement before any native cache exists")
    original = getattr(pipeline, "_initialize_kv_cache", None)
    if not callable(original):
        raise ValueError("native pipeline has no cache initializer")

    def initialize(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        relocate(pipeline.kv_cache1, torch=torch)
        return result

    pipeline._initialize_kv_cache = initialize
    pipeline._cfexp_host_kv_cursors_installed = True
    return evidence
