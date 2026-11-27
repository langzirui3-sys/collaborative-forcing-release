"""Coalesce small audit transfers without changing sampled cache evidence."""
from __future__ import annotations


def host_lists(tensors, *, torch, float32=False):
    """Return per-tensor flat lists, with one host transfer per device/dtype.

    No model tensor is modified. Keeping source dtypes separate preserves integer
    cursors exactly; in particular, int64 cursors never pass through a float.
    Samples are cast at the same host-transfer boundary as the serial witness.
    """
    groups = {}
    for index, value in enumerate(tensors):
        groups.setdefault((str(value.device), str(value.dtype)), []).append((index, value))
    result = [None] * len(tensors)
    for group in groups.values():
        counts = [int(value.numel()) for _, value in group]
        joined = torch.cat([value.detach().reshape(-1) for _, value in group])
        options = {"device": "cpu"}
        if float32:
            options["dtype"] = torch.float32
        values = joined.to(**options).tolist()
        start = 0
        for (index, _), count in zip(group, counts):
            result[index] = values[start:start + count]
            start += count
        if start != len(values):
            raise ValueError("batched witness transfer length changed")
    return result
