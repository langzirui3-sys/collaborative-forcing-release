"""Motion energy of an accepted three-frame LongLive Draft latent."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

from .correction_state import LatentSegment


MOTION_ENERGY_FLOOR = 0.24
_SHAPE = [1, 3, 16, 60, 104]
_FRAME_ELEMENTS = 16 * 60 * 104
_SAMPLE_OFFSETS = tuple(
    2 * ((channel * 60 + row) * 104 + column)
    for channel in range(16)
    for row in range(0, 60, 8)
    for column in range(0, 104, 8)
)
_BF16 = struct.unpack('<65536f', b''.join(struct.pack('<I', i << 16) for i in range(65536)))


def latent_motion_energy(root: Path, segment: LatentSegment) -> float:
    """Read the committed BF16 tensor and normalize frame differences by magnitude."""
    if segment.producer_role != 'draft' or segment.model_id != 'longlive_1.3b':
        raise ValueError('motion energy requires an accepted LongLive Draft latent')
    root = root.resolve()
    path = root / segment.latent_path
    if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('unsafe motion latent path')
    raw = path.read_bytes()
    if len(raw) != segment.latent_nbytes or hashlib.sha256(raw).hexdigest() != segment.latent_sha256:
        raise ValueError('motion latent differs from the committed segment')
    header_bytes = struct.unpack_from('<Q', raw)[0]
    if not 8 < header_bytes < 4096:
        raise ValueError('invalid motion latent header size')
    start = 8 + header_bytes
    tensor = json.loads(raw[8:start])['latent']
    data_bytes = 6 * _FRAME_ELEMENTS
    if (tensor['shape'] != _SHAPE or tensor['dtype'] != 'BF16'
            or tensor['data_offsets'] != [0, data_bytes] or len(raw) != start + data_bytes):
        raise ValueError('unexpected LongLive latent layout')
    frames = []
    for index in range(3):
        offset = start + 2 * _FRAME_ELEMENTS * index
        frames.append([_BF16[raw[offset + i] + 256 * raw[offset + i + 1]] for i in _SAMPLE_OFFSETS])
    samples = len(_SAMPLE_OFFSETS)
    magnitude = math.fsum(abs(value) for frame in frames for value in frame) / (3 * samples)
    difference = math.fsum(abs(a - b) for first, second in zip(frames, frames[1:])
                          for a, b in zip(first, second, strict=True)) / (2 * samples)
    if not math.isfinite(magnitude) or not math.isfinite(difference):
        raise ValueError('nonfinite LongLive latent motion energy')
    return difference / magnitude if magnitude else 0.0


def low_motion_pair(previous: float | None, current: float | None) -> bool:
    return (previous is not None and current is not None
            and previous < MOTION_ENERGY_FLOOR and current < MOTION_ENERGY_FLOOR)
