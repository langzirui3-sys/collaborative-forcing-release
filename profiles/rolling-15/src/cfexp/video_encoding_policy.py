"""Explicit SF encoding policies; legacy runs are never reinterpreted.

Serial ABR keeps torchvision's bitrate/pixel format and fixes encoder threads.
It is a reproducibility control, not an algorithmic quality improvement.
"""
from __future__ import annotations
import os
from pathlib import Path

LEGACY = 'legacy_abr_v1'
SERIAL = 'serial_abr_v1'
ENV = 'CFEXP_VIDEO_ENCODING'


def validate_policy(value: str) -> str:
    if value not in (LEGACY, SERIAL):
        raise ValueError(f'{ENV} must be {LEGACY} or {SERIAL}')
    return value


def policy() -> str:
    return validate_policy(os.environ.get(ENV, LEGACY))


def writer_options() -> dict:
    return {} if policy() == LEGACY else {'video_codec': 'libx264', 'options': {'threads': '1'}}


def verify_video(path: Path, selected: str | None = None) -> str:
    """Require actual x264 option evidence for the new opt-in policy."""
    selected = policy() if selected is None else validate_policy(selected)
    if selected == SERIAL:
        raw = path.read_bytes()
        start = raw.find(b'x264 - core ')
        if start < 0:
            raise ValueError('serial encoding has no x264 option evidence')
        options = raw[start:start+2000]
        for item in (b' threads=1 ', b' lookahead_threads=1 ', b' sliced_threads=0 ',
                     b' rc=abr ', b' bitrate=1024 '):
            if item not in options:
                raise ValueError('serial encoder options differ from the frozen ABR policy')
    return selected
