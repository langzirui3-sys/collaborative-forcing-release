"""Native LingBot 2.0 temporal geometry; never silently accept a short action path.

Pinned source: Robbyant/lingbot-world-v2@1895d300d8ac936401689b26389f51cbd36530eb.
wan/configs/shared_config.py uses 16 FPS; wan/image2video.py rounds RGB frames
by the VAE stride and then rounds latent frames down to a full four-frame block (the official CLI default).
"""
from __future__ import annotations

from fractions import Fraction

SOURCE_REVISION = '1895d300d8ac936401689b26389f51cbd36530eb'
FPS = 16


class LingBotGeometryError(ValueError):
    pass


def temporal_contract(duration_seconds: int, pose_count: int) -> dict:
    if type(duration_seconds) is not int or duration_seconds not in (15, 30, 60):
        raise LingBotGeometryError('nominal duration must be 15, 30, or 60 seconds')
    if type(pose_count) is not int or pose_count < 1:
        raise LingBotGeometryError('pose count must be a positive integer')
    requested = ((duration_seconds * FPS - 1) // 4) * 4 + 1
    latent = ((requested - 1) // 4 + 1) // 4 * 4
    requested = (latent - 1) * 4 + 1
    available = ((pose_count - 1) // 4) * 4 + 1
    clamped = min(requested, available)
    actual_latent = ((clamped - 1) // 4 + 1) // 4 * 4
    actual_rgb = (actual_latent - 1) * 4 + 1 if actual_latent else 0
    return {
        'schema': 'lingbot_native_temporal_contract_v1',
        'source_revision': SOURCE_REVISION,
        'nominal_duration_seconds': duration_seconds,
        'fps': FPS, 'requested_rgb_frames': requested, 'requested_latent_frames': latent,
        'pose_count': pose_count, 'native_rgb_frames': actual_rgb,
        'native_latent_frames': actual_latent,
        'exact_duration_seconds': str(Fraction(actual_rgb, FPS)),
        'would_truncate': actual_rgb != requested,
        'input_policy': 'official_unmodified_prefix',
    }


def require_horizon(duration_seconds: int, pose_count: int) -> dict:
    contract = temporal_contract(duration_seconds, pose_count)
    if contract['would_truncate']:
        raise LingBotGeometryError(
            f"official action path would truncate {contract['requested_rgb_frames']} "
            f"to {contract['native_rgb_frames']} frames at {FPS} FPS; "
            'a separately frozen input extension policy is required')
    return contract


def validate_decoded_video(contract: dict, *, frame_count: int, fps: int) -> None:
    expected = require_horizon(contract['nominal_duration_seconds'], contract['pose_count'])
    if contract != expected:
        raise LingBotGeometryError('temporal contract differs from the native input binding')
    if type(frame_count) is not int or type(fps) is not int or frame_count != expected['requested_rgb_frames'] or fps != FPS:
        raise LingBotGeometryError('decoded frame count or FPS differs from the native temporal contract')
