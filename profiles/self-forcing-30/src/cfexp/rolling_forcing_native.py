"""Pinned Causal-Forcing long-video loader; independent of SF/LongLive loaders.

The caller materializes the verified official ``long_video`` source tree and
its Wan runtime files before importing its pipeline in an isolated worker.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .rolling_forcing_stream import RollingForcingError

SOURCE_COMMIT = 'eb8705b2f00b7c361193b25aa3aae9e7edc94667'
CHECKPOINT_SHA256 = '1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98'
CHECKPOINT_SIZE = 17028883814
CONFIG_HASHES = {
    'configs/default_config.yaml': 'e43b0c52c16398413ed11ea2c8b6029c4fd550573788b46b864b668d69ca6fa3',
    'configs/rolling_forcing_dmd.yaml': '4e31cac08a1b200f85d4bde01fa2427f1938241589831b04b8c2bbd02e29ccd7',
}


def _check_file(path, digest, size=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RollingForcingError(f'expected regular native artifact: {path.name}')
    actual = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            actual.update(block)
        count = handle.tell()
    if actual.hexdigest() != digest or (size is not None and count != size):
        raise RollingForcingError(f'native artifact hash/size mismatch: {path.name}')


def native_config(source_root, omega_conf):
    source_root = Path(source_root)
    for relative, digest in CONFIG_HASHES.items():
        _check_file(source_root/relative, digest)
    config = omega_conf.merge(omega_conf.load(source_root/'configs/default_config.yaml'),
                              omega_conf.load(source_root/'configs/rolling_forcing_dmd.yaml'))
    if (config.num_frame_per_block != 3 or config.independent_first_frame
            or config.context_noise != 0 or not config.warp_denoising_step
            or tuple(config.denoising_step_list) != (1000,750,500,250)
            or config.model_kwargs.timestep_shift != 5.0):
        raise RollingForcingError('merged native long-video configuration drifted')
    return config


def _ema_state(checkpoint, torch):
    if not isinstance(checkpoint, Mapping):
        raise RollingForcingError('checkpoint must be a state envelope')
    state = checkpoint.get('generator_ema')
    if not isinstance(state, Mapping) or not state:
        raise RollingForcingError('official long-video EMA state is missing')
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key or not torch.is_tensor(value):
            raise RollingForcingError('invalid EMA state entry')
        # Match the official CLI --use_ema path, but reject ambiguous keys.
        name = key.replace('_fsdp_wrapped_module.', '')
        if not name or name in cleaned:
            raise RollingForcingError('duplicate EMA key after FSDP prefix removal')
        cleaned[name] = value
    return cleaned


def load_native_pipeline(*, source_root, checkpoint_path, torch, omega_conf,
                         pipeline_type, device):
    """Use native config + EMA weights, with strict matching before GPU transfer."""
    config = native_config(source_root, omega_conf)
    _check_file(checkpoint_path, CHECKPOINT_SHA256, CHECKPOINT_SIZE)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = _ema_state(checkpoint, torch)
    pipeline = pipeline_type(config, device=device)
    # No fallback to base Wan weights, a non-EMA envelope, or strict=False.
    pipeline.generator.load_state_dict(state, strict=True)
    key_count = len(state)
    del state, checkpoint
    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval().requires_grad_(False)
    if tuple(float(x) for x in pipeline.denoising_step_list) != (1000.,937.5,833.3333129882812,625.):
        raise RollingForcingError('native warped schedule drifted')
    return pipeline, {
        'source_commit': SOURCE_COMMIT,
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'checkpoint_envelope': 'generator_ema',
        'strict_state_load': True,
        'state_key_count': key_count,
        'official_output_fps': 16,
        'model_variant': 'Causal Forcing official long-video extension based on Rolling Forcing',
    }
