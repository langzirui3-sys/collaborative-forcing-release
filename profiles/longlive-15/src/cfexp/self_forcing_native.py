"""Pure Self-Forcing baseline, without correction or hidden-state capture.

The formal worker must verify source, config, and weights before calling this
building block. This module is not a standalone formal experiment launcher.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .native_autoregressive import generate_autoregressive

SEED = 20260911


class SelfForcingNativeError(ValueError):
    pass


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        'num_frame_per_block': 3,
        'independent_first_frame': False,
        'denoising_step_list': [1000, 750, 500, 250],
        'warp_denoising_step': True,
        'context_noise': 0,
        'causal': True,
        'i2v': False,
        'same_step_across_blocks': True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SelfForcingNativeError(f'native Self-Forcing configuration drift: {key}')
    model = config.get('model_kwargs', {})
    if (model.get('local_attn_size') != 21 or model.get('sink_size') != 0
            or model.get('model_name') != 'Wan2.1-T2V-1.3B'
            or model.get('timestep_shift') != 5.0):
        raise SelfForcingNativeError('native Self-Forcing requires local context 21 and sink 0')


def load_native_weights(*, pipeline: Any, checkpoint: Any, state_key: str,
                        torch: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact configured state, never fall back between EMA/non-EMA."""
    pipeline._cfexp_native_self_forcing_loaded = False
    validate_config(config)
    if state_key not in ('generator', 'generator_ema'):
        raise SelfForcingNativeError('unsupported configured checkpoint state')
    weights = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    if not isinstance(weights, Mapping) or state_key not in weights:
        raise SelfForcingNativeError('checkpoint lacks configured generator state')
    state = weights[state_key]
    if not isinstance(state, Mapping) or not state:
        raise SelfForcingNativeError('generator state must be a nonempty mapping')
    incompatible = pipeline.generator.load_state_dict(state, strict=True)
    if incompatible is None or incompatible.missing_keys or incompatible.unexpected_keys:
        raise SelfForcingNativeError('native generator weights were not loaded completely')
    pipeline._cfexp_native_self_forcing_loaded = True
    return {'state_key': state_key, 'strict': True, 'tensor_count': len(state),
            'missing_keys': [], 'unexpected_keys': []}


def generate_native(*, pipeline: Any, prompt: str, duration_seconds: int,
                    torch: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    """Start native Draft from noise; no Target root, proposals, or cache rebuild."""
    if type(duration_seconds) is not int or duration_seconds not in (15, 30, 60):
        raise SelfForcingNativeError('native horizon must be 15, 30, or 60 seconds')
    if not getattr(pipeline, '_cfexp_native_self_forcing_loaded', False):
        raise SelfForcingNativeError('native generation requires verified weight loading')
    # Reinitialize the private RNG for each prompt, independent of batch order.
    generator = torch.Generator(device=device).manual_seed(SEED)
    noise = torch.randn((1, duration_seconds * 6, 16, 60, 104),
                        generator=generator, device=device, dtype=torch.bfloat16)
    video, timing = generate_autoregressive(
        pipeline=pipeline, noise=noise, prompt=prompt, torch=torch, device=device,
        error_type=SelfForcingNativeError)
    return video, dict(timing, seed=SEED, target_calls=0,
                       initialization='native_draft_noise')
