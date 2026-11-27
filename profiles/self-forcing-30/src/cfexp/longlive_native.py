"""Pinned LongLive 1.0 configuration and strict two-stage weight loading.

This is the native backend building block, not a formal experiment entry point.
A worker must materialize verified upstream sources and Wan assets before
constructing its native pipeline; formal runs still enter through run_remote.sh.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE_REVISION = "2aa857f595d35c81526938d603ae7b1082922ac6"
WEIGHT_REVISION = "cda9138d93872ef109d9a74c270184eb3a5acc6e"
SEED = 20260911


class LongLiveNativeError(ValueError):
    pass


def native_config(*, duration_seconds: int, base_checkpoint: Path,
                  lora_checkpoint: Path) -> dict[str, Any]:
    """Freeze the official non-interactive longlive_inference.yaml settings."""
    if type(duration_seconds) is not int or duration_seconds not in (15, 30, 60):
        raise LongLiveNativeError("LongLive requires a 15, 30, or 60 second horizon")
    return {
        "denoising_step_list": [1000, 750, 500, 250],
        "warp_denoising_step": True,
        "num_frame_per_block": 3,
        "model_name": "Wan2.1-T2V-1.3B",
        "model_kwargs": {"local_attn_size": 12, "timestep_shift": 5.0, "sink_size": 3},
        "num_output_frames": duration_seconds * 6,
        "use_ema": False,
        "seed": SEED,
        "num_samples": 1,
        "global_sink": True,
        "context_noise": 0,
        "distributed": False,
        "generator_ckpt": str(base_checkpoint),
        "lora_ckpt": str(lora_checkpoint),
        "adapter": {"type": "lora", "rank": 256, "alpha": 256,
                    "dropout": 0.0, "dtype": "bfloat16", "verbose": False},
    }


def _state(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value or any(not isinstance(k, str) for k in value):
        raise LongLiveNativeError(f"{label} must be a nonempty state dictionary")
    return value


def load_native_weights(*, pipeline: Any, config: Any, torch: Any,
                        configure_lora: Any, peft: Any) -> dict[str, Any]:
    """Load base strictly, attach official LoRA, then require complete LoRA coverage.

    The caller verifies file hashes before calling this function. Loading is
    deliberately before dtype/device conversion, matching official inference.py.
    A failed load must discard the partially initialized pipeline.
    """
    if config.use_ema is not False or dict(config.adapter) != native_config(
        duration_seconds=15, base_checkpoint=Path("base"), lora_checkpoint=Path("lora")
    )["adapter"]:
        raise LongLiveNativeError("LongLive native weight configuration drifted")
    pipeline.is_lora_enabled = False
    base = _state(torch.load(config.generator_ckpt, map_location="cpu", weights_only=True), "base checkpoint")
    # Follow upstream's generator/model envelope choice, never use generator_ema.
    key = "generator" if "generator" in base else "model" if "model" in base else None
    if key is None:
        raise LongLiveNativeError("base checkpoint lacks generator/model state")
    pipeline.generator.load_state_dict(_state(base[key], "generator"), strict=True)
    del base
    pipeline.generator.model = configure_lora(
        pipeline.generator.model, model_name="generator", lora_config=config.adapter,
        is_main_process=False,
    )
    checkpoint = _state(torch.load(config.lora_ckpt, map_location="cpu", weights_only=True), "LoRA checkpoint")
    supplied = _state(checkpoint.get("generator_lora", checkpoint), "LoRA")
    expected = _state(peft.get_peft_model_state_dict(pipeline.generator.model), "native LoRA structure")
    if set(expected) != set(supplied):
        raise LongLiveNativeError("LoRA checkpoint keys do not exactly cover the native adapter")
    for name in expected:
        if not isinstance(supplied[name], torch.Tensor) or tuple(supplied[name].shape) != tuple(expected[name].shape):
            raise LongLiveNativeError(f"LoRA tensor shape/type mismatch: {name}")
    result = peft.set_peft_model_state_dict(pipeline.generator.model, supplied)
    if result is None or result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
        raise LongLiveNativeError("native LoRA loader reported incompatible adapter keys")
    pipeline.is_lora_enabled = True
    return {"source_revision": SOURCE_REVISION, "weight_revision": WEIGHT_REVISION,
            "base_state_key": key, "lora_tensor_count": len(supplied),
            "lora_rank": 256, "lora_alpha": 256, "use_ema": False}


def generate_native(*, pipeline: Any, noise: Any, prompt: str, torch: Any,
                    device: Any) -> tuple[Any, dict[str, Any]]:
    from .native_autoregressive import generate_autoregressive
    if not getattr(pipeline, "is_lora_enabled", False):
        raise LongLiveNativeError("native generation requires verified LoRA loading")
    return generate_autoregressive(pipeline=pipeline, noise=noise, prompt=prompt,
                                   torch=torch, device=device, error_type=LongLiveNativeError)
