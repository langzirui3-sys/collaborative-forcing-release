"""Pinned ImageReward routing support for the SDVG main-table reproduction.

The public SDVG paper does not ship video-generation code.  This adapter
therefore implements only the paper-described reward gate around the already
audited Self-Forcing Draft backend.  It deliberately keeps the Draft VAE cache
on the Draft trajectory: every proposal is decoded, including proposals later
rejected by the Target trajectory.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

from .self_forcing_frame_adapter import _safe_extract_tar, _verified_artifacts


SDVG_ADAPTER_SCHEMA = "sdvg_adapter_config_v1"
SDVG_REWARD_SCHEMA = "sdvg_image_reward_block_v1"
SDVG_SOURCE_COMMIT = "78f8051059b34fef8e0032ccb40561c23a2ddbbb"


class SdvgAdapterError(RuntimeError):
    """Raised when the frozen SDVG reward path cannot be proved."""


def _load_reward_classes(source_root: Path):
    """Load pinned inference modules without package-level training/CLIP imports.

    ImageReward's __init__ eagerly imports ReFL and unrelated score models. The
    SDVG gate uses only its original ImageReward class and BLIP implementation.
    """
    import importlib
    import importlib.machinery
    import types
    for name, relative in (("ImageReward", "ImageReward"),
                           ("ImageReward.models", "ImageReward/models"),
                           ("ImageReward.models.BLIP", "ImageReward/models/BLIP")):
        directory = (source_root / relative).resolve(strict=True)
        existing = sys.modules.get(name)
        if existing is not None:
            _require(list(getattr(existing, "__path__", [])) == [str(directory)],
                     "ImageReward package is already loaded from a different source")
            continue
        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    reward_module = importlib.import_module("ImageReward.ImageReward")
    blip_module = importlib.import_module("ImageReward.models.BLIP.blip_pretrain")
    return reward_module.ImageReward, blip_module


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SdvgAdapterError(message)


def _artifact(
    artifacts: Mapping[str, Path], name: Any, *, label: str
) -> Path:
    _require(
        isinstance(name, str) and name in artifacts,
        f"{label} names an unverified artifact",
    )
    return artifacts[str(name)]


def _link_or_copy(source: Path, destination: Path) -> None:
    _require(not destination.exists() and not destination.is_symlink(), "destination exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _materialize_reward_source(
    *, run_dir: Path, artifacts: Mapping[str, Path], config: Mapping[str, Any]
) -> tuple[Path, Path]:
    scratch = run_dir / "tmp" / f"sdvg-image-reward-{os.getpid()}"
    _require(not scratch.exists(), "SDVG ImageReward scratch path already exists")
    reward = config.get("image_reward")
    _require(isinstance(reward, Mapping), "SDVG image_reward config is absent")
    _require(
        reward.get("source_commit") == SDVG_SOURCE_COMMIT,
        "SDVG ImageReward source commit drifted",
    )
    archive = _artifact(
        artifacts, reward.get("source_archive_artifact"), label="ImageReward source"
    )
    _safe_extract_tar(archive, scratch)
    subdirectory = reward.get("source_archive_subdirectory")
    _require(
        isinstance(subdirectory, str)
        and subdirectory
        and "/" not in subdirectory
        and "\\" not in subdirectory,
        "ImageReward source subdirectory is invalid",
    )
    source_root = (scratch / subdirectory).resolve(strict=True)
    _require(
        source_root.parent == scratch.resolve(strict=True)
        and (source_root / "ImageReward" / "ImageReward.py").is_file()
        and (source_root / "ImageReward" / "models" / "BLIP" / "blip.py").is_file(),
        "ImageReward source archive has an unexpected layout",
    )

    bert_root = scratch / "bert-base-uncased"
    bert_root.mkdir(mode=0o700)
    bert_files = {
        "bert_config": "config.json",
        "bert_model": "model.safetensors",
        "bert_tokenizer": "tokenizer.json",
        "bert_tokenizer_config": "tokenizer_config.json",
        "bert_vocab": "vocab.txt",
    }
    for artifact_name, filename in bert_files.items():
        _link_or_copy(
            _artifact(artifacts, artifact_name, label="ImageReward BERT artifact"),
            bert_root / filename,
        )
    return source_root, bert_root


class SdvgDraftBackend:
    """Delegate Self-Forcing generation and add an offline ImageReward gate."""

    def __init__(
        self,
        *,
        base_backend: Any,
        run_dir: Path,
        adapter_config: Mapping[str, Any],
        verified_artifacts: Sequence[Mapping[str, Any]],
    ) -> None:
        _require(
            adapter_config.get("schema") == SDVG_ADAPTER_SCHEMA,
            "SDVG adapter schema drifted",
        )
        routing = adapter_config.get("routing")
        _require(isinstance(routing, Mapping), "SDVG routing config is absent")
        threshold = routing.get("threshold")
        _require(
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold)),
            "SDVG threshold is invalid",
        )
        _require(
            routing.get("aggregation")
            == "minimum_reward_over_every_decoded_frame_in_block"
            and routing.get("comparison")
            == "accept_if_score_greater_than_or_equal_to_threshold"
            and routing.get("force_reject_chunk_indices") == [0],
            "SDVG routing semantics drifted",
        )
        benchmark = adapter_config.get("benchmark_adaptations")
        _require(
            isinstance(benchmark, Mapping)
            and type(benchmark.get("chunk_count")) is int
            and benchmark.get("chunk_count") in (30, 60, 120)
            and benchmark.get("latent_frames_per_chunk") == 3
            and benchmark.get("rgb_frame_count") == benchmark["chunk_count"] * 12 - 3
            and benchmark.get("fps") == 24
            and benchmark.get("resolution") == [832, 480],
            "SDVG benchmark adaptation drifted",
        )
        self.base = base_backend
        self.chunk_count = int(getattr(base_backend, "chunk_count", -1))
        _require(self.chunk_count == benchmark["chunk_count"], "SDVG Draft horizon differs from declared benchmark")
        self.threshold = float(threshold)
        artifacts = _verified_artifacts({"verified_artifacts": list(verified_artifacts)})
        source_root, bert_root = _materialize_reward_source(
            run_dir=run_dir,
            artifacts=artifacts,
            config=adapter_config,
        )
        reward_config = adapter_config["image_reward"]
        checkpoint = _artifact(
            artifacts,
            reward_config.get("checkpoint_artifact"),
            label="ImageReward checkpoint",
        )
        med_config = _artifact(
            artifacts,
            reward_config.get("med_config_artifact"),
            label="ImageReward med config",
        )

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import torch  # type: ignore
            from transformers import BertTokenizer  # type: ignore
            ImageReward, blip_pretrain = _load_reward_classes(source_root)
        except ImportError as error:
            raise SdvgAdapterError(
                "the frozen Draft runtime lacks ImageReward dependencies"
            ) from error

        tokenizer = BertTokenizer.from_pretrained(
            str(bert_root), local_files_only=True
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        original_init_tokenizer = blip_pretrain.init_tokenizer
        blip_pretrain.init_tokenizer = lambda: tokenizer
        try:
            reward_model = ImageReward(
                med_config=str(med_config), device=base_backend.device
            ).to(base_backend.device)
        finally:
            blip_pretrain.init_tokenizer = original_init_tokenizer
        try:
            state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(str(checkpoint), map_location="cpu")
        message = reward_model.load_state_dict(state_dict, strict=False)
        _require(
            not message.missing_keys and not message.unexpected_keys,
            "ImageReward checkpoint did not load exactly",
        )
        reward_model.eval()
        self.torch = torch
        self.reward_model = reward_model
        self.to_pil_image = __import__(
            "torchvision.transforms.functional", fromlist=["to_pil_image"]
        ).to_pil_image
        self._active_prompt: str | None = None
        self._active_prompt_id: str | None = None
        self._next_chunk_index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def start_sdvg_trajectory(
        self, *, prompt_id: str, prompt: str, chunk_count: int
    ) -> Mapping[str, Any]:
        _require(self._active_prompt is None, "an SDVG trajectory is already active")
        _require(prompt_id and prompt.strip(), "SDVG prompt identity is invalid")
        _require(chunk_count == self.chunk_count, "SDVG chunk horizon drifted")
        self.base.pipeline.vae.model.clear_cache()
        self._active_prompt_id = prompt_id
        self._active_prompt = prompt
        self._next_chunk_index = 0
        return {
            "prompt_id": prompt_id,
            "chunk_count": chunk_count,
            "threshold": self.threshold,
        }

    def score_sdvg_candidate(
        self, *, prompt: str, clean_latent: Any, chunk_index: int
    ) -> Mapping[str, Any]:
        torch = self.torch
        _require(self._active_prompt == prompt, "SDVG prompt drifted")
        _require(
            chunk_index == self._next_chunk_index,
            "SDVG candidates must be scored exactly once in order",
        )
        _require(
            isinstance(clean_latent, torch.Tensor)
            and tuple(clean_latent.shape) == (1, 3, 16, 60, 104),
            "SDVG candidate latent shape is invalid",
        )
        torch.cuda.synchronize(self.base.device)
        decode_started = time.perf_counter()
        with torch.inference_mode():
            video = self.base.pipeline.vae.decode_to_pixel(
                clean_latent.to(
                    device=self.base.device, dtype=torch.bfloat16
                ).contiguous(),
                use_cache=True,
            )
        torch.cuda.synchronize(self.base.device)
        decode_seconds = time.perf_counter() - decode_started
        expected_frames = 9 if chunk_index == 0 else 12
        _require(
            video.ndim == 5
            and tuple(video.shape[:1]) == (1,)
            and int(video.shape[1]) == expected_frames
            and int(video.shape[2]) == 3,
            "SDVG cached VAE emitted an unexpected frame block",
        )
        frames = (
            video[0]
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
        )
        reward_started = time.perf_counter()
        processed = torch.stack(
            [
                self.reward_model.preprocess(self.to_pil_image(frame))
                for frame in frames
            ],
            dim=0,
        ).to(device=self.base.device)
        text = self.reward_model.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.base.device)
        with torch.inference_mode():
            rewards_tensor = self.reward_model.score_gard(
                text.input_ids.repeat(expected_frames, 1),
                text.attention_mask.repeat(expected_frames, 1),
                processed,
            ).detach().reshape(-1)
        torch.cuda.synchronize(self.base.device)
        reward_seconds = time.perf_counter() - reward_started
        rewards = [float(value) for value in rewards_tensor.to("cpu").tolist()]
        _require(
            len(rewards) == expected_frames
            and all(math.isfinite(value) for value in rewards),
            "ImageReward did not return one finite score per frame",
        )
        block_reward = min(rewards)
        self._next_chunk_index += 1
        return {
            "schema": SDVG_REWARD_SCHEMA,
            "chunk_index": chunk_index,
            "rgb_frame_count": expected_frames,
            "frame_rewards": rewards,
            "block_reward": block_reward,
            "threshold": self.threshold,
            "accepted_by_threshold": block_reward >= self.threshold,
            "forced_reject": chunk_index == 0,
            "decode_wall_seconds": decode_seconds,
            "reward_wall_seconds": reward_seconds,
        }

    def finish_sdvg_trajectory(self, *, prompt_id: str) -> Mapping[str, Any]:
        _require(
            self._active_prompt_id == prompt_id,
            "SDVG completion prompt identity drifted",
        )
        _require(
            self._next_chunk_index == self.chunk_count,
            "SDVG trajectory ended before every chunk was scored",
        )
        self.base.pipeline.vae.model.clear_cache()
        self._active_prompt = None
        self._active_prompt_id = None
        count = self._next_chunk_index
        self._next_chunk_index = 0
        return {"prompt_id": prompt_id, "scored_chunk_count": count}

    def close(self) -> None:
        try:
            self.base.pipeline.vae.model.clear_cache()
        finally:
            self._active_prompt = None
            self._active_prompt_id = None
            self.reward_model = None
            self.base.close()


def draft_generation_backend(backend):
    """Return the generation backend behind the known SDVG reward wrapper."""
    return backend.base if type(backend) is SdvgDraftBackend else backend


__all__ = [
    "SDVG_ADAPTER_SCHEMA",
    "SDVG_REWARD_SCHEMA",
    "SDVG_SOURCE_COMMIT",
    "SdvgAdapterError",
    "SdvgDraftBackend",
    "draft_generation_backend",
]
