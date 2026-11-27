"""Fail-closed adapter for the pinned official Light Forcing implementation.

The adapter accepts immutable prompt requests, verifies every artifact before
importing ML dependencies, materializes the official source only below
``CFEXP_RUN_DIR/tmp``, and writes all generated evidence below
``CFEXP_RUN_DIR``.  It refuses to report Light Forcing when the official sparse
attention dispatcher would fall back to dense attention.  Batch execution
loads the official model once per worker and then runs multiple prompts.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import math
from fractions import Fraction
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tarfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence


REQUEST_SCHEMA = "light_forcing_adapter_request_v1"
BATCH_REQUEST_SCHEMA = "light_forcing_adapter_batch_request_v1"
BATCH_RESULT_SCHEMA = "light_forcing_adapter_batch_result_v1"
RESULT_SCHEMA = "light_forcing_adapter_result_v1"
ARTIFACT_SPEC_SCHEMA = "main_table_baseline_artifacts_v1"
RUNTIME_CONFIG_SCHEMA = "light_forcing_adapter_config_v1"
METHOD_ID = "light_forcing"
OFFICIAL_SOURCE_REPOSITORY = "chengtao-lv/LightForcing"
OFFICIAL_SOURCE_COMMIT = "d1e633337fff7e8a0ebf9f643dc37cd481fb299d"

FPS = 24
LATENT_FRAMES = 360
RGB_FRAMES = 1437
HEIGHT = 480
WIDTH = 832
LATENT_HEIGHT = 60
LATENT_WIDTH = 104
SEED = 20260911
NUM_FRAME_PER_BLOCK = 3
DENOISING_STEPS = (1000, 750, 500, 250)
CHECKPOINT_STATE_KEY = "generator_ema"
SPARSE_PARAMETERS = {
    "sparsity": 0.85,
    "sparsity_base": 0.95,
    "keep_frames": 3,
    "keep_sink": 1,
    "keep_near": 1,
}
MODEL_PARAMETERS = {
    "local_attn_size": 12,
    "timestep_shift": 5.0,
    "long_video_gen": True,
    "sink_size": 1,
    "sparse_config": SPARSE_PARAMETERS,
    "efficient_deployment": {
        "lightvae_path": None,
        "quant_fp8": False,
        "rmsnorm_kernel": False,
        "rope_kernel": False,
        "scale_shift_kernel": False,
    },
}

_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SUBDIR_RE = re.compile(rf"^LightForcing-{OFFICIAL_SOURCE_COMMIT}$")
_EXPECTED_RUNTIME_DESTINATIONS = frozenset(
    {
        "wan_models/Wan2.1-T2V-1.3B/config.json",
        "wan_models/Wan2.1-T2V-1.3B/configuration.json",
        "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json",
        "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model",
        "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json",
        "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json",
    }
)


class LightForcingAdapterError(RuntimeError):
    """Raised when the adapter cannot prove the frozen Light Forcing contract."""


@dataclass(frozen=True)
class ValidatedRequest:
    run_dir: Path
    worker_index: int
    prompt_id: str
    prompt: str
    prompt_sha256: str
    seed: int
    video_path: Path
    result_path: Path
    artifacts: dict[str, Path]
    config: dict[str, Any]
    artifact_spec_sha256: str


@dataclass(frozen=True)
class GenerationOutput:
    frames: Any
    total_seconds: float
    prompt_encode_seconds: float
    generation_cuda_seconds: float
    per_chunk_total_seconds: tuple[float, ...]
    decode_seconds: float
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int
    sparse_dispatch: Mapping[str, Any]


class LightForcingBackend(Protocol):
    def generate(self, *, prompt: str, seed: int) -> GenerationOutput: ...

    def frame_shape(self, frames: Any) -> tuple[int, int, int, int]: ...

    def write_video(self, frames: Any, path: Path, fps: int) -> None: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LightForcingAdapterError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                LightForcingAdapterError(
                    f"{label} contains invalid JSON constant {item!r}"
                )
            ),
        )
    except LightForcingAdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LightForcingAdapterError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise LightForcingAdapterError(f"{label} must be a JSON object")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LightForcingAdapterError(f"value is not canonical JSON: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise LightForcingAdapterError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(value) + b"\n")
        os.replace(temporary, path)
    except OSError as error:
        raise LightForcingAdapterError(f"cannot write {path}: {error}") from error


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise LightForcingAdapterError(
            f"{label} keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LightForcingAdapterError(f"{label} must be a non-empty POSIX path")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise LightForcingAdapterError(f"{label} must be a safe relative path")
    return path


def _inside(root: Path, candidate: Path, label: str, *, strict: bool) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=strict)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise LightForcingAdapterError(f"{label} escapes CFEXP_RUN_DIR: {candidate}") from error
    return resolved


def _run_dir_from_environment() -> Path:
    raw = os.environ.get("CFEXP_RUN_DIR")
    if not raw:
        raise LightForcingAdapterError("CFEXP_RUN_DIR is required")
    unresolved = Path(raw).expanduser()
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise LightForcingAdapterError(
            "CFEXP_RUN_DIR must be an absolute non-symlink directory"
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise LightForcingAdapterError(f"CFEXP_RUN_DIR is unavailable: {error}") from error
    if not resolved.is_dir():
        raise LightForcingAdapterError("CFEXP_RUN_DIR must be a directory")
    return resolved


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            members = handle.getmembers()
            if not members:
                raise LightForcingAdapterError("official source archive is empty")
            for member in members:
                relative = Path(member.name)
                if (
                    not member.name
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or member.name in seen
                    or not (member.isdir() or member.isfile())
                ):
                    raise LightForcingAdapterError(
                        f"unsafe or unsupported tar member: {member.name!r}"
                    )
                seen.add(member.name)
            for member in members:
                target = _inside(
                    destination,
                    destination / member.name,
                    "source archive member",
                    strict=False,
                )
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise LightForcingAdapterError(
                        f"cannot read tar member {member.name!r}"
                    )
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except LightForcingAdapterError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise LightForcingAdapterError(
            f"cannot safely extract source archive {archive}: {error}"
        ) from error


def _verify_artifacts(request: Mapping[str, Any]) -> dict[str, Path]:
    values = request.get("verified_artifacts")
    if not isinstance(values, list) or not values:
        raise LightForcingAdapterError(
            "request.verified_artifacts must be a non-empty list"
        )
    result: dict[str, Path] = {}
    for index, value in enumerate(values):
        label = f"request.verified_artifacts[{index}]"
        if not isinstance(value, Mapping):
            raise LightForcingAdapterError(f"{label} must be an object")
        _strict_keys(value, {"name", "resolved_path", "sha256", "size"}, label)
        name = value["name"]
        path_text = value["resolved_path"]
        digest = value["sha256"]
        size = value["size"]
        if not isinstance(name, str) or not name or name in result:
            raise LightForcingAdapterError(f"{label}.name must be unique and non-empty")
        if not isinstance(path_text, str) or not path_text:
            raise LightForcingAdapterError(f"{label}.resolved_path is invalid")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise LightForcingAdapterError(f"{label}.sha256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise LightForcingAdapterError(f"{label}.size is invalid")
        try:
            path = Path(path_text).expanduser().resolve(strict=True)
        except OSError as error:
            raise LightForcingAdapterError(f"{label} is unavailable: {error}") from error
        if path.is_symlink() or not path.is_file():
            raise LightForcingAdapterError(
                f"{label} must resolve to a regular non-symlink file"
            )
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise LightForcingAdapterError(
                f"verified artifact changed after preflight: {name}"
            )
        result[name] = path
    return result


def _artifact(artifacts: Mapping[str, Path], name: Any, label: str) -> Path:
    if not isinstance(name, str) or name not in artifacts:
        raise LightForcingAdapterError(f"{label} names an unverified artifact: {name!r}")
    return artifacts[name]


def _validate_runtime_config(
    config: Mapping[str, Any], artifacts: Mapping[str, Path]
) -> dict[str, Any]:
    _strict_keys(
        config,
        {
            "schema",
            "official_source_commit",
            "source_archive_artifact",
            "source_archive_subdirectory",
            "runtime_files",
            "generation",
        },
        "adapter_config",
    )
    if config.get("schema") != RUNTIME_CONFIG_SCHEMA:
        raise LightForcingAdapterError(
            f"adapter_config.schema must be {RUNTIME_CONFIG_SCHEMA!r}"
        )
    if config.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT:
        raise LightForcingAdapterError(
            f"official source commit must be {OFFICIAL_SOURCE_COMMIT}"
        )
    _artifact(
        artifacts,
        config.get("source_archive_artifact"),
        "adapter_config.source_archive_artifact",
    )
    source_subdirectory = config.get("source_archive_subdirectory")
    if (
        not isinstance(source_subdirectory, str)
        or _SOURCE_SUBDIR_RE.fullmatch(source_subdirectory) is None
    ):
        raise LightForcingAdapterError(
            "source archive subdirectory must bind the exact official commit"
        )
    runtime_files = config.get("runtime_files")
    if not isinstance(runtime_files, list):
        raise LightForcingAdapterError("adapter_config.runtime_files must be a list")
    destinations: set[str] = set()
    for index, value in enumerate(runtime_files):
        label = f"adapter_config.runtime_files[{index}]"
        if not isinstance(value, Mapping):
            raise LightForcingAdapterError(f"{label} must be an object")
        _strict_keys(value, {"artifact", "destination"}, label)
        _artifact(artifacts, value["artifact"], f"{label}.artifact")
        destination = _safe_relative(value["destination"], f"{label}.destination")
        normalized = destination.as_posix()
        if normalized in destinations:
            raise LightForcingAdapterError(
                f"duplicate runtime destination: {normalized}"
            )
        destinations.add(normalized)
    if destinations != _EXPECTED_RUNTIME_DESTINATIONS:
        raise LightForcingAdapterError(
            "runtime files must pin the complete Wan2.1-T2V-1.3B base model"
        )
    generation = config.get("generation")
    if not isinstance(generation, Mapping):
        raise LightForcingAdapterError("adapter_config.generation must be an object")
    _strict_keys(
        generation,
        {
            "checkpoint_artifact",
            "checkpoint_state_key",
            "dtype",
            "model_id",
            "noise_shape",
            "rgb_frames",
            "fps",
            "num_frame_per_block",
            "denoising_step_list",
            "warp_denoising_step",
            "context_noise",
            "model_kwargs",
        },
        "adapter_config.generation",
    )
    _artifact(
        artifacts,
        generation.get("checkpoint_artifact"),
        "adapter_config.generation.checkpoint_artifact",
    )
    noise_shape = generation.get("noise_shape")
    if not isinstance(noise_shape, list) or len(noise_shape) != 4 or type(noise_shape[0]) is not int or noise_shape[0] not in (90, 180, 360):
        raise LightForcingAdapterError("noise_shape must select 90, 180, or 360 latent frames")
    latent_frames = noise_shape[0]
    rgb_frames = latent_frames * 4 - 3
    expected = {
        "checkpoint_state_key": CHECKPOINT_STATE_KEY,
        "dtype": "bfloat16",
        "model_id": "Wan2.1-T2V-1.3B",
        "noise_shape": [latent_frames, 16, LATENT_HEIGHT, LATENT_WIDTH],
        "rgb_frames": rgb_frames,
        "fps": FPS,
        "num_frame_per_block": NUM_FRAME_PER_BLOCK,
        "denoising_step_list": list(DENOISING_STEPS),
        "warp_denoising_step": True,
        "context_noise": 0,
        "model_kwargs": MODEL_PARAMETERS,
    }
    for field, frozen in expected.items():
        if generation.get(field) != frozen:
            raise LightForcingAdapterError(
                f"adapter_config.generation.{field} must equal {frozen!r}"
            )
    return dict(config)


def _load_artifact_config(
    request: Mapping[str, Any], artifacts: Mapping[str, Path]
) -> tuple[dict[str, Any], str]:
    path_text = request.get("artifact_spec_path")
    digest = request.get("artifact_spec_sha256")
    if not isinstance(path_text, str) or not isinstance(digest, str):
        raise LightForcingAdapterError("request artifact-spec identity is invalid")
    if _SHA256_RE.fullmatch(digest) is None:
        raise LightForcingAdapterError("request.artifact_spec_sha256 is invalid")
    try:
        path = Path(path_text).expanduser().resolve(strict=True)
    except OSError as error:
        raise LightForcingAdapterError(f"artifact specification is unavailable: {error}") from error
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
        raise LightForcingAdapterError(
            "artifact specification changed after campaign preflight"
        )
    spec = _load_json(path, "artifact specification")
    _strict_keys(
        spec,
        {"schema", "status", "method_id", "adapter_module", "adapter_config", "artifacts"},
        "artifact specification",
    )
    if (
        spec.get("schema") != ARTIFACT_SPEC_SCHEMA
        or spec.get("status") != "frozen"
        or spec.get("method_id") != METHOD_ID
        or spec.get("adapter_module") != "cfexp.light_forcing_adapter"
    ):
        raise LightForcingAdapterError(
            "artifact specification does not select the frozen Light Forcing adapter"
        )
    config = spec.get("adapter_config")
    if not isinstance(config, Mapping):
        raise LightForcingAdapterError("artifact specification has no adapter_config")
    return _validate_runtime_config(config, artifacts), digest


def validate_request(request: Mapping[str, Any]) -> ValidatedRequest:
    _strict_keys(
        request,
        {
            "schema",
            "method_id",
            "worker_index",
            "prompt",
            "seed",
            "output",
            "artifact_spec_path",
            "artifact_spec_sha256",
            "verified_artifacts",
        },
        "request",
    )
    if request.get("schema") != REQUEST_SCHEMA or request.get("method_id") != METHOD_ID:
        raise LightForcingAdapterError("request does not select Light Forcing")
    worker_index = request.get("worker_index")
    if isinstance(worker_index, bool) or not isinstance(worker_index, int) or worker_index < 0:
        raise LightForcingAdapterError("request.worker_index must be non-negative")
    seed = request.get("seed")
    if seed != SEED:
        raise LightForcingAdapterError(f"Light Forcing main-table seed must be {SEED}")
    prompt = request.get("prompt")
    if not isinstance(prompt, Mapping):
        raise LightForcingAdapterError("request.prompt must be an object")
    _strict_keys(prompt, {"id", "text", "sha256"}, "request.prompt")
    prompt_id = prompt.get("id")
    prompt_text = prompt.get("text")
    prompt_sha256 = prompt.get("sha256")
    if not isinstance(prompt_id, str) or _PROMPT_ID_RE.fullmatch(prompt_id) is None:
        raise LightForcingAdapterError("request.prompt.id is invalid")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise LightForcingAdapterError("request.prompt.text must be non-empty")
    actual_prompt_sha256 = _sha256_bytes(prompt_text.encode("utf-8"))
    if prompt_sha256 != actual_prompt_sha256:
        raise LightForcingAdapterError("request.prompt.sha256 does not bind its text")
    output = request.get("output")
    if not isinstance(output, Mapping):
        raise LightForcingAdapterError("request.output must be an object")
    _strict_keys(output, {"video_path", "result_path"}, "request.output")
    expected_video = Path("videos") / METHOD_ID / f"{prompt_id}.mp4"
    expected_result = Path("provenance") / METHOD_ID / f"{prompt_id}.json"
    if _safe_relative(output.get("video_path"), "request.output.video_path") != expected_video:
        raise LightForcingAdapterError(
            f"video path must be {expected_video.as_posix()}"
        )
    if _safe_relative(output.get("result_path"), "request.output.result_path") != expected_result:
        raise LightForcingAdapterError(
            f"result path must be {expected_result.as_posix()}"
        )
    run_dir = _run_dir_from_environment()
    video_path = _inside(run_dir, run_dir / expected_video, "video path", strict=False)
    result_path = _inside(run_dir, run_dir / expected_result, "result path", strict=False)
    if video_path.exists() or result_path.exists():
        raise LightForcingAdapterError(
            "Light Forcing output already exists; immutable prompt outputs cannot be overwritten"
        )
    artifacts = _verify_artifacts(request)
    config, artifact_spec_sha256 = _load_artifact_config(request, artifacts)
    return ValidatedRequest(
        run_dir=run_dir,
        worker_index=worker_index,
        prompt_id=prompt_id,
        prompt=prompt_text,
        prompt_sha256=actual_prompt_sha256,
        seed=seed,
        video_path=video_path,
        result_path=result_path,
        artifacts=artifacts,
        config=config,
        artifact_spec_sha256=artifact_spec_sha256,
    )


def _materialize_source(validated: ValidatedRequest) -> tuple[Path, Path]:
    scratch = validated.run_dir / "tmp" / (
        f"light-forcing-worker-{validated.worker_index:02d}-{validated.prompt_id}"
    )
    if scratch.exists() or scratch.is_symlink():
        raise LightForcingAdapterError(f"worker scratch path already exists: {scratch}")
    archive = _artifact(
        validated.artifacts,
        validated.config["source_archive_artifact"],
        "source archive",
    )
    _safe_extract_tar(archive, scratch)
    subdirectory = _safe_relative(
        validated.config["source_archive_subdirectory"],
        "source archive subdirectory",
    )
    source_root = _inside(
        scratch, scratch / subdirectory, "official source root", strict=True
    )
    required = (
        source_root / "pipeline" / "causal_inference.py",
        source_root / "wan" / "modules" / "causal_model.py",
        source_root / "wan" / "modules" / "sparse_attention.py",
        source_root / "utils" / "wan_wrapper.py",
        source_root / "configs" / "default_config.yaml",
        source_root / "configs" / "light_forcing_long.yaml",
    )
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise LightForcingAdapterError(
            "official source archive lacks the required Light Forcing implementation"
        )
    for mapping in validated.config["runtime_files"]:
        source = _artifact(validated.artifacts, mapping["artifact"], "runtime artifact")
        relative = _safe_relative(mapping["destination"], "runtime destination")
        destination = _inside(
            source_root, source_root / relative, "runtime destination", strict=False
        )
        if destination.exists() or destination.is_symlink():
            raise LightForcingAdapterError(
                f"runtime destination already exists: {relative.as_posix()}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if _sha256_file(destination) != _sha256_file(source):
            raise LightForcingAdapterError(
                f"materialized runtime file changed: {relative.as_posix()}"
            )
    return source_root, source_root / "wan_models" / "Wan2.1-T2V-1.3B"


def _validate_sparse_profile(
    sparse_module: Any, pipeline: Any, chunk_sparsities: list[float], *, latent_frames: int = LATENT_FRAMES
) -> dict[str, Any]:
    """Prove the official sparse path is selected rather than dense fallback."""

    device_sm = getattr(sparse_module, "DEVICE_SM", None)
    if device_sm not in (90, 100):
        raise LightForcingAdapterError(
            f"official Hopper sparse path requires SM90/SM100, got {device_sm!r}"
        )
    if not bool(getattr(sparse_module, "FLASH_ATTN_4_AVAILABLE", False)):
        raise LightForcingAdapterError(
            "Flash Attention 4 is unavailable; official code would fall back to dense attention"
        )
    if len(chunk_sparsities) != latent_frames // NUM_FRAME_PER_BLOCK:
        raise LightForcingAdapterError(
            "official chunk-aware sparsity schedule has the wrong length"
        )
    if chunk_sparsities[0] != 0.0 or any(
        not math.isfinite(value) or not 0.0 <= value < 1.0
        for value in chunk_sparsities
    ):
        raise LightForcingAdapterError(
            "official chunk-aware sparsity schedule is invalid"
        )
    model = pipeline.generator.model
    configured = dict(getattr(model, "sparse_config", {}) or {})
    if configured.get("sparsity_list") != chunk_sparsities:
        raise LightForcingAdapterError(
            "official generator did not bind the chunk-aware sparsity schedule"
        )
    for key, value in SPARSE_PARAMETERS.items():
        if configured.get(key) != value:
            raise LightForcingAdapterError(
                f"official generator sparse parameter {key} drifted"
            )
    return {
        "device_sm": device_sm,
        "dispatcher": "flash_attention_4_block_sparse",
        "dense_fallback": False,
        "chunk_count": len(chunk_sparsities),
        "chunk_sparsities": chunk_sparsities,
    }


class OfficialLightForcingBackend:
    """One pinned official Light Forcing pipeline, isolated to one visible GPU."""

    def __init__(
        self,
        *,
        source_root: Path,
        model_root: Path,
        artifacts: Mapping[str, Path],
        config: Mapping[str, Any],
    ) -> None:
        del model_root  # Runtime files are materialized at upstream's relative paths.
        self.latent_frames = config["generation"]["noise_shape"][0]
        source_text = str(source_root)
        if source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)
        self._source_text = source_text
        self._old_cwd = Path.cwd()
        os.chdir(source_root)
        try:
            import torch  # type: ignore
            from omegaconf import OmegaConf  # type: ignore
            from torchvision.io import write_video  # type: ignore
        except ImportError as error:
            self._restore_import_state()
            raise LightForcingAdapterError(
                "Light Forcing runtime requires torch, OmegaConf, and torchvision"
            ) from error
        self.torch = torch
        self.write_video_impl = write_video
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            self._restore_import_state()
            raise LightForcingAdapterError(
                "each Light Forcing adapter worker must see exactly one CUDA GPU"
            )
        self.device = torch.device("cuda:0")
        try:
            sparse_module = importlib.import_module("wan.modules.sparse_attention")
            pipeline_module = importlib.import_module("pipeline.causal_inference")
            pipeline_type = getattr(pipeline_module, "CausalInferencePipeline")
            generation = config["generation"]
            official_config = OmegaConf.load(
                str(source_root / "configs" / "default_config.yaml")
            )
            adapted = OmegaConf.create(
                {
                    "denoising_step_list": list(DENOISING_STEPS),
                    "warp_denoising_step": True,
                    "num_frame_per_block": NUM_FRAME_PER_BLOCK,
                    "model_name": "Wan2.1-T2V-1.3B",
                    "model_kwargs": dict(MODEL_PARAMETERS),
                    "num_output_frames": self.latent_frames,
                    "use_ema": True,
                    "seed": SEED,
                    "num_samples": 1,
                    "context_noise": 0,
                    "height": HEIGHT,
                    "width": WIDTH,
                    "generator_ckpt": "verified-by-cfexp",
                }
            )
            merged = OmegaConf.merge(official_config, adapted)
            calculate = getattr(sparse_module, "calculate_chunk_sparsities")
            sparse_config = dict(merged.model_kwargs.sparse_config)
            chunk_sparsities = [
                float(value)
                for value in calculate(
                    self.latent_frames,
                    NUM_FRAME_PER_BLOCK,
                    MODEL_PARAMETERS["local_attn_size"],
                    sparse_config,
                )
            ]
            sparse_config["sparsity_list"] = chunk_sparsities
            merged.model_kwargs.sparse_config = sparse_config
            pipeline = pipeline_type(merged, device=self.device)
            checkpoint_path = _artifact(
                artifacts, generation["checkpoint_artifact"], "generator checkpoint"
            )
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            state_key = generation["checkpoint_state_key"]
            if not isinstance(checkpoint, Mapping) or state_key not in checkpoint:
                raise LightForcingAdapterError(
                    f"checkpoint does not contain configured state key {state_key!r}"
                )
            state = checkpoint[state_key]
            if not isinstance(state, Mapping):
                raise LightForcingAdapterError("checkpoint generator state is not a mapping")
            pipeline.generator.load_state_dict(state, strict=True)
            del checkpoint, state
            pipeline = pipeline.to(dtype=torch.bfloat16)
            pipeline.text_encoder.to(device=self.device)
            pipeline.generator.to(device=self.device)
            pipeline.vae.to(device=self.device)
            pipeline.eval().requires_grad_(False)
            if tuple(float(item) for item in pipeline.denoising_step_list.tolist()) != (
                1000.0,
                937.5,
                833.3333129882812,
                625.0,
            ):
                raise LightForcingAdapterError(
                    "official warped denoising schedule drifted"
                )
            self.sparse_dispatch = _validate_sparse_profile(
                sparse_module, pipeline, chunk_sparsities, latent_frames=self.latent_frames
            )
            self.pipeline = pipeline
        except BaseException:
            self._restore_import_state()
            raise

    def _restore_import_state(self) -> None:
        os.chdir(self._old_cwd)
        if self._source_text in sys.path:
            sys.path.remove(self._source_text)

    def generate(self, *, prompt: str, seed: int) -> GenerationOutput:
        torch = self.torch
        if seed != SEED or not prompt.strip():
            raise LightForcingAdapterError("prompt/seed drifted after request validation")
        random.seed(seed)
        try:
            import numpy as np  # type: ignore
        except ImportError as error:
            raise LightForcingAdapterError("Light Forcing runtime requires NumPy") from error
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        noise = torch.randn(
            (1, self.latent_frames, 16, LATENT_HEIGHT, LATENT_WIDTH),
            device=self.device,
            dtype=torch.bfloat16,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        overall_cuda_start = torch.cuda.Event(enable_timing=True)
        overall_cuda_end = torch.cuda.Event(enable_timing=True)
        prompt_cuda_start = torch.cuda.Event(enable_timing=True)
        prompt_cuda_end = torch.cuda.Event(enable_timing=True)
        decode_cuda_start = torch.cuda.Event(enable_timing=True)
        decode_cuda_end = torch.cuda.Event(enable_timing=True)
        chunk_events: dict[int, tuple[Any, Any]] = {}
        chunk_call_counts: dict[int, int] = {}
        prompt_encode_seconds = 0.0
        decode_seconds = 0.0
        real_text_forward = self.pipeline.text_encoder.forward
        real_generator_forward = self.pipeline.generator.forward
        real_decode = self.pipeline.vae.decode_to_pixel

        def timed_text_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal prompt_encode_seconds
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            prompt_cuda_start.record()
            output = real_text_forward(*args, **kwargs)
            prompt_cuda_end.record()
            torch.cuda.synchronize(self.device)
            prompt_encode_seconds += time.perf_counter() - started
            return output

        def timed_generator_forward(*args: Any, **kwargs: Any) -> Any:
            current_start = kwargs.get("current_start")
            if isinstance(current_start, bool) or not isinstance(current_start, int):
                raise LightForcingAdapterError(
                    "official generator call omitted integer current_start"
                )
            if current_start not in chunk_events:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                chunk_events[current_start] = (start_event, end_event)
                chunk_call_counts[current_start] = 0
            output = real_generator_forward(*args, **kwargs)
            chunk_events[current_start][1].record()
            chunk_call_counts[current_start] += 1
            return output

        def timed_decode(*args: Any, **kwargs: Any) -> Any:
            nonlocal decode_seconds
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            decode_cuda_start.record()
            output = real_decode(*args, **kwargs)
            decode_cuda_end.record()
            torch.cuda.synchronize(self.device)
            decode_seconds += time.perf_counter() - started
            return output

        self.pipeline.text_encoder.forward = timed_text_forward
        self.pipeline.generator.forward = timed_generator_forward
        self.pipeline.vae.decode_to_pixel = timed_decode
        try:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            overall_cuda_start.record()
            with torch.inference_mode():
                video = self.pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    return_latents=False,
                    profile=False,
                    low_memory=False,
                )
            overall_cuda_end.record()
            torch.cuda.synchronize(self.device)
            total_seconds = time.perf_counter() - started
        finally:
            self.pipeline.text_encoder.forward = real_text_forward
            self.pipeline.generator.forward = real_generator_forward
            self.pipeline.vae.decode_to_pixel = real_decode
        overall_cuda_seconds = overall_cuda_start.elapsed_time(overall_cuda_end) / 1000.0
        prompt_encode_cuda_seconds = (
            prompt_cuda_start.elapsed_time(prompt_cuda_end) / 1000.0
        )
        decode_cuda_seconds = decode_cuda_start.elapsed_time(decode_cuda_end) / 1000.0
        generation_cuda_seconds = (
            overall_cuda_seconds - prompt_encode_cuda_seconds - decode_cuda_seconds
        )
        if not math.isfinite(generation_cuda_seconds) or generation_cuda_seconds < 0:
            raise LightForcingAdapterError("CUDA event timing produced an invalid generation interval")
        if len(chunk_events) != self.latent_frames // NUM_FRAME_PER_BLOCK or any(
            count != len(DENOISING_STEPS) + 1 for count in chunk_call_counts.values()
        ):
            raise LightForcingAdapterError(
                "official generator call trace does not cover the requested four-step chunks plus KV updates"
            )
        per_chunk_total_seconds = tuple(
            start.elapsed_time(end) / 1000.0
            for start, end in chunk_events.values()
        )
        if any(not math.isfinite(value) or value < 0 for value in per_chunk_total_seconds):
            raise LightForcingAdapterError("per-chunk CUDA timing is invalid")
        return GenerationOutput(
            frames=video,
            total_seconds=total_seconds,
            prompt_encode_seconds=prompt_encode_seconds,
            generation_cuda_seconds=generation_cuda_seconds,
            per_chunk_total_seconds=per_chunk_total_seconds,
            decode_seconds=decode_seconds,
            peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)),
            sparse_dispatch=dict(self.sparse_dispatch),
        )

    def frame_shape(self, frames: Any) -> tuple[int, int, int, int]:
        if not isinstance(frames, self.torch.Tensor) or frames.ndim != 5:
            raise LightForcingAdapterError(
                "official pipeline video must be [1,frames,channels,height,width]"
            )
        if int(frames.shape[0]) != 1:
            raise LightForcingAdapterError("Light Forcing batch size must be one")
        return tuple(int(item) for item in frames.shape[1:])

    def write_video(self, frames: Any, path: Path, fps: int) -> None:
        if fps != FPS:
            raise LightForcingAdapterError(f"Light Forcing output FPS must be {FPS}")
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = (
            frames[0]
            .permute(0, 2, 3, 1)
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(device="cpu", dtype=self.torch.uint8)
            .contiguous()
        )
        self.write_video_impl(str(path), pixels, fps=fps, video_codec="h264")

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def close(self) -> None:
        try:
            if hasattr(self, "pipeline"):
                del self.pipeline
            gc.collect()
            if hasattr(self, "torch") and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        finally:
            self._restore_import_state()


def _execute_validated(
    validated: ValidatedRequest,
    backend: LightForcingBackend,
    *,
    model_load_seconds: float,
) -> dict[str, Any]:
    latent_frames = validated.config["generation"]["noise_shape"][0]
    rgb_frames = latent_frames * 4 - 3
    output = backend.generate(prompt=validated.prompt, seed=validated.seed)
    shape = backend.frame_shape(output.frames)
    if shape != (rgb_frames, 3, HEIGHT, WIDTH):
        raise LightForcingAdapterError(
            f"decoded video shape {shape} != {(rgb_frames, 3, HEIGHT, WIDTH)}"
        )
    if (
        not math.isfinite(output.total_seconds)
        or not math.isfinite(output.prompt_encode_seconds)
        or not math.isfinite(output.generation_cuda_seconds)
        or not math.isfinite(output.decode_seconds)
        or output.total_seconds <= 0
        or output.prompt_encode_seconds < 0
        or output.generation_cuda_seconds < 0
        or output.decode_seconds < 0
        or output.prompt_encode_seconds + output.decode_seconds > output.total_seconds
        or len(output.per_chunk_total_seconds) != latent_frames // NUM_FRAME_PER_BLOCK
        or any(
            not math.isfinite(value) or value < 0
            for value in output.per_chunk_total_seconds
        )
    ):
        raise LightForcingAdapterError("backend returned invalid timing evidence")
    dispatch = dict(output.sparse_dispatch)
    if dispatch.get("dense_fallback") is not False:
        raise LightForcingAdapterError(
            "backend did not prove official sparse attention execution"
        )
    video_encode_started = time.perf_counter()
    backend.write_video(output.frames, validated.video_path, FPS)
    video_encode_seconds = time.perf_counter() - video_encode_started
    if (
        not validated.video_path.is_file()
        or validated.video_path.is_symlink()
        or validated.video_path.stat().st_size < 1
    ):
        raise LightForcingAdapterError("backend did not produce a regular MP4 file")
    generation_seconds = (
        output.total_seconds - output.prompt_encode_seconds - output.decode_seconds
    )
    result = {
        "schema": RESULT_SCHEMA,
        "method_id": METHOD_ID,
        "implementation": {
            "kind": "official_code_adapted_to_main_table_contract",
            "source_repository": OFFICIAL_SOURCE_REPOSITORY,
            "source_commit": OFFICIAL_SOURCE_COMMIT,
            "artifact_spec_sha256": validated.artifact_spec_sha256,
        },
        "prompt": {
            "id": validated.prompt_id,
            "sha256": validated.prompt_sha256,
        },
        "seed": validated.seed,
        "temporal_contract": {
            "latent_frames": latent_frames,
            "rgb_frames": rgb_frames,
            "fps": FPS,
            "exact_duration_seconds": str(Fraction(rgb_frames, FPS)),
            "height": HEIGHT,
            "width": WIDTH,
        },
        "method_contract": {
            "backbone": "Wan2.1-T2V-1.3B",
            "denoising_step_list": list(DENOISING_STEPS),
            "num_frame_per_block": NUM_FRAME_PER_BLOCK,
            "model_kwargs": dict(MODEL_PARAMETERS),
            "hardware_profile": "official_hopper_recommendation",
            "sparse_dispatch": dispatch,
        },
        "timing": {
            "scope": "prompt_encode_plus_generation_plus_vae_decode_excluding_model_load_and_mp4_encode",
            "model_load_seconds": model_load_seconds,
            "total_seconds": output.total_seconds,
            "prompt_encode_seconds": output.prompt_encode_seconds,
            "generation_wall_seconds": generation_seconds,
            "generation_cuda_seconds": output.generation_cuda_seconds,
            "per_chunk_total_seconds": list(output.per_chunk_total_seconds),
            "vae_decode_seconds": output.decode_seconds,
            "video_encode_seconds": video_encode_seconds,
        },
        "memory": {
            "peak_allocated_bytes": output.peak_memory_allocated_bytes,
            "peak_reserved_bytes": output.peak_memory_reserved_bytes,
        },
        "video": {
            "path": validated.video_path.relative_to(validated.run_dir).as_posix(),
            "sha256": _sha256_file(validated.video_path),
            "nbytes": validated.video_path.stat().st_size,
            "container": "mp4",
            "fps": FPS,
            "frames": rgb_frames,
            "height": HEIGHT,
            "width": WIDTH,
        },
    }
    _write_json(validated.result_path, result)
    return result


def _validate_batch_identity(validated: Sequence[ValidatedRequest]) -> None:
    if not validated:
        raise LightForcingAdapterError("batch must contain at least one request")
    first = validated[0]
    prompt_ids: set[str] = set()
    output_paths: set[Path] = set()
    for item in validated:
        if item.run_dir != first.run_dir or item.worker_index != first.worker_index:
            raise LightForcingAdapterError(
                "all batch requests must use the same run directory and worker index"
            )
        if (
            item.artifact_spec_sha256 != first.artifact_spec_sha256
            or item.artifacts != first.artifacts
            or item.config != first.config
        ):
            raise LightForcingAdapterError(
                "all batch requests must bind identical verified artifacts and config"
            )
        if item.prompt_id in prompt_ids:
            raise LightForcingAdapterError("batch prompt IDs must be unique")
        prompt_ids.add(item.prompt_id)
        for path in (item.video_path, item.result_path):
            if path in output_paths:
                raise LightForcingAdapterError("batch output paths must be unique")
            output_paths.add(path)


def execute_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    backend_factory: Callable[..., LightForcingBackend] = OfficialLightForcingBackend,
) -> list[dict[str, Any]]:
    """Run a prompt shard with one verified source tree and one model load."""

    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise LightForcingAdapterError("requests must be a sequence")
    if any(not isinstance(request, Mapping) for request in requests):
        raise LightForcingAdapterError("every batch request must be an object")
    validated = [validate_request(request) for request in requests]
    _validate_batch_identity(validated)
    first = validated[0]
    source_root, model_root = _materialize_source(first)
    model_load_started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        backend = backend_factory(
            source_root=source_root,
            model_root=model_root,
            artifacts=first.artifacts,
            config=first.config,
        )
    try:
        backend.synchronize()
        model_load_seconds = time.perf_counter() - model_load_started
        with redirect_stdout(sys.stderr):
            return [
                _execute_validated(
                    item, backend, model_load_seconds=model_load_seconds
                )
                for item in validated
            ]
    finally:
        backend.close()


def execute_request(
    request: Mapping[str, Any],
    *,
    backend_factory: Callable[..., LightForcingBackend] = OfficialLightForcingBackend,
) -> dict[str, Any]:
    return execute_requests([request], backend_factory=backend_factory)[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.light_forcing_adapter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", type=Path)
    group.add_argument("--batch-request", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.request is not None:
            request = _load_json(args.request.expanduser().resolve(strict=True), "request")
            result: Any = execute_request(request)
        else:
            batch = _load_json(
                args.batch_request.expanduser().resolve(strict=True), "batch request"
            )
            _strict_keys(batch, {"schema", "requests"}, "batch request")
            if batch.get("schema") != BATCH_REQUEST_SCHEMA:
                raise LightForcingAdapterError(
                    f"batch request schema must be {BATCH_REQUEST_SCHEMA!r}"
                )
            requests = batch.get("requests")
            if not isinstance(requests, list):
                raise LightForcingAdapterError("batch request requests must be a list")
            result = {
                "schema": BATCH_RESULT_SCHEMA,
                "results": execute_requests(requests),
            }
    except (LightForcingAdapterError, OSError) as error:
        print(f"light-forcing-adapter: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
