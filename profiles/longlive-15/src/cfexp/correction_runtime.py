"""Production runtime bridge for the Insight 2 correction campaign.

This module is deliberately the only layer that knows both the model-agnostic
campaign contracts and the persistent JSONL worker protocol.  The campaign
passes immutable latent histories; a worker reconstructs its own model-local
cache from those latents and returns a cache witness.  No cache object crosses
the process boundary.

The two backend factories are also kept here so worker startup has one small,
fail-closed path: bind preflight records back to a digest-pinned artifact
specification, re-hash every artifact, materialize source below
``CFEXP_RUN_DIR``, and only then construct the heavy backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, TextIO
import uuid

from .stage_timing import stage, timing_session

from .correction_state import (
    GenerationContext,
    InitialNoiseDescriptor,
    KVCacheDescriptor,
    LatentHistory,
    LatentSegment,
    derive_initial_noise_seed,
)
from .correction_worker import (
    BOUNDARY_VELOCITY_ACCUMULATOR_DTYPE,
    BOUNDARY_VELOCITY_METRIC_ID,
    BOUNDARY_VELOCITY_ROUND_DECIMALS,
    BOUNDARY_VELOCITY_WITNESS_SCHEMA,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
)
from .insight2_campaign import (
    ChunkGenerationRequest,
    GeneratedChunk,
    NoiseMaterialization,
    NOISE_NAMESPACE,
    PreparedFullQuality,
    QualityMeasurement,
)
from .vbench_long import (
    OfficialVBenchLongScorer,
    VBENCH_LONG_DIMENSIONS,
    VBenchLongMeasurement,
    VBenchLongVideoRequest,
)


BACKEND_CONFIG_SCHEMA = "insight2_runtime_backend_config_v1"
DECODE_EVIDENCE_SCHEMA = "insight2_decode_evidence_v1"
QUALITY_MEASUREMENT_SCHEMA = "insight2_quality_measurement_v1"
PRIMARY_QUALITY_METRIC = "musiq_spaq_frame_mean_raw_div_100"
TEMPORAL_CONSISTENCY_METRIC = "adjacent_frame_l1_similarity_v1"
VBENCH_LONG_PRIMARY_QUALITY_METRIC = (
    "vbench_long_custom_six_dimension_macro_mean"
)
VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA = (
    "insight2_vbench_long_full_quality_v1"
)
MODEL_IDS = {
    "draft": "self_forcing_1.3b",
    "target": "krea-realtime-video-14b",
}
NOISE_SHAPE = (1, 3, 16, 60, 104)
SUPPORTED_CHUNK_COUNTS = frozenset({16, 30, 60, 120})
DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS = 3600.0
DEFAULT_WORKER_REQUEST_TIMEOUT_SECONDS = 1800.0
WORKER_WATCHDOG_POLL_SECONDS = 0.1
WORKER_LOG_TAIL_BYTES = 16 * 1024
WORKER_LOG_TAIL_LINES = 20

_VBENCH_RUNTIME_CACHE_LAYOUT = {
    "MPLCONFIGDIR": "matplotlib",
    "XDG_CACHE_HOME": "xdg",
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    "HUGGING_FACE_HUB_CACHE": "huggingface/hub",
    "HF_DATASETS_CACHE": "huggingface/datasets",
    "HF_ASSETS_CACHE": "huggingface/assets",
    "HUGGINGFACE_ASSETS_CACHE": "huggingface/assets",
    "HF_MODULES_CACHE": "huggingface/modules",
    "TRANSFORMERS_CACHE": "transformers",
    "DIFFUSERS_CACHE": "diffusers",
    "TORCH_HOME": "torch",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MEASUREMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_BOUNDARY_VELOCITY_WITNESS_KEYS = frozenset(
    {
        "schema",
        "metric_id",
        "accumulator_dtype",
        "round_decimals",
        "input_dtype",
        "history_frame_count",
        "candidate_frame_count",
        "previous_velocity_norm",
        "boundary_velocity_norm",
        "candidate_velocity_norm",
        "incoming_discontinuity",
        "outgoing_discontinuity",
        "score",
    }
)


class RuntimeBridgeError(RuntimeError):
    """A backend factory, worker exchange, or immutable sidecar is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeBridgeError(message)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
            size = handle.tell()
    except OSError as error:
        raise RuntimeBridgeError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest(), size


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeBridgeError(f"value is not canonical JSON: {error}") from error


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise RuntimeBridgeError(f"cannot verify {path}: {error}") from error
        _require(existing == raw, f"immutable JSON conflicts with existing file: {path}")
        return
    except OSError as error:
        raise RuntimeBridgeError(f"cannot create {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _strict_json_file(path: Path) -> dict[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RuntimeBridgeError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RuntimeBridgeError(f"invalid JSON constant {item!r} in {path}")
            ),
        )
    except RuntimeBridgeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeBridgeError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def _relative_path(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    _require("\\" not in value and "\x00" not in value, f"{label} is not canonical")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    _require(
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and str(posix) == value
        and all(part not in {"", ".", ".."} for part in posix.parts),
        f"{label} must be a canonical relative POSIX path",
    )
    return value


def _run_file(run_dir: Path, relative: str, *, must_exist: bool) -> Path:
    relative = _relative_path(relative, label="run-relative path")
    root = run_dir.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise RuntimeBridgeError(f"path escapes CFEXP_RUN_DIR: {relative}") from error
    if must_exist:
        _require(resolved.is_file() and not candidate.is_symlink(), f"missing file: {relative}")
    return resolved


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )


def _boundary_velocity_number(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"boundary velocity witness {label} must be finite",
    )
    parsed = float(value)
    _require(
        parsed >= minimum and (maximum is None or parsed <= maximum),
        f"boundary velocity witness {label} is out of range",
    )
    _require(
        parsed == round(parsed, BOUNDARY_VELOCITY_ROUND_DECIMALS),
        f"boundary velocity witness {label} is not rounded to "
        f"{BOUNDARY_VELOCITY_ROUND_DECIMALS} decimals",
    )
    return parsed


def _validated_boundary_velocity_witness(
    result: Mapping[str, Any],
    *,
    chunk_index: int,
    history_chunk_count: int,
    input_dtype: str,
    candidate_frame_count: int,
) -> dict[str, Any] | None:
    """Validate worker-owned boundary evidence before campaign state sees it."""

    _require(
        "boundary_velocity_witness" in result,
        "worker generate result omitted boundary velocity witness",
    )
    value = result["boundary_velocity_witness"]
    _require(
        (chunk_index == 0) == (history_chunk_count == 0),
        "generate history and chunk index disagree at the boundary",
    )
    no_history = chunk_index == 0
    if no_history:
        _require(
            value is None,
            "boundary velocity witness must be null when generation has no history",
        )
        return None
    _require(
        isinstance(value, Mapping) and set(value) == _BOUNDARY_VELOCITY_WITNESS_KEYS,
        "boundary velocity witness fields are invalid",
    )
    assert isinstance(value, Mapping)
    expected_history_frames = history_chunk_count * candidate_frame_count
    _require(
        value.get("schema") == BOUNDARY_VELOCITY_WITNESS_SCHEMA
        and value.get("metric_id") == BOUNDARY_VELOCITY_METRIC_ID
        and value.get("accumulator_dtype")
        == BOUNDARY_VELOCITY_ACCUMULATOR_DTYPE
        and isinstance(value.get("round_decimals"), int)
        and not isinstance(value.get("round_decimals"), bool)
        and value.get("round_decimals") == BOUNDARY_VELOCITY_ROUND_DECIMALS
        and value.get("input_dtype") == input_dtype
        and isinstance(value.get("history_frame_count"), int)
        and not isinstance(value.get("history_frame_count"), bool)
        and value.get("history_frame_count") == expected_history_frames
        and isinstance(value.get("candidate_frame_count"), int)
        and not isinstance(value.get("candidate_frame_count"), bool)
        and value.get("candidate_frame_count") == candidate_frame_count,
        "boundary velocity witness contract or frame binding is invalid",
    )
    for field in (
        "previous_velocity_norm",
        "boundary_velocity_norm",
        "candidate_velocity_norm",
    ):
        _boundary_velocity_number(value.get(field), label=field, minimum=0.0)
    incoming = _boundary_velocity_number(
        value.get("incoming_discontinuity"),
        label="incoming_discontinuity",
        minimum=0.0,
        maximum=1.0,
    )
    outgoing = _boundary_velocity_number(
        value.get("outgoing_discontinuity"),
        label="outgoing_discontinuity",
        minimum=0.0,
        maximum=1.0,
    )
    score = _boundary_velocity_number(
        value.get("score"), label="score", minimum=0.0, maximum=1.0
    )
    _require(
        score == max(incoming, outgoing),
        "boundary velocity witness score is inconsistent",
    )
    return dict(value)


def _factory_config(raw: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "backend config must be an object")
    expected = {
        "schema",
        "role",
        "run_dir",
        "worker_label",
        "device",
        "artifact_spec_path",
        "artifact_spec_sha256",
        "verified_artifacts",
        "_verified_config_sha256",
    }
    actual = set(raw)
    optional = {"chunk_count", "conditioning_cache", "sdvg", "rolling"}
    _require(
        expected <= actual <= expected | optional,
        "backend config keys differ from the frozen runtime contract",
    )
    _require(raw["schema"] == BACKEND_CONFIG_SCHEMA, "backend config schema is invalid")
    _require(raw["role"] == role, f"backend role must be {role!r}")
    _require(raw["device"] == "cuda:0", "isolated workers must use device='cuda:0'")
    label = raw["worker_label"]
    _require(isinstance(label, str) and _LABEL_RE.fullmatch(label) is not None,
             "worker_label is invalid")
    for field in ("artifact_spec_sha256", "_verified_config_sha256"):
        _require(
            isinstance(raw[field], str) and _SHA256_RE.fullmatch(raw[field]) is not None,
            f"{field} must be a lowercase SHA-256",
        )
    run_dir = Path(str(raw["run_dir"])).expanduser().resolve(strict=True)
    _require(run_dir.is_dir(), "backend run_dir must be an existing directory")
    environment_run = os.environ.get("CFEXP_RUN_DIR")
    _require(environment_run is not None, "CFEXP_RUN_DIR is required by backend factory")
    _require(
        Path(environment_run).expanduser().resolve(strict=True) == run_dir,
        "backend run_dir disagrees with CFEXP_RUN_DIR",
    )
    spec_path = Path(str(raw["artifact_spec_path"])).expanduser()
    _require(not spec_path.is_symlink(), "artifact specification may not be a symlink")
    spec_path = spec_path.resolve(strict=True)
    _require(spec_path.is_file(), "artifact specification must be a regular file")
    _require(
        _sha256_file(spec_path)[0] == raw["artifact_spec_sha256"],
        "artifact specification changed after preflight",
    )
    _require(isinstance(raw["verified_artifacts"], list), "verified_artifacts must be a list")
    result = dict(raw)
    chunk_count = result.get("chunk_count", 16)
    _require(
        isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count in SUPPORTED_CHUNK_COUNTS,
        "backend chunk_count must be one of 16, 30, 60, or 120",
    )
    result["chunk_count"] = chunk_count
    result["run_dir"] = run_dir
    result["artifact_spec_path"] = spec_path
    if "rolling" in result:
        from .rolling_forcing_assets import SPEC_SHA256 as rolling_spec_sha256
        _require(role == "draft" and raw["artifact_spec_sha256"] == rolling_spec_sha256
                 and isinstance(result["rolling"], dict),
                 "Rolling configuration requires its pinned Draft artifact specification")
    if "sdvg" in result:
        sdvg = result["sdvg"]
        _require(isinstance(sdvg, Mapping), "backend SDVG config must be an object")
        _exact_keys(
            sdvg,
            {
                "artifact_spec_path",
                "artifact_spec_sha256",
                "adapter_config",
                "verified_artifacts",
            },
            label="backend SDVG config",
        )
        sdvg_path = Path(str(sdvg["artifact_spec_path"])).expanduser()
        _require(not sdvg_path.is_symlink(), "SDVG artifact specification may not be a symlink")
        sdvg_path = sdvg_path.resolve(strict=True)
        _require(sdvg_path.is_file(), "SDVG artifact specification must be a regular file")
        sdvg_sha = sdvg["artifact_spec_sha256"]
        _require(
            isinstance(sdvg_sha, str)
            and _SHA256_RE.fullmatch(sdvg_sha) is not None
            and _sha256_file(sdvg_path)[0] == sdvg_sha,
            "SDVG artifact specification digest mismatch",
        )
        _require(isinstance(sdvg["adapter_config"], Mapping), "SDVG adapter config is invalid")
        _require(
            isinstance(sdvg["verified_artifacts"], list)
            and bool(sdvg["verified_artifacts"]),
            "SDVG verified artifacts are absent",
        )
        result["sdvg"] = {
            **dict(sdvg),
            "artifact_spec_path": sdvg_path,
        }
    return result


def _bind_records_to_spec(
    spec: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    artifacts = spec.get("artifacts")
    _require(isinstance(artifacts, list) and bool(artifacts), "artifact spec has no artifacts")
    expected: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        _require(isinstance(item, Mapping), "artifact spec contains a non-object record")
        name = item.get("name")
        _require(isinstance(name, str) and name not in expected, "artifact name is invalid")
        expected[name] = item
    _require(len(records) == len(expected), "verified artifact count differs from spec")
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    record_keys = {
        "name",
        "role",
        "root",
        "requested_path",
        "resolved_path",
        "size",
        "sha256",
    }
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"verified_artifacts[{index}] is not an object")
        _exact_keys(record, record_keys, label=f"verified_artifacts[{index}]")
        name = record["name"]
        _require(isinstance(name, str) and name in expected and name not in seen,
                 f"verified_artifacts[{index}].name is invalid")
        frozen = expected[name]
        for record_field, spec_field in (
            ("role", "role"),
            ("root", "root"),
            ("requested_path", "path"),
            ("sha256", "sha256"),
        ):
            _require(
                record[record_field] == frozen.get(spec_field),
                f"verified artifact {name!r} is not bound to its frozen {record_field}",
            )
        digest = record["sha256"]
        _require(isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
                 f"verified artifact {name!r} has an invalid digest")
        hashes[name] = digest
        seen.add(name)
    _require(seen == set(expected), "verified artifact names differ from spec")
    return hashes


_CONDITIONING_TOKENIZER_ARTIFACTS = (
    "umt5_special_tokens_map",
    "umt5_spiece_model",
    "umt5_tokenizer_config",
    "umt5_tokenizer_json",
)


def _shared_conditioning_configs(
    *,
    run_dir: Path,
    prompts: Sequence[Mapping[str, Any]],
    draft_spec: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    draft_verified_artifacts: Sequence[Mapping[str, Any]],
    target_verified_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one prompt cache to the exact shared UMT5 implementation inputs."""

    from .conditioning_cache import (
        CONDITIONING_CONFIG_SCHEMA,
        CONDITIONING_PROVENANCE_SCHEMA,
        canonical_prompt_records,
        prompt_set_sha256,
    )

    draft_runtime = draft_spec.get("adapter_config")
    if draft_spec.get("schema") == "rolling_native_artifacts_v1":
        from .rolling_forcing_native import SOURCE_COMMIT
        _require(draft_spec.get("source_commit") == SOURCE_COMMIT,
                 "Rolling conditioning source revision mismatch")
        draft_runtime = {"source_archive_artifact": "rolling_source",
                         "official_source_commit": SOURCE_COMMIT}
    target_runtime = target_spec.get("backend_config")
    _require(isinstance(draft_runtime, Mapping), "Draft adapter config is unavailable")
    _require(isinstance(target_runtime, Mapping), "Target backend config is unavailable")
    draft_hashes = {
        str(item.get("name")): item.get("sha256") for item in draft_verified_artifacts
    }
    target_hashes = {
        str(item.get("name")): item.get("sha256") for item in target_verified_artifacts
    }
    shared_names = ("wan_t5_checkpoint", *_CONDITIONING_TOKENIZER_ARTIFACTS)
    for name in shared_names:
        _require(
            isinstance(draft_hashes.get(name), str)
            and draft_hashes.get(name) == target_hashes.get(name)
            and _SHA256_RE.fullmatch(str(draft_hashes[name])) is not None,
            f"Draft and Target conditioning artifact {name!r} is not byte-identical",
        )
    draft_source_name = draft_runtime.get("source_archive_artifact")
    target_source_name = target_runtime.get("source_archive_artifact")
    _require(
        isinstance(draft_source_name, str)
        and isinstance(target_source_name, str)
        and _SHA256_RE.fullmatch(str(draft_hashes.get(draft_source_name, ""))) is not None
        and _SHA256_RE.fullmatch(str(target_hashes.get(target_source_name, ""))) is not None,
        "conditioning source archive bindings are unavailable",
    )
    prompt_values = [
        {"prompt_id": item.get("prompt_id"), "prompt": item.get("prompt")}
        for item in prompts
    ]
    records = canonical_prompt_records(prompt_values)
    provenance = {
        "schema": CONDITIONING_PROVENANCE_SCHEMA,
        "dtype": "bfloat16",
        "encoder_implementation": "wan.modules.t5.umt5_xxl",
        "draft_source_commit": draft_runtime.get("official_source_commit"),
        "target_source_commit": target_runtime.get("official_source_commit"),
        "draft_source_archive_sha256": draft_hashes[draft_source_name],
        "target_source_archive_sha256": target_hashes[target_source_name],
        "text_encoder_checkpoint_sha256": draft_hashes["wan_t5_checkpoint"],
        "tokenizer_artifacts": [
            {"name": name, "sha256": draft_hashes[name]}
            for name in sorted(_CONDITIONING_TOKENIZER_ARTIFACTS)
        ],
    }
    common = {
        "schema": CONDITIONING_CONFIG_SCHEMA,
        "run_dir": str(run_dir),
        "manifest_path": "provenance/shared-umt5-conditioning/manifest.json",
        "tensor_path": "provenance/shared-umt5-conditioning/prompt-embeds.safetensors",
        "prompt_set_sha256": prompt_set_sha256(records),
        "prompts": prompt_values,
        "provenance": provenance,
    }
    return ({**common, "mode": "producer"}, {**common, "mode": "consumer"})


def build_draft_backend(raw_config: Mapping[str, Any]) -> Any:
    """Worker factory for the pinned Self-Forcing 1.3B Draft."""

    config = _factory_config(raw_config, role="draft")
    # Private helpers are reused intentionally: this is the same frozen loader
    # boundary used by the existing eRank campaign, not a second implementation.
    from .self_forcing_frame_adapter import (
        OFFICIAL_SOURCE_COMMIT,
        OfficialSelfForcingBackend,
        _load_json,
        _load_runtime_config,
        _materialize_source,
        _verified_artifacts,
    )

    spec_path = config["artifact_spec_path"]
    spec = _load_json(spec_path, "Draft artifact specification")
    artifact_hashes = _bind_records_to_spec(spec, config["verified_artifacts"])
    adapter_request = {
        "artifact_spec_path": str(spec_path),
        "artifact_spec_sha256": config["artifact_spec_sha256"],
        "verified_artifacts": config["verified_artifacts"],
    }
    artifacts = _verified_artifacts(adapter_request)
    runtime_config = _load_runtime_config(adapter_request, artifacts)
    source_root, python_site = _materialize_source(
        run_dir=config["run_dir"],
        worker_index=0,
        artifacts=artifacts,
        config=runtime_config,
        scratch_suffix=f"correction-{os.getpid()}",
    )
    backend: Any = OfficialSelfForcingBackend(
        source_root=source_root,
        artifacts=artifacts,
        config=runtime_config,
        device_name="cuda:0",
        python_site=python_site,
        chunk_count=config["chunk_count"],
        conditioning_cache_config=config.get("conditioning_cache"),
    )
    sdvg = config.get("sdvg")
    if sdvg is not None:
        from .sdvg_adapter import SdvgDraftBackend

        sdvg_document = _strict_json_file(sdvg["artifact_spec_path"])
        _require(
            sdvg_document.get("schema") == "main_table_baseline_artifacts_v1"
            and sdvg_document.get("status") == "frozen"
            and sdvg_document.get("method_id") == "sdvg"
            and sdvg_document.get("adapter_module") == "cfexp.sdvg_adapter"
            and sdvg_document.get("adapter_config") == sdvg["adapter_config"],
            "SDVG worker config is not bound to its frozen artifact specification",
        )
        sdvg_artifacts = sdvg_document.get("artifacts")
        _require(isinstance(sdvg_artifacts, list), "SDVG artifact list is invalid")
        expected_sdvg_hashes = {
            str(item.get("name")): item.get("sha256")
            for item in sdvg_artifacts
            if isinstance(item, Mapping)
        }
        actual_sdvg_hashes = {
            str(item.get("name")): item.get("sha256")
            for item in sdvg["verified_artifacts"]
            if isinstance(item, Mapping)
        }
        _require(
            len(expected_sdvg_hashes) == len(sdvg_artifacts)
            and actual_sdvg_hashes == expected_sdvg_hashes,
            "SDVG verified artifacts differ from the frozen specification",
        )

        backend = SdvgDraftBackend(
            base_backend=backend,
            run_dir=config["run_dir"],
            adapter_config=sdvg["adapter_config"],
            verified_artifacts=sdvg["verified_artifacts"],
        )
    generation = runtime_config["generation"]
    musiq = runtime_config["musiq"]
    backend.model_state = {
        "host_kv_cursor_optimization": backend.host_kv_cursor_optimization,
        "model_id": MODEL_IDS["draft"],
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "checkpoint_sha256": artifact_hashes[generation["checkpoint_artifact"]],
        "base_model_sha256": artifact_hashes["wan_model_weights"],
        "vae_sha256": artifact_hashes["wan_vae_checkpoint"],
        "text_encoder_sha256": artifact_hashes["wan_t5_checkpoint"],
        "musiq_spaq_sha256": artifact_hashes[musiq["checkpoint_artifact"]],
        "artifact_spec_sha256": config["artifact_spec_sha256"],
        "backend_config_sha256": config["_verified_config_sha256"],
        "dtype": generation["dtype"],
        "denoising_steps": 4,
        "native_first_timestep": generation["first_denoising_timestep"],
        "hidden_layer_zero_based": 14,
        "hidden_token_sample_count_max": 256,
        "effective_rank_relative_cutoff": 1e-7,
        "kv_policy": "rolling_local_attention_21_sink_0_fresh_rebuild",
        "kv_local_attention_size": 21,
        "kv_sink_size": 0,
        "main_table_t_stitch_step_supported": True,
        "main_table_sdvg_supported": sdvg is not None,
        "quality_metric": "musiq_spaq_frame_mean_raw_div_100",
        "chunk_count": config["chunk_count"],
    }
    if sdvg is not None:
        backend.model_state["sdvg"] = {
            "officiality_label": "unofficial_paper_faithful_reproduction",
            "image_reward_source_commit": sdvg["adapter_config"]["image_reward"][
                "source_commit"
            ],
            "image_reward_source_sha256": actual_sdvg_hashes[
                "image_reward_source"
            ],
            "image_reward_checkpoint_sha256": actual_sdvg_hashes[
                "image_reward_checkpoint"
            ],
            "image_reward_med_config_sha256": actual_sdvg_hashes[
                "image_reward_med_config"
            ],
            "bert_revision": sdvg["adapter_config"]["image_reward"][
                "bert_revision"
            ],
            "threshold": sdvg["adapter_config"]["routing"]["threshold"],
            "artifact_spec_sha256": sdvg["artifact_spec_sha256"],
        }
    if "conditioning_cache" in config:
        binding = getattr(backend, "conditioning_cache_binding", None)
        _require(isinstance(binding, Mapping), "Draft conditioning cache binding is absent")
        backend.model_state["conditioning_cache"] = dict(binding)
    return backend


def build_target_backend(raw_config: Mapping[str, Any]) -> Any:
    """Worker factory for the pinned Krea Realtime 14B Target."""

    config = _factory_config(raw_config, role="target")
    from .correction_artifacts import (
        materialize_target_runtime,
        reverify_target_artifacts,
        verify_bound_target_spec,
    )
    from .krea_chunk_backend import KreaChunkBackend

    spec = verify_bound_target_spec(
        config["artifact_spec_path"], config["artifact_spec_sha256"]
    )
    frozen_hashes = _bind_records_to_spec(spec, config["verified_artifacts"])
    artifacts, actual_hashes = reverify_target_artifacts(config["verified_artifacts"])
    _require(actual_hashes == frozen_hashes, "Target artifact re-verification drifted")
    runtime_config = spec["backend_config"]
    paths = materialize_target_runtime(
        run_dir=config["run_dir"],
        worker_label=f"{config['worker_label']}-{os.getpid()}",
        artifacts=artifacts,
        config=runtime_config,
    )
    backend = KreaChunkBackend(
        source_root=paths.source_root,
        python_site=paths.python_site,
        artifacts=artifacts,
        artifact_hashes=actual_hashes,
        config=runtime_config,
        device_name="cuda:0",
        chunk_count=config["chunk_count"],
        conditioning_cache_config=config.get("conditioning_cache"),
    )
    prior = getattr(backend, "model_state", {})
    _require(isinstance(prior, Mapping), "Target backend model_state is invalid")
    if "conditioning_cache" in config:
        _require(
            isinstance(prior.get("conditioning_cache"), Mapping),
            "Target conditioning cache binding is absent",
        )
    backend.model_state = {
        **dict(prior),
        "model_id": MODEL_IDS["target"],
        "artifact_spec_sha256": config["artifact_spec_sha256"],
        "backend_config_sha256": config["_verified_config_sha256"],
        "chunk_count": config["chunk_count"],
    }
    return backend


class NoiseTensorWriter(Protocol):
    def write_standard_normal(
        self, path: Path, *, seed: int, shape: Sequence[int], dtype: str
    ) -> None: ...


class TorchSafeTensorNoiseWriter:
    """Lazy production implementation; importing this module remains CPU-light."""

    def write_standard_normal(
        self, path: Path, *, seed: int, shape: Sequence[int], dtype: str
    ) -> None:
        _require(dtype == "bfloat16", "production correction noise must be bfloat16")
        try:
            import torch  # type: ignore
            from safetensors.torch import save_file  # type: ignore
        except ImportError as error:
            raise RuntimeBridgeError(
                "noise materialization requires torch and safetensors"
            ) from error
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        # Draw in float32 and quantize explicitly.  This avoids a device-specific
        # bfloat16 normal kernel while still persisting the exact tensor consumed
        # by both workers.
        with stage("noise_generate"):
            tensor = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
            tensor = tensor.to(dtype=torch.bfloat16).contiguous()
        with stage("noise_persist", detail="temporary_tensor"):
            save_file({"noise": tensor}, str(path))


class DeterministicNoiseProvider:
    """Materialize one immutable explicit-noise tensor per trajectory chunk."""

    namespace = NOISE_NAMESPACE

    def __init__(
        self,
        *,
        run_dir: str | Path,
        writer: NoiseTensorWriter | None = None,
        shape: Sequence[int] = NOISE_SHAPE,
        namespace: str = NOISE_NAMESPACE,
        directory: str = "noise",
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve(strict=True)
        _require(self.run_dir.is_dir(), "noise run_dir must be an existing directory")
        self.writer = writer if writer is not None else TorchSafeTensorNoiseWriter()
        self.shape = tuple(int(item) for item in shape)
        _require(self.shape == NOISE_SHAPE, f"noise shape must remain {NOISE_SHAPE}")
        _require(
            isinstance(namespace, str) and _LABEL_RE.fullmatch(namespace) is not None,
            "noise namespace must be a portable label",
        )
        _require(
            isinstance(directory, str) and _LABEL_RE.fullmatch(directory) is not None,
            "noise directory must be a portable label",
        )
        self.namespace = namespace
        self.directory = directory

    def materialize(
        self, *, prompt_id: str, trajectory_seed: int, chunk_index: int
    ) -> NoiseMaterialization:
        relative = (f"{self.directory}/{prompt_id}/seed-{trajectory_seed}/"
                    f"chunk-{chunk_index:02d}.safetensors")
        with timing_session(self.run_dir, "controller-noise", {
            "action": "noise_materialize", "role": "controller",
            "chunk_index": chunk_index, "history_chunk_count": None,
            "output_noise_path": relative, "trajectory_seed": trajectory_seed,
            # This API knows the prompt identifier, not its private prompt text.
            "prompt_sha256": None,
            "prompt_id_sha256": hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()
            if isinstance(prompt_id, str) else None,
        }) as session:
            with stage("noise_materialize"):
                result = self._materialize_impl(prompt_id=prompt_id,
                    trajectory_seed=trajectory_seed, chunk_index=chunk_index)
            if session is not None:
                session.metadata["noise_sha256"] = result.descriptor.tensor_sha256
            return result

    def _materialize_impl(
        self, *, prompt_id: str, trajectory_seed: int, chunk_index: int
    ) -> NoiseMaterialization:
        seed = derive_initial_noise_seed(
            namespace=self.namespace,
            prompt_id=prompt_id,
            trajectory_seed=trajectory_seed,
            chunk_index=chunk_index,
        )
        relative = (
            f"{self.directory}/{prompt_id}/seed-{trajectory_seed}/"
            f"chunk-{chunk_index:02d}.safetensors"
        )
        target = _run_file(self.run_dir, relative, must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with stage("noise_write"):
                self.writer.write_standard_normal(
                    temporary, seed=seed, shape=self.shape, dtype="bfloat16"
                )
            with stage("noise_hash", detail="temporary_tensor"):
                expected_sha256, expected_size = _sha256_file(temporary)
            if target.exists() or target.is_symlink():
                _require(
                    not target.is_symlink() and target.is_file(),
                    "noise target is not regular",
                )
                with stage("noise_hash", detail="existing_tensor"):
                    actual_sha256, actual_size = _sha256_file(target)
                _require(
                    (actual_sha256, actual_size) == (expected_sha256, expected_size),
                    f"existing deterministic noise conflicts at {relative}",
                )
            else:
                try:
                    with stage("noise_persist", detail="commit_tensor"):
                        os.link(temporary, target)
                except FileExistsError:
                    with stage("noise_hash", detail="existing_tensor"):
                        actual_sha256, actual_size = _sha256_file(target)
                    _require(
                        (actual_sha256, actual_size) == (expected_sha256, expected_size),
                        f"concurrent deterministic noise conflicts at {relative}",
                    )
            descriptor = InitialNoiseDescriptor.create(
                namespace=self.namespace,
                prompt_id=prompt_id,
                trajectory_seed=trajectory_seed,
                chunk_index=chunk_index,
                shape=self.shape,
                dtype="bfloat16",
                tensor_sha256=expected_sha256,
                tensor_nbytes=expected_size,
            )
            return NoiseMaterialization(descriptor=descriptor, tensor_path=relative)
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

class ProcessLike(Protocol):
    stdin: TextIO | None
    stdout: TextIO | None

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


@dataclass(frozen=True)
class DecodeResult:
    frames: Mapping[str, Any]
    video: Mapping[str, Any] | None


class JsonlWorkerClient:
    """Synchronous client for one persistent isolated correction worker."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        role: str,
        backend_factory: str,
        backend_config_path: str | Path,
        backend_config_sha256: str,
        visible_device: str,
        process_factory: Callable[..., ProcessLike] = subprocess.Popen,
        startup_timeout_seconds: float = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS,
        request_timeout_seconds: float = DEFAULT_WORKER_REQUEST_TIMEOUT_SECONDS,
        chunk_count: int = 16,
        expected_model_id: str | None = None,
    ) -> None:
        _require(role in MODEL_IDS, "worker role must be draft or target")
        _require(bool(backend_factory) and backend_factory.count(":") == 1,
                 "backend_factory must be MODULE:CALLABLE")
        _require(_SHA256_RE.fullmatch(backend_config_sha256) is not None,
                 "backend_config_sha256 is invalid")
        _require(isinstance(visible_device, str) and bool(visible_device.strip()),
                 "visible_device must be non-empty")
        _require(
            isinstance(chunk_count, int)
            and not isinstance(chunk_count, bool)
            and chunk_count in SUPPORTED_CHUNK_COUNTS,
            "chunk_count must be one of 16, 30, 60, or 120",
        )
        expected_model_id = MODEL_IDS[role] if expected_model_id is None else expected_model_id
        allowed_models = ({MODEL_IDS["draft"], "longlive_1.3b", "rolling_forcing_1.3b"}
                          if role == "draft" else {MODEL_IDS["target"]})
        _require(expected_model_id in allowed_models, "unsupported model for worker role")
        self.expected_model_id = expected_model_id
        self.chunk_count = chunk_count
        self.run_dir = Path(run_dir).expanduser().resolve(strict=True)
        _require(self.run_dir.is_dir(), "worker run_dir must be an existing directory")
        self.role = role
        self.requires_target_rpc_lifecycle = role == "target"
        self._persistent_session_id: str | None = None
        self._sdvg_prompt_id: str | None = None
        self._counter = 0
        self._closed = False
        self._failed = False
        self._startup_timeout_seconds = self._validate_timeout(
            startup_timeout_seconds, label="startup_timeout_seconds"
        )
        self._request_timeout_seconds = self._validate_timeout(
            request_timeout_seconds, label="request_timeout_seconds"
        )
        config_path = Path(backend_config_path).expanduser().resolve(strict=True)
        _require(not config_path.is_symlink() and config_path.is_file(),
                 "backend config must be a regular non-symlink file")
        actual, _size = _sha256_file(config_path)
        _require(actual == backend_config_sha256, "backend config digest mismatch")
        log_path = self.run_dir / "logs" / f"correction-{role}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path
        self._log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        environment = dict(os.environ)
        environment.update(
            {
                "CFEXP_RUN_DIR": str(self.run_dir),
                "CUDA_VISIBLE_DEVICES": visible_device,
            }
        )
        try:
            self.process = process_factory(
                [
                    sys.executable,
                    "-m",
                    "cfexp.correction_worker",
                    "--role",
                    role,
                    "--backend-factory",
                    backend_factory,
                    "--backend-config",
                    str(config_path),
                    "--backend-config-sha256",
                    backend_config_sha256,
                ],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                text=True,
                bufsize=1,
            )
            ready = self._read_response(
                stage="startup:ready",
                timeout_seconds=self._startup_timeout_seconds,
            )
            self._validate_envelope(
                ready, request_id="worker-startup", action="ready", allow_failure=False
            )
            payload = ready.get("payload")
            _require(isinstance(payload, Mapping), "ready response has no payload")
            state = payload.get("model_state")
            _require(isinstance(state, Mapping), "ready response has no model_state")
            _require(state.get("role") == role and state.get("poisoned") is False,
                     "worker ready state is invalid")
            backend = state.get("backend")
            _require(isinstance(backend, Mapping), "worker backend state is invalid")
            _require(backend.get("model_id") == self.expected_model_id, "worker loaded wrong model")
            _require(
                backend.get("chunk_count", 16) == self.chunk_count,
                "worker loaded a different chunk horizon",
            )
            artifact = backend.get("checkpoint_sha256")
            _require(isinstance(artifact, str) and _SHA256_RE.fullmatch(artifact) is not None,
                     "worker model state has no checkpoint digest")
            self.model_state = dict(backend)
            self.model_id = str(backend["model_id"])
            self.model_artifact_sha256 = artifact
            persistent_supported = (
                role == "target"
                and backend.get("main_table_persistent_session_supported") is True
            )
            self.session_policy = (
                "persistent_native_session"
                if persistent_supported
                else "history_reconstruction_per_chunk"
            )
            self.decode_during_generation = not (
                role == "target"
                and backend.get("main_table_interleaved_real_vae_decode") is False
            )
            self.t_stitch_step_supported = (
                role == "draft"
                and backend.get("main_table_t_stitch_step_supported") is True
            )
            self.sdvg_supported = (
                role == "draft" and backend.get("main_table_sdvg_supported") is True
            )
        except BaseException:
            self._abort_process()
            self._close_process_pipes()
            self._log_handle.close()
            raise

    @staticmethod
    def _validate_timeout(value: float, *, label: str) -> float:
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0,
            f"{label} must be a finite positive number",
        )
        return float(value)

    def _abort_process(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, TimeoutError):
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, TimeoutError):
                pass

    def _close_process_pipes(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            if stream is None:
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def _last_worker_log(self) -> str:
        try:
            with self._log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - WORKER_LOG_TAIL_BYTES))
                raw = handle.read()
        except OSError as error:
            return f"<unavailable: {type(error).__name__}: {error}>"
        text = raw.decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "<no worker stderr captured>"
        return " | ".join(lines[-WORKER_LOG_TAIL_LINES:])

    def _diagnostic_error(
        self, *, stage: str, reason: str, abort: bool = True
    ) -> RuntimeBridgeError:
        self._failed = True
        if abort:
            self._abort_process()
        process = getattr(self, "process", None)
        exit_code = None if process is None else process.poll()
        log_tail = json.dumps(self._last_worker_log(), ensure_ascii=False)
        return RuntimeBridgeError(
            f"{self.role} worker failure "
            f"[stage={stage}; exit={exit_code}; last_worker_log={log_tail}]: {reason}"
        )

    def _read_response(
        self, *, stage: str, timeout_seconds: float
    ) -> dict[str, Any]:
        stream = self.process.stdout
        _require(stream is not None, "worker stdout pipe is unavailable")
        result: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def read_one_line() -> None:
            try:
                result.put(("line", stream.readline()))
            except BaseException as error:
                result.put(("error", error))

        reader = threading.Thread(
            target=read_one_line,
            name=f"cfexp-{self.role}-{stage}-stdout",
            daemon=True,
        )
        reader.start()
        deadline = time.monotonic() + timeout_seconds
        event: tuple[str, Any] | None = None
        while event is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                try:
                    event = result.get_nowait()
                except queue.Empty:
                    raise self._diagnostic_error(
                        stage=stage,
                        reason=f"stdout response deadline exceeded after {timeout_seconds:.3f}s",
                    )
                break
            try:
                event = result.get(
                    timeout=min(WORKER_WATCHDOG_POLL_SECONDS, remaining)
                )
            except queue.Empty:
                exit_code = self.process.poll()
                if exit_code is None:
                    continue
                # The worker may have flushed its final line immediately before
                # exiting.  Give the reader one short scheduling quantum to
                # publish that already-buffered response before declaring loss.
                try:
                    event = result.get(
                        timeout=min(WORKER_WATCHDOG_POLL_SECONDS, remaining)
                    )
                except queue.Empty:
                    raise self._diagnostic_error(
                        stage=stage,
                        reason=f"process exited before a response (exit={exit_code})",
                        abort=False,
                    )
        assert event is not None
        event_type, event_value = event
        if event_type == "error":
            raise self._diagnostic_error(
                stage=stage,
                reason=(
                    "stdout reader failed: "
                    f"{type(event_value).__name__}: {event_value}"
                ),
            )
        line = event_value
        if not isinstance(line, str):
            raise self._diagnostic_error(
                stage=stage,
                reason=f"stdout reader returned {type(line).__name__}, not text",
            )
        if not line:
            code = self.process.poll()
            raise self._diagnostic_error(
                stage=stage,
                reason=f"worker protocol ended unexpectedly (exit={code})",
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise self._diagnostic_error(
                stage=stage,
                reason=f"worker emitted invalid JSON: {error}",
            ) from error
        if not isinstance(value, dict):
            raise self._diagnostic_error(
                stage=stage, reason="worker response must be an object"
            )
        return value

    def _validate_envelope(
        self,
        response: Mapping[str, Any],
        *,
        request_id: str,
        action: str,
        allow_failure: bool = False,
    ) -> None:
        _require(response.get("schema") == RESPONSE_SCHEMA, "worker response schema mismatch")
        _require(response.get("request_id") == request_id, "worker response request_id mismatch")
        _require(response.get("action") == action, "worker response action mismatch")
        _require(response.get("role") == self.role, "worker response role mismatch")
        ok = response.get("ok")
        _require(isinstance(ok, bool), "worker response ok flag is invalid")
        if not ok and not allow_failure:
            error = response.get("error")
            if isinstance(error, Mapping):
                raise RuntimeBridgeError(
                    f"{self.role} worker {action} failed: "
                    f"{error.get('type', 'Error')}: {error.get('message', '')}"
                )
            raise RuntimeBridgeError(f"{self.role} worker {action} failed")

    def _request(self, action: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        _require(not self._closed, "worker client is closed")
        _require(not self._failed, "worker client is unusable after a prior failure")
        self._counter += 1
        request_id = f"{self.role}-{self._counter:08d}"
        request_stage = f"request:{action}:{request_id}"
        request = {
            "schema": REQUEST_SCHEMA,
            "request_id": request_id,
            "action": action,
            "role": self.role,
            **dict(fields),
        }
        prompt = fields.get("prompt")
        output_paths = fields.get("output_paths")
        history_paths = fields.get("history_latent_paths")
        metadata = {
            "action": action, "request_id": request_id, "role": self.role,
            "chunk_index": fields.get("chunk_index"),
            "history_chunk_count": len(history_paths) if isinstance(history_paths, list) else None,
            "output_latent_path": output_paths.get("latent") if isinstance(output_paths, Mapping)
            else fields.get("output_latent_path"),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if isinstance(prompt, str) else None,
        }
        with timing_session(self.run_dir, f"client-{self.role}", metadata):
            with stage("rpc_total"):
                try:
                    with stage("rpc_serialize"):
                        raw = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                except (TypeError, ValueError) as error:
                    raise RuntimeBridgeError(f"worker request is not JSON: {error}") from error
                stream = self.process.stdin
                _require(stream is not None, "worker stdin pipe is unavailable")
                exit_code = self.process.poll()
                if exit_code is not None:
                    raise self._diagnostic_error(
                        stage=request_stage,
                        reason=f"process had already exited before request send (exit={exit_code})",
                        abort=False,
                    )
                try:
                    with stage("rpc_send"):
                        stream.write(raw + "\n")
                        stream.flush()
                except (BrokenPipeError, OSError) as error:
                    raise self._diagnostic_error(
                        stage=request_stage,
                        reason=f"cannot send request to worker: {error}",
                    ) from error
                with stage("rpc_wait"):
                    response = self._read_response(
                        stage=request_stage,
                        timeout_seconds=self._request_timeout_seconds,
                    )
                with stage("rpc_validate"):
                    self._validate_envelope(response, request_id=request_id, action=action)
                    payload = response.get("payload")
                    _require(isinstance(payload, dict), "worker response payload is invalid")
                return payload

    def _history_kv_widths(self, history: LatentHistory) -> list[int]:
        """Bind clean-cache writes to the consumer model, not just the producer.

        LongLive's native clean update consumes all three frames even when
        Krea supplied the block. Self-Forcing keeps its existing owner policy.
        The same vector is used for the request and returned reuse evidence.
        """
        _require(self.role == "draft", "history widths belong to the Draft consumer")
        return [3 if self.model_id in {"longlive_1.3b", "rolling_forcing_1.3b"} or item.producer_role == "draft" else 1
                for item in history.segments]

    def generate(self, request: ChunkGenerationRequest) -> GeneratedChunk:
        _require(isinstance(request, ChunkGenerationRequest), "generate request type is invalid")
        with timing_session(self.run_dir, f"controller-{self.role}", {
            "action": "generate", "role": self.role,
            "request_id": f"{self.role}-{self._counter + 1:08d}",
            "chunk_index": request.history.next_chunk_index,
            "history_chunk_count": len(request.history.segments),
            "output_latent_path": request.output_latent_path,
            "prompt_sha256": hashlib.sha256(request.prompt.prompt.encode("utf-8")).hexdigest(),
            "target_call_phase": request.target_call_phase,
        }):
            with stage("controller_generate"):
                return self._generate_impl(request)

    def generate_early(self, request: ChunkGenerationRequest, *, session_id: str, control: Mapping[str, Any]):
        from .early_protocol import validate_control
        _require(self.role == "draft" and not request.capture_hidden
                 and request.kv_branch_action is None, "early generation requires plain Draft")
        control = validate_control(control, chunk_index=request.history.next_chunk_index)
        _require(isinstance(session_id, str) and _SHA256_RE.fullmatch(session_id) is not None,
                 "invalid early session ID")
        with timing_session(self.run_dir, "controller-draft", {
                "action": "generate_early", "chunk_index": request.history.next_chunk_index,
                "output_latent_path": request.output_latent_path}):
            with stage("controller_generate"):
                return self._generate_impl(request, early_session_id=session_id, early_control=control)

    def commit_early_target(self, stopped, target: GeneratedChunk):
        from .early_protocol import StoppedDraft
        _require(self.role == "draft" and isinstance(stopped, StoppedDraft), "invalid early commit")
        request = stopped.request
        _require(target.context.role == "target"
                 and target.context.history_head_sha256 == request.history.head_sha256
                 and target.context.next_chunk_index == request.history.next_chunk_index
                 and target.context.noise == request.noise.descriptor,
                 "Target replacement did not use the same history and initial noise")
        payload = self._request("commit_early_target", {
            "early_session_id": stopped.session_id, "pending_token": stopped.pending_token,
            "target_latent_path": target.segment.latent_path,
            "target_latent_sha256": target.segment.latent_sha256})
        _require(payload.get("early_session_id") == stopped.session_id
                 and payload.get("pending_token") == stopped.pending_token
                 and payload.get("target_latent_path") == target.segment.latent_path
                 and payload.get("target_latent_sha256") == target.segment.latent_sha256,
                 "early commit returned the wrong accepted Target binding")
        _require(payload.get("result") == {"external_clean_forwards": 1,
                 "history_replay_chunks": 0, "branch_snapshot_bytes": 0,
                 "committed_chunks": request.history.next_chunk_index+1},
                 "early commit lacks native same-slot update evidence")
        return payload

    def _generate_impl(self, request: ChunkGenerationRequest, *, early_session_id=None, early_control=None):
        _require(isinstance(request, ChunkGenerationRequest), "generate request type is invalid")
        if self.role == "target":
            _require(request.kv_branch_action is None,
                     "Target generation cannot carry a Draft KV branch action")
            _require(
                request.target_logical_call_id is not None,
                "Target generate request has no logical call ID",
            )
            _require(request.target_call_phase is not None, "Target generate request has no phase")
        else:
            _require(
                request.target_logical_call_id is None
                and request.target_call_scope_id is None
                and request.target_call_phase is None,
                "Draft generate request carries Target lifecycle IDs",
            )
        chunk_index = request.history.next_chunk_index
        output_paths: dict[str, str] = {"latent": request.output_latent_path}
        if request.capture_hidden:
            stem = request.output_latent_path
            if stem.endswith(".safetensors"):
                stem = stem[: -len(".safetensors")]
            output_paths.update(
                {
                    "sampled_hidden": f"{stem}-sampled-hidden.safetensors",
                    "full_hidden": f"{stem}-full-hidden.safetensors",
                }
            )
        payload = self._request(
            "generate_early" if early_control is not None else "generate",
            {
                **({"early_session_id": early_session_id, "early_control": early_control}
                   if early_control is not None else {}),
                "prompt": request.prompt.prompt,
                "chunk_index": chunk_index,
                "internal_seed": request.noise.descriptor.derived_seed,
                "capture_hidden": request.capture_hidden,
                "history_latent_paths": [item.latent_path for item in request.history.segments],
                **({"history_kv_widths": self._history_kv_widths(request.history)} if self.role == "draft" else {}),
                "initial_noise_path": request.noise.tensor_path,
                "output_paths": output_paths,
                "target_logical_call_id": request.target_logical_call_id,
                "target_call_scope_id": request.target_call_scope_id,
                "target_call_phase": request.target_call_phase,
                **({"kv_branch_action": request.kv_branch_action}
                   if request.kv_branch_action is not None else {}),
            },
        )
        with stage("post_rpc_validation"):
            _require(payload.get("chunk_index") == chunk_index, "worker generated wrong chunk")
            _require(payload.get("history_chunk_count") == chunk_index,
                     "worker consumed wrong history length")
            result = payload.get("result")
            _require(isinstance(result, Mapping), "generate payload has no result")
            if self.role == "draft" and "kv_reuse" in result:
                from .draft_reuse_evidence import validate_reuse_evidence
                validate_reuse_evidence(result["kv_reuse"], history_chunks=chunk_index,
                    history_file_sha256=[x.latent_sha256 for x in request.history.segments],
                    prompt_sha256=hashlib.sha256(request.prompt.prompt.encode()).hexdigest(),
                    kv_branch_action=request.kv_branch_action,
                    history_kv_widths=self._history_kv_widths(request.history))
            if early_control is not None:
                from .early_protocol import decide, StoppedDraft
                decision = result.get("early_decision")
                _require(isinstance(decision, Mapping), "missing early decision")
                _require(decide(early_control, decision.get("effective_rank")) == decision,
                         "worker early decision differs from controller rule")
                evidence = result.get("early_execution")
                _require(isinstance(evidence, Mapping)
                         and evidence.get("schema") == "draft_early_execution_v1"
                         and evidence.get("stopped") is decision["replace_with_target"]
                         and evidence.get("branch_snapshot_bytes") == 0
                         and isinstance(result.get("kv_reuse"), Mapping),
                         "missing native early execution evidence")
                if decision["replace_with_target"]:
                    probe = result.get("probe")
                    _require(payload.get("stopped") is True and "latent" not in result
                             and not (self.run_dir / request.output_latent_path).exists()
                             and payload.get("early_session_id") == early_session_id
                             and isinstance(payload.get("pending_token"), str)
                             and _SHA256_RE.fullmatch(payload["pending_token"]) is not None,
                             "stopped Draft returned an output or invalid capability")
                    _require(isinstance(probe, Mapping) and probe.get("layer_index") == 14
                             and probe.get("timestep") == 1000 and probe.get("current_start") == chunk_index*4680
                             and probe.get("completed_native_forward") is False
                             and probe.get("effective_rank") == decision["effective_rank"],
                             "invalid partial Draft probe")
                    return StoppedDraft(request, early_session_id, payload["pending_token"],
                                        decision, dict(result))
                _require(result.get("rank") == decision["effective_rank"],
                         "continued Draft changed its probe rank")
            if request.kv_branch_action is not None:
                _require(isinstance(result.get("kv_reuse"), Mapping)
                         and isinstance(result["kv_reuse"].get("branch_anchor"), Mapping),
                         "worker did not return requested Draft branch-anchor evidence")
            latent = result.get("latent")
            _require(isinstance(latent, Mapping), "worker did not persist a latent tensor")
            _require(latent.get("path") == request.output_latent_path,
                     "worker wrote latent to an unexpected path")
            latent_path = _run_file(self.run_dir, request.output_latent_path, must_exist=True)
            with stage("post_rpc_hash"):
                latent_sha, latent_size = _sha256_file(latent_path)
            _require(
                latent.get("sha256") == latent_sha and latent.get("nbytes") == latent_size,
                "worker latent metadata does not match persisted bytes",
            )
            _require(latent.get("shape") == list(NOISE_SHAPE), "worker latent shape is invalid")
            _require(latent.get("dtype") == "bfloat16", "worker latent dtype is invalid")
            boundary_velocity_witness = _validated_boundary_velocity_witness(
                result,
                chunk_index=chunk_index,
                history_chunk_count=len(request.history.segments),
                input_dtype=str(latent["dtype"]),
                candidate_frame_count=int(latent["shape"][1]),
            )
            kv_hash = result.get("kv_witness_sha256")
            _require(isinstance(kv_hash, str) and _SHA256_RE.fullmatch(kv_hash) is not None,
                     "worker did not return a KV witness digest")
            kv = KVCacheDescriptor(
                owner_role=self.role,
                model_id=self.model_id,
                model_artifact_sha256=self.model_artifact_sha256,
                history_head_sha256=request.history.head_sha256,
                cache_sha256=kv_hash,
            )
            context = GenerationContext(
                role=self.role,
                model_id=self.model_id,
                model_artifact_sha256=self.model_artifact_sha256,
                history_head_sha256=request.history.head_sha256,
                next_chunk_index=chunk_index,
                noise=request.noise.descriptor,
                kv=kv,
            )
            with stage("post_rpc_segment_binding"):
                segment = LatentSegment.from_file(
                    history=request.history,
                    context=context,
                    latent_root=self.run_dir,
                    latent_path=request.output_latent_path,
                )
            rank_raw = result.get("rank")
            rank: float | None = None
            if rank_raw is not None:
                rank = float(rank_raw)
                _require(math.isfinite(rank) and rank >= 1.0, "worker effective rank is invalid")
            hidden_path: str | None = None
            hidden_sha: str | None = None
            if request.capture_hidden:
                full = result.get("full_hidden")
                sampled = result.get("sampled_hidden")
                _require(isinstance(full, Mapping) and isinstance(sampled, Mapping),
                         "captured generation omitted hidden tensors")
                _require(full.get("path") == output_paths["full_hidden"],
                         "worker full-hidden path mismatch")
                _require(sampled.get("path") == output_paths["sampled_hidden"],
                         "worker sampled-hidden path mismatch")
                hidden_path = str(full["path"])
                hidden_file = _run_file(self.run_dir, hidden_path, must_exist=True)
                with stage("post_rpc_hash"):
                    hidden_sha, hidden_size = _sha256_file(hidden_file)
                _require(
                    full.get("sha256") == hidden_sha and full.get("nbytes") == hidden_size,
                    "worker hidden-state metadata does not match persisted bytes",
                )
                _require(rank is not None, "captured Draft generation omitted effective rank")
            diagnostics = {
                key: value
                for key, value in result.items()
                if key not in {"latent", "full_hidden", "sampled_hidden", "rank"}
            }
            diagnostics["boundary_velocity_witness"] = boundary_velocity_witness
            if request.capture_hidden:
                diagnostics["sampled_hidden"] = result["sampled_hidden"]
            target_attempt_id = payload.get("target_attempt_id")
            target_logical_call_id = payload.get("target_logical_call_id")
            if self.role == "target":
                _require(
                    isinstance(target_attempt_id, str)
                    and _SHA256_RE.fullmatch(target_attempt_id) is not None,
                    "Target worker returned no valid lifecycle attempt ID",
                )
                _require(
                    target_logical_call_id == request.target_logical_call_id,
                    "Target worker returned the wrong logical call ID",
                )
            else:
                _require(
                    target_attempt_id is None and target_logical_call_id is None,
                    "Draft worker returned Target lifecycle IDs",
                )
            return GeneratedChunk(
                context=context,
                segment=segment,
                effective_rank=rank,
                hidden_state_path=hidden_path,
                hidden_state_sha256=hidden_sha,
                diagnostics=diagnostics,
                target_attempt_id=target_attempt_id,
                target_logical_call_id=target_logical_call_id,
            )

    def start_trajectory(
        self, *, prompt_id: str, prompt: str, seed: int, chunk_count: int
    ) -> Mapping[str, Any]:
        """Start one native Target session for the Full-Target reference."""

        _require(self.role == "target", "only Target client can start a trajectory")
        _require(
            self.session_policy == "persistent_native_session",
            "Target backend does not advertise persistent-session support",
        )
        _require(self._persistent_session_id is None, "a Target trajectory is already active")
        _require(
            isinstance(prompt_id, str) and bool(prompt_id),
            "persistent prompt_id must be non-empty",
        )
        _require(
            isinstance(prompt, str) and bool(prompt.strip()),
            "persistent prompt must be non-empty",
        )
        _require(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            "persistent seed must be a non-negative integer",
        )
        _require(chunk_count == self.chunk_count, "persistent chunk horizon drifted")
        payload = self._request(
            "start_trajectory",
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": seed,
                "chunk_count": chunk_count,
            },
        )
        _exact_keys(
            payload,
            {"session_id", "prompt_encode_seconds"},
            label="persistent trajectory startup payload",
        )
        session_id = payload["session_id"]
        prompt_encode_seconds = payload["prompt_encode_seconds"]
        _require(
            isinstance(session_id, str) and _SHA256_RE.fullmatch(session_id) is not None,
            "persistent trajectory returned an invalid session ID",
        )
        _require(
            isinstance(prompt_encode_seconds, (int, float))
            and not isinstance(prompt_encode_seconds, bool)
            and math.isfinite(float(prompt_encode_seconds))
            and float(prompt_encode_seconds) >= 0.0,
            "persistent trajectory returned invalid prompt timing",
        )
        self._persistent_session_id = session_id
        return {
            "session_id": session_id,
            "prompt_encode_seconds": float(prompt_encode_seconds),
        }

    def start_sdvg_trajectory(
        self, *, prompt_id: str, prompt: str, chunk_count: int
    ) -> Mapping[str, Any]:
        """Reset the Draft streaming VAE cache for one SDVG prompt."""

        _require(self.role == "draft", "only Draft client supports SDVG")
        _require(self.sdvg_supported, "Draft backend does not advertise SDVG support")
        _require(self._sdvg_prompt_id is None, "an SDVG trajectory is already active")
        _require(isinstance(prompt_id, str) and bool(prompt_id), "SDVG prompt_id is empty")
        _require(isinstance(prompt, str) and bool(prompt.strip()), "SDVG prompt is empty")
        _require(chunk_count == self.chunk_count, "SDVG chunk horizon drifted")
        payload = self._request(
            "start_sdvg_trajectory",
            {"prompt_id": prompt_id, "prompt": prompt, "chunk_count": chunk_count},
        )
        _require(
            payload.get("prompt_id") == prompt_id
            and payload.get("chunk_count") == chunk_count,
            "Draft worker returned invalid SDVG startup evidence",
        )
        threshold = payload.get("threshold")
        _require(
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold)),
            "Draft worker returned an invalid SDVG threshold",
        )
        self._sdvg_prompt_id = prompt_id
        return {**payload, "threshold": float(threshold)}

    def score_sdvg_candidate(
        self, *, prompt: str, chunk_index: int, clean_latent_path: str
    ) -> Mapping[str, Any]:
        """Decode and ImageReward-score every RGB frame in one Draft block."""

        _require(self.role == "draft" and self.sdvg_supported, "SDVG is unavailable")
        _require(self._sdvg_prompt_id is not None, "no SDVG trajectory is active")
        _require(isinstance(prompt, str) and bool(prompt.strip()), "SDVG prompt is empty")
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and 0 <= chunk_index < self.chunk_count,
            "SDVG chunk index is invalid",
        )
        latent_path = _relative_path(clean_latent_path, label="clean_latent_path")
        payload = self._request(
            "score_sdvg_candidate",
            {
                "prompt": prompt,
                "chunk_index": chunk_index,
                "clean_latent_path": latent_path,
            },
        )
        _require(
            payload.get("clean_latent_path") == latent_path,
            "SDVG worker scored the wrong latent",
        )
        result = payload.get("result")
        _require(isinstance(result, Mapping), "SDVG worker omitted reward evidence")
        expected_frames = 9 if chunk_index == 0 else 12
        rewards_raw = result.get("frame_rewards")
        _require(
            result.get("schema") == "sdvg_image_reward_block_v1"
            and result.get("chunk_index") == chunk_index
            and result.get("rgb_frame_count") == expected_frames
            and isinstance(rewards_raw, list)
            and len(rewards_raw) == expected_frames,
            "SDVG reward evidence shape is invalid",
        )
        rewards = [float(value) for value in rewards_raw]
        _require(
            all(math.isfinite(value) for value in rewards),
            "SDVG frame reward is non-finite",
        )
        block_reward = float(result.get("block_reward"))
        threshold = float(result.get("threshold"))
        _require(
            math.isfinite(block_reward)
            and math.isfinite(threshold)
            and block_reward == min(rewards)
            and result.get("accepted_by_threshold") == (block_reward >= threshold)
            and result.get("forced_reject") is (chunk_index == 0),
            "SDVG reward decision is inconsistent",
        )
        for field in ("decode_wall_seconds", "reward_wall_seconds"):
            value = result.get(field)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0,
                f"SDVG {field} is invalid",
            )
        return dict(result)

    def finish_sdvg_trajectory(self, *, prompt_id: str) -> Mapping[str, Any]:
        """Close one complete SDVG reward stream and clear its VAE cache."""

        _require(
            self.role == "draft"
            and self.sdvg_supported
            and self._sdvg_prompt_id == prompt_id,
            "SDVG completion does not match the active prompt",
        )
        payload = self._request("finish_sdvg_trajectory", {"prompt_id": prompt_id})
        _require(
            payload.get("prompt_id") == prompt_id
            and payload.get("scored_chunk_count") == self.chunk_count,
            "Draft worker returned invalid SDVG completion evidence",
        )
        self._sdvg_prompt_id = None
        return dict(payload)

    def generate_persistent(self, request: ChunkGenerationRequest) -> GeneratedChunk:
        _require(isinstance(request, ChunkGenerationRequest), "generate_persistent request type is invalid")
        with timing_session(self.run_dir, f"controller-{self.role}", {
            "action": "generate_persistent", "role": self.role,
            "request_id": f"{self.role}-{self._counter + 1:08d}",
            "chunk_index": request.history.next_chunk_index,
            "history_chunk_count": len(request.history.segments),
            "output_latent_path": request.output_latent_path,
            "prompt_sha256": hashlib.sha256(request.prompt.prompt.encode("utf-8")).hexdigest(),
            "target_call_phase": request.target_call_phase,
        }):
            with stage("controller_generate"):
                return self._generate_persistent_impl(request)

    def _generate_persistent_impl(self, request: ChunkGenerationRequest) -> GeneratedChunk:
        """Generate the next chunk without transporting or rebuilding history."""

        _require(self.role == "target", "only Target client supports persistent generation")
        _require(
            self._persistent_session_id is not None,
            "persistent Target generation requires an active trajectory",
        )
        _require(
            isinstance(request, ChunkGenerationRequest),
            "persistent generate request type is invalid",
        )
        _require(not request.capture_hidden, "persistent Target may not capture Draft hidden state")
        _require(
            request.target_logical_call_id is not None
            and request.target_call_phase is not None,
            "persistent Target request lacks lifecycle identity",
        )
        chunk_index = request.history.next_chunk_index
        payload = self._request(
            "generate_persistent",
            {
                "prompt": request.prompt.prompt,
                "chunk_index": chunk_index,
                "internal_seed": request.noise.descriptor.derived_seed,
                # These paths are an audit binding only.  The worker validates
                # them but deliberately does not load or concatenate them.
                "history_latent_paths": [
                    item.latent_path for item in request.history.segments
                ],
                "initial_noise_path": request.noise.tensor_path,
                "output_latent_path": request.output_latent_path,
                "target_logical_call_id": request.target_logical_call_id,
                "target_call_scope_id": request.target_call_scope_id,
                "target_call_phase": request.target_call_phase,
            },
        )
        with stage("post_rpc_validation"):
            _require(payload.get("chunk_index") == chunk_index, "worker generated wrong chunk")
            _require(
                payload.get("history_chunk_count") == chunk_index,
                "worker bound the wrong persistent history length",
            )
            result = payload.get("result")
            _require(isinstance(result, Mapping), "persistent generate payload has no result")
            latent = result.get("latent")
            _require(isinstance(latent, Mapping), "persistent worker did not persist a latent")
            _require(
                latent.get("path") == request.output_latent_path,
                "persistent worker wrote latent to an unexpected path",
            )
            latent_path = _run_file(self.run_dir, request.output_latent_path, must_exist=True)
            with stage("post_rpc_hash"):
                latent_sha, latent_size = _sha256_file(latent_path)
            _require(
                latent.get("sha256") == latent_sha and latent.get("nbytes") == latent_size,
                "persistent latent metadata does not match persisted bytes",
            )
            _require(latent.get("shape") == list(NOISE_SHAPE), "persistent latent shape is invalid")
            _require(latent.get("dtype") == "bfloat16", "persistent latent dtype is invalid")
            kv_hash = result.get("kv_witness_sha256")
            _require(
                isinstance(kv_hash, str) and _SHA256_RE.fullmatch(kv_hash) is not None,
                "persistent worker did not return a KV witness digest",
            )
            _require(
                result.get("native_session_id") == self._persistent_session_id,
                "persistent worker returned the wrong native session ID",
            )
            kv = KVCacheDescriptor(
                owner_role=self.role,
                model_id=self.model_id,
                model_artifact_sha256=self.model_artifact_sha256,
                history_head_sha256=request.history.head_sha256,
                cache_sha256=kv_hash,
            )
            context = GenerationContext(
                role=self.role,
                model_id=self.model_id,
                model_artifact_sha256=self.model_artifact_sha256,
                history_head_sha256=request.history.head_sha256,
                next_chunk_index=chunk_index,
                noise=request.noise.descriptor,
                kv=kv,
            )
            with stage("post_rpc_segment_binding"):
                segment = LatentSegment.from_file(
                    history=request.history,
                    context=context,
                    latent_root=self.run_dir,
                    latent_path=request.output_latent_path,
                )
            diagnostics = {key: value for key, value in result.items() if key != "latent"}
            target_attempt_id = payload.get("target_attempt_id")
            _require(
                isinstance(target_attempt_id, str)
                and _SHA256_RE.fullmatch(target_attempt_id) is not None,
                "persistent Target worker returned no valid lifecycle attempt ID",
            )
            _require(
                payload.get("target_logical_call_id") == request.target_logical_call_id,
                "persistent Target worker returned the wrong logical call ID",
            )
            return GeneratedChunk(
                context=context,
                segment=segment,
                effective_rank=None,
                hidden_state_path=None,
                hidden_state_sha256=None,
                diagnostics=diagnostics,
                target_attempt_id=target_attempt_id,
                target_logical_call_id=request.target_logical_call_id,
            )

    def append_persistent_clean(
        self,
        *,
        prompt: str,
        chunk_index: int,
        history_latent_paths: Sequence[str],
        clean_latent_path: str,
    ) -> Mapping[str, Any]:
        """Mirror one SDVG-accepted Draft block into the live Target history."""

        _require(self.role == "target", "only Target client supports clean commits")
        _require(
            self._persistent_session_id is not None,
            "persistent Target clean commit requires an active trajectory",
        )
        _require(isinstance(prompt, str) and bool(prompt.strip()), "prompt is empty")
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and 1 <= chunk_index < self.chunk_count,
            "persistent clean commit chunk index is invalid",
        )
        canonical_history = [
            _relative_path(value, label=f"history_latent_paths[{index}]")
            for index, value in enumerate(history_latent_paths)
        ]
        _require(
            len(canonical_history) == chunk_index
            and len(set(canonical_history)) == len(canonical_history),
            "persistent clean commit history binding is invalid",
        )
        clean_path = _relative_path(clean_latent_path, label="clean_latent_path")
        _require(
            clean_path not in set(canonical_history),
            "persistent clean latent may not alias Target history",
        )
        payload = self._request(
            "append_persistent_clean",
            {
                "prompt": prompt,
                "chunk_index": chunk_index,
                "history_latent_paths": canonical_history,
                "clean_latent_path": clean_path,
            },
        )
        _require(payload.get("chunk_index") == chunk_index, "worker committed wrong chunk")
        _require(
            payload.get("history_chunk_count") == chunk_index,
            "worker bound wrong clean-commit history length",
        )
        clean = payload.get("clean_latent")
        _require(isinstance(clean, Mapping), "clean commit response has no input evidence")
        _require(clean.get("path") == clean_path, "clean commit bound the wrong latent")
        clean_file = _run_file(self.run_dir, clean_path, must_exist=True)
        clean_sha, clean_size = _sha256_file(clean_file)
        _require(
            clean.get("sha256") == clean_sha and clean.get("nbytes") == clean_size,
            "clean commit input metadata does not match persisted bytes",
        )
        result = payload.get("result")
        _require(isinstance(result, Mapping), "clean commit response has no result")
        _require(
            result.get("native_session_id") == self._persistent_session_id,
            "clean commit returned the wrong native session ID",
        )
        _require(
            result.get("native_session_block_index") == chunk_index,
            "clean commit returned the wrong native block index",
        )
        kv_hash = result.get("kv_witness_sha256")
        _require(
            isinstance(kv_hash, str) and _SHA256_RE.fullmatch(kv_hash) is not None,
            "clean commit returned no valid KV witness digest",
        )
        return {
            "chunk_index": chunk_index,
            "history_chunk_count": chunk_index,
            "clean_latent": dict(clean),
            "result": dict(result),
        }

    def generate_t_stitch_step(
        self,
        request: ChunkGenerationRequest,
        *,
        renoise_750_path: str,
        draft_clean_output_path: str,
        noisy_latent_750_output_path: str,
    ) -> Mapping[str, Any]:
        """Run the one-step Draft leg and persist its clean/handoff tensors."""

        _require(self.role == "draft", "only Draft client supports the T-Stitch step")
        _require(
            self.t_stitch_step_supported,
            "Draft backend does not advertise T-Stitch step support",
        )
        _require(isinstance(request, ChunkGenerationRequest), "T-Stitch request is invalid")
        _require(not request.capture_hidden, "T-Stitch Draft step may not capture hidden state")
        _require(
            request.target_logical_call_id is None
            and request.target_call_scope_id is None
            and request.target_call_phase is None,
            "T-Stitch Draft step may not carry Target lifecycle identity",
        )
        chunk_index = request.history.next_chunk_index
        renoise_path = _relative_path(renoise_750_path, label="renoise_750_path")
        clean_output = _relative_path(
            draft_clean_output_path, label="draft_clean_output_path"
        )
        handoff_output = _relative_path(
            noisy_latent_750_output_path,
            label="noisy_latent_750_output_path",
        )
        _require(clean_output != handoff_output, "T-Stitch Draft outputs must be distinct")
        payload = self._request(
            "generate_t_stitch_step",
            {
                "prompt": request.prompt.prompt,
                "chunk_index": chunk_index,
                "internal_seed": request.noise.descriptor.derived_seed,
                "history_latent_paths": [
                    item.latent_path for item in request.history.segments
                ],
                "initial_noise_path": request.noise.tensor_path,
                "renoise_750_path": renoise_path,
                "output_paths": {
                    "draft_clean_latent": clean_output,
                    "noisy_latent_750": handoff_output,
                },
            },
        )
        _require(payload.get("chunk_index") == chunk_index, "worker ran wrong Draft chunk")
        _require(
            payload.get("history_chunk_count") == chunk_index,
            "worker bound wrong T-Stitch Draft history length",
        )
        result = payload.get("result")
        _require(isinstance(result, Mapping), "T-Stitch Draft response has no result")
        for field, expected_path in (
            ("draft_clean_latent", clean_output),
            ("noisy_latent_750", handoff_output),
        ):
            metadata = result.get(field)
            _require(isinstance(metadata, Mapping), f"T-Stitch Draft omitted {field}")
            _require(metadata.get("path") == expected_path, f"T-Stitch Draft {field} path drifted")
            path = _run_file(self.run_dir, expected_path, must_exist=True)
            sha256, nbytes = _sha256_file(path)
            _require(
                metadata.get("sha256") == sha256 and metadata.get("nbytes") == nbytes,
                f"T-Stitch Draft {field} metadata does not match persisted bytes",
            )
            _require(metadata.get("shape") == list(NOISE_SHAPE), f"T-Stitch Draft {field} shape is invalid")
            _require(metadata.get("dtype") == "bfloat16", f"T-Stitch Draft {field} dtype is invalid")
        kv_hash = result.get("kv_witness_sha256")
        _require(
            isinstance(kv_hash, str) and _SHA256_RE.fullmatch(kv_hash) is not None,
            "T-Stitch Draft returned no valid KV witness digest",
        )
        trace = result.get("denoising_forward_trace")
        _require(
            isinstance(trace, list)
            and len(trace) == 1
            and isinstance(trace[0], Mapping)
            and float(trace[0].get("timestep", float("nan"))) == 1000.0,
            "T-Stitch Draft response did not prove one step at 1000",
        )
        return {
            "chunk_index": chunk_index,
            "history_chunk_count": chunk_index,
            "result": dict(result),
        }

    def generate_persistent_suffix(
        self,
        request: ChunkGenerationRequest,
        *,
        noisy_latent_750_path: str,
        renoise_500_path: str,
        renoise_250_path: str,
    ) -> GeneratedChunk:
        """Run the exact Target 750/500/250 suffix for one T-Stitch block."""

        _require(self.role == "target", "only Target client supports suffix generation")
        _require(
            self._persistent_session_id is not None,
            "persistent Target suffix requires an active trajectory",
        )
        _require(isinstance(request, ChunkGenerationRequest), "suffix request type is invalid")
        _require(not request.capture_hidden, "Target suffix may not capture Draft hidden state")
        _require(
            request.target_logical_call_id is not None
            and request.target_call_phase is not None,
            "Target suffix request lacks lifecycle identity",
        )
        chunk_index = request.history.next_chunk_index
        noisy_path = _relative_path(
            noisy_latent_750_path, label="noisy_latent_750_path"
        )
        noise_500_path = _relative_path(renoise_500_path, label="renoise_500_path")
        noise_250_path = _relative_path(renoise_250_path, label="renoise_250_path")
        _require(
            len({noisy_path, noise_500_path, noise_250_path}) == 3,
            "Target suffix tensor inputs must be distinct",
        )
        payload = self._request(
            "generate_persistent_suffix",
            {
                "prompt": request.prompt.prompt,
                "chunk_index": chunk_index,
                "internal_seed": request.noise.descriptor.derived_seed,
                "history_latent_paths": [
                    item.latent_path for item in request.history.segments
                ],
                "noisy_latent_750_path": noisy_path,
                "renoise_500_path": noise_500_path,
                "renoise_250_path": noise_250_path,
                "output_latent_path": request.output_latent_path,
                "target_logical_call_id": request.target_logical_call_id,
                "target_call_scope_id": request.target_call_scope_id,
                "target_call_phase": request.target_call_phase,
            },
        )
        _require(payload.get("chunk_index") == chunk_index, "worker generated wrong suffix chunk")
        _require(
            payload.get("history_chunk_count") == chunk_index,
            "worker bound the wrong suffix history length",
        )
        result = payload.get("result")
        _require(isinstance(result, Mapping), "persistent suffix payload has no result")
        latent = result.get("latent")
        _require(isinstance(latent, Mapping), "persistent suffix did not persist a latent")
        _require(
            latent.get("path") == request.output_latent_path,
            "persistent suffix wrote latent to an unexpected path",
        )
        latent_path = _run_file(self.run_dir, request.output_latent_path, must_exist=True)
        latent_sha, latent_size = _sha256_file(latent_path)
        _require(
            latent.get("sha256") == latent_sha and latent.get("nbytes") == latent_size,
            "persistent suffix latent metadata does not match persisted bytes",
        )
        _require(latent.get("shape") == list(NOISE_SHAPE), "persistent suffix shape is invalid")
        _require(latent.get("dtype") == "bfloat16", "persistent suffix dtype is invalid")
        kv_hash = result.get("kv_witness_sha256")
        _require(
            isinstance(kv_hash, str) and _SHA256_RE.fullmatch(kv_hash) is not None,
            "persistent suffix returned no valid KV witness digest",
        )
        _require(
            result.get("native_session_id") == self._persistent_session_id,
            "persistent suffix returned the wrong native session ID",
        )
        kv = KVCacheDescriptor(
            owner_role=self.role,
            model_id=self.model_id,
            model_artifact_sha256=self.model_artifact_sha256,
            history_head_sha256=request.history.head_sha256,
            cache_sha256=kv_hash,
        )
        context = GenerationContext(
            role=self.role,
            model_id=self.model_id,
            model_artifact_sha256=self.model_artifact_sha256,
            history_head_sha256=request.history.head_sha256,
            next_chunk_index=chunk_index,
            noise=request.noise.descriptor,
            kv=kv,
        )
        segment = LatentSegment.from_file(
            history=request.history,
            context=context,
            latent_root=self.run_dir,
            latent_path=request.output_latent_path,
        )
        target_attempt_id = payload.get("target_attempt_id")
        _require(
            isinstance(target_attempt_id, str)
            and _SHA256_RE.fullmatch(target_attempt_id) is not None,
            "persistent suffix returned no valid lifecycle attempt ID",
        )
        _require(
            payload.get("target_logical_call_id") == request.target_logical_call_id,
            "persistent suffix returned the wrong logical call ID",
        )
        return GeneratedChunk(
            context=context,
            segment=segment,
            effective_rank=None,
            hidden_state_path=None,
            hidden_state_sha256=None,
            diagnostics={key: value for key, value in result.items() if key != "latent"},
            target_attempt_id=target_attempt_id,
            target_logical_call_id=request.target_logical_call_id,
        )

    def finish_trajectory(self, *, session_id: str) -> Mapping[str, Any]:
        """Finish and release the active native Target session."""

        _require(self.role == "target", "only Target client can finish a trajectory")
        _require(
            self._persistent_session_id is not None
            and session_id == self._persistent_session_id,
            "persistent Target session ID drifted",
        )
        payload = self._request("finish_trajectory", {"session_id": session_id})
        _exact_keys(
            payload,
            {"session_id", "generated_block_count"},
            label="persistent trajectory completion payload",
        )
        _require(payload.get("session_id") == session_id, "worker finished the wrong session")
        generated = payload.get("generated_block_count")
        _require(
            isinstance(generated, int)
            and not isinstance(generated, bool)
            and generated == self.chunk_count,
            "worker finished an incomplete persistent trajectory",
        )
        self._persistent_session_id = None
        return {"session_id": session_id, "generated_block_count": generated}

    def decode_video(self, *, history: LatentHistory, output_video_path: str,
                     fps: int = 24) -> DecodeResult:
        _require(self.role == 'draft', 'only Draft client can decode video')
        payload = self._request('decode_video', {
            'history_latent_paths': [item.latent_path for item in history.segments],
            'output_video_path': output_video_path, 'fps': fps})
        _require(set(payload) == {'latent_chunk_count', 'pixels', 'video'}
                 and payload['latent_chunk_count'] == history.next_chunk_index,
                 'video-only decoder consumed the wrong history')
        _require(isinstance(payload['pixels'], Mapping)
                 and isinstance(payload['video'], Mapping)
                 and payload['video'].get('path') == output_video_path,
                 'video-only decoder returned invalid metadata')
        return DecodeResult(frames=dict(payload['pixels']), video=dict(payload['video']))

    def decode(
        self,
        *,
        history: LatentHistory,
        output_frames_path: str,
        output_video_path: str | None,
        fps: int = 24,
    ) -> DecodeResult:
        _require(self.role == "draft", "only Draft client can decode")
        payload = self._request(
            "decode",
            {
                "history_latent_paths": [item.latent_path for item in history.segments],
                "output_frames_path": output_frames_path,
                "output_video_path": output_video_path,
                "fps": fps,
            },
        )
        _require(payload.get("latent_chunk_count") == history.next_chunk_index,
                 "decoder consumed wrong latent history")
        frames = payload.get("frames")
        _require(isinstance(frames, Mapping), "decode response has no frame tensor")
        _require(frames.get("path") == output_frames_path, "decoded frame path mismatch")
        video = payload.get("video")
        _require(video is None or isinstance(video, Mapping), "decode video metadata is invalid")
        return DecodeResult(frames=dict(frames), video=None if video is None else dict(video))

    def score(
        self,
        *,
        frames_path: str,
        frame_indices: Sequence[int],
        temporal_window: tuple[int, int],
    ) -> QualityMeasurement:
        _require(self.role == "draft", "only Draft client can score")
        payload = self._request(
            "score",
            {
                "frames_path": frames_path,
                "frame_indices": list(frame_indices),
                "temporal_window": {
                    "start": temporal_window[0],
                    "stop": temporal_window[1],
                },
            },
        )
        _require(payload.get("frame_indices") == list(frame_indices),
                 "quality worker scored different frames")
        _require(payload.get("frames_path") == frames_path,
                 "quality worker scored a different frame tensor")
        returned_window = payload.get("temporal_window")
        _require(
            returned_window
            == {"start": temporal_window[0], "stop": temporal_window[1]},
            "quality worker used a different temporal window",
        )
        scores_raw = payload.get("quality_scores")
        _require(
            isinstance(scores_raw, list)
            and len(scores_raw) == len(frame_indices)
            and bool(scores_raw),
            "quality worker omitted one score per requested frame",
        )
        scores: list[float] = []
        for index, value in enumerate(scores_raw):
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0,
                f"quality worker score {index} is not finite in [0,1]",
            )
            scores.append(float(value))
        quality_mean_raw = payload.get("quality_mean")
        _require(
            isinstance(quality_mean_raw, (int, float))
            and not isinstance(quality_mean_raw, bool)
            and math.isfinite(float(quality_mean_raw))
            and 0.0 <= float(quality_mean_raw) <= 1.0,
            "quality worker returned an invalid mean",
        )
        quality_mean = float(quality_mean_raw)
        expected_mean = math.fsum(scores) / len(scores)
        _require(
            quality_mean == expected_mean,
            "quality worker mean differs from its individual scores",
        )
        temporal_raw = payload.get("temporal_consistency")
        _require(
            isinstance(temporal_raw, (int, float))
            and not isinstance(temporal_raw, bool)
            and math.isfinite(float(temporal_raw))
            and 0.0 <= float(temporal_raw) <= 1.0,
            "quality worker omitted or returned invalid temporal consistency",
        )
        return QualityMeasurement(
            primary=quality_mean,
            temporal_consistency=float(temporal_raw),
            frame_scores=tuple(scores),
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._failed:
                self._abort_process()
                self._closed = True
                return
            self._request("shutdown", {})
            self._closed = True
            try:
                code = self.process.wait(timeout=30.0)
            except (subprocess.TimeoutExpired, TimeoutError) as error:
                raise self._diagnostic_error(
                    stage="shutdown:process-exit",
                    reason="process did not exit within 30.000s after shutdown response",
                ) from error
            if code != 0:
                raise self._diagnostic_error(
                    stage="shutdown:process-exit",
                    reason=f"worker exited with status {code}",
                    abort=False,
                )
        finally:
            if not self._closed:
                self._abort_process()
                self._closed = True
            self._close_process_pipes()
            self._log_handle.close()

    def __enter__(self) -> JsonlWorkerClient:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


class DraftQualityEvaluator:
    """Decode losslessly per compared prefix and apply the frozen frame windows."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        client: JsonlWorkerClient,
        write_full_videos: bool = False,
        vbench_scorer: OfficialVBenchLongScorer | None = None,
        full_quality_metric_id: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve(strict=True)
        _require(self.run_dir.is_dir(), "quality run_dir must be an existing directory")
        _require(client.role == "draft", "quality evaluator requires the Draft worker")
        self.client = client
        self.write_full_videos = bool(write_full_videos)
        self.vbench_scorer = vbench_scorer
        if vbench_scorer is None:
            _require(
                full_quality_metric_id in {None, PRIMARY_QUALITY_METRIC},
                "legacy full-video quality metric must remain MUSIQ-SPAQ",
            )
            self.full_quality_metric_id = PRIMARY_QUALITY_METRIC
        else:
            _require(
                full_quality_metric_id == VBENCH_LONG_PRIMARY_QUALITY_METRIC,
                "official VBench-Long mode requires its exact metric identity",
            )
            self.full_quality_metric_id = VBENCH_LONG_PRIMARY_QUALITY_METRIC
        self._local_baseline_cache_key: str | None = None

        state = client.model_state
        _require(
            state.get("quality_metric") == PRIMARY_QUALITY_METRIC,
            "Draft worker quality metric identity is not frozen",
        )
        for field in ("musiq_spaq_sha256", "vae_sha256"):
            value = state.get(field)
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"Draft worker model state has no valid {field}",
            )

    @staticmethod
    def _frame_count(history: LatentHistory) -> int:
        latent_frames = 3 * history.next_chunk_index
        _require(latent_frames >= 1, "cannot evaluate an empty latent history")
        return 1 + 4 * (latent_frames - 1)

    @staticmethod
    def _decode_evidence_relative(
        history: LatentHistory, *, measurement_id: str | None = None
    ) -> str:
        if measurement_id is None:
            return f"decoded/{history.manifest_sha256}.json"
        return f"decoded/vbench-long/{measurement_id}.json"

    def _full_decode_evidence_relative(
        self, *, history: LatentHistory, measurement_id: str
    ) -> str:
        """Resolve canonical or measurement-specific evidence for one video."""

        expected_video = f"videos/{measurement_id}.mp4"
        alias_relative = self._decode_evidence_relative(
            history, measurement_id=measurement_id
        )
        alias_path = self.run_dir / alias_relative
        if alias_path.exists():
            return alias_relative
        canonical_relative = self._decode_evidence_relative(history)
        canonical_path = self.run_dir / canonical_relative
        if canonical_path.exists():
            canonical = _strict_json_file(canonical_path)
            video = canonical.get("video")
            if isinstance(video, Mapping) and video.get("path") == expected_video:
                return canonical_relative
        return alias_relative

    def _materialize_shared_history_video_alias(
        self,
        *,
        history: LatentHistory,
        measurement_id: str,
        canonical_evidence: Mapping[str, Any],
    ) -> str:
        """Bind the same deterministic decode to another audited route ID."""

        expected_frames = self._frame_count(history)
        _require(
            canonical_evidence.get("schema") == DECODE_EVIDENCE_SCHEMA
            and canonical_evidence.get("history_manifest_sha256")
            == history.manifest_sha256
            and canonical_evidence.get("latent_chunk_count")
            == history.next_chunk_index
            and canonical_evidence.get("expected_frames") == expected_frames,
            "shared-history decode evidence is bound to another history",
        )
        frames = canonical_evidence.get("frames")
        source_video = canonical_evidence.get("video")
        _require(
            isinstance(frames, Mapping)
            and frames.get("path")
            == f"decoded/{history.manifest_sha256}.safetensors"
            and isinstance(source_video, Mapping),
            "shared-history decode evidence is incomplete",
        )
        self._verify_video_metadata(source_video)
        source_relative = _relative_path(
            source_video.get("path"), label="shared-history source video path"
        )
        source_path = _run_file(self.run_dir, source_relative, must_exist=True)
        video_relative = f"videos/{measurement_id}.mp4"
        video_path = self.run_dir / video_relative
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if video_path.exists():
            _require(
                video_path.is_file()
                and not video_path.is_symlink()
                and video_path.samefile(source_path),
                "shared-history route video conflicts with the frozen decode",
            )
        else:
            try:
                os.link(source_path, video_path)
            except OSError as error:
                raise RuntimeBridgeError(
                    "cannot hard-link a shared-history video to its route identity"
                ) from error
        aliased_video = {**dict(source_video), "path": video_relative}
        self._verify_video_metadata(aliased_video)
        alias_relative = self._decode_evidence_relative(
            history, measurement_id=measurement_id
        )
        _write_immutable_json(
            self.run_dir / alias_relative,
            {
                **dict(canonical_evidence),
                "video": aliased_video,
            },
        )
        return alias_relative

    def _ensure_decoded(
        self,
        *,
        history: LatentHistory,
        measurement_id: str,
        video: bool,
    ) -> str:
        _require(_MEASUREMENT_RE.fullmatch(measurement_id) is not None,
                 "measurement_id is not portable")
        key = history.manifest_sha256
        frames_relative = f"decoded/{key}.safetensors"
        evidence_path = self.run_dir / "decoded" / f"{key}.json"
        video_relative = f"videos/{measurement_id}.mp4" if video else None
        if evidence_path.exists():
            evidence = _strict_json_file(evidence_path)
            _exact_keys(
                evidence,
                {
                    "schema",
                    "history_manifest_sha256",
                    "latent_chunk_count",
                    "expected_frames",
                    "frames",
                    "video",
                },
                label="decode evidence",
            )
            _require(evidence.get("schema") == DECODE_EVIDENCE_SCHEMA,
                     "decode evidence schema is invalid")
            _require(evidence.get("history_manifest_sha256") == key,
                     "decode evidence is bound to another history")
            _require(
                evidence.get("latent_chunk_count") == history.next_chunk_index
                and evidence.get("expected_frames") == self._frame_count(history),
                "decode evidence history dimensions drifted",
            )
            frames = evidence.get("frames")
            _require(isinstance(frames, Mapping) and frames.get("path") == frames_relative,
                     "decode evidence frame path is invalid")
            if video:
                recorded_video = evidence.get("video")
                if (
                    isinstance(recorded_video, Mapping)
                    and recorded_video.get("path") != video_relative
                ):
                    evidence_path = self.run_dir / self._materialize_shared_history_video_alias(
                        history=history,
                        measurement_id=measurement_id,
                        canonical_evidence=evidence,
                    )
                    evidence = _strict_json_file(evidence_path)
                    frames = evidence.get("frames")
                    recorded_video = evidence.get("video")
                _require(
                    isinstance(recorded_video, Mapping)
                    and recorded_video.get("path") == video_relative,
                    "cached decode has no requested full-video evidence",
                )
                self._verify_video_metadata(recorded_video)
            frame_path = self.run_dir / frames_relative
            if not frame_path.exists():
                _require(not frame_path.is_symlink(), "decoded frame path is a dangling symlink")
                decoded = self.client.decode(
                    history=history,
                    output_frames_path=frames_relative,
                    # An existing video is immutable and already covered by
                    # the evidence.  Re-materialize only the transient tensor.
                    output_video_path=None,
                    fps=24,
                )
                try:
                    self._verify_frames_metadata(
                        decoded.frames,
                        frames_relative=frames_relative,
                        expected_frames=self._frame_count(history),
                    )
                    _require(
                        dict(decoded.frames) == dict(frames),
                        "deterministic re-decode differs from frozen evidence",
                    )
                    _require(
                        decoded.video is None,
                        "transient re-decode unexpectedly emitted a video",
                    )
                except BaseException:
                    try:
                        frame_path.unlink()
                    except OSError:
                        pass
                    raise
            else:
                self._verify_frames_metadata(
                    frames,
                    frames_relative=frames_relative,
                    expected_frames=self._frame_count(history),
                )
            return frames_relative
        _require(not (self.run_dir / frames_relative).exists(),
                 "decoded frames exist without immutable evidence")
        decoded = self.client.decode(
            history=history,
            output_frames_path=frames_relative,
            output_video_path=video_relative,
            fps=24,
        )
        expected_frames = self._frame_count(history)
        self._verify_frames_metadata(
            decoded.frames,
            frames_relative=frames_relative,
            expected_frames=expected_frames,
        )
        if decoded.video is not None:
            self._verify_video_metadata(decoded.video)
        try:
            _write_immutable_json(
                evidence_path,
                {
                    "schema": DECODE_EVIDENCE_SCHEMA,
                    "history_manifest_sha256": key,
                    "latent_chunk_count": history.next_chunk_index,
                    "expected_frames": expected_frames,
                    "frames": dict(decoded.frames),
                    "video": None if decoded.video is None else dict(decoded.video),
                },
            )
        except BaseException:
            try:
                (self.run_dir / frames_relative).unlink()
            except OSError:
                pass
            raise
        return frames_relative

    def _verify_frames_metadata(
        self,
        frames: Mapping[str, Any],
        *,
        frames_relative: str,
        expected_frames: int,
    ) -> None:
        _exact_keys(
            frames,
            {"path", "key", "sha256", "nbytes", "shape", "dtype"},
            label="decoded frame metadata",
        )
        _require(
            frames.get("path") == frames_relative
            and frames.get("key") == "frames"
            and frames.get("shape") == [expected_frames, 3, 480, 832]
            and frames.get("dtype") == "uint8",
            "decoded frame metadata disagrees with the frozen VAE output contract",
        )
        digest = frames.get("sha256")
        size = frames.get("nbytes")
        _require(
            isinstance(digest, str)
            and _SHA256_RE.fullmatch(digest) is not None
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0,
            "decoded frame digest metadata is invalid",
        )
        frame_path = _run_file(self.run_dir, frames_relative, must_exist=True)
        actual_digest, actual_size = _sha256_file(frame_path)
        _require(
            (digest, size) == (actual_digest, actual_size),
            "decoded frame bytes differ from metadata",
        )

    def _verify_video_metadata(self, video: Mapping[str, Any]) -> None:
        _exact_keys(
            video,
            {"path", "sha256", "nbytes", "fps", "frames"},
            label="decoded video metadata",
        )
        relative = _relative_path(video.get("path"), label="decoded video path")
        _require(
            video.get("fps") == 24
            and isinstance(video.get("frames"), int)
            and not isinstance(video.get("frames"), bool)
            and video["frames"] > 0,
            "decoded video metadata is invalid",
        )
        path = _run_file(self.run_dir, relative, must_exist=True)
        digest, size = _sha256_file(path)
        _require(
            video.get("sha256") == digest and video.get("nbytes") == size,
            "decoded video bytes differ from metadata",
        )

    def _write_measurement(
        self,
        *,
        history: LatentHistory,
        measurement_id: str,
        evaluation_kind: str,
        chunk_index: int | None,
        frame_indices: Sequence[int],
        temporal_window: tuple[int, int],
        measurement: QualityMeasurement,
    ) -> None:
        _require(
            _MEASUREMENT_RE.fullmatch(measurement_id) is not None,
            "measurement_id is not portable",
        )
        _require(
            evaluation_kind in {"local", "full"},
            "quality evaluation kind is invalid",
        )
        evidence_relative = f"decoded/{history.manifest_sha256}.json"
        evidence_path = _run_file(self.run_dir, evidence_relative, must_exist=True)
        evidence = _strict_json_file(evidence_path)
        _require(
            evidence.get("schema") == DECODE_EVIDENCE_SCHEMA
            and evidence.get("history_manifest_sha256") == history.manifest_sha256,
            "quality measurement decode evidence is not bound to its history",
        )
        evidence_sha256, _evidence_size = _sha256_file(evidence_path)
        frames = evidence.get("frames")
        _require(isinstance(frames, Mapping), "decode evidence has no frame metadata")
        indices = list(frame_indices)
        _require(
            bool(indices)
            and len(indices) == len(measurement.frame_scores)
            and indices == sorted(set(indices)),
            "quality measurement frame indices/scores are inconsistent",
        )
        state = self.client.model_state
        sidecar = {
            "schema": QUALITY_MEASUREMENT_SCHEMA,
            "measurement_id": measurement_id,
            "history": {
                "manifest_sha256": history.manifest_sha256,
                "head_sha256": history.head_sha256,
                "latent_chunk_count": history.next_chunk_index,
                "latents": [
                    {
                        "chunk_index": segment.chunk_index,
                        "path": segment.latent_path,
                        "sha256": segment.latent_sha256,
                        "nbytes": segment.latent_nbytes,
                        "segment_sha256": segment.segment_sha256,
                    }
                    for segment in history.segments
                ],
            },
            "decode_evidence": {
                "path": evidence_relative,
                "sha256": evidence_sha256,
            },
            "frames": dict(frames),
            "evaluation": {
                "kind": evaluation_kind,
                "chunk_index": chunk_index,
                "decoded_frame_count": self._frame_count(history),
                "frame_indices": indices,
                "temporal_window": {
                    "start": temporal_window[0],
                    "stop": temporal_window[1],
                },
            },
            "scores": {
                "quality_scores": list(measurement.frame_scores),
                "quality_mean": measurement.primary,
                "temporal_consistency": measurement.temporal_consistency,
            },
            "draft_model": {
                "role": "draft",
                "model_id": self.client.model_id,
                "model_artifact_sha256": self.client.model_artifact_sha256,
                "musiq_spaq_sha256": state["musiq_spaq_sha256"],
                "vae_sha256": state["vae_sha256"],
            },
            "metrics": {
                "primary": {
                    "id": PRIMARY_QUALITY_METRIC,
                    "definition": (
                        "MUSIQ-SPAQ raw score divided by 100 on each exact lossless "
                        "decoded RGB frame; aggregate is math.fsum(scores)/len(scores)"
                    ),
                    "range": [0.0, 1.0],
                },
                "temporal_consistency": {
                    "id": TEMPORAL_CONSISTENCY_METRIC,
                    "definition": (
                        "1 - mean absolute uint8 RGB difference between adjacent frames "
                        "in the exact half-open temporal window, divided by 255"
                    ),
                    "range": [0.0, 1.0],
                },
            },
        }
        _write_immutable_json(
            self.run_dir / "measurements" / "quality" / f"{measurement_id}.json",
            sidecar,
        )

    def _discard_decode(self, key: str) -> None:
        """Remove only the large frame tensor; immutable evidence remains."""

        _require(_SHA256_RE.fullmatch(key) is not None, "decode cache key is invalid")
        evidence_path = self.run_dir / "decoded" / f"{key}.json"
        frames_path = self.run_dir / "decoded" / f"{key}.safetensors"
        if evidence_path.exists():
            evidence = _strict_json_file(evidence_path)
            _require(evidence.get("history_manifest_sha256") == key,
                     "cannot discard decode evidence bound to another history")
            frames = evidence.get("frames")
            _require(
                isinstance(frames, Mapping)
                and frames.get("path") == f"decoded/{key}.safetensors",
                "cannot discard malformed decode evidence",
            )
        try:
            frames_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RuntimeBridgeError(
                f"cannot remove transient decode {frames_path}: {error}"
            ) from error

    def _write_vbench_measurement(
        self,
        *,
        history: LatentHistory,
        measurement_id: str,
        measurement: VBenchLongMeasurement,
        decode_evidence_relative: str | None = None,
    ) -> None:
        evidence_relative = (
            self._full_decode_evidence_relative(
                history=history, measurement_id=measurement_id
            )
            if decode_evidence_relative is None
            else decode_evidence_relative
        )
        from .video_only_decode import enabled, evidence_path as video_only_evidence_path, recover, SCHEMA as VIDEO_ONLY_SCHEMA
        video_only = enabled() and evidence_relative == video_only_evidence_path(history, measurement_id)
        if video_only:
            recover(self, history=history, measurement_id=measurement_id)
        _require(
            video_only or evidence_relative
            in {
                self._decode_evidence_relative(history),
                self._decode_evidence_relative(
                    history, measurement_id=measurement_id
                ),
            },
            "VBench-Long decode evidence path is not bound to this measurement",
        )
        evidence_path = _run_file(
            self.run_dir, evidence_relative, must_exist=True
        )
        decode_evidence = _strict_json_file(evidence_path)
        _require(
            decode_evidence.get("schema") == (VIDEO_ONLY_SCHEMA if video_only else DECODE_EVIDENCE_SCHEMA)
            and decode_evidence.get("history_manifest_sha256")
            == history.manifest_sha256,
            "VBench-Long decode evidence is not bound to its history",
        )
        video = decode_evidence.get("video")
        _require(
            isinstance(video, Mapping),
            "VBench-Long full-video measurement has no decoded video",
        )
        self._verify_video_metadata(video)
        official_path = _run_file(
            self.run_dir,
            measurement.evidence_path,
            must_exist=True,
        )
        official_sha256, official_nbytes = _sha256_file(official_path)
        scores = measurement.scores()
        _require(
            tuple(scores) == VBENCH_LONG_DIMENSIONS,
            "VBench-Long scorer returned different dimensions or order",
        )
        sidecar = {
            "schema": VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA,
            "measurement_id": measurement_id,
            "metric_id": self.full_quality_metric_id,
            "history": {
                "manifest_sha256": history.manifest_sha256,
                "head_sha256": history.head_sha256,
                "prompt_id": history.prompt_id,
                "latent_chunk_count": history.next_chunk_index,
            },
            "decode_evidence": {
                "path": evidence_relative,
                "sha256": _sha256_file(evidence_path)[0],
            },
            "video": dict(video),
            "official_vbench_long_evidence": {
                "path": measurement.evidence_path,
                "sha256": official_sha256,
                "nbytes": official_nbytes,
            },
            "dimensions": list(VBENCH_LONG_DIMENSIONS),
            "aggregation": (
                "unweighted_arithmetic_mean_of_six_official_dimension_scores"
            ),
            "scores": {
                "dimension_scores": scores,
                "macro_mean": measurement.macro_mean,
                "motion_smoothness": measurement.motion_smoothness,
            },
        }
        _write_immutable_json(
            self.run_dir
            / "measurements"
            / "vbench-long-quality"
            / f"{measurement_id}.json",
            sidecar,
        )

    def _prepared_full_from_evidence(
        self, *, history: LatentHistory, measurement_id: str
    ) -> PreparedFullQuality:
        """Recover a prepared video without rematerializing its frame tensor."""

        from .video_only_decode import enabled, recover
        if enabled():
            return recover(self, history=history, measurement_id=measurement_id)

        _require(
            _MEASUREMENT_RE.fullmatch(measurement_id) is not None,
            "measurement_id is not portable",
        )
        evidence_relative = self._full_decode_evidence_relative(
            history=history, measurement_id=measurement_id
        )
        evidence_path = _run_file(
            self.run_dir, evidence_relative, must_exist=True
        )
        evidence = _strict_json_file(evidence_path)
        _exact_keys(
            evidence,
            {
                "schema",
                "history_manifest_sha256",
                "latent_chunk_count",
                "expected_frames",
                "frames",
                "video",
            },
            label="prepared decode evidence",
        )
        expected_frames = self._frame_count(history)
        _require(
            evidence.get("schema") == DECODE_EVIDENCE_SCHEMA
            and evidence.get("history_manifest_sha256")
            == history.manifest_sha256
            and evidence.get("latent_chunk_count") == history.next_chunk_index
            and evidence.get("expected_frames") == expected_frames,
            "prepared decode evidence is bound to another history",
        )
        frames = evidence.get("frames")
        _require(isinstance(frames, Mapping), "prepared decode evidence has no frames")
        _exact_keys(
            frames,
            {"path", "key", "sha256", "nbytes", "shape", "dtype"},
            label="prepared decoded frame metadata",
        )
        _require(
            frames.get("path")
            == f"decoded/{history.manifest_sha256}.safetensors"
            and frames.get("key") == "frames"
            and frames.get("shape") == [expected_frames, 3, 480, 832]
            and frames.get("dtype") == "uint8"
            and isinstance(frames.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(frames["sha256"])) is not None
            and isinstance(frames.get("nbytes"), int)
            and not isinstance(frames.get("nbytes"), bool)
            and int(frames["nbytes"]) > 0,
            "prepared decoded frame metadata is invalid",
        )
        video = evidence.get("video")
        _require(isinstance(video, Mapping), "prepared decode evidence has no video")
        _require(
            video.get("path") == f"videos/{measurement_id}.mp4",
            "prepared video is bound to another measurement",
        )
        self._verify_video_metadata(video)
        evidence_sha256, _evidence_nbytes = _sha256_file(evidence_path)
        return PreparedFullQuality(
            measurement_id=measurement_id,
            history_manifest_sha256=history.manifest_sha256,
            history_head_sha256=history.head_sha256,
            prompt_id=history.prompt_id,
            latent_chunk_count=history.next_chunk_index,
            decode_evidence_path=evidence_relative,
            decode_evidence_sha256=evidence_sha256,
            video_path=str(video["path"]),
            video_sha256=str(video["sha256"]),
            video_nbytes=int(video["nbytes"]),
            fps=int(video["fps"]),
            frame_count=int(video["frames"]),
        )

    def prepare_full(
        self, *, history: LatentHistory, measurement_id: str
    ) -> PreparedFullQuality:
        """Decode and seal one full video for scoring after worker shutdown."""

        from .video_only_decode import enabled, prepare
        if enabled():
            return prepare(self, history=history, measurement_id=measurement_id)

        pending_path = (
            self.run_dir
            / "measurements"
            / "vbench-long-pending"
            / f"{measurement_id}.json"
        )
        if pending_path.exists():
            prepared = PreparedFullQuality.from_dict(_strict_json_file(pending_path))
            current = self._prepared_full_from_evidence(
                history=history, measurement_id=measurement_id
            )
            _require(
                prepared == current,
                "prepared full-video manifest differs from current immutable evidence",
            )
            self._discard_decode(history.manifest_sha256)
            return prepared

        evidence_path = self.run_dir / self._decode_evidence_relative(history)
        try:
            if not evidence_path.exists():
                self._ensure_decoded(
                    history=history,
                    measurement_id=measurement_id,
                    video=True,
                )
            elif not (
                self.run_dir
                / self._full_decode_evidence_relative(
                    history=history, measurement_id=measurement_id
                )
            ).exists():
                self._materialize_shared_history_video_alias(
                    history=history,
                    measurement_id=measurement_id,
                    canonical_evidence=_strict_json_file(evidence_path),
                )
            prepared = self._prepared_full_from_evidence(
                history=history, measurement_id=measurement_id
            )
            _write_immutable_json(pending_path, prepared.to_dict())
            return prepared
        finally:
            self._discard_decode(history.manifest_sha256)

    def score_prepared_full(
        self,
        *,
        history: LatentHistory,
        prepared: PreparedFullQuality,
        scorer: OfficialVBenchLongScorer,
    ) -> QualityMeasurement:
        """Score a sealed video without consulting the closed Draft worker."""

        pending_path = (
            self.run_dir
            / "measurements"
            / "vbench-long-pending"
            / f"{prepared.measurement_id}.json"
        )
        frozen = PreparedFullQuality.from_dict(_strict_json_file(pending_path))
        _require(frozen == prepared, "prepared full-video input is not the sealed manifest")
        current = self._prepared_full_from_evidence(
            history=history, measurement_id=prepared.measurement_id
        )
        _require(current == prepared, "prepared full-video bytes drifted before scoring")
        _require(
            Path(scorer.run_dir).resolve(strict=True) == self.run_dir,
            "prepared scorer belongs to another run directory",
        )
        official = scorer.score_video(
            video_path=_run_file(
                self.run_dir, prepared.video_path, must_exist=True
            ),
            prompt=prepared.prompt_id,
            measurement_id=prepared.measurement_id,
        )
        measurement = QualityMeasurement(
            primary=official.macro_mean,
            temporal_consistency=official.motion_smoothness,
        )
        self._write_vbench_measurement(
            history=history,
            measurement_id=prepared.measurement_id,
            measurement=official,
            decode_evidence_relative=prepared.decode_evidence_path,
        )
        return measurement

    def score_prepared_full_batch(
        self,
        *,
        entries: Sequence[tuple[LatentHistory, PreparedFullQuality]],
        scorer: OfficialVBenchLongScorer,
    ) -> dict[str, QualityMeasurement]:
        """Score one GPU lane in one process, retaining route-local bridges."""

        _require(bool(entries), "prepared full-video batch is empty")
        _require(
            Path(scorer.run_dir).resolve(strict=True) == self.run_dir,
            "prepared batch scorer belongs to another run directory",
        )
        requests: list[VBenchLongVideoRequest] = []
        frozen_entries: dict[
            str, tuple[LatentHistory, PreparedFullQuality]
        ] = {}
        for history, prepared in entries:
            measurement_id = prepared.measurement_id
            _require(
                measurement_id not in frozen_entries,
                "prepared full-video batch repeats a measurement ID",
            )
            pending_path = (
                self.run_dir
                / "measurements"
                / "vbench-long-pending"
                / f"{measurement_id}.json"
            )
            frozen = PreparedFullQuality.from_dict(_strict_json_file(pending_path))
            _require(
                frozen == prepared,
                "prepared batch input is not the sealed manifest",
            )
            current = self._prepared_full_from_evidence(
                history=history, measurement_id=measurement_id
            )
            _require(
                current == prepared,
                "prepared full-video bytes drifted before batch scoring",
            )
            video_path = _run_file(
                self.run_dir, prepared.video_path, must_exist=True
            )
            requests.append(
                VBenchLongVideoRequest(
                    video_path=video_path,
                    prompt=prepared.prompt_id,
                    measurement_id=measurement_id,
                )
            )
            frozen_entries[measurement_id] = (history, prepared)

        official = scorer.score_videos(requests)
        _require(
            set(official) == set(frozen_entries),
            "official VBench batch did not exactly cover prepared routes",
        )
        completed: dict[str, QualityMeasurement] = {}
        for measurement_id in sorted(frozen_entries):
            history, prepared = frozen_entries[measurement_id]
            item = official[measurement_id]
            measurement = QualityMeasurement(
                primary=item.macro_mean,
                temporal_consistency=item.motion_smoothness,
            )
            self._write_vbench_measurement(
                history=history,
                measurement_id=measurement_id,
                measurement=item,
                decode_evidence_relative=prepared.decode_evidence_path,
            )
            completed[measurement_id] = measurement
        return completed

    def _retain_local_baseline(self, key: str) -> None:
        prior = self._local_baseline_cache_key
        if prior is not None and prior != key:
            self._discard_decode(prior)
        self._local_baseline_cache_key = key

    def score_local(
        self, *, history: LatentHistory, chunk_index: int, measurement_id: str
    ) -> QualityMeasurement:
        _require(isinstance(chunk_index, int) and not isinstance(chunk_index, bool),
                 "chunk_index must be an integer")
        _require(0 <= chunk_index < history.next_chunk_index, "chunk_index is out of history")
        frames_path = self._ensure_decoded(
            history=history, measurement_id=measurement_id, video=False
        )
        if chunk_index == 0:
            direct_start, direct_stop = 0, 9
            temporal_start = 0
        else:
            direct_start = 12 * chunk_index - 3
            direct_stop = 12 * chunk_index + 9
            temporal_start = direct_start - 1
        _require(direct_stop <= self._frame_count(history),
                 "local quality window exceeds decoded history")
        key = history.manifest_sha256
        try:
            measurement = self.client.score(
                frames_path=frames_path,
                frame_indices=tuple(range(direct_start, direct_stop)),
                temporal_window=(temporal_start, direct_stop),
            )
            self._write_measurement(
                history=history,
                measurement_id=measurement_id,
                evaluation_kind="local",
                chunk_index=chunk_index,
                frame_indices=tuple(range(direct_start, direct_stop)),
                temporal_window=(temporal_start, direct_stop),
                measurement=measurement,
            )
        except BaseException:
            self._discard_decode(key)
            if self._local_baseline_cache_key == key:
                self._local_baseline_cache_key = None
            raise
        # Campaign 2(a) names branch A ``...-draft``.  Retain only the latest
        # branch-A decode until its paired Target branch has been measured;
        # the next candidate replaces it, keeping GPU-host storage bounded.
        if measurement_id.endswith("-draft"):
            self._retain_local_baseline(key)
        elif key != self._local_baseline_cache_key:
            self._discard_decode(key)
        return measurement

    def score_full(
        self, *, history: LatentHistory, measurement_id: str
    ) -> QualityMeasurement:
        use_vbench = self.vbench_scorer is not None
        frames_path = self._ensure_decoded(
            history=history,
            measurement_id=measurement_id,
            video=use_vbench or self.write_full_videos,
        )
        count = self._frame_count(history)
        key = history.manifest_sha256
        try:
            if use_vbench:
                assert self.vbench_scorer is not None
                decode_evidence_relative = self._full_decode_evidence_relative(
                    history=history, measurement_id=measurement_id
                )
                decode_evidence = _strict_json_file(
                    _run_file(
                        self.run_dir,
                        decode_evidence_relative,
                        must_exist=True,
                    )
                )
                video = decode_evidence.get("video")
                _require(
                    isinstance(video, Mapping),
                    "official VBench-Long requires a decoded full video",
                )
                video_relative = _relative_path(
                    video.get("path"), label="VBench-Long decoded video path"
                )
                official = self.vbench_scorer.score_video(
                    video_path=_run_file(
                        self.run_dir, video_relative, must_exist=True
                    ),
                    prompt=history.prompt_id,
                    measurement_id=measurement_id,
                )
                measurement = QualityMeasurement(
                    primary=official.macro_mean,
                    temporal_consistency=official.motion_smoothness,
                )
                self._write_vbench_measurement(
                    history=history,
                    measurement_id=measurement_id,
                    measurement=official,
                    decode_evidence_relative=decode_evidence_relative,
                )
                return measurement
            measurement = self.client.score(
                frames_path=frames_path,
                frame_indices=tuple(range(count)),
                temporal_window=(0, count),
            )
            self._write_measurement(
                history=history,
                measurement_id=measurement_id,
                evaluation_kind="full",
                chunk_index=None,
                frame_indices=tuple(range(count)),
                temporal_window=(0, count),
                measurement=measurement,
            )
            return measurement
        finally:
            self._discard_decode(key)
            if self._local_baseline_cache_key == key:
                self._local_baseline_cache_key = None

    def close(self) -> None:
        if self._local_baseline_cache_key is not None:
            self._discard_decode(self._local_baseline_cache_key)
            self._local_baseline_cache_key = None


def isolate_vbench_scorer_runtime_cache(
    scorer: OfficialVBenchLongScorer, *, scorer_id: str
) -> OfficialVBenchLongScorer:
    """Bind one scorer to a private run-local cache before concurrent use.

    ``OfficialVBenchLongScorer`` deliberately keeps its subprocess environment
    as instance state.  Rebinding that state before the first ``score_video``
    call lets two post-generation scorers share only the immutable asset bundle,
    never writable Matplotlib/Hugging Face/Torch caches.  The dependency
    preflight is repeated after rebinding so each private ``MPLCONFIGDIR`` is
    populated before the fail-closed official launcher blocks all unbound
    subprocesses (including Matplotlib's ambient ``fc-list`` fallback).
    """

    _require(
        isinstance(scorer_id, str)
        and _LABEL_RE.fullmatch(scorer_id) is not None,
        "VBench scorer_id must be a portable lowercase label",
    )
    run_dir = Path(scorer.run_dir).resolve(strict=True)
    _require(run_dir.is_dir(), "VBench scorer run directory is invalid")
    cache_root = run_dir / "tmp" / "vbench-long-runtime-cache" / scorer_id
    cache_root.mkdir(parents=True, exist_ok=True)
    _require(
        cache_root.is_dir() and not cache_root.is_symlink(),
        "isolated VBench runtime cache is invalid",
    )
    cache_environment: dict[str, str] = {}
    for variable, relative in _VBENCH_RUNTIME_CACHE_LAYOUT.items():
        cache = cache_root / relative
        cache.mkdir(parents=True, exist_ok=True)
        _require(
            cache.is_dir() and not cache.is_symlink(),
            f"isolated VBench cache directory is invalid: {cache}",
        )
        cache_environment[variable] = str(cache)

    offline_environment = dict(scorer.offline_environment)
    offline_environment.update(cache_environment)
    environment = dict(scorer.environment)
    environment.update(offline_environment)
    scorer.runtime_cache = cache_root
    scorer.offline_environment = offline_environment
    scorer.environment = environment
    scorer._validate_checkout_and_python()
    return scorer


@dataclass
class CorrectionRuntimeBundle:
    """The two persistent workers and campaign-facing adapters as one lifetime."""

    draft: JsonlWorkerClient
    target: JsonlWorkerClient
    noise_provider: DeterministicNoiseProvider
    evaluator: DraftQualityEvaluator

    @property
    def model_state(self) -> dict[str, Mapping[str, Any]]:
        return {
            "draft": dict(self.draft.model_state),
            "target": dict(self.target.model_state),
        }

    def close(self) -> None:
        errors: list[BaseException] = []
        try:
            self.evaluator.close()
        except BaseException as error:
            errors.append(error)
        # Close the larger Target first; always attempt both workers.
        for client in (self.target, self.draft):
            try:
                client.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeBridgeError(
                "correction runtime close failed: "
                + "; ".join(f"{type(item).__name__}: {item}" for item in errors)
            )

    def __enter__(self) -> CorrectionRuntimeBundle:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _spec_below_repo(path: str | Path, repo_root: Path, *, label: str) -> Path:
    unresolved = Path(path).expanduser()
    _require(not unresolved.is_symlink(), f"{label} may not be a symlink")
    resolved = unresolved.resolve(strict=True)
    try:
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise RuntimeBridgeError(f"{label} must be repository-tracked") from error
    _require(resolved.is_file(), f"{label} must be a regular file")
    return resolved


def _write_backend_config(
    *,
    run_dir: Path,
    role: str,
    spec_path: Path,
    spec_sha256: str,
    verified_artifacts: Sequence[Mapping[str, Any]],
    chunk_count: int = 16,
    conditioning_cache: Mapping[str, Any] | None = None,
    sdvg: Mapping[str, Any] | None = None,
    rolling: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    _require(_SHA256_RE.fullmatch(spec_sha256) is not None,
             f"{role} artifact specification digest is invalid")
    _require(_sha256_file(spec_path)[0] == spec_sha256,
             f"{role} artifact specification digest mismatch")
    path = run_dir / "provenance" / f"{role}-backend-config.json"
    _require(
        isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count in SUPPORTED_CHUNK_COUNTS,
        "backend chunk_count must be one of 16, 30, 60, or 120",
    )
    payload: dict[str, Any] = {
            "schema": BACKEND_CONFIG_SCHEMA,
            "role": role,
            "run_dir": str(run_dir),
            "worker_label": role,
            "device": "cuda:0",
            "artifact_spec_path": str(spec_path),
            "artifact_spec_sha256": spec_sha256,
            "verified_artifacts": [dict(item) for item in verified_artifacts],
    }
    if chunk_count != 16:
        payload["chunk_count"] = chunk_count
    if conditioning_cache is not None:
        payload["conditioning_cache"] = dict(conditioning_cache)
    if sdvg is not None:
        payload["sdvg"] = dict(sdvg)
    if rolling is not None:
        from .rolling_forcing_assets import SPEC_SHA256 as rolling_spec_sha256
        _require(role == "draft" and spec_sha256 == rolling_spec_sha256,
                 "Rolling worker config requires its pinned Draft specification")
        payload["rolling"] = dict(rolling)
    _write_immutable_json(path, payload)
    digest, _size = _sha256_file(path)
    return path, digest


def _draft_worker_factory(model_id: str) -> str:
    factories = {
        "self_forcing_1.3b": "cfexp.correction_runtime:build_draft_backend",
        "longlive_1.3b": "cfexp.longlive_policy_worker:build_backend",
        "rolling_forcing_1.3b": "cfexp.rolling_policy_worker:build_backend",
    }
    _require(model_id in factories, "unsupported Draft model identity")
    return factories[model_id]


def _validate_rolling_launch_config(*, model_id, rolling, run_dir, repo_root, chunk_count):
    _require((rolling is not None) == (model_id == "rolling_forcing_1.3b"),
             "Rolling model requires its explicit native configuration; other models must omit it")
    if rolling is None:
        return
    _require(isinstance(rolling, Mapping) and set(rolling) == {
        "repository_root", "model_root", "musiq_relative_path", "noise_schedules"},
        "invalid Rolling launch configuration")
    _require(Path(rolling["repository_root"]).resolve(strict=True) == repo_root,
             "Rolling launch repository differs from worker repository")
    _require(Path(rolling["model_root"]).resolve(strict=True).is_dir(),
             "Rolling model root is not a directory")
    bindings = rolling["noise_schedules"]
    _require(isinstance(bindings, list) and bool(bindings), "Rolling launch has no noise schedules")
    from .rolling_forcing_noise import verify_schedule
    identities = set()
    for binding in bindings:
        _require(isinstance(binding, Mapping)
                 and Path(binding["run_dir"]).resolve(strict=True) == run_dir
                 and binding["chunk_count"] == chunk_count,
                 "Rolling launch noise run or horizon mismatch")
        identity = (binding["prompt_id"], binding["trajectory_seed"])
        _require(identity not in identities, "duplicate Rolling noise trajectory")
        identities.add(identity)
        verify_schedule(**binding)


def launch_correction_runtime(
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    draft_artifact_spec_path: str | Path,
    draft_artifact_spec_sha256: str,
    draft_verified_artifacts: Sequence[Mapping[str, Any]],
    target_artifact_spec_path: str | Path,
    target_artifact_spec_sha256: str,
    target_verified_artifacts: Sequence[Mapping[str, Any]],
    devices: Sequence[str],
    write_full_videos: bool = False,
    process_factory: Callable[..., ProcessLike] = subprocess.Popen,
    startup_timeout_seconds: float = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_WORKER_REQUEST_TIMEOUT_SECONDS,
    draft_model_id: str = "self_forcing_1.3b",
    chunk_count: int = 16,
    rolling: Mapping[str, Any] | None = None,
    vbench_checkout: str | Path | None = None,
    vbench_python: str | Path | None = None,
    vbench_asset_manifest: str | Path | None = None,
    vbench_asset_spec: str | Path | None = None,
    vbench_metric_id: str | None = None,
    conditioning_prompts: Sequence[Mapping[str, Any]] | None = None,
    shared_precomputed_conditioning: bool = False,
    allow_shared_device: bool = False,
) -> CorrectionRuntimeBundle:
    """Create the complete production bridge on isolated worker processes.

    ``devices`` contains host-visible CUDA identifiers.  Each child sees one
    identifier through ``CUDA_VISIBLE_DEVICES`` and therefore addresses it as
    ``cuda:0``.  The default contract still requires two distinct devices.  A
    duplicated identifier is accepted only behind the explicit shared-device
    gate and only when both workers consume one precomputed UMT5 cache.
    Backend configs and their hashes are persisted below the run directory
    before either model process starts.
    """

    root = Path(run_dir).expanduser().resolve(strict=True)
    _require(root.is_dir(), "run_dir must be an existing directory")
    repository = Path(repo_root).expanduser().resolve(strict=True)
    _require(repository.is_dir(), "repo_root must be an existing directory")
    _require(
        isinstance(devices, Sequence)
        and not isinstance(devices, (str, bytes))
        and len(devices) == 2
        and all(isinstance(item, str) and bool(item.strip()) for item in devices)
        and (len(set(devices)) == 1 if allow_shared_device else len(set(devices)) == 2),
        (
            "shared-device mode requires two references to one CUDA identifier"
            if allow_shared_device
            else "devices must contain exactly two distinct CUDA identifiers"
        ),
    )
    _require(isinstance(shared_precomputed_conditioning, bool),
             "shared_precomputed_conditioning must be boolean")
    _require(isinstance(allow_shared_device, bool), "allow_shared_device must be boolean")
    _require(
        not allow_shared_device or shared_precomputed_conditioning,
        "shared-device mode requires shared precomputed UMT5 conditioning",
    )
    _require(
        (conditioning_prompts is not None) == shared_precomputed_conditioning,
        "conditioning_prompts must be set exactly when shared conditioning is enabled",
    )
    _require(
        isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count in SUPPORTED_CHUNK_COUNTS,
        "chunk_count must be one of 16, 30, 60, or 120",
    )
    vbench_values = (
        vbench_checkout,
        vbench_python,
        vbench_asset_manifest,
        vbench_asset_spec,
        vbench_metric_id,
    )
    _require(
        all(value is None for value in vbench_values)
        or all(value is not None for value in vbench_values),
        "vbench_checkout, vbench_python, vbench_asset_manifest, "
        "vbench_asset_spec, and "
        "vbench_metric_id must be set together",
    )
    _require(
        vbench_metric_id in {None, VBENCH_LONG_PRIMARY_QUALITY_METRIC},
        "vbench_metric_id is not the frozen official VBench-Long metric",
    )
    _validate_rolling_launch_config(model_id=draft_model_id, rolling=rolling,
        run_dir=root, repo_root=repository, chunk_count=chunk_count)
    draft_spec = _spec_below_repo(
        draft_artifact_spec_path, repository, label="Draft artifact specification"
    )
    target_spec = _spec_below_repo(
        target_artifact_spec_path, repository, label="Target artifact specification"
    )
    draft_conditioning: Mapping[str, Any] | None = None
    target_conditioning: Mapping[str, Any] | None = None
    if shared_precomputed_conditioning:
        assert conditioning_prompts is not None
        draft_conditioning, target_conditioning = _shared_conditioning_configs(
            run_dir=root,
            prompts=conditioning_prompts,
            draft_spec=_strict_json_file(draft_spec),
            target_spec=_strict_json_file(target_spec),
            draft_verified_artifacts=draft_verified_artifacts,
            target_verified_artifacts=target_verified_artifacts,
        )
    draft_config, draft_digest = _write_backend_config(
        run_dir=root,
        role="draft",
        spec_path=draft_spec,
        spec_sha256=draft_artifact_spec_sha256,
        verified_artifacts=draft_verified_artifacts,
        chunk_count=chunk_count,
        rolling=rolling,
        conditioning_cache=draft_conditioning,
    )
    target_config, target_digest = _write_backend_config(
        run_dir=root,
        role="target",
        spec_path=target_spec,
        spec_sha256=target_artifact_spec_sha256,
        verified_artifacts=target_verified_artifacts,
        chunk_count=chunk_count,
        conditioning_cache=target_conditioning,
    )
    draft: JsonlWorkerClient | None = None
    target: JsonlWorkerClient | None = None
    try:
        draft = JsonlWorkerClient(
            run_dir=root,
            role="draft",
            backend_factory=_draft_worker_factory(draft_model_id),
            expected_model_id=draft_model_id,
            backend_config_path=draft_config,
            backend_config_sha256=draft_digest,
            visible_device=devices[0],
            process_factory=process_factory,
            startup_timeout_seconds=startup_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            chunk_count=chunk_count,
        )
        target = JsonlWorkerClient(
            run_dir=root,
            role="target",
            backend_factory="cfexp.correction_runtime:build_target_backend",
            backend_config_path=target_config,
            backend_config_sha256=target_digest,
            visible_device=devices[1],
            process_factory=process_factory,
            startup_timeout_seconds=startup_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            chunk_count=chunk_count,
        )
        noise = DeterministicNoiseProvider(run_dir=root)
        vbench_scorer: OfficialVBenchLongScorer | None = None
        if vbench_checkout is not None:
            assert vbench_python is not None
            assert vbench_asset_manifest is not None
            assert vbench_asset_spec is not None
            vbench_scorer = OfficialVBenchLongScorer(
                run_dir=root,
                checkout=vbench_checkout,
                python_executable=vbench_python,
                asset_manifest=vbench_asset_manifest,
                asset_spec=vbench_asset_spec,
                visible_device=devices[0],
            )
        evaluator = DraftQualityEvaluator(
            run_dir=root,
            client=draft,
            write_full_videos=write_full_videos,
            vbench_scorer=vbench_scorer,
            full_quality_metric_id=vbench_metric_id,
        )
        return CorrectionRuntimeBundle(
            draft=draft,
            target=target,
            noise_provider=noise,
            evaluator=evaluator,
        )
    except BaseException:
        for client in (target, draft):
            if client is not None:
                try:
                    client.close()
                except BaseException:
                    pass
        raise


def launch_draft_decoder_worker(
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    draft_artifact_spec_path: str | Path,
    draft_artifact_spec_sha256: str,
    draft_verified_artifacts: Sequence[Mapping[str, Any]],
    visible_device: str,
    process_factory: Callable[..., ProcessLike] = subprocess.Popen,
    startup_timeout_seconds: float = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_WORKER_REQUEST_TIMEOUT_SECONDS,
    draft_model_id: str = "self_forcing_1.3b",
    chunk_count: int = 120,
    rolling: Mapping[str, Any] | None = None,
) -> JsonlWorkerClient:
    """Launch only the audited Draft worker for post-generation VAE decode.

    Main-table Full Target generation owns a single Krea GPU process.  Its
    completed latent histories are decoded only after the generation timer has
    closed; launching this isolated worker avoids loading a second Target or
    changing the established correction-worker protocol.
    """

    root = Path(run_dir).expanduser().resolve(strict=True)
    repository = Path(repo_root).expanduser().resolve(strict=True)
    _require(root.is_dir(), "run_dir must be an existing directory")
    _require(repository.is_dir(), "repo_root must be an existing directory")
    _require(
        isinstance(visible_device, str) and bool(visible_device.strip()),
        "visible_device must be non-empty",
    )
    _require(
        isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count in SUPPORTED_CHUNK_COUNTS,
        "chunk_count must be one of 16, 30, 60, or 120",
    )
    _validate_rolling_launch_config(model_id=draft_model_id, rolling=rolling,
        run_dir=root, repo_root=repository, chunk_count=chunk_count)
    draft_spec = _spec_below_repo(
        draft_artifact_spec_path,
        repository,
        label="Draft artifact specification",
    )
    draft_config, draft_digest = _write_backend_config(
        run_dir=root,
        role="draft",
        spec_path=draft_spec,
        spec_sha256=draft_artifact_spec_sha256,
        verified_artifacts=draft_verified_artifacts,
        chunk_count=chunk_count,
        rolling=rolling,
    )
    return JsonlWorkerClient(
        run_dir=root,
        role="draft",
        backend_factory=_draft_worker_factory(draft_model_id),
        expected_model_id=draft_model_id,
        backend_config_path=draft_config,
        backend_config_sha256=draft_digest,
        visible_device=visible_device,
        process_factory=process_factory,
        startup_timeout_seconds=startup_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        chunk_count=chunk_count,
    )


def launch_sdvg_draft_worker(
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    draft_artifact_spec_path: str | Path,
    draft_artifact_spec_sha256: str,
    draft_verified_artifacts: Sequence[Mapping[str, Any]],
    sdvg_artifact_spec_path: str | Path,
    sdvg_artifact_spec_sha256: str,
    sdvg_verified_artifacts: Sequence[Mapping[str, Any]],
    visible_device: str,
    process_factory: Callable[..., ProcessLike] = subprocess.Popen,
    startup_timeout_seconds: float = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_WORKER_REQUEST_TIMEOUT_SECONDS,
    draft_model_id: str = "self_forcing_1.3b",
    chunk_count: int = 120,
    rolling: Mapping[str, Any] | None = None,
) -> JsonlWorkerClient:
    """Launch the Draft worker with the pinned offline ImageReward gate."""

    root = Path(run_dir).expanduser().resolve(strict=True)
    repository = Path(repo_root).expanduser().resolve(strict=True)
    _require(root.is_dir(), "run_dir must be an existing directory")
    _require(repository.is_dir(), "repo_root must be an existing directory")
    _require(
        isinstance(visible_device, str) and bool(visible_device.strip()),
        "visible_device must be non-empty",
    )
    _require(type(chunk_count) is int and chunk_count in (30, 60, 120),
             "SDVG requires 30, 60, or 120 chunks")
    _validate_rolling_launch_config(model_id=draft_model_id, rolling=rolling,
        run_dir=root, repo_root=repository, chunk_count=chunk_count)
    draft_spec = _spec_below_repo(
        draft_artifact_spec_path,
        repository,
        label="Draft artifact specification",
    )
    sdvg_spec = _spec_below_repo(
        sdvg_artifact_spec_path,
        repository,
        label="SDVG artifact specification",
    )
    _require(
        isinstance(sdvg_artifact_spec_sha256, str)
        and _SHA256_RE.fullmatch(sdvg_artifact_spec_sha256) is not None
        and _sha256_file(sdvg_spec)[0] == sdvg_artifact_spec_sha256,
        "SDVG artifact specification digest mismatch",
    )
    sdvg_document = _strict_json_file(sdvg_spec)
    _require(
        sdvg_document.get("schema") == "main_table_baseline_artifacts_v1"
        and sdvg_document.get("status") == "frozen"
        and sdvg_document.get("method_id") == "sdvg"
        and sdvg_document.get("adapter_module") == "cfexp.sdvg_adapter",
        "SDVG artifact specification identity is invalid",
    )
    adapter_config = sdvg_document.get("adapter_config")
    _require(isinstance(adapter_config, Mapping), "SDVG adapter config is absent")
    _require(adapter_config.get("benchmark_adaptations", {}).get("chunk_count") == chunk_count,
             "SDVG declared horizon differs from worker horizon")
    draft_config, draft_digest = _write_backend_config(
        run_dir=root,
        role="draft",
        spec_path=draft_spec,
        spec_sha256=draft_artifact_spec_sha256,
        verified_artifacts=draft_verified_artifacts,
        chunk_count=chunk_count,
        rolling=rolling,
        sdvg={
            "artifact_spec_path": str(sdvg_spec),
            "artifact_spec_sha256": sdvg_artifact_spec_sha256,
            "adapter_config": dict(adapter_config),
            "verified_artifacts": [dict(item) for item in sdvg_verified_artifacts],
        },
    )
    return JsonlWorkerClient(
        run_dir=root,
        role="draft",
        backend_factory=_draft_worker_factory(draft_model_id),
        expected_model_id=draft_model_id,
        backend_config_path=draft_config,
        backend_config_sha256=draft_digest,
        visible_device=visible_device,
        process_factory=process_factory,
        startup_timeout_seconds=startup_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        chunk_count=chunk_count,
    )


def launch_target_generation_worker(
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    target_artifact_spec_path: str | Path,
    target_artifact_spec_sha256: str,
    target_verified_artifacts: Sequence[Mapping[str, Any]],
    visible_device: str,
    process_factory: Callable[..., ProcessLike] = subprocess.Popen,
    startup_timeout_seconds: float = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_WORKER_REQUEST_TIMEOUT_SECONDS,
    chunk_count: int = 120,
) -> JsonlWorkerClient:
    """Launch only the persistent Target worker for Full-Target generation."""

    root = Path(run_dir).expanduser().resolve(strict=True)
    repository = Path(repo_root).expanduser().resolve(strict=True)
    _require(root.is_dir(), "run_dir must be an existing directory")
    _require(repository.is_dir(), "repo_root must be an existing directory")
    _require(
        isinstance(visible_device, str) and bool(visible_device.strip()),
        "visible_device must be non-empty",
    )
    _require(
        isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count in SUPPORTED_CHUNK_COUNTS,
        "chunk_count must be one of 16, 30, 60, or 120",
    )
    target_spec = _spec_below_repo(
        target_artifact_spec_path,
        repository,
        label="Target artifact specification",
    )
    target_config, target_digest = _write_backend_config(
        run_dir=root,
        role="target",
        spec_path=target_spec,
        spec_sha256=target_artifact_spec_sha256,
        verified_artifacts=target_verified_artifacts,
        chunk_count=chunk_count,
    )
    return JsonlWorkerClient(
        run_dir=root,
        role="target",
        backend_factory="cfexp.correction_runtime:build_target_backend",
        backend_config_path=target_config,
        backend_config_sha256=target_digest,
        visible_device=visible_device,
        process_factory=process_factory,
        startup_timeout_seconds=startup_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        chunk_count=chunk_count,
    )
