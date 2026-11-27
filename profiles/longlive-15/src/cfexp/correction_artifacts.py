"""Frozen Target artifacts and safe runtime materialization for Insight 2.

The correction campaign deliberately keeps the Krea source tree and model
weights outside Git.  This module is the small fail-closed boundary between a
formal campaign and those external inputs:

* a repository-tracked specification pins every file by SHA-256;
* preflight resolves files only below the declared model/repository roots;
* a worker re-verifies size and digest to close the preflight/import gap; and
* only regular files from the pinned source archive and wheel are materialized
  below ``CFEXP_RUN_DIR/tmp``.

No ML package is imported here.  Model construction remains in
``cfexp.krea_chunk_backend`` and the main inference framework is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Mapping, Sequence

from .krea_chunk_backend import (
    KREA_CHECKPOINT_REVISION,
    KREA_CHECKPOINT_SHA256,
    KREA_SOURCE_COMMIT,
    WAN14B_CONFIG_REVISION,
    WAN14B_CONFIG_SHA256,
)
from .self_forcing_frame_adapter import (
    AdapterError,
    _safe_extract_tar,
    _safe_extract_wheel,
)


TARGET_ARTIFACT_SPEC_SCHEMA = "insight2_target_artifacts_v1"
TARGET_BACKEND_CONFIG_SCHEMA = "krea_chunk_backend_config_v1"
TARGET_BACKEND_MODULE = "cfexp.krea_chunk_backend"
KREA_SOURCE_SUBDIRECTORY = (
    "realtime-video-acf4b7b4e1a3049a0e04782e751d12203334038d"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WORKER_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_ARTIFACT_ROLES = {
    "official_krea_source_archive": "official_source_archive",
    "flash_attn_wheel": "binary_runtime_wheel",
    "krea_14b_checkpoint": "target_generation_checkpoint",
    "wan14b_config": "target_architecture_config",
    "wan_t5_checkpoint": "text_encoder_checkpoint",
    "umt5_special_tokens_map": "tokenizer_file",
    "umt5_spiece_model": "tokenizer_file",
    "umt5_tokenizer_json": "tokenizer_file",
    "umt5_tokenizer_config": "tokenizer_file",
}

_EXPECTED_RUNTIME_FILES = [
    {
        "artifact": "wan_t5_checkpoint",
        "destination": (
            "wan_models/Wan2.1-T2V-1.3B/"
            "models_t5_umt5-xxl-enc-bf16.pth"
        ),
    },
    {
        "artifact": "umt5_special_tokens_map",
        "destination": (
            "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/"
            "special_tokens_map.json"
        ),
    },
    {
        "artifact": "umt5_spiece_model",
        "destination": (
            "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"
        ),
    },
    {
        "artifact": "umt5_tokenizer_json",
        "destination": (
            "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"
        ),
    },
    {
        "artifact": "umt5_tokenizer_config",
        "destination": (
            "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/"
            "tokenizer_config.json"
        ),
    },
]

_EXPECTED_GENERATION = {
    "model_id": "krea-realtime-video-14b",
    "width": 832,
    "height": 480,
    "dtype": "bfloat16",
    "denoising_steps": 4,
    "chunk_count": 16,
    "latent_frames_per_chunk": 3,
    "latent_shape_per_chunk": [1, 3, 16, 60, 104],
    "keep_first_frame": True,
    "kv_cache_num_frames": 3,
    "history_transport": "latent_history_only",
    "cache_rebuild": "fresh_target_session_from_full_latent_prefix",
    "checkpoint_loading": "strict_assign_into_meta_wan14b_scaffold",
}


class TargetArtifactError(AdapterError):
    """Raised when the Krea Target input contract cannot be proven."""


@dataclass(frozen=True)
class TargetRuntimePaths:
    """Materialized, run-confined source and Python wheel roots."""

    source_root: Path
    python_site: Path
    scratch_root: Path


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TargetArtifactError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                TargetArtifactError(
                    f"{label} contains invalid JSON constant {item!r}"
                )
            ),
        )
    except TargetArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TargetArtifactError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TargetArtifactError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_artifact_spec_sha256(path: str | Path) -> str:
    """Return the digest workers bind to before reading a target spec."""

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise TargetArtifactError("target artifact specification may not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file():
        raise TargetArtifactError("target artifact specification must be a regular file")
    return _sha256_file(resolved)


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise TargetArtifactError(
            f"{label} keys differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TargetArtifactError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or path == Path(".")
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise TargetArtifactError(f"{label} must be a safe relative path")
    return path


def _inside(root: Path, candidate: Path, label: str, *, strict: bool) -> Path:
    try:
        resolved_root = root.expanduser().resolve(strict=True)
        resolved = candidate.expanduser().resolve(strict=strict)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise TargetArtifactError(
            f"{label} resolves outside the declared root {root}: {candidate}"
        ) from error
    return resolved


def _require_regular_non_symlink(path: Path, label: str) -> os.stat_result:
    if path.is_symlink():
        raise TargetArtifactError(f"{label} may not be a symlink: {path}")
    try:
        metadata = path.stat()
    except OSError as error:
        raise TargetArtifactError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise TargetArtifactError(f"{label} must be a regular file: {path}")
    return metadata


def validate_target_backend_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact runtime contract consumed by ``KreaChunkBackend``."""

    if not isinstance(config, Mapping):
        raise TargetArtifactError("backend_config must be an object")
    _strict_keys(
        config,
        {
            "schema",
            "official_source_commit",
            "source_archive_artifact",
            "source_archive_subdirectory",
            "checkpoint_revision",
            "checkpoint_artifact",
            "wan14b_config_revision",
            "wan14b_config_artifact",
            "text_encoder_artifact",
            "python_wheels",
            "runtime_files",
            "generation",
        },
        "backend_config",
    )
    expected_scalars = {
        "schema": TARGET_BACKEND_CONFIG_SCHEMA,
        "official_source_commit": KREA_SOURCE_COMMIT,
        "source_archive_artifact": "official_krea_source_archive",
        "source_archive_subdirectory": KREA_SOURCE_SUBDIRECTORY,
        "checkpoint_revision": KREA_CHECKPOINT_REVISION,
        "checkpoint_artifact": "krea_14b_checkpoint",
        "wan14b_config_revision": WAN14B_CONFIG_REVISION,
        "wan14b_config_artifact": "wan14b_config",
        "text_encoder_artifact": "wan_t5_checkpoint",
    }
    for field, expected in expected_scalars.items():
        if config[field] != expected:
            raise TargetArtifactError(
                f"backend_config.{field} must equal {expected!r}"
            )
    _safe_relative(
        config["source_archive_subdirectory"],
        "backend_config.source_archive_subdirectory",
    )
    if config["python_wheels"] != [
        {
            "artifact": "flash_attn_wheel",
            "module": "flash_attn",
            "version": "2.8.3",
        }
    ]:
        raise TargetArtifactError(
            "backend_config.python_wheels must pin flash_attn 2.8.3 exactly"
        )
    if config["runtime_files"] != _EXPECTED_RUNTIME_FILES:
        raise TargetArtifactError(
            "backend_config.runtime_files must materialize the frozen T5/tokenizer set"
        )
    if config["generation"] != _EXPECTED_GENERATION:
        raise TargetArtifactError(
            "backend_config.generation differs from the frozen Target contract"
        )
    for index, item in enumerate(config["runtime_files"]):
        _safe_relative(item["destination"], f"runtime_files[{index}].destination")
    return dict(config)


def validate_target_artifact_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable Krea contract without touching model files."""

    if not isinstance(spec, Mapping):
        raise TargetArtifactError("target artifact specification must be an object")
    _strict_keys(
        spec,
        {"schema", "status", "backend_module", "backend_config", "artifacts"},
        "target artifact specification",
    )
    if spec["schema"] != TARGET_ARTIFACT_SPEC_SCHEMA:
        raise TargetArtifactError(
            f"target schema must be {TARGET_ARTIFACT_SPEC_SCHEMA!r}"
        )
    if spec["status"] != "frozen":
        raise TargetArtifactError("target artifact specification is not frozen")
    if spec["backend_module"] != TARGET_BACKEND_MODULE:
        raise TargetArtifactError(
            f"target backend_module must be {TARGET_BACKEND_MODULE!r}"
        )

    validate_target_backend_config(spec["backend_config"])

    artifacts = spec["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(_ARTIFACT_ROLES):
        raise TargetArtifactError(
            "artifacts must contain exactly the frozen Krea runtime inputs"
        )
    names: set[str] = set()
    paths: set[tuple[str, str]] = set()
    for index, item in enumerate(artifacts):
        label = f"artifacts[{index}]"
        if not isinstance(item, Mapping):
            raise TargetArtifactError(f"{label} must be an object")
        _strict_keys(item, {"name", "root", "path", "sha256", "role"}, label)
        name = item["name"]
        if not isinstance(name, str) or name not in _ARTIFACT_ROLES or name in names:
            raise TargetArtifactError(f"{label}.name is unknown or duplicated")
        if item["role"] != _ARTIFACT_ROLES[name]:
            raise TargetArtifactError(
                f"{label}.role must equal {_ARTIFACT_ROLES[name]!r}"
            )
        if item["root"] not in {"model", "repository"}:
            raise TargetArtifactError(f"{label}.root is invalid")
        relative = _safe_relative(item["path"], f"{label}.path")
        key = (str(item["root"]), relative.as_posix())
        if key in paths:
            raise TargetArtifactError(f"{label}.path duplicates another artifact")
        if not isinstance(item["sha256"], str) or not _SHA256_RE.fullmatch(
            item["sha256"]
        ):
            raise TargetArtifactError(f"{label}.sha256 must be lowercase 64-hex")
        names.add(name)
        paths.add(key)
    if names != set(_ARTIFACT_ROLES):
        raise TargetArtifactError("artifact names differ from the frozen Target set")

    by_name = {str(item["name"]): item for item in artifacts}
    if by_name["krea_14b_checkpoint"]["sha256"] != KREA_CHECKPOINT_SHA256:
        raise TargetArtifactError("Krea checkpoint digest differs from the audited pin")
    if by_name["wan14b_config"]["sha256"] != WAN14B_CONFIG_SHA256:
        raise TargetArtifactError("Wan 14B config digest differs from the audited pin")
    return dict(spec)


def load_target_artifact_spec(path: str | Path) -> dict[str, Any]:
    """Load a duplicate-key-checked target spec and validate its frozen shape."""

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise TargetArtifactError("target artifact specification may not be a symlink")
    resolved = unresolved.resolve(strict=True)
    _require_regular_non_symlink(resolved, "target artifact specification")
    return validate_target_artifact_spec(
        _load_json(resolved, "target artifact specification")
    )


def verify_target_artifact_spec(
    path: str | Path,
    *,
    model_root: str | Path,
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Hash every frozen Target file before a worker is started.

    The return shape intentionally mirrors the existing Self-Forcing campaign
    preflight: validated spec, JSON-serializable verified records, backend
    module name.
    """

    spec = load_target_artifact_spec(path)
    roots = {
        "model": Path(model_root).expanduser().resolve(strict=True),
        "repository": Path(repo_root).expanduser().resolve(strict=True),
    }
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(spec["artifacts"]):
        label = f"artifacts[{index}]"
        root_name = str(item["root"])
        root = roots[root_name]
        relative = _safe_relative(item["path"], f"{label}.path")
        unresolved = root / relative
        if unresolved.is_symlink():
            raise TargetArtifactError(f"{label}.path may not be a symlink")
        resolved = _inside(root, unresolved, f"{label}.path", strict=True)
        metadata = _require_regular_non_symlink(resolved, f"artifact {item['name']!r}")
        actual = _sha256_file(resolved)
        if actual != item["sha256"]:
            raise TargetArtifactError(
                f"artifact hash mismatch for {item['name']!r}: "
                f"expected {item['sha256']}, got {actual}"
            )
        verified.append(
            {
                "name": item["name"],
                "role": item["role"],
                "root": root_name,
                "requested_path": relative.as_posix(),
                "resolved_path": str(resolved),
                "size": metadata.st_size,
                "sha256": actual,
            }
        )
    return spec, verified, TARGET_BACKEND_MODULE


def reverify_target_artifacts(
    verified: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Path], dict[str, str]]:
    """Re-hash campaign-preflight records immediately before model import."""

    if not isinstance(verified, Sequence) or isinstance(verified, (str, bytes)):
        raise TargetArtifactError("verified Target artifacts must be a sequence")
    if len(verified) != len(_ARTIFACT_ROLES):
        raise TargetArtifactError("verified Target artifact count is invalid")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for index, item in enumerate(verified):
        label = f"verified_artifacts[{index}]"
        if not isinstance(item, Mapping):
            raise TargetArtifactError(f"{label} must be an object")
        _strict_keys(
            item,
            {
                "name",
                "role",
                "root",
                "requested_path",
                "resolved_path",
                "size",
                "sha256",
            },
            label,
        )
        name = item["name"]
        if not isinstance(name, str) or name not in _ARTIFACT_ROLES or name in paths:
            raise TargetArtifactError(f"{label}.name is unknown or duplicated")
        if item["role"] != _ARTIFACT_ROLES[name]:
            raise TargetArtifactError(f"{label}.role changed after preflight")
        if item["root"] not in {"model", "repository"}:
            raise TargetArtifactError(f"{label}.root is invalid")
        _safe_relative(item["requested_path"], f"{label}.requested_path")
        if not isinstance(item["resolved_path"], str) or not item["resolved_path"]:
            raise TargetArtifactError(f"{label}.resolved_path is invalid")
        unresolved = Path(item["resolved_path"]).expanduser()
        if unresolved.is_symlink():
            raise TargetArtifactError(f"{label} may not resolve through a final symlink")
        path = unresolved.resolve(strict=True)
        metadata = _require_regular_non_symlink(path, label)
        expected_size = item["size"]
        expected_digest = item["sha256"]
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise TargetArtifactError(f"{label}.size is invalid")
        if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(
            expected_digest
        ):
            raise TargetArtifactError(f"{label}.sha256 is invalid")
        if metadata.st_size != expected_size:
            raise TargetArtifactError(
                f"artifact size changed after Target preflight: {name}"
            )
        actual = _sha256_file(path)
        if actual != expected_digest:
            raise TargetArtifactError(
                f"artifact hash changed after Target preflight: {name}"
            )
        paths[name] = path
        hashes[name] = actual
    if set(paths) != set(_ARTIFACT_ROLES):
        raise TargetArtifactError("verified Target artifact names are incomplete")
    return paths, hashes


def verify_bound_target_spec(
    path: str | Path, expected_sha256: str
) -> dict[str, Any]:
    """Load a worker's repository spec only when its preflight digest matches."""

    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise TargetArtifactError("expected target artifact-spec digest is invalid")
    if target_artifact_spec_sha256(path) != expected_sha256:
        raise TargetArtifactError(
            "target artifact specification changed after campaign preflight"
        )
    return load_target_artifact_spec(path)


def materialize_target_runtime(
    *,
    run_dir: str | Path,
    worker_label: str,
    artifacts: Mapping[str, Path],
    config: Mapping[str, Any],
) -> TargetRuntimePaths:
    """Safely materialize the pinned Krea source tree under one run directory."""

    if not isinstance(worker_label, str) or not _WORKER_LABEL_RE.fullmatch(
        worker_label
    ):
        raise TargetArtifactError("worker_label must be a safe lowercase identifier")
    root = Path(run_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise TargetArtifactError("run_dir must be an existing directory")
    config = validate_target_backend_config(config)
    scratch = _inside(
        root,
        root / "tmp" / f"krea-target-{worker_label}",
        "Target scratch",
        strict=False,
    )
    if scratch.exists() or scratch.is_symlink():
        raise TargetArtifactError(f"Target scratch path already exists: {scratch}")

    required_names = set(_ARTIFACT_ROLES)
    if set(artifacts) != required_names:
        raise TargetArtifactError("materializer received the wrong Target artifact set")
    archive = artifacts.get("official_krea_source_archive")
    wheel = artifacts.get("flash_attn_wheel")
    if not isinstance(archive, Path) or not isinstance(wheel, Path):
        raise TargetArtifactError("materializer artifact paths must be pathlib.Path values")

    try:
        _safe_extract_tar(archive, scratch)
    except AdapterError as error:
        raise TargetArtifactError(str(error)) from error
    source_relative = _safe_relative(
        config.get("source_archive_subdirectory"),
        "backend_config.source_archive_subdirectory",
    )
    source_root = _inside(
        scratch, scratch / source_relative, "official Krea source root", strict=True
    )
    required_source_files = (
        source_root / "release_server.py",
        source_root / "configs" / "self_forcing_server_14b.yaml",
        source_root / "pipeline" / "causal_inference.py",
        source_root / "utils" / "wan_wrapper.py",
        source_root / "wan" / "modules" / "causal_model.py",
    )
    for path in required_source_files:
        _require_regular_non_symlink(path, "required Krea source interface")

    python_site = source_root / ".cfexp-site-packages"
    try:
        _safe_extract_wheel(wheel, python_site)
    except AdapterError as error:
        raise TargetArtifactError(str(error)) from error
    runtime_files = config.get("runtime_files")
    if runtime_files != _EXPECTED_RUNTIME_FILES:
        raise TargetArtifactError("materializer requires the frozen runtime-file map")
    for index, mapping in enumerate(runtime_files):
        artifact_name = mapping["artifact"]
        source = artifacts[artifact_name]
        _require_regular_non_symlink(source, f"runtime artifact {artifact_name!r}")
        relative = _safe_relative(
            mapping["destination"], f"runtime_files[{index}].destination"
        )
        destination = _inside(
            source_root,
            source_root / relative,
            "Target runtime destination",
            strict=False,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise TargetArtifactError(
                f"Target runtime destination already exists: {relative.as_posix()}"
            )
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if _sha256_file(destination) != _sha256_file(source):
            raise TargetArtifactError(
                f"materialized Target runtime file changed: {relative.as_posix()}"
            )
    return TargetRuntimePaths(
        source_root=source_root,
        python_site=python_site,
        scratch_root=scratch,
    )
