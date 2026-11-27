"""Experiment manifest loading, normalization, hashing, and validation.

The module intentionally depends only on the Python standard library.  Loading
and validation are separate operations: a planning command can load a manifest
containing unresolved environment references, show the resulting plan, and
then call :func:`validate_manifest` before execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit


API_VERSION = "cfexp/v1"
KIND = "Experiment"

_VARIABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IMAGE_RE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_HF_URI_RE = re.compile(r"^(?:hf://|https://huggingface\.co/)")
_ENV_REFERENCE_RE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")
_PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_RCLONE_REMOTE_RE = re.compile(r"^[A-Za-z0-9._-]+:.*")
_SECRET_LITERAL_RE = re.compile(
    r"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)
_RESERVED_COMMAND_ENV_NAMES = frozenset(
    {
        "CFEXP_ATTEMPT_ID",
        "CFEXP_CACHE_DIR",
        "CFEXP_MODEL_ROOT",
        "CFEXP_RUN_DIR",
        "CFEXP_UV_BIN",
        "CUDA_VISIBLE_DEVICES",
        "HF_ENDPOINT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "HF_XET_CACHE",
        "MODEL_ROOT",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "TMPDIR",
        "TORCH_HOME",
        "UV_CACHE_DIR",
        "UV_PROJECT_ENVIRONMENT",
    }
)


class ManifestError(ValueError):
    """Raised when a manifest cannot be parsed or violates the v1 contract."""


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ManifestError(f"invalid JSON numeric constant: {value}")


def _expand_allowed_environment(data: dict[str, Any]) -> dict[str, Any]:
    """Expand only the non-secret artifact destination.

    Expanding arbitrary manifest strings is unsafe: a token referenced from
    ``argv`` or ``command.env`` would then be persisted in the resolved
    manifest and uploaded. Secrets are injected exclusively by name through
    ``command.secret_env``.
    """

    expanded = copy.deepcopy(data)
    try:
        remote = expanded["spec"]["storage"]["remote"]
    except (KeyError, TypeError):
        return expanded
    if isinstance(remote, str):
        expanded["spec"]["storage"]["remote"] = os.path.expandvars(remote)
    return expanded


def normalize_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical, non-mutating representation of *data*.

    Normalization fills defaults only for optional fields and canonicalizes
    order-insensitive lists and hexadecimal digests.  It does not make an
    invalid manifest valid; callers should use :func:`validate_manifest` for
    strict checking.
    """

    if not isinstance(data, Mapping):
        raise ManifestError("$: expected a JSON object")

    normalized = copy.deepcopy(dict(data))

    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata.setdefault("tags", [])
        tags = metadata.get("tags")
        if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
            tags.sort()

    spec = normalized.get("spec")
    if not isinstance(spec, dict):
        return normalized

    spec.setdefault("inputs", [])
    spec.setdefault("parameters", {})

    code = spec.get("code")
    if isinstance(code, dict):
        revision = code.get("revision")
        if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            code["revision"] = revision.lower()

    runtime = spec.get("runtime")
    if isinstance(runtime, dict):
        image = runtime.get("image")
        if isinstance(image, str) and "@sha256:" in image.lower():
            prefix, digest = re.split(r"@sha256:", image, maxsplit=1, flags=re.IGNORECASE)
            runtime["image"] = f"{prefix}@sha256:{digest.lower()}"

    command = spec.get("command")
    if isinstance(command, dict):
        command.setdefault("cwd", ".")
        command.setdefault("env", {})
        command.setdefault("secret_env", [])
        secret_env = command.get("secret_env")
        if isinstance(secret_env, list) and all(isinstance(name, str) for name in secret_env):
            secret_env.sort()

    inputs = spec.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            if not isinstance(item, dict):
                continue
            digest = item.get("sha256")
            if isinstance(digest, str):
                item["sha256"] = digest.lower()
            revision = item.get("revision")
            if isinstance(revision, str):
                item["revision"] = revision.lower()

    outputs = spec.get("outputs")
    if isinstance(outputs, dict):
        outputs.setdefault("exclude", [])

    return normalized


def load_manifest(
    path: str | os.PathLike[str], *, expand_environment: bool = True
) -> dict[str, Any]:
    """Load and normalize a JSON experiment manifest.

    This function performs syntactic JSON loading but intentionally does not
    call :func:`validate_manifest`.  Keeping those operations separate lets a
    planner display unresolved ``$VARS`` and lets callers report validation
    errors in their own command context.
    """

    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc

    try:
        data = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_invalid_json_constant,
        )
    except ManifestError:
        raise
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"invalid JSON in {manifest_path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ManifestError("$: expected a JSON object")
    if expand_environment:
        data = _expand_allowed_environment(data)
    return normalize_manifest(data)


def canonical_json(data: Any) -> str:
    """Serialize JSON data deterministically, without insignificant spacing."""

    _validate_json_value(data, "$")
    try:
        return json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # Defensive; checked above as well.
        raise ManifestError(f"cannot encode canonical JSON: {exc}") from exc


def manifest_hash(data: Mapping[str, Any]) -> str:
    """Return the lowercase SHA-256 hex digest of the normalized manifest."""

    payload = canonical_json(normalize_manifest(data)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# The artifact and documentation call this value ``spec_hash``.  Keep the
# descriptive public alias while retaining ``manifest_hash`` as the stable API.
spec_hash = manifest_hash


def _fail(path: str, message: str) -> None:
    raise ManifestError(f"{path}: {message}")


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "non-finite numbers are not valid manifest values")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    _fail(path, f"value of type {type(value).__name__} is not JSON-compatible")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    return value


def _required(obj: dict[str, Any], keys: set[str], path: str) -> None:
    missing = sorted(keys - obj.keys())
    if missing:
        _fail(path, f"missing required field(s): {', '.join(missing)}")


def _keys(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(obj.keys() - allowed)
    if unknown:
        _fail(path, f"unknown field(s): {', '.join(unknown)}")


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    if nonempty and not value.strip():
        _fail(path, "must not be empty")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer")
    if value < minimum:
        _fail(path, f"must be at least {minimum}")
    return value


def _number(value: Any, path: str, *, exclusive_minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a number")
    if not math.isfinite(value) or value <= exclusive_minimum:
        _fail(path, f"must be greater than {exclusive_minimum:g}")
    return float(value)


def _safe_relative_path(value: Any, path: str) -> str:
    text = _string(value, path)
    if "\x00" in text:
        _fail(path, "must not contain a NUL byte")
    if (
        text.startswith("~")
        or text.startswith("\\\\")
        or PurePosixPath(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
    ):
        _fail(path, "must be a relative path")
    if text == "~" or text.startswith(("~/", "~\\")):
        _fail(path, "must not use a home-directory shorthand")
    # Treat both separators as path boundaries so manifests remain portable.
    if ".." in re.split(r"[\\/]", text):
        _fail(path, "must not contain a '..' path segment")
    return text


def _string_list(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    if nonempty and not value:
        _fail(path, "must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_string(item, f"{path}[{index}]"))
    return result


def _unique(values: list[str], path: str) -> None:
    if len(values) != len(set(values)):
        _fail(path, "must not contain duplicates")


def _validate_metadata(value: Any) -> None:
    path = "$.metadata"
    obj = _object(value, path)
    _required(obj, {"name"}, path)
    _keys(obj, {"name", "tags"}, path)
    _string(obj["name"], f"{path}.name")
    if "tags" in obj:
        tags = _string_list(obj["tags"], f"{path}.tags")
        _unique(tags, f"{path}.tags")


def _validate_code(value: Any) -> None:
    path = "$.spec.code"
    obj = _object(value, path)
    _required(obj, {"revision", "require_clean"}, path)
    _keys(obj, {"revision", "require_clean"}, path)
    _string(obj["revision"], f"{path}.revision")
    _boolean(obj["require_clean"], f"{path}.require_clean")


def _validate_runtime(value: Any) -> None:
    path = "$.spec.runtime"
    obj = _object(value, path)
    _required(obj, {"backend"}, path)
    _keys(obj, {"backend", "python", "lockfile", "image", "engine"}, path)
    backend = _string(obj["backend"], f"{path}.backend")
    if backend not in {"uv", "image"}:
        _fail(f"{path}.backend", "must be 'uv' or 'image'")

    if "python" in obj:
        _string(obj["python"], f"{path}.python")
    if "lockfile" in obj:
        lockfile = _safe_relative_path(obj["lockfile"], f"{path}.lockfile")
        if lockfile != "uv.lock":
            _fail(f"{path}.lockfile", "v1 supports only the repository-root uv.lock")
    if "engine" in obj:
        engine = _string(obj["engine"], f"{path}.engine")
        if engine != "docker":
            _fail(f"{path}.engine", "v1 supports only 'docker'")

    if backend == "uv" and "lockfile" not in obj:
        _fail(path, "backend 'uv' requires lockfile")
    if backend == "image" and "image" not in obj:
        _fail(path, "backend 'image' requires image")
    if "image" in obj:
        image = _string(obj["image"], f"{path}.image")
        if not _IMAGE_RE.fullmatch(image):
            _fail(
                f"{path}.image",
                "must be pinned by an @sha256: digest containing 64 hex digits",
            )


def _validate_command(value: Any) -> None:
    path = "$.spec.command"
    obj = _object(value, path)
    _required(obj, {"argv"}, path)
    _keys(obj, {"argv", "cwd", "env", "secret_env"}, path)
    argv = _string_list(obj["argv"], f"{path}.argv", nonempty=True)
    for index, argument in enumerate(argv):
        if _ENV_REFERENCE_RE.search(argument):
            _fail(
                f"{path}.argv[{index}]",
                "must not reference environment variables; use secret_env for secrets",
            )
        if _SECRET_LITERAL_RE.search(argument):
            _fail(
                f"{path}.argv[{index}]",
                "appears to contain a literal access token; use secret_env",
            )
    if "cwd" in obj:
        _safe_relative_path(obj["cwd"], f"{path}.cwd")

    if "env" in obj:
        env = _object(obj["env"], f"{path}.env")
        for name, env_value in env.items():
            if not _VARIABLE_NAME_RE.fullmatch(name):
                _fail(f"{path}.env", f"invalid environment variable name: {name!r}")
            if name in _RESERVED_COMMAND_ENV_NAMES:
                _fail(
                    f"{path}.env.{name}",
                    "is managed by cfexp and may not be set by the manifest",
                )
            text = _string(env_value, f"{path}.env.{name}", nonempty=False)
            if _ENV_REFERENCE_RE.search(text):
                _fail(
                    f"{path}.env.{name}",
                    "must be a literal value; declare secret variable names in secret_env",
                )
            if _SECRET_LITERAL_RE.search(text):
                _fail(
                    f"{path}.env.{name}",
                    "appears to contain a literal access token; use secret_env",
                )

    if "secret_env" in obj:
        secret_env = _string_list(obj["secret_env"], f"{path}.secret_env")
        _unique(secret_env, f"{path}.secret_env")
        for index, name in enumerate(secret_env):
            if not _VARIABLE_NAME_RE.fullmatch(name):
                _fail(
                    f"{path}.secret_env[{index}]",
                    "must contain only an environment variable name, not a value",
                )
            if name in _RESERVED_COMMAND_ENV_NAMES:
                _fail(
                    f"{path}.secret_env[{index}]",
                    "is managed by cfexp and may not be injected into experiment code",
                )
        overlap = sorted(set(secret_env) & set(obj.get("env", {})))
        if overlap:
            _fail(f"{path}.secret_env", f"must not overlap env: {', '.join(overlap)}")


def _validate_resources(value: Any) -> tuple[int, int]:
    path = "$.spec.resources"
    obj = _object(value, path)
    _required(obj, {"gpus", "expected_duration_minutes"}, path)
    _keys(
        obj,
        {"gpus", "min_gpu_memory_gb", "cpu", "memory_gb", "expected_duration_minutes"},
        path,
    )
    gpus = _integer(obj["gpus"], f"{path}.gpus", minimum=0)
    duration = _integer(
        obj["expected_duration_minutes"],
        f"{path}.expected_duration_minutes",
        minimum=1,
    )
    if "min_gpu_memory_gb" in obj:
        _number(obj["min_gpu_memory_gb"], f"{path}.min_gpu_memory_gb")
    if "cpu" in obj:
        _integer(obj["cpu"], f"{path}.cpu", minimum=1)
    if "memory_gb" in obj:
        _number(obj["memory_gb"], f"{path}.memory_gb")
    return gpus, duration


def _validate_inputs(value: Any) -> None:
    path = "$.spec.inputs"
    if not isinstance(value, list):
        _fail(path, "expected an array")
    names: list[str] = []
    targets: list[str] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        obj = _object(item, item_path)
        _required(obj, {"name", "uri", "target"}, item_path)
        _keys(obj, {"name", "uri", "target", "sha256", "revision"}, item_path)
        name = _string(obj["name"], f"{item_path}.name")
        uri = _string(obj["uri"], f"{item_path}.uri")
        if _ENV_REFERENCE_RE.search(uri):
            _fail(f"{item_path}.uri", "must not contain environment-variable references")
        target = _safe_relative_path(obj["target"], f"{item_path}.target")
        names.append(name)
        targets.append(target)

        has_sha256 = "sha256" in obj
        has_revision = "revision" in obj
        if has_sha256 == has_revision:
            _fail(item_path, "must contain exactly one of sha256 or revision")
        if has_sha256:
            digest = _string(obj["sha256"], f"{item_path}.sha256")
            if not _SHA256_RE.fullmatch(digest):
                _fail(f"{item_path}.sha256", "must contain exactly 64 hex digits")
        else:
            revision = _string(obj["revision"], f"{item_path}.revision")
            if not _HF_URI_RE.match(uri):
                _fail(item_path, "revision pins are valid only for Hugging Face URIs")
            if not _HF_REVISION_RE.fullmatch(revision):
                _fail(
                    f"{item_path}.revision",
                    "must be a full 40-hex-character Hugging Face revision",
                )
    _unique(names, path + ".name")
    _unique(targets, path + ".target")


def _validate_outputs(value: Any) -> None:
    path = "$.spec.outputs"
    obj = _object(value, path)
    _required(obj, {"include"}, path)
    _keys(obj, {"include", "exclude"}, path)
    include = _string_list(obj["include"], f"{path}.include", nonempty=True)
    for index, pattern in enumerate(include):
        _safe_relative_path(pattern, f"{path}.include[{index}]")
    if "exclude" in obj:
        exclude = _string_list(obj["exclude"], f"{path}.exclude")
        for index, pattern in enumerate(exclude):
            _safe_relative_path(pattern, f"{path}.exclude[{index}]")


def _validate_checkpoint(value: Any) -> None:
    path = "$.spec.checkpoint"
    obj = _object(value, path)
    _required(obj, {"enabled"}, path)
    _keys(
        obj,
        {"enabled", "glob", "publish_interval_sec", "stable_for_sec", "resume_argv"},
        path,
    )
    _boolean(obj["enabled"], f"{path}.enabled")
    if "glob" in obj:
        _safe_relative_path(obj["glob"], f"{path}.glob")
    if "publish_interval_sec" in obj:
        _integer(
            obj["publish_interval_sec"],
            f"{path}.publish_interval_sec",
            minimum=1,
        )
    if "stable_for_sec" in obj:
        _integer(obj["stable_for_sec"], f"{path}.stable_for_sec", minimum=0)
    if "resume_argv" in obj:
        _string_list(obj["resume_argv"], f"{path}.resume_argv", nonempty=True)


def _validate_storage(value: Any) -> None:
    path = "$.spec.storage"
    obj = _object(value, path)
    _required(obj, {"project", "remote"}, path)
    _keys(obj, {"project", "remote"}, path)
    project = _string(obj["project"], f"{path}.project")
    if not _PROJECT_RE.fullmatch(project):
        _fail(f"{path}.project", "must be a lowercase path-safe project name")
    remote = _string(obj["remote"], f"{path}.remote")
    if remote == "${CFEXP_ARTIFACT_REMOTE}":
        return
    if _ENV_REFERENCE_RE.search(remote):
        _fail(f"{path}.remote", "contains an unresolved or unsupported environment variable")
    if remote.startswith("file:///"):
        parsed = urlsplit(remote)
        normalized_path = Path(
            os.path.abspath(os.path.normpath(unquote(parsed.path)))
        )
        if normalized_path.parent == normalized_path:
            _fail(f"{path}.remote", "must not name the filesystem root")
        return
    if remote.startswith("hf://"):
        parsed = urlsplit(remote)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if (
            parsed.scheme != "hf"
            or parsed.netloc != "datasets"
            or parsed.query
            or parsed.fragment
            or len(parts) < 2
            or any(part in {".", ".."} for part in parts)
        ):
            _fail(
                f"{path}.remote",
                "must use hf://datasets/<namespace>/<private-dataset>[/prefix]",
            )
        return
    if "://" in remote or not _RCLONE_REMOTE_RE.fullmatch(remote):
        _fail(
            f"{path}.remote",
            "must be file:///..., hf://datasets/..., or rclone syntax such as "
            "remote:bucket/prefix",
        )


def _validate_policy(value: Any, *, gpus: int, duration: int) -> None:
    path = "$.spec.policy"
    obj = _object(value, path)
    _keys(obj, {"coordination_notice_ref"}, path)
    if "coordination_notice_ref" in obj:
        _string(obj["coordination_notice_ref"], f"{path}.coordination_notice_ref")
    if gpus >= 4 and duration > 120 and not obj.get("coordination_notice_ref"):
        _fail(
            f"{path}.coordination_notice_ref",
            "is required when gpus >= 4 and expected_duration_minutes > 120",
        )


def validate_manifest(data: Mapping[str, Any]) -> None:
    """Strictly validate an experiment manifest.

    Unknown fields are rejected at the top level and at every structural layer.
    The function returns ``None`` on success and raises :class:`ManifestError`
    with a JSON-path-like location on the first violation.
    """

    _validate_json_value(data, "$")
    root = _object(data, "$")
    _required(root, {"api_version", "kind", "metadata", "spec"}, "$")
    _keys(root, {"api_version", "kind", "metadata", "spec"}, "$")
    if root["api_version"] != API_VERSION:
        _fail("$.api_version", f"must be {API_VERSION!r}")
    if root["kind"] != KIND:
        _fail("$.kind", f"must be {KIND!r}")
    _validate_metadata(root["metadata"])

    spec_path = "$.spec"
    spec = _object(root["spec"], spec_path)
    required = {
        "code",
        "runtime",
        "command",
        "resources",
        "outputs",
        "storage",
        "policy",
    }
    allowed = required | {"inputs", "parameters", "checkpoint"}
    _required(spec, required, spec_path)
    _keys(spec, allowed, spec_path)

    _validate_code(spec["code"])
    _validate_runtime(spec["runtime"])
    _validate_command(spec["command"])
    gpus, duration = _validate_resources(spec["resources"])
    if "inputs" in spec:
        _validate_inputs(spec["inputs"])
    if "parameters" in spec:
        _validate_json_value(spec["parameters"], "$.spec.parameters")
    _validate_outputs(spec["outputs"])
    if "checkpoint" in spec:
        _validate_checkpoint(spec["checkpoint"])
    _validate_storage(spec["storage"])
    _validate_policy(spec["policy"], gpus=gpus, duration=duration)


__all__ = [
    "API_VERSION",
    "KIND",
    "ManifestError",
    "canonical_json",
    "load_manifest",
    "manifest_hash",
    "normalize_manifest",
    "spec_hash",
    "validate_manifest",
]
