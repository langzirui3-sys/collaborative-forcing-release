"""Persistent JSONL worker for Draft/Target correction experiments.

The worker deliberately has no machine-learning imports at module import time.
One process owns one already-loaded model backend and accepts auditable requests
over stdin.  The original stdout file descriptor is reserved for protocol
responses before a real backend is imported; stdout from model code is then
redirected to stderr so a stray ``print`` cannot corrupt the JSONL stream.

All request tensor inputs and every generated output are relative to the
resolved ``CFEXP_RUN_DIR``.  Tensor files are staged beside their destination
and are published only after every tensor returned by a backend has been
validated and written successfully.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
import traceback
from typing import Any, Callable, Mapping, Protocol, Sequence, TextIO
import uuid


from .stage_timing import stage, timing_session


REQUEST_SCHEMA = "correction_worker_request_v1"
RESPONSE_SCHEMA = "correction_worker_response_v2"
TARGET_RPC_LIFECYCLE_SCHEMA = "insight2_target_rpc_lifecycle_v2"
BOUNDARY_VELOCITY_WITNESS_SCHEMA = "correction_boundary_velocity_witness_v1"
BOUNDARY_VELOCITY_METRIC_ID = "latent_velocity_transition_discontinuity_v1"
BOUNDARY_VELOCITY_ACCUMULATOR_DTYPE = "float64"
BOUNDARY_VELOCITY_ROUND_DECIMALS = 12
MODEL_ROLES = frozenset({"draft", "target"})
MAX_REQUEST_BYTES = 1024 * 1024
SUPPORTED_CHUNK_COUNTS = frozenset({16, 30, 60, 120})

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_FIELD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TARGET_RPC_STATUSES = ("started", "backend_returned", "completed")
_TARGET_RPC_PHASES = frozenset(
    {
        "unscoped",
        "single_point_candidate",
        "calibration_baseline",
        "evaluation_baseline",
        "route_materialization",
        "selection_search",
    }
)
_TARGET_RPC_LIFECYCLE_KEYS = {
    "schema",
    "event_id",
    "attempt_id",
    "logical_call_id",
    "scope_id",
    "phase",
    "status",
    "request_id",
    "model_id",
    "model_artifact_sha256",
    "prompt_sha256",
    "chunk_index",
    "internal_seed",
    "history_latent_paths",
    "output_latent_path",
    "history_chunk_count",
    "initial_noise_path",
    "output_latent_sha256",
    "kv_witness_sha256",
}


class WorkerError(RuntimeError):
    """Base class for correction-worker failures."""


class WorkerProtocolError(WorkerError):
    """A request or run-directory path is invalid; the backend was not called."""


class WorkerExecutionError(WorkerError):
    """The backend or output publication failed and the worker is poisoned."""


class WorkerPoisonedError(WorkerError):
    """A state-changing action was attempted after an execution failure."""


class TensorCodec(Protocol):
    """Minimal tensor persistence API, injectable for CPU-only unit tests."""

    def is_tensor(self, value: Any) -> bool: ...

    def load(self, path: Path, *, key: str) -> Any: ...

    def save(self, path: Path, *, key: str, value: Any) -> None: ...

    def concatenate(self, values: Sequence[Any], *, dim: int) -> Any: ...

    def describe(self, value: Any) -> tuple[tuple[int, ...], str]: ...

    def boundary_velocity_witness(
        self, *, history: Any, candidate: Any
    ) -> Mapping[str, Any]: ...


class SafeTensorCodec:
    """Lazy Torch/safetensors implementation used by formal GPU workers."""

    def __init__(self) -> None:
        self._torch: Any | None = None
        self._load_file: Any | None = None
        self._save_file: Any | None = None

    def _ensure_runtime(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch  # type: ignore
            from safetensors.torch import load_file, save_file  # type: ignore
        except ImportError as error:
            raise WorkerError("formal workers require torch and safetensors") from error
        self._torch = torch
        self._load_file = load_file
        self._save_file = save_file

    def is_tensor(self, value: Any) -> bool:
        self._ensure_runtime()
        assert self._torch is not None
        return isinstance(value, self._torch.Tensor)

    def load(self, path: Path, *, key: str) -> Any:
        self._ensure_runtime()
        assert self._load_file is not None
        tensors = self._load_file(str(path), device="cpu")
        if set(tensors) != {key}:
            raise WorkerProtocolError(
                f"tensor input {path.name!r} must contain exactly key {key!r}"
            )
        return tensors[key]

    def save(self, path: Path, *, key: str, value: Any) -> None:
        self._ensure_runtime()
        assert self._save_file is not None
        if not self.is_tensor(value):
            raise WorkerExecutionError(f"backend field {key!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        self._save_file({key: tensor}, str(path))

    def concatenate(self, values: Sequence[Any], *, dim: int) -> Any:
        self._ensure_runtime()
        assert self._torch is not None
        if not values:
            raise WorkerProtocolError("cannot concatenate an empty tensor sequence")
        return self._torch.cat(list(values), dim=dim).contiguous()

    def describe(self, value: Any) -> tuple[tuple[int, ...], str]:
        self._ensure_runtime()
        if not self.is_tensor(value):
            raise WorkerExecutionError("cannot describe a non-tensor backend value")
        dtype = str(value.dtype)
        if dtype.startswith("torch."):
            dtype = dtype[len("torch.") :]
        return tuple(int(item) for item in value.shape), dtype

    def boundary_velocity_witness(
        self, *, history: Any, candidate: Any
    ) -> Mapping[str, Any]:
        """Measure the generated boundary with deterministic CPU-float64 math.

        The witness compares the velocity immediately before the boundary, the
        velocity across it, and the first velocity inside the candidate.  It
        deliberately consumes only the already-materialized history and the
        current candidate; no decode or model call is involved.
        """

        self._ensure_runtime()
        assert self._torch is not None
        torch = self._torch
        if not self.is_tensor(history) or not self.is_tensor(candidate):
            raise WorkerExecutionError(
                "boundary velocity witness requires tensor history and candidate"
            )
        history_shape, history_dtype = self.describe(history)
        candidate_shape, candidate_dtype = self.describe(candidate)
        if (
            len(history_shape) != 5
            or len(candidate_shape) != 5
            or history_shape[0] != 1
            or candidate_shape[0] != 1
            or history_shape[1] < 2
            or candidate_shape[1] < 2
            or history_shape[2:] != candidate_shape[2:]
            or history_dtype != candidate_dtype
        ):
            raise WorkerExecutionError(
                "boundary velocity witness inputs have incompatible shapes or dtypes"
            )
        if history_dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise WorkerExecutionError(
                "boundary velocity witness inputs must use a floating dtype"
            )
        history64 = history[:, -2:].detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        candidate64 = candidate[:, :2].detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        if not bool(torch.isfinite(history64).all().item()) or not bool(
            torch.isfinite(candidate64).all().item()
        ):
            raise WorkerExecutionError(
                "boundary velocity witness inputs contain non-finite values"
            )

        previous_velocity = history64[:, 1] - history64[:, 0]
        boundary_velocity = candidate64[:, 0] - history64[:, -1]
        candidate_velocity = candidate64[:, 1] - candidate64[:, 0]

        def norm(value: Any) -> float:
            result = float(torch.linalg.vector_norm(value.reshape(-1), ord=2).item())
            if not math.isfinite(result) or result < 0.0:
                raise WorkerExecutionError(
                    "boundary velocity witness produced an invalid norm"
                )
            return result

        previous_norm = norm(previous_velocity)
        boundary_norm = norm(boundary_velocity)
        candidate_norm = norm(candidate_velocity)

        def discontinuity(
            left: Any,
            right: Any,
            left_norm: float,
            right_norm: float,
        ) -> float:
            denominator = left_norm + right_norm
            raw = 0.0 if denominator == 0.0 else norm(left - right) / denominator
            rounded = round(raw, BOUNDARY_VELOCITY_ROUND_DECIMALS)
            if not math.isfinite(rounded) or not 0.0 <= rounded <= 1.0:
                raise WorkerExecutionError(
                    "boundary velocity witness produced an out-of-range discontinuity"
                )
            return rounded

        incoming = discontinuity(
            previous_velocity,
            boundary_velocity,
            previous_norm,
            boundary_norm,
        )
        outgoing = discontinuity(
            boundary_velocity,
            candidate_velocity,
            boundary_norm,
            candidate_norm,
        )
        return {
            "schema": BOUNDARY_VELOCITY_WITNESS_SCHEMA,
            "metric_id": BOUNDARY_VELOCITY_METRIC_ID,
            "accumulator_dtype": BOUNDARY_VELOCITY_ACCUMULATOR_DTYPE,
            "round_decimals": BOUNDARY_VELOCITY_ROUND_DECIMALS,
            "input_dtype": history_dtype,
            "history_frame_count": history_shape[1],
            "candidate_frame_count": candidate_shape[1],
            "previous_velocity_norm": round(
                previous_norm, BOUNDARY_VELOCITY_ROUND_DECIMALS
            ),
            "boundary_velocity_norm": round(
                boundary_norm, BOUNDARY_VELOCITY_ROUND_DECIMALS
            ),
            "candidate_velocity_norm": round(
                candidate_norm, BOUNDARY_VELOCITY_ROUND_DECIMALS
            ),
            "incoming_discontinuity": incoming,
            "outgoing_discontinuity": outgoing,
            "score": max(incoming, outgoing),
        }


@dataclass(frozen=True)
class _StagedOutput:
    relative_path: str
    target: Path
    temporary: Path
    metadata: dict[str, Any]


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise WorkerProtocolError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                WorkerProtocolError(f"{label} contains invalid constant {item!r}")
            ),
        )
    except WorkerProtocolError:
        raise
    except json.JSONDecodeError as error:
        raise WorkerProtocolError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise WorkerProtocolError(f"{label} must be a JSON object")
    return value


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerProtocolError(f"{label} must be a non-empty relative path")
    if "\\" in value or "\x00" in value:
        raise WorkerProtocolError(f"{label} must be a canonical POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or str(posix) != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise WorkerProtocolError(f"{label} must remain below CFEXP_RUN_DIR")
    return value


def _ensure_run_root(run_dir: Path) -> Path:
    try:
        root = run_dir.expanduser().resolve(strict=True)
    except OSError as error:
        raise WorkerProtocolError(f"CFEXP_RUN_DIR is unavailable: {error}") from error
    if not root.is_dir():
        raise WorkerProtocolError("CFEXP_RUN_DIR must be a directory")
    return root


def _input_path(root: Path, relative_path: Any, *, label: str) -> tuple[str, Path]:
    relative = _canonical_relative_path(relative_path, label=label)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise WorkerProtocolError(f"{label} is unavailable: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise WorkerProtocolError(f"{label} may not traverse a symlink")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise WorkerProtocolError(f"{label} escapes CFEXP_RUN_DIR") from error
    if not resolved.is_file():
        raise WorkerProtocolError(f"{label} must name a regular file")
    return relative, resolved


def _output_path(root: Path, relative_path: Any, *, label: str) -> tuple[str, Path]:
    relative = _canonical_relative_path(relative_path, label=label)
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts[:-1]:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                info = current.lstat()
            except OSError as error:
                raise WorkerProtocolError(f"cannot inspect {label}: {error}") from error
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise WorkerProtocolError(f"{label} parent must be a real directory")
        else:
            try:
                current.mkdir(mode=0o750)
            except OSError as error:
                raise WorkerProtocolError(f"cannot create {label} parent: {error}") from error
    target = root.joinpath(*parts)
    try:
        target.resolve(strict=False).relative_to(root)
    except (OSError, ValueError) as error:
        raise WorkerProtocolError(f"{label} escapes CFEXP_RUN_DIR") from error
    if target.exists() or target.is_symlink():
        raise WorkerProtocolError(f"{label} already exists")
    return relative, target


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise WorkerExecutionError("staged output is not a regular file")
            while block := handle.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise WorkerExecutionError(f"cannot hash staged output: {error}") from error
    if (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise WorkerExecutionError("staged output changed while it was hashed")
    return digest.hexdigest(), after.st_size


def _temporary_sibling(target: Path) -> Path:
    return target.with_name(f".{target.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{target.suffix}")


def _publish_staged(outputs: Sequence[_StagedOutput]) -> None:
    published: list[Path] = []
    try:
        for output in outputs:
            # Hard linking is an atomic no-overwrite publication on the same filesystem.
            os.link(output.temporary, output.target)
            published.append(output.target)
        for output in outputs:
            output.temporary.unlink()
    except OSError as error:
        for path in published:
            try:
                path.unlink()
            except OSError:
                pass
        for output in outputs:
            try:
                output.temporary.unlink()
            except OSError:
                pass
        raise WorkerExecutionError(f"cannot publish staged outputs atomically: {error}") from error


def _discard_staged(outputs: Sequence[_StagedOutput]) -> None:
    for output in outputs:
        try:
            output.temporary.unlink()
        except OSError:
            pass


def _json_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkerExecutionError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkerExecutionError(f"{label} contains a non-string object key")
            normalized[key] = _json_value(item, label=f"{label}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label=f"{label}[]") for item in value]
    raise WorkerExecutionError(f"{label} contains non-JSON value {type(value).__name__}")


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


def _finite_witness_number(value: Any, *, label: str, minimum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise WorkerExecutionError(
            f"boundary velocity witness {label} must be finite and >= {minimum}"
        )
    parsed = float(value)
    if parsed != round(parsed, BOUNDARY_VELOCITY_ROUND_DECIMALS):
        raise WorkerExecutionError(
            f"boundary velocity witness {label} is not rounded to "
            f"{BOUNDARY_VELOCITY_ROUND_DECIMALS} decimals"
        )
    return parsed


def _validated_boundary_velocity_witness(
    value: Any,
    *,
    history_frame_count: int,
    candidate_frame_count: int,
    input_dtype: str,
) -> dict[str, Any]:
    """Validate a codec-produced witness before it enters worker evidence."""

    if (
        isinstance(history_frame_count, bool)
        or not isinstance(history_frame_count, int)
        or history_frame_count < 2
        or isinstance(candidate_frame_count, bool)
        or not isinstance(candidate_frame_count, int)
        or candidate_frame_count < 2
    ):
        raise WorkerExecutionError(
            "boundary velocity witness frame counts must both be integers >= 2"
        )
    if not isinstance(value, Mapping) or set(value) != _BOUNDARY_VELOCITY_WITNESS_KEYS:
        raise WorkerExecutionError("boundary velocity witness fields are invalid")
    if (
        value.get("schema") != BOUNDARY_VELOCITY_WITNESS_SCHEMA
        or value.get("metric_id") != BOUNDARY_VELOCITY_METRIC_ID
        or value.get("accumulator_dtype")
        != BOUNDARY_VELOCITY_ACCUMULATOR_DTYPE
        or not isinstance(value.get("round_decimals"), int)
        or isinstance(value.get("round_decimals"), bool)
        or value.get("round_decimals") != BOUNDARY_VELOCITY_ROUND_DECIMALS
        or value.get("input_dtype") != input_dtype
        or not isinstance(value.get("history_frame_count"), int)
        or isinstance(value.get("history_frame_count"), bool)
        or value.get("history_frame_count") != history_frame_count
        or not isinstance(value.get("candidate_frame_count"), int)
        or isinstance(value.get("candidate_frame_count"), bool)
        or value.get("candidate_frame_count") != candidate_frame_count
    ):
        raise WorkerExecutionError(
            "boundary velocity witness metric or input binding is invalid"
        )
    for field in (
        "previous_velocity_norm",
        "boundary_velocity_norm",
        "candidate_velocity_norm",
    ):
        _finite_witness_number(value.get(field), label=field, minimum=0.0)
    incoming = _finite_witness_number(
        value.get("incoming_discontinuity"),
        label="incoming_discontinuity",
        minimum=0.0,
    )
    outgoing = _finite_witness_number(
        value.get("outgoing_discontinuity"),
        label="outgoing_discontinuity",
        minimum=0.0,
    )
    score = _finite_witness_number(
        value.get("score"), label="score", minimum=0.0
    )
    if incoming > 1.0 or outgoing > 1.0 or score > 1.0:
        raise WorkerExecutionError(
            "boundary velocity witness discontinuity must be in [0, 1]"
        )
    if score != max(incoming, outgoing):
        raise WorkerExecutionError(
            "boundary velocity witness score must equal the maximum discontinuity"
        )
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise WorkerProtocolError(
            f"{label} must contain exactly {sorted(expected)!r}; got {sorted(actual)!r}"
        )


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerProtocolError(f"{label} must be an integer >= {minimum}")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise WorkerProtocolError(f"{label} is not a portable identifier")
    return value


class CorrectionWorker:
    """One persistent model backend with fail-closed request handling."""

    _COMMON_KEYS = {"schema", "request_id", "action", "role"}

    def __init__(
        self,
        *,
        role: str,
        backend: Any,
        run_dir: Path,
        tensor_codec: TensorCodec | None = None,
    ) -> None:
        if role not in MODEL_ROLES:
            raise WorkerProtocolError("worker role must be 'draft' or 'target'")
        self.role = role
        self.backend = backend
        self.run_dir = _ensure_run_root(run_dir)
        self.codec = tensor_codec if tensor_codec is not None else SafeTensorCodec()
        self.poisoned = False
        self.poison_reason: str | None = None
        self.closed = False
        state = getattr(backend, "model_state", None)
        if state is None:
            state = {"backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}"}
        if not isinstance(state, Mapping):
            raise WorkerExecutionError("backend.model_state must be a mapping")
        self._backend_model_state = _json_value(state, label="backend.model_state")
        chunk_count = getattr(
            backend, "chunk_count", self._backend_model_state.get("chunk_count", 16)
        )
        if (
            not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count not in SUPPORTED_CHUNK_COUNTS
        ):
            raise WorkerExecutionError(
                "backend chunk_count must be one of 16, 30, 60, or 120"
            )
        self._chunk_count = chunk_count
        self._target_lifecycle_path = self.run_dir / "logs" / "target-rpc-lifecycle.jsonl"
        self._target_lifecycle_records: dict[str, dict[str, Any]] = {}
        self._target_attempt_ids: set[str] = set()
        self._worker_instance_id = uuid.uuid4().hex
        self._target_attempt_counter = 0
        if self.role == "target":
            self._load_target_lifecycle()

    def _load_target_lifecycle(self) -> None:
        path = self._target_lifecycle_path
        if not path.exists():
            return
        if path.is_symlink() or not path.is_file():
            raise WorkerExecutionError("Target RPC lifecycle log must be a regular file")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise WorkerExecutionError(f"cannot read Target RPC lifecycle log: {error}") from error
        for line_number, raw in enumerate(lines, 1):
            if not raw:
                raise WorkerExecutionError(
                    f"Target RPC lifecycle log has a blank line at {line_number}"
                )
            try:
                record = _parse_json_object(
                    raw, label=f"Target RPC lifecycle line {line_number}"
                )
            except WorkerProtocolError as error:
                raise WorkerExecutionError(str(error)) from error
            if set(record) != _TARGET_RPC_LIFECYCLE_KEYS:
                raise WorkerExecutionError("Target RPC lifecycle fields are invalid")
            if record.get("schema") != TARGET_RPC_LIFECYCLE_SCHEMA:
                raise WorkerExecutionError("Target RPC lifecycle schema is invalid")
            event_id = record.get("event_id")
            if not isinstance(event_id, str) or _SHA256_RE.fullmatch(event_id) is None:
                raise WorkerExecutionError("Target RPC lifecycle event_id is invalid")
            expected_event_id = hashlib.sha256(
                json.dumps(
                    {key: value for key, value in record.items() if key != "event_id"},
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if event_id != expected_event_id:
                raise WorkerExecutionError("Target RPC lifecycle event digest is invalid")
            if event_id in self._target_lifecycle_records:
                raise WorkerExecutionError("Target RPC lifecycle event_id is duplicated")
            self._target_lifecycle_records[event_id] = record
            attempt_id = record.get("attempt_id")
            if not isinstance(attempt_id, str) or _SHA256_RE.fullmatch(attempt_id) is None:
                raise WorkerExecutionError("Target RPC lifecycle attempt_id is invalid")
            self._target_attempt_ids.add(attempt_id)

        attempts: dict[str, dict[str, Any]] = {}
        for record in self._target_lifecycle_records.values():
            attempt_id = record["attempt_id"]
            logical_call_id = record["logical_call_id"]
            scope_id = record["scope_id"]
            phase = record["phase"]
            status = record["status"]
            if (
                not isinstance(logical_call_id, str)
                or _SHA256_RE.fullmatch(logical_call_id) is None
                or (
                    scope_id is not None
                    and (not isinstance(scope_id, str) or _SHA256_RE.fullmatch(scope_id) is None)
                )
                or phase not in _TARGET_RPC_PHASES
                or (scope_id is None) != (phase == "unscoped")
                or status not in _TARGET_RPC_STATUSES
            ):
                raise WorkerExecutionError("Target RPC lifecycle identity is invalid")
            history_paths = record["history_latent_paths"]
            chunk_index = record["chunk_index"]
            history_count = record["history_chunk_count"]
            if (
                not isinstance(history_paths, list)
                or any(not isinstance(item, str) for item in history_paths)
                or len(set(history_paths)) != len(history_paths)
                or not isinstance(chunk_index, int)
                or isinstance(chunk_index, bool)
                or not 0 <= chunk_index < self._chunk_count
                or not isinstance(history_count, int)
                or isinstance(history_count, bool)
                or history_count != chunk_index
                or len(history_paths) != history_count
            ):
                raise WorkerExecutionError("Target RPC lifecycle history binding is invalid")
            try:
                _identifier(record["request_id"], label="lifecycle request_id")
                for index, history_path in enumerate(history_paths):
                    _canonical_relative_path(
                        history_path,
                        label=f"lifecycle history_latent_paths[{index}]",
                    )
                _canonical_relative_path(
                    record["output_latent_path"], label="lifecycle output_latent_path"
                )
                _canonical_relative_path(
                    record["initial_noise_path"], label="lifecycle initial_noise_path"
                )
            except WorkerProtocolError as error:
                raise WorkerExecutionError(str(error)) from error
            model_id = record["model_id"]
            model_artifact = record["model_artifact_sha256"]
            prompt_sha = record["prompt_sha256"]
            internal_seed = record["internal_seed"]
            if (
                not isinstance(model_id, str)
                or not model_id
                or not isinstance(model_artifact, str)
                or _SHA256_RE.fullmatch(model_artifact) is None
                or not isinstance(prompt_sha, str)
                or _SHA256_RE.fullmatch(prompt_sha) is None
                or not isinstance(internal_seed, int)
                or isinstance(internal_seed, bool)
                or internal_seed < 0
            ):
                raise WorkerExecutionError("Target RPC lifecycle request evidence is invalid")
            output_sha = record["output_latent_sha256"]
            kv_sha = record["kv_witness_sha256"]
            if status == "completed":
                if (
                    not isinstance(output_sha, str)
                    or _SHA256_RE.fullmatch(output_sha) is None
                    or not isinstance(kv_sha, str)
                    or _SHA256_RE.fullmatch(kv_sha) is None
                ):
                    raise WorkerExecutionError(
                        "completed Target RPC lifecycle event lacks output evidence"
                    )
            elif output_sha is not None or kv_sha is not None:
                raise WorkerExecutionError(
                    "pre-completion Target RPC lifecycle event carries output evidence"
                )
            binding = {
                key: record[key]
                for key in _TARGET_RPC_LIFECYCLE_KEYS
                if key
                not in {
                    "schema",
                    "event_id",
                    "status",
                    "output_latent_sha256",
                    "kv_witness_sha256",
                }
            }
            state = attempts.setdefault(attempt_id, {"binding": binding, "statuses": []})
            if state["binding"] != binding:
                raise WorkerExecutionError(
                    "Target RPC lifecycle request binding changed within one attempt"
                )
            statuses = state["statuses"]
            statuses.append(status)
            expected = list(_TARGET_RPC_STATUSES[: len(statuses)])
            if statuses != expected:
                raise WorkerExecutionError("Target RPC lifecycle stages are out of order")

    def _new_target_attempt_id(self, logical_call_id: str) -> str:
        while True:
            self._target_attempt_counter += 1
            attempt_id = hashlib.sha256(
                json.dumps(
                    [
                        TARGET_RPC_LIFECYCLE_SCHEMA,
                        self._worker_instance_id,
                        self._target_attempt_counter,
                        logical_call_id,
                    ],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if attempt_id not in self._target_attempt_ids:
                self._target_attempt_ids.add(attempt_id)
                return attempt_id

    def _record_target_lifecycle(
        self,
        *,
        request: Mapping[str, Any],
        attempt_id: str,
        status: str,
        output_latent_path: str,
        history_chunk_count: int,
        initial_noise_path: str,
        output_latent_sha256: str | None = None,
        kv_witness_sha256: str | None = None,
    ) -> None:
        if self.role != "target":
            return
        if status not in _TARGET_RPC_STATUSES:
            raise WorkerExecutionError("Target RPC lifecycle status is invalid")
        logical_call_id = request.get("target_logical_call_id")
        scope_id = request.get("target_call_scope_id")
        phase = request.get("target_call_phase")
        if not isinstance(attempt_id, str) or _SHA256_RE.fullmatch(attempt_id) is None:
            raise WorkerProtocolError("target attempt_id must be a lowercase SHA-256")
        if not isinstance(logical_call_id, str) or _SHA256_RE.fullmatch(logical_call_id) is None:
            raise WorkerProtocolError("target_logical_call_id must be a lowercase SHA-256")
        if scope_id is not None and (
            not isinstance(scope_id, str) or _SHA256_RE.fullmatch(scope_id) is None
        ):
            raise WorkerProtocolError("target_call_scope_id must be null or a lowercase SHA-256")
        if phase not in _TARGET_RPC_PHASES:
            raise WorkerProtocolError("target_call_phase is invalid")
        if (scope_id is None) != (phase == "unscoped"):
            raise WorkerProtocolError("Target scope and lifecycle phase disagree")
        if status == "completed":
            if (
                not isinstance(output_latent_sha256, str)
                or _SHA256_RE.fullmatch(output_latent_sha256) is None
                or not isinstance(kv_witness_sha256, str)
                or _SHA256_RE.fullmatch(kv_witness_sha256) is None
            ):
                raise WorkerExecutionError("completed Target lifecycle event lacks output evidence")
        elif output_latent_sha256 is not None or kv_witness_sha256 is not None:
            raise WorkerExecutionError("pre-completion Target lifecycle event carries output evidence")
        backend_model_id = self._backend_model_state.get("model_id")
        backend_artifact = self._backend_model_state.get("checkpoint_sha256")
        if not isinstance(backend_model_id, str) or not backend_model_id:
            raise WorkerExecutionError("Target backend model_id is unavailable")
        if not isinstance(backend_artifact, str) or _SHA256_RE.fullmatch(backend_artifact) is None:
            raise WorkerExecutionError("Target backend checkpoint digest is unavailable")
        record = {
            "schema": TARGET_RPC_LIFECYCLE_SCHEMA,
            "attempt_id": attempt_id,
            "logical_call_id": logical_call_id,
            "scope_id": scope_id,
            "phase": phase,
            "status": status,
            "request_id": request["request_id"],
            "model_id": backend_model_id,
            "model_artifact_sha256": backend_artifact,
            "prompt_sha256": hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest(),
            "chunk_index": request["chunk_index"],
            "internal_seed": request["internal_seed"],
            "history_latent_paths": list(request["history_latent_paths"]),
            "output_latent_path": output_latent_path,
            "history_chunk_count": history_chunk_count,
            "initial_noise_path": initial_noise_path,
            "output_latent_sha256": output_latent_sha256,
            "kv_witness_sha256": kv_witness_sha256,
        }
        event_id = hashlib.sha256(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        record["event_id"] = event_id
        existing = self._target_lifecycle_records.get(event_id)
        if existing is not None:
            if existing != record:
                raise WorkerExecutionError("Target RPC lifecycle event content changed")
            return
        path = self._target_lifecycle_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise WorkerExecutionError("Target RPC lifecycle log must be a regular file")
        raw = (
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        try:
            with stage("lifecycle_persist"):
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(descriptor, "ab") as handle:
                    opened = os.fstat(handle.fileno())
                    if not stat.S_ISREG(opened.st_mode):
                        raise WorkerExecutionError("Target RPC lifecycle log is not regular")
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError as error:
            raise WorkerExecutionError(f"cannot append Target RPC lifecycle log: {error}") from error
        self._target_lifecycle_records[event_id] = record

    def _state_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "poisoned": self.poisoned,
            "poison_reason": self.poison_reason,
            "backend": self._backend_model_state,
        }

    def ready_payload(self) -> dict[str, Any]:
        return {"model_state": self._state_payload()}

    def _poison(self, error: BaseException) -> WorkerExecutionError:
        self.poisoned = True
        self.poison_reason = f"{type(error).__name__}: {error}"
        if isinstance(error, WorkerExecutionError):
            return error
        return WorkerExecutionError(self.poison_reason)

    def _validate_common(self, request: Mapping[str, Any]) -> tuple[str, str]:
        if request.get("schema") != REQUEST_SCHEMA:
            raise WorkerProtocolError(f"request.schema must equal {REQUEST_SCHEMA!r}")
        request_id = _identifier(request.get("request_id"), label="request.request_id")
        action = request.get("action")
        if action not in {
            "model_state",
            "start_trajectory",
            "generate_persistent",
            "append_persistent_clean",
            "generate_persistent_suffix",
            "generate_t_stitch_step",
            "start_sdvg_trajectory",
            "score_sdvg_candidate",
            "finish_sdvg_trajectory",
            "finish_trajectory",
            "generate",
            "generate_early",
            "commit_early_target",
            "generate_target_prefix",
            "resume_early_target_prefix",
            "decode",
            "decode_video",
            "score",
            "shutdown",
        }:
            raise WorkerProtocolError("request.action is unsupported")
        if request.get("role") != self.role:
            raise WorkerProtocolError(
                f"request.role must match the {self.role!r} worker"
            )
        if self.closed:
            raise WorkerPoisonedError("worker is already closed")
        if self.poisoned and action not in {"model_state", "shutdown"}:
            raise WorkerPoisonedError(
                "worker is poisoned after a backend/output failure; "
                "only model_state or shutdown is allowed"
            )
        return request_id, str(action)

    def handle(self, request: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
        request_id, action = self._validate_common(request)
        if action == "model_state":
            _exact_keys(request, self._COMMON_KEYS, label="model_state request")
            return request_id, action, {"model_state": self._state_payload()}
        if action == "shutdown":
            _exact_keys(request, self._COMMON_KEYS, label="shutdown request")
            self.close()
            return request_id, action, {"closed": True}
        if action == "start_trajectory":
            return request_id, action, self._timed_backend_action(request, action=action)
        if action == "generate_persistent":
            return request_id, action, self._timed_backend_action(request, action=action)
        if action == "append_persistent_clean":
            return request_id, action, self._append_persistent_clean(request)
        if action == "generate_persistent_suffix":
            return request_id, action, self._generate_persistent_suffix(request)
        if action == "generate_t_stitch_step":
            return request_id, action, self._generate_t_stitch_step(request)
        if action == "start_sdvg_trajectory":
            return request_id, action, self._start_sdvg_trajectory(request)
        if action == "score_sdvg_candidate":
            return request_id, action, self._score_sdvg_candidate(request)
        if action == "finish_sdvg_trajectory":
            return request_id, action, self._finish_sdvg_trajectory(request)
        if action == "finish_trajectory":
            return request_id, action, self._finish_trajectory(request)
        if action == "commit_early_target":
            from .early_worker import commit
            with timing_session(self.run_dir, "worker-draft", {"action": action, "request_id": request_id}):
                return request_id, action, commit(self, request)
        if action in {"generate_target_prefix", "resume_early_target_prefix"}:
            from .prefix_worker import generate_prefix, resume_prefix
            with timing_session(self.run_dir, "worker-"+self.role,
                                {"action": action, "request_id": request_id}):
                fn = generate_prefix if action == "generate_target_prefix" else resume_prefix
                return request_id, action, fn(self, request)
        if action in {"generate", "generate_early"}:
            if action == "generate" and getattr(self, "_early_session", None) is not None:
                raise WorkerProtocolError("cannot mix ordinary and early Draft sessions")
            return request_id, action, self._timed_backend_action(request, action=action)
        if action == "decode":
            return request_id, action, self._decode(request)
        if action == "decode_video":
            return request_id, action, self._decode_video(request)
        return request_id, action, self._score(request)

    def _timed_backend_action(self, request: Mapping[str, Any], *, action: str) -> dict[str, Any]:
        # Sidecars carry join keys only; the generation protocol remains unchanged.
        prompt = request.get("prompt")
        output_paths = request.get("output_paths")
        history_paths = request.get("history_latent_paths")
        metadata = {
            "action": request.get("action"),
            "request_id": request.get("request_id"),
            "role": self.role,
            "chunk_index": request.get("chunk_index"),
            "history_chunk_count": len(history_paths) if isinstance(history_paths, list) else None,
            "output_latent_path": (
                output_paths.get("latent") if isinstance(output_paths, Mapping)
                else request.get("output_latent_path")
            ),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if isinstance(prompt, str) else None,
        }
        with timing_session(self.run_dir, f"worker-{self.role}", metadata) as session:
            if session is not None:
                torch = getattr(self.backend, "torch", None)
                device = getattr(self.backend, "device", None)
                if torch is not None:
                    for name in ("get_num_threads", "get_num_interop_threads"):
                        getter = getattr(torch, name, None)
                        if callable(getter):
                            session.metadata["torch_" + name.removeprefix("get_")] = int(getter())
                session.metadata["cpu_thread_environment"] = {
                    key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
                }
                if torch is not None and device is not None:
                    session.attach_cuda(torch, device)
                session.note_not_applicable(
                    "snapshot", "This path does not create a reusable KV snapshot."
                )
                session.note_not_applicable(
                    "kv_restore", "This path does not restore a reusable KV snapshot."
                )
                if action in {"generate", "generate_persistent"} and request.get("chunk_index") == 0:
                    session.note_not_applicable("prefill", "Chunk zero has no historical prefix.")
                if self.role == "target":
                    session.note_not_applicable("rank", "Target does not compute Draft effective rank.")
                if action == "generate_persistent":
                    session.note_not_applicable(
                        "history_read", "Persistent Target history is already resident in the worker."
                    )
                    session.note_not_applicable(
                        "history_concat", "Persistent Target does not concatenate full history inputs."
                    )
            operation = {
                "generate": self._generate,
                "generate_early": self._generate,
                "generate_persistent": self._generate_persistent,
                "start_trajectory": self._start_trajectory,
            }[action]
            from .kernel_profile import profile_request
            with stage("worker_start_trajectory" if action == "start_trajectory" else "worker_generate"), \
                    profile_request(self.backend, run_dir=self.run_dir, metadata=metadata):
                return operation(request)

    def _start_sdvg_trajectory(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {"prompt_id", "prompt", "chunk_count"}
        _exact_keys(request, expected, label="start_sdvg_trajectory request")
        if self.role != "draft":
            raise WorkerProtocolError("only Draft supports SDVG reward trajectories")
        prompt_id = _identifier(request["prompt_id"], label="prompt_id")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        chunk_count = _integer(request["chunk_count"], label="chunk_count", minimum=1)
        if chunk_count != self._chunk_count:
            raise WorkerProtocolError("SDVG trajectory chunk_count drifted")
        start = getattr(self.backend, "start_sdvg_trajectory", None)
        if not callable(start):
            raise WorkerExecutionError("Draft backend lacks SDVG reward support")
        try:
            result = start(
                prompt_id=prompt_id, prompt=prompt, chunk_count=chunk_count
            )
        except BaseException as error:
            raise self._poison(error) from error
        if not isinstance(result, Mapping):
            raise self._poison(WorkerExecutionError("SDVG startup returned no evidence"))
        normalized = _json_value(result, label="sdvg_start_result")
        if not isinstance(normalized, dict):
            raise self._poison(WorkerExecutionError("SDVG startup evidence is invalid"))
        return normalized

    def _score_sdvg_candidate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {"prompt", "chunk_index", "clean_latent_path"}
        _exact_keys(request, expected, label="score_sdvg_candidate request")
        if self.role != "draft":
            raise WorkerProtocolError("only Draft supports SDVG reward scoring")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        chunk_index = _integer(request["chunk_index"], label="chunk_index")
        if chunk_index >= self._chunk_count:
            raise WorkerProtocolError(
                f"chunk_index must be in [0,{self._chunk_count - 1}]"
            )
        clean_path, clean_latent = self._load_input_tensor(
            request["clean_latent_path"], label="clean_latent_path", key="latent"
        )
        score = getattr(self.backend, "score_sdvg_candidate", None)
        if not callable(score):
            raise WorkerExecutionError("Draft backend lacks SDVG reward support")
        try:
            result = score(
                prompt=prompt,
                clean_latent=clean_latent,
                chunk_index=chunk_index,
            )
        except BaseException as error:
            raise self._poison(error) from error
        if not isinstance(result, Mapping):
            raise self._poison(WorkerExecutionError("SDVG scoring returned no evidence"))
        normalized = _json_value(result, label="sdvg_score_result")
        if not isinstance(normalized, dict):
            raise self._poison(WorkerExecutionError("SDVG score evidence is invalid"))
        return {"clean_latent_path": clean_path, "result": normalized}

    def _finish_sdvg_trajectory(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {"prompt_id"}
        _exact_keys(request, expected, label="finish_sdvg_trajectory request")
        if self.role != "draft":
            raise WorkerProtocolError("only Draft supports SDVG reward trajectories")
        prompt_id = _identifier(request["prompt_id"], label="prompt_id")
        finish = getattr(self.backend, "finish_sdvg_trajectory", None)
        if not callable(finish):
            raise WorkerExecutionError("Draft backend lacks SDVG reward support")
        try:
            result = finish(prompt_id=prompt_id)
        except BaseException as error:
            raise self._poison(error) from error
        if not isinstance(result, Mapping):
            raise self._poison(WorkerExecutionError("SDVG completion returned no evidence"))
        normalized = _json_value(result, label="sdvg_finish_result")
        if not isinstance(normalized, dict):
            raise self._poison(WorkerExecutionError("SDVG completion evidence is invalid"))
        return normalized

    def _start_trajectory(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {"prompt_id", "prompt", "seed", "chunk_count"}
        _exact_keys(request, expected, label="start_trajectory request")
        if self.role != "target":
            raise WorkerProtocolError("only Target supports persistent trajectories")
        prompt_id = _identifier(request["prompt_id"], label="prompt_id")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        seed = _integer(request["seed"], label="seed")
        chunk_count = _integer(request["chunk_count"], label="chunk_count", minimum=1)
        if chunk_count != self._chunk_count:
            raise WorkerProtocolError("persistent trajectory chunk_count drifted")
        start = getattr(self.backend, "start_persistent_trajectory", None)
        if not callable(start):
            raise WorkerExecutionError("Target backend lacks persistent trajectory support")
        try:
            result = start(
                prompt_id=prompt_id,
                prompt=prompt,
                seed=seed,
                chunk_count=chunk_count,
            )
        except BaseException as error:
            raise self._poison(error) from error
        if not isinstance(result, Mapping):
            raise self._poison(
                WorkerExecutionError("persistent trajectory startup returned no evidence")
            )
        normalized = _json_value(result, label="persistent_start_result")
        if not isinstance(normalized, dict):
            raise self._poison(
                WorkerExecutionError("persistent trajectory startup evidence is invalid")
            )
        return normalized

    def _finish_trajectory(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {"session_id"}
        _exact_keys(request, expected, label="finish_trajectory request")
        if self.role != "target":
            raise WorkerProtocolError("only Target supports persistent trajectories")
        session_id = _identifier(request["session_id"], label="session_id")
        finish = getattr(self.backend, "finish_persistent_trajectory", None)
        if not callable(finish):
            raise WorkerExecutionError("Target backend lacks persistent trajectory support")
        try:
            result = finish(session_id=session_id)
        except BaseException as error:
            raise self._poison(error) from error
        if not isinstance(result, Mapping):
            raise self._poison(
                WorkerExecutionError("persistent trajectory completion returned no evidence")
            )
        normalized = _json_value(result, label="persistent_finish_result")
        if not isinstance(normalized, dict):
            raise self._poison(
                WorkerExecutionError("persistent trajectory completion evidence is invalid")
            )
        return normalized

    def _history(
        self, raw_paths: Any, *, expected_count: int | None
    ) -> tuple[Any | None, list[str]]:
        if not isinstance(raw_paths, list) or any(not isinstance(item, str) for item in raw_paths):
            raise WorkerProtocolError("history_latent_paths must be a list of relative paths")
        if expected_count is not None and len(raw_paths) != expected_count:
            raise WorkerProtocolError(
                f"history_latent_paths must contain exactly {expected_count} chunks"
            )
        if len(set(raw_paths)) != len(raw_paths):
            raise WorkerProtocolError("history_latent_paths may not contain duplicates")
        tensors: list[Any] = []
        canonical: list[str] = []
        with stage("history_read", detail={"chunk_count": len(raw_paths)}):
            for index, raw in enumerate(raw_paths):
                relative, path = _input_path(
                    self.run_dir, raw, label=f"history_latent_paths[{index}]"
                )
                try:
                    tensor = self.codec.load(path, key="latent")
                except WorkerProtocolError:
                    raise
                except BaseException as error:
                    raise WorkerProtocolError(
                        f"cannot load history_latent_paths[{index}]: {error}"
                    ) from error
                tensors.append(tensor)
                canonical.append(relative)
        if not tensors:
            return None, canonical
        try:
            with stage("history_concat", detail={"chunk_count": len(tensors)}):
                return self.codec.concatenate(tensors, dim=1), canonical
        except WorkerProtocolError:
            raise
        except BaseException as error:
            raise WorkerProtocolError(f"cannot concatenate latent history: {error}") from error

    def _load_input_tensor(self, raw_path: Any, *, label: str, key: str) -> tuple[str, Any]:
        relative, path = _input_path(self.run_dir, raw_path, label=label)
        try:
            with stage("noise_read" if key == "noise" else "tensor_input_read"):
                value = self.codec.load(path, key=key)
        except WorkerProtocolError:
            raise
        except BaseException as error:
            raise WorkerProtocolError(f"cannot load {label}: {error}") from error
        return relative, value

    def _output_map(self, raw: Any, *, required: set[str]) -> dict[str, tuple[str, Path]]:
        if not isinstance(raw, Mapping):
            raise WorkerProtocolError("output_paths must be an object")
        actual = set(raw)
        if actual != required:
            raise WorkerProtocolError(
                f"output_paths must contain exactly {sorted(required)!r}; got {sorted(actual)!r}"
            )
        result: dict[str, tuple[str, Path]] = {}
        for field, path in raw.items():
            if not isinstance(field, str) or _FIELD_RE.fullmatch(field) is None:
                raise WorkerProtocolError("output_paths contains an unsafe field name")
            result[field] = _output_path(
                self.run_dir, path, label=f"output_paths.{field}"
            )
        relative_values = [item[0] for item in result.values()]
        if len(relative_values) != len(set(relative_values)):
            raise WorkerProtocolError("output_paths must be unique")
        return result

    def _stage_tensor(
        self, *, relative: str, target: Path, key: str, value: Any
    ) -> _StagedOutput:
        temporary = _temporary_sibling(target)
        try:
            with stage("tensor_persist", detail={"tensor_field": key}):
                self.codec.save(temporary, key=key, value=value)
            with stage("tensor_hash", detail={"tensor_field": key}):
                digest, size = _sha256_file(temporary)
            shape, dtype = self.codec.describe(value)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        return _StagedOutput(
            relative_path=relative,
            target=target,
            temporary=temporary,
            metadata={
                "path": relative,
                "key": key,
                "sha256": digest,
                "nbytes": size,
                "shape": list(shape),
                "dtype": dtype,
            },
        )

    def _generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {
            "prompt",
            "chunk_index",
            "internal_seed",
            "capture_hidden",
            "history_latent_paths",
            "initial_noise_path",
            "output_paths",
            "target_logical_call_id",
            "target_call_scope_id",
            "target_call_phase",
        }
        early = request.get("action") == "generate_early"
        if early:
            expected = expected | {"early_session_id", "early_control", "history_kv_widths"}
        widths = request.get("history_kv_widths")
        if "history_kv_widths" in request:
            if self.role != "draft":
                raise WorkerProtocolError("history KV widths belong only to Draft")
            expected = expected | {"history_kv_widths"}
        branch_action = request.get("kv_branch_action")
        if "kv_branch_action" in request:
            if (self.role != "draft" or os.environ.get("CFEXP_DRAFT_KV_FORK") != "1"
                    or not isinstance(branch_action, str)
                    or branch_action not in ("none", "preserve", "consume")):
                raise WorkerProtocolError("invalid or disabled Draft KV branch action")
            expected = expected | {"kv_branch_action"}
        _exact_keys(request, expected, label="generate request")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        chunk_index = _integer(request["chunk_index"], label="chunk_index")
        if chunk_index >= self._chunk_count:
            raise WorkerProtocolError(
                f"chunk_index must be in [0,{self._chunk_count - 1}]"
            )
        if "history_kv_widths" in request and (not isinstance(widths, list) or len(widths) != chunk_index
                or any(type(w) is not int or w not in (1, 3) for w in widths)):
            raise WorkerProtocolError("history KV widths must bind every chunk")
        internal_seed = _integer(request["internal_seed"], label="internal_seed")
        capture_hidden = request["capture_hidden"]
        if not isinstance(capture_hidden, bool):
            raise WorkerProtocolError("capture_hidden must be boolean")
        if self.role == "target" and capture_hidden:
            raise WorkerProtocolError("Target generation cannot capture Draft hidden states")
        target_logical_call_id = request["target_logical_call_id"]
        target_call_scope_id = request["target_call_scope_id"]
        target_call_phase = request["target_call_phase"]
        if self.role == "target":
            if (
                not isinstance(target_logical_call_id, str)
                or _SHA256_RE.fullmatch(target_logical_call_id) is None
            ):
                raise WorkerProtocolError("target_logical_call_id must be a lowercase SHA-256")
            if target_call_scope_id is not None and (
                not isinstance(target_call_scope_id, str)
                or _SHA256_RE.fullmatch(target_call_scope_id) is None
            ):
                raise WorkerProtocolError(
                    "target_call_scope_id must be null or a lowercase SHA-256"
                )
            if target_call_phase not in _TARGET_RPC_PHASES:
                raise WorkerProtocolError("target_call_phase is invalid")
            if (target_call_scope_id is None) != (target_call_phase == "unscoped"):
                raise WorkerProtocolError("Target scope and lifecycle phase disagree")
        elif (
            target_logical_call_id is not None
            or target_call_scope_id is not None
            or target_call_phase is not None
        ):
            raise WorkerProtocolError("Draft generation may not carry Target RPC lifecycle IDs")
        output_fields = (
            {"latent", "sampled_hidden", "full_hidden"}
            if capture_hidden
            else {"latent"}
        )
        outputs = self._output_map(request["output_paths"], required=output_fields)
        history, history_paths = self._history(
            request["history_latent_paths"], expected_count=chunk_index
        )
        noise_path, initial_noise = self._load_input_tensor(
            request["initial_noise_path"], label="initial_noise_path", key="noise"
        )
        input_paths = set(history_paths) | {noise_path}
        if input_paths.intersection(item[0] for item in outputs.values()):
            raise WorkerProtocolError("an output path may not overwrite an input path")
        target_attempt_id: str | None = None
        if self.role == "target":
            assert isinstance(target_logical_call_id, str)
            target_attempt_id = self._new_target_attempt_id(target_logical_call_id)
            self._record_target_lifecycle(
                request=request,
                attempt_id=target_attempt_id,
                status="started",
                output_latent_path=outputs["latent"][0],
                history_chunk_count=len(history_paths),
                initial_noise_path=noise_path,
            )
        kwargs = {
            "prompt": prompt,
            "history_latents": history,
            "initial_noise": initial_noise,
            "internal_seed": internal_seed,
            "chunk_index": chunk_index,
        }
        if self.role == "draft":
            kwargs["capture_hidden"] = capture_hidden
            if getattr(self.backend, "supports_history_kv_widths", False):
                kwargs["history_kv_widths"] = tuple(widths) if widths is not None else None
            if branch_action is not None:
                kwargs["kv_branch_action"] = branch_action
        if early:
            from .early_worker import prepare, finish
            callback, observations = prepare(self, request)
            kwargs["early_should_replace"] = callback
        try:
            with stage("backend_call", cuda=True):
                result = self.backend.generate_chunk(**kwargs)
            if self.role == "draft" and isinstance(result, Mapping) and "kv_reuse" in result:
                from .draft_reuse_evidence import validate_reuse_evidence
                evidence = dict(result["kv_reuse"])
                if "history_file_sha256" in evidence:
                    raise WorkerExecutionError("backend may not supply worker-owned history file bindings")
                evidence["history_file_sha256"] = [
                    _sha256_file(_input_path(self.run_dir, path, label="reuse history")[1])[0]
                    for path in history_paths]
                validate_reuse_evidence(evidence, history_chunks=chunk_index,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    kv_branch_action=branch_action, history_kv_widths=widths)
                result = {**result, "kv_reuse": evidence}
            if early:
                if not isinstance(result, Mapping) or "kv_reuse" not in result:
                    raise WorkerExecutionError("early Draft omitted native reuse evidence")
                result, stopped_payload = finish(self, request, result, observations, history_paths, noise_path)
                if stopped_payload is not None:
                    return stopped_payload
            if branch_action is not None:
                if not (isinstance(result, Mapping)
                        and isinstance(result.get("kv_reuse"), Mapping)
                        and isinstance(result["kv_reuse"].get("branch_anchor"), Mapping)):
                    raise WorkerExecutionError("backend omitted requested branch-anchor evidence")
        except BaseException as error:
            raise self._poison(error) from error
        with stage("generation_finalize"):
            return self._finalize_generate_result(
                request=request,
                target_attempt_id=target_attempt_id,
                target_logical_call_id=target_logical_call_id,
                outputs=outputs,
                history=history,
                history_paths=history_paths,
                noise_path=noise_path,
                result=result,
                capture_hidden=capture_hidden,
                chunk_index=chunk_index,
            )

    def _generate_persistent(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {
            "prompt",
            "chunk_index",
            "internal_seed",
            "history_latent_paths",
            "initial_noise_path",
            "output_latent_path",
            "target_logical_call_id",
            "target_call_scope_id",
            "target_call_phase",
        }
        _exact_keys(request, expected, label="generate_persistent request")
        if self.role != "target":
            raise WorkerProtocolError("only Target supports persistent generation")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        chunk_index = _integer(request["chunk_index"], label="chunk_index")
        if chunk_index >= self._chunk_count:
            raise WorkerProtocolError(
                f"chunk_index must be in [0,{self._chunk_count - 1}]"
            )
        internal_seed = _integer(request["internal_seed"], label="internal_seed")
        logical_call_id = request["target_logical_call_id"]
        scope_id = request["target_call_scope_id"]
        phase = request["target_call_phase"]
        if not isinstance(logical_call_id, str) or _SHA256_RE.fullmatch(logical_call_id) is None:
            raise WorkerProtocolError("target_logical_call_id must be a lowercase SHA-256")
        if scope_id is not None and (
            not isinstance(scope_id, str) or _SHA256_RE.fullmatch(scope_id) is None
        ):
            raise WorkerProtocolError("target_call_scope_id must be null or a lowercase SHA-256")
        if phase not in _TARGET_RPC_PHASES or (scope_id is None) != (phase == "unscoped"):
            raise WorkerProtocolError("Target persistent scope/lifecycle phase is invalid")
        raw_history = request["history_latent_paths"]
        if (
            not isinstance(raw_history, list)
            or len(raw_history) != chunk_index
            or any(not isinstance(item, str) for item in raw_history)
            or len(set(raw_history)) != len(raw_history)
        ):
            raise WorkerProtocolError(
                "persistent history_latent_paths must bind every prior chunk exactly once"
            )
        history_paths: list[str] = []
        for index, raw_path in enumerate(raw_history):
            relative, _path = _input_path(
                self.run_dir,
                raw_path,
                label=f"history_latent_paths[{index}]",
            )
            history_paths.append(relative)
        noise_path, initial_noise = self._load_input_tensor(
            request["initial_noise_path"], label="initial_noise_path", key="noise"
        )
        output_relative, output_target = _output_path(
            self.run_dir,
            request["output_latent_path"],
            label="output_latent_path",
        )
        if output_relative in set(history_paths) | {noise_path}:
            raise WorkerProtocolError("persistent output may not overwrite an input")
        attempt_id = self._new_target_attempt_id(logical_call_id)
        lifecycle_request = {
            **dict(request),
            "history_latent_paths": history_paths,
        }
        self._record_target_lifecycle(
            request=lifecycle_request,
            attempt_id=attempt_id,
            status="started",
            output_latent_path=output_relative,
            history_chunk_count=len(history_paths),
            initial_noise_path=noise_path,
        )
        generate = getattr(self.backend, "generate_persistent_chunk", None)
        if not callable(generate):
            raise self._poison(
                WorkerExecutionError("Target backend lacks persistent generation support")
            )
        try:
            with stage("backend_call", cuda=True):
                result = generate(
                    prompt=prompt,
                    initial_noise=initial_noise,
                    internal_seed=internal_seed,
                    chunk_index=chunk_index,
                )
        except BaseException as error:
            raise self._poison(error) from error
        with stage("generation_finalize"):
            self._record_target_lifecycle(
                request=lifecycle_request,
                attempt_id=attempt_id,
                status="backend_returned",
                output_latent_path=output_relative,
                history_chunk_count=len(history_paths),
                initial_noise_path=noise_path,
            )
            staged: list[_StagedOutput] = []
            try:
                if not isinstance(result, Mapping) or not self.codec.is_tensor(result.get("latent")):
                    raise WorkerExecutionError(
                        "persistent Target backend did not return tensor 'latent'"
                    )
                normalized: dict[str, Any] = {}
                for field, value in result.items():
                    if not isinstance(field, str) or _FIELD_RE.fullmatch(field) is None:
                        raise WorkerExecutionError("persistent backend returned an unsafe field")
                    if self.codec.is_tensor(value):
                        if field != "latent":
                            raise WorkerExecutionError(
                                f"unexpected persistent tensor field {field!r}"
                            )
                        item = self._stage_tensor(
                            relative=output_relative,
                            target=output_target,
                            key="latent",
                            value=value,
                        )
                        staged.append(item)
                        normalized[field] = item.metadata
                    else:
                        normalized[field] = _json_value(
                            value, label=f"persistent_backend_result.{field}"
                        )
                kv_sha = normalized.get("kv_witness_sha256")
                if not isinstance(kv_sha, str) or _SHA256_RE.fullmatch(kv_sha) is None:
                    raise WorkerExecutionError("persistent Target omitted its KV witness digest")
                _publish_staged(staged)
                latent_metadata = normalized["latent"]
                assert isinstance(latent_metadata, Mapping)
                self._record_target_lifecycle(
                    request=lifecycle_request,
                    attempt_id=attempt_id,
                    status="completed",
                    output_latent_path=output_relative,
                    history_chunk_count=len(history_paths),
                    initial_noise_path=noise_path,
                    output_latent_sha256=latent_metadata.get("sha256"),
                    kv_witness_sha256=kv_sha,
                )
                return {
                    "chunk_index": chunk_index,
                    "history_chunk_count": len(history_paths),
                    "target_attempt_id": attempt_id,
                    "target_logical_call_id": logical_call_id,
                    "result": normalized,
                }
            except BaseException as error:
                _discard_staged(staged)
                raise self._poison(error) from error

    def _append_persistent_clean(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Commit an accepted Draft latent into the live Target session.

        This state transition is intentionally separate from Target generation:
        SDVG acceptance performs a clean-context Target forward but creates no
        new latent file.  The request therefore binds the already-published
        Draft latent by path and digest and returns the post-commit KV witness.
        """

        expected = self._COMMON_KEYS | {
            "prompt",
            "chunk_index",
            "history_latent_paths",
            "clean_latent_path",
        }
        _exact_keys(request, expected, label="append_persistent_clean request")
        if self.role != "target":
            raise WorkerProtocolError("only Target supports persistent clean commits")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        chunk_index = _integer(request["chunk_index"], label="chunk_index")
        if not 1 <= chunk_index < self._chunk_count:
            raise WorkerProtocolError(
                f"chunk_index must be in [1,{self._chunk_count - 1}]"
            )
        raw_history = request["history_latent_paths"]
        if (
            not isinstance(raw_history, list)
            or len(raw_history) != chunk_index
            or any(not isinstance(item, str) for item in raw_history)
            or len(set(raw_history)) != len(raw_history)
        ):
            raise WorkerProtocolError(
                "persistent history_latent_paths must bind every prior chunk exactly once"
            )
        history_paths = [
            _input_path(
                self.run_dir,
                raw_path,
                label=f"history_latent_paths[{index}]",
            )[0]
            for index, raw_path in enumerate(raw_history)
        ]
        clean_path, clean_latent = self._load_input_tensor(
            request["clean_latent_path"], label="clean_latent_path", key="latent"
        )
        if clean_path in set(history_paths):
            raise WorkerProtocolError("clean_latent_path may not alias Target history")
        _relative, clean_file = _input_path(
            self.run_dir, clean_path, label="clean_latent_path"
        )
        clean_sha256, clean_nbytes = _sha256_file(clean_file)
        append = getattr(self.backend, "append_persistent_clean_chunk", None)
        if not callable(append):
            raise self._poison(
                WorkerExecutionError("Target backend lacks persistent clean commit support")
            )
        try:
            result = append(
                prompt=prompt,
                clean_latent=clean_latent,
                chunk_index=chunk_index,
            )
        except BaseException as error:
            raise self._poison(error) from error
        if not isinstance(result, Mapping):
            raise self._poison(
                WorkerExecutionError("persistent clean commit returned no evidence")
            )
        normalized = _json_value(result, label="persistent_clean_commit_result")
        if not isinstance(normalized, dict):
            raise self._poison(
                WorkerExecutionError("persistent clean commit evidence is invalid")
            )
        kv_sha = normalized.get("kv_witness_sha256")
        if not isinstance(kv_sha, str) or _SHA256_RE.fullmatch(kv_sha) is None:
            raise self._poison(
                WorkerExecutionError("persistent clean commit omitted its KV witness digest")
            )
        return {
            "chunk_index": chunk_index,
            "history_chunk_count": len(history_paths),
            "clean_latent": {
                "path": clean_path,
                "sha256": clean_sha256,
                "nbytes": clean_nbytes,
            },
            "result": normalized,
        }

    def _generate_persistent_suffix(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run the exact three-step Target suffix of the T-Stitch schedule."""

        expected = self._COMMON_KEYS | {
            "prompt",
            "chunk_index",
            "internal_seed",
            "history_latent_paths",
            "noisy_latent_750_path",
            "renoise_500_path",
            "renoise_250_path",
            "output_latent_path",
            "target_logical_call_id",
            "target_call_scope_id",
            "target_call_phase",
        }
        _exact_keys(request, expected, label="generate_persistent_suffix request")
        if self.role != "target":
            raise WorkerProtocolError("only Target supports persistent suffix generation")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        chunk_index = _integer(request["chunk_index"], label="chunk_index")
        if chunk_index >= self._chunk_count:
            raise WorkerProtocolError(
                f"chunk_index must be in [0,{self._chunk_count - 1}]"
            )
        internal_seed = _integer(request["internal_seed"], label="internal_seed")
        logical_call_id = request["target_logical_call_id"]
        scope_id = request["target_call_scope_id"]
        phase = request["target_call_phase"]
        if not isinstance(logical_call_id, str) or _SHA256_RE.fullmatch(logical_call_id) is None:
            raise WorkerProtocolError("target_logical_call_id must be a lowercase SHA-256")
        if scope_id is not None and (
            not isinstance(scope_id, str) or _SHA256_RE.fullmatch(scope_id) is None
        ):
            raise WorkerProtocolError("target_call_scope_id must be null or a lowercase SHA-256")
        if phase not in _TARGET_RPC_PHASES or (scope_id is None) != (phase == "unscoped"):
            raise WorkerProtocolError("Target persistent suffix scope/lifecycle phase is invalid")
        raw_history = request["history_latent_paths"]
        if (
            not isinstance(raw_history, list)
            or len(raw_history) != chunk_index
            or any(not isinstance(item, str) for item in raw_history)
            or len(set(raw_history)) != len(raw_history)
        ):
            raise WorkerProtocolError(
                "persistent history_latent_paths must bind every prior chunk exactly once"
            )
        history_paths = [
            _input_path(
                self.run_dir,
                raw_path,
                label=f"history_latent_paths[{index}]",
            )[0]
            for index, raw_path in enumerate(raw_history)
        ]
        noisy_path, noisy_latent = self._load_input_tensor(
            request["noisy_latent_750_path"],
            label="noisy_latent_750_path",
            key="noisy_latent_750",
        )
        renoise_500_path, renoise_500 = self._load_input_tensor(
            request["renoise_500_path"], label="renoise_500_path", key="noise"
        )
        renoise_250_path, renoise_250 = self._load_input_tensor(
            request["renoise_250_path"], label="renoise_250_path", key="noise"
        )
        input_paths = {noisy_path, renoise_500_path, renoise_250_path}
        if len(input_paths) != 3 or input_paths.intersection(history_paths):
            raise WorkerProtocolError("persistent suffix tensor inputs must be distinct")
        output_relative, output_target = _output_path(
            self.run_dir,
            request["output_latent_path"],
            label="output_latent_path",
        )
        if output_relative in input_paths or output_relative in set(history_paths):
            raise WorkerProtocolError("persistent suffix output may not overwrite an input")
        attempt_id = self._new_target_attempt_id(logical_call_id)
        lifecycle_request = {
            **dict(request),
            "history_latent_paths": history_paths,
            "initial_noise_path": noisy_path,
        }
        self._record_target_lifecycle(
            request=lifecycle_request,
            attempt_id=attempt_id,
            status="started",
            output_latent_path=output_relative,
            history_chunk_count=len(history_paths),
            initial_noise_path=noisy_path,
        )
        generate = getattr(self.backend, "generate_persistent_suffix_chunk", None)
        if not callable(generate):
            raise self._poison(
                WorkerExecutionError("Target backend lacks persistent suffix support")
            )
        try:
            result = generate(
                prompt=prompt,
                noisy_latent_750=noisy_latent,
                renoise_500=renoise_500,
                renoise_250=renoise_250,
                chunk_index=chunk_index,
            )
        except BaseException as error:
            raise self._poison(error) from error
        self._record_target_lifecycle(
            request=lifecycle_request,
            attempt_id=attempt_id,
            status="backend_returned",
            output_latent_path=output_relative,
            history_chunk_count=len(history_paths),
            initial_noise_path=noisy_path,
        )
        staged: list[_StagedOutput] = []
        try:
            if not isinstance(result, Mapping) or not self.codec.is_tensor(result.get("latent")):
                raise WorkerExecutionError(
                    "persistent Target suffix did not return tensor 'latent'"
                )
            normalized: dict[str, Any] = {}
            for field, value in result.items():
                if not isinstance(field, str) or _FIELD_RE.fullmatch(field) is None:
                    raise WorkerExecutionError("persistent suffix returned an unsafe field")
                if self.codec.is_tensor(value):
                    if field != "latent":
                        raise WorkerExecutionError(
                            f"unexpected persistent suffix tensor field {field!r}"
                        )
                    item = self._stage_tensor(
                        relative=output_relative,
                        target=output_target,
                        key="latent",
                        value=value,
                    )
                    staged.append(item)
                    normalized[field] = item.metadata
                else:
                    normalized[field] = _json_value(
                        value, label=f"persistent_suffix_result.{field}"
                    )
            kv_sha = normalized.get("kv_witness_sha256")
            if not isinstance(kv_sha, str) or _SHA256_RE.fullmatch(kv_sha) is None:
                raise WorkerExecutionError("persistent Target suffix omitted its KV witness")
            _publish_staged(staged)
            latent_metadata = normalized["latent"]
            assert isinstance(latent_metadata, Mapping)
            self._record_target_lifecycle(
                request=lifecycle_request,
                attempt_id=attempt_id,
                status="completed",
                output_latent_path=output_relative,
                history_chunk_count=len(history_paths),
                initial_noise_path=noisy_path,
                output_latent_sha256=latent_metadata.get("sha256"),
                kv_witness_sha256=kv_sha,
            )
            return {
                "chunk_index": chunk_index,
                "history_chunk_count": len(history_paths),
                "target_attempt_id": attempt_id,
                "target_logical_call_id": logical_call_id,
                "result": normalized,
            }
        except BaseException as error:
            _discard_staged(staged)
            raise self._poison(error) from error

    def _generate_t_stitch_step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run and persist the single Draft timestep-1000 T-Stitch leg."""

        expected = self._COMMON_KEYS | {
            "prompt",
            "chunk_index",
            "internal_seed",
            "history_latent_paths",
            "initial_noise_path",
            "renoise_750_path",
            "output_paths",
        }
        _exact_keys(request, expected, label="generate_t_stitch_step request")
        if self.role != "draft":
            raise WorkerProtocolError("only Draft supports the T-Stitch first step")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerProtocolError("prompt must be non-empty text")
        chunk_index = _integer(request["chunk_index"], label="chunk_index")
        if chunk_index >= self._chunk_count:
            raise WorkerProtocolError(
                f"chunk_index must be in [0,{self._chunk_count - 1}]"
            )
        internal_seed = _integer(request["internal_seed"], label="internal_seed")
        history, history_paths = self._history(
            request["history_latent_paths"], expected_count=chunk_index
        )
        initial_path, initial_noise = self._load_input_tensor(
            request["initial_noise_path"], label="initial_noise_path", key="noise"
        )
        renoise_path, renoise_750 = self._load_input_tensor(
            request["renoise_750_path"], label="renoise_750_path", key="noise"
        )
        if initial_path == renoise_path or initial_path in set(history_paths) or renoise_path in set(history_paths):
            raise WorkerProtocolError("T-Stitch Draft tensor inputs must be distinct")
        outputs = self._output_map(
            request["output_paths"],
            required={"draft_clean_latent", "noisy_latent_750"},
        )
        output_paths = {relative for relative, _target in outputs.values()}
        if output_paths.intersection(set(history_paths) | {initial_path, renoise_path}):
            raise WorkerProtocolError("T-Stitch Draft output may not overwrite an input")
        generate = getattr(self.backend, "generate_t_stitch_step_chunk", None)
        if not callable(generate):
            raise self._poison(
                WorkerExecutionError("Draft backend lacks T-Stitch step support")
            )
        try:
            result = generate(
                prompt=prompt,
                history_latents=history,
                initial_noise=initial_noise,
                renoise_750=renoise_750,
                internal_seed=internal_seed,
                chunk_index=chunk_index,
            )
        except BaseException as error:
            raise self._poison(error) from error
        staged: list[_StagedOutput] = []
        try:
            if not isinstance(result, Mapping):
                raise WorkerExecutionError("T-Stitch Draft backend returned no result")
            normalized: dict[str, Any] = {}
            for field, value in result.items():
                if not isinstance(field, str) or _FIELD_RE.fullmatch(field) is None:
                    raise WorkerExecutionError("T-Stitch Draft returned an unsafe field")
                if self.codec.is_tensor(value):
                    if field not in outputs:
                        raise WorkerExecutionError(
                            f"unexpected T-Stitch Draft tensor field {field!r}"
                        )
                    relative, target = outputs[field]
                    item = self._stage_tensor(
                        relative=relative,
                        target=target,
                        key=field,
                        value=value,
                    )
                    staged.append(item)
                    normalized[field] = item.metadata
                else:
                    normalized[field] = _json_value(
                        value, label=f"t_stitch_draft_result.{field}"
                    )
            for field in outputs:
                if not isinstance(normalized.get(field), Mapping):
                    raise WorkerExecutionError(
                        f"T-Stitch Draft omitted tensor {field!r}"
                    )
            kv_sha = normalized.get("kv_witness_sha256")
            if not isinstance(kv_sha, str) or _SHA256_RE.fullmatch(kv_sha) is None:
                raise WorkerExecutionError("T-Stitch Draft omitted its KV witness digest")
            schedule = normalized.get("denoising_forward_trace")
            if (
                not isinstance(schedule, list)
                or len(schedule) != 1
                or not isinstance(schedule[0], Mapping)
                or float(schedule[0].get("timestep", float("nan"))) != 1000.0
            ):
                raise WorkerExecutionError("T-Stitch Draft did not prove one step at 1000")
            _publish_staged(staged)
            return {
                "chunk_index": chunk_index,
                "history_chunk_count": len(history_paths),
                "result": normalized,
            }
        except BaseException as error:
            _discard_staged(staged)
            raise self._poison(error) from error

    def _finalize_generate_result(
        self,
        *,
        request: Mapping[str, Any],
        target_attempt_id: str | None,
        target_logical_call_id: Any,
        outputs: Mapping[str, tuple[str, Path]],
        history: Any | None,
        history_paths: Sequence[str],
        noise_path: str,
        result: Any,
        capture_hidden: bool,
        chunk_index: int,
    ) -> dict[str, Any]:
        if self.role == "target":
            assert target_attempt_id is not None
            self._record_target_lifecycle(
                request=request,
                attempt_id=target_attempt_id,
                status="backend_returned",
                output_latent_path=outputs["latent"][0],
                history_chunk_count=len(history_paths),
                initial_noise_path=noise_path,
            )
        staged: list[_StagedOutput] = []
        try:
            if not isinstance(result, Mapping):
                raise WorkerExecutionError("backend.generate_chunk must return a mapping")
            if "boundary_velocity_witness" in result:
                raise WorkerExecutionError(
                    "backend may not supply the worker-owned boundary velocity witness"
                )
            if "latent" not in result or not self.codec.is_tensor(result["latent"]):
                raise WorkerExecutionError("backend.generate_chunk did not return tensor 'latent'")
            if capture_hidden:
                for field in ("sampled_hidden", "full_hidden"):
                    if field not in result or not self.codec.is_tensor(result[field]):
                        raise WorkerExecutionError(
                            f"capturing hidden state requires tensor {field!r}"
                        )
            normalized: dict[str, Any] = {}
            for field, value in result.items():
                if not isinstance(field, str) or _FIELD_RE.fullmatch(field) is None:
                    raise WorkerExecutionError("backend returned an unsafe field name")
                if self.codec.is_tensor(value):
                    if field not in outputs:
                        raise WorkerExecutionError(
                            f"backend tensor field {field!r} has no requested output path"
                        )
                    relative, target = outputs[field]
                    item = self._stage_tensor(
                        relative=relative, target=target, key=field, value=value
                    )
                    staged.append(item)
                    normalized[field] = item.metadata
                else:
                    normalized[field] = _json_value(value, label=f"backend_result.{field}")
            for field in outputs:
                if field not in normalized:
                    raise WorkerExecutionError(
                        f"backend result omitted requested tensor field {field!r}"
                    )
            candidate_shape, candidate_dtype = self.codec.describe(result["latent"])
            if history is None:
                boundary_velocity_witness = None
            else:
                history_shape, history_dtype = self.codec.describe(history)
                if history_dtype != candidate_dtype:
                    raise WorkerExecutionError(
                        "history and candidate latent dtypes differ at the boundary"
                    )
                raw_witness = self.codec.boundary_velocity_witness(
                    history=history,
                    candidate=result["latent"],
                )
                boundary_velocity_witness = _validated_boundary_velocity_witness(
                    raw_witness,
                    history_frame_count=history_shape[1],
                    candidate_frame_count=candidate_shape[1],
                    input_dtype=candidate_dtype,
                )
            normalized["boundary_velocity_witness"] = boundary_velocity_witness
            _publish_staged(staged)
            if self.role == "target":
                assert target_attempt_id is not None
                latent_metadata = normalized["latent"]
                kv_witness_sha256 = normalized.get("kv_witness_sha256")
                if not isinstance(latent_metadata, Mapping):
                    raise WorkerExecutionError("Target latent metadata is invalid")
                self._record_target_lifecycle(
                    request=request,
                    attempt_id=target_attempt_id,
                    status="completed",
                    output_latent_path=outputs["latent"][0],
                    history_chunk_count=len(history_paths),
                    initial_noise_path=noise_path,
                    output_latent_sha256=latent_metadata.get("sha256"),
                    kv_witness_sha256=kv_witness_sha256,
                )
            return {
                "chunk_index": chunk_index,
                "history_chunk_count": len(history_paths),
                "target_attempt_id": target_attempt_id,
                "target_logical_call_id": target_logical_call_id,
                "result": normalized,
            }
        except BaseException as error:
            _discard_staged(staged)
            raise self._poison(error) from error

    def _decode_video(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Encode the native pixels without a transient tensor-file round trip."""
        from .video_only_decode import pixel_witness
        _exact_keys(request, self._COMMON_KEYS | {
            "history_latent_paths", "output_video_path", "fps"}, label="video-only decode request")
        if self.role != "draft":
            raise WorkerProtocolError("only Draft owns the offline VAE decoder")
        history, paths = self._history(request['history_latent_paths'], expected_count=None)
        if history is None or not paths:
            raise WorkerProtocolError("video-only decode requires nonempty history")
        relative, target = _output_path(self.run_dir, request['output_video_path'], label='output_video_path')
        fps = _integer(request['fps'], label='fps', minimum=1)
        if fps > 240 or relative in paths:
            raise WorkerProtocolError("invalid video-only decode output")
        temporary = _temporary_sibling(target)
        staged = []
        try:
            frames = self.backend.decode_latents(history)
            with stage('pixel_content_hash'):
                pixels = pixel_witness(frames)
            self.backend.write_video(frames, pixels['shape'][0], temporary, fps)
            digest, size = _sha256_file(temporary)
            video = {'path': relative, 'sha256': digest, 'nbytes': size,
                     'fps': fps, 'frames': pixels['shape'][0]}
            staged.append(_StagedOutput(relative_path=relative, target=target,
                temporary=temporary, metadata=video))
            _publish_staged(staged)
            return {'latent_chunk_count': len(paths), 'pixels': pixels, 'video': video}
        except BaseException as error:
            _discard_staged(staged)
            try:
                temporary.unlink()
            except OSError:
                pass
            raise self._poison(error) from error

    def _decode(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {
            "history_latent_paths",
            "output_frames_path",
            "output_video_path",
            "fps",
        }
        _exact_keys(request, expected, label="decode request")
        if self.role != "draft":
            raise WorkerProtocolError("only the Draft worker owns the offline VAE decoder")
        history, history_paths = self._history(request["history_latent_paths"], expected_count=None)
        if history is None or not history_paths:
            raise WorkerProtocolError("decode requires at least one latent chunk")
        frames_relative, frames_target = _output_path(
            self.run_dir, request["output_frames_path"], label="output_frames_path"
        )
        video_raw = request["output_video_path"]
        video_output: tuple[str, Path] | None = None
        if video_raw is not None:
            video_output = _output_path(
                self.run_dir, video_raw, label="output_video_path"
            )
            if video_output[0] == frames_relative:
                raise WorkerProtocolError("frame and video outputs must be distinct")
        fps = _integer(request["fps"], label="fps", minimum=1)
        if fps > 240:
            raise WorkerProtocolError("fps must be <= 240")
        if frames_relative in set(history_paths) or (
            video_output is not None and video_output[0] in set(history_paths)
        ):
            raise WorkerProtocolError("decode output may not overwrite a latent input")
        try:
            frames = self.backend.decode_latents(history)
        except BaseException as error:
            raise self._poison(error) from error
        staged: list[_StagedOutput] = []
        try:
            if not self.codec.is_tensor(frames):
                raise WorkerExecutionError("backend.decode_latents must return a tensor")
            frame_stage = self._stage_tensor(
                relative=frames_relative,
                target=frames_target,
                key="frames",
                value=frames,
            )
            staged.append(frame_stage)
            shape, _dtype = self.codec.describe(frames)
            if len(shape) != 4 or shape[0] < 1:
                raise WorkerExecutionError("decoded frames must have shape [T,C,H,W]")
            video_metadata: dict[str, Any] | None = None
            if video_output is not None:
                if not hasattr(self.backend, "write_video"):
                    raise WorkerExecutionError("Draft backend does not implement write_video")
                video_relative, video_target = video_output
                video_temporary = _temporary_sibling(video_target)
                self.backend.write_video(frames, int(shape[0]), video_temporary, fps)
                digest, size = _sha256_file(video_temporary)
                video_metadata = {
                    "path": video_relative,
                    "sha256": digest,
                    "nbytes": size,
                    "fps": fps,
                    "frames": int(shape[0]),
                }
                staged.append(
                    _StagedOutput(
                        relative_path=video_relative,
                        target=video_target,
                        temporary=video_temporary,
                        metadata=video_metadata,
                    )
                )
            _publish_staged(staged)
            return {
                "latent_chunk_count": len(history_paths),
                "frames": frame_stage.metadata,
                "video": video_metadata,
            }
        except BaseException as error:
            _discard_staged(staged)
            raise self._poison(error) from error

    def _score(self, request: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._COMMON_KEYS | {"frames_path", "frame_indices", "temporal_window"}
        _exact_keys(request, expected, label="score request")
        if self.role != "draft":
            raise WorkerProtocolError("only the Draft worker owns quality evaluation")
        frames_path, frames = self._load_input_tensor(
            request["frames_path"], label="frames_path", key="frames"
        )
        indices_raw = request["frame_indices"]
        if (
            not isinstance(indices_raw, list)
            or not indices_raw
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in indices_raw
            )
            or indices_raw != sorted(set(indices_raw))
        ):
            raise WorkerProtocolError(
                "frame_indices must be a non-empty sorted list of unique non-negative integers"
            )
        temporal_raw = request["temporal_window"]
        temporal: tuple[int, int] | None = None
        if temporal_raw is not None:
            if not isinstance(temporal_raw, Mapping):
                raise WorkerProtocolError("temporal_window must be null or an object")
            _exact_keys(temporal_raw, {"start", "stop"}, label="temporal_window")
            start = _integer(temporal_raw["start"], label="temporal_window.start")
            stop = _integer(temporal_raw["stop"], label="temporal_window.stop")
            if stop - start < 2:
                raise WorkerProtocolError("temporal_window must contain at least two frames")
            temporal = (start, stop)
        try:
            scores_raw = self.backend.score_frame_indices(frames, indices_raw)
            scores = [float(value) for value in scores_raw]
            if len(scores) != len(indices_raw):
                raise WorkerExecutionError("quality backend did not return one score per frame")
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
                raise WorkerExecutionError("quality scores must be finite values in [0,1]")
            temporal_score: float | None = None
            if temporal is not None:
                temporal_score = float(
                    self.backend.temporal_consistency(
                        frames, start=temporal[0], stop=temporal[1]
                    )
                )
                if not math.isfinite(temporal_score) or not 0.0 <= temporal_score <= 1.0:
                    raise WorkerExecutionError(
                        "temporal consistency must be a finite value in [0,1]"
                    )
            return {
                "frames_path": frames_path,
                "frame_indices": list(indices_raw),
                "quality_scores": scores,
                "quality_mean": math.fsum(scores) / len(scores),
                "temporal_window": (
                    None if temporal is None else {"start": temporal[0], "stop": temporal[1]}
                ),
                "temporal_consistency": temporal_score,
            }
        except BaseException as error:
            raise self._poison(error) from error

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        close = getattr(self.backend, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as error:
                self.poisoned = True
                self.poison_reason = f"{type(error).__name__}: {error}"
                raise WorkerExecutionError(f"backend close failed: {error}") from error


def _response(
    *,
    request_id: str,
    action: str,
    role: str,
    ok: bool,
    payload: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
    poisoned: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": RESPONSE_SCHEMA,
        "request_id": request_id,
        "action": action,
        "role": role,
        "ok": ok,
        "worker_poisoned": poisoned,
    }
    if ok:
        value["payload"] = _json_value(payload or {}, label="response.payload")
    else:
        assert error is not None
        value["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "retryable": False,
        }
    return value


def _emit(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()


def serve_jsonl(
    *, worker: CorrectionWorker, input_stream: TextIO, output_stream: TextIO
) -> None:
    """Serve until a successful shutdown request or EOF."""

    _emit(
        output_stream,
        _response(
            request_id="worker-startup",
            action="ready",
            role=worker.role,
            ok=True,
            payload=worker.ready_payload(),
            poisoned=worker.poisoned,
        ),
    )
    try:
        for raw_line in input_stream:
            request_id = "unknown"
            action = "invalid"
            try:
                if len(raw_line.encode("utf-8")) > MAX_REQUEST_BYTES:
                    raise WorkerProtocolError(
                        f"request exceeds {MAX_REQUEST_BYTES} UTF-8 bytes"
                    )
                request = _parse_json_object(raw_line, label="request")
                if isinstance(request.get("request_id"), str):
                    request_id = request["request_id"][:256]
                if isinstance(request.get("action"), str):
                    action = request["action"][:64]
                request_id, action, payload = worker.handle(request)
                response = _response(
                    request_id=request_id,
                    action=action,
                    role=worker.role,
                    ok=True,
                    payload=payload,
                    poisoned=worker.poisoned,
                )
            except (WorkerProtocolError, WorkerPoisonedError) as error:
                response = _response(
                    request_id=request_id,
                    action=action,
                    role=worker.role,
                    ok=False,
                    error=error,
                    poisoned=worker.poisoned,
                )
            except BaseException as error:
                traceback.print_exc(file=sys.stderr)
                if not worker.poisoned:
                    worker._poison(error)
                response = _response(
                    request_id=request_id,
                    action=action,
                    role=worker.role,
                    ok=False,
                    error=error,
                    poisoned=True,
                )
            _emit(output_stream, response)
            if response["ok"] and response["action"] == "shutdown":
                return
    finally:
        if not worker.closed:
            try:
                worker.close()
            except WorkerError as error:
                print(f"correction worker close failure: {error}", file=sys.stderr, flush=True)


def _load_backend_factory(value: str) -> Callable[[Mapping[str, Any]], Any]:
    if value.count(":") != 1:
        raise WorkerProtocolError("--backend-factory must be MODULE:CALLABLE")
    module_name, attribute_name = value.split(":", 1)
    if not module_name or not attribute_name:
        raise WorkerProtocolError("--backend-factory must be MODULE:CALLABLE")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
    except (ImportError, AttributeError) as error:
        raise WorkerProtocolError(f"cannot import backend factory {value!r}: {error}") from error
    if not callable(factory):
        raise WorkerProtocolError("backend factory is not callable")
    return factory


def _load_backend_config(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise WorkerProtocolError("backend config may not be a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
        raw = resolved.read_bytes()
    except OSError as error:
        raise WorkerProtocolError(f"cannot read backend config: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise WorkerProtocolError("--backend-config-sha256 must be a lowercase SHA-256")
        if digest != expected_sha256:
            raise WorkerProtocolError("backend config SHA-256 mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise WorkerProtocolError("backend config is not UTF-8") from error
    value = _parse_json_object(text, label="backend config")
    value["_verified_config_sha256"] = digest
    return value


def _protocol_stream_and_redirect_stdout() -> TextIO:
    """Reserve original fd 1, then make all ordinary stdout writes go to stderr."""

    try:
        protocol_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except (OSError, io.UnsupportedOperation) as error:
        raise WorkerError(f"cannot isolate the JSONL protocol stream: {error}") from error
    # Rebind sys.stdout too; sys.__stdout__ now also targets redirected fd 1.
    sys.stdout = os.fdopen(os.dup(sys.stderr.fileno()), "w", buffering=1, encoding="utf-8")
    return os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent Insight 2 correction worker")
    parser.add_argument("--role", required=True, choices=sorted(MODEL_ROLES))
    parser.add_argument("--backend-factory", required=True, metavar="MODULE:CALLABLE")
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--backend-config-sha256")
    return parser


def _run_worker_main(args: argparse.Namespace, protocol: TextIO) -> int:
    role = str(args.role)
    try:
        raw_run_dir = os.environ.get("CFEXP_RUN_DIR")
        if not raw_run_dir:
            raise WorkerProtocolError("CFEXP_RUN_DIR is required")
        run_dir = _ensure_run_root(Path(raw_run_dir))
        config = _load_backend_config(args.backend_config, args.backend_config_sha256)
        factory = _load_backend_factory(args.backend_factory)
        backend = factory(config)
        worker = CorrectionWorker(role=role, backend=backend, run_dir=run_dir)
    except BaseException as error:
        _emit(
            protocol,
            _response(
                request_id="worker-startup",
                action="ready",
                role=role,
                ok=False,
                error=error,
                poisoned=True,
            ),
        )
        traceback.print_exc(file=sys.stderr)
        print(f"correction worker startup failed: {error}", file=sys.stderr, flush=True)
        return 2
    serve_jsonl(worker=worker, input_stream=sys.stdin, output_stream=protocol)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = _protocol_stream_and_redirect_stdout()
    result = 2
    try:
        result = _run_worker_main(args, protocol)
    finally:
        # This process is terminal, unlike CorrectionWorker.close() when used
        # in-process.  Reap library helpers before the controller's wait()
        # succeeds; waiting only for this Python PID cannot reap descendants.
        from .runtime_helpers import shutdown_owned_runtime_helpers

        try:
            shutdown_owned_runtime_helpers(label=f"worker-{args.role}")
        except BaseException as error:
            print(
                f"correction worker runtime helper shutdown failed: {type(error).__name__}",
                file=sys.stderr,
                flush=True,
            )
            result = 2
        finally:
            protocol.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
