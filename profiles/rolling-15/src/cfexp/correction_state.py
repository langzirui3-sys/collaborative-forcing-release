"""Fail-closed state contracts for Draft/Target latent correction.

The objects in this module deliberately know nothing about Torch or a model
runtime.  They bind opaque latent and initial-noise bytes to SHA-256 digests,
chain latent segments through a parent history digest, and make correction
transitions explicit.  A caller may therefore rebuild each model's KV cache
from the same clean latent history without ever passing a Draft cache to the
Target (or vice versa).

All state-bearing dataclasses are frozen.  Methods return new values instead
of mutating a baseline trajectory or an in-progress branch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


HISTORY_SCHEMA = "correction_latent_history_v1"
SEGMENT_SCHEMA = "correction_latent_segment_v1"
NOISE_SCHEMA = "correction_initial_noise_v1"
KV_SCHEMA = "correction_kv_rebuild_v1"
MODEL_ROLES = frozenset({"draft", "target"})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_NOISE_ALGORITHM = "sha256-counter-seed-v1"
_NOISE_DISTRIBUTION = "standard_normal"
_KV_CONSTRUCTION = "rebuild_from_clean_latent_history"


class CorrectionStateError(ValueError):
    """Raised when a correction state or transition violates the contract."""


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise CorrectionStateError(f"{path}: {message}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_sha256(value: Any, path: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, path,
             "must be a lowercase SHA-256")
    return value


def _validate_identifier(value: Any, path: str) -> str:
    _require(isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None, path,
             "must be a portable identifier")
    return value


def _validate_role(value: Any, path: str) -> str:
    _require(isinstance(value, str) and value in MODEL_ROLES, path,
             "must be 'draft' or 'target'")
    return value


def _validate_relative_path(value: Any, path: str) -> str:
    _require(isinstance(value, str) and value != "", path, "must be a non-empty string")
    _require("\\" not in value and "\x00" not in value, path,
             "must be a canonical POSIX relative path")
    pure = PurePosixPath(value)
    windows = PureWindowsPath(value)
    _require(not pure.is_absolute() and not windows.is_absolute() and not windows.drive, path,
             "must be relative")
    _require(str(pure) == value, path, "must be canonical")
    _require(all(part not in {"", ".", ".."} for part in pure.parts), path,
             "must not contain empty, '.' or '..' components")
    return value


def _validate_exact_keys(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    actual = set(value)
    _require(actual == keys, path,
             f"expected exactly keys {sorted(keys)!r}; got {sorted(actual)!r}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_object(raw: bytes, path: str) -> Mapping[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CorrectionStateError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except CorrectionStateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionStateError(f"{path}: invalid UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), path, "expected a JSON object")
    return value


def _sha256_regular_file(root: Path, relative_path: str) -> tuple[str, int]:
    """Hash a regular, non-symlink file confined below ``root``."""

    _validate_relative_path(relative_path, "$relative_path")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise CorrectionStateError(f"latent root is unavailable: {root}: {exc}") from exc
    _require(resolved_root.is_dir(), "$latent_root", "must be a directory")

    current = resolved_root
    for component in PurePosixPath(relative_path).parts:
        current = current / component
        try:
            info = current.lstat()
        except OSError as exc:
            raise CorrectionStateError(
                f"latent file is unavailable: {relative_path}: {exc}"
            ) from exc
        _require(not stat.S_ISLNK(info.st_mode), "$relative_path", "symlinks are forbidden")

    try:
        resolved_file = current.resolve(strict=True)
        resolved_file.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise CorrectionStateError(
            f"$relative_path: file escapes latent root: {relative_path}"
        ) from exc
    _require(resolved_file.is_file(), "$relative_path", "must name a regular file")

    digest = hashlib.sha256()
    try:
        with resolved_file.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            _require(stat.S_ISREG(opened.st_mode), "$relative_path", "must name a regular file")
            while block := handle.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise CorrectionStateError(f"cannot hash latent file {relative_path}: {exc}") from exc
    _require(
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "$relative_path",
        "file changed while hashing",
    )
    return digest.hexdigest(), after.st_size


def _noise_seed_payload(
    namespace: str, prompt_id: str, trajectory_seed: int, chunk_index: int
) -> dict[str, Any]:
    return {
        "algorithm": _NOISE_ALGORITHM,
        "namespace": namespace,
        "prompt_id": prompt_id,
        "trajectory_seed": trajectory_seed,
        "chunk_index": chunk_index,
    }


def derive_initial_noise_seed(
    *, namespace: str, prompt_id: str, trajectory_seed: int, chunk_index: int
) -> int:
    """Derive a stable non-negative 63-bit seed for one absolute chunk."""

    _validate_identifier(namespace, "$namespace")
    _validate_identifier(prompt_id, "$prompt_id")
    _require(_is_int(trajectory_seed) and 0 <= trajectory_seed < 2**63, "$trajectory_seed",
             "must be an integer in [0, 2**63)")
    _require(_is_int(chunk_index) and chunk_index >= 0, "$chunk_index",
             "must be a non-negative integer")
    digest = hashlib.sha256(
        _canonical_json_bytes(
            _noise_seed_payload(namespace, prompt_id, trajectory_seed, chunk_index)
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class InitialNoiseDescriptor:
    """Identity of the exact initial-noise tensor shared by both models."""

    namespace: str
    prompt_id: str
    trajectory_seed: int
    chunk_index: int
    derived_seed: int
    shape: tuple[int, ...]
    dtype: str
    tensor_sha256: str
    tensor_nbytes: int

    def __post_init__(self) -> None:
        _validate_identifier(self.namespace, "$.noise.namespace")
        _validate_identifier(self.prompt_id, "$.noise.prompt_id")
        _require(_is_int(self.trajectory_seed) and 0 <= self.trajectory_seed < 2**63,
                 "$.noise.trajectory_seed", "must be an integer in [0, 2**63)")
        _require(_is_int(self.chunk_index) and self.chunk_index >= 0, "$.noise.chunk_index",
                 "must be a non-negative integer")
        expected_seed = derive_initial_noise_seed(
            namespace=self.namespace,
            prompt_id=self.prompt_id,
            trajectory_seed=self.trajectory_seed,
            chunk_index=self.chunk_index,
        )
        _require(self.derived_seed == expected_seed, "$.noise.derived_seed",
                 "does not match deterministic derivation")
        _require(
            isinstance(self.shape, tuple)
            and bool(self.shape)
            and all(_is_int(item) and item > 0 for item in self.shape),
            "$.noise.shape",
            "must be a non-empty tuple of positive integers",
        )
        _require(isinstance(self.dtype, str) and self.dtype in _DTYPES, "$.noise.dtype",
                 f"must be one of {sorted(_DTYPES)!r}")
        _validate_sha256(self.tensor_sha256, "$.noise.tensor_sha256")
        _require(_is_int(self.tensor_nbytes) and self.tensor_nbytes > 0,
                 "$.noise.tensor_nbytes", "must be a positive integer")

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        prompt_id: str,
        trajectory_seed: int,
        chunk_index: int,
        shape: Sequence[int],
        dtype: str,
        tensor_sha256: str,
        tensor_nbytes: int,
    ) -> InitialNoiseDescriptor:
        shape_tuple = tuple(shape)
        return cls(
            namespace=namespace,
            prompt_id=prompt_id,
            trajectory_seed=trajectory_seed,
            chunk_index=chunk_index,
            derived_seed=derive_initial_noise_seed(
                namespace=namespace,
                prompt_id=prompt_id,
                trajectory_seed=trajectory_seed,
                chunk_index=chunk_index,
            ),
            shape=shape_tuple,
            dtype=dtype,
            tensor_sha256=tensor_sha256,
            tensor_nbytes=tensor_nbytes,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": NOISE_SCHEMA,
            "algorithm": _NOISE_ALGORITHM,
            "distribution": _NOISE_DISTRIBUTION,
            "namespace": self.namespace,
            "prompt_id": self.prompt_id,
            "trajectory_seed": self.trajectory_seed,
            "chunk_index": self.chunk_index,
            "derived_seed": self.derived_seed,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "tensor_sha256": self.tensor_sha256,
            "tensor_nbytes": self.tensor_nbytes,
        }

    @property
    def descriptor_sha256(self) -> str:
        return _payload_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["descriptor_sha256"] = self.descriptor_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Any, *, path: str = "$.noise") -> InitialNoiseDescriptor:
        _require(isinstance(value, dict), path, "expected an object")
        keys = {
            "schema", "algorithm", "distribution", "namespace", "prompt_id",
            "trajectory_seed", "chunk_index", "derived_seed", "shape", "dtype",
            "tensor_sha256", "tensor_nbytes", "descriptor_sha256",
        }
        _validate_exact_keys(value, keys, path)
        _require(value["schema"] == NOISE_SCHEMA, f"{path}.schema", f"expected {NOISE_SCHEMA!r}")
        _require(value["algorithm"] == _NOISE_ALGORITHM, f"{path}.algorithm",
                 f"expected {_NOISE_ALGORITHM!r}")
        _require(value["distribution"] == _NOISE_DISTRIBUTION, f"{path}.distribution",
                 f"expected {_NOISE_DISTRIBUTION!r}")
        _require(isinstance(value["shape"], list), f"{path}.shape", "expected a list")
        result = cls(
            namespace=value["namespace"],
            prompt_id=value["prompt_id"],
            trajectory_seed=value["trajectory_seed"],
            chunk_index=value["chunk_index"],
            derived_seed=value["derived_seed"],
            shape=tuple(value["shape"]),
            dtype=value["dtype"],
            tensor_sha256=value["tensor_sha256"],
            tensor_nbytes=value["tensor_nbytes"],
        )
        _require(value["descriptor_sha256"] == result.descriptor_sha256,
                 f"{path}.descriptor_sha256", "descriptor digest mismatch")
        return result


@dataclass(frozen=True)
class KVCacheDescriptor:
    """Evidence that a model rebuilt its own KV from a clean history head."""

    owner_role: str
    model_id: str
    model_artifact_sha256: str
    history_head_sha256: str
    cache_sha256: str

    def __post_init__(self) -> None:
        _validate_role(self.owner_role, "$.kv.owner_role")
        _validate_identifier(self.model_id, "$.kv.model_id")
        _validate_sha256(self.model_artifact_sha256, "$.kv.model_artifact_sha256")
        _validate_sha256(self.history_head_sha256, "$.kv.history_head_sha256")
        _validate_sha256(self.cache_sha256, "$.kv.cache_sha256")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": KV_SCHEMA,
            "construction": _KV_CONSTRUCTION,
            "owner_role": self.owner_role,
            "model_id": self.model_id,
            "model_artifact_sha256": self.model_artifact_sha256,
            "history_head_sha256": self.history_head_sha256,
            "cache_sha256": self.cache_sha256,
        }

    @property
    def descriptor_sha256(self) -> str:
        return _payload_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["descriptor_sha256"] = self.descriptor_sha256
        return payload

    def verify_for(
        self,
        *,
        role: str,
        model_id: str,
        model_artifact_sha256: str,
        history_head_sha256: str,
    ) -> None:
        _validate_role(role, "$role")
        _require(self.owner_role == role, "$.kv.owner_role",
                 "cross-model KV input is forbidden")
        _require(self.model_id == model_id, "$.kv.model_id",
                 "KV was built by a different model")
        _require(self.model_artifact_sha256 == model_artifact_sha256,
                 "$.kv.model_artifact_sha256", "KV model artifact mismatch")
        _require(self.history_head_sha256 == history_head_sha256,
                 "$.kv.history_head_sha256", "KV was built from a different history head")

    @classmethod
    def from_dict(cls, value: Any, *, path: str = "$.kv") -> KVCacheDescriptor:
        _require(isinstance(value, dict), path, "expected an object")
        keys = {
            "schema", "construction", "owner_role", "model_id",
            "model_artifact_sha256", "history_head_sha256", "cache_sha256",
            "descriptor_sha256",
        }
        _validate_exact_keys(value, keys, path)
        _require(value["schema"] == KV_SCHEMA, f"{path}.schema", f"expected {KV_SCHEMA!r}")
        _require(value["construction"] == _KV_CONSTRUCTION, f"{path}.construction",
                 f"expected {_KV_CONSTRUCTION!r}")
        result = cls(
            owner_role=value["owner_role"],
            model_id=value["model_id"],
            model_artifact_sha256=value["model_artifact_sha256"],
            history_head_sha256=value["history_head_sha256"],
            cache_sha256=value["cache_sha256"],
        )
        _require(value["descriptor_sha256"] == result.descriptor_sha256,
                 f"{path}.descriptor_sha256", "descriptor digest mismatch")
        return result


@dataclass(frozen=True)
class GenerationContext:
    """The model-local state allowed to produce one candidate segment."""

    role: str
    model_id: str
    model_artifact_sha256: str
    history_head_sha256: str
    next_chunk_index: int
    noise: InitialNoiseDescriptor
    kv: KVCacheDescriptor

    def __post_init__(self) -> None:
        _validate_role(self.role, "$.context.role")
        _validate_identifier(self.model_id, "$.context.model_id")
        _validate_sha256(self.model_artifact_sha256, "$.context.model_artifact_sha256")
        _validate_sha256(self.history_head_sha256, "$.context.history_head_sha256")
        _require(_is_int(self.next_chunk_index) and self.next_chunk_index >= 0,
                 "$.context.next_chunk_index", "must be a non-negative integer")
        _require(self.noise.chunk_index == self.next_chunk_index, "$.context.noise.chunk_index",
                 "must equal next_chunk_index")
        self.kv.verify_for(
            role=self.role,
            model_id=self.model_id,
            model_artifact_sha256=self.model_artifact_sha256,
            history_head_sha256=self.history_head_sha256,
        )

    def verify_history(self, history: LatentHistory) -> None:
        _require(self.history_head_sha256 == history.head_sha256,
                 "$.context.history_head_sha256", "does not match current history")
        _require(self.next_chunk_index == history.next_chunk_index,
                 "$.context.next_chunk_index", "does not match current history")
        _require(self.noise.prompt_id == history.prompt_id, "$.context.noise.prompt_id",
                 "does not match trajectory")
        _require(self.noise.trajectory_seed == history.trajectory_seed,
                 "$.context.noise.trajectory_seed", "does not match trajectory")


@dataclass(frozen=True)
class LatentSegment:
    """Content-addressed append-only latent segment manifest."""

    trajectory_id: str
    prompt_id: str
    trajectory_seed: int
    chunk_index: int
    parent_history_sha256: str
    producer_role: str
    model_id: str
    model_artifact_sha256: str
    initial_noise_descriptor_sha256: str
    kv_descriptor_sha256: str
    latent_path: str
    latent_sha256: str
    latent_nbytes: int

    def __post_init__(self) -> None:
        _validate_identifier(self.trajectory_id, "$.segment.trajectory_id")
        _validate_identifier(self.prompt_id, "$.segment.prompt_id")
        _require(_is_int(self.trajectory_seed) and 0 <= self.trajectory_seed < 2**63,
                 "$.segment.trajectory_seed", "must be an integer in [0, 2**63)")
        _require(_is_int(self.chunk_index) and self.chunk_index >= 0,
                 "$.segment.chunk_index", "must be a non-negative integer")
        _validate_sha256(self.parent_history_sha256, "$.segment.parent_history_sha256")
        _validate_role(self.producer_role, "$.segment.producer_role")
        _validate_identifier(self.model_id, "$.segment.model_id")
        _validate_sha256(self.model_artifact_sha256, "$.segment.model_artifact_sha256")
        _validate_sha256(self.initial_noise_descriptor_sha256,
                         "$.segment.initial_noise_descriptor_sha256")
        _validate_sha256(self.kv_descriptor_sha256, "$.segment.kv_descriptor_sha256")
        _validate_relative_path(self.latent_path, "$.segment.latent_path")
        _validate_sha256(self.latent_sha256, "$.segment.latent_sha256")
        _require(_is_int(self.latent_nbytes) and self.latent_nbytes > 0,
                 "$.segment.latent_nbytes", "must be a positive integer")

    @classmethod
    def from_file(
        cls,
        *,
        history: LatentHistory,
        context: GenerationContext,
        latent_root: str | Path,
        latent_path: str,
    ) -> LatentSegment:
        context.verify_history(history)
        digest, size = _sha256_regular_file(Path(latent_root), latent_path)
        return cls(
            trajectory_id=history.trajectory_id,
            prompt_id=history.prompt_id,
            trajectory_seed=history.trajectory_seed,
            chunk_index=history.next_chunk_index,
            parent_history_sha256=history.head_sha256,
            producer_role=context.role,
            model_id=context.model_id,
            model_artifact_sha256=context.model_artifact_sha256,
            initial_noise_descriptor_sha256=context.noise.descriptor_sha256,
            kv_descriptor_sha256=context.kv.descriptor_sha256,
            latent_path=latent_path,
            latent_sha256=digest,
            latent_nbytes=size,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SEGMENT_SCHEMA,
            "trajectory_id": self.trajectory_id,
            "prompt_id": self.prompt_id,
            "trajectory_seed": self.trajectory_seed,
            "chunk_index": self.chunk_index,
            "parent_history_sha256": self.parent_history_sha256,
            "producer_role": self.producer_role,
            "model_id": self.model_id,
            "model_artifact_sha256": self.model_artifact_sha256,
            "initial_noise_descriptor_sha256": self.initial_noise_descriptor_sha256,
            "kv_descriptor_sha256": self.kv_descriptor_sha256,
            "latent_path": self.latent_path,
            "latent_sha256": self.latent_sha256,
            "latent_nbytes": self.latent_nbytes,
        }

    @property
    def segment_sha256(self) -> str:
        return _payload_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["segment_sha256"] = self.segment_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Any, *, path: str = "$.segment") -> LatentSegment:
        _require(isinstance(value, dict), path, "expected an object")
        keys = {
            "schema", "trajectory_id", "prompt_id", "trajectory_seed", "chunk_index",
            "parent_history_sha256", "producer_role", "model_id", "model_artifact_sha256",
            "initial_noise_descriptor_sha256", "kv_descriptor_sha256", "latent_path",
            "latent_sha256", "latent_nbytes", "segment_sha256",
        }
        _validate_exact_keys(value, keys, path)
        _require(value["schema"] == SEGMENT_SCHEMA, f"{path}.schema",
                 f"expected {SEGMENT_SCHEMA!r}")
        result = cls(**{key: value[key] for key in keys - {"schema", "segment_sha256"}})
        _require(value["segment_sha256"] == result.segment_sha256,
                 f"{path}.segment_sha256", "segment digest mismatch")
        return result

    def verify_file(self, latent_root: str | Path) -> None:
        actual_sha256, actual_size = _sha256_regular_file(Path(latent_root), self.latent_path)
        _require(actual_sha256 == self.latent_sha256, "$.segment.latent_sha256",
                 "latent content digest mismatch")
        _require(actual_size == self.latent_nbytes, "$.segment.latent_nbytes",
                 "latent byte size mismatch")

    def verify_context(self, history: LatentHistory, context: GenerationContext) -> None:
        context.verify_history(history)
        _require(self.trajectory_id == history.trajectory_id, "$.segment.trajectory_id",
                 "does not match history")
        _require(self.prompt_id == history.prompt_id, "$.segment.prompt_id",
                 "does not match history")
        _require(self.trajectory_seed == history.trajectory_seed, "$.segment.trajectory_seed",
                 "does not match history")
        _require(self.chunk_index == history.next_chunk_index, "$.segment.chunk_index",
                 "does not match next history index")
        _require(self.parent_history_sha256 == history.head_sha256,
                 "$.segment.parent_history_sha256", "does not match history head")
        _require(self.producer_role == context.role, "$.segment.producer_role",
                 "does not match generation context")
        _require(self.model_id == context.model_id, "$.segment.model_id",
                 "does not match generation context")
        _require(self.model_artifact_sha256 == context.model_artifact_sha256,
                 "$.segment.model_artifact_sha256", "does not match generation context")
        _require(self.initial_noise_descriptor_sha256 == context.noise.descriptor_sha256,
                 "$.segment.initial_noise_descriptor_sha256",
                 "does not match generation context")
        _require(self.kv_descriptor_sha256 == context.kv.descriptor_sha256,
                 "$.segment.kv_descriptor_sha256", "does not match generation context")


@dataclass(frozen=True)
class LatentHistory:
    """An immutable contiguous chain of latent segments starting at chunk zero."""

    trajectory_id: str
    prompt_id: str
    trajectory_seed: int
    segments: tuple[LatentSegment, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.trajectory_id, "$.history.trajectory_id")
        _validate_identifier(self.prompt_id, "$.history.prompt_id")
        _require(_is_int(self.trajectory_seed) and 0 <= self.trajectory_seed < 2**63,
                 "$.history.trajectory_seed", "must be an integer in [0, 2**63)")
        _require(isinstance(self.segments, tuple), "$.history.segments", "must be a tuple")
        expected_parent = self.root_sha256
        for index, segment in enumerate(self.segments):
            path = f"$.history.segments[{index}]"
            _require(isinstance(segment, LatentSegment), path, "expected LatentSegment")
            _require(segment.trajectory_id == self.trajectory_id, f"{path}.trajectory_id",
                     "does not match history")
            _require(segment.prompt_id == self.prompt_id, f"{path}.prompt_id",
                     "does not match history")
            _require(segment.trajectory_seed == self.trajectory_seed,
                     f"{path}.trajectory_seed", "does not match history")
            _require(segment.chunk_index == index, f"{path}.chunk_index",
                     "segments must be contiguous and unique from index zero")
            _require(segment.parent_history_sha256 == expected_parent,
                     f"{path}.parent_history_sha256", "parent history digest mismatch")
            expected_parent = segment.segment_sha256

    @property
    def root_sha256(self) -> str:
        return _payload_sha256({
            "schema": "correction_latent_history_root_v1",
            "trajectory_id": self.trajectory_id,
            "prompt_id": self.prompt_id,
            "trajectory_seed": self.trajectory_seed,
        })

    @property
    def head_sha256(self) -> str:
        return self.segments[-1].segment_sha256 if self.segments else self.root_sha256

    @property
    def next_chunk_index(self) -> int:
        return len(self.segments)

    def append(self, segment: LatentSegment) -> LatentHistory:
        _require(isinstance(segment, LatentSegment), "$segment", "expected LatentSegment")
        _require(segment.chunk_index == self.next_chunk_index, "$.segment.chunk_index",
                 f"expected unique next index {self.next_chunk_index}")
        _require(segment.parent_history_sha256 == self.head_sha256,
                 "$.segment.parent_history_sha256", "parent history digest mismatch")
        return replace(self, segments=self.segments + (segment,))

    def prefix_before(self, chunk_index: int) -> LatentHistory:
        _require(_is_int(chunk_index) and 0 <= chunk_index <= self.next_chunk_index,
                 "$chunk_index", f"must be in [0, {self.next_chunk_index}]")
        return replace(self, segments=self.segments[:chunk_index])

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": HISTORY_SCHEMA,
            "trajectory_id": self.trajectory_id,
            "prompt_id": self.prompt_id,
            "trajectory_seed": self.trajectory_seed,
            "root_sha256": self.root_sha256,
            "head_sha256": self.head_sha256,
            "next_chunk_index": self.next_chunk_index,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @property
    def manifest_sha256(self) -> str:
        return _payload_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Any, *, path: str = "$.history") -> LatentHistory:
        _require(isinstance(value, dict), path, "expected an object")
        keys = {
            "schema", "trajectory_id", "prompt_id", "trajectory_seed", "root_sha256",
            "head_sha256", "next_chunk_index", "segments", "manifest_sha256",
        }
        _validate_exact_keys(value, keys, path)
        _require(value["schema"] == HISTORY_SCHEMA, f"{path}.schema",
                 f"expected {HISTORY_SCHEMA!r}")
        _require(isinstance(value["segments"], list), f"{path}.segments", "expected a list")
        segments = tuple(
            LatentSegment.from_dict(item, path=f"{path}.segments[{index}]")
            for index, item in enumerate(value["segments"])
        )
        result = cls(
            trajectory_id=value["trajectory_id"],
            prompt_id=value["prompt_id"],
            trajectory_seed=value["trajectory_seed"],
            segments=segments,
        )
        _require(value["root_sha256"] == result.root_sha256, f"{path}.root_sha256",
                 "root digest mismatch")
        _require(value["head_sha256"] == result.head_sha256, f"{path}.head_sha256",
                 "head digest mismatch")
        _require(value["next_chunk_index"] == result.next_chunk_index,
                 f"{path}.next_chunk_index", "next index mismatch")
        _require(value["manifest_sha256"] == result.manifest_sha256,
                 f"{path}.manifest_sha256", "manifest digest mismatch")
        return result

    def verify_files(self, latent_root: str | Path) -> None:
        for segment in self.segments:
            segment.verify_file(latent_root)


def write_history_manifest(path: str | Path, history: LatentHistory) -> None:
    """Write a new manifest without ever overwriting an existing object."""

    destination = Path(path)
    _require(destination.parent.is_dir(), "$path", "parent directory must already exist")
    raw = _canonical_json_bytes(history.to_dict()) + b"\n"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise CorrectionStateError(f"cannot create history manifest {destination}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def load_history_manifest(
    path: str | Path, *, latent_root: str | Path | None = None
) -> LatentHistory:
    """Load a strict manifest and optionally verify all referenced latent bytes."""

    source = Path(path)
    try:
        _require(not source.is_symlink(), "$path", "manifest symlinks are forbidden")
        raw = source.read_bytes()
    except OSError as exc:
        raise CorrectionStateError(f"cannot read history manifest {source}: {exc}") from exc
    history = LatentHistory.from_dict(_strict_json_object(raw, str(source)), path="$.history")
    if latent_root is not None:
        history.verify_files(latent_root)
    return history


@dataclass(frozen=True)
class SinglePointFork:
    """An isolated 2(a) fork from a fixed baseline prefix."""

    baseline: LatentHistory
    chunk_index: int
    baseline_manifest_sha256: str

    @classmethod
    def open(cls, baseline: LatentHistory, chunk_index: int) -> SinglePointFork:
        _require(_is_int(chunk_index) and 0 <= chunk_index < baseline.next_chunk_index,
                 "$chunk_index", "must identify an existing baseline segment")
        _require(baseline.segments[chunk_index].producer_role == "draft",
                 "$chunk_index", "single-point correction requires a Draft baseline chunk")
        return cls(
            baseline=baseline,
            chunk_index=chunk_index,
            baseline_manifest_sha256=baseline.manifest_sha256,
        )

    def __post_init__(self) -> None:
        _require(self.baseline_manifest_sha256 == self.baseline.manifest_sha256,
                 "$.fork.baseline_manifest_sha256", "baseline changed")
        _require(_is_int(self.chunk_index) and 0 <= self.chunk_index < len(self.baseline.segments),
                 "$.fork.chunk_index", "does not identify a baseline segment")
        _require(self.baseline.segments[self.chunk_index].producer_role == "draft",
                 "$.fork.chunk_index", "single-point correction requires a Draft baseline chunk")

    @property
    def prefix(self) -> LatentHistory:
        return self.baseline.prefix_before(self.chunk_index)

    @property
    def original_draft(self) -> LatentSegment:
        return self.baseline.segments[self.chunk_index]

    def commit_draft(
        self, *, context: GenerationContext, segment: LatentSegment
    ) -> LatentHistory:
        """Commit branch A only if replay exactly reproduces the baseline Draft."""

        _require(context.role == "draft", "$.context.role", "branch A must use Draft")
        segment.verify_context(self.prefix, context)
        _require(segment.segment_sha256 == self.original_draft.segment_sha256,
                 "$.segment.segment_sha256", "Draft replay differs from baseline")
        return self.prefix.append(segment)

    def commit_target(
        self, *, context: GenerationContext, segment: LatentSegment
    ) -> LatentHistory:
        """Commit branch B from the same prefix and exact initial noise as branch A."""

        _require(context.role == "target", "$.context.role", "branch B must use Target")
        segment.verify_context(self.prefix, context)
        _require(
            segment.initial_noise_descriptor_sha256
            == self.original_draft.initial_noise_descriptor_sha256,
            "$.segment.initial_noise_descriptor_sha256",
            "Target branch must use the baseline Draft's exact initial noise",
        )
        return self.prefix.append(segment)


@dataclass(frozen=True)
class CorrectionState:
    """Immutable 2(b) stage/accept-or-replace state machine."""

    history: LatentHistory
    staged_draft: LatentSegment | None = None
    staged_noise_descriptor_sha256: str | None = None

    def __post_init__(self) -> None:
        _require((self.staged_draft is None) == (self.staged_noise_descriptor_sha256 is None),
                 "$.state", "staged segment and noise digest must appear together")
        if self.staged_draft is not None:
            _require(self.staged_draft.producer_role == "draft", "$.state.staged_draft",
                     "only a Draft candidate may be staged")
            _require(self.staged_draft.chunk_index == self.history.next_chunk_index,
                     "$.state.staged_draft.chunk_index", "does not match next history index")
            _require(self.staged_draft.parent_history_sha256 == self.history.head_sha256,
                     "$.state.staged_draft.parent_history_sha256",
                     "does not match history head")
            _require(
                self.staged_draft.initial_noise_descriptor_sha256
                == self.staged_noise_descriptor_sha256,
                "$.state.staged_noise_descriptor_sha256",
                "does not match staged Draft",
            )

    def stage_draft(
        self, *, context: GenerationContext, segment: LatentSegment
    ) -> CorrectionState:
        _require(self.staged_draft is None, "$.state.staged_draft",
                 "resolve the current candidate before staging another")
        _require(context.role == "draft", "$.context.role", "must stage a Draft candidate")
        segment.verify_context(self.history, context)
        return replace(
            self,
            staged_draft=segment,
            staged_noise_descriptor_sha256=context.noise.descriptor_sha256,
        )

    def accept_draft(self) -> CorrectionState:
        _require(self.staged_draft is not None, "$.state.staged_draft",
                 "no Draft candidate is staged")
        return CorrectionState(history=self.history.append(self.staged_draft))

    def replace_with_target(
        self, *, context: GenerationContext, segment: LatentSegment
    ) -> CorrectionState:
        _require(self.staged_draft is not None, "$.state.staged_draft",
                 "a Draft candidate must be staged before replacement")
        _require(context.role == "target", "$.context.role", "replacement must use Target")
        segment.verify_context(self.history, context)
        _require(context.noise.descriptor_sha256 == self.staged_noise_descriptor_sha256,
                 "$.context.noise", "Target replacement must reuse staged Draft initial noise")
        _require(segment.initial_noise_descriptor_sha256 == self.staged_noise_descriptor_sha256,
                 "$.segment.initial_noise_descriptor_sha256",
                 "Target replacement must reuse staged Draft initial noise")
        return CorrectionState(history=self.history.append(segment))
