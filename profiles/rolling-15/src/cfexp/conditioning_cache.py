"""Immutable shared UMT5 prompt-conditioning cache for Insight 2.

The production Draft and Target implementations use the same frozen UMT5
checkpoint, tokenizer files, and implementation files.  This module provides a
small fail-closed boundary that lets the Draft worker compute each frozen
prompt embedding once, persist the exact bfloat16 tensors, and lets both model
workers consume those tensors without retaining an UMT5 model.

Torch and safetensors are deliberately imported lazily.  The manifest logic is
therefore unit-testable on CPU-only control hosts with an injected tensor
codec.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid


CONDITIONING_CONFIG_SCHEMA = "insight2_shared_umt5_conditioning_config_v1"
CONDITIONING_MANIFEST_SCHEMA = "insight2_shared_umt5_conditioning_manifest_v1"
CONDITIONING_PROVENANCE_SCHEMA = "insight2_shared_umt5_provenance_v1"
CONDITIONING_DTYPE = "bfloat16"
CONDITIONING_SHAPE = (1, 512, 4096)
CONDITIONING_IMPLEMENTATION_PATHS = (
    "wan/modules/t5.py",
    "wan/modules/tokenizers.py",
)
CONDITIONING_TENSOR_DIGEST_ALGORITHM = (
    "sha256(canonical_json({dtype,shape})||contiguous_uint8_tensor_bytes)"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ConditioningCacheError(RuntimeError):
    """The shared prompt-conditioning artifact is absent or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConditioningCacheError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConditioningCacheError(f"value is not canonical JSON: {error}") from error


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
            size = handle.tell()
    except OSError as error:
        raise ConditioningCacheError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest(), size


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ConditioningCacheError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ConditioningCacheError(f"invalid JSON constant {item!r} in {path}")
            ),
        )
    except ConditioningCacheError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConditioningCacheError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )


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


def _run_path(run_dir: Path, relative: str, *, must_exist: bool) -> Path:
    relative = _relative_path(relative, label="conditioning artifact path")
    root = run_dir.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        _require(not current.is_symlink(), "conditioning artifact path traverses a symlink")
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except FileNotFoundError as error:
        raise ConditioningCacheError(f"conditioning cache miss: {relative}") from error
    except (OSError, ValueError) as error:
        raise ConditioningCacheError(
            f"conditioning artifact escapes run directory: {relative}"
        ) from error
    if must_exist:
        _require(resolved.is_file(), f"conditioning cache miss: {relative}")
    return resolved


def canonical_prompt_records(values: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    """Validate and canonically bind the exact frozen prompt order."""

    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and bool(values),
        "conditioning prompts must be a non-empty sequence",
    )
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for index, raw in enumerate(values):
        _require(isinstance(raw, Mapping), f"conditioning prompt {index} is not an object")
        _exact_keys(raw, {"prompt_id", "prompt"}, label=f"conditioning prompt {index}")
        prompt_id = raw["prompt_id"]
        prompt = raw["prompt"]
        _require(
            isinstance(prompt_id, str) and _PROMPT_ID_RE.fullmatch(prompt_id) is not None,
            f"conditioning prompt {index} has an invalid prompt_id",
        )
        _require(
            isinstance(prompt, str) and bool(prompt.strip()),
            f"conditioning prompt {index} has empty text",
        )
        _require(prompt_id not in seen_ids, f"duplicate conditioning prompt_id {prompt_id!r}")
        _require(prompt not in seen_text, "conditioning prompt text must be unique")
        seen_ids.add(prompt_id)
        seen_text.add(prompt)
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return tuple(records)


def prompt_set_sha256(records: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(_canonical_bytes(list(records))).hexdigest()


def validate_conditioning_config(
    raw: Mapping[str, Any], *, run_dir: str | Path, role: str
) -> dict[str, Any]:
    """Validate the worker-bound cache request without importing Torch."""

    _require(isinstance(raw, Mapping), "conditioning_cache must be an object")
    _exact_keys(
        raw,
        {
            "schema",
            "mode",
            "run_dir",
            "manifest_path",
            "tensor_path",
            "prompt_set_sha256",
            "prompts",
            "provenance",
        },
        label="conditioning_cache",
    )
    _require(raw["schema"] == CONDITIONING_CONFIG_SCHEMA, "conditioning schema is invalid")
    expected_mode = "producer" if role == "draft" else "consumer"
    _require(raw["mode"] == expected_mode, f"{role} conditioning mode must be {expected_mode}")
    root = Path(run_dir).expanduser().resolve(strict=True)
    _require(
        Path(str(raw["run_dir"])).expanduser().resolve(strict=True) == root,
        "conditioning run_dir disagrees with backend run_dir",
    )
    manifest_relative = _relative_path(raw["manifest_path"], label="manifest_path")
    tensor_relative = _relative_path(raw["tensor_path"], label="tensor_path")
    _require(manifest_relative != tensor_relative, "manifest and tensor paths must differ")
    records = canonical_prompt_records(raw["prompts"])
    expected_set = prompt_set_sha256(records)
    _require(
        raw["prompt_set_sha256"] == expected_set,
        "conditioning prompt_set_sha256 does not bind the prompt records",
    )
    provenance = raw["provenance"]
    _require(isinstance(provenance, Mapping), "conditioning provenance must be an object")
    _exact_keys(
        provenance,
        {
            "schema",
            "dtype",
            "encoder_implementation",
            "draft_source_commit",
            "target_source_commit",
            "draft_source_archive_sha256",
            "target_source_archive_sha256",
            "text_encoder_checkpoint_sha256",
            "tokenizer_artifacts",
        },
        label="conditioning provenance",
    )
    _require(
        provenance["schema"] == CONDITIONING_PROVENANCE_SCHEMA,
        "conditioning provenance schema is invalid",
    )
    _require(provenance["dtype"] == CONDITIONING_DTYPE, "conditioning dtype must be bfloat16")
    _require(
        provenance["encoder_implementation"] == "wan.modules.t5.umt5_xxl",
        "conditioning encoder implementation drifted",
    )
    for field in (
        "draft_source_archive_sha256",
        "target_source_archive_sha256",
        "text_encoder_checkpoint_sha256",
    ):
        _require(
            isinstance(provenance[field], str)
            and _SHA256_RE.fullmatch(provenance[field]) is not None,
            f"conditioning provenance {field} is not a SHA-256",
        )
    for field in ("draft_source_commit", "target_source_commit"):
        _require(
            isinstance(provenance[field], str) and re.fullmatch(r"[0-9a-f]{40}", provenance[field]),
            f"conditioning provenance {field} is not a full commit",
        )
    tokenizer_rows = provenance["tokenizer_artifacts"]
    _require(isinstance(tokenizer_rows, list) and len(tokenizer_rows) == 4,
             "conditioning provenance must bind four tokenizer files")
    tokenizer_names: list[str] = []
    for index, item in enumerate(tokenizer_rows):
        _require(isinstance(item, Mapping), f"tokenizer artifact {index} is not an object")
        _exact_keys(item, {"name", "sha256"}, label=f"tokenizer artifact {index}")
        _require(isinstance(item["name"], str) and bool(item["name"]),
                 f"tokenizer artifact {index} has an invalid name")
        _require(isinstance(item["sha256"], str) and _SHA256_RE.fullmatch(item["sha256"]),
                 f"tokenizer artifact {index} has an invalid SHA-256")
        tokenizer_names.append(str(item["name"]))
    _require(tokenizer_names == sorted(tokenizer_names) and len(set(tokenizer_names)) == 4,
             "tokenizer artifacts must be unique and sorted")
    result = dict(raw)
    result["prompts"] = records
    result["provenance"] = dict(provenance)
    result["manifest_file"] = _run_path(root, manifest_relative, must_exist=False)
    result["tensor_file"] = _run_path(root, tensor_relative, must_exist=False)
    result["run_dir"] = root
    return result


class ConditioningTensorCodec(Protocol):
    def normalize(self, value: Any) -> Any: ...

    def describe(self, value: Any) -> tuple[tuple[int, ...], str]: ...

    def tensor_sha256(self, value: Any) -> str: ...

    def save(self, path: Path, tensors: Mapping[str, Any]) -> None: ...

    def load(self, path: Path) -> Mapping[str, Any]: ...

    def equal(self, left: Any, right: Any) -> bool: ...


class TorchConditioningTensorCodec:
    """Production bfloat16 safetensors codec, imported only in GPU workers."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        try:
            from safetensors.torch import load_file, save_file  # type: ignore
        except ImportError as error:
            raise ConditioningCacheError("conditioning cache requires safetensors") from error
        self._load_file = load_file
        self._save_file = save_file

    def normalize(self, value: Any) -> Any:
        _require(isinstance(value, self.torch.Tensor), "prompt embedding is not a tensor")
        tensor = value.detach().to(device="cpu", dtype=self.torch.bfloat16).contiguous()
        _require(bool(self.torch.isfinite(tensor.float()).all().item()),
                 "prompt embedding contains non-finite values")
        return tensor

    def describe(self, value: Any) -> tuple[tuple[int, ...], str]:
        _require(isinstance(value, self.torch.Tensor), "cached prompt embedding is not a tensor")
        dtype = str(value.dtype)
        if dtype.startswith("torch."):
            dtype = dtype[len("torch.") :]
        return tuple(int(item) for item in value.shape), dtype

    def tensor_sha256(self, value: Any) -> str:
        tensor = self.normalize(value)
        shape, dtype = self.describe(tensor)
        digest = hashlib.sha256(_canonical_bytes({"dtype": dtype, "shape": list(shape)}))
        digest.update(tensor.view(self.torch.uint8).numpy().tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: Path, tensors: Mapping[str, Any]) -> None:
        ordered = {key: self.normalize(tensors[key]) for key in sorted(tensors)}
        self._save_file(ordered, str(path))

    def load(self, path: Path) -> Mapping[str, Any]:
        return self._load_file(str(path), device="cpu")

    def equal(self, left: Any, right: Any) -> bool:
        return bool(self.torch.equal(left, right))


def _source_implementation(source_root: Path) -> dict[str, str]:
    root = source_root.resolve(strict=True)
    result: dict[str, str] = {}
    for relative in CONDITIONING_IMPLEMENTATION_PATHS:
        path = root.joinpath(*PurePosixPath(relative).parts)
        _require(not path.is_symlink(), f"conditioning source file is a symlink: {relative}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise ConditioningCacheError(
                f"conditioning source file is unavailable: {relative}"
            ) from error
        _require(resolved.is_file(), f"conditioning source file is not regular: {relative}")
        result[relative] = _sha256_file(resolved)[0]
    return result


def _publish_immutable_file(temporary: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(not target.parent.is_symlink(), "conditioning cache parent may not be a symlink")
    expected = _sha256_file(temporary)
    try:
        os.link(temporary, target)
    except FileExistsError:
        _require(not target.is_symlink() and target.is_file(),
                 f"immutable conditioning artifact is not regular: {target}")
        _require(_sha256_file(target) == expected,
                 f"immutable conditioning artifact conflicts: {target}")
    except OSError as error:
        raise ConditioningCacheError(f"cannot publish conditioning artifact {target}: {error}") from error
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ConditioningCacheError(f"cannot verify {path}: {error}") from error
        _require(existing == raw, f"immutable conditioning manifest conflicts: {path}")
        return
    except OSError as error:
        raise ConditioningCacheError(f"cannot create {path}: {error}") from error
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


@dataclass(frozen=True)
class LoadedConditioningCache:
    manifest: Mapping[str, Any]
    tensors: Mapping[str, Any]
    manifest_path: str

    def audit_binding(self) -> dict[str, Any]:
        tensor_file = self.manifest["tensor_file"]
        return {
            "schema": CONDITIONING_MANIFEST_SCHEMA,
            "manifest_path": self.manifest_path,
            "tensor_file_path": tensor_file["path"],
            "prompt_set_sha256": self.manifest["prompt_set_sha256"],
            "prompt_count": len(self.manifest["prompts"]),
            "manifest_sha256": hashlib.sha256(
                _canonical_bytes(dict(self.manifest)) + b"\n"
            ).hexdigest(),
            "tensor_file_sha256": tensor_file["sha256"],
            "tensor_file_nbytes": tensor_file["nbytes"],
            "dtype": CONDITIONING_DTYPE,
            "live_roundtrip_bitwise_equal": self.manifest["equivalence"]["all_equal"],
            "provenance": self.manifest["provenance"],
            "implementation_files": self.manifest["implementation_files"],
        }


def seal_shared_conditioning_cache(
    raw_config: Mapping[str, Any],
    *,
    run_dir: str | Path,
    source_root: str | Path,
    live_encode: Callable[[Sequence[str]], Mapping[str, Any]],
    codec: ConditioningTensorCodec,
) -> LoadedConditioningCache:
    """Encode every prompt once and seal an exact, round-trip-verified cache."""

    config = validate_conditioning_config(raw_config, run_dir=run_dir, role="draft")
    implementation = _source_implementation(Path(source_root))
    tensors: dict[str, Any] = {}
    prompt_rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(config["prompts"]):
        result = live_encode([prompt["prompt"]])
        _require(isinstance(result, Mapping) and set(result) == {"prompt_embeds"},
                 "live UMT5 encoder must return only prompt_embeds")
        tensor = codec.normalize(result["prompt_embeds"])
        shape, dtype = codec.describe(tensor)
        _require(shape == CONDITIONING_SHAPE, f"UMT5 embedding shape is {shape}")
        _require(dtype == CONDITIONING_DTYPE, f"UMT5 embedding dtype is {dtype}")
        key = f"prompt_{index:04d}"
        tensor_digest = codec.tensor_sha256(tensor)
        tensors[key] = tensor
        prompt_rows.append(
            {
                **dict(prompt),
                "tensor_key": key,
                "shape": list(shape),
                "dtype": dtype,
                "tensor_sha256": tensor_digest,
            }
        )

    tensor_path: Path = config["tensor_file"]
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_name(
        f".{tensor_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        codec.save(temporary, tensors)
        temporary.chmod(0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        staged = codec.load(temporary)
        _require(set(staged) == set(tensors), "staged conditioning tensor keys drifted")
        for key, tensor in tensors.items():
            _require(codec.equal(tensor, staged[key]),
                     f"conditioning tensor {key} changed during safetensors round trip")
        _publish_immutable_file(temporary, tensor_path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise

    tensor_sha256, tensor_nbytes = _sha256_file(tensor_path)
    manifest = {
        "schema": CONDITIONING_MANIFEST_SCHEMA,
        "prompt_set_sha256": config["prompt_set_sha256"],
        "prompts": prompt_rows,
        "provenance": dict(config["provenance"]),
        "implementation_files": implementation,
        "tensor_file": {
            "path": raw_config["tensor_path"],
            "sha256": tensor_sha256,
            "nbytes": tensor_nbytes,
            "format": "safetensors",
            "dtype": CONDITIONING_DTYPE,
            "tensor_digest_algorithm": CONDITIONING_TENSOR_DIGEST_ALGORITHM,
        },
        "equivalence": {
            "method": "exact_tensor_sha256_and_codec_roundtrip_equality",
            "prompt_count": len(prompt_rows),
            "all_equal": True,
        },
    }
    _write_immutable_json(config["manifest_file"], manifest)
    return load_shared_conditioning_cache(
        raw_config,
        run_dir=run_dir,
        source_root=source_root,
        role="draft",
        codec=codec,
    )


def load_shared_conditioning_cache(
    raw_config: Mapping[str, Any],
    *,
    run_dir: str | Path,
    source_root: str | Path,
    role: str,
    codec: ConditioningTensorCodec,
) -> LoadedConditioningCache:
    """Load and fully re-verify a shared cache; cache misses never fall back."""

    config = validate_conditioning_config(raw_config, run_dir=run_dir, role=role)
    manifest_path = _run_path(config["run_dir"], raw_config["manifest_path"], must_exist=True)
    tensor_path = _run_path(config["run_dir"], raw_config["tensor_path"], must_exist=True)
    manifest = _strict_json(manifest_path)
    _exact_keys(
        manifest,
        {
            "schema",
            "prompt_set_sha256",
            "prompts",
            "provenance",
            "implementation_files",
            "tensor_file",
            "equivalence",
        },
        label="conditioning manifest",
    )
    _require(manifest["schema"] == CONDITIONING_MANIFEST_SCHEMA,
             "conditioning manifest schema is invalid")
    _require(manifest["prompt_set_sha256"] == config["prompt_set_sha256"],
             "conditioning manifest binds another prompt set")
    _require(manifest["provenance"] == config["provenance"],
             "conditioning manifest provenance drifted")
    _require(manifest["implementation_files"] == _source_implementation(Path(source_root)),
             "Draft and Target UMT5 implementation file hashes differ")
    equivalence = manifest["equivalence"]
    _require(
        isinstance(equivalence, Mapping)
        and equivalence
        == {
            "method": "exact_tensor_sha256_and_codec_roundtrip_equality",
            "prompt_count": len(config["prompts"]),
            "all_equal": True,
        },
        "conditioning live/precomputed equivalence evidence is invalid",
    )
    tensor_file = manifest["tensor_file"]
    _require(isinstance(tensor_file, Mapping), "conditioning tensor_file is invalid")
    _exact_keys(
        tensor_file,
        {"path", "sha256", "nbytes", "format", "dtype", "tensor_digest_algorithm"},
        label="conditioning tensor_file",
    )
    actual_sha256, actual_size = _sha256_file(tensor_path)
    _require(
        tensor_file
        == {
            "path": raw_config["tensor_path"],
            "sha256": actual_sha256,
            "nbytes": actual_size,
            "format": "safetensors",
            "dtype": CONDITIONING_DTYPE,
            "tensor_digest_algorithm": CONDITIONING_TENSOR_DIGEST_ALGORITHM,
        },
        "conditioning tensor file metadata or bytes drifted",
    )
    prompt_rows = manifest["prompts"]
    _require(isinstance(prompt_rows, list) and len(prompt_rows) == len(config["prompts"]),
             "conditioning manifest prompt count drifted")
    expected_keys: set[str] = set()
    entries: list[dict[str, Any]] = []
    for index, (record, frozen) in enumerate(zip(prompt_rows, config["prompts"])):
        _require(isinstance(record, Mapping), f"conditioning prompt record {index} is invalid")
        _exact_keys(
            record,
            {"prompt_id", "prompt", "prompt_sha256", "tensor_key", "shape", "dtype", "tensor_sha256"},
            label=f"conditioning prompt record {index}",
        )
        key = f"prompt_{index:04d}"
        _require(
            {name: record[name] for name in ("prompt_id", "prompt", "prompt_sha256")}
            == dict(frozen),
            f"conditioning prompt record {index} drifted",
        )
        _require(record["tensor_key"] == key, f"conditioning tensor key {index} drifted")
        _require(record["shape"] == list(CONDITIONING_SHAPE),
                 f"conditioning tensor shape {index} drifted")
        _require(record["dtype"] == CONDITIONING_DTYPE,
                 f"conditioning tensor dtype {index} drifted")
        _require(isinstance(record["tensor_sha256"], str)
                 and _SHA256_RE.fullmatch(record["tensor_sha256"]),
                 f"conditioning tensor digest {index} is invalid")
        expected_keys.add(key)
        entries.append(dict(record))
    tensors = dict(codec.load(tensor_path))
    _require(set(tensors) == expected_keys, "conditioning tensor keys differ from manifest")
    for entry in entries:
        tensor = tensors[entry["tensor_key"]]
        shape, dtype = codec.describe(tensor)
        _require(shape == tuple(entry["shape"]) and dtype == entry["dtype"],
                 f"conditioning tensor {entry['tensor_key']} metadata drifted")
        _require(codec.tensor_sha256(tensor) == entry["tensor_sha256"],
                 f"conditioning tensor {entry['tensor_key']} bytes drifted")
    return LoadedConditioningCache(
        manifest=manifest,
        tensors=tensors,
        manifest_path=str(raw_config["manifest_path"]),
    )


def make_torch_conditioner(
    loaded: LoadedConditioningCache, *, torch_module: Any
) -> Any:
    """Create a tiny ``nn.Module`` with the same prompt->dict interface as UMT5."""

    entries = list(loaded.manifest["prompts"])

    class FrozenPromptConditioner(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._prompt_to_key: dict[str, str] = {}
            for entry in entries:
                key = str(entry["tensor_key"])
                prompt = str(entry["prompt"])
                tensor = loaded.tensors[key].detach().to(
                    device="cpu", dtype=torch_module.bfloat16
                ).contiguous()
                self.register_buffer(key, tensor, persistent=False)
                self._prompt_to_key[prompt] = key

        @property
        def device(self) -> Any:
            return next(self.buffers()).device

        def forward(self, text_prompts: Sequence[str]) -> dict[str, Any]:
            if not isinstance(text_prompts, Sequence) or isinstance(text_prompts, (str, bytes)):
                raise ConditioningCacheError("text_prompts must be a non-empty sequence")
            prompts = list(text_prompts)
            if not prompts:
                raise ConditioningCacheError("text_prompts must be a non-empty sequence")
            missing = [prompt for prompt in prompts if prompt not in self._prompt_to_key]
            if missing:
                digest = hashlib.sha256(str(missing[0]).encode("utf-8")).hexdigest()
                raise ConditioningCacheError(
                    f"conditioning cache miss for prompt sha256={digest}"
                )
            values = [getattr(self, self._prompt_to_key[prompt]) for prompt in prompts]
            return {"prompt_embeds": torch_module.cat(values, dim=0).contiguous()}

    return FrozenPromptConditioner().eval().requires_grad_(False)


__all__ = [
    "CONDITIONING_CONFIG_SCHEMA",
    "CONDITIONING_DTYPE",
    "CONDITIONING_MANIFEST_SCHEMA",
    "CONDITIONING_PROVENANCE_SCHEMA",
    "CONDITIONING_SHAPE",
    "ConditioningCacheError",
    "ConditioningTensorCodec",
    "LoadedConditioningCache",
    "TorchConditioningTensorCodec",
    "canonical_prompt_records",
    "load_shared_conditioning_cache",
    "make_torch_conditioner",
    "prompt_set_sha256",
    "seal_shared_conditioning_cache",
    "validate_conditioning_config",
]
