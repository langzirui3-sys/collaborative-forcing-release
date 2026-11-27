"""Sealed, content-addressed reuse of Insight 2 calibration evidence.

The long-video campaign spends most of its calibration time rebuilding the
same ten calibration trajectories in smoke and in every prompt shard.  This
module turns the smoke run's completed calibration population into a sealed
receipt.  A later run may *read* that receipt and materialize an exact local
copy below its own ``CFEXP_RUN_DIR``; the source run is never modified.

The receipt is not a shortcut around the normal campaign audits.  Imported
checkpoints, hidden states, latents, noise tensors, and Target RPC evidence are
put back at their original run-relative paths, after which
``Insight2Campaign.run_calibration`` and the terminal Target/KV audits replay
their existing validation unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence


CALIBRATION_RECEIPT_SCHEMA = "insight2_calibration_receipt_v1"
CALIBRATION_RECEIPT_SEAL_SCHEMA = "insight2_calibration_receipt_seal_v1"
CALIBRATION_RECEIPT_IMPORT_SCHEMA = "insight2_calibration_receipt_import_v1"
CALIBRATION_RECEIPT_RELATIVE = "calibration/shared-receipt-v1"
CALIBRATION_RECEIPT_MANIFEST = f"{CALIBRATION_RECEIPT_RELATIVE}/receipt.json"
CALIBRATION_RECEIPT_SEAL = f"{CALIBRATION_RECEIPT_RELATIVE}/SEALED.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT_SCHEMA = "insight2_campaign_checkpoint_v2"
_CALIBRATION_SCHEMA = "insight2_rank_calibration_v1"
_TARGET_RPC_EVENT_SCHEMA = "insight2_target_rpc_event_v3"
_TARGET_RPC_LIFECYCLE_SCHEMA = "insight2_target_rpc_lifecycle_v2"


class CalibrationReceiptError(RuntimeError):
    """A calibration receipt is missing, mutable, or bound to another run."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationReceiptError(message)


def _duplicate_checked_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        _require(key not in value, f"duplicate JSON key {key!r}")
        value[key] = item
    return value


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
        raise CalibrationReceiptError(f"value is not canonical JSON: {error}") from error


def _load_json(path: Path, *, label: str, require_canonical: bool = False) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CalibrationReceiptError(f"{label} contains invalid constant {item!r}")
            ),
        )
    except CalibrationReceiptError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationReceiptError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    if require_canonical:
        _require(
            raw == _canonical_bytes(value) + b"\n",
            f"{label} is not canonical JSON",
        )
    return value


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                size += len(block)
    except OSError as error:
        raise CalibrationReceiptError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest(), size


def _safe_relative(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty path")
    _require("\\" not in value and "\x00" not in value, f"{label} is not canonical")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    _require(
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and str(posix) == value
        and all(part not in {"", ".", ".."} for part in posix.parts),
        f"{label} must be a canonical run-relative POSIX path",
    )
    return value


def _safe_root(value: str | Path, *, label: str) -> Path:
    unresolved = Path(value).expanduser()
    _require(unresolved.is_absolute(), f"{label} must be absolute")
    _require(not unresolved.is_symlink(), f"{label} may not be a symlink")
    try:
        root = unresolved.resolve(strict=True)
    except OSError as error:
        raise CalibrationReceiptError(f"{label} is unavailable: {error}") from error
    _require(root.is_dir(), f"{label} must be a directory")
    return root


def _safe_file(root: Path, relative: Any, *, label: str) -> Path:
    text = _safe_relative(relative, label=label)
    current = root
    for part in PurePosixPath(text).parts:
        current = current / part
        _require(not current.is_symlink(), f"{label} may not traverse a symlink")
    try:
        result = current.resolve(strict=True)
        result.relative_to(root)
    except (OSError, ValueError) as error:
        raise CalibrationReceiptError(f"{label} escapes or is unavailable: {text}") from error
    _require(result.is_file(), f"{label} must be a regular file")
    return result


def _safe_destination(root: Path, relative: Any, *, label: str) -> Path:
    text = _safe_relative(relative, label=label)
    current = root
    for part in PurePosixPath(text).parts[:-1]:
        current = current / part
        _require(not current.is_symlink(), f"{label} may not traverse a symlink")
        if current.exists():
            _require(current.is_dir(), f"{label} parent is not a directory")
            try:
                current.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as error:
                raise CalibrationReceiptError(
                    f"{label} parent escapes or is unavailable"
                ) from error
    return current / PurePosixPath(text).name


def _prepare_destination(root: Path, relative: Any, *, label: str) -> Path:
    """Create destination parents one component at a time without symlinks."""

    text = _safe_relative(relative, label=label)
    current = root
    for part in PurePosixPath(text).parts[:-1]:
        candidate = current / part
        if not candidate.exists() and not candidate.is_symlink():
            try:
                candidate.mkdir(mode=0o700)
            except FileExistsError:
                pass
        _require(
            candidate.is_dir() and not candidate.is_symlink(),
            f"{label} may not traverse a symlink or non-directory",
        )
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise CalibrationReceiptError(f"{label} parent escapes the run") from error
        current = candidate
    return current / PurePosixPath(text).name


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _jsonl_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CalibrationReceiptError(f"cannot read {label}: {error}") from error
    _require(bool(raw) and raw.endswith(b"\n"), f"{label} is empty or incomplete")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), 1):
        _require(bool(line), f"{label} line {index} is blank")
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_duplicate_checked_object,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    CalibrationReceiptError(
                        f"{label} line {index} contains invalid constant {item!r}"
                    )
                ),
            )
        except CalibrationReceiptError:
            raise
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CalibrationReceiptError(f"{label} line {index} is invalid: {error}") from error
        _require(isinstance(value, dict), f"{label} line {index} is not an object")
        records.append(value)
    return records


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(record)) + b"\n" for record in records)


def _validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(binding)
    expected = {
        "schema",
        "code_revision",
        "config",
        "protocol",
        "geometry",
        "models",
        "artifact_specs",
        "verified_artifacts",
        "noise",
        "calibration_population",
        "hidden_risk",
    }
    _require(set(value) == expected, "calibration receipt binding fields are invalid")
    _require(
        value.get("schema") == "insight2_calibration_binding_v1",
        "calibration receipt binding schema is invalid",
    )
    _require(
        isinstance(value.get("code_revision"), str)
        and _COMMIT_RE.fullmatch(str(value["code_revision"])) is not None,
        "calibration receipt code revision is not a full commit",
    )
    # Canonicalization rejects non-finite values and non-JSON objects.  The
    # experiment runner builds the remaining nested fields from trusted
    # preflight facts and exact protocol data; equality is checked on import.
    _canonical_bytes(value)
    return value


def _checkpoint_paths(binding: Mapping[str, Any]) -> tuple[str, ...]:
    population = binding.get("calibration_population")
    geometry = binding.get("geometry")
    _require(isinstance(population, Mapping), "calibration population binding is invalid")
    _require(isinstance(geometry, Mapping), "calibration geometry binding is invalid")
    prompts = population.get("prompts")
    seeds = population.get("seeds")
    chunk_count = geometry.get("chunk_count")
    _require(isinstance(prompts, list) and bool(prompts), "calibration prompts are invalid")
    _require(
        isinstance(seeds, list)
        and bool(seeds)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds),
        "calibration seeds are invalid",
    )
    _require(
        isinstance(chunk_count, int) and not isinstance(chunk_count, bool) and chunk_count >= 2,
        "calibration chunk count is invalid",
    )
    result: list[str] = []
    prompt_ids: set[str] = set()
    for prompt in prompts:
        _require(
            isinstance(prompt, Mapping)
            and isinstance(prompt.get("prompt_id"), str)
            and bool(prompt["prompt_id"])
            and isinstance(prompt.get("prompt"), str)
            and bool(str(prompt["prompt"]).strip()),
            "calibration prompt binding is invalid",
        )
        prompt_id = str(prompt["prompt_id"])
        _require(prompt_id not in prompt_ids, "calibration prompt IDs are duplicated")
        prompt_ids.add(prompt_id)
        prompt_sha256 = hashlib.sha256(str(prompt["prompt"]).encode("utf-8")).hexdigest()
        for seed in seeds:
            key = f"{prompt_id}-{seed}-{prompt_sha256[:12]}"
            result.extend(
                f"state/calibration-baselines/{key}/chunk-{chunk_index:02d}.json"
                for chunk_index in range(chunk_count)
            )
    return tuple(result)


def _allowed_restore_path(path: str, *, kind: str, prompt_ids: frozenset[str]) -> None:
    allowed_exact = {
        "calibration": {"calibration/ranks.json"},
        "target_rpc_lifecycle": {"logs/target-rpc-lifecycle.jsonl"},
        "target_rpc_events": {"logs/target-rpc-events.jsonl"},
    }
    if kind in allowed_exact:
        _require(path in allowed_exact[kind], f"receipt {kind} restore path is invalid")
        return
    if kind == "checkpoint":
        _require(path.startswith("state/calibration-baselines/"), "checkpoint path is invalid")
        return
    if kind == "latent":
        _require(path.startswith("latents/calibration-baselines/"), "latent path is invalid")
        return
    if kind == "hidden":
        _require(
            path.startswith("hidden/calibration-baselines/")
            or path.startswith("latents/calibration-baselines/"),
            "hidden-state path is invalid",
        )
        return
    if kind == "noise":
        _require(
            any(path.startswith(f"noise/{prompt_id}/") for prompt_id in prompt_ids),
            "noise path is outside the calibration prompt population",
        )
        return
    if kind == "target_rpc_diagnostics":
        _require(
            path.startswith("provenance/target-rpc-diagnostics/")
            and path.endswith(".json"),
            "Target diagnostics path is invalid",
        )
        return
    raise CalibrationReceiptError(f"unknown receipt file kind {kind!r}")


def _inventory_entry(
    *, physical: Path, source_path: str, restore_path: str, kind: str
) -> dict[str, Any]:
    source_path = _safe_relative(source_path, label="receipt source_path")
    restore_path = _safe_relative(restore_path, label="receipt restore_path")
    sha256, nbytes = _sha256_file(physical)
    return {
        "source_path": source_path,
        "restore_path": restore_path,
        "kind": kind,
        "sha256": sha256,
        "nbytes": nbytes,
    }


@dataclass(frozen=True)
class VerifiedCalibrationReceipt:
    source_run_dir: Path
    receipt_sha256: str
    binding: Mapping[str, Any]
    files: tuple[Mapping[str, Any], ...]
    checkpoint_count: int
    target_baseline_count: int

    def public_binding(self) -> dict[str, Any]:
        geometry = self.binding["geometry"]
        return {
            "schema": CALIBRATION_RECEIPT_SCHEMA,
            "receipt_sha256": self.receipt_sha256,
            "code_revision": self.binding["code_revision"],
            "protocol_id": self.binding["protocol"]["id"],
            "chunk_count": geometry["chunk_count"],
            "duration_seconds_exact": geometry["duration_seconds_exact"],
            "file_count": len(self.files),
            "checkpoint_count": self.checkpoint_count,
            "target_baseline_count": self.target_baseline_count,
        }


def _add_original_file(
    *,
    root: Path,
    rows: dict[str, dict[str, Any]],
    relative: str,
    kind: str,
) -> None:
    relative = _safe_relative(relative, label=f"{kind} path")
    physical = _safe_file(root, relative, label=f"{kind} file")
    entry = _inventory_entry(
        physical=physical,
        source_path=relative,
        restore_path=relative,
        kind=kind,
    )
    prior = rows.get(relative)
    if prior is None:
        rows[relative] = entry
    else:
        _require(
            prior["sha256"] == entry["sha256"]
            and prior["nbytes"] == entry["nbytes"],
            f"receipt restore path {relative} has conflicting identities",
        )


def _collect_calibration_files(
    *, source_run_dir: Path, binding: Mapping[str, Any], staging: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    checkpoint_relatives = _checkpoint_paths(binding)
    expected_checkpoint_set = set(checkpoint_relatives)
    state_root = source_run_dir / "state" / "calibration-baselines"
    _require(state_root.is_dir() and not state_root.is_symlink(), "calibration state is missing")
    actual_checkpoint_set = {
        path.relative_to(source_run_dir).as_posix()
        for path in state_root.rglob("chunk-*.json")
        if path.is_file() and not path.is_symlink()
    }
    _require(
        actual_checkpoint_set == expected_checkpoint_set,
        "calibration checkpoint population is incomplete or contains extras",
    )

    population = binding["calibration_population"]
    prompt_ids = frozenset(str(item["prompt_id"]) for item in population["prompts"])
    seeds = tuple(int(seed) for seed in population["seeds"])
    geometry = binding["geometry"]
    chunk_count = int(geometry["chunk_count"])

    checkpoints: list[dict[str, Any]] = []
    for relative in checkpoint_relatives:
        path = _safe_file(source_run_dir, relative, label="calibration checkpoint")
        checkpoint = _load_json(path, label="calibration checkpoint", require_canonical=True)
        _require(
            checkpoint.get("schema") == _CHECKPOINT_SCHEMA,
            "calibration checkpoint schema is invalid",
        )
        metadata = checkpoint.get("metadata")
        generated = checkpoint.get("generated")
        history = checkpoint.get("history")
        _require(
            isinstance(metadata, Mapping)
            and metadata.get("kind") == "baseline"
            and isinstance(metadata.get("chunk_index"), int),
            "calibration checkpoint metadata is invalid",
        )
        _require(isinstance(generated, Mapping), "calibration generated record is invalid")
        _require(isinstance(history, Mapping), "calibration history is invalid")
        _add_original_file(
            root=source_run_dir, rows=rows, relative=relative, kind="checkpoint"
        )
        checkpoint_entry = rows[relative]
        checkpoints.append(
            {
                "path": relative,
                "sha256": checkpoint_entry["sha256"],
                "nbytes": checkpoint_entry["nbytes"],
            }
        )

        segments = history.get("segments")
        _require(isinstance(segments, list) and bool(segments), "checkpoint history is empty")
        for segment in segments:
            _require(
                isinstance(segment, Mapping) and isinstance(segment.get("latent_path"), str),
                "checkpoint latent path is invalid",
            )
            _add_original_file(
                root=source_run_dir,
                rows=rows,
                relative=str(segment["latent_path"]),
                kind="latent",
            )
        hidden_path = generated.get("hidden_state_path")
        if hidden_path is not None:
            _require(isinstance(hidden_path, str), "checkpoint hidden-state path is invalid")
            _add_original_file(
                root=source_run_dir,
                rows=rows,
                relative=hidden_path,
                kind="hidden",
            )
        diagnostics = generated.get("diagnostics")
        _require(isinstance(diagnostics, Mapping), "checkpoint diagnostics are invalid")
        sampled = diagnostics.get("sampled_hidden")
        if sampled is not None:
            _require(
                isinstance(sampled, Mapping) and isinstance(sampled.get("path"), str),
                "checkpoint sampled hidden-state path is invalid",
            )
            _add_original_file(
                root=source_run_dir,
                rows=rows,
                relative=str(sampled["path"]),
                kind="hidden",
            )

    _add_original_file(
        root=source_run_dir,
        rows=rows,
        relative="calibration/ranks.json",
        kind="calibration",
    )
    ranks = _load_json(
        source_run_dir / "calibration" / "ranks.json",
        label="calibration ranks",
        require_canonical=True,
    )
    _require(ranks.get("schema") == _CALIBRATION_SCHEMA, "calibration ranks schema is invalid")

    for prompt_id in prompt_ids:
        for seed in seeds:
            for chunk_index in range(chunk_count):
                _add_original_file(
                    root=source_run_dir,
                    rows=rows,
                    relative=(
                        f"noise/{prompt_id}/seed-{seed}/"
                        f"chunk-{chunk_index:02d}.safetensors"
                    ),
                    kind="noise",
                )

    lifecycle_records = [
        record
        for record in _jsonl_records(
            source_run_dir / "logs" / "target-rpc-lifecycle.jsonl",
            label="Target RPC lifecycle",
        )
        if record.get("phase") == "calibration_baseline"
    ]
    event_records = [
        record
        for record in _jsonl_records(
            source_run_dir / "logs" / "target-rpc-events.jsonl",
            label="Target RPC success journal",
        )
        if record.get("phase") == "calibration_baseline"
    ]
    expected_target_baselines = len(prompt_ids) * len(seeds)
    attempts: dict[str, list[str]] = {}
    for record in lifecycle_records:
        _require(
            record.get("schema") == _TARGET_RPC_LIFECYCLE_SCHEMA
            and isinstance(record.get("attempt_id"), str),
            "calibration Target lifecycle record is invalid",
        )
        attempts.setdefault(str(record["attempt_id"]), []).append(str(record.get("status")))
    _require(
        len(attempts) == expected_target_baselines
        and all(
            statuses == ["started", "backend_returned", "completed"]
            for statuses in attempts.values()
        ),
        "calibration Target lifecycle does not contain one complete attempt per prompt/seed",
    )
    _require(
        len(event_records) == expected_target_baselines
        and {str(record.get("attempt_id")) for record in event_records} == set(attempts)
        and all(record.get("schema") == _TARGET_RPC_EVENT_SCHEMA for record in event_records),
        "calibration Target success journal disagrees with lifecycle attempts",
    )

    lifecycle_snapshot = staging / "calibration-target-rpc-lifecycle.jsonl"
    events_snapshot = staging / "calibration-target-rpc-events.jsonl"
    _write_exclusive(lifecycle_snapshot, _jsonl_bytes(lifecycle_records))
    _write_exclusive(events_snapshot, _jsonl_bytes(event_records))
    lifecycle_source = f"{CALIBRATION_RECEIPT_RELATIVE}/calibration-target-rpc-lifecycle.jsonl"
    events_source = f"{CALIBRATION_RECEIPT_RELATIVE}/calibration-target-rpc-events.jsonl"
    rows["logs/target-rpc-lifecycle.jsonl"] = _inventory_entry(
        physical=lifecycle_snapshot,
        source_path=lifecycle_source,
        restore_path="logs/target-rpc-lifecycle.jsonl",
        kind="target_rpc_lifecycle",
    )
    rows["logs/target-rpc-events.jsonl"] = _inventory_entry(
        physical=events_snapshot,
        source_path=events_source,
        restore_path="logs/target-rpc-events.jsonl",
        kind="target_rpc_events",
    )

    for event in event_records:
        diagnostics_path = event.get("diagnostics_path")
        _require(
            isinstance(diagnostics_path, str) and bool(diagnostics_path),
            "calibration Target event omitted diagnostics_path",
        )
        _add_original_file(
            root=source_run_dir,
            rows=rows,
            relative=diagnostics_path,
            kind="target_rpc_diagnostics",
        )

    for entry in rows.values():
        _allowed_restore_path(
            str(entry["restore_path"]),
            kind=str(entry["kind"]),
            prompt_ids=prompt_ids,
        )
    ordered = [rows[key] for key in sorted(rows)]
    return ordered, checkpoints, expected_target_baselines


def seal_calibration_receipt(
    *, source_run_dir: str | Path, binding: Mapping[str, Any]
) -> VerifiedCalibrationReceipt:
    """Seal the complete C10 calibration population inside its smoke run."""

    root = _safe_root(source_run_dir, label="calibration source run")
    expected_binding = _validate_binding(binding)
    destination = root / CALIBRATION_RECEIPT_RELATIVE
    if destination.exists() or destination.is_symlink():
        return verify_calibration_receipt(
            source_run_dir=root,
            expected_binding=expected_binding,
        )

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    _require(not parent.is_symlink(), "calibration receipt parent may not be a symlink")
    staging = parent / f".{destination.name}.{os.getpid()}.tmp"
    _require(
        not staging.exists() and not staging.is_symlink(),
        "calibration receipt staging exists",
    )
    staging.mkdir(mode=0o700)
    try:
        files, checkpoints, target_baseline_count = _collect_calibration_files(
            source_run_dir=root,
            binding=expected_binding,
            staging=staging,
        )
        receipt = {
            "schema": CALIBRATION_RECEIPT_SCHEMA,
            "binding": expected_binding,
            "files": files,
            "checkpoints": checkpoints,
            "checkpoint_count": len(checkpoints),
            "target_baseline_count": target_baseline_count,
            "payload_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
        }
        receipt_bytes = _canonical_bytes(receipt) + b"\n"
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        _write_exclusive(staging / "receipt.json", receipt_bytes)
        _write_exclusive(
            staging / "SEALED.json",
            _canonical_bytes(
                {
                    "schema": CALIBRATION_RECEIPT_SEAL_SCHEMA,
                    "receipt_sha256": receipt_sha256,
                    "file_count": len(files),
                    "checkpoint_count": len(checkpoints),
                }
            )
            + b"\n",
        )
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_calibration_receipt(
        source_run_dir=root,
        expected_binding=expected_binding,
        expected_receipt_sha256=receipt_sha256,
    )


def _receipt_metadata(root, binding, expected_receipt_sha256):
    if expected_receipt_sha256 is not None:
        _require(
            isinstance(expected_receipt_sha256, str)
            and _SHA256_RE.fullmatch(expected_receipt_sha256) is not None,
            "expected calibration receipt SHA-256 is invalid",
        )
    receipt_path = _safe_file(root, CALIBRATION_RECEIPT_MANIFEST, label="calibration receipt")
    seal_path = _safe_file(root, CALIBRATION_RECEIPT_SEAL, label="calibration receipt seal")
    receipt = _load_json(receipt_path, label="calibration receipt", require_canonical=True)
    seal = _load_json(seal_path, label="calibration receipt seal", require_canonical=True)
    receipt_sha256, _ = _sha256_file(receipt_path)
    _require(
        expected_receipt_sha256 is None or receipt_sha256 == expected_receipt_sha256,
        "calibration receipt SHA-256 differs from the externally frozen digest",
    )
    _require(
        set(receipt)
        == {
            "schema",
            "binding",
            "files",
            "checkpoints",
            "checkpoint_count",
            "target_baseline_count",
            "payload_sha256",
        }
        and receipt.get("schema") == CALIBRATION_RECEIPT_SCHEMA,
        "calibration receipt fields or schema are invalid",
    )
    _require(
        set(seal) == {"schema", "receipt_sha256", "file_count", "checkpoint_count"}
        and seal.get("schema") == CALIBRATION_RECEIPT_SEAL_SCHEMA
        and seal.get("receipt_sha256") == receipt_sha256,
        "calibration receipt seal is invalid",
    )
    _require(
        receipt.get("binding") == binding,
        "calibration receipt binding mismatch (duration, commit, config, protocol, model, "
        "artifact, noise, prompt/seed, or hidden-risk definition drifted)",
    )
    return receipt, seal, receipt_sha256


def load_sealed_calibration_ranks(
    *,
    source_run_dir: str | Path,
    expected_binding: Mapping[str, Any],
    expected_receipt_sha256: str,
):
    """Load the original C10 rank distribution from its sealed run provenance."""
    from .correction_policy import SuddenDropRiskCalibrator, causal_erank_sudden_drop
    from .main_table_merge import MainTableMergeError, _load_frozen_index

    root = _safe_root(source_run_dir, label="sealed calibration attempt")
    binding = _validate_binding(expected_binding)
    _require(bool(expected_receipt_sha256), "sealed ranks require a frozen receipt digest")
    _require(root.parent.name == "attempts", "sealed ranks require an original run attempt")
    try:
        frozen, marker = _load_frozen_index(root.parent.parent)
        _require(marker["attempt_id"] == root.name and marker["run_id"] == frozen.run_dir.name
                 and marker["git_sha"] == binding["code_revision"],
                 "sealed rank source identity differs from its expected calibration")
        result = _load_json(frozen.file(
            f"attempts/{root.name}/result.json", label="sealed calibration result"),
            label="sealed calibration result")
        terminal = _load_json(frozen.file("run.json", label="terminal calibration execution"),
                              label="terminal calibration execution")
    except MainTableMergeError as error:
        raise CalibrationReceiptError("cannot verify the original calibration run seal") from error
    _require(terminal.get("execution_status") == "SUCCEEDED" and terminal.get("exit_code") == 0
             and terminal.get("run_id") == marker["run_id"]
             and terminal.get("latest_attempt_id") == root.name and result.get("status") == "succeeded",
             "sealed rank source execution differs from its terminal marker")

    receipt, seal, digest = _receipt_metadata(root, binding, expected_receipt_sha256)
    public = result.get("calibration_receipt")
    _require(isinstance(public, dict) and public.get("mode") == "sealed_by_this_run"
             and public.get("schema") == CALIBRATION_RECEIPT_SCHEMA
             and public.get("receipt_sha256") == digest
             and public.get("code_revision") == marker["git_sha"]
             and public.get("protocol_id") == binding["protocol"]["id"],
             "original indexed result does not bind this calibration receipt")
    files = receipt["files"]
    _require(isinstance(files, list) and all(isinstance(entry, dict) for entry in files)
             and len(files) == seal["file_count"] == public.get("file_count")
             and hashlib.sha256(_canonical_bytes(files)).hexdigest() == receipt["payload_sha256"],
             "sealed calibration file inventory differs")
    by_path = {entry["restore_path"]: entry for entry in files}
    _require(len(by_path) == len(files), "duplicate sealed calibration file path")
    rank_entry = by_path.get("calibration/ranks.json")
    _require(isinstance(rank_entry, dict) and rank_entry.get("kind") == "calibration",
             "sealed receipt omits its rank data")
    rank_path = _safe_file(root, rank_entry["source_path"], label="sealed ranks")
    rank_sha256, rank_bytes = _sha256_file(rank_path)
    _require((rank_sha256, rank_bytes) == (rank_entry["sha256"], rank_entry["nbytes"]),
             "sealed rank bytes differ from the original receipt")
    ranks = _load_json(rank_path, label="sealed ranks", require_canonical=True)
    chunk_count = binding["geometry"]["chunk_count"]
    _require(chunk_count == 120, "sealed eRank startup requires the 120-block calibration")
    expected_pairs = [[prompt["prompt_id"], seed]
                      for prompt in binding["calibration_population"]["prompts"]
                      for seed in binding["calibration_population"]["seeds"]]
    rows = ranks.get("rows")
    _require(ranks.get("schema") == _CALIBRATION_SCHEMA
             and ranks.get("protocol_id") == binding["protocol"]["id"]
             and ranks.get("prompt_seed_pairs") == expected_pairs
             and isinstance(rows, list) and len(rows) == len(expected_pairs) * (chunk_count - 1)
             and receipt["checkpoint_count"] == seal["checkpoint_count"]
                == public.get("checkpoint_count") == len(expected_pairs) * chunk_count,
             "sealed rank population or checkpoint coverage differs")
    calibration_id = "cal-" + hashlib.sha256(_canonical_bytes(rows)).hexdigest()[:40]
    _require(ranks.get("calibration_id") == calibration_id, "sealed calibration identifier differs")
    drops = []
    offset = 0
    for prompt_id, seed in expected_pairs:
        prior = []
        for index in range(1, chunk_count):
            row = rows[offset]
            offset += 1
            _require(isinstance(row, dict), "sealed rank row must be an object")
            rank = row.get("rank")
            hidden = by_path.get(row.get("hidden_state_path"), {})
            _require((row.get("prompt_id"), row.get("seed"), row.get("chunk_index"))
                     == (prompt_id, seed, index) and type(rank) in (int, float) and math.isfinite(rank)
                     and hidden.get("kind") == "hidden"
                     and hidden.get("sha256") == row.get("hidden_state_sha256"),
                     "sealed rank row differs from its original checkpoint inventory")
            prior.append(float(rank))
            drop = causal_erank_sudden_drop(prior, lookback_chunks=4)
            _require(type(row.get("erank_drop_score")) in (int, float)
                     and row["erank_drop_score"] == drop, "sealed causal rank drop differs")
            drops.append(drop)
    calibrator = SuddenDropRiskCalibrator(calibration_id, tuple(drops))
    return calibrator, {
        **public,
        "mode": "sealed_rank_distribution",
        "source_run_id": marker["run_id"],
        "source_attempt_id": marker["attempt_id"],
        "source_artifact_index_sha256": marker["artifact_index_sha256"],
        "rank_sha256": rank_sha256,
        "calibration_id": calibration_id,
        "rank_count": len(drops),
        "drop_scores_sha256": hashlib.sha256(_canonical_bytes(drops)).hexdigest(),
    }


def verify_calibration_receipt(
    *,
    source_run_dir: str | Path,
    expected_binding: Mapping[str, Any],
    expected_receipt_sha256: str | None = None,
) -> VerifiedCalibrationReceipt:
    """Verify a sealed receipt and every source byte without modifying it."""

    root = _safe_root(source_run_dir, label="calibration receipt source run")
    binding = _validate_binding(expected_binding)
    receipt, seal, receipt_sha256 = _receipt_metadata(root, binding, expected_receipt_sha256)
    expected_checkpoints = _checkpoint_paths(binding)
    checkpoint_rows = receipt.get("checkpoints")
    file_rows = receipt.get("files")
    _require(isinstance(file_rows, list) and bool(file_rows), "receipt files are invalid")
    _require(isinstance(checkpoint_rows, list), "receipt checkpoints are invalid")
    _require(
        receipt.get("checkpoint_count") == len(expected_checkpoints)
        and seal.get("checkpoint_count") == len(expected_checkpoints)
        and len(checkpoint_rows) == len(expected_checkpoints),
        "calibration receipt checkpoint count drifted",
    )
    population = binding["calibration_population"]
    prompt_ids = frozenset(str(item["prompt_id"]) for item in population["prompts"])
    restore_paths: set[str] = set()
    source_paths: set[str] = set()
    parsed_files: list[dict[str, Any]] = []
    by_restore: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(file_rows):
        _require(
            isinstance(raw, dict)
            and set(raw) == {"source_path", "restore_path", "kind", "sha256", "nbytes"},
            f"receipt file row {index} fields are invalid",
        )
        source_path = _safe_relative(raw["source_path"], label=f"receipt file {index} source")
        restore_path = _safe_relative(raw["restore_path"], label=f"receipt file {index} restore")
        kind = raw["kind"]
        sha256 = raw["sha256"]
        nbytes = raw["nbytes"]
        _require(isinstance(kind, str) and bool(kind), f"receipt file {index} kind is invalid")
        _require(
            isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256) is not None,
            f"receipt file {index} SHA-256 is invalid",
        )
        _require(
            isinstance(nbytes, int) and not isinstance(nbytes, bool) and nbytes > 0,
            f"receipt file {index} size is invalid",
        )
        _require(source_path not in source_paths, "receipt source paths are duplicated")
        _require(restore_path not in restore_paths, "receipt restore paths are duplicated")
        source_paths.add(source_path)
        restore_paths.add(restore_path)
        _allowed_restore_path(restore_path, kind=kind, prompt_ids=prompt_ids)
        physical = _safe_file(root, source_path, label=f"receipt payload {index}")
        actual_sha256, actual_nbytes = _sha256_file(physical)
        _require(
            (actual_sha256, actual_nbytes) == (sha256, nbytes),
            f"calibration receipt payload drifted: {source_path}",
        )
        parsed = dict(raw)
        parsed_files.append(parsed)
        by_restore[restore_path] = parsed
    _require(
        seal.get("file_count") == len(parsed_files)
        and receipt.get("payload_sha256")
        == hashlib.sha256(_canonical_bytes(parsed_files)).hexdigest(),
        "calibration receipt payload inventory digest drifted",
    )
    expected_checkpoint_rows: list[dict[str, Any]] = []
    for path in expected_checkpoints:
        entry = by_restore.get(path)
        _require(
            entry is not None and entry["kind"] == "checkpoint",
            "checkpoint is absent from receipt files",
        )
        expected_checkpoint_rows.append(
            {"path": path, "sha256": entry["sha256"], "nbytes": entry["nbytes"]}
        )
    _require(
        checkpoint_rows == expected_checkpoint_rows,
        "per-checkpoint calibration hashes drifted",
    )
    expected_target_baselines = len(prompt_ids) * len(population["seeds"])
    _require(
        receipt.get("target_baseline_count") == expected_target_baselines,
        "calibration Target baseline count drifted",
    )
    receipt_dir = root / CALIBRATION_RECEIPT_RELATIVE
    actual_internal: set[str] = set()
    for path in receipt_dir.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "calibration receipt directory contains a non-file or symlink",
        )
        actual_internal.add(path.name)
    _require(
        actual_internal
        == {
            "receipt.json",
            "SEALED.json",
            "calibration-target-rpc-lifecycle.jsonl",
            "calibration-target-rpc-events.jsonl",
        },
        "calibration receipt directory contains unexpected or missing files",
    )
    return VerifiedCalibrationReceipt(
        source_run_dir=root,
        receipt_sha256=receipt_sha256,
        binding=binding,
        files=tuple(parsed_files),
        checkpoint_count=len(expected_checkpoints),
        target_baseline_count=expected_target_baselines,
    )


def _copy_verified_file(
    *, source: Path, destination: Path, sha256: str, nbytes: int
) -> None:
    if destination.exists() or destination.is_symlink():
        _require(
            destination.is_file()
            and not destination.is_symlink()
            and _sha256_file(destination) == (sha256, nbytes),
            f"calibration import conflicts with existing file: {destination}",
        )
        return
    _require(
        destination.parent.is_dir() and not destination.parent.is_symlink(),
        "calibration import parent is unavailable or a symlink",
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    _require(
        not temporary.exists() and not temporary.is_symlink(),
        "calibration import temporary exists",
    )
    digest = hashlib.sha256()
    size = 0
    temporary_created = False
    try:
        with source.open("rb") as source_handle:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            temporary_created = True
            with os.fdopen(descriptor, "wb") as destination_handle:
                while block := source_handle.read(1024 * 1024):
                    destination_handle.write(block)
                    digest.update(block)
                    size += len(block)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        _require(
            (digest.hexdigest(), size) == (sha256, nbytes),
            f"calibration source changed while copying: {source}",
        )
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _require(
                destination.is_file()
                and not destination.is_symlink()
                and _sha256_file(destination) == (sha256, nbytes),
                f"concurrent calibration import conflicts: {destination}",
            )
        temporary.unlink()
        temporary_created = False
    except BaseException:
        if temporary_created:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise


def restore_calibration_receipt(
    *,
    source_run_dir: str | Path,
    destination_run_dir: str | Path,
    expected_binding: Mapping[str, Any],
    expected_receipt_sha256: str,
    preverified: VerifiedCalibrationReceipt | None = None,
) -> VerifiedCalibrationReceipt:
    """Copy verified evidence into a new run without modifying the receipt.

    ``preverified`` is the result of the formal preflight verification.  Using
    it avoids hashing every (potentially very large) hidden state and latent a
    second time before the copy.  Every source byte is still hashed while it
    is copied, and the small externally addressed receipt envelope is checked
    immediately before and after the import.
    """

    destination = _safe_root(destination_run_dir, label="calibration destination run")
    source = _safe_root(source_run_dir, label="calibration receipt source run")
    binding = _validate_binding(expected_binding)
    _require(
        isinstance(expected_receipt_sha256, str)
        and _SHA256_RE.fullmatch(expected_receipt_sha256) is not None,
        "expected calibration receipt SHA-256 is invalid",
    )
    if preverified is None:
        verified = verify_calibration_receipt(
            source_run_dir=source,
            expected_binding=binding,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    else:
        _require(
            isinstance(preverified, VerifiedCalibrationReceipt),
            "preverified calibration receipt has an invalid type",
        )
        _require(
            preverified.source_run_dir == source
            and preverified.receipt_sha256 == expected_receipt_sha256
            and preverified.binding == binding,
            "preverified calibration receipt differs from the requested source or binding",
        )
        verified = preverified

    def verify_envelope() -> None:
        receipt_path = _safe_file(
            source, CALIBRATION_RECEIPT_MANIFEST, label="calibration receipt"
        )
        seal_path = _safe_file(
            source, CALIBRATION_RECEIPT_SEAL, label="calibration receipt seal"
        )
        receipt = _load_json(
            receipt_path, label="calibration receipt", require_canonical=True
        )
        seal = _load_json(
            seal_path, label="calibration receipt seal", require_canonical=True
        )
        actual_digest, _ = _sha256_file(receipt_path)
        _require(
            actual_digest == expected_receipt_sha256,
            "calibration receipt SHA-256 differs from the externally frozen digest",
        )
        _require(
            receipt.get("binding") == binding
            and receipt.get("files") == [dict(row) for row in verified.files],
            "calibration receipt envelope changed after preflight",
        )
        _require(
            seal.get("schema") == CALIBRATION_RECEIPT_SEAL_SCHEMA
            and seal.get("receipt_sha256") == actual_digest,
            "calibration receipt seal changed after preflight",
        )

    verify_envelope()
    _require(
        destination != verified.source_run_dir,
        "calibration receipt source and destination runs must differ",
    )
    for row in verified.files:
        source_file = _safe_file(
            verified.source_run_dir,
            row["source_path"],
            label="calibration import source",
        )
        target = _prepare_destination(
            destination,
            row["restore_path"],
            label="calibration import destination",
        )
        checked_target = _safe_destination(
            destination,
            row["restore_path"],
            label="calibration import destination",
        )
        _require(checked_target == target, "calibration import destination changed")
        _copy_verified_file(
            source=source_file,
            destination=target,
            sha256=str(row["sha256"]),
            nbytes=int(row["nbytes"]),
        )
    # Payloads are content-checked during their copies.  Re-checking only the
    # envelope here closes receipt/inventory replacement without a third full
    # read of all hidden states and latents.
    verify_envelope()
    import_log = destination / "logs" / "calibration-receipt-import.json"
    value = {
        "schema": CALIBRATION_RECEIPT_IMPORT_SCHEMA,
        "receipt_sha256": verified.receipt_sha256,
        "binding_sha256": hashlib.sha256(_canonical_bytes(verified.binding)).hexdigest(),
        "file_count": len(verified.files),
        "checkpoint_count": verified.checkpoint_count,
        "target_baseline_count": verified.target_baseline_count,
        "source_read_only": True,
    }
    payload = _canonical_bytes(value) + b"\n"
    if import_log.exists() or import_log.is_symlink():
        _require(
            import_log.is_file()
            and not import_log.is_symlink()
            and import_log.read_bytes() == payload,
            "calibration receipt import log conflicts",
        )
    else:
        _write_exclusive(import_log, payload)
    return verified


__all__ = [
    "CALIBRATION_RECEIPT_IMPORT_SCHEMA",
    "CALIBRATION_RECEIPT_MANIFEST",
    "CALIBRATION_RECEIPT_RELATIVE",
    "CALIBRATION_RECEIPT_SCHEMA",
    "CALIBRATION_RECEIPT_SEAL",
    "CALIBRATION_RECEIPT_SEAL_SCHEMA",
    "CalibrationReceiptError",
    "VerifiedCalibrationReceipt",
    "load_sealed_calibration_ranks",
    "restore_calibration_receipt",
    "seal_calibration_receipt",
    "verify_calibration_receipt",
]
