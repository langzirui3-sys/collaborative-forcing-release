"""Strict merger for the six sealed ``main-table-v1`` method runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

from .main_table_analysis import MainTableAnalysisError, RAW_SCHEMA, analyze_rows
from .main_table_baseline_campaign import RESULT_SCHEMA as METHOD_RESULT_SCHEMA
from .main_table_v1 import METHOD_IDS, PROMPT_IDS


RESULT_SCHEMA = "main_table_merge_result_v1"
SOURCE_SCHEMA = "main_table_merge_sources_v1"
ANALYSIS_FILENAME = "main-table-analysis.json"
RAW_FILENAME = "main-table-raw-rows.json"
RESULT_FILENAME = "result.json"
SOURCE_FILENAME = "sources.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MainTableMergeError(RuntimeError):
    """Raised when sealed method evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MainTableMergeError(message)


def _duplicate_checked_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        _require(key not in value, f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                MainTableMergeError(f"{label} contains invalid constant {item!r}")
            ),
        )
    except MainTableMergeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MainTableMergeError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
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
        raise MainTableMergeError(f"value is not canonical JSON: {error}") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
            size = handle.tell()
    except OSError as error:
        raise MainTableMergeError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest(), size


@dataclass(frozen=True)
class _FrozenIndex:
    run_dir: Path
    entries: Mapping[str, tuple[int, str]]

    def file(self, relative: str, *, label: str) -> Path:
        posix = PurePosixPath(relative)
        _require(
            relative == str(posix)
            and not posix.is_absolute()
            and all(part not in {"", ".", ".."} for part in posix.parts),
            f"{label} has an unsafe indexed path",
        )
        _require(relative in self.entries, f"{label} is absent from artifact-index.json")
        candidate = self.run_dir.joinpath(*posix.parts)
        current = self.run_dir
        for part in posix.parts:
            current = current / part
            _require(not current.is_symlink(), f"{label} traverses a symlink")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.run_dir)
        except (OSError, ValueError) as error:
            raise MainTableMergeError(f"{label} is unavailable or escapes its run") from error
        _require(resolved.is_file(), f"{label} is not a regular file")
        expected_size, expected_sha256 = self.entries[relative]
        actual_sha256, actual_size = _sha256_file(resolved)
        _require(
            (actual_size, actual_sha256) == (expected_size, expected_sha256),
            f"{label} differs from the sealed artifact index",
        )
        return resolved

    def verify_all(self) -> None:
        for relative in sorted(self.entries):
            self.file(relative, label=f"indexed artifact {relative!r}")


def _load_frozen_index(run_dir: Path) -> tuple[_FrozenIndex, dict[str, Any]]:
    try:
        run_dir = run_dir.resolve(strict=True)
    except OSError as error:
        raise MainTableMergeError(f"formal method run is unavailable: {error}") from error
    _require(run_dir.is_dir() and not run_dir.is_symlink(), "formal run must be a directory")
    _require(not (run_dir / "_FAILED.json").exists(), "formal method run has a failure marker")
    index_path = run_dir / "artifact-index.json"
    marker_path = run_dir / "_SUCCESS.json"
    index = _load_json(index_path, label="artifact index")
    marker = _load_json(marker_path, label="success marker")
    _require(
        set(index) == {"schema_version", "algorithm", "files"}
        and index.get("schema_version") == 1
        and index.get("algorithm") == "sha256"
        and isinstance(index.get("files"), list),
        "formal method artifact index schema is invalid",
    )
    entries: dict[str, tuple[int, str]] = {}
    ordered_paths: list[str] = []
    for item in index["files"]:
        _require(
            isinstance(item, dict)
            and set(item) == {"path", "size", "sha256"}
            and isinstance(item.get("path"), str)
            and isinstance(item.get("size"), int)
            and not isinstance(item.get("size"), bool)
            and int(item["size"]) >= 0
            and isinstance(item.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(item["sha256"])) is not None,
            "formal method artifact index contains an invalid entry",
        )
        relative = str(item["path"])
        posix = PurePosixPath(relative)
        _require(
            relative == str(posix)
            and not posix.is_absolute()
            and all(part not in {"", ".", ".."} for part in posix.parts)
            and relative not in {"artifact-index.json", "_SUCCESS.json", "_FAILED.json"},
            "formal method artifact index contains an unsafe or reserved path",
        )
        _require(relative not in entries, "formal method artifact index repeats a path")
        entries[relative] = (int(item["size"]), str(item["sha256"]))
        ordered_paths.append(relative)
    _require(ordered_paths == sorted(ordered_paths), "artifact index paths are not sorted")
    index_sha256, _ = _sha256_file(index_path)
    _require(
        set(marker)
        == {
            "artifact_index_sha256",
            "artifact_publication",
            "attempt_id",
            "created_at",
            "exit_code",
            "git_sha",
            "run_id",
            "spec_hash",
            "status",
        }
        and marker.get("status") == "SUCCEEDED"
        and marker.get("exit_code") == 0
        and marker.get("artifact_index_sha256") == index_sha256
        and isinstance(marker.get("artifact_publication"), dict)
        and isinstance(marker["artifact_publication"].get("destination"), str)
        and bool(marker["artifact_publication"]["destination"])
        and isinstance(marker.get("git_sha"), str)
        and re.fullmatch(r"[0-9a-f]{40}", str(marker["git_sha"])) is not None,
        "formal method success marker is invalid, unsealed, or local-only",
    )
    frozen = _FrozenIndex(run_dir=run_dir, entries=entries)
    frozen.verify_all()
    return frozen, marker


@dataclass(frozen=True)
class _ValidatedMethod:
    method_id: str
    rows: tuple[dict[str, Any], ...]
    source: Mapping[str, Any]


def _indexed_attempt_file(
    frozen: _FrozenIndex, attempt_id: str, relative: str, *, label: str
) -> Path:
    return frozen.file(
        (PurePosixPath("attempts") / attempt_id / PurePosixPath(relative)).as_posix(),
        label=label,
    )


def validate_method_attempt(attempt_dir: Path, expected_method: str) -> _ValidatedMethod:
    try:
        attempt_dir = attempt_dir.resolve(strict=True)
    except OSError as error:
        raise MainTableMergeError(f"{expected_method} attempt is unavailable: {error}") from error
    _require(
        attempt_dir.is_dir()
        and not attempt_dir.is_symlink()
        and attempt_dir.parent.name == "attempts",
        f"{expected_method} source must be an attempt directory",
    )
    attempt_id = attempt_dir.name
    run_dir = attempt_dir.parent.parent
    frozen, marker = _load_frozen_index(run_dir)
    run_path = frozen.file("run.json", label=f"{expected_method} run.json")
    run = _load_json(run_path, label=f"{expected_method} run.json")
    _require(
        set(run) == {"execution_status", "exit_code", "latest_attempt_id", "run_id", "updated_at"}
        and run.get("execution_status") == "SUCCEEDED"
        and run.get("exit_code") == 0
        and run.get("latest_attempt_id") == attempt_id
        and run.get("run_id") == marker.get("run_id")
        and marker.get("attempt_id") == attempt_id,
        f"{expected_method} run identity or terminal status is invalid",
    )
    result_relative = (PurePosixPath("attempts") / attempt_id / "result.json").as_posix()
    result_path = frozen.file(result_relative, label=f"{expected_method} result.json")
    result = _load_json(result_path, label=f"{expected_method} result.json")
    expected_result_fields = {
        "schema",
        "status",
        "method_id",
        "prompt_count",
        "row_count",
        "seed_count",
        "raw_rows",
        "vbench_wall_seconds",
        "vbench_measurements",
        "generation_device",
        "quality_device",
    }
    _require(
        set(result) == expected_result_fields
        and result.get("schema") == METHOD_RESULT_SCHEMA
        and result.get("status") == "succeeded"
        and result.get("method_id") == expected_method
        and result.get("prompt_count") == len(PROMPT_IDS)
        and result.get("row_count") == len(PROMPT_IDS)
        and result.get("seed_count") == 1,
        f"{expected_method} method result schema or coverage is invalid",
    )
    raw_binding = result.get("raw_rows")
    _require(
        isinstance(raw_binding, dict)
        and set(raw_binding) == {"path", "sha256", "nbytes"}
        and isinstance(raw_binding.get("path"), str)
        and isinstance(raw_binding.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(raw_binding["sha256"])) is not None
        and isinstance(raw_binding.get("nbytes"), int)
        and not isinstance(raw_binding.get("nbytes"), bool),
        f"{expected_method} raw-row binding is invalid",
    )
    raw_path = _indexed_attempt_file(
        frozen, attempt_id, str(raw_binding["path"]), label=f"{expected_method} raw rows"
    )
    raw_sha256, raw_size = _sha256_file(raw_path)
    _require(
        (raw_sha256, raw_size) == (raw_binding["sha256"], raw_binding["nbytes"]),
        f"{expected_method} raw-row result binding drifted",
    )
    raw = _load_json(raw_path, label=f"{expected_method} raw rows")
    _require(
        set(raw) == {"schema", "rows"}
        and raw.get("schema") == RAW_SCHEMA
        and isinstance(raw.get("rows"), list),
        f"{expected_method} raw-row document is invalid",
    )
    rows = raw["rows"]
    _require(
        len(rows) == len(PROMPT_IDS)
        and all(isinstance(row, dict) for row in rows)
        and tuple(row.get("prompt_id") for row in rows) == PROMPT_IDS
        and all(row.get("method_id") == expected_method for row in rows),
        f"{expected_method} raw-row order, identity, or coverage drifted",
    )
    measurements = result.get("vbench_measurements")
    _require(
        isinstance(measurements, list)
        and len(measurements) == len(PROMPT_IDS)
        and tuple(
            item.get("prompt_id") if isinstance(item, dict) else None
            for item in measurements
        )
        == PROMPT_IDS,
        f"{expected_method} official VBench-Long measurement coverage drifted",
    )
    for prompt_id, item in zip(PROMPT_IDS, measurements):
        _require(
            isinstance(item, dict)
            and set(item) == {"prompt_id", "measurement_id", "path", "sha256", "nbytes"}
            and item.get("prompt_id") == prompt_id
            and item.get("measurement_id") == f"mt-{expected_method}-{prompt_id}"
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(item["sha256"])) is not None
            and isinstance(item.get("nbytes"), int)
            and not isinstance(item.get("nbytes"), bool),
            f"{expected_method}/{prompt_id} VBench evidence binding is invalid",
        )
        evidence_path = _indexed_attempt_file(
            frozen,
            attempt_id,
            str(item["path"]),
            label=f"{expected_method}/{prompt_id} VBench evidence",
        )
        evidence_sha256, evidence_size = _sha256_file(evidence_path)
        _require(
            (evidence_sha256, evidence_size) == (item["sha256"], item["nbytes"]),
            f"{expected_method}/{prompt_id} VBench evidence binding drifted",
        )
    result_sha256, result_size = _sha256_file(result_path)
    index_sha256, index_size = _sha256_file(run_dir / "artifact-index.json")
    return _ValidatedMethod(
        method_id=expected_method,
        rows=tuple(dict(row) for row in rows),
        source={
            "method_id": expected_method,
            "run_id": run["run_id"],
            "attempt_id": attempt_id,
            "attempt_dir": str(attempt_dir),
            "git_sha": marker["git_sha"],
            "artifact_publication_destination": marker["artifact_publication"]["destination"],
            "artifact_index": {"sha256": index_sha256, "nbytes": index_size},
            "result": {"sha256": result_sha256, "nbytes": result_size},
            "raw_rows": {"sha256": raw_sha256, "nbytes": raw_size},
            "official_vbench_long_verified": True,
        },
    )


def merge_method_attempts(*, attempt_dirs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    _require(
        len(attempt_dirs) == len(METHOD_IDS),
        f"exactly {len(METHOD_IDS)} method attempts are required",
    )
    output_dir = output_dir.expanduser().absolute()
    _require(not output_dir.exists(), "output directory must be new and absent")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        validated = [
            validate_method_attempt(Path(path), method_id)
            for method_id, path in zip(METHOD_IDS, attempt_dirs)
        ]
        git_shas = {str(item.source["git_sha"]) for item in validated}
        _require(len(git_shas) == 1, "method runs were not produced from one exact Git SHA")
        rows = [row for item in validated for row in item.rows]
        try:
            analysis = analyze_rows(rows)
        except MainTableAnalysisError as error:
            raise MainTableMergeError(f"global main-table analysis failed: {error}") from error
        _require(
            analysis.get("method_count") == len(METHOD_IDS)
            and analysis.get("prompt_count") == len(PROMPT_IDS)
            and analysis.get("seed_count") == 1
            and analysis.get("row_count") == len(METHOD_IDS) * len(PROMPT_IDS)
            and analysis.get("shard_means_used") is False,
            "global main-table analysis summary drifted",
        )
        raw_document = {"schema": RAW_SCHEMA, "rows": rows}
        source_document = {
            "schema": SOURCE_SCHEMA,
            "experiment_git_sha": next(iter(git_shas)),
            "methods": [dict(item.source) for item in validated],
        }
        _write_json(staging / RAW_FILENAME, raw_document)
        _write_json(staging / ANALYSIS_FILENAME, analysis)
        _write_json(staging / SOURCE_FILENAME, source_document)
        artifacts: dict[str, dict[str, Any]] = {}
        for name in (RAW_FILENAME, ANALYSIS_FILENAME, SOURCE_FILENAME):
            sha256, nbytes = _sha256_file(staging / name)
            artifacts[name] = {"sha256": sha256, "nbytes": nbytes}
        result = {
            "schema": RESULT_SCHEMA,
            "status": "succeeded",
            "experiment_git_sha": next(iter(git_shas)),
            "method_count": len(METHOD_IDS),
            "prompt_count": len(PROMPT_IDS),
            "seed_count": 1,
            "row_count": len(rows),
            "method_ids": list(METHOD_IDS),
            "official_vbench_long_verified": True,
            "shard_means_used": False,
            "artifacts": artifacts,
        }
        _write_json(staging / RESULT_FILENAME, result)
        os.replace(staging, output_dir)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.main_table_merge")
    parser.add_argument(
        "--attempt-dir",
        action="append",
        type=Path,
        required=True,
        help="sealed attempt directory, repeated once in frozen method order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = merge_method_attempts(
            attempt_dirs=args.attempt_dir,
            output_dir=args.output_dir,
        )
    except (MainTableMergeError, OSError) as error:
        print(f"main-table-merge: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MainTableMergeError",
    "RESULT_SCHEMA",
    "SOURCE_SCHEMA",
    "merge_method_attempts",
    "validate_method_attempt",
]
