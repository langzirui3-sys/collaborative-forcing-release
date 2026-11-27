"""Read-only stage timing analysis; output is diagnostic, never a speedup claim."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

SCHEMA = "stage_timing_v1"
REPORT_SCHEMA = "stage_timing_report_v1"
CORE_STAGES = ("history_read", "history_cat", "prefill", "denoise", "rank", "snapshot", "transfer", "rpc", "persist")
MARKERS = ("_SUCCESS.json", "_FAILED.json", "artifact-index.json")


class StageTimingAnalysisError(ValueError):
    """Invalid, incomplete or unsafe timing input/output."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageTimingAnalysisError(message)


def _number(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             f"{label} must be finite and nonnegative")
    return float(value)


def _text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _safe_path(path: Path, *, existing: bool = True) -> Path:
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        _require(not item.is_symlink(), f"symlink path is forbidden: {item}")
    if existing:
        _require(path.exists(), f"path does not exist: {path}")
    return path


def _validate(record: Any) -> dict[str, Any]:
    _require(isinstance(record, dict), "timing record must be an object")
    required = {"schema", "source", "record_id", "pid", "metadata", "status", "wall_seconds",
                "spans", "not_applicable", "cuda_sync_wall_seconds"}
    _require(required <= record.keys() and record["schema"] == SCHEMA, "timing record schema/fields mismatch")
    for key in ("source", "record_id"):
        _text(record[key], key)
    _require(type(record["pid"]) is int and record["pid"] > 0, "pid must be positive")
    _require(isinstance(record["metadata"], dict), "metadata must be an object")
    _require(record["status"] in ("completed", "error"), "invalid record status")
    wall = _number(record["wall_seconds"], "record wall_seconds")
    sync = _number(record["cuda_sync_wall_seconds"], "cuda_sync_wall_seconds")
    _require(sync <= wall + 1e-6, "CUDA sync wall exceeds record wall")
    for key in ("chunk_index", "history_chunk_count"):
        value = record["metadata"].get(key)
        _require(value is None or (type(value) is int and value >= 0), f"invalid metadata {key}")
    _require(isinstance(record["spans"], list), "spans must be a list")
    spans = {}
    for span in record["spans"]:
        required_span = {"span_id", "parent_span_id", "name", "detail", "wall_seconds", "cuda_seconds",
                         "start_offset_seconds", "end_offset_seconds", "status"}
        _require(isinstance(span, dict) and required_span <= span.keys(), "invalid span fields")
        identifier = span["span_id"]
        _require(type(identifier) is int and identifier >= 0, "span_id must be a nonnegative integer")
        _require(identifier not in spans, "duplicate span_id")
        _text(span["name"], "span name")
        _require(span["parent_span_id"] is None or (type(span["parent_span_id"]) is int and span["parent_span_id"] >= 0), "invalid parent_span_id")
        _require(span["status"] in ("completed", "error", "interrupted"), "invalid span status")
        if span["status"] == "interrupted":
            _require(span.get("interruption") == "DraftProbeStop"
                     and span["name"] == "denoise" and span["detail"] in ("forward_0", "forward_1")
                     and record["source"] == "worker-draft"
                     and record["metadata"].get("action") == "generate_early",
                     "invalid intentional Draft interruption")
        seconds = _number(span["wall_seconds"], "span wall_seconds")
        start = _number(span["start_offset_seconds"], "span start")
        end = _number(span["end_offset_seconds"], "span end")
        _require(start <= end and end <= wall + 1e-6, "span interval outside record")
        _require(math.isclose(seconds, end - start, rel_tol=1e-6, abs_tol=1e-6), "span wall/interval mismatch")
        if span["cuda_seconds"] is not None:
            _number(span["cuda_seconds"], "span CUDA seconds")
        spans[identifier] = span
    for identifier, span in spans.items():
        visited = {identifier}
        parent_id = span["parent_span_id"]
        while parent_id is not None:
            _require(parent_id in spans, "span parent is missing")
            _require(parent_id not in visited, "span parent cycle")
            visited.add(parent_id)
            parent = spans[parent_id]
            _require(parent["start_offset_seconds"] <= span["start_offset_seconds"] + 1e-6
                     and parent["end_offset_seconds"] + 1e-6 >= span["end_offset_seconds"],
                     "child interval outside parent")
            parent_id = parent["parent_span_id"]
    if record["status"] == "completed":
        _require(all(span["status"] in ("completed", "interrupted") for span in spans.values()),
                 "completed record contains failed span")
    _require(isinstance(record["not_applicable"], list), "not_applicable must be a list")
    na_names = set()
    for item in record["not_applicable"]:
        _require(isinstance(item, dict) and {"name", "reason"} <= item.keys(), "invalid not_applicable entry")
        name = _text(item["name"], "not_applicable name")
        _text(item["reason"], "not_applicable reason")
        _require(name not in na_names, "duplicate not_applicable name")
        _require(name not in {span["name"] for span in spans.values()}, "stage is both measured and not applicable")
        na_names.add(name)
    try:
        _json(record)
    except (TypeError, ValueError) as error:
        raise StageTimingAnalysisError("record contains non-JSON or nonfinite values") from error
    return record


def _family(name: str) -> str | None:
    # Deliberately do not classify descriptor_snapshot as physical KV snapshot.
    aliases = {"history_concat": "history_cat", "tensor_persist": "persist",
               "checkpoint_persist": "persist", "lifecycle_persist": "persist",
               "noise_persist": "persist", "tensor_hash": "persist",
               "noise_hash": "persist", "post_rpc_hash": "persist"}
    if name in aliases:
        return aliases[name]
    for family in CORE_STAGES:
        if name == family or name.startswith(family + ".") or name.startswith(family + "_"):
            return family
    return None


def _identity(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    return {"source": record["source"], "action": metadata.get("action"),
            "role": metadata.get("role"), "prompt_sha256": metadata.get("prompt_sha256"),
            "output_latent_path": metadata.get("output_latent_path"),
            "output_noise_path": metadata.get("output_noise_path"),
            "prompt_id_sha256": metadata.get("prompt_id_sha256"),
            "branch": metadata.get("branch", metadata.get("branch_id")),
            "chunk_index": metadata.get("chunk_index"),
            "history_chunk_count": metadata.get("history_chunk_count"), "status": record["status"]}


def _stats(values: Sequence[float]) -> dict[str, Any]:
    return {"count": len(values), "minimum": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None}



def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _interval_union(spans: Sequence[Mapping[str, Any]]) -> float:
    intervals = sorted((span["start_offset_seconds"], span["end_offset_seconds"])
                       for span in spans)
    total, last_start, last_end = 0.0, None, None
    for start, end in intervals:
        if last_start is None:
            last_start, last_end = start, end
        elif start <= last_end:
            last_end = max(last_end, end)
        else:
            total += last_end - last_start
            last_start, last_end = start, end
    return total + (last_end - last_start if last_start is not None else 0.0)


def analyze_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(records), "no timing records found")
    seen = set()
    request_groups: dict[str, Any] = {}
    stage_groups: dict[str, Any] = {}
    requests, span_rows, stage_totals = [], [], []
    for raw in records:
        record = _validate(raw)
        key = (record["source"], record["record_id"])
        _require(key not in seen, "duplicate timing record")
        seen.add(key)
        identity = _identity(record)
        measured = {_family(span["name"]) for span in record["spans"]}
        na = {_family(item["name"]) for item in record["not_applicable"]}
        coverage = {name: "measured" if name in measured else "not_applicable" if name in na else "missing"
                    for name in CORE_STAGES}
        request = {**identity, "record_id": record["record_id"], "pid": record["pid"],
                   "metadata": record["metadata"], "wall_seconds": record["wall_seconds"],
                   "cuda_sync_wall_seconds": record["cuda_sync_wall_seconds"], "coverage": coverage,
                   "not_applicable": record["not_applicable"]}
        requests.append(request)
        group = request_groups.setdefault(_json(identity), {"identity": identity, "wall": [], "sync": []})
        group["wall"].append(record["wall_seconds"])
        group["sync"].append(record["cuda_sync_wall_seconds"])
        for name in sorted({span["name"] for span in record["spans"]}):
            matching = [span for span in record["spans"] if span["name"] == name]
            stage_totals.append({**identity, "record_id": record["record_id"],
                "request_id": record["metadata"].get("request_id"), "stage": name,
                "span_count": len(matching),
                "failed_span_count": sum(span["status"] == "error" for span in matching),
                "inclusive_wall_interval_union_seconds": _interval_union(matching),
                "nonadditive_across_stages_or_sources": True})
        span_by_id = {span["span_id"]: span for span in record["spans"]}
        for span in record["spans"]:
            parent = span_by_id.get(span["parent_span_id"])
            row = {**identity, "record_id": record["record_id"], "pid": record["pid"],
                   "request_id": record["metadata"].get("request_id"), "metadata": record["metadata"],
                   "parent_name": parent["name"] if parent else None,
                   "record_status": record["status"], **span}
            span_rows.append(row)
            grouping = {**identity, "stage": span["name"], "stage_status": span["status"],
                        "parent_name": row["parent_name"], "detail": span["detail"]}
            group = stage_groups.setdefault(_json(grouping), {"identity": grouping, "wall": [], "cuda": [], "cuda_missing": 0})
            group["wall"].append(span["wall_seconds"])
            if span["cuda_seconds"] is None:
                group["cuda_missing"] += 1
            else:
                group["cuda"].append(span["cuda_seconds"])
    return {
        "schema": REPORT_SCHEMA, "record_count": len(requests),
        "failed_record_count": sum(row["status"] == "error" for row in requests),
        "interpretation": {
            "wall_policy": "inclusive spans; never add nested spans or client RPC to worker wall",
            "cuda_policy": "event elapsed intervals may include stream idle/host gaps; not pure GPU busy time",
            "missing_policy": "missing/unavailable is not zero; snapshot descriptors are not physical KV snapshots",
            "comparison_policy": "diagnostic instrumented runs; not a formal speedup or quality result",
        },
        "requests": requests, "spans": span_rows, "per_request_stage_totals": stage_totals,
        "request_summaries": [{**group["identity"], "wall_seconds": _stats(group["wall"]),
                               "cuda_sync_wall_seconds": _stats(group["sync"])}
                              for _, group in sorted(request_groups.items())],
        "stage_summaries": [{**group["identity"], "inclusive_wall_seconds": _stats(group["wall"]),
                             "cuda_elapsed_seconds": _stats(group["cuda"]),
                             "unavailable_cuda_count": group["cuda_missing"]}
                            for _, group in sorted(stage_groups.items())],
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    root = _safe_path(run_dir)
    directory = _safe_path(root / "provenance" / "stage-timing")
    _require(directory.is_dir(), "stage timing input is not a directory")
    records, sources = [], []
    for path in sorted(directory.glob("*.jsonl")):
        _safe_path(path)
        _require(path.is_file(), "timing input is not a file")
        payload = path.read_bytes()
        _require(not payload or payload.endswith(b"\n"), f"incomplete JSONL input: {path}")
        sources.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "nbytes": len(payload)})
        try:
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
                _require(bool(line.strip()), f"empty record in {path.name}:{line_number}")
                records.append(json.loads(line, object_pairs_hook=_unique_object))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise StageTimingAnalysisError(f"invalid/incomplete JSONL input: {path}") from error
    return {**analyze_records(records), "source_run_dir": str(root), "source_files": sources}


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    active_text = os.environ.get("CFEXP_RUN_DIR")
    _require(bool(active_text), "writing requires CFEXP_RUN_DIR")
    active = _safe_path(Path(active_text))
    output = _safe_path(output_dir, existing=False)
    _require(output != active and output.is_relative_to(active), "output must be below CFEXP_RUN_DIR")
    for directory in (output, *output.parents):
        _require(not any((directory / marker).exists() or (directory / marker).is_symlink() for marker in MARKERS),
                 "refusing to write inside a sealed run")
    _require(not output.exists(), "output directory already exists; reports are immutable")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")
    for name in ("requests", "spans", "request_summaries", "stage_summaries", "per_request_stage_totals"):
        rows = report[name]
        columns = sorted({key for row in rows for key in row})
        with (output / f"{name}.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _json(value) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="new directory below active CFEXP_RUN_DIR; omit for read-only stdout")
    args = parser.parse_args()
    report = analyze_run(args.run_dir)
    if args.output_dir is None:
        print(json.dumps(report, allow_nan=False, sort_keys=True))
    else:
        write_report(report, args.output_dir)


if __name__ == "__main__":
    main()
