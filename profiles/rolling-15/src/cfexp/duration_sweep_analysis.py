"""Paired duration analysis; formal inputs must be sealed and indexed."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .duration_sweep import RAW_SCHEMA, RESULT_SCHEMA, duration_profile
from .main_table_analysis import _number
from .main_table_v1 import PROMPT_IDS, QUALITY_DIMENSIONS
from .main_table_merge import (
    _indexed_attempt_file, _load_frozen_index, _load_json, _require, _sha256_file,
)
from . import optimized_duration_control as optimized_control


def _resolved_experiment_manifest(resolved, marker):
    # runner.run_experiment writes an execution envelope around the manifest.
    # Bind that envelope to this sealed attempt before checking its command.
    _require(isinstance(resolved, dict) and isinstance(resolved.get("manifest"), dict),
             "resolved manifest must contain the runner execution envelope")
    _require(all(resolved.get(key) == marker[key] for key in ("run_id", "attempt_id", "spec_hash")),
             "resolved manifest terminal identity mismatch")
    return resolved["manifest"]


def _verify_optimized_control(result, preflight, manifest) -> None:
    expected = optimized_control.contract()
    _require(result.get("comparison_contract") == expected
             and preflight.get("comparison_contract") == expected,
             "optimized comparison contract mismatch")
    method = result["method_id"]
    seconds = result["duration"]["nominal_seconds"]
    command = manifest["spec"]["command"]
    argv = command["argv"]
    _require(manifest["metadata"]["name"] == f"optimized-target-duration-{seconds}s-{method.replace('_', '-')}-ntu-v2"
             and argv.count("--optimized-duration-control") == 1
             and argv.count("--duration-seconds") == 1
             and argv[argv.index("--duration-seconds") + 1] == str(seconds)
             and "--smoke-prompt-id" not in argv,
             "optimized comparison invocation mismatch")
    if method == "ours":
        _require(all(command["env"].get(k) == v for k, v in optimized_control.OURS_FLAGS.items()),
                 "optimized feature flags mismatch")
        receipt = result["calibration_receipt"]
        _require(receipt.get("receipt_sha256") == optimized_control.CALIBRATION_RECEIPT_SHA256
                 and receipt.get("code_revision") == optimized_control.CALIBRATION_REVISION,
                 "optimized comparison calibration source mismatch")


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require the complete 3 x 2 x 12 paired population before aggregation."""
    indexed = {}
    fields = {"method_id", "prompt_id", "seed", "generation_wall_seconds",
              "quality_scores", "official_vbench_long_verified", "duration"}
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == fields, "duration row fields drifted")
        seconds = row["duration"].get("nominal_seconds") if isinstance(row["duration"], dict) else None
        profile = duration_profile(seconds)
        _require(row["duration"] == profile.metadata(), "duration geometry/budget drifted")
        _require(row["method_id"] in ("ours", "full_target") and row["prompt_id"] in PROMPT_IDS,
                 "unknown duration method/prompt")
        _require(type(row["seed"]) is int and row["seed"] == 20260911
                 and row["official_vbench_long_verified"] is True, "seed or quality provenance drifted")
        _number(row["generation_wall_seconds"], "generation_wall_seconds", positive=True)
        scores = row["quality_scores"]
        _require(isinstance(scores, dict) and set(scores) == set(QUALITY_DIMENSIONS), "quality dimensions drifted")
        for score in scores.values():
            _require(0 <= _number(score, "quality score") <= 1, "quality score outside [0,1]")
        key = (seconds, row["method_id"], row["prompt_id"])
        _require(key not in indexed, "duplicate duration/method/prompt")
        indexed[key] = row
    _require(len(indexed) == 72, "duration analysis requires all 72 rows")
    results = []
    for seconds in (15, 30, 60):
        pairs = []
        for prompt in PROMPT_IDS:
            ours = indexed[(seconds, "ours", prompt)]
            target = indexed[(seconds, "full_target", prompt)]
            pairs.append({"prompt_id": prompt,
                          "ours_seconds": ours["generation_wall_seconds"],
                          "target_seconds": target["generation_wall_seconds"],
                          "speedup": target["generation_wall_seconds"] / ours["generation_wall_seconds"]})
        speedup = math.exp(math.fsum(math.log(p["speedup"]) for p in pairs) / len(pairs))
        quality = {method: {dim: math.fsum(indexed[(seconds, method, prompt)]["quality_scores"][dim]
                                          for prompt in PROMPT_IDS) / len(PROMPT_IDS)
                            for dim in QUALITY_DIMENSIONS}
                   for method in ("ours", "full_target")}
        results.append({"duration": duration_profile(seconds).metadata(),
                        "paired_geometric_mean_speedup": speedup,
                        "speedup_at_least_2x": speedup >= 2.0,
                        "per_prompt": pairs, "quality_dimension_means": quality,
                        "quality_dimension_deltas_ours_minus_target": {
                            dim: quality["ours"][dim] - quality["full_target"][dim]
                            for dim in QUALITY_DIMENSIONS}})
    return {"schema": "ours_target_duration_analysis_v1", "row_count": 72,
            "calibration_in_generation_latency": False, "durations": results}


def analyze_attempts(attempt_dirs: Sequence[Path], *, require_optimized_control: bool = False) -> dict[str, Any]:
    _require(len(attempt_dirs) == 6, "six formal duration attempts are required")
    rows, sources, receipts, controls = [], [], [], []
    for requested in attempt_dirs:
        _require(not requested.is_symlink(), "attempt may not be a symlink")
        attempt = requested.resolve(strict=True)
        _require(attempt.parent.name == "attempts", "source must be an attempt directory")
        frozen, marker = _load_frozen_index(attempt.parent.parent)
        _require(marker["attempt_id"] == attempt.name, "marker attempt mismatch")
        run = _load_json(frozen.file("run.json", label="run status"), label="run status")
        _require(run.get("execution_status") == "SUCCEEDED" and run.get("exit_code") == 0
                 and run.get("latest_attempt_id") == attempt.name and run.get("run_id") == marker["run_id"],
                 "run terminal identity mismatch")
        def bound_file(binding):
            path = _indexed_attempt_file(frozen, attempt.name, binding["path"], label="bound evidence")
            _require(_sha256_file(path) == (binding["sha256"], binding["nbytes"]), "evidence binding mismatch")
            return path
        result = _load_json(_indexed_attempt_file(frozen, attempt.name, "result.json", label="result"), label="result")
        _require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "succeeded"
                 and result.get("run_kind") == "formal_campaign" and result.get("prompt_count") == 12
                 and result.get("row_count") == 12 and result.get("seed_count") == 1,
                 "not a complete formal duration result")
        raw = _load_json(bound_file(result["raw_rows"]), label="raw rows")
        _require(raw.get("schema") == RAW_SCHEMA and len(raw.get("rows", [])) == 12, "raw duration coverage mismatch")
        _require(tuple(row["prompt_id"] for row in raw["rows"]) == PROMPT_IDS
                 and all(row["method_id"] == result["method_id"] and row["duration"] == result["duration"] for row in raw["rows"]),
                 "result/raw identity mismatch")
        evidence = result["vbench_measurements"]
        _require(tuple(item["prompt_id"] for item in evidence) == PROMPT_IDS, "quality evidence coverage mismatch")
        for item in evidence:
            _require(item["measurement_id"] == f'mt-{result["method_id"]}-{item["prompt_id"]}', "quality measurement identity mismatch")
            bound_file(item)
        if result["method_id"] == "ours":
            _require(result["calibration_receipt"]["mode"] == "read_only_reuse", "formal duration must reuse frozen calibration")
            receipts.append(result["calibration_receipt"]["receipt_sha256"])
        control = result.get("comparison_contract")
        controls.append(control)
        if control is not None or require_optimized_control:
            preflight = _load_json(_indexed_attempt_file(frozen, attempt.name, "provenance/preflight.json", label="preflight"), label="preflight")
            resolved = _load_json(frozen.file("resolved-manifest.json", label="manifest"), label="manifest")
            manifest = _resolved_experiment_manifest(resolved, marker)
            _verify_optimized_control(result, preflight, manifest)
        rows.extend(raw["rows"])
        sources.append({"attempt_dir": str(attempt), "git_sha": marker["git_sha"],
                        "destination": marker["artifact_publication"]["destination"],
                        "artifact_index_sha256": marker["artifact_index_sha256"]})
    _require(len({s["git_sha"] for s in sources}) == 1, "duration methods must use the same exact revision")
    _require(len(receipts) == 3 and len(set(receipts)) == 1, "Ours horizons must share the exact frozen calibration receipt")
    _require(all(c == controls[0] for c in controls), "duration methods mix different comparison controls")
    result = analyze_rows(rows)
    return {**result, "sources": sources, "comparison_contract": controls[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, action="append", required=True)
    parser.add_argument("--require-optimized-control", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze_attempts(args.attempt_dir, require_optimized_control=args.require_optimized_control), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
