"""Sealed, paired quality comparison for the frozen three-model queue.

This reports a custom six-dimension mean, not an official VBench total. Timing
observations are retained but speedup requires a separate matched-Target audit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from .baseline_population import load_baseline_population
from .duration_sweep_analysis import _resolved_experiment_manifest
from .main_table_merge import (
    _indexed_attempt_file, _load_frozen_index, _load_json, _require, _sha256_file,
)
from .main_table_v1 import QUALITY_DIMENSIONS
from .manifest import normalize_manifest

PAIRS = {"sf": "krea_self_forcing", "ll": "krea_longlive", "rolling": "krea_rolling_forcing"}
METHODS = ("ours", "sdvg", "random")
CELLS = tuple(f"{model}-{method}-{seconds}" for model in PAIRS
              for seconds in (15, 30, 60) for method in METHODS)


def _number(value, *, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value), "non-finite numeric measurement")
    _require(value > 0 if positive else 0 <= value <= 1, "measurement outside valid range")
    return float(value)


def _validate_duration(duration: dict, parameters: dict, method: str, seconds: int) -> None:
    # SDVG's frozen campaign schema names this field differently from Ours/Random.
    nominal_key = "nominal_duration_seconds" if method == "sdvg" else "nominal_seconds"
    _require(parameters.get(nominal_key) == seconds and duration.get("nominal_seconds") == seconds,
             "duration mismatch")
    _require(all(k in parameters and duration.get(k) == parameters[k]
                 for k in ("chunk_count", "frame_count", "fps")),
             "duration geometry differs from pinned manifest")


def validate_cell(attempt: Path, task: dict, repo: Path) -> dict:
    """Bind an entire 24-prompt result to its exact published queue invocation."""
    _require(not attempt.is_symlink(), "attempt may not be a symlink")
    attempt = attempt.resolve(strict=True)
    _require(attempt.parent.name == "attempts", "source must be an attempt directory")
    frozen, marker = _load_frozen_index(attempt.parent.parent)
    _require(marker["attempt_id"] == attempt.name and marker["git_sha"] == task["commit"],
             "attempt or published revision mismatch")
    run = _load_json(frozen.file("run.json", label="run"), label="run")
    _require(run.get("execution_status") == "SUCCEEDED" and run.get("exit_code") == 0
             and run.get("latest_attempt_id") == attempt.name and run.get("run_id") == marker["run_id"],
             "run terminal identity mismatch")
    resolved = _load_json(frozen.file("resolved-manifest.json", label="resolved"), label="resolved")
    manifest = _resolved_experiment_manifest(resolved, marker)
    expected = json.loads(subprocess.check_output(
        ["git", "show", task["commit"] + ":" + task["manifest"]], cwd=repo))
    _require(manifest == normalize_manifest(expected),
             "resolved invocation differs from pinned queue manifest")
    model, method, seconds_text = task["id"].split("-")
    seconds = int(seconds_text)
    prompts, population = load_baseline_population(repo, model_pair=PAIRS[model])
    ids = [p["id"] for p in prompts]
    benchmark = "VBench-Long" if seconds == 60 else "VBench"

    def read(relative):
        return _load_json(_indexed_attempt_file(frozen, attempt.name, relative, label=relative), label=relative)

    def bound(binding):
        path = _indexed_attempt_file(frozen, attempt.name, binding["path"], label="measurement")
        _require(_sha256_file(path) == (binding["sha256"], binding["nbytes"]), "measurement binding mismatch")
        return _load_json(path, label="measurement")

    result = read("result.json")
    _require(result.get("status") == "succeeded" and result.get("run_kind") == "formal_campaign"
             and result.get("method_id") == method and result.get("prompt_count") == 24
             and result.get("row_count") == 24 and result.get("seed_count") == 1
             and result.get("population") == population and result.get("benchmark") == benchmark,
             "not a matched complete 24-prompt result")
    duration = result.get("duration", {})
    parameters = manifest["spec"]["parameters"]
    _validate_duration(duration, parameters, method, seconds)
    rows = bound(result["raw_rows"])["rows"]
    _require(len(rows) == 24 and [r.get("prompt_id") for r in rows] == ids,
             "prompt coverage or order mismatch")
    evidence = result["vbench_measurements"]
    _require(len(evidence) == 24 and [e.get("prompt_id") for e in evidence] == ids,
             "quality evidence coverage mismatch")
    flag = "official_vbench_long_verified" if seconds == 60 else "official_vbench_verified"
    quality, times = {}, {}
    for row, item in zip(rows, evidence):
        pid = row["prompt_id"]
        _require(row.get("method_id") == method and type(row.get("seed")) is int
                 and row["seed"] == 20260911 and row.get(flag) is True
                 and row.get("duration") == duration, "row identity or provenance mismatch")
        scores = row.get("quality_scores", {})
        _require(set(scores) == set(QUALITY_DIMENSIONS), "quality dimensions mismatch")
        scores = {d: _number(scores[d]) for d in QUALITY_DIMENSIONS}
        sidecar = bound(item)
        _require(item["measurement_id"] == f"mt-{method}-{pid}"
                 and sidecar.get("measurement_id") == item["measurement_id"]
                 and sidecar.get("dimension_scores") == scores,
                 "quality row differs from evaluator evidence")
        quality[pid] = scores
        times[pid] = _number(row.get("generation_wall_seconds"), positive=True)
    return {"task": task["id"], "quality": quality, "generation_seconds": times,
            "benchmark": benchmark, "duration": duration,
            "source": {"attempt_dir": str(attempt), "git_sha": marker["git_sha"],
                       "artifact_index_sha256": marker["artifact_index_sha256"],
                       "destination": marker["artifact_publication"]["destination"]}}


def summarize(cells: dict) -> dict:
    _require(set(cells) <= set(CELLS), "unknown comparison cell")
    comparisons = []
    for model in PAIRS:
        for seconds in (15, 30, 60):
            keys = [f"{model}-{m}-{seconds}" for m in METHODS]
            if not all(k in cells for k in keys):
                continue
            arms = {m: cells[k] for m, k in zip(METHODS, keys)}
            ids = set(arms["ours"]["quality"])
            _require(len(ids) == 24 and all(set(a["quality"]) == ids for a in arms.values()),
                     "comparison prompt pairing mismatch")
            per_prompt = {m: {p: 100 * math.fsum(a["quality"][p].values()) / 6
                              for p in sorted(ids)} for m, a in arms.items()}
            means = {m: math.fsum(v.values()) / 24 for m, v in per_prompt.items()}
            comparisons.append({"model_pair": PAIRS[model], "nominal_seconds": seconds,
                "benchmark": arms["ours"]["benchmark"], "quality_mean_percent": means,
                "ours_minus_baseline_points": {m: means["ours"] - means[m] for m in ("sdvg", "random")},
                "per_prompt_quality_percent": per_prompt,
                "ours_higher_than_both_on_this_population": all(means["ours"] > means[m] for m in ("sdvg", "random")),
                "generation_seconds_by_method_and_prompt": {m: a["generation_seconds"] for m, a in arms.items()},
                "sources": {m: a["source"] for m, a in arms.items()}})
    return {"schema": "erank3_paired_quality_v1", "complete_quality_matrix": set(cells) == set(CELLS),
            "missing_cells": sorted(set(CELLS) - set(cells)),
            "quality_metric": "custom_unweighted_six_dimension_mean_percent_not_official_total",
            "speedup_verified": False,
            "speedup_note": "Requires separate matched hardware, timing scope and native KV runtime audit against pure Target.",
            "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True, help="JSON task ID to sealed attempt directory")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    tasks = {t["id"]: t for t in _load_json(args.plan, label="plan")["tasks"]}
    sources = _load_json(args.sources, label="sources")
    _require(set(sources) <= set(CELLS), "sources include an unknown comparison task")
    cells = {k: validate_cell(Path(v), tasks[k], args.repo) for k, v in sources.items()}
    print(json.dumps(summarize(cells), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
