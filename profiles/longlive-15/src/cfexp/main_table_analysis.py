"""Strict global analysis for the ``main-table-v1`` benchmark.

The analyzer consumes one sealed row for every method/prompt pair.  It keeps
quality aggregation separate from timing, computes speedup from paired prompt
latencies against Full 14B Target, and never averages pre-aggregated shard
means.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .main_table_v1 import METHOD_IDS, PROMPT_IDS, QUALITY_DIMENSIONS


RESULT_SCHEMA = "main_table_analysis_result_v1"
RAW_SCHEMA = "main_table_raw_rows_v1"
REFERENCE_METHOD_ID = "full_target"
SEED = 20260911


class MainTableAnalysisError(ValueError):
    """Raised when raw main-table evidence is incomplete or inconsistent."""


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MainTableAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise MainTableAnalysisError(f"{label} must be {qualifier}")
    return result


def analyze_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate sealed raw rows and calculate global quality and speedup data."""

    expected_count = len(METHOD_IDS) * len(PROMPT_IDS)
    if len(rows) != expected_count:
        raise MainTableAnalysisError(
            f"expected {expected_count} raw rows, observed {len(rows)}"
        )

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise MainTableAnalysisError(f"rows[{index}] must be an object")
        allowed = {
            "method_id",
            "prompt_id",
            "seed",
            "generation_wall_seconds",
            "quality_scores",
            "official_vbench_long_verified",
        }
        if set(raw) != allowed:
            raise MainTableAnalysisError(
                f"rows[{index}] fields must be exactly {sorted(allowed)!r}"
            )
        method_id = raw["method_id"]
        prompt_id = raw["prompt_id"]
        if method_id not in METHOD_IDS:
            raise MainTableAnalysisError(f"rows[{index}].method_id is unknown")
        if prompt_id not in PROMPT_IDS:
            raise MainTableAnalysisError(f"rows[{index}].prompt_id is unknown")
        if raw["seed"] != SEED:
            raise MainTableAnalysisError(f"rows[{index}].seed drifted")
        if raw["official_vbench_long_verified"] is not True:
            raise MainTableAnalysisError(
                f"rows[{index}] lacks verified official VBench-Long evidence"
            )
        key = (str(method_id), str(prompt_id))
        if key in indexed:
            raise MainTableAnalysisError(f"duplicate method/prompt row: {key!r}")
        latency = _number(
            raw["generation_wall_seconds"],
            f"rows[{index}].generation_wall_seconds",
            positive=True,
        )
        scores = raw["quality_scores"]
        if not isinstance(scores, Mapping) or set(scores) != set(QUALITY_DIMENSIONS):
            raise MainTableAnalysisError(
                f"rows[{index}].quality_scores must contain the six frozen dimensions"
            )
        normalized_scores = {
            dimension: _number(scores[dimension], f"rows[{index}].{dimension}")
            for dimension in QUALITY_DIMENSIONS
        }
        if any(not 0.0 <= value <= 1.0 for value in normalized_scores.values()):
            raise MainTableAnalysisError(
                f"rows[{index}].quality_scores must remain in [0,1]"
            )
        indexed[key] = {
            "latency": latency,
            "scores": normalized_scores,
            "prompt_macro_mean": sum(normalized_scores.values())
            / len(QUALITY_DIMENSIONS),
        }

    expected_keys = {
        (method_id, prompt_id)
        for method_id in METHOD_IDS
        for prompt_id in PROMPT_IDS
    }
    missing = sorted(expected_keys - set(indexed))
    unexpected = sorted(set(indexed) - expected_keys)
    if missing or unexpected:
        raise MainTableAnalysisError(
            f"method/prompt coverage mismatch: missing={missing!r}, unexpected={unexpected!r}"
        )

    methods: list[dict[str, Any]] = []
    for method_id in METHOD_IDS:
        dimension_means = {
            dimension: sum(
                indexed[(method_id, prompt_id)]["scores"][dimension]
                for prompt_id in PROMPT_IDS
            )
            / len(PROMPT_IDS)
            for dimension in QUALITY_DIMENSIONS
        }
        prompt_quality = [
            indexed[(method_id, prompt_id)]["prompt_macro_mean"]
            for prompt_id in PROMPT_IDS
        ]
        paired_speedups = [
            indexed[(REFERENCE_METHOD_ID, prompt_id)]["latency"]
            / indexed[(method_id, prompt_id)]["latency"]
            for prompt_id in PROMPT_IDS
        ]
        geometric_speedup = math.exp(
            sum(math.log(value) for value in paired_speedups) / len(paired_speedups)
        )
        if method_id == REFERENCE_METHOD_ID and not math.isclose(
            geometric_speedup, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise MainTableAnalysisError("Full Target self-speedup must equal one")
        methods.append(
            {
                "method_id": method_id,
                "prompt_count": len(PROMPT_IDS),
                "dimension_means": dimension_means,
                "quality_macro_mean": sum(dimension_means.values())
                / len(QUALITY_DIMENSIONS),
                "generation_wall_seconds_arithmetic_mean": sum(
                    indexed[(method_id, prompt_id)]["latency"]
                    for prompt_id in PROMPT_IDS
                )
                / len(PROMPT_IDS),
                "paired_speedups": paired_speedups,
                "paired_speedup_geometric_mean": geometric_speedup,
                "paired_quality_difference_vs_full_target_mean": sum(
                    prompt_quality[position]
                    - indexed[(REFERENCE_METHOD_ID, prompt_id)]["prompt_macro_mean"]
                    for position, prompt_id in enumerate(PROMPT_IDS)
                )
                / len(PROMPT_IDS),
            }
        )

    return {
        "schema": RESULT_SCHEMA,
        "status": "succeeded",
        "method_count": len(METHOD_IDS),
        "prompt_count": len(PROMPT_IDS),
        "seed_count": 1,
        "row_count": expected_count,
        "reference_method_id": REFERENCE_METHOD_ID,
        "speedup_aggregation": "geometric_mean_of_paired_per_prompt_speedups",
        "shard_means_used": False,
        "methods": methods,
    }


__all__ = [
    "MainTableAnalysisError",
    "RAW_SCHEMA",
    "REFERENCE_METHOD_ID",
    "RESULT_SCHEMA",
    "SEED",
    "analyze_rows",
]
