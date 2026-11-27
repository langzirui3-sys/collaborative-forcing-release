"""Strict validation, prompt-cluster statistics, and figures for Insight 2.

No function in this module fabricates or imputes measurements.  Missing
branches, route repeats, scores, or Target-call accounting fail before a
summary or figure is written.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from .correction_policy import candidate_chunks, random_positions, uniform_positions
from .erank_quality import ProtocolError, spearman


INSIGHT2A_ANALYSIS_SCHEMA = "insight2a_analysis_v1"
INSIGHT2B_ANALYSIS_SCHEMA = "insight2b_analysis_v2"
FIGURE2A_DATA_SCHEMA = "insight2a_figure_data_v1"
FIGURE2B_DATA_SCHEMA = "insight2b_figure_data_v2"
RISK_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
STRATEGIES_2B = (
    "random",
    "uniform",
    "online_rank_guided",
    "greedy_oracle",
)
ALLOWED_STRATEGIES_2B = frozenset(
    (
        *STRATEGIES_2B,
        "online_coverage_guided",
        "online_lookahead_coverage_guided",
        "online_lookahead_coverage_boundary_gated",
        "online_erank_drop",
    )
)


class AnalysisError(ProtocolError):
    """Raised when raw experiment output cannot support the promised claim."""


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise AnalysisError(f"{path}: {message}")


def _finite_float(value: Any, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{path}: expected a finite number") from error
    _require(math.isfinite(result), path, "expected a finite number")
    return result


def _integer(value: Any, path: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{path}: expected an integer") from error
    _require(str(result) == str(value).strip(), path, "expected canonical integer")
    return result


def _sha256(value: Any, path: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        path,
        "expected lowercase SHA-256",
    )
    return value


@dataclass(frozen=True)
class Insight2ARow:
    prompt_id: str
    seed: int
    chunk_index: int
    rank: float
    risk_score: float
    q_d: float
    q_t: float
    delta_q: float
    tc_d: float
    tc_t: float
    delta_tc: float
    prefix_hash: str
    condition_hash: str
    noise_hash: str
    draft_kv_hash: str
    target_kv_hash: str
    hidden_state_path: str
    hidden_state_sha256: str


@dataclass(frozen=True)
class Insight2BRow:
    prompt_id: str
    seed: int
    budget: int
    strategy: str
    route_repeat: int | None
    selected_chunks: tuple[int, ...]
    proposed_target_chunks: tuple[int, ...]
    full_quality: float
    temporal_consistency: float
    budgeted_target_calls: int
    committed_target_chunks: int
    physical_target_calls: int
    target_rpc_attempted: int
    target_rpc_backend_returned: int
    target_rpc_completed: int
    route_materialized_logical_target_calls: int
    selection_search_rollout_count: int
    selection_search_target_calls: int
    selection_search_target_rpc_attempted: int
    selection_search_target_rpc_backend_returned: int
    selection_search_target_rpc_completed: int
    final_history_hash: str

    @property
    def route_materialization_target_calls(self) -> int:
        """Successful physical calls attributable to the reported 1+K route.

        Greedy Oracle performs an offline selection search before reporting its
        chosen route.  Those search calls are deliberately excluded here so
        they cannot be mistaken for the scientific correction budget.
        """

        return self.physical_target_calls - self.selection_search_target_calls

    @property
    def route_cache_hits(self) -> int:
        """Budgeted Target calls served without a new successful Target RPC."""

        return (
            self.budgeted_target_calls
            - self.route_materialized_logical_target_calls
        )

    @property
    def route_materialization_target_rpc_attempted(self) -> int:
        """RPC attempts attributable to route materialization, excluding search."""

        return self.target_rpc_attempted - self.selection_search_target_rpc_attempted


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = Path(path)
    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise AnalysisError(f"cannot read CSV {csv_path}: {error}") from error
    _require(bool(fields), "$csv", "missing header")
    _require(len(fields) == len(set(fields)), "$csv.header", "duplicate column")
    return fields, rows


def load_insight2a_rows(
    path: str | Path,
    *,
    expected_prompt_ids: Sequence[str],
    expected_seeds: Sequence[int],
    chunk_count: int = 16,
) -> list[Insight2ARow]:
    """Load the complete prompt × seed × candidate-chunk correction grid."""

    required = {
        "prompt_id",
        "seed",
        "chunk_index",
        "rank",
        "risk_score",
        "Q_D",
        "Q_T",
        "delta_Q",
        "TC_D",
        "TC_T",
        "delta_TC",
        "prefix_hash",
        "condition_hash",
        "noise_hash",
        "draft_kv_hash",
        "target_kv_hash",
        "hidden_state_path",
        "hidden_state_sha256",
    }
    fields, raw_rows = _read_csv(path)
    _require(
        required <= set(fields), "$csv.header", f"missing columns {sorted(required - set(fields))}"
    )
    prompts = tuple(expected_prompt_ids)
    seeds = tuple(expected_seeds)
    candidates = candidate_chunks(chunk_count)
    _require(
        prompts and len(prompts) == len(set(prompts)),
        "$expected_prompt_ids",
        "must be non-empty and unique",
    )
    _require(
        seeds and len(seeds) == len(set(seeds)), "$expected_seeds", "must be non-empty and unique"
    )
    expected_grid = {
        (prompt_id, seed, chunk_index)
        for prompt_id in prompts
        for seed in seeds
        for chunk_index in candidates
    }
    rows: list[Insight2ARow] = []
    keys: list[tuple[str, int, int]] = []
    for index, raw in enumerate(raw_rows):
        base = f"$rows[{index}]"
        prompt_id = raw["prompt_id"]
        seed = _integer(raw["seed"], f"{base}.seed")
        chunk_index = _integer(raw["chunk_index"], f"{base}.chunk_index")
        _require(prompt_id in prompts, f"{base}.prompt_id", "not in the frozen split")
        _require(seed in seeds, f"{base}.seed", "not in the frozen seed set")
        _require(
            chunk_index in candidates,
            f"{base}.chunk_index",
            f"must be in [1, {chunk_count - 1}]",
        )
        rank = _finite_float(raw["rank"], f"{base}.rank")
        risk = _finite_float(raw["risk_score"], f"{base}.risk_score")
        _require(rank >= 1.0, f"{base}.rank", "effective rank must be at least one")
        _require(0.0 <= risk <= 1.0, f"{base}.risk_score", "must be in [0, 1]")
        q_d = _finite_float(raw["Q_D"], f"{base}.Q_D")
        q_t = _finite_float(raw["Q_T"], f"{base}.Q_T")
        delta_q = _finite_float(raw["delta_Q"], f"{base}.delta_Q")
        tc_d = _finite_float(raw["TC_D"], f"{base}.TC_D")
        tc_t = _finite_float(raw["TC_T"], f"{base}.TC_T")
        delta_tc = _finite_float(raw["delta_TC"], f"{base}.delta_TC")
        _require(abs(delta_q - (q_t - q_d)) <= 1e-9, f"{base}.delta_Q", "must equal Q_T - Q_D")
        _require(
            abs(delta_tc - (tc_t - tc_d)) <= 1e-9, f"{base}.delta_TC", "must equal TC_T - TC_D"
        )
        for label, value in (("Q_D", q_d), ("Q_T", q_t), ("TC_D", tc_d), ("TC_T", tc_t)):
            _require(0.0 <= value <= 1.0, f"{base}.{label}", "must be in [0, 1]")
        prefix_hash = _sha256(raw["prefix_hash"], f"{base}.prefix_hash")
        condition_hash = _sha256(raw["condition_hash"], f"{base}.condition_hash")
        noise_hash = _sha256(raw["noise_hash"], f"{base}.noise_hash")
        draft_kv_hash = _sha256(raw["draft_kv_hash"], f"{base}.draft_kv_hash")
        target_kv_hash = _sha256(raw["target_kv_hash"], f"{base}.target_kv_hash")
        hidden_path = raw["hidden_state_path"]
        _require(
            bool(hidden_path)
            and not Path(hidden_path).is_absolute()
            and ".." not in Path(hidden_path).parts,
            f"{base}.hidden_state_path",
            "must be a safe relative path",
        )
        hidden_hash = _sha256(raw["hidden_state_sha256"], f"{base}.hidden_state_sha256")
        rows.append(
            Insight2ARow(
                prompt_id=prompt_id,
                seed=seed,
                chunk_index=chunk_index,
                rank=rank,
                risk_score=risk,
                q_d=q_d,
                q_t=q_t,
                delta_q=delta_q,
                tc_d=tc_d,
                tc_t=tc_t,
                delta_tc=delta_tc,
                prefix_hash=prefix_hash,
                condition_hash=condition_hash,
                noise_hash=noise_hash,
                draft_kv_hash=draft_kv_hash,
                target_kv_hash=target_kv_hash,
                hidden_state_path=hidden_path,
                hidden_state_sha256=hidden_hash,
            )
        )
        keys.append((prompt_id, seed, chunk_index))
    _require(len(keys) == len(set(keys)), "$rows", "duplicate prompt/seed/chunk row")
    actual_grid = set(keys)
    _require(
        actual_grid == expected_grid,
        "$rows",
        f"incomplete grid: missing={len(expected_grid - actual_grid)}, extra={len(actual_grid - expected_grid)}",
    )
    return sorted(
        rows, key=lambda row: (prompts.index(row.prompt_id), seeds.index(row.seed), row.chunk_index)
    )


T = TypeVar("T")


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    _require(values, "$quantile", "cannot summarize an empty sequence")
    _require(0.0 <= probability <= 1.0, "$probability", "must be in [0, 1]")
    ordered = sorted(_finite_float(value, "$quantile.value") for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap(
    rows: Sequence[T],
    *,
    cluster: Callable[[T], str],
    statistic: Callable[[Sequence[T]], float],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    _require(replicates >= 1, "$bootstrap.replicates", "must be positive")
    clusters: dict[str, list[T]] = {}
    for row in rows:
        clusters.setdefault(cluster(row), []).append(row)
    labels = sorted(clusters)
    _require(len(labels) >= 2, "$bootstrap.clusters", "need at least two prompt clusters")
    generator = random.Random(seed)
    estimates: list[float] = []
    attempts = 0
    maximum_attempts = replicates * 100
    while len(estimates) < replicates and attempts < maximum_attempts:
        attempts += 1
        sampled: list[T] = []
        for label in generator.choices(labels, k=len(labels)):
            sampled.extend(clusters[label])
        try:
            value = float(statistic(sampled))
        except (AnalysisError, ProtocolError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(value):
            estimates.append(value)
    _require(len(estimates) == replicates, "$bootstrap", "could not obtain all finite replicates")
    return _type7_quantile(estimates, 0.025), _type7_quantile(estimates, 0.975)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    _require(bool(materialized), "$mean", "cannot average an empty sequence")
    return math.fsum(materialized) / len(materialized)


def _risk_bin(value: float) -> int:
    for index, (left, right) in enumerate(zip(RISK_BIN_EDGES, RISK_BIN_EDGES[1:])):
        if left <= value < right or (index == 3 and value == 1.0):
            return index
    raise AnalysisError(f"risk value outside [0,1]: {value}")


def analyze_insight2a(
    rows: Sequence[Insight2ARow],
    *,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20261012,
    chunk_count: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Return summary, per-video Top-1 rows, and canonical Figure 2(a) data."""

    _require(bool(rows), "$rows", "no measurements")
    candidates = candidate_chunks(chunk_count)
    pooled = spearman([row.risk_score for row in rows], [row.delta_q for row in rows])
    correlation_ci = _cluster_bootstrap(
        rows,
        cluster=lambda row: row.prompt_id,
        statistic=lambda sample: spearman(
            [row.risk_score for row in sample], [row.delta_q for row in sample]
        ),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    bin_summaries: list[dict[str, Any]] = []
    for bin_index, (left, right) in enumerate(zip(RISK_BIN_EDGES, RISK_BIN_EDGES[1:])):
        selected = [row for row in rows if _risk_bin(row.risk_score) == bin_index]
        _require(bool(selected), f"$risk_bins[{bin_index}]", "empty risk bin")
        ci = _cluster_bootstrap(
            selected,
            cluster=lambda row: row.prompt_id,
            statistic=lambda sample: _mean(row.delta_q for row in sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 100 + bin_index,
        )
        bin_summaries.append(
            {
                "left": left,
                "right": right,
                "right_inclusive": bin_index == 3,
                "n_chunks": len(selected),
                "n_prompts": len({row.prompt_id for row in selected}),
                "mean_risk": _mean(row.risk_score for row in selected),
                "mean_delta_Q": _mean(row.delta_q for row in selected),
                "ci95": list(ci),
            }
        )

    grouped: dict[tuple[str, int], list[Insight2ARow]] = {}
    for row in rows:
        grouped.setdefault((row.prompt_id, row.seed), []).append(row)
    top1_rows: list[dict[str, Any]] = []
    within_video_correlations: list[float] = []
    for (prompt_id, seed), sample in sorted(grouped.items()):
        ordered = sorted(sample, key=lambda row: row.chunk_index)
        _require(
            [row.chunk_index for row in ordered] == list(candidates),
            "$top1",
            f"each video must have all {len(candidates)} positions",
        )
        # Risk is a calibrated empirical CDF and can saturate/tie.  The deployable
        # ranking signal itself is lower effective rank, so Top-1 must select on
        # the original continuous rank rather than its discretized risk mapping.
        rank_choice = min(ordered, key=lambda row: (row.rank, row.chunk_index))
        oracle_choice = min(ordered, key=lambda row: (-row.delta_q, row.chunk_index))
        random_gain = _mean(row.delta_q for row in ordered)
        top1_rows.append(
            {
                "prompt_id": prompt_id,
                "seed": seed,
                "rank_top1_chunk": rank_choice.chunk_index,
                "rank_top1_gain": rank_choice.delta_q,
                "random_top1_expected_gain": random_gain,
                "oracle_top1_chunk": oracle_choice.chunk_index,
                "oracle_top1_gain": oracle_choice.delta_q,
            }
        )
        try:
            within_video_correlations.append(
                spearman(
                    [row.risk_score for row in ordered],
                    [row.delta_q for row in ordered],
                )
            )
        except ProtocolError:
            pass

    strategy_fields = {
        "rank_top1": "rank_top1_gain",
        "random_top1_exact_expectation": "random_top1_expected_gain",
        "oracle_top1": "oracle_top1_gain",
    }
    top1_summary: dict[str, Any] = {}
    for offset, (strategy, field) in enumerate(strategy_fields.items()):
        mean_value = _mean(float(row[field]) for row in top1_rows)
        ci = _cluster_bootstrap(
            top1_rows,
            cluster=lambda row: str(row["prompt_id"]),
            statistic=lambda sample, field=field: _mean(float(row[field]) for row in sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1000 + offset,
        )
        top1_summary[strategy] = {"mean_gain": mean_value, "ci95": list(ci)}

    temporal_fields = {
        "TC_D": lambda row: row.tc_d,
        "TC_T": lambda row: row.tc_t,
        "delta_TC": lambda row: row.delta_tc,
    }
    temporal_summary: dict[str, Any] = {}
    for offset, (label, accessor) in enumerate(temporal_fields.items()):
        ci = _cluster_bootstrap(
            rows,
            cluster=lambda row: row.prompt_id,
            statistic=lambda sample, accessor=accessor: _mean(accessor(row) for row in sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 2000 + offset,
        )
        temporal_summary[label] = {
            "mean": _mean(accessor(row) for row in rows),
            "ci95": list(ci),
        }

    # Temporal gain can be exactly constant for a degenerate pilot.  Keep the
    # paired means and their CIs in that case, while marking the optional
    # correlation as undefined instead of inventing a number.
    try:
        temporal_rho = spearman(
            [row.risk_score for row in rows],
            [row.delta_tc for row in rows],
        )
        temporal_rho_ci = _cluster_bootstrap(
            rows,
            cluster=lambda row: row.prompt_id,
            statistic=lambda sample: spearman(
                [row.risk_score for row in sample],
                [row.delta_tc for row in sample],
            ),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 2100,
        )
        temporal_summary["risk_vs_delta_TC_spearman"] = {
            "estimate": temporal_rho,
            "ci95": list(temporal_rho_ci),
        }
    except (AnalysisError, ProtocolError):
        temporal_summary["risk_vs_delta_TC_spearman"] = {
            "estimate": None,
            "ci95": None,
            "undefined_reason": "constant_or_degenerate_vector",
        }

    summary = {
        "schema": INSIGHT2A_ANALYSIS_SCHEMA,
        "n_rows": len(rows),
        "n_prompts": len({row.prompt_id for row in rows}),
        "n_videos": len(grouped),
        "bootstrap": {
            "cluster": "prompt_id",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "ci": "two_sided_percentile_95",
            "quantile": "Hyndman-Fan type 7",
        },
        "spearman": {"estimate": pooled, "ci95": list(correlation_ci)},
        "mean_within_video_spearman": (
            _mean(within_video_correlations) if within_video_correlations else None
        ),
        "risk_bins": bin_summaries,
        "top1": top1_summary,
        "temporal_consistency": temporal_summary,
    }
    figure_data = {
        "schema": FIGURE2A_DATA_SCHEMA,
        "x": "representation_degradation_risk",
        "y": "delta_Q",
        "points": [
            {
                "prompt_id": row.prompt_id,
                "seed": row.seed,
                "chunk_index": row.chunk_index,
                "risk_score": row.risk_score,
                "delta_Q": row.delta_q,
            }
            for row in rows
        ],
        "risk_bins": bin_summaries,
        "spearman": summary["spearman"],
    }
    return summary, top1_rows, figure_data


def _parse_selected_chunks(
    value: str,
    *,
    budget: int,
    path: str,
    chunk_count: int = 16,
    require_exact_budget: bool = True,
) -> tuple[int, ...]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise AnalysisError(f"{path}: expected a JSON integer list") from error
    _require(isinstance(raw, list), path, "expected list")
    chunks = tuple(_integer(item, f"{path}[]") for item in raw)
    if require_exact_budget:
        _require(len(chunks) == budget, path, "length must equal budget")
    else:
        _require(len(chunks) <= budget, path, "length must not exceed budget")
    _require(len(chunks) == len(set(chunks)), path, "positions must be unique")
    candidates = candidate_chunks(chunk_count)
    _require(
        all(item in candidates for item in chunks),
        path,
        f"positions must be in [1,{chunk_count - 1}]",
    )
    _require(tuple(sorted(chunks)) == chunks, path, "positions must be sorted")
    return chunks


def load_insight2b_rows(
    path: str | Path,
    *,
    protocol_id: str,
    expected_prompt_ids: Sequence[str],
    expected_seeds: Sequence[int],
    random_repeats: int = 10,
    chunk_count: int = 16,
    expected_budgets: Sequence[int] = (2, 3, 4),
    expected_strategies: Sequence[str] = STRATEGIES_2B,
) -> list[Insight2BRow]:
    required = {
        "prompt_id",
        "seed",
        "budget",
        "strategy",
        "route_repeat",
        "selected_chunks",
        "proposed_target_chunks",
        "full_quality",
        "temporal_consistency",
        "budgeted_target_calls",
        "committed_target_chunks",
        "physical_target_calls",
        "target_rpc_attempted",
        "target_rpc_backend_returned",
        "target_rpc_completed",
        "route_materialized_logical_target_calls",
        "selection_search_rollout_count",
        "selection_search_target_calls",
        "selection_search_target_rpc_attempted",
        "selection_search_target_rpc_backend_returned",
        "selection_search_target_rpc_completed",
        "final_history_hash",
    }
    fields, raw_rows = _read_csv(path)
    _require(
        required <= set(fields), "$csv.header", f"missing columns {sorted(required - set(fields))}"
    )
    prompts = tuple(expected_prompt_ids)
    seeds = tuple(expected_seeds)
    budgets = tuple(expected_budgets)
    strategies = tuple(expected_strategies)
    _require(
        budgets
        and len(budgets) == len(set(budgets))
        and all(isinstance(item, int) and not isinstance(item, bool) for item in budgets),
        "$expected_budgets",
        "must contain unique integer budgets",
    )
    _require(
        all(1 <= item < chunk_count for item in budgets),
        "$expected_budgets",
        f"budgets must be in [1,{chunk_count - 1}]",
    )
    _require(
        strategies
        and len(strategies) == len(set(strategies))
        and all(item in ALLOWED_STRATEGIES_2B for item in strategies),
        "$expected_strategies",
        "must contain unique supported strategies",
    )
    _require(
        isinstance(random_repeats, int) and not isinstance(random_repeats, bool),
        "$random_repeats",
        "must be an integer",
    )
    if "random" in strategies:
        _require(random_repeats >= 2, "$random_repeats", "must be at least 2")
    else:
        _require(
            random_repeats == 0,
            "$random_repeats",
            "must be 0 when the Random strategy is absent",
        )
    rows: list[Insight2BRow] = []
    seen: set[tuple[str, int, int, str, int | None]] = set()
    for index, raw in enumerate(raw_rows):
        base = f"$rows[{index}]"
        prompt_id = raw["prompt_id"]
        seed = _integer(raw["seed"], f"{base}.seed")
        budget = _integer(raw["budget"], f"{base}.budget")
        strategy = raw["strategy"]
        _require(prompt_id in prompts, f"{base}.prompt_id", "not in frozen split")
        _require(seed in seeds, f"{base}.seed", "not in frozen seeds")
        _require(budget in budgets, f"{base}.budget", "not in the frozen budget set")
        _require(strategy in strategies, f"{base}.strategy", "not in the frozen strategy set")
        repeat_text = raw["route_repeat"].strip()
        route_repeat = _integer(repeat_text, f"{base}.route_repeat") if repeat_text else None
        if strategy == "random":
            _require(
                route_repeat is not None and 0 <= route_repeat < random_repeats,
                f"{base}.route_repeat",
                f"random needs repeat 0..{random_repeats - 1}",
            )
        else:
            _require(
                route_repeat is None,
                f"{base}.route_repeat",
                "deterministic strategies leave repeat blank",
            )
        boundary_gated = (
            strategy == "online_lookahead_coverage_boundary_gated"
        )
        selected = _parse_selected_chunks(
            raw["selected_chunks"],
            budget=budget,
            path=f"{base}.selected_chunks",
            chunk_count=chunk_count,
            require_exact_budget=not boundary_gated,
        )
        proposed = _parse_selected_chunks(
            raw["proposed_target_chunks"],
            budget=budget,
            path=f"{base}.proposed_target_chunks",
            chunk_count=chunk_count,
        )
        if boundary_gated:
            _require(
                set(selected) <= set(proposed),
                f"{base}.selected_chunks",
                "accepted commits must be a subset of proposed_target_chunks",
            )
        if strategy == "random":
            assert route_repeat is not None
            expected_selected = random_positions(
                protocol_id=protocol_id,
                prompt_id=prompt_id,
                generation_seed=seed,
                budget=budget,
                repeat_id=route_repeat,
                chunk_count=chunk_count,
            )
            _require(
                selected == expected_selected,
                f"{base}.selected_chunks",
                "does not match the frozen Random route",
            )
        elif strategy == "uniform":
            _require(
                selected == uniform_positions(budget, chunk_count=chunk_count),
                f"{base}.selected_chunks",
                "does not match the frozen Uniform route",
            )
        elif strategy in {
            "online_coverage_guided",
            "online_lookahead_coverage_guided",
            "online_lookahead_coverage_boundary_gated",
        }:
            coverage_chunks = proposed if boundary_gated else selected
            coverage_windows = (
                (13, 21),
                (30, 38),
                (47, 55),
                (65, 73),
                (82, 90),
                (99, 107),
            )
            _require(
                chunk_count == 120 and budget == 6,
                (
                    f"{base}.proposed_target_chunks"
                    if boundary_gated
                    else f"{base}.selected_chunks"
                ),
                "coverage-guided rows require the frozen N=120, K=6 profile",
            )
            _require(
                all(
                    left <= chunk <= right
                    for chunk, (left, right) in zip(
                        coverage_chunks, coverage_windows
                    )
                ),
                (
                    f"{base}.proposed_target_chunks"
                    if boundary_gated
                    else f"{base}.selected_chunks"
                ),
                "coverage-guided route must select exactly one chunk in every frozen window",
            )
            coverage_gaps = (
                coverage_chunks[0],
                *(
                    right - left
                    for left, right in zip(
                        coverage_chunks, coverage_chunks[1:]
                    )
                ),
                (chunk_count - 1) - coverage_chunks[-1],
            )
            _require(
                all(8 <= gap <= 20 for gap in coverage_gaps),
                (
                    f"{base}.proposed_target_chunks"
                    if boundary_gated
                    else f"{base}.selected_chunks"
                ),
                "coverage-guided route violates the frozen 8..20 chunk gap",
            )
        if not boundary_gated:
            _require(
                proposed == selected,
                f"{base}.proposed_target_chunks",
                "legacy exact-K strategies require proposed_target_chunks == selected_chunks",
            )
        quality = _finite_float(raw["full_quality"], f"{base}.full_quality")
        temporal = _finite_float(raw["temporal_consistency"], f"{base}.temporal_consistency")
        _require(0.0 <= quality <= 1.0, f"{base}.full_quality", "must be in [0,1]")
        _require(0.0 <= temporal <= 1.0, f"{base}.temporal_consistency", "must be in [0,1]")
        budgeted = _integer(
            raw["budgeted_target_calls"], f"{base}.budgeted_target_calls"
        )
        committed = _integer(raw["committed_target_chunks"], f"{base}.committed_target_chunks")
        physical = _integer(raw["physical_target_calls"], f"{base}.physical_target_calls")
        rpc_attempted = _integer(raw["target_rpc_attempted"], f"{base}.target_rpc_attempted")
        rpc_backend_returned = _integer(
            raw["target_rpc_backend_returned"],
            f"{base}.target_rpc_backend_returned",
        )
        rpc_completed = _integer(raw["target_rpc_completed"], f"{base}.target_rpc_completed")
        route_logical_calls = _integer(
            raw["route_materialized_logical_target_calls"],
            f"{base}.route_materialized_logical_target_calls",
        )
        search_rollouts = _integer(
            raw["selection_search_rollout_count"],
            f"{base}.selection_search_rollout_count",
        )
        search_calls = _integer(
            raw["selection_search_target_calls"],
            f"{base}.selection_search_target_calls",
        )
        search_rpc_attempted = _integer(
            raw["selection_search_target_rpc_attempted"],
            f"{base}.selection_search_target_rpc_attempted",
        )
        search_rpc_backend_returned = _integer(
            raw["selection_search_target_rpc_backend_returned"],
            f"{base}.selection_search_target_rpc_backend_returned",
        )
        search_rpc_completed = _integer(
            raw["selection_search_target_rpc_completed"],
            f"{base}.selection_search_target_rpc_completed",
        )
        _require(
            budgeted == 1 + len(proposed) == 1 + budget,
            f"{base}.budgeted_target_calls",
            "must count chunk 0 plus exactly K proposed Target calls",
        )
        expected_committed = 1 + len(selected)
        _require(
            committed == expected_committed,
            f"{base}.committed_target_chunks",
            (
                "must count chunk 0 plus accepted boundary-gated commits"
                if boundary_gated
                else "must count chunk 0 plus exactly K corrections"
            ),
        )
        _require(
            physical >= 0,
            f"{base}.physical_target_calls",
            "must be a non-negative actual RPC count",
        )
        _require(
            0 <= rpc_completed <= rpc_backend_returned <= rpc_attempted,
            f"{base}.target_rpc_attempted",
            "lifecycle counts must satisfy 0 <= completed <= backend_returned <= attempted",
        )
        _require(
            physical == rpc_completed,
            f"{base}.physical_target_calls",
            "must equal target_rpc_completed",
        )
        _require(
            search_rollouts >= 0,
            f"{base}.selection_search_rollout_count",
            "must be a non-negative rollout count",
        )
        _require(
            search_calls >= 0,
            f"{base}.selection_search_target_calls",
            "must be a non-negative successful RPC count",
        )
        _require(
            0
            <= search_rpc_completed
            <= search_rpc_backend_returned
            <= search_rpc_attempted,
            f"{base}.selection_search_target_rpc_attempted",
            "search lifecycle counts must satisfy completed <= backend_returned <= attempted",
        )
        _require(
            search_calls == search_rpc_completed,
            f"{base}.selection_search_target_calls",
            "must equal selection_search_target_rpc_completed",
        )
        _require(
            search_rpc_attempted <= rpc_attempted,
            f"{base}.selection_search_target_rpc_attempted",
            "cannot exceed total target_rpc_attempted",
        )
        _require(
            search_rpc_backend_returned <= rpc_backend_returned,
            f"{base}.selection_search_target_rpc_backend_returned",
            "cannot exceed total target_rpc_backend_returned",
        )
        _require(
            search_rpc_completed <= rpc_completed,
            f"{base}.selection_search_target_rpc_completed",
            "cannot exceed total target_rpc_completed",
        )
        if strategy != "greedy_oracle":
            _require(
                search_rollouts == 0,
                f"{base}.selection_search_rollout_count",
                "must be zero outside Greedy Oracle",
            )
            _require(
                search_calls == 0,
                f"{base}.selection_search_target_calls",
                "must be zero outside Greedy Oracle",
            )
            _require(
                search_rpc_attempted == 0,
                f"{base}.selection_search_target_rpc_attempted",
                "must be zero outside Greedy Oracle",
            )
            _require(
                search_rpc_backend_returned == 0,
                f"{base}.selection_search_target_rpc_backend_returned",
                "must be zero outside Greedy Oracle",
            )
            _require(
                search_rpc_completed == 0,
                f"{base}.selection_search_target_rpc_completed",
                "must be zero outside Greedy Oracle",
            )
        _require(
            search_rollouts > 0 or search_rpc_attempted == 0,
            f"{base}.selection_search_target_rpc_attempted",
            "cannot be positive when no selection-search rollout was evaluated",
        )
        _require(
            search_calls <= physical,
            f"{base}.selection_search_target_calls",
            "cannot exceed the row's incremental successful physical Target calls",
        )
        route_calls = physical - search_calls
        route_rpc_attempted = rpc_attempted - search_rpc_attempted
        route_rpc_backend_returned = (
            rpc_backend_returned - search_rpc_backend_returned
        )
        _require(
            0 <= route_calls <= route_rpc_backend_returned <= route_rpc_attempted,
            f"{base}.target_rpc_attempted",
            "route lifecycle counts must satisfy completed <= backend_returned <= attempted",
        )
        _require(
            route_logical_calls <= route_calls,
            f"{base}.route_materialized_logical_target_calls",
            "cannot exceed completed physical route attempts",
        )
        _require(
            route_logical_calls <= budgeted,
            f"{base}.route_materialized_logical_target_calls",
            "unique route materializations exceed the scientific Target-call budget",
        )
        route_cache_hits = budgeted - route_logical_calls
        _require(
            route_cache_hits >= 0,
            f"{base}.physical_target_calls",
            "derived route cache hits must be non-negative",
        )
        final_hash = _sha256(raw["final_history_hash"], f"{base}.final_history_hash")
        key = (prompt_id, seed, budget, strategy, route_repeat)
        _require(key not in seen, base, "duplicate rollout key")
        seen.add(key)
        rows.append(
            Insight2BRow(
                prompt_id=prompt_id,
                seed=seed,
                budget=budget,
                strategy=strategy,
                route_repeat=route_repeat,
                selected_chunks=selected,
                proposed_target_chunks=proposed,
                full_quality=quality,
                temporal_consistency=temporal,
                budgeted_target_calls=budgeted,
                committed_target_chunks=committed,
                physical_target_calls=physical,
                target_rpc_attempted=rpc_attempted,
                target_rpc_backend_returned=rpc_backend_returned,
                target_rpc_completed=rpc_completed,
                route_materialized_logical_target_calls=route_logical_calls,
                selection_search_rollout_count=search_rollouts,
                selection_search_target_calls=search_calls,
                selection_search_target_rpc_attempted=search_rpc_attempted,
                selection_search_target_rpc_backend_returned=(
                    search_rpc_backend_returned
                ),
                selection_search_target_rpc_completed=search_rpc_completed,
                final_history_hash=final_hash,
            )
        )
    expected = set()
    for prompt_id in prompts:
        for seed in seeds:
            for budget in budgets:
                for strategy in strategies:
                    repeats: Iterable[int | None] = (
                        range(random_repeats) if strategy == "random" else (None,)
                    )
                    for repeat in repeats:
                        expected.add((prompt_id, seed, budget, strategy, repeat))
    _require(
        seen == expected,
        "$rows",
        f"incomplete 2(b) grid: missing={len(expected - seen)}, extra={len(seen - expected)}",
    )
    return rows


@dataclass(frozen=True)
class _CollapsedB:
    prompt_id: str
    seed: int
    budget: int
    strategy: str
    quality: float
    temporal: float
    route_variance_quality: float | None
    budgeted_target_calls: float
    committed_target_calls: float
    physical_target_calls: float
    target_rpc_attempted: float
    target_rpc_backend_returned: float
    target_rpc_completed: float
    route_materialized_logical_target_calls: float
    selection_search_rollout_count: float
    selection_search_target_calls: float
    selection_search_target_rpc_attempted: float
    selection_search_target_rpc_backend_returned: float
    selection_search_target_rpc_completed: float
    route_materialization_target_calls: float
    route_materialization_target_rpc_attempted: float
    route_cache_hits: float


def analyze_insight2b(
    rows: Sequence[Insight2BRow],
    *,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20261013,
    descriptive_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collapse Random repeats per video, then summarize and plot 2(b).

    ``descriptive_only`` is reserved for a one-prompt smoke run.  Such a run
    can exercise the complete generation, scoring, export, and plotting path,
    but it cannot support a prompt-cluster confidence interval.  The emitted
    CI fields are therefore null rather than a fabricated zero-width interval.
    """

    _require(bool(rows), "$rows", "no measurements")
    if descriptive_only:
        _require(
            len({row.prompt_id for row in rows}) == 1,
            "$descriptive_only",
            "is reserved for a single-prompt smoke run",
        )
    budgets = tuple(sorted({row.budget for row in rows}))
    present_strategies = {row.strategy for row in rows}
    strategy_order = (
        "random",
        "uniform",
        "online_rank_guided",
        "online_coverage_guided",
        "online_lookahead_coverage_guided",
        "online_lookahead_coverage_boundary_gated",
        "greedy_oracle",
        "online_erank_drop",
    )
    strategies = tuple(
        strategy for strategy in strategy_order if strategy in present_strategies
    )
    _require(
        set(strategies) == present_strategies,
        "$rows.strategy",
        "contains an unsupported strategy",
    )
    grouped: dict[tuple[str, int, int, str], list[Insight2BRow]] = {}
    for row in rows:
        grouped.setdefault((row.prompt_id, row.seed, row.budget, row.strategy), []).append(row)
    collapsed: list[_CollapsedB] = []
    for (prompt_id, seed, budget, strategy), sample in sorted(grouped.items()):
        qualities = [row.full_quality for row in sample]
        temporals = [row.temporal_consistency for row in sample]
        variance = statistics.variance(qualities) if strategy == "random" else None
        collapsed.append(
            _CollapsedB(
                prompt_id=prompt_id,
                seed=seed,
                budget=budget,
                strategy=strategy,
                quality=_mean(qualities),
                temporal=_mean(temporals),
                route_variance_quality=variance,
                budgeted_target_calls=_mean(
                    float(row.budgeted_target_calls) for row in sample
                ),
                committed_target_calls=_mean(float(row.committed_target_chunks) for row in sample),
                physical_target_calls=_mean(float(row.physical_target_calls) for row in sample),
                target_rpc_attempted=_mean(float(row.target_rpc_attempted) for row in sample),
                target_rpc_backend_returned=_mean(
                    float(row.target_rpc_backend_returned) for row in sample
                ),
                target_rpc_completed=_mean(float(row.target_rpc_completed) for row in sample),
                route_materialized_logical_target_calls=_mean(
                    float(row.route_materialized_logical_target_calls)
                    for row in sample
                ),
                selection_search_rollout_count=_mean(
                    float(row.selection_search_rollout_count) for row in sample
                ),
                selection_search_target_calls=_mean(
                    float(row.selection_search_target_calls) for row in sample
                ),
                selection_search_target_rpc_attempted=_mean(
                    float(row.selection_search_target_rpc_attempted) for row in sample
                ),
                selection_search_target_rpc_backend_returned=_mean(
                    float(row.selection_search_target_rpc_backend_returned)
                    for row in sample
                ),
                selection_search_target_rpc_completed=_mean(
                    float(row.selection_search_target_rpc_completed) for row in sample
                ),
                route_materialization_target_calls=_mean(
                    float(row.route_materialization_target_calls) for row in sample
                ),
                route_materialization_target_rpc_attempted=_mean(
                    float(row.route_materialization_target_rpc_attempted) for row in sample
                ),
                route_cache_hits=_mean(float(row.route_cache_hits) for row in sample),
            )
        )

    points: list[dict[str, Any]] = []
    for budget in budgets:
        for strategy_index, strategy in enumerate(strategies):
            sample = [row for row in collapsed if row.budget == budget and row.strategy == strategy]
            raw_sample = [row for row in rows if row.budget == budget and row.strategy == strategy]
            _require(bool(sample), f"$summary.{budget}.{strategy}", "empty group")
            _require(bool(raw_sample), f"$summary.{budget}.{strategy}", "empty raw group")
            if descriptive_only:
                quality_ci = None
                temporal_ci = None
            else:
                quality_ci = _cluster_bootstrap(
                    sample,
                    cluster=lambda row: row.prompt_id,
                    statistic=lambda chosen: _mean(row.quality for row in chosen),
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + budget * 100 + strategy_index,
                )
                temporal_ci = _cluster_bootstrap(
                    sample,
                    cluster=lambda row: row.prompt_id,
                    statistic=lambda chosen: _mean(row.temporal for row in chosen),
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + 1000 + budget * 100 + strategy_index,
                )
            random_variances = [
                row.route_variance_quality
                for row in sample
                if row.route_variance_quality is not None
            ]
            points.append(
                {
                    "budget": budget,
                    "budget_label": f"1+{budget}",
                    "strategy": strategy,
                    "n_videos": len(sample),
                    "n_raw_rollouts": len(raw_sample),
                    "mean_full_quality": _mean(row.quality for row in sample),
                    "quality_ci95": (
                        None if quality_ci is None else list(quality_ci)
                    ),
                    "mean_temporal_consistency": _mean(row.temporal for row in sample),
                    "temporal_ci95": (
                        None if temporal_ci is None else list(temporal_ci)
                    ),
                    "mean_budgeted_target_calls": _mean(
                        row.budgeted_target_calls for row in sample
                    ),
                    "mean_scientific_budgeted_target_calls": _mean(
                        row.budgeted_target_calls for row in sample
                    ),
                    "mean_committed_target_calls": _mean(
                        row.committed_target_calls for row in sample
                    ),
                    "mean_scientific_committed_target_calls": _mean(
                        row.committed_target_calls for row in sample
                    ),
                    "mean_physical_target_calls": _mean(
                        row.physical_target_calls for row in sample
                    ),
                    "mean_incremental_successful_physical_target_calls": _mean(
                        row.physical_target_calls for row in sample
                    ),
                    "mean_target_rpc_attempted": _mean(row.target_rpc_attempted for row in sample),
                    "mean_target_rpc_backend_returned": _mean(
                        row.target_rpc_backend_returned for row in sample
                    ),
                    "mean_target_rpc_completed": _mean(row.target_rpc_completed for row in sample),
                    "mean_route_materialized_logical_target_calls": _mean(
                        row.route_materialized_logical_target_calls for row in sample
                    ),
                    "mean_selection_search_rollout_count": _mean(
                        row.selection_search_rollout_count for row in sample
                    ),
                    "mean_selection_search_target_calls": _mean(
                        row.selection_search_target_calls for row in sample
                    ),
                    "mean_selection_search_target_rpc_attempted": _mean(
                        row.selection_search_target_rpc_attempted for row in sample
                    ),
                    "mean_selection_search_target_rpc_backend_returned": _mean(
                        row.selection_search_target_rpc_backend_returned for row in sample
                    ),
                    "mean_selection_search_target_rpc_completed": _mean(
                        row.selection_search_target_rpc_completed for row in sample
                    ),
                    "mean_route_materialization_target_calls": _mean(
                        row.route_materialization_target_calls for row in sample
                    ),
                    "mean_route_materialization_target_rpc_attempted": _mean(
                        row.route_materialization_target_rpc_attempted for row in sample
                    ),
                    "mean_route_cache_hits": _mean(row.route_cache_hits for row in sample),
                    "total_scientific_committed_target_calls": sum(
                        row.committed_target_chunks for row in raw_sample
                    ),
                    "total_scientific_budgeted_target_calls": sum(
                        row.budgeted_target_calls for row in raw_sample
                    ),
                    "total_incremental_successful_physical_target_calls": sum(
                        row.physical_target_calls for row in raw_sample
                    ),
                    "total_target_rpc_attempted": sum(
                        row.target_rpc_attempted for row in raw_sample
                    ),
                    "total_target_rpc_backend_returned": sum(
                        row.target_rpc_backend_returned for row in raw_sample
                    ),
                    "total_target_rpc_completed": sum(
                        row.target_rpc_completed for row in raw_sample
                    ),
                    "total_route_materialized_logical_target_calls": sum(
                        row.route_materialized_logical_target_calls
                        for row in raw_sample
                    ),
                    "total_selection_search_rollout_count": sum(
                        row.selection_search_rollout_count for row in raw_sample
                    ),
                    "total_selection_search_target_calls": sum(
                        row.selection_search_target_calls for row in raw_sample
                    ),
                    "total_selection_search_target_rpc_attempted": sum(
                        row.selection_search_target_rpc_attempted for row in raw_sample
                    ),
                    "total_selection_search_target_rpc_backend_returned": sum(
                        row.selection_search_target_rpc_backend_returned
                        for row in raw_sample
                    ),
                    "total_selection_search_target_rpc_completed": sum(
                        row.selection_search_target_rpc_completed for row in raw_sample
                    ),
                    "total_route_materialization_target_calls": sum(
                        row.route_materialization_target_calls for row in raw_sample
                    ),
                    "total_route_materialization_target_rpc_attempted": sum(
                        row.route_materialization_target_rpc_attempted for row in raw_sample
                    ),
                    "total_route_cache_hits": sum(row.route_cache_hits for row in raw_sample),
                    "random_route_variance_mean": (
                        _mean(float(value) for value in random_variances)
                        if random_variances
                        else None
                    ),
                }
            )

    lookup = {(row.prompt_id, row.seed, row.budget, row.strategy): row for row in collapsed}
    paired: list[dict[str, Any]] = []
    for budget in budgets:
        identities = sorted(
            {(row.prompt_id, row.seed) for row in collapsed if row.budget == budget}
        )
        for strategy in strategies:
            if strategy == "random":
                continue
            references = [
                item for item in ("random", "uniform") if item in strategies
            ]
            if (
                strategy == "online_coverage_guided"
                and "online_rank_guided" in strategies
            ):
                references.append("online_rank_guided")
            if (
                strategy == "online_lookahead_coverage_guided"
                and "online_coverage_guided" in strategies
            ):
                references.append("online_coverage_guided")
            if (
                strategy == "online_lookahead_coverage_boundary_gated"
                and "online_lookahead_coverage_guided" in strategies
            ):
                references.append("online_lookahead_coverage_guided")
            for reference in references:
                if strategy == reference:
                    continue
                diffs = [
                    {
                        "prompt_id": prompt_id,
                        "delta": lookup[(prompt_id, seed, budget, strategy)].quality
                        - lookup[(prompt_id, seed, budget, reference)].quality,
                    }
                    for prompt_id, seed in identities
                ]
                ci = (
                    None
                    if descriptive_only
                    else _cluster_bootstrap(
                        diffs,
                        cluster=lambda row: str(row["prompt_id"]),
                        statistic=lambda chosen: _mean(
                            float(row["delta"]) for row in chosen
                        ),
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed + 5000 + budget * 100 + len(paired),
                    )
                )
                paired.append(
                    {
                        "budget": budget,
                        "strategy": strategy,
                        "reference": reference,
                        "mean_paired_quality_difference": _mean(
                            float(row["delta"]) for row in diffs
                        ),
                        "ci95": None if ci is None else list(ci),
                    }
                )

    summary = {
        "schema": INSIGHT2B_ANALYSIS_SCHEMA,
        "n_raw_rollouts": len(rows),
        "n_collapsed_video_strategy_budget_rows": len(collapsed),
        "bootstrap": (
            {
                "status": "not_computed",
                "reason": "single_prompt_smoke_has_fewer_than_two_prompt_clusters",
                "cluster": "prompt_id",
                "replicates": 0,
            }
            if descriptive_only
            else {
                "cluster": "prompt_id",
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "ci": "two_sided_percentile_95",
                "quantile": "Hyndman-Fan type 7",
            }
        ),
        "random_repeat_handling": "collapsed_within_prompt_seed_budget_before_bootstrap",
        "budgets": list(budgets),
        "strategies": list(strategies),
        "target_call_accounting": {
            "scientific_budgeted": (
                "budgeted_target_calls; exactly 1+K proposed Target calls for every route"
            ),
            "scientific_committed": (
                "committed_target_chunks; exactly 1+K for legacy exact-K routes, "
                "and 1 plus accepted proposed_target_chunks for boundary-gated routes"
            ),
            "incremental_successful_physical": (
                "physical_target_calls == target_rpc_completed; fully completed RPCs newly "
                "attributed to the row's persistent journal scope and therefore "
                "execution-order/cache dependent"
            ),
            "rpc_lifecycle": (
                "target_rpc_completed <= target_rpc_backend_returned <= "
                "target_rpc_attempted; backend_returned records completed GPU inference "
                "even when output publication did not complete"
            ),
            "greedy_selection_search": (
                "not present"
                if "greedy_oracle" not in strategies
                else (
                    "selection_search_rollout_count, selection_search_target_rpc_attempted, "
                    "selection_search_target_rpc_backend_returned, and "
                    "selection_search_target_rpc_completed == "
                    "selection_search_target_calls; reported separately and excluded from "
                    "the scientific 1+K budget"
                )
            ),
            "route_materialization": (
                "physical attempts: target_rpc_completed - "
                "selection_search_target_rpc_completed; may exceed 1+K after retries"
            ),
            "route_materialized_logical": (
                "distinct completed logical_call_id values in route_materialization; "
                "used for cache-hit accounting"
            ),
            "route_materialization_attempted": (
                "target_rpc_attempted - selection_search_target_rpc_attempted"
            ),
            "route_cache_hits": (
                "budgeted_target_calls - route_materialized_logical_target_calls"
            ),
        },
        "points": points,
        "paired_quality_differences": paired,
    }
    figure_data = {
        "schema": FIGURE2B_DATA_SCHEMA,
        "x": "additional_target_budget",
        "y": "full_video_quality",
        "budgets": list(budgets),
        "strategies": list(strategies),
        "oracle_label": "Greedy Oracle" if "greedy_oracle" in strategies else None,
        "monotonicity_constraint": False,
        "points": points,
    }
    if descriptive_only:
        figure_data["confidence_intervals"] = False
    return summary, figure_data


def render_figure2a(plot_data: Mapping[str, Any], output: str | Path) -> None:
    """Render a paper-ready scatter, least-squares trend, and binned means."""

    _require(
        plot_data.get("schema") == FIGURE2A_DATA_SCHEMA, "$.schema", "invalid Figure 2(a) data"
    )
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise AnalysisError("Matplotlib and NumPy are required to render Figure 2(a)") from error
    points = plot_data["points"]
    _require(bool(points), "$.points", "empty")
    x = np.asarray([float(item["risk_score"]) for item in points])
    y = np.asarray([float(item["delta_Q"]) for item in points])
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    axis.scatter(x, y, s=15, alpha=0.24, color="#3B82F6", linewidths=0, rasterized=True)
    if len(set(x.tolist())) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(0.0, 1.0, 200)
        axis.plot(
            grid, intercept + slope * grid, color="#1D4ED8", linewidth=2, label="Linear trend"
        )
    bins = plot_data["risk_bins"]
    bx = np.asarray([float(item["mean_risk"]) for item in bins])
    by = np.asarray([float(item["mean_delta_Q"]) for item in bins])
    lower = by - np.asarray([float(item["ci95"][0]) for item in bins])
    upper = np.asarray([float(item["ci95"][1]) for item in bins]) - by
    axis.errorbar(
        bx,
        by,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        color="#F97316",
        capsize=3,
        label="Risk quartile mean ± 95% CI",
    )
    rho = float(plot_data["spearman"]["estimate"])
    axis.text(0.02, 0.98, f"Spearman ρ = {rho:.3f}", transform=axis.transAxes, ha="left", va="top")
    axis.axhline(0.0, color="#9CA3AF", linewidth=0.8, zorder=0)
    axis.set_xlim(-0.02, 1.02)
    axis.set_xlabel("Representation Degradation Risk")
    axis.set_ylabel("Correction Gain ΔQ")
    axis.set_title("(a) Single-point Correction Gain (1+1)")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=300)
    plt.close(figure)


def render_figure2b(plot_data: Mapping[str, Any], output: str | Path) -> None:
    """Render exactly the measured strategy/budget groups without fitting."""

    _require(
        plot_data.get("schema") == FIGURE2B_DATA_SCHEMA, "$.schema", "invalid Figure 2(b) data"
    )
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise AnalysisError("Matplotlib and NumPy are required to render Figure 2(b)") from error
    styles = {
        "random": ("Random", "#3B82F6", "o"),
        "uniform": ("Uniform", "#22C55E", "o"),
        "online_rank_guided": ("Rank-guided", "#F97316", "o"),
        "online_coverage_guided": ("Coverage-guided", "#A855F7", "o"),
        "online_lookahead_coverage_guided": (
            "Lookahead coverage-guided",
            "#DC2626",
            "o",
        ),
        "online_lookahead_coverage_boundary_gated": (
            "Boundary-gated lookahead",
            "#0891B2",
            "o",
        ),
        "online_erank_drop": ("Online eRank-drop", "#F97316", "o"),
        "greedy_oracle": ("Greedy Oracle", "#EAB308", "o"),
    }
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    points = plot_data["points"]
    budgets = list(plot_data.get("budgets") or sorted({item["budget"] for item in points}))
    strategies = list(
        plot_data.get("strategies")
        or [item for item in styles if any(point["strategy"] == item for point in points)]
    )
    for strategy in strategies:
        _require(strategy in styles, f"$.strategies.{strategy}", "unknown strategy")
        chosen = sorted(
            (item for item in points if item["strategy"] == strategy),
            key=lambda item: item["budget"],
        )
        _require(
            [item["budget"] for item in chosen] == budgets,
            f"$.points.{strategy}",
            "missing budget",
        )
        x = np.asarray([item["budget"] for item in chosen], dtype=float)
        y = np.asarray([item["mean_full_quality"] for item in chosen], dtype=float)
        ci_values = [item.get("quality_ci95") for item in chosen]
        _require(
            all(value is None for value in ci_values)
            or all(value is not None for value in ci_values),
            f"$.points.{strategy}.quality_ci95",
            "must be present for every budget or omitted for every budget",
        )
        yerr = None
        if all(value is not None for value in ci_values):
            lower = y - np.asarray(
                [value[0] for value in ci_values if value is not None], dtype=float
            )
            upper = np.asarray(
                [value[1] for value in ci_values if value is not None], dtype=float
            ) - y
            yerr = np.vstack([lower, upper])
        label, color, marker = styles[strategy]
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )
    axis.set_xticks(budgets, [f"1+{budget}" for budget in budgets])
    axis.set_xlabel("Correction Budget")
    axis.set_ylabel("Full-video Quality")
    axis.set_title("(b) Multi-point Correction Budget")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="best")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=300)
    plt.close(figure)


def render_figure2_combined(
    plot_data_a: Mapping[str, Any],
    plot_data_b: Mapping[str, Any],
    output: str | Path,
) -> None:
    """Render the measured two-panel Figure 2 using the supplied layout only.

    The reference image fixes the panel arrangement and visual vocabulary; it
    never supplies coordinates, axis limits, or monotonicity constraints.
    """

    _require(
        plot_data_a.get("schema") == FIGURE2A_DATA_SCHEMA,
        "$.figure2a.schema",
        "invalid Figure 2(a) data",
    )
    _require(
        plot_data_b.get("schema") == FIGURE2B_DATA_SCHEMA,
        "$.figure2b.schema",
        "invalid Figure 2(b) data",
    )
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise AnalysisError("Matplotlib and NumPy are required to render Figure 2") from error

    figure, (axis_a, axis_b) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.2),
        constrained_layout=True,
    )
    points_a = plot_data_a["points"]
    _require(bool(points_a), "$.figure2a.points", "empty")
    x = np.asarray([float(item["risk_score"]) for item in points_a])
    y = np.asarray([float(item["delta_Q"]) for item in points_a])
    axis_a.scatter(
        x,
        y,
        s=14,
        alpha=0.22,
        color="#3B82F6",
        linewidths=0,
        rasterized=True,
    )
    if len(set(x.tolist())) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(0.0, 1.0, 200)
        axis_a.plot(grid, intercept + slope * grid, color="#1D4ED8", linewidth=2)
    bins = plot_data_a["risk_bins"]
    bin_x = np.asarray([float(item["mean_risk"]) for item in bins])
    bin_y = np.asarray([float(item["mean_delta_Q"]) for item in bins])
    bin_lower = bin_y - np.asarray([float(item["ci95"][0]) for item in bins])
    bin_upper = np.asarray([float(item["ci95"][1]) for item in bins]) - bin_y
    axis_a.errorbar(
        bin_x,
        bin_y,
        yerr=np.vstack([bin_lower, bin_upper]),
        fmt="o",
        color="#F97316",
        capsize=3,
        label="Risk-bin mean ± 95% CI",
    )
    rho = float(plot_data_a["spearman"]["estimate"])
    axis_a.text(
        0.02,
        0.98,
        f"Spearman ρ = {rho:.3f}",
        transform=axis_a.transAxes,
        ha="left",
        va="top",
    )
    axis_a.axhline(0.0, color="#9CA3AF", linewidth=0.8, zorder=0)
    axis_a.set_xlim(-0.02, 1.02)
    axis_a.set_xlabel("Representation Degradation Risk")
    axis_a.set_ylabel("Correction Gain ΔQ")
    axis_a.set_title("(a) Single-point Correction Gain (1+1)")
    axis_a.grid(alpha=0.18, linewidth=0.6)
    axis_a.legend(frameon=False, loc="best")

    styles = {
        "random": ("Random", "#3B82F6"),
        "uniform": ("Uniform", "#22C55E"),
        "online_rank_guided": ("Rank-guided", "#F97316"),
        "online_coverage_guided": ("Coverage-guided", "#A855F7"),
        "online_lookahead_coverage_guided": (
            "Lookahead coverage-guided",
            "#DC2626",
        ),
        "online_lookahead_coverage_boundary_gated": (
            "Boundary-gated lookahead",
            "#0891B2",
        ),
        "online_erank_drop": ("Online eRank-drop", "#F97316"),
        "greedy_oracle": ("Greedy Oracle", "#EAB308"),
    }
    points_b = plot_data_b["points"]
    budgets_b = list(
        plot_data_b.get("budgets")
        or sorted({item["budget"] for item in points_b})
    )
    strategies_b = list(
        plot_data_b.get("strategies")
        or [item for item in styles if any(point["strategy"] == item for point in points_b)]
    )
    for strategy in strategies_b:
        _require(strategy in styles, f"$.figure2b.strategies.{strategy}", "unknown strategy")
        chosen = sorted(
            (item for item in points_b if item["strategy"] == strategy),
            key=lambda item: item["budget"],
        )
        _require(
            [item["budget"] for item in chosen] == budgets_b,
            f"$.figure2b.points.{strategy}",
            "missing budget",
        )
        budget_x = np.asarray([item["budget"] for item in chosen], dtype=float)
        quality_y = np.asarray([item["mean_full_quality"] for item in chosen], dtype=float)
        ci_values = [item.get("quality_ci95") for item in chosen]
        _require(
            all(value is None for value in ci_values)
            or all(value is not None for value in ci_values),
            f"$.figure2b.points.{strategy}.quality_ci95",
            "must be present for every budget or omitted for every budget",
        )
        quality_yerr = None
        if all(value is not None for value in ci_values):
            quality_lower = quality_y - np.asarray(
                [value[0] for value in ci_values if value is not None], dtype=float
            )
            quality_upper = np.asarray(
                [value[1] for value in ci_values if value is not None], dtype=float
            ) - quality_y
            quality_yerr = np.vstack([quality_lower, quality_upper])
        label, color = styles[strategy]
        axis_b.errorbar(
            budget_x,
            quality_y,
            yerr=quality_yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )
    axis_b.set_xticks(budgets_b, [f"1+{budget}" for budget in budgets_b])
    axis_b.set_xlabel("Correction Budget")
    axis_b.set_ylabel("Full-video Quality")
    axis_b.set_title("(b) Multi-point Correction Budget")
    axis_b.grid(alpha=0.18, linewidth=0.6)
    axis_b.legend(frameon=False, ncol=2, loc="best")
    figure.suptitle(
        "Figure 2. Representation Degradation and Correction Benefit",
        fontsize=14,
        x=0.01,
        ha="left",
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=300)
    plt.close(figure)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def write_top1_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "prompt_id",
        "seed",
        "rank_top1_chunk",
        "rank_top1_gain",
        "random_top1_expected_gain",
        "oracle_top1_chunk",
        "oracle_top1_gain",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
