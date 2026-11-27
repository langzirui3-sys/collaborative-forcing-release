"""Validation and statistics for the frozen eRank--quality protocol.

The implementation intentionally uses only the Python standard library.  A
confirmatory run is therefore analysable in the same locked environment as the
rest of :mod:`cfexp`, without silently pulling a different SciPy build onto a
GPU worker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = "erank_quality_correlation_protocol_v1"
REQUIRED_COLUMNS = {
    "prompt_id",
    "seed",
    "model_id",
    "trajectory_seconds",
    "window_index",
    "window_start_s",
    "window_end_s",
    "erank_primary",
    "erank_centered_singular_secondary",
    "erank_legacy_centered_squared_normalized",
    "erank_krea_l8_prefix2_secondary",
    "imaging_quality",
}


class ProtocolError(ValueError):
    """Raised when the protocol or collected metric table violates its contract."""


@dataclass(frozen=True)
class WindowRow:
    """One non-overlapping quality window from one generated trajectory."""

    prompt_id: str
    seed: int
    model_id: str
    trajectory_seconds: int
    window_index: int
    window_start_s: float
    window_end_s: float
    erank_primary: float | None
    erank_centered_singular_secondary: float | None
    erank_legacy_centered_squared_normalized: float | None
    erank_krea_l8_prefix2_secondary: float | None
    imaging_quality: float


@dataclass(frozen=True)
class BlockERank:
    """Absolute eRank and exact half-open RGB-frame support of one latent block."""

    block_index: int
    frame_start: int
    frame_end: int
    value: float


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ProtocolError(f"{path}: {message}")


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate an eRank quality protocol JSON file."""

    protocol_path = Path(path)
    try:
        value = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load protocol {protocol_path}: {exc}") from exc
    validate_protocol(value)
    return value


def validate_protocol(value: Any) -> None:
    """Validate the frozen v1 protocol, including the pre-registered method."""

    _require(isinstance(value, dict), "$", "expected an object")
    _require(value.get("schema") == PROTOCOL_SCHEMA, "$.schema", f"expected {PROTOCOL_SCHEMA!r}")
    _require(value.get("status") == "frozen", "$.status", "must be 'frozen'")

    provenance = value.get("provenance")
    _require(isinstance(provenance, dict), "$.provenance", "expected an object")
    _require(
        provenance.get("source_prompt_manifest_sha256")
        == "8a2f89312402c2688cb00074194d748e264dd47e5a2f2e6fe010ca2afb0d9d94",
        "$.provenance.source_prompt_manifest_sha256",
        "source prompt population drift",
    )

    design = value.get("design")
    _require(isinstance(design, dict), "$.design", "expected an object")
    _require(design.get("prompt_count") == 48, "$.design.prompt_count", "must equal 48")
    _require(
        design.get("seeds") == [20260911, 20260912, 20260913],
        "$.design.seeds",
        "must equal the three pre-registered seeds",
    )
    _require(
        design.get("trajectory_seconds") == 30,
        "$.design.trajectory_seconds",
        "must equal 30",
    )
    _require(
        design.get("derived_durations_seconds") == [15, 30],
        "$.design.derived_durations_seconds",
        "must be [15, 30]",
    )
    _require(
        design.get("window_seconds") == 5 and design.get("window_stride_seconds") == 5,
        "$.design",
        "windows must be non-overlapping 5-second windows",
    )
    _require(
        design.get("fps") == 24 and design.get("trajectory_frames") == 720,
        "$.design",
        "the frozen 30-second trajectory must contain 720 frames at 24 fps",
    )
    _require(
        design.get("trajectory_reuse") == "one_30s_trajectory_with_15s_prefix",
        "$.design.trajectory_reuse",
        "must derive 15 s from the same 30 s trajectory",
    )

    models = value.get("models")
    _require(isinstance(models, list), "$.models", "expected a list")
    model_ids = [item.get("id") for item in models if isinstance(item, dict)]
    _require(
        model_ids == ["self_forcing_1.3b", "krea_realtime_14b"],
        "$.models",
        "must contain the frozen Self-Forcing 1.3B and Krea 14B arms in order",
    )

    prompts = value.get("prompts")
    _require(isinstance(prompts, list), "$.prompts", "expected a list")
    _require(len(prompts) == 48, "$.prompts", "must contain exactly 48 prompts")
    prompt_ids: list[str] = []
    dimensions: Counter[str] = Counter()
    ordinals: list[int] = []
    for index, prompt in enumerate(prompts):
        path = f"$.prompts[{index}]"
        _require(isinstance(prompt, dict), path, "expected an object")
        for field in (
            "prompt_id",
            "prompt",
            "assigned_dimension",
            "selection_ordinal",
            "benchmark_prompt_index",
            "prompt_file_sha256",
            "selection_sha256",
        ):
            _require(field in prompt, path, f"missing {field!r}")
        prompt_id = prompt["prompt_id"]
        _require(isinstance(prompt_id, str) and prompt_id, f"{path}.prompt_id", "invalid")
        _require(
            isinstance(prompt["prompt"], str) and prompt["prompt"],
            f"{path}.prompt",
            "must be non-empty",
        )
        _require(
            hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest()
            == prompt["prompt_file_sha256"],
            f"{path}.prompt_file_sha256",
            "does not match the frozen prompt text",
        )
        for digest_field in ("prompt_file_sha256", "selection_sha256"):
            digest = prompt[digest_field]
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"{path}.{digest_field}",
                "must be a lowercase 64-hex SHA-256",
            )
        prompt_ids.append(prompt_id)
        dimensions[str(prompt["assigned_dimension"])] += 1
        ordinals.append(int(prompt["selection_ordinal"]))
    _require(len(set(prompt_ids)) == 48, "$.prompts", "prompt_id values must be unique")
    _require(
        len(dimensions) == 16 and set(dimensions.values()) == {3},
        "$.prompts",
        "must contain the first three frozen cases for each of 16 dimensions",
    )
    expected_ordinals = [
        ordinal
        for group in range(16)
        for ordinal in (8 * group + 1, 8 * group + 2, 8 * group + 3)
    ]
    _require(
        ordinals == expected_ordinals,
        "$.prompts",
        "must contain the first three source ordinals for every dimension",
    )

    metric = value.get("primary_erank")
    _require(isinstance(metric, dict), "$.primary_erank", "expected an object")
    frozen_metric = {
        "population_model_id": "self_forcing_1.3b",
        "source_model": "Self-Forcing-1.3B",
        "layer_number_one_based": 15,
        "layer_index_zero_based": 14,
        "measurement_context": "online_denoising_step_0",
        "current_blocks": 1,
        "latent_frames_per_block": 3,
        "online_denoising_step_index": 0,
        "actual_timestep": 1000,
        "noise_source": "native_absolute_chunk_noise_derived_from_seed_and_chunk",
        "extra_probe_forward": False,
        "max_sampled_tokens": 256,
        "token_sampling": "deterministic_even_floor_i_times_T_over_N",
        "token_sampling_exact": "for N=min(256,T), indices_i=floor(i*T/N), i=0,...,N-1",
        "centering": "none",
        "gram": "H_sampled@H_sampled.T",
        "spectrum": "singular_values_from_sqrt_of_clamped_gram_eigenvalues",
        "gram_eigenvalue_relative_cutoff": 1e-7,
        "gram_eigenvalue_cutoff_rule": (
            "set lambda_i to zero when lambda_i < 1e-7 * max_j(lambda_j), "
            "before square root"
        ),
        "field": "sampled_paper_effective_rank",
        "window_reducer": "rgb_frame_overlap_weighted_mean",
        "block_to_window_assignment": (
            "weight each block by the count of its generated RGB frames overlapping "
            "[5w,5(w+1)); trim support beyond frame 720"
        ),
        "decoder_frame_support": (
            "block 0 emits frames [0,6); every later block emits the next 12 frames"
        ),
        "uses_absolute_level": True,
    }
    for field, expected in frozen_metric.items():
        _require(
            metric.get(field) == expected,
            f"$.primary_erank.{field}",
            f"must equal {expected!r}",
        )

    secondary = value.get("secondary_eranks")
    _require(isinstance(secondary, list), "$.secondary_eranks", "expected a list")
    _require(
        [item.get("name") for item in secondary if isinstance(item, dict)]
        == [
            "sf_l15_centered_singular",
            "sf_l15_legacy_centered_squared_normalized",
            "krea_l8_clean_t0_prefix2",
        ],
        "$.secondary_eranks",
        "secondary metrics must be explicit and remain in frozen order",
    )
    _require(
        all(item.get("confirmatory") is False for item in secondary),
        "$.secondary_eranks",
        "secondary metrics cannot be confirmatory",
    )
    secondary_contracts = [
        {
            "centering": "subtract_sampled_token_mean_per_channel",
            "spectrum": "singular_values_from_sqrt_of_clamped_gram_eigenvalues",
            "normalization": "none",
        },
        {
            "centering": "subtract_sampled_token_mean_per_channel",
            "spectrum": "Gram eigenvalues, equivalent to squared singular values",
            "normalization": (
                "divide entropy effective rank by min(sample_count-1, channel_count)"
            ),
            "field": "sampled_normalized_entropy_effective_rank",
        },
        {
            "source_model": "Krea-Realtime-14B",
            "layer_number_one_based": 8,
            "layer_index_zero_based": 7,
            "latent_condition": "clean",
            "actual_timestep": 0,
            "candidate_prefix_blocks": 2,
        },
    ]
    for index, contract in enumerate(secondary_contracts):
        for field, expected in contract.items():
            _require(
                secondary[index].get(field) == expected,
                f"$.secondary_eranks[{index}].{field}",
                f"must equal {expected!r}",
            )

    quality = value.get("primary_quality")
    _require(isinstance(quality, dict), "$.primary_quality", "expected an object")
    _require(
        quality.get("benchmark") == "VBench-Long"
        and quality.get("dimension") == "imaging_quality"
        and quality.get("window_seconds") == 5,
        "$.primary_quality",
        "must be 5-second VBench-Long imaging_quality",
    )

    analysis = value.get("analysis")
    _require(isinstance(analysis, dict), "$.analysis", "expected an object")
    _require(analysis.get("primary_statistic") == "spearman", "$.analysis", "invalid statistic")
    _require(analysis.get("cluster") == "prompt_id", "$.analysis.cluster", "must be prompt_id")
    _require(analysis.get("null_hypothesis") == "rho<=0.6", "$.analysis", "invalid null")
    _require(analysis.get("alternative") == "rho>0.6", "$.analysis", "invalid alternative")
    _require(analysis.get("alpha") == 0.05, "$.analysis.alpha", "must equal 0.05")
    _require(
        analysis.get("two_way_fe_low_threshold") == 0.3,
        "$.analysis.two_way_fe_low_threshold",
        "must equal 0.3",
    )
    _require(
        isinstance(analysis.get("bootstrap_replicates"), int)
        and analysis["bootstrap_replicates"] >= 1000,
        "$.analysis.bootstrap_replicates",
        "must be at least 1000",
    )
    _require(
        isinstance(analysis.get("bootstrap_seed"), int),
        "$.analysis.bootstrap_seed",
        "invalid",
    )


def _finite_float(value: str, path: str, *, allow_blank: bool = False) -> float | None:
    if allow_blank and not value.strip():
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{path}: expected a finite number") from exc
    _require(math.isfinite(result), path, "expected a finite number")
    return result


def _integer(value: str, path: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{path}: expected an integer") from exc
    return result


def load_window_rows(
    path: str | Path,
    protocol: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> list[WindowRow]:
    """Load a window-level CSV and enforce its frozen experimental grid."""

    csv_path = Path(path)
    try:
        handle = csv_path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"cannot open metrics CSV {csv_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        _require(not missing, "$csv.header", f"missing column(s): {', '.join(missing)}")
        records = list(reader)

    prompts = {item["prompt_id"] for item in protocol["prompts"]}
    seeds = set(protocol["design"]["seeds"])
    models = {item["id"] for item in protocol["models"]}
    duration = int(protocol["design"]["trajectory_seconds"])
    window = float(protocol["design"]["window_seconds"])
    expected_windows = int(duration / window)
    primary_model = str(protocol["primary_erank"]["population_model_id"])

    rows: list[WindowRow] = []
    seen: set[tuple[str, int, str, int]] = set()
    for line_number, raw in enumerate(records, start=2):
        path_prefix = f"$csv.line[{line_number}]"
        prompt_id = raw["prompt_id"].strip()
        model_id = raw["model_id"].strip()
        seed = _integer(raw["seed"], f"{path_prefix}.seed")
        trajectory_seconds = _integer(
            raw["trajectory_seconds"], f"{path_prefix}.trajectory_seconds"
        )
        window_index = _integer(raw["window_index"], f"{path_prefix}.window_index")
        window_start = _finite_float(raw["window_start_s"], f"{path_prefix}.window_start_s")
        window_end = _finite_float(raw["window_end_s"], f"{path_prefix}.window_end_s")
        erank_primary = _finite_float(
            raw["erank_primary"], f"{path_prefix}.erank_primary", allow_blank=True
        )
        centered_singular = _finite_float(
            raw["erank_centered_singular_secondary"],
            f"{path_prefix}.erank_centered_singular_secondary",
            allow_blank=True,
        )
        legacy_normalized = _finite_float(
            raw["erank_legacy_centered_squared_normalized"],
            f"{path_prefix}.erank_legacy_centered_squared_normalized",
            allow_blank=True,
        )
        krea_secondary = _finite_float(
            raw["erank_krea_l8_prefix2_secondary"],
            f"{path_prefix}.erank_krea_l8_prefix2_secondary",
            allow_blank=True,
        )
        quality = _finite_float(raw["imaging_quality"], f"{path_prefix}.imaging_quality")
        assert window_start is not None and window_end is not None and quality is not None

        _require(prompt_id in prompts, f"{path_prefix}.prompt_id", "not in frozen prompt set")
        _require(seed in seeds, f"{path_prefix}.seed", "not in frozen seed set")
        _require(model_id in models, f"{path_prefix}.model_id", "not in frozen model set")
        _require(
            trajectory_seconds == duration,
            f"{path_prefix}.trajectory_seconds",
            f"must equal {duration}",
        )
        _require(
            0 <= window_index < expected_windows,
            f"{path_prefix}.window_index",
            f"must be in [0, {expected_windows - 1}]",
        )
        expected_start = window_index * window
        _require(
            math.isclose(window_start, expected_start, abs_tol=1e-9)
            and math.isclose(window_end, expected_start + window, abs_tol=1e-9),
            path_prefix,
            "window bounds do not match the non-overlapping 5-second grid",
        )
        _require(0.0 <= quality <= 1.0, f"{path_prefix}.imaging_quality", "must be in [0, 1]")
        if model_id == primary_model:
            _require(erank_primary is not None, f"{path_prefix}.erank_primary", "is required")
            assert erank_primary is not None
            _require(
                1.0 <= erank_primary <= 256.0,
                f"{path_prefix}.erank_primary",
                "paper-faithful absolute eRank must be in [1, 256]",
            )
        else:
            _require(
                erank_primary is None,
                f"{path_prefix}.erank_primary",
                "must be blank outside the pre-registered Self-Forcing population",
            )
        for field_name, rank in (
            ("erank_centered_singular_secondary", centered_singular),
            ("erank_krea_l8_prefix2_secondary", krea_secondary),
        ):
            if rank is not None:
                _require(
                    1.0 <= rank <= 256.0,
                    f"{path_prefix}.{field_name}",
                    "absolute effective rank must be in [1, 256]",
                )
        if legacy_normalized is not None:
            _require(
                0.0 <= legacy_normalized <= 1.0,
                f"{path_prefix}.erank_legacy_centered_squared_normalized",
                "must be in [0, 1]",
            )

        key = (prompt_id, seed, model_id, window_index)
        _require(key not in seen, path_prefix, f"duplicate experimental cell {key!r}")
        seen.add(key)
        rows.append(
            WindowRow(
                prompt_id=prompt_id,
                seed=seed,
                model_id=model_id,
                trajectory_seconds=trajectory_seconds,
                window_index=window_index,
                window_start_s=window_start,
                window_end_s=window_end,
                erank_primary=erank_primary,
                erank_centered_singular_secondary=centered_singular,
                erank_legacy_centered_squared_normalized=legacy_normalized,
                erank_krea_l8_prefix2_secondary=krea_secondary,
                imaging_quality=quality,
            )
        )

    if require_complete:
        expected = {
            (prompt_id, seed, model_id, window_index)
            for prompt_id in prompts
            for seed in seeds
            for model_id in models
            for window_index in range(expected_windows)
        }
        missing_cells = expected - seen
        extra_cells = seen - expected
        _require(
            not missing_cells and not extra_cells,
            "$csv",
            f"incomplete grid: missing={len(missing_cells)}, extra={len(extra_cells)}",
        )
    _require(rows, "$csv", "contains no data rows")
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ProtocolError("cannot take mean of an empty sequence")
    return math.fsum(values) / len(values)


def aggregate_block_eranks_to_windows(
    blocks: Sequence[BlockERank],
    *,
    fps: int = 24,
    trajectory_seconds: int = 30,
    window_seconds: int = 5,
    first_block_frames: int = 6,
    later_block_frames: int = 12,
) -> list[float]:
    """Aggregate block eRanks by exact RGB-frame overlap with each quality window.

    The production decoder emits six RGB frames for block zero and twelve for
    every later block.  A later block can straddle a five-second boundary; its
    eRank then contributes to both adjacent windows in proportion to the number
    of output frames in each.  Frames beyond the requested trajectory endpoint
    are trimmed rather than assigned to the final window.
    """

    _require(fps > 0, "$blocks.fps", "must be positive")
    _require(trajectory_seconds > 0, "$blocks.trajectory_seconds", "must be positive")
    _require(window_seconds > 0, "$blocks.window_seconds", "must be positive")
    total_frames = fps * trajectory_seconds
    window_frames = fps * window_seconds
    _require(
        total_frames % window_frames == 0,
        "$blocks",
        "trajectory must divide exactly into quality windows",
    )
    _require(blocks, "$blocks", "must not be empty")

    prior_end = 0
    for index, block in enumerate(blocks):
        path = f"$blocks[{index}]"
        _require(block.block_index == index, f"{path}.block_index", "must be contiguous from zero")
        expected_width = first_block_frames if index == 0 else later_block_frames
        _require(
            block.frame_start == prior_end,
            f"{path}.frame_start",
            "block supports must be contiguous",
        )
        _require(
            block.frame_end - block.frame_start == expected_width,
            path,
            f"expected exactly {expected_width} RGB frames",
        )
        _require(math.isfinite(block.value) and block.value > 0.0, f"{path}.value", "invalid eRank")
        prior_end = block.frame_end
    _require(prior_end >= total_frames, "$blocks", "blocks do not cover the full trajectory")
    _require(
        len(blocks) == 1 + math.ceil((total_frames - first_block_frames) / later_block_frames),
        "$blocks",
        "unexpected number of blocks for the requested trajectory",
    )

    results: list[float] = []
    for window_index in range(total_frames // window_frames):
        window_start = window_index * window_frames
        window_end = window_start + window_frames
        weighted_sum = 0.0
        covered = 0
        for block in blocks:
            overlap = max(
                0,
                min(block.frame_end, window_end, total_frames)
                - max(block.frame_start, window_start),
            )
            if overlap:
                weighted_sum += block.value * overlap
                covered += overlap
        _require(
            covered == window_frames,
            f"$blocks.window[{window_index}]",
            f"expected {window_frames} covered frames, got {covered}",
        )
        results.append(weighted_sum / covered)
    return results


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the product-moment correlation, rejecting degenerate inputs."""

    if len(left) != len(right) or len(left) < 3:
        raise ProtocolError("correlation requires equal vectors with at least three values")
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = math.fsum(a * b for a, b in zip(left_centered, right_centered))
    left_ss = math.fsum(value * value for value in left_centered)
    right_ss = math.fsum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        raise ProtocolError("correlation is undefined for a constant vector")
    return numerator / math.sqrt(left_ss * right_ss)


def average_ranks(values: Sequence[float]) -> list[float]:
    """Rank values from one with average ranks for ties."""

    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for offset in range(start, stop):
            result[order[offset]] = rank
        start = stop
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman's rank correlation with average tie handling."""

    return pearson(average_ranks(left), average_ranks(right))


def partial_spearman_by_time_bin(rows: Sequence[WindowRow]) -> float:
    """Spearman correlation after residualizing both ranks on window-bin fixed effects."""

    erank = [float(row.erank_primary) for row in rows if row.erank_primary is not None]
    quality = [row.imaging_quality for row in rows if row.erank_primary is not None]
    bins = [row.window_index for row in rows if row.erank_primary is not None]
    rank_erank = average_ranks(erank)
    rank_quality = average_ranks(quality)
    indices: dict[int, list[int]] = defaultdict(list)
    for index, time_bin in enumerate(bins):
        indices[time_bin].append(index)
    residual_erank = list(rank_erank)
    residual_quality = list(rank_quality)
    for members in indices.values():
        erank_mean = _mean([rank_erank[index] for index in members])
        quality_mean = _mean([rank_quality[index] for index in members])
        for index in members:
            residual_erank[index] -= erank_mean
            residual_quality[index] -= quality_mean
    return pearson(residual_erank, residual_quality)


def partial_spearman_two_way_fe(rows: Sequence[WindowRow]) -> float:
    """Rank correlation after trajectory and window-index fixed effects.

    For the pre-registered balanced panel this is the Pearson correlation of
    globally ranked eRank and quality after double demeaning by
    ``(prompt_id, seed)`` trajectory and by ``window_index``.
    """

    selected = [row for row in rows if row.erank_primary is not None]
    _require(selected, "$analysis.two_way_fe", "no primary observations")
    trajectory_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    time_indices: dict[int, list[int]] = defaultdict(list)
    cells: set[tuple[str, int, int]] = set()
    for index, row in enumerate(selected):
        cell = (row.prompt_id, row.seed, row.window_index)
        _require(cell not in cells, "$analysis.two_way_fe", f"duplicate panel cell {cell!r}")
        cells.add(cell)
        trajectory_indices[(row.prompt_id, row.seed)].append(index)
        time_indices[row.window_index].append(index)
    expected_bins = set(time_indices)
    _require(len(expected_bins) >= 2, "$analysis.two_way_fe", "requires at least two time bins")
    for trajectory, members in trajectory_indices.items():
        observed = {selected[index].window_index for index in members}
        _require(
            observed == expected_bins and len(members) == len(expected_bins),
            "$analysis.two_way_fe",
            f"unbalanced trajectory {trajectory!r}",
        )

    rank_erank = average_ranks([float(row.erank_primary) for row in selected])
    rank_quality = average_ranks([row.imaging_quality for row in selected])
    grand_erank = _mean(rank_erank)
    grand_quality = _mean(rank_quality)
    trajectory_erank = {
        key: _mean([rank_erank[index] for index in members])
        for key, members in trajectory_indices.items()
    }
    trajectory_quality = {
        key: _mean([rank_quality[index] for index in members])
        for key, members in trajectory_indices.items()
    }
    time_erank = {
        key: _mean([rank_erank[index] for index in members])
        for key, members in time_indices.items()
    }
    time_quality = {
        key: _mean([rank_quality[index] for index in members])
        for key, members in time_indices.items()
    }
    residual_erank: list[float] = []
    residual_quality: list[float] = []
    for index, row in enumerate(selected):
        trajectory = (row.prompt_id, row.seed)
        residual_erank.append(
            rank_erank[index]
            - trajectory_erank[trajectory]
            - time_erank[row.window_index]
            + grand_erank
        )
        residual_quality.append(
            rank_quality[index]
            - trajectory_quality[trajectory]
            - time_quality[row.window_index]
            + grand_quality
        )
    return pearson(residual_erank, residual_quality)


def _correlations(rows: Sequence[WindowRow]) -> dict[str, Any]:
    erank = [float(row.erank_primary) for row in rows if row.erank_primary is not None]
    quality = [row.imaging_quality for row in rows if row.erank_primary is not None]
    clusters = {row.prompt_id for row in rows if row.erank_primary is not None}
    if len(erank) < 3:
        raise ProtocolError("a correlation subset contains fewer than three observations")
    return {
        "observations": len(erank),
        "prompt_clusters": len(clusters),
        "spearman": spearman(erank, quality),
        "pearson": pearson(erank, quality),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ProtocolError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_spearman(
    rows: Sequence[WindowRow],
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    """Prompt-cluster percentile bootstrap for a one-sided lower bound."""

    by_prompt: dict[str, list[WindowRow]] = defaultdict(list)
    for row in rows:
        if row.erank_primary is not None:
            by_prompt[row.prompt_id].append(row)
    prompt_ids = sorted(by_prompt)
    _require(len(prompt_ids) >= 3, "$analysis", "bootstrap requires at least three prompts")
    _require(replicates >= 100, "$analysis", "bootstrap requires at least 100 replicates")
    generator = random.Random(seed)
    samples: list[float] = []
    failed = 0
    for _ in range(replicates):
        sampled_rows: list[WindowRow] = []
        for _cluster_index in range(len(prompt_ids)):
            sampled_rows.extend(by_prompt[generator.choice(prompt_ids)])
        try:
            result = _correlations(sampled_rows)["spearman"]
        except ProtocolError:
            failed += 1
            continue
        if math.isfinite(result):
            samples.append(result)
        else:
            failed += 1
    _require(
        len(samples) >= max(100, math.ceil(replicates * 0.95)),
        "$analysis",
        "too many degenerate bootstrap replicates",
    )
    return {
        "method": "prompt_cluster_percentile",
        "requested_replicates": replicates,
        "valid_replicates": len(samples),
        "failed_replicates": failed,
        "seed": seed,
        "one_sided_alpha": alpha,
        "lower_confidence_bound": _percentile(samples, alpha),
    }


def _subset_with_bootstrap(
    rows: Sequence[WindowRow],
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    result = _correlations(rows)
    result["bootstrap"] = cluster_bootstrap_spearman(
        rows,
        replicates=replicates,
        seed=seed,
        alpha=alpha,
    )
    return result


def analyze_rows(
    rows: Sequence[WindowRow],
    protocol: Mapping[str, Any],
    *,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Run the pre-registered primary analysis and named robustness checks."""

    primary_model = str(protocol["primary_erank"]["population_model_id"])
    primary = [row for row in rows if row.model_id == primary_model]
    _require(primary, "$analysis", "no rows exist for the primary model")
    analysis = protocol["analysis"]
    frozen_replicates = int(analysis["bootstrap_replicates"])
    replicates = frozen_replicates if bootstrap_replicates is None else bootstrap_replicates
    alpha = float(analysis["alpha"])
    seed = int(analysis["bootstrap_seed"])
    threshold = 0.6

    primary_result = _subset_with_bootstrap(
        primary,
        replicates=replicates,
        seed=seed,
        alpha=alpha,
    )
    lower_bound = primary_result["bootstrap"]["lower_confidence_bound"]
    primary_result.update(
        {
            "population_model_id": primary_model,
            "point_target_met": primary_result["spearman"] > threshold,
            "reject_h0_rho_le_0_6": lower_bound > threshold,
            "decision_rule": "one-sided prompt-cluster bootstrap LCB > 0.6",
        }
    )

    strata = {
        "prefix_15s": [row for row in primary if row.window_end_s <= 15.0],
        "full_30s": [row for row in primary if row.window_end_s <= 30.0],
        "late_15_to_30s": [row for row in primary if row.window_start_s >= 15.0],
        "end_window_15s": [row for row in primary if math.isclose(row.window_end_s, 15.0)],
        "end_window_30s": [row for row in primary if math.isclose(row.window_end_s, 30.0)],
    }
    stratified: dict[str, Any] = {}
    for offset, (name, subset) in enumerate(strata.items(), start=1):
        stratified[name] = _subset_with_bootstrap(
            subset,
            replicates=replicates,
            seed=seed + offset,
            alpha=alpha,
        )

    time_bins: dict[str, Any] = {}
    for index in sorted({row.window_index for row in primary}):
        subset = [row for row in primary if row.window_index == index]
        time_bins[str(index)] = _correlations(subset)

    model_quality: dict[str, Any] = {}
    for model in protocol["models"]:
        model_id = model["id"]
        values = [row.imaging_quality for row in rows if row.model_id == model_id]
        model_quality[model_id] = {"observations": len(values)}
        model_quality[model_id].update(
            {
                "mean_imaging_quality": _mean(values) if values else None,
                "minimum_imaging_quality": min(values) if values else None,
                "maximum_imaging_quality": max(values) if values else None,
            }
        )

    two_way_fe = partial_spearman_two_way_fe(primary)
    guardrail_threshold = float(analysis["two_way_fe_low_threshold"])
    guardrail_triggered = (
        primary_result["spearman"] > threshold and two_way_fe < guardrail_threshold
    )

    return {
        "schema": "erank_quality_correlation_analysis_v1",
        "confirmatory": bootstrap_replicates is None or replicates == frozen_replicates,
        "primary": primary_result,
        "duration_and_end_window_strata": stratified,
        "robustness": {
            "partial_spearman_controlling_window_index": partial_spearman_by_time_bin(primary),
            "two_way_fe_partial_spearman": two_way_fe,
            "per_time_bin": time_bins,
            "deployment_confounding_guardrail": {
                "low_threshold": guardrail_threshold,
                "triggered": guardrail_triggered,
                "interpretation": (
                    "When triggered, raw correlation may be driven by prompt or time drift and "
                    "does not by itself establish a deployable bad-block detector."
                ),
            },
        },
        "quality_descriptives_by_model": model_quality,
    }


def dump_analysis(result: Mapping[str, Any], path: str | Path | None = None) -> str:
    """Serialize an analysis deterministically and optionally write it."""

    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is not None:
        Path(path).write_text(payload, encoding="utf-8")
    return payload
