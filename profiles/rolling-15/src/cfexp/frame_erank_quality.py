"""Frame-aligned eRank/quality pilot validation, statistics, and plot data.

This module is intentionally independent from :mod:`cfexp.erank_quality`'s
frozen five-second-window study.  The new pilot has one row per generated
latent block, six simultaneously observed transformer layers, and one
lossless center-frame MUSIQ-SPAQ score for each complete block.

Only the Python standard library is required for validation and analysis.
Matplotlib is an optional renderer; the canonical plot-data JSON can always be
produced in the locked ``cfexp`` environment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

from .erank_quality import ProtocolError, average_ranks, pearson, spearman


PROTOCOL_SCHEMA = "erank_frame_quality_layer_ablation_protocol_v1"
ANALYSIS_SCHEMA = "erank_frame_quality_layer_ablation_analysis_v1"
PLOT_DATA_SCHEMA = "erank_across_blocks_plot_data_v1"
CORRELATION_PLOT_DATA_SCHEMA = "erank_frame_quality_correlation_plot_data_v1"
LAYER_NUMBERS = (1, 5, 10, 15, 20, 25)
LAYER_FIELDS = {layer: f"erank_l{layer}" for layer in LAYER_NUMBERS}
CSV_FIELDS = (
    "prompt_id",
    "analysis_split",
    "assigned_dimension",
    "seed",
    "model_id",
    "trajectory_seconds",
    "fps",
    "trajectory_frames",
    "block_index",
    "frame_start",
    "frame_end_exclusive",
    "analysis_eligible",
    "anchor_frame_index",
    "anchor_png_path",
    "anchor_png_sha256",
    "erank_l1",
    "erank_l5",
    "erank_l10",
    "erank_l15",
    "erank_l20",
    "erank_l25",
    "musiq_spaq_raw",
    "musiq_spaq",
)


@dataclass(frozen=True)
class FrameBlockRow:
    """All six eRanks and optional frame quality for one trajectory block."""

    prompt_id: str
    analysis_split: str
    assigned_dimension: str
    seed: int
    model_id: str
    trajectory_seconds: int
    fps: int
    trajectory_frames: int
    block_index: int
    frame_start: int
    frame_end_exclusive: int
    analysis_eligible: bool
    anchor_frame_index: int | None
    anchor_png_path: str | None
    anchor_png_sha256: str | None
    eranks: tuple[tuple[int, float], ...]
    musiq_spaq_raw: float | None
    musiq_spaq: float | None

    def erank(self, layer_number_one_based: int) -> float:
        for layer, value in self.eranks:
            if layer == layer_number_one_based:
                return value
        raise ProtocolError(f"row does not contain layer {layer_number_one_based}")


@dataclass(frozen=True)
class _Observation:
    cluster: str
    block_index: int
    erank: float
    quality: float


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ProtocolError(f"{path}: {message}")


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
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{path}: expected an integer") from exc


def _boolean(value: str, path: str) -> bool:
    normalized = value.strip().lower()
    _require(normalized in {"true", "false"}, path, "must be 'true' or 'false'")
    return normalized == "true"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def expected_block_support(block_index: int, *, trajectory_frames: int = 360) -> tuple[int, int]:
    """Return the retained half-open RGB-frame support for a block."""

    _require(0 <= block_index <= 30, "$block_index", "must be in [0, 30]")
    if block_index == 0:
        return 0, 9
    start = 12 * block_index - 3
    return start, min(12 * block_index + 9, trajectory_frames)


def expected_anchor_frame(block_index: int) -> int | None:
    """Return the zero-based lower-middle frame for complete blocks 1..29."""

    _require(0 <= block_index <= 30, "$block_index", "must be in [0, 30]")
    return 12 * block_index + 2 if 1 <= block_index <= 29 else None


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen frame-level pilot protocol."""

    protocol_path = Path(path)
    try:
        value = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load protocol {protocol_path}: {exc}") from exc
    validate_protocol(value)
    source_path = protocol_path.with_name(value["provenance"]["source_protocol_file"])
    try:
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load frozen prompt source {source_path}: {exc}") from exc
    _require(
        hashlib.sha256(source_bytes).hexdigest()
        == value["provenance"]["source_protocol_sha256"],
        "$.provenance.source_protocol_sha256",
        "does not match the referenced prompt protocol",
    )
    source_prompts = source.get("prompts")
    _require(isinstance(source_prompts, list), "$.provenance.source_protocol_file", "has no prompts")
    source_identity = source_prompts
    pilot_identity = [
        {key: item_value for key, item_value in item.items() if key != "analysis_split"}
        for item in value["prompts"]
    ]
    _require(
        pilot_identity == source_identity,
        "$.prompts",
        "does not match every field of the pinned source protocol prompts and order",
    )
    return value


def validate_protocol(value: Any) -> None:
    """Validate all outcome-independent choices in the frame-level pilot."""

    _require(isinstance(value, dict), "$", "expected an object")
    _require(value.get("schema") == PROTOCOL_SCHEMA, "$.schema", f"expected {PROTOCOL_SCHEMA!r}")
    _require(value.get("status") == "frozen", "$.status", "must be 'frozen'")
    _require(value.get("study_phase") == "pilot", "$.study_phase", "must be 'pilot'")

    provenance = value.get("provenance")
    _require(isinstance(provenance, dict), "$.provenance", "expected an object")
    _require(
        provenance.get("source_prompt_manifest_sha256")
        == "8a2f89312402c2688cb00074194d748e264dd47e5a2f2e6fe010ca2afb0d9d94",
        "$.provenance.source_prompt_manifest_sha256",
        "source prompt population drift",
    )
    _require(
        provenance.get("source_protocol_file") == "erank-quality-correlation-v1.protocol.json",
        "$.provenance.source_protocol_file",
        "must reference the frozen 48-prompt source protocol",
    )
    _require(
        _is_sha256(provenance.get("source_protocol_sha256")),
        "$.provenance.source_protocol_sha256",
        "must be a lowercase SHA-256",
    )

    design = value.get("design")
    _require(isinstance(design, dict), "$.design", "expected an object")
    frozen_design = {
        "prompt_count": 48,
        "dimension_count": 16,
        "prompts_per_dimension": 3,
        "seeds": [20260911],
        "trajectory_seconds": 15,
        "fps": 24,
        "trajectory_frames": 360,
        "block_indices": list(range(31)),
        "analysis_block_indices": list(range(1, 30)),
        "blocks_per_trajectory": 31,
        "eligible_blocks_per_trajectory": 29,
        "gpu_shards": 4,
        "prompts_per_gpu_shard": 12,
    }
    for field, expected in frozen_design.items():
        _require(design.get(field) == expected, f"$.design.{field}", f"must equal {expected!r}")
    expected_counts = design.get("expected_counts")
    _require(isinstance(expected_counts, dict), "$.design.expected_counts", "expected an object")
    frozen_counts = {
        "generated_trajectories": 48,
        "block_rows": 1488,
        "anchor_pngs": 1392,
        "raw_erank_scalars": 8928,
        "eligible_layer_observations": 8352,
        "development_prompts": 16,
        "test_prompts": 32,
        "development_pairs_per_layer": 464,
        "test_pairs_per_layer": 928,
    }
    for field, expected in frozen_counts.items():
        _require(
            expected_counts.get(field) == expected,
            f"$.design.expected_counts.{field}",
            f"must equal {expected}",
        )

    model = value.get("model")
    _require(isinstance(model, dict), "$.model", "expected an object")
    _require(model.get("id") == "self_forcing_1.3b", "$.model.id", "invalid model")
    _require(
        model.get("display_name") == "Self-Forcing 1.3B",
        "$.model.display_name",
        "invalid model display name",
    )

    probe = value.get("erank_probe")
    _require(isinstance(probe, dict), "$.erank_probe", "expected an object")
    expected_layers = [
        {
            "layer_number_one_based": layer,
            "layer_index_zero_based": index,
            "csv_field": LAYER_FIELDS[layer],
        }
        for layer, index in zip(LAYER_NUMBERS, (0, 4, 9, 14, 19, 24))
    ]
    _require(probe.get("layers") == expected_layers, "$.erank_probe.layers", "layer set drift")
    frozen_probe = {
        "capture_mode": "same_native_online_forward",
        "measurement_context": "online_denoising_step_0",
        "hidden_state_site": "post_transformer_block_output",
        "current_blocks": 1,
        "latent_frames_per_block": 3,
        "online_denoising_step_index": 0,
        "actual_timestep": 1000,
        "noise_source": "native_absolute_chunk_noise_derived_from_seed_and_chunk",
        "extra_probe_forward": False,
        "max_sampled_tokens": 256,
        "token_sampling": "deterministic_even_floor_i_times_T_over_N",
        "centering": "none",
        "gram": "H_sampled@H_sampled.T",
        "gram_dtype": "float32",
        "spectrum": "singular_values_from_sqrt_of_clamped_gram_eigenvalues",
        "gram_eigenvalue_relative_cutoff": 1e-7,
        "entropy_rank": "exp(-sum(p*log(p))) where p_i=sigma_i/sum_j(sigma_j)",
        "normalization": "none",
        "uses_absolute_level": True,
    }
    for field, expected in frozen_probe.items():
        _require(probe.get(field) == expected, f"$.erank_probe.{field}", f"must equal {expected!r}")

    alignment = value.get("frame_alignment")
    _require(isinstance(alignment, dict), "$.frame_alignment", "expected an object")
    frozen_alignment = {
        "frame_indexing": "zero_based",
        "support_interval": "half_open",
        "first_block_support": "[0,9)",
        "later_block_support_formula": "[12*i-3,12*i+9)",
        "trajectory_clip_frame": 360,
        "eligible_block_indices": list(range(1, 30)),
        "anchor_formula": "12*i+2",
        "anchor_choice": "lower_middle_frame",
        "boundary_blocks_excluded": [0, 30],
    }
    for field, expected in frozen_alignment.items():
        _require(
            alignment.get(field) == expected,
            f"$.frame_alignment.{field}",
            f"must equal {expected!r}",
        )

    quality = value.get("primary_quality")
    _require(isinstance(quality, dict), "$.primary_quality", "expected an object")
    frozen_quality = {
        "metric": "MUSIQ-SPAQ",
        "implementation": "VBench-Long imaging_quality frame scorer",
        "unit": "one_lossless_rgb_png",
        "input_source": "vae_decoded_rgb_before_video_encoding",
        "aggregation": "none_one_score_per_anchor_frame",
        "score_scale": "normalized_0_1",
        "resize_mode": "longer",
        "raw_score_field": "musiq_spaq_raw",
        "normalization": "musiq_spaq=musiq_spaq_raw/100",
        "direction": "higher_is_better",
    }
    for field, expected in frozen_quality.items():
        _require(quality.get(field) == expected, f"$.primary_quality.{field}", f"must equal {expected!r}")

    selection = value.get("split_and_layer_selection")
    _require(isinstance(selection, dict), "$.split_and_layer_selection", "expected an object")
    frozen_selection = {
        "development_prompts_per_dimension": 1,
        "test_prompts_per_dimension": 2,
        "development_rule": "first_selected_prompt_within_each_dimension",
        "test_rule": "remaining_two_selected_prompts_within_each_dimension",
        "candidate_raw_spearman_direction": "positive",
        "candidate_two_way_fe_minimum": 0.3,
        "selection_statistic": "development_raw_spearman",
        "near_tie_tolerance": 0.02,
        "near_tie_rule": "choose_shallower_layer_within_tolerance_of_maximum",
        "test_set_must_not_affect_selection": True,
    }
    for field, expected in frozen_selection.items():
        _require(
            selection.get(field) == expected,
            f"$.split_and_layer_selection.{field}",
            f"must equal {expected!r}",
        )

    analysis = value.get("analysis")
    _require(isinstance(analysis, dict), "$.analysis", "expected an object")
    frozen_analysis = {
        "primary_statistic": "spearman",
        "descriptive_statistic": "pearson",
        "cluster": "prompt_id",
        "cluster_contents": "all_29_eligible_blocks_for_one_prompt",
        "bootstrap_method": "percentile_prompt_cluster_bootstrap",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261012,
        "alpha": 0.05,
        "null_hypothesis": "rho<=0.6",
        "alternative": "rho>0.6",
        "point_target": 0.6,
        "two_way_fe_target": 0.3,
        "two_way_fe_bootstrap_lower_target": 0.0,
        "two_way_fixed_effects": ["prompt_id", "block_index"],
    }
    for field, expected in frozen_analysis.items():
        _require(analysis.get(field) == expected, f"$.analysis.{field}", f"must equal {expected!r}")

    plots = value.get("plots")
    _require(isinstance(plots, dict), "$.plots", "expected an object")
    _require(plots.get("trend_layers") == list(LAYER_NUMBERS), "$.plots.trend_layers", "layer drift")
    _require(plots.get("trend_blocks") == list(range(31)), "$.plots.trend_blocks", "block drift")
    _require(plots.get("raw_trajectory_count_per_layer") == 48, "$.plots", "must show 48 tracks")
    _require(plots.get("summary") == "cross_prompt_median", "$.plots.summary", "invalid")
    _require(plots.get("confidence_interval") == "pointwise_prompt_bootstrap_95pct", "$.plots", "invalid")
    _require(plots.get("bootstrap_replicates") == 10000, "$.plots", "invalid")
    _require(plots.get("bootstrap_seed") == 20261013, "$.plots", "invalid")
    _require(plots.get("smoothing") == "none", "$.plots.smoothing", "must be none")
    development_correlation = plots.get("development_correlation")
    _require(
        isinstance(development_correlation, dict),
        "$.plots.development_correlation",
        "expected an object",
    )
    frozen_development_plot = {
        "population": "development_only",
        "layout": "two_by_three_layer_facets_shared_axes",
        "mark": "hexbin",
        "shared_x_limits": True,
        "shared_y_limits": True,
        "annotations": ["raw_spearman", "two_way_fe_partial_spearman", "observations"],
    }
    for field, expected in frozen_development_plot.items():
        _require(
            development_correlation.get(field) == expected,
            f"$.plots.development_correlation.{field}",
            f"must equal {expected!r}",
        )
    held_out_correlation = plots.get("held_out_correlation")
    _require(
        isinstance(held_out_correlation, dict),
        "$.plots.held_out_correlation",
        "expected an object",
    )
    frozen_held_out_plot = {
        "population": "test_only_selected_layer",
        "mark": "scatter_colored_by_block_index",
        "block_color_range": [1, 29],
        "annotations": [
            "raw_spearman",
            "raw_spearman_one_sided_95pct_lcb",
            "two_way_fe_partial_spearman",
            "observations",
        ],
        "test_set_must_not_affect_layer_selection": True,
    }
    for field, expected in frozen_held_out_plot.items():
        _require(
            held_out_correlation.get(field) == expected,
            f"$.plots.held_out_correlation.{field}",
            f"must equal {expected!r}",
        )

    prompts = value.get("prompts")
    _require(isinstance(prompts, list) and len(prompts) == 48, "$.prompts", "must contain 48 prompts")
    prompt_ids: list[str] = []
    dimensions: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for index, prompt in enumerate(prompts):
        path = f"$.prompts[{index}]"
        _require(isinstance(prompt, dict), path, "expected an object")
        expected_fields = {
            "prompt_id",
            "selection_ordinal",
            "assigned_dimension",
            "benchmark_prompt_index",
            "prompt",
            "prompt_file_sha256",
            "selection_sha256",
            "analysis_split",
        }
        _require(set(prompt) == expected_fields, path, "unexpected prompt fields")
        expected_group = index // 3
        expected_offset = index % 3
        expected_ordinal = 8 * expected_group + expected_offset + 1
        expected_split = "development" if expected_offset == 0 else "test"
        _require(prompt.get("selection_ordinal") == expected_ordinal, f"{path}.selection_ordinal", "drift")
        _require(prompt.get("analysis_split") == expected_split, f"{path}.analysis_split", "drift")
        prompt_id = prompt.get("prompt_id")
        dimension = prompt.get("assigned_dimension")
        _require(isinstance(prompt_id, str) and prompt_id, f"{path}.prompt_id", "invalid")
        _require(isinstance(dimension, str) and dimension, f"{path}.assigned_dimension", "invalid")
        _require(
            isinstance(prompt.get("benchmark_prompt_index"), int),
            f"{path}.benchmark_prompt_index",
            "must be an integer",
        )
        prompt_text = prompt.get("prompt")
        _require(isinstance(prompt_text, str) and prompt_text, f"{path}.prompt", "must be non-empty")
        _require(
            hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            == prompt.get("prompt_file_sha256"),
            f"{path}.prompt_file_sha256",
            "does not match prompt text",
        )
        _require(
            _is_sha256(prompt.get("selection_sha256")),
            f"{path}.selection_sha256",
            "must be lowercase SHA-256",
        )
        if expected_offset:
            _require(
                dimension == prompts[index - expected_offset]["assigned_dimension"],
                f"{path}.assigned_dimension",
                "three-prompt dimension group drift",
            )
        prompt_ids.append(prompt_id)
        dimensions[dimension] += 1
        splits[expected_split] += 1
    _require(len(set(prompt_ids)) == 48, "$.prompts", "prompt IDs must be unique")
    _require(len(dimensions) == 16 and set(dimensions.values()) == {3}, "$.prompts", "must be 16x3")
    _require(splits == {"development": 16, "test": 32}, "$.prompts", "split must be 16/32")

    csv_contract = value.get("csv_contract")
    _require(isinstance(csv_contract, dict), "$.csv_contract", "expected an object")
    _require(csv_contract.get("unit") == "one_prompt_trajectory_block", "$.csv_contract.unit", "invalid")
    _require(csv_contract.get("required_columns") == list(CSV_FIELDS), "$.csv_contract.required_columns", "drift")
    _require(csv_contract.get("exact_columns") is True, "$.csv_contract.exact_columns", "must be true")


def _safe_relative_png(value: str, path: str) -> str:
    _require(value and "\x00" not in value, path, "invalid path")
    _require(
        not value.startswith(("~", "/", "\\\\"))
        and not PurePosixPath(value).is_absolute()
        and not PureWindowsPath(value).is_absolute(),
        path,
        "must be a relative path",
    )
    _require(".." not in value.replace("\\", "/").split("/"), path, "must not traverse")
    _require(value.lower().endswith(".png"), path, "must name a PNG")
    return value


def load_frame_rows(
    path: str | Path,
    protocol: Mapping[str, Any],
    *,
    frames_root: str | Path | None = None,
    require_complete: bool = True,
) -> list[FrameBlockRow]:
    """Load the wide block table and enforce the frozen 48x31 grid."""

    csv_path = Path(path)
    try:
        handle = csv_path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"cannot open frame metrics CSV {csv_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        _require(columns == CSV_FIELDS, "$csv.header", "must exactly match the frozen column order")
        records = list(reader)

    prompts = {item["prompt_id"]: item for item in protocol["prompts"]}
    seed = int(protocol["design"]["seeds"][0])
    model_id = str(protocol["model"]["id"])
    duration = int(protocol["design"]["trajectory_seconds"])
    fps = int(protocol["design"]["fps"])
    trajectory_frames = int(protocol["design"]["trajectory_frames"])
    root = Path(frames_root).resolve() if frames_root is not None else None

    rows: list[FrameBlockRow] = []
    seen: set[tuple[str, int, int]] = set()
    for line_number, raw in enumerate(records, start=2):
        prefix = f"$csv.line[{line_number}]"
        prompt_id = raw["prompt_id"].strip()
        _require(prompt_id in prompts, f"{prefix}.prompt_id", "not in frozen prompt set")
        prompt = prompts[prompt_id]
        _require(raw["analysis_split"].strip() == prompt["analysis_split"], f"{prefix}.analysis_split", "drift")
        _require(raw["assigned_dimension"].strip() == prompt["assigned_dimension"], f"{prefix}.assigned_dimension", "drift")
        row_seed = _integer(raw["seed"], f"{prefix}.seed")
        _require(row_seed == seed, f"{prefix}.seed", f"must equal {seed}")
        _require(raw["model_id"].strip() == model_id, f"{prefix}.model_id", "drift")
        row_duration = _integer(raw["trajectory_seconds"], f"{prefix}.trajectory_seconds")
        row_fps = _integer(raw["fps"], f"{prefix}.fps")
        row_frames = _integer(raw["trajectory_frames"], f"{prefix}.trajectory_frames")
        _require((row_duration, row_fps, row_frames) == (duration, fps, trajectory_frames), prefix, "trajectory contract drift")
        block_index = _integer(raw["block_index"], f"{prefix}.block_index")
        _require(0 <= block_index <= 30, f"{prefix}.block_index", "must be in [0,30]")
        frame_start = _integer(raw["frame_start"], f"{prefix}.frame_start")
        frame_end = _integer(raw["frame_end_exclusive"], f"{prefix}.frame_end_exclusive")
        expected_start, expected_end = expected_block_support(block_index, trajectory_frames=trajectory_frames)
        _require((frame_start, frame_end) == (expected_start, expected_end), prefix, "block frame support drift")
        eligible = _boolean(raw["analysis_eligible"], f"{prefix}.analysis_eligible")
        expected_anchor = expected_anchor_frame(block_index)
        _require(eligible == (expected_anchor is not None), f"{prefix}.analysis_eligible", "does not match blocks 1..29")

        anchor_text = raw["anchor_frame_index"].strip()
        png_path_text = raw["anchor_png_path"].strip()
        png_hash_text = raw["anchor_png_sha256"].strip()
        raw_quality = _finite_float(
            raw["musiq_spaq_raw"], f"{prefix}.musiq_spaq_raw", allow_blank=True
        )
        quality = _finite_float(raw["musiq_spaq"], f"{prefix}.musiq_spaq", allow_blank=True)
        if eligible:
            anchor = _integer(anchor_text, f"{prefix}.anchor_frame_index")
            _require(anchor == expected_anchor, f"{prefix}.anchor_frame_index", "anchor formula drift")
            _require(frame_start <= anchor < frame_end, f"{prefix}.anchor_frame_index", "outside block")
            png_path = _safe_relative_png(png_path_text, f"{prefix}.anchor_png_path")
            _require(_is_sha256(png_hash_text), f"{prefix}.anchor_png_sha256", "must be lowercase SHA-256")
            _require(
                raw_quality is not None and 0.0 <= raw_quality <= 100.0,
                f"{prefix}.musiq_spaq_raw",
                "must be in [0,100]",
            )
            _require(quality is not None and 0.0 <= quality <= 1.0, f"{prefix}.musiq_spaq", "must be in [0,1]")
            _require(
                math.isclose(quality, raw_quality / 100.0, rel_tol=0.0, abs_tol=1e-12),
                f"{prefix}.musiq_spaq",
                "must equal musiq_spaq_raw/100",
            )
            if root is not None:
                candidate = (root / png_path).resolve()
                _require(candidate != root and root in candidate.parents, f"{prefix}.anchor_png_path", "escapes frames root")
                try:
                    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                except OSError as exc:
                    raise ProtocolError(f"{prefix}.anchor_png_path: cannot read {candidate}: {exc}") from exc
                _require(digest == png_hash_text, f"{prefix}.anchor_png_sha256", "PNG digest mismatch")
        else:
            _require(not anchor_text and not png_path_text and not png_hash_text, prefix, "boundary anchor fields must be blank")
            _require(
                raw_quality is None and quality is None,
                f"{prefix}.musiq_spaq",
                "boundary quality must be blank",
            )
            anchor = None
            png_path = None
            png_hash_text = ""

        eranks: list[tuple[int, float]] = []
        for layer in LAYER_NUMBERS:
            rank = _finite_float(raw[LAYER_FIELDS[layer]], f"{prefix}.{LAYER_FIELDS[layer]}")
            assert rank is not None
            _require(1.0 <= rank <= 256.0, f"{prefix}.{LAYER_FIELDS[layer]}", "must be in [1,256]")
            eranks.append((layer, rank))

        key = (prompt_id, row_seed, block_index)
        _require(key not in seen, prefix, f"duplicate experimental cell {key!r}")
        seen.add(key)
        rows.append(
            FrameBlockRow(
                prompt_id=prompt_id,
                analysis_split=str(prompt["analysis_split"]),
                assigned_dimension=str(prompt["assigned_dimension"]),
                seed=row_seed,
                model_id=model_id,
                trajectory_seconds=row_duration,
                fps=row_fps,
                trajectory_frames=row_frames,
                block_index=block_index,
                frame_start=frame_start,
                frame_end_exclusive=frame_end,
                analysis_eligible=eligible,
                anchor_frame_index=anchor,
                anchor_png_path=png_path,
                anchor_png_sha256=png_hash_text or None,
                eranks=tuple(eranks),
                musiq_spaq_raw=raw_quality,
                musiq_spaq=quality,
            )
        )

    if require_complete:
        expected = {
            (prompt_id, seed, block_index)
            for prompt_id in prompts
            for block_index in range(31)
        }
        _require(seen == expected, "$csv", f"incomplete grid: missing={len(expected-seen)}, extra={len(seen-expected)}")
    _require(rows, "$csv", "contains no rows")
    return rows


def _observations(rows: Sequence[FrameBlockRow], *, split: str, layer: int) -> list[_Observation]:
    result: list[_Observation] = []
    for row in rows:
        if row.analysis_split != split or not row.analysis_eligible:
            continue
        _require(row.musiq_spaq is not None, "$analysis", "eligible row lacks quality")
        result.append(
            _Observation(
                cluster=row.prompt_id,
                block_index=row.block_index,
                erank=row.erank(layer),
                quality=float(row.musiq_spaq),
            )
        )
    return result


def _two_way_fe_spearman(observations: Sequence[_Observation]) -> float:
    _require(observations, "$analysis.two_way_fe", "no observations")
    prompts: dict[str, list[int]] = defaultdict(list)
    blocks: dict[int, list[int]] = defaultdict(list)
    cells: set[tuple[str, int]] = set()
    for index, observation in enumerate(observations):
        cell = (observation.cluster, observation.block_index)
        _require(cell not in cells, "$analysis.two_way_fe", f"duplicate panel cell {cell!r}")
        cells.add(cell)
        prompts[observation.cluster].append(index)
        blocks[observation.block_index].append(index)
    expected_blocks = set(blocks)
    _require(expected_blocks == set(range(1, 30)), "$analysis.two_way_fe", "requires blocks 1..29")
    for prompt_id, members in prompts.items():
        _require(
            {observations[index].block_index for index in members} == expected_blocks,
            "$analysis.two_way_fe",
            f"unbalanced prompt {prompt_id!r}",
        )

    rank_x = average_ranks([item.erank for item in observations])
    rank_y = average_ranks([item.quality for item in observations])
    grand_x = statistics.fmean(rank_x)
    grand_y = statistics.fmean(rank_y)
    prompt_x = {key: statistics.fmean(rank_x[i] for i in members) for key, members in prompts.items()}
    prompt_y = {key: statistics.fmean(rank_y[i] for i in members) for key, members in prompts.items()}
    block_x = {key: statistics.fmean(rank_x[i] for i in members) for key, members in blocks.items()}
    block_y = {key: statistics.fmean(rank_y[i] for i in members) for key, members in blocks.items()}
    residual_x = [
        rank_x[index] - prompt_x[item.cluster] - block_x[item.block_index] + grand_x
        for index, item in enumerate(observations)
    ]
    residual_y = [
        rank_y[index] - prompt_y[item.cluster] - block_y[item.block_index] + grand_y
        for index, item in enumerate(observations)
    ]
    return pearson(residual_x, residual_y)


def _percentile(values: Sequence[float], probability: float) -> float:
    _require(values, "$analysis.bootstrap", "no valid bootstrap values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    observations: Sequence[_Observation],
    *,
    statistic: Callable[[Sequence[_Observation]], float],
    replicates: int,
    seed: int,
    alpha: float,
    two_sided: bool = False,
) -> dict[str, Any]:
    _require(replicates > 0, "$analysis.bootstrap.replicates", "must be positive")
    by_prompt: dict[str, list[_Observation]] = defaultdict(list)
    for item in observations:
        by_prompt[item.cluster].append(item)
    prompt_ids = sorted(by_prompt)
    _require(len(prompt_ids) >= 3, "$analysis.bootstrap", "requires at least three prompt clusters")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled: list[_Observation] = []
        for draw_index in range(len(prompt_ids)):
            prompt_id = prompt_ids[rng.randrange(len(prompt_ids))]
            synthetic_cluster = f"{draw_index}:{prompt_id}"
            sampled.extend(
                _Observation(synthetic_cluster, item.block_index, item.erank, item.quality)
                for item in by_prompt[prompt_id]
            )
        try:
            estimates.append(statistic(sampled))
        except ProtocolError:
            continue
    _require(len(estimates) >= max(1, int(0.9 * replicates)), "$analysis.bootstrap", "too many degenerate replicates")
    lower_probability = alpha / 2.0 if two_sided else alpha
    result: dict[str, Any] = {
        "replicates_requested": replicates,
        "replicates_valid": len(estimates),
        "seed": seed,
        "lower_confidence_bound": _percentile(estimates, lower_probability),
    }
    if two_sided:
        result["upper_confidence_bound"] = _percentile(estimates, 1.0 - alpha / 2.0)
    return result


def _raw_spearman(observations: Sequence[_Observation]) -> float:
    return spearman([item.erank for item in observations], [item.quality for item in observations])


def _raw_pearson(observations: Sequence[_Observation]) -> float:
    return pearson([item.erank for item in observations], [item.quality for item in observations])


def select_layer(
    rows: Sequence[FrameBlockRow],
    protocol: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
) -> tuple[int | None, list[dict[str, Any]]]:
    """Select a layer using development prompts only."""

    analysis = protocol["analysis"]
    selection = protocol["split_and_layer_selection"]
    summaries: list[dict[str, Any]] = []
    for layer in LAYER_NUMBERS:
        observations = _observations(rows, split="development", layer=layer)
        _require(len(observations) == 464, "$analysis.development", "requires 16x29 observations")
        raw = _raw_spearman(observations)
        fixed_effect = _two_way_fe_spearman(observations)
        eligible = raw > 0.0 and fixed_effect >= float(selection["candidate_two_way_fe_minimum"])
        summaries.append(
            {
                "layer_number_one_based": layer,
                "observations": len(observations),
                "prompt_clusters": len({item.cluster for item in observations}),
                "raw_spearman": raw,
                "pearson": _raw_pearson(observations),
                "two_way_fe_partial_spearman": fixed_effect,
                "eligible_for_selection": eligible,
                "raw_spearman_bootstrap": _bootstrap(
                    observations,
                    statistic=_raw_spearman,
                    replicates=bootstrap_replicates,
                    seed=int(analysis["bootstrap_seed"]) + layer,
                    alpha=float(analysis["alpha"]),
                    two_sided=True,
                ),
            }
        )
    candidates = [item for item in summaries if item["eligible_for_selection"]]
    if not candidates:
        return None, summaries
    maximum = max(float(item["raw_spearman"]) for item in candidates)
    tolerance = float(selection["near_tie_tolerance"])
    selected = min(
        int(item["layer_number_one_based"])
        for item in candidates
        if maximum - float(item["raw_spearman"]) <= tolerance
    )
    return selected, summaries


def analyze_rows(
    rows: Sequence[FrameBlockRow],
    protocol: Mapping[str, Any],
    *,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Run leakage-safe development selection and held-out test analysis."""

    frozen_replicates = int(protocol["analysis"]["bootstrap_replicates"])
    replicates = frozen_replicates if bootstrap_replicates is None else bootstrap_replicates
    selected, development = select_layer(rows, protocol, bootstrap_replicates=replicates)
    result: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "study_phase": "pilot",
        "confirmatory": False,
        "fixed_seed": int(protocol["design"]["seeds"][0]),
        "bootstrap_replicates": replicates,
        "selected_layer_number_one_based": selected,
        "development_layer_ablation": development,
        "held_out_test": None,
    }
    if selected is None:
        result["selection_outcome"] = "no_layer_met_pre_registered_candidate_guardrails"
        return result

    observations = _observations(rows, split="test", layer=selected)
    _require(len(observations) == 928, "$analysis.test", "requires 32x29 observations")
    analysis = protocol["analysis"]
    raw = _raw_spearman(observations)
    fixed_effect = _two_way_fe_spearman(observations)
    raw_bootstrap = _bootstrap(
        observations,
        statistic=_raw_spearman,
        replicates=replicates,
        seed=int(analysis["bootstrap_seed"]),
        alpha=float(analysis["alpha"]),
    )
    fe_bootstrap = _bootstrap(
        observations,
        statistic=_two_way_fe_spearman,
        replicates=replicates,
        seed=int(analysis["bootstrap_seed"]) + 1000,
        alpha=float(analysis["alpha"]),
    )
    point_target = float(analysis["point_target"])
    fe_target = float(analysis["two_way_fe_target"])
    fe_lower_target = float(analysis["two_way_fe_bootstrap_lower_target"])
    strict_target_met = (
        raw > point_target
        and float(raw_bootstrap["lower_confidence_bound"]) > point_target
        and fixed_effect >= fe_target
        and float(fe_bootstrap["lower_confidence_bound"]) > fe_lower_target
    )
    result["selection_outcome"] = "selected_on_development_only"
    result["held_out_test"] = {
        "layer_number_one_based": selected,
        "observations": len(observations),
        "prompt_clusters": len({item.cluster for item in observations}),
        "raw_spearman": raw,
        "pearson": _raw_pearson(observations),
        "raw_spearman_bootstrap": raw_bootstrap,
        "two_way_fe_partial_spearman": fixed_effect,
        "two_way_fe_bootstrap": fe_bootstrap,
        "strict_pilot_target_met": strict_target_met,
        "interpretation": (
            "This fixed-seed pilot can motivate a multi-seed confirmation, but cannot by "
            "itself establish seed-general correlation."
        ),
    }
    return result


def build_erank_trend_plot_data(
    rows: Sequence[FrameBlockRow],
    protocol: Mapping[str, Any],
    *,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Build canonical six-panel eRank trajectory data without plotting deps."""

    plot = protocol["plots"]
    replicates = int(plot["bootstrap_replicates"]) if bootstrap_replicates is None else bootstrap_replicates
    _require(replicates > 0, "$plots.bootstrap_replicates", "must be positive")
    prompt_order = [item["prompt_id"] for item in protocol["prompts"]]
    row_map = {(row.prompt_id, row.block_index): row for row in rows}
    _require(
        all((prompt_id, block) in row_map for prompt_id in prompt_order for block in range(31)),
        "$plots",
        "trend plot requires the complete 48x31 grid",
    )
    rng = random.Random(int(plot["bootstrap_seed"]))
    bootstrap_draws = [
        [rng.randrange(len(prompt_order)) for _ in prompt_order]
        for _ in range(replicates)
    ]
    layers: list[dict[str, Any]] = []
    for layer in LAYER_NUMBERS:
        matrix = [
            [row_map[(prompt_id, block)].erank(layer) for block in range(31)]
            for prompt_id in prompt_order
        ]
        bootstrap_by_block: list[list[float]] = [[] for _ in range(31)]
        for draw in bootstrap_draws:
            for block in range(31):
                bootstrap_by_block[block].append(
                    statistics.median(matrix[prompt_index][block] for prompt_index in draw)
                )
        summary = []
        for block in range(31):
            values = [track[block] for track in matrix]
            summary.append(
                {
                    "block_index": block,
                    "n": len(values),
                    "median": statistics.median(values),
                    "ci_lower": _percentile(bootstrap_by_block[block], 0.025),
                    "ci_upper": _percentile(bootstrap_by_block[block], 0.975),
                    "analysis_eligible": 1 <= block <= 29,
                }
            )
        layers.append(
            {
                "layer_number_one_based": layer,
                "trajectories": [
                    {
                        "prompt_id": prompt_id,
                        "analysis_split": protocol["prompts"][index]["analysis_split"],
                        "values": matrix[index],
                    }
                    for index, prompt_id in enumerate(prompt_order)
                ],
                "summary": summary,
            }
        )
    return {
        "schema": PLOT_DATA_SCHEMA,
        "x_axis": "block_index",
        "y_axis": "absolute_effective_rank",
        "layers": layers,
        "bootstrap": {
            "unit": "prompt_id",
            "replicates": replicates,
            "seed": int(plot["bootstrap_seed"]),
            "interval": "pointwise_two_sided_95_percentile",
        },
        "excluded_boundary_blocks": [0, 30],
        "smoothing": "none",
    }


def _axis_limits(values: Sequence[float]) -> list[float]:
    """Return finite non-degenerate canonical limits before render-time padding."""

    _require(values, "$plots.axis_limits", "cannot derive limits from no values")
    lower = min(values)
    upper = max(values)
    _require(math.isfinite(lower) and math.isfinite(upper), "$plots.axis_limits", "non-finite")
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1e-6)
        return [lower - padding, upper + padding]
    return [lower, upper]


def build_correlation_plot_data(
    rows: Sequence[FrameBlockRow],
    protocol: Mapping[str, Any],
    analysis_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build canonical data for the six-layer and held-out correlation figures.

    Development and test observations remain in separate objects.  The
    selected layer and every annotation are copied from the already completed
    leakage-safe analysis instead of being recalculated from plot choices.
    """

    summaries = analysis_result.get("development_layer_ablation")
    _require(
        isinstance(summaries, list) and len(summaries) == len(LAYER_NUMBERS),
        "$correlation_plot.analysis.development_layer_ablation",
        "expected six layer summaries",
    )
    summary_by_layer = {
        int(item["layer_number_one_based"]): item
        for item in summaries
        if isinstance(item, Mapping)
    }
    _require(
        set(summary_by_layer) == set(LAYER_NUMBERS),
        "$correlation_plot.analysis.development_layer_ablation",
        "layer summary set drift",
    )

    development_layers: list[dict[str, Any]] = []
    all_development_x: list[float] = []
    all_development_y: list[float] = []
    for layer in LAYER_NUMBERS:
        observations = _observations(rows, split="development", layer=layer)
        _require(len(observations) == 464, "$correlation_plot.development", "requires 16x29")
        points = [
            {
                "prompt_id": item.cluster,
                "block_index": item.block_index,
                "erank": item.erank,
                "musiq_spaq": item.quality,
            }
            for item in observations
        ]
        all_development_x.extend(item.erank for item in observations)
        all_development_y.extend(item.quality for item in observations)
        summary = summary_by_layer[layer]
        development_layers.append(
            {
                "layer_number_one_based": layer,
                "points": points,
                "annotations": {
                    "raw_spearman": float(summary["raw_spearman"]),
                    "two_way_fe_partial_spearman": float(
                        summary["two_way_fe_partial_spearman"]
                    ),
                    "observations": int(summary["observations"]),
                    "prompt_clusters": int(summary["prompt_clusters"]),
                    "eligible_for_selection": bool(summary["eligible_for_selection"]),
                },
            }
        )

    selected_raw = analysis_result.get("selected_layer_number_one_based")
    selected = int(selected_raw) if selected_raw is not None else None
    held_out: dict[str, Any] | None = None
    if selected is not None:
        _require(selected in LAYER_NUMBERS, "$correlation_plot.selected_layer", "invalid")
        held_out_analysis = analysis_result.get("held_out_test")
        _require(isinstance(held_out_analysis, Mapping), "$correlation_plot.held_out_test", "missing")
        observations = _observations(rows, split="test", layer=selected)
        _require(len(observations) == 928, "$correlation_plot.held_out_test", "requires 32x29")
        test_points = [
            {
                "prompt_id": item.cluster,
                "block_index": item.block_index,
                "erank": item.erank,
                "musiq_spaq": item.quality,
            }
            for item in observations
        ]
        bootstrap = held_out_analysis.get("raw_spearman_bootstrap")
        _require(isinstance(bootstrap, Mapping), "$correlation_plot.held_out_test.bootstrap", "missing")
        held_out = {
            "layer_number_one_based": selected,
            "points": test_points,
            "axis_limits": {
                "x": _axis_limits([item.erank for item in observations]),
                "y": _axis_limits([item.quality for item in observations]),
            },
            "color": {
                "field": "block_index",
                "minimum": 1,
                "maximum": 29,
                "palette": "viridis",
            },
            "annotations": {
                "raw_spearman": float(held_out_analysis["raw_spearman"]),
                "raw_spearman_one_sided_95pct_lcb": float(
                    bootstrap["lower_confidence_bound"]
                ),
                "two_way_fe_partial_spearman": float(
                    held_out_analysis["two_way_fe_partial_spearman"]
                ),
                "observations": int(held_out_analysis["observations"]),
                "prompt_clusters": int(held_out_analysis["prompt_clusters"]),
                "strict_pilot_target_met": bool(
                    held_out_analysis["strict_pilot_target_met"]
                ),
            },
        }

    return {
        "schema": CORRELATION_PLOT_DATA_SCHEMA,
        "axes": {
            "x": "absolute_effective_rank",
            "y": "single_frame_musiq_spaq",
        },
        "development": {
            "population": "development_only",
            "shared_axis_limits": {
                "x": _axis_limits(all_development_x),
                "y": _axis_limits(all_development_y),
            },
            "layers": development_layers,
            "mark": "hexbin",
        },
        "held_out_test": held_out,
        "selection_provenance": {
            "selected_layer_number_one_based": selected,
            "selection_population": "development_only",
            "test_set_used_for_selection": False,
        },
    }


def render_erank_trend_plot(plot_data: Mapping[str, Any], output: str | Path) -> None:
    """Optionally render the canonical 2x3 trend figure with Matplotlib."""

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProtocolError(
            "optional Matplotlib renderer requested, but matplotlib is not installed; "
            "use the canonical plot-data JSON or install a separately pinned renderer"
        ) from exc

    layers = plot_data.get("layers")
    _require(isinstance(layers, list) and len(layers) == 6, "$plot_data.layers", "expected six layers")
    colors = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
    for axis, layer, color in zip(axes.flat, layers, colors):
        for trajectory in layer["trajectories"]:
            axis.plot(range(31), trajectory["values"], color=color, alpha=0.09, linewidth=0.7)
        summary = layer["summary"]
        x = [item["block_index"] for item in summary]
        median = [item["median"] for item in summary]
        lower = [item["ci_lower"] for item in summary]
        upper = [item["ci_upper"] for item in summary]
        axis.fill_between(x, lower, upper, color=color, alpha=0.22, linewidth=0)
        axis.plot(x, median, color=color, linewidth=2.4, label="median")
        axis.axvspan(-0.5, 0.5, color="#9ca3af", alpha=0.18)
        axis.axvspan(29.5, 30.5, color="#9ca3af", alpha=0.18)
        axis.set_title(f"Layer {layer['layer_number_one_based']}")
        axis.grid(alpha=0.2)
    figure.supxlabel("Autoregressive block index")
    figure.supylabel("Absolute effective rank")
    figure.suptitle("Hidden-state Effective Rank Across Blocks")
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _padded_limits(limits: Sequence[float]) -> tuple[float, float]:
    _require(len(limits) == 2, "$plot_data.axis_limits", "expected [minimum, maximum]")
    lower = float(limits[0])
    upper = float(limits[1])
    span = upper - lower
    _require(math.isfinite(span) and span > 0.0, "$plot_data.axis_limits", "invalid span")
    padding = span * 0.04
    return lower - padding, upper + padding


def render_correlation_plots(
    plot_data: Mapping[str, Any],
    *,
    development_output: str | Path | None = None,
    held_out_output: str | Path | None = None,
) -> None:
    """Optionally render the development ablation and held-out main figure.

    Output formats are selected by the file suffix understood by Matplotlib,
    so both PNG and vector PDF are supported without changing plot data.
    """

    _require(
        development_output is not None or held_out_output is not None,
        "$correlation_plot.output",
        "at least one output path is required",
    )
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProtocolError(
            "optional Matplotlib renderer requested, but matplotlib is not installed; "
            "use the canonical correlation plot-data JSON or install a separately pinned renderer"
        ) from exc

    _require(
        plot_data.get("schema") == CORRELATION_PLOT_DATA_SCHEMA,
        "$correlation_plot.schema",
        f"expected {CORRELATION_PLOT_DATA_SCHEMA!r}",
    )
    development = plot_data.get("development")
    _require(isinstance(development, Mapping), "$correlation_plot.development", "missing")

    if development_output is not None:
        layers = development.get("layers")
        _require(
            isinstance(layers, list) and len(layers) == 6,
            "$correlation_plot.development.layers",
            "expected six layers",
        )
        shared = development.get("shared_axis_limits")
        _require(isinstance(shared, Mapping), "$correlation_plot.development.shared_axis_limits", "missing")
        x_limits = _padded_limits(shared["x"])
        y_limits = _padded_limits(shared["y"])
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
        for axis, layer in zip(axes.flat, layers):
            points = layer["points"]
            x = [point["erank"] for point in points]
            y = [point["musiq_spaq"] for point in points]
            axis.hexbin(x, y, gridsize=26, mincnt=1, cmap="Blues", linewidths=0.15)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            annotation = layer["annotations"]
            axis.text(
                0.03,
                0.97,
                (
                    f"raw ρ={annotation['raw_spearman']:.3f}\n"
                    f"FE ρ={annotation['two_way_fe_partial_spearman']:.3f}\n"
                    f"n={annotation['observations']} points / "
                    f"{annotation['prompt_clusters']} prompts"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
            )
            axis.set_title(f"Layer {layer['layer_number_one_based']}")
            axis.grid(alpha=0.15)
        figure.supxlabel("Hidden-state absolute effective rank")
        figure.supylabel("Single-frame MUSIQ-SPAQ")
        figure.suptitle("Development-set Layer Ablation: eRank vs. Frame Quality")
        figure.tight_layout()
        destination = Path(development_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(figure)

    if held_out_output is not None:
        held_out = plot_data.get("held_out_test")
        _require(
            isinstance(held_out, Mapping),
            "$correlation_plot.held_out_test",
            "no development-selected layer exists, so the held-out plot is unavailable",
        )
        points = held_out["points"]
        x = [point["erank"] for point in points]
        y = [point["musiq_spaq"] for point in points]
        blocks = [point["block_index"] for point in points]
        figure, axis = plt.subplots(figsize=(7.4, 6.2))
        scatter = axis.scatter(
            x,
            y,
            c=blocks,
            cmap="viridis",
            vmin=1,
            vmax=29,
            s=15,
            alpha=0.48,
            linewidths=0,
        )
        axis.set_xlim(*_padded_limits(held_out["axis_limits"]["x"]))
        axis.set_ylim(*_padded_limits(held_out["axis_limits"]["y"]))
        annotation = held_out["annotations"]
        axis.text(
            0.03,
            0.97,
            (
                f"raw ρ={annotation['raw_spearman']:.3f}\n"
                f"95% LCB={annotation['raw_spearman_one_sided_95pct_lcb']:.3f}\n"
                f"FE ρ={annotation['two_way_fe_partial_spearman']:.3f}\n"
                f"n={annotation['observations']} points / "
                f"{annotation['prompt_clusters']} prompts"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.84},
        )
        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label("Autoregressive block index")
        axis.set_xlabel("Hidden-state absolute effective rank")
        axis.set_ylabel("Single-frame MUSIQ-SPAQ")
        axis.set_title(
            "Held-out Test Correlation "
            f"(Development-selected Layer {held_out['layer_number_one_based']})"
        )
        axis.grid(alpha=0.18)
        figure.tight_layout()
        destination = Path(held_out_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)


def dump_json(value: Mapping[str, Any], path: str | Path | None = None) -> str:
    """Serialize analysis or plot data deterministically."""

    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is not None:
        Path(path).write_text(payload, encoding="utf-8")
    return payload
