"""Frozen, outcome-independent protocol helpers for Insight 2(a)/(b).

The experiment runner and the statistics code both consume this module so the
VAE time mapping, prompt split, quality windows, and candidate positions cannot
silently drift apart.  This module intentionally has no ML dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .erank_quality import ProtocolError


PROTOCOL_SCHEMA = "insight2_correction_protocol_v1"
PROTOCOL_OVERLAY_SCHEMA = "insight2_correction_protocol_overlay_v1"
DEFAULT_CHUNK_COUNT = 16
CHUNK_INDICES = tuple(range(16))
CANDIDATE_CHUNK_INDICES = tuple(range(1, 16))
LATENT_FRAMES_PER_CHUNK = 3
VAE_TEMPORAL_DOWNSAMPLE = 4
RGB_FRAME_COUNT = 189
DEADLINE_EVALUATION_PROMPT_IDS = (
    "001_subject_consistency_vbl0268",
    "002_subject_consistency_vbl0276",
    "003_subject_consistency_vbl0257",
    "009_background_consistency_vbl0805",
    "010_background_consistency_vbl0819",
    "011_background_consistency_vbl0823",
    "017_temporal_flickering_vbl0042",
    "018_temporal_flickering_vbl0004",
    "019_temporal_flickering_vbl0008",
    "025_motion_smoothness_vbl0291",
    "026_motion_smoothness_vbl0282",
    "027_motion_smoothness_vbl0327",
    "033_dynamic_degree_vbl0309",
    "034_dynamic_degree_vbl0308",
    "035_dynamic_degree_vbl0325",
    "041_aesthetic_quality_vbl0746",
    "042_aesthetic_quality_vbl0719",
    "043_aesthetic_quality_vbl0689",
    "049_imaging_quality_vbl0699",
    "050_imaging_quality_vbl0757",
    "051_imaging_quality_vbl0745",
    "057_object_class_vbl0375",
    "058_object_class_vbl0331",
    "059_object_class_vbl0379",
)
DEADLINE_PROTOCOL_IDS = frozenset(
    {
        "insight2-correction-long30-h800-deadline24-v1",
        "insight2-correction-long60-h800-deadline24-v1",
        "insight2-correction-long60-h800-coverage-v1",
        "insight2-correction-long60-h800-lookahead-coverage-v1",
        "insight2-correction-long60-h800-lookahead-boundary-v1",
    }
)
DEADLINE_PROTOCOL_CHUNK_COUNTS = {
    "insight2-correction-long30-h800-deadline24-v1": 60,
    "insight2-correction-long60-h800-deadline24-v1": 120,
    "insight2-correction-long60-h800-coverage-v1": 120,
    "insight2-correction-long60-h800-lookahead-coverage-v1": 120,
    "insight2-correction-long60-h800-lookahead-boundary-v1": 120,
}
COVERAGE_CONSTRAINED_PROTOCOL_ID = (
    "insight2-correction-long60-h800-coverage-v1"
)
LOOKAHEAD_COVERAGE_PROTOCOL_ID = (
    "insight2-correction-long60-h800-lookahead-coverage-v1"
)
LOOKAHEAD_BOUNDARY_PROTOCOL_ID = (
    "insight2-correction-long60-h800-lookahead-boundary-v1"
)
LOOKAHEAD_PROTOCOL_IDS = frozenset(
    {LOOKAHEAD_COVERAGE_PROTOCOL_ID, LOOKAHEAD_BOUNDARY_PROTOCOL_ID}
)
COVERAGE_PROTOCOL_IDS = frozenset(
    {
        COVERAGE_CONSTRAINED_PROTOCOL_ID,
        LOOKAHEAD_COVERAGE_PROTOCOL_ID,
        LOOKAHEAD_BOUNDARY_PROTOCOL_ID,
    }
)


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ProtocolError(f"{path}: {message}")


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: Any, path: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), path, "expected integer")
    return value


def _finite_number(value: Any, path: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        path,
        "expected number",
    )
    result = float(value)
    _require(math.isfinite(result), path, "expected finite number")
    return result


def _normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively apply a small frozen overlay without mutating its base."""

    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class ResolvedInsight2Protocol:
    """Validated protocol plus the fully materialized prompt partitions."""

    raw: dict[str, Any]
    eval_prompts: tuple[dict[str, Any], ...]
    calibration_prompts: tuple[dict[str, Any], ...]
    smoke_prompts: tuple[dict[str, Any], ...]
    source_protocol_path: Path


def rgb_frame_count(latent_frame_count: int) -> int:
    """Wan causal-VAE mapping ``1 + 4 * (L - 1)``."""

    _require(latent_frame_count >= 1, "$latent_frame_count", "must be positive")
    return 1 + VAE_TEMPORAL_DOWNSAMPLE * (latent_frame_count - 1)


def chunk_indices(chunk_count: int = DEFAULT_CHUNK_COUNT) -> tuple[int, ...]:
    """Return all chunk indices for a validated trajectory length."""

    parsed = _integer(chunk_count, "$chunk_count")
    _require(parsed >= 2, "$chunk_count", "must be at least 2")
    return tuple(range(parsed))


def candidate_chunk_indices(
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> tuple[int, ...]:
    """Return all post-prefix chunks eligible for Draft/Target selection."""

    return chunk_indices(chunk_count)[1:]


def frame_count_for_chunks(chunk_count: int) -> int:
    """Return the exact causal-VAE RGB frame count for ``chunk_count``."""

    parsed = len(chunk_indices(chunk_count))
    return rgb_frame_count(parsed * LATENT_FRAMES_PER_CHUNK)


def chunk_rgb_support(
    chunk_index: int, *, chunk_count: int = DEFAULT_CHUNK_COUNT
) -> tuple[int, int]:
    """Return the exact, non-overlapping half-open RGB support for one chunk."""

    _require(
        chunk_index in chunk_indices(chunk_count),
        "$chunk_index",
        f"must be in [0, {chunk_count - 1}]",
    )
    if chunk_index == 0:
        return 0, 9
    return 12 * chunk_index - 3, 12 * chunk_index + 9


def local_temporal_window(
    chunk_index: int, *, chunk_count: int = DEFAULT_CHUNK_COUNT
) -> tuple[int, int]:
    """Return 13 frames: the preceding context frame plus a candidate chunk."""

    _require(
        chunk_index in candidate_chunk_indices(chunk_count),
        "$chunk_index",
        f"must be a correction candidate in [1, {chunk_count - 1}]",
    )
    start, stop = chunk_rgb_support(chunk_index, chunk_count=chunk_count)
    return start - 1, stop


def duration_seconds(
    chunk_count: int = DEFAULT_CHUNK_COUNT, *, fps: int = 24
) -> Fraction:
    """Exact video playback duration at the frozen 24 FPS."""

    parsed_fps = _integer(fps, "$fps")
    _require(parsed_fps > 0, "$fps", "must be positive")
    return Fraction(frame_count_for_chunks(chunk_count), parsed_fps)


def _validate_extra_prompt(item: Any, index: int) -> dict[str, Any]:
    path = f"$.prompt_sources.official_vbench2_long.selected[{index}]"
    _require(isinstance(item, Mapping), path, "expected object")
    prompt_id = item.get("prompt_id")
    prompt = item.get("prompt")
    source_index = item.get("source_index_zero_based")
    split = item.get("split")
    dimensions = item.get("dimensions")
    _require(isinstance(prompt_id, str) and prompt_id.strip(), f"{path}.prompt_id", "invalid")
    _require(isinstance(prompt, str) and prompt.strip(), f"{path}.prompt", "invalid")
    _integer(source_index, f"{path}.source_index_zero_based")
    _require(split in {"evaluation", "calibration"}, f"{path}.split", "invalid")
    _require(
        isinstance(dimensions, list)
        and dimensions
        and all(isinstance(value, str) and value for value in dimensions),
        f"{path}.dimensions",
        "must be a non-empty string list",
    )
    selection_hash = item.get("selection_sha256")
    _require(_is_sha256(selection_hash), f"{path}.selection_sha256", "invalid SHA-256")
    namespace = "insight2-prompt-v1"
    expected = hashlib.sha256(f"{namespace}\0{source_index}\0{prompt}".encode()).hexdigest()
    _require(selection_hash == expected, f"{path}.selection_sha256", "does not match prompt")
    return dict(item)


def validate_protocol(value: Any) -> None:
    """Fail closed on every outcome-independent choice used by the experiment."""

    _require(isinstance(value, Mapping), "$", "expected object")
    _require(value.get("schema") == PROTOCOL_SCHEMA, "$.schema", f"expected {PROTOCOL_SCHEMA!r}")
    _require(value.get("status") == "frozen", "$.status", "must be 'frozen'")
    protocol_id = value.get("protocol_id")
    _require(isinstance(protocol_id, str) and protocol_id, "$.protocol_id", "invalid")
    deadline_profile = protocol_id in DEADLINE_PROTOCOL_IDS
    design = value.get("design")
    _require(isinstance(design, Mapping), "$.design", "expected object")
    chunk_count = _integer(design.get("chunk_count"), "$.design.chunk_count")
    _require(
        chunk_count in {16, 60, 120},
        "$.design.chunk_count",
        "must be one of the frozen 16, 60, or 120 chunk horizons",
    )
    is_long = chunk_count in {60, 120}
    expected_frames = frame_count_for_chunks(chunk_count)
    _require(
        not deadline_profile or is_long,
        "$.protocol_id",
        "the H800 deadline profile is only valid for a long-video horizon",
    )
    if deadline_profile:
        _require(
            chunk_count == DEADLINE_PROTOCOL_CHUNK_COUNTS[protocol_id],
            "$.design.chunk_count",
            "does not match the frozen H800 deadline protocol horizon",
        )
    if deadline_profile:
        expected_eval_count = 24
    elif is_long:
        expected_eval_count = 48
    else:
        expected_eval_count = 50
    expected_smoke_count = 1 if is_long else 10
    expected_design = {
        "width": 832,
        "height": 480,
        "fps": 24,
        "denoising_steps": 4,
        "chunk_count": chunk_count,
        "latent_frames_per_chunk": 3,
        "latent_frame_count": 3 * chunk_count,
        "rgb_frame_count": expected_frames,
        "candidate_chunk_indices": list(candidate_chunk_indices(chunk_count)),
        "seeds": [20260911] if deadline_profile else [20260911, 20260912],
        "evaluation_prompt_count": expected_eval_count,
        "calibration_prompt_count": 10,
        "smoke_prompt_count": expected_smoke_count,
    }
    for key, expected in expected_design.items():
        _require(design.get(key) == expected, f"$.design.{key}", f"must equal {expected!r}")
    if deadline_profile:
        _require(
            design.get("evaluation_selection")
            == "first_24_base_prompts_in_frozen_source_order",
            "$.design.evaluation_selection",
            "must freeze the quality-blind first-24 rule",
        )
        _require(
            design.get("evaluation_prompt_ids")
            == list(DEADLINE_EVALUATION_PROMPT_IDS),
            "$.design.evaluation_prompt_ids",
            "must freeze the exact first 24 base prompt IDs",
        )
    expected_smoke_selection = (
        "first_1_evaluation_prompt_in_frozen_source_order"
        if is_long
        else "first_10_evaluation_prompts_in_frozen_source_order"
    )
    _require(
        design.get("smoke_selection") == expected_smoke_selection,
        "$.design.smoke_selection",
        "must freeze the quality-blind evaluation-prefix rule",
    )
    smoke_ids = design.get("smoke_prompt_ids")
    _require(
        isinstance(smoke_ids, list)
        and len(smoke_ids) == expected_smoke_count
        and len(set(smoke_ids)) == expected_smoke_count
        and all(isinstance(item, str) and item for item in smoke_ids),
        "$.design.smoke_prompt_ids",
        f"must contain {expected_smoke_count} unique prompt IDs",
    )
    expected_duration = str(Fraction(expected_frames, 24))
    _require(
        design.get("exact_duration_seconds") == expected_duration,
        "$.design.exact_duration_seconds",
        f"must be the exact rational {expected_frames}/24",
    )
    _require(
        rgb_frame_count(design["latent_frame_count"]) == design["rgb_frame_count"],
        "$.design.rgb_frame_count",
        "does not satisfy the causal-VAE temporal relation",
    )

    models = value.get("models")
    _require(isinstance(models, Mapping), "$.models", "expected object")
    draft_id = models.get("draft", {}).get("id")
    _require(draft_id in {"self_forcing_1.3b", "longlive_1.3b", "rolling_forcing_1.3b"}, "$.models.draft.id", "invalid")
    if draft_id == "longlive_1.3b":
        _require(protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID,
                 "$.protocol_id", "LongLive requires the frozen boundary-policy profile")
        _require(models["draft"] == {
            "id": "longlive_1.3b",
            "native_cache_policy": "local_attn_size_12_sink_size_3",
            "checkpoint_role": "longlive_generator_with_lora",
        }, "$.models.draft", "LongLive native identity/cache/LoRA binding drifted")
        _require(models.get("branch_cache_policy") == "model_specific_history_bound_incremental_cache",
                 "$.models.branch_cache_policy", "LongLive requires history-bound native branches")
    if draft_id == "rolling_forcing_1.3b":
        _require(protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID,
                 "$.protocol_id", "Rolling requires the frozen boundary-policy profile")
        _require(models["draft"] == {
            "id": "rolling_forcing_1.3b",
            "native_cache_policy": "native_rolling_accepted_history_bound_future_state",
            "checkpoint_role": "rolling_ema_longvideo",
        }, "$.models.draft", "Rolling native identity/cache/checkpoint binding drifted")
        _require(models.get("branch_cache_policy") == "model_specific_history_bound_incremental_cache",
                 "$.models.branch_cache_policy", "Rolling requires history-bound native branches")
    _require(models.get("target", {}).get("id") == "krea-realtime-video-14b", "$.models.target.id", "invalid")
    _require(
        models.get("chunk_zero_policy") == "target",
        "$.models.chunk_zero_policy",
        "chunk 0 must use Target",
    )
    _require(
        models.get("cross_model_state") == "latent_history_only",
        "$.models.cross_model_state",
        "Draft and Target may share only latent history",
    )

    risk = value.get("risk")
    _require(isinstance(risk, Mapping), "$.risk", "expected object")
    expected_risk = {
        "layer_number_one_based": 15,
        "layer_index_zero_based": 14,
        "denoising_step_index": 0,
        "actual_timestep": 1000,
        "token_cap": 256,
        "center_hidden": False,
        "gram_dtype": "float32",
        "relative_gram_eigenvalue_cutoff": 1e-7,
        "direction": (
            "larger_causal_relative_drop_is_higher_risk"
            if is_long
            else "lower_rank_is_higher_risk"
        ),
        "calibration_cdf": (
            "midrank_less_fraction"
            if is_long
            else "midrank_greater_fraction"
        ),
    }
    for key, expected in expected_risk.items():
        actual = risk.get(key)
        if isinstance(expected, float):
            _require(_finite_number(actual, f"$.risk.{key}") == expected, f"$.risk.{key}", "invalid")
        else:
            _require(actual == expected, f"$.risk.{key}", f"must equal {expected!r}")
    if is_long:
        long_risk = {
            "signal": "causal_trailing_median_relative_erank_drop",
            "lookback_chunks": 4,
            "formula": "max(0,(median(previous_at_most_4_draft_eranks)-current_draft_erank)/median(previous_at_most_4_draft_eranks))",
            "cold_start": "zero_when_no_previous_draft_erank",
            "history_visibility": "current_and_past_draft_proposals_only",
            "decision_timing": "after_current_draft_proposal_before_accept_or_target_replace",
        }
        for key, expected in long_risk.items():
            _require(
                risk.get(key) == expected,
                f"$.risk.{key}",
                f"must equal {expected!r}",
            )
        expected_lookahead_gate = {
            "preview_distance_chunks": 1,
            "guarded_chunk": "pending_draft_t",
            "signal_chunk": "frontier_draft_t_plus_1",
            "accept_rule": "commit_pending_and_reuse_frontier_as_next_pending",
            "reject_rule": (
                "discard_pending_and_frontier_then_target_regenerate_pending_"
                "from_last_committed_prefix_with_same_noise"
            ),
            "successor_rule_after_reject": (
                "regenerate_frontier_from_target_corrected_prefix_with_same_noise"
            ),
            "risk_history": (
                "active_branch_committed_draft_ranks_plus_pending_and_frontier"
            ),
            "discarded_branch_history": "audit_only_excluded_from_future_risk",
            "after_exact_k_budget_exhaustion": (
                "commit_remaining_drafts_without_unused_preview"
            ),
            "terminal_out_of_range_preview": "forbidden",
        }
        if protocol_id in LOOKAHEAD_PROTOCOL_IDS:
            _require(
                risk.get("lookahead_gate") == expected_lookahead_gate,
                "$.risk.lookahead_gate",
                "does not match the frozen one-chunk lookahead gate",
            )
        else:
            _require(
                risk.get("lookahead_gate") is None,
                "$.risk.lookahead_gate",
                "is reserved for the frozen lookahead-coverage protocol",
            )
        expected_boundary_gate = {
            "strategy": "online_lookahead_coverage_boundary_gated",
            "route_id_template": (
                "online-lookahead-coverage-boundary-gated-erank-drop-k{budget}"
            ),
            "decision_metric_id": "paired_successor_erank_boundary_witness",
            "decision_metric_version": "v1",
            "proposal_semantics": {
                "proposal_opportunity_budget": 6,
                "controller_action": "open_target_proposal_opportunity",
                "proposed_target_chunks": (
                    "exactly_6_controller_selected_chunks_one_per_frozen_window"
                ),
                "accepted_target_chunks": (
                    "ordered_subset_of_proposed_target_chunks"
                ),
                "rejected_target_chunks": (
                    "ordered_complement_within_proposed_target_chunks"
                ),
                "opportunity_consumption": (
                    "every_materialized_target_candidate_consumes_one_"
                    "opportunity_regardless_of_acceptance"
                ),
            },
            "same_noise_branches": {
                "pending_and_target_chunk_t": "same_noise_t",
                "draft_and_target_successor_chunk_t_plus_1": (
                    "same_noise_t_plus_1"
                ),
            },
            "erank_dominance": {
                "metric": (
                    "target_conditioned_successor_erank_minus_"
                    "draft_conditioned_successor_erank"
                ),
                "threshold": 0,
                "comparison": "greater_than_or_equal",
                "equality": "pass",
            },
            "boundary_dominance": {
                "result_field": "boundary_velocity_witness",
                "witness_schema": "correction_boundary_velocity_witness_v1",
                "metric_id": "latent_velocity_transition_discontinuity_v1",
                "chunk_zero_witness": None,
                "witness_fields": [
                    "schema",
                    "metric_id",
                    "accumulator_dtype",
                    "round_decimals",
                    "input_dtype",
                    "history_frame_count",
                    "candidate_frame_count",
                    "previous_velocity_norm",
                    "boundary_velocity_norm",
                    "candidate_velocity_norm",
                    "incoming_discontinuity",
                    "outgoing_discontinuity",
                    "score",
                ],
                "latent_layout": "batch_time_channels_height_width",
                "accumulator_dtype": "float64",
                "adjacent_frames_only": True,
                "previous_velocity": "history_last_minus_history_penultimate",
                "boundary_velocity": "candidate_first_minus_history_last",
                "candidate_velocity": "candidate_second_minus_candidate_first",
                "distance_formula": (
                    "l2_norm_a_minus_b_divided_by_l2_norm_a_plus_l2_norm_b"
                ),
                "both_zero_distance": 0,
                "incoming": "distance_previous_velocity_boundary_velocity",
                "outgoing": "distance_boundary_velocity_candidate_velocity",
                "score": "max_incoming_outgoing",
                "round_decimals": 12,
                "comparison_operands": "rounded_12_decimal_witness_components",
                "entry_metric": (
                    "target_candidate_witness_score_minus_pending_draft_"
                    "witness_score"
                ),
                "exit_metric": (
                    "target_conditioned_successor_witness_score_minus_draft_"
                    "conditioned_successor_witness_score"
                ),
                "entry_threshold": 0,
                "exit_threshold": 0,
                "comparison": "less_than_or_equal",
                "equality": "pass",
                "all_components_required": True,
            },
            "accept_rule": (
                "accept_target_only_when_erank_and_boundary_paired_"
                "dominance_both_pass"
            ),
            "accept_transition": (
                "commit_target_discard_pending_draft_and_draft_frontier_and_"
                "carry_target_conditioned_successor"
            ),
            "reject_transition": (
                "commit_pending_draft_reuse_draft_frontier_and_discard_"
                "target_candidate_and_target_conditioned_successor"
            ),
            "discarded_branch_history": "audit_only_excluded_from_future_risk",
            "threshold_source": "preregistered_zero_paired_dominance",
            "old_smoke_tuning": "forbidden",
        }
        if protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID:
            _require(
                risk.get("boundary_acceptance_gate") == expected_boundary_gate,
                "$.risk.boundary_acceptance_gate",
                "does not match the frozen boundary acceptance gate",
            )
        else:
            _require(
                risk.get("boundary_acceptance_gate") is None,
                "$.risk.boundary_acceptance_gate",
                "is reserved for the frozen boundary-gated protocol",
            )

    quality = value.get("quality")
    _require(isinstance(quality, Mapping), "$.quality", "expected object")
    if is_long:
        _require(
            quality.get("primary") == "vbench_long_custom_six_dimension_macro_mean",
            "$.quality.primary",
            "invalid",
        )
        _require(quality.get("implementation") == "official_vbench_long", "$.quality.implementation", "invalid")
        _require(quality.get("mode") == "long_custom_input", "$.quality.mode", "invalid")
        _require(
            quality.get("dimensions")
            == [
                "subject_consistency",
                "background_consistency",
                "motion_smoothness",
                "dynamic_degree",
                "aesthetic_quality",
                "imaging_quality",
            ],
            "$.quality.dimensions",
            "invalid",
        )
        _require(
            quality.get("aggregation")
            == "unweighted_arithmetic_mean_of_six_official_dimension_scores",
            "$.quality.aggregation",
            "invalid",
        )
        _require(quality.get("full_video_frames") == f"all_{expected_frames}_frames", "$.quality.full_video_frames", "invalid")
    else:
        _require(quality.get("primary") == "musiq_spaq_lossless_rgb_mean", "$.quality.primary", "invalid")
        _require(quality.get("primary_divisor") == 100, "$.quality.primary_divisor", "invalid")
        _require(
            quality.get("temporal") == "lossless_rgb_temporal_flicker_proxy",
            "$.quality.temporal",
            "invalid",
        )
        _require(quality.get("local_boundary_rule") == "preceding_plus_chunk", "$.quality.local_boundary_rule", "invalid")

    multi = value.get("multi_budget")
    _require(isinstance(multi, Mapping), "$.multi_budget", "expected object")
    expected_budget = 3 if chunk_count == 60 else 6 if chunk_count == 120 else None
    expected_budgets = [expected_budget] if is_long else [2, 3, 4]
    _require(multi.get("budgets") == expected_budgets, "$.multi_budget.budgets", "invalid")
    if protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID:
        expected_random_repeats = 0
    elif deadline_profile:
        expected_random_repeats = 3
    else:
        expected_random_repeats = 10
    _require(
        multi.get("random_repeats") == expected_random_repeats,
        "$.multi_budget.random_repeats",
        f"must equal {expected_random_repeats}",
    )
    if is_long:
        if protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID:
            expected_strategies = [
                "uniform",
                "online_lookahead_coverage_guided",
                "online_lookahead_coverage_boundary_gated",
            ]
        else:
            expected_strategies = ["random", "uniform", "online_rank_guided"]
            if protocol_id in COVERAGE_PROTOCOL_IDS:
                expected_strategies.append("online_coverage_guided")
            if protocol_id == LOOKAHEAD_COVERAGE_PROTOCOL_ID:
                expected_strategies.append("online_lookahead_coverage_guided")
        _require(
            multi.get("strategies") == expected_strategies,
            "$.multi_budget.strategies",
            "invalid",
        )
        _require(multi.get("oracle") == "not_in_scope", "$.multi_budget.oracle", "invalid")
        expected_uniform = [
            math.floor(ordinal * chunk_count / (expected_budget + 1) + 0.5)
            for ordinal in range(1, expected_budget + 1)
        ]
        _require(
            isinstance(multi.get("uniform_positions"), Mapping)
            and multi["uniform_positions"].get(str(expected_budget)) == expected_uniform,
            "$.multi_budget.uniform_positions",
            "invalid",
        )
        coverage = multi.get("coverage_constraint")
        if protocol_id in COVERAGE_PROTOCOL_IDS:
            coverage_path = "$.multi_budget.coverage_constraint"
            _require(isinstance(coverage, Mapping), coverage_path, "expected object")
            for field in (
                "window_radius_chunks",
                "selection_per_window",
                "minimum_gap_chunks",
                "maximum_gap_chunks",
            ):
                _integer(coverage.get(field), f"{coverage_path}.{field}")
            anchors = coverage.get("anchors")
            _require(
                isinstance(anchors, Mapping),
                f"{coverage_path}.anchors",
                "expected object",
            )
            anchor_values = anchors.get("6")
            _require(
                isinstance(anchor_values, list),
                f'{coverage_path}.anchors["6"]',
                "expected list",
            )
            for index, anchor in enumerate(anchor_values):
                _integer(anchor, f'{coverage_path}.anchors["6"][{index}]')
            windows = coverage.get("windows")
            _require(
                isinstance(windows, Mapping),
                f"{coverage_path}.windows",
                "expected object",
            )
            window_values = windows.get("6")
            _require(
                isinstance(window_values, list),
                f'{coverage_path}.windows["6"]',
                "expected list",
            )
            for window_index, window in enumerate(window_values):
                _require(
                    isinstance(window, list),
                    f'{coverage_path}.windows["6"][{window_index}]',
                    "expected list",
                )
                for endpoint_index, endpoint in enumerate(window):
                    _integer(
                        endpoint,
                        (
                            f'{coverage_path}.windows["6"]'
                            f"[{window_index}][{endpoint_index}]"
                        ),
                    )
            expected_coverage = {
                "anchor_rule": "round_half_up_chunk_count_times_j_over_K_plus_1",
                "anchors": {"6": [17, 34, 51, 69, 86, 103]},
                "window_radius_chunks": 4,
                "windows": {
                    "6": [
                        [13, 21],
                        [30, 38],
                        [47, 55],
                        [65, 73],
                        [82, 90],
                        [99, 107],
                    ]
                },
                "selection_per_window": 1,
                "minimum_gap_chunks": 8,
                "maximum_gap_chunks": 20,
                "dynamic_eligibility_rule": (
                    "max(static_window_start,last_target_chunk_plus_minimum_gap)"
                ),
                "dynamic_deadline_rule": (
                    "min(static_window_end,last_target_chunk_plus_maximum_gap)"
                ),
                "threshold_quantile_method": "higher",
                "threshold_probability": (
                    "one_minus_one_over_remaining_eligible_slots_through_deadline"
                ),
                "history_visibility": "current_and_past_draft_proposals_only",
                "emergency_override": "none",
                "age_bonus": "none",
            }
            _require(
                coverage == expected_coverage,
                "$.multi_budget.coverage_constraint",
                "does not match the frozen anchored causal coverage policy",
            )
        else:
            _require(
                coverage is None,
                "$.multi_budget.coverage_constraint",
                "is reserved for the frozen coverage-constrained protocol",
            )
        expected_lookahead_constraint = {
            "controller": "same_frozen_coverage_controller",
            "risk_source": "one_chunk_frontier_preview",
            "commit_lag_chunks": 1,
            "target_budget": "chunk_zero_plus_exactly_6_replacements",
            "latest_possible_final_correction_chunk": 107,
            "preview_after_budget_exhaustion": False,
        }
        lookahead_constraint = multi.get("lookahead_constraint")
        if protocol_id in LOOKAHEAD_PROTOCOL_IDS:
            _require(
                lookahead_constraint == expected_lookahead_constraint,
                "$.multi_budget.lookahead_constraint",
                "does not match the frozen lookahead coverage controller",
            )
        else:
            _require(
                lookahead_constraint is None,
                "$.multi_budget.lookahead_constraint",
                "is reserved for the frozen lookahead-coverage protocol",
            )
        expected_boundary_constraint = {
            "controller": "same_frozen_coverage_controller",
            "proposal_opportunity_budget": 6,
            "proposal_opportunities": "exactly_one_per_frozen_window",
            "budget_accounting": "proposal_opportunities_not_acceptances",
            "target_call_budget": (
                "shared_chunk_zero_plus_exactly_6_target_proposal_calls"
            ),
            "proposed_target_chunks": "exactly_6",
            "accepted_target_chunks": "ordered_subset_size_0_through_6",
            "latest_possible_final_proposal_chunk": 107,
            "preview_after_proposal_budget_exhaustion": False,
            "route_revision": "same_exact_git_sha_for_all_three_routes",
        }
        boundary_constraint = multi.get("boundary_gated_constraint")
        if protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID:
            _require(
                boundary_constraint == expected_boundary_constraint,
                "$.multi_budget.boundary_gated_constraint",
                "does not match the frozen proposal-opportunity budget",
            )
        else:
            _require(
                boundary_constraint is None,
                "$.multi_budget.boundary_gated_constraint",
                "is reserved for the frozen boundary-gated protocol",
            )
    else:
        _require(
            multi.get("figure_oracle_label") == "Greedy Oracle",
            "$.multi_budget.figure_oracle_label",
            "the main curve must not claim a global Oracle",
        )
        _require(
            multi.get("exact_oracle_subset_status") == "not_run",
            "$.multi_budget.exact_oracle_subset_status",
            "must not claim an exhaustive Oracle experiment that is not implemented",
        )

    analysis = value.get("analysis")
    _require(isinstance(analysis, Mapping), "$.analysis", "expected object")
    _require(analysis.get("bootstrap_cluster") == "prompt_id", "$.analysis.bootstrap_cluster", "invalid")
    _require(analysis.get("bootstrap_replicates") == 10000, "$.analysis.bootstrap_replicates", "invalid")
    _require(analysis.get("bootstrap_ci") == 0.95, "$.analysis.bootstrap_ci", "invalid")
    _require(analysis.get("quantile_method") == "type7", "$.analysis.quantile_method", "invalid")
    if protocol_id == COVERAGE_CONSTRAINED_PROTOCOL_ID:
        _require(
            analysis.get("primary_contrast")
            == (
                "online_coverage_guided_minus_online_rank_guided_and_"
                "online_coverage_guided_minus_uniform"
            ),
            "$.analysis.primary_contrast",
            "must freeze the coverage-versus-online-and-uniform contrast",
        )
    elif protocol_id == LOOKAHEAD_COVERAGE_PROTOCOL_ID:
        _require(
            analysis.get("primary_contrast")
            == (
                "online_lookahead_coverage_guided_minus_online_coverage_guided_"
                "and_online_lookahead_coverage_guided_minus_uniform"
            ),
            "$.analysis.primary_contrast",
            "must freeze the lookahead-versus-current-and-uniform contrast",
        )
    elif protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID:
        _require(
            analysis.get("primary_contrast")
            == (
                "online_lookahead_coverage_boundary_gated_minus_"
                "online_lookahead_coverage_guided_and_online_lookahead_"
                "coverage_boundary_gated_minus_uniform"
            ),
            "$.analysis.primary_contrast",
            "must freeze the boundary-gated-versus-lookahead-and-uniform contrast",
        )
        _require(
            analysis.get("comparison_controls")
            == {
                "code_revision": "same_exact_git_sha",
                "prompt_seed": "same_frozen_prompt_and_seed",
                "model_artifacts": "same_digest_pinned_draft_and_target",
                "chunk_noise": "same_prompt_seed_chunk_noise_per_absolute_chunk",
                "scorer": "same_official_vbench_long_six_dimension_contract",
            },
            "$.analysis.comparison_controls",
            "must freeze the same-SHA three-arm controls",
        )
    if deadline_profile:
        expected_random_analysis = (
            "not_applicable_three_deterministic_routes_only"
            if protocol_id == LOOKAHEAD_BOUNDARY_PROTOCOL_ID
            else "collapse_3_route_repeats_within_video_before_prompt_cluster_bootstrap"
        )
        _require(
            analysis.get("random_multi_budget") == expected_random_analysis,
            "$.analysis.random_multi_budget",
            "does not match the frozen deterministic/random route profile",
        )

    prompt_sources = value.get("prompt_sources")
    _require(isinstance(prompt_sources, Mapping), "$.prompt_sources", "expected object")
    base = prompt_sources.get("base_48")
    _require(isinstance(base, Mapping), "$.prompt_sources.base_48", "expected object")
    _require(base.get("prompt_count") == 48, "$.prompt_sources.base_48.prompt_count", "invalid")
    _require(_is_sha256(base.get("sha256")), "$.prompt_sources.base_48.sha256", "invalid")
    official = prompt_sources.get("official_vbench2_long")
    _require(isinstance(official, Mapping), "$.prompt_sources.official_vbench2_long", "expected object")
    _require(_is_git_commit(official.get("commit")), "$.prompt_sources.official_vbench2_long.commit", "invalid commit")
    _require(_is_sha256(official.get("source_sha256")), "$.prompt_sources.official_vbench2_long.source_sha256", "invalid")
    selected = official.get("selected")
    _require(isinstance(selected, list) and len(selected) == 12, "$.prompt_sources.official_vbench2_long.selected", "must contain 12 prompts")
    extras = [_validate_extra_prompt(item, index) for index, item in enumerate(selected)]
    _require(sum(item["split"] == "evaluation" for item in extras) == 2, "$.prompt_sources", "need two evaluation extras")
    _require(sum(item["split"] == "calibration" for item in extras) == 10, "$.prompt_sources", "need ten calibration prompts")
    ids = [item["prompt_id"] for item in extras]
    _require(len(ids) == len(set(ids)), "$.prompt_sources", "duplicate extra prompt_id")
    texts = [_normalized_prompt(item["prompt"]) for item in extras]
    _require(len(texts) == len(set(texts)), "$.prompt_sources", "duplicate extra prompt text")


def load_protocol(path: str | Path) -> ResolvedInsight2Protocol:
    """Load, validate, and resolve the frozen 48+2/10 prompt partitions."""

    protocol_path = Path(path).expanduser().resolve(strict=True)
    try:
        loaded = json.loads(
            protocol_path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"invalid JSON constant {value!r}")
            ),
        )
    except ProtocolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load protocol {protocol_path}: {error}") from error
    raw = loaded
    if isinstance(loaded, Mapping) and loaded.get("schema") == PROTOCOL_OVERLAY_SCHEMA:
        _require(loaded.get("status") == "frozen", "$.status", "must be 'frozen'")
        base_file = loaded.get("base_protocol_file")
        _require(
            isinstance(base_file, str) and base_file,
            "$.base_protocol_file",
            "must be a non-empty relative path",
        )
        relative_base = Path(base_file)
        _require(
            not relative_base.is_absolute() and ".." not in relative_base.parts,
            "$.base_protocol_file",
            "must be a safe relative path",
        )
        base_path = (protocol_path.parent / relative_base).resolve(strict=True)
        _require(
            base_path.parent == protocol_path.parent,
            "$.base_protocol_file",
            "must stay in the protocol directory",
        )
        _require(
            _is_sha256(loaded.get("base_protocol_sha256")),
            "$.base_protocol_sha256",
            "invalid SHA-256",
        )
        _require(
            _sha256_file(base_path) == loaded["base_protocol_sha256"],
            "$.base_protocol_sha256",
            "base protocol hash mismatch",
        )
        try:
            base_raw = json.loads(
                base_path.read_text(encoding="utf-8"),
                object_pairs_hook=_duplicate_checked_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ProtocolError(f"invalid JSON constant {value!r}")
                ),
            )
        except ProtocolError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProtocolError(f"cannot load base protocol {base_path}: {error}") from error
        _require(
            isinstance(base_raw, Mapping) and base_raw.get("schema") == PROTOCOL_SCHEMA,
            "$.base_protocol_file",
            "must reference a full Insight 2 protocol",
        )
        overrides = loaded.get("overrides")
        _require(isinstance(overrides, Mapping), "$.overrides", "expected object")
        _require(
            "schema" not in overrides and "prompt_sources" not in overrides,
            "$.overrides",
            "may not replace schema or prompt provenance",
        )
        raw = _deep_merge(base_raw, overrides)
    validate_protocol(raw)

    base_spec = raw["prompt_sources"]["base_48"]
    relative = Path(base_spec["file"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "$.prompt_sources.base_48.file", "must be a safe relative path")
    source_path = (protocol_path.parent / relative).resolve(strict=True)
    _require(source_path.parent == protocol_path.parent, "$.prompt_sources.base_48.file", "must stay in the protocol directory")
    _require(_sha256_file(source_path) == base_spec["sha256"], "$.prompt_sources.base_48.sha256", "source protocol hash mismatch")
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load source prompt protocol {source_path}: {error}") from error
    prompts = source.get("prompts")
    _require(isinstance(prompts, list) and len(prompts) == 48, "$.prompt_sources.base_48", "source must contain 48 prompts")
    eval_prompts = [dict(item, insight2_split="evaluation") for item in prompts]

    extras = raw["prompt_sources"]["official_vbench2_long"]["selected"]
    expected_eval_count = int(raw["design"]["evaluation_prompt_count"])
    if expected_eval_count == 24:
        eval_prompts = eval_prompts[:24]
        _require(
            [str(item["prompt_id"]) for item in eval_prompts]
            == raw["design"]["evaluation_prompt_ids"],
            "$.design.evaluation_prompt_ids",
            "does not match the first 24 frozen source prompts",
        )
    elif expected_eval_count == 50:
        eval_prompts.extend(dict(item, insight2_split="evaluation") for item in extras if item["split"] == "evaluation")
    calibration = [dict(item, insight2_split="calibration") for item in extras if item["split"] == "calibration"]
    _require(
        len(eval_prompts) == expected_eval_count,
        "$.prompt_sources",
        f"resolved evaluation set must have {expected_eval_count} prompts",
    )
    _require(len(calibration) == 10, "$.prompt_sources", "resolved calibration set must have 10 prompts")
    smoke_ids = raw["design"]["smoke_prompt_ids"]
    smoke_count = int(raw["design"]["smoke_prompt_count"])
    smoke = eval_prompts[:smoke_count]
    _require(
        [item["prompt_id"] for item in smoke] == smoke_ids,
        "$.design.smoke_prompt_ids",
        f"does not match the frozen first {smoke_count} evaluation prompt(s)",
    )
    all_ids = [str(item["prompt_id"]) for item in eval_prompts + calibration]
    _require(len(all_ids) == len(set(all_ids)), "$.prompt_sources", "resolved prompt IDs overlap")
    all_texts = [_normalized_prompt(str(item["prompt"])) for item in eval_prompts + calibration]
    _require(len(all_texts) == len(set(all_texts)), "$.prompt_sources", "resolved prompt texts overlap")
    return ResolvedInsight2Protocol(
        raw=dict(raw),
        eval_prompts=tuple(eval_prompts),
        calibration_prompts=tuple(calibration),
        # The pilot is the first, outcome-independent slice of E50.  C10 is
        # reserved exclusively for calibrating the risk CDF and online policy,
        # so no pilot/evaluation decision is made on calibration examples.
        smoke_prompts=tuple(smoke),
        source_protocol_path=source_path,
    )


def verify_frame_partition(chunk_count: int = DEFAULT_CHUNK_COUNT) -> None:
    """Internal invariant used by tests and campaign preflight."""

    indices = chunk_indices(chunk_count)
    candidates = candidate_chunk_indices(chunk_count)
    expected_frames = frame_count_for_chunks(chunk_count)
    supports = [
        chunk_rgb_support(index, chunk_count=chunk_count) for index in indices
    ]
    _require(supports[0][0] == 0, "$supports", "partition must start at frame 0")
    for previous, current in zip(supports, supports[1:]):
        _require(previous[1] == current[0], "$supports", "partition has a gap or overlap")
    _require(
        supports[-1][1] == expected_frames,
        "$supports",
        f"partition must end at frame {expected_frames}",
    )
    _require(all(stop - start == 12 for start, stop in supports[1:]), "$supports", "candidate chunks must contain 12 frames")
    _require(
        all(
            stop - start == 13
            for start, stop in (
                local_temporal_window(index, chunk_count=chunk_count)
                for index in candidates
            )
        ),
        "$temporal_windows",
        "candidate temporal windows must contain 13 frames",
    )
