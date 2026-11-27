"""Frozen contract bundle for the six-method ``main-table-v1`` benchmark.

This module is deliberately an execution gate, not a generation backend.  It
validates the frozen protocol, method provenance, artifact requirements,
two-GPU roles, output/timing contracts, and the generic cfexp manifest.  The
checked-in bundle remains fail-closed until every method adapter and every
required artifact binding exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from typing import Any, Mapping, Sequence

from .manifest import ManifestError, load_manifest, validate_manifest


PROTOCOL_SCHEMA = "main_table_protocol_v1"
CONFIG_SCHEMA = "main_table_config_v1"
PROTOCOL_ID = "main-table-v1"
CONFIG_ID = "main-table-v1"

PROTOCOL_RELATIVE_PATH = Path("experiments/main-table-v1.protocol.json")
CONFIG_RELATIVE_PATH = Path("experiments/main-table-v1.config.json")
MANIFEST_RELATIVE_PATH = Path("experiments/main-table-v1.manifest.json")

PROMPT_SOURCE_RELATIVE_PATH = Path(
    "experiments/erank-quality-correlation-v1.protocol.json"
)
PROMPT_SOURCE_SHA256 = (
    "98dfe2b98bc50a54b92f88e227130b2a28addfd8fccc2ec47a982e8bb8fe0ccb"
)
DRAFT_ARTIFACT_RELATIVE_PATH = Path(
    "experiments/erank-frame-quality-layer-ablation-v1.artifacts.json"
)
DRAFT_ARTIFACT_SHA256 = (
    "6339b9adbf7e0ca0c1ea7b5095839a20125a592ece1957def9c350e76752508f"
)
TARGET_ARTIFACT_RELATIVE_PATH = Path("experiments/insight2-krea14b.artifacts.json")
TARGET_ARTIFACT_SHA256 = (
    "fb9908334d8abb14f25b52811751e4637509781b635322c5c09e00bd82bf43c8"
)
VBENCH_ASSETS_RELATIVE_PATH = Path("experiments/vbench-long-offline-assets-v1.json")
VBENCH_ASSETS_SHA256 = (
    "7393c2611c150733a6b4637774c3b76590e22c097f1e3b41239d45685cbc1831"
)
OURS_BOUNDARY_PROTOCOL_RELATIVE_PATH = Path(
    "experiments/insight2-correction-long60-h800-lookahead-boundary-v1.protocol.json"
)
OURS_BOUNDARY_PROTOCOL_SHA256 = (
    "8c42f4bbbdd54a8e74e09c962899a29ec37be3c603e5d98d3a19136d88dba33b"
)
DUMMY_FORCING_ARTIFACT_RELATIVE_PATH = Path(
    "experiments/main-table-dummy-forcing-v1.artifacts.json"
)
DUMMY_FORCING_ARTIFACT_SHA256 = (
    "02844fe3ee6f45f9528d72b1f311fb5e3503c216b4397a3842ca4a6e6b228d98"
)
LIGHT_FORCING_ARTIFACT_RELATIVE_PATH = Path(
    "experiments/main-table-light-forcing-v1.artifacts.json"
)
LIGHT_FORCING_ARTIFACT_SHA256 = (
    "c04bcf7f8b49a2759c88e0df62e86e14e0874ac11f24a73c567a65263d735692"
)
SDVG_ARTIFACT_RELATIVE_PATH = Path(
    "experiments/main-table-sdvg-v1.artifacts.json"
)
SDVG_ARTIFACT_SHA256 = (
    "008d1fd4b33ad8afaad98cf957bdf49697a44097cfd5ae34cf8bd8d4825f6689"
)
T_STITCH_ARTIFACT_RELATIVE_PATH = Path(
    "experiments/main-table-t-stitch-v1.artifacts.json"
)
T_STITCH_ARTIFACT_SHA256 = (
    "a81ef329630a9066bd7d543c782ea78c7c0d27f35e49a885686f535dca23ce09"
)

# Filled with the digest of the checked-in protocol.  Keeping the digest here,
# in the config, and in the manifest makes silent protocol replacement fail at
# every entry point.
PROTOCOL_SHA256 = "e2dce6e1142dc11e20ce86a773a6d0de37da233506c74bc6180f7f7890eaf2a1"

METHOD_IDS = (
    "sdvg",
    "t_stitch",
    "light_forcing",
    "dummy_forcing",
    "ours",
    "full_target",
)
IMPLEMENTATION_KINDS = {
    "sdvg": "paper_faithful_reproduction",
    "t_stitch": "official_method_adaptation",
    "light_forcing": "official_implementation",
    "dummy_forcing": "official_implementation",
    "ours": "repository_native_implementation",
    "full_target": "repository_native_reference",
}
PROMPT_IDS = (
    "001_subject_consistency_vbl0268",
    "002_subject_consistency_vbl0276",
    "009_background_consistency_vbl0805",
    "010_background_consistency_vbl0819",
    "025_motion_smoothness_vbl0291",
    "026_motion_smoothness_vbl0282",
    "033_dynamic_degree_vbl0309",
    "034_dynamic_degree_vbl0308",
    "041_aesthetic_quality_vbl0746",
    "042_aesthetic_quality_vbl0719",
    "049_imaging_quality_vbl0699",
    "050_imaging_quality_vbl0757",
)
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
DENOSING_TIMESTEPS = (1000, 750, 500, 250)
VBENCH_LONG_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class MainTableContractError(ValueError):
    """Raised when any part of the frozen main-table bundle drifts."""


def _fail(path: str, message: str) -> None:
    raise MainTableContractError(f"{path}: {message}")


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        _fail(path, message)


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MainTableContractError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise MainTableContractError(f"invalid JSON numeric constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MainTableContractError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_invalid_json_constant,
        )
    except MainTableContractError:
        raise
    except json.JSONDecodeError as exc:
        raise MainTableContractError(
            f"invalid JSON in {path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    _require(isinstance(value, dict), "$", "expected object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MainTableContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _object(
    value: Any,
    path: str,
    *,
    required: set[str],
    allowed: set[str] | None = None,
) -> dict[str, Any]:
    _require(isinstance(value, dict), path, "expected object")
    obj = value
    missing = sorted(required - obj.keys())
    if missing:
        _fail(path, f"missing required field(s): {', '.join(missing)}")
    permitted = required if allowed is None else allowed
    unknown = sorted(obj.keys() - permitted)
    if unknown:
        _fail(path, f"unknown field(s): {', '.join(unknown)}")
    return obj


def _string(value: Any, path: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), path, "expected non-empty string")
    return value


def _same_json_value(value: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercions."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _same_json_value(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _same_json_value(item, expected_item)
            for item, expected_item in zip(value, expected)
        )
    return value == expected


def _expect(value: Any, expected: Any, path: str) -> None:
    _require(_same_json_value(value, expected), path, f"must equal {expected!r}")


def _safe_relative_path(value: Any, path: str) -> Path:
    text = _string(value, path)
    components = re.split(r"[\\/]", text)
    _require(
        not text.startswith(("~", "/", "\\\\"))
        and not PurePosixPath(text).is_absolute()
        and not PureWindowsPath(text).is_absolute()
        and ".." not in components,
        path,
        "must be a repository-relative path without '..'",
    )
    return Path(text)


def _verify_file(
    repository_root: Path,
    relative_path: Any,
    expected_path: Path,
    expected_sha256: Any,
    frozen_sha256: str,
    path: str,
) -> None:
    parsed = _safe_relative_path(relative_path, f"{path}.file")
    _expect(parsed.as_posix(), expected_path.as_posix(), f"{path}.file")
    _expect(expected_sha256, frozen_sha256, f"{path}.sha256")
    actual = _sha256_file(repository_root / parsed)
    _expect(actual, frozen_sha256, f"{path}.sha256")


def _validate_exact_object(value: Any, expected: Mapping[str, Any], path: str) -> None:
    obj = _object(value, path, required=set(expected))
    for key, expected_value in expected.items():
        _expect(obj[key], expected_value, f"{path}.{key}")


EXPECTED_PROMPT_POPULATION = {
    "source_file": PROMPT_SOURCE_RELATIVE_PATH.as_posix(),
    "source_sha256": PROMPT_SOURCE_SHA256,
    "selection_rule": "deterministic_first_two_per_primary_dimension_in_frozen_source_order",
    "selection_was_quality_blind": True,
    "prompt_count": 12,
    "prompt_ids": list(PROMPT_IDS),
}

EXPECTED_GENERATION_CONTRACT = {
    "seed": 20260911,
    "width": 832,
    "height": 480,
    "fps": 24,
    "chunk_count": 120,
    "latent_frames_per_chunk": 3,
    "latent_frame_count": 360,
    "rgb_frame_count": 1437,
    "exact_duration_seconds": "479/8",
    "denoising_steps": 4,
    "denoising_timesteps": list(DENOSING_TIMESTEPS),
    "scheduler": "shared_wan_four_step_schedule",
    "same_prompt_seed_chunk_noise_across_methods": True,
}

EXPECTED_QUALITY_CONTRACT = {
    "benchmark": "VBench-Long",
    "implementation": "official_vbench_long",
    "mode": "long_custom_input",
    "source_repository_url": "https://github.com/Vchitect/VBench",
    "source_revision": VBENCH_LONG_REVISION,
    "offline_assets_file": VBENCH_ASSETS_RELATIVE_PATH.as_posix(),
    "offline_assets_sha256": VBENCH_ASSETS_SHA256,
    "dimensions": list(QUALITY_DIMENSIONS),
    "aggregation": "unweighted_arithmetic_mean_of_six_official_dimension_scores",
    "dimension_scores_reported_separately": True,
    "full_video_frame_count": 1437,
    "quality_evaluation_must_not_overlap_generation_timing": True,
}

EXPECTED_COMPARISON_CONTROLS = {
    "same_exact_code_revision": True,
    "same_prompt_order": True,
    "same_seed": True,
    "same_resolution_fps_and_horizon": True,
    "same_four_step_schedule": True,
    "same_official_vbench_long_scorer": True,
    "one_video_per_prompt_method": True,
}

EXPECTED_SPEEDUP_CONTRACT = {
    "reference_method_id": "full_target",
    "per_prompt_formula": "full_target_generation_wall_seconds/method_generation_wall_seconds",
    "reported_aggregate": "geometric_mean_of_paired_per_prompt_speedups",
    "paired_prompt_seed_required": True,
    "primary_latency_field": "generation_wall_seconds",
    "excluded_from_primary_latency": [
        "model_load_seconds",
        "prompt_encode_seconds",
        "vae_decode_seconds",
        "video_encode_seconds",
        "vbench_wall_seconds",
    ],
    "full_target_self_speedup": 1.0,
    "missing_or_nonpositive_latency_policy": "fail_closed",
}

EXPECTED_OFFICIAL_SOURCES = {
    "sdvg": {
        "source_type": "paper",
        "url": "https://arxiv.org/abs/2604.17397",
        "revision": "2604.17397v1",
        "code_release_status": "no_official_code_release_located",
        "official_video_implementation": False,
    },
    "t_stitch": {
        "source_type": "git_repository",
        "url": "https://github.com/NVlabs/T-Stitch",
        "revision": "264b60b68d013031c24f51ea3baeffd8641a87de",
        "code_release_status": "official_repository_available",
        "official_video_implementation": False,
    },
    "light_forcing": {
        "source_type": "git_repository",
        "url": "https://github.com/chengtao-lv/LightForcing",
        "revision": "d1e633337fff7e8a0ebf9f643dc37cd481fb299d",
        "code_release_status": "official_repository_available",
        "official_video_implementation": True,
    },
    "dummy_forcing": {
        "source_type": "git_repository",
        "url": "https://github.com/csguoh/DummyForcing",
        "revision": "28128ed6204dcb4c572bf8719955124a5f2dbe20",
        "code_release_status": "official_repository_available",
        "official_video_implementation": True,
    },
    "ours": {
        "source_type": "repository_native",
        "url": "https://github.com/langzirui3-sys/collaborative-forcing-release",
        "revision": "78b55f5529b65cf6062b00c2dd4fc71d4d1f318f",
        "code_release_status": "repository_native_reference_revision",
        "official_video_implementation": True,
    },
    "full_target": {
        "source_type": "repository_native",
        "url": "https://github.com/langzirui3-sys/collaborative-forcing-release",
        "revision": "78b55f5529b65cf6062b00c2dd4fc71d4d1f318f",
        "code_release_status": "repository_native_reference_revision",
        "official_video_implementation": True,
    },
}

EXPECTED_ARTIFACT_REQUIREMENTS = {
    "sdvg": {
        "required_artifact_ids": [
            "draft_1_3b",
            "target_14b",
            "image_reward_checkpoint",
            "sdvg_reproduction_source",
            "vbench_long_offline_assets",
        ],
        "bound_artifact_spec_ids": [
            "draft_1_3b",
            "target_14b",
            "sdvg_artifacts",
            "vbench_long_offline_assets",
        ],
        "unresolved_artifact_ids": [],
        "all_required_artifacts_bound": True,
    },
    "t_stitch": {
        "required_artifact_ids": [
            "draft_1_3b",
            "target_14b",
            "t_stitch_source_archive",
            "vbench_long_offline_assets",
        ],
        "bound_artifact_spec_ids": [
            "draft_1_3b",
            "target_14b",
            "t_stitch_artifacts",
            "vbench_long_offline_assets",
        ],
        "unresolved_artifact_ids": [],
        "all_required_artifacts_bound": True,
    },
    "light_forcing": {
        "required_artifact_ids": [
            "light_forcing_generator_artifacts",
            "light_forcing_source_archive",
            "vbench_long_offline_assets",
        ],
        "bound_artifact_spec_ids": [
            "light_forcing_artifacts",
            "vbench_long_offline_assets",
        ],
        "unresolved_artifact_ids": [],
        "all_required_artifacts_bound": True,
    },
    "dummy_forcing": {
        "required_artifact_ids": [
            "dummy_forcing_generator_artifacts",
            "dummy_forcing_source_archive",
            "vbench_long_offline_assets",
        ],
        "bound_artifact_spec_ids": [
            "dummy_forcing_artifacts",
            "vbench_long_offline_assets",
        ],
        "unresolved_artifact_ids": [],
        "all_required_artifacts_bound": True,
    },
    "ours": {
        "required_artifact_ids": [
            "draft_1_3b",
            "target_14b",
            "ours_boundary_protocol",
            "vbench_long_offline_assets",
        ],
        "bound_artifact_spec_ids": [
            "draft_1_3b",
            "target_14b",
            "ours_boundary_protocol",
            "vbench_long_offline_assets",
        ],
        "unresolved_artifact_ids": [],
        "all_required_artifacts_bound": True,
    },
    "full_target": {
        "required_artifact_ids": [
            "draft_1_3b",
            "target_14b",
            "vbench_long_offline_assets",
        ],
        "bound_artifact_spec_ids": [
            "draft_1_3b",
            "target_14b",
            "vbench_long_offline_assets",
        ],
        "unresolved_artifact_ids": [],
        "all_required_artifacts_bound": True,
    },
}

EXPECTED_DUAL_MODEL_GPU_ROLES = {
    "required_visible_gpu_count": 2,
    "gpu_0_generation_role": "wan_1_3b_draft",
    "gpu_1_generation_role": "krea_14b_target",
    "gpu_1_generation_timing_policy": "active_target_generator",
    "quality_evaluation_gpu_role": "gpu_1_after_generation_timer_closed",
    "quality_evaluation_must_not_overlap_generation_timing": True,
}

EXPECTED_TWO_GPU_ROLES = {
    "sdvg": EXPECTED_DUAL_MODEL_GPU_ROLES,
    "t_stitch": EXPECTED_DUAL_MODEL_GPU_ROLES,
    "light_forcing": {
        "required_visible_gpu_count": 2,
        "gpu_0_generation_role": "light_forcing_generator",
        "gpu_1_generation_role": "reserved_idle",
        "gpu_1_generation_timing_policy": "idle_for_entire_generation_interval",
        "quality_evaluation_gpu_role": "gpu_1_after_generation_timer_closed",
        "quality_evaluation_must_not_overlap_generation_timing": True,
    },
    "dummy_forcing": {
        "required_visible_gpu_count": 2,
        "gpu_0_generation_role": "dummy_forcing_generator",
        "gpu_1_generation_role": "reserved_idle",
        "gpu_1_generation_timing_policy": "idle_for_entire_generation_interval",
        "quality_evaluation_gpu_role": "gpu_1_after_generation_timer_closed",
        "quality_evaluation_must_not_overlap_generation_timing": True,
    },
    "ours": EXPECTED_DUAL_MODEL_GPU_ROLES,
    "full_target": {
        "required_visible_gpu_count": 2,
        "gpu_0_generation_role": "wan_1_3b_offline_vae_decoder_idle_during_generation",
        "gpu_1_generation_role": "krea_14b_target",
        "gpu_1_generation_timing_policy": "active_target_generator",
        "quality_evaluation_gpu_role": "gpu_1_after_generation_timer_closed",
        "quality_evaluation_must_not_overlap_generation_timing": True,
    },
}

EXPECTED_METHOD_CONTRACTS = {
    "sdvg": {
        "officiality_label": "unofficial_paper_faithful_reproduction",
        "draft_model_role": "wan_1_3b_draft",
        "target_model_role": "krea_14b_target",
        "acceptance_gate": "image_reward_accept_draft_else_target_regenerate",
        "target_kv_maintenance": "maintain_target_kv_for_accepted_draft_and_target_chunks",
        "benchmark_adaptations": [
            "paper_schedule_replaced_by_shared_four_step_schedule",
            "paper_horizon_replaced_by_shared_120_chunk_horizon",
            "paper_metrics_replaced_by_shared_official_vbench_long_six_dimension_macro_mean",
        ],
    },
    "t_stitch": {
        "adaptation_label": "official_image_method_adapted_to_wan_video",
        "switch_unit": "denoising_step_within_each_chunk",
        "chunk_step_assignment": [
            {"step_index": 0, "timestep": 1000, "model_role": "wan_1_3b_draft"},
            {"step_index": 1, "timestep": 750, "model_role": "krea_14b_target"},
            {"step_index": 2, "timestep": 500, "model_role": "krea_14b_target"},
            {"step_index": 3, "timestep": 250, "model_role": "krea_14b_target"},
        ],
        "draft_step_count_per_chunk": 1,
        "target_step_count_per_chunk": 3,
        "same_latent_shape_for_both_models": True,
        "accept_reject_gate": "forbidden",
    },
    "light_forcing": {
        "generation_mode": "official_chunk_aware_sparse_attention",
        "hierarchical_historical_local_selection": True,
        "hardware_profile": "official_hopper_recommendation",
        "sparsity": 0.85,
        "sparsity_base": 0.95,
        "keep_frames": 3,
        "keep_sink": 1,
        "keep_near": 1,
        "fp8_enabled": False,
        "lightvae_enabled": False,
        "teacache_enabled": False,
        "additional_unreported_acceleration": "forbidden",
    },
    "dummy_forcing": {
        "generation_mode": "official_dummy_forcing",
        "kv_cache_enabled": True,
        "online_dummy_head_classification": True,
        "heterogeneous_memory_allocation": True,
        "teacache_enabled": False,
        "additional_unreported_acceleration": "forbidden",
    },
    "ours": {
        "reference_protocol_file": OURS_BOUNDARY_PROTOCOL_RELATIVE_PATH.as_posix(),
        "reference_protocol_sha256": OURS_BOUNDARY_PROTOCOL_SHA256,
        "strategy": "online_lookahead_coverage_boundary_gated",
        "target_proposal_opportunity_budget": 6,
        "preview_distance_chunks": 1,
        "target_acceptance_rule": "successor_erank_and_boundary_paired_dominance",
        "same_noise_pairing_required": True,
        "target_cache_rebuild": "fresh_target_session_from_full_latent_prefix",
    },
    "full_target": {
        "reference_role": "quality_and_speedup_denominator",
        "generation_mode": "all_chunks_all_steps_krea_14b_target",
        "target_model_role": "krea_14b_target",
        "native_session_policy": "one_persistent_krea_generation_session_per_prompt",
        "history_transport_during_generation": "forbidden",
        "interleaved_real_vae_decode": "forbidden",
        "offline_decoder_role": "wan_1_3b_vae_after_generation_timer_closed",
        "chunk_count": 120,
        "denoising_timesteps": list(DENOSING_TIMESTEPS),
        "draft_model_calls": 0,
        "accept_reject_gate": "forbidden",
    },
}

EXPECTED_OUTPUT_CONTRACT = {
    "required_artifacts": [
        "full_video_mp4",
        "video_metadata_json",
        "video_sha256",
        "per_chunk_trace_jsonl",
        "method_provenance_json",
        "official_vbench_long_measurement_evidence",
        "six_dimension_scores_json",
        "six_dimension_macro_mean_json",
        "timing_events_jsonl",
        "timing_summary_json",
    ],
    "video_container": "mp4",
    "video_frame_count": 1437,
    "video_fps": 24,
    "video_exact_duration_seconds": "479/8",
    "dimension_score_order": list(QUALITY_DIMENSIONS),
    "macro_mean": "unweighted_arithmetic_mean_in_dimension_score_order",
}

COMMON_TIMING_FIELDS = [
    "model_load_seconds",
    "prompt_encode_seconds",
    "generation_wall_seconds",
    "generation_cuda_seconds",
    "per_chunk_total_seconds",
    "vae_decode_seconds",
    "video_encode_seconds",
    "vbench_wall_seconds",
    "peak_gpu_memory_bytes_by_role",
]
METHOD_TIMING_FIELDS = {
    "sdvg": [
        "draft_generation_seconds",
        "reward_scoring_seconds",
        "target_kv_append_seconds",
        "target_regeneration_seconds",
    ],
    "t_stitch": ["draft_step_0_seconds", "target_steps_1_through_3_seconds"],
    "light_forcing": ["sparse_mask_selection_seconds", "sparse_attention_seconds"],
    "dummy_forcing": [
        "dummy_head_classification_seconds",
        "kv_cache_update_seconds",
        "heterogeneous_context_packing_seconds",
    ],
    "ours": [
        "draft_generation_seconds",
        "lookahead_preview_seconds",
        "target_proposal_seconds",
        "boundary_witness_seconds",
        "target_cache_rebuild_seconds",
    ],
    "full_target": ["target_generation_seconds"],
}


def _expected_timing_contract(method_id: str) -> dict[str, Any]:
    return {
        "required_summary_fields": COMMON_TIMING_FIELDS,
        "required_method_stage_fields": METHOD_TIMING_FIELDS[method_id],
        "primary_latency_field": "generation_wall_seconds",
        "primary_latency_excludes": [
            "model_load_seconds",
            "prompt_encode_seconds",
            "vae_decode_seconds",
            "video_encode_seconds",
            "vbench_wall_seconds",
        ],
        "cuda_timing": "synchronize_before_start_and_after_stop",
        "wall_timing": "synchronize_all_generation_gpus_before_start_and_after_stop",
        "quality_scorer_overlap_with_generation": "forbidden",
    }


EXPECTED_BACKEND_STATUS = {
    "sdvg": "implemented_awaiting_gpu_smoke",
    "t_stitch": "implemented_awaiting_gpu_smoke",
    "light_forcing": "implemented_awaiting_gpu_smoke",
    "dummy_forcing": "implemented_awaiting_gpu_smoke",
    "ours": "implemented_awaiting_gpu_smoke",
    "full_target": "implemented_awaiting_gpu_smoke",
}

EXPECTED_BACKEND_MODULE = {
    "sdvg": "cfexp.main_table_sdvg_campaign",
    "t_stitch": "cfexp.main_table_t_stitch_campaign",
    "light_forcing": "cfexp.light_forcing_adapter",
    "dummy_forcing": "cfexp.dummy_forcing_adapter",
    "ours": "cfexp.main_table_ours_campaign",
    "full_target": "cfexp.main_table_full_target_campaign",
}

EXPECTED_ARTIFACT_SPECS = {
    "draft_1_3b": {
        "file": DRAFT_ARTIFACT_RELATIVE_PATH.as_posix(),
        "sha256": DRAFT_ARTIFACT_SHA256,
    },
    "target_14b": {
        "file": TARGET_ARTIFACT_RELATIVE_PATH.as_posix(),
        "sha256": TARGET_ARTIFACT_SHA256,
    },
    "vbench_long_offline_assets": {
        "file": VBENCH_ASSETS_RELATIVE_PATH.as_posix(),
        "sha256": VBENCH_ASSETS_SHA256,
    },
    "ours_boundary_protocol": {
        "file": OURS_BOUNDARY_PROTOCOL_RELATIVE_PATH.as_posix(),
        "sha256": OURS_BOUNDARY_PROTOCOL_SHA256,
    },
    "sdvg_artifacts": {
        "file": SDVG_ARTIFACT_RELATIVE_PATH.as_posix(),
        "sha256": SDVG_ARTIFACT_SHA256,
    },
    "t_stitch_artifacts": {
        "file": T_STITCH_ARTIFACT_RELATIVE_PATH.as_posix(),
        "sha256": T_STITCH_ARTIFACT_SHA256,
    },
    "dummy_forcing_artifacts": {
        "file": DUMMY_FORCING_ARTIFACT_RELATIVE_PATH.as_posix(),
        "sha256": DUMMY_FORCING_ARTIFACT_SHA256,
    },
    "light_forcing_artifacts": {
        "file": LIGHT_FORCING_ARTIFACT_RELATIVE_PATH.as_posix(),
        "sha256": LIGHT_FORCING_ARTIFACT_SHA256,
    },
}

EXPECTED_EXECUTION_GATE = {
    "status": "blocked",
    "reasons": [
        "sdvg_backend_awaiting_gpu_smoke",
        "light_forcing_backend_awaiting_gpu_smoke",
        "dummy_forcing_backend_awaiting_gpu_smoke",
        "ours_backend_awaiting_gpu_smoke",
        "full_target_backend_awaiting_gpu_smoke",
    ],
    "required_for_runnable": [
        "all_method_backend_modules_bound_to_reviewed_adapters",
        "all_method_backend_statuses_implemented",
        "all_required_artifact_bindings_digest_pinned",
        "manifest_code_revision_is_the_published_full_commit",
        "frozen_contract_tests_pass",
    ],
}


def validate_protocol(value: Any, *, repository_root: Path | str = REPOSITORY_ROOT) -> None:
    """Validate the frozen benchmark population and shared comparison rules."""

    root = _object(
        value,
        "$",
        required={
            "schema",
            "protocol_id",
            "status",
            "frozen_date",
            "claim_scope",
            "prompt_population",
            "generation_contract",
            "quality_contract",
            "method_order",
            "speedup_contract",
            "comparison_controls",
        },
    )
    _expect(root["schema"], PROTOCOL_SCHEMA, "$.schema")
    _expect(root["protocol_id"], PROTOCOL_ID, "$.protocol_id")
    _expect(root["status"], "frozen", "$.status")
    _expect(root["frozen_date"], "2026-10-21", "$.frozen_date")
    _string(root["claim_scope"], "$.claim_scope")
    _validate_exact_object(
        root["prompt_population"], EXPECTED_PROMPT_POPULATION, "$.prompt_population"
    )
    _validate_exact_object(
        root["generation_contract"], EXPECTED_GENERATION_CONTRACT, "$.generation_contract"
    )
    _validate_exact_object(
        root["quality_contract"], EXPECTED_QUALITY_CONTRACT, "$.quality_contract"
    )
    _expect(root["method_order"], list(METHOD_IDS), "$.method_order")
    _validate_exact_object(
        root["speedup_contract"], EXPECTED_SPEEDUP_CONTRACT, "$.speedup_contract"
    )
    _validate_exact_object(
        root["comparison_controls"],
        EXPECTED_COMPARISON_CONTROLS,
        "$.comparison_controls",
    )

    repo = Path(repository_root).resolve()
    source_path = repo / PROMPT_SOURCE_RELATIVE_PATH
    _expect(_sha256_file(source_path), PROMPT_SOURCE_SHA256, "$.prompt_population.source_sha256")
    source = _load_json(source_path)
    prompts = source.get("prompts")
    _require(isinstance(prompts, list), "$.prompt_population.source_file", "missing prompts")
    source_ids: list[Any] = []
    for dimension in QUALITY_DIMENSIONS:
        matches = [
            item.get("prompt_id")
            for item in prompts
            if isinstance(item, dict) and item.get("assigned_dimension") == dimension
        ]
        _require(
            len(matches) >= 2,
            "$.prompt_population.source_file",
            f"fewer than two prompts for dimension {dimension!r}",
        )
        source_ids.extend(matches[:2])
    _expect(source_ids, list(PROMPT_IDS), "$.prompt_population.prompt_ids")
    _expect(_sha256_file(repo / VBENCH_ASSETS_RELATIVE_PATH), VBENCH_ASSETS_SHA256, "$.quality_contract.offline_assets_sha256")


def load_protocol(
    path: Path | str = REPOSITORY_ROOT / PROTOCOL_RELATIVE_PATH,
    *,
    repository_root: Path | str = REPOSITORY_ROOT,
) -> dict[str, Any]:
    value = _load_json(Path(path))
    validate_protocol(value, repository_root=repository_root)
    return value


def validate_config(
    value: Any,
    *,
    repository_root: Path | str = REPOSITORY_ROOT,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    """Validate all six method contracts and the fail-closed execution gate."""

    root = _object(
        value,
        "$",
        required={
            "schema",
            "config_id",
            "status",
            "protocol",
            "manifest_file",
            "artifact_specs",
            "methods",
            "execution_gate",
        },
    )
    _expect(root["schema"], CONFIG_SCHEMA, "$.schema")
    _expect(root["config_id"], CONFIG_ID, "$.config_id")
    _expect(root["status"], "frozen", "$.status")
    _validate_exact_object(
        root["protocol"],
        {"file": PROTOCOL_RELATIVE_PATH.as_posix(), "sha256": PROTOCOL_SHA256},
        "$.protocol",
    )
    _expect(root["manifest_file"], MANIFEST_RELATIVE_PATH.as_posix(), "$.manifest_file")

    artifact_specs = _object(
        root["artifact_specs"],
        "$.artifact_specs",
        required=set(EXPECTED_ARTIFACT_SPECS),
    )
    repo = Path(repository_root).resolve()
    artifact_paths = {
        "draft_1_3b": (DRAFT_ARTIFACT_RELATIVE_PATH, DRAFT_ARTIFACT_SHA256),
        "target_14b": (TARGET_ARTIFACT_RELATIVE_PATH, TARGET_ARTIFACT_SHA256),
        "vbench_long_offline_assets": (VBENCH_ASSETS_RELATIVE_PATH, VBENCH_ASSETS_SHA256),
        "ours_boundary_protocol": (
            OURS_BOUNDARY_PROTOCOL_RELATIVE_PATH,
            OURS_BOUNDARY_PROTOCOL_SHA256,
        ),
        "dummy_forcing_artifacts": (
            DUMMY_FORCING_ARTIFACT_RELATIVE_PATH,
            DUMMY_FORCING_ARTIFACT_SHA256,
        ),
        "light_forcing_artifacts": (
            LIGHT_FORCING_ARTIFACT_RELATIVE_PATH,
            LIGHT_FORCING_ARTIFACT_SHA256,
        ),
    }
    for artifact_id, (relative_path, digest) in artifact_paths.items():
        spec = _object(
            artifact_specs[artifact_id],
            f"$.artifact_specs.{artifact_id}",
            required={"file", "sha256"},
        )
        _verify_file(
            repo,
            spec["file"],
            relative_path,
            spec["sha256"],
            digest,
            f"$.artifact_specs.{artifact_id}",
        )

    methods = root["methods"]
    _require(isinstance(methods, list), "$.methods", "expected array")
    _expect(len(methods), len(METHOD_IDS), "$.methods")
    observed_ids: list[str] = []
    method_keys = {
        "method_id",
        "display_name",
        "implementation_kind",
        "official_source",
        "artifact_requirements",
        "two_gpu_roles",
        "method_contract",
        "output_contract",
        "timing_contract",
        "backend_module",
        "backend_status",
    }
    for index, method_id in enumerate(METHOD_IDS):
        path = f"$.methods[{index}]"
        method = _object(methods[index], path, required=method_keys)
        _expect(method["method_id"], method_id, f"{path}.method_id")
        observed_ids.append(method["method_id"])
        _string(method["display_name"], f"{path}.display_name")
        _expect(
            method["implementation_kind"],
            IMPLEMENTATION_KINDS[method_id],
            f"{path}.implementation_kind",
        )
        _validate_exact_object(
            method["official_source"],
            EXPECTED_OFFICIAL_SOURCES[method_id],
            f"{path}.official_source",
        )
        _validate_exact_object(
            method["artifact_requirements"],
            EXPECTED_ARTIFACT_REQUIREMENTS[method_id],
            f"{path}.artifact_requirements",
        )
        _validate_exact_object(
            method["two_gpu_roles"],
            EXPECTED_TWO_GPU_ROLES[method_id],
            f"{path}.two_gpu_roles",
        )
        _validate_exact_object(
            method["method_contract"],
            EXPECTED_METHOD_CONTRACTS[method_id],
            f"{path}.method_contract",
        )
        _validate_exact_object(
            method["output_contract"],
            EXPECTED_OUTPUT_CONTRACT,
            f"{path}.output_contract",
        )
        _validate_exact_object(
            method["timing_contract"],
            _expected_timing_contract(method_id),
            f"{path}.timing_contract",
        )
        _expect(
            method["backend_module"],
            EXPECTED_BACKEND_MODULE[method_id],
            f"{path}.backend_module",
        )
        _expect(
            method["backend_status"],
            EXPECTED_BACKEND_STATUS[method_id],
            f"{path}.backend_status",
        )
    _expect(observed_ids, list(METHOD_IDS), "$.methods")
    _require(len(set(observed_ids)) == len(observed_ids), "$.methods", "duplicate method_id")
    _validate_exact_object(root["execution_gate"], EXPECTED_EXECUTION_GATE, "$.execution_gate")

    protocol_path = repo / PROTOCOL_RELATIVE_PATH
    _expect(_sha256_file(protocol_path), PROTOCOL_SHA256, "$.protocol.sha256")
    resolved_protocol = protocol if protocol is not None else load_protocol(
        protocol_path, repository_root=repo
    )
    _expect(resolved_protocol.get("protocol_id"), PROTOCOL_ID, "$.protocol")
    _expect(resolved_protocol.get("method_order"), list(METHOD_IDS), "$.methods")


def load_config(
    path: Path | str = REPOSITORY_ROOT / CONFIG_RELATIVE_PATH,
    *,
    repository_root: Path | str = REPOSITORY_ROOT,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = _load_json(Path(path))
    validate_config(value, repository_root=repository_root, protocol=protocol)
    return value


EXPECTED_MANIFEST_TAGS = {
    "main-table-v1",
    "six-method",
    "long-video",
    "vbench-long",
    "blocked",
}
EXPECTED_MANIFEST_OUTPUTS = {
    "include": [
        "attempts/**/videos/**/*.mp4",
        "attempts/**/metadata/**/*.json",
        "attempts/**/traces/**/*.jsonl",
        "attempts/**/provenance/**/*.json",
        "attempts/**/timing/**/*.json",
        "attempts/**/timing/**/*.jsonl",
        "attempts/**/vbench-long/**/*.json",
        "attempts/**/result.json",
        "run.json",
        "resolved-manifest.json",
    ],
    "exclude": [
        "attempts/**/tmp/**",
        "attempts/**/vbench-long/**/videos/**",
        "**/*.tmp",
        "**/*.part",
        "**/__pycache__/**",
    ],
}


def _expected_manifest_inputs(repository_root: Path) -> list[dict[str, Any]]:
    paths = (
        ("main_table_protocol", PROTOCOL_RELATIVE_PATH),
        ("main_table_config", CONFIG_RELATIVE_PATH),
        ("draft_artifact_spec", DRAFT_ARTIFACT_RELATIVE_PATH),
        ("target_artifact_spec", TARGET_ARTIFACT_RELATIVE_PATH),
        ("vbench_long_offline_assets", VBENCH_ASSETS_RELATIVE_PATH),
        ("ours_boundary_protocol", OURS_BOUNDARY_PROTOCOL_RELATIVE_PATH),
        ("sdvg_artifact_spec", SDVG_ARTIFACT_RELATIVE_PATH),
        ("t_stitch_artifact_spec", T_STITCH_ARTIFACT_RELATIVE_PATH),
        ("dummy_forcing_artifact_spec", DUMMY_FORCING_ARTIFACT_RELATIVE_PATH),
        ("light_forcing_artifact_spec", LIGHT_FORCING_ARTIFACT_RELATIVE_PATH),
    )
    return [
        {
            "name": name,
            "uri": f"repository://{relative_path.as_posix()}",
            "target": relative_path.as_posix(),
            "sha256": _sha256_file(repository_root / relative_path),
        }
        for name, relative_path in paths
    ]


def validate_manifest_binding(
    value: Mapping[str, Any],
    *,
    repository_root: Path | str = REPOSITORY_ROOT,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Validate the generic manifest and its exact binding to this bundle."""

    try:
        validate_manifest(value)
    except ManifestError as exc:
        raise MainTableContractError(str(exc)) from exc
    repo = Path(repository_root).resolve()
    metadata = value["metadata"]
    _expect(metadata["name"], "main-table-v1", "$.metadata.name")
    _expect(set(metadata.get("tags", [])), EXPECTED_MANIFEST_TAGS, "$.metadata.tags")
    spec = value["spec"]
    _expect(spec["code"], {"revision": "HEAD", "require_clean": True}, "$.spec.code")
    _expect(
        spec["runtime"],
        {"backend": "uv", "python": "3.11", "lockfile": "uv.lock"},
        "$.spec.runtime",
    )
    _expect(
        spec["command"],
        {
            "argv": [
                "python",
                "-m",
                "cfexp.main_table_v1",
                "--config",
                CONFIG_RELATIVE_PATH.as_posix(),
                "--require-runnable",
            ],
            "cwd": ".",
            "env": {"PYTHONUNBUFFERED": "1"},
            "secret_env": [],
        },
        "$.spec.command",
    )
    _expect(
        spec["resources"],
        {
            "gpus": 2,
            "min_gpu_memory_gb": 75,
            "cpu": 16,
            "memory_gb": 192,
            "expected_duration_minutes": 10080,
        },
        "$.spec.resources",
    )
    _expect(spec.get("inputs"), _expected_manifest_inputs(repo), "$.spec.inputs")
    _expect(
        spec.get("parameters"),
        {
            "protocol_id": PROTOCOL_ID,
            "method_order": list(METHOD_IDS),
            "prompt_count": 12,
            "seed": 20260911,
            "chunk_count": 120,
            "frame_count": 1437,
            "fps": 24,
            "execution_gate_status": "blocked",
        },
        "$.spec.parameters",
    )
    _expect(spec["outputs"], EXPECTED_MANIFEST_OUTPUTS, "$.spec.outputs")
    _expect(spec.get("checkpoint"), {"enabled": False}, "$.spec.checkpoint")
    _expect(
        spec["storage"],
        {
            "project": "collaborative-forcing",
            "remote": "${CFEXP_ARTIFACT_REMOTE}",
        },
        "$.spec.storage",
    )
    _expect(spec["policy"], {}, "$.spec.policy")
    resolved_config = config if config is not None else load_config(
        repo / CONFIG_RELATIVE_PATH, repository_root=repo
    )
    _expect(
        resolved_config["execution_gate"]["status"],
        spec["parameters"]["execution_gate_status"],
        "$.spec.parameters.execution_gate_status",
    )


@dataclass(frozen=True)
class MainTableBundle:
    repository_root: Path
    protocol: dict[str, Any]
    config: dict[str, Any]
    manifest: dict[str, Any]

    def assert_runnable(self) -> None:
        assert_runnable(self)


def _discover_repository_root(path: Path) -> Path:
    start = path.resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "experiments").is_dir():
            return candidate
    raise MainTableContractError(f"cannot discover repository root from {path}")


def load_bundle(
    repository_root: Path | str | None = None,
    *,
    protocol_path: Path | str | None = None,
    config_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> MainTableBundle:
    """Load and cross-check the complete protocol/config/manifest bundle."""

    if repository_root is None:
        hint = Path(config_path) if config_path is not None else Path.cwd()
        repository_root = _discover_repository_root(hint)
    repo = Path(repository_root).resolve()
    protocol_file = Path(protocol_path) if protocol_path is not None else repo / PROTOCOL_RELATIVE_PATH
    config_file = Path(config_path) if config_path is not None else repo / CONFIG_RELATIVE_PATH
    manifest_file = Path(manifest_path) if manifest_path is not None else repo / MANIFEST_RELATIVE_PATH
    protocol = load_protocol(protocol_file, repository_root=repo)
    config = load_config(config_file, repository_root=repo, protocol=protocol)
    try:
        manifest = load_manifest(manifest_file, expand_environment=False)
    except ManifestError as exc:
        raise MainTableContractError(str(exc)) from exc
    validate_manifest_binding(manifest, repository_root=repo, config=config)
    return MainTableBundle(repo, protocol, config, manifest)


def assert_runnable(value: MainTableBundle | Mapping[str, Any]) -> None:
    """Require an executable bundle; the checked-in contract intentionally fails."""

    config = value.config if isinstance(value, MainTableBundle) else value
    gate = config.get("execution_gate") if isinstance(config, Mapping) else None
    if not isinstance(gate, Mapping):
        _fail("$.execution_gate", "missing execution gate")
    status = gate.get("status")
    reasons = gate.get("reasons")
    if status != "runnable":
        detail = ", ".join(str(item) for item in reasons) if isinstance(reasons, list) else "unspecified"
        _fail("$.execution_gate.status", f"bundle is not runnable ({status!r}): {detail}")
    methods = config.get("methods")
    _require(isinstance(methods, list), "$.methods", "expected array")
    for index, method in enumerate(methods):
        _require(isinstance(method, Mapping), f"$.methods[{index}]", "expected object")
        _require(
            isinstance(method.get("backend_module"), str) and bool(method["backend_module"].strip()),
            f"$.methods[{index}].backend_module",
            "must bind a reviewed backend module",
        )
        _expect(method.get("backend_status"), "implemented", f"$.methods[{index}].backend_status")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=CONFIG_RELATIVE_PATH.as_posix(),
        help="path to the main-table-v1 config",
    )
    parser.add_argument(
        "--require-runnable",
        action="store_true",
        help="fail unless all backends and artifact bindings are executable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config_path = Path(args.config)
        repo = _discover_repository_root(config_path)
        bundle = load_bundle(repo, config_path=config_path)
        if args.require_runnable:
            bundle.assert_runnable()
    except MainTableContractError as exc:
        print(f"main-table-v1 contract error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "protocol_id": bundle.protocol["protocol_id"],
                "method_order": bundle.protocol["method_order"],
                "execution_gate_status": bundle.config["execution_gate"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests.
    raise SystemExit(main())


__all__ = [
    "CONFIG_ID",
    "CONFIG_SCHEMA",
    "IMPLEMENTATION_KINDS",
    "METHOD_IDS",
    "MainTableBundle",
    "MainTableContractError",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "PROMPT_IDS",
    "QUALITY_DIMENSIONS",
    "assert_runnable",
    "load_bundle",
    "load_config",
    "load_protocol",
    "main",
    "validate_config",
    "validate_manifest_binding",
    "validate_protocol",
]
