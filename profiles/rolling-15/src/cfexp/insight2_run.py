"""Formal command-line entry point for the frozen Insight 2 experiments.

This module owns only the experiment boundary: configuration validation,
artifact preflight, phase selection, durable exports, statistics, and audit
summaries.  Model loading and JSONL worker transport live in
``cfexp.correction_runtime``; trajectory semantics live in
``cfexp.insight2_campaign``.  Keeping those concerns separate makes this
entry point small enough to validate without importing Torch on a CPU host.

No placeholder metrics are ever emitted.  A failed preflight or runtime
leaves a ``logs/failure.json`` record and returns a non-zero status.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

from .calibration_receipt import (
    CALIBRATION_RECEIPT_RELATIVE,
    CalibrationReceiptError,
    VerifiedCalibrationReceipt,
    restore_calibration_receipt,
    seal_calibration_receipt,
    verify_calibration_receipt,
)
from .correction_artifacts import (
    TARGET_BACKEND_MODULE,
    target_artifact_spec_sha256,
    verify_target_artifact_spec,
)
from .correction_policy import (
    BoundarySmoothnessGate,
    CoverageConstrainedOnlineRankGuidedPolicy,
    LookaheadCoverageConstrainedOnlineRankGuidedPolicy,
    OnlineRankGuidedPolicy,
    RankRiskCalibrator,
    SuddenDropRiskCalibrator,
    causal_erank_sudden_drop,
)
from .correction_state import (
    CorrectionStateError,
    GenerationContext,
    InitialNoiseDescriptor,
    KVCacheDescriptor,
    LatentHistory,
    LatentSegment,
)
from .erank_frame_campaign import verify_artifact_spec
from .insight2_analysis import (
    analyze_insight2a,
    analyze_insight2b,
    load_insight2a_rows,
    load_insight2b_rows,
    render_figure2a,
    render_figure2b,
    write_json as write_analysis_json,
    write_top1_csv,
)
from .insight2_campaign import (
    BOUNDARY_VELOCITY_METRIC_ID,
    BOUNDARY_VELOCITY_ROUND_DECIMALS,
    BOUNDARY_VELOCITY_WITNESS_SCHEMA,
    CALIBRATION_SCHEMA,
    CAMPAIGN_SCHEMA,
    CHECKPOINT_SCHEMA,
    FORK_EVIDENCE_SCHEMA,
    NOISE_NAMESPACE,
    RAW_2A_SCHEMA,
    RAW_2B_SCHEMA,
    ROLLOUT_SCORE_SCHEMA,
    TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS,
    TARGET_RPC_DIAGNOSTICS_SCHEMA,
    TARGET_RPC_PHASES,
    TARGET_RPC_SCHEMA,
    CampaignError,
    Insight2Campaign,
    PendingMultiBudgetPlan,
    PromptCase,
    QualityMeasurement,
    load_target_rpc_lifecycle,
)
from .insight2_protocol import ResolvedInsight2Protocol, load_protocol
from .krea_chunk_backend import MAIN_TABLE_FOUR_STEP_SCHEDULE, validate_target_forward_trace
from .metadata import collect_git_metadata
from .self_forcing_frame_adapter import (
    AdapterError,
    SELF_FORCING_CROSS_KV_SHAPE,
    SELF_FORCING_KV_DTYPE,
    SELF_FORCING_KV_LAYER_COUNT,
    SELF_FORCING_SELF_KV_SHAPE,
    SELF_FORCING_TOKENS_PER_CHUNK,
    validate_effective_rank_evidence,
    validate_sampled_kv_cache_witness,
    validate_self_forcing_forward_trace,
)
from .vbench_long import (
    VBENCH_DISTRIBUTED_ENVIRONMENT,
    BATCH_CONFIG_SCHEMA as VBENCH_LONG_BATCH_CONFIG_SCHEMA,
    BATCH_MEASUREMENT_SCHEMA as VBENCH_LONG_BATCH_MEASUREMENT_SCHEMA,
    BATCH_ROOT_RELATIVE as VBENCH_LONG_BATCH_ROOT_RELATIVE,
    CHECKOUT_TREE_HASH_ALGORITHM as VBENCH_LONG_CHECKOUT_TREE_HASH_ALGORITHM,
    CONFIG_SCHEMA as VBENCH_LONG_CONFIG_SCHEMA,
    MEASUREMENT_SCHEMA as VBENCH_LONG_OFFICIAL_MEASUREMENT_SCHEMA,
    OFFLINE_ASSET_SCHEMA as VBENCH_LONG_OFFLINE_ASSET_SCHEMA,
    OFFLINE_NETWORK_POLICY as VBENCH_LONG_OFFLINE_NETWORK_POLICY,
    OFFLINE_TREE_HASH_ALGORITHM as VBENCH_LONG_OFFLINE_TREE_HASH_ALGORITHM,
    STAGED_ALIAS_ALGORITHM as VBENCH_LONG_STAGED_ALIAS_ALGORITHM,
    VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR,
    VBENCH_LONG_COMMIT,
    VBENCH_LONG_DIMENSIONS,
    VBENCH_LONG_OFFLINE_FILE_PATHS,
    VBENCH_LONG_OFFLINE_TREE_PATHS,
    VBENCH_LONG_SCRUBBED_ENVIRONMENT_VARIABLES,
    VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR,
    vbench_long_staged_alias,
    validate_vbench_long_batch_measurement,
    validate_vbench_long_checkout,
    validate_vbench_long_offline_assets,
    validate_vbench_long_official_result,
    validate_vbench_long_success_attempt,
)


RUN_CONFIG_SCHEMA = "insight2_run_config_v1"
LONG_VIDEO_RUN_CONFIG_SCHEMA = "insight2_run_config_v2"
LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA = "insight2_run_config_v3"
LONG_VIDEO_RUN_CONFIG_SCHEMAS = frozenset(
    {LONG_VIDEO_RUN_CONFIG_SCHEMA, LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA}
)
RUN_RESULT_SCHEMA = "insight2_run_result_v1"
PREFLIGHT_SCHEMA = "insight2_preflight_v1"
PROMPT_SHARD_SCHEMA = "insight2_prompt_shard_v1"
PARTIAL_SHARD_RESULT_SCHEMA = "insight2_partial_prompt_shard_v1"
PROMPT_SHARD_TERMINAL_AUDIT_SCHEMA = "insight2_prompt_shard_terminal_audit_v2"
KV_AUDIT_SCHEMA = "insight2_kv_consistency_audit_v1"
TARGET_RPC_AUDIT_SCHEMA = "insight2_target_rpc_accounting_audit_v2"
MODEL_STATE_SCHEMA = "insight2_model_state_log_v1"
EVALUATOR_STATE_SCHEMA = "insight2_evaluator_state_v1"
DECODE_EVIDENCE_SCHEMA = "insight2_decode_evidence_v1"
QUALITY_MEASUREMENT_SCHEMA = "insight2_quality_measurement_v1"
VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA = "insight2_vbench_long_full_quality_v1"
PRIMARY_QUALITY_METRIC = "musiq_spaq_frame_mean_raw_div_100"
VBENCH_LONG_PRIMARY_QUALITY_METRIC = (
    "vbench_long_custom_six_dimension_macro_mean"
)
TEMPORAL_CONSISTENCY_METRIC = "adjacent_frame_l1_similarity_v1"
PRIMARY_QUALITY_DEFINITION = (
    "MUSIQ-SPAQ raw score divided by 100 on each exact lossless decoded RGB frame; "
    "aggregate is math.fsum(scores)/len(scores)"
)
TEMPORAL_CONSISTENCY_DEFINITION = (
    "1 - mean absolute uint8 RGB difference between adjacent frames in the exact "
    "half-open temporal window, divided by 255"
)
PHASES = ("2a-smoke", "2a-eval", "2b-smoke", "2b-eval")
LEGACY_STRATEGIES = (
    "random",
    "uniform",
    "online_rank_guided",
    "greedy_oracle",
)
LONG_VIDEO_STRATEGIES = ("random", "uniform", "online_rank_guided")
COVERAGE_CONSTRAINED_LONG_VIDEO_STRATEGIES = (
    *LONG_VIDEO_STRATEGIES,
    "online_coverage_guided",
)
LOOKAHEAD_COVERAGE_LONG_VIDEO_STRATEGIES = (
    *COVERAGE_CONSTRAINED_LONG_VIDEO_STRATEGIES,
    "online_lookahead_coverage_guided",
)
BOUNDARY_GATED_LONG_VIDEO_STRATEGIES = (
    "uniform",
    "online_lookahead_coverage_guided",
    "online_lookahead_coverage_boundary_gated",
)
LONG_VIDEO_PROMPT_SHARD_COUNT = 4
LONG_VIDEO_PROMPTS_PER_SHARD = 12
LONG_VIDEO_H800_DEADLINE_PROMPTS_PER_SHARD = 6
VBENCH_LONG_CHECKOUT = "/data/chenjiayu/cfexp/vendor/VBench-fd18b3d055cb"
VBENCH_LONG_PYTHON = "/data/chenjiayu/miniconda3/envs/vbench/bin/python"
VBENCH_LONG_ASSET_MANIFEST = (
    "/data/chenjiayu/cfexp/models/vbench-long-offline-v1/asset-manifest.json"
)
VBENCH_LONG_ASSET_SPEC = str(
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "vbench-long-offline-assets-v1.json"
)
CALIBRATION_RECEIPT_SEAL_ENV = "CFEXP_INSIGHT2_SEAL_CALIBRATION_RECEIPT"
CALIBRATION_RECEIPT_REQUIRE_ENV = "CFEXP_INSIGHT2_REQUIRE_CALIBRATION_RECEIPT"
CALIBRATION_RECEIPT_SOURCE_ENV = "CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SOURCE"
CALIBRATION_RECEIPT_SHA256_ENV = "CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SHA256"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATED_CHUNK_PATH_RE = re.compile(r"(?:^|/)chunk-([0-9]+)(?=[^0-9]|$)")
_LOOKAHEAD_PREVIEW_STEM_RE = re.compile(
    r"^chunk-([0-9]{2,})-draft-preview-from-([0-9]{2,})$"
)
_LOOKAHEAD_AFTER_TARGET_STEM_RE = re.compile(
    r"^chunk-([0-9]{2,})-draft-after-target-from-([0-9]{2,})$"
)
_LOOKAHEAD_ROUTE_STEM_RE = re.compile(
    r"^chunk-([0-9]{2,})-(draft|target)$"
)
_GENERATED_ATTEMPT_FILENAME_RE = re.compile(
    r"^attempt-[0-9]{4,}\.safetensors$"
)
_PHASE_RE = re.compile(r"^2[ab]-(?:smoke|eval)$")
_KV_CHECKPOINT_CATEGORIES = frozenset(
    {
        "calibration-baselines",
        "evaluation-baselines",
        "rollout-roots",
        "rollouts",
    }
)
_KV_CHECKPOINT_FILENAME_RE = re.compile(r"^chunk-([0-9]{2,})\.json$")
_KV_PENDING_PLAN_SPLITS = {
    "pending/insight2b-smoke.json": "smoke",
    "pending/insight2b-evaluation.json": "evaluation",
}
_TARGET_RPC_DIAGNOSTICS_KEYS = frozenset(
    {
        "schema",
        "attempt_id",
        "logical_call_id",
        "scope_id",
        "phase",
        "model_id",
        "model_artifact_sha256",
        "prompt_id",
        "prompt_sha256",
        "chunk_index",
        "internal_seed",
        "history_head_sha256",
        "history_latent_paths",
        "noise_descriptor",
        "noise_descriptor_sha256",
        "initial_noise_path",
        "latent_path",
        "latent_sha256",
        "segment_sha256",
        "kv_cache_sha256",
        "backend_diagnostics",
    }
)

_GENERATED_RECORD_KEYS = frozenset(
    {
        "context",
        "segment",
        "effective_rank",
        "hidden_state_path",
        "hidden_state_sha256",
        "diagnostics",
        "target_attempt_id",
        "target_logical_call_id",
    }
)

_DRAFT_BACKEND_DIAGNOSTIC_KEYS = frozenset(
    {
        "kv_witness_sha256",
        "kv_witness",
        "forward_trace",
        "rank_evidence",
        "boundary_velocity_witness",
    }
)

_BOUNDARY_VELOCITY_WITNESS_KEYS = frozenset(
    {
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
    }
)

_BOUNDARY_CHECKPOINT_METADATA_ROOT_KEYS = frozenset(
    {
        "kind",
        "route_id",
        "chunk_index",
        "corrected",
        "risk_score",
        "draft_proposal",
        "draft_kv_hash",
        "committed_role",
        "committed_kv_hash",
        "initial_noise_descriptor_sha256",
        "gate_signal_chunk_index",
        "gate_signal_rank",
        "gate_probe",
        "gate_probe_parent_history_hash",
        "gate_probe_kv_hash",
        "gate_probe_noise_descriptor_sha256",
        "target_candidate_proposed",
        "target_candidate",
        "target_candidate_parent_history_hash",
        "target_candidate_kv_hash",
        "target_candidate_noise_descriptor_sha256",
        "corrected_frontier_probe",
        "corrected_frontier_parent_history_hash",
        "corrected_frontier_kv_hash",
        "corrected_frontier_noise_descriptor_sha256",
        "boundary_gate_decision",
        "target_candidate_accepted",
        "carried_frontier_kind",
        "probe_reused_as_successor",
        "terminal_flush",
    }
)
_BOUNDARY_CHECKPOINT_METADATA_NONROOT_KEYS = (
    _BOUNDARY_CHECKPOINT_METADATA_ROOT_KEYS
    | {"rank", "decision", "draft_parent_history_hash"}
)

_CSV_2A_FIELDS = (
    "prompt_id",
    "seed",
    "chunk_index",
    "risk_score",
    "rank",
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
    "target_rpc_attempt_id",
    "target_logical_call_id",
    "target_latent_path",
    "target_latent_sha256",
    "target_noise_descriptor_sha256",
    "hidden_state_path",
    "hidden_state_sha256",
    "event_id",
    "split",
    "schema",
    "draft_segment_hash",
    "target_segment_hash",
    "fork_evidence_path",
    "fork_evidence_sha256",
)

_CSV_2B_FIELDS = (
    "prompt_id",
    "seed",
    "budget",
    "strategy",
    "route_repeat",
    "selected_chunks",
    "proposed_target_chunks",
    "full_quality",
    "temporal_consistency",
    "committed_target_chunks",
    "budgeted_target_calls",
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
    "rollout_id",
    "event_id",
    "split",
    "schema",
)


class Insight2RunError(RuntimeError):
    """A formal run cannot satisfy its frozen, fail-closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Insight2RunError(message)


def _duplicate_checked_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise Insight2RunError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Insight2RunError(f"value is not canonical JSON: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise Insight2RunError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _safe_repository_path(repo_root: Path, value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty path")
    _require("\\" not in value and "\x00" not in value, f"{label} is not canonical")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    _require(
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and str(posix) == value
        and all(part not in {"", ".", ".."} for part in posix.parts),
        f"{label} must be a canonical repository-relative POSIX path",
    )
    unresolved = repo_root.joinpath(*posix.parts)
    _require(not unresolved.is_symlink(), f"{label} may not be a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as error:
        raise Insight2RunError(f"{label} escapes or is unavailable: {value}") from error
    _require(resolved.is_file(), f"{label} must be a regular file")
    return resolved


def _safe_run_file(run_dir: Path, value: Any, *, label: str) -> Path:
    """Resolve one canonical run-relative regular file without symlink traversal."""

    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty path")
    _require("\\" not in value and "\x00" not in value, f"{label} is not canonical")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    _require(
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and str(posix) == value
        and all(part not in {"", ".", ".."} for part in posix.parts),
        f"{label} must be a canonical run-relative POSIX path",
    )
    root = run_dir.resolve(strict=True)
    unresolved = root.joinpath(*posix.parts)
    current = root
    for part in posix.parts:
        current = current / part
        _require(not current.is_symlink(), f"{label} may not traverse a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise Insight2RunError(f"{label} escapes or is unavailable: {value}") from error
    _require(resolved.is_file(), f"{label} must be a regular file")
    return resolved


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{label} may not be a symlink")
    try:
        raw = path.resolve(strict=True).read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                Insight2RunError(f"{label} contains invalid constant {item!r}")
            ),
        )
    except Insight2RunError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Insight2RunError(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json_bytes(value) + b"\n")


@dataclass(frozen=True)
class Insight2RunConfig:
    schema: str
    path: Path
    protocol_path: Path
    draft_artifact_spec_path: Path
    target_artifact_spec_path: Path
    draft_gpu_slot: int
    target_gpu_slot: int
    budgets: tuple[int, ...]
    random_repeats: int
    strategies: tuple[str, ...]
    bootstrap_replicates: int
    shared_precomputed_umt5_conditioning: bool = False
    allow_draft_target_same_gpu: bool = False
    coverage_window_radius_chunks: int | None = None
    coverage_minimum_gap_chunks: int | None = None
    coverage_maximum_gap_chunks: int | None = None


@dataclass(frozen=True)
class PromptShard:
    """One fixed, zero-based contiguous shard of a frozen prompt order."""

    index: int
    count: int
    start: int
    stop: int
    prompt_ids: tuple[str, ...]


@dataclass(frozen=True)
class Preflight:
    repo_root: Path
    run_dir: Path
    model_root: Path
    phase: str
    split: str
    protocol: ResolvedInsight2Protocol
    config: Insight2RunConfig
    visible_devices: tuple[str, ...]
    protocol_sha256: str
    draft_spec_sha256: str
    target_spec_sha256: str
    draft_spec: Mapping[str, Any]
    target_spec: Mapping[str, Any]
    draft_verified_artifacts: tuple[Mapping[str, Any], ...]
    target_verified_artifacts: tuple[Mapping[str, Any], ...]
    vbench_offline_assets: Mapping[str, Any] | None
    vbench_checkout: Mapping[str, Any] | None
    prompt_shard: PromptShard | None = None
    seal_calibration_receipt: bool = False
    calibration_receipt_source: Path | None = None
    calibration_receipt_sha256: str | None = None
    calibration_receipt_binding: Mapping[str, Any] | None = None
    calibration_receipt_verified: VerifiedCalibrationReceipt | None = None


def load_run_config(path: str | Path, *, repo_root: str | Path) -> Insight2RunConfig:
    """Load the one repository-tracked, frozen phase-independent run config."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    unresolved = Path(path).expanduser()
    if not unresolved.is_absolute():
        unresolved = root / unresolved
    _require(not unresolved.is_symlink(), "run config may not be a symlink")
    try:
        config_path = unresolved.resolve(strict=True)
        config_path.relative_to(root)
    except (OSError, ValueError) as error:
        raise Insight2RunError("run config must be a repository file") from error
    value = _load_json_object(config_path, label="Insight 2 run config")
    expected = {
        "schema",
        "status",
        "protocol",
        "draft_artifact_spec",
        "target_artifact_spec",
        "gpu_roles",
        "multi_budget",
        "analysis",
    }
    optional = {"runtime_acceleration", "coverage_constraint"}
    _require(
        expected <= set(value) <= expected | optional,
        f"run config keys differ: missing={sorted(expected - set(value))}, "
        f"unknown={sorted(set(value) - expected - optional)}",
    )
    schema = value["schema"]
    _require(
        schema in {RUN_CONFIG_SCHEMA, *LONG_VIDEO_RUN_CONFIG_SCHEMAS},
        "config.schema must select a supported frozen Insight 2 run contract",
    )
    _require(value["status"] == "frozen", "run config must have status='frozen'")
    shared_conditioning = False
    allow_same_gpu = False
    coverage_radius: int | None = None
    coverage_minimum_gap: int | None = None
    coverage_maximum_gap: int | None = None
    if "runtime_acceleration" in value:
        acceleration = value["runtime_acceleration"]
        _require(
            isinstance(acceleration, dict)
            and set(acceleration)
            == {
                "shared_precomputed_umt5_conditioning",
                "allow_draft_target_same_gpu",
            },
            "runtime_acceleration keys are invalid",
        )
        shared_conditioning = acceleration["shared_precomputed_umt5_conditioning"]
        allow_same_gpu = acceleration["allow_draft_target_same_gpu"]
        _require(
            isinstance(shared_conditioning, bool) and isinstance(allow_same_gpu, bool),
            "runtime_acceleration switches must be boolean",
        )
        _require(
            schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA,
            "runtime_acceleration is restricted to the v3 H800 profile",
        )
        _require(
            not allow_same_gpu or shared_conditioning,
            "same-GPU Draft/Target requires shared precomputed UMT5 conditioning",
        )
    if "coverage_constraint" in value:
        coverage = value["coverage_constraint"]
        _require(
            isinstance(coverage, dict)
            and set(coverage)
            == {
                "window_radius_chunks",
                "min_gap_chunks",
                "max_gap_chunks",
            },
            "coverage_constraint keys are invalid",
        )
        coverage_radius = coverage["window_radius_chunks"]
        coverage_minimum_gap = coverage["min_gap_chunks"]
        coverage_maximum_gap = coverage["max_gap_chunks"]
        _require(
            all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in (
                    coverage_radius,
                    coverage_minimum_gap,
                    coverage_maximum_gap,
                )
            ),
            "coverage_constraint values must be exact integers",
        )
        _require(
            (
                coverage_radius,
                coverage_minimum_gap,
                coverage_maximum_gap,
            )
            == (4, 8, 20),
            "coverage_constraint must freeze radius=4, minimum_gap=8, maximum_gap=20",
        )
        _require(
            schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA,
            "coverage constraints are restricted to the v3 H800 profile",
        )
    gpu_roles = value["gpu_roles"]
    _require(
        isinstance(gpu_roles, dict) and set(gpu_roles) == {"draft", "target"},
        "gpu_roles must contain exactly draft and target",
    )
    draft_slot = gpu_roles["draft"]
    target_slot = gpu_roles["target"]
    _require(
        isinstance(draft_slot, int)
        and not isinstance(draft_slot, bool)
        and isinstance(target_slot, int)
        and not isinstance(target_slot, bool)
        and (
            (draft_slot == target_slot == 0)
            if allow_same_gpu
            else {draft_slot, target_slot} == {0, 1}
        ),
        (
            "same-GPU gpu_roles must assign both models to slot 0"
            if allow_same_gpu
            else "gpu_roles must assign distinct slots 0 and 1"
        ),
    )
    multi = value["multi_budget"]
    expected_multi_keys = (
        {"budgets", "random_repeats"}
        if schema == RUN_CONFIG_SCHEMA
        else {"budgets", "random_repeats", "strategies"}
    )
    _require(
        isinstance(multi, dict) and set(multi) == expected_multi_keys,
        "multi_budget keys are invalid",
    )
    budgets_raw = multi["budgets"]
    repeats = multi["random_repeats"]
    if schema == RUN_CONFIG_SCHEMA:
        expected_repeats = 10
        _require(
            repeats == expected_repeats,
            f"multi_budget.random_repeats must equal {expected_repeats}",
        )
        _require(budgets_raw == [2, 3, 4], "multi_budget.budgets must equal [2,3,4]")
        strategies = LEGACY_STRATEGIES
    else:
        _require(
            budgets_raw in ([3], [6]),
            "long-video multi_budget.budgets must be exactly [3] or [6]",
        )
        if coverage_radius is None:
            expected_profiles = (LONG_VIDEO_STRATEGIES,)
        else:
            expected_profiles = (
                COVERAGE_CONSTRAINED_LONG_VIDEO_STRATEGIES,
                LOOKAHEAD_COVERAGE_LONG_VIDEO_STRATEGIES,
                BOUNDARY_GATED_LONG_VIDEO_STRATEGIES,
            )
        parsed_strategies = tuple(multi["strategies"])
        _require(
            parsed_strategies in expected_profiles,
            "long-video strategies do not match a frozen policy profile",
        )
        expected_repeats = (
            0
            if parsed_strategies == BOUNDARY_GATED_LONG_VIDEO_STRATEGIES
            else 3
            if schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA
            else 10
        )
        _require(
            repeats == expected_repeats,
            f"multi_budget.random_repeats must equal {expected_repeats}",
        )
        _require(
            parsed_strategies != BOUNDARY_GATED_LONG_VIDEO_STRATEGIES
            or (
                schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA
                and budgets_raw == [6]
                and coverage_radius is not None
            ),
            "boundary-gated profile requires the v3 N=120 K=6 coverage contract",
        )
        strategies = parsed_strategies
    analysis = value["analysis"]
    _require(
        isinstance(analysis, dict) and set(analysis) == {"bootstrap_replicates"},
        "analysis keys are invalid",
    )
    bootstrap = analysis["bootstrap_replicates"]
    _require(bootstrap == 10_000, "analysis.bootstrap_replicates must equal 10000")
    return Insight2RunConfig(
        schema=str(schema),
        path=config_path,
        protocol_path=_safe_repository_path(root, value["protocol"], label="config.protocol"),
        draft_artifact_spec_path=_safe_repository_path(
            root, value["draft_artifact_spec"], label="config.draft_artifact_spec"
        ),
        target_artifact_spec_path=_safe_repository_path(
            root, value["target_artifact_spec"], label="config.target_artifact_spec"
        ),
        draft_gpu_slot=draft_slot,
        target_gpu_slot=target_slot,
        budgets=tuple(budgets_raw),
        random_repeats=repeats,
        strategies=strategies,
        bootstrap_replicates=bootstrap,
        shared_precomputed_umt5_conditioning=shared_conditioning,
        allow_draft_target_same_gpu=allow_same_gpu,
        coverage_window_radius_chunks=coverage_radius,
        coverage_minimum_gap_chunks=coverage_minimum_gap,
        coverage_maximum_gap_chunks=coverage_maximum_gap,
    )


def _protocol_geometry(protocol: ResolvedInsight2Protocol) -> tuple[int, int, str, float]:
    """Return and cross-check the protocol-owned temporal geometry."""

    design = protocol.raw["design"]
    chunk_count = design.get("chunk_count")
    latent_frames_per_chunk = design.get("latent_frames_per_chunk")
    fps = design.get("fps")
    _require(
        isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count >= 2,
        "protocol chunk_count must be an integer of at least two",
    )
    _require(
        isinstance(latent_frames_per_chunk, int)
        and not isinstance(latent_frames_per_chunk, bool)
        and latent_frames_per_chunk >= 1,
        "protocol latent_frames_per_chunk must be positive",
    )
    _require(
        isinstance(fps, int) and not isinstance(fps, bool) and fps >= 1,
        "protocol fps must be positive",
    )
    latent_frame_count = chunk_count * latent_frames_per_chunk
    rgb_frame_count = 1 + 4 * (latent_frame_count - 1)
    duration = Fraction(rgb_frame_count, fps)
    duration_exact = str(duration)
    _require(
        design.get("latent_frame_count") == latent_frame_count,
        "protocol latent frame count does not match chunk geometry",
    )
    _require(
        design.get("rgb_frame_count") == rgb_frame_count,
        "protocol RGB frame count does not match the causal-VAE mapping",
    )
    _require(
        design.get("exact_duration_seconds") == duration_exact,
        "protocol exact duration does not match frame count and FPS",
    )
    _require(
        design.get("candidate_chunk_indices") == list(range(1, chunk_count)),
        "protocol candidate chunk grid does not match chunk_count",
    )
    return chunk_count, rgb_frame_count, duration_exact, float(duration)


def _phase_seeds(
    protocol: ResolvedInsight2Protocol,
    config: Insight2RunConfig,
    *,
    split: str,
) -> tuple[int, ...]:
    """Resolve the frozen seed slice; v2 smoke is deliberately one seed."""

    seeds = tuple(int(item) for item in protocol.raw["design"]["seeds"])
    _require(bool(seeds) and len(seeds) == len(set(seeds)), "protocol seeds are invalid")
    if split == "smoke" and config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS:
        return seeds[:1]
    return seeds


def _visible_devices(config: Insight2RunConfig) -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    _require(
        (
            len(devices) == 1
            if config.allow_draft_target_same_gpu
            else len(devices) == 2 and len(set(devices)) == 2
        ),
        (
            "same-GPU Insight 2 requires exactly one visible GPU; "
            f"got {list(devices)!r}"
            if config.allow_draft_target_same_gpu
            else "formal Insight 2 requires exactly two unique visible GPUs; "
            f"got {list(devices)!r}"
        ),
    )
    _require(
        all(value.isdigit() for value in devices),
        "CUDA_VISIBLE_DEVICES entries must be non-negative integer device IDs",
    )
    return devices


def _run_directory() -> Path:
    value = os.environ.get("CFEXP_RUN_DIR")
    _require(bool(value), "CFEXP_RUN_DIR is required; use scripts/run_remote.sh for formal runs")
    unresolved = Path(str(value)).expanduser()
    _require(unresolved.is_absolute(), "CFEXP_RUN_DIR must be absolute")
    _require(not unresolved.is_symlink(), "CFEXP_RUN_DIR may not be a symlink")
    try:
        result = unresolved.resolve(strict=True)
    except OSError as error:
        raise Insight2RunError(f"CFEXP_RUN_DIR is unavailable: {error}") from error
    _require(result.is_dir(), "CFEXP_RUN_DIR must be a directory")
    return result


def _model_root() -> Path:
    value = os.environ.get("CFEXP_MODEL_ROOT") or os.environ.get("MODEL_ROOT")
    _require(bool(value), "CFEXP_MODEL_ROOT is required")
    unresolved = Path(str(value)).expanduser()
    _require(unresolved.is_absolute(), "CFEXP_MODEL_ROOT must be absolute")
    _require(not unresolved.is_symlink(), "CFEXP_MODEL_ROOT may not be a symlink")
    try:
        result = unresolved.resolve(strict=True)
    except OSError as error:
        raise Insight2RunError(f"CFEXP_MODEL_ROOT is unavailable: {error}") from error
    _require(result.is_dir(), "CFEXP_MODEL_ROOT must be a directory")
    return result


def _public_artifact_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove host-specific absolute paths from a published preflight report."""

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                key: row[key]
                for key in ("name", "role", "root", "requested_path", "size", "sha256")
                if key in row
            }
        )
    return result


def _strict_environment_flag(name: str) -> bool:
    value = os.environ.get(name)
    _require(
        value in {None, "", "1"},
        f"{name} must be unset or exactly '1'",
    )
    return value == "1"


def _calibration_receipt_binding(
    *,
    repo_root: Path,
    config: Insight2RunConfig,
    protocol: ResolvedInsight2Protocol,
    protocol_sha256: str,
    draft_spec_sha256: str,
    target_spec_sha256: str,
    draft_verified_artifacts: Sequence[Mapping[str, Any]],
    target_verified_artifacts: Sequence[Mapping[str, Any]],
    phase_seeds: Sequence[int],
) -> dict[str, Any]:
    """Bind one C10 receipt to every fact that can change its risk signal."""

    git = collect_git_metadata(repo_root)
    revision = git.get("sha")
    _require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "calibration receipt cannot resolve a full Git commit",
    )
    chunk_count, frame_count, duration_exact, _duration_decimal = _protocol_geometry(
        protocol
    )
    design = protocol.raw["design"]
    risk = protocol.raw["risk"]
    prompts = [dict(item) for item in protocol.calibration_prompts]
    _require(
        len(prompts) == 10,
        "calibration receipt requires the frozen C10 prompt population",
    )
    public_draft = sorted(
        _public_artifact_rows(draft_verified_artifacts), key=lambda item: str(item["name"])
    )
    public_target = sorted(
        _public_artifact_rows(target_verified_artifacts), key=lambda item: str(item["name"])
    )
    resolved_protocol_sha256 = hashlib.sha256(
        _canonical_json_bytes(protocol.raw)
    ).hexdigest()
    risk_sha256 = hashlib.sha256(_canonical_json_bytes(risk)).hexdigest()
    return {
        "schema": "insight2_calibration_binding_v1",
        "code_revision": revision,
        "config": {
            "path": config.path.relative_to(repo_root).as_posix(),
            "sha256": _sha256_file(config.path),
            "schema": config.schema,
        },
        "protocol": {
            "path": config.protocol_path.relative_to(repo_root).as_posix(),
            "sha256": protocol_sha256,
            "resolved_sha256": resolved_protocol_sha256,
            "id": str(protocol.raw["protocol_id"]),
        },
        "geometry": {
            "width": int(design["width"]),
            "height": int(design["height"]),
            "fps": int(design["fps"]),
            "denoising_steps": int(design["denoising_steps"]),
            "latent_frames_per_chunk": int(design["latent_frames_per_chunk"]),
            "chunk_count": chunk_count,
            "rgb_frame_count": frame_count,
            "duration_seconds_exact": duration_exact,
        },
        "models": dict(protocol.raw["models"]),
        "artifact_specs": {
            "draft": {
                "path": config.draft_artifact_spec_path.relative_to(repo_root).as_posix(),
                "sha256": draft_spec_sha256,
            },
            "target": {
                "path": config.target_artifact_spec_path.relative_to(repo_root).as_posix(),
                "sha256": target_spec_sha256,
            },
        },
        "verified_artifacts": {
            "draft": public_draft,
            "target": public_target,
        },
        "noise": {
            "namespace": NOISE_NAMESPACE,
            "seed_derivation": "sha256_namespace_prompt_id_trajectory_seed_chunk_index_v1",
            "shape": [1, 3, 16, 60, 104],
            "dtype": "bfloat16",
            "writer": "cpu_float32_standard_normal_then_bfloat16_safetensors",
        },
        "calibration_population": {
            "prompts": prompts,
            "seeds": [int(seed) for seed in phase_seeds],
        },
        "hidden_risk": {
            "definition": dict(risk),
            "sha256": risk_sha256,
        },
    }


def _resolve_calibration_receipt_policy(
    *,
    phase: str,
    config: Insight2RunConfig,
    expected_binding: Mapping[str, Any],
) -> tuple[
    bool,
    Path | None,
    str | None,
    VerifiedCalibrationReceipt | None,
]:
    """Resolve the opt-in receipt mode; legacy run-local behavior is default."""

    seal = _strict_environment_flag(CALIBRATION_RECEIPT_SEAL_ENV)
    required = _strict_environment_flag(CALIBRATION_RECEIPT_REQUIRE_ENV)
    source_value = os.environ.get(CALIBRATION_RECEIPT_SOURCE_ENV)
    digest_value = os.environ.get(CALIBRATION_RECEIPT_SHA256_ENV)
    source_set = isinstance(source_value, str) and bool(source_value)
    digest_set = isinstance(digest_value, str) and bool(digest_value)
    enabled = seal or required or source_set or digest_set
    if not enabled:
        return False, None, None, None
    _require(
        config.schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA,
        "shared calibration receipts are enabled only for the H800 long-video profile",
    )
    if seal:
        _require(
            phase == "2b-smoke" and not required and not source_set and not digest_set,
            "calibration receipt sealing is valid only for smoke and cannot consume a receipt",
        )
        return True, None, None, None
    _require(
        phase == "2b-eval",
        "calibration receipt consumption is valid only for formal 2b-eval",
    )
    _require(required, "formal calibration receipt source must be explicitly required")
    _require(
        source_set and digest_set,
        "formal calibration receipt source and externally frozen SHA-256 are required",
    )
    assert isinstance(source_value, str) and isinstance(digest_value, str)
    _require(
        _SHA256_RE.fullmatch(digest_value) is not None,
        "formal calibration receipt SHA-256 is invalid",
    )
    unresolved = Path(source_value).expanduser()
    _require(unresolved.is_absolute(), "formal calibration receipt source must be absolute")
    _require(not unresolved.is_symlink(), "formal calibration receipt source may not be a symlink")
    try:
        source = unresolved.resolve(strict=True)
    except OSError as error:
        raise Insight2RunError(
            f"formal calibration receipt source is unavailable: {error}"
        ) from error
    try:
        verified = verify_calibration_receipt(
            source_run_dir=source,
            expected_binding=expected_binding,
            expected_receipt_sha256=digest_value,
        )
    except CalibrationReceiptError as error:
        raise Insight2RunError(f"calibration receipt preflight failed: {error}") from error
    return False, source, digest_value, verified


def _long_video_prompts_per_shard(config_schema: str) -> int:
    if config_schema == LONG_VIDEO_RUN_CONFIG_SCHEMA:
        return LONG_VIDEO_PROMPTS_PER_SHARD
    if config_schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA:
        return LONG_VIDEO_H800_DEADLINE_PROMPTS_PER_SHARD
    raise Insight2RunError("prompt sharding is disabled for the legacy Insight 2 contract")


def _resolve_prompt_shard(
    *,
    config_schema: str,
    phase: str,
    evaluation_prompts: Sequence[Mapping[str, Any]],
    shard_index: int | None,
) -> PromptShard | None:
    """Resolve one of four fixed long-video 2(b)-evaluation prompt shards."""

    if shard_index is None:
        return None
    _require(
        config_schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS,
        "prompt sharding is disabled for the legacy Insight 2 contract",
    )
    _require(
        phase == "2b-eval",
        "prompt sharding is only valid for long-video 2b-eval; smoke and 2a are disabled",
    )
    _require(
        isinstance(shard_index, int)
        and not isinstance(shard_index, bool)
        and 0 <= shard_index < LONG_VIDEO_PROMPT_SHARD_COUNT,
        "prompt shard index must be a zero-based integer in [0,3]",
    )
    prompts_per_shard = _long_video_prompts_per_shard(config_schema)
    expected_prompt_count = LONG_VIDEO_PROMPT_SHARD_COUNT * prompts_per_shard
    _require(
        len(evaluation_prompts) == expected_prompt_count,
        f"prompt sharding requires the frozen {expected_prompt_count}-prompt "
        "long-video evaluation split",
    )
    prompt_ids: list[str] = []
    for item in evaluation_prompts:
        prompt_id = item.get("prompt_id") if isinstance(item, Mapping) else None
        _require(
            isinstance(prompt_id, str) and bool(prompt_id),
            "prompt sharding requires non-empty evaluation prompt IDs",
        )
        prompt_ids.append(prompt_id)
    _require(
        len(prompt_ids) == len(set(prompt_ids)),
        "prompt sharding requires unique evaluation prompt IDs",
    )
    start = shard_index * prompts_per_shard
    stop = start + prompts_per_shard
    return PromptShard(
        index=shard_index,
        count=LONG_VIDEO_PROMPT_SHARD_COUNT,
        start=start,
        stop=stop,
        prompt_ids=tuple(prompt_ids[start:stop]),
    )


def _prompt_shard_metadata(shard: PromptShard) -> dict[str, Any]:
    return {
        "schema": PROMPT_SHARD_SCHEMA,
        "status": "partial",
        "selection": "contiguous_frozen_protocol_order",
        "index_zero_based": shard.index,
        "shard_count": shard.count,
        "start_inclusive": shard.start,
        "stop_exclusive": shard.stop,
        "prompt_count": len(shard.prompt_ids),
        "prompt_ids": list(shard.prompt_ids),
        "full_evaluation_prompt_count": shard.count * len(shard.prompt_ids),
        "complete_dataset_statistics_allowed": False,
    }


def _selected_evaluation_prompt_values(
    preflight_result: Preflight,
) -> tuple[Mapping[str, Any], ...]:
    values = tuple(preflight_result.protocol.eval_prompts)
    shard = getattr(preflight_result, "prompt_shard", None)
    if shard is None:
        return values
    _require(isinstance(shard, PromptShard), "trusted prompt shard has an invalid type")
    resolved = _resolve_prompt_shard(
        config_schema=preflight_result.config.schema,
        phase=preflight_result.phase,
        evaluation_prompts=values,
        shard_index=shard.index,
    )
    _require(resolved == shard, "trusted prompt shard differs from the frozen protocol")
    selected = values[shard.start : shard.stop]
    _require(
        tuple(str(item["prompt_id"]) for item in selected) == shard.prompt_ids,
        "trusted prompt shard prompt IDs drifted",
    )
    return selected


def _selected_smoke_prompt_values(
    preflight_result: Preflight,
) -> tuple[Mapping[str, Any], ...]:
    shard = getattr(preflight_result, "prompt_shard", None)
    if shard is None:
        return tuple(preflight_result.protocol.smoke_prompts)
    # Insight2Campaign requires the smoke view to be the first prompt of its
    # evaluation view.  Sharded execution never runs smoke, but keeping the
    # manifest internally consistent makes the terminal audit unambiguous.
    return _selected_evaluation_prompt_values(preflight_result)[:1]


def _preflight_log_value(preflight_result: Preflight) -> dict[str, Any]:
    protocol = preflight_result.protocol
    config = preflight_result.config
    root = preflight_result.repo_root
    chunk_count, rgb_frame_count, duration_exact, _duration_decimal = (
        _protocol_geometry(protocol)
    )
    selected_evaluation = _selected_evaluation_prompt_values(preflight_result)
    value: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "phase": preflight_result.phase,
        "split": preflight_result.split,
        "config_path": config.path.relative_to(root).as_posix(),
        "config_sha256": _sha256_file(config.path),
        "protocol_path": config.protocol_path.relative_to(root).as_posix(),
        "protocol_sha256": preflight_result.protocol_sha256,
        "draft_artifact_spec_path": config.draft_artifact_spec_path.relative_to(
            root
        ).as_posix(),
        "draft_artifact_spec_sha256": preflight_result.draft_spec_sha256,
        "target_artifact_spec_path": config.target_artifact_spec_path.relative_to(
            root
        ).as_posix(),
        "target_artifact_spec_sha256": preflight_result.target_spec_sha256,
        "visible_gpu_count": len(preflight_result.visible_devices),
        "visible_devices": list(preflight_result.visible_devices),
        "gpu_role_slots": {
            "draft": config.draft_gpu_slot,
            "target": config.target_gpu_slot,
        },
        "draft_artifacts": _public_artifact_rows(
            preflight_result.draft_verified_artifacts
        ),
        "target_artifacts": _public_artifact_rows(
            preflight_result.target_verified_artifacts
        ),
        "vbench_offline_assets": preflight_result.vbench_offline_assets,
        "vbench_checkout": preflight_result.vbench_checkout,
        "calibration_prompt_count": len(protocol.calibration_prompts),
        "evaluation_prompt_count": len(selected_evaluation),
        "seeds": list(
            _phase_seeds(protocol, config, split=preflight_result.split)
        ),
        "chunk_count": chunk_count,
        "rgb_frame_count": rgb_frame_count,
        "duration_seconds_exact": duration_exact,
        "status": "passed",
    }
    if config.shared_precomputed_umt5_conditioning:
        value["runtime_acceleration"] = {
            "shared_precomputed_umt5_conditioning": True,
            "allow_draft_target_same_gpu": config.allow_draft_target_same_gpu,
        }
    shard = getattr(preflight_result, "prompt_shard", None)
    if shard is not None:
        _require(isinstance(shard, PromptShard), "trusted prompt shard has an invalid type")
        value["prompt_shard"] = _prompt_shard_metadata(shard)
    receipt_binding = getattr(preflight_result, "calibration_receipt_binding", None)
    if receipt_binding is not None:
        _require(
            isinstance(receipt_binding, Mapping),
            "trusted calibration receipt binding is invalid",
        )
        expected_binding_sha256 = hashlib.sha256(
            _canonical_json_bytes(receipt_binding)
        ).hexdigest()
        verified = getattr(preflight_result, "calibration_receipt_verified", None)
        if preflight_result.seal_calibration_receipt:
            value["calibration_receipt"] = {
                "mode": "seal",
                "relative_path": CALIBRATION_RECEIPT_RELATIVE,
                "binding_sha256": expected_binding_sha256,
            }
        else:
            _require(
                isinstance(verified, VerifiedCalibrationReceipt),
                "trusted consumed calibration receipt is missing",
            )
            value["calibration_receipt"] = {
                "mode": "consume",
                "receipt_sha256": verified.receipt_sha256,
                "binding_sha256": expected_binding_sha256,
                "checkpoint_count": verified.checkpoint_count,
                "target_baseline_count": verified.target_baseline_count,
            }
    return value


def preflight(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    phase: str,
    prompt_shard_index: int | None = None,
) -> Preflight:
    """Verify protocol, all model bytes, GPU isolation, and output confinement."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    _require((root / ".git").exists(), "Insight 2 must run from the Git repository")
    _require(phase in PHASES and _PHASE_RE.fullmatch(phase) is not None, "invalid phase")
    split = "smoke" if phase.endswith("smoke") else "evaluation"
    run_dir = _run_directory()
    model_root = _model_root()
    _require(
        root != run_dir and root not in run_dir.parents and run_dir not in root.parents,
        "CFEXP_RUN_DIR must be outside the Git repository",
    )
    config = load_run_config(config_path, repo_root=root)
    protocol = load_protocol(config.protocol_path)
    prompt_shard = _resolve_prompt_shard(
        config_schema=config.schema,
        phase=phase,
        evaluation_prompts=protocol.eval_prompts,
        shard_index=prompt_shard_index,
    )
    devices = _visible_devices(config)

    try:
        draft_spec, draft_verified, draft_module = verify_artifact_spec(
            config.draft_artifact_spec_path,
            model_root=model_root,
            repo_root=root,
        )
    except BaseException as error:
        raise Insight2RunError(f"Draft artifact preflight failed: {error}") from error
    _require(
        draft_module == "cfexp.self_forcing_frame_adapter",
        "Draft artifact spec selected an unexpected adapter",
    )
    try:
        target_spec, target_verified, target_module = verify_target_artifact_spec(
            config.target_artifact_spec_path,
            model_root=model_root,
            repo_root=root,
        )
    except BaseException as error:
        raise Insight2RunError(f"Target artifact preflight failed: {error}") from error
    _require(target_module == TARGET_BACKEND_MODULE, "Target backend module drifted")

    protocol_hash = _sha256_file(config.protocol_path)
    draft_hash = _sha256_file(config.draft_artifact_spec_path)
    target_hash = target_artifact_spec_sha256(config.target_artifact_spec_path)
    design = protocol.raw["design"]
    multi_budget = protocol.raw["multi_budget"]
    seeds = design["seeds"]
    expected_seeds = (
        [20260911]
        if config.schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA
        else [20260911, 20260912]
    )
    _require(seeds == expected_seeds, "protocol seed set drifted")
    chunk_count, _rgb_frame_count, _duration_exact, _duration_decimal = (
        _protocol_geometry(protocol)
    )
    _require(len(protocol.calibration_prompts) == 10, "protocol calibration split drifted")
    vbench_offline_assets: Mapping[str, Any] | None = None
    vbench_checkout: Mapping[str, Any] | None = None
    if config.schema == RUN_CONFIG_SCHEMA:
        _require(len(protocol.eval_prompts) == 50, "protocol evaluation split drifted")
        _require(len(protocol.smoke_prompts) == 10, "protocol smoke split drifted")
        _require(chunk_count == 16, "legacy protocol chunk count drifted")
    else:
        expected_budget_by_horizon = {60: 3, 120: 6}
        _require(
            phase.startswith("2b-"),
            "long-video configs are only valid for the multi-strategy 2(b) phase",
        )
        _require(
            chunk_count in expected_budget_by_horizon,
            "long-video protocol horizon must be N=60 or N=120",
        )
        _require(
            config.budgets == (expected_budget_by_horizon[chunk_count],),
            "long-video budget does not match its protocol horizon",
        )
        _require(
            len(protocol.eval_prompts)
            == LONG_VIDEO_PROMPT_SHARD_COUNT
            * _long_video_prompts_per_shard(config.schema),
            "long-video protocol evaluation prompt count differs from its run profile",
        )
        _require(
            len(protocol.smoke_prompts) == 1,
            "long-video smoke must be the frozen first evaluation prompt",
        )
        _require(
            multi_budget.get("strategies") == list(config.strategies),
            "long-video protocol strategy set drifted",
        )
        protocol_coverage = multi_budget.get("coverage_constraint")
        if config.coverage_window_radius_chunks is None:
            _require(
                protocol_coverage is None,
                "baseline long-video protocol unexpectedly enables coverage constraints",
            )
        else:
            _require(
                isinstance(protocol_coverage, Mapping)
                and protocol_coverage.get("window_radius_chunks")
                == config.coverage_window_radius_chunks
                and protocol_coverage.get("minimum_gap_chunks")
                == config.coverage_minimum_gap_chunks
                and protocol_coverage.get("maximum_gap_chunks")
                == config.coverage_maximum_gap_chunks,
                "run config coverage constraint differs from its protocol",
            )
        _require(
            protocol.raw.get("quality", {}).get("primary")
            not in {"musiq_spaq_lossless_rgb_mean", PRIMARY_QUALITY_METRIC},
            "long-video protocol must freeze VBench-Long rather than the MUSIQ proxy",
        )
        try:
            vbench_offline_assets = validate_vbench_long_offline_assets(
                VBENCH_LONG_ASSET_MANIFEST,
                spec_path=VBENCH_LONG_ASSET_SPEC,
            ).config_binding()
        except BaseException as error:
            raise Insight2RunError(
                f"VBench-Long offline asset preflight failed: {error}"
            ) from error
        try:
            vbench_checkout = validate_vbench_long_checkout(
                VBENCH_LONG_CHECKOUT
            ).config_binding()
        except BaseException as error:
            raise Insight2RunError(
                f"VBench-Long checkout preflight failed: {error}"
            ) from error
    _require(
        tuple(multi_budget.get("budgets", ())) == config.budgets
        and multi_budget.get("random_repeats") == config.random_repeats,
        "run config multi-budget design differs from its protocol",
    )
    receipt_environment_enabled = any(
        os.environ.get(name) not in {None, ""}
        for name in (
            CALIBRATION_RECEIPT_SEAL_ENV,
            CALIBRATION_RECEIPT_REQUIRE_ENV,
            CALIBRATION_RECEIPT_SOURCE_ENV,
            CALIBRATION_RECEIPT_SHA256_ENV,
        )
    )
    receipt_binding: Mapping[str, Any] | None = None
    receipt_seal = False
    receipt_source: Path | None = None
    receipt_sha256: str | None = None
    receipt_verified: VerifiedCalibrationReceipt | None = None
    if receipt_environment_enabled:
        receipt_binding = _calibration_receipt_binding(
            repo_root=root,
            config=config,
            protocol=protocol,
            protocol_sha256=protocol_hash,
            draft_spec_sha256=draft_hash,
            target_spec_sha256=target_hash,
            draft_verified_artifacts=tuple(draft_verified),
            target_verified_artifacts=tuple(target_verified),
            phase_seeds=_phase_seeds(protocol, config, split=split),
        )
        (
            receipt_seal,
            receipt_source,
            receipt_sha256,
            receipt_verified,
        ) = _resolve_calibration_receipt_policy(
            phase=phase,
            config=config,
            expected_binding=receipt_binding,
        )
    result = Preflight(
        repo_root=root,
        run_dir=run_dir,
        model_root=model_root,
        phase=phase,
        split=split,
        protocol=protocol,
        config=config,
        visible_devices=devices,
        protocol_sha256=protocol_hash,
        draft_spec_sha256=draft_hash,
        target_spec_sha256=target_hash,
        draft_spec=draft_spec,
        target_spec=target_spec,
        draft_verified_artifacts=tuple(draft_verified),
        target_verified_artifacts=tuple(target_verified),
        vbench_offline_assets=vbench_offline_assets,
        vbench_checkout=vbench_checkout,
        prompt_shard=prompt_shard,
        seal_calibration_receipt=receipt_seal,
        calibration_receipt_source=receipt_source,
        calibration_receipt_sha256=receipt_sha256,
        calibration_receipt_binding=receipt_binding,
        calibration_receipt_verified=receipt_verified,
    )
    _write_json(
        run_dir / "logs" / "preflight.json",
        _preflight_log_value(result),
    )
    return result


def _prompt_cases(values: Sequence[Mapping[str, Any]], *, split: str) -> tuple[PromptCase, ...]:
    return tuple(
        PromptCase(
            prompt_id=str(item["prompt_id"]),
            prompt=str(item["prompt"]),
            split=split,
        )
        for item in values
    )


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if isinstance(value, float):
        _require(math.isfinite(value), "raw result contains a non-finite float")
        return format(value, ".17g")
    return value


def export_raw_rows(
    *, run_dir: Path, phase: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[Path, Path]:
    """Write deterministic phase-scoped CSV and JSON exports from measured JSONL."""

    _require(phase in PHASES, "cannot export an unknown phase")
    is_a = phase.startswith("2a-")
    schema = RAW_2A_SCHEMA if is_a else RAW_2B_SCHEMA
    fields = _CSV_2A_FIELDS if is_a else _CSV_2B_FIELDS
    expected_split = "smoke" if phase.endswith("smoke") else "evaluation"
    normalized = [dict(row) for row in rows]
    _require(bool(normalized), "measured phase returned no raw rows")
    _require(
        all(
            row.get("schema") == schema and row.get("split") == expected_split for row in normalized
        ),
        "raw rows have a schema or split mismatch",
    )
    if is_a:
        normalized.sort(
            key=lambda row: (str(row["prompt_id"]), int(row["seed"]), int(row["chunk_index"]))
        )
    else:
        normalized.sort(
            key=lambda row: (
                str(row["prompt_id"]),
                int(row["seed"]),
                int(row["budget"]),
                str(row["strategy"]),
                -1 if row["route_repeat"] is None else int(row["route_repeat"]),
            )
        )
    stem = "insight2a" if is_a else "insight2b"
    suffix = "smoke" if expected_split == "smoke" else "evaluation"
    json_path = run_dir / "raw" / f"{stem}-{suffix}.json"
    csv_path = run_dir / "raw" / f"{stem}-{suffix}.csv"
    _write_json(json_path, {"schema": f"{stem}_raw_export_v1", "phase": phase, "rows": normalized})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            for row in normalized:
                writer.writerow({field: _csv_cell(row.get(field)) for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return csv_path, json_path


def analyze_export(
    *,
    preflight_result: Preflight,
    csv_path: Path,
) -> dict[str, str]:
    """Run only strict measured-data analysis and render publication figures."""

    _require(
        getattr(preflight_result, "prompt_shard", None) is None,
        "a partial prompt shard cannot emit full-dataset statistics or Figure 2(b)",
    )
    phase = preflight_result.phase
    protocol = preflight_result.protocol
    prompts = protocol.smoke_prompts if preflight_result.split == "smoke" else protocol.eval_prompts
    prompt_ids = [str(item["prompt_id"]) for item in prompts]
    seeds = list(
        _phase_seeds(
            protocol,
            preflight_result.config,
            split=preflight_result.split,
        )
    )
    output = preflight_result.run_dir / "analysis" / phase
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if phase.startswith("2a-"):
        rows = load_insight2a_rows(
            csv_path,
            expected_prompt_ids=prompt_ids,
            expected_seeds=seeds,
        )
        summary, top1, plot_data = analyze_insight2a(
            rows,
            bootstrap_replicates=preflight_result.config.bootstrap_replicates,
        )
        summary_path = output / "insight2a_summary.json"
        plot_path = output / "figure2a.plot-data.json"
        top1_path = output / "insight2a_top1_per_video.csv"
        png_path = output / "figure2a.png"
        pdf_path = output / "figure2a.pdf"
        write_analysis_json(summary_path, summary)
        write_analysis_json(plot_path, plot_data)
        write_top1_csv(top1_path, top1)
        render_figure2a(plot_data, png_path)
        render_figure2a(plot_data, pdf_path)
        artifacts.update(
            summary=str(summary_path.relative_to(preflight_result.run_dir)),
            plot_data=str(plot_path.relative_to(preflight_result.run_dir)),
            top1=str(top1_path.relative_to(preflight_result.run_dir)),
            figure_png=str(png_path.relative_to(preflight_result.run_dir)),
            figure_pdf=str(pdf_path.relative_to(preflight_result.run_dir)),
        )
    else:
        chunk_count, _frame_count, _duration_exact, _duration_decimal = (
            _protocol_geometry(protocol)
        )
        rows = load_insight2b_rows(
            csv_path,
            protocol_id=str(protocol.raw["protocol_id"]),
            expected_prompt_ids=prompt_ids,
            expected_seeds=seeds,
            random_repeats=preflight_result.config.random_repeats,
            chunk_count=chunk_count,
            expected_budgets=preflight_result.config.budgets,
            expected_strategies=preflight_result.config.strategies,
        )
        summary, plot_data = analyze_insight2b(
            rows,
            bootstrap_replicates=preflight_result.config.bootstrap_replicates,
            descriptive_only=(
                preflight_result.split == "smoke" and len(set(prompt_ids)) == 1
            ),
        )
        summary_path = output / "insight2b_summary.json"
        plot_path = output / "figure2b.plot-data.json"
        png_path = output / "figure2b.png"
        pdf_path = output / "figure2b.pdf"
        write_analysis_json(summary_path, summary)
        write_analysis_json(plot_path, plot_data)
        render_figure2b(plot_data, png_path)
        render_figure2b(plot_data, pdf_path)
        artifacts.update(
            summary=str(summary_path.relative_to(preflight_result.run_dir)),
            plot_data=str(plot_path.relative_to(preflight_result.run_dir)),
            figure_png=str(png_path.relative_to(preflight_result.run_dir)),
            figure_pdf=str(pdf_path.relative_to(preflight_result.run_dir)),
        )
    return artifacts


def _write_partial_shard_result(
    *,
    preflight_result: Preflight,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Seal a shard-local completion marker without computing global statistics."""

    shard = getattr(preflight_result, "prompt_shard", None)
    _require(isinstance(shard, PromptShard), "partial result requires a prompt shard")
    _require(
        preflight_result.phase == "2b-eval"
        and preflight_result.split == "evaluation"
        and preflight_result.config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS,
        "partial prompt-shard results are only valid for long-video 2b-eval",
    )
    selected = _selected_evaluation_prompt_values(preflight_result)
    selected_ids = tuple(str(item["prompt_id"]) for item in selected)
    seeds = _phase_seeds(
        preflight_result.protocol,
        preflight_result.config,
        split=preflight_result.split,
    )
    _require(
        preflight_result.config.strategies
        in {
            LONG_VIDEO_STRATEGIES,
            COVERAGE_CONSTRAINED_LONG_VIDEO_STRATEGIES,
            LOOKAHEAD_COVERAGE_LONG_VIDEO_STRATEGIES,
            BOUNDARY_GATED_LONG_VIDEO_STRATEGIES,
        }
        and len(preflight_result.config.budgets) == 1,
        "partial prompt sharding requires the frozen long-video strategy grid",
    )
    expected_routes = {
        (budget, "random", repeat)
        for budget in preflight_result.config.budgets
        for repeat in range(preflight_result.config.random_repeats)
    }
    expected_routes.update(
        (budget, strategy, None)
        for budget in preflight_result.config.budgets
        for strategy in preflight_result.config.strategies
        if strategy != "random"
    )
    routes_per_prompt_seed = len(expected_routes)
    expected_prompt_seed_pairs = len(selected_ids) * len(seeds)
    expected_row_count = expected_prompt_seed_pairs * routes_per_prompt_seed
    _require(
        len(rows) == expected_row_count,
        "partial prompt shard does not contain its complete route grid",
    )
    routes_by_pair: dict[tuple[str, int], set[tuple[int, str, int | None]]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        seed = row.get("seed")
        budget = row.get("budget")
        strategy = row.get("strategy")
        repeat = row.get("route_repeat")
        _require(
            row.get("schema") == RAW_2B_SCHEMA
            and row.get("split") == "evaluation"
            and isinstance(prompt_id, str)
            and bool(prompt_id)
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(budget, int)
            and not isinstance(budget, bool)
            and isinstance(strategy, str)
            and (repeat is None or (isinstance(repeat, int) and not isinstance(repeat, bool))),
            "partial prompt shard contains a malformed route row",
        )
        key = (prompt_id, seed)
        route = (budget, strategy, repeat)
        pair_routes = routes_by_pair.setdefault(key, set())
        _require(route not in pair_routes, "partial prompt shard contains a duplicate route")
        pair_routes.add(route)
    expected_pairs = {(prompt_id, seed) for prompt_id in selected_ids for seed in seeds}
    _require(
        set(routes_by_pair) == expected_pairs
        and all(value == expected_routes for value in routes_by_pair.values()),
        "partial prompt shard split a prompt/seed route grid",
    )

    path = preflight_result.run_dir / "analysis" / "2b-eval" / "partial-shard.json"
    _write_json(
        path,
        {
            "schema": PARTIAL_SHARD_RESULT_SCHEMA,
            "status": "partial",
            "phase": "2b-eval",
            "protocol_id": str(preflight_result.protocol.raw["protocol_id"]),
            "prompt_shard": _prompt_shard_metadata(shard),
            "seeds": list(seeds),
            "routes_per_prompt_seed": routes_per_prompt_seed,
            "expected_prompt_seed_pairs": expected_prompt_seed_pairs,
            "observed_prompt_seed_pairs": len(routes_by_pair),
            "expected_row_count": expected_row_count,
            "observed_row_count": len(rows),
            "full_evaluation_statistics_ready": False,
            "full_evaluation_analysis_performed": False,
            "merge_required": True,
            "notice": (
                "This run contains one of four prompt shards only. It must not be "
                "reported as the full evaluation or used to render Figure 2(b)."
            ),
        },
    )
    return {"partial_shard": path.relative_to(preflight_result.run_dir).as_posix()}


def _receipt_file_binding(
    run_dir: Path, relative: str, *, label: str
) -> dict[str, Any]:
    """Return one canonical, content-addressed binding below an attempt root."""

    path = _safe_run_file(run_dir, relative, label=label)
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "nbytes": path.stat().st_size,
    }


def _read_target_success_evidence(
    run_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read, validate, and summarize the immutable Target success ledgers.

    This helper is deliberately read-only.  The terminal shard receipt and the
    later merger both use it, so every raw-row scope is tied to the same
    completed lifecycle attempt and fsynced success event bytes.
    """

    try:
        lifecycle = load_target_rpc_lifecycle(run_dir, required=True)
    except Exception as error:
        raise Insight2RunError(
            f"cannot read terminal Target RPC lifecycle: {error}"
        ) from error
    success_path = _safe_run_file(
        run_dir,
        "logs/target-rpc-events.jsonl",
        label="terminal Target RPC success journal",
    )
    try:
        lines = success_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise Insight2RunError(
            f"cannot read terminal Target RPC success journal: {error}"
        ) from error
    expected_success_keys = {
        "schema",
        "event_id",
        "attempt_id",
        "logical_call_id",
        "scope_id",
        "phase",
        "chunk_index",
        "latent_path",
        "latent_sha256",
        "segment_sha256",
        "history_head_sha256",
        "kv_cache_sha256",
        "noise_descriptor_sha256",
        "diagnostics_path",
        "diagnostics_sha256",
    }
    successes: dict[str, dict[str, Any]] = {}
    success_event_ids: set[str] = set()
    for line_number, raw in enumerate(lines, 1):
        _require(bool(raw), f"Target success journal line {line_number} is blank")
        try:
            event = json.loads(
                raw,
                object_pairs_hook=_duplicate_checked_object,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    Insight2RunError(
                        f"Target success journal contains invalid constant {item!r}"
                    )
                ),
            )
        except Insight2RunError:
            raise
        except (json.JSONDecodeError, UnicodeError) as error:
            raise Insight2RunError(
                f"Target success journal line {line_number} is invalid: {error}"
            ) from error
        _require(
            isinstance(event, dict)
            and set(event) == expected_success_keys
            and event.get("schema") == TARGET_RPC_SCHEMA,
            "Target success journal event schema drifted",
        )
        event_id = event.get("event_id")
        attempt_id = event.get("attempt_id")
        _require(
            isinstance(event_id, str)
            and _SHA256_RE.fullmatch(event_id) is not None
            and event_id
            == hashlib.sha256(
                _canonical_json_bytes(
                    {key: value for key, value in event.items() if key != "event_id"}
                )
            ).hexdigest()
            and event_id not in success_event_ids
            and isinstance(attempt_id, str)
            and _SHA256_RE.fullmatch(attempt_id) is not None
            and attempt_id not in successes,
            "Target success journal identity is invalid or duplicated",
        )
        success_event_ids.add(event_id)
        successes[attempt_id] = event

    attempts: dict[str, dict[str, Any]] = {}
    for record in lifecycle:
        attempt_id = str(record["attempt_id"])
        state = attempts.setdefault(
            attempt_id,
            {
                "logical_call_id": record["logical_call_id"],
                "scope_id": record["scope_id"],
                "phase": record["phase"],
                "statuses": [],
                "completed": None,
            },
        )
        state["statuses"].append(record["status"])
        if record["status"] == "completed":
            state["completed"] = record
    _require(
        bool(attempts)
        and all(
            state["statuses"] == ["started", "backend_returned", "completed"]
            and state["completed"] is not None
            and state["scope_id"] is not None
            and state["phase"] != "unscoped"
            for state in attempts.values()
        )
        and set(successes) == set(attempts),
        "terminal Target evidence contains an incomplete, unscoped, or unmatched attempt",
    )

    by_scope: dict[str, list[dict[str, Any]]] = {}
    diagnostic_bindings: dict[str, str] = {}
    for attempt_id in sorted(attempts):
        state = attempts[attempt_id]
        completed = state["completed"]
        success = successes[attempt_id]
        assert isinstance(completed, Mapping)
        for lifecycle_key, success_key in (
            ("logical_call_id", "logical_call_id"),
            ("scope_id", "scope_id"),
            ("phase", "phase"),
            ("chunk_index", "chunk_index"),
            ("output_latent_path", "latent_path"),
            ("output_latent_sha256", "latent_sha256"),
            ("kv_witness_sha256", "kv_cache_sha256"),
        ):
            _require(
                completed[lifecycle_key] == success[success_key],
                f"Target success event disagrees with completed {success_key}",
            )
        scope_id = str(state["scope_id"])
        attempt = {
            "attempt_id": attempt_id,
            "logical_call_id": success["logical_call_id"],
            "success_event_id": success["event_id"],
            "phase": success["phase"],
            "chunk_index": success["chunk_index"],
            "latent_path": success["latent_path"],
            "latent_sha256": success["latent_sha256"],
            "segment_sha256": success["segment_sha256"],
            "history_head_sha256": success["history_head_sha256"],
            "kv_cache_sha256": success["kv_cache_sha256"],
            "noise_descriptor_sha256": success["noise_descriptor_sha256"],
            "diagnostics_path": success["diagnostics_path"],
            "diagnostics_sha256": success["diagnostics_sha256"],
        }
        by_scope.setdefault(scope_id, []).append(attempt)
        diagnostic_bindings[attempt_id] = str(success["diagnostics_sha256"])
    for values in by_scope.values():
        values.sort(key=lambda item: str(item["attempt_id"]))

    completed_by_phase = {
        lifecycle_phase: sum(
            state["phase"] == lifecycle_phase for state in attempts.values()
        )
        for lifecycle_phase in sorted(TARGET_RPC_PHASES)
    }
    summary = {
        "attempts": len(attempts),
        "attempted": len(attempts),
        "backend_returned": len(attempts),
        "completed": len(attempts),
        "unique_completed_logical_calls": len(
            {str(state["logical_call_id"]) for state in attempts.values()}
        ),
        "successful_events": len(successes),
        "diagnostics_files": len(
            {str(event["diagnostics_path"]) for event in successes.values()}
        ),
        "diagnostics_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(diagnostic_bindings)
        ).hexdigest(),
        "scoped_attempts": len(attempts),
        "completed_by_phase": completed_by_phase,
    }
    return by_scope, summary


def _validate_target_successes_for_raw_row(
    row: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]]
) -> None:
    """Bind one raw 2(b) row's physical counters to completed Target attempts."""

    _require(bool(attempts), "raw 2(b) route has no completed Target attempt")
    _require(
        all(
            isinstance(item, Mapping)
            and item.get("phase") in {"route_materialization", "selection_search"}
            for item in attempts
        ),
        "raw 2(b) route contains a non-route Target attempt",
    )
    selection = [item for item in attempts if item["phase"] == "selection_search"]
    materialized_logical_calls = {
        str(item["logical_call_id"])
        for item in attempts
        if item["phase"] == "route_materialization"
    }
    _require(
        all(
            row.get(field) == len(attempts)
            for field in (
                "physical_target_calls",
                "target_rpc_attempted",
                "target_rpc_backend_returned",
                "target_rpc_completed",
            )
        )
        and row.get("route_materialized_logical_target_calls")
        == len(materialized_logical_calls)
        and all(
            row.get(field) == len(selection)
            for field in (
                "selection_search_target_calls",
                "selection_search_target_rpc_attempted",
                "selection_search_target_rpc_backend_returned",
                "selection_search_target_rpc_completed",
            )
        ),
        "raw 2(b) Target counters differ from completed success evidence",
    )


def _shard_vbench_offline_contract(
    *,
    run_dir: Path,
    asset_manifest_path: str,
    scorer_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    asset_root = str(PurePosixPath(asset_manifest_path).parent)
    runtime_cache = run_dir.resolve(strict=True) / "tmp" / "vbench-long-runtime-cache"
    if scorer_id is not None:
        _require(
            scorer_id in {"scorer-0", "scorer-1"},
            "VBench scorer cache identity is invalid",
        )
        runtime_cache /= scorer_id
    policy = {
        "network": VBENCH_LONG_OFFLINE_NETWORK_POLICY,
        "load_ckpt_from_local": True,
        "subject_clip2clip_extractor": VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR,
        "background_clip2clip_extractor": VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR,
        "dev_flag_slow_fast_weights": "frozen_upstream_yaml",
        "scrubbed_parent_environment": list(VBENCH_LONG_SCRUBBED_ENVIRONMENT_VARIABLES),
    }
    environment = {
        **VBENCH_DISTRIBUTED_ENVIRONMENT,
        "HOME": f"{asset_root}/home",
        "VBENCH_CACHE_DIR": f"{asset_root}/vbench-cache",
        "MPLCONFIGDIR": str(runtime_cache / "matplotlib"),
        "XDG_CACHE_HOME": str(runtime_cache / "xdg"),
        "HF_HOME": str(runtime_cache / "huggingface"),
        "HF_HUB_CACHE": str(runtime_cache / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(runtime_cache / "huggingface" / "hub"),
        "HUGGING_FACE_HUB_CACHE": str(runtime_cache / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(runtime_cache / "huggingface" / "datasets"),
        "HF_ASSETS_CACHE": str(runtime_cache / "huggingface" / "assets"),
        "HUGGINGFACE_ASSETS_CACHE": str(runtime_cache / "huggingface" / "assets"),
        "HF_MODULES_CACHE": str(runtime_cache / "huggingface" / "modules"),
        "TRANSFORMERS_CACHE": str(runtime_cache / "transformers"),
        "DIFFUSERS_CACHE": str(runtime_cache / "diffusers"),
        "TORCH_HOME": str(runtime_cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(runtime_cache / "torch-extensions"),
        "PYTHONPYCACHEPREFIX": str(runtime_cache / "python-bytecode"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "socks5://127.0.0.1:9",
    }
    return policy, environment


def _validate_shard_vbench_batch_route(
    *,
    run_dir: Path,
    measurement_id: str,
    prompt_id: str,
    video_path: Path,
    config_path: Path,
    result_path: Path,
    official: Mapping[str, Any],
    config: Mapping[str, Any],
    configured_device: str,
    scorer_id: str | None,
    expected_checkout: Mapping[str, Any],
    expected_assets: Mapping[str, Any],
    expected_batch_measurement_ids: Sequence[str] | None,
) -> dict[str, float]:
    """Audit a route extracted from one shared lane-level official process."""

    batch_id = official.get("batch_id")
    _require(
        isinstance(batch_id, str) and re.fullmatch(r"batch-[0-9a-f]{64}", batch_id) is not None,
        "route official VBench batch identity is invalid",
    )
    batch_root = run_dir.resolve(strict=True) / VBENCH_LONG_BATCH_ROOT_RELATIVE / batch_id
    _require(
        config_path.resolve(strict=True) == batch_root / "config.json"
        and config.get("schema") == VBENCH_LONG_BATCH_CONFIG_SCHEMA
        and config.get("batch_id") == batch_id
        and config.get("upstream") == expected_checkout
        and config.get("offline_assets") == expected_assets
        and config.get("visible_device") == configured_device
        and config.get("mode") == "long_custom_input"
        and config.get("dev_flag") is True
        and config.get("dimensions") == list(VBENCH_LONG_DIMENSIONS),
        "route official VBench batch config identity drifted",
    )
    policy, environment = _shard_vbench_offline_contract(
        run_dir=run_dir,
        asset_manifest_path=VBENCH_LONG_ASSET_MANIFEST,
        scorer_id=scorer_id,
    )
    _require(
        config.get("offline_policy") == policy
        and config.get("offline_environment") == environment,
        "route official VBench batch policy or environment drifted",
    )
    python = config.get("python_executable")
    launcher = config.get("launcher")
    _require(
        isinstance(python, str)
        and PurePosixPath(python).is_absolute()
        and PurePosixPath(python).parent == PurePosixPath(VBENCH_LONG_PYTHON).parent
        and PurePosixPath(python).name.startswith("python")
        and isinstance(launcher, Mapping)
        and set(launcher) == {"path", "sha256", "nbytes"}
        and isinstance(launcher.get("path"), str)
        and PurePosixPath(str(launcher["path"])).is_absolute()
        and PurePosixPath(str(launcher["path"])).name == "vbench_offline_entry.py"
        and launcher.get("sha256")
        == _sha256_file(Path(__file__).with_name("vbench_offline_entry.py"))
        and launcher.get("nbytes")
        == Path(__file__).with_name("vbench_offline_entry.py").stat().st_size,
        "route official VBench batch Python or launcher binding drifted",
    )
    runtime_cache = run_dir.resolve(strict=True) / "tmp" / "vbench-long-runtime-cache"
    if scorer_id is not None:
        runtime_cache /= scorer_id
    expected_command = [
        python,
        str(launcher["path"]),
        "--asset-manifest",
        str(expected_assets["manifest"]["path"]),
        "--asset-spec",
        str(expected_assets["git_spec"]["path"]),
        "--checkout",
        VBENCH_LONG_CHECKOUT,
        "--eval-script",
        f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/eval_long.py",
        "--eval-sha256",
        str(expected_checkout["eval_script"]["sha256"]),
        "--eval-nbytes",
        str(expected_checkout["eval_script"]["nbytes"]),
        "--full-info",
        f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/VBench_full_info.json",
        "--full-info-sha256",
        str(expected_checkout["full_info"]["sha256"]),
        "--full-info-nbytes",
        str(expected_checkout["full_info"]["nbytes"]),
        "--runtime-cache-root",
        str(runtime_cache),
        "--",
        "--videos_path",
        str(batch_root / "videos"),
        "--dimension",
        *VBENCH_LONG_DIMENSIONS,
        "--mode",
        "long_custom_input",
        "--dev_flag",
        "--load_ckpt_from_local",
        "True",
        "--sb_clip2clip_feat_extractor",
        VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR,
        "--bg_clip2clip_feat_extractor",
        VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR,
        "--num_of_samples_per_prompt",
        "1",
        "--output_path",
        str(batch_root / "results"),
        "--full_json_dir",
        f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/VBench_full_info.json",
    ]
    _require(
        config.get("command") == expected_command,
        "route official VBench batch command drifted",
    )
    inputs = config.get("inputs")
    _require(isinstance(inputs, list) and bool(inputs), "route VBench batch has no inputs")
    if expected_batch_measurement_ids is not None:
        expected_measurements = sorted(expected_batch_measurement_ids)
        observed_measurements = sorted(
            str(item.get("measurement_id"))
            for item in inputs
            if isinstance(item, Mapping)
            and isinstance(item.get("measurement_id"), str)
        )
        _require(
            len(observed_measurements) == len(inputs)
            and len(expected_measurements) == len(set(expected_measurements))
            and observed_measurements == expected_measurements,
            "route VBench batch input set differs from its deterministic GPU lane",
        )
    selected = [
        item
        for item in inputs
        if isinstance(item, Mapping) and item.get("measurement_id") == measurement_id
    ]
    _require(len(selected) == 1, "route VBench batch does not uniquely bind its input")
    selected_input = selected[0]
    source_sha256 = _sha256_file(video_path)
    alias = vbench_long_staged_alias(
        prompt_id=prompt_id,
        measurement_id=measurement_id,
        source_video_sha256=source_sha256,
    )
    staged_relative = selected_input.get("staged_video")
    _require(
        selected_input.get("alias") == alias
        and selected_input.get("prompt") == prompt_id
        and selected_input.get("source_video")
        == {
            "path": str(video_path.resolve(strict=True)),
            "sha256": source_sha256,
            "nbytes": video_path.stat().st_size,
        }
        and isinstance(staged_relative, str),
        "route VBench batch input differs from its decoded video",
    )
    staged_path = _safe_run_file(
        run_dir, str(staged_relative), label="route staged VBench batch video"
    )
    _require(
        staged_path.samefile(video_path)
        and _sha256_file(staged_path) == source_sha256,
        "route staged VBench batch video differs from the decoded video",
    )
    sidecar_path = _safe_run_file(
        run_dir,
        f"vbench-long/{measurement_id}/measurement.json",
        label="route official VBench batch measurement",
    )
    try:
        measured = validate_vbench_long_batch_measurement(
            sidecar_path,
            run_dir=run_dir,
            measurement_id=measurement_id,
            prompt=prompt_id,
            video_path=video_path,
        )
    except BaseException as error:
        raise Insight2RunError(
            f"route official VBench batch provenance is invalid: {error}"
        ) from error
    _require(
        result_path.resolve(strict=True).parent == batch_root / "results"
        and official.get("official_result")
        == {
            "path": result_path.resolve(strict=True)
            .relative_to(run_dir.resolve(strict=True))
            .as_posix(),
            "sha256": _sha256_file(result_path),
            "nbytes": result_path.stat().st_size,
        },
        "route official VBench batch result binding drifted",
    )
    return measured.scores()


def validate_shard_vbench_route(
    *,
    run_dir: Path,
    measurement_id: str,
    prompt_id: str,
    video_path: Path,
    config_path: Path,
    result_path: Path,
    official: Mapping[str, Any],
    config: Mapping[str, Any],
    expected_device: str | Mapping[str, str],
    expected_checkout: Mapping[str, Any],
    expected_assets: Mapping[str, Any],
    expected_scorer_id: str | None = None,
    expected_batch_measurement_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Recheck a route's official bytes and exact invocation at sealing and merge."""

    configured_device = config.get("visible_device")
    if isinstance(expected_device, str):
        _require(
            configured_device == expected_device,
            "route official VBench device differs from its assigned GPU",
        )
        scorer_id: str | None = None
    else:
        _require(
            isinstance(expected_device, Mapping)
            and set(expected_device) == {"scorer-0", "scorer-1"}
            and all(
                isinstance(value, str) and bool(value) and "," not in value
                for value in expected_device.values()
            )
            and len(set(expected_device.values())) == 2,
            "route official VBench scorer/device map is invalid",
        )
        matches = [
            key for key, value in expected_device.items() if value == configured_device
        ]
        _require(
            len(matches) == 1,
            "route official VBench device is outside the two assigned scorer GPUs",
        )
        scorer_id = matches[0]
    _require(
        expected_scorer_id is None or scorer_id == expected_scorer_id,
        "route official VBench scorer slot differs from deterministic assignment",
    )

    if config.get("schema") == VBENCH_LONG_BATCH_CONFIG_SCHEMA:
        return _validate_shard_vbench_batch_route(
            run_dir=run_dir,
            measurement_id=measurement_id,
            prompt_id=prompt_id,
            video_path=video_path,
            config_path=config_path,
            result_path=result_path,
            official=official,
            config=config,
            configured_device=str(configured_device),
            scorer_id=scorer_id,
            expected_checkout=expected_checkout,
            expected_assets=expected_assets,
            expected_batch_measurement_ids=expected_batch_measurement_ids,
        )

    root = run_dir.resolve(strict=True) / "vbench-long" / measurement_id
    source_sha256 = _sha256_file(video_path)
    alias = vbench_long_staged_alias(
        prompt_id=prompt_id,
        measurement_id=measurement_id,
        source_video_sha256=source_sha256,
    )
    staged_relative = f"vbench-long/{measurement_id}/videos/{alias}.mp4"
    staged_path = _safe_run_file(run_dir, staged_relative, label="route staged VBench video")
    _require(
        staged_path.samefile(video_path)
        and _sha256_file(staged_path) == source_sha256,
        "route staged VBench video differs from the current decoded video",
    )
    _require(
        config_path.resolve(strict=True) == root / "config.json"
        and result_path.parent.resolve(strict=True) == root / "results"
        and set(config)
        == {
            "schema", "measurement_id", "upstream", "python_executable",
            "launcher", "visible_device", "mode", "dev_flag", "dimensions",
            "offline_assets", "offline_policy", "offline_environment", "prompt",
            "staging", "source_video", "staged_video", "command",
        }
        and config.get("schema") == VBENCH_LONG_CONFIG_SCHEMA
        and config.get("measurement_id") == measurement_id
        and config.get("upstream") == expected_checkout
        and config.get("offline_assets") == expected_assets
        and config.get("visible_device") == configured_device
        and config.get("mode") == "long_custom_input"
        and config.get("dev_flag") is True
        and config.get("dimensions") == list(VBENCH_LONG_DIMENSIONS)
        and config.get("prompt") == prompt_id
        and config.get("staging")
        == {
            "algorithm": VBENCH_LONG_STAGED_ALIAS_ALGORITHM,
            "alias": alias,
            "original_prompt_id": prompt_id,
        }
        and config.get("source_video")
        == {
            "path": str(video_path.resolve(strict=True)),
            "sha256": source_sha256,
            "nbytes": video_path.stat().st_size,
        }
        and config.get("staged_video") == staged_relative,
        "route official VBench config is not bound to its video, alias, and device",
    )
    python = config["python_executable"]
    launcher = config["launcher"]
    checkout = expected_checkout
    assets = expected_assets
    asset_manifest = assets.get("manifest")
    asset_spec = assets.get("git_spec")
    _require(
        isinstance(python, str)
        and PurePosixPath(python).is_absolute()
        and PurePosixPath(python).parent == PurePosixPath(VBENCH_LONG_PYTHON).parent
        and PurePosixPath(python).name.startswith("python")
        and isinstance(launcher, Mapping)
        and set(launcher) == {"path", "sha256", "nbytes"}
        and isinstance(launcher.get("path"), str)
        and PurePosixPath(str(launcher["path"])).is_absolute()
        and PurePosixPath(str(launcher["path"])).name == "vbench_offline_entry.py"
        and launcher.get("sha256") == _sha256_file(Path(__file__).with_name("vbench_offline_entry.py"))
        and launcher.get("nbytes") == Path(__file__).with_name("vbench_offline_entry.py").stat().st_size
        and isinstance(checkout.get("eval_script"), Mapping)
        and isinstance(checkout.get("full_info"), Mapping)
        and isinstance(asset_manifest, Mapping)
        and asset_manifest.get("path") == VBENCH_LONG_ASSET_MANIFEST
        and isinstance(asset_spec, Mapping)
        and isinstance(asset_spec.get("path"), str)
        and PurePosixPath(str(asset_spec["path"])).name == "vbench-long-offline-assets-v1.json"
        and checkout.get("repository") == "https://github.com/Vchitect/VBench"
        and checkout.get("commit") == VBENCH_LONG_COMMIT
        and checkout.get("checkout") == VBENCH_LONG_CHECKOUT,
        "route official VBench Python, launcher, or frozen assets are invalid",
    )
    policy, environment = _shard_vbench_offline_contract(
        run_dir=run_dir,
        asset_manifest_path=VBENCH_LONG_ASSET_MANIFEST,
        scorer_id=scorer_id,
    )
    _require(
        config.get("offline_policy") == policy
        and config.get("offline_environment") == environment,
        "route official VBench offline policy or environment drifted",
    )
    runtime_cache = run_dir.resolve(strict=True) / "tmp" / "vbench-long-runtime-cache"
    if scorer_id is not None:
        runtime_cache /= scorer_id
    expected_command = [
        python, str(launcher["path"]),
        "--asset-manifest", str(assets["manifest"]["path"]),
        "--asset-spec", str(assets["git_spec"]["path"]),
        "--checkout", VBENCH_LONG_CHECKOUT,
        "--eval-script", f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/eval_long.py",
        "--eval-sha256", str(checkout["eval_script"]["sha256"]),
        "--eval-nbytes", str(checkout["eval_script"]["nbytes"]),
        "--full-info", f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/VBench_full_info.json",
        "--full-info-sha256", str(checkout["full_info"]["sha256"]),
        "--full-info-nbytes", str(checkout["full_info"]["nbytes"]),
        "--runtime-cache-root", str(runtime_cache), "--",
        "--videos_path", str(root / "videos"), "--dimension", *VBENCH_LONG_DIMENSIONS,
        "--mode", "long_custom_input", "--dev_flag", "--load_ckpt_from_local", "True",
        "--sb_clip2clip_feat_extractor", VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR,
        "--bg_clip2clip_feat_extractor", VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR,
        "--num_of_samples_per_prompt", "1", "--output_path", str(root / "results"),
        "--full_json_dir", f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/VBench_full_info.json",
    ]
    _require(config.get("command") == expected_command, "route official VBench command drifted")
    _require(
        official.get("config_sha256") == _sha256_file(config_path),
        "route official VBench config digest drifted",
    )
    result_relative = result_path.resolve(strict=True).relative_to(
        run_dir.resolve(strict=True)
    ).as_posix()
    _require(
        set(official) == {
            "schema", "measurement_id", "config_sha256", "official_result",
            "dimension_scores", "aggregation", "macro_mean", "motion_smoothness",
            "staging",
        }
        and official.get("schema") == VBENCH_LONG_OFFICIAL_MEASUREMENT_SCHEMA
        and official.get("measurement_id") == measurement_id
        and official.get("staging") == config["staging"]
        and official.get("aggregation")
        == "unweighted_arithmetic_mean_of_six_official_dimension_scores"
        and official.get("official_result") == {
            "path": result_relative,
            "sha256": _sha256_file(result_path),
            "nbytes": result_path.stat().st_size,
        },
        "route official VBench measurement binding drifted",
    )
    try:
        scores = validate_vbench_long_official_result(
            result_path,
            measurement_root=root,
            staged_video=staged_path,
            staged_alias=alias,
        )
        successful = validate_vbench_long_success_attempt(
            root,
            run_dir=run_dir,
            config_sha256=str(official["config_sha256"]),
            command=expected_command,
        )
    except BaseException as error:
        raise Insight2RunError(f"route official VBench provenance is invalid: {error}") from error
    _require(successful, "route official VBench has no bound successful attempt")
    _require(
        official.get("dimension_scores") == scores
        and official.get("macro_mean") == math.fsum(scores.values()) / len(scores)
        and official.get("motion_smoothness") == scores["motion_smoothness"],
        "route official VBench detail scores differ from the measurement",
    )
    return scores


def _write_prompt_shard_terminal_audit(
    *,
    preflight_result: Preflight,
    rows: Sequence[Mapping[str, Any]],
    raw_csv_path: Path,
    raw_json_path: Path,
    target_rpc_report: Mapping[str, Any],
    kv_report: Mapping[str, Any],
    partial_path: str,
) -> dict[str, Any]:
    """Freeze the already-audited route evidence used by a later shard merger.

    ``audit_target_rpc_accounting`` has already replayed every checkpoint,
    lifecycle event, model-local KV witness, and VBench-Long measurement.  This
    receipt does not replace that audit.  It content-addresses its terminal
    reports and the exact per-route quality evidence so a merger can verify the
    published, sealed bytes without mutating or replaying a shard.
    """

    shard = getattr(preflight_result, "prompt_shard", None)
    _require(isinstance(shard, PromptShard), "terminal audit requires a prompt shard")
    _require(
        preflight_result.phase == "2b-eval"
        and preflight_result.split == "evaluation"
        and preflight_result.config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS,
        "terminal shard audit is only valid for formal long-video 2b-eval",
    )
    _require(
        target_rpc_report.get("schema") == TARGET_RPC_AUDIT_SCHEMA
        and target_rpc_report.get("status") == "passed"
        and target_rpc_report.get("phase") == "2b-eval"
        and target_rpc_report.get("trusted_preflight_bound") is True
        and target_rpc_report.get("reconciled_raw_scopes") == len(rows),
        "terminal shard audit requires the trusted full Target/quality audit",
    )
    _require(
        kv_report.get("schema") == KV_AUDIT_SCHEMA
        and kv_report.get("status") == "passed",
        "terminal shard audit requires the passed model-state/KV audit",
    )
    git = collect_git_metadata(preflight_result.repo_root)
    revision = git.get("sha")
    _require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "terminal shard audit cannot resolve a full Git commit",
    )

    run_dir = preflight_result.run_dir
    target_attempts_by_scope, target_attempt_summary = _read_target_success_evidence(
        run_dir
    )
    _require(
        all(target_rpc_report.get(key) == value for key, value in target_attempt_summary.items())
        and target_rpc_report.get("lifecycle_sha256")
        == _sha256_file(run_dir / "logs" / "target-rpc-lifecycle.jsonl")
        and target_rpc_report.get("success_journal_sha256")
        == _sha256_file(run_dir / "logs" / "target-rpc-events.jsonl"),
        "terminal Target success ledgers differ from the passed accounting audit",
    )
    expected_raw_csv = raw_csv_path.relative_to(run_dir).as_posix()
    expected_raw_json = raw_json_path.relative_to(run_dir).as_posix()
    bindings = {
        "preflight": _receipt_file_binding(
            run_dir, "logs/preflight.json", label="terminal-audit preflight"
        ),
        "model_state": _receipt_file_binding(
            run_dir, "logs/model-state.json", label="terminal-audit model state"
        ),
        "evaluator_state": _receipt_file_binding(
            run_dir, "logs/evaluator-state.json", label="terminal-audit evaluator state"
        ),
        "target_rpc_audit": _receipt_file_binding(
            run_dir,
            "logs/target-rpc-accounting.json",
            label="terminal-audit Target RPC report",
        ),
        "target_rpc_lifecycle": _receipt_file_binding(
            run_dir,
            "logs/target-rpc-lifecycle.jsonl",
            label="terminal-audit Target RPC lifecycle",
        ),
        "target_rpc_success_journal": _receipt_file_binding(
            run_dir,
            "logs/target-rpc-events.jsonl",
            label="terminal-audit Target RPC success journal",
        ),
        "kv_audit": _receipt_file_binding(
            run_dir, "logs/kv-consistency.json", label="terminal-audit KV report"
        ),
        "campaign": _receipt_file_binding(
            run_dir, "campaign.json", label="terminal-audit campaign"
        ),
        "partial_shard": _receipt_file_binding(
            run_dir, partial_path, label="terminal-audit partial marker"
        ),
    }
    _require(
        _load_json_object(
            _safe_run_file(
                run_dir,
                bindings["target_rpc_audit"]["path"],
                label="terminal-audit Target RPC report",
            ),
            label="terminal-audit Target RPC report",
        )
        == dict(target_rpc_report)
        and _load_json_object(
            _safe_run_file(
                run_dir,
                bindings["kv_audit"]["path"],
                label="terminal-audit KV report",
            ),
            label="terminal-audit KV report",
        )
        == dict(kv_report),
        "terminal audit report bytes differ from the passed in-memory audits",
    )

    route_receipts: list[dict[str, Any]] = []
    ordered_rows = sorted(
        (dict(item) for item in rows),
        key=lambda item: (
            str(item["prompt_id"]),
            int(item["seed"]),
            int(item["budget"]),
            str(item["strategy"]),
            -1 if item.get("route_repeat") is None else int(item["route_repeat"]),
        ),
    )
    expected_scorer_by_measurement: dict[str, str] = {}
    if (
        preflight_result.config.schema
        == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA
    ):
        unique_measurements: set[str] = set()
        for row in ordered_rows:
            route_id = row.get("rollout_id")
            _require(
                isinstance(route_id, str) and bool(route_id),
                "terminal audit encountered an invalid v3 rollout identity",
            )
            route_key = hashlib.sha256(
                _canonical_json_bytes(
                    [
                        str(preflight_result.protocol.raw["protocol_id"]),
                        str(row["prompt_id"]),
                        int(row["seed"]),
                        route_id,
                    ]
                )
            ).hexdigest()[:48]
            unique_measurements.add(f"2b-{route_key}")
        expected_scorer_by_measurement = {
            measurement_id: f"scorer-{index % 2}"
            for index, measurement_id in enumerate(sorted(unique_measurements))
        }
    for row in ordered_rows:
        route_id = row.get("rollout_id")
        event_id = row.get("event_id")
        _require(
            isinstance(route_id, str)
            and bool(route_id)
            and isinstance(event_id, str)
            and _SHA256_RE.fullmatch(event_id) is not None,
            "terminal audit encountered an invalid route identity",
        )
        successful_attempts = target_attempts_by_scope.get(event_id, [])
        _validate_target_successes_for_raw_row(row, successful_attempts)
        route_key = hashlib.sha256(
            _canonical_json_bytes(
                [
                    str(preflight_result.protocol.raw["protocol_id"]),
                    str(row["prompt_id"]),
                    int(row["seed"]),
                    route_id,
                ]
            )
        ).hexdigest()[:48]
        measurement_id = f"2b-{route_key}"
        score_relative = f"measurements/rollouts/{route_key}.json"
        bridge_relative = (
            f"measurements/vbench-long-quality/{measurement_id}.json"
        )
        score = _load_json_object(
            _safe_run_file(run_dir, score_relative, label="terminal-audit rollout score"),
            label="terminal-audit rollout score",
        )
        bridge = _load_json_object(
            _safe_run_file(
                run_dir, bridge_relative, label="terminal-audit VBench bridge"
            ),
            label="terminal-audit VBench bridge",
        )
        bridge_scores = bridge.get("scores")
        _require(
            score.get("schema") == ROLLOUT_SCORE_SCHEMA
            and score.get("route_id") == route_id
            and score.get("corrected_chunks") == row.get("selected_chunks")
            and score.get("proposed_target_chunks")
            == row.get("proposed_target_chunks")
            and score.get("target_call_count")
            == row.get("committed_target_chunks")
            and score.get("budgeted_target_call_count")
            == row.get("budgeted_target_calls")
            and score.get("full_video_quality") == row.get("full_quality")
            and score.get("temporal_consistency") == row.get("temporal_consistency")
            and score.get("final_history_hash") == row.get("final_history_hash")
            and bridge.get("schema") == VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA
            and bridge.get("measurement_id") == measurement_id
            and bridge.get("metric_id") == VBENCH_LONG_PRIMARY_QUALITY_METRIC
            and isinstance(bridge_scores, Mapping)
            and bridge_scores.get("macro_mean") == row.get("full_quality")
            and bridge_scores.get("motion_smoothness")
            == row.get("temporal_consistency"),
            "terminal audit route quality differs from its audited evidence",
        )
        decode = bridge.get("decode_evidence")
        video = bridge.get("video")
        official = bridge.get("official_vbench_long_evidence")
        _require(
            isinstance(decode, Mapping)
            and isinstance(decode.get("path"), str)
            and isinstance(video, Mapping)
            and isinstance(video.get("path"), str)
            and isinstance(official, Mapping)
            and isinstance(official.get("path"), str),
            "terminal audit route omitted VBench decode/video/official evidence",
        )
        official_value = _load_json_object(
            _safe_run_file(
                run_dir,
                str(official["path"]),
                label="terminal-audit official VBench measurement",
            ),
            label="terminal-audit official VBench measurement",
        )
        official_result = official_value.get("official_result")
        _require(
            official_value.get("schema")
            in {
                VBENCH_LONG_OFFICIAL_MEASUREMENT_SCHEMA,
                VBENCH_LONG_BATCH_MEASUREMENT_SCHEMA,
            }
            and official_value.get("measurement_id") == measurement_id
            and isinstance(official_result, Mapping)
            and isinstance(official_result.get("path"), str),
            "terminal audit official VBench evidence is invalid",
        )
        if official_value.get("schema") == VBENCH_LONG_BATCH_MEASUREMENT_SCHEMA:
            batch_config_binding = official_value.get("batch_config")
            _require(
                isinstance(batch_config_binding, Mapping)
                and isinstance(batch_config_binding.get("path"), str),
                "terminal audit batch measurement omitted its shared config",
            )
            config_relative = str(batch_config_binding["path"])
        else:
            config_relative = f"vbench-long/{measurement_id}/config.json"
        official_config = _load_json_object(
            _safe_run_file(
                run_dir, config_relative, label="terminal-audit official VBench config"
            ),
            label="terminal-audit official VBench config",
        )
        staging_value = official_value.get("staging")
        staged_relative = (
            staging_value.get("staged_video")
            if official_value.get("schema") == VBENCH_LONG_BATCH_MEASUREMENT_SCHEMA
            and isinstance(staging_value, Mapping)
            else official_config.get("staged_video")
        )
        _require(
            isinstance(staged_relative, str) and bool(staged_relative),
            "terminal audit official VBench config omitted its staged video",
        )
        _require(
            isinstance(preflight_result.vbench_checkout, Mapping)
            and isinstance(preflight_result.vbench_offline_assets, Mapping),
            "terminal audit omitted trusted VBench preflight bindings",
        )
        parsed_scores = validate_shard_vbench_route(
            run_dir=run_dir,
            measurement_id=measurement_id,
            prompt_id=str(row["prompt_id"]),
            video_path=_safe_run_file(run_dir, str(video["path"]), label="terminal-audit decoded video"),
            config_path=_safe_run_file(run_dir, config_relative, label="terminal-audit VBench config"),
            result_path=_safe_run_file(run_dir, str(official_result["path"]), label="terminal-audit VBench result"),
            official=official_value,
            config=official_config,
            expected_device=(
                _deferred_vbench_scorer_devices(preflight_result)
                if preflight_result.config.schema
                == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA
                else preflight_result.visible_devices[
                    preflight_result.config.draft_gpu_slot
                ]
            ),
            expected_checkout=preflight_result.vbench_checkout,
            expected_assets=preflight_result.vbench_offline_assets,
            expected_scorer_id=expected_scorer_by_measurement.get(measurement_id),
            expected_batch_measurement_ids=(
                tuple(
                    sorted(
                        item
                        for item, scorer in expected_scorer_by_measurement.items()
                        if scorer
                        == expected_scorer_by_measurement.get(measurement_id)
                    )
                )
                if official_value.get("schema")
                == VBENCH_LONG_BATCH_MEASUREMENT_SCHEMA
                else None
            ),
        )
        _require(
            isinstance(bridge_scores, Mapping)
            and bridge_scores.get("dimension_scores") == parsed_scores
            and official_value.get("dimension_scores") == parsed_scores
            and math.fsum(parsed_scores.values()) / len(parsed_scores) == row["full_quality"],
            "terminal audit official VBench detail scores differ from route quality",
        )
        route_receipts.append(
            {
                "event_id": event_id,
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "budget": row["budget"],
                "strategy": row["strategy"],
                "route_repeat": row.get("route_repeat"),
                "rollout_id": route_id,
                "route_key": route_key,
                "selected_chunks": row["selected_chunks"],
                "proposed_target_chunks": row["proposed_target_chunks"],
                "budgeted_target_calls": row["budgeted_target_calls"],
                "full_quality": row["full_quality"],
                "temporal_consistency": row["temporal_consistency"],
                "final_history_hash": row["final_history_hash"],
                "target_successful_attempts": successful_attempts,
                "row_sha256": hashlib.sha256(_canonical_json_bytes(row)).hexdigest(),
                "evidence": {
                    "rollout_score": _receipt_file_binding(
                        run_dir, score_relative, label="terminal-audit rollout score"
                    ),
                    "quality_measurement": _receipt_file_binding(
                        run_dir, bridge_relative, label="terminal-audit VBench bridge"
                    ),
                    "decode_evidence": _receipt_file_binding(
                        run_dir,
                        str(decode["path"]),
                        label="terminal-audit decode evidence",
                    ),
                    "decoded_video": _receipt_file_binding(
                        run_dir,
                        str(video["path"]),
                        label="terminal-audit decoded video",
                    ),
                    "official_measurement": _receipt_file_binding(
                        run_dir,
                        str(official["path"]),
                        label="terminal-audit official measurement",
                    ),
                    "official_config": _receipt_file_binding(
                        run_dir,
                        config_relative,
                        label="terminal-audit official config",
                    ),
                    "official_result": _receipt_file_binding(
                        run_dir,
                        str(official_result["path"]),
                        label="terminal-audit official result",
                    ),
                    "staged_video": _receipt_file_binding(
                        run_dir,
                        staged_relative,
                        label="terminal-audit staged video",
                    ),
                },
            }
        )

    _require(
        {str(row["event_id"]) for row in ordered_rows}
        == {
            scope_id
            for scope_id, attempts in target_attempts_by_scope.items()
            if attempts and attempts[0]["phase"] in {"route_materialization", "selection_search"}
        },
        "terminal Target route scopes do not match the raw shard rows",
    )

    receipt = {
        "schema": PROMPT_SHARD_TERMINAL_AUDIT_SCHEMA,
        "status": "passed",
        "phase": "2b-eval",
        "source_revision": revision,
        "protocol_id": str(preflight_result.protocol.raw["protocol_id"]),
        "prompt_shard": _prompt_shard_metadata(shard),
        "config": {
            "path": preflight_result.config.path.relative_to(
                preflight_result.repo_root
            ).as_posix(),
            "sha256": _sha256_file(preflight_result.config.path),
        },
        "protocol": {
            "path": preflight_result.config.protocol_path.relative_to(
                preflight_result.repo_root
            ).as_posix(),
            "sha256": _sha256_file(preflight_result.config.protocol_path),
        },
        "draft_artifact_spec_sha256": preflight_result.draft_spec_sha256,
        "target_artifact_spec_sha256": preflight_result.target_spec_sha256,
        "quality": _long_video_evaluator_contract(
            phase="2b-eval", metric_id=VBENCH_LONG_PRIMARY_QUALITY_METRIC
        ),
        "raw": {
            "csv": _receipt_file_binding(
                run_dir, expected_raw_csv, label="terminal-audit raw CSV"
            ),
            "json": _receipt_file_binding(
                run_dir, expected_raw_json, label="terminal-audit raw JSON"
            ),
            "row_count": len(route_receipts),
        },
        "bindings": bindings,
        "routes": route_receipts,
    }
    relative = "logs/prompt-shard-terminal-audit.json"
    path = run_dir / relative
    _write_json(path, receipt)
    return _receipt_file_binding(run_dir, relative, label="prompt-shard terminal audit")


def _evidence_context(value: Any, *, label: str) -> GenerationContext:
    """Strictly reconstruct a persisted generation context."""

    _require(isinstance(value, dict), f"{label} must be an object")
    expected = {
        "role",
        "model_id",
        "model_artifact_sha256",
        "history_head_sha256",
        "next_chunk_index",
        "noise",
        "kv",
    }
    _require(set(value) == expected, f"{label} has unexpected or missing fields")
    try:
        return GenerationContext(
            role=value["role"],
            model_id=value["model_id"],
            model_artifact_sha256=value["model_artifact_sha256"],
            history_head_sha256=value["history_head_sha256"],
            next_chunk_index=value["next_chunk_index"],
            noise=InitialNoiseDescriptor.from_dict(
                value["noise"], path=f"{label}.noise"
            ),
            kv=KVCacheDescriptor.from_dict(value["kv"], path=f"{label}.kv"),
        )
    except CorrectionStateError as error:
        raise Insight2RunError(f"invalid {label}: {error}") from error


def _audit_single_point_evidence(
    *,
    run_dir: Path,
    row: Mapping[str, Any],
    verified_latents: dict[str, tuple[str, int]],
) -> tuple[dict[str, Any], str]:
    """Reconstruct and validate one complete paired 2(a) fork."""

    evidence_sha256 = row.get("fork_evidence_sha256")
    _require(
        isinstance(evidence_sha256, str)
        and _SHA256_RE.fullmatch(evidence_sha256) is not None,
        "2(a) row has invalid fork_evidence_sha256",
    )
    evidence_path = _safe_run_file(
        run_dir,
        row.get("fork_evidence_path"),
        label="2(a) fork_evidence_path",
    )
    _require(
        _sha256_file(evidence_path) == evidence_sha256,
        "2(a) fork evidence SHA-256 mismatch",
    )
    evidence = _load_json_object(evidence_path, label="2(a) fork evidence")
    expected_top = {
        "schema",
        "protocol_id",
        "event_id",
        "split",
        "prompt_id",
        "trajectory_seed",
        "chunk_index",
        "condition_sha256",
        "prefix",
        "initial_noise",
        "draft_branch",
        "target_branch",
    }
    _require(set(evidence) == expected_top, "2(a) fork evidence fields are invalid")
    _require(
        evidence.get("schema") == FORK_EVIDENCE_SCHEMA,
        "2(a) fork evidence schema mismatch",
    )
    for evidence_field, row_field in (
        ("event_id", "event_id"),
        ("split", "split"),
        ("prompt_id", "prompt_id"),
        ("trajectory_seed", "seed"),
        ("chunk_index", "chunk_index"),
        ("condition_sha256", "condition_hash"),
    ):
        _require(
            evidence.get(evidence_field) == row.get(row_field),
            f"2(a) fork evidence {evidence_field} disagrees with its raw row",
        )
    _require(
        isinstance(evidence.get("protocol_id"), str) and bool(evidence["protocol_id"]),
        "2(a) fork evidence protocol_id is invalid",
    )

    try:
        prefix = LatentHistory.from_dict(evidence["prefix"], path="$.fork.prefix")
    except CorrectionStateError as error:
        raise Insight2RunError(f"invalid 2(a) fork prefix: {error}") from error
    _require(prefix.prompt_id == row.get("prompt_id"), "2(a) fork prefix prompt mismatch")
    _require(prefix.trajectory_seed == row.get("seed"), "2(a) fork prefix seed mismatch")
    _require(
        prefix.next_chunk_index == row.get("chunk_index"),
        "2(a) fork prefix length mismatch",
    )
    _require(prefix.head_sha256 == row.get("prefix_hash"), "2(a) fork prefix hash mismatch")

    initial_noise = evidence["initial_noise"]
    _require(
        isinstance(initial_noise, dict)
        and set(initial_noise) == {"descriptor", "tensor_path"},
        "2(a) fork initial-noise evidence is invalid",
    )
    try:
        noise = InitialNoiseDescriptor.from_dict(
            initial_noise["descriptor"], path="$.fork.initial_noise.descriptor"
        )
    except CorrectionStateError as error:
        raise Insight2RunError(f"invalid 2(a) initial-noise descriptor: {error}") from error
    _require(noise.prompt_id == row.get("prompt_id"), "2(a) fork noise prompt mismatch")
    _require(noise.trajectory_seed == row.get("seed"), "2(a) fork noise seed mismatch")
    _require(noise.chunk_index == row.get("chunk_index"), "2(a) fork noise chunk mismatch")
    _require(noise.tensor_sha256 == row.get("noise_hash"), "2(a) fork noise hash mismatch")
    noise_path = _safe_run_file(
        run_dir,
        initial_noise["tensor_path"],
        label="2(a) initial-noise tensor_path",
    )
    _require(
        _sha256_file(noise_path) == noise.tensor_sha256
        and noise_path.stat().st_size == noise.tensor_nbytes,
        "2(a) initial-noise tensor bytes disagree with the descriptor",
    )

    def verify_segment_file(segment: LatentSegment) -> None:
        identity = (segment.latent_sha256, segment.latent_nbytes)
        prior = verified_latents.get(segment.latent_path)
        if prior is not None:
            _require(prior == identity, "one latent path has conflicting evidence")
            return
        try:
            segment.verify_file(run_dir)
        except CorrectionStateError as error:
            raise Insight2RunError(f"2(a) latent file verification failed: {error}") from error
        verified_latents[segment.latent_path] = identity

    for segment in prefix.segments:
        verify_segment_file(segment)

    branch_objects: dict[str, tuple[GenerationContext, LatentSegment, LatentHistory]] = {}
    for branch_name, expected_role in (("draft_branch", "draft"), ("target_branch", "target")):
        branch = evidence[branch_name]
        _require(
            isinstance(branch, dict)
            and set(branch) == {"context", "segment", "history"},
            f"2(a) {branch_name} evidence is invalid",
        )
        context = _evidence_context(branch["context"], label=f"2(a) {branch_name}.context")
        _require(context.role == expected_role, f"2(a) {branch_name} has the wrong model role")
        _require(
            context.noise.descriptor_sha256 == noise.descriptor_sha256,
            f"2(a) {branch_name} does not bind the shared initial noise",
        )
        context.kv.verify_for(
            role=context.role,
            model_id=context.model_id,
            model_artifact_sha256=context.model_artifact_sha256,
            history_head_sha256=prefix.head_sha256,
        )
        try:
            context.verify_history(prefix)
            segment = LatentSegment.from_dict(
                branch["segment"], path=f"$.fork.{branch_name}.segment"
            )
            segment.verify_context(prefix, context)
            history = LatentHistory.from_dict(
                branch["history"], path=f"$.fork.{branch_name}.history"
            )
        except CorrectionStateError as error:
            raise Insight2RunError(f"invalid 2(a) {branch_name}: {error}") from error
        verify_segment_file(segment)
        expected_history = prefix.append(segment)
        _require(
            history.to_dict() == expected_history.to_dict(),
            f"2(a) {branch_name} history is not exactly prefix plus its segment",
        )
        for history_segment in history.segments:
            verify_segment_file(history_segment)
        branch_objects[branch_name] = (context, segment, history)

    draft_context, draft_segment, _draft_history = branch_objects["draft_branch"]
    target_context, target_segment, _target_history = branch_objects["target_branch"]
    _require(
        draft_context.kv.cache_sha256 == row.get("draft_kv_hash"),
        "2(a) Draft KV witness disagrees with its raw row",
    )
    _require(
        target_context.kv.cache_sha256 == row.get("target_kv_hash"),
        "2(a) Target KV witness disagrees with its raw row",
    )
    _require(
        target_context.noise.descriptor_sha256
        == row.get("target_noise_descriptor_sha256"),
        "2(a) Target noise descriptor disagrees with its raw row",
    )
    _require(
        draft_segment.segment_sha256 == row.get("draft_segment_hash"),
        "2(a) Draft segment disagrees with its raw row",
    )
    _require(
        target_segment.segment_sha256 == row.get("target_segment_hash"),
        "2(a) Target segment disagrees with its raw row",
    )
    _require(
        target_segment.latent_path == row.get("target_latent_path"),
        "2(a) Target latent path disagrees with its raw row",
    )
    _require(
        target_segment.latent_sha256 == row.get("target_latent_sha256"),
        "2(a) Target latent SHA-256 disagrees with its raw row",
    )
    return (
        {
            "draft_kv": draft_context.kv.cache_sha256,
            "target_kv": target_context.kv.cache_sha256,
            "prefix": prefix.head_sha256,
            "noise": noise.tensor_sha256,
            "noise_descriptor": noise.descriptor_sha256,
            "noise_descriptor_payload": noise.to_dict(),
            "noise_derived_seed": noise.derived_seed,
            "noise_path": initial_noise["tensor_path"],
            "target_model_id": target_context.model_id,
            "target_model_artifact_sha256": target_context.model_artifact_sha256,
            "target_latent_path": target_segment.latent_path,
            "target_latent_sha256": target_segment.latent_sha256,
            "target_segment_sha256": target_segment.segment_sha256,
            "prefix_latent_paths": [item.latent_path for item in prefix.segments],
            "evidence": evidence_sha256,
        },
        evidence_sha256,
    )


def _audit_target_rpc_diagnostics(
    *,
    run_dir: Path,
    attempt_id: str,
    completed: Mapping[str, Any],
    success: Mapping[str, Any],
    protocol_id: str,
    prompt_bindings: set[tuple[str, str]],
    target_model_binding: tuple[str, str],
    chunk_count: int,
    verified_noise_files: dict[str, tuple[str, int]],
    verified_latent_files: dict[str, tuple[str, int]],
) -> tuple[dict[str, Any], str]:
    """Load and independently revalidate one immutable Target diagnostics file."""

    expected_relative = f"provenance/target-rpc-diagnostics/{attempt_id}.json"
    _require(
        success.get("diagnostics_path") == expected_relative,
        "Target success event diagnostics path is not attempt-bound",
    )
    expected_sha256 = success.get("diagnostics_sha256")
    _require(
        isinstance(expected_sha256, str)
        and _SHA256_RE.fullmatch(expected_sha256) is not None,
        "Target success event diagnostics SHA-256 is invalid",
    )
    path = _safe_run_file(
        run_dir,
        expected_relative,
        label="Target RPC diagnostics",
    )
    _require(
        _sha256_file(path) == expected_sha256,
        "Target RPC diagnostics SHA-256 disagrees with the success event",
    )
    diagnostics = _load_json_object(path, label="Target RPC diagnostics")
    _require(
        set(diagnostics) == _TARGET_RPC_DIAGNOSTICS_KEYS,
        "Target RPC diagnostics fields are invalid",
    )
    _require(
        diagnostics.get("schema") == TARGET_RPC_DIAGNOSTICS_SCHEMA,
        "Target RPC diagnostics schema drifted",
    )
    try:
        canonical_file = _canonical_json_bytes(diagnostics) + b"\n"
        actual_file = path.read_bytes()
    except OSError as error:
        raise Insight2RunError(f"cannot reread Target RPC diagnostics: {error}") from error
    _require(
        actual_file == canonical_file,
        "Target RPC diagnostics file is not canonical JSON",
    )
    for field in (
        "attempt_id",
        "logical_call_id",
        "model_artifact_sha256",
        "prompt_sha256",
        "history_head_sha256",
        "noise_descriptor_sha256",
        "latent_sha256",
        "segment_sha256",
        "kv_cache_sha256",
    ):
        value = diagnostics.get(field)
        _require(
            isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
            f"Target RPC diagnostics {field} is invalid",
        )
    scope_id = diagnostics.get("scope_id")
    _require(
        isinstance(scope_id, str) and _SHA256_RE.fullmatch(scope_id) is not None,
        "Target RPC diagnostics scope_id is invalid",
    )
    _require(
        diagnostics.get("phase") in TARGET_RPC_PHASES - {"unscoped"},
        "Target RPC diagnostics phase is unscoped or invalid",
    )
    _require(
        isinstance(diagnostics.get("model_id"), str)
        and bool(diagnostics["model_id"]),
        "Target RPC diagnostics model_id is invalid",
    )
    _require(
        (
            str(diagnostics["model_id"]),
            str(diagnostics["model_artifact_sha256"]),
        )
        == target_model_binding,
        "Target RPC diagnostics model binding differs from the campaign manifest",
    )
    _require(
        isinstance(diagnostics.get("prompt_id"), str)
        and bool(diagnostics["prompt_id"]),
        "Target RPC diagnostics prompt_id is invalid",
    )
    _require(
        (str(diagnostics["prompt_id"]), str(diagnostics["prompt_sha256"]))
        in prompt_bindings,
        "Target RPC diagnostics prompt ID/hash is not in the campaign manifest",
    )
    chunk_index = diagnostics.get("chunk_index")
    _require(
        isinstance(chunk_index, int)
        and not isinstance(chunk_index, bool)
        and 0 <= chunk_index < chunk_count,
        "Target RPC diagnostics chunk_index is invalid",
    )
    internal_seed = diagnostics.get("internal_seed")
    _require(
        isinstance(internal_seed, int)
        and not isinstance(internal_seed, bool)
        and internal_seed >= 0,
        "Target RPC diagnostics internal_seed is invalid",
    )
    try:
        noise_descriptor = InitialNoiseDescriptor.from_dict(
            diagnostics.get("noise_descriptor"),
            path="$.target_rpc_diagnostics.noise_descriptor",
        )
    except CorrectionStateError as error:
        raise Insight2RunError(
            f"Target RPC diagnostics noise descriptor is invalid: {error}"
        ) from error
    _require(
        noise_descriptor.descriptor_sha256
        == diagnostics.get("noise_descriptor_sha256"),
        "Target RPC diagnostics noise descriptor SHA-256 drifted",
    )
    _require(
        noise_descriptor.namespace == NOISE_NAMESPACE
        and noise_descriptor.prompt_id == diagnostics.get("prompt_id")
        and noise_descriptor.chunk_index == chunk_index
        and noise_descriptor.derived_seed == internal_seed,
        "Target RPC diagnostics noise identity disagrees with the request",
    )
    history_paths = diagnostics.get("history_latent_paths")
    _require(
        isinstance(history_paths, list)
        and len(history_paths) == chunk_index
        and len(set(history_paths)) == len(history_paths),
        "Target RPC diagnostics history paths are invalid",
    )
    for index, value in enumerate(history_paths):
        _safe_run_file(
            run_dir,
            value,
            label=f"Target diagnostics history_latent_paths[{index}]",
        )
    noise_path = _safe_run_file(
        run_dir,
        diagnostics.get("initial_noise_path"),
        label="Target diagnostics initial_noise_path",
    )
    noise_identity = (
        noise_descriptor.tensor_sha256,
        noise_descriptor.tensor_nbytes,
    )
    prior_noise = verified_noise_files.get(str(diagnostics["initial_noise_path"]))
    if prior_noise is None:
        _require(
            _sha256_file(noise_path) == noise_identity[0]
            and noise_path.stat().st_size == noise_identity[1],
            "Target RPC diagnostics noise bytes disagree with the descriptor",
        )
        verified_noise_files[str(diagnostics["initial_noise_path"])] = noise_identity
    else:
        _require(
            prior_noise == noise_identity,
            "one Target noise path has conflicting immutable descriptors",
        )
    latent_path = _safe_run_file(
        run_dir,
        diagnostics.get("latent_path"),
        label="Target diagnostics latent_path",
    )
    latent_identity = (
        str(diagnostics["latent_sha256"]),
        latent_path.stat().st_size,
    )
    prior_latent = verified_latent_files.get(str(diagnostics["latent_path"]))
    if prior_latent is None:
        _require(
            _sha256_file(latent_path) == latent_identity[0],
            "Target RPC diagnostics latent SHA-256 disagrees with persisted bytes",
        )
        verified_latent_files[str(diagnostics["latent_path"])] = latent_identity
    else:
        _require(
            prior_latent[0] == diagnostics["latent_sha256"],
            "one Target latent path has conflicting SHA-256 identities",
        )

    lifecycle_bindings = {
        "attempt_id": "attempt_id",
        "logical_call_id": "logical_call_id",
        "scope_id": "scope_id",
        "phase": "phase",
        "model_id": "model_id",
        "model_artifact_sha256": "model_artifact_sha256",
        "prompt_sha256": "prompt_sha256",
        "chunk_index": "chunk_index",
        "internal_seed": "internal_seed",
        "history_latent_paths": "history_latent_paths",
        "initial_noise_path": "initial_noise_path",
        "latent_path": "output_latent_path",
        "latent_sha256": "output_latent_sha256",
        "kv_cache_sha256": "kv_witness_sha256",
    }
    for diagnostics_key, lifecycle_key in lifecycle_bindings.items():
        _require(
            diagnostics[diagnostics_key] == completed.get(lifecycle_key),
            f"Target RPC diagnostics {diagnostics_key} disagrees with lifecycle",
        )
    success_bindings = {
        "attempt_id": "attempt_id",
        "logical_call_id": "logical_call_id",
        "scope_id": "scope_id",
        "phase": "phase",
        "chunk_index": "chunk_index",
        "history_head_sha256": "history_head_sha256",
        "noise_descriptor_sha256": "noise_descriptor_sha256",
        "latent_path": "latent_path",
        "latent_sha256": "latent_sha256",
        "segment_sha256": "segment_sha256",
        "kv_cache_sha256": "kv_cache_sha256",
    }
    for diagnostics_key, success_key in success_bindings.items():
        _require(
            diagnostics[diagnostics_key] == success.get(success_key),
            f"Target RPC diagnostics {diagnostics_key} disagrees with success event",
        )

    backend = diagnostics.get("backend_diagnostics")
    _require(
        isinstance(backend, dict)
        and TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS <= set(backend)
        and set(backend) <= TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS | {"native_correction_state"},
        "Target backend diagnostics fields are invalid",
    )
    if "native_correction_state" in backend:
        from .krea_chunk_backend import validate_native_correction_state
        try:
            validate_native_correction_state(backend["native_correction_state"], chunk_index=chunk_index)
        except RuntimeError as error:
            raise Insight2RunError(f"invalid native Target state: {error}") from error
    kv_witness_sha256 = backend.get("kv_witness_sha256")
    _require(
        isinstance(kv_witness_sha256, str)
        and _SHA256_RE.fullmatch(kv_witness_sha256) is not None,
        "Target backend KV witness SHA-256 is invalid",
    )
    kv_witness = backend.get("kv_witness")
    try:
        validate_sampled_kv_cache_witness(
            kv_witness,
            expected_layer_count=40,
            expected_self_shape=(1, 9360, 40, 128),
            expected_cross_shape=(1, 512, 40, 128),
            expected_dtype="torch.bfloat16",
            expected_self_used_tokens=0 if chunk_index == 0 else 4680,
            expected_cross_initialized=chunk_index > 0,
        )
    except AdapterError as error:
        raise Insight2RunError(f"Target backend KV witness is invalid: {error}") from error
    _require(
        hashlib.sha256(_canonical_json_bytes(kv_witness)).hexdigest()
        == kv_witness_sha256,
        "Target backend KV witness canonical hash is invalid",
    )
    _require(
        kv_witness_sha256 == diagnostics["kv_cache_sha256"],
        "Target backend KV witness differs from the persisted KV hash",
    )
    _require(
        backend.get("explicit_initial_noise_verified") is True,
        "Target backend did not verify explicit initial noise",
    )
    full_trace = backend.get("forward_trace")
    denoising_trace = backend.get("denoising_forward_trace")
    schedule = backend.get("denoising_schedule")
    _require(isinstance(full_trace, list), "Target backend full forward trace is invalid")
    _require(
        isinstance(denoising_trace, list),
        "Target backend denoising forward trace is invalid",
    )
    _require(isinstance(schedule, list), "Target backend denoising schedule is invalid")
    try:
        expected_denoising = validate_target_forward_trace(
            full_trace,
            schedule,
            history_frames=3 * chunk_index,
            expected_schedule=MAIN_TABLE_FOUR_STEP_SCHEDULE,
        )
    except RuntimeError as error:
        raise Insight2RunError(f"Target backend forward trace is invalid: {error}") from error
    _require(
        denoising_trace == expected_denoising,
        "Target backend denoising trace differs from validated full trace",
    )
    expected_logical_call_id = hashlib.sha256(
        _canonical_json_bytes(
            [
                "target-logical-call-v2",
                protocol_id,
                diagnostics["model_id"],
                diagnostics["model_artifact_sha256"],
                diagnostics["prompt_id"],
                diagnostics["prompt_sha256"],
                diagnostics["history_head_sha256"],
                diagnostics["chunk_index"],
                diagnostics["noise_descriptor_sha256"],
            ]
        )
    ).hexdigest()
    _require(
        diagnostics["logical_call_id"] == expected_logical_call_id,
        "Target RPC diagnostics logical call ID is not its scientific identity",
    )
    # Keep only compact bindings in memory. A 40-layer witness carries more
    # than ten thousand sampled values, and a multi-budget Greedy run can
    # legitimately produce thousands of physical Target attempts.
    compact = {
        key: value for key, value in diagnostics.items() if key != "backend_diagnostics"
    }
    compact["backend_diagnostics_sha256"] = hashlib.sha256(
        _canonical_json_bytes(backend)
    ).hexdigest()
    return compact, expected_relative


def _load_audited_checkpoint(
    path: Path, *, label: str
) -> tuple[dict[str, Any], LatentHistory, GenerationContext, LatentSegment, dict[str, Any]]:
    """Load one strict campaign checkpoint without trusting its filename."""

    value = _load_json_object(path, label=label)
    _require(
        set(value) == {"schema", "history", "generated", "metadata"}
        and value.get("schema") == CHECKPOINT_SCHEMA,
        f"{label} fields or schema are invalid",
    )
    generated = value.get("generated")
    _require(
        isinstance(generated, dict)
        and set(generated) == _GENERATED_RECORD_KEYS,
        f"{label} generated record is invalid",
    )
    metadata = value.get("metadata")
    _require(isinstance(metadata, dict), f"{label} metadata is invalid")
    try:
        history = LatentHistory.from_dict(value.get("history"), path=f"$.{label}.history")
        context = _evidence_context(
            generated.get("context"), label=f"{label} generated context"
        )
        segment = LatentSegment.from_dict(
            generated.get("segment"), path=f"$.{label}.generated.segment"
        )
        _require(
            history.next_chunk_index >= 1,
            f"{label} history contains no generated chunk",
        )
        prefix = history.prefix_before(history.next_chunk_index - 1)
        segment.verify_context(prefix, context)
        context.kv.verify_for(
            role=context.role,
            model_id=context.model_id,
            model_artifact_sha256=context.model_artifact_sha256,
            history_head_sha256=prefix.head_sha256,
        )
    except CorrectionStateError as error:
        raise Insight2RunError(f"invalid {label}: {error}") from error
    _require(
        segment.to_dict() == history.segments[-1].to_dict(),
        f"{label} latest segment differs from its history",
    )
    return value, history, context, segment, metadata


def _audit_trusted_run_inputs(
    *,
    run_dir: Path,
    phase: str,
    campaign_manifest: Mapping[str, Any],
    preflight_result: Preflight,
) -> None:
    """Cross-bind mutable run evidence to the in-memory frozen preflight.

    Run-directory logs are useful evidence, but they are not an independent
    trust root.  A formal execution therefore supplies the already verified
    :class:`Preflight` object and this audit requires the persisted preflight
    and campaign manifests to equal the facts derived from it exactly.
    """

    _require(
        run_dir.resolve(strict=True) == preflight_result.run_dir.resolve(strict=True)
        and phase == preflight_result.phase,
        "terminal audit does not match its trusted preflight run or phase",
    )
    protocol = preflight_result.protocol
    config = preflight_result.config
    chunk_count, _rgb_frame_count, _duration_exact, _duration_decimal = (
        _protocol_geometry(protocol)
    )
    phase_seeds = _phase_seeds(protocol, config, split=preflight_result.split)
    _require(
        _sha256_file(config.protocol_path) == preflight_result.protocol_sha256
        and _sha256_file(config.draft_artifact_spec_path)
        == preflight_result.draft_spec_sha256
        and target_artifact_spec_sha256(config.target_artifact_spec_path)
        == preflight_result.target_spec_sha256,
        "a frozen terminal-audit input changed after preflight",
    )
    expected_preflight = _preflight_log_value(preflight_result)
    preflight_path = _safe_run_file(
        run_dir, "logs/preflight.json", label="trusted preflight log"
    )
    persisted_preflight = _load_json_object(
        preflight_path, label="trusted preflight log"
    )
    _require(
        persisted_preflight == expected_preflight
        and preflight_path.read_bytes()
        == _canonical_json_bytes(expected_preflight) + b"\n",
        "persisted preflight differs from the trusted in-memory preflight",
    )

    draft_artifacts = {
        str(row["name"]): row for row in preflight_result.draft_verified_artifacts
    }
    target_artifacts = {
        str(row["name"]): row for row in preflight_result.target_verified_artifacts
    }
    _require(
        "self_forcing_checkpoint" in draft_artifacts
        and "krea_14b_checkpoint" in target_artifacts,
        "trusted preflight omitted a required generation checkpoint",
    )
    calibration = _prompt_cases(protocol.calibration_prompts, split="calibration")
    evaluation = _prompt_cases(
        _selected_evaluation_prompt_values(preflight_result), split="evaluation"
    )
    smoke = _prompt_cases(
        _selected_smoke_prompt_values(preflight_result), split="evaluation"
    )
    expected_campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "protocol_id": str(protocol.raw["protocol_id"]),
        "chunk_count": chunk_count,
        "seeds": list(phase_seeds),
        "noise_namespace": NOISE_NAMESPACE,
        "models": {
            "draft": {
                "id": str(protocol.raw["models"]["draft"]["id"]),
                "artifact_sha256": draft_artifacts["self_forcing_checkpoint"][
                    "sha256"
                ],
            },
            "target": {
                "id": str(protocol.raw["models"]["target"]["id"]),
                "artifact_sha256": target_artifacts["krea_14b_checkpoint"][
                    "sha256"
                ],
            },
        },
        "calibration_prompts": [asdict(item) for item in calibration],
        "evaluation_prompts": [asdict(item) for item in evaluation],
        "smoke_prompts": [asdict(item) for item in smoke],
    }
    campaign_path = _safe_run_file(run_dir, "campaign.json", label="campaign manifest")
    _require(
        campaign_manifest == expected_campaign
        and campaign_path.read_bytes()
        == _canonical_json_bytes(expected_campaign) + b"\n",
        "campaign manifest differs from the trusted frozen protocol or artifacts",
    )


@lru_cache(maxsize=16)
def _audit_shared_conditioning_files(
    run_dir_text: str,
    manifest_relative: str,
    manifest_sha256: str,
    tensor_relative: str,
    tensor_sha256: str,
    tensor_nbytes: int,
    prompt_set_sha256_value: str,
) -> None:
    """Re-hash the sealed cache once during terminal quality audit."""

    run_dir = Path(run_dir_text).resolve(strict=True)
    manifest_path = _safe_run_file(
        run_dir, manifest_relative, label="shared conditioning manifest"
    )
    _require(
        _sha256_file(manifest_path) == manifest_sha256,
        "shared conditioning manifest bytes drifted after model startup",
    )
    manifest = _load_json_object(
        manifest_path, label="shared conditioning manifest"
    )
    _require(
        manifest_path.read_bytes() == _canonical_json_bytes(manifest) + b"\n"
        and manifest.get("schema")
        == "insight2_shared_umt5_conditioning_manifest_v1"
        and manifest.get("prompt_set_sha256") == prompt_set_sha256_value,
        "shared conditioning manifest is non-canonical or binds another prompt set",
    )
    tensor_record = manifest.get("tensor_file")
    _require(
        isinstance(tensor_record, Mapping)
        and tensor_record.get("path") == tensor_relative
        and tensor_record.get("sha256") == tensor_sha256
        and tensor_record.get("nbytes") == tensor_nbytes,
        "shared conditioning tensor binding differs from model state",
    )
    tensor_path = _safe_run_file(
        run_dir, tensor_relative, label="shared conditioning tensor file"
    )
    _require(
        tensor_path.stat().st_size == tensor_nbytes
        and _sha256_file(tensor_path) == tensor_sha256,
        "shared conditioning tensor bytes drifted after model startup",
    )


def _load_quality_model_binding(
    *,
    run_dir: Path,
    phase: str,
    draft_model_binding: tuple[str, str],
) -> dict[str, str]:
    """Bind quality measurements to the actually loaded Draft/VAE/MUSIQ bytes."""

    preflight_path = _safe_run_file(
        run_dir, "logs/preflight.json", label="preflight log"
    )
    preflight = _load_json_object(preflight_path, label="preflight log")
    _require(
        preflight.get("schema") == PREFLIGHT_SCHEMA
        and preflight.get("phase") == phase
        and preflight.get("status") == "passed",
        "preflight log schema, phase, or status drifted",
    )
    _require(
        preflight_path.read_bytes() == _canonical_json_bytes(preflight) + b"\n",
        "preflight log is not canonical JSON",
    )
    artifact_rows = preflight.get("draft_artifacts")
    _require(isinstance(artifact_rows, list), "preflight Draft artifacts are invalid")
    artifacts: dict[str, Mapping[str, Any]] = {}
    for row in artifact_rows:
        _require(
            isinstance(row, Mapping)
            and isinstance(row.get("name"), str)
            and bool(row["name"])
            and isinstance(row.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(row["sha256"])) is not None,
            "preflight Draft artifact entry is invalid",
        )
        name = str(row["name"])
        _require(name not in artifacts, "preflight Draft artifact names are not unique")
        artifacts[name] = row

    path = _safe_run_file(run_dir, "logs/model-state.json", label="model-state log")
    value = _load_json_object(path, label="model-state log")
    _require(
        set(value) == {"schema", "phase", "models"}
        and value.get("schema") == MODEL_STATE_SCHEMA
        and value.get("phase") == phase,
        "model-state log fields, schema, or phase drifted",
    )
    _require(
        path.read_bytes() == _canonical_json_bytes(value) + b"\n",
        "model-state log is not canonical JSON",
    )
    models = value.get("models")
    draft = models.get("draft") if isinstance(models, Mapping) else None
    target = models.get("target") if isinstance(models, Mapping) else None
    _require(isinstance(draft, Mapping), "model-state log has no Draft state")
    _require(
        draft.get("model_id") == draft_model_binding[0]
        and draft.get("checkpoint_sha256") == draft_model_binding[1]
        and draft.get("artifact_spec_sha256")
        == preflight.get("draft_artifact_spec_sha256")
        and draft.get("quality_metric") == PRIMARY_QUALITY_METRIC,
        "quality model state differs from the campaign Draft binding",
    )
    acceleration = preflight.get("runtime_acceleration")
    if acceleration is not None:
        _require(
            acceleration
            == {
                "shared_precomputed_umt5_conditioning": True,
                "allow_draft_target_same_gpu": True,
            }
            or acceleration
            == {
                "shared_precomputed_umt5_conditioning": True,
                "allow_draft_target_same_gpu": False,
            },
            "preflight runtime acceleration audit is invalid",
        )
        draft_conditioning = draft.get("conditioning_cache")
        target_conditioning = (
            target.get("conditioning_cache") if isinstance(target, Mapping) else None
        )
        _require(
            isinstance(draft_conditioning, Mapping)
            and draft_conditioning == target_conditioning,
            "terminal audit found different Draft/Target conditioning bindings",
        )
        required_fields = {
            "manifest_path": str,
            "manifest_sha256": str,
            "tensor_file_path": str,
            "tensor_file_sha256": str,
            "tensor_file_nbytes": int,
            "prompt_set_sha256": str,
        }
        _require(
            all(
                isinstance(draft_conditioning.get(field), expected_type)
                and not isinstance(draft_conditioning.get(field), bool)
                for field, expected_type in required_fields.items()
            ),
            "terminal conditioning binding is incomplete",
        )
        _audit_shared_conditioning_files(
            str(run_dir),
            str(draft_conditioning["manifest_path"]),
            str(draft_conditioning["manifest_sha256"]),
            str(draft_conditioning["tensor_file_path"]),
            str(draft_conditioning["tensor_file_sha256"]),
            int(draft_conditioning["tensor_file_nbytes"]),
            str(draft_conditioning["prompt_set_sha256"]),
        )
    result: dict[str, str] = {
        "model_id": draft_model_binding[0],
        "model_artifact_sha256": draft_model_binding[1],
    }
    artifact_for_field = {
        "checkpoint_sha256": "self_forcing_checkpoint",
        "musiq_spaq_sha256": "musiq_spaq_checkpoint",
        "vae_sha256": "wan_vae_checkpoint",
    }
    for field, artifact_name in artifact_for_field.items():
        item = draft.get(field)
        _require(
            isinstance(item, str) and _SHA256_RE.fullmatch(item) is not None,
            f"quality model state {field} is invalid",
        )
        _require(
            artifact_name in artifacts
            and item == artifacts[artifact_name].get("sha256"),
            f"quality model state {field} disagrees with artifact preflight",
        )
        result[field] = item
    return result


@lru_cache(maxsize=1)
def _validated_vbench_assets_for_terminal_audit() -> Mapping[str, Any]:
    """Hash the 3.7-GB bundle once, then reuse that exact binding per route."""

    try:
        return validate_vbench_long_offline_assets(
            VBENCH_LONG_ASSET_MANIFEST,
            spec_path=VBENCH_LONG_ASSET_SPEC,
        ).config_binding()
    except BaseException as error:
        raise Insight2RunError(
            f"VBench-Long offline asset terminal audit failed: {error}"
        ) from error


@lru_cache(maxsize=1)
def _validated_vbench_checkout_for_terminal_audit() -> Mapping[str, Any]:
    """Revalidate the frozen checkout at terminal audit time."""

    try:
        return validate_vbench_long_checkout(
            VBENCH_LONG_CHECKOUT
        ).config_binding()
    except BaseException as error:
        raise Insight2RunError(
            f"VBench-Long checkout terminal audit failed: {error}"
        ) from error


def _audit_vbench_long_quality_measurement(
    *,
    run_dir: Path,
    measurement_id: str,
    history: LatentHistory,
    evaluation_kind: str,
    chunk_index: int | None,
    measurement_schema: str,
    primary_metric_id: str,
    verified_decode_files: dict[str, str],
) -> tuple[float, float]:
    """Audit the bridge sidecar plus the retained official VBench evidence."""

    _require(
        evaluation_kind == "full" and chunk_index is None,
        "VBench-Long quality is only valid for a complete-video measurement",
    )
    _require(
        measurement_schema == VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA
        and primary_metric_id == VBENCH_LONG_PRIMARY_QUALITY_METRIC,
        "VBench-Long measurement schema or metric identity is invalid",
    )
    relative = f"measurements/vbench-long-quality/{measurement_id}.json"
    path = _safe_run_file(
        run_dir, relative, label="VBench-Long quality measurement sidecar"
    )
    value = _load_json_object(path, label="VBench-Long quality measurement sidecar")
    _require(
        path.read_bytes() == _canonical_json_bytes(value) + b"\n",
        "VBench-Long quality sidecar is not canonical JSON",
    )
    _require(
        set(value)
        == {
            "schema",
            "measurement_id",
            "metric_id",
            "history",
            "decode_evidence",
            "video",
            "official_vbench_long_evidence",
            "dimensions",
            "aggregation",
            "scores",
        }
        and value.get("schema") == measurement_schema
        and value.get("measurement_id") == measurement_id
        and value.get("metric_id") == primary_metric_id,
        "VBench-Long quality sidecar fields or identity are invalid",
    )
    _require(
        value.get("history")
        == {
            "manifest_sha256": history.manifest_sha256,
            "head_sha256": history.head_sha256,
            "prompt_id": history.prompt_id,
            "latent_chunk_count": history.next_chunk_index,
        },
        "VBench-Long quality history differs from the scored trajectory",
    )

    frame_count = 1 + 4 * (3 * history.next_chunk_index - 1)
    decode_binding = value.get("decode_evidence")
    canonical_decode_relative = f"decoded/{history.manifest_sha256}.json"
    measurement_decode_relative = f"decoded/vbench-long/{measurement_id}.json"
    decode_relative = (
        str(decode_binding.get("path"))
        if isinstance(decode_binding, Mapping)
        else ""
    )
    _require(
        isinstance(decode_binding, Mapping)
        and set(decode_binding) == {"path", "sha256"}
        and decode_relative
        in {canonical_decode_relative, measurement_decode_relative}
        and isinstance(decode_binding.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(decode_binding["sha256"])) is not None,
        "VBench-Long decode binding is invalid",
    )
    decode_path = _safe_run_file(
        run_dir, decode_relative, label="VBench-Long decode evidence"
    )
    decode_sha256 = str(decode_binding["sha256"])
    _require(
        _sha256_file(decode_path) == decode_sha256,
        "VBench-Long decode evidence SHA-256 mismatch",
    )
    prior_decode = verified_decode_files.get(decode_relative)
    if prior_decode is None:
        verified_decode_files[decode_relative] = decode_sha256
    else:
        _require(
            prior_decode == decode_sha256,
            "one decode evidence path has conflicting immutable identities",
        )
    decode = _load_json_object(decode_path, label="VBench-Long decode evidence")
    _require(
        decode_path.read_bytes() == _canonical_json_bytes(decode) + b"\n",
        "VBench-Long decode evidence is not canonical JSON",
    )
    frames = decode.get("frames")
    video = value.get("video")
    _require(
        set(decode)
        == {
            "schema",
            "history_manifest_sha256",
            "latent_chunk_count",
            "expected_frames",
            "frames",
            "video",
        }
        and decode.get("schema") == DECODE_EVIDENCE_SCHEMA
        and decode.get("history_manifest_sha256") == history.manifest_sha256
        and decode.get("latent_chunk_count") == history.next_chunk_index
        and decode.get("expected_frames") == frame_count
        and decode.get("video") == video,
        "VBench-Long decode evidence differs from its scored history or video",
    )
    _require(
        isinstance(frames, Mapping)
        and set(frames) == {"path", "key", "sha256", "nbytes", "shape", "dtype"}
        and frames.get("path") == f"decoded/{history.manifest_sha256}.safetensors"
        and frames.get("key") == "frames"
        and isinstance(frames.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(frames["sha256"])) is not None
        and isinstance(frames.get("nbytes"), int)
        and not isinstance(frames.get("nbytes"), bool)
        and int(frames["nbytes"]) > 0
        and frames.get("shape") == [frame_count, 3, 480, 832]
        and frames.get("dtype") == "uint8",
        "VBench-Long decoded-frame metadata violates the VAE output contract",
    )
    video_relative = f"videos/{measurement_id}.mp4"
    _require(
        isinstance(video, Mapping)
        and set(video) == {"path", "sha256", "nbytes", "fps", "frames"}
        and video.get("path") == video_relative
        and isinstance(video.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(video["sha256"])) is not None
        and isinstance(video.get("nbytes"), int)
        and not isinstance(video.get("nbytes"), bool)
        and int(video["nbytes"]) > 0
        and video.get("fps") == 24
        and video.get("frames") == frame_count,
        "VBench-Long decoded-video metadata is invalid",
    )
    video_path = _safe_run_file(
        run_dir, video_relative, label="VBench-Long decoded video"
    )
    _require(
        _sha256_file(video_path) == video["sha256"]
        and video_path.stat().st_size == video["nbytes"],
        "VBench-Long decoded video bytes differ from metadata",
    )

    dimensions = list(VBENCH_LONG_DIMENSIONS)
    aggregation = "unweighted_arithmetic_mean_of_six_official_dimension_scores"
    _require(
        value.get("dimensions") == dimensions
        and value.get("aggregation") == aggregation,
        "VBench-Long dimension grid or aggregation drifted",
    )
    scores = value.get("scores")
    _require(
        isinstance(scores, Mapping)
        and set(scores) == {"dimension_scores", "macro_mean", "motion_smoothness"},
        "VBench-Long bridge scores are invalid",
    )
    raw_dimension_scores = scores.get("dimension_scores")
    _require(
        isinstance(raw_dimension_scores, Mapping)
        and set(raw_dimension_scores) == set(VBENCH_LONG_DIMENSIONS),
        "VBench-Long bridge does not contain the exact six dimensions",
    )
    dimension_scores: dict[str, float] = {}
    for dimension in VBENCH_LONG_DIMENSIONS:
        item = raw_dimension_scores[dimension]
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and 0.0 <= float(item) <= 1.0,
            f"VBench-Long {dimension} score is invalid",
        )
        dimension_scores[dimension] = float(item)
    macro_mean = math.fsum(dimension_scores.values()) / len(VBENCH_LONG_DIMENSIONS)
    motion_smoothness = dimension_scores["motion_smoothness"]
    recorded_macro = scores.get("macro_mean")
    recorded_motion = scores.get("motion_smoothness")
    _require(
        isinstance(recorded_macro, (int, float))
        and not isinstance(recorded_macro, bool)
        and math.isfinite(float(recorded_macro))
        and float(recorded_macro) == macro_mean
        and isinstance(recorded_motion, (int, float))
        and not isinstance(recorded_motion, bool)
        and math.isfinite(float(recorded_motion))
        and float(recorded_motion) == motion_smoothness,
        "VBench-Long bridge macro mean or motion smoothness drifted",
    )

    official_binding = value.get("official_vbench_long_evidence")
    official_relative = f"vbench-long/{measurement_id}/measurement.json"
    _require(
        isinstance(official_binding, Mapping)
        and set(official_binding) == {"path", "sha256", "nbytes"}
        and official_binding.get("path") == official_relative
        and isinstance(official_binding.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(official_binding["sha256"])) is not None
        and isinstance(official_binding.get("nbytes"), int)
        and not isinstance(official_binding.get("nbytes"), bool)
        and int(official_binding["nbytes"]) > 0,
        "official VBench-Long evidence binding is invalid",
    )
    official_path = _safe_run_file(
        run_dir, official_relative, label="official VBench-Long measurement evidence"
    )
    _require(
        _sha256_file(official_path) == official_binding["sha256"]
        and official_path.stat().st_size == official_binding["nbytes"],
        "official VBench-Long evidence bytes differ from the bridge binding",
    )
    official = _load_json_object(
        official_path, label="official VBench-Long measurement evidence"
    )
    _require(
        official_path.read_bytes() == _canonical_json_bytes(official) + b"\n",
        "official VBench-Long measurement evidence is not canonical JSON",
    )
    staged_alias = vbench_long_staged_alias(
        prompt_id=history.prompt_id,
        measurement_id=measurement_id,
        source_video_sha256=str(video["sha256"]),
    )
    staging_binding = {
        "algorithm": VBENCH_LONG_STAGED_ALIAS_ALGORITHM,
        "alias": staged_alias,
        "original_prompt_id": history.prompt_id,
    }
    if official.get("schema") == VBENCH_LONG_BATCH_MEASUREMENT_SCHEMA:
        try:
            batch_measurement = validate_vbench_long_batch_measurement(
                official_path,
                run_dir=run_dir,
                measurement_id=measurement_id,
                prompt=history.prompt_id,
                video_path=video_path,
            )
        except BaseException as error:
            raise Insight2RunError(
                f"official VBench-Long batch result provenance is invalid: {error}"
            ) from error
        _require(
            batch_measurement.scores() == dimension_scores
            and batch_measurement.macro_mean == macro_mean
            and batch_measurement.motion_smoothness == motion_smoothness,
            "official VBench-Long batch detail scores differ from recorded scores",
        )
        return macro_mean, motion_smoothness
    _require(
        set(official)
        == {
            "schema",
            "measurement_id",
            "config_sha256",
            "official_result",
            "dimension_scores",
            "aggregation",
            "macro_mean",
            "motion_smoothness",
            "staging",
        }
        and official.get("schema") == VBENCH_LONG_OFFICIAL_MEASUREMENT_SCHEMA
        and official.get("measurement_id") == measurement_id
        and official.get("staging") == staging_binding
        and official.get("dimension_scores") == dimension_scores
        and official.get("aggregation") == aggregation
        and official.get("macro_mean") == macro_mean
        and official.get("motion_smoothness") == motion_smoothness
        and isinstance(official.get("config_sha256"), str)
        and _SHA256_RE.fullmatch(str(official["config_sha256"])) is not None,
        "official VBench-Long measurement JSON drifted",
    )
    official_dimension_scores = official["dimension_scores"]
    _require(
        isinstance(official_dimension_scores, Mapping)
        and all(
            isinstance(official_dimension_scores[dimension], (int, float))
            and not isinstance(official_dimension_scores[dimension], bool)
            and math.isfinite(float(official_dimension_scores[dimension]))
            and float(official_dimension_scores[dimension])
            == dimension_scores[dimension]
            for dimension in VBENCH_LONG_DIMENSIONS
        ),
        "official VBench-Long dimension score types drifted",
    )

    config_relative = f"vbench-long/{measurement_id}/config.json"
    config_path = _safe_run_file(
        run_dir, config_relative, label="official VBench-Long config evidence"
    )
    _require(
        _sha256_file(config_path) == official["config_sha256"],
        "official VBench-Long config SHA-256 mismatch",
    )
    official_config = _load_json_object(
        config_path, label="official VBench-Long config evidence"
    )
    _require(
        config_path.read_bytes() == _canonical_json_bytes(official_config) + b"\n",
        "official VBench-Long config evidence is not canonical JSON",
    )
    official_root = run_dir.resolve(strict=True) / "vbench-long" / measurement_id
    staged_relative = (
        f"vbench-long/{measurement_id}/videos/{staged_alias}.mp4"
    )
    configured_python = official_config.get("python_executable")
    _require(
        isinstance(configured_python, str)
        and PurePosixPath(configured_python).is_absolute()
        and PurePosixPath(configured_python).parent
        == PurePosixPath(VBENCH_LONG_PYTHON).parent
        and PurePosixPath(configured_python).name.startswith("python"),
        "official VBench-Long Python executable escaped the frozen environment",
    )
    checkout_binding = _validated_vbench_checkout_for_terminal_audit()
    tracked_tree = checkout_binding.get("tracked_tree")
    _require(
        isinstance(tracked_tree, Mapping)
        and set(tracked_tree) == {"sha256", "file_count", "nbytes", "algorithm"}
        and isinstance(tracked_tree.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(tracked_tree["sha256"])) is not None
        and isinstance(tracked_tree.get("file_count"), int)
        and not isinstance(tracked_tree.get("file_count"), bool)
        and int(tracked_tree["file_count"]) > 0
        and isinstance(tracked_tree.get("nbytes"), int)
        and not isinstance(tracked_tree.get("nbytes"), bool)
        and int(tracked_tree["nbytes"]) > 0
        and tracked_tree.get("algorithm")
        == VBENCH_LONG_CHECKOUT_TREE_HASH_ALGORITHM
        and official_config.get("upstream") == checkout_binding,
        "official VBench-Long checkout differs from current verified bytes",
    )
    current_launcher = Path(__file__).with_name("vbench_offline_entry.py").resolve(
        strict=True
    )
    launcher = official_config.get("launcher")
    _require(
        isinstance(launcher, Mapping)
        and set(launcher) == {"path", "sha256", "nbytes"}
        and launcher.get("path") == str(current_launcher)
        and isinstance(launcher.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(launcher["sha256"])) is not None
        and isinstance(launcher.get("nbytes"), int)
        and not isinstance(launcher.get("nbytes"), bool)
        and int(launcher["nbytes"]) > 0,
        "official VBench-Long offline launcher binding is invalid",
    )
    assets = official_config.get("offline_assets")
    asset_root = str(PurePosixPath(VBENCH_LONG_ASSET_MANIFEST).parent)
    _require(
        isinstance(assets, Mapping)
        and set(assets)
        == {"schema", "manifest", "git_spec", "root", "files", "trees"}
        and assets.get("schema") == VBENCH_LONG_OFFLINE_ASSET_SCHEMA
        and assets.get("root") == asset_root,
        "official VBench-Long offline asset binding is invalid",
    )
    asset_manifest = assets.get("manifest")
    _require(
        isinstance(asset_manifest, Mapping)
        and set(asset_manifest) == {"path", "sha256", "nbytes"}
        and asset_manifest.get("path") == VBENCH_LONG_ASSET_MANIFEST
        and isinstance(asset_manifest.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(asset_manifest["sha256"])) is not None
        and isinstance(asset_manifest.get("nbytes"), int)
        and not isinstance(asset_manifest.get("nbytes"), bool)
        and int(asset_manifest["nbytes"]) > 0,
        "official VBench-Long asset manifest binding is invalid",
    )
    asset_spec = assets.get("git_spec")
    _require(
        isinstance(asset_spec, Mapping)
        and set(asset_spec) == {"path", "sha256", "nbytes"}
        and asset_spec.get("path") == VBENCH_LONG_ASSET_SPEC
        and isinstance(asset_spec.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(asset_spec["sha256"])) is not None
        and isinstance(asset_spec.get("nbytes"), int)
        and not isinstance(asset_spec.get("nbytes"), bool)
        and int(asset_spec["nbytes"]) > 0
        and _sha256_file(Path(VBENCH_LONG_ASSET_SPEC)) == asset_spec["sha256"]
        and Path(VBENCH_LONG_ASSET_SPEC).stat().st_size == asset_spec["nbytes"],
        "official VBench-Long Git asset specification binding is invalid",
    )
    asset_files = assets.get("files")
    _require(
        isinstance(asset_files, Mapping)
        and set(asset_files) == set(VBENCH_LONG_OFFLINE_FILE_PATHS),
        "official VBench-Long offline file set drifted",
    )
    for name, relative in VBENCH_LONG_OFFLINE_FILE_PATHS.items():
        record = asset_files[name]
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "sha256", "nbytes"}
            and record.get("path") == relative
            and isinstance(record.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(record["sha256"])) is not None
            and isinstance(record.get("nbytes"), int)
            and not isinstance(record.get("nbytes"), bool)
            and int(record["nbytes"]) > 0,
            f"official VBench-Long offline file binding drifted: {name}",
        )
    asset_trees = assets.get("trees")
    _require(
        isinstance(asset_trees, Mapping)
        and set(asset_trees) == set(VBENCH_LONG_OFFLINE_TREE_PATHS),
        "official VBench-Long offline source-tree set drifted",
    )
    for name, relative in VBENCH_LONG_OFFLINE_TREE_PATHS.items():
        record = asset_trees[name]
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {"path", "sha256", "file_count", "nbytes", "algorithm"}
            and record.get("path") == relative
            and isinstance(record.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(record["sha256"])) is not None
            and isinstance(record.get("file_count"), int)
            and not isinstance(record.get("file_count"), bool)
            and int(record["file_count"]) > 0
            and isinstance(record.get("nbytes"), int)
            and not isinstance(record.get("nbytes"), bool)
            and int(record["nbytes"]) > 0
            and record.get("algorithm") == VBENCH_LONG_OFFLINE_TREE_HASH_ALGORITHM,
            f"official VBench-Long offline source-tree binding drifted: {name}",
        )
    _require(
        assets == _validated_vbench_assets_for_terminal_audit(),
        "official VBench-Long offline assets differ from current verified bytes",
    )
    try:
        launcher_path = Path(str(launcher["path"])).resolve(strict=True)
    except OSError as error:
        raise Insight2RunError(
            f"official VBench-Long offline launcher is unavailable: {error}"
        ) from error
    _require(
        launcher_path == current_launcher
        and launcher_path.is_file()
        and not launcher_path.is_symlink()
        and _sha256_file(launcher_path) == launcher["sha256"]
        and launcher_path.stat().st_size == launcher["nbytes"],
        "official VBench-Long offline launcher bytes drifted",
    )
    matched_runtime_contracts: list[
        tuple[str | None, dict[str, Any], dict[str, str]]
    ] = []
    for candidate_scorer_id in (None, "scorer-0", "scorer-1"):
        candidate_policy, candidate_environment = _shard_vbench_offline_contract(
            run_dir=run_dir,
            asset_manifest_path=VBENCH_LONG_ASSET_MANIFEST,
            scorer_id=candidate_scorer_id,
        )
        if (
            official_config.get("offline_policy") == candidate_policy
            and official_config.get("offline_environment") == candidate_environment
        ):
            matched_runtime_contracts.append(
                (candidate_scorer_id, candidate_policy, candidate_environment)
            )
    _require(
        len(matched_runtime_contracts) == 1,
        "official VBench-Long runtime cache is not a frozen base/scorer cache",
    )
    scorer_id, expected_offline_policy, expected_offline_environment = (
        matched_runtime_contracts[0]
    )
    runtime_cache_root = (
        run_dir.resolve(strict=True) / "tmp" / "vbench-long-runtime-cache"
    )
    if scorer_id is not None:
        runtime_cache_root /= scorer_id
    expected_command = [
        configured_python,
        launcher["path"],
        "--asset-manifest",
        VBENCH_LONG_ASSET_MANIFEST,
        "--asset-spec",
        VBENCH_LONG_ASSET_SPEC,
        "--checkout",
        VBENCH_LONG_CHECKOUT,
        "--eval-script",
        f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/eval_long.py",
        "--eval-sha256",
        str(checkout_binding["eval_script"]["sha256"]),
        "--eval-nbytes",
        str(checkout_binding["eval_script"]["nbytes"]),
        "--full-info",
        f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/VBench_full_info.json",
        "--full-info-sha256",
        str(checkout_binding["full_info"]["sha256"]),
        "--full-info-nbytes",
        str(checkout_binding["full_info"]["nbytes"]),
        "--runtime-cache-root",
        str(runtime_cache_root),
        "--",
        "--videos_path",
        str(official_root / "videos"),
        "--dimension",
        *VBENCH_LONG_DIMENSIONS,
        "--mode",
        "long_custom_input",
        "--dev_flag",
        "--load_ckpt_from_local",
        "True",
        "--sb_clip2clip_feat_extractor",
        VBENCH_LONG_SUBJECT_CLIP2CLIP_EXTRACTOR,
        "--bg_clip2clip_feat_extractor",
        VBENCH_LONG_BACKGROUND_CLIP2CLIP_EXTRACTOR,
        "--num_of_samples_per_prompt",
        "1",
        "--output_path",
        str(official_root / "results"),
        "--full_json_dir",
        f"{VBENCH_LONG_CHECKOUT}/vbench2_beta_long/VBench_full_info.json",
    ]
    _require(
        set(official_config)
        == {
            "schema",
            "measurement_id",
            "upstream",
            "python_executable",
            "launcher",
            "visible_device",
            "mode",
            "dev_flag",
            "dimensions",
            "offline_assets",
            "offline_policy",
            "offline_environment",
            "prompt",
            "staging",
            "source_video",
            "staged_video",
            "command",
        }
        and official_config.get("schema") == VBENCH_LONG_CONFIG_SCHEMA
        and official_config.get("measurement_id") == measurement_id
        and official_config.get("upstream") == checkout_binding
        and official_config.get("python_executable") == configured_python
        and isinstance(official_config.get("visible_device"), str)
        and bool(str(official_config["visible_device"]).strip())
        and "," not in str(official_config["visible_device"])
        and official_config.get("mode") == "long_custom_input"
        and official_config.get("dev_flag") is True
        and official_config.get("dimensions") == dimensions
        and official_config.get("offline_assets") == assets
        and official_config.get("offline_policy") == expected_offline_policy
        and official_config.get("offline_environment")
        == expected_offline_environment
        and official_config.get("prompt") == history.prompt_id
        and official_config.get("staging") == staging_binding
        and official_config.get("source_video")
        == {
            "path": str(video_path),
            "sha256": video["sha256"],
            "nbytes": video["nbytes"],
        }
        and official_config.get("staged_video") == staged_relative
        and official_config.get("command") == expected_command,
        "official VBench-Long config is not bound to the frozen evaluator and video",
    )
    staged_path = _safe_run_file(
        run_dir, staged_relative, label="official VBench-Long staged video"
    )
    _require(
        staged_path.samefile(video_path)
        and _sha256_file(staged_path) == video["sha256"]
        and staged_path.stat().st_size == video["nbytes"],
        "official VBench-Long staged video is not the scored decoded video",
    )

    result_binding = official.get("official_result")
    _require(
        isinstance(result_binding, Mapping)
        and set(result_binding) == {"path", "sha256", "nbytes"}
        and isinstance(result_binding.get("path"), str)
        and isinstance(result_binding.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(result_binding["sha256"])) is not None
        and isinstance(result_binding.get("nbytes"), int)
        and not isinstance(result_binding.get("nbytes"), bool)
        and int(result_binding["nbytes"]) > 0,
        "official VBench-Long result binding is invalid",
    )
    result_path = _safe_run_file(
        run_dir,
        result_binding["path"],
        label="official VBench-Long result JSON",
    )
    _require(
        result_path.parent == official_root / "results"
        and result_path.name.endswith("_eval_results.json")
        and _sha256_file(result_path) == result_binding["sha256"]
        and result_path.stat().st_size == result_binding["nbytes"],
        "official VBench-Long result JSON path or bytes drifted",
    )
    try:
        parsed_scores = validate_vbench_long_official_result(
            result_path,
            measurement_root=official_root,
            staged_video=staged_path,
            staged_alias=staged_alias,
        )
        successful_attempt = validate_vbench_long_success_attempt(
            official_root,
            run_dir=run_dir,
            config_sha256=str(official["config_sha256"]),
            command=expected_command,
        )
    except BaseException as error:
        raise Insight2RunError(
            f"official VBench-Long result provenance is invalid: {error}"
        ) from error
    _require(
        parsed_scores == dimension_scores,
        "official VBench-Long recomputed scores differ from recorded scores",
    )
    _require(
        successful_attempt,
        "official VBench-Long result has no successful current attempt evidence",
    )
    return macro_mean, motion_smoothness


def _audit_quality_measurement(
    *,
    run_dir: Path,
    measurement_id: str,
    history: LatentHistory,
    evaluation_kind: str,
    chunk_index: int | None,
    quality_model_binding: Mapping[str, str],
    verified_decode_files: dict[str, str],
    measurement_schema: str,
    primary_metric_id: str,
) -> tuple[float, float]:
    """Recompute a quality aggregate and bind its exact decode/window provenance."""

    if measurement_schema == VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA:
        return _audit_vbench_long_quality_measurement(
            run_dir=run_dir,
            measurement_id=measurement_id,
            history=history,
            evaluation_kind=evaluation_kind,
            chunk_index=chunk_index,
            measurement_schema=measurement_schema,
            primary_metric_id=primary_metric_id,
            verified_decode_files=verified_decode_files,
        )
    _require(
        measurement_schema == QUALITY_MEASUREMENT_SCHEMA
        and primary_metric_id == PRIMARY_QUALITY_METRIC,
        "legacy quality measurement schema or metric identity is invalid",
    )

    relative = f"measurements/quality/{measurement_id}.json"
    path = _safe_run_file(run_dir, relative, label="quality measurement sidecar")
    value = _load_json_object(path, label="quality measurement sidecar")
    _require(
        path.read_bytes() == _canonical_json_bytes(value) + b"\n",
        "quality measurement sidecar is not canonical JSON",
    )
    _require(
        set(value)
        == {
            "schema",
            "measurement_id",
            "history",
            "decode_evidence",
            "frames",
            "evaluation",
            "scores",
            "draft_model",
            "metrics",
        }
        and value.get("schema") == measurement_schema
        and value.get("measurement_id") == measurement_id,
        "quality measurement fields or identity are invalid",
    )

    expected_latents = [
        {
            "chunk_index": segment.chunk_index,
            "path": segment.latent_path,
            "sha256": segment.latent_sha256,
            "nbytes": segment.latent_nbytes,
            "segment_sha256": segment.segment_sha256,
        }
        for segment in history.segments
    ]
    _require(
        value.get("history")
        == {
            "manifest_sha256": history.manifest_sha256,
            "head_sha256": history.head_sha256,
            "latent_chunk_count": history.next_chunk_index,
            "latents": expected_latents,
        },
        "quality measurement history differs from the scored latent trajectory",
    )
    frame_count = 1 + 4 * (3 * history.next_chunk_index - 1)
    decode_relative = f"decoded/{history.manifest_sha256}.json"
    decode_binding = value.get("decode_evidence")
    _require(
        isinstance(decode_binding, Mapping)
        and set(decode_binding) == {"path", "sha256"}
        and decode_binding.get("path") == decode_relative
        and isinstance(decode_binding.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(decode_binding["sha256"])) is not None,
        "quality measurement decode binding is invalid",
    )
    decode_path = _safe_run_file(
        run_dir, decode_relative, label="quality decode evidence"
    )
    decode_sha256 = str(decode_binding["sha256"])
    _require(
        _sha256_file(decode_path) == decode_sha256,
        "quality decode evidence SHA-256 mismatch",
    )
    prior_decode = verified_decode_files.get(decode_relative)
    if prior_decode is None:
        verified_decode_files[decode_relative] = decode_sha256
    else:
        _require(
            prior_decode == decode_sha256,
            "one decode evidence path has conflicting immutable identities",
        )
    decode = _load_json_object(decode_path, label="quality decode evidence")
    _require(
        decode_path.read_bytes() == _canonical_json_bytes(decode) + b"\n",
        "quality decode evidence is not canonical JSON",
    )
    frames = value.get("frames")
    _require(
        set(decode)
        == {
            "schema",
            "history_manifest_sha256",
            "latent_chunk_count",
            "expected_frames",
            "frames",
            "video",
        }
        and decode.get("schema") == DECODE_EVIDENCE_SCHEMA
        and decode.get("history_manifest_sha256") == history.manifest_sha256
        and decode.get("latent_chunk_count") == history.next_chunk_index
        and decode.get("expected_frames") == frame_count
        and decode.get("frames") == frames
        and decode.get("video") is None,
        "quality decode evidence differs from its scored history or frame metadata",
    )
    _require(
        isinstance(frames, Mapping)
        and set(frames) == {"path", "key", "sha256", "nbytes", "shape", "dtype"}
        and frames.get("path") == f"decoded/{history.manifest_sha256}.safetensors"
        and frames.get("key") == "frames"
        and isinstance(frames.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(frames["sha256"])) is not None
        and isinstance(frames.get("nbytes"), int)
        and not isinstance(frames.get("nbytes"), bool)
        and int(frames["nbytes"]) > 0
        and frames.get("shape") == [frame_count, 3, 480, 832]
        and frames.get("dtype") == "uint8",
        "quality decoded-frame metadata violates the VAE output contract",
    )

    if evaluation_kind == "local":
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and 0 <= chunk_index < history.next_chunk_index,
            "local quality chunk index is invalid",
        )
        if chunk_index == 0:
            direct_start, direct_stop, temporal_start = 0, 9, 0
        else:
            direct_start = 12 * chunk_index - 3
            direct_stop = 12 * chunk_index + 9
            temporal_start = direct_start - 1
        expected_indices = list(range(direct_start, direct_stop))
        expected_window = {"start": temporal_start, "stop": direct_stop}
    else:
        _require(
            evaluation_kind == "full" and chunk_index is None,
            "full quality evaluation identity is invalid",
        )
        expected_indices = list(range(frame_count))
        expected_window = {"start": 0, "stop": frame_count}
    _require(
        value.get("evaluation")
        == {
            "kind": evaluation_kind,
            "chunk_index": chunk_index,
            "decoded_frame_count": frame_count,
            "frame_indices": expected_indices,
            "temporal_window": expected_window,
        },
        "quality measurement used a non-frozen frame or temporal window",
    )

    scores = value.get("scores")
    _require(
        isinstance(scores, Mapping)
        and set(scores)
        == {"quality_scores", "quality_mean", "temporal_consistency"},
        "quality measurement scores fields are invalid",
    )
    quality_scores = scores.get("quality_scores")
    _require(
        isinstance(quality_scores, list)
        and len(quality_scores) == len(expected_indices)
        and bool(quality_scores),
        "quality measurement omitted one score per selected frame",
    )
    parsed_scores: list[float] = []
    for index, item in enumerate(quality_scores):
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and 0.0 <= float(item) <= 1.0,
            f"quality measurement score {index} is invalid",
        )
        parsed_scores.append(float(item))
    quality_mean = scores.get("quality_mean")
    temporal = scores.get("temporal_consistency")
    recomputed_mean = math.fsum(parsed_scores) / len(parsed_scores)
    _require(
        isinstance(quality_mean, (int, float))
        and not isinstance(quality_mean, bool)
        and float(quality_mean) == recomputed_mean
        and isinstance(temporal, (int, float))
        and not isinstance(temporal, bool)
        and math.isfinite(float(temporal))
        and 0.0 <= float(temporal) <= 1.0,
        "quality aggregate or temporal consistency is invalid",
    )
    _require(
        value.get("draft_model")
        == {
            "role": "draft",
            "model_id": quality_model_binding["model_id"],
            "model_artifact_sha256": quality_model_binding["model_artifact_sha256"],
            "musiq_spaq_sha256": quality_model_binding["musiq_spaq_sha256"],
            "vae_sha256": quality_model_binding["vae_sha256"],
        },
        "quality measurement model/checkpoint binding drifted",
    )
    _require(
        value.get("metrics")
        == {
            "primary": {
                "id": primary_metric_id,
                "definition": PRIMARY_QUALITY_DEFINITION,
                "range": [0.0, 1.0],
            },
            "temporal_consistency": {
                "id": TEMPORAL_CONSISTENCY_METRIC,
                "definition": TEMPORAL_CONSISTENCY_DEFINITION,
                "range": [0.0, 1.0],
            },
        },
        "quality metric definitions drifted",
    )
    return recomputed_mean, float(temporal)


def _audit_boundary_velocity_witness(
    diagnostics: Any, *, label: str
) -> Mapping[str, Any]:
    """Validate the deterministic latent-boundary witness used by the gate."""

    _require(isinstance(diagnostics, Mapping), f"{label} diagnostics are invalid")
    witness = diagnostics.get("boundary_velocity_witness")
    _require(
        isinstance(witness, Mapping)
        and set(witness) == _BOUNDARY_VELOCITY_WITNESS_KEYS,
        f"{label} boundary velocity witness fields are invalid",
    )
    _require(
        witness.get("schema") == BOUNDARY_VELOCITY_WITNESS_SCHEMA
        and witness.get("metric_id") == BOUNDARY_VELOCITY_METRIC_ID
        and witness.get("accumulator_dtype") == "float64"
        and witness.get("round_decimals") == BOUNDARY_VELOCITY_ROUND_DECIMALS
        and witness.get("input_dtype")
        in {"float16", "bfloat16", "float32", "float64"},
        f"{label} boundary velocity witness contract drifted",
    )
    for field in ("history_frame_count", "candidate_frame_count"):
        value = witness.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 2,
            f"{label} boundary velocity witness {field} is invalid",
        )
    for field in (
        "previous_velocity_norm",
        "boundary_velocity_norm",
        "candidate_velocity_norm",
    ):
        value = witness.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            and round(float(value), BOUNDARY_VELOCITY_ROUND_DECIMALS)
            == float(value),
            f"{label} boundary velocity witness {field} is invalid",
        )
    for field in ("incoming_discontinuity", "outgoing_discontinuity", "score"):
        value = witness.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
            and round(float(value), BOUNDARY_VELOCITY_ROUND_DECIMALS)
            == float(value),
            f"{label} boundary velocity witness {field} is invalid",
        )
    _require(
        float(witness["score"])
        == max(
            float(witness["incoming_discontinuity"]),
            float(witness["outgoing_discontinuity"]),
        ),
        f"{label} boundary velocity witness score is inconsistent",
    )
    return witness


def _recompute_boundary_velocity_witness(
    *,
    run_dir: Path,
    prefix: LatentHistory,
    segment: LatentSegment,
    label: str,
) -> Mapping[str, Any]:
    """Recompute one boundary witness directly from indexed safetensors.

    Imports stay lazy so config/preflight-only use remains Torch-free.  Formal
    post-run audit is fail-closed if the pinned tensor runtime is unavailable.
    """

    _require(
        bool(prefix.segments),
        f"{label} boundary witness requires a non-empty history",
    )
    try:
        import torch  # type: ignore
        from safetensors.torch import load_file  # type: ignore
    except ImportError as error:
        raise Insight2RunError(
            f"{label} boundary witness recomputation requires torch/safetensors"
        ) from error

    history_frame_count = 0
    history_tail: Any | None = None
    reference_shape: tuple[int, ...] | None = None
    reference_dtype: Any | None = None
    for index, history_segment in enumerate(prefix.segments):
        history_file = _safe_run_file(
            run_dir,
            history_segment.latent_path,
            label=f"{label} history latent {index}",
        )
        try:
            payload = load_file(str(history_file), device="cpu")
        except BaseException as error:
            raise Insight2RunError(
                f"cannot load {label} history latent {index}: {error}"
            ) from error
        _require(
            set(payload) == {"latent"},
            f"{label} history latent {index} must contain exactly key 'latent'",
        )
        tensor = payload["latent"]
        shape = tuple(int(item) for item in tensor.shape)
        _require(
            len(shape) == 5 and shape[0] == 1 and shape[1] >= 2,
            f"{label} history latent {index} has an invalid shape",
        )
        if reference_shape is None:
            reference_shape = shape
            reference_dtype = tensor.dtype
        _require(
            shape[0] == reference_shape[0]
            and shape[2:] == reference_shape[2:]
            and tensor.dtype == reference_dtype,
            f"{label} history latent shapes or dtypes are inconsistent",
        )
        history_frame_count += shape[1]
        history_tail = tensor[:, -2:].detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()

    candidate_file = _safe_run_file(
        run_dir, segment.latent_path, label=f"{label} candidate latent"
    )
    try:
        candidate_payload = load_file(str(candidate_file), device="cpu")
    except BaseException as error:
        raise Insight2RunError(f"cannot load {label} candidate latent: {error}") from error
    _require(
        set(candidate_payload) == {"latent"},
        f"{label} candidate latent must contain exactly key 'latent'",
    )
    candidate = candidate_payload["latent"]
    candidate_shape = tuple(int(item) for item in candidate.shape)
    _require(
        reference_shape is not None
        and reference_dtype is not None
        and history_tail is not None
        and len(candidate_shape) == 5
        and candidate_shape[0] == 1
        and candidate_shape[1] >= 2
        and candidate_shape[2:] == reference_shape[2:]
        and candidate.dtype == reference_dtype,
        f"{label} candidate latent shape or dtype is incompatible with history",
    )
    dtype_name = str(reference_dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name[len("torch.") :]
    _require(
        dtype_name in {"float16", "bfloat16", "float32", "float64"},
        f"{label} boundary witness input dtype is unsupported",
    )
    candidate_head = candidate[:, :2].detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    _require(
        bool(torch.isfinite(history_tail).all().item())
        and bool(torch.isfinite(candidate_head).all().item()),
        f"{label} boundary witness inputs contain non-finite values",
    )
    previous_velocity = history_tail[:, 1] - history_tail[:, 0]
    boundary_velocity = candidate_head[:, 0] - history_tail[:, -1]
    candidate_velocity = candidate_head[:, 1] - candidate_head[:, 0]

    def norm(value: Any) -> float:
        result = float(torch.linalg.vector_norm(value.reshape(-1), ord=2).item())
        _require(
            math.isfinite(result) and result >= 0.0,
            f"{label} boundary witness produced an invalid norm",
        )
        return result

    previous_norm = norm(previous_velocity)
    boundary_norm = norm(boundary_velocity)
    candidate_norm = norm(candidate_velocity)

    def discontinuity(
        left: Any,
        right: Any,
        left_norm: float,
        right_norm: float,
    ) -> float:
        denominator = left_norm + right_norm
        raw = 0.0 if denominator == 0.0 else norm(left - right) / denominator
        rounded = round(raw, BOUNDARY_VELOCITY_ROUND_DECIMALS)
        _require(
            math.isfinite(rounded) and 0.0 <= rounded <= 1.0,
            f"{label} boundary witness produced an invalid discontinuity",
        )
        return rounded

    incoming = discontinuity(
        previous_velocity, boundary_velocity, previous_norm, boundary_norm
    )
    outgoing = discontinuity(
        boundary_velocity, candidate_velocity, boundary_norm, candidate_norm
    )
    return {
        "schema": BOUNDARY_VELOCITY_WITNESS_SCHEMA,
        "metric_id": BOUNDARY_VELOCITY_METRIC_ID,
        "accumulator_dtype": "float64",
        "round_decimals": BOUNDARY_VELOCITY_ROUND_DECIMALS,
        "input_dtype": dtype_name,
        "history_frame_count": history_frame_count,
        "candidate_frame_count": candidate_shape[1],
        "previous_velocity_norm": round(
            previous_norm, BOUNDARY_VELOCITY_ROUND_DECIMALS
        ),
        "boundary_velocity_norm": round(
            boundary_norm, BOUNDARY_VELOCITY_ROUND_DECIMALS
        ),
        "candidate_velocity_norm": round(
            candidate_norm, BOUNDARY_VELOCITY_ROUND_DECIMALS
        ),
        "incoming_discontinuity": incoming,
        "outgoing_discontinuity": outgoing,
        "score": max(incoming, outgoing),
    }


def _audit_recomputed_boundary_velocity_witness(
    *,
    run_dir: Path,
    prefix: LatentHistory,
    segment: LatentSegment,
    diagnostics: Any,
    label: str,
) -> Mapping[str, Any]:
    stored = _audit_boundary_velocity_witness(diagnostics, label=label)
    recomputed = _recompute_boundary_velocity_witness(
        run_dir=run_dir,
        prefix=prefix,
        segment=segment,
        label=label,
    )
    _require(
        dict(stored) == dict(recomputed),
        f"{label} boundary velocity witness differs from indexed latent bytes",
    )
    return stored


def _audit_target_checkpoint_binding(
    *,
    generated: Mapping[str, Any],
    context: GenerationContext,
    segment: LatentSegment,
    prefix: LatentHistory,
    prompt_sha256: str,
    protocol_id: str,
    target_model_binding: tuple[str, str],
    diagnostic_records: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """Bind a committed Target checkpoint to one fully audited physical call."""

    _require(context.role == "target", "Target checkpoint has a non-Target context")
    _require(
        (context.model_id, context.model_artifact_sha256) == target_model_binding,
        "Target checkpoint model differs from the campaign binding",
    )
    attempt_id = generated.get("target_attempt_id")
    logical_call_id = generated.get("target_logical_call_id")
    _require(
        isinstance(attempt_id, str)
        and _SHA256_RE.fullmatch(attempt_id) is not None
        and attempt_id in diagnostic_records,
        "Target checkpoint does not name an audited physical attempt",
    )
    _require(
        isinstance(logical_call_id, str)
        and _SHA256_RE.fullmatch(logical_call_id) is not None,
        "Target checkpoint logical call ID is invalid",
    )
    diagnostics = diagnostic_records[attempt_id]
    backend = generated.get("diagnostics")
    _require(
        isinstance(backend, dict)
        and TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS <= set(backend)
        and set(backend) <= TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS | {"native_correction_state"},
        "Target checkpoint backend diagnostics fields are invalid",
    )
    _require(
        hashlib.sha256(_canonical_json_bytes(backend)).hexdigest()
        == diagnostics.get("backend_diagnostics_sha256"),
        "Target checkpoint backend diagnostics differ from the audited sidecar",
    )
    expected_logical_call_id = hashlib.sha256(
        _canonical_json_bytes(
            [
                "target-logical-call-v2",
                protocol_id,
                context.model_id,
                context.model_artifact_sha256,
                prefix.prompt_id,
                prompt_sha256,
                prefix.head_sha256,
                segment.chunk_index,
                context.noise.descriptor_sha256,
            ]
        )
    ).hexdigest()
    _require(
        logical_call_id == expected_logical_call_id,
        "Target checkpoint logical call ID is not its deterministic scientific identity",
    )
    expected = {
        "attempt_id": attempt_id,
        "logical_call_id": logical_call_id,
        "model_id": context.model_id,
        "model_artifact_sha256": context.model_artifact_sha256,
        "prompt_id": prefix.prompt_id,
        "prompt_sha256": prompt_sha256,
        "chunk_index": segment.chunk_index,
        "internal_seed": context.noise.derived_seed,
        "history_head_sha256": prefix.head_sha256,
        "history_latent_paths": [item.latent_path for item in prefix.segments],
        "noise_descriptor": context.noise.to_dict(),
        "noise_descriptor_sha256": context.noise.descriptor_sha256,
        "latent_path": segment.latent_path,
        "latent_sha256": segment.latent_sha256,
        "segment_sha256": segment.segment_sha256,
        "kv_cache_sha256": context.kv.cache_sha256,
    }
    for field, expected_value in expected.items():
        _require(
            diagnostics.get(field) == expected_value,
            f"Target checkpoint {field} differs from the audited physical call",
        )
    return attempt_id, logical_call_id


def _audit_lookahead_target_candidate(
    *,
    run_dir: Path,
    generated: Any,
    prefix: LatentHistory,
    expected_noise: InitialNoiseDescriptor,
    prompt_sha256: str,
    protocol_id: str,
    target_model_binding: tuple[str, str],
    diagnostic_records: Mapping[str, Mapping[str, Any]],
    verified_segment_files: dict[str, tuple[str, int]],
    label: str,
) -> dict[str, Any]:
    """Audit a materialized Target proposal whether accepted or rejected."""

    _require(
        isinstance(generated, Mapping)
        and set(generated) == _GENERATED_RECORD_KEYS,
        f"{label} fields are invalid",
    )
    try:
        context = _evidence_context(generated.get("context"), label=f"{label} context")
        segment = LatentSegment.from_dict(
            generated.get("segment"), path=f"$.{label}.segment"
        )
        segment.verify_context(prefix, context)
        context.kv.verify_for(
            role="target",
            model_id=context.model_id,
            model_artifact_sha256=context.model_artifact_sha256,
            history_head_sha256=prefix.head_sha256,
        )
    except CorrectionStateError as error:
        raise Insight2RunError(f"invalid {label}: {error}") from error
    _require(
        context.role == "target"
        and (context.model_id, context.model_artifact_sha256)
        == target_model_binding
        and context.history_head_sha256 == prefix.head_sha256
        and context.next_chunk_index == prefix.next_chunk_index
        and context.noise.to_dict() == expected_noise.to_dict()
        and generated.get("effective_rank") is None
        and generated.get("hidden_state_path") is None
        and generated.get("hidden_state_sha256") is None,
        f"{label} is not the same-prefix/same-noise Target proposal",
    )
    segment_identity = (segment.latent_sha256, segment.latent_nbytes)
    prior_segment = verified_segment_files.get(segment.latent_path)
    if prior_segment is None:
        try:
            segment.verify_file(run_dir)
        except CorrectionStateError as error:
            raise Insight2RunError(f"{label} latent verification failed: {error}") from error
        verified_segment_files[segment.latent_path] = segment_identity
    else:
        _require(
            prior_segment == segment_identity,
            f"{label} reuses a latent path with a conflicting identity",
        )
    _audit_boundary_velocity_witness(generated.get("diagnostics"), label=label)
    attempt_id, logical_call_id = _audit_target_checkpoint_binding(
        generated=generated,
        context=context,
        segment=segment,
        prefix=prefix,
        prompt_sha256=prompt_sha256,
        protocol_id=protocol_id,
        target_model_binding=target_model_binding,
        diagnostic_records=diagnostic_records,
    )
    return {
        "context": context,
        "segment": segment,
        "attempt_id": attempt_id,
        "logical_call_id": logical_call_id,
    }


def _audit_draft_generated_record(
    *,
    run_dir: Path,
    generated: Mapping[str, Any],
    prefix: LatentHistory,
    expected_noise: InitialNoiseDescriptor,
    draft_model_binding: tuple[str, str],
    require_hidden_evidence: bool,
    verified_segment_files: dict[str, tuple[str, int]],
    verified_hidden_files: dict[str, tuple[str, int]],
    label: str,
) -> dict[str, Any]:
    """Validate one persisted Draft generation, including its real KV rebuild."""

    _require(
        isinstance(generated, Mapping)
        and set(generated) == _GENERATED_RECORD_KEYS,
        f"{label} fields are invalid",
    )
    try:
        context = _evidence_context(
            generated.get("context"), label=f"{label} context"
        )
        segment = LatentSegment.from_dict(
            generated.get("segment"), path=f"$.{label}.segment"
        )
        segment.verify_context(prefix, context)
        context.kv.verify_for(
            role="draft",
            model_id=context.model_id,
            model_artifact_sha256=context.model_artifact_sha256,
            history_head_sha256=prefix.head_sha256,
        )
    except CorrectionStateError as error:
        raise Insight2RunError(f"invalid {label}: {error}") from error
    chunk_index = prefix.next_chunk_index
    _require(
        context.role == "draft"
        and (context.model_id, context.model_artifact_sha256)
        == draft_model_binding
        and context.history_head_sha256 == prefix.head_sha256
        and context.next_chunk_index == chunk_index
        and context.noise.to_dict() == expected_noise.to_dict(),
        f"{label} context is not the same-prefix/same-noise Draft proposal",
    )
    _require(
        expected_noise.namespace == NOISE_NAMESPACE
        and expected_noise.prompt_id == prefix.prompt_id
        and expected_noise.trajectory_seed == prefix.trajectory_seed
        and expected_noise.chunk_index == chunk_index
        and expected_noise.shape == (1, 3, 16, 60, 104)
        and expected_noise.dtype == "bfloat16",
        f"{label} noise identity is invalid",
    )
    _require(
        generated.get("target_attempt_id") is None
        and generated.get("target_logical_call_id") is None,
        f"{label} carries Target lifecycle identities",
    )
    rank = generated.get("effective_rank")
    _require(
        isinstance(rank, (int, float))
        and not isinstance(rank, bool)
        and math.isfinite(float(rank))
        and float(rank) >= 1.0,
        f"{label} effective rank is invalid",
    )

    segment_identity = (segment.latent_sha256, segment.latent_nbytes)
    prior_segment = verified_segment_files.get(segment.latent_path)
    if prior_segment is None:
        try:
            segment.verify_file(run_dir)
        except CorrectionStateError as error:
            raise Insight2RunError(f"{label} latent verification failed: {error}") from error
        verified_segment_files[segment.latent_path] = segment_identity
    else:
        _require(
            prior_segment == segment_identity,
            f"{label} reuses a latent path with a conflicting identity",
        )

    diagnostics = generated.get("diagnostics")
    expected_diagnostic_keys = set(_DRAFT_BACKEND_DIAGNOSTIC_KEYS)
    if isinstance(diagnostics, Mapping) and "kv_reuse" in diagnostics:
        expected_diagnostic_keys.add("kv_reuse")
    if require_hidden_evidence:
        expected_diagnostic_keys.add("sampled_hidden")
    _require(
        isinstance(diagnostics, Mapping)
        and set(diagnostics) == expected_diagnostic_keys,
        f"{label} backend diagnostics fields are invalid",
    )
    _audit_boundary_velocity_witness(diagnostics, label=label)
    try:
        recomputed_rank = validate_effective_rank_evidence(
            diagnostics.get("rank_evidence"),
            expected_rank=float(rank),
            expected_eigenvalue_count=256,
        )
    except AdapterError as error:
        raise Insight2RunError(f"{label} rank evidence is invalid: {error}") from error
    _require(
        recomputed_rank == float(rank),
        f"{label} effective rank differs from its Gram spectrum",
    )
    kv_sha256 = diagnostics.get("kv_witness_sha256")
    _require(
        isinstance(kv_sha256, str)
        and _SHA256_RE.fullmatch(kv_sha256) is not None
        and kv_sha256 == context.kv.cache_sha256,
        f"{label} KV witness digest disagrees with its context",
    )
    kv_witness = diagnostics.get("kv_witness")
    try:
        validate_sampled_kv_cache_witness(
            kv_witness,
            expected_layer_count=SELF_FORCING_KV_LAYER_COUNT,
            expected_self_shape=SELF_FORCING_SELF_KV_SHAPE,
            expected_cross_shape=SELF_FORCING_CROSS_KV_SHAPE,
            expected_dtype=SELF_FORCING_KV_DTYPE,
            expected_self_used_tokens=min(
                chunk_index * SELF_FORCING_TOKENS_PER_CHUNK,
                SELF_FORCING_SELF_KV_SHAPE[1],
            ),
            expected_cross_initialized=chunk_index > 0,
        )
    except AdapterError as error:
        raise Insight2RunError(f"{label} KV witness is invalid: {error}") from error
    _require(
        hashlib.sha256(_canonical_json_bytes(kv_witness)).hexdigest() == kv_sha256,
        f"{label} KV witness canonical hash is invalid",
    )
    trace = diagnostics.get("forward_trace")
    _require(isinstance(trace, list), f"{label} forward trace is invalid")
    try:
        from .draft_reuse_evidence import validate_reuse_evidence
        reused = validate_reuse_evidence(diagnostics.get("kv_reuse"), history_chunks=chunk_index,
            history_file_sha256=[x.latent_sha256 for x in prefix.segments])
        validate_self_forcing_forward_trace(trace, history_chunks=chunk_index, reused_history_chunks=reused)
    except (AdapterError, ValueError) as error:
        raise Insight2RunError(f"{label} forward trace is invalid: {error}") from error

    hidden_path = generated.get("hidden_state_path")
    hidden_sha256 = generated.get("hidden_state_sha256")
    if require_hidden_evidence:
        _require(
            isinstance(hidden_path, str)
            and isinstance(hidden_sha256, str)
            and _SHA256_RE.fullmatch(hidden_sha256) is not None,
            f"{label} full hidden-state evidence is invalid",
        )
        hidden_file = _safe_run_file(
            run_dir, hidden_path, label=f"{label} hidden_state_path"
        )
        hidden_identity = (hidden_sha256, hidden_file.stat().st_size)
        prior_hidden = verified_hidden_files.get(hidden_path)
        if prior_hidden is None:
            _require(
                _sha256_file(hidden_file) == hidden_sha256,
                f"{label} full hidden-state SHA-256 mismatch",
            )
            verified_hidden_files[hidden_path] = hidden_identity
        else:
            _require(
                prior_hidden == hidden_identity,
                f"{label} hidden path has a conflicting immutable identity",
            )
        sampled = diagnostics.get("sampled_hidden")
        _require(
            isinstance(sampled, Mapping)
            and isinstance(sampled.get("path"), str)
            and isinstance(sampled.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(sampled["sha256"])) is not None
            and isinstance(sampled.get("nbytes"), int)
            and not isinstance(sampled.get("nbytes"), bool)
            and int(sampled["nbytes"]) > 0,
            f"{label} sampled hidden-state evidence is invalid",
        )
        sampled_path = str(sampled["path"])
        sampled_file = _safe_run_file(
            run_dir, sampled_path, label=f"{label} sampled_hidden.path"
        )
        sampled_identity = (str(sampled["sha256"]), int(sampled["nbytes"]))
        prior_sampled = verified_hidden_files.get(sampled_path)
        if prior_sampled is None:
            _require(
                _sha256_file(sampled_file) == sampled_identity[0]
                and sampled_file.stat().st_size == sampled_identity[1],
                f"{label} sampled hidden-state bytes drifted",
            )
            verified_hidden_files[sampled_path] = sampled_identity
        else:
            _require(
                prior_sampled == sampled_identity,
                f"{label} sampled hidden path has a conflicting identity",
            )
    else:
        _require(
            hidden_path is None and hidden_sha256 is None,
            f"{label} unexpectedly carries full hidden-state sidecars",
        )

    return {
        "context": context,
        "segment": segment,
        "rank": float(rank),
        "kv_sha256": kv_sha256,
    }


def _audit_generated_chunk_path_horizon(
    *, run_dir: Path, chunk_count: int
) -> dict[str, int]:
    """Reject generated noise/latent paths outside the protocol horizon.

    Route replay validates every referenced proposal and lookahead probe. This
    complementary scan covers unreferenced crash/retry residue without
    requiring one file per logical call: extra in-range attempts are allowed,
    but a synthetic terminal lookahead such as ``chunk-120`` for an N=120
    protocol is never publishable evidence.
    """

    _require(
        isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count >= 2,
        "generated-artifact chunk horizon is invalid",
    )
    counts = {"noise": 0, "latents": 0}
    resolved_run_dir = run_dir.resolve(strict=True)
    for category in counts:
        root = run_dir / category
        if not root.exists() and not root.is_symlink():
            continue
        _require(
            not root.is_symlink() and root.is_dir(),
            f"generated-artifact {category} root must be a real directory",
        )
        try:
            paths = sorted(root.rglob("*"))
        except OSError as error:
            raise Insight2RunError(
                f"cannot enumerate generated-artifact {category} paths: {error}"
            ) from error
        for path in paths:
            _require(
                not path.is_symlink(),
                f"generated-artifact path may not be a symlink: {path}",
            )
            if path.is_dir():
                continue
            _require(
                path.is_file(),
                f"generated-artifact path must be a regular file: {path}",
            )
            try:
                relative = path.resolve(strict=True).relative_to(resolved_run_dir)
            except (OSError, ValueError) as error:
                raise Insight2RunError(
                    f"generated-artifact path escapes the run directory: {path}: "
                    f"{error}"
                ) from error
            matches = tuple(
                _GENERATED_CHUNK_PATH_RE.finditer(relative.as_posix())
            )
            for match in matches:
                chunk_index = int(match.group(1))
                _require(
                    chunk_index < chunk_count,
                    f"generated-artifact path names out-of-horizon chunk "
                    f"{chunk_index} for N={chunk_count}: {relative.as_posix()}",
                )
            if matches:
                counts[category] += 1
    return counts


def _audit_lookahead_preview_path_budget(
    *,
    run_dir: Path,
    route_key: str,
    corrected_chunks: Sequence[int],
    chunk_count: int,
    proposed_target_chunks: Sequence[int] | None = None,
) -> int:
    """Reject route-local preview assets created after budget exhaustion.

    Crash/retry attempts below a legitimate preview stem remain valid audit
    residue.  What is forbidden is a preview *stem* whose source chunk is
    later than the final proposal opportunity, because the exact-K controller
    has no remaining decision to guard at that point.
    """

    _require(
        isinstance(route_key, str) and bool(route_key),
        "lookahead route key is invalid",
    )
    parsed_corrections = tuple(corrected_chunks)
    _require(
        (bool(parsed_corrections) or proposed_target_chunks is not None)
        and all(
            isinstance(chunk, int)
            and not isinstance(chunk, bool)
            and 1 <= chunk < chunk_count - 1
            for chunk in parsed_corrections
        )
        and tuple(sorted(set(parsed_corrections))) == parsed_corrections,
        "lookahead correction set is invalid",
    )
    selected_corrections = frozenset(parsed_corrections)
    parsed_proposals = (
        parsed_corrections
        if proposed_target_chunks is None
        else tuple(proposed_target_chunks)
    )
    _require(
        bool(parsed_proposals)
        and all(
            isinstance(chunk, int)
            and not isinstance(chunk, bool)
            and 1 <= chunk < chunk_count - 1
            for chunk in parsed_proposals
        )
        and tuple(sorted(set(parsed_proposals))) == parsed_proposals
        and selected_corrections <= frozenset(parsed_proposals),
        "lookahead Target proposal set is invalid",
    )
    boundary_gated = proposed_target_chunks is not None
    proposal_chunks = frozenset(parsed_proposals)
    last_proposal_chunk = parsed_proposals[-1]
    root = run_dir / "latents" / "rollouts" / route_key
    _require(
        root.is_dir() and not root.is_symlink(),
        "lookahead latent route root must be a real directory",
    )
    preview_files = 0
    for stem in sorted(root.iterdir()):
        _require(
            stem.is_dir() and not stem.is_symlink(),
            f"lookahead route latent stem must be a real directory: {stem}",
        )
        preview_match = _LOOKAHEAD_PREVIEW_STEM_RE.fullmatch(stem.name)
        corrected_match = _LOOKAHEAD_AFTER_TARGET_STEM_RE.fullmatch(stem.name)
        route_match = _LOOKAHEAD_ROUTE_STEM_RE.fullmatch(stem.name)
        _require(
            preview_match is not None
            or (boundary_gated and corrected_match is not None)
            or route_match is not None,
            f"lookahead route contains an unknown latent stem: {stem.name}",
        )
        if preview_match is not None:
            signal_chunk = int(preview_match.group(1))
            source_chunk = int(preview_match.group(2))
            _require(
                stem.name
                == (
                    f"chunk-{signal_chunk:02d}-draft-preview-from-"
                    f"{source_chunk:02d}"
                ),
                f"lookahead preview stem is not canonically encoded: {stem.name}",
            )
            _require(
                signal_chunk == source_chunk + 1
                and 1 <= source_chunk < chunk_count - 1,
                f"lookahead preview stem is not an in-horizon immediate "
                f"successor: {stem.name}",
            )
            _require(
                source_chunk <= last_proposal_chunk,
                f"lookahead preview was generated after Target budget "
                f"exhaustion at chunk {last_proposal_chunk}: {stem.name}",
            )
        elif corrected_match is not None:
            signal_chunk = int(corrected_match.group(1))
            source_chunk = int(corrected_match.group(2))
            _require(
                stem.name
                == (
                    f"chunk-{signal_chunk:02d}-draft-after-target-from-"
                    f"{source_chunk:02d}"
                ),
                f"lookahead corrected frontier stem is not canonically encoded: "
                f"{stem.name}",
            )
            _require(
                signal_chunk == source_chunk + 1
                and source_chunk in proposal_chunks
                and 1 <= source_chunk < chunk_count - 1,
                f"lookahead corrected frontier is not an immediate successor of "
                f"a proposed Target: {stem.name}",
            )
        else:
            assert route_match is not None
            route_chunk = int(route_match.group(1))
            route_role = route_match.group(2)
            _require(
                stem.name.startswith(f"chunk-{route_chunk:02d}-")
                and 1 <= route_chunk < chunk_count,
                f"lookahead route latent stem is not canonical or in horizon: "
                f"{stem.name}",
            )
            _require(
                route_role != "target"
                or route_chunk
                in (proposal_chunks if boundary_gated else selected_corrections),
                f"lookahead route contains an orphan Target latent stem: "
                f"{stem.name}",
            )
        for attempt in sorted(stem.iterdir()):
            _require(
                attempt.is_file()
                and not attempt.is_symlink()
                and _GENERATED_ATTEMPT_FILENAME_RE.fullmatch(attempt.name)
                is not None,
                f"lookahead route latent attempt is invalid: {attempt}",
            )
            if preview_match is not None or corrected_match is not None:
                preview_files += 1
    return preview_files


def _audit_rank_calibration(
    *,
    run_dir: Path,
    protocol_id: str,
    calibration_prompts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    chunk_count: int,
    model_bindings: Mapping[str, tuple[str, str]],
    verified_segment_files: dict[str, tuple[str, int]],
    verified_hidden_files: dict[str, tuple[str, int]],
) -> RankRiskCalibrator | SuddenDropRiskCalibrator:
    """Rebuild the independent rank calibration from its persisted evidence."""

    path = _safe_run_file(
        run_dir, "calibration/ranks.json", label="rank calibration"
    )
    value = _load_json_object(path, label="rank calibration")
    _require(
        set(value)
        == {
            "schema",
            "protocol_id",
            "calibration_id",
            "prompt_seed_pairs",
            "rows",
        }
        and value.get("schema") == CALIBRATION_SCHEMA
        and value.get("protocol_id") == protocol_id,
        "rank calibration fields or protocol are invalid",
    )
    try:
        actual_bytes = path.read_bytes()
    except OSError as error:
        raise Insight2RunError(f"cannot reread rank calibration: {error}") from error
    _require(
        actual_bytes == _canonical_json_bytes(value) + b"\n",
        "rank calibration is not canonical JSON",
    )
    expected_pairs = [
        [prompt["prompt_id"], seed]
        for prompt in calibration_prompts
        for seed in seeds
    ]
    _require(
        value.get("prompt_seed_pairs") == expected_pairs,
        "rank calibration prompt/seed population drifted",
    )
    rows = value.get("rows")
    _require(isinstance(rows, list), "rank calibration rows are invalid")
    expected_identities = [
        (prompt, seed, chunk_index)
        for prompt in calibration_prompts
        for seed in seeds
        for chunk_index in range(1, chunk_count)
    ]
    _require(
        len(rows) == len(expected_identities),
        "rank calibration row count drifted",
    )
    audited_rows: list[dict[str, Any]] = []
    ranks: list[float] = []
    drop_scores: list[float] = []
    rank_history_by_pair: dict[tuple[str, int], list[float]] = {}
    previous_by_pair: dict[tuple[str, int], LatentHistory] = {}
    for row_index, (row, identity) in enumerate(zip(rows, expected_identities)):
        prompt_entry, seed, chunk_index = identity
        expected_row_keys = {
            "prompt_id",
            "seed",
            "chunk_index",
            "rank",
            "hidden_state_path",
            "hidden_state_sha256",
        }
        if chunk_count != 16:
            expected_row_keys.add("erank_drop_score")
        _require(
            isinstance(row, dict)
            and set(row) == expected_row_keys
            and row.get("prompt_id") == prompt_entry["prompt_id"]
            and row.get("seed") == seed
            and row.get("chunk_index") == chunk_index,
            f"rank calibration row {row_index} identity drifted",
        )
        prompt_sha256 = hashlib.sha256(
            str(prompt_entry["prompt"]).encode("utf-8")
        ).hexdigest()
        key = f"{prompt_entry['prompt_id']}-{seed}-{prompt_sha256[:12]}"
        pair = (str(prompt_entry["prompt_id"]), seed)
        previous = previous_by_pair.get(pair)
        if previous is None:
            root_path = _safe_run_file(
                run_dir,
                f"state/calibration-baselines/{key}/chunk-00.json",
                label=f"rank calibration root {row_index}",
            )
            _root_raw, root_history, root_context, root_segment, root_metadata = (
                _load_audited_checkpoint(
                    root_path, label=f"rank calibration root {row_index}"
                )
            )
            trajectory_id = "i2-" + hashlib.sha256(
                _canonical_json_bytes(
                    [
                        protocol_id,
                        "baseline",
                        "calibration",
                        prompt_entry["prompt_id"],
                        seed,
                    ]
                )
            ).hexdigest()[:40]
            _require(
                root_history.next_chunk_index == 1
                and root_history.trajectory_id == trajectory_id
                and root_history.prompt_id == prompt_entry["prompt_id"]
                and root_history.trajectory_seed == seed
                and root_metadata == {"kind": "baseline", "chunk_index": 0}
                and root_context.role == "target"
                and (
                    root_context.model_id,
                    root_context.model_artifact_sha256,
                )
                == model_bindings["target"],
                f"rank calibration root {row_index} identity drifted",
            )
            root_identity = (root_segment.latent_sha256, root_segment.latent_nbytes)
            prior_root = verified_segment_files.get(root_segment.latent_path)
            if prior_root is None:
                try:
                    root_segment.verify_file(run_dir)
                except CorrectionStateError as error:
                    raise Insight2RunError(
                        f"rank calibration root latent verification failed: {error}"
                    ) from error
                verified_segment_files[root_segment.latent_path] = root_identity
            else:
                _require(
                    prior_root == root_identity,
                    "rank calibration root latent identity conflicts",
                )
            previous = root_history
        checkpoint_path = _safe_run_file(
            run_dir,
            f"state/calibration-baselines/{key}/chunk-{chunk_index:02d}.json",
            label=f"rank calibration checkpoint {row_index}",
        )
        raw, history, context, _segment, metadata = _load_audited_checkpoint(
            checkpoint_path, label=f"rank calibration checkpoint {row_index}"
        )
        trajectory_id = "i2-" + hashlib.sha256(
            _canonical_json_bytes(
                [
                    protocol_id,
                    "baseline",
                    "calibration",
                    prompt_entry["prompt_id"],
                    seed,
                ]
            )
        ).hexdigest()[:40]
        _require(
            history.next_chunk_index == chunk_index + 1
            and history.trajectory_id == trajectory_id
            and history.prompt_id == prompt_entry["prompt_id"]
            and history.trajectory_seed == seed
            and metadata == {"kind": "baseline", "chunk_index": chunk_index}
            and context.role == "draft",
            f"rank calibration checkpoint {row_index} identity drifted",
        )
        _require(
            history.prefix_before(chunk_index).to_dict() == previous.to_dict(),
            f"rank calibration checkpoint {row_index} breaks trajectory continuity",
        )
        prefix = history.prefix_before(chunk_index)
        audited = _audit_draft_generated_record(
            run_dir=run_dir,
            generated=raw["generated"],
            prefix=prefix,
            expected_noise=context.noise,
            draft_model_binding=model_bindings["draft"],
            require_hidden_evidence=True,
            verified_segment_files=verified_segment_files,
            verified_hidden_files=verified_hidden_files,
            label=f"rank calibration Draft {row_index}",
        )
        _require(
            row.get("rank") == audited["rank"]
            and row.get("hidden_state_path")
            == raw["generated"]["hidden_state_path"]
            and row.get("hidden_state_sha256")
            == raw["generated"]["hidden_state_sha256"],
            f"rank calibration row {row_index} differs from its Draft evidence",
        )
        rank_history = rank_history_by_pair.setdefault(pair, [])
        rank_history.append(float(audited["rank"]))
        if chunk_count != 16:
            drop_score = causal_erank_sudden_drop(
                rank_history,
                lookback_chunks=4,
            )
            _require(
                row.get("erank_drop_score") == drop_score,
                f"rank calibration row {row_index} eRank-drop score drifted",
            )
            drop_scores.append(drop_score)
        previous_by_pair[pair] = history
        audited_rows.append(dict(row))
        ranks.append(float(audited["rank"]))
    calibration_id = f"cal-{hashlib.sha256(_canonical_json_bytes(audited_rows)).hexdigest()[:40]}"
    _require(
        value.get("calibration_id") == calibration_id,
        "rank calibration ID is not its canonical row digest",
    )
    if chunk_count == 16:
        return RankRiskCalibrator(calibration_id=calibration_id, ranks=tuple(ranks))
    return SuddenDropRiskCalibrator(
        calibration_id=calibration_id,
        drop_scores=tuple(drop_scores),
    )


def _audit_2a_baseline_binding(
    *,
    run_dir: Path,
    row: Mapping[str, Any],
    protocol_id: str,
    expected_split: str,
    prompt_entry: Mapping[str, Any],
    calibrator: RankRiskCalibrator,
    chunk_count: int,
    model_bindings: Mapping[str, tuple[str, str]],
    verified_segment_files: dict[str, tuple[str, int]],
    verified_hidden_files: dict[str, tuple[str, int]],
) -> tuple[LatentHistory, LatentHistory]:
    """Bind one paired fork to its immutable evaluation-baseline Draft chunk."""

    prompt_id = str(prompt_entry["prompt_id"])
    prompt_sha256 = hashlib.sha256(str(prompt_entry["prompt"]).encode("utf-8")).hexdigest()
    seed = row.get("seed")
    chunk_index = row.get("chunk_index")
    _require(
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and isinstance(chunk_index, int)
        and not isinstance(chunk_index, bool)
        and 1 <= chunk_index < chunk_count,
        "2(a) row seed or candidate chunk is invalid",
    )
    expected_event_id = hashlib.sha256(
        _canonical_json_bytes(
            [protocol_id, "2a", expected_split, prompt_id, seed, chunk_index]
        )
    ).hexdigest()
    _require(
        row.get("event_id") == expected_event_id
        and row.get("split") == expected_split
        and row.get("prompt_id") == prompt_id
        and row.get("condition_hash") == prompt_sha256,
        "2(a) row deterministic identity drifted",
    )

    evidence_path = _safe_run_file(
        run_dir, row.get("fork_evidence_path"), label="2(a) baseline fork evidence"
    )
    evidence = _load_json_object(evidence_path, label="2(a) baseline fork evidence")
    _require(
        evidence.get("protocol_id") == protocol_id,
        "2(a) fork evidence belongs to another protocol",
    )
    try:
        prefix = LatentHistory.from_dict(evidence.get("prefix"), path="$.2a.prefix")
        noise = InitialNoiseDescriptor.from_dict(
            evidence.get("initial_noise", {}).get("descriptor"),
            path="$.2a.initial_noise.descriptor",
        )
    except (CorrectionStateError, AttributeError) as error:
        raise Insight2RunError(f"2(a) baseline evidence is invalid: {error}") from error
    _require(
        noise.namespace == NOISE_NAMESPACE
        and noise.shape == (1, 3, 16, 60, 104)
        and noise.dtype == "bfloat16",
        "2(a) fork noise protocol drifted",
    )

    key = f"{prompt_id}-{seed}-{prompt_sha256[:12]}"
    checkpoint_path = _safe_run_file(
        run_dir,
        f"state/evaluation-baselines/{key}/chunk-{chunk_index:02d}.json",
        label="2(a) evaluation baseline checkpoint",
    )
    raw, history, context, _segment, metadata = _load_audited_checkpoint(
        checkpoint_path, label="2(a) evaluation baseline checkpoint"
    )
    expected_trajectory_id = "i2-" + hashlib.sha256(
        _canonical_json_bytes(
            [protocol_id, "baseline", "evaluation", prompt_id, seed]
        )
    ).hexdigest()[:40]
    _require(
        history.next_chunk_index == chunk_index + 1
        and history.trajectory_id == expected_trajectory_id
        and history.prompt_id == prompt_id
        and history.trajectory_seed == seed
        and history.prefix_before(chunk_index).to_dict() == prefix.to_dict()
        and metadata == {"kind": "baseline", "chunk_index": chunk_index}
        and context.role == "draft",
        "2(a) evaluation baseline checkpoint identity or prefix drifted",
    )
    audited = _audit_draft_generated_record(
        run_dir=run_dir,
        generated=raw["generated"],
        prefix=prefix,
        expected_noise=noise,
        draft_model_binding=model_bindings["draft"],
        require_hidden_evidence=True,
        verified_segment_files=verified_segment_files,
        verified_hidden_files=verified_hidden_files,
        label="2(a) evaluation baseline Draft",
    )
    draft_branch = evidence.get("draft_branch")
    _require(
        isinstance(draft_branch, Mapping)
        and draft_branch.get("context") == raw["generated"]["context"]
        and draft_branch.get("segment") == raw["generated"]["segment"]
        and draft_branch.get("history") == history.to_dict(),
        "2(a) fork Draft branch differs from its baseline checkpoint",
    )
    rank = audited["rank"]
    _require(
        row.get("rank") == rank
        and row.get("risk_score") == calibrator.risk(rank)
        and row.get("draft_kv_hash") == audited["kv_sha256"]
        and row.get("draft_segment_hash") == audited["segment"].segment_sha256
        and row.get("hidden_state_path")
        == raw["generated"]["hidden_state_path"]
        and row.get("hidden_state_sha256")
        == raw["generated"]["hidden_state_sha256"],
        "2(a) rank, risk, hidden state, or Draft witness differs from baseline",
    )
    target_branch = evidence.get("target_branch")
    _require(
        isinstance(target_branch, Mapping),
        "2(a) fork Target branch is invalid",
    )
    try:
        target_history = LatentHistory.from_dict(
            target_branch.get("history"), path="$.2a.target_branch.history"
        )
    except CorrectionStateError as error:
        raise Insight2RunError(f"2(a) Target measurement history is invalid: {error}") from error
    return history, target_history


def _audit_2b_rollout_route(
    *,
    run_dir: Path,
    protocol_id: str,
    prompt_id: str,
    prompt_sha256: str,
    seed: int,
    route_id: str,
    corrected_chunks: tuple[int, ...],
    proposed_target_chunks: tuple[int, ...] | None = None,
    chunk_count: int,
    model_bindings: Mapping[str, tuple[str, str]],
    quality_model_binding: Mapping[str, str],
    quality_measurement_schema: str,
    primary_quality_metric_id: str,
    online_calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator | None,
    diagnostic_records: Mapping[str, Mapping[str, Any]],
    route_cache: dict[str, dict[str, Any]],
    canonical_root_cache: dict[tuple[str, int], str],
    verified_noise_files: dict[str, tuple[str, int]],
    verified_segment_files: dict[str, tuple[str, int]],
    verified_hidden_files: dict[str, tuple[str, int]],
    verified_decode_files: dict[str, str],
    coverage_constraint: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Reconstruct one complete sequential rollout and its full-video score."""

    proposed_chunks = (
        corrected_chunks
        if proposed_target_chunks is None
        else proposed_target_chunks
    )
    route_key = hashlib.sha256(
        _canonical_json_bytes([protocol_id, prompt_id, seed, route_id])
    ).hexdigest()[:48]
    cached = route_cache.get(route_key)
    if cached is not None:
        _require(
            cached["identity"]
            == [
                prompt_id,
                seed,
                route_id,
                list(corrected_chunks),
                list(proposed_chunks),
            ],
            "one 2(b) route key resolves to conflicting identities",
        )
        return cached

    trajectory_id = "i2-" + hashlib.sha256(
        _canonical_json_bytes([protocol_id, "rollout", prompt_id, seed])
    ).hexdigest()[:40]
    previous = LatentHistory(
        trajectory_id=trajectory_id,
        prompt_id=prompt_id,
        trajectory_seed=seed,
    )
    target_attempt_ids: set[str] = set()
    target_logical_call_ids: set[str] = set()
    proposal_target_attempt_ids: set[str] = set()
    proposal_target_logical_call_ids: set[str] = set()
    chunk_zero_binding_sha256: str | None = None
    is_boundary_coverage_online = route_id.startswith(
        "online-lookahead-coverage-boundary-gated-erank-drop-k"
    )
    is_plain_lookahead_coverage_online = route_id.startswith(
        "online-lookahead-coverage-erank-drop-k"
    )
    is_lookahead_coverage_online = (
        is_plain_lookahead_coverage_online or is_boundary_coverage_online
    )
    is_current_coverage_online = route_id.startswith(
        "online-coverage-erank-drop-k"
    )
    is_coverage_online = (
        is_current_coverage_online or is_lookahead_coverage_online
    )
    _require(
        tuple(sorted(set(proposed_chunks))) == proposed_chunks
        and all(
            isinstance(chunk, int)
            and not isinstance(chunk, bool)
            and 1 <= chunk < chunk_count
            for chunk in proposed_chunks
        )
        and (
            set(corrected_chunks).issubset(proposed_chunks)
            if is_boundary_coverage_online
            else proposed_chunks == corrected_chunks
        ),
        "2(b) route proposed/accepted Target sets are invalid",
    )
    is_online = is_coverage_online or route_id.startswith(
        ("online-rank-guided-k", "online-erank-drop-k")
    )
    if is_online:
        _require(
            online_calibrator is not None,
            "2(b) online route has no independently audited calibration",
        )
        if is_coverage_online:
            _require(
                coverage_constraint is not None,
                "2(b) coverage route has no trusted frozen constraint",
            )
            window_radius, minimum_gap, maximum_gap = coverage_constraint
            online_policy: (
                OnlineRankGuidedPolicy
                | CoverageConstrainedOnlineRankGuidedPolicy
                | LookaheadCoverageConstrainedOnlineRankGuidedPolicy
                | None
            )
            if is_lookahead_coverage_online:
                policy_budget = (
                    len(proposed_chunks)
                    if is_boundary_coverage_online
                    else len(corrected_chunks)
                )
                online_policy = LookaheadCoverageConstrainedOnlineRankGuidedPolicy(
                    budget=policy_budget,
                    calibration=online_calibrator,
                    evaluation_id=(
                        f"eval-lookahead-coverage-"
                        f"{'boundary-' if is_boundary_coverage_online else ''}"
                        f"{prompt_id}-{seed}-k{policy_budget}"
                    ),
                    chunk_count=chunk_count,
                    window_radius_chunks=window_radius,
                    min_gap_chunks=minimum_gap,
                    max_gap_chunks=maximum_gap,
                )
            else:
                online_policy = CoverageConstrainedOnlineRankGuidedPolicy(
                    budget=len(corrected_chunks),
                    calibration=online_calibrator,
                    evaluation_id=(
                        f"eval-coverage-{prompt_id}-{seed}-k{len(corrected_chunks)}"
                    ),
                    chunk_count=chunk_count,
                    window_radius_chunks=window_radius,
                    min_gap_chunks=minimum_gap,
                    max_gap_chunks=maximum_gap,
                )
        else:
            online_policy = OnlineRankGuidedPolicy(
                budget=len(corrected_chunks),
                calibration=online_calibrator,
                evaluation_id=f"eval-{prompt_id}-{seed}-k{len(corrected_chunks)}",
                chunk_count=chunk_count,
            )
    else:
        online_policy = None

    proposal_ranks: list[float] = []
    carried_lookahead_probe: Mapping[str, Any] | None = None
    discarded_lookahead_probes: tuple[Mapping[str, Any], ...] = ()
    replayed_proposed_chunks: list[int] = []
    replayed_accepted_chunks: list[int] = []
    boundary_gate = BoundarySmoothnessGate() if is_boundary_coverage_online else None
    for chunk_index in range(chunk_count):
        relative = f"state/rollouts/{route_key}/chunk-{chunk_index:02d}.json"
        checkpoint_path = _safe_run_file(
            run_dir, relative, label=f"2(b) route checkpoint {chunk_index}"
        )
        raw, history, context, segment, metadata = _load_audited_checkpoint(
            checkpoint_path, label=f"2(b) route checkpoint {chunk_index}"
        )
        _require(
            history.next_chunk_index == chunk_index + 1
            and history.prompt_id == prompt_id
            and history.trajectory_seed == seed
            and history.trajectory_id == trajectory_id,
            "2(b) route checkpoint history identity or length drifted",
        )
        _require(
            history.prefix_before(chunk_index).to_dict() == previous.to_dict(),
            "2(b) route checkpoint does not extend the preceding committed history",
        )
        corrected = chunk_index == 0 or chunk_index in corrected_chunks
        expected_role = "target" if corrected else "draft"
        _require(
            context.role == segment.producer_role == expected_role,
            "2(b) route committed the wrong model at a chunk",
        )
        segment_identity = (segment.latent_sha256, segment.latent_nbytes)
        prior_segment = verified_segment_files.get(segment.latent_path)
        if prior_segment is None:
            try:
                segment.verify_file(run_dir)
            except CorrectionStateError as error:
                raise Insight2RunError(
                    f"2(b) committed latent verification failed: {error}"
                ) from error
            verified_segment_files[segment.latent_path] = segment_identity
        else:
            _require(
                prior_segment == segment_identity,
                "one 2(b) latent path has conflicting immutable identities",
            )
        _require(
            (context.model_id, context.model_artifact_sha256)
            == model_bindings[expected_role],
            "2(b) route checkpoint model binding drifted",
        )
        _require(
            context.history_head_sha256 == previous.head_sha256
            and context.next_chunk_index == chunk_index,
            "2(b) route checkpoint context is bound to the wrong prefix",
        )
        noise = context.noise
        _require(
            noise.namespace == NOISE_NAMESPACE
            and noise.prompt_id == prompt_id
            and noise.trajectory_seed == seed
            and noise.chunk_index == chunk_index
            and noise.shape == (1, 3, 16, 60, 104)
            and noise.dtype == "bfloat16",
            "2(b) route checkpoint noise identity drifted",
        )
        noise_relative = (
            f"noise/{prompt_id}/seed-{seed}/chunk-{chunk_index:02d}.safetensors"
        )
        noise_path = _safe_run_file(
            run_dir, noise_relative, label=f"2(b) chunk {chunk_index} noise"
        )
        noise_identity = (noise.tensor_sha256, noise.tensor_nbytes)
        prior_noise = verified_noise_files.get(noise_relative)
        if prior_noise is None:
            _require(
                _sha256_file(noise_path) == noise.tensor_sha256
                and noise_path.stat().st_size == noise.tensor_nbytes,
                "2(b) route noise bytes disagree with the committed descriptor",
            )
            verified_noise_files[noise_relative] = noise_identity
        else:
            _require(
                prior_noise == noise_identity,
                "2(b) route reuses one noise path with conflicting descriptors",
            )

        expected_kind = (
            "online_lookahead_coverage_boundary_gated"
            if is_boundary_coverage_online
            else "online_lookahead_coverage_guided"
            if is_plain_lookahead_coverage_online
            else "online_coverage_guided"
            if is_current_coverage_online
            else "online_rank_guided"
            if is_online
            else "fixed_rollout"
        )
        _require(
            metadata.get("kind") == expected_kind
            and metadata.get("route_id") == route_id
            and metadata.get("chunk_index") == chunk_index
            and metadata.get("corrected") is corrected
            and metadata.get("committed_role") == expected_role
            and metadata.get("committed_kv_hash") == context.kv.cache_sha256
            and metadata.get("initial_noise_descriptor_sha256")
            == noise.descriptor_sha256,
            "2(b) route checkpoint metadata disagrees with the committed chunk",
        )
        generated = raw["generated"]
        if online_policy is not None:
            if chunk_index == 0:
                _require(
                    metadata.get("risk_score") is None
                    and metadata.get("draft_proposal") is None,
                    "2(b) online chunk 0 carries a Draft decision",
                )
                if is_lookahead_coverage_online:
                    if is_boundary_coverage_online:
                        _require(
                            set(metadata)
                            == _BOUNDARY_CHECKPOINT_METADATA_ROOT_KEYS,
                            "2(b) boundary-gated root metadata fields are invalid",
                        )
                    _require(
                        metadata.get("gate_signal_chunk_index") is None
                        and metadata.get("gate_signal_rank") is None
                        and metadata.get("gate_probe") is None
                        and metadata.get("gate_probe_parent_history_hash") is None
                        and metadata.get("gate_probe_kv_hash") is None
                        and metadata.get("gate_probe_noise_descriptor_sha256") is None
                        and metadata.get("probe_reused_as_successor") is False
                        and metadata.get("terminal_flush") is False,
                        "2(b) lookahead root carries speculative gate evidence",
                    )
                    if is_boundary_coverage_online:
                        _require(
                            metadata.get("target_candidate_proposed") is False
                            and metadata.get("target_candidate") is None
                            and metadata.get("target_candidate_parent_history_hash")
                            is None
                            and metadata.get("target_candidate_kv_hash") is None
                            and metadata.get(
                                "target_candidate_noise_descriptor_sha256"
                            )
                            is None
                            and metadata.get("corrected_frontier_probe") is None
                            and metadata.get(
                                "corrected_frontier_parent_history_hash"
                            )
                            is None
                            and metadata.get("corrected_frontier_kv_hash") is None
                            and metadata.get(
                                "corrected_frontier_noise_descriptor_sha256"
                            )
                            is None
                            and metadata.get("boundary_gate_decision") is None
                            and metadata.get("target_candidate_accepted") is None
                            and metadata.get("carried_frontier_kind") is None,
                            "2(b) boundary-gated root carries proposal evidence",
                        )
            else:
                proposal = metadata.get("draft_proposal")
                _require(
                    isinstance(proposal, dict),
                    "2(b) online checkpoint omitted its Draft proposal evidence",
                )
                audited_proposal = _audit_draft_generated_record(
                    run_dir=run_dir,
                    generated=proposal,
                    prefix=previous,
                    expected_noise=noise,
                    draft_model_binding=model_bindings["draft"],
                    require_hidden_evidence=False,
                    verified_segment_files=verified_segment_files,
                    verified_hidden_files=verified_hidden_files,
                    label=f"2(b) online Draft proposal chunk {chunk_index}",
                )
                rank = audited_proposal["rank"]
                if is_lookahead_coverage_online:
                    assert isinstance(
                        online_policy,
                        LookaheadCoverageConstrainedOnlineRankGuidedPolicy,
                    )
                    if carried_lookahead_probe is not None:
                        _require(
                            proposal == carried_lookahead_probe,
                            "2(b) carried lookahead frontier was not reused as the "
                            "next pending Draft",
                        )
                    if discarded_lookahead_probes:
                        _require(
                            all(
                                proposal != discarded
                                for discarded in discarded_lookahead_probes
                            ),
                            "2(b) discarded lookahead branch was reused after its "
                            "paired decision",
                        )

                    needs_preview = online_policy.preview_required(
                        chunk_index=chunk_index
                    )
                    terminal_flush = chunk_index == chunk_count - 1
                    gate_probe = metadata.get("gate_probe")
                    signal_chunk_index: int | None = None
                    signal_rank: float | None = None
                    risk: float | None = None
                    drop_score: float | None = None
                    audited_gate: dict[str, Any] | None = None
                    if needs_preview:
                        _require(
                            isinstance(gate_probe, Mapping),
                            "2(b) lookahead checkpoint omitted its frontier probe",
                        )
                        provisional = previous.append(audited_proposal["segment"])
                        try:
                            gate_context = _evidence_context(
                                gate_probe.get("context"),
                                label=(
                                    f"2(b) lookahead gate context chunk "
                                    f"{chunk_index + 1}"
                                ),
                            )
                        except (CorrectionStateError, AttributeError) as error:
                            raise Insight2RunError(
                                "2(b) lookahead gate context is invalid: "
                                f"{error}"
                            ) from error
                        audited_gate = _audit_draft_generated_record(
                            run_dir=run_dir,
                            generated=gate_probe,
                            prefix=provisional,
                            expected_noise=gate_context.noise,
                            draft_model_binding=model_bindings["draft"],
                            require_hidden_evidence=False,
                            verified_segment_files=verified_segment_files,
                            verified_hidden_files=verified_hidden_files,
                            label=(
                                f"2(b) lookahead frontier Draft chunk "
                                f"{chunk_index + 1}"
                            ),
                        )
                        gate_latent = PurePosixPath(
                            audited_gate["segment"].latent_path
                        )
                        expected_gate_parent = PurePosixPath(
                            f"latents/rollouts/{route_key}/"
                            f"chunk-{chunk_index + 1:02d}-draft-preview-from-"
                            f"{chunk_index:02d}"
                        )
                        _require(
                            gate_latent.parent == expected_gate_parent
                            and re.fullmatch(
                                r"attempt-[0-9]{4,}\.safetensors",
                                gate_latent.name,
                            )
                            is not None,
                            "2(b) lookahead frontier latent path is not bound to "
                            "its route and guarded source chunk",
                        )
                        signal_chunk_index = chunk_index + 1
                        signal_rank = audited_gate["rank"]
                        gate_noise = audited_gate["context"].noise
                        gate_noise_relative = (
                            f"noise/{prompt_id}/seed-{seed}/"
                            f"chunk-{signal_chunk_index:02d}.safetensors"
                        )
                        gate_noise_path = _safe_run_file(
                            run_dir,
                            gate_noise_relative,
                            label=(
                                f"2(b) lookahead frontier noise "
                                f"{signal_chunk_index}"
                            ),
                        )
                        gate_noise_identity = (
                            gate_noise.tensor_sha256,
                            gate_noise.tensor_nbytes,
                        )
                        prior_gate_noise = verified_noise_files.get(
                            gate_noise_relative
                        )
                        if prior_gate_noise is None:
                            _require(
                                _sha256_file(gate_noise_path)
                                == gate_noise.tensor_sha256
                                and gate_noise_path.stat().st_size
                                == gate_noise.tensor_nbytes,
                                "2(b) lookahead frontier noise bytes drifted",
                            )
                            verified_noise_files[gate_noise_relative] = (
                                gate_noise_identity
                            )
                        else:
                            _require(
                                prior_gate_noise == gate_noise_identity,
                                "2(b) lookahead frontier reused one noise path "
                                "with a conflicting descriptor",
                            )
                        active_ranks = (*proposal_ranks, rank, signal_rank)
                        if isinstance(
                            online_calibrator, SuddenDropRiskCalibrator
                        ):
                            drop_score = causal_erank_sudden_drop(
                                active_ranks,
                                lookback_chunks=4,
                            )
                            risk = online_calibrator.risk(drop_score)
                        else:
                            risk = online_calibrator.risk(signal_rank)
                    else:
                        _require(
                            online_policy.remaining_budget == 0
                            and gate_probe is None,
                            "2(b) lookahead route reached an unguarded pending "
                            "chunk before exhausting its Target budget",
                        )

                    decision = online_policy.decide(
                        chunk_index=chunk_index,
                        signal_chunk_index=signal_chunk_index,
                        risk_score=risk,
                    )
                    expected_probe_parent = (
                        None
                        if audited_gate is None
                        else audited_gate["context"].history_head_sha256
                    )
                    expected_probe_kv = (
                        None if audited_gate is None else audited_gate["kv_sha256"]
                    )
                    expected_probe_noise = (
                        None
                        if audited_gate is None
                        else audited_gate["context"].noise.descriptor_sha256
                    )
                    if is_boundary_coverage_online:
                        assert boundary_gate is not None
                        target_candidate_raw = metadata.get("target_candidate")
                        corrected_frontier_raw = metadata.get(
                            "corrected_frontier_probe"
                        )
                        audited_target: dict[str, Any] | None = None
                        audited_corrected: dict[str, Any] | None = None
                        gate_decision = None
                        accepted = False
                        if decision.replace_with_target:
                            _require(
                                audited_gate is not None
                                and isinstance(target_candidate_raw, Mapping),
                                "2(b) boundary-gated proposal omitted a paired branch",
                            )
                            audited_target = _audit_lookahead_target_candidate(
                                run_dir=run_dir,
                                generated=target_candidate_raw,
                                prefix=previous,
                                expected_noise=noise,
                                prompt_sha256=prompt_sha256,
                                protocol_id=protocol_id,
                                target_model_binding=model_bindings["target"],
                                diagnostic_records=diagnostic_records,
                                verified_segment_files=verified_segment_files,
                                label=(
                                    f"2(b) boundary-gated Target candidate chunk "
                                    f"{chunk_index}"
                                ),
                            )
                            target_latent = PurePosixPath(
                                audited_target["segment"].latent_path
                            )
                            expected_target_parent = PurePosixPath(
                                f"latents/rollouts/{route_key}/"
                                f"chunk-{chunk_index:02d}-target"
                            )
                            _require(
                                target_latent.parent == expected_target_parent
                                and _GENERATED_ATTEMPT_FILENAME_RE.fullmatch(
                                    target_latent.name
                                )
                                is not None,
                                "2(b) boundary-gated Target candidate path is not "
                                "bound to its route and proposal chunk",
                            )
                            saved_gate = metadata.get("boundary_gate_decision")
                            short_circuit = (
                                isinstance(saved_gate, Mapping)
                                and saved_gate.get("short_circuit") is True
                            )
                            if short_circuit:
                                _require(
                                    corrected_frontier_raw is None,
                                    "2(b) entry short-circuit carries a corrected successor",
                                )
                                gate_decision = boundary_gate.reject_if_entry_worse(
                                    draft_entry_witness_score=(
                                        _audit_recomputed_boundary_velocity_witness(
                                            run_dir=run_dir,
                                            prefix=previous,
                                            segment=audited_proposal["segment"],
                                            diagnostics=proposal.get("diagnostics"),
                                            label=f"2(b) entry short-circuit Draft chunk {chunk_index}",
                                        )["score"]
                                    ),
                                    target_entry_witness_score=(
                                        _audit_recomputed_boundary_velocity_witness(
                                            run_dir=run_dir,
                                            prefix=previous,
                                            segment=audited_target["segment"],
                                            diagnostics=target_candidate_raw.get("diagnostics"),
                                            label=f"2(b) entry short-circuit Target chunk {chunk_index}",
                                        )["score"]
                                    ),
                                )
                                _require(
                                    gate_decision is not None,
                                    "2(b) entry short-circuit has no valid failing entry evidence",
                                )
                            else:
                                _require(
                                    isinstance(corrected_frontier_raw, Mapping),
                                    "2(b) boundary-gated proposal omitted a paired branch",
                                )
                                corrected_prefix = previous.append(
                                    audited_target["segment"]
                                )
                                try:
                                    corrected_context = _evidence_context(
                                        corrected_frontier_raw.get("context"),
                                        label=(
                                            f"2(b) boundary-gated corrected frontier "
                                            f"context chunk {chunk_index + 1}"
                                        ),
                                    )
                                except (CorrectionStateError, AttributeError) as error:
                                    raise Insight2RunError(
                                        "2(b) boundary-gated corrected frontier context "
                                        f"is invalid: {error}"
                                    ) from error
                                audited_corrected = _audit_draft_generated_record(
                                    run_dir=run_dir,
                                    generated=corrected_frontier_raw,
                                    prefix=corrected_prefix,
                                    expected_noise=corrected_context.noise,
                                    draft_model_binding=model_bindings["draft"],
                                    require_hidden_evidence=False,
                                    verified_segment_files=verified_segment_files,
                                    verified_hidden_files=verified_hidden_files,
                                    label=(
                                        f"2(b) boundary-gated corrected frontier Draft "
                                        f"chunk {chunk_index + 1}"
                                    ),
                                )
                                corrected_latent = PurePosixPath(
                                    audited_corrected["segment"].latent_path
                                )
                                expected_corrected_parent = PurePosixPath(
                                    f"latents/rollouts/{route_key}/"
                                    f"chunk-{chunk_index + 1:02d}-"
                                    f"draft-after-target-from-{chunk_index:02d}"
                                )
                                _require(
                                    corrected_latent.parent
                                    == expected_corrected_parent
                                    and _GENERATED_ATTEMPT_FILENAME_RE.fullmatch(
                                        corrected_latent.name
                                    )
                                    is not None
                                    and audited_corrected[
                                        "context"
                                    ].noise.to_dict()
                                    == audited_gate["context"].noise.to_dict(),
                                    "2(b) boundary-gated corrected successor path or "
                                    "same-noise binding drifted",
                                )
                                gate_decision = boundary_gate.decide(
                                    original_successor_erank=audited_gate["rank"],
                                    corrected_successor_erank=audited_corrected[
                                        "rank"
                                    ],
                                    draft_entry_witness_score=float(
                                        _audit_recomputed_boundary_velocity_witness(
                                            run_dir=run_dir,
                                            prefix=previous,
                                            segment=audited_proposal["segment"],
                                            diagnostics=proposal.get("diagnostics"),
                                            label=(
                                                f"2(b) boundary-gated Draft candidate "
                                                f"chunk {chunk_index}"
                                            ),
                                        )["score"]
                                    ),
                                    target_entry_witness_score=float(
                                        _audit_recomputed_boundary_velocity_witness(
                                            run_dir=run_dir,
                                            prefix=previous,
                                            segment=audited_target["segment"],
                                            diagnostics=target_candidate_raw.get(
                                                "diagnostics"
                                            ),
                                            label=(
                                                f"2(b) boundary-gated Target candidate "
                                                f"chunk {chunk_index}"
                                            ),
                                        )["score"]
                                    ),
                                    draft_exit_witness_score=float(
                                        _audit_recomputed_boundary_velocity_witness(
                                            run_dir=run_dir,
                                            prefix=provisional,
                                            segment=audited_gate["segment"],
                                            diagnostics=gate_probe.get("diagnostics"),
                                            label=(
                                                f"2(b) boundary-gated original successor "
                                                f"chunk {chunk_index + 1}"
                                            ),
                                        )["score"]
                                    ),
                                    target_exit_witness_score=float(
                                        _audit_recomputed_boundary_velocity_witness(
                                            run_dir=run_dir,
                                            prefix=corrected_prefix,
                                            segment=audited_corrected["segment"],
                                            diagnostics=corrected_frontier_raw.get(
                                                "diagnostics"
                                            ),
                                            label=(
                                                f"2(b) boundary-gated corrected successor "
                                                f"chunk {chunk_index + 1}"
                                            ),
                                        )["score"]
                                    ),
                                )
                            _require(
                                gate_decision.evidence_valid,
                                "2(b) boundary-gated decision evidence is invalid",
                            )
                            accepted = gate_decision.accept_target
                            replayed_proposed_chunks.append(chunk_index)
                            target_attempt_ids.add(audited_target["attempt_id"])
                            target_logical_call_ids.add(
                                audited_target["logical_call_id"]
                            )
                            proposal_target_attempt_ids.add(
                                audited_target["attempt_id"]
                            )
                            proposal_target_logical_call_ids.add(
                                audited_target["logical_call_id"]
                            )
                            if accepted:
                                replayed_accepted_chunks.append(chunk_index)
                        else:
                            _require(
                                target_candidate_raw is None
                                and corrected_frontier_raw is None
                                and metadata.get("boundary_gate_decision") is None
                                and metadata.get("target_candidate_accepted") is None,
                                "2(b) boundary-gated non-proposal carries paired "
                                "Target evidence",
                            )

                        expected_target_parent_hash = (
                            None
                            if audited_target is None
                            else audited_target["context"].history_head_sha256
                        )
                        expected_target_kv = (
                            None
                            if audited_target is None
                            else audited_target["context"].kv.cache_sha256
                        )
                        expected_target_noise = (
                            None
                            if audited_target is None
                            else audited_target[
                                "context"
                            ].noise.descriptor_sha256
                        )
                        expected_corrected_parent_hash = (
                            None
                            if audited_corrected is None
                            else audited_corrected[
                                "context"
                            ].history_head_sha256
                        )
                        expected_corrected_kv = (
                            None
                            if audited_corrected is None
                            else audited_corrected["kv_sha256"]
                        )
                        expected_corrected_noise = (
                            None
                            if audited_corrected is None
                            else audited_corrected[
                                "context"
                            ].noise.descriptor_sha256
                        )
                        carried_kind = (
                            None
                            if audited_gate is None
                            else "corrected"
                            if accepted
                            else "original"
                        )
                        expected_reuse = carried_kind is not None
                        _require(
                            metadata.get("target_candidate_proposed")
                            is decision.replace_with_target
                            and metadata.get(
                                "target_candidate_parent_history_hash"
                            )
                            == expected_target_parent_hash
                            and metadata.get("target_candidate_kv_hash")
                            == expected_target_kv
                            and metadata.get(
                                "target_candidate_noise_descriptor_sha256"
                            )
                            == expected_target_noise
                            and metadata.get(
                                "corrected_frontier_parent_history_hash"
                            )
                            == expected_corrected_parent_hash
                            and metadata.get("corrected_frontier_kv_hash")
                            == expected_corrected_kv
                            and metadata.get(
                                "corrected_frontier_noise_descriptor_sha256"
                            )
                            == expected_corrected_noise
                            and metadata.get("boundary_gate_decision")
                            == (
                                None
                                if gate_decision is None
                                else asdict(gate_decision)
                            )
                            and metadata.get("target_candidate_accepted")
                            == (
                                accepted
                                if decision.replace_with_target
                                else None
                            )
                            and metadata.get("carried_frontier_kind")
                            == carried_kind
                            and metadata.get("probe_reused_as_successor")
                            is expected_reuse
                            and metadata.get("corrected") is accepted
                            and (
                                generated == target_candidate_raw
                                if accepted
                                else generated == proposal
                            ),
                            "2(b) boundary-gated paired decision or branch reuse "
                            "drifted",
                        )
                        if accepted:
                            assert corrected_frontier_raw is not None
                            carried_lookahead_probe = corrected_frontier_raw
                            discarded_lookahead_probes = (gate_probe,)
                        else:
                            carried_lookahead_probe = gate_probe
                            discarded_lookahead_probes = (
                                ()
                                if corrected_frontier_raw is None
                                else (corrected_frontier_raw,)
                            )
                            proposal_ranks.append(rank)
                    else:
                        expected_reuse = (
                            audited_gate is not None
                            and not decision.replace_with_target
                        )
                        _require(
                            metadata.get("probe_reused_as_successor")
                            is expected_reuse
                            and metadata.get("corrected")
                            is decision.replace_with_target,
                            "2(b) lookahead branch reuse or correction decision "
                            "drifted",
                        )
                    _require(
                        metadata.get("rank") == rank
                        and metadata.get("risk_score") == risk
                        and metadata.get("decision") == asdict(decision)
                        and metadata.get("draft_parent_history_hash")
                        == previous.head_sha256
                        and metadata.get("draft_kv_hash")
                        == audited_proposal["kv_sha256"]
                        and metadata.get("gate_signal_chunk_index")
                        == signal_chunk_index
                        and metadata.get("gate_signal_rank") == signal_rank
                        and metadata.get("gate_probe_parent_history_hash")
                        == expected_probe_parent
                        and metadata.get("gate_probe_kv_hash")
                        == expected_probe_kv
                        and metadata.get(
                            "gate_probe_noise_descriptor_sha256"
                        )
                        == expected_probe_noise
                        and metadata.get("terminal_flush") is terminal_flush,
                        "2(b) lookahead rank, probe binding, or chronological "
                        "decision drifted",
                    )
                    if drop_score is None:
                        _require(
                            "erank_drop_score" not in metadata,
                            "2(b) lookahead checkpoint has an unexpected "
                            "eRank-drop score",
                        )
                    else:
                        _require(
                            metadata.get("erank_drop_score") == drop_score,
                            "2(b) lookahead causal eRank-drop evidence drifted",
                        )
                    if is_boundary_coverage_online:
                        expected_boundary_metadata_keys = (
                            _BOUNDARY_CHECKPOINT_METADATA_NONROOT_KEYS
                            | ({"erank_drop_score"} if drop_score is not None else set())
                        )
                        _require(
                            set(metadata) == expected_boundary_metadata_keys,
                            "2(b) boundary-gated checkpoint metadata fields are invalid",
                        )
                    if not is_boundary_coverage_online:
                        if decision.replace_with_target:
                            carried_lookahead_probe = None
                            discarded_lookahead_probes = (gate_probe,)
                        else:
                            _require(
                                generated == proposal,
                                "2(b) lookahead accepted Draft differs from its "
                                "pending proposal",
                            )
                            proposal_ranks.append(rank)
                            carried_lookahead_probe = gate_probe
                            discarded_lookahead_probes = ()
                else:
                    proposal_ranks.append(rank)
                    if isinstance(online_calibrator, SuddenDropRiskCalibrator):
                        drop_score = causal_erank_sudden_drop(
                            proposal_ranks,
                            lookback_chunks=4,
                        )
                        risk = online_calibrator.risk(drop_score)
                        _require(
                            metadata.get("erank_drop_score") == drop_score,
                            "2(b) online causal eRank-drop evidence drifted",
                        )
                    else:
                        risk = online_calibrator.risk(rank)
                    decision = online_policy.decide(
                        chunk_index=chunk_index, risk_score=risk
                    )
                    _require(
                        metadata.get("rank") == rank
                        and metadata.get("risk_score") == risk
                        and metadata.get("decision") == asdict(decision)
                        and metadata.get("draft_parent_history_hash")
                        == previous.head_sha256
                        and metadata.get("draft_kv_hash")
                        == audited_proposal["kv_sha256"]
                        and metadata.get("corrected")
                        is decision.replace_with_target,
                        "2(b) online rank, risk, or chronological decision drifted",
                    )
                    if not decision.replace_with_target:
                        _require(
                            generated == proposal,
                            "2(b) online accepted Draft differs from its decision "
                            "proposal",
                        )
        if expected_role == "target":
            attempt_id, logical_call_id = _audit_target_checkpoint_binding(
                generated=generated,
                context=context,
                segment=segment,
                prefix=previous,
                prompt_sha256=prompt_sha256,
                protocol_id=protocol_id,
                target_model_binding=model_bindings["target"],
                diagnostic_records=diagnostic_records,
            )
            target_attempt_ids.add(attempt_id)
            target_logical_call_ids.add(logical_call_id)
        else:
            audited_committed_draft = _audit_draft_generated_record(
                run_dir=run_dir,
                generated=generated,
                prefix=previous,
                expected_noise=noise,
                draft_model_binding=model_bindings["draft"],
                require_hidden_evidence=False,
                verified_segment_files=verified_segment_files,
                verified_hidden_files=verified_hidden_files,
                label=f"2(b) committed Draft chunk {chunk_index}",
            )
            _require(
                metadata.get("committed_kv_hash")
                == audited_committed_draft["kv_sha256"]
                and metadata.get("draft_kv_hash")
                == audited_committed_draft["kv_sha256"]
                and metadata.get("draft_parent_history_hash")
                == previous.head_sha256
                and metadata.get("rank" if is_online else "draft_rank")
                == audited_committed_draft["rank"],
                "2(b) committed Draft evidence differs from checkpoint metadata",
            )
        if chunk_index == 0:
            chunk_zero_binding_sha256 = hashlib.sha256(
                _canonical_json_bytes(
                    {"history": raw["history"], "generated": raw["generated"]}
                )
            ).hexdigest()
        previous = history

    if online_policy is not None:
        replayed_route = online_policy.finalize()
        _require(
            replayed_route
            == (proposed_chunks if is_boundary_coverage_online else corrected_chunks),
            "2(b) online replay selected different proposal positions",
        )
        if is_lookahead_coverage_online:
            _require(
                bool(proposed_chunks),
                "2(b) lookahead route has no final proposal boundary",
            )
            if is_boundary_coverage_online:
                _require(
                    tuple(replayed_proposed_chunks) == proposed_chunks
                    and tuple(replayed_accepted_chunks) == corrected_chunks,
                    "2(b) boundary-gated replay proposed or accepted a different set",
                )
            _audit_lookahead_preview_path_budget(
                run_dir=run_dir,
                route_key=route_key,
                corrected_chunks=corrected_chunks,
                chunk_count=chunk_count,
                proposed_target_chunks=(
                    proposed_chunks if is_boundary_coverage_online else None
                ),
            )
    _require(
        previous.next_chunk_index == chunk_count,
        "2(b) route did not produce the protocol-owned chunk horizon",
    )
    actual_corrected = tuple(
        item.chunk_index
        for item in previous.segments[1:]
        if item.producer_role == "target"
    )
    _require(
        actual_corrected == corrected_chunks,
        "2(b) route Target positions differ from the selected chunks",
    )

    canonical_key = hashlib.sha256(
        _canonical_json_bytes([protocol_id, "rollout-root", prompt_id, seed])
    ).hexdigest()[:48]
    canonical_relative = f"state/rollout-roots/{canonical_key}/chunk-00.json"
    canonical_path = _safe_run_file(
        run_dir, canonical_relative, label="2(b) canonical rollout root"
    )
    canonical_digest = canonical_root_cache.get((prompt_id, seed))
    if canonical_digest is None:
        canonical_raw, canonical_history, canonical_context, _segment, canonical_metadata = (
            _load_audited_checkpoint(
                canonical_path, label="2(b) canonical rollout root"
            )
        )
        _require(
            canonical_history.next_chunk_index == 1
            and canonical_history.prompt_id == prompt_id
            and canonical_history.trajectory_seed == seed
            and canonical_context.role == "target"
            and canonical_metadata
            == {
                "kind": "canonical_rollout_root",
                "chunk_index": 0,
                "corrected": True,
            },
            "2(b) canonical rollout root is invalid",
        )
        canonical_digest = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "history": canonical_raw["history"],
                    "generated": canonical_raw["generated"],
                }
            )
        ).hexdigest()
        canonical_root_cache[(prompt_id, seed)] = canonical_digest
    _require(
        chunk_zero_binding_sha256 == canonical_digest,
        "2(b) route chunk 0 differs from the shared canonical Target root",
    )

    measured_quality, measured_temporal = _audit_quality_measurement(
        run_dir=run_dir,
        measurement_id=f"2b-{route_key}",
        history=previous,
        evaluation_kind="full",
        chunk_index=None,
        quality_model_binding=quality_model_binding,
        verified_decode_files=verified_decode_files,
        measurement_schema=quality_measurement_schema,
        primary_metric_id=primary_quality_metric_id,
    )

    score_relative = f"measurements/rollouts/{route_key}.json"
    score_path = _safe_run_file(run_dir, score_relative, label="2(b) rollout score")
    score = _load_json_object(score_path, label="2(b) rollout score")
    _require(
        set(score)
        == {
            "schema",
            "route_id",
            "corrected_chunks",
            "proposed_target_chunks",
            "full_video_quality",
            "temporal_consistency",
            "target_call_count",
            "budgeted_target_call_count",
            "final_history_hash",
        }
        and score.get("schema") == ROLLOUT_SCORE_SCHEMA
        and score.get("route_id") == route_id
        and score.get("corrected_chunks") == list(corrected_chunks)
        and score.get("proposed_target_chunks") == list(proposed_chunks)
        and score.get("target_call_count") == 1 + len(corrected_chunks)
        and score.get("budgeted_target_call_count") == 1 + len(proposed_chunks)
        and score.get("final_history_hash") == previous.head_sha256
        and score.get("full_video_quality") == measured_quality
        and score.get("temporal_consistency") == measured_temporal,
        "2(b) rollout score does not bind the reconstructed route",
    )
    for field in ("full_video_quality", "temporal_consistency"):
        value = score.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"2(b) rollout score {field} is invalid",
        )
    summary = {
        "identity": [
            prompt_id,
            seed,
            route_id,
            list(corrected_chunks),
            list(proposed_chunks),
        ],
        "route_key": route_key,
        "full_video_quality": float(score["full_video_quality"]),
        "temporal_consistency": float(score["temporal_consistency"]),
        "final_history_hash": previous.head_sha256,
        "target_attempt_ids": frozenset(target_attempt_ids),
        "target_logical_call_ids": frozenset(target_logical_call_ids),
        "proposal_target_attempt_ids": frozenset(
            proposal_target_attempt_ids
        ),
        "proposal_target_logical_call_ids": frozenset(
            proposal_target_logical_call_ids
        ),
    }
    route_cache[route_key] = summary
    return summary


def _audit_boundary_proposal_lifecycle(
    *,
    route_summary: Mapping[str, Any],
    attempts: Mapping[str, Mapping[str, Any]],
    scope_id: str,
    proposal_count: int,
) -> None:
    """Bind every accepted or rejected Target proposal to its result-row scope."""

    attempt_ids = route_summary.get("proposal_target_attempt_ids")
    logical_call_ids = route_summary.get("proposal_target_logical_call_ids")
    _require(
        isinstance(attempt_ids, frozenset)
        and isinstance(logical_call_ids, frozenset)
        and len(attempt_ids) == proposal_count
        and len(logical_call_ids) == proposal_count,
        "2(b) boundary-gated Target proposal lifecycle count drifted",
    )
    for attempt_id in attempt_ids:
        state = attempts.get(attempt_id)
        _require(
            isinstance(state, Mapping)
            and state.get("scope_id") == scope_id
            and state.get("phase") == "route_materialization"
            and state.get("completed") is not None
            and state.get("logical_call_id") in logical_call_ids,
            "2(b) boundary-gated Target proposal is not bound to its raw-row "
            "route-materialization lifecycle scope",
        )


def audit_target_rpc_accounting(
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    preflight_result: Preflight | None = None,
) -> dict[str, Any]:
    """Reconcile worker-fsynced Target attempts with success events and raw rows."""

    _require(phase in PHASES, "cannot audit Target RPCs for an unknown phase")
    try:
        lifecycle = load_target_rpc_lifecycle(run_dir, required=True)
    except Exception as error:
        raise Insight2RunError(f"Target RPC lifecycle audit failed: {error}") from error

    attempts: dict[str, dict[str, Any]] = {}
    for record in lifecycle:
        attempt_id = str(record["attempt_id"])
        state = attempts.setdefault(
            attempt_id,
            {
                "logical_call_id": record["logical_call_id"],
                "scope_id": record["scope_id"],
                "phase": record["phase"],
                "statuses": [],
                "completed": None,
            },
        )
        state["statuses"].append(record["status"])
        if record["status"] == "completed":
            state["completed"] = record
    _require(
        all(
            state["scope_id"] is not None and state["phase"] != "unscoped"
            for state in attempts.values()
        ),
        "formal Target RPC lifecycle contains an unscoped attempt",
    )
    allowed_phases = (
        {"calibration_baseline", "evaluation_baseline", "single_point_candidate"}
        if phase.startswith("2a-")
        else {"calibration_baseline", "route_materialization", "selection_search"}
    )
    _require(
        all(state["phase"] in allowed_phases for state in attempts.values()),
        "formal Target RPC lifecycle contains a phase outside the selected experiment",
    )
    _require(
        all(
            state["statuses"] == ["started", "backend_returned", "completed"]
            for state in attempts.values()
        ),
        "successful formal run contains an incomplete Target RPC attempt",
    )
    attempts_by_scope: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for attempt_id, state in attempts.items():
        scope_id = state["scope_id"]
        assert isinstance(scope_id, str)
        attempts_by_scope.setdefault(scope_id, []).append((attempt_id, state))

    campaign_path = _safe_run_file(run_dir, "campaign.json", label="campaign manifest")
    campaign_manifest = _load_json_object(campaign_path, label="campaign manifest")
    campaign_keys = {
        "schema",
        "protocol_id",
        "seeds",
        "noise_namespace",
        "models",
        "calibration_prompts",
        "evaluation_prompts",
        "smoke_prompts",
    }
    if "chunk_count" in campaign_manifest:
        campaign_keys.add("chunk_count")
    _require(
        set(campaign_manifest) == campaign_keys
        and (preflight_result is None or "chunk_count" in campaign_manifest)
        and campaign_manifest.get("schema") == CAMPAIGN_SCHEMA
        and campaign_path.read_bytes()
        == _canonical_json_bytes(campaign_manifest) + b"\n",
        "campaign manifest fields, schema, or canonical encoding drifted",
    )
    if preflight_result is not None:
        _audit_trusted_run_inputs(
            run_dir=run_dir,
            phase=phase,
            campaign_manifest=campaign_manifest,
            preflight_result=preflight_result,
        )
    campaign_protocol_id = campaign_manifest.get("protocol_id")
    _require(
        isinstance(campaign_protocol_id, str) and bool(campaign_protocol_id),
        "campaign manifest protocol_id is invalid",
    )
    campaign_chunk_count = campaign_manifest.get("chunk_count", 16)
    _require(
        isinstance(campaign_chunk_count, int)
        and not isinstance(campaign_chunk_count, bool)
        and campaign_chunk_count >= 2,
        "campaign manifest chunk_count is invalid",
    )
    _audit_generated_chunk_path_horizon(
        run_dir=run_dir,
        chunk_count=campaign_chunk_count,
    )
    if preflight_result is not None:
        _require(
            campaign_chunk_count
            == int(preflight_result.protocol.raw["design"]["chunk_count"]),
            "campaign manifest chunk_count differs from the trusted protocol",
        )
        audit_budgets = frozenset(preflight_result.config.budgets)
        audit_strategies = frozenset(preflight_result.config.strategies)
        audit_random_repeats = preflight_result.config.random_repeats
        coverage_values = (
            getattr(
                preflight_result.config,
                "coverage_window_radius_chunks",
                None,
            ),
            getattr(
                preflight_result.config,
                "coverage_minimum_gap_chunks",
                None,
            ),
            getattr(
                preflight_result.config,
                "coverage_maximum_gap_chunks",
                None,
            ),
        )
        audit_coverage_constraint = (
            coverage_values
            if all(isinstance(value, int) for value in coverage_values)
            else None
        )
        if preflight_result.config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS:
            protocol_quality = preflight_result.protocol.raw.get("quality")
            _require(
                isinstance(protocol_quality, Mapping)
                and protocol_quality.get("primary")
                == VBENCH_LONG_PRIMARY_QUALITY_METRIC,
                "long-video protocol quality identity is invalid",
            )
            quality_measurement_schema = VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA
            primary_quality_metric_id = str(protocol_quality["primary"])
        else:
            quality_measurement_schema = QUALITY_MEASUREMENT_SCHEMA
            primary_quality_metric_id = PRIMARY_QUALITY_METRIC
    else:
        # Backward-compatible standalone evidence audit for v1 fixtures.  A
        # formal run always supplies ``preflight_result`` and therefore binds
        # these choices to its frozen config and protocol.
        audit_budgets = frozenset((2, 3, 4))
        audit_strategies = frozenset(LEGACY_STRATEGIES)
        audit_random_repeats = 10
        audit_coverage_constraint = None
        quality_measurement_schema = QUALITY_MEASUREMENT_SCHEMA
        primary_quality_metric_id = PRIMARY_QUALITY_METRIC
    models = campaign_manifest.get("models")
    target_model = models.get("target") if isinstance(models, dict) else None
    draft_model = models.get("draft") if isinstance(models, dict) else None
    _require(
        isinstance(target_model, dict)
        and isinstance(target_model.get("id"), str)
        and bool(target_model["id"])
        and isinstance(target_model.get("artifact_sha256"), str)
        and _SHA256_RE.fullmatch(target_model["artifact_sha256"]) is not None,
        "campaign manifest Target model binding is invalid",
    )
    _require(
        isinstance(draft_model, dict)
        and isinstance(draft_model.get("id"), str)
        and bool(draft_model["id"])
        and isinstance(draft_model.get("artifact_sha256"), str)
        and _SHA256_RE.fullmatch(draft_model["artifact_sha256"]) is not None,
        "campaign manifest Draft model binding is invalid",
    )
    target_model_binding = (
        str(target_model["id"]),
        str(target_model["artifact_sha256"]),
    )
    draft_model_binding = (
        str(draft_model["id"]),
        str(draft_model["artifact_sha256"]),
    )
    quality_model_binding = _load_quality_model_binding(
        run_dir=run_dir,
        phase=phase,
        draft_model_binding=draft_model_binding,
    )
    verified_decode_files: dict[str, str] = {}
    prompt_bindings: set[tuple[str, str]] = set()
    for split_key in ("calibration_prompts", "evaluation_prompts"):
        prompt_entries = campaign_manifest.get(split_key)
        _require(
            isinstance(prompt_entries, list),
            f"campaign manifest {split_key} is invalid",
        )
        for entry in prompt_entries:
            _require(
                isinstance(entry, dict)
                and isinstance(entry.get("prompt_id"), str)
                and bool(entry["prompt_id"])
                and isinstance(entry.get("prompt"), str)
                and bool(str(entry["prompt"]).strip()),
                f"campaign manifest {split_key} entry is invalid",
            )
            binding = (
                str(entry["prompt_id"]),
                hashlib.sha256(str(entry["prompt"]).encode("utf-8")).hexdigest(),
            )
            _require(
                binding not in prompt_bindings,
                "campaign manifest contains a duplicate prompt binding",
            )
            prompt_bindings.add(binding)

    success_path = run_dir / "logs" / "target-rpc-events.jsonl"
    successes: dict[str, dict[str, Any]] = {}
    if success_path.exists():
        _require(
            success_path.is_file() and not success_path.is_symlink(),
            "Target RPC success journal must be a regular file",
        )
        try:
            lines = success_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise Insight2RunError(f"cannot read Target RPC success journal: {error}") from error
        expected_keys = {
            "schema",
            "event_id",
            "attempt_id",
            "logical_call_id",
            "scope_id",
            "phase",
            "chunk_index",
            "latent_path",
            "latent_sha256",
            "segment_sha256",
            "history_head_sha256",
            "kv_cache_sha256",
            "noise_descriptor_sha256",
            "diagnostics_path",
            "diagnostics_sha256",
        }
        event_ids: set[str] = set()
        for line_number, raw in enumerate(lines, 1):
            _require(bool(raw), f"Target RPC success journal line {line_number} is blank")
            try:
                event = json.loads(
                    raw,
                    object_pairs_hook=_duplicate_checked_object,
                    parse_constant=lambda item: (_ for _ in ()).throw(
                        Insight2RunError(
                            f"Target RPC success journal contains invalid constant {item!r}"
                        )
                    ),
                )
            except Insight2RunError:
                raise
            except (json.JSONDecodeError, UnicodeError) as error:
                raise Insight2RunError(
                    f"Target RPC success journal line {line_number} is invalid: {error}"
                ) from error
            _require(
                isinstance(event, dict) and set(event) == expected_keys,
                "Target RPC success event fields are invalid",
            )
            _require(event["schema"] == TARGET_RPC_SCHEMA, "Target RPC success schema drifted")
            event_id = event["event_id"]
            _require(
                isinstance(event_id, str)
                and _SHA256_RE.fullmatch(event_id) is not None
                and event_id
                == hashlib.sha256(
                    _canonical_json_bytes(
                        {key: value for key, value in event.items() if key != "event_id"}
                    )
                ).hexdigest(),
                "Target RPC success event digest drifted",
            )
            _require(event_id not in event_ids, "Target RPC success event is duplicated")
            event_ids.add(event_id)
            attempt_id = event["attempt_id"]
            _require(
                isinstance(attempt_id, str) and _SHA256_RE.fullmatch(attempt_id) is not None,
                "Target RPC success attempt_id is invalid",
            )
            _require(attempt_id not in successes, "Target RPC attempt has duplicate success events")
            successes[attempt_id] = event
    _require(
        success_path.exists() or not any(state["completed"] for state in attempts.values()),
        "completed Target attempts have no success journal",
    )

    diagnostic_records: dict[str, dict[str, Any]] = {}
    referenced_diagnostic_paths: set[str] = set()
    verified_noise_files: dict[str, tuple[str, int]] = {}
    verified_target_latent_files: dict[str, tuple[str, int]] = {}
    for attempt_id, state in attempts.items():
        completed = state["completed"]
        success = successes.get(attempt_id)
        _require(
            (completed is None) == (success is None),
            "Target completed attempt and success journal disagree",
        )
        if completed is None:
            continue
        assert success is not None
        for lifecycle_key, success_key in (
            ("logical_call_id", "logical_call_id"),
            ("scope_id", "scope_id"),
            ("phase", "phase"),
            ("chunk_index", "chunk_index"),
            ("output_latent_path", "latent_path"),
            ("output_latent_sha256", "latent_sha256"),
            ("kv_witness_sha256", "kv_cache_sha256"),
        ):
            _require(
                completed[lifecycle_key] == success[success_key],
                f"Target success event disagrees on {success_key}",
            )
        diagnostics, diagnostics_path = _audit_target_rpc_diagnostics(
            run_dir=run_dir,
            attempt_id=attempt_id,
            completed=completed,
            success=success,
            protocol_id=campaign_protocol_id,
            prompt_bindings=prompt_bindings,
            target_model_binding=target_model_binding,
            chunk_count=campaign_chunk_count,
            verified_noise_files=verified_noise_files,
            verified_latent_files=verified_target_latent_files,
        )
        _require(
            diagnostics_path not in referenced_diagnostic_paths,
            "Target RPC diagnostics path is referenced more than once",
        )
        referenced_diagnostic_paths.add(diagnostics_path)
        diagnostic_records[attempt_id] = diagnostics
    _require(
        set(successes) <= set(attempts),
        "Target success journal references an unknown lifecycle attempt",
    )

    diagnostics_root = run_dir / "provenance" / "target-rpc-diagnostics"
    if diagnostics_root.exists():
        _require(
            diagnostics_root.is_dir() and not diagnostics_root.is_symlink(),
            "Target RPC diagnostics root is not a regular directory",
        )
        actual_diagnostic_paths: set[str] = set()
        for path in diagnostics_root.iterdir():
            _require(
                path.is_file() and not path.is_symlink(),
                "Target RPC diagnostics root contains a non-regular entry",
            )
            _require(
                path.suffix == ".json"
                and _SHA256_RE.fullmatch(path.stem) is not None,
                "Target RPC diagnostics root contains an invalid filename",
            )
            actual_diagnostic_paths.add(path.relative_to(run_dir).as_posix())
        _require(
            actual_diagnostic_paths == referenced_diagnostic_paths,
            "Target RPC diagnostics contain an orphan or omit a completed attempt",
        )
    else:
        _require(
            not referenced_diagnostic_paths,
            "completed Target attempts omitted their diagnostics directory",
        )

    scoped_attempts = {
        attempt_id: state
        for attempt_id, state in attempts.items()
        if state["scope_id"] is not None
    }
    reconciled_non_row_scopes = 0
    if phase.startswith("2a-"):
        campaign = _load_json_object(run_dir / "campaign.json", label="campaign manifest")
        protocol_id = campaign.get("protocol_id")
        _require(
            isinstance(protocol_id, str) and bool(protocol_id),
            "campaign manifest protocol_id is invalid",
        )
        models = campaign.get("models")
        target_model = models.get("target") if isinstance(models, dict) else None
        draft_model = models.get("draft") if isinstance(models, dict) else None
        _require(
            isinstance(target_model, dict)
            and isinstance(target_model.get("id"), str)
            and bool(target_model["id"])
            and isinstance(target_model.get("artifact_sha256"), str)
            and _SHA256_RE.fullmatch(target_model["artifact_sha256"]) is not None,
            "campaign manifest Target model binding is invalid",
        )
        _require(
            isinstance(draft_model, dict)
            and isinstance(draft_model.get("id"), str)
            and bool(draft_model["id"])
            and isinstance(draft_model.get("artifact_sha256"), str)
            and _SHA256_RE.fullmatch(draft_model["artifact_sha256"]) is not None,
            "campaign manifest Draft model binding is invalid",
        )
        seeds_for_rank = campaign.get("seeds")
        calibration_for_rank = campaign.get("calibration_prompts")
        evaluation_for_rows = campaign.get("evaluation_prompts")
        smoke_for_rows = campaign.get("smoke_prompts")
        _require(
            isinstance(seeds_for_rank, list)
            and all(
                isinstance(seed, int) and not isinstance(seed, bool)
                for seed in seeds_for_rank
            )
            and isinstance(calibration_for_rank, list)
            and isinstance(evaluation_for_rows, list)
            and isinstance(smoke_for_rows, list),
            "campaign manifest cannot define 2(a) rank evidence",
        )
        model_bindings = {
            "draft": (
                str(draft_model["id"]),
                str(draft_model["artifact_sha256"]),
            ),
            "target": (
                str(target_model["id"]),
                str(target_model["artifact_sha256"]),
            ),
        }
        verified_hidden_files: dict[str, tuple[str, int]] = {}
        verified_latents = verified_target_latent_files
        calibrator = _audit_rank_calibration(
            run_dir=run_dir,
            protocol_id=protocol_id,
            calibration_prompts=calibration_for_rank,
            seeds=seeds_for_rank,
            chunk_count=campaign_chunk_count,
            model_bindings=model_bindings,
            verified_segment_files=verified_latents,
            verified_hidden_files=verified_hidden_files,
        )
        expected_result_split = "smoke" if phase == "2a-smoke" else "evaluation"
        selected_prompt_rows = (
            smoke_for_rows if expected_result_split == "smoke" else evaluation_for_rows
        )
        evaluation_prompt_entries = {
            str(entry.get("prompt_id")): entry
            for entry in selected_prompt_rows
            if isinstance(entry, dict)
            and isinstance(entry.get("prompt_id"), str)
            and isinstance(entry.get("prompt"), str)
            and entry.get("split") == "evaluation"
        }
        _require(
            len(evaluation_prompt_entries) == len(selected_prompt_rows),
            "campaign selected evaluation prompt bindings are invalid",
        )
        expected_pairs = {
            (prompt_id, seed)
            for prompt_id in evaluation_prompt_entries
            for seed in seeds_for_rank
        }
        if preflight_result is not None:
            _require(
                len(rows) == len(expected_pairs) * (campaign_chunk_count - 1),
                "2(a) raw rows do not contain the exact selected prompt/seed grid",
        )
        raw_scope_ids: set[str] = set()
        previous_baseline_by_pair: dict[tuple[str, int], LatentHistory] = {}
        _require(
            all(
                isinstance(item, Mapping)
                and isinstance(item.get("prompt_id"), str)
                and isinstance(item.get("seed"), int)
                and not isinstance(item.get("seed"), bool)
                and isinstance(item.get("chunk_index"), int)
                and not isinstance(item.get("chunk_index"), bool)
                for item in rows
            ),
            "2(a) raw row sort identities are invalid",
        )
        ordered_rows = sorted(
            rows,
            key=lambda item: (
                str(item.get("prompt_id")),
                int(item.get("seed", -1)),
                int(item.get("chunk_index", -1)),
            ),
        )
        for row in ordered_rows:
            _require(row.get("schema") == RAW_2A_SCHEMA, "2(a) raw row schema drifted")
            scope_id = row.get("event_id")
            _require(
                isinstance(scope_id, str) and _SHA256_RE.fullmatch(scope_id) is not None,
                "2(a) raw row event_id is invalid",
            )
            _require(scope_id not in raw_scope_ids, "2(a) raw row event_id is duplicated")
            raw_scope_ids.add(scope_id)
            evidence, _evidence_sha256 = _audit_single_point_evidence(
                run_dir=run_dir,
                row=row,
                verified_latents=verified_latents,
            )
            _require(
                evidence["target_model_id"] == target_model["id"]
                and evidence["target_model_artifact_sha256"]
                == target_model["artifact_sha256"],
                "2(a) fork Target model differs from the campaign binding",
            )
            prompt_entry = evaluation_prompt_entries.get(str(row.get("prompt_id")))
            _require(
                prompt_entry is not None,
                "2(a) raw row is not bound to an evaluation prompt",
            )
            draft_history, target_history = _audit_2a_baseline_binding(
                run_dir=run_dir,
                row=row,
                protocol_id=protocol_id,
                expected_split=expected_result_split,
                prompt_entry=prompt_entry,
                calibrator=calibrator,
                chunk_count=campaign_chunk_count,
                model_bindings=model_bindings,
                verified_segment_files=verified_latents,
                verified_hidden_files=verified_hidden_files,
            )
            pair = (str(row["prompt_id"]), int(row["seed"]))
            chunk_index = int(row["chunk_index"])
            previous_baseline = previous_baseline_by_pair.get(pair)
            if previous_baseline is None:
                _require(
                    pair in expected_pairs
                    and chunk_index == 1
                    and draft_history.prefix_before(chunk_index).next_chunk_index == 1,
                    "2(a) baseline trajectory does not start at candidate chunk 1",
                )
            else:
                _require(
                    chunk_index == previous_baseline.next_chunk_index
                    and draft_history.prefix_before(chunk_index).to_dict()
                    == previous_baseline.to_dict(),
                    "2(a) baseline checkpoints break prompt/seed trajectory continuity",
                )
            previous_baseline_by_pair[pair] = draft_history
            draft_quality, draft_temporal = _audit_quality_measurement(
                run_dir=run_dir,
                measurement_id=f"2a-{scope_id}-draft",
                history=draft_history,
                evaluation_kind="local",
                chunk_index=int(row["chunk_index"]),
                quality_model_binding=quality_model_binding,
                verified_decode_files=verified_decode_files,
                measurement_schema=quality_measurement_schema,
                primary_metric_id=primary_quality_metric_id,
            )
            target_quality, target_temporal = _audit_quality_measurement(
                run_dir=run_dir,
                measurement_id=f"2a-{scope_id}-target",
                history=target_history,
                evaluation_kind="local",
                chunk_index=int(row["chunk_index"]),
                quality_model_binding=quality_model_binding,
                verified_decode_files=verified_decode_files,
                measurement_schema=quality_measurement_schema,
                primary_metric_id=primary_quality_metric_id,
            )
            _require(
                row.get("Q_D") == draft_quality
                and row.get("Q_T") == target_quality
                and row.get("delta_Q") == target_quality - draft_quality
                and row.get("TC_D") == draft_temporal
                and row.get("TC_T") == target_temporal
                and row.get("delta_TC") == target_temporal - draft_temporal,
                "2(a) raw quality or temporal gain differs from audited measurements",
            )
            scoped = attempts_by_scope.get(scope_id, [])
            selected = [state for _attempt_id, state in scoped]
            _require(bool(selected), "2(a) raw row has no Target lifecycle scope")
            _require(
                all(state["phase"] == "single_point_candidate" for state in selected),
                "2(a) raw row scope contains a non-candidate Target phase",
            )
            completed_selected = [
                (attempt_id, state)
                for attempt_id, state in scoped
                if state["completed"] is not None
            ]
            _require(
                len(completed_selected) == 1,
                "2(a) raw row must bind exactly one completed Target call",
            )
            attempt_id, state = completed_selected[0]
            completed = state["completed"]
            success = successes.get(attempt_id)
            assert isinstance(completed, dict) and isinstance(success, dict)
            logical_call_id = row.get("target_logical_call_id")
            _require(
                isinstance(logical_call_id, str)
                and _SHA256_RE.fullmatch(logical_call_id) is not None,
                "2(a) raw target_logical_call_id is invalid",
            )
            _require(
                row.get("target_rpc_attempt_id") == attempt_id,
                "2(a) raw Target attempt ID disagrees with its unique completed call",
            )
            _require(
                state["logical_call_id"] == logical_call_id,
                "2(a) raw Target logical call ID disagrees with lifecycle",
            )
            expected_logical_call_id = hashlib.sha256(
                _canonical_json_bytes(
                    [
                        "target-logical-call-v2",
                        protocol_id,
                        evidence["target_model_id"],
                        evidence["target_model_artifact_sha256"],
                        row.get("prompt_id"),
                        row.get("condition_hash"),
                        row.get("prefix_hash"),
                        row.get("chunk_index"),
                        evidence["noise_descriptor"],
                    ]
                )
            ).hexdigest()
            _require(
                logical_call_id == expected_logical_call_id,
                "2(a) Target logical call ID is not the deterministic fork identity",
            )
            expected_bindings = {
                "scope_id": scope_id,
                "phase": "single_point_candidate",
                "chunk_index": row.get("chunk_index"),
                "logical_call_id": logical_call_id,
                "model_id": evidence["target_model_id"],
                "model_artifact_sha256": evidence["target_model_artifact_sha256"],
                "prompt_sha256": row.get("condition_hash"),
                "internal_seed": evidence["noise_derived_seed"],
                "history_latent_paths": evidence["prefix_latent_paths"],
                "history_chunk_count": row.get("chunk_index"),
                "initial_noise_path": evidence["noise_path"],
                "output_latent_path": row.get("target_latent_path"),
                "output_latent_sha256": row.get("target_latent_sha256"),
                "kv_witness_sha256": row.get("target_kv_hash"),
            }
            for field, expected in expected_bindings.items():
                _require(
                    completed.get(field) == expected,
                    f"2(a) Target lifecycle {field} disagrees with its raw fork",
                )
            for field, expected in (
                ("scope_id", scope_id),
                ("phase", "single_point_candidate"),
                ("chunk_index", row.get("chunk_index")),
                ("logical_call_id", logical_call_id),
                ("latent_path", row.get("target_latent_path")),
                ("latent_sha256", row.get("target_latent_sha256")),
                ("segment_sha256", row.get("target_segment_hash")),
                ("history_head_sha256", row.get("prefix_hash")),
                ("kv_cache_sha256", row.get("target_kv_hash")),
                ("noise_descriptor_sha256", evidence["noise_descriptor"]),
            ):
                _require(
                    success.get(field) == expected,
                    f"2(a) Target success {field} disagrees with its raw fork",
                )
            diagnostics = diagnostic_records[attempt_id]
            for field, expected in (
                ("prompt_id", row.get("prompt_id")),
                ("prompt_sha256", row.get("condition_hash")),
                ("chunk_index", row.get("chunk_index")),
                ("history_head_sha256", row.get("prefix_hash")),
                ("history_latent_paths", evidence["prefix_latent_paths"]),
                ("noise_descriptor", evidence["noise_descriptor_payload"]),
                ("noise_descriptor_sha256", evidence["noise_descriptor"]),
                ("initial_noise_path", evidence["noise_path"]),
                ("latent_path", row.get("target_latent_path")),
                ("latent_sha256", row.get("target_latent_sha256")),
                ("segment_sha256", row.get("target_segment_hash")),
                ("kv_cache_sha256", row.get("target_kv_hash")),
            ):
                _require(
                    diagnostics.get(field) == expected,
                    f"2(a) Target diagnostics {field} disagrees with its raw fork",
                )

        if preflight_result is not None:
            _require(
                set(previous_baseline_by_pair) == expected_pairs
                and all(
                    history.next_chunk_index == campaign_chunk_count
                    for history in previous_baseline_by_pair.values()
                ),
                "2(a) baseline trajectories do not cover the full candidate horizon",
            )

        orphan_candidate_scopes = {
            str(state["scope_id"])
            for state in attempts.values()
            if state["phase"] == "single_point_candidate"
            and state["completed"] is not None
            and state["scope_id"] not in raw_scope_ids
        }
        _require(
            not orphan_candidate_scopes,
            "Target lifecycle contains an orphan 2(a) result-row scope",
        )
        completed_non_row = {
            attempt_id: state
            for attempt_id, state in attempts.items()
            if state["completed"] is not None
            and state["phase"] in {"calibration_baseline", "evaluation_baseline"}
        }
        unexpected_completed = [
            state
            for state in attempts.values()
            if state["completed"] is not None
            and state["phase"]
            not in {
                "single_point_candidate",
                "calibration_baseline",
                "evaluation_baseline",
            }
        ]
        _require(
            not unexpected_completed,
            "2(a) contains an extra completed Target call outside candidate/baseline phases",
        )

        seeds = campaign.get("seeds")
        calibration_prompts = campaign.get("calibration_prompts")
        evaluation_prompts = campaign.get("evaluation_prompts")
        _require(
            isinstance(seeds, list)
            and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
            and isinstance(calibration_prompts, list)
            and isinstance(evaluation_prompts, list),
            "campaign manifest cannot define expected 2(a) baseline scopes",
        )
        prompt_entries: dict[str, dict[str, Any]] = {}
        for expected_split, values in (
            ("calibration", calibration_prompts),
            ("evaluation", evaluation_prompts),
        ):
            for value in values:
                _require(
                    isinstance(value, dict)
                    and value.get("split") == expected_split
                    and isinstance(value.get("prompt_id"), str)
                    and isinstance(value.get("prompt"), str),
                    "campaign prompt entry is invalid",
                )
                prompt_entries[str(value["prompt_id"])] = value

        expected_baselines: dict[str, tuple[str, dict[str, Any], int]] = {}

        def add_expected_baseline(
            *, baseline_phase: str, prompt_entry: dict[str, Any], seed: int
        ) -> None:
            scope_id = hashlib.sha256(
                _canonical_json_bytes(
                    [
                        protocol_id,
                        "target-non-row-v1",
                        baseline_phase,
                        prompt_entry["prompt_id"],
                        seed,
                        0,
                    ]
                )
            ).hexdigest()
            _require(scope_id not in expected_baselines, "expected baseline scope is duplicated")
            expected_baselines[scope_id] = (baseline_phase, prompt_entry, seed)

        for prompt_entry in calibration_prompts:
            assert isinstance(prompt_entry, dict)
            for seed in seeds:
                add_expected_baseline(
                    baseline_phase="calibration_baseline",
                    prompt_entry=prompt_entry,
                    seed=seed,
                )
        evaluation_pairs = {
            (str(row.get("prompt_id")), int(row.get("seed"))) for row in rows
        }
        for prompt_id, seed in evaluation_pairs:
            prompt_entry = prompt_entries.get(prompt_id)
            _require(
                prompt_entry is not None and prompt_entry.get("split") == "evaluation",
                "2(a) raw row is not in the campaign evaluation prompt set",
            )
            expected_condition_hash = hashlib.sha256(
                str(prompt_entry["prompt"]).encode("utf-8")
            ).hexdigest()
            _require(
                all(
                    row.get("condition_hash") == expected_condition_hash
                    for row in rows
                    if row.get("prompt_id") == prompt_id and row.get("seed") == seed
                ),
                "2(a) raw condition hash differs from the campaign prompt",
            )
            add_expected_baseline(
                baseline_phase="evaluation_baseline",
                prompt_entry=prompt_entry,
                seed=seed,
            )

        actual_baseline_scopes = {
            str(state["scope_id"]) for state in completed_non_row.values()
        }
        _require(
            actual_baseline_scopes == set(expected_baselines),
            "2(a) baseline Target scopes are missing or orphaned",
        )
        reconciled_non_row_scopes = len(expected_baselines)
        for scope_id, (baseline_phase, prompt_entry, seed) in expected_baselines.items():
            completed_for_scope = [
                (attempt_id, state)
                for attempt_id, state in attempts_by_scope.get(scope_id, [])
                if attempt_id in completed_non_row
            ]
            _require(
                len(completed_for_scope) == 1,
                "2(a) baseline scope must bind exactly one completed Target call",
            )
            attempt_id, state = completed_for_scope[0]
            completed = state["completed"]
            success = successes.get(attempt_id)
            diagnostics = diagnostic_records.get(attempt_id)
            assert isinstance(completed, dict) and isinstance(success, dict)
            assert isinstance(diagnostics, dict)
            prompt_sha256 = hashlib.sha256(
                str(prompt_entry["prompt"]).encode("utf-8")
            ).hexdigest()
            _require(
                completed.get("phase") == baseline_phase
                and completed.get("scope_id") == scope_id
                and completed.get("chunk_index") == 0
                and completed.get("history_chunk_count") == 0
                and completed.get("history_latent_paths") == []
                and completed.get("prompt_sha256") == prompt_sha256,
                "2(a) baseline Target lifecycle binding is invalid",
            )
            _require(
                completed.get("model_id") == target_model["id"]
                and completed.get("model_artifact_sha256")
                == target_model["artifact_sha256"],
                "2(a) baseline Target model differs from the campaign binding",
            )
            _require(
                success.get("scope_id") == scope_id
                and success.get("phase") == baseline_phase
                and success.get("chunk_index") == 0
                and diagnostics.get("prompt_id") == prompt_entry["prompt_id"]
                and diagnostics.get("prompt_sha256") == prompt_sha256
                and diagnostics.get("noise_descriptor", {}).get("trajectory_seed")
                == seed
                and diagnostics.get("noise_descriptor", {}).get("chunk_index") == 0,
                "2(a) baseline Target success binding is invalid",
            )
            expected_logical_call_id = hashlib.sha256(
                _canonical_json_bytes(
                    [
                        "target-logical-call-v2",
                        protocol_id,
                        completed["model_id"],
                        completed["model_artifact_sha256"],
                        prompt_entry["prompt_id"],
                        prompt_sha256,
                        success["history_head_sha256"],
                        0,
                        success["noise_descriptor_sha256"],
                    ]
                )
            ).hexdigest()
            _require(
                state["logical_call_id"] == expected_logical_call_id,
                "2(a) baseline Target logical call ID drifted",
            )
            baseline_split = (
                "calibration"
                if baseline_phase == "calibration_baseline"
                else "evaluation"
            )
            baseline_category = (
                "calibration-baselines"
                if baseline_split == "calibration"
                else "evaluation-baselines"
            )
            baseline_key = (
                f"{prompt_entry['prompt_id']}-{seed}-{prompt_sha256[:12]}"
            )
            baseline_path = _safe_run_file(
                run_dir,
                f"state/{baseline_category}/{baseline_key}/chunk-00.json",
                label="2(a) baseline chunk-0 checkpoint",
            )
            baseline_raw, baseline_history, baseline_context, baseline_segment, baseline_metadata = (
                _load_audited_checkpoint(
                    baseline_path, label="2(a) baseline chunk-0 checkpoint"
                )
            )
            expected_trajectory_id = "i2-" + hashlib.sha256(
                _canonical_json_bytes(
                    [
                        protocol_id,
                        "baseline",
                        baseline_split,
                        prompt_entry["prompt_id"],
                        seed,
                    ]
                )
            ).hexdigest()[:40]
            _require(
                baseline_history.next_chunk_index == 1
                and baseline_history.trajectory_id == expected_trajectory_id
                and baseline_history.prompt_id == prompt_entry["prompt_id"]
                and baseline_history.trajectory_seed == seed
                and baseline_metadata == {"kind": "baseline", "chunk_index": 0},
                "2(a) baseline chunk-0 checkpoint identity drifted",
            )
            checkpoint_attempt_id, checkpoint_logical_call_id = (
                _audit_target_checkpoint_binding(
                    generated=baseline_raw["generated"],
                    context=baseline_context,
                    segment=baseline_segment,
                    prefix=baseline_history.prefix_before(0),
                    prompt_sha256=prompt_sha256,
                    protocol_id=protocol_id,
                    target_model_binding=target_model_binding,
                    diagnostic_records=diagnostic_records,
                )
            )
            _require(
                checkpoint_attempt_id == attempt_id
                and checkpoint_logical_call_id == state["logical_call_id"],
                "2(a) baseline checkpoint differs from its lifecycle scope",
            )
    else:
        raw_scope_ids = set()

        protocol_id = campaign_manifest.get("protocol_id")
        seeds = campaign_manifest.get("seeds")
        calibration_prompts = campaign_manifest.get("calibration_prompts")
        evaluation_prompts = campaign_manifest.get("evaluation_prompts")
        smoke_prompts = campaign_manifest.get("smoke_prompts")
        draft_model = models.get("draft") if isinstance(models, dict) else None
        _require(
            isinstance(protocol_id, str)
            and bool(protocol_id)
            and isinstance(seeds, list)
            and all(
                isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
            )
            and isinstance(calibration_prompts, list)
            and isinstance(evaluation_prompts, list)
            and isinstance(smoke_prompts, list)
            and isinstance(draft_model, dict)
            and isinstance(draft_model.get("id"), str)
            and bool(draft_model["id"])
            and isinstance(draft_model.get("artifact_sha256"), str)
            and _SHA256_RE.fullmatch(draft_model["artifact_sha256"]) is not None,
            "campaign manifest cannot define expected 2(b) identities",
        )
        model_bindings = {
            "draft": (
                str(draft_model["id"]),
                str(draft_model["artifact_sha256"]),
            ),
            "target": target_model_binding,
        }
        expected_split = "smoke" if phase == "2b-smoke" else "evaluation"
        selected_prompt_entries = smoke_prompts if expected_split == "smoke" else evaluation_prompts
        prompt_entries: dict[str, dict[str, Any]] = {}
        for entry in selected_prompt_entries:
            _require(
                isinstance(entry, dict)
                and isinstance(entry.get("prompt_id"), str)
                and isinstance(entry.get("prompt"), str)
                and entry.get("split") == "evaluation",
                "campaign manifest selected 2(b) prompt entry is invalid",
            )
            prompt_id = str(entry["prompt_id"])
            _require(prompt_id not in prompt_entries, "selected 2(b) prompt ID is duplicated")
            prompt_entries[prompt_id] = entry

        expected_calibration_scopes: dict[str, tuple[dict[str, Any], int]] = {}
        for prompt_entry in calibration_prompts:
            _require(
                isinstance(prompt_entry, dict)
                and prompt_entry.get("split") == "calibration"
                and isinstance(prompt_entry.get("prompt_id"), str)
                and isinstance(prompt_entry.get("prompt"), str),
                "2(b) calibration prompt entry is invalid",
            )
            for seed in seeds:
                scope_id = hashlib.sha256(
                    _canonical_json_bytes(
                        [
                            protocol_id,
                            "target-non-row-v1",
                            "calibration_baseline",
                            prompt_entry["prompt_id"],
                            seed,
                            0,
                        ]
                    )
                ).hexdigest()
                _require(
                    scope_id not in expected_calibration_scopes,
                    "expected 2(b) calibration scope is duplicated",
                )
                expected_calibration_scopes[scope_id] = (prompt_entry, seed)

        calibration_attempts = {
            attempt_id: state
            for attempt_id, state in attempts.items()
            if state["phase"] == "calibration_baseline"
        }
        actual_calibration_scopes = {
            str(state["scope_id"]) for state in calibration_attempts.values()
        }
        _require(
            actual_calibration_scopes == set(expected_calibration_scopes),
            "2(b) calibration Target scopes are missing or orphaned",
        )
        reconciled_non_row_scopes = len(expected_calibration_scopes)
        for scope_id, (prompt_entry, calibration_seed) in expected_calibration_scopes.items():
            selected_calibration = [
                (attempt_id, state)
                for attempt_id, state in attempts_by_scope.get(scope_id, [])
                if attempt_id in calibration_attempts
            ]
            _require(
                len(selected_calibration) == 1,
                "2(b) calibration scope must bind exactly one Target call",
            )
            attempt_id, state = selected_calibration[0]
            completed = state["completed"]
            success = successes.get(attempt_id)
            diagnostics = diagnostic_records.get(attempt_id)
            assert isinstance(completed, dict)
            assert isinstance(success, dict)
            assert isinstance(diagnostics, dict)
            expected_prompt_sha256 = hashlib.sha256(
                str(prompt_entry["prompt"]).encode("utf-8")
            ).hexdigest()
            _require(
                completed.get("scope_id") == scope_id
                and completed.get("chunk_index") == 0
                and completed.get("history_chunk_count") == 0
                and completed.get("history_latent_paths") == []
                and completed.get("prompt_sha256") == expected_prompt_sha256
                and completed.get("model_id") == target_model_binding[0]
                and completed.get("model_artifact_sha256")
                == target_model_binding[1],
                "2(b) calibration Target lifecycle binding is invalid",
            )
            _require(
                success.get("scope_id") == scope_id
                and success.get("phase") == "calibration_baseline"
                and success.get("chunk_index") == 0
                and diagnostics.get("prompt_id") == prompt_entry["prompt_id"]
                and diagnostics.get("prompt_sha256") == expected_prompt_sha256
                and diagnostics.get("noise_descriptor", {}).get("trajectory_seed")
                == calibration_seed,
                "2(b) calibration Target evidence binding is invalid",
            )

            calibration_key = (
                f"{prompt_entry['prompt_id']}-{calibration_seed}-"
                f"{expected_prompt_sha256[:12]}"
            )
            calibration_path = _safe_run_file(
                run_dir,
                f"state/calibration-baselines/{calibration_key}/chunk-00.json",
                label="2(b) calibration baseline checkpoint",
            )
            calibration_raw, calibration_history, calibration_context, calibration_segment, calibration_metadata = (
                _load_audited_checkpoint(
                    calibration_path,
                    label="2(b) calibration baseline checkpoint",
                )
            )
            expected_trajectory_id = "i2-" + hashlib.sha256(
                _canonical_json_bytes(
                    [
                        protocol_id,
                        "baseline",
                        "calibration",
                        prompt_entry["prompt_id"],
                        calibration_seed,
                    ]
                )
            ).hexdigest()[:40]
            _require(
                calibration_history.next_chunk_index == 1
                and calibration_history.trajectory_id == expected_trajectory_id
                and calibration_history.prompt_id == prompt_entry["prompt_id"]
                and calibration_history.trajectory_seed == calibration_seed
                and calibration_metadata == {"kind": "baseline", "chunk_index": 0},
                "2(b) calibration checkpoint identity is invalid",
            )
            calibration_prefix = calibration_history.prefix_before(0)
            checkpoint_attempt_id, checkpoint_logical_call_id = (
                _audit_target_checkpoint_binding(
                    generated=calibration_raw["generated"],
                    context=calibration_context,
                    segment=calibration_segment,
                    prefix=calibration_prefix,
                    prompt_sha256=expected_prompt_sha256,
                    protocol_id=protocol_id,
                    target_model_binding=target_model_binding,
                    diagnostic_records=diagnostic_records,
                )
            )
            _require(
                checkpoint_attempt_id == attempt_id
                and checkpoint_logical_call_id == state["logical_call_id"],
                "2(b) calibration checkpoint differs from its lifecycle scope",
            )

        verified_hidden_files: dict[str, tuple[str, int]] = {}
        online_calibrator = (
            _audit_rank_calibration(
                run_dir=run_dir,
                protocol_id=protocol_id,
                calibration_prompts=calibration_prompts,
                seeds=seeds,
                chunk_count=campaign_chunk_count,
                model_bindings=model_bindings,
                verified_segment_files=verified_target_latent_files,
                verified_hidden_files=verified_hidden_files,
            )
            if any(
                row.get("strategy")
                in {
                    "online_rank_guided",
                    "online_coverage_guided",
                    "online_lookahead_coverage_guided",
                    "online_lookahead_coverage_boundary_gated",
                }
                for row in rows
            )
            else None
        )
        route_cache: dict[str, dict[str, Any]] = {}
        canonical_root_cache: dict[tuple[str, int], str] = {}
        for row in rows:
            _require(
                set(row) == set(_CSV_2B_FIELDS)
                and row.get("schema") == RAW_2B_SCHEMA,
                "2(b) raw row fields or schema drifted",
            )
            scope_id = row.get("event_id")
            _require(
                isinstance(scope_id, str) and _SHA256_RE.fullmatch(scope_id) is not None,
                "2(b) raw row event_id is invalid",
            )
            _require(scope_id not in raw_scope_ids, "2(b) raw row event_id is duplicated")
            raw_scope_ids.add(scope_id)
            prompt_id = row.get("prompt_id")
            seed = row.get("seed")
            budget = row.get("budget")
            strategy = row.get("strategy")
            route_repeat = row.get("route_repeat")
            selected_chunks_value = row.get("selected_chunks")
            proposed_chunks_value = row.get("proposed_target_chunks")
            _require(
                isinstance(prompt_id, str)
                and prompt_id in prompt_entries
                and isinstance(seed, int)
                and not isinstance(seed, bool)
                and seed in seeds
                and isinstance(budget, int)
                and not isinstance(budget, bool)
                and budget in audit_budgets
                and strategy in audit_strategies
                and isinstance(selected_chunks_value, list)
                and isinstance(proposed_chunks_value, list)
                and all(
                    isinstance(chunk, int)
                    and not isinstance(chunk, bool)
                    and 1 <= chunk < campaign_chunk_count
                    for chunk in selected_chunks_value
                )
                and all(
                    isinstance(chunk, int)
                    and not isinstance(chunk, bool)
                    and 1 <= chunk < campaign_chunk_count
                    for chunk in proposed_chunks_value
                ),
                "2(b) raw row scientific identity is invalid",
            )
            selected_chunks = tuple(selected_chunks_value)
            proposed_chunks = tuple(proposed_chunks_value)
            is_boundary_gated = (
                strategy == "online_lookahead_coverage_boundary_gated"
            )
            _require(
                selected_chunks == tuple(sorted(set(selected_chunks)))
                and proposed_chunks == tuple(sorted(set(proposed_chunks)))
                and len(proposed_chunks) == budget
                and (
                    set(selected_chunks).issubset(proposed_chunks)
                    if is_boundary_gated
                    else selected_chunks == proposed_chunks
                    and len(selected_chunks) == budget
                ),
                "2(b) raw proposed/selected chunks violate their frozen budget",
            )
            _require(
                (strategy == "random" and isinstance(route_repeat, int)
                 and not isinstance(route_repeat, bool)
                 and 0 <= route_repeat < audit_random_repeats)
                or (strategy != "random" and route_repeat is None),
                "2(b) raw route_repeat is invalid for its strategy",
            )
            expected_event_id = hashlib.sha256(
                _canonical_json_bytes(
                    [
                        protocol_id,
                        "2b",
                        expected_split,
                        prompt_id,
                        seed,
                        budget,
                        strategy,
                        route_repeat,
                    ]
                )
            ).hexdigest()
            _require(
                row.get("split") == expected_split and scope_id == expected_event_id,
                "2(b) raw event ID or split is not its frozen deterministic identity",
            )
            expected_route_id = (
                f"online-lookahead-coverage-boundary-gated-erank-drop-k{budget}"
                if is_boundary_gated
                else f"online-lookahead-coverage-erank-drop-k{budget}"
                if strategy == "online_lookahead_coverage_guided"
                else f"online-coverage-erank-drop-k{budget}"
                if strategy == "online_coverage_guided"
                else (
                    f"online-erank-drop-k{budget}"
                    if quality_measurement_schema
                    == VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA
                    else f"online-rank-guided-k{budget}"
                )
                if strategy == "online_rank_guided"
                else "fixed-" + "-".join(str(chunk) for chunk in selected_chunks)
            )
            _require(
                row.get("rollout_id") == expected_route_id
                and row.get("committed_target_chunks")
                == 1 + len(selected_chunks)
                and row.get("budgeted_target_calls")
                == 1 + len(proposed_chunks),
                "2(b) raw rollout ID or committed Target budget drifted",
            )
            prompt_sha256 = hashlib.sha256(
                str(prompt_entries[prompt_id]["prompt"]).encode("utf-8")
            ).hexdigest()
            route_summary = _audit_2b_rollout_route(
                run_dir=run_dir,
                protocol_id=protocol_id,
                prompt_id=prompt_id,
                prompt_sha256=prompt_sha256,
                seed=seed,
                route_id=expected_route_id,
                corrected_chunks=selected_chunks,
                proposed_target_chunks=proposed_chunks,
                chunk_count=campaign_chunk_count,
                model_bindings=model_bindings,
                quality_model_binding=quality_model_binding,
                quality_measurement_schema=quality_measurement_schema,
                primary_quality_metric_id=primary_quality_metric_id,
                online_calibrator=online_calibrator,
                diagnostic_records=diagnostic_records,
                route_cache=route_cache,
                canonical_root_cache=canonical_root_cache,
                verified_noise_files=verified_noise_files,
                verified_segment_files=verified_target_latent_files,
                verified_hidden_files=verified_hidden_files,
                verified_decode_files=verified_decode_files,
                coverage_constraint=audit_coverage_constraint,
            )
            if is_boundary_gated:
                _audit_boundary_proposal_lifecycle(
                    route_summary=route_summary,
                    attempts=attempts,
                    scope_id=scope_id,
                    proposal_count=len(proposed_chunks),
                )
            for row_field, summary_field in (
                ("full_quality", "full_video_quality"),
                ("temporal_consistency", "temporal_consistency"),
                ("final_history_hash", "final_history_hash"),
            ):
                _require(
                    row.get(row_field) == route_summary[summary_field],
                    f"2(b) raw {row_field} differs from its audited full rollout",
                )
            scoped = attempts_by_scope.get(scope_id, [])
            selected = [state for _attempt_id, state in scoped]

            def count(status: str, *, selected_phase: str | None = None) -> int:
                return sum(
                    1
                    for state in selected
                    if (selected_phase is None or state["phase"] == selected_phase)
                    and status in state["statuses"]
                )

            expected_total = {
                "target_rpc_attempted": count("started"),
                "target_rpc_backend_returned": count("backend_returned"),
                "target_rpc_completed": count("completed"),
                "physical_target_calls": count("completed"),
                "route_materialized_logical_target_calls": len(
                    {
                        state["logical_call_id"]
                        for state in selected
                        if state["phase"] == "route_materialization"
                        and "completed" in state["statuses"]
                    }
                ),
            }
            expected_search = {
                "selection_search_target_rpc_attempted": count(
                    "started", selected_phase="selection_search"
                ),
                "selection_search_target_rpc_backend_returned": count(
                    "backend_returned", selected_phase="selection_search"
                ),
                "selection_search_target_rpc_completed": count(
                    "completed", selected_phase="selection_search"
                ),
                "selection_search_target_calls": count(
                    "completed", selected_phase="selection_search"
                ),
            }
            for field, expected in {**expected_total, **expected_search}.items():
                _require(
                    row.get(field) == expected,
                    f"2(b) raw {field} disagrees with Target lifecycle scope",
                )
            _require(
                expected_total["route_materialized_logical_target_calls"]
                <= 1 + len(proposed_chunks),
                "2(b) route materialized more logical Target calls than its "
                "proposal budget",
            )
            search_attempts = expected_search["selection_search_target_rpc_attempted"]
            _require(
                (strategy == "greedy_oracle") or search_attempts == 0,
                "non-Greedy strategy contains selection-search Target attempts",
            )
            route_materialization_ids = {
                str(state["logical_call_id"])
                for state in selected
                if state["phase"] == "route_materialization"
            }
            _require(
                route_materialization_ids
                <= route_summary["target_logical_call_ids"],
                "2(b) route-materialization scope contains a call outside its audited route",
            )

            expected_search_ids: set[str] = set()
            expected_search_rollouts = 0
            if strategy == "greedy_oracle":
                greedy_selected: tuple[int, ...] = ()
                for _stage in range(budget):
                    stage_results: list[tuple[tuple[int, ...], dict[str, Any]]] = []
                    for candidate in range(1, campaign_chunk_count):
                        if candidate in greedy_selected:
                            continue
                        candidate_chunks = tuple(sorted((*greedy_selected, candidate)))
                        candidate_route_id = "fixed-" + "-".join(
                            str(chunk) for chunk in candidate_chunks
                        )
                        candidate_summary = _audit_2b_rollout_route(
                            run_dir=run_dir,
                            protocol_id=protocol_id,
                            prompt_id=prompt_id,
                            prompt_sha256=prompt_sha256,
                            seed=seed,
                            route_id=candidate_route_id,
                            corrected_chunks=candidate_chunks,
                            chunk_count=campaign_chunk_count,
                            model_bindings=model_bindings,
                            quality_model_binding=quality_model_binding,
                            quality_measurement_schema=quality_measurement_schema,
                            primary_quality_metric_id=primary_quality_metric_id,
                            online_calibrator=online_calibrator,
                            diagnostic_records=diagnostic_records,
                            route_cache=route_cache,
                            canonical_root_cache=canonical_root_cache,
                            verified_noise_files=verified_noise_files,
                            verified_segment_files=verified_target_latent_files,
                            verified_hidden_files=verified_hidden_files,
                            verified_decode_files=verified_decode_files,
                        )
                        stage_results.append((candidate_chunks, candidate_summary))
                        expected_search_ids.update(
                            candidate_summary["target_logical_call_ids"]
                        )
                        expected_search_rollouts += 1
                    _require(bool(stage_results), "Greedy Oracle audit stage is empty")
                    greedy_selected = min(
                        stage_results,
                        key=lambda item: (
                            -float(item[1]["full_video_quality"]),
                            item[0],
                        ),
                    )[0]
                _require(
                    greedy_selected == selected_chunks
                    and row.get("selection_search_rollout_count")
                    == expected_search_rollouts,
                    "2(b) Greedy Oracle selection or search count is not reproducible",
                )
            else:
                _require(
                    row.get("selection_search_rollout_count") == 0,
                    "non-Greedy strategy reports selection-search rollouts",
                )
            selection_search_ids = {
                str(state["logical_call_id"])
                for state in selected
                if state["phase"] == "selection_search"
            }
            _require(
                selection_search_ids <= expected_search_ids,
                "2(b) selection-search scope contains a call outside audited Greedy routes",
            )
            for attempt_id, state in scoped:
                diagnostics = diagnostic_records.get(attempt_id)
                _require(
                    isinstance(diagnostics, dict)
                    and diagnostics.get("prompt_id") == prompt_id
                    and diagnostics.get("prompt_sha256") == prompt_sha256
                    and diagnostics.get("noise_descriptor", {}).get("trajectory_seed")
                    == seed,
                    "2(b) scoped Target call belongs to another prompt or seed",
                )
        lifecycle_scope_ids = {
            str(state["scope_id"])
            for state in scoped_attempts.values()
            if state["phase"] in {"route_materialization", "selection_search"}
        }
        _require(
            lifecycle_scope_ids <= raw_scope_ids,
            "Target lifecycle contains an orphan result-row scope",
        )

    report = {
        "schema": TARGET_RPC_AUDIT_SCHEMA,
        "status": "passed",
        "phase": phase,
        "trusted_preflight_bound": preflight_result is not None,
        "attempts": len(attempts),
        "attempted": sum("started" in state["statuses"] for state in attempts.values()),
        "backend_returned": sum(
            "backend_returned" in state["statuses"] for state in attempts.values()
        ),
        "completed": sum("completed" in state["statuses"] for state in attempts.values()),
        "unique_completed_logical_calls": len(
            {
                state["logical_call_id"]
                for state in attempts.values()
                if "completed" in state["statuses"]
            }
        ),
        "successful_events": len(successes),
        "diagnostics_files": len(referenced_diagnostic_paths),
        "diagnostics_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(
                {
                    attempt_id: successes[attempt_id]["diagnostics_sha256"]
                    for attempt_id in sorted(diagnostic_records)
                }
            )
        ).hexdigest(),
        "scoped_attempts": len(scoped_attempts),
        "reconciled_raw_scopes": len(raw_scope_ids),
        "reconciled_non_row_scopes": reconciled_non_row_scopes,
        "completed_by_phase": {
            lifecycle_phase: sum(
                state["phase"] == lifecycle_phase and state["completed"] is not None
                for state in attempts.values()
            )
            for lifecycle_phase in sorted(TARGET_RPC_PHASES)
        },
        "lifecycle_sha256": _sha256_file(run_dir / "logs" / "target-rpc-lifecycle.jsonl"),
        "success_journal_sha256": (
            _sha256_file(success_path) if success_path.exists() else None
        ),
    }
    _write_json(run_dir / "logs" / "target-rpc-accounting.json", report)
    return report


def audit_kv_consistency(*, run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Revalidate every persisted model-specific context without retaining it."""

    checked = 0
    checked_pending_plans = 0
    by_role = {"draft": 0, "target": 0}
    chain = hashlib.sha256()
    checkpoint_root = run_dir / "state"
    canonical_rollout_roots: dict[tuple[str, int], str] = {}
    route_rollout_roots: list[tuple[tuple[str, int], str, Path]] = []
    pending_plan_event_ids: set[str] | None = None
    for path in sorted(checkpoint_root.rglob("*.json")) if checkpoint_root.exists() else ():
        relative = path.relative_to(checkpoint_root).as_posix()
        pending_split = _KV_PENDING_PLAN_SPLITS.get(relative)
        if pending_split is not None:
            value = _load_json_object(path, label="pending campaign plan")
            try:
                plan = PendingMultiBudgetPlan.from_dict(
                    value, manifest_path=f"state/{relative}"
                )
            except CampaignError as error:
                raise Insight2RunError(
                    f"pending campaign plan {path} is invalid: {error}"
                ) from error
            _require(
                plan.split == pending_split and plan.to_dict() == value,
                f"pending campaign plan {path} is not canonical",
            )
            _require(
                pending_plan_event_ids is None,
                "multiple pending campaign plans are not valid in one run",
            )
            pending_plan_event_ids = {entry.event_id for entry in plan.entries}
            checked_pending_plans += 1
            continue
        relative_parts = path.relative_to(checkpoint_root).parts
        filename_match = (
            _KV_CHECKPOINT_FILENAME_RE.fullmatch(relative_parts[2])
            if len(relative_parts) == 3
            else None
        )
        _require(
            len(relative_parts) == 3
            and relative_parts[0] in _KV_CHECKPOINT_CATEGORIES
            and bool(relative_parts[1])
            and filename_match is not None
            and relative_parts[2]
            == f"chunk-{int(filename_match.group(1)):02d}.json"
            and (
                relative_parts[0] != "rollout-roots"
                or relative_parts[2] == "chunk-00.json"
            ),
            f"unsupported state JSON for KV audit: {path}",
        )
        value, history, context_object, segment_object, _metadata = (
            _load_audited_checkpoint(path, label="campaign checkpoint")
        )
        _require(
            history.next_chunk_index == int(filename_match.group(1)) + 1,
            f"checkpoint {path} index differs from its history",
        )
        generated = value["generated"]
        context = generated["context"]
        segment = generated["segment"]
        role = context_object.role
        _require(role in by_role, f"checkpoint {path} has invalid model role")
        noise = context_object.noise
        kv = context_object.kv
        kv.verify_for(
            role=str(role),
            model_id=context_object.model_id,
            model_artifact_sha256=context_object.model_artifact_sha256,
            history_head_sha256=context_object.history_head_sha256,
        )
        _require(
            segment.get("parent_history_sha256") == context.get("history_head_sha256"),
            f"checkpoint {path} context does not bind the segment parent",
        )
        _require(
            segment.get("kv_descriptor_sha256") == kv.descriptor_sha256,
            f"checkpoint {path} segment does not bind its model-local KV descriptor",
        )
        _require(
            segment.get("initial_noise_descriptor_sha256") == noise.descriptor_sha256,
            f"checkpoint {path} segment does not bind its initial noise",
        )
        _require(
            segment.get("chunk_index") == context.get("next_chunk_index") == noise.chunk_index,
            f"checkpoint {path} chunk indices disagree",
        )
        chain.update(
            _canonical_json_bytes(
                {
                    "role": role,
                    "history": context["history_head_sha256"],
                    "kv": kv.cache_sha256,
                    "noise": noise.tensor_sha256,
                    "chunk": noise.chunk_index,
                }
            )
        )
        checked += 1
        by_role[str(role)] += 1

        if len(relative_parts) == 3 and relative_parts[2] == "chunk-00.json":
            identity = (history.prompt_id, history.trajectory_seed)
            root_binding_sha256 = hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "history": history.prefix_before(1).to_dict(),
                        "generated": generated,
                    }
                )
            ).hexdigest()
            if relative_parts[0] == "rollout-roots":
                _require(
                    history.next_chunk_index == 1,
                    f"canonical rollout root {path} must contain exactly chunk 0",
                )
                _require(
                    identity not in canonical_rollout_roots,
                    f"duplicate canonical rollout root for {identity}",
                )
                canonical_rollout_roots[identity] = root_binding_sha256
            elif relative_parts[0] == "rollouts":
                route_rollout_roots.append((identity, root_binding_sha256, path))

    for identity, route_binding_sha256, path in route_rollout_roots:
        _require(
            identity in canonical_rollout_roots,
            f"route checkpoint {path} has no canonical Target root",
        )
        _require(
            route_binding_sha256 == canonical_rollout_roots[identity],
            f"route checkpoint {path} does not share canonical Target chunk 0",
        )

    paired_rows = 0
    hidden_files = 0
    single_point_contexts = 0
    verified_latents: dict[str, tuple[str, int]] = {}
    raw_2b_event_ids: set[str] = set()
    for row in rows:
        if row.get("schema") != RAW_2B_SCHEMA:
            continue
        event_id = row.get("event_id")
        _require(
            isinstance(event_id, str)
            and _SHA256_RE.fullmatch(event_id) is not None
            and event_id not in raw_2b_event_ids,
            "2(b) rows contain an invalid or duplicate event_id",
        )
        raw_2b_event_ids.add(event_id)
    if pending_plan_event_ids is not None:
        _require(
            raw_2b_event_ids == pending_plan_event_ids,
            "pending campaign plan does not exactly match finalized 2(b) rows",
        )
    for row in rows:
        if row.get("schema") != RAW_2A_SCHEMA:
            continue
        for field in ("prefix_hash", "draft_kv_hash", "target_kv_hash", "noise_hash"):
            value = row.get(field)
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"2(a) row has invalid {field}",
            )
        evidence_binding, _evidence_sha256 = _audit_single_point_evidence(
            run_dir=run_dir,
            row=row,
            verified_latents=verified_latents,
        )
        by_role["draft"] += 1
        by_role["target"] += 1
        single_point_contexts += 2
        hidden_sha256 = row.get("hidden_state_sha256")
        _require(
            isinstance(hidden_sha256, str)
            and _SHA256_RE.fullmatch(hidden_sha256) is not None,
            "2(a) row has invalid hidden_state_sha256",
        )
        hidden_path = _safe_run_file(
            run_dir,
            row.get("hidden_state_path"),
            label="2(a) hidden_state_path",
        )
        _require(
            _sha256_file(hidden_path) == hidden_sha256,
            "2(a) hidden-state evidence SHA-256 mismatch",
        )
        chain.update(
            _canonical_json_bytes(
                {
                    **evidence_binding,
                    "hidden_state": hidden_sha256,
                }
            )
        )
        paired_rows += 1
        hidden_files += 1
    _require(
        checked + single_point_contexts > 0,
        "no persisted generation contexts were available for KV audit",
    )
    report = {
        "schema": KV_AUDIT_SCHEMA,
        "status": "passed",
        "construction": "fresh_model_specific_rebuild_from_clean_latent_history",
        "cross_model_cache_reuse": False,
        "checked_checkpoint_contexts": checked,
        "checked_pending_plan_manifests": checked_pending_plans,
        "checked_contexts_by_role": by_role,
        "checked_single_point_pairs": paired_rows,
        "checked_single_point_contexts": single_point_contexts,
        "checked_unique_latent_files": len(verified_latents),
        "checked_hidden_state_files": hidden_files,
        "checked_canonical_rollout_roots": len(canonical_rollout_roots),
        "checked_route_rollout_roots": len(route_rollout_roots),
        "evidence_chain_sha256": chain.hexdigest(),
    }
    _write_json(run_dir / "logs" / "kv-consistency.json", report)
    return report


def _runtime_model_state(runtime: Any) -> Mapping[str, Any]:
    value = getattr(runtime, "model_state", None)
    if callable(value):
        value = value()
    if value is None:
        draft = getattr(runtime, "draft")
        target = getattr(runtime, "target")
        value = {
            "draft": getattr(draft, "model_state", {}),
            "target": getattr(target, "model_state", {}),
        }
    _require(isinstance(value, Mapping), "runtime model_state must be an object")
    return value


def verify_runtime_model_state(
    preflight_result: Preflight, value: Mapping[str, Any]
) -> None:
    """Bind the two loaded workers back to the frozen protocol and artifacts."""

    _require(set(value) == {"draft", "target"}, "runtime must report draft and target")
    draft = value["draft"]
    target = value["target"]
    _require(isinstance(draft, Mapping), "Draft runtime model state is invalid")
    _require(isinstance(target, Mapping), "Target runtime model state is invalid")
    models = preflight_result.protocol.raw["models"]
    _require(
        draft.get("model_id") == models["draft"]["id"],
        "loaded Draft model ID disagrees with the frozen protocol",
    )
    _require(
        target.get("model_id") == models["target"]["id"],
        "loaded Target model ID disagrees with the frozen protocol",
    )
    draft_artifacts = {
        str(item["name"]): item for item in preflight_result.draft_verified_artifacts
    }
    target_artifacts = {
        str(item["name"]): item for item in preflight_result.target_verified_artifacts
    }
    _require(
        draft.get("checkpoint_sha256")
        == draft_artifacts["self_forcing_checkpoint"]["sha256"],
        "loaded Draft checkpoint disagrees with artifact preflight",
    )
    _require(
        draft.get("artifact_spec_sha256") == preflight_result.draft_spec_sha256
        and draft.get("vae_sha256")
        == draft_artifacts["wan_vae_checkpoint"]["sha256"]
        and draft.get("musiq_spaq_sha256")
        == draft_artifacts["musiq_spaq_checkpoint"]["sha256"],
        "loaded Draft artifact specification, VAE, or quality model disagrees with preflight",
    )
    _require(
        target.get("checkpoint_sha256")
        == target_artifacts["krea_14b_checkpoint"]["sha256"],
        "loaded Target checkpoint disagrees with artifact preflight",
    )
    _require(
        target.get("artifact_spec_sha256") == preflight_result.target_spec_sha256,
        "loaded Target artifact specification disagrees with preflight",
    )
    _require(
        draft.get("denoising_steps") == 4
        and draft.get("native_first_timestep") == 1000
        and draft.get("hidden_layer_zero_based") == 14
        and draft.get("hidden_token_sample_count_max") == 256
        and draft.get("effective_rank_relative_cutoff") == 1e-7
        and draft.get("kv_local_attention_size") == 21
        and draft.get("kv_sink_size") == 0,
        "loaded Draft generation/risk/KV settings disagree with the protocol",
    )
    _require(
        target.get("denoising_steps") == 4
        and target.get("kv_cache_num_frames") == 3,
        "loaded Target generation/KV settings disagree with the protocol",
    )
    runtime_config = getattr(preflight_result, "config", None)
    shared_conditioning = bool(
        getattr(runtime_config, "shared_precomputed_umt5_conditioning", False)
    )
    if shared_conditioning:
        from .conditioning_cache import canonical_prompt_records, prompt_set_sha256

        draft_conditioning = draft.get("conditioning_cache")
        target_conditioning = target.get("conditioning_cache")
        _require(
            isinstance(draft_conditioning, Mapping)
            and draft_conditioning == target_conditioning,
            "Draft and Target did not load the same conditioning artifact",
        )
        _require(
            draft_conditioning.get("schema")
            == "insight2_shared_umt5_conditioning_manifest_v1"
            and draft_conditioning.get("dtype") == "bfloat16"
            and draft_conditioning.get("live_roundtrip_bitwise_equal") is True
            and isinstance(draft_conditioning.get("manifest_sha256"), str)
            and _SHA256_RE.fullmatch(draft_conditioning["manifest_sha256"]) is not None
            and isinstance(draft_conditioning.get("tensor_file_sha256"), str)
            and _SHA256_RE.fullmatch(draft_conditioning["tensor_file_sha256"])
            is not None,
            "shared conditioning audit binding is incomplete",
        )
        selected_prompts = (
            _selected_smoke_prompt_values(preflight_result)
            if preflight_result.split == "smoke"
            else _selected_evaluation_prompt_values(preflight_result)
        )
        expected_prompt_values = [
            {"prompt_id": str(item["prompt_id"]), "prompt": str(item["prompt"])}
            for item in (*preflight_result.protocol.calibration_prompts, *selected_prompts)
        ]
        expected_prompt_set = prompt_set_sha256(
            canonical_prompt_records(expected_prompt_values)
        )
        _require(
            draft_conditioning.get("prompt_set_sha256") == expected_prompt_set
            and draft_conditioning.get("prompt_count") == len(expected_prompt_values),
            "shared conditioning artifact binds another frozen prompt set",
        )
        conditioning_provenance = draft_conditioning.get("provenance")
        _require(
            isinstance(conditioning_provenance, Mapping),
            "shared conditioning provenance is absent",
        )
        conditioning_artifact_names = (
            "wan_t5_checkpoint",
            "umt5_special_tokens_map",
            "umt5_spiece_model",
            "umt5_tokenizer_config",
            "umt5_tokenizer_json",
        )
        _require(
            all(name in draft_artifacts for name in conditioning_artifact_names),
            "Draft artifact preflight omitted shared conditioning inputs",
        )
        expected_tokenizers = [
            {"name": name, "sha256": draft_artifacts[name]["sha256"]}
            for name in sorted(
                conditioning_artifact_names[1:]
            )
        ]
        _require(
            conditioning_provenance.get("text_encoder_checkpoint_sha256")
            == draft_artifacts["wan_t5_checkpoint"]["sha256"]
            and conditioning_provenance.get("tokenizer_artifacts")
            == expected_tokenizers,
            "shared conditioning checkpoint/tokenizer provenance differs from preflight",
        )
    elif runtime_config is not None:
        _require(
            "conditioning_cache" not in draft and "conditioning_cache" not in target,
            "runtime loaded a conditioning cache without the explicit config switch",
        )


def _verify_runtime_quality_contract(preflight_result: Preflight, runtime: Any) -> None:
    """Prevent long-video rows from relabelling the legacy MUSIQ proxy as VBench."""

    if preflight_result.config.schema not in LONG_VIDEO_RUN_CONFIG_SCHEMAS:
        return
    quality = preflight_result.protocol.raw.get("quality")
    _require(isinstance(quality, Mapping), "long-video protocol quality contract is invalid")
    primary = quality.get("primary")
    _require(
        isinstance(primary, str)
        and bool(primary)
        and primary
        not in {
            "musiq_spaq_lossless_rgb_mean",
            PRIMARY_QUALITY_METRIC,
        },
        "long-video formal quality must be a frozen VBench-Long metric, not MUSIQ",
    )
    evaluator = getattr(runtime, "evaluator", None)
    evaluator_metric = getattr(evaluator, "full_quality_metric_id", None)
    _require(
        evaluator_metric == primary,
        "long-video VBench-Long evaluator is not connected to the frozen quality contract",
    )


def _long_video_evaluator_contract(*, phase: str, metric_id: str) -> dict[str, Any]:
    _require(phase in {"2b-smoke", "2b-eval"}, "invalid long-video evaluator phase")
    _require(
        metric_id == VBENCH_LONG_PRIMARY_QUALITY_METRIC,
        "invalid frozen VBench-Long evaluator metric",
    )
    return {
        "schema": EVALUATOR_STATE_SCHEMA,
        "phase": phase,
        "metric_id": metric_id,
        "measurement_schema": VBENCH_LONG_QUALITY_MEASUREMENT_SCHEMA,
        "official_vbench_long_commit": VBENCH_LONG_COMMIT,
        "dimensions": list(VBENCH_LONG_DIMENSIONS),
        "aggregation": "unweighted_six_dimension_macro_mean",
    }


def _long_video_evaluator_state(
    preflight_result: Preflight, runtime: Any
) -> dict[str, Any]:
    """Return the frozen evaluator identity persisted by every long-video shard."""

    _require(
        preflight_result.config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS,
        "VBench-Long evaluator state is only defined for the long-video contract",
    )
    _verify_runtime_quality_contract(preflight_result, runtime)
    return _long_video_evaluator_contract(
        phase=preflight_result.phase,
        metric_id=str(preflight_result.protocol.raw["quality"]["primary"]),
    )


def _launch_runtime(preflight_result: Preflight) -> Any:
    """Late import keeps config/preflight unit tests Torch-free."""

    try:
        from .correction_runtime import launch_correction_runtime
    except ImportError as error:
        raise Insight2RunError(
            "cfexp.correction_runtime is unavailable; the measured run cannot start"
        ) from error
    config = preflight_result.config
    devices = preflight_result.visible_devices
    chunk_count, _rgb_frames, _duration_exact, _duration_decimal = _protocol_geometry(
        preflight_result.protocol
    )
    vbench_options: dict[str, Any] = {}
    if config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS:
        quality = preflight_result.protocol.raw.get("quality")
        _require(
            isinstance(quality, Mapping)
            and quality.get("primary") == VBENCH_LONG_PRIMARY_QUALITY_METRIC,
            "long-video runtime requires the frozen official VBench-Long metric",
        )
        if config.schema == LONG_VIDEO_RUN_CONFIG_SCHEMA:
            vbench_options = {
                "vbench_checkout": VBENCH_LONG_CHECKOUT,
                "vbench_python": VBENCH_LONG_PYTHON,
                "vbench_asset_manifest": VBENCH_LONG_ASSET_MANIFEST,
                "vbench_asset_spec": VBENCH_LONG_ASSET_SPEC,
                "vbench_metric_id": str(quality["primary"]),
            }
    shared_conditioning = bool(
        getattr(config, "shared_precomputed_umt5_conditioning", False)
    )
    shared_device = bool(getattr(config, "allow_draft_target_same_gpu", False))
    if shared_conditioning:
        selected = (
            _selected_smoke_prompt_values(preflight_result)
            if preflight_result.split == "smoke"
            else _selected_evaluation_prompt_values(preflight_result)
        )
        prompt_values = [
            {"prompt_id": str(item["prompt_id"]), "prompt": str(item["prompt"])}
            for item in (*preflight_result.protocol.calibration_prompts, *selected)
        ]
        vbench_options.update(
            {
                "conditioning_prompts": prompt_values,
                "shared_precomputed_conditioning": True,
                "allow_shared_device": shared_device,
            }
        )
    runtime = launch_correction_runtime(
        run_dir=preflight_result.run_dir,
        repo_root=preflight_result.repo_root,
        draft_artifact_spec_path=config.draft_artifact_spec_path,
        draft_artifact_spec_sha256=preflight_result.draft_spec_sha256,
        draft_verified_artifacts=preflight_result.draft_verified_artifacts,
        target_artifact_spec_path=config.target_artifact_spec_path,
        target_artifact_spec_sha256=preflight_result.target_spec_sha256,
        target_verified_artifacts=preflight_result.target_verified_artifacts,
        devices=(
            devices[config.draft_gpu_slot],
            devices[config.target_gpu_slot],
        ),
        chunk_count=chunk_count,
        # The v2 arms publish each route video for the frozen VBench-Long
        # evaluator.  Legacy greedy-oracle runs keep their historical bounded
        # storage behavior.
        write_full_videos=(config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS),
        **vbench_options,
    )
    if config.schema == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA:
        evaluator = getattr(runtime, "evaluator", None)
        _require(
            evaluator is not None
            and getattr(evaluator, "vbench_scorer", None) is None
            and callable(getattr(evaluator, "prepare_full", None))
            and callable(getattr(evaluator, "score_prepared_full", None)),
            "v3 runtime does not support deferred official VBench-Long scoring",
        )
        # The runtime factory deliberately constructs no official scorer for
        # v3, so both model workers can release their GPU allocations before
        # either evaluator process is created.  ``prepare_full`` seals the MP4
        # while the VAE worker is alive; the metric identity below records the
        # post-runtime evaluator that will consume those sealed bytes.
        evaluator.full_quality_metric_id = VBENCH_LONG_PRIMARY_QUALITY_METRIC
    return runtime


def _deferred_vbench_scorer_devices(
    preflight_result: Preflight,
) -> dict[str, str]:
    """Return the exact cache/GPU identity of the two v3 scorers."""

    _require(
        preflight_result.config.schema
        == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA,
        "deferred VBench scorers are restricted to the v3 H800 profile",
    )
    devices = preflight_result.visible_devices
    if getattr(preflight_result.config, "allow_draft_target_same_gpu", False):
        _require(
            len(devices) == 1,
            "same-GPU deferred VBench scoring requires exactly one lane GPU",
        )
        return {"scorer-0": devices[0]}
    _require(
        len(devices) == 2 and len(set(devices)) == 2,
        "deferred VBench scoring requires exactly two distinct lane GPUs",
    )
    return {"scorer-0": devices[0], "scorer-1": devices[1]}


def _score_deferred_vbench_plan(
    *,
    preflight_result: Preflight,
    evaluator: Any,
    plan: PendingMultiBudgetPlan,
) -> dict[str, QualityMeasurement]:
    """Score all sealed v3 routes on every configured post-runtime scorer."""

    try:
        from .correction_runtime import isolate_vbench_scorer_runtime_cache
        from .vbench_long import OfficialVBenchLongScorer
    except ImportError as error:
        raise Insight2RunError(
            "deferred official VBench-Long scorer is unavailable"
        ) from error

    scorer_devices = _deferred_vbench_scorer_devices(preflight_result)
    scorers: list[tuple[str, Any]] = []
    # Construction is intentionally sequential: the upstream scorer performs
    # checkout/Python preflight in its constructor and initially touches the
    # legacy cache root.  Isolation repeats that preflight under the scorer's
    # final private cache before the first scoring call.
    for scorer_id, device in scorer_devices.items():
        scorer = OfficialVBenchLongScorer(
            run_dir=preflight_result.run_dir,
            checkout=VBENCH_LONG_CHECKOUT,
            python_executable=VBENCH_LONG_PYTHON,
            asset_manifest=VBENCH_LONG_ASSET_MANIFEST,
            asset_spec=VBENCH_LONG_ASSET_SPEC,
            visible_device=device,
        )
        scorers.append(
            (
                scorer_id,
                isolate_vbench_scorer_runtime_cache(
                    scorer, scorer_id=scorer_id
                ),
            )
        )

    unique: dict[str, Any] = {}
    for entry in plan.entries:
        pending = entry.rollout
        measurement_id = pending.prepared_quality.measurement_id
        previous = unique.get(measurement_id)
        _require(
            previous is None or previous == pending,
            "one deferred measurement ID is bound to conflicting routes",
        )
        unique[measurement_id] = pending
    _require(bool(unique), "deferred VBench plan has no unique routes")
    ordered = tuple(unique[key] for key in sorted(unique))
    scorer_count = len(scorers)
    _require(scorer_count in {1, 2}, "deferred VBench scorer count is invalid")
    buckets = tuple(ordered[index::scorer_count] for index in range(scorer_count))

    def score_bucket(scorer: Any, pending_routes: Sequence[Any]) -> dict[str, QualityMeasurement]:
        batch_method = getattr(evaluator, "score_prepared_full_batch", None)
        if callable(batch_method) and pending_routes:
            return batch_method(
                entries=tuple(
                    (pending.history, pending.prepared_quality)
                    for pending in pending_routes
                ),
                scorer=scorer,
            )
        completed: dict[str, QualityMeasurement] = {}
        for pending in pending_routes:
            measurement_id = pending.prepared_quality.measurement_id
            completed[measurement_id] = evaluator.score_prepared_full(
                history=pending.history,
                prepared=pending.prepared_quality,
                scorer=scorer,
            )
        return completed

    completed: dict[str, QualityMeasurement] = {}
    with ThreadPoolExecutor(
        max_workers=scorer_count, thread_name_prefix="insight2-vbench"
    ) as executor:
        futures = [
            executor.submit(score_bucket, scorer, bucket)
            for (_scorer_id, scorer), bucket in zip(scorers, buckets, strict=True)
        ]
        for future in futures:
            shard = future.result()
            _require(
                not (set(completed) & set(shard)),
                "deferred VBench scorers returned duplicate measurements",
            )
            completed.update(shard)
    _require(
        set(completed) == set(unique),
        "deferred VBench scores do not exactly cover the pending route set",
    )
    return completed


def execute(preflight_result: Preflight) -> dict[str, Any]:
    """Execute exactly one requested phase, then export and analyze its real rows."""

    calibration_receipt_result = getattr(
        preflight_result, "calibration_receipt_verified", None
    )
    receipt_source = getattr(preflight_result, "calibration_receipt_source", None)
    if receipt_source is not None:
        _require(
            getattr(preflight_result, "calibration_receipt_binding", None) is not None
            and isinstance(
                getattr(preflight_result, "calibration_receipt_sha256", None), str
            ),
            "calibration receipt import lost its trusted preflight binding",
        )
        try:
            calibration_receipt_result = restore_calibration_receipt(
                source_run_dir=receipt_source,
                destination_run_dir=preflight_result.run_dir,
                expected_binding=preflight_result.calibration_receipt_binding,
                expected_receipt_sha256=preflight_result.calibration_receipt_sha256,
                preverified=calibration_receipt_result,
            )
        except CalibrationReceiptError as error:
            raise Insight2RunError(
                f"calibration receipt import failed before runtime launch: {error}"
            ) from error
    protocol = preflight_result.protocol
    calibration = _prompt_cases(protocol.calibration_prompts, split="calibration")
    evaluation = _prompt_cases(
        _selected_evaluation_prompt_values(preflight_result), split="evaluation"
    )
    smoke = _prompt_cases(
        _selected_smoke_prompt_values(preflight_result), split="evaluation"
    )
    phase = preflight_result.phase
    chunk_count, frame_count, duration_exact, duration_decimal = _protocol_geometry(
        protocol
    )
    phase_seeds = _phase_seeds(
        protocol,
        preflight_result.config,
        split=preflight_result.split,
    )
    deferred_plan: PendingMultiBudgetPlan | None = None
    deferred_evaluator: Any | None = None
    with _launch_runtime(preflight_result) as runtime:
        model_state = _runtime_model_state(runtime)
        verify_runtime_model_state(preflight_result, model_state)
        _verify_runtime_quality_contract(preflight_result, runtime)
        _write_json(
            preflight_result.run_dir / "logs" / "model-state.json",
            {
                "schema": MODEL_STATE_SCHEMA,
                "phase": phase,
                "models": model_state,
            },
        )
        if preflight_result.config.schema in LONG_VIDEO_RUN_CONFIG_SCHEMAS:
            _write_json(
                preflight_result.run_dir / "logs" / "evaluator-state.json",
                _long_video_evaluator_state(preflight_result, runtime),
            )
        campaign = Insight2Campaign(
            run_dir=preflight_result.run_dir,
            protocol_id=str(protocol.raw["protocol_id"]),
            chunk_count=chunk_count,
            seeds=phase_seeds,
            calibration_prompts=calibration,
            evaluation_prompts=evaluation,
            smoke_prompts=smoke,
            draft=runtime.draft,
            target=runtime.target,
            noise_provider=runtime.noise_provider,
            evaluator=runtime.evaluator,
        )
        if getattr(preflight_result, "seal_calibration_receipt", False):
            _require(
                preflight_result.calibration_receipt_binding is not None,
                "calibration receipt sealing lost its trusted preflight binding",
            )
            # Generate/validate C10 exactly once, before any smoke evaluation
            # route can append non-calibration Target lifecycle records.
            campaign.run_calibration()
            try:
                calibration_receipt_result = seal_calibration_receipt(
                    source_run_dir=preflight_result.run_dir,
                    binding=preflight_result.calibration_receipt_binding,
                )
            except CalibrationReceiptError as error:
                raise Insight2RunError(
                    f"calibration receipt sealing failed: {error}"
                ) from error
        if phase.startswith("2a-"):
            rows = campaign.run_single_point(split=preflight_result.split)
        elif (
            preflight_result.config.schema
            == LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA
        ):
            deferred_plan = campaign.prepare_multi_budget(
                split=preflight_result.split,
                budgets=preflight_result.config.budgets,
                random_repeats=preflight_result.config.random_repeats,
                strategies=preflight_result.config.strategies,
                coverage_window_radius_chunks=getattr(
                    preflight_result.config,
                    "coverage_window_radius_chunks",
                    None,
                ),
                coverage_minimum_gap_chunks=getattr(
                    preflight_result.config,
                    "coverage_minimum_gap_chunks",
                    None,
                ),
                coverage_maximum_gap_chunks=getattr(
                    preflight_result.config,
                    "coverage_maximum_gap_chunks",
                    None,
                ),
            )
            deferred_evaluator = runtime.evaluator
        else:
            rows = campaign.run_multi_budget(
                split=preflight_result.split,
                budgets=preflight_result.config.budgets,
                random_repeats=preflight_result.config.random_repeats,
                strategies=preflight_result.config.strategies,
                coverage_window_radius_chunks=getattr(
                    preflight_result.config,
                    "coverage_window_radius_chunks",
                    None,
                ),
                coverage_minimum_gap_chunks=getattr(
                    preflight_result.config,
                    "coverage_minimum_gap_chunks",
                    None,
                ),
                coverage_maximum_gap_chunks=getattr(
                    preflight_result.config,
                    "coverage_maximum_gap_chunks",
                    None,
                ),
                include_greedy_oracle=(
                    "greedy_oracle" in preflight_result.config.strategies
                ),
            )

    if deferred_plan is not None:
        _require(
            deferred_evaluator is not None,
            "deferred VBench plan lost its post-runtime evaluator",
        )
        completed_qualities = _score_deferred_vbench_plan(
            preflight_result=preflight_result,
            evaluator=deferred_evaluator,
            plan=deferred_plan,
        )
        # Raw rows are appended only after every unique route has a bound
        # official score.  A scorer exception therefore cannot produce a
        # partial raw dataset or any downstream success/terminal artifact.
        rows = campaign.finalize_multi_budget(
            plan=deferred_plan,
            qualities=completed_qualities,
        )

    target_rpc_report = audit_target_rpc_accounting(
        run_dir=preflight_result.run_dir,
        rows=rows,
        phase=phase,
        preflight_result=preflight_result,
    )
    csv_path, json_path = export_raw_rows(
        run_dir=preflight_result.run_dir,
        phase=phase,
        rows=rows,
    )
    kv_report = audit_kv_consistency(run_dir=preflight_result.run_dir, rows=rows)
    shard = getattr(preflight_result, "prompt_shard", None)
    terminal_audit: dict[str, Any] | None = None
    if shard is None:
        analysis_artifacts = analyze_export(
            preflight_result=preflight_result, csv_path=csv_path
        )
    else:
        analysis_artifacts = _write_partial_shard_result(
            preflight_result=preflight_result,
            rows=rows,
        )
        terminal_audit = _write_prompt_shard_terminal_audit(
            preflight_result=preflight_result,
            rows=rows,
            raw_csv_path=csv_path,
            raw_json_path=json_path,
            target_rpc_report=target_rpc_report,
            kv_report=kv_report,
            partial_path=analysis_artifacts["partial_shard"],
        )
    result = {
        "schema": RUN_RESULT_SCHEMA,
        "status": "succeeded",
        "phase": phase,
        "split": preflight_result.split,
        "protocol_id": protocol.raw["protocol_id"],
        "protocol_sha256": preflight_result.protocol_sha256,
        "row_count": len(rows),
        "raw_csv": csv_path.relative_to(preflight_result.run_dir).as_posix(),
        "raw_json": json_path.relative_to(preflight_result.run_dir).as_posix(),
        "append_only_jsonl": (
            "raw/insight2a.jsonl" if phase.startswith("2a-") else "raw/insight2b.jsonl"
        ),
        "analysis": analysis_artifacts,
        "kv_audit": {
            "path": "logs/kv-consistency.json",
            "checked_checkpoint_contexts": kv_report["checked_checkpoint_contexts"],
            "checked_single_point_pairs": kv_report["checked_single_point_pairs"],
        },
        "target_rpc_audit": {
            "path": "logs/target-rpc-accounting.json",
            "attempted": target_rpc_report["attempted"],
            "backend_returned": target_rpc_report["backend_returned"],
            "completed": target_rpc_report["completed"],
            "unique_completed_logical_calls": target_rpc_report[
                "unique_completed_logical_calls"
            ],
            "reconciled_raw_scopes": target_rpc_report["reconciled_raw_scopes"],
        },
        "chunk_count": chunk_count,
        "frame_count": frame_count,
        "fps": int(protocol.raw["design"]["fps"]),
        "duration_seconds_exact": duration_exact,
        "duration_seconds_decimal": duration_decimal,
    }
    if shard is not None:
        _require(isinstance(shard, PromptShard), "trusted prompt shard has an invalid type")
        result.update(
            {
                "completion_scope": "partial_prompt_shard",
                "partial": True,
                "prompt_shard": _prompt_shard_metadata(shard),
                "complete_dataset_statistics_allowed": False,
                "terminal_audit": terminal_audit,
            }
        )
    if calibration_receipt_result is not None:
        result["calibration_receipt"] = {
            **calibration_receipt_result.public_binding(),
            "mode": (
                "sealed_by_this_run"
                if getattr(preflight_result, "seal_calibration_receipt", False)
                else "read_only_reuse"
            ),
            "relative_path": (
                CALIBRATION_RECEIPT_RELATIVE
                if getattr(preflight_result, "seal_calibration_receipt", False)
                else None
            ),
        }
    _write_json(preflight_result.run_dir / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen Self-Forcing/Krea Insight 2 phase")
    parser.add_argument("--config", required=True, help="repository-relative frozen run config")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--prompt-shard-index",
        type=int,
        help=(
            "zero-based fixed prompt shard (0-3); valid only for long-video 2b-eval"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify all frozen inputs without loading a model or emitting metrics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    run_dir: Path | None = None
    try:
        run_dir = _run_directory()
        checked = preflight(
            repo_root=repo_root,
            config_path=args.config,
            phase=str(args.phase),
            prompt_shard_index=args.prompt_shard_index,
        )
        if args.preflight_only:
            result: dict[str, Any] = {
                "schema": RUN_RESULT_SCHEMA,
                "status": "preflight_only",
                "phase": checked.phase,
                "metrics_emitted": False,
            }
            if checked.prompt_shard is not None:
                result.update(
                    {
                        "completion_scope": "partial_prompt_shard",
                        "partial": True,
                        "prompt_shard": _prompt_shard_metadata(
                            checked.prompt_shard
                        ),
                        "complete_dataset_statistics_allowed": False,
                    }
                )
            _write_json(checked.run_dir / "result.json", result)
            return 0
        execute(checked)
        return 0
    except KeyboardInterrupt:
        error: BaseException = Insight2RunError("interrupted by operator")
        exit_code = 130
    except BaseException as caught:
        error = caught
        exit_code = 2
    if run_dir is not None and run_dir.is_dir():
        try:
            _write_json(
                run_dir / "logs" / "failure.json",
                {
                    "schema": "insight2_failure_v1",
                    "phase": str(args.phase),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exception(
                        type(error), error, error.__traceback__
                    ),
                    "metrics_fabricated": False,
                },
            )
        except BaseException as log_error:
            print(f"insight2-run: could not write failure log: {log_error}", file=sys.stderr)
    print(f"insight2-run: {type(error).__name__}: {error}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUNDARY_GATED_LONG_VIDEO_STRATEGIES",
    "COVERAGE_CONSTRAINED_LONG_VIDEO_STRATEGIES",
    "LOOKAHEAD_COVERAGE_LONG_VIDEO_STRATEGIES",
    "Insight2RunConfig",
    "Insight2RunError",
    "LONG_VIDEO_H800_DEADLINE_RUN_CONFIG_SCHEMA",
    "LONG_VIDEO_PROMPT_SHARD_COUNT",
    "LONG_VIDEO_RUN_CONFIG_SCHEMAS",
    "LONG_VIDEO_STRATEGIES",
    "PHASES",
    "PROMPT_SHARD_TERMINAL_AUDIT_SCHEMA",
    "Preflight",
    "PromptShard",
    "analyze_export",
    "audit_target_rpc_accounting",
    "audit_kv_consistency",
    "execute",
    "export_raw_rows",
    "load_run_config",
    "main",
    "preflight",
    "verify_runtime_model_state",
]
