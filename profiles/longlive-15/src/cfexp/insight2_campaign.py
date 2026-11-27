"""Resumable orchestration for the frozen Insight 2 correction experiments.

This module is intentionally model-runtime agnostic.  It coordinates two
persistent backend clients through small protocols, while
``correction_state`` remains the authority for latent-history and KV-cache
ownership invariants.  In particular:

* every 2(a) Target candidate forks from the immutable baseline prefix;
* every 2(b) candidate first stages the current Draft result and a Target
  replacement, when selected, is committed into the history used by every
  later Draft chunk;
* calibration prompts and evaluation prompts are disjoint by default; an
  explicitly labelled full-calibration development replay is never held out; and
* checkpoints and raw records are append-only, so an interrupted campaign can
  resume without rewriting a completed object.

The concrete JSONL worker adapter is deliberately outside this module.  Unit
tests use an in-process fake, and a production adapter only has to implement
the protocols defined below.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .correction_policy import (
    BoundarySmoothnessGate,
    CoverageConstrainedOnlineRankGuidedPolicy,
    FullVideoRollout,
    LookaheadCoverageConstrainedOnlineRankGuidedPolicy,
    OnlineRankGuidedPolicy,
    RankRiskCalibrator,
    SuddenDropRiskCalibrator,
    causal_erank_sudden_drop,
    greedy_oracle_plan,
    random_positions,
    uniform_positions,
)
from .stage_timing import stage, timing_session
from .motion_repair_policy import MotionAcceptanceGate

from .correction_state import (
    CorrectionState,
    CorrectionStateError,
    GenerationContext,
    InitialNoiseDescriptor,
    KVCacheDescriptor,
    LatentHistory,
    LatentSegment,
    SinglePointFork,
)
from .self_forcing_frame_adapter import (
    AdapterError,
    validate_effective_rank_evidence,
    validate_self_forcing_forward_trace,
    validate_sampled_kv_cache_witness,
)


CAMPAIGN_SCHEMA = "insight2_campaign_manifest_v1"
CHECKPOINT_SCHEMA = "insight2_campaign_checkpoint_v2"
CALIBRATION_SCHEMA = "insight2_rank_calibration_v1"
RAW_2A_SCHEMA = "insight2a_raw_row_v1"
RAW_2B_SCHEMA = "insight2b_raw_row_v3"
ROLLOUT_SCORE_SCHEMA = "insight2_full_rollout_score_v2"
PREPARED_FULL_QUALITY_SCHEMA = "insight2_prepared_full_quality_v1"
PENDING_ROLLOUT_SCHEMA = "insight2_pending_rollout_v2"
PENDING_MULTI_BUDGET_PLAN_SCHEMA = "insight2_pending_multi_budget_plan_v2"
FORK_EVIDENCE_SCHEMA = "insight2a_single_point_fork_evidence_v1"
TARGET_RPC_SCHEMA = "insight2_target_rpc_event_v3"
TARGET_RPC_LIFECYCLE_SCHEMA = "insight2_target_rpc_lifecycle_v2"
TARGET_RPC_DIAGNOSTICS_SCHEMA = "insight2_target_rpc_diagnostics_v3"
NOISE_NAMESPACE = "insight2-correction-v1"
SUPPORTED_CHUNK_COUNTS = frozenset({16, 30, 60, 120})
SMOKE_PROMPT_COUNT_BY_CHUNK_COUNT = {16: 10, 30: 1, 60: 1, 120: 1}

TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS = frozenset(
    {
        "kv_witness_sha256",
        "kv_witness",
        "forward_trace",
        "denoising_forward_trace",
        "denoising_schedule",
        "explicit_initial_noise_verified",
        "boundary_velocity_witness",
    }
)
_GENERATED_CHUNK_KEYS = frozenset(
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
_DRAFT_PROPOSAL_DIAGNOSTIC_KEYS = frozenset(
    {
        "kv_witness_sha256",
        "kv_witness",
        "forward_trace",
        "rank_evidence",
        "boundary_velocity_witness",
    }
)
BOUNDARY_VELOCITY_WITNESS_SCHEMA = "correction_boundary_velocity_witness_v1"
BOUNDARY_VELOCITY_METRIC_ID = "latent_velocity_transition_discontinuity_v1"
BOUNDARY_VELOCITY_ROUND_DECIMALS = 12
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

TARGET_RPC_PHASES = frozenset(
    {
        "unscoped",
        "single_point_candidate",
        "calibration_baseline",
        "evaluation_baseline",
        "route_materialization",
        "selection_search",
    }
)
TARGET_RPC_LIFECYCLE_KEYS = frozenset(
    {
        "schema",
        "event_id",
        "attempt_id",
        "logical_call_id",
        "scope_id",
        "phase",
        "status",
        "request_id",
        "model_id",
        "model_artifact_sha256",
        "prompt_sha256",
        "chunk_index",
        "internal_seed",
        "history_latent_paths",
        "output_latent_path",
        "history_chunk_count",
        "initial_noise_path",
        "output_latent_sha256",
        "kv_witness_sha256",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class CampaignError(RuntimeError):
    """Raised when campaign state is incomplete, inconsistent, or mutable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CampaignError(f"value is not canonical JSON: {error}") from error


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_target_rpc_lifecycle_record(
    record: Any,
    *,
    event_ids: set[str],
    attempts: dict[str, dict[str, Any]],
    line_number: int,
) -> dict[str, Any]:
    """Validate one ledger record against all preceding records.

    The same stateful validator is used by the terminal full-file loader and
    the campaign's incremental accounting index.  Keeping one validator avoids
    making the O(total-records) fast path weaker than the final audit.
    """

    _require(
        isinstance(record, dict),
        f"Target RPC lifecycle line {line_number} must be an object",
    )
    _require(
        set(record) == TARGET_RPC_LIFECYCLE_KEYS,
        "Target RPC lifecycle fields are invalid",
    )
    _require(
        record["schema"] == TARGET_RPC_LIFECYCLE_SCHEMA,
        "Target RPC lifecycle schema drifted",
    )
    event_id = record["event_id"]
    _require(
        isinstance(event_id, str) and _SHA256_RE.fullmatch(event_id) is not None,
        "Target RPC lifecycle event_id is invalid",
    )
    expected_event_id = _digest_json(
        {key: value for key, value in record.items() if key != "event_id"}
    )
    _require(event_id == expected_event_id, "Target RPC lifecycle event digest drifted")
    _require(event_id not in event_ids, "Target RPC lifecycle event_id is duplicated")
    event_ids.add(event_id)
    attempt_id = record["attempt_id"]
    logical_call_id = record["logical_call_id"]
    for label, value in (
        ("attempt_id", attempt_id),
        ("logical_call_id", logical_call_id),
        ("model_artifact_sha256", record["model_artifact_sha256"]),
        ("prompt_sha256", record["prompt_sha256"]),
    ):
        _require(
            isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
            f"Target RPC lifecycle {label} is invalid",
        )
    scope_id = record["scope_id"]
    _require(
        scope_id is None
        or (isinstance(scope_id, str) and _SHA256_RE.fullmatch(scope_id) is not None),
        "Target RPC lifecycle scope_id is invalid",
    )
    phase = record["phase"]
    _require(phase in TARGET_RPC_PHASES, "Target RPC lifecycle phase is invalid")
    _require(
        (scope_id is None) == (phase == "unscoped"),
        "Target RPC lifecycle scope and phase disagree",
    )
    status = record["status"]
    _require(
        status in {"started", "backend_returned", "completed"},
        "Target RPC lifecycle status is invalid",
    )
    _require(
        isinstance(record["request_id"], str) and bool(record["request_id"]),
        "Target RPC lifecycle request_id is invalid",
    )
    _require(
        isinstance(record["model_id"], str) and bool(record["model_id"]),
        "Target RPC lifecycle model_id is invalid",
    )
    chunk_index = record["chunk_index"]
    history_count = record["history_chunk_count"]
    _require(
        isinstance(chunk_index, int)
        and not isinstance(chunk_index, bool)
        and 0 <= chunk_index < max(SUPPORTED_CHUNK_COUNTS)
        and history_count == chunk_index,
        "Target RPC lifecycle chunk/history count is invalid",
    )
    _require(
        isinstance(record["internal_seed"], int)
        and not isinstance(record["internal_seed"], bool)
        and record["internal_seed"] >= 0,
        "Target RPC lifecycle internal_seed is invalid",
    )
    history_paths = record["history_latent_paths"]
    _require(
        isinstance(history_paths, list)
        and len(history_paths) == history_count
        and len(set(history_paths)) == len(history_paths),
        "Target RPC lifecycle history paths are invalid",
    )
    for index, value in enumerate(history_paths):
        _safe_relative_path(value, label=f"lifecycle history_latent_paths[{index}]")
    _safe_relative_path(record["output_latent_path"], label="lifecycle output path")
    _safe_relative_path(record["initial_noise_path"], label="lifecycle noise path")
    output_sha = record["output_latent_sha256"]
    kv_sha = record["kv_witness_sha256"]
    if status == "completed":
        _require(
            isinstance(output_sha, str)
            and _SHA256_RE.fullmatch(output_sha) is not None
            and isinstance(kv_sha, str)
            and _SHA256_RE.fullmatch(kv_sha) is not None,
            "completed Target RPC lifecycle event lacks output evidence",
        )
    else:
        _require(
            output_sha is None and kv_sha is None,
            "pre-completion Target RPC lifecycle event carries output evidence",
        )
    binding = {
        key: record[key]
        for key in TARGET_RPC_LIFECYCLE_KEYS
        if key
        not in {
            "schema",
            "event_id",
            "status",
            "output_latent_sha256",
            "kv_witness_sha256",
        }
    }
    state = attempts.setdefault(attempt_id, {"binding": binding, "statuses": []})
    _require(
        state["binding"] == binding,
        "Target RPC lifecycle request binding changed within one attempt",
    )
    statuses = state["statuses"]
    _require(isinstance(statuses, list), "Target RPC lifecycle state is invalid")
    expected_status = (
        ("started", "backend_returned", "completed")[len(statuses)]
        if len(statuses) < 3
        else None
    )
    _require(
        status == expected_status,
        "Target RPC lifecycle stages are duplicated or out of order",
    )
    statuses.append(status)
    return record


def load_target_rpc_lifecycle(
    run_dir: str | Path, *, required: bool
) -> tuple[dict[str, Any], ...]:
    """Load the worker ledger with strict digest, binding, and stage-order checks."""

    path = Path(run_dir) / "logs" / "target-rpc-lifecycle.jsonl"
    if not path.exists():
        _require(not required, "formal Target worker omitted its RPC lifecycle log")
        return ()
    _require(path.is_file() and not path.is_symlink(), "Target RPC lifecycle log is not regular")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CampaignError(f"cannot read Target RPC lifecycle log: {error}") from error
    records: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    attempts: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(lines, 1):
        _require(bool(raw), f"Target RPC lifecycle line {line_number} is blank")
        try:
            record = json.loads(raw, object_pairs_hook=_duplicate_checked_object)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise CampaignError(
                f"Target RPC lifecycle line {line_number} is invalid JSON: {error}"
            ) from error
        records.append(
            _validate_target_rpc_lifecycle_record(
                record,
                event_ids=event_ids,
                attempts=attempts,
                line_number=line_number,
            )
        )
    return tuple(records)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
            size = handle.tell()
    except OSError as error:
        raise CampaignError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest(), size


def _safe_relative_path(value: str, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    _require("\\" not in value and "\x00" not in value, f"{label} is not canonical")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    _require(
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and str(posix) == value
        and all(part not in {"", ".", ".."} for part in posix.parts),
        f"{label} must be a canonical relative POSIX path",
    )
    return value


def _verify_run_file(
    run_dir: Path,
    relative: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    """Verify an immutable evidence file without following run-local symlinks."""

    relative = _safe_relative_path(relative, label=label)
    _require(
        isinstance(expected_sha256, str)
        and _SHA256_RE.fullmatch(expected_sha256) is not None,
        f"{label} SHA-256 is invalid",
    )
    try:
        root = run_dir.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot resolve run directory: {error}") from error
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        _require(not current.is_symlink(), f"{label} may not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise CampaignError(f"{label} escapes or is unavailable: {relative}") from error
    _require(resolved.is_file(), f"{label} must be a regular file")
    actual_sha256, size = _sha256_file(resolved)
    _require(size > 0, f"{label} must not be empty")
    _require(actual_sha256 == expected_sha256, f"{label} SHA-256 mismatch")
    return resolved


def _finite_unit(value: float, *, label: str) -> float:
    parsed = float(value)
    _require(math.isfinite(parsed) and 0.0 <= parsed <= 1.0, f"{label} must be in [0,1]")
    return parsed


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one canonical JSON object, or verify an identical existing one."""

    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise CampaignError(f"cannot verify existing object {path}: {error}") from error
        _require(existing == raw, f"immutable object conflicts with existing file: {path}")
        return
    except OSError as error:
        raise CampaignError(f"cannot create immutable object {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot load JSON object {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


class AppendOnlyJsonl:
    """Small durable event log with unique, content-bound event IDs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise CampaignError(f"cannot read append-only log {path}: {error}") from error
        for line_number, line in enumerate(lines, 1):
            _require(bool(line), f"{path}:{line_number}: blank JSONL line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CampaignError(f"{path}:{line_number}: invalid JSON: {error}") from error
            _require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            event_id = value.get("event_id")
            _require(
                isinstance(event_id, str) and _SHA256_RE.fullmatch(event_id) is not None,
                f"{path}:{line_number}: invalid event_id",
            )
            _require(event_id not in self._records, f"{path}: duplicate event_id {event_id}")
            self._records[event_id] = value

    def has(self, event_id: str) -> bool:
        return event_id in self._records

    def get(self, event_id: str) -> dict[str, Any] | None:
        value = self._records.get(event_id)
        return None if value is None else dict(value)

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(value) for value in self._records.values())

    def append(self, value: Mapping[str, Any]) -> bool:
        record = dict(value)
        event_id = record.get("event_id")
        _require(
            isinstance(event_id, str) and _SHA256_RE.fullmatch(event_id) is not None,
            "append-only row requires a lowercase SHA-256 event_id",
        )
        existing = self._records.get(event_id)
        if existing is not None:
            _require(existing == record, f"event_id {event_id} has conflicting content")
            return False
        raw = _canonical_bytes(record) + b"\n"
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise CampaignError(f"cannot append {self.path}: {error}") from error
        self._records[event_id] = record
        return True


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    prompt: str
    split: str

    def __post_init__(self) -> None:
        _require(
            isinstance(self.prompt_id, str)
            and _IDENTIFIER_RE.fullmatch(self.prompt_id) is not None,
            "prompt_id must be a portable identifier",
        )
        _require(isinstance(self.prompt, str) and bool(self.prompt.strip()), "prompt is empty")
        _require(self.split in {"calibration", "evaluation"}, "prompt split is invalid")


@dataclass(frozen=True)
class NoiseMaterialization:
    descriptor: InitialNoiseDescriptor
    tensor_path: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.tensor_path, label="noise tensor_path")


@dataclass(frozen=True)
class ChunkGenerationRequest:
    prompt: PromptCase
    history: LatentHistory
    noise: NoiseMaterialization
    output_latent_path: str
    capture_hidden: bool
    target_logical_call_id: str | None = None
    target_call_scope_id: str | None = None
    target_call_phase: str | None = None
    kv_branch_action: str | None = None

    def __post_init__(self) -> None:
        _require(
            self.kv_branch_action is None
            or (isinstance(self.kv_branch_action, str)
                and self.kv_branch_action in ("none", "preserve", "consume")),
            "kv_branch_action is invalid",
        )
        _safe_relative_path(self.output_latent_path, label="output_latent_path")
        _require(
            self.noise.descriptor.prompt_id == self.prompt.prompt_id,
            "noise prompt does not match generation prompt",
        )
        _require(
            self.noise.descriptor.chunk_index == self.history.next_chunk_index,
            "noise chunk does not match next history chunk",
        )
        for label, value in (
            ("target_logical_call_id", self.target_logical_call_id),
            ("target_call_scope_id", self.target_call_scope_id),
        ):
            _require(
                value is None
                or (isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None),
                f"{label} must be null or a lowercase SHA-256",
            )
        _require(
            self.target_call_phase is None or self.target_call_phase in TARGET_RPC_PHASES,
            "target_call_phase is invalid",
        )


@dataclass(frozen=True)
class GeneratedChunk:
    context: GenerationContext
    segment: LatentSegment
    effective_rank: float | None = None
    hidden_state_path: str | None = None
    hidden_state_sha256: str | None = None
    diagnostics: Mapping[str, Any] | None = None
    target_attempt_id: str | None = None
    target_logical_call_id: str | None = None


@dataclass(frozen=True)
class QualityMeasurement:
    primary: float
    temporal_consistency: float
    frame_scores: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary", _finite_unit(self.primary, label="primary quality"))
        object.__setattr__(
            self,
            "temporal_consistency",
            _finite_unit(self.temporal_consistency, label="temporal consistency"),
        )
        _require(
            isinstance(self.frame_scores, tuple),
            "frame_scores must be a tuple",
        )
        parsed_scores = tuple(
            _finite_unit(value, label=f"frame_scores[{index}]")
            for index, value in enumerate(self.frame_scores)
        )
        if parsed_scores:
            _require(
                math.fsum(parsed_scores) / len(parsed_scores) == self.primary,
                "frame_scores mean differs from primary quality",
            )
        object.__setattr__(self, "frame_scores", parsed_scores)


class NoiseProvider(Protocol):
    namespace: str

    def materialize(
        self, *, prompt_id: str, trajectory_seed: int, chunk_index: int
    ) -> NoiseMaterialization: ...


class ChunkBackendClient(Protocol):
    role: str
    model_id: str
    model_artifact_sha256: str

    def generate(self, request: ChunkGenerationRequest) -> GeneratedChunk: ...


class QualityEvaluator(Protocol):
    def score_local(
        self, *, history: LatentHistory, chunk_index: int, measurement_id: str
    ) -> QualityMeasurement: ...

    def score_full(
        self, *, history: LatentHistory, measurement_id: str
    ) -> QualityMeasurement: ...

    def prepare_full(
        self, *, history: LatentHistory, measurement_id: str
    ) -> PreparedFullQuality: ...


@dataclass(frozen=True)
class BaselineTrajectory:
    history: LatentHistory
    generated: tuple[GeneratedChunk, ...]


@dataclass(frozen=True)
class SequentialRollout:
    result: FullVideoRollout
    temporal_consistency: float
    history: LatentHistory
    proposed_target_chunks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        proposed = (
            self.result.corrected_chunks
            if self.proposed_target_chunks is None
            else self.proposed_target_chunks
        )
        _require(
            proposed == tuple(sorted(set(proposed)))
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 0 < item < self.history.next_chunk_index
                for item in proposed
            )
            and set(self.result.corrected_chunks).issubset(proposed),
            "sequential rollout proposed Target chunks are invalid",
        )
        object.__setattr__(self, "proposed_target_chunks", proposed)


@dataclass(frozen=True)
class PreparedFullQuality:
    """Immutable decoded-video input that can be scored after model shutdown."""

    measurement_id: str
    history_manifest_sha256: str
    history_head_sha256: str
    prompt_id: str
    latent_chunk_count: int
    decode_evidence_path: str
    decode_evidence_sha256: str
    video_path: str
    video_sha256: str
    video_nbytes: int
    fps: int
    frame_count: int

    def __post_init__(self) -> None:
        _require(
            isinstance(self.measurement_id, str)
            and _IDENTIFIER_RE.fullmatch(self.measurement_id) is not None,
            "prepared measurement_id must be portable",
        )
        _require(
            isinstance(self.prompt_id, str)
            and _IDENTIFIER_RE.fullmatch(self.prompt_id) is not None,
            "prepared prompt_id must be portable",
        )
        for label, value in (
            ("history_manifest_sha256", self.history_manifest_sha256),
            ("history_head_sha256", self.history_head_sha256),
            ("decode_evidence_sha256", self.decode_evidence_sha256),
            ("video_sha256", self.video_sha256),
        ):
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"prepared {label} is invalid",
            )
        _safe_relative_path(
            self.decode_evidence_path, label="prepared decode_evidence_path"
        )
        _safe_relative_path(self.video_path, label="prepared video_path")
        _require(
            isinstance(self.latent_chunk_count, int)
            and not isinstance(self.latent_chunk_count, bool)
            and 1 <= self.latent_chunk_count <= max(SUPPORTED_CHUNK_COUNTS),
            "prepared latent_chunk_count is invalid",
        )
        _require(
            isinstance(self.video_nbytes, int)
            and not isinstance(self.video_nbytes, bool)
            and self.video_nbytes > 0,
            "prepared video_nbytes is invalid",
        )
        _require(self.fps == 24, "prepared video fps must equal 24")
        _require(
            isinstance(self.frame_count, int)
            and not isinstance(self.frame_count, bool)
            and self.frame_count == 12 * self.latent_chunk_count - 3,
            "prepared video frame_count disagrees with the VAE geometry",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PREPARED_FULL_QUALITY_SCHEMA,
            "measurement_id": self.measurement_id,
            "history_manifest_sha256": self.history_manifest_sha256,
            "history_head_sha256": self.history_head_sha256,
            "prompt_id": self.prompt_id,
            "latent_chunk_count": self.latent_chunk_count,
            "decode_evidence_path": self.decode_evidence_path,
            "decode_evidence_sha256": self.decode_evidence_sha256,
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "video_nbytes": self.video_nbytes,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> PreparedFullQuality:
        _require(isinstance(value, Mapping), "prepared full-quality input must be an object")
        expected = {
            "schema",
            "measurement_id",
            "history_manifest_sha256",
            "history_head_sha256",
            "prompt_id",
            "latent_chunk_count",
            "decode_evidence_path",
            "decode_evidence_sha256",
            "video_path",
            "video_sha256",
            "video_nbytes",
            "fps",
            "frame_count",
        }
        _require(set(value) == expected, "prepared full-quality fields are invalid")
        _require(
            value.get("schema") == PREPARED_FULL_QUALITY_SCHEMA,
            "prepared full-quality schema mismatch",
        )
        return cls(
            measurement_id=value["measurement_id"],
            history_manifest_sha256=value["history_manifest_sha256"],
            history_head_sha256=value["history_head_sha256"],
            prompt_id=value["prompt_id"],
            latent_chunk_count=value["latent_chunk_count"],
            decode_evidence_path=value["decode_evidence_path"],
            decode_evidence_sha256=value["decode_evidence_sha256"],
            video_path=value["video_path"],
            video_sha256=value["video_sha256"],
            video_nbytes=value["video_nbytes"],
            fps=value["fps"],
            frame_count=value["frame_count"],
        )


@dataclass(frozen=True)
class PendingSequentialRollout:
    """A complete generated trajectory awaiting only full-video scoring."""

    route_key: str
    route_id: str
    corrected_chunks: tuple[int, ...]
    history: LatentHistory
    prepared_quality: PreparedFullQuality
    proposed_target_chunks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.route_key, str)
            and re.fullmatch(r"[0-9a-f]{48}", self.route_key) is not None,
            "pending rollout route_key is invalid",
        )
        _require(
            isinstance(self.route_id, str) and bool(self.route_id),
            "pending rollout route_id is invalid",
        )
        _require(
            self.corrected_chunks == tuple(sorted(set(self.corrected_chunks)))
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 0 < item < self.history.next_chunk_index
                for item in self.corrected_chunks
            ),
            "pending rollout corrected_chunks are invalid",
        )
        proposed = (
            self.corrected_chunks
            if self.proposed_target_chunks is None
            else self.proposed_target_chunks
        )
        _require(
            proposed == tuple(sorted(set(proposed)))
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 0 < item < self.history.next_chunk_index
                for item in proposed
            )
            and set(self.corrected_chunks).issubset(proposed),
            "pending rollout proposed_target_chunks are invalid",
        )
        object.__setattr__(self, "proposed_target_chunks", proposed)
        _require(
            self.prepared_quality.measurement_id == f"2b-{self.route_key}"
            and self.prepared_quality.history_manifest_sha256
            == self.history.manifest_sha256
            and self.prepared_quality.history_head_sha256 == self.history.head_sha256
            and self.prepared_quality.prompt_id == self.history.prompt_id
            and self.prepared_quality.latent_chunk_count == self.history.next_chunk_index,
            "pending rollout prepared quality is bound to another history",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PENDING_ROLLOUT_SCHEMA,
            "route_key": self.route_key,
            "route_id": self.route_id,
            "corrected_chunks": list(self.corrected_chunks),
            "proposed_target_chunks": list(self.proposed_target_chunks or ()),
            "history": self.history.to_dict(),
            "prepared_quality": self.prepared_quality.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PendingSequentialRollout:
        _require(isinstance(value, Mapping), "pending rollout must be an object")
        corrected_chunks = value.get("corrected_chunks")
        proposed_target_chunks = value.get("proposed_target_chunks")
        _require(
            set(value)
            == {
                "schema",
                "route_key",
                "route_id",
                "corrected_chunks",
                "proposed_target_chunks",
                "history",
                "prepared_quality",
            }
            and value.get("schema") == PENDING_ROLLOUT_SCHEMA
            and isinstance(corrected_chunks, list)
            and isinstance(proposed_target_chunks, list)
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in corrected_chunks
            )
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in proposed_target_chunks
            ),
            "pending rollout fields are invalid",
        )
        try:
            history = LatentHistory.from_dict(value["history"])
        except CorrectionStateError as error:
            raise CampaignError(f"pending rollout history is invalid: {error}") from error
        return cls(
            route_key=value["route_key"],
            route_id=value["route_id"],
            corrected_chunks=tuple(corrected_chunks),
            proposed_target_chunks=tuple(proposed_target_chunks),
            history=history,
            prepared_quality=PreparedFullQuality.from_dict(
                value["prepared_quality"]
            ),
        )


_RPC_COUNT_FIELDS = (
    "attempted",
    "backend_returned",
    "completed",
    "logical_completed",
)


def _freeze_rpc_counts(value: Mapping[str, int], *, label: str) -> tuple[int, ...]:
    _require(set(value) == set(_RPC_COUNT_FIELDS), f"{label} fields are invalid")
    result = tuple(value[field] for field in _RPC_COUNT_FIELDS)
    _require(
        all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in result
        )
        and result[2] <= result[1] <= result[0]
        and result[3] <= result[2],
        f"{label} values are invalid",
    )
    return result


def _thaw_rpc_counts(value: tuple[int, ...]) -> dict[str, int]:
    _require(len(value) == len(_RPC_COUNT_FIELDS), "frozen RPC counts are invalid")
    return dict(zip(_RPC_COUNT_FIELDS, value, strict=True))


@dataclass(frozen=True)
class PendingStrategyRow:
    """One deterministic raw-row intent whose route is waiting for quality."""

    event_id: str
    split: str
    prompt_id: str
    seed: int
    budget: int
    strategy: str
    route_repeat: int | None
    rollout: PendingSequentialRollout
    target_rpc_counts: tuple[int, ...]
    route_materialization_rpc_counts: tuple[int, ...]
    selection_search_rollout_count: int = 0
    selection_search_rpc_counts: tuple[int, ...] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        _require(
            isinstance(self.event_id, str)
            and _SHA256_RE.fullmatch(self.event_id) is not None,
            "pending strategy event_id is invalid",
        )
        _require(self.split in {"smoke", "evaluation"}, "pending split is invalid")
        _require(
            isinstance(self.prompt_id, str)
            and _IDENTIFIER_RE.fullmatch(self.prompt_id) is not None
            and self.prompt_id == self.rollout.history.prompt_id,
            "pending prompt_id is invalid",
        )
        _require(
            isinstance(self.seed, int)
            and not isinstance(self.seed, bool)
            and self.seed >= 0
            and self.seed == self.rollout.history.trajectory_seed,
            "pending seed is invalid",
        )
        _require(
            isinstance(self.budget, int)
            and not isinstance(self.budget, bool)
            and self.budget
            == (
                len(self.rollout.proposed_target_chunks or ())
                if self.strategy == "online_lookahead_coverage_boundary_gated"
                else len(self.rollout.corrected_chunks)
            ),
            "pending budget differs from the strategy Target budget",
        )
        _require(
            self.strategy
            in {
                "random",
                "uniform",
                "online_rank_guided",
                "online_coverage_guided",
                "online_lookahead_coverage_guided",
                "online_lookahead_coverage_boundary_gated",
            },
            "deferred quality supports only deployed non-oracle strategies",
        )
        _require(
            (self.strategy == "random")
            == (
                isinstance(self.route_repeat, int)
                and not isinstance(self.route_repeat, bool)
                and self.route_repeat >= 0
            ),
            "pending route_repeat is inconsistent with its strategy",
        )
        for label, counts in (
            ("target_rpc_counts", self.target_rpc_counts),
            (
                "route_materialization_rpc_counts",
                self.route_materialization_rpc_counts,
            ),
            ("selection_search_rpc_counts", self.selection_search_rpc_counts),
        ):
            _freeze_rpc_counts(_thaw_rpc_counts(counts), label=label)
        _require(
            self.selection_search_rollout_count == 0
            and self.selection_search_rpc_counts == (0, 0, 0, 0),
            "deferred non-oracle route cannot carry selection-search work",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "split": self.split,
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "budget": self.budget,
            "strategy": self.strategy,
            "route_repeat": self.route_repeat,
            "rollout": self.rollout.to_dict(),
            "target_rpc_counts": _thaw_rpc_counts(self.target_rpc_counts),
            "route_materialization_rpc_counts": _thaw_rpc_counts(
                self.route_materialization_rpc_counts
            ),
            "selection_search_rollout_count": self.selection_search_rollout_count,
            "selection_search_rpc_counts": _thaw_rpc_counts(
                self.selection_search_rpc_counts
            ),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PendingStrategyRow:
        _require(isinstance(value, Mapping), "pending strategy row must be an object")
        expected = {
            "event_id",
            "split",
            "prompt_id",
            "seed",
            "budget",
            "strategy",
            "route_repeat",
            "rollout",
            "target_rpc_counts",
            "route_materialization_rpc_counts",
            "selection_search_rollout_count",
            "selection_search_rpc_counts",
        }
        _require(set(value) == expected, "pending strategy row fields are invalid")
        counts: dict[str, tuple[int, ...]] = {}
        for field in (
            "target_rpc_counts",
            "route_materialization_rpc_counts",
            "selection_search_rpc_counts",
        ):
            raw_counts = value[field]
            _require(
                isinstance(raw_counts, Mapping),
                f"pending strategy {field} must be an object",
            )
            counts[field] = _freeze_rpc_counts(raw_counts, label=field)
        selection_search_rollout_count = value["selection_search_rollout_count"]
        _require(
            isinstance(selection_search_rollout_count, int)
            and not isinstance(selection_search_rollout_count, bool)
            and selection_search_rollout_count >= 0,
            "pending selection_search_rollout_count is invalid",
        )
        return cls(
            event_id=value["event_id"],
            split=value["split"],
            prompt_id=value["prompt_id"],
            seed=value["seed"],
            budget=value["budget"],
            strategy=value["strategy"],
            route_repeat=value["route_repeat"],
            rollout=PendingSequentialRollout.from_dict(value["rollout"]),
            target_rpc_counts=counts["target_rpc_counts"],
            route_materialization_rpc_counts=counts[
                "route_materialization_rpc_counts"
            ],
            selection_search_rollout_count=selection_search_rollout_count,
            selection_search_rpc_counts=counts["selection_search_rpc_counts"],
        )


@dataclass(frozen=True)
class PendingMultiBudgetPlan:
    """Content-addressed generation-complete plan for post-runtime scoring."""

    manifest_path: str
    protocol_id: str
    split: str
    budgets: tuple[int, ...]
    random_repeats: int
    entries: tuple[PendingStrategyRow, ...]

    def __post_init__(self) -> None:
        _safe_relative_path(self.manifest_path, label="pending plan manifest_path")
        _require(
            self.manifest_path == f"state/pending/insight2b-{self.split}.json"
            and isinstance(self.protocol_id, str)
            and _IDENTIFIER_RE.fullmatch(self.protocol_id) is not None
            and self.split in {"smoke", "evaluation"}
            and bool(self.budgets)
            and self.budgets == tuple(dict.fromkeys(self.budgets))
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 1
                for item in self.budgets
            )
            and isinstance(self.random_repeats, int)
            and not isinstance(self.random_repeats, bool)
            and self.random_repeats >= 0
            and bool(self.entries)
            and all(isinstance(entry, PendingStrategyRow) for entry in self.entries),
            "pending plan fields are invalid",
        )
        _require(
            len({entry.event_id for entry in self.entries}) == len(self.entries)
            and all(
                entry.split == self.split
                and entry.budget in self.budgets
                and entry.event_id
                == _digest_json(
                    [
                        self.protocol_id,
                        "2b",
                        entry.split,
                        entry.prompt_id,
                        entry.seed,
                        entry.budget,
                        entry.strategy,
                        entry.route_repeat,
                    ]
                )
                for entry in self.entries
            ),
            "pending plan entries are duplicated or have the wrong split",
        )
        for entry in self.entries:
            if entry.strategy == "online_rank_guided":
                expected_route_id = f"online-erank-drop-k{entry.budget}"
            elif entry.strategy == "online_coverage_guided":
                expected_route_id = f"online-coverage-erank-drop-k{entry.budget}"
            elif entry.strategy == "online_lookahead_coverage_guided":
                expected_route_id = (
                    f"online-lookahead-coverage-erank-drop-k{entry.budget}"
                )
            elif entry.strategy == "online_lookahead_coverage_boundary_gated":
                expected_route_id = (
                    "online-lookahead-coverage-boundary-gated-erank-drop-"
                    f"k{entry.budget}"
                )
            else:
                expected_route_id = "fixed-" + (
                    "none"
                    if not entry.rollout.corrected_chunks
                    else "-".join(map(str, entry.rollout.corrected_chunks))
                )
            _require(
                entry.rollout.route_id == expected_route_id
                and entry.rollout.route_key
                == _digest_json(
                    [
                        self.protocol_id,
                        entry.prompt_id,
                        entry.seed,
                        expected_route_id,
                    ]
                )[:48],
                "pending plan rollout identity is invalid",
            )
        observed_strategies = {entry.strategy for entry in self.entries}
        supported_strategy_sets = {
            frozenset({"random", "uniform", "online_rank_guided"}),
            frozenset(
                {"random", "uniform", "online_rank_guided", "online_coverage_guided"}
            ),
            frozenset(
                {
                    "random",
                    "uniform",
                    "online_rank_guided",
                    "online_coverage_guided",
                    "online_lookahead_coverage_guided",
                }
            ),
            frozenset(
                {
                    "uniform",
                    "online_lookahead_coverage_guided",
                    "online_lookahead_coverage_boundary_gated",
                }
            ),
        }
        _require(
            frozenset(observed_strategies) in supported_strategy_sets
            and (("random" in observed_strategies) == (self.random_repeats >= 1))
            and (self.random_repeats == 0 or "random" in observed_strategies),
            "pending plan strategy profile or random repeat count is invalid",
        )
        expected_variants = {
            *(("random", repeat) for repeat in range(self.random_repeats)),
            *((strategy, None) for strategy in observed_strategies if strategy != "random"),
        }
        prompt_seeds = {(entry.prompt_id, entry.seed) for entry in self.entries}
        grouped_variants: dict[tuple[str, int, int], set[tuple[str, int | None]]] = {}
        for entry in self.entries:
            grouped_variants.setdefault(
                (entry.prompt_id, entry.seed, entry.budget), set()
            ).add((entry.strategy, entry.route_repeat))
        _require(
            set(grouped_variants)
            == {
                (prompt_id, seed, budget)
                for prompt_id, seed in prompt_seeds
                for budget in self.budgets
            }
            and all(
                variants == expected_variants
                for variants in grouped_variants.values()
            ),
            "pending plan does not exactly cover every strategy and budget",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PENDING_MULTI_BUDGET_PLAN_SCHEMA,
            "protocol_id": self.protocol_id,
            "split": self.split,
            "budgets": list(self.budgets),
            "random_repeats": self.random_repeats,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(
        cls, value: Any, *, manifest_path: str
    ) -> PendingMultiBudgetPlan:
        _require(isinstance(value, Mapping), "pending multi-budget plan must be an object")
        expected = {
            "schema",
            "protocol_id",
            "split",
            "budgets",
            "random_repeats",
            "entries",
        }
        budgets = value.get("budgets")
        entries = value.get("entries")
        _require(
            set(value) == expected
            and value.get("schema") == PENDING_MULTI_BUDGET_PLAN_SCHEMA
            and isinstance(budgets, list)
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in budgets
            )
            and isinstance(entries, list),
            "pending multi-budget plan fields are invalid",
        )
        return cls(
            manifest_path=manifest_path,
            protocol_id=value["protocol_id"],
            split=value["split"],
            budgets=tuple(budgets),
            random_repeats=value["random_repeats"],
            entries=tuple(PendingStrategyRow.from_dict(item) for item in entries),
        )


def _context_dict(context: GenerationContext) -> dict[str, Any]:
    return {
        "role": context.role,
        "model_id": context.model_id,
        "model_artifact_sha256": context.model_artifact_sha256,
        "history_head_sha256": context.history_head_sha256,
        "next_chunk_index": context.next_chunk_index,
        "noise": context.noise.to_dict(),
        "kv": context.kv.to_dict(),
    }


def _context_from_dict(value: Any) -> GenerationContext:
    _require(isinstance(value, Mapping), "checkpoint context must be an object")
    return GenerationContext(
        role=value["role"],
        model_id=value["model_id"],
        model_artifact_sha256=value["model_artifact_sha256"],
        history_head_sha256=value["history_head_sha256"],
        next_chunk_index=value["next_chunk_index"],
        noise=InitialNoiseDescriptor.from_dict(value["noise"]),
        kv=KVCacheDescriptor.from_dict(value["kv"]),
    )


def _generated_dict(value: GeneratedChunk) -> dict[str, Any]:
    return {
        "context": _context_dict(value.context),
        "segment": value.segment.to_dict(),
        "effective_rank": value.effective_rank,
        "hidden_state_path": value.hidden_state_path,
        "hidden_state_sha256": value.hidden_state_sha256,
        "diagnostics": dict(value.diagnostics or {}),
        "target_attempt_id": value.target_attempt_id,
        "target_logical_call_id": value.target_logical_call_id,
    }


def _generated_from_dict(value: Any) -> GeneratedChunk:
    _require(isinstance(value, Mapping), "checkpoint generation must be an object")
    return GeneratedChunk(
        context=_context_from_dict(value["context"]),
        segment=LatentSegment.from_dict(value["segment"]),
        effective_rank=value["effective_rank"],
        hidden_state_path=value["hidden_state_path"],
        hidden_state_sha256=value["hidden_state_sha256"],
        diagnostics=value.get("diagnostics", {}),
        target_attempt_id=value.get("target_attempt_id"),
        target_logical_call_id=value.get("target_logical_call_id"),
    )


def _boundary_velocity_witness(value: GeneratedChunk, *, label: str) -> Mapping[str, Any]:
    diagnostics = value.diagnostics
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
        item = witness.get(field)
        _require(
            isinstance(item, int) and not isinstance(item, bool) and item >= 2,
            f"{label} boundary velocity witness {field} is invalid",
        )
    for field in (
        "previous_velocity_norm",
        "boundary_velocity_norm",
        "candidate_velocity_norm",
    ):
        item = witness.get(field)
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and float(item) >= 0.0
            and round(float(item), BOUNDARY_VELOCITY_ROUND_DECIMALS)
            == float(item),
            f"{label} boundary velocity witness {field} is invalid",
        )
    for field in ("incoming_discontinuity", "outgoing_discontinuity", "score"):
        item = witness.get(field)
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and 0.0 <= float(item) <= 1.0
            and round(float(item), BOUNDARY_VELOCITY_ROUND_DECIMALS)
            == float(item),
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


def validate_prompt_partition(calibration_prompts, evaluation_prompts, *,
                              calibration_replay_development=False):
    """Keep held-out splits strict; disclose an intentional complete C10 replay."""
    _require(type(calibration_replay_development) is bool, "invalid development replay mode")
    calibration_prompts=tuple(calibration_prompts);evaluation_prompts=tuple(evaluation_prompts)
    _require(bool(calibration_prompts), "calibration prompts are empty")
    _require(bool(evaluation_prompts), "evaluation prompts are empty")
    _require(all(p.split=='calibration' for p in calibration_prompts), "calibration prompt has the wrong split")
    _require(all(p.split=='evaluation' for p in evaluation_prompts), "evaluation prompt has the wrong split")
    calibration_ids={p.prompt_id for p in calibration_prompts}
    evaluation_ids={p.prompt_id for p in evaluation_prompts}
    _require(len(calibration_ids)==len(calibration_prompts) and len(evaluation_ids)==len(evaluation_prompts),
             "prompt IDs must be unique within each split")
    calibration_texts={' '.join(p.prompt.casefold().split()) for p in calibration_prompts}
    evaluation_texts={' '.join(p.prompt.casefold().split()) for p in evaluation_prompts}
    if calibration_replay_development:
        from .paper_population import enabled
        _require(not enabled(), "calibration replay is development only, never paper evaluation")
        _require(tuple((p.prompt_id,p.prompt) for p in calibration_prompts)==
                 tuple((p.prompt_id,p.prompt) for p in evaluation_prompts)
                 and len(calibration_texts)==len(calibration_prompts),
                 "development replay requires the complete ordered calibration population")
    else:
        _require(not calibration_ids.intersection(evaluation_ids), "prompt splits overlap by ID")
        _require(not calibration_texts.intersection(evaluation_texts), "prompt splits overlap by text")


class Insight2Campaign:
    """Coordinate calibration, paired 1+1 forks, and sequential multi-budget runs."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        protocol_id: str,
        seeds: Sequence[int],
        calibration_prompts: Sequence[PromptCase],
        evaluation_prompts: Sequence[PromptCase],
        smoke_prompts: Sequence[PromptCase] | None = None,
        draft: ChunkBackendClient,
        target: ChunkBackendClient,
        noise_provider: NoiseProvider,
        evaluator: QualityEvaluator,
        chunk_count: int = 16,
        calibration_replay_development: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _require(self.run_dir.is_dir(), "run_dir must be a directory")
        _require(
            isinstance(protocol_id, str) and _IDENTIFIER_RE.fullmatch(protocol_id) is not None,
            "protocol_id must be a portable identifier",
        )
        self.protocol_id = protocol_id
        _require(
            isinstance(chunk_count, int)
            and not isinstance(chunk_count, bool)
            and chunk_count in SUPPORTED_CHUNK_COUNTS,
            "chunk_count must be one of 16, 30, 60, or 120",
        )
        self.chunk_count = chunk_count
        self.candidate_chunks = tuple(range(1, chunk_count))
        self.seeds = tuple(seeds)
        _require(
            bool(self.seeds)
            and len(self.seeds) == len(set(self.seeds))
            and all(
                isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
                for seed in self.seeds
            ),
            "seeds must be unique non-negative integers",
        )
        self.calibration_prompts = tuple(calibration_prompts)
        self.evaluation_prompts = tuple(evaluation_prompts)
        frozen_smoke_count = SMOKE_PROMPT_COUNT_BY_CHUNK_COUNT[self.chunk_count]
        self.smoke_prompts = (
            tuple(smoke_prompts)
            if smoke_prompts is not None
            else self.evaluation_prompts[:frozen_smoke_count]
        )
        _require(bool(self.calibration_prompts), "calibration prompts are empty")
        _require(bool(self.evaluation_prompts), "evaluation prompts are empty")
        expected_smoke = self.evaluation_prompts[
            : min(frozen_smoke_count, len(self.evaluation_prompts))
        ]
        _require(
            self.smoke_prompts == expected_smoke,
            "smoke prompts must be the frozen first "
            f"{frozen_smoke_count} evaluation prompt(s) for the chunk horizon",
        )
        validate_prompt_partition(self.calibration_prompts,self.evaluation_prompts,
            calibration_replay_development=calibration_replay_development)
        self.calibration_replay_development=calibration_replay_development

        _require(draft.role == "draft", "Draft client role is invalid")
        _require(target.role == "target", "Target client role is invalid")
        _require(
            _SHA256_RE.fullmatch(draft.model_artifact_sha256) is not None,
            "Draft artifact SHA is invalid",
        )
        _require(
            _SHA256_RE.fullmatch(target.model_artifact_sha256) is not None,
            "Target artifact SHA is invalid",
        )
        _require(noise_provider.namespace == NOISE_NAMESPACE, "noise namespace is not frozen")
        self.draft = draft
        self.target = target
        self.noise_provider = noise_provider
        self.evaluator = evaluator

        manifest = {
            "schema": CAMPAIGN_SCHEMA,
            "protocol_id": self.protocol_id,
            "chunk_count": self.chunk_count,
            "seeds": list(self.seeds),
            "noise_namespace": noise_provider.namespace,
            "models": {
                "draft": {
                    "id": draft.model_id,
                    "artifact_sha256": draft.model_artifact_sha256,
                },
                "target": {
                    "id": target.model_id,
                    "artifact_sha256": target.model_artifact_sha256,
                },
            },
            "calibration_prompts": [asdict(item) for item in self.calibration_prompts],
            "evaluation_prompts": [asdict(item) for item in self.evaluation_prompts],
            "smoke_prompts": [asdict(item) for item in self.smoke_prompts],
        }
        if calibration_replay_development:
            manifest.update(population_role='calibration_replay_development',development_only=True,
                            held_out=False,may_replace_paper128_results=False)
        _write_immutable_json(self.run_dir / "campaign.json", manifest)
        self.raw_2a = AppendOnlyJsonl(self.run_dir / "raw" / "insight2a.jsonl")
        self.raw_2b = AppendOnlyJsonl(self.run_dir / "raw" / "insight2b.jsonl")
        self.target_rpc_events = AppendOnlyJsonl(
            self.run_dir / "logs" / "target-rpc-events.jsonl"
        )
        self._active_target_rpc_scope: str | None = None
        self._active_target_rpc_phase: str | None = None
        # The worker ledger can contain hundreds of thousands of lines in the
        # multi-budget campaign.  Validate each appended line exactly once and
        # retain only per-attempt state for O(total ledger size) accounting.
        self._target_lifecycle_offset = 0
        self._target_lifecycle_identity: tuple[int, int] | None = None
        self._target_lifecycle_line_count = 0
        self._target_lifecycle_event_ids: set[str] = set()
        self._target_lifecycle_attempts: dict[str, dict[str, Any]] = {}
        self._target_lifecycle_scope_attempt_ids: dict[
            tuple[str, str | None], set[str]
        ] = {}

    @property
    def condition_hashes(self) -> dict[str, str]:
        return {
            item.prompt_id: _digest_text(item.prompt)
            for item in (*self.calibration_prompts, *self.evaluation_prompts)
        }

    def _case_key(self, prompt: PromptCase, seed: int) -> str:
        return f"{prompt.prompt_id}-{seed}-{_digest_text(prompt.prompt)[:12]}"

    def _trajectory_id(self, *parts: Any) -> str:
        digest = _digest_json([self.protocol_id, *parts])
        return f"i2-{digest[:40]}"

    @contextmanager
    def _target_rpc_scope(
        self, scope_id: str, *, phase: str = "route_materialization"
    ) -> Iterator[None]:
        """Persistently attribute Target RPCs to one deterministic scope."""

        _require(
            isinstance(scope_id, str) and _SHA256_RE.fullmatch(scope_id) is not None,
            "Target RPC scope must be a lowercase SHA-256 event ID",
        )
        _require(
            phase in TARGET_RPC_PHASES - {"unscoped"},
            "scoped Target RPC phase is invalid",
        )
        _require(self._active_target_rpc_scope is None, "Target RPC scopes may not nest")
        self._active_target_rpc_scope = scope_id
        self._active_target_rpc_phase = phase
        try:
            yield
        finally:
            self._active_target_rpc_scope = None
            self._active_target_rpc_phase = None

    def _target_rpc_count(self, scope_id: str) -> int:
        return sum(
            1
            for record in self.target_rpc_events.records()
            if record.get("scope_id") == scope_id
        )

    def _target_rpc_lifecycle_counts(
        self, scope_id: str, *, phase: str | None = None
    ) -> dict[str, int]:
        """Index worker-fsynced lifecycle stages for one deterministic result row."""

        path = self.run_dir / "logs" / "target-rpc-lifecycle.jsonl"
        if not path.exists():
            _require(
                not bool(getattr(self.target, "requires_target_rpc_lifecycle", False)),
                "formal Target worker omitted its RPC lifecycle log",
            )
            successful_events = [
                record
                for record in self.target_rpc_events.records()
                if record.get("scope_id") == scope_id
                and (phase is None or record.get("phase") == phase)
            ]
            successful = len(successful_events)
            return {
                "attempted": successful,
                "backend_returned": successful,
                "completed": successful,
                "logical_completed": len(
                    {
                        record.get("logical_call_id")
                        for record in successful_events
                        if isinstance(record.get("logical_call_id"), str)
                    }
                ),
            }
        _require(
            phase is None or phase in TARGET_RPC_PHASES - {"unscoped"},
            "Target RPC lifecycle count phase is invalid",
        )
        try:
            stat = path.stat()
        except OSError as error:
            raise CampaignError(f"cannot stat Target RPC lifecycle log: {error}") from error
        _require(path.is_file() and not path.is_symlink(), "Target RPC lifecycle log is not regular")
        identity = (int(stat.st_dev), int(stat.st_ino))
        if self._target_lifecycle_identity is None:
            self._target_lifecycle_identity = identity
        _require(
            self._target_lifecycle_identity == identity,
            "Target RPC lifecycle log was replaced during the campaign",
        )
        _require(
            stat.st_size >= self._target_lifecycle_offset,
            "Target RPC lifecycle log was truncated during the campaign",
        )
        try:
            with path.open("rb") as handle:
                handle.seek(self._target_lifecycle_offset)
                appended = handle.read()
        except OSError as error:
            raise CampaignError(f"cannot incrementally read Target RPC lifecycle log: {error}") from error
        if appended:
            _require(
                appended.endswith(b"\n"),
                "Target RPC lifecycle log ends with an incomplete record",
            )
            try:
                lines = appended.decode("utf-8").splitlines()
            except UnicodeError as error:
                raise CampaignError(
                    f"Target RPC lifecycle append is not UTF-8: {error}"
                ) from error
            for raw in lines:
                self._target_lifecycle_line_count += 1
                _require(
                    bool(raw),
                    f"Target RPC lifecycle line {self._target_lifecycle_line_count} is blank",
                )
                try:
                    record = json.loads(raw, object_pairs_hook=_duplicate_checked_object)
                except (json.JSONDecodeError, UnicodeError) as error:
                    raise CampaignError(
                        "Target RPC lifecycle line "
                        f"{self._target_lifecycle_line_count} is invalid JSON: {error}"
                    ) from error
                _validate_target_rpc_lifecycle_record(
                    record,
                    event_ids=self._target_lifecycle_event_ids,
                    attempts=self._target_lifecycle_attempts,
                    line_number=self._target_lifecycle_line_count,
                )
                attempt_id = str(record["attempt_id"])
                record_scope = record["scope_id"]
                record_phase = str(record["phase"])
                if isinstance(record_scope, str):
                    for key in (
                        (record_scope, None),
                        (record_scope, record_phase),
                    ):
                        self._target_lifecycle_scope_attempt_ids.setdefault(
                            key, set()
                        ).add(attempt_id)
            self._target_lifecycle_offset += len(appended)
        selected_ids = self._target_lifecycle_scope_attempt_ids.get(
            (scope_id, phase), set()
        )
        selected = [
            self._target_lifecycle_attempts[attempt_id]
            for attempt_id in selected_ids
        ]
        return {
            "attempted": len(selected),
            "backend_returned": sum(
                1 for state in selected if "backend_returned" in state["statuses"]
            ),
            "completed": sum(
                1 for state in selected if "completed" in state["statuses"]
            ),
            "logical_completed": len(
                {
                    state["binding"]["logical_call_id"]
                    for state in selected
                    if "completed" in state["statuses"]
                }
            ),
        }

    def _fresh_output_path(self, stem: str, suffix: str = ".safetensors") -> str:
        _safe_relative_path(stem, label="output stem")
        index = 0
        while True:
            candidate = f"{stem}/attempt-{index:04d}{suffix}"
            if not (self.run_dir / candidate).exists():
                return candidate
            index += 1

    def _materialize_noise(
        self, *, prompt: PromptCase, seed: int, chunk_index: int
    ) -> NoiseMaterialization:
        value = self.noise_provider.materialize(
            prompt_id=prompt.prompt_id,
            trajectory_seed=seed,
            chunk_index=chunk_index,
        )
        _require(isinstance(value, NoiseMaterialization), "noise provider returned the wrong type")
        descriptor = value.descriptor
        _require(descriptor.namespace == NOISE_NAMESPACE, "noise descriptor namespace drifted")
        _require(descriptor.prompt_id == prompt.prompt_id, "noise descriptor prompt drifted")
        _require(descriptor.trajectory_seed == seed, "noise descriptor seed drifted")
        _require(descriptor.chunk_index == chunk_index, "noise descriptor chunk drifted")
        noise_path = self.run_dir / value.tensor_path
        try:
            resolved_noise = noise_path.resolve(strict=True)
            resolved_noise.relative_to(self.run_dir)
        except (OSError, ValueError) as error:
            raise CampaignError("noise tensor path escapes or is unavailable") from error
        _require(noise_path.is_file() and not noise_path.is_symlink(), "noise path is not a file")
        actual_sha256, actual_nbytes = _sha256_file(noise_path)
        _require(
            actual_sha256 == descriptor.tensor_sha256
            and actual_nbytes == descriptor.tensor_nbytes,
            "noise tensor bytes do not match their immutable descriptor",
        )
        return value

    def _generate(
        self,
        *,
        backend: ChunkBackendClient,
        prompt: PromptCase,
        history: LatentHistory,
        noise: NoiseMaterialization,
        output_stem: str,
        capture_hidden: bool,
        kv_branch_action: str | None = None,
    ) -> GeneratedChunk:
        _require(kv_branch_action is None or backend.role == "draft",
                 "KV branch actions belong only to Draft")
        relative = self._fresh_output_path(output_stem)
        target_logical_call_id = (
            self._target_logical_call_id(
                backend=backend,
                prompt=prompt,
                history=history,
                noise=noise,
            )
            if backend.role == "target"
            else None
        )
        request = ChunkGenerationRequest(
            prompt=prompt,
            history=history,
            noise=noise,
            output_latent_path=relative,
            capture_hidden=capture_hidden,
            kv_branch_action=kv_branch_action,
            target_logical_call_id=target_logical_call_id,
            target_call_scope_id=(
                self._active_target_rpc_scope if backend.role == "target" else None
            ),
            target_call_phase=(
                (self._active_target_rpc_phase or "unscoped")
                if backend.role == "target"
                else None
            ),
        )
        result = backend.generate(request)
        _require(isinstance(result, GeneratedChunk), "backend returned the wrong result type")
        _require(result.context.role == backend.role, "backend context role mismatch")
        _require(result.context.model_id == backend.model_id, "backend context model mismatch")
        _require(
            result.context.model_artifact_sha256 == backend.model_artifact_sha256,
            "backend context artifact mismatch",
        )
        _require(
            result.context.noise.descriptor_sha256 == noise.descriptor.descriptor_sha256,
            "backend did not use the requested initial noise",
        )
        result.segment.verify_context(history, result.context)
        _require(result.segment.latent_path == relative, "backend wrote an unexpected latent path")
        result.segment.verify_file(self.run_dir)
        if backend.role == "draft":
            rank = result.effective_rank
            diagnostics = result.diagnostics
            _require(
                isinstance(rank, (int, float))
                and not isinstance(rank, bool)
                and math.isfinite(float(rank))
                and float(rank) >= 1.0
                and isinstance(diagnostics, Mapping),
                "Draft chunk has no valid effective-rank evidence",
            )
            try:
                validate_effective_rank_evidence(
                    diagnostics.get("rank_evidence"),
                    expected_rank=float(rank),
                    expected_eigenvalue_count=256,
                )
            except AdapterError as error:
                raise CampaignError(f"Draft rank evidence is invalid: {error}") from error
        if backend.role == "draft" and capture_hidden:
            _require(
                result.effective_rank is not None
                and math.isfinite(float(result.effective_rank))
                and float(result.effective_rank) >= 1.0,
                "captured Draft chunk has no valid effective rank",
            )
            _require(
                isinstance(result.hidden_state_path, str)
                and isinstance(result.hidden_state_sha256, str)
                and _SHA256_RE.fullmatch(result.hidden_state_sha256) is not None,
                "captured Draft chunk has no hidden-state evidence",
            )
            _safe_relative_path(result.hidden_state_path, label="hidden_state_path")
            _verify_run_file(
                self.run_dir,
                result.hidden_state_path,
                result.hidden_state_sha256,
                label="hidden_state_path",
            )
        if backend.role == "target":
            if bool(getattr(backend, "requires_target_rpc_lifecycle", False)):
                _require(
                    isinstance(result.target_attempt_id, str)
                    and _SHA256_RE.fullmatch(result.target_attempt_id) is not None,
                    "formal Target result has no lifecycle attempt ID",
                )
                _require(
                    result.target_logical_call_id == target_logical_call_id,
                    "formal Target result logical call ID drifted",
                )
            diagnostics_path: str | None = None
            diagnostics_sha256: str | None = None
            if result.target_attempt_id is not None:
                diagnostics_path, diagnostics_sha256 = (
                    self._persist_target_rpc_diagnostics(
                        backend=backend,
                        prompt=prompt,
                        history=history,
                        noise=noise,
                        result=result,
                        logical_call_id=target_logical_call_id,
                    )
                )
            target_event = {
                "schema": TARGET_RPC_SCHEMA,
                "attempt_id": result.target_attempt_id,
                "logical_call_id": target_logical_call_id,
                "scope_id": self._active_target_rpc_scope,
                "phase": self._active_target_rpc_phase or "unscoped",
                "chunk_index": result.segment.chunk_index,
                "latent_path": result.segment.latent_path,
                "latent_sha256": result.segment.latent_sha256,
                "segment_sha256": result.segment.segment_sha256,
                "history_head_sha256": result.context.history_head_sha256,
                "kv_cache_sha256": result.context.kv.cache_sha256,
                "noise_descriptor_sha256": result.context.noise.descriptor_sha256,
                "diagnostics_path": diagnostics_path,
                "diagnostics_sha256": diagnostics_sha256,
            }
            target_event["event_id"] = _digest_json(target_event)
            self.target_rpc_events.append(target_event)
        return result

    def _persist_target_rpc_diagnostics(
        self,
        *,
        backend: ChunkBackendClient,
        prompt: PromptCase,
        history: LatentHistory,
        noise: NoiseMaterialization,
        result: GeneratedChunk,
        logical_call_id: str | None,
    ) -> tuple[str, str]:
        """Persist the complete, replay-verifiable evidence for one Target RPC."""

        attempt_id = result.target_attempt_id
        _require(
            isinstance(attempt_id, str) and _SHA256_RE.fullmatch(attempt_id) is not None,
            "Target diagnostics require a valid lifecycle attempt ID",
        )
        _require(
            isinstance(logical_call_id, str)
            and _SHA256_RE.fullmatch(logical_call_id) is not None,
            "Target diagnostics require a valid logical call ID",
        )
        diagnostics = result.diagnostics
        _require(isinstance(diagnostics, Mapping), "Target result omitted backend diagnostics")
        _require(
            TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS <= set(diagnostics),
            "Target result omitted required backend diagnostics",
        )
        backend_diagnostics = {
            key: diagnostics[key] for key in sorted(TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS)
        }
        if "native_correction_state" in diagnostics:
            from .krea_chunk_backend import validate_native_correction_state
            validate_native_correction_state(
                diagnostics["native_correction_state"], chunk_index=result.segment.chunk_index
            )
            backend_diagnostics["native_correction_state"] = dict(diagnostics["native_correction_state"])
        kv_witness_sha256 = backend_diagnostics["kv_witness_sha256"]
        _require(
            isinstance(kv_witness_sha256, str)
            and _SHA256_RE.fullmatch(kv_witness_sha256) is not None,
            "Target backend KV witness digest is invalid",
        )
        kv_witness = backend_diagnostics["kv_witness"]
        try:
            validate_sampled_kv_cache_witness(
                kv_witness,
                expected_layer_count=40,
                expected_self_shape=(1, 9360, 40, 128),
                expected_cross_shape=(1, 512, 40, 128),
                expected_dtype="torch.bfloat16",
                expected_self_used_tokens=(
                    0 if result.segment.chunk_index == 0 else 4680
                ),
                expected_cross_initialized=result.segment.chunk_index > 0,
            )
        except AdapterError as error:
            raise CampaignError(f"Target backend KV witness is invalid: {error}") from error
        _require(
            _digest_json(kv_witness) == kv_witness_sha256,
            "Target backend KV witness digest does not bind its witness",
        )
        _require(
            result.context.kv.cache_sha256 == kv_witness_sha256,
            "Target backend KV witness differs from its generation context",
        )
        _require(
            backend_diagnostics["explicit_initial_noise_verified"] is True,
            "Target backend did not verify explicit initial noise",
        )
        full_trace = backend_diagnostics["forward_trace"]
        denoising_trace = backend_diagnostics["denoising_forward_trace"]
        schedule = backend_diagnostics["denoising_schedule"]
        _require(isinstance(full_trace, list), "Target forward trace must be a list")
        _require(isinstance(denoising_trace, list), "Target denoising trace must be a list")
        _require(isinstance(schedule, list), "Target denoising schedule must be a list")
        try:
            from .krea_timestep_adapter import (UPSTREAM_TIMESTEP_POLICY, target_timestep_policy,
                                               target_four_step_schedule)
            chunk_index = result.segment.chunk_index
            if target_timestep_policy() == UPSTREAM_TIMESTEP_POLICY:
                # The upstream four-update policy includes its terminal zero
                # step, including at the canonical root. Validate that exact
                # opt-in schedule and native context trace without relabelling
                # it as the historical uniform comparison policy.
                from .krea_chunk_backend import validate_target_forward_trace
                expected_denoising = validate_target_forward_trace(
                    full_trace, schedule, history_frames=3 * chunk_index,
                    expected_schedule=target_four_step_schedule(),
                )
            else:
                from .sf_target_two_step import (MAIN_TABLE_FOUR_STEP_SCHEDULE, SCHEDULE, burst_schedules,
                                                 early_native_burst_count, enabled, validate_correction_trace)
                expected_schedule = MAIN_TABLE_FOUR_STEP_SCHEDULE
                if enabled() and chunk_index > 0:
                    allowed = burst_schedules() or (SCHEDULE,)
                    if early_native_burst_count():
                        allowed = (*allowed, MAIN_TABLE_FOUR_STEP_SCHEDULE)
                    expected_schedule = tuple(schedule)
                    _require(expected_schedule in allowed, "Target schedule is outside the declared burst")
                expected_denoising = validate_correction_trace(
                    full_trace, schedule, chunk_index=chunk_index,
                    expected_nonroot_schedule=expected_schedule,
                )
        except RuntimeError as error:
            raise CampaignError(f"Target forward diagnostics are invalid: {error}") from error
        _require(
            denoising_trace == expected_denoising,
            "Target denoising trace differs from the validated full trace",
        )
        relative = f"provenance/target-rpc-diagnostics/{attempt_id}.json"
        payload = {
            "schema": TARGET_RPC_DIAGNOSTICS_SCHEMA,
            "attempt_id": attempt_id,
            "logical_call_id": logical_call_id,
            "scope_id": self._active_target_rpc_scope,
            "phase": self._active_target_rpc_phase or "unscoped",
            "model_id": backend.model_id,
            "model_artifact_sha256": backend.model_artifact_sha256,
            "prompt_id": prompt.prompt_id,
            "prompt_sha256": _digest_text(prompt.prompt),
            "chunk_index": result.segment.chunk_index,
            "internal_seed": noise.descriptor.derived_seed,
            "history_head_sha256": history.head_sha256,
            "history_latent_paths": [item.latent_path for item in history.segments],
            "noise_descriptor": noise.descriptor.to_dict(),
            "noise_descriptor_sha256": noise.descriptor.descriptor_sha256,
            "initial_noise_path": noise.tensor_path,
            "latent_path": result.segment.latent_path,
            "latent_sha256": result.segment.latent_sha256,
            "segment_sha256": result.segment.segment_sha256,
            "kv_cache_sha256": result.context.kv.cache_sha256,
            "backend_diagnostics": backend_diagnostics,
        }
        path = self.run_dir / relative
        _write_immutable_json(path, payload)
        actual_sha256, _size = _sha256_file(path)
        return relative, actual_sha256

    def _target_logical_call_id(
        self,
        *,
        backend: ChunkBackendClient,
        prompt: PromptCase,
        history: LatentHistory,
        noise: NoiseMaterialization,
    ) -> str:
        """Return the immutable scientific identity of one Target generation."""

        _require(backend.role == "target", "logical Target call ID requires Target backend")
        return _digest_json(
            [
                "target-logical-call-v2",
                self.protocol_id,
                backend.model_id,
                backend.model_artifact_sha256,
                prompt.prompt_id,
                _digest_text(prompt.prompt),
                history.head_sha256,
                history.next_chunk_index,
                noise.descriptor.descriptor_sha256,
            ]
        )

    def _baseline_target_scope_id(
        self, *, prompt: PromptCase, seed: int, phase: str
    ) -> str:
        """Return a stable non-row scope for a baseline Target chunk 0."""

        _require(
            phase in {"calibration_baseline", "evaluation_baseline"},
            "baseline Target scope phase is invalid",
        )
        return self._row_id(
            "target-non-row-v1",
            phase,
            prompt.prompt_id,
            seed,
            0,
        )

    def _checkpoint_path(self, category: str, key: str, chunk_index: int) -> Path:
        return self.run_dir / "state" / category / key / f"chunk-{chunk_index:02d}.json"

    def _write_checkpoint(
        self,
        *,
        path: Path,
        history: LatentHistory,
        generated: GeneratedChunk,
        metadata: Mapping[str, Any],
    ) -> None:
        with timing_session(self.run_dir, "campaign", {
            "action": "checkpoint", "role": generated.context.role,
            "checkpoint_path": path.relative_to(self.run_dir).as_posix(),
            "chunk_index": generated.segment.chunk_index,
            "history_chunk_count": history.next_chunk_index,
            "output_latent_path": generated.segment.latent_path,
        }) as session:
            if session is not None:
                session.note_not_applicable(
                    "snapshot", "Checkpoint stores descriptors; it does not copy actual KV tensors."
                )
            with stage("checkpoint_total"):
                with stage("history_serialize"):
                    serialized_history = history.to_dict()
                with stage("descriptor_snapshot"):
                    serialized_generated = _generated_dict(generated)
                    serialized_metadata = dict(metadata)
                with stage("checkpoint_persist"):
                    _write_immutable_json(path, {
                        "schema": CHECKPOINT_SCHEMA,
                        "history": serialized_history,
                        "generated": serialized_generated,
                        "metadata": serialized_metadata,
                    })

    def _load_checkpoint(self, path: Path) -> tuple[LatentHistory, GeneratedChunk, dict[str, Any]]:
        value = _load_json_object(path)
        _require(value.get("schema") == CHECKPOINT_SCHEMA, f"invalid checkpoint schema: {path}")
        history = LatentHistory.from_dict(value["history"])
        history.verify_files(self.run_dir)
        generated = _generated_from_dict(value["generated"])
        prefix = history.prefix_before(history.next_chunk_index - 1)
        _require(
            generated.segment.to_dict() == history.segments[-1].to_dict(),
            f"checkpoint latest segment mismatch: {path}",
        )
        try:
            generated.segment.verify_context(prefix, generated.context)
        except CorrectionStateError as error:
            raise CampaignError(f"checkpoint segment/context mismatch: {path}: {error}") from error
        hidden_path = generated.hidden_state_path
        hidden_sha256 = generated.hidden_state_sha256
        _require(
            (hidden_path is None and hidden_sha256 is None)
            or (isinstance(hidden_path, str) and isinstance(hidden_sha256, str)),
            f"checkpoint hidden-state evidence is incomplete: {path}",
        )
        if isinstance(hidden_path, str) and isinstance(hidden_sha256, str):
            _verify_run_file(
                self.run_dir,
                hidden_path,
                hidden_sha256,
                label="checkpoint hidden_state_path",
            )
        metadata = value.get("metadata")
        _require(isinstance(metadata, dict), f"checkpoint metadata is invalid: {path}")
        return history, generated, metadata

    def _baseline(self, *, prompt: PromptCase, seed: int) -> BaselineTrajectory:
        category = (
            "calibration-baselines"
            if prompt.split == "calibration"
            else "evaluation-baselines"
        )
        key = self._case_key(prompt, seed)
        history = LatentHistory(
            trajectory_id=self._trajectory_id("baseline", prompt.split, prompt.prompt_id, seed),
            prompt_id=prompt.prompt_id,
            trajectory_seed=seed,
        )
        generated: list[GeneratedChunk] = []
        for chunk_index in range(self.chunk_count):
            checkpoint = self._checkpoint_path(category, key, chunk_index)
            if checkpoint.exists():
                restored, latest, metadata = self._load_checkpoint(checkpoint)
                _require(
                    restored.next_chunk_index == chunk_index + 1,
                    "baseline checkpoint index drift",
                )
                _require(
                    restored.prefix_before(chunk_index).manifest_sha256 == history.manifest_sha256,
                    "baseline checkpoint does not extend the resumed history",
                )
                _require(metadata.get("kind") == "baseline", "baseline checkpoint kind drift")
                history = restored
                generated.append(latest)
                continue
            noise = self._materialize_noise(prompt=prompt, seed=seed, chunk_index=chunk_index)
            backend = self.target if chunk_index == 0 else self.draft
            if backend.role == "target":
                baseline_phase = (
                    "calibration_baseline"
                    if prompt.split == "calibration"
                    else "evaluation_baseline"
                )
                baseline_scope = self._baseline_target_scope_id(
                    prompt=prompt,
                    seed=seed,
                    phase=baseline_phase,
                )
                with self._target_rpc_scope(baseline_scope, phase=baseline_phase):
                    result = self._generate(
                        backend=backend,
                        prompt=prompt,
                        history=history,
                        noise=noise,
                        output_stem=(
                            f"latents/{category}/{key}/chunk-{chunk_index:02d}-{backend.role}"
                        ),
                        capture_hidden=False,
                    )
            else:
                result = self._generate(
                    backend=backend,
                    prompt=prompt,
                    history=history,
                    noise=noise,
                    output_stem=(
                        f"latents/{category}/{key}/chunk-{chunk_index:02d}-{backend.role}"
                    ),
                    capture_hidden=True,
                )
            history = history.append(result.segment)
            self._write_checkpoint(
                path=checkpoint,
                history=history,
                generated=result,
                metadata={"kind": "baseline", "chunk_index": chunk_index},
            )
            generated.append(result)
        _require(
            len(generated) == self.chunk_count,
            f"baseline did not produce {self.chunk_count} chunks",
        )
        _require(generated[0].segment.producer_role == "target", "chunk 0 is not Target")
        _require(
            all(item.segment.producer_role == "draft" for item in generated[1:]),
            "baseline candidate chunk is not Draft",
        )
        return BaselineTrajectory(history=history, generated=tuple(generated))

    def _calibration_row(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        chunk_index: int,
        generated: GeneratedChunk,
        prior_ranks: Sequence[float],
    ) -> dict[str, Any]:
        _require(generated.effective_rank is not None, "calibration rank is missing")
        rank = float(generated.effective_rank)
        row: dict[str, Any] = {
            "prompt_id": prompt.prompt_id,
            "seed": seed,
            "chunk_index": chunk_index,
            "rank": rank,
            "hidden_state_path": generated.hidden_state_path,
            "hidden_state_sha256": generated.hidden_state_sha256,
        }
        if self.chunk_count != 16:
            row["erank_drop_score"] = causal_erank_sudden_drop(
                (*prior_ranks, rank), lookback_chunks=4
            )
        return row

    def _calibrator_from_rows(
        self, *, calibration_id: str, rows: Sequence[Mapping[str, Any]]
    ) -> RankRiskCalibrator | SuddenDropRiskCalibrator:
        if self.chunk_count == 16:
            return RankRiskCalibrator(
                calibration_id=calibration_id,
                ranks=tuple(float(item["rank"]) for item in rows),
            )
        return SuddenDropRiskCalibrator(
            calibration_id=calibration_id,
            drop_scores=tuple(float(item["erank_drop_score"]) for item in rows),
        )

    def _risk_from_rank_history(
        self,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        ranks: Sequence[float],
    ) -> tuple[float, float | None]:
        _require(bool(ranks), "risk calculation requires the current Draft rank")
        if isinstance(calibrator, SuddenDropRiskCalibrator):
            drop = causal_erank_sudden_drop(ranks, lookback_chunks=4)
            return calibrator.risk(drop), drop
        return calibrator.risk(float(ranks[-1])), None

    def run_calibration(self) -> RankRiskCalibrator | SuddenDropRiskCalibrator:
        """Generate/load only C10 baselines and freeze their pooled Draft ranks."""

        destination = self.run_dir / "calibration" / "ranks.json"
        expected_pairs = [
            (prompt.prompt_id, seed)
            for prompt in self.calibration_prompts
            for seed in self.seeds
        ]
        if destination.exists():
            value = _load_json_object(destination)
            _require(
                set(value)
                == {
                    "schema",
                    "protocol_id",
                    "calibration_id",
                    "prompt_seed_pairs",
                    "rows",
                },
                "calibration object has unexpected or missing fields",
            )
            _require(value.get("schema") == CALIBRATION_SCHEMA, "calibration schema mismatch")
            _require(value.get("protocol_id") == self.protocol_id, "calibration protocol mismatch")
            _require(
                value.get("prompt_seed_pairs") == [list(item) for item in expected_pairs],
                "calibration split or seed set mismatch",
            )
            persisted_rows = value.get("rows")
            _require(isinstance(persisted_rows, list), "calibration rows must be a list")
            expected_rows: list[dict[str, Any]] = []
            for prompt in self.calibration_prompts:
                for seed in self.seeds:
                    key = self._case_key(prompt, seed)
                    for chunk_index in range(self.chunk_count):
                        _require(
                            self._checkpoint_path(
                                "calibration-baselines", key, chunk_index
                            ).is_file(),
                            "calibration file exists without complete baseline checkpoints",
                        )
                    baseline = self._baseline(prompt=prompt, seed=seed)
                    prior_ranks: list[float] = []
                    for chunk_index in self.candidate_chunks:
                        generated = baseline.generated[chunk_index]
                        expected_rows.append(
                            self._calibration_row(
                                prompt=prompt,
                                seed=seed,
                                chunk_index=chunk_index,
                                generated=generated,
                                prior_ranks=prior_ranks,
                            )
                        )
                        assert generated.effective_rank is not None
                        prior_ranks.append(float(generated.effective_rank))
            _require(
                persisted_rows == expected_rows,
                "calibration rows do not match baseline rank/hidden evidence",
            )
            expected_id = f"cal-{_digest_json(expected_rows)[:40]}"
            _require(
                value.get("calibration_id") == expected_id,
                "calibration_id does not match canonical rows",
            )
            return self._calibrator_from_rows(
                calibration_id=value["calibration_id"], rows=expected_rows
            )

        rows: list[dict[str, Any]] = []
        for prompt in self.calibration_prompts:
            for seed in self.seeds:
                baseline = self._baseline(prompt=prompt, seed=seed)
                prior_ranks: list[float] = []
                for chunk_index in self.candidate_chunks:
                    generated = baseline.generated[chunk_index]
                    rows.append(
                        self._calibration_row(
                            prompt=prompt,
                            seed=seed,
                            chunk_index=chunk_index,
                            generated=generated,
                            prior_ranks=prior_ranks,
                        )
                    )
                    assert generated.effective_rank is not None
                    prior_ranks.append(float(generated.effective_rank))
        calibration_id = f"cal-{_digest_json(rows)[:40]}"
        value = {
            "schema": CALIBRATION_SCHEMA,
            "protocol_id": self.protocol_id,
            "calibration_id": calibration_id,
            "prompt_seed_pairs": [list(item) for item in expected_pairs],
            "rows": rows,
        }
        _write_immutable_json(destination, value)
        return self._calibrator_from_rows(
            calibration_id=calibration_id, rows=rows
        )

    def _row_id(self, *parts: Any) -> str:
        return _digest_json([self.protocol_id, *parts])

    def run_single_point(self, *, split: str = "evaluation") -> tuple[dict[str, Any], ...]:
        """Run 2(a); ``smoke`` uses frozen E50[:10], formal evaluation uses E50."""

        _require(split in {"smoke", "evaluation"}, "2(a) split must be smoke or evaluation")
        calibrator = self.run_calibration()
        prompts = self.smoke_prompts if split == "smoke" else self.evaluation_prompts
        for prompt in prompts:
            for seed in self.seeds:
                baseline = self._baseline(prompt=prompt, seed=seed)
                fork_condition_hash = _digest_text(prompt.prompt)
                baseline_draft_ranks: list[float] = []
                for chunk_index in self.candidate_chunks:
                    draft = baseline.generated[chunk_index]
                    _require(draft.effective_rank is not None, "baseline Draft rank is missing")
                    baseline_draft_ranks.append(float(draft.effective_rank))
                    event_id = self._row_id("2a", split, prompt.prompt_id, seed, chunk_index)
                    if self.raw_2a.has(event_id):
                        continue
                    fork = SinglePointFork.open(baseline.history, chunk_index)
                    risk_score, erank_drop_score = self._risk_from_rank_history(
                        calibrator, baseline_draft_ranks
                    )
                    noise = self._materialize_noise(
                        prompt=prompt,
                        seed=seed,
                        chunk_index=chunk_index,
                    )
                    _require(
                        noise.descriptor.descriptor_sha256
                        == fork.original_draft.initial_noise_descriptor_sha256,
                        "2(a) Target branch does not reuse baseline initial noise",
                    )
                    target_logical_call_id = self._target_logical_call_id(
                        backend=self.target,
                        prompt=prompt,
                        history=fork.prefix,
                        noise=noise,
                    )
                    with self._target_rpc_scope(
                        event_id,
                        phase="single_point_candidate",
                    ):
                        target = self._generate(
                            backend=self.target,
                            prompt=prompt,
                            history=fork.prefix,
                            noise=noise,
                            output_stem=(
                                f"latents/2a-{split}/{self._case_key(prompt, seed)}/"
                                f"chunk-{chunk_index:02d}-target"
                            ),
                            capture_hidden=False,
                        )
                    target_history = fork.commit_target(
                        context=target.context,
                        segment=target.segment,
                    )
                    draft_history = fork.commit_draft(
                        context=draft.context,
                        segment=draft.segment,
                    )
                    evidence_relative = (
                        f"provenance/2a-forks/{split}/{self._case_key(prompt, seed)}/"
                        f"chunk-{chunk_index:02d}-{target.segment.segment_sha256[:16]}.json"
                    )
                    evidence_path = self.run_dir / evidence_relative
                    _write_immutable_json(
                        evidence_path,
                        {
                            "schema": FORK_EVIDENCE_SCHEMA,
                            "protocol_id": self.protocol_id,
                            "event_id": event_id,
                            "split": split,
                            "prompt_id": prompt.prompt_id,
                            "trajectory_seed": seed,
                            "chunk_index": chunk_index,
                            "condition_sha256": fork_condition_hash,
                            "prefix": fork.prefix.to_dict(),
                            "initial_noise": {
                                "descriptor": noise.descriptor.to_dict(),
                                "tensor_path": noise.tensor_path,
                            },
                            "draft_branch": {
                                "context": _context_dict(draft.context),
                                "segment": draft.segment.to_dict(),
                                "history": draft_history.to_dict(),
                            },
                            "target_branch": {
                                "context": _context_dict(target.context),
                                "segment": target.segment.to_dict(),
                                "history": target_history.to_dict(),
                            },
                        },
                    )
                    evidence_sha256, _evidence_nbytes = _sha256_file(evidence_path)
                    q_d = self.evaluator.score_local(
                        # Decode branch A at the same prefix length as branch
                        # B.  This does not rely on future-latent invariance of
                        # the causal VAE to make the local comparison paired.
                        history=draft_history,
                        chunk_index=chunk_index,
                        measurement_id=f"2a-{event_id}-draft",
                    )
                    q_t = self.evaluator.score_local(
                        history=target_history,
                        chunk_index=chunk_index,
                        measurement_id=f"2a-{event_id}-target",
                    )
                    record = {
                        "schema": RAW_2A_SCHEMA,
                        "event_id": event_id,
                        "split": split,
                        "prompt_id": prompt.prompt_id,
                        "seed": seed,
                        "chunk_index": chunk_index,
                        "rank": float(draft.effective_rank),
                        "risk_score": risk_score,
                        "Q_D": q_d.primary,
                        "Q_T": q_t.primary,
                        "delta_Q": q_t.primary - q_d.primary,
                        "TC_D": q_d.temporal_consistency,
                        "TC_T": q_t.temporal_consistency,
                        "delta_TC": q_t.temporal_consistency - q_d.temporal_consistency,
                        "prefix_hash": fork.prefix.head_sha256,
                        "condition_hash": fork_condition_hash,
                        "noise_hash": noise.descriptor.tensor_sha256,
                        "draft_kv_hash": draft.context.kv.cache_sha256,
                        "target_kv_hash": target.context.kv.cache_sha256,
                        "target_rpc_attempt_id": target.target_attempt_id,
                        "target_logical_call_id": target_logical_call_id,
                        "target_latent_path": target.segment.latent_path,
                        "target_latent_sha256": target.segment.latent_sha256,
                        "target_noise_descriptor_sha256": (
                            target.context.noise.descriptor_sha256
                        ),
                        "hidden_state_path": draft.hidden_state_path,
                        "hidden_state_sha256": draft.hidden_state_sha256,
                        "draft_segment_hash": fork.original_draft.segment_sha256,
                        "target_segment_hash": target.segment.segment_sha256,
                        "fork_evidence_path": evidence_relative,
                        "fork_evidence_sha256": evidence_sha256,
                    }
                    if erank_drop_score is not None:
                        record["erank_drop_score"] = erank_drop_score
                    self.raw_2a.append(record)
        selected_ids = {item.prompt_id for item in prompts}
        return tuple(
            row
            for row in self.raw_2a.records()
            if row.get("split") == split and row.get("prompt_id") in selected_ids
        )

    def _rollout_key(self, prompt: PromptCase, seed: int, route_id: str) -> str:
        return _digest_json([self.protocol_id, prompt.prompt_id, seed, route_id])[:48]

    def _canonical_rollout_root_checkpoint(self, prompt: PromptCase, seed: int) -> Path:
        key = _digest_json([self.protocol_id, "rollout-root", prompt.prompt_id, seed])[:48]
        return self._checkpoint_path("rollout-roots", key, 0)

    def _canonical_rollout_root(
        self, *, prompt: PromptCase, seed: int
    ) -> tuple[LatentHistory, GeneratedChunk]:
        """Generate/load the one exact Target chunk 0 shared by every strategy."""

        trajectory_id = self._trajectory_id("rollout", prompt.prompt_id, seed)
        key = _digest_json([self.protocol_id, "rollout-root", prompt.prompt_id, seed])[:48]
        checkpoint = self._canonical_rollout_root_checkpoint(prompt, seed)
        empty = LatentHistory(
            trajectory_id=trajectory_id,
            prompt_id=prompt.prompt_id,
            trajectory_seed=seed,
        )
        if checkpoint.exists():
            history, generated, metadata = self._load_checkpoint(checkpoint)
            _require(history.next_chunk_index == 1, "canonical rollout root has wrong length")
            _require(
                history.prefix_before(0).manifest_sha256 == empty.manifest_sha256,
                "canonical rollout root is bound to another empty history",
            )
            _require(
                metadata
                == {
                    "kind": "canonical_rollout_root",
                    "chunk_index": 0,
                    "corrected": True,
                },
                "canonical rollout root metadata drifted",
            )
            _require(generated.context.role == "target", "canonical chunk 0 is not Target")
            return history, generated

        noise = self._materialize_noise(prompt=prompt, seed=seed, chunk_index=0)
        generated = self._generate(
            backend=self.target,
            prompt=prompt,
            history=empty,
            noise=noise,
            output_stem=f"latents/rollout-roots/{key}/chunk-00-target",
            capture_hidden=False,
        )
        history = empty.append(generated.segment)
        self._write_checkpoint(
            path=checkpoint,
            history=history,
            generated=generated,
            metadata={
                "kind": "canonical_rollout_root",
                "chunk_index": 0,
                "corrected": True,
            },
        )
        return history, generated

    def _load_rollout_prefix(
        self, *, key: str, trajectory_id: str, prompt: PromptCase, seed: int
    ) -> tuple[LatentHistory, list[dict[str, Any]], list[GeneratedChunk]]:
        history = LatentHistory(
            trajectory_id=trajectory_id,
            prompt_id=prompt.prompt_id,
            trajectory_seed=seed,
        )
        metadata: list[dict[str, Any]] = []
        generated_chunks: list[GeneratedChunk] = []
        for chunk_index in range(self.chunk_count):
            path = self._checkpoint_path("rollouts", key, chunk_index)
            if not path.exists():
                break
            restored, latest, item = self._load_checkpoint(path)
            _require(restored.next_chunk_index == chunk_index + 1, "rollout checkpoint index drift")
            _require(
                restored.prefix_before(chunk_index).manifest_sha256 == history.manifest_sha256,
                "rollout checkpoint does not extend the resumed history",
            )
            history = restored
            metadata.append(item)
            generated_chunks.append(latest)
        return history, metadata, generated_chunks

    def _verify_canonical_rollout_root(
        self,
        *,
        history: LatentHistory,
        metadata: Sequence[Mapping[str, Any]],
        generated_chunks: Sequence[GeneratedChunk],
        canonical_history: LatentHistory,
        canonical_generated: GeneratedChunk,
    ) -> None:
        """Fail closed if a resumed route did not copy the frozen Target root."""

        _require(history.next_chunk_index >= 1, "resumed rollout has no chunk 0")
        _require(bool(metadata) and bool(generated_chunks), "resumed rollout root is incomplete")
        _require(
            history.prefix_before(1).to_dict() == canonical_history.to_dict(),
            "resumed rollout chunk 0 differs from the canonical Target root",
        )
        _require(
            _generated_dict(generated_chunks[0]) == _generated_dict(canonical_generated),
            "resumed rollout chunk-0 context differs from the canonical Target root",
        )
        root_metadata = metadata[0]
        _require(
            root_metadata.get("chunk_index") == 0
            and root_metadata.get("corrected") is True
            and root_metadata.get("committed_role") == "target",
            "resumed rollout chunk-0 metadata is invalid",
        )
        _require(
            root_metadata.get("committed_kv_hash")
            == canonical_generated.context.kv.cache_sha256,
            "resumed rollout chunk-0 KV differs from the canonical Target root",
        )
        _require(
            root_metadata.get("initial_noise_descriptor_sha256")
            == canonical_generated.context.noise.descriptor_sha256,
            "resumed rollout chunk-0 noise differs from the canonical Target root",
        )

    def _validate_online_draft_proposal(
        self,
        *,
        chunk_index: int,
        metadata: Mapping[str, Any],
        committed: GeneratedChunk,
        prefix: LatentHistory,
    ) -> GeneratedChunk | None:
        """Validate the decision-time Draft evidence stored by an online route."""

        _require(
            "draft_proposal" in metadata,
            f"online chunk {chunk_index} is missing draft_proposal evidence",
        )
        raw = metadata["draft_proposal"]
        if chunk_index == 0:
            _require(raw is None, "online chunk 0 draft_proposal must be null")
            return None

        _require(
            isinstance(raw, Mapping) and set(raw) == _GENERATED_CHUNK_KEYS,
            f"online chunk {chunk_index} draft_proposal fields are invalid",
        )
        diagnostics = raw.get("diagnostics")
        _require(
            isinstance(diagnostics, Mapping)
            and set(diagnostics) == (_DRAFT_PROPOSAL_DIAGNOSTIC_KEYS | ({"kv_reuse"} if "kv_reuse" in diagnostics else set())),
            f"online chunk {chunk_index} draft_proposal diagnostics are invalid",
        )
        try:
            proposal = _generated_from_dict(raw)
            proposal.segment.verify_context(prefix, proposal.context)
            proposal.segment.verify_file(self.run_dir)
        except (CampaignError, CorrectionStateError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"online chunk {chunk_index} draft_proposal is invalid: {error}"
            ) from error
        _require(
            dict(raw) == _generated_dict(proposal),
            f"online chunk {chunk_index} draft_proposal is not canonical",
        )
        _require(
            proposal.context.role == "draft"
            and proposal.context.model_id == self.draft.model_id
            and proposal.context.model_artifact_sha256
            == self.draft.model_artifact_sha256,
            f"online chunk {chunk_index} draft_proposal model binding is invalid",
        )
        _require(
            proposal.target_attempt_id is None
            and proposal.target_logical_call_id is None,
            f"online chunk {chunk_index} draft_proposal carries Target lifecycle IDs",
        )
        _require(
            proposal.hidden_state_path is None
            and proposal.hidden_state_sha256 is None,
            f"online chunk {chunk_index} draft_proposal unexpectedly carries hidden tensors",
        )
        rank = proposal.effective_rank
        _require(
            isinstance(rank, (int, float))
            and not isinstance(rank, bool)
            and math.isfinite(float(rank))
            and float(rank) >= 1.0,
            f"online chunk {chunk_index} draft_proposal rank is invalid",
        )
        try:
            validate_effective_rank_evidence(
                diagnostics.get("rank_evidence"),
                expected_rank=float(rank),
                expected_eigenvalue_count=256,
            )
        except AdapterError as error:
            raise CampaignError(
                f"online chunk {chunk_index} draft_proposal rank evidence is invalid: {error}"
            ) from error
        _boundary_velocity_witness(
            proposal,
            label=f"online chunk {chunk_index} draft_proposal",
        )
        kv_witness_sha256 = diagnostics.get("kv_witness_sha256")
        kv_witness = diagnostics.get("kv_witness")
        _require(
            isinstance(kv_witness_sha256, str)
            and _SHA256_RE.fullmatch(kv_witness_sha256) is not None
            and kv_witness_sha256 == proposal.context.kv.cache_sha256,
            f"online chunk {chunk_index} draft_proposal KV witness is inconsistent",
        )
        _require(
            isinstance(kv_witness, Mapping)
            and _digest_json(kv_witness) == kv_witness_sha256,
            f"online chunk {chunk_index} draft_proposal KV witness digest is invalid",
        )
        forward_trace = diagnostics.get("forward_trace")
        _require(
            isinstance(forward_trace, list),
            f"online chunk {chunk_index} draft_proposal forward trace is invalid",
        )
        try:
            from .draft_reuse_evidence import validate_reuse_evidence
            reused = validate_reuse_evidence(diagnostics.get("kv_reuse"), history_chunks=chunk_index,
                history_file_sha256=[x.latent_sha256 for x in prefix.segments])
            validate_self_forcing_forward_trace(
                forward_trace, history_chunks=chunk_index, reused_history_chunks=reused,
            )
        except (AdapterError, ValueError) as error:
            raise CampaignError(
                f"online chunk {chunk_index} draft_proposal forward trace is invalid: {error}"
            ) from error

        corrected = metadata.get("corrected")
        _require(
            isinstance(corrected, bool),
            f"online chunk {chunk_index} corrected flag is invalid",
        )
        _require(
            metadata.get("rank") == float(rank)
            and metadata.get("draft_parent_history_hash")
            == proposal.context.history_head_sha256
            and metadata.get("draft_kv_hash") == proposal.context.kv.cache_sha256
            and metadata.get("initial_noise_descriptor_sha256")
            == proposal.context.noise.descriptor_sha256,
            f"online chunk {chunk_index} metadata disagrees with draft_proposal",
        )
        expected_role = "target" if corrected else "draft"
        _require(
            committed.context.role == expected_role
            and metadata.get("committed_role") == expected_role
            and metadata.get("committed_kv_hash")
            == committed.context.kv.cache_sha256,
            f"online chunk {chunk_index} committed decision is inconsistent",
        )
        _require(
            committed.context.history_head_sha256
            == proposal.context.history_head_sha256
            and committed.context.noise.to_dict() == proposal.context.noise.to_dict(),
            f"online chunk {chunk_index} committed result is not same-prefix/same-noise",
        )
        if not corrected:
            _require(
                _generated_dict(committed) == _generated_dict(proposal),
                f"online chunk {chunk_index} accepted Draft differs from draft_proposal",
            )
        return proposal

    def _validate_lookahead_probe(
        self,
        *,
        raw: object,
        chunk_index: int,
        prefix: LatentHistory,
    ) -> GeneratedChunk:
        """Validate a speculative Draft successor without treating it as committed."""

        _require(
            isinstance(raw, Mapping) and set(raw) == _GENERATED_CHUNK_KEYS,
            f"lookahead probe chunk {chunk_index} fields are invalid",
        )
        try:
            parsed = _generated_from_dict(raw)
        except (CampaignError, CorrectionStateError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"lookahead probe chunk {chunk_index} is invalid: {error}"
            ) from error
        synthetic_metadata = {
            "draft_proposal": dict(raw),
            "corrected": False,
            "rank": parsed.effective_rank,
            "draft_parent_history_hash": parsed.context.history_head_sha256,
            "draft_kv_hash": parsed.context.kv.cache_sha256,
            "initial_noise_descriptor_sha256": (
                parsed.context.noise.descriptor_sha256
            ),
            "committed_role": "draft",
            "committed_kv_hash": parsed.context.kv.cache_sha256,
        }
        validated = self._validate_online_draft_proposal(
            chunk_index=chunk_index,
            metadata=synthetic_metadata,
            committed=parsed,
            prefix=prefix,
        )
        _require(validated is not None, "lookahead probe unexpectedly resolved to null")
        return validated

    def _validate_lookahead_target_candidate(
        self,
        *,
        raw: object,
        chunk_index: int,
        prefix: LatentHistory,
        expected_noise: InitialNoiseDescriptor,
    ) -> GeneratedChunk:
        """Validate an audit-only Target proposal, including rejected proposals."""

        _require(
            isinstance(raw, Mapping) and set(raw) == _GENERATED_CHUNK_KEYS,
            f"lookahead Target candidate chunk {chunk_index} fields are invalid",
        )
        try:
            candidate = _generated_from_dict(raw)
            candidate.segment.verify_context(prefix, candidate.context)
            candidate.segment.verify_file(self.run_dir)
        except (CampaignError, CorrectionStateError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                f"lookahead Target candidate chunk {chunk_index} is invalid: {error}"
            ) from error
        _require(
            dict(raw) == _generated_dict(candidate)
            and candidate.context.role == "target"
            and candidate.context.model_id == self.target.model_id
            and candidate.context.model_artifact_sha256
            == self.target.model_artifact_sha256
            and candidate.context.noise.to_dict() == expected_noise.to_dict()
            and candidate.effective_rank is None
            and candidate.hidden_state_path is None
            and candidate.hidden_state_sha256 is None,
            f"lookahead Target candidate chunk {chunk_index} binding is invalid",
        )
        diagnostics = candidate.diagnostics
        _require(
            isinstance(diagnostics, Mapping)
            and TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS <= set(diagnostics),
            f"lookahead Target candidate chunk {chunk_index} diagnostics are invalid",
        )
        _boundary_velocity_witness(
            candidate,
            label=f"lookahead Target candidate chunk {chunk_index}",
        )
        return candidate

    def _score_rollout(
        self,
        *,
        key: str,
        route_id: str,
        corrected_chunks: tuple[int, ...],
        proposed_target_chunks: tuple[int, ...] | None = None,
        history: LatentHistory,
    ) -> SequentialRollout:
        proposed = corrected_chunks if proposed_target_chunks is None else proposed_target_chunks
        destination = self.run_dir / "measurements" / "rollouts" / f"{key}.json"
        if destination.exists():
            value = _load_json_object(destination)
            _require(value.get("schema") == ROLLOUT_SCORE_SCHEMA, "rollout score schema mismatch")
            _require(tuple(value["corrected_chunks"]) == corrected_chunks, "rollout route mismatch")
            _require(
                tuple(value.get("proposed_target_chunks", ())) == proposed
                and value.get("budgeted_target_call_count") == 1 + len(proposed),
                "rollout proposal budget mismatch",
            )
            _require(value["final_history_hash"] == history.head_sha256, "rollout history drift")
            quality = QualityMeasurement(
                primary=value["full_video_quality"],
                temporal_consistency=value["temporal_consistency"],
            )
        else:
            quality = self.evaluator.score_full(
                history=history,
                measurement_id=f"2b-{key}",
            )
            value = {
                "schema": ROLLOUT_SCORE_SCHEMA,
                "route_id": route_id,
                "corrected_chunks": list(corrected_chunks),
                "proposed_target_chunks": list(proposed),
                "full_video_quality": quality.primary,
                "temporal_consistency": quality.temporal_consistency,
                "target_call_count": 1 + len(corrected_chunks),
                "budgeted_target_call_count": 1 + len(proposed),
                "final_history_hash": history.head_sha256,
            }
            _write_immutable_json(destination, value)
        result = FullVideoRollout(
            corrected_chunks=corrected_chunks,
            full_video_quality=quality.primary,
            target_call_count=1 + len(corrected_chunks),
            rollout_id=route_id,
            chunk_count=self.chunk_count,
        )
        return SequentialRollout(
            result=result,
            temporal_consistency=quality.temporal_consistency,
            history=history,
            proposed_target_chunks=proposed,
        )

    def _prepare_rollout(
        self,
        *,
        key: str,
        route_id: str,
        corrected_chunks: tuple[int, ...],
        proposed_target_chunks: tuple[int, ...] | None = None,
        history: LatentHistory,
    ) -> PendingSequentialRollout:
        prepare = getattr(self.evaluator, "prepare_full", None)
        _require(
            callable(prepare),
            "deferred full quality requires an evaluator.prepare_full implementation",
        )
        prepared = prepare(history=history, measurement_id=f"2b-{key}")
        _require(
            isinstance(prepared, PreparedFullQuality),
            "evaluator.prepare_full returned the wrong result type",
        )
        pending = PendingSequentialRollout(
            route_key=key,
            route_id=route_id,
            corrected_chunks=corrected_chunks,
            proposed_target_chunks=(
                corrected_chunks
                if proposed_target_chunks is None
                else proposed_target_chunks
            ),
            history=history,
            prepared_quality=prepared,
        )
        _write_immutable_json(
            self.run_dir / "measurements" / "rollouts" / f"{key}.pending.json",
            pending.to_dict(),
        )
        return pending

    def _materialize_fixed_positions(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        corrected_chunks: Sequence[int],
        bypass_draft_on_target_chunks: bool = False,
        on_commit: Callable[[LatentHistory], None] | None = None,
        before_generate: Callable[[str], None] | None = None,
    ) -> tuple[str, str, tuple[int, ...], LatentHistory]:
        """Materialize one true sequential trajectory for a fixed correction set."""

        selected = tuple(corrected_chunks)
        _require((on_commit is None and before_generate is None) or bypass_draft_on_target_chunks,
                 'fixed-route render callbacks require direct committed Target generation')
        _require(
            selected == tuple(sorted(set(selected)))
            and all(chunk in self.candidate_chunks for chunk in selected),
            f"corrected_chunks must be sorted unique values in [1,{self.chunk_count - 1}]",
        )
        route_id = "fixed-" + ("none" if not selected else "-".join(map(str, selected)))
        if bypass_draft_on_target_chunks:
            route_id += "-direct-target"
        key = self._rollout_key(prompt, seed, route_id)
        trajectory_id = self._trajectory_id("rollout", prompt.prompt_id, seed)
        history, restored_metadata, restored_generated = self._load_rollout_prefix(
            key=key,
            trajectory_id=trajectory_id,
            prompt=prompt,
            seed=seed,
        )
        if history.next_chunk_index == 0:
            history, target_root = self._canonical_rollout_root(prompt=prompt, seed=seed)
            root_metadata = {
                "kind": "fixed_rollout",
                "route_id": route_id,
                "chunk_index": 0,
                "corrected": True,
                "draft_rank": None,
                "draft_parent_history_hash": None,
                "draft_kv_hash": None,
                "committed_role": "target",
                "committed_kv_hash": target_root.context.kv.cache_sha256,
                "initial_noise_descriptor_sha256": (
                    target_root.context.noise.descriptor_sha256
                ),
            }
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, 0),
                history=history,
                generated=target_root,
                metadata=root_metadata,
            )
            restored_metadata = [root_metadata]
            restored_generated = [target_root]
        else:
            _require(
                self._canonical_rollout_root_checkpoint(prompt, seed).is_file(),
                "resumed rollout is missing its canonical Target root checkpoint",
            )
            canonical_history, target_root = self._canonical_rollout_root(
                prompt=prompt, seed=seed
            )
            self._verify_canonical_rollout_root(
                history=history,
                metadata=restored_metadata,
                generated_chunks=restored_generated,
                canonical_history=canonical_history,
                canonical_generated=target_root,
            )
        for index, metadata in enumerate(restored_metadata):
            expected = index in selected if index > 0 else True
            _require(metadata.get("corrected") is expected, "resumed fixed route changed")
        state = CorrectionState(history=history)
        if on_commit is not None:
            for count in range(1, history.next_chunk_index+1):
                on_commit(history.prefix_before(count))
        for chunk_index in range(history.next_chunk_index, self.chunk_count):
            if before_generate is not None:
                before_generate('target' if chunk_index in selected else 'draft')
            noise = self._materialize_noise(prompt=prompt, seed=seed, chunk_index=chunk_index)
            if chunk_index == 0:
                target = self._generate(
                    backend=self.target,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=f"latents/rollouts/{key}/chunk-00-target",
                    capture_hidden=False,
                )
                state = CorrectionState(history=state.history.append(target.segment))
                committed = target
                draft: GeneratedChunk | None = None
                corrected = True
            elif bypass_draft_on_target_chunks and chunk_index in selected:
                # The schedule is known before this chunk. No rejected Draft
                # proposal is needed; bind Target directly to accepted history.
                target = self._generate(
                    backend=self.target, prompt=prompt, history=state.history, noise=noise,
                    output_stem=f"latents/rollouts/{key}/chunk-{chunk_index:02d}-target",
                    capture_hidden=False,
                )
                state = CorrectionState(history=state.history.append(target.segment))
                committed, draft, corrected = target, None, True
            else:
                draft = self._generate(
                    backend=self.draft,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=f"latents/rollouts/{key}/chunk-{chunk_index:02d}-draft",
                    # Multi-budget rollouts need the scalar rank but not a
                    # persisted hidden tensor for every candidate trajectory.
                    capture_hidden=False,
                )
                state = state.stage_draft(context=draft.context, segment=draft.segment)
                if chunk_index in selected:
                    target = self._generate(
                        backend=self.target,
                        prompt=prompt,
                        history=state.history,
                        noise=noise,
                        output_stem=f"latents/rollouts/{key}/chunk-{chunk_index:02d}-target",
                        capture_hidden=False,
                    )
                    state = state.replace_with_target(
                        context=target.context,
                        segment=target.segment,
                    )
                    committed = target
                    corrected = True
                else:
                    state = state.accept_draft()
                    committed = draft
                    corrected = False
            metadata = {
                "kind": "fixed_rollout",
                "route_id": route_id,
                "chunk_index": chunk_index,
                "corrected": corrected,
                "draft_proposal": None if draft is None else _generated_dict(draft),
                "draft_rank": None if draft is None else draft.effective_rank,
                "draft_parent_history_hash": (
                    None if draft is None else draft.context.history_head_sha256
                ),
                "draft_kv_hash": None if draft is None else draft.context.kv.cache_sha256,
                "committed_role": committed.context.role,
                "committed_kv_hash": committed.context.kv.cache_sha256,
                "initial_noise_descriptor_sha256": noise.descriptor.descriptor_sha256,
            }
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, chunk_index),
                history=state.history,
                generated=committed,
                metadata=metadata,
            )
            if on_commit is not None:
                on_commit(state.history)
        _require(
            state.history.next_chunk_index == self.chunk_count,
            f"rollout did not reach {self.chunk_count} chunks",
        )
        actual = tuple(
            segment.chunk_index
            for segment in state.history.segments[1:]
            if segment.producer_role == "target"
        )
        _require(actual == selected, "committed Target chunks do not match the fixed route")
        return key, route_id, selected, state.history

    def _materialize_online_lookahead_coverage_boundary_gated(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
        acceptance_rule: str = "paired_boundary",
    ) -> tuple[
        str,
        str,
        tuple[int, ...],
        tuple[int, ...],
        LatentHistory,
    ]:
        """Materialize fixed-budget Target proposals with a paired boundary gate.

        The coverage controller chooses exactly ``budget`` proposal positions.
        At each position, the campaign compares the original Draft successor
        against a same-noise Draft successor generated from the Target-corrected
        prefix.  The Target candidate is committed only when the corrected
        successor recovers eRank and both its entry and exit boundary witnesses
        are no worse than the original Draft branch.  A rejection consumes the
        proposal opportunity and continues from the original branch.
        """

        _require(acceptance_rule in {"paired_boundary", "rank_only", "proposal_always"},
                 "unknown paired correction acceptance rule")
        variant = acceptance_rule != "paired_boundary"
        _require(not variant or os.environ.get("CFEXP_BOUNDARY_EARLY_REJECT") != "1",
                 "motion acceptance ablations require entry short-circuit disabled")
        gate_name = acceptance_rule.replace("_", "-") if variant else "boundary-gated"
        route_id = f"online-lookahead-coverage-{gate_name}-erank-drop-k{budget}"
        policy = LookaheadCoverageConstrainedOnlineRankGuidedPolicy(
            budget=budget,
            calibration=calibrator,
            evaluation_id=(
                f"eval-lookahead-coverage-boundary-{prompt.prompt_id}-{seed}-k{budget}"
            ),
            chunk_count=self.chunk_count,
            window_radius_chunks=window_radius_chunks,
            min_gap_chunks=minimum_gap_chunks,
            max_gap_chunks=maximum_gap_chunks,
        )
        gate = MotionAcceptanceGate(acceptance_rule) if variant else BoundarySmoothnessGate()
        early_reject_enabled = os.environ.get("CFEXP_BOUNDARY_EARLY_REJECT") == "1"
        fork_enabled = os.environ.get("CFEXP_DRAFT_KV_FORK") == "1"
        fork_window_starts = (
            {start for start, _end in policy.static_windows} if fork_enabled else set()
        )
        strategy = (f"online_lookahead_coverage_{acceptance_rule}"
                    if variant else "online_lookahead_coverage_boundary_gated")
        key = self._rollout_key(prompt, seed, route_id)
        trajectory_id = self._trajectory_id("rollout", prompt.prompt_id, seed)
        history, restored_metadata, restored_generated = self._load_rollout_prefix(
            key=key,
            trajectory_id=trajectory_id,
            prompt=prompt,
            seed=seed,
        )
        if history.next_chunk_index == 0:
            history, target_root = self._canonical_rollout_root(prompt=prompt, seed=seed)
            root_metadata = {
                "kind": strategy,
                "route_id": route_id,
                "chunk_index": 0,
                "corrected": True,
                "risk_score": None,
                "draft_proposal": None,
                "draft_kv_hash": None,
                "committed_role": "target",
                "committed_kv_hash": target_root.context.kv.cache_sha256,
                "initial_noise_descriptor_sha256": (
                    target_root.context.noise.descriptor_sha256
                ),
                "gate_signal_chunk_index": None,
                "gate_signal_rank": None,
                "gate_probe": None,
                "gate_probe_parent_history_hash": None,
                "gate_probe_kv_hash": None,
                "gate_probe_noise_descriptor_sha256": None,
                "target_candidate_proposed": False,
                "target_candidate": None,
                "target_candidate_parent_history_hash": None,
                "target_candidate_kv_hash": None,
                "target_candidate_noise_descriptor_sha256": None,
                "corrected_frontier_probe": None,
                "corrected_frontier_parent_history_hash": None,
                "corrected_frontier_kv_hash": None,
                "corrected_frontier_noise_descriptor_sha256": None,
                "boundary_gate_decision": None,
                "target_candidate_accepted": None,
                "carried_frontier_kind": None,
                "probe_reused_as_successor": False,
                "terminal_flush": False,
            }
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, 0),
                history=history,
                generated=target_root,
                metadata=root_metadata,
            )
            restored_metadata = [root_metadata]
            restored_generated = [target_root]
        else:
            _require(
                self._canonical_rollout_root_checkpoint(prompt, seed).is_file(),
                "resumed boundary-gated route is missing its canonical Target root",
            )
            canonical_history, target_root = self._canonical_rollout_root(
                prompt=prompt,
                seed=seed,
            )
            self._verify_canonical_rollout_root(
                history=history,
                metadata=restored_metadata,
                generated_chunks=restored_generated,
                canonical_history=canonical_history,
                canonical_generated=target_root,
            )

        resolved_proposal_ranks: list[float] = []
        proposed_chunks: list[int] = []
        accepted_chunks: list[int] = []
        carried_probe: GeneratedChunk | None = None
        for chunk_index, (metadata, committed) in enumerate(
            zip(restored_metadata, restored_generated, strict=True)
        ):
            _require(
                metadata.get("kind") == strategy
                and metadata.get("route_id") == route_id
                and metadata.get("chunk_index") == chunk_index,
                "resumed boundary-gated route metadata changed",
            )
            prefix = history.prefix_before(chunk_index)
            proposal = self._validate_online_draft_proposal(
                chunk_index=chunk_index,
                metadata=metadata,
                committed=committed,
                prefix=prefix,
            )
            if chunk_index == 0:
                _require(
                    metadata.get("corrected") is True
                    and metadata.get("gate_probe") is None
                    and metadata.get("target_candidate") is None
                    and metadata.get("corrected_frontier_probe") is None
                    and metadata.get("boundary_gate_decision") is None,
                    "boundary-gated chunk 0 is not the canonical Target root",
                )
                continue
            _require(proposal is not None, "boundary-gated pending Draft is missing")
            if carried_probe is not None:
                _require(
                    _generated_dict(proposal) == _generated_dict(carried_probe),
                    "boundary-gated carried frontier was not reused as successor",
                )

            pending_rank = float(proposal.effective_rank)
            needs_preview = policy.preview_required(chunk_index=chunk_index)
            terminal_flush = chunk_index == self.chunk_count - 1
            original_probe: GeneratedChunk | None = None
            signal_chunk_index: int | None = None
            signal_rank: float | None = None
            risk: float | None = None
            erank_drop_score: float | None = None
            if needs_preview:
                signal_chunk_index = chunk_index + 1
                original_probe = self._validate_lookahead_probe(
                    raw=metadata.get("gate_probe"),
                    chunk_index=signal_chunk_index,
                    prefix=prefix.append(proposal.segment),
                )
                signal_rank = float(original_probe.effective_rank)
                risk, erank_drop_score = self._risk_from_rank_history(
                    calibrator,
                    (*resolved_proposal_ranks, pending_rank, signal_rank),
                )
            else:
                _require(
                    metadata.get("gate_probe") is None,
                    "budget-exhausted boundary decision carries an original preview",
                )
            controller_decision = policy.decide(
                chunk_index=chunk_index,
                signal_chunk_index=signal_chunk_index,
                risk_score=risk,
            )
            is_proposal = controller_decision.replace_with_target
            target_candidate: GeneratedChunk | None = None
            corrected_probe: GeneratedChunk | None = None
            gate_decision = None
            accepted = False
            if is_proposal:
                _require(
                    original_probe is not None,
                    "boundary-gated Target proposal has no original successor",
                )
                target_candidate = self._validate_lookahead_target_candidate(
                    raw=metadata.get("target_candidate"),
                    chunk_index=chunk_index,
                    prefix=prefix,
                    expected_noise=proposal.context.noise,
                )
                saved_gate = metadata.get("boundary_gate_decision")
                short_circuit = (
                    isinstance(saved_gate, Mapping)
                    and saved_gate.get("short_circuit") is True
                )
                if short_circuit:
                    # Replay the saved execution path, independent of the current
                    # flag.  Missing successors are valid only after this exact
                    # entry failure is independently re-established.
                    _require(
                        metadata.get("corrected_frontier_probe") is None,
                        "entry short-circuit checkpoint carries a corrected successor",
                    )
                    gate_decision = gate.reject_if_entry_worse(
                        draft_entry_witness_score=_boundary_velocity_witness(
                            proposal,
                            label=f"boundary-gated Draft candidate chunk {chunk_index}",
                        )["score"],
                        target_entry_witness_score=_boundary_velocity_witness(
                            target_candidate,
                            label=f"boundary-gated Target candidate chunk {chunk_index}",
                        )["score"],
                    )
                    _require(
                        gate_decision is not None,
                        "entry short-circuit checkpoint has no valid failing entry evidence",
                    )
                else:
                    corrected_prefix = prefix.append(target_candidate.segment)
                    corrected_probe = self._validate_lookahead_probe(
                        raw=metadata.get("corrected_frontier_probe"),
                        chunk_index=chunk_index + 1,
                        prefix=corrected_prefix,
                    )
                    _require(
                        corrected_probe.context.noise.to_dict()
                        == original_probe.context.noise.to_dict(),
                        "boundary-gated successor branches do not reuse the same noise",
                    )
                    gate_decision = gate.decide(
                        original_successor_erank=original_probe.effective_rank,
                        corrected_successor_erank=corrected_probe.effective_rank,
                        draft_entry_witness_score=float(
                            _boundary_velocity_witness(
                                proposal,
                                label=f"boundary-gated Draft candidate chunk {chunk_index}",
                            )["score"]
                        ),
                        target_entry_witness_score=float(
                            _boundary_velocity_witness(
                                target_candidate,
                                label=f"boundary-gated Target candidate chunk {chunk_index}",
                            )["score"]
                        ),
                        draft_exit_witness_score=float(
                            _boundary_velocity_witness(
                                original_probe,
                                label=f"boundary-gated original successor chunk {chunk_index + 1}",
                            )["score"]
                        ),
                        target_exit_witness_score=float(
                            _boundary_velocity_witness(
                                corrected_probe,
                                label=f"boundary-gated corrected successor chunk {chunk_index + 1}",
                            )["score"]
                        ),
                    )
                accepted = gate_decision.accept_target
                proposed_chunks.append(chunk_index)
                if accepted:
                    accepted_chunks.append(chunk_index)
                    _require(
                        _generated_dict(committed)
                        == _generated_dict(target_candidate),
                        "accepted boundary-gated Target differs from committed chunk",
                    )
                carried_probe = corrected_probe if accepted else original_probe
            else:
                _require(
                    metadata.get("target_candidate") is None
                    and metadata.get("corrected_frontier_probe") is None
                    and metadata.get("boundary_gate_decision") is None
                    and metadata.get("target_candidate_accepted") is None,
                    "non-proposal boundary checkpoint carries Target gate evidence",
                )
                carried_probe = original_probe

            expected_frontier_kind = (
                None
                if carried_probe is None
                else "corrected"
                if accepted
                else "original"
            )
            expected_original_parent = (
                None
                if original_probe is None
                else original_probe.context.history_head_sha256
            )
            expected_original_kv = (
                None if original_probe is None else original_probe.context.kv.cache_sha256
            )
            expected_original_noise = (
                None
                if original_probe is None
                else original_probe.context.noise.descriptor_sha256
            )
            expected_target_parent = (
                None
                if target_candidate is None
                else target_candidate.context.history_head_sha256
            )
            expected_target_kv = (
                None
                if target_candidate is None
                else target_candidate.context.kv.cache_sha256
            )
            expected_target_noise = (
                None
                if target_candidate is None
                else target_candidate.context.noise.descriptor_sha256
            )
            expected_corrected_parent = (
                None
                if corrected_probe is None
                else corrected_probe.context.history_head_sha256
            )
            expected_corrected_kv = (
                None
                if corrected_probe is None
                else corrected_probe.context.kv.cache_sha256
            )
            expected_corrected_noise = (
                None
                if corrected_probe is None
                else corrected_probe.context.noise.descriptor_sha256
            )
            _require(
                metadata.get("risk_score") == risk
                and metadata.get("decision") == asdict(controller_decision)
                and metadata.get("gate_signal_chunk_index") == signal_chunk_index
                and metadata.get("gate_signal_rank") == signal_rank
                and metadata.get("gate_probe_parent_history_hash")
                == expected_original_parent
                and metadata.get("gate_probe_kv_hash") == expected_original_kv
                and metadata.get("gate_probe_noise_descriptor_sha256")
                == expected_original_noise
                and metadata.get("target_candidate_proposed") is is_proposal
                and metadata.get("target_candidate_parent_history_hash")
                == expected_target_parent
                and metadata.get("target_candidate_kv_hash") == expected_target_kv
                and metadata.get("target_candidate_noise_descriptor_sha256")
                == expected_target_noise
                and metadata.get("corrected_frontier_parent_history_hash")
                == expected_corrected_parent
                and metadata.get("corrected_frontier_kv_hash")
                == expected_corrected_kv
                and metadata.get("corrected_frontier_noise_descriptor_sha256")
                == expected_corrected_noise
                and metadata.get("boundary_gate_decision")
                == (None if gate_decision is None else asdict(gate_decision))
                and metadata.get("target_candidate_accepted")
                == (accepted if is_proposal else None)
                and metadata.get("carried_frontier_kind") == expected_frontier_kind
                and metadata.get("probe_reused_as_successor")
                is (expected_frontier_kind is not None)
                and metadata.get("terminal_flush") is terminal_flush
                and metadata.get("corrected") is accepted,
                "resumed boundary-gated decision or evidence binding changed",
            )
            if erank_drop_score is None:
                _require(
                    "erank_drop_score" not in metadata,
                    "boundary-gated checkpoint has an unexpected eRank-drop score",
                )
            else:
                _require(
                    metadata.get("erank_drop_score") == erank_drop_score,
                    "resumed boundary-gated eRank-drop score changed",
                )
            if not accepted:
                resolved_proposal_ranks.append(pending_rank)

        state = CorrectionState(history=history)
        pending = carried_probe
        for chunk_index in range(history.next_chunk_index, self.chunk_count):
            noise = self._materialize_noise(
                prompt=prompt,
                seed=seed,
                chunk_index=chunk_index,
            )
            if pending is None:
                pending = self._generate(
                    backend=self.draft,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=(
                        f"latents/rollouts/{key}/chunk-{chunk_index:02d}-draft"
                    ),
                    capture_hidden=False,
                    **({"kv_branch_action": "preserve"}
                       if chunk_index in fork_window_starts else {}),
                )
            else:
                _require(
                    pending.segment.chunk_index == chunk_index
                    and pending.context.history_head_sha256 == state.history.head_sha256
                    and pending.context.noise.to_dict() == noise.descriptor.to_dict(),
                    "carried boundary frontier is not bound to the current successor",
                )
            _require(
                pending.effective_rank is not None,
                "boundary-gated pending Draft rank is missing",
            )
            state = state.stage_draft(context=pending.context, segment=pending.segment)
            pending_rank = float(pending.effective_rank)
            needs_preview = policy.preview_required(chunk_index=chunk_index)
            terminal_flush = chunk_index == self.chunk_count - 1
            original_probe: GeneratedChunk | None = None
            preview_noise: NoiseMaterialization | None = None
            signal_chunk_index: int | None = None
            signal_rank: float | None = None
            risk: float | None = None
            erank_drop_score: float | None = None
            if needs_preview:
                signal_chunk_index = chunk_index + 1
                preview_noise = self._materialize_noise(
                    prompt=prompt,
                    seed=seed,
                    chunk_index=signal_chunk_index,
                )
                original_probe = self._generate(
                    backend=self.draft,
                    prompt=prompt,
                    history=state.history.append(pending.segment),
                    noise=preview_noise,
                    output_stem=(
                        f"latents/rollouts/{key}/chunk-{signal_chunk_index:02d}-"
                        f"draft-preview-from-{chunk_index:02d}"
                    ),
                    capture_hidden=False,
                    **({"kv_branch_action": "preserve"}
                       if signal_chunk_index in fork_window_starts else {}),
                )
                _require(
                    original_probe.effective_rank is not None,
                    "boundary-gated original successor rank is missing",
                )
                signal_rank = float(original_probe.effective_rank)
                risk, erank_drop_score = self._risk_from_rank_history(
                    calibrator,
                    (*resolved_proposal_ranks, pending_rank, signal_rank),
                )
            controller_decision = policy.decide(
                chunk_index=chunk_index,
                signal_chunk_index=signal_chunk_index,
                risk_score=risk,
            )
            is_proposal = controller_decision.replace_with_target
            target_candidate: GeneratedChunk | None = None
            corrected_probe: GeneratedChunk | None = None
            gate_decision = None
            accepted = False
            if is_proposal:
                _require(
                    original_probe is not None and preview_noise is not None,
                    "boundary-gated Target proposal has no paired successor noise",
                )
                target_candidate = self._generate(
                    backend=self.target,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=(
                        f"latents/rollouts/{key}/chunk-{chunk_index:02d}-target"
                    ),
                    capture_hidden=False,
                )
                if early_reject_enabled:
                    gate_decision = gate.reject_if_entry_worse(
                        draft_entry_witness_score=_boundary_velocity_witness(
                            pending,
                            label=f"boundary-gated Draft candidate chunk {chunk_index}",
                        )["score"],
                        target_entry_witness_score=_boundary_velocity_witness(
                            target_candidate,
                            label=f"boundary-gated Target candidate chunk {chunk_index}",
                        )["score"],
                    )
                if gate_decision is None:
                    corrected_probe = self._generate(
                        backend=self.draft,
                        prompt=prompt,
                        history=state.history.append(target_candidate.segment),
                        noise=preview_noise,
                        output_stem=(
                            f"latents/rollouts/{key}/chunk-{chunk_index + 1:02d}-"
                            f"draft-after-target-from-{chunk_index:02d}"
                        ),
                        capture_hidden=False,
                        **({"kv_branch_action": "consume"} if fork_enabled else {}),
                    )
                    _require(
                        corrected_probe.effective_rank is not None,
                        "boundary-gated corrected successor rank is missing",
                    )
                    gate_decision = gate.decide(
                        original_successor_erank=original_probe.effective_rank,
                        corrected_successor_erank=corrected_probe.effective_rank,
                        draft_entry_witness_score=float(
                            _boundary_velocity_witness(
                                pending,
                                label=f"boundary-gated Draft candidate chunk {chunk_index}",
                            )["score"]
                        ),
                        target_entry_witness_score=float(
                            _boundary_velocity_witness(
                                target_candidate,
                                label=f"boundary-gated Target candidate chunk {chunk_index}",
                            )["score"]
                        ),
                        draft_exit_witness_score=float(
                            _boundary_velocity_witness(
                                original_probe,
                                label=f"boundary-gated original successor chunk {chunk_index + 1}",
                            )["score"]
                        ),
                        target_exit_witness_score=float(
                            _boundary_velocity_witness(
                                corrected_probe,
                                label=f"boundary-gated corrected successor chunk {chunk_index + 1}",
                            )["score"]
                        ),
                    )
                _require(
                    gate_decision.evidence_valid,
                    "boundary-gated Target decision has invalid evidence",
                )
                accepted = gate_decision.accept_target
                proposed_chunks.append(chunk_index)
                if accepted:
                    accepted_chunks.append(chunk_index)
                    state = state.replace_with_target(
                        context=target_candidate.context,
                        segment=target_candidate.segment,
                    )
                    committed = target_candidate
                    next_pending = corrected_probe
                else:
                    state = state.accept_draft()
                    committed = pending
                    next_pending = original_probe
            else:
                state = state.accept_draft()
                committed = pending
                next_pending = original_probe

            carried_frontier_kind = (
                None
                if next_pending is None
                else "corrected"
                if accepted
                else "original"
            )
            metadata = {
                "kind": strategy,
                "route_id": route_id,
                "chunk_index": chunk_index,
                "corrected": accepted,
                "risk_score": risk,
                "rank": pending_rank,
                "decision": asdict(controller_decision),
                "draft_proposal": _generated_dict(pending),
                "draft_parent_history_hash": pending.context.history_head_sha256,
                "draft_kv_hash": pending.context.kv.cache_sha256,
                "committed_role": committed.context.role,
                "committed_kv_hash": committed.context.kv.cache_sha256,
                "initial_noise_descriptor_sha256": noise.descriptor.descriptor_sha256,
                "gate_signal_chunk_index": signal_chunk_index,
                "gate_signal_rank": signal_rank,
                "gate_probe": (
                    None if original_probe is None else _generated_dict(original_probe)
                ),
                "gate_probe_parent_history_hash": (
                    None
                    if original_probe is None
                    else original_probe.context.history_head_sha256
                ),
                "gate_probe_kv_hash": (
                    None
                    if original_probe is None
                    else original_probe.context.kv.cache_sha256
                ),
                "gate_probe_noise_descriptor_sha256": (
                    None
                    if original_probe is None
                    else original_probe.context.noise.descriptor_sha256
                ),
                "target_candidate_proposed": is_proposal,
                "target_candidate": (
                    None
                    if target_candidate is None
                    else _generated_dict(target_candidate)
                ),
                "target_candidate_parent_history_hash": (
                    None
                    if target_candidate is None
                    else target_candidate.context.history_head_sha256
                ),
                "target_candidate_kv_hash": (
                    None
                    if target_candidate is None
                    else target_candidate.context.kv.cache_sha256
                ),
                "target_candidate_noise_descriptor_sha256": (
                    None
                    if target_candidate is None
                    else target_candidate.context.noise.descriptor_sha256
                ),
                "corrected_frontier_probe": (
                    None
                    if corrected_probe is None
                    else _generated_dict(corrected_probe)
                ),
                "corrected_frontier_parent_history_hash": (
                    None
                    if corrected_probe is None
                    else corrected_probe.context.history_head_sha256
                ),
                "corrected_frontier_kv_hash": (
                    None
                    if corrected_probe is None
                    else corrected_probe.context.kv.cache_sha256
                ),
                "corrected_frontier_noise_descriptor_sha256": (
                    None
                    if corrected_probe is None
                    else corrected_probe.context.noise.descriptor_sha256
                ),
                "boundary_gate_decision": (
                    None if gate_decision is None else asdict(gate_decision)
                ),
                "target_candidate_accepted": accepted if is_proposal else None,
                "carried_frontier_kind": carried_frontier_kind,
                "probe_reused_as_successor": carried_frontier_kind is not None,
                "terminal_flush": terminal_flush,
            }
            if erank_drop_score is not None:
                metadata["erank_drop_score"] = erank_drop_score
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, chunk_index),
                history=state.history,
                generated=committed,
                metadata=metadata,
            )
            if not accepted:
                resolved_proposal_ranks.append(pending_rank)
            pending = next_pending

        proposed = policy.finalize()
        accepted_tuple = tuple(accepted_chunks)
        actual = tuple(
            segment.chunk_index
            for segment in state.history.segments[1:]
            if segment.producer_role == "target"
        )
        _require(
            tuple(proposed_chunks) == proposed
            and len(proposed) == budget
            and set(accepted_tuple).issubset(proposed)
            and actual == accepted_tuple,
            "boundary-gated proposed/accepted Target history is inconsistent",
        )
        return key, route_id, accepted_tuple, proposed, state.history

    def rollout_fixed_positions(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        corrected_chunks: Sequence[int],
    ) -> SequentialRollout:
        """Run and synchronously score a fixed correction trajectory."""

        key, route_id, selected, history = self._materialize_fixed_positions(
            prompt=prompt,
            seed=seed,
            corrected_chunks=corrected_chunks,
        )
        return self._score_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def prepare_fixed_positions(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        corrected_chunks: Sequence[int],
    ) -> PendingSequentialRollout:
        """Run a fixed trajectory and seal its video without scoring it."""

        key, route_id, selected, history = self._materialize_fixed_positions(
            prompt=prompt,
            seed=seed,
            corrected_chunks=corrected_chunks,
        )
        return self._prepare_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def _materialize_online_rank_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        strategy: str = "online_rank_guided",
        coverage_window_radius_chunks: int | None = None,
        coverage_minimum_gap_chunks: int | None = None,
        coverage_maximum_gap_chunks: int | None = None,
    ) -> tuple[str, str, tuple[int, ...], LatentHistory]:
        """Materialize the deployable policy without future-risk access."""

        _require(
            strategy in {"online_rank_guided", "online_coverage_guided"},
            "online strategy is invalid",
        )
        coverage_values = (
            coverage_window_radius_chunks,
            coverage_minimum_gap_chunks,
            coverage_maximum_gap_chunks,
        )
        if strategy == "online_rank_guided":
            _require(
                coverage_values == (None, None, None),
                "unconstrained online strategy cannot carry coverage parameters",
            )
            route_id = f"online-erank-drop-k{budget}"
            policy = OnlineRankGuidedPolicy(
                budget=budget,
                calibration=calibrator,
                evaluation_id=f"eval-{prompt.prompt_id}-{seed}-k{budget}",
                chunk_count=self.chunk_count,
            )
        else:
            _require(
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 1
                    for value in coverage_values
                ),
                "coverage-guided strategy requires positive coverage parameters",
            )
            route_id = f"online-coverage-erank-drop-k{budget}"
            policy = CoverageConstrainedOnlineRankGuidedPolicy(
                budget=budget,
                calibration=calibrator,
                evaluation_id=(
                    f"eval-coverage-{prompt.prompt_id}-{seed}-k{budget}"
                ),
                chunk_count=self.chunk_count,
                window_radius_chunks=coverage_window_radius_chunks,
                min_gap_chunks=coverage_minimum_gap_chunks,
                max_gap_chunks=coverage_maximum_gap_chunks,
            )
        key = self._rollout_key(prompt, seed, route_id)
        trajectory_id = self._trajectory_id("rollout", prompt.prompt_id, seed)
        history, restored_metadata, restored_generated = self._load_rollout_prefix(
            key=key,
            trajectory_id=trajectory_id,
            prompt=prompt,
            seed=seed,
        )
        if history.next_chunk_index == 0:
            history, target_root = self._canonical_rollout_root(prompt=prompt, seed=seed)
            root_metadata = {
                "kind": strategy,
                "route_id": route_id,
                "chunk_index": 0,
                "corrected": True,
                "risk_score": None,
                "draft_proposal": None,
                "draft_kv_hash": None,
                "committed_role": "target",
                "committed_kv_hash": target_root.context.kv.cache_sha256,
                "initial_noise_descriptor_sha256": (
                    target_root.context.noise.descriptor_sha256
                ),
            }
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, 0),
                history=history,
                generated=target_root,
                metadata=root_metadata,
            )
            restored_metadata = [root_metadata]
            restored_generated = [target_root]
        else:
            _require(
                self._canonical_rollout_root_checkpoint(prompt, seed).is_file(),
                "resumed rollout is missing its canonical Target root checkpoint",
            )
            canonical_history, target_root = self._canonical_rollout_root(
                prompt=prompt, seed=seed
            )
            self._verify_canonical_rollout_root(
                history=history,
                metadata=restored_metadata,
                generated_chunks=restored_generated,
                canonical_history=canonical_history,
                canonical_generated=target_root,
            )
        proposal_ranks: list[float] = []
        for chunk_index, (metadata, committed) in enumerate(
            zip(restored_metadata, restored_generated, strict=True)
        ):
            _require(
                metadata.get("kind") == strategy
                and metadata.get("route_id") == route_id
                and metadata.get("chunk_index") == chunk_index,
                "resumed online route metadata changed",
            )
            proposal = self._validate_online_draft_proposal(
                chunk_index=chunk_index,
                metadata=metadata,
                committed=committed,
                prefix=history.prefix_before(chunk_index),
            )
            if chunk_index == 0:
                _require(metadata.get("corrected") is True, "online chunk 0 is not Target")
                continue
            _require(proposal is not None, "online Draft proposal is missing")
            proposal_ranks.append(float(proposal.effective_rank))
            risk, erank_drop_score = self._risk_from_rank_history(
                calibrator, proposal_ranks
            )
            _require(
                metadata.get("risk_score") == risk,
                "resumed online risk differs from its Draft proposal",
            )
            if erank_drop_score is not None:
                _require(
                    metadata.get("erank_drop_score") == erank_drop_score,
                    "resumed online eRank-drop score differs from its Draft proposals",
                )
            decision = policy.decide(chunk_index=chunk_index, risk_score=risk)
            _require(
                asdict(decision) == metadata.get("decision")
                and decision.replace_with_target is metadata.get("corrected"),
                "resumed online decision changed",
            )
        state = CorrectionState(history=history)
        for chunk_index in range(history.next_chunk_index, self.chunk_count):
            noise = self._materialize_noise(prompt=prompt, seed=seed, chunk_index=chunk_index)
            if chunk_index == 0:
                target = self._generate(
                    backend=self.target,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=f"latents/rollouts/{key}/chunk-00-target",
                    capture_hidden=False,
                )
                state = CorrectionState(history=state.history.append(target.segment))
                committed = target
                metadata = {
                    "kind": strategy,
                    "route_id": route_id,
                    "chunk_index": 0,
                    "corrected": True,
                    "risk_score": None,
                    "draft_proposal": None,
                    "draft_kv_hash": None,
                    "committed_role": "target",
                    "committed_kv_hash": target.context.kv.cache_sha256,
                    "initial_noise_descriptor_sha256": noise.descriptor.descriptor_sha256,
                }
            else:
                draft = self._generate(
                    backend=self.draft,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=f"latents/rollouts/{key}/chunk-{chunk_index:02d}-draft",
                    # The backend still computes effective rank online; avoid
                    # materializing full hidden-state sidecars for 2(b).
                    capture_hidden=False,
                )
                _require(draft.effective_rank is not None, "online Draft rank is missing")
                proposal_ranks.append(float(draft.effective_rank))
                risk, erank_drop_score = self._risk_from_rank_history(
                    calibrator, proposal_ranks
                )
                decision = policy.decide(chunk_index=chunk_index, risk_score=risk)
                state = state.stage_draft(context=draft.context, segment=draft.segment)
                if decision.replace_with_target:
                    target = self._generate(
                        backend=self.target,
                        prompt=prompt,
                        history=state.history,
                        noise=noise,
                        output_stem=f"latents/rollouts/{key}/chunk-{chunk_index:02d}-target",
                        capture_hidden=False,
                    )
                    state = state.replace_with_target(
                        context=target.context,
                        segment=target.segment,
                    )
                    committed = target
                else:
                    state = state.accept_draft()
                    committed = draft
                metadata = {
                    "kind": strategy,
                    "route_id": route_id,
                    "chunk_index": chunk_index,
                    "corrected": decision.replace_with_target,
                    "risk_score": risk,
                    "rank": float(draft.effective_rank),
                    "decision": asdict(decision),
                    "draft_proposal": _generated_dict(draft),
                    "draft_parent_history_hash": draft.context.history_head_sha256,
                    "draft_kv_hash": draft.context.kv.cache_sha256,
                    "committed_role": committed.context.role,
                    "committed_kv_hash": committed.context.kv.cache_sha256,
                    "initial_noise_descriptor_sha256": noise.descriptor.descriptor_sha256,
                }
                if erank_drop_score is not None:
                    metadata["erank_drop_score"] = erank_drop_score
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, chunk_index),
                history=state.history,
                generated=committed,
                metadata=metadata,
            )
        selected = policy.finalize()
        actual = tuple(
            segment.chunk_index
            for segment in state.history.segments[1:]
            if segment.producer_role == "target"
        )
        _require(actual == selected, "online selection and committed history disagree")
        return key, route_id, selected, state.history

    def _materialize_online_lookahead_coverage_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
    ) -> tuple[str, str, tuple[int, ...], LatentHistory]:
        """Guard each pending Draft with one real successor eRank observation.

        A preview is generated from the provisional history containing the
        pending Draft.  Accepting the pending chunk makes that preview the next
        pending proposal without recomputation.  Replacing the pending chunk
        with Target invalidates the preview's parent history, so the preview is
        retained only as decision evidence and the successor is regenerated
        from the corrected prefix.  No out-of-range sentinel chunk is ever
        generated: the coverage controller must exhaust the exact budget by
        chunk 107, after which preview generation stops and the final real
        chunk is an audit-only terminal flush.  An unconsumed budget at that
        terminal boundary fails closed.
        """

        route_id = f"online-lookahead-coverage-erank-drop-k{budget}"
        policy = LookaheadCoverageConstrainedOnlineRankGuidedPolicy(
            budget=budget,
            calibration=calibrator,
            evaluation_id=(
                f"eval-lookahead-coverage-{prompt.prompt_id}-{seed}-k{budget}"
            ),
            chunk_count=self.chunk_count,
            window_radius_chunks=window_radius_chunks,
            min_gap_chunks=minimum_gap_chunks,
            max_gap_chunks=maximum_gap_chunks,
        )
        strategy = "online_lookahead_coverage_guided"
        key = self._rollout_key(prompt, seed, route_id)
        trajectory_id = self._trajectory_id("rollout", prompt.prompt_id, seed)
        history, restored_metadata, restored_generated = self._load_rollout_prefix(
            key=key,
            trajectory_id=trajectory_id,
            prompt=prompt,
            seed=seed,
        )
        if history.next_chunk_index == 0:
            history, target_root = self._canonical_rollout_root(prompt=prompt, seed=seed)
            root_metadata = {
                "kind": strategy,
                "route_id": route_id,
                "chunk_index": 0,
                "corrected": True,
                "risk_score": None,
                "draft_proposal": None,
                "draft_kv_hash": None,
                "committed_role": "target",
                "committed_kv_hash": target_root.context.kv.cache_sha256,
                "initial_noise_descriptor_sha256": (
                    target_root.context.noise.descriptor_sha256
                ),
                "gate_signal_chunk_index": None,
                "gate_signal_rank": None,
                "gate_probe": None,
                "gate_probe_parent_history_hash": None,
                "gate_probe_kv_hash": None,
                "gate_probe_noise_descriptor_sha256": None,
                "probe_reused_as_successor": False,
                "terminal_flush": False,
            }
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, 0),
                history=history,
                generated=target_root,
                metadata=root_metadata,
            )
            restored_metadata = [root_metadata]
            restored_generated = [target_root]
        else:
            _require(
                self._canonical_rollout_root_checkpoint(prompt, seed).is_file(),
                "resumed rollout is missing its canonical Target root checkpoint",
            )
            canonical_history, target_root = self._canonical_rollout_root(
                prompt=prompt, seed=seed
            )
            self._verify_canonical_rollout_root(
                history=history,
                metadata=restored_metadata,
                generated_chunks=restored_generated,
                canonical_history=canonical_history,
                canonical_generated=target_root,
            )

        resolved_proposal_ranks: list[float] = []
        carried_probe: GeneratedChunk | None = None
        for chunk_index, (metadata, committed) in enumerate(
            zip(restored_metadata, restored_generated, strict=True)
        ):
            _require(
                metadata.get("kind") == strategy
                and metadata.get("route_id") == route_id
                and metadata.get("chunk_index") == chunk_index,
                "resumed lookahead route metadata changed",
            )
            prefix = history.prefix_before(chunk_index)
            proposal = self._validate_online_draft_proposal(
                chunk_index=chunk_index,
                metadata=metadata,
                committed=committed,
                prefix=prefix,
            )
            if chunk_index == 0:
                _require(
                    metadata.get("corrected") is True
                    and metadata.get("gate_probe") is None,
                    "lookahead chunk 0 is not the canonical Target root",
                )
                continue
            _require(proposal is not None, "lookahead pending Draft proposal is missing")
            if carried_probe is not None:
                _require(
                    _generated_dict(proposal) == _generated_dict(carried_probe),
                    "accepted lookahead probe was not reused as its successor proposal",
                )

            pending_rank = float(proposal.effective_rank)
            needs_preview = policy.preview_required(chunk_index=chunk_index)
            terminal_flush = chunk_index == self.chunk_count - 1
            probe: GeneratedChunk | None = None
            signal_chunk_index: int | None = None
            signal_rank: float | None = None
            risk: float | None = None
            erank_drop_score: float | None = None
            if needs_preview:
                provisional = prefix.append(proposal.segment)
                probe = self._validate_lookahead_probe(
                    raw=metadata.get("gate_probe"),
                    chunk_index=chunk_index + 1,
                    prefix=provisional,
                )
                signal_chunk_index = chunk_index + 1
                signal_rank = float(probe.effective_rank)
                risk, erank_drop_score = self._risk_from_rank_history(
                    calibrator,
                    (*resolved_proposal_ranks, pending_rank, signal_rank),
                )
            else:
                _require(
                    metadata.get("gate_probe") is None,
                    "budget-exhausted decision unexpectedly carries a preview",
                )

            decision = policy.decide(
                chunk_index=chunk_index,
                signal_chunk_index=signal_chunk_index,
                risk_score=risk,
            )
            expected_probe_parent = (
                None if probe is None else probe.context.history_head_sha256
            )
            expected_probe_kv = None if probe is None else probe.context.kv.cache_sha256
            expected_probe_noise = (
                None if probe is None else probe.context.noise.descriptor_sha256
            )
            expected_reuse = probe is not None and not decision.replace_with_target
            _require(
                metadata.get("risk_score") == risk
                and metadata.get("decision") == asdict(decision)
                and metadata.get("gate_signal_chunk_index") == signal_chunk_index
                and metadata.get("gate_signal_rank") == signal_rank
                and metadata.get("gate_probe_parent_history_hash")
                == expected_probe_parent
                and metadata.get("gate_probe_kv_hash") == expected_probe_kv
                and metadata.get("gate_probe_noise_descriptor_sha256")
                == expected_probe_noise
                and metadata.get("probe_reused_as_successor") is expected_reuse
                and metadata.get("terminal_flush") is terminal_flush
                and metadata.get("corrected") is decision.replace_with_target,
                "resumed lookahead gate decision or probe binding changed",
            )
            if erank_drop_score is None:
                _require(
                    "erank_drop_score" not in metadata,
                    "lookahead checkpoint has an unexpected eRank-drop score",
                )
            else:
                _require(
                    metadata.get("erank_drop_score") == erank_drop_score,
                    "resumed lookahead eRank-drop score changed",
                )
            if not decision.replace_with_target:
                resolved_proposal_ranks.append(pending_rank)
            carried_probe = probe if expected_reuse else None

        state = CorrectionState(history=history)
        pending = carried_probe
        for chunk_index in range(history.next_chunk_index, self.chunk_count):
            noise = self._materialize_noise(
                prompt=prompt, seed=seed, chunk_index=chunk_index
            )
            if pending is None:
                pending = self._generate(
                    backend=self.draft,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=(
                        f"latents/rollouts/{key}/chunk-{chunk_index:02d}-draft"
                    ),
                    capture_hidden=False,
                )
            else:
                _require(
                    pending.segment.chunk_index == chunk_index
                    and pending.context.history_head_sha256 == state.history.head_sha256
                    and pending.context.noise.to_dict() == noise.descriptor.to_dict(),
                    "carried lookahead probe is not bound to the current successor",
                )
            _require(
                pending.effective_rank is not None,
                "lookahead pending Draft rank is missing",
            )
            state = state.stage_draft(
                context=pending.context,
                segment=pending.segment,
            )
            pending_rank = float(pending.effective_rank)
            needs_preview = policy.preview_required(chunk_index=chunk_index)
            terminal_flush = chunk_index == self.chunk_count - 1
            probe: GeneratedChunk | None = None
            signal_chunk_index: int | None = None
            signal_rank: float | None = None
            risk: float | None = None
            erank_drop_score: float | None = None
            if needs_preview:
                signal_chunk_index = chunk_index + 1
                provisional = state.history.append(pending.segment)
                preview_noise = self._materialize_noise(
                    prompt=prompt,
                    seed=seed,
                    chunk_index=signal_chunk_index,
                )
                probe = self._generate(
                    backend=self.draft,
                    prompt=prompt,
                    history=provisional,
                    noise=preview_noise,
                    output_stem=(
                        f"latents/rollouts/{key}/chunk-{signal_chunk_index:02d}-"
                        f"draft-preview-from-{chunk_index:02d}"
                    ),
                    capture_hidden=False,
                )
                _require(
                    probe.effective_rank is not None,
                    "lookahead preview Draft rank is missing",
                )
                signal_rank = float(probe.effective_rank)
                risk, erank_drop_score = self._risk_from_rank_history(
                    calibrator,
                    (*resolved_proposal_ranks, pending_rank, signal_rank),
                )
            decision = policy.decide(
                chunk_index=chunk_index,
                signal_chunk_index=signal_chunk_index,
                risk_score=risk,
            )
            if decision.replace_with_target:
                target = self._generate(
                    backend=self.target,
                    prompt=prompt,
                    history=state.history,
                    noise=noise,
                    output_stem=(
                        f"latents/rollouts/{key}/chunk-{chunk_index:02d}-target"
                    ),
                    capture_hidden=False,
                )
                state = state.replace_with_target(
                    context=target.context,
                    segment=target.segment,
                )
                committed = target
                next_pending = None
            else:
                state = state.accept_draft()
                committed = pending
                next_pending = probe

            metadata = {
                "kind": strategy,
                "route_id": route_id,
                "chunk_index": chunk_index,
                "corrected": decision.replace_with_target,
                "risk_score": risk,
                "rank": pending_rank,
                "decision": asdict(decision),
                "draft_proposal": _generated_dict(pending),
                "draft_parent_history_hash": pending.context.history_head_sha256,
                "draft_kv_hash": pending.context.kv.cache_sha256,
                "committed_role": committed.context.role,
                "committed_kv_hash": committed.context.kv.cache_sha256,
                "initial_noise_descriptor_sha256": noise.descriptor.descriptor_sha256,
                "gate_signal_chunk_index": signal_chunk_index,
                "gate_signal_rank": signal_rank,
                "gate_probe": None if probe is None else _generated_dict(probe),
                "gate_probe_parent_history_hash": (
                    None if probe is None else probe.context.history_head_sha256
                ),
                "gate_probe_kv_hash": (
                    None if probe is None else probe.context.kv.cache_sha256
                ),
                "gate_probe_noise_descriptor_sha256": (
                    None if probe is None else probe.context.noise.descriptor_sha256
                ),
                "probe_reused_as_successor": (
                    probe is not None and not decision.replace_with_target
                ),
                "terminal_flush": terminal_flush,
            }
            if erank_drop_score is not None:
                metadata["erank_drop_score"] = erank_drop_score
            self._write_checkpoint(
                path=self._checkpoint_path("rollouts", key, chunk_index),
                history=state.history,
                generated=committed,
                metadata=metadata,
            )
            if not decision.replace_with_target:
                resolved_proposal_ranks.append(pending_rank)
            pending = next_pending

        selected = policy.finalize()
        actual = tuple(
            segment.chunk_index
            for segment in state.history.segments[1:]
            if segment.producer_role == "target"
        )
        _require(
            actual == selected,
            "lookahead selection and committed history disagree",
        )
        return key, route_id, selected, state.history

    def rollout_online_rank_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
    ) -> SequentialRollout:
        """Run and synchronously score the deployable online policy."""

        key, route_id, selected, history = self._materialize_online_rank_guided(
            prompt=prompt,
            seed=seed,
            budget=budget,
            calibrator=calibrator,
        )
        return self._score_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def prepare_online_rank_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
    ) -> PendingSequentialRollout:
        """Run the online policy and seal its video without scoring it."""

        key, route_id, selected, history = self._materialize_online_rank_guided(
            prompt=prompt,
            seed=seed,
            budget=budget,
            calibrator=calibrator,
        )
        return self._prepare_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def rollout_online_coverage_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
    ) -> SequentialRollout:
        """Run and synchronously score the causal coverage-constrained policy."""

        key, route_id, selected, history = self._materialize_online_rank_guided(
            prompt=prompt,
            seed=seed,
            budget=budget,
            calibrator=calibrator,
            strategy="online_coverage_guided",
            coverage_window_radius_chunks=window_radius_chunks,
            coverage_minimum_gap_chunks=minimum_gap_chunks,
            coverage_maximum_gap_chunks=maximum_gap_chunks,
        )
        return self._score_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def prepare_online_coverage_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
    ) -> PendingSequentialRollout:
        """Run the coverage-constrained policy and seal its video for scoring."""

        key, route_id, selected, history = self._materialize_online_rank_guided(
            prompt=prompt,
            seed=seed,
            budget=budget,
            calibrator=calibrator,
            strategy="online_coverage_guided",
            coverage_window_radius_chunks=window_radius_chunks,
            coverage_minimum_gap_chunks=minimum_gap_chunks,
            coverage_maximum_gap_chunks=maximum_gap_chunks,
        )
        return self._prepare_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def rollout_online_lookahead_coverage_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
    ) -> SequentialRollout:
        """Run and synchronously score the one-chunk lookahead policy."""

        key, route_id, selected, history = (
            self._materialize_online_lookahead_coverage_guided(
                prompt=prompt,
                seed=seed,
                budget=budget,
                calibrator=calibrator,
                window_radius_chunks=window_radius_chunks,
                minimum_gap_chunks=minimum_gap_chunks,
                maximum_gap_chunks=maximum_gap_chunks,
            )
        )
        return self._score_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def prepare_online_lookahead_coverage_guided(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
    ) -> PendingSequentialRollout:
        """Materialize and seal the one-chunk lookahead policy for scoring."""

        key, route_id, selected, history = (
            self._materialize_online_lookahead_coverage_guided(
                prompt=prompt,
                seed=seed,
                budget=budget,
                calibrator=calibrator,
                window_radius_chunks=window_radius_chunks,
                minimum_gap_chunks=minimum_gap_chunks,
                maximum_gap_chunks=maximum_gap_chunks,
            )
        )
        return self._prepare_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=selected,
            history=history,
        )

    def rollout_online_lookahead_coverage_boundary_gated(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
    ) -> SequentialRollout:
        """Run and score the proposal-budgeted boundary-gated lookahead policy."""

        key, route_id, accepted, proposed, history = (
            self._materialize_online_lookahead_coverage_boundary_gated(
                prompt=prompt,
                seed=seed,
                budget=budget,
                calibrator=calibrator,
                window_radius_chunks=window_radius_chunks,
                minimum_gap_chunks=minimum_gap_chunks,
                maximum_gap_chunks=maximum_gap_chunks,
            )
        )
        return self._score_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=accepted,
            proposed_target_chunks=proposed,
            history=history,
        )

    def prepare_online_lookahead_coverage_boundary_gated(
        self,
        *,
        prompt: PromptCase,
        seed: int,
        budget: int,
        calibrator: RankRiskCalibrator | SuddenDropRiskCalibrator,
        window_radius_chunks: int,
        minimum_gap_chunks: int,
        maximum_gap_chunks: int,
    ) -> PendingSequentialRollout:
        """Materialize and seal the boundary-gated lookahead route."""

        key, route_id, accepted, proposed, history = (
            self._materialize_online_lookahead_coverage_boundary_gated(
                prompt=prompt,
                seed=seed,
                budget=budget,
                calibrator=calibrator,
                window_radius_chunks=window_radius_chunks,
                minimum_gap_chunks=minimum_gap_chunks,
                maximum_gap_chunks=maximum_gap_chunks,
            )
        )
        return self._prepare_rollout(
            key=key,
            route_id=route_id,
            corrected_chunks=accepted,
            proposed_target_chunks=proposed,
            history=history,
        )

    def _append_strategy_row(
        self,
        *,
        split: str,
        prompt: PromptCase,
        seed: int,
        budget: int,
        strategy: str,
        route_repeat: int | None,
        rollout: SequentialRollout,
        target_rpc_counts: Mapping[str, int],
        route_materialization_rpc_counts: Mapping[str, int],
        selection_search_rollout_count: int = 0,
        selection_search_rpc_counts: Mapping[str, int] | None = None,
    ) -> None:
        event_id = self._row_id(
            "2b", split, prompt.prompt_id, seed, budget, strategy, route_repeat
        )
        proposed_chunks = tuple(rollout.proposed_target_chunks or ())
        committed_calls = 1 + len(rollout.result.corrected_chunks)
        budgeted_calls = 1 + len(proposed_chunks)
        if strategy == "online_lookahead_coverage_boundary_gated":
            _require(
                len(proposed_chunks) == budget
                and set(rollout.result.corrected_chunks).issubset(proposed_chunks),
                "boundary-gated raw row proposal budget is invalid",
            )
        else:
            _require(
                proposed_chunks == rollout.result.corrected_chunks
                and len(proposed_chunks) == budget,
                "legacy raw row proposal budget differs from committed corrections",
            )
        expected_count_keys = {
            "attempted",
            "backend_returned",
            "completed",
            "logical_completed",
        }
        _require(
            set(target_rpc_counts) == expected_count_keys
            and set(route_materialization_rpc_counts) == expected_count_keys,
            "Target RPC lifecycle count fields are incomplete",
        )
        attempted_calls = target_rpc_counts["attempted"]
        returned_calls = target_rpc_counts["backend_returned"]
        physical_calls = target_rpc_counts["completed"]
        route_logical_calls = route_materialization_rpc_counts["logical_completed"]
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (
                    attempted_calls,
                    returned_calls,
                    physical_calls,
                    route_logical_calls,
                )
            )
            and physical_calls <= returned_calls <= attempted_calls,
            "Target RPC lifecycle counts are invalid",
        )
        search_counts = selection_search_rpc_counts or {
            "attempted": 0,
            "backend_returned": 0,
            "completed": 0,
            "logical_completed": 0,
        }
        _require(
            set(search_counts) == expected_count_keys
            and selection_search_rollout_count >= 0
            and 0 <= search_counts["completed"]
            <= search_counts["backend_returned"]
            <= search_counts["attempted"]
            and search_counts["attempted"] <= attempted_calls
            and search_counts["backend_returned"] <= returned_calls,
            "selection-search audit counts cannot be negative",
        )
        _require(
            0
            <= route_logical_calls
            <= route_materialization_rpc_counts["completed"]
            <= route_materialization_rpc_counts["backend_returned"]
            <= route_materialization_rpc_counts["attempted"]
            and route_materialization_rpc_counts["attempted"]
            + search_counts["attempted"]
            == attempted_calls
            and route_materialization_rpc_counts["backend_returned"]
            + search_counts["backend_returned"]
            == returned_calls
            and route_materialization_rpc_counts["completed"]
            + search_counts["completed"]
            == physical_calls
            and route_logical_calls <= budgeted_calls,
            "route-materialization Target RPC counts are invalid",
        )
        self.raw_2b.append(
            {
                "schema": RAW_2B_SCHEMA,
                "event_id": event_id,
                "split": split,
                "prompt_id": prompt.prompt_id,
                "seed": seed,
                "budget": budget,
                "strategy": strategy,
                "route_repeat": route_repeat,
                "selected_chunks": list(rollout.result.corrected_chunks),
                "proposed_target_chunks": list(proposed_chunks),
                "full_quality": rollout.result.full_video_quality,
                "temporal_consistency": rollout.temporal_consistency,
                # Includes the mandatory Target chunk 0 and the K committed
                # corrections.  The analysis schema intentionally rejects K.
                "committed_target_chunks": committed_calls,
                "budgeted_target_calls": budgeted_calls,
                # Actual successful Target generation RPCs made while
                # materializing this row.  This may be zero on a verified
                # checkpoint/cache hit; committed_target_chunks remains the
                # invariant scientific budget.
                "physical_target_calls": physical_calls,
                "target_rpc_attempted": attempted_calls,
                "target_rpc_backend_returned": returned_calls,
                "target_rpc_completed": physical_calls,
                # Physical completions may exceed 1+K after a response or
                # checkpoint loss.  Cache-hit accounting therefore uses only
                # distinct logical calls completed in the route phase.
                "route_materialized_logical_target_calls": route_logical_calls,
                # Greedy Oracle is an offline search, not a budget-free
                # deployed policy.  These fields expose its search separately
                # instead of hiding it in the committed 1+K route budget.
                "selection_search_rollout_count": selection_search_rollout_count,
                "selection_search_target_calls": search_counts["completed"],
                "selection_search_target_rpc_attempted": search_counts["attempted"],
                "selection_search_target_rpc_backend_returned": search_counts[
                    "backend_returned"
                ],
                "selection_search_target_rpc_completed": search_counts["completed"],
                "final_history_hash": rollout.history.head_sha256,
                "rollout_id": rollout.result.rollout_id,
            }
        )

    def prepare_multi_budget(
        self,
        *,
        split: str = "evaluation",
        budgets: Sequence[int],
        random_repeats: int,
        strategies: Sequence[str] = (
            "random",
            "uniform",
            "online_rank_guided",
        ),
        coverage_window_radius_chunks: int | None = None,
        coverage_minimum_gap_chunks: int | None = None,
        coverage_maximum_gap_chunks: int | None = None,
    ) -> PendingMultiBudgetPlan:
        """Generate non-oracle 2(b) routes and seal videos without scoring.

        This entry point is intentionally separate from ``run_multi_budget``;
        legacy callers therefore retain their exact synchronous behavior.  A
        v3 runner can call this method while model workers are alive, close the
        runtime, score each unique ``prepared_quality`` on either lane GPU, and
        finally call ``finalize_multi_budget`` in deterministic plan order.
        """

        _require(split in {"smoke", "evaluation"}, "2(b) split must be smoke or evaluation")
        parsed_budgets = tuple(budgets)
        _require(
            parsed_budgets
            and len(parsed_budgets) == len(set(parsed_budgets))
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 1 <= item < self.chunk_count
                for item in parsed_budgets
            ),
            f"budgets must be unique integers in [1,{self.chunk_count - 1}]",
        )
        _require(
            isinstance(random_repeats, int)
            and not isinstance(random_repeats, bool)
            and random_repeats >= 0,
            "random_repeats must be non-negative",
        )
        parsed_strategies = tuple(strategies)
        base_strategies = ("random", "uniform", "online_rank_guided")
        coverage_strategies = (*base_strategies, "online_coverage_guided")
        lookahead_strategies = (
            *coverage_strategies,
            "online_lookahead_coverage_guided",
        )
        boundary_strategies = (
            "uniform",
            "online_lookahead_coverage_guided",
            "online_lookahead_coverage_boundary_gated",
        )
        _require(
            parsed_strategies
            in {
                base_strategies,
                coverage_strategies,
                lookahead_strategies,
                boundary_strategies,
            }
            and (
                (random_repeats >= 1)
                if "random" in parsed_strategies
                else random_repeats == 0
            ),
            "deferred strategies must match a supported online policy profile",
        )
        coverage_values = (
            coverage_window_radius_chunks,
            coverage_minimum_gap_chunks,
            coverage_maximum_gap_chunks,
        )
        if parsed_strategies in {
            coverage_strategies,
            lookahead_strategies,
            boundary_strategies,
        }:
            _require(
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 1
                    for value in coverage_values
                ),
                "coverage strategy requires positive coverage parameters",
            )
        else:
            _require(
                coverage_values == (None, None, None),
                "baseline strategies cannot carry coverage parameters",
            )
        calibrator = self.run_calibration()
        prompts = self.smoke_prompts if split == "smoke" else self.evaluation_prompts
        entries: list[PendingStrategyRow] = []

        def append_pending(
            *,
            event_id: str,
            prompt: PromptCase,
            seed: int,
            budget: int,
            strategy: str,
            route_repeat: int | None,
            rollout: PendingSequentialRollout,
        ) -> None:
            _require(
                not self.raw_2b.has(event_id),
                "deferred plan cannot replace an already finalized raw row",
            )
            entries.append(
                PendingStrategyRow(
                    event_id=event_id,
                    split=split,
                    prompt_id=prompt.prompt_id,
                    seed=seed,
                    budget=budget,
                    strategy=strategy,
                    route_repeat=route_repeat,
                    rollout=rollout,
                    target_rpc_counts=_freeze_rpc_counts(
                        self._target_rpc_lifecycle_counts(event_id),
                        label="pending target_rpc_counts",
                    ),
                    route_materialization_rpc_counts=_freeze_rpc_counts(
                        self._target_rpc_lifecycle_counts(
                            event_id, phase="route_materialization"
                        ),
                        label="pending route_materialization_rpc_counts",
                    ),
                )
            )

        for prompt in prompts:
            for seed in self.seeds:
                for budget in parsed_budgets:
                    for repeat in range(random_repeats):
                        event_id = self._row_id(
                            "2b", split, prompt.prompt_id, seed, budget, "random", repeat
                        )
                        selected = random_positions(
                            protocol_id=self.protocol_id,
                            prompt_id=prompt.prompt_id,
                            generation_seed=seed,
                            budget=budget,
                            repeat_id=repeat,
                            chunk_count=self.chunk_count,
                        )
                        with self._target_rpc_scope(event_id):
                            rollout = self.prepare_fixed_positions(
                                prompt=prompt,
                                seed=seed,
                                corrected_chunks=selected,
                            )
                            append_pending(
                                event_id=event_id,
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                strategy="random",
                                route_repeat=repeat,
                                rollout=rollout,
                            )

                    uniform_event = self._row_id(
                        "2b", split, prompt.prompt_id, seed, budget, "uniform", None
                    )
                    with self._target_rpc_scope(uniform_event):
                        uniform = self.prepare_fixed_positions(
                            prompt=prompt,
                            seed=seed,
                            corrected_chunks=uniform_positions(
                                budget, chunk_count=self.chunk_count
                            ),
                        )
                        append_pending(
                            event_id=uniform_event,
                            prompt=prompt,
                            seed=seed,
                            budget=budget,
                            strategy="uniform",
                            route_repeat=None,
                            rollout=uniform,
                        )

                    if "online_rank_guided" in parsed_strategies:
                        online_strategy = "online_rank_guided"
                        rank_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            online_strategy,
                            None,
                        )
                        with self._target_rpc_scope(rank_event):
                            rank_guided = self.prepare_online_rank_guided(
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                calibrator=calibrator,
                            )
                            append_pending(
                                event_id=rank_event,
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                strategy=online_strategy,
                                route_repeat=None,
                                rollout=rank_guided,
                            )

                    if "online_coverage_guided" in parsed_strategies:
                        coverage_strategy = "online_coverage_guided"
                        coverage_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            coverage_strategy,
                            None,
                        )
                        with self._target_rpc_scope(coverage_event):
                            coverage_guided = self.prepare_online_coverage_guided(
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                calibrator=calibrator,
                                window_radius_chunks=coverage_window_radius_chunks,
                                minimum_gap_chunks=coverage_minimum_gap_chunks,
                                maximum_gap_chunks=coverage_maximum_gap_chunks,
                            )
                            append_pending(
                                event_id=coverage_event,
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                strategy=coverage_strategy,
                                route_repeat=None,
                                rollout=coverage_guided,
                            )

                    if "online_lookahead_coverage_guided" in parsed_strategies:
                        lookahead_strategy = "online_lookahead_coverage_guided"
                        lookahead_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            lookahead_strategy,
                            None,
                        )
                        with self._target_rpc_scope(lookahead_event):
                            lookahead_guided = (
                                self.prepare_online_lookahead_coverage_guided(
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    calibrator=calibrator,
                                    window_radius_chunks=(
                                        coverage_window_radius_chunks
                                    ),
                                    minimum_gap_chunks=(
                                        coverage_minimum_gap_chunks
                                    ),
                                    maximum_gap_chunks=(
                                        coverage_maximum_gap_chunks
                                    ),
                                )
                            )
                            append_pending(
                                event_id=lookahead_event,
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                strategy=lookahead_strategy,
                                route_repeat=None,
                                rollout=lookahead_guided,
                            )

                    if (
                        "online_lookahead_coverage_boundary_gated"
                        in parsed_strategies
                    ):
                        boundary_strategy = (
                            "online_lookahead_coverage_boundary_gated"
                        )
                        boundary_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            boundary_strategy,
                            None,
                        )
                        with self._target_rpc_scope(boundary_event):
                            boundary_guided = (
                                self.prepare_online_lookahead_coverage_boundary_gated(
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    calibrator=calibrator,
                                    window_radius_chunks=(
                                        coverage_window_radius_chunks
                                    ),
                                    minimum_gap_chunks=(
                                        coverage_minimum_gap_chunks
                                    ),
                                    maximum_gap_chunks=(
                                        coverage_maximum_gap_chunks
                                    ),
                                )
                            )
                            append_pending(
                                event_id=boundary_event,
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                strategy=boundary_strategy,
                                route_repeat=None,
                                rollout=boundary_guided,
                            )

        manifest_relative = f"state/pending/insight2b-{split}.json"
        plan = PendingMultiBudgetPlan(
            manifest_path=manifest_relative,
            protocol_id=self.protocol_id,
            split=split,
            budgets=parsed_budgets,
            random_repeats=random_repeats,
            entries=tuple(entries),
        )
        _write_immutable_json(self.run_dir / manifest_relative, plan.to_dict())
        return plan

    def _finalize_pending_rollout(
        self,
        *,
        pending: PendingSequentialRollout,
        quality: QualityMeasurement | None,
    ) -> SequentialRollout:
        destination = (
            self.run_dir
            / "measurements"
            / "rollouts"
            / f"{pending.route_key}.json"
        )
        if destination.exists():
            value = _load_json_object(destination)
            _require(
                value.get("schema") == ROLLOUT_SCORE_SCHEMA
                and value.get("route_id") == pending.route_id
                and tuple(value.get("corrected_chunks", ()))
                == pending.corrected_chunks
                and tuple(value.get("proposed_target_chunks", ()))
                == pending.proposed_target_chunks
                and value.get("target_call_count")
                == 1 + len(pending.corrected_chunks)
                and value.get("budgeted_target_call_count")
                == 1 + len(pending.proposed_target_chunks or ())
                and value.get("final_history_hash") == pending.history.head_sha256,
                "completed rollout score differs from its pending route",
            )
            frozen_quality = QualityMeasurement(
                primary=value["full_video_quality"],
                temporal_consistency=value["temporal_consistency"],
            )
            if quality is not None:
                _require(
                    quality == frozen_quality,
                    "provided quality differs from the immutable rollout score",
                )
            quality = frozen_quality
        else:
            _require(
                isinstance(quality, QualityMeasurement),
                "pending rollout has no completed quality measurement",
            )
            _write_immutable_json(
                destination,
                {
                    "schema": ROLLOUT_SCORE_SCHEMA,
                    "route_id": pending.route_id,
                    "corrected_chunks": list(pending.corrected_chunks),
                    "proposed_target_chunks": list(
                        pending.proposed_target_chunks or ()
                    ),
                    "full_video_quality": quality.primary,
                    "temporal_consistency": quality.temporal_consistency,
                    "target_call_count": 1 + len(pending.corrected_chunks),
                    "budgeted_target_call_count": 1
                    + len(pending.proposed_target_chunks or ()),
                    "final_history_hash": pending.history.head_sha256,
                },
            )
        assert quality is not None
        return SequentialRollout(
            result=FullVideoRollout(
                corrected_chunks=pending.corrected_chunks,
                full_video_quality=quality.primary,
                target_call_count=1 + len(pending.corrected_chunks),
                rollout_id=pending.route_id,
                chunk_count=self.chunk_count,
            ),
            temporal_consistency=quality.temporal_consistency,
            history=pending.history,
            proposed_target_chunks=pending.proposed_target_chunks,
        )

    def finalize_multi_budget(
        self,
        *,
        plan: PendingMultiBudgetPlan,
        qualities: Mapping[str, QualityMeasurement],
    ) -> tuple[dict[str, Any], ...]:
        """Append raw rows after all prepared routes have official scores."""

        _require(
            plan.protocol_id == self.protocol_id,
            "pending plan belongs to another protocol",
        )
        manifest = _load_json_object(self.run_dir / plan.manifest_path)
        _require(
            manifest == plan.to_dict(),
            "pending plan differs from its immutable generation-complete manifest",
        )
        unique_measurements = {
            entry.rollout.prepared_quality.measurement_id for entry in plan.entries
        }
        _require(
            set(qualities) == unique_measurements
            and all(isinstance(value, QualityMeasurement) for value in qualities.values()),
            "completed qualities do not exactly cover the pending unique routes",
        )
        prompt_by_id = {
            prompt.prompt_id: prompt
            for prompt in (*self.calibration_prompts, *self.evaluation_prompts)
        }
        for entry in plan.entries:
            _require(
                entry.event_id
                == self._row_id(
                    "2b",
                    entry.split,
                    entry.prompt_id,
                    entry.seed,
                    entry.budget,
                    entry.strategy,
                    entry.route_repeat,
                ),
                "pending strategy event identity drifted",
            )
            rollout = self._finalize_pending_rollout(
                pending=entry.rollout,
                quality=qualities[entry.rollout.prepared_quality.measurement_id],
            )
            self._append_strategy_row(
                split=entry.split,
                prompt=prompt_by_id[entry.prompt_id],
                seed=entry.seed,
                budget=entry.budget,
                strategy=entry.strategy,
                route_repeat=entry.route_repeat,
                rollout=rollout,
                target_rpc_counts=_thaw_rpc_counts(entry.target_rpc_counts),
                route_materialization_rpc_counts=_thaw_rpc_counts(
                    entry.route_materialization_rpc_counts
                ),
                selection_search_rollout_count=entry.selection_search_rollout_count,
                selection_search_rpc_counts=_thaw_rpc_counts(
                    entry.selection_search_rpc_counts
                ),
            )
        selected_ids = {
            item.prompt_id
            for item in (
                self.smoke_prompts if plan.split == "smoke" else self.evaluation_prompts
            )
        }
        return tuple(
            row
            for row in self.raw_2b.records()
            if row.get("split") == plan.split
            and row.get("prompt_id") in selected_ids
        )

    def run_multi_budget(
        self,
        *,
        split: str = "evaluation",
        budgets: Sequence[int] = (2, 3, 4),
        random_repeats: int = 10,
        strategies: Sequence[str] = (
            "random",
            "uniform",
            "online_rank_guided",
            "greedy_oracle",
        ),
        coverage_window_radius_chunks: int | None = None,
        coverage_minimum_gap_chunks: int | None = None,
        coverage_maximum_gap_chunks: int | None = None,
        include_greedy_oracle: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        """Run Random, Uniform, online Rank-guided, and full-rollout Greedy Oracle."""

        _require(split in {"smoke", "evaluation"}, "2(b) split must be smoke or evaluation")
        parsed_budgets = tuple(budgets)
        _require(
            parsed_budgets
            and len(parsed_budgets) == len(set(parsed_budgets))
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 1 <= item < self.chunk_count
                for item in parsed_budgets
            ),
            f"budgets must be unique integers in [1,{self.chunk_count - 1}]",
        )
        _require(
            isinstance(random_repeats, int)
            and not isinstance(random_repeats, bool)
            and random_repeats >= 0,
            "random_repeats must be non-negative",
        )
        _require(
            isinstance(include_greedy_oracle, bool),
            "include_greedy_oracle must be boolean",
        )
        parsed_strategies = tuple(strategies)
        legacy_strategies = (
            "random",
            "uniform",
            "online_rank_guided",
            "greedy_oracle",
        )
        long_strategies = ("random", "uniform", "online_rank_guided")
        coverage_strategies = (*long_strategies, "online_coverage_guided")
        lookahead_strategies = (
            *coverage_strategies,
            "online_lookahead_coverage_guided",
        )
        boundary_strategies = (
            "uniform",
            "online_lookahead_coverage_guided",
            "online_lookahead_coverage_boundary_gated",
        )
        _require(
            parsed_strategies
            in {
                legacy_strategies,
                long_strategies,
                coverage_strategies,
                lookahead_strategies,
                boundary_strategies,
            },
            "strategies must match a supported policy profile",
        )
        _require(
            include_greedy_oracle == ("greedy_oracle" in parsed_strategies),
            "include_greedy_oracle differs from strategies",
        )
        coverage_values = (
            coverage_window_radius_chunks,
            coverage_minimum_gap_chunks,
            coverage_maximum_gap_chunks,
        )
        _require(
            ((random_repeats >= 1) if "random" in parsed_strategies else random_repeats == 0),
            "random_repeats differs from the selected strategy profile",
        )
        if parsed_strategies in {
            coverage_strategies,
            lookahead_strategies,
            boundary_strategies,
        }:
            _require(
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 1
                    for value in coverage_values
                ),
                "coverage strategy requires positive coverage parameters",
            )
        else:
            _require(
                coverage_values == (None, None, None),
                "baseline strategies cannot carry coverage parameters",
            )
        calibrator = self.run_calibration()
        prompts = self.smoke_prompts if split == "smoke" else self.evaluation_prompts
        for prompt in prompts:
            for seed in self.seeds:
                for budget in parsed_budgets:
                    for repeat in range(random_repeats):
                        event_id = self._row_id(
                            "2b", split, prompt.prompt_id, seed, budget, "random", repeat
                        )
                        if not self.raw_2b.has(event_id):
                            selected = random_positions(
                                protocol_id=self.protocol_id,
                                prompt_id=prompt.prompt_id,
                                generation_seed=seed,
                                budget=budget,
                                repeat_id=repeat,
                                chunk_count=self.chunk_count,
                            )
                            with self._target_rpc_scope(event_id):
                                rollout = self.rollout_fixed_positions(
                                    prompt=prompt,
                                    seed=seed,
                                    corrected_chunks=selected,
                                )
                                self._append_strategy_row(
                                    split=split,
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    strategy="random",
                                    route_repeat=repeat,
                                    rollout=rollout,
                                    target_rpc_counts=self._target_rpc_lifecycle_counts(
                                        event_id
                                    ),
                                    route_materialization_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            event_id, phase="route_materialization"
                                        )
                                    ),
                                )

                    uniform_event = self._row_id(
                        "2b", split, prompt.prompt_id, seed, budget, "uniform", None
                    )
                    if not self.raw_2b.has(uniform_event):
                        with self._target_rpc_scope(uniform_event):
                            uniform = self.rollout_fixed_positions(
                                prompt=prompt,
                                seed=seed,
                                corrected_chunks=uniform_positions(
                                    budget, chunk_count=self.chunk_count
                                ),
                            )
                            self._append_strategy_row(
                                split=split,
                                prompt=prompt,
                                seed=seed,
                                budget=budget,
                                strategy="uniform",
                                route_repeat=None,
                                rollout=uniform,
                                target_rpc_counts=self._target_rpc_lifecycle_counts(
                                    uniform_event
                                ),
                                route_materialization_rpc_counts=(
                                    self._target_rpc_lifecycle_counts(
                                        uniform_event, phase="route_materialization"
                                    )
                                ),
                            )

                    if "online_rank_guided" in parsed_strategies:
                        online_strategy = "online_rank_guided"
                        rank_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            online_strategy,
                            None,
                        )
                        if not self.raw_2b.has(rank_event):
                            with self._target_rpc_scope(rank_event):
                                rank_guided = self.rollout_online_rank_guided(
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    calibrator=calibrator,
                                )
                                self._append_strategy_row(
                                    split=split,
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    strategy=online_strategy,
                                    route_repeat=None,
                                    rollout=rank_guided,
                                    target_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(rank_event)
                                    ),
                                    route_materialization_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            rank_event, phase="route_materialization"
                                        )
                                    ),
                                )

                    if "online_coverage_guided" in parsed_strategies:
                        coverage_strategy = "online_coverage_guided"
                        coverage_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            coverage_strategy,
                            None,
                        )
                        if not self.raw_2b.has(coverage_event):
                            with self._target_rpc_scope(coverage_event):
                                coverage_guided = self.rollout_online_coverage_guided(
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    calibrator=calibrator,
                                    window_radius_chunks=coverage_window_radius_chunks,
                                    minimum_gap_chunks=coverage_minimum_gap_chunks,
                                    maximum_gap_chunks=coverage_maximum_gap_chunks,
                                )
                                self._append_strategy_row(
                                    split=split,
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    strategy=coverage_strategy,
                                    route_repeat=None,
                                    rollout=coverage_guided,
                                    target_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            coverage_event
                                        )
                                    ),
                                    route_materialization_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            coverage_event,
                                            phase="route_materialization",
                                        )
                                    ),
                                )

                    if "online_lookahead_coverage_guided" in parsed_strategies:
                        lookahead_strategy = "online_lookahead_coverage_guided"
                        lookahead_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            lookahead_strategy,
                            None,
                        )
                        if not self.raw_2b.has(lookahead_event):
                            with self._target_rpc_scope(lookahead_event):
                                lookahead_guided = (
                                    self.rollout_online_lookahead_coverage_guided(
                                        prompt=prompt,
                                        seed=seed,
                                        budget=budget,
                                        calibrator=calibrator,
                                        window_radius_chunks=(
                                            coverage_window_radius_chunks
                                        ),
                                        minimum_gap_chunks=(
                                            coverage_minimum_gap_chunks
                                        ),
                                        maximum_gap_chunks=(
                                            coverage_maximum_gap_chunks
                                        ),
                                    )
                                )
                                self._append_strategy_row(
                                    split=split,
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    strategy=lookahead_strategy,
                                    route_repeat=None,
                                    rollout=lookahead_guided,
                                    target_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            lookahead_event
                                        )
                                    ),
                                    route_materialization_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            lookahead_event,
                                            phase="route_materialization",
                                        )
                                    ),
                                )

                    if (
                        "online_lookahead_coverage_boundary_gated"
                        in parsed_strategies
                    ):
                        boundary_strategy = (
                            "online_lookahead_coverage_boundary_gated"
                        )
                        boundary_event = self._row_id(
                            "2b",
                            split,
                            prompt.prompt_id,
                            seed,
                            budget,
                            boundary_strategy,
                            None,
                        )
                        if not self.raw_2b.has(boundary_event):
                            with self._target_rpc_scope(boundary_event):
                                boundary_guided = (
                                    self.rollout_online_lookahead_coverage_boundary_gated(
                                        prompt=prompt,
                                        seed=seed,
                                        budget=budget,
                                        calibrator=calibrator,
                                        window_radius_chunks=(
                                            coverage_window_radius_chunks
                                        ),
                                        minimum_gap_chunks=(
                                            coverage_minimum_gap_chunks
                                        ),
                                        maximum_gap_chunks=(
                                            coverage_maximum_gap_chunks
                                        ),
                                    )
                                )
                                self._append_strategy_row(
                                    split=split,
                                    prompt=prompt,
                                    seed=seed,
                                    budget=budget,
                                    strategy=boundary_strategy,
                                    route_repeat=None,
                                    rollout=boundary_guided,
                                    target_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            boundary_event
                                        )
                                    ),
                                    route_materialization_rpc_counts=(
                                        self._target_rpc_lifecycle_counts(
                                            boundary_event,
                                            phase="route_materialization",
                                        )
                                    ),
                                )

                    oracle_event = self._row_id(
                        "2b", split, prompt.prompt_id, seed, budget, "greedy_oracle", None
                    )
                    if include_greedy_oracle and not self.raw_2b.has(oracle_event):
                        with self._target_rpc_scope(
                            oracle_event, phase="selection_search"
                        ):
                            cache: dict[tuple[int, ...], SequentialRollout] = {}

                            def scorer(chunks: tuple[int, ...]) -> FullVideoRollout:
                                if chunks not in cache:
                                    cache[chunks] = self.rollout_fixed_positions(
                                        prompt=prompt,
                                        seed=seed,
                                        corrected_chunks=chunks,
                                    )
                                return cache[chunks].result

                            plan = greedy_oracle_plan(
                                budget=budget,
                                rollout_scorer=scorer,
                                chunk_count=self.chunk_count,
                            )
                            search_rpc_counts = self._target_rpc_lifecycle_counts(
                                oracle_event, phase="selection_search"
                            )
                        selected = plan.selected_chunks
                        if selected not in cache:
                            with self._target_rpc_scope(
                                oracle_event, phase="route_materialization"
                            ):
                                cache[selected] = self.rollout_fixed_positions(
                                    prompt=prompt,
                                    seed=seed,
                                    corrected_chunks=selected,
                                )
                        self._append_strategy_row(
                            split=split,
                            prompt=prompt,
                            seed=seed,
                            budget=budget,
                            strategy="greedy_oracle",
                            route_repeat=None,
                            rollout=cache[selected],
                            target_rpc_counts=self._target_rpc_lifecycle_counts(
                                oracle_event
                            ),
                            route_materialization_rpc_counts=(
                                self._target_rpc_lifecycle_counts(
                                    oracle_event, phase="route_materialization"
                                )
                            ),
                            selection_search_rollout_count=len(plan.evaluations),
                            selection_search_rpc_counts=search_rpc_counts,
                        )
        selected_ids = {item.prompt_id for item in prompts}
        return tuple(
            row
            for row in self.raw_2b.records()
            if row.get("split") == split and row.get("prompt_id") in selected_ids
        )


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CALIBRATION_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "FORK_EVIDENCE_SCHEMA",
    "NOISE_NAMESPACE",
    "PENDING_MULTI_BUDGET_PLAN_SCHEMA",
    "PENDING_ROLLOUT_SCHEMA",
    "PREPARED_FULL_QUALITY_SCHEMA",
    "RAW_2A_SCHEMA",
    "RAW_2B_SCHEMA",
    "TARGET_RPC_BACKEND_DIAGNOSTIC_KEYS",
    "TARGET_RPC_DIAGNOSTICS_SCHEMA",
    "TARGET_RPC_LIFECYCLE_SCHEMA",
    "TARGET_RPC_SCHEMA",
    "AppendOnlyJsonl",
    "BaselineTrajectory",
    "CampaignError",
    "ChunkBackendClient",
    "ChunkGenerationRequest",
    "GeneratedChunk",
    "Insight2Campaign",
    "NoiseMaterialization",
    "NoiseProvider",
    "PendingMultiBudgetPlan",
    "PendingSequentialRollout",
    "PendingStrategyRow",
    "PreparedFullQuality",
    "PromptCase",
    "QualityEvaluator",
    "QualityMeasurement",
    "SequentialRollout",
    "SUPPORTED_CHUNK_COUNTS",
]
