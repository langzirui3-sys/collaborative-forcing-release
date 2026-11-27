"""Frozen, CPU-only selection policies for the Insight 2 experiments.

The functions in this module deliberately know nothing about model execution.
They turn already-recorded representation risks into correction positions, or
ask a caller-owned *full rollout* scorer to evaluate a correction set.  This
separation makes it difficult to accidentally leak future risks into the
online policy or to report a sum of 1+1 gains as a multi-point experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


FIRST_TARGET_CHUNK = 0
DEFAULT_CHUNK_COUNT = 16
CANDIDATE_CHUNKS = tuple(range(1, 16))
RANDOM_SELECTION_HASH_FIELDS = (
    "protocol_id",
    "prompt_id",
    "generation_seed",
    "K",
    "repeat_id",
    "chunk_index",
)
BOUNDARY_GATE_METRIC_ID = "paired_successor_erank_boundary_witness"
BOUNDARY_GATE_METRIC_VERSION = "v1"
BOUNDARY_GATE_ERANK_DELTA_THRESHOLD = 0.0
BOUNDARY_GATE_ENTRY_DELTA_THRESHOLD = 0.0
BOUNDARY_GATE_EXIT_DELTA_THRESHOLD = 0.0
BOUNDARY_GATE_MIN_ERANK = 1.0
BOUNDARY_GATE_MAX_ERANK = 256.0
BOUNDARY_GATE_MIN_WITNESS_SCORE = 0.0
BOUNDARY_GATE_MAX_WITNESS_SCORE = 1.0


class PolicyContractError(ValueError):
    """Raised when an Insight 2 selection contract would be ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyContractError(message)


def _require_nonempty_string(value: object, field: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{field} must be a non-empty string")
    return value


def _require_integer(value: object, field: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{field} must be an integer >= {minimum}",
    )
    return value


def _require_finite(value: object, field: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    return float(value)


def candidate_chunks(chunk_count: int = DEFAULT_CHUNK_COUNT) -> tuple[int, ...]:
    """Return correction candidates for a trajectory whose chunk 0 is Target."""

    parsed_count = _require_integer(chunk_count, "chunk_count", minimum=2)
    return tuple(range(1, parsed_count))


def _validate_budget(
    budget: object, *, chunk_count: int = DEFAULT_CHUNK_COUNT
) -> int:
    parsed = _require_integer(budget, "budget", minimum=1)
    candidates = candidate_chunks(chunk_count)
    _require(
        parsed <= len(candidates),
        f"budget exceeds the {len(candidates)} candidate chunks",
    )
    return parsed


def _validate_chunk_tuple(
    chunks: Sequence[int],
    field: str,
    *,
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> tuple[int, ...]:
    _require(
        isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)),
        f"{field} must be a sequence",
    )
    candidates = candidate_chunks(chunk_count)
    parsed: list[int] = []
    for index, chunk in enumerate(chunks):
        _require_integer(chunk, f"{field}[{index}]", minimum=1)
        _require(
            chunk in candidates,
            f"{field}[{index}] is not in chunk 1..{chunk_count - 1}",
        )
        parsed.append(chunk)
    _require(len(parsed) == len(set(parsed)), f"{field} contains duplicate chunks")
    _require(parsed == sorted(parsed), f"{field} must be in chronological order")
    return tuple(parsed)


def _complete_chunk_values(
    values_by_chunk: Mapping[int, float],
    field: str,
    *,
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> dict[int, float]:
    candidates = candidate_chunks(chunk_count)
    _require(isinstance(values_by_chunk, Mapping), f"{field} must be a mapping")
    _require(
        set(values_by_chunk) == set(candidates),
        f"{field} must contain each candidate chunk 1..{chunk_count - 1} exactly once",
    )
    return {
        chunk: _require_finite(values_by_chunk[chunk], f"{field}[{chunk}]")
        for chunk in candidates
    }


def uniform_positions(
    budget: int, *, chunk_count: int = DEFAULT_CHUNK_COUNT
) -> tuple[int, ...]:
    """Return the pre-registered, outcome-independent half-up uniform grid.

    For selection ordinal ``j=1..K``, the position is
    ``floor(j * chunk_count / (K + 1) + 0.5)``.  With the legacy 16 chunks,
    K=2,3,4 yield ``(5, 11)``, ``(4, 8, 12)``, and ``(3, 6, 10, 13)``.
    """

    parsed_budget = _validate_budget(budget, chunk_count=chunk_count)
    positions = tuple(
        math.floor(ordinal * chunk_count / (parsed_budget + 1) + 0.5)
        for ordinal in range(1, parsed_budget + 1)
    )
    _validate_chunk_tuple(positions, "uniform positions", chunk_count=chunk_count)
    _require(len(positions) == parsed_budget, "uniform rule did not produce exactly K chunks")
    return positions


def random_selection_digest(
    *,
    protocol_id: str,
    prompt_id: str,
    generation_seed: int,
    budget: int,
    repeat_id: int,
    chunk_index: int,
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> str:
    """Hash the six frozen Random-policy fields with an unambiguous encoding.

    The digest uses no process PRNG and no generation-noise namespace.  Model
    noise derivation can therefore change without changing a Random-policy
    assignment (and vice versa).
    """

    parsed_protocol = _require_nonempty_string(protocol_id, "protocol_id")
    parsed_prompt = _require_nonempty_string(prompt_id, "prompt_id")
    parsed_seed = _require_integer(generation_seed, "generation_seed")
    parsed_budget = _validate_budget(budget, chunk_count=chunk_count)
    parsed_repeat = _require_integer(repeat_id, "repeat_id")
    parsed_chunk = _require_integer(chunk_index, "chunk_index", minimum=1)
    _require(
        parsed_chunk in candidate_chunks(chunk_count),
        f"chunk_index must be in 1..{chunk_count - 1}",
    )
    payload = json.dumps(
        [
            parsed_protocol,
            parsed_prompt,
            parsed_seed,
            parsed_budget,
            parsed_repeat,
            parsed_chunk,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def random_positions(
    *,
    protocol_id: str,
    prompt_id: str,
    generation_seed: int,
    budget: int,
    repeat_id: int,
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> tuple[int, ...]:
    """Select K chunks uniformly without replacement by SHA-256 ordering."""

    parsed_budget = _validate_budget(budget, chunk_count=chunk_count)
    candidates = candidate_chunks(chunk_count)
    scored = [
        (
            random_selection_digest(
                protocol_id=protocol_id,
                prompt_id=prompt_id,
                generation_seed=generation_seed,
                budget=parsed_budget,
                repeat_id=repeat_id,
                chunk_index=chunk,
                chunk_count=chunk_count,
            ),
            chunk,
        )
        for chunk in candidates
    ]
    # The chunk index is an explicit, deterministic tie-breaker for the
    # vanishingly unlikely event of a SHA-256 collision.
    selected = tuple(sorted(chunk for _, chunk in sorted(scored)[:parsed_budget]))
    _validate_chunk_tuple(selected, "random positions", chunk_count=chunk_count)
    _require(len(selected) == parsed_budget, "Random policy did not produce exactly K chunks")
    return selected


def random_p8_k2_positions(
    *, protocol_id: str, prompt_id: str, generation_seed: int, repeat_id: int
) -> tuple[int, int]:
    """Choose one of eight Draft observations before a two-chunk Target burst."""
    observation, = random_positions(
        protocol_id=f"{protocol_id}/native-random-p8-k2-v1",
        prompt_id=prompt_id,
        generation_seed=generation_seed,
        budget=1,
        repeat_id=repeat_id,
        chunk_count=9,
    )
    return observation + 1, observation + 2


def calibration_rank_cdf_risk(rank: float, calibration_ranks: Sequence[float]) -> float:
    """Map effective rank to lower-tail degradation risk.

    The frozen mid-rank definition is ``(#greater + 0.5 * #equal) / N``;
    lower effective rank consequently means higher degradation risk.
    """

    parsed_rank = _require_finite(rank, "rank")
    _require(parsed_rank > 0.0, "rank must be positive")
    _require(
        isinstance(calibration_ranks, Sequence)
        and not isinstance(calibration_ranks, (str, bytes))
        and len(calibration_ranks) > 0,
        "calibration_ranks must be a non-empty sequence",
    )
    reference = [
        _require_finite(value, f"calibration_ranks[{index}]")
        for index, value in enumerate(calibration_ranks)
    ]
    _require(all(value > 0.0 for value in reference), "calibration ranks must be positive")
    greater = sum(value > parsed_rank for value in reference)
    equal = sum(value == parsed_rank for value in reference)
    return (greater + 0.5 * equal) / len(reference)


def causal_erank_sudden_drop(
    rank_history_including_current: Sequence[float],
    *,
    lookback_chunks: int = 4,
) -> float:
    """Return the pre-registered causal eRank-drop score for the current chunk.

    Let ``r_t`` be the just-observed Draft eRank and let ``m_t`` be the median
    of the previous at most ``lookback_chunks`` Draft eRanks.  The score is
    ``max(0, (m_t - r_t) / m_t)``.  It is zero when no previous Draft chunk
    exists.  Thus the function consumes current/past ranks only and cannot
    leak a future chunk into an online decision.
    """

    _require(
        isinstance(rank_history_including_current, Sequence)
        and not isinstance(rank_history_including_current, (str, bytes))
        and len(rank_history_including_current) > 0,
        "rank_history_including_current must be a non-empty sequence",
    )
    parsed_lookback = _require_integer(
        lookback_chunks, "lookback_chunks", minimum=1
    )
    ranks = tuple(
        _require_finite(value, f"rank_history_including_current[{index}]")
        for index, value in enumerate(rank_history_including_current)
    )
    _require(all(value > 0.0 for value in ranks), "effective ranks must be positive")
    if len(ranks) == 1:
        return 0.0
    previous = ranks[max(0, len(ranks) - 1 - parsed_lookback) : -1]
    baseline = float(statistics.median(previous))
    score = max(0.0, (baseline - ranks[-1]) / baseline)
    # Positive ranks imply score < 1; clamp only to absorb floating-point noise.
    return min(1.0, score)


def calibration_drop_cdf_risk(
    drop_score: float, calibration_drop_scores: Sequence[float]
) -> float:
    """Map a causal eRank drop to an upper-tail mid-rank percentile.

    The definition is ``(#less + 0.5 * #equal) / N``.  Larger sudden drops
    therefore yield higher risk while ties remain deterministic.
    """

    parsed_drop = _require_finite(drop_score, "drop_score")
    _require(0.0 <= parsed_drop <= 1.0, "drop_score must be in [0, 1]")
    _require(
        isinstance(calibration_drop_scores, Sequence)
        and not isinstance(calibration_drop_scores, (str, bytes))
        and len(calibration_drop_scores) > 0,
        "calibration_drop_scores must be a non-empty sequence",
    )
    reference = tuple(
        _require_finite(value, f"calibration_drop_scores[{index}]")
        for index, value in enumerate(calibration_drop_scores)
    )
    _require(
        all(0.0 <= value <= 1.0 for value in reference),
        "calibration drop scores must be in [0, 1]",
    )
    less = sum(value < parsed_drop for value in reference)
    equal = sum(value == parsed_drop for value in reference)
    return (less + 0.5 * equal) / len(reference)


def higher_empirical_quantile(values: Sequence[float], probability: float) -> float:
    """Return the empirical quantile using NumPy's ``method='higher'`` rule."""

    _require(
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and len(values) > 0,
        "values must be a non-empty sequence",
    )
    parsed_values = sorted(
        _require_finite(value, f"values[{index}]") for index, value in enumerate(values)
    )
    parsed_probability = _require_finite(probability, "probability")
    _require(0.0 <= parsed_probability <= 1.0, "probability must be in [0, 1]")
    index = math.ceil(parsed_probability * (len(parsed_values) - 1))
    return parsed_values[index]


@dataclass(frozen=True)
class RankRiskCalibrator:
    """A named independent calibration population for rank-based risk."""

    calibration_id: str
    ranks: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.calibration_id, "calibration_id")
        _require(
            isinstance(self.ranks, tuple) and bool(self.ranks),
            "ranks must be a non-empty tuple",
        )
        parsed = tuple(
            _require_finite(value, f"ranks[{index}]")
            for index, value in enumerate(self.ranks)
        )
        _require(all(value > 0.0 for value in parsed), "calibration ranks must be positive")
        object.__setattr__(self, "ranks", parsed)

    def risk(self, rank: float) -> float:
        """Transform one observed rank without mutating the calibration set."""

        return calibration_rank_cdf_risk(rank, self.ranks)

    @property
    def calibration_risks(self) -> tuple[float, ...]:
        """Return leave-in risk scores used solely to freeze online thresholds."""

        return tuple(self.risk(rank) for rank in self.ranks)

    def as_risk_distribution(self) -> CalibrationRiskDistribution:
        """Materialize the independent risk population used for thresholds."""

        return CalibrationRiskDistribution(
            calibration_id=self.calibration_id,
            risks=self.calibration_risks,
        )


@dataclass(frozen=True)
class SuddenDropRiskCalibrator:
    """Independent calibration population for causal eRank-drop scores."""

    calibration_id: str
    drop_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.calibration_id, "calibration_id")
        _require(
            isinstance(self.drop_scores, tuple) and bool(self.drop_scores),
            "drop_scores must be a non-empty tuple",
        )
        parsed = tuple(
            _require_finite(value, f"drop_scores[{index}]")
            for index, value in enumerate(self.drop_scores)
        )
        _require(
            all(0.0 <= value <= 1.0 for value in parsed),
            "drop_scores must be in [0, 1]",
        )
        object.__setattr__(self, "drop_scores", parsed)

    def risk(self, drop_score: float) -> float:
        return calibration_drop_cdf_risk(drop_score, self.drop_scores)

    @property
    def calibration_risks(self) -> tuple[float, ...]:
        return tuple(self.risk(score) for score in self.drop_scores)

    def as_risk_distribution(self) -> CalibrationRiskDistribution:
        return CalibrationRiskDistribution(
            calibration_id=self.calibration_id,
            risks=self.calibration_risks,
        )


@dataclass(frozen=True)
class CalibrationRiskDistribution:
    """Named independent risk scores used by the online budget controller.

    This direct representation also supports chunk-conditional rank CDFs: a
    campaign may map each calibration row against the reference ranks for its
    own chunk, then combine those already-calibrated risks here.
    """

    calibration_id: str
    risks: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.calibration_id, "calibration_id")
        _require(
            isinstance(self.risks, tuple) and bool(self.risks),
            "risks must be a non-empty tuple",
        )
        parsed = tuple(
            _require_finite(value, f"risks[{index}]")
            for index, value in enumerate(self.risks)
        )
        _require(all(0.0 <= value <= 1.0 for value in parsed), "risks must be in [0, 1]")
        object.__setattr__(self, "risks", parsed)


@dataclass(frozen=True)
class OnlineDecision:
    """Auditable decision emitted after one current Draft chunk."""

    chunk_index: int
    risk_score: float
    replace_with_target: bool
    threshold: float | None
    quantile_probability: float | None
    remaining_budget_before: int
    remaining_slots_including_current: int
    reason: str


class OnlineRankGuidedPolicy:
    """Sequential Rank-guided policy with no future-risk input.

    The caller must invoke :meth:`decide` exactly once for every candidate
    chunk in order, after the Draft has produced each current chunk.  At a regular
    decision the upper-tail threshold is the independent calibration-risk
    empirical quantile ``1 - remaining_budget / remaining_slots`` using
    ``method='higher'``.  Budget exhaustion and the final forced slots ensure
    exactly K Target replacements.
    """

    def __init__(
        self,
        *,
        budget: int,
        calibration: (
            CalibrationRiskDistribution
            | RankRiskCalibrator
            | SuddenDropRiskCalibrator
        ),
        evaluation_id: str,
        chunk_count: int = DEFAULT_CHUNK_COUNT,
    ) -> None:
        self._chunk_count = _require_integer(
            chunk_count, "chunk_count", minimum=2
        )
        self._candidate_chunks = candidate_chunks(self._chunk_count)
        self._budget = _validate_budget(budget, chunk_count=self._chunk_count)
        _require(
            isinstance(
                calibration,
                (
                    CalibrationRiskDistribution,
                    RankRiskCalibrator,
                    SuddenDropRiskCalibrator,
                ),
            ),
            "calibration must be an independent calibration risk distribution",
        )
        distribution = (
            calibration.as_risk_distribution()
            if isinstance(calibration, (RankRiskCalibrator, SuddenDropRiskCalibrator))
            else calibration
        )
        self._evaluation_id = _require_nonempty_string(evaluation_id, "evaluation_id")
        _require(
            self._evaluation_id != distribution.calibration_id,
            "evaluation_id must differ from the independent calibration_id",
        )
        self._calibration = distribution
        self._cursor = 0
        self._selected: list[int] = []
        self._decisions: list[OnlineDecision] = []

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def selected_chunks(self) -> tuple[int, ...]:
        return tuple(self._selected)

    @property
    def decisions(self) -> tuple[OnlineDecision, ...]:
        return tuple(self._decisions)

    @property
    def is_complete(self) -> bool:
        return self._cursor == len(self._candidate_chunks)

    def decide(self, *, chunk_index: int, risk_score: float,
               additional_trigger: bool = False, allow_threshold_trigger: bool = True) -> OnlineDecision:
        """Decide for the current chunk using only its scalar risk and state."""

        _require(not self.is_complete, "all candidate chunks have already been consumed")
        expected_chunk = self._candidate_chunks[self._cursor]
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and chunk_index == expected_chunk,
            f"expected chunk {expected_chunk}; online decisions must be chronological",
        )
        parsed_risk = _require_finite(risk_score, "risk_score")
        _require(0.0 <= parsed_risk <= 1.0, "risk_score must be in [0, 1]")
        _require(type(additional_trigger) is bool, "additional trigger must be boolean")
        _require(type(allow_threshold_trigger) is bool, "threshold trigger gate must be boolean")

        remaining_budget = self._budget - len(self._selected)
        remaining_slots = len(self._candidate_chunks) - self._cursor
        _require(0 <= remaining_budget <= remaining_slots, "online budget state is inconsistent")

        threshold: float | None
        probability: float | None
        if remaining_budget == 0:
            replace = False
            threshold = None
            probability = None
            reason = "budget_exhausted"
        elif remaining_budget == remaining_slots:
            replace = True
            threshold = None
            probability = None
            reason = "forced_remaining_slots_equal_budget"
        else:
            probability = 1.0 - remaining_budget / remaining_slots
            threshold = higher_empirical_quantile(
                self._calibration.risks,
                probability,
            )
            replace = parsed_risk >= threshold and allow_threshold_trigger
            reason = "risk_at_or_above_calibration_threshold" if replace else "risk_below_threshold"
            if not allow_threshold_trigger and parsed_risk >= threshold:
                reason = "threshold_trigger_gated"
            if additional_trigger and not replace:
                replace, reason = True, "additional_trigger"

        decision = OnlineDecision(
            chunk_index=chunk_index,
            risk_score=parsed_risk,
            replace_with_target=replace,
            threshold=threshold,
            quantile_probability=probability,
            remaining_budget_before=remaining_budget,
            remaining_slots_including_current=remaining_slots,
            reason=reason,
        )
        if replace:
            self._selected.append(chunk_index)
        self._decisions.append(decision)
        self._cursor += 1
        return decision

    def finalize(self) -> tuple[int, ...]:
        """Return the exact-K selection only after the full online horizon."""

        _require(
            self.is_complete,
            f"cannot finalize before chunks 1..{self._chunk_count - 1} are consumed",
        )
        _require(
            len(self._selected) == self._budget,
            "online policy did not consume exactly K replacements",
        )
        return tuple(self._selected)


@dataclass(frozen=True)
class CoverageConstrainedOnlineDecision:
    """Auditable decision for one coverage-constrained online chunk.

    ``static_window_*`` records the outcome-independent window around the
    uniform anchor.  ``eligible_start`` and ``deadline`` are the causal bounds
    after intersecting that window with the already-committed Target history
    and the suffix feasibility envelope.  No future risk is represented here.
    """

    chunk_index: int
    risk_score: float
    replace_with_target: bool
    threshold: float | None
    quantile_probability: float | None
    remaining_budget_before: int
    remaining_slots_including_current: int
    correction_ordinal: int | None
    anchor_chunk: int | None
    static_window_start: int | None
    static_window_end: int | None
    feasible_window_start: int | None
    feasible_window_end: int | None
    eligible_start: int | None
    deadline: int | None
    local_slots_including_current: int | None
    previous_target_chunk: int
    reason: str


class CoverageConstrainedOnlineRankGuidedPolicy:
    """Causal risk selection with one correction near each uniform anchor.

    The K outcome-independent anchors are :func:`uniform_positions`.  Anchor
    ``j`` owns the inclusive static window ``anchor[j] +/- window_radius`` and
    exactly one Target replacement.  A backward feasibility pass narrows those
    windows just enough to guarantee the terminal max-gap constraint and a
    feasible correction in every later window.  At runtime the previous
    committed Target further determines the current ``eligible_start`` and
    ``deadline`` from the frozen min/max gaps.

    Between those causal bounds, the current risk is compared with a *local*
    higher empirical quantile, ``1 - 1 / remaining_window_slots``.  The policy
    therefore takes an unusually risky current proposal early, but forces the
    correction at the deadline.  It observes no future risk and consumes
    exactly one correction per anchor window.
    """

    def __init__(
        self,
        *,
        budget: int,
        calibration: (
            CalibrationRiskDistribution
            | RankRiskCalibrator
            | SuddenDropRiskCalibrator
        ),
        evaluation_id: str,
        chunk_count: int,
        window_radius_chunks: int,
        min_gap_chunks: int,
        max_gap_chunks: int,
    ) -> None:
        self._chunk_count = _require_integer(
            chunk_count, "chunk_count", minimum=2
        )
        self._candidate_chunks = candidate_chunks(self._chunk_count)
        self._budget = _validate_budget(budget, chunk_count=self._chunk_count)
        self._window_radius_chunks = _require_integer(
            window_radius_chunks, "window_radius_chunks"
        )
        self._min_gap_chunks = _require_integer(
            min_gap_chunks, "min_gap_chunks", minimum=1
        )
        self._max_gap_chunks = _require_integer(
            max_gap_chunks, "max_gap_chunks", minimum=1
        )
        _require(
            self._min_gap_chunks <= self._max_gap_chunks,
            "min_gap_chunks must not exceed max_gap_chunks",
        )
        _require(
            isinstance(
                calibration,
                (
                    CalibrationRiskDistribution,
                    RankRiskCalibrator,
                    SuddenDropRiskCalibrator,
                ),
            ),
            "calibration must be an independent calibration risk distribution",
        )
        distribution = (
            calibration.as_risk_distribution()
            if isinstance(calibration, (RankRiskCalibrator, SuddenDropRiskCalibrator))
            else calibration
        )
        self._evaluation_id = _require_nonempty_string(evaluation_id, "evaluation_id")
        _require(
            self._evaluation_id != distribution.calibration_id,
            "evaluation_id must differ from the independent calibration_id",
        )
        self._calibration = distribution

        horizon_end = self._chunk_count - 1
        self._anchors = uniform_positions(
            self._budget, chunk_count=self._chunk_count
        )
        self._static_windows = tuple(
            (
                max(1, anchor - self._window_radius_chunks),
                min(horizon_end, anchor + self._window_radius_chunks),
            )
            for anchor in self._anchors
        )
        _require(
            all(
                left[1] < right[0]
                for left, right in zip(
                    self._static_windows, self._static_windows[1:]
                )
            ),
            "coverage windows must be disjoint and chronological",
        )

        # Compute the complete suffix-feasible envelope without looking at any
        # evaluation risk.  If x is selected in window i, the next selected y
        # must satisfy min_gap <= y-x <= max_gap.  The backward interval below
        # is exactly the set of integer x for which such a y remains possible.
        feasible_reversed: list[tuple[int, int]] = []
        next_start: int | None = None
        next_end: int | None = None
        for index in range(self._budget - 1, -1, -1):
            static_start, static_end = self._static_windows[index]
            if next_start is None:
                feasible_start = max(
                    static_start, horizon_end - self._max_gap_chunks
                )
                feasible_end = min(
                    static_end, horizon_end - self._min_gap_chunks
                )
            else:
                assert next_end is not None
                feasible_start = max(
                    static_start, next_start - self._max_gap_chunks
                )
                feasible_end = min(
                    static_end, next_end - self._min_gap_chunks
                )
            _require(
                feasible_start <= feasible_end,
                f"coverage window {index + 1} has no suffix-feasible chunk",
            )
            feasible_reversed.append((feasible_start, feasible_end))
            next_start, next_end = feasible_start, feasible_end
        feasible_windows = list(reversed(feasible_reversed))
        first_start, first_end = feasible_windows[0]
        first_start = max(first_start, self._min_gap_chunks)
        first_end = min(first_end, self._max_gap_chunks)
        _require(
            first_start <= first_end,
            "first coverage window cannot satisfy the chunk-0 gap",
        )
        feasible_windows[0] = (first_start, first_end)
        self._feasible_windows = tuple(feasible_windows)

        self._cursor = 0
        self._selected: list[int] = []
        self._decisions: list[CoverageConstrainedOnlineDecision] = []

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def anchors(self) -> tuple[int, ...]:
        return self._anchors

    @property
    def static_windows(self) -> tuple[tuple[int, int], ...]:
        return self._static_windows

    @property
    def selected_chunks(self) -> tuple[int, ...]:
        return tuple(self._selected)

    @property
    def decisions(self) -> tuple[CoverageConstrainedOnlineDecision, ...]:
        return tuple(self._decisions)

    @property
    def is_complete(self) -> bool:
        return self._cursor == len(self._candidate_chunks)

    def _current_bounds(self) -> tuple[int, int, int, int, int]:
        ordinal_index = len(self._selected)
        _require(ordinal_index < self._budget, "coverage budget is already exhausted")
        anchor = self._anchors[ordinal_index]
        static_start, static_end = self._static_windows[ordinal_index]
        feasible_start, feasible_end = self._feasible_windows[ordinal_index]
        previous_target = self._selected[-1] if self._selected else FIRST_TARGET_CHUNK
        eligible_start = max(
            feasible_start, previous_target + self._min_gap_chunks
        )
        deadline = min(
            feasible_end, previous_target + self._max_gap_chunks
        )
        _require(
            eligible_start <= deadline,
            f"coverage window {ordinal_index + 1} became infeasible",
        )
        return anchor, static_start, static_end, eligible_start, deadline

    def decide(
        self, *, chunk_index: int, risk_score: float
    ) -> CoverageConstrainedOnlineDecision:
        """Decide causally for the current Draft proposal."""

        _require(not self.is_complete, "all candidate chunks have already been consumed")
        expected_chunk = self._candidate_chunks[self._cursor]
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and chunk_index == expected_chunk,
            f"expected chunk {expected_chunk}; online decisions must be chronological",
        )
        parsed_risk = _require_finite(risk_score, "risk_score")
        _require(0.0 <= parsed_risk <= 1.0, "risk_score must be in [0, 1]")

        remaining_budget = self._budget - len(self._selected)
        remaining_slots = len(self._candidate_chunks) - self._cursor
        previous_target = self._selected[-1] if self._selected else FIRST_TARGET_CHUNK
        _require(
            0 <= remaining_budget <= remaining_slots,
            "online budget state is inconsistent",
        )

        anchor: int | None = None
        static_start: int | None = None
        static_end: int | None = None
        feasible_start: int | None = None
        feasible_end: int | None = None
        eligible_start: int | None = None
        deadline: int | None = None
        local_slots: int | None = None
        threshold: float | None = None
        probability: float | None = None
        correction_ordinal: int | None = None

        if remaining_budget == 0:
            replace = False
            reason = "budget_exhausted"
        else:
            ordinal_index = len(self._selected)
            correction_ordinal = ordinal_index + 1
            anchor, static_start, static_end, eligible_start, deadline = (
                self._current_bounds()
            )
            feasible_start, feasible_end = self._feasible_windows[ordinal_index]
            _require(
                chunk_index <= deadline,
                f"coverage window {correction_ordinal} missed its deadline",
            )
            if chunk_index < eligible_start:
                replace = False
                reason = "before_dynamic_eligible_start"
            elif chunk_index == deadline:
                replace = True
                local_slots = 1
                reason = "forced_at_dynamic_deadline"
            else:
                local_slots = deadline - chunk_index + 1
                probability = 1.0 - 1.0 / local_slots
                threshold = higher_empirical_quantile(
                    self._calibration.risks, probability
                )
                replace = parsed_risk >= threshold
                reason = (
                    "risk_at_or_above_local_threshold"
                    if replace
                    else "risk_below_local_threshold"
                )

        decision = CoverageConstrainedOnlineDecision(
            chunk_index=chunk_index,
            risk_score=parsed_risk,
            replace_with_target=replace,
            threshold=threshold,
            quantile_probability=probability,
            remaining_budget_before=remaining_budget,
            remaining_slots_including_current=remaining_slots,
            correction_ordinal=correction_ordinal,
            anchor_chunk=anchor,
            static_window_start=static_start,
            static_window_end=static_end,
            feasible_window_start=feasible_start,
            feasible_window_end=feasible_end,
            eligible_start=eligible_start,
            deadline=deadline,
            local_slots_including_current=local_slots,
            previous_target_chunk=previous_target,
            reason=reason,
        )
        if replace:
            self._selected.append(chunk_index)
        self._decisions.append(decision)
        self._cursor += 1
        return decision

    def finalize(self) -> tuple[int, ...]:
        """Return an exact-K route only after validating full coverage."""

        _require(
            self.is_complete,
            f"cannot finalize before chunks 1..{self._chunk_count - 1} are consumed",
        )
        _require(
            len(self._selected) == self._budget,
            "coverage policy did not consume exactly K replacements",
        )
        _require(
            all(
                start <= selected <= end
                for selected, (start, end) in zip(
                    self._selected, self._static_windows
                )
            ),
            "coverage policy did not select exactly once inside every static window",
        )
        correction_points = (FIRST_TARGET_CHUNK, *self._selected)
        correction_gaps = tuple(
            current - previous
            for previous, current in zip(
                correction_points, correction_points[1:]
            )
        )
        _require(
            all(
                self._min_gap_chunks <= gap <= self._max_gap_chunks
                for gap in correction_gaps
            ),
            "coverage correction gaps violate the frozen min/max bounds",
        )
        tail_gap = (self._chunk_count - 1) - self._selected[-1]
        _require(
            self._min_gap_chunks <= tail_gap <= self._max_gap_chunks,
            "coverage terminal tail violates the frozen min/max bounds",
        )
        return tuple(self._selected)


@dataclass(frozen=True)
class BoundarySmoothnessGateDecision:
    """Auditable outcome of the frozen paired Target-acceptance gate.

    The gate compares a corrected successor against the already-generated
    original successor, and compares Target boundary witnesses against the
    corresponding Draft witnesses.  All three paired deltas use a frozen zero
    threshold.  This object deliberately does not select a correction chunk
    or mutate an online controller; it only answers whether a caller-owned
    Target candidate has complete evidence and is safe to accept.
    """

    metric_id: str
    metric_version: str
    erank_delta_threshold: float
    entry_delta_threshold: float
    exit_delta_threshold: float
    original_successor_erank: float | None
    corrected_successor_erank: float | None
    draft_entry_witness_score: float | None
    target_entry_witness_score: float | None
    draft_exit_witness_score: float | None
    target_exit_witness_score: float | None
    erank_delta: float | None
    entry_delta: float | None
    exit_delta: float | None
    erank_recovered: bool
    entry_not_worse: bool
    exit_not_worse: bool
    evidence_valid: bool
    accept_target: bool
    reason: str


@dataclass(frozen=True)
class BoundaryEntryRejectionDecision(BoundarySmoothnessGateDecision):
    """A decisive entry failure with deliberately incomplete successor evidence.

    The original full-gate serialization stays unchanged.  Only this opt-in
    short circuit has these extra fields, and unmeasured checks remain null.
    ``evidence_valid`` describes the measured entry evidence, not completeness.
    """

    erank_recovered: None
    exit_not_worse: None
    evidence_complete: bool = False
    short_circuit: bool = True
    evaluation_stage: str = "entry_only"


class BoundarySmoothnessGate:
    """Fail-closed causal gate for a fully materialized Target candidate.

    Effective ranks are bounded by the frozen 256-value rank witness used by
    Insight 2.  Entry and exit witness scores are normalized to ``[0, 1]``.
    Missing, non-finite, boolean, or out-of-range values produce a rejected
    decision instead of raising or partially evaluating the conjunction.
    """

    metric_id = BOUNDARY_GATE_METRIC_ID
    metric_version = BOUNDARY_GATE_METRIC_VERSION
    erank_delta_threshold = BOUNDARY_GATE_ERANK_DELTA_THRESHOLD
    entry_delta_threshold = BOUNDARY_GATE_ENTRY_DELTA_THRESHOLD
    exit_delta_threshold = BOUNDARY_GATE_EXIT_DELTA_THRESHOLD

    @staticmethod
    def _bounded_value(
        value: object,
        *,
        field: str,
        minimum: float,
        maximum: float,
    ) -> tuple[float | None, str | None]:
        if value is None:
            return None, f"{field}_missing"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, f"{field}_invalid"
        try:
            parsed = float(value)
        except (OverflowError, TypeError, ValueError):
            return None, f"{field}_invalid"
        if not math.isfinite(parsed):
            return None, f"{field}_invalid"
        if not minimum <= parsed <= maximum:
            return None, f"{field}_out_of_range"
        return parsed, None

    def reject_if_entry_worse(
        self,
        *,
        draft_entry_witness_score: object = None,
        target_entry_witness_score: object = None,
    ) -> BoundaryEntryRejectionDecision | None:
        """Short-circuit only a valid entry failure of the unchanged AND gate.

        Invalid or passing entry evidence returns None so callers retain the
        complete generation/validation path.  No successor evidence is inferred
        and this method can never authorize a Target commit.
        """

        draft_entry, draft_error = self._bounded_value(
            draft_entry_witness_score,
            field="draft_entry_witness_score",
            minimum=BOUNDARY_GATE_MIN_WITNESS_SCORE,
            maximum=BOUNDARY_GATE_MAX_WITNESS_SCORE,
        )
        target_entry, target_error = self._bounded_value(
            target_entry_witness_score,
            field="target_entry_witness_score",
            minimum=BOUNDARY_GATE_MIN_WITNESS_SCORE,
            maximum=BOUNDARY_GATE_MAX_WITNESS_SCORE,
        )
        if draft_error is not None or target_error is not None:
            return None
        assert draft_entry is not None and target_entry is not None
        entry_delta = target_entry - draft_entry
        if entry_delta <= self.entry_delta_threshold:
            return None
        return BoundaryEntryRejectionDecision(
            metric_id=self.metric_id,
            metric_version=self.metric_version,
            erank_delta_threshold=self.erank_delta_threshold,
            entry_delta_threshold=self.entry_delta_threshold,
            exit_delta_threshold=self.exit_delta_threshold,
            original_successor_erank=None,
            corrected_successor_erank=None,
            draft_entry_witness_score=draft_entry,
            target_entry_witness_score=target_entry,
            draft_exit_witness_score=None,
            target_exit_witness_score=None,
            erank_delta=None,
            entry_delta=entry_delta,
            exit_delta=None,
            erank_recovered=None,
            entry_not_worse=False,
            exit_not_worse=None,
            evidence_valid=True,
            accept_target=False,
            reason="rejected:target_entry_worse",
        )

    def decide(
        self,
        *,
        original_successor_erank: object = None,
        corrected_successor_erank: object = None,
        draft_entry_witness_score: object = None,
        target_entry_witness_score: object = None,
        draft_exit_witness_score: object = None,
        target_exit_witness_score: object = None,
    ) -> BoundarySmoothnessGateDecision:
        fields = (
            (
                "original_successor_erank",
                original_successor_erank,
                BOUNDARY_GATE_MIN_ERANK,
                BOUNDARY_GATE_MAX_ERANK,
            ),
            (
                "corrected_successor_erank",
                corrected_successor_erank,
                BOUNDARY_GATE_MIN_ERANK,
                BOUNDARY_GATE_MAX_ERANK,
            ),
            (
                "draft_entry_witness_score",
                draft_entry_witness_score,
                BOUNDARY_GATE_MIN_WITNESS_SCORE,
                BOUNDARY_GATE_MAX_WITNESS_SCORE,
            ),
            (
                "target_entry_witness_score",
                target_entry_witness_score,
                BOUNDARY_GATE_MIN_WITNESS_SCORE,
                BOUNDARY_GATE_MAX_WITNESS_SCORE,
            ),
            (
                "draft_exit_witness_score",
                draft_exit_witness_score,
                BOUNDARY_GATE_MIN_WITNESS_SCORE,
                BOUNDARY_GATE_MAX_WITNESS_SCORE,
            ),
            (
                "target_exit_witness_score",
                target_exit_witness_score,
                BOUNDARY_GATE_MIN_WITNESS_SCORE,
                BOUNDARY_GATE_MAX_WITNESS_SCORE,
            ),
        )
        parsed: dict[str, float | None] = {}
        invalid: list[str] = []
        for field, value, minimum, maximum in fields:
            parsed_value, error = self._bounded_value(
                value,
                field=field,
                minimum=minimum,
                maximum=maximum,
            )
            parsed[field] = parsed_value
            if error is not None:
                invalid.append(error)

        evidence_valid = not invalid
        if evidence_valid:
            original_rank = parsed["original_successor_erank"]
            corrected_rank = parsed["corrected_successor_erank"]
            draft_entry = parsed["draft_entry_witness_score"]
            target_entry = parsed["target_entry_witness_score"]
            draft_exit = parsed["draft_exit_witness_score"]
            target_exit = parsed["target_exit_witness_score"]
            assert original_rank is not None and corrected_rank is not None
            assert draft_entry is not None and target_entry is not None
            assert draft_exit is not None and target_exit is not None
            erank_delta = corrected_rank - original_rank
            entry_delta = target_entry - draft_entry
            exit_delta = target_exit - draft_exit
            erank_recovered = erank_delta >= self.erank_delta_threshold
            entry_not_worse = entry_delta <= self.entry_delta_threshold
            exit_not_worse = exit_delta <= self.exit_delta_threshold
            failed_checks = tuple(
                label
                for label, passed in (
                    ("erank_not_recovered", erank_recovered),
                    ("target_entry_worse", entry_not_worse),
                    ("target_exit_worse", exit_not_worse),
                )
                if not passed
            )
            accept_target = not failed_checks
            reason = (
                "accepted"
                if accept_target
                else "rejected:" + ",".join(failed_checks)
            )
        else:
            erank_delta = None
            entry_delta = None
            exit_delta = None
            erank_recovered = False
            entry_not_worse = False
            exit_not_worse = False
            accept_target = False
            reason = "invalid_evidence:" + ",".join(invalid)

        return BoundarySmoothnessGateDecision(
            metric_id=self.metric_id,
            metric_version=self.metric_version,
            erank_delta_threshold=self.erank_delta_threshold,
            entry_delta_threshold=self.entry_delta_threshold,
            exit_delta_threshold=self.exit_delta_threshold,
            original_successor_erank=parsed["original_successor_erank"],
            corrected_successor_erank=parsed["corrected_successor_erank"],
            draft_entry_witness_score=parsed["draft_entry_witness_score"],
            target_entry_witness_score=parsed["target_entry_witness_score"],
            draft_exit_witness_score=parsed["draft_exit_witness_score"],
            target_exit_witness_score=parsed["target_exit_witness_score"],
            erank_delta=erank_delta,
            entry_delta=entry_delta,
            exit_delta=exit_delta,
            erank_recovered=erank_recovered,
            entry_not_worse=entry_not_worse,
            exit_not_worse=exit_not_worse,
            evidence_valid=evidence_valid,
            accept_target=accept_target,
            reason=reason,
        )


@dataclass(frozen=True)
class LookaheadCoverageConstrainedOnlineDecision:
    """One guarded commit driven by the immediately following Draft chunk.

    ``chunk_index`` is the pending Draft chunk being accepted or replaced.
    While budget remains, every non-terminal pending chunk is guarded by the
    eRank risk of ``signal_chunk_index == chunk_index + 1``.  The coverage
    controller must consume the exact Target budget before the final real
    chunk (by chunk 107 in the frozen 120-chunk protocol).  The final chunk is
    then an audit-only terminal flush with no preview or signal; reaching it
    with an unconsumed budget fails closed.  Once the exact Target budget has
    been consumed, no preview is requested and the remaining Draft suffix is
    committed directly.

    The nested controller decision preserves the complete, independently
    replayable coverage-window audit trail without duplicating those fields in
    a second policy implementation.
    """

    chunk_index: int
    signal_chunk_index: int | None
    risk_score: float | None
    replace_with_target: bool
    preview_required: bool
    terminal_flush: bool
    reason: str
    controller_decision: CoverageConstrainedOnlineDecision


class LookaheadCoverageConstrainedOnlineRankGuidedPolicy:
    """One-chunk lookahead gate around the exact-K coverage controller.

    This class owns only the temporal binding between a pending chunk and its
    signal.  Coverage feasibility, quantile thresholds, deadline forcing, and
    exact-K accounting remain delegated to
    :class:`CoverageConstrainedOnlineRankGuidedPolicy`.
    """

    def __init__(
        self,
        *,
        budget: int,
        calibration: (
            CalibrationRiskDistribution
            | RankRiskCalibrator
            | SuddenDropRiskCalibrator
        ),
        evaluation_id: str,
        chunk_count: int,
        window_radius_chunks: int,
        min_gap_chunks: int,
        max_gap_chunks: int,
    ) -> None:
        self._controller = CoverageConstrainedOnlineRankGuidedPolicy(
            budget=budget,
            calibration=calibration,
            evaluation_id=evaluation_id,
            chunk_count=chunk_count,
            window_radius_chunks=window_radius_chunks,
            min_gap_chunks=min_gap_chunks,
            max_gap_chunks=max_gap_chunks,
        )
        self._candidate_chunks = candidate_chunks(chunk_count)
        self._cursor = 0
        self._decisions: list[LookaheadCoverageConstrainedOnlineDecision] = []

    @property
    def budget(self) -> int:
        return self._controller.budget

    @property
    def chunk_count(self) -> int:
        return self._controller.chunk_count

    @property
    def static_windows(self) -> tuple[tuple[int, int], ...]:
        """Expose immutable proposal windows for optional KV fork retention."""

        return self._controller.static_windows

    @property
    def selected_chunks(self) -> tuple[int, ...]:
        return self._controller.selected_chunks

    @property
    def decisions(self) -> tuple[LookaheadCoverageConstrainedOnlineDecision, ...]:
        return tuple(self._decisions)

    @property
    def is_complete(self) -> bool:
        return self._cursor == len(self._candidate_chunks)

    @property
    def remaining_budget(self) -> int:
        return self.budget - len(self.selected_chunks)

    def preview_required(self, *, chunk_index: int) -> bool:
        """Return whether the current pending chunk needs a real successor."""

        _require(not self.is_complete, "all candidate chunks have already been consumed")
        expected_chunk = self._candidate_chunks[self._cursor]
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and chunk_index == expected_chunk,
            f"expected chunk {expected_chunk}; lookahead decisions must be chronological",
        )
        return self.remaining_budget > 0 and chunk_index < self._candidate_chunks[-1]

    def decide(
        self,
        *,
        chunk_index: int,
        signal_chunk_index: int | None,
        risk_score: float | None,
    ) -> LookaheadCoverageConstrainedOnlineDecision:
        """Resolve one pending chunk with a bound preview or terminal flush."""

        needs_preview = self.preview_required(chunk_index=chunk_index)
        terminal = chunk_index == self._candidate_chunks[-1]
        remaining_budget = self.remaining_budget
        if remaining_budget == 0:
            _require(
                signal_chunk_index is None and risk_score is None,
                "budget-exhausted lookahead decisions must not carry a preview signal",
            )
            controller_risk = 0.0
            reason_prefix = "budget_exhausted_without_preview"
        elif terminal:
            raise PolicyContractError(
                "terminal flush reached with unconsumed Target budget"
            )
        else:
            _require(needs_preview, "non-terminal guarded decision requires a preview")
            _require(
                signal_chunk_index == chunk_index + 1,
                "lookahead signal must be the immediate successor chunk",
            )
            controller_risk = _require_finite(risk_score, "risk_score")
            _require(
                0.0 <= controller_risk <= 1.0,
                "risk_score must be in [0, 1]",
            )
            reason_prefix = "one_chunk_preview"

        controller_decision = self._controller.decide(
            chunk_index=chunk_index,
            risk_score=controller_risk,
        )
        decision = LookaheadCoverageConstrainedOnlineDecision(
            chunk_index=chunk_index,
            signal_chunk_index=signal_chunk_index,
            risk_score=None if remaining_budget == 0 else controller_risk,
            replace_with_target=controller_decision.replace_with_target,
            preview_required=needs_preview,
            terminal_flush=terminal,
            reason=f"{reason_prefix}:{controller_decision.reason}",
            controller_decision=controller_decision,
        )
        self._decisions.append(decision)
        self._cursor += 1
        return decision

    def finalize(self) -> tuple[int, ...]:
        _require(
            self.is_complete,
            f"cannot finalize before chunks 1..{self.chunk_count - 1} are consumed",
        )
        return self._controller.finalize()


def offline_rank_top_k(
    risk_by_chunk: Mapping[int, float],
    budget: int,
    *,
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> tuple[int, ...]:
    """Offline-only top-K curve; this is not the deployable online policy."""

    parsed_budget = _validate_budget(budget, chunk_count=chunk_count)
    candidates = candidate_chunks(chunk_count)
    risks = _complete_chunk_values(
        risk_by_chunk, "risk_by_chunk", chunk_count=chunk_count
    )
    selected = sorted(
        sorted(candidates, key=lambda chunk: (-risks[chunk], chunk))[:parsed_budget]
    )
    return tuple(selected)


def exact_random_top1_expectation(
    delta_by_chunk: Mapping[int, float],
    *,
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> float:
    """Return the exact Random Top-1 expectation over all 15 positions."""

    deltas = _complete_chunk_values(
        delta_by_chunk, "delta_by_chunk", chunk_count=chunk_count
    )
    return math.fsum(deltas.values()) / len(candidate_chunks(chunk_count))


@dataclass(frozen=True)
class FullVideoRollout:
    """Result of one actual sequential rollout for a complete correction set."""

    corrected_chunks: tuple[int, ...]
    full_video_quality: float
    target_call_count: int
    rollout_id: str
    chunk_count: int = DEFAULT_CHUNK_COUNT

    def __post_init__(self) -> None:
        parsed_count = _require_integer(self.chunk_count, "chunk_count", minimum=2)
        object.__setattr__(self, "chunk_count", parsed_count)
        parsed_chunks = _validate_chunk_tuple(
            self.corrected_chunks,
            "corrected_chunks",
            chunk_count=parsed_count,
        )
        object.__setattr__(self, "corrected_chunks", parsed_chunks)
        object.__setattr__(
            self,
            "full_video_quality",
            _require_finite(self.full_video_quality, "full_video_quality"),
        )
        parsed_calls = _require_integer(self.target_call_count, "target_call_count", minimum=1)
        _require(
            parsed_calls == 1 + len(parsed_chunks),
            "target_call_count must include chunk 0 plus every corrected chunk",
        )
        _require_nonempty_string(self.rollout_id, "rollout_id")


@dataclass(frozen=True)
class GreedyOraclePlan:
    """Audit record for a sequential full-rollout Greedy Oracle search."""

    selected_chunks: tuple[int, ...]
    stage_full_video_quality: tuple[float, ...]
    evaluations: tuple[FullVideoRollout, ...]
    label: str = "Greedy Oracle"
    is_global_upper_bound: bool = False


def greedy_oracle_plan(
    *,
    budget: int,
    rollout_scorer: Callable[[tuple[int, ...]], FullVideoRollout],
    chunk_count: int = DEFAULT_CHUNK_COUNT,
) -> GreedyOraclePlan:
    """Plan K corrections by repeatedly scoring real full-video rollouts.

    The API intentionally accepts neither per-position gains nor a mapping of
    scores.  At every greedy stage it invokes ``rollout_scorer`` for each
    complete candidate correction set and compares the returned *full-video*
    quality directly.  The result is explicitly labelled ``Greedy Oracle`` and
    is never represented as a global upper bound.
    """

    parsed_budget = _validate_budget(budget, chunk_count=chunk_count)
    candidates = candidate_chunks(chunk_count)
    _require(callable(rollout_scorer), "rollout_scorer must be callable")
    selected: tuple[int, ...] = ()
    stage_quality: list[float] = []
    evaluations: list[FullVideoRollout] = []

    for _ in range(parsed_budget):
        stage_results: list[FullVideoRollout] = []
        for candidate in candidates:
            if candidate in selected:
                continue
            requested = tuple(sorted((*selected, candidate)))
            result = rollout_scorer(requested)
            _require(
                isinstance(result, FullVideoRollout),
                "rollout_scorer must return FullVideoRollout, not a scalar or single gain",
            )
            _require(
                result.corrected_chunks == requested,
                "rollout_scorer returned a result for a different correction set",
            )
            _require(
                result.chunk_count == chunk_count,
                "rollout_scorer returned a result for a different chunk_count",
            )
            stage_results.append(result)
            evaluations.append(result)
        _require(bool(stage_results), "Greedy Oracle stage has no candidates")
        best = min(
            stage_results,
            key=lambda result: (-result.full_video_quality, result.corrected_chunks),
        )
        selected = best.corrected_chunks
        stage_quality.append(best.full_video_quality)

    return GreedyOraclePlan(
        selected_chunks=selected,
        stage_full_video_quality=tuple(stage_quality),
        evaluations=tuple(evaluations),
    )


__all__ = [
    "BOUNDARY_GATE_ENTRY_DELTA_THRESHOLD",
    "BOUNDARY_GATE_ERANK_DELTA_THRESHOLD",
    "BOUNDARY_GATE_EXIT_DELTA_THRESHOLD",
    "BOUNDARY_GATE_MAX_ERANK",
    "BOUNDARY_GATE_MAX_WITNESS_SCORE",
    "BOUNDARY_GATE_METRIC_ID",
    "BOUNDARY_GATE_METRIC_VERSION",
    "BOUNDARY_GATE_MIN_ERANK",
    "BOUNDARY_GATE_MIN_WITNESS_SCORE",
    "BoundaryEntryRejectionDecision",
    "BoundarySmoothnessGate",
    "BoundarySmoothnessGateDecision",
    "CANDIDATE_CHUNKS",
    "DEFAULT_CHUNK_COUNT",
    "FIRST_TARGET_CHUNK",
    "RANDOM_SELECTION_HASH_FIELDS",
    "CalibrationRiskDistribution",
    "CoverageConstrainedOnlineDecision",
    "CoverageConstrainedOnlineRankGuidedPolicy",
    "FullVideoRollout",
    "GreedyOraclePlan",
    "OnlineDecision",
    "OnlineRankGuidedPolicy",
    "LookaheadCoverageConstrainedOnlineDecision",
    "LookaheadCoverageConstrainedOnlineRankGuidedPolicy",
    "PolicyContractError",
    "RankRiskCalibrator",
    "SuddenDropRiskCalibrator",
    "calibration_drop_cdf_risk",
    "calibration_rank_cdf_risk",
    "candidate_chunks",
    "causal_erank_sudden_drop",
    "exact_random_top1_expectation",
    "greedy_oracle_plan",
    "higher_empirical_quantile",
    "offline_rank_top_k",
    "random_positions",
    "random_selection_digest",
    "uniform_positions",
]
