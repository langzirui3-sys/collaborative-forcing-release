"""Explicit horizons for the paired Ours/Target duration experiment.

This experiment is separate from the frozen main-table-v1 (120 chunks only).
All horizons share a same-revision, independently calibrated 120-chunk C10 CDF.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

RESULT_SCHEMA = "ours_target_duration_result_v1"
RAW_SCHEMA = "ours_target_duration_rows_v1"


@dataclass(frozen=True)
class DurationProfile:
    nominal_seconds: int
    chunk_count: int
    proposal_budget: int

    @property
    def frame_count(self) -> int:
        return self.chunk_count * 12 - 3

    def metadata(self) -> dict:
        return {
            "experiment_id": "ours-target-duration-v1",
            "nominal_seconds": self.nominal_seconds,
            "chunk_count": self.chunk_count,
            "frame_count": self.frame_count,
            "fps": 24,
            "exact_duration_seconds": str(Fraction(self.frame_count, 24)),
            "ours_proposal_budget": self.proposal_budget,
            "budget_rule": "ceil_chunk_count_divided_by_20",
            "generation_timing_policy": "continuous_generation_loop_excluding_load_conditioning_decode_encode_vbench_calibration",
            "calibration_chunk_count": 120,
            "calibration_policy": "independent_C10_same_revision_shared_across_horizons",
        }


def duration_profile(seconds: int) -> DurationProfile:
    if type(seconds) is not int or seconds not in (15, 30, 60):
        raise ValueError("duration must be exactly 15, 30, or 60 nominal seconds")
    chunks = seconds * 2
    return DurationProfile(seconds, chunks, (chunks + 19) // 20)
