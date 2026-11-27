"""Frozen system-only comparison against Target, with an explicit historical C10.

This is a named experimental control, not a general relaxation of receipt
verification. All binding fields except the declared calibration producer SHA
must still match, and both the receipt and the resulting binding are pinned.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from typing import Any, Mapping

from .calibration_receipt import _canonical_bytes

CONTROL_ID = "optimized-target-duration-v2"
CALIBRATION_REVISION = "3afcbb2bb2bfb945ae0e18b3ca9f6b6047641b87"
CALIBRATION_RECEIPT_SHA256 = "6faca2ca1ff0406daafd3ac766aa34d3ce4c35506c8bc93a81abb513c6e4f51b"
CALIBRATION_BINDING_SHA256 = "1ffa52e8e43f41bf055b3f39865135ed8dd614cddb1970c28ce70418be377ad5"
OURS_FLAGS = {"CFEXP_DRAFT_KV_REUSE": "1", "CFEXP_DRAFT_KV_FORK": "1",
              "CFEXP_BOUNDARY_EARLY_REJECT": "1"}


class OptimizedDurationError(RuntimeError):
    pass


def contract() -> dict[str, Any]:
    return {"id": CONTROL_ID, "calibration_code_revision": CALIBRATION_REVISION,
            "calibration_receipt_sha256": CALIBRATION_RECEIPT_SHA256,
            "calibration_binding_sha256": CALIBRATION_BINDING_SHA256,
            "ours_feature_flags": dict(OURS_FLAGS), "algorithm_changed": False,
            "quality_measured": True, "prompt_count": 12, "seed": 20260911}


def validate_invocation(*, method: str, seconds: int | None, smoke: str | None) -> None:
    if type(seconds) is not int or seconds not in (15, 30, 60) or smoke is not None:
        raise OptimizedDurationError("optimized comparison requires a complete 15/30/60s formal population")
    if method not in {"ours", "full_target"}:
        raise OptimizedDurationError("unknown optimized comparison method")
    if method == "ours":
        if any(os.environ.get(key) != value for key, value in OURS_FLAGS.items()):
            raise OptimizedDurationError("optimized comparison requires all three explicit feature flags")
        if os.environ.get("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SHA256") != CALIBRATION_RECEIPT_SHA256:
            raise OptimizedDurationError("optimized comparison requires the exact frozen C10 receipt")


def frozen_control_binding(current: Mapping[str, Any]) -> dict[str, Any]:
    binding = deepcopy(dict(current))
    binding["code_revision"] = CALIBRATION_REVISION
    if hashlib.sha256(_canonical_bytes(binding)).hexdigest() != CALIBRATION_BINDING_SHA256:
        raise OptimizedDurationError("frozen C10 control binding drifted outside the declared producer revision")
    return binding


class ReadOnlyCalibrationBackend:
    """Receipt restoration must fail instead of silently regenerating evidence."""
    def __init__(self, wrapped: Any):
        self.wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        if name in {"generate", "generate_persistent", "materialize", "start_trajectory"}:
            raise OptimizedDurationError("frozen C10 revalidation attempted new calibration compute")
        return getattr(self.wrapped, name)
