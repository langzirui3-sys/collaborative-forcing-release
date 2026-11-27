"""Serializable causal eRank decision evaluated inside the first Draft forward.

This uses the same drop/CDF/budget rule as the late replacement control. It does
not predict future quality and never accepts a latent from a stopped Draft.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping

from .correction_policy import (SuddenDropRiskCalibrator, causal_erank_sudden_drop,
                                higher_empirical_quantile)


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_control(value, *, chunk_index):
    keys = {'schema', 'prior_ranks', 'calibration_id', 'calibration_drop_scores',
            'remaining_budget', 'remaining_slots'}
    require(isinstance(value, Mapping) and set(value) == keys
            and value['schema'] == 'early_erank_control_v1', 'invalid early control fields')
    ranks = value['prior_ranks']
    require(isinstance(ranks, list) and len(ranks) == max(0, chunk_index-1)
            and all(type(x) in (int, float) and math.isfinite(x) and x >= 1 for x in ranks),
            'early control must bind every previous candidate rank')
    scores = value['calibration_drop_scores']
    require(isinstance(scores, list) and bool(scores)
            and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in scores),
            'invalid early calibration scores')
    SuddenDropRiskCalibrator(value['calibration_id'], tuple(scores))
    k, n = value['remaining_budget'], value['remaining_slots']
    require(type(k) is int and type(n) is int and 0 <= k <= n and n > 0,
            'invalid early remaining budget')
    return dict(value)


def decide(control, rank):
    require(type(rank) in (float, int) and math.isfinite(rank) and rank >= 1,
            'invalid early effective rank')
    cal = SuddenDropRiskCalibrator(control['calibration_id'], tuple(control['calibration_drop_scores']))
    drop = causal_erank_sudden_drop([*control['prior_ranks'], rank], lookback_chunks=4)
    risk = cal.risk(drop)
    k, n = control['remaining_budget'], control['remaining_slots']
    threshold = None if k in (0, n) else higher_empirical_quantile(cal.calibration_risks, 1-k/n)
    replace = False if k == 0 else True if k == n else risk >= threshold
    return {'effective_rank': float(rank), 'drop_score': drop, 'risk_score': risk,
            'replace_with_target': replace, 'threshold': threshold}


@dataclass(frozen=True)
class StoppedDraft:
    """A probe and a pending commit capability, deliberately not GeneratedChunk."""
    request: object
    session_id: str
    pending_token: str
    decision: Mapping
    evidence: Mapping
