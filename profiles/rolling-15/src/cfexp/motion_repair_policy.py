"""Explicit acceptance ablations for diagnosing long-horizon motion loss.

These are different algorithms, not parity-preserving system optimizations.
Both retain the measured boundary reference, even when it does not veto the
candidate. Complete paired evidence is required in every arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .correction_policy import BoundarySmoothnessGate, BoundarySmoothnessGateDecision


@dataclass(frozen=True)
class MotionAcceptanceDecision:
    metric_id: str
    acceptance_rule: str
    evidence_valid: bool
    accept_target: bool
    reason: str
    boundary_reference: BoundarySmoothnessGateDecision


class MotionAcceptanceGate:
    def __init__(self, rule: str):
        if rule not in {"rank_only", "proposal_always"}:
            raise ValueError("unknown motion acceptance ablation")
        self.rule = rule

    def reject_if_entry_worse(self, **_kwargs: Any) -> None:
        # A boundary measurement cannot decide either ablation. In particular,
        # old entry-only checkpoints cannot be replayed into these trajectories.
        return None

    def decide(self, **kwargs: Any) -> MotionAcceptanceDecision:
        reference = BoundarySmoothnessGate().decide(**kwargs)
        accepted = reference.evidence_valid and (
            self.rule == "proposal_always" or reference.erank_recovered
        )
        reason = ("accepted:" + self.rule if accepted else
                  "rejected:erank_not_recovered" if reference.evidence_valid else reference.reason)
        return MotionAcceptanceDecision(
            metric_id="paired_successor_acceptance_ablation_v1",
            acceptance_rule=self.rule, evidence_valid=reference.evidence_valid,
            accept_target=accepted, reason=reason, boundary_reference=reference,
        )
