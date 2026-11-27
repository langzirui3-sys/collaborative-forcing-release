from __future__ import annotations

import inspect
import random
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cfexp.correction_policy import (  # noqa: E402
    BOUNDARY_GATE_METRIC_ID,
    BOUNDARY_GATE_METRIC_VERSION,
    BoundarySmoothnessGate,
    CANDIDATE_CHUNKS,
    CalibrationRiskDistribution,
    CoverageConstrainedOnlineRankGuidedPolicy,
    FullVideoRollout,
    LookaheadCoverageConstrainedOnlineRankGuidedPolicy,
    OnlineRankGuidedPolicy,
    PolicyContractError,
    RankRiskCalibrator,
    SuddenDropRiskCalibrator,
    calibration_drop_cdf_risk,
    calibration_rank_cdf_risk,
    candidate_chunks,
    causal_erank_sudden_drop,
    exact_random_top1_expectation,
    greedy_oracle_plan,
    higher_empirical_quantile,
    offline_rank_top_k,
    random_positions,
    random_selection_digest,
    uniform_positions,
)


class BoundarySmoothnessGateTests(unittest.TestCase):
    @staticmethod
    def evidence(**overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "original_successor_erank": 10.0,
            "corrected_successor_erank": 11.0,
            "draft_entry_witness_score": 0.4,
            "target_entry_witness_score": 0.3,
            "draft_exit_witness_score": 0.5,
            "target_exit_witness_score": 0.4,
        }
        values.update(overrides)
        return values

    def test_accepts_only_when_all_three_paired_checks_pass(self) -> None:
        gate = BoundarySmoothnessGate()
        decision = gate.decide(**self.evidence())

        self.assertTrue(decision.evidence_valid)
        self.assertTrue(decision.erank_recovered)
        self.assertTrue(decision.entry_not_worse)
        self.assertTrue(decision.exit_not_worse)
        self.assertTrue(decision.accept_target)
        self.assertEqual(decision.reason, "accepted")

    def test_rejects_every_failed_check_and_their_combinations(self) -> None:
        gate = BoundarySmoothnessGate()
        cases = {
            "erank": {"corrected_successor_erank": 9.0},
            "entry": {"target_entry_witness_score": 0.6},
            "exit": {"target_exit_witness_score": 0.7},
            "all": {
                "corrected_successor_erank": 9.0,
                "target_entry_witness_score": 0.6,
                "target_exit_witness_score": 0.7,
            },
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                decision = gate.decide(**self.evidence(**overrides))
                self.assertTrue(decision.evidence_valid)
                self.assertFalse(decision.accept_target)
                self.assertTrue(decision.reason.startswith("rejected:"))

    def test_equality_passes_and_metric_identity_and_thresholds_are_frozen(self) -> None:
        decision = BoundarySmoothnessGate().decide(
            **self.evidence(
                corrected_successor_erank=10.0,
                target_entry_witness_score=0.4,
                target_exit_witness_score=0.5,
            )
        )

        self.assertTrue(decision.accept_target)
        self.assertEqual(decision.metric_id, BOUNDARY_GATE_METRIC_ID)
        self.assertEqual(decision.metric_version, BOUNDARY_GATE_METRIC_VERSION)
        self.assertEqual(decision.erank_delta, 0.0)
        self.assertEqual(decision.entry_delta, 0.0)
        self.assertEqual(decision.exit_delta, 0.0)
        self.assertEqual(decision.erank_delta_threshold, 0.0)
        self.assertEqual(decision.entry_delta_threshold, 0.0)
        self.assertEqual(decision.exit_delta_threshold, 0.0)

    def test_missing_nonfinite_and_out_of_range_evidence_fail_closed(self) -> None:
        gate = BoundarySmoothnessGate()
        cases = (
            {},
            {"original_successor_erank": None},
            {"corrected_successor_erank": float("nan")},
            {"corrected_successor_erank": float("inf")},
            {"original_successor_erank": 0.0},
            {"corrected_successor_erank": 257.0},
            {"draft_entry_witness_score": -0.1},
            {"target_entry_witness_score": 1.1},
            {"draft_exit_witness_score": True},
            {"target_exit_witness_score": "0.4"},
            {"target_exit_witness_score": 10**10000},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(case=index):
                values = overrides if not overrides else self.evidence(**overrides)
                decision = gate.decide(**values)
                self.assertFalse(decision.evidence_valid)
                self.assertFalse(decision.accept_target)
                self.assertFalse(decision.erank_recovered)
                self.assertFalse(decision.entry_not_worse)
                self.assertFalse(decision.exit_not_worse)
                self.assertTrue(decision.reason.startswith("invalid_evidence:"))


class FixedPositionPolicyTests(unittest.TestCase):
    def test_uniform_half_up_positions_are_frozen(self) -> None:
        self.assertEqual(uniform_positions(2), (5, 11))
        self.assertEqual(uniform_positions(3), (4, 8, 12))
        self.assertEqual(uniform_positions(4), (3, 6, 10, 13))

    def test_uniform_rejects_an_invalid_budget(self) -> None:
        for invalid in (0, 16, True, 2.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(PolicyContractError):
                    uniform_positions(invalid)  # type: ignore[arg-type]

    def test_long_horizon_uniform_positions_and_random_budget(self) -> None:
        self.assertEqual(uniform_positions(3, chunk_count=60), (15, 30, 45))
        self.assertEqual(
            uniform_positions(6, chunk_count=120),
            (17, 34, 51, 69, 86, 103),
        )
        selected = random_positions(
            protocol_id="insight2-long60-v1",
            prompt_id="prompt-001",
            generation_seed=17,
            budget=6,
            repeat_id=0,
            chunk_count=120,
        )
        self.assertEqual(len(selected), 6)
        self.assertTrue(set(selected) <= set(candidate_chunks(120)))

    def test_random_is_reproducible_unique_and_hash_ordered(self) -> None:
        kwargs = {
            "protocol_id": "insight2-v1",
            "prompt_id": "prompt-007",
            "generation_seed": 20261012,
            "budget": 4,
            "repeat_id": 9,
        }
        first = random_positions(**kwargs)
        second = random_positions(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first, (1, 3, 4, 5))
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        digests = {
            chunk: random_selection_digest(**kwargs, chunk_index=chunk)
            for chunk in CANDIDATE_CHUNKS
        }
        expected = tuple(sorted(sorted(CANDIDATE_CHUNKS, key=digests.get)[:4]))
        self.assertEqual(first, expected)
        self.assertEqual(
            digests[1],
            "1618000da4765a695c546c39082216788e93434273a05e69bedd1ebd9f13442f",
        )

    def test_random_selection_is_independent_of_noise_rng_namespace(self) -> None:
        kwargs = {
            "protocol_id": "insight2-v1",
            "prompt_id": "prompt-012",
            "generation_seed": 123456,
            "budget": 3,
            "repeat_id": 4,
        }
        before = random_positions(**kwargs)
        # Simulate unrelated changes in model-noise streams.  The policy has no
        # RNG object or noise namespace input and must remain byte-for-byte stable.
        for namespace in ("draft-initial-noise-v1", "target-initial-noise-v2"):
            noise_rng = random.Random(f"{namespace}:{kwargs['generation_seed']}")
            _ = [noise_rng.getrandbits(64) for _ in range(100)]
            random.seed(namespace)
            _ = [random.random() for _ in range(100)]
        after = random_positions(**kwargs)
        self.assertEqual(before, after)

    def test_random_contract_rejects_ambiguous_fields(self) -> None:
        with self.assertRaisesRegex(PolicyContractError, "protocol_id"):
            random_positions(
                protocol_id="",
                prompt_id="p",
                generation_seed=1,
                budget=2,
                repeat_id=0,
            )
        with self.assertRaisesRegex(PolicyContractError, "repeat_id"):
            random_positions(
                protocol_id="v1",
                prompt_id="p",
                generation_seed=1,
                budget=2,
                repeat_id=-1,
            )


class RiskCalibrationTests(unittest.TestCase):
    def test_rank_cdf_uses_midrank_ties(self) -> None:
        calibration = (1.0, 2.0, 2.0, 4.0)
        # One greater, two equal: (1 + 0.5 * 2) / 4.
        self.assertEqual(calibration_rank_cdf_risk(2.0, calibration), 0.5)
        self.assertEqual(calibration_rank_cdf_risk(0.5, calibration), 1.0)
        self.assertEqual(calibration_rank_cdf_risk(5.0, calibration), 0.0)

    def test_cached_calibration_preserves_midrank_values_and_order(self) -> None:
        ranks = (4.0, 2.0, 1.0, 2.0)
        drops = (0.75, 0.25, 0.0, 0.25)
        rank = RankRiskCalibrator("rank", ranks)
        drop = SuddenDropRiskCalibrator("drop", drops)
        for calibrator, reference, original, probes in (
            (rank, ranks, calibration_rank_cdf_risk, (0.5, 2.0, 3.0, 5.0)),
            (drop, drops, calibration_drop_cdf_risk, (0.0, 0.25, 0.5, 1.0)),
        ):
            expected = tuple(original(value, reference) for value in reference)
            self.assertEqual(calibrator.calibration_risks, expected)
            self.assertEqual(calibrator.as_risk_distribution().risks, expected)
            for value in probes:
                self.assertEqual(calibrator.risk(value), original(value, reference))
            with self.assertRaises(PolicyContractError):
                calibrator.risk(float("nan"))

    def test_higher_quantile_ties_and_boundaries(self) -> None:
        values = (0.1, 0.4, 0.4, 0.9)
        self.assertEqual(higher_empirical_quantile(values, 0.0), 0.1)
        self.assertEqual(higher_empirical_quantile(values, 0.5), 0.4)
        self.assertEqual(higher_empirical_quantile(values, 0.75), 0.9)
        self.assertEqual(higher_empirical_quantile(values, 1.0), 0.9)

    def test_calibration_fails_closed_for_empty_or_nonfinite_input(self) -> None:
        with self.assertRaises(PolicyContractError):
            calibration_rank_cdf_risk(2.0, ())
        with self.assertRaises(PolicyContractError):
            RankRiskCalibrator(calibration_id="cal", ranks=(1.0, float("nan")))
        with self.assertRaises(PolicyContractError):
            CalibrationRiskDistribution(calibration_id="cal", risks=(0.2, 1.1))

    def test_causal_erank_drop_uses_only_trailing_past(self) -> None:
        self.assertEqual(causal_erank_sudden_drop((8.0,)), 0.0)
        self.assertEqual(causal_erank_sudden_drop((8.0, 8.0, 4.0)), 0.5)
        # With a four-chunk lookback, the ancient 100 cannot affect the score.
        self.assertEqual(
            causal_erank_sudden_drop((100.0, 8.0, 8.0, 8.0, 8.0, 4.0)),
            0.5,
        )
        self.assertEqual(causal_erank_sudden_drop((4.0, 8.0)), 0.0)

    def test_drop_calibration_is_upper_tail_midrank(self) -> None:
        calibration = (0.0, 0.25, 0.25, 0.75)
        self.assertEqual(calibration_drop_cdf_risk(0.25, calibration), 0.5)
        self.assertEqual(calibration_drop_cdf_risk(1.0, calibration), 1.0)
        calibrator = SuddenDropRiskCalibrator("cal-drop", calibration)
        self.assertEqual(calibrator.risk(0.25), 0.5)


class OnlineRankGuidedTests(unittest.TestCase):
    @staticmethod
    def calibrator() -> RankRiskCalibrator:
        return RankRiskCalibrator(
            calibration_id="independent-calibration-split-v1",
            ranks=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        )

    def test_decisions_cannot_observe_future_risks(self) -> None:
        first = OnlineRankGuidedPolicy(
            budget=3,
            calibration=self.calibrator(),
            evaluation_id="evaluation-split-v1",
        )
        second = OnlineRankGuidedPolicy(
            budget=3,
            calibration=self.calibrator(),
            evaluation_id="evaluation-split-v1",
        )
        common_prefix = (0.1, 0.99, 0.3, 0.8)
        first_prefix = [
            first.decide(chunk_index=chunk, risk_score=risk)
            for chunk, risk in zip(CANDIDATE_CHUNKS, common_prefix)
        ]
        second_prefix = [
            second.decide(chunk_index=chunk, risk_score=risk)
            for chunk, risk in zip(CANDIDATE_CHUNKS, common_prefix)
        ]
        # Hypothetical futures differ, but are never passed to decide().
        future_a = [0.0] * 11
        future_b = [1.0] * 11
        self.assertNotEqual(future_a, future_b)
        self.assertEqual(first_prefix, second_prefix)

    def test_exact_k_is_forced_even_when_all_risks_are_low(self) -> None:
        policy = OnlineRankGuidedPolicy(
            budget=4,
            calibration=self.calibrator(),
            evaluation_id="evaluation-split-v1",
        )
        decisions = [
            policy.decide(chunk_index=chunk, risk_score=0.0)
            for chunk in CANDIDATE_CHUNKS
        ]
        self.assertEqual(policy.finalize(), (12, 13, 14, 15))
        self.assertEqual(sum(decision.replace_with_target for decision in decisions), 4)
        self.assertTrue(
            all(
                decision.reason == "forced_remaining_slots_equal_budget"
                for decision in decisions[-4:]
            )
        )

    def test_budget_exhaustion_never_selects_more_than_k(self) -> None:
        policy = OnlineRankGuidedPolicy(
            budget=2,
            calibration=self.calibrator(),
            evaluation_id="evaluation-split-v1",
        )
        decisions = [
            policy.decide(chunk_index=chunk, risk_score=1.0)
            for chunk in CANDIDATE_CHUNKS
        ]
        self.assertEqual(policy.finalize(), (1, 2))
        self.assertEqual(sum(decision.replace_with_target for decision in decisions), 2)
        self.assertTrue(all(not item.replace_with_target for item in decisions[2:]))

    def test_threshold_is_higher_empirical_calibration_quantile(self) -> None:
        calibration = self.calibrator().as_risk_distribution()
        policy = OnlineRankGuidedPolicy(
            budget=3,
            calibration=calibration,
            evaluation_id="evaluation-split-v1",
        )
        decision = policy.decide(chunk_index=1, risk_score=0.5)
        expected_probability = 1.0 - 3 / 15
        self.assertAlmostEqual(decision.quantile_probability or -1, expected_probability)
        self.assertEqual(
            decision.threshold,
            higher_empirical_quantile(calibration.risks, expected_probability),
        )

    def test_online_order_and_independent_split_are_enforced(self) -> None:
        with self.assertRaisesRegex(PolicyContractError, "differ"):
            OnlineRankGuidedPolicy(
                budget=2,
                calibration=self.calibrator(),
                evaluation_id="independent-calibration-split-v1",
            )
        policy = OnlineRankGuidedPolicy(
            budget=2,
            calibration=self.calibrator(),
            evaluation_id="evaluation-split-v1",
        )
        with self.assertRaisesRegex(PolicyContractError, "expected chunk 1"):
            policy.decide(chunk_index=2, risk_score=0.5)
        with self.assertRaisesRegex(PolicyContractError, "cannot finalize"):
            policy.finalize()

    def test_long_policy_consumes_exact_horizon_and_budget(self) -> None:
        calibration = SuddenDropRiskCalibrator(
            calibration_id="long-calibration",
            drop_scores=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        )
        policy = OnlineRankGuidedPolicy(
            budget=3,
            calibration=calibration,
            evaluation_id="long-evaluation",
            chunk_count=60,
        )
        decisions = [
            policy.decide(chunk_index=chunk, risk_score=0.0)
            for chunk in candidate_chunks(60)
        ]
        self.assertEqual(policy.finalize(), (57, 58, 59))
        self.assertEqual(sum(item.replace_with_target for item in decisions), 3)


class CoverageConstrainedOnlineRankGuidedTests(unittest.TestCase):
    @staticmethod
    def calibration() -> CalibrationRiskDistribution:
        return CalibrationRiskDistribution(
            calibration_id="coverage-calibration-v1",
            risks=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        )

    def policy(self, **overrides: object) -> CoverageConstrainedOnlineRankGuidedPolicy:
        kwargs: dict[str, object] = {
            "budget": 6,
            "calibration": self.calibration(),
            "evaluation_id": "coverage-evaluation-v1",
            "chunk_count": 120,
            "window_radius_chunks": 4,
            "min_gap_chunks": 8,
            "max_gap_chunks": 20,
        }
        kwargs.update(overrides)
        return CoverageConstrainedOnlineRankGuidedPolicy(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def consume(
        policy: CoverageConstrainedOnlineRankGuidedPolicy, risk: float
    ) -> tuple[int, ...]:
        for chunk in candidate_chunks(policy.chunk_count):
            policy.decide(chunk_index=chunk, risk_score=risk)
        return policy.finalize()

    def test_uniform_anchors_windows_and_local_threshold_are_auditable(self) -> None:
        policy = self.policy()
        self.assertEqual(policy.anchors, (17, 34, 51, 69, 86, 103))
        self.assertEqual(
            policy.static_windows,
            ((13, 21), (30, 38), (47, 55), (65, 73), (82, 90), (99, 107)),
        )

        decisions = [
            policy.decide(chunk_index=chunk, risk_score=1.0)
            for chunk in range(1, 14)
        ]
        decision = decisions[-1]
        self.assertTrue(decision.replace_with_target)
        self.assertEqual(decision.correction_ordinal, 1)
        self.assertEqual(decision.anchor_chunk, 17)
        self.assertEqual(
            (decision.static_window_start, decision.static_window_end),
            (13, 21),
        )
        self.assertEqual(
            (decision.feasible_window_start, decision.feasible_window_end),
            (13, 20),
        )
        self.assertEqual((decision.eligible_start, decision.deadline), (13, 20))
        self.assertEqual(decision.local_slots_including_current, 8)
        self.assertAlmostEqual(decision.quantile_probability or -1.0, 0.875)
        self.assertEqual(
            decision.threshold,
            higher_empirical_quantile(self.calibration().risks, 0.875),
        )
        self.assertEqual(decision.reason, "risk_at_or_above_local_threshold")

    def test_all_high_selects_earliest_eligible_chunk_in_each_window(self) -> None:
        policy = self.policy()
        selected = self.consume(policy, 1.0)
        self.assertEqual(selected, (13, 30, 47, 65, 82, 99))
        self.assertEqual(len(selected), policy.budget)

    def test_all_low_forces_each_dynamic_deadline_and_exact_k(self) -> None:
        policy = self.policy()
        selected = self.consume(policy, 0.0)
        self.assertEqual(selected, (20, 38, 55, 73, 90, 107))
        self.assertEqual(len(selected), policy.budget)
        forced = [
            item.chunk_index
            for item in policy.decisions
            if item.reason == "forced_at_dynamic_deadline"
        ]
        self.assertEqual(tuple(forced), selected)

    def test_selection_has_one_per_window_and_frozen_gap_bounds(self) -> None:
        policy = self.policy()
        risks = (0.0, 1.0, 0.4, 0.95, 0.2, 0.8, 0.3)
        for chunk in candidate_chunks(policy.chunk_count):
            policy.decide(
                chunk_index=chunk,
                risk_score=risks[(chunk - 1) % len(risks)],
            )
        selected = policy.finalize()
        self.assertEqual(len(selected), 6)
        self.assertTrue(
            all(
                start <= chunk <= end
                for chunk, (start, end) in zip(selected, policy.static_windows)
            )
        )
        points = (0, *selected, policy.chunk_count - 1)
        gaps = tuple(right - left for left, right in zip(points, points[1:]))
        self.assertTrue(all(8 <= gap <= 20 for gap in gaps))

    def test_common_history_decisions_are_independent_of_future_risks(self) -> None:
        first = self.policy()
        second = self.policy()
        common_prefix = [0.0] * 35
        first_prefix = [
            first.decide(chunk_index=chunk, risk_score=risk)
            for chunk, risk in zip(candidate_chunks(120), common_prefix)
        ]
        second_prefix = [
            second.decide(chunk_index=chunk, risk_score=risk)
            for chunk, risk in zip(candidate_chunks(120), common_prefix)
        ]
        self.assertEqual(first_prefix, second_prefix)

        for chunk in range(36, 120):
            first.decide(chunk_index=chunk, risk_score=1.0)
            second.decide(chunk_index=chunk, risk_score=0.0)
        self.assertNotEqual(first.finalize(), second.finalize())

    def test_bad_constraint_parameters_and_split_identity_fail_closed(self) -> None:
        invalid_overrides = (
            {"window_radius_chunks": -1},
            {"window_radius_chunks": True},
            {"min_gap_chunks": 0},
            {"min_gap_chunks": 21, "max_gap_chunks": 20},
            {"window_radius_chunks": 9},
            {"max_gap_chunks": 10},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(PolicyContractError):
                    self.policy(**overrides)
        with self.assertRaisesRegex(PolicyContractError, "differ"):
            self.policy(evaluation_id="coverage-calibration-v1")


class LookaheadCoverageConstrainedOnlineRankGuidedTests(unittest.TestCase):
    @staticmethod
    def policy() -> LookaheadCoverageConstrainedOnlineRankGuidedPolicy:
        return LookaheadCoverageConstrainedOnlineRankGuidedPolicy(
            budget=6,
            calibration=CalibrationRiskDistribution(
                calibration_id="lookahead-calibration-v1",
                risks=(0.1, 0.3, 0.5, 0.7, 0.9),
            ),
            evaluation_id="lookahead-evaluation-v1",
            chunk_count=120,
            window_radius_chunks=4,
            min_gap_chunks=8,
            max_gap_chunks=20,
        )

    def test_preview_binding_exact_k_and_budget_exhausted_suffix(self) -> None:
        policy = self.policy()
        decisions = []
        for chunk in candidate_chunks(120):
            if policy.preview_required(chunk_index=chunk):
                signal = chunk + 1
                risk = 0.0
            elif policy.remaining_budget > 0:
                signal = chunk
                risk = 0.0
            else:
                signal = None
                risk = None
            decisions.append(
                policy.decide(
                    chunk_index=chunk,
                    signal_chunk_index=signal,
                    risk_score=risk,
                )
            )

        self.assertEqual(policy.finalize(), (20, 38, 55, 73, 90, 107))
        self.assertTrue(decisions[0].preview_required)
        self.assertEqual(decisions[0].signal_chunk_index, 2)
        exhausted = decisions[107]
        self.assertEqual(exhausted.chunk_index, 108)
        self.assertFalse(exhausted.preview_required)
        self.assertIsNone(exhausted.signal_chunk_index)
        self.assertIsNone(exhausted.risk_score)
        terminal = decisions[-1]
        self.assertTrue(terminal.terminal_flush)
        self.assertIsNone(terminal.signal_chunk_index)
        self.assertIn("budget_exhausted_without_preview", terminal.reason)

    def test_preview_must_be_immediate_successor(self) -> None:
        policy = self.policy()
        with self.assertRaisesRegex(PolicyContractError, "immediate successor"):
            policy.decide(
                chunk_index=1,
                signal_chunk_index=3,
                risk_score=0.5,
            )

    def test_terminal_flush_never_requires_out_of_range_signal(self) -> None:
        policy = self.policy()
        for chunk in range(1, 119):
            needs_preview = policy.preview_required(chunk_index=chunk)
            policy.decide(
                chunk_index=chunk,
                signal_chunk_index=chunk + 1 if needs_preview else None,
                risk_score=0.0 if needs_preview else None,
            )
        self.assertFalse(policy.preview_required(chunk_index=119))
        decision = policy.decide(
            chunk_index=119,
            signal_chunk_index=None,
            risk_score=None,
        )
        self.assertTrue(decision.terminal_flush)
        self.assertNotEqual(decision.signal_chunk_index, 120)
        self.assertEqual(policy.finalize(), (20, 38, 55, 73, 90, 107))


class OfflineAndOracleTests(unittest.TestCase):
    def test_offline_top_k_is_explicit_and_tie_deterministic(self) -> None:
        risks = {chunk: 0.1 for chunk in CANDIDATE_CHUNKS}
        risks[8] = 0.9
        risks[4] = 0.9
        risks[12] = 0.8
        self.assertEqual(offline_rank_top_k(risks, 2), (4, 8))
        self.assertIn("offline", offline_rank_top_k.__name__)

    def test_random_top1_expectation_is_all_position_mean(self) -> None:
        deltas = {chunk: float(chunk) for chunk in CANDIDATE_CHUNKS}
        self.assertEqual(exact_random_top1_expectation(deltas), 8.0)
        with self.assertRaisesRegex(PolicyContractError, "each candidate"):
            exact_random_top1_expectation({1: 1.0})

    def test_greedy_oracle_scores_real_combinations_not_single_gain_sums(self) -> None:
        calls: list[tuple[int, ...]] = []

        def scorer(chunks: tuple[int, ...]) -> FullVideoRollout:
            calls.append(chunks)
            if len(chunks) == 1:
                quality = {1: 0.90, 2: 0.80}.get(chunks[0], 0.10)
            elif chunks == (1, 3):
                quality = 1.20  # Interaction wins despite chunk 3's low singleton score.
            elif chunks == (1, 2):
                quality = 0.70
            else:
                quality = 0.20
            return FullVideoRollout(
                corrected_chunks=chunks,
                full_video_quality=quality,
                target_call_count=1 + len(chunks),
                rollout_id="rollout-" + "-".join(str(chunk) for chunk in chunks),
            )

        plan = greedy_oracle_plan(budget=2, rollout_scorer=scorer)
        self.assertEqual(plan.selected_chunks, (1, 3))
        self.assertEqual(plan.stage_full_video_quality, (0.90, 1.20))
        self.assertEqual(plan.label, "Greedy Oracle")
        self.assertFalse(plan.is_global_upper_bound)
        self.assertEqual(len(calls), 15 + 14)
        self.assertIn((1, 2), calls)
        self.assertIn((1, 3), calls)
        self.assertNotIn("single", inspect.signature(greedy_oracle_plan).parameters)

    def test_greedy_oracle_rejects_scalar_or_mismatched_results(self) -> None:
        with self.assertRaisesRegex(PolicyContractError, "FullVideoRollout"):
            greedy_oracle_plan(budget=1, rollout_scorer=lambda _: 0.5)  # type: ignore[arg-type]

        def mismatched(_: tuple[int, ...]) -> FullVideoRollout:
            return FullVideoRollout(
                corrected_chunks=(15,),
                full_video_quality=0.5,
                target_call_count=2,
                rollout_id="wrong-set",
            )

        with self.assertRaisesRegex(PolicyContractError, "different correction set"):
            greedy_oracle_plan(budget=1, rollout_scorer=mismatched)

    def test_full_rollout_requires_chunk_zero_plus_all_corrections(self) -> None:
        with self.assertRaisesRegex(PolicyContractError, "chunk 0"):
            FullVideoRollout(
                corrected_chunks=(2, 5),
                full_video_quality=0.7,
                target_call_count=2,
                rollout_id="missing-call",
            )


if __name__ == "__main__":
    unittest.main()
