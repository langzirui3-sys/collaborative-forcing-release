"""Budget causality, exactly-once generation and resumable burst routing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cfexp.rank_burst import RankBurstPolicy, materialize_rank_bursts
from cfexp.correction_policy import CalibrationRiskDistribution, SuddenDropRiskCalibrator
from test_insight2_campaign import CampaignFixture


class BurstPolicyTests(unittest.TestCase):
    def test_short_positive_opening_delays_zero_drop_and_keeps_the_budget(self):
        from cfexp.erank_acceptance_pilot import (
            SHORT_POSITIVE_OPENING_ARM, burst_length_for_arm,
            observation_horizon_for_arm, split_transfer_options,
        )
        from cfexp.rank_burst import validate_burst_schedule

        arm = SHORT_POSITIVE_OPENING_ARM
        options = split_transfer_options(arm, 30)
        self.assertEqual(options, {"split_target_burst": True, "opening_target_burst": True,
                                   "require_positive_drop": True})
        self.assertEqual(split_transfer_options(arm, 60), {})
        horizon = observation_horizon_for_arm(arm)
        length = burst_length_for_arm(arm)
        for drop, start, reason in (
            (0.0, 9, "threshold_trigger_gated"),
            (0.01, 8, "risk_at_or_above_calibration_threshold"),
        ):
            with self.subTest(drop=drop):
                policy = RankBurstPolicy(
                    chunk_count=60, target_budget=36, burst_length=length,
                    calibration=CalibrationRiskDistribution("frozen", (0.25, 0.25)),
                    evaluation_id="positive-opening", observation_horizon=horizon, **options,
                )
                decisions = []
                for index in range(1, 60):
                    role = policy.next_role
                    risk = 0.25 if role == "draft" else None
                    observed_drop = drop if role == "draft" else None
                    decision = policy.advance(
                        chunk_index=index, role=role, risk_score=risk,
                        erank_drop_score=observed_drop,
                    )
                    decisions.append(dict(
                        chunk_index=index, decision=decision, risk_score=risk,
                        rank=10.0 if role == "draft" else None, erank_drop_score=observed_drop,
                    ))
                selected = policy.finalize()
                self.assertEqual(len(selected), 36)
                self.assertEqual(selected[:5], (1, 2, 3, 4, start))
                self.assertEqual(decisions[6]["decision"]["observation_decision"]["reason"], reason)
                validate_burst_schedule(
                    dict(decisions=decisions, accepted_chunks=list(selected),
                         proposed_chunks=list(selected)),
                    budget=36, burst_length=length, chunk_count=60,
                    observation_horizon=horizon, **options,
                )

    def test_long_self_forcing_single_block_routes_preserve_budget_and_cache_reuse(self):
        from cfexp.erank_acceptance_pilot import (
            SINGLE_BLOCK_TRANSFER_ARMS, burst_length_for_arm,
            observation_horizon_for_arm, split_transfer_options,
        )
        from cfexp.rank_burst import validate_burst_schedule

        for seconds, budget in ((30, 6), (60, 8)):
            with self.subTest(seconds=seconds), tempfile.TemporaryDirectory() as directory:
                arm = SINGLE_BLOCK_TRANSFER_ARMS[seconds]
                options = split_transfer_options(arm, seconds)
                chunks = seconds * 2
                horizon = observation_horizon_for_arm(arm)
                self.assertEqual(options, {"split_target_burst": True})
                self.assertEqual(split_transfer_options(arm, 90 - seconds), {})
                self.assertEqual(burst_length_for_arm(arm), 1)
                fixture = CampaignFixture(Path(directory))
                campaign = fixture.campaign(chunk_count=chunks)
                args = dict(prompt=fixture.evaluation[0], seed=17, budget=budget,
                    calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)),
                    burst_length=1, observation_horizon=horizon, **options)
                key, route, selected, history = materialize_rank_bursts(campaign, **args)
                self.assertEqual(history.next_chunk_index, chunks)
                self.assertEqual(len(selected), budget)
                self.assertTrue(all(b - a > 1 for a, b in zip(selected, selected[1:])))
                self.assertEqual(materialize_rank_bursts(campaign, **args),
                                 (key, route, selected, history))
                self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)),
                                 (chunks - budget - 1, budget + 1))
                policy = RankBurstPolicy(chunk_count=chunks, target_budget=budget,
                    burst_length=1, calibration=args['calibrator'],
                    evaluation_id='single', observation_horizon=horizon, **options)
                decisions = []
                for index in range(1, chunks):
                    role = policy.next_role
                    risk = 0. if role == 'draft' else None
                    decision = policy.advance(chunk_index=index, role=role, risk_score=risk)
                    decisions.append(dict(chunk_index=index, decision=decision,
                        rank=10. if role == 'draft' else None, risk_score=risk))
                positions = list(policy.finalize())
                validate_burst_schedule(dict(decisions=decisions, accepted_chunks=positions,
                    proposed_chunks=positions), budget=budget, burst_length=1,
                    chunk_count=chunks, observation_horizon=horizon, **options)

    def test_low_motion_reserves_the_same_burst_budget_and_checks_evidence(self):
        from cfexp.rank_burst import validate_burst_schedule

        policy = RankBurstPolicy(chunk_count=60, target_budget=6, burst_length=2,
            calibration=CalibrationRiskDistribution('frozen', (0.5, 0.6)),
            evaluation_id='motion', observation_horizon=8, split_target_burst=True,
            motion_trigger=True)
        decisions = []
        for i in range(1, 60):
            role = policy.next_role
            risk = 0.1 if role == 'draft' else None
            decision = policy.advance(chunk_index=i, role=role, risk_score=risk,
                                      motion_energy=0.2 if policy.observes_motion else None)
            decisions.append(dict(chunk_index=i, decision=decision,
                                  rank=10. if role == 'draft' else None, risk_score=risk))
        selected = policy.finalize()
        self.assertEqual(selected, (3, 4, 22, 23, 42, 43))
        for i in selected[::2]:
            self.assertEqual(decisions[i-2]['decision']['observation_decision']['reason'],
                             'additional_trigger')
        evidence = dict(decisions=decisions, accepted_chunks=list(selected), proposed_chunks=list(selected))
        geometry = dict(budget=6, burst_length=2, chunk_count=60,
                        observation_horizon=8, split_target_burst=True)
        validate_burst_schedule(evidence, **geometry, motion_trigger=True)
        with self.assertRaisesRegex(ValueError, 'unexpected motion evidence'):
            validate_burst_schedule(evidence, **geometry)
        decisions[0]['decision']['motion_energy'] = 0.25
        with self.assertRaisesRegex(ValueError, 'lacks two low-motion'):
            validate_burst_schedule(evidence, **geometry, motion_trigger=True)

    def test_motion_route_resumes_from_its_committed_draft_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            campaign = fixture.campaign(chunk_count=60)
            args = dict(prompt=fixture.evaluation[0], seed=17, budget=6, burst_length=2,
                observation_horizon=8, split_target_burst=True, motion_trigger=True,
                calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            with patch('cfexp.rank_burst.latent_motion_energy', return_value=0.2) as probe:
                first = materialize_rank_bursts(campaign, **args)
                self.assertTrue(first[1].endswith('-split-motion'))
                self.assertEqual(probe.call_count, 24)
                self.assertEqual(materialize_rank_bursts(campaign, **args), first)
                self.assertEqual(probe.call_count, 48)
            self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)), (53, 7))

    def test_tail_aligned_k6_shifts_only_the_last_window_and_preserves_budget(self):
        from cfexp.rank_burst import validate_burst_schedule

        geometry = dict(budget=6, burst_length=2, chunk_count=60,
                        observation_horizon=8, split_target_burst=True)
        calibration = SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2))
        for risk, expected_start in ((0., 58), (1., 51)):
            with self.subTest(risk=risk):
                routes = []
                for tail in (False, True):
                    policy = RankBurstPolicy(chunk_count=60, target_budget=6,
                        burst_length=2, calibration=calibration, evaluation_id='k6',
                        observation_horizon=8, split_target_burst=True, final_window_at_tail=tail)
                    decisions = []
                    for i in range(1, 60):
                        role = policy.next_role
                        value = risk if role == 'draft' else None
                        decision = policy.advance(chunk_index=i, role=role, risk_score=value)
                        decisions.append(dict(chunk_index=i, decision=decision,
                            rank=10. if role == 'draft' else None, risk_score=value))
                    selected = policy.finalize()
                    routes.append(selected)
                    trajectory = dict(decisions=decisions, accepted_chunks=list(selected),
                                      proposed_chunks=list(selected))
                    validate_burst_schedule(trajectory, **geometry, final_window_at_tail=tail)
                    with self.assertRaises(ValueError):
                        validate_burst_schedule(trajectory, **geometry, final_window_at_tail=not tail)
                self.assertEqual(routes[0][:4], routes[1][:4])
                self.assertEqual(routes[1][4:], (expected_start, expected_start + 1))

    def test_tail_aligned_k6_rollout_has_its_own_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            campaign = fixture.campaign(chunk_count=60)
            args = dict(prompt=fixture.evaluation[0], seed=17, budget=6, burst_length=2,
                observation_horizon=8, split_target_burst=True, final_window_at_tail=True,
                calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            first = materialize_rank_bursts(campaign, **args)
            self.assertTrue(first[1].endswith('-split-tail'))
            self.assertIn(first[2][-2], range(51, 59))
            self.assertEqual(materialize_rank_bursts(campaign, **args), first)
            self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)), (53, 7))
            with self.assertRaisesRegex(ValueError, 'three-window K6'):
                materialize_rank_bursts(campaign, **dict(args, budget=4))

    def test_later_final_window_preserves_earlier_bursts_and_four_draft_blocks(self):
        from cfexp.rank_burst import validate_burst_schedule

        for chunks, budget, low_start, high_start in ((60, 6, 54, 47), (120, 8, 114, 107)):
            for risk, start in ((0., low_start), (1., high_start)):
                with self.subTest(chunks=chunks, risk=risk):
                    common = dict(chunk_count=chunks, burst_length=2, observation_horizon=8,
                                  split_target_burst=True)
                    tail = dict(final_window_at_tail=True, tail_draft_reserve=4)
                    routes = []
                    for options in ({}, tail):
                        policy = RankBurstPolicy(**common, **options, target_budget=budget,
                            calibration=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)),
                            evaluation_id='later-final-burst')
                        decisions = []
                        for index in range(1, chunks):
                            role = policy.next_role
                            score = risk if role == 'draft' else None
                            decision = policy.advance(chunk_index=index, role=role, risk_score=score)
                            decisions.append(dict(chunk_index=index, decision=decision, risk_score=score,
                                rank=1. if role == 'draft' else None))
                        selected = policy.finalize()
                        routes.append(selected)
                        trajectory = dict(decisions=decisions, accepted_chunks=list(selected),
                                          proposed_chunks=list(selected))
                        validate_burst_schedule(trajectory, **common, **options, budget=budget)
                    self.assertEqual(routes[0][:-2], routes[1][:-2])
                    self.assertEqual(routes[1][-2:], (start, start + 1))
                    self.assertGreaterEqual(chunks - routes[1][-1] - 1, 4)
                    with self.assertRaises(ValueError):
                        validate_burst_schedule(trajectory, **common, budget=budget)

    def test_longlive_k8_windows_spread_equal_budget_with_single_or_paired_calls(self):
        from cfexp.rank_burst import valid_split_burst_count, validate_burst_schedule

        for width, count in ((1, 8), (2, 4)):
            with self.subTest(width=width):
                self.assertTrue(valid_split_burst_count(
                    count, burst_length=width, seconds=60, split=True))
                self.assertFalse(valid_split_burst_count(
                    count, burst_length=width, seconds=30, split=True))
                calibration = SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2))
                for risk in (0., 1.):
                    policy = RankBurstPolicy(chunk_count=120, target_budget=8,
                        burst_length=width, calibration=calibration, evaluation_id='k8',
                        observation_horizon=8, split_target_burst=True)
                    decisions = []
                    for index in range(1, 120):
                        role = policy.next_role
                        value = risk if role == 'draft' else None
                        decision = policy.advance(chunk_index=index, role=role, risk_score=value)
                        decisions.append(dict(chunk_index=index, decision=decision,
                            rank=10. if role == 'draft' else None, risk_score=value))
                    selected = policy.finalize()
                    self.assertEqual(len(selected), 8)
                    self.assertGreater(selected[-1], 90)
                    for i in range(count):
                        start = 111 * i // count + width * i + 2
                        self.assertIn(selected[width * i], range(start, start + 8))
                    validate_burst_schedule(dict(decisions=decisions,
                        accepted_chunks=list(selected), proposed_chunks=list(selected)),
                        budget=8, burst_length=width, chunk_count=120,
                        observation_horizon=8, split_target_burst=True)

    def test_longlive_k6_three_block_bursts_have_two_windows_and_preserve_history(self):
        from cfexp.rank_burst import valid_split_burst_count, validate_burst_schedule

        self.assertTrue(valid_split_burst_count(2, burst_length=3, seconds=30, split=True))
        self.assertFalse(valid_split_burst_count(2, burst_length=3, seconds=15, split=True))
        self.assertFalse(valid_split_burst_count(2, burst_length=3, seconds=60, split=True))
        self.assertFalse(valid_split_burst_count(3, burst_length=3, seconds=30, split=True))
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            campaign = fixture.campaign(chunk_count=60)
            geometry = dict(budget=6, burst_length=3, observation_horizon=8,
                            split_target_burst=True)
            options = dict(geometry, prompt=fixture.evaluation[0], seed=17,
                           calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            key, route, selected, history = materialize_rank_bursts(campaign, **options)
            self.assertEqual(route, 'committed-rank-burst3-k6-probe8-split')
            self.assertIn(selected[0], range(2, 10))
            self.assertIn(selected[3], range(31, 39))
            for start in (0, 3):
                self.assertEqual(selected[start:start + 3],
                                 tuple(range(selected[start], selected[start] + 3)))
            decisions = [json.loads(campaign._checkpoint_path('rollouts', key, index).read_text())['metadata']
                         for index in range(1, 60)]
            trace = dict(decisions=decisions, accepted_chunks=list(selected),
                         proposed_chunks=list(selected))
            validate_burst_schedule(trace, chunk_count=60, **geometry)
            with self.assertRaises(ValueError):
                validate_burst_schedule(trace, chunk_count=60, **dict(geometry, burst_length=2))
            self.assertEqual(materialize_rank_bursts(campaign, **options),
                             (key, route, selected, history))
            self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)), (53, 7))

    def test_split_bursts_use_two_disjoint_observation_windows_and_resume(self):
        from cfexp.rank_burst import validate_burst_schedule

        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            campaign = fixture.campaign(chunk_count=30)
            options = dict(prompt=fixture.evaluation[0], seed=17, budget=4, burst_length=2,
                           observation_horizon=8, split_target_burst=True,
                           calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            key, route, selected, history = materialize_rank_bursts(campaign, **options)
            self.assertTrue(route.endswith('-split'))
            first, second, third, fourth = selected
            self.assertIn(first, range(2, 10))
            self.assertIn(third, range(16, 24))
            self.assertEqual((second, fourth), (first + 1, third + 1))
            decisions = []
            for index in range(1, 30):
                row = json.loads(campaign._checkpoint_path('rollouts', key, index).read_text())
                decisions.append(row['metadata'])
            trajectory = dict(decisions=decisions, accepted_chunks=list(selected),
                              proposed_chunks=list(selected))
            validate_burst_schedule(trajectory, budget=4, burst_length=2,
                                    observation_horizon=8, split_target_burst=True)
            with self.assertRaises(ValueError):
                validate_burst_schedule(trajectory, budget=4, burst_length=2, observation_horizon=8)
            self.assertEqual(materialize_rank_bursts(campaign, **options),
                             (key, route, selected, history))
            self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)), (25, 5))

    def test_six_target_blocks_use_three_causal_windows_without_discarding_draft(self):
        from cfexp.rank_burst import audit_native_model_burst

        with tempfile.TemporaryDirectory() as td:
            fixture = CampaignFixture(Path(td))
            campaign = fixture.campaign(chunk_count=60)
            key, _, selected, history = materialize_rank_bursts(
                campaign, prompt=fixture.evaluation[0], seed=17, budget=6, burst_length=2,
                observation_horizon=8, split_target_burst=True,
                calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            first, middle, late = selected[::2]
            self.assertIn(first, range(2, 10))
            self.assertIn(middle, range(21, 29))
            self.assertIn(late, range(41, 49))
            self.assertEqual(selected[1::2], (first + 1, middle + 1, late + 1))
            self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)), (53, 7))
            previous = None
            for i, segment in enumerate(history.segments):
                path = campaign._checkpoint_path('rollouts', key, i)
                row = json.loads(path.read_text())
                diagnostics = row['generated']['diagnostics']
                if segment.producer_role == 'target':
                    diagnostics['native_correction_state'] = {
                        'native_session_id': 'one', 'session_recreated': False}
                else:
                    replayed = i if previous is None else i - previous - 1
                    diagnostics['kv_reuse'] = {'reused_chunks': i - replayed,
                                               'replayed_chunks': replayed}
                    previous = i
                path.write_text(json.dumps(row))
            options = dict(key=key, history=history, selected=selected, draft_model_id='longlive_1.3b',
                           burst_length=2, observation_horizon=8, split_target_burst=True)
            with patch('cfexp.krea_chunk_backend.validate_native_correction_state'):
                audit = audit_native_model_burst(campaign, **options, split_burst_count=3)
                self.assertEqual(audit['split_burst_count'], 3)
                self.assertEqual(audit['target_chunks'], [0, *selected])
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    audit_native_model_burst(campaign, **options)

    def test_prefill_callbacks_bind_three_accepted_target_prefixes_before_one_draft_handoff(self):
        with tempfile.TemporaryDirectory() as td:
            f=CampaignFixture(Path(td));c=f.campaign(chunk_count=30);prefixes=[];committed=[]
            args=dict(prompt=f.evaluation[0],seed=17,budget=4,burst_length=4,
                observation_horizon=8,calibrator=SuddenDropRiskCalibrator('frozen',(0.,.05,.1,.2)))
            def prefill(**kw):
                self.assertEqual(committed[-1],kw['history'])
                self.assertEqual(kw['history'].segments[-1].producer_role,'target')
                prefixes.append(kw)
            _,_,selected,h=materialize_rank_bursts(c,**args,on_target_prefix=prefill,on_commit=committed.append)
            self.assertEqual(len(prefixes),3)
            self.assertEqual([p['history'].next_chunk_index for p in prefixes],[i+1 for i in selected[:3]])
            self.assertEqual({p['next_draft_chunk_index'] for p in prefixes},{selected[-1]+1})
            self.assertEqual((len(f.draft.calls),len(f.target.calls)),(25,5))
            with self.assertRaisesRegex(ValueError,'supported fresh burst geometry'):
                materialize_rank_bursts(c,**args,on_target_prefix=prefill)

    def test_split_prefill_schedules_the_first_committed_block_of_each_burst(self):
        for chunks, budget in ((30, 4), (60, 6), (120, 8)):
            with self.subTest(chunks=chunks, budget=budget), tempfile.TemporaryDirectory() as td:
                fixture = CampaignFixture(Path(td))
                campaign = fixture.campaign(chunk_count=chunks)
                updates = []
                key, _, selected, history = materialize_rank_bursts(
                    campaign, prompt=fixture.evaluation[0], seed=17, budget=budget, burst_length=2,
                    observation_horizon=8, split_target_burst=True,
                    calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)),
                    on_target_prefix=lambda **request: updates.append(request))
                self.assertEqual(len(updates), budget // 2)
                self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)),
                    (chunks - 1 - budget, budget + 1))
                for update, start in zip(updates, selected[::2]):
                    self.assertEqual(update['history'], history.prefix_before(start + 1))
                    self.assertEqual(update['next_draft_chunk_index'], start + 2)
                    checkpoint = campaign._checkpoint_path('rollouts', key, start + 1)
                    first, last = json.loads(checkpoint.read_text())['metadata']['target_rpc_interval']
                    self.assertLessEqual(first, last)

    def test_dense_routes_prefill_each_accepted_prefix_after_the_opening(self):
        configurations = ((60, 4, 36, False, 27), (60, 4, 28, False, 21),
                          (60, 4, 28, True, 18), (60, 4, 36, True, 24),
                          (120, 8, 88, True, 70), (120, 8, 72, True, 56))
        for chunks, width, budget, opening, expected_updates in configurations:
            with self.subTest(chunks=chunks, budget=budget, opening=opening), \
                    tempfile.TemporaryDirectory() as directory:
                fixture = CampaignFixture(Path(directory))
                campaign = fixture.campaign(chunk_count=chunks)
                updates = []
                _, _, selected, history = materialize_rank_bursts(
                    campaign, prompt=fixture.evaluation[0], seed=17, budget=budget, burst_length=width,
                    observation_horizon=2, split_target_burst=True, opening_target_burst=opening,
                    calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)),
                    on_target_prefix=lambda **request: updates.append(request))
                self.assertEqual((len(fixture.draft.calls), len(fixture.target.calls)),
                                 (chunks - 1 - budget, budget + 1))
                self.assertEqual(len(updates), expected_updates)
                starts = selected[::width]
                if opening:
                    self.assertEqual(starts[0], 1)
                    starts = starts[1:]
                prefills = width - 1
                for offset, start in enumerate(starts):
                    burst = updates[offset * prefills:(offset + 1) * prefills]
                    for index, request in enumerate(burst, start=start + 1):
                        self.assertEqual(request['history'], history.prefix_before(index))
                        self.assertEqual(request['next_draft_chunk_index'], start + width)

    def test_model_native_audit_distinguishes_rolling_restart_from_longlive_suffix_replay(self):
        from cfexp.rank_burst import audit_native_model_burst
        with tempfile.TemporaryDirectory() as td:
            fixture = CampaignFixture(Path(td))
            campaign = fixture.campaign(chunk_count=30)
            key, _, selected, history = materialize_rank_bursts(
                campaign, prompt=fixture.evaluation[0], seed=17, budget=4, burst_length=4,
                observation_horizon=8, calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            for model_id in ('longlive_1.3b', 'rolling_forcing_1.3b'):
                previous = None
                for index, segment in enumerate(history.segments):
                    path = campaign._checkpoint_path('rollouts', key, index)
                    row = json.loads(path.read_text())
                    diagnostics = row['generated']['diagnostics']
                    if segment.producer_role == 'target':
                        diagnostics['native_correction_state'] = {
                            'native_session_id': 'one', 'session_recreated': False}
                    else:
                        missing = index if previous is None else index - previous - 1
                        reused = previous is not None and missing == 0
                        diagnostics.update(
                            kv_reuse={'reused_chunks': index-missing, 'replayed_chunks': missing},
                            rolling_state={'chunk_index': index, 'reused_rolling_state': reused,
                                           'replayed_history_blocks': 0 if reused else index,
                                           'discarded_future_state': previous is not None and not reused,
                                           'state_policy': 'single_active_branch_clean_history_restart_v1'})
                        previous = index
                    path.write_text(json.dumps(row))
                with patch('cfexp.krea_chunk_backend.validate_native_correction_state'):
                    audit = audit_native_model_burst(campaign, key=key, history=history,
                                                     selected=selected, draft_model_id=model_id)
                    self.assertEqual(audit['target_chunks'], [0, *selected])
                    self.assertEqual(len(audit['draft_states']), 25)
                    path = campaign._checkpoint_path('rollouts', key, selected[-1]+1)
                    row = json.loads(path.read_text())
                    row['generated']['diagnostics']['kv_reuse']['replayed_chunks'] = 0
                    row['generated']['diagnostics']['rolling_state']['replayed_history_blocks'] = 0
                    path.write_text(json.dumps(row))
                    with self.assertRaisesRegex(ValueError, 'LongLive|Rolling'):
                        audit_native_model_burst(campaign, key=key, history=history,
                                                 selected=selected, draft_model_id=model_id)

    def test_commit_callback_covers_root_resumed_prefix_and_each_new_chunk_once(self):
        with tempfile.TemporaryDirectory() as td:
            f=CampaignFixture(Path(td));c=f.campaign(chunk_count=30)
            args=dict(prompt=f.evaluation[0],seed=17,budget=4,burst_length=4,
                observation_horizon=8,calibrator=SuddenDropRiskCalibrator('frozen',(0.,.05,.1,.2)))
            seen=[];roles=[]
            _,_,_,history=materialize_rank_bursts(c,**args,on_commit=seen.append,before_generate=roles.append)
            self.assertEqual([h.next_chunk_index for h in seen],list(range(1,31)))
            self.assertEqual(seen,[history.prefix_before(i) for i in range(1,31)])
            self.assertEqual(roles,[s.producer_role for s in history.segments[1:]])
            calls=(len(f.draft.calls),len(f.target.calls))
            resumed=[]
            materialize_rank_bursts(c,**args,on_commit=resumed.append)
            self.assertEqual(resumed,seen)
            self.assertEqual((len(f.draft.calls),len(f.target.calls)),calls)


    def test_early_observation_window_forces_exact_budget_without_new_generations(self):
        from cfexp.rank_burst import validate_burst_schedule
        from cfexp.erank_acceptance_pilot import observation_horizon_for_arm
        for horizon in (8,12):
            self.assertEqual(observation_horizon_for_arm(f'rank_burst4_p{horizon}_k4'),horizon)
            for risk in (0.,1.):
                p=RankBurstPolicy(chunk_count=30,target_budget=4,burst_length=4,
                    observation_horizon=horizon,calibration=CalibrationRiskDistribution('independent',(.2,.4,.6,.8)),
                    evaluation_id='evaluation')
                for i in range(1,30):
                    role=p.next_role;p.advance(chunk_index=i,role=role,risk_score=risk if role=='draft' else None)
                selected=p.finalize()
                self.assertEqual(len(selected),4)
                self.assertLessEqual(selected[0],horizon+1)
                self.assertEqual(selected,tuple(range(selected[0],selected[0]+4)))
            with tempfile.TemporaryDirectory() as td:
                f=CampaignFixture(Path(td));c=f.campaign(chunk_count=30)
                args=dict(prompt=f.evaluation[0],seed=17,budget=4,burst_length=4,
                    observation_horizon=horizon,calibrator=SuddenDropRiskCalibrator('frozen',(0.,.05,.1,.2)))
                key,_,selected,h=materialize_rank_bursts(c,**args)
                self.assertEqual((len(f.draft.calls),len(f.target.calls)),(25,5))
                self.assertLessEqual(selected[0],horizon+1)
                for call in f.draft.calls+f.target.calls:
                    self.assertEqual(call['history_hash'],h.prefix_before(call['chunk_index']).head_sha256)
                trajectory={'accepted_chunks':list(selected),'proposed_chunks':list(selected),
                    'decisions':[json.loads(c._checkpoint_path('rollouts',key,i).read_text())['metadata'] for i in range(1,30)]}
                validate_burst_schedule(trajectory,budget=4,burst_length=4,observation_horizon=horizon)
                with self.assertRaises(ValueError):
                    validate_burst_schedule(trajectory,budget=4,burst_length=4)
                again=materialize_rank_bursts(f.campaign(chunk_count=30),**args)
                self.assertEqual(again[3].head_sha256,h.head_sha256)
                self.assertEqual(len(f.draft.calls)+len(f.target.calls),30)
        for horizon in (0,26,True):
            with self.assertRaises(ValueError):
                RankBurstPolicy(chunk_count=30,target_budget=4,burst_length=4,observation_horizon=horizon,
                    calibration=CalibrationRiskDistribution('independent',(.2,.4,.6,.8)),evaluation_id='evaluation')

    def test_delayed_observation_preserves_budget_and_rejects_early_decisions(self):
        from cfexp.rank_burst import validate_burst_schedule

        with tempfile.TemporaryDirectory() as td:
            fixture = CampaignFixture(Path(td))
            campaign = fixture.campaign(chunk_count=30)
            args = dict(prompt=fixture.evaluation[0], seed=17, budget=4, burst_length=4,
                        observation_horizon=4, observation_offset=20,
                        calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            key, _, selected, history = materialize_rank_bursts(campaign, **args)
            self.assertIn(selected[0], range(22, 26))
            self.assertEqual(len(fixture.target.calls), 5)
            self.assertEqual(len(fixture.draft.calls), 25)
            for call in fixture.draft.calls + fixture.target.calls:
                self.assertEqual(call['history_hash'],
                                 history.prefix_before(call['chunk_index']).head_sha256)
            decisions = [json.loads(campaign._checkpoint_path('rollouts', key, index).read_text())['metadata']
                         for index in range(1, 30)]
            trajectory = dict(accepted_chunks=list(selected), proposed_chunks=list(selected),
                              decisions=decisions)
            validate_burst_schedule(trajectory, budget=4, burst_length=4,
                                    observation_horizon=4, observation_offset=20)
            with self.assertRaises(ValueError):
                validate_burst_schedule(trajectory, budget=4, burst_length=4, observation_horizon=4)
            self.assertEqual(materialize_rank_bursts(campaign, **args)[3].head_sha256,
                             history.head_sha256)
            self.assertEqual(len(fixture.draft.calls) + len(fixture.target.calls), 30)

    def test_opening_target_burst_hands_off_once_and_preserves_accepted_prefix(self):
        from cfexp.rank_burst import validate_burst_schedule

        with tempfile.TemporaryDirectory() as td:
            fixture = CampaignFixture(Path(td))
            campaign = fixture.campaign(chunk_count=30)
            args = dict(prompt=fixture.evaluation[0], seed=17, budget=4, burst_length=4,
                        observation_horizon=8, opening_target_burst=True,
                        calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            key, _, selected, history = materialize_rank_bursts(campaign, **args)
            self.assertEqual(selected, (1, 2, 3, 4))
            self.assertEqual([x['chunk_index'] for x in fixture.target.calls], list(range(5)))
            self.assertEqual([x['chunk_index'] for x in fixture.draft.calls], list(range(5, 30)))
            for call in fixture.draft.calls + fixture.target.calls:
                self.assertEqual(call['history_hash'],
                                 history.prefix_before(call['chunk_index']).head_sha256)
            decisions = [json.loads(campaign._checkpoint_path('rollouts', key, index).read_text())['metadata']
                         for index in range(1, 30)]
            trajectory = dict(accepted_chunks=list(selected), proposed_chunks=list(selected),
                              decisions=decisions)
            validate_burst_schedule(trajectory, budget=4, burst_length=4, observation_horizon=8,
                                    opening_target_burst=True)
            with self.assertRaises(ValueError):
                validate_burst_schedule(trajectory, budget=4, burst_length=4, observation_horizon=8)
            self.assertEqual(materialize_rank_bursts(campaign, **args)[3].head_sha256,
                             history.head_sha256)
            self.assertEqual(len(fixture.draft.calls) + len(fixture.target.calls), 30)

    def test_dense_opening_prefills_later_bursts_only(self):
        from cfexp.erank_acceptance_pilot import SHORT_TARGET_OPENING_ARM, split_transfer_options

        options = split_transfer_options(SHORT_TARGET_OPENING_ARM, 30)
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            campaign = fixture.campaign(chunk_count=60)
            prefills = []

            def prefill(*, prompt, history, next_draft_chunk_index):
                prefills.append((history.next_chunk_index, next_draft_chunk_index))

            _, _, selected, history = materialize_rank_bursts(
                campaign, prompt=fixture.evaluation[0], seed=17,
                budget=36, burst_length=4, observation_horizon=2, on_target_prefix=prefill,
                calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)), **options,
            )
            self.assertEqual(selected[:4], (1, 2, 3, 4))
            self.assertEqual((len(fixture.target.calls), len(fixture.draft.calls)), (37, 23))
            self.assertEqual(fixture.draft.calls[0]['chunk_index'], 5)
            self.assertEqual(history.prefix_before(5).head_sha256, fixture.draft.calls[0]['history_hash'])
            self.assertEqual(prefills, [(start + offset, start + 4)
                                       for start in selected[4::4] for offset in (1, 2, 3)])

    def test_five_block_bursts_cover_more_of_the_video_with_native_prefill(self):
        from cfexp.erank_acceptance_pilot import WIDE_TARGET_BURST_ARM, split_transfer_options

        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            campaign = fixture.campaign(chunk_count=60)
            prefills = []

            def prefill(*, prompt, history, next_draft_chunk_index):
                prefills.append((history.next_chunk_index, next_draft_chunk_index))

            _, _, selected, _ = materialize_rank_bursts(
                campaign, prompt=fixture.evaluation[0], seed=17,
                budget=40, burst_length=5, observation_horizon=2, on_target_prefix=prefill,
                calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)),
                **split_transfer_options(WIDE_TARGET_BURST_ARM, 30),
            )
            self.assertEqual((len(fixture.target.calls), len(fixture.draft.calls)), (41, 19))
            self.assertEqual(prefills, [(start + offset, start + 5)
                                       for start in selected[::5] for offset in (1, 2, 3, 4)])

    def test_long_opening_uses_one_of_the_eleven_reserved_target_bursts(self):
        from cfexp.erank_acceptance_pilot import (LONG_TARGET_OPENING_ARM,
            burst_length_for_arm, observation_horizon_for_arm, split_transfer_options)
        from cfexp.rank_burst import validate_burst_schedule

        arm = LONG_TARGET_OPENING_ARM
        options = split_transfer_options(arm, 60)
        self.assertEqual(options, {"split_target_burst": True, "opening_target_burst": True})
        self.assertEqual(split_transfer_options(arm, 30), {})
        width = burst_length_for_arm(arm)
        horizon = observation_horizon_for_arm(arm)
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            campaign = fixture.campaign(chunk_count=120)
            key, _, selected, history = materialize_rank_bursts(
                campaign, prompt=fixture.evaluation[0], seed=17, budget=88, burst_length=width,
                observation_horizon=horizon, **options,
                calibrator=SuddenDropRiskCalibrator('frozen', (0., .05, .1, .2)))
            self.assertEqual(selected[:8], tuple(range(1, 9)))
            self.assertEqual((len(fixture.target.calls), len(fixture.draft.calls)), (89, 31))
            first_draft = fixture.draft.calls[0]
            self.assertEqual(first_draft['chunk_index'], 9)
            self.assertEqual(first_draft['history_hash'], history.prefix_before(9).head_sha256)
            decisions = [json.loads(campaign._checkpoint_path('rollouts', key, i).read_text())['metadata']
                         for i in range(1, 120)]
            validate_burst_schedule(
                dict(accepted_chunks=list(selected), proposed_chunks=list(selected), decisions=decisions),
                budget=88, burst_length=width, observation_horizon=horizon, chunk_count=120, **options)

    def make(self, budget=14):
        return RankBurstPolicy(chunk_count=30, target_budget=budget, burst_length=2,
            calibration=CalibrationRiskDistribution('independent',(.2,.4,.6,.8)), evaluation_id='evaluation')


    def test_low_budget_groupings_commit_once_and_enforce_causal_schedule(self):
        from cfexp.rank_burst import validate_burst_schedule
        for budget in (4,8):
            for length in (1,2,4):
                with self.subTest(budget=budget,length=length),tempfile.TemporaryDirectory() as td:
                    f=CampaignFixture(Path(td));c=f.campaign(chunk_count=30)
                    args=dict(prompt=f.evaluation[0],seed=17,budget=budget,burst_length=length,
                        calibrator=SuddenDropRiskCalibrator('frozen',(0.,.05,.1,.2)))
                    key,route,selected,h=materialize_rank_bursts(c,**args)
                    self.assertEqual(len(f.draft.calls),29-budget)
                    self.assertEqual(len(f.target.calls),budget+1)
                    self.assertEqual(len(selected),budget)
                    for call in f.draft.calls+f.target.calls:
                        self.assertEqual(call['history_hash'],h.prefix_before(call['chunk_index']).head_sha256)
                    trajectory={"accepted_chunks":list(selected),"proposed_chunks":list(selected),
                        "decisions":[json.loads(c._checkpoint_path("rollouts",key,i).read_text())["metadata"] for i in range(1,30)]}
                    validate_burst_schedule(trajectory,budget=budget,burst_length=length)
                    with self.assertRaises(ValueError):
                        validate_burst_schedule(trajectory,budget=budget,burst_length=2 if length!=2 else 4)
                    again=materialize_rank_bursts(f.campaign(chunk_count=30),**args)
                    self.assertEqual(again[3].head_sha256,h.head_sha256)
                    self.assertEqual(len(f.draft.calls)+len(f.target.calls),30)


    def test_exact_budget_no_discarded_drafts_and_no_future_signal(self):
        for budget in (2,8,14):
            for risk in (0.,.5,1.):
                p=self.make(budget);roles=[];signals=[]
                for i in range(1,30):
                    role=p.next_role;roles.append(role)
                    d=p.advance(chunk_index=i,role=role,risk_score=risk if role=='draft' else None)
                    if d['scheduled_target_chunks']:
                        self.assertEqual(d['scheduled_target_chunks'],[i+1,i+2]);signals.append(i)
                selected=p.finalize()
                self.assertEqual(len(selected),budget)
                self.assertEqual(roles.count('draft'),29-budget)
                self.assertEqual(roles.count('target'),budget)
                self.assertTrue(all(roles[i-1]=='draft' for i in signals))
                self.assertEqual(tuple(i for i,r in enumerate(roles,1) if r=='target'),selected)

    def test_native_audit_rejects_full_history_rebuild(self):
        from types import SimpleNamespace
        from cfexp.rank_burst import verify_burst_reuse
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'checkpoint.json'
            campaign=SimpleNamespace(chunk_count=1,_checkpoint_path=lambda *args:p)
            p.write_text(json.dumps({'generated':{'context':{'role':'draft'},'diagnostics':{'kv_reuse':{'replayed_chunks':20}}}}))
            with self.assertRaisesRegex(ValueError,'missing Target suffix'):verify_burst_reuse(campaign,'key')
            p.write_text(json.dumps({'generated':{'context':{'role':'draft'},'diagnostics':{'kv_reuse':{'replayed_chunks':2}}}}))
            self.assertTrue(verify_burst_reuse(campaign,'key')['suffix_replay_bound_verified'])

    def test_invalid_budget_order_and_fabricated_risk_fail(self):
        for budget in (0,1,20,30):
            with self.assertRaises(ValueError):self.make(budget)
        p=self.make()
        with self.assertRaises(ValueError):p.advance(chunk_index=2,role='draft',risk_score=.5)
        with self.assertRaises(ValueError):p.finalize()
        p.advance(chunk_index=1,role='draft',risk_score=1.)
        with self.assertRaises(ValueError):p.advance(chunk_index=2,role='target',risk_score=.5)


class BurstCampaignTests(unittest.TestCase):
    def test_committed_history_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as td:
            f=CampaignFixture(Path(td));c=f.campaign(chunk_count=30)
            args=dict(prompt=f.evaluation[0],seed=17,budget=14,
                calibrator=SuddenDropRiskCalibrator('independent',(0.,.05,.1,.2)))
            key,route,selected,h=materialize_rank_bursts(c,**args)
            self.assertEqual(len(f.draft.calls),15);self.assertEqual(len(f.target.calls),15)
            for call in f.draft.calls+f.target.calls:
                i=call['chunk_index']
                self.assertEqual(call['history_hash'],h.prefix_before(i).head_sha256)
            self.assertEqual(len(selected),14)
            again=materialize_rank_bursts(f.campaign(chunk_count=30),**args)
            self.assertEqual(again[3].head_sha256,h.head_sha256)
            self.assertEqual(len(f.draft.calls)+len(f.target.calls),30)
            cp=c._checkpoint_path('rollouts',key,1)
            data=json.loads(cp.read_text());data['metadata']['rank']+=1;cp.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'resumed burst'):
                materialize_rank_bursts(f.campaign(chunk_count=30),**args)

    def test_resume_inside_target_burst_does_not_regenerate_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            f=CampaignFixture(Path(td));c=f.campaign(chunk_count=30)
            args=dict(prompt=f.evaluation[0],seed=17,budget=8,
                calibrator=SuddenDropRiskCalibrator('independent',(0.,)))
            generate=f.target.generate
            def interrupt(request):
                if request.history.next_chunk_index==3:raise RuntimeError('simulated stop')
                return generate(request)
            with patch.object(f.target,'generate',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated stop'):materialize_rank_bursts(c,**args)
            self.assertEqual([x['chunk_index'] for x in f.target.calls],[0,2])
            result=materialize_rank_bursts(f.campaign(chunk_count=30),**args)
            self.assertEqual(result[3].next_chunk_index,30)
            self.assertEqual(len(f.draft.calls),21);self.assertEqual(len(f.target.calls),9)
            for backend in (f.draft,f.target):
                ids=[x['chunk_index'] for x in backend.calls]
                self.assertEqual(len(ids),len(set(ids)))


