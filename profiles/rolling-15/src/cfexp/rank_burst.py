"""Causal routing of future Target bursts from committed Draft observations.

There is no speculative candidate, current-block rejection, or trained predictor.
The exact budget counts Target blocks after the independently generated root.
The empirical threshold is an explicitly frozen calibration control, not a
calibration claim for the changed interleaved observation distribution.
"""
from dataclasses import asdict
import math
import time
from .correction_policy import OnlineRankGuidedPolicy
from .longlive_motion import latent_motion_energy, low_motion_pair


NATIVE_BURST_WORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 8: 'eight'}
SPLIT_BURST_WORD = {2: 'two', 3: 'three', 4: 'four', 8: 'eight'}


def valid_split_burst_count(count, *, burst_length, seconds, split):
    if type(count) is not int:
        return False
    if not split:
        return count == 2
    return ((burst_length, count) == (2, 2)
            or (seconds == 30 and (burst_length, count) in ((2, 3), (3, 2), (4, 2)))
            or (seconds == 60 and (burst_length, count) in ((2, 4), (1, 8))))


def _observation_windows(*, draft_count, offset, horizon, budget, burst_length, split,
                         final_window_at_tail=False, tail_draft_reserve=0, opening_target_burst=False):
    if type(split) is not bool:
        raise ValueError('split Target burst flag must be boolean')
    if type(tail_draft_reserve) is not int or tail_draft_reserve not in (0, 4) or (
            tail_draft_reserve and not final_window_at_tail):
        raise ValueError('invalid Draft reserve for the final observation window')
    if type(final_window_at_tail) is not bool or (final_window_at_tail and not (
            split and burst_length == 2 and horizon == 8 and offset == 0
            and (draft_count, budget, tail_draft_reserve) in ((53, 6, 0), (53, 6, 4), (111, 8, 4)))):
        raise ValueError('tail-aligned observation requires the three-window K6 or four-window K8 route')
    if type(opening_target_burst) is not bool or (opening_target_burst and (
            offset != 0 or final_window_at_tail or
            (split and (burst_length, budget, draft_count, horizon)
             not in ((2, 6, 53, 8), (2, 8, 111, 8), (4, 36, 23, 2))) or
            (not split and budget != burst_length))):
        raise ValueError('unsupported opening Target burst geometry')
    if not split:
        return () if opening_target_burst else ((offset, horizon),)
    if (burst_length, budget) not in ((2, 4), (2, 6), (2, 8), (1, 8), (3, 6), (4, 8), (4, 12), (4, 16), (4, 36)) or offset != 0:
        raise ValueError('unsupported split Target burst geometry')
    if burst_length == 4 and not ((draft_count + budget == 59 and horizon == 8 and budget != 36)
                                or (draft_count, budget, horizon) == (23, 36, 2)):
        raise ValueError('four-block split bursts require an explicit 30-second P8 route')
    if burst_length == 3 and (draft_count != 53 or horizon != 8):
        raise ValueError('three-block bursts require the 30-second two-window K6 route')
    count = budget // burst_length
    if horizon > draft_count // count:
        raise ValueError('split Target observation windows must be disjoint')
    windows = tuple((draft_count * i // count, horizon) for i in range(count))
    if final_window_at_tail:
        return (*windows[:-1], (draft_count - horizon - tail_draft_reserve, horizon))
    return windows[1:] if opening_target_burst else windows


def _observation_slot(observed, windows):
    for index, (offset, horizon) in enumerate(windows):
        if offset < observed <= offset + horizon:
            return index, observed - offset
    return None


class RankBurstPolicy:
    def __init__(self, *, chunk_count, target_budget, burst_length, calibration, evaluation_id,
                 observation_horizon=None, observation_offset=0, opening_target_burst=False,
                 split_target_burst=False, final_window_at_tail=False, tail_draft_reserve=0,
                 motion_trigger=False, require_positive_drop=False):
        for value in (chunk_count, target_budget, burst_length):
            if type(value) is not int or value <= 0:
                raise ValueError("burst geometry must contain positive integers")
        if target_budget % burst_length:
            raise ValueError("Target budget must contain whole bursts")
        self.draft_count = chunk_count - 1 - target_budget
        burst_count = target_budget // burst_length
        if not 1 <= burst_count <= self.draft_count:
            raise ValueError("each burst needs its own preceding Draft observation")
        self.chunk_count, self.target_budget, self.burst_length = chunk_count, target_budget, burst_length
        if type(observation_offset) is not int or not 0 <= observation_offset < self.draft_count:
            raise ValueError("invalid Draft observation offset")
        self.observation_offset = observation_offset
        available = self.draft_count - observation_offset
        self.observation_horizon = available if observation_horizon is None else observation_horizon
        if (type(self.observation_horizon) is not int
                or not (1 if split_target_burst else burst_count) <= self.observation_horizon <= available):
            raise ValueError("observation horizon cannot fit the frozen event budget")
        windows = _observation_windows(draft_count=self.draft_count, offset=observation_offset,
            horizon=self.observation_horizon, budget=target_budget, burst_length=burst_length,
            split=split_target_burst, final_window_at_tail=final_window_at_tail, tail_draft_reserve=tail_draft_reserve,
            opening_target_burst=opening_target_burst)
        if type(motion_trigger) is not bool or (motion_trigger and not (
                chunk_count == 60 and target_budget == 6 and burst_length == 2
                and self.observation_horizon == 8 and split_target_burst
                and not final_window_at_tail and not opening_target_burst)):
            raise ValueError('motion trigger requires the original 30-second K6 observation windows')
        self.motion_trigger = motion_trigger
        if type(require_positive_drop) is not bool or (require_positive_drop and not (
                split_target_burst and not motion_trigger and burst_length == 4
                and self.observation_horizon == 2 and (chunk_count, target_budget) == (60, 36))):
            raise ValueError('positive eRank drop gate requires the frozen SF30 K36 route')
        self.require_positive_drop = require_positive_drop
        self.previous_motion_energy = None
        self.windows = windows
        remaining_bursts = burst_count - int(opening_target_burst)
        self.policies = tuple(OnlineRankGuidedPolicy(
            budget=remaining_bursts // len(self.windows), calibration=calibration,
            evaluation_id=evaluation_id, chunk_count=horizon + 1) for _, horizon in self.windows)
        self.cursor, self.observed = 1, 0
        self.pending = burst_length if opening_target_burst else 0
        self.selected = list(range(1, self.pending + 1))

    @property
    def next_role(self):
        if self.cursor >= self.chunk_count:
            raise ValueError("burst trajectory is complete")
        return "target" if self.pending else "draft"

    @property
    def observes_motion(self):
        return (self.motion_trigger and self.next_role == 'draft'
                and _observation_slot(self.observed + 1, self.windows) is not None)

    def advance(self, *, chunk_index, role, risk_score=None, motion_energy=None,
                erank_drop_score=None):
        if type(chunk_index) is not int or chunk_index != self.cursor or role != self.next_role:
            raise ValueError("burst chunk index or generation role drifted")
        if self.observes_motion:
            if type(motion_energy) not in (int, float) or not math.isfinite(motion_energy) or motion_energy < 0:
                raise ValueError('motion observation requires a finite nonnegative energy')
        elif motion_energy is not None:
            raise ValueError('motion energy outside the active Draft observation window')
        decision = None
        selected = []
        if role == "target":
            if risk_score is not None:
                raise ValueError("Target blocks cannot fabricate a Draft risk")
            self.pending -= 1
        else:
            self.observed += 1
            slot = _observation_slot(self.observed, self.windows)
            if slot is not None:
                window, local_observation = slot
                motion = (local_observation > 1
                          and low_motion_pair(self.previous_motion_energy, motion_energy))
                self.previous_motion_energy = motion_energy
                if self.require_positive_drop and (
                        type(erank_drop_score) not in (int, float)
                        or not math.isfinite(erank_drop_score) or not 0 <= erank_drop_score <= 1):
                    raise ValueError('positive-drop route requires a valid Draft drop score')
                decision = asdict(self.policies[window].decide(
                    chunk_index=local_observation, risk_score=risk_score, additional_trigger=motion,
                    allow_threshold_trigger=not self.require_positive_drop or erank_drop_score > 0))
            if decision is not None and decision["replace_with_target"]:
                selected = list(range(chunk_index + 1, chunk_index + 1 + self.burst_length))
                if selected[-1] >= self.chunk_count:
                    raise ValueError("burst would extend beyond the frozen video horizon")
                self.pending = self.burst_length
                self.selected.extend(selected)
        self.cursor += 1
        return {"schema": "committed_draft_rank_burst_decision_v1",
                "chunk_index": chunk_index, "committed_role": role,
                "draft_observations_so_far": self.observed,
                "observation_decision": decision,
                "scheduled_target_chunks": selected,
                "pending_target_chunks": self.pending,
                **({'motion_energy': motion_energy} if motion_energy is not None else {})}

    def finalize(self):
        if (self.cursor != self.chunk_count or self.pending or self.observed != self.draft_count
                or len(self.selected) != self.target_budget):
            raise ValueError("burst trajectory is incomplete")
        for policy in self.policies:
            policy.finalize()
        return tuple(self.selected)


def materialize_rank_bursts(campaign, *, prompt, seed, budget, calibrator, burst_length=2,
                            observation_horizon=None, on_commit=None, before_generate=None,
                            on_target_prefix=None, observation_offset=0, opening_target_burst=False,
                            split_target_burst=False, final_window_at_tail=False, tail_draft_reserve=0,
                            motion_trigger=False, require_positive_drop=False):
    """Generate each committed chunk exactly once, with resumable causal decisions."""
    route = f"committed-rank-burst{burst_length}-k{budget}"
    if observation_horizon is not None:
        route += f"-probe{observation_horizon}"
    if observation_offset:
        route += f"-offset{observation_offset}"
    if opening_target_burst:
        route += "-opening"
    if split_target_burst:
        route += "-split"
    if final_window_at_tail:
        route += "-tail"
    if tail_draft_reserve:
        route += f"-reserve{tail_draft_reserve}"
    if motion_trigger:
        route += "-motion"
    if require_positive_drop:
        route += "-positive-drop"
    policy = RankBurstPolicy(chunk_count=campaign.chunk_count, target_budget=budget,
        burst_length=burst_length, calibration=calibrator,
        evaluation_id=f"eval-burst-{prompt.prompt_id}-{seed}-{budget}-{burst_length}",
        observation_horizon=observation_horizon, observation_offset=observation_offset,
        opening_target_burst=opening_target_burst, split_target_burst=split_target_burst,
        final_window_at_tail=final_window_at_tail, tail_draft_reserve=tail_draft_reserve,
        motion_trigger=motion_trigger, require_positive_drop=require_positive_drop)
    key = campaign._rollout_key(prompt, seed, route)
    history, restored_meta, restored_generated = campaign._load_rollout_prefix(
        key=key, trajectory_id=campaign._trajectory_id("rollout", prompt.prompt_id, seed),
        prompt=prompt, seed=seed)
    if on_target_prefix is not None and (history.next_chunk_index > 1 or opening_target_burst
            or (final_window_at_tail and not tail_draft_reserve)
            or (burst_length, split_target_burst, budget) not in
            ((4, False, 4), (2, True, 4), (2, True, 6), (2, True, 8))):
        raise ValueError('accepted-prefix scheduling requires fresh four-Target or split two-Target bursts')
    canonical, root = campaign._canonical_rollout_root(prompt=prompt, seed=seed)
    root_meta = {"kind": "committed_rank_burst", "route_id": route, "chunk_index": 0,
                 "corrected": True, "committed_role": "target", "rank": None,
                 "committed_kv_hash": root.context.kv.cache_sha256,
                 "risk_score": None, "erank_drop_score": None, "decision": None,
                 "initial_noise_descriptor_sha256": root.context.noise.descriptor_sha256}
    if history.next_chunk_index == 0:
        history = canonical
        campaign._write_checkpoint(path=campaign._checkpoint_path("rollouts", key, 0),
            history=history, generated=root, metadata=root_meta)
        restored_meta, restored_generated = [root_meta], [root]
    else:
        campaign._verify_canonical_rollout_root(history=history, metadata=restored_meta,
            generated_chunks=restored_generated, canonical_history=canonical, canonical_generated=root)
        if restored_meta[0] != root_meta:
            raise ValueError("resumed burst root metadata changed")
    ranks = []
    if on_commit is not None:
        for index in range(1, history.next_chunk_index+1):
            on_commit(history.prefix_before(index))

    def metadata_for(index, generated):
        role = generated.context.role
        rank, risk, drop = None, None, None
        if role == "draft":
            if generated.effective_rank is None:
                raise ValueError("committed Draft rank is missing")
            rank = float(generated.effective_rank)
            ranks.append(rank)
            risk, drop = campaign._risk_from_rank_history(calibrator, ranks)
        energy = (latent_motion_energy(campaign.run_dir, generated.segment)
                  if policy.observes_motion else None)
        decision = policy.advance(chunk_index=index, role=role, risk_score=risk,
                                  motion_energy=energy, erank_drop_score=drop)
        return {"kind": "committed_rank_burst", "route_id": route, "chunk_index": index,
                "corrected": role == "target", "committed_role": role, "rank": rank,
                "risk_score": risk, "erank_drop_score": drop, "decision": decision,
                "initial_noise_descriptor_sha256": generated.context.noise.descriptor_sha256}

    for index in range(1, history.next_chunk_index):
        expected = metadata_for(index, restored_generated[index])
        if expected != restored_meta[index]:
            raise ValueError("resumed burst rank, budget or decision changed")
    for index in range(history.next_chunk_index, campaign.chunk_count):
        role = policy.next_role
        if before_generate is not None:
            before_generate(role)
        noise = campaign._materialize_noise(prompt=prompt, seed=seed, chunk_index=index)
        measure_target = on_target_prefix is not None and split_target_burst and role == 'target'
        target_started = time.monotonic() if measure_target else None
        generated = campaign._generate(backend=campaign.draft if role == "draft" else campaign.target,
            prompt=prompt, history=history, noise=noise,
            output_stem=f"latents/rollouts/{key}/chunk-{index:02d}-{role}", capture_hidden=False)
        target_completed = time.monotonic() if measure_target else None
        metadata = metadata_for(index, generated)
        if measure_target:
            metadata['target_rpc_interval'] = [target_started, target_completed]
        history = history.append(generated.segment)
        campaign._write_checkpoint(path=campaign._checkpoint_path("rollouts", key, index),
            history=history, generated=generated, metadata=metadata)
        if on_commit is not None:
            on_commit(history)
        if on_target_prefix is not None and role=='target' and policy.pending:
            on_target_prefix(prompt=prompt,history=history,
                             next_draft_chunk_index=index+policy.pending+1)
    selected = policy.finalize()
    actual = tuple(s.chunk_index for s in history.segments[1:] if s.producer_role == "target")
    if actual != selected:
        raise ValueError("scheduled bursts differ from committed history")
    return key, route, selected, history


def verify_burst_reuse(campaign, key, *, burst_length=2, budget=4, opening_target_burst=False):
    """Bound cache reconstruction to the initial prefix and missing suffixes."""
    import json
    import os
    from .cpu_history_binding import ENV, validate_evidence
    cpu_binding = os.environ.get(ENV, '0') == '1'
    from .rope_worker import ENV as ROPE_ENV, validate_execution
    rope = os.environ.get(ROPE_ENV,'0')=='1'
    from .full_attention_worker import ENV as ATTENTION_ENV, validate_execution as validate_attention
    attention=os.environ.get(ATTENTION_ENV,'0')=='1'
    from .accepted_prefill import enabled as prefill_enabled
    prefill=prefill_enabled();prefill_paths=set()
    evidence = []
    for i in range(campaign.chunk_count):
        x = json.loads(campaign._checkpoint_path('rollouts', key, i).read_text())['generated']
        if x['context']['role'] == 'draft':
            e = x.get('diagnostics', {}).get('kv_reuse')
            if not isinstance(e, dict) or type(e.get('replayed_chunks')) is not int:
                raise ValueError('burst Draft omitted replay evidence')
            if opening_target_burst and not evidence:
                if i != burst_length + 1 or e['replayed_chunks'] != i:
                    raise ValueError('first Draft must assimilate the complete Target opening')
            elif not 0 <= e['replayed_chunks'] <= burst_length:
                raise ValueError('burst Draft rebuilt more than the missing Target suffix')
            if cpu_binding:
                validate_evidence(x.get('diagnostics', {}).get('history_binding_execution'),
                    history_chunks=i, tensor_digests=e['history_tensor_sha256'])
            if rope:
                validate_execution(x.get('diagnostics',{}).get('rope_execution_evidence'))
            if attention:
                validate_attention(x.get('diagnostics',{}).get('full_attention_execution_evidence'))
            if prefill:
                from .accepted_prefill_audit import canonical_trace,local_receipt_reader
                from .correction_state import LatentHistory
                checkpoint=json.loads(campaign._checkpoint_path('rollouts',key,i).read_text())
                h=LatentHistory.from_dict(checkpoint['history']).prefix_before(i)
                prompt=next(p.prompt for p in campaign.evaluation_prompts if p.prompt_id==h.prompt_id)
                _,paths=canonical_trace(x['diagnostics'],history=h,prompt=prompt, burst_length=burst_length,
                    read_receipt=local_receipt_reader(campaign.run_dir))
                if prefill_paths.intersection(paths):raise ValueError('accepted prefill receipt reused')
                prefill_paths.update(paths)
            evidence.append(e)
    if not evidence:
        raise ValueError('burst has no measured Draft continuation')
    if prefill and len(prefill_paths) != (budget // burst_length) * (burst_length - 1):
        raise ValueError('accepted prefill must account for every prefetched Target suffix block')
    return {**({'initial_replayed_chunks': evidence[0]['replayed_chunks']} if opening_target_burst else {}),
            **({'cpu_history_binding_verified':True} if cpu_binding else {}),
            **({'native_rope_coefficients_verified':True} if rope else {}),
            **({'native_full_length_attention_verified':True} if attention else {}),
            **({'accepted_target_prefill_verified':True,'separate_clean_prefix_calls':len(prefill_paths)} if prefill else {}),
            'schema': 'committed_rank_burst_reuse_v1', 'maximum_replayed_chunks':
            max(e['replayed_chunks'] for e in evidence), 'draft_calls': len(evidence),
            'burst_length': burst_length, 'suffix_replay_bound_verified': True}


def validate_burst_schedule(trajectory, *, budget, burst_length=2, chunk_count=30,
                            observation_horizon=None, observation_offset=0, opening_target_burst=False,
                            split_target_burst=False, final_window_at_tail=False, tail_draft_reserve=0,
                            motion_trigger=False, require_positive_drop=False):
    """Validate that decisions schedule future blocks, never reject current Draft."""
    decisions = trajectory['decisions']
    if len(decisions) != chunk_count-1:
        raise ValueError('burst decision coverage is incomplete')
    available = chunk_count - 1 - budget
    if type(observation_offset) is not int or not 0 <= observation_offset < available:
        raise ValueError("invalid Draft observation offset")
    available -= observation_offset
    horizon = available if observation_horizon is None else observation_horizon
    minimum_horizon = 1 if split_target_burst else budget // burst_length
    if type(horizon) is not int or not minimum_horizon <= horizon <= available:
        raise ValueError('invalid burst observation horizon')
    windows = _observation_windows(draft_count=chunk_count - 1 - budget, offset=observation_offset,
        horizon=horizon, budget=budget, burst_length=burst_length, split=split_target_burst,
        final_window_at_tail=final_window_at_tail, tail_draft_reserve=tail_draft_reserve,
        opening_target_burst=opening_target_burst)
    pending = burst_length if opening_target_burst else 0
    observed, selected = 0, list(range(1, pending + 1))
    triggered = [0] * len(windows)
    previous_motion = None
    for i, row in enumerate(decisions, 1):
        d = row['decision']
        role = 'target' if pending else 'draft'
        if (row['chunk_index'] != i or d['chunk_index'] != i
                or d['schema'] != 'committed_draft_rank_burst_decision_v1'
                or d['committed_role'] != role):
            raise ValueError('burst decision order or committed role differs')
        scheduled = []
        energy = d.get('motion_energy')
        if role == 'target':
            pending -= 1
            if (d['observation_decision'] is not None or row['rank'] is not None
                    or row['risk_score'] is not None or energy is not None):
                raise ValueError('Target fabricated a Draft observation')
        else:
            observed += 1
            o = d['observation_decision']
            slot = _observation_slot(observed, windows)
            if slot is None:
                if o is not None:
                    raise ValueError('routing occurred outside the frozen observation window')
            elif o is None or o['chunk_index'] != slot[1] or type(o['replace_with_target']) is not bool:
                raise ValueError('burst observation order differs')
            if motion_trigger and slot is not None:
                if type(energy) not in (int, float) or not math.isfinite(energy) or energy < 0:
                    raise ValueError('burst motion evidence is missing or invalid')
                motion = slot[1] > 1 and low_motion_pair(previous_motion, energy)
                previous_motion = energy
                if o['reason'] == 'additional_trigger' and not (
                        motion and o['replace_with_target'] and o['threshold'] is not None
                        and row['risk_score'] < o['threshold']):
                    raise ValueError('motion trigger lacks two low-motion Draft observations')
                if motion and o['remaining_budget_before'] > 0 and not o['replace_with_target']:
                    raise ValueError('motion trigger did not schedule its reserved burst')
            elif energy is not None or (o is not None and o.get('reason') == 'additional_trigger'):
                raise ValueError('unexpected motion evidence in a non-motion observation')
            if o is not None and require_positive_drop:
                drop = row.get('erank_drop_score')
                if type(drop) not in (int, float) or not math.isfinite(drop) or not 0 <= drop <= 1:
                    raise ValueError('missing eRank drop evidence')
                if (drop == 0 and o['threshold'] is not None
                        and o['risk_score'] >= o['threshold']
                        and (o['replace_with_target'] or o['reason'] != 'threshold_trigger_gated')):
                    raise ValueError('threshold trigger accepted a zero eRank drop')
                if o['reason'] == 'threshold_trigger_gated' and (
                        drop > 0 or o['threshold'] is None or o['risk_score'] < o['threshold']):
                    raise ValueError('eRank drop gate evidence differs')
            if o is not None and o['replace_with_target']:
                triggered[slot[0]] += 1
                scheduled = list(range(i+1,i+1+burst_length))
                if scheduled[-1] >= chunk_count:
                    raise ValueError('burst exceeds horizon')
                selected.extend(scheduled);pending = burst_length
        if (d['scheduled_target_chunks'] != scheduled or d['pending_target_chunks'] != pending
                or d['draft_observations_so_far'] != observed):
            raise ValueError('burst state or future schedule differs')
    if split_target_burst and triggered != [1] * len(windows):
        raise ValueError('split Target route must schedule exactly one burst per window')
    if (pending or observed != chunk_count-1-budget or len(selected) != budget
            or trajectory['accepted_chunks'] != selected or trajectory['proposed_chunks'] != selected):
        raise ValueError('burst budget or committed positions differ')


def audit_native_model_burst(campaign, *, key, history, selected, draft_model_id,
                            burst_length=4, observation_horizon=8, observation_offset=0,
                            opening_target_burst=False, split_target_burst=False,
                            split_burst_count=2, longlive_prefill=False,
                            fixed_target_positions=False, final_window_at_tail=False, motion_trigger=False,
                            rolling_prefix_transactions=False, rolling_committed_prefix=False):
    """Audit each committed generation against the selected backend's own cache."""
    import json

    from .krea_chunk_backend import validate_native_correction_state

    if (type(rolling_prefix_transactions) is not bool or type(rolling_committed_prefix) is not bool
            or (rolling_prefix_transactions and rolling_committed_prefix)):
        raise ValueError('Rolling committed prefix and rollback flags must be distinct booleans')
    if (rolling_prefix_transactions or rolling_committed_prefix) and (
            draft_model_id!='rolling_forcing_1.3b' or fixed_target_positions
            or burst_length!=4 or opening_target_burst or split_target_burst):
        raise ValueError('Rolling prefix audit requires its native eRank B4 route')

    if type(fixed_target_positions) is not bool or (fixed_target_positions and
            (longlive_prefill or split_target_burst or opening_target_burst or observation_offset
             or final_window_at_tail or motion_trigger)):
        raise ValueError('fixed native route cannot inherit burst or prefill policy')
    if (list(selected)!=sorted(set(selected)) or any(type(i) is not int or
            not 1<=i<campaign.chunk_count for i in selected)):
        raise ValueError('native route positions must be unique non-root chunks')

    target_positions = {0, *selected}
    target_sessions = set()
    decisions, draft_states = [], []
    previous_draft = None
    last_target_interval = None
    for index, segment in enumerate(history.segments):
        record = json.loads(campaign._checkpoint_path('rollouts', key, index).read_text())
        generated = record['generated']
        role = 'target' if index in target_positions else 'draft'
        if (generated['context']['role'] != role or segment.producer_role != role
                or generated['segment']['segment_sha256'] != segment.segment_sha256):
            raise ValueError('native burst checkpoint differs from committed producer or segment')
        metadata = record['metadata']
        if fixed_target_positions and (metadata.get('corrected') is not (index in target_positions)
                or (role == 'target' and metadata.get('draft_proposal') is not None)):
            raise ValueError('fixed native route computed an unused Draft proposal or changed positions')
        if index and not fixed_target_positions:
            decisions.append({name: metadata[name] for name in
                              ('chunk_index', 'decision', 'rank', 'risk_score')})
            energy = metadata['decision'].get('motion_energy')
            if motion_trigger and energy is not None and energy != latent_motion_energy(campaign.run_dir, segment):
                raise ValueError('motion energy differs from the accepted Draft latent')
        diagnostics = generated['diagnostics']
        if role == 'target':
            last_target_interval = metadata.get('target_rpc_interval')
            state = diagnostics['native_correction_state']
            validate_native_correction_state(state, chunk_index=index)
            if index and state['session_recreated']:
                raise ValueError('Target correction session was recreated within the rollout')
            target_sessions.add(state['native_session_id'])
            continue
        missing = index if previous_draft is None else index - previous_draft - 1
        if draft_model_id == 'rolling_forcing_1.3b':
            state = diagnostics['rolling_state']
            reused = previous_draft is not None and missing == 0
            if rolling_committed_prefix:
                from .rolling_committed_execution import validate_state
                validate_state(state,index=index,previous=previous_draft)
            elif rolling_prefix_transactions:
                from .rolling_prefix_execution import validate_state
                validate_state(state,index=index,previous=previous_draft)
            elif (state['chunk_index'] != index
                    or state['state_policy'] != 'single_active_branch_clean_history_restart_v1'
                    or state['reused_rolling_state'] is not reused
                    or state['replayed_history_blocks'] != (0 if reused else index)
                    or state['discarded_future_state'] is not (previous_draft is not None and not reused)):
                raise ValueError('Rolling continuation did not respect its native accepted-history window')
            draft_states.append({'chunk_index': index, **state})
        elif draft_model_id == 'longlive_1.3b':
            state = diagnostics['kv_reuse']
            prefills = []
            if longlive_prefill:
                from .longlive_accepted_prefill import audit_handoff
                prefills = audit_handoff(campaign.run_dir, history.prefix_before(index),
                    diagnostics, expected=int(missing == 2), target_interval=last_target_interval)
            replayed = missing - len(prefills)
            if (type(state['replayed_chunks']) is not int or state['replayed_chunks'] != replayed
                    or type(state['reused_chunks']) is not int or state['reused_chunks'] != index - replayed):
                raise ValueError('LongLive replayed history outside the missing Target suffix')
            draft_states.append({'chunk_index': index, 'replayed_chunks': state['replayed_chunks'],
                                 'reused_chunks': state['reused_chunks'],
                                 **({'accepted_prefills': prefills} if prefills else {})})
        else:
            raise ValueError('unsupported native burst Draft backend')
        previous_draft = index

    budget = len(selected) if fixed_target_positions else (split_burst_count * burst_length if split_target_burst else burst_length)
    if len(target_sessions) != 1 or len(draft_states) != campaign.chunk_count - 1 - budget:
        raise ValueError('native burst cache evidence is incomplete')
    if longlive_prefill and sum(len(s.get('accepted_prefills', ())) for s in draft_states) != 2:
        raise ValueError('LongLive split route must prefill one block per burst')
    if not fixed_target_positions:
        validate_burst_schedule({'decisions': decisions, 'accepted_chunks': list(selected),
                             'proposed_chunks': list(selected)}, budget=budget, burst_length=burst_length,
        chunk_count=campaign.chunk_count, observation_horizon=observation_horizon,
        observation_offset=observation_offset, opening_target_burst=opening_target_burst,
        split_target_burst=split_target_burst, final_window_at_tail=final_window_at_tail,
        motion_trigger=motion_trigger)
    return {'schema': ('native_fixed_route_cache_audit_v1' if fixed_target_positions else
                      f'native_model_e4_p{observation_horizon}_k{budget}_cache_audit_v1'), 'draft_model_id': draft_model_id,
            'target_chunks': sorted(target_positions), 'target_session_id': next(iter(target_sessions)),
            'draft_states': draft_states, 'decisions': decisions,
            **({'observation_offset': observation_offset} if observation_offset else {}),
            **({'opening_target_burst': True} if opening_target_burst else {}),
            **({'split_target_burst': True} if split_target_burst else {}),
            **({'split_burst_count': split_burst_count} if split_burst_count != 2 else {}),
            **({'final_window_at_tail': True} if final_window_at_tail else {}),
            **({'motion_trigger': True} if motion_trigger else {}),
            **({'rolling_prefix_transactions':True} if rolling_prefix_transactions else {}),
            **({'rolling_committed_prefix':True} if rolling_committed_prefix else {}),
            **({'longlive_prefill': True} if longlive_prefill else {})}
