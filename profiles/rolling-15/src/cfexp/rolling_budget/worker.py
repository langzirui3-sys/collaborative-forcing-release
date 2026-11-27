"""Ordinary streaming worker with an explicitly admitted Rolling Draft only."""
from __future__ import annotations

import os
from functools import partial

from .. import rolling_forcing_policy_backend as policy
from ..rolling_policy_worker import build_backend as build_native
from ..rolling_joint_handoff import ENV as JOINT_ENV
from ..rolling_joint_stream import JointCommittedSession
from .admission import PINNED_GATES, SOURCE, require

WIDTH_ENV = 'CFEXP_ROLLING_BUDGET_WIDTH'
STATE = 'rolling_verified_joint_budget_trial_v1'


def width():
    value = os.environ.get(WIDTH_ENV)
    require(value in ('2', '4'), 'Rolling trial worker requires a frozen budget')
    return int(value)


def binding(blocks):
    return dict(source=SOURCE, artifact_index_sha256=PINNED_GATES[blocks][1], width=blocks)


class BudgetBackend:
    """Keep the stock worker protocol, noise owner, and serialized witnesses."""
    def __init__(self, native, blocks):
        self._native = native
        self._blocks = blocks

    def __getattr__(self, name):
        return getattr(self._native, name)

    def generate_chunk(self, **kwargs):
        # The isolated worker handles one request at a time. Scope this
        # injection to the native generator call; no stock factory is changed.
        original = policy.RollingForcingSession
        policy.RollingForcingSession = partial(JointCommittedSession,
            target_suffix_blocks=self._blocks)
        try:
            with self._native.torch.no_grad():
                value = self._native.generate_chunk(**kwargs)
        finally:
            policy.RollingForcingSession = original
        state = value['rolling_state']
        require(state['state_policy'] == 'diagnostic_joint_committed_rolling_window_v1',
                'native worker did not use the admitted Rolling joint session')
        session = self._native.session
        require(isinstance(session, JointCommittedSession)
                and session.target_suffix_blocks == self._blocks,
                'wrong owner for joint Rolling request')
        resumed = state['discarded_future_state']
        require(not resumed or (session.stream.joint_calls == 1 and session.stream.joint_suffix is None),
                'Target suffix was not consumed exactly once')
        state.update(state_policy=STATE, proof=binding(self._blocks),
            fused_clean_blocks=self._blocks if resumed else 0,
            joint_first_window_calls=int(resumed), separate_suffix_clean_calls=0)
        return value


def build_backend(raw_config):
    return build_joint_backend(raw_config, blocks=width(), chunk_count=30)


def build_joint_backend(raw_config, *, blocks, chunk_count):
    require((blocks, chunk_count) in ((2, 30), (4, 30), (4, 60)),
            'unsupported native Rolling joint width or horizon')
    require(os.environ.get(JOINT_ENV) == '1'
            and all(os.environ.get(k,'0') == '0' for k in (
                'CFEXP_ROLLING_HOST_KV_CURSORS','CFEXP_ROLLING_HOST_WINDOW',
                'CFEXP_ROLLING_CPU_HISTORY_IDENTITY','CFEXP_ROLLING_COMMITTED_PREFIX',
                'CFEXP_ROLLING_PREFIX_TRANSACTIONS')),
            'Rolling budget trial must preserve the validated tensor-metadata path')
    native = build_native(raw_config)
    try:
        require(native.chunk_count == chunk_count and not native.cpu_history and
                not native.prefix_transactions and not native.committed_prefix,
                'Rolling joint worker requires the declared native video horizon')
        native.model_state['joint_budget_proof'] = binding(blocks)
        return BudgetBackend(native, blocks)
    except BaseException:
        native.close()
        raise


def validate_state(state, *, index, previous, blocks):
    initial = previous is None
    missing = index if initial else index-previous-1
    resumed = not initial and missing != 0
    reused = not initial and not resumed
    require((missing == 1 if initial else missing in (0, blocks))
            and state == dict(chunk_index=index, reused_rolling_state=reused,
              replayed_history_blocks=missing, discarded_future_state=resumed,
              state_policy=STATE, proof=binding(blocks),
              fused_clean_blocks=blocks if resumed else 0,
              joint_first_window_calls=int(resumed), separate_suffix_clean_calls=0),
            'Rolling joint budget reuse, suffix, or proof differs')
