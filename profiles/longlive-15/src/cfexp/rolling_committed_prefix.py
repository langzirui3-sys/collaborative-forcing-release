"""Native Rolling restart after a committed Draft block and four Target blocks.

RankPulse keeps its observed Draft block. Its cache is already clean and belongs
to accepted history; undoing that block is the wrong handoff. This separate,
diagnostic-only session retains that post-commit prefix, discards speculative
future windows and appends exactly the missing Target B3 history. The original
reject-current transaction session and all production defaults are unchanged.
"""
from __future__ import annotations

import hashlib

from .incremental_draft import latent_digest
from .rolling_forcing_observation import RollingStepZeroObserver
from .rolling_forcing_session import RollingForcingSession
from .rolling_forcing_stream import RollingForcingError, RollingForcingStream
from .rolling_prefix_transaction import AcceptedPrefix
from .stage_timing import stage

SCHEMA = 'rolling_committed_clean_prefix_v1'
POLICY = 'single_active_branch_committed_clean_prefix_v1'


def require(value, message):
    if not value:
        raise RollingForcingError(message)


class CommittedPrefixSession(RollingForcingSession):
    """Keep the last returned Draft block; never accept a current-block rejection.

    No rollback snapshot is allocated. A request that changes an accepted
    prefix fails before any cache or observer is mutated. This is not a generic
    replacement backend and does not enable any production factory.
    """

    def __init__(self, **kwargs):
        require(kwargs.get('prefix_transactions', False) is False
                and kwargs.get('cpu_history', False) is False,
                'committed prefix requires GPU history and no rollback transactions')
        super().__init__(**kwargs)

    def _restart_after_target(self, history, index, digest):
        t, p = self.torch, self.pipeline
        retained = self.next_index
        condition = self.stream.condition
        prefix = history[:, :retained*3]
        # Validate the live owner, populated cursors and accepted prefix before
        # destroying the old future. No K/V from Target crosses this boundary.
        lease = AcceptedPrefix(pipeline=p, torch=t, history=prefix,
                               prompt=self.prompt, condition=condition)
        lease.consume(pipeline=p, torch=t, history=prefix, prompt=self.prompt)
        try:
            if self.observer is not None:
                self.observer.close()
                self.observer = None
            condition = self.stream.abandon_future()
            self.stream = None
            devices = ([self.noise.device.index if self.noise.device.index is not None
                        else t.cuda.current_device()]
                       if self.noise.device.type == 'cuda' else [])
            restart_seed = int.from_bytes(hashlib.sha256(
                f'rolling-restart-v1:{self.seed}:{index}'.encode()).digest()[:8], 'big') % (2**63)
            with t.random.fork_rng(devices=devices), t.no_grad():
                t.manual_seed(restart_seed)
                if devices:
                    t.cuda.manual_seed_all(restart_seed)
                with stage('rolling_committed_suffix', detail={
                        'retained_blocks': retained, 'target_blocks': index-retained}, cuda=True):
                    for block in range(retained, index):
                        p.generator(noisy_image_or_video=history[:, block*3:(block+1)*3],
                            conditional_dict=condition,
                            timestep=t.zeros((1,3), device=self.noise.device, dtype=t.float32),
                            kv_cache=p.kv_cache_clean, crossattn_cache=p.crossattn_cache,
                            current_start=block*3*p.frame_seq_length, updating_cache=True)
                if self.observe_step_zero:
                    self.observer = RollingStepZeroObserver(pipeline=p, torch=t,
                        retain_hidden=self.retain_hidden, capture_cache=self.capture_cache)
                accepted = AcceptedPrefix(pipeline=p, torch=t, history=history,
                    prompt=self.prompt, condition=condition)
                self.stream = RollingForcingStream(pipeline=p, torch=t,
                    noise=self.noise[:, index*3:], prompt=self.prompt, history=history,
                    accepted_prefix=accepted)
            # The base session now continues this freshly restarted window.
            # The returned receipt below separately discloses the handoff.
            self.next_index, self.expected_history = index, digest
        except BaseException:
            self.close()
            raise

    def generate(self, *, history=None):
        require(not self.closed, 'committed prefix session is closed')
        t = self.torch
        if history is not None:
            require(history.ndim == 5 and history.shape[0] == 1
                    and history.shape[1] >= 3 and history.shape[1] % 3 == 0
                    and history.shape[2:] == self.noise.shape[2:]
                    and history.dtype == self.noise.dtype and history.device == self.noise.device,
                    'invalid committed accepted history')
        index = 0 if history is None else history.shape[1] // 3
        require(index < self.blocks, 'committed history exhausts the noise schedule')
        initial = self.stream is None
        retained = self.next_index
        digest = None if history is None else latent_digest(history, t)
        handoff = not initial and index != retained
        if initial:
            require(index in (0,1), 'committed session must start at its native or Target root')
        elif handoff:
            require(index == retained+4 and history is not None
                    and latent_digest(history[:, :retained*3], t) == self.expected_history,
                    'committed handoff requires the unchanged accepted Draft prefix and four Target blocks')
            self._restart_after_target(history, index, digest)
        else:
            require(digest == self.expected_history,
                    'committed Draft history cannot be replaced or rejected')
        result = super().generate(history=history)
        if handoff:
            result.update(reused_rolling_state=False, replayed_history_blocks=4,
                          discarded_future_state=True)
        result['state_policy'] = POLICY
        result['committed_prefix'] = dict(schema=SCHEMA,
            action=('initial_clean_history' if initial else
                    'accepted_prefix_and_target_suffix' if handoff else 'continued_native_window'),
            retained_prefix_blocks=0 if initial else retained,
            clean_suffix_blocks=4 if handoff else 0,
            full_history_replay_blocks=index if initial else 0,
            rollback_performed=False, checkpoint_undo_bytes=0,
            last_returned_draft_retained=not initial)
        return result
