"""Single-active-branch rolling session with a bound full noise schedule.

Experimental until native CUDA/history validation passes. Common campaign
workers do not route here yet. An accepted Target latent causes clean-history
replay and suffix restart, never reuse of an abandoned speculative future.
"""
from __future__ import annotations

import hashlib
import os

from .incremental_draft import latent_digest
from .stage_timing import stage
from .rolling_forcing_observation import RollingStepZeroObserver
from .rolling_forcing_stream import RollingForcingError, RollingForcingStream


CPU_HISTORY_ENV = 'CFEXP_ROLLING_CPU_HISTORY_IDENTITY'


def cpu_history_enabled():
    flag = os.environ.get(CPU_HISTORY_ENV, '0')
    if flag not in ('0', '1'):
        raise RollingForcingError(f'{CPU_HISTORY_ENV} must be 0 or 1')
    return flag == '1'


def _cpu_history_digest(history, appended, torch):
    """Hash the native BF16 history identity without concatenating its bytes."""
    parts = [value for value in (history, appended) if value is not None]
    frames = sum(value.shape[1] for value in parts)
    shape = (1, frames, *parts[0].shape[2:])
    digest = hashlib.sha256(str(shape).encode('ascii'))
    for value in parts:
        payload = value.detach().to(dtype=torch.bfloat16).contiguous()
        digest.update(memoryview(payload.view(torch.uint8).numpy()))
    return digest.hexdigest()


class RollingForcingSession:
    def __init__(self, *, pipeline, torch, noise, prompt, seed,
                 observe_step_zero=False, retain_hidden=False, capture_cache=False,
                 cpu_history=False):
        if type(observe_step_zero) is not bool or type(retain_hidden) is not bool:
            raise RollingForcingError('observation options must be booleans')
        if type(capture_cache) is not bool:
            raise RollingForcingError("cache observation option must be boolean")
        if type(cpu_history) is not bool:
            raise RollingForcingError('CPU history option must be a boolean')
        if capture_cache and not observe_step_zero:
            raise RollingForcingError("cache observation requires step-zero observation")
        if retain_hidden and not observe_step_zero:
            raise RollingForcingError('retained hidden requires step-zero observation')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise RollingForcingError('invalid bound trajectory seed')
        if not isinstance(prompt, str) or not prompt.strip():
            raise RollingForcingError('empty bound prompt')
        if noise.ndim != 5 or noise.shape[0] != 1 or noise.shape[1] < 12 or noise.shape[1] % 3:
            raise RollingForcingError('full bound noise schedule requires at least four blocks')
        if pipeline.kv_cache_clean is not None or getattr(pipeline, 'crossattn_cache', None) is not None:
            raise RollingForcingError('session requires exclusive empty native caches')
        self.pipeline, self.torch = pipeline, torch
        self.noise, self.prompt, self.seed = noise.clone(), prompt, seed
        self.blocks = noise.shape[1] // 3
        self.stream = None
        self.observer = None
        self.observe_step_zero = observe_step_zero
        self.retain_hidden = retain_hidden
        self.capture_cache = capture_cache
        self.cpu_history = cpu_history
        self.expected_history = None
        self.next_index = 0
        self.closed = False

    def generate(self, *, history=None):
        if self.closed:
            raise RollingForcingError('session is closed')
        t = self.torch
        history_device = t.device('cpu') if self.cpu_history else self.noise.device
        if history is not None and (history.ndim != 5 or history.shape[0] != 1
                or history.shape[1] < 3 or history.shape[1] % 3
                or history.shape[2:] != self.noise.shape[2:]
                or history.device != history_device or history.dtype != self.noise.dtype):
            raise RollingForcingError('invalid accepted history')
        index = 0 if history is None else history.shape[1] // 3
        if index >= self.blocks:
            raise RollingForcingError('accepted history exhausts bound noise schedule')
        with stage('rolling_history_identity', detail='input'):
            if history is None:
                digest = None
            elif self.cpu_history:
                digest = _cpu_history_digest(history, None, t)
            else:
                digest = latent_digest(history, t)
        reuse = self.stream is not None and index == self.next_index and digest == self.expected_history
        discarded = self.stream is not None and not reuse
        if not reuse:
            # A native window can already contain observations of future
            # blocks. They belong to the abandoned branch, even when their
            # absolute positions match the replacement branch.
            if self.observer is not None:
                self.observer.close()
                self.observer = None
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            # Same prompt/trajectory and restart position share the same
            # re-noising stream across policy arms, regardless of latent values.
            restart_seed = int.from_bytes(hashlib.sha256(
                f'rolling-restart-v1:{self.seed}:{index}'.encode()).digest()[:8], 'big') % (2**63)
            if index == 0:
                restart_seed = self.seed
            devices = [self.noise.device.index if self.noise.device.index is not None
                       else t.cuda.current_device()] if self.noise.device.type == 'cuda' else []
            try:
                if self.observe_step_zero:
                    self.observer = RollingStepZeroObserver(pipeline=self.pipeline,
                        torch=t, retain_hidden=self.retain_hidden, capture_cache=self.capture_cache)
                with t.random.fork_rng(devices=devices), t.no_grad():
                    t.manual_seed(restart_seed)
                    if devices:
                        t.cuda.manual_seed_all(restart_seed)
                    with stage("rolling_restart", detail={"history_blocks":index}, cuda=True):
                        replay_history = history
                        if history is not None and self.cpu_history:
                            replay_history = history.to(device=self.noise.device).contiguous()
                        self.stream = RollingForcingStream(pipeline=self.pipeline, torch=t,
                            noise=self.noise[:,index*3:], prompt=self.prompt, history=replay_history)
            except BaseException:
                self.close()
                raise
        try:
            with stage("rolling_next_block", detail={"index":index,"reuse":reuse}, cuda=True):
                actual_index, latent = self.stream.next_block()
            if actual_index != index:
                raise RollingForcingError('rolling history cursor mismatch')
            observation = self.observer.take(index) if self.observer is not None else None
            with stage('rolling_history_identity', detail='output'):
                if self.cpu_history:
                    latent = latent.detach().to(device='cpu').contiguous()
                    self.expected_history = _cpu_history_digest(history, latent, t)
                else:
                    accepted = latent if history is None else t.cat([history, latent], dim=1)
                    self.expected_history = latent_digest(accepted, t)
            self.next_index = index + 1
            result = {'latent':latent, 'chunk_index':index, 'reused_rolling_state':reuse,
                    'replayed_history_blocks':0 if reuse else index,
                    'discarded_future_state':discarded,
                    'state_policy':'single_active_branch_clean_history_restart_v1'}
            if self.cpu_history:
                result['cpu_history_identity'] = {
                    'schema': 'rolling_cpu_history_identity_v1',
                    'input_sha256': digest,
                    'output_sha256': self.expected_history,
                    'history_blocks_transferred': 0 if reuse else index,
                }
            if self.stream.host_kv_cursors:
                from .rolling_host_cursors import evidence
                result['host_cursor_execution'] = evidence(self.pipeline.kv_cache_clean, torch=t)
            if observation is not None:
                result['step_zero_observation'] = observation
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.observer is not None:
            self.observer.close()
            self.observer = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        else:
            self.pipeline.kv_cache_clean = None
            self.pipeline.crossattn_cache = None
        self.noise = None
        self.expected_history = None
        self.closed = True
