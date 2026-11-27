"""Diagnostic Rolling suffix assimilation inside the original window sampler.

Only the first noisy forward is replaced. The original stream still owns every
re-noising draw, first-block clean write, later window, drain and RNG fork.
Ordinary sessions/factories do not import this module. Trained CUDA and complete
policy parity are still required before production selection.
"""
import hashlib
import json
import os
from types import SimpleNamespace

from .rolling_forcing_observation import RollingStepZeroObserver
from .rolling_forcing_stream import RollingForcingStream
from .incremental_draft import latent_digest
from .rolling_committed_prefix import CommittedPrefixSession
from .rolling_joint_handoff import ENV, active_handoff, joint_first_window, require
from .rolling_prefix_transaction import AcceptedPrefix
from .self_forcing_frame_adapter import sampled_kv_cache_witness
from .stage_timing import stage


class RollingCleanWitness:
    """Canonical samples from each layer before its first noisy slot write."""
    def __init__(self, *, pipeline, torch, expected_start):
        require(type(expected_start) is int and expected_start > 0
                and expected_start % (3*pipeline.frame_seq_length) == 0,
                'joint Rolling witness requires the next logical Draft position')
        self.pipeline, self.torch = pipeline, torch
        self.expected_start = expected_start
        self.layers = []
        self.finished = False

    def observe(self, index, k, v, global_end, local_end):
        t, p = self.torch, self.pipeline
        require(not self.finished and type(index) is int and index == len(self.layers) and index < 30
                and global_end == self.expected_start
                and local_end == min(global_end, p.kv_cache_clean[index]['k'].shape[1])
                and isinstance(k, t.Tensor) and isinstance(v, t.Tensor)
                and k.shape == v.shape == p.kv_cache_clean[index]['k'].shape
                and k.ndim == 4 and k.shape[0] == 1 and k.dtype == v.dtype == t.bfloat16
                and k.device == v.device,
                'joint Rolling accepted-history witness order or cache geometry differs')
        _, descriptor = sampled_kv_cache_witness([dict(k=k, v=v,
            global_end_index=global_end, local_end_index=local_end)],
            batched=False, reuse_indices=False)
        record = descriptor['layers'][0]; record['layer'] = index
        self.layers.append(record)

    def finish(self):
        p = self.pipeline
        require(not self.finished and len(self.layers) == 30
                and len(p.kv_cache_clean) == len(p.crossattn_cache) == 30
                and all(row.get('is_init') is True for row in p.crossattn_cache),
                'joint Rolling witness is incomplete or conditioning changed')
        first = p.kv_cache_clean[0]
        # Current self KV is already noisy; read only immutable cross KV here.
        _, descriptor = sampled_kv_cache_witness([dict(k=first['k'], v=first['v'],
            global_end_index=0, local_end_index=0)], p.crossattn_cache,
            batched=False, reuse_indices=False)
        descriptor['layers'] = self.layers
        digest = hashlib.sha256(json.dumps(descriptor, ensure_ascii=False,
            allow_nan=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.finished = True
        return digest, descriptor


class JointRollingObserver(RollingStepZeroObserver):
    """Observe the noisy B3 at layer15; exclude all accepted clean tokens."""
    def __init__(self, *, pipeline, **kwargs):
        self.pipeline = pipeline
        self.clean_witness = None
        super().__init__(pipeline=pipeline, **kwargs)

    def _begin(self, module, args, kwargs):
        handoff = active_handoff()
        if handoff is None:
            return super()._begin(module, args, kwargs)
        start, groups = handoff
        steps = kwargs.get('timestep')
        t = self.torch
        require(type(groups) is int and 1 <= groups <= 4 and type(start) is int
                and start > 3*groups*self.frame_tokens and start % (3*self.frame_tokens) == 0
                and not self.closed and self.active is None and self.clean_witness is None
                and isinstance(steps, t.Tensor) and tuple(steps.shape) == (1, 3*(groups+1))
                and tuple(steps.detach().cpu().reshape(-1).tolist()) == (0.,)*(3*groups)+(1000.,)*3
                and kwargs.get('current_start') == start-3*groups*self.frame_tokens
                and kwargs.get('updating_cache') is False,
                'joint Rolling observation does not match its clean/noisy partition')
        index = start//(3*self.frame_tokens)
        require(index not in self.pending and not self.pending,
                'joint Rolling requires the abandoned future observations to be discarded')
        self.active = (index, 3*groups*self.frame_tokens, 3*(groups+1)*self.frame_tokens)
        self.cache_witness = None
        if self.capture_cache:
            self.clean_witness = RollingCleanWitness(pipeline=self.pipeline, torch=t,
                                                     expected_start=start)

    def observe_clean(self, *args):
        if self.capture_cache:
            require(self.clean_witness is not None, 'joint Rolling clean observation has no owner')
            self.clean_witness.observe(*args)

    def _end(self, module, args, output):
        if active_handoff() is not None and self.capture_cache:
            require(self.active is not None and self.active[0] in self.pending
                    and self.clean_witness is not None,
                    'joint Rolling rank or accepted-history witness is missing')
            self.pending[self.active[0]]['kv_witness'] = self.clean_witness.finish()
            self.clean_witness = None
        return super()._end(module, args, output)

    def close(self):
        self.clean_witness = None
        super().close()


class JointRollingStream(RollingForcingStream):
    """Consume an owned prefix and 1--4 accepted Target B3 blocks once.

    The caller first discards the previous stream's speculative future and its
    pending observer. This constructor transfers only that same Draft owner's
    retained prefix. It never imports Target KV or snapshots an entire cache.
    """
    def __init__(self, *, pipeline, torch, noise, prompt, retained_history,
                 target_suffix, condition, clean_observer=None):
        require(os.environ.get(ENV) == '1' and not torch.is_grad_enabled()
                and os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS', '0') == '0'
                and retained_history.ndim == target_suffix.ndim == noise.ndim == 5
                and retained_history.shape[0] == target_suffix.shape[0] == noise.shape[0] == 1
                and retained_history.shape[1] >= 3 and retained_history.shape[1] % 3 == 0
                and target_suffix.shape[1] in (3,6,9,12)
                and retained_history.shape[2:] == target_suffix.shape[2:] == noise.shape[2:]
                and retained_history.dtype == target_suffix.dtype == noise.dtype == torch.bfloat16
                and retained_history.device == target_suffix.device == noise.device
                and (clean_observer is None or callable(clean_observer)),
                'joint Rolling stream requires native diagnostic ownership and compatible BF16 blocks')
        accepted = AcceptedPrefix(pipeline=pipeline, torch=torch, history=retained_history,
                                   prompt=prompt, condition=condition)
        super().__init__(pipeline=pipeline, torch=torch, noise=noise, prompt=prompt,
                         history=retained_history, accepted_prefix=accepted)
        self.joint_suffix = target_suffix.clone()
        self.joint_history_start = self.prefix_blocks*3*pipeline.frame_seq_length
        self.prefix_blocks += target_suffix.shape[1]//3
        self.clean_observer = clean_observer
        self.joint_calls = 0

    def _advance_window(self):
        if self.joint_suffix is None:
            return super()._advance_window()
        require(self.window == 0 and self.joint_calls == 0,
                'joint Rolling suffix must be consumed at the first window only')
        p, t, original = self.pipeline, self.torch, self.pipeline.generator
        view = SimpleNamespace(**{name:getattr(p, name) for name in (
            'denoising_step_list','num_frame_per_block','independent_first_frame',
            'args','frame_seq_length','kv_cache_clean','crossattn_cache')}, generator=original)
        calls = 0

        def dispatch(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                start = self.prefix_blocks*3*p.frame_seq_length
                require(kwargs.get('current_start') == start
                        and not kwargs.get('updating_cache', False)
                        and t.equal(kwargs['noisy_image_or_video'], self.noise[:, :3])
                        and tuple(kwargs['timestep'].reshape(-1).tolist()) == (1000.,)*3,
                        'joint Rolling first native noisy call differs')
                with stage('rolling_joint_suffix_first_window', detail={
                        'clean_blocks': self.joint_suffix.shape[1]//3}, cuda=True):
                    clean = joint_first_window(view, torch=t, history=self.joint_suffix,
                        noise=kwargs['noisy_image_or_video'], conditional=self.condition,
                        history_start=self.joint_history_start, draft_start=start,
                        clean_observer=self.clean_observer)
                self.joint_calls += 1
                return None, clean
            require(calls == 2 and kwargs.get('updating_cache') is True,
                    'joint Rolling must retain its original first-block clean write')
            return original(**kwargs)

        # The real pipeline registers generator as a Torch child module.
        # Feed a local view to the unchanged window sampler instead of
        # replacing that child; physical calls still reach its original
        # module, including rank hooks and the CUDA forward observer.
        window_view = SimpleNamespace(**vars(view))
        window_view.generator = dispatch
        window_view.scheduler = p.scheduler
        self.pipeline = window_view
        try:
            result = super()._advance_window()
            require(calls == 2 and self.joint_calls == 1,
                    'joint Rolling did not execute one joint call and its native clean write')
            self.joint_suffix = self.clean_observer = None
            return result
        finally:
            self.pipeline = p

    def close(self):
        self.joint_suffix = self.clean_observer = None
        super().close()

    def abandon_future(self):
        self.joint_suffix = self.clean_observer = None
        return super().abandon_future()


class JointCommittedSession(CommittedPrefixSession):
    """Diagnostic keep-current route with a frozen two- or four-block suffix."""
    def __init__(self, *, target_suffix_blocks=4, **kwargs):
        require(os.environ.get(ENV) == '1'
                and os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS', '0') == '0',
                'joint Rolling session requires explicit native tensor-cursor diagnostics')
        require(type(target_suffix_blocks) is int and target_suffix_blocks in (2, 4),
                'joint Rolling comparison requires a fixed B2 or B4 budget')
        super().__init__(**kwargs)
        self.target_suffix_blocks = target_suffix_blocks

    def _restart_after_target(self, history, index, digest):
        t, p = self.torch, self.pipeline
        retained = self.next_index
        prefix = history[:, :retained*3]
        lease = AcceptedPrefix(pipeline=p, torch=t, history=prefix, prompt=self.prompt,
                               condition=self.stream.condition)
        lease.consume(pipeline=p, torch=t, history=prefix, prompt=self.prompt)
        try:
            if self.observer is not None:
                self.observer.close(); self.observer = None
            condition = self.stream.abandon_future()
            self.stream = None
            devices = ([self.noise.device.index if self.noise.device.index is not None
                        else t.cuda.current_device()] if self.noise.device.type == 'cuda' else [])
            # Match the original restart exactly; constructing the diagnostic
            # stream must not draw or resize any sampler randomness.
            restart_seed = int.from_bytes(hashlib.sha256(
                f'rolling-restart-v1:{self.seed}:{index}'.encode()).digest()[:8], 'big') % (2**63)
            with t.random.fork_rng(devices=devices), t.no_grad():
                t.manual_seed(restart_seed)
                if devices: t.cuda.manual_seed_all(restart_seed)
                if self.observe_step_zero:
                    self.observer = JointRollingObserver(pipeline=p, torch=t,
                        retain_hidden=self.retain_hidden, capture_cache=self.capture_cache)
                self.stream = JointRollingStream(pipeline=p, torch=t,
                    noise=self.noise[:, index*3:], prompt=self.prompt, retained_history=prefix,
                    target_suffix=history[:, retained*3:], condition=condition,
                    clean_observer=(self.observer.observe_clean if self.observer is not None else None))
            self.next_index, self.expected_history = index, digest
        except BaseException:
            self.close()
            raise

    def generate(self, *, history=None):
        retained = self.next_index
        width = self.target_suffix_blocks
        handoff = (self.stream is not None and isinstance(history, self.torch.Tensor)
                   and history.ndim == 5 and history.shape[1]//3 != retained)
        # The ordinary committed session admits exactly B4. Keep that path
        # intact; the B2 diagnostic transfers the owner only after imposing
        # the same history check at its explicitly frozen width.
        if width == 2 and handoff:
            t = self.torch
            index = retained + width
            require(not self.closed and isinstance(history, t.Tensor)
                    and history.shape == (1,index*3,*self.noise.shape[2:])
                    and history.dtype == self.noise.dtype and history.device == self.noise.device
                    and index < self.blocks
                    and latent_digest(history[:, :retained*3], t) == self.expected_history,
                    'B2 handoff requires the unchanged accepted Draft prefix and two Target blocks')
            self._restart_after_target(history, index, latent_digest(history, t))

        result = super().generate(history=history)
        if width == 2 and handoff:
            result.update(reused_rolling_state=False, replayed_history_blocks=2,
                          discarded_future_state=True)
            result['committed_prefix'].update(action='accepted_prefix_and_target_suffix',
                retained_prefix_blocks=retained, clean_suffix_blocks=2)
        result['state_policy'] = 'diagnostic_joint_committed_rolling_window_v1'
        result['joint_execution'] = dict(schema='rolling_joint_window_execution_v1',
            fused_clean_blocks=width if handoff else 0,
            joint_first_window_calls=1 if handoff else 0,
            separate_suffix_clean_calls=0,
            native_rolling_windows_and_drains_preserved=True,
            native_first_block_clean_writes_preserved=True,
            trained_cuda_parity_established=False)
        if handoff:
            require(self.stream.joint_calls == 1 and self.stream.joint_suffix is None,
                    'joint committed restart did not consume the accepted suffix exactly once')
        return result
