"""Native Rolling sessions exposed as history-bound common policy requests.

No Self-Forcing inference loop is used. The caller owns the verified native
backend, conditioning and quality scorer. Formal factory routing still requires
trained-model Target-handoff validation; this class alone is not that evidence.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .correction_runtime import NOISE_SHAPE
from .rolling_forcing_noise import load_schedule, verify_schedule
from .rolling_forcing_session import RollingForcingSession, cpu_history_enabled
from .rolling_forcing_stream import RollingForcingError
from .self_forcing_frame_adapter import (
    OfficialSelfForcingBackend as _Observation, effective_rank_evidence_from_capture,
)


def _require(value, message):
    if not value:
        raise RollingForcingError(message)


class RollingPolicyBackend:
    """Own one active rolling branch, with at most four pending observations.

    Future noises come exclusively from sealed common-controller schedules.
    A chunk-local internal seed selects an existing schedule; it never seeds an
    invented future. Returning to an earlier branch restarts from its exact
    accepted latents and discards the abandoned partially denoised suffix.
    """

    supports_history_kv_widths = True
    decode_latents = _Observation.decode_latents
    temporal_consistency = _Observation.temporal_consistency
    write_video = _Observation.write_video
    score_frame_indices = _Observation.score_frame_indices

    def __init__(self, *, native_backend, run_dir, schedule_bindings,
                 chunk_count, write_video_impl):
        _require(type(chunk_count) is int and chunk_count in (16,30,60,120),
                 'unsupported Rolling policy horizon')
        _require(isinstance(schedule_bindings, (list,tuple)) and bool(schedule_bindings),
                 'Rolling policy needs frozen full noise schedules')
        self.torch, self.device = native_backend.torch, native_backend.device
        self.pipeline = native_backend.pipeline
        self.run_dir = Path(run_dir).resolve(strict=True)
        self.chunk_count = chunk_count
        self.cpu_history = cpu_history_enabled()
        self.write_video_impl = write_video_impl
        self.bindings, self.payloads, self.requests = [], [], {}
        for number, item in enumerate(schedule_bindings):
            binding = dict(item)
            _require(Path(binding['run_dir']).resolve(strict=True) == self.run_dir,
                     'noise schedule belongs to another run')
            _require(binding['chunk_count'] == chunk_count, 'noise horizon differs from worker')
            payload, _ = verify_schedule(**binding)
            for index, row in enumerate(payload['chunks']):
                key = (payload['prompt'], index, row['noise']['derived_seed'])
                _require(key not in self.requests, 'ambiguous or duplicate Rolling noise request')
                self.requests[key] = number
            self.bindings.append(binding)
            self.payloads.append(payload)
        self.session = None
        self.active_schedule = None
        self.closed = False

    def generate_chunk(self, *, prompt, history_latents, initial_noise,
                       internal_seed, chunk_index, capture_hidden,
                       kv_branch_action='none', history_kv_widths=None):
        _require(not self.closed, 'Rolling policy backend is closed')
        _require(kv_branch_action == 'none', 'Rolling does not support SF KV fork actions')
        _require(type(chunk_index) is int and 0 <= chunk_index < self.chunk_count,
                 'invalid Rolling chunk index')
        _require(type(internal_seed) is int and internal_seed >= 0,
                 'invalid Rolling internal seed')
        _require(type(capture_hidden) is bool, 'capture_hidden must be boolean')
        _require(isinstance(prompt,str) and bool(prompt.strip()), 'invalid Rolling prompt')
        if history_kv_widths is not None:
            _require(isinstance(history_kv_widths,tuple)
                     and len(history_kv_widths) == chunk_index
                     and all(type(x) is int and x == 3 for x in history_kv_widths),
                     'Rolling requires three-frame native clean history updates')
        t = self.torch
        _require(isinstance(initial_noise,t.Tensor)
                 and tuple(initial_noise.shape) == NOISE_SHAPE
                 and initial_noise.dtype == t.bfloat16
                 and bool(t.isfinite(initial_noise).all()), 'invalid explicit Rolling noise')
        history = None
        if chunk_index:
            _require(isinstance(history_latents,t.Tensor)
                     and tuple(history_latents.shape) == (1,chunk_index*3,*NOISE_SHAPE[2:])
                     and history_latents.dtype == t.bfloat16
                     and bool(t.isfinite(history_latents).all()), 'invalid accepted Rolling history')
            if self.cpu_history:
                _require(history_latents.device.type == 'cpu',
                         'Rolling CPU history must use the worker-loaded accepted latents')
                history = history_latents
            else:
                history = history_latents.to(device=self.device).contiguous()
        else:
            _require(history_latents is None, 'first Rolling chunk must have no history')
        key = (prompt,chunk_index,internal_seed)
        _require(key in self.requests, 'request is not bound to a frozen Rolling schedule')
        selected = self.requests[key]
        if self.active_schedule != selected:
            from safetensors.torch import load
            # Check all bytes before discarding the previous valid session.
            noise, seed = load_schedule(torch=t, load_file=load, device=self.device,
                                        **self.bindings[selected])
            expected = noise[:,chunk_index*3:(chunk_index+1)*3]
            _require(t.equal(expected,initial_noise.to(device=self.device)),
                     'current noise differs from its frozen future schedule')
            if self.session is not None:
                self.session.close()
            self.session = None
            self.active_schedule = None
            self.session = RollingForcingSession(pipeline=self.pipeline,torch=t,
                noise=noise,prompt=prompt,seed=seed,observe_step_zero=True,
                retain_hidden=True,capture_cache=True,cpu_history=self.cpu_history)
            self.active_schedule = selected
        else:
            expected = self.session.noise[:,chunk_index*3:(chunk_index+1)*3]
            _require(t.equal(expected,initial_noise.to(device=self.device)),
                     'current noise differs from active Rolling schedule')
        try:
            generated = self.session.generate(history=history)
            observation = generated['step_zero_observation']
            _require(observation['chunk_index'] == chunk_index,
                     'Rolling observation belongs to another block')
            latent = generated['latent'].detach().to(device='cpu',dtype=t.bfloat16).contiguous()
            _require(tuple(latent.shape) == NOISE_SHAPE and bool(t.isfinite(latent).all()),
                     'invalid native Rolling output')
            witness = observation['kv_witness']
            _require(witness is not None, 'native Rolling cache witness is missing')
            evidence = effective_rank_evidence_from_capture(SimpleNamespace(
                retained_rank_layer=14, retained_gram_eigenvalues=observation['gram_eigenvalues'],
                ranks={14:observation['effective_rank']}))
            result = dict(latent=latent, rank=observation['effective_rank'],rank_evidence=evidence,
                sampled_hidden=observation['sampled_hidden'] if capture_hidden else None,
                full_hidden=observation['hidden'] if capture_hidden else None,
                kv_witness_sha256=witness[0],kv_witness=witness[1],
                rolling_state={k:generated[k] for k in ('chunk_index','reused_rolling_state',
                    'replayed_history_blocks','discarded_future_state','state_policy')})
            if 'host_cursor_execution' in generated:
                result['rolling_state']['host_cursor_execution'] = generated['host_cursor_execution']
            if self.cpu_history:
                result['rolling_state']['cpu_history_identity'] = generated['cpu_history_identity']
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.active_schedule = None
        self.closed = True
