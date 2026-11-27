"""Opt-in D-prefix -> one final T clean prediction; a new sampling policy.

None of these policies is four-step-equivalent. Only noisy latents cross models; the
accepted prediction is assimilated by one native LongLive B3 clean forward.
"""
from __future__ import annotations

from .early_draft import EarlyDraftExecutor
from .longlive_low_noise import require, SCHEDULE

ENV = 'CFEXP_LONGLIVE_LATE_HANDOFF_DIAGNOSTIC'
POLICIES = {'d1_t750': (1, 750.), 'd1_t625': (1, 625.), 'd3_t625': (3, 625.)}


def validate_trace(trace, steps, *, start):
    require(trace == [dict(timesteps=[float(s)] * 3, current_start=start) for s in steps],
            'late handoff executed a different native forward sequence')


class LateDraft:
    def __init__(self, backend):
        require(not backend.session.host_cursors.enabled,
                'late handoff gate requires native device cursors')
        self.backend = backend
        self.p = backend.session.executor.pipeline
        require([float(x) for x in self.p.denoising_step_list] == SCHEDULE,
                'late handoff requires the exact native LongLive schedule')
        self.t = backend.torch
        self.early = EarlyDraftExecutor(pipeline=self.p, torch=self.t,
            scope_sha256=backend.session.executor.scope_sha256)

    def _record(self, trace):
        def before(_module, _args, kwargs):
            trace.append(dict(timesteps=self.backend._timestep_values(kwargs['timestep']),
                              current_start=kwargs['current_start']))
        return self.p.generator.register_forward_pre_hook(before, with_kwargs=True)

    def _allocate(self, noise):
        p, t = self.p, self.t
        p._initialize_kv_cache(batch_size=1, dtype=t.bfloat16, device=noise.device)
        p._initialize_crossattn_cache(batch_size=1, dtype=t.bfloat16, device=noise.device)
        return dict(self=p.kv_cache1, cross=p.crossattn_cache)

    def _propose(self, *, history, widths, noise, noises, policy, prompt):
        e, p, t = self.early, self.p, self.t
        require(policy in POLICIES and not e.poisoned and e.pending is None,
                'invalid late proposal or unresolved previous proposal')
        updates, handoff = POLICIES[policy]
        require(len(noises) == updates and widths == (3,) * len(widths),
                'late handoff needs explicit renoising and native B3 history')
        for value in (noise, *noises):
            require(isinstance(value, t.Tensor) and tuple(value.shape) == (1,3,16,60,104)
                and value.dtype == t.bfloat16 and value.device == noise.device
                and bool(t.isfinite(value.float()).all()),
                'invalid late handoff noise')
        scope, binding = e._identity(prompt, history, widths)
        require(e.payload is None or (e.scope == scope and binding[:len(e.binding)] == e.binding),
                'late handoff accepted history changed without a new session')
        start = len(e.binding) if e.payload is not None else 0
        require(len(binding) - start in (0, 1), 'late handoff cannot replay an old accepted prefix')
        if e.payload is None:
            e.payload = self._allocate(noise)
            from .incremental_draft import tensor_bytes
            require(tensor_bytes(e.payload,t) <= e.max_bytes, 'late handoff cache exceeds its memory bound')
            e.scope = scope
        e._attach()
        e._cursor(start)
        cond = p.text_encoder(text_prompts=[prompt])
        for i in range(start, len(binding)):
            e._clean(history[:,3*i:3*i+3], cond, i, 3)
        e.binding = binding
        e._cursor(len(binding))
        trace = []
        hook = self._record(trace)
        try:
            noisy = noise
            for i, step in enumerate(p.denoising_step_list[:updates]):
                _, clean = p.generator(noisy_image_or_video=noisy, conditional_dict=cond,
                    timestep=t.ones((1,3), device=noise.device, dtype=t.int64)*step,
                    kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                    current_start=len(binding)*e.tokens_per_chunk)
                next_step = p.denoising_step_list[i+1] if i+1 < updates else handoff
                noisy = p.scheduler.add_noise(clean.flatten(0,1), noises[i].flatten(0,1),
                    next_step*t.ones((3,),device=noise.device,dtype=t.long)).unflatten(0,clean.shape[:2])
        finally:
            hook.remove()
        validate_trace(trace, SCHEDULE[:updates], start=len(binding)*e.tokens_per_chunk)
        e._cursor(len(binding), tentative=True)
        e.pending = dict(scope=scope, binding=binding, shape=tuple(noise.shape),
                         device=noise.device, policy=policy)
        return dict(noisy_latent=noisy, policy=policy, handoff_timestep=handoff,
            draft_steps=SCHEDULE[:updates], forward_trace=trace,
            reused_chunks=start, replayed_chunks=len(binding)-start,
            draft_clean_forwards=0, branch_snapshot_bytes=0)

    def propose(self, **kwargs):
        try:
            with self.t.inference_mode():
                return self._propose(**kwargs)
        except BaseException:
            self.early.payload = self.early.pending = None
            self.early.poisoned = True
            raise
        finally:
            self.early._detach()

    def commit(self, *, prompt, history, widths, latent):
        e, t = self.early, self.t
        require(not e.poisoned and e.pending is not None, 'late commit has no pending proposal')
        scope, binding = e._identity(prompt, history, widths)
        require(scope == e.pending['scope'] and binding == e.pending['binding'],
                'late commit belongs to another accepted history')
        require(isinstance(latent, t.Tensor) and tuple(latent.shape) == e.pending['shape']
            and latent.dtype == t.bfloat16 and bool(t.isfinite(latent.float()).all()),
            'invalid late Target clean prediction')
        trace = []
        hook = self._record(trace)
        try:
            with t.inference_mode():
                e._attach()
                e._cursor(len(binding), tentative=True)
                clean = latent.to(e.pending['device']).contiguous()
                cond = self.p.text_encoder(text_prompts=[prompt])
                e._clean(clean, cond, len(binding), 3)
                e._cursor(len(binding)+1)
                e.binding = binding+(e._chunk_binding(clean,3),)
                e.pending = None
            validate_trace(trace, [0.], start=len(binding)*e.tokens_per_chunk)
            return dict(forward_trace=trace, draft_clean_forwards=1,
                history_replay_chunks=0, committed_kv_width=3,
                committed_chunks=len(e.binding), branch_snapshot_bytes=0)
        except BaseException:
            e.payload = e.pending = None
            e.poisoned = True
            raise
        finally:
            hook.remove()
            e._detach()


def target_prediction(backend, *, prompt, history, noisy, timestep, chunk_index, internal_seed):
    """One clean prediction at a declared noise level, using native bounded context.

The ordinary full Target root must have initialized this correction session.
No scheduler step, VAE decode or Draft KV transfer occurs here.
"""
    from .self_forcing_frame_adapter import sampled_kv_cache_witness
    t, models = backend.torch, backend.models
    require(type(timestep) is float and timestep in {750.,625.} and type(chunk_index) is int
        and 1 <= chunk_index < backend.chunk_count and type(internal_seed) is int and internal_seed >= 0,
        'unsupported late Target input')
    require(backend._persistent_session is None and backend._correction_session is not None
        and backend._correction_prompt == prompt, 'late Target requires its own full Target root session')
    require(isinstance(history,t.Tensor) and tuple(history.shape) == (1,3*chunk_index,16,60,104)
        and history.dtype == t.bfloat16 and isinstance(noisy,t.Tensor)
        and tuple(noisy.shape) == (1,3,16,60,104) and noisy.dtype == t.bfloat16
        and bool(t.isfinite(history.float()).all()) and bool(t.isfinite(noisy.float()).all()),
        'late Target history/noisy latent differs')
    session = backend._correction_session
    trace = []
    def before(_module, _args, kwargs):
        values = backend.torch.unique(kwargs['timestep']).detach().cpu().tolist()
        require(len(values) == 1, 'unequal late Target frame timesteps')
        trace.append(dict(timestep=float(values[0]), current_start=kwargs['current_start'],
                          batch_timesteps_equal=True))
    hook = models.transformer.register_forward_pre_hook(before, with_kwargs=True)
    try:
        with t.inference_mode():
            accepted = history.to(backend.device).contiguous()
            session.all_latents[:,:3*chunk_index].copy_(accepted)
            require(bool(t.equal(session.all_latents[:,:3*chunk_index],accepted)),
                    'late Target did not bind the accepted history')
            session.block_idx = chunk_index
            session.current_start_frame = 3*chunk_index
            session.resume_latents = None
            session.rnd.manual_seed(internal_seed)
            condition = backend._condition(prompt)
            session.conditional_dict = condition
            session.current_prompt_embeds = condition['prompt_embeds']
            start = session.recompute_kv_cache(models)
            require(start == 3, 'late Target native bounded context changed')
            witness = sampled_kv_cache_witness(models.pipeline.kv_cache1, models.pipeline.crossattn_cache)
            _, clean = models.transformer(noisy_image_or_video=noisy.to(backend.device).contiguous(),
                conditional_dict=condition,
                timestep=t.full((1,3),timestep,device=backend.device,dtype=t.int64),
                kv_cache=models.pipeline.kv_cache1, crossattn_cache=models.pipeline.crossattn_cache,
                current_start=start*models.pipeline.frame_seq_length)
            require(tuple(clean.shape) == (1,3,16,60,104) and bool(t.isfinite(clean.float()).all()),
                    'late Target returned an invalid clean prediction')
            clean = clean.detach().to(device='cpu',dtype=t.bfloat16).contiguous()
    finally:
        hook.remove()
    expected = [dict(timestep=step,current_start=4680,batch_timesteps_equal=True) for step in (0.,timestep)]
    require(trace == expected, 'late Target ran more than native context plus one denoising forward')
    return dict(latent=clean, forward_trace=trace, denoising_schedule=[timestep],
        native_session_id=backend._correction_session_id, accepted_history_frames=3*chunk_index,
        bounded_context_rebase=True, context_frames=3, input_representation='noisy_latent_at_timestep',
        completed_four_step_target_chunk=False, sampling_policy='one_final_target_clean_prediction',
        kv_witness_sha256=witness[0], kv_witness=witness[1])
