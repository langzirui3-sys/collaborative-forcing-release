"""Study-specific Krea factory. Original strict artifact loader is unchanged.

Both full Target and intervention use the same persistent four-step Target
schedule. An intervention primes canonical clean history once, then uses the
native recompute-KV path. No Draft model is used by the full Target reference.
"""
from __future__ import annotations
import gc
import types

from .cross_model_delay import HiddenRankRecorder, require


def build_backend(config):
    from .correction_runtime import build_target_backend
    backend = build_target_backend(config)
    t = backend.torch
    recorder = HiddenRankRecorder(t)
    native = backend.generate_persistent_chunk
    def pre(_m, _a, kw):
        recorder.active = bool(float(kw['timestep'].reshape(-1)[0]) == 1000)
    def post(_m, _a, _kw, _out):
        recorder.active = False
    handles = [backend.models.transformer.register_forward_pre_hook(pre, with_kwargs=True),
               backend.models.transformer.register_forward_hook(post, with_kwargs=True),
               backend.models.transformer.model.blocks[14].register_forward_hook(recorder.observe)]
    def persistent(self, **kw):
        recorder.begin(producer='krea_14b', chunk=kw['chunk_index'])
        from .incremental_draft import latent_digest
        scheduler = self.models.pipeline.scheduler
        original_add = scheduler.add_noise
        noises = []
        def add(clean, noise, timestep):
            noises.append(latent_digest(noise,t))
            return original_add(clean, noise, timestep)
        scheduler.add_noise = add
        try:
            result = native(**kw)
        finally:
            scheduler.add_noise = original_add
        require(len(noises) == 3, 'Target actual renoise coverage')
        result['renoise_tensor_sha256'] = noises
        result['erank_observation'] = recorder.end()
        return result
    backend.generate_persistent_chunk = types.MethodType(persistent, backend)
    def chunk(self, *, prompt, history_latents, initial_noise, internal_seed, chunk_index, capture_hidden=False):
        require(not capture_hidden, 'study does not transport full hidden states')
        require(self._persistent_session is None, 'intervention conflicts with live Target trajectory')
        started = self.start_persistent_trajectory(prompt_id='cross-model-intervention', prompt=prompt,
                                                 seed=internal_seed, chunk_count=self.chunk_count)
        session = self._persistent_session
        try:
            if chunk_index:
                require(history_latents is not None and tuple(history_latents.shape) == (1,3*chunk_index,16,60,104), 'invalid intervention history')
                history = history_latents.to(device=self.device, dtype=t.bfloat16).contiguous()
                # The native session creates all_latents under inference_mode.
                # Preserve that native allocation and exact values; perform its
                # canonical-prefix write in the same mode rather than replacing
                # or cloning the whole history buffer.
                with t.inference_mode():
                    session.all_latents[:, :3*chunk_index].copy_(history)
                session.current_start_frame = 3*chunk_index
                session.block_idx = chunk_index
                session.resume_latents = None
                require(t.equal(session.all_latents[:, :3*chunk_index], history), 'Target canonical prefix differs')
            else:
                require(history_latents is None, 'root Target requires empty history')
            result = self.generate_persistent_chunk(prompt=prompt, initial_noise=initial_noise,
                       internal_seed=internal_seed, chunk_index=chunk_index)
            result.update(standalone_external_prefix_session=True, session_recreated=True, history_reconstructed=bool(chunk_index))
            result['history_priming'] = {'chunk_count':chunk_index, 'exact_canonical_history':True,
                                        'native_recompute_kv':True}
            return result
        finally:
            # This request owns one standalone block, not a full-Target
            # trajectory. Discard its private session exactly as the original
            # generate_chunk does; never fake a completed horizon or weaken
            # finish_persistent_trajectory's full-horizon guard.
            require(self._persistent_session is session, 'private intervention session ownership drift')
            self._persistent_session = None
            self._persistent_session_id = None
            self._persistent_prompt = None
            self._persistent_seed = None
            del session
            gc.collect()
            t.cuda.empty_cache()
    backend.generate_chunk = types.MethodType(chunk, backend)
    old_close = backend.close
    def close(self):
        for handle in handles:
            handle.remove()
        old_close()
    backend.close = types.MethodType(close, backend)
    backend.model_state = dict(backend.model_state, cross_model_delay_rank=True,
                              intervention_schedule='same_as_full_target_MAIN_TABLE_FOUR_STEP_SCHEDULE')
    return backend
