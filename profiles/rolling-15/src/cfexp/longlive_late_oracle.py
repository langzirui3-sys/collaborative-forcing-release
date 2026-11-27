"""Fresh-state CUDA oracles for late handoff; never part of timed execution."""
from __future__ import annotations

from .longlive_late_handoff import POLICIES


def draft_reference(backend, *, prompt, history, noise, noises=(), policy=None, accepted=None):
    p, t = backend.session.executor.pipeline, backend.torch
    p._initialize_kv_cache(batch_size=1, dtype=t.bfloat16, device=noise.device)
    p._initialize_crossattn_cache(batch_size=1, dtype=t.bfloat16, device=noise.device)
    payload = dict(self=p.kv_cache1, cross=p.crossattn_cache)
    n = 0 if history is None else history.shape[1]//3
    try:
        with t.inference_mode():
            cond = p.text_encoder(text_prompts=[prompt])
            for i in range(n):
                p.generator(noisy_image_or_video=history[:,3*i:3*i+3],conditional_dict=cond,
                    timestep=t.zeros((1,3),device=noise.device,dtype=t.int64),
                    kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                    current_start=3*i*p.frame_seq_length)
            if accepted is not None:
                p.generator(noisy_image_or_video=accepted,conditional_dict=cond,
                    timestep=t.zeros((1,3),device=noise.device,dtype=t.int64),
                    kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                    current_start=3*n*p.frame_seq_length)
                return accepted, payload
            updates, handoff = POLICIES[policy]
            current = noise
            for index in range(updates):
                _, prediction = p.generator(noisy_image_or_video=current,conditional_dict=cond,
                    timestep=t.ones((1,3),device=noise.device,dtype=t.int64)*p.denoising_step_list[index],
                    kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                    current_start=3*n*p.frame_seq_length)
                destination = p.denoising_step_list[index+1] if index+1 < updates else handoff
                current = p.scheduler.add_noise(prediction.flatten(0,1),noises[index].flatten(0,1),
                    destination*t.ones((3,),device=noise.device,dtype=t.long)).unflatten(0,prediction.shape[:2])
            return current, payload
    finally:
        p.kv_cache1 = p.crossattn_cache = None


def target_reference(backend, *, prompt, history, noisy, timestep, chunk_index, internal_seed):
    """Run the unmodified upstream block loop with one explicit schedule entry.

This establishes equality to that declared new policy, never equivalence to
the checkpoint's four-update sampler or its quality.
"""
    from .krea_chunk_backend import copy_explicit_session_noise
    t = backend.torch
    params = backend.release_server.GenerateParams(prompt=prompt,width=832,height=480,
        seed=internal_seed,context_noise=0.,keep_first_frame=True,kv_cache_num_frames=3,
        num_blocks=backend.chunk_count,num_denoising_steps=4,timestep_shift=5.)
    session = backend.release_server.GenerationSession(params,backend.config,
        frame_callback=lambda *_a,**_k:None,models=backend.models)
    with t.inference_mode():
        session.all_latents[:,:3*chunk_index].copy_(history.to(backend.device))
        session.block_idx = chunk_index
        session.current_start_frame = 3*chunk_index
        session.resume_latents = None
        session.denoising_step_list = [timestep]
        copy_explicit_session_noise(session=session,noise=noisy.to(backend.device),
            history_frames=3*chunk_index,torch_module=t)
        session.rnd.manual_seed(internal_seed)
        condition = backend._condition(prompt)
        session.conditional_dict = condition
        session.current_prompt_embeds = condition['prompt_embeds']
        session.generate_block(backend.models)
        return session.last_pred.detach().to(device='cpu',dtype=t.bfloat16).contiguous()
