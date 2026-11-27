"""Experimental T1000 clean prediction -> LongLive's last native update.

This is a new sampling policy, not an equivalent implementation of four-step
LongLive. Native accepted-history KV ownership and B3 clean updates are retained.
"""
from __future__ import annotations

from .longlive_native import LongLiveNativeError

ENV = 'CFEXP_LONGLIVE_LOW_NOISE_DIAGNOSTIC'
ACTION = 'resume_longlive_low_noise'
SCHEMA = 'longlive_low_noise_bridge_v1'
SCHEDULE = [1000., 937.5, 833.3333129882812, 625.]


def require(ok, message):
    if not ok:
        raise LongLiveNativeError(message)


def validate_evidence(value):
    require(value.get('schema') == SCHEMA
        and value.get('draft_native_schedule') == SCHEDULE
        and value.get('draft_denoising_schedule') == [625.]
        and value.get('target_steps') == [1000.]
        and value.get('handoff_timestep') == 625.
        and value.get('draft_clean_forwards') == 1
        and value.get('history_replay_chunks') == 0
        and value.get('branch_snapshot_bytes') == 0
        and value.get('committed_kv_width') == 3,
        'low-noise bridge execution evidence differs')
    n = value.get('committed_chunks')
    tokens = value.get('tokens_per_chunk')
    require(type(n) is int and n > 0 and type(tokens) is int and tokens > 0,
            'invalid low-noise native cursor identity')
    require(value.get('forward_trace') == [
        dict(timesteps=[625.]*3,current_start=(n-1)*tokens),
        dict(timesteps=[0.]*3,current_start=(n-1)*tokens)],
        'low-noise bridge must execute exactly one 625 forward and one B3 clean forward')
    return True


def oracle(*, pipeline, torch, prompt, history, widths, noisy):
    """Diagnostic-only full-prefix replay, independent of the tentative cache."""
    p, t = pipeline, torch
    p._initialize_kv_cache(batch_size=1, dtype=t.bfloat16, device=noisy.device)
    p._initialize_crossattn_cache(batch_size=1, dtype=t.bfloat16, device=noisy.device)
    cache = dict(self=p.kv_cache1, cross=p.crossattn_cache)
    try:
        with t.inference_mode():
            cond = p.text_encoder(text_prompts=[prompt])
            for i, width in enumerate(widths):
                p.generator(noisy_image_or_video=history[:,3*i:3*i+3].to(noisy.device),
                    conditional_dict=cond, timestep=t.zeros((1,width),device=noisy.device,dtype=t.int64),
                    kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                    current_start=3*i*p.frame_seq_length)
            _, clean = p.generator(noisy_image_or_video=noisy, conditional_dict=cond,
                timestep=t.ones((1,3),device=noisy.device,dtype=t.int64)*p.denoising_step_list[-1],
                kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                current_start=3*len(widths)*p.frame_seq_length)
            p.generator(noisy_image_or_video=clean, conditional_dict=cond,
                timestep=t.zeros((1,3),device=noisy.device,dtype=t.int64),
                kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                current_start=3*len(widths)*p.frame_seq_length)
        return clean, cache
    finally:
        p.kv_cache1 = p.crossattn_cache = None


def execute(backend, *, prompt, history, widths, prediction, noise, verify_parity=False):
    """Resolve one stopped native LongLive block; never copy Target KV."""
    from .target_prefix import native_draft_schedule
    from .self_forcing_frame_adapter import StepZeroCapture, LAYER_INDICES, EFFECTIVE_RANK_LAYER
    from .self_forcing_frame_adapter import sampled_kv_cache_witness
    early = getattr(backend, '_early_draft', None)
    require(early is not None and early.pending is not None and not early.poisoned
        and backend._active_capture is None, 'low-noise handoff needs one idle pending Draft')
    require(type(verify_parity) is bool, 'invalid parity flag')
    require(isinstance(widths,tuple) and all(type(w) is int and w == 3 for w in widths),
            'LongLive low-noise bridge requires native B3 history')
    p, t = early.pipeline, backend.torch
    steps, values = native_draft_schedule(p)
    require(values == SCHEDULE, 'low-noise bridge requires the exact warped LongLive schedule')
    scope, binding = early._identity(prompt, history, widths)
    require(scope == early.pending['scope'] and binding == early.pending['binding'],
            'low-noise handoff belongs to another accepted history')
    for x in (prediction, noise):
        require(isinstance(x,t.Tensor) and tuple(x.shape) == early.pending['shape']
            and x.dtype == t.bfloat16 and bool(t.isfinite(x.float()).all().item()),
            'low-noise handoff tensor is nonfinite or has incompatible geometry')
    backend._active_capture = StepZeroCapture(layers=LAYER_INDICES,target_timestep=1000,
        sample_hidden=backend._sample_hidden,rank_sample=backend._rank_sample,
        retain_rank_layer=EFFECTIVE_RANK_LAYER,policy_rank_only=True)
    handle = None
    trace = []
    try:
        with t.inference_mode():
            clean_input, epsilon = (x.to(early.pending['device']).contiguous() for x in (prediction,noise))
            noisy = p.scheduler.add_noise(clean_input.flatten(0,1),epsilon.flatten(0,1),
                steps[-1]*t.ones((3,),device=clean_input.device,dtype=t.long)).unflatten(0,clean_input.shape[:2])
            if verify_parity:
                expected, cache = oracle(pipeline=p,torch=t,prompt=prompt,history=history,
                                         widths=widths,noisy=noisy)
            def record(_module, _args, kwargs):
                trace.append(dict(timesteps=backend._timestep_values(kwargs['timestep']),
                                  current_start=kwargs['current_start']))
            handle = p.generator.register_forward_pre_hook(record,with_kwargs=True)
            early._attach()
            early._cursor(len(binding),tentative=True)
            cond = p.text_encoder(text_prompts=[prompt])
            _, clean = p.generator(noisy_image_or_video=noisy,conditional_dict=cond,
                timestep=t.ones((1,3),device=noisy.device,dtype=t.int64)*steps[-1],
                kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                current_start=len(binding)*early.tokens_per_chunk)
            early._clean(clean,cond,len(binding),3)
            early._cursor(len(binding)+1)
            early.binding = binding+(early._chunk_binding(clean,3),)
            early.pending = None
            result = dict(schema=SCHEMA,latent=clean,target_steps=[1000.],
                draft_native_schedule=values,draft_denoising_schedule=[625.],handoff_timestep=625.,
                draft_clean_forwards=1,history_replay_chunks=0,branch_snapshot_bytes=0,
                committed_kv_width=3,committed_chunks=len(early.binding),
                tokens_per_chunk=early.tokens_per_chunk,forward_trace=trace)
            if verify_parity:
                from .prefix_oracle import check_suffix
                result['diagnostic_oracle'] = check_suffix(actual=clean,expected=expected,
                    actual_cache=early.payload,expected_cache=cache,torch=t)
            witness = sampled_kv_cache_witness(early.payload['self'],early.payload['cross'])
            result['kv_witness_sha256'], result['kv_witness'] = witness
            result['latent'] = clean.detach().to('cpu').contiguous()
            validate_evidence(result)
            return result
    except BaseException:
        early.payload = early.pending = None
        early.poisoned = True
        raise
    finally:
        if handle is not None:
            handle.remove()
        early._detach()
        backend._active_capture = None
