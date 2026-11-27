"""Opt-in accepted-history assimilation; no proposals or changed model math.

The native prefix primitive is separate from its eventual worker scheduling.
Nothing in this module enables itself in an existing generation campaign.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ENV='CFEXP_ACCEPTED_TARGET_PREFILL'
BURST_LENGTH_ENV = 'CFEXP_ACCEPTED_TARGET_PREFILL_BURST_LENGTH'


def enabled():
    value=os.environ.get(ENV,'0')
    if value not in ('0','1'):
        raise ValueError('accepted Target prefill flag must be 0 or 1')
    return value=='1'


def burst_length(env=None):
    value = (os.environ if env is None else env).get(BURST_LENGTH_ENV, '4')
    if value not in ('2', '4'):
        raise ValueError('Self-Forcing accepted prefill burst length must be 2 or 4')
    return int(value)


def object_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,allow_nan=False,
                                    sort_keys=True,separators=(',',':')).encode()).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def native_prefill(backend, *, run_dir, prompt, history_latents, history_kv_widths):
    """Run only the exact missing clean forwards on an existing native prefix.

    The caller must own the worker's model lock. Stream completion is explicit
    before returning, so the next owner can safely run on another CPU thread.
    All actual model calls and RNG/stream/helper witnesses are returned for an
    immutable worker receipt; this function writes no generation artifacts.
    """
    from .self_forcing_frame_adapter import (OfficialSelfForcingBackend,StepZeroCapture,
        LAYER_INDICES,EFFECTIVE_RANK_LAYER,attach_adapter_timing,timed_forward_stages)
    from .incremental_draft import IncrementalDraftExecutor
    from .rope_coefficients import RopeCoefficientCache
    from .full_length_attention import FullLengthAttention
    from .rope_worker import validate_execution as validate_rope,validate_frequency
    from .full_attention_worker import validate_execution as validate_attention
    require(type(backend) is OfficialSelfForcingBackend
        and type(backend._incremental_draft) is IncrementalDraftExecutor
        and getattr(backend,'_cpu_history_binding_installed',False)
        and getattr(backend,'_rope_coefficients_installed',False)
        and getattr(backend,'_full_length_attention_installed',False),
        'accepted prefill requires the probed native SF execution path')
    require(backend._active_capture is None,'accepted prefill cannot interrupt another native call')
    torch=backend.torch;p=backend.pipeline;model=p.generator.model
    require(model.training is False,'accepted prefill requires the native evaluation model')
    require(isinstance(history_latents,torch.Tensor) and history_latents.device.type=='cpu'
        and history_latents.dtype==torch.bfloat16 and not history_latents.requires_grad
        and tuple(history_latents.shape[:1])==(1,) and history_latents.ndim==5
        and tuple(history_latents.shape[2:])==(16,60,104),
        'accepted prefill requires worker-owned native BF16 CPU history')
    causal=sys.modules['wan.modules.causal_model'];attention=sys.modules['wan.modules.attention']
    rope=causal.causal_rope_apply;fast=attention.flash_attention
    require(type(rope) is RopeCoefficientCache and type(fast) is FullLengthAttention
        and sys.modules['wan.modules.model'].flash_attention is fast,
        'accepted prefill native helper ownership changed')
    root=Path(run_dir)
    def proof(name):
        value=json.loads((root/'provenance'/name).read_text());sha=value.pop('sha256')
        require(hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()==sha,
                'accepted prefill execution proof changed')
        return value,sha
    _,rope_gate=proof('native-rope-probe-gate.json')
    _,attention_gate=proof('native-full-attention-probe-gate.json')
    frequency,frequency_sha=proof('native-rope-frequency-initialization.json');validate_frequency(frequency)
    require(rope._source is model.freqs and model.freqs._version==frequency['prepared_version'],
        'accepted prefill frequency binding changed')
    before_rope=rope.evidence();before_attention=fast.evidence()
    trace=[];audit=None
    started=time.monotonic()
    with torch.cuda.device(backend.device):
        attach_adapter_timing(torch,backend.device)
        cpu_rng=torch.random.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state(backend.device).clone()
        history=history_latents.to(device=backend.device,dtype=torch.bfloat16)
        backend._active_capture=StepZeroCapture(layers=LAYER_INDICES,target_timestep=1000,
            sample_hidden=backend._sample_hidden,rank_sample=backend._rank_sample,
            retain_rank_layer=EFFECTIVE_RANK_LAYER,policy_rank_only=True)
        def before(_module,_args,kwargs):
            values=backend._timestep_values(kwargs['timestep']);position=kwargs['current_start']
            require(values and all(v==0. for v in values) and type(position) is int,
                    'accepted prefill attempted a denoising forward')
            trace.append({'timestep':values[0],'batch_timesteps_equal':True,'current_start':position})
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        try:
            audit=p.generator.register_forward_pre_hook(before,prepend=False,with_kwargs=True)
            begin.record(torch.cuda.current_stream(backend.device))
            result=backend._incremental_draft.prefill_accepted(prompt=prompt,
                history=history,history_kv_widths=history_kv_widths,
                history_digest_source=history_latents,
                forward_context=lambda count:timed_forward_stages(p.generator,
                    prefill_count=count,denoise_count=0,cache_update_count=0))
            end.record(torch.cuda.current_stream(backend.device));end.synchronize()
            require(not backend._active_capture.events,'accepted prefill captured a Draft proposal')
            require(torch.equal(cpu_rng,torch.random.get_rng_state())
                and torch.equal(cuda_rng,torch.cuda.get_rng_state(backend.device)),
                'native clean prefill unexpectedly consumed Torch RNG')
            require(model.freqs is rope._source and model.freqs._version==frequency['prepared_version'],
                    'accepted prefill changed the bound frequency source')
        finally:
            if audit is not None:audit.remove()
            backend._active_capture=None
        stream_ms=begin.elapsed_time(end)
    require(result['replayed_chunks']>0 and len(trace)==result['replayed_chunks']
        and trace==[{'timestep':0.,'batch_timesteps_equal':True,
                    'current_start':i*backend._incremental_draft.tokens_per_chunk}
                   for i in range(result['reused_chunks'],result['next_chunk_index'])],
            'accepted prefill native forward coverage differs')
    after_rope=rope.evidence();after_attention=fast.evidence()
    rope_evidence={'schema':'native_rope_execution_v2','probe_gate_sha256':rope_gate,
        'frequency_initialization_sha256':frequency_sha,'helper_sha256':
            'e63063c9ce2f74208e98482838513941288de16f6b8520a511826b5919d1db1e',
        'native_float64_complex128_math':True,
        'counters':{k:after_rope[k]-before_rope[k] for k in ('calls','hits','misses','fallbacks')},
        'peak_coefficient_bytes':after_rope['peak_coefficient_bytes'],'bound_bytes':after_rope['bound_bytes']}
    attention_evidence={'schema':'native_full_attention_execution_v1','probe_gate_sha256':attention_gate,
        'helper_sha256':'93e031d8da9c7dfe7ea0554a571d1b90e7562b41760ff08ee42c1d8b2c1a4dc7',
        'native_attention_parameters_and_kv_slices_preserved':True,
        'counters':{k:after_attention[k]-before_attention[k] for k in before_attention}}
    validate_rope(rope_evidence);validate_attention(attention_evidence)
    return dict(result,forward_trace=trace,torch_cpu_rng_preserved=True,torch_cuda_rng_preserved=True,
        cuda_stream_completed=True,started_monotonic=started,completed_monotonic=time.monotonic(),
        cuda_stream_elapsed_ms_not_kernel_busy=stream_ms,
        rope_execution_evidence=rope_evidence,full_attention_execution_evidence=attention_evidence)
