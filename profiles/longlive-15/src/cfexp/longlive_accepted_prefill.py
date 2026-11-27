"""Opt-in native LongLive clean updates on the idle Draft worker."""
from __future__ import annotations

import math
import os
import time

from .accepted_prefill import require

ENV = 'CFEXP_LONGLIVE_ASYNC_ACCEPTED_PREFILL'


def enabled():
    value = os.environ.get(ENV, '0')
    require(value in ('0', '1'), 'invalid LongLive accepted prefill flag')
    return value == '1'


def native_prefill(backend, *, run_dir, prompt, history_latents, history_kv_widths):
    """Update the native B3 prefix without denoising, sampling, or allocation."""
    from .longlive_policy_backend import LongLivePolicyBackend
    from .self_forcing_frame_adapter import EFFECTIVE_RANK_LAYER, LAYER_INDICES, StepZeroCapture

    require(isinstance(backend, LongLivePolicyBackend) and backend._active_capture is None,
            'LongLive prefill requires the idle native policy model')
    torch = backend.torch
    count = len(history_kv_widths)
    require(isinstance(history_latents, torch.Tensor) and history_latents.device.type == 'cpu'
            and history_latents.dtype == torch.bfloat16 and not history_latents.requires_grad
            and tuple(history_latents.shape) == (1, count * 3, 16, 60, 104)
            and history_kv_widths == (3,) * count,
            'LongLive prefill requires the bound native BF16 CPU history')
    trace = []

    def before(_module, _args, kwargs):
        timestep = backend._timestep_values(kwargs['timestep'])
        require(len(timestep) == 3 and all(v == 0 for v in timestep),
                'LongLive prefill attempted a non-clean forward')
        trace.append({'timestep': 0., 'batch_timesteps_equal': True,
                      'current_start': kwargs['current_start']})

    started = time.monotonic()
    with torch.cuda.device(backend.device), torch.inference_mode():
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(backend.device)
        history = history_latents.to(backend.device)
        capture = StepZeroCapture(
            layers=LAYER_INDICES, target_timestep=1000,
            sample_hidden=backend._sample_hidden, rank_sample=backend._rank_sample,
            retain_rank_layer=EFFECTIVE_RANK_LAYER, policy_rank_only=True,
        )
        hook = backend.pipeline.generator.register_forward_pre_hook(before, with_kwargs=True)
        # The policy hooks require a capture scope even on a clean KV update.
        backend._active_capture = capture
        try:
            result = backend.session.prefill_accepted(
                prompt=prompt, history=history, history_digest_source=history_latents)
        finally:
            backend._active_capture = None
            hook.remove()
        torch.cuda.current_stream(backend.device).synchronize()
        require(not capture.events, 'LongLive clean prefill captured a Draft proposal')
        require(torch.equal(cpu_rng, torch.get_rng_state())
                and torch.equal(cuda_rng, torch.cuda.get_rng_state(backend.device)),
                'LongLive clean prefix update consumed RNG')
    require(trace == [{'timestep': 0., 'batch_timesteps_equal': True,
                       'current_start': (count - 1) * backend.session.executor.tokens_per_chunk}],
            'LongLive prefill did not perform exactly one native clean forward')
    return dict(result, forward_trace=trace, started_monotonic=started,
                completed_monotonic=time.monotonic(), cuda_stream_completed=True)


def audit_handoff(run_dir, history, diagnostics, *, expected, target_interval=None):
    """Bind a finished update to the exact accepted input and its next consumer."""
    from .accepted_prefill import object_sha
    from .accepted_prefill_audit import local_receipt_reader
    from .correction_state import LatentHistory

    evidence = diagnostics['accepted_prefill_execution']
    reuse = diagnostics['kv_reuse']
    trace = diagnostics['forward_trace']
    index = history.next_chunk_index
    bindings = evidence['receipts']
    require(evidence['schema'] == 'accepted_prefill_generation_v1'
            and evidence['chunk_index'] == index
            and evidence['prompt_sha256'] == reuse['prompt_sha256']
            and evidence['actual_generate_trace_sha256'] == object_sha(trace)
            and evidence['prefill_count'] == len(bindings) == expected,
            'LongLive accepted-prefill consumer identity differs')
    if not expected:
        require(evidence['canonical_native_trace_sha256'] == object_sha(trace),
                'LongLive ordinary Draft operation trace differs')
        return []
    binding, = bindings
    receipt = local_receipt_reader(run_dir)(binding)
    prefix = history.prefix_before(index - 1)
    ident = object_sha({'request_id': receipt['request_id'],
        'history_manifest_sha256': prefix.manifest_sha256, 'next_draft_chunk_index': index,
        'prompt_sha256': evidence['prompt_sha256']})
    require(receipt['schema'] == 'accepted_target_prefill_receipt_v1'
            and receipt['prefill_id'] == ident
            and binding['path'] == 'provenance/accepted-target-prefill/' + ident + '.json'
            and receipt['prompt_sha256'] == evidence['prompt_sha256']
            and receipt['next_draft_chunk_index'] == index
            and LatentHistory.from_dict(receipt['history']) == prefix
            and all(s.producer_role == 'target' for s in history.segments[-2:]),
            'LongLive prefill receipt differs from the accepted Target prefix')
    native = receipt['native']
    require(native['schema'] == 'native_accepted_prefix_prefill_v1'
            and native['next_chunk_index'] == index - 1
            and native['reused_chunks'] == index - 2 and native['replayed_chunks'] == 1
            and native['history_binding'] == reuse['history_tensor_sha256'][:-1]
            and native['history_kv_widths'] == reuse['history_kv_widths'][:-1] == [3] * (index - 1)
            and native['scope_sha256'] == reuse['scope_sha256']
            and native['cuda_stream_completed'] is True
            and native['native_payload_reused'] is True
            and native['noise_created'] is False
            and native['branch_snapshot_clone_bytes'] == 0
            and native['scheduler_updates'] == native['generated_chunks'] == 0
            and evidence['canonical_native_trace_sha256'] == object_sha(native['forward_trace'] + trace),
            'LongLive native prefix or operation trace differs')
    start, end = native['started_monotonic'], native['completed_monotonic']
    require(isinstance(target_interval, list) and len(target_interval) == 2,
            'LongLive prefill is missing the corresponding Target RPC interval')
    target_start, target_end = target_interval
    require(all(type(t) in (int, float) and math.isfinite(t)
                for t in (start, end, target_start, target_end))
            and 0 < start <= end and 0 < target_start <= target_end,
            'LongLive prefill timing interval differs')
    return [{'receipt': binding, 'target_rpc_interval': target_interval, 'overlapped_target_rpc_seconds':
             max(0., min(end, target_end) - max(start, target_start))}]
