"""Joint Self-Forcing history prefill and first Draft denoising forward."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import json
import math
import sys
from types import MethodType

ENV = 'CFEXP_SF_JOINT_HANDOFF'
_active_handoff = ContextVar('cfexp_sf_joint_handoff', default=None)


def active_handoff():
    return _active_handoff.get()


def window_bounds(old_tokens, tokens_per_chunk, chunk_count, capacity):
    """Ranges in [old visible KV, new KV], one native window per query block."""
    if not (0 <= old_tokens <= capacity and 0 < tokens_per_chunk <= capacity
            and 0 < chunk_count * tokens_per_chunk <= capacity):
        raise ValueError('unsupported native rolling KV geometry')
    ends = [old_tokens + tokens_per_chunk * i for i in range(1, chunk_count + 1)]
    return tuple((max(0, end - capacity), end) for end in ends)


def batched_attention(self, x, seq_lens, grid_sizes, freqs, block_mask,
                      kv_cache=None, current_start=0, cache_start=None, *, keep_clean_window=False):
    """Apply native FlashAttention to each block's own contiguous KV range."""
    torch = sys.modules['torch']
    native = sys.modules['wan.modules.causal_model']
    batch, tokens, _ = x.shape
    frame_tokens = math.prod(grid_sizes[0, 1:].tolist())
    chunk_tokens = 3 * frame_tokens
    capacity = kv_cache['k'].shape[1]
    old_end = int(kv_cache['local_end_index'].item())
    if (batch != 1 or not tokens or tokens % chunk_tokens or self.sink_size != 0
            or self.local_attn_size != 21 or capacity != self.max_attention_size
            or current_start != int(kv_cache['global_end_index'].item())
            or old_end != min(current_start, capacity)):
        raise ValueError('batched handoff requires contiguous native sink-free KV')
    bounds = window_bounds(old_end, chunk_tokens, tokens // chunk_tokens, capacity)
    shape = (batch, tokens, self.num_heads, self.head_dim)
    v = self.v(x).view(shape)
    q = self.norm_q(self.q(x)).view(shape)
    k = self.norm_k(self.k(x)).view(shape)
    frame_start = current_start // frame_tokens
    q = native.causal_rope_apply(q, grid_sizes, freqs, start_frame=frame_start).type_as(v)
    k = native.causal_rope_apply(k, grid_sizes, freqs, start_frame=frame_start).type_as(v)
    all_k = torch.cat((kv_cache['k'][:, :old_end], k), dim=1)
    all_v = torch.cat((kv_cache['v'][:, :old_end], v), dim=1)
    outputs = []
    for index, (lo, hi) in enumerate(bounds):
        query = q[:, index * chunk_tokens:(index + 1) * chunk_tokens]
        outputs.append(native.attention(query, all_k[:, lo:hi], all_v[:, lo:hi]))
    # The second denoise can append its own slot to the clean prefix. Keep
    # only accepted history resident so the native clean KV witness still
    # describes exactly the same state as before the first denoising step.
    lo, hi = bounds[-2 if keep_clean_window else -1]
    kv_cache['k'][:, :hi-lo].copy_(all_k[:, lo:hi])
    kv_cache['v'][:, :hi-lo].copy_(all_v[:, lo:hi])
    committed_tokens = tokens - chunk_tokens if keep_clean_window else tokens
    kv_cache['global_end_index'].fill_(current_start + committed_tokens)
    kv_cache['local_end_index'].fill_(hi - lo)
    return self.o(torch.cat(outputs, dim=1).flatten(2))


@contextmanager
def batch_attention(model, *, keep_clean_window=False):
    layers = [block.self_attn for block in model.blocks]
    if any('forward' in vars(layer) for layer in layers):
        raise ValueError('joint handoff requires unpatched native self-attention')
    try:
        for layer in layers:
            layer.forward = MethodType(
                partial(batched_attention, keep_clean_window=keep_clean_window), layer)
        yield
    finally:
        for layer in layers:
            if 'forward' in vars(layer):
                del layer.forward


@contextmanager
def native_timestep_shapes(model, history_frames, *, mixed):
    """Embed and project each timestep at the row count of the native call."""
    torch = sys.modules['torch']
    modules = (model.time_embedding, model.time_projection)
    if history_frames < 3 or history_frames % 3:
        raise ValueError('expected complete three-frame clean history blocks')
    if any('forward' in vars(module) for module in modules):
        raise ValueError('expected unpatched native timestep modules')

    def wrap(forward):
        def apply(rows):
            if rows.ndim != 2 or rows.shape[0] != history_frames + 3 * mixed:
                raise ValueError('unexpected fused timestep shape')
            # The clean timestep is zero for every history block. Run one row
            # before broadcasting so BF16 uses the same GEMM shape as B1.
            history = forward(rows[:1]).expand(history_frames, -1)
            if not mixed:
                return history
            # The Draft's native first denoise uses all three frame timesteps.
            return torch.cat((history, forward(rows[-3:].contiguous())), dim=0)
        return apply

    try:
        for module in modules:
            module.forward = wrap(module.forward)
        yield
    finally:
        for module in modules:
            if 'forward' in vars(module):
                del module.forward


def joint_first_step(pipeline, *, torch, history, noise, conditional,
                     step, history_start, draft_start):
    """Advance one leased native cache and return only the resumed Draft block."""
    model = pipeline.generator.model
    history_frames = int(history.shape[1])
    video = torch.cat((history, noise), dim=1)
    timestep = torch.zeros((1, history_frames + 3), device=noise.device, dtype=torch.int64)
    timestep[:, -3:] = step
    handoff = (draft_start, history_frames // 3)
    token = _active_handoff.set(handoff)
    try:
        with batch_attention(model, keep_clean_window=True), native_timestep_shapes(
            model, history_frames, mixed=True,
        ):
            _, clean = pipeline.generator(
                noisy_image_or_video=video, conditional_dict=conditional,
                timestep=timestep, kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache, current_start=history_start,
            )
        return clean[:, -3:].contiguous()
    finally:
        _active_handoff.reset(token)


def verify_rollout(campaign, key, *, selected):
    """Require one physically observed joint forward at each eligible handoff."""
    from .self_forcing_frame_adapter import (
        SELF_FORCING_DENOISING_SCHEDULE, SELF_FORCING_TOKENS_PER_CHUNK,
        validate_self_forcing_forward_trace,
    )

    target = {0, *selected}
    handoffs = []
    for index in range(campaign.chunk_count):
        if index in target:
            continue
        checkpoint = campaign._checkpoint_path('rollouts', key, index)
        diagnostics = json.loads(checkpoint.read_bytes())['generated']['diagnostics']
        trace = diagnostics['forward_trace']
        reused = diagnostics['kv_reuse']['reused_chunks']
        start = index
        while start - 1 in target:
            start -= 1
        # An opening Target run has no reusable Draft prefix.
        expected = index - start if start > 0 else 0
        actual = trace[0].get('fused_history_chunks', 0)
        if actual != expected:
            raise ValueError('physical joint forwards differ from the accepted Target bursts')
        if actual:
            entry = trace[0]
            if reused != start or entry != {
                'timestep': SELF_FORCING_DENOISING_SCHEDULE[0],
                'batch_timesteps_equal': False,
                'current_start': start * SELF_FORCING_TOKENS_PER_CHUNK,
                'fused_history_chunks': actual,
            }:
                raise ValueError('joint forward does not bind the exact leased prefix')
            native_first = dict(entry, current_start=index * SELF_FORCING_TOKENS_PER_CHUNK,
                                batch_timesteps_equal=True)
            del native_first['fused_history_chunks']
            validate_self_forcing_forward_trace([native_first, *trace[1:]],
                history_chunks=index, reused_history_chunks=index)
            handoffs.append({'draft_chunk': index, 'target_history_chunks': actual,
                             'physical_joint_forwards': 1, 'forward_calls_saved': actual})
        else:
            validate_self_forcing_forward_trace(trace,
                history_chunks=index, reused_history_chunks=reused)
    if not handoffs:
        raise ValueError('joint handoff run did not execute a joint forward')
    return {'handoffs': handoffs, 'forward_calls_saved': sum(row['forward_calls_saved'] for row in handoffs)}
