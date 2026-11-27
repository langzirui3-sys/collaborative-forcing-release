"""Diagnostic native LongLive B3 prefill + first denoise fusion.

This is deliberately separate from Self-Forcing's sink-free implementation.
Native LongLive commits the current noisy slot through deferred model-level
updates. A joint forward must do the same, while each query block sees only
its own sink-three/local-twelve window. CUDA parity is required before any
production backend may use this primitive.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import math
import os
import sys
from types import MethodType

ENV = "CFEXP_LONGLIVE_JOINT_HANDOFF_DIAGNOSTIC"
_active = ContextVar("cfexp_longlive_joint_handoff", default=None)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def active_handoff():
    return _active.get()


def joint_attention(self, x, seq_lens, grid_sizes, freqs, block_mask,
                    kv_cache=None, current_start=0, cache_start=None,
                    sink_recache_after_switch=False, *, clean_observer=None):
    """Return native deferred updates; never mutate the borrowed KV directly."""
    torch = sys.modules["torch"]
    native = sys.modules["wan.modules.causal_model"]
    require(kv_cache is not None and x.ndim == 3 and grid_sizes.shape == (1, 3),
            "joint LongLive requires one cached native sequence")
    batch, tokens, _ = x.shape
    frame_tokens = math.prod(grid_sizes[0, 1:].tolist())
    chunk_tokens = 3 * frame_tokens
    capacity = int(kv_cache["k"].shape[1])
    global_end = int(kv_cache["global_end_index"].item())
    local_end = int(kv_cache["local_end_index"].item())
    require(batch == 1 and 2 * chunk_tokens <= tokens <= 5 * chunk_tokens
            and tokens % chunk_tokens == 0 and int(grid_sizes[0, 0]) * frame_tokens == tokens
            and self.local_attn_size == 12 and self.sink_size == 3
            and capacity == self.max_attention_size == 12 * frame_tokens
            and kv_cache["k"].shape == kv_cache["v"].shape
            and 0 < current_start == global_end and current_start % chunk_tokens == 0
            and local_end == min(global_end, capacity)
            and cache_start in (None, current_start) and not sink_recache_after_switch,
            "joint LongLive requires contiguous native local12/sink3/B3 state")
    shape = (batch, tokens, self.num_heads, self.head_dim)
    q = self.norm_q(self.q(x)).view(shape)
    k = self.norm_k(self.k(x)).view(shape)
    v = self.v(x).view(shape)
    q = native.causal_rope_apply(q, grid_sizes, freqs,
                                start_frame=current_start // frame_tokens).type_as(v)
    k = native.causal_rope_apply(k, grid_sizes, freqs,
                                start_frame=current_start // frame_tokens).type_as(v)
    temp_k, temp_v = kv_cache["k"].clone(), kv_cache["v"].clone()
    outputs = []
    groups = tokens // chunk_tokens
    for group in range(groups):
        lo, hi = group * chunk_tokens, (group + 1) * chunk_tokens
        # Each group advances exactly one native B3 append. This is not a
        # single large-window attention, which would expose future blocks.
        evicted = max(0, local_end + chunk_tokens - capacity)
        if evicted:
            rolled = local_end - evicted - chunk_tokens
            require(rolled >= 0, "joint LongLive would evict its sink")
            temp_k[:, chunk_tokens:chunk_tokens + rolled] = temp_k[
                :, chunk_tokens + evicted:chunk_tokens + evicted + rolled].clone()
            temp_v[:, chunk_tokens:chunk_tokens + rolled] = temp_v[
                :, chunk_tokens + evicted:chunk_tokens + evicted + rolled].clone()
        local_end += chunk_tokens - evicted
        temp_k[:, local_end - chunk_tokens:local_end] = k[:, lo:hi]
        temp_v[:, local_end - chunk_tokens:local_end] = v[:, lo:hi]
        window_start = max(chunk_tokens, local_end - (capacity - chunk_tokens))
        keys = torch.cat((temp_k[:, :chunk_tokens], temp_k[:, window_start:local_end]), dim=1)
        values = torch.cat((temp_v[:, :chunk_tokens], temp_v[:, window_start:local_end]), dim=1)
        outputs.append(native.attention(q[:, lo:hi], keys, values))
        if clean_observer is not None and group == groups - 2:
            # Diagnostic only: the observer must own its copy before the next
            # group overwrites the temporary cache. Production pays no copy.
            clean_observer(temp_k, temp_v, current_start + hi, local_end)
    current_end = current_start + tokens
    update = dict(action="direct_insert", local_start_index=0,
                  local_end_index=local_end, write_start_index=0,
                  write_end_index=local_end, new_k=temp_k[:, :local_end],
                  new_v=temp_v[:, :local_end], current_end=current_end,
                  is_recompute=False)
    return self.o(torch.cat(outputs, dim=1).flatten(2)), (current_end, local_end, update)


@contextmanager
def native_attention(model, *, clean_observer=None):
    layers = [block.self_attn for block in model.blocks]
    require(len(layers) == 30 and all("forward" not in vars(layer) for layer in layers),
            "joint LongLive requires thirty unpatched native attention layers")
    try:
        for index, layer in enumerate(layers):
            observer = None if clean_observer is None else partial(clean_observer, index)
            layer.forward = MethodType(partial(joint_attention, clean_observer=observer), layer)
        yield
    finally:
        for layer in layers:
            if "forward" in vars(layer):
                del layer.forward


@contextmanager
def native_timestep_shapes(model, history_frames):
    """Keep LongLive's native three-row clean and noisy timestep GEMMs."""
    torch = sys.modules["torch"]
    modules = (model.time_embedding, model.time_projection)
    require(3 <= history_frames <= 12 and history_frames % 3 == 0
            and all("forward" not in vars(module) for module in modules),
            "joint LongLive requires one to four B3 clean groups and native time modules")

    def wrap(forward):
        def apply(rows):
            require(rows.ndim == 2 and rows.shape[0] == history_frames + 3,
                    "unexpected joint LongLive timestep shape")
            first = rows[:3].contiguous()
            require(torch.equal(rows[:history_frames], first.repeat(history_frames // 3, 1)),
                    "joint clean groups have different timestep embeddings")
            clean = forward(first).repeat(history_frames // 3, 1)
            noisy = forward(rows[-3:].contiguous())
            return torch.cat((clean, noisy), dim=0)
        return apply

    try:
        for module in modules:
            module.forward = wrap(module.forward)
        yield
    finally:
        for module in modules:
            if "forward" in vars(module):
                del module.forward


@contextmanager
def native_ffn_output_shapes(model, *, chunk_tokens, groups):
    """Keep the final FFN projection at its native B3 GEMM geometry.

    The trained CUDA trace first diverged at blocks.0.ffn.2 for four clean
    groups. Each call below evaluates that projection at the ordinary B3
    shape; it neither substitutes reference tensors nor changes attention.
    This adds projection launches, so its speed must be measured separately.
    """
    torch = sys.modules["torch"]
    require(type(chunk_tokens) is int and chunk_tokens > 0
            and type(groups) is int and 2 <= groups <= 5
            and len(model.blocks) == 30,
            "joint LongLive requires native B3 FFN geometry")
    modules = tuple(block.ffn[2] for block in model.blocks)
    require(len({id(module) for module in modules}) == 30
            and all(native_output_projection(module, torch=torch) for module in modules),
            "joint LongLive requires thirty unpatched native FFN output projections")

    def wrap(forward):
        def apply(rows):
            require(rows.ndim == 3 and rows.shape[0] == 1
                    and rows.shape[1] == chunk_tokens * groups,
                    "unexpected joint LongLive FFN output input shape")
            return torch.cat([forward(rows[:, lo:lo+chunk_tokens].contiguous())
                              for lo in range(0, chunk_tokens * groups, chunk_tokens)], dim=1)
        return apply

    try:
        for module in modules:
            module.forward = wrap(module.forward)
        yield
    finally:
        for module in modules:
            if "forward" in vars(module):
                del module.forward


def native_output_projection(module, *, torch):
    """Accept the pinned upstream plain or unmerged rank256 PEFT projection.

    LongLive's official configure_lora_for_model wraps all Linear layers in
    each CausalWanAttentionBlock, including ffn.2. Calling its bound original
    forward preserves the base projection AND loaded adapter; extracting or
    merging base_layer would silently change the native computation.
    """
    if 'forward' in vars(module):
        return False
    if type(module) is torch.nn.Linear:
        return True
    try:
        from peft.tuners.lora.layer import Linear
    except ImportError:
        return False
    if (type(module) is not Linear or type(module.base_layer) is not torch.nn.Linear
            or module.merged or module.disable_adapters or module.fan_in_fan_out
            or list(module.active_adapters) != ['default']
            or module.r != {'default':256} or module.lora_alpha != {'default':256}
            or module.scaling != {'default':1.0}
            or module.use_dora != {'default':False}
            or set(module.lora_A) != {'default'} or set(module.lora_B) != {'default'}
            or set(module.lora_dropout) != {'default'}):
        return False
    base, a, b, dropout = (module.base_layer, module.lora_A['default'],
                          module.lora_B['default'], module.lora_dropout['default'])
    return (type(a) is torch.nn.Linear and type(b) is torch.nn.Linear
            and all('forward' not in vars(x) for x in (base,a,b))
            and a.bias is None and b.bias is None
            and tuple(a.weight.shape) == (256,base.in_features)
            and tuple(b.weight.shape) == (base.out_features,256)
            and a.weight.dtype == b.weight.dtype == base.weight.dtype
            and a.weight.device == b.weight.device == base.weight.device
            and (type(dropout) is torch.nn.Identity
                 or (type(dropout) is torch.nn.Dropout and dropout.p == 0)))


def joint_first_step(pipeline, *, torch, history, noise, conditional,
                     history_start, draft_start, clean_observer=None):
    """Diagnostic call only; ordinary LongLive execution remains unchanged."""
    require(os.environ.get(ENV) == "1", "native LongLive fusion requires its explicit diagnostic flag")
    require(not torch.is_grad_enabled(), "joint LongLive is inference only")
    frames = int(history.shape[1])
    frame_tokens = int(pipeline.frame_seq_length)
    require(history.ndim == noise.ndim == 5 and history.shape[0] == noise.shape[0] == 1
            and tuple(history.shape[2:]) == tuple(noise.shape[2:]) and noise.shape[1] == 3
            and history.dtype == noise.dtype and history.device == noise.device
            and 3 <= frames <= 12 and frames % 3 == 0
            and type(history_start) is int and type(draft_start) is int
            and history_start > 0 and history_start % (3 * frame_tokens) == 0
            and draft_start == history_start + frames * frame_tokens,
            "joint LongLive history/noise geometry or absolute position differs")
    model = pipeline.generator.model
    video = torch.cat((history, noise), dim=1)
    timestep = torch.zeros((1, frames + 3), device=noise.device, dtype=torch.int64)
    timestep[:, -3:] = 1000
    token = _active.set((draft_start, frames // 3))
    try:
        with native_attention(model, clean_observer=clean_observer), \
                native_timestep_shapes(model, frames), \
                native_ffn_output_shapes(model, chunk_tokens=3*frame_tokens, groups=frames//3+1):
            _, clean = pipeline.generator(noisy_image_or_video=video,
                conditional_dict=conditional, timestep=timestep,
                kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                current_start=history_start)
        return clean[:, -3:].contiguous()
    finally:
        _active.reset(token)
