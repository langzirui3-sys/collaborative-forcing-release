# SPDX-License-Identifier: Apache-2.0
# Adapted from thu-ml/Causal-Forcing; see docs/licenses/causal-forcing/NOTICE.md.
"""Isolated Rolling clean-history/first-window primitive, not a production path.

Unlike LongLive, native Rolling mutates the first B3 cache slot on a noisy
forward, keeps its global anchor unrotated, and rotates that anchor differently
for clean updates and noisy windows. Every group below uses that native rule.
No ordinary session or factory imports or selects this diagnostic module.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import os
import sys
from types import MethodType

ENV = 'CFEXP_ROLLING_JOINT_HANDOFF_DIAGNOSTIC'
SCHEDULE = (1000., 937.5, 833.3333129882812, 625.)
_active = ContextVar('cfexp_rolling_joint_handoff', default=None)


def require(value, message):
    if not value:
        raise ValueError(message)


def active_handoff():
    return _active.get()


def joint_attention(self, x, seq_lens, grid_sizes, freqs, block_mask,
                    kv_cache=None, current_start=0, cache_start=None,
                    updating_cache=False, *, clean_observer=None):
    """Execute 1--4 accepted clean groups and one noisy B3 group together.

    The final cache includes the noisy first-window slot, just as the upstream
    noisy forward does. Its following ordinary clean update must still run.
    Observers receive the temporary accepted-history state before that slot.
    A runtime failure invalidates this owner; callers must discard it rather
    than retrying on partially modified caches.
    """
    t = sys.modules['torch']
    native = sys.modules['wan.modules.causal_model']
    require(os.environ.get(ENV) == '1' and not t.is_grad_enabled(),
            'Rolling joint primitive requires explicit inference diagnostic opt-in')
    require(kv_cache is not None and x.ndim == 3 and x.shape[0] == 1
            and tuple(grid_sizes.shape) == (1, 3) and not updating_cache
            and cache_start in (None, current_start), 'unsupported joint Rolling call')
    frames, height, width = (int(v) for v in grid_sizes[0].tolist())
    frame_tokens = height*width; block = 3*frame_tokens; tokens = int(x.shape[1])
    require(frame_tokens > 0 and frames*frame_tokens == tokens
            and tokens % block == 0 and 2 <= tokens//block <= 5
            and self.frame_length == frame_tokens and self.block_length == block
            and self.max_attention_size == 21*frame_tokens,
            'joint Rolling requires native frame geometry and a clean/noisy B3 partition')
    k_cache, v_cache = kv_cache['k'], kv_cache['v']
    require(k_cache.ndim == 4 and k_cache.shape == v_cache.shape
            and k_cache.shape[0] == 1 and k_cache.shape[2:] == (self.num_heads, self.head_dim)
            and k_cache.dtype == v_cache.dtype == x.dtype
            and k_cache.device == v_cache.device == x.device
            and k_cache.shape[1] >= 2*block and k_cache.shape[1] % block == 0,
            'joint Rolling cache geometry/precision/device differs')
    for name in ('global_end_index', 'local_end_index'):
        cursor = kv_cache[name]
        require(isinstance(cursor, t.Tensor) and cursor.dtype == t.int64
                and tuple(cursor.shape) == (1,) and not cursor.requires_grad,
                'joint Rolling requires native scalar tensor cursors')
    capacity = int(k_cache.shape[1])
    global_end = int(kv_cache['global_end_index'].item())
    local_end = int(kv_cache['local_end_index'].item())
    require(type(current_start) is int and current_start > 0 and current_start % block == 0
            and global_end == current_start and local_end == min(global_end, capacity)
            and (clean_observer is None or callable(clean_observer)),
            'joint Rolling requires a contiguous owned accepted prefix')
    shape = (1, tokens, self.num_heads, self.head_dim)
    # Keep native B3 projection geometry. A single enlarged CPU GEMM already
    # changed populated K values in the differential test. Model-level fusion
    # is not permission to change those numerical kernels or its cache state.
    groups_x = tuple(x[:, lo:lo+block] for lo in range(0, tokens, block))
    q = t.cat([self.norm_q(self.q(value)) for value in groups_x], dim=1).view(shape)
    k = t.cat([self.norm_k(self.k(value)) for value in groups_x], dim=1).view(shape)
    v = t.cat([self.v(value) for value in groups_x], dim=1).view(shape)
    q = native.causal_rope_apply(q, grid_sizes, freqs,
                                start_frame=current_start//frame_tokens).type_as(v)
    k = native.causal_rope_apply(k, grid_sizes, freqs,
                                start_frame=current_start//frame_tokens).type_as(v)
    grid_one = grid_sizes.clone(); grid_one[:, 0] = 3
    outputs = []; groups = tokens//block
    for group in range(groups):
        lo, hi = group*block, (group+1)*block
        absolute_start = current_start+lo
        # Each native append evicts only working state; the first B3 anchor
        # remains raw and resident even after many bounded-cache evictions.
        evicted = max(0, local_end+block-capacity)
        if evicted:
            rolled = local_end-evicted-block
            require(rolled >= 0, 'joint Rolling would overwrite the raw anchor')
            k_cache[:, block:block+rolled] = k_cache[:, block+evicted:block+evicted+rolled].clone()
            v_cache[:, block:block+rolled] = v_cache[:, block+evicted:block+evicted+rolled].clone()
        local_end += block-evicted
        local_start = local_end-block
        k_cache[:, local_start:local_end] = k[:, lo:hi]
        v_cache[:, local_start:local_end] = v[:, lo:hi]
        kv_cache['global_end_index'].fill_(absolute_start+block)
        kv_cache['local_end_index'].fill_(local_end)
        if group < groups-1:
            # Upstream updating_cache=True: current clean slot is included,
            # and the raw first anchor is rotated at absolute frame zero.
            begin = max(0, local_end-self.max_attention_size)
            keys = k_cache[:, begin:local_end].clone()
            values = v_cache[:, begin:local_end]
            if begin == 0:
                keys[:, :block] = native.causal_rope_apply(
                    keys[:, :block], grid_one, freqs, start_frame=0).type_as(v)
        else:
            # Upstream updating_cache=False: current noisy K/V is separate
            # from the working prefix; rebase the raw anchor to the native
            # past-frame position, NOT to absolute zero or to its cached RoPE.
            working_limit = self.max_attention_size-block-block
            begin = max(block, local_start-working_limit)
            working_k = k_cache[:, begin:local_start]
            working_v = v_cache[:, begin:local_start]
            anchor_frame = absolute_start//frame_tokens-working_k.shape[1]//frame_tokens-3
            anchor = native.causal_rope_apply(k_cache[:, :block], grid_one, freqs,
                                               start_frame=anchor_frame).type_as(v)
            keys = t.cat((anchor, working_k, k[:, lo:hi]), dim=1)
            values = t.cat((v_cache[:, :block], working_v, v[:, lo:hi]), dim=1)
        outputs.append(native.attention(q[:, lo:hi], keys, values))
        if clean_observer is not None and group == groups-2:
            # The observer must take its own samples before the noisy group
            # mutates this store. No full cache snapshot is owned here.
            clean_observer(k_cache, v_cache, absolute_start+block, local_end)
    return t.cat([self.o(value.flatten(2)) for value in outputs], dim=1)


@contextmanager
def native_attention(model, *, clean_observer=None):
    require(os.environ.get(ENV) == '1', 'joint Rolling attention is diagnostic only')
    layers = tuple(block.self_attn for block in model.blocks)
    require(len(layers) == 30 and len({id(x) for x in layers}) == 30
            and all('forward' not in vars(x) for x in layers)
            and (clean_observer is None or callable(clean_observer)),
            'joint Rolling requires thirty unpatched owned attention layers')
    try:
        for index, layer in enumerate(layers):
            observer = None if clean_observer is None else partial(clean_observer, index)
            layer.forward = MethodType(partial(joint_attention, clean_observer=observer), layer)
        yield
    finally:
        for layer in layers:
            if 'forward' in vars(layer):
                del layer.forward


@contextmanager
def native_timestep_shapes(model, history_frames):
    """Evaluate each clean and noisy timestep with the original B3 row count."""
    t = sys.modules['torch']; modules = (model.time_embedding, model.time_projection)
    require(history_frames in (3, 6, 9, 12)
            and all('forward' not in vars(x) for x in modules),
            'joint Rolling requires native B3 timestep modules')
    def wrap(forward):
        def apply(rows):
            require(rows.ndim == 2 and rows.shape[0] == history_frames+3
                    and t.equal(rows[:history_frames], rows[:3].repeat(history_frames//3, 1)),
                    'joint Rolling clean timestep rows differ')
            return t.cat((forward(rows[:3].contiguous()).repeat(history_frames//3, 1),
                          forward(rows[-3:].contiguous())), dim=0)
        return apply
    try:
        for module in modules: module.forward = wrap(module.forward)
        yield
    finally:
        for module in modules:
            if 'forward' in vars(module): del module.forward


def joint_first_window(pipeline, *, torch, history, noise, conditional,
                       history_start, draft_start, clean_observer=None):
    """Only the initial noisy B3 forward; caller retains native window updates.

    This primitive is deliberately not wired into RollingForcingStream. Its
    re-noising draws, clean warmup writes, drains and policy observer require
    a separate integrated trained-model gate before any production selection.
    """
    require(os.environ.get(ENV) == '1' and not torch.is_grad_enabled(),
            'joint Rolling first window requires explicit inference diagnostic opt-in')
    require(tuple(float(x) for x in pipeline.denoising_step_list) == SCHEDULE
            and pipeline.num_frame_per_block == 3 and not pipeline.independent_first_frame
            and pipeline.args.context_noise == 0, 'joint Rolling native schedule/config differs')
    require(history.ndim == noise.ndim == 5 and history.shape[0] == noise.shape[0] == 1
            and history.shape[1] in (3, 6, 9, 12) and noise.shape[1] == 3
            and history.shape[2:] == noise.shape[2:] and history.dtype == noise.dtype
            and history.device == noise.device and history.dtype == torch.bfloat16,
            'joint Rolling requires one to four accepted B3 blocks and one BF16 noise block')
    block_tokens = 3*pipeline.frame_seq_length
    require(type(history_start) is int and history_start > 0 and history_start % block_tokens == 0
            and type(draft_start) is int
            and draft_start == history_start+history.shape[1]*pipeline.frame_seq_length,
            'joint Rolling logical positions differ')
    caches = pipeline.kv_cache_clean
    require(isinstance(caches, list) and len(caches) == 30 and all(
        int(x['global_end_index'].item()) == history_start
        and int(x['local_end_index'].item()) == min(history_start, x['k'].shape[1]) for x in caches),
        'joint Rolling does not own the declared retained prefix')
    video = torch.cat((history, noise), dim=1)
    timestep = torch.zeros((1, video.shape[1]), device=noise.device, dtype=torch.float32)
    timestep[:, -3:] = 1000.
    token = _active.set((draft_start, history.shape[1]//3))
    try:
        with native_attention(pipeline.generator.model, clean_observer=clean_observer), \
                native_timestep_shapes(pipeline.generator.model, history.shape[1]):
            _, clean = pipeline.generator(noisy_image_or_video=video, conditional_dict=conditional,
                timestep=timestep, kv_cache=caches, crossattn_cache=pipeline.crossattn_cache,
                current_start=history_start, updating_cache=False)
        return clean[:, -3:].contiguous()
    finally:
        _active.reset(token)
