# SPDX-License-Identifier: Apache-2.0
# Adapted from thu-ml/Causal-Forcing; see docs/licenses/causal-forcing/NOTICE.md.
"""Resumable pure-Draft Rolling Forcing loop, pending native CUDA parity.

Preserves the pinned Causal-Forcing rolling loop, including full-window random
re-noising draws and warm-up clean-cache writes. It is not a sequential
four-step chunk sampler. History-conditioned restart is experimental: replay
accepted clean history into empty caches and discard partially denoised future
states. No comparison uses this extension before CUDA and history gates pass.
"""
from __future__ import annotations

import os



class RollingForcingError(RuntimeError):
    pass


class RollingForcingStream:
    def __init__(self, *, pipeline, torch, noise, prompt: str, history=None):
        if (pipeline.num_frame_per_block != 3 or pipeline.independent_first_frame
                or pipeline.args.context_noise != 0):
            raise RollingForcingError('requires native three-frame zero-context configuration')
        steps = tuple(float(x) for x in pipeline.denoising_step_list)
        if steps != (1000., 937.5, 833.3333129882812, 625.):
            raise RollingForcingError('requires pinned warped four-step schedule')
        minimum = 12 if history is None else 3
        if noise.ndim != 5 or noise.shape[0] != 1 or noise.shape[1] % 3 or noise.shape[1] < minimum:
            raise RollingForcingError('noise must contain complete native blocks (four for unconditioned parity)')
        if history is not None and (history.ndim != 5 or history.shape[0] != 1
                or history.shape[1] < 3 or history.shape[1] % 3
                or history.shape[2:] != noise.shape[2:]
                or history.dtype != noise.dtype or history.device != noise.device):
            raise RollingForcingError('history must contain compatible complete accepted blocks')
        if pipeline.kv_cache_clean is not None or getattr(pipeline, 'crossattn_cache', None) is not None:
            raise RollingForcingError('requires exclusive ownership of empty native caches')
        mode = os.environ.get("CFEXP_ROLLING_HOST_WINDOW", "0")
        if mode not in ("0", "1"):
            raise RollingForcingError("CFEXP_ROLLING_HOST_WINDOW must be 0 or 1")
        self.host_window = mode == "1"
        from .rolling_host_cursors import enabled
        self.host_kv_cursors = enabled()
        self.pipeline, self.torch = pipeline, torch
        self.noise = noise.clone()
        self.output = torch.zeros_like(noise)
        self.noisy_cache = torch.zeros_like(noise)
        self.blocks = int(noise.shape[1]) // 3
        self.prefix_blocks = 0 if history is None else int(history.shape[1]) // 3
        self.window = 0
        self.closed = False
        self.cuda_devices = [noise.device.index if noise.device.index is not None else torch.cuda.current_device()] if noise.device.type == "cuda" else []
        self.rng_state = (torch.cuda.get_rng_state(noise.device) if self.cuda_devices else torch.get_rng_state()).clone()
        self.condition = pipeline.text_encoder(text_prompts=[prompt])
        pipeline._initialize_kv_cache(batch_size=noise.shape[0], dtype=noise.dtype, device=noise.device)
        if self.host_kv_cursors:
            from .rolling_host_cursors import initialize
            initialize(pipeline.kv_cache_clean, torch=torch)
        pipeline._initialize_crossattn_cache(batch_size=noise.shape[0], dtype=noise.dtype, device=noise.device)
        if history is not None:
            # Only accepted latents cross the model boundary, never another
            # model's KV or the abandoned branch's partially denoised future.
            with torch.no_grad():
                for index in range(self.prefix_blocks):
                    pipeline.generator(noisy_image_or_video=history[:,index*3:(index+1)*3],
                        conditional_dict=self.condition,
                        timestep=torch.zeros((1,3), device=noise.device, dtype=torch.float32),
                        kv_cache=pipeline.kv_cache_clean, crossattn_cache=pipeline.crossattn_cache,
                        current_start=index*3*pipeline.frame_seq_length, updating_cache=True)

    def _advance_window(self):
        p, t, noise = self.pipeline, self.torch, self.noise
        start = max(0, self.window - 3)
        end = min(self.blocks - 1, self.window) + 1
        first, last = start * 3, end * 3
        length = last - first
        if self.window < self.blocks:
            noisy_input = t.cat([self.noisy_cache[:, first:last-3], noise[:, last-3:last]], dim=1)
        else:
            noisy_input = self.noisy_cache[:, first:last]
        # Each block advances one stage per window. Equivalent to upstream
        # for >=4 blocks; also drains 1-3 remaining blocks after intervention.
        timestep = t.cat([t.ones((noise.shape[0],3), device=noise.device, dtype=t.float32)
            * p.denoising_step_list[self.window-block] for block in range(start,end)], dim=1)
        _, clean = p.generator(noisy_image_or_video=noisy_input, conditional_dict=self.condition,
            timestep=timestep, kv_cache=p.kv_cache_clean, crossattn_cache=p.crossattn_cache,
            current_start=(first+self.prefix_blocks*3)*p.frame_seq_length)
        self.output[:, first:last] = clean
        # Preserve native RNG consumption: draw for the ENTIRE window once
        # per unfinished block, then keep only that block's slice.
        with t.no_grad():
            for block in range(start, end):
                if self.host_window:
                    # The timestep above was constructed from this exact host
                    # window/block pair. No tensor value is approximated and
                    # no RNG draw is removed, resized or reordered.
                    step_index = self.window - block
                    if not 0 <= step_index < 4:
                        raise RollingForcingError('invalid host window cursor')
                    if step_index == 3:
                        continue
                    next_step = p.denoising_step_list[step_index+1:step_index+2].to(noise.device)
                else:
                    block_step = timestep[:, (block-start)*3:(block-start+1)*3].mean().item()
                    indices = t.nonzero(t.abs(p.denoising_step_list-block_step) < 1e-4, as_tuple=True)[0]
                    if len(indices) != 1:
                        raise RollingForcingError('ambiguous denoising timestep')
                    if indices == len(p.denoising_step_list)-1:
                        continue
                    next_step = p.denoising_step_list[indices+1].to(noise.device)
                renoised = p.scheduler.add_noise(clean.flatten(0,1), t.randn_like(clean.flatten(0,1)),
                    next_step*t.ones([noise.shape[0]*length], device=noise.device, dtype=t.long))
                self.noisy_cache[:, block*3:(block+1)*3] = renoised.unflatten(0,clean.shape[:2])[:, (block-start)*3:(block-start+1)*3]
            p.generator(noisy_image_or_video=clean[:, :3], conditional_dict=self.condition,
                timestep=(t.ones_like(timestep)*p.args.context_noise)[:, :3],
                kv_cache=p.kv_cache_clean, crossattn_cache=p.crossattn_cache,
                current_start=(first+self.prefix_blocks*3)*p.frame_seq_length, updating_cache=True)
        self.window += 1
        return start

    def next_block(self):
        if self.closed:
            raise RollingForcingError('stream is closed')
        if self.window >= self.blocks+3:
            raise StopIteration
        t = self.torch
        try:
            with t.random.fork_rng(devices=self.cuda_devices), t.no_grad():
                if self.cuda_devices:
                    t.cuda.set_rng_state(self.rng_state, self.noise.device)
                else:
                    t.set_rng_state(self.rng_state)
                while True:
                    index = self._advance_window()
                    if self.window >= 4:
                        self.rng_state = (t.cuda.get_rng_state(self.noise.device) if self.cuda_devices else t.get_rng_state()).clone()
                        return index+self.prefix_blocks, self.output[:, index*3:(index+1)*3].clone()
        except Exception:
            self.close()
            raise

    def close(self):
        self.pipeline.kv_cache_clean = None
        self.pipeline.crossattn_cache = None
        self.noise = self.noisy_cache = self.output = self.condition = None
        self.closed = True
