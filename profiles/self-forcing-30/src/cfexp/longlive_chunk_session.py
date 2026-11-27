"""Branch-aware LongLive 1.0 latent execution; CUDA parity is a launch gate.

The pinned native inference loop has no initial_latent argument. This bridge
uses the shared branch store, but preserves LongLive's attention setup and
three-frame clean-cache updates. It is not enabled by any formal manifest.
"""
from __future__ import annotations

from typing import Any

from .incremental_draft import IncrementalDraftExecutor
from .longlive_native import LongLiveNativeError


class _NativePipelineView:
    """Expose the chunk executor interface without modifying native attributes."""

    independent_first_frame = False

    def __init__(self, pipeline: Any, host_cursors=None):
        self.native = pipeline
        self.num_frame_per_block = pipeline.num_frame_per_block
        self.frame_seq_length = pipeline.frame_seq_length
        self.args = pipeline.args
        self.generator = pipeline.generator
        self.host_cursors = host_cursors
        if host_cursors is not None and host_cursors.enabled:
            from .longlive_host_cursors import CheckedGenerator
            self.generator = CheckedGenerator(pipeline.generator, host_cursors)
        self.text_encoder = pipeline.text_encoder
        self.scheduler = pipeline.scheduler
        self.denoising_step_list = pipeline.denoising_step_list

    def __getattr__(self, name):
        # Instrumentation/diagnostic code can use the same native VAE and
        # model hooks while cache allocation still goes through this view.
        return getattr(self.native, name)

    @property
    def kv_cache1(self):
        return self.native.kv_cache1

    @kv_cache1.setter
    def kv_cache1(self, value):
        self.native.kv_cache1 = value

    @property
    def crossattn_cache(self):
        return self.native.crossattn_cache

    @crossattn_cache.setter
    def crossattn_cache(self, value):
        self.native.crossattn_cache = value

    def _initialize_kv_cache(self, **kwargs):
        # Match LongLive1.0/pipeline/causal_inference.py's local-window allocation.
        self.native._initialize_kv_cache(
            **kwargs, kv_cache_size_override=12 * self.frame_seq_length)
        if self.host_cursors is not None:
            self.host_cursors.initialize(self.native.kv_cache1)

    def _initialize_crossattn_cache(self, **kwargs):
        self.native._initialize_crossattn_cache(**kwargs)


class LongLiveChunkSession:
    """Generate a chunk from an explicitly selected, clean latent history.

    All history blocks, including external Target blocks, use [B,3] clean
    timesteps. The caller owns explicit initial noise and the per-chunk RNG
    state for the three re-noising draws. A scope digest must bind the verified
    model/LoRA/configuration. The shared executor binds prompt and latent bytes
    to each cached branch and clears native cache aliases after every call.
    """

    def __init__(self, *, pipeline: Any, torch: Any, scope_sha256: str,
                 max_bytes: int = 16 * 1024**3, fork_enabled: bool = True):
        config = pipeline.args
        kwargs = config.model_kwargs
        if (not getattr(pipeline, 'is_lora_enabled', False)
                or pipeline.num_frame_per_block != 3
                or config.context_noise != 0
                or pipeline.local_attn_size != 12
                or kwargs.local_attn_size != 12
                or kwargs.timestep_shift != 5.0
                or kwargs.sink_size != 3
                or tuple(float(x) for x in pipeline.denoising_step_list)
                   != (1000.0, 937.5, 833.3333129882812, 625.0)):
            raise LongLiveNativeError('chunk session requires the pinned native LongLive 1.0 configuration')
        if pipeline.kv_cache1 is not None or getattr(pipeline, 'crossattn_cache', None) is not None:
            raise LongLiveNativeError('chunk session requires exclusive ownership of an empty native cache')
        self.closed = False
        self.pipeline = pipeline
        # Native inference does this before the first block. Restored branches
        # retain the same window; this session never changes attention policy.
        pipeline.generator.model.local_attn_size = pipeline.local_attn_size
        pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)
        from .longlive_host_cursors import HostCursors
        self.host_cursors = HostCursors(torch=torch)
        self.last_host_cursor_evidence = None
        self.executor = IncrementalDraftExecutor(
            pipeline=_NativePipelineView(pipeline, self.host_cursors), torch=torch,
            scope_sha256=scope_sha256, max_bytes=max_bytes, fork_enabled=fork_enabled)

    def generate(self, *, prompt: str, history: Any, noise: Any,
                 kv_branch_action: str = 'none', **kwargs):
        if self.closed:
            raise LongLiveNativeError('chunk session is closed')
        if history is not None and (history.ndim != noise.ndim or history.shape[1] % 3):
            raise LongLiveNativeError('LongLive history must contain complete three-frame blocks')
        count = 0 if history is None else int(history.shape[1]) // 3
        if 'history_kv_widths' in kwargs:
            raise LongLiveNativeError('LongLive history always uses native three-frame clean updates')
        self.last_host_cursor_evidence = None
        before = self.host_cursors.snapshot()
        result = self.executor.generate(
            prompt=prompt, history=history, noise=noise,
            kv_branch_action=kv_branch_action, history_kv_widths=(3,) * count, **kwargs)
        if self.host_cursors.enabled:
            self.last_host_cursor_evidence = self.host_cursors.receipt(before)
            result['longlive_host_cursor_execution'] = dict(self.last_host_cursor_evidence)
        return result

    def prefill_accepted(self, *, prompt: str, history: Any, history_digest_source: Any = None):
        """Assimilate one accepted external block with the native B3 update."""
        if self.closed:
            raise LongLiveNativeError('chunk session is closed')
        count = int(history.shape[1]) // 3
        return self.executor.prefill_accepted(
            prompt=prompt, history=history, history_kv_widths=(3,) * count,
            history_digest_source=history_digest_source, maximum_suffix_chunks=1,
            target_kv_width=3)

    def close(self):
        self.executor.cache.clear()
        self.pipeline.kv_cache1 = None
        self.pipeline.crossattn_cache = None
        self.closed = True
