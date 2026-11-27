"""LongLive's native chunk session with the common policy observations.

The Self-Forcing loader and inference loop are never used here. Only its
model-independent request validation, step-zero instrumentation and RGB
utilities are shared. Formal policy routing still requires worker integration.
"""
from __future__ import annotations

from typing import Any

from .longlive_chunk_session import LongLiveChunkSession
from .longlive_native import LongLiveNativeError
from .self_forcing_frame_adapter import OfficialSelfForcingBackend as _Observation


class _NativeExecutor:
    """Translate common request metadata without changing native KV semantics."""

    def __init__(self, session):
        self.session = session
        self.fork_enabled = session.executor.fork_enabled

    def generate(self, *, history, history_kv_widths=None, **kwargs):
        count = 0 if history is None else int(history.shape[1]) // 3
        if history_kv_widths is not None and (
            not isinstance(history_kv_widths, tuple)
            or len(history_kv_widths) != count
            or any(type(w) is not int or w != 3 for w in history_kv_widths)
        ):
            raise LongLiveNativeError('LongLive requires three-frame clean updates for every model owner')
        return self.session.generate(history=history, **kwargs)


class LongLivePolicyBackend:
    """Observe an already verified native LongLive model; own only its session.

    Closing removes hooks and KV branches, but the caller still owns the native
    backend and its import/CUDA lifecycle. No Self-Forcing model identity or
    attention setting is inherited. MUSIQ is deliberately not silently loaded.
    """

    supports_history_kv_widths = True
    _timestep_values = _Observation._timestep_values
    _rank_sample = _Observation._rank_sample
    _install_hooks = _Observation._install_hooks
    decode_latents = _Observation.decode_latents
    temporal_consistency = _Observation.temporal_consistency
    write_video = _Observation.write_video

    @staticmethod
    def _native_hidden(output):
        # CausalWanAttentionBlock returns (x, cache_update_info). The parent
        # applies the deferred updates after all blocks have run. Observe x
        # only; never replace the hook return or mutate the metadata tuple.
        if (not isinstance(output, tuple) or len(output) != 2
                or not isinstance(output[1], tuple) or len(output[1]) != 3):
            raise LongLiveNativeError('unexpected native LongLive cached block output')
        return output[0]

    def _sample_hidden(self, output):
        return _Observation._sample_hidden(self, self._native_hidden(output))

    def _retain_full_hidden(self, output):
        return _Observation._retain_full_hidden(self, self._native_hidden(output))

    def __init__(self, *, native_backend: Any, scope_sha256: str, chunk_count: int):
        if type(chunk_count) is not int or chunk_count not in (16, 30, 60, 120):
            raise LongLiveNativeError('unsupported policy trajectory length')
        self.torch = native_backend.torch
        self.pipeline = native_backend.pipeline
        self.device = native_backend.device
        self.write_video_impl = native_backend.write_video_impl
        self.chunk_count = chunk_count
        self._active_capture = None
        self._hook_handles = []
        self.closed = False
        self.session = LongLiveChunkSession(
            pipeline=self.pipeline, torch=self.torch, scope_sha256=scope_sha256)
        self._incremental_draft = _NativeExecutor(self.session)
        try:
            self._install_hooks(self.pipeline.generator.model.blocks)
        except BaseException:
            self.close()
            raise

    def generate_chunk(self, **kwargs):
        if self.closed:
            raise LongLiveNativeError('policy backend is closed')
        result = _Observation.generate_chunk(self, **kwargs)
        if self.session.host_cursors.enabled:
            from .longlive_host_cursors import validate_evidence
            proof = self.session.last_host_cursor_evidence
            validate_evidence(proof)
            result['longlive_host_cursor_execution'] = dict(proof)
        return result

    def close(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.session.close()
        self.closed = True
