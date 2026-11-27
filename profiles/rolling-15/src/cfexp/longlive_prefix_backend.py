"""Diagnostic-only Target first-prediction/native LongLive suffix bridge.

The Target output is a clean prediction, not x_750 or a completed block.
It is re-noised at LongLive's own 937.5 timestep by the existing explicit
noise handoff. Attention allocation remains local-12/sink-3 and all clean
updates have width three. No Self-Forcing model, schedule or KV is loaded.
"""
from __future__ import annotations

from .longlive_native import LongLiveNativeError
from .self_forcing_frame_adapter import OfficialSelfForcingBackend as _Observation


class LongLivePrefixMixin:
    reset_early_draft = _Observation.reset_early_draft
    resume_early_target_prefix = _Observation.resume_early_target_prefix

    def enable_prefix_diagnostic(self):
        if self.session.host_cursors.enabled:
            raise LongLiveNativeError('prefix diagnostic requires the native device cursor path')
        # EarlyDraftExecutor is model-independent. This view is essential:
        # using the raw pipeline would omit LongLive's native KV-size override.
        self.pipeline = self.session.executor.pipeline
        self._incremental_draft.scope_sha256 = self.session.executor.scope_sha256

    def generate_chunk(self, **kwargs):
        widths = kwargs.get('history_kv_widths')
        if widths is not None and any(type(w) is not int or w != 3 for w in widths):
            raise LongLiveNativeError('LongLive prefix history requires native B3 clean updates')
        return super().generate_chunk(**kwargs)

    def commit_early_target(self, **_kwargs):
        # The common full-Target commit uses width one. This experimental
        # bridge intentionally supports partial-prediction handoff only.
        raise LongLiveNativeError('LongLive prefix diagnostic cannot use a width-one Target commit')

    def close(self):
        early = getattr(self, '_early_draft', None)
        if early is not None:
            early.payload = None
            early.pending = None
        super().close()
