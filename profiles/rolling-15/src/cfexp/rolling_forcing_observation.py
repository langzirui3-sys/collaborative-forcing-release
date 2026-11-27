"""Read-only, block-local step-zero observations of native rolling windows.

The four live blocks have different timesteps. Only the three frames at 1000
belong to the new proposal; neighbouring blocks must not enter its eRank.
Observations stay pending until the corresponding block finishes denoising.
"""
from __future__ import annotations

from .rolling_forcing_stream import RollingForcingError
from .self_forcing_frame_adapter import (
    EFFECTIVE_RANK_LAYER, OfficialSelfForcingBackend, sampled_kv_cache_witness,
)


class RollingStepZeroObserver:
    _sample_hidden = OfficialSelfForcingBackend._sample_hidden
    _rank_sample = OfficialSelfForcingBackend._rank_sample
    _retain_full_hidden = OfficialSelfForcingBackend._retain_full_hidden

    def __init__(self, *, pipeline, torch, retain_hidden=False, capture_cache=False):
        if type(retain_hidden) is not bool or type(capture_cache) is not bool:
            raise RollingForcingError("observation options must be booleans")
        self.torch = torch
        self.frame_tokens = pipeline.frame_seq_length
        if type(self.frame_tokens) is not int or self.frame_tokens <= 0:
            raise RollingForcingError('invalid native frame token length')
        self.retain_hidden = retain_hidden
        self.capture_cache = capture_cache
        self.cache_witness = None
        self.pending = {}
        self.active = None
        self.handles = []
        self.closed = False
        try:
            self.handles.append(pipeline.generator.register_forward_pre_hook(
                self._begin, with_kwargs=True))
            self.handles.append(pipeline.generator.model.blocks[EFFECTIVE_RANK_LAYER]
                                .register_forward_hook(self._observe))
            self.handles.append(pipeline.generator.register_forward_hook(self._end))
        except BaseException:
            self.close()
            raise

    def _begin(self, module, args, kwargs):
        if self.closed or self.active is not None:
            raise RollingForcingError('observer is closed or previous window did not finish')
        if kwargs.get('updating_cache', False):
            return
        t = self.torch
        steps = kwargs.get('timestep')
        start = kwargs.get('current_start')
        if (not isinstance(steps, t.Tensor) or steps.ndim != 2 or steps.shape[0] != 1
                or steps.shape[1] not in (3, 6, 9, 12)
                or type(start) is not int or start < 0 or start % (3*self.frame_tokens)):
            raise RollingForcingError('invalid rolling observation geometry')
        values = steps.detach().cpu().reshape(-1).tolist()
        if any(len(set(values[i:i+3])) != 1 for i in range(0,len(values),3)):
            raise RollingForcingError('rolling block has mixed frame timesteps')
        slots = [i for i in range(0,len(values),3) if values[i] == 1000.]
        if not slots:  # Drain windows introduce no new block.
            return
        if len(slots) != 1 or slots[0] != len(values)-3:
            raise RollingForcingError('step-zero block must be the unique last window block')
        slot = slots[0]
        index = start // (3*self.frame_tokens) + slot//3
        if index in self.pending or len(self.pending) >= 4:
            raise RollingForcingError('duplicate or unconsumed rolling observations')
        self.cache_witness = (sampled_kv_cache_witness(
            kwargs["kv_cache"], kwargs.get("crossattn_cache"))
            if self.capture_cache else None)
        self.active = (index, slot*self.frame_tokens, len(values)*self.frame_tokens)

    def _observe(self, module, args, hidden):
        if self.active is None:
            return
        index, first, total = self.active
        if index in self.pending:
            raise RollingForcingError('step-zero layer executed twice in one window')
        if (not isinstance(hidden, self.torch.Tensor) or hidden.ndim != 3
                or hidden.shape[0] != 1 or hidden.shape[1] != total):
            raise RollingForcingError('rolling hidden tokens do not match the visual window')
        block = hidden[:,first:first+3*self.frame_tokens]
        sample = self._sample_hidden(block)
        measurement = self._rank_sample(sample)
        self.pending[index] = dict(chunk_index=index, layer=EFFECTIVE_RANK_LAYER,
            timestep=1000., token_count=int(block.shape[1]),
            effective_rank=measurement.effective_rank,
            gram_eigenvalues=measurement.gram_eigenvalues,
            hidden=self._retain_full_hidden(block) if self.retain_hidden else None,
            sampled_hidden=sample.detach().to(device="cpu", dtype=self.torch.bfloat16).contiguous()
                if self.retain_hidden else None,
            kv_witness=self.cache_witness)

    def _end(self, module, args, output):
        if self.active is not None and self.active[0] not in self.pending:
            raise RollingForcingError('step-zero transformer layer was not observed')
        self.active = None
        self.cache_witness = None

    def take(self, index):
        if type(index) is not int or index not in self.pending:
            raise RollingForcingError('missing block-local step-zero observation')
        return self.pending.pop(index)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()
        self.active = None
        self.closed = True
        self.cache_witness = None
