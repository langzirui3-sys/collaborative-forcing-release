"""Own one native Rolling Forcing pipeline in a fresh dedicated process."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

from .rolling_forcing_assets import materialize
from .rolling_forcing_native import load_native_pipeline
from .rolling_forcing_stream import RollingForcingError


class RollingNativeBackend:
    def __init__(self, *, run_dir, repository_root, model_root):
        # These generic upstream package names must never refer to SF/LongLive.
        if any(name in sys.modules for name in ('pipeline', 'wan', 'utils', 'flash_attn', 'flash_attn_2_cuda')):
            raise RollingForcingError('native Rolling backend requires a fresh isolated process')
        self.pipeline = None
        self.closed = False
        self._cwd, self._path = Path.cwd(), list(sys.path)
        self.source, self.artifacts, self.records = materialize(
            run_dir=run_dir, repository_root=repository_root, model_root=model_root)
        try:
            sys.path[:0] = [str(self.source/'_runtime_site'), str(self.source)]
            os.chdir(self.source)
            import torch
            from omegaconf import OmegaConf
            from .longlive_adapter import _verify_flash_attention
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RollingForcingError('native Rolling worker requires exactly one visible GPU')
            self.torch, self.device = torch, torch.device('cuda:0')
            _verify_flash_attention(self.source/'_runtime_site')
            module = importlib.import_module('pipeline.rolling_forcing_inference')
            if not Path(module.__file__).resolve().is_relative_to(self.source.resolve()):
                raise RollingForcingError('native pipeline import escaped pinned source')
            self.pipeline, self.audit = load_native_pipeline(
                source_root=self.source, checkpoint_path=self.artifacts['rolling_ema'],
                torch=torch, omega_conf=OmegaConf, pipeline_type=module.CausalInferencePipeline,
                device=self.device)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        if self.pipeline is not None:
            self.pipeline.kv_cache_clean = None
            self.pipeline.crossattn_cache = None
            self.pipeline = None
        os.chdir(self._cwd)
        sys.path[:] = self._path
        self.closed = True
        # Upstream imports are deliberately not reusable: this process exits.
