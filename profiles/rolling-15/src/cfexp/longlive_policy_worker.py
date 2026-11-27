"""Isolated native LongLive worker with pinned MUSIQ policy measurements.

This factory is deliberately separate from the Self-Forcing loader. The native
weights, LoRA, attention window and clean-cache semantics remain LongLive's.
"""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

from .correction_runtime import _bind_records_to_spec, _factory_config, _require, _strict_json_file
from .longlive_adapter import (
    OfficialLongLiveBackend, _load_artifact_config, _materialize_source,
    _verify_artifacts,
)
from .longlive_policy_backend import LongLivePolicyBackend
from .longlive_prefix_backend import LongLivePrefixMixin
from .self_forcing_frame_adapter import OfficialSelfForcingBackend as _Observation

MUSIQ_SHA256 = '358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50'


def _bind_policy_records(spec, records):
    """Bind compact baseline-preflight records without discarding rich checks.

    Baseline preflight emits name/path/hash/size; policy workers historically
    expected role/root/requested_path too. Those fields come only from the
    already frozen spec, then pass the same strict binder. Actual file bytes
    are still reverified by each native/scorer loader before model loading.
    """
    expanded = []
    for record in records:
        if set(record) == {'name', 'resolved_path', 'sha256', 'size'}:
            frozen = next((item for item in spec.get('artifacts', [])
                           if item.get('name') == record['name']), None)
            _require(frozen is not None, 'unknown compact policy artifact')
            record = {**record, 'role': frozen['role'], 'root': frozen['root'],
                      'requested_path': frozen['path']}
        expanded.append(record)
    return _bind_records_to_spec(spec, expanded)


def _load_quality(path, device):
    from pyiqa.archs.musiq_arch import MUSIQ
    from torchvision.transforms import Resize
    metric = MUSIQ(pretrained_model_path=str(path))
    metric.to(device).eval().requires_grad_(False)
    return metric, Resize


def _bind_conditioning(native, config, source_root):
    """Seal embeddings from LongLive's own encoder before replacing it.

    The existing cache verifier still binds source files and exact tensor
    bytes. A different Target implementation is rejected by its consumer.
    """
    if 'conditioning_cache' not in config:
        return None
    from .conditioning_cache import (
        TorchConditioningTensorCodec, make_torch_conditioner,
        seal_shared_conditioning_cache,
    )
    live_encoder = native.pipeline.text_encoder
    with native.torch.inference_mode():
        loaded = seal_shared_conditioning_cache(config['conditioning_cache'],
            run_dir=config['run_dir'], source_root=source_root,
            live_encode=lambda prompts: live_encoder(text_prompts=list(prompts)),
            codec=TorchConditioningTensorCodec(native.torch))
    frozen = make_torch_conditioner(loaded, torch_module=native.torch)
    frozen = frozen.to(device=native.device)
    binding = loaded.audit_binding()
    # Commit replacement only after the seal, device transfer and audit pass.
    native.pipeline.text_encoder = frozen
    del live_encoder
    native.torch.cuda.empty_cache()
    return binding


class OwnedLongLivePolicyBackend(LongLivePolicyBackend):
    score_frame_indices = _Observation.score_frame_indices

    def __init__(self, *, native_backend, scope_sha256, chunk_count, musiq_path):
        self._owned_native = native_backend
        try:
            super().__init__(native_backend=native_backend, scope_sha256=scope_sha256,
                             chunk_count=chunk_count)
            self.musiq, self.resize_type = _load_quality(musiq_path, self.device)
        except BaseException:
            self.close()
            raise

    def close(self):
        # Base construction may fail before a session exists; all paths must
        # still release the owned native import/cwd state exactly once.
        try:
            if hasattr(self, 'session') and not getattr(self, 'closed', False):
                super().close()
            if hasattr(self, 'musiq'):
                del self.musiq
        finally:
            native = self._owned_native
            self._owned_native = None
            if native is not None:
                native.close()


class PrefixDiagnosticLongLiveBackend(LongLivePrefixMixin, OwnedLongLivePolicyBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.enable_prefix_diagnostic()
        except BaseException:
            self.close()
            raise


def build_backend(raw_config):
    config = _factory_config(raw_config, role='draft')
    prefix_mode = os.environ.get('CFEXP_LONGLIVE_PREFIX_DIAGNOSTIC', '0')
    _require(prefix_mode in ('0', '1'), 'invalid LongLive prefix diagnostic flag')
    spec = json.loads(config['artifact_spec_path'].read_text(encoding='utf-8'))
    hashes = _bind_policy_records(spec, config['verified_artifacts'])
    _require(hashes.get('musiq_spaq_checkpoint') == MUSIQ_SHA256,
             'LongLive policy requires the frozen MUSIQ-SPAQ checkpoint')
    # Rehash actual bytes, not only preflight metadata, before loading anything.
    request = {
        'artifact_spec_path': str(config['artifact_spec_path']),
        'artifact_spec_sha256': config['artifact_spec_sha256'],
        'verified_artifacts': [{k: record[k] for k in ('name','resolved_path','sha256','size')}
                               for record in config['verified_artifacts']],
    }
    artifacts = _verify_artifacts(request)
    native_config, _ = _load_artifact_config(request, artifacts)
    source, assets = _materialize_source(SimpleNamespace(
        run_dir=config['run_dir'], worker_index=0, prompt_id='policy-worker',
        artifacts=artifacts, config=native_config))
    native = OfficialLongLiveBackend(source_root=source, model_root=assets,
                                    artifacts=artifacts, config=native_config)
    try:
        conditioning_binding = _bind_conditioning(native, config, source)
        scope = hashlib.sha256(json.dumps({
            'artifact_spec_sha256': config['artifact_spec_sha256'],
            'backend_config_sha256': config['_verified_config_sha256'],
            'native_load': native.native_load,
        }, sort_keys=True).encode()).hexdigest()
    except BaseException:
        native.close()
        raise
    backend_type = PrefixDiagnosticLongLiveBackend if prefix_mode == '1' else OwnedLongLivePolicyBackend
    backend = backend_type(native_backend=native, scope_sha256=scope,
        chunk_count=config['chunk_count'], musiq_path=artifacts['musiq_spaq_checkpoint'])
    generation = native_config['generation']
    backend.model_state = {
        'model_id': 'longlive_1.3b',
        'source_commit': native_config['official_source_commit'],
        'checkpoint_sha256': hashes[generation['checkpoint_artifact']],
        'lora_sha256': hashes[generation['lora_artifact']],
        'base_model_sha256': hashes['wan_model_weights'],
        'vae_sha256': hashes['wan_vae_checkpoint'],
        'text_encoder_sha256': hashes['wan_t5_checkpoint'],
        'musiq_spaq_sha256': hashes['musiq_spaq_checkpoint'],
        'artifact_spec_sha256': config['artifact_spec_sha256'],
        'backend_config_sha256': config['_verified_config_sha256'],
        'native_load': native.native_load,
        'dtype': 'bfloat16', 'denoising_steps': 4,
        'native_first_timestep': 1000.0,
        'hidden_layer_zero_based': 14, 'hidden_token_sample_count_max': 256,
        'effective_rank_relative_cutoff': 1e-7,
        'kv_policy': 'longlive_native_local_12_sink_3_incremental',
        'kv_local_attention_size': 12, 'kv_sink_size': 3,
        'history_clean_update_width': 3,
        'main_table_t_stitch_step_supported': False,
        'main_table_sdvg_supported': False,
        'quality_metric': 'musiq_spaq_frame_mean_raw_div_100',
        'chunk_count': config['chunk_count'],
    }
    if conditioning_binding is not None:
        backend.conditioning_cache_binding = conditioning_binding
        backend.model_state['conditioning_cache'] = dict(conditioning_binding)
    if prefix_mode == '1':
        backend.model_state['prefix_diagnostic'] = {
            'target_steps': [1000.0],
            'draft_suffix_steps': [937.5, 833.3333129882812, 625.0],
            'history_clean_update_width': 3,
            'formal_quality_or_speed_claim': False,
        }
    mode = os.environ.get('CFEXP_LONGLIVE_CPU_HISTORY_BINDING', '0')
    if mode not in ('0', '1'):
        backend.close()
        raise ValueError('CFEXP_LONGLIVE_CPU_HISTORY_BINDING must be 0 or 1')
    if mode == '1':
        try:
            from .cpu_history_binding import install_longlive
            install_longlive(backend)
            backend.model_state['cpu_history_binding'] = 'longlive_native_cpu_history_binding_v1'
        except BaseException:
            backend.close()
            raise
    if 'sdvg' in config:
        try:
            backend = _wrap_sdvg(backend, config)
        except BaseException:
            backend.close()
            raise
    return backend


def _wrap_sdvg(backend, config):
    from .sdvg_adapter import SdvgDraftBackend
    sdvg = config['sdvg']
    spec = _strict_json_file(sdvg['artifact_spec_path'])
    _require(spec.get('schema') == 'main_table_baseline_artifacts_v1'
             and spec.get('status') == 'frozen' and spec.get('method_id') == 'sdvg'
             and spec.get('adapter_module') == 'cfexp.sdvg_adapter'
             and spec.get('adapter_config') == sdvg['adapter_config'],
             'LongLive SDVG config differs from its frozen specification')
    hashes = _bind_policy_records(spec, sdvg['verified_artifacts'])
    wrapped = SdvgDraftBackend(base_backend=backend, run_dir=config['run_dir'],
        adapter_config=sdvg['adapter_config'], verified_artifacts=sdvg['verified_artifacts'])
    wrapped.model_state = dict(backend.model_state)
    wrapped.model_state['main_table_sdvg_supported'] = True
    wrapped.model_state['sdvg'] = {
        'officiality_label': 'unofficial_paper_faithful_reproduction',
        'image_reward_source_commit': sdvg['adapter_config']['image_reward']['source_commit'],
        'image_reward_source_sha256': hashes['image_reward_source'],
        'image_reward_checkpoint_sha256': hashes['image_reward_checkpoint'],
        'image_reward_med_config_sha256': hashes['image_reward_med_config'],
        'bert_revision': sdvg['adapter_config']['image_reward']['bert_revision'],
        'threshold': sdvg['adapter_config']['routing']['threshold'],
        'artifact_spec_sha256': sdvg['artifact_spec_sha256'],
    }
    return wrapped
