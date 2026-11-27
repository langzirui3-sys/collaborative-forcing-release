"""Isolated Rolling policy worker with native state and frozen MUSIQ scoring.

The full future-noise bindings are supplied in the hashed worker config. This
factory does not infer future seeds or replace native Rolling with an SF loop.
"""
from pathlib import Path

from .correction_runtime import _factory_config, _require, _strict_json_file
from .longlive_policy_worker import MUSIQ_SHA256, _bind_policy_records, _load_quality, _bind_conditioning, _wrap_sdvg
from .rolling_forcing_assets import SPEC_SHA256, _contained_file
from .rolling_forcing_backend import RollingNativeBackend
from .rolling_forcing_native import _check_file
from .rolling_forcing_noise import verify_schedule
from .rolling_forcing_policy_backend import RollingPolicyBackend


class OwnedRollingPolicyBackend(RollingPolicyBackend):
    def __init__(self, *, native_backend, musiq_path, **kwargs):
        self._owned_native = native_backend
        try:
            super().__init__(native_backend=native_backend, **kwargs)
            self.musiq, self.resize_type = _load_quality(musiq_path, self.device)
        except BaseException:
            self.close()
            raise

    def close(self):
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


def _load_video_writer():
    from torchvision.io import write_video
    return write_video


def build_backend(raw_config):
    config = _factory_config(raw_config, role='draft')
    rolling = config.get('rolling')
    _require(isinstance(rolling, dict) and set(rolling) == {
        'repository_root', 'model_root', 'musiq_relative_path', 'noise_schedules'},
        'Rolling worker needs explicit native roots, scorer and noise schedules')
    _require(config['artifact_spec_sha256'] == SPEC_SHA256,
             'Rolling worker requires the pinned native artifact specification')
    repo = Path(rolling['repository_root']).resolve(strict=True)
    models = Path(rolling['model_root']).resolve(strict=True)
    _require(config['artifact_spec_path'] == repo/'experiments/erank3-rolling-native-v1.artifacts.json',
             'Rolling specification belongs to another repository')
    spec = _strict_json_file(config['artifact_spec_path'])
    hashes = _bind_policy_records(spec, config['verified_artifacts'])
    scorer = _contained_file(models, rolling['musiq_relative_path'])
    _check_file(scorer, MUSIQ_SHA256)
    bindings = rolling['noise_schedules']
    _require(isinstance(bindings, list) and bool(bindings), 'Rolling noise schedules are missing')
    for binding in bindings:
        _require(isinstance(binding, dict)
                 and Path(binding['run_dir']).resolve(strict=True) == config['run_dir']
                 and binding['chunk_count'] == config['chunk_count'],
                 'Rolling noise schedule run or horizon differs from worker')
        verify_schedule(**binding)
    native = RollingNativeBackend(run_dir=config['run_dir'], repository_root=repo, model_root=models)
    backend = None
    try:
        conditioning = _bind_conditioning(native, config, native.source)
        write_video = _load_video_writer()
        backend = OwnedRollingPolicyBackend(native_backend=native, musiq_path=scorer,
            run_dir=config['run_dir'], schedule_bindings=bindings,
            chunk_count=config['chunk_count'], write_video_impl=write_video)
        backend.model_state = {
            'model_id': 'rolling_forcing_1.3b', 'source_commit': spec['source_commit'],
            'checkpoint_sha256': hashes['rolling_ema'], 'native_load': native.audit,
            'vae_sha256': hashes['wan_vae_checkpoint'],
            'text_encoder_sha256': hashes['wan_t5_checkpoint'],
            'musiq_spaq_sha256': MUSIQ_SHA256,
            'artifact_spec_sha256': config['artifact_spec_sha256'],
            'backend_config_sha256': config['_verified_config_sha256'],
            'dtype': 'bfloat16', 'denoising_steps': 4,
            'hidden_layer_zero_based': 14, 'effective_rank_relative_cutoff': 1e-7,
            'kv_policy': 'native_rolling_accepted_history_bound_future_state',
            'history_clean_update_width': 3, 'chunk_count': config['chunk_count'],
            'quality_metric': 'musiq_spaq_frame_mean_raw_div_100',
            'main_table_sdvg_supported': False, 'main_table_t_stitch_step_supported': False,
        }
        if conditioning is not None:
            backend.conditioning_cache_binding = conditioning
            backend.model_state['conditioning_cache'] = dict(conditioning)
        from .rolling_host_cursors import enabled, SCHEMA
        if enabled():
            backend.model_state['native_host_kv_cursors'] = SCHEMA
        if 'sdvg' in config:
            backend = _wrap_sdvg(backend, config)
        return backend
    except BaseException:
        if backend is not None:
            backend.close()
        else:
            native.close()
        raise
