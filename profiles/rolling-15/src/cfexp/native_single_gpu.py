"""Explicit one-card execution for the coupled Rolling/LongLive renderer.

Both generation workers remain resident. Rendering starts only after generation
finishes; no model KV is shared, substituted or rebuilt by this execution mode.
"""
import os

ENV = 'CFEXP_NATIVE_SINGLE_GPU'
TIMING_SCOPE = 'native_warm_cached_conditioning_generation_then_coupled_vae_ready_v1'
MODELS = {'rolling': 'rolling_forcing_1.3b', 'longlive': 'longlive_1.3b'}


def enabled():
    flag = os.environ.get(ENV, '0')
    if flag not in ('0', '1'):
        raise ValueError('invalid native single-GPU switch')
    if flag == '0':
        return False
    if (os.environ.get('CFEXP_NATIVE_VIDEO_STREAM_MODEL') not in MODELS
            or os.environ.get('CFEXP_SF_SINGLE_GPU', '0') != '0'):
        raise ValueError('native single-GPU mode requires exactly its coupled native model')
    return True


def validate(model_id, renderer, *, enforce_idle):
    single = enabled()
    if single and (MODELS.get(renderer) != model_id or not enforce_idle
                   or os.environ.get('CFEXP_ACCEPTED_VIDEO_STREAM') != '1'
                   or os.environ.get('CFEXP_RENDER_TARGET_PRIORITY') != '1'):
        raise ValueError('native single-GPU execution requires matched renderer and ownership checks')
    return single


def devices(single, dual_devices):
    if not single:
        return dual_devices()
    from .accepted_stream import stream_devices
    return stream_devices()
