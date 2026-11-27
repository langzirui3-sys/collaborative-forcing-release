"""Guide Draft motion from a recent accepted Target block."""
from __future__ import annotations

import math
import os
from pathlib import PurePosixPath
import subprocess

from .accepted_stream import ENV as STREAM_ENV
from .accepted_stream_worker import StreamingCorrectionWorker
from .correction_worker import WorkerProtocolError
from .longlive_late_handoff import require

CONTROL_ARM = 'full_target_bursts_k6'
GUIDED_ARM = 'target_motion_continuation_k6'
FLOOR_ARM = 'target_motion_floor_k6'
AMPLITUDE_ARM = 'target_motion_amplitude_k6'
GUIDANCE_ARMS = (GUIDED_ARM, FLOOR_ARM, AMPLITUDE_ARM)
TARGET_POSITIONS = (1, 2, 20, 21, 40, 41)
GAIN_BY_LAG = (0.25, 0.125, 0.0625)
SCHEMA = 'longlive_target_motion_continuation_v1'
FLOOR_OPTIONS = dict(target_energy_quantile=0.5, max_draft_to_target_rms=1.0,
                     min_signed_projection=-0.35, projection_floor=0.55,
                     initial_gain=0.75, initial_extra_cap=0.35)
AMPLITUDE_OPTIONS = dict(patch_size=3, target_energy_quantile=0.5, max_rms_multiplier=1.5)


def reference_chunk(chunk):
    """Return the last Target chunk while its three-block guidance is active."""
    ref = max((n for n in TARGET_POSITIONS if n < chunk), default=0)
    if ref and chunk not in TARGET_POSITIONS and chunk - ref <= len(GAIN_BY_LAG):
        return ref
    return None


def _validate_latents(clean, reference):
    require(clean.ndim == 5 and clean.shape[1] == 3 and clean.shape == reference.shape
            and clean.dtype == reference.dtype and clean.device == reference.device,
            'motion continuation requires two native three-frame latents')


def guide_prediction(clean, reference, gain, *, motion_floor=False):
    """Extend aligned temporal changes, with an optional bounded weak-motion floor.

    The first Draft frame and all orthogonal components are preserved before
    bfloat16 rounding. The accepted block remains a three-frame clean latent.
    """
    _validate_latents(clean, reference)
    velocity = (reference[:, 2:3].float() - reference[:, :1].float()) * 0.5
    draft = clean.float()
    change = draft[:, 1:] - draft[:, :-1]
    energy = velocity.square().sum(dim=2, keepdim=True)
    projection = (change * velocity).sum(dim=2, keepdim=True) / energy.clamp_min(1e-8)
    aligned = projection.clamp(0, 1)
    scale = (gain * aligned).minimum(1 - aligned)
    if motion_floor:
        p = FLOOR_OPTIONS
        sites = energy.flatten(start_dim=2)
        # Higher empirical median. Ignore the index returned for tied values.
        k = math.ceil(p['target_energy_quantile'] * (sites.shape[-1] - 1)) + 1
        median = sites.kthvalue(k, dim=-1).values
        cutoff = median[:, :, None, None, None].clamp_min(1e-8)
        weak = change.square().sum(dim=2, keepdim=True) <= energy * p['max_draft_to_target_rms'] ** 2
        eligible = ((energy >= cutoff) & weak & (projection >= p['min_signed_projection'])
                    & (projection < p['projection_floor']))
        taper = gain / GAIN_BY_LAG[0]
        boost = ((p['projection_floor'] - projection) * p['initial_gain']).clamp(
            min=0, max=p['initial_extra_cap']) * taper
        scale = scale.maximum(boost.where(eligible, 0))
    extra = scale * velocity
    output = clean.clone()
    output[:, 1:] = (draft[:, 1:] + extra.cumsum(dim=1)).to(clean.dtype)
    return output


def match_motion_amplitude(clean, reference, gain):
    """Amplify existing Draft changes using local RMS from the last Target block."""
    from torch.nn.functional import avg_pool2d

    _validate_latents(clean, reference)
    draft = clean.float()
    target = reference.float()
    draft_change = draft[:, 1:] - draft[:, :-1]
    target_change = target[:, 1:] - target[:, :-1]
    patch_size = AMPLITUDE_OPTIONS['patch_size']
    draft_energy = avg_pool2d(draft_change.square().mean(dim=(1, 2)).unsqueeze(1),
                             patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    target_energy = avg_pool2d(target_change.square().mean(dim=(1, 2)).unsqueeze(1),
                              patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    sites = target_energy.flatten(start_dim=2)
    k = math.ceil(AMPLITUDE_OPTIONS['target_energy_quantile'] * (sites.shape[-1] - 1)) + 1
    cutoff = sites.kthvalue(k, dim=-1).values[:, :, None, None].clamp_min(1e-8)
    multiplier = (target_energy / draft_energy.clamp_min(1e-8)).sqrt().clamp(
        min=1, max=AMPLITUDE_OPTIONS['max_rms_multiplier'])
    extra_scale = (multiplier - 1).where(target_energy >= cutoff, 0) * (gain / GAIN_BY_LAG[0])
    extra = draft_change * extra_scale.unsqueeze(1)
    output = clean.clone()
    output[:, 1:] = (draft[:, 1:] + extra.cumsum(dim=1)).to(clean.dtype)
    return output


def receipt(chunk, ref, *, arm=GUIDED_ARM):
    require(arm in GUIDANCE_ARMS, 'unknown motion guidance arm')
    result = dict(schema=SCHEMA, chunk_index=chunk, target_reference_chunk=ref,
                gain=GAIN_BY_LAG[chunk-ref-1], native_denoising_forwards=4,
                native_clean_forwards=1, clean_update_width=3)
    if arm == FLOOR_ARM:
        result.update(schema='longlive_target_motion_floor_v1', floor=FLOOR_OPTIONS.copy())
    if arm == AMPLITUDE_ARM:
        result.update(schema='longlive_target_motion_amplitude_v1', amplitude=AMPLITUDE_OPTIONS.copy())
    return result


def generate_guided(session, generate, *, chunk, reference, arm=GUIDED_ARM, **kwargs):
    """Transform the final prediction before the existing clean-cache forward."""
    history = kwargs['history']
    require(history is not None and history.shape[1] == 3 * chunk,
            'guided Draft did not receive its accepted history')
    require(not session.host_cursors.enabled, 'guided Draft requires native CUDA cursors')
    view = session.executor.pipeline
    ref = history[:, 3*reference:3*reference+3]
    gain = GAIN_BY_LAG[chunk-reference-1]
    current_start = chunk * session.executor.tokens_per_chunk
    calls = 0
    guided = None

    def hook(_module, _args, fields, output):
        nonlocal calls, guided
        if fields['current_start'] != current_start:
            return None
        calls += 1
        if calls == 4:
            require(isinstance(output, tuple) and len(output) == 2,
                    'unexpected LongLive clean-prediction output')
            if arm == AMPLITUDE_ARM:
                guided = match_motion_amplitude(output[1], ref, gain)
            else:
                guided = guide_prediction(output[1], ref, gain, motion_floor=arm == FLOOR_ARM)
            return output[0], guided
        if calls == 5:
            require(fields['noisy_image_or_video'] is guided,
                    'the native clean-cache update did not consume the guided latent')
        return None

    handle = view.generator.register_forward_hook(hook, with_kwargs=True)
    try:
        result = generate(**kwargs)
    finally:
        handle.remove()
    require(calls == 5 and guided is not None and result['latent'] is guided,
            'guided latent was not accepted by the native Draft executor')
    return result


class MotionContinuationWorker(StreamingCorrectionWorker):
    def _generate(self, request):
        paths = request.get('output_paths')
        output = paths.get('latent') if isinstance(paths, dict) else None
        if self.role != 'draft' or not isinstance(output, str):
            return super()._generate(request)
        parts = PurePosixPath(output).parts
        if len(parts) != 4 or parts[0] != 'latents' or parts[1] not in GUIDANCE_ARMS:
            return super()._generate(request)
        arm = parts[1]
        chunk = request.get('chunk_index')
        if type(chunk) is not int or not 0 < chunk < 60 or parts[3] != f'{chunk:03d}.safetensors':
            raise WorkerProtocolError('guided Draft requires its bound 30-second output path')
        if request.get('action') != 'generate' or request.get('capture_hidden') is not False:
            raise WorkerProtocolError('guided Draft requires one ordinary uncaptured chunk')
        ref = reference_chunk(chunk)
        if ref is None:
            return super()._generate(request)
        expected = f'latents/{arm}/{parts[2]}/{ref:03d}.safetensors'
        history_paths = request.get('history_latent_paths')
        if (not isinstance(history_paths, list) or len(history_paths) != chunk
                or history_paths[ref] != expected):
            raise WorkerProtocolError('guided Draft has no bound accepted Target reference')

        from .longlive_chunk_session import LongLiveChunkSession
        session = self.backend.session
        if not isinstance(session, LongLiveChunkSession):
            raise WorkerProtocolError('guided Draft requires the native LongLive session')
        generate = session.generate

        def guided_generate(**kwargs):
            return generate_guided(session, generate, chunk=chunk, reference=ref, arm=arm, **kwargs)

        session.generate = guided_generate
        try:
            payload = super()._generate(request)
        finally:
            del session.generate
        result = payload['result']
        result['motion_continuation'] = dict(receipt(chunk, ref, arm=arm),
            reference_latent_sha256=result['kv_reuse']['history_file_sha256'][ref])
        return payload


def process_factory(argv, **kwargs):
    if os.environ.get(STREAM_ENV) != '1' or list(argv[1:3]) != ['-m', 'cfexp.correction_worker']:
        raise ValueError('motion continuation requires the explicit native stream launcher')
    return subprocess.Popen([argv[0], '-m', __name__, *argv[3:]], **kwargs)


def main():
    from . import correction_worker as base

    require(os.environ.get(STREAM_ENV) == '1', 'motion continuation requires native streaming')
    original = base.CorrectionWorker
    base.CorrectionWorker = MotionContinuationWorker
    try:
        return base.main()
    finally:
        base.CorrectionWorker = original


if __name__ == '__main__':
    raise SystemExit(main())
