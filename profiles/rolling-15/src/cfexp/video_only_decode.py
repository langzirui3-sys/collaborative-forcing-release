"""Explicit video-only decode evidence; raw pixels stay in worker memory.

This path changes transport only. It uses the same VAE output and video writer,
and keeps pixel-content evidence distinct from a serialized tensor-file hash.
"""
from __future__ import annotations

import hashlib
import os
import re

ENV = 'CFEXP_VIDEO_ONLY_DECODE'
SCHEMA = 'insight2_video_only_decode_evidence_v1'


def enabled():
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{ENV} must be 0 or 1')
    return value == '1'


def pixel_witness(frames):
    import torch
    if (not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8
            or frames.device.type != 'cpu' or frames.ndim != 4
            or frames.shape[0] < 1 or frames.shape[1] != 3):
        raise ValueError('video-only decode requires CPU uint8 TCHW pixels')
    contiguous = frames.detach().contiguous()
    pixels = memoryview(contiguous.numpy()).cast('B')
    return {'layout': 'TCHW', 'dtype': 'uint8', 'shape': list(frames.shape),
            'raw_sha256': hashlib.sha256(pixels).hexdigest(), 'raw_nbytes': len(pixels)}


def evidence_path(history, measurement_id):
    if not isinstance(measurement_id, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,191}', measurement_id) is None:
        raise ValueError('invalid video-only measurement identity')
    return f'decoded/video-only/{history.manifest_sha256}/{measurement_id}.json'


def validate_pixels(value, expected_frames):
    if (not isinstance(value, dict)
            or set(value) != {'layout', 'dtype', 'shape', 'raw_sha256', 'raw_nbytes'}
            or value['layout'] != 'TCHW' or value['dtype'] != 'uint8'
            or value['shape'] != [expected_frames, 3, 480, 832]
            or type(value['raw_nbytes']) is not int
            or value['raw_nbytes'] != expected_frames * 3 * 480 * 832
            or not isinstance(value['raw_sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', value['raw_sha256']) is None):
        raise ValueError('video-only raw-pixel evidence is invalid')


def recover(evaluator, *, history, measurement_id):
    from .correction_runtime import _run_file, _strict_json_file, _sha256_file, _require
    from .insight2_campaign import PreparedFullQuality
    relative = evidence_path(history, measurement_id)
    path = _run_file(evaluator.run_dir, relative, must_exist=True)
    value = _strict_json_file(path)
    expected = evaluator._frame_count(history)
    _require(set(value) == {'schema', 'history_manifest_sha256', 'history_head_sha256',
                'latent_chunk_count', 'expected_frames', 'pixels', 'video'}
             and value['schema'] == SCHEMA
             and value['history_manifest_sha256'] == history.manifest_sha256
             and value['history_head_sha256'] == history.head_sha256
             and value['latent_chunk_count'] == history.next_chunk_index
             and value['expected_frames'] == expected,
             'video-only evidence belongs to another history')
    validate_pixels(value['pixels'], expected)
    video = value['video']
    _require(isinstance(video, dict) and video.get('path') == f'videos/{measurement_id}.mp4'
             and type(video.get('fps')) is int and video['fps'] == 24
             and type(video.get('frames')) is int and video['frames'] == expected,
             'video-only evidence belongs to another measurement')
    evaluator._verify_video_metadata(video)
    digest, _ = _sha256_file(path)
    return PreparedFullQuality(measurement_id=measurement_id,
        history_manifest_sha256=history.manifest_sha256,
        history_head_sha256=history.head_sha256, prompt_id=history.prompt_id,
        latent_chunk_count=history.next_chunk_index, decode_evidence_path=relative,
        decode_evidence_sha256=digest, video_path=video['path'],
        video_sha256=video['sha256'], video_nbytes=video['nbytes'],
        fps=video['fps'], frame_count=video['frames'])


def prepare(evaluator, *, history, measurement_id):
    from .correction_runtime import _strict_json_file, _write_immutable_json, _require
    from .insight2_campaign import PreparedFullQuality
    relative = evidence_path(history, measurement_id)
    evidence = evaluator.run_dir / relative
    pending = evaluator.run_dir / 'measurements/vbench-long-pending' / (measurement_id+'.json')
    if not evidence.exists():
        _require(not pending.exists(), 'pending video-only result has no decode evidence')
        decoded = evaluator.client.decode_video(history=history,
            output_video_path=f'videos/{measurement_id}.mp4', fps=24)
        validate_pixels(dict(decoded.frames), evaluator._frame_count(history))
        evaluator._verify_video_metadata(decoded.video)
        _write_immutable_json(evidence, {'schema': SCHEMA,
            'history_manifest_sha256': history.manifest_sha256,
            'history_head_sha256': history.head_sha256,
            'latent_chunk_count': history.next_chunk_index,
            'expected_frames': evaluator._frame_count(history),
            'pixels': dict(decoded.frames), 'video': dict(decoded.video)})
    result = recover(evaluator, history=history, measurement_id=measurement_id)
    if pending.exists():
        _require(PreparedFullQuality.from_dict(_strict_json_file(pending)) == result,
                 'pending video-only result differs from immutable evidence')
    else:
        _write_immutable_json(pending, result.to_dict())
    return result
