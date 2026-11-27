"""Instrument the pinned LingBot causal-fast loop without changing its sampling.

This building block is not a formal launcher. Callers must verify sources,
weights, prepared inputs and GPU ownership before constructing the pipeline.
"""
from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

from .lingbot_geometry import require_horizon, validate_decoded_video
from .lingbot_inputs import read_npy

SEED = 20260911
CHUNK_SIZE = 4  # generate.py CLI default, shared by both official model examples.


def native_config(model_size: str) -> dict:
    if model_size not in ('1.3b', '14b'):
        raise ValueError('unsupported LingBot model size')
    return {'model_size': model_size, 'infer_mode': 'causal_fast',
            'chunk_size': CHUNK_SIZE, 'local_attn_size': 18, 'sink_size': 6,
            'shift': 5.0 if model_size == '1.3b' else 10.0,
            'timesteps_index': [0, 179, 358, 679], 'seed': SEED,
            'max_area': 480 * 832, 'offload_model': False}


class _TimedCallable:
    def __init__(self, target, invoke):
        self.target, self.invoke = target, invoke

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        return self.invoke(self.target, *args, **kwargs)


def generate_native(*, pipeline: Any, prompt: str, image: Any,
                    action_path: Path, duration_seconds: int, model_size: str,
                    torch: Any) -> tuple[Any, dict]:
    """Run one single-GPU native baseline; report generation and full-pipeline time.

    The official image encode and video decode remain unchanged. Generation
    wall latency excludes text and VAE stages, but includes camera conditioning,
    allocation and the sampling loop. Hooks are restored even on failure.
    """
    config = native_config(model_size)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('empty prompt')
    if (pipeline.infer_mode != 'causal_fast' or pipeline.rank != 0
            or pipeline.sp_size != 1 or pipeline.local_attn_size != 18
            or pipeline.sink_size != 6):
        raise ValueError('native baseline requires verified single-GPU causal-fast configuration')
    expected_dim, expected_heads, expected_layers = (1536,12,30) if model_size == '1.3b' else (5120,40,40)
    if (pipeline.model.config.dim, pipeline.model.config.num_heads, pipeline.model.config.num_layers) != (expected_dim, expected_heads, expected_layers):
        raise ValueError('model architecture does not match baseline label')
    header, _ = read_npy((action_path/'poses.npy').read_bytes())
    contract = require_horizon(duration_seconds, header['shape'][0])
    frames, latent_frames = contract['requested_rgb_frames'], contract['requested_latent_frames']
    if header['shape'] != (frames,4,4):
        raise ValueError('prepared camera input must have exact horizon')
    for filename in ('intrinsics.npy','action.npy','wasd_action.npy','ijkl_action.npy'):
        if filename == 'ijkl_action.npy' and not (action_path/filename).exists():
            continue
        h, _ = read_npy((action_path/filename).read_bytes())
        if h['shape'] != (frames,4):
            raise ValueError('prepared inputs have inconsistent temporal geometry')
    # Keep prompt encoding cold and consistent across repeated methods/prompts.
    pipeline._t5_cache.clear()
    original_text = pipeline.text_encoder
    original_encode, original_decode = pipeline.vae.encode, pipeline.vae.decode
    original_forward = pipeline.model.forward
    stages = {'text':0.0,'image_encode':0.0,'video_decode':0.0}
    counts = dict.fromkeys(stages,0)
    chunks = []
    device = pipeline.device

    def measure(name, fn, *args, **kwargs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize(device)
        stages[name] += time.perf_counter() - start
        counts[name] += 1
        return result

    def forward(*args, **kwargs):
        start, width = kwargs.get('current_start'), kwargs.get('frame_seqlen')
        if type(width) is not int or width <= 0 or type(start) is not int or start % (CHUNK_SIZE*width):
            raise ValueError('invalid native model cursor')
        index = start // (CHUNK_SIZE*width)
        if index == len(chunks):
            if chunks and chunks[-1]['forward_calls'] != 5:
                raise ValueError('incomplete previous native chunk')
            chunks.append({'chunk':index,'current_start':start,'frame_seqlen':width,'forward_calls':0})
        if index != len(chunks)-1 or index >= latent_frames//CHUNK_SIZE or chunks[index]['forward_calls'] >= 5:
            raise ValueError('native model reordered or exceeded chunk horizon')
        if chunks[index]['frame_seqlen'] != width:
            raise ValueError('native frame sequence width changed')
        result = original_forward(*args, **kwargs)
        chunks[index]['forward_calls'] += 1
        return result

    pipeline.text_encoder = _TimedCallable(original_text, lambda fn,*a,**kw:measure('text',fn,*a,**kw))
    pipeline.vae.encode = lambda *a,**kw:measure('image_encode',original_encode,*a,**kw)
    pipeline.vae.decode = lambda *a,**kw:measure('video_decode',original_decode,*a,**kw)
    pipeline.model.forward = forward
    try:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        video = pipeline.generate(prompt, image, action_path=str(action_path),
            chunk_size=CHUNK_SIZE, max_area=config['max_area'], frame_num=frames,
            timesteps_index=config['timesteps_index'], shift=config['shift'],
            seed=SEED, offload_model=False, max_attention_size=None)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-start
    finally:
        pipeline.text_encoder = original_text
        pipeline.vae.encode, pipeline.vae.decode = original_encode, original_decode
        pipeline.model.forward = original_forward
        pipeline._t5_cache.clear()
    if counts != {'text':1,'image_encode':1,'video_decode':1}:
        raise ValueError('native text/VAE stage count drifted')
    if len(chunks) != latent_frames//CHUNK_SIZE or any(c['forward_calls'] != 5 for c in chunks):
        raise ValueError('native loop did not complete four denoise plus one clean-cache call per chunk')
    if (not isinstance(video,torch.Tensor) or len(video.shape) != 4
            or video.shape[0] != 3 or min(video.shape[2:]) <= 0
            or any(n % 16 for n in video.shape[2:])
            or video.shape[2]*video.shape[3] > config['max_area']
            or not bool(torch.isfinite(video).all())):
        raise ValueError('invalid native decoded video')
    if any(c['frame_seqlen'] != video.shape[2]*video.shape[3]//256 for c in chunks):
        raise ValueError('decoded spatial geometry differs from native model sequence')
    validate_decoded_video(contract,frame_count=int(video.shape[1]),fps=16)
    generation = elapsed-sum(stages.values())
    if not math.isfinite(generation) or generation <= 0:
        raise ValueError('invalid measured generation time')
    return video, {'schema':'lingbot_native_generation_v1','config':config,
                   'temporal_contract':contract,'chunks':chunks,'stage_calls':counts,
                   'stage_wall_seconds':stages,'generation_wall_seconds':generation,
                   'pipeline_wall_seconds':elapsed,'decoded_shape':list(video.shape),
                   'timing_scope':'single_gpu_generation_excludes_text_and_vae_includes_camera_and_allocation'}
