"""Measured native three-latent autoregressive loop shared by table baselines."""
from __future__ import annotations
from typing import Any


def generate_autoregressive(*, pipeline: Any, noise: Any, prompt: str, torch: Any,
                    device: Any, error_type: type[ValueError] = ValueError) -> tuple[Any, dict[str, Any]]:
    """Run the pinned native temporal loop and measure its real execution.

    Noise must be produced by the worker using its frozen per-prompt seed.
    Generation wall latency excludes text encoding and VAE decoding, matching
    the other table baselines; full pipeline wall time is retained separately.
    No branch/cache manipulation is introduced here.
    """
    import math
    import time

    if (not isinstance(noise, torch.Tensor) or tuple(noise.shape) not in
            ((1, 90, 16, 60, 104), (1, 180, 16, 60, 104), (1, 360, 16, 60, 104))
            or noise.dtype != torch.bfloat16 or not isinstance(prompt, str) or not prompt.strip()):
        raise error_type("invalid native noise geometry, dtype, or prompt")
    latent_frames = int(noise.shape[1])
    real_text = pipeline.text_encoder.forward
    real_forward = pipeline.generator.forward
    real_decode = pipeline.vae.decode_to_pixel
    durations = {"text": 0.0, "decode": 0.0}
    calls = {"text": 0, "decode": 0}
    chunks: dict[int, list[Any]] = {}

    def timed_stage(fn, name):
        def invoke(*args, **kwargs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            value = fn(*args, **kwargs)
            torch.cuda.synchronize(device)
            durations[name] += time.perf_counter() - start
            calls[name] += 1
            return value
        return invoke

    def forward(*args, **kwargs):
        position = kwargs.get("current_start")
        if type(position) is not int or position not in range(0, latent_frames * 1560, 3 * 1560):
            raise error_type("native generator cursor is outside the requested horizon")
        if position not in chunks:
            if position != len(chunks) * 3 * 1560:
                raise error_type("native generator skipped or reordered a chunk")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            chunks[position] = [start, end, 0]
        elif position != (len(chunks) - 1) * 3 * 1560:
            raise error_type("native generator revisited a completed chunk")
        value = real_forward(*args, **kwargs)
        chunks[position][1].record()
        chunks[position][2] += 1
        return value

    pipeline.text_encoder.forward = timed_stage(real_text, "text")
    pipeline.generator.forward = forward
    pipeline.vae.decode_to_pixel = timed_stage(real_decode, "decode")
    try:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            video = pipeline.inference(noise=noise, text_prompts=[prompt],
                                       return_latents=False, profile=False, low_memory=False)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        pipeline.text_encoder.forward = real_text
        pipeline.generator.forward = real_forward
        pipeline.vae.decode_to_pixel = real_decode
    if calls != {"text": 1, "decode": 1}:
        raise error_type("native generation did not encode/decode exactly once")
    if len(chunks) != latent_frames // 3 or any(row[2] != 5 for row in chunks.values()):
        raise error_type("native trace lacks four denoising calls plus clean KV update per chunk")
    if not isinstance(video, torch.Tensor) or tuple(video.shape) != (1, latent_frames * 4 - 3, 3, 480, 832):
        raise error_type("native decoded video geometry differs from requested horizon")
    per_chunk = [row[0].elapsed_time(row[1]) / 1000 for row in chunks.values()]
    generation = elapsed - durations['text'] - durations['decode']
    if generation <= 0 or any(not math.isfinite(v) or v < 0 for v in [elapsed, generation, *durations.values(), *per_chunk]):
        raise error_type("native timing is invalid")
    return video, {"pipeline_wall_seconds": elapsed, "generation_wall_seconds": generation,
                   "prompt_encode_seconds": durations['text'], "decode_seconds": durations['decode'],
                   "per_chunk_cuda_seconds": per_chunk, "latent_frames": latent_frames,
                   "rgb_frames": latent_frames * 4 - 3, "fps": 24,
                   "native_generator_calls": sum(row[2] for row in chunks.values())}
