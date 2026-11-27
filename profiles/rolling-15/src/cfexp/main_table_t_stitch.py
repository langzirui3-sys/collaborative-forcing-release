"""T-Stitch 1S+3T rollout core for the frozen main-table benchmark.

Each chunk uses one Self-Forcing Draft forward at timestep 1000 followed by
exactly three Krea Target forwards at 750, 500, and 250.  The handoff remains
in latent space and is bound by immutable tensor files; video-level splicing
is impossible on this path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Protocol

from .correction_runtime import DecodeResult
from .correction_state import LatentHistory
from .insight2_campaign import (
    ChunkGenerationRequest,
    GeneratedChunk,
    NoiseProvider,
    PromptCase,
)


SCHEMA = "main_table_t_stitch_rollout_v1"
METHOD_ID = "t_stitch_1s3t"
DRAFT_MODEL_ID = "self_forcing_1.3b"
TARGET_MODEL_ID = "krea-realtime-video-14b"
SEED = 20260911
CHUNK_COUNT = 120
LATENT_FRAMES_PER_CHUNK = 3
RGB_FRAME_COUNT = 1437
FPS = 24
HEIGHT = 480
WIDTH = 832
TARGET_RPC_PHASE = "route_materialization"


class TStitchError(RuntimeError):
    """Raised when the exact 1S+3T trajectory or its evidence drifts."""


class TStitchDraftClient(Protocol):
    role: str
    model_id: str
    t_stitch_step_supported: bool

    def generate_t_stitch_step(
        self,
        request: ChunkGenerationRequest,
        *,
        renoise_750_path: str,
        draft_clean_output_path: str,
        noisy_latent_750_output_path: str,
    ) -> Mapping[str, Any]: ...

    def decode(
        self,
        *,
        history: LatentHistory,
        output_frames_path: str,
        output_video_path: str | None,
        fps: int = 24,
    ) -> DecodeResult: ...


class TStitchTargetClient(Protocol):
    role: str
    model_id: str
    session_policy: str
    decode_during_generation: bool

    def start_trajectory(
        self, *, prompt_id: str, prompt: str, seed: int, chunk_count: int
    ) -> Mapping[str, Any]: ...

    def generate_persistent_suffix(
        self,
        request: ChunkGenerationRequest,
        *,
        noisy_latent_750_path: str,
        renoise_500_path: str,
        renoise_250_path: str,
    ) -> GeneratedChunk: ...

    def finish_trajectory(self, *, session_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TStitchGeneration:
    history: LatentHistory
    generated: tuple[GeneratedChunk, ...]
    result: Mapping[str, Any]


@dataclass(frozen=True)
class TStitchRollout:
    history: LatentHistory
    generated: tuple[GeneratedChunk, ...]
    decode: DecodeResult
    result: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TStitchError(message)


def _canonical_sha256(*values: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            list(values),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TStitchError(f"{label} is invalid") from error
    _require(math.isfinite(parsed) and parsed >= 0.0, f"{label} is invalid")
    return parsed


def generate_t_stitch_trajectory(
    *,
    prompt_id: str,
    prompt: str,
    draft: TStitchDraftClient,
    target: TStitchTargetClient,
    initial_noise_provider: NoiseProvider,
    renoise_750_provider: NoiseProvider,
    renoise_500_provider: NoiseProvider,
    renoise_250_provider: NoiseProvider,
    seed: int = SEED,
    chunk_count: int = CHUNK_COUNT,
    clock: Callable[[], float] = time.perf_counter,
    safety_check: Callable[[str, int], None] | None = None,
) -> TStitchGeneration:
    """Generate one supported-duration trajectory with the exact 1S+3T schedule."""

    _require(seed == SEED, f"main-table seed must be {SEED}")
    _require(type(chunk_count) is int and chunk_count in (30, 60, 120), "T-Stitch requires 30, 60, or 120 chunks")
    _require(draft.role == "draft" and draft.model_id == DRAFT_MODEL_ID, "wrong Draft model")
    _require(draft.t_stitch_step_supported is True, "Draft lacks T-Stitch support")
    _require(target.role == "target" and target.model_id == TARGET_MODEL_ID, "wrong Target model")
    _require(
        target.session_policy == "persistent_native_session",
        "T-Stitch requires one persistent native Target session",
    )
    _require(
        target.decode_during_generation is False,
        "T-Stitch generation timing must exclude VAE decode",
    )
    case = PromptCase(prompt_id=prompt_id, prompt=prompt, split="evaluation")
    trajectory_id = _canonical_sha256(METHOD_ID, prompt_id, seed, chunk_count)
    scope_id = _canonical_sha256("main-table-target-scope-v1", METHOD_ID, prompt_id, seed)
    session = target.start_trajectory(
        prompt_id=prompt_id,
        prompt=prompt,
        seed=seed,
        chunk_count=chunk_count,
    )
    _require(
        isinstance(session, Mapping)
        and set(session) == {"session_id", "prompt_encode_seconds"}
        and isinstance(session.get("session_id"), str)
        and bool(session["session_id"]),
        "Target persistent-session startup evidence is invalid",
    )
    session_id = str(session["session_id"])
    prompt_encode_seconds = _finite_nonnegative(
        session["prompt_encode_seconds"], "Target prompt encode time"
    )
    history = LatentHistory(
        trajectory_id=trajectory_id,
        prompt_id=prompt_id,
        trajectory_seed=seed,
    )
    generated: list[GeneratedChunk] = []
    chunk_timings: list[dict[str, Any]] = []

    for chunk_index in range(chunk_count):
        if safety_check is not None:
            safety_check("before_chunk", chunk_index)
        initial = initial_noise_provider.materialize(
            prompt_id=prompt_id, trajectory_seed=seed, chunk_index=chunk_index
        )
        noise_750 = renoise_750_provider.materialize(
            prompt_id=prompt_id, trajectory_seed=seed, chunk_index=chunk_index
        )
        noise_500 = renoise_500_provider.materialize(
            prompt_id=prompt_id, trajectory_seed=seed, chunk_index=chunk_index
        )
        noise_250 = renoise_250_provider.materialize(
            prompt_id=prompt_id, trajectory_seed=seed, chunk_index=chunk_index
        )
        draft_request = ChunkGenerationRequest(
            prompt=case,
            history=history,
            noise=initial,
            output_latent_path=(
                f"latents/{METHOD_ID}/{prompt_id}/chunk-{chunk_index:03d}-draft-unused.safetensors"
            ),
            capture_hidden=False,
        )
        clean_path = (
            f"handoff/{METHOD_ID}/{prompt_id}/chunk-{chunk_index:03d}-draft-clean.safetensors"
        )
        noisy_750_path = (
            f"handoff/{METHOD_ID}/{prompt_id}/chunk-{chunk_index:03d}-noisy-750.safetensors"
        )
        started = clock()
        draft_result = draft.generate_t_stitch_step(
            draft_request,
            renoise_750_path=noise_750.tensor_path,
            draft_clean_output_path=clean_path,
            noisy_latent_750_output_path=noisy_750_path,
        )
        draft_elapsed = _finite_nonnegative(clock() - started, "Draft step wall time")
        draft_evidence = draft_result.get("result")
        _require(isinstance(draft_evidence, Mapping), "Draft step evidence is absent")
        _require(
            draft_evidence.get("draft_timestep") == 1000.0
            and draft_evidence.get("target_handoff_timestep") == 750.0,
            "Draft handoff schedule drifted",
        )
        logical_call_id = _canonical_sha256(
            "main-table-target-call-v1",
            METHOD_ID,
            prompt_id,
            seed,
            chunk_index,
            history.head_sha256,
            initial.descriptor.descriptor_sha256,
            noise_750.descriptor.descriptor_sha256,
            noise_500.descriptor.descriptor_sha256,
            noise_250.descriptor.descriptor_sha256,
        )
        target_request = ChunkGenerationRequest(
            prompt=case,
            history=history,
            noise=initial,
            output_latent_path=(
                f"latents/{METHOD_ID}/{prompt_id}/chunk-{chunk_index:03d}-target.safetensors"
            ),
            capture_hidden=False,
            target_logical_call_id=logical_call_id,
            target_call_scope_id=scope_id,
            target_call_phase=TARGET_RPC_PHASE,
        )
        target_started = clock()
        item = target.generate_persistent_suffix(
            target_request,
            noisy_latent_750_path=noisy_750_path,
            renoise_500_path=noise_500.tensor_path,
            renoise_250_path=noise_250.tensor_path,
        )
        target_finished = clock()
        target_elapsed = _finite_nonnegative(
            target_finished - target_started, "Target suffix wall time"
        )
        chunk_elapsed = _finite_nonnegative(
            target_finished - started, "T-Stitch chunk wall time"
        )
        if safety_check is not None:
            safety_check("after_chunk", chunk_index)
        _require(item.context.role == "target", "T-Stitch returned a non-Target final chunk")
        _require(item.context.model_id == TARGET_MODEL_ID, "T-Stitch Target identity drifted")
        _require(item.target_logical_call_id == logical_call_id, "Target logical call drifted")
        diagnostics = item.diagnostics
        _require(
            isinstance(diagnostics, Mapping)
            and diagnostics.get("native_session_id") == session_id
            and diagnostics.get("native_session_block_index") == chunk_index
            and diagnostics.get("draft_handoff_timestep") == 1000.0
            and diagnostics.get("denoising_schedule") == [750.0, 500.0, 250.0]
            and diagnostics.get("session_recreated") is False
            and diagnostics.get("vae_decode_performed") is False,
            "Target suffix did not prove the exact 750/500/250 execution",
        )
        draft_cuda = _finite_nonnegative(
            draft_evidence.get("generation_cuda_seconds"), "Draft CUDA time"
        )
        target_cuda = _finite_nonnegative(
            diagnostics.get("generation_cuda_seconds"), "Target CUDA time"
        )
        item.segment.verify_context(history, item.context)
        history = history.append(item.segment)
        generated.append(item)
        chunk_timings.append(
            {
                "chunk_index": chunk_index,
                "generation_wall_seconds": chunk_elapsed,
                "draft_wall_seconds": draft_elapsed,
                "target_wall_seconds": target_elapsed,
                "generation_cuda_seconds": draft_cuda + target_cuda,
                "draft_cuda_seconds": draft_cuda,
                "target_cuda_seconds": target_cuda,
                "draft_clean_path": clean_path,
                "noisy_latent_750_path": noisy_750_path,
                "target_attempt_id": item.target_attempt_id,
                "target_logical_call_id": logical_call_id,
                "latent_sha256": item.segment.latent_sha256,
                "history_head_sha256": history.head_sha256,
            }
        )

    finished = target.finish_trajectory(session_id=session_id)
    _require(
        isinstance(finished, Mapping)
        and set(finished) == {"session_id", "generated_block_count"}
        and finished.get("session_id") == session_id
        and finished.get("generated_block_count") == chunk_count,
        "Target persistent-session completion evidence is invalid",
    )
    _require(history.next_chunk_index == chunk_count, "T-Stitch horizon is incomplete")
    generation_wall_seconds = math.fsum(
        float(item["generation_wall_seconds"]) for item in chunk_timings
    )
    generation_cuda_seconds = math.fsum(
        float(item["generation_cuda_seconds"]) for item in chunk_timings
    )
    result = {
        "schema": SCHEMA,
        "method_id": METHOD_ID,
        "prompt_id": prompt_id,
        "seed": seed,
        "draft_model_id": DRAFT_MODEL_ID,
        "target_model_id": TARGET_MODEL_ID,
        "trajectory_id": trajectory_id,
        "history_manifest_sha256": history.manifest_sha256,
        "history_head_sha256": history.head_sha256,
        "chunk_count": chunk_count,
        "latent_frame_count": chunk_count * LATENT_FRAMES_PER_CHUNK,
        "rgb_frame_count": chunk_count * 12 - 3,
        "fps": FPS,
        "height": HEIGHT,
        "width": WIDTH,
        "draft_model_calls": chunk_count,
        "target_model_calls": 3 * chunk_count,
        "schedule": [
            {"step_index": 0, "timestep": 1000, "model_role": "draft"},
            {"step_index": 1, "timestep": 750, "model_role": "target"},
            {"step_index": 2, "timestep": 500, "model_role": "target"},
            {"step_index": 3, "timestep": 250, "model_role": "target"},
        ],
        "latent_handoff": True,
        "video_splicing": False,
        "target_session_policy": "persistent_native_session",
        "target_session_count": 1,
        "target_native_session_id": session_id,
        "interleaved_decode": False,
        "timing": {
            "primary_latency_field": "generation_wall_seconds",
            "generation_wall_seconds": generation_wall_seconds,
            "generation_cuda_seconds": generation_cuda_seconds,
            "target_prompt_encode_seconds": prompt_encode_seconds,
            "model_load_included": False,
            "noise_materialization_included": False,
            "decode_included": False,
            "video_encode_included": False,
            "vbench_included": False,
            "per_chunk": chunk_timings,
        },
    }
    return TStitchGeneration(
        history=history,
        generated=tuple(generated),
        result=result,
    )


def decode_t_stitch_trajectory(
    *,
    generation: TStitchGeneration,
    decoder: TStitchDraftClient,
    clock: Callable[[], float] = time.perf_counter,
) -> TStitchRollout:
    _require(decoder.role == "draft", "offline VAE decode must use the Draft worker")
    chunk_count = generation.result["chunk_count"]
    _require(type(chunk_count) is int and chunk_count in (30, 60, 120), "invalid decode horizon")
    _require(generation.history.next_chunk_index == chunk_count, "decode history/horizon mismatch")
    prompt_id = str(generation.result["prompt_id"])
    started = clock()
    decode = decoder.decode(
        history=generation.history,
        output_frames_path=f"decoded/{METHOD_ID}/{prompt_id}.safetensors",
        output_video_path=f"videos/{METHOD_ID}/{prompt_id}.mp4",
        fps=FPS,
    )
    elapsed = _finite_nonnegative(clock() - started, "T-Stitch decode wall time")
    _require(decode.video is not None, "T-Stitch decoder did not emit an MP4")
    video = dict(decode.video)
    _require(video.get("fps") == FPS, "T-Stitch video FPS drifted")
    _require(video.get("frames") == chunk_count * 12 - 3, "T-Stitch frame count drifted")
    timing = dict(generation.result["timing"])
    timing["vae_decode_and_video_encode_wall_seconds"] = elapsed
    return TStitchRollout(
        history=generation.history,
        generated=generation.generated,
        decode=decode,
        result={**dict(generation.result), "timing": timing, "video": video},
    )
