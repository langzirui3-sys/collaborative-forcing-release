"""Paper-faithful SDVG rollout for the frozen main-table benchmark.

SDVG maintains two histories.  The Self-Forcing Draft always advances on its
own proposals so its KV cache and streaming VAE cache remain internally
consistent.  The selected Target history advances with an accepted Draft
proposal or with a same-noise Krea regeneration after rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Protocol

from .correction_runtime import DecodeResult
from .correction_state import LatentHistory, LatentSegment
from .insight2_campaign import (
    ChunkGenerationRequest,
    GeneratedChunk,
    NoiseProvider,
    PromptCase,
)


SCHEMA = "main_table_sdvg_rollout_v1"
METHOD_ID = "sdvg"
DRAFT_MODEL_ID = "self_forcing_1.3b"
TARGET_MODEL_ID = "krea-realtime-video-14b"
SEED = 20260911
CHUNK_COUNT = 120
RGB_FRAME_COUNT = 1437
FPS = 24
THRESHOLD = -0.7
TARGET_RPC_PHASE = "route_materialization"


class SdvgError(RuntimeError):
    """Raised when the frozen SDVG route contract drifts."""


class SdvgDraftClient(Protocol):
    role: str
    model_id: str
    sdvg_supported: bool

    def generate(self, request: ChunkGenerationRequest) -> GeneratedChunk: ...

    def start_sdvg_trajectory(
        self, *, prompt_id: str, prompt: str, chunk_count: int
    ) -> Mapping[str, Any]: ...

    def score_sdvg_candidate(
        self, *, prompt: str, chunk_index: int, clean_latent_path: str
    ) -> Mapping[str, Any]: ...

    def finish_sdvg_trajectory(self, *, prompt_id: str) -> Mapping[str, Any]: ...

    def decode(
        self,
        *,
        history: LatentHistory,
        output_frames_path: str,
        output_video_path: str | None,
        fps: int = 24,
    ) -> DecodeResult: ...


class SdvgTargetClient(Protocol):
    role: str
    model_id: str
    session_policy: str
    decode_during_generation: bool

    def start_trajectory(
        self, *, prompt_id: str, prompt: str, seed: int, chunk_count: int
    ) -> Mapping[str, Any]: ...

    def generate_persistent(self, request: ChunkGenerationRequest) -> GeneratedChunk: ...

    def append_persistent_clean(
        self,
        *,
        prompt: str,
        chunk_index: int,
        history_latent_paths: list[str],
        clean_latent_path: str,
    ) -> Mapping[str, Any]: ...

    def finish_trajectory(self, *, session_id: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SdvgGeneration:
    draft_history: LatentHistory
    selected_history: LatentHistory
    draft_generated: tuple[GeneratedChunk, ...]
    selected_generated: tuple[GeneratedChunk | None, ...]
    result: Mapping[str, Any]


@dataclass(frozen=True)
class SdvgRollout:
    draft_history: LatentHistory
    selected_history: LatentHistory
    decode: DecodeResult
    result: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SdvgError(message)


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
        raise SdvgError(f"{label} is invalid") from error
    _require(math.isfinite(parsed) and parsed >= 0.0, f"{label} is invalid")
    return parsed


def _rechain_accepted_draft(
    *, selected_history: LatentHistory, candidate: GeneratedChunk
) -> LatentSegment:
    """Represent an accepted Draft file under the selected-history parent.

    The KV descriptor remains the digest of the real Draft-local context; only
    the selection-chain parent changes.  This preserves both histories without
    claiming that Draft generation consumed the Target-selected prefix.
    """

    source = candidate.segment
    _require(source.chunk_index == selected_history.next_chunk_index, "chunk index drifted")
    _require(source.producer_role == "draft", "accepted candidate is not Draft-produced")
    return LatentSegment(
        trajectory_id=selected_history.trajectory_id,
        prompt_id=selected_history.prompt_id,
        trajectory_seed=selected_history.trajectory_seed,
        chunk_index=selected_history.next_chunk_index,
        parent_history_sha256=selected_history.head_sha256,
        producer_role=source.producer_role,
        model_id=source.model_id,
        model_artifact_sha256=source.model_artifact_sha256,
        initial_noise_descriptor_sha256=source.initial_noise_descriptor_sha256,
        kv_descriptor_sha256=source.kv_descriptor_sha256,
        latent_path=source.latent_path,
        latent_sha256=source.latent_sha256,
        latent_nbytes=source.latent_nbytes,
    )


def generate_sdvg_trajectory(
    *,
    prompt_id: str,
    prompt: str,
    draft: SdvgDraftClient,
    target: SdvgTargetClient,
    noise_provider: NoiseProvider,
    seed: int = SEED,
    chunk_count: int = CHUNK_COUNT,
    threshold: float = THRESHOLD,
    draft_model_id: str = DRAFT_MODEL_ID,
    clock: Callable[[], float] = time.perf_counter,
    safety_check: Callable[[str, int], None] | None = None,
    on_commit: Callable[[LatentHistory], None] | None = None,
    before_generate: Callable[[str], None] | None = None,
) -> SdvgGeneration:
    """Generate one strict two-history SDVG trajectory."""

    _require(seed == SEED, f"main-table seed must be {SEED}")
    _require(type(chunk_count) is int and chunk_count in (30, 60, 120), "SDVG requires 30, 60, or 120 chunks")
    _require(threshold == THRESHOLD, f"SDVG threshold must be {THRESHOLD}")
    _require(draft_model_id in {DRAFT_MODEL_ID, "longlive_1.3b", "rolling_forcing_1.3b"}, "unsupported SDVG Draft model")
    _require(
        draft.role == "draft"
        and draft.model_id == draft_model_id
        and draft.sdvg_supported is True,
        "SDVG loaded the wrong Draft backend",
    )
    _require(
        target.role == "target"
        and target.model_id == TARGET_MODEL_ID
        and target.session_policy == "persistent_native_session"
        and target.decode_during_generation is False,
        "SDVG loaded an incompatible Target backend",
    )
    case = PromptCase(prompt_id=prompt_id, prompt=prompt, split="evaluation")
    draft_trajectory_id = _canonical_sha256(
        "main-table-sdvg-draft-v1", prompt_id, seed, chunk_count
    )
    selected_trajectory_id = _canonical_sha256(
        "main-table-sdvg-selected-v1", prompt_id, seed, chunk_count
    )
    scope_id = _canonical_sha256(
        "main-table-target-scope-v1", METHOD_ID, prompt_id, seed
    )
    reward_start = draft.start_sdvg_trajectory(
        prompt_id=prompt_id, prompt=prompt, chunk_count=chunk_count
    )
    _require(
        reward_start.get("threshold") == threshold,
        "Draft worker loaded a different SDVG threshold",
    )
    target_start = target.start_trajectory(
        prompt_id=prompt_id, prompt=prompt, seed=seed, chunk_count=chunk_count
    )
    _require(
        isinstance(target_start.get("session_id"), str)
        and bool(target_start["session_id"]),
        "Target session startup evidence is invalid",
    )
    session_id = str(target_start["session_id"])
    prompt_encode_seconds = _finite_nonnegative(
        target_start.get("prompt_encode_seconds"), "Target prompt encode time"
    )
    draft_history = LatentHistory(
        trajectory_id=draft_trajectory_id,
        prompt_id=prompt_id,
        trajectory_seed=seed,
    )
    selected_history = LatentHistory(
        trajectory_id=selected_trajectory_id,
        prompt_id=prompt_id,
        trajectory_seed=seed,
    )
    draft_generated: list[GeneratedChunk] = []
    selected_generated: list[GeneratedChunk | None] = []
    routes: list[dict[str, Any]] = []
    generation_wall_seconds = 0.0

    for chunk_index in range(chunk_count):
        if safety_check is not None:
            safety_check("before_chunk", chunk_index)
        noise = noise_provider.materialize(
            prompt_id=prompt_id, trajectory_seed=seed, chunk_index=chunk_index
        )
        draft_request = ChunkGenerationRequest(
            prompt=case,
            history=draft_history,
            noise=noise,
            output_latent_path=(
                f"latents/{METHOD_ID}/{prompt_id}/chunk-{chunk_index:03d}-draft.safetensors"
            ),
            capture_hidden=False,
        )
        started = clock()
        if before_generate is not None:
            before_generate("draft")
        candidate = draft.generate(draft_request)
        _require(candidate.segment.producer_role == "draft", "Draft returned a Target chunk")
        if draft_model_id == "rolling_forcing_1.3b":
            state = candidate.diagnostics.get("rolling_state")
            _require(isinstance(state, Mapping)
                     and state.get("chunk_index") == chunk_index
                     and state.get("reused_rolling_state") is (chunk_index > 0)
                     and state.get("replayed_history_blocks") == 0
                     and state.get("discarded_future_state") is False,
                     "SDVG Rolling Draft did not preserve its independent native trajectory")
        candidate.segment.verify_context(draft_history, candidate.context)
        draft_history = draft_history.append(candidate.segment)
        reward = draft.score_sdvg_candidate(
            prompt=prompt,
            chunk_index=chunk_index,
            clean_latent_path=candidate.segment.latent_path,
        )
        block_reward = float(reward.get("block_reward"))
        _require(
            math.isfinite(block_reward)
            and float(reward.get("threshold")) == threshold,
            "SDVG reward evidence drifted",
        )
        accepted = chunk_index != 0 and block_reward >= threshold
        _require(
            reward.get("accepted_by_threshold") == (block_reward >= threshold)
            and reward.get("forced_reject") is (chunk_index == 0),
            "SDVG reward decision evidence is inconsistent",
        )
        target_item: GeneratedChunk | None = None
        target_append: Mapping[str, Any] | None = None
        logical_call_id: str | None = None
        # Root is forced Target and there is no accepted renderer stream yet.
        # Later Target generation *and* external-clean appends share its GPU.
        if before_generate is not None and chunk_index > 0:
            before_generate("target")
        if accepted:
            target_append = target.append_persistent_clean(
                prompt=prompt,
                chunk_index=chunk_index,
                history_latent_paths=[
                    item.latent_path for item in selected_history.segments
                ],
                clean_latent_path=candidate.segment.latent_path,
            )
            append_result = target_append.get("result")
            _require(
                isinstance(append_result, Mapping)
                and append_result.get("committed_external_clean_latent") is True
                and append_result.get("native_session_id") == session_id
                and append_result.get("native_session_block_index") == chunk_index,
                "Target did not commit the accepted Draft chunk",
            )
            selected_history = selected_history.append(
                _rechain_accepted_draft(
                    selected_history=selected_history, candidate=candidate
                )
            )
        else:
            logical_call_id = _canonical_sha256(
                "main-table-target-call-v1",
                METHOD_ID,
                prompt_id,
                seed,
                chunk_index,
                selected_history.head_sha256,
                noise.descriptor.descriptor_sha256,
            )
            target_request = ChunkGenerationRequest(
                prompt=case,
                history=selected_history,
                noise=noise,
                output_latent_path=(
                    f"latents/{METHOD_ID}/{prompt_id}/chunk-{chunk_index:03d}-target.safetensors"
                ),
                capture_hidden=False,
                target_logical_call_id=logical_call_id,
                target_call_scope_id=scope_id,
                target_call_phase=TARGET_RPC_PHASE,
            )
            target_item = target.generate_persistent(target_request)
            _require(
                target_item.segment.producer_role == "target"
                and target_item.target_logical_call_id == logical_call_id
                and target_item.diagnostics.get("native_session_id") == session_id
                and target_item.diagnostics.get("native_session_block_index")
                == chunk_index,
                "Target regeneration evidence is invalid",
            )
            target_item.segment.verify_context(selected_history, target_item.context)
            selected_history = selected_history.append(target_item.segment)
        if on_commit is not None:
            on_commit(selected_history)
        if safety_check is not None:
            safety_check("after_chunk", chunk_index)
        chunk_wall_seconds = _finite_nonnegative(
            clock() - started, "SDVG chunk wall time"
        )
        generation_wall_seconds += chunk_wall_seconds
        draft_generated.append(candidate)
        selected_generated.append(target_item)
        routes.append(
            {
                "chunk_index": chunk_index,
                "block_reward": block_reward,
                "threshold": threshold,
                "forced_reject": chunk_index == 0,
                "route": "draft" if accepted else "target",
                "same_initial_noise_verified": (
                    candidate.context.noise.descriptor_sha256
                    == noise.descriptor.descriptor_sha256
                    and (
                        target_item is None
                        or target_item.context.noise.descriptor_sha256
                        == noise.descriptor.descriptor_sha256
                    )
                ),
                # Preserve backend evidence; missing evidence stays missing rather
                # than inferring reuse from a manifest flag or a KV descriptor.
                "draft_cache_evidence": {
                    key: candidate.diagnostics[key]
                    for key in ("kv_reuse", "native_session_id", "native_session_block_index",
                                "session_recreated", "history_reconstructed", "rolling_state")
                    if key in candidate.diagnostics
                },
                "target_cache_evidence": ({
                    key: target_item.diagnostics[key]
                    for key in ("native_session_id", "native_session_block_index",
                                "session_recreated", "history_reconstructed", "kv_witness_sha256",
                                "denoising_schedule", "denoising_forward_trace")
                    if key in target_item.diagnostics
                } if target_item is not None else None),
                "draft_latent_path": candidate.segment.latent_path,
                "selected_latent_path": selected_history.segments[-1].latent_path,
                "selected_latent_sha256": selected_history.segments[-1].latent_sha256,
                "target_logical_call_id": logical_call_id,
                "target_append": target_append,
                "reward": dict(reward),
                "generation_wall_seconds": chunk_wall_seconds,
            }
        )

    reward_finish = draft.finish_sdvg_trajectory(prompt_id=prompt_id)
    target_finish = target.finish_trajectory(session_id=session_id)
    _require(
        reward_finish.get("scored_chunk_count") == chunk_count
        and target_finish.get("generated_block_count") == chunk_count,
        "SDVG trajectory completion evidence is invalid",
    )
    _require(
        draft_history.next_chunk_index == selected_history.next_chunk_index == chunk_count,
        "SDVG histories are incomplete",
    )
    _require(
        routes[0]["route"] == "target"
        and all(item["same_initial_noise_verified"] is True for item in routes),
        "SDVG force-reject or same-noise contract failed",
    )
    result = {
        "schema": SCHEMA,
        "method_id": METHOD_ID,
        "draft_model_id": draft_model_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "chunk_count": chunk_count,
        "threshold": threshold,
        "draft_accept_count": sum(item["route"] == "draft" for item in routes),
        "target_regeneration_count": sum(item["route"] == "target" for item in routes),
        "prompt_encode_seconds": prompt_encode_seconds,
        "timing": {"generation_wall_seconds": generation_wall_seconds},
        "routes": routes,
        "draft_history_manifest_sha256": draft_history.manifest_sha256,
        "selected_history_manifest_sha256": selected_history.manifest_sha256,
    }
    return SdvgGeneration(
        draft_history=draft_history,
        selected_history=selected_history,
        draft_generated=tuple(draft_generated),
        selected_generated=tuple(selected_generated),
        result=result,
    )


def decode_sdvg_trajectory(
    *, generation: SdvgGeneration, decoder: SdvgDraftClient, video_only: bool = False
) -> SdvgRollout:
    """Decode only the final selected trajectory after generation timing closes."""

    chunk_count = generation.result["chunk_count"]
    _require(type(chunk_count) is int and chunk_count in (30, 60, 120), "invalid decode horizon")
    _require(generation.selected_history.next_chunk_index == chunk_count, "decode history/horizon mismatch")
    prompt_id = generation.selected_history.prompt_id
    _require(type(video_only) is bool, "video-only decode switch must be boolean")
    arguments = dict(history=generation.selected_history,
        output_video_path=f"videos/{METHOD_ID}/{prompt_id}.mp4", fps=FPS)
    if video_only:
        decoded = decoder.decode_video(**arguments)
    else:
        decoded = decoder.decode(output_frames_path=f"decoded/{METHOD_ID}/{prompt_id}.safetensors",
                                 **arguments)
    _require(decoded.video is not None, "SDVG final decode omitted the MP4")
    _require(
        decoded.video.get("frames") == chunk_count * 12 - 3
        and decoded.video.get("fps") == FPS,
        "SDVG final video geometry drifted",
    )
    result = {
        **dict(generation.result),
        "frames": dict(decoded.frames),
        "video": dict(decoded.video),
    }
    return SdvgRollout(
        draft_history=generation.draft_history,
        selected_history=generation.selected_history,
        decode=decoded,
        result=result,
    )


__all__ = [
    "CHUNK_COUNT",
    "METHOD_ID",
    "SEED",
    "THRESHOLD",
    "SdvgError",
    "SdvgGeneration",
    "SdvgRollout",
    "decode_sdvg_trajectory",
    "generate_sdvg_trajectory",
]
