"""Random block placement with Self-Forcing's native accepted history."""

from types import SimpleNamespace

from .correction_policy import random_positions
from .correction_state import LatentHistory
from .insight2_campaign import ChunkGenerationRequest, PromptCase
from .krea_chunk_backend import validate_native_correction_state
from .main_table_full_target import _canonical_sha256, SEED, TARGET_MODEL_ID
from .native_video_ready_draft import SELF_FORCING

PROTOCOL_ID = "insight2-correction-long60-h800-lookahead-boundary-v1"
ROUTING_SEED = 0


def target_positions(prompt_id, profile):
    return (0, *random_positions(
        protocol_id=PROTOCOL_ID, prompt_id=prompt_id, generation_seed=SEED,
        budget=profile.proposal_budget, repeat_id=ROUTING_SEED, chunk_count=profile.chunk_count,
    ))


def verify_draft_suffix(state, *, index, target_chunks, cached_through):
    widths = [1 if j in target_chunks else 3 for j in range(index)]
    if (state.get("schema") != "draft_kv_reuse_v2"
            or state.get("history_kv_widths") != widths
            or state.get("reused_chunks") != cached_through
            or state.get("replayed_chunks") != index - cached_through):
        raise ValueError("native Random replayed beyond the newly accepted Target history")


def generate(*, draft, target, noise_provider, prompt_id, prompt, profile, renderer, run_dir):
    if (draft.role, draft.model_id, target.role, target.model_id) != (
        "draft", SELF_FORCING, "target", TARGET_MODEL_ID
    ):
        raise ValueError("native Random requires the Self-Forcing and Krea workers")
    budget = profile.proposal_budget
    target_chunks = set(target_positions(prompt_id, profile))
    scope = _canonical_sha256("native-random-sf-v1", prompt_id, SEED, profile.chunk_count)
    case = PromptCase(prompt_id, prompt, "evaluation")
    history = LatentHistory(scope, prompt_id, SEED)
    states = []
    target_session = None
    cached_through = 0
    for index in range(profile.chunk_count):
        backend = target if index in target_chunks else draft
        noise = noise_provider.materialize(
            prompt_id=prompt_id, trajectory_seed=SEED, chunk_index=index)
        target_call = {}
        if backend is target:
            target_call = {
                "target_logical_call_id": _canonical_sha256(scope, index, history.head_sha256),
                "target_call_scope_id": scope,
                "target_call_phase": "route_materialization",
            }
        path = f"latents/native-random/{prompt_id}/{index:03d}.safetensors"
        request = ChunkGenerationRequest(
            prompt=case, history=history, noise=noise, output_latent_path=path,
            capture_hidden=False, **target_call,
        )
        if index > 0:
            renderer.before_generate(backend.role)
        item = backend.generate(request)
        if (item.context.role != backend.role or item.context.model_id != backend.model_id
                or item.context.noise.descriptor_sha256 != noise.descriptor.descriptor_sha256
                or item.segment.latent_path != path):
            raise ValueError("native Random worker changed the requested chunk")
        item.segment.verify_context(history, item.context)
        item.segment.verify_file(run_dir)
        if backend is target:
            state = item.diagnostics["native_correction_state"]
            validate_native_correction_state(state, chunk_index=index)
            if index == 0:
                target_session = state["native_session_id"]
            if state["native_session_id"] != target_session or (index and state["session_recreated"]):
                raise ValueError("native Random recreated the Target correction session")
        else:
            state = item.diagnostics["kv_reuse"]
            verify_draft_suffix(state, index=index, target_chunks=target_chunks,
                                cached_through=cached_through)
            cached_through = index + 1
        history = history.append(item.segment)
        renderer.append(history)
        states.append(dict(chunk_index=index, role=backend.role, cache=state,
                           latent_sha256=item.segment.latent_sha256,
                           initial_noise_descriptor_sha256=item.segment.initial_noise_descriptor_sha256))
    return SimpleNamespace(history=history, result=dict(
        schema="native_sf_random_trajectory_v1", draft_model_id=SELF_FORCING,
        initialization="target_root_then_fixed_random_positions", prompt_id=prompt_id,
        seed=SEED, routing_repeat=ROUTING_SEED, target_chunks=sorted(target_chunks),
        chunk_count=profile.chunk_count, draft_model_calls=profile.chunk_count-budget-1,
        target_model_calls=budget+1, history_head_sha256=history.head_sha256,
        history_manifest_sha256=history.manifest_sha256, native_cache_states=states,
    ))
