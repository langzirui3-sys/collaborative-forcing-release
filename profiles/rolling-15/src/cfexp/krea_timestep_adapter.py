"""Explicit upstream schedule adapter; legacy Target source and gates stay unchanged.

The three ordinary generation methods retain the pinned legacy operator logic.
Only their schedule selection/validation is different; an AST regression test
checks this narrow delta. There is no module-global patch or cache transfer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

from . import krea_chunk_backend as legacy
from .krea_chunk_backend import (KREA_FOUR_STEP_SCHEDULE, MAIN_TABLE_FOUR_STEP_SCHEDULE,
    _require, copy_explicit_session_noise, validate_target_forward_trace, validate_native_correction_state)
from .self_forcing_frame_adapter import (AdapterError, attach_adapter_timing,
    timed_forward_stages, sampled_kv_cache_witness, witness_index_execution_payload)
from .stage_timing import stage

LEGACY_SOURCE_SHA256 = "70f251edb7dab8eddb9a80bc731fa4165eb98112f151244a21e69a9325feb9c0"

TIMESTEP_POLICY_ENV = "CFEXP_KREA_TIMESTEP_POLICY"
COMPARISON_TIMESTEP_POLICY = "main_comparison_v1"
UPSTREAM_TIMESTEP_POLICY = "upstream_release_four_v1"


def target_timestep_policy(environment: Mapping[str, str] | None = None) -> str:
    env = os.environ if environment is None else environment
    policy = env.get(TIMESTEP_POLICY_ENV, COMPARISON_TIMESTEP_POLICY)
    if policy not in (COMPARISON_TIMESTEP_POLICY, UPSTREAM_TIMESTEP_POLICY):
        raise AdapterError("unknown explicit Krea timestep policy")
    if policy == UPSTREAM_TIMESTEP_POLICY:
        flags = ("CFEXP_SF_TARGET_TWO_STEP", "CFEXP_SF_TARGET_BURST_INTERIOR_ONE_STEP",
                 "CFEXP_SF_TARGET_BURST_EXIT_TWO_STEP", "CFEXP_SF_TARGET_LOW_NOISE_REFINEMENT")
        if (any(env.get(k, "0") != "0" for k in flags)
                or env.get("CFEXP_SF_TARGET_EARLY_NATIVE_BURSTS", "0") != "0"):
            raise AdapterError("upstream Krea schedule cannot use shortened Target updates")
    return policy


def target_four_step_schedule(environment: Mapping[str, str] | None = None) -> tuple[float, ...]:
    return (KREA_FOUR_STEP_SCHEDULE if target_timestep_policy(environment) == UPSTREAM_TIMESTEP_POLICY
            else MAIN_TABLE_FOUR_STEP_SCHEDULE)


def configure_target_session_schedule(session: Any) -> None:
    """Keep the upstream session's own schedule in the explicit diagnostic mode.

    Historical/default comparisons retain their original uniform schedule. The
    opt-in must match the pinned constructor; it never fabricates a replacement
    upstream schedule or changes the selected four values after construction.
    """
    if target_timestep_policy() == COMPARISON_TIMESTEP_POLICY:
        session.denoising_step_list = list(MAIN_TABLE_FOUR_STEP_SCHEDULE)
    else:
        actual = tuple(float(v) for v in session.denoising_step_list)
        if actual != KREA_FOUR_STEP_SCHEDULE:
            raise AdapterError("pinned upstream four-step session schedule differs")


def require_comparison_timestep_policy(action: str) -> None:
    if target_timestep_policy() != COMPARISON_TIMESTEP_POLICY:
        raise AdapterError("upstream Krea schedule does not support partial Target action: " + action)


def start_persistent_trajectory(
    self, *, prompt_id: str, prompt: str, seed: int, chunk_count: int
) -> dict[str, Any]:
    """Create one native Krea session and encode its prompt outside timing.

    This path is reserved for the Full 14B Target main-table reference.  It
    keeps the official GenerationSession alive for the full 120 blocks and
    uses the backend's latent-only decoder, so measured generation never
    includes a real VAE decode or full-history transport.
    """

    torch = self.torch
    attach_adapter_timing(torch, self.device)
    _require(self._persistent_session is None, "a persistent Target session is active")
    # The model owns one KV workspace. A new explicit trajectory supersedes
    # the correction session; the two APIs must never share live cursors.
    self._correction_session = None
    self._correction_prompt = None
    self._correction_session_id = None
    _require(isinstance(prompt_id, str) and bool(prompt_id), "prompt_id is invalid")
    _require(isinstance(prompt, str) and bool(prompt.strip()), "prompt is invalid")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "persistent Target seed is invalid",
    )
    _require(chunk_count == self.chunk_count, "persistent Target horizon drifted")
    params = self.release_server.GenerateParams(
        prompt=prompt,
        width=832,
        height=480,
        seed=seed,
        context_noise=0.0,
        keep_first_frame=True,
        kv_cache_num_frames=3,
        num_blocks=chunk_count,
        num_denoising_steps=4,
        timestep_shift=5.0,
    )
    with stage("session_init", detail="target", cuda=True):
        session = self.release_server.GenerationSession(
            params,
            self.config,
            frame_callback=lambda *_args, **_kwargs: None,
            models=self.models,
        )
    torch.cuda.synchronize(self.device)
    prompt_started = time.perf_counter()
    with stage("conditioning", detail="target", cuda=True):
        condition = self._condition(prompt)
    torch.cuda.synchronize(self.device)
    prompt_encode_seconds = time.perf_counter() - prompt_started
    session.conditional_dict = condition
    session.current_prompt_embeds = condition["prompt_embeds"]
    configure_target_session_schedule(session)
    session_id = hashlib.sha256(
        json.dumps(
            ["main-table-krea-native-session-v1", prompt_id, prompt, seed, chunk_count],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    self._persistent_session = session
    self._persistent_session_id = session_id
    self._persistent_prompt = prompt
    self._persistent_seed = seed
    return {
        "session_id": session_id,
        "prompt_encode_seconds": prompt_encode_seconds,
    }


def generate_persistent_chunk(
    self,
    *,
    prompt: str,
    initial_noise: Any,
    internal_seed: int,
    chunk_index: int,
) -> dict[str, Any]:
    """Generate the next latent block in the active native Krea session."""

    torch = self.torch
    attach_adapter_timing(torch, self.device)
    session = self._persistent_session
    _require(session is not None, "no persistent Target session is active")
    _require(prompt == self._persistent_prompt, "persistent Target prompt drifted")
    _require(
        isinstance(chunk_index, int)
        and not isinstance(chunk_index, bool)
        and chunk_index == session.block_idx
        and 0 <= chunk_index < self.chunk_count,
        "persistent Target chunk order drifted",
    )
    _require(
        isinstance(internal_seed, int)
        and not isinstance(internal_seed, bool)
        and internal_seed >= 0,
        "persistent Target internal seed is invalid",
    )
    _require(
        isinstance(initial_noise, torch.Tensor)
        and tuple(initial_noise.shape) == (1, 3, 16, 60, 104),
        "persistent Target initial noise has an invalid shape",
    )
    history_frames = 3 * chunk_index
    with stage("transfer", detail="target_noise_h2d", cuda=True):
        noise = initial_noise.to(device=self.device, dtype=torch.bfloat16).contiguous()
    with stage("transfer", detail="target_noise_session_copy", cuda=True):
        copy_explicit_session_noise(
            session=session,
            noise=noise,
            history_frames=history_frames,
            torch_module=torch,
        )
    random.seed(internal_seed)
    torch.manual_seed(internal_seed)
    torch.cuda.manual_seed_all(internal_seed)
    session.rnd.manual_seed(internal_seed)

    trace: list[dict[str, Any]] = []
    witness: tuple[str, dict[str, Any]] | None = None

    def pre_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
        nonlocal witness
        timestep = kwargs.get("timestep")
        current_start = kwargs.get("current_start")
        if timestep is None or isinstance(current_start, bool) or not isinstance(current_start, int):
            raise AdapterError(
                "persistent Target audit hook did not receive timestep/current_start"
            )
        values = [
            float(value)
            for value in timestep.detach().reshape(-1).to("cpu").tolist()
        ]
        trace.append(
            {
                "timestep": values[0],
                "batch_timesteps_equal": len(set(values)) == 1,
                "current_start": current_start,
            }
        )
        if values and values[0] > 0.0 and witness is None:
            with stage("kv_witness", detail="target", cuda=True):
                witness = self._kv_witness_sampler(
                    self.models.pipeline.kv_cache1,
                    getattr(self.models.pipeline, "crossattn_cache", None),
                )

    handle = self.models.transformer.register_forward_pre_hook(
        pre_hook, prepend=False, with_kwargs=True
    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    try:
        start_event.record()
        with timed_forward_stages(
            self.models.transformer, prefill_count=int(bool(history_frames)),
        ), torch.inference_mode():
            session.generate_block(self.models)
        end_event.record()
        end_event.synchronize()
    finally:
        handle.remove()
    generation_cuda_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
    _require(
        math.isfinite(generation_cuda_seconds) and generation_cuda_seconds >= 0.0,
        "persistent Target CUDA time is invalid",
    )
    schedule = [float(value) for value in session.denoising_step_list]
    denoising_trace = validate_target_forward_trace(
        trace,
        schedule,
        history_frames=history_frames,
        expected_schedule=target_four_step_schedule(),
    )
    _require(witness is not None, "persistent Target did not capture a KV witness")
    candidate = session.last_pred
    _require(
        isinstance(candidate, torch.Tensor)
        and tuple(candidate.shape) == (1, 3, 16, 60, 104),
        "persistent Target generated an unexpected latent shape",
    )
    with stage("transfer", detail="target_result_d2h", cuda=True):
        candidate = candidate.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
    _require(
        bool(torch.isfinite(candidate.float()).all().item()),
        "persistent Target generated non-finite latent values",
    )
    _require(
        session.block_idx == chunk_index + 1
        and session.current_start_frame == 3 * (chunk_index + 1),
        "persistent Target session cursor drifted",
    )
    return {
        "latent": candidate,
        "forward_trace": trace,
        "denoising_forward_trace": denoising_trace,
        "denoising_schedule": schedule,
        "explicit_initial_noise_verified": True,
        "native_session_id": self._persistent_session_id,
        "native_session_block_index": chunk_index,
        "session_recreated": False,
        "history_reconstructed": False,
        "vae_decode_performed": False,
        "generation_cuda_seconds": generation_cuda_seconds,
        "kv_witness_sha256": witness[0],
        "kv_witness": witness[1],
        **witness_index_execution_payload(witness),
    }


def _generate_correction(
    self,
    *,
    prompt: str,
    history_latents: Any | None,
    initial_noise: Any,
    internal_seed: int,
    chunk_index: int,
    first_update_only: bool = False,
) -> dict[str, Any]:
    if first_update_only:
        require_comparison_timestep_policy("first_update")
    torch = self.torch
    attach_adapter_timing(torch, self.device)
    if not isinstance(prompt, str) or not prompt.strip():
        raise AdapterError("prompt text must be non-empty")
    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or not 0 <= chunk_index < self.chunk_count
    ):
        raise AdapterError(
            f"chunk_index must be in [0,{self.chunk_count - 1}]"
        )
    if isinstance(internal_seed, bool) or not isinstance(internal_seed, int) or internal_seed < 0:
        raise AdapterError("internal_seed must be non-negative")
    if not isinstance(initial_noise, torch.Tensor) or tuple(initial_noise.shape) != (1, 3, 16, 60, 104):
        raise AdapterError("Target initial noise must have shape [1,3,16,60,104]")
    history_frames = 3 * chunk_index
    if chunk_index == 0:
        history = None
        if history_latents is not None and int(history_latents.shape[1]) != 0:
            raise AdapterError("Target chunk 0 requires empty history")
    else:
        expected = (1, history_frames, 16, 60, 104)
        if not isinstance(history_latents, torch.Tensor) or tuple(history_latents.shape) != expected:
            raise AdapterError(f"Target history must have shape {expected}")
        with stage("transfer", detail="target_history_h2d", cuda=True):
            history = history_latents.to(device=self.device, dtype=torch.bfloat16).contiguous()
    with stage("transfer", detail="target_noise_h2d", cuda=True):
        noise = initial_noise.to(device=self.device, dtype=torch.bfloat16).contiguous()
    params = self.release_server.GenerateParams(
        prompt=prompt,
        width=832,
        height=480,
        seed=internal_seed,
        context_noise=0.0,
        keep_first_frame=True,
        kv_cache_num_frames=3,
        num_blocks=self.chunk_count,
        num_denoising_steps=4,
        timestep_shift=5.0,
    )
    random.seed(internal_seed)
    torch.manual_seed(internal_seed)
    torch.cuda.manual_seed_all(internal_seed)
    _require(self._persistent_session is None,
             "ordinary correction cannot overlap an explicit Target trajectory")
    session = getattr(self, "_correction_session", None)
    recreated = session is None or self._correction_prompt != prompt or chunk_index == 0
    if recreated:
        with stage("session_init", detail="target", cuda=True):
            session = self.release_server.GenerationSession(
                params,
                self.config,
                frame_callback=lambda *_args, **_kwargs: None,
                models=self.models,
            )
        self._correction_session_counter = getattr(self, "_correction_session_counter", 0) + 1
        self._correction_session_id = hashlib.sha256(json.dumps(
            [os.getpid(), prompt, internal_seed, self._correction_session_counter],
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        self._correction_session = session
        self._correction_prompt = prompt
    # Bind the session to the caller's ACCEPTED prefix, not the previous
    # proposal. This also handles rejected corrections and backwards forks.
    # Native generate_block rebases only its three-frame clean context.
    with stage("history_bind", detail="target", cuda=True), torch.inference_mode():
        if history is not None:
            _require(bool(torch.isfinite(history.float()).all().item()),
                     "Target accepted history contains non-finite values")
            destination = session.all_latents[:, :history_frames]
            destination.copy_(history)
            _require(bool(torch.equal(destination, history)),
                     "Target session did not preserve the accepted history")
        session.block_idx = chunk_index
        session.current_start_frame = history_frames
        session.resume_latents = None
    # Both execution paths use the same explicitly selected policy.
    configure_target_session_schedule(session)
    with stage("transfer", detail="target_noise_session_copy", cuda=True):
        copy_explicit_session_noise(
            session=session,
            noise=noise,
            history_frames=history_frames,
            torch_module=torch,
        )
    session.rnd.manual_seed(internal_seed)
    with stage("conditioning", detail="target", cuda=True):
        condition = self._condition(prompt)
    session.conditional_dict = condition
    session.current_prompt_embeds = condition["prompt_embeds"]

    trace: list[dict[str, Any]] = []
    witness: tuple[str, dict[str, Any]] | None = None

    def pre_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
        nonlocal witness
        timestep = kwargs.get("timestep")
        current_start = kwargs.get("current_start")
        if timestep is None or isinstance(current_start, bool) or not isinstance(current_start, int):
            raise AdapterError("Target audit hook did not receive timestep/current_start")
        values = [float(value) for value in timestep.detach().reshape(-1).to("cpu").tolist()]
        trace.append(
            {
                "timestep": values[0],
                "batch_timesteps_equal": len(set(values)) == 1,
                "current_start": current_start,
            }
        )
        if values and values[0] > 0.0 and witness is None:
            with stage("kv_witness", detail="target", cuda=True):
                witness = sampled_kv_cache_witness(
                    self.models.pipeline.kv_cache1,
                    getattr(self.models.pipeline, "crossattn_cache", None),
                )

    handle = self.models.transformer.register_forward_pre_hook(
        pre_hook, prepend=False, with_kwargs=True
    )
    try:
        with timed_forward_stages(
            self.models.transformer, prefill_count=int(bool(history_frames)),
        ), torch.inference_mode():
            if first_update_only:
                from .target_prefix import run_first_update
                prefix_prediction = run_first_update(session=session, models=self.models,
                    initial_noise=noise, torch=torch, device=self.device, history_frames=history_frames)
            else:
                session.generate_block(self.models)
    finally:
        handle.remove()
    schedule = [float(value) for value in session.denoising_step_list]
    if first_update_only:
        from .target_prefix import validate_prefix_trace
        denoising_trace = validate_prefix_trace(trace, history_frames=history_frames)
        schedule = [1000.0]
    else:
        denoising_trace = validate_target_forward_trace(
            trace, schedule, history_frames=history_frames,
            expected_schedule=target_four_step_schedule(),
        )
    if witness is None:
        raise AdapterError("Target did not capture a pre-denoising KV witness")
    candidate = prefix_prediction if first_update_only else session.last_pred
    if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != (1, 3, 16, 60, 104):
        raise AdapterError("Target generated an unexpected latent shape")
    with stage("transfer", detail="target_result_d2h", cuda=True):
        candidate = candidate.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if not bool(torch.isfinite(candidate.float()).all().item()):
        raise AdapterError("Target generated non-finite latent values")
    result = {
        "latent": candidate,
        "kv_witness_sha256": witness[0],
        "kv_witness": witness[1],
        **witness_index_execution_payload(witness),
        "forward_trace": trace,
        "denoising_forward_trace": denoising_trace,
        "denoising_schedule": schedule,
        "explicit_initial_noise_verified": True,
    }
    result["native_correction_state"] = {
        "schema": "krea_native_correction_state_v1",
        "native_session_id": self._correction_session_id,
        "session_recreated": recreated,
        "chunk_index": chunk_index,
        "accepted_history_frames": history_frames,
        "bounded_context_rebase": bool(history_frames),
        "context_frames": min(history_frames, 3),
    }
    validate_native_correction_state(result["native_correction_state"], chunk_index=chunk_index)
    if first_update_only:
        result['first_clean_prediction'] = result.pop('latent')
        result['completed_target_chunk'] = False
        result['handoff_representation'] = 'clean_prediction_requires_receiver_native_renoise'
    return result


def unsupported_partial(self, *args, **kwargs):
    raise AdapterError("upstream Krea schedule does not support partial Target action")


def install(backend):
    """Install only on a cold verified native owner, without touching class/globals."""
    if (target_timestep_policy()!=UPSTREAM_TIMESTEP_POLICY
            or type(backend) is not legacy.KreaChunkBackend
            or getattr(backend,'_persistent_session',None) is not None
            or getattr(backend,'_correction_session',None) is not None
            or getattr(backend,'_upstream_timestep_adapter',False)):
        raise AdapterError('upstream schedule adapter requires its explicit cold native owner')
    path=Path(legacy.__file__)
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=LEGACY_SOURCE_SHA256:
        raise AdapterError('legacy Target source changed; a new upstream adapter proof is required')
    state=getattr(backend,'model_state',None)
    if not isinstance(state,Mapping) or state.get('checkpoint_sha256')!=legacy.KREA_CHECKPOINT_SHA256:
        raise AdapterError('upstream adapter lacks the verified original Krea checkpoint')
    methods={name:MethodType(fn,backend) for name,fn in (
        ('start_persistent_trajectory',start_persistent_trajectory),
        ('generate_persistent_chunk',generate_persistent_chunk),
        ('_generate_correction',_generate_correction),
        ('generate_persistent_suffix_chunk',unsupported_partial))}
    for name,method in methods.items():setattr(backend,name,method)
    backend.model_state={**dict(state),'timestep_policy':UPSTREAM_TIMESTEP_POLICY,
        'denoising_schedule':list(KREA_FOUR_STEP_SCHEDULE),'legacy_source_sha256':LEGACY_SOURCE_SHA256,
        'timestep_adapter_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    backend._upstream_timestep_adapter=True


def build_backend(config):
    from .correction_runtime import build_target_backend
    backend=build_target_backend(config)
    try:install(backend)
    except BaseException:
        backend.close()
        raise
    return backend


def process_factory(argv, **kwargs):
    from .tracked_sdvg import process_factory as observed
    command=list(argv)
    role=command[command.index('--role')+1]
    if target_timestep_policy()==UPSTREAM_TIMESTEP_POLICY and role=='target':
        index=command.index('--backend-factory')+1
        if command[index]!='cfexp.correction_runtime:build_target_backend':
            raise ValueError('upstream schedule requires the verified original Target builder')
        command[index]='cfexp.krea_timestep_adapter:build_backend'
    return observed(command,**kwargs)
