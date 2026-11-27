"""Capture rank and cache evidence for the opt-in Self-Forcing handoff."""

from __future__ import annotations

import hashlib
import os
import random
from types import MethodType
from typing import Any, Callable, Mapping, Sequence

from .self_forcing_frame_adapter import (
    AdapterError, EffectiveRankMeasurement, EFFECTIVE_RANK_LAYER, LAYER_INDICES,
    SELF_FORCING_TOKENS_PER_CHUNK, StepZeroCapture, attach_adapter_timing,
    effective_rank_evidence_from_capture, sampled_kv_cache_witness, stage,
    timed_forward_stages, witness_index_execution_payload,
)
from .sf_joint_handoff import active_handoff


def build_backend(config):
    from .correction_runtime import build_draft_backend

    backend = build_draft_backend(config)
    try:
        install(backend)
    except BaseException:
        backend.close()
        raise
    return backend


def process_factory(argv, **kwargs):
    from .accepted_stream_worker import process_factory as launch

    argv = list(argv)
    if argv[argv.index('--role') + 1] == 'draft':
        index = argv.index('--backend-factory') + 1
        if argv[index] != 'cfexp.correction_runtime:build_draft_backend':
            raise ValueError('joint handoff requires the native Self-Forcing Draft')
        argv[index] = 'cfexp.sf_joint_handoff_adapter:build_backend'
    return launch(argv, **kwargs)


def install(backend):
    """Bind the joint path after the native model and executor have loaded."""
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend

    if (type(backend) is not OfficialSelfForcingBackend or backend._incremental_draft is None
            or os.environ.get('CFEXP_SF_SINGLE_GPU') != '1'):
        raise AdapterError('joint handoff requires single-GPU Self-Forcing native KV reuse')
    if backend._incremental_draft.joint_handoff:
        raise AdapterError('joint handoff is already installed')
    for handle in backend._hook_handles:
        handle.remove()
    backend._hook_handles.clear()
    install_capture_hooks(backend, backend.pipeline.generator.model.blocks)
    backend.generate_chunk = MethodType(generate_chunk, backend)
    backend._incremental_draft.joint_handoff = True


def install_capture_hooks(self, blocks: Sequence[Any]) -> None:
    def generator_pre_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
        if self._active_capture is None:
            raise AdapterError("generator ran without an active prompt capture")
        timestep = kwargs.get("timestep")
        current_start = kwargs.get("current_start")
        if (
            timestep is None
            or isinstance(current_start, bool)
            or not isinstance(current_start, int)
        ):
            raise AdapterError("native generator call did not expose timestep/current_start")
        values = self._timestep_values(timestep)
        handoff = active_handoff()
        if handoff is not None:
            draft_start, chunks = handoff
            if (values != [0.0] * (3 * chunks) + [1000.0] * 3
                    or current_start != draft_start - chunks * SELF_FORCING_TOKENS_PER_CHUNK):
                raise AdapterError("joint handoff used unexpected timestep or positions")
            current_start, values = draft_start, values[-3:]
        self._active_capture.begin_call(values, current_start=current_start)

    def generator_post_hook(
        _module: Any, _args: Any, _kwargs: Mapping[str, Any], output: Any
    ) -> Any:
        if self._active_capture is None:
            raise AdapterError("generator ended without an active prompt capture")
        self._active_capture.end_call()
        return output

    self._hook_handles.append(
        self.pipeline.generator.register_forward_pre_hook(
            generator_pre_hook, prepend=True, with_kwargs=True
        )
    )
    self._hook_handles.append(
        self.pipeline.generator.register_forward_hook(
            generator_post_hook, prepend=False, with_kwargs=True
        )
    )
    for layer_index in LAYER_INDICES:
        def layer_hook(
            _module: Any,
            _args: Any,
            output: Any,
            *,
            index: int = layer_index,
        ) -> None:
            if self._active_capture is None:
                raise AdapterError("transformer block ran without an active prompt capture")
            if active_handoff() is not None:
                output = output[:, -SELF_FORCING_TOKENS_PER_CHUNK:]
            self._active_capture.record_layer(index, output)

        self._hook_handles.append(blocks[layer_index].register_forward_hook(layer_hook))

def generate_chunk(
    self,
    *,
    prompt: str,
    history_latents: Any | None,
    initial_noise: Any,
    internal_seed: int,
    chunk_index: int,
    capture_hidden: bool,
    kv_branch_action: str = "none",
    history_kv_widths: tuple[int, ...] | None = None,
    early_should_replace: Callable[[EffectiveRankMeasurement], bool] | None = None,
) -> dict[str, Any]:
    torch = self.torch
    rank_index_helper = getattr(self, "rank_indices", None)
    rank_index_before = (rank_index_helper.snapshot()
        if rank_index_helper is not None and rank_index_helper.enabled else None)
    attach_adapter_timing(torch, self.device)
    if early_should_replace is not None or capture_hidden:
        raise AdapterError("joint handoff requires committed Draft eRank routing")
    if kv_branch_action not in ("none", "preserve", "consume"):
        raise AdapterError("unknown Draft KV branch action")
    if kv_branch_action != "none" and not getattr(
        getattr(self, "_incremental_draft", None), "fork_enabled", False
    ):
        raise AdapterError("Draft KV branch action requires fork-enabled reuse")
    if not isinstance(prompt, str) or not prompt.strip():
        raise AdapterError("prompt text must be non-empty")
    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or not 0 <= chunk_index < self.chunk_count
    ):
        raise AdapterError(
            f"chunk_index must be in [0, {self.chunk_count - 1}]"
        )
    if isinstance(internal_seed, bool) or not isinstance(internal_seed, int) or internal_seed < 0:
        raise AdapterError("internal_seed must be a non-negative integer")
    expected_noise_shape = (1, 3, 16, 60, 104)
    if not isinstance(initial_noise, torch.Tensor) or tuple(initial_noise.shape) != expected_noise_shape:
        raise AdapterError(f"initial_noise must have shape {expected_noise_shape}")
    expected_history_frames = 3 * chunk_index
    if chunk_index == 0:
        if history_latents is not None and int(history_latents.shape[1]) != 0:
            raise AdapterError("chunk 0 requires an empty latent history")
        history = None
    else:
        expected_history_shape = (1, expected_history_frames, 16, 60, 104)
        if not isinstance(history_latents, torch.Tensor) or tuple(history_latents.shape) != expected_history_shape:
            raise AdapterError(f"history_latents must have shape {expected_history_shape}")
        with stage("transfer", detail="draft_history_h2d", cuda=True):
            history = history_latents.to(device=self.device, dtype=torch.bfloat16).contiguous()
    with stage("transfer", detail="draft_noise_h2d", cuda=True):
        noise = initial_noise.to(device=self.device, dtype=torch.bfloat16).contiguous()

    random.seed(internal_seed)
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise AdapterError("the frozen Self-Forcing runtime is missing NumPy") from error
    np.random.seed(internal_seed % (2**32))
    torch.manual_seed(internal_seed)
    torch.cuda.manual_seed_all(internal_seed)

    self._active_capture = StepZeroCapture(
        layers=LAYER_INDICES,
        target_timestep=1000,
        sample_hidden=self._sample_hidden,
        rank_sample=self._rank_sample,
        retain_layer=14 if capture_hidden else None,
        retain_full_hidden=self._retain_full_hidden if capture_hidden else None,
        retain_rank_layer=EFFECTIVE_RANK_LAYER,
        policy_rank_only=not capture_hidden,
        early_should_replace=early_should_replace,
    )
    forward_trace: list[dict[str, Any]] = []
    kv_witness: tuple[str, dict[str, Any]] | None = None

    def audit_pre_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
        nonlocal kv_witness
        timestep = kwargs.get("timestep")
        current_start = kwargs.get("current_start")
        if timestep is None or isinstance(current_start, bool) or not isinstance(current_start, int):
            raise AdapterError("generator audit hook did not receive timestep/current_start")
        values = self._timestep_values(timestep)
        handoff = active_handoff()
        if handoff is not None:
            forward_trace.append({
                "timestep": values[-1], "batch_timesteps_equal": False,
                "current_start": current_start, "fused_history_chunks": handoff[1],
            })
            return
        forward_trace.append(
            {
                "timestep": values[0],
                "batch_timesteps_equal": len(set(values)) == 1,
                "current_start": current_start,
            }
        )
        if values and len(set(values)) == 1 and values[0] == 1000.0 and kv_witness is None:
            with stage("kv_witness", detail="draft", cuda=True):
                kv_witness = sampled_kv_cache_witness(
                    self.pipeline.kv_cache1,
                    getattr(self.pipeline, "crossattn_cache", None),
                )

    audit_handle = self.pipeline.generator.register_forward_pre_hook(
        audit_pre_hook, prepend=False, with_kwargs=True
    )

    def audit_post_hook(_module: Any, _args: Any, _output: Any) -> None:
        nonlocal kv_witness
        handoff = active_handoff()
        if handoff is not None:
            if kv_witness is not None or any(
                layer["global_end_index"].item() != handoff[0] for layer in self.pipeline.kv_cache1
            ):
                raise AdapterError("joint handoff did not retain the exact clean KV position")
            with stage("kv_witness", detail="draft", cuda=True):
                kv_witness = sampled_kv_cache_witness(
                    self.pipeline.kv_cache1, self.pipeline.crossattn_cache,
                )

    audit_post_handle = self.pipeline.generator.register_forward_hook(audit_post_hook)
    real_decode = self.pipeline.vae.decode_to_pixel

    def latent_only_decode(latent: Any, use_cache: bool = False) -> Any:
        del use_cache
        return torch.zeros(
            (int(latent.shape[0]), 1, 3, 1, 1),
            device=latent.device,
            dtype=torch.float32,
        )

    self.pipeline.vae.decode_to_pixel = latent_only_decode
    try:
        reuse_evidence = None
        executor = getattr(self, "_incremental_draft", None)
        if executor is not None:
            incremental = executor.generate(prompt=prompt, history=history, noise=noise,
                history_kv_widths=history_kv_widths,
                **({"kv_branch_action": kv_branch_action} if kv_branch_action != "none" else {}),
                forward_context=lambda count: timed_forward_stages(
                    self.pipeline.generator, prefill_count=count, cache_update_count=1))
            clean = incremental["latent"]
            output = clean if history is None else torch.cat([history, clean], dim=1)
            reuse_evidence = {
                "schema": "draft_kv_reuse_v2",
                "history_kv_widths": list(incremental["history_kv_widths"]),
                "reused_chunks": incremental["reused_chunks"],
                "replayed_chunks": incremental["replayed_chunks"],
                "scope_sha256": incremental["scope_sha256"],
                "history_tensor_sha256": list(incremental["history_binding"]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                **({"branch_anchor": incremental["branch_anchor"]}
                   if "branch_anchor" in incremental else {}),
            }
        else:
            raise AdapterError("joint handoff requires an incremental Draft executor")
        events = list(self._active_capture.events)
    finally:
        self.pipeline.vae.decode_to_pixel = real_decode
        audit_handle.remove()
        audit_post_handle.remove()
        self._active_capture = None
    if len(events) != 1:
        raise AdapterError(f"one-chunk Draft request produced {len(events)} step-0 captures")
    event = events[0]
    expected_start = expected_history_frames * 1560
    if event.current_start != expected_start:
        raise AdapterError(
            f"Draft chunk {chunk_index} used current_start={event.current_start}, expected {expected_start}"
        )
    positive_calls = [item for item in forward_trace if float(item["timestep"]) > 0.0]
    if len(positive_calls) != 4:
        raise AdapterError(f"Draft chunk used {len(positive_calls)} denoising forwards, expected 4")
    if kv_witness is None:
        raise AdapterError("Draft chunk did not capture its pre-denoising KV witness")
    expected_output_shape = (1, expected_history_frames + 3, 16, 60, 104)
    if tuple(output.shape) != expected_output_shape:
        raise AdapterError(f"Draft output shape {tuple(output.shape)} != {expected_output_shape}")
    if history is not None and not torch.equal(output[:, :expected_history_frames], history):
        raise AdapterError("Draft output did not preserve the exact input history")
    with stage("transfer", detail="draft_result_d2h", cuda=True):
        candidate = output[:, -3:].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if not bool(torch.isfinite(candidate.float()).all().item()):
        raise AdapterError("Draft generated non-finite latent values")
    rank_evidence = effective_rank_evidence_from_capture(event)
    return {
        **({"kv_reuse": reuse_evidence} if reuse_evidence is not None else {}),
        "latent": candidate,
        "rank": event.ranks[14],
        "rank_evidence": rank_evidence,
        "sampled_hidden": (
            event.retained_sample.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            if event.retained_sample is not None
            else None
        ),
        "full_hidden": event.retained_full_hidden,
        "kv_witness_sha256": kv_witness[0],
        "kv_witness": kv_witness[1],
        **witness_index_execution_payload(kv_witness),
        **({"rank_index_reuse": rank_index_helper.receipt(rank_index_before)}
            if rank_index_before is not None else {}),
        "forward_trace": forward_trace,
    }

