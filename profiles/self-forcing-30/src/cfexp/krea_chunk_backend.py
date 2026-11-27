"""Thin latent-only adapter around the pinned official Krea runtime.

The Krea source tree remains an external, digest-pinned runtime artifact.  We
reuse its ``GenerationSession`` instead of copying its serving state machine.
Only model construction is supplied here because the upstream loader
incorrectly requires a second 14B base-weight copy before loading the Krea
checkpoint.  A meta-device model built from the pinned 250-byte Wan config is
strictly assigned the Krea state dict once.

This module imports no ML package at import time so CPU validation and the
formal runner remain lightweight.
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

from .self_forcing_frame_adapter import (
    AdapterError, sampled_kv_cache_witness, attach_adapter_timing, timed_forward_stages,
    witness_index_execution_payload,
)
from .stage_timing import stage


KREA_SOURCE_COMMIT = "acf4b7b4e1a3049a0e04782e751d12203334038d"
KREA_CHECKPOINT_REVISION = "9a2ceea280e5d584c6a28d384a9c19e38c810f8c"
KREA_CHECKPOINT_SHA256 = "792f6042a645e227f5d71851845dc4098a7338fec9ec55279f88ebbc3527eec7"
WAN14B_CONFIG_REVISION = "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"
WAN14B_CONFIG_SHA256 = "7c9d7887442cb46716380bdf4426933d7cf88cbb31abcb434ea219322f5580ad"
KREA_PARAMETER_COUNT = 14_288_491_584
EXPECTED_KREA_STATE_KEYS = 1095
KREA_KV_LAYER_COUNT = 40
KREA_SELF_KV_SHAPE = (1, 9360, 40, 128)
KREA_CROSS_KV_SHAPE = (1, 512, 40, 128)
KREA_KV_DTYPE = "torch.bfloat16"
SUPPORTED_CORRECTION_CHUNK_COUNTS = frozenset({16, 30, 60, 120})
KREA_FOUR_STEP_SCHEDULE = (
    1000.0,
    908.8427124023438,
    713.9794311523438,
    0.0,
)
MAIN_TABLE_FOUR_STEP_SCHEDULE = (1000.0, 750.0, 500.0, 250.0)
MAIN_TABLE_TARGET_SUFFIX_SCHEDULE = (750.0, 500.0, 250.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def install_flash_attention2_compatibility(attention_module: Any) -> dict[str, Any]:
    """Bind the symbol used by the pinned Krea FA2 fast path.

    The pinned source first tries a top-level ``flash_attn_interface`` (FA3).
    When that import fails it later imports the ``flash_attn`` package (FA2),
    but its early FA2 return still references the name ``flash_attn_func``
    created only by the failed first ``try`` block.  The frozen FA2 wheel
    exposes the same function as ``flash_attn.flash_attn_func``.  Binding that
    existing function is the narrowest possible compatibility repair and does
    not replace the upstream attention implementation.
    """

    fa2_available = getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", None)
    fa3_available = getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", None)
    _require(isinstance(fa2_available, bool), "Krea FA2 availability flag is invalid")
    _require(isinstance(fa3_available, bool), "Krea FA3 availability flag is invalid")
    patched = False
    if fa2_available and not fa3_available:
        current = getattr(attention_module, "flash_attn_func", None)
        if not callable(current):
            package = getattr(attention_module, "flash_attn", None)
            replacement = getattr(package, "flash_attn_func", None)
            _require(
                callable(replacement),
                "Krea reports FA2 available but exposes no flash_attn_func",
            )
            attention_module.flash_attn_func = replacement
            patched = True
        _require(
            callable(getattr(attention_module, "flash_attn_func", None)),
            "Krea FA2 fast-path symbol is unavailable",
        )
    return {
        "flash_attention_2_available": fa2_available,
        "flash_attention_3_available": fa3_available,
        "flash_attention_2_symbol_compatibility_shim": patched,
    }


def materialize_wan_nonparameter_runtime_state(
    *,
    transformer: Any,
    architecture: Mapping[str, Any],
    torch_module: Any,
    rope_params: Any,
    scheduler_class: Any,
) -> dict[str, Any]:
    """Recreate ordinary tensors that a meta state-dict load cannot assign.

    ``CausalWanModel.freqs`` and ``FlowMatchScheduler`` tensors are deliberately
    not parameters or registered buffers in the official runtime.  Creating
    the scaffold under ``torch.device('meta')`` therefore leaves them meta even
    after a strict ``load_state_dict(assign=True)`` and ``Module.to``.  Rebuild
    exactly those official values outside the meta context; model parameters
    continue to be loaded once from the Krea checkpoint.
    """

    dim = architecture.get("dim")
    heads = architecture.get("num_heads")
    _require(
        isinstance(dim, int)
        and not isinstance(dim, bool)
        and isinstance(heads, int)
        and not isinstance(heads, bool)
        and dim > 0
        and heads > 0
        and dim % heads == 0,
        "Wan architecture has invalid attention dimensions",
    )
    head_dim = dim // heads
    _require(head_dim % 2 == 0, "Wan attention head dimension must be even")
    freqs = torch_module.cat(
        [
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    )
    _require(not bool(getattr(freqs, "is_meta", False)), "Wan RoPE frequencies remained meta")
    transformer.model.freqs = freqs

    scheduler = scheduler_class(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    for name in ("sigmas", "timesteps", "linear_timesteps_weights"):
        value = getattr(scheduler, name, None)
        _require(value is not None, f"Wan scheduler omitted {name}")
        _require(
            not bool(getattr(value, "is_meta", False)),
            f"Wan scheduler {name} remained meta",
        )
    transformer.scheduler = scheduler
    transformer.seq_len = 32760
    transformer.post_init()
    return {
        "wan_rope_frequencies_materialized_outside_meta": True,
        "wan_scheduler_materialized_outside_meta": True,
    }



def validate_native_correction_state(value: Any, *, chunk_index: int) -> None:
    """Validate disclosed native rebasing; it is not append-only KV reuse."""
    _require(isinstance(value, Mapping) and set(value) == {
        "schema", "native_session_id", "session_recreated", "chunk_index",
        "accepted_history_frames", "bounded_context_rebase", "context_frames",
    }, "invalid Target native correction evidence")
    _require(value["schema"] == "krea_native_correction_state_v1",
             "unknown Target native correction schema")
    identity = value["native_session_id"]
    _require(isinstance(identity, str) and len(identity) == 64
             and all(c in "0123456789abcdef" for c in identity),
             "invalid Target native session identity")
    _require(type(value["session_recreated"]) is bool,
             "invalid Target session recreation flag")
    for key, expected in (("chunk_index", chunk_index),
                          ("accepted_history_frames", 3 * chunk_index),
                          ("context_frames", min(3 * chunk_index, 3))):
        _require(type(value[key]) is int and value[key] == expected,
                 "Target native history/cursor evidence drifted")
    _require(value["bounded_context_rebase"] is bool(chunk_index),
             "Target native context rebase evidence drifted")
    _require(chunk_index != 0 or value["session_recreated"] is True,
             "Target root must start a new trajectory")


def validate_target_forward_trace(
    trace: Sequence[Mapping[str, Any]],
    schedule: Sequence[float],
    *,
    history_frames: int,
    expected_schedule: Sequence[float] = KREA_FOUR_STEP_SCHEDULE,
) -> list[dict[str, Any]]:
    """Validate the one context rebuild plus all four denoising forwards.

    The official four-step schedule includes its terminal timestep ``0``.
    Counting only positive timesteps would therefore reject every successful
    four-step generation as having three forwards.
    """

    _require(len(schedule) == 4, f"Target denoising schedule has {len(schedule)} steps")
    parsed_schedule = [float(value) for value in schedule]
    _require(
        all(math.isfinite(value) for value in parsed_schedule)
        and all(
            earlier > later
            for earlier, later in zip(parsed_schedule, parsed_schedule[1:])
        ),
        f"Target denoising schedule is not finite and strictly decreasing: {list(schedule)}",
    )
    _require(
        parsed_schedule == [float(value) for value in expected_schedule],
        f"Target denoising schedule differs from the required schedule: {list(schedule)}",
    )
    expected_total = 4 + (1 if history_frames else 0)
    _require(
        len(trace) == expected_total,
        f"Target chunk used {len(trace)} total forwards, expected {expected_total}",
    )
    expected_start = min(history_frames, 3) * 1560
    generation = [dict(item) for item in trace[-4:]]
    actual_schedule = [float(item.get("timestep", float("nan"))) for item in generation]
    _require(
        actual_schedule == parsed_schedule,
        f"Target denoising forwards differ from schedule: {actual_schedule}",
    )
    _require(
        all(item.get("batch_timesteps_equal") is True for item in generation),
        "Target denoising forward used unequal per-frame timesteps",
    )
    _require(
        all(item.get("current_start") == expected_start for item in generation),
        "Target denoising forward used an invalid KV cursor",
    )
    if history_frames:
        context = trace[0]
        _require(
            float(context.get("timestep", float("nan"))) == 0.0,
            "Target context rebuild must use timestep zero",
        )
        _require(
            context.get("batch_timesteps_equal") is True,
            "Target context rebuild used unequal timesteps",
        )
        _require(
            context.get("current_start") == expected_start,
            "Target context rebuild used an invalid KV cursor",
        )
    return generation


def validate_target_suffix_forward_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    history_frames: int,
) -> list[dict[str, Any]]:
    """Validate a T-Stitch Target suffix after the Draft step at 1000.

    The suffix is deliberately a separate contract from
    :func:`validate_target_forward_trace`: the Target must rebuild its own KV
    context and then execute exactly timesteps 750, 500, and 250.  Accepting a
    four-step Target trace here would silently turn the proposed 1S+3T
    baseline back into full Target generation.
    """

    expected_total = 3 + (1 if history_frames else 0)
    _require(
        len(trace) == expected_total,
        f"Target suffix used {len(trace)} total forwards, expected {expected_total}",
    )
    expected_start = min(history_frames, 3) * 1560
    generation = [dict(item) for item in trace[-3:]]
    actual_schedule = [
        float(item.get("timestep", float("nan"))) for item in generation
    ]
    _require(
        actual_schedule == list(MAIN_TABLE_TARGET_SUFFIX_SCHEDULE),
        f"Target suffix forwards differ from the required schedule: {actual_schedule}",
    )
    _require(
        all(item.get("batch_timesteps_equal") is True for item in generation),
        "Target suffix forward used unequal per-frame timesteps",
    )
    _require(
        all(item.get("current_start") == expected_start for item in generation),
        "Target suffix forward used an invalid KV cursor",
    )
    if history_frames:
        context = trace[0]
        _require(
            float(context.get("timestep", float("nan"))) == 0.0,
            "Target suffix context rebuild must use timestep zero",
        )
        _require(
            context.get("batch_timesteps_equal") is True,
            "Target suffix context rebuild used unequal timesteps",
        )
        _require(
            context.get("current_start") == expected_start,
            "Target suffix context rebuild used an invalid KV cursor",
        )
    return generation


def install_torch25_compatibility(torch_module: Any) -> dict[str, Any]:
    """Install the two narrow shims required by Krea on pinned Torch 2.5.1.

    Upstream imports assign ``torch._dynamo.config.recompile_limit``, a key
    introduced after 2.5.  Unknown config keys raise during module import.
    Compilation is frozen off for this runtime, so registering that otherwise
    inert key preserves upstream's assignment without changing inference.
    """

    version = str(getattr(torch_module, "__version__", ""))
    is_pinned_25 = version.split("+", 1)[0] == "2.5.1"
    compiler = getattr(torch_module, "compiler", None)
    _require(compiler is not None, "Torch runtime has no compiler namespace")
    stance_shim = not hasattr(compiler, "set_stance")
    if stance_shim:
        _require(is_pinned_25, "set_stance compatibility is allowed only on Torch 2.5.1")
        compiler.set_stance = lambda _stance: nullcontext()

    dynamo = getattr(torch_module, "_dynamo", None)
    if dynamo is None:
        dynamo = importlib.import_module("torch._dynamo")
    _require(dynamo is not None and hasattr(dynamo, "config"), "Torch has no Dynamo config")
    dynamo_config = dynamo.config
    recompile_shim = not hasattr(dynamo_config, "recompile_limit")
    if recompile_shim:
        _require(
            is_pinned_25,
            "recompile_limit compatibility is allowed only on Torch 2.5.1",
        )
        config_store = getattr(dynamo_config, "_config", None)
        defaults = getattr(dynamo_config, "_default", None)
        allowed_keys = getattr(dynamo_config, "_allowed_keys", None)
        compile_ignored = getattr(dynamo_config, "_compile_ignored_keys", None)
        _require(isinstance(config_store, dict), "Torch Dynamo config store is unavailable")
        _require(isinstance(defaults, dict), "Torch Dynamo default store is unavailable")
        _require(isinstance(allowed_keys, set), "Torch Dynamo allowed-key set is unavailable")
        _require(
            isinstance(compile_ignored, set),
            "Torch Dynamo compile-ignored set is unavailable",
        )
        _require(
            all(
                "recompile_limit" not in collection
                for collection in (config_store, defaults, allowed_keys, compile_ignored)
            ),
            "Dynamo compatibility key collided",
        )
        # Torch 2.5 ConfigModule.__setattr__ checks _allowed_keys and its
        # code generator indexes _default for every _config entry.  Register
        # the compatibility key in all of those stores before the pinned Krea
        # module executes its ordinary assignment.
        config_store["recompile_limit"] = 32
        defaults["recompile_limit"] = 32
        allowed_keys.add("recompile_limit")
        compile_ignored.add("recompile_limit")
        if hasattr(dynamo_config, "_is_dirty"):
            dynamo_config._is_dirty = True
        if hasattr(dynamo_config, "_hash_digest"):
            dynamo_config._hash_digest = None
        _require(
            getattr(dynamo_config, "recompile_limit", None) == 32,
            "could not register the Dynamo compatibility key",
        )
    return {
        "torch_version": version,
        "torch_compiler_set_stance_nullcontext_shim": stance_shim,
        "torch_dynamo_recompile_limit_registration_shim": recompile_shim,
        "torch_compile_enabled": False,
    }


def validate_wan14b_config(value: Any) -> dict[str, Any]:
    """Validate the exact architecture needed by the Krea checkpoint."""

    _require(isinstance(value, dict), "Wan 14B config must be a JSON object")
    expected = {
        "model_type": "t2v",
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_dim": 16,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "text_len": 512,
        "eps": 1e-6,
    }
    for field, frozen in expected.items():
        _require(value.get(field) == frozen, f"Wan 14B config {field} must equal {frozen!r}")
    return dict(value)


def load_wan14b_config(path: str | Path) -> dict[str, Any]:
    unresolved_path = Path(path).expanduser()
    if unresolved_path.is_symlink():
        raise AdapterError("Wan 14B config must be a regular non-symlink file")
    config_path = unresolved_path.resolve(strict=True)
    if not config_path.is_file():
        raise AdapterError("Wan 14B config must be a regular non-symlink file")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot load Wan 14B config: {error}") from error
    return validate_wan14b_config(value)


def copy_explicit_session_noise(
    *,
    session: Any,
    noise: Any,
    history_frames: int,
    torch_module: Any,
) -> None:
    """Write caller-owned noise without leaving the official inference boundary.

    ``GenerationSession.__init__`` is decorated with ``torch.inference_mode``
    in the pinned Krea runtime, so ``session.noise`` is an inference tensor.
    PyTorch rejects an in-place update to that tensor outside InferenceMode;
    keeping both the write and its equality check in the same narrow scope
    preserves the official session tensor and the explicitly supplied noise.
    """

    with torch_module.inference_mode():
        destination = session.noise[:, history_frames : history_frames + 3]
        destination.copy_(noise)
        preserved = torch_module.equal(destination, noise)
    _require(bool(preserved), "Target session did not preserve the explicit initial noise")


class KreaChunkBackend:
    """One-GPU Target backend with accepted-history-bound native sessions."""

    def __init__(
        self,
        *,
        source_root: Path,
        python_site: Path,
        artifacts: Mapping[str, Path],
        artifact_hashes: Mapping[str, str],
        config: Mapping[str, Any],
        device_name: str = "cuda:0",
        chunk_count: int = 16,
        conditioning_cache_config: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            isinstance(chunk_count, bool)
            or not isinstance(chunk_count, int)
            or chunk_count not in SUPPORTED_CORRECTION_CHUNK_COUNTS
        ):
            raise AdapterError("chunk_count must be one of 16, 30, 60, or 120")
        self.chunk_count = chunk_count
        source_text = str(source_root)
        site_text = str(python_site)
        if site_text in sys.path:
            sys.path.remove(site_text)
        sys.path.insert(0, site_text)
        if source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)
        self._old_cwd = Path.cwd()
        self._source_root = source_root
        self._python_site = python_site
        os.chdir(source_root)
        try:
            import torch  # type: ignore
            from safetensors.torch import load_file as safe_load_file  # type: ignore
        except ImportError as error:
            self._restore_process_state()
            raise AdapterError(
                "the Krea worker requires torch, OmegaConf, and safetensors"
            ) from error
        self.torch = torch
        self.device = torch.device(device_name)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self._restore_process_state()
            raise AdapterError("Krea Realtime Target execution requires CUDA")
        if torch.cuda.device_count() != 1:
            self._restore_process_state()
            raise AdapterError("each Target worker must see exactly one CUDA device")

        compatibility = install_torch25_compatibility(torch)
        os.environ["DO_COMPILE"] = "false"
        os.environ["MODEL_FOLDER"] = str(source_root / "wan_models")
        try:
            import release_server  # type: ignore
            from utils.scheduler import FlowMatchScheduler  # type: ignore
            from utils.wan_wrapper import WanDiffusionWrapper  # type: ignore
            import wan.modules.attention as wan_attention  # type: ignore
            from wan.modules.causal_model import CausalWanModel  # type: ignore
            from wan.modules.model import rope_params  # type: ignore
            if conditioning_cache_config is None:
                from wan.modules.t5 import umt5_xxl  # type: ignore
                from wan.modules.tokenizers import HuggingfaceTokenizer  # type: ignore

            attention_compatibility = install_flash_attention2_compatibility(wan_attention)

            checkpoint_name = str(config["checkpoint_artifact"])
            architecture_name = str(config["wan14b_config_artifact"])
            checkpoint_path = artifacts[checkpoint_name]
            architecture = load_wan14b_config(artifacts[architecture_name])

            with torch.device("meta"):
                causal_model = CausalWanModel.from_config(
                    architecture,
                    local_attn_size=-1,
                    sink_size=0,
                )
                transformer = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
                torch.nn.Module.__init__(transformer)
                transformer.model = causal_model
                transformer.uniform_timestep = False
            runtime_materialization = materialize_wan_nonparameter_runtime_state(
                transformer=transformer,
                architecture=architecture,
                torch_module=torch,
                rope_params=rope_params,
                scheduler_class=FlowMatchScheduler,
            )
            state = safe_load_file(str(checkpoint_path), device="cpu")
            _require(len(state) == EXPECTED_KREA_STATE_KEYS, "Krea checkpoint key count mismatch")
            incompatible = transformer.load_state_dict(state, strict=True, assign=True)
            _require(not incompatible.missing_keys, f"Krea checkpoint missing keys: {incompatible.missing_keys[:8]}")
            _require(not incompatible.unexpected_keys, f"Krea checkpoint unexpected keys: {incompatible.unexpected_keys[:8]}")
            parameter_count = sum(parameter.numel() for parameter in transformer.parameters())
            _require(parameter_count == KREA_PARAMETER_COUNT, f"Krea parameter count is {parameter_count}")
            _require(
                all(not parameter.is_meta for parameter in transformer.parameters()),
                "Krea strict assignment left meta parameters",
            )
            del state
            transformer.to(device=self.device, dtype=torch.bfloat16)
            transformer.eval().requires_grad_(False)
            for block in transformer.model.blocks:
                block.self_attn.fuse_projections()

            conditioning_binding: Mapping[str, Any] | None = None
            if conditioning_cache_config is None:
                text_name = str(config["text_encoder_artifact"])
                tokenizer_root = (
                    source_root
                    / "wan_models"
                    / "Wan2.1-T2V-1.3B"
                    / "google"
                    / "umt5-xxl"
                )

                class PthTextEncoder(torch.nn.Module):
                    def __init__(inner_self: Any) -> None:
                        super().__init__()
                        inner_self.text_encoder = umt5_xxl(
                            encoder_only=True,
                            return_tokenizer=False,
                            dtype=torch.float32,
                            device=torch.device("cpu"),
                        ).eval().requires_grad_(False)
                        raw_state = torch.load(
                            str(artifacts[text_name]),
                            map_location="cpu",
                            weights_only=False,
                        )
                        inner_self.text_encoder.load_state_dict(raw_state, strict=True)
                        del raw_state
                        inner_self.tokenizer = HuggingfaceTokenizer(
                            name=str(tokenizer_root),
                            seq_len=512,
                            clean="whitespace",
                            local_files_only=True,
                        )

                    @property
                    def device(inner_self: Any) -> Any:
                        return self.device

                    def forward(inner_self: Any, text_prompts: Sequence[str]) -> dict[str, Any]:
                        ids, mask = inner_self.tokenizer(
                            list(text_prompts),
                            return_mask=True,
                            add_special_tokens=True,
                        )
                        ids = ids.to(self.device)
                        mask = mask.to(self.device)
                        lengths = mask.gt(0).sum(dim=1).long()
                        context = inner_self.text_encoder(ids, mask)
                        for embedding, length in zip(context, lengths):
                            embedding[length:] = 0.0
                        return {"prompt_embeds": context}

                text_encoder = PthTextEncoder()
            else:
                from .conditioning_cache import (
                    TorchConditioningTensorCodec,
                    load_shared_conditioning_cache,
                    make_torch_conditioner,
                )

                loaded_conditioning = load_shared_conditioning_cache(
                    conditioning_cache_config,
                    run_dir=str(conditioning_cache_config["run_dir"]),
                    source_root=source_root,
                    role="target",
                    codec=TorchConditioningTensorCodec(torch),
                )
                text_encoder = make_torch_conditioner(
                    loaded_conditioning, torch_module=torch
                )
                conditioning_binding = loaded_conditioning.audit_binding()
            text_encoder = text_encoder.to(device=self.device, dtype=torch.bfloat16)
            text_encoder.eval().requires_grad_(False)

            class LatentOnlyDecoder(torch.nn.Module):
                def forward(inner_self: Any, z: Any, *feature_cache: Any) -> tuple[Any, list[Any]]:
                    del inner_self
                    pixels = torch.zeros(
                        (int(z.shape[0]), 12, 3, 1, 1),
                        device=z.device,
                        dtype=torch.float32,
                    )
                    return pixels, list(feature_cache)

            dummy_decoder = LatentOnlyDecoder().to(self.device)
            server_config = release_server.load_merge_config(
                source_root / "configs" / "self_forcing_server_14b.yaml"
            )
            _require(server_config.enable_fp8 is False, "Krea FP8 must remain disabled")
            pipeline = release_server.load_pipeline(
                server_config,
                torch.cuda.current_device(),
                transformer,
                text_encoder,
                dummy_decoder,
            )
            models = release_server.Models(
                text_encoder,
                transformer,
                pipeline,
                None,
                dummy_decoder,
            )
            self.release_server = release_server
            self.config = server_config
            self.models = models
            from .host_kv_cursors import install as install_host_kv_cursors
            host_kv_cursor_optimization = install_host_kv_cursors(models.pipeline, torch=torch)
            self._prompt_cache: dict[str, dict[str, Any]] = {}
            self._kv_witness_sampler = sampled_kv_cache_witness
            self._persistent_session: Any | None = None
            self._persistent_session_id: str | None = None
            self._persistent_prompt: str | None = None
            self._persistent_seed: int | None = None
            self._correction_session: Any | None = None
            self._correction_prompt: str | None = None
            self._correction_session_id: str | None = None
            self._correction_session_counter = 0
            self.checkpoint_sha256 = artifact_hashes[checkpoint_name]
            self.model_state = {
                "host_kv_cursor_optimization": host_kv_cursor_optimization,
                "source_commit": KREA_SOURCE_COMMIT,
                "checkpoint_revision": KREA_CHECKPOINT_REVISION,
                "checkpoint_sha256": self.checkpoint_sha256,
                "wan14b_config_revision": WAN14B_CONFIG_REVISION,
                "wan14b_config_sha256": artifact_hashes[architecture_name],
                "parameter_count": parameter_count,
                "checkpoint_key_count": EXPECTED_KREA_STATE_KEYS,
                "dtype": "bfloat16",
                **compatibility,
                **attention_compatibility,
                **runtime_materialization,
                "kv_policy": "native_session_with_accepted_history_and_bounded_context_rebase",
                "kv_cache_num_frames": 3,
                "denoising_steps": 4,
                "initial_noise_transport": "exact_bfloat16_copy_into_absolute_chunk_slice",
                "renoise_rng_policy": "reset_to_explicit_chunk_seed_after_session_initialization",
                "main_table_persistent_session_supported": True,
                "main_table_interleaved_real_vae_decode": False,
            }
            if conditioning_binding is not None:
                self.model_state["conditioning_cache"] = dict(conditioning_binding)
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException:
            self._restore_process_state()
            raise

    def _restore_process_state(self) -> None:
        os.chdir(self._old_cwd)
        for value in (str(self._source_root), str(self._python_site)):
            if value in sys.path:
                sys.path.remove(value)

    def _condition(self, prompt: str) -> dict[str, Any]:
        torch = self.torch
        if prompt not in self._prompt_cache:
            with torch.inference_mode():
                condition = self.models.text_encoder(text_prompts=[prompt])
            self._prompt_cache[prompt] = {
                key: value.to(dtype=torch.bfloat16).contiguous()
                for key, value in condition.items()
            }
        return self._prompt_cache[prompt]

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
        session.denoising_step_list = list(MAIN_TABLE_FOUR_STEP_SCHEDULE)
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
            expected_schedule=MAIN_TABLE_FOUR_STEP_SCHEDULE,
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

    def append_persistent_clean_chunk(
        self,
        *,
        prompt: str,
        clean_latent: Any,
        chunk_index: int,
    ) -> dict[str, Any]:
        """Commit one accepted external chunk to the active Target trajectory.

        SDVG keeps an independent Draft trajectory.  When its router accepts a
        Draft block, the exact clean latent must nevertheless become part of
        the Target history before any later Target regeneration.  The pinned
        Krea runtime rebuilds its short KV window from ``all_latents``; this
        method updates that canonical history and performs the corresponding
        context rebuild immediately so the acceptance cost is measured rather
        than hidden in a later rejection.
        """

        torch = self.torch
        session = self._persistent_session
        _require(session is not None, "no persistent Target session is active")
        _require(prompt == self._persistent_prompt, "persistent Target prompt drifted")
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and chunk_index == session.block_idx
            and 1 <= chunk_index < self.chunk_count,
            "external Target commit chunk order drifted",
        )
        _require(
            isinstance(clean_latent, torch.Tensor)
            and tuple(clean_latent.shape) == (1, 3, 16, 60, 104),
            "external Target clean latent has an invalid shape",
        )
        candidate = clean_latent.to(
            device=self.device, dtype=torch.bfloat16
        ).contiguous()
        _require(
            bool(torch.isfinite(candidate.float()).all().item()),
            "external Target clean latent contains non-finite values",
        )
        start_frame = int(session.current_start_frame)
        _require(
            start_frame == 3 * chunk_index,
            "external Target commit cursor drifted",
        )
        with torch.inference_mode():
            session.all_latents[
                :, start_frame : start_frame + 3
            ].copy_(candidate)
        session.last_pred = candidate
        session.current_start_frame = start_frame + 3
        session.block_idx = chunk_index + 1
        session.resume_latents = None

        trace: list[dict[str, Any]] = []

        def pre_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
            timestep = kwargs.get("timestep")
            current_start = kwargs.get("current_start")
            if (
                timestep is None
                or isinstance(current_start, bool)
                or not isinstance(current_start, int)
            ):
                raise AdapterError(
                    "external Target commit audit hook omitted timestep/current_start"
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

        handle = self.models.transformer.register_forward_pre_hook(
            pre_hook, prepend=False, with_kwargs=True
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        try:
            start_event.record()
            with torch.inference_mode():
                model_input_start = session.recompute_kv_cache(self.models)
            end_event.record()
            end_event.synchronize()
        finally:
            handle.remove()
        _require(model_input_start == 3, "external Target KV rebuild cursor drifted")
        _require(
            len(trace) == 1
            and float(trace[0]["timestep"]) == 0.0
            and trace[0]["batch_timesteps_equal"] is True
            and trace[0]["current_start"] == 3 * 1560,
            "external Target commit did not perform one valid clean-context rebuild",
        )
        witness = self._kv_witness_sampler(
            self.models.pipeline.kv_cache1,
            getattr(self.models.pipeline, "crossattn_cache", None),
        )
        elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        _require(
            math.isfinite(elapsed) and elapsed >= 0.0,
            "external Target KV append CUDA time is invalid",
        )
        return {
            "native_session_id": self._persistent_session_id,
            "native_session_block_index": chunk_index,
            "committed_external_clean_latent": True,
            "target_kv_append_cuda_seconds": elapsed,
            "context_forward_trace": trace,
            "kv_witness_sha256": witness[0],
            "kv_witness": witness[1],
            **witness_index_execution_payload(witness),
        }

    def generate_persistent_suffix_chunk(
        self,
        *,
        prompt: str,
        noisy_latent_750: Any,
        renoise_500: Any,
        renoise_250: Any,
        chunk_index: int,
    ) -> dict[str, Any]:
        """Run exactly the Target 750/500/250 suffix for T-Stitch.

        ``noisy_latent_750`` is produced from the Draft timestep-1000 clean
        prediction using the shared scheduler.  The two subsequent explicit
        noise tensors keep the stochastic trajectory bound across the process
        and GPU boundary instead of assuming CUDA generators advance
        identically in two independent workers.
        """

        torch = self.torch
        session = self._persistent_session
        _require(session is not None, "no persistent Target session is active")
        _require(prompt == self._persistent_prompt, "persistent Target prompt drifted")
        _require(
            isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and chunk_index == session.block_idx
            and 0 <= chunk_index < self.chunk_count,
            "persistent Target suffix chunk order drifted",
        )
        expected_shape = (1, 3, 16, 60, 104)
        for label, value in (
            ("noisy_latent_750", noisy_latent_750),
            ("renoise_500", renoise_500),
            ("renoise_250", renoise_250),
        ):
            _require(
                isinstance(value, torch.Tensor) and tuple(value.shape) == expected_shape,
                f"persistent Target suffix {label} has an invalid shape",
            )
        noisy_input = noisy_latent_750.to(
            device=self.device, dtype=torch.bfloat16
        ).contiguous()
        future_noise = (
            renoise_500.to(device=self.device, dtype=torch.bfloat16).contiguous(),
            renoise_250.to(device=self.device, dtype=torch.bfloat16).contiguous(),
        )
        _require(
            bool(torch.isfinite(noisy_input.float()).all().item())
            and all(
                bool(torch.isfinite(value.float()).all().item())
                for value in future_noise
            ),
            "persistent Target suffix input contains non-finite values",
        )
        history_frames = 3 * chunk_index
        trace: list[dict[str, Any]] = []
        witness: tuple[str, dict[str, Any]] | None = None

        def pre_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
            nonlocal witness
            timestep = kwargs.get("timestep")
            current_start = kwargs.get("current_start")
            if (
                timestep is None
                or isinstance(current_start, bool)
                or not isinstance(current_start, int)
            ):
                raise AdapterError(
                    "persistent Target suffix audit hook omitted timestep/current_start"
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
            with torch.inference_mode():
                model_input_start = session.recompute_kv_cache(self.models)
                _require(
                    model_input_start == min(history_frames, 3),
                    "persistent Target suffix KV cursor drifted",
                )
                session.conditional_dict["prompt_embeds"] = (
                    session.current_prompt_embeds
                )
                for offset, current_timestep in enumerate(
                    MAIN_TABLE_TARGET_SUFFIX_SCHEDULE
                ):
                    timestep = torch.ones(
                        [1, 3], device=self.device, dtype=torch.int64
                    ) * current_timestep
                    _, denoised_pred = self.models.transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=session.conditional_dict,
                        timestep=timestep,
                        kv_cache=self.models.pipeline.kv_cache1,
                        crossattn_cache=self.models.pipeline.crossattn_cache,
                        current_start=model_input_start
                        * self.models.pipeline.frame_seq_length,
                    )
                    if offset < 2:
                        next_timestep = MAIN_TABLE_TARGET_SUFFIX_SCHEDULE[offset + 1]
                        noise = future_noise[offset]
                        noisy_input = self.models.pipeline.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            noise.flatten(0, 1),
                            next_timestep
                            * torch.ones(
                                [3], device=self.device, dtype=torch.long
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                session.all_latents[
                    :, history_frames : history_frames + 3
                ].copy_(denoised_pred)
                session.last_pred = denoised_pred
                session.current_start_frame = history_frames + 3
                session.block_idx = chunk_index + 1
                session.resume_latents = None
            end_event.record()
            end_event.synchronize()
        finally:
            handle.remove()
        generation_cuda_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
        _require(
            math.isfinite(generation_cuda_seconds)
            and generation_cuda_seconds >= 0.0,
            "persistent Target suffix CUDA time is invalid",
        )
        denoising_trace = validate_target_suffix_forward_trace(
            trace, history_frames=history_frames
        )
        _require(witness is not None, "persistent Target suffix omitted its KV witness")
        candidate = denoised_pred.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        _require(
            tuple(candidate.shape) == expected_shape
            and bool(torch.isfinite(candidate.float()).all().item()),
            "persistent Target suffix generated an invalid clean latent",
        )
        return {
            "latent": candidate,
            "forward_trace": trace,
            "denoising_forward_trace": denoising_trace,
            "denoising_schedule": list(MAIN_TABLE_TARGET_SUFFIX_SCHEDULE),
            "draft_handoff_timestep": 1000.0,
            "native_session_id": self._persistent_session_id,
            "native_session_block_index": chunk_index,
            "session_recreated": False,
            "history_reconstructed": False,
            "target_kv_context_recomputed": True,
            "vae_decode_performed": False,
            "generation_cuda_seconds": generation_cuda_seconds,
            "kv_witness_sha256": witness[0],
            "kv_witness": witness[1],
            **witness_index_execution_payload(witness),
        }

    def finish_persistent_trajectory(self, *, session_id: str) -> dict[str, Any]:
        session = self._persistent_session
        _require(session is not None, "no persistent Target session is active")
        _require(session_id == self._persistent_session_id, "persistent session ID drifted")
        generated = int(session.block_idx)
        _require(generated == self.chunk_count, "persistent Target horizon is incomplete")
        self._persistent_session = None
        self._persistent_session_id = None
        self._persistent_prompt = None
        self._persistent_seed = None
        del session
        gc.collect()
        self.torch.cuda.empty_cache()
        return {"session_id": session_id, "generated_block_count": generated}

    def generate_chunk(
        self, *, prompt: str, history_latents: Any | None, initial_noise: Any,
        internal_seed: int, chunk_index: int,
    ) -> dict[str, Any]:
        return self._generate_correction(prompt=prompt, history_latents=history_latents,
            initial_noise=initial_noise, internal_seed=internal_seed, chunk_index=chunk_index)

    def generate_first_update(
        self, *, prompt: str, history_latents: Any | None, initial_noise: Any,
        internal_seed: int, chunk_index: int,
    ) -> dict[str, Any]:
        """Experimental Target prefix; output is not an accepted video chunk."""
        return self._generate_correction(prompt=prompt, history_latents=history_latents,
            initial_noise=initial_noise, internal_seed=internal_seed, chunk_index=chunk_index,
            first_update_only=True)

    def check_first_update(self, **kwargs) -> dict[str, Any]:
        """Diagnostic only: compare prefix with the first output of full Target."""
        prefix=self.generate_first_update(**kwargs)
        captured=[];active=[False]
        def before(_module,_args,kw):
            active[0]=float(kw['timestep'].flatten()[0].item())==1000.0
        def after(_module,_args,out):
            if active[0]:captured.append(out[1].detach().to('cpu').contiguous())
        pre=self.models.transformer.register_forward_pre_hook(before,with_kwargs=True)
        post=self.models.transformer.register_forward_hook(after)
        try:
            full=self.generate_chunk(**kwargs)
        finally:
            pre.remove();post.remove()
        _require(len(captured)==1,'Target oracle did not capture exactly one first update')
        exact=bool(self.torch.equal(captured[0],prefix['first_clean_prediction']))
        maximum=float((captured[0].float()-prefix['first_clean_prediction'].float()).abs().max().item())
        _require(exact,'Target prefix diverged from full Target first update')
        prefix['diagnostic_oracle']={'first_prediction_exact':exact,'maximum_absolute_difference':maximum,
            'extra_target_denoising_forwards':4,'full_forward_trace':full['forward_trace'],
            'no_quality_or_speed_claim':True}
        return prefix

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
        # Corrections and the persistent SDVG/pure-Target reference must use
        # identical denoising steps. The upstream session default is different.
        session.denoising_step_list = list(MAIN_TABLE_FOUR_STEP_SCHEDULE)
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
                expected_schedule=MAIN_TABLE_FOUR_STEP_SCHEDULE,
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

    def close(self) -> None:
        self._correction_session = None
        self._correction_prompt = None
        self._correction_session_id = None
        self._persistent_session = None
        self._persistent_session_id = None
        self._persistent_prompt = None
        self._persistent_seed = None
        self._prompt_cache.clear()
        self._restore_process_state()
