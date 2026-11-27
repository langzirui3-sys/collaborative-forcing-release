"""Official Self-Forcing adapter for the frame-level eRank campaign.

The adapter deliberately imports no ML dependency at module import time.  A
formal worker first verifies its request and every pinned artifact, safely
materializes the exact official Self-Forcing source archive, and only then
imports torch/OmegaConf/torchvision from the frozen runtime.

The generation path uses the public interface at Self-Forcing commit
``33593df3e81fa3ec10239271dd2c100facac6de1`` without patching its source:
``CausalInferencePipeline.inference`` performs the native block loop.  Forward
hooks on six transformer blocks observe only the native denoising call whose
actual timestep is 1000.  No probe forward is issued.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import sys
import tarfile
from typing import Any, Callable, Mapping, Protocol, Sequence
import zipfile


from .stage_timing import current_timing, stage


REQUEST_SCHEMA = "erank_frame_adapter_request_v1"
RESULT_SCHEMA = "erank_frame_adapter_result_v1"
RUNTIME_CONFIG_SCHEMA = "self_forcing_frame_adapter_config_v1"
OFFICIAL_SOURCE_COMMIT = "33593df3e81fa3ec10239271dd2c100facac6de1"
LAYER_INDICES = (0, 4, 9, 14, 19, 24)
LAYER_FIELDS = {
    0: "erank_l1",
    4: "erank_l5",
    9: "erank_l10",
    14: "erank_l15",
    19: "erank_l20",
    24: "erank_l25",
}
METRICS_COLUMNS = (
    "prompt_id",
    "analysis_split",
    "assigned_dimension",
    "seed",
    "model_id",
    "trajectory_seconds",
    "fps",
    "trajectory_frames",
    "block_index",
    "frame_start",
    "frame_end_exclusive",
    "analysis_eligible",
    "anchor_frame_index",
    "anchor_png_path",
    "anchor_png_sha256",
    "erank_l1",
    "erank_l5",
    "erank_l10",
    "erank_l15",
    "erank_l20",
    "erank_l25",
    "musiq_spaq_raw",
    "musiq_spaq",
)
FROZEN_PARAMETERS: dict[str, Any] = {
    "worker_count": 4,
    "prompt_count": 48,
    "seed": 20260911,
    "duration_seconds": 15,
    "fps": 24,
    "layer_indices_zero_based": list(LAYER_INDICES),
    "expected_blocks": 31,
    "eligible_block_start": 1,
    "eligible_block_end": 29,
}
_SHA256_HEX = frozenset("0123456789abcdef")
SAMPLED_KV_CACHE_WITNESS_SCHEMA = "sampled_kv_cache_witness_v1"
SAMPLED_KV_CACHE_WITNESS_SAMPLING = (
    "up_to_64_evenly_spaced_values_from_each_populated_k_and_v"
)
SELF_FORCING_KV_LAYER_COUNT = 30
SELF_FORCING_SELF_KV_SHAPE = (1, 32760, 12, 128)
SELF_FORCING_CROSS_KV_SHAPE = (1, 512, 12, 128)
SELF_FORCING_KV_DTYPE = "torch.bfloat16"
SELF_FORCING_TOKENS_PER_CHUNK = 4680


def draft_kv_feature_flags(environ: Mapping[str, str] | None = None) -> tuple[bool, bool]:
    values = os.environ if environ is None else environ
    reuse = values.get("CFEXP_DRAFT_KV_REUSE", "0")
    fork = values.get("CFEXP_DRAFT_KV_FORK", "0")
    if reuse not in {"0", "1"} or fork not in {"0", "1"}:
        raise AdapterError("CFEXP_DRAFT_KV_REUSE and CFEXP_DRAFT_KV_FORK must be 0 or 1")
    if fork == "1" and reuse != "1":
        raise AdapterError("CFEXP_DRAFT_KV_FORK requires CFEXP_DRAFT_KV_REUSE=1")
    return reuse == "1", fork == "1"


SELF_FORCING_DENOISING_SCHEDULE = (
    1000.0,
    937.5,
    833.3333129882812,
    625.0,
    0.0,
)
EFFECTIVE_RANK_EVIDENCE_SCHEMA = "self_forcing_effective_rank_evidence_v1"
EFFECTIVE_RANK_LAYER = 14
EFFECTIVE_RANK_RELATIVE_CUTOFF = 1e-7
SUPPORTED_CORRECTION_CHUNK_COUNTS = frozenset({16, 30, 60, 120})
EFFECTIVE_RANK_ALGORITHM = (
    "entropy_exp_of_l1_normalized_singular_values_from_retained_gram_spectrum_v1"
)
EFFECTIVE_RANK_SAMPLING = {
    "axis": "visual_tokens",
    "maximum_tokens": 256,
    "sample_count": "min(current_visual_token_count, maximum_tokens)",
    "token_indices": (
        "floor(i * current_visual_token_count / sample_count), "
        "i=0..sample_count-1"
    ),
    "sampled_hidden_dtype": "float32",
    "gram_matrix": "sampled_hidden @ sampled_hidden.T",
}


class AdapterError(RuntimeError):
    """Raised when a formal worker cannot prove the adapter contract."""


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdapterError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AdapterError(f"{label} contains invalid constant {item!r}")
            ),
        )
    except AdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _inside(root: Path, candidate: Path, label: str, *, strict: bool = True) -> Path:
    try:
        resolved_root = root.expanduser().resolve(strict=True)
        resolved = candidate.expanduser().resolve(strict=strict)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise AdapterError(f"{label} escapes {root}: {candidate}") from error
    return resolved


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AdapterError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise AdapterError(f"{label} must be a safe relative path")
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def deterministic_token_indices(token_count: int, maximum: int = 256) -> tuple[int, ...]:
    """Protocol sampling: ``floor(i*T/N)`` for ``i=0..N-1``."""

    if token_count < 1 or maximum < 1:
        raise AdapterError("token_count and maximum must be positive")
    count = min(token_count, maximum)
    return tuple((index * token_count) // count for index in range(count))


def entropy_effective_rank_from_gram_eigenvalues(
    eigenvalues: Sequence[float], *, relative_cutoff: float = 1e-7
) -> float:
    """Return the protocol eRank from eigenvalues of ``H @ H.T``.

    Tiny negative eigenvalues caused by finite precision are clamped to zero.
    The cutoff is applied to Gram eigenvalues before their square root, exactly
    as frozen in the pilot protocol.
    """

    if not eigenvalues:
        raise AdapterError("the Gram spectrum is empty")
    if not math.isfinite(relative_cutoff) or not 0.0 <= relative_cutoff < 1.0:
        raise AdapterError("relative_cutoff must be finite and in [0, 1)")
    values: list[float] = []
    for raw in eigenvalues:
        value = float(raw)
        if not math.isfinite(value):
            raise AdapterError("the Gram spectrum contains a non-finite value")
        values.append(max(0.0, value))
    maximum = max(values)
    if maximum <= 0.0:
        raise AdapterError("cannot define eRank for an all-zero hidden state")
    retained = [value for value in values if value > maximum * relative_cutoff]
    singular_values = [math.sqrt(value) for value in retained]
    total = math.fsum(singular_values)
    if total <= 0.0:
        raise AdapterError("the retained singular spectrum is empty")
    probabilities = [value / total for value in singular_values]
    entropy = -math.fsum(probability * math.log(probability) for probability in probabilities)
    result = math.exp(entropy)
    if not math.isfinite(result) or not 1.0 <= result <= len(retained) + 1e-5:
        raise AdapterError(f"invalid effective rank {result!r}")
    return min(result, float(len(retained)))


@dataclass(frozen=True)
class EffectiveRankMeasurement:
    """One rank and the exact float64 Gram spectrum used to compute it."""

    effective_rank: float
    gram_eigenvalues: tuple[float, ...]


def validate_effective_rank_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_rank: float | None = None,
    expected_eigenvalue_count: int | None = None,
) -> float:
    """Validate and recompute a persisted Draft effective-rank witness."""

    expected_keys = {
        "schema",
        "layer",
        "relative_cutoff",
        "algorithm",
        "sampling",
        "gram_eigenvalue_dtype",
        "gram_eigenvalues",
        "effective_rank",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise AdapterError("effective-rank evidence has invalid fields")
    if evidence.get("schema") != EFFECTIVE_RANK_EVIDENCE_SCHEMA:
        raise AdapterError("effective-rank evidence has an invalid schema")
    if evidence.get("layer") != EFFECTIVE_RANK_LAYER:
        raise AdapterError(
            f"effective-rank evidence layer must equal {EFFECTIVE_RANK_LAYER}"
        )
    if evidence.get("relative_cutoff") != EFFECTIVE_RANK_RELATIVE_CUTOFF:
        raise AdapterError("effective-rank evidence has an invalid relative cutoff")
    if evidence.get("algorithm") != EFFECTIVE_RANK_ALGORITHM:
        raise AdapterError("effective-rank evidence has an invalid algorithm")
    if evidence.get("sampling") != EFFECTIVE_RANK_SAMPLING:
        raise AdapterError("effective-rank evidence has an invalid sampling rule")
    if evidence.get("gram_eigenvalue_dtype") != "float64":
        raise AdapterError("effective-rank Gram eigenvalues must be float64")
    eigenvalues = evidence.get("gram_eigenvalues")
    if not isinstance(eigenvalues, list) or not 1 <= len(eigenvalues) <= 256:
        raise AdapterError(
            "effective-rank Gram eigenvalues must contain between 1 and 256 values"
        )
    if (
        expected_eigenvalue_count is not None
        and len(eigenvalues) != expected_eigenvalue_count
    ):
        raise AdapterError(
            "effective-rank Gram eigenvalue count does not match the expected count"
        )
    parsed: list[float] = []
    for index, value in enumerate(eigenvalues):
        if not isinstance(value, float) or not math.isfinite(value):
            raise AdapterError(
                f"effective-rank Gram eigenvalue {index} must be a finite float"
            )
        if parsed and value < parsed[-1]:
            raise AdapterError("effective-rank Gram eigenvalues must be nondecreasing")
        parsed.append(value)
    stored_rank = evidence.get("effective_rank")
    if not isinstance(stored_rank, float) or not math.isfinite(stored_rank):
        raise AdapterError("effective-rank evidence rank must be a finite float")
    recomputed = entropy_effective_rank_from_gram_eigenvalues(
        parsed, relative_cutoff=EFFECTIVE_RANK_RELATIVE_CUTOFF
    )
    if stored_rank != recomputed:
        raise AdapterError("effective-rank evidence rank does not recompute exactly")
    if expected_rank is not None:
        if (
            isinstance(expected_rank, bool)
            or not isinstance(expected_rank, (int, float))
            or not math.isfinite(float(expected_rank))
            or float(expected_rank) != recomputed
        ):
            raise AdapterError("effective-rank evidence does not match the returned rank")
    return recomputed


@dataclass(frozen=True)
class CaptureEvent:
    """Six layer measurements produced by one top-level generator forward."""

    event_index: int
    current_start: int
    ranks: dict[int, float]
    retained_layer: int | None = None
    retained_sample: Any | None = None
    retained_full_hidden: Any | None = None
    retained_rank_layer: int | None = None
    retained_gram_eigenvalues: tuple[float, ...] | None = None


def effective_rank_evidence_from_capture(event: CaptureEvent) -> dict[str, Any]:
    """Build the canonical JSON witness for layer-14 Draft rank."""

    if event.retained_rank_layer != EFFECTIVE_RANK_LAYER:
        raise AdapterError("capture did not retain the protocol effective-rank layer")
    eigenvalues = event.retained_gram_eigenvalues
    if eigenvalues is None:
        raise AdapterError("capture did not retain the protocol Gram spectrum")
    rank = event.ranks.get(EFFECTIVE_RANK_LAYER)
    if rank is None:
        raise AdapterError("capture did not retain the protocol effective rank")
    evidence = {
        "schema": EFFECTIVE_RANK_EVIDENCE_SCHEMA,
        "layer": EFFECTIVE_RANK_LAYER,
        "relative_cutoff": EFFECTIVE_RANK_RELATIVE_CUTOFF,
        "algorithm": EFFECTIVE_RANK_ALGORITHM,
        "sampling": dict(EFFECTIVE_RANK_SAMPLING),
        "gram_eigenvalue_dtype": "float64",
        "gram_eigenvalues": list(eigenvalues),
        "effective_rank": float(rank),
    }
    validate_effective_rank_evidence(
        evidence,
        expected_rank=rank,
        expected_eigenvalue_count=len(eigenvalues),
    )
    return evidence


class StepZeroCapture:
    """State machine proving all layer values came from one native forward."""

    def __init__(
        self,
        *,
        layers: Sequence[int],
        target_timestep: int,
        sample_hidden: Callable[[Any], Any],
        rank_sample: Callable[[Any], float | EffectiveRankMeasurement],
        retain_layer: int | None = None,
        retain_full_hidden: Callable[[Any], Any] | None = None,
        retain_rank_layer: int | None = None,
        policy_rank_only: bool = False,
        early_should_replace: Callable[[EffectiveRankMeasurement], bool] | None = None,
    ) -> None:
        if tuple(layers) != LAYER_INDICES:
            raise AdapterError(f"capture layers must be exactly {LAYER_INDICES}")
        if not isinstance(policy_rank_only, bool):
            raise AdapterError("policy_rank_only must be boolean")
        if policy_rank_only and (
            retain_rank_layer != EFFECTIVE_RANK_LAYER
            or retain_layer not in (None, EFFECTIVE_RANK_LAYER)
        ):
            raise AdapterError("policy-only capture requires the frozen layer-14 rank")
        self.policy_rank_only = policy_rank_only
        if early_should_replace is not None and not policy_rank_only:
            raise AdapterError("early probe requires policy-only capture")
        self.early_should_replace = early_should_replace
        self._early_measurement: EffectiveRankMeasurement | None = None
        self.layers = tuple(layers)
        self.target_timestep = target_timestep
        self.sample_hidden = sample_hidden
        self.rank_sample = rank_sample
        if retain_layer is not None and retain_layer not in self.layers:
            raise AdapterError("retain_layer must be one of the captured layers")
        if retain_full_hidden is not None and retain_layer is None:
            raise AdapterError("retain_full_hidden requires retain_layer")
        if retain_rank_layer is not None and retain_rank_layer not in self.layers:
            raise AdapterError("retain_rank_layer must be one of the captured layers")
        self.retain_layer = retain_layer
        self.retain_full_hidden = retain_full_hidden
        self.retain_rank_layer = retain_rank_layer
        self.events: list[CaptureEvent] = []
        self._active = False
        self._current_start: int | None = None
        self._samples: dict[int, Any] = {}
        self._retained_full: Any | None = None

    def begin_call(self, timestep_values: Sequence[float], *, current_start: int) -> None:
        if self._active or self._samples:
            raise AdapterError("a generator call began before the prior call was closed")
        if not timestep_values:
            raise AdapterError("generator timestep tensor is empty")
        normalized = [float(item) for item in timestep_values]
        if any(not math.isfinite(item) for item in normalized) or len(set(normalized)) != 1:
            raise AdapterError("a native generator forward must use one finite timestep")
        capture = normalized[0] == float(self.target_timestep)
        self._active = capture
        self._current_start = current_start if capture else None

    def record_layer(self, layer_index: int, hidden: Any) -> None:
        if not self._active:
            return
        if layer_index not in self.layers:
            raise AdapterError(f"unexpected hooked layer {layer_index}")
        if layer_index in self._samples:
            raise AdapterError(f"layer {layer_index} ran twice in one generator forward")
        # Keep presence/duplicate checks for every hook. Only policy-relevant
        # activations need sampling and a Gram eigensolve in chunk generation.
        self._samples[layer_index] = (
            self.sample_hidden(hidden)
            if not self.policy_rank_only or layer_index == EFFECTIVE_RANK_LAYER
            else None
        )
        if layer_index == self.retain_layer and self.retain_full_hidden is not None:
            self._retained_full = self.retain_full_hidden(hidden)
        if layer_index == EFFECTIVE_RANK_LAYER and self.early_should_replace is not None:
            from .early_draft import DraftProbeStop
            measurement = self.rank_sample(self._samples[layer_index])
            if not isinstance(measurement, EffectiveRankMeasurement):
                raise AdapterError("early probe requires an auditable Gram spectrum")
            rank = entropy_effective_rank_from_gram_eigenvalues(
                measurement.gram_eigenvalues, relative_cutoff=EFFECTIVE_RANK_RELATIVE_CUTOFF)
            if not math.isfinite(rank) or rank != measurement.effective_rank:
                raise AdapterError("early probe rank does not match its spectrum")
            self._early_measurement = measurement
            replace = self.early_should_replace(measurement)
            if type(replace) is not bool:
                raise AdapterError("early decision must be a boolean")
            if replace:
                raise DraftProbeStop({"layer_index": layer_index,
                    "current_start": self._current_start, "timestep": self.target_timestep,
                    "effective_rank": rank, "gram_eigenvalues": list(measurement.gram_eigenvalues),
                    "completed_native_forward": False})

    def end_call(self) -> CaptureEvent | None:
        if not self._active:
            self._current_start = None
            self._samples.clear()
            self._retained_full = None
            return None
        missing = [layer for layer in self.layers if layer not in self._samples]
        if missing:
            self._active = False
            self._current_start = None
            self._samples.clear()
            self._retained_full = None
            raise AdapterError(f"step-0 forward missed hooked layers {missing}")
        assert self._current_start is not None
        ranks: dict[int, float] = {}
        retained_gram_eigenvalues: tuple[float, ...] | None = None
        for layer in self.layers:
            if self.policy_rank_only and layer != EFFECTIVE_RANK_LAYER:
                continue
            measurement = (self._early_measurement
                if layer == EFFECTIVE_RANK_LAYER and self._early_measurement is not None
                else self.rank_sample(self._samples[layer]))
            if isinstance(measurement, EffectiveRankMeasurement):
                rank = float(measurement.effective_rank)
                spectrum = tuple(float(value) for value in measurement.gram_eigenvalues)
                recomputed = entropy_effective_rank_from_gram_eigenvalues(
                    spectrum, relative_cutoff=EFFECTIVE_RANK_RELATIVE_CUTOFF
                )
                if rank != recomputed:
                    raise AdapterError(
                        f"layer {layer} effective rank does not match its Gram spectrum"
                    )
                if layer == self.retain_rank_layer:
                    retained_gram_eigenvalues = spectrum
            else:
                rank = float(measurement)
            if not math.isfinite(rank):
                raise AdapterError(f"layer {layer} produced a non-finite effective rank")
            ranks[layer] = rank
        event = CaptureEvent(
            event_index=len(self.events),
            current_start=self._current_start,
            ranks=ranks,
            retained_layer=self.retain_layer,
            retained_sample=(
                self._samples[self.retain_layer]
                if self.retain_layer is not None
                else None
            ),
            retained_full_hidden=self._retained_full,
            retained_rank_layer=self.retain_rank_layer,
            retained_gram_eigenvalues=retained_gram_eigenvalues,
        )
        self.events.append(event)
        self._active = False
        self._current_start = None
        self._samples.clear()
        self._retained_full = None
        self._early_measurement = None
        return event


_LAST_WITNESS_INDEX_EXECUTION = ContextVar("cfexp_witness_index_execution", default=None)


def sampled_kv_cache_witness(
    kv_cache: Sequence[Mapping[str, Any]],
    crossattn_cache: Sequence[Mapping[str, Any]] | None = None,
    *, batched: bool | None = None, reuse_indices: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Hash deterministic samples of a model-local KV cache.

    This is an auditable witness, not a byte-for-byte dump of several GiB of
    cache tensors.  It binds every layer's shape, dtype, cursor, and evenly
    spaced samples from the populated prefix.  Model/checkpoint and history
    identities are bound separately by :class:`KVCacheDescriptor`.
    """

    _LAST_WITNESS_INDEX_EXECUTION.set(None)
    if not kv_cache:
        raise AdapterError("KV cache witness requires at least one layer")
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise AdapterError("PyTorch is required to witness a KV cache") from error

    if batched is None:
        selected = os.environ.get("CFEXP_BATCH_KV_WITNESS", "0")
        if selected not in {"0", "1"}:
            raise AdapterError("CFEXP_BATCH_KV_WITNESS must be 0 or 1")
        batched = selected == "1"
    if type(batched) is not bool:
        raise AdapterError("batched witness selection must be boolean")
    if reuse_indices is None:
        from .native_rank_indices import enabled
        reuse_indices = enabled("CFEXP_KV_WITNESS_INDEX_REUSE")
    if type(reuse_indices) is not bool:
        raise AdapterError("witness index reuse selection must be boolean")
    # Call-local constants only: never retain cache values, samples or CUDA
    # tensors across calls/streams. At most 64 layouts (32 KiB of int64).
    index_constants = {}
    cursor_values = {}
    pending_samples = []
    index_counts = {"sample_tensors": 0, "allocations": 0, "hits": 0}
    if batched:
        from .kv_witness_batch import host_lists
        cursors = []
        for layer_index, layer in enumerate(kv_cache):
            if not isinstance(layer, Mapping) or "k" not in layer or "v" not in layer:
                raise AdapterError(f"KV cache layer {layer_index} is invalid")
            for name in ("global_end_index", "local_end_index"):
                value = layer.get(name, 0)
                if hasattr(value, "detach"):
                    if int(value.numel()) != 1:
                        raise AdapterError(f"{name} must be scalar")
                    cursors.append(value)
        for value, host in zip(cursors, host_lists(cursors, torch=torch)):
            cursor_values[id(value)] = host[0]

    def scalar_integer(value: Any, label: str) -> int:
        if hasattr(value, "detach"):
            flattened = value.detach().reshape(-1)
            if int(flattened.numel()) != 1:
                raise AdapterError(f"{label} must be scalar")
            value = cursor_values[id(value)] if batched else flattened.to(device="cpu").item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdapterError(f"{label} must be numeric")
        result = int(value)
        if float(value) != float(result) or result < 0:
            raise AdapterError(f"{label} must be a non-negative integer")
        return result

    def tensor_record(tensor: Any, used_tokens: int, label: str) -> dict[str, Any]:
        if not hasattr(tensor, "detach") or not hasattr(tensor, "shape"):
            raise AdapterError(f"{label} is not a tensor")
        shape = [int(value) for value in tensor.shape]
        if len(shape) < 2 or any(value < 1 for value in shape):
            raise AdapterError(f"{label} has an invalid shape")
        used = min(max(used_tokens, 0), shape[1])
        if used == 0:
            samples: list[float] = []
        else:
            index_counts["sample_tensors"] += 1
            populated = tensor.detach()[:, :used].reshape(-1)
            sample_count = min(64, int(populated.numel()))
            key = (tensor.device, int(populated.numel()), sample_count)
            index_tensor = index_constants.get(key) if reuse_indices else None
            if index_tensor is None:
                index_counts["allocations"] += 1
                indices = [
                    (index * int(populated.numel())) // sample_count
                    for index in range(sample_count)
                ]
                index_tensor = torch.tensor(indices, device=tensor.device, dtype=torch.long)
                if reuse_indices and len(index_constants) < 64:
                    index_constants[key] = index_tensor
            else:
                index_counts["hits"] += 1
            selected_values = populated.index_select(0, index_tensor)
            if batched:
                samples = []
                pending_samples.append((samples, selected_values))
            else:
                samples = [float(value) for value in selected_values.to(
                    device="cpu", dtype=torch.float32).tolist()]
        return {
            "shape": shape,
            "dtype": str(tensor.dtype),
            "used_tokens": used,
            "samples": samples,
        }

    layers: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(kv_cache):
        if not isinstance(layer, Mapping) or "k" not in layer or "v" not in layer:
            raise AdapterError(f"KV cache layer {layer_index} is invalid")
        global_end = scalar_integer(layer.get("global_end_index", 0), "global_end_index")
        local_end = scalar_integer(layer.get("local_end_index", global_end), "local_end_index")
        used = max(global_end, local_end)
        layers.append(
            {
                "layer": layer_index,
                "global_end_index": global_end,
                "local_end_index": local_end,
                "k": tensor_record(layer["k"], used, f"kv[{layer_index}].k"),
                "v": tensor_record(layer["v"], used, f"kv[{layer_index}].v"),
            }
        )
    cross_layers: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(crossattn_cache or ()):
        if not isinstance(layer, Mapping) or "k" not in layer or "v" not in layer:
            raise AdapterError(f"cross-attention cache layer {layer_index} is invalid")
        initialized = bool(layer.get("is_init", False))
        used = int(layer["k"].shape[1]) if initialized else 0
        cross_layers.append(
            {
                "layer": layer_index,
                "is_init": initialized,
                "k": tensor_record(layer["k"], used, f"cross[{layer_index}].k"),
                "v": tensor_record(layer["v"], used, f"cross[{layer_index}].v"),
            }
        )
    if batched:
        values = host_lists([value for _, value in pending_samples], torch=torch, float32=True)
        for (destination, _), samples in zip(pending_samples, values):
            destination.extend(float(value) for value in samples)
    descriptor = {
        "schema": SAMPLED_KV_CACHE_WITNESS_SCHEMA,
        "sampling": SAMPLED_KV_CACHE_WITNESS_SAMPLING,
        "layers": layers,
        "cross_attention_layers": cross_layers,
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if reuse_indices:
        _LAST_WITNESS_INDEX_EXECUTION.set({
            "schema": "call_local_witness_index_execution_v1", "enabled": True,
            **index_counts, "resident_entries": len(index_constants),
            "witness_sha256": digest, "retained_model_values": False})
    return digest, descriptor


def witness_index_execution_payload(witness):
    from .native_rank_indices import enabled
    if not enabled("CFEXP_KV_WITNESS_INDEX_REUSE"):
        return {}
    evidence = _LAST_WITNESS_INDEX_EXECUTION.get()
    if evidence is None or evidence["witness_sha256"] != witness[0]:
        raise AdapterError("enabled witness index reuse lacks matching physical evidence")
    return {"kv_witness_index_reuse": dict(evidence)}


def validate_sampled_kv_cache_witness(
    witness: Mapping[str, Any],
    *,
    expected_layer_count: int,
    expected_self_shape: Sequence[int],
    expected_cross_shape: Sequence[int],
    expected_dtype: str,
    expected_self_used_tokens: int | None = None,
    expected_cross_initialized: bool | None = None,
) -> None:
    """Validate the complete JSON schema of a sampled KV-cache witness.

    This validator intentionally has no Torch dependency.  It is suitable for
    terminal audits that must validate a persisted witness without importing
    the heavyweight model runtime.  ``used_tokens`` follows the witness
    writer's rolling-cache semantics: self-attention cursors are clamped to
    the cache capacity, while a cross-attention cache is either empty or fully
    populated according to its exact boolean ``is_init`` flag.
    """

    def positive_integer(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AdapterError(f"{label} must be a positive integer")
        return value

    def nonnegative_integer(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AdapterError(f"{label} must be a non-negative integer")
        return value

    def expected_shape(value: Sequence[int], label: str) -> list[int]:
        if isinstance(value, (str, bytes)):
            raise AdapterError(f"{label} must be an integer sequence")
        try:
            result = list(value)
        except TypeError as error:
            raise AdapterError(f"{label} must be an integer sequence") from error
        if len(result) < 2:
            raise AdapterError(f"{label} must contain at least two dimensions")
        return [positive_integer(item, f"{label}[{index}]") for index, item in enumerate(result)]

    layer_count = positive_integer(expected_layer_count, "expected_layer_count")
    self_shape = expected_shape(expected_self_shape, "expected_self_shape")
    cross_shape = expected_shape(expected_cross_shape, "expected_cross_shape")
    if not isinstance(expected_dtype, str) or not expected_dtype:
        raise AdapterError("expected_dtype must be a non-empty string")
    if expected_self_used_tokens is not None:
        expected_self_used_tokens = nonnegative_integer(
            expected_self_used_tokens, "expected_self_used_tokens"
        )
        if expected_self_used_tokens > self_shape[1]:
            raise AdapterError("expected_self_used_tokens exceeds self cache capacity")
    if expected_cross_initialized is not None and not isinstance(
        expected_cross_initialized, bool
    ):
        raise AdapterError("expected_cross_initialized must be a boolean")
    if not isinstance(witness, Mapping):
        raise AdapterError("KV cache witness must be an object")
    expected_top_keys = {
        "schema",
        "sampling",
        "layers",
        "cross_attention_layers",
    }
    if set(witness) != expected_top_keys:
        raise AdapterError("KV cache witness has invalid top-level fields")
    if witness.get("schema") != SAMPLED_KV_CACHE_WITNESS_SCHEMA:
        raise AdapterError("KV cache witness has an invalid schema")
    if witness.get("sampling") != SAMPLED_KV_CACHE_WITNESS_SAMPLING:
        raise AdapterError("KV cache witness has an invalid sampling rule")

    layers = witness.get("layers")
    cross_layers = witness.get("cross_attention_layers")
    if not isinstance(layers, list) or len(layers) != layer_count:
        raise AdapterError(
            f"KV cache witness must contain exactly {layer_count} self-attention layers"
        )
    if not isinstance(cross_layers, list) or len(cross_layers) != layer_count:
        raise AdapterError(
            f"KV cache witness must contain exactly {layer_count} cross-attention layers"
        )

    def validate_tensor(
        value: Any,
        *,
        label: str,
        shape: list[int],
        expected_used_tokens: int,
    ) -> None:
        if not isinstance(value, Mapping) or set(value) != {
            "shape",
            "dtype",
            "used_tokens",
            "samples",
        }:
            raise AdapterError(f"{label} must have the exact tensor witness fields")
        actual_shape = value.get("shape")
        if not isinstance(actual_shape, list):
            raise AdapterError(f"{label}.shape must be a list")
        parsed_shape = [
            positive_integer(item, f"{label}.shape[{index}]")
            for index, item in enumerate(actual_shape)
        ]
        if parsed_shape != shape:
            raise AdapterError(
                f"{label}.shape is {parsed_shape!r}, expected {shape!r}"
            )
        if value.get("dtype") != expected_dtype:
            raise AdapterError(
                f"{label}.dtype is {value.get('dtype')!r}, expected {expected_dtype!r}"
            )
        used_tokens = nonnegative_integer(value.get("used_tokens"), f"{label}.used_tokens")
        if used_tokens != expected_used_tokens:
            raise AdapterError(
                f"{label}.used_tokens is {used_tokens}, expected {expected_used_tokens}"
            )
        samples = value.get("samples")
        if not isinstance(samples, list):
            raise AdapterError(f"{label}.samples must be a list")
        populated_numel = used_tokens
        for dimension_index, dimension in enumerate(parsed_shape):
            if dimension_index != 1:
                populated_numel *= dimension
        expected_sample_count = 0 if used_tokens == 0 else min(64, populated_numel)
        if len(samples) != expected_sample_count:
            raise AdapterError(
                f"{label}.samples has length {len(samples)}, expected "
                f"{expected_sample_count}"
            )
        for sample_index, sample in enumerate(samples):
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise AdapterError(f"{label}.samples[{sample_index}] must be numeric")
            try:
                finite = math.isfinite(float(sample))
            except (OverflowError, TypeError, ValueError):
                finite = False
            if not finite:
                raise AdapterError(f"{label}.samples[{sample_index}] must be finite")

    for index, layer in enumerate(layers):
        label = f"KV cache witness self-attention layer {index}"
        if not isinstance(layer, Mapping) or set(layer) != {
            "layer",
            "global_end_index",
            "local_end_index",
            "k",
            "v",
        }:
            raise AdapterError(f"{label} has invalid fields")
        if nonnegative_integer(layer.get("layer"), f"{label}.layer") != index:
            raise AdapterError(f"{label}.layer must equal {index}")
        global_end = nonnegative_integer(
            layer.get("global_end_index"), f"{label}.global_end_index"
        )
        local_end = nonnegative_integer(
            layer.get("local_end_index"), f"{label}.local_end_index"
        )
        used_tokens = min(max(global_end, local_end), self_shape[1])
        if (
            expected_self_used_tokens is not None
            and used_tokens != expected_self_used_tokens
        ):
            raise AdapterError(
                f"{label} populated prefix is {used_tokens}, expected "
                f"{expected_self_used_tokens}"
            )
        validate_tensor(
            layer.get("k"),
            label=f"{label}.k",
            shape=self_shape,
            expected_used_tokens=used_tokens,
        )
        validate_tensor(
            layer.get("v"),
            label=f"{label}.v",
            shape=self_shape,
            expected_used_tokens=used_tokens,
        )

    for index, layer in enumerate(cross_layers):
        label = f"KV cache witness cross-attention layer {index}"
        if not isinstance(layer, Mapping) or set(layer) != {
            "layer",
            "is_init",
            "k",
            "v",
        }:
            raise AdapterError(f"{label} has invalid fields")
        if nonnegative_integer(layer.get("layer"), f"{label}.layer") != index:
            raise AdapterError(f"{label}.layer must equal {index}")
        initialized = layer.get("is_init")
        if not isinstance(initialized, bool):
            raise AdapterError(f"{label}.is_init must be a boolean")
        if (
            expected_cross_initialized is not None
            and initialized is not expected_cross_initialized
        ):
            raise AdapterError(
                f"{label}.is_init is {initialized}, expected "
                f"{expected_cross_initialized}"
            )
        used_tokens = cross_shape[1] if initialized else 0
        validate_tensor(
            layer.get("k"),
            label=f"{label}.k",
            shape=cross_shape,
            expected_used_tokens=used_tokens,
        )
        validate_tensor(
            layer.get("v"),
            label=f"{label}.v",
            shape=cross_shape,
            expected_used_tokens=used_tokens,
        )


@contextmanager
def timed_forward_stages(
    module: Any, *, prefill_count: int, denoise_count: int = 4,
    cache_update_count: int = 0,
) -> Any:
    """Observe forwards without changing model arguments, outputs or RNG.

    Classification uses the audited call position, never timestep positivity:
    Krea's correction schedule includes a real denoising forward at zero.
    Forward spans are inclusive of rank and KV-witness hooks nested inside them.
    The existing trace validators, not this observer, enforce the call schedule.
    """
    timing = current_timing()
    if timing is None:
        yield
        return
    names = (
        ["prefill"] * prefill_count + ["denoise"] * denoise_count
        + ["cache_update"] * cache_update_count
    )
    handles: list[Any] = []
    active: list[Any] = []
    cursor = 0

    def before(_module: Any, _args: Any, _kwargs: Any) -> None:
        nonlocal cursor
        name = names[cursor] if cursor < len(names) else "forward_unclassified"
        active.append(timing.start_span(name, detail=f"forward_{cursor}", cuda=True))
        cursor += 1

    def after(_module: Any, _args: Any, _kwargs: Any, _output: Any) -> None:
        if active:
            timing.end_span(
                active.pop(), status="error" if sys.exc_info()[0] is not None else "completed",
            )

    try:
        handles.append(module.register_forward_pre_hook(before, prepend=True, with_kwargs=True))
        handles.append(module.register_forward_hook(
            after, prepend=False, with_kwargs=True, always_call=True,
        ))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        while active:
            timing.end_span(
                active.pop(), status="error" if sys.exc_info()[0] is not None else "completed",
            )


def attach_adapter_timing(torch: Any, device: Any) -> None:
    timing = current_timing()
    if timing is not None:
        timing.attach_cuda(torch, device)


def validate_self_forcing_forward_trace(
    trace: Sequence[Mapping[str, Any]], *, history_chunks: int, reused_history_chunks: int = 0
) -> tuple[Mapping[str, Any], ...]:
    """Validate one Draft request's actual history replay and denoising trace.

    Legacy calls replay every preceding three-latent
    chunk once at timestep zero to rebuild its own rolling KV cache, then
    executes the frozen four denoising forwards and terminal clean-cache
    forward at the current position.  Checking the complete sequence makes a
    persisted Draft proposal evidence that its KV belongs to its exact prefix,
    rather than merely trusting the reported cache digest. Incremental callers
    must first validate their reuse evidence and supply reused_history_chunks;
    omitted prefix forwards are never synthesized in the physical trace.
    """

    if (
        isinstance(history_chunks, bool)
        or not isinstance(history_chunks, int)
        or not 0 <= history_chunks < max(SUPPORTED_CORRECTION_CHUNK_COUNTS)
    ):
        raise AdapterError("history_chunks must be an integer in [0, 119]")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise AdapterError("Draft forward trace must be a sequence")
    if type(reused_history_chunks) is not int or not 0 <= reused_history_chunks <= history_chunks:
        raise AdapterError("invalid reused history length")
    replayed_chunks = history_chunks - reused_history_chunks
    expected_length = replayed_chunks + len(SELF_FORCING_DENOISING_SCHEDULE)
    if len(trace) != expected_length:
        raise AdapterError(
            f"Draft forward trace has length {len(trace)}, expected {expected_length}"
        )

    def validate_entry(
        value: Mapping[str, Any],
        *,
        index: int,
        timestep: float,
        current_start: int,
    ) -> None:
        if not isinstance(value, Mapping) or set(value) != {
            "timestep",
            "batch_timesteps_equal",
            "current_start",
        }:
            raise AdapterError(f"Draft forward trace entry {index} has invalid fields")
        actual_timestep = value.get("timestep")
        if (
            isinstance(actual_timestep, bool)
            or not isinstance(actual_timestep, (int, float))
            or not math.isfinite(float(actual_timestep))
            or float(actual_timestep) != timestep
        ):
            raise AdapterError(
                f"Draft forward trace entry {index} timestep is not {timestep}"
            )
        if value.get("batch_timesteps_equal") is not True:
            raise AdapterError(
                f"Draft forward trace entry {index} has unequal batch timesteps"
            )
        actual_start = value.get("current_start")
        if (
            isinstance(actual_start, bool)
            or not isinstance(actual_start, int)
            or actual_start != current_start
        ):
            raise AdapterError(
                f"Draft forward trace entry {index} current_start is not "
                f"{current_start}"
            )

    for offset, chunk_index in enumerate(range(reused_history_chunks, history_chunks)):
        validate_entry(
            trace[offset],
            index=offset,
            timestep=0.0,
            current_start=chunk_index * SELF_FORCING_TOKENS_PER_CHUNK,
        )
    current_start = history_chunks * SELF_FORCING_TOKENS_PER_CHUNK
    denoising = trace[replayed_chunks:]
    for offset, timestep in enumerate(SELF_FORCING_DENOISING_SCHEDULE):
        validate_entry(
            denoising[offset],
            index=replayed_chunks + offset,
            timestep=timestep,
            current_start=current_start,
        )
    return tuple(denoising)


def _verified_artifacts(request: Mapping[str, Any]) -> dict[str, Path]:
    items = request.get("verified_artifacts")
    if not isinstance(items, list) or not items:
        raise AdapterError("request.verified_artifacts must be a non-empty list")
    result: dict[str, Path] = {}
    for index, item in enumerate(items):
        label = f"request.verified_artifacts[{index}]"
        if not isinstance(item, Mapping):
            raise AdapterError(f"{label} must be an object")
        name = item.get("name")
        resolved_path = item.get("resolved_path")
        expected = item.get("sha256")
        expected_size = item.get("size")
        if not isinstance(name, str) or not name or name in result:
            raise AdapterError(f"{label}.name must be non-empty and unique")
        if not isinstance(resolved_path, str) or not resolved_path:
            raise AdapterError(f"{label}.resolved_path is invalid")
        if not _is_sha256(expected):
            raise AdapterError(f"{label}.sha256 is invalid")
        path = Path(resolved_path).expanduser().resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise AdapterError(f"{label} is not a regular non-symlink file")
        metadata = path.stat()
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise AdapterError(f"{label}.size is invalid")
        if metadata.st_size != expected_size:
            raise AdapterError(f"artifact size changed after campaign preflight: {name}")
        actual = _sha256_file(path)
        if actual != expected:
            raise AdapterError(f"artifact hash changed after campaign preflight: {name}")
        result[name] = path
    return result


def _artifact(artifacts: Mapping[str, Path], name: Any, label: str) -> Path:
    if not isinstance(name, str) or name not in artifacts:
        raise AdapterError(f"{label} names an unverified artifact: {name!r}")
    return artifacts[name]


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract regular files/directories only, rejecting links and traversal."""

    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            members = handle.getmembers()
            if not members:
                raise AdapterError("official source archive is empty")
            for member in members:
                member_path = Path(member.name)
                if (
                    not member.name
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.name in seen
                ):
                    raise AdapterError(f"unsafe or duplicate tar member: {member.name!r}")
                if not (member.isdir() or member.isfile()):
                    raise AdapterError(
                        "source archive may contain only directories and regular files: "
                        f"{member.name!r}"
                    )
                seen.add(member.name)
            for member in members:
                target = _inside(destination, destination / member.name, "tar member", strict=False)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise AdapterError(f"cannot read tar member {member.name!r}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(target, member.mode & 0o777)
    except (tarfile.TarError, OSError) as error:
        raise AdapterError(f"cannot safely extract source archive {archive}: {error}") from error


def _safe_extract_wheel(wheel: Path, destination: Path) -> None:
    """Extract a pinned binary wheel without accepting links or path traversal."""

    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(wheel, mode="r") as handle:
            members = handle.infolist()
            if not members:
                raise AdapterError("runtime wheel is empty")
            for member in members:
                member_path = Path(member.filename)
                unix_mode = member.external_attr >> 16
                if (
                    not member.filename
                    or "\\" in member.filename
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.filename in seen
                ):
                    raise AdapterError(
                        f"unsafe or duplicate wheel member: {member.filename!r}"
                    )
                if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                    raise AdapterError(
                        f"runtime wheel may not contain links: {member.filename!r}"
                    )
                seen.add(member.filename)
            for member in members:
                target = _inside(
                    destination,
                    destination / member.filename,
                    "wheel member",
                    strict=False,
                )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member, mode="r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                unix_mode = member.external_attr >> 16
                if unix_mode:
                    os.chmod(target, unix_mode & 0o777)
    except (zipfile.BadZipFile, OSError) as error:
        raise AdapterError(f"cannot safely extract runtime wheel {wheel}: {error}") from error


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise AdapterError(
            f"{label} keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _load_runtime_config(
    request: Mapping[str, Any], artifacts: Mapping[str, Path]
) -> dict[str, Any]:
    spec_path_text = request.get("artifact_spec_path")
    expected_digest = request.get("artifact_spec_sha256")
    if not isinstance(spec_path_text, str) or not _is_sha256(expected_digest):
        raise AdapterError("request artifact-spec identity is invalid")
    spec_path = Path(spec_path_text).expanduser().resolve(strict=True)
    if spec_path.is_symlink() or not spec_path.is_file():
        raise AdapterError("artifact specification must be a regular non-symlink file")
    if _sha256_file(spec_path) != expected_digest:
        raise AdapterError("artifact specification changed after campaign preflight")
    spec = _load_json(spec_path, "artifact specification")
    if spec.get("schema") != "erank_frame_artifacts_v1" or spec.get("status") != "frozen":
        raise AdapterError("adapter requires a frozen erank_frame_artifacts_v1 specification")
    if spec.get("adapter_module") != "cfexp.self_forcing_frame_adapter":
        raise AdapterError("artifact specification selects a different adapter module")
    config = spec.get("adapter_config")
    if not isinstance(config, dict):
        raise AdapterError("frozen artifact specification has no adapter_config object")
    _strict_keys(
        config,
        {
            "schema",
            "official_source_commit",
            "source_archive_artifact",
            "source_archive_subdirectory",
            "python_wheels",
            "runtime_files",
            "generation",
            "musiq",
        },
        "adapter_config",
    )
    if config["schema"] != RUNTIME_CONFIG_SCHEMA:
        raise AdapterError(f"adapter_config.schema must be {RUNTIME_CONFIG_SCHEMA!r}")
    if config["official_source_commit"] != OFFICIAL_SOURCE_COMMIT:
        raise AdapterError(f"official source commit must be {OFFICIAL_SOURCE_COMMIT}")
    _artifact(artifacts, config["source_archive_artifact"], "source_archive_artifact")
    _safe_relative(config["source_archive_subdirectory"], "source_archive_subdirectory")
    python_wheels = config["python_wheels"]
    expected_wheels = [
        {
            "artifact": "flash_attn_wheel",
            "module": "flash_attn",
            "version": "2.8.3",
        }
    ]
    if python_wheels != expected_wheels:
        raise AdapterError(
            "adapter_config.python_wheels must pin flash_attn 2.8.3 exactly"
        )
    _artifact(artifacts, python_wheels[0]["artifact"], "python_wheels[0].artifact")
    runtime_files = config["runtime_files"]
    if not isinstance(runtime_files, list) or not runtime_files:
        raise AdapterError("adapter_config.runtime_files must pin the base-model files")
    destinations: set[str] = set()
    for index, mapping in enumerate(runtime_files):
        if not isinstance(mapping, dict):
            raise AdapterError(f"runtime_files[{index}] must be an object")
        _strict_keys(mapping, {"artifact", "destination"}, f"runtime_files[{index}]")
        _artifact(artifacts, mapping["artifact"], f"runtime_files[{index}].artifact")
        destination = _safe_relative(
            mapping["destination"], f"runtime_files[{index}].destination"
        ).as_posix()
        if destination in destinations:
            raise AdapterError(f"duplicate runtime destination {destination!r}")
        destinations.add(destination)
    generation = config["generation"]
    if not isinstance(generation, dict):
        raise AdapterError("adapter_config.generation must be an object")
    _strict_keys(
        generation,
        {
            "config_artifact",
            "checkpoint_artifact",
            "checkpoint_state_key",
            "dtype",
            "noise_shape",
            "decoded_frames_before_trim",
            "output_frames",
            "expected_blocks",
            "num_frame_per_block",
            "independent_first_frame",
            "first_denoising_timestep",
            "seed_mode",
            "low_memory",
            "model_id",
            "required_config_values",
        },
        "adapter_config.generation",
    )
    _artifact(artifacts, generation["config_artifact"], "generation.config_artifact")
    _artifact(
        artifacts, generation["checkpoint_artifact"], "generation.checkpoint_artifact"
    )
    if not isinstance(generation["checkpoint_state_key"], str) or not generation[
        "checkpoint_state_key"
    ]:
        raise AdapterError("generation.checkpoint_state_key must be explicit")
    if generation["dtype"] != "bfloat16":
        raise AdapterError("this formal adapter requires dtype='bfloat16'")
    noise_shape = generation["noise_shape"]
    if noise_shape != [93, 16, 60, 104]:
        raise AdapterError("generation.noise_shape must be [93,16,60,104] for 31x3 blocks")
    frozen_generation = {
        "output_frames": 360,
        "expected_blocks": 31,
        "num_frame_per_block": 3,
        "independent_first_frame": False,
        "first_denoising_timestep": 1000,
        "seed_mode": "reset_per_prompt_full_tensor_before_native_loop",
        "low_memory": False,
        "model_id": "self_forcing_1.3b",
    }
    for field, expected in frozen_generation.items():
        if generation[field] != expected:
            raise AdapterError(f"generation.{field} must equal {expected!r}")
    decoded = generation["decoded_frames_before_trim"]
    if isinstance(decoded, bool) or not isinstance(decoded, int) or decoded < 360:
        raise AdapterError("generation.decoded_frames_before_trim must be an integer >=360")
    required_values = generation["required_config_values"]
    if not isinstance(required_values, dict) or not required_values:
        raise AdapterError("generation.required_config_values must be a non-empty object")
    musiq = config["musiq"]
    if not isinstance(musiq, dict):
        raise AdapterError("adapter_config.musiq must be an object")
    _strict_keys(
        musiq,
        {"checkpoint_artifact", "implementation", "resize_mode", "raw_score_divisor"},
        "adapter_config.musiq",
    )
    _artifact(artifacts, musiq["checkpoint_artifact"], "musiq.checkpoint_artifact")
    if musiq["implementation"] != "vbench_long_pyiqa_musiq_spaq":
        raise AdapterError("unsupported MUSIQ implementation")
    if musiq["resize_mode"] != "longer" or musiq["raw_score_divisor"] != 100:
        raise AdapterError("MUSIQ must use VBench 'longer' preprocessing and raw/100")
    return config


def _verify_protocol_shard(request: Mapping[str, Any]) -> None:
    protocol_text = request.get("protocol_path")
    expected_digest = request.get("protocol_sha256")
    if not isinstance(protocol_text, str) or not _is_sha256(expected_digest):
        raise AdapterError("request protocol identity is invalid")
    protocol_path = Path(protocol_text).expanduser().resolve(strict=True)
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise AdapterError("protocol must be a regular non-symlink file")
    if _sha256_file(protocol_path) != expected_digest:
        raise AdapterError("protocol changed after campaign preflight")
    protocol = _load_json(protocol_path, "campaign protocol")
    if protocol.get("status") != "frozen":
        raise AdapterError("campaign protocol is not frozen")
    protocol_prompts = protocol.get("prompts")
    request_prompts = request.get("prompts")
    worker_index = request.get("worker_index")
    if (
        not isinstance(protocol_prompts, list)
        or len(protocol_prompts) != 48
        or not isinstance(request_prompts, list)
        or isinstance(worker_index, bool)
        or not isinstance(worker_index, int)
        or not 0 <= worker_index < 4
    ):
        raise AdapterError("protocol/request prompt shard identity is invalid")
    expected_shard = protocol_prompts[worker_index * 12 : (worker_index + 1) * 12]
    if request_prompts != expected_shard:
        raise AdapterError("request prompts do not equal the frozen contiguous protocol shard")


def _verify_frozen_parameters(request: Mapping[str, Any]) -> None:
    parameters = request.get("parameters")
    if not isinstance(parameters, Mapping):
        raise AdapterError("request.parameters must be an object")
    for field, expected in FROZEN_PARAMETERS.items():
        if parameters.get(field) != expected:
            raise AdapterError(f"request parameter {field} must equal {expected!r}")


def _verify_result_contract(request: Mapping[str, Any], output_dir: Path) -> None:
    contract = request.get("result_contract")
    if not isinstance(contract, Mapping):
        raise AdapterError("request.result_contract must be an object")
    if contract.get("schema") != RESULT_SCHEMA or contract.get("expected_rows") != 12 * 31:
        raise AdapterError("request result contract schema/row count is invalid")
    result_path = contract.get("path")
    if not isinstance(result_path, str) or Path(result_path).resolve() != output_dir / "result.json":
        raise AdapterError("request result path must be the worker result.json")


def _materialize_source(
    *,
    run_dir: Path,
    worker_index: int,
    artifacts: Mapping[str, Path],
    config: Mapping[str, Any],
    scratch_suffix: str | None = None,
) -> tuple[Path, Path]:
    if scratch_suffix is not None and (
        not isinstance(scratch_suffix, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", scratch_suffix) is None
    ):
        raise AdapterError("scratch_suffix must be a safe lowercase identifier")
    scratch_name = f"self-forcing-worker-{worker_index:02d}"
    if scratch_suffix is not None:
        scratch_name = f"{scratch_name}-{scratch_suffix}"
    scratch = run_dir / "tmp" / scratch_name
    if scratch.exists():
        raise AdapterError(f"worker scratch path already exists: {scratch}")
    archive = _artifact(artifacts, config["source_archive_artifact"], "source archive")
    _safe_extract_tar(archive, scratch)
    subdirectory = _safe_relative(config["source_archive_subdirectory"], "source subdirectory")
    source_root = _inside(scratch, scratch / subdirectory, "official source root")
    required_source_files = (
        source_root / "inference.py",
        source_root / "pipeline" / "causal_inference.py",
    )
    if any(path.is_symlink() or not path.is_file() for path in required_source_files):
        raise AdapterError(
            "official source archive does not contain the expected inference interface"
        )
    python_site = source_root / ".cfexp-site-packages"
    wheel_name = config["python_wheels"][0]["artifact"]
    _safe_extract_wheel(
        _artifact(artifacts, wheel_name, "runtime Python wheel"), python_site
    )
    for mapping in config["runtime_files"]:
        source = _artifact(artifacts, mapping["artifact"], "runtime artifact")
        relative = _safe_relative(mapping["destination"], "runtime destination")
        destination = _inside(
            source_root, source_root / relative, "runtime destination", strict=False
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise AdapterError(f"runtime destination already exists: {relative.as_posix()}")
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if _sha256_file(destination) != _sha256_file(source):
            raise AdapterError(f"materialized runtime file changed: {relative.as_posix()}")
    return source_root, python_site


def _nested_value(value: Mapping[str, Any], dotted: str) -> Any:
    if not dotted or any(not part for part in dotted.split(".")):
        raise AdapterError(f"invalid required config path {dotted!r}")
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise AdapterError(f"generation config is missing required field {dotted!r}")
        current = current[part]
    return current


@dataclass
class GeneratedTrajectory:
    frames: Any
    captures: list[CaptureEvent]


class FrameBackend(Protocol):
    def generate(self, *, prompt: str, seed: int) -> GeneratedTrajectory: ...

    def frame_count(self, frames: Any) -> int: ...

    def write_png(self, frames: Any, frame_index: int, path: Path) -> None: ...

    def score_png_raw(self, path: Path) -> float: ...

    def write_video(self, frames: Any, frame_count: int, path: Path, fps: int) -> None: ...

    def close(self) -> None: ...


class OfficialSelfForcingBackend:
    """Heavy runtime loaded only after all formal inputs have been verified."""

    supports_history_kv_widths = True

    def __init__(
        self,
        *,
        source_root: Path,
        artifacts: Mapping[str, Path],
        config: Mapping[str, Any],
        device_name: str,
        python_site: Path,
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
        generation = config["generation"]
        musiq_config = config["musiq"]
        python_site_text = str(python_site)
        if python_site_text in sys.path:
            sys.path.remove(python_site_text)
        sys.path.insert(0, python_site_text)
        try:
            import flash_attn  # type: ignore
            import torch  # type: ignore
            from omegaconf import OmegaConf  # type: ignore
            from torchvision.io import read_image, write_png, write_video  # type: ignore
            from torchvision.transforms import Resize  # type: ignore
            from pyiqa.archs.musiq_arch import MUSIQ  # type: ignore
        except ImportError as error:
            if python_site_text in sys.path:
                sys.path.remove(python_site_text)
            raise AdapterError(
                "the frozen runtime is missing torch/flash-attn/OmegaConf/torchvision/pyiqa"
            ) from error
        expected_flash_version = config["python_wheels"][0]["version"]
        if getattr(flash_attn, "__version__", None) != expected_flash_version:
            if python_site_text in sys.path:
                sys.path.remove(python_site_text)
            raise AdapterError(
                "the pinned flash-attn module version is not "
                f"{expected_flash_version}"
            )
        self.torch = torch
        self.read_image = read_image
        self.write_png_impl = write_png
        self.write_video_impl = write_video
        self.resize_type = Resize
        self.device = torch.device(device_name)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise AdapterError("official Self-Forcing execution requires CUDA")
        if torch.cuda.device_count() != 1:
            raise AdapterError("each adapter worker must see exactly one CUDA device")

        old_cwd = Path.cwd()
        source_text = str(source_root)
        if source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)
        os.chdir(source_root)
        self._source_root = source_root
        self._python_site = python_site
        self._old_cwd = old_cwd
        try:
            pipeline_module = importlib.import_module("pipeline.causal_inference")
            pipeline_type = getattr(pipeline_module, "CausalInferencePipeline")
            config_path = _artifact(
                artifacts, generation["config_artifact"], "generation config"
            )
            user_config = OmegaConf.load(str(config_path))
            default_config = OmegaConf.load(str(source_root / "configs" / "default_config.yaml"))
            merged = OmegaConf.merge(default_config, user_config)
            container = OmegaConf.to_container(merged, resolve=True)
            if not isinstance(container, dict):
                raise AdapterError("merged Self-Forcing config is not an object")
            for dotted, expected in generation["required_config_values"].items():
                if _nested_value(container, dotted) != expected:
                    raise AdapterError(
                        f"Self-Forcing config {dotted!r} does not equal frozen value {expected!r}"
                    )
            if container.get("num_frame_per_block") != 3:
                raise AdapterError("Self-Forcing config must set num_frame_per_block=3")
            if container.get("independent_first_frame") is not False:
                raise AdapterError("Self-Forcing config must set independent_first_frame=false")
            model_kwargs = container.get("model_kwargs")
            if not isinstance(model_kwargs, Mapping):
                raise AdapterError("Self-Forcing config must explicitly define model_kwargs")
            local_attention = model_kwargs.get("local_attn_size")
            sink_size = model_kwargs.get("sink_size")
            if local_attention != 21:
                raise AdapterError("long generation requires local_attn_size=21")
            if sink_size != 0:
                raise AdapterError("the classic Self-Forcing context requires sink_size=0")
            conditioning_binding: Mapping[str, Any] | None = None
            cached_text_encoder: Any | None = None
            if conditioning_cache_config is not None:
                from .conditioning_cache import (
                    TorchConditioningTensorCodec,
                    make_torch_conditioner,
                    seal_shared_conditioning_cache,
                )

                live_text_encoder_type = getattr(pipeline_module, "WanTextEncoder")
                live_text_encoder: Any | None = None
                try:
                    live_text_encoder = live_text_encoder_type()
                    live_text_encoder = live_text_encoder.to(dtype=torch.bfloat16)
                    live_text_encoder.to(device=self.device)
                    live_text_encoder.eval().requires_grad_(False)
                    codec = TorchConditioningTensorCodec(torch)
                    loaded_conditioning = seal_shared_conditioning_cache(
                        conditioning_cache_config,
                        run_dir=str(conditioning_cache_config["run_dir"]),
                        source_root=source_root,
                        live_encode=lambda prompts: live_text_encoder(
                            text_prompts=list(prompts)
                        ),
                        codec=codec,
                    )
                    cached_text_encoder = make_torch_conditioner(
                        loaded_conditioning, torch_module=torch
                    )
                    conditioning_binding = loaded_conditioning.audit_binding()
                finally:
                    if live_text_encoder is not None:
                        del live_text_encoder
                    gc.collect()
                    torch.cuda.empty_cache()
            if cached_text_encoder is None:
                # Preserve the established constructor path byte-for-byte when
                # the acceleration gate is disabled.  In particular, do not
                # rely on an optional upstream ``text_encoder=None`` contract.
                pipeline = pipeline_type(merged, device=self.device)
            else:
                pipeline = pipeline_type(
                    merged,
                    device=self.device,
                    text_encoder=cached_text_encoder,
                )
            checkpoint_path = _artifact(
                artifacts, generation["checkpoint_artifact"], "generation checkpoint"
            )
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            state_key = generation["checkpoint_state_key"]
            if not isinstance(checkpoint, Mapping) or state_key not in checkpoint:
                available = (
                    sorted(str(key) for key in checkpoint)
                    if isinstance(checkpoint, Mapping)
                    else []
                )
                raise AdapterError(
                    f"checkpoint does not contain configured state key {state_key!r}; "
                    f"available={available[:20]}"
                )
            pipeline.generator.load_state_dict(checkpoint[state_key], strict=True)
            del checkpoint
            pipeline = pipeline.to(dtype=torch.bfloat16)
            pipeline.text_encoder.to(device=self.device)
            pipeline.generator.to(device=self.device)
            pipeline.vae.to(device=self.device)
            pipeline.eval()
            from .host_kv_cursors import install as install_host_kv_cursors
            self.host_kv_cursor_optimization = install_host_kv_cursors(pipeline, torch=torch)
            actual_first = float(pipeline.denoising_step_list[0].item())
            if actual_first != float(generation["first_denoising_timestep"]):
                raise AdapterError(
                    f"native first denoising timestep is {actual_first}, expected 1000"
                )
            blocks = pipeline.generator.model.blocks
            if len(blocks) <= max(LAYER_INDICES):
                raise AdapterError(f"backbone exposes only {len(blocks)} transformer blocks")
            self.pipeline = pipeline
            from .native_rank_indices import NativeRankIndices
            self.rank_indices = NativeRankIndices(torch=torch, device=self.device,
                index_function=deterministic_token_indices)
            reuse_enabled, fork_enabled = draft_kv_feature_flags()
            self._incremental_draft = None
            if reuse_enabled:
                from .incremental_draft import IncrementalDraftExecutor
                scope = hashlib.sha256(json.dumps({
                    "config": config, "checkpoint": _sha256_file(checkpoint_path),
                    "source": OFFICIAL_SOURCE_COMMIT, "dtype": "bfloat16",
                }, sort_keys=True).encode()).hexdigest()
                self._incremental_draft = IncrementalDraftExecutor(
                    pipeline=pipeline, torch=torch, scope_sha256=scope, fork_enabled=fork_enabled)
            self.conditioning_cache_binding = conditioning_binding
            self.generation = generation
            self._torch_generator = None
            self._active_capture: StepZeroCapture | None = None
            self._hook_handles: list[Any] = []
            self._install_hooks(blocks)

            musiq_checkpoint = _artifact(
                artifacts, musiq_config["checkpoint_artifact"], "MUSIQ checkpoint"
            )
            self.musiq = MUSIQ(pretrained_model_path=str(musiq_checkpoint))
            self.musiq.to(self.device)
            self.musiq.eval()
            for parameter in self.musiq.parameters():
                parameter.requires_grad_(False)
        except BaseException:
            os.chdir(old_cwd)
            if source_text in sys.path:
                sys.path.remove(source_text)
            if python_site_text in sys.path:
                sys.path.remove(python_site_text)
            raise

    def _timestep_values(self, value: Any) -> list[float]:
        tensor = value.detach().reshape(-1).to(device="cpu")
        return [float(item) for item in tensor.tolist()]

    def _sample_hidden(self, hidden: Any) -> Any:
        with stage("rank", detail="sample", cuda=True):
            torch = self.torch
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise AdapterError(
                    "hooked transformer output must be a [1,current_visual_tokens,hidden] tensor"
                )
            helper = getattr(self, "rank_indices", None)
            if helper is not None and helper.enabled:
                return helper.sample(hidden)
            indices = deterministic_token_indices(int(hidden.shape[1]), 256)
            index_tensor = torch.tensor(indices, dtype=torch.long, device=hidden.device)
            return hidden[0].index_select(0, index_tensor).detach().to(dtype=torch.float32)

    def _rank_sample(self, sample: Any) -> EffectiveRankMeasurement:
        with stage("rank", detail="eigensolve_and_entropy", cuda=True):
            torch = self.torch
            gram = sample @ sample.transpose(0, 1)
            eigenvalues = torch.linalg.eigvalsh(gram).detach().to(device="cpu", dtype=torch.float64)
            spectrum = tuple(float(item) for item in eigenvalues.tolist())
            rank = entropy_effective_rank_from_gram_eigenvalues(
                spectrum, relative_cutoff=EFFECTIVE_RANK_RELATIVE_CUTOFF
            )
            return EffectiveRankMeasurement(
                effective_rank=rank,
                gram_eigenvalues=spectrum,
            )

    def _retain_full_hidden(self, hidden: Any) -> Any:
        torch = self.torch
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise AdapterError(
                "hooked transformer output must be a [1,current_visual_tokens,hidden] tensor"
            )
        with stage("transfer", detail="draft_full_hidden_d2h", cuda=True):
            return hidden[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()

    def _install_hooks(self, blocks: Sequence[Any]) -> None:
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
            self._active_capture.begin_call(
                self._timestep_values(timestep), current_start=current_start
            )

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
        """Generate one native three-latent Draft chunk from an explicit prefix.

        The native pipeline is the default. Opt-in incremental execution reuses
        its generator and scheduler with bound clean KV. The VAE call is
        replaced only for this latent-only request; :meth:`decode_latents`
        retains the real decoder for offline scoring.
        """

        torch = self.torch
        rank_index_helper = getattr(self, "rank_indices", None)
        rank_index_before = (rank_index_helper.snapshot()
            if rank_index_helper is not None and rank_index_helper.enabled else None)
        attach_adapter_timing(torch, self.device)
        if early_should_replace is not None and (
            capture_hidden or kv_branch_action != "none" or self._incremental_draft is None
        ):
            raise AdapterError("early execution requires native reuse, policy-only capture, and one branch")
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
            if early_should_replace is not None:
                from .early_draft import EarlyDraftExecutor
                early = getattr(self, "_early_draft", None)
                if early is None:
                    early = self._early_draft = EarlyDraftExecutor(pipeline=self.pipeline,
                        torch=torch, scope_sha256=executor.scope_sha256)
                incremental = early.generate(prompt=prompt, history=history, noise=noise,
                    history_kv_widths=history_kv_widths,
                    forward_context=lambda count: timed_forward_stages(
                        self.pipeline.generator, prefill_count=count, cache_update_count=1))
                early_execution = {k:v for k,v in incremental.items() if k not in ("latent", "probe")}
                early_execution["schema"] = "draft_early_execution_v1"
                from .incremental_draft import latent_digest
                reuse_evidence = {
                    "schema": "draft_kv_reuse_v2",
                    "history_kv_widths": list(history_kv_widths),
                    "reused_chunks": incremental["reused_chunks"],
                    "replayed_chunks": incremental["replayed_chunks"],
                    "scope_sha256": early.scope,
                    "history_tensor_sha256": [latent_digest(history[:,3*i:3*i+3], torch)
                                              for i in range(chunk_index)],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
                if incremental["stopped"]:
                    return {"stopped": True, "probe": incremental["probe"],
                            "kv_reuse": reuse_evidence, "early_execution": early_execution, "forward_trace": forward_trace}
                clean = incremental["latent"]
                output = clean if history is None else torch.cat([history, clean], dim=1)
            elif executor is not None:
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
                with timed_forward_stages(
                    self.pipeline.generator, prefill_count=chunk_index, cache_update_count=1,
                ), torch.inference_mode():
                    _ignored_video, output = self.pipeline.inference(
                        noise=noise, text_prompts=[prompt], initial_latent=history,
                        return_latents=True, profile=False, low_memory=False)
            events = list(self._active_capture.events)
        finally:
            self.pipeline.vae.decode_to_pixel = real_decode
            audit_handle.remove()
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
            **({"early_execution": early_execution} if early_should_replace is not None else {}),
        }

    def reset_early_draft(self):
        early = getattr(self, "_early_draft", None)
        if self._active_capture is not None or (early is not None and early.pending is not None):
            raise AdapterError("cannot reset an active or tentative early Draft")
        self._early_draft = None

    def commit_early_target(self, *, prompt: str, history_latents: Any | None,
                            history_kv_widths: tuple[int, ...], latent: Any) -> dict[str, Any]:
        """Accept a replacement through Draft's native clean forward, not Target KV."""
        early = getattr(self, "_early_draft", None)
        if early is None or self._active_capture is not None:
            raise AdapterError("external commit requires an idle early Draft session")
        attach_adapter_timing(self.torch, self.device)
        self._active_capture = StepZeroCapture(layers=LAYER_INDICES, target_timestep=1000,
            sample_hidden=self._sample_hidden, rank_sample=self._rank_sample,
            retain_rank_layer=EFFECTIVE_RANK_LAYER, policy_rank_only=True)
        try:
            return early.commit_external(prompt=prompt, history=history_latents,
                history_kv_widths=history_kv_widths, latent=latent,
                forward_context=lambda count: timed_forward_stages(
                    self.pipeline.generator, prefill_count=0, denoise_count=0, cache_update_count=count))
        finally:
            self._active_capture = None

    def resume_early_target_prefix(self, *, prompt, history_latents, history_kv_widths,
            first_clean_prediction, renoise_first, renoise_second, renoise_third, verify_parity=False):
        """Resume Draft at its native second timestep; never import Target KV or accept a prefix."""
        early=getattr(self,'_early_draft',None)
        if early is None or early.pending is None or self._active_capture is not None:
            raise AdapterError('Target prefix requires an idle pending early Draft')
        t,p=self.torch,self.pipeline
        from .target_prefix import native_draft_schedule
        steps,_=native_draft_schedule(p)
        shape=early.pending['shape']
        for value in (first_clean_prediction,renoise_first,renoise_second,renoise_third):
            if (not isinstance(value,t.Tensor) or tuple(value.shape)!=shape or value.dtype!=t.bfloat16
                    or not bool(t.isfinite(value.float()).all().item())):
                raise AdapterError('Target handoff has invalid tensor geometry or values')
        self._active_capture=StepZeroCapture(layers=LAYER_INDICES,target_timestep=1000,
            sample_hidden=self._sample_hidden,rank_sample=self._rank_sample,
            retain_rank_layer=EFFECTIVE_RANK_LAYER,policy_rank_only=True)
        try:
            with t.inference_mode():
                clean,noise_first,noise_second,noise_third=(v.to(self.device).contiguous() for v in
                    (first_clean_prediction,renoise_first,renoise_second,renoise_third))
                noisy_suffix=p.scheduler.add_noise(clean.flatten(0,1),noise_first.flatten(0,1),
                    steps[1]*t.ones((3,),device=self.device,dtype=t.long)).unflatten(0,clean.shape[:2])
                if verify_parity:
                    from .prefix_oracle import suffix_oracle,check_suffix
                    expected,reference=suffix_oracle(pipeline=p,torch=t,prompt=prompt,history=history_latents,
                        widths=history_kv_widths,noisy_suffix=noisy_suffix,noise_second=noise_second,noise_third=noise_third)
                result=early.resume_external_prefix(prompt=prompt,history=history_latents,
                    history_kv_widths=history_kv_widths,noisy_latent_suffix=noisy_suffix,
                    renoise_second=noise_second,renoise_third=noise_third,
                    forward_context=lambda count:timed_forward_stages(
                        p.generator,prefill_count=0,denoise_count=3,cache_update_count=1))
                if verify_parity:
                    result['diagnostic_oracle']=check_suffix(actual=result['latent'],expected=expected,
                        actual_cache=early.payload,expected_cache=reference,torch=t)
                    del reference,expected
                witness=sampled_kv_cache_witness(early.payload['self'],early.payload['cross'])
                result['kv_witness_sha256'],result['kv_witness']=witness
                result['latent']=result['latent'].detach().to('cpu').contiguous()
                return result
        finally:
            self._active_capture=None

    def generate_t_stitch_step_chunk(
        self,
        *,
        prompt: str,
        history_latents: Any | None,
        initial_noise: Any,
        renoise_750: Any,
        internal_seed: int,
        chunk_index: int,
    ) -> dict[str, Any]:
        """Run exactly the Draft timestep-1000 leg of T-Stitch.

        The official causal pipeline is reused with its denoising schedule
        temporarily narrowed to one step.  The resulting clean Draft estimate
        is re-noised to timestep 750 with an explicit caller-provided tensor;
        both tensors are returned so the process boundary cannot conceal a
        second Draft step or substitute a stitched video.
        """

        torch = self.torch
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
        if (
            isinstance(internal_seed, bool)
            or not isinstance(internal_seed, int)
            or internal_seed < 0
        ):
            raise AdapterError("internal_seed must be a non-negative integer")
        expected_shape = (1, 3, 16, 60, 104)
        for label, value in (
            ("initial_noise", initial_noise),
            ("renoise_750", renoise_750),
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise AdapterError(f"{label} must have shape {expected_shape}")
        expected_history_frames = 3 * chunk_index
        if chunk_index == 0:
            if history_latents is not None and int(history_latents.shape[1]) != 0:
                raise AdapterError("chunk 0 requires an empty latent history")
            history = None
        else:
            expected_history_shape = (1, expected_history_frames, 16, 60, 104)
            if (
                not isinstance(history_latents, torch.Tensor)
                or tuple(history_latents.shape) != expected_history_shape
            ):
                raise AdapterError(
                    f"history_latents must have shape {expected_history_shape}"
                )
            history = history_latents.to(
                device=self.device, dtype=torch.bfloat16
            ).contiguous()
        noise = initial_noise.to(device=self.device, dtype=torch.bfloat16).contiguous()
        noise_750 = renoise_750.to(
            device=self.device, dtype=torch.bfloat16
        ).contiguous()
        if not bool(torch.isfinite(noise.float()).all().item()) or not bool(
            torch.isfinite(noise_750.float()).all().item()
        ):
            raise AdapterError("T-Stitch Draft noise contains non-finite values")

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
            retain_rank_layer=EFFECTIVE_RANK_LAYER,
        )
        forward_trace: list[dict[str, Any]] = []
        kv_witness: tuple[str, dict[str, Any]] | None = None

        def audit_pre_hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
            nonlocal kv_witness
            timestep = kwargs.get("timestep")
            current_start = kwargs.get("current_start")
            if (
                timestep is None
                or isinstance(current_start, bool)
                or not isinstance(current_start, int)
            ):
                raise AdapterError(
                    "T-Stitch Draft audit hook omitted timestep/current_start"
                )
            values = self._timestep_values(timestep)
            forward_trace.append(
                {
                    "timestep": values[0],
                    "batch_timesteps_equal": len(set(values)) == 1,
                    "current_start": current_start,
                }
            )
            if values and len(set(values)) == 1 and values[0] == 1000.0:
                if kv_witness is not None:
                    raise AdapterError("T-Stitch Draft executed timestep 1000 twice")
                kv_witness = sampled_kv_cache_witness(
                    self.pipeline.kv_cache1,
                    getattr(self.pipeline, "crossattn_cache", None),
                )

        audit_handle = self.pipeline.generator.register_forward_pre_hook(
            audit_pre_hook, prepend=False, with_kwargs=True
        )
        real_decode = self.pipeline.vae.decode_to_pixel
        original_schedule = self.pipeline.denoising_step_list

        def latent_only_decode(latent: Any, use_cache: bool = False) -> Any:
            del use_cache
            return torch.zeros(
                (int(latent.shape[0]), 1, 3, 1, 1),
                device=latent.device,
                dtype=torch.float32,
            )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self.pipeline.vae.decode_to_pixel = latent_only_decode
        try:
            self.pipeline.denoising_step_list = original_schedule.new_tensor([1000])
            start_event.record()
            with torch.inference_mode():
                _ignored_video, output = self.pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    initial_latent=history,
                    return_latents=True,
                    profile=False,
                    low_memory=False,
                )
                draft_clean = output[:, -3:].contiguous()
                noisy_750 = self.pipeline.scheduler.add_noise(
                    draft_clean.flatten(0, 1),
                    noise_750.flatten(0, 1),
                    750
                    * torch.ones(
                        [3], device=self.device, dtype=torch.long
                    ),
                ).unflatten(0, draft_clean.shape[:2])
            end_event.record()
            end_event.synchronize()
            events = list(self._active_capture.events)
        finally:
            self.pipeline.denoising_step_list = original_schedule
            self.pipeline.vae.decode_to_pixel = real_decode
            audit_handle.remove()
            self._active_capture = None

        positive = [item for item in forward_trace if float(item["timestep"]) > 0.0]
        if len(positive) != 1 or float(positive[0]["timestep"]) != 1000.0:
            raise AdapterError(
                "T-Stitch Draft must execute exactly one positive-timestep forward at 1000"
            )
        if len(events) != 1 or events[0].current_start != expected_history_frames * 1560:
            raise AdapterError("T-Stitch Draft step-1000 capture is not bound to its history")
        if kv_witness is None:
            raise AdapterError("T-Stitch Draft omitted its pre-denoising KV witness")
        expected_output_shape = (1, expected_history_frames + 3, 16, 60, 104)
        if tuple(output.shape) != expected_output_shape:
            raise AdapterError(
                f"T-Stitch Draft output shape {tuple(output.shape)} != {expected_output_shape}"
            )
        if history is not None and not torch.equal(
            output[:, :expected_history_frames], history
        ):
            raise AdapterError("T-Stitch Draft did not preserve the exact input history")
        draft_cpu = draft_clean.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        noisy_cpu = noisy_750.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        if not bool(torch.isfinite(draft_cpu.float()).all().item()) or not bool(
            torch.isfinite(noisy_cpu.float()).all().item()
        ):
            raise AdapterError("T-Stitch Draft produced non-finite handoff tensors")
        elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise AdapterError("T-Stitch Draft CUDA time is invalid")
        return {
            "draft_clean_latent": draft_cpu,
            "noisy_latent_750": noisy_cpu,
            "draft_timestep": 1000.0,
            "target_handoff_timestep": 750.0,
            "denoising_forward_trace": positive,
            "forward_trace": forward_trace,
            "generation_cuda_seconds": elapsed,
            "kv_witness_sha256": kv_witness[0],
            "kv_witness": kv_witness[1],
            "rank": events[0].ranks[EFFECTIVE_RANK_LAYER],
            "rank_evidence": effective_rank_evidence_from_capture(events[0]),
        }

    def decode_latents(self, latents: Any) -> Any:
        """Decode a complete clean latent prefix to lossless uint8 RGB frames."""

        torch = self.torch
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            raise AdapterError("latents must be a five-dimensional tensor")
        if tuple(latents.shape[:1] + latents.shape[2:]) != (1, 16, 60, 104):
            raise AdapterError("latents must have shape [1,L,16,60,104]")
        latent_frames = int(latents.shape[1])
        if latent_frames < 1:
            raise AdapterError("cannot decode an empty latent history")
        with torch.inference_mode():
            video = self.pipeline.vae.decode_to_pixel(
                latents.to(device=self.device, dtype=torch.bfloat16).contiguous(),
                use_cache=False,
            )
        expected_frames = 1 + 4 * (latent_frames - 1)
        if video.ndim != 5 or tuple(video.shape[:1]) != (1,) or int(video.shape[1]) != expected_frames or int(video.shape[2]) != 3:
            raise AdapterError(
                f"VAE decoded shape {tuple(video.shape)}, expected [1,{expected_frames},3,H,W]"
            )
        frames = (
            video[0]
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
        )
        self.pipeline.vae.model.clear_cache()
        return frames

    def score_frame_indices(self, frames: Any, indices: Sequence[int]) -> list[float]:
        """Score direct, lossless uint8 frames with frozen MUSIQ-SPAQ."""

        torch = self.torch
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or int(frames.shape[1]) != 3:
            raise AdapterError("frames must have shape [T,3,H,W]")
        selected_indices = [int(value) for value in indices]
        if not selected_indices or any(value < 0 or value >= int(frames.shape[0]) for value in selected_indices):
            raise AdapterError("score_frame_indices received an out-of-range or empty index set")
        selected = frames.index_select(0, torch.tensor(selected_indices, dtype=torch.long))
        _, _, height, width = selected.shape
        if max(height, width) > 512:
            scale = 512.0 / max(height, width)
            selected = self.resize_type(
                size=(int(scale * height), int(scale * width)), antialias=False
            )(selected)
        normalized = selected.to(device=self.device, dtype=torch.float32) / 255.0
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, int(normalized.shape[0]), 16):
                batch = self.musiq(normalized[start : start + 16]).detach().reshape(-1)
                scores.extend(float(value) for value in batch.to(device="cpu").tolist())
        if len(scores) != len(selected_indices):
            raise AdapterError("MUSIQ did not return one score per frame")
        if any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in scores):
            raise AdapterError("MUSIQ returned an invalid raw score")
        return [value / 100.0 for value in scores]

    def temporal_consistency(self, frames: Any, *, start: int, stop: int) -> float:
        """VBench-style lossless adjacent-frame flicker proxy, higher is better."""

        torch = self.torch
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
            raise AdapterError("frames must be a [T,3,H,W] tensor")
        if isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or not isinstance(stop, int):
            raise AdapterError("temporal window bounds must be integers")
        if not 0 <= start < stop <= int(frames.shape[0]) or stop - start < 2:
            raise AdapterError("temporal window must contain at least two in-range frames")
        window = frames[start:stop].to(dtype=torch.float32)
        mae = (window[1:] - window[:-1]).abs().mean().item()
        result = 1.0 - float(mae) / 255.0
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise AdapterError("temporal consistency is outside [0,1]")
        return result

    def generate(self, *, prompt: str, seed: int) -> GeneratedTrajectory:
        torch = self.torch
        if not isinstance(prompt, str) or not prompt.strip():
            raise AdapterError("prompt text must be non-empty")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AdapterError("seed must be an integer")
        random.seed(seed)
        try:
            import numpy as np  # type: ignore
        except ImportError as error:
            raise AdapterError("the frozen Self-Forcing runtime is missing NumPy") from error
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self._active_capture = StepZeroCapture(
            layers=LAYER_INDICES,
            target_timestep=1000,
            sample_hidden=self._sample_hidden,
            rank_sample=self._rank_sample,
        )
        shape = [1, *self.generation["noise_shape"]]
        noise = torch.randn(shape, device=self.device, dtype=torch.bfloat16)
        try:
            with torch.inference_mode():
                video, latents = self.pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    initial_latent=None,
                    return_latents=True,
                    profile=False,
                    low_memory=False,
                )
            captures = list(self._active_capture.events)
        finally:
            self._active_capture = None
        expected_blocks = self.generation["expected_blocks"]
        if len(captures) != expected_blocks:
            raise AdapterError(
                f"native inference produced {len(captures)} step-0 forwards, "
                f"expected {expected_blocks}"
            )
        expected_stride = self.generation["num_frame_per_block"] * 1560
        for block_index, event in enumerate(captures):
            if (
                event.event_index != block_index
                or event.current_start != block_index * expected_stride
            ):
                raise AdapterError(
                    f"capture {block_index} has unexpected native "
                    f"current_start={event.current_start}"
                )
        if int(latents.shape[1]) != self.generation["noise_shape"][0]:
            raise AdapterError("native inference returned an unexpected latent-frame count")
        expected_decoded = self.generation["decoded_frames_before_trim"]
        if video.ndim != 5 or int(video.shape[0]) != 1 or int(video.shape[2]) != 3:
            raise AdapterError("native VAE output must have shape [1,T,3,H,W]")
        if int(video.shape[1]) != expected_decoded:
            raise AdapterError(
                f"native VAE decoded {int(video.shape[1])} frames, expected {expected_decoded}"
            )
        frames = (
            video[0, : self.generation["output_frames"]]
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(dtype=torch.uint8, device="cpu")
            .contiguous()
        )
        self.pipeline.vae.model.clear_cache()
        return GeneratedTrajectory(frames=frames, captures=captures)

    def frame_count(self, frames: Any) -> int:
        return int(frames.shape[0])

    def write_png(self, frames: Any, frame_index: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write_png_impl(frames[frame_index], str(path), compression_level=6)

    def score_png_raw(self, path: Path) -> float:
        torch = self.torch
        image = self.read_image(str(path)).unsqueeze(0)
        _, _, height, width = image.shape
        if max(height, width) > 512:
            scale = 512.0 / max(height, width)
            image = self.resize_type(
                size=(int(scale * height), int(scale * width)), antialias=False
            )(image)
        image = image.to(device=self.device, dtype=torch.float32) / 255.0
        with torch.inference_mode():
            score = self.musiq(image)
        raw = float(score.detach().reshape(-1)[0].item())
        if not math.isfinite(raw):
            raise AdapterError("MUSIQ returned a non-finite score")
        return raw

    def write_video(self, frames: Any, frame_count: int, path: Path, fps: int) -> None:
        from .video_encoding_policy import writer_options, verify_video
        options = writer_options()
        path.parent.mkdir(parents=True, exist_ok=True)
        video = frames[:frame_count].permute(0, 2, 3, 1).contiguous()
        self.write_video_impl(str(path), video, fps=fps, **options)
        verify_video(path)

    def close(self) -> None:
        executor = getattr(self, "_incremental_draft", None)
        if executor is not None:
            executor.cache.clear()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        os.chdir(self._old_cwd)
        source_text = str(self._source_root)
        if source_text in sys.path:
            sys.path.remove(source_text)
        python_site_text = str(self._python_site)
        if python_site_text in sys.path:
            sys.path.remove(python_site_text)


def _frame_support(block_index: int, trajectory_frames: int) -> tuple[int, int]:
    if block_index == 0:
        return 0, 9
    return 12 * block_index - 3, min(12 * block_index + 9, trajectory_frames)


def execute_worker(
    *,
    request: Mapping[str, Any],
    backend: FrameBackend,
    run_dir: Path,
    output_dir: Path,
    model_id: str,
) -> dict[str, Any]:
    """Generate one worker shard.  The injectable backend enables CPU-only tests."""

    parameters = request.get("parameters")
    prompts = request.get("prompts")
    worker_index = request.get("worker_index")
    if not isinstance(parameters, Mapping) or not isinstance(prompts, list):
        raise AdapterError("request parameters/prompts are invalid")
    if isinstance(worker_index, bool) or not isinstance(worker_index, int):
        raise AdapterError("request.worker_index is invalid")
    _verify_frozen_parameters(request)
    if len(prompts) != 12:
        raise AdapterError("each worker must receive exactly twelve prompts")
    metrics_path = output_dir / "metrics" / "block_frame_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, str]] = []
    videos: list[dict[str, Any]] = []
    prompt_ids: list[str] = []
    try:
        for prompt_position, prompt_item in enumerate(prompts):
            if not isinstance(prompt_item, Mapping):
                raise AdapterError(f"prompts[{prompt_position}] must be an object")
            prompt_id = prompt_item.get("prompt_id")
            prompt_text = prompt_item.get("prompt")
            analysis_split = prompt_item.get("analysis_split")
            dimension = prompt_item.get("assigned_dimension")
            if (
                not isinstance(prompt_id, str)
                or not prompt_id
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                    for character in prompt_id
                )
            ):
                raise AdapterError(f"prompts[{prompt_position}].prompt_id is unsafe")
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                raise AdapterError(f"prompts[{prompt_position}].prompt is invalid")
            if analysis_split not in {"development", "test"}:
                raise AdapterError(f"prompts[{prompt_position}].analysis_split is invalid")
            if not isinstance(dimension, str) or not dimension:
                raise AdapterError(f"prompts[{prompt_position}].assigned_dimension is invalid")
            if prompt_id in prompt_ids:
                raise AdapterError(f"duplicate prompt_id {prompt_id!r}")
            prompt_ids.append(prompt_id)
            trajectory = backend.generate(prompt=prompt_text, seed=20260911)
            if backend.frame_count(trajectory.frames) != 360:
                raise AdapterError(
                    f"prompt {prompt_id} did not produce exactly 360 retained frames"
                )
            if len(trajectory.captures) != 31:
                raise AdapterError(f"prompt {prompt_id} did not produce exactly 31 captures")
            capture_by_block = {event.event_index: event for event in trajectory.captures}
            if set(capture_by_block) != set(range(31)):
                raise AdapterError(f"prompt {prompt_id} capture grid is incomplete")
            for block_index in range(31):
                start, end = _frame_support(block_index, 360)
                eligible = 1 <= block_index <= 29
                anchor_index = 12 * block_index + 2 if eligible else None
                anchor_relative = ""
                anchor_digest = ""
                musiq_raw = ""
                musiq = ""
                if anchor_index is not None:
                    anchor_path = output_dir / "frames" / prompt_id / f"block-{block_index:02d}.png"
                    backend.write_png(trajectory.frames, anchor_index, anchor_path)
                    if anchor_path.is_symlink() or not anchor_path.is_file():
                        raise AdapterError("backend did not write a regular anchor PNG")
                    _inside(output_dir, anchor_path, "anchor PNG")
                    anchor_digest = _sha256_file(anchor_path)
                    anchor_relative = anchor_path.relative_to(run_dir).as_posix()
                    raw_score = float(backend.score_png_raw(anchor_path))
                    if not math.isfinite(raw_score):
                        raise AdapterError("MUSIQ returned a non-finite raw score")
                    normalized_score = raw_score / 100.0
                    if not 0.0 <= raw_score <= 100.0 or not 0.0 <= normalized_score <= 1.0:
                        raise AdapterError("MUSIQ-SPAQ raw score is outside [0,100]")
                    musiq_raw = format(raw_score, ".17g")
                    musiq = format(normalized_score, ".17g")
                event = capture_by_block[block_index]
                if set(event.ranks) != set(LAYER_INDICES):
                    raise AdapterError(
                        f"prompt {prompt_id} block {block_index} layer grid is invalid"
                    )
                row = {
                    "prompt_id": prompt_id,
                    "analysis_split": str(analysis_split),
                    "assigned_dimension": dimension,
                    "seed": "20260911",
                    "model_id": model_id,
                    "trajectory_seconds": "15",
                    "fps": "24",
                    "trajectory_frames": "360",
                    "block_index": str(block_index),
                    "frame_start": str(start),
                    "frame_end_exclusive": str(end),
                    "analysis_eligible": "true" if eligible else "false",
                    "anchor_frame_index": "" if anchor_index is None else str(anchor_index),
                    "anchor_png_path": anchor_relative,
                    "anchor_png_sha256": anchor_digest,
                    "musiq_spaq_raw": musiq_raw,
                    "musiq_spaq": musiq,
                }
                for layer_index, field in LAYER_FIELDS.items():
                    rank = float(event.ranks[layer_index])
                    if not math.isfinite(rank) or not 1.0 <= rank <= 256.0:
                        raise AdapterError(f"invalid {field} for {prompt_id} block {block_index}")
                    row[field] = format(rank, ".17g")
                rows.append(row)
            video_path = output_dir / "videos" / f"{prompt_id}.mp4"
            backend.write_video(trajectory.frames, 360, video_path, 24)
            if video_path.is_symlink() or not video_path.is_file():
                raise AdapterError("backend did not write a regular video")
            _inside(output_dir, video_path, "video")
            videos.append(
                {
                    "prompt_id": prompt_id,
                    "path": video_path.relative_to(run_dir).as_posix(),
                    "sha256": _sha256_file(video_path),
                    "frame_count": 360,
                    "fps": 24,
                }
            )
    finally:
        backend.close()
    temporary = metrics_path.with_name(f".{metrics_path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, metrics_path)
    if len(rows) != 12 * 31:
        raise AdapterError("worker metric row count is incomplete")
    result = {
        "schema": RESULT_SCHEMA,
        "worker_index": worker_index,
        "prompt_ids": prompt_ids,
        "metrics_csv": metrics_path.relative_to(output_dir).as_posix(),
        "videos": videos,
    }
    _write_json(output_dir / "result.json", result)
    return result


def run_request(request_path: Path, *, backend: FrameBackend | None = None) -> dict[str, Any]:
    request = _load_json(request_path, "adapter request")
    if request.get("schema") != REQUEST_SCHEMA:
        raise AdapterError(f"request schema must be {REQUEST_SCHEMA!r}")
    run_text = request.get("run_dir")
    output_text = request.get("output_dir")
    worker_index = request.get("worker_index")
    if not isinstance(run_text, str) or not isinstance(output_text, str):
        raise AdapterError("request run_dir/output_dir are invalid")
    if isinstance(worker_index, bool) or not isinstance(worker_index, int):
        raise AdapterError("request.worker_index is invalid")
    run_dir = Path(run_text).expanduser().resolve(strict=True)
    output_dir = _inside(run_dir, Path(output_text), "worker output directory")
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise AdapterError("worker output directory must be a real directory")
    expected_output = run_dir / "workers" / f"worker-{worker_index:02d}"
    if output_dir != expected_output:
        raise AdapterError("worker output directory disagrees with worker_index")
    environment_index = os.environ.get("CFEXP_WORKER_INDEX")
    environment_count = os.environ.get("CFEXP_WORKER_COUNT")
    environment_dir = os.environ.get("CFEXP_WORKER_DIR")
    if environment_index is not None and environment_index != str(worker_index):
        raise AdapterError("CFEXP_WORKER_INDEX disagrees with the request")
    if environment_dir is not None and Path(environment_dir).resolve() != output_dir:
        raise AdapterError("CFEXP_WORKER_DIR disagrees with the request")
    if environment_count is not None and environment_count != "4":
        raise AdapterError("CFEXP_WORKER_COUNT disagrees with the frozen campaign")
    _verify_frozen_parameters(request)
    _verify_result_contract(request, output_dir)
    _verify_protocol_shard(request)
    artifacts = _verified_artifacts(request)
    config = _load_runtime_config(request, artifacts)
    created_backend = backend
    if created_backend is None:
        source_root, python_site = _materialize_source(
            run_dir=run_dir,
            worker_index=worker_index,
            artifacts=artifacts,
            config=config,
        )
        device = request.get("device")
        if device != "cuda:0":
            raise AdapterError("each isolated worker must use request.device='cuda:0'")
        created_backend = OfficialSelfForcingBackend(
            source_root=source_root,
            artifacts=artifacts,
            config=config,
            device_name=device,
            python_site=python_site,
        )
    return execute_worker(
        request=request,
        backend=created_backend,
        run_dir=run_dir,
        output_dir=output_dir,
        model_id=config["generation"]["model_id"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.self_forcing_frame_adapter")
    parser.add_argument("--cfexp-request", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_request(Path(args.cfexp_request))
        return 0
    except (AdapterError, OSError, ValueError, RuntimeError) as error:
        print(f"self-forcing-frame-adapter: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
