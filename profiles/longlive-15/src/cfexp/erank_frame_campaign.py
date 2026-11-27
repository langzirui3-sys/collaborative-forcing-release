"""Fail-closed orchestration for the frame-level eRank pilot.

This module deliberately does not implement Self-Forcing or MUSIQ.  A real,
repository-tracked adapter must implement the subprocess contract documented by
``ADAPTER_REQUEST_SCHEMA`` and be named by a frozen artifact specification.
The orchestrator owns immutable input verification, deterministic GPU sharding,
output confinement, worker supervision, and final metric-grid assembly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .erank_quality import ProtocolError
from .frame_erank_quality import (
    analyze_rows,
    build_correlation_plot_data,
    build_erank_trend_plot_data,
    load_frame_rows,
    load_protocol,
    render_correlation_plots,
    render_erank_trend_plot,
)
from .metadata import collect_git_metadata, find_git_root, utc_now


ARTIFACT_SPEC_SCHEMA = "erank_frame_artifacts_v1"
ADAPTER_REQUEST_SCHEMA = "erank_frame_adapter_request_v1"
ADAPTER_RESULT_SCHEMA = "erank_frame_adapter_result_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
METRICS_COLUMNS = [
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
]


class CampaignError(RuntimeError):
    """Raised before incomplete output can be mistaken for a valid campaign."""


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CampaignError(f"{label} contains invalid JSON constant {item!r}")
            ),
        )
    except CampaignError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path, label: str) -> Path:
    root = root.expanduser().resolve(strict=True)
    candidate = candidate.expanduser().resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise CampaignError(f"{label} resolves outside {root}: {candidate}") from error
    return candidate


def _repository_file(repo_root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignError(f"{label} must be a repository-relative path")
    path = _inside(repo_root, repo_root / relative, label)
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"{label} must be a regular, non-symlink file: {path}")
    return path


def _output_path(run_dir: Path, *parts: str) -> Path:
    if any(not part or Path(part).is_absolute() or ".." in Path(part).parts for part in parts):
        raise CampaignError("output paths must be non-empty relative paths without '..'")
    candidate = run_dir.joinpath(*parts)
    resolved_parent = candidate.parent.resolve(strict=False)
    try:
        resolved_parent.relative_to(run_dir)
    except ValueError as error:
        raise CampaignError(f"output path escapes CFEXP_RUN_DIR: {candidate}") from error
    return candidate


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_prompt_shards(
    prompts: Sequence[Mapping[str, Any]], worker_count: int
) -> list[list[dict[str, Any]]]:
    """Split prompts into stable contiguous shards with size difference at most one."""

    if worker_count < 1:
        raise CampaignError("worker_count must be positive")
    if len(prompts) < worker_count:
        raise CampaignError("worker_count cannot exceed the prompt count")
    quotient, remainder = divmod(len(prompts), worker_count)
    shards: list[list[dict[str, Any]]] = []
    start = 0
    for worker_index in range(worker_count):
        size = quotient + (1 if worker_index < remainder else 0)
        stop = start + size
        shards.append([dict(item) for item in prompts[start:stop]])
        start = stop
    return shards


def _finite_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignError(f"{label} must be an integer")
    return value


def _protocol_seed(design: Mapping[str, Any]) -> int:
    if "seed" in design:
        return _finite_integer(design["seed"], "protocol design.seed")
    seeds = design.get("seeds")
    if isinstance(seeds, list) and len(seeds) == 1:
        return _finite_integer(seeds[0], "protocol design.seeds[0]")
    raise CampaignError("protocol must freeze exactly one seed")


def _protocol_layer_indices(protocol: Mapping[str, Any]) -> list[int]:
    candidates: list[Any] = []
    probe = protocol.get("erank_probe")
    if isinstance(probe, Mapping):
        layers = probe.get("layers")
        if isinstance(layers, list) and layers:
            indices: list[int] = []
            for index, layer in enumerate(layers):
                if not isinstance(layer, Mapping):
                    raise CampaignError(f"protocol erank_probe.layers[{index}] is invalid")
                indices.append(
                    _finite_integer(
                        layer.get("layer_index_zero_based"),
                        f"protocol erank_probe.layers[{index}].layer_index_zero_based",
                    )
                )
            return indices
    for section_name in ("layer_ablation", "erank", "hidden_state_erank"):
        section = protocol.get(section_name)
        if isinstance(section, Mapping):
            candidates.extend(
                [
                    section.get("layer_indices_zero_based"),
                    section.get("layers_zero_based"),
                ]
            )
    candidates.append(protocol.get("layer_indices_zero_based"))
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            return [
                _finite_integer(item, "protocol zero-based layer index")
                for item in candidate
            ]
    raise CampaignError("protocol does not declare zero-based layer indices")


def load_campaign_protocol(
    path: Path,
    *,
    expected_prompts: int,
    seed: int,
    duration_seconds: int,
    fps: int,
    layer_indices: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _load_json_object(path, "campaign protocol")
    design = protocol.get("design")
    if not isinstance(design, Mapping):
        raise CampaignError("protocol design must be an object")
    prompts = protocol.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != expected_prompts:
        raise CampaignError(
            f"protocol must contain exactly {expected_prompts} prompts"
        )
    normalized_prompts: list[dict[str, Any]] = []
    prompt_ids: list[str] = []
    for index, item in enumerate(prompts):
        if not isinstance(item, Mapping):
            raise CampaignError(f"protocol prompts[{index}] must be an object")
        prompt_id = item.get("prompt_id")
        prompt_text = item.get("prompt")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise CampaignError(f"protocol prompts[{index}].prompt_id is invalid")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise CampaignError(f"protocol prompts[{index}].prompt is invalid")
        prompt_ids.append(prompt_id)
        normalized_prompts.append(dict(item))
    if len(set(prompt_ids)) != len(prompt_ids):
        raise CampaignError("protocol prompt_id values must be unique")
    if design.get("prompt_count") != expected_prompts:
        raise CampaignError("protocol design.prompt_count disagrees with the manifest")
    if _protocol_seed(design) != seed:
        raise CampaignError("protocol seed disagrees with the manifest")
    if design.get("trajectory_seconds") != duration_seconds:
        raise CampaignError("protocol trajectory_seconds disagrees with the manifest")
    if design.get("fps") != fps:
        raise CampaignError("protocol fps disagrees with the manifest")
    if design.get("trajectory_frames") != duration_seconds * fps:
        raise CampaignError("protocol trajectory_frames disagrees with duration and fps")
    if _protocol_layer_indices(protocol) != list(layer_indices):
        raise CampaignError("protocol layer indices disagree with the manifest")
    return protocol, normalized_prompts


def verify_artifact_spec(
    path: Path, *, model_root: Path, repo_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Verify every frozen model/scorer file before importing the adapter."""

    spec = _load_json_object(path, "artifact specification")
    if spec.get("schema") != ARTIFACT_SPEC_SCHEMA:
        raise CampaignError(
            f"artifact specification schema must be {ARTIFACT_SPEC_SCHEMA!r}"
        )
    if spec.get("status") != "frozen":
        raise CampaignError("artifact specification is not frozen; real pins are required")
    adapter_module = spec.get("adapter_module")
    if not isinstance(adapter_module, str) or not _MODULE_RE.fullmatch(adapter_module):
        raise CampaignError("artifact specification adapter_module is invalid")
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CampaignError("artifact specification must pin at least one file")

    roots = {"model": model_root, "repository": repo_root}
    verified: list[dict[str, Any]] = []
    names: set[str] = set()
    requested_paths: set[tuple[str, str]] = set()
    for index, item in enumerate(artifacts):
        label = f"artifacts[{index}]"
        if not isinstance(item, Mapping):
            raise CampaignError(f"{label} must be an object")
        unknown = set(item) - {"name", "root", "path", "sha256", "role"}
        if unknown:
            raise CampaignError(f"{label} has unknown fields: {sorted(unknown)}")
        name = item.get("name")
        root_name = item.get("root")
        relative_text = item.get("path")
        expected_digest = item.get("sha256")
        role = item.get("role", "runtime_input")
        if not isinstance(name, str) or not name or name in names:
            raise CampaignError(f"{label}.name must be non-empty and unique")
        if root_name not in roots:
            raise CampaignError(f"{label}.root must be 'model' or 'repository'")
        if not isinstance(relative_text, str) or not relative_text:
            raise CampaignError(f"{label}.path must be a non-empty relative path")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise CampaignError(f"{label}.path must stay below its declared root")
        if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(
            expected_digest
        ):
            raise CampaignError(f"{label}.sha256 must be lowercase 64-hex")
        if not isinstance(role, str) or not role:
            raise CampaignError(f"{label}.role must be a non-empty string")
        path_key = (str(root_name), relative.as_posix())
        if path_key in requested_paths:
            raise CampaignError(f"{label}.path duplicates another artifact")
        root = roots[str(root_name)].expanduser().resolve(strict=True)
        resolved = _inside(root, root / relative, f"{label}.path")
        try:
            metadata = resolved.stat()
        except OSError as error:
            raise CampaignError(f"cannot inspect artifact {resolved}: {error}") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise CampaignError(f"artifact is not a regular file: {resolved}")
        actual_digest = _sha256_file(resolved)
        if actual_digest != expected_digest:
            raise CampaignError(
                f"artifact hash mismatch for {name!r}: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        names.add(name)
        requested_paths.add(path_key)
        verified.append(
            {
                "name": name,
                "role": role,
                "root": root_name,
                "requested_path": relative.as_posix(),
                "resolved_path": str(resolved),
                "size": metadata.st_size,
                "sha256": actual_digest,
            }
        )
    return spec, verified, adapter_module


def _adapter_origin(adapter_module: str, repo_root: Path) -> Path:
    try:
        module_spec = importlib.util.find_spec(adapter_module)
    except (ImportError, AttributeError, ValueError) as error:
        raise CampaignError(f"cannot resolve adapter module {adapter_module!r}: {error}") from error
    if module_spec is None or not module_spec.origin:
        raise CampaignError(f"adapter module is not installed: {adapter_module}")
    origin = Path(module_spec.origin)
    try:
        origin = _inside(repo_root, origin, "adapter module")
    except (OSError, CampaignError) as error:
        raise CampaignError(
            "the real adapter must be a repository-tracked Python module"
        ) from error
    if origin.is_symlink() or not origin.is_file():
        raise CampaignError("adapter module must be a regular, non-symlink file")
    return origin


def _visible_devices(worker_count: int) -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) != worker_count or len(devices) != len(set(devices)):
        raise CampaignError(
            f"expected {worker_count} unique CUDA_VISIBLE_DEVICES entries, got {devices}"
        )
    return devices


def _parse_layer_indices(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layer indices must be comma-separated integers"
        ) from error
    if not values or any(item < 0 for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("layer indices must be unique non-negative integers")
    return values


def _terminate(processes: Sequence[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def _run_workers(
    *,
    adapter_module: str,
    repo_root: Path,
    run_dir: Path,
    requests: Sequence[Path],
    devices: Sequence[str],
) -> None:
    processes: list[subprocess.Popen[str]] = []
    handles: list[Any] = []
    try:
        for worker_index, (request_path, device) in enumerate(zip(requests, devices)):
            worker_dir = _output_path(run_dir, "workers", f"worker-{worker_index:02d}")
            log_path = _output_path(
                run_dir, "workers", f"worker-{worker_index:02d}", "logs", "adapter.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8", buffering=1)
            handles.append(handle)
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": device,
                    "CFEXP_WORKER_INDEX": str(worker_index),
                    "CFEXP_WORKER_COUNT": str(len(requests)),
                    "CFEXP_WORKER_DIR": str(worker_dir),
                }
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    adapter_module,
                    "--cfexp-request",
                    str(request_path),
                ],
                cwd=repo_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)

        pending = set(range(len(processes)))
        while pending:
            for worker_index in list(pending):
                return_code = processes[worker_index].poll()
                if return_code is None:
                    continue
                pending.remove(worker_index)
                if return_code != 0:
                    _terminate(processes)
                    raise CampaignError(
                        f"adapter worker {worker_index} failed with exit code {return_code}"
                    )
            if pending:
                time.sleep(0.2)
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


def merge_worker_metrics(
    *,
    run_dir: Path,
    shards: Sequence[Sequence[Mapping[str, Any]]],
    seed: int,
    expected_blocks: int,
    duration_seconds: int = 15,
    fps: int = 24,
    eligible_block_start: int = 1,
    eligible_block_end: int = 29,
) -> tuple[Path, int]:
    """Validate adapter result contracts and assemble a deterministic wide CSV."""

    expected_prompt_order = [
        str(prompt["prompt_id"]) for shard in shards for prompt in shard
    ]
    rows_by_key: dict[tuple[str, int], dict[str, str]] = {}
    header: list[str] | None = None
    for worker_index, shard in enumerate(shards):
        worker_dir = _output_path(run_dir, "workers", f"worker-{worker_index:02d}")
        result_path = worker_dir / "result.json"
        result = _load_json_object(result_path, f"worker {worker_index} result")
        expected_ids = [str(prompt["prompt_id"]) for prompt in shard]
        if result.get("schema") != ADAPTER_RESULT_SCHEMA:
            raise CampaignError(f"worker {worker_index} result schema is invalid")
        if result.get("worker_index") != worker_index:
            raise CampaignError(f"worker {worker_index} result index is invalid")
        if result.get("prompt_ids") != expected_ids:
            raise CampaignError(f"worker {worker_index} processed the wrong prompt shard")
        videos = result.get("videos")
        if not isinstance(videos, list) or len(videos) != len(expected_ids):
            raise CampaignError(f"worker {worker_index} returned an invalid video inventory")
        video_prompt_ids: list[str] = []
        for video_index, video in enumerate(videos):
            if not isinstance(video, Mapping):
                raise CampaignError(
                    f"worker {worker_index} videos[{video_index}] must be an object"
                )
            video_prompt_id = video.get("prompt_id")
            video_path_text = video.get("path")
            video_digest = video.get("sha256")
            if not isinstance(video_prompt_id, str):
                raise CampaignError(f"worker {worker_index} has an invalid video prompt_id")
            if not isinstance(video_path_text, str) or not video_path_text:
                raise CampaignError(f"worker {worker_index} has an invalid video path")
            relative_video = Path(video_path_text)
            if relative_video.is_absolute() or ".." in relative_video.parts:
                raise CampaignError(f"worker {worker_index} has an unsafe video path")
            requested_video = run_dir / relative_video
            if requested_video.is_symlink():
                raise CampaignError(f"worker {worker_index} video may not be a symlink")
            video_path = _inside(worker_dir, requested_video, "worker video")
            if not video_path.is_file() or video_path.suffix.casefold() not in {
                ".mp4",
                ".mkv",
                ".webm",
                ".mov",
            }:
                raise CampaignError(f"worker {worker_index} video is not a supported file")
            if not isinstance(video_digest, str) or not _SHA256_RE.fullmatch(video_digest):
                raise CampaignError(f"worker {worker_index} has an invalid video hash")
            if _sha256_file(video_path) != video_digest:
                raise CampaignError(f"worker {worker_index} video hash mismatch")
            if video.get("frame_count") != duration_seconds * fps or video.get("fps") != fps:
                raise CampaignError(f"worker {worker_index} video metadata is invalid")
            video_prompt_ids.append(video_prompt_id)
        if video_prompt_ids != expected_ids:
            raise CampaignError(f"worker {worker_index} video inventory is out of order")
        metrics_name = result.get("metrics_csv")
        if not isinstance(metrics_name, str) or not metrics_name:
            raise CampaignError(f"worker {worker_index} did not return metrics_csv")
        relative = Path(metrics_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise CampaignError(f"worker {worker_index} metrics_csv is not relative")
        metrics_path = _inside(worker_dir, worker_dir / relative, "worker metrics_csv")
        if metrics_path.is_symlink() or not metrics_path.is_file():
            raise CampaignError(f"worker {worker_index} metrics_csv is not a regular file")
        try:
            handle = metrics_path.open("r", encoding="utf-8", newline="")
        except (OSError, UnicodeError) as error:
            raise CampaignError(f"cannot read worker metrics {metrics_path}: {error}") from error
        with handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if fields != METRICS_COLUMNS:
                raise CampaignError(
                    f"worker {worker_index} metrics header does not match the frozen contract"
                )
            if header is None:
                header = fields
            elif fields != header:
                raise CampaignError("worker metrics CSV headers are inconsistent")
            for row_number, row in enumerate(reader, start=2):
                prompt_id = row.get("prompt_id", "")
                if prompt_id not in expected_ids:
                    raise CampaignError(
                        f"worker {worker_index} row {row_number} has unexpected prompt_id"
                    )
                try:
                    row_seed = int(row.get("seed", ""))
                    block_index = int(row.get("block_index", ""))
                except ValueError as error:
                    raise CampaignError(
                        f"worker {worker_index} row {row_number} has invalid integer fields"
                    ) from error
                if row_seed != seed or not 0 <= block_index < expected_blocks:
                    raise CampaignError(
                        f"worker {worker_index} row {row_number} is outside the frozen grid"
                    )
                prompt_contract = next(
                    prompt for prompt in shard if str(prompt["prompt_id"]) == prompt_id
                )
                expected_start = 0 if block_index == 0 else 12 * block_index - 3
                expected_end = min(
                    duration_seconds * fps,
                    9 if block_index == 0 else 12 * block_index + 9,
                )
                expected_eligible = eligible_block_start <= block_index <= eligible_block_end
                expected_anchor = 12 * block_index + 2 if expected_eligible else None
                exact_values = {
                    "analysis_split": str(prompt_contract.get("analysis_split", "")),
                    "assigned_dimension": str(prompt_contract.get("assigned_dimension", "")),
                    "model_id": "self_forcing_1.3b",
                    "trajectory_seconds": str(duration_seconds),
                    "fps": str(fps),
                    "trajectory_frames": str(duration_seconds * fps),
                    "frame_start": str(expected_start),
                    "frame_end_exclusive": str(expected_end),
                    "analysis_eligible": "true" if expected_eligible else "false",
                    "anchor_frame_index": "" if expected_anchor is None else str(expected_anchor),
                }
                for field, expected_value in exact_values.items():
                    if row.get(field, "").strip().lower() != expected_value.lower():
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} has invalid {field}"
                        )
                key = (prompt_id, block_index)
                if key in rows_by_key:
                    raise CampaignError(f"duplicate metrics row for {key}")
                for field, value in row.items():
                    if field is None or value is None:
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} is malformed"
                        )
                    if field.startswith("erank_"):
                        try:
                            number = float(value)
                        except ValueError as error:
                            raise CampaignError(
                                f"worker {worker_index} row {row_number} has invalid {field}"
                            ) from error
                        if not math.isfinite(number) or not 1.0 <= number <= 256.0:
                            raise CampaignError(
                                f"worker {worker_index} row {row_number} has invalid {field}"
                            )
                raw_quality_text = row["musiq_spaq_raw"].strip()
                quality_text = row["musiq_spaq"].strip()
                if expected_eligible:
                    try:
                        raw_quality = float(raw_quality_text)
                        quality = float(quality_text)
                    except ValueError as error:
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} has invalid musiq_spaq"
                        ) from error
                    if (
                        not math.isfinite(raw_quality)
                        or not 0.0 <= raw_quality <= 100.0
                        or not math.isfinite(quality)
                        or not 0.0 <= quality <= 1.0
                        or not math.isclose(
                            quality,
                            raw_quality / 100.0,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} has invalid musiq_spaq"
                        )
                    digest = row["anchor_png_sha256"]
                    if not _SHA256_RE.fullmatch(digest):
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} has invalid anchor hash"
                        )
                    anchor_text = row["anchor_png_path"].strip()
                    if not anchor_text:
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} is missing anchor_png_path"
                        )
                    anchor_relative = Path(anchor_text)
                    if anchor_relative.is_absolute() or ".." in anchor_relative.parts:
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} has unsafe anchor_png_path"
                        )
                    requested_anchor = run_dir / anchor_relative
                    if requested_anchor.is_symlink():
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} anchor may not be a symlink"
                        )
                    anchor_path = _inside(
                        worker_dir, requested_anchor, "anchor_png_path"
                    )
                    if not anchor_path.is_file():
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} anchor is not a file"
                        )
                    if anchor_path.suffix.casefold() != ".png":
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} anchor is not PNG"
                        )
                    if _sha256_file(anchor_path) != digest:
                        raise CampaignError(
                            f"worker {worker_index} row {row_number} anchor hash mismatch"
                        )
                elif any(
                    row[field].strip()
                    for field in (
                        "anchor_png_path",
                        "anchor_png_sha256",
                        "musiq_spaq_raw",
                        "musiq_spaq",
                    )
                ):
                    raise CampaignError(
                        f"worker {worker_index} row {row_number} has ineligible frame metrics"
                    )
                rows_by_key[key] = dict(row)

    expected_keys = {
        (prompt_id, block_index)
        for prompt_id in expected_prompt_order
        for block_index in range(expected_blocks)
    }
    if set(rows_by_key) != expected_keys:
        missing = sorted(expected_keys - set(rows_by_key))
        unexpected = sorted(set(rows_by_key) - expected_keys)
        raise CampaignError(
            f"incomplete final metric grid: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    assert header is not None
    output_path = _output_path(run_dir, "metrics", "block_frame_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for prompt_id in expected_prompt_order:
            for block_index in range(expected_blocks):
                writer.writerow(rows_by_key[(prompt_id, block_index)])
    os.replace(temporary, output_path)
    return output_path, len(rows_by_key)


def _require_nonempty_file(path: Path, label: str) -> None:
    """Fail closed when a renderer returned without producing its artifact."""

    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"{label} was not written as a regular file: {path}")
    if path.stat().st_size <= 0:
        raise CampaignError(f"{label} is empty: {path}")


def run_post_analysis(
    *,
    run_dir: Path,
    protocol_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Validate the complete campaign and atomically publish canonical analysis.

    Formal runs always use the protocol-frozen 10,000 prompt-cluster bootstrap
    replicates for both the correlation analysis and trend bands.  Every PNG is
    read and checked again by :func:`load_frame_rows` before the held-out test is
    opened.  Plot files are staged outside ``plots/`` and the directory is
    renamed into place only after every applicable figure has rendered in both
    PNG and PDF formats.  A pre-registered no-selection outcome is successful
    and intentionally has no held-out figure.
    """

    plots_dir = _output_path(run_dir, "plots")
    if plots_dir.exists() or plots_dir.is_symlink():
        raise CampaignError(f"post-analysis output already exists: {plots_dir}")
    staging_dir = _output_path(run_dir, f".post-analysis-{os.getpid()}")
    if staging_dir.exists() or staging_dir.is_symlink():
        raise CampaignError(f"post-analysis staging path already exists: {staging_dir}")

    try:
        protocol = load_protocol(protocol_path)
        rows = load_frame_rows(
            metrics_path,
            protocol,
            frames_root=run_dir,
            require_complete=True,
        )
        # Passing no override is intentional: the frozen protocol requires
        # exactly 10,000 prompt-cluster bootstrap replicates in formal runs.
        analysis = analyze_rows(rows, protocol)
        trend_data = build_erank_trend_plot_data(rows, protocol)
        correlation_data = build_correlation_plot_data(rows, protocol, analysis)

        staging_dir.mkdir(parents=False, exist_ok=False)
        has_held_out = isinstance(analysis.get("held_out_test"), Mapping)
        staged_files: dict[str, Path] = {
            "analysis_json": staging_dir / "frame_erank_analysis.json",
            "trend_plot_data_json": staging_dir / "erank_across_blocks.plot-data.json",
            "correlation_plot_data_json": (
                staging_dir / "erank_frame_correlation.plot-data.json"
            ),
            "trend_png": staging_dir / "erank_across_blocks.png",
            "trend_pdf": staging_dir / "erank_across_blocks.pdf",
            "development_correlation_png": staging_dir / "dev_layer_correlations.png",
            "development_correlation_pdf": staging_dir / "dev_layer_correlations.pdf",
        }
        if has_held_out:
            staged_files.update(
                {
                    "held_out_correlation_png": staging_dir / "held_out_correlation.png",
                    "held_out_correlation_pdf": staging_dir / "held_out_correlation.pdf",
                }
            )
        _write_json(staged_files["analysis_json"], analysis)
        _write_json(staged_files["trend_plot_data_json"], trend_data)
        _write_json(staged_files["correlation_plot_data_json"], correlation_data)

        for suffix in ("png", "pdf"):
            render_erank_trend_plot(trend_data, staged_files[f"trend_{suffix}"])
            render_correlation_plots(
                correlation_data,
                development_output=staged_files[f"development_correlation_{suffix}"],
                held_out_output=(
                    staged_files[f"held_out_correlation_{suffix}"]
                    if has_held_out
                    else None
                ),
            )

        for label, path in staged_files.items():
            _require_nonempty_file(path, label)

        # A single same-filesystem rename prevents a successful campaign from
        # exposing a partially rendered plot set.
        os.replace(staging_dir, plots_dir)
        relative_outputs = {
            label: (plots_dir / path.name).relative_to(run_dir).as_posix()
            for label, path in staged_files.items()
        }
        return {
            "bootstrap_replicates": int(analysis["bootstrap_replicates"]),
            "trend_bootstrap_replicates": int(trend_data["bootstrap"]["replicates"]),
            "selected_layer_number_one_based": analysis[
                "selected_layer_number_one_based"
            ],
            "selection_outcome": analysis.get("selection_outcome"),
            "strict_pilot_target_met": (
                analysis["held_out_test"]["strict_pilot_target_met"]
                if isinstance(analysis.get("held_out_test"), Mapping)
                else None
            ),
            "outputs": relative_outputs,
        }
    except (ProtocolError, KeyError, TypeError, ValueError) as error:
        raise CampaignError(f"post-analysis failed: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.erank_frame_campaign")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--artifact-spec", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration-seconds", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--layer-indices", type=_parse_layer_indices, required=True)
    parser.add_argument("--expected-blocks", type=int, required=True)
    parser.add_argument("--eligible-block-start", type=int, required=True)
    parser.add_argument("--eligible-block-end", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_value = os.environ.get("CFEXP_RUN_DIR")
        model_value = os.environ.get("CFEXP_MODEL_ROOT")
        if not run_value:
            raise CampaignError("CFEXP_RUN_DIR is required")
        if not model_value:
            raise CampaignError("CFEXP_MODEL_ROOT is required")
        run_dir = Path(run_value).expanduser().resolve(strict=True)
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise CampaignError("CFEXP_RUN_DIR must be a regular directory")
        model_root = Path(model_value).expanduser().resolve(strict=True)
        if not model_root.is_dir():
            raise CampaignError("CFEXP_MODEL_ROOT must be a directory")
        repo_root = find_git_root(Path.cwd())
        protocol_path = _repository_file(repo_root, args.protocol, "protocol")
        artifact_spec_path = _repository_file(
            repo_root, args.artifact_spec, "artifact specification"
        )
        if args.workers != 4:
            raise CampaignError("this frozen pilot requires exactly four workers")
        if args.expected_prompts != 48:
            raise CampaignError("this frozen pilot requires exactly 48 prompts")
        if args.expected_blocks < 1:
            raise CampaignError("expected_blocks must be positive")
        if not (
            0 <= args.eligible_block_start <= args.eligible_block_end < args.expected_blocks
        ):
            raise CampaignError("eligible block bounds are invalid")

        protocol, prompts = load_campaign_protocol(
            protocol_path,
            expected_prompts=args.expected_prompts,
            seed=args.seed,
            duration_seconds=args.duration_seconds,
            fps=args.fps,
            layer_indices=args.layer_indices,
        )
        artifact_spec, verified_artifacts, adapter_module = verify_artifact_spec(
            artifact_spec_path,
            model_root=model_root,
            repo_root=repo_root,
        )
        adapter_origin = _adapter_origin(adapter_module, repo_root)
        devices = _visible_devices(args.workers)
        shards = stable_prompt_shards(prompts, args.workers)
        if [len(shard) for shard in shards] != [12] * 4:
            raise CampaignError("the 48 prompts must map to exactly twelve prompts per worker")
        design = protocol["design"]
        assert isinstance(design, Mapping)
        if design.get("block_indices") != list(range(args.expected_blocks)):
            raise CampaignError("protocol block_indices disagrees with the manifest")
        if design.get("analysis_block_indices") != list(
            range(args.eligible_block_start, args.eligible_block_end + 1)
        ):
            raise CampaignError("protocol analysis_block_indices disagrees with the manifest")

        git = collect_git_metadata(repo_root)
        preflight = {
            "schema": "erank_frame_campaign_preflight_v1",
            "created_at": utc_now(),
            "git": git,
            "parameters": {
                "worker_count": args.workers,
                "prompt_count": args.expected_prompts,
                "seed": args.seed,
                "duration_seconds": args.duration_seconds,
                "fps": args.fps,
                "layer_indices_zero_based": args.layer_indices,
                "expected_blocks": args.expected_blocks,
                "eligible_block_start": args.eligible_block_start,
                "eligible_block_end": args.eligible_block_end,
            },
            "protocol": {
                "path": protocol_path.relative_to(repo_root).as_posix(),
                "sha256": _sha256_file(protocol_path),
                "protocol_id": protocol.get("protocol_id"),
            },
            "artifact_spec": {
                "path": artifact_spec_path.relative_to(repo_root).as_posix(),
                "sha256": _sha256_file(artifact_spec_path),
                "schema": artifact_spec.get("schema"),
            },
            "adapter": {
                "module": adapter_module,
                "origin": adapter_origin.relative_to(repo_root).as_posix(),
                "sha256": _sha256_file(adapter_origin),
            },
            "verified_artifacts": verified_artifacts,
            "shards": [
                {
                    "worker_index": index,
                    "prompt_ids": [str(prompt["prompt_id"]) for prompt in shard],
                }
                for index, shard in enumerate(shards)
            ],
            # Record only stable worker ordinals. Physical GPU identifiers are
            # already captured by cfexp metadata and need not enter requests.
            "worker_device_ordinals": list(range(args.workers)),
        }
        preflight_path = _output_path(run_dir, "provenance", "preflight.json")
        _write_json(preflight_path, preflight)

        requests: list[Path] = []
        for worker_index, shard in enumerate(shards):
            worker_dir = _output_path(run_dir, "workers", f"worker-{worker_index:02d}")
            worker_dir.mkdir(parents=True, exist_ok=False)
            request_path = _output_path(
                run_dir, "provenance", "requests", f"worker-{worker_index:02d}.json"
            )
            _write_json(
                request_path,
                {
                    "schema": ADAPTER_REQUEST_SCHEMA,
                    "worker_index": worker_index,
                    "worker_count": args.workers,
                    "device": "cuda:0",
                    "output_dir": str(worker_dir),
                    "run_dir": str(run_dir),
                    "protocol_path": str(protocol_path),
                    "protocol_sha256": _sha256_file(protocol_path),
                    "artifact_spec_path": str(artifact_spec_path),
                    "artifact_spec_sha256": _sha256_file(artifact_spec_path),
                    "verified_artifacts": verified_artifacts,
                    "parameters": preflight["parameters"],
                    "prompts": shard,
                    "result_contract": {
                        "schema": ADAPTER_RESULT_SCHEMA,
                        "path": str(worker_dir / "result.json"),
                        "metrics_unit": "one row per prompt and generated block",
                        "expected_rows": len(shard) * args.expected_blocks,
                        "video_inventory": {
                            "one_per_prompt": True,
                            "path_base": "CFEXP_RUN_DIR",
                            "required_fields": [
                                "prompt_id",
                                "path",
                                "sha256",
                                "frame_count",
                                "fps",
                            ],
                        },
                    },
                },
            )
            requests.append(request_path)

        _run_workers(
            adapter_module=adapter_module,
            repo_root=repo_root,
            run_dir=run_dir,
            requests=requests,
            devices=devices,
        )
        metrics_path, row_count = merge_worker_metrics(
            run_dir=run_dir,
            shards=shards,
            seed=args.seed,
            expected_blocks=args.expected_blocks,
            duration_seconds=args.duration_seconds,
            fps=args.fps,
            eligible_block_start=args.eligible_block_start,
            eligible_block_end=args.eligible_block_end,
        )
        post_analysis = run_post_analysis(
            run_dir=run_dir,
            protocol_path=protocol_path,
            metrics_path=metrics_path,
        )
        summary = {
            "schema": "erank_frame_campaign_summary_v1",
            "status": "SUCCEEDED",
            "finished_at": utc_now(),
            "prompt_count": args.expected_prompts,
            "worker_count": args.workers,
            "rows": row_count,
            "metrics_csv": metrics_path.relative_to(run_dir).as_posix(),
            "analysis": post_analysis,
            "preflight": preflight_path.relative_to(run_dir).as_posix(),
        }
        _write_json(_output_path(run_dir, "provenance", "campaign_summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except (CampaignError, OSError, subprocess.SubprocessError) as error:
        print(f"erank-frame-campaign: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
