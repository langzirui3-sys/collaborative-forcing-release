"""Formal one-method campaign for the official Light/Dummy main-table baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .main_table_paper_figures import RAW_SCHEMA
from .main_table_v1 import METHOD_IDS, PROMPT_IDS, QUALITY_DIMENSIONS, load_bundle
from .metadata import collect_git_metadata, find_git_root, utc_now
from .duration_sweep import duration_profile
from .baseline_population import load_baseline_population, select_prompt_shard
from .vbench_short import OfficialVBenchScorer
from .vbench_long import OfficialVBenchLongScorer, VBenchLongVideoRequest


RESULT_SCHEMA = "main_table_method_campaign_result_v1"
SMOKE_RESULT_SCHEMA = "main_table_method_smoke_result_v1"
PREFLIGHT_SCHEMA = "main_table_baseline_campaign_preflight_v1"
ARTIFACT_SPEC_SCHEMA = "main_table_baseline_artifacts_v1"
SUPPORTED_METHODS = ("light_forcing", "dummy_forcing", "draft_only")
ADAPTER_MODULES = {
    "draft_only": "cfexp.longlive_adapter",
    "sdvg": "cfexp.sdvg_adapter",
    "light_forcing": "cfexp.light_forcing_adapter",
    "dummy_forcing": "cfexp.dummy_forcing_adapter",
    "t_stitch": "cfexp.t_stitch_adapter",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MainTableBaselineCampaignError(RuntimeError):
    """Raised when a baseline campaign cannot prove its formal contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MainTableBaselineCampaignError(
            f"value is not canonical JSON: {error}"
        ) from error


def _duplicate_checked_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise MainTableBaselineCampaignError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                MainTableBaselineCampaignError(
                    f"{label} contains invalid JSON constant {item!r}"
                )
            ),
        )
    except MainTableBaselineCampaignError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MainTableBaselineCampaignError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise MainTableBaselineCampaignError(f"{label} must contain one object")
    return value


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
        size = handle.tell()
    return digest.hexdigest(), size


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise MainTableBaselineCampaignError(
            f"immutable output already exists: {path}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _run_dir() -> Path:
    value = os.environ.get("CFEXP_RUN_DIR")
    if not value:
        raise MainTableBaselineCampaignError("CFEXP_RUN_DIR is required")
    unresolved = Path(value).expanduser()
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise MainTableBaselineCampaignError(
            "CFEXP_RUN_DIR must be an absolute non-symlink directory"
        )
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_dir():
        raise MainTableBaselineCampaignError("CFEXP_RUN_DIR must be a directory")
    return resolved


def _model_root() -> Path:
    value = os.environ.get("CFEXP_MODEL_ROOT")
    if not value:
        raise MainTableBaselineCampaignError("CFEXP_MODEL_ROOT is required")
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise MainTableBaselineCampaignError("CFEXP_MODEL_ROOT must be a regular directory")
    return root


def _visible_devices() -> tuple[str, str]:
    devices = tuple(
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    )
    if len(devices) != 2 or len(set(devices)) != 2:
        raise MainTableBaselineCampaignError(
            "formal baseline run requires exactly two unique visible GPUs"
        )
    return devices[0], devices[1]


def verify_artifact_spec(
    path: Path, *, method_id: str, model_root: Path, repository_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise MainTableBaselineCampaignError("artifact spec must be a regular file")
    spec = _load_json(path, "artifact spec")
    if method_id not in ADAPTER_MODULES:
        raise MainTableBaselineCampaignError("unsupported artifact method identity")
    if (
        spec.get("schema") != ARTIFACT_SPEC_SCHEMA
        or spec.get("status") != "frozen"
        or spec.get("method_id") != method_id
        or spec.get("adapter_module") != ADAPTER_MODULES[method_id]
    ):
        raise MainTableBaselineCampaignError("artifact spec identity is invalid")
    if set(spec) != {
        "schema",
        "status",
        "method_id",
        "adapter_module",
        "adapter_config",
        "artifacts",
    }:
        raise MainTableBaselineCampaignError("artifact spec field schema drifted")
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise MainTableBaselineCampaignError("artifact spec contains no artifacts")
    roots = {"model": model_root, "repository": repository_root}
    names: set[str] = set()
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "root",
            "path",
            "sha256",
            "role",
        }:
            raise MainTableBaselineCampaignError(
                f"artifact {index} field schema drifted"
            )
        name = item.get("name")
        root_name = item.get("root")
        relative_text = item.get("path")
        expected_sha256 = item.get("sha256")
        if not isinstance(name, str) or not name or name in names:
            raise MainTableBaselineCampaignError(f"artifact {index} name is invalid")
        if root_name not in roots or not isinstance(relative_text, str):
            raise MainTableBaselineCampaignError(f"artifact {name} root/path is invalid")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise MainTableBaselineCampaignError(f"artifact {name} path is unsafe")
        if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
            raise MainTableBaselineCampaignError(f"artifact {name} SHA-256 is invalid")
        root = roots[str(root_name)].resolve(strict=True)
        resolved = (root / relative).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise MainTableBaselineCampaignError(f"artifact {name} escapes its root") from error
        metadata = resolved.stat()
        if resolved.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise MainTableBaselineCampaignError(f"artifact {name} is not a regular file")
        actual_sha256, size = _sha256_file(resolved)
        if actual_sha256 != expected_sha256 or size < 1:
            raise MainTableBaselineCampaignError(f"artifact {name} changed after freezing")
        names.add(name)
        verified.append(
            {
                "name": name,
                "resolved_path": str(resolved),
                "sha256": actual_sha256,
                "size": size,
            }
        )
    return spec, verified, str(spec["adapter_module"])


def _selected_prompts(repository_root: Path, protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    population = protocol.get("prompt_population")
    if not isinstance(population, Mapping):
        raise MainTableBaselineCampaignError("protocol has no prompt population")
    source_relative = population.get("source_file")
    if not isinstance(source_relative, str):
        raise MainTableBaselineCampaignError("protocol prompt source is invalid")
    source = (repository_root / source_relative).resolve(strict=True)
    source_sha256, _ = _sha256_file(source)
    if source_sha256 != population.get("source_sha256"):
        raise MainTableBaselineCampaignError("protocol prompt source hash drifted")
    value = _load_json(source, "prompt source")
    raw_prompts = value.get("prompts")
    if not isinstance(raw_prompts, list):
        raise MainTableBaselineCampaignError("prompt source has no prompts")
    indexed: dict[str, str] = {}
    for item in raw_prompts:
        if isinstance(item, Mapping) and isinstance(item.get("prompt_id"), str) and isinstance(item.get("prompt"), str):
            indexed[str(item["prompt_id"])] = str(item["prompt"])
    if tuple(population.get("prompt_ids", ())) != PROMPT_IDS or any(
        prompt_id not in indexed for prompt_id in PROMPT_IDS
    ):
        raise MainTableBaselineCampaignError("selected prompt population drifted")
    return [{"id": prompt_id, "text": indexed[prompt_id]} for prompt_id in PROMPT_IDS]


def _default_adapter_runner(
    *,
    module: str,
    batch_request: Path,
    run_dir: Path,
    generation_device: str,
    quality_device: str,
) -> Mapping[str, Any]:
    stdout = run_dir / "logs" / "adapter.stdout.log"
    stderr = run_dir / "logs" / "adapter.stderr.log"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = generation_device
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(name, None)
    generation_uuid = _gpu_uuid(generation_device)
    quality_uuid = _gpu_uuid(quality_device)
    if _gpu_compute_pids(generation_uuid) or _gpu_compute_pids(quality_uuid):
        raise MainTableBaselineCampaignError(
            "a selected GPU already has a compute process immediately before generation"
        )
    violation: str | None = None
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--batch-request", str(batch_request)],
            cwd=str(find_git_root(Path.cwd())),
            env=environment,
            stdout=out,
            stderr=err,
            text=True,
        )
        while process.poll() is None:
            quality_pids = _gpu_compute_pids(quality_uuid)
            generation_pids = _gpu_compute_pids(generation_uuid)
            foreign_generation = generation_pids - {process.pid}
            if quality_pids:
                violation = "reserved quality GPU acquired a compute process during generation"
            elif foreign_generation:
                violation = "generation GPU acquired an unrelated compute process"
            if violation is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                continue
        returncode = process.wait()
    if violation is not None:
        raise MainTableBaselineCampaignError(violation)
    if returncode != 0:
        raise MainTableBaselineCampaignError(
            f"baseline adapter exited with status {returncode}"
        )
    value = _load_json(stdout, "adapter stdout")
    if not isinstance(value.get("results"), list):
        raise MainTableBaselineCampaignError("adapter stdout omitted batch results")
    return value


def _nvidia_smi(arguments: Sequence[str]) -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise MainTableBaselineCampaignError(f"cannot execute nvidia-smi: {error}") from error
    if completed.returncode != 0:
        raise MainTableBaselineCampaignError(
            f"nvidia-smi failed with status {completed.returncode}"
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _gpu_uuid(device: str) -> str:
    lines = _nvidia_smi(
        ["-i", device, "--query-gpu=uuid", "--format=csv,noheader,nounits"]
    )
    if len(lines) != 1 or not lines[0].startswith("GPU-"):
        raise MainTableBaselineCampaignError(
            f"cannot resolve one physical GPU UUID for device {device!r}"
        )
    return lines[0]


def _gpu_compute_pids(gpu_uuid: str) -> set[int]:
    lines = _nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    pids: set[int] = set()
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise MainTableBaselineCampaignError("nvidia-smi returned an invalid process row")
        if fields[0] != gpu_uuid:
            continue
        if fields[1] in {"N/A", "[N/A]"}:
            # Some NVIDIA driver releases emit one placeholder row for an
            # otherwise idle GPU instead of omitting it from the compute-app
            # query.  This is not a process and therefore has no PID to own.
            continue
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise MainTableBaselineCampaignError("nvidia-smi returned an invalid PID") from error
        if pid <= 0:
            raise MainTableBaselineCampaignError("nvidia-smi returned a non-positive PID")
        pids.add(pid)
    return pids


AdapterRunner = Callable[..., Mapping[str, Any]]
ScorerFactory = Callable[..., Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MainTableBaselineCampaignError(message)


def run_campaign(
    *,
    method_id: str,
    artifact_spec_path: Path,
    vbench_checkout: Path,
    vbench_python: Path,
    vbench_asset_manifest: Path,
    vbench_asset_spec: Path,
    adapter_runner: AdapterRunner = _default_adapter_runner,
    scorer_factory: ScorerFactory = OfficialVBenchLongScorer,
    smoke_prompt_id: str | None = None,
    model_pair: str = "krea_self_forcing",
    baseline_population_v2: bool = False,
    duration_seconds: int | None = None,
    prompt_shard_index: int = 0,
    prompt_shard_count: int = 1,
) -> dict[str, Any]:
    if method_id not in SUPPORTED_METHODS:
        raise MainTableBaselineCampaignError(
            f"method must be one of {SUPPORTED_METHODS!r}"
        )
    run_dir = _run_dir()
    _require((model_pair == "krea_longlive" and method_id == "draft_only" and baseline_population_v2)
             or (model_pair == "krea_self_forcing" and method_id != "draft_only"),
             "method and model pair are incompatible")
    model_root = _model_root()
    generation_device, quality_device = _visible_devices()
    repository_root = find_git_root(Path.cwd())
    population_binding = None
    duration_metadata = None
    chunk_count = 120
    benchmark = "VBench-Long"
    if baseline_population_v2:
        _require(duration_seconds in (15, 30, 60), "baseline v2 requires a duration")
        profile = duration_profile(duration_seconds)
        chunk_count = profile.chunk_count
        prompts, population_binding = load_baseline_population(repository_root, model_pair=model_pair)
        benchmark = "VBench" if duration_seconds in (15, 30) else "VBench-Long"
        duration_metadata = {k: v for k, v in profile.metadata().items()
                             if not k.startswith(('ours_', 'calibration_')) and k != 'budget_rule'}
        duration_metadata.update(experiment_id='main-table-population-v2', benchmark=benchmark)
        if benchmark == "VBench" and scorer_factory is OfficialVBenchLongScorer:
            scorer_factory = OfficialVBenchScorer
        if prompt_shard_count != 1 or prompt_shard_index != 0:
            _require(smoke_prompt_id is None, "cannot mix smoke with prompt sharding")
            prompts = select_prompt_shard(prompts, prompt_shard_index, prompt_shard_count)
            population_binding.update(shard_index=prompt_shard_index,
                                      shard_count=prompt_shard_count, full_population_count=24)
    else:
        _require(duration_seconds is None and prompt_shard_count == 1 and prompt_shard_index == 0,
                 "duration and sharding require baseline v2")
        bundle = load_bundle(repository_root)
        prompts = _selected_prompts(repository_root, bundle.protocol)
        _require(tuple(p['id'] for p in prompts) == PROMPT_IDS, "frozen population drifted")
    if smoke_prompt_id is not None:
        _require(smoke_prompt_id in {p['id'] for p in prompts}, "smoke prompt is outside the population")
        prompts = [item for item in prompts if item["id"] == smoke_prompt_id]
    active_prompt_ids = tuple(item["id"] for item in prompts)
    metric_key = "official_vbench" if benchmark == "VBench" else "official_vbench_long"
    spec, verified, adapter_module = verify_artifact_spec(
        artifact_spec_path,
        method_id=method_id,
        model_root=model_root,
        repository_root=repository_root,
    )
    if baseline_population_v2:
        geometry = spec["adapter_config"]["generation"]
        _require(geometry["noise_shape"][0] == chunk_count * 3 and geometry["rgb_frames"] == chunk_count * 12 - 3, "artifact horizon differs from requested duration")
    artifact_spec_sha256, _ = _sha256_file(artifact_spec_path.resolve(strict=True))
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": utc_now(),
        "git": collect_git_metadata(repository_root),
        "method_id": method_id,
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "prompt_ids": list(active_prompt_ids),
        "seed": 20260911,
        "generation_device": generation_device,
        "quality_device": quality_device,
        "artifact_spec_sha256": artifact_spec_sha256,
        "verified_artifacts": verified,
        "adapter_module": adapter_module,
    }
    if baseline_population_v2:
        preflight.update(population=population_binding, duration=duration_metadata, benchmark=benchmark, model_pair=model_pair)
    _write_new_json(run_dir / "provenance" / "preflight.json", preflight)

    requests: list[dict[str, Any]] = []
    adapter = importlib.import_module(adapter_module)
    for prompt in prompts:
        prompt_sha256 = hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
        requests.append(
            {
                "schema": adapter.REQUEST_SCHEMA,
                "method_id": method_id,
                "worker_index": 0,
                "prompt": {**prompt, "sha256": prompt_sha256},
                "seed": 20260911,
                "output": {
                    "video_path": f"videos/{method_id}/{prompt['id']}.mp4",
                    "result_path": f"provenance/{method_id}/{prompt['id']}.json",
                },
                "artifact_spec_path": str(artifact_spec_path.resolve(strict=True)),
                "artifact_spec_sha256": artifact_spec_sha256,
                "verified_artifacts": verified,
            }
        )
    batch_path = run_dir / "provenance" / "adapter-batch-request.json"
    _write_new_json(
        batch_path,
        {"schema": adapter.BATCH_REQUEST_SCHEMA, "requests": requests},
    )
    adapter_result = adapter_runner(
        module=adapter_module,
        batch_request=batch_path,
        run_dir=run_dir,
        generation_device=generation_device,
        quality_device=quality_device,
    )
    results = adapter_result.get("results")
    if not isinstance(results, list) or len(results) != len(active_prompt_ids):
        raise MainTableBaselineCampaignError("adapter result coverage is incomplete")
    by_prompt: dict[str, Mapping[str, Any]] = {}
    for item in results:
        if not isinstance(item, Mapping) or item.get("method_id") != method_id:
            raise MainTableBaselineCampaignError("adapter returned invalid method evidence")
        prompt = item.get("prompt")
        timing = item.get("timing")
        video = item.get("video")
        if not isinstance(prompt, Mapping) or prompt.get("id") not in active_prompt_ids:
            raise MainTableBaselineCampaignError("adapter returned an unknown prompt")
        prompt_id = str(prompt["id"])
        if prompt_id in by_prompt or not isinstance(timing, Mapping) or not isinstance(video, Mapping):
            raise MainTableBaselineCampaignError("adapter result schema or uniqueness failed")
        latency = timing.get("generation_wall_seconds")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(float(latency)) or float(latency) <= 0:
            raise MainTableBaselineCampaignError("adapter generation wall time is invalid")
        if baseline_population_v2:
            if method_id == "draft_only":
                _require(item.get("model_pair") == model_pair, "adapter model pair differs from requested pair")
            _require(video.get("frames") == chunk_count * 12 - 3 and video.get("fps") == 24, "adapter output duration differs from requested horizon")
        video_path = (run_dir / str(video.get("path"))).resolve(strict=True)
        try:
            video_path.relative_to(run_dir)
        except ValueError as error:
            raise MainTableBaselineCampaignError("adapter video escaped the run directory") from error
        sha256, nbytes = _sha256_file(video_path)
        if video.get("sha256") != sha256 or video.get("nbytes") != nbytes:
            raise MainTableBaselineCampaignError("adapter video binding is invalid")
        by_prompt[prompt_id] = item
    if tuple(by_prompt) != active_prompt_ids:
        raise MainTableBaselineCampaignError("adapter prompt order or coverage drifted")

    if adapter_runner is _default_adapter_runner and _gpu_compute_pids(
        _gpu_uuid(quality_device)
    ):
        raise MainTableBaselineCampaignError(
            "quality GPU is no longer idle immediately before VBench-Long"
        )
    scorer = scorer_factory(
        run_dir=run_dir,
        checkout=vbench_checkout,
        python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest,
        asset_spec=vbench_asset_spec,
        visible_device=quality_device,
    )
    score_requests = [
        VBenchLongVideoRequest(
            video_path=run_dir / str(by_prompt[prompt_id]["video"]["path"]),
            prompt=next(
                item["text"] for item in prompts if item["id"] == prompt_id
            ),
            measurement_id=f"mt-{method_id}-{prompt_id}",
        )
        for prompt_id in active_prompt_ids
    ]
    vbench_started = time.perf_counter()
    measurements = scorer.score_videos(score_requests)
    vbench_wall_seconds = time.perf_counter() - vbench_started
    if set(measurements) != {request.measurement_id for request in score_requests}:
        raise MainTableBaselineCampaignError("official VBench-Long coverage is incomplete")

    rows: list[dict[str, Any]] = []
    vbench_evidence: list[dict[str, Any]] = []
    for prompt_id in active_prompt_ids:
        measurement_id = f"mt-{method_id}-{prompt_id}"
        measurement = measurements[measurement_id]
        scores = dict(measurement.dimension_scores)
        if tuple(scores) != QUALITY_DIMENSIONS:
            raise MainTableBaselineCampaignError("official VBench-Long dimensions drifted")
        rows.append(
            {
                "method_id": method_id,
                "prompt_id": prompt_id,
                "seed": 20260911,
                "generation_wall_seconds": float(
                    by_prompt[prompt_id]["timing"]["generation_wall_seconds"]
                ),
                "quality_scores": scores,
                metric_key + "_verified": True,
                **({"duration": duration_metadata, "model_pair": model_pair} if duration_metadata else {}),
            }
        )
        evidence_path = (run_dir / measurement.evidence_path).resolve(strict=True)
        evidence_sha256, evidence_nbytes = _sha256_file(evidence_path)
        vbench_evidence.append(
            {
                "prompt_id": prompt_id,
                "measurement_id": measurement_id,
                "path": evidence_path.relative_to(run_dir).as_posix(),
                "sha256": evidence_sha256,
                "nbytes": evidence_nbytes,
            }
        )
    raw_path = run_dir / "raw" / f"main-table-{method_id}.json"
    _write_new_json(raw_path, {"schema": RAW_SCHEMA, "rows": rows})
    raw_sha256, raw_nbytes = _sha256_file(raw_path)
    result = {
        "schema": SMOKE_RESULT_SCHEMA if smoke_prompt_id else RESULT_SCHEMA,
        "status": "succeeded",
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "method_id": method_id,
        "prompt_count": len(active_prompt_ids),
        "row_count": len(rows),
        "seed_count": 1,
        "raw_rows": {
            "path": raw_path.relative_to(run_dir).as_posix(),
            "sha256": raw_sha256,
            "nbytes": raw_nbytes,
        },
        "vbench_wall_seconds": vbench_wall_seconds,
        "vbench_measurements": vbench_evidence,
        "generation_device": generation_device,
        "quality_device": quality_device,
    }
    if baseline_population_v2:
        result.update(schema="baseline_population_v2_method_result", population=population_binding,
                      benchmark=benchmark, duration=duration_metadata, model_pair=model_pair)
        if prompt_shard_count > 1:
            result["run_kind"] = "formal_prompt_shard"
    _write_new_json(run_dir / "result.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.main_table_baseline_campaign")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--artifact-spec", type=Path, required=True)
    parser.add_argument("--vbench-checkout", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-asset-manifest", type=Path, required=True)
    parser.add_argument("--vbench-asset-spec", type=Path, required=True)
    parser.add_argument("--smoke-prompt-id")
    parser.add_argument("--model-pair", choices=("krea_self_forcing", "krea_longlive"), default="krea_self_forcing")
    parser.add_argument("--baseline-population-v2", action="store_true")
    parser.add_argument("--duration-seconds", type=int, choices=(15,30,60))
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_campaign(
            method_id=args.method,
            artifact_spec_path=args.artifact_spec,
            vbench_checkout=args.vbench_checkout,
            vbench_python=args.vbench_python,
            vbench_asset_manifest=args.vbench_asset_manifest,
            vbench_asset_spec=args.vbench_asset_spec,
            smoke_prompt_id=args.smoke_prompt_id,
            model_pair=args.model_pair,
            baseline_population_v2=args.baseline_population_v2,
            duration_seconds=args.duration_seconds,
            prompt_shard_index=args.prompt_shard_index,
            prompt_shard_count=args.prompt_shard_count,
        )
    except (MainTableBaselineCampaignError, OSError, subprocess.SubprocessError) as error:
        print(f"main-table-baseline-campaign: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MainTableBaselineCampaignError",
    "PREFLIGHT_SCHEMA",
    "RESULT_SCHEMA",
    "SMOKE_RESULT_SCHEMA",
    "SUPPORTED_METHODS",
    "run_campaign",
    "verify_artifact_spec",
]
