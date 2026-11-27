"""Formal 12-prompt Full 14B Target main-table campaign.

Generation, offline decode, and official VBench-Long evaluation are deliberately
three disjoint phases.  Only one Krea Target worker exists during generation;
the Draft worker is launched afterwards solely for VAE decode.  This keeps the
speed denominator free of model loading, prompt encoding, decode, video encode,
and quality-evaluation time while preserving immutable per-prompt evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .correction_artifacts import (
    target_artifact_spec_sha256,
    verify_target_artifact_spec,
)
from .correction_runtime import (
    DeterministicNoiseProvider,
    JsonlWorkerClient,
    launch_draft_decoder_worker,
    launch_target_generation_worker,
)
from .correction_state import write_history_manifest
from .erank_frame_campaign import verify_artifact_spec as verify_draft_artifact_spec
from .main_table_baseline_campaign import (
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    _gpu_compute_pids,
    _gpu_uuid,
    _model_root,
    _run_dir,
    _selected_prompts,
    _sha256_file,
    _visible_devices,
    _write_new_json,
)
from .main_table_full_target import (
    CHUNK_COUNT,
    METHOD_ID,
    SEED,
    FullTargetGeneration,
    FullTargetRollout,
    decode_full_target_trajectory,
    generate_full_target_trajectory,
)
from .main_table_paper_figures import RAW_SCHEMA
from .main_table_v1 import PROMPT_IDS, QUALITY_DIMENSIONS, load_bundle
from .duration_sweep import (
    duration_profile,
    RESULT_SCHEMA as DURATION_RESULT_SCHEMA,
    RAW_SCHEMA as DURATION_RAW_SCHEMA,
)
from .metadata import collect_git_metadata, find_git_root, utc_now
from . import optimized_duration_control as optimized_control
from .vbench_long import OfficialVBenchLongScorer, VBenchLongVideoRequest
from .vbench_short import OfficialVBenchScorer
from .baseline_population import load_baseline_population, select_prompt_shard


PREFLIGHT_SCHEMA = "main_table_full_target_campaign_preflight_v1"
METHOD_RESULT_SCHEMA = "main_table_full_target_prompt_result_v1"


class FullTargetCampaignError(RuntimeError):
    """Raised when Full Target cannot prove the frozen campaign contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullTargetCampaignError(message)


def _finite_positive(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"{label} must be finite and positive",
    )
    return float(value)


def _assert_idle_pair(decoder_device: str, target_device: str) -> None:
    occupied = {
        decoder_device: _gpu_compute_pids(_gpu_uuid(decoder_device)),
        target_device: _gpu_compute_pids(_gpu_uuid(target_device)),
    }
    _require(
        not occupied[decoder_device] and not occupied[target_device],
        "selected Full Target GPU pair is not idle immediately before launch",
    )


def _assert_worker_owns_gpu(
    *,
    worker: JsonlWorkerClient,
    worker_device: str,
    reserved_idle_device: str,
    phase: str,
) -> None:
    process = getattr(worker, "process", None)
    pid = getattr(process, "pid", None)
    _require(isinstance(pid, int) and pid > 0, f"{phase} worker PID is unavailable")
    worker_pids = _gpu_compute_pids(_gpu_uuid(worker_device))
    idle_pids = _gpu_compute_pids(_gpu_uuid(reserved_idle_device))
    _require(not idle_pids,
             f"reserved idle GPU acquired a compute process during {phase}: "
             f"reserved_device={reserved_idle_device}, pids={sorted(idle_pids)}, "
             f"worker_pid={pid}, worker_device={worker_device}, worker_pids={sorted(worker_pids)}")
    _require(
        worker_pids <= {pid},
        f"generation GPU acquired an unrelated compute process during {phase}",
    )


TargetLauncher = Callable[..., JsonlWorkerClient]
DecoderLauncher = Callable[..., JsonlWorkerClient]
ScorerFactory = Callable[..., Any]


def run_campaign(
    *,
    draft_artifact_spec_path: Path,
    target_artifact_spec_path: Path,
    vbench_checkout: Path,
    vbench_python: Path,
    vbench_asset_manifest: Path,
    vbench_asset_spec: Path,
    target_launcher: TargetLauncher = launch_target_generation_worker,
    decoder_launcher: DecoderLauncher = launch_draft_decoder_worker,
    scorer_factory: ScorerFactory = OfficialVBenchLongScorer,
    enforce_gpu_idle: bool = True,
    smoke_prompt_id: str | None = None,
    duration_seconds: int | None = None,
    optimized_duration_control: bool = False,
    baseline_population_v2: bool = False,
    prompt_shard_index: int = 0,
    prompt_shard_count: int = 1,
) -> dict[str, Any]:
    """Run all twelve prompts and write one strict method result."""

    if optimized_duration_control:
        optimized_control.validate_invocation(method=METHOD_ID, seconds=duration_seconds, smoke=smoke_prompt_id)
    profile = duration_profile(duration_seconds) if duration_seconds is not None else None
    chunk_count = profile.chunk_count if profile else CHUNK_COUNT
    run_dir = _run_dir()
    model_root = _model_root()
    decoder_device, target_device = _visible_devices()
    repository_root = find_git_root(Path.cwd())
    population_binding = None
    if baseline_population_v2:
        _require(profile is not None and not optimized_duration_control, "baseline v2 requires an explicit duration and no optimized control")
        prompts, population_binding = load_baseline_population(repository_root)
        expected_prompt_ids = tuple(p['id'] for p in prompts)
        if duration_seconds in (15, 30) and scorer_factory is OfficialVBenchLongScorer:
            scorer_factory = OfficialVBenchScorer
    else:
        bundle = load_bundle(repository_root)
        prompts = _selected_prompts(repository_root, bundle.protocol)
        expected_prompt_ids = PROMPT_IDS
    benchmark = "VBench" if baseline_population_v2 and duration_seconds in (15, 30) else "VBench-Long"
    metric_key = "official_vbench" if benchmark == "VBench" else "official_vbench_long"
    duration_metadata = profile.metadata() if profile else None
    if baseline_population_v2:
        duration_metadata = {k: v for k, v in duration_metadata.items()
                             if not k.startswith(('ours_', 'calibration_')) and k != 'budget_rule'}
        duration_metadata['experiment_id'] = 'main-table-population-v2'
        duration_metadata['benchmark'] = benchmark
    if prompt_shard_count != 1 or prompt_shard_index != 0:
        _require(baseline_population_v2 and smoke_prompt_id is None,
                 "prompt sharding requires baseline v2 and cannot be mixed with smoke")
        prompts = select_prompt_shard(prompts, prompt_shard_index, prompt_shard_count)
        expected_prompt_ids = tuple(p['id'] for p in prompts)
        population_binding['shard_index'] = prompt_shard_index
        population_binding['shard_count'] = prompt_shard_count
        population_binding['full_population_count'] = 24
    if smoke_prompt_id is not None:
        _require(
            smoke_prompt_id in expected_prompt_ids,
            "smoke prompt must belong to the frozen main-table population",
        )
        prompts = [item for item in prompts if item["id"] == smoke_prompt_id]
    active_prompt_ids = tuple(item["id"] for item in prompts)
    _require(
        active_prompt_ids == ((smoke_prompt_id,) if smoke_prompt_id else expected_prompt_ids),
        "active prompt selection drifted",
    )

    try:
        _draft_spec, draft_verified, draft_module = verify_draft_artifact_spec(
            draft_artifact_spec_path,
            model_root=model_root,
            repo_root=repository_root,
        )
        _target_spec, target_verified, target_module = verify_target_artifact_spec(
            target_artifact_spec_path,
            model_root=model_root,
            repo_root=repository_root,
        )
    except BaseException as error:
        raise FullTargetCampaignError(f"artifact preflight failed: {error}") from error
    _require(
        draft_module == "cfexp.self_forcing_frame_adapter",
        "Draft artifact spec selected the wrong adapter",
    )
    _require(
        target_module == "cfexp.krea_chunk_backend",
        "Target artifact spec selected the wrong backend",
    )
    draft_spec_sha256, _ = _sha256_file(draft_artifact_spec_path.resolve(strict=True))
    target_spec_sha256 = target_artifact_spec_sha256(target_artifact_spec_path)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": utc_now(),
        "git": collect_git_metadata(repository_root),
        "method_id": METHOD_ID,
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "prompt_ids": list(active_prompt_ids),
        "seed": SEED,
        "chunk_count": chunk_count,
        "decoder_device": decoder_device,
        "target_device": target_device,
        "draft_artifact_spec_sha256": draft_spec_sha256,
        "target_artifact_spec_sha256": target_spec_sha256,
        "draft_verified_artifacts": draft_verified,
        "target_verified_artifacts": target_verified,
    }
    if profile:
        preflight["duration"] = duration_metadata
    if population_binding:
        preflight["population"] = population_binding
    if optimized_duration_control:
        preflight["comparison_contract"] = optimized_control.contract()
    _write_new_json(run_dir / "provenance" / "preflight.json", preflight)

    if enforce_gpu_idle:
        _assert_idle_pair(decoder_device, target_device)
    target_load_started = time.perf_counter()
    target = target_launcher(
        run_dir=run_dir,
        repo_root=repository_root,
        target_artifact_spec_path=target_artifact_spec_path,
        target_artifact_spec_sha256=target_spec_sha256,
        target_verified_artifacts=target_verified,
        visible_device=target_device,
        chunk_count=chunk_count,
    )
    target_model_load_seconds = time.perf_counter() - target_load_started
    if enforce_gpu_idle:
        _assert_worker_owns_gpu(
            worker=target,
            worker_device=target_device,
            reserved_idle_device=decoder_device,
            phase="Target generation",
        )
    generations: dict[str, FullTargetGeneration] = {}
    noise = DeterministicNoiseProvider(run_dir=run_dir)
    try:
        for prompt in prompts:
            if enforce_gpu_idle and profile:
                _assert_worker_owns_gpu(
                    worker=target, worker_device=target_device,
                    reserved_idle_device=decoder_device, phase="Target generation",
                )
            generation = generate_full_target_trajectory(
                prompt_id=prompt["id"],
                prompt=prompt["text"],
                target=target,
                noise_provider=noise,
                chunk_count=chunk_count,
                duration_seconds=duration_seconds,
                safety_check=(
                    (
                        lambda _stage, _chunk: _assert_worker_owns_gpu(
                            worker=target,
                            worker_device=target_device,
                            reserved_idle_device=decoder_device,
                            phase="Target generation",
                        )
                    )
                    if enforce_gpu_idle and not profile
                    else None
                ),
            )
            history_path = (
                run_dir / "provenance" / METHOD_ID / f"{prompt['id']}-history.json"
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            write_history_manifest(history_path, generation.history)
            generations[prompt["id"]] = generation
    finally:
        target.close()

    _require(
        tuple(generations) == active_prompt_ids,
        "Full Target generation coverage drifted",
    )
    if enforce_gpu_idle:
        _require(
            not _gpu_compute_pids(_gpu_uuid(target_device)),
            "Target GPU retained a compute process after generation shutdown",
        )
    decoder_load_started = time.perf_counter()
    decoder = decoder_launcher(
        run_dir=run_dir,
        repo_root=repository_root,
        draft_artifact_spec_path=draft_artifact_spec_path,
        draft_artifact_spec_sha256=draft_spec_sha256,
        draft_verified_artifacts=draft_verified,
        visible_device=decoder_device,
        chunk_count=chunk_count,
    )
    decoder_model_load_seconds = time.perf_counter() - decoder_load_started
    if enforce_gpu_idle:
        _assert_worker_owns_gpu(
            worker=decoder,
            worker_device=decoder_device,
            reserved_idle_device=target_device,
            phase="offline decode",
        )
    rollouts: dict[str, FullTargetRollout] = {}
    try:
        for prompt_id in active_prompt_ids:
            if enforce_gpu_idle:
                _assert_worker_owns_gpu(
                    worker=decoder,
                    worker_device=decoder_device,
                    reserved_idle_device=target_device,
                    phase="offline decode",
                )
            rollouts[prompt_id] = decode_full_target_trajectory(
                generation=generations[prompt_id],
                decoder=decoder,
            )
            if enforce_gpu_idle:
                _assert_worker_owns_gpu(
                    worker=decoder,
                    worker_device=decoder_device,
                    reserved_idle_device=target_device,
                    phase="offline decode",
                )
    finally:
        decoder.close()

    if enforce_gpu_idle:
        _require(
            not _gpu_compute_pids(_gpu_uuid(decoder_device)),
            "decoder GPU retained a compute process after shutdown",
        )
        _require(
            not _gpu_compute_pids(_gpu_uuid(target_device)),
            "quality GPU is not idle immediately before VBench-Long",
        )
    scorer = scorer_factory(
        run_dir=run_dir,
        checkout=vbench_checkout,
        python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest,
        asset_spec=vbench_asset_spec,
        visible_device=target_device,
    )
    prompt_text = {item["id"]: item["text"] for item in prompts}
    score_requests = [
        VBenchLongVideoRequest(
            video_path=run_dir / str(rollouts[prompt_id].result["video"]["path"]),
            prompt=prompt_text[prompt_id],
            measurement_id=f"mt-{METHOD_ID}-{prompt_id}",
        )
        for prompt_id in active_prompt_ids
    ]
    vbench_started = time.perf_counter()
    measurements = scorer.score_videos(score_requests)
    vbench_wall_seconds = time.perf_counter() - vbench_started
    _require(
        set(measurements) == {item.measurement_id for item in score_requests},
        "official VBench-Long coverage is incomplete",
    )

    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for prompt_id in active_prompt_ids:
        rollout = rollouts[prompt_id]
        measurement_id = f"mt-{METHOD_ID}-{prompt_id}"
        measurement = measurements[measurement_id]
        scores = dict(measurement.dimension_scores)
        _require(tuple(scores) == QUALITY_DIMENSIONS, "VBench dimension order drifted")
        generation_wall_seconds = _finite_positive(
            rollout.result["timing"]["generation_wall_seconds"],
            "generation wall time",
        )
        rows.append(
            {
                "method_id": METHOD_ID,
                "prompt_id": prompt_id,
                "seed": SEED,
                "generation_wall_seconds": generation_wall_seconds,
                "quality_scores": scores,
                metric_key + "_verified": True,
            }
        )
        evidence_path = (run_dir / measurement.evidence_path).resolve(strict=True)
        evidence_sha256, evidence_nbytes = _sha256_file(evidence_path)
        evidence_item = {
            "prompt_id": prompt_id,
            "measurement_id": measurement_id,
            "path": evidence_path.relative_to(run_dir).as_posix(),
            "sha256": evidence_sha256,
            "nbytes": evidence_nbytes,
        }
        evidence.append(evidence_item)
        prompt_result = {
            "schema": METHOD_RESULT_SCHEMA,
            "method_id": METHOD_ID,
            "prompt_id": prompt_id,
            "seed": SEED,
            "generation": dict(rollout.result),
            "model_load_seconds": {
                "target": target_model_load_seconds,
                "offline_decoder": decoder_model_load_seconds,
            },
            metric_key: {
                "verified": True,
                "dimension_scores": scores,
                "macro_mean": measurement.macro_mean,
                "motion_smoothness": measurement.motion_smoothness,
                "evidence": evidence_item,
            },
        }
        _write_new_json(
            run_dir / "provenance" / METHOD_ID / f"{prompt_id}.json",
            prompt_result,
        )

    if profile:
        for row in rows:
            row["duration"] = duration_metadata
    raw_path = run_dir / "raw" / f"main-table-{METHOD_ID}.json"
    _write_new_json(raw_path, {"schema": DURATION_RAW_SCHEMA if profile else RAW_SCHEMA, "rows": rows})
    raw_sha256, raw_nbytes = _sha256_file(raw_path)
    result = {
        "schema": DURATION_RESULT_SCHEMA if profile else (SMOKE_RESULT_SCHEMA if smoke_prompt_id else RESULT_SCHEMA),
        "status": "succeeded",
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "method_id": METHOD_ID,
        "prompt_count": len(active_prompt_ids),
        "row_count": len(rows),
        "seed_count": 1,
        "raw_rows": {
            "path": raw_path.relative_to(run_dir).as_posix(),
            "sha256": raw_sha256,
            "nbytes": raw_nbytes,
        },
        "vbench_wall_seconds": vbench_wall_seconds,
        "vbench_measurements": evidence,
        "generation_device": target_device,
        "quality_device": target_device,
        "offline_decoder_device": decoder_device,
        "target_model_load_seconds": target_model_load_seconds,
        "offline_decoder_model_load_seconds": decoder_model_load_seconds,
    }
    if profile:
        result["duration"] = duration_metadata
    if population_binding:
        result["population"] = population_binding
        result["benchmark"] = benchmark
        result["schema"] = "baseline_population_v2_method_result"
        if prompt_shard_count > 1:
            result["run_kind"] = "formal_prompt_shard"
    if optimized_duration_control:
        result["comparison_contract"] = optimized_control.contract()
    _write_new_json(run_dir / "result.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cfexp.main_table_full_target_campaign"
    )
    parser.add_argument("--draft-artifact-spec", type=Path, required=True)
    parser.add_argument("--target-artifact-spec", type=Path, required=True)
    parser.add_argument("--vbench-checkout", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-asset-manifest", type=Path, required=True)
    parser.add_argument("--vbench-asset-spec", type=Path, required=True)
    parser.add_argument("--smoke-prompt-id")
    parser.add_argument("--baseline-population-v2", action="store_true")
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    parser.add_argument("--duration-seconds", type=int, choices=(15, 30, 60))
    parser.add_argument("--optimized-duration-control", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_campaign(
            draft_artifact_spec_path=args.draft_artifact_spec,
            target_artifact_spec_path=args.target_artifact_spec,
            vbench_checkout=args.vbench_checkout,
            vbench_python=args.vbench_python,
            vbench_asset_manifest=args.vbench_asset_manifest,
            vbench_asset_spec=args.vbench_asset_spec,
            smoke_prompt_id=args.smoke_prompt_id,
            duration_seconds=args.duration_seconds,
            optimized_duration_control=args.optimized_duration_control,
            baseline_population_v2=args.baseline_population_v2,
            prompt_shard_index=args.prompt_shard_index,
            prompt_shard_count=args.prompt_shard_count,
        )
    except (FullTargetCampaignError, optimized_control.OptimizedDurationError, OSError, subprocess.SubprocessError) as error:
        print(f"main-table-full-target-campaign: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FullTargetCampaignError",
    "METHOD_RESULT_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "run_campaign",
]
