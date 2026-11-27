"""Formal 12-prompt T-Stitch 1S+3T main-table campaign."""

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
    verify_artifact_spec as verify_main_table_artifact_spec,
)
from .main_table_paper_figures import RAW_SCHEMA
from .main_table_t_stitch import (
    CHUNK_COUNT,
    METHOD_ID,
    SEED,
    TStitchGeneration,
    TStitchRollout,
    decode_t_stitch_trajectory,
    generate_t_stitch_trajectory,
)
from .main_table_v1 import PROMPT_IDS, QUALITY_DIMENSIONS, load_bundle
from .metadata import collect_git_metadata, find_git_root, utc_now
from .vbench_long import OfficialVBenchLongScorer, VBenchLongVideoRequest
from .vbench_short import OfficialVBenchScorer
from .duration_sweep import duration_profile
from .baseline_population import load_baseline_population, select_prompt_shard


PREFLIGHT_SCHEMA = "main_table_t_stitch_campaign_preflight_v1"
METHOD_RESULT_SCHEMA = "main_table_t_stitch_prompt_result_v1"


class TStitchCampaignError(RuntimeError):
    """Raised when T-Stitch cannot prove the frozen campaign contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TStitchCampaignError(message)


def _finite_positive(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"{label} must be finite and positive",
    )
    return float(value)


def _assert_idle_pair(draft_device: str, target_device: str) -> None:
    occupied = {
        draft_device: _gpu_compute_pids(_gpu_uuid(draft_device)),
        target_device: _gpu_compute_pids(_gpu_uuid(target_device)),
    }
    _require(
        not occupied[draft_device] and not occupied[target_device],
        "selected T-Stitch GPU pair is not idle immediately before launch",
    )


def _assert_worker_pair_owns_gpus(
    *,
    draft: JsonlWorkerClient,
    target: JsonlWorkerClient,
    draft_device: str,
    target_device: str,
    phase: str,
) -> None:
    draft_pid = getattr(getattr(draft, "process", None), "pid", None)
    target_pid = getattr(getattr(target, "process", None), "pid", None)
    _require(
        isinstance(draft_pid, int)
        and draft_pid > 0
        and isinstance(target_pid, int)
        and target_pid > 0,
        f"{phase} worker PID is unavailable",
    )
    _require(
        _gpu_compute_pids(_gpu_uuid(draft_device)) <= {draft_pid},
        f"Draft GPU acquired an unrelated compute process during {phase}",
    )
    _require(
        _gpu_compute_pids(_gpu_uuid(target_device)) <= {target_pid},
        f"Target GPU acquired an unrelated compute process during {phase}",
    )


DraftLauncher = Callable[..., JsonlWorkerClient]
TargetLauncher = Callable[..., JsonlWorkerClient]
ScorerFactory = Callable[..., Any]


def run_campaign(
    *,
    t_stitch_artifact_spec_path: Path,
    draft_artifact_spec_path: Path,
    target_artifact_spec_path: Path,
    vbench_checkout: Path,
    vbench_python: Path,
    vbench_asset_manifest: Path,
    vbench_asset_spec: Path,
    draft_launcher: DraftLauncher = launch_draft_decoder_worker,
    target_launcher: TargetLauncher = launch_target_generation_worker,
    scorer_factory: ScorerFactory = OfficialVBenchLongScorer,
    enforce_gpu_idle: bool = True,
    smoke_prompt_id: str | None = None,
    baseline_population_v2: bool = False,
    duration_seconds: int | None = None,
    prompt_shard_index: int = 0,
    prompt_shard_count: int = 1,
) -> dict[str, Any]:
    run_dir = _run_dir()
    model_root = _model_root()
    draft_device, target_device = _visible_devices()
    repository_root = find_git_root(Path.cwd())
    population_binding = None
    duration_metadata = None
    chunk_count = CHUNK_COUNT
    benchmark = "VBench-Long"
    if baseline_population_v2:
        _require(duration_seconds in (15, 30, 60), "baseline v2 requires a duration")
        profile = duration_profile(duration_seconds)
        chunk_count = profile.chunk_count
        prompts, population_binding = load_baseline_population(repository_root)
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
    try:
        t_stitch_spec, t_stitch_verified, t_stitch_module = (
            verify_main_table_artifact_spec(
                t_stitch_artifact_spec_path,
                method_id="t_stitch",
                model_root=model_root,
                repository_root=repository_root,
            )
        )
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
        raise TStitchCampaignError(f"artifact preflight failed: {error}") from error
    _require(
        t_stitch_module == "cfexp.t_stitch_adapter",
        "T-Stitch artifact spec selected the wrong adapter identity",
    )
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
    t_stitch_spec_sha256, _ = _sha256_file(
        t_stitch_artifact_spec_path.resolve(strict=True)
    )
    adapter_config = t_stitch_spec.get("adapter_config")
    _require(isinstance(adapter_config, Mapping), "T-Stitch adapter config is absent")
    _require(
        adapter_config.get("draft_artifact_spec") == "draft_artifact_spec"
        and adapter_config.get("target_artifact_spec") == "target_artifact_spec"
        and adapter_config.get("switch_unit")
        == "denoising_step_within_each_autoregressive_chunk"
        and adapter_config.get("step_assignment")
        == [
            {"step_index": 0, "timestep": 1000, "model_role": "wan_1_3b_draft"},
            {"step_index": 1, "timestep": 750, "model_role": "krea_14b_target"},
            {"step_index": 2, "timestep": 500, "model_role": "krea_14b_target"},
            {"step_index": 3, "timestep": 250, "model_role": "krea_14b_target"},
        ],
        "T-Stitch frozen adapter contract drifted",
    )
    geometry = adapter_config.get("benchmark_geometry", {})
    _require(geometry.get("chunk_count") == chunk_count and geometry.get("rgb_frame_count") == chunk_count * 12 - 3,
             "T-Stitch artifact horizon differs from selected duration")
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": utc_now(),
        "git": collect_git_metadata(repository_root),
        "method_id": METHOD_ID,
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "prompt_ids": list(active_prompt_ids),
        "seed": SEED,
        "chunk_count": chunk_count,
        "draft_device": draft_device,
        "target_device": target_device,
        "draft_artifact_spec_sha256": draft_spec_sha256,
        "target_artifact_spec_sha256": target_spec_sha256,
        "t_stitch_artifact_spec_sha256": t_stitch_spec_sha256,
        "draft_verified_artifacts": draft_verified,
        "target_verified_artifacts": target_verified,
        "t_stitch_verified_artifacts": t_stitch_verified,
        "schedule": [
            {"step_index": 0, "timestep": 1000, "model_role": "draft"},
            {"step_index": 1, "timestep": 750, "model_role": "target"},
            {"step_index": 2, "timestep": 500, "model_role": "target"},
            {"step_index": 3, "timestep": 250, "model_role": "target"},
        ],
    }
    if baseline_population_v2:
        preflight.update(population=population_binding, duration=duration_metadata, benchmark=benchmark)
    _write_new_json(run_dir / "provenance" / "preflight.json", preflight)

    if enforce_gpu_idle:
        _assert_idle_pair(draft_device, target_device)
    draft_load_started = time.perf_counter()
    draft = draft_launcher(
        run_dir=run_dir,
        repo_root=repository_root,
        draft_artifact_spec_path=draft_artifact_spec_path,
        draft_artifact_spec_sha256=draft_spec_sha256,
        draft_verified_artifacts=draft_verified,
        visible_device=draft_device,
        chunk_count=chunk_count,
    )
    draft_model_load_seconds = time.perf_counter() - draft_load_started
    target: JsonlWorkerClient | None = None
    try:
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
            _assert_worker_pair_owns_gpus(
                draft=draft,
                target=target,
                draft_device=draft_device,
                target_device=target_device,
                phase="T-Stitch generation",
            )
        providers = {
            "initial": DeterministicNoiseProvider(run_dir=run_dir),
            "renoise_750": DeterministicNoiseProvider(
                run_dir=run_dir,
                namespace="main-table-t-stitch-renoise-750-v1",
                directory="noise-t-stitch-750",
            ),
            "renoise_500": DeterministicNoiseProvider(
                run_dir=run_dir,
                namespace="main-table-t-stitch-renoise-500-v1",
                directory="noise-t-stitch-500",
            ),
            "renoise_250": DeterministicNoiseProvider(
                run_dir=run_dir,
                namespace="main-table-t-stitch-renoise-250-v1",
                directory="noise-t-stitch-250",
            ),
        }
        generations: dict[str, TStitchGeneration] = {}
        for prompt in prompts:
            generation = generate_t_stitch_trajectory(
                prompt_id=prompt["id"],
                prompt=prompt["text"],
                draft=draft,
                target=target,
                chunk_count=chunk_count,
                initial_noise_provider=providers["initial"],
                renoise_750_provider=providers["renoise_750"],
                renoise_500_provider=providers["renoise_500"],
                renoise_250_provider=providers["renoise_250"],
                safety_check=(
                    (
                        lambda _stage, _chunk: _assert_worker_pair_owns_gpus(
                            draft=draft,
                            target=target,
                            draft_device=draft_device,
                            target_device=target_device,
                            phase="T-Stitch generation",
                        )
                    )
                    if enforce_gpu_idle
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
        pending_exception = sys.exc_info()[0] is not None
        try:
            if target is not None:
                target.close()
        except BaseException:
            draft.close()
            raise
        if pending_exception:
            draft.close()

    rollouts: dict[str, TStitchRollout] = {}
    try:
        _require(
            tuple(generations) == active_prompt_ids,
            "T-Stitch generation coverage drifted",
        )
        if enforce_gpu_idle:
            _require(
                not _gpu_compute_pids(_gpu_uuid(target_device)),
                "Target GPU retained a compute process after generation shutdown",
            )
            draft_pid = getattr(getattr(draft, "process", None), "pid", None)
            _require(
                isinstance(draft_pid, int)
                and _gpu_compute_pids(_gpu_uuid(draft_device)) <= {draft_pid},
                "Draft GPU acquired an unrelated process before offline decode",
            )
        for prompt_id in active_prompt_ids:
            rollouts[prompt_id] = decode_t_stitch_trajectory(
                generation=generations[prompt_id],
                decoder=draft,
            )
    finally:
        draft.close()

    if enforce_gpu_idle:
        _require(
            not _gpu_compute_pids(_gpu_uuid(draft_device))
            and not _gpu_compute_pids(_gpu_uuid(target_device)),
            "T-Stitch GPU pair is not idle before VBench-Long",
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
                **({"duration": duration_metadata} if duration_metadata else {}),
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
        _write_new_json(
            run_dir / "provenance" / METHOD_ID / f"{prompt_id}.json",
            {
                "schema": METHOD_RESULT_SCHEMA,
                "method_id": METHOD_ID,
                "prompt_id": prompt_id,
                "seed": SEED,
                "generation": dict(rollout.result),
                "model_load_seconds": {
                    "draft": draft_model_load_seconds,
                    "target": target_model_load_seconds,
                },
                metric_key: {
                    "verified": True,
                    "dimension_scores": scores,
                    "macro_mean": measurement.macro_mean,
                    "motion_smoothness": measurement.motion_smoothness,
                    "evidence": evidence_item,
                },
            },
        )

    raw_path = run_dir / "raw" / f"main-table-{METHOD_ID}.json"
    _write_new_json(raw_path, {"schema": RAW_SCHEMA, "rows": rows})
    raw_sha256, raw_nbytes = _sha256_file(raw_path)
    result = {
        "schema": SMOKE_RESULT_SCHEMA if smoke_prompt_id else RESULT_SCHEMA,
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
        "draft_generation_device": draft_device,
        "target_generation_device": target_device,
        "quality_device": target_device,
        "draft_model_load_seconds": draft_model_load_seconds,
        "target_model_load_seconds": target_model_load_seconds,
    }
    if baseline_population_v2:
        result.update(schema="baseline_population_v2_method_result", population=population_binding,
                      benchmark=benchmark, duration=duration_metadata)
        if prompt_shard_count > 1:
            result["run_kind"] = "formal_prompt_shard"
    _write_new_json(run_dir / "result.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cfexp.main_table_t_stitch_campaign"
    )
    parser.add_argument("--t-stitch-artifact-spec", type=Path, required=True)
    parser.add_argument("--draft-artifact-spec", type=Path, required=True)
    parser.add_argument("--target-artifact-spec", type=Path, required=True)
    parser.add_argument("--vbench-checkout", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-asset-manifest", type=Path, required=True)
    parser.add_argument("--vbench-asset-spec", type=Path, required=True)
    parser.add_argument("--smoke-prompt-id")
    parser.add_argument("--baseline-population-v2", action="store_true")
    parser.add_argument("--duration-seconds", type=int, choices=(15,30,60))
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_campaign(
            t_stitch_artifact_spec_path=args.t_stitch_artifact_spec,
            draft_artifact_spec_path=args.draft_artifact_spec,
            target_artifact_spec_path=args.target_artifact_spec,
            vbench_checkout=args.vbench_checkout,
            vbench_python=args.vbench_python,
            vbench_asset_manifest=args.vbench_asset_manifest,
            vbench_asset_spec=args.vbench_asset_spec,
            smoke_prompt_id=args.smoke_prompt_id,
            baseline_population_v2=args.baseline_population_v2,
            duration_seconds=args.duration_seconds,
            prompt_shard_index=args.prompt_shard_index,
            prompt_shard_count=args.prompt_shard_count,
        )
    except (TStitchCampaignError, OSError, subprocess.SubprocessError) as error:
        print(f"main-table-t-stitch-campaign: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHOD_RESULT_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "TStitchCampaignError",
    "run_campaign",
]
