"""Formal one-prompt smoke and 12-prompt SDVG main-table campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import (
    DeterministicNoiseProvider,
    JsonlWorkerClient,
    launch_sdvg_draft_worker,
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
from .main_table_sdvg import (
    CHUNK_COUNT,
    METHOD_ID,
    SEED,
    THRESHOLD,
    SdvgGeneration,
    SdvgRollout,
    decode_sdvg_trajectory,
    generate_sdvg_trajectory,
)
from .main_table_v1 import PROMPT_IDS, QUALITY_DIMENSIONS, load_bundle
from .metadata import collect_git_metadata, find_git_root, utc_now
from .vbench_long import OfficialVBenchLongScorer, VBenchLongVideoRequest
from .vbench_short import OfficialVBenchScorer
from .duration_sweep import duration_profile
from .rolling_campaign import verify_draft_spec as verify_rolling_spec, prepare_launch as prepare_rolling_launch
from .baseline_population import model_pair_for_draft, load_baseline_population, select_prompt_shard


PREFLIGHT_SCHEMA = "main_table_sdvg_campaign_preflight_v1"
METHOD_RESULT_SCHEMA = "main_table_sdvg_prompt_result_v1"


class SdvgCampaignError(RuntimeError):
    """Raised when the formal SDVG campaign cannot prove its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SdvgCampaignError(message)


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
    _require(
        not _gpu_compute_pids(_gpu_uuid(draft_device))
        and not _gpu_compute_pids(_gpu_uuid(target_device)),
        "selected SDVG GPU pair is not idle immediately before launch",
    )


def _assert_worker_pair_owns_gpus(
    *,
    draft: JsonlWorkerClient,
    target: JsonlWorkerClient,
    draft_device: str,
    target_device: str,
    phase: str,
) -> None:
    if draft_device == target_device:
        from types import SimpleNamespace
        from .main_table_ours_campaign import _assert_worker_pair_owns_gpus as assert_shared_workers

        assert_shared_workers(runtime=SimpleNamespace(draft=draft, target=target),
                              draft_device=draft_device, target_device=target_device, phase=phase)
        return
    draft_pid = getattr(getattr(draft, "process", None), "pid", None)
    target_pid = getattr(getattr(target, "process", None), "pid", None)
    _require(
        isinstance(draft_pid, int)
        and isinstance(target_pid, int)
        and _gpu_compute_pids(_gpu_uuid(draft_device)) <= {draft_pid}
        and _gpu_compute_pids(_gpu_uuid(target_device)) <= {target_pid},
        f"an SDVG GPU acquired an unrelated compute process during {phase}",
    )


DraftLauncher = Callable[..., JsonlWorkerClient]
TargetLauncher = Callable[..., JsonlWorkerClient]
ScorerFactory = Callable[..., Any]


def run_campaign(
    *,
    sdvg_artifact_spec_path: Path,
    draft_artifact_spec_path: Path,
    draft_model_id: str = "self_forcing_1.3b",
    target_artifact_spec_path: Path,
    vbench_checkout: Path,
    vbench_python: Path,
    vbench_asset_manifest: Path,
    vbench_asset_spec: Path,
    draft_launcher: DraftLauncher = launch_sdvg_draft_worker,
    target_launcher: TargetLauncher = launch_target_generation_worker,
    scorer_factory: ScorerFactory = OfficialVBenchLongScorer,
    enforce_gpu_idle: bool = True,
    smoke_prompt_id: str | None = None,
    baseline_population_v2: bool = False,
    duration_seconds: int | None = None,
    prompt_shard_index: int = 0,
    prompt_shard_count: int = 1,
    record_video_ready_timing: bool = False,
    overlap_native_vae: bool = False,
) -> dict[str, Any]:
    run_dir = _run_dir()
    model_root = _model_root()
    from .accepted_stream import SINGLE_GPU_TIMING_SCOPE, single_gpu_enabled, stream_devices
    single_gpu = single_gpu_enabled()
    draft_device, target_device = stream_devices() if single_gpu else _visible_devices()
    repository_root = find_git_root(Path.cwd())
    from .native_sdvg import video_ready_protocol
    _require(type(record_video_ready_timing) is bool and type(overlap_native_vae) is bool,
             "native SDVG timing switches must be boolean")
    _require(record_video_ready_timing == overlap_native_vae,
             "native SDVG video-ready timing requires explicit native VAE overlap")
    native_render, timing_scope = video_ready_protocol(draft_model_id, overlap=overlap_native_vae)
    from . import native_single_gpu
    native_single = native_single_gpu.validate(draft_model_id, native_render, enforce_idle=enforce_gpu_idle)
    if single_gpu:
        _require(enforce_gpu_idle and (native_single or native_render == 'self-forcing'),
                 "single-GPU SDVG requires its matched native stream and GPU ownership checks")
        timing_scope = native_single_gpu.TIMING_SCOPE if native_single else SINGLE_GPU_TIMING_SCOPE
    from .paper_population import validate_execution
    validate_execution(single_gpu=single_gpu, seconds=duration_seconds, full_population=baseline_population_v2)
    _require(not native_render or (baseline_population_v2 and duration_seconds in (15, 30, 60)),
             "native SDVG rendering requires the frozen baseline population and duration")
    population_binding = None
    duration_metadata = None
    chunk_count = CHUNK_COUNT
    benchmark = "VBench-Long"
    if baseline_population_v2:
        _require(duration_seconds in (15, 30, 60), "baseline v2 requires a duration")
        profile = duration_profile(duration_seconds)
        chunk_count = profile.chunk_count
        prompts, population_binding = load_baseline_population(repository_root, model_pair=model_pair_for_draft(draft_model_id))
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
                                      shard_count=prompt_shard_count, full_population_count=population_binding.get("full_population_count", 24))
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
        sdvg_spec, sdvg_verified, sdvg_module = verify_main_table_artifact_spec(
            sdvg_artifact_spec_path,
            method_id="sdvg",
            model_root=model_root,
            repository_root=repository_root,
        )
        _require(draft_model_id in {"self_forcing_1.3b", "longlive_1.3b", "rolling_forcing_1.3b"}, "unknown Draft model")
        if draft_model_id == "rolling_forcing_1.3b":
            _draft_spec, draft_verified, draft_module = verify_rolling_spec(
                draft_artifact_spec_path, model_root=model_root, repo_root=repository_root)
        elif draft_model_id == "longlive_1.3b":
            from .main_table_baseline_campaign import verify_artifact_spec as verify_native
            _draft_spec, draft_verified, draft_module = verify_native(
                draft_artifact_spec_path, method_id="draft_only",
                model_root=model_root, repository_root=repository_root)
        else:
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
        raise SdvgCampaignError(f"artifact preflight failed: {error}") from error
    _require(sdvg_module == "cfexp.sdvg_adapter", "wrong SDVG adapter identity")
    _require(
        draft_module == {"longlive_1.3b": "cfexp.longlive_adapter",
                         "self_forcing_1.3b": "cfexp.self_forcing_frame_adapter",
                         "rolling_forcing_1.3b": "cfexp.rolling_policy_worker"}[draft_model_id],
        "wrong Draft adapter identity",
    )
    _require(target_module == "cfexp.krea_chunk_backend", "wrong Target backend identity")
    adapter_config = sdvg_spec.get("adapter_config")
    _require(isinstance(adapter_config, Mapping), "SDVG adapter config is absent")
    routing = adapter_config.get("routing")
    _require(
        isinstance(routing, Mapping)
        and routing.get("threshold") == THRESHOLD
        and routing.get("force_reject_chunk_indices") == [0]
        and routing.get("draft_history_policy") == "always_append_draft_candidate"
        and routing.get("target_history_policy")
        == "append_accepted_draft_else_target_regeneration"
        and routing.get("rejection_noise_policy")
        == "target_regenerates_from_same_initial_noise",
        "SDVG routing config drifted",
    )
    sdvg_spec_sha256, _ = _sha256_file(sdvg_artifact_spec_path.resolve(strict=True))
    draft_spec_sha256, _ = _sha256_file(draft_artifact_spec_path.resolve(strict=True))
    target_spec_sha = target_artifact_spec_sha256(target_artifact_spec_path)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": utc_now(),
        "git": collect_git_metadata(repository_root),
        "method_id": METHOD_ID,
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "prompt_ids": list(active_prompt_ids),
        "seed": SEED,
        "chunk_count": chunk_count,
        "threshold": THRESHOLD,
        "draft_device": draft_device,
        "target_device": target_device,
        "sdvg_artifact_spec_sha256": sdvg_spec_sha256,
        "draft_model_id": draft_model_id,
        "draft_artifact_spec_sha256": draft_spec_sha256,
        "target_artifact_spec_sha256": target_spec_sha,
        "sdvg_verified_artifacts": sdvg_verified,
        "draft_verified_artifacts": draft_verified,
        "target_verified_artifacts": target_verified,
    }
    _write_new_json(run_dir / "provenance" / "preflight.json", preflight)

    if enforce_gpu_idle:
        _assert_idle_pair(draft_device, target_device)
    rolling_kwargs = {}
    if draft_model_id == "rolling_forcing_1.3b":
        rolling_kwargs["rolling"] = prepare_rolling_launch(
            run_dir=run_dir, repo_root=repository_root, model_root=model_root,
            prompts=[{"prompt_id": p["id"], "prompt": p["text"]} for p in prompts],
            seeds=(SEED,), chunk_count=chunk_count)
    draft_extra, target_extra = {}, {}
    if native_render:
        from .correction_runtime import _shared_conditioning_configs
        from .accepted_stream_worker import process_factory
        from .native_sdvg import launch_conditioned_draft_worker, launch_conditioned_target_worker
        if draft_launcher is launch_sdvg_draft_worker:
            draft_launcher = launch_conditioned_draft_worker
        if target_launcher is launch_target_generation_worker:
            target_launcher = launch_conditioned_target_worker
        draft_conditioning, target_conditioning = _shared_conditioning_configs(
            run_dir=run_dir,
            prompts=[dict(prompt_id=p['id'], prompt=p['text']) for p in prompts],
            draft_spec=_draft_spec, target_spec=_target_spec,
            draft_verified_artifacts=draft_verified, target_verified_artifacts=target_verified)
        draft_extra.update(conditioning_cache=draft_conditioning, process_factory=process_factory)
        target_extra.update(conditioning_cache=target_conditioning, process_factory=process_factory)
    draft_load_started = time.perf_counter()
    draft = draft_launcher(
        run_dir=run_dir,
        repo_root=repository_root,
        draft_artifact_spec_path=draft_artifact_spec_path,
        draft_model_id=draft_model_id,
        draft_artifact_spec_sha256=draft_spec_sha256,
        draft_verified_artifacts=draft_verified,
        sdvg_artifact_spec_path=sdvg_artifact_spec_path,
        sdvg_artifact_spec_sha256=sdvg_spec_sha256,
        sdvg_verified_artifacts=sdvg_verified,
        visible_device=draft_device,
        chunk_count=chunk_count,
        **rolling_kwargs,
        **draft_extra,
    )
    draft_model_load_seconds = time.perf_counter() - draft_load_started
    target: JsonlWorkerClient | None = None
    generations: dict[str, SdvgGeneration] = {}
    rollouts: dict[str, SdvgRollout] = {}
    ready_seconds, render_bindings, native_audits = {}, {}, {}
    renderer = None
    try:
        target_load_started = time.perf_counter()
        target = target_launcher(
            run_dir=run_dir,
            repo_root=repository_root,
            target_artifact_spec_path=target_artifact_spec_path,
            target_artifact_spec_sha256=target_spec_sha,
            target_verified_artifacts=target_verified,
            visible_device=target_device,
            chunk_count=chunk_count,
            **target_extra,
        )
        target_model_load_seconds = time.perf_counter() - target_load_started
        if enforce_gpu_idle:
            _assert_worker_pair_owns_gpus(
                draft=draft,
                target=target,
                draft_device=draft_device,
                target_device=target_device,
                phase="SDVG generation",
            )
        noise = DeterministicNoiseProvider(run_dir=run_dir)
        if native_render:
            from .accepted_stream_client import StreamDecoderProxy
            renderer = StreamDecoderProxy(draft, target, run_dir, chunk_count=chunk_count)
        for prompt in prompts:
            ready_started = time.perf_counter()
            generation = generate_sdvg_trajectory(
                prompt_id=prompt["id"],
                prompt=prompt["text"],
                draft=draft,
                target=target,
                noise_provider=noise,
                chunk_count=chunk_count,
                draft_model_id=draft_model_id,
                **(dict(on_commit=renderer.append,
                        before_generate=None if single_gpu else renderer.before_generate)
                   if renderer is not None else {}),
                safety_check=(
                    (
                        lambda _stage, _chunk: _assert_worker_pair_owns_gpus(
                            draft=draft,
                            target=target,
                            draft_device=draft_device,
                            target_device=target_device,
                            phase="SDVG generation",
                        )
                    )
                    if enforce_gpu_idle
                    else None
                ),
            )
            if renderer is not None:
                from .erank_acceptance_pilot import bound_file
                rollouts[prompt['id']] = decode_sdvg_trajectory(
                    generation=generation, decoder=renderer, video_only=True)
                ready_seconds[prompt['id']] = _finite_positive(
                    time.perf_counter() - ready_started, 'SDVG video-ready wall time')
                receipt = renderer.last_evidence
                _require(receipt is not None and receipt['decoder_and_encoder_joined'] is True
                    and receipt['history_manifest_sha256'] == generation.selected_history.manifest_sha256
                    and receipt['video']['sha256'] == rollouts[prompt['id']].decode.video['sha256'],
                    'SDVG renderer did not finish the selected history')
                render_bindings[prompt['id']] = bound_file(run_dir, run_dir/'provenance'/
                    'accepted-stream'/(generation.selected_history.manifest_sha256+'.json'))
                from .native_sdvg import audit_native_draft_history
                native_audits[prompt['id']] = audit_native_draft_history(generation.result)
            history_root = run_dir / "provenance" / METHOD_ID
            history_root.mkdir(parents=True, exist_ok=True)
            write_history_manifest(
                history_root / f"{prompt['id']}-draft-history.json",
                generation.draft_history,
            )
            write_history_manifest(
                history_root / f"{prompt['id']}-selected-history.json",
                generation.selected_history,
            )
            # Persist latency and routing before decoding/scoring can fail. These
            # records are sealed with the run and never claim completed quality.
            _write_new_json(
                history_root / f"{prompt['id']}-generation.json",
                {"schema": "sdvg_generation_checkpoint_v1",
                 "method_id": METHOD_ID, "prompt_id": prompt["id"],
                 "seed": SEED, "duration": duration_metadata,
                 "quality_evaluated": False, "generation": dict(generation.result)},
            )
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

    try:
        _require(tuple(generations) == active_prompt_ids, "SDVG generation coverage drifted")
        if enforce_gpu_idle:
            expected_pids = {draft.process.pid} if single_gpu else set()
            _require(
                _gpu_compute_pids(_gpu_uuid(target_device)) <= expected_pids,
                "Unexpected GPU process after the SDVG Target worker closed",
            )
        for prompt_id in active_prompt_ids:
            if renderer is None:
                rollouts[prompt_id] = decode_sdvg_trajectory(
                    generation=generations[prompt_id], decoder=draft)
            _write_new_json(
                run_dir / "provenance" / METHOD_ID / f"{prompt_id}-decoded.json",
                {"schema": "sdvg_decoded_checkpoint_v1",
                 "method_id": METHOD_ID, "prompt_id": prompt_id,
                 "seed": SEED, "duration": duration_metadata,
                 "quality_evaluated": False, "generation": dict(rollouts[prompt_id].result)},
            )
    finally:
        draft.close()

    if enforce_gpu_idle:
        _require(
            not _gpu_compute_pids(_gpu_uuid(draft_device))
            and not _gpu_compute_pids(_gpu_uuid(target_device)),
            "SDVG GPU pair is not idle before VBench-Long",
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
        measurement = measurements[f"mt-{METHOD_ID}-{prompt_id}"]
        scores = dict(measurement.dimension_scores)
        _require(tuple(scores) == QUALITY_DIMENSIONS, "VBench dimension order drifted")
        generation_seconds = _finite_positive(
            rollout.result["timing"]["generation_wall_seconds"],
            "SDVG generation wall time",
        )
        rows.append(
            {
                "method_id": METHOD_ID,
                "prompt_id": prompt_id,
                "seed": SEED,
                "generation_wall_seconds": generation_seconds,
                "quality_scores": scores,
                **(dict(video_ready_wall_seconds=ready_seconds[prompt_id],
                        timing_scope=timing_scope,
                        accepted_stream_evidence=render_bindings[prompt_id],
                        native_cache_audit=native_audits[prompt_id]) if native_render else {}),
                metric_key + "_verified": True,
                **({"duration": duration_metadata} if duration_metadata else {}),
            }
        )
        evidence_path = (run_dir / measurement.evidence_path).resolve(strict=True)
        evidence_sha, evidence_nbytes = _sha256_file(evidence_path)
        evidence_item = {
            "prompt_id": prompt_id,
            "measurement_id": f"mt-{METHOD_ID}-{prompt_id}",
            "path": evidence_path.relative_to(run_dir).as_posix(),
            "sha256": evidence_sha,
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
                **(dict(video_ready_wall_seconds=ready_seconds[prompt_id],
                        timing_scope=timing_scope,
                        accepted_stream_evidence=render_bindings[prompt_id],
                        native_cache_audit=native_audits[prompt_id]) if native_render else {}),
                "model_load_seconds": {
                    "draft_and_image_reward": draft_model_load_seconds,
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
    raw_sha, raw_nbytes = _sha256_file(raw_path)
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
            "sha256": raw_sha,
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
    if native_render:
        from .stage_timing_probe import _hardware
        result.update(timing_scope=timing_scope,
            code_revision=collect_git_metadata(repository_root)['sha'],
            hardware=_hardware((draft_device,) if single_gpu else (draft_device, target_device)),
            native_model=native_render,
            draft_history_policy='always_append_draft_candidate',
            selected_history_policy='append_accepted_draft_else_target_regeneration',
            formal_speedup_claim=False)
    _write_new_json(run_dir / "result.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.main_table_sdvg_campaign")
    parser.add_argument("--sdvg-artifact-spec", type=Path, required=True)
    parser.add_argument("--draft-artifact-spec", type=Path, required=True)
    parser.add_argument("--draft-model-id", choices=("self_forcing_1.3b", "longlive_1.3b", "rolling_forcing_1.3b"), default="self_forcing_1.3b")
    parser.add_argument("--target-artifact-spec", type=Path, required=True)
    parser.add_argument("--vbench-checkout", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-asset-manifest", type=Path, required=True)
    parser.add_argument("--vbench-asset-spec", type=Path, required=True)
    parser.add_argument("--smoke-prompt-id")
    parser.add_argument("--baseline-population-v2", action="store_true")
    parser.add_argument("--duration-seconds", type=int, choices=(15, 30, 60))
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    parser.add_argument("--record-video-ready-timing", action="store_true")
    parser.add_argument("--overlap-native-vae", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_campaign(
            sdvg_artifact_spec_path=args.sdvg_artifact_spec,
            draft_artifact_spec_path=args.draft_artifact_spec,
            draft_model_id=args.draft_model_id,
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
            record_video_ready_timing=args.record_video_ready_timing,
            overlap_native_vae=args.overlap_native_vae,
        )
    except (SdvgCampaignError, OSError, subprocess.SubprocessError) as error:
        print(f"main-table-sdvg-campaign: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["METHOD_RESULT_SCHEMA", "PREFLIGHT_SCHEMA", "SdvgCampaignError", "run_campaign"]
