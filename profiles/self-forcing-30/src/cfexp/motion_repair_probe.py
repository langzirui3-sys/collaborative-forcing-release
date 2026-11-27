"""Development-only decoded-video acceptance ablation on frozen C10 prompts.

Run with the formal remote wrapper. This is algorithm development evidence,
not a held-out evaluation or a Target speed/quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

from . import optimized_duration_control as control
from .branch_gate_probe import (_calibrator_from_restored, collect_trajectory,
                                feature_environment)
from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import VBENCH_LONG_PRIMARY_QUALITY_METRIC, launch_correction_runtime
from .duration_sweep import duration_profile
from .erank_frame_campaign import verify_artifact_spec as verify_draft_artifact_spec
from .insight2_campaign import Insight2Campaign, PromptCase
from .insight2_protocol import load_protocol
from .insight2_run import _calibration_receipt_binding, load_run_config
from .main_table_baseline_campaign import (_model_root, _run_dir, _sha256_file,
                                          _visible_devices, _write_new_json)
from .main_table_ours_campaign import (BOUNDARY_CONFIG, MAXIMUM_GAP_CHUNKS,
    MINIMUM_GAP_CHUNKS, WINDOW_RADIUS_CHUNKS, _assert_idle_pair,
    _assert_worker_pair_owns_gpus, _receipt_policy)
from .main_table_v1 import QUALITY_DIMENSIONS
from .metadata import collect_git_metadata, find_git_root, utc_now
from .runtime_helpers import shutdown_owned_runtime_helpers
from .vbench_long import OfficialVBenchLongScorer, VBenchLongVideoRequest

PROTOCOL_ID = "motion-acceptance-development-v1"
SEED = 20260911
ARMS = ("draft_only", "paired_boundary", "rank_only", "proposal_always")
PROMPT_IDS = ("c002_vbench2long_0247", "c004_vbench2long_0283",
              "c009_vbench2long_0206", "c008_vbench2long_0052")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def bound_file(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_relative_to(root.resolve()) and not path.is_symlink(),
             "diagnostic evidence escapes run directory")
    sha, size = _sha256_file(resolved)
    return {"path": resolved.relative_to(root).as_posix(), "sha256": sha, "nbytes": size}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(r["arm"], r["prompt_id"]): r for r in rows}
    expected = {(a, p) for a in ARMS for p in PROMPT_IDS}
    _require(len(rows) == len(indexed) == len(expected) and set(indexed) == expected,
             "missing or duplicate motion diagnostic cells")
    for row in rows:
        _require(math.isfinite(row["generation_wall_seconds"]) and row["generation_wall_seconds"] > 0,
                 "invalid generation timing")
        scores = row["quality_scores"]
        _require(set(scores) == set(QUALITY_DIMENSIONS) and all(
            type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in scores.values()),
            "invalid or incomplete VBench quality")
    # Keep static-control motion separate: higher motion there is not a success.
    return {arm: {group: {dimension: sum(indexed[arm, p]["quality_scores"][dimension]
                     for p in prompts) / len(prompts) for dimension in QUALITY_DIMENSIONS}
                 for group, prompts in (("action_prompts", PROMPT_IDS[:3]),
                                        ("static_control", PROMPT_IDS[3:]))}
            for arm in ARMS}


def run_probe(*, draft_artifact_spec_path: Path, target_artifact_spec_path: Path,
              vbench_checkout: Path, vbench_python: Path, vbench_asset_manifest: Path,
              vbench_asset_spec: Path, boundary_config_path: Path = BOUNDARY_CONFIG,
              runtime_launcher: Callable[..., Any] = launch_correction_runtime,
              receipt_policy: Callable[..., Any] = _receipt_policy,
              scorer_factory: Callable[..., Any] = OfficialVBenchLongScorer,
              enforce_gpu_idle: bool = True) -> dict[str, Any]:
    run_dir, model_root = _run_dir(), _model_root()
    _require(not (run_dir / "cells").exists() and not (run_dir / "result.json").exists(),
             "motion diagnostic requires a fresh attempt")
    _require(all(not (parent / name).exists() for parent in (run_dir, *run_dir.parents)
                 for name in ("_SUCCESS.json", "_FAILED.json", "artifact-index.json")),
             "refusing to write inside a sealed run")
    _require(os.environ.get("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SHA256") == control.CALIBRATION_RECEIPT_SHA256,
             "motion diagnostic requires the pinned C10 receipt")
    devices, profile = _visible_devices(), duration_profile(60)
    duration = {**profile.metadata(), "experiment_id": PROTOCOL_ID,
                "calibration_policy": "explicit_historical_C10_receipt"}
    repository = find_git_root(Path.cwd())
    config = load_run_config(boundary_config_path, repo_root=repository)
    protocol = load_protocol(config.protocol_path)
    original_cal = {str(p["prompt_id"]): p for p in protocol.calibration_prompts}
    _require(set(PROMPT_IDS).issubset(original_cal), "development prompts missing from C10")
    prompts = tuple(PromptCase(pid, str(original_cal[pid]["prompt"]), "evaluation") for pid in PROMPT_IDS)
    remaining = tuple(PromptCase(pid, str(p["prompt"]), "calibration")
                      for pid, p in original_cal.items() if pid not in PROMPT_IDS)
    # 'evaluation' is the campaign's trajectory namespace only. The population
    # remains calibration/development; the historical CDF still uses all C10.
    original_eval = tuple(PromptCase(str(p["prompt_id"]), str(p["prompt"]), "evaluation")
                          for p in protocol.eval_prompts)
    _, draft_verified, draft_module = verify_draft_artifact_spec(draft_artifact_spec_path,
        model_root=model_root, repo_root=repository)
    _, target_verified, target_module = verify_target_artifact_spec(target_artifact_spec_path,
        model_root=model_root, repo_root=repository)
    _require((draft_module, target_module) == ("cfexp.self_forcing_frame_adapter", "cfexp.krea_chunk_backend"),
             "diagnostic backend drift")
    draft_sha, _ = _sha256_file(draft_artifact_spec_path.resolve(strict=True))
    target_sha = target_artifact_spec_sha256(target_artifact_spec_path)
    protocol_sha, _ = _sha256_file(config.protocol_path)
    current = _calibration_receipt_binding(repo_root=repository, config=config, protocol=protocol,
        protocol_sha256=protocol_sha, draft_spec_sha256=draft_sha, target_spec_sha256=target_sha,
        draft_verified_artifacts=draft_verified, target_verified_artifacts=target_verified, phase_seeds=(SEED,))
    binding = control.frozen_control_binding(current)
    calibration_dir = run_dir / "calibration-control"
    calibration_dir.mkdir(parents=True, exist_ok=False)
    sealed, receipt = receipt_policy(smoke_prompt_id=None, run_dir=calibration_dir, binding=binding)
    _require(sealed is False and receipt is not None, "verified historical C10 is required")
    preflight = {"schema": "motion_acceptance_preflight_v1", "created_at": utc_now(),
        "git": collect_git_metadata(repository), "code_revision": current["code_revision"],
        "calibration_code_revision": binding["code_revision"], "calibration_receipt": receipt.public_binding(),
        "population": "C10_development_not_held_out", "formal_ours_result": False,
        "prompt_ids": list(PROMPT_IDS), "prompts": [{"id": p.prompt_id, "text": p.prompt} for p in prompts],
        "arms": list(ARMS), "seed": SEED, "duration": duration,
        "proposal_budget": profile.proposal_budget, "worker_reuse": "within_arm_only",
        "generation_timer": "materialization_only_excludes_load_decode_quality_and_publication",
        "conditioning_policy": "precomputed_before_generation_timer",
        "candidate_selection": "same coverage rule; realized positions may diverge after branch changes"}
    _write_new_json(run_dir / "provenance/preflight.json", preflight)
    cells, requests, calibrator = [], [], None
    with feature_environment("fork_only") as features:
        for arm in ARMS:
            directory = run_dir / "cells" / arm
            directory.mkdir(parents=True, exist_ok=False)
            if enforce_gpu_idle:
                _assert_idle_pair(*devices)
            started = time.perf_counter()
            runtime = runtime_launcher(run_dir=directory, repo_root=repository,
                draft_artifact_spec_path=draft_artifact_spec_path, draft_artifact_spec_sha256=draft_sha,
                draft_verified_artifacts=draft_verified, target_artifact_spec_path=target_artifact_spec_path,
                target_artifact_spec_sha256=target_sha, target_verified_artifacts=target_verified,
                devices=devices, write_full_videos=True, chunk_count=profile.chunk_count,
                conditioning_prompts=[{"prompt_id": p.prompt_id, "prompt": p.prompt} for p in prompts],
                shared_precomputed_conditioning=True)
            load_seconds = time.perf_counter() - started
            try:
                runtime.evaluator.full_quality_metric_id = VBENCH_LONG_PRIMARY_QUALITY_METRIC
                if calibrator is None:
                    calibrator = _calibrator_from_restored(directory=calibration_dir, runtime=runtime,
                        protocol=protocol, evaluation=original_eval)
                campaign = Insight2Campaign(run_dir=directory, protocol_id=PROTOCOL_ID, seeds=(SEED,),
                    chunk_count=profile.chunk_count, calibration_prompts=remaining, evaluation_prompts=prompts,
                    draft=runtime.draft, target=runtime.target, noise_provider=runtime.noise_provider,
                    evaluator=runtime.evaluator)
                for prompt in prompts:
                    if enforce_gpu_idle:
                        _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                            target_device=devices[1], phase=f"motion:{arm}:{prompt.prompt_id}")
                    event_id = hashlib.sha256(f"{PROTOCOL_ID}:{arm}:{prompt.prompt_id}".encode()).hexdigest()
                    started = time.perf_counter()
                    with campaign._target_rpc_scope(event_id):
                        if arm == "draft_only":
                            key, route, accepted, history = campaign._materialize_fixed_positions(
                                prompt=prompt, seed=SEED, corrected_chunks=())
                            proposed = ()
                        else:
                            key, route, accepted, proposed, history = campaign._materialize_online_lookahead_coverage_boundary_gated(
                                prompt=prompt, seed=SEED, budget=profile.proposal_budget, calibrator=calibrator,
                                window_radius_chunks=WINDOW_RADIUS_CHUNKS, minimum_gap_chunks=MINIMUM_GAP_CHUNKS,
                                maximum_gap_chunks=MAXIMUM_GAP_CHUNKS, acceptance_rule=arm)
                    elapsed = time.perf_counter() - started
                    trajectory = collect_trajectory(campaign, key=key, history=history, accepted=accepted, proposed=proposed)
                    pending = campaign._prepare_rollout(key=key, route_id=route, corrected_chunks=accepted,
                        proposed_target_chunks=proposed, history=history)
                    prepared = pending.prepared_quality
                    video = bound_file(run_dir, directory / prepared.video_path)
                    _require(video["sha256"] == prepared.video_sha256 and video["nbytes"] == prepared.video_nbytes,
                             "decoded video differs from prepared evidence")
                    decode = bound_file(run_dir, directory / prepared.decode_evidence_path)
                    _require(decode["sha256"] == prepared.decode_evidence_sha256, "decode evidence changed")
                    _require(prepared.frame_count == profile.frame_count and prepared.fps == 24,
                             "decoded horizon changed")
                    measurement_id = f"motion-{arm}-{prompt.prompt_id}".replace("_", "-")
                    cell = {"arm": arm, "prompt_id": prompt.prompt_id, "seed": SEED, "measurement_id": measurement_id,
                        "features": dict(features), "generation_wall_seconds": elapsed, "trajectory": trajectory,
                        "video": video, "decode_evidence": decode,
                        "model_load_wall_seconds_per_arm": load_seconds}
                    _write_new_json(directory / "provenance" / f"{prompt.prompt_id}.json", cell)
                    cells.append(cell)
                    requests.append(VBenchLongVideoRequest(video_path=run_dir / video["path"],
                        prompt=prompt.prompt, measurement_id=measurement_id))
            finally:
                runtime.close()
    if enforce_gpu_idle:
        _assert_idle_pair(*devices)
    scorer = scorer_factory(run_dir=run_dir, checkout=vbench_checkout, python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest, asset_spec=vbench_asset_spec, visible_device=devices[1])
    started = time.perf_counter()
    measurements = scorer.score_videos(requests)
    vbench_seconds = time.perf_counter() - started
    _require(set(measurements) == {r.measurement_id for r in requests}, "incomplete VBench coverage")
    rows = []
    for cell in cells:
        measurement = measurements[cell["measurement_id"]]
        evidence = bound_file(run_dir, run_dir / measurement.evidence_path)
        # Score requests, prepared evidence and final rows bind the same bytes.
        _require(bound_file(run_dir, run_dir / cell["video"]["path"]) == cell["video"],
                 "video changed during scoring")
        rows.append({**cell, "quality_scores": dict(measurement.dimension_scores), "quality_evidence": evidence})
    result = {"schema": "motion_acceptance_development_v1", "status": "succeeded",
        "formal_ours_result": False, "population": preflight["population"], "code_revision": current["code_revision"],
        "calibration_receipt": preflight["calibration_receipt"], "duration": duration,
        "row_count": len(rows), "rows": rows, "quality_summary": summarize(rows),
        "vbench_wall_seconds": vbench_seconds, "automatic_winner_selection": False}
    _write_new_json(run_dir / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft-artifact-spec", "target-artifact-spec", "vbench-checkout", "vbench-python",
                 "vbench-asset-manifest", "vbench-asset-spec"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--boundary-config", type=Path, default=BOUNDARY_CONFIG)
    args = parser.parse_args(argv)
    try:
        try:
            result = run_probe(draft_artifact_spec_path=args.draft_artifact_spec,
                target_artifact_spec_path=args.target_artifact_spec, boundary_config_path=args.boundary_config,
                vbench_checkout=args.vbench_checkout, vbench_python=args.vbench_python,
                vbench_asset_manifest=args.vbench_asset_manifest, vbench_asset_spec=args.vbench_asset_spec)
        finally:
            shutdown_owned_runtime_helpers(label="motion-acceptance-controller")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"motion-repair-probe: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
