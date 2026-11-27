"""Explicit eRank acceptance diagnostics on C10 development or main24 prompts.

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

from .branch_gate_probe import (_calibrator_from_restored, collect_trajectory,
                                feature_environment, calibration_control_binding)
from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import launch_correction_runtime
from .krea_chunk_backend import validate_native_correction_state
from .duration_sweep import duration_profile
from .correction_policy import random_positions, uniform_positions
from .erank_frame_campaign import verify_artifact_spec as verify_draft_artifact_spec
from .insight2_campaign import Insight2Campaign, PromptCase
from .insight2_protocol import load_protocol
from .insight2_run import _calibration_receipt_binding, load_run_config
from .main_table_baseline_campaign import (_model_root, _run_dir, _sha256_file,
                                          _visible_devices, _write_new_json)
from .main_table_ours_campaign import (BOUNDARY_CONFIG, MAXIMUM_GAP_CHUNKS,
    MINIMUM_GAP_CHUNKS, WINDOW_RADIUS_CHUNKS, _assert_idle_pair,
    _assert_worker_pair_owns_gpus, _read_sealed_calibration, _receipt_policy)
from .main_table_v1 import QUALITY_DIMENSIONS
from .metadata import collect_git_metadata, find_git_root, utc_now
from .runtime_helpers import shutdown_owned_runtime_helpers
from .vbench_long import VBenchLongVideoRequest
from .vbench_short import OfficialVBenchScorer
from .baseline_population import load_baseline_population
from .stage_timing_probe import _hardware

PROTOCOL_ID = "erank-acceptance-pilot-30s-v1"
from .paper_population import generation_seed, CALIBRATION_SEED
SEED = generation_seed()
ARMS = ("paired_boundary", "proposal_always", "random", "uniform")
MAIN24_ARMS = ("paired_boundary", "rank_only", "proposal_always", "random")
P8_TRANSFER_ARM = "rank_burst4_p8_k4"
PAPER30_ARM = "rank_burst4_opening_p2_positive_k36"
SPLIT_TRANSFER_ARMS = {30: "rank_burst2_p8_k6", 60: "rank_burst2_p8_k8"}
NATIVE_B4_TRANSFER_ARMS = {30: "rank_burst4_p8_k8"}
RECOVERY_30_ARMS = ("rank_burst2_p8_k8", "rank_burst4_p8_k12", "rank_burst4_p8_k16")
OPENING_SPLIT_TRANSFER_ARMS = {30: "rank_burst2_opening_p8_k6", 60: "rank_burst2_opening_p8_k8"}
TAIL_SPLIT_TRANSFER_ARMS = {30: "rank_burst2_tail_p8_k6", 60: "rank_burst2_tail_p8_k8"}
SPLIT_ARMS = (*SPLIT_TRANSFER_ARMS.values(), *NATIVE_B4_TRANSFER_ARMS.values(), *RECOVERY_30_ARMS, *OPENING_SPLIT_TRANSFER_ARMS.values(),
              *TAIL_SPLIT_TRANSFER_ARMS.values(), PAPER30_ARM)
BURST_ARMS = ("rank_burst2_k8", "rank_burst2_k10", "rank_burst2_k12", "rank_burst2_k14",
              "rank_burst1_k4", "rank_burst1_k8", "rank_burst2_k4",
              "rank_burst4_k4", "rank_burst4_k8",
              "rank_burst3_k3", "rank_burst5_k5", "rank_burst6_k6",
              P8_TRANSFER_ARM, "rank_burst4_p12_k4")


def split_transfer_options(arm, seconds):
    if arm is None:
        return {}
    if seconds == 30 and arm == PAPER30_ARM:
        return {"split_target_burst": True, "opening_target_burst": True,
                "require_positive_drop": True}
    if (arm == SPLIT_TRANSFER_ARMS.get(seconds) or arm == NATIVE_B4_TRANSFER_ARMS.get(seconds)
            or (seconds == 30 and arm in RECOVERY_30_ARMS)):
        return {"split_target_burst": True}
    if arm == OPENING_SPLIT_TRANSFER_ARMS.get(seconds):
        return {"split_target_burst": True, "opening_target_burst": True}
    if arm == TAIL_SPLIT_TRANSFER_ARMS.get(seconds):
        return {"split_target_burst": True, "final_window_at_tail": True, "tail_draft_reserve": 4}
    return {}


def observation_horizon_for_arm(arm):
    if arm == PAPER30_ARM:
        return 2
    if arm in SPLIT_ARMS:
        return 8
    if arm not in BURST_ARMS:
        raise ValueError("unknown frozen burst arm")
    return {"rank_burst4_p8_k4": 8, "rank_burst4_p12_k4": 12}.get(arm)

def burst_length_for_arm(arm):
    if arm not in (*BURST_ARMS, *SPLIT_ARMS):
        raise ValueError("unknown frozen burst arm")
    return int(arm.split("_")[1][len("burst"):])

PREFIX_ARMS = ("prefix_rank_k14", "prefix_all_k29")
EARLY_ARMS = ("early_current_k8", "early_current_k14")
DIRECT_ARMS = ("causal_direct_k2", "causal_direct_k4", "causal_direct_k14")
BLOCK_ARMS = ("paired_direct_k8", "random_direct_k8", "random_direct_k4", "target_prefix_k4", "uniform_direct_k4", "uniform_direct_k14", "paired_direct_k14", "random_direct_k14")
FIXED_BURSTS = {
    "fixed_deadline8_k4": (9,10,11,12),
    "fixed_burst1_k4": (5,10,15,20),
    "fixed_burst2_k4": (5,6,15,16),
    "fixed_burst4_k4": (5,6,7,8),
    "fixed_burst1_k8": (3,6,9,12,15,18,21,24),
    "fixed_burst2_k8": (5,6,12,13,19,20,26,27),
    "fixed_burst4_k8": (5,6,7,8,19,20,21,22),
}
BLOCK_ARMS += tuple(FIXED_BURSTS) + ("target_root_only_k0",)
RANDOM_P8_B4 = "random_burst4_p8_k4"
RANDOM_P8_B4_ARMS = tuple(f"random_burst4_p8_r{repeat}_k4" for repeat in range(3))
RANDOM_P8_B4_PROTOCOL = "sf-selector-random-p8-b4-v1"
RANDOM_P8_B4_HORIZON = 8
COHORTS = ("development30", "main24_15s", "main12_direct_15s", "main12_block_15s", "main12_burst_15s", "main12_early_15s", "main12_prefix_15s", "main24_burst")


def block_positions(arm: str, *, chunk_count: int = 30, prompt_id: str | None = None) -> tuple[int, ...]:
    """Frozen quality controls: equal Target budgets, different temporal grouping."""
    _require(arm in (*BLOCK_ARMS, *RANDOM_P8_B4_ARMS) and chunk_count == 30,
             "unknown block-control schedule")
    if arm in RANDOM_P8_B4_ARMS:
        _require(isinstance(prompt_id, str) and bool(prompt_id), "Random P8 requires prompt identity")
        observation, = random_positions(protocol_id=RANDOM_P8_B4_PROTOCOL,
            prompt_id=prompt_id, generation_seed=SEED, budget=1,
            repeat_id=RANDOM_P8_B4_ARMS.index(arm), chunk_count=1 + RANDOM_P8_B4_HORIZON)
        return tuple(range(observation + 1, observation + 5))
    if arm == "target_root_only_k0":
        return ()
    if arm in FIXED_BURSTS:
        return FIXED_BURSTS[arm]
    if arm == "target_prefix_k4":
        return (1, 2, 3, 4)
    if arm == "paired_direct_k8":
        return (5, 6, 12, 13, 19, 20, 26, 27)
    if arm in {"random_direct_k4", "random_direct_k8"}:
        _require(isinstance(prompt_id, str) and bool(prompt_id), "random control requires prompt identity")
        return random_positions(protocol_id="committed-burst-budget-matched-random-v1",
            prompt_id=prompt_id, generation_seed=SEED, budget=int(arm.rsplit('_k',1)[1]), repeat_id=0, chunk_count=chunk_count)
    if arm == "paired_direct_k14":
        return (3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28)
    if arm == "random_direct_k14":
        _require(isinstance(prompt_id, str) and bool(prompt_id), "random control requires prompt identity")
        return random_positions(protocol_id="quality-first-half-target-random-v1",
            prompt_id=prompt_id, generation_seed=SEED, budget=14, repeat_id=0, chunk_count=chunk_count)
    return uniform_positions(int(arm.rsplit("_k", 1)[1]), chunk_count=chunk_count)
READY_TIMING_SCOPE = "warm_cached_conditioning_generation_routing_state_sync_decode_encode_v1"
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
    return {"path": resolved.relative_to(root.resolve()).as_posix(), "sha256": sha, "nbytes": size}


def verify_native_cache(campaign: Insight2Campaign, key: str, trajectory: dict[str, Any]) -> dict[str, Any]:
    """Require measured reuse, including persistent Target correction sessions."""
    _require(trajectory["counters"]["reused_chunks"] > 0,
             "Draft reported no cross-chunk KV reuse")
    targets = {}
    for index in range(campaign.chunk_count):
        item = json.loads(campaign._checkpoint_path("rollouts", key, index).read_text())
        for value in (item["generated"], item["metadata"].get("target_candidate")):
            if value is None or value["context"]["role"] != "target":
                continue
            state = value.get("diagnostics", {}).get("native_correction_state")
            validate_native_correction_state(state, chunk_index=index)
            targets[index] = state
    _require(len(targets) == trajectory["counters"]["physical_calls"]["target"],
             "missing physical Target native-state evidence")
    _require(len({v["native_session_id"] for v in targets.values()}) == 1
             and all(not v["session_recreated"] for k, v in targets.items() if k),
             "Target correction session was recreated between chunks")
    return {"draft_reused_chunks": trajectory["counters"]["reused_chunks"],
            "target_native_states": targets,
            "target_bounded_context_rebases": sum(v["bounded_context_rebase"] for v in targets.values())}


def collect_root_only_ranks(campaign, key, trajectory):
    """Expose already measured ranks; never execute a diagnostic model forward."""
    from .self_forcing_frame_adapter import validate_effective_rank_evidence
    _require(len(trajectory['decisions']) == 29, 'root-only rank horizon differs')
    for row in trajectory['decisions']:
        index=row['chunk_index']
        item=json.loads(campaign._checkpoint_path('rollouts',key,index).read_text())
        value=item['generated'];rank=value['effective_rank']
        _require(value['context']['role']=='draft' and rank==item['metadata']['draft_rank'],
                 'root-only observation differs from committed Draft')
        validate_effective_rank_evidence(value['diagnostics']['rank_evidence'],
            expected_rank=rank,expected_eigenvalue_count=256)
        row['rank']=rank
        row['initial_noise_tensor_sha256']=value['context']['noise']['tensor_sha256']


def summarize(rows: Sequence[dict[str, Any]], arms: Sequence[str] = ARMS,
              prompt_ids: Sequence[str] = PROMPT_IDS, *, main24: bool = False) -> dict[str, Any]:
    indexed = {(r["arm"], r["prompt_id"]): r for r in rows}
    expected = {(a, p) for a in arms for p in prompt_ids}
    _require(len(rows) == len(indexed) == len(expected) and set(indexed) == expected,
             "missing or duplicate acceptance pilot cells")
    for row in rows:
        _require(math.isfinite(row["generation_wall_seconds"]) and row["generation_wall_seconds"] > 0,
                 "invalid generation timing")
        scores = row["quality_scores"]
        _require(set(scores) == set(QUALITY_DIMENSIONS) and all(
            type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in scores.values()),
            "invalid or incomplete VBench quality")
    # Keep static-control motion separate: higher motion there is not a success.
    groups = (("main24", prompt_ids),) if main24 else (
        ("action_prompts", PROMPT_IDS[:3]), ("static_control", PROMPT_IDS[3:]))
    return {arm: {group: {dimension: sum(indexed[arm, p]["quality_scores"][dimension]
                     for p in prompts) / len(prompts) for dimension in QUALITY_DIMENSIONS}
                 for group, prompts in groups}
            for arm in arms}


def run_probe(*, draft_artifact_spec_path: Path, target_artifact_spec_path: Path,
              vbench_checkout: Path, vbench_python: Path, vbench_asset_manifest: Path,
              vbench_asset_spec: Path, boundary_config_path: Path = BOUNDARY_CONFIG,
              runtime_launcher: Callable[..., Any] = launch_correction_runtime,
              receipt_policy: Callable[..., Any] = _receipt_policy,
              scorer_factory: Callable[..., Any] = OfficialVBenchScorer,
              enforce_gpu_idle: bool = True, arm: str | None = None,
              cohort: str = "development30", calibration_code_revision: str | None = None,
              prefix_gate_run: Path | None = None,
              duration_seconds: int | None = None, smoke_prompt_id: str | None = None,
              defer_video_delivery: bool = False,
              sealed_calibration_ranks: bool = False,
              development_c10: bool = False) -> dict[str, Any]:
    _require(cohort in COHORTS, "unknown acceptance cohort")
    _require(type(development_c10) is bool, "invalid C10 development switch")
    if development_c10:
        from .native_c10_development import validate_options
        validate_options(cohort=cohort, arm=arm, seconds=duration_seconds,
                         sealed_ranks=sealed_calibration_ranks, smoke=smoke_prompt_id)
    from .stage_timing import enabled as timing_enabled
    diagnostic_timing = timing_enabled()
    prefix = cohort == "main12_prefix_15s"
    early = cohort == "main12_early_15s"
    transfer = cohort == "main24_burst"
    split_options = split_transfer_options(arm, duration_seconds) if transfer else {}
    split_transfer = bool(split_options)
    opening_split = split_options.get("opening_target_burst", False)
    stream_burst = arm == P8_TRANSFER_ARM or split_transfer
    random_p8 = arm == RANDOM_P8_B4
    _require(not random_p8 or cohort == "main12_block_15s",
             "Random P8 requires the first12 block-control cohort")

    burst = cohort == "main12_burst_15s" or transfer
    from .accepted_stream_client import enabled as stream_enabled, StreamDecoderProxy
    from .accepted_stream import (TIMING_SCOPE as STREAM_TIMING_SCOPE, SINGLE_GPU_TIMING_SCOPE,
                                  single_gpu_enabled, stream_devices)
    single_gpu = single_gpu_enabled()
    single_gpu_placement = single_gpu and cohort == "main12_block_15s" and (
        random_p8 or arm == "fixed_deadline8_k4"
    )
    single_gpu_first12_rank = single_gpu and cohort == "main12_burst_15s" and arm == P8_TRANSFER_ARM
    _require((duration_seconds is not None) == transfer
             and (smoke_prompt_id is None or transfer or single_gpu_placement),
             "explicit duration and smoke selection require a supported burst cohort")
    if single_gpu:
        STREAM_TIMING_SCOPE = SINGLE_GPU_TIMING_SCOPE
    streaming = stream_enabled()
    from .accepted_prefill import (ENV as PREFILL_ENV, burst_length as prefill_burst_length,
                                   enabled as prefill_enabled)
    accepted_prefill = prefill_enabled()
    from .sf_joint_handoff import ENV as JOINT_HANDOFF_ENV
    from .native_rank_indices import enabled as index_flag
    joint_handoff = index_flag(JOINT_HANDOFF_ENV)
    from . import sf_target_two_step
    mixed_target = sf_target_two_step.enabled()
    target_schedule = sf_target_two_step.schedule_metadata() if mixed_target else {}
    _require(not mixed_target or (
        single_gpu and transfer and streaming and duration_seconds == 30 and arm == PAPER30_ARM
        and sealed_calibration_ranks and sf_target_two_step.early_native_burst_count() == 2
        and sf_target_two_step.burst_schedules() == (
            sf_target_two_step.LOW_NOISE_REFINEMENT, *(sf_target_two_step.FOLLOWUP_SCHEDULE,) * 3)
    ), 'mixed Target steps require the exact single-GPU paper SF30 K36 configuration')
    _require(arm != PAPER30_ARM or mixed_target,
             'paper SF30 K36 requires its declared mixed Target schedule')
    _require(not joint_handoff or (single_gpu and burst and stream_burst
             and not accepted_prefill and not early and not prefix),
             'joint handoff requires the single-GPU committed Self-Forcing burst route')
    _require(not accepted_prefill or (streaming and burst and stream_burst and not opening_split
             and prefill_burst_length() == burst_length_for_arm(arm)),
             'accepted Target prefill requires a matching streaming SF burst after a Draft block')
    _require(not single_gpu or (streaming and (burst or single_gpu_placement) and not accepted_prefill
             and not defer_video_delivery and enforce_gpu_idle),
             'single-GPU mode requires a native streaming burst without concurrent prefill')
    from .paper_population import validate_execution
    validate_execution(single_gpu=single_gpu, seconds=duration_seconds, full_population=cohort == 'main24_burst')
    stream_process_options = {}
    if streaming:
        if mixed_target:
            process_factory = sf_target_two_step.process_factory
        elif joint_handoff:
            from .sf_joint_handoff_adapter import process_factory
        else:
            from .accepted_stream_worker import process_factory
        stream_process_options = {'process_factory': process_factory}
    _require(not streaming or (burst and stream_burst) or single_gpu_placement,
             'streaming pilot requires a frozen P8 burst route')
    _require(not transfer or (streaming and stream_burst),
             "main24 burst requires a streaming P8 arm frozen for this horizon")
    _require(type(sealed_calibration_ranks) is bool and
             (not sealed_calibration_ranks or ((transfer or random_p8 or single_gpu_placement or single_gpu_first12_rank)
              and calibration_code_revision is not None)),
             "sealed rank startup requires a frozen P8 run with explicit C10 revision")
    _require(type(defer_video_delivery) is bool and
             (not defer_video_delivery or (transfer and streaming and not split_transfer and smoke_prompt_id is None)),
             "deferred delivery requires the complete native P8 transfer cohort")
    block_control = cohort == "main12_block_15s"
    direct = transfer or cohort in {"main12_direct_15s", "main12_block_15s", "main12_burst_15s", "main12_early_15s", "main12_prefix_15s"}
    main24 = cohort != "development30"
    _require(direct or calibration_code_revision is None, "historical calibration is only allowed for the explicit direct diagnostic")
    allowed_arms = PREFIX_ARMS if prefix else EARLY_ARMS if early else BURST_ARMS if burst else BLOCK_ARMS if block_control else DIRECT_ARMS if direct else MAIN24_ARMS if main24 else ARMS
    if split_transfer:
        allowed_arms = (arm,)
    _require(arm is None or arm in allowed_arms or random_p8, "unknown pilot arm")
    arms = RANDOM_P8_B4_ARMS if random_p8 else (arm,) if arm is not None else allowed_arms
    run_dir, model_root = _run_dir(), _model_root()
    _require(not (run_dir / "cells").exists() and not (run_dir / "result.json").exists(),
             "acceptance pilot requires a fresh attempt")
    _require(all(not (parent / name).exists() for parent in (run_dir, *run_dir.parents)
                 for name in ("_SUCCESS.json", "_FAILED.json", "artifact-index.json")),
             "refusing to write inside a sealed run")
    _require(os.environ.get("CFEXP_DRAFT_KV_REUSE") == "1",
             "pilot requires explicit native Draft KV reuse")
    devices = stream_devices() if single_gpu else _visible_devices()
    profile = duration_profile(duration_seconds if transfer else (15 if main24 else 30))
    benchmark = "VBench-Long" if profile.nominal_seconds == 60 else "VBench"
    if benchmark == "VBench-Long" and scorer_factory is OfficialVBenchScorer:
        from .vbench_long import OfficialVBenchLongScorer
        scorer_factory = OfficialVBenchLongScorer
    experiment_id = "erank-causal-direct-main12-15s-v1" if direct else "erank-acceptance-main24-15s-v1" if main24 else PROTOCOL_ID
    if block_control:
        experiment_id = "target-block-placement-main12-15s-v1"
    if burst:
        experiment_id = (f"committed-rank-burst-main24-{profile.nominal_seconds}s-v1" if transfer
                         else "committed-rank-burst-main12-15s-v1")
    if early:
        experiment_id = "early-current-rank-main12-15s-v1"
    if prefix:
        experiment_id = "target-first-update-main12-15s-v1"
    duration = {**profile.metadata(), "experiment_id": experiment_id,
                "calibration_policy": "explicit_frozen_C10_control" if calibration_code_revision else "same_revision_C10_receipt"}
    if direct:
        duration.update(ours_proposal_budget={a: int(a.rsplit("_k", 1)[1]) for a in arms},
                        budget_rule="explicit_per_arm_budget_after_target_root")
    duration.update(split_options)
    repository = find_git_root(Path.cwd())
    prefix_gate = None
    if prefix:
        from .prefix_quality_gate import verify_gate
        prefix_gate = verify_gate(prefix_gate_run, repository)
    config = load_run_config(boundary_config_path, repo_root=repository)
    protocol = load_protocol(config.protocol_path)
    from .paper_population import assert_calibration_compatible, enabled as paper_enabled
    paper_calibration = assert_calibration_compatible(repository, protocol)
    original_cal = {str(p["prompt_id"]): p for p in protocol.calibration_prompts}
    if not paper_enabled() and not development_c10:
        _require(set(PROMPT_IDS).issubset(original_cal), "development prompts missing from C10")
    population_binding = None
    if main24:
        if development_c10:
            from .native_c10_development import select_population
            selected, population_binding = select_population(repository, protocol)
        else:
            selected, population_binding = load_baseline_population(repository)
        if direct and not transfer:
            selected = selected[:12]
        if smoke_prompt_id is not None:
            selected = [p for p in selected if p["id"] == smoke_prompt_id]
            _require(len(selected) == 1, "smoke prompt is outside the frozen population")
        prompts = tuple(PromptCase(p["id"], p["text"], "evaluation") for p in selected)
        remaining = tuple(PromptCase(pid, str(p["prompt"]), "calibration") for pid, p in original_cal.items())
    else:
        prompts = tuple(PromptCase(pid, str(original_cal[pid]["prompt"]), "evaluation") for pid in PROMPT_IDS)
        remaining = tuple(PromptCase(pid, str(p["prompt"]), "calibration")
                          for pid, p in original_cal.items() if pid not in PROMPT_IDS)
    prompt_ids = tuple(p.prompt_id for p in prompts)
    if development_c10:
        from .native_c10_development import validate_development_population
        validate_development_population(repository,protocol,remaining,prompts)
    hardware = _hardware(tuple(dict.fromkeys(devices))) if enforce_gpu_idle else {"unverified_test_fixture": True}
    # 'evaluation' is the trajectory namespace. Both cohorts are now method
    # development populations; the frozen calibration CDF still uses all C10.
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
        draft_verified_artifacts=draft_verified, target_verified_artifacts=target_verified, phase_seeds=(CALIBRATION_SEED,))
    binding = calibration_control_binding(current, calibration_code_revision)
    calibration_dir = run_dir / "calibration-control"
    calibration_dir.mkdir(parents=True, exist_ok=False)
    calibrator = None
    if sealed_calibration_ranks:
        started = time.perf_counter()
        calibrator, calibration_evidence = _read_sealed_calibration(binding)
        calibration_read_seconds = time.perf_counter() - started
    else:
        sealed, receipt = receipt_policy(smoke_prompt_id=None, run_dir=calibration_dir, binding=binding)
        _require(sealed is False and receipt is not None, "verified C10 control receipt is required")
        calibration_evidence = receipt.public_binding()
    preflight = {"schema": "erank_acceptance_pilot_preflight_v1", "created_at": utc_now(),
        "git": collect_git_metadata(repository), "code_revision": current["code_revision"],
        "calibration_code_revision": binding["code_revision"], "calibration_receipt": calibration_evidence,
        "population": ("independent_C10_development_not_held_out" if development_c10 else
                       "main24_smoke" if smoke_prompt_id else "main24_method_development_not_held_out"
                       if transfer else "main24_fixed_first12_method_development_not_held_out"
                       if direct else "main24_method_development_not_held_out" if main24 else "C10_development_not_held_out"),
        "population_binding": population_binding, "hardware": hardware, "formal_ours_result": False,
        "cohort": cohort, "prompt_ids": list(prompt_ids), "prompts": [{"id": p.prompt_id, "text": p.prompt} for p in prompts],
        "arms": list(arms), "seed": SEED, "duration": duration,
        "proposal_budget": {a: int(a.rsplit("_k", 1)[1]) for a in arms} if direct else profile.proposal_budget, "worker_reuse": "within_arm_only",
        "generation_timer": "materialization_only_excludes_load_decode_quality_and_publication",
        "video_ready_timer": STREAM_TIMING_SCOPE if streaming else READY_TIMING_SCOPE,
        "conditioning_policy": "precomputed_before_generation_timer",
        "candidate_selection": "current-chunk causal relative eRank drop with fixed budget; no lookahead or coverage window" if direct else "eRank arms share coverage selector; Random and uniform have frozen independent positions",
        "benchmark": benchmark, "matched_target_speedup_available": False,
        "paired_successor_probes": "none: current-chunk causal eRank, unconditional correction acceptance" if direct else "retained in all eRank arms to isolate acceptance; not optimized direct-accept execution",
        "calibration_revision_explicit": calibration_code_revision is not None,
        "target_cache_semantics": "persistent native correction session with disclosed bounded context rebase"}
    from .paper_population import population_label
    preflight["population"] = population_label(preflight["population"])
    if paper_calibration is not None:
        preflight['paper_calibration'] = paper_calibration
    if burst:
        preflight.update(candidate_selection="committed Draft eRank schedules the next frozen burst of Target blocks",
            paired_successor_probes="none; every generated block is committed",
            signal_policy="frozen C10 drop control on committed Draft observations; no trained predictor",
            burst_length=(burst_length_for_arm(arms[0]) if len({burst_length_for_arm(a) for a in arms}) == 1 else None),
            burst_lengths_by_arm={a:burst_length_for_arm(a) for a in arms}, discard_generated_draft=False)
        preflight["observation_horizons_by_arm"] = {a:observation_horizon_for_arm(a) for a in arms}
    if block_control:
        preflight.update(candidate_selection="frozen per-arm Target positions; shared Target root at chunk zero",
                         target_positions_by_arm_prompt={a: {p.prompt_id: list(block_positions(a, chunk_count=profile.chunk_count, prompt_id=p.prompt_id)) for p in prompts} for a in arms},
                         paired_successor_probes="none; no Draft proposal on preselected Target chunks",
                         signal_policy="no eRank selection; mechanism control, not the proposed method")
    if random_p8:
        preflight.update(signal_policy="uniform SHA-256 selection of one eligible Draft observation per prompt and repeat",
                         random_selection_protocol=RANDOM_P8_B4_PROTOCOL,
                         observation_horizon=RANDOM_P8_B4_HORIZON,
                         routing_repeats=list(range(len(RANDOM_P8_B4_ARMS))))
    if early:
        preflight.update(candidate_selection="current-chunk eRank at layer14 timestep1000; stop Draft before Target",
            paired_successor_probes="none", same_signal_as_late_current_chunk=True,
            cache_update="one current-slot clean forward after each accepted Target, no history replay",
            stopped_probe_is_generated_latent=False)
    if prefix:
        preflight.update(candidate_selection="early eRank at layer14, or fixed every chunk when budget29",
            signal_policy="prefix_all_k29 is a mechanism control; prefix_rank_k14 uses frozen C10 eRank",
            intervention="Target first clean prediction at timestep1000, then Draft's resolved native suffix; exact steps are bound in runtime evidence; partial Target is never decoded directly",
            prefix_gate=prefix_gate, paired_successor_probes="none")
    preflight["stage_timing_requested"] = diagnostic_timing
    preflight["instrumented_latency_is_diagnostic_only"] = diagnostic_timing
    preflight["accepted_target_prefill"] = accepted_prefill
    preflight.update(target_schedule)
    if joint_handoff:
        preflight["joint_handoff"] = True
    if single_gpu:
        preflight["accepted_video_schedule"] = "decode_after_generation"
    if defer_video_delivery:
        preflight["accepted_video_schedule"] = "after_generation"
    if sealed_calibration_ranks:
        preflight["sealed_calibration_read_seconds"] = calibration_read_seconds
    _write_new_json(run_dir / "provenance/preflight.json", preflight)
    cells, requests = [], []
    with feature_environment("baseline" if early or prefix else "fork_only",
                             stage_timing=diagnostic_timing) as features:
        if accepted_prefill:
            features[PREFILL_ENV] = '1'
        for arm in arms:
            budget = int(arm.rsplit("_k", 1)[1]) if direct else profile.proposal_budget
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
                shared_precomputed_conditioning=True,
                **({"allow_shared_device": True} if single_gpu else {}),
                **stream_process_options)
            load_seconds = time.perf_counter() - started
            try:
                decoder_load_started = time.perf_counter()
                stream_decoder = (StreamDecoderProxy(runtime.draft, runtime.target, directory,
                                                     chunk_count=profile.chunk_count)
                                  if streaming else None)
                if stream_decoder is not None:
                    load_seconds += time.perf_counter() - decoder_load_started
                    runtime.evaluator.client = stream_decoder
                live_decoder = None if defer_video_delivery else stream_decoder
                from .accepted_prefill_worker import AcceptedPrefillProxy
                prefix_prefiller = AcceptedPrefillProxy(runtime.draft) if accepted_prefill else None
                runtime.evaluator.full_quality_metric_id = ("vbench_long_custom_six_dimension_macro_mean"
                    if benchmark == "VBench-Long" else "vbench_custom_six_dimension_macro_mean")
                if calibrator is None:
                    calibrator = _calibrator_from_restored(directory=calibration_dir, runtime=runtime,
                        protocol=protocol, evaluation=original_eval)
                campaign_factory=Insight2Campaign
                campaign_options=dict(protocol_id=f"ours-target-duration-v1-{profile.nominal_seconds}s" if main24 else PROTOCOL_ID)
                if development_c10:
                    from .native_c10_development import make_development_campaign
                    campaign_factory=make_development_campaign
                    campaign_options=dict(repo=repository,protocol=protocol)
                campaign = campaign_factory(run_dir=directory,**campaign_options,seeds=(SEED,),
                    chunk_count=profile.chunk_count, calibration_prompts=remaining, evaluation_prompts=prompts,
                    draft=runtime.draft, target=runtime.target, noise_provider=runtime.noise_provider,
                    evaluator=runtime.evaluator)
                for prompt in prompts:
                    if enforce_gpu_idle:
                        _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                            target_device=devices[1], phase=f"acceptance-pilot:{arm}:{prompt.prompt_id}")
                    event_id = hashlib.sha256(f"{PROTOCOL_ID}:{arm}:{prompt.prompt_id}".encode()).hexdigest()
                    started = time.perf_counter()
                    with campaign._target_rpc_scope(event_id):
                        if prefix:
                            from .prefix_rank_route import materialize_prefix_rank
                            key, route, accepted, history = materialize_prefix_rank(
                                campaign, prompt=prompt, seed=SEED, budget=budget, calibrator=calibrator)
                            proposed = accepted
                        elif early:
                            from .early_rank_route import materialize_early_rank
                            key, route, accepted, history = materialize_early_rank(
                                campaign, prompt=prompt, seed=SEED, budget=budget, calibrator=calibrator)
                            proposed = accepted
                        elif block_control:
                            positions = block_positions(arm, chunk_count=profile.chunk_count, prompt_id=prompt.prompt_id)
                            key, route, accepted, history = campaign._materialize_fixed_positions(
                                prompt=prompt, seed=SEED, corrected_chunks=positions,
                                bypass_draft_on_target_chunks=True,
                                **({"on_commit": live_decoder.append} if single_gpu_placement else {}))
                            _require(tuple(accepted) == positions, "block-control positions changed")
                            proposed = accepted
                        elif burst:
                            from .rank_burst import materialize_rank_bursts
                            key, route, accepted, history = materialize_rank_bursts(
                                campaign, prompt=prompt, seed=SEED, budget=budget, calibrator=calibrator,
                                burst_length=burst_length_for_arm(arm),
                                observation_horizon=observation_horizon_for_arm(arm),
                                on_commit=live_decoder.append if live_decoder is not None else None,
                                before_generate=(live_decoder.before_generate
                                                 if live_decoder is not None and not single_gpu else None),
                                on_target_prefix=prefix_prefiller.start if prefix_prefiller is not None else None,
                                **split_options)
                            proposed = accepted
                        elif direct:
                            key, route, accepted, history = campaign._materialize_online_rank_guided(
                                prompt=prompt, seed=SEED, budget=budget, calibrator=calibrator)
                            proposed = accepted
                        elif arm in {"random", "uniform"}:
                            positions = (random_positions(protocol_id=str(protocol.raw["protocol_id"]) if main24 else PROTOCOL_ID,
                                prompt_id=prompt.prompt_id, generation_seed=SEED,
                                budget=budget, repeat_id=0, chunk_count=profile.chunk_count)
                                if arm == "random" else uniform_positions(profile.proposal_budget,
                                    chunk_count=profile.chunk_count))
                            key, route, accepted, history = campaign._materialize_fixed_positions(
                                prompt=prompt, seed=SEED, corrected_chunks=positions)
                            _require(tuple(accepted) == positions, "fixed intervention positions changed")
                            proposed = accepted
                        else:
                            key, route, accepted, proposed, history = campaign._materialize_online_lookahead_coverage_boundary_gated(
                                prompt=prompt, seed=SEED, budget=budget, calibrator=calibrator,
                                window_radius_chunks=WINDOW_RADIUS_CHUNKS, minimum_gap_chunks=MINIMUM_GAP_CHUNKS,
                                maximum_gap_chunks=MAXIMUM_GAP_CHUNKS, acceptance_rule=arm)
                    elapsed = time.perf_counter() - started
                    if defer_video_delivery:
                        # Preserve the worker's ordered per-chunk append protocol and
                        # its decoder state while moving every delivery call after generation.
                        for count in range(1, history.next_chunk_index + 1):
                            stream_decoder.append(history.prefix_before(count))
                    pending = campaign._prepare_rollout(key=key, route_id=route, corrected_chunks=accepted,
                        proposed_target_chunks=proposed, history=history)
                    ready_elapsed = time.perf_counter() - started
                    if prefix:
                        from .prefix_rank_route import collect_prefix_trajectory
                        trajectory = collect_prefix_trajectory(campaign, key=key, history=history,
                                                               accepted=accepted, proposed=proposed)
                    elif early:
                        from .early_rank_route import collect_early_trajectory
                        trajectory = collect_early_trajectory(campaign, key=key, history=history,
                                                              accepted=accepted, proposed=proposed)
                    else:
                        trajectory = collect_trajectory(campaign, key=key, history=history, accepted=accepted, proposed=proposed)
                    if arm == "target_root_only_k0":
                        collect_root_only_ranks(campaign, key, trajectory)
                    _require(len(proposed) == budget, "Target proposal budget changed")
                    _require(trajectory["counters"]["physical_calls"]["target"] == (1 if prefix else budget + 1),
                             "Target physical call budget differs from frozen proposals plus root")
                    if direct or arm in {"proposal_always", "random", "uniform"}:
                        _require(tuple(accepted) == tuple(proposed), "direct acceptance discarded a proposal")
                    cache_audit = ({"draft_reused_chunks": trajectory['counters']['reused_chunks'],
                        "prefix_execution": trajectory['prefix_execution_audit'],
                        "target_native_states": trajectory['prefix_native_states']} if prefix
                        else verify_native_cache(campaign, key, trajectory))
                    if mixed_target:
                        cache_audit['target_step_allocation'] = sf_target_two_step.verify_paper30_execution(
                            campaign, key, trajectory)
                    if early:
                        cache_audit['early_execution'] = trajectory['early_execution_audit']
                    if burst:
                        from .rank_burst import verify_burst_reuse, validate_burst_schedule
                        cache_audit["burst_reuse"] = verify_burst_reuse(campaign, key,
                            burst_length=burst_length_for_arm(arm), budget=budget,
                            opening_target_burst=opening_split)
                        if joint_handoff:
                            from .sf_joint_handoff import verify_rollout
                            cache_audit["joint_handoff"] = verify_rollout(
                                campaign, key, selected=trajectory["accepted_chunks"])
                        validate_burst_schedule(trajectory, budget=budget, burst_length=burst_length_for_arm(arm),
                            chunk_count=profile.chunk_count,
                            observation_horizon=observation_horizon_for_arm(arm), **split_options)
                    if direct:
                        _require(trajectory["counters"]["physical_calls"]["draft"] == profile.chunk_count - 1 - (budget if block_control or burst else 0),
                                 "direct route executed unexpected Draft calls")
                    if arm == "target_root_only_k0":
                        _require(trajectory["counters"]["replayed_chunks"] == 1,
                                 "root-only Draft must replay the Target root once and reuse its accepted suffix")
                    prepared = pending.prepared_quality
                    video = bound_file(run_dir, directory / prepared.video_path)
                    from .video_encoding_policy import verify_video
                    encoding_policy = verify_video(run_dir / video['path'])
                    _require(video["sha256"] == prepared.video_sha256 and video["nbytes"] == prepared.video_nbytes,
                             "decoded video differs from prepared evidence")
                    decode = bound_file(run_dir, directory / prepared.decode_evidence_path)
                    _require(decode["sha256"] == prepared.decode_evidence_sha256, "decode evidence changed")
                    _require(prepared.frame_count == profile.frame_count and prepared.fps == 24,
                             "decoded horizon changed")
                    measurement_id = f"acceptance-pilot-{arm}-{prompt.prompt_id}".replace("_", "-")
                    cell = {"arm": arm, "proposal_budget": budget, "prompt_id": prompt.prompt_id, "seed": SEED, "prompt": prompt.prompt, "measurement_id": measurement_id,
                        "features": dict(features), "generation_wall_seconds": elapsed, "trajectory": trajectory, "native_cache_audit": cache_audit,
                        "video_ready_wall_seconds": ready_elapsed,
                        "timing_scope": STREAM_TIMING_SCOPE if streaming else READY_TIMING_SCOPE,
                        "decode_encode_and_prepare_wall_seconds": ready_elapsed - elapsed,
                        "video": video, "decode_evidence": decode, "encoding_policy": encoding_policy,
                        "model_load_wall_seconds_per_arm": load_seconds}
                    if streaming:
                        cell['accepted_stream'] = stream_decoder.last_evidence
                        cell['stream_decoder_model_load_excluded'] = True
                        cell['post_generation_drain_and_prepare_wall_seconds'] = ready_elapsed-elapsed
                        cell['decode_encode_and_prepare_wall_seconds_semantics'] = 'tail_after_generation_not_total_render_work'
                    if defer_video_delivery:
                        cell['accepted_video_schedule'] = 'after_generation'
                        cell['decode_encode_and_prepare_wall_seconds_semantics'] = 'all_video_delivery_after_generation'
                    if single_gpu:
                        cell['decode_encode_and_prepare_wall_seconds_semantics'] = 'gpu_decode_and_encoder_drain_after_generation'
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
    result = {"schema": "erank_acceptance_pilot_v1", "status": "succeeded",
        "formal_ours_result": False, "population": preflight["population"], "code_revision": current["code_revision"],
        "calibration_receipt": preflight["calibration_receipt"],
        "calibration_code_revision": binding["code_revision"], "duration": duration,
        "cohort": cohort, "benchmark": benchmark, "population_binding": population_binding, "hardware": hardware,
        "prefix_gate": prefix_gate,
        "timing_scope": STREAM_TIMING_SCOPE if streaming else READY_TIMING_SCOPE,
        "row_count": len(rows), "rows": rows, "arms": list(arms),
        "quality_summary": summarize(rows, arms, prompt_ids, main24=main24),
        "vbench_wall_seconds": vbench_seconds, "automatic_winner_selection": False}
    if direct and not transfer:
        result["quality_summary"] = {a: {"fixed_first12": v["main24"]} for a, v in result["quality_summary"].items()}
    if development_c10:
        result["quality_summary"] = {a: {"independent_C10": v["main24"]}
                                     for a, v in result["quality_summary"].items()}
        result["development_only"] = True
    if single_gpu:
        result["accepted_video_schedule"] = "decode_after_generation"
    result.update(target_schedule)
    if defer_video_delivery:
        result["accepted_video_schedule"] = "after_generation"
    _write_new_json(run_dir / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft-artifact-spec", "target-artifact-spec", "vbench-checkout", "vbench-python",
                 "vbench-asset-manifest", "vbench-asset-spec"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--boundary-config", type=Path, default=BOUNDARY_CONFIG)
    arm_choices = (*ARMS, *MAIN24_ARMS, *DIRECT_ARMS, *BLOCK_ARMS, RANDOM_P8_B4,
                   *BURST_ARMS, *SPLIT_ARMS, *EARLY_ARMS, *PREFIX_ARMS)
    parser.add_argument("--arm", choices=tuple(dict.fromkeys(arm_choices)))
    parser.add_argument("--cohort", choices=COHORTS, default="development30")
    parser.add_argument("--calibration-code-revision", help="Explicit frozen C10 control for direct diagnostic only")
    parser.add_argument("--prefix-gate-run", type=Path)
    parser.add_argument("--duration-seconds", type=int, choices=(15, 30, 60))
    parser.add_argument("--smoke-prompt-id")
    parser.add_argument("--defer-video-delivery", action="store_true")
    parser.add_argument("--sealed-calibration-ranks", action="store_true",
                        help="Read the exact frozen eRank distribution from its original sealed C10 run")
    parser.add_argument("--development-c10", action="store_true",
                        help="Explicit separate native-four-step C10 development; never paper128 evidence")
    args = parser.parse_args(argv)
    try:
        try:
            result = run_probe(draft_artifact_spec_path=args.draft_artifact_spec,
                target_artifact_spec_path=args.target_artifact_spec, boundary_config_path=args.boundary_config,
                vbench_checkout=args.vbench_checkout, vbench_python=args.vbench_python,
                vbench_asset_manifest=args.vbench_asset_manifest, vbench_asset_spec=args.vbench_asset_spec,
                arm=args.arm, cohort=args.cohort, calibration_code_revision=args.calibration_code_revision,
                prefix_gate_run=args.prefix_gate_run, duration_seconds=args.duration_seconds,
                smoke_prompt_id=args.smoke_prompt_id, defer_video_delivery=args.defer_video_delivery,
                sealed_calibration_ranks=args.sealed_calibration_ranks,
                development_c10=args.development_c10)
        finally:
            shutdown_owned_runtime_helpers(label="acceptance-pilot-controller")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"acceptance-pilot-repair-probe: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
