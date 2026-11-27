"""Paired systems diagnostic of the unchanged Boundary-gated policy.

Run through scripts/run_remote.sh at a published full SHA. This is neither a
formal Ours quality evaluation nor a claim against Full Target. All four arms
use the same frozen CDF and incremental Draft cache; only fork preservation and
entry short-circuiting vary. Every timed cell starts with fresh native workers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import launch_correction_runtime
from .duration_sweep import duration_profile
from .erank_frame_campaign import verify_artifact_spec as verify_draft_artifact_spec
from .insight2_campaign import Insight2Campaign, PromptCase
from .insight2_protocol import load_protocol
from .insight2_run import _calibration_receipt_binding, load_run_config
from .main_table_baseline_campaign import (_model_root, _run_dir, _selected_prompts,
                                          _sha256_file, _visible_devices, _write_new_json)
from .main_table_ours_campaign import (BOUNDARY_CONFIG, MAXIMUM_GAP_CHUNKS,
    MINIMUM_GAP_CHUNKS, WINDOW_RADIUS_CHUNKS, _assert_idle_pair,
    _assert_worker_pair_owns_gpus, _receipt_policy)
from .main_table_v1 import PROMPT_IDS, load_bundle
from .metadata import collect_git_metadata, find_git_root, utc_now
from .runtime_helpers import shutdown_owned_runtime_helpers
from .stage_timing_probe import _hardware

PROTOCOL_ID = "branch-gate-systems-diagnostic-v1"
SEED = 20260911
PROMPT_ID = PROMPT_IDS[0]
ARMS = {"baseline": (False, False), "fork_only": (True, False),
        "early_only": (False, True), "both": (True, True)}


class BranchGateProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BranchGateProbeError(message)


@contextmanager
def feature_environment(arm: str, *, stage_timing: bool = False):
    _require(arm in ARMS, "unknown diagnostic arm")
    _require(type(stage_timing) is bool, "stage timing selection must be boolean")
    fork, early = ARMS[arm]
    values = {"CFEXP_DRAFT_KV_REUSE": "1", "CFEXP_DRAFT_KV_FORK": str(int(fork)),
              "CFEXP_BOUNDARY_EARLY_REJECT": str(int(early)), "CFEXP_STAGE_TIMING": str(int(stage_timing))}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def calibration_control_binding(binding: Mapping[str, Any], revision: str | None) -> dict[str, Any]:
    """Explicit diagnostic-only choice; generic receipt validation stays strict."""
    result = deepcopy(dict(binding))
    if revision is not None:
        _require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                 "calibration code revision must be a full lowercase SHA")
        result["code_revision"] = revision
    return result


class _ReadOnlyBackend:
    """A complete restored receipt must never execute missing calibration work."""
    def __init__(self, wrapped: Any):
        self.wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        if name in {"generate", "generate_persistent", "materialize", "start_trajectory"}:
            raise BranchGateProbeError("receipt revalidation attempted new calibration compute")
        return getattr(self.wrapped, name)


def _calibrator_from_restored(*, directory: Path, runtime: Any, protocol: Any,
                              evaluation: Sequence[PromptCase]) -> Any:
    campaign = Insight2Campaign(run_dir=directory, protocol_id=str(protocol.raw["protocol_id"]),
        seeds=(SEED,), chunk_count=120,
        calibration_prompts=tuple(PromptCase(str(p["prompt_id"]), str(p["prompt"]), "calibration")
                                  for p in protocol.calibration_prompts),
        evaluation_prompts=evaluation, draft=_ReadOnlyBackend(runtime.draft),
        target=_ReadOnlyBackend(runtime.target), noise_provider=_ReadOnlyBackend(runtime.noise_provider),
        evaluator=runtime.evaluator)
    return campaign.run_calibration()


def _bound_latent(root: Path, segment: Mapping[str, Any]) -> dict[str, Any]:
    unresolved = root / str(segment["latent_path"])
    path = unresolved.resolve(strict=True)
    _require(path.is_relative_to(root.resolve()) and not unresolved.is_symlink(), "latent escapes cell")
    digest, size = _sha256_file(path)
    _require(digest == segment["latent_sha256"], "committed latent bytes changed")
    return {"chunk_index": segment["chunk_index"], "producer_role": segment["producer_role"],
            "latent_sha256": digest, "nbytes": size}


def validate_anchor_bounds(evidence: Sequence[Mapping[str, Any]], *, window_count: int) -> None:
    anchors = [x["branch_anchor"] for x in evidence if "branch_anchor" in x]
    _require(all(type(x.get(name)) is int and x[name] >= 0 for x in anchors
                 for name in ("anchor_clone_count", "anchor_clone_bytes", "peak_resident_bytes", "state_count")),
             "invalid branch-anchor counters")
    _require(all(x["state_count"] <= 2 and x["peak_resident_bytes"] <= 16 * 1024**3 for x in anchors),
             "branch anchor exceeds two states or 16 GiB budget")
    _require(sum(x["anchor_clone_count"] for x in anchors) <= window_count,
             "branch anchor cloned more than once per coverage window")
    for item in evidence:
        anchor = item.get("branch_anchor")
        if anchor is not None and anchor.get("action") == "consume":
            _require(anchor.get("anchor_consumed") is True and anchor.get("anchor_miss_reason") is None,
                     "fresh diagnostic missed a branch anchor on consume")
            _require(item["replayed_chunks"] <= 2 * WINDOW_RADIUS_CHUNKS + 1,
                     "anchor consumption exceeded the nine-chunk window replay bound")


def collect_trajectory(campaign: Insight2Campaign, *, key: str, history: Any,
                       accepted: Sequence[int], proposed: Sequence[int]) -> dict[str, Any]:
    """Read exact committed bytes and unique physical generations after timing."""
    decisions, generated = [], {}
    for index in range(campaign.chunk_count):
        item = json.loads(campaign._checkpoint_path("rollouts", key, index).read_text())
        metadata = item["metadata"]
        values = [item["generated"]] + [metadata.get(name) for name in
            ("draft_proposal", "gate_probe", "target_candidate", "corrected_frontier_probe")]
        for value in values:
            if value is not None:
                path = value["segment"]["latent_path"]
                if path in generated:
                    _require(generated[path] == value, "a generation changed across checkpoints")
                generated[path] = value
        if index:
            decisions.append({name: metadata.get(name) for name in
                ("chunk_index", "decision", "rank", "risk_score", "erank_drop_score",
                 "gate_signal_rank", "target_candidate_proposed", "target_candidate_accepted",
                 "boundary_gate_decision", "carried_frontier_kind")})
    calls = {role: sum(x["context"]["role"] == role for x in generated.values())
             for role in ("draft", "target")}
    evidence = [x.get("diagnostics", {}).get("kv_reuse") for x in generated.values()
                if x["context"]["role"] == "draft"]
    _require(bool(evidence) and all(isinstance(x, dict) for x in evidence),
             "every Draft call must report incremental cache evidence")
    _require(all(type(x.get(name)) is int and x[name] >= 0 for x in evidence
                 for name in ("reused_chunks", "replayed_chunks")), "invalid incremental cache counters")
    counters = {name: sum(x[name] for x in evidence)
                for name in ("reused_chunks", "replayed_chunks")}
    anchors = [x["branch_anchor"] for x in evidence if "branch_anchor" in x]
    validate_anchor_bounds(evidence, window_count=len(proposed))
    consumed_replays = [x["replayed_chunks"] for x in evidence
                       if x.get("branch_anchor", {}).get("action") == "consume"]
    counters.update({"physical_calls": calls, "draft_evidence_count": len(evidence),
        "anchor_evidence_count": len(anchors),
        "anchor_clone_count": sum(x["anchor_clone_count"] for x in anchors) if anchors else None,
        "anchor_clone_bytes": sum(x["anchor_clone_bytes"] for x in anchors) if anchors else None,
        "anchor_consumed_count": sum(bool(x["anchor_consumed"]) for x in anchors) if anchors else None,
        "anchor_action_counts": {action: sum(x.get("action") == action for x in anchors)
                                 for action in ("none", "preserve", "consume")} if anchors else None,
        "maximum_consume_replayed_chunks": max(consumed_replays) if consumed_replays else None,
        "peak_cache_resident_bytes": max(x["peak_resident_bytes"] for x in anchors) if anchors else None,
        "peak_cache_resident_bytes_definition": "Native KV state ledger high-water; not total CUDA allocation or process VRAM.",
        "rollback_restore": {"status": "not_applicable", "reason": "This implementation uses branch anchors, not rollback restoration."}})
    committed = [_bound_latent(campaign.run_dir, segment.to_dict()) for segment in history.segments]
    return {"accepted_chunks": list(accepted), "proposed_chunks": list(proposed),
            "committed": committed, "decisions": decisions, "counters": counters,
            "final_history_head_sha256": history.head_sha256}


def assert_equivalent(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    for name in ("accepted_chunks", "proposed_chunks", "committed"):
        _require(candidate[name] == reference[name], f"trajectory parity failed: {name}")
    _require(len(candidate["decisions"]) == len(reference["decisions"]), "decision coverage changed")
    for old, new in zip(reference["decisions"], candidate["decisions"]):
        _require({k: v for k, v in old.items() if k != "boundary_gate_decision"}
                 == {k: v for k, v in new.items() if k != "boundary_gate_decision"},
                 f"controller/rank parity failed at chunk {old['chunk_index']}")
        full, gate = old["boundary_gate_decision"], new["boundary_gate_decision"]
        if gate is not None and gate.get("evaluation_stage") == "entry_only":
            _require(full is not None and full["accept_target"] is False
                     and full["evidence_valid"] is True and full["entry_not_worse"] is False
                     and gate["accept_target"] is False and gate["short_circuit"] is True
                     and gate["evidence_complete"] is False and gate["evidence_valid"] is True,
                     "entry short circuit is not an equivalent known rejection")
            for name in ("draft_entry_witness_score", "target_entry_witness_score",
                         "entry_delta", "entry_delta_threshold"):
                _require(full[name] == gate[name], f"entry short-circuit evidence changed: {name}")
            _require(gate["entry_delta"] > gate["entry_delta_threshold"], "entry rejection threshold not crossed")
            _require(gate.get("reason") == "rejected:target_entry_worse" and gate.get("entry_not_worse") is False
                     and all(gate.get(name) is None for name in ("original_successor_erank",
                         "corrected_successor_erank", "erank_delta", "erank_recovered", "draft_exit_witness_score",
                         "target_exit_witness_score", "exit_delta", "exit_not_worse")),
                     "entry rejection fabricated unmeasured successor evidence")
        else:
            _require(gate == full, f"full gate parity failed at chunk {old['chunk_index']}")


def summarize_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(cells) == 8, "diagnostic requires all four arms in two rounds")
    indexed = {(x["repeat"], x["arm"]): x for x in cells}
    _require(len(indexed) == 8 and set(indexed) == {(r, a) for r in range(2) for a in ARMS},
             "duplicate or missing diagnostic cell")
    reference = indexed[(0, "baseline")]
    for cell in cells:
        _require(cell["duration"] == reference["duration"], "cell horizon changed")
        _require(math.isfinite(cell["generation_wall_seconds"]) and cell["generation_wall_seconds"] > 0,
                 "invalid generation duration")
        assert_equivalent(reference["trajectory"], cell["trajectory"])
    summary = {}
    for arm in ARMS:
        elapsed = [indexed[(r, arm)]["generation_wall_seconds"] for r in range(2)]
        ratios = [indexed[(r, "baseline")]["generation_wall_seconds"] / elapsed[r] for r in range(2)]
        summary[arm] = {"wall_seconds": elapsed, "median_wall_seconds": statistics.median(elapsed),
                        "paired_speedup_vs_incremental_baseline": ratios,
                        "paired_geometric_mean_speedup": math.exp(sum(map(math.log, ratios)) / 2)}
    return {"arms": summary, "bitwise_committed_file_parity": True,
            "controller_rank_gate_semantics_parity": True,
            "accepted_count": len(reference["trajectory"]["accepted_chunks"]),
            "rejected_count": len(reference["trajectory"]["proposed_chunks"]) - len(reference["trajectory"]["accepted_chunks"]),
            "both_gate_outcomes_covered": 0 < len(reference["trajectory"]["accepted_chunks"]) < len(reference["trajectory"]["proposed_chunks"]),
            "early_rejection_counts": {a: sum(bool((d["boundary_gate_decision"] or {}).get("short_circuit"))
                                              for d in indexed[(0, a)]["trajectory"]["decisions"]) for a in ARMS},
            "interpretation": "One prompt, one seed, two cold-trajectory repeats; not a quality score, Full Target speedup, or confidence interval."}


def run_probe(*, draft_artifact_spec_path: Path, target_artifact_spec_path: Path,
              boundary_config_path: Path = BOUNDARY_CONFIG, duration_seconds: int = 60,
              calibration_code_revision: str | None = None,
              runtime_launcher: Callable[..., Any] = launch_correction_runtime,
              receipt_policy: Callable[..., Any] = _receipt_policy,
              enforce_gpu_idle: bool = True) -> dict[str, Any]:
    profile = duration_profile(duration_seconds)
    run_dir, model_root = _run_dir(), _model_root()
    _require(not (run_dir / "cells").exists() and not (run_dir / "result.json").exists(),
             "diagnostic requires a fresh attempt, not resume")
    _require(all(not (parent / marker).exists() for parent in (run_dir, *run_dir.parents)
                 for marker in ("_SUCCESS.json", "_FAILED.json", "artifact-index.json")),
             "refusing to write inside a sealed run")
    _require(os.environ.get("CFEXP_STAGE_TIMING") in {None, "", "0"},
             "systems comparison requires stage timing disabled")
    devices = _visible_devices()
    repository = find_git_root(Path.cwd())
    bundle = load_bundle(repository)
    selected = _selected_prompts(repository, bundle.protocol)[0]
    _require(selected["id"] == PROMPT_ID, "frozen diagnostic prompt changed")
    prompt = PromptCase(PROMPT_ID, selected["text"], "evaluation")
    config = load_run_config(boundary_config_path, repo_root=repository)
    protocol = load_protocol(config.protocol_path)
    _require(protocol.raw["protocol_id"] == "insight2-correction-long60-h800-lookahead-boundary-v1",
             "wrong boundary protocol")
    calibration_prompts = tuple(PromptCase(str(p["prompt_id"]), str(p["prompt"]), "calibration")
                                for p in protocol.calibration_prompts)
    evaluation = tuple(PromptCase(str(p["prompt_id"]), str(p["prompt"]), "evaluation")
                       for p in protocol.eval_prompts)
    _require(prompt in evaluation, "diagnostic prompt differs from boundary protocol")
    _, draft_verified, draft_module = verify_draft_artifact_spec(draft_artifact_spec_path,
        model_root=model_root, repo_root=repository)
    _, target_verified, target_module = verify_target_artifact_spec(target_artifact_spec_path,
        model_root=model_root, repo_root=repository)
    _require((draft_module, target_module) == ("cfexp.self_forcing_frame_adapter", "cfexp.krea_chunk_backend"),
             "diagnostic artifact backend drift")
    draft_sha, _ = _sha256_file(draft_artifact_spec_path.resolve(strict=True))
    target_sha = target_artifact_spec_sha256(target_artifact_spec_path)
    protocol_sha, _ = _sha256_file(config.protocol_path)
    current_binding = _calibration_receipt_binding(repo_root=repository, config=config, protocol=protocol,
        protocol_sha256=protocol_sha, draft_spec_sha256=draft_sha, target_spec_sha256=target_sha,
        draft_verified_artifacts=draft_verified, target_verified_artifacts=target_verified, phase_seeds=(SEED,))
    binding = calibration_control_binding(current_binding, calibration_code_revision)
    calibration_dir = run_dir / "calibration-control"
    calibration_dir.mkdir(parents=True, exist_ok=False)
    sealed, receipt = receipt_policy(smoke_prompt_id=None, run_dir=calibration_dir, binding=binding)
    _require(sealed is False and receipt is not None, "diagnostic requires a verified frozen calibration receipt")
    preflight = {"schema": "branch_gate_probe_preflight_v1", "created_at": utc_now(),
        "git": collect_git_metadata(repository), "code_revision": current_binding["code_revision"],
        "calibration_code_revision": binding["code_revision"], "calibration_receipt": receipt.public_binding(),
        "calibration_revision_explicit": calibration_code_revision is not None,
        "current_revision_calibration": binding["code_revision"] == current_binding["code_revision"],
        "duration": profile.metadata(), "prompt_id": PROMPT_ID, "seed": SEED,
        "hardware": _hardware(devices), "formal_ours_result": False, "quality_measured": False,
        "stage_timing_enabled": False, "worker_reuse_between_cells": False,
        "conditioning_policy": "same precomputed prompt conditioning before each cell timer",
        "draft_artifact_spec_sha256": draft_sha, "target_artifact_spec_sha256": target_sha}
    _write_new_json(run_dir / "provenance/preflight.json", preflight)
    cells, calibrator, calibration_seconds = [], None, None
    for repeat in range(2):
        order = list(ARMS) if repeat == 0 else list(reversed(ARMS))
        for arm in order:
            cell_dir = run_dir / "cells" / f"repeat-{repeat}" / arm
            cell_dir.mkdir(parents=True, exist_ok=False)
            with feature_environment(arm) as features:
                if enforce_gpu_idle:
                    _assert_idle_pair(*devices)
                loaded = time.perf_counter()
                runtime = runtime_launcher(run_dir=cell_dir, repo_root=repository,
                    draft_artifact_spec_path=draft_artifact_spec_path, draft_artifact_spec_sha256=draft_sha,
                    draft_verified_artifacts=draft_verified, target_artifact_spec_path=target_artifact_spec_path,
                    target_artifact_spec_sha256=target_sha, target_verified_artifacts=target_verified,
                    devices=devices, write_full_videos=False, chunk_count=profile.chunk_count,
                    conditioning_prompts=[{"prompt_id": PROMPT_ID, "prompt": prompt.prompt}],
                    shared_precomputed_conditioning=True)
                load_seconds = time.perf_counter() - loaded
                try:
                    if calibrator is None:
                        started = time.perf_counter()
                        calibrator = _calibrator_from_restored(directory=calibration_dir, runtime=runtime,
                            protocol=protocol, evaluation=evaluation)
                        calibration_seconds = time.perf_counter() - started
                    if enforce_gpu_idle:
                        _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                            target_device=devices[1], phase=f"diagnostic:{repeat}:{arm}")
                    campaign = Insight2Campaign(run_dir=cell_dir, protocol_id=PROTOCOL_ID, seeds=(SEED,),
                        calibration_prompts=calibration_prompts, evaluation_prompts=(prompt,),
                        draft=runtime.draft, target=runtime.target, noise_provider=runtime.noise_provider,
                        evaluator=runtime.evaluator, chunk_count=profile.chunk_count)
                    _require(not campaign._canonical_rollout_root_checkpoint(prompt, SEED).exists(),
                             "cell must execute a fresh trajectory")
                    started = time.perf_counter()
                    with campaign._target_rpc_scope(hashlib.sha256(f"{PROTOCOL_ID}:{repeat}:{arm}".encode()).hexdigest()):
                        key, _, accepted, proposed, history = campaign._materialize_online_lookahead_coverage_boundary_gated(
                            prompt=prompt, seed=SEED, budget=profile.proposal_budget, calibrator=calibrator,
                            window_radius_chunks=WINDOW_RADIUS_CHUNKS, minimum_gap_chunks=MINIMUM_GAP_CHUNKS,
                            maximum_gap_chunks=MAXIMUM_GAP_CHUNKS)
                    elapsed = time.perf_counter() - started
                    if enforce_gpu_idle:
                        _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                            target_device=devices[1], phase=f"diagnostic-complete:{repeat}:{arm}")
                    trajectory = collect_trajectory(campaign, key=key, history=history, accepted=accepted, proposed=proposed)
                    if ARMS[arm][0]:
                        _require(trajectory["counters"]["anchor_evidence_count"] == trajectory["counters"]["draft_evidence_count"],
                                 "fork-enabled arm lacks branch-anchor evidence")
                    cell = {"schema": "branch_gate_probe_cell_v1", "repeat": repeat, "arm": arm,
                        "duration": profile.metadata(), "features": dict(features), "generation_wall_seconds": elapsed,
                        "model_load_wall_seconds": load_seconds, "trajectory": trajectory}
                    _write_new_json(cell_dir / "result.json", cell)
                    if cells:
                        assert_equivalent(cells[0]["trajectory"], trajectory)
                    cells.append(cell)
                finally:
                    runtime.close()
    result = {"schema": "branch_gate_systems_probe_v1", "status": "succeeded",
        "run_kind": "diagnostic_boundary_policy_systems_comparison", "formal_ours_result": False,
        "quality_measured": False, "code_revision": preflight["code_revision"],
        "calibration_code_revision": preflight["calibration_code_revision"],
        "calibration_receipt": preflight["calibration_receipt"],
        "current_revision_calibration": preflight["current_revision_calibration"],
        "calibration_revalidation_wall_seconds": calibration_seconds,
        "duration": profile.metadata(), "prompt_id": PROMPT_ID, "seed": SEED,
        "cells": cells, "summary": summarize_cells(cells)}
    _write_new_json(run_dir / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-artifact-spec", type=Path, required=True)
    parser.add_argument("--target-artifact-spec", type=Path, required=True)
    parser.add_argument("--boundary-config", type=Path, default=BOUNDARY_CONFIG)
    parser.add_argument("--duration-seconds", type=int, choices=(15, 30, 60), default=60)
    parser.add_argument("--calibration-code-revision", help="Explicit frozen CDF control source SHA; default requires current SHA")
    args = parser.parse_args(argv)
    try:
        try:
            result = run_probe(draft_artifact_spec_path=args.draft_artifact_spec,
                target_artifact_spec_path=args.target_artifact_spec, boundary_config_path=args.boundary_config,
                duration_seconds=args.duration_seconds, calibration_code_revision=args.calibration_code_revision)
        finally:
            # run_probe has completed its per-cell shutdown attempts.
            # Finalize parent-side helpers once, outside every generation timer.
            shutdown_owned_runtime_helpers(label="branch-gate-controller")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"branch-gate-probe: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
