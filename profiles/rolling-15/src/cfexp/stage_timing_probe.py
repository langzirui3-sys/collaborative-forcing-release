"""Fixed-route 15s instrumentation probe, never an Ours or quality result.

Launch only through scripts/run_remote.sh at a full published revision. The
normal runner owns artifact indexing, terminal markers, and publication. This
module profiles existing native compute paths without calibration or learned
selection. The default route rejects chunk-10 correction and accepts chunk-20;
continuation controls use only a Target root, followed by 29 Draft blocks.
Prefix cost controls force alternating or every-block single Target updates;
these diagnose switching overhead and do not measure a learned routing policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import launch_correction_runtime
from .erank_frame_campaign import verify_artifact_spec as verify_draft_artifact_spec
from .insight2_campaign import Insight2Campaign, PromptCase, _generated_dict
from .insight2_protocol import load_protocol
from .insight2_run import load_run_config
from .main_table_baseline_campaign import (_model_root, _run_dir, _selected_prompts,
                                           _sha256_file, _visible_devices, _write_new_json)
from .main_table_full_target import generate_full_target_trajectory
from .main_table_ours_campaign import (_assert_idle_pair, _assert_worker_pair_owns_gpus,
                                        BOUNDARY_CONFIG)
from .main_table_v1 import PROMPT_IDS, load_bundle
from .metadata import collect_git_metadata, find_git_root, utc_now
from .stage_timing_analysis import analyze_run, write_report

PROTOCOL_ID = "stage-timing-fixed-route-15s-v1"
RESULT_SCHEMA = "stage_timing_fixed_route_probe_v1"
PROMPT_ID = PROMPT_IDS[0]
SEED = 20260911
CHUNK_COUNT = 30
BRANCH_CHOICES = {10: "original", 20: "corrected"}


class StageTimingProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageTimingProbeError(message)


def generate_fixed_route(campaign: Insight2Campaign, prompt: PromptCase) -> dict[str, Any]:
    """Exercise both fork outcomes without reading or fitting a calibrator."""
    _require(campaign.chunk_count == CHUNK_COUNT, "diagnostic horizon must be 30 chunks")
    _require(prompt.prompt_id == PROMPT_ID, "diagnostic prompt must be the frozen first prompt")
    _require(campaign.seeds == (SEED,), "diagnostic seed drifted")
    _require(not campaign._canonical_rollout_root_checkpoint(prompt, SEED).exists(),
             "diagnostic profiling requires a fresh trajectory, never a checkpoint replay")
    scope = hashlib.sha256(f"{PROTOCOL_ID}:{PROMPT_ID}:{SEED}".encode()).hexdigest()
    calls = {"draft": 0, "target": 1}
    branches: list[dict[str, Any]] = []
    carried = None
    started = time.perf_counter()
    with campaign._target_rpc_scope(scope):
        history, root = campaign._canonical_rollout_root(prompt=prompt, seed=SEED)
        for chunk_index in range(1, CHUNK_COUNT):
            noise = campaign._materialize_noise(prompt=prompt, seed=SEED, chunk_index=chunk_index)
            if carried is None:
                pending = campaign._generate(backend=campaign.draft, prompt=prompt, history=history,
                    noise=noise, output_stem=f"latents/diagnostic/chunk-{chunk_index:02d}-draft", capture_hidden=False)
                calls["draft"] += 1
            else:
                pending, carried = carried, None
                pending.segment.verify_context(history, pending.context)
                _require(pending.context.noise.descriptor_sha256 == noise.descriptor.descriptor_sha256,
                         "carried successor noise drifted")
            committed = pending
            choice = BRANCH_CHOICES.get(chunk_index)
            if choice is not None:
                successor_noise = campaign._materialize_noise(prompt=prompt, seed=SEED, chunk_index=chunk_index + 1)
                original_prefix = history.append(pending.segment)
                original = campaign._generate(backend=campaign.draft, prompt=prompt, history=original_prefix,
                    noise=successor_noise, output_stem=f"latents/diagnostic/fork-{chunk_index:02d}-original-successor", capture_hidden=False)
                candidate = campaign._generate(backend=campaign.target, prompt=prompt, history=history,
                    noise=noise, output_stem=f"latents/diagnostic/fork-{chunk_index:02d}-target", capture_hidden=False)
                corrected_prefix = history.append(candidate.segment)
                corrected = campaign._generate(backend=campaign.draft, prompt=prompt, history=corrected_prefix,
                    noise=successor_noise, output_stem=f"latents/diagnostic/fork-{chunk_index:02d}-corrected-successor", capture_hidden=False)
                calls["draft"] += 2
                calls["target"] += 1
                _require(original.context.noise.to_dict() == corrected.context.noise.to_dict(),
                         "fork successors must consume the exact same noise")
                _require(candidate.context.noise.to_dict() == pending.context.noise.to_dict(),
                         "fork candidates must consume the exact same noise")
                committed = pending if choice == "original" else candidate
                carried = original if choice == "original" else corrected
                branches.append({
                    "chunk_index": chunk_index, "fixed_choice": choice,
                    "decision_source": "frozen_diagnostic_route_not_rank_or_quality",
                    "parent_history_head_sha256": history.head_sha256,
                    "draft_candidate": _generated_dict(pending),
                    "target_candidate": _generated_dict(candidate),
                    "original_successor": _generated_dict(original),
                    "corrected_successor": _generated_dict(corrected),
                    "chosen_successor_latent_path": carried.segment.latent_path,
                })
            history = history.append(committed.segment)
            campaign._write_checkpoint(
                path=campaign.run_dir / "state/diagnostic" / f"chunk-{chunk_index:02d}.json",
                history=history, generated=committed,
                metadata={"kind": PROTOCOL_ID, "fixed_choice": choice,
                          "selection_policy_executed": False,
                          "carried_successor": carried.segment.latent_path if carried else None})
    elapsed = time.perf_counter() - started
    _require(history.next_chunk_index == CHUNK_COUNT, "diagnostic trajectory incomplete")
    _require(calls == {"draft": 31, "target": 3}, "diagnostic physical call count drifted")
    _require(tuple(s.chunk_index for s in history.segments if s.producer_role == "target") == (0, 20),
             "diagnostic committed route drifted")
    return {"schema": "stage_timing_fixed_route_trajectory_v1", "protocol_id": PROTOCOL_ID,
            "prompt_id": PROMPT_ID, "seed": SEED, "chunk_count": CHUNK_COUNT,
            "physical_generate_calls": calls, "committed_target_chunks": [0, 20],
            "generation_wall_seconds_instrumented": elapsed,
            "history": history.to_dict(), "root": _generated_dict(root), "branches": branches,
            "calibration_executed": False, "selection_policy_executed": False,
            "quality_measured": False, "formal_ours_result": False}


def _hardware(devices: Sequence[str]) -> dict[str, Any]:
    result = []
    for device in devices:
        text = subprocess.run(["nvidia-smi", "--id", device,
            "--query-gpu=uuid,name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=20).stdout.strip()
        fields = [part.strip() for part in text.split(",")]
        _require(len(fields) == 5 and fields[0].startswith("GPU-"), "GPU identity query failed")
        result.append(dict(zip(("uuid", "name", "memory_total", "driver_version", "pci_bus_id"), fields)))
    return {"hostname": platform.node(), "gpus": result}


def generate_continue_route(campaign, prompt, *, early, prefix_chunks=()):
    """Target root followed by 29 completed Draft blocks, with fixed prefix choices.

    The zero-budget early control still measures rank at layer14, but must
    continue the live forward. Its latent bytes can be compared with the native
    path in a separate run; neither path is a quality or speed benchmark.
    """
    from .insight2_campaign import ChunkGenerationRequest, GeneratedChunk
    from .early_protocol import StoppedDraft
    _require(isinstance(prefix_chunks, tuple) and len(set(prefix_chunks)) == len(prefix_chunks)
             and all(type(i) is int and 1 <= i < CHUNK_COUNT for i in prefix_chunks)
             and prefix_chunks == tuple(range(1, len(prefix_chunks)+1))
             and (early or not prefix_chunks), "invalid diagnostic prefix positions")
    _require(campaign.chunk_count == CHUNK_COUNT and campaign.seeds == (SEED,)
             and prompt.prompt_id == PROMPT_ID, "continuation diagnostic input drift")
    _require(not campaign._canonical_rollout_root_checkpoint(prompt, SEED).exists(),
             "diagnostic profiling requires a fresh trajectory")
    session = hashlib.sha256(f"{PROTOCOL_ID}:continue:{PROMPT_ID}:{SEED}".encode()).hexdigest()
    ranks = []
    started = time.perf_counter()
    with campaign._target_rpc_scope(session):
        history, root = campaign._canonical_rollout_root(prompt=prompt, seed=SEED)
        for index in range(1, CHUNK_COUNT):
            noise = campaign._materialize_noise(prompt=prompt, seed=SEED, chunk_index=index)
            stem = f"latents/diagnostic/chunk-{index:02d}-draft"
            if early:
                request = ChunkGenerationRequest(prompt=prompt, history=history, noise=noise,
                    output_latent_path=campaign._fresh_output_path(stem), capture_hidden=False)
                result = campaign.draft.generate_early(request, session_id=session, control={
                    "schema": "early_erank_control_v1", "prior_ranks": list(ranks),
                    "calibration_id": "diagnostic-fixed-choice" if prefix_chunks else "diagnostic-zero-budget",
                    "calibration_drop_scores": [0.0],
                    # Zero-score calibration selects the initial K blocks; the budget
                    # decrements monotonically, as enforced by the worker session.
                    "remaining_budget": max(0, len(prefix_chunks)-index+1),
                    "remaining_slots": CHUNK_COUNT-index})
                if index in prefix_chunks:
                    _require(isinstance(result, StoppedDraft), "fixed prefix probe did not stop")
                    _require(result.request == request and result.session_id == session,
                             "stopped prefix changed requested context")
                    rank = result.decision['effective_rank']
                    from .prefix_client import target_prefix, resume_prefix
                    prefix = target_prefix(campaign.target, request,
                        output_path=campaign._fresh_output_path(f'prefix/diagnostic/chunk-{index:02d}'))
                    noises = [campaign.noise_provider.materialize(
                        prompt_id=prompt.prompt_id+f'-suffix-noise-{step}', trajectory_seed=SEED,
                        chunk_index=index).tensor_path for step in (1,2,3)]
                    result = resume_prefix(campaign.draft, result, prefix, renoise_paths=noises)
                    _require(isinstance(result, GeneratedChunk) and result.context.role == 'draft',
                             'partial Target must finish as a Draft block')
                else:
                    _require(isinstance(result, GeneratedChunk), "zero-budget probe stopped Draft")
                    evidence = result.diagnostics.get("early_execution", {})
                    _require(evidence.get("stopped") is False and evidence.get("branch_snapshot_bytes") == 0,
                             "missing live continuation evidence")
                    rank = result.effective_rank
            else:
                result = campaign._generate(backend=campaign.draft, prompt=prompt, history=history,
                    noise=noise, output_stem=stem, capture_hidden=False)
            result.segment.verify_context(history, result.context)
            _require(result.context.noise == noise.descriptor, "continuation noise changed")
            ranks.append(rank if early else result.effective_rank)
            history = history.append(result.segment)
            campaign._write_checkpoint(path=campaign.run_dir/"state/diagnostic"/f"chunk-{index:02d}.json",
                history=history, generated=result, metadata={"early_continue": early,
                    "fixed_target_prefix": index in prefix_chunks,
                    "selection_policy_executed": False})
    return {"schema": "stage_timing_continue_trajectory_v1", "protocol_id": PROTOCOL_ID,
        "early_continue": early, "physical_generate_calls": {"draft": 29, "target": 1},
        "partial_target_calls": len(prefix_chunks), "fixed_prefix_chunks": list(prefix_chunks),
        "committed_target_chunks": [0], "history": history.to_dict(), "root": _generated_dict(root),
        "generation_wall_seconds_instrumented": time.perf_counter()-started,
        "quality_measured": False, "formal_ours_result": False}



def generate_early_target_route(campaign, prompt):
    """Fixed first14 full replacements through the production early route."""
    from .early_rank_route import materialize_early_rank, collect_early_trajectory
    from .correction_policy import SuddenDropRiskCalibrator
    _require(campaign.chunk_count == CHUNK_COUNT and campaign.seeds == (SEED,)
             and prompt.prompt_id == PROMPT_ID, "early Target diagnostic input drift")
    scope = hashlib.sha256(b"stage-timing-early-target-first14-v1").hexdigest()
    started = time.perf_counter()
    with campaign._target_rpc_scope(scope):
        key, _, selected, history = materialize_early_rank(campaign, prompt=prompt,
            seed=SEED, budget=14, calibrator=SuddenDropRiskCalibrator(
                "diagnostic-fixed-first14", (0.0,)))
    elapsed = time.perf_counter()-started
    _require(selected == tuple(range(1,15)), "fixed early Target positions drifted")
    audit = collect_early_trajectory(campaign, key=key, history=history,
                                   accepted=selected, proposed=selected)
    return {"schema":"stage_timing_early_target_trajectory_v1",
        "generation_wall_seconds_instrumented":elapsed, "history":history.to_dict(),
        "fixed_target_chunks":list(selected), "trajectory_audit":audit,
        "quality_measured":False, "formal_ours_result":False}


def run_probe(*, draft_artifact_spec_path: Path, target_artifact_spec_path: Path,
              boundary_config_path: Path = BOUNDARY_CONFIG,
              runtime_launcher: Callable[..., Any] = launch_correction_runtime,
              enforce_gpu_idle: bool = True, route_kind: str = "fork",
              prefix_gate_run: Path | None = None) -> dict[str, Any]:
    _require(route_kind in ("fork", "native_continue", "early_continue", "prefix_first14", "prefix_all", "early_target_first14"), "invalid timing route")
    _require(os.environ.get("CFEXP_STAGE_TIMING") == "1", "diagnostic requires CFEXP_STAGE_TIMING=1")
    run_dir, model_root = _run_dir(), _model_root()
    _require(not (run_dir / "campaign.json").exists(), "diagnostic requires a fresh attempt directory")
    devices = _visible_devices()
    repository = find_git_root(Path.cwd())
    prefix_chunks = (tuple(range(1, 15)) if route_kind == 'prefix_first14' else
                     tuple(range(1, 30)) if route_kind == 'prefix_all' else ())
    gate = None
    if prefix_chunks:
        from .prefix_quality_gate import verify_gate
        gate = verify_gate(prefix_gate_run, repository)
    bundle = load_bundle(repository)
    selected = _selected_prompts(repository, bundle.protocol)[0]
    _require(selected["id"] == PROMPT_ID, "frozen first prompt drifted")
    prompt = PromptCase(PROMPT_ID, selected["text"], "evaluation")
    config = load_run_config(boundary_config_path, repo_root=repository)
    protocol = load_protocol(config.protocol_path)
    _draft, draft_verified, draft_module = verify_draft_artifact_spec(
        draft_artifact_spec_path, model_root=model_root, repo_root=repository)
    _target, target_verified, target_module = verify_target_artifact_spec(
        target_artifact_spec_path, model_root=model_root, repo_root=repository)
    _require(draft_module == "cfexp.self_forcing_frame_adapter" and target_module == "cfexp.krea_chunk_backend",
             "diagnostic backend artifact binding drifted")
    draft_sha, _ = _sha256_file(draft_artifact_spec_path.resolve(strict=True))
    target_sha = target_artifact_spec_sha256(target_artifact_spec_path)
    preflight = {"schema": "stage_timing_probe_preflight_v1", "created_at": utc_now(),
        "run_kind": "diagnostic_fixed_route_profile", "git": collect_git_metadata(repository),
        "protocol_id": PROTOCOL_ID, "prompt_id": PROMPT_ID, "seed": SEED,
        "chunk_count": CHUNK_COUNT, "height": 480, "width": 832, "latent_frames": 90,
        "equivalent_rgb_frames": 357, "fps": 24, "video_decode_performed": False,
        "branch_choices": BRANCH_CHOICES if route_kind == "fork" else {},
        "route_kind": route_kind, "calibration_executed": False,
        "prefix_gate": gate, "fixed_prefix_chunks": list(prefix_chunks),
        "formal_ours_result": False, "quality_measured": False,
        "draft_artifact_spec_sha256": draft_sha, "target_artifact_spec_sha256": target_sha,
        "draft_verified_artifacts": draft_verified, "target_verified_artifacts": target_verified,
        "hardware": _hardware(devices)}
    _write_new_json(run_dir / "provenance/preflight.json", preflight)
    if enforce_gpu_idle:
        _assert_idle_pair(*devices)
    started = time.perf_counter()
    runtime = runtime_launcher(run_dir=run_dir, repo_root=repository,
        draft_artifact_spec_path=draft_artifact_spec_path, draft_artifact_spec_sha256=draft_sha,
        draft_verified_artifacts=draft_verified, target_artifact_spec_path=target_artifact_spec_path,
        target_artifact_spec_sha256=target_sha, target_verified_artifacts=target_verified,
        devices=devices, write_full_videos=False, chunk_count=CHUNK_COUNT,
        conditioning_prompts=[{"prompt_id": PROMPT_ID, "prompt": prompt.prompt}],
        shared_precomputed_conditioning=True)
    load_seconds = time.perf_counter() - started
    try:
        if enforce_gpu_idle:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0], target_device=devices[1], phase="diagnostic_fixed_route")
        # Constructor declares the frozen calibration population, but the probe
        # never invokes run_calibration and never fits or consumes a calibrator.
        campaign = Insight2Campaign(run_dir=run_dir, protocol_id=PROTOCOL_ID, seeds=(SEED,),
            calibration_prompts=tuple(PromptCase(str(p["prompt_id"]), str(p["prompt"]), "calibration") for p in protocol.calibration_prompts),
            evaluation_prompts=(prompt,), draft=runtime.draft, target=runtime.target,
            noise_provider=runtime.noise_provider, evaluator=runtime.evaluator, chunk_count=CHUNK_COUNT)
        route = (generate_early_target_route(campaign, prompt) if route_kind == "early_target_first14"
                 else generate_fixed_route(campaign, prompt) if route_kind == "fork"
                 else generate_continue_route(campaign, prompt,
                     early=route_kind != "native_continue", prefix_chunks=prefix_chunks))
        _write_new_json(run_dir / "raw/fixed-route.json", route)
        if enforce_gpu_idle:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0], target_device=devices[1], phase="diagnostic_full_target")
        reference = generate_full_target_trajectory(prompt_id=PROMPT_ID, prompt=prompt.prompt,
            target=runtime.target, noise_provider=runtime.noise_provider,
            seed=SEED, chunk_count=CHUNK_COUNT, duration_seconds=15)
        _write_new_json(run_dir / "raw/target-reference.json", {
            "schema": "stage_timing_target_reference_v1", "run_kind": "diagnostic_instrumented_reference",
            "formal_benchmark_result": False, "history": reference.history.to_dict(),
            "timing": dict(reference.result["timing"]), "quality_measured": False})
        if enforce_gpu_idle:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0], target_device=devices[1], phase="diagnostic_complete")
    finally:
        runtime.close()
    report = analyze_run(run_dir)
    _require(report["failed_record_count"] == 0, "diagnostic timing contains failed requests")
    worker_rows = [row for row in report["requests"] if row["source"].startswith("worker-")]
    draft_action = "generate" if route_kind in ("fork", "native_continue") else "generate_early"
    counts = {("draft", draft_action): 0, ("target", "generate"): 0, ("target", "generate_persistent"): 0}
    for row in worker_rows:
        key = (row["metadata"].get("role"), row["metadata"].get("action"))
        if key in counts:
            counts[key] += 1
    _require(counts == {("draft", draft_action): 31 if route_kind == "fork" else 29,
                        ("target", "generate"): 3 if route_kind == "fork" else 15 if route_kind == "early_target_first14" else 1,
                        ("target", "generate_persistent"): 30}, "timing worker coverage incomplete")
    for action, source in [('generate_target_prefix', 'worker-target'),
                           ('resume_early_target_prefix', 'worker-draft')]:
        _require(sum(row['source'] == source and row['metadata'].get('action') == action
                     for row in worker_rows) == len(prefix_chunks), 'prefix timing coverage incomplete')
    write_report(report, run_dir / "provenance/stage-timing-report")
    result = {"schema": RESULT_SCHEMA, "status": "succeeded", "run_kind": "diagnostic_fixed_route_profile",
        "protocol_id": PROTOCOL_ID, "route_kind": route_kind,
        "prompt_id": PROMPT_ID, "seed": SEED, "chunk_count": CHUNK_COUNT,
        "calibration_executed": False, "formal_ours_result": False, "quality_measured": False,
        "model_load_wall_seconds": load_seconds,
        "fixed_route_generation_wall_seconds_instrumented": route["generation_wall_seconds_instrumented"],
        "target_reference_generation_wall_seconds_instrumented": reference.result["timing"]["generation_wall_seconds"],
        "worker_profile_sync_wall_seconds_sum": sum(row["cuda_sync_wall_seconds"] for row in worker_rows),
        "timing_report": "provenance/stage-timing-report/report.json",
        "interpretation": "Fixed decisions exercise systems paths; instrumented times are not a speedup or quality claim. Nested spans and client/worker durations must not be added."}
    _write_new_json(run_dir / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-artifact-spec", type=Path, required=True)
    parser.add_argument("--target-artifact-spec", type=Path, required=True)
    parser.add_argument("--boundary-config", type=Path, default=BOUNDARY_CONFIG)
    parser.add_argument("--route-kind", choices=("fork", "native_continue", "early_continue", "prefix_first14", "prefix_all", "early_target_first14"), default="fork")
    parser.add_argument("--prefix-gate-run", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_probe(draft_artifact_spec_path=args.draft_artifact_spec,
            target_artifact_spec_path=args.target_artifact_spec, boundary_config_path=args.boundary_config,
            route_kind=args.route_kind, prefix_gate_run=args.prefix_gate_run)
    except (StageTimingProbeError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"stage-timing-probe: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
