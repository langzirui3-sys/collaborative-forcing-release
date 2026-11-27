"""Short native worker-lifetime diagnostic; not a performance comparison.

The formal runner owns process-group enforcement, sealing and publication.
This probe records safe Linux process identities before/after three real model
requests, closes its workers, and joins only helpers owned by this controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import launch_correction_runtime
from .correction_state import LatentHistory
from .erank_frame_campaign import verify_artifact_spec as verify_draft_artifact_spec
from .insight2_campaign import Insight2Campaign, PromptCase, _generated_dict
from .main_table_baseline_campaign import (_model_root, _run_dir, _selected_prompts,
    _sha256_file, _visible_devices, _write_new_json)
from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
from .main_table_v1 import PROMPT_IDS, load_bundle
from .metadata import collect_git_metadata, find_git_root, utc_now
from .stage_timing_probe import _hardware

PROTOCOL_ID = "worker-lifecycle-diagnostic-v1"
RESULT_SCHEMA = "worker_lifecycle_probe_v1"
PROMPT_ID, SEED = PROMPT_IDS[0], 20260911
BACKEND_CHUNK_COUNT = 30


class WorkerLifecycleProbeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerLifecycleProbeError(message)


def _members(snapshot: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    _require(snapshot.get("schema") == "linux_process_group_snapshot_v1"
             and snapshot.get("status") == "available", "complete Linux process snapshot required")
    members = snapshot.get("members")
    _require(isinstance(members, list), "process snapshot members missing")
    result = {}
    for member in members:
        _require(isinstance(member, Mapping), "invalid process snapshot member")
        _require(all(type(member.get(key)) is int for key in ("pid", "ppid", "pgid", "starttime")),
                 "invalid process identity")
        _require(member["pid"] not in result and member["pgid"] == snapshot["pgid"],
                 "duplicate or foreign process identity")
        result[member["pid"]] = member
    return result


def baseline_identities(snapshot: Mapping[str, Any], *, controller_pid: int) -> set[tuple[int, int]]:
    """Allow the controller and its uv ancestors, never pre-existing children."""
    members = _members(snapshot)
    _require(controller_pid in members, "controller absent from initial process snapshot")
    ancestors, cursor = set(), controller_pid
    while cursor in members:
        _require(cursor not in ancestors, "cycle in process ancestry")
        ancestors.add(cursor)
        cursor = members[cursor]["ppid"]
    _require(set(members) == ancestors, "initial group already contains non-ancestor processes")
    _require(all(members[pid].get("state") not in {"Z", "X"} for pid in ancestors),
             "initial process ancestry contains exited members")
    return {(pid, members[pid]["starttime"]) for pid in ancestors}


def assert_clean_snapshot(snapshot: Mapping[str, Any], *, baseline: set[tuple[int, int]],
                          controller_pid: int) -> None:
    members = _members(snapshot)
    _require(controller_pid in members, "controller absent from final process snapshot")
    unexpected = {(pid, member["starttime"]) for pid, member in members.items()} - baseline
    _require(not unexpected, "new process-group members remain after owned cleanup")
    _require(all(member.get("state") not in {"Z", "X"} for member in members.values()),
             "exited process-group members remain after owned cleanup")


def generate_lifecycle_requests(campaign: Insight2Campaign, prompt: PromptCase,
                                record: Callable[[str], Any]) -> dict[str, Any]:
    """Draft0, Draft1 and Target1 exercise both workers and parent-side noise."""
    history = LatentHistory(trajectory_id=hashlib.sha256(
        f"{PROTOCOL_ID}:{PROMPT_ID}:{SEED}".encode()).hexdigest(),
        prompt_id=PROMPT_ID, trajectory_seed=SEED)
    scope = hashlib.sha256(f"{PROTOCOL_ID}:target-rpc".encode()).hexdigest()
    generated = {}
    with campaign._target_rpc_scope(scope):
        noise0 = campaign._materialize_noise(prompt=prompt, seed=SEED, chunk_index=0)
        record("parent-noise-0")
        root = campaign._generate(backend=campaign.draft, prompt=prompt, history=history,
            noise=noise0, output_stem="latents/lifecycle/draft-00", capture_hidden=False)
        generated["draft_0"] = _generated_dict(root)
        record("draft-0-complete")
        history = history.append(root.segment)
        noise1 = campaign._materialize_noise(prompt=prompt, seed=SEED, chunk_index=1)
        record("parent-noise-1")
        for name, backend in (("draft", campaign.draft), ("target", campaign.target)):
            chunk = campaign._generate(backend=backend, prompt=prompt, history=history,
                noise=noise1, output_stem=f"latents/lifecycle/{name}-01", capture_hidden=False)
            generated[f"{name}_1"] = _generated_dict(chunk)
            record(f"{name}-1-complete")
    _require(generated["draft_1"]["context"]["history_head_sha256"]
             == generated["target_1"]["context"]["history_head_sha256"], "diagnostic histories differ")
    _require(generated["draft_1"]["context"]["noise"] == generated["target_1"]["context"]["noise"],
             "diagnostic request noise differs")
    kv_evidence = [generated[name]["diagnostics"].get("kv_reuse") for name in ("draft_0", "draft_1")]
    _require(all(isinstance(value, Mapping) and value.get("schema") in ("draft_kv_reuse_v1", "draft_kv_reuse_v2")
                 and all(type(value.get(key)) is int and value[key] >= 0
                         for key in ("reused_chunks", "replayed_chunks")) for value in kv_evidence),
             "native Draft cache evidence is missing or invalid")
    return {"schema": "worker_lifecycle_requests_v1", "physical_generate_calls": {"draft": 2, "target": 1},
        "parent_noise_materializations": 2, "noninitial_history_chunks": 1,
        "history": history.to_dict(), "generated": generated, "draft_native_kv_evidence": kv_evidence,
        "calibration_executed": False, "quality_measured": False, "selection_policy_executed": False}


def run_probe(*, draft_artifact_spec_path: Path, target_artifact_spec_path: Path,
              runtime_launcher: Callable[..., Any] = launch_correction_runtime,
              snapshotter: Callable[[int], Mapping[str, Any]] | None = None,
              helper_shutdown: Callable[..., Any] | None = None,
              enforce_gpu_idle: bool = True) -> dict[str, Any]:
    _require(os.environ.get("CFEXP_STAGE_TIMING") in {None, "", "0"}, "stage timing must be disabled")
    _require(os.environ.get("CFEXP_DRAFT_KV_REUSE") == "1", "native KV reuse must be enabled")
    run_dir, model_root = _run_dir(), _model_root()
    _require(not any((parent / marker).exists() for parent in (run_dir, *run_dir.parents)
                     for marker in ("_SUCCESS.json", "_FAILED.json", "artifact-index.json")),
             "cannot write a lifecycle probe inside a sealed run")
    _require(not (run_dir / "provenance/lifecycle").exists() and not (run_dir / "result.json").exists(),
             "lifecycle probe requires a fresh attempt")
    if snapshotter is None:
        from .runner import _snapshot_process_group
        snapshotter = _snapshot_process_group
    if helper_shutdown is None:
        from .runtime_helpers import shutdown_owned_runtime_helpers
        helper_shutdown = shutdown_owned_runtime_helpers
    devices, repository = _visible_devices(), find_git_root(Path.cwd())
    bundle = load_bundle(repository)
    selected = _selected_prompts(repository, bundle.protocol)[0]
    _require(selected["id"] == PROMPT_ID, "frozen prompt differs")
    prompt = PromptCase(PROMPT_ID, selected["text"], "evaluation")
    _, draft_verified, draft_module = verify_draft_artifact_spec(
        draft_artifact_spec_path, model_root=model_root, repo_root=repository)
    _, target_verified, target_module = verify_target_artifact_spec(
        target_artifact_spec_path, model_root=model_root, repo_root=repository)
    _require((draft_module, target_module) == ("cfexp.self_forcing_frame_adapter", "cfexp.krea_chunk_backend"),
             "frozen native backend differs")
    draft_sha, _ = _sha256_file(draft_artifact_spec_path.resolve(strict=True))
    target_sha = target_artifact_spec_sha256(target_artifact_spec_path)
    _write_new_json(run_dir / "provenance/preflight.json", {"schema": "worker_lifecycle_preflight_v1",
        "created_at": utc_now(), "git": collect_git_metadata(repository), "hardware": _hardware(devices),
        "protocol_id": PROTOCOL_ID, "prompt_id": PROMPT_ID, "seed": SEED,
        "draft_artifact_spec_sha256": draft_sha, "target_artifact_spec_sha256": target_sha,
        "backend_chunk_count": BACKEND_CHUNK_COUNT, "stage_timing_enabled": False,
        "quality_measured": False, "calibration_executed": False, "formal_ours_result": False})
    snapshots: list[str] = []
    def record(phase: str) -> Mapping[str, Any]:
        snapshot = dict(snapshotter(os.getpgrp()))
        path = f"provenance/lifecycle/{len(snapshots):02d}-{phase}.json"
        _write_new_json(run_dir / path, {"schema": "worker_lifecycle_snapshot_v1", "phase": phase,
            "created_at": utc_now(), "controller_pid": os.getpid(), "snapshot": snapshot})
        snapshots.append(path)
        return snapshot
    if enforce_gpu_idle:
        _assert_idle_pair(*devices)
    baseline = baseline_identities(record("before-runtime"), controller_pid=os.getpid())
    runtime, route, cleanup, errors = None, None, None, []
    final_clean = False
    try:
        runtime = runtime_launcher(run_dir=run_dir, repo_root=repository,
            draft_artifact_spec_path=draft_artifact_spec_path, draft_artifact_spec_sha256=draft_sha,
            draft_verified_artifacts=draft_verified, target_artifact_spec_path=target_artifact_spec_path,
            target_artifact_spec_sha256=target_sha, target_verified_artifacts=target_verified,
            devices=devices, write_full_videos=False, chunk_count=BACKEND_CHUNK_COUNT,
            conditioning_prompts=[{"prompt_id": PROMPT_ID, "prompt": prompt.prompt}],
            shared_precomputed_conditioning=True)
        ready = _members(record("workers-ready"))
        worker_pids = [runtime.draft.process.pid, runtime.target.process.pid]
        _require(len(set(worker_pids)) == 2 and all(pid in ready for pid in worker_pids),
                 "native workers absent from the owned process group")
        if enforce_gpu_idle:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                target_device=devices[1], phase="lifecycle-start")
        campaign = Insight2Campaign(run_dir=run_dir, protocol_id=PROTOCOL_ID, seeds=(SEED,),
            calibration_prompts=(PromptCase("unused-lifecycle-calibration", "Unused diagnostic placeholder", "calibration"),),
            evaluation_prompts=(prompt,), draft=runtime.draft, target=runtime.target,
            noise_provider=runtime.noise_provider, evaluator=runtime.evaluator, chunk_count=BACKEND_CHUNK_COUNT)
        route = generate_lifecycle_requests(campaign, prompt, record)
        _write_new_json(run_dir / "raw/requests.json", route)
        if enforce_gpu_idle:
            _assert_worker_pair_owns_gpus(runtime=runtime, draft_device=devices[0],
                target_device=devices[1], phase="lifecycle-complete")
    except BaseException as error:
        errors.append(error)
    finally:
        if runtime is not None:
            try:
                record("before-runtime-close")
            except BaseException as error:
                errors.append(error)
            try:
                runtime.close()
            except BaseException as error:
                errors.append(error)
        try:
            record("after-runtime-close")
        except BaseException as error:
            errors.append(error)
        try:
            cleanup = helper_shutdown(label="lifecycle-controller")
        except BaseException as error:
            errors.append(error)
        try:
            final = record("after-controller-cleanup")
            assert_clean_snapshot(final, baseline=baseline, controller_pid=os.getpid())
            final_clean = True
        except BaseException as error:
            errors.append(error)
    result = {"schema": RESULT_SCHEMA, "status": "failed" if errors else "succeeded",
        "protocol_id": PROTOCOL_ID, "run_kind": "diagnostic_worker_lifecycle",
        "formal_ours_result": False, "quality_measured": False, "calibration_executed": False,
        "performance_measured": False, "replaces_eight_cell_comparison": False,
        "request_counts": route["physical_generate_calls"] if route else None,
        "process_snapshots": snapshots, "owned_helper_cleanup": cleanup,
        "post_cleanup_snapshot_clean": final_clean,
        "formal_runner_exit_verification_pending": True,
        "error_types": [type(error).__name__ for error in errors]}
    _write_new_json(run_dir / "result.json", result)
    if errors:
        raise WorkerLifecycleProbeError("lifecycle validation failed: " + ", ".join(result["error_types"])) from errors[0]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-artifact-spec", type=Path, required=True)
    parser.add_argument("--target-artifact-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(draft_artifact_spec_path=args.draft_artifact_spec,
                           target_artifact_spec_path=args.target_artifact_spec)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"worker-lifecycle-probe: failed ({type(error).__name__})", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
