"""Formal Boundary-gated Ours main-table campaign.

This adapter deliberately reuses :class:`Insight2Campaign` for trajectory
semantics.  It adds only the main-table population, timing boundary, deferred
official VBench-Long scoring, and strict result projection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .calibration_receipt import (
    CalibrationReceiptError,
    load_sealed_calibration_ranks,
    restore_calibration_receipt,
    seal_calibration_receipt,
    verify_calibration_receipt,
)
from .rolling_campaign import verify_draft_spec as verify_rolling_spec, prepare_launch as prepare_rolling_launch
from .baseline_population import model_pair_for_draft, load_baseline_population
from .vbench_short import OfficialVBenchScorer
from .correction_policy import random_p8_k2_positions, random_positions
from .correction_artifacts import target_artifact_spec_sha256, verify_target_artifact_spec
from .correction_runtime import (
    VBENCH_LONG_PRIMARY_QUALITY_METRIC,
    launch_correction_runtime,
)
from .erank_frame_campaign import verify_artifact_spec as verify_draft_artifact_spec
from .insight2_campaign import Insight2Campaign, PromptCase, QualityMeasurement
from .insight2_protocol import load_protocol
from .insight2_run import _calibration_receipt_binding, load_run_config
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
from .main_table_paper_figures import RAW_SCHEMA
from .main_table_v1 import PROMPT_IDS, QUALITY_DIMENSIONS, load_bundle
from .duration_sweep import (
    duration_profile,
    RESULT_SCHEMA as DURATION_RESULT_SCHEMA,
    RAW_SCHEMA as DURATION_RAW_SCHEMA,
)
from .metadata import collect_git_metadata, find_git_root, utc_now
from .native_vae_component import (OVERLAP_READY_SCOPE, SERIAL_READY_SCOPE,
    campaign_model, completed_render_binding)
from . import optimized_duration_control as optimized_control
from .vbench_long import OfficialVBenchLongScorer, VBenchLongVideoRequest


METHOD_ID = "ours"
from .paper_population import generation_seed, CALIBRATION_SEED
SEED = generation_seed()
CHUNK_COUNT = 120
BUDGET = 6
WINDOW_RADIUS_CHUNKS = 4
MINIMUM_GAP_CHUNKS = 8
MAXIMUM_GAP_CHUNKS = 20
STRATEGY = "online_lookahead_coverage_boundary_gated"
BOUNDARY_CONFIG = Path(
    "experiments/insight2-run-60s-h800-lookahead-boundary-v1.config.json"
)
PREFLIGHT_SCHEMA = "main_table_ours_campaign_preflight_v1"
METHOD_RESULT_SCHEMA = "main_table_ours_prompt_result_v1"


class OursCampaignError(RuntimeError):
    """Raised when the Ours campaign cannot prove the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OursCampaignError(message)


def _finite_positive(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"{label} must be finite and positive",
    )
    return float(value)


def _strict_flag(name: str) -> bool:
    value = os.environ.get(name)
    _require(value in {None, "", "1"}, f"{name} must be unset or exactly '1'")
    return value == "1"


def _assert_idle_pair(draft_device: str, target_device: str) -> None:
    _require(
        not _gpu_compute_pids(_gpu_uuid(draft_device))
        and not _gpu_compute_pids(_gpu_uuid(target_device)),
        "selected Ours GPU pair is not idle immediately before launch",
    )


def _ownership_process_record(pid: int) -> dict[str, Any]:
    """Failure-only bounded process metadata; never read argv or environment."""
    record: dict[str, Any] = {"pid": pid}
    try:
        text = (Path('/proc') / str(pid) / 'status').read_text()
        for line in text.splitlines():
            key, _, value = line.partition(':')
            if key in {'Name', 'State'}:
                record[key.lower()] = value.strip()[:128]
            elif key == 'PPid' and value.strip().isdigit():
                record['ppid'] = int(value.strip())
        record['proc_status_available'] = True
    except (OSError, UnicodeError):
        record['proc_status_available'] = False
    return record


def _ownership_failure_snapshot(*, runtime: Any, draft_pid: Any, target_pid: Any,
                                gpu_uuid: str | None, observed: set[int] | None,
                                phase: str) -> dict[str, Any]:
    expected = {p for p in (draft_pid, target_pid) if type(p) is int and p > 0}
    workers = {}
    for role, pid in (('draft', draft_pid), ('target', target_pid)):
        row = _ownership_process_record(pid) if type(pid) is int and pid > 0 else {
            'pid': None, 'invalid_pid_type': type(pid).__name__}
        process = getattr(getattr(runtime, role, None), 'process', None)
        poll = getattr(process, 'poll', None)
        try:
            code = poll() if callable(poll) else 'unavailable'
            row['returncode'] = code if code is None or type(code) is int else 'unavailable'
        except (OSError, RuntimeError):
            row['returncode'] = 'unavailable'
        workers[role] = row
    return dict(schema='single_gpu_ownership_failure_v1', phase=phase,
        observed_unix=time.time(), gpu_uuid=gpu_uuid,
        expected_worker_pids=sorted(expected),
        observed_compute_pids=sorted(observed) if observed is not None else None,
        missing_worker_pids=sorted(expected-observed) if observed is not None else None,
        unexpected_compute_pids=sorted(observed-expected) if observed is not None else None,
        workers=workers,
        unexpected_processes=[_ownership_process_record(p) for p in sorted(observed-expected)]
            if observed is not None else [])


def _assert_worker_pair_owns_gpus(
    *, runtime: Any, draft_device: str, target_device: str, phase: str
) -> None:
    draft_pid = getattr(getattr(runtime.draft, "process", None), "pid", None)
    target_pid = getattr(getattr(runtime.target, "process", None), "pid", None)
    if draft_device == target_device:
        valid = (type(draft_pid) is int and type(target_pid) is int
                 and min(draft_pid, target_pid) > 0 and draft_pid != target_pid)
        uuid = _gpu_uuid(draft_device) if valid else None
        observed = _gpu_compute_pids(uuid) if valid else None
        if not valid or observed != {draft_pid, target_pid}:
            snapshot = _ownership_failure_snapshot(runtime=runtime, draft_pid=draft_pid,
                target_pid=target_pid, gpu_uuid=uuid, observed=observed, phase=phase)
            raise OursCampaignError(
                f"single-GPU ownership differs from the Draft and Target workers during {phase}; "
                'ownership_snapshot=' + json.dumps(snapshot, sort_keys=True))
        return
    _require(
        isinstance(draft_pid, int)
        and draft_pid > 0
        and isinstance(target_pid, int)
        and target_pid > 0
        and _gpu_compute_pids(_gpu_uuid(draft_device)) <= {draft_pid}
        and _gpu_compute_pids(_gpu_uuid(target_device)) <= {target_pid},
        f"an Ours GPU acquired an unrelated compute process during {phase}",
    )


def _prompt_cases(values: Sequence[Mapping[str, Any]], *, split: str) -> tuple[PromptCase, ...]:
    return tuple(
        PromptCase(prompt_id=str(item["prompt_id"]), prompt=str(item["prompt"]), split=split)
        for item in values
    )


def _receipt_policy(
    *,
    smoke_prompt_id: str | None,
    run_dir: Path,
    binding: Mapping[str, Any],
) -> tuple[bool, Any | None]:
    seal = _strict_flag("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SEAL")
    require = _strict_flag("CFEXP_INSIGHT2_REQUIRE_CALIBRATION_RECEIPT")
    source_text = os.environ.get("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SOURCE")
    digest = os.environ.get("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SHA256")
    if smoke_prompt_id is not None:
        _require(seal and not require and not source_text and not digest, "Ours smoke must seal a new same-revision calibration receipt")
        return True, None
    source, digest = _frozen_calibration_source(seal, require, source_text, digest)
    try:
        verified = verify_calibration_receipt(
            source_run_dir=source,
            expected_binding=binding,
            expected_receipt_sha256=digest,
        )
        restored = restore_calibration_receipt(
            source_run_dir=source,
            destination_run_dir=run_dir,
            expected_binding=binding,
            expected_receipt_sha256=digest,
            preverified=verified,
        )
    except CalibrationReceiptError as error:
        raise OursCampaignError(f"calibration receipt verification failed: {error}") from error
    return False, restored


def _frozen_calibration_source(seal, require, source_text, digest):
    _require(not seal and require and bool(source_text) and bool(digest),
             "formal Ours requires an externally frozen calibration receipt")
    source = Path(str(source_text)).expanduser()
    _require(source.is_absolute() and not source.is_symlink(), "calibration receipt source must be an absolute non-symlink path")
    return source, str(digest)


def _read_sealed_calibration(binding: Mapping[str, Any]):
    source, digest = _frozen_calibration_source(
        _strict_flag("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SEAL"),
        _strict_flag("CFEXP_INSIGHT2_REQUIRE_CALIBRATION_RECEIPT"),
        os.environ.get("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SOURCE"),
        os.environ.get("CFEXP_INSIGHT2_CALIBRATION_RECEIPT_SHA256"),
    )
    try:
        return load_sealed_calibration_ranks(
            source_run_dir=source, expected_binding=binding, expected_receipt_sha256=digest)
    except CalibrationReceiptError as error:
        raise OursCampaignError(f"sealed rank calibration failed: {error}") from error


RuntimeLauncher = Callable[..., Any]
ScorerFactory = Callable[..., Any]
CampaignFactory = Callable[..., Any]
ReceiptPolicy = Callable[..., tuple[bool, Any]]


def run_campaign(
    *,
    draft_artifact_spec_path: Path,
    draft_model_id: str = "self_forcing_1.3b",
    target_artifact_spec_path: Path,
    boundary_config_path: Path,
    vbench_checkout: Path,
    vbench_python: Path,
    vbench_asset_manifest: Path,
    vbench_asset_spec: Path,
    runtime_launcher: RuntimeLauncher = launch_correction_runtime,
    scorer_factory: ScorerFactory = OfficialVBenchLongScorer,
    campaign_factory: CampaignFactory = Insight2Campaign,
    receipt_policy: ReceiptPolicy = _receipt_policy,
    enforce_gpu_idle: bool = True,
    smoke_prompt_id: str | None = None,
    duration_seconds: int | None = None,
    optimized_duration_control: bool = False,
    selection_strategy: str = "erank",
    random_repeat: int = 0,
    baseline_population_v2: bool = False,
    burst_calibration_revision: str | None = None,
    record_video_ready_timing: bool = False,
    overlap_native_vae: bool = False,
    native_burst_length: int = 4,
    native_burst_observation_horizon: int = 8,
    native_burst_observation_offset: int = 0,
    native_opening_target_burst: bool = False,
    native_split_target_burst: bool = False,
    native_split_burst_count: int = 2,
    native_final_window_at_tail: bool = False,
    native_motion_trigger: bool = False,
    sealed_calibration_ranks: bool = False,
    native_random_p8_k2: bool = False,
) -> dict[str, Any]:
    _require(selection_strategy in {"erank", "random", "rank_burst"}, "unknown selection strategy")
    burst = selection_strategy == "rank_burst"
    from .rank_burst import NATIVE_BURST_WORD, SPLIT_BURST_WORD, valid_split_burst_count
    _require(type(native_burst_length) is int and native_burst_length in NATIVE_BURST_WORD
             and (native_burst_length == 4
                  or (native_burst_length == 1 and native_split_target_burst)
                  or (native_burst_length == 2 and burst and overlap_native_vae
                      and (duration_seconds == 15 or native_split_target_burst))
                  or (native_burst_length == 3 and native_split_target_burst
                      and duration_seconds == 30)
                  or (native_burst_length == 8 and burst and overlap_native_vae
                      and draft_model_id == "longlive_1.3b" and duration_seconds == 15
                      and smoke_prompt_id is not None)),
             "native burst length requires its declared backend, horizon and experiment scope")
    _require(type(native_burst_observation_horizon) is int and
             (native_burst_observation_horizon == 8 or
              (native_burst_observation_horizon == 2 and burst and native_burst_length == 4
               and draft_model_id == "longlive_1.3b" and duration_seconds == 30 and overlap_native_vae)),
             "early burst is a 30-second LongLive four-block overlap control")
    _require(type(native_burst_observation_offset) is int and
             (native_burst_observation_offset == 0 or
              (native_burst_observation_offset == 20 and burst and native_burst_length == 4
               and native_burst_observation_horizon == 8 and draft_model_id == "longlive_1.3b"
               and duration_seconds == 60 and overlap_native_vae and smoke_prompt_id is not None)),
             "delayed burst requires the 60-second LongLive same-budget smoke")
    _require(type(native_opening_target_burst) is bool and
             (not native_opening_target_burst or
              (burst and native_burst_length == 4 and native_burst_observation_horizon == 8
               and native_burst_observation_offset == 0 and draft_model_id == "longlive_1.3b"
               and duration_seconds in (15, 30, 60) and overlap_native_vae)),
             "opening Target burst requires the LongLive same-budget overlap path")
    _require(type(native_split_target_burst) is bool and
             (not native_split_target_burst or
              (burst and native_burst_length in (1, 2, 3) and native_burst_observation_horizon == 8
               and native_burst_observation_offset == 0 and not native_opening_target_burst
               and draft_model_id == "longlive_1.3b" and duration_seconds in (15, 30, 60)
               and overlap_native_vae)),
             "split Target bursts require the native LongLive overlap path")
    _require(valid_split_burst_count(native_split_burst_count, burst_length=native_burst_length,
                                    seconds=duration_seconds, split=native_split_target_burst),
             "unsupported native Target burst count for the declared width and horizon")
    _require(type(native_final_window_at_tail) is bool and
             (not native_final_window_at_tail or
              (native_split_target_burst and native_split_burst_count == 3)),
             "tail-aligned observation requires the 30-second LongLive K6 route")
    _require(type(native_motion_trigger) is bool and
             (not native_motion_trigger or (native_split_target_burst and native_split_burst_count == 3
                                            and not native_final_window_at_tail)),
             "motion trigger requires the original 30-second LongLive K6 route")
    native_random = selection_strategy == 'random' and overlap_native_vae
    _require(type(record_video_ready_timing) is bool and
             (not record_video_ready_timing or burst or native_random),
             "video-ready measurement requires the native burst path or direct native Random generation")
    _require(not overlap_native_vae or ((burst or native_random) and record_video_ready_timing),
             "native renderer requires measured native burst generation")
    native_render = campaign_model(draft_model_id, overlap=overlap_native_vae)
    from . import native_single_gpu
    single_gpu = native_single_gpu.validate(draft_model_id, native_render,
                                             enforce_idle=enforce_gpu_idle)
    from .paper_population import validate_execution
    validate_execution(single_gpu=single_gpu, seconds=duration_seconds, full_population=baseline_population_v2)
    _require(not native_random or (native_render in ('longlive','rolling') and baseline_population_v2
                                   and duration_seconds in (15,30,60)),
             'native Random requires a frozen native model/duration population')
    _require(type(native_random_p8_k2) is bool and (
        not native_random_p8_k2 or (native_random and native_render == 'rolling'
                                   and duration_seconds == 15 and random_repeat == 0)),
        'Random P8/K2 requires the frozen 15-second native Rolling Random cohort')
    from .longlive_accepted_prefill import enabled as longlive_prefill_enabled
    longlive_prefill = longlive_prefill_enabled()
    _require(not longlive_prefill or (native_render == 'longlive' and duration_seconds == 30
                                    and native_burst_length == 2
                                    and native_split_target_burst
                                    and native_split_burst_count == 2 and enforce_gpu_idle),
             'LongLive accepted prefill requires the audited 30-second two-window route')
    timing_scope = OVERLAP_READY_SCOPE if native_render else SERIAL_READY_SCOPE
    if single_gpu:
        timing_scope = native_single_gpu.TIMING_SCOPE
    render_worker = {}
    if native_render:
        from .accepted_stream_worker import process_factory
        render_worker['process_factory'] = process_factory
    _require((burst_calibration_revision is not None) == burst,
             "native-model burst requires an explicitly pinned calibration revision")
    _require(not burst or (baseline_population_v2 and not optimized_duration_control
                           and draft_model_id in {"longlive_1.3b", "rolling_forcing_1.3b"}),
             "native-model burst requires the LongLive or Rolling frozen population")
    _require(type(sealed_calibration_ranks) is bool
             and (not sealed_calibration_ranks or (burst and native_render in ("longlive", "rolling"))),
             "sealed rank startup requires a native LongLive or Rolling burst")
    _require(type(random_repeat) is int and random_repeat >= 0, "invalid Random repeat")
    _require(selection_strategy == "random" or random_repeat == 0, "repeat applies only to Random")
    _require(not (selection_strategy == "random" and optimized_duration_control),
             "Random cannot use the optimized Ours control")
    _require(selection_strategy != "random" or duration_seconds is None or baseline_population_v2,
             "Random duration profiles require the frozen baseline population")
    _require(not (baseline_population_v2 and optimized_duration_control),
             "baseline population is separate from the optimized duration control")
    _require(not baseline_population_v2 or duration_seconds in (15, 30, 60),
             "baseline v2 requires a duration")
    method_id = "random" if selection_strategy == "random" else METHOD_ID
    from .rolling_prefix_execution import enabled as prefix_enabled
    rolling_prefix_transactions=prefix_enabled()
    from .rolling_committed_execution import enabled as committed_enabled
    rolling_committed_prefix=committed_enabled()
    _require(not (rolling_prefix_transactions and rolling_committed_prefix),
             'Rolling committed prefix and rollback execution are mutually exclusive')
    _require(not (rolling_prefix_transactions or rolling_committed_prefix) or (single_gpu and burst
             and draft_model_id=='rolling_forcing_1.3b' and selection_strategy=='rank_burst'
             and duration_seconds==30 and native_burst_length==4
             and native_burst_observation_horizon==8 and native_burst_observation_offset==0
             and not native_opening_target_burst and not native_split_target_burst
             and not native_motion_trigger),
             'Rolling prefix execution requires the verified single-GPU 30-second eRank B4 configuration')
    native_budget = native_split_burst_count * native_burst_length if native_split_target_burst else native_burst_length
    strategy = (f"committed_rank_burst{native_burst_length}_probe{native_burst_observation_horizon}_k{native_budget}" if burst else
                "random" if selection_strategy == "random" else STRATEGY)
    if native_random:
        strategy = 'native_random_p8_k2' if native_random_p8_k2 else 'native_random_direct'
    route_repeat = random_repeat if selection_strategy == "random" else None
    if optimized_duration_control:
        optimized_control.validate_invocation(method=method_id, seconds=duration_seconds, smoke=smoke_prompt_id)
    profile = duration_profile(duration_seconds) if duration_seconds is not None else None
    chunk_count = profile.chunk_count if profile else CHUNK_COUNT
    run_dir = _run_dir()
    model_root = _model_root()
    draft_device, target_device = native_single_gpu.devices(single_gpu, _visible_devices)
    repository_root = find_git_root(Path.cwd())
    population_binding = None
    benchmark = "VBench-Long"
    if baseline_population_v2:
        selected, population_binding = load_baseline_population(repository_root, model_pair=model_pair_for_draft(draft_model_id))
        benchmark = "VBench" if duration_seconds in (15, 30) else "VBench-Long"
        if benchmark == "VBench" and scorer_factory is OfficialVBenchLongScorer:
            scorer_factory = OfficialVBenchScorer
    else:
        bundle = load_bundle(repository_root)
        selected = _selected_prompts(repository_root, bundle.protocol)
        _require(tuple(item["id"] for item in selected) == PROMPT_IDS,
                 "active prompt selection drifted")
    population_ids = tuple(item["id"] for item in selected)
    if smoke_prompt_id is not None:
        _require(smoke_prompt_id in population_ids, "smoke prompt is outside the population")
        selected = [item for item in selected if item["id"] == smoke_prompt_id]
    active_prompt_ids = tuple(item["id"] for item in selected)
    metric_key = "official_vbench" if benchmark == "VBench" else "official_vbench_long"
    duration_metadata = profile.metadata() if profile else None
    if baseline_population_v2:
        duration_metadata.update(experiment_id="erank-baseline24-v1", benchmark=benchmark)
        if selection_strategy == "random":
            for key in ("calibration_chunk_count", "calibration_policy"):
                duration_metadata.pop(key, None)
            duration_metadata["target_replacement_budget"] = duration_metadata.pop("ours_proposal_budget")
            if native_random_p8_k2:
                duration_metadata["random_burst_observation_horizon"] = 8

    # Keep the original C10 calibration contract and receipt binding at N=120.
    # Short evaluation trajectories use a separate campaign/runtime afterwards.
    calibration_dir = run_dir / "calibration-60s" if profile else run_dir
    calibration_dir.mkdir(parents=True, exist_ok=True)
    budget = native_budget if burst else (profile.proposal_budget if profile else BUDGET)
    if burst:
        duration_metadata.update(experiment_id=f"native-model-e4-p{native_burst_observation_horizon}-k{budget}-v1",
            ours_proposal_budget=budget,
            budget_rule=f"one_target_root_and_one_{NATIVE_BURST_WORD[native_burst_length]}_chunk_burst",
            calibration_policy="explicit_sealed_model_specific_C10",
            calibration_code_revision=burst_calibration_revision, observation_horizon=native_burst_observation_horizon)
        if native_split_target_burst:
            burst_word = SPLIT_BURST_WORD[native_split_burst_count]
            width = NATIVE_BURST_WORD[native_burst_length]
            duration_metadata.update(split_target_burst=True,
                                     budget_rule=f"one_target_root_and_{burst_word}_{width}_chunk_bursts")
            if native_split_burst_count != 2:
                duration_metadata["split_burst_count"] = native_split_burst_count
            duration_metadata["experiment_id"] += "-split"
            strategy += "_split"
        if native_final_window_at_tail:
            duration_metadata["final_window_at_tail"] = True
            duration_metadata["experiment_id"] += "-tail"
            strategy += "_tail"
        if native_motion_trigger:
            from .longlive_motion import MOTION_ENERGY_FLOOR
            duration_metadata.update(motion_trigger=True, motion_energy_floor=MOTION_ENERGY_FLOOR)
            duration_metadata["experiment_id"] += "-motion"
            strategy += "_motion"
        if longlive_prefill:
            duration_metadata["longlive_accepted_prefill"] = True
            duration_metadata["experiment_id"] += "-prefill"
            strategy += "_prefill"
        if native_burst_observation_offset:
            duration_metadata["observation_offset"] = native_burst_observation_offset
            duration_metadata["experiment_id"] += f"-offset{native_burst_observation_offset}"
            strategy += f"_offset{native_burst_observation_offset}"
        if native_opening_target_burst:
            duration_metadata["opening_target_burst"] = True
            duration_metadata["experiment_id"] += "-opening"
            strategy += "_opening"
    _require(not (profile and smoke_prompt_id and duration_seconds != 60 and selection_strategy == "erank"),
             "duration smoke must calibrate at 60s; short formal runs reuse its receipt")
    config = load_run_config(boundary_config_path, repo_root=repository_root)
    protocol = load_protocol(config.protocol_path)
    from .paper_population import assert_calibration_compatible
    paper_calibration = assert_calibration_compatible(
        repository_root, protocol, calibration_run=smoke_prompt_id is not None)
    _require(protocol.raw["models"]["draft"]["id"] == draft_model_id,
             "Draft worker selection differs from calibration protocol model")
    if draft_model_id in {"longlive_1.3b", "rolling_forcing_1.3b"}:
        _require(config.draft_artifact_spec_path.resolve() == draft_artifact_spec_path.resolve(),
                 "Native calibration config differs from selected Draft artifact spec")
    _require(protocol.raw["protocol_id"] == "insight2-correction-long60-h800-lookahead-boundary-v1", "wrong Boundary protocol")
    _require(tuple(protocol.raw["design"]["seeds"]) == (CALIBRATION_SEED,), "Boundary calibration seed drifted")
    if baseline_population_v2:
        evaluation_values = [{"prompt_id": item["id"], "prompt": item["text"]} for item in selected]
        calibration_ids = {item["prompt_id"] for item in protocol.calibration_prompts}
        calibration_texts = {item["prompt"] for item in protocol.calibration_prompts}
        _require(not (calibration_ids & set(active_prompt_ids)) and
                 not (calibration_texts & {item["text"] for item in selected}),
                 "evaluation population overlaps calibration")
    else:
        eval_by_id = {str(item["prompt_id"]): item for item in protocol.eval_prompts}
        evaluation_values = [eval_by_id[prompt_id] for prompt_id in active_prompt_ids]
        _require([str(item["prompt"]) for item in evaluation_values]
                 == [str(item["text"]) for item in selected],
                 "main-table prompts differ from the audited Boundary protocol")
    try:
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
                draft_artifact_spec_path, model_root=model_root, repo_root=repository_root
            )
        _target_spec, target_verified, target_module = verify_target_artifact_spec(
            target_artifact_spec_path, model_root=model_root, repo_root=repository_root
        )
    except BaseException as error:
        raise OursCampaignError(f"artifact preflight failed: {error}") from error
    _require(draft_module == {"longlive_1.3b": "cfexp.longlive_adapter", "self_forcing_1.3b": "cfexp.self_forcing_frame_adapter", "rolling_forcing_1.3b": "cfexp.rolling_policy_worker"}[draft_model_id], "wrong Draft adapter")
    _require(target_module == "cfexp.krea_chunk_backend", "wrong Target backend")
    draft_spec_sha256, _ = _sha256_file(draft_artifact_spec_path.resolve(strict=True))
    target_spec_sha256 = target_artifact_spec_sha256(target_artifact_spec_path)
    protocol_sha256, _ = _sha256_file(config.protocol_path)
    binding = _calibration_receipt_binding(
        repo_root=repository_root,
        config=config,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        draft_spec_sha256=draft_spec_sha256,
        target_spec_sha256=target_spec_sha256,
        draft_verified_artifacts=draft_verified,
        target_verified_artifacts=target_verified,
        phase_seeds=(CALIBRATION_SEED,),
    )
    if optimized_duration_control:
        binding = optimized_control.frozen_control_binding(binding)
    if burst:
        from .branch_gate_probe import calibration_control_binding
        binding = calibration_control_binding(binding, burst_calibration_revision)
    rank_proof = None
    calibrator = None
    calibration_wall_seconds = 0.0
    if sealed_calibration_ranks:
        started = time.perf_counter()
        calibrator, rank_proof = _read_sealed_calibration(binding)
        calibration_wall_seconds = time.perf_counter() - started
        seal_receipt, receipt = False, None
    else:
        seal_receipt, receipt = (False, None) if selection_strategy == "random" else receipt_policy(
            smoke_prompt_id=None if burst else smoke_prompt_id, run_dir=calibration_dir, binding=binding
        )
    _require(not burst or (not seal_receipt and (receipt is not None or rank_proof is not None)),
             "native-model burst requires a read-only model calibration receipt")
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at": utc_now(),
        "git": collect_git_metadata(repository_root),
        "method_id": method_id,
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "prompt_ids": list(active_prompt_ids),
        "seed": SEED,
        "chunk_count": chunk_count,
        "budget": budget,
        "strategy": strategy,
        "draft_device": draft_device,
        "target_device": target_device,
        "boundary_protocol_sha256": protocol_sha256,
        "draft_model_id": draft_model_id,
        "draft_artifact_spec_sha256": draft_spec_sha256,
        "target_artifact_spec_sha256": target_spec_sha256,
        "calibration_receipt_mode": ("sealed_rank_distribution" if rank_proof is not None else
            "not_used_by_random" if receipt is None else ("seal" if seal_receipt else "read_only_reuse")),
        "random_repeat": route_repeat,
    }
    if baseline_population_v2:
        preflight.update(population=population_binding, benchmark=benchmark)
    if paper_calibration is not None:
        preflight['paper_calibration'] = paper_calibration
    if profile:
        preflight["duration"] = duration_metadata
    if (burst or native_random) and enforce_gpu_idle:
        preflight["gpu_uuids"] = {role: _gpu_uuid(device) for role, device in
                                  (("draft", draft_device), ("target", target_device))}
    if single_gpu:
        from .stage_timing_probe import _hardware
        preflight['hardware'] = _hardware((draft_device,))
        preflight['accepted_video_schedule'] = 'decode_after_generation'
    if optimized_duration_control:
        preflight["comparison_contract"] = optimized_control.contract()
    if rank_proof is not None:
        preflight["sealed_calibration_ranks"] = rank_proof
    _write_new_json(run_dir / "provenance" / "preflight.json", preflight)

    if enforce_gpu_idle:
        _assert_idle_pair(draft_device, target_device)
    conditioning = [
        {"prompt_id": str(item["prompt_id"]), "prompt": str(item["prompt"])}
        for item in (*protocol.calibration_prompts, *evaluation_values)
    ]
    def rolling_arguments(directory, horizon):
        if draft_model_id != "rolling_forcing_1.3b":
            return {}
        return {"rolling": prepare_rolling_launch(run_dir=directory,
            repo_root=repository_root, model_root=model_root, prompts=conditioning,
            seeds=(SEED,), chunk_count=horizon)}

    def launch_runtime(directory, horizon, **options):
        return runtime_launcher(
            run_dir=directory, repo_root=repository_root,
            draft_artifact_spec_path=draft_artifact_spec_path,
            draft_model_id=draft_model_id,
            draft_artifact_spec_sha256=draft_spec_sha256,
            draft_verified_artifacts=draft_verified,
            target_artifact_spec_path=target_artifact_spec_path,
            target_artifact_spec_sha256=target_spec_sha256,
            target_verified_artifacts=target_verified,
            devices=(draft_device, target_device), write_full_videos=True,
            chunk_count=horizon, conditioning_prompts=conditioning,
            shared_precomputed_conditioning=True,
            **({'allow_shared_device': True} if single_gpu else {}),
            **rolling_arguments(directory, horizon), **options,
        )

    runtime = None
    model_load_seconds = 0.0
    pending: dict[str, Any] = {}
    generation_seconds: dict[str, float] = {}
    ready_seconds: dict[str, float] = {}
    event_ids: dict[str, str] = {}
    native_audits: dict[str, Any] = {}
    render_audits: dict[str, Any] = {}
    renderer = None
    prefiller = None
    campaign: Any | None = None
    try:
        if rank_proof is None:
            load_started = time.perf_counter()
            runtime = launch_runtime(calibration_dir, CHUNK_COUNT)
            model_load_seconds = time.perf_counter() - load_started
            if enforce_gpu_idle:
                _assert_worker_pair_owns_gpus(
                    runtime=runtime, draft_device=draft_device,
                    target_device=target_device, phase="generation")
            runtime.evaluator.full_quality_metric_id = VBENCH_LONG_PRIMARY_QUALITY_METRIC
            campaign = campaign_factory(
                run_dir=calibration_dir, protocol_id=str(protocol.raw["protocol_id"]),
                chunk_count=CHUNK_COUNT, seeds=(SEED,),
                calibration_prompts=_prompt_cases(protocol.calibration_prompts, split="calibration"),
                evaluation_prompts=_prompt_cases(evaluation_values, split="evaluation"),
                smoke_prompts=_prompt_cases(evaluation_values[:1], split="evaluation"),
                draft=optimized_control.ReadOnlyCalibrationBackend(runtime.draft) if optimized_duration_control or burst else runtime.draft,
                target=optimized_control.ReadOnlyCalibrationBackend(runtime.target) if optimized_duration_control or burst else runtime.target,
                noise_provider=optimized_control.ReadOnlyCalibrationBackend(runtime.noise_provider) if optimized_duration_control or burst else runtime.noise_provider,
                evaluator=runtime.evaluator,
            )
            calibration_started = time.perf_counter()
            calibrator = None if selection_strategy == "random" else campaign.run_calibration()
            calibration_wall_seconds = time.perf_counter() - calibration_started
            if seal_receipt:
                try:
                    receipt = seal_calibration_receipt(source_run_dir=calibration_dir, binding=binding)
                except CalibrationReceiptError as error:
                    raise OursCampaignError(f"calibration receipt sealing failed: {error}") from error
            if profile:
                runtime.close()
                runtime = None
        if profile:
            if enforce_gpu_idle:
                _assert_idle_pair(draft_device, target_device)
            load_started = time.perf_counter()
            runtime = launch_runtime(run_dir, chunk_count, **render_worker)
            if native_render:
                from .accepted_stream_client import StreamDecoderProxy
                renderer = StreamDecoderProxy(runtime.draft, runtime.target, run_dir, chunk_count=chunk_count)
                runtime.evaluator.client = renderer
            if longlive_prefill:
                from .accepted_prefill_worker import AcceptedPrefillProxy
                prefiller = AcceptedPrefillProxy(runtime.draft, longlive_split=True)
            model_load_seconds += time.perf_counter() - load_started
            runtime.evaluator.full_quality_metric_id = (
                "vbench_custom_six_dimension_macro_mean" if benchmark == "VBench"
                else VBENCH_LONG_PRIMARY_QUALITY_METRIC
            )
            campaign = campaign_factory(
                run_dir=run_dir,
                protocol_id=f"ours-target-duration-v1-{duration_seconds}s",
                chunk_count=chunk_count, seeds=(SEED,),
                calibration_prompts=_prompt_cases(protocol.calibration_prompts, split="calibration"),
                evaluation_prompts=_prompt_cases(evaluation_values, split="evaluation"),
                smoke_prompts=_prompt_cases(evaluation_values[:1], split="evaluation"),
                draft=runtime.draft, target=runtime.target,
                noise_provider=runtime.noise_provider, evaluator=runtime.evaluator,
            )
        for prompt in campaign.evaluation_prompts:
            if enforce_gpu_idle:
                _assert_worker_pair_owns_gpus(
                    runtime=runtime,
                    draft_device=draft_device,
                    target_device=target_device,
                    phase=f"generation:{prompt.prompt_id}",
                )
            split = "smoke" if smoke_prompt_id else "evaluation"
            event_id = campaign._row_id(
                "2b", split, prompt.prompt_id, SEED, budget, strategy, route_repeat
            )
            event_ids[prompt.prompt_id] = event_id
            started = time.perf_counter()
            with campaign._target_rpc_scope(event_id):
                if selection_strategy == "random":
                    random_input = dict(protocol_id=str(protocol.raw["protocol_id"]),
                        prompt_id=prompt.prompt_id, generation_seed=SEED, repeat_id=random_repeat)
                    if native_random_p8_k2:
                        positions = random_p8_k2_positions(**random_input)
                    else:
                        positions = random_positions(**random_input, budget=budget, chunk_count=chunk_count)
                    key, route_id, accepted, history = campaign._materialize_fixed_positions(
                        prompt=prompt, seed=SEED, corrected_chunks=positions,
                        **({'bypass_draft_on_target_chunks':True,
                            'on_commit':renderer.append,'before_generate':renderer.before_generate}
                           if native_random else {}))
                    _require(tuple(accepted) == positions, "Random route changed selected positions")
                    proposed = accepted
                elif burst:
                    from .rank_burst import materialize_rank_bursts
                    key, route_id, accepted, history = materialize_rank_bursts(
                        campaign, prompt=prompt, seed=SEED, budget=budget, calibrator=calibrator,
                        burst_length=native_burst_length, observation_horizon=native_burst_observation_horizon,
                        observation_offset=native_burst_observation_offset,
                        opening_target_burst=native_opening_target_burst,
                        split_target_burst=native_split_target_burst,
                        final_window_at_tail=native_final_window_at_tail,
                        motion_trigger=native_motion_trigger,
                        on_target_prefix=prefiller.start if prefiller is not None else None,
                        **({'on_commit': renderer.append, 'before_generate': renderer.before_generate}
                           if renderer is not None else {}))
                    proposed = accepted
                else:
                    key, route_id, accepted, proposed, history = campaign._materialize_online_lookahead_coverage_boundary_gated(
                        prompt=prompt,
                        seed=SEED,
                        budget=budget,
                        calibrator=calibrator,
                        window_radius_chunks=WINDOW_RADIUS_CHUNKS,
                        minimum_gap_chunks=MINIMUM_GAP_CHUNKS,
                        maximum_gap_chunks=MAXIMUM_GAP_CHUNKS,
                    )
            generation_seconds[prompt.prompt_id] = _finite_positive(
                time.perf_counter() - started, "generation wall time"
            )
            pending[prompt.prompt_id] = campaign._prepare_rollout(
                key=key,
                route_id=route_id,
                corrected_chunks=accepted,
                proposed_target_chunks=proposed,
                history=history,
            )
            if record_video_ready_timing:
                ready_seconds[prompt.prompt_id] = _finite_positive(
                    time.perf_counter() - started, "video-ready wall time")
                _require(ready_seconds[prompt.prompt_id] >= generation_seconds[prompt.prompt_id],
                         "video-ready time precedes generation completion")
            if renderer is not None:
                render_audits[prompt.prompt_id] = completed_render_binding(
                    renderer, history, pending[prompt.prompt_id].prepared_quality)
            if burst or native_random:
                from .rank_burst import audit_native_model_burst
                native_audits[prompt.prompt_id] = audit_native_model_burst(
                    campaign, key=key, history=history, selected=accepted,
                    draft_model_id=draft_model_id, burst_length=native_burst_length,
                    observation_horizon=native_burst_observation_horizon,
                    observation_offset=native_burst_observation_offset,
                    opening_target_burst=native_opening_target_burst,
                    split_target_burst=native_split_target_burst,
                    split_burst_count=native_split_burst_count,
                    final_window_at_tail=native_final_window_at_tail,
                    longlive_prefill=longlive_prefill,
                    motion_trigger=native_motion_trigger,
                    **({'rolling_prefix_transactions':True} if rolling_prefix_transactions else {}),
                    **({'rolling_committed_prefix':True} if rolling_committed_prefix else {}),
                    **({'fixed_target_positions':True} if native_random else {}))
    finally:
        if runtime is not None:
            runtime.close()

    if enforce_gpu_idle:
        _require(
            not _gpu_compute_pids(_gpu_uuid(draft_device))
            and not _gpu_compute_pids(_gpu_uuid(target_device)),
            "Ours workers retained compute processes after shutdown",
        )
    scorer = scorer_factory(
        run_dir=run_dir,
        checkout=vbench_checkout,
        python_executable=vbench_python,
        asset_manifest=vbench_asset_manifest,
        asset_spec=vbench_asset_spec,
        visible_device=target_device,
    )
    prompt_text = {item["id"]: item["text"] for item in selected}
    requests = [
        VBenchLongVideoRequest(
            video_path=run_dir / pending[prompt_id].prepared_quality.video_path,
            prompt=prompt_text[prompt_id],
            measurement_id=f"mt-{method_id}-{prompt_id}",
        )
        for prompt_id in active_prompt_ids
    ]
    vbench_started = time.perf_counter()
    measurements = scorer.score_videos(requests)
    vbench_wall_seconds = time.perf_counter() - vbench_started
    _require(set(measurements) == {item.measurement_id for item in requests}, "official VBench-Long coverage is incomplete")

    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    _require(campaign is not None, "Ours campaign disappeared after runtime shutdown")
    prompt_cases = {item.prompt_id: item for item in campaign.evaluation_prompts}
    for prompt_id in active_prompt_ids:
        measurement_id = f"mt-{method_id}-{prompt_id}"
        measurement = measurements[measurement_id]
        scores = dict(measurement.dimension_scores)
        _require(tuple(scores) == QUALITY_DIMENSIONS, "VBench dimension order drifted")
        rows.append(
            {
                "method_id": method_id,
                "prompt_id": prompt_id,
                "seed": SEED,
                "generation_wall_seconds": generation_seconds[prompt_id],
                **({"video_ready_wall_seconds": ready_seconds[prompt_id],
                    "timing_scope": timing_scope}
                   if record_video_ready_timing else {}),
                "quality_scores": scores,
                f"{metric_key}_verified": True,
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
        route = pending[prompt_id]
        completed = campaign._finalize_pending_rollout(
            pending=route,
            quality=QualityMeasurement(
                primary=measurement.macro_mean,
                temporal_consistency=measurement.motion_smoothness,
            ),
        )
        event_id = event_ids[prompt_id]
        campaign._append_strategy_row(
            split="smoke" if smoke_prompt_id else "evaluation",
            prompt=prompt_cases[prompt_id],
            seed=SEED,
            budget=budget,
            strategy=strategy,
            route_repeat=route_repeat,
            rollout=completed,
            target_rpc_counts=campaign._target_rpc_lifecycle_counts(event_id),
            route_materialization_rpc_counts=campaign._target_rpc_lifecycle_counts(
                event_id, phase="route_materialization"
            ),
        )
        _write_new_json(
            run_dir / "provenance" / method_id / f"{prompt_id}.json",
            {
                "schema": METHOD_RESULT_SCHEMA,
                "method_id": method_id,
                "prompt_id": prompt_id,
                "seed": SEED,
                "strategy": strategy,
                "budget": budget,
                "accepted_target_chunks": list(route.corrected_chunks),
                "proposed_target_chunks": list(route.proposed_target_chunks),
                "final_history_hash": route.history.head_sha256,
                **({"native_cache_audit": native_audits[prompt_id]} if burst or native_random else {}),
                **({"accepted_stream": render_audits[prompt_id]} if native_render else {}),
                "timing": {"generation_wall_seconds": generation_seconds[prompt_id],
                    **({"video_ready_wall_seconds": ready_seconds[prompt_id],
                        "timing_scope": timing_scope}
                       if record_video_ready_timing else {})},
                metric_key: {
                    "verified": True,
                    "dimension_scores": scores,
                    "macro_mean": measurement.macro_mean,
                    "motion_smoothness": measurement.motion_smoothness,
                    "evidence": evidence_item,
                },
            },
        )

    if profile:
        for row in rows:
            row["duration"] = duration_metadata
    raw_path = run_dir / "raw" / f"main-table-{method_id}.json"
    _write_new_json(raw_path, {"schema": "erank_baseline24_rows_v1" if baseline_population_v2 else (DURATION_RAW_SCHEMA if profile else RAW_SCHEMA), "rows": rows})
    raw_sha256, raw_nbytes = _sha256_file(raw_path)
    result = {
        "schema": "erank_baseline24_result_v1" if baseline_population_v2 else (DURATION_RESULT_SCHEMA if profile else (SMOKE_RESULT_SCHEMA if smoke_prompt_id else RESULT_SCHEMA)),
        "status": "succeeded",
        "run_kind": "single_prompt_smoke" if smoke_prompt_id else "formal_campaign",
        "method_id": method_id,
        "prompt_count": len(active_prompt_ids),
        "row_count": len(rows),
        "seed_count": 1,
        "raw_rows": {"path": raw_path.relative_to(run_dir).as_posix(), "sha256": raw_sha256, "nbytes": raw_nbytes},
        "vbench_wall_seconds": vbench_wall_seconds,
        "vbench_measurements": evidence,
        "draft_generation_device": draft_device,
        "target_generation_device": target_device,
        "quality_device": target_device,
        "model_load_seconds": model_load_seconds,
        "calibration_receipt": rank_proof if receipt is None else {
            **receipt.public_binding(),
            "mode": "sealed_by_this_run" if seal_receipt else "read_only_reuse",
        },
        "random_repeat": route_repeat,
    }
    if baseline_population_v2:
        result.update(population=population_binding, benchmark=benchmark)
    if burst or native_random:
        result.update(strategy=strategy, draft_model_id=draft_model_id,
                      code_revision=preflight["git"]["sha"],
                      native_cache_audit_count=len(native_audits))
    if record_video_ready_timing:
        result["timing_scope"] = timing_scope
    if native_render:
        result["accepted_stream_count"] = len(render_audits)
    if single_gpu:
        result.update(hardware=preflight['hardware'],
                      accepted_video_schedule='decode_after_generation')
    if profile:
        result["duration"] = duration_metadata
        result["calibration_or_receipt_revalidation_wall_seconds"] = calibration_wall_seconds
    if optimized_duration_control:
        result["comparison_contract"] = optimized_control.contract()
    _write_new_json(run_dir / "result.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfexp.main_table_ours_campaign")
    parser.add_argument("--draft-artifact-spec", type=Path, required=True)
    parser.add_argument("--draft-model-id", choices=("self_forcing_1.3b", "longlive_1.3b", "rolling_forcing_1.3b"), default="self_forcing_1.3b")
    parser.add_argument("--target-artifact-spec", type=Path, required=True)
    parser.add_argument("--boundary-config", type=Path, default=BOUNDARY_CONFIG)
    parser.add_argument("--vbench-checkout", type=Path, required=True)
    parser.add_argument("--vbench-python", type=Path, required=True)
    parser.add_argument("--vbench-asset-manifest", type=Path, required=True)
    parser.add_argument("--vbench-asset-spec", type=Path, required=True)
    parser.add_argument("--smoke-prompt-id")
    parser.add_argument("--baseline-population-v2", action="store_true")
    parser.add_argument("--duration-seconds", type=int, choices=(15, 30, 60))
    parser.add_argument("--optimized-duration-control", action="store_true")
    parser.add_argument("--selection-strategy", choices=("erank", "random", "rank_burst"), default="erank")
    parser.add_argument("--burst-calibration-revision")
    parser.add_argument("--random-repeat", type=int, default=0)
    parser.add_argument("--native-random-p8-k2", action="store_true")
    parser.add_argument("--record-video-ready-timing", action="store_true")
    parser.add_argument("--overlap-native-vae", action="store_true")
    parser.add_argument("--native-burst-length", type=int, choices=(1, 2, 3, 4, 8), default=4)
    parser.add_argument("--native-burst-observation-horizon", type=int, choices=(2, 8), default=8)
    parser.add_argument("--native-burst-observation-offset", type=int, choices=(0, 20), default=0)
    parser.add_argument("--native-opening-target-burst", action="store_true")
    parser.add_argument("--native-split-target-burst", action="store_true")
    parser.add_argument("--native-split-burst-count", type=int, choices=(2, 3, 4, 8), default=2)
    parser.add_argument("--native-final-window-at-tail", action="store_true")
    parser.add_argument("--native-motion-trigger", action="store_true")
    parser.add_argument("--sealed-calibration-ranks", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_campaign(
            draft_artifact_spec_path=args.draft_artifact_spec,
            draft_model_id=args.draft_model_id,
            target_artifact_spec_path=args.target_artifact_spec,
            boundary_config_path=args.boundary_config,
            vbench_checkout=args.vbench_checkout,
            vbench_python=args.vbench_python,
            vbench_asset_manifest=args.vbench_asset_manifest,
            vbench_asset_spec=args.vbench_asset_spec,
            smoke_prompt_id=args.smoke_prompt_id,
            duration_seconds=args.duration_seconds,
            optimized_duration_control=args.optimized_duration_control,
            selection_strategy=args.selection_strategy,
            random_repeat=args.random_repeat,
            native_random_p8_k2=args.native_random_p8_k2,
            baseline_population_v2=args.baseline_population_v2,
            burst_calibration_revision=args.burst_calibration_revision,
            record_video_ready_timing=args.record_video_ready_timing,
            overlap_native_vae=args.overlap_native_vae,
            native_burst_length=args.native_burst_length,
            native_burst_observation_horizon=args.native_burst_observation_horizon,
            native_burst_observation_offset=args.native_burst_observation_offset,
            native_opening_target_burst=args.native_opening_target_burst,
            native_split_target_burst=args.native_split_target_burst,
            native_split_burst_count=args.native_split_burst_count,
            native_final_window_at_tail=args.native_final_window_at_tail,
            native_motion_trigger=args.native_motion_trigger,
            sealed_calibration_ranks=args.sealed_calibration_ranks,
        )
    except (OursCampaignError, optimized_control.OptimizedDurationError, OSError, subprocess.SubprocessError) as error:
        print(f"main-table-ours-campaign: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["METHOD_RESULT_SCHEMA", "OursCampaignError", "PREFLIGHT_SCHEMA", "run_campaign"]
