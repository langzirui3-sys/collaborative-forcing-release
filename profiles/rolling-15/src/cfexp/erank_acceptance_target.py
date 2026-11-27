"""Matched native Full Target reference for the main24 acceptance experiment.

Uses the existing persistent Full Target generator, never the bounded-context
correction entrypoint. Timing is the sum of measured warm generation and offline
decode/encode phases; model loading, prompt encoding and queue waiting are excluded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from . import main_table_full_target_campaign as target
from .baseline_population import load_baseline_population
from .erank_acceptance_pilot import READY_TIMING_SCOPE, bound_file
from .main_table_baseline_campaign import _run_dir, _visible_devices, _write_new_json
from .main_table_full_target import METHOD_ID
from .metadata import collect_git_metadata, find_git_root
from .runtime_helpers import shutdown_owned_runtime_helpers
from .stage_timing_probe import _hardware


def project_reference(run_dir: Path, prompts: list[dict], hardware: dict, revision: str) -> dict:
    rows = []
    for prompt in prompts:
        path = run_dir / "provenance" / METHOD_ID / (prompt["id"] + ".json")
        payload = json.loads(path.read_text())
        g = payload["generation"]
        if not (g["prompt_id"] == prompt["id"] and g["seed"] == 20260911
                and g["chunk_count"] == 30 and g["rgb_frame_count"] == 357
                and g["draft_model_calls"] == 0 and g["target_model_calls"] == 30
                and g["target_session_policy"] == "persistent_native_session"
                and g["target_session_count"] == 1 and g["history_reconstruction_calls"] == 0
                and payload["official_vbench"]["verified"] is True):
            raise ValueError("Full Target native reference contract changed")
        timing = g["timing"]
        generation = timing["generation_wall_seconds"]
        decode = timing["vae_decode_and_video_encode_wall_seconds"]
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (generation, decode)):
            raise ValueError("invalid Target video-ready timing")
        video = bound_file(run_dir, run_dir / g["video"]["path"])
        from .video_encoding_policy import verify_video
        encoding_policy = verify_video(run_dir / video['path'])
        if any(video[k] != g["video"][k] for k in ("sha256", "nbytes")):
            raise ValueError("Full Target video binding changed")
        rows.append({"arm": "target", "prompt_id": prompt["id"], "prompt": prompt["text"],
            "seed": 20260911, "generation_wall_seconds": generation,
            "video_ready_wall_seconds": generation + decode,
            "decode_encode_and_prepare_wall_seconds": decode,
            "timing_scope": READY_TIMING_SCOPE,
            "quality_scores": payload["official_vbench"]["dimension_scores"],
            "video": video, "native_evidence": bound_file(run_dir, path), "encoding_policy": encoding_policy,
            "root_latent_sha256": timing["per_chunk"][0]["latent_sha256"]})
    return {"schema": "erank_acceptance_target_reference_v1", "code_revision": revision,
        "population": "main24_method_development_not_held_out", "hardware": hardware,
        "timing_scope": READY_TIMING_SCOPE, "timing_measurement": "sum_of_disjoint_measured_warm_phases",
        "target_semantics": "persistent_full_target_native_session_not_correction_mode",
        "rows": rows, "row_count": len(rows)}


def main(argv=None):
    args = target._build_parser().parse_args(argv)
    if not (args.baseline_population_v2 and args.duration_seconds == 15
            and args.smoke_prompt_id is None and args.prompt_shard_count == 1
            and args.prompt_shard_index == 0 and not args.optimized_duration_control):
        raise ValueError("acceptance Target requires the complete frozen main24 15s population")
    root, run_dir = find_git_root(Path.cwd()), _run_dir()
    prompts, _ = load_baseline_population(root)
    hardware = _hardware(_visible_devices())
    _write_new_json(run_dir / "provenance/acceptance-hardware.json", hardware)
    try:
        campaign_args = vars(args).copy()
        campaign_args["draft_artifact_spec_path"] = campaign_args.pop("draft_artifact_spec")
        campaign_args["target_artifact_spec_path"] = campaign_args.pop("target_artifact_spec")
        result = target.run_campaign(**campaign_args)
        reference = project_reference(run_dir, prompts, hardware, collect_git_metadata(root)["sha"])
        _write_new_json(run_dir / "provenance/acceptance-target-reference.json", reference)
    finally:
        shutdown_owned_runtime_helpers(label="acceptance-target-reference")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
