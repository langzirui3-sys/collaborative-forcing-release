"""Audit the SF execution ablation across the frozen C10 startup change."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .accepted_stream_report import compare, load_run, require, validate_stream
from .baseline_population import load_baseline_population
from .baseline_shard_report import source_manifest
from .main_table_v1 import QUALITY_DIMENSIONS
from .sf_execution_ablation import (
    SECONDS, SETTINGS, manifest_path, parity, source_binding, validate_invocations,
)

ORIGINAL = "6f2572b865e57919e4a71697ca00d4d80db33c58"
PROJECTED = "d68feb197c0f607d5ee4bba40e202f2a69876758"
PROJECT_FLAG = "--sealed-calibration-ranks"
CALIBRATION_SOURCE = {
    "mode": "sealed_rank_distribution",
    "source_run_id": "erank3-sf-ours-kv-smoke-ntu-v2-20261103T002901Z-bea6721e-df450492-b6951140",
    "source_attempt_id": "attempt-20261103T002901Z-a4f04a4039164fb3bfb9ae699b168d63",
    "source_artifact_index_sha256": "044dfb2ff9bf10966c1b7ef92f1bbec39c8abe09a390a083dedd7bf404124dab",
    "receipt_sha256": "1d6681e30f41c041129b81edca176c543a56cddc959287aa755c3b8b7489856f",
    "rank_sha256": "edbb7fe498839bf3f80c6c5dd9be94bbc007db805592fc2b007f0cdbd895e761",
    "drop_scores_sha256": "3e1b902427354bf326a1916050d807e10506ebb7877f038c75dd1df4e0887da4",
    "rank_count": 1190,
}


def verify_startup_change(repo):
    for revision in (ORIGINAL, PROJECTED):
        validate_invocations(repo, revision)
    for setting in SETTINGS:
        before, after = (source_manifest(repo, revision, manifest_path(setting))["spec"]
                         for revision in (ORIGINAL, PROJECTED))
        command = after["command"]
        argv = command["argv"]
        input_groups = [[item for item in spec["inputs"] if not item["target"].endswith(".py")]
                        for spec in (before, after)]
        require(input_groups[0] == input_groups[1]
                and argv.count(PROJECT_FLAG) == 1
                and {**command, "argv": [arg for arg in argv if arg != PROJECT_FLAG]} == before["command"]
                and after["parameters"] == before["parameters"]
                and after["resources"] == before["resources"],
                "projected startup changed the timed command or ablation setting")


def verify_calibration(original, projected):
    require(all(projected.get(key) == value for key, value in CALIBRATION_SOURCE.items())
            and original.get("receipt_sha256") == CALIBRATION_SOURCE["receipt_sha256"]
            and all(projected.get(key) == value for key, value in original.items()),
            "projected startup does not use the same original C10 rank distribution")


def compare_arm(full, candidate, setting):
    for key in ("hardware", "timing_scope", "duration", "arms", "cohort", "benchmark",
                "population", "population_binding", "row_count", "audited_codec_options",
                "audited_noise_bindings"):
        require(full[key] == candidate[key], "projected ablation changed " + key)
    video, prefill = SETTINGS[setting]
    schedule = "interleaved" if video else "after_generation"
    require(candidate.get("accepted_video_schedule", "interleaved") == schedule
            and ("sealed_accepted_target_prefill_audit" in candidate) is prefill,
            "projected ablation did not use its declared overlap flags")
    for key in ("sealed_native_rope_audit", "sealed_native_full_attention_audit",
                "sealed_native_index_constants_audit"):
        require(key in full and key in candidate, "missing native execution audit: " + key)
    verify_calibration(full["calibration_receipt"], candidate["calibration_receipt"])
    before = {row["prompt_id"]: row for row in full["rows"]}
    differences = {}
    scores = []
    for row in candidate["rows"]:
        prompt_id = row["prompt_id"]
        require(prompt_id in before and prompt_id not in differences,
                "projected ablation prompt coverage differs")
        old = before[prompt_id]
        require(row["prompt"] == old["prompt"] and row["seed"] == old["seed"]
                and row["timing_scope"] == old["timing_scope"]
                and set(row["quality_scores"]) == set(QUALITY_DIMENSIONS)
                and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                        for value in row["quality_scores"].values())
                and row["native_cache_audit"]["burst_reuse"].get(
                    "accepted_target_prefill_verified", False) is prefill,
                "projected ablation changed the prompt, timing scope, or prefill evidence")
        ready, generation = row["video_ready_wall_seconds"], row["generation_wall_seconds"]
        require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in (ready, generation)) and ready >= generation,
                "invalid projected ablation timing")
        validate_stream(row, target=False)
        differences[prompt_id] = ready - old["video_ready_wall_seconds"]
        scores.append(math.fsum(row["quality_scores"].values()) / len(QUALITY_DIMENSIONS) * 100)
    require(len(differences) == len(before) == 24, "projected ablation is incomplete")
    exact, rows = parity(full, candidate)
    require(all(exact[key] for key in ("target_positions_exact", "rank_and_decisions_exact",
                                      "accepted_latents_exact", "raw_rgb_exact")),
            "projected startup did not preserve the exact route, latents, and decoded pixels")
    return {
        "mean_ready_delta_vs_original_full_seconds": math.fsum(differences.values()) / len(differences),
        "mean_video_ready_seconds": math.fsum(row["video_ready_wall_seconds"]
                                            for row in candidate["rows"]) / len(differences),
        "mean_quality_100": math.fsum(scores) / len(scores),
        "mean_dynamic_degree_100": 100 * math.fsum(row["quality_scores"]["dynamic_degree"]
                                                  for row in candidate["rows"]) / len(scores),
        "per_prompt_ready_delta_seconds": differences,
        "parity_vs_full": exact,
        "per_prompt_parity": rows,
    }


def report(repo, runs):
    require(set(runs) == {"target", *SETTINGS}, "all five ablation arms are required")
    verify_startup_change(repo)
    data, sources = {}, {}
    for setting, root in runs.items():
        data[setting], marker = load_run(root, target=setting == "target", seconds=SECONDS,
                                         repo=repo, manifest=manifest_path(setting))
        revision = ORIGINAL if setting in ("target", "full") else PROJECTED
        sources[setting] = {**source_binding(marker, revision), "code_revision": revision}
    full, target = data["full"], data["target"]
    require(full.get("accepted_video_schedule", "interleaved") == "interleaved"
            and "sealed_accepted_target_prefill_audit" in full
            and all(row["native_cache_audit"]["burst_reuse"].get(
                "accepted_target_prefill_verified") is True for row in full["rows"])
            and full["audited_codec_options"] == target["audited_codec_options"]
            and full["audited_noise_bindings"] == target["audited_noise_bindings"],
            "original full arm and Target do not bind the expected execution and inputs")
    population, binding = load_baseline_population(repo)
    original = compare(full, target, population, seconds=SECONDS, population_binding=binding)
    return {
        "schema": "sf_execution_projected_ablation_v1",
        "status": "verified",
        "original_matched_full_and_target": original,
        "projected_module_arms": {
            setting: compare_arm(full, data[setting], setting)
            for setting in SETTINGS if setting != "full"
        },
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    for name in ("target", *SETTINGS):
        parser.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    repo = args.pop("repo")
    runs = {setting: args[setting.replace("-", "_")] for setting in ("target", *SETTINGS)}
    print(json.dumps(report(repo, runs), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
