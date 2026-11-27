"""Read the sealed same-pair Self-Forcing execution-module ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .accepted_prefill import ENV as PREFILL_ENV
from .accepted_stream_report import compare, load_run, require
from .baseline_population import load_baseline_population
from .baseline_shard_report import source_manifest

SECONDS = 15
SETTINGS = {
    "full": (True, True),
    "no-video": (False, True),
    "no-prefill": (True, False),
    "neither": (False, False),
}
DEFER = "--defer-video-delivery"


def manifest_path(setting):
    return f"experiments/sf-execution-ablation-{setting}-15s-24p-ntu-v1.manifest.json"


def validate_invocations(repo, commit):
    """Check that the four published commands vary only the planned switches."""
    baseline = source_manifest(repo, commit, manifest_path("full"))["spec"]["command"]
    require(DEFER not in baseline["argv"], "the full execution control must overlap delivery")
    for setting, (video, prefill) in SETTINGS.items():
        spec = source_manifest(repo, commit, manifest_path(setting))["spec"]
        command = spec["command"]
        require(command["argv"] == baseline["argv"] + ([] if video else [DEFER])
                and command["env"] == {**baseline["env"], PREFILL_ENV: str(int(prefill))}
                and command["cwd"] == baseline["cwd"],
                "execution ablation changed generation or another execution feature")
        require(spec["parameters"]["execution_ablation"] ==
                {"video_overlap": video, "prefill_overlap": prefill},
                "execution ablation manifest label differs from its command")


def parity(full, candidate):
    original = {row["prompt_id"]: row for row in full["rows"]}
    observed = {row["prompt_id"]: row for row in candidate["rows"]}
    require(bool(original) and len(original) == len(full["rows"]) and len(observed) == len(candidate["rows"])
            and original.keys() == observed.keys(), "execution parity population differs")
    evidence = []
    for pid, before in original.items():
        after = observed[pid]
        before_trajectory, after_trajectory = (r["trajectory"] for r in (before, after))
        before_stream, after_stream = (r["accepted_stream"] for r in (before, after))
        before_latents, after_latents = ([c["latent_sha256"] for c in s["chunks"]]
                                         for s in (before_stream, after_stream))
        evidence.append({
            "prompt_id": pid,
            "target_positions_exact": before_trajectory["accepted_chunks"] == after_trajectory["accepted_chunks"],
            "rank_and_decisions_exact": before_trajectory["decisions"] == after_trajectory["decisions"],
            "accepted_latents_exact": before_latents == after_latents,
            "raw_rgb_exact": before_stream["pixels"]["raw_sha256"] == after_stream["pixels"]["raw_sha256"],
            "encoded_bytes_exact": before["video"]["sha256"] == after["video"]["sha256"],
            "quality_scores_exact": before["quality_scores"] == after["quality_scores"],
        })
    return {k: all(row[k] for row in evidence) for k in evidence[0] if k != "prompt_id"}, evidence


def source_binding(marker, commit):
    require(marker["git_sha"] == commit, "sealed execution source has another revision")
    return {k: marker[k] for k in ("run_id", "attempt_id", "artifact_index_sha256")}


def report(*, repo, commit, target, runs):
    require(set(runs) == set(SETTINGS), "all four ablation settings are required")
    validate_invocations(repo, commit)
    population, population_binding = load_baseline_population(repo)
    reference, marker = load_run(target, target=True, seconds=SECONDS, repo=repo,
                                 manifest=manifest_path("target"))
    sources = {"target": source_binding(marker, commit)}
    results = {}
    full = None
    for setting, (video, prefill) in SETTINGS.items():
        candidate, marker = load_run(runs[setting], target=False, seconds=SECONDS, repo=repo,
                                     manifest=manifest_path(setting))
        sources[setting] = source_binding(marker, commit)
        schedule = "interleaved" if video else "after_generation"
        require(candidate.get("accepted_video_schedule", "interleaved") == schedule,
                "sealed video overlap setting differs")
        require(("sealed_accepted_target_prefill_audit" in candidate) is prefill
                and all(row["native_cache_audit"]["burst_reuse"].get(
                    "accepted_target_prefill_verified", False) is prefill for row in candidate["rows"]),
                "sealed prefill overlap setting differs")
        require(candidate["audited_codec_options"] is not None
                and candidate["audited_codec_options"] == reference["audited_codec_options"]
                and candidate["audited_noise_bindings"] == reference["audited_noise_bindings"],
                "ablation changed codec options or initial noise")
        metrics = compare(candidate, reference, population, seconds=SECONDS,
                          population_binding=population_binding)
        if full is None:
            full = candidate
        require(candidate["calibration_receipt"] == full["calibration_receipt"],
                "execution settings used different C10 receipts")
        exact, rows = parity(full, candidate)
        results[setting] = {
            "video_overlap": video,
            "prefill_overlap": prefill,
            "matched": metrics,
            "parity_vs_full": exact,
            "per_prompt_parity": rows,
            "preserves_route_latents_and_raw_rgb": all(exact[k] for k in
                ("target_positions_exact", "rank_and_decisions_exact", "accepted_latents_exact", "raw_rgb_exact")),
        }
    return {"schema": "sf_execution_ablation_verified_v1", "status": "verified",
            "generation_commit": commit, "sources": sources, "settings": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "target", *SETTINGS):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    options = vars(parser.parse_args())
    runs = {setting: options.pop(setting.replace("-", "_")) for setting in SETTINGS}
    print(json.dumps(report(runs=runs, **options), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
