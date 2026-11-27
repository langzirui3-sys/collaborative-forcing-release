# Collaborative Forcing

**eRank-Guided Burst Scheduling for Autoregressive Video Generation**

RankPulse schedules short bursts from a stronger Target using signals from
already committed Draft chunks. StateBridge resumes from the accepted
history, and FuseStep combines the missing-history prefill with the first
resumed Draft denoising step.

[Project page](https://langzirui3-sys.github.io/collaborative-forcing-project/) ·
[Preprint](https://langzirui3-sys.github.io/collaborative-forcing-project/paper.pdf) ·
[Video comparisons](https://langzirui3-sys.github.io/collaborative-forcing-project/demo.html)

## Release profiles

| Profile | Duration | Draft cache and Target budget |
| --- | ---: | --- |
| `rolling-15` | 15 s | Native rolling cache, one Target root and four later Target blocks |
| `longlive-15` | 15 s | Native LongLive cache with FuseStep, one Target root and four later Target blocks |
| `self-forcing-30` | 30 s | Native incremental Self-Forcing KV reuse, 37 Target blocks including the root |

Each profile ships its own source snapshot and frozen dependency lock.
`reproduction/source-revisions.json` records the archived source revisions.
`reproduction/profiles.json` records the historical run settings, entry
modules, and runtime inputs. The source package metadata uses Apache-2.0.
Machine-local publication defaults and private archival tools were removed
from the exported copies.

## Install and check the method

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/langzirui3-sys/collaborative-forcing-release.git
cd collaborative-forcing-release/profiles/rolling-15
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m cfexp.rolling_budget.paper_main --help
```

The included tests exercise causal burst selection, accepted-history
binding, and the stream interfaces with portable fixtures. Backend tests
that use PyTorch run when PyTorch is present. Install each profile in its
own environment. The `longlive-15` and `self-forcing-30` entry modules are
`cfexp.longlive_matched_main` and `cfexp.erank_acceptance_pilot`, respectively.

## GPU inputs and experiment settings

For an inference environment, use the pinned Linux x86-64 and Python 3.11
dependencies in the selected profile. The model specifications list the
upstream revisions, relative local paths, and SHA256 checks for the Draft
and Krea Target files. Obtain the model sources and checkpoints from their
providers under their respective terms. [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) ·
[LongLive](https://github.com/NVlabs/LongLive) · [Self-Forcing](https://github.com/guandeh17/Self-Forcing) ·
[Krea Realtime 14B](https://github.com/krea-ai/realtime-video)

From the repository root, check the model files against the selected
profile's exact artifact specifications.

```sh
python3 reproduction/check_assets.py rolling-15 --model-root /path/to/models
```

The profile JSON lists machine-local inputs by name. Replace the file
argument placeholders with your own paths and use a separate
`CFEXP_RUN_DIR` for generated output. The original benchmark entry
modules enforce sealed calibration and hardware-specific admission
records. Those records and the evaluation population belong to the
research environment. A new GPU reproduction needs its own validated
environment and upstream evaluator assets. Before a new cohort, run a
bounded warm-up and profile, and track the run with W&B using your own
credentials or offline mode. This release was checked locally with CPU
tests and entry-point checks. End-to-end GPU execution in an external
environment remains unverified.

## License

Apache-2.0 covers the project source and included documentation. See
`THIRD_PARTY.md` and the preserved source headers for upstream attribution.
