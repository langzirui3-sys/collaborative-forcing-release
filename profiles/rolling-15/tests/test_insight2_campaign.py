from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cfexp.correction_state import (  # noqa: E402
    GenerationContext,
    InitialNoiseDescriptor,
    KVCacheDescriptor,
    LatentHistory,
    LatentSegment,
)
from cfexp.correction_policy import causal_erank_sudden_drop  # noqa: E402
from cfexp.insight2_campaign import (  # noqa: E402
    CampaignError,
    ChunkGenerationRequest,
    GeneratedChunk,
    Insight2Campaign,
    NOISE_NAMESPACE,
    NoiseMaterialization,
    PreparedFullQuality,
    PromptCase,
    QualityMeasurement,
)
from cfexp.insight2_run import Insight2RunError, audit_kv_consistency  # noqa: E402
from cfexp.self_forcing_frame_adapter import (  # noqa: E402
    EFFECTIVE_RANK_ALGORITHM,
    EFFECTIVE_RANK_EVIDENCE_SCHEMA,
    EFFECTIVE_RANK_LAYER,
    EFFECTIVE_RANK_RELATIVE_CUTOFF,
    EFFECTIVE_RANK_SAMPLING,
    entropy_effective_rank_from_gram_eigenvalues,
    validate_effective_rank_evidence,
)


DRAFT_SHA = "d" * 64
TARGET_SHA = "e" * 64


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: object) -> str:
    return digest(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def boundary_velocity_witness(score: float) -> dict[str, object]:
    return {
        "schema": "correction_boundary_velocity_witness_v1",
        "metric_id": "latent_velocity_transition_discontinuity_v1",
        "accumulator_dtype": "float64",
        "round_decimals": 12,
        "input_dtype": "bfloat16",
        "history_frame_count": 16,
        "candidate_frame_count": 16,
        "previous_velocity_norm": 1.0,
        "boundary_velocity_norm": 1.0,
        "candidate_velocity_norm": 1.0,
        "incoming_discontinuity": score,
        "outgoing_discontinuity": score,
        "score": score,
    }


def rank_evidence(chunk_index: int) -> dict[str, object]:
    """Return a 256-value spectrum whose rank is independently recomputable."""

    cardinality = max(1, 32 - chunk_index)
    eigenvalues = [0.0] * (256 - cardinality) + [1.0] * cardinality
    rank = entropy_effective_rank_from_gram_eigenvalues(eigenvalues)
    evidence: dict[str, object] = {
        "schema": EFFECTIVE_RANK_EVIDENCE_SCHEMA,
        "layer": EFFECTIVE_RANK_LAYER,
        "relative_cutoff": EFFECTIVE_RANK_RELATIVE_CUTOFF,
        "algorithm": EFFECTIVE_RANK_ALGORITHM,
        "sampling": dict(EFFECTIVE_RANK_SAMPLING),
        "gram_eigenvalue_dtype": "float64",
        "gram_eigenvalues": eigenvalues,
        "effective_rank": rank,
    }
    validate_effective_rank_evidence(
        evidence,
        expected_rank=rank,
        expected_eigenvalue_count=256,
    )
    return evidence


def target_diagnostics(chunk_index: int) -> tuple[str, dict[str, object]]:
    self_used = 0 if chunk_index == 0 else 4680
    cross_initialized = chunk_index > 0

    def tensor(shape: list[int], used_tokens: int) -> dict[str, object]:
        return {
            "shape": shape,
            "dtype": "torch.bfloat16",
            "used_tokens": used_tokens,
            "samples": (
                []
                if used_tokens == 0
                else [float(index) for index in range(64)]
            ),
        }

    witness = {
        "schema": "sampled_kv_cache_witness_v1",
        "sampling": "up_to_64_evenly_spaced_values_from_each_populated_k_and_v",
        "layers": [
            {
                "layer": index,
                "global_end_index": self_used,
                "local_end_index": 0,
                "k": tensor([1, 9360, 40, 128], self_used),
                "v": tensor([1, 9360, 40, 128], self_used),
            }
            for index in range(40)
        ],
        "cross_attention_layers": [
            {
                "layer": index,
                "is_init": cross_initialized,
                "k": tensor(
                    [1, 512, 40, 128], 512 if cross_initialized else 0
                ),
                "v": tensor(
                    [1, 512, 40, 128], 512 if cross_initialized else 0
                ),
            }
            for index in range(40)
        ],
    }
    witness_sha256 = canonical_digest(witness)
    schedule = [1000.0, 750.0, 500.0, 250.0]
    cursor = 0 if chunk_index == 0 else 4680
    generation = [
        {
            "timestep": timestep,
            "batch_timesteps_equal": True,
            "current_start": cursor,
        }
        for timestep in schedule
    ]
    full_trace = (
        generation
        if chunk_index == 0
        else [
            {
                "timestep": 0.0,
                "batch_timesteps_equal": True,
                "current_start": cursor,
            },
            *generation,
        ]
    )
    return witness_sha256, {
        "kv_witness_sha256": witness_sha256,
        "kv_witness": witness,
        "forward_trace": full_trace,
        "denoising_forward_trace": generation,
        "denoising_schedule": schedule,
        "explicit_initial_noise_verified": True,
        "native_correction_state": {
            "schema": "krea_native_correction_state_v1",
            "native_session_id": "a" * 64,
            "session_recreated": chunk_index == 0,
            "chunk_index": chunk_index,
            "accepted_history_frames": 3 * chunk_index,
            "bounded_context_rebase": bool(chunk_index),
            "context_frames": min(3 * chunk_index, 3),
        },
    }


def draft_diagnostics(chunk_index: int) -> tuple[str, dict[str, object]]:
    witness = {
        "schema": "test_draft_kv_witness_v1",
        "chunk_index": chunk_index,
        "history_tokens": chunk_index * 4680,
    }
    witness_sha256 = canonical_digest(witness)
    trace = [
        {
            "timestep": 0.0,
            "batch_timesteps_equal": True,
            "current_start": index * 4680,
        }
        for index in range(chunk_index)
    ]
    trace.extend(
        {
            "timestep": timestep,
            "batch_timesteps_equal": True,
            "current_start": chunk_index * 4680,
        }
        for timestep in (1000.0, 937.5, 833.3333129882812, 625.0, 0.0)
    )
    return witness_sha256, {
        "kv_witness_sha256": witness_sha256,
        "kv_witness": witness,
        "forward_trace": trace,
        "rank_evidence": rank_evidence(chunk_index),
    }


class FakeNoiseProvider:
    namespace = NOISE_NAMESPACE

    def __init__(self, root: Path) -> None:
        self.root = root

    def materialize(
        self, *, prompt_id: str, trajectory_seed: int, chunk_index: int
    ) -> NoiseMaterialization:
        relative = f"noise/{prompt_id}-{trajectory_seed}/chunk-{chunk_index:02d}.bin"
        path = self.root / relative
        payload = f"noise|{prompt_id}|{trajectory_seed}|{chunk_index}".encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise AssertionError("noise file changed")
        else:
            path.write_bytes(payload)
        return NoiseMaterialization(
            descriptor=InitialNoiseDescriptor.create(
                namespace=self.namespace,
                prompt_id=prompt_id,
                trajectory_seed=trajectory_seed,
                chunk_index=chunk_index,
                shape=(1, 16, 3, 2, 2),
                dtype="bfloat16",
                tensor_sha256=digest(payload),
                tensor_nbytes=len(payload),
            ),
            tensor_path=relative,
        )


class FakeBackend:
    def __init__(self, root: Path, role: str) -> None:
        self.root = root
        self.role = role
        if role == "draft":
            self.model_id = "self-forcing-1.3b"
            self.model_artifact_sha256 = DRAFT_SHA
        else:
            self.model_id = "krea-realtime-video-14b"
            self.model_artifact_sha256 = TARGET_SHA
        self.calls: list[dict[str, object]] = []

    def generate(self, request: ChunkGenerationRequest) -> GeneratedChunk:
        chunk_index = request.history.next_chunk_index
        if self.role == "target":
            kv_sha256, backend_diagnostics = target_diagnostics(chunk_index)
        else:
            kv_sha256, backend_diagnostics = draft_diagnostics(chunk_index)
        if chunk_index == 0:
            backend_diagnostics["boundary_velocity_witness"] = None
        else:
            if self.role == "target":
                boundary_score = 0.3 if chunk_index % 2 == 0 else 0.7
            elif "draft-preview-from" in request.output_latent_path:
                boundary_score = 0.5
            elif "draft-after-target-from" in request.output_latent_path:
                proposal_chunk = chunk_index - 1
                boundary_score = 0.3 if proposal_chunk % 2 == 0 else 0.7
            else:
                boundary_score = 0.5
            backend_diagnostics["boundary_velocity_witness"] = (
                boundary_velocity_witness(boundary_score)
            )
        history_roles = tuple(segment.producer_role for segment in request.history.segments)
        self.calls.append(
            {
                "prompt_id": request.prompt.prompt_id,
                "chunk_index": chunk_index,
                "history_roles": history_roles,
                "history_hash": request.history.head_sha256,
                "noise_descriptor": request.noise.descriptor.descriptor_sha256,
                "output": request.output_latent_path,
                "capture_hidden": request.capture_hidden,
                "target_logical_call_id": request.target_logical_call_id,
                "target_call_scope_id": request.target_call_scope_id,
                "target_call_phase": request.target_call_phase,
            }
        )
        context = GenerationContext(
            role=self.role,
            model_id=self.model_id,
            model_artifact_sha256=self.model_artifact_sha256,
            history_head_sha256=request.history.head_sha256,
            next_chunk_index=chunk_index,
            noise=request.noise.descriptor,
            kv=KVCacheDescriptor(
                owner_role=self.role,
                model_id=self.model_id,
                model_artifact_sha256=self.model_artifact_sha256,
                history_head_sha256=request.history.head_sha256,
                cache_sha256=kv_sha256,
            ),
        )
        latent_path = self.root / request.output_latent_path
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            f"latent|{self.role}|{request.prompt.prompt_id}|{chunk_index}|"
            f"{request.history.head_sha256}|{request.noise.descriptor.tensor_sha256}"
        ).encode()
        latent_path.write_bytes(payload)
        segment = LatentSegment.from_file(
            history=request.history,
            context=context,
            latent_root=self.root,
            latent_path=request.output_latent_path,
        )
        if self.role == "draft":
            rank = float(backend_diagnostics["rank_evidence"]["effective_rank"])
            if request.capture_hidden:
                hidden_relative = request.output_latent_path + ".hidden"
                hidden_path = self.root / hidden_relative
                hidden_payload = f"hidden|{request.prompt.prompt_id}|{chunk_index}".encode()
                hidden_path.write_bytes(hidden_payload)
                hidden_hash = digest(hidden_payload)
            else:
                hidden_relative = None
                hidden_hash = None
        else:
            hidden_relative = None
            rank = None
            hidden_hash = None
        return GeneratedChunk(
            context=context,
            segment=segment,
            effective_rank=rank,
            hidden_state_path=hidden_relative,
            hidden_state_sha256=hidden_hash,
            diagnostics=backend_diagnostics,
            target_attempt_id=(
                digest(f"attempt|{request.output_latent_path}".encode())
                if self.role == "target"
                else None
            ),
            target_logical_call_id=(
                request.target_logical_call_id if self.role == "target" else None
            ),
        )


class FakeEvaluator:
    def __init__(self) -> None:
        self.local_calls: list[tuple[str, int, str, int]] = []
        self.full_calls: list[tuple[str, str]] = []
        self.prepare_calls: list[tuple[str, str]] = []

    def score_local(
        self, *, history: LatentHistory, chunk_index: int, measurement_id: str
    ) -> QualityMeasurement:
        producer = history.segments[chunk_index].producer_role
        self.local_calls.append(
            (history.head_sha256, chunk_index, measurement_id, history.next_chunk_index)
        )
        target_bonus = 0.2 if producer == "target" else 0.0
        return QualityMeasurement(
            primary=0.4 + chunk_index / 100.0 + target_bonus,
            temporal_consistency=0.6 + (0.1 if producer == "target" else 0.0),
        )

    def score_full(self, *, history: LatentHistory, measurement_id: str) -> QualityMeasurement:
        corrected = [
            segment.chunk_index
            for segment in history.segments[1:]
            if segment.producer_role == "target"
        ]
        self.full_calls.append((history.head_sha256, measurement_id))
        return QualityMeasurement(
            primary=0.4 + sum(corrected) / 200.0,
            temporal_consistency=0.7 + len(corrected) / 100.0,
        )

    def prepare_full(
        self, *, history: LatentHistory, measurement_id: str
    ) -> PreparedFullQuality:
        self.prepare_calls.append((history.head_sha256, measurement_id))
        return PreparedFullQuality(
            measurement_id=measurement_id,
            history_manifest_sha256=history.manifest_sha256,
            history_head_sha256=history.head_sha256,
            prompt_id=history.prompt_id,
            latent_chunk_count=history.next_chunk_index,
            decode_evidence_path=f"decoded/{history.manifest_sha256}.json",
            decode_evidence_sha256=digest(
                f"decode|{history.manifest_sha256}".encode()
            ),
            video_path=f"videos/{measurement_id}.mp4",
            video_sha256=digest(f"video|{measurement_id}".encode()),
            video_nbytes=123,
            fps=24,
            frame_count=12 * history.next_chunk_index - 3,
        )


class CampaignFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.noise = FakeNoiseProvider(root)
        self.draft = FakeBackend(root, "draft")
        self.target = FakeBackend(root, "target")
        self.evaluator = FakeEvaluator()
        self.calibration = (PromptCase("cal-001", "calibration prompt", "calibration"),)
        self.evaluation = (PromptCase("eval-001", "evaluation prompt", "evaluation"),)

    def campaign(
        self,
        *,
        chunk_count: int = 16,
        evaluation_prompts: tuple[PromptCase, ...] | None = None,
        smoke_prompts: tuple[PromptCase, ...] | None = None,
        run_dir: Path | None = None,
    ) -> Insight2Campaign:
        return Insight2Campaign(
            run_dir=run_dir or self.root,
            protocol_id="insight2-test-v1",
            seeds=(17,),
            calibration_prompts=self.calibration,
            evaluation_prompts=evaluation_prompts or self.evaluation,
            smoke_prompts=smoke_prompts,
            draft=self.draft,
            target=self.target,
            noise_provider=self.noise,
            evaluator=self.evaluator,
            chunk_count=chunk_count,
        )


class Insight2CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = CampaignFixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_smoke_prefix_contract_depends_on_chunk_horizon(self) -> None:
        evaluation = tuple(
            PromptCase(
                f"eval-{index:03d}",
                f"frozen evaluation prompt {index}",
                "evaluation",
            )
            for index in range(48)
        )

        long_campaign = self.fixture.campaign(
            chunk_count=60,
            evaluation_prompts=evaluation,
            smoke_prompts=evaluation[:1],
            run_dir=self.root / "long",
        )
        self.assertEqual(long_campaign.smoke_prompts, evaluation[:1])
        long_default_campaign = self.fixture.campaign(
            chunk_count=60,
            evaluation_prompts=evaluation,
            run_dir=self.root / "long-default",
        )
        self.assertEqual(long_default_campaign.smoke_prompts, evaluation[:1])

        for wrong_smoke in (evaluation[:10], evaluation[1:2]):
            with self.subTest(long_wrong_smoke=wrong_smoke):
                with self.assertRaisesRegex(
                    CampaignError,
                    "frozen first 1 evaluation prompt",
                ):
                    self.fixture.campaign(
                        chunk_count=60,
                        evaluation_prompts=evaluation,
                        smoke_prompts=wrong_smoke,
                        run_dir=self.root
                        / f"long-wrong-smoke-{wrong_smoke[0].prompt_id}",
                    )

        legacy_campaign = self.fixture.campaign(
            chunk_count=16,
            evaluation_prompts=evaluation,
            smoke_prompts=evaluation[:10],
            run_dir=self.root / "legacy",
        )
        self.assertEqual(legacy_campaign.smoke_prompts, evaluation[:10])

        with self.assertRaisesRegex(
            CampaignError,
            "frozen first 10 evaluation prompt",
        ):
            self.fixture.campaign(
                chunk_count=16,
                evaluation_prompts=evaluation,
                smoke_prompts=evaluation[:1],
                run_dir=self.root / "legacy-wrong-smoke",
            )

    def test_long_horizon_uses_causal_erank_drop_and_exact_budget(self) -> None:
        campaign = self.fixture.campaign(chunk_count=60)
        calibrator = campaign.run_calibration()
        self.assertEqual(type(calibrator).__name__, "SuddenDropRiskCalibrator")
        rollout = campaign.rollout_online_rank_guided(
            prompt=self.fixture.evaluation[0],
            seed=17,
            budget=3,
            calibrator=calibrator,
        )
        self.assertEqual(rollout.history.next_chunk_index, 60)
        self.assertEqual(len(rollout.result.corrected_chunks), 3)
        self.assertEqual(rollout.result.target_call_count, 4)
        self.assertEqual(rollout.result.chunk_count, 60)
        self.assertIn("online-erank-drop-k3", rollout.result.rollout_id)
        calibration = json.loads(
            (self.root / "calibration" / "ranks.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(calibration["rows"]), 59)
        self.assertTrue(
            all("erank_drop_score" in row for row in calibration["rows"])
        )

    def test_coverage_guided_long_horizon_is_causal_exact_k_and_resumable(self) -> None:
        fixture = CampaignFixture(self.root / "coverage-rollout")
        campaign = fixture.campaign(chunk_count=120)
        calibrator = campaign.run_calibration()
        rollout = campaign.prepare_online_coverage_guided(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )

        selected = rollout.corrected_chunks
        windows = ((13, 21), (30, 38), (47, 55), (65, 73), (82, 90), (99, 107))
        self.assertEqual(len(selected), 6)
        self.assertTrue(
            all(left <= chunk <= right for chunk, (left, right) in zip(selected, windows))
        )
        gaps = (
            selected[0],
            *(right - left for left, right in zip(selected, selected[1:])),
            119 - selected[-1],
        )
        self.assertTrue(all(8 <= gap <= 20 for gap in gaps))
        self.assertEqual(
            rollout.route_id,
            "online-coverage-erank-drop-k6",
        )

        checkpoint_paths = sorted(
            (fixture.root / "state" / "rollouts").glob("*/chunk-*.json"),
            key=lambda path: int(path.stem.removeprefix("chunk-")),
        )
        self.assertEqual(len(checkpoint_paths), 120)
        decisions = []
        for checkpoint_path in checkpoint_paths[1:]:
            metadata = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )["metadata"]
            self.assertEqual(metadata["kind"], "online_coverage_guided")
            decisions.append(metadata["decision"])
        self.assertEqual(
            [decision["chunk_index"] for decision in decisions],
            list(range(1, 120)),
        )
        self.assertTrue(
            all(
                decision["deadline"] is None
                or decision["deadline"] >= decision["chunk_index"]
                for decision in decisions
                if decision["reason"] not in {"before_eligible_window", "after_budget_complete"}
            )
        )

        calls_before_resume = len(fixture.draft.calls) + len(fixture.target.calls)
        resumed = fixture.campaign(chunk_count=120).prepare_online_coverage_guided(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )
        self.assertEqual(resumed.history.head_sha256, rollout.history.head_sha256)
        self.assertEqual(
            len(fixture.draft.calls) + len(fixture.target.calls),
            calls_before_resume,
        )

    def test_lookahead_coverage_reuses_or_discards_bound_preview_and_flushes_tail(self) -> None:
        fixture = CampaignFixture(self.root / "lookahead-rollout")
        campaign = fixture.campaign(chunk_count=120)
        calibrator = campaign.run_calibration()
        rollout = campaign.prepare_online_lookahead_coverage_guided(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )

        self.assertEqual(len(rollout.corrected_chunks), 6)
        windows = ((13, 21), (30, 38), (47, 55), (65, 73), (82, 90), (99, 107))
        self.assertTrue(
            all(
                left <= chunk <= right
                for chunk, (left, right) in zip(rollout.corrected_chunks, windows)
            )
        )
        checkpoint_root = (
            fixture.root / "state" / "rollouts" / rollout.route_key
        )
        metadata = [
            json.loads(
                (checkpoint_root / f"chunk-{chunk:02d}.json").read_text(
                    encoding="utf-8"
                )
            )["metadata"]
            for chunk in range(120)
        ]

        first_accept = next(
            chunk
            for chunk in range(1, 119)
            if not metadata[chunk]["corrected"]
            and metadata[chunk]["gate_probe"] is not None
        )
        self.assertTrue(metadata[first_accept]["probe_reused_as_successor"])
        self.assertEqual(
            metadata[first_accept]["gate_probe"],
            metadata[first_accept + 1]["draft_proposal"],
        )

        first_reject = rollout.corrected_chunks[0]
        rejected_probe = metadata[first_reject]["gate_probe"]
        self.assertIsNotNone(rejected_probe)
        self.assertFalse(metadata[first_reject]["probe_reused_as_successor"])
        successor = metadata[first_reject + 1]["draft_proposal"]
        self.assertNotEqual(rejected_probe, successor)
        self.assertEqual(
            successor["context"]["history_head_sha256"],
            json.loads(
                (checkpoint_root / f"chunk-{first_reject:02d}.json").read_text(
                    encoding="utf-8"
                )
            )["history"]["head_sha256"],
        )

        active_before_reject = [
            float(metadata[chunk]["rank"])
            for chunk in range(1, first_reject)
            if not metadata[chunk]["corrected"]
        ]
        successor_metadata = metadata[first_reject + 1]
        expected_active_drop = causal_erank_sudden_drop(
            (
                *active_before_reject,
                float(successor_metadata["rank"]),
                float(successor_metadata["gate_signal_rank"]),
            ),
            lookback_chunks=4,
        )
        stale_branch_drop = causal_erank_sudden_drop(
            (
                *active_before_reject,
                float(metadata[first_reject]["rank"]),
                float(metadata[first_reject]["gate_signal_rank"]),
                float(successor_metadata["rank"]),
                float(successor_metadata["gate_signal_rank"]),
            ),
            lookback_chunks=4,
        )
        self.assertEqual(
            successor_metadata["erank_drop_score"], expected_active_drop
        )
        self.assertNotEqual(expected_active_drop, stale_branch_drop)

        terminal = metadata[119]
        self.assertTrue(terminal["terminal_flush"])
        self.assertIsNone(terminal["gate_probe"])
        self.assertIsNone(terminal["gate_signal_chunk_index"])
        last_corrected = rollout.corrected_chunks[-1]
        self.assertLessEqual(last_corrected, 107)
        for chunk in range(last_corrected + 1, 120):
            with self.subTest(exhausted_budget_chunk=chunk):
                exhausted = metadata[chunk]
                self.assertIsNone(exhausted["risk_score"])
                self.assertIsNone(exhausted["gate_probe"])
                self.assertIsNone(exhausted["gate_signal_chunk_index"])
                self.assertIsNone(exhausted["gate_signal_rank"])
                self.assertIsNone(exhausted["gate_probe_parent_history_hash"])
                self.assertIsNone(exhausted["gate_probe_kv_hash"])
                self.assertIsNone(
                    exhausted["gate_probe_noise_descriptor_sha256"]
                )
                self.assertFalse(exhausted["probe_reused_as_successor"])
                self.assertFalse(exhausted["decision"]["preview_required"])
                self.assertIn(
                    "budget_exhausted_without_preview",
                    exhausted["decision"]["reason"],
                )
        route_draft_calls = [
            call
            for call in fixture.draft.calls
            if rollout.route_key in str(call["output"])
        ]
        self.assertTrue(route_draft_calls)
        self.assertEqual(len(route_draft_calls), 119 + 6)
        self.assertLessEqual(
            max(int(call["chunk_index"]) for call in route_draft_calls),
            119,
        )
        self.assertFalse(
            any(int(call["chunk_index"]) == 120 for call in route_draft_calls)
        )

    def test_lookahead_partial_resume_reuses_valid_probe(self) -> None:
        root = self.root / "lookahead-resume-reuse"
        fixture = CampaignFixture(root)
        campaign = fixture.campaign(chunk_count=120)
        calibrator = campaign.run_calibration()
        rollout = campaign.prepare_online_lookahead_coverage_guided(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )
        checkpoint_root = root / "state" / "rollouts" / rollout.route_key
        first = json.loads(
            (checkpoint_root / "chunk-01.json").read_text(encoding="utf-8")
        )["metadata"]
        self.assertTrue(first["probe_reused_as_successor"])
        for chunk in range(2, 120):
            (checkpoint_root / f"chunk-{chunk:02d}.json").unlink()
        (
            root
            / "measurements"
            / "rollouts"
            / f"{rollout.route_key}.pending.json"
        ).unlink()

        resumed_fixture = CampaignFixture(root)
        resumed_campaign = resumed_fixture.campaign(chunk_count=120)
        resumed_calibrator = resumed_campaign.run_calibration()
        resumed = resumed_campaign.prepare_online_lookahead_coverage_guided(
            prompt=resumed_fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=resumed_calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )
        self.assertEqual(resumed.corrected_chunks, rollout.corrected_chunks)
        self.assertEqual(resumed.history.next_chunk_index, 120)
        route_calls = [
            call
            for call in resumed_fixture.draft.calls
            if rollout.route_key in str(call["output"])
        ]
        self.assertTrue(route_calls)
        self.assertEqual(route_calls[0]["chunk_index"], 3)
        self.assertFalse(any(call["chunk_index"] == 2 for call in route_calls))

    def test_boundary_gated_lookahead_consumes_proposals_and_keeps_rejections_inactive(self) -> None:
        fixture = CampaignFixture(self.root / "boundary-gated-lookahead")
        campaign = fixture.campaign(chunk_count=120)
        calibrator = campaign.run_calibration()
        rollout = campaign.prepare_online_lookahead_coverage_boundary_gated(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )

        proposed = rollout.proposed_target_chunks
        accepted = rollout.corrected_chunks
        self.assertIsNotNone(proposed)
        assert proposed is not None
        self.assertEqual(len(proposed), 6)
        self.assertTrue(set(accepted).issubset(proposed))
        self.assertGreater(len(accepted), 0)
        self.assertLess(len(accepted), len(proposed))

        checkpoint_root = fixture.root / "state" / "rollouts" / rollout.route_key
        for chunk in proposed:
            metadata = json.loads(
                (checkpoint_root / f"chunk-{chunk:02d}.json").read_text(
                    encoding="utf-8"
                )
            )["metadata"]
            self.assertTrue(metadata["target_candidate_proposed"])
            self.assertIsNotNone(metadata["target_candidate"])
            self.assertIsNotNone(metadata["corrected_frontier_probe"])
            self.assertEqual(
                metadata["target_candidate_accepted"],
                metadata["boundary_gate_decision"]["accept_target"],
            )
            self.assertEqual(
                metadata["gate_probe"]["context"]["noise"],
                metadata["corrected_frontier_probe"]["context"]["noise"],
            )
            self.assertEqual(
                metadata["carried_frontier_kind"],
                "corrected" if chunk in accepted else "original",
            )
            self.assertTrue(metadata["probe_reused_as_successor"])

        route_target_calls = [
            call
            for call in fixture.target.calls
            if rollout.route_key in str(call["output"])
        ]
        self.assertEqual(len(route_target_calls), 6)

        calls_before_resume = len(fixture.target.calls) + len(fixture.draft.calls)
        resumed = fixture.campaign(chunk_count=120).prepare_online_lookahead_coverage_boundary_gated(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )
        self.assertEqual(resumed.to_dict(), rollout.to_dict())
        self.assertEqual(
            len(fixture.target.calls) + len(fixture.draft.calls),
            calls_before_resume,
        )

    def test_boundary_gated_resume_rejects_tampered_acceptance_evidence(self) -> None:
        fixture = CampaignFixture(self.root / "boundary-gated-tamper")
        campaign = fixture.campaign(chunk_count=120)
        calibrator = campaign.run_calibration()
        rollout = campaign.prepare_online_lookahead_coverage_boundary_gated(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )
        proposed = rollout.proposed_target_chunks or ()
        self.assertEqual(len(proposed), 6)
        checkpoint = (
            fixture.root
            / "state"
            / "rollouts"
            / rollout.route_key
            / f"chunk-{proposed[0]:02d}.json"
        )
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        value["metadata"]["boundary_gate_decision"]["accept_target"] = not value[
            "metadata"
        ]["boundary_gate_decision"]["accept_target"]
        checkpoint.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            CampaignError,
            "boundary-gated decision or evidence binding changed",
        ):
            fixture.campaign(chunk_count=120).prepare_online_lookahead_coverage_boundary_gated(
                prompt=fixture.evaluation[0],
                seed=17,
                budget=6,
                calibrator=calibrator,
                window_radius_chunks=4,
                minimum_gap_chunks=8,
                maximum_gap_chunks=20,
            )

    def test_lookahead_partial_resume_discards_stale_probe_after_target(self) -> None:
        root = self.root / "lookahead-resume-rollback"
        fixture = CampaignFixture(root)
        campaign = fixture.campaign(chunk_count=120)
        calibrator = campaign.run_calibration()
        rollout = campaign.prepare_online_lookahead_coverage_guided(
            prompt=fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )
        corrected = rollout.corrected_chunks[0]
        checkpoint_root = root / "state" / "rollouts" / rollout.route_key
        rejected = json.loads(
            (checkpoint_root / f"chunk-{corrected:02d}.json").read_text(
                encoding="utf-8"
            )
        )["metadata"]
        self.assertIsNotNone(rejected["gate_probe"])
        self.assertFalse(rejected["probe_reused_as_successor"])
        for chunk in range(corrected + 1, 120):
            (checkpoint_root / f"chunk-{chunk:02d}.json").unlink()
        (
            root
            / "measurements"
            / "rollouts"
            / f"{rollout.route_key}.pending.json"
        ).unlink()

        resumed_fixture = CampaignFixture(root)
        resumed_campaign = resumed_fixture.campaign(chunk_count=120)
        resumed_calibrator = resumed_campaign.run_calibration()
        resumed = resumed_campaign.prepare_online_lookahead_coverage_guided(
            prompt=resumed_fixture.evaluation[0],
            seed=17,
            budget=6,
            calibrator=resumed_calibrator,
            window_radius_chunks=4,
            minimum_gap_chunks=8,
            maximum_gap_chunks=20,
        )
        self.assertEqual(resumed.corrected_chunks, rollout.corrected_chunks)
        self.assertEqual(resumed.history.next_chunk_index, 120)
        route_calls = [
            call
            for call in resumed_fixture.draft.calls
            if rollout.route_key in str(call["output"])
        ]
        self.assertTrue(route_calls)
        first_call = route_calls[0]
        self.assertEqual(first_call["chunk_index"], corrected + 1)
        self.assertEqual(first_call["history_roles"][-1], "target")
        self.assertNotIn("preview-from", str(first_call["output"]))

    def test_deferred_coverage_grid_contains_all_six_frozen_routes(self) -> None:
        fixture = CampaignFixture(self.root / "coverage-grid")
        campaign = fixture.campaign(chunk_count=120)
        plan = campaign.prepare_multi_budget(
            split="smoke",
            budgets=(6,),
            random_repeats=3,
            strategies=(
                "random",
                "uniform",
                "online_rank_guided",
                "online_coverage_guided",
            ),
            coverage_window_radius_chunks=4,
            coverage_minimum_gap_chunks=8,
            coverage_maximum_gap_chunks=20,
        )

        self.assertEqual(len(plan.entries), 6)
        self.assertEqual(
            [entry.strategy for entry in plan.entries],
            [
                "random",
                "random",
                "random",
                "uniform",
                "online_rank_guided",
                "online_coverage_guided",
            ],
        )
        coverage = plan.entries[-1]
        self.assertEqual(
            coverage.rollout.route_id,
            "online-coverage-erank-drop-k6",
        )
        self.assertEqual(len(coverage.rollout.corrected_chunks), 6)
        self.assertEqual(len({entry.rollout.route_key for entry in plan.entries}), 6)

    def test_deferred_lookahead_grid_contains_five_strategy_profile(self) -> None:
        fixture = CampaignFixture(self.root / "lookahead-grid")
        campaign = fixture.campaign(chunk_count=120)
        plan = campaign.prepare_multi_budget(
            split="smoke",
            budgets=(6,),
            random_repeats=3,
            strategies=(
                "random",
                "uniform",
                "online_rank_guided",
                "online_coverage_guided",
                "online_lookahead_coverage_guided",
            ),
            coverage_window_radius_chunks=4,
            coverage_minimum_gap_chunks=8,
            coverage_maximum_gap_chunks=20,
        )

        self.assertEqual(len(plan.entries), 7)
        self.assertEqual(
            [entry.strategy for entry in plan.entries],
            [
                "random",
                "random",
                "random",
                "uniform",
                "online_rank_guided",
                "online_coverage_guided",
                "online_lookahead_coverage_guided",
            ],
        )
        lookahead = plan.entries[-1]
        self.assertEqual(
            lookahead.rollout.route_id,
            "online-lookahead-coverage-erank-drop-k6",
        )
        self.assertEqual(len(lookahead.rollout.corrected_chunks), 6)
        self.assertEqual(len({entry.rollout.route_key for entry in plan.entries}), 7)

    def test_deferred_boundary_gate_grid_uses_three_arm_proposal_budget_profile(self) -> None:
        fixture = CampaignFixture(self.root / "boundary-gate-grid")
        campaign = fixture.campaign(chunk_count=120)
        plan = campaign.prepare_multi_budget(
            split="smoke",
            budgets=(6,),
            random_repeats=0,
            strategies=(
                "uniform",
                "online_lookahead_coverage_guided",
                "online_lookahead_coverage_boundary_gated",
            ),
            coverage_window_radius_chunks=4,
            coverage_minimum_gap_chunks=8,
            coverage_maximum_gap_chunks=20,
        )

        self.assertEqual(
            [entry.strategy for entry in plan.entries],
            [
                "uniform",
                "online_lookahead_coverage_guided",
                "online_lookahead_coverage_boundary_gated",
            ],
        )
        boundary = plan.entries[-1]
        self.assertEqual(
            boundary.rollout.route_id,
            "online-lookahead-coverage-boundary-gated-erank-drop-k6",
        )
        self.assertEqual(len(boundary.rollout.proposed_target_chunks or ()), 6)
        self.assertLessEqual(len(boundary.rollout.corrected_chunks), 6)
        self.assertTrue(
            set(boundary.rollout.corrected_chunks).issubset(
                boundary.rollout.proposed_target_chunks or ()
            )
        )

    def test_target_success_persists_attempt_bound_backend_diagnostics_first(self) -> None:
        campaign = self.fixture.campaign()
        campaign.run_calibration()

        diagnostics_files = list(
            (self.root / "provenance" / "target-rpc-diagnostics").glob("*.json")
        )
        self.assertEqual(len(diagnostics_files), 1)
        diagnostics_path = diagnostics_files[0]
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        self.assertEqual(diagnostics["schema"], "insight2_target_rpc_diagnostics_v3")
        self.assertEqual(diagnostics["attempt_id"], diagnostics_path.stem)
        self.assertEqual(diagnostics["prompt_id"], "cal-001")
        self.assertEqual(diagnostics["chunk_index"], 0)
        self.assertEqual(diagnostics["noise_descriptor"]["prompt_id"], "cal-001")
        self.assertEqual(diagnostics["noise_descriptor"]["trajectory_seed"], 17)
        self.assertEqual(diagnostics["noise_descriptor"]["chunk_index"], 0)
        self.assertEqual(
            diagnostics["noise_descriptor"]["descriptor_sha256"],
            diagnostics["noise_descriptor_sha256"],
        )
        self.assertEqual(
            set(diagnostics["backend_diagnostics"]),
            {
                "kv_witness_sha256",
                "kv_witness",
                "forward_trace",
                "denoising_forward_trace",
                "denoising_schedule",
                "explicit_initial_noise_verified",
                "boundary_velocity_witness",
                "native_correction_state",
            },
        )
        events = [
            json.loads(line)
            for line in (self.root / "logs" / "target-rpc-events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0]["diagnostics_path"],
            diagnostics_path.relative_to(self.root).as_posix(),
        )
        self.assertEqual(events[0]["diagnostics_sha256"], digest(diagnostics_path.read_bytes()))

    def test_single_point_target_branches_share_baseline_prefix_and_noise(self) -> None:
        campaign = self.fixture.campaign()
        rows = campaign.run_single_point(split="evaluation")

        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["prompt_id"] == "eval-001" for row in rows))
        self.assertTrue(all(abs(row["delta_Q"] - 0.2) < 1e-12 for row in rows))
        self.assertTrue(all(abs(row["delta_TC"] - 0.1) < 1e-12 for row in rows))
        self.assertEqual(len(self.fixture.evaluator.local_calls), 30)
        self.assertEqual(len({row["fork_evidence_path"] for row in rows}), 15)
        for row in rows:
            evidence_path = self.root / row["fork_evidence_path"]
            self.assertTrue(evidence_path.is_file())
            self.assertEqual(digest(evidence_path.read_bytes()), row["fork_evidence_sha256"])
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["event_id"], row["event_id"])
            self.assertEqual(evidence["prefix"]["head_sha256"], row["prefix_hash"])
            self.assertEqual(
                evidence["draft_branch"]["context"]["noise"]["descriptor_sha256"],
                evidence["target_branch"]["context"]["noise"]["descriptor_sha256"],
            )
        for index in range(0, 30, 2):
            draft_score = self.fixture.evaluator.local_calls[index]
            target_score = self.fixture.evaluator.local_calls[index + 1]
            self.assertTrue(draft_score[2].endswith("-draft"))
            self.assertTrue(target_score[2].endswith("-target"))
            self.assertEqual(draft_score[1], target_score[1])
            self.assertEqual(draft_score[3], draft_score[1] + 1)
            self.assertEqual(target_score[3], target_score[1] + 1)

        target_forks = [
            call
            for call in self.fixture.target.calls
            if call["prompt_id"] == "eval-001" and int(call["chunk_index"]) > 0
        ]
        self.assertEqual(len(target_forks), 15)
        rows_by_chunk = {int(row["chunk_index"]): row for row in rows}
        for call in target_forks:
            chunk_index = int(call["chunk_index"])
            row = rows_by_chunk[chunk_index]
            self.assertEqual(
                call["history_roles"],
                ("target",) + ("draft",) * (chunk_index - 1),
            )
            self.assertEqual(call["target_call_scope_id"], row["event_id"])
            self.assertEqual(call["target_call_phase"], "single_point_candidate")
            self.assertEqual(
                call["target_logical_call_id"], row["target_logical_call_id"]
            )

        target_roots = [
            call for call in self.fixture.target.calls if int(call["chunk_index"]) == 0
        ]
        self.assertEqual(len(target_roots), 2)
        self.assertEqual(
            {call["target_call_phase"] for call in target_roots},
            {"calibration_baseline", "evaluation_baseline"},
        )
        self.assertTrue(
            all(
                isinstance(call["target_call_scope_id"], str)
                and len(str(call["target_call_scope_id"])) == 64
                for call in target_roots
            )
        )

        draft_baseline_calls = {
            int(call["chunk_index"]): call
            for call in self.fixture.draft.calls
            if call["prompt_id"] == "eval-001"
        }
        for call in target_forks:
            chunk_index = int(call["chunk_index"])
            self.assertEqual(
                call["noise_descriptor"],
                draft_baseline_calls[chunk_index]["noise_descriptor"],
            )

    def test_resume_rejects_tampered_hidden_state_evidence(self) -> None:
        campaign = self.fixture.campaign()
        rows = campaign.run_single_point(split="evaluation")
        hidden = self.root / rows[0]["hidden_state_path"]
        hidden.write_bytes(b"tampered")

        with self.assertRaisesRegex(CampaignError, "hidden_state_path SHA-256 mismatch"):
            self.fixture.campaign().run_single_point(split="evaluation")

    def test_smoke_and_formal_rows_remain_split_and_resume_without_rewrite(self) -> None:
        campaign = self.fixture.campaign()
        smoke = campaign.run_single_point(split="smoke")
        formal = campaign.run_single_point(split="evaluation")
        self.assertEqual({row["prompt_id"] for row in smoke}, {"eval-001"})
        self.assertEqual({row["prompt_id"] for row in formal}, {"eval-001"})
        self.assertTrue(all(row["split"] == "smoke" for row in smoke))
        self.assertTrue(all(row["split"] == "evaluation" for row in formal))

        raw_path = self.root / "raw" / "insight2a.jsonl"
        before_bytes = raw_path.read_bytes()
        before_generation_calls = len(self.fixture.draft.calls) + len(self.fixture.target.calls)
        before_score_calls = len(self.fixture.evaluator.local_calls)

        resumed = self.fixture.campaign()
        resumed_rows = resumed.run_single_point(split="evaluation")
        self.assertEqual(len(resumed_rows), 15)
        self.assertEqual(raw_path.read_bytes(), before_bytes)
        self.assertEqual(
            len(self.fixture.draft.calls) + len(self.fixture.target.calls),
            before_generation_calls,
        )
        self.assertEqual(len(self.fixture.evaluator.local_calls), before_score_calls)

        calibration = __import__("json").loads(
            (self.root / "calibration" / "ranks.json").read_text()
        )
        calibration_ids = {row["prompt_id"] for row in calibration["rows"]}
        self.assertEqual(calibration_ids, {"cal-001"})
        self.assertTrue(calibration_ids.isdisjoint({row["prompt_id"] for row in smoke}))

    def test_calibration_resume_rejects_rows_not_bound_to_baseline_evidence(self) -> None:
        campaign = self.fixture.campaign()
        campaign.run_calibration()
        path = self.root / "calibration" / "ranks.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["rows"][0]["rank"] += 1.0
        path.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaisesRegex(CampaignError, "baseline rank/hidden evidence"):
            self.fixture.campaign().run_calibration()

    def test_fixed_multi_point_replacement_updates_every_later_history(self) -> None:
        campaign = self.fixture.campaign()
        prompt = self.fixture.evaluation[0]
        rollout = campaign.rollout_fixed_positions(
            prompt=prompt,
            seed=17,
            corrected_chunks=(2, 4),
        )

        self.assertEqual(rollout.result.corrected_chunks, (2, 4))
        self.assertEqual(rollout.result.target_call_count, 3)
        self.assertEqual(
            tuple(segment.producer_role for segment in rollout.history.segments[:6]),
            ("target", "draft", "target", "draft", "target", "draft"),
        )
        rollout_draft_calls = [
            call for call in self.fixture.draft.calls if "/rollouts/" in str(call["output"])
        ]
        self.assertTrue(all(call["capture_hidden"] is False for call in rollout_draft_calls))
        by_chunk = {int(call["chunk_index"]): call for call in rollout_draft_calls}
        self.assertEqual(by_chunk[3]["history_roles"], ("target", "draft", "target"))
        self.assertEqual(
            by_chunk[5]["history_roles"],
            ("target", "draft", "target", "draft", "target"),
        )

        before_generation = len(self.fixture.draft.calls) + len(self.fixture.target.calls)
        before_scores = len(self.fixture.evaluator.full_calls)
        resumed = campaign.rollout_fixed_positions(
            prompt=prompt,
            seed=17,
            corrected_chunks=(2, 4),
        )
        self.assertEqual(resumed.history.head_sha256, rollout.history.head_sha256)
        self.assertEqual(
            len(self.fixture.draft.calls) + len(self.fixture.target.calls),
            before_generation,
        )
        self.assertEqual(len(self.fixture.evaluator.full_calls), before_scores)

    def test_all_multi_budget_routes_share_exact_canonical_target_chunk_zero(self) -> None:
        campaign = self.fixture.campaign()
        prompt = self.fixture.evaluation[0]
        first = campaign.rollout_fixed_positions(
            prompt=prompt,
            seed=17,
            corrected_chunks=(2, 4),
        )
        second = campaign.rollout_fixed_positions(
            prompt=prompt,
            seed=17,
            corrected_chunks=(3, 5),
        )
        online = campaign.rollout_online_rank_guided(
            prompt=prompt,
            seed=17,
            budget=2,
            calibrator=campaign.run_calibration(),
        )
        self.assertEqual(
            first.history.segments[0].segment_sha256,
            second.history.segments[0].segment_sha256,
        )
        self.assertEqual(
            first.history.segments[0].latent_path,
            second.history.segments[0].latent_path,
        )
        self.assertEqual(
            first.history.segments[0].to_dict(),
            online.history.segments[0].to_dict(),
        )
        chunk_zero_calls = [
            call
            for call in self.fixture.target.calls
            if call["prompt_id"] == "eval-001" and call["chunk_index"] == 0
        ]
        self.assertEqual(len(chunk_zero_calls), 1)

    def test_resumed_rollout_requires_its_canonical_target_root(self) -> None:
        campaign = self.fixture.campaign()
        prompt = self.fixture.evaluation[0]
        campaign.rollout_fixed_positions(
            prompt=prompt,
            seed=17,
            corrected_chunks=(2, 4),
        )
        roots = list((self.root / "state" / "rollout-roots").glob("*/chunk-00.json"))
        self.assertEqual(len(roots), 1)
        roots[0].unlink()

        with self.assertRaisesRegex(CampaignError, "missing its canonical Target root"):
            self.fixture.campaign().rollout_fixed_positions(
                prompt=prompt,
                seed=17,
                corrected_chunks=(2, 4),
            )

    def test_checkpoint_resume_rejects_segment_context_mismatch(self) -> None:
        campaign = self.fixture.campaign()
        prompt = self.fixture.evaluation[0]
        campaign.rollout_fixed_positions(
            prompt=prompt,
            seed=17,
            corrected_chunks=(2, 4),
        )
        checkpoints = list((self.root / "state" / "rollouts").glob("*/chunk-01.json"))
        self.assertEqual(len(checkpoints), 1)
        checkpoint = checkpoints[0]
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        context = value["generated"]["context"]
        context["role"] = "target"
        context["model_id"] = "krea-realtime-video-14b"
        context["model_artifact_sha256"] = TARGET_SHA
        kv = context["kv"]
        kv["owner_role"] = "target"
        kv["model_id"] = "krea-realtime-video-14b"
        kv["model_artifact_sha256"] = TARGET_SHA
        kv_payload = {key: item for key, item in kv.items() if key != "descriptor_sha256"}
        kv["descriptor_sha256"] = digest(
            json.dumps(
                kv_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        checkpoint.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CampaignError, "does not match generation context"):
            self.fixture.campaign().rollout_fixed_positions(
                prompt=prompt,
                seed=17,
                corrected_chunks=(2, 4),
            )

    def test_target_rpc_journal_survives_resume_and_does_not_recount_cache_hits(self) -> None:
        campaign = self.fixture.campaign()
        scope_id = "f" * 64
        with campaign._target_rpc_scope(scope_id):
            campaign.rollout_fixed_positions(
                prompt=self.fixture.evaluation[0],
                seed=17,
                corrected_chunks=(2, 4),
            )
        self.assertEqual(campaign._target_rpc_count(scope_id), 3)
        journal = self.root / "logs" / "target-rpc-events.jsonl"
        before = journal.read_bytes()

        resumed = self.fixture.campaign()
        with resumed._target_rpc_scope(scope_id):
            resumed.rollout_fixed_positions(
                prompt=self.fixture.evaluation[0],
                seed=17,
                corrected_chunks=(2, 4),
            )
        self.assertEqual(resumed._target_rpc_count(scope_id), 3)
        self.assertEqual(journal.read_bytes(), before)

    def test_online_policy_uses_current_draft_rank_and_commits_exact_budget(self) -> None:
        campaign = self.fixture.campaign()
        calibrator = campaign.run_calibration()
        prompt = self.fixture.evaluation[0]
        rollout = campaign.rollout_online_rank_guided(
            prompt=prompt,
            seed=17,
            budget=3,
            calibrator=calibrator,
        )
        self.assertEqual(len(rollout.result.corrected_chunks), 3)
        self.assertEqual(rollout.result.target_call_count, 4)
        committed = tuple(
            segment.chunk_index
            for segment in rollout.history.segments[1:]
            if segment.producer_role == "target"
        )
        self.assertEqual(committed, rollout.result.corrected_chunks)

        draft_calls = [
            call for call in self.fixture.draft.calls if "/rollouts/" in str(call["output"])
        ]
        self.assertTrue(all(call["capture_hidden"] is False for call in draft_calls))
        for previous, current in zip(draft_calls, draft_calls[1:]):
            previous_chunk = int(previous["chunk_index"])
            current_chunk = int(current["chunk_index"])
            if current_chunk == previous_chunk + 1:
                self.assertEqual(len(current["history_roles"]), current_chunk)

        checkpoint_paths = sorted(
            (self.root / "state" / "rollouts").glob("*/chunk-*.json")
        )
        self.assertEqual(len(checkpoint_paths), 16)
        for chunk_index, checkpoint_path in enumerate(checkpoint_paths):
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            metadata = checkpoint["metadata"]
            self.assertIn("draft_proposal", metadata)
            if chunk_index == 0:
                self.assertIsNone(metadata["draft_proposal"])
                continue
            proposal = metadata["draft_proposal"]
            self.assertEqual(proposal["context"]["role"], "draft")
            self.assertEqual(proposal["segment"]["producer_role"], "draft")
            self.assertIsNone(proposal["target_attempt_id"])
            self.assertIsNone(proposal["target_logical_call_id"])
            self.assertEqual(metadata["rank"], proposal["effective_rank"])
            self.assertEqual(
                metadata["draft_parent_history_hash"],
                proposal["context"]["history_head_sha256"],
            )
            self.assertEqual(
                metadata["draft_kv_hash"],
                proposal["diagnostics"]["kv_witness_sha256"],
            )
            self.assertEqual(
                metadata["initial_noise_descriptor_sha256"],
                proposal["context"]["noise"]["descriptor_sha256"],
            )
            self.assertEqual(
                set(proposal["diagnostics"]),
                {
                    "kv_witness_sha256",
                    "kv_witness",
                    "forward_trace",
                    "rank_evidence",
                    "boundary_velocity_witness",
                },
            )
            self.assertEqual(
                checkpoint["generated"]["context"]["role"],
                "target" if metadata["corrected"] else "draft",
            )
            self.assertEqual(
                checkpoint["generated"]["context"]["noise"],
                proposal["context"]["noise"],
            )

        corrected_checkpoint = json.loads(
            checkpoint_paths[rollout.result.corrected_chunks[0]].read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(corrected_checkpoint["generated"]["context"]["role"], "target")
        self.assertEqual(
            corrected_checkpoint["metadata"]["draft_proposal"]["context"]["role"],
            "draft",
        )

        generation_calls = len(self.fixture.draft.calls) + len(self.fixture.target.calls)
        resumed = self.fixture.campaign().rollout_online_rank_guided(
            prompt=prompt,
            seed=17,
            budget=3,
            calibrator=calibrator,
        )
        self.assertEqual(resumed.history.head_sha256, rollout.history.head_sha256)
        self.assertEqual(
            len(self.fixture.draft.calls) + len(self.fixture.target.calls),
            generation_calls,
        )

    def test_online_resume_rejects_missing_draft_proposal_evidence(self) -> None:
        campaign = self.fixture.campaign()
        calibrator = campaign.run_calibration()
        prompt = self.fixture.evaluation[0]
        campaign.rollout_online_rank_guided(
            prompt=prompt,
            seed=17,
            budget=2,
            calibrator=calibrator,
        )
        checkpoint = next(
            (self.root / "state" / "rollouts").glob("*/chunk-01.json")
        )
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        del value["metadata"]["draft_proposal"]
        checkpoint.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CampaignError, "missing draft_proposal evidence"):
            self.fixture.campaign().rollout_online_rank_guided(
                prompt=prompt,
                seed=17,
                budget=2,
                calibrator=calibrator,
            )

    def test_online_resume_rejects_target_ids_in_draft_proposal(self) -> None:
        campaign = self.fixture.campaign()
        calibrator = campaign.run_calibration()
        prompt = self.fixture.evaluation[0]
        rollout = campaign.rollout_online_rank_guided(
            prompt=prompt,
            seed=17,
            budget=2,
            calibrator=calibrator,
        )
        corrected_chunk = rollout.result.corrected_chunks[0]
        checkpoint = next(
            (self.root / "state" / "rollouts").glob(
                f"*/chunk-{corrected_chunk:02d}.json"
            )
        )
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        value["metadata"]["draft_proposal"]["target_logical_call_id"] = "a" * 64
        checkpoint.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(CampaignError, "carries Target lifecycle IDs"):
            self.fixture.campaign().rollout_online_rank_guided(
                prompt=prompt,
                seed=17,
                budget=2,
                calibrator=calibrator,
            )

    def test_multi_budget_smoke_uses_evaluation_prefix_and_audits_greedy_search_calls(self) -> None:
        campaign = self.fixture.campaign()
        rows = campaign.run_multi_budget(
            split="smoke",
            budgets=(2,),
            random_repeats=1,
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["prompt_id"] for row in rows}, {"eval-001"})
        self.assertTrue(
            {row["prompt_id"] for row in rows}.isdisjoint(
                {prompt.prompt_id for prompt in self.fixture.calibration}
            )
        )
        self.assertEqual(
            {row["strategy"] for row in rows},
            {"random", "uniform", "online_rank_guided", "greedy_oracle"},
        )
        self.assertTrue(all(row["committed_target_chunks"] == 3 for row in rows))
        ordinary = [row for row in rows if row["strategy"] != "greedy_oracle"]
        self.assertTrue(all(0 <= row["physical_target_calls"] <= 3 for row in ordinary))
        self.assertEqual(
            next(row for row in rows if row["strategy"] == "online_rank_guided")[
                "physical_target_calls"
            ],
            2,
        )
        oracle = next(row for row in rows if row["strategy"] == "greedy_oracle")
        self.assertEqual(oracle["selection_search_rollout_count"], 29)
        self.assertGreaterEqual(oracle["selection_search_target_calls"], 0)
        self.assertLessEqual(oracle["selection_search_target_calls"], 72)
        self.assertEqual(
            oracle["physical_target_calls"], oracle["selection_search_target_calls"]
        )

    def test_synchronous_multi_budget_supports_five_strategy_lookahead_profile(self) -> None:
        class LongHorizonEvaluator(FakeEvaluator):
            def score_full(
                self, *, history: LatentHistory, measurement_id: str
            ) -> QualityMeasurement:
                corrected = [
                    segment.chunk_index
                    for segment in history.segments[1:]
                    if segment.producer_role == "target"
                ]
                self.full_calls.append((history.head_sha256, measurement_id))
                return QualityMeasurement(
                    primary=0.4 + sum(corrected) / 2000.0,
                    temporal_consistency=0.7 + len(corrected) / 100.0,
                )

        fixture = CampaignFixture(self.root / "lookahead-sync-grid")
        fixture.evaluator = LongHorizonEvaluator()
        campaign = fixture.campaign(chunk_count=120)
        rows = campaign.run_multi_budget(
            split="smoke",
            budgets=(6,),
            random_repeats=1,
            strategies=(
                "random",
                "uniform",
                "online_rank_guided",
                "online_coverage_guided",
                "online_lookahead_coverage_guided",
            ),
            coverage_window_radius_chunks=4,
            coverage_minimum_gap_chunks=8,
            coverage_maximum_gap_chunks=20,
            include_greedy_oracle=False,
        )

        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [row["strategy"] for row in rows],
            [
                "random",
                "uniform",
                "online_rank_guided",
                "online_coverage_guided",
                "online_lookahead_coverage_guided",
            ],
        )
        lookahead = rows[-1]
        self.assertEqual(
            lookahead["rollout_id"],
            "online-lookahead-coverage-erank-drop-k6",
        )
        self.assertEqual(lookahead["committed_target_chunks"], 7)

    def test_deferred_multi_budget_seals_plan_then_appends_rows_in_plan_order(self) -> None:
        campaign = self.fixture.campaign()
        plan = campaign.prepare_multi_budget(
            split="smoke",
            budgets=(2,),
            random_repeats=1,
        )

        self.assertEqual(len(plan.entries), 3)
        self.assertEqual(
            [entry.strategy for entry in plan.entries],
            ["random", "uniform", "online_rank_guided"],
        )
        self.assertEqual(self.fixture.evaluator.full_calls, [])
        self.assertEqual(campaign.raw_2b.records(), ())
        manifest = self.root / plan.manifest_path
        self.assertTrue(manifest.is_file())
        self.assertEqual(
            json.loads(manifest.read_text(encoding="utf-8")), plan.to_dict()
        )
        self.assertTrue(
            all(
                (
                    self.root
                    / "measurements"
                    / "rollouts"
                    / f"{entry.rollout.route_key}.pending.json"
                ).is_file()
                for entry in plan.entries
            )
        )

        qualities: dict[str, QualityMeasurement] = {}
        for entry in plan.entries:
            corrected = entry.rollout.corrected_chunks
            qualities[entry.rollout.prepared_quality.measurement_id] = QualityMeasurement(
                primary=0.4 + sum(corrected) / 200.0,
                temporal_consistency=0.7 + len(corrected) / 100.0,
            )
        rows = campaign.finalize_multi_budget(plan=plan, qualities=qualities)
        kv_report = audit_kv_consistency(run_dir=self.root, rows=rows)
        self.assertEqual(kv_report["status"], "passed")
        self.assertGreater(kv_report["checked_checkpoint_contexts"], 0)
        self.assertEqual(kv_report["checked_pending_plan_manifests"], 1)
        original_manifest = manifest.read_bytes()
        malformed_manifests = []
        for field, value in (
            ("schema", "wrong"),
            ("split", "evaluation"),
            ("entries", []),
        ):
            malformed = plan.to_dict()
            malformed[field] = value
            malformed_manifests.append((field, malformed))
        malformed_nested = plan.to_dict()
        malformed_nested["entries"][0]["rollout"].pop("prepared_quality")
        malformed_manifests.append(("prepared_quality", malformed_nested))
        for label, malformed_manifest in malformed_manifests:
            with self.subTest(label=label):
                manifest.write_text(json.dumps(malformed_manifest), encoding="utf-8")
                with self.assertRaisesRegex(
                    Insight2RunError, "pending campaign plan.*is invalid"
                ):
                    audit_kv_consistency(run_dir=self.root, rows=rows)
        manifest.write_bytes(original_manifest)
        with self.assertRaisesRegex(
            Insight2RunError, "does not exactly match finalized 2\\(b\\) rows"
        ):
            audit_kv_consistency(run_dir=self.root, rows=rows[:-1])

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["strategy"] for row in campaign.raw_2b.records()],
            ["random", "uniform", "online_rank_guided"],
        )
        self.assertTrue(all(row["committed_target_chunks"] == 3 for row in rows))
        before = (self.root / "raw" / "insight2b.jsonl").read_bytes()
        resumed = campaign.finalize_multi_budget(plan=plan, qualities=qualities)
        self.assertEqual(resumed, rows)
        self.assertEqual(
            (self.root / "raw" / "insight2b.jsonl").read_bytes(), before
        )


if __name__ == "__main__":
    unittest.main()
