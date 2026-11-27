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
    CorrectionState,
    CorrectionStateError,
    GenerationContext,
    InitialNoiseDescriptor,
    KVCacheDescriptor,
    LatentHistory,
    LatentSegment,
    SinglePointFork,
    derive_initial_noise_seed,
    load_history_manifest,
    write_history_manifest,
)


DRAFT_MODEL_SHA256 = "d" * 64
TARGET_MODEL_SHA256 = "e" * 64


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root

    def noise(self, chunk_index: int) -> InitialNoiseDescriptor:
        payload = f"noise-{chunk_index}".encode()
        return InitialNoiseDescriptor.create(
            namespace="insight2",
            prompt_id="prompt-001",
            trajectory_seed=17,
            chunk_index=chunk_index,
            shape=(1, 16, 3, 60, 104),
            dtype="bfloat16",
            tensor_sha256=digest(payload),
            tensor_nbytes=len(payload),
        )

    def context(
        self,
        history: LatentHistory,
        role: str,
        *,
        noise: InitialNoiseDescriptor | None = None,
        kv_role: str | None = None,
    ) -> GenerationContext:
        model_id = "self-forcing-1.3b" if role == "draft" else "krea-realtime-14b"
        model_sha = DRAFT_MODEL_SHA256 if role == "draft" else TARGET_MODEL_SHA256
        owner = role if kv_role is None else kv_role
        kv_model_id = (
            "self-forcing-1.3b" if owner == "draft" else "krea-realtime-14b"
        )
        kv_model_sha = DRAFT_MODEL_SHA256 if owner == "draft" else TARGET_MODEL_SHA256
        selected_noise = noise or self.noise(history.next_chunk_index)
        return GenerationContext(
            role=role,
            model_id=model_id,
            model_artifact_sha256=model_sha,
            history_head_sha256=history.head_sha256,
            next_chunk_index=history.next_chunk_index,
            noise=selected_noise,
            kv=KVCacheDescriptor(
                owner_role=owner,
                model_id=kv_model_id,
                model_artifact_sha256=kv_model_sha,
                history_head_sha256=history.head_sha256,
                cache_sha256=digest(
                    f"kv-{owner}-{history.head_sha256}".encode()
                ),
            ),
        )

    def segment(
        self,
        history: LatentHistory,
        role: str,
        payload: bytes,
        *,
        noise: InitialNoiseDescriptor | None = None,
        filename: str | None = None,
    ) -> tuple[GenerationContext, LatentSegment]:
        context = self.context(history, role, noise=noise)
        relative = filename or f"latents/chunk-{history.next_chunk_index}-{role}.bin"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        segment = LatentSegment.from_file(
            history=history,
            context=context,
            latent_root=self.root,
            latent_path=relative,
        )
        return context, segment

    def baseline(self, count: int = 4) -> LatentHistory:
        history = LatentHistory(
            trajectory_id="trajectory-001",
            prompt_id="prompt-001",
            trajectory_seed=17,
        )
        for index in range(count):
            role = "target" if index == 0 else "draft"
            _, segment = self.segment(
                history,
                role,
                f"baseline-{index}".encode(),
            )
            history = history.append(segment)
        return history


class CorrectionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_noise_seed_and_descriptor_are_model_independent_and_tamper_evident(self) -> None:
        noise = self.fixture.noise(3)
        self.assertEqual(
            noise.derived_seed,
            derive_initial_noise_seed(
                namespace="insight2",
                prompt_id="prompt-001",
                trajectory_seed=17,
                chunk_index=3,
            ),
        )
        self.assertEqual(InitialNoiseDescriptor.from_dict(noise.to_dict()), noise)

        tampered = noise.to_dict()
        tampered["shape"] = [1, 16, 4, 60, 104]
        with self.assertRaisesRegex(CorrectionStateError, "descriptor digest mismatch"):
            InitialNoiseDescriptor.from_dict(tampered)

    def test_parent_mismatch_and_duplicate_index_fail_closed(self) -> None:
        history = self.fixture.baseline(1)
        context, candidate = self.fixture.segment(history, "draft", b"chunk-one")
        wrong_parent = LatentSegment(
            **{
                **candidate.__dict__,
                "parent_history_sha256": "0" * 64,
            }
        )
        with self.assertRaisesRegex(CorrectionStateError, "parent history digest mismatch"):
            history.append(wrong_parent)

        committed = history.append(candidate)
        with self.assertRaisesRegex(CorrectionStateError, "expected unique next index 2"):
            committed.append(candidate)

        candidate.verify_context(history, context)

    def test_single_point_branches_are_isolated_from_baseline(self) -> None:
        baseline = self.fixture.baseline(4)
        baseline_manifest = baseline.manifest_sha256
        baseline_head = baseline.head_sha256
        fork = SinglePointFork.open(baseline, 2)

        prefix = fork.prefix
        original = fork.original_draft
        draft_context = self.fixture.context(prefix, "draft", noise=self.fixture.noise(2))
        draft_branch = fork.commit_draft(context=draft_context, segment=original)

        shared_noise = self.fixture.noise(2)
        target_context, target = self.fixture.segment(
            prefix,
            "target",
            b"corrected-two",
            noise=shared_noise,
            filename="branches/two-target.bin",
        )
        target_branch = fork.commit_target(context=target_context, segment=target)

        self.assertEqual(baseline.manifest_sha256, baseline_manifest)
        self.assertEqual(baseline.head_sha256, baseline_head)
        self.assertEqual(len(baseline.segments), 4)
        self.assertEqual(draft_branch.segments[-1].segment_sha256, original.segment_sha256)
        self.assertEqual(target_branch.segments[-1].producer_role, "target")
        self.assertNotEqual(target_branch.head_sha256, draft_branch.head_sha256)
        self.assertEqual(len(target_branch.segments), 3)

    def test_target_replacement_becomes_the_parent_of_subsequent_draft(self) -> None:
        history = self.fixture.baseline(1)
        state = CorrectionState(history=history)

        context_one, draft_one = self.fixture.segment(history, "draft", b"draft-one")
        state = state.stage_draft(context=context_one, segment=draft_one).accept_draft()
        self.assertEqual(state.history.segments[-1].producer_role, "draft")

        noise_two = self.fixture.noise(2)
        draft_context_two, draft_two = self.fixture.segment(
            state.history, "draft", b"draft-two", noise=noise_two
        )
        staged = state.stage_draft(context=draft_context_two, segment=draft_two)
        target_context_two, target_two = self.fixture.segment(
            state.history,
            "target",
            b"target-two",
            noise=noise_two,
            filename="latents/chunk-2-target-replacement.bin",
        )
        replaced = staged.replace_with_target(
            context=target_context_two,
            segment=target_two,
        )

        self.assertIsNone(replaced.staged_draft)
        self.assertEqual(replaced.history.segments[-1].segment_sha256, target_two.segment_sha256)
        self.assertNotIn(draft_two.segment_sha256, {
            segment.segment_sha256 for segment in replaced.history.segments
        })

        next_context, next_draft = self.fixture.segment(
            replaced.history, "draft", b"draft-after-target"
        )
        self.assertEqual(next_context.kv.history_head_sha256, target_two.segment_sha256)
        self.assertEqual(next_draft.parent_history_sha256, target_two.segment_sha256)
        continued = replaced.stage_draft(
            context=next_context,
            segment=next_draft,
        ).accept_draft()
        self.assertEqual(continued.history.segments[-2].producer_role, "target")
        self.assertEqual(
            continued.history.segments[-1].parent_history_sha256,
            continued.history.segments[-2].segment_sha256,
        )

    def test_cross_model_or_stale_kv_is_never_accepted(self) -> None:
        history = self.fixture.baseline(1)
        with self.assertRaisesRegex(CorrectionStateError, "cross-model KV input is forbidden"):
            self.fixture.context(history, "draft", kv_role="target")

        stale_kv = KVCacheDescriptor(
            owner_role="draft",
            model_id="self-forcing-1.3b",
            model_artifact_sha256=DRAFT_MODEL_SHA256,
            history_head_sha256="a" * 64,
            cache_sha256="b" * 64,
        )
        with self.assertRaisesRegex(CorrectionStateError, "different history head"):
            GenerationContext(
                role="draft",
                model_id="self-forcing-1.3b",
                model_artifact_sha256=DRAFT_MODEL_SHA256,
                history_head_sha256=history.head_sha256,
                next_chunk_index=1,
                noise=self.fixture.noise(1),
                kv=stale_kv,
            )

    def test_manifest_round_trip_verifies_content_and_refuses_overwrite(self) -> None:
        history = self.fixture.baseline(3)
        manifest = self.root / "history.json"
        write_history_manifest(manifest, history)
        loaded = load_history_manifest(manifest, latent_root=self.root)
        self.assertEqual(loaded, history)

        with self.assertRaisesRegex(CorrectionStateError, "cannot create history manifest"):
            write_history_manifest(manifest, history)

        first_path = self.root / history.segments[0].latent_path
        first_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(CorrectionStateError, "latent content digest mismatch"):
            load_history_manifest(manifest, latent_root=self.root)

    def test_manifest_rejects_digest_drift_and_duplicate_json_keys(self) -> None:
        history = self.fixture.baseline(2)
        manifest = self.root / "history.json"
        value = history.to_dict()
        value["head_sha256"] = "0" * 64
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(CorrectionStateError, "head digest mismatch"):
            load_history_manifest(manifest)

        manifest.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        with self.assertRaisesRegex(CorrectionStateError, "duplicate JSON key"):
            load_history_manifest(manifest)

    def test_unsafe_or_symlink_latent_paths_are_rejected(self) -> None:
        history = LatentHistory(
            trajectory_id="trajectory-001",
            prompt_id="prompt-001",
            trajectory_seed=17,
        )
        context = self.fixture.context(history, "target")
        outside = self.root.parent / "outside-latent.bin"
        outside.write_bytes(b"outside")
        try:
            link = self.root / "linked.bin"
            link.symlink_to(outside)
            with self.assertRaisesRegex(CorrectionStateError, "symlinks are forbidden"):
                LatentSegment.from_file(
                    history=history,
                    context=context,
                    latent_root=self.root,
                    latent_path="linked.bin",
                )
            with self.assertRaisesRegex(CorrectionStateError, "canonical POSIX relative path"):
                LatentSegment.from_file(
                    history=history,
                    context=context,
                    latent_root=self.root,
                    latent_path="..\\outside-latent.bin",
                )
        finally:
            outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
