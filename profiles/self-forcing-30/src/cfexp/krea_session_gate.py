"""Trained single-GPU Target session/accepted-history parity diagnostic."""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from pathlib import Path

from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
)

PROMPT = 'a motorcycle accelerating to gain speed'
SEED = 20260911
BLOCKS = 16


def run():
    from .metadata import find_git_root
    from .correction_artifacts import verify_target_artifact_spec
    from .correction_runtime import DeterministicNoiseProvider, launch_target_generation_worker
    from .correction_state import LatentHistory
    from .insight2_campaign import ChunkGenerationRequest, PromptCase
    from .krea_chunk_backend import MAIN_TABLE_FOUR_STEP_SCHEDULE, validate_native_correction_state

    output, repo, models = _run_dir(), find_git_root(Path.cwd()), _model_root()
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('Target session gate requires one exclusive GPU')
    uuid = _gpu_uuid(devices[0])
    if _gpu_compute_pids(uuid):
        raise ValueError('assigned GPU is occupied')
    spec = repo / 'experiments/insight2-krea14b.artifacts.json'
    _, records, adapter = verify_target_artifact_spec(spec, model_root=models, repo_root=repo)
    if adapter != 'cfexp.krea_chunk_backend':
        raise ValueError('unexpected Target adapter')
    provider = DeterministicNoiseProvider(run_dir=output)
    case = PromptCase(prompt_id='target-session-gate', prompt=PROMPT, split='evaluation')
    rows = []
    client = None

    def launch():
        if _gpu_compute_pids(uuid):
            raise ValueError('GPU is not released between reference workers')
        return launch_target_generation_worker(
            run_dir=output, repo_root=repo, target_artifact_spec_path=spec,
            target_artifact_spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
            target_verified_artifacts=records, visible_device=devices[0], chunk_count=BLOCKS,
        )

    def empty(seed):
        return LatentHistory(trajectory_id=f'gate-{seed}', prompt_id=case.prompt_id,
                             trajectory_seed=seed)

    def generate(history, label, persistent=False):
        if not _gpu_compute_pids(uuid) <= {client.process.pid}:
            raise ValueError('GPU acquired an unrelated process')
        noise = provider.materialize(prompt_id=case.prompt_id,
            trajectory_seed=history.trajectory_seed, chunk_index=history.next_chunk_index)
        request = ChunkGenerationRequest(prompt=case, history=history, noise=noise,
            output_latent_path=f'latents/{label}.safetensors', capture_hidden=False,
            target_logical_call_id=hashlib.sha256(label.encode()).hexdigest(),
            target_call_scope_id=hashlib.sha256(b'target-session-gate').hexdigest(),
            target_call_phase='route_materialization')
        item = (client.generate_persistent if persistent else client.generate)(request)
        if item.diagnostics['denoising_schedule'] != list(MAIN_TABLE_FOUR_STEP_SCHEDULE):
            raise ValueError('Target schedule mismatch')
        if not persistent:
            validate_native_correction_state(item.diagnostics['native_correction_state'],
                                             chunk_index=history.next_chunk_index)
        rows.append(dict(label=label, history_sha256=history.head_sha256,
                         noise_sha256=noise.descriptor.descriptor_sha256,
                         segment=asdict(item.segment), diagnostics=item.diagnostics))
        _write_new_json(output / f'parity/{label}.json', rows[-1])
        return item

    try:
        client = launch()
        root_a = generate(empty(SEED), 'root-a')
        root_b = generate(empty(SEED + 1), 'root-b')
        if root_a.segment.latent_sha256 == root_b.segment.latent_sha256:
            raise ValueError('branch diagnostic requires distinct accepted histories')
        history_a = empty(SEED).append(root_a.segment)
        history_b = empty(SEED + 1).append(root_b.segment)
        successor_a = generate(history_a, 'successor-a')
        successor_b = generate(history_b, 'successor-b')
        restored_a = generate(history_a, 'restored-a')
        identities = {item.diagnostics['native_correction_state']['native_session_id']
                      for item in (root_b, successor_a, successor_b, restored_a)}
        if len(identities) != 1 or any(
            item.diagnostics['native_correction_state']['session_recreated']
            for item in (successor_a, successor_b, restored_a)
        ):
            raise ValueError('correction path recreated its session across branch changes')
        if successor_a.segment.latent_sha256 != restored_a.segment.latent_sha256:
            raise ValueError('restoring accepted history changed Target output')
        client.close()
        client = None
        client = launch()
        fresh_a = generate(history_a, 'fresh-a')
        if fresh_a.segment.latent_sha256 != successor_a.segment.latent_sha256:
            raise ValueError('reused Target session differs from fresh accepted-history reference')
        client.close()
        client = None
        client = launch()
        client.start_trajectory(prompt_id=case.prompt_id, prompt=PROMPT,
                                seed=SEED, chunk_count=BLOCKS)
        native_root = generate(empty(SEED), 'native-root', persistent=True)
        native_next = generate(empty(SEED).append(native_root.segment), 'native-next', persistent=True)
        if (native_root.segment.latent_sha256 != root_a.segment.latent_sha256
                or native_next.segment.latent_sha256 != successor_a.segment.latent_sha256):
            raise ValueError('correction and pure-Target native paths differ')
        result = dict(schema='krea_session_gate_v1', status='succeeded', prompt=PROMPT,
            branch_restore_equal=True, fresh_reference_equal=True, pure_target_equal=True,
            requests=rows, claim_scope='trained short parity gate; not a quality or speed comparison')
        _write_new_json(output / 'result.json', result)
    finally:
        if client is not None:
            client.close()


if __name__ == '__main__':
    run()
