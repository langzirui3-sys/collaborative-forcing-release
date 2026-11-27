"""GPU gate for LongLive JSONL generation, eRank, VAE and MUSIQ integration."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path

from .correction_runtime import DeterministicNoiseProvider, JsonlWorkerClient, _write_backend_config
from .correction_state import LatentHistory
from .insight2_campaign import ChunkGenerationRequest, PromptCase
from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
    verify_artifact_spec,
)
from .metadata import find_git_root

PROMPT = 'a motorcycle accelerating to gain speed'
SEED = 20260911


def run():
    output = _run_dir()
    repository = find_git_root(Path.cwd())
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('LongLive worker gate requires one assigned GPU')
    if _gpu_compute_pids(_gpu_uuid(devices[0])):
        raise ValueError('assigned GPU is already occupied')
    path = repository/'experiments/erank3-longlive-policy-v1.artifacts.json'
    _, records, module = verify_artifact_spec(path, method_id='draft_only',
        model_root=_model_root(), repository_root=repository)
    if module != 'cfexp.longlive_adapter':
        raise ValueError('unexpected LongLive artifact loader')
    config, digest = _write_backend_config(run_dir=output, role='draft', spec_path=path,
        spec_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        verified_artifacts=records, chunk_count=120)
    client = None
    try:
        client = JsonlWorkerClient(run_dir=output, role='draft',
            backend_factory='cfexp.longlive_policy_worker:build_backend',
            backend_config_path=config, backend_config_sha256=digest,
            visible_device=devices[0], chunk_count=120, expected_model_id='longlive_1.3b')
        _write_new_json(output/'provenance/worker.json', client.model_state)
        provider = DeterministicNoiseProvider(run_dir=output)
        prompt = PromptCase(prompt_id='longlive-worker-gate', prompt=PROMPT, split='evaluation')
        history = LatentHistory(trajectory_id='longlive-worker-gate',
                               prompt_id=prompt.prompt_id, trajectory_seed=SEED)
        rows = []
        for index in range(2):
            generated = client.generate(ChunkGenerationRequest(prompt=prompt, history=history,
                noise=provider.materialize(prompt_id=prompt.prompt_id, trajectory_seed=SEED,
                                           chunk_index=index),
                output_latent_path=f'latents/block-{index:03d}.safetensors', capture_hidden=True))
            if generated.effective_rank is None or generated.hidden_state_path is None:
                raise ValueError('LongLive worker omitted rank/hidden observations')
            history = history.append(generated.segment)
            rows.append({'chunk':index, 'erank':generated.effective_rank,
                         'segment':asdict(generated.segment), 'diagnostics':generated.diagnostics})
        decoded = client.decode(history=history, output_frames_path='decoded/frames.safetensors',
                                output_video_path='videos/worker-gate.mp4', fps=24)
        if decoded.frames.get('shape') != [21, 3, 480, 832]:
            # Six latent frames decode to 1 + 4*(6-1) RGB frames.
            raise ValueError('LongLive worker decoded unexpected video geometry')
        quality = client.score(frames_path='decoded/frames.safetensors',
                               frame_indices=[0,10,20], temporal_window=(0,21))
        result = {'schema':'longlive_worker_gate_v1', 'status':'succeeded',
                  'prompt':PROMPT, 'seed':SEED, 'chunks':rows,
                  'decode':asdict(decoded), 'quality':asdict(quality),
                  'claim_scope':'two-block integration gate; not a quality comparison'}
        _write_new_json(output/'result.json', result)
        return result
    finally:
        if client is not None:
            client.close()


def main():
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        run()
        return 0
    finally:
        shutdown_owned_runtime_helpers(label='longlive-worker-gate-controller')


if __name__ == '__main__':
    raise SystemExit(main())
