"""GPU gate for Rolling JSONL generation, eRank, VAE and MUSIQ integration."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
import json
from pathlib import Path

from .correction_runtime import DeterministicNoiseProvider, JsonlWorkerClient, _write_backend_config
from .correction_state import LatentHistory
from .insight2_campaign import ChunkGenerationRequest, PromptCase
from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
)
from .metadata import find_git_root
from .rolling_forcing_assets import SPEC_SHA256, _contained_file
from .rolling_forcing_native import _check_file
from .rolling_forcing_noise import seal_schedule

PROMPT = 'a motorcycle accelerating to gain speed'
SEED = 20260911


def run(*, sdvg=False):
    horizon = 30 if sdvg else 16
    output = _run_dir()
    repository = find_git_root(Path.cwd())
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('Rolling worker gate requires one assigned GPU')
    if _gpu_compute_pids(_gpu_uuid(devices[0])):
        raise ValueError('assigned GPU is already occupied')
    models = _model_root()
    path = repository/'experiments/erank3-rolling-native-v1.artifacts.json'
    _check_file(path, SPEC_SHA256)
    spec = json.loads(path.read_text())
    records = []
    for item in spec['artifacts']:
        asset = _contained_file(models, item['path'])
        _check_file(asset, item['sha256'])
        records.append(dict(name=item['name'], resolved_path=str(asset),
                            sha256=item['sha256'], size=asset.stat().st_size))
    provider = DeterministicNoiseProvider(run_dir=output)
    prompt = PromptCase(prompt_id='rolling-worker-gate', prompt=PROMPT, split='evaluation')
    receipt = seal_schedule(provider=provider, prompt_id=prompt.prompt_id, prompt=PROMPT,
                            trajectory_seed=SEED, chunk_count=horizon)
    binding = dict(run_dir=str(output), relative_path=receipt['path'], sha256=receipt['sha256'],
        prompt_id=prompt.prompt_id, prompt=PROMPT, trajectory_seed=SEED,
        namespace=provider.namespace, chunk_count=horizon)
    sdvg_config = None
    if sdvg:
        from .main_table_baseline_campaign import verify_artifact_spec
        score_path = repository/'experiments/erank3-rolling-sdvg-15s-v1.artifacts.json'
        score_spec, score_records, _ = verify_artifact_spec(score_path, method_id='sdvg',
            model_root=models, repository_root=repository)
        sdvg_config = dict(artifact_spec_path=str(score_path),
            artifact_spec_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest(),
            adapter_config=score_spec['adapter_config'], verified_artifacts=score_records)
    config, digest = _write_backend_config(run_dir=output, role='draft', spec_path=path,
        spec_sha256=SPEC_SHA256, verified_artifacts=records, chunk_count=horizon,
        sdvg=sdvg_config,
        rolling=dict(repository_root=str(repository), model_root=str(models),
            musiq_relative_path='metrics/musiq/musiq_spaq_ckpt-358bb6af.pth',
            noise_schedules=[binding]))
    client = None
    try:
        client = JsonlWorkerClient(run_dir=output, role='draft',
            backend_factory='cfexp.rolling_policy_worker:build_backend',
            backend_config_path=config, backend_config_sha256=digest,
            visible_device=devices[0], chunk_count=horizon, expected_model_id='rolling_forcing_1.3b')
        _write_new_json(output/'provenance/worker.json', client.model_state)
        history = LatentHistory(trajectory_id='rolling-worker-gate',
                               prompt_id=prompt.prompt_id, trajectory_seed=SEED)
        if sdvg:
            client.start_sdvg_trajectory(prompt_id=prompt.prompt_id, prompt=PROMPT, chunk_count=horizon)
        rows = []
        for index in range(2):
            generated = client.generate(ChunkGenerationRequest(prompt=prompt, history=history,
                noise=provider.materialize(prompt_id=prompt.prompt_id, trajectory_seed=SEED,
                                           chunk_index=index),
                output_latent_path=f'latents/block-{index:03d}.safetensors', capture_hidden=True))
            if generated.effective_rank is None or generated.hidden_state_path is None:
                raise ValueError('Rolling worker omitted rank/hidden observations')
            state = generated.diagnostics.get('rolling_state', {})
            if state.get('chunk_index') != index or (index and not state.get('reused_rolling_state')):
                raise ValueError('Rolling RPC omitted native continuation evidence')
            reward = (client.score_sdvg_candidate(prompt=PROMPT, chunk_index=index,
                clean_latent_path=generated.segment.latent_path) if sdvg else None)
            history = history.append(generated.segment)
            rows.append({'chunk':index, 'erank':generated.effective_rank,
                         'segment':asdict(generated.segment), 'diagnostics':generated.diagnostics,
                         'sdvg_reward':reward})
        decoded = client.decode(history=history, output_frames_path='decoded/frames.safetensors',
                                output_video_path='videos/worker-gate.mp4', fps=24)
        if decoded.frames.get('shape') != [21, 3, 480, 832]:
            # Six latent frames decode to 1 + 4*(6-1) RGB frames.
            raise ValueError('Rolling worker decoded unexpected video geometry')
        quality = client.score(frames_path='decoded/frames.safetensors',
                               frame_indices=[0,10,20], temporal_window=(0,21))
        result = {'schema':'rolling_worker_gate_v1', 'status':'succeeded',
                  'sdvg_scoring_enabled':sdvg, 'prompt':PROMPT, 'seed':SEED, 'chunks':rows,
                  'decode':asdict(decoded), 'quality':asdict(quality),
                  'claim_scope':'two-block integration gate; not a quality comparison'}
        _write_new_json(output/'result.json', result)
        return result
    finally:
        if client is not None:
            client.close()


def main():
    from .runtime_helpers import shutdown_owned_runtime_helpers
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdvg', action='store_true')
    args = parser.parse_args()
    try:
        run(sdvg=args.sdvg)
        return 0
    finally:
        shutdown_owned_runtime_helpers(label='rolling-worker-gate-controller')


if __name__ == '__main__':
    raise SystemExit(main())
