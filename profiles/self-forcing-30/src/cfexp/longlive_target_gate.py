"""Krea-root/LongLive continuation gate; not a benchmark or policy selection."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import time

from .correction_artifacts import verify_target_artifact_spec
from .correction_runtime import launch_correction_runtime
from .correction_state import LatentHistory
from .insight2_campaign import ChunkGenerationRequest, PromptCase
from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
    verify_artifact_spec,
)
from .metadata import find_git_root

PROMPT = 'a motorcycle accelerating to gain speed'
SEED = 20260911


def exercise(runtime, output, ownership=lambda: None):
    """Compare two histories with identical per-position noise, then restore."""
    case = PromptCase(prompt_id='longlive-target-gate', prompt=PROMPT, split='evaluation')
    empty = LatentHistory(trajectory_id=case.prompt_id, prompt_id=case.prompt_id,
                          trajectory_seed=SEED)
    rows = []
    scope = hashlib.sha256(f'longlive-target-gate:{case.prompt_id}:{SEED}'.encode()).hexdigest()

    def generate(role, history, label):
        ownership()
        noise = runtime.noise_provider.materialize(prompt_id=case.prompt_id,
            trajectory_seed=SEED, chunk_index=history.next_chunk_index)
        logical = hashlib.sha256((label + history.head_sha256).encode()).hexdigest()
        request = ChunkGenerationRequest(prompt=case, history=history, noise=noise,
            output_latent_path=f'latents/{label}.safetensors', capture_hidden=role == 'draft',
            target_logical_call_id=logical if role == 'target' else None,
            target_call_scope_id=scope if role == 'target' else None,
            target_call_phase='route_materialization' if role == 'target' else None)
        tick = time.perf_counter()
        item = getattr(runtime, role).generate(request)
        ownership()
        if role == 'draft' and (item.effective_rank is None or item.hidden_state_path is None):
            raise ValueError('LongLive omitted eRank or hidden observation')
        rows.append({'label':label, 'role':role, 'chunk':history.next_chunk_index,
            'history_sha256':history.head_sha256,
            'noise_sha256':noise.descriptor.descriptor_sha256,
            'generation_seconds':time.perf_counter()-tick,
            'erank':item.effective_rank, 'segment':asdict(item.segment),
            'diagnostics':item.diagnostics})
        return item

    root = empty.append(generate('target', empty, 'target-root').segment)
    draft_history = root.append(generate('draft', root, 'draft-candidate').segment)
    original = generate('draft', draft_history, 'draft-successor')
    target_history = root.append(generate('target', root, 'target-replacement').segment)
    corrected = generate('draft', target_history, 'corrected-successor')
    restored = generate('draft', draft_history, 'restored-successor')
    equal = original.segment.latent_sha256 == restored.segment.latent_sha256
    _write_new_json(output/'parity/branch-restore.json', {
        'equal':equal, 'original_sha256':original.segment.latent_sha256,
        'restored_sha256':restored.segment.latent_sha256, 'requests':rows})
    if not equal:
        raise ValueError('LongLive restored branch differs under identical history and noise')
    arms = {}
    for name, history in (('target_root',draft_history.append(original.segment)),
                          ('target_root_and_replacement',target_history.append(corrected.segment))):
        ownership()
        decoded = runtime.draft.decode(history=history,
            output_frames_path=f'decoded/{name}.safetensors',
            output_video_path=f'videos/{name}.mp4', fps=24)
        if decoded.frames.get('shape') != [33,3,480,832]:
            raise ValueError('unexpected three-block decoded geometry')
        arms[name] = {'decode':asdict(decoded), 'history':history.head_sha256}
        ownership()
    result = {'schema':'longlive_target_gate_v1', 'status':'succeeded',
        'prompt':PROMPT, 'seed':SEED, 'requests':rows, 'arms':arms,
        'restored_branch_equal':equal,
        'timing_scope':'individual generation RPCs; excludes model loading and decoding',
        'claim_scope':'three-block shared-conditioning and branch gate; no quality superiority claim'}
    _write_new_json(output/'result.json',result)
    return result


def run():
    output, repo, models = _run_dir(), find_git_root(Path.cwd()), _model_root()
    devices = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('integration gate requires one exclusive GPU')
    if _gpu_compute_pids(_gpu_uuid(devices[0])):
        raise ValueError('assigned GPU is occupied')
    draft = repo/'experiments/erank3-longlive-policy-v1.artifacts.json'
    target = repo/'experiments/insight2-krea14b.artifacts.json'
    _, dr, dm = verify_artifact_spec(draft, method_id='draft_only',
        model_root=models, repository_root=repo)
    _, tr, tm = verify_target_artifact_spec(target, model_root=models, repo_root=repo)
    if dm != 'cfexp.longlive_adapter' or tm != 'cfexp.krea_chunk_backend':
        raise ValueError('unexpected model pair')
    tick = time.perf_counter()
    with launch_correction_runtime(run_dir=output, repo_root=repo,
            draft_artifact_spec_path=draft,
            draft_artifact_spec_sha256=hashlib.sha256(draft.read_bytes()).hexdigest(),
            draft_verified_artifacts=dr, draft_model_id='longlive_1.3b',
            target_artifact_spec_path=target,
            target_artifact_spec_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            target_verified_artifacts=tr, devices=(devices[0],devices[0]),
            chunk_count=120, allow_shared_device=True,
            shared_precomputed_conditioning=True,
            conditioning_prompts=[{'prompt_id':'longlive-target-gate','prompt':PROMPT}]) as runtime:
        _write_new_json(output/'provenance/models.json', {
            'model_load_seconds':time.perf_counter()-tick, 'models':runtime.model_state,
            'device_policy':'one shared GPU; not comparable to two-GPU benchmark timing'})
        def ownership():
            allowed = {runtime.draft.process.pid, runtime.target.process.pid}
            if not _gpu_compute_pids(_gpu_uuid(devices[0])) <= allowed:
                raise ValueError('shared GPU acquired an unrelated compute process')
        return exercise(runtime,output,ownership)


def main():
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        run()
        return 0
    finally:
        shutdown_owned_runtime_helpers(label='longlive-target-gate')


if __name__ == '__main__':
    raise SystemExit(main())
