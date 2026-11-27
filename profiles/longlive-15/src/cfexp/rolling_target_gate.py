"""Trained Rolling/Krea handoff diagnostic, not a benchmark or policy selection.

Rolling restoration intentionally discards speculative future state. Its oracle
is a fresh native continuation from identical accepted history and frozen noise,
not the uninterrupted speculative continuation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
)

PROMPT = 'a motorcycle accelerating to gain speed'
PROMPT_ID = 'rolling-target-handoff'
SEED = 20260911
BLOCKS = 16


def compare_branches(*, torch, root, replacement, factory, request, record):
    """Exercise real branch changes; close each session before its oracle."""
    if torch.equal(root, replacement):
        raise ValueError('handoff probe needs distinct Target blocks')
    backend = factory()
    try:
        candidate = request(backend, root, 1)
        record('draft-candidate', candidate)
        if torch.equal(candidate['latent'], replacement):
            raise ValueError('Target replacement did not change accepted history')
        original_history = torch.cat([root, candidate['latent']], dim=1)
        original = request(backend, original_history, 2)
        record('uninterrupted-successor', original)
        if not original['rolling_state']['reused_rolling_state']:
            raise ValueError('native Rolling continuation did not reuse its session')
        corrected_history = torch.cat([root, replacement], dim=1)
        corrected = request(backend, corrected_history, 2)
        record('corrected-successor', corrected)
        restored = request(backend, original_history, 2)
        record('restored-successor', restored)
        for item in (corrected, restored):
            state = item['rolling_state']
            if not state['discarded_future_state'] or state['replayed_history_blocks'] != 2:
                raise ValueError('branch change omitted native history replay evidence')
    finally:
        backend.close()
    comparisons = {}
    for label, history, actual in (
        ('corrected', corrected_history, corrected),
        ('restored', original_history, restored),
    ):
        backend = factory()
        try:
            reference = request(backend, history, 2)
            record(label + '-fresh-reference', reference)
            equal = bool(torch.equal(actual['latent'], reference['latent']))
            rank_equal = actual['rank_evidence'] == reference['rank_evidence']
            comparisons[label] = {'latent_equal': equal, 'rank_evidence_equal': rank_equal}
            if not equal or not rank_equal:
                raise ValueError(label + ' branch differs from fresh native continuation')
        finally:
            backend.close()
    return comparisons


def decode_probe(*, backend, latents, torch, output):
    """Exercise the native VAE on an accepted prefix, preserving RGB evidence."""
    from PIL import Image
    frames = backend.decode_latents(latents)
    expected = (1 + 4 * (int(latents.shape[1]) - 1), 3, 480, 832)
    if tuple(frames.shape) != expected or frames.dtype != torch.uint8:
        raise ValueError('Rolling decoded RGB dimensions or dtype differ from frozen protocol')
    pixels = frames.contiguous().cpu().numpy()
    evidence = dict(shape=list(frames.shape), dtype='uint8',
                    rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    minimum=int(pixels.min()), maximum=int(pixels.max()),
                    frame_indices=[0, len(pixels)//2, len(pixels)-1])
    output.mkdir(parents=True, exist_ok=False)
    for index in evidence['frame_indices']:
        Image.fromarray(pixels[index].transpose(1,2,0)).save(output/f'frame-{index:03d}.png')
    _write_new_json(output/'decode.json', evidence)
    return evidence


def run():
    from .metadata import find_git_root
    from .correction_artifacts import verify_target_artifact_spec
    from .correction_runtime import DeterministicNoiseProvider, launch_target_generation_worker
    from .correction_state import LatentHistory
    from .insight2_campaign import ChunkGenerationRequest, PromptCase
    from .rolling_forcing_backend import RollingNativeBackend
    from .rolling_forcing_noise import seal_schedule
    from .rolling_forcing_policy_backend import RollingPolicyBackend
    from .incremental_draft import latent_digest

    output, repo, models = _run_dir(), find_git_root(Path.cwd()), _model_root()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = visible.split(',')
    if len(devices) != 2 or any(not x.strip() for x in devices) or devices[0] == devices[1]:
        raise ValueError('Rolling/Target gate requires two distinct exclusive GPUs')
    uuids = [_gpu_uuid(x) for x in devices]
    if any(_gpu_compute_pids(x) for x in uuids):
        raise ValueError('assigned GPU is occupied')
    target_spec = repo/'experiments/insight2-krea14b.artifacts.json'
    _, records, adapter = verify_target_artifact_spec(target_spec, model_root=models, repo_root=repo)
    if adapter != 'cfexp.krea_chunk_backend':
        raise ValueError('unexpected Target adapter')
    target = native = None
    rows = []
    try:
        target = launch_target_generation_worker(run_dir=output, repo_root=repo,
            target_artifact_spec_path=target_spec,
            target_artifact_spec_sha256=hashlib.sha256(target_spec.read_bytes()).hexdigest(),
            target_verified_artifacts=records, visible_device=devices[1], chunk_count=BLOCKS)
        # CUDA visibility must be restricted before main-process torch init.
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
        native = RollingNativeBackend(run_dir=output, repository_root=repo, model_root=models)
        t = native.torch
        from safetensors.torch import load, save_file
        provider = DeterministicNoiseProvider(run_dir=output)
        receipt = seal_schedule(provider=provider, prompt_id=PROMPT_ID, prompt=PROMPT,
                                trajectory_seed=SEED, chunk_count=BLOCKS)
        binding = dict(run_dir=output, relative_path=receipt['path'], sha256=receipt['sha256'],
            prompt_id=PROMPT_ID, prompt=PROMPT, trajectory_seed=SEED,
            namespace=provider.namespace, chunk_count=BLOCKS)
        _write_new_json(output/'provenance/models.json', {
            'rolling':native.records, 'native_load':native.audit, 'target':records,
            'target_policy':'fresh-history diagnostic RPC; not optimized benchmark timing',
            'noise_schedule':receipt})

        def ownership():
            for uuid, allowed in zip(uuids, ({os.getpid()}, {target.process.pid})):
                if not _gpu_compute_pids(uuid) <= allowed:
                    raise ValueError('assigned GPU acquired an unrelated process')

        def noise(index):
            return provider.materialize(prompt_id=PROMPT_ID,trajectory_seed=SEED,chunk_index=index)

        def read_tensor(relative, digest, key):
            raw = (output/relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('handoff input bytes changed')
            return load(raw)[key]

        case = PromptCase(prompt_id=PROMPT_ID, prompt=PROMPT, split='evaluation')
        history = LatentHistory(trajectory_id=PROMPT_ID,prompt_id=PROMPT_ID,trajectory_seed=SEED)
        target_values = []
        for label in ('target-root', 'target-replacement'):
            ownership()
            logical = hashlib.sha256(label.encode()).hexdigest()
            item = target.generate(ChunkGenerationRequest(prompt=case,history=history,
                noise=noise(history.next_chunk_index),output_latent_path=f'latents/{label}.safetensors',
                capture_hidden=False,target_logical_call_id=logical,
                target_call_scope_id=hashlib.sha256(PROMPT_ID.encode()).hexdigest(),
                target_call_phase='route_materialization'))
            target_values.append(read_tensor(item.segment.latent_path,item.segment.latent_sha256,'latent'))
            history = history.append(item.segment)

        def factory():
            return RollingPolicyBackend(native_backend=native,run_dir=output,
                schedule_bindings=[binding],chunk_count=BLOCKS,write_video_impl=None)

        def request(backend, accepted_history, index):
            ownership()
            n = noise(index)
            with t.inference_mode():
                result = backend.generate_chunk(prompt=PROMPT,history_latents=accepted_history,
                    initial_noise=read_tensor(n.tensor_path,n.descriptor.tensor_sha256,'noise'),
                    internal_seed=n.descriptor.derived_seed,chunk_index=index,capture_hidden=True,
                    history_kv_widths=tuple([3]*index))
            ownership()
            return result

        def record(label, item):
            if item['rank'] is None or item['full_hidden'] is None or not item['kv_witness_sha256']:
                raise ValueError('Rolling observation missing')
            path = output/f'latents/{label}.safetensors'
            if path.exists():
                raise ValueError('diagnostic artifact already exists')
            save_file({'latent':item['latent']},str(path))
            row = {k:item[k] for k in ('rank','rank_evidence','rolling_state','kv_witness_sha256')}
            row.update(label=label,latent_sha256=latent_digest(item['latent'],t))
            rows.append(row)
            _write_new_json(output/f'parity/{label}.json',row)

        comparisons = compare_branches(torch=t, root=target_values[0],replacement=target_values[1],
            factory=factory,request=request,record=record)
        # A parity pass alone cannot establish that the policy's output is
        # consumable by the native decoder and subsequent quality evaluation.
        corrected_path = output/'latents/corrected-successor.safetensors'
        corrected_latent = load(corrected_path.read_bytes())['latent']
        backend = factory()
        try:
            ownership()
            decoded = decode_probe(backend=backend, torch=t, output=output/'decoded-probe',
                latents=t.cat([*target_values, corrected_latent], dim=1))
            ownership()
        finally:
            backend.close()
        result = dict(schema='rolling_target_handoff_v2',status='succeeded',prompt=PROMPT,seed=SEED,
            comparisons=comparisons,requests=rows,decoded=decoded,
            claim_scope='three-block trained-model handoff and native restart parity; no quality or speed claim')
        _write_new_json(output/'result.json',result)
        return result
    finally:
        if native is not None:
            native.close()
        if target is not None:
            target.close()
        os.environ['CUDA_VISIBLE_DEVICES'] = visible


def main():
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        run()
        return 0
    finally:
        shutdown_owned_runtime_helpers(label='rolling-target-gate')


if __name__ == '__main__':
    raise SystemExit(main())
