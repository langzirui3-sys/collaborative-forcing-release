"""Frozen 6-prompt x 2-seed forced-prevention pairs; eRank never controls routing."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from .single_target_recovery import (CHUNKS, SAMPLE_FRAMES, contact_sheet,
    intervention_schedule, run_arm, video_time, write_json)
from .motion_video_audit import digest, png, require
from .motion_vae_audit import SPEC_SHA
from .model_compatibility import generate_arm
from .runtime_helpers import shutdown_owned_runtime_helpers

ARMS = {'draft_only': (), 'target_three_5s': (10, 11, 12),
        'target_three_10s': (20, 21, 22)}
PROMPT_IDS = ('c008_vbench2long_0052', 'c004_vbench2long_0283',
              'c001_vbench2long_0655', 'c002_vbench2long_0247',
              'c009_vbench2long_0206', 'c006_vbench2long_0409')
SEEDS = (20260911, 20260912)


def load_plan(path, expected_sha256, pair_index):
    require(digest(path)[0] == expected_sha256, 'paired protocol digest drift')
    p = json.loads(path.read_text())
    require(p['schema'] == 'paired_prevention_6x2_v1' and p['status'] == 'frozen', 'unfrozen protocol')
    require(tuple(x['prompt_id'] for x in p['prompts']) == PROMPT_IDS, 'prompt population drift')
    require(all(isinstance(x['prompt'], str) and bool(x['prompt'].strip()) for x in p['prompts']), 'empty prompt')
    require(tuple(p['seeds']) == SEEDS and p['chunk_count'] == CHUNKS, 'seed or horizon drift')
    require(p['arms'] == {k: list(v) for k, v in ARMS.items()}, 'intervention policy drift')
    require(type(pair_index) is int and 0 <= pair_index < 12, 'invalid pair index')
    prompt = p['prompts'][pair_index // 2]
    seed = p['seeds'][pair_index % 2]
    names = list(ARMS)
    shift = pair_index % 3
    return p, prompt, seed, names[shift:] + names[:shift]


def decode_arm(backend, history, directory, output, arm, trace):
    """Identical decoder; timing separate from instrumented latent generation."""
    torch = backend.torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    frames = backend.decode_latents(history)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - start
    require(tuple(frames.shape) == (1437, 3, 480, 832), 'decoded geometry drift')
    video = directory / 'video.mp4'
    start = time.perf_counter()
    backend.write_video(frames, len(frames), video, 24)
    encode_seconds = time.perf_counter() - start
    for frame in sorted(set(SAMPLE_FRAMES) | set(range(0, 1437, 24))):
        rgb = frames[frame, :, ::4, ::4].permute(1, 2, 0).contiguous().numpy()
        png(directory / f'frame-{frame:04d}.png', 208, 120, rgb.tobytes())
    schedule = ARMS[arm]
    result = {'arm': arm, 'intervention_chunks': list(schedule),
        'intervention_seconds': [video_time(i) for i in schedule],
        'target_block_count': len(trace['target_calls']),
        'generation_with_instrumentation_seconds': sum(x['generation_with_instrumentation_seconds'] for x in trace['chunks']),
        'decode_seconds': decode_seconds, 'encode_seconds': encode_seconds,
        'video': {'path': video.relative_to(output).as_posix(), 'sha256': digest(video)[0],
                  'frames': 1437, 'fps': 24},
        'review_status': 'pending_rank_hidden_visual_review',
        'collapse_onset_seconds': None, 'right_censored': None}
    write_json(directory / 'result.json', result)
    del frames
    gc.collect(); torch.cuda.empty_cache()
    return result


def run(protocol_path, protocol_sha256, pair_index, output, repository, model_root):
    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend, _materialize_source
    from .main_table_baseline_campaign import _gpu_uuid, _gpu_compute_pids, _visible_devices
    from .correction_artifacts import verify_target_artifact_spec, target_artifact_spec_sha256
    from .correction_runtime import launch_target_generation_worker
    from .stage_timing import _unsealed_directory

    plan, item, seed, order = load_plan(protocol_path, protocol_sha256, pair_index)
    output = output.resolve(strict=True)
    _unsealed_directory(output)
    require(not (output / 'preflight.json').exists(), 'pair already started')
    devices = _visible_devices()
    require(len(devices) == 2 and len(set(devices)) == 2, 'two distinct GPUs required')
    uuids = [_gpu_uuid(d) for d in devices]
    require(all(not _gpu_compute_pids(u) for u in uuids), 'GPU busy')
    draft_path = repository / 'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    target_path = repository / 'experiments/insight2-krea14b.artifacts.json'
    require(digest(draft_path)[0] == SPEC_SHA, 'Draft artifact specification drift')
    spec, verified, module = verify_artifact_spec(draft_path, model_root=model_root, repo_root=repository)
    _, tv, tm = verify_target_artifact_spec(target_path, model_root=model_root, repo_root=repository)
    require((module, tm) == ('cfexp.self_forcing_frame_adapter', 'cfexp.krea_chunk_backend'), 'model backend drift')
    started = time.perf_counter()
    target = launch_target_generation_worker(run_dir=output, repo_root=repository,
        target_artifact_spec_path=target_path, target_artifact_spec_sha256=target_artifact_spec_sha256(target_path),
        target_verified_artifacts=tv, visible_device=devices[1], chunk_count=CHUNKS)
    backend = None
    previous_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
        artifacts = {r['name']: Path(r['resolved_path']) for r in verified}
        source_root, python_site = _materialize_source(run_dir=output, worker_index=0, artifacts=artifacts,
            config=spec['adapter_config'], scratch_suffix='paired-prevention')
        backend = OfficialSelfForcingBackend(source_root=source_root, python_site=python_site, artifacts=artifacts,
            config=spec['adapter_config'], device_name='cuda:0', chunk_count=CHUNKS)
        initialization_seconds = time.perf_counter() - started
        from safetensors.torch import save_file
        from .incremental_draft import latent_digest
        import random
        import numpy as np
        torch = backend.torch
        def ownership():
            require(_gpu_compute_pids(uuids[0]) <= {os.getpid()}, 'unrelated Draft GPU workload')
            require(_gpu_compute_pids(uuids[1]) <= {target.process.pid}, 'unrelated Target GPU workload')
        ownership()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        noise = torch.randn((1, 360, 16, 60, 104), device=backend.device, dtype=torch.bfloat16)
        save_file({'noise': noise.cpu()}, str(output / 'initial-noise.safetensors'))
        noise_hash = latent_digest(noise, torch)
        write_json(output / 'preflight.json', {'schema': 'paired_prevention_preflight_v1',
            'protocol_sha256': protocol_sha256, 'pair_index': pair_index,
            'prompt_id': item['prompt_id'], 'prompt': item['prompt'], 'seed': seed,
            'arm_execution_order': order, 'devices': list(devices), 'gpu_uuids': uuids,
            'draft_artifacts': verified, 'target_artifacts': tv,
            'initial_noise_sha256': noise_hash, 'initialization_seconds': initialization_seconds,
            'erank_used_for_acceptance': False, 'forced_target_blocks': 3})
        # Untouched native continuous generation supplies all 360 actual scheduler
        # noise tensors. The requested Draft arm must reproduce every latent.
        control = output / 'native-control'; control.mkdir()
        torch.cuda.synchronize(); start = time.perf_counter()
        generate_arm(backend, arm='native_draft', prompt=item['prompt'], initial_noise=noise,
            root=None, tape_dir=control / 'noise-tape', output=control)
        torch.cuda.synchronize(); native_control_seconds = time.perf_counter() - start
        results = {}
        for arm in order:
            directory = output / 'arms' / arm
            schedule = intervention_schedule(ARMS[arm])
            def target_generate(index, arm_dir, initial, path):
                input_path = output / 'inputs' / f'{arm}-{index:03d}-noise.safetensors'
                input_path.parent.mkdir(exist_ok=True)
                save_file({'noise': initial.detach().cpu().contiguous()}, str(input_path))
                relative = path.relative_to(output).as_posix()
                identity = hashlib.sha256(f'paired-prevention:{pair_index}:{seed}:{arm}:{index}'.encode()).hexdigest()
                payload = {'prompt': item['prompt'], 'chunk_index': index, 'internal_seed': seed, 'capture_hidden': False,
                    'history_latent_paths': [(arm_dir / f'{i:03d}.safetensors').relative_to(output).as_posix() for i in range(index)],
                    'initial_noise_path': input_path.relative_to(output).as_posix(), 'output_paths': {'latent': relative},
                    'target_logical_call_id': identity, 'target_call_scope_id': identity, 'target_call_phase': 'single_point_candidate'}
                response = target._request('generate', payload)
                require(response.get('chunk_index') == index and response.get('history_chunk_count') == index,
                        'Target history or position drift')
                evidence = response['result']
                require(evidence.get('explicit_initial_noise_verified') is True, 'Target initial noise not verified')
                require(evidence['latent']['path'] == relative and digest(path)[0] == evidence['latent']['sha256'], 'Target output binding drift')
                require(len(evidence['denoising_forward_trace']) == 4, 'Target step schedule drift')
                write_json(output / 'target-evidence' / f'{arm}-{index:03d}.json',
                    {'request': payload, 'response': response, 'forced_accept': True})
                return {'chunk': index, 'logical_call_id': identity, 'latent': evidence['latent']}
            history, trace = run_arm(backend, prompt=item['prompt'], noise=noise,
                tape_dir=control / 'noise-tape', directory=directory, source_dir=control / 'native_draft',
                intervention=schedule, target_generate=target_generate, ownership_check=ownership, internal_seed=seed)
            require(latent_digest(noise, torch) == noise_hash, 'shared initial noise mutated')
            ownership()
            results[arm] = decode_arm(backend, history, directory, output, arm, trace)
            del history
            gc.collect(); torch.cuda.empty_cache()
        from PIL import Image
        for page in range(6):
            rows = [[Image.open(output / 'arms' / arm / f'frame-{second*24:04d}.png').convert('RGB').tobytes()
                     for second in range(page*10, (page+1)*10)] for arm in ARMS]
            contact_sheet(output / f'comparison-seconds-{page*10:02d}-{page*10+9:02d}.png', rows)
        result = {'schema': 'paired_prevention_result_v1', 'status': 'generation_and_decode_complete',
            'protocol_sha256': protocol_sha256, 'pair_index': pair_index, 'prompt_id': item['prompt_id'],
            'prompt': item['prompt'], 'seed': seed, 'comparison_row_order': list(ARMS),
            'arm_execution_order': order, 'arms': [results[a] for a in ARMS],
            'initial_noise_sha256': noise_hash, 'initialization_seconds': initialization_seconds,
            'native_control_validation_seconds': native_control_seconds,
            'timing_policy': 'instrumented generation, decode, encode, initialization and native validation reported separately; no pure-inference subtraction',
            'conclusion': 'pending rank-hidden review; eRank was observed but never routed generation'}
        write_json(output / 'result.json', result)
        return result
    finally:
        if backend is not None: backend.close()
        target.close()
        if previous_cuda is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else: os.environ['CUDA_VISIBLE_DEVICES'] = previous_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--pair-index', type=int, required=True)
    args = parser.parse_args()
    from .main_table_baseline_campaign import _run_dir, _model_root
    from .metadata import find_git_root
    try: run(args.protocol, args.protocol_sha256, args.pair_index, _run_dir(), find_git_root(Path.cwd()), _model_root())
    finally: shutdown_owned_runtime_helpers(label='paired-prevention')

if __name__ == '__main__': main()
