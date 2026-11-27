"""Decode fixed, hash-bound development latents; never generate new latents."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

from .motion_video_audit import (SOURCE_SHA, SOURCE_INDEX, FRAME_INDICES,
    bound_path, digest, png, require, sheet, video_samples)
from .main_table_merge import _load_frozen_index
from .runtime_helpers import shutdown_owned_runtime_helpers

PROMPTS = ('c004_vbench2long_0283', 'c008_vbench2long_0052')
SPEC_SHA = '6339b9adbf7e0ca0c1ea7b5095839a20125a592ece1957def9c350e76752508f'
MODES = ('bf16_full', 'fp32_full', 'bf16_tail')
TAIL_START_LATENT = 270


def frame_plan(mode):
    """A reset latent prefix has 1+4*(L-1) frames, not 4*L frames."""
    require(mode in MODES, 'unknown decode mode')
    start = TAIL_START_LATENT if mode == 'bf16_tail' else 0
    # First reset frame is a boundary diagnostic. Only the final frame is
    # compared quantitatively, after 89 latent frames of native warmup.
    absolute = (1080, 1436) if start else FRAME_INDICES
    return start, tuple(i - 4 * start for i in absolute), tuple(absolute)


def restore_history(cell, history, committed, output):
    segments = history['segments']
    require(len(segments) == len(committed) == 120 and history['next_chunk_index'] == 120,
            'incomplete source history')
    paths = []
    output.mkdir(parents=True, exist_ok=False)
    for i, (segment, record) in enumerate(zip(segments, committed, strict=True)):
        require(segment['chunk_index'] == record['chunk_index'] == i
                and segment['producer_role'] == record['producer_role']
                and segment['latent_sha256'] == record['latent_sha256']
                and segment['latent_nbytes'] == record['nbytes'], 'committed history mismatch')
        src = bound_path(cell, {'path': segment['latent_path'],
            'sha256': record['latent_sha256'], 'nbytes': record['nbytes']})
        dst = output / f'{i:03d}.safetensors'
        with src.open('rb') as reader, dst.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        require(digest(dst) == (record['latent_sha256'], record['nbytes']), 'copy changed')
        paths.append(dst)
    return paths


def run(source, output, repository, model_root):
    source, output = source.resolve(strict=True), output.resolve(strict=True)
    require(not source.is_relative_to(output) and not output.is_relative_to(source), 'overlap')
    require(not any((p / name).exists() for p in (output, *output.parents)
                    for name in ('_SUCCESS.json', '_FAILED.json', 'artifact-index.json')), 'sealed output')
    require(not (output / 'source-latents').exists(), 'audit already started')
    frozen, marker = _load_frozen_index(source)
    require(marker['git_sha'] == SOURCE_SHA and marker['artifact_index_sha256'] == SOURCE_INDEX,
            'unexpected source seal')
    attempt = source / 'attempts' / marker['attempt_id']
    result_path = frozen.file(str((attempt / 'result.json').relative_to(source)), label='source result')
    rows = json.loads(result_path.read_text())['rows']
    selected = [r for r in rows if r['arm'] == 'draft_only' and r['prompt_id'] in PROMPTS]
    require(len(selected) == 2 and {r['prompt_id'] for r in selected} == set(PROMPTS), 'source population')
    inputs, evidence, original_samples = {}, [], {}
    cell = attempt / 'cells/draft_only'
    for row in selected:
        pid = row['prompt_id']
        histories = []
        for path in (cell / 'state/rollouts').glob('*/chunk-119.json'):
            path = frozen.file(str(path.relative_to(source)), label='final checkpoint')
            history = json.loads(path.read_text())['history']
            if history['prompt_id'] == pid:
                histories.append((path, history))
        require(len(histories) == 1, 'ambiguous history')
        path, history = histories[0]
        require(history['head_sha256'] == row['trajectory']['final_history_head_sha256'], 'history head drift')
        inputs[pid] = restore_history(cell, history, row['trajectory']['committed'], output / 'source-latents' / pid)
        video = bound_path(attempt, row['video'])
        original_samples[pid] = video_samples(video, Path(sys.executable))
        evidence.append({'prompt_id': pid, 'checkpoint_sha256': digest(path)[0],
                         'history_head': history['head_sha256'], 'original_video': row['video']})

    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import _materialize_source
    spec_path = repository / 'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    require(digest(spec_path)[0] == SPEC_SHA, 'Draft artifact spec changed')
    spec, verified, module = verify_artifact_spec(spec_path, model_root=model_root, repo_root=repository)
    require(module == 'cfexp.self_forcing_frame_adapter', 'unexpected adapter')
    artifacts = {r['name']: Path(r['resolved_path']) for r in verified}
    source_root, python_site = _materialize_source(run_dir=output, worker_index=0,
        artifacts=artifacts, config=spec['adapter_config'], scratch_suffix='vae-audit')
    old_cwd = Path.cwd()
    sys.path[:0] = [str(python_site), str(source_root)]
    os.chdir(source_root)
    try:
        from .main_table_baseline_campaign import _gpu_uuid, _gpu_compute_pids
        device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(bool(device) and ',' not in device, 'exactly one explicit device required')
        gpu_uuid = _gpu_uuid(device)
        require(not _gpu_compute_pids(gpu_uuid), 'allocated decode GPU is busy')
        import torch
        from safetensors.torch import load_file
        from torchvision.io import write_video
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one allocated GPU required')
        wrapper = importlib.import_module('utils.wan_wrapper').WanVAEWrapper
        results, samples = [], {}
        for mode in MODES:
            # Reload original weights for FP32; upcasting rounded BF16 weights
            # would not test a true FP32 decoder.
            dtype = torch.float32 if mode == 'fp32_full' else torch.bfloat16
            vae = wrapper().eval().requires_grad_(False)
            checkpoint_dtypes = sorted({str(p.dtype) for p in vae.parameters()})
            vae = vae.to(device='cuda:0', dtype=dtype)
            for pid in PROMPTS:
                require(_gpu_compute_pids(gpu_uuid) <= {os.getpid()}, 'unrelated GPU workload appeared')
                tensors = []
                for path in inputs[pid]:
                    values = load_file(str(path), device='cpu')
                    require(set(values) == {'latent'}, 'unexpected latent tensor keys')
                    t = values['latent']
                    require(tuple(t.shape) == (1, 3, 16, 60, 104) and t.dtype == torch.bfloat16,
                            'unexpected latent tensor layout')
                    require(bool(torch.isfinite(t).all()), 'nonfinite input latent')
                    tensors.append(t)
                start, local_frames, absolute_frames = frame_plan(mode)
                latent = torch.cat(tensors, dim=1)[:, start:].to(device='cuda:0', dtype=dtype)
                with torch.inference_mode():
                    pixel = vae.decode_to_pixel(latent.contiguous(), use_cache=False)
                require(tuple(pixel.shape) == (1, 1+4*(360-start-1), 3, 480, 832), 'decode shape')
                require(bool(torch.isfinite(pixel).all()), 'nonfinite decoded RGB')
                frames = pixel[0].add(1).mul(127.5).round().clamp(0,255).to(device='cpu', dtype=torch.uint8).contiguous()
                directory = output / 'decoded' / pid / mode
                directory.mkdir(parents=True, exist_ok=False)
                video_path = directory / 'video.mp4'
                write_video(str(video_path), frames.permute(0,2,3,1).contiguous(), fps=24)
                frame_samples = []
                for local, absolute in zip(local_frames, absolute_frames, strict=True):
                    rgb = frames[local,:,::4,::4].permute(1,2,0).contiguous().numpy().tobytes()
                    png(directory / f'frame-{absolute:04d}.png', 208, 120, rgb)
                    frame_samples.append(rgb)
                samples[pid,mode] = frame_samples
                results.append({'prompt_id':pid, 'mode':mode, 'dtype':str(dtype),
                    'latent_start':start, 'local_frame_indices':local_frames, 'absolute_frame_indices':absolute_frames,
                    'checkpoint_parameter_dtypes':checkpoint_dtypes,
                    'video':str(video_path.relative_to(output)), 'video_sha256':digest(video_path)[0],
                    'sample_clipped_channel_fraction':[sum(v in (0,255) for v in b)/len(b) for b in frame_samples]})
                print('DECODE_AUDITED', pid, mode, flush=True)
                del pixel, frames, latent, tensors, t
                vae.model.clear_cache()
                gc.collect(); torch.cuda.empty_cache()
            del vae
            gc.collect(); torch.cuda.empty_cache()
        comparisons = []
        for pid in PROMPTS:
            sheet(output / f'{pid}-precision.png', [samples[pid,'bf16_full'],samples[pid,'fp32_full']])
            comparisons.append({'prompt_id':pid,'reference':'original_mp4_cpu_decode','candidate':'bf16_full',
                'absolute_frame_indices':FRAME_INDICES,'sample_rgb_mae':[
                    sum(abs(x-y) for x,y in zip(a,b,strict=True))/len(a)
                    for a,b in zip(original_samples[pid],samples[pid,'bf16_full'],strict=True)]})
            for other in ('fp32_full','bf16_tail'):
                a,b = samples[pid,'bf16_full'][-1],samples[pid,other][-1]
                comparisons.append({'prompt_id':pid,'reference':'bf16_full','candidate':other,
                    'absolute_frame':1436,'sample_rgb_mae':sum(abs(x-y) for x,y in zip(a,b,strict=True))/len(a)})
        report = {'schema':'motion_saved_latent_vae_audit_v1','source_marker':marker,
            'source_result_sha256':digest(result_path)[0], 'source_evidence':evidence,
            'artifact_spec_sha256':SPEC_SHA,'vae_sha256':digest(artifacts['wan_vae_checkpoint'])[0],
            'gpu_uuid':gpu_uuid,
            'generation_performed':False,'rows':results,'comparisons':comparisons,
            'interpretation':'Decode diagnostic only. Tail reset changes boundary context; compare only after warmup. No quality winner inferred.'}
        with (output/'result.json').open('x') as f: json.dump(report,f,indent=2)
        return report
    finally:
        os.chdir(old_cwd)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True)
    args=p.parse_args(argv)
    from .main_table_baseline_campaign import _model_root, _run_dir
    from .metadata import find_git_root
    try:
        run(args.source_run,_run_dir(),find_git_root(Path.cwd()),_model_root())
    finally:
        shutdown_owned_runtime_helpers(label='saved-latent-vae-audit')


if __name__ == '__main__':
    main()
