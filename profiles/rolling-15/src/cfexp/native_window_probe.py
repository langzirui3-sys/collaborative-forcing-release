"""Native-only rolling-history controls, replaying a sealed stochastic tape.

This diagnoses current checkpoint/configuration; it is not a quality benchmark
or a production attention-policy change.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path

from .motion_video_audit import require, digest, png, sheet
from .motion_vae_audit import PROMPTS, SPEC_SHA
from .model_compatibility import NoiseTape, instrument, first_trace_difference
from .runtime_helpers import shutdown_owned_runtime_helpers

SOURCE_SHA = '0557ad016ca7109af8423877a4fd9b33b860264b'
SOURCE_INDEX = 'dee12a85aa647c9981182067e61fa2791cb83c9afe5f3b1999fadf2b7c404301'
CHUNKS = 60
ARMS = {'window21': (21, 0), 'window21_sink3': (21, 3), 'window42': (42, 0)}
FULL_HISTORY_ARMS = {'window21': (21, 0), 'full_history180': (CHUNKS*3, 0)}
FRAMES = (0, 72, 144, 240, 360, 480, 600, 716)


@contextmanager
def attention_policy(pipeline, window, sink):
    """Change all native configuration sites together; restore even on failure."""
    require(type(window) is int and window >= 3 and window % 3 == 0, 'invalid window')
    require(type(sink) is int and sink >= 0 and sink % 3 == 0 and sink <= window-3, 'invalid sink')
    require(pipeline.kv_cache1 is None and pipeline.crossattn_cache is None, 'live native cache')
    model = pipeline.generator.model
    require(len(model.blocks) == 30, 'unexpected native block count')
    edits = [(pipeline, 'local_attn_size', window), (model, 'local_attn_size', window)]
    for block in model.blocks:
        edits.extend([(block, 'local_attn_size', window),
                      (block.self_attn, 'local_attn_size', window),
                      (block.self_attn, 'sink_size', sink),
                      (block.self_attn, 'max_attention_size', window*pipeline.frame_seq_length)])
    before = [(obj, key, getattr(obj, key)) for obj, key, value in edits]
    try:
        for obj, key, value in edits: setattr(obj, key, value)
        yield
    finally:
        pipeline.kv_cache1 = pipeline.crossattn_cache = None
        for obj, key, value in reversed(before): setattr(obj, key, value)


def verify_full_history_cache(caches, *, expected_end, capacity):
    require(len(caches) == 30, 'full history layer coverage')
    for cache in caches:
        require(int(cache['global_end_index'].item()) == expected_end and
                int(cache['local_end_index'].item()) == expected_end,
                'full history cache rolled or lost tokens')
        require(int(cache['k'].shape[1]) == capacity and
                int(cache['v'].shape[1]) == capacity and expected_end <= capacity,
                'full history cache capacity')


def horizon_plan(chunks, full_history):
    require(type(chunks) is int and chunks in (60, 120), 'unsupported diagnostic horizon')
    arms = ({'window21': (21, 0), f'full_history{chunks*3}': (chunks*3, 0)}
            if full_history else dict(ARMS))
    frames = FRAMES if chunks == 60 else (0, 144, 360, 600, 720, 960, 1200, 1436)
    return arms, frames


def native_arm(backend, prompt, noise, tape_dir, directory, window, sink,
               *, full_history=False):
    from .self_forcing_frame_adapter import StepZeroCapture, LAYER_INDICES, EFFECTIVE_RANK_LAYER
    torch, p = backend.torch, backend.pipeline
    chunks = noise.shape[1] // 3
    require(noise.shape[1] == chunks*3 and chunks > 0, 'invalid noise horizon')
    backend._incremental_draft.cache.clear()
    p.kv_cache1 = p.crossattn_cache = None
    tape = NoiseTape(tape_dir, torch, False)
    with attention_policy(p, window, sink), instrument(backend, directory, tape) as trace:
        def stats(_module, _args, kw, result):
            value = result[1].detach().float()
            require(bool(torch.isfinite(value).all()), 'nonfinite model prediction')
            trace[-1]['prediction_rms'] = value.square().mean().sqrt().item()
            trace[-1]['prediction_max_abs'] = value.abs().max().item()
            if trace[-1]['timestep'] == 0:
                if full_history:
                    verify_full_history_cache(p.kv_cache1,
                        expected_end=(trace[-1]['chunk']+1)*3*p.frame_seq_length,
                        capacity=chunks*3*p.frame_seq_length)
                    trace[-1]['all_layer_history_extent_verified'] = True
                cache = p.kv_cache1[0]
                trace[-1]['layer0_cache'] = {
                    'global_end': int(cache['global_end_index'].item()),
                    'local_end': int(cache['local_end_index'].item()),
                    'capacity': int(cache['k'].shape[1]),
                    'k_rms': cache['k'].float().square().mean().sqrt().item(),
                    'v_rms': cache['v'].float().square().mean().sqrt().item(),
                }
        handle = p.generator.register_forward_hook(stats, with_kwargs=True)
        backend._active_capture = StepZeroCapture(layers=LAYER_INDICES, target_timestep=1000,
            sample_hidden=backend._sample_hidden, rank_sample=backend._rank_sample,
            retain_rank_layer=EFFECTIVE_RANK_LAYER, policy_rank_only=True)
        try:
            with torch.inference_mode():
                _, result = p.inference(noise=noise, text_prompts=[prompt], initial_latent=None,
                    return_latents=True, profile=False, low_memory=False)
            require(result.shape == noise.shape, 'native output layout')
            require(len(backend._active_capture.events) == chunks, 'capture coverage')
            del result
        finally:
            handle.remove()
            backend._active_capture = None
    tape.finish(0, chunks)
    require(len(trace) == chunks*5, 'forward coverage')
    with (directory/'trace.json').open('x') as f:
        json.dump({'forwards': trace, 'noise': tape.calls, 'window': window, 'sink': sink}, f, indent=2)
    gc.collect(); torch.cuda.empty_cache()
    return trace, tape.calls


def run(source, output, repository, model_root, *, full_history_control=False, chunks=60, prompt_id=None):
    arms, sample_frames = horizon_plan(chunks, full_history_control)
    require(prompt_id is None or prompt_id in PROMPTS, 'unknown diagnostic prompt')
    selected_prompts = PROMPTS if prompt_id is None else (prompt_id,)
    from .main_table_merge import _load_frozen_index
    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend, _materialize_source
    from .main_table_baseline_campaign import _gpu_uuid, _gpu_compute_pids
    from .insight2_run import load_run_config
    from .insight2_protocol import load_protocol
    from .main_table_ours_campaign import BOUNDARY_CONFIG
    source, output = source.resolve(strict=True), output.resolve(strict=True)
    require(not source.is_relative_to(output) and not output.is_relative_to(source), 'overlap')
    require(not any((p/n).exists() for p in (output,*output.parents)
        for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')), 'sealed output')
    frozen, marker = _load_frozen_index(source)
    require(marker['git_sha'] == SOURCE_SHA and marker['artifact_index_sha256'] == SOURCE_INDEX, 'source drift')
    attempt = source/'attempts'/marker['attempt_id']
    protocol = load_protocol(load_run_config(repository/BOUNDARY_CONFIG,repo_root=repository).protocol_path)
    prompts = {r['prompt_id']:r['prompt'] for r in protocol.calibration_prompts}
    spec_path = repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    require(digest(spec_path)[0] == SPEC_SHA, 'spec drift')
    spec, verified, module = verify_artifact_spec(spec_path,model_root=model_root,repo_root=repository)
    require(module == 'cfexp.self_forcing_frame_adapter', 'adapter drift')
    artifacts = {r['name']:Path(r['resolved_path']) for r in verified}
    source_root, python_site = _materialize_source(run_dir=output,worker_index=0,artifacts=artifacts,
        config=spec['adapter_config'],scratch_suffix='native-window')
    device = os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(bool(device) and ',' not in device, 'one GPU required')
    uuid = _gpu_uuid(device); require(not _gpu_compute_pids(uuid), 'GPU busy')
    backend = OfficialSelfForcingBackend(source_root=source_root,python_site=python_site,artifacts=artifacts,
        config=spec['adapter_config'],device_name='cuda:0',chunk_count=chunks)
    reports = []
    try:
        from safetensors.torch import load_file
        from .kv_generation_parity import compare
        from .incremental_draft import latent_digest
        torch = backend.torch
        for pid in selected_prompts:
            src = attempt/pid
            directory = output/pid; directory.mkdir(exist_ok=False)
            noise_path = frozen.file(str((src/'initial-noise.safetensors').relative_to(source)), label='initial noise')
            noise = load_file(str(noise_path))['noise'][:,:chunks*3].contiguous().to(backend.device)
            require(noise.shape == (1,chunks*3,16,60,104) and noise.dtype == torch.bfloat16, 'noise layout')
            initial_hash = latent_digest(noise,torch)
            source_trace = json.loads((src/'native_draft/trace.json').read_text())
            traces, tapes, decoded, images, comparisons = {}, {}, [], {}, []
            for arm, (window, sink) in arms.items():
                require(_gpu_compute_pids(uuid) <= {os.getpid()}, 'unrelated GPU workload')
                traces[arm], tapes[arm] = native_arm(backend,prompts[pid],noise,src/'noise-tape',directory/arm,window,sink,
                    full_history=arm.startswith('full_history'))
                require(latent_digest(noise,torch) == initial_hash, 'input mutated')
                require(tapes[arm] == [r for r in source_trace['noise'] if r['chunk'] < chunks], 'noise drift')
                baseline = src/'native_draft' if arm == 'window21' else directory/'window21'
                comparisons.append({'arm':arm,'reference':'sealed_native_draft' if arm == 'window21' else 'window21',
                    'chunks':[{'chunk':i,**compare(load_file(str(baseline/f'{i:03d}.safetensors'))['latent'],
                        load_file(str(directory/arm/f'{i:03d}.safetensors'))['latent'],torch)} for i in range(chunks)]})
                if arm == 'window21':
                    require(all(c['exact'] for c in comparisons[-1]['chunks']), 'baseline failed to reproduce sealed trajectory')
                else:
                    require(all(c['exact'] for c in comparisons[-1]['chunks'][:7]), 'intervention changed pre-roll trajectory')
                latents = torch.cat([load_file(str(directory/arm/f'{i:03d}.safetensors'))['latent'] for i in range(chunks)],dim=1)
                frames = backend.decode_latents(latents)
                require(frames.shape == (chunks*12-3,3,480,832), 'decoded layout')
                video = directory/arm/'video.mp4'; backend.write_video(frames,len(frames),video,24)
                samples = []
                for i in sample_frames:
                    rgb=frames[i,:,::4,::4].permute(1,2,0).contiguous().numpy().tobytes()
                    png(directory/arm/f'frame-{i:04d}.png',208,120,rgb); samples.append(rgb)
                images[arm] = samples
                decoded.append({'arm':arm,'video_sha256':digest(video)[0],
                    'latent_rms_per_frame':latents.float().square().mean(dim=(0,2,3,4)).sqrt().tolist(),
                    'clipped_channel_fraction':[sum(v in (0,255) for v in x)/len(x) for x in samples]})
                del latents,frames
                gc.collect();torch.cuda.empty_cache()
            sheet(directory/'comparison.png',[images[a] for a in arms])
            row={'prompt_id':pid,'input_noise_sha256':initial_hash,'arms':arms,'comparisons':comparisons,
                'decoded':decoded,'first_differences':{a:first_trace_difference(traces['window21'],traces[a])
                    for a in arms if a != 'window21'}}
            with (directory/'result.json').open('x') as f:json.dump(row,f,indent=2)
            reports.append(row)
        report={'schema':'native_window_probe_v1','source_marker':marker,'rows':reports,'gpu_uuid':uuid,
            'full_history_control':full_history_control,
            'chunk_count':chunks,'sample_frame_indices':sample_frames,'formal_quality_result':False,
            'native_source_sha256':digest(source_root/'pipeline/causal_inference.py')[0],
            'attention_source_sha256':digest(source_root/'wan/modules/causal_model.py')[0]}
        with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
        return report
    finally:backend.close()


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--full-history-control',action='store_true')
    parser.add_argument('--chunks',type=int,choices=(60,120),default=60)
    parser.add_argument('--prompt-id',choices=PROMPTS)
    args=parser.parse_args(argv)
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root
    try:run(args.source_run,_run_dir(),find_git_root(Path.cwd()),_model_root(),
            full_history_control=args.full_history_control,chunks=args.chunks,prompt_id=args.prompt_id)
    finally:shutdown_owned_runtime_helpers(label='native-window-probe')

if __name__ == '__main__':main()
