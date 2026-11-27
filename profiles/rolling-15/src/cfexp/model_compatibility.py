"""Development-only native/chunked and Draft/Target-root factorial diagnostic.

The untouched native Draft call supplies an explicit stochastic noise tape.
Other arms replay the tape, rather than relying on equal seeds across different
call boundaries. Generation, VAE decoding and quality conclusions stay separate.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
from contextlib import contextmanager

from .motion_video_audit import require, digest, png, sheet, FRAME_INDICES
from .motion_vae_audit import PROMPTS, SPEC_SHA
from .kv_generation_parity import compare
from .runtime_helpers import shutdown_owned_runtime_helpers

SOURCE_SHA = 'bb857cd2f76748c1fe4862fc2cc6c88b7e401fb6'
SOURCE_INDEX = '18ab989f5529b737494f7eb95d1334a7728a84b0f1d0e37d4c891393813fc88a'
ARMS = ('native_draft', 'native_target', 'chunked_draft', 'chunked_target')


def first_trace_difference(left, right):
    """Compare generation steps, excluding prefix replay and clean KV updates."""
    a = [r for r in left if r['timestep'] > 0]
    b = [r for r in right if r['timestep'] > 0]
    require(len(a) == len(b), 'different number of generation steps')
    for x, y in zip(a, b, strict=True):
        require((x['chunk'], x['step'], x['timestep']) ==
                (y['chunk'], y['step'], y['timestep']), 'schedule mismatch')
        for field in ('input_sha256', 'clean_sha256'):
            if x[field] != y[field]:
                return {'chunk': x['chunk'], 'step': x['step'], 'field': field,
                        'reference_sha256': x[field], 'candidate_sha256': y[field]}
    return None


class NoiseTape:
    """Capture/replay the scheduler's actual noise argument, with strict cursors."""
    def __init__(self, directory, torch, record):
        self.directory, self.torch, self.record = directory, torch, record
        self.chunk = None
        self.step = 0
        self.calls = []

    def start(self, chunk):
        if self.chunk is not None:
            require(self.step == 3, 'incomplete previous re-noise block')
        self.chunk, self.step = chunk, 0

    def apply(self, original, clean, noise, timestep):
        from safetensors.torch import save_file, load_file
        from .incremental_draft import latent_digest
        require(self.chunk is not None and self.step < 3, 'unexpected scheduler call')
        path = self.directory / f'{self.chunk:03d}-{self.step}.safetensors'
        if self.record:
            require(not path.exists(), 'noise tape overwrite')
            save_file({'noise': noise.detach().cpu().contiguous()}, str(path))
        else:
            saved = load_file(str(path))['noise']
            require(saved.shape == noise.shape and saved.dtype == noise.dtype, 'noise layout drift')
            noise = saved.to(device=noise.device)
        self.calls.append({'chunk': self.chunk, 'step': self.step,
            'sha256': digest(path)[0], 'tensor_sha256': latent_digest(noise, self.torch),
            'timestep': float(timestep.reshape(-1)[0])})
        self.step += 1
        return original(clean, noise, timestep)

    def finish(self, start, stop):
        require(self.step == 3, 'last noise block incomplete')
        require([(r['chunk'], r['step']) for r in self.calls] ==
                [(i, j) for i in range(start, stop) for j in range(3)], 'noise tape coverage')


@contextmanager
def instrument(backend, directory, tape, *, native_clean_shape=False):
    """All patches are restored on failure; every observed tensor is hashed."""
    from .incremental_draft import latent_digest
    torch, p = backend.torch, backend.pipeline
    trace = []
    active = {}
    scheduler = p.scheduler.add_noise
    decode = p.vae.decode_to_pixel
    directory.mkdir(parents=True, exist_ok=False)

    def pre(_module, _args, kw):
        timestep = kw['timestep']
        chunk = int(kw['current_start']) // (3 * p.frame_seq_length)
        value = float(timestep.reshape(-1)[0])
        if value > 0:
            step = active.get(chunk, 0)
            if step == 0:
                tape.start(chunk)
            active[chunk] = step + 1
        else:
            step = None
            # Diagnostic only: match native continuous clean-update shape.
            # Never alter prefix prefill (no preceding positive forwards).
            if native_clean_shape and active.get(chunk) == 4:
                kw['timestep'] = timestep.expand(timestep.shape[0], 3).contiguous()
        trace.append({'chunk': chunk, 'step': step, 'timestep': value,
            'timestep_shape': list(kw['timestep'].shape),
            'position': int(kw['current_start']),
            'input_sha256': latent_digest(kw['noisy_image_or_video'], torch)})
        return _args, kw

    def post(_module, _args, kw, result):
        from safetensors.torch import save_file
        trace[-1]['clean_sha256'] = latent_digest(result[1], torch)
        if trace[-1]['step'] == 3:
            chunk = trace[-1]['chunk']
            save_file({'latent': result[1].detach().cpu().contiguous()},
                      str(directory / f'{chunk:03d}.safetensors'))
            print('COMPAT_CHUNK', directory.parent.name, directory.name, chunk, flush=True)

    prehandle = p.generator.register_forward_pre_hook(pre, with_kwargs=True)
    handle = p.generator.register_forward_hook(post, with_kwargs=True)
    p.scheduler.add_noise = lambda clean, noise, timestep: tape.apply(scheduler, clean, noise, timestep)
    p.vae.decode_to_pixel = lambda latent, **kw: torch.zeros((1, 1, 3, 1, 1), device=latent.device)
    try:
        yield trace
    finally:
        prehandle.remove(); handle.remove()
        p.scheduler.add_noise = scheduler
        p.vae.decode_to_pixel = decode


def generate_arm(backend, *, arm, prompt, initial_noise, root, tape_dir, output,
                 native_clean_shape=False):
    from safetensors.torch import save_file
    torch, p = backend.torch, backend.pipeline
    require(arm in ARMS, 'unknown arm')
    target = arm.endswith('target')
    start, stop = (1 if target else 0), initial_noise.shape[1] // 3
    backend._incremental_draft.cache.clear()
    p.kv_cache1 = p.crossattn_cache = None
    tape = NoiseTape(tape_dir, torch, record=arm == 'native_draft')
    directory = output / arm
    if arm == 'native_draft': tape_dir.mkdir(parents=True, exist_ok=False)
    with instrument(backend, directory, tape, native_clean_shape=native_clean_shape) as trace:
        if target: save_file({'latent': root.contiguous()}, str(directory/'000.safetensors'))
        if arm.startswith('native'):
            from .self_forcing_frame_adapter import StepZeroCapture, LAYER_INDICES, EFFECTIVE_RANK_LAYER
            backend._active_capture = StepZeroCapture(layers=LAYER_INDICES,target_timestep=1000,
                sample_hidden=backend._sample_hidden,rank_sample=backend._rank_sample,
                retain_rank_layer=EFFECTIVE_RANK_LAYER,policy_rank_only=True)
            try:
                with torch.inference_mode():
                    _, generated = p.inference(noise=initial_noise[:, start*3:], text_prompts=[prompt],
                        initial_latent=root.to(backend.device) if target else None,
                        return_latents=True, profile=False, low_memory=False)
                require(len(backend._active_capture.events) == stop-start, 'native capture coverage')
            finally:
                backend._active_capture = None
            require(generated.shape == initial_noise.shape, 'native output shape')
            if target: require(torch.equal(generated[:, :3].cpu(), root), 'native root changed')
            del generated
        else:
            history = root.clone() if target else None
            for i in range(start, stop):
                result = backend.generate_chunk(prompt=prompt, history_latents=history,
                    initial_noise=initial_noise[:, i*3:(i+1)*3], internal_seed=20260911,
                    chunk_index=i, capture_hidden=False,
                    history_kv_widths=((1,)+(3,)*(i-1) if target else (3,)*i))
                require(result['kv_reuse']['replayed_chunks'] == (1 if target and i == 1 else 0),
                        'unexpected replay in chunked path')
                history = result['latent'] if history is None else torch.cat((history,result['latent']),dim=1)
            del history
    tape.finish(start, stop)
    require(len([r for r in trace if r['timestep'] > 0]) == 4*(stop-start), 'trace coverage')
    with (directory/'trace.json').open('x') as f: json.dump({'forwards':trace,'noise':tape.calls},f,indent=2)
    backend._incremental_draft.cache.clear()
    p.kv_cache1 = p.crossattn_cache = None
    gc.collect(); torch.cuda.empty_cache()
    return trace, tape.calls


def run(source, output, repository, model_root):
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
    require(marker['git_sha'] == SOURCE_SHA and marker['artifact_index_sha256'] == SOURCE_INDEX,'source drift')
    attempt = source/'attempts'/marker['attempt_id']
    roots = {pid:frozen.file(str((attempt/'source-latents'/pid/'000.safetensors').relative_to(source)),
                           label='canonical Target root') for pid in PROMPTS}
    protocol = load_protocol(load_run_config(repository/BOUNDARY_CONFIG,repo_root=repository).protocol_path)
    prompts = {r['prompt_id']:r['prompt'] for r in protocol.calibration_prompts}
    spec_path = repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    require(digest(spec_path)[0] == SPEC_SHA, 'spec drift')
    spec, verified, module = verify_artifact_spec(spec_path,model_root=model_root,repo_root=repository)
    require(module == 'cfexp.self_forcing_frame_adapter', 'adapter drift')
    artifacts = {r['name']:Path(r['resolved_path']) for r in verified}
    source_root, python_site = _materialize_source(run_dir=output,worker_index=0,artifacts=artifacts,
        config=spec['adapter_config'],scratch_suffix='model-compatibility')
    device = os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(bool(device) and ',' not in device, 'one GPU required')
    uuid = _gpu_uuid(device); require(not _gpu_compute_pids(uuid),'GPU busy')
    backend = OfficialSelfForcingBackend(source_root=source_root,python_site=python_site,artifacts=artifacts,
        config=spec['adapter_config'],device_name='cuda:0',chunk_count=120)
    reports = []
    try:
        torch = backend.torch
        from safetensors.torch import save_file, load_file
        from .incremental_draft import latent_digest
        for pid in PROMPTS:
            directory = output/pid; directory.mkdir(exist_ok=False)
            root = load_file(str(roots[pid]))['latent']
            require(root.shape == (1,3,16,60,104) and root.dtype == torch.bfloat16,'root layout')
            # Match upstream CLI: seed, full initial-noise draw, continuous RNG.
            import random
            import numpy as np
            random.seed(20260911); np.random.seed(20260911)
            torch.manual_seed(20260911); torch.cuda.manual_seed_all(20260911)
            noise = torch.randn((1,360,16,60,104),device=backend.device,dtype=torch.bfloat16)
            save_file({'noise':noise.cpu()},str(directory/'initial-noise.safetensors'))
            traces, tapes = {}, {}
            input_binding = (latent_digest(noise,torch), latent_digest(root,torch))
            for arm in ARMS:
                require(_gpu_compute_pids(uuid) <= {os.getpid()},'unrelated GPU workload')
                traces[arm], tapes[arm] = generate_arm(backend,arm=arm,prompt=prompts[pid],
                    initial_noise=noise,root=root,tape_dir=directory/'noise-tape',output=directory)
            require(input_binding == (latent_digest(noise,torch),latent_digest(root,torch)), 'shared input mutated')
            require(tapes['native_target'] == [r for r in tapes['native_draft'] if r['chunk'] > 0],
                    'root intervention changed future noise')
            comparisons = []
            for root_name in ('draft','target'):
                left, right = 'native_'+root_name, 'chunked_'+root_name
                rows=[]
                for i in range(120):
                    a=load_file(str(directory/left/f'{i:03d}.safetensors'))['latent']
                    b=load_file(str(directory/right/f'{i:03d}.safetensors'))['latent']
                    rows.append({'chunk':i,**compare(a,b,torch)})
                first=first_trace_difference(traces[left],traces[right])
                comparisons.append({'reference':left,'candidate':right,'first_step_difference':first,'chunks':rows})
                require(tapes[left] == tapes[right], 'paired noise tapes differ')
                if first is not None:
                    extra = directory/'native-clean-shape'; extra.mkdir(exist_ok=True)
                    xt, xn = generate_arm(backend,arm=right,prompt=prompts[pid],initial_noise=noise,
                        root=root,tape_dir=directory/'noise-tape',output=extra,native_clean_shape=True)
                    require(xn == tapes[left],'diagnostic noise drift')
                    comparisons.append({'reference':left,'candidate':'native-clean-shape/'+right,
                        'first_step_difference':first_trace_difference(traces[left],xt),'chunks':[
                            {'chunk':i,**compare(load_file(str(directory/left/f'{i:03d}.safetensors'))['latent'],
                            load_file(str(extra/right/f'{i:03d}.safetensors'))['latent'],torch)} for i in range(120)]})
            root_effects = []
            for mode in ('native','chunked'):
                root_effects.append({'mode':mode,'interpretation':'different roots, independently evolving histories',
                    'chunks':[{'chunk':i,**compare(
                        load_file(str(directory/(mode+'_draft')/f'{i:03d}.safetensors'))['latent'],
                        load_file(str(directory/(mode+'_target')/f'{i:03d}.safetensors'))['latent'],torch)}
                        for i in range(120)]})
            # Same decoder and native scaling for every trajectory, no scoring claim.
            samples, decoded = {}, []
            arms = list(ARMS) + [c['candidate'] for c in comparisons if '/' in c['candidate']]
            for arm in arms:
                latents = torch.cat([load_file(str(directory/arm/f'{i:03d}.safetensors'))['latent']
                                     for i in range(120)],dim=1)
                energy = latents.float().square().mean(dim=(0,2,3,4)).sqrt().tolist()
                frames = backend.decode_latents(latents)
                require(frames.shape == (1437,3,480,832),'decoded geometry')
                video = directory/arm/'video.mp4'
                backend.write_video(frames,len(frames),video,24)
                sampled=[]
                for i in FRAME_INDICES:
                    rgb=frames[i,:,::4,::4].permute(1,2,0).contiguous().numpy().tobytes()
                    png(directory/arm/f'frame-{i:04d}.png',208,120,rgb);sampled.append(rgb)
                samples[arm]=sampled
                decoded.append({'arm':arm,'video':str(video.relative_to(output)),
                    'video_sha256':digest(video)[0],'latent_rms_per_frame':energy,
                    'clipped_channel_fraction':[sum(v in (0,255) for v in x)/len(x) for x in sampled]})
                del latents,frames
                gc.collect();torch.cuda.empty_cache()
            sheet(directory/'comparison.png',[samples[a] for a in arms])
            report={'prompt_id':pid,'target_root_sha256':latent_digest(root,torch),
                'arms_in_contact_sheet_order':arms,'comparisons':comparisons,'root_effects':root_effects,'decoded':decoded,
                'initial_noise_sha256':digest(directory/'initial-noise.safetensors')[0]}
            with (directory/'result.json').open('x') as f:json.dump(report,f,indent=2)
            reports.append(report)
        report={'schema':'model_compatibility_v1','source_marker':marker,'rows':reports,'gpu_uuid':uuid,
            'native_source_sha256':digest(source_root/'pipeline/causal_inference.py')[0],
            'artifact_spec_sha256':SPEC_SHA,'chunk_count':120,
            'noise_policy':'untouched native Draft RNG tape; replay explicit scheduler noises in other arms',
            'formal_quality_result':False}
        with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
        return report
    finally: backend.close()


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True)
    args=p.parse_args(argv)
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root
    try:run(args.source_run,_run_dir(),find_git_root(Path.cwd()),_model_root())
    finally:shutdown_owned_runtime_helpers(label='model-compatibility')

if __name__ == '__main__': main()
