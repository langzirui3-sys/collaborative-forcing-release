"""Native LongLive 1.0 rollout with explicit Krea chunk substitution.

The native LongLive pipeline owns its continuous sampling and KV loop. On a
scheduled chunk only its four denoising model calls are replaced; its native
clean forward commits the accepted Krea latent to LongLive KV before returning
control to Draft. The pure Target arm uses a live native Krea session.
"""
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .cross_model_delay import (SEED, ARMS, HiddenRankRecorder, audit_video,
    geometry, require, schedule, sha, video_interval, write_new)
from .incremental_draft import latent_digest


def launch_target(output, repository, model_root, device):
    from .correction_artifacts import verify_target_artifact_spec, target_artifact_spec_sha256
    from .correction_runtime import _write_backend_config, JsonlWorkerClient
    path = repository/'experiments/insight2-krea14b.artifacts.json'
    _s, records, _m = verify_target_artifact_spec(path, model_root=model_root, repo_root=repository)
    config, digest = _write_backend_config(run_dir=output, role='target', spec_path=path,
        spec_sha256=target_artifact_spec_sha256(path), verified_artifacts=records, chunk_count=120)
    worker = JsonlWorkerClient(run_dir=output, role='target',
        backend_factory='cfexp.cross_model_krea:build_backend', backend_config_path=config,
        backend_config_sha256=digest, visible_device=device, chunk_count=120)
    return worker, records


def build_longlive(output, repository, model_root):
    from .main_table_baseline_campaign import verify_artifact_spec
    from .longlive_adapter import _materialize_source, OfficialLongLiveBackend
    path = repository/'experiments/baseline24-longlive-draft-60s-v2.artifacts.json'
    spec, records, module = verify_artifact_spec(path, method_id='draft_only', model_root=model_root, repository_root=repository)
    require(module == 'cfexp.longlive_adapter', 'wrong Draft adapter')
    artifacts = {r['name']:Path(r['resolved_path']) for r in records}
    validated = SimpleNamespace(run_dir=output, worker_index=0, prompt_id='cross-model',
                                artifacts=artifacts, config=spec['adapter_config'])
    source, assets = _materialize_source(validated)
    backend = OfficialLongLiveBackend(source_root=source, model_root=assets,
                                      artifacts=artifacts, config=spec['adapter_config'])
    return backend, records


def target_request(worker, output, directory, prompt, index, noise_path, *, persistent):
    rel = lambda p: p.relative_to(output).as_posix()
    path = directory/f'{index:03d}.safetensors'
    identity = hashlib.sha256(f'cross-model:{directory.name}:{index}'.encode()).hexdigest()
    payload = {'prompt':prompt, 'chunk_index':index, 'internal_seed':SEED+index,
        'history_latent_paths':[rel(directory/f'{i:03d}.safetensors') for i in range(index)],
        'initial_noise_path':rel(noise_path), 'target_logical_call_id':identity,
        'target_call_scope_id':identity, 'target_call_phase':'route_materialization'}
    if persistent:
        payload['output_latent_path'] = rel(path)
    else:
        payload.update(capture_hidden=False, output_paths={'latent':rel(path)})
    response = worker._request('generate_persistent' if persistent else 'generate', payload)
    require(response['chunk_index'] == index and response['history_chunk_count'] == index, 'Target history cursor drift')
    result = response['result']
    require(result['explicit_initial_noise_verified'] is True, 'unbound Target initial noise')
    require(result['latent']['sha256'] == sha(path), 'Target latent binding mismatch')
    require(result['denoising_schedule'] == [1000.0,750.0,500.0,250.0], 'Target schedule differs from full Target')
    require(len(result['denoising_forward_trace']) == 4, 'Target denoising count drift')
    require('erank_observation' in result, 'Target rank missing')
    write_new(output/'target-evidence'/f'{directory.name}-{index:03d}.json', {'request':payload,'response':response})
    return path, result


def observe_native(backend, worker, *, arm, prompt, output, noise, reference, ownership,
                   target_chunks=None):
    """Delegate unchanged native sampling; commit every external clean block."""
    from safetensors.torch import save_file, load_file
    t, p = backend.torch, backend.pipeline
    directory = output/'arms'/arm
    directory.mkdir(parents=True, exist_ok=False)
    targets = schedule('longlive_krea', arm) if target_chunks is None else tuple(target_chunks)
    require(all(type(i) is int and 0<=i<120 for i in targets)
            and targets==tuple(sorted(set(targets))), 'invalid explicit Target chunk schedule')
    rows, noise_calls, produced = [], [], {}
    current, step, started = None, 0, None
    rank = HiddenRankRecorder(t)
    handle = p.generator.model.blocks[14].register_forward_hook(rank.observe)
    original_forward = p.generator.forward
    original_add = p.scheduler.add_noise
    original_decode = p.vae.decode_to_pixel
    stages = {'decode_seconds':0.0}
    def add(clean, actual_noise, timestep):
        noise_calls.append({'chunk':current,'step':step-1,'sha256':latent_digest(actual_noise,t)})
        return original_add(clean, actual_noise, timestep)
    def forward(*args, **kw):
        nonlocal current, step, started
        i = kw['current_start']//(3*p.frame_seq_length)
        value = float(kw['timestep'].reshape(-1)[0])
        if value > 0:
            if current != i:
                require(len(rows) == i, 'native chunk order drift')
                ownership()
                current, step = i, 0
                t.cuda.synchronize(); started = time.perf_counter()
            require(step < 4, 'too many native denoising calls')
            if i in targets:
                if step == 0:
                    path, evidence = target_request(worker, output, directory, prompt, i,
                        output/'inputs'/f'{i:03d}.safetensors', persistent=False)
                    latent = load_file(str(path))['latent'].to(backend.device)
                    require(tuple(latent.shape) == (1,3,16,60,104), 'external latent geometry')
                    produced[i] = (latent, evidence)
                result = (None, produced[i][0])
            else:
                if step == 0:
                    rank.begin(producer='longlive_1.3b', chunk=i)
                rank.active = step == 0
                result = original_forward(*args, **kw)
                rank.active = False
                if step == 0:
                    produced[i] = (None, {'erank_observation':rank.end()})
            step += 1
            return result
        require(current == i and step == 4, 'clean-cache update before completed denoising')
        latent = kw['noisy_image_or_video']
        if i in targets:
            require(t.equal(latent, produced[i][0]), 'clean Draft cache did not receive accepted Target latent')
        # This is the upstream native clean forward, including width-three t=0.
        result = original_forward(*args, **kw)
        require(all(int(c['global_end_index'].item()) == (i+1)*3*p.frame_seq_length for c in p.kv_cache1), 'Draft KV cursor differs from canonical history')
        clean = latent.detach().cpu().contiguous()
        require(bool(t.isfinite(clean).all()), 'nonfinite LongLive clean latent')
        path = directory/f'{i:03d}.safetensors'
        if i not in targets:
            save_file({'latent':clean},str(path))
        else:
            require(t.equal(load_file(str(path))['latent'], clean), 'external clean changed at Draft commit')
        comparison = None
        if reference is not None and i < min(targets, default=120):
            comparison = t.equal(load_file(str(reference/f'{i:03d}.safetensors'))['latent'],clean)
            require(comparison, 'pre-intervention prefix differs from pure Draft')
        t.cuda.synchronize()
        rows.append({'chunk':i,'producer':'target' if i in targets else 'draft',
            'video_interval_seconds':video_interval('longlive_krea',i),
            'generation_with_instrumentation_seconds':time.perf_counter()-started,
            'latent_sha256':latent_digest(clean,t), 'erank_observation':produced[i][1]['erank_observation'],
            'native_clean_cache_update':True, 'clean_cache_timestep_shape':list(kw['timestep'].shape),
            'native_prefix_exact':comparison})
        del produced[i]
        print('CROSS_MODEL_CHUNK', 'longlive_krea', arm, i, flush=True)
        return result
    def decode(latent, **kw):
        t.cuda.synchronize(); start=time.perf_counter()
        result=original_decode(latent,**kw)
        t.cuda.synchronize(); stages['decode_seconds']+=time.perf_counter()-start
        return result
    p.generator.forward=forward; p.scheduler.add_noise=add; p.vae.decode_to_pixel=decode
    try:
        pipeline_started=time.perf_counter()
        with t.inference_mode():
            video, latents = p.inference(noise=noise, text_prompts=[prompt], return_latents=True,
                                         profile=False, low_memory=False)
        require(len(rows)==120 and len(noise_calls)==360, 'native trajectory coverage drift')
        require([x['chunk'] for x in rows if x['producer']=='target']==list(targets), 'forced Target budget drift')
        if reference is not None:
            import json
            require(noise_calls==json.loads((reference/'trace.json').read_text())['renoise'], 'native actual scheduler noise differs across arms')
        # Native pipeline returns [B,T,C,H,W], already normalized to [0,1].
        frames=(video[0].mul(255).round().clamp(0,255).to(device='cpu',dtype=t.uint8).contiguous())
        require(tuple(frames.shape)==(1437,3,480,832), 'native LongLive video layout')
        stages['pipeline_with_monitoring_wall_seconds']=time.perf_counter()-pipeline_started
        p.vae.model.clear_cache()
        write_new(directory/'trace.json', {'chunks':rows,'renoise':noise_calls,'native_pipeline':True,
                   'target_chunks':list(targets),'rank_routes_generation':False})
        return directory, frames, rows, stages
    finally:
        handle.remove(); p.generator.forward=original_forward; p.scheduler.add_noise=original_add
        p.vae.decode_to_pixel=original_decode
        p.kv_cache1=None; p.crossattn_cache=None


def full_target(backend, worker, *, prompt, output, ownership):
    from safetensors.torch import load_file
    t = backend.torch
    directory=output/'arms/target_only';directory.mkdir(parents=True,exist_ok=False)
    pipeline_started=time.perf_counter()
    session=worker.start_trajectory(prompt_id='cross-model-target-only',prompt=prompt,seed=SEED,chunk_count=120)
    rows=[];latents=[]
    try:
        for i in range(120):
            ownership();start=time.perf_counter()
            path,evidence=target_request(worker,output,directory,prompt,i,output/'inputs'/f'{i:03d}.safetensors',persistent=True)
            latent=load_file(str(path))['latent'];latents.append(latent)
            rows.append({'chunk':i,'producer':'target','video_interval_seconds':video_interval('longlive_krea',i),
                'generation_with_instrumentation_seconds':time.perf_counter()-start,
                'latent_sha256':latent_digest(latent,t),'erank_observation':evidence['erank_observation'],
                'persistent_native_target_session':True})
            print('CROSS_MODEL_CHUNK','longlive_krea','target_only',i,flush=True)
    finally:
        worker.finish_trajectory(session_id=session['session_id'])
    history=t.cat(latents,dim=1).to(backend.device)
    t.cuda.synchronize();start=time.perf_counter()
    with t.inference_mode():video=backend.pipeline.vae.decode_to_pixel(history,use_cache=False)
    t.cuda.synchronize();decode=time.perf_counter()-start
    frames=video[0].add(1).mul(127.5).round().clamp(0,255).to(device='cpu',dtype=t.uint8).contiguous()
    require(tuple(frames.shape)==(1437,3,480,832),'Target decode layout')
    backend.pipeline.vae.model.clear_cache()
    write_new(directory/'trace.json',{'chunks':rows,'native_pipeline':True,'rank_routes_generation':False})
    return directory,frames,rows,{'decode_seconds':decode,'pipeline_with_monitoring_wall_seconds':time.perf_counter()-pipeline_started}


def save_arm(backend, directory, frames, rows, stages):
    from torchvision.io import write_video
    from PIL import Image
    Image.fromarray(frames[0].permute(1,2,0).numpy()).save(directory/'first-frame.png')
    start=time.perf_counter()
    write_video(str(directory/'video.mp4'),frames.permute(0,2,3,1).contiguous(),fps=24,
                video_codec='libx264',options={'crf':'18'})
    encode=time.perf_counter()-start
    for second in range(60):
        image=Image.fromarray(frames[second*24].permute(1,2,0).numpy()).resize((416,240))
        image.save(directory/f'second-{second:02d}.png')
    audit=audit_video(directory/'video.mp4',1437,24)
    result={'arm':directory.name,'video':audit,'chunk_count':120,
        'generation_with_instrumentation_seconds':sum(r['generation_with_instrumentation_seconds'] for r in rows),
        **stages,'encode_seconds':encode,'visual_review_status':'pending_rank_hidden_review'}
    write_new(directory/'result.json',result)
    return result


def run(plan, item, output, repository, model_root, ownership, devices, *, arms=None, target_schedules=None):
    from safetensors.torch import save_file
    initialization_started=time.perf_counter()
    target, records = launch_target(output,repository,model_root,devices[1])
    prior=os.environ.get('CUDA_VISIBLE_DEVICES')
    backend=None
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=devices[0]
        backend, draft_records = build_longlive(output,repository,model_root)
        initialization_seconds=time.perf_counter()-initialization_started
        t=backend.torch
        t.manual_seed(SEED);t.cuda.manual_seed_all(SEED)
        noise=t.randn((1,360,16,60,104),device=backend.device,dtype=t.bfloat16)
        noise_hash=latent_digest(noise,t)
        (output/'inputs').mkdir()
        for i in range(120):save_file({'noise':noise[:,i*3:(i+1)*3].cpu().contiguous()},str(output/'inputs'/f'{i:03d}.safetensors'))
        write_new(output/'provenance.json',{'draft_artifacts':draft_records,'target_artifacts':records,
            'native_load':backend.native_load,'initial_noise_sha256':noise_hash,
            'target_schedule':[1000,750,500,250],'Draft_external_clean_policy':'native width-three clean-cache commit'})
        warmup_started=time.perf_counter()
        with t.inference_mode():
            warm_video,warm_latents=backend.pipeline.inference(noise=noise[:,:3],text_prompts=[item['prompt']],return_latents=True,profile=False,low_memory=False)
        del warm_video,warm_latents
        backend.pipeline.vae.model.clear_cache();backend.pipeline.kv_cache1=None;backend.pipeline.crossattn_cache=None
        warm_directory=output/'warmup';warm_directory.mkdir()
        # A standalone warmup must not claim completion of a 120-block native
        # persistent trajectory. Use the ordinary single-block request.
        target_request(target,output,warm_directory,item['prompt'],0,output/'inputs/000.safetensors',persistent=False)
        warmup_seconds=time.perf_counter()-warmup_started
        results=[];reference=None
        # Pure Draft is generated natively first to bind every later prefix and RNG call.
        selected=ARMS if arms is None else tuple(arms)
        others=list(selected[1:]);shift=plan['prompts'].index(item)%len(others)
        order=['draft_only']+others[shift:]+others[:shift]
        for arm in order:
            t.manual_seed(SEED);t.cuda.manual_seed_all(SEED)
            # Match the RNG state after the one shared native initial-noise draw.
            check=t.randn(noise.shape,device=backend.device,dtype=t.bfloat16)
            require(latent_digest(check,t)==noise_hash,'initial noise changed');del check
            ownership(target.process.pid)
            if arm=='target_only':data=full_target(backend,target,prompt=item['prompt'],output=output,ownership=lambda:ownership(target.process.pid))
            else:data=observe_native(backend,target,arm=arm,prompt=item['prompt'],output=output,noise=noise,
                    reference=reference,ownership=lambda:ownership(target.process.pid),
                    target_chunks=None if target_schedules is None else target_schedules[arm])
            directory,frames,rows,stages=data
            results.append(save_arm(backend,directory,frames,rows,stages))
            if arm=='draft_only':reference=directory
            require(latent_digest(noise,t)==noise_hash,'shared initial noise mutated')
            del frames;gc.collect();t.cuda.empty_cache()
        import json
        for arm in selected:
            if arm in ('draft_only','target_only'):continue
            for i in (schedule('longlive_krea',arm) if target_schedules is None else target_schedules[arm]):
                candidate=json.loads((output/'target-evidence'/f'{arm}-{i:03d}.json').read_text())['response']['result']
                baseline=json.loads((output/'target-evidence'/f'target_only-{i:03d}.json').read_text())['response']['result']
                require(candidate['renoise_tensor_sha256']==baseline['renoise_tensor_sha256'],'Target actual stochastic inputs differ by strategy')
        return {'arms':results,'execution_order':order,'initial_noise_sha256':noise_hash,'initialization_seconds':initialization_seconds,'warmup_seconds':warmup_seconds,'all_actual_noise_checks_passed':True}
    finally:
        if backend is not None:backend.close()
        target.close()
        if prior is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=prior
