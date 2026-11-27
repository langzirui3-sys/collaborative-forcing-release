"""Forced Target interventions on the sealed native Draft noise tape.

This is a development intervention, with no acceptance gate, tuning, or VBench
claim. Every arm starts from a Draft root. Native baseline parity is mandatory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from .motion_video_audit import require, digest, png
from .motion_vae_audit import PROMPTS, SPEC_SHA
from .model_compatibility import NoiseTape, instrument
from .native_window_probe import SOURCE_SHA, SOURCE_INDEX
from .stage_timing import timing_session
from .runtime_helpers import shutdown_owned_runtime_helpers

CHUNKS = 120
SEED = 20260911
ARMS = {'draft_only': None, 'one_target_10s': 20, 'one_target_15s': 30}
EARLY_PREVENTION_ARMS = {
    'draft_only': (),
    'prevention_one_5s': (10,),
    'prevention_three_5s': (10,11,12),
}
PREVENTION_RECOVERY_ARMS = {
    'draft_only': (),
    'early_one_10s': (20,),
    'early_three_10s': (20,21,22),
    'late_one_25s': (50,),
    'late_three_25s': (50,51,52),
}
SAMPLE_FRAMES = (0, 120, 240, 288, 336, 360, 384, 432, 480, 600, 720, 960, 1200, 1436)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def video_time(chunk):
    require(type(chunk) is int and 0 <= chunk < CHUNKS, 'invalid chunk')
    return max(0, 12 * chunk - 3) / 24


def intervention_schedule(intervention):
    values = () if intervention is None else ((intervention,) if type(intervention) is int else intervention)
    require(isinstance(values, tuple), 'schedule must be an immutable tuple')
    require(all(type(i) is int and 0 < i < CHUNKS-1 for i in values), 'invalid intervention chunk')
    require(tuple(sorted(set(values))) == values, 'schedule must be unique and ordered')
    return values


def validate_noise_calls(calls, reference, intervention):
    schedule = intervention_schedule(intervention)
    expected = [r for r in reference if r['chunk'] not in schedule]
    require(calls == expected, 'future Draft scheduler noise changed')


def contact_sheet(path, rows):
    require(bool(rows) and bool(rows[0]), 'empty comparison sheet')
    columns = len(rows[0])
    require(all(len(row) == columns for row in rows), 'ragged comparison sheet')
    require(all(len(rgb) == 208*120*3 for row in rows for rgb in row), 'invalid sample RGB')
    pixels = bytearray()
    for row in rows:
        for y in range(120):
            for rgb in row:
                pixels.extend(rgb[y*208*3:(y+1)*208*3])
    png(path,208*columns,120*len(rows),pixels)


def draft_rank_observation(result, index):
    from .self_forcing_frame_adapter import validate_effective_rank_evidence
    rank = validate_effective_rank_evidence(result['rank_evidence'],
        expected_rank=result['rank'], expected_eigenvalue_count=256)
    return {'model': 'draft', 'layer_index_zero_based': 14,
            'timestep': 1000, 'rank': rank, 'evidence': result['rank_evidence'],
            'video_seconds': video_time(index)}


def run_arm(backend, *, prompt, noise, tape_dir, directory, source_dir,
            intervention, target_generate, ownership_check, internal_seed=SEED):
    """Keep native stochastic inputs fixed, force the scheduled external clean latents.

    A Target-produced block is reconstructed with external-prefix width one;
    Draft blocks retain native generated-clean width three. This provenance is
    passed explicitly on every continuation, never inferred from matching bytes.
    """
    from safetensors.torch import load_file
    from .incremental_draft import latent_digest
    from .kv_generation_parity import compare
    torch, pipeline = backend.torch, backend.pipeline
    require(noise.shape[1] == CHUNKS * 3, 'wrong noise horizon')
    schedule = intervention_schedule(intervention)
    first = schedule[0] if schedule else CHUNKS
    backend._incremental_draft.cache.clear()
    pipeline.kv_cache1 = pipeline.crossattn_cache = None
    tape = NoiseTape(tape_dir, torch, False)
    history, widths, target_calls, rows = None, [], [], []
    source_trace = json.loads((source_dir/'trace.json').read_text())
    with instrument(backend, directory, tape) as trace:
        for index in range(CHUNKS):
            ownership_check()
            initial = noise[:, index*3:(index+1)*3].contiguous()
            started = time.perf_counter()
            rank_observation = None
            if index in schedule:
                path = directory/f'{index:03d}.safetensors'
                target = target_generate(index, directory, initial, path)
                latent = load_file(str(path))['latent']
                require(tuple(latent.shape) == (1,3,16,60,104) and latent.dtype == torch.bfloat16,
                        'Target output layout')
                target_calls.append(target)
                backend._incremental_draft.cache.clear()
                pipeline.kv_cache1 = pipeline.crossattn_cache = None
                widths.append(1)
                reuse = None
            else:
                with timing_session(directory, 'intervention-draft',
                        {'chunk': index, 'video_seconds': video_time(index),
                         'arm': directory.name}):
                    result = backend.generate_chunk(prompt=prompt, history_latents=history,
                        initial_noise=initial, internal_seed=internal_seed, chunk_index=index,
                        capture_hidden=False, history_kv_widths=tuple(widths))
                latent = result['latent'].detach().cpu().contiguous()
                reuse = result['kv_reuse']
                rank_observation = draft_rank_observation(result, index)
                # One cold reconstruction immediately after the intervention,
                # then actual KV reuse. A cache miss elsewhere is an error.
                expected_replays = index if index-1 in schedule else 0
                require(reuse['replayed_chunks'] == expected_replays,
                        'unexpected Draft cache reconstruction')
                widths.append(3)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - started
            expected = load_file(str(source_dir/f'{index:03d}.safetensors'))['latent']
            comparison = compare(latent, expected, torch)
            if index < first:
                require(comparison['exact'], 'Draft baseline or pre-intervention prefix drifted')
            require(bool(torch.isfinite(latent).all()), 'nonfinite latent')
            history = latent if history is None else torch.cat((history, latent), dim=1)
            rows.append({'chunk':index, 'producer':'target' if index in schedule else 'draft',
                         'generation_with_instrumentation_seconds':elapsed,
                         'draft_erank_observation':rank_observation,
                         'native_reference':comparison, 'latent_sha256':latent_digest(latent, torch),
                         'history_kv_widths':list(widths), 'kv_reuse':reuse})
            print('RECOVERY_CHUNK', directory.parent.name, directory.name, index, flush=True)
    require(len(target_calls) == len(schedule), 'forced Target budget violated')
    validate_noise_calls(tape.calls, source_trace['noise'], intervention)
    require(history.shape == noise.shape, 'incomplete recovered history')
    record = {'intervention_chunks':list(schedule), 'intervention_chunk':schedule[0] if schedule else None,
              'intervention_video_seconds':video_time(schedule[0]) if schedule else None,
              'forced_acceptance':bool(schedule), 'target_calls':target_calls,
              'chunks':rows, 'forwards':trace, 'noise':tape.calls,
              'initial_noise_sha256':latent_digest(noise, torch)}
    write_json(directory/'trace.json', record)
    backend._incremental_draft.cache.clear()
    pipeline.kv_cache1 = pipeline.crossattn_cache = None
    gc.collect(); torch.cuda.empty_cache()
    return history, record


def run(source, output, repository, model_root, prompt_id, *, prevention_recovery=False, early_prevention=False, reference_run=None, reference_index=None):
    require(not (prevention_recovery and early_prevention), 'choose one intervention study')
    schedules = EARLY_PREVENTION_ARMS if early_prevention else (PREVENTION_RECOVERY_ARMS if prevention_recovery else ARMS)
    from .main_table_merge import _load_frozen_index
    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend, _materialize_source
    from .main_table_baseline_campaign import _gpu_uuid, _gpu_compute_pids, _visible_devices
    from .correction_artifacts import verify_target_artifact_spec, target_artifact_spec_sha256
    from .correction_runtime import launch_target_generation_worker
    from .insight2_run import load_run_config
    from .insight2_protocol import load_protocol
    from .main_table_ours_campaign import BOUNDARY_CONFIG

    source, output = source.resolve(strict=True), output.resolve(strict=True)
    require(prompt_id in PROMPTS, 'unknown development prompt')
    require(not source.is_relative_to(output) and not output.is_relative_to(source), 'overlap')
    require(not any((p/n).exists() for p in (output,*output.parents)
        for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')), 'sealed output')
    frozen, marker = _load_frozen_index(source)
    require(marker['git_sha'] == SOURCE_SHA and marker['artifact_index_sha256'] == SOURCE_INDEX,
            'source seal drift')
    reference_traces = {}
    if reference_run is not None:
        reference_run = reference_run.resolve(strict=True)
        ref_frozen, ref_marker = _load_frozen_index(reference_run)
        require(ref_marker['artifact_index_sha256'] == reference_index, 'paired reference index drift')
        for arm in schedules:
            path = reference_run/'attempts'/ref_marker['attempt_id']/'arms'/arm/'trace.json'
            ref_frozen.file(path.relative_to(reference_run).as_posix(), label='paired trajectory')
            rows = json.loads(path.read_text())['chunks']
            require(len(rows) == CHUNKS and [r['chunk'] for r in rows] == list(range(CHUNKS)),
                    'paired reference chunk coverage')
            reference_traces[arm] = rows
    source_attempt = source/'attempts'/marker['attempt_id']
    src = source_attempt/prompt_id
    # Verify every consumed native latent, explicit noise and source trace.
    required = [src/'initial-noise.safetensors', src/'native_draft/trace.json']
    required += [src/'native_draft'/f'{i:03d}.safetensors' for i in range(CHUNKS)]
    required += [src/'noise-tape'/f'{i:03d}-{j}.safetensors'
                 for i in range(CHUNKS) for j in range(3)]
    for path in required:
        frozen.file(path.relative_to(source).as_posix(), label='single-correction source')
    protocol = load_protocol(load_run_config(repository/BOUNDARY_CONFIG,repo_root=repository).protocol_path)
    prompt = next(p['prompt'] for p in protocol.calibration_prompts if p['prompt_id'] == prompt_id)
    draft_path = repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    target_path = repository/'experiments/insight2-krea14b.artifacts.json'
    require(digest(draft_path)[0] == SPEC_SHA, 'Draft specification drift')
    spec, verified, module = verify_artifact_spec(draft_path,model_root=model_root,repo_root=repository)
    _, target_verified, target_module = verify_target_artifact_spec(target_path,model_root=model_root,repo_root=repository)
    require((module,target_module) == ('cfexp.self_forcing_frame_adapter','cfexp.krea_chunk_backend'),
            'unexpected model backend')
    devices = _visible_devices()
    require(len(devices) == 2 and len(set(devices)) == 2, 'two distinct reserved GPUs required')
    uuids = [_gpu_uuid(device) for device in devices]
    require(all(not _gpu_compute_pids(u) for u in uuids), 'GPU busy')
    target = launch_target_generation_worker(run_dir=output,repo_root=repository,
        target_artifact_spec_path=target_path,target_artifact_spec_sha256=target_artifact_spec_sha256(target_path),
        target_verified_artifacts=target_verified,visible_device=devices[1],chunk_count=CHUNKS)
    backend = None
    previous_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        # The controller becomes the single-GPU Draft process. Target already
        # owns its separately isolated child GPU, with the normal RPC lifecycle.
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
        artifacts = {r['name']:Path(r['resolved_path']) for r in verified}
        source_root, python_site = _materialize_source(run_dir=output,worker_index=0,artifacts=artifacts,
            config=spec['adapter_config'],scratch_suffix='single-target-recovery')
        backend = OfficialSelfForcingBackend(source_root=source_root,python_site=python_site,artifacts=artifacts,
            config=spec['adapter_config'],device_name='cuda:0',chunk_count=CHUNKS)
        from safetensors.torch import load_file, save_file
        from .incremental_draft import latent_digest
        torch = backend.torch
        noise = load_file(str(src/'initial-noise.safetensors'))['noise'].to(backend.device)
        noise_hash = latent_digest(noise,torch)
        def ownership():
            require(_gpu_compute_pids(uuids[0]) <= {os.getpid()}, 'unrelated Draft GPU workload')
            require(_gpu_compute_pids(uuids[1]) <= {target.process.pid}, 'unrelated Target GPU workload')
        ownership()
        write_json(output/'preflight.json',{'schema':'single_target_recovery_preflight_v1',
            'prompt_id':prompt_id,'prompt':prompt,'seed':SEED,'arms':schedules,
            'source_sha':SOURCE_SHA,'source_index':SOURCE_INDEX,'devices':list(devices),
            'gpu_uuids':uuids,'draft_artifacts':verified,'target_artifacts':target_verified,
            'initial_noise_sha256':noise_hash,'formal_quality_result':False,
            'acceptance_gate':'disabled; fixed forced Target schedule for each intervention arm',
            'root_policy':'pure native Draft for every arm',
            'collapse_judgment':'human_review_pending; fixed one-second samples; sustained severe color or structural failure for at least three consecutive samples',
            'same_noise_policy':'sealed native initial tensor and all future Draft scheduler noises'})
        images, arms = {}, []
        for arm, correction in schedules.items():
            schedule = intervention_schedule(correction)
            directory = output/'arms'/arm
            def target_generate(index, arm_dir, initial, path):
                relative = path.relative_to(output).as_posix()
                input_path = output/'inputs'/f'{arm}-{index:03d}-noise.safetensors'
                input_path.parent.mkdir(exist_ok=True)
                save_file({'noise':initial.detach().cpu().contiguous()},str(input_path))
                inputs = [(arm_dir/f'{i:03d}.safetensors').relative_to(output).as_posix() for i in range(index)]
                identity = hashlib.sha256(f'single-target:{prompt_id}:{arm}:{index}'.encode()).hexdigest()
                payload = {'prompt':prompt,'chunk_index':index,'internal_seed':SEED,'capture_hidden':False,
                    'history_latent_paths':inputs,'initial_noise_path':input_path.relative_to(output).as_posix(),
                    'output_paths':{'latent':relative},'target_logical_call_id':identity,
                    'target_call_scope_id':identity,'target_call_phase':'single_point_candidate'}
                response = target._request('generate', payload)
                require(response.get('chunk_index') == index and response.get('history_chunk_count') == index,
                        'Target consumed incorrect history')
                result = response['result']
                require(result.get('explicit_initial_noise_verified') is True,'Target did not verify initial noise')
                require(result['latent']['path'] == relative and digest(path)[0] == result['latent']['sha256'],
                        'Target latent binding mismatch')
                require(len(result['denoising_forward_trace']) == 4, 'Target four-step schedule mismatch')
                record={'request':payload,'response':response,'forced_accept':True}
                write_json(output/'target-evidence'/f'{arm}-{index:03d}.json',record)
                return {'chunk':index,'logical_call_id':identity,'latent':result['latent']}
            history, trace = run_arm(backend,prompt=prompt,noise=noise,tape_dir=src/'noise-tape',
                directory=directory,source_dir=src/'native_draft',intervention=correction,
                target_generate=target_generate,ownership_check=ownership)
            require(latent_digest(noise,torch) == noise_hash,'shared initial noise mutated')
            ownership()
            if reference_traces:
                require(all(a['latent_sha256'] == b['latent_sha256'] and a['producer'] == b['producer']
                    for a,b in zip(trace['chunks'], reference_traces[arm])),
                    'eRank persistence/timing changed paired generated trajectory')
            frames = backend.decode_latents(history)
            require(tuple(frames.shape) == (1437,3,480,832),'decoded duration or shape drift')
            video = directory/'video.mp4';backend.write_video(frames,len(frames),video,24)
            samples=[];per_second=[]
            for frame in sorted(set(SAMPLE_FRAMES) | set(range(0,1437,24))):
                rgb = frames[frame,:,::4,::4].permute(1,2,0).contiguous().numpy()
                png(directory/f'frame-{frame:04d}.png',208,120,rgb.tobytes())
                values=rgb.astype('float32')/255
                high_chroma=((values.max(axis=-1)-values.min(axis=-1) > .8) & (values.max(axis=-1) > .95)).mean()
                per_second.append({'frame':frame,'video_seconds':frame/24,'high_chroma_fraction':float(high_chroma),
                                   'interpretation':'descriptive only; not an automatic collapse label'})
                if frame in SAMPLE_FRAMES:samples.append(rgb.tobytes())
            images[arm]=samples
            entry={'arm':arm,'target_block_count':len(trace['target_calls']),
                'intervention_chunks':list(schedule),
                'intervention_chunk':schedule[0] if schedule else None,
                'intervention_seconds':[video_time(i) for i in schedule],
                'video':{'path':video.relative_to(output).as_posix(),'sha256':digest(video)[0],
                         'frames':1437,'fps':24},'samples':per_second,
                'collapse_onset_seconds':None,'recovery_onset_seconds':None,
                'sustained_improvement_seconds':None,'review_status':'pending_visual_comparison'}
            arms.append(entry);write_json(directory/'result.json',entry)
            del history,frames;gc.collect();torch.cuda.empty_cache()
        contact_sheet(output/'comparison.png',[images[a] for a in schedules])
        # Dense fixed one-second pages allow an onset and relapse interval to
        # be reviewed without cherry-picking frames after seeing the result.
        for page in range(6):
            rows=[]
            for arm in schedules:
                from PIL import Image
                rows.append([Image.open(output/'arms'/arm/f'frame-{second*24:04d}.png').convert('RGB').tobytes()
                             for second in range(page*10,(page+1)*10)])
            contact_sheet(output/f'comparison-seconds-{page*10:02d}-{page*10+9:02d}.png',rows)
        result={'schema':'single_target_recovery_result_v1','status':'generation_and_decode_complete',
            'prompt_id':prompt_id,'arms':arms,'comparison_row_order':list(schedules),
            'comparison_frame_indices':list(SAMPLE_FRAMES),'source_index':SOURCE_INDEX,
            'formal_quality_result':False,'conclusion':'pending blinded/synchronized visual review; no automatic winner'}
        write_json(output/'result.json',result)
        return result
    finally:
        if backend is not None:backend.close()
        target.close()
        if previous_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=previous_cuda


def main(argv=None):
    parser=argparse.ArgumentParser()
    study = parser.add_mutually_exclusive_group()
    study.add_argument('--prevention-recovery',action='store_true')
    study.add_argument('--early-prevention',action='store_true')
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--prompt-id',choices=PROMPTS,required=True)
    parser.add_argument('--reference-run',type=Path)
    parser.add_argument('--reference-index')
    args=parser.parse_args(argv)
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root
    try:run(args.source_run,_run_dir(),find_git_root(Path.cwd()),_model_root(),args.prompt_id,prevention_recovery=args.prevention_recovery,early_prevention=args.early_prevention,reference_run=args.reference_run,reference_index=args.reference_index)
    finally:shutdown_owned_runtime_helpers(label='single-target-recovery')

if __name__ == '__main__':main()
