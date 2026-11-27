"""Paired long-horizon native rebuild versus retained KV diagnostic.

One shared history follows the reuse candidate. Stop at the first differing
forward input/output; never compare independently evolving trajectories.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from .motion_vae_audit import PROMPTS, SPEC_SHA, restore_history
from .motion_video_audit import SOURCE_SHA, SOURCE_INDEX, digest, require
from .main_table_merge import _load_frozen_index
from .runtime_helpers import shutdown_owned_runtime_helpers


def compare(a, b, torch):
    require(a.shape == b.shape and a.dtype == b.dtype, 'tensor layout mismatch')
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'nonfinite tensor')
    delta = a.float() - b.float()
    differing = (a != b).nonzero()
    return {'exact': bool(torch.equal(a, b)), 'max_abs': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / b.float().norm().clamp_min(1e-30)),
            'different_elements': len(differing),
            'first_coordinate': differing[0].tolist() if len(differing) else None}


def paired_call(backend, *, prompt, history, noise, seed, index):
    """Separate buffer ownership, identical explicit inputs and backend RNG reset."""
    torch, pipeline = backend.torch, backend.pipeline
    executor = backend._incremental_draft
    require(executor is not None, 'reuse executor required')
    from .incremental_draft import latent_digest
    before = (latent_digest(history, torch), latent_digest(noise, torch))
    outputs, traces = {}, {}
    from .branch_kv_state import payloads_alias
    try:
        for mode in ('reuse', 'rebuild'):
            pipeline.kv_cache1 = pipeline.crossattn_cache = None
            backend._incremental_draft = executor if mode == 'reuse' else None
            steps = []
            def before_forward(_module, _args, kwargs):
                if mode == 'rebuild':
                    for state in executor.cache._states.values():
                        require(not payloads_alias(state.payload,
                            {'self': kwargs['kv_cache'], 'cross': kwargs['crossattn_cache']}),
                            'rebuild aliases retained cache')
                timestep = kwargs['timestep']
                if bool((timestep > 0).any()):
                    steps.append({'timestep': float(timestep.reshape(-1)[0]),
                        'position': int(kwargs['current_start']),
                        'input': kwargs['noisy_image_or_video'].detach().cpu().clone()})
            def capture(_module, _args, kwargs, result):
                if bool((kwargs['timestep'] > 0).any()):
                    steps[-1]['clean'] = result[1].detach().cpu().clone()
            pre_handle = pipeline.generator.register_forward_pre_hook(before_forward, with_kwargs=True)
            handle = pipeline.generator.register_forward_hook(capture, with_kwargs=True)
            try:
                outputs[mode] = backend.generate_chunk(prompt=prompt,
                    history_latents=history.clone(), initial_noise=noise.clone(),
                    internal_seed=seed, chunk_index=index, capture_hidden=False,
                    history_kv_widths=(1,)+(3,)*(index-1))
            finally:
                handle.remove()
                pre_handle.remove()
                pipeline.kv_cache1 = pipeline.crossattn_cache = None
            require(len(steps) == 4, 'incomplete denoise trace')
            traces[mode] = steps
        require(before == (latent_digest(history, torch), latent_digest(noise, torch)), 'input mutation')
        require(outputs['reuse']['kv_reuse']['reused_chunks'] == (0 if index == 1 else index),
                'expected retained history was replayed instead')
        return outputs, traces, before
    finally:
        pipeline.kv_cache1 = pipeline.crossattn_cache = None
        backend._incremental_draft = executor


def run(source, output, repository, model_root):
    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend, _materialize_source
    from .correction_runtime import DeterministicNoiseProvider
    from .insight2_run import load_run_config
    from .insight2_protocol import load_protocol
    from .main_table_ours_campaign import BOUNDARY_CONFIG
    from .main_table_baseline_campaign import _gpu_uuid, _gpu_compute_pids
    source, output = source.resolve(strict=True), output.resolve(strict=True)
    require(not source.is_relative_to(output) and not output.is_relative_to(source), 'overlap')
    require(not any((p/n).exists() for p in (output,*output.parents)
        for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')), 'sealed output')
    frozen, marker = _load_frozen_index(source)
    require(marker['git_sha'] == SOURCE_SHA and marker['artifact_index_sha256'] == SOURCE_INDEX,
            'source seal drift')
    attempt = source/'attempts'/marker['attempt_id']
    result = frozen.file(str((attempt/'result.json').relative_to(source)), label='source result')
    rows = json.loads(result.read_text())['rows']
    selected = [r for r in rows if r['arm']=='draft_only' and r['prompt_id'] in PROMPTS]
    require(len(selected)==2 and {r['prompt_id'] for r in selected}==set(PROMPTS), 'source population')
    paths = {}
    for row in selected:
        pid=row['prompt_id']; histories=[]
        for p in (attempt/'cells/draft_only/state/rollouts').glob('*/chunk-119.json'):
            p=frozen.file(str(p.relative_to(source)),label='checkpoint')
            h=json.loads(p.read_text())['history']
            if h['prompt_id']==pid: histories.append(h)
        require(len(histories)==1, 'ambiguous source history')
        h=histories[0]
        require(h['head_sha256']==row['trajectory']['final_history_head_sha256'], 'history drift')
        paths[pid]=restore_history(attempt/'cells/draft_only',h,row['trajectory']['committed'],output/'source-latents'/pid)
    config=load_run_config(repository/BOUNDARY_CONFIG,repo_root=repository)
    protocol=load_protocol(config.protocol_path)
    prompts={p['prompt_id']:p['prompt'] for p in protocol.calibration_prompts}
    spec_path=repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    require(digest(spec_path)[0]==SPEC_SHA, 'artifact spec drift')
    spec, verified, module=verify_artifact_spec(spec_path,model_root=model_root,repo_root=repository)
    require(module=='cfexp.self_forcing_frame_adapter','adapter drift')
    artifacts={r['name']:Path(r['resolved_path']) for r in verified}
    source_root,python_site=_materialize_source(run_dir=output,worker_index=0,
        artifacts=artifacts,config=spec['adapter_config'],scratch_suffix='kv-parity')
    device=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(bool(device) and ',' not in device,'one GPU required')
    uuid=_gpu_uuid(device);require(not _gpu_compute_pids(uuid),'GPU busy')
    backend=OfficialSelfForcingBackend(source_root=source_root,python_site=python_site,
        artifacts=artifacts,config=spec['adapter_config'],device_name='cuda:0',chunk_count=120)
    try:
        torch=backend.torch
        from safetensors.torch import load_file, save_file
        from .incremental_draft import latent_digest
        provider=DeterministicNoiseProvider(run_dir=output)
        reports=[]
        for pid in PROMPTS:
            backend._incremental_draft.cache.clear()
            history=load_file(str(paths[pid][0]))['latent']
            entries=[]; first=None
            directory=output/'comparisons'/pid;directory.mkdir(parents=True,exist_ok=False)
            for index in range(1,120):
                require(_gpu_compute_pids(uuid)<={os.getpid()},'unrelated GPU workload appeared')
                material=provider.materialize(prompt_id=pid,trajectory_seed=20260911,chunk_index=index)
                noise=load_file(str(output/material.tensor_path))['noise']
                values,traces,inputs=paired_call(backend,prompt=prompts[pid],history=history,
                    noise=noise,seed=material.descriptor.derived_seed,index=index)
                steps=[]
                for n,(a,b) in enumerate(zip(traces['reuse'],traces['rebuild'],strict=True)):
                    require((a['timestep'],a['position'])==(b['timestep'],b['position']),'schedule drift')
                    for field in ('input','clean'):
                        comparison=compare(a[field],b[field],torch)
                        steps.append({'step':n,'timestep':a['timestep'],'position':a['position'],
                                      'field':field,**comparison})
                        if first is None and not comparison['exact']:
                            first={'chunk_index':index,'step':n,'field':field,**comparison}
                row={'chunk_index':index,'history_sha256':inputs[0],'noise_tensor_sha256':inputs[1],
                    'noise':material.descriptor.to_dict(),'steps':steps,
                    'latent':compare(values['reuse']['latent'],values['rebuild']['latent'],torch),
                    'reuse':values['reuse']['kv_reuse'],
                    'kv_witness_equal':values['reuse']['kv_witness_sha256']==values['rebuild']['kv_witness_sha256'],
                    'witnesses':{m:values[m]['kv_witness'] for m in values},
                    'forward_traces':{m:values[m]['forward_trace'] for m in values},
                    'source_latent_equal':torch.equal(values['reuse']['latent'],load_file(str(paths[pid][index]))['latent'])}
                if first is None and not row['latent']['exact']:
                    first={'chunk_index':index,'step':None,'field':'final_latent',**row['latent']}
                save_file({m:values[m]['latent'] for m in values},str(directory/f'{index:03d}.safetensors'))
                with (directory/f'{index:03d}.json').open('x') as f: json.dump(row,f,indent=2)
                entries.append(row)
                print('KV_PAIRED',pid,index,'exact',row['latent']['exact'],'first',first,flush=True)
                if first is not None:
                    save_file({f'{m}_{n}_{field}':s[field] for m,t in traces.items()
                        for n,s in enumerate(t) for field in ('input','clean')},str(directory/'first-divergence.safetensors'))
                    save_file({'history':history,'noise':noise},str(directory/'first-divergence-inputs.safetensors'))
                    break
                history=torch.cat((history,values['reuse']['latent']),dim=1)
            reports.append({'prompt_id':pid,'checked_chunks':len(entries),'first_divergence':first,
                            'completed_60s':len(entries)==119,'rows':entries})
        report={'schema':'kv_generation_parity_v1','source_marker':marker,'gpu_uuid':uuid,
            'shared_history_policy':'canonical Target root then shared reuse outputs; stop at first step divergence',
            'artifact_spec_sha256':SPEC_SHA,'rows':reports}
        with (output/'result.json').open('x') as f: json.dump(report,f,indent=2)
        return report
    finally:
        backend.close()


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True)
    args=p.parse_args(argv)
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root
    try: run(args.source_run,_run_dir(),find_git_root(Path.cwd()),_model_root())
    finally: shutdown_owned_runtime_helpers(label='kv-generation-parity')

if __name__=='__main__': main()
