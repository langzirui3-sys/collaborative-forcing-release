"""Forced branch-choice correctness with real candidates from a sealed run.

The fixture chooses both branches regardless of the score gate. It measures
cache-state equivalence, never acceptance quality or speed versus Target.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
from .motion_video_audit import require, digest
from .motion_vae_audit import SPEC_SHA
from .runtime_helpers import shutdown_owned_runtime_helpers

SOURCE_SHA = '8bc7ace01d952001e5b2f51e4235f52d30b259ba'
SOURCE_RUN_ID = 'branch-gate-systems-60s-ntu-v1-20261025T062854Z-8bc7ace0-a8a73f9b-6ddaffa6'
CENTERS = (13, 69)
CHOICES = ('original', 'corrected')


def exercise_choice(*, torch, call, history, widths, target, center, choice, expected):
    """Keep both branch outputs, then continue the explicitly selected history."""
    require(choice in CHOICES, 'unknown choice')
    anchor = center-4
    require(history.shape[1] == anchor*3 and len(widths) == anchor, 'anchor history')
    rows = []
    for index in range(anchor, center+1):
        value = call(history, widths, index, f'prefix-{index}',
                     action='preserve' if index == anchor else 'none')
        require(torch.equal(value['latent'], expected(index)), 'original prefix drift')
        history = torch.cat((history, value['latent']), dim=1)
        widths += (3,)
    original = call(history, widths, center+1, 'original-successor')
    corrected_history = history.clone()
    corrected_history[:, -3:] = target
    corrected_widths = widths[:-1]+(1,)
    corrected = call(corrected_history, corrected_widths, center+1,
                     'corrected-successor', action='consume')
    evidence = corrected['kv_reuse']
    require(evidence['branch_anchor']['anchor_consumed'] is True and
            evidence['reused_chunks'] == anchor, 'anchor was not consumed at the shared prefix')
    for label, h, w, value in (('original', history, widths, original),
                              ('corrected', corrected_history, corrected_widths, corrected)):
        rebuilt = call(h, w, center+1, label+'-cold', cold=True)
        require(torch.equal(value['latent'], rebuilt['latent']), 'successor cold parity')
        rows.append({'stage': label+'-successor', 'exact': True,
                     'warm': value['kv_reuse'], 'cold': rebuilt['kv_reuse']})
    h, w, value = ((history, widths, original) if choice == 'original' else
                   (corrected_history, corrected_widths, corrected))
    h = torch.cat((h, value['latent']), dim=1); w += (3,)
    for index in (center+2, center+3):
        warm = call(h, w, index, f'continue-{index}')
        cold = call(h, w, index, f'continue-{index}-cold', cold=True)
        require(torch.equal(warm['latent'], cold['latent']), 'chosen continuation cold parity')
        require(warm['kv_reuse']['reused_chunks'] == index, 'selected branch did not retain its state')
        rows.append({'stage': f'continue-{index}', 'exact': True,
                     'warm': warm['kv_reuse'], 'cold': cold['kv_reuse']})
        h = torch.cat((h, warm['latent']), dim=1); w += (3,)
    return rows


def run(source, output, repository, model_root):
    from .main_table_merge import _load_frozen_index
    from .main_table_baseline_campaign import _gpu_uuid, _gpu_compute_pids, _selected_prompts
    from .main_table_v1 import load_bundle
    from .erank_frame_campaign import verify_artifact_spec
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend, _materialize_source
    from .incremental_draft import IncrementalDraftExecutor, latent_digest
    from .correction_runtime import DeterministicNoiseProvider
    from safetensors.torch import load_file, save_file
    source, output = source.resolve(strict=True), output.resolve(strict=True)
    require(source.name == SOURCE_RUN_ID, 'wrong source run')
    require(not source.is_relative_to(output) and not output.is_relative_to(source), 'overlap')
    require(not any((p/n).exists() for p in (output,*output.parents)
        for n in ('artifact-index.json','_SUCCESS.json','_FAILED.json')), 'sealed output')
    frozen, marker = _load_frozen_index(source)
    require(marker['git_sha'] == SOURCE_SHA, 'source revision')
    base = source/'attempts'/marker['attempt_id']/'cells/repeat-0/baseline'
    def read_json(path):
        bound=frozen.file(str(path.relative_to(source)),label='branch fixture metadata')
        return json.loads(bound.read_text())
    result = read_json(base/'result.json')
    require(all(i in result['trajectory']['proposed_chunks'] for i in CENTERS), 'candidate coverage')
    checkpoints = list(base.glob('state/rollouts/*/chunk-119.json'))
    require(len(checkpoints) == 1, 'ambiguous source trajectory')
    state = checkpoints[0].parent
    segments = read_json(checkpoints[0])['history']['segments']
    require(len(segments)==120, 'source horizon')
    prompt = _selected_prompts(repository, load_bundle(repository).protocol)[0]
    preflight=read_json(source/'attempts'/marker['attempt_id']/'provenance/preflight.json')
    require(preflight['prompt_id']==prompt['id'] and preflight['seed']==20260911, 'prompt/seed drift')
    def tensor(segment):
        path = (base/segment['latent_path']).resolve(strict=True)
        require(path.is_relative_to(base) and digest(path)[0] == segment['latent_sha256'], 'source latent binding')
        frozen.file(str(path.relative_to(source)), label='branch fixture latent')
        return load_file(str(path))['latent']
    spec_path = repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    require(digest(spec_path)[0] == SPEC_SHA, 'spec drift')
    spec, verified, module = verify_artifact_spec(spec_path,model_root=model_root,repo_root=repository)
    require(module == 'cfexp.self_forcing_frame_adapter', 'adapter drift')
    artifacts = {r['name']:Path(r['resolved_path']) for r in verified}
    source_root, python_site = _materialize_source(run_dir=output,worker_index=0,artifacts=artifacts,
        config=spec['adapter_config'],scratch_suffix='branch-choice')
    device = os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(bool(device) and ',' not in device, 'one GPU required')
    uuid = _gpu_uuid(device); require(not _gpu_compute_pids(uuid), 'GPU busy')
    backend = OfficialSelfForcingBackend(source_root=source_root,python_site=python_site,artifacts=artifacts,
        config=spec['adapter_config'],device_name='cuda:0',chunk_count=120)
    reports=[]
    try:
        torch, pipeline = backend.torch, backend.pipeline
        executor = backend._incremental_draft
        require(executor is not None and executor.fork_enabled, 'fork-enabled executor required')
        provider=DeterministicNoiseProvider(run_dir=output)
        for center in CENTERS:
            checkpoint=read_json(state/f'chunk-{center:02d}.json')
            candidate=checkpoint['metadata']['target_candidate']['segment']
            require(candidate['producer_role']=='target' and candidate['chunk_index']==center, 'real Target candidate required')
            target=tensor(candidate)
            for choice in CHOICES:
                executor.cache.clear();pipeline.kv_cache1=pipeline.crossattn_cache=None
                directory=output/f'center-{center}'/choice;directory.mkdir(parents=True,exist_ok=False)
                def call(history,widths,index,label,action='none',cold=False):
                    require(_gpu_compute_pids(uuid)<={os.getpid()}, 'unrelated GPU workload')
                    material=provider.materialize(prompt_id=prompt['id'],trajectory_seed=20260911,chunk_index=index)
                    noise=load_file(str(output/material.tensor_path))['noise']
                    before=latent_digest(history,torch)
                    selected=IncrementalDraftExecutor(pipeline=pipeline,torch=torch,scope_sha256=executor.scope_sha256,
                        max_bytes=executor.cache.max_bytes,fork_enabled=True) if cold else executor
                    backend._incremental_draft=selected
                    pipeline.kv_cache1=pipeline.crossattn_cache=None
                    try:
                        value=backend.generate_chunk(prompt=prompt['text'],history_latents=history,
                            initial_noise=noise,internal_seed=material.descriptor.derived_seed,
                            chunk_index=index,capture_hidden=False,kv_branch_action=action,history_kv_widths=widths)
                        require(before==latent_digest(history,torch), 'history mutation')
                        if cold:require(value['kv_reuse']['reused_chunks']==0, 'cold path reused cache')
                        save_file({'latent':value['latent'].detach().cpu().contiguous()},str(directory/f'{label}.safetensors'))
                        evidence={'chunk':index,'history_sha256':before,'widths':widths,'noise':material.descriptor.to_dict(),
                            'cold':cold,'action':action,'reuse':value['kv_reuse'],'forward_trace':value['forward_trace']}
                        with (directory/f'{label}.json').open('x') as f:json.dump(evidence,f,indent=2)
                        return value
                    finally:
                        if cold:selected.cache.clear()
                        backend._incremental_draft=executor
                        pipeline.kv_cache1=pipeline.crossattn_cache=None
                anchor=center-4
                history=torch.cat([tensor(s) for s in segments[:anchor]],dim=1)
                widths=tuple(3 if s['producer_role']=='draft' else 1 for s in segments[:anchor])
                rows=exercise_choice(torch=torch,call=call,history=history,widths=widths,target=target,
                    center=center,choice=choice,expected=lambda i:tensor(segments[i]))
                report={'center':center,'forced_choice':choice,'target_candidate':candidate,'rows':rows}
                with (directory/'result.json').open('x') as f:json.dump(report,f,indent=2)
                reports.append(report);print('BRANCH_CHOICE_VERIFIED',center,choice,flush=True)
                executor.cache.clear();gc.collect();torch.cuda.empty_cache()
        report={'schema':'branch_choice_probe_v1','source_marker':marker,'cases':reports,
                'forced_choices':True,'policy_acceptance_measured':False,'quality_measured':False}
        with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
        return report
    finally:backend.close()


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True)
    args=parser.parse_args(argv)
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root
    try:run(args.source_run,_run_dir(),find_git_root(Path.cwd()),_model_root())
    finally:shutdown_owned_runtime_helpers(label='branch-choice-probe')

if __name__=='__main__':main()
