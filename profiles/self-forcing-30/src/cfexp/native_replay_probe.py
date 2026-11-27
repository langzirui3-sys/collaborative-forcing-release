"""Cold-history reconstruction against sealed native continuous trajectories."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
from .motion_video_audit import require, digest
from .motion_vae_audit import PROMPTS, SPEC_SHA
from .model_compatibility import NoiseTape, instrument, first_trace_difference
from .runtime_helpers import shutdown_owned_runtime_helpers

SOURCE_SHA = '0557ad016ca7109af8423877a4fd9b33b860264b'
SOURCE_INDEX = 'dee12a85aa647c9981182067e61fa2791cb83c9afe5f3b1999fadf2b7c404301'
CHECKPOINTS = (1, 7, 30, 59)
POLICIES = ('legacy_prefill', 'native_committed')

@contextmanager
def replay_geometry(pipeline, *, history_chunks, external_chunks, policy):
    require(type(history_chunks) is int and history_chunks > 0, 'history chunks')
    require(type(external_chunks) is int and 0 <= external_chunks <= history_chunks, 'external chunks')
    require(policy in POLICIES, 'replay policy')
    def pre(_module, args, kw):
        index = int(kw['current_start']) // (3*pipeline.frame_seq_length)
        if float(kw['timestep'].reshape(-1)[0]) == 0 and index < history_chunks:
            width = 3 if policy == 'native_committed' and index >= external_chunks else 1
            require(kw['timestep'].shape[1] == 1, 'unexpected original prefix geometry')
            kw['timestep'] = kw['timestep'].expand(kw['timestep'].shape[0], width).contiguous()
        return args, kw
    handle = pipeline.generator.register_forward_pre_hook(pre, with_kwargs=True, prepend=True)
    try: yield
    finally: handle.remove()

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
        config=spec['adapter_config'],scratch_suffix='native-replay')
    device = os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(bool(device) and ',' not in device, 'one GPU required')
    uuid = _gpu_uuid(device); require(not _gpu_compute_pids(uuid), 'GPU busy')
    backend = OfficialSelfForcingBackend(source_root=source_root,python_site=python_site,artifacts=artifacts,
        config=spec['adapter_config'],device_name='cuda:0',chunk_count=120)
    reports = []
    try:
        from safetensors.torch import load_file
        from .kv_generation_parity import compare
        from .incremental_draft import latent_digest
        torch = backend.torch
        for pid in PROMPTS:
            src = attempt/pid
            noise = load_file(str(src/'initial-noise.safetensors'))['noise']
            for arm in ('native_draft', 'native_target'):
                source_trace = json.loads((src/arm/'trace.json').read_text())
                external = int(arm == 'native_target')
                for index in CHECKPOINTS:
                    history = torch.cat([load_file(str(src/arm/f'{i:03d}.safetensors'))['latent']
                        for i in range(index)], dim=1)
                    expected = load_file(str(src/arm/f'{index:03d}.safetensors'))['latent']
                    for policy in POLICIES:
                        require(_gpu_compute_pids(uuid) <= {os.getpid()}, 'unrelated GPU workload')
                        backend._incremental_draft.cache.clear()
                        directory = output/pid/arm/f'{index:03d}'/policy
                        tape = NoiseTape(src/'noise-tape', torch, False)
                        before = latent_digest(history, torch)
                        widths=tuple(3 if policy == 'native_committed' and j >= external else 1
                                     for j in range(index))
                        with instrument(backend,directory,tape) as trace:
                            result = backend.generate_chunk(prompt=prompts[pid],history_latents=history,
                                initial_noise=noise[:,index*3:(index+1)*3],internal_seed=20260911,
                                chunk_index=index,capture_hidden=False,history_kv_widths=widths)
                        tape.finish(index,index+1)
                        require(before == latent_digest(history,torch), 'history mutated')
                        require(result['kv_reuse']['reused_chunks']==0 and
                            result['kv_reuse']['replayed_chunks']==index, 'not a cold replay')
                        require(tape.calls==[x for x in source_trace['noise'] if x['chunk']==index], 'noise drift')
                        reference=[x for x in source_trace['forwards'] if x['chunk']==index and x['timestep']>0]
                        row={'prompt_id':pid,'source_arm':arm,'chunk':index,'policy':policy,
                            'external_chunks':external,'history_sha256':before,
                            'comparison':compare(expected,result['latent'].detach().cpu(),torch),
                            'first_difference':first_trace_difference(reference,trace),
                            'trace':trace,'noise':tape.calls}
                        with (directory/'result.json').open('x') as f:json.dump(row,f,indent=2)
                        reports.append(row)
                        print('REPLAY_RESULT',pid,arm,index,policy,row['comparison'],flush=True)
                        backend._incremental_draft.cache.clear()
                        gc.collect();torch.cuda.empty_cache()
        report={'schema':'native_replay_probe_v1','source_marker':marker,'rows':reports,
            'checkpoints':CHECKPOINTS,'policies':POLICIES,'formal_quality_result':False}
        with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
        return report
    finally:backend.close()

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True)
    args=parser.parse_args(argv)
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root
    try:run(args.source_run,_run_dir(),find_git_root(Path.cwd()),_model_root())
    finally:shutdown_owned_runtime_helpers(label='native-replay-probe')

if __name__ == '__main__':main()
