"""Native CUDA correctness gate for late Target predictions; no quality claim."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from dataclasses import replace

from .correction_runtime import launch_correction_runtime,launch_draft_decoder_worker
from .correction_state import LatentHistory
from .early_draft_diagnostic import PROMPT
from .insight2_campaign import PromptCase,ChunkGenerationRequest
from .longlive_late_handoff import ENV,require,POLICIES
from .longlive_late_worker import process_factory
from .longlive_late_client import propose,target,commit,generated,continue_draft
from .longlive_low_noise_gate import SOURCE_PATHS as PREVIOUS_SOURCE_PATHS
from .runtime_helpers import shutdown_owned_runtime_helpers

SOURCE_PATHS=tuple(sorted(set(PREVIOUS_SOURCE_PATHS) | {
    'src/cfexp/longlive_late_handoff.py','src/cfexp/longlive_late_worker.py',
    'src/cfexp/longlive_late_client.py','src/cfexp/longlive_late_oracle.py',
    'src/cfexp/longlive_late_gate.py'}))


def run(output,repository,model_root,devices):
    from .main_table_baseline_campaign import verify_artifact_spec,_sha256_file
    from .correction_artifacts import verify_target_artifact_spec
    from .main_table_ours_campaign import _assert_idle_pair,_assert_worker_pair_owns_gpus
    require(os.environ.get(ENV)=='1' and os.environ.get('CFEXP_LONGLIVE_PREFIX_DIAGNOSTIC')=='1'
        and os.environ.get('CFEXP_LONGLIVE_HOST_CURSORS')=='0', 'late gate requires explicit native diagnostic flags')
    draft_spec=repository/'experiments/erank3-longlive-policy-v1.artifacts.json'
    _,dv,_=verify_artifact_spec(draft_spec,method_id='draft_only',model_root=model_root,repository_root=repository)
    target_spec=repository/'experiments/insight2-krea14b.artifacts.json'
    _,tv,_=verify_target_artifact_spec(target_spec,model_root=model_root,repo_root=repository)
    live=output/'live';reference=output/'reference';live.mkdir();reference.mkdir()
    prompt=PromptCase('longlive-late-native-gate',PROMPT,'evaluation')
    history=LatentHistory('longlive-late-native-gate',prompt.prompt_id,20260911)
    identity=hashlib.sha256(b'longlive-late-native-gate-v1').hexdigest()
    report=dict(schema='longlive_late_native_gate_v1',passed=False,formal_quality_result=False,
        no_end_to_end_speed_claim=True,changed_sampling_policy=True,
        source_sha256={path:_sha256_file(repository/path)[0] for path in SOURCE_PATHS},
        prompt=PROMPT,seed=20260911,chunk_count=16,intervention_chunks=list(range(1,10)),
        policies={k:dict(draft_updates=v[0],target_timestep=v[1]) for k,v in POLICIES.items()},
        rows=[],following_draft=[])
    runtime=oracle=None;following=[]
    try:
        _assert_idle_pair(*devices)
        runtime=launch_correction_runtime(run_dir=live,repo_root=repository,
            draft_artifact_spec_path=draft_spec,draft_artifact_spec_sha256=_sha256_file(draft_spec)[0],
            draft_model_id='longlive_1.3b',draft_verified_artifacts=dv,
            target_artifact_spec_path=target_spec,target_artifact_spec_sha256=_sha256_file(target_spec)[0],
            target_verified_artifacts=tv,devices=devices,chunk_count=16,process_factory=process_factory)
        for n in range(16):
            _assert_worker_pair_owns_gpus(runtime=runtime,draft_device=devices[0],target_device=devices[1],
                                         phase='late handoff correctness gate')
            noise=runtime.noise_provider.materialize(prompt_id=prompt.prompt_id,trajectory_seed=20260911,chunk_index=n)
            request=ChunkGenerationRequest(prompt=prompt,history=history,noise=noise,
                output_latent_path=f'latents/chunk-{n:02d}.safetensors',capture_hidden=False)
            if n==0:
                item=runtime.target.generate(replace(request,
                    target_logical_call_id=hashlib.sha256(b'late-native-root').hexdigest(),target_call_phase='unscoped'))
                report['root']=dict(latent_sha256=item.segment.latent_sha256,
                    denoising_schedule=item.diagnostics['denoising_schedule'])
            elif n<=9:
                policy=tuple(POLICIES)[(n-1)%len(POLICIES)]
                noises=[runtime.noise_provider.materialize(prompt_id=prompt.prompt_id+f'-late-renoise-{j}',
                    trajectory_seed=20260911,chunk_index=n).tensor_path for j in range(POLICIES[policy][0])]
                proposal=propose(runtime.draft,request,session_id=identity,policy=policy,
                    renoise_paths=noises,output_path=f'handoff/chunk-{n:02d}.safetensors',verify_parity=True)
                prediction=target(runtime.target,request,proposal,verify_parity=True)
                committed=commit(runtime.draft,session_id=identity,proposal=proposal,
                    target_receipt=prediction,verify_parity=True)
                item=generated(runtime.target,request,prediction['result'])
                row=dict(chunk_index=n,policy=policy,proposal=proposal,target=prediction,commit=committed)
                report['rows'].append(row)
                print('LATE_NATIVE_GATE',n,policy,'proposal/target/B3 exact',flush=True)
            else:
                item=continue_draft(runtime.draft,request,session_id=identity)
                following.append((request,item))
            history=history.append(item.segment)
        runtime.close();runtime=None
        for directory in ('latents','noise'):
            shutil.copytree(live/directory,reference/directory)
        oracle=launch_draft_decoder_worker(run_dir=reference,repo_root=repository,
            draft_artifact_spec_path=draft_spec,draft_artifact_spec_sha256=_sha256_file(draft_spec)[0],
            draft_model_id='longlive_1.3b',draft_verified_artifacts=dv,visible_device=devices[0],chunk_count=16)
        from safetensors.torch import load_file
        import torch
        for request,expected in following:
            actual=oracle.generate(replace(request,
                output_latent_path=f'oracle/chunk-{request.history.next_chunk_index:02d}.safetensors'))
            a=load_file(str(reference/actual.segment.latent_path))['latent']
            b=load_file(str(live/expected.segment.latent_path))['latent']
            proof=dict(chunk_index=request.history.next_chunk_index,latent_exact=bool(torch.equal(a,b)),
                rank_exact=actual.effective_rank==expected.effective_rank,
                maximum_absolute_difference=float((a.float()-b.float()).abs().max()))
            report['following_draft'].append(proof)
            require(proof['latent_exact'] and proof['rank_exact'], 'late handoff corrupted subsequent native Draft')
        require(len(report['rows'])==9 and len(report['following_draft'])==6, 'late gate coverage incomplete')
        report['passed']=True
        return report
    finally:
        try:
            try:
                if oracle is not None:oracle.close()
            finally:
                try:
                    if runtime is not None:runtime.close()
                finally:shutdown_owned_runtime_helpers(label='longlive-late-native-gate')
        except BaseException as error:
            report['passed']=False;report['cleanup_error']=str(error)
            raise
        finally:
            with (output/'longlive-late-native-gate.json').open('x') as out:
                json.dump(report,out,indent=2,allow_nan=False)


def main():
    from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices
    from .metadata import find_git_root
    run(_run_dir(),find_git_root(Path.cwd()),_model_root(),_visible_devices())


if __name__=='__main__':
    main()
