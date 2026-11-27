"""Independent GPU parity gate for the experimental T1000 -> D625 bridge."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from .early_protocol import StoppedDraft
from .early_draft_diagnostic import PROMPT
from .correction_runtime import launch_correction_runtime,launch_draft_decoder_worker
from .correction_state import LatentHistory
from .insight2_campaign import ChunkGenerationRequest,PromptCase
from .prefix_client import target_prefix
from .longlive_low_noise_client import resume_low_noise
from .longlive_low_noise_worker import process_factory
from .runtime_helpers import shutdown_owned_runtime_helpers


SOURCE_PATHS = ('src/cfexp/accepted_stream_worker.py', 'src/cfexp/correction_runtime.py', 'src/cfexp/correction_worker.py', 'src/cfexp/early_draft.py', 'src/cfexp/early_protocol.py', 'src/cfexp/early_worker.py', 'src/cfexp/host_kv_cursors.py', 'src/cfexp/kernel_profile.py', 'src/cfexp/krea_chunk_backend.py', 'src/cfexp/kv_witness_batch.py', 'src/cfexp/longlive_chunk_session.py', 'src/cfexp/longlive_low_noise.py', 'src/cfexp/longlive_low_noise_client.py', 'src/cfexp/longlive_low_noise_gate.py', 'src/cfexp/longlive_low_noise_worker.py', 'src/cfexp/longlive_native.py', 'src/cfexp/longlive_policy_backend.py', 'src/cfexp/longlive_policy_worker.py', 'src/cfexp/longlive_prefix_backend.py', 'src/cfexp/longlive_prefix_gate.py', 'src/cfexp/prefix_client.py', 'src/cfexp/prefix_oracle.py', 'src/cfexp/prefix_worker.py', 'src/cfexp/self_forcing_frame_adapter.py', 'src/cfexp/target_prefix.py', 'src/cfexp/target_prefix_gate.py', 'src/cfexp/video_encoding_policy.py', 'src/cfexp/video_only_decode.py')

def run(output,repository,model_root,devices, *, draft_model_id='longlive_1.3b'):
    from .erank_frame_campaign import verify_artifact_spec
    from .correction_artifacts import verify_target_artifact_spec
    from .main_table_baseline_campaign import _sha256_file
    from .main_table_ours_campaign import _assert_idle_pair,_assert_worker_pair_owns_gpus
    if draft_model_id != 'longlive_1.3b' or os.environ.get('CFEXP_LONGLIVE_LOW_NOISE_DIAGNOSTIC') != '1':
        raise ValueError('low-noise gate requires the explicit native LongLive diagnostic')
    draft_spec=repository/'experiments/erank3-longlive-policy-v1.artifacts.json'
    from .main_table_baseline_campaign import verify_artifact_spec as verify_native
    _,dv,_=verify_native(draft_spec,method_id='draft_only',model_root=model_root,repository_root=repository)
    target_spec=repository/'experiments/insight2-krea14b.artifacts.json'
    _,tv,_=verify_target_artifact_spec(target_spec,model_root=model_root,repo_root=repository)
    live=output/'live';reference=output/'reference';live.mkdir();reference.mkdir()
    prompt=PromptCase('longlive-low-noise-gate',PROMPT,'evaluation')
    history=LatentHistory('longlive-low-noise-gate',prompt.prompt_id,20260911)
    session=hashlib.sha256(b'longlive-low-noise-live-gate-v1').hexdigest()
    report={'schema':'longlive_low_noise_rpc_gate_v1','passed':False,'formal_quality_result':False,
        'no_end_to_end_speed_claim':True,'prompt':PROMPT,'seed':20260911,
        'draft_model_id':draft_model_id,
        'source_sha256':{path:_sha256_file(repository/path)[0] for path in SOURCE_PATHS},
        'prefix_chunks':list(range(1,9)),'chunk_count':16,'rows':[],
        'diagnostic_replay':'independent full-prefix oracle only; optimized cache remains incremental'}
    runtime=None;oracle=None;following=[]
    try:
        _assert_idle_pair(*devices)
        runtime=launch_correction_runtime(run_dir=live,repo_root=repository,
            draft_artifact_spec_path=draft_spec,draft_artifact_spec_sha256=_sha256_file(draft_spec)[0],
            draft_model_id=draft_model_id,
            draft_verified_artifacts=dv,target_artifact_spec_path=target_spec,
            target_artifact_spec_sha256=_sha256_file(target_spec)[0],target_verified_artifacts=tv,
            devices=devices,chunk_count=16,process_factory=process_factory)
        ranks=[];remaining=8
        for i in range(16):
            _assert_worker_pair_owns_gpus(runtime=runtime,draft_device=devices[0],target_device=devices[1],
                                         phase='Target-prefix correctness gate')
            noise=runtime.noise_provider.materialize(prompt_id=prompt.prompt_id,trajectory_seed=20260911,chunk_index=i)
            req=ChunkGenerationRequest(prompt=prompt,history=history,noise=noise,
                output_latent_path=f'latents/chunk-{i:02d}.safetensors',capture_hidden=False)
            if not i:
                accepted=runtime.target.generate(replace(req,target_logical_call_id=hashlib.sha256(b'prefix-gate-root').hexdigest(),
                                                        target_call_phase='unscoped'))
            else:
                control={'schema':'early_erank_control_v1','prior_ranks':list(ranks),
                    'calibration_id':'synthetic-correctness-only','calibration_drop_scores':[0.],
                    'remaining_budget':remaining,'remaining_slots':16-i}
                outcome=runtime.draft.generate_early(req,session_id=session,control=control)
                row={'chunk_index':i,'prefix':i<=8}
                if i<=8:
                    if not isinstance(outcome,StoppedDraft):raise ValueError('expected pending Draft prefix')
                    ranks.append(outcome.decision['effective_rank']);remaining-=1
                    prefix=target_prefix(runtime.target,req,output_path=f'prefix/chunk-{i:02d}.safetensors',verify_parity=True)
                    noise_625=runtime.noise_provider.materialize(prompt_id=prompt.prompt_id+'-low-noise-625',
                        trajectory_seed=20260911,chunk_index=i).tensor_path
                    accepted=resume_low_noise(runtime.draft,outcome,prefix,renoise_path=noise_625,verify_parity=True)
                    row['target_oracle']=prefix['result']['diagnostic_oracle']
                    row['draft_oracle']=accepted.diagnostics['diagnostic_oracle']
                    row['native_execution']={k:accepted.diagnostics[k] for k in
                        ('schema','target_steps','draft_native_schedule','draft_denoising_schedule','handoff_timestep',
                         'draft_clean_forwards','history_replay_chunks','branch_snapshot_bytes','committed_kv_width',
                         'forward_trace','tokens_per_chunk','committed_chunks')}
                    evidence=outcome.evidence['kv_reuse']
                else:
                    if isinstance(outcome,StoppedDraft):raise ValueError('unexpected extra Target intervention')
                    accepted=outcome;ranks.append(outcome.effective_rank);following.append((req,accepted))
                    evidence=accepted.diagnostics['kv_reuse']
                if evidence['replayed_chunks']!=(1 if i==1 else 0):raise ValueError('optimized Draft replayed accepted history')
                row['reuse']=evidence;report['rows'].append(row)
                print('LONGLIVE_LOW_NOISE_GATE',json.dumps(row,sort_keys=True),flush=True)
            history=history.append(accepted.segment)
        if remaining:raise ValueError('prefix gate budget incomplete')
        runtime.close();runtime=None
        for directory in ('latents','noise'):shutil.copytree(live/directory,reference/directory)
        oracle=launch_draft_decoder_worker(run_dir=reference,repo_root=repository,
            draft_artifact_spec_path=draft_spec,draft_artifact_spec_sha256=_sha256_file(draft_spec)[0],
            draft_model_id=draft_model_id,
            draft_verified_artifacts=dv,visible_device=devices[0],chunk_count=16)
        from safetensors.torch import load_file
        import torch
        report['following_draft']=[]
        for req,expected in following:
            actual=oracle.generate(replace(req,output_latent_path=f'oracle/chunk-{req.history.next_chunk_index:02d}.safetensors'))
            a=load_file(str(reference/actual.segment.latent_path))['latent']
            b=load_file(str(live/expected.segment.latent_path))['latent']
            row={'chunk_index':req.history.next_chunk_index,'latent_exact':bool(torch.equal(a,b)),
                 'rank_exact':actual.effective_rank==expected.effective_rank,
                 'maximum_absolute_difference':float((a.float()-b.float()).abs().max())}
            report['following_draft'].append(row)
            if not row['latent_exact'] or not row['rank_exact']:raise ValueError('following Draft continuation diverged')
        report['passed']=True
        return report
    finally:
        try:
            try:
                if oracle is not None:oracle.close()
            finally:
                try:
                    if runtime is not None:runtime.close()
                finally:shutdown_owned_runtime_helpers(label='target-prefix-rpc-gate')
        except BaseException as error:
            report['passed']=False;report['cleanup_error']=str(error);raise
        finally:
            with (output/'longlive-low-noise-gate.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--draft-model-id',default='longlive_1.3b',
                        choices=('longlive_1.3b',))
    args=parser.parse_args()
    from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices
    from .metadata import find_git_root
    run(_run_dir(),find_git_root(Path.cwd()),_model_root(),_visible_devices(),draft_model_id=args.draft_model_id)


if __name__=='__main__':main()
