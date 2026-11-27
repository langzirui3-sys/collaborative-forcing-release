"""Diagnostic gate for T1000 -> native LongLive warped suffix, not quality."""
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
from .prefix_client import target_prefix,resume_prefix
from .runtime_helpers import shutdown_owned_runtime_helpers


def run(output,repository,model_root,devices, *, draft_model_id='longlive_1.3b'):
    from .erank_frame_campaign import verify_artifact_spec
    from .correction_artifacts import verify_target_artifact_spec
    from .main_table_baseline_campaign import _sha256_file
    from .main_table_ours_campaign import _assert_idle_pair,_assert_worker_pair_owns_gpus
    if draft_model_id not in ('self_forcing_1.3b', 'longlive_1.3b'):
        raise ValueError('unsupported prefix gate model')
    longlive = draft_model_id == 'longlive_1.3b'
    if longlive:
        if os.environ.get('CFEXP_LONGLIVE_PREFIX_DIAGNOSTIC') != '1':
            raise ValueError('LongLive prefix gate requires its explicit diagnostic flag')
        draft_spec=repository/'experiments/erank3-longlive-policy-v1.artifacts.json'
    else:
        if os.environ.get('CFEXP_DRAFT_KV_REUSE')!='1':raise ValueError('native Draft KV reuse required')
        draft_spec=repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    target_spec=repository/'experiments/insight2-krea14b.artifacts.json'
    if longlive:
        from .main_table_baseline_campaign import verify_artifact_spec as verify_native
        _,dv,_=verify_native(draft_spec,method_id='draft_only',model_root=model_root,repository_root=repository)
    else:
        _,dv,_=verify_artifact_spec(draft_spec,model_root=model_root,repo_root=repository)
    _,tv,_=verify_target_artifact_spec(target_spec,model_root=model_root,repo_root=repository)
    live=output/'live';reference=output/'reference';live.mkdir();reference.mkdir()
    prompt=PromptCase('target-prefix-gate',PROMPT,'evaluation')
    history=LatentHistory('target-prefix-gate',prompt.prompt_id,20260911)
    session=hashlib.sha256(b'target-prefix-live-gate-v1').hexdigest()
    report={'schema':'target_prefix_rpc_gate_v2','passed':False,'formal_quality_result':False,
        'no_end_to_end_speed_claim':True,'prompt':PROMPT,'seed':20260911,
        'draft_model_id':draft_model_id,
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
            devices=devices,chunk_count=16)
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
                    noises=[runtime.noise_provider.materialize(prompt_id=prompt.prompt_id+f'-suffix-noise-{step}',
                        trajectory_seed=20260911,chunk_index=i).tensor_path for step in (1,2,3)]
                    accepted=resume_prefix(runtime.draft,outcome,prefix,renoise_paths=noises,verify_parity=True)
                    row['target_oracle']=prefix['result']['diagnostic_oracle']
                    row['draft_oracle']=accepted.diagnostics['diagnostic_oracle']
                    row['native_execution']={k:accepted.diagnostics[k] for k in
                        ('draft_native_schedule','draft_denoising_schedule','draft_clean_forwards','history_replay_chunks',
                         'branch_snapshot_bytes','committed_kv_width')}
                    evidence=outcome.evidence['kv_reuse']
                else:
                    if isinstance(outcome,StoppedDraft):raise ValueError('unexpected extra Target intervention')
                    accepted=outcome;ranks.append(outcome.effective_rank);following.append((req,accepted))
                    evidence=accepted.diagnostics['kv_reuse']
                if evidence['replayed_chunks']!=(1 if i==1 else 0):raise ValueError('optimized Draft replayed accepted history')
                row['reuse']=evidence;report['rows'].append(row)
                print('TARGET_PREFIX_GATE',json.dumps(row,sort_keys=True),flush=True)
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
            with (output/'target-prefix-gate.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--draft-model-id',default='longlive_1.3b',
                        choices=('self_forcing_1.3b','longlive_1.3b'))
    args=parser.parse_args()
    from .main_table_baseline_campaign import _run_dir,_model_root,_visible_devices
    from .metadata import find_git_root
    run(_run_dir(),find_git_root(Path.cwd()),_model_root(),_visible_devices(),draft_model_id=args.draft_model_id)


if __name__=='__main__':main()
