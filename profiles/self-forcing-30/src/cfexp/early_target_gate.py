"""Actual-Target RPC gate; paired history/noise, no quality or speed claim."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from .early_protocol import StoppedDraft, decide
from .early_draft_diagnostic import PROMPT
from .correction_runtime import launch_correction_runtime, launch_draft_decoder_worker
from .correction_state import LatentHistory
from .insight2_campaign import ChunkGenerationRequest, PromptCase
from .runtime_helpers import shutdown_owned_runtime_helpers


def run(output, repository, model_root, devices):
    from .erank_frame_campaign import verify_artifact_spec
    from .correction_artifacts import verify_target_artifact_spec
    from .main_table_baseline_campaign import _sha256_file
    from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
    if os.environ.get('CFEXP_DRAFT_KV_REUSE') != '1':
        raise ValueError('actual Target gate requires native Draft KV reuse')
    draft_spec=repository/'experiments/erank-frame-quality-layer-ablation-v1.artifacts.json'
    target_spec=repository/'experiments/insight2-krea14b.artifacts.json'
    _,draft_verified,_=verify_artifact_spec(draft_spec,model_root=model_root,repo_root=repository)
    _,target_verified,_=verify_target_artifact_spec(target_spec,model_root=model_root,repo_root=repository)
    late_dir=output/'late';early_dir=output/'early'
    late_dir.mkdir();early_dir.mkdir()
    prompt=PromptCase('actual-target-gate',PROMPT,'evaluation')
    history=LatentHistory('actual-target-gate','actual-target-gate',20260911)
    report={'schema':'early_actual_target_rpc_gate_v1','passed':False,'formal_quality_result':False,
            'prompt':PROMPT,'seed':20260911,'chunk_count':16,'rows':[],
            'control':'synthetic all-zero drop calibration forces the first eight candidate replacements',
            'no_end_to_end_speed_claim':True}
    records=[];runtime=None;early=None
    try:
        _assert_idle_pair(*devices)
        runtime=launch_correction_runtime(run_dir=late_dir,repo_root=repository,
            draft_artifact_spec_path=draft_spec,draft_artifact_spec_sha256=_sha256_file(draft_spec)[0],
            draft_verified_artifacts=draft_verified,target_artifact_spec_path=target_spec,
            target_artifact_spec_sha256=_sha256_file(target_spec)[0],target_verified_artifacts=target_verified,
            devices=devices,chunk_count=16)
        ranks=[];remaining=8
        for i in range(16):
            _assert_worker_pair_owns_gpus(runtime=runtime,draft_device=devices[0],target_device=devices[1],
                                         phase='actual Target early gate late reference')
            noise=runtime.noise_provider.materialize(prompt_id=prompt.prompt_id,trajectory_seed=20260911,chunk_index=i)
            req=ChunkGenerationRequest(prompt=prompt,history=history,noise=noise,
                output_latent_path=f'latents/chunk-{i:02d}-draft.safetensors',capture_hidden=False)
            if i:
                draft=runtime.draft.generate(req)
                control={'schema':'early_erank_control_v1','prior_ranks':list(ranks),
                    'calibration_id':'synthetic-correctness-only','calibration_drop_scores':[0.],
                    'remaining_budget':remaining,'remaining_slots':16-i}
                decision=decide(control,draft.effective_rank)
                ranks.append(draft.effective_rank)
                selected=decision['replace_with_target'];remaining-=int(selected)
            else:
                draft=None;control=None;selected=True
            if selected:
                accepted=runtime.target.generate(replace(req,
                    output_latent_path=f'latents/chunk-{i:02d}-target.safetensors',
                    target_logical_call_id=hashlib.sha256(f'actual-target-gate-{i}'.encode()).hexdigest(),
                    target_call_phase='unscoped'))
            else:
                accepted=draft
            if i:records.append((req,control,draft,accepted,selected))
            history=history.append(accepted.segment)
        if remaining or [r[0].history.next_chunk_index for r in records if r[-1]] != list(range(1,9)):
            raise ValueError('diagnostic replacement schedule drifted')
        runtime.close();runtime=None
        # Separate immutable phase roots prevent scratch/cache collisions. The
        # reference history retains its original relative file names and bytes.
        for subdir in ('latents','noise'):
            shutil.copytree(late_dir/subdir,early_dir/subdir)
        early=launch_draft_decoder_worker(run_dir=early_dir,repo_root=repository,
            draft_artifact_spec_path=draft_spec,draft_artifact_spec_sha256=_sha256_file(draft_spec)[0],
            draft_verified_artifacts=draft_verified,visible_device=devices[0],chunk_count=16)
        from safetensors.torch import load_file
        import torch
        session=hashlib.sha256(b'actual-target-gate-early-session-v1').hexdigest()
        for req,control,draft,accepted,selected in records:
            i=req.history.next_chunk_index
            # Noise paths are relative to each phase root in the RPC protocol.
            request=replace(req,output_latent_path=f'early-latents/chunk-{i:02d}.safetensors')
            outcome=early.generate_early(request,session_id=session,control=control)
            row={'chunk_index':i,'replace':selected,'same_history_head':req.history.head_sha256,
                 'same_noise':req.noise.descriptor.descriptor_sha256}
            if selected:
                if not isinstance(outcome,StoppedDraft):raise ValueError('early route failed to stop')
                row['rank_equal']=outcome.decision['effective_rank']==draft.effective_rank
                row['commit']=early.commit_early_target(outcome,accepted)
                evidence=outcome.evidence['kv_reuse']
            else:
                if isinstance(outcome,StoppedDraft):raise ValueError('unexpected early replacement')
                row['rank_equal']=outcome.effective_rank==draft.effective_rank
                x=load_file(str(early_dir/outcome.segment.latent_path))['latent']
                y=load_file(str(late_dir/draft.segment.latent_path))['latent']
                row['latent_exact']=bool(torch.equal(x,y))
                row['max_abs']=float((x.float()-y.float()).abs().max())
                evidence=outcome.diagnostics['kv_reuse']
            row['reuse']=evidence
            if evidence['replayed_chunks'] != (1 if i==1 else 0):
                raise ValueError('early path replayed accepted history after initialization')
            report['rows'].append(row)
            print('EARLY_ACTUAL_TARGET',json.dumps(row,sort_keys=True),flush=True)
            if not row['rank_equal'] or not row.get('latent_exact',True):
                raise ValueError(f'first early/late divergence at chunk {i}')
        report['passed']=True
        return report
    finally:
        try:
            try:
                if early is not None:early.close()
            finally:
                try:
                    if runtime is not None:runtime.close()
                finally:
                    shutdown_owned_runtime_helpers(label='early-actual-target-rpc-gate')
        except BaseException as error:
            report['passed']=False
            report['cleanup_error']=f'{type(error).__name__}: {error}'
            raise
        finally:
            (output/'early-actual-target-gate.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


def main():
    from .main_table_baseline_campaign import _model_root, _run_dir, _visible_devices
    from .metadata import find_git_root
    run(_run_dir(),find_git_root(Path.cwd()),_model_root(),_visible_devices())


if __name__=='__main__':main()
