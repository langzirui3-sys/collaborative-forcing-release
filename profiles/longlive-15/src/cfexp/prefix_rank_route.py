"""Experimental T1000/native-Draft-suffix route with explicit partial-call accounting."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from .early_protocol import StoppedDraft, decide, validate_control, require
from .correction_policy import SuddenDropRiskCalibrator
from .insight2_campaign import ChunkGenerationRequest, GeneratedChunk
from .self_forcing_frame_adapter import validate_effective_rank_evidence


def materialize_prefix_rank(campaign, *, prompt, seed, budget, calibrator):
    require(isinstance(calibrator, SuddenDropRiskCalibrator), 'early route requires frozen drop calibration')
    require(type(budget) is int and 0 < budget <= campaign.chunk_count-1, 'invalid early route budget')
    route = f'target-first-update-erank-k{budget}'
    key = campaign._rollout_key(prompt, seed, route)
    # The live RPC capability cannot be resumed from a half-written checkpoint.
    # Never silently replay a run or mix ordinary Draft with its early session.
    require(not campaign._checkpoint_path('rollouts', key, 0).exists(),
            'early route requires a fresh trajectory; existing output is preserved')
    history, root = campaign._canonical_rollout_root(prompt=prompt, seed=seed)
    campaign._write_checkpoint(path=campaign._checkpoint_path('rollouts', key, 0),
        history=history, generated=root, metadata={'kind': 'target_first_update_rank',
            'route_id': route, 'chunk_index': 0, 'corrected': True, 'committed_role': 'target'})
    session = hashlib.sha256(f'{key}:{history.head_sha256}:early-v1'.encode()).hexdigest()
    ranks, selected = [], []
    for index in range(1, campaign.chunk_count):
        noise = campaign._materialize_noise(prompt=prompt, seed=seed, chunk_index=index)
        output = campaign._fresh_output_path(f'latents/rollouts/{key}/chunk-{index:02d}-early-draft')
        request = ChunkGenerationRequest(prompt=prompt, history=history, noise=noise,
            output_latent_path=output, capture_hidden=False)
        control = {'schema': 'early_erank_control_v1', 'prior_ranks': list(ranks),
            'calibration_id': calibrator.calibration_id, 'calibration_drop_scores': list(calibrator.drop_scores),
            'remaining_budget': budget-len(selected), 'remaining_slots': campaign.chunk_count-index}
        outcome = campaign.draft.generate_early(request, session_id=session, control=control)
        stopped = isinstance(outcome, StoppedDraft)
        require(stopped or isinstance(outcome, GeneratedChunk), 'invalid early Draft result')
        rank = outcome.decision['effective_rank'] if stopped else outcome.effective_rank
        decision = decide(control, rank)
        evidence = dict(outcome.evidence if stopped else outcome.diagnostics)
        require(evidence.get('early_decision') == decision and decision['replace_with_target'] is stopped,
                'early worker and controller decisions disagree')
        commit = None
        if stopped:
            require(outcome.request == request and outcome.session_id == session
                    and not (campaign.run_dir/output).exists(), 'stopped Draft has an output or wrong input binding')
            from .prefix_client import target_prefix, resume_prefix
            prefix = target_prefix(campaign.target, request,
                output_path=campaign._fresh_output_path(f'prefix/{key}/chunk-{index:02d}'))
            noises = [campaign.noise_provider.materialize(prompt_id=prompt.prompt_id+f'-suffix-noise-{step}',
                trajectory_seed=seed, chunk_index=index).tensor_path for step in (1,2,3)]
            accepted = resume_prefix(campaign.draft, outcome, prefix, renoise_paths=noises)
            # Prefix is a Target computation, but NOT a completed Target chunk.
            accepted = replace(accepted, diagnostics={**accepted.diagnostics,
                'kv_reuse': evidence['kv_reuse']})
            commit = {'target_prefix': prefix, 'renoise_paths': noises}
            selected.append(index)
        else:
            accepted = outcome
            require(accepted.context.role == 'draft' and accepted.context.model_id == campaign.draft.model_id
                    and accepted.context.model_artifact_sha256 == campaign.draft.model_artifact_sha256
                    and accepted.context.noise == noise.descriptor
                    and accepted.segment.latent_path == output, 'completed Draft changed requested context/output')
            accepted.segment.verify_context(history, accepted.context)
            accepted.segment.verify_file(campaign.run_dir)
            validate_effective_rank_evidence(evidence.get('rank_evidence'), expected_rank=rank,
                                            expected_eigenvalue_count=256)
        ranks.append(rank)
        metadata = {'kind': 'target_first_update_rank', 'route_id': route, 'chunk_index': index,
            'rank': rank, 'risk_score': decision['risk_score'], 'erank_drop_score': decision['drop_score'],
            'decision': decision, 'corrected': stopped, 'committed_role': accepted.context.role,
            'early_control': control, 'early_evidence': evidence, 'early_commit': commit,
            'early_output_path': output, 'early_session_id': session,
            'parent_history_sha256': history.head_sha256,
            'initial_noise_descriptor_sha256': noise.descriptor.descriptor_sha256}
        history = history.append(accepted.segment)
        campaign._write_checkpoint(path=campaign._checkpoint_path('rollouts', key, index),
            history=history, generated=accepted, metadata=metadata)
    require(len(selected) == budget, 'early route did not consume its frozen Target budget')
    return key, route, tuple(selected), history


def collect_prefix_trajectory(campaign, *, key, history, accepted, proposed):
    from .branch_gate_probe import collect_trajectory
    from .krea_chunk_backend import validate_native_correction_state
    from .target_prefix import validate_prefix_trace,validate_draft_suffix_evidence
    trajectory = collect_trajectory(campaign, key=key, history=history, accepted=accepted, proposed=proposed)
    ranks, selected, states = [], [], {}
    root = json.loads(campaign._checkpoint_path('rollouts', key, 0).read_text())
    states[0] = root['generated']['diagnostics']['native_correction_state']
    validate_native_correction_state(states[0], chunk_index=0)
    calibration = session = None
    for i in range(1, campaign.chunk_count):
        item = json.loads(campaign._checkpoint_path('rollouts', key, i).read_text())
        m, g = item['metadata'], item['generated']; d = g['diagnostics']
        require(m['kind']=='target_first_update_rank' and m['chunk_index']==i, 'prefix checkpoint drift')
        c = validate_control(m['early_control'], chunk_index=i)
        require(c['prior_ranks']==ranks and c['remaining_slots']==campaign.chunk_count-i
                and c['remaining_budget']==len(proposed)-len(selected), 'prefix causal control drift')
        current = (c['calibration_id'],c['calibration_drop_scores'])
        if calibration is None: calibration,session = current,m['early_session_id']
        require(current==calibration and session==m['early_session_id'], 'prefix calibration/session drift')
        decision=decide(c,m['rank']);stop=decision['replace_with_target'];e=m['early_evidence'];kv=e['kv_reuse']
        require(decision==m['decision']==e['early_decision'] and stop is m['corrected']
                and e['early_execution']['stopped'] is stop
                and e['early_execution']['branch_snapshot_bytes']==0, 'prefix decision/execution drift')
        require(kv['reused_chunks']==(0 if i==1 else i) and kv['replayed_chunks']==(1 if i==1 else 0)
                and d['kv_reuse']==kv, 'prefix lost native KV reuse')
        require(g['context']['role']=='draft' and g['segment']['latent_path']==m['early_output_path']
                and g['context']['history_head_sha256']==m['parent_history_sha256']==history.prefix_before(i).head_sha256
                and g['context']['noise']['descriptor_sha256']==m['initial_noise_descriptor_sha256'],
                'prefix output history/noise binding drift')
        if stop:
            p=e['probe'];prefix=m['early_commit']['target_prefix'];result=prefix['result'];binding=prefix['input_binding']
            require(p['layer_index']==14 and p['timestep']==1000 and p['current_start']==i*4680
                    and p['completed_native_forward'] is False and p['effective_rank']==m['rank'], 'invalid prefix probe')
            h=history.prefix_before(i)
            require(binding['chunk_index']==i and binding['hashes'][:-1]==[s.latent_sha256 for s in h.segments]
                    and binding['hashes'][-1]==g['context']['noise']['tensor_sha256'], 'Target prefix input binding drift')
            require(result['completed_target_chunk'] is False and 'diagnostic_oracle' not in result
                    and d['target_prefix']==prefix and validate_draft_suffix_evidence(d)
                    and d['draft_clean_forwards']==1 and d['history_replay_chunks']==0
                    and d['branch_snapshot_bytes']==0 and d['committed_chunks']==i+1
                    and d['committed_kv_width']==3 and 'diagnostic_oracle' not in d,
                    'prefix computed extra steps, replay or oracle work')
            validate_prefix_trace(result['forward_trace'],history_frames=i*3)
            states[i]=result['native_correction_state'];validate_native_correction_state(states[i],chunk_index=i)
            selected.append(i)
        else: require(m['early_commit'] is None, 'unselected chunk has a prefix')
        ranks.append(m['rank'])
    require(selected==list(accepted)==list(proposed), 'prefix intervention budget drift')
    require(len({s['native_session_id'] for s in states.values()})==1
            and all(not s['session_recreated'] for i,s in states.items() if i), 'Target session recreated')
    n=campaign.chunk_count-1;k=len(selected)
    trajectory['counters'].update(physical_calls={'draft':n,'target':1,'target_first_update':k},
        target_denoising_forwards=4+k, draft_complete_four_step_blocks=n-k,
        draft_stopped_probes=k, draft_three_step_suffixes=k, draft_clean_updates=n)
    trajectory['prefix_execution_audit']={'schema':'target_prefix_route_audit_v1',
        'probe_count':n,'prefix_count':k,'full_target_chunks':1,'history_replay_after_root':0,
        'branch_snapshot_bytes':0,'diagnostic_oracle_forwards':0,'validated':True}
    trajectory['prefix_native_states']=states
    return trajectory

