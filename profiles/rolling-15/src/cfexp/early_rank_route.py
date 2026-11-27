"""Current-block early routing, without representing partial probes as latents."""
from __future__ import annotations

import hashlib
import json

from .early_protocol import StoppedDraft, decide, validate_control, require
from .correction_policy import SuddenDropRiskCalibrator
from .insight2_campaign import ChunkGenerationRequest, GeneratedChunk
from .self_forcing_frame_adapter import validate_effective_rank_evidence


def materialize_early_rank(campaign, *, prompt, seed, budget, calibrator):
    require(isinstance(calibrator, SuddenDropRiskCalibrator), 'early route requires frozen drop calibration')
    require(type(budget) is int and 0 < budget < campaign.chunk_count-1, 'invalid early route budget')
    route = f'early-current-erank-k{budget}'
    key = campaign._rollout_key(prompt, seed, route)
    # The live RPC capability cannot be resumed from a half-written checkpoint.
    # Never silently replay a run or mix ordinary Draft with its early session.
    require(not campaign._checkpoint_path('rollouts', key, 0).exists(),
            'early route requires a fresh trajectory; existing output is preserved')
    history, root = campaign._canonical_rollout_root(prompt=prompt, seed=seed)
    campaign._write_checkpoint(path=campaign._checkpoint_path('rollouts', key, 0),
        history=history, generated=root, metadata={'kind': 'early_current_rank',
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
            accepted = campaign._generate(backend=campaign.target, prompt=prompt, history=history, noise=noise,
                output_stem=f'latents/rollouts/{key}/chunk-{index:02d}-target', capture_hidden=False)
            committed = campaign.draft.commit_early_target(outcome, accepted)
            # Only persist consumed-result evidence, not the transient capability.
            commit = {k: committed[k] for k in ('target_latent_path', 'target_latent_sha256', 'result')}
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
        metadata = {'kind': 'early_current_rank', 'route_id': route, 'chunk_index': index,
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


def collect_early_trajectory(campaign, *, key, history, accepted, proposed):
    from .branch_gate_probe import collect_trajectory
    trajectory = collect_trajectory(campaign, key=key, history=history, accepted=accepted, proposed=proposed)
    ranks, selected, probes, commits = [], [], [], []
    calibration = session = None
    for index in range(1, campaign.chunk_count):
        item = json.loads(campaign._checkpoint_path('rollouts', key, index).read_text())
        m, generated = item['metadata'], item['generated']
        require(m['kind'] == 'early_current_rank' and m['chunk_index'] == index, 'early checkpoint changed')
        c = validate_control(m['early_control'], chunk_index=index)
        require(c['prior_ranks'] == ranks and c['remaining_slots'] == campaign.chunk_count-index
                and c['remaining_budget'] == len(proposed)-len(selected), 'early causal control drift')
        current = (c['calibration_id'], c['calibration_drop_scores'])
        if calibration is None: calibration, session = current, m['early_session_id']
        require(current == calibration and m['early_session_id'] == session, 'early session/calibration drift')
        decision = decide(c, m['rank']); stop = decision['replace_with_target']
        e = m['early_evidence']; execution = e['early_execution']; kv = e['kv_reuse']
        require(m['decision'] == e['early_decision'] == decision and m['corrected'] is stop
                and execution['schema'] == 'draft_early_execution_v1' and execution['stopped'] is stop
                and execution['branch_snapshot_bytes'] == 0, 'early execution evidence disagrees')
        require(type(kv['reused_chunks']) is int and kv['reused_chunks'] == (0 if index==1 else index)
                and type(kv['replayed_chunks']) is int and kv['replayed_chunks'] == (1 if index==1 else 0),
                'early route replayed history after its initial root')
        prefix = history.prefix_before(index)
        require(m['parent_history_sha256'] == prefix.head_sha256
                and generated['context']['history_head_sha256'] == prefix.head_sha256
                and m['initial_noise_descriptor_sha256'] == generated['context']['noise']['descriptor_sha256'],
                'early accepted context is not bound to the observed history/noise')
        if stop:
            p = e['probe']; commit = m['early_commit']
            require(p['layer_index'] == 14 and p['timestep'] == 1000
                    and p['current_start'] == index*4680 and p['completed_native_forward'] is False
                    and p['effective_rank'] == m['rank'] and 'latent' not in e
                    and not (campaign.run_dir/m['early_output_path']).exists(), 'invalid stopped Draft probe')
            require(generated['context']['role'] == 'target'
                    and commit['target_latent_path'] == generated['segment']['latent_path']
                    and commit['target_latent_sha256'] == generated['segment']['latent_sha256']
                    and commit['result'] == {'external_clean_forwards':1, 'history_replay_chunks':0,
                        'branch_snapshot_bytes':0, 'committed_chunks':index+1}, 'invalid accepted Target cache commit')
            selected.append(index); commits.append(commit)
        else:
            require(generated['context']['role'] == 'draft' and m['early_commit'] is None
                    and generated['segment']['latent_path'] == m['early_output_path'], 'invalid continued Draft')
        ranks.append(m['rank']); probes.append(kv)
    require(selected == list(accepted) == list(proposed), 'early accepted budget/positions drift')
    counters = trajectory['counters']
    counters.update(physical_calls={'draft':len(probes), 'target':1+len(selected)},
        completed_draft_generations=len(probes)-len(selected), stopped_draft_probes=len(selected),
        current_slot_clean_updates=len(commits), draft_evidence_count=len(probes),
        reused_chunks=sum(x['reused_chunks'] for x in probes), replayed_chunks=sum(x['replayed_chunks'] for x in probes))
    trajectory['early_execution_audit'] = {'schema':'early_route_audit_v1', 'probe_count':len(probes),
        'stopped_probe_count':len(selected), 'completed_draft_count':len(probes)-len(selected),
        'current_slot_clean_updates':len(commits), 'history_replay_after_root':0,
        'branch_snapshot_bytes':0, 'validated':True}
    return trajectory
