"""Two-GPU trained Target/Rolling rollback gate; no quality or speed claim.

Target root, seven accepted Draft blocks, rejected Draft block 8, four real
Target replacements at 8..11, then Draft 12..15. Compare the resumed latents,
eRank, full native K/V and RNG with independent fresh accepted-history replay.
Full-cache oracle clones are diagnostic overhead, never an execution strategy.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from .incremental_draft import clone_native_payload, latent_digest
from .main_table_baseline_campaign import (
    _gpu_compute_pids, _gpu_uuid, _model_root, _run_dir, _write_new_json,
)
from .rolling_clean_prefix_probe import compare_caches
from .rolling_forcing_session import RollingForcingSession

PROMPT = 'a car stuck in traffic during rush hour'
PROMPT_ID = 'rolling-transaction-gate'
SEED, BLOCKS = 20260911, 16
PREREQUISITE_SOURCE = '2bebf9ce7c33bf5322b113f93e6e61c626e708b6'
PREREQUISITE_INDEX = '413b2c962e148744984385ef4f9f790efa5532f519e78a064ffb437eb2b66d75'
ENV = 'CFEXP_ROLLING_PREFIX_TRANSACTION_DIAGNOSTIC'
FAST_ENV = 'CFEXP_ROLLING_TRANSACTION_FAST_METADATA'
COMMITTED_ENV = 'CFEXP_ROLLING_COMMITTED_PREFIX_GATE'


def metadata_mode():
    fast = os.environ.get(FAST_ENV, '0')
    require(fast in ('0','1'), 'transaction metadata mode must be 0 or 1')
    require(os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS','0') == fast
            and os.environ.get('CFEXP_ROLLING_HOST_WINDOW','0') == fast,
            'transaction metadata configuration must match its declared mode')
    return dict(host_kv_cursors=fast == '1', host_window=fast == '1')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def process_identity(pid):
    """Read identity only, never argv/environment, and never grant ownership."""
    path = Path('/proc')/str(pid)
    try:
        raw = (path/'stat').read_text()
        name, _, rest = raw.partition('(')[2].rpartition(')')
        fields = rest.split()
        return dict(pid=pid, name=name, state=fields[0], parent_pid=int(fields[1]),
                    process_group=int(fields[2]), session=int(fields[3]),
                    start_ticks=int(fields[19]), executable=os.readlink(path/'exe'))
    except (OSError, ValueError, IndexError):
        return dict(pid=pid, identity_unavailable=True)


def verify_gpu_ownership(output, sequence, uuids, owners, *, query=_gpu_compute_pids):
    """Seal the observed PIDs before rejecting; children are not auto-authorized."""
    require(len(uuids) == len(owners) == 2, 'two GPU owners required')
    rows = []
    for uuid, owner in zip(uuids, owners):
        actual = query(uuid)
        unexpected = actual - {owner}
        rows.append(dict(gpu_uuid=uuid, allowed_pids=[owner], actual_pids=sorted(actual),
            unexpected_pids=sorted(unexpected),
            identities=[process_identity(pid) for pid in sorted(actual | {owner})]))
    record = dict(schema='rolling_transaction_gpu_ownership_v1', sequence=sequence,
                  exclusive=not any(row['unexpected_pids'] for row in rows), devices=rows)
    _write_new_json(output/f'provenance/gpu-ownership-{sequence:03d}.json', record)
    require(record['exclusive'], 'diagnostic GPU acquired an unrelated process: '+
            '; '.join(f"{r['gpu_uuid']} actual={r['actual_pids']} allowed={r['allowed_pids']}"
                      for r in rows if r['unexpected_pids']))


def prerequisite(run):
    from .native_ready_report import _load_frozen_index
    frozen, seal = _load_frozen_index(Path(run))
    frozen.verify_all()
    require(seal['git_sha'] == PREREQUISITE_SOURCE
            and seal['artifact_index_sha256'] == PREREQUISITE_INDEX,
            'original native clean-prefix prerequisite seal differs')
    path = frozen.file('attempts/'+seal['attempt_id']+'/result.json', label='prefix gate')
    import json
    data = json.loads(path.read_text())
    require(data['status'] == 'succeeded' and data['prefix_marks'] == [1,4,8,12]
            and data['generated_blocks'] == 13 and data['rng_equal'] is True
            and len(data['continuation']) == 13
            and all(x['equal'] is True for x in data['continuation']),
            'native clean-prefix prerequisite did not pass')
    return dict(run_id=seal['run_id'], attempt_id=seal['attempt_id'],
                artifact_index_sha256=PREREQUISITE_INDEX, source=PREREQUISITE_SOURCE,
                result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def rank_capture(observation):
    from .self_forcing_frame_adapter import effective_rank_evidence_from_capture
    return effective_rank_evidence_from_capture(SimpleNamespace(retained_rank_layer=14,
        retained_gram_eigenvalues=observation['gram_eigenvalues'],
        ranks={14:observation['effective_rank']}))


def run():
    from .metadata import find_git_root
    from .correction_artifacts import verify_target_artifact_spec
    from .correction_runtime import DeterministicNoiseProvider, launch_target_generation_worker
    from .correction_state import LatentHistory, GenerationContext, KVCacheDescriptor, LatentSegment
    from .insight2_campaign import ChunkGenerationRequest, PromptCase
    from .rolling_forcing_backend import RollingNativeBackend
    from .rolling_forcing_native import CHECKPOINT_SHA256
    from .rolling_forcing_noise import seal_schedule, load_schedule

    require(os.environ.get(ENV) == '1', 'native transaction gate requires diagnostic opt-in')
    metadata = metadata_mode()
    committed_flag = os.environ.get(COMMITTED_ENV, '0')
    require(committed_flag in ('0','1'), 'committed gate switch must be 0 or 1')
    committed_route = committed_flag == '1'
    require(not committed_route or metadata['host_kv_cursors'],
            'committed route gate requires the native host metadata composition')
    proof = prerequisite(os.environ['CFEXP_ROLLING_PREFIX_GATE_RUN'])
    output, repo, models = _run_dir(), find_git_root(Path.cwd()), _model_root()
    trained_proof = None
    if metadata['host_kv_cursors']:
        from .rolling_transaction_proof import verify
        trained_proof = verify(os.environ['CFEXP_ROLLING_TRAINED_PREFIX_GATE_RUN'],
                               repository_root=repo)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = visible.split(',')
    require(len(devices) == 2 and all(x.strip() for x in devices) and devices[0] != devices[1],
            'two exclusive diagnostic GPUs required; this is not a single-card benchmark')
    uuids = [_gpu_uuid(x) for x in devices]
    require(not any(_gpu_compute_pids(x) for x in uuids), 'diagnostic GPU is occupied')
    spec = repo/'experiments/insight2-krea14b.artifacts.json'
    _, records, adapter = verify_target_artifact_spec(spec, model_root=models, repo_root=repo)
    require(adapter == 'cfexp.krea_chunk_backend', 'unexpected real Target adapter')
    native = target = candidate = reference = None
    try:
        target = launch_target_generation_worker(run_dir=output, repo_root=repo,
            target_artifact_spec_path=spec,
            target_artifact_spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
            target_verified_artifacts=records, visible_device=devices[1], chunk_count=BLOCKS)
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
        native = RollingNativeBackend(run_dir=output, repository_root=repo, model_root=models)
        t, p = native.torch, native.pipeline
        from safetensors.torch import load, save_file

        ownership_sequence = 0

        def ownership():
            nonlocal ownership_sequence
            verify_gpu_ownership(output, ownership_sequence, uuids,
                                 (os.getpid(), target.process.pid))
            ownership_sequence += 1

        provider = DeterministicNoiseProvider(run_dir=output)
        receipt = seal_schedule(provider=provider, prompt_id=PROMPT_ID, prompt=PROMPT,
            trajectory_seed=SEED, chunk_count=BLOCKS)
        binding = dict(run_dir=output, relative_path=receipt['path'], sha256=receipt['sha256'],
            prompt_id=PROMPT_ID, prompt=PROMPT, trajectory_seed=SEED,
            namespace=provider.namespace, chunk_count=BLOCKS)
        noise, seed = load_schedule(torch=t, load_file=load, device=native.device, **binding)
        require(seed == SEED, 'native transaction noise seed differs')
        _write_new_json(output/'provenance/models.json', dict(rolling=native.records,
            native_load=native.audit, target=records, target_worker=target.model_state,
            target_policy='fresh-history diagnostic RPC, no optimized timing claim',
            prerequisite=proof, trained_prerequisite=trained_proof,
            execution_metadata=metadata, noise_schedule=receipt, gpu_uuids=uuids))

        def read_latent(segment):
            segment.verify_file(output)
            return load((output/segment.latent_path).read_bytes())['latent']

        def initial(index):
            return provider.materialize(prompt_id=PROMPT_ID, trajectory_seed=SEED, chunk_index=index)

        prompt = PromptCase(prompt_id=PROMPT_ID, prompt=PROMPT, split='evaluation')
        accepted = LatentHistory(trajectory_id=PROMPT_ID, prompt_id=PROMPT_ID, trajectory_seed=SEED)
        values, requests, following, snapshots = [], [], [], []
        target_sessions = []

        def target_block(history):
            index = history.next_chunk_index
            ownership()
            result = target.generate(ChunkGenerationRequest(prompt=prompt, history=history,
                noise=initial(index), output_latent_path=f'latents/target-{index:02d}.safetensors',
                capture_hidden=False, target_logical_call_id=hashlib.sha256(f'target-{index}'.encode()).hexdigest(),
                target_call_scope_id=hashlib.sha256(PROMPT_ID.encode()).hexdigest(),
                target_call_phase='route_materialization'))
            from .krea_chunk_backend import validate_target_forward_trace, validate_native_correction_state
            diagnostics = result.diagnostics
            validate_target_forward_trace(diagnostics['forward_trace'],
                diagnostics['denoising_schedule'], history_frames=3*index,
                expected_schedule=(1000.,750.,500.,250.))
            validate_native_correction_state(diagnostics['native_correction_state'], chunk_index=index)
            require(diagnostics['explicit_initial_noise_verified'] is True,
                    'real Target omitted explicit initial-noise verification')
            target_sessions.append(diagnostics['native_correction_state']['native_session_id'])
            require(len(set(target_sessions)) == 1, 'Target correction session was recreated')
            requests.append(dict(role='target', chunk=index, segment=result.segment.to_dict(),
                                 diagnostics=dict(result.diagnostics)))
            return result.segment

        root = target_block(accepted);accepted = accepted.append(root);values.append(read_latent(root))
        if committed_route:
            from .rolling_committed_prefix import CommittedPrefixSession
            candidate = CommittedPrefixSession(pipeline=p, torch=t, noise=noise, prompt=PROMPT,
                seed=SEED, observe_step_zero=True, capture_cache=True)
        else:
            candidate = RollingForcingSession(pipeline=p, torch=t, noise=noise, prompt=PROMPT,
                seed=SEED, observe_step_zero=True, capture_cache=True, prefix_transactions=True)

        def generate(session, history_values):
            ownership()
            history = t.cat(history_values, dim=1).to(device=native.device)
            with t.inference_mode():
                result = session.generate(history=history)
            require(session.stream.host_kv_cursors is metadata['host_kv_cursors']
                    and session.stream.host_window is metadata['host_window'],
                    'actual native metadata execution differs from declared mode')
            if metadata['host_kv_cursors']:
                from .rolling_host_cursors import validate_evidence
                validate_evidence(result['host_cursor_execution'])
            else:
                require('host_cursor_execution' not in result,
                        'unexpected host cursor execution in original metadata mode')
            result['metadata_execution'] = dict(metadata)
            ownership()
            observation = result.pop('step_zero_observation')
            result['rank_evidence'] = rank_capture(observation)
            result['kv_witness_sha256'] = observation['kv_witness'][0]
            return result

        def store_draft(result, history, *, committed):
            index = history.next_chunk_index
            path = f'latents/draft-{index:02d}.safetensors'
            tensor = result['latent'].detach().cpu().contiguous()
            require(tuple(tensor.shape) == (1,3,16,60,104) and tensor.dtype == t.bfloat16
                    and bool(t.isfinite(tensor).all()), 'invalid native Draft latent')
            require(not (output/path).exists(), 'diagnostic Draft output already exists')
            save_file({'latent':tensor}, str(output/path))
            kv = KVCacheDescriptor(owner_role='draft', model_id='rolling_forcing_1.3b',
                model_artifact_sha256=CHECKPOINT_SHA256, history_head_sha256=history.head_sha256,
                cache_sha256=result['kv_witness_sha256'])
            context = GenerationContext(role='draft', model_id='rolling_forcing_1.3b',
                model_artifact_sha256=CHECKPOINT_SHA256, history_head_sha256=history.head_sha256,
                next_chunk_index=index, noise=initial(index).descriptor, kv=kv)
            segment = LatentSegment.from_file(history=history, context=context,
                latent_root=output, latent_path=path)
            row = dict(role='draft', chunk=index, committed=committed, segment=segment.to_dict(),
                       diagnostics={k:v for k,v in result.items() if k != 'latent'})
            requests.append(row);_write_new_json(output/f'parity/draft-{index:02d}.json', row)
            return segment, tensor

        # The production router retains Draft 7 and assigns 8..11 to Target.
        # The original rollback gate additionally proposed/rejected Draft 8.
        for index in range(1,8 if committed_route else 9):
            result = generate(candidate, values)
            segment, value = store_draft(result, accepted, committed=index != 8)
            if index != 8:
                accepted = accepted.append(segment);values.append(value)
        require(accepted.next_chunk_index == 8, 'rejected candidate entered accepted history')
        for index in range(8,12):
            segment = target_block(accepted)
            accepted = accepted.append(segment);values.append(read_latent(segment))
        reference_prefix = list(values)
        for index in range(12,16):
            result = generate(candidate, values)
            if index == 12:
                if committed_route:
                    handoff = result['committed_prefix']
                    require(handoff['schema'] == 'rolling_committed_clean_prefix_v1'
                            and handoff['action'] == 'accepted_prefix_and_target_suffix'
                            and handoff['retained_prefix_blocks'] == 8
                            and handoff['clean_suffix_blocks'] == 4
                            and handoff['full_history_replay_blocks'] == 0
                            and handoff['rollback_performed'] is False
                            and handoff['checkpoint_undo_bytes'] == 0
                            and handoff['last_returned_draft_retained'] is True
                            and result['replayed_history_blocks'] == 4,
                            'committed route did not retain its accepted Draft and append four Target blocks')
                else:
                    handoff = result['prefix_transaction']
                    require(handoff['action'] == 'retained_after_rollback'
                            and handoff['retained_prefix_blocks'] == 8
                            and handoff['clean_suffix_blocks'] == 4
                            and handoff['full_history_replay_blocks'] == 0
                            and result['replayed_history_blocks'] == 4
                            and handoff['rollback']['evicted_tokens_per_layer'] == [4680]*30,
                            'real Target handoff did not retain its native prefix')
            segment, value = store_draft(result, accepted, committed=True)
            accepted = accepted.append(segment);values.append(value)
            following.append(dict(latent=value, rank=result['rank_evidence'], witness=result['kv_witness_sha256']))
            with t.no_grad():snapshots.append(clone_native_payload((p.kv_cache_clean,p.crossattn_cache),t))
        candidate_rng = candidate.stream.rng_state.clone()
        candidate.close();candidate = None
        reference = RollingForcingSession(pipeline=p, torch=t, noise=noise, prompt=PROMPT,
            seed=SEED, observe_step_zero=True, capture_cache=True)
        comparisons = []
        for index, expected in enumerate(following, 12):
            actual = generate(reference, reference_prefix)
            require(t.equal(actual['latent'].detach().cpu(), expected['latent'])
                    and actual['rank_evidence'] == expected['rank']
                    and actual['kv_witness_sha256'] == expected['witness'],
                    f'real Target handoff latent/eRank/witness divergence at block {index}')
            caches = compare_caches(snapshots[index-12], (p.kv_cache_clean,p.crossattn_cache), torch=t)
            snapshots[index-12] = None
            comparisons.append(dict(chunk=index, latent_equal=True, rank_equal=True,
                cache_witness_equal=True, full_cache_equality=caches,
                latent_sha256=latent_digest(expected['latent'],t)))
            reference_prefix.append(actual['latent'].detach().cpu().contiguous())
        require(t.equal(candidate_rng, reference.stream.rng_state), 'native handoff continuation RNG differs')
        accepted.verify_files(output)
        _write_new_json(output/'provenance/accepted-history.json', accepted.to_dict())
        result = dict(schema=('rolling_trained_committed_prefix_gate_v1' if committed_route else
                              'rolling_trained_prefix_transaction_gate_v1'), status='succeeded',
            prompt=PROMPT, seed=SEED, blocks=BLOCKS, target_chunks=[0,8,9,10,11],
            rejected_draft_chunks=[] if committed_route else [8], resumed_draft_chunks=[12,13,14,15],
            prerequisite=proof, handoff=handoff, comparisons=comparisons, rng_equal=True,
            trained_prerequisite=trained_proof, execution_metadata=metadata,
            requests=requests, gpu_uuids=uuids, accepted_history=accepted.to_dict(),
            claim_scope=('two-GPU committed Draft prefix and Target append correctness only; fusion, quality and speed untested'
                         if committed_route else
                         'two-GPU trained Target rollback/append correctness only; fusion, quality and speed untested'))
        _write_new_json(output/'result.json',result)
        return result
    finally:
        if reference is not None:reference.close()
        if candidate is not None:candidate.close()
        if native is not None:native.close()
        if target is not None:target.close()
        os.environ['CUDA_VISIBLE_DEVICES'] = visible


def main():
    from .runtime_helpers import shutdown_owned_runtime_helpers
    try:
        run()
        return 0
    finally:
        shutdown_owned_runtime_helpers(label='rolling-transaction-gate')


if __name__ == '__main__':raise SystemExit(main())
