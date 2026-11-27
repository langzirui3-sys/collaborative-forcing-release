"""Read the original sealed keep-current-Draft handoff correctness gate.

This reader cannot enable a production factory or establish quality/speed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE = 'a8315de1f6f485d55602b730a2496f3b3b86edf8'
INDEX = '0a3c907de2c19487005c4f3057ed55c42fe1dc670058f4aecb41ccbb10caea2c'
RESULT = '0dd5311cb0e11eeb44df0c25568e61897a929e1778c495a4c7171cef263e30e7'
MANIFEST = 'experiments/rolling-committed-prefix-fast-gate-ntu-v1.manifest.json'
PROMPT = 'a car stuck in traffic during rush hour'


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_comparisons(data):
    require(data['schema'] == 'rolling_trained_committed_prefix_gate_v1'
            and data['status'] == 'succeeded' and data['prompt'] == PROMPT
            and data['seed'] == 20260911 and data['blocks'] == 16
            and data['target_chunks'] == [0,8,9,10,11]
            and data['rejected_draft_chunks'] == []
            and data['resumed_draft_chunks'] == [12,13,14,15]
            and data['rng_equal'] is True
            and data['execution_metadata'] == dict(host_kv_cursors=True,host_window=True),
            'committed handoff scenario or metadata differs')
    require(data['handoff'] == dict(schema='rolling_committed_clean_prefix_v1',
            action='accepted_prefix_and_target_suffix',retained_prefix_blocks=8,
            clean_suffix_blocks=4,full_history_replay_blocks=0,
            rollback_performed=False,checkpoint_undo_bytes=0,
            last_returned_draft_retained=True),
            'committed handoff did not retain eight blocks and append four without undo')
    require([r['chunk'] for r in data['comparisons']] == [12,13,14,15],
            'committed following-block coverage differs')
    for row in data['comparisons']:
        require(all(row[k] is True for k in ('latent_equal','rank_equal','cache_witness_equal')),
                'committed following latent/rank/witness mismatch')
        cache = row['full_cache_equality']
        require(cache['schema'] == 'rolling_full_populated_cache_equality_v1'
                and cache['equal'] is True and cache['unused_storage_compared'] is False
                and [r['layer'] for r in cache['layers']] == list(range(30)),
                'committed full cache coverage differs')
        for layer in cache['layers']:
            require(layer['global_end_index'] == (row['chunk']+1)*4680
                    and layer['local_end_index'] == 37440
                    and layer['compared_elements'] == 2*(37440+512)*12*128,
                    'committed cache cursors or compared element count differs')


def verify(run, *, repository_root):
    from .baseline_shard_report import source_manifest
    from .correction_state import LatentHistory, LatentSegment
    from .krea_chunk_backend import validate_target_forward_trace, validate_native_correction_state
    from .main_table_merge import _load_frozen_index
    from .rolling_host_cursors import validate_evidence
    from .rolling_transaction_gate import prerequisite
    from .rolling_transaction_proof import verify as verify_trained

    repo, run = Path(repository_root).resolve(strict=True), Path(run).resolve(strict=True)
    frozen, seal = _load_frozen_index(run)
    require(seal['git_sha'] == SOURCE and seal['artifact_index_sha256'] == INDEX,
            'committed original source/index differs')
    frozen.verify_all()
    prefix = 'attempts/'+seal['attempt_id']+'/'

    def read(name):
        return json.loads(frozen.file(name,label='committed handoff proof').read_text())

    manifest = source_manifest(repo,SOURCE,MANIFEST)
    envelope = read('resolved-manifest.json')
    require(envelope['manifest'] == manifest and all(envelope[k] == seal[k]
            for k in ('run_id','attempt_id','spec_hash')), 'committed source invocation differs')
    for item in manifest['spec']['inputs']:
        path = repo/item['target']
        require(not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],
                'committed implementation changed: '+item['target'])
    result_path = frozen.file(prefix+'result.json',label='committed result')
    require(hashlib.sha256(result_path.read_bytes()).hexdigest() == RESULT,
            'committed original result differs')
    data = read(prefix+'result.json');validate_comparisons(data)
    env = manifest['spec']['command']['env']
    original = prerequisite(env['CFEXP_ROLLING_PREFIX_GATE_RUN'])
    trained = verify_trained(env['CFEXP_ROLLING_TRAINED_PREFIX_GATE_RUN'],repository_root=repo)
    require(data['prerequisite'] == original and data['trained_prerequisite'] == trained,
            'committed prerequisite chain differs')
    models = read(prefix+'provenance/models.json')
    assets = read('resolved-manifest.json')['manifest']['spec']['inputs'][0]
    require(assets['target'] == 'experiments/erank3-rolling-native-v1.artifacts.json',
            'committed model specification differs')
    expected_models = json.loads((repo/assets['target']).read_text())['artifacts']
    require(len(models['rolling']) == len(expected_models)
            and [{k:r[k] for k in expected} for r,expected in zip(models['rolling'],expected_models)]
                == expected_models
            and models['target_worker']['checkpoint_sha256']
                == '792f6042a645e227f5d71851845dc4098a7338fec9ec55279f88ebbc3527eec7'
            and models['target_worker']['source_commit'] == 'acf4b7b4e1a3049a0e04782e751d12203334038d',
            'committed checkpoint identity differs')
    history = LatentHistory.from_dict(data['accepted_history']);history.verify_files(run/prefix)
    require(history.next_chunk_index == 16 and history.trajectory_seed == 20260911,
            'committed accepted history coverage differs')
    require(read(prefix+'provenance/accepted-history.json') == data['accepted_history'],
            'committed separate accepted history differs')
    accepted = LatentHistory(history.trajectory_id,history.prompt_id,history.trajectory_seed)
    expected = [('target',0),*[('draft',i) for i in range(1,8)],
                *[('target',i) for i in range(8,12)],*[('draft',i) for i in range(12,16)]]
    require([(r['role'],r['chunk']) for r in data['requests']] == expected,
            'committed actual generation request coverage differs')
    sessions = set()
    for row in data['requests']:
        segment = LatentSegment.from_dict(row['segment']);segment.verify_file(run/prefix)
        require(segment.chunk_index == row['chunk'] and segment.producer_role == row['role']
                and segment.model_id == ('rolling_forcing_1.3b' if row['role']=='draft'
                                         else 'krea-realtime-video-14b'),
                'committed segment model/role differs')
        accepted = accepted.append(segment)
        diag = row['diagnostics']
        if row['role'] == 'target':
            validate_target_forward_trace(diag['forward_trace'],diag['denoising_schedule'],
                history_frames=3*row['chunk'],expected_schedule=(1000.,750.,500.,250.))
            validate_native_correction_state(diag['native_correction_state'],chunk_index=row['chunk'])
            require(diag['explicit_initial_noise_verified'] is True,'committed Target noise differs')
            sessions.add(diag['native_correction_state']['native_session_id'])
        else:
            require(row['committed'] is True
                    and read(prefix+f"parity/draft-{row['chunk']:02d}.json") == row
                    and diag['metadata_execution'] == data['execution_metadata'],
                    'committed Draft receipt or metadata differs')
            cursor = diag['host_cursor_execution'];validate_evidence(cursor)
            require(all(r['global_end_index'] == (row['chunk']+1)*4680
                        and r['local_end_index'] == min((row['chunk']+1)*4680,37440)
                        for r in cursor['layers']), 'committed actual native cursor differs')
            if row['chunk'] == 12:
                require(diag['committed_prefix'] == data['handoff']
                        and diag['replayed_history_blocks'] == 4,
                        'committed actual handoff receipt differs')
    require(accepted == history and len(sessions) == 1,'committed accepted branch/Target owner differs')
    names = sorted(n for n in frozen.entries if n.startswith(prefix+'provenance/gpu-ownership-'))
    require(names == [prefix+f'provenance/gpu-ownership-{i:03d}.json' for i in range(35)],
            'committed GPU ownership coverage differs')
    owners = None
    for index,name in enumerate(names):
        row = read(name)
        require(row['schema'] == 'rolling_transaction_gpu_ownership_v1'
                and row['sequence'] == index and row['exclusive'] is True
                and [d['gpu_uuid'] for d in row['devices']] == data['gpu_uuids']
                and len(set(data['gpu_uuids'])) == len(row['devices']) == 2,
                'committed GPU identity differs')
        current = [d['allowed_pids'] for d in row['devices']]
        owners = current if owners is None else owners
        require(current == owners and all(len(d['allowed_pids']) == 1
                and not d['unexpected_pids'] and set(d['actual_pids']) <= set(d['allowed_pids'])
                for d in row['devices']), 'committed GPU ownership differs')
    return dict(schema='rolling_committed_prefix_proof_v1',source=SOURCE,
                artifact_index_sha256=INDEX,result_sha256=RESULT,run_id=seal['run_id'],
                attempt_id=seal['attempt_id'],retained_blocks=8,appended_blocks=4,
                following_blocks=[12,13,14,15],compared_layers=30,gpu_ownership_checks=35,
                native_sources_verified=True,rollback_performed=False,checkpoint_undo_bytes=0,
                original_proof=trained,
                claim_scope='trained committed handoff correctness only; no quality or speed claim')
