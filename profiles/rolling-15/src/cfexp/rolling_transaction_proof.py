"""Read the original sealed trained handoff proof, without generating a video.

The fixed source/index identify one correctness gate, not a performance result.
Keep this separate from a caller's current source and never rebuild its index.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE = 'b2017b4b02e9dcbfc43d75c41faa7a51a2523499'
INDEX = 'd834d08afa6cbeca491c0a33b3e555815053b350da14560bf026eed2bdc77295'
MANIFEST = 'experiments/rolling-trained-prefix-transaction-gate-ntu-v1.manifest.json'
PROMPT = 'a car stuck in traffic during rush hour'
FAST_SOURCE = 'd5ac60f0ab8215397c6a981eef50c2da1f7f0189'
FAST_INDEX = '7dd21f1af40d06b1eea84f28826fb51c534c648115848a810df004ee375a3fab'
FAST_MANIFEST = 'experiments/rolling-trained-prefix-fast-metadata-gate-ntu-v1.manifest.json'
FAST_RESULT = '58152819148d0cce931453a5c946a6330e0022b6cbb6806e7e5b47cc558bb5f8'


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_comparisons(data):
    """Check coverage and exact comparison scope; this alone is not a seal check."""
    require(data['schema'] == 'rolling_trained_prefix_transaction_gate_v1'
            and data['status'] == 'succeeded' and data['prompt'] == PROMPT
            and data['seed'] == 20260911 and data['blocks'] == 16
            and data['target_chunks'] == [0,8,9,10,11]
            and data['rejected_draft_chunks'] == [8]
            and data['resumed_draft_chunks'] == [12,13,14,15]
            and data['rng_equal'] is True, 'trained handoff scenario differs')
    handoff = data['handoff']
    undo = handoff['rollback']
    require(handoff['schema'] == 'rolling_native_prefix_transaction_v1'
            and handoff['action'] == 'retained_after_rollback'
            and handoff['retained_prefix_blocks'] == 8
            and handoff['clean_suffix_blocks'] == 4
            and handoff['full_history_replay_blocks'] == 0
            and handoff['checkpoint_undo_bytes'] == 862617600
            and undo['schema'] == 'rolling_native_prefix_rollback_v1'
            and undo['prefix_blocks'] == 8 and undo['unused_slots_restored'] is False
            and undo['copied_undo_bytes'] == 862617600
            and undo['evicted_tokens_per_layer'] == [4680]*30,
            'trained handoff did not retain eight blocks and append four')
    require([r['chunk'] for r in data['comparisons']] == [12,13,14,15],
            'trained following-block coverage differs')
    for row in data['comparisons']:
        require(all(row[k] is True for k in ('latent_equal','rank_equal','cache_witness_equal')),
                'trained following latent/rank/witness mismatch')
        cache = row['full_cache_equality']
        require(cache['schema'] == 'rolling_full_populated_cache_equality_v1'
                and cache['equal'] is True and cache['unused_storage_compared'] is False
                and [r['layer'] for r in cache['layers']] == list(range(30)),
                'trained full cache coverage differs')
        for layer in cache['layers']:
            require(layer['global_end_index'] == (row['chunk']+1)*4680
                    and layer['local_end_index'] == 37440
                    and layer['compared_elements'] == 2*(37440+512)*12*128,
                    'trained cache cursors or compared element count differs')


def verify(run, *, repository_root):
    from .baseline_shard_report import source_manifest
    from .correction_state import LatentHistory, LatentSegment
    from .krea_chunk_backend import validate_target_forward_trace, validate_native_correction_state
    from .main_table_merge import _load_frozen_index
    from .rolling_transaction_gate import prerequisite

    repo, run = Path(repository_root).resolve(strict=True), Path(run).resolve(strict=True)
    frozen, seal = _load_frozen_index(run)
    require(seal['git_sha'] == SOURCE and seal['artifact_index_sha256'] == INDEX,
            'original trained handoff source/index differs')
    frozen.verify_all()
    prefix = 'attempts/'+seal['attempt_id']+'/'

    def read(relative):
        path = frozen.file(relative, label='original trained handoff proof')
        return json.loads(path.read_text())

    envelope = read('resolved-manifest.json')
    manifest = source_manifest(repo, SOURCE, MANIFEST)
    require(envelope['manifest'] == manifest and all(envelope[k] == seal[k]
            for k in ('run_id','attempt_id','spec_hash')), 'trained invocation/source manifest differs')
    # Bind the mathematical/native execution implementation. The diagnostic
    # driver may evolve to exercise additional metadata configurations.
    for item in manifest['spec']['inputs']:
        if item['target'] == 'src/cfexp/rolling_transaction_gate.py':
            continue
        path = repo/item['target']
        require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],
                'trained native implementation changed: '+item['target'])
    data = read(prefix+'result.json')
    validate_comparisons(data)
    models = read(prefix+'provenance/models.json')
    spec = json.loads((repo/'experiments/erank3-rolling-native-v1.artifacts.json').read_text())
    require([{k:r[k] for k in expected} for r,expected in zip(models['rolling'], spec['artifacts'])]
            == spec['artifacts'] and len(models['rolling']) == len(spec['artifacts'])
            and models['target_worker']['checkpoint_sha256']
            == '792f6042a645e227f5d71851845dc4098a7338fec9ec55279f88ebbc3527eec7'
            and models['target_worker']['source_commit'] == 'acf4b7b4e1a3049a0e04782e751d12203334038d',
            'trained model/checkpoint identity differs')
    original_prefix = prerequisite(manifest['spec']['command']['env']['CFEXP_ROLLING_PREFIX_GATE_RUN'])
    require(data['prerequisite'] == original_prefix, 'original clean-prefix proof differs')
    history = LatentHistory.from_dict(data['accepted_history'])
    require(history.next_chunk_index == 16 and history.trajectory_seed == 20260911,
            'trained accepted history coverage differs')
    history.verify_files(run/prefix)
    accepted = LatentHistory(trajectory_id=history.trajectory_id, prompt_id=history.prompt_id,
                            trajectory_seed=history.trajectory_seed)
    expected = [('target',0), *[('draft',i) for i in range(1,9)],
                *[('target',i) for i in range(8,12)], *[('draft',i) for i in range(12,16)]]
    require([(r['role'],r['chunk']) for r in data['requests']] == expected,
            'trained actual generation request coverage differs')
    sessions = set()
    for row in data['requests']:
        segment = LatentSegment.from_dict(row['segment'])
        segment.verify_file(run/prefix)
        advanced = accepted.append(segment)
        require(segment.chunk_index == row['chunk'] and segment.producer_role == row['role']
                and segment.model_id == ('rolling_forcing_1.3b' if row['role'] == 'draft'
                                         else 'krea-realtime-video-14b'),
                'trained segment model/role differs')
        diagnostics = row['diagnostics']
        if row['role'] == 'target':
            validate_target_forward_trace(diagnostics['forward_trace'], diagnostics['denoising_schedule'],
                history_frames=3*row['chunk'], expected_schedule=(1000.,750.,500.,250.))
            validate_native_correction_state(diagnostics['native_correction_state'], chunk_index=row['chunk'])
            require(diagnostics['explicit_initial_noise_verified'] is True,
                    'trained Target initial noise was not verified')
            sessions.add(diagnostics['native_correction_state']['native_session_id'])
        else:
            require(row['committed'] is (row['chunk'] != 8)
                    and read(prefix+f"parity/draft-{row['chunk']:02d}.json") == row,
                    'trained Draft checkpoint differs')
        if row['role'] != 'draft' or row['chunk'] != 8:
            accepted = advanced
    require(accepted == history and len(sessions) == 1, 'trained history or persistent Target differs')
    names = sorted(n for n in frozen.entries if n.startswith(prefix+'provenance/gpu-ownership-'))
    require(names == [prefix+f'provenance/gpu-ownership-{i:03d}.json' for i in range(37)],
            'trained GPU ownership coverage differs')
    owners = None
    for index, name in enumerate(names):
        row = read(name)
        require(row['schema'] == 'rolling_transaction_gpu_ownership_v1'
                and row['sequence'] == index and row['exclusive'] is True
                and [d['gpu_uuid'] for d in row['devices']] == data['gpu_uuids']
                and len(set(data['gpu_uuids'])) == 2, 'trained GPU identity differs')
        current = [d['allowed_pids'] for d in row['devices']]
        owners = current if owners is None else owners
        require(current == owners and all(len(d['allowed_pids']) == 1
                and set(d['actual_pids']) <= set(d['allowed_pids'])
                and d['unexpected_pids'] == [] for d in row['devices']),
                'trained GPU ownership differs')
    return dict(schema='rolling_trained_prefix_transaction_proof_v1', source=SOURCE,
                artifact_index_sha256=INDEX, run_id=seal['run_id'], attempt_id=seal['attempt_id'],
                result_sha256=hashlib.sha256(frozen.file(prefix+'result.json',label='result').read_bytes()).hexdigest(),
                retained_blocks=8, appended_blocks=4, following_blocks=[12,13,14,15],
                compared_layers=30, gpu_ownership_checks=37, native_sources_verified=True,
                claim_scope='trained handoff correctness only; no single-GPU quality or speed claim')


def verify_fast(run, *, repository_root):
    """Bind the separately sealed host-metadata composition, never a caller's claim."""
    from .baseline_shard_report import source_manifest
    from .correction_state import LatentHistory, LatentSegment
    from .krea_chunk_backend import validate_target_forward_trace, validate_native_correction_state
    from .main_table_merge import _load_frozen_index
    from .rolling_host_cursors import validate_evidence

    repo, run = Path(repository_root).resolve(strict=True), Path(run).resolve(strict=True)
    frozen, seal = _load_frozen_index(run)
    require(seal['git_sha'] == FAST_SOURCE and seal['artifact_index_sha256'] == FAST_INDEX,
            'original fast-metadata source/index differs')
    frozen.verify_all()
    prefix = 'attempts/'+seal['attempt_id']+'/'

    def read(relative):
        return json.loads(frozen.file(relative,label='fast metadata proof').read_text())

    manifest = source_manifest(repo, FAST_SOURCE, FAST_MANIFEST)
    envelope = read('resolved-manifest.json')
    require(envelope['manifest'] == manifest and all(envelope[k] == seal[k]
            for k in ('run_id','attempt_id','spec_hash')), 'fast-metadata invocation differs')
    for item in manifest['spec']['inputs']:
        # Drivers and readers can add checks; native computation remains bound.
        if item['target'] in ('src/cfexp/rolling_transaction_gate.py',
                              'src/cfexp/rolling_transaction_proof.py'):
            continue
        path = repo/item['target']
        require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],
                'fast-metadata native implementation changed: '+item['target'])
    data = read(prefix+'result.json')
    validate_comparisons(data)
    require(hashlib.sha256(frozen.file(prefix+'result.json',label='result').read_bytes()).hexdigest()
            == FAST_RESULT, 'original fast-metadata result differs')
    original = verify(manifest['spec']['command']['env']['CFEXP_ROLLING_TRAINED_PREFIX_GATE_RUN'],
                      repository_root=repo)
    require(data['trained_prerequisite'] == original and data['execution_metadata']
            == dict(host_kv_cursors=True,host_window=True), 'fast-metadata prerequisite or mode differs')
    history = LatentHistory.from_dict(data['accepted_history'])
    history.verify_files(run/prefix)
    accepted = LatentHistory(history.trajectory_id,history.prompt_id,history.trajectory_seed)
    expected = [('target',0), *[('draft',i) for i in range(1,9)],
                *[('target',i) for i in range(8,12)], *[('draft',i) for i in range(12,16)]]
    require([(r['role'],r['chunk']) for r in data['requests']]==expected,
            'fast-metadata request coverage differs')
    sessions = set()
    for row in data['requests']:
        segment = LatentSegment.from_dict(row['segment']);segment.verify_file(run/prefix)
        advanced = accepted.append(segment)
        diagnostics = row['diagnostics']
        if row['role']=='target':
            validate_target_forward_trace(diagnostics['forward_trace'],diagnostics['denoising_schedule'],
                history_frames=3*row['chunk'],expected_schedule=(1000.,750.,500.,250.))
            validate_native_correction_state(diagnostics['native_correction_state'],chunk_index=row['chunk'])
            require(diagnostics['explicit_initial_noise_verified'] is True,'fast Target noise differs')
            sessions.add(diagnostics['native_correction_state']['native_session_id'])
        else:
            require(diagnostics['metadata_execution']==data['execution_metadata']
                    and read(prefix+f"parity/draft-{row['chunk']:02d}.json")==row,
                    'fast-metadata Draft receipt differs')
            cursor = diagnostics['host_cursor_execution'];validate_evidence(cursor)
            require(all(r['global_end_index']==(row['chunk']+1)*4680
                        and r['local_end_index']==min((row['chunk']+1)*4680,37440)
                        for r in cursor['layers']), 'fast-metadata cursor position differs')
        if row['role']!='draft' or row['chunk']!=8:
            accepted = advanced
    require(accepted==history and len(sessions)==1,'fast-metadata accepted history differs')
    names=sorted(n for n in frozen.entries if n.startswith(prefix+'provenance/gpu-ownership-'))
    require(names==[prefix+f'provenance/gpu-ownership-{i:03d}.json' for i in range(37)],
            'fast-metadata ownership coverage differs')
    for i,name in enumerate(names):
        row=read(name)
        require(row['sequence']==i and row['exclusive'] is True and len(row['devices'])==2
                and all(len(d['allowed_pids'])==1 and not d['unexpected_pids']
                        and set(d['actual_pids'])<=set(d['allowed_pids']) for d in row['devices']),
                'fast-metadata GPU ownership differs')
    return dict(schema='rolling_prefix_execution_proof_v1',source=FAST_SOURCE,
                artifact_index_sha256=FAST_INDEX,result_sha256=FAST_RESULT,
                run_id=seal['run_id'],attempt_id=seal['attempt_id'],native_sources_verified=True,
                execution_metadata=data['execution_metadata'],following_blocks=[12,13,14,15],
                compared_layers=30,original_proof=original,
                claim_scope='trained native-prefix correctness; not a performance measurement')
