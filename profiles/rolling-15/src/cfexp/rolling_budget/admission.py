"""Fail-closed admission of the two trained CUDA gates for a video trial.

Only the diagnostic receipts are read here. Successful admission permits
full-video verification of the same kernel, not a quality or speed claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from ..main_table_merge import _load_frozen_index
from ..rolling_joint_case_gate import BUDGET_ENV, ORIGINAL_RUN, RUN_ENV
from ..rolling_joint_case_profile import validate as validate_profile
from ..rolling_transaction_proof import INDEX as ORIGINAL_INDEX, PROMPT, SOURCE as ORIGINAL_SOURCE

SOURCE = '916b59100db5d366c53474cd41aec18fb5be7df9'
MANIFESTS = {
    2: 'rolling-joint-b2-case-single-h200-gate-ntu-v1',
    4: 'rolling-joint-original-case-single-h200-gate-ntu-v1',
}
PINNED_GATES = {
    2: ('rolling-joint-b2-case-single-h200-gate-ntu-v1-20261122T220520Z-916b5910-e9c36a91-5e747d11',
        '5965178a8df063cc044026542b9e718265408aea6d0ae4093d5d46813ecb41cd'),
    4: ('rolling-joint-original-case-single-h200-gate-ntu-v1-20261122T224012Z-916b5910-8ec6c5ba-39f926a1',
        '421f0a918010de22b0326934fe9be09b0a856f856bcd59cd1bcb1061be673336'),
}
PHYSICAL_CALLS = {2: 225, 4: 207}
CHECKS = ('independent_native_latent_exact', 'layer15_hidden_exact',
          'full_spectrum_and_rank_exact', 'accepted_history_witness_exact',
          'private_rng_exact', 'lookahead_and_drain_exact',
          'retained_cache_storage', 'ambient_rng_preserved')
REQUIRED_SOURCES = frozenset({
    'src/cfexp/rolling_forcing_native.py', 'src/cfexp/rolling_forcing_stream.py',
    'src/cfexp/rolling_forcing_session.py', 'src/cfexp/rolling_joint_handoff.py',
    'src/cfexp/rolling_joint_stream.py', 'src/cfexp/rolling_committed_prefix.py',
    'src/cfexp/rolling_prefix_transaction.py', 'src/cfexp/rolling_forcing_observation.py',
    'src/cfexp/rolling_joint_case_oracle.py', 'src/cfexp/rolling_joint_case_profile.py',
    'src/cfexp/rolling_joint_case_gate.py',
    'experiments/erank3-rolling-native-v1.artifacts.json',
    'experiments/insight2-krea14b.artifacts.json',
})


def require(test, reason):
    if not test:
        raise ValueError(reason)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_bindings(manifest, repo):
    result = {}
    root = Path(repo).resolve(strict=True)
    for entry in manifest['spec']['inputs']:
        name = entry['target']
        posix = PurePosixPath(name)
        require(str(posix) == name and not posix.is_absolute()
                and '..' not in posix.parts and name not in result
                and entry['uri'] == 'repository://'+name, 'invalid source binding')
        path = root/name
        require(path.is_file() and path.resolve(strict=True) == path
                and digest(path) == entry['sha256'], 'trained gate source differs: '+name)
        result[name] = entry['sha256']
    require(REQUIRED_SOURCES <= result.keys(), 'trained gate lacks native source coverage')
    return result


def validate_case(case, width):
    resume = 8+width
    require(case['schema'] == 'rolling_joint_original_case_oracle_v1'
            and case['passed'] is True and case['executed_device_type'] == 'cuda'
            and case['prompt'] == PROMPT and case['seed'] == 20260911
            and case['retained_blocks'] == 8 and case['accepted_target_blocks'] == list(range(8,resume))
            and case['following_blocks'] == list(range(resume,16))
            and case['original_resumed_latents_required'] is (width == 4)
            and case['retained_prefix_oracle']['comparison']['equal'] is True
            and case['prologue'] == [dict(chunk=i, original_latent_exact=True) for i in range(1,8)]
            and case['no_quality_or_speedup_claim'] is True,
            'trained case horizon, prologue, or scope differs')
    require([row['chunk'] for row in case['following']] == list(range(resume,16)),
            'trained case has incomplete continuation')
    for row in case['following']:
        cache = row['full_cache_equality']
        require(row['original_latent_exact'] is (True if width == 4 else None)
                and all(row[key] is True for key in CHECKS)
                and cache['equal'] is True and [x['layer'] for x in cache['layers']] == list(range(30))
                and all(x['global_end_index'] == (row['chunk']+1)*4680
                        and x['local_end_index'] == 37440 for x in cache['layers']),
                'trained continuation/cache equality is incomplete')


def admit_one(repo, path, *, width, pinned_index):
    require(ORIGINAL_RUN is not None, 'the original Rolling CUDA admission archive is required')
    require(type(width) is int and width in MANIFESTS, 'unknown Rolling width')
    supplied = Path(path)
    require(supplied.is_absolute() and supplied.resolve(strict=True) == supplied
            and not supplied.is_symlink(), 'gate path must be an absolute original directory')
    frozen, marker = _load_frozen_index(supplied)
    require(marker['git_sha'] == SOURCE and marker['artifact_index_sha256'] == pinned_index
            and marker['run_id'] == supplied.name, 'trained budget gate identity differs')
    def read(rel):
        return json.loads(frozen.file(rel,label='Rolling budget gate').read_text())
    env = read('resolved-manifest.json')
    require(all(env[k] == marker[k] for k in ('run_id','attempt_id','spec_hash')),
            'trained invocation identity differs')
    manifest = env['manifest']; spec = manifest['spec']; flags = spec['command']['env']
    require(manifest['metadata']['name'] == MANIFESTS[width]
            and spec['resources']['gpus'] == 1
            and spec['command']['argv'] == ['python','-m','cfexp.rolling_joint_case_gate']
            and flags.get(RUN_ENV) == ORIGINAL_RUN and flags.get(BUDGET_ENV,'4') == str(width)
            and flags.get('CFEXP_ROLLING_HOST_KV_CURSORS','0') == '0'
            and flags.get('CFEXP_ROLLING_HOST_WINDOW','0') == '0'
            and spec['parameters']['original_case_source'] == ORIGINAL_SOURCE
            and spec['parameters']['original_case_index'] == ORIGINAL_INDEX,
            'trained gate changed the native budget configuration')
    sources = source_bindings(manifest, repo)
    prefix = 'attempts/'+marker['attempt_id']+'/'
    result = read(prefix+'result.json')
    profile = read(prefix+'provenance/tracking/profile-summary.json')
    track = read(prefix+'provenance/tracking/session.json')
    finish = read(prefix+'provenance/tracking/finished.json')
    gpus = result['hardware']['gpus']
    require(result['schema'] == 'rolling_joint_trained_case_gate_v1'
            and result['source'] == SOURCE and result['passed'] is True
            and result['target_suffix_blocks'] == width
            and result['original_resumed_latents_required'] is (width == 4)
            and result['actual_post_warmup_cuda_profiles'] is True
            and result['original_target_outputs_not_regenerated'] is True
            and result['no_quality_or_end_to_end_speed_claim'] is True
            and len(gpus) == 1 and gpus[0]['name'] == 'NVIDIA H200'
            and type(gpus[0]['uuid']) is str and gpus[0]['uuid'].startswith('GPU-')
            and track['config']['target_suffix_blocks'] == width
            and finish['execution_succeeded'] is True
            and track['mode'] == finish['mode'] in ('online','offline'),
            'trained result is missing physical device or execution evidence')
    repetitions = result['repetitions']
    require(len(repetitions) == 3, 'expected three actual trained repetitions')
    for i, case in enumerate(repetitions):
        require(case == read(prefix+f'parity/repetition-{i}.json')
                and case == repetitions[0], 'trained repetitions disagree')
        validate_case(case, width)
    records = [json.loads(line) for line in frozen.file(
        prefix+'provenance/tracking/physical-forwards.jsonl',label='physical forwards').read_text().splitlines()]
    require(len(records) == result['physical_forward_count']
            == profile['physical_forward_count'] == PHYSICAL_CALLS[width]
            and profile['executed_actual_cuda'] is True
            and profile['samples'] == validate_profile(records, target_suffix_blocks=width),
            'trained post-warmup profile coverage differs')
    return dict(run_id=marker['run_id'], index=pinned_index, source=SOURCE, width=width,
                uuid=gpus[0]['uuid'], case=result['original_case'], source_bindings=sources,
                input_tensor_sha256=repetitions[0]['input_tensor_sha256'],
                warmup_profiles_verified=True, tracking_mode=finish['mode'])


def admit_pair(repo, runs: Mapping[int,tuple[Path,str]]):
    require(set(runs) == {2,4}, 'both pinned budget gates are required')
    reports = {width: admit_one(repo, path, width=width, pinned_index=index)
               for width,(path,index) in sorted(runs.items())}
    for field in ('source','uuid','case','source_bindings','input_tensor_sha256'):
        require(reports[2][field] == reports[4][field], 'budget gate pairing differs: '+field)
    return dict(schema='rolling_matched_budget_kernel_admission_v1', gates=reports,
                one_matched_device_and_source=True, no_video_quality_or_latency_claim=True)

def admit_original_pair(repo, run_root):
    """Admit exactly the predetermined pair, never pick a replacement by score."""
    root = Path(run_root).resolve(strict=True)
    return admit_pair(repo, {width: (root/name, index)
                             for width,(name,index) in PINNED_GATES.items()})
