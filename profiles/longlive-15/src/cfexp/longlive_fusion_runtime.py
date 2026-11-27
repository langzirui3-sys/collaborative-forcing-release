"""Admit the sealed native LongLive policy kernel into a normal stream worker.

Admission and source checks run before model loading, outside measured video
generation. No profiler, oracle, extra model, or diagnostic RPC is installed by
this factory. The ordinary result serializer retains its execution evidence.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

ENV = 'CFEXP_LONGLIVE_FUSESTEP'
GATE_ENV = 'CFEXP_LONGLIVE_FUSESTEP_GATE_RUN'
ANCESTOR = 'b2a37e4a9a9ce1bf3717537f985f090c949f696d'
BRIDGES = {
    'longlive_fusion_handoff': 'longlive_joint_handoff',
    'longlive_fusion_executor': 'longlive_joint_executor',
    'longlive_fusion_capture': 'longlive_joint_policy_capture',
}
DIAGNOSTIC_FLAGS = (
    'CFEXP_LONGLIVE_JOINT_HANDOFF_DIAGNOSTIC',
    'CFEXP_LONGLIVE_JOINT_POLICY_DIAGNOSTIC',
    'CFEXP_LONGLIVE_PREFIX_DIAGNOSTIC',
    'CFEXP_LONGLIVE_LATE_HANDOFF_DIAGNOSTIC',
)
UNPROVEN_OPTIONS = (
    'CFEXP_LONGLIVE_HOST_CURSORS', 'CFEXP_LONGLIVE_CPU_HISTORY_BINDING',
    'CFEXP_CPU_HISTORY_BINDING', 'CFEXP_KV_WITNESS_INDEX_REUSE',
    'CFEXP_BATCH_KV_WITNESS', 'CFEXP_RANK_INDEX_REUSE',
    'CFEXP_NATIVE_ROPE_COEFFICIENT_REUSE', 'CFEXP_NATIVE_FULL_LENGTH_ATTENTION',
    'CFEXP_LONGLIVE_ACCEPTED_PREFILL', 'CFEXP_ACCEPTED_TARGET_PREFILL',
    'CFEXP_DRAFT_KV_FORK',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_flags(env=None):
    env = os.environ if env is None else env
    require(env.get(ENV) == '1', 'LongLive FuseStep requires its explicit normal-worker opt-in')
    require(all(env.get(k, '0') == '0' for k in DIAGNOSTIC_FLAGS + UNPROVEN_OPTIONS),
            'LongLive FuseStep cannot mix unverified diagnostic/cache execution options')
    require(isinstance(env.get(GATE_ENV), str) and bool(env[GATE_ENV]),
            'LongLive FuseStep requires the original sealed full-policy gate')


def _normalized(source, *, production, module):
    """Normalize only a top docstring, known imports and one ENV assignment.

    No function, assertion, numerical constant, message, or statement is dropped.
    A change to tensor computation or validation therefore fails the bridge.
    """
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(
            tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        tree.body.pop(0)
    if production:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in BRIDGES:
                node.module = BRIDGES[node.module]
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'ENV'):
                require(isinstance(node.value, ast.Constant) and node.value.value == ENV,
                        'production kernel changed its opt-in namespace')
                old = {'longlive_fusion_handoff': 'CFEXP_LONGLIVE_JOINT_HANDOFF_DIAGNOSTIC',
                       'longlive_fusion_capture': 'CFEXP_LONGLIVE_JOINT_POLICY_DIAGNOSTIC'}
                require(module in old, 'unexpected environment assignment in bridged executor')
                node.value = ast.Constant(value=old[module])
    return ast.dump(tree, include_attributes=False)


def validate_bridge(repo):
    repo = Path(repo)
    files = []
    for new, old in BRIDGES.items():
        current, original = (repo / ('src/cfexp/' + name + '.py') for name in (new, old))
        require(current.is_file() and original.is_file()
                and not current.is_symlink() and not original.is_symlink(), 'missing regular kernel source')
        require(_normalized(current.read_text(), production=True, module=new)
                == _normalized(original.read_text(), production=False, module=old),
                'LongLive production kernel differs from the frozen diagnostic computation: ' + new)
        files.append(dict(production=str(current.relative_to(repo)), production_sha256=sha(current),
                          original=str(original.relative_to(repo)), original_sha256=sha(original)))
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema='longlive_fusestep_exact_source_bridge_v1', files=files,
                normalized_bridge_sha256=digest,
                allowed_differences=['module docstring', 'three declared relative module imports',
                                     'explicit normal-worker ENV constant'])


def admit_gate(repo, run):
    """Read only; never invoke a helper that writes into a sealed producer."""
    from .main_table_merge import _load_frozen_index
    from .longlive_joint_policy_gate import validate_report, expected_forward_trace
    from .joint_gate_profile import validate_forward_records
    repo, run = Path(repo), Path(run)
    frozen, marker = _load_frozen_index(run)
    require(marker.get('git_sha') == ANCESTOR and marker.get('status') == 'SUCCEEDED'
            and marker.get('exit_code') == 0, 'wrong or unsuccessful original full-policy gate')
    frozen.verify_all()
    prefix = 'attempts/' + marker['attempt_id'] + '/'
    def load(relative):
        return json.loads(frozen.file(prefix + relative, label=relative).read_text())
    resolved = json.loads(frozen.file('resolved-manifest.json', label='gate manifest').read_text())
    manifest = resolved['manifest']
    require(manifest['metadata']['name'] == 'longlive-joint-policy-profile-single-h200-gate-ntu-v1'
            and manifest['spec']['resources']['gpus'] == 1, 'wrong original policy/physical-device gate')
    # All old source files are retained byte-for-byte. Newly added isolated
    # modules are subsequently bound by the exact AST bridge.
    for item in manifest['spec']['inputs']:
        if item['uri'].startswith('repository://'):
            path = repo / item['target']
            require(path.is_file() and not path.is_symlink() and sha(path) == item['sha256'],
                    'original full-policy source/artifact changed: ' + item['target'])
    driver = load('longlive-joint-policy-native-gate.json')
    require(driver.get('passed') is True and driver.get('physical_gpu_count') == 1
            and driver.get('real_target_history_blocks') == 12, 'incomplete original native gate')
    proof = driver['gate']
    files = proof['input_files']
    require(len(files) == 19 and files[:12] == [dict(path=x['path'], sha256=x['sha256'])
            for x in driver['target_history']], 'original Target/noise bindings differ')
    for item in files:
        require(sha(frozen.file(prefix + 'live/' + item['path'], label='gate input')) == item['sha256'],
                'original gate input differs')
    validate_report(proof, files=files)
    records = []
    directory = run / 'attempts' / marker['attempt_id'] / 'live/provenance/tracking'
    for path in directory.glob('draft-forwards-*.jsonl'):
        path = frozen.file(prefix + str(path.relative_to(run / 'attempts' / marker['attempt_id'])),
                           label='original forward records')
        records.extend(json.loads(x) for x in path.read_text().splitlines())
    samples = validate_forward_records(records, expected_forward_trace())
    profile = load('provenance/tracking/profile-summary.json')
    require(len(records) == 410 and profile['samples'] == samples
            and profile.get('executed_actual_cuda') is True
            and profile.get('original_full_oracle_passed') is True,
            'original actual trained CUDA/profile evidence differs')
    session, finish = load('provenance/tracking/session.json'), load('provenance/tracking/finished.json')
    hardware = session['hardware']['gpus']
    require(len(hardware) == 1 and hardware[0]['name'] == 'NVIDIA H200'
            and finish.get('execution_succeeded') is True, 'original hardware/tracking completion differs')
    bridge = validate_bridge(repo)
    return dict(schema='longlive_fusestep_ancestor_admission_v1', run_id=marker['run_id'],
                gate_source=ANCESTOR, artifact_index_sha256=marker['artifact_index_sha256'],
                full_policy_pairs=35, actual_draft_forwards=410, source_bridge=bridge,
                no_quality_or_video_ready_speed_claim=True)


def build_backend(raw_config):
    validate_flags()
    require(isinstance(raw_config, Mapping) and 'sdvg' not in raw_config,
            'LongLive FuseStep is not an SDVG baseline wrapper')
    repo = Path(__file__).resolve().parents[2]
    require(raw_config.get('artifact_spec_sha256')
            == sha(repo / 'experiments/erank3-longlive-policy-v1.artifacts.json'),
            'LongLive normal worker selected a different checkpoint specification')
    admission = admit_gate(repo, os.environ[GATE_ENV])
    from .longlive_policy_worker import build_backend as native, OwnedLongLivePolicyBackend
    from .longlive_fusion_capture import install
    backend = native(raw_config)
    try:
        require(type(backend) is OwnedLongLivePolicyBackend
                and backend.model_state['model_id'] == 'longlive_1.3b'
                and backend.model_state['artifact_spec_sha256'] == raw_config['artifact_spec_sha256']
                and backend.model_state['kv_local_attention_size'] == 12
                and backend.model_state['kv_sink_size'] == 3
                and backend.model_state['history_clean_update_width'] == 3,
                'normal worker did not load the admitted native LongLive model')
        install(backend)
        backend.fusestep_admission = admission
        backend.model_state['native_fusestep'] = admission
        return backend
    except BaseException:
        backend.close()
        raise


def worker_argv(argv):
    require(os.environ.get(ENV) == '1' and list(argv[1:3]) == ['-m', 'cfexp.correction_worker'],
            'LongLive fusion launcher requires its explicit standard-worker entry')
    command = list(argv)
    role = command[command.index('--role') + 1]
    require(role in ('draft', 'target'), 'unexpected fusion worker role')
    index = command.index('--backend-factory') + 1
    expected = ('cfexp.longlive_policy_worker:build_backend' if role == 'draft'
                else 'cfexp.correction_runtime:build_target_backend')
    require(command[index] == expected,
            'unexpected original fusion worker factory')
    if role == 'draft':
        command[index] = 'cfexp.longlive_fusion_runtime:build_backend'
    return command


def process_factory(argv, **kwargs):
    from .accepted_stream_worker import process_factory as ordinary
    return ordinary(worker_argv(argv), **kwargs)


def launch_runtime(**kwargs):
    from .accepted_stream_worker import process_factory as ordinary
    from .correction_runtime import launch_correction_runtime
    require(kwargs.pop('process_factory') is ordinary
            and kwargs.get('draft_model_id') == 'longlive_1.3b',
            'normal LongLive campaign must retain its original coupled stream runtime')
    return launch_correction_runtime(process_factory=process_factory, **kwargs)
