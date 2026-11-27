"""Quality experiments require sealed real-model parity for this backend."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .main_table_merge import _load_frozen_index
from .target_prefix import validate_draft_suffix_evidence


def _load_spec(repository, native_model=None):
    if native_model not in (None, 'longlive'):
        raise ValueError('unsupported Target-prefix gate model')
    name = ('longlive-target-prefix-quality-gate-v1.json' if native_model == 'longlive'
            else 'target-prefix-quality-gate-v1.json')
    return json.loads((repository/'experiments'/name).read_text())


def verify_gate(root, repository, *, native_model=None):
    if root is None:
        raise ValueError('Target-prefix quality requires an explicit sealed gate run')
    spec=_load_spec(repository, native_model)
    if spec.get('revalidation_required', False):
        raise ValueError('prefix backend requires a new sealed real-model gate')
    for name,digest in spec['source_sha256'].items():
        if hashlib.sha256((repository/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('prefix backend changed since the real-model gate')
    if root.is_file():
        if hashlib.sha256(root.read_bytes()).hexdigest()!=spec['queue_plan_sha256']:
            raise ValueError('Target-prefix gate queue plan changed')
        state=json.loads((root.parent/'state.json').read_text())
        task_id=spec.get('queue_task_id','tpfx2-native')
        if not isinstance(task_id,str) or task_id not in state.get('tasks',{}):
            raise ValueError('Target-prefix gate queue task missing')
        task=state['tasks'][task_id]
        if state['plan_sha256']!=spec['queue_plan_sha256'] or task['status']!='succeeded':
            raise ValueError('Target-prefix gate queue has not succeeded')
        root=Path(task['run_dir'])
    frozen,marker=_load_frozen_index(root)
    if marker['git_sha']!=spec['gate_commit'] or marker['status']!='SUCCEEDED':
        raise ValueError('wrong or failed Target-prefix gate')
    if native_model == 'longlive' and (marker['run_id'] != spec['gate_run_id']
            or marker['artifact_index_sha256'] != spec['gate_index_sha256']):
        raise ValueError('LongLive Target-prefix gate seal differs')
    relative='attempts/'+marker['attempt_id']+'/target-prefix-gate.json'
    path=frozen.file(relative,label='Target-prefix gate');report=json.loads(path.read_text())
    if native_model == 'longlive' and (report.get('draft_model_id') != 'longlive_1.3b'
            or hashlib.sha256(path.read_bytes()).hexdigest() != spec['gate_report_sha256']):
        raise ValueError('LongLive Target-prefix gate report differs')
    if (report.get('schema')!='target_prefix_rpc_gate_v2' or report.get('passed') is not True
            or report.get('prefix_chunks')!=list(range(1,9)) or report.get('chunk_count')!=16
            or len(report.get('rows',[]))!=15 or len(report.get('following_draft',[]))!=7
            or any(not r['latent_exact'] or not r['rank_exact'] for r in report['following_draft'])
            or 'cleanup_error' in report):
        raise ValueError('incomplete Target-prefix parity proof')
    if ([r['chunk_index'] for r in report['rows']]!=list(range(1,16))
            or [r['chunk_index'] for r in report['following_draft']]!=list(range(9,16))):
        raise ValueError('Target-prefix gate chunk coverage differs')
    native_schedule=None
    for i,row in enumerate(report['rows'],1):
        if row['prefix'] is not (i<=8) or row['reuse']['replayed_chunks']!=(1 if i==1 else 0):
            raise ValueError('Target-prefix gate cache proof differs')
        if i<=8 and (row['target_oracle']['first_prediction_exact'] is not True
                or row['draft_oracle']['latent_exact'] is not True
                or row['draft_oracle']['cache_differences']):
            raise ValueError('Target-prefix oracle mismatch')
        if i<=8:
            execution=row['native_execution'];validate_draft_suffix_evidence(execution)
            if native_model == 'longlive' and execution['draft_native_schedule'] != [1000.,937.5,833.3333129882812,625.]:
                raise ValueError('LongLive gate did not use its native warped schedule')
            if native_schedule is None:native_schedule=execution['draft_native_schedule']
            if execution['draft_native_schedule']!=native_schedule:
                raise ValueError('Target-prefix native schedule changed across chunks')
    return {'run_id':marker['run_id'],'git_sha':marker['git_sha'],
            'artifact_index_sha256':marker['artifact_index_sha256'],
            'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_sha256':spec['source_sha256']}
