"""Fail closed until the separately sealed L625 CUDA gate has passed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .longlive_low_noise import validate_evidence
from .longlive_low_noise_gate import SOURCE_PATHS
from .main_table_merge import _load_frozen_index
from .baseline_shard_report import _published,source_manifest


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_report(report,spec):
    require(report.get('schema')=='longlive_low_noise_rpc_gate_v1'
        and report.get('passed') is True and report.get('formal_quality_result') is False
        and report.get('no_end_to_end_speed_claim') is True
        and report.get('draft_model_id')=='longlive_1.3b'
        and report.get('source_sha256')==spec['source_sha256']
        and report.get('prefix_chunks')==list(range(1,9)) and report.get('chunk_count')==16
        and 'cleanup_error' not in report,'failed or incompatible L625 gate')
    rows=report.get('rows',[]);following=report.get('following_draft',[])
    require([r.get('chunk_index') for r in rows]==list(range(1,16))
        and [r.get('chunk_index') for r in following]==list(range(9,16)),
        'L625 gate omitted an intervention or following continuation')
    for i,row in enumerate(rows,1):
        require(type(row['chunk_index']) is int and row['prefix'] is (i<=8)
            and row['reuse']['replayed_chunks']==(1 if i==1 else 0),
            'L625 gate changed the accepted-history reuse policy')
        if i<=8:
            require(row['target_oracle']['first_prediction_exact'] is True
                and row['draft_oracle']['latent_exact'] is True
                and row['draft_oracle']['maximum_absolute_difference']==0
                and row['draft_oracle']['cache_differences']==[],
                'L625 native Target prediction, latent or complete KV differs from the oracle')
            x=row['native_execution'];validate_evidence(x)
            require(x['committed_chunks']==i+1 and x['tokens_per_chunk']==4680,
                    'L625 gate did not use the native 832x480 B3 geometry')
    require(all(r['latent_exact'] is True and r['rank_exact'] is True
                and r['maximum_absolute_difference']==0 for r in following),
            'ordinary Draft continuation diverged after L625')
    return True


def verify_gate(plan_path,repository):
    repository=Path(repository).resolve(strict=True);plan_path=Path(plan_path)
    spec=json.loads((repository/'experiments/longlive-low-noise-quality-gate-v1.json').read_text())
    require(spec.get('schema')=='longlive_low_noise_quality_approval_v1','invalid L625 approval schema')
    require(set(spec['source_sha256'])==set(SOURCE_PATHS),'incomplete L625 source approval')
    for path,digest in spec['source_sha256'].items():
        require(hashlib.sha256((repository/path).read_bytes()).hexdigest()==digest,
                'L625 implementation changed after its candidate CUDA gate')
    require(plan_path.is_file() and not plan_path.is_symlink()
        and hashlib.sha256(plan_path.read_bytes()).hexdigest()==spec['queue_plan_sha256'],
        'L625 gate requires its original immutable queue plan')
    plan=json.loads(plan_path.read_text());state=json.loads((Path(plan['ops'])/'state.json').read_text())
    require(state['plan_sha256']==spec['queue_plan_sha256'],'L625 gate queue/state binding differs')
    tasks=[t for t in plan['tasks'] if t['id']==spec['queue_task_id']]
    require(len(tasks)==1 and tasks[0]['commit']==spec['gate_commit'],'wrong L625 gate source task')
    task=tasks[0];row=state['tasks'][task['id']]
    require(row['status']=='succeeded','L625 CUDA gate has not completed successfully')
    run=Path(row['run_dir'])
    require(not run.is_symlink() and run.resolve().parent==(Path(plan['root'])/'runs').resolve(),
            'L625 gate escaped its formal run root')
    frozen,marker=_load_frozen_index(run);frozen.verify_all();_published(row,run,marker)
    require(marker['status']=='SUCCEEDED' and marker['git_sha']==spec['gate_commit'],
            'wrong or failed sealed L625 gate')
    envelope=json.loads(frozen.file('resolved-manifest.json',label='L625 manifest').read_text())
    require(envelope['manifest']==source_manifest(repository,spec['gate_commit'],task['manifest'])
        and all(envelope[k]==marker[k] for k in ('run_id','attempt_id','spec_hash')),
        'L625 gate invocation differs from the published source')
    terminal=json.loads(frozen.file('run.json',label='L625 terminal run').read_text())
    require(terminal['execution_status']=='SUCCEEDED' and terminal['exit_code']==0
        and terminal['latest_attempt_id']==marker['attempt_id'],'L625 gate terminal state differs')
    path=frozen.file('attempts/'+marker['attempt_id']+'/longlive-low-noise-gate.json',label='L625 report')
    report=json.loads(path.read_text());validate_report(report,spec)
    return dict(run_id=marker['run_id'],git_sha=marker['git_sha'],attempt_id=marker['attempt_id'],
        artifact_index_sha256=marker['artifact_index_sha256'],report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        queue_plan_sha256=spec['queue_plan_sha256'],source_sha256=spec['source_sha256'])
