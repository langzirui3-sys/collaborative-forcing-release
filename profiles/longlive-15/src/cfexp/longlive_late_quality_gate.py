"""Bind a late Target trial to its separately sealed CUDA gate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .longlive_late_gate import SOURCE_PATHS
from .main_table_merge import _load_frozen_index
from .baseline_shard_report import _published,source_manifest


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_report(report,spec):
    from .longlive_late_handoff import POLICIES, validate_trace
    from .longlive_late_worker import validate_proposal
    require(report.get('schema')=='longlive_late_native_gate_v1'
        and report.get('passed') is True and report.get('formal_quality_result') is False
        and report.get('no_end_to_end_speed_claim') is True
        and report.get('changed_sampling_policy') is True
        and report.get('source_sha256')==spec['source_sha256']
        and report.get('intervention_chunks')==list(range(1,10))
        and report.get('chunk_count')==16 and 'cleanup_error' not in report,
        'failed or incompatible late Target gate')
    require(report.get('policies')=={k:dict(draft_updates=v[0],target_timestep=v[1])
        for k,v in POLICIES.items()}, 'late gate policy coverage changed')
    rows=report.get('rows',[]); following=report.get('following_draft',[])
    require([r.get('chunk_index') for r in rows]==list(range(1,10))
        and [r.get('chunk_index') for r in following]==list(range(10,16)),
        'late gate omitted interventions or continuation')
    require(report['root']['denoising_schedule']==[1000.,750.,500.,250.],
        'late gate did not retain the full native Target root')
    def exact(proof, cache=False):
        require(proof.get('latent_exact') is True
            and proof.get('maximum_absolute_difference')==0
            and (not cache or proof.get('cache_differences')==[]),
            'late gate native tensor or complete KV parity failed')
    for n,row in enumerate(rows,1):
        policy=tuple(POLICIES)[(n-1)%len(POLICIES)]
        require(row['policy']==policy, 'late gate did not cycle all policies')
        proposal=row['proposal']; validate_proposal(proposal)
        require(proposal['input_binding']['chunk_index']==n
            and proposal['input_binding']['policy']==policy,
            'late gate used another proposal position or policy')
        exact(proposal['result']['proposal_oracle'],cache=True)
        target=row['target']; require(target['proposal']==proposal,
            'late gate Target belongs to another Draft proposal')
        tv=target['result']; tau=POLICIES[policy][1]
        require(tv['denoising_schedule']==[tau]
            and tv['completed_four_step_target_chunk'] is False
            and tv['context_frames']==3 and tv['accepted_history_frames']==n*3
            and tv['forward_trace']==[dict(timestep=t,current_start=4680,
                batch_timesteps_equal=True) for t in (0.,tau)],
            'late Target changed its native context or compute')
        exact(tv['target_oracle'])
        committed=row['commit']; cv=committed['result']
        require(committed['accepted_latent_sha256']==tv['latent']['sha256']
            and cv['draft_clean_forwards']==1 and cv['history_replay_chunks']==0
            and cv['committed_kv_width']==3 and cv['committed_chunks']==n+1
            and cv['branch_snapshot_bytes']==0,
            'late gate did not commit the accepted Target through one native B3 update')
        validate_trace(cv['forward_trace'],[0.],start=n*4680)
        exact(cv['commit_oracle'],cache=True)
    for row in following:
        exact(row);require(row.get('rank_exact') is True,
            'late handoff changed ordinary native Draft continuation')
    return True


def verify_gate(plan_path,repository):
    repository=Path(repository).resolve(strict=True);plan_path=Path(plan_path)
    spec=json.loads((repository/'experiments/longlive-late-quality-gate-v1.json').read_text())
    require(spec.get('schema')=='longlive_late_quality_approval_v1','invalid late Target approval schema')
    require(set(spec['source_sha256'])==set(SOURCE_PATHS),'incomplete late Target source approval')
    for path,digest in spec['source_sha256'].items():
        require(hashlib.sha256((repository/path).read_bytes()).hexdigest()==digest,
                'late Target implementation changed after its candidate CUDA gate')
    require(plan_path.is_file() and not plan_path.is_symlink()
        and hashlib.sha256(plan_path.read_bytes()).hexdigest()==spec['queue_plan_sha256'],
        'late Target gate requires its original immutable queue plan')
    plan=json.loads(plan_path.read_text());state=json.loads((Path(plan['ops'])/'state.json').read_text())
    require(state['plan_sha256']==spec['queue_plan_sha256'],'late Target gate queue/state binding differs')
    tasks=[t for t in plan['tasks'] if t['id']==spec['queue_task_id']]
    require(len(tasks)==1 and tasks[0]['commit']==spec['gate_commit'],'wrong late Target gate source task')
    task=tasks[0];row=state['tasks'][task['id']]
    published = row['status'] == 'succeeded'
    pinned_seal = spec.get('publication_pending_seal')
    if published:
        run_id = Path(row['run_dir']).name
    else:
        require(pinned_seal is not None and row['status'] == 'failed'
            and row.get('exit_code') == 2
            and row.get('reason') == 'publication reconciliation exhausted; original run preserved'
            and row.get('launch_id') == pinned_seal['launch_id'],
            'late Target CUDA gate has not completed successfully')
        run_id = pinned_seal['run_id']
    run = Path(plan['root']) / 'runs' / run_id
    require(not run.is_symlink() and run.resolve().parent == (Path(plan['root'])/'runs').resolve(),
            'late Target gate escaped its formal run root')
    frozen,marker = _load_frozen_index(run)
    if published:
        require(run == Path(row['run_dir']), 'published late gate run path differs')
        _published(row,run,marker)
    else:
        require(hashlib.sha256((run/'_SUCCESS.json').read_bytes()).hexdigest() == pinned_seal['marker_sha256']
            and marker['artifact_index_sha256'] == pinned_seal['artifact_index_sha256']
            and marker['attempt_id'] == pinned_seal['attempt_id'],
            'local late Target gate differs from the approved original seal')
    require(marker['status']=='SUCCEEDED' and marker['git_sha']==spec['gate_commit'],
            'wrong or failed sealed late Target gate')
    envelope=json.loads(frozen.file('resolved-manifest.json',label='late Target manifest').read_text())
    require(envelope['manifest']==source_manifest(repository,spec['gate_commit'],task['manifest'])
        and all(envelope[k]==marker[k] for k in ('run_id','attempt_id','spec_hash')),
        'late Target gate invocation differs from the published source')
    terminal=json.loads(frozen.file('run.json',label='late Target terminal run').read_text())
    require(terminal['execution_status']=='SUCCEEDED' and terminal['exit_code']==0
        and terminal['latest_attempt_id']==marker['attempt_id']
        and terminal['run_id']==marker['run_id']==run_id,'late Target gate terminal state differs')
    path=frozen.file('attempts/'+marker['attempt_id']+'/longlive-late-native-gate.json',label='late Target report')
    report_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if not published:
        require(report_sha256 == pinned_seal['report_sha256'],
                'local late Target gate report differs from the approved report')
    report=json.loads(path.read_text());validate_report(report,spec)
    return dict(run_id=marker['run_id'],git_sha=marker['git_sha'],attempt_id=marker['attempt_id'],
        artifact_index_sha256=marker['artifact_index_sha256'],report_sha256=report_sha256, publication_verified=published,
        queue_plan_sha256=spec['queue_plan_sha256'],source_sha256=spec['source_sha256'])
