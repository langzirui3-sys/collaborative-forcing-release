"""Audit the complete SF128x3 method and matched one-H200 Target controls.

Read original seals only. Recompute causal routing from the original C10 CDF
and replay producer-side cache/forward audits without executing either model.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from . import native_ready_report as audit
from . import paper128_reference_report as reference
from . import paper_population as population
from .accepted_stream_report import _single_gpu_render_row
from .baseline_shard_report import source_manifest
from .branch_gate_probe import collect_trajectory
from .calibration_receipt import load_sealed_calibration_ranks, CALIBRATION_RECEIPT_MANIFEST
from .correction_policy import causal_erank_sudden_drop
from .erank_acceptance_pilot import (split_transfer_options, verify_native_cache,
                                   observation_horizon_for_arm)
from .insight2_campaign import PendingSequentialRollout, _generated_from_dict
from .main_table_merge import _load_frozen_index, _load_json, _require, _sha256_file
from .rank_burst import RankBurstPolicy, validate_burst_schedule, verify_burst_reuse
from .sf_joint_handoff import verify_rollout
from .sf_target_two_step import verify_paper30_execution, validate_correction_trace
from .krea_chunk_backend import MAIN_TABLE_FOUR_STEP_SCHEDULE

MATRIX='experiments/paper128-sf-ours-v1.protocol.json'
MATRIX_SHA='fcfe18b422f7c4538392df9cf1c8a1c7354477dfd92523c0a1eca880e1d9c170'
# These diagnostics/readers are not imported by the SF production entrypoints.
# Any difference in the generation/renderer/runtime files still fails closed.
NON_EXECUTED_SOURCE_DIFFERENCES={
    'src/cfexp/longlive_joint_gate.py','src/cfexp/longlive_joint_oracle.py',
    'src/cfexp/longlive_joint_trace.py','src/cfexp/paper128_reference_report.py',
    'src/cfexp/sf_handoff_continuity.py','src/cfexp/sf_handoff_legacy_executor.py'}


def matrix(repo):
    p=Path(repo)/MATRIX
    _require(not p.is_symlink() and _sha256_file(p)[0]==MATRIX_SHA,'frozen SF method matrix differs')
    return _load_json(p,label='method matrix')


def normalized(value):
    return json.loads(json.dumps(value,allow_nan=False,sort_keys=True))


def source(run, *, repo, seconds):
    requested=Path(run);_require(not requested.is_symlink(),'method run is a symlink')
    run=requested.resolve(strict=True);frozen,marker=_load_frozen_index(run)
    terminal=_load_json(frozen.file('run.json',label='terminal method'),label='terminal method')
    _require(terminal.get('execution_status')=='SUCCEEDED' and terminal.get('exit_code')==0
        and terminal.get('latest_attempt_id')==marker['attempt_id']
        and terminal.get('run_id')==marker['run_id']==run.name,'method execution has not sealed success')
    envelope=_load_json(frozen.file('resolved-manifest.json',label='method manifest'),label='method manifest')
    invocation=envelope['manifest'];relative='experiments/'+invocation['metadata']['name']+'.manifest.json'
    selected=[x for x in matrix(repo)['runs'] if x['manifest']==relative]
    _require(len(selected)==1 and selected[0]['nominal_seconds']==seconds
        and all(envelope.get(k)==marker[k] for k in ('run_id','attempt_id','spec_hash'))
        and invocation==source_manifest(repo,marker['git_sha'],relative),'method source/invocation/horizon differs')
    prefix='attempts/'+marker['attempt_id']+'/'
    result=_load_json(frozen.file(prefix+'result.json',label='method result'),label='method result')
    _require(result.get('status')=='succeeded' and result.get('code_revision')==marker['git_sha'],
             'method status/revision differs')
    return SimpleNamespace(run=run,frozen=frozen,marker=marker,prefix=prefix,result=result,
                           manifest=invocation,definition=selected[0])


def selection(src, *, repo, seconds):
    d=src.definition;spec=src.manifest['spec'];env=spec['command']['env'];argv=spec['command']['argv']
    policy=matrix(repo)['policies'][str(seconds)];arm=policy['arm']
    _require(spec['resources']['gpus']==1 and env.get(population.ENV)=='1'
        and env.get(population.SEED_ENV)==str(d['generation_seed'])
        and env.get('CFEXP_PAPER_PROMPT_SHARD_COUNT')=='4'
        and env.get('CFEXP_PAPER_PROMPT_SHARD_INDEX')==str(d['logical_matched_gpu_slot'])
        and env.get('CFEXP_SF_SINGLE_GPU')==env.get('CFEXP_DRAFT_KV_REUSE')=='1'
        and env.get('CFEXP_SF_JOINT_HANDOFF')=='1' and env.get('CFEXP_STAGE_TIMING')=='0'
        and '--smoke-prompt-id' not in argv and '--defer-video-delivery' not in argv
        and argv[2]=='cfexp.erank_acceptance_pilot' and '--sealed-calibration-ranks' in argv
        and argv[argv.index('--duration-seconds')+1]==str(seconds)
        and argv[argv.index('--arm')+1]==arm,'method is not its frozen complete one-card shard')
    with patch.dict(os.environ,env,clear=True):prompts,binding=population.load(repo,model_pair='krea_self_forcing')
    result=src.result;profile=reference.duration_profile(seconds)
    duration=dict(profile.metadata(),experiment_id=f'committed-rank-burst-main24-{seconds}s-v1',
        calibration_policy='explicit_frozen_C10_control',ours_proposal_budget={arm:policy['post_root_target_blocks']},
        budget_rule='explicit_per_arm_budget_after_target_root',**split_transfer_options(arm,seconds))
    _require([p['id'] for p in prompts]==d['prompt_ids'] and len(prompts)==32
        and result.get('population_binding')==binding and result.get('population')=='paper128_three_seed_replication'
        and result.get('duration')==duration and result.get('benchmark')=='VBench'
        and result.get('timing_scope')==reference.scope('self-forcing')
        and result.get('cohort')=='main24_burst' and result.get('arms')==[arm]
        and result.get('row_count')==len(result.get('rows',[]))==32
        and result.get('accepted_video_schedule')=='decode_after_generation'
        and result.get('automatic_winner_selection') is False,'method population/policy/timing differs')
    gpus=result['hardware']['gpus']
    _require(len(gpus)==1 and gpus[0].get('uuid','').startswith('GPU-')
        and gpus[0].get('name')=='NVIDIA H200' and not result['hardware'].get('unverified_test_fixture'),
        'method is not an actual one-H200 allocation')
    return prompts,binding,policy


def calibration(src, *, repo, run=None):
    expected=matrix(repo)['calibration']
    root=Path(run) if run is not None else src.run.parent/expected['run_id']
    _require(not root.is_symlink(),'C10 source is a symlink')
    frozen,marker=_load_frozen_index(root)
    _require(marker['run_id']==expected['run_id'] and marker['attempt_id']==expected['attempt_id']
        and marker['git_sha']==expected['code_revision'],'original C10 source identity differs')
    relative='attempts/'+marker['attempt_id']+'/'+CALIBRATION_RECEIPT_MANIFEST
    receipt_path=frozen.file(relative,label='original C10 receipt')
    _require(_sha256_file(receipt_path)[0]==expected['receipt_sha256'],'original C10 receipt digest differs')
    receipt=_load_json(receipt_path,label='C10 receipt')
    calibrator,evidence=load_sealed_calibration_ranks(source_run_dir=root/'attempts'/marker['attempt_id'],
        expected_binding=receipt['binding'],expected_receipt_sha256=expected['receipt_sha256'])
    return calibrator,evidence


class RecordedCampaign:
    """Only resolves already indexed checkpoint paths; has no generation API."""
    def __init__(self, cell, count):
        self.cell=cell;self.run_dir=cell.run/cell.prefix;self.chunk_count=count
    def _checkpoint_path(self, category, key, index):
        _require(category=='rollouts' and '/' not in key and '..' not in key
            and type(index) is int and 0<=index<self.chunk_count,'unsafe recorded checkpoint')
        return self.cell.frozen.file(self.cell.prefix+f'state/rollouts/{key}/chunk-{index:02d}.json',
                                     label='SF generated checkpoint')


def replay_route(row, generated, *, calibrator, seconds):
    arm=row['arm'];budget=int(arm.rsplit('_k',1)[1]);count=seconds*2
    options=split_transfer_options(arm,seconds);horizon=observation_horizon_for_arm(arm)
    policy=RankBurstPolicy(chunk_count=count,target_budget=budget,burst_length=4,calibration=calibrator,
        evaluation_id=f'eval-burst-{row["prompt_id"]}-{row["seed"]}-{budget}-4',
        observation_horizon=horizon,**options)
    ranks=[]
    _require(len(generated)==count and len(row['trajectory']['decisions'])==count-1,
             'rank/route checkpoint coverage is incomplete')
    for i,(chunk,record) in enumerate(zip(generated[1:],row['trajectory']['decisions']),1):
        role=chunk.context.role;rank=risk=drop=None
        if role=='draft':
            rank=chunk.effective_rank;ranks.append(rank)
            drop=causal_erank_sudden_drop(ranks,lookback_chunks=4);risk=calibrator.risk(drop)
        expected=policy.advance(chunk_index=i,role=role,risk_score=risk,erank_drop_score=drop)
        _require(record['chunk_index']==i and record['rank']==rank and record['risk_score']==risk
            and record['erank_drop_score']==drop and record['decision']==expected,
            'recorded causal eRank/CDF/future-burst decision differs')
    validate_burst_schedule(row['trajectory'],budget=budget,burst_length=4,chunk_count=count,
                           observation_horizon=horizon,**options)


def history_and_execution(src,row,*,calibrator,seconds):
    arm=row['arm'];prefix=src.prefix+'cells/'+arm+'/'
    cell=SimpleNamespace(run=src.run,frozen=src.frozen,prefix=prefix)
    if not hasattr(src,'pending_by_prompt'):
        src.pending_by_prompt={}
        for relative in src.frozen.entries:
            if relative.startswith(prefix+'measurements/rollouts/') and relative.endswith('.pending.json'):
                pending=PendingSequentialRollout.from_dict(_load_json(src.frozen.file(relative,label='pending rollout'),label='pending rollout'))
                _require(pending.history.prompt_id not in src.pending_by_prompt,'ambiguous pending prompt')
                src.pending_by_prompt[pending.history.prompt_id]=pending
        _require(set(src.pending_by_prompt)==set(src.definition['prompt_ids']),'pending rollout population differs')
    pending=src.pending_by_prompt[row['prompt_id']];history=pending.history;count=seconds*2
    _require(history.prompt_id==row['prompt_id'] and history.trajectory_seed==row['seed']
        and history.next_chunk_index==count and list(pending.corrected_chunks)==row['trajectory']['accepted_chunks']
        and list(pending.proposed_target_chunks)==row['trajectory']['proposed_chunks']
        and history.head_sha256==row['trajectory']['final_history_head_sha256'],'pending accepted history differs')
    campaign=RecordedCampaign(cell,count);generated=[];targets={0,*pending.corrected_chunks}
    for i,segment in enumerate(history.segments):
        checkpoint=_load_json(campaign._checkpoint_path('rollouts',pending.route_key,i),label='checkpoint')
        chunk=_generated_from_dict(checkpoint['generated']);generated.append(chunk)
        _require(checkpoint['schema']=='insight2_campaign_checkpoint_v2'
            and checkpoint['history']==history.prefix_before(i+1).to_dict()
            and chunk.segment==segment and segment.producer_role==('target' if i in targets else 'draft')
            and segment.model_id==('krea-realtime-video-14b' if i in targets else 'self_forcing_1.3b'),
            'checkpoint prefix/accepted producer/model differs')
        segment.verify_context(history.prefix_before(i),chunk.context)
        audit.referenced_file(cell,dict(path=segment.latent_path,sha256=segment.latent_sha256,
                                       nbytes=segment.latent_nbytes),'accepted SF latent')
    replay_route(row,generated,calibrator=calibrator,seconds=seconds)
    trace=collect_trajectory(campaign,key=pending.route_key,history=history,
        accepted=pending.corrected_chunks,proposed=pending.proposed_target_chunks)
    _require(normalized(trace)==row['trajectory'],'sealed trajectory does not match every actual generation')
    env=src.manifest['spec']['command']['env'];budget=len(pending.corrected_chunks)
    with patch.dict(os.environ,env,clear=True):
        cache=verify_native_cache(campaign,pending.route_key,trace)
        cache['burst_reuse']=verify_burst_reuse(campaign,pending.route_key,burst_length=4,budget=budget,
                                              opening_target_burst=seconds==30)
        cache['joint_handoff']=verify_rollout(campaign,pending.route_key,selected=pending.corrected_chunks)
        if seconds==30:cache['target_step_allocation']=verify_paper30_execution(campaign,pending.route_key,trace)
    _require(normalized(cache)==row['native_cache_audit'],'cache or physical fused-forward audit differs')
    updates=0
    for i in targets:
        diag=generated[i].diagnostics
        if seconds==15:
            observed=validate_correction_trace(diag['forward_trace'],tuple(diag['denoising_schedule']),
                chunk_index=i,expected_nonroot_schedule=MAIN_TABLE_FOUR_STEP_SCHEDULE)
            _require(observed==diag['denoising_forward_trace'],'native Target denoise trace differs')
            updates+=len(observed)
    if seconds==30:updates=cache['target_step_allocation']['target_denoising_forwards']
    expected=matrix(src.repo)['policies'][str(seconds)]
    _require(updates==expected['target_denoising_updates']
        and trace['counters']['physical_calls']=={'draft':count-expected['target_total_blocks'],
                                                 'target':expected['target_total_blocks']},
        'actual Target/Draft calls or Target updates differ from paper policy')
    prepared=pending.prepared_quality
    _require(prepared.video_sha256==row['video']['sha256'] and prepared.video_nbytes==row['video']['nbytes']
        and prepared.decode_evidence_sha256==row['decode_evidence']['sha256']
        and prepared.frame_count==count*12-3 and prepared.fps==24,'prepared/scored video evidence differs')
    return history,cell,updates


def verified_rows(src, *, repo, seconds, calibrator, calibration_evidence):
    src.repo=Path(repo);prompts,binding,policy=selection(src,repo=repo,seconds=seconds)
    _require(src.result['calibration_receipt']==calibration_evidence
        and src.result['calibration_code_revision']==matrix(repo)['calibration']['code_revision'],
        'method did not use the original independently sealed C10 distribution')
    output=[];batches={};profile=reference.duration_profile(seconds)
    for row,prompt in zip(src.result['rows'],prompts):
        _require(row['prompt_id']==prompt['id'] and row['prompt']==prompt['text']
            and row['seed']==binding['seed'] and row['arm']==policy['arm']
            and row['proposal_budget']==policy['post_root_target_blocks']
            and row['timing_scope']==reference.scope('self-forcing')
            and audit.positive(row['video_ready_wall_seconds'],'ready')>=
                audit.positive(row['generation_wall_seconds'],'generation'),'method row input/timing differs')
        _single_gpu_render_row(row)
        original=audit.read_json(src,f'cells/{row["arm"]}/provenance/{prompt["id"]}.json')
        _require(original=={k:v for k,v in row.items() if k not in ('quality_scores','quality_evidence')},
                 'method timing/trajectory differs from pre-scoring evidence')
        mid=f'acceptance-pilot-{row["arm"]}-{prompt["id"]}'.replace('_','-')
        _require(row['measurement_id']==mid,'scoring measurement id differs')
        video,actual,codec=audit.quality(src,row,row['quality_evidence'],prompt,mid,seconds)
        _require((actual['sha256'],actual['nbytes'])==(row['video']['sha256'],row['video']['nbytes'])
            and video==src.run/src.prefix/row['video']['path'],'scored and timed video differ')
        reference.legacy.official_scores(src,row,prompt,mid,seconds,video,short_batches=batches)
        audit.verify_geometry(video,profile)
        history,cell,updates=history_and_execution(src,row,calibrator=calibrator,seconds=seconds)
        decode=_load_json(audit.referenced_file(src,row['decode_evidence'],'SF video decode'),label='SF decode')
        _require(decode['history_manifest_sha256']==history.manifest_sha256
            and decode['expected_frames']==profile.frame_count and decode['latent_chunk_count']==profile.chunk_count,
            'SF decode history/horizon differs')
        reference.verify_renderer(cell,row['accepted_stream'],history,decode,
                                   model='self-forcing',role='target',profile=profile)
        output.append(dict(row=row,history=history,codec=codec,target_updates=updates))
    return output


def execution_gate(run, *, repo):
    run=Path(run);_require(not run.is_symlink(),'execution gate is a symlink')
    frozen,marker=_load_frozen_index(run)
    terminal=_load_json(frozen.file('run.json',label='gate terminal'),label='gate terminal')
    _require(terminal.get('execution_status')=='SUCCEEDED' and terminal.get('exit_code')==0
        and terminal.get('latest_attempt_id')==marker['attempt_id'],'execution gate did not succeed')
    env=_load_json(frozen.file('resolved-manifest.json',label='gate manifest'),label='gate manifest')
    relative='experiments/sf-single-h200-handoff-integrated-parity-ntu-v1.manifest.json'
    spec=env['manifest']['spec'];argv=spec['command']['argv']
    _require(env['manifest']==source_manifest(repo,marker['git_sha'],relative)
        and spec['resources']['gpus']==1 and argv[2]=='cfexp.sf_handoff_fusion_probe'
        and '--integrated' in argv and '--native-timestep-shapes' in argv
        and all(env.get(k)==marker[k] for k in ('run_id','attempt_id','spec_hash')),
        'gate is not the original single-device integrated native parity experiment')
    result=_load_json(frozen.file('attempts/'+marker['attempt_id']+'/result.json',label='gate result'),label='gate result')
    arms={'sequential':5,'batched_prefill':2,'fused_first_step':1}
    _require(result.get('schema')=='sf_handoff_fusion_probe_v1'
        and result.get('integrated_joint_forward') is True and result.get('native_timestep_shapes') is True
        and result.get('timing_includes_rank_and_kv_witness') is True
        and result.get('target_suffix_chunks')==4 and result.get('model_forward_counts')==arms
        and len(result['hardware']['gpus'])==1
        and result['hardware']['gpus'][0]['name']=='NVIDIA H200', 'gate operator/hardware differs')
    rows=result['rows']
    _require(len(rows)==36 and {(r['arm'],r['repetition']) for r in rows}
        =={(arm,i) for arm in arms for i in range(12)}
        and result['source']['index_sha256']==argv[argv.index('--source-index-sha256')+1]
        and all(type(r['rank']) in (int,float) and math.isfinite(r['rank']) and r['rank']>0
            and re.fullmatch('[0-9a-f]{64}',r['clean_kv_witness_sha256']) for r in rows)
        and all(r['model_forward_calls']==arms[r['arm']]
            and audit.positive(r['wall_seconds'],'gate latency') for r in rows)
        and len({(r['rank'],r['clean_kv_witness_sha256']) for r in rows})==1,
        'gate lacks all real forward counts or changes the observed rank/clean KV witness')
    _require(set(result['parity'])=={'batched_prefill','fused_first_step'},'gate parity modes missing')
    for item in result['parity'].values():
        _require(item['first_output']['exact'] is True and item['final_output']['exact'] is True
            and item['committed_kv']['tensor_count']==item['committed_kv']['tensors_exact']==120
            and item['committed_kv']['cursors_exact'] is True,'gate latent/full30-layer KV/cursor parity failed')
    return dict(**{k:marker[k] for k in ('run_id','attempt_id','git_sha','artifact_index_sha256')},
        verified_native_integrated_parity=True,actual_gpus=1,measured_cells=36,
        scope=result['scope'],not_main_video_ready_speedup=True)


def compatible_source(repo, candidate, target):
    _require(all(isinstance(x,str) and re.fullmatch('[0-9a-f]{40}',x) for x in (candidate,target)),
             'full generation source revisions required')
    env=dict(PATH='/usr/bin:/bin',HOME=os.path.expanduser('~'),GIT_CONFIG_GLOBAL='/dev/null',
        GIT_CONFIG_SYSTEM='/dev/null',GIT_CONFIG_NOSYSTEM='1',GIT_NO_REPLACE_OBJECTS='1')
    changed=subprocess.check_output(['git','--no-replace-objects','-c','core.hooksPath=/dev/null',
        '-C',str(repo),'diff','--name-only',candidate,target,'--','src/cfexp','pyproject.toml','uv.lock'],
        env=env,text=True).splitlines()
    _require(set(changed)<=NON_EXECUTED_SOURCE_DIFFERENCES,
             'compared generation/renderer/runtime source differs; matched reference required')
    return dict(candidate_revision=candidate,target_revision=target,
                verified_non_executed_source_differences=changed)


def compare(ours, targets, *, repo, seconds, calibrator, calibration_evidence, gate):
    expected=[r for r in matrix(repo)['runs'] if r['nominal_seconds']==seconds]
    _require(len(ours)==len(targets)==len(expected)==12 and len({s.run for s in ours+targets})==24
        and {s.definition['id'] for s in ours}=={r['id'] for r in expected},'need all twelve method and twelve Target shards')
    _require(gate['git_sha']==ours[0].marker['git_sha']
        and all(s.marker['git_sha']==gate['git_sha'] for s in ours),
        'method source does not match the independently verified execution gate')
    method_cases={};target_cases={};compatibility={}
    for s in ours:
        for item in verified_rows(s,repo=repo,seconds=seconds,calibrator=calibrator,calibration_evidence=calibration_evidence):
            k=item['row']['prompt_id'],item['row']['seed'];_require(k not in method_cases,'duplicate method case')
            method_cases[k]=item,s
    for s in targets:
        _require(s.definition['method_id']=='target_only','control is not pure Target')
        for item in reference.verified_rows(s,repo=repo,model='self-forcing',seconds=seconds):
            k=item['row']['prompt_id'],item['row']['seed'];_require(k not in target_cases,'duplicate Target case')
            target_cases[k]=item,s
    keys=[(p['id'],seed) for p in _load_json(Path(repo)/population.PROTOCOL,label='paper population')['prompts']
          for seed in population.SEEDS]
    _require(len(keys)==384 and set(method_cases)==set(target_cases)==set(keys),'complete384-case pairing differs')
    pairs=[]
    for key in keys:
        a,sa=method_cases[key];b,sb=target_cases[key]
        revisions=sa.marker['git_sha'],sb.marker['git_sha']
        if revisions not in compatibility:compatibility[revisions]=compatible_source(repo,*revisions)
        _require(sa.result['hardware']==sb.result['hardware'] and a['codec']==b['codec']
            and [x.initial_noise_descriptor_sha256 for x in a['history'].segments]==
                [x.initial_noise_descriptor_sha256 for x in b['history'].segments]
            and a['history'].segments[0].latent_sha256==b['history'].segments[0].latent_sha256
            and all(segment.model_artifact_sha256==b['history'].segments[0].model_artifact_sha256
                    for segment in a['history'].segments if segment.producer_role=='target'),
            'physical GPU, codec, noise or common Target root differs')
        pairs.append(dict(prompt_id=key[0],seed=key[1],gpu_uuid=sa.result['hardware']['gpus'][0]['uuid'],
            method_quality_100=math.fsum(a['row']['quality_scores'].values())*100/6,
            target_quality_100=math.fsum(b['row']['quality_scores'].values())*100/6,
            method_ready_seconds=a['row']['video_ready_wall_seconds'],target_ready_seconds=b['row']['video_ready_wall_seconds']))
    mean=lambda k:math.fsum(x[k] for x in pairs)/384
    return dict(schema='paper128_sf_single_h200_matched_report_v1',status='verified',nominal_seconds=seconds,
        prompt_count=128,seeds=list(population.SEEDS),case_count=384,actual_gpus_per_video=1,
        policy=matrix(repo)['policies'][str(seconds)],timing_scope=reference.scope('self-forcing'),
        method_quality_mean_100=mean('method_quality_100'),target_quality_mean_100=mean('target_quality_100'),
        method_mean_ready_seconds=mean('method_ready_seconds'),target_mean_ready_seconds=mean('target_ready_seconds'),
        speedup_ratio_of_mean_ready_times=mean('target_ready_seconds')/mean('method_ready_seconds'),pairs=pairs,
        source_compatibility=list(compatibility.values()),calibration=calibration_evidence,execution_gate=gate,
        sources=[{k:s.marker[k] for k in ('run_id','attempt_id','git_sha','artifact_index_sha256')} for s in ours+targets],
        dataset_publication_verified=False,replication_not_recovered_original_submission_record=True,
        source_producer_formal_ours_flags=[s.result.get('formal_ours_result') for s in ours],caveats=['Six-dimension macro mean, not official full VBench total.',
            'Actual384 cases and physical GPU/codec/noise are paired; no selection by score.',
            'Target retains one native session with disclosed bounded-context rebases.',
            'Local seals are verified; upload and W&B are separate evidence. Existing runs were not retroactively tracked.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ours',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--seconds',type=int,choices=(15,30),required=True)
    p.add_argument('--execution-gate',type=Path,required=True)
    p.add_argument('--calibration-run',type=Path);a=p.parse_args()
    ours=[source(r,repo=a.repo,seconds=a.seconds) for r in json.loads(a.ours.read_text())]
    targets=[reference.source(r,repo=a.repo,model='self-forcing',seconds=a.seconds)
             for r in json.loads(a.targets.read_text())]
    _require(ours,'method sources missing');calibrator,evidence=calibration(ours[0],repo=a.repo,run=a.calibration_run)
    print(json.dumps(compare(ours,targets,repo=a.repo,seconds=a.seconds,
        calibrator=calibrator,calibration_evidence=evidence,gate=execution_gate(a.execution_gate,repo=a.repo)),allow_nan=False))


if __name__=='__main__':main()
