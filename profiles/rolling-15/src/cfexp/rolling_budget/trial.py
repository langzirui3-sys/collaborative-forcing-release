"""Fixed 24-prompt, single-H200 Rolling B2/B4 video-ready comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
from types import SimpleNamespace

from ..baseline_population import load_baseline_population
from ..longlive_fullvideo_preflight import bound, environment
from ..main_table_v1 import QUALITY_DIMENSIONS
from ..tracked_sdvg import (write_new, tracking_mode, materialize_sdk_log_links,
                            require_profile_samples, log_result)
from ..rolling_joint_handoff import ENV as JOINT_ENV
from .admission import admit_original_pair, require
from .worker import WIDTH_ENV, validate_state

SEED = 20260911
PROFILE_PROMPT = '001_subject_consistency_vbl0268'
CALIBRATION_REVISION = '1a06af650ca0e495d75a59fc9bef3541b0bf413f'
RUN_ROOT_ENV = 'CFEXP_ROLLING_BUDGET_ORIGINAL_RUNS'
CELLS = (('b2-profile',2,True),('b4-profile',4,True),
         ('b2-timed',2,False),('b4-timed',4,False))
WORKER_FACTORY = 'cfexp.rolling_budget.worker:build_backend'


def process_factory(argv, *, draft_factory=WORKER_FACTORY, **kwargs):
    command = list(argv)
    require(command[1:3] == ['-m','cfexp.correction_worker'],
            'unexpected worker entry for Rolling budget')
    role = command[command.index('--role')+1]
    index = command.index('--backend-factory')+1
    expected = {'draft':'cfexp.rolling_policy_worker:build_backend',
                'target':'cfexp.correction_runtime:build_target_backend'}
    require(role in expected and command[index] == expected[role],
            'unexpected native model factory')
    if role == 'draft':
        command[index] = draft_factory or expected['draft']
    command[2] = 'cfexp.tracked_sdvg_worker'
    return subprocess.Popen(command, **kwargs)


def launch_runtime(**kwargs):
    from ..accepted_stream_worker import process_factory as ordinary
    from ..correction_runtime import launch_correction_runtime
    require(kwargs.pop('process_factory') is ordinary
            and kwargs.get('draft_model_id') == 'rolling_forcing_1.3b'
            and kwargs.get('chunk_count') == 30 and kwargs.get('allow_shared_device') is True
            and len(set(kwargs['devices'])) == 1, 'wrong device or native Rolling horizon')
    return launch_correction_runtime(process_factory=process_factory, **kwargs)


def audit(campaign, *, key, history, selected, blocks, chunk_count=30, **kwargs):
    from ..krea_chunk_backend import validate_native_correction_state, validate_target_forward_trace
    from ..rank_burst import validate_burst_schedule
    require(kwargs['draft_model_id'] == 'rolling_forcing_1.3b'
            and kwargs['burst_length'] == blocks and kwargs['observation_horizon'] == 8
            and campaign.chunk_count == chunk_count and chunk_count in (30, 60)
            and not any(kwargs.get(x,False) for x in ('opening_target_burst','split_target_burst', 'observation_offset',
                'fixed_target_positions','rolling_prefix_transactions','rolling_committed_prefix',
                'final_window_at_tail','longlive_prefill','motion_trigger')),
            'audit received a changed placement or cache policy')
    require(len(history.segments)==chunk_count and len(selected)==blocks and
            list(selected)==list(range(selected[0], selected[0]+blocks)),
            'expected a complete native trajectory with one burst')
    target = {0,*selected}
    sessions = set(); draft_states = []; decisions = []; previous = None
    for i,segment in enumerate(history.segments):
        row=json.loads(campaign._checkpoint_path('rollouts',key,i).read_text())
        gen=row['generated']; role='target' if i in target else 'draft'
        require(gen['context']['role']==segment.producer_role==role
                and gen['segment']['segment_sha256']==segment.segment_sha256,
                'accepted segment or owner differs')
        if i:
            decisions.append({k:row['metadata'][k]
                              for k in ('chunk_index','decision','rank','risk_score')})
        d=gen['diagnostics']
        if role=='target':
            state=d['native_correction_state'];validate_native_correction_state(state,chunk_index=i)
            validate_target_forward_trace(d['forward_trace'],d['denoising_schedule'],
                                          history_frames=3*i,expected_schedule=(1000.,750.,500.,250.))
            require(d['explicit_initial_noise_verified'] is True
                    and (i==0 or not state['session_recreated']),
                    'Target noise or persistent native owner differs')
            sessions.add(state['native_session_id'])
        else:
            state=d['rolling_state']
            validate_state(state,index=i,previous=previous,blocks=blocks)
            draft_states.append(state);previous=i
    require(len(sessions)==1 and sum(x['joint_first_window_calls'] for x in draft_states)==1
            and len(draft_states)==chunk_count-1-blocks, 'expected one joint handoff and one persistent Target owner')
    validate_burst_schedule(dict(decisions=decisions,accepted_chunks=list(selected),
        proposed_chunks=list(selected)),budget=blocks,burst_length=blocks,
        chunk_count=chunk_count,observation_horizon=8)
    return dict(schema='rolling_joint_budget_trial_cache_audit_v1',
        target_chunks=sorted(target),target_session_id=next(iter(sessions)),
        draft_states=draft_states,decisions=decisions)


@contextmanager
def scoped_auditor(blocks, *, chunk_count=30):
    """Use the budget-specific audit only inside this dedicated controller."""
    from .. import rank_burst
    original=rank_burst.audit_native_model_burst
    rank_burst.audit_native_model_burst=partial(audit,blocks=blocks,chunk_count=chunk_count)
    try:
        yield
    finally:
        rank_burst.audit_native_model_burst=original


def snapshot(root, result, *, blocks, ids, hardware):
    """Validate the exact accepted path, common prefix, noise and complete MP4."""
    rows = json.loads(bound(root,result['raw_rows']).read_text())['rows']
    require(result['hardware'] == hardware and len(rows) == len(ids)
            and {x['prompt_id'] for x in rows} == set(ids)
            and result['duration']['frame_count'] == 357,
            'Rolling budget population, device or video horizon differs')
    chunks = {pid:{} for pid in ids}
    for file in sorted((root/'state/rollouts').glob('*/chunk-*.json')):
        r=json.loads(file.read_text()); pid=r['history']['prompt_id']; g=r['generated']
        seg=g['segment']; i=seg['chunk_index']; n=g['context']['noise']
        require(pid in chunks and 0<=i<30 and i not in chunks[pid],
                'foreign or duplicate checkpoint in paired rollout')
        bound(root,dict(path=seg['latent_path'],sha256=seg['latent_sha256']))
        bound(root,dict(path=f'noise/{pid}/seed-{SEED}/chunk-{i:02d}.safetensors',
                        sha256=n['tensor_sha256']))
        chunks[pid][i]=dict(latent=seg['latent_sha256'],role=g['context']['role'],noise=n)
    cases={}
    for row in rows:
        pid=row['prompt_id']; doc=json.loads((root/'provenance/ours'/f'{pid}.json').read_text())
        target=doc['accepted_target_chunks']; start=target[0]
        require(row['seed']==SEED and set(chunks[pid])==set(range(30))
                and target==list(range(start,start+blocks)) and 2<=start<=9
                and doc['native_cache_audit']['target_chunks']==[0,*target]
                and [i for i in range(30) if chunks[pid][i]['role']=='target']==[0,*target],
                'incorrect Target root, burst placement, or final accepted history')
        stream=json.loads(bound(root,doc['accepted_stream']).read_text()); bound(root,stream['video'])
        require(stream['decoder_and_encoder_joined'] and stream['gpu_decode_deferred_until_finish']
                and stream['pixels']['shape']==[357,3,480,832], 'incomplete native video')
        quality=row['quality_scores']; ready=row['video_ready_wall_seconds']
        require(set(quality)==set(QUALITY_DIMENSIONS)
                and all(math.isfinite(x) and 0<=x<=1 for x in quality.values())
                and math.isfinite(ready) and ready>0, 'invalid official quality or latency')
        decisions=doc['native_cache_audit']['decisions']
        cases[pid]=dict(target_start=start, target_chunks=target, quality=quality, ready=ready,
            mp4_sha256=stream['video']['sha256'],
            accepted=[chunks[pid][i] for i in range(30)],
            pre_target_decisions=[{k:d[k] for k in ('chunk_index','rank','risk_score')}
                                   for d in decisions if d['chunk_index']<start])
    return dict(hardware=result['hardware'], population=result['population'],
                calibration=result['calibration_receipt'], cases=cases,
                timing_scope=result['timing_scope'])


def paired(b2,b4,*,expected=24):
    for key in ('hardware','population','calibration','timing_scope'):
        require(b2[key]==b4[key], 'Rolling budget pair differs: '+key)
    require(b2['cases'].keys()==b4['cases'].keys() and len(b2['cases'])==expected,
            'budget comparison is missing paired prompts')
    result=[]
    for pid,x in b2['cases'].items():
        y=b4['cases'][pid]; s=x['target_start']
        require(s==y['target_start'] and x['pre_target_decisions']==y['pre_target_decisions']
                and x['accepted'][:s+2]==y['accepted'][:s+2]
                and [r['noise'] for r in x['accepted']]==[r['noise'] for r in y['accepted']],
                'burst start, common accepted prefix, or rollout noise differs: '+pid)
        result.append(dict(prompt_id=pid,start=s, b2_quality=statistics.mean(x['quality'].values())*100,
            b4_quality=statistics.mean(y['quality'].values())*100,
            b2_dynamic=x['quality']['dynamic_degree']*100,b4_dynamic=y['quality']['dynamic_degree']*100,
            b2_ready=x['ready'],b4_ready=y['ready']))
    return result


def interval(values):
    rng=random.Random(SEED); n=len(values)
    draws=sorted(statistics.mean(values[rng.randrange(n)] for _ in values) for _ in range(10000))
    def q(p):
        i=(len(draws)-1)*p;lo=math.floor(i)
        return draws[lo]+(draws[lo+1]-draws[lo])*(i-lo)
    return [q(.025),q(.975)]


def execute(args, root, hardware, prompts, run):
    from ..main_table_ours_campaign import run_campaign
    from ..krea_long_session_gate import validate_operator_profile
    captures={}
    for name,width,profiled in CELLS:
        cell=root/'cells'/name; cell.mkdir(parents=True,exist_ok=False)
        ids=[PROFILE_PROMPT] if profiled else [p['id'] for p in prompts]
        with environment({'CFEXP_RUN_DIR':str(cell),WIDTH_ENV:str(width),
                          'CFEXP_SDVG_BOUNDED_PROFILE':str(int(profiled))}), scoped_auditor(width):
            result=run_campaign(draft_artifact_spec_path=args.draft_artifact_spec,
                draft_model_id='rolling_forcing_1.3b',target_artifact_spec_path=args.target_artifact_spec,
                boundary_config_path=args.boundary_config,vbench_checkout=args.vbench_checkout,
                vbench_python=args.vbench_python,vbench_asset_manifest=args.vbench_asset_manifest,
                vbench_asset_spec=args.vbench_asset_spec,baseline_population_v2=True,
                duration_seconds=15,selection_strategy='rank_burst',
                burst_calibration_revision=CALIBRATION_REVISION,sealed_calibration_ranks=True,
                native_burst_length=width,record_video_ready_timing=True,overlap_native_vae=True,
                runtime_launcher=launch_runtime,
                smoke_prompt_id=PROFILE_PROMPT if profiled else None)
        if profiled:
            require_profile_samples(cell,config=dict(cell=name),result=result)
            summary=json.loads((cell/'provenance/tracking/profile-summary.json').read_text())
            require(summary['coverage_complete_for_this_campaign'] is True,
                    'Draft and Target both require post-warmup CUDA evidence')
            for sample in summary['samples']:
                validate_operator_profile(sample,role=sample['role'],action='generate',root=cell)
        log_result(SimpleNamespace(log=lambda row:run.log({**row,'cell':name,'profiled':profiled})),
                   cell,result)
        captures[name]=snapshot(cell,result,blocks=width,ids=ids,hardware=hardware)
        if name=='b4-profile':
            paired(captures['b2-profile'],captures[name],expected=1)
    rows=paired(captures['b2-timed'],captures['b4-timed'])
    changes={}
    for metric in ('quality','dynamic','ready'):
        v=[row['b4_'+metric]-row['b2_'+metric] for row in rows]
        changes[metric]=dict(mean=statistics.mean(v),bootstrap95=interval(v))
    return dict(schema='rolling_single_h200_budget_ablation_v1',frames=357,
        population=captures['b2-timed']['population'],hardware=hardware,prompt_count=24,
        generation_seed=SEED,same_prefix_and_burst_start=True,rows=rows,changes=changes,
        means={k:statistics.mean(row[k] for row in rows)
               for k in rows[0] if k.startswith(('b2_','b4_'))},
        profile_cases_excluded_from_means=True,not_paper128_main=True)


def main(argv=None):
    from ..accepted_stream import stream_devices
    from ..metadata import find_git_root, collect_git_metadata
    from ..runtime_helpers import shutdown_owned_runtime_helpers
    from ..stage_timing import _unsealed_directory
    from ..stage_timing_probe import _hardware
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('draft-artifact-spec','target-artifact-spec','boundary-config','vbench-checkout',
                 'vbench-python','vbench-asset-manifest','vbench-asset-spec'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv)
    require(all(os.environ.get(k)=='1' for k in (
        JOINT_ENV,'CFEXP_NATIVE_SINGLE_GPU','CFEXP_ACCEPTED_VIDEO_STREAM',
        'CFEXP_RENDER_TARGET_PRIORITY','CFEXP_SDVG_TRACKING','CFEXP_STAGE_TIMING'))
        and os.environ.get('CFEXP_PAPER_REPRODUCTION','0')=='0'
        and os.environ.get('CFEXP_GENERATION_SEED',str(SEED))==str(SEED),
        'Rolling budget trial needs the original single-device 24-prompt scope')
    root=_unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']))
    repo=find_git_root(Path.cwd());hardware=_hardware((stream_devices()[0],))
    require(len(hardware['gpus'])==1 and hardware['gpus'][0]['name']=='NVIDIA H200',
            'exactly one physical H200 is required')
    proof=admit_original_pair(repo,os.environ[RUN_ROOT_ENV])
    prompts,binding=load_baseline_population(repo,model_pair='krea_rolling_forcing')
    journal=root/'tracking/wandb';journal.mkdir(parents=True)
    config=dict(source=collect_git_metadata(repo)['sha'],hardware=hardware,protocol=binding,
        prompts=prompts,actual_gpu_count=1,batch_size=1,rank=0,precision='BF16',
        nominal_seconds=15,frame_count=357,resolution=[832,480],fps=24,
        generation_seed=SEED, target_root=True, post_root_budgets=[2,4], observation_horizon=8,
        cell_order=[x[0] for x in CELLS],gate_indexes={str(k):v['index'] for k,v in proof['gates'].items()},
        draft_artifact_spec_sha256=hashlib.sha256(args.draft_artifact_spec.read_bytes()).hexdigest(),
        target_artifact_spec_sha256=hashlib.sha256(args.target_artifact_spec.read_bytes()).hexdigest(),
        timing_scope='end-to-end video ready; separate operator profile cells',
        backward='not applicable: inference',optimizer='not applicable: inference',
        loss='not applicable: inference',learning_rate='not applicable: inference',
        gradient_norm='not applicable: inference',distributed_communication='none')
    import wandb
    options=dict(project='collaborative-forcing-reproduction',name=root.name,config=config,
        dir=str(journal),save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,
            disable_code=True,symlink=False,console='off',_disable_stats=True))
    mode=tracking_mode(os.environ);fallback=False
    try: run=wandb.init(mode=mode,**options)
    except Exception:
        if mode!='online':raise
        mode='offline';fallback=True;run=wandb.init(mode=mode,**options)
    write_new(root/'provenance/tracking/session.json',dict(mode=mode,run_id=run.id,config=config,
        online_fallback_to_offline=fallback,synced_evidence_verified=False))
    ok=False
    try:
        result=execute(args,root,hardware,prompts,run)
        write_new(root/'result.json',result);ok=True;return 0
    finally:
        try:shutdown_owned_runtime_helpers(label='rolling-budget-trial')
        finally:
            run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(journal)
            write_new(root/'provenance/tracking/finished.json',dict(mode=mode,
                execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':
    raise SystemExit(main())
