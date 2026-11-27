"""Original full native LongLive equality gate plus real offline-capable W&B.

Separate instrumented diagnostic only. No quality or E2E speedup result.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from . import longlive_joint_policy_gate as gate
from .joint_gate_profile import ENV,validate_forward_records
from .longlive_joint_handoff import require
from .tracked_sdvg import tracking_mode,materialize_sdk_log_links,write_new


def process_factory(argv,**kwargs):
    require(os.environ.get(ENV)=='1' and list(argv[1:3])==['-m','cfexp.correction_worker'],
            'tracked joint gate requires original native-worker launch')
    command=list(argv);command[2]='cfexp.longlive_joint_policy_worker'
    return subprocess.Popen(command,**kwargs)


def validate_measurements(root, gate_report):
    observations=[];requests=[]
    for path in (root/'live'/'provenance'/'tracking').glob('*.jsonl'):
        values=[json.loads(x) for x in path.read_text().splitlines()]
        (observations if path.name.startswith('draft-forwards-') else requests).extend(values)
    samples=validate_forward_records(observations,gate.expected_forward_trace())
    require(len({x['pid'] for x in observations})==1,'joint forward telemetry has multiple owners')
    for row in samples:
        trace=root/'live'/row['trace_path']
        require(not trace.is_symlink() and trace.is_file()
                and hashlib.sha256(trace.read_bytes()).hexdigest()==row['trace_sha256'],
                'joint profiler trace missing or changed')
    require(len([x for x in requests if x['role']=='target' and x['action']=='generate'])==12
        and len([x for x in requests if x['role']=='draft' and x['action']=='longlive_joint_policy_gate'])==1,
        'joint gate RPC telemetry differs from original Target/gate workload')
    require(all(type(x.get('request_wall_seconds')) in (float,int)
        and math.isfinite(x['request_wall_seconds']) and x['request_wall_seconds']>0
        and type(x.get('peak_allocated_bytes')) is int and x['peak_allocated_bytes']>0
        and type(x.get('peak_reserved_bytes')) is int
        and x['peak_reserved_bytes']>=x['peak_allocated_bytes'] for x in requests),
        'joint RPC timing/memory telemetry is missing or invalid')
    require(gate_report['passed'] is True and gate_report['gate']['passed'] is True,
            'native parity did not pass; profiling cannot substitute for correctness')
    target_profiles=list((root/'live'/'provenance'/'kernel-profile').glob('target-generate-chunk-03.json'))
    require(len(target_profiles)==1,'post-warmup Target kernel profile is missing')
    target_profile=json.loads(target_profiles[0].read_text())
    require(target_profile.get('schema')=='native_kernel_profile_v1' and target_profile.get('role')=='target'
        and target_profile.get('action')=='generate' and target_profile.get('chunk_index')==3
        and target_profile.get('cuda_kernel_us_sum',0)>0,'Target profile is not actual third-or-later generation')
    target_requests=[x for x in requests if x['role']=='target' and x['action']=='generate']
    require(target_requests[3]['request_id']==target_profile['request_id'],
            'Target profile request differs from fourth fixed-history generation')
    from .stage_timing_analysis import analyze_records
    stages=[]
    for path in (root/'live'/'provenance'/'stage-timing').glob('*.jsonl'):
        stages.extend(json.loads(x) for x in path.read_text().splitlines())
    require(stages,'actual stage costs are missing')
    analysis=analyze_records(stages)
    from .tracked_sdvg import write_new
    write_new(root/'provenance'/'tracking'/'stage-analysis.json',analysis)
    return dict(schema='longlive_joint_profile_evidence_v1',executed_actual_cuda=True,
        native_forward_count=len(observations),matching_forward_warmup=2,samples=samples,
        request_observation_count=len(requests),no_end_to_end_speed_claim=True,
        target_kernel_profile=target_profile,
        target_unprofiled_post_warmup_request_seconds=[x['request_wall_seconds'] for x in target_requests[2:]
            if x['request_id']!=target_profile['request_id']],
        original_full_oracle_passed=True,physical_device_count=1,
        timing_scope='instrumented generator forwards; native oracle hooks included; not a matched speedup',
        memory_scope='per-process allocator samples; no simultaneous physical-device peak claim')


def main():
    import argparse
    from .main_table_baseline_campaign import _run_dir,_model_root
    from .metadata import find_git_root,collect_git_metadata
    from .stage_timing_probe import _hardware
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--single-gpu',action='store_true');a=parser.parse_args()
    require(a.single_gpu and os.environ.get(ENV)=='1' and os.environ.get('CFEXP_STAGE_TIMING')=='1',
            'tracked native gate requires explicit one-H200 diagnostic and stage costs')
    root=_run_dir();repo=find_git_root(Path.cwd());devices=gate.gate_devices(True)
    hardware=_hardware((devices[0],));require(len(hardware['gpus'])==1 and hardware['gpus'][0]['name']=='NVIDIA H200',
                                           'tracked native gate requires actual H200')
    config=dict(method='LongLive native full-policy FuseStep observation gate',phase='bounded_profile_not_main',
        git_sha=collect_git_metadata(repo)['sha'],physical_device_count=1,batch_size=1,rank=0,
        output_resolution=[480,832],comparison_fps=24,precision='bfloat16',prompt=gate.PROMPT,
        generation_seed=20260911,native_schedule=list(gate.SCHEDULE),native_local_frames=12,native_sink_frames=3,
        clean_update_frames=3,cases=[list(x) for x in gate.CASES],following_draft_chunks_per_case=6,
        warmup_matching_forwards=2,full_410_forward_trace_required=True,
        no_quality_result=True,no_end_to_end_speed_claim=True,
        backward='not applicable: inference',optimizer='not applicable: inference',loss='not applicable: inference',
        learning_rate='not applicable: inference',gradient_norm='not applicable: inference',
        decoding='not applicable: latent/cache correctness gate',distributed_communication='not applicable: single device',
        checkpoints={label:hashlib.sha256((repo/path).read_bytes()).hexdigest() for label,path in
            [('draft','experiments/erank3-longlive-policy-v1.artifacts.json'),('target','experiments/insight2-krea14b.artifacts.json')]})
    import wandb
    mode=tracking_mode(os.environ);directory=root/'tracking'/'wandb';directory.mkdir(parents=True,exist_ok=True)
    kwargs=dict(project='collaborative-forcing-reproduction',name=root.name,dir=str(directory),config=config,
        save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,disable_code=True,symlink=False,
                                             console='off',_disable_stats=True))
    fallback=False
    try:run=wandb.init(mode=mode,**kwargs)
    except Exception:
        if mode!='online':raise
        fallback=True;mode='offline';run=wandb.init(mode='offline',**kwargs)
    write_new(root/'provenance'/'tracking'/'session.json',dict(schema='wandb_session_v1',mode=mode,
        run_id=run.id,directory='tracking/wandb',online_fallback_to_offline=fallback,synced_evidence_verified=False,config=config,hardware=hardware))
    original=gate.process_factory;gate.process_factory=process_factory;ok=False
    try:
        report=gate.run(root,repo,_model_root(),devices,single_gpu=True)
        evidence=validate_measurements(root,report);write_new(root/'provenance'/'tracking'/'profile-summary.json',evidence)
        run.log({'correctness/full_native_oracle_passed':1,'correctness/actual_forwards':evidence['native_forward_count']})
        for row in evidence['samples']:
            run.log({f'profile/{row["kind"]}/{k}':row[k] for k in ['request_wall_seconds','cuda_stream_elapsed_seconds',
                'cuda_kernel_us_sum','cuda_transfer_us_sum','peak_allocated_bytes','peak_reserved_bytes',
                'stream_window_frame_equivalents_per_second','warmup_matching_forwards_before']})
        for seconds in evidence['target_unprofiled_post_warmup_request_seconds']:
            run.log({'target/post_warmup_unprofiled_request_seconds':seconds})
        run.log({'target/profile_cuda_kernel_us_sum':evidence['target_kernel_profile']['cuda_kernel_us_sum']})
        ok=True;return 0
    finally:
        gate.process_factory=original
        run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
        write_new(root/'provenance'/'tracking'/'finished.json',dict(schema='wandb_finish_v1',mode=mode,
            execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':raise SystemExit(main())
