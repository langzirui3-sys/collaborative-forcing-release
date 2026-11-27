"""Bounded full SDVG reward/RPC integration trial after native VAE parity.

Separate opt-in entry only. Full main-table expansion is deliberately unavailable
until this reward-policy integration has also been inspected and validated.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .sdvg_accepted_vae import ENV,POLICY
from .sdvg_accepted_client import AcceptedDraftClient,generate_selected_vae_trajectory
from .sdvg_vae_transaction import require,FIELDS,is_sha
from .sdvg_vae_transaction_gate import DECISIONS
from .tracked_sdvg import (ENV as TRACKING_ENV,PROFILE,tracking_mode,write_new,
    materialize_sdk_log_links,require_profile_samples,log_result)

NATIVE_GATE_COMMIT='02876fa7ae7817fa6a54300f7b60aa4a21bf4f06'


def validate_native_gate(run_dir,index_sha256):
    from .main_table_merge import _load_frozen_index
    from .main_table_baseline_campaign import _sha256_file
    from .streaming_vae_probe import VAE_SHA,WEIGHTS_SHA
    require(is_sha(index_sha256) and run_dir.is_absolute() and not run_dir.is_symlink()
        and run_dir.resolve(strict=True)==run_dir,'native VAE proof requires canonical bound source')
    require(_sha256_file(run_dir/'artifact-index.json')[0]==index_sha256,'native VAE proof index changed')
    frozen,marker=_load_frozen_index(run_dir)
    require(marker.get('git_sha')==NATIVE_GATE_COMMIT and marker.get('status')=='SUCCEEDED',
            'native VAE proof has wrong implementation or unsuccessful marker')
    prefix='attempts/'+marker['attempt_id']+'/'
    path=frozen.file(prefix+'sdvg-accepted-vae-native-gate.json',label='native VAE parity proof')
    report=json.loads(path.read_text())
    require(report.get('schema')=='sdvg_accepted_vae_native_gate_v1' and report.get('passed') is True
        and report.get('physical_device_count')==1 and report.get('quality_measured') is False
        and report.get('production_sdvg_enabled') is False and report.get('reward_model_executed') is False,
        'native VAE proof scope differs')
    component=report.get('native_component',{})
    require(component.get('native_vae_sha256')==VAE_SHA and component.get('checkpoint_sha256')==WEIGHTS_SHA
        and len(report.get('owner_bindings',[]))==2 and all(
            r.get('source_sha256')==VAE_SHA and r.get('snapshot_fields')==list(FIELDS)
            for r in report['owner_bindings']),'native VAE proof source/weights/cache fields differ')
    samples=report.get('samples');require(type(samples) is list and len(samples)==2,'native VAE proof samples missing')
    for sample in samples:
        require(sample.get('warmup_full_trajectories')==2 and sample.get('peak_allocated_bytes',0)>0
            and sample.get('peak_reserved_bytes',-1)>=sample['peak_allocated_bytes'],
            'native VAE proof warmup/memory differs')
        rows=sample.get('rows');require(type(rows) is list and len(rows)==len(DECISIONS),'native VAE proof incomplete')
        for i,(row,accepted) in enumerate(zip(rows,DECISIONS)):
            receipt=row.get('receipt',{})
            require(row.get('chunk_index')==i and row.get('accepted_draft') is accepted
                and row.get('float_rgb_exact') is True and row.get('pixels_exact') is True
                and row.get('rng_unchanged') is True and row.get('all_decoder_fields_exact')==list(FIELDS)
                and row.get('rgb_frames')==(9 if i==0 else 12)
                and receipt.get('selected_decode_calls')==int(not accepted),
                'native VAE proof does not cover each fixed-route RGB/cache/RNG comparison')
    require(any(x.get('kind')=='kernel' and x.get('cuda_us',0)>0 for x in report.get('kernels',[])),
            'native VAE proof has no actual CUDA kernels')
    return dict(schema='sdvg_native_vae_prerequisite_v1',run_dir=str(run_dir),artifact_index_sha256=index_sha256,
        report_sha256=_sha256_file(path)[0],source_commit=NATIVE_GATE_COMMIT,marker=marker,
        scope='decoder transaction only; actual reward/RPC trial still required')


def process_factory(argv,**kwargs):
    require(os.environ.get(ENV)=='1' and os.environ.get(TRACKING_ENV)=='1'
        and list(argv[1:3])==['-m','cfexp.correction_worker'],
        'selected SDVG launcher requires explicit standard-worker tracking/feature entry')
    require('env' in kwargs and isinstance(kwargs['env'],dict),'selected SDVG worker environment missing')
    command=list(argv);command[2]='cfexp.sdvg_accepted_worker'
    kwargs=dict(kwargs,env=dict(kwargs['env'],**{ENV:'1'}))
    return subprocess.Popen(command,**kwargs)


def launch_draft(**kwargs):
    from .native_sdvg import launch_conditioned_draft_worker
    kwargs['process_factory']=process_factory
    client=launch_conditioned_draft_worker(**kwargs)
    try:return AcceptedDraftClient(client)
    except BaseException:client.close();raise


def launch_target(**kwargs):
    from .native_sdvg import launch_conditioned_target_worker
    kwargs['process_factory']=process_factory
    return launch_conditioned_target_worker(**kwargs)


def main(argv=None):
    from . import main_table_sdvg_campaign as campaign
    from .main_table_baseline_campaign import _run_dir
    from .metadata import find_git_root,collect_git_metadata
    from .stage_timing import _unsealed_directory
    from .accepted_stream import single_gpu_enabled
    import sys
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--native-vae-gate-run',type=Path,required=True)
    parser.add_argument('--native-vae-gate-index-sha256',required=True)
    extra,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    require(os.environ.get(ENV)=='1' and os.environ.get(TRACKING_ENV)=='1' and os.environ.get(PROFILE)=='1'
        and os.environ.get('CFEXP_STAGE_TIMING')=='1',
            'selected-history SDVG entry currently permits only tracked bounded integration trials')
    args=campaign._build_parser().parse_args(rest)
    require(args.smoke_prompt_id is not None and single_gpu_enabled() and args.duration_seconds in (15,30),
            'selected-history integration requires one explicit prompt and one GPU')
    proof=validate_native_gate(extra.native_vae_gate_run,extra.native_vae_gate_index_sha256)
    root=_unsealed_directory(_run_dir());repo=find_git_root(Path.cwd())
    config=dict(method='SDVG selected-history scoring VAE',scoring_vae_policy=POLICY,
        phase='bounded_reward_rpc_integration_not_main',git_sha=collect_git_metadata(repo)['sha'],
        physical_device_count=1,rank=0,batch_size=1,resolution=[480,832],fps=24,
        generation_seed=int(os.environ['CFEXP_GENERATION_SEED']),prompt_id=args.smoke_prompt_id,
        nominal_seconds=args.duration_seconds,chunk_count=args.duration_seconds*2,threshold=-.7,
        root_forced_target=True,draft_diffusion_history='independent proposals',
        target_diffusion_history='accepted outputs',scoring_vae_history='accepted outputs',
        native_vae_gate=proof,native_kv=True,unchanged_sampler=True,
        warmup_generator_forwards=2,main_quality_or_speedup_claim=False,
        selected_commit_inside_generation_timing=True,selected_scoring_decode_and_output_renderer_separate=True,
        optimizer='not applicable: inference',backward='not applicable: inference',
        loss='not applicable: inference',learning_rate='not applicable: inference',gradient_norm='not applicable: inference',
        precision='native BF16 generator/VAE and pinned reward checkpoint configuration',
        qualitative_comparison='fixed prompt/seed, native videos and prompt provenance below run directory',
        distributed_communication='not applicable: one physical device')
    for label,path in [('draft',args.draft_artifact_spec),('target',args.target_artifact_spec),('reward',args.sdvg_artifact_spec)]:
        config[label+'_artifact_spec_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    write_new(root/'provenance'/'sdvg-selected-vae-policy.json',config)
    import wandb
    directory=root/'tracking'/'wandb';directory.mkdir(parents=True,exist_ok=True)
    mode=tracking_mode(os.environ);fallback=False
    kwargs=dict(project='collaborative-forcing-reproduction',name=root.name,dir=str(directory),config=config,
        save_code=False,settings=wandb.Settings(init_timeout=25,disable_git=True,disable_code=True,
                                             symlink=False,console='off',_disable_stats=True))
    try:run=wandb.init(mode=mode,**kwargs)
    except Exception:
        if mode!='online':raise
        mode='offline';fallback=True;run=wandb.init(mode=mode,**kwargs)
    write_new(root/'provenance'/'tracking'/'session.json',dict(schema='wandb_session_v1',mode=mode,
        run_id=run.id,config=config,online_fallback_to_offline=fallback,synced_evidence_verified=False))
    original=campaign.generate_sdvg_trajectory;ok=False
    campaign.generate_sdvg_trajectory=partial(generate_selected_vae_trajectory,original=original)
    try:
        options=dict(vars(args))
        for key in ('draft_artifact_spec','target_artifact_spec','sdvg_artifact_spec'):
            options[key+'_path']=options.pop(key)
        result=campaign.run_campaign(**options,draft_launcher=launch_draft,target_launcher=launch_target)
        require_profile_samples(root,config=config,result=result);log_result(run,root,result)
        ok=True;return 0
    finally:
        campaign.generate_sdvg_trajectory=original
        run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
        write_new(root/'provenance'/'tracking'/'finished.json',dict(schema='wandb_finish_v1',mode=mode,
            execution_succeeded=ok,synced_evidence_verified=False))


if __name__=='__main__':raise SystemExit(main())
