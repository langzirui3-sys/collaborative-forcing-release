"""Full paper128 SDVG replication after sealed native/reward/cache/profile gates.

This is a distinct entry; historical A915 outputs and bounded integration
manifests are unchanged. Profiling is a prerequisite, never the main latency.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

from .sdvg_accepted_vae import ENV,POLICY
from .sdvg_accepted_client import generate_selected_vae_trajectory
from .tracked_sdvg_accepted import launch_draft,launch_target
from .tracked_sdvg import (ENV as TRACKING_ENV,PROFILE,tracking_mode,write_new,
                          materialize_sdk_log_links,log_result)
from .sdvg_selected_profile_proof import validate_profiles
from .sdvg_vae_transaction import require
from . import paper_population as population


def artifact_tracking_fields(label,path):
    raw=path.read_bytes();value=json.loads(raw)
    result={label+'_artifact_spec_sha256':hashlib.sha256(raw).hexdigest(),
            label+'_artifact_definition':value}
    for name in ('adapter_config','backend_config'):
        if name in value:result[label+'_'+name]=value[name]
    return result


def workload(args,repo):
    from .accepted_stream import single_gpu_enabled
    from .baseline_population import model_pair_for_draft
    require(os.environ.get(ENV)=='1' and os.environ.get(TRACKING_ENV)=='1'
        and os.environ.get(PROFILE)=='0' and os.environ.get('CFEXP_STAGE_TIMING')=='0'
        and population.enabled() and single_gpu_enabled()
        and args.smoke_prompt_id is None and args.duration_seconds in (15,30)
        and args.baseline_population_v2 and args.record_video_ready_timing and args.overlap_native_vae
        and args.prompt_shard_count==1 and args.prompt_shard_index==0,
        'selected SDVG main requires an unprofiled full paper128 single-GPU shard')
    require(os.environ.get('CFEXP_PAPER_PROMPT_SHARD_COUNT')=='4'
        and os.environ.get('CFEXP_PAPER_PROMPT_SHARD_INDEX') in ('0','1','2','3'),
        'selected SDVG main requires one of the four original paper shards')
    prompts,binding=population.load(repo,model_pair=model_pair_for_draft(args.draft_model_id))
    require(len(prompts)==32 and binding['seed'] in population.SEEDS,
            'selected SDVG main population/seed differs')
    return prompts,binding


def main(argv=None):
    from . import main_table_sdvg_campaign as campaign
    from .metadata import find_git_root,collect_git_metadata
    from .stage_timing import _unsealed_directory
    import sys
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--profile-runs-root',type=Path,required=True)
    extra,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    args=campaign._build_parser().parse_args(rest);repo=find_git_root(Path.cwd())
    prompts,binding=workload(args,repo)
    root=_unsealed_directory(Path(os.environ['CFEXP_RUN_DIR']))
    before=time.perf_counter();proof=validate_profiles(repo=repo,runs_root=extra.profile_runs_root)
    validation_seconds=time.perf_counter()-before
    write_new(root/'provenance'/'sdvg-selected-profile-admission.json',proof)
    config=dict(method='sdvg',phase='main',scoring_vae_policy=POLICY,
        git_sha=collect_git_metadata(repo)['sha'],model_pair=args.draft_model_id,
        generation_seed=binding['seed'],source_population_sha256=population.SOURCE_SHA256,
        prompt_shard=binding['shard_index'],nominal_seconds=args.duration_seconds,
        physical_device_count=1,rank=0,batch_size=1,output_resolution=[480,832],output_fps=24,
        chunk_count=args.duration_seconds*2,root_target=True,threshold=-.7,
        draft_history='independent proposals',target_history='accepted outputs',
        scoring_vae_history='accepted outputs',warmup_forward_calls_per_worker=2,native_kv=True,
        selected_commit_inside_generation_timing=True,selected_scoring_decode_and_output_renderer_separate=True,
        prerequisite_verification_seconds=validation_seconds,
        prerequisite_verification_scope='once before model loading; not part of warm video-ready latency',
        profile_sources=[{k:p[k] for k in ('run_id','git_sha','artifact_index_sha256')} for p in proof['profiles']],
        profile_protocol_sha256=proof['prerequisite_protocol_sha256'],
        no_imputed_trajectory_target_latencies=True,
        optimizer='not applicable: inference',backward='not applicable: inference',loss='not applicable: inference',
        learning_rate='not applicable: inference',gradient_norm='not applicable: inference',
        precision='model-specific; frozen artifact adapter_config is authoritative',
        qualitative_comparison='original fixed prompts/seeds; selected MP4/hash and paired-reference metadata remain on server',
        distributed_communication='not applicable: one physical GPU; no model or media transfer between devices')
    for label,path in [('draft',args.draft_artifact_spec),('target',args.target_artifact_spec),('sdvg',args.sdvg_artifact_spec)]:
        config.update(artifact_tracking_fields(label,path))
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
        run_id=run.id,directory='tracking/wandb',online_fallback_to_offline=fallback,
        online_session_url=run.url if mode=='online' else None,synced_evidence_verified=False,config=config))
    original=campaign.generate_sdvg_trajectory;ok=False
    campaign.generate_sdvg_trajectory=partial(generate_selected_vae_trajectory,original=original)
    try:
        options=dict(vars(args))
        for key in ('draft_artifact_spec','target_artifact_spec','sdvg_artifact_spec'):
            options[key+'_path']=options.pop(key)
        result=campaign.run_campaign(**options,draft_launcher=launch_draft,target_launcher=launch_target)
        log_result(run,root,result);ok=True;return 0
    finally:
        campaign.generate_sdvg_trajectory=original
        run.finish(exit_code=0 if ok else 1);materialize_sdk_log_links(directory)
        write_new(root/'provenance'/'tracking'/'finished.json',dict(schema='wandb_finish_v1',mode=mode,
            execution_succeeded=ok,synced_evidence_verified=False,
            note='SDK/profiler files stay on server; synchronization is not independently verified.'))


if __name__=='__main__':raise SystemExit(main())
