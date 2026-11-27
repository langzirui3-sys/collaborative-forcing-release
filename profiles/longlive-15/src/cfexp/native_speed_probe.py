"""One-input native video-ready cost probe; no quality or routing claim.

Fixed Target root + blocks 8..11 isolates intervention/restart cost without
fitting or importing calibration. It is not an eRank method result.
"""
import argparse
import json
import os
from pathlib import Path
import time

from .baseline_population import load_baseline_population, model_pair_for_draft
from .correction_artifacts import verify_target_artifact_spec, target_artifact_spec_sha256
from .correction_runtime import launch_correction_runtime
from .duration_sweep import duration_profile
from .insight2_campaign import Insight2Campaign, PromptCase
from .insight2_protocol import load_protocol
from .insight2_run import load_run_config
from .main_table_baseline_campaign import (_run_dir, _model_root, _visible_devices,
    _sha256_file, _write_new_json, verify_artifact_spec)
from .main_table_full_target import generate_full_target_trajectory
from .main_table_ours_campaign import _assert_idle_pair, _assert_worker_pair_owns_gpus
from .metadata import find_git_root, collect_git_metadata
from .rolling_campaign import prepare_launch, verify_draft_spec
from .stage_timing import _unsealed_directory
from .stage_timing_probe import _hardware
from .erank_acceptance_pilot import bound_file

SCOPE='native_fixed_route_warm_models_cached_text_video_ready_v1'
RENDER_SCOPE='native_fixed_route_warm_models_cached_text_coupled_vae_ready_v1'


def run(args):
    if os.environ.get('CFEXP_STAGE_TIMING') != '1':
        raise ValueError('cost probe requires stage timing; not a production benchmark')
    if args.model not in ('longlive','rolling') or args.arm not in ('fixed_burst','target'):
        raise ValueError('unsupported native cost probe')
    render_mode = getattr(args, 'render_mode', 'legacy')
    from .native_vae_component import selected_model, validate_render_flags
    native_render = selected_model()
    if render_mode not in ('legacy','serial','overlap'):
        raise ValueError('unknown native rendering mode')
    render_kwargs = {}
    if render_mode != 'legacy':
        if native_render != args.model:
            raise ValueError('native render probe model binding differs')
        validate_render_flags()
        from .accepted_stream_worker import process_factory
        render_kwargs['process_factory'] = process_factory
    elif native_render is not None:
        raise ValueError('native rendering mode must be explicitly selected')
    from .longlive_host_cursors import enabled as host_cursor_enabled
    host_cursors = host_cursor_enabled()
    if host_cursors and (args.model, args.arm) != ('longlive', 'fixed_burst'):
        raise ValueError('LongLive host cursor cost probe requires its fixed-burst arm')
    from .rolling_host_cursors import enabled as rolling_cursor_enabled
    rolling_cursors = rolling_cursor_enabled()
    if rolling_cursors and (args.model,args.arm) != ('rolling','fixed_burst'):
        raise ValueError('Rolling host cursor cost probe requires its fixed-burst arm')
    from .rolling_forcing_session import cpu_history_enabled
    rolling_cpu_history = cpu_history_enabled()
    if rolling_cpu_history and (args.model, args.arm) != ('rolling', 'fixed_burst'):
        raise ValueError('Rolling CPU history cost probe requires its fixed-burst arm')
    root=_unsealed_directory(_run_dir());models=_model_root();repo=find_git_root(Path.cwd())
    if (root/'result.json').exists():raise ValueError('fresh diagnostic required')
    profile=duration_profile(args.seconds);devices=_visible_devices()
    _assert_idle_pair(*devices)
    hardware=_hardware(devices)
    model_id='longlive_1.3b' if args.model=='longlive' else 'rolling_forcing_1.3b'
    prompts,population=load_baseline_population(repo,model_pair=model_pair_for_draft(model_id))
    prompt=prompts[0];case=PromptCase(prompt['id'],prompt['text'],'evaluation')
    ds=repo/f'experiments/erank3-{args.model}-policy-v1.artifacts.json'
    if args.model=='rolling':ds=repo/'experiments/erank3-rolling-native-v1.artifacts.json'
    ts=repo/'experiments/insight2-krea14b.artifacts.json'
    if args.model=='rolling':
        _,dv,dm=verify_draft_spec(ds,model_root=models,repo_root=repo)
    else:
        _,dv,dm=verify_artifact_spec(ds,method_id='draft_only',model_root=models,repository_root=repo)
    _,tv,tm=verify_target_artifact_spec(ts,model_root=models,repo_root=repo)
    expected='cfexp.longlive_adapter' if args.model=='longlive' else 'cfexp.rolling_policy_worker'
    if dm!=expected or tm!='cfexp.krea_chunk_backend':raise ValueError('wrong native backend')
    conditioning=[{'prompt_id':prompt['id'],'prompt':prompt['text']}]
    extra=({'rolling':prepare_launch(run_dir=root,repo_root=repo,model_root=models,
        prompts=conditioning,seeds=(20260911,),chunk_count=profile.chunk_count)}
        if args.model=='rolling' else {})
    runtime=launch_correction_runtime(run_dir=root,repo_root=repo,
        draft_artifact_spec_path=ds,draft_model_id=model_id,
        draft_artifact_spec_sha256=_sha256_file(ds)[0],draft_verified_artifacts=dv,
        target_artifact_spec_path=ts,target_artifact_spec_sha256=target_artifact_spec_sha256(ts),
        target_verified_artifacts=tv,devices=devices,write_full_videos=True,
        chunk_count=profile.chunk_count,conditioning_prompts=conditioning,
        shared_precomputed_conditioning=True,**extra,**render_kwargs)
    selected=(8,9,10,11)
    try:
        _assert_worker_pair_owns_gpus(runtime=runtime,draft_device=devices[0],target_device=devices[1],phase='native-cost-probe')
        # Protocol/config loading is before ready time, just like model/text preparation.
        config=load_run_config(repo/f'experiments/erank3-{args.model}-boundary-v1.config.json',repo_root=repo)
        protocol=load_protocol(config.protocol_path)
        campaign=Insight2Campaign(run_dir=root,protocol_id='native-fixed-route-cost-v1',
            seeds=(20260911,),calibration_prompts=tuple(PromptCase(str(x['prompt_id']),str(x['prompt']),'calibration') for x in protocol.calibration_prompts),
            evaluation_prompts=(case,),draft=runtime.draft,target=runtime.target,
            noise_provider=runtime.noise_provider,evaluator=runtime.evaluator,
            chunk_count=profile.chunk_count)
        decoder = None
        if render_mode != 'legacy':
            from .accepted_stream_client import StreamDecoderProxy
            renderer = runtime.draft if args.arm == 'target' else runtime.target
            decoder = StreamDecoderProxy(runtime.draft, renderer, root, chunk_count=profile.chunk_count)
            runtime.evaluator.client = decoder
        callbacks = (dict(on_commit=decoder.append, before_generate=decoder.before_generate)
                     if render_mode == 'overlap' else {})
        started=time.perf_counter()
        if args.arm=='target':
            generated=generate_full_target_trajectory(prompt_id=case.prompt_id,prompt=case.prompt,
                target=runtime.target,noise_provider=runtime.noise_provider,
                chunk_count=profile.chunk_count,duration_seconds=args.seconds,
                **({'on_commit':decoder.append} if render_mode == 'overlap' else {}))
            history=generated.history
        else:
            key,_,accepted,history=campaign._materialize_fixed_positions(prompt=case,seed=20260911,
                corrected_chunks=selected,bypass_draft_on_target_chunks=True,**callbacks)
            if tuple(accepted)!=selected:raise ValueError('fixed diagnostic route drifted')
        generation=time.perf_counter()-started
        if render_mode == 'serial':
            for count in range(1, history.next_chunk_index+1):
                decoder.append(history.prefix_before(count))
        prepared=runtime.evaluator.prepare_full(history=history,measurement_id='native-cost-'+args.arm)
        ready=time.perf_counter()-started
        if decoder is not None and decoder.last_evidence is None:
            raise ValueError('native renderer did not produce a joined execution receipt')
        _assert_worker_pair_owns_gpus(runtime=runtime,draft_device=devices[0],target_device=devices[1],phase='native-cost-probe-complete')
        _write_new_json(root/'provenance/history.json',history.to_dict())
        video=bound_file(root,root/prepared.video_path)
        decode=bound_file(root,root/prepared.decode_evidence_path)
    finally:
        runtime.close()
    _assert_idle_pair(*devices)
    host_audit = None
    if host_cursors:
        from .longlive_host_cursors import audit_checkpoints
        host_audit = audit_checkpoints(campaign, key, profile.chunk_count)
    rolling_audit = None
    if rolling_cursors:
        from .rolling_host_cursors import audit_checkpoints
        rolling_audit = audit_checkpoints(campaign,key,profile.chunk_count)
    cpu_history_audit = None
    if rolling_cpu_history:
        receipts = []
        previous_draft = None
        for index in range(1, profile.chunk_count):
            if index in selected:
                previous_draft = None
                continue
            checkpoint = campaign._checkpoint_path('rollouts', key, index)
            generated = json.loads(checkpoint.read_text())['generated']
            proof = generated['diagnostics']['rolling_state']['cpu_history_identity']
            if (generated['context']['role'] != 'draft'
                    or generated['context']['next_chunk_index'] != index
                    or proof['schema'] != 'rolling_cpu_history_identity_v1'
                    or proof['history_blocks_transferred'] != (index if previous_draft is None else 0)
                    or (previous_draft is not None and proof['input_sha256'] != previous_draft)):
                raise ValueError('Rolling CPU identity differs from the accepted fixed route')
            previous_draft = proof['output_sha256']
            receipts.append({'chunk_index': index, 'execution': proof,
                             'checkpoint': bound_file(root, checkpoint)})
        cpu_history_audit = {'schema': 'rolling_cpu_history_probe_audit_v1', 'rows': receipts}
    thread_audit = None
    if os.environ.get('OMP_NUM_THREADS') == '4' and os.environ.get('MKL_NUM_THREADS') == '4':
        from .native_cpu_budget import audit
        thread_audit = audit(root,chunk_count=profile.chunk_count,arm=args.arm)
    result={'schema':'native_fixed_route_cost_probe_v1','status':'succeeded',
        'formal_speedup_claim':False,'quality_measured':False,'erank_routing_evaluated':False,
        'model':model_id,'arm':args.arm,'seed':20260911,'prompt':prompt,'population_binding':population,
        'duration':profile.metadata(),'hardware':hardware,'code_revision':collect_git_metadata(repo)['sha'],
        'timing_scope':SCOPE if render_mode == 'legacy' else RENDER_SCOPE,
        'generation_wall_seconds':generation,'video_ready_wall_seconds':ready,
        'target_chunks':list(range(profile.chunk_count)) if args.arm=='target' else [0,*selected],
        'video':video,'decode_evidence':decode,'history':bound_file(root,root/'provenance/history.json'),
        **({'longlive_host_cursor_audit':host_audit} if host_audit is not None else {}),
        **({'rolling_host_cursor_audit':rolling_audit} if rolling_audit is not None else {}),
        **({'rolling_cpu_history_audit': cpu_history_audit} if cpu_history_audit is not None else {}),
        **({'cpu_thread_audit':thread_audit} if thread_audit is not None else {}),
        **({'native_render':{'mode':render_mode,'execution':decoder.last_evidence,
            'generation_kv_semantics_unchanged':True,'final_encoder_drain_in_ready_time':True}}
           if decoder is not None else {})}
    _write_new_json(root/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=('longlive','rolling'))
    p.add_argument('--arm',required=True,choices=('fixed_burst','target'))
    p.add_argument('--seconds',required=True,type=int,choices=(15,30,60))
    p.add_argument('--render-mode',default='legacy',choices=('legacy','serial','overlap'))
    args=p.parse_args()
    try:run(args)
    finally:
        from .runtime_helpers import shutdown_owned_runtime_helpers
        shutdown_owned_runtime_helpers(label='native-cost-probe')

if __name__=='__main__':main()
