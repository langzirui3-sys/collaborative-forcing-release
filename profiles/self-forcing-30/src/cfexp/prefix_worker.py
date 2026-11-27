"""Explicit intermediate-latent RPCs; partial Target is never a final chunk."""
from __future__ import annotations
from collections.abc import Mapping
from .correction_worker import (WorkerProtocolError, WorkerExecutionError, _exact_keys,
    _integer, _input_path, _sha256_file, _json_value, _publish_staged, _discard_staged)
from .target_prefix import validate_prefix_trace,validate_draft_suffix_evidence


def check(ok,message):
    if not ok:raise WorkerProtocolError(message)


def file_hash(worker,path):
    return _sha256_file(_input_path(worker.run_dir,path,label='handoff input')[1])[0]


def generate_prefix(worker,request):
    _exact_keys(request,worker._COMMON_KEYS|{'prompt','chunk_index','internal_seed',
        'history_latent_paths','initial_noise_path','output_paths','verify_parity'},label='Target prefix')
    check(worker.role=='target','only Target can generate a Target prefix')
    prompt=request['prompt'];check(isinstance(prompt,str) and bool(prompt.strip()),'invalid prompt')
    n=_integer(request['chunk_index'],label='chunk_index');check(n<worker._chunk_count,'invalid chunk')
    seed=_integer(request['internal_seed'],label='internal_seed')
    verify=request['verify_parity'];check(type(verify) is bool,'verify_parity must be boolean')
    history,paths=worker._history(request['history_latent_paths'],expected_count=n)
    noise_path,noise=worker._load_input_tensor(request['initial_noise_path'],label='initial noise',key='noise')
    outputs=worker._output_map(request['output_paths'],required={'first_clean_prediction'})
    files=[*paths,noise_path]
    check(not set(files).intersection(p[0] for p in outputs.values()),'prefix output aliases input')
    binding={'prompt':prompt,'chunk_index':n,'internal_seed':seed,'files':files,
             'hashes':[file_hash(worker,p) for p in files]}
    staged=[]
    try:
        fn=worker.backend.check_first_update if verify else worker.backend.generate_first_update
        result=fn(prompt=prompt,history_latents=history,initial_noise=noise,
                  internal_seed=seed,chunk_index=n)
        if not isinstance(result,Mapping) or result.get('completed_target_chunk') is not False or 'latent' in result:
            raise WorkerExecutionError('Target prefix cannot be a completed chunk')
        validate_prefix_trace(result['forward_trace'],history_frames=n*3)
        value=result['first_clean_prediction']
        if not worker.codec.is_tensor(value):raise WorkerExecutionError('missing Target first prediction')
        relative,target=outputs['first_clean_prediction']
        item=worker._stage_tensor(relative=relative,target=target,key='first_clean_prediction',value=value)
        staged.append(item)
        normalized=_json_value({k:v for k,v in result.items() if k!='first_clean_prediction'},label='prefix evidence')
        normalized['first_clean_prediction']=item.metadata
        _publish_staged(staged)
        return {'schema':'target_first_update_handoff_v1','input_binding':binding,'result':normalized}
    except BaseException as error:
        _discard_staged(staged);raise worker._poison(error) from error


def resume_prefix(worker,request):
    _exact_keys(request,worker._COMMON_KEYS|{'early_session_id','pending_token','prefix',
        'renoise_paths','output_paths','verify_parity'},label='Draft suffix')
    pending=getattr(worker,'_early_pending',None)
    check(worker.role=='draft' and pending is not None,'no pending early Draft block')
    check(request['early_session_id']==pending['session_id'] and request['pending_token']==pending['token'],
          'prefix commit capability mismatch')
    verify=request['verify_parity'];check(type(verify) is bool,'verify_parity must be boolean')
    prefix=request['prefix']
    check(isinstance(prefix,Mapping) and set(prefix)=={'schema','input_binding','result'}
          and prefix['schema']=='target_first_update_handoff_v1','invalid Target prefix receipt')
    binding={k:pending[k] for k in ('prompt','chunk_index','internal_seed','files','hashes')}
    check(prefix['input_binding']==binding,'Target prefix used a different history, noise or seed')
    for path,digest in zip(pending['files'],pending['hashes']):
        check(file_hash(worker,path)==digest,'pending early input changed')
    result=prefix['result'];check(isinstance(result,Mapping) and result.get('completed_target_chunk') is False,
                                'Target prefix is not intermediate')
    validate_prefix_trace(result['forward_trace'],history_frames=pending['chunk_index']*3)
    meta=result['first_clean_prediction']
    check(isinstance(meta,Mapping) and meta.get('key')=='first_clean_prediction','invalid prefix tensor metadata')
    path,prediction=worker._load_input_tensor(meta['path'],label='Target prefix',key='first_clean_prediction')
    check(file_hash(worker,path)==meta['sha256'],'Target prefix bytes changed')
    noises=request['renoise_paths'];check(isinstance(noises,list) and len(noises)==3,'three explicit noises required')
    loaded=[worker._load_input_tensor(p,label='suffix noise',key='noise') for p in noises]
    all_inputs=[*pending['files'],path,*[p for p,_ in loaded]]
    check(len(set(all_inputs))==len(all_inputs),'handoff tensor inputs must be distinct')
    outputs=worker._output_map(request['output_paths'],required={'latent'})
    check(not set(all_inputs).intersection(p[0] for p in outputs.values()),'suffix output aliases input')
    history,paths=worker._history(pending['history_paths'],expected_count=pending['chunk_index'])
    try:
        suffix=worker.backend.resume_early_target_prefix(prompt=pending['prompt'],history_latents=history,
            history_kv_widths=tuple(pending['widths']),first_clean_prediction=prediction,
            renoise_first=loaded[0][1],renoise_second=loaded[1][1],renoise_third=loaded[2][1],verify_parity=verify)
        validate_draft_suffix_evidence(suffix)
        if (suffix.get('draft_clean_forwards')!=1
                or suffix.get('history_replay_chunks')!=0 or suffix.get('branch_snapshot_bytes')!=0
                or suffix.get('committed_kv_width')!=3
                or suffix.get('committed_chunks')!=pending['chunk_index']+1):
            raise WorkerExecutionError('Draft suffix omitted native execution evidence')
        suffix={**suffix,'target_prefix':dict(prefix),'explicit_renoise':[
            {'path':p,'sha256':file_hash(worker,p)} for p,_ in loaded]}
        payload=worker._finalize_generate_result(request=request,target_attempt_id=None,
            target_logical_call_id=None,outputs=outputs,history=history,history_paths=paths,
            noise_path=pending['files'][-1],result=suffix,capture_hidden=False,chunk_index=pending['chunk_index'])
        worker._early_pending=None;worker._early_session['next']+=1
        return payload
    except BaseException as error:
        raise worker._poison(error) from error
