"""Coordinator validation for the experimental single-update handoff."""
from __future__ import annotations
from collections.abc import Mapping
from .early_protocol import StoppedDraft
from .target_prefix import validate_draft_suffix_evidence
from .correction_state import KVCacheDescriptor,GenerationContext,LatentSegment
from .insight2_campaign import GeneratedChunk


def target_prefix(client,request,*,output_path,verify_parity=False):
    from .correction_runtime import _require,_run_file,_sha256_file,NOISE_SHAPE
    _require(client.role=='target','prefix requires Target client')
    binding={'prompt':request.prompt.prompt,'chunk_index':request.history.next_chunk_index,
        'internal_seed':request.noise.descriptor.derived_seed,
        'files':[*(x.latent_path for x in request.history.segments),request.noise.tensor_path],
        'hashes':[*(x.latent_sha256 for x in request.history.segments),request.noise.descriptor.tensor_sha256]}
    payload=client._request('generate_target_prefix',{
        'prompt':binding['prompt'],'chunk_index':binding['chunk_index'],'internal_seed':binding['internal_seed'],
        'history_latent_paths':binding['files'][:-1],'initial_noise_path':binding['files'][-1],
        'output_paths':{'first_clean_prediction':output_path},'verify_parity':verify_parity})
    _require(payload.get('schema')=='target_first_update_handoff_v1'
             and payload.get('input_binding')==binding,'Target prefix input binding mismatch')
    meta=payload['result']['first_clean_prediction']
    digest,size=_sha256_file(_run_file(client.run_dir,output_path,must_exist=True))
    _require(meta.get('path')==output_path and meta.get('sha256')==digest and meta.get('nbytes')==size
             and meta.get('shape')==list(NOISE_SHAPE) and meta.get('dtype')=='bfloat16',
             'Target prefix tensor metadata mismatch')
    _require(payload['result'].get('completed_target_chunk') is False,'Target prefix was labelled complete')
    return payload


def resume_prefix(client,stopped,prefix,*,renoise_paths,verify_parity=False):
    from .correction_runtime import _require,_run_file,_sha256_file,NOISE_SHAPE,_validated_boundary_velocity_witness
    _require(client.role=='draft' and isinstance(stopped,StoppedDraft),'suffix requires stopped Draft')
    request=stopped.request;n=request.history.next_chunk_index
    payload=client._request('resume_early_target_prefix',{'early_session_id':stopped.session_id,
        'pending_token':stopped.pending_token,'prefix':prefix,'renoise_paths':list(renoise_paths),
        'output_paths':{'latent':request.output_latent_path},'verify_parity':verify_parity})
    result=payload.get('result')
    _require(payload.get('chunk_index')==n and payload.get('history_chunk_count')==n
             and isinstance(result,Mapping),'suffix returned wrong chunk')
    meta=result['latent'];digest,size=_sha256_file(_run_file(client.run_dir,request.output_latent_path,must_exist=True))
    _require(meta.get('path')==request.output_latent_path and meta.get('sha256')==digest
             and meta.get('nbytes')==size and meta.get('shape')==list(NOISE_SHAPE)
             and meta.get('dtype')=='bfloat16','suffix tensor binding mismatch')
    validate_draft_suffix_evidence(result)
    _require(result.get('target_prefix')==prefix
             and result.get('history_replay_chunks')==0 and result.get('branch_snapshot_bytes')==0
             and result.get('committed_kv_width')==3 and result.get('committed_chunks')==n+1
             and result.get('draft_clean_forwards')==1,'suffix execution evidence mismatch')
    noises=result.get('explicit_renoise')
    _require(isinstance(noises,list) and [x['path'] for x in noises]==list(renoise_paths),
             'suffix noise paths mismatch')
    for item in noises:
        _require(item['sha256']==_sha256_file(_run_file(client.run_dir,item['path'],must_exist=True))[0],
                 'suffix noise bytes mismatch')
    _validated_boundary_velocity_witness(result,chunk_index=n,history_chunk_count=n,
                                         input_dtype='bfloat16',candidate_frame_count=3)
    kv=KVCacheDescriptor(owner_role='draft',model_id=client.model_id,
        model_artifact_sha256=client.model_artifact_sha256,history_head_sha256=request.history.head_sha256,
        cache_sha256=result['kv_witness_sha256'])
    context=GenerationContext(role='draft',model_id=client.model_id,model_artifact_sha256=client.model_artifact_sha256,
        history_head_sha256=request.history.head_sha256,next_chunk_index=n,noise=request.noise.descriptor,kv=kv)
    segment=LatentSegment.from_file(history=request.history,context=context,latent_root=client.run_dir,
                                    latent_path=request.output_latent_path)
    return GeneratedChunk(context=context,segment=segment,effective_rank=stopped.decision['effective_rank'],
        hidden_state_path=None,hidden_state_sha256=None,diagnostics={
            **{k:v for k,v in result.items() if k!='latent'},'intervention':'T1000-D-native-suffix',
            'rank_origin':'pre_intervention_draft_probe'})
