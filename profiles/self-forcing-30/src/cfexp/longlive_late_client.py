"""Coordinator bindings for the opt-in late-handoff CUDA gate."""
from __future__ import annotations

from .longlive_late_handoff import POLICIES, require
from .longlive_late_worker import validate_proposal


def base_inputs(request):
    return dict(prompt=request.prompt.prompt,chunk_index=request.history.next_chunk_index,
        internal_seed=request.noise.descriptor.derived_seed,
        history_latent_paths=[x.latent_path for x in request.history.segments],
        initial_noise_path=request.noise.tensor_path)


def validate_input_binding(binding, request):
    expected=base_inputs(request)
    require(all(binding.get(k)==expected[k] for k in ('prompt','chunk_index','internal_seed'))
        and binding.get('files')==expected['history_latent_paths']+[expected['initial_noise_path']]
        and binding.get('hashes')==[x.latent_sha256 for x in request.history.segments]+
            [request.noise.descriptor.tensor_sha256],
        'late response used another accepted history or initial noise')


def check_tensor(client, metadata, *, path, key):
    from .correction_runtime import _run_file,_sha256_file,NOISE_SHAPE
    actual=_run_file(client.run_dir,path,must_exist=True)
    require(metadata.get('key')==key and metadata.get('path')==path
        and metadata.get('shape')==list(NOISE_SHAPE) and metadata.get('dtype')=='bfloat16'
        and _sha256_file(actual)==(metadata.get('sha256'),metadata.get('nbytes')),
        'late tensor output does not match its published bytes')


def propose(client, request, *, session_id, policy, renoise_paths, output_path, verify_parity):
    from .correction_runtime import _run_file,_sha256_file
    require(client.role=='draft' and policy in POLICIES, 'invalid late proposer')
    b=base_inputs(request)
    value=client._request('late_propose',dict(b,session_id=session_id,policy=policy,
        renoise_paths=list(renoise_paths),output_paths={'noisy_latent':output_path},verify_parity=verify_parity))
    validate_proposal(value)
    binding=value['input_binding']
    validate_input_binding(binding,request)
    files=b['history_latent_paths']+[b['initial_noise_path']]
    require({k:binding[k] for k in ('prompt','chunk_index','internal_seed')}==
                {k:b[k] for k in ('prompt','chunk_index','internal_seed')}
        and binding['files']==files
        and binding['hashes']==[x.latent_sha256 for x in request.history.segments]+[request.noise.descriptor.tensor_sha256]
        and binding['renoise_paths']==list(renoise_paths)
        and binding['renoise_sha256']==[_sha256_file(_run_file(client.run_dir,p,must_exist=True))[0] for p in renoise_paths],
        'late proposal did not use the requested history and explicit noises')
    check_tensor(client,value['result']['noisy_latent'],path=output_path,key='noisy_latent')
    return value


def target(client, request, proposal, *, verify_parity):
    require(client.role=='target', 'late prediction requires the Target client')
    validate_proposal(proposal)
    validate_input_binding(proposal['input_binding'],request)
    value=client._request('late_target',dict(proposal=proposal,
        output_paths={'latent':request.output_latent_path},verify_parity=verify_parity))
    require(value.get('schema')=='longlive_late_target_v1' and value.get('proposal')==proposal,
            'late Target receipt belongs to another proposal')
    check_tensor(client,value['result']['latent'],path=request.output_latent_path,key='latent')
    require(value['result']['denoising_schedule']==[POLICIES[proposal['input_binding']['policy']][1]]
        and value['result']['completed_four_step_target_chunk'] is False,
        'late Target response mislabels its sampling policy')
    return value


def commit(client, *, session_id, proposal, target_receipt, verify_parity):
    require(client.role=='draft', 'late B3 commit requires Draft')
    result=client._request('late_commit',dict(session_id=session_id,pending_token=proposal['pending_token'],
        target=target_receipt,verify_parity=verify_parity))
    n=proposal['input_binding']['chunk_index'];value=result.get('result',{})
    require(result.get('schema')=='longlive_late_commit_v1'
        and result.get('accepted_latent_sha256')==target_receipt['result']['latent']['sha256']
        and value.get('draft_clean_forwards')==1 and value.get('history_replay_chunks')==0
        and value.get('committed_kv_width')==3 and value.get('committed_chunks')==n+1
        and value.get('branch_snapshot_bytes')==0
        and value.get('forward_trace')==[dict(timesteps=[0.]*3,current_start=n*4680)],
        'late Target was not committed through one native Draft B3 update')
    return result


def generated(client, request, result):
    from .correction_state import GenerationContext,KVCacheDescriptor,LatentSegment
    from .insight2_campaign import GeneratedChunk
    from .self_forcing_frame_adapter import validate_effective_rank_evidence
    check_tensor(client,result['latent'],path=request.output_latent_path,key='latent')
    if client.role=='draft':
        validate_effective_rank_evidence(result['rank_evidence'],expected_rank=result['rank'])
    context=GenerationContext(role=client.role,model_id=client.model_id,
        model_artifact_sha256=client.model_artifact_sha256,
        history_head_sha256=request.history.head_sha256,next_chunk_index=request.history.next_chunk_index,
        noise=request.noise.descriptor,
        kv=KVCacheDescriptor(owner_role=client.role,model_id=client.model_id,
            model_artifact_sha256=client.model_artifact_sha256,history_head_sha256=request.history.head_sha256,
            cache_sha256=result['kv_witness_sha256']))
    segment=LatentSegment.from_file(history=request.history,context=context,
        latent_root=client.run_dir,latent_path=request.output_latent_path)
    return GeneratedChunk(context=context,segment=segment,effective_rank=result.get('rank'),
        hidden_state_path=None,hidden_state_sha256=None,
        diagnostics={k:v for k,v in result.items() if k!='latent'})


def continue_draft(client, request, *, session_id):
    require(client.role=='draft', 'late continuation requires Draft')
    b=base_inputs(request)
    value=client._request('late_continue',dict(b,session_id=session_id,
        output_paths={'latent':request.output_latent_path}))
    require(value.get('schema')=='longlive_late_following_v1'
        and value.get('input_binding',{}).get('prompt')==b['prompt']
        and value['input_binding'].get('chunk_index')==b['chunk_index']
        and value['input_binding'].get('internal_seed')==b['internal_seed'],
        'late continuation request binding differs')
    validate_input_binding(value['input_binding'],request)
    return generated(client,request,value['result'])
