"""Full trained-model lease/RNG/hidden/cache gate; never a quality benchmark."""
import hashlib
import json

from .branch_kv_state import NativeBranchState, PrefixBinding, payloads_alias
from .incremental_draft import IncrementalDraftExecutor, latent_digest, tensor_bytes
from .longlive_joint_executor import JointLongLiveExecutor
from .longlive_joint_handoff import require
from .longlive_joint_oracle import CASES, SCHEDULE
from .prefix_oracle import check_suffix
from .self_forcing_frame_adapter import OfficialSelfForcingBackend as Observation
from .stage_timing import stage

PREFIX_BOOTSTRAP='cold_parent_plus_final_native_clean_append_v2'


def construct_native_prefix(executor,*,prompt,history,count):
    """Cold oracle setup through the ordinary one-block append contract.

    A cold allocation is intentionally forbidden by commit_prefill. Request the
    prefix before its last known clean block, replay native B3 history including
    that final block, then commit exactly that one append. This seeds only the
    diagnostic oracle; it does not add a cold-prefill production capability.
    """
    t,p=executor.torch,executor.pipeline;tokens=3*p.frame_seq_length
    require(type(count) is int and 0<count<=int(history.shape[1])//3
        and executor.cache.state_count==0 and executor.cache.pinned_binding is None
        and p.kv_cache1 is None and getattr(p,'crossattn_cache',None) is None,
        'gate must construct independent fresh native prefixes')
    scope=hashlib.sha256(('native-committed-v2\0'+executor.scope_sha256+'\0'+prompt).encode()).hexdigest()
    hashes=tuple(latent_digest(history[:,3*i:3*i+3],t) for i in range(count))
    binding=PrefixBinding(scope,tuple(hashlib.sha256((h+':3').encode()).hexdigest() for h in hashes))
    parent=PrefixBinding(scope,binding.chunk_sha256[:-1])
    with executor.cache.checkout(parent) as lease:
        require(lease.state is None,'fresh oracle unexpectedly reused a retained native state')
        lease.reserve_allocation(executor.cache.max_bytes)
        try:
            p._initialize_kv_cache(batch_size=1,dtype=t.bfloat16,device=history.device)
            p._initialize_crossattn_cache(batch_size=1,dtype=t.bfloat16,device=history.device)
            payload={'self':p.kv_cache1,'cross':p.crossattn_cache};size=tensor_bytes(payload,t)
            executor.allocation_bytes=size;lease.note_allocated(size)
            conditional=p.text_encoder(text_prompts=[prompt])
            with t.inference_mode():
                for i in range(count):
                    p.generator(noisy_image_or_video=history[:,3*i:3*i+3],conditional_dict=conditional,
                        timestep=t.zeros((1,3),device=history.device,dtype=t.int64),
                        kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,current_start=i*tokens)
            executor._validate_native_cursor(count)
            lease.commit(NativeBranchState(binding,payload,count,count*tokens,size))
        finally:p.kv_cache1=p.crossattn_cache=None
    return binding


def expected_forward_trace():
    trace=[]
    def add(index,steps):trace.append(dict(current_start=index*4680,latent_frames=len(steps),timesteps=list(steps)))
    for retained,external in CASES:
        n=retained+external
        for _ in range(2):
            for i in range(retained):add(i,[0.]*3)
        for i in range(retained,n):add(i,[0.]*3)
        for step in (*SCHEDULE,0.):add(n,[step]*3)
        add(retained,[0.]*(3*external)+[1000.]*3)
        for step in (*SCHEDULE[1:],0.):add(n,[step]*3)
        for following in range(1,7):
            for _ in range(2):
                for step in (*SCHEDULE,0.):add(n+following,[step]*3)
    return trace


def run(backend,*,prompt,history,noises,report):
    t=backend.torch;session=backend.session;p=session.executor.pipeline
    require(not session.host_cursors.enabled and p.kv_cache1 is None
        and getattr(p,'crossattn_cache',None) is None and session.executor.cache.resident_bytes==0
        and len(noises)==7,'lease gate requires one empty native owner and seven explicit initial noises')
    report['native_prefix_bootstrap_policy']=PREFIX_BOOTSTRAP
    for handle in backend._hook_handles:handle.remove()
    backend._hook_handles.clear()
    joint=JointLongLiveExecutor(session)
    reference=IncrementalDraftExecutor(pipeline=p,torch=t,scope_sha256=joint.scope_sha256,
                                      max_bytes=joint.cache.max_bytes,fork_enabled=False)
    trace=[];hidden=[];current={};handles=[];tokens=3*p.frame_seq_length
    def pre(_m,_a,kw):
        steps=kw['timestep'].detach().flatten().cpu().tolist()
        current['capture']=1000. in steps
        trace.append(dict(current_start=kw['current_start'],latent_frames=int(kw['noisy_image_or_video'].shape[1]),timesteps=steps))
    def layer(_m,_a,out):
        if current.get('capture'):hidden.append(backend._native_hidden(out)[:,-tokens:].detach().clone())
    handles.append(p.generator.register_forward_pre_hook(pre,with_kwargs=True))
    handles.append(p.generator.model.blocks[14].register_forward_hook(layer))

    def prefix(executor,count):
        construct_native_prefix(executor,prompt=prompt,history=history,count=count)

    def payload(executor):
        require(executor.cache.state_count==1,'lease gate did not retain exactly one committed prefix')
        return next(iter(executor.cache._states.values())).payload

    def pair(h,noise,seed,expected_fused):
        require(not hidden,'hidden capture from another request survived')
        args=dict(prompt=prompt,history=h,noise=noise,history_kv_widths=(3,)*(int(h.shape[1])//3))
        t.manual_seed(seed)
        with stage('lease_reference_chunk',cuda=True):expected=reference.generate(**args)
        rng=t.cuda.get_rng_state().clone()
        expected_hidden=hidden.pop();cpu_rng=t.get_rng_state().clone()
        t.manual_seed(seed)
        with stage('lease_joint_chunk',cuda=True):actual=joint.generate(**args)
        actual_hidden=hidden.pop()
        require(not hidden and t.equal(rng,t.cuda.get_rng_state()) and t.equal(cpu_rng,t.get_rng_state()),
                'joint lease changed the native three-draw RNG trajectory')
        require(t.equal(actual_hidden,expected_hidden),'joint lease changed first-denoise layer15 hidden')
        a=backend._rank_sample(Observation._sample_hidden(backend,actual_hidden))
        b=backend._rank_sample(Observation._sample_hidden(backend,expected_hidden))
        require(a==b,'joint lease changed native eRank/spectrum')
        require(actual['joint_execution']['fused_history_chunks']==expected_fused
            and all(actual[k]==expected[k] for k in ('reused_chunks','replayed_chunks','history_binding',
                                                     'history_kv_widths','scope_sha256')),
            'joint lease changed native identity/reuse semantics')
        ap,bp=payload(joint),payload(reference)
        require(not payloads_alias(ap,bp),'native reference and joint executor alias cache storage')
        require(len(ap['self'])==len(bp['self'])==len(ap['cross'])==len(bp['cross'])==30,
                'lease gate did not compare all native self/cross layers')
        proof=check_suffix(actual=actual['latent'],expected=expected['latent'],actual_cache=ap,expected_cache=bp,torch=t)
        proof['oracle']='original_incremental_native_executor_with_independent_fresh_prefix'
        proof.update(hidden15_exact=True,rank_spectrum_exact=True,rng_state_exact=True,
                     exact_history_binding=True,joint_execution=actual['joint_execution'],
                     self_cache_layers=30,cross_cache_layers=30,effective_rank=a.effective_rank,
                     gram_spectrum_sha256=hashlib.sha256(json.dumps(a.gram_eigenvalues,allow_nan=False).encode()).hexdigest(),
                     latent_sha256=latent_digest(actual['latent'],t))
        return actual['latent'],proof

    try:
        with t.inference_mode():
            for case,(retained,external) in enumerate(CASES):
                joint.cache.clear();reference.cache.clear();prefix(joint,retained);prefix(reference,retained)
                h=history[:,:3*(retained+external)];row=dict(retained_blocks=retained,
                    external_target_blocks=external,chunk_index=retained+external,following=[])
                report['rows'].append(row)
                latent,row['current']=pair(h,noises[0],20260911+case*100,external)
                h=t.cat((h,latent),dim=1)
                for following in range(1,7):
                    latent,proof=pair(h,noises[following],20260911+case*100+following,0)
                    row['following'].append(dict(offset=following,**proof));h=t.cat((h,latent),dim=1)
                print('LONGLIVE_JOINT_LEASE_GATE',retained,external,'cache/hidden/rank/RNG exact',flush=True)
            require(trace==expected_forward_trace(),'lease gate actual physical forwards differ')
            report.update(passed=True,forward_trace=trace);return report
    finally:
        for handle in handles:handle.remove()
        joint.cache.clear();reference.cache.clear();p.kv_cache1=p.crossattn_cache=None
        report['forward_trace']=trace
