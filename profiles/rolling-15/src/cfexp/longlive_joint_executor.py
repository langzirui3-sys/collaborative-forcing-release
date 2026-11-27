"""Diagnostic lease integration for the CUDA-validated LongLive primitive.

This is not installed by the policy loader. It preserves the original exact
prefix bindings, native B3 construction and original four-step sampler. A
separate integrated CUDA/observation gate is required before production use.
"""
from contextlib import nullcontext
import hashlib
import os

from .branch_kv_state import NativeBranchState, PrefixBinding
from .cpu_history_binding import history_digests
from .incremental_draft import IncrementalDraftExecutor, latent_digest, tensor_bytes
from .longlive_chunk_session import LongLiveChunkSession, _NativePipelineView
from .longlive_joint_handoff import ENV, joint_first_step, require
from .longlive_joint_oracle import SCHEDULE


class JointLongLiveExecutor(IncrementalDraftExecutor):
    """Own an exact existing native branch store, never copy another model's KV.

    Missing suffixes of one to four blocks can share a forward with the next
    Draft first denoise. An empty/mismatched prefix or longer suffix follows
    the ordinary native sequence. Both paths use exactly three random draws.
    """

    def __init__(self, session):
        require(os.environ.get(ENV)=='1', 'joint executor requires explicit diagnostic opt-in')
        require(type(session) is LongLiveChunkSession and not session.closed,
                'joint executor requires its live native LongLive session')
        native_executor=session.executor
        require(type(native_executor) is IncrementalDraftExecutor
            and type(native_executor.pipeline) is _NativePipelineView
            and not native_executor.pipeline.host_cursors.enabled
            and not native_executor.joint_handoff
            and native_executor.cache.pinned_binding is None
            and native_executor.pipeline.kv_cache1 is None
            and getattr(native_executor.pipeline,'crossattn_cache',None) is None
            and tuple(float(x) for x in native_executor.pipeline.denoising_step_list)==SCHEDULE,
            'joint executor requires an idle unpatched native LongLive owner')
        # Transfer use of the same store; no tensor allocation or clone here.
        # The session must replace its executor, so its original close path
        # continues to clear exactly this store.
        self.__dict__.update(native_executor.__dict__)
        session.executor=self

    def generate(self, *, prompt, history, noise, history_kv_widths=None,
                 history_digest_source=None, kv_branch_action='none',
                 forward_context=lambda count:nullcontext(), clean_observer=None):
        t,p=self.torch,self.pipeline
        require(os.environ.get(ENV)=='1' and kv_branch_action=='none'
            and self.cache.pinned_binding is None
            and tuple(float(x) for x in p.denoising_step_list)==SCHEDULE,
            'joint LongLive diagnostic requires a single committed branch')
        require(isinstance(prompt,str) and bool(prompt.strip())
            and isinstance(noise,t.Tensor) and noise.ndim>=3
            and noise.shape[0]==1 and noise.shape[1]==3 and noise.dtype==t.bfloat16
            and not noise.requires_grad,'joint LongLive requires one native BF16 noise block')
        require(history is None or (isinstance(history,t.Tensor) and history.ndim==noise.ndim
            and history.shape[0]==1 and history.shape[2:]==noise.shape[2:]
            and history.shape[1]%3==0 and history.dtype==noise.dtype
            and history.device==noise.device and not history.requires_grad),
            'joint LongLive history geometry/precision/device differs')
        n=0 if history is None else int(history.shape[1])//3
        require(isinstance(history_kv_widths,tuple) and len(history_kv_widths)==n
            and all(type(w) is int and w==3 for w in history_kv_widths),
            'every LongLive history owner requires native B3 clean construction')
        require(clean_observer is None or callable(clean_observer),'invalid clean-cache observer')
        scope=hashlib.sha256(('native-committed-v2\0'+self.scope_sha256+'\0'+prompt).encode()).hexdigest()
        chunks=history_digests(history,t,source=history_digest_source)
        bound=tuple(hashlib.sha256((h+':3').encode('ascii')).hexdigest() for h in chunks)
        requested=PrefixBinding(scope,bound)
        with self.cache.checkout(requested) as lease:
            start=lease.replay_from_chunk
            fused=lease.state is not None and start>0 and 1<=n-start<=4
            try:
                if lease.state is None:
                    lease.reserve_allocation(self.allocation_bytes or self.cache.max_bytes)
                    p._initialize_kv_cache(batch_size=1,dtype=noise.dtype,device=noise.device)
                    p._initialize_crossattn_cache(batch_size=1,dtype=noise.dtype,device=noise.device)
                    payload={'self':p.kv_cache1,'cross':p.crossattn_cache};size=tensor_bytes(payload,t)
                    require(size<=self.cache.max_bytes,'native LongLive allocation exceeds budget')
                    self.allocation_bytes=size;lease.note_allocated(size)
                else:
                    payload=lease.state.payload;size=lease.state.resident_bytes
                    p.kv_cache1,p.crossattn_cache=payload['self'],payload['cross']
                    self._validate_native_cursor(start)
                conditional=p.text_encoder(text_prompts=[prompt])
                with forward_context(0 if fused else n-start),t.inference_mode():
                    for i in range(n if fused else start,n):
                        p.generator(noisy_image_or_video=history[:,3*i:3*i+3],conditional_dict=conditional,
                            timestep=t.zeros((1,3),device=noise.device,dtype=t.int64),
                            kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                            current_start=i*self.tokens_per_chunk)
                    noisy=noise
                    for index,step in enumerate(p.denoising_step_list):
                        if fused and index==0:
                            clean=joint_first_step(p,torch=t,history=history[:,3*start:3*n],
                                noise=noisy,conditional=conditional,
                                history_start=start*self.tokens_per_chunk,draft_start=n*self.tokens_per_chunk,
                                clean_observer=clean_observer)
                        else:
                            _,clean=p.generator(noisy_image_or_video=noisy,conditional_dict=conditional,
                                timestep=t.ones((1,3),device=noise.device,dtype=t.int64)*step,
                                kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                                current_start=n*self.tokens_per_chunk)
                        if index<3:
                            noisy=p.scheduler.add_noise(clean.flatten(0,1),t.randn_like(clean.flatten(0,1)),
                                p.denoising_step_list[index+1]*t.ones((3,),device=noise.device,dtype=t.long)
                                ).unflatten(0,clean.shape[:2])
                    p.generator(noisy_image_or_video=clean,conditional_dict=conditional,
                        timestep=t.zeros((1,3),device=noise.device,dtype=t.int64),
                        kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                        current_start=n*self.tokens_per_chunk)
                self._validate_native_cursor(n+1)
                generated=hashlib.sha256((latent_digest(clean,t)+':3').encode('ascii')).hexdigest()
                lease.commit(NativeBranchState(PrefixBinding(scope,bound+(generated,)),payload,
                    n+1,(n+1)*self.tokens_per_chunk,size))
                return dict(latent=clean,reused_chunks=start,replayed_chunks=n-start,
                    history_binding=chunks,history_kv_widths=history_kv_widths,scope_sha256=scope,
                    joint_execution=dict(schema='longlive_joint_lease_execution_v1',
                        fused_history_chunks=n-start if fused else 0,
                        expected_physical_generator_calls=5 if fused else n-start+5,
                        native_sampling_updates=4,native_renoise_draws=3,
                        branch_snapshot_bytes=0,clean_update_width=3,
                        requires_independent_physical_trace_verification=True))
            finally:
                p.kv_cache1=p.crossattn_cache=None
