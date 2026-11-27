"""Experimental native Draft early exit and same-slot external clean commit.

This executor is not wired into a formal manifest yet. A synchronous layer
hook may raise DraftProbeStop; continuing the hook preserves the original call
stack and RNG stream. No KV payload is cloned. A stopped payload is tentative,
never advertised as an accepted prefix, until commit_external overwrites the
current block through a complete native clean forward.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib

from .incremental_draft import latent_digest, tensor_bytes


class DraftProbeStop(Exception):
    def __init__(self, evidence):
        super().__init__('Draft stopped at the requested probe')
        self.evidence = evidence


class EarlyDraftExecutor:
    """One exclusively owned native cache, with explicit tentative state."""
    def __init__(self, *, pipeline, torch, scope_sha256, max_bytes=16 * 1024**3):
        if pipeline.num_frame_per_block != 3 or pipeline.independent_first_frame:
            raise ValueError('early Draft requires native three-frame blocks')
        if pipeline.args.context_noise != 0:
            raise ValueError('early Draft requires clean accepted history')
        self.pipeline, self.torch = pipeline, torch
        self.scope_sha256, self.max_bytes = scope_sha256, max_bytes
        self.tokens_per_chunk = 3 * pipeline.frame_seq_length
        self.payload = None
        self.binding = ()
        self.scope = None
        self.pending = None
        self.poisoned = False

    def _identity(self, prompt, history, widths):
        torch = self.torch
        count = 0 if history is None else int(history.shape[1]) // 3
        if history is not None and (history.dtype != torch.bfloat16 or history.shape[1] != 3*count):
            raise ValueError('history must contain native bfloat16 whole chunks')
        if not isinstance(widths, tuple) or len(widths) != count or any(type(w) is not int or w not in (1,3) for w in widths):
            raise ValueError('history KV widths must bind every accepted chunk')
        scope = hashlib.sha256((self.scope_sha256+'\0'+prompt).encode()).hexdigest()
        binding = tuple(self._chunk_binding(history[:,3*i:3*i+3], widths[i]) for i in range(count))
        return scope, binding

    def _chunk_binding(self, tensor, width):
        return hashlib.sha256((latent_digest(tensor,self.torch)+':'+str(width)).encode()).hexdigest()

    def _attach(self):
        self.pipeline.kv_cache1 = self.payload['self']
        self.pipeline.crossattn_cache = self.payload['cross']

    def _detach(self):
        self.pipeline.kv_cache1 = None
        self.pipeline.crossattn_cache = None

    def _cursor(self, chunks, *, tentative=False):
        seen_later = False
        for block in self.payload['self']:
            end = int(block['global_end_index'].item())
            allowed = (chunks, chunks+1) if tentative else (chunks,)
            if end not in tuple(i*self.tokens_per_chunk for i in allowed):
                raise ValueError('native KV cursor escaped the current chunk')
            local = int(block['local_end_index'].item())
            if local != min(end,block['k'].shape[1]):
                raise ValueError('native rolling KV cursor is inconsistent')
            if tentative:
                # A synchronous early exit can only have visited a layer prefix.
                if end == chunks*self.tokens_per_chunk:
                    seen_later = True
                elif seen_later:
                    raise ValueError('tentative layers are not a contiguous prefix')

    def _clean(self, latent, conditional, chunk_index, width):
        t = self.torch
        self.pipeline.generator(noisy_image_or_video=latent, conditional_dict=conditional,
            timestep=t.zeros([latent.shape[0],width],device=latent.device,dtype=t.int64),
            kv_cache=self.pipeline.kv_cache1,crossattn_cache=self.pipeline.crossattn_cache,
            current_start=chunk_index*self.tokens_per_chunk)

    def generate(self, *, prompt, history, noise, history_kv_widths,
                 forward_context=lambda count:nullcontext()):
        t,p = self.torch,self.pipeline
        if self.poisoned or self.pending is not None:
            raise ValueError('resolve the pending external commit before generating')
        if noise.dtype != t.bfloat16 or noise.shape[1] != 3:
            raise ValueError('noise must be one native bfloat16 chunk')
        if history is not None and (history.shape[0] != noise.shape[0] or history.shape[2:] != noise.shape[2:]):
            raise ValueError('history and noise geometry differ')
        scope,binding = self._identity(prompt,history,history_kv_widths)
        n = len(binding)
        if self.payload is not None:
            if scope != self.scope or binding[:len(self.binding)] != self.binding:
                raise ValueError('accepted history changed without an explicit new executor')
        start = len(self.binding) if self.payload is not None else 0
        try:
            if self.payload is None:
                p._initialize_kv_cache(batch_size=noise.shape[0],dtype=noise.dtype,device=noise.device)
                p._initialize_crossattn_cache(batch_size=noise.shape[0],dtype=noise.dtype,device=noise.device)
                self.payload={'self':p.kv_cache1,'cross':p.crossattn_cache}
                if tensor_bytes(self.payload,t)>self.max_bytes:
                    raise ValueError('native early cache exceeds its memory bound')
                self.scope=scope
            self._attach()
            self._cursor(start)
            conditional=p.text_encoder(text_prompts=[prompt])
            with forward_context(n-start),t.inference_mode():
                for i in range(start,n):
                    self._clean(history[:,3*i:3*i+3],conditional,i,history_kv_widths[i])
                self.binding=binding
                self._cursor(n)
                noisy=noise
                try:
                    for i,step in enumerate(p.denoising_step_list):
                        _,clean=p.generator(noisy_image_or_video=noisy,conditional_dict=conditional,
                            timestep=t.ones([noise.shape[0],3],device=noise.device,dtype=t.int64)*step,
                            kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,current_start=n*self.tokens_per_chunk)
                        if i<len(p.denoising_step_list)-1:
                            noisy=p.scheduler.add_noise(clean.flatten(0,1),t.randn_like(clean.flatten(0,1)),
                                p.denoising_step_list[i+1]*t.ones([noise.shape[0]*3],device=noise.device,dtype=t.long)
                            ).unflatten(0,clean.shape[:2])
                except DraftProbeStop as stop:
                    if i != 0:
                        raise ValueError('early probe is only allowed in the first denoising forward') from stop
                    self._cursor(n,tentative=True)
                    self.pending={'scope':scope,'binding':binding,'chunk_index':n,
                                  'shape':tuple(noise.shape),'device':noise.device}
                    return {'stopped':True,'probe':stop.evidence,'replayed_chunks':n-start,
                            'reused_chunks':start,'branch_snapshot_bytes':0}
                self._clean(clean,conditional,n,3)
                self._cursor(n+1)
                self.binding=binding+(self._chunk_binding(clean,3),)
                return {'stopped':False,'latent':clean,'replayed_chunks':n-start,
                        'reused_chunks':start,'branch_snapshot_bytes':0}
        except BaseException:
            self.payload=None;self.pending=None;self.poisoned=True
            raise
        finally:
            self._detach()

    def commit_external(self, *, prompt, history, history_kv_widths, latent, width=1,
                        forward_context=lambda count:nullcontext()):
        """Overwrite tentative current-slot KV with the accepted Target chunk.

        Already visited layers do not roll twice; untouched layers roll once.
        The native implementation must be parity-tested before formal use.
        """
        t,p=self.torch,self.pipeline
        if self.poisoned or self.pending is None:
            raise ValueError('no tentative Draft block to replace')
        scope,binding=self._identity(prompt,history,history_kv_widths)
        pending=self.pending
        if scope!=pending['scope'] or binding!=pending['binding']:
            raise ValueError('external commit belongs to a different history')
        if type(width) is not int or width!=1 or tuple(latent.shape)!=pending['shape'] or latent.dtype!=t.bfloat16:
            raise ValueError('external commit must use the native Target history geometry')
        if not bool(t.isfinite(latent.float()).all().item()):
            raise ValueError('external latent is nonfinite')
        try:
            self._attach()
            self._cursor(len(binding),tentative=True)
            clean=latent.to(device=pending['device']).contiguous()
            conditional=p.text_encoder(text_prompts=[prompt])
            with forward_context(1),t.inference_mode():
                self._clean(clean,conditional,len(binding),width)
            self._cursor(len(binding)+1)
            self.binding=binding+(self._chunk_binding(clean,width),)
            self.pending=None
            return {'external_clean_forwards':1,'history_replay_chunks':0,'branch_snapshot_bytes':0,
                    'committed_chunks':len(self.binding)}
        except BaseException:
            self.payload=None;self.pending=None;self.poisoned=True
            raise
        finally:
            self._detach()

    def resume_external_prefix(self, *, prompt, history, history_kv_widths,
                               noisy_latent_suffix, renoise_second, renoise_third,
                               forward_context=lambda count:nullcontext()):
        """Finish a stopped block from an externally produced native suffix input.

        This is a distinct experimental action from accepting a complete Target
        block. Target's first clean prediction must be re-noised at the native second timestep before
        this call. All later noise is explicit; another process's CUDA RNG is
        never assumed to match this process. Only the final Draft prediction is
        committed, using native Draft width 3. No Target KV is copied.

        The real-model gate must establish partial-cache overwrite correctness
        before this operation is used in a quality/speed experiment.
        """
        t,p=self.torch,self.pipeline
        if self.poisoned or self.pending is None:
            raise ValueError('no tentative Draft block to resume')
        scope,binding=self._identity(prompt,history,history_kv_widths)
        pending=self.pending
        if scope!=pending['scope'] or binding!=pending['binding']:
            raise ValueError('external prefix belongs to a different history')
        from .target_prefix import native_draft_schedule
        steps,schedule=native_draft_schedule(p)
        values=(noisy_latent_suffix,renoise_second,renoise_third)
        for value in values:
            if (not isinstance(value,t.Tensor) or tuple(value.shape)!=pending['shape']
                    or value.dtype!=t.bfloat16 or not bool(t.isfinite(value.float()).all().item())):
                raise ValueError('external prefix tensors must have finite native geometry')
        try:
            self._attach()
            self._cursor(len(binding),tentative=True)
            noisy,noise_second,noise_third=(v.to(device=pending['device']).contiguous() for v in values)
            conditional=p.text_encoder(text_prompts=[prompt])
            with forward_context(0),t.inference_mode():
                for i,step in enumerate(steps[1:]):
                    _,clean=p.generator(noisy_image_or_video=noisy,conditional_dict=conditional,
                        timestep=t.ones((noisy.shape[0],3),device=noisy.device,dtype=t.int64)*step,
                        kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                        current_start=len(binding)*self.tokens_per_chunk)
                    if i<2:
                        noisy=p.scheduler.add_noise(clean.flatten(0,1),(noise_second,noise_third)[i].flatten(0,1),
                            steps[i+2]*t.ones((clean.shape[0]*3,),device=clean.device,dtype=t.long)
                        ).unflatten(0,clean.shape[:2])
                self._clean(clean,conditional,len(binding),3)
            self._cursor(len(binding)+1)
            self.binding=binding+(self._chunk_binding(clean,3),)
            self.pending=None
            return {'latent':clean,'draft_denoising_schedule':schedule[1:],'draft_native_schedule':schedule,
                    'draft_clean_forwards':1,'history_replay_chunks':0,'branch_snapshot_bytes':0,
                    'committed_kv_width':3,'committed_chunks':len(self.binding)}
        except BaseException:
            self.payload=None;self.pending=None;self.poisoned=True
            raise
        finally:
            self._detach()
