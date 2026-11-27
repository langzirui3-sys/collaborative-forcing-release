"""Incremental latent-only execution of the pinned Self-Forcing schedule."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from typing import Any, Callable

from .branch_kv_state import BranchKVCache, NativeBranchState, PrefixBinding
from .stage_timing import stage


def latent_digest(tensor: Any, torch: Any) -> str:
    value = tensor.detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
    h = hashlib.sha256(str(tuple(value.shape)).encode('ascii'))
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tensor_bytes(payload: Any, torch: Any) -> int:
    if isinstance(payload, torch.Tensor):
        return payload.numel() * payload.element_size()
    if isinstance(payload, dict):
        return sum(tensor_bytes(x, torch) for x in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(tensor_bytes(x, torch) for x in payload)
    return 0


def clone_native_payload(payload: Any, torch: Any) -> Any:
    """Copy every mutable native KV tensor, including cursor and cross KV."""
    if isinstance(payload, torch.Tensor):
        return payload.detach().clone()
    if isinstance(payload, dict):
        return {key: clone_native_payload(value, torch) for key, value in payload.items()}
    if isinstance(payload, list):
        return [clone_native_payload(value, torch) for value in payload]
    if isinstance(payload, tuple):
        return tuple(clone_native_payload(value, torch) for value in payload)
    if payload is None or type(payload) in (bool, int, float, str):
        return payload
    raise ValueError("unsupported mutable value in native KV payload")


class IncrementalDraftExecutor:
    def __init__(self, *, pipeline: Any, torch: Any, scope_sha256: str,
                 max_bytes: int = 16 * 1024**3, fork_enabled: bool = False,
                 joint_handoff: bool = False):
        self.pipeline, self.torch = pipeline, torch
        self.scope_sha256 = scope_sha256
        PrefixBinding(scope_sha256, ())
        if pipeline.num_frame_per_block != 3 or pipeline.independent_first_frame:
            raise ValueError('incremental execution requires three-frame blocks')
        if pipeline.args.context_noise != 0:
            raise ValueError('clean committed cache requires context_noise=0')
        self.tokens_per_chunk = 3 * pipeline.frame_seq_length
        self.cache = BranchKVCache(max_bytes=max_bytes, tokens_per_chunk=self.tokens_per_chunk)
        self.allocation_bytes: int | None = None
        if type(fork_enabled) is not bool:
            raise ValueError('fork_enabled must be a boolean')
        self.fork_enabled = fork_enabled
        self.joint_handoff = joint_handoff

    def generate(self, *, prompt: str, history: Any, noise: Any,
                 forward_context: Callable[[int], Any] = lambda n: nullcontext(),
                 kv_branch_action: str = 'none', history_kv_widths: tuple[int, ...] | None = None,
                 history_digest_source: Any = None) -> dict[str, Any]:
        torch, p = self.torch, self.pipeline
        if kv_branch_action not in ('none', 'preserve', 'consume'):
            raise ValueError('unknown Draft KV branch action')
        if kv_branch_action != 'none' and not self.fork_enabled:
            raise ValueError('Draft KV branch action requires fork-enabled reuse')
        if noise.dtype != torch.bfloat16 or (history is not None and history.dtype != torch.bfloat16):
            raise ValueError('incremental cache bindings require native bfloat16 latents')
        if history is not None and (history.shape[0] != noise.shape[0] or history.shape[2:] != noise.shape[2:]):
            raise ValueError('history and noise geometry differ')
        if noise.shape[1] != 3:
            raise ValueError('exactly one three-frame noise chunk is required')
        n = 0 if history is None else int(history.shape[1]) // 3
        if history is not None and history.shape[1] != n * 3:
            raise ValueError('history must contain whole chunks')
        if history_kv_widths is None:
            if n:
                raise ValueError('nonempty history requires explicit KV update widths')
            history_kv_widths = ()
        if (not isinstance(history_kv_widths, tuple) or len(history_kv_widths) != n
                or any(type(w) is not int or w not in (1, 3) for w in history_kv_widths)):
            raise ValueError('history KV widths must bind every chunk with 1 or 3')
        scope = hashlib.sha256(('native-committed-v2\0' + self.scope_sha256 + '\0' + prompt).encode('utf-8')).hexdigest()
        from .cpu_history_binding import history_digests
        chunks = history_digests(history, torch, source=history_digest_source)
        # Cache identity binds construction semantics as well as latent bytes.
        bound_chunks = tuple(hashlib.sha256((h + ':' + str(w)).encode('ascii')).hexdigest()
                             for h, w in zip(chunks, history_kv_widths))
        requested = PrefixBinding(scope, bound_chunks)
        action_anchor = self.cache.pinned_binding
        anchor_clone_bytes = 0
        anchor_replaced = False
        with self.cache.checkout(requested, consume_anchor=kv_branch_action == 'consume') as lease:
            start = lease.replay_from_chunk
            anchor_consumed = kv_branch_action == 'consume' and lease.state is not None
            anchor_miss_reason = ('absent' if kv_branch_action == 'consume'
                                  and not anchor_consumed else None)
            try:
                if lease.state is None:
                    # On a miss, free the oldest branch BEFORE allocating a new
                    # native buffer set. The first allocation defines exact bytes.
                    lease.reserve_allocation(self.allocation_bytes or self.cache.max_bytes)
                    p._initialize_kv_cache(batch_size=noise.shape[0], dtype=noise.dtype, device=noise.device)
                    p._initialize_crossattn_cache(batch_size=noise.shape[0], dtype=noise.dtype, device=noise.device)
                    payload = {'self': p.kv_cache1, 'cross': p.crossattn_cache}
                    size = tensor_bytes(payload, torch)
                    if size > self.cache.max_bytes:
                        raise ValueError('native KV allocation exceeds budget')
                    self.allocation_bytes = size
                    lease.note_allocated(size)
                else:
                    payload = lease.state.payload
                    p.kv_cache1, p.crossattn_cache = payload['self'], payload['cross']
                    size = lease.state.resident_bytes
                    self._validate_native_cursor(start)
                conditional = p.text_encoder(text_prompts=[prompt])
                fuse = (self.joint_handoff and start > 0 and kv_branch_action == 'none'
                        and 1 <= n-start <= 4 and history_kv_widths[start:] == (1,) * (n-start))
                with forward_context(0 if fuse else n-start), torch.inference_mode():
                    # Replay only the suffix absent from this exact prefix state.
                    for i in range(n if fuse else start, n):
                        p.generator(noisy_image_or_video=history[:, i*3:(i+1)*3],
                            conditional_dict=conditional,
                            timestep=torch.zeros([noise.shape[0], history_kv_widths[i]], device=noise.device, dtype=torch.int64),
                            kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                            current_start=i*self.tokens_per_chunk)
                    if kv_branch_action == 'preserve':
                        self._validate_native_cursor(n)
                        with stage('anchor_clone', detail={'history_chunks': n, 'bytes': size}, cuda=True):
                            anchor_replaced = lease.preserve_input(
                                NativeBranchState(requested, payload, n, n*self.tokens_per_chunk, size),
                                lambda value: clone_native_payload(value, torch))
                        anchor_clone_bytes = size
                        action_anchor = requested
                    noisy = noise
                    for index, step in enumerate(p.denoising_step_list):
                        timestep = torch.ones([noise.shape[0], 3], device=noise.device, dtype=torch.int64) * step
                        if fuse and index == 0:
                            from .sf_joint_handoff import joint_first_step
                            clean = joint_first_step(p, torch=torch, history=history[:, start*3:n*3],
                                noise=noisy, conditional=conditional, step=step,
                                history_start=start*self.tokens_per_chunk, draft_start=n*self.tokens_per_chunk)
                        else:
                            _, clean = p.generator(noisy_image_or_video=noisy, conditional_dict=conditional,
                                timestep=timestep, kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                                current_start=n*self.tokens_per_chunk)
                        if index < len(p.denoising_step_list)-1:
                            next_step = p.denoising_step_list[index+1]
                            noisy = p.scheduler.add_noise(clean.flatten(0,1), torch.randn_like(clean.flatten(0,1)),
                                next_step * torch.ones([noise.shape[0]*3], device=noise.device, dtype=torch.long)
                            ).unflatten(0, clean.shape[:2])
                    p.generator(noisy_image_or_video=clean, conditional_dict=conditional,
                        # Match the native continuous loop's generated-clean
                        # update. Replayed history above follows each chunk's
                        # declared construction; external chunks may use B1.
                        timestep=torch.zeros([noise.shape[0], 3], device=noise.device, dtype=torch.int64),
                        kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                        current_start=n*self.tokens_per_chunk)
                self._validate_native_cursor(n+1)
                generated_binding = hashlib.sha256((latent_digest(clean, torch) + ':3').encode('ascii')).hexdigest()
                result_binding = PrefixBinding(scope, bound_chunks+(generated_binding,))
                lease.commit(NativeBranchState(result_binding,payload,n+1,(n+1)*self.tokens_per_chunk,size))
                result = {'latent': clean, 'replayed_chunks': n-start, 'reused_chunks': start,
                          'history_binding': chunks, 'history_kv_widths': history_kv_widths, 'scope_sha256': scope}
                if self.fork_enabled:
                    result['branch_anchor'] = {
                        'schema': 'draft_kv_branch_anchor_v1', 'action': kv_branch_action,
                        'anchor_chunks': (len(action_anchor.chunk_sha256)
                                          if action_anchor is not None else None),
                        'anchor_clone_count': int(anchor_clone_bytes > 0),
                        'anchor_clone_bytes': anchor_clone_bytes,
                        'anchor_replaced': anchor_replaced, 'anchor_consumed': anchor_consumed,
                        'anchor_miss_reason': anchor_miss_reason,
                        'anchor_retained': self.cache.pinned_binding is not None,
                        'resident_bytes': self.cache.resident_bytes,
                        'peak_resident_bytes': self.cache.peak_resident_bytes,
                        'state_count': self.cache.state_count,
                    }
                return result
            finally:
                # Native pipeline aliases must not survive a lease, especially a
                # failed forward. A later legacy call cannot mutate retained KV.
                p.kv_cache1 = None
                p.crossattn_cache = None

    def prefill_accepted(self, *, prompt: str, history: Any,
                         history_kv_widths: tuple[int, ...],
                         history_digest_source: Any = None,
                         forward_context: Callable[[int], Any] = lambda n: nullcontext(),
                         maximum_suffix_chunks: int = 4,
                         target_kv_width: int = 1) -> dict[str, Any]:
        """Advance clean native KV to an accepted Target suffix, with no sampling.

        This opt-in primitive never reconstructs a missing root, clones a KV
        allocation, creates noise, calls the scheduler or generates a latent.
        The worker must serialize access and complete its CUDA stream before
        making the advanced prefix available to another thread.
        """
        torch,p=self.torch,self.pipeline
        if (not isinstance(prompt,str) or not prompt.strip()
                or not isinstance(history,torch.Tensor) or history.ndim < 3
                or history.dtype != torch.bfloat16 or history.requires_grad
                or history.shape[0] != 1 or history.shape[1] <= 0
                or history.shape[1] % 3):
            raise ValueError('accepted prefill requires one native BF16 history')
        n=int(history.shape[1])//3
        if (not isinstance(history_kv_widths,tuple) or len(history_kv_widths)!=n
                or any(type(w) is not int or w not in (1,3) for w in history_kv_widths)
                or type(maximum_suffix_chunks) is not int or not 1<=maximum_suffix_chunks<=4
                or type(target_kv_width) is not int or target_kv_width not in (1, 3)):
            raise ValueError('accepted prefill requires bound native widths and suffix limit')
        if self.cache.pinned_binding is not None:
            raise ValueError('accepted prefill forbids an active branch anchor')
        scope=hashlib.sha256(('native-committed-v2\0'+self.scope_sha256+'\0'+prompt).encode()).hexdigest()
        from .cpu_history_binding import history_digests
        chunks=history_digests(history,torch,source=history_digest_source)
        requested=PrefixBinding(scope,tuple(hashlib.sha256((h+':'+str(w)).encode('ascii')).hexdigest()
                                           for h,w in zip(chunks,history_kv_widths)))
        with self.cache.checkout(requested) as lease:
            start=lease.replay_from_chunk
            if lease.state is None:
                raise ValueError('accepted prefill requires an existing exact native prefix')
            if n-start>maximum_suffix_chunks or any(w!=target_kv_width for w in history_kv_widths[start:]):
                raise ValueError('accepted prefill may append only the bounded external Target suffix')
            payload=lease.state.payload;size=lease.state.resident_bytes
            try:
                p.kv_cache1,p.crossattn_cache=payload['self'],payload['cross']
                self._validate_native_cursor(start)
                conditional=p.text_encoder(text_prompts=[prompt])
                with forward_context(n-start),torch.inference_mode():
                    for i in range(start,n):
                        p.generator(noisy_image_or_video=history[:,i*3:(i+1)*3],
                            conditional_dict=conditional,
                            timestep=torch.zeros([history.shape[0],history_kv_widths[i]],
                                                 device=history.device,dtype=torch.int64),
                            kv_cache=p.kv_cache1,crossattn_cache=p.crossattn_cache,
                            current_start=i*self.tokens_per_chunk)
                self._validate_native_cursor(n)
                lease.commit_prefill(NativeBranchState(requested,payload,n,n*self.tokens_per_chunk,size))
                return {'schema':'native_accepted_prefix_prefill_v1','reused_chunks':start,
                    'replayed_chunks':n-start,'history_binding':chunks,
                    'history_kv_widths':history_kv_widths,'scope_sha256':scope,
                    'next_chunk_index':n,'native_allocation_bytes':size,
                    'noise_created':False,'scheduler_updates':0,'generated_chunks':0,
                    'native_payload_reused':True,'branch_snapshot_clone_bytes':0}
            finally:
                p.kv_cache1=None;p.crossattn_cache=None

    def _validate_native_cursor(self, chunks: int) -> None:
        for block in self.pipeline.kv_cache1:
            if int(block['global_end_index'].item()) != chunks*self.tokens_per_chunk:
                raise ValueError('native global KV cursor differs from bound history')
            expected_local = min(chunks*self.tokens_per_chunk, block['k'].shape[1])
            if int(block['local_end_index'].item()) != expected_local:
                raise ValueError('native rolling KV cursor differs from bound history')
