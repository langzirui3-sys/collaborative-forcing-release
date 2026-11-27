"""Bind native KV identity from the exact CPU input before its native H2D copy.

No digest is accepted from a caller. Tensor bytes are still hashed on every
request; only the redundant GPU-to-CPU trip is removed. The frozen adapter
continues to own conversion, prefix checks, model execution and RNG handling.
"""
from contextvars import ContextVar
import os

ENV = 'CFEXP_CPU_HISTORY_BINDING'


def validate_evidence(value, *, history_chunks, tensor_digests):
    if value != {'schema':'native_cpu_history_binding_v1',
                 'source':'worker_loaded_native_bfloat16_cpu_history',
                 'history_chunks':history_chunks,
                 'history_tensor_sha256':list(tensor_digests),
                 'native_h2d_and_prefix_checks_preserved':True}:
        raise ValueError('missing or inconsistent CPU history binding evidence')


def validate_longlive_evidence(value, *, history_chunks, tensor_digests):
    if not isinstance(value, dict) or value.get('schema') != 'longlive_native_cpu_history_binding_v1':
        raise ValueError('missing native LongLive CPU binding evidence')
    validate_evidence(dict(value, schema='native_cpu_history_binding_v1'),
                      history_chunks=history_chunks, tensor_digests=tensor_digests)


def history_digests(history, torch, *, source=None):
    from .incremental_draft import latent_digest
    if source is not None:
        if (history is None or not isinstance(source, torch.Tensor)
                or source.device.type != 'cpu' or source.dtype != torch.bfloat16
                or history.dtype != torch.bfloat16 or source.shape != history.shape
                or source.requires_grad):
            raise ValueError('CPU history binding requires the exact native BF16 CPU input geometry')
    value = history if source is None else source
    if value is None:
        return ()
    return tuple(latent_digest(value[:, i*3:(i+1)*3], torch)
                 for i in range(int(value.shape[1]) // 3))


def install(backend):
    from .self_forcing_frame_adapter import OfficialSelfForcingBackend
    from .incremental_draft import IncrementalDraftExecutor
    from .sdvg_adapter import draft_generation_backend

    backend = draft_generation_backend(backend)
    if type(backend) is not OfficialSelfForcingBackend:
        raise ValueError('CPU binding requires the frozen native Self-Forcing backend')
    executor = backend._incremental_draft
    if type(executor) is not IncrementalDraftExecutor or os.environ.get('CFEXP_DRAFT_KV_REUSE') != '1':
        raise ValueError('CPU binding requires native incremental Draft KV reuse')
    _install_backend(backend, executor, schema='native_cpu_history_binding_v1')


def install_longlive(backend):
    """Bind LongLive's own session, without installing SF cache semantics."""
    from .longlive_policy_backend import LongLivePolicyBackend, _NativeExecutor
    from .longlive_chunk_session import LongLiveChunkSession
    from .incremental_draft import IncrementalDraftExecutor
    if not isinstance(backend, LongLivePolicyBackend):
        raise ValueError('CPU binding requires native LongLive policy backend')
    executor = backend._incremental_draft
    if (type(executor) is not _NativeExecutor
            or type(executor.session) is not LongLiveChunkSession
            or type(executor.session.executor) is not IncrementalDraftExecutor):
        raise ValueError('CPU binding requires LongLive native incremental session')
    _install_backend(backend, executor, schema='longlive_native_cpu_history_binding_v1')


def _install_backend(backend, executor, *, schema):
    if getattr(backend, '_cpu_history_binding_installed', False):
        raise ValueError('CPU history binding already installed')
    active = ContextVar('native_cpu_history_binding', default=None)
    original_generate = backend.generate_chunk
    original_execute = executor.generate

    def execute(**kwargs):
        context = active.get()
        if context is None or context['called']:
            raise ValueError('CPU history binding must belong to one native backend invocation')
        context['called'] = True
        if 'history_digest_source' in kwargs:
            raise ValueError('caller may not replace worker-owned CPU history source')
        result = original_execute(**kwargs, history_digest_source=context['source'])
        context['binding'] = result['history_binding']
        return result

    def generate(**kwargs):
        if active.get() is not None:
            raise ValueError('nested CPU history binding is not supported')
        source = kwargs['history_latents']
        context = {'source':source, 'called':False}
        token = active.set(context)
        try:
            result = original_generate(**kwargs)
            if not context['called'] or list(context['binding']) != result['kv_reuse']['history_tensor_sha256']:
                raise ValueError('native KV binding differs from its CPU source')
            return dict(result, history_binding_execution={
                'schema':schema,
                'source':'worker_loaded_native_bfloat16_cpu_history',
                'history_chunks':kwargs['chunk_index'],
                'history_tensor_sha256':list(context['binding']),
                'native_h2d_and_prefix_checks_preserved':True})
        finally:
            active.reset(token)

    executor.generate = execute
    backend.generate_chunk = generate
    backend._cpu_history_binding_installed = True
