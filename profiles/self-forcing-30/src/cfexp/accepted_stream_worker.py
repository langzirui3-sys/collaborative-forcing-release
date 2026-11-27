"""Opt-in worker protocol; existing worker PID owns rendering and generation."""
import os
import subprocess
import threading

from .accepted_stream import STREAM_CHUNK_COUNTS, ENV, PRIORITY_ENV, AcceptedStream, NativeStreamDecoder
from .correction_state import LatentHistory
from .correction_worker import CorrectionWorker, WorkerProtocolError, WorkerPoisonedError

ACTIONS = {'stream_prepare', 'stream_append', 'stream_finish', 'stream_abort', 'stream_set_role'}
from .accepted_prefill_worker import ACTION as PREFILL_ACTION


class StreamingCorrectionWorker(CorrectionWorker):
    """Separate opt-in protocol, leaving the frozen original worker unchanged."""
    def __init__(self, **kwargs):
        from .native_vae_component import selected_model, validate_flags
        if selected_model() is not None:
            validate_flags()
        super().__init__(**kwargs)
        self._render_gpu_lock = threading.RLock()
        from .cpu_history_binding import ENV as BINDING_ENV, install
        binding = os.environ.get(BINDING_ENV, '0')
        if binding not in ('0', '1'):
            raise ValueError('invalid CPU history binding flag')
        if binding == '1' and self.role == 'draft':
            install(self.backend)
        from .rope_worker import ENV as ROPE_ENV, install as install_rope
        rope=os.environ.get(ROPE_ENV,'0')
        if rope not in ('0','1'):
            raise ValueError('invalid native RoPE coefficient reuse flag')
        self._restore_rope=None
        if rope=='1' and self.role=='draft':
            self._restore_rope=install_rope(self.backend,self.run_dir)
        from .full_attention_worker import ENV as ATTENTION_ENV, install as install_attention
        attention=os.environ.get(ATTENTION_ENV,'0')
        if attention not in ('0','1'):
            raise ValueError('invalid full native attention flag')
        self._restore_attention=None
        if attention=='1' and self.role=='draft':
            self._restore_attention=install_attention(self.backend,self.run_dir)
        from .accepted_prefill import enabled as prefill_enabled,require
        from .longlive_accepted_prefill import enabled as longlive_prefill_enabled
        self._accepted_prefill=None
        if prefill_enabled() and self.role=='draft':
            from .self_forcing_frame_adapter import OfficialSelfForcingBackend
            from .accepted_prefill_worker import AcceptedPrefillWorker
            require(type(self.backend) is OfficialSelfForcingBackend
                and all(os.environ.get(k)=='1' for k in ('CFEXP_DRAFT_KV_REUSE',
                    'CFEXP_CPU_HISTORY_BINDING','CFEXP_NATIVE_ROPE_COEFFICIENT_REUSE',
                    'CFEXP_NATIVE_FULL_LENGTH_ATTENTION')),
                'accepted prefill requires explicit probed native SF cache execution')
            self._accepted_prefill=AcceptedPrefillWorker(self)
        if longlive_prefill_enabled():
            require(selected_model() == 'longlive' and self._chunk_count == 60
                    and not prefill_enabled(),
                    'LongLive prefill requires the 30-second native stream worker')
            if self.role == 'draft':
                from .accepted_prefill_worker import AcceptedPrefillWorker
                from .longlive_accepted_prefill import native_prefill
                from .native_vae_component import validate_worker
                validate_worker(self)
                self._accepted_prefill = AcceptedPrefillWorker(
                    self, operation=native_prefill, longlive_split=True)

    def _validate_common(self, request):
        if request.get('action') not in ACTIONS | {PREFILL_ACTION}:
            return super()._validate_common(request)
        if self.poisoned:
            raise WorkerPoisonedError('poisoned stream worker permits only state/shutdown')
        request_id, _ = super()._validate_common(dict(request, action='model_state'))
        return request_id, request['action']

    def handle(self, request):
        request_id, action = self._validate_common(request)
        prefill=getattr(self,'_accepted_prefill',None)
        if action==PREFILL_ACTION:
            if prefill is None:
                raise WorkerProtocolError('accepted prefill is not enabled on this Draft worker')
            try:return request_id,action,prefill.start(request)
            except WorkerProtocolError:raise
            except BaseException as error:raise self._poison(error) from error
        if prefill is not None and action!='shutdown':
            try:prefill.barrier()
            except BaseException as error:raise self._poison(error) from error
        if action in ACTIONS:
            try:
                return request_id, action, handle(self, request)
            except WorkerProtocolError:
                raise
            except BaseException as error:
                raise self._poison(error) from error
        # Joining a renderer must never hold the GPU lock it needs to finish.
        if action == 'shutdown':
            return super().handle(request)
        with self._render_gpu_lock:
            return super().handle(request)

    def close(self):
        prefill=getattr(self,'_accepted_prefill',None)
        if prefill is not None:
            prefill.close()
            self._accepted_prefill=None
        stream = getattr(self, '_accepted_stream', None)
        if stream is not None:
            stream.abort()
            self._accepted_stream = None
        restore=getattr(self,'_restore_attention',None)
        if restore is not None:
            restore()
            self._restore_attention=None
        restore=getattr(self,'_restore_rope',None)
        if restore is not None:
            restore()
            self._restore_rope=None
        super().close()


def process_factory(argv, **kwargs):
    """Select this explicit entrypoint without editing frozen runtime launchers."""
    if os.environ.get(ENV) != '1' or list(argv[1:3]) != ['-m', 'cfexp.correction_worker']:
        raise ValueError('stream launcher requires the explicit standard-worker opt-in')
    command = list(argv)
    command[2] = 'cfexp.accepted_stream_worker'
    return subprocess.Popen(command, **kwargs)


def handle(worker, request):
    from .correction_worker import _exact_keys, WorkerProtocolError
    from .video_encoding_policy import policy, SERIAL
    from .native_vae_component import selected_model, validate_worker
    native_model = selected_model()
    if native_model is not None:
        validate_worker(worker)
    elif (os.environ.get(ENV) != '1' or os.environ.get('CFEXP_DRAFT_KV_REUSE') != '1'
            or policy() != SERIAL or worker._chunk_count not in STREAM_CHUNK_COUNTS):
        raise WorkerProtocolError('accepted streaming requires explicit native-KV/serial-ABR opt-in and a supported duration')
    action = request['action']
    priority=os.environ.get(PRIORITY_ENV,'0')
    if priority not in ('0','1'):
        raise WorkerProtocolError('invalid role-aware priority flag')
    fields = ({'generation_role'} if action=='stream_set_role' else
              set() if action in {'stream_prepare', 'stream_abort'} else {'history'})
    if action == 'stream_prepare' and native_model is not None:
        fields = {'native_repository_root'}
    _exact_keys(request, worker._COMMON_KEYS | fields, label=action+' request')
    if action == 'stream_prepare':
        if getattr(worker, '_accepted_decoder', None) is None:
            with worker._render_gpu_lock:
                worker._accepted_decoder = (NativeStreamDecoder(worker.run_dir)
                    if native_model is None else NativeStreamDecoder(worker.run_dir,
                        native_model=native_model, repository=request['native_repository_root']))
        return dict(worker._accepted_decoder.evidence)
    if action == 'stream_abort':
        stream = getattr(worker, '_accepted_stream', None)
        if stream is not None:
            stream.abort()
            worker._accepted_stream = None
        return {'aborted': True}
    if action == 'stream_set_role':
        role=request['generation_role']
        if priority!='1' or role not in ('draft','target'):
            raise WorkerProtocolError('invalid role-aware render scheduling request')
        allowed=role!=worker.role
        stream=getattr(worker,'_accepted_stream',None)
        if stream is None:
            raise WorkerProtocolError('accepted root must precede render scheduling')
        stream.set_render_allowed(allowed)
        return {'allowed':allowed,'generation_role':role,'renderer_worker_role':worker.role}
    history = LatentHistory.from_dict(request['history'])
    if getattr(worker, '_accepted_decoder', None) is None:
        raise WorkerProtocolError('prepare the native stream decoder before timing generation')
    stream = getattr(worker, '_accepted_stream', None)
    if action == 'stream_append':
        if stream is None:
            if history.next_chunk_index != 1:
                raise WorkerProtocolError('a new accepted stream must start at root chunk zero')
            stream = AcceptedStream(worker.run_dir, history=history.prefix_before(0),
                output_path='videos/accepted-stream-'+history.trajectory_id+'.mp4',
                chunk_count=worker._chunk_count, decoder=worker._accepted_decoder,
                gpu_lock=worker._render_gpu_lock,priority=priority=='1')
            worker._accepted_stream = stream
        stream.append(history)
        return {'history_head_sha256': history.head_sha256,
                'latent_chunk_count': history.next_chunk_index}
    if stream is None:
        raise WorkerProtocolError('accepted stream has not started')
    result = stream.finish(history)
    worker._accepted_stream = None
    return dict(result, decoder_worker_role=worker.role, decoder_pid=os.getpid())


def main(argv=None):
    from . import correction_worker as base
    if os.environ.get(ENV) != '1':
        raise ValueError('accepted stream worker requires explicit opt-in')
    original = base.CorrectionWorker
    base.CorrectionWorker = StreamingCorrectionWorker
    try:
        return base.main(argv)
    finally:
        base.CorrectionWorker = original


if __name__ == '__main__':
    raise SystemExit(main())
