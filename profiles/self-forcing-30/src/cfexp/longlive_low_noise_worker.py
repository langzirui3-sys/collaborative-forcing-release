"""Separate opt-in RPC endpoint; frozen prefix workers remain unchanged."""
from __future__ import annotations
from collections.abc import Mapping
import os
import subprocess

from .accepted_stream_worker import StreamingCorrectionWorker
from .correction_worker import _exact_keys, WorkerProtocolError, WorkerExecutionError
from .longlive_low_noise import ACTION, ENV, execute, require
from .prefix_worker import file_hash
from .target_prefix import validate_prefix_trace


class LowNoiseWorker(StreamingCorrectionWorker):
    def __init__(self, **kwargs):
        require(os.environ.get(ENV) == '1', 'low-noise worker requires explicit opt-in')
        super().__init__(**kwargs)
        if self.role == 'draft':
            from .longlive_policy_worker import PrefixDiagnosticLongLiveBackend
            try:
                require(type(self.backend) is PrefixDiagnosticLongLiveBackend,
                        'low-noise worker requires the native LongLive prefix backend')
                require(not self.backend.session.host_cursors.enabled,
                        'low-noise gate has not validated host cursors')
            except BaseException:
                self.close()
                raise

    def _validate_common(self, request):
        if request.get('action') == ACTION:
            identity, _ = super()._validate_common(dict(request,action='model_state'))
            if self.poisoned:
                raise WorkerProtocolError('poisoned low-noise worker cannot generate')
            return identity, ACTION
        return super()._validate_common(request)

    def handle(self, request):
        identity, action = self._validate_common(request)
        if action != ACTION:
            return super().handle(request)
        try:
            with self._render_gpu_lock:
                return identity, action, resume(self,request)
        except BaseException as error:
            raise self._poison(error) from error


def resume(worker, request):
    _exact_keys(request,worker._COMMON_KEYS | {'early_session_id','pending_token','prefix',
        'renoise_path','output_paths','verify_parity'},label='LongLive low-noise request')
    pending = getattr(worker,'_early_pending',None)
    require(worker.role == 'draft' and pending is not None,'missing stopped native Draft')
    require(request['early_session_id'] == pending['session_id']
        and request['pending_token'] == pending['token'],'low-noise pending capability differs')
    verify = request['verify_parity']
    require(type(verify) is bool,'invalid low-noise parity flag')
    prefix = request['prefix']
    require(isinstance(prefix,Mapping) and set(prefix) == {'schema','input_binding','result'}
        and prefix['schema'] == 'target_first_update_handoff_v1','invalid low-noise Target receipt')
    require(prefix['input_binding'] == {k:pending[k] for k in
        ('prompt','chunk_index','internal_seed','files','hashes')},'Target used another history/noise')
    for path,digest in zip(pending['files'],pending['hashes']):
        require(file_hash(worker,path) == digest,'pending low-noise input changed')
    value = prefix['result']
    require(isinstance(value,Mapping) and value.get('completed_target_chunk') is False,
            'low-noise Target prediction was labelled as a complete block')
    validate_prefix_trace(value['forward_trace'],history_frames=3*pending['chunk_index'])
    meta = value['first_clean_prediction']
    require(meta.get('key') == 'first_clean_prediction','wrong Target tensor key')
    path,prediction = worker._load_input_tensor(meta['path'],label='Target prediction',key='first_clean_prediction')
    require(file_hash(worker,path) == meta['sha256'],'Target prediction bytes changed')
    noise_path,noise = worker._load_input_tensor(request['renoise_path'],label='625 noise',key='noise')
    inputs = [*pending['files'],path,noise_path]
    require(len(set(inputs)) == len(inputs),'low-noise inputs alias')
    outputs = worker._output_map(request['output_paths'],required={'latent'})
    require(not set(inputs).intersection(v[0] for v in outputs.values()),'low-noise output aliases an input')
    history,paths = worker._history(pending['history_paths'],expected_count=pending['chunk_index'])
    result = execute(worker.backend,prompt=pending['prompt'],history=history,
        widths=tuple(pending['widths']),prediction=prediction,noise=noise,verify_parity=verify)
    result.update(target_prefix=dict(prefix),explicit_renoise=[dict(path=noise_path,sha256=file_hash(worker,noise_path))])
    payload = worker._finalize_generate_result(request=request,target_attempt_id=None,
        target_logical_call_id=None,outputs=outputs,history=history,history_paths=paths,
        noise_path=pending['files'][-1],result=result,capture_hidden=False,chunk_index=pending['chunk_index'])
    worker._early_pending = None
    worker._early_session['next'] += 1
    return payload


def process_factory(argv, **kwargs):
    require(os.environ.get(ENV) == '1' and list(argv[1:3]) == ['-m','cfexp.correction_worker'],
            'low-noise launcher requires its explicit worker entrypoint')
    command = list(argv)
    command[2] = 'cfexp.longlive_low_noise_worker'
    return subprocess.Popen(command,**kwargs)


def main(argv=None):
    from . import correction_worker as base
    require(os.environ.get(ENV) == '1','low-noise worker requires explicit opt-in')
    original = base.CorrectionWorker
    base.CorrectionWorker = LowNoiseWorker
    try:
        return base.main(argv)
    finally:
        base.CorrectionWorker = original


if __name__ == '__main__':
    raise SystemExit(main())
