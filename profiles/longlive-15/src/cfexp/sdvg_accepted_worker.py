"""Opt-in SDVG worker with one reward-cache commit per selected latent.

Ordinary native generation classes and the original SDVG wrapper remain intact.
"""
import os
import time

from .accepted_stream_worker import StreamingCorrectionWorker
from .correction_worker import (_exact_keys,_input_path,_sha256_file,_json_value,WorkerProtocolError)
from .sdvg_accepted_vae import AcceptedScoringVae,ENV,POLICY,ACTION
from .sdvg_vae_transaction import require


class AcceptedVaeWorker(StreamingCorrectionWorker):
    def __init__(self,**kwargs):
        require(os.environ.get(ENV)=='1','selected VAE worker requires explicit opt-in')
        super().__init__(**kwargs)
        self.accepted_vae=None;self._vae_candidate_path=None
        if self.role=='draft':
            self.accepted_vae=AcceptedScoringVae(self.backend,scope=str(self.run_dir))
            self._backend_model_state=dict(self._backend_model_state,sdvg_scoring_vae_policy=POLICY)
            if 'sdvg' in self._backend_model_state:
                self._backend_model_state['sdvg']=dict(self._backend_model_state['sdvg'],
                    officiality_label='benchmark_adaptation_with_selected_history_scoring_vae_v2')

    def _validate_common(self,request):
        if request.get('action')==ACTION:
            identity,_=super()._validate_common(dict(request,action='model_state'))
            require(not self.poisoned and self.role=='draft' and self.accepted_vae is not None,
                    'selected VAE commit requires a live Draft service')
            return identity,ACTION
        return super()._validate_common(request)

    def handle(self,request):
        identity,action=self._validate_common(request)
        service=self.accepted_vae
        if service is not None and service.pending is not None and action not in {
                ACTION,'shutdown','model_state','stream_set_role'}:
            raise WorkerProtocolError('selected scoring VAE must commit before the next operation')
        if action!=ACTION:return super().handle(request)
        _exact_keys(request,self._COMMON_KEYS|{'prompt_id','chunk_index','pending_token','accepted',
            'selected_role','selected_sha256','selected_latent_path'},label='selected VAE commit request')
        try:
            with self._render_gpu_lock:
                return identity,action,self._commit_selected_vae(request)
        except WorkerProtocolError:raise
        except BaseException as error:raise self._poison(error) from error

    def _start_sdvg_trajectory(self,request):
        _exact_keys(request,self._COMMON_KEYS|{'prompt_id','prompt','chunk_count'},label='selected VAE start')
        require(self.role=='draft' and request['chunk_count']==self._chunk_count,
                'selected VAE startup role/horizon differs')
        require(type(request['prompt_id']) is str and request['prompt_id'] and
                type(request['prompt']) is str and request['prompt'].strip(),'selected VAE prompt is invalid')
        try:return self.accepted_vae.start(prompt_id=request['prompt_id'],prompt=request['prompt'],
            chunk_count=request['chunk_count'])
        except BaseException as error:raise self._poison(error) from error

    def _score_sdvg_candidate(self,request):
        _exact_keys(request,self._COMMON_KEYS|{'prompt','chunk_index','clean_latent_path'},label='selected VAE score')
        require(self.role=='draft','selected VAE scoring requires Draft')
        relative,path=_input_path(self.run_dir,request['clean_latent_path'],label='selected VAE proposal')
        digest,_=_sha256_file(path)
        _,tensor=self._load_input_tensor(relative,label='selected VAE proposal',key='latent')
        try:
            result=self.accepted_vae.score(prompt=request['prompt'],chunk_index=request['chunk_index'],
                candidate_sha256=digest,clean_latent=tensor)
            require(_sha256_file(path)[0]==digest,'scored candidate file changed')
            self._vae_candidate_path=relative
            return dict(clean_latent_path=relative,result=_json_value(result,label='selected VAE reward'))
        except BaseException as error:raise self._poison(error) from error

    def _commit_selected_vae(self,request):
        from .stage_timing import timing_session,stage
        relative,path=_input_path(self.run_dir,request['selected_latent_path'],label='selected VAE selected latent')
        require(_sha256_file(path)[0]==request['selected_sha256'],'selected VAE latent hash differs')
        tensor=None
        if request['accepted'] is True:
            require(relative==self._vae_candidate_path,'accepted VAE latent path differs from scored proposal')
        else:
            from .correction_runtime import NOISE_SHAPE
            _,tensor=self._load_input_tensor(relative,label='selected VAE Target latent',key='latent')
            shape,dtype=self.codec.describe(tensor)
            require(tuple(shape)==tuple(NOISE_SHAPE) and dtype=='bfloat16',
                    'selected VAE Target latent geometry differs')
        started=time.perf_counter()
        with timing_session(self.run_dir,'worker-draft-selected-vae',dict(action=ACTION,role=self.role,
                request_id=request['request_id'],chunk_index=request['chunk_index'])) as session:
            if session is not None:session.attach_cuda(self.backend.torch,self.backend.base.device)
            with stage('selected_vae_cache_commit',cuda=True):
                receipt=self.accepted_vae.commit(prompt_id=request['prompt_id'],chunk_index=request['chunk_index'],
                    pending_token=request['pending_token'],accepted=request['accepted'],selected_role=request['selected_role'],
                    selected_sha256=request['selected_sha256'],selected_latent=tensor)
        require(_sha256_file(path)[0]==request['selected_sha256'],'selected VAE latent changed during commit')
        self._vae_candidate_path=None
        return dict(receipt,selected_latent_path=relative,worker_commit_wall_seconds=time.perf_counter()-started)

    def _finish_sdvg_trajectory(self,request):
        _exact_keys(request,self._COMMON_KEYS|{'prompt_id'},label='selected VAE finish')
        require(self.role=='draft','selected VAE completion requires Draft')
        try:return self.accepted_vae.finish(prompt_id=request['prompt_id'])
        except BaseException as error:raise self._poison(error) from error

    def close(self):
        service=getattr(self,'accepted_vae',None)
        if service is not None:
            tx=service.transaction
            if tx is not None:tx.pending=None;tx.closed=True
            service.pending=None;service.transaction=None;service.failed=True
            self.accepted_vae=None
        super().close()


def main(argv=None):
    from . import accepted_stream_worker as stream
    from .tracked_sdvg import observed_worker_class
    original=stream.StreamingCorrectionWorker
    stream.StreamingCorrectionWorker=observed_worker_class(AcceptedVaeWorker)
    try:return stream.main(argv)
    finally:stream.StreamingCorrectionWorker=original


if __name__=='__main__':raise SystemExit(main())
