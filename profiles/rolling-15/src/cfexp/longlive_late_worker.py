"""Isolated opt-in RPCs for the late-handoff correctness gate."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import subprocess

from .accepted_stream_worker import StreamingCorrectionWorker
from .correction_worker import (_exact_keys, _json_value, _publish_staged,
    _discard_staged, _integer)
from .longlive_late_handoff import ENV, POLICIES, LateDraft, target_prediction, require, validate_trace, SCHEDULE
from .prefix_worker import file_hash

ACTIONS = {'late_propose', 'late_target', 'late_commit', 'late_continue'}


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_proposal(proposal):
    from .self_forcing_frame_adapter import validate_effective_rank_evidence
    require(isinstance(proposal,dict) and set(proposal)=={'schema','input_binding','result','pending_token'}
        and proposal['schema']=='longlive_late_proposal_v1', 'invalid late proposal receipt')
    b,v = proposal['input_binding'],proposal['result']
    require(isinstance(b,dict) and set(b)=={'prompt','chunk_index','internal_seed','files','hashes',
        'policy','renoise_paths','renoise_sha256'} and b['policy'] in POLICIES,
        'invalid late input binding')
    n=b['chunk_index'];updates,tau=POLICIES[b['policy']]
    require(type(n) is int and n>0 and type(b['internal_seed']) is int and b['internal_seed']>=0
        and isinstance(b['prompt'],str) and bool(b['prompt'].strip())
        and isinstance(b['files'],list) and isinstance(b['hashes'],list)
        and len(b['files'])==len(b['hashes'])==n+1
        and isinstance(b['renoise_paths'],list) and isinstance(b['renoise_sha256'],list)
        and len(b['renoise_paths'])==len(b['renoise_sha256'])==updates
        and len(set(b['files']+b['renoise_paths']))==n+1+updates,
        'incomplete late input identity or noise coverage')
    for digest in [*b['hashes'],*b['renoise_sha256'],proposal['pending_token']]:
        require(isinstance(digest,str) and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),
                'invalid late receipt digest')
    require(all(type(v[k]) is int for k in ('draft_clean_forwards','branch_snapshot_bytes','replayed_chunks','reused_chunks'))
        and v['policy']==b['policy'] and v['handoff_timestep']==tau
        and v['draft_steps']==SCHEDULE[:updates] and v['draft_clean_forwards']==0
        and v['branch_snapshot_bytes']==0 and v['replayed_chunks']==(1 if n==1 else 0)
        and v['reused_chunks']==(0 if n==1 else n), 'late proposal compute/reuse evidence differs')
    validate_trace(v['forward_trace'],SCHEDULE[:updates],start=n*4680)
    validate_effective_rank_evidence(v['rank_evidence'],expected_rank=v['rank'])
    meta=v['noisy_latent']
    require(meta['key']=='noisy_latent' and meta['shape']==[1,3,16,60,104] and meta['dtype']=='bfloat16',
            'late proposal intermediate tensor metadata differs')
    return True


@contextmanager
def capture(backend):
    from .self_forcing_frame_adapter import StepZeroCapture,LAYER_INDICES,EFFECTIVE_RANK_LAYER
    require(backend._active_capture is None, 'late operation overlapped another Draft capture')
    c = StepZeroCapture(layers=LAYER_INDICES,target_timestep=1000,
        sample_hidden=backend._sample_hidden,rank_sample=backend._rank_sample,
        retain_rank_layer=EFFECTIVE_RANK_LAYER,policy_rank_only=True)
    backend._active_capture = c
    try:
        yield c
    finally:
        backend._active_capture = None


def publish(worker, request, result, *, key):
    outputs = worker._output_map(request['output_paths'],required={key})
    relative, path = outputs[key]
    staged = []
    try:
        item = worker._stage_tensor(relative=relative,target=path,key=key,value=result[key])
        staged.append(item)
        value = _json_value({k:v for k,v in result.items() if k != key},label='late execution')
        value[key] = item.metadata
        _publish_staged(staged)
        return value
    except BaseException:
        _discard_staged(staged)
        raise


def inputs(worker, request):
    n = _integer(request['chunk_index'],label='late chunk index')
    require(1 <= n < worker._chunk_count, 'late handoff requires a full Target root')
    prompt = request['prompt']
    require(isinstance(prompt,str) and bool(prompt.strip()), 'missing late prompt')
    seed = _integer(request['internal_seed'],label='late seed')
    history, paths = worker._history(request['history_latent_paths'],expected_count=n)
    noise_path, noise = worker._load_input_tensor(request['initial_noise_path'],label='late initial noise',key='noise')
    files = [*paths,noise_path]
    binding = dict(prompt=prompt,chunk_index=n,internal_seed=seed,files=files,
                   hashes=[file_hash(worker,p) for p in files])
    return binding, history, noise


def session(worker, request, binding):
    identity = request['session_id']
    require(isinstance(identity,str) and len(identity)==64
        and all(x in '0123456789abcdef' for x in identity), 'invalid late session')
    current = getattr(worker,'_late_session',None)
    if current is None or identity != current['id']:
        require(binding['chunk_index']==1 and getattr(worker,'_late_pending',None) is None,
                'late session can only start after its own Target root')
        previous=getattr(worker,'_late_engine',None)
        if previous is not None:
            previous.early.payload=None
            worker.backend._early_draft=None
        worker._late_session = dict(id=identity,prompt=binding['prompt'],next=1)
        worker._late_engine = LateDraft(worker.backend)
        current = worker._late_session
    require(current['prompt']==binding['prompt'] and current['next']==binding['chunk_index'],
            'late session cursor or prompt differs')
    require(getattr(worker,'_late_pending',None) is None, 'late proposal still awaits its Target commit')
    return worker._late_engine


def propose(worker, request):
    from .self_forcing_frame_adapter import effective_rank_evidence_from_capture
    _exact_keys(request,worker._COMMON_KEYS | {'session_id','prompt','chunk_index','internal_seed',
        'history_latent_paths','initial_noise_path','renoise_paths','policy','output_paths','verify_parity'},label='late proposal')
    require(worker.role=='draft' and request['policy'] in POLICIES
        and type(request['verify_parity']) is bool, 'invalid late proposal role/policy')
    binding, history, noise = inputs(worker,request)
    engine = session(worker,request,binding)
    updates, tau = POLICIES[request['policy']]
    require(isinstance(request['renoise_paths'],list) and len(request['renoise_paths'])==updates,
            'late proposal requires every explicit re-noising draw')
    loaded = [worker._load_input_tensor(p,label='late renoise',key='noise') for p in request['renoise_paths']]
    paths = binding['files']+[p for p,_ in loaded]
    require(len(set(paths))==len(paths), 'late noise inputs alias')
    binding.update(policy=request['policy'],renoise_paths=[p for p,_ in loaded],
                   renoise_sha256=[file_hash(worker,p) for p,_ in loaded])
    require(not set(paths).intersection(x[0] for x in worker._output_map(request['output_paths'],required={'noisy_latent'}).values()),
            'late output aliases an input')
    t = worker.backend.torch
    args = dict(prompt=binding['prompt'],history=history.to(worker.backend.device),widths=(3,)*binding['chunk_index'],
        noise=noise.to(worker.backend.device),noises=[x.to(worker.backend.device) for _,x in loaded],policy=request['policy'])
    with capture(worker.backend) as c:
        result = engine.propose(**args)
        require(len(c.events)==1, 'late proposal must observe exactly its first Draft update')
        result['rank_evidence'] = effective_rank_evidence_from_capture(c.events[0])
        result['rank'] = c.events[0].ranks[14]
        if request['verify_parity']:
            from .longlive_late_oracle import draft_reference
            from .prefix_oracle import check_suffix
            reference_args = {k:v for k,v in args.items() if k!='widths'}
            expected, cache = draft_reference(worker.backend,**reference_args)
            result['proposal_oracle'] = check_suffix(actual=result['noisy_latent'],expected=expected,
                actual_cache=engine.early.payload,expected_cache=cache,torch=t)
    result['noisy_latent'] = result['noisy_latent'].detach().to('cpu').contiguous()
    result = publish(worker,request,result,key='noisy_latent')
    receipt = dict(schema='longlive_late_proposal_v1',input_binding=binding,result=result)
    receipt['pending_token'] = object_hash(dict(request_id=request['request_id'],proposal=receipt))
    validate_proposal(receipt)
    worker._late_pending = receipt
    return receipt


def target(worker, request):
    _exact_keys(request,worker._COMMON_KEYS | {'proposal','output_paths','verify_parity'},label='late Target')
    require(worker.role=='target' and type(request['verify_parity']) is bool, 'invalid late Target role')
    proposal = request['proposal']
    validate_proposal(proposal)
    b = proposal['input_binding'];value=proposal['result'];policy=b['policy']
    require(policy in POLICIES and value['policy']==policy and value['handoff_timestep']==POLICIES[policy][1],
            'late Target noise level differs')
    for path,digest in zip(b['files']+b['renoise_paths'],b['hashes']+b['renoise_sha256']):
        require(file_hash(worker,path)==digest, 'late proposal input changed')
    history, _ = worker._history(b['files'][:-1],expected_count=b['chunk_index'])
    meta = value['noisy_latent']
    path, noisy = worker._load_input_tensor(meta['path'],label='late noisy latent',key='noisy_latent')
    require(file_hash(worker,path)==meta['sha256'], 'late noisy latent changed')
    all_inputs = [*b['files'],*b['renoise_paths'],path]
    require(not set(all_inputs).intersection(x[0] for x in worker._output_map(request['output_paths'],required={'latent'}).values()),
            'late Target output aliases an input')
    args = dict(prompt=b['prompt'],history=history,noisy=noisy,timestep=POLICIES[policy][1],
                chunk_index=b['chunk_index'],internal_seed=b['internal_seed'])
    result = target_prediction(worker.backend,**args)
    if request['verify_parity']:
        from .longlive_late_oracle import target_reference
        expected = target_reference(worker.backend,**args)
        t = worker.backend.torch
        exact = bool(t.equal(result['latent'],expected))
        maximum = float((result['latent'].float()-expected.float()).abs().max())
        require(exact, 'late Target differs from the native one-entry block loop')
        result['target_oracle'] = dict(latent_exact=exact,maximum_absolute_difference=maximum,
            oracle='fresh_native_session_one_entry_schedule',no_quality_or_speed_claim=True)
    result = publish(worker,request,result,key='latent')
    return dict(schema='longlive_late_target_v1',proposal=proposal,result=result)


def commit(worker, request):
    _exact_keys(request,worker._COMMON_KEYS | {'session_id','pending_token','target','verify_parity'},label='late commit')
    pending = getattr(worker,'_late_pending',None)
    require(worker.role=='draft' and pending is not None
        and request['session_id']==worker._late_session['id']
        and request['pending_token']==pending['pending_token'] and type(request['verify_parity']) is bool,
        'late commit capability differs')
    validate_proposal(pending)
    response = request['target']
    require(response.get('schema')=='longlive_late_target_v1' and response.get('proposal')==pending,
            'late Target belongs to another proposal')
    b = pending['input_binding']
    for path,digest in zip(b['files']+b['renoise_paths'],b['hashes']+b['renoise_sha256']):
        require(file_hash(worker,path)==digest, 'late commit input changed')
    v = response['result'];meta=v['latent']
    require(v['denoising_schedule']==[POLICIES[b['policy']][1]]
        and v['completed_four_step_target_chunk'] is False, 'late Target sampling policy changed')
    path, clean = worker._load_input_tensor(meta['path'],label='late Target clean prediction',key='latent')
    require(file_hash(worker,path)==meta['sha256'], 'late Target prediction changed')
    history, _ = worker._history(b['files'][:-1],expected_count=b['chunk_index'])
    engine = worker._late_engine;t=worker.backend.torch
    args = dict(prompt=b['prompt'],history=history.to(worker.backend.device),
                widths=(3,)*b['chunk_index'],latent=clean)
    with capture(worker.backend):
        result = engine.commit(**args)
        if request['verify_parity']:
            from .longlive_late_oracle import draft_reference
            from .prefix_oracle import check_suffix
            accepted = clean.to(worker.backend.device)
            _,cache = draft_reference(worker.backend,prompt=b['prompt'],history=args['history'],
                                     noise=accepted,accepted=accepted)
            result['commit_oracle'] = check_suffix(actual=accepted,expected=accepted,
                actual_cache=engine.early.payload,expected_cache=cache,torch=t)
    worker._late_pending = None
    worker._late_session['next'] += 1
    return dict(schema='longlive_late_commit_v1',accepted_latent_sha256=meta['sha256'],result=result)


def continue_draft(worker, request):
    _exact_keys(request,worker._COMMON_KEYS | {'session_id','prompt','chunk_index','internal_seed',
        'history_latent_paths','initial_noise_path','output_paths'},label='late following Draft')
    require(worker.role=='draft', 'late continuation requires Draft')
    b,history,noise = inputs(worker,request)
    engine = session(worker,request,b)
    paths = b['files']
    require(not set(paths).intersection(x[0] for x in worker._output_map(request['output_paths'],required={'latent'}).values()),
            'late Draft output aliases an input')
    worker.backend._early_draft = engine.early
    result = worker.backend.generate_chunk(prompt=b['prompt'],history_latents=history,initial_noise=noise,
        internal_seed=b['internal_seed'],chunk_index=b['chunk_index'],capture_hidden=False,
        history_kv_widths=(3,)*b['chunk_index'],early_should_replace=lambda _m:False)
    require(result['kv_reuse']['replayed_chunks']==0, 'following late Draft replayed accepted history')
    result = publish(worker,request,result,key='latent')
    worker._late_session['next'] += 1
    return dict(schema='longlive_late_following_v1',input_binding=b,result=result)


class LateWorker(StreamingCorrectionWorker):
    def __init__(self, **kwargs):
        require(os.environ.get(ENV)=='1', 'late worker needs explicit diagnostic opt-in')
        super().__init__(**kwargs)
        if self.role=='draft':
            from .longlive_policy_worker import PrefixDiagnosticLongLiveBackend
            require(type(self.backend) is PrefixDiagnosticLongLiveBackend,
                    'late worker requires the native LongLive diagnostic backend')

    def close(self):
        engine=getattr(self,'_late_engine',None)
        if engine is not None:
            engine.early.payload=engine.early.pending=None
        return super().close()

    def _validate_common(self, request):
        if request.get('action') in ACTIONS:
            identity,_ = super()._validate_common(dict(request,action='model_state'))
            require(not self.poisoned, 'late worker is poisoned')
            return identity,request['action']
        return super()._validate_common(request)

    def handle(self, request):
        identity,action = self._validate_common(request)
        if action not in ACTIONS:
            require(not (self.role=='draft' and getattr(self,'_late_engine',None) is not None
                and action in {'generate','generate_early','commit_early_target','resume_early_target_prefix'}),
                'late session cannot share its native Draft cache with another execution path')
            return super().handle(request)
        try:
            with self._render_gpu_lock:
                fn = dict(late_propose=propose,late_target=target,late_commit=commit,late_continue=continue_draft)[action]
                return identity,action,fn(self,request)
        except BaseException as error:
            raise self._poison(error) from error


def process_factory(argv, **kwargs):
    require(os.environ.get(ENV)=='1' and list(argv[1:3])==['-m','cfexp.correction_worker'],
            'late launcher requires its explicit worker entrypoint')
    command = list(argv);command[2]='cfexp.longlive_late_worker'
    return subprocess.Popen(command,**kwargs)


def main(argv=None):
    from . import correction_worker as base
    require(os.environ.get(ENV)=='1', 'late worker needs explicit opt-in')
    original = base.CorrectionWorker;base.CorrectionWorker=LateWorker
    try:
        return base.main(argv)
    finally:
        base.CorrectionWorker=original


if __name__=='__main__':
    raise SystemExit(main())
