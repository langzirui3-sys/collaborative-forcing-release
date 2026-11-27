"""Bounded asynchronous accepted-prefix work on the existing Draft worker."""
from __future__ import annotations

import hashlib
import queue
import threading
from types import SimpleNamespace

from .accepted_prefill import burst_length, native_prefill, object_sha, require
from .correction_state import LatentHistory

ACTION='accepted_target_prefill_start'


class AcceptedPrefillWorker:
    """One persistent thread, one active job and at most three pending receipts.

    Every model operation holds the existing worker GPU lock. A normal request
    waits for the previous prefill before acquiring that lock. CUDA completion
    is owned by ``native_prefill``; an acknowledgement never claims completion.
    """
    def __init__(self,worker,*,operation=native_prefill,timeout=120.,longlive_split=False):
        self.worker=worker;self.operation=operation;self.timeout=timeout
        self.prefills_per_burst = 1 if longlive_split else burst_length() - 1
        self.target_kv_width = 3 if longlive_split else 1
        self.jobs=queue.Queue(maxsize=1);self.current=None;self.thread=None
        self.pending=[];self.error=None;self.closed=False
        self.original_generate=worker.backend.generate_chunk
        def generate(**kwargs):
            require(self.current is None,'prefill barrier must precede the model lock')
            require(not self.closed and self.error is None,'accepted prefill worker is not healthy')
            pending=list(self.pending)
            if pending:
                require(len(pending)==self.prefills_per_burst,
                        'accepted Target burst requires all intermediate prefix updates')
                require(all(p['receipt']['next_draft_chunk_index']==kwargs['chunk_index']
                    and p['receipt']['prompt_sha256']==hashlib.sha256(kwargs['prompt'].encode()).hexdigest()
                    for p in pending),'prefill receipts belong to another generation')
            result=self.original_generate(**kwargs)
            trace=result['forward_trace'];prefix=[]
            if pending:
                history=result['kv_reuse']['history_tensor_sha256']
                widths=result['kv_reuse']['history_kv_widths']
                for item in pending:
                    native=item['receipt']['native'];n=native['next_chunk_index']
                    require(list(native['history_binding'])==history[:n]
                        and list(native['history_kv_widths'])==widths[:n],
                        'prefilled history differs from the final accepted Draft input')
                    prefix.extend(native['forward_trace'])
                require(result['kv_reuse']['replayed_chunks']==1
                    and result['kv_reuse']['reused_chunks']==kwargs['chunk_index']-1,
                    'handoff did not reuse the prefetched Target blocks')
            evidence={'schema':'accepted_prefill_generation_v1','chunk_index':kwargs['chunk_index'],
                'prompt_sha256':hashlib.sha256(kwargs['prompt'].encode()).hexdigest(),
                'prefill_count':len(pending),'receipts':[p['binding'] for p in pending],
                'actual_generate_trace_sha256':object_sha(trace),
                'canonical_native_trace_sha256':object_sha(prefix+trace)}
            self.pending.clear()
            return dict(result,accepted_prefill_execution=evidence)
        self.generate=generate;worker.backend.generate_chunk=generate

    def barrier(self):
        if self.current is not None:
            require(self.current.done.wait(self.timeout),'accepted prefill completion timed out')
            if self.current.error is not None:
                raise RuntimeError('native accepted prefill failed') from self.current.error
            self.current=None
        require(self.error is None,'accepted prefill thread previously failed')

    def start(self,request):
        from .correction_worker import _exact_keys,WorkerProtocolError
        _exact_keys(request,self.worker._COMMON_KEYS|{'prompt','history','next_draft_chunk_index'},
                    label='accepted prefill request')
        self.barrier()
        require(not self.closed,'accepted prefill worker is closed')
        history=LatentHistory.from_dict(request['history']);future=request['next_draft_chunk_index']
        prompt=request['prompt'];n=history.next_chunk_index
        if (not isinstance(prompt,str) or not prompt.strip() or n<2
                or history.segments[-1].producer_role!='target' or type(future) is not int
                or not n<future<=n+self.prefills_per_burst or future>=self.worker._chunk_count):
            raise WorkerProtocolError('prefill must bind an accepted Target prefix before a future Draft block')
        if self.pending:
            last=self.pending[-1]['receipt']
            require(len(self.pending)<self.prefills_per_burst and last['next_draft_chunk_index']==future
                and last['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
                and last['history']['next_chunk_index']+1==n,
                'prefill queue crossed its accepted burst, prompt or bound')
        ident=object_sha({'request_id':request['request_id'],
            'history_manifest_sha256':history.manifest_sha256,'next_draft_chunk_index':future,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
        op=SimpleNamespace(ident=ident,request_id=request['request_id'],history=history,
            future=future,prompt=prompt,done=threading.Event(),error=None)
        if self.thread is None:
            self.thread=threading.Thread(target=self._loop,name='cfexp-accepted-prefill',daemon=False)
            self.thread.start()
        require(self.thread.is_alive(),'accepted prefill thread stopped')
        self.current=op;self.jobs.put(op,timeout=1.)
        return {'schema':'accepted_prefill_scheduled_v1','prefill_id':ident,
            'history_manifest_sha256':history.manifest_sha256,
            'next_draft_chunk_index':future,'queued':True}

    def _loop(self):
        while True:
            op=self.jobs.get()
            if op is None:return
            try:
                with self.worker._render_gpu_lock:self._perform(op)
            except BaseException as error:
                op.error=error;self.error=error
            finally:op.done.set()
            if op.error is not None:return

    def _perform(self,op):
        from .correction_runtime import _write_immutable_json
        from .correction_worker import _sha256_file
        from .stage_timing import _unsealed_directory,timing_session
        worker=self.worker;root=_unsealed_directory(worker.run_dir);history=op.history
        history.verify_files(root)
        value,paths=worker._history([s.latent_path for s in history.segments],
                                    expected_count=history.next_chunk_index)
        require(paths==[s.latent_path for s in history.segments],'prefill file identity changed')
        history.verify_files(root)
        widths=tuple(3 if s.producer_role=='draft' else self.target_kv_width for s in history.segments)
        with timing_session(root,'worker-draft-accepted-prefill',{'action':ACTION,'role':'draft',
                'request_id':op.request_id,'chunk_index':history.next_chunk_index,
                'prompt_sha256':hashlib.sha256(op.prompt.encode()).hexdigest()}):
            native=self.operation(worker.backend,run_dir=root,prompt=op.prompt,
                                 history_latents=value,history_kv_widths=widths)
        require(native['replayed_chunks']==1 and native['reused_chunks']==history.next_chunk_index-1
            and native['next_chunk_index']==history.next_chunk_index,
            'asynchronous accepted prefill must advance exactly one Target block')
        receipt={'schema':'accepted_target_prefill_receipt_v1','prefill_id':op.ident,
            'request_id':op.request_id,'prompt_sha256':hashlib.sha256(op.prompt.encode()).hexdigest(),
            'next_draft_chunk_index':op.future,'history':history.to_dict(),'native':native}
        relative='provenance/accepted-target-prefill/'+op.ident+'.json';path=root/relative
        _write_immutable_json(path,receipt);sha,size=_sha256_file(path)
        self.pending.append({'binding':{'path':relative,'sha256':sha,'nbytes':size},'receipt':receipt})

    def close(self):
        if self.closed:return
        failure=None
        try:self.barrier()
        except BaseException as error:failure=error
        if self.thread is not None:
            if self.thread.is_alive() and self.current is None:
                self.jobs.put(None,timeout=1.)
            self.thread.join(self.timeout)
            require(not self.thread.is_alive(),'accepted prefill thread failed to stop')
        require(self.worker.backend.generate_chunk is self.generate,'accepted prefill wrapper changed')
        self.worker.backend.generate_chunk=self.original_generate;self.pending.clear();self.closed=True
        if failure is not None:raise failure


class AcceptedPrefillProxy:
    def __init__(self,client,*,longlive_split=False):
        model_id = 'longlive_1.3b' if longlive_split else 'self_forcing_1.3b'
        require(client.role=='draft' and client.model_id==model_id,
                'accepted prefill requires its declared native Draft consumer')
        self.client=client

    def start(self,*,prompt,history,next_draft_chunk_index):
        result=self.client._request(ACTION,{'prompt':prompt.prompt,'history':history.to_dict(),
                                           'next_draft_chunk_index':next_draft_chunk_index})
        require(result.get('schema')=='accepted_prefill_scheduled_v1' and result.get('queued') is True
            and result.get('history_manifest_sha256')==history.manifest_sha256
            and result.get('next_draft_chunk_index')==next_draft_chunk_index
            and isinstance(result.get('prefill_id'),str) and len(result['prefill_id'])==64,
            'worker did not acknowledge the same accepted prefix prefill')
        return result
