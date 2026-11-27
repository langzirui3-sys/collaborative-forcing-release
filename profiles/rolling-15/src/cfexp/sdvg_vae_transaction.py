"""Accepted-history transaction for the pinned causal Wan scoring VAE.

This module is not installed by existing SDVG launchers. It owns only decoder
feature state; Draft/Target diffusion KV, model parameters and RNG are untouched.
Native source verification belongs at the adapter/gate boundary below.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path

from .streaming_vae_probe import VAE_SHA

FIELDS=('_conv_num','_conv_idx','_feat_map')
SCHEMA='sdvg_accepted_vae_transaction_v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def native_owner_binding(owner):
    """Match actual loaded implementation bytes, not a claimed model name."""
    require(type(owner).__name__=='WanVAE_','transaction requires the pinned native WanVAE_')
    paths=[Path(inspect.getfile(getattr(type(owner),name))) for name in ('cached_decode','clear_cache')]
    require(len(set(paths))==1 and not paths[0].is_symlink() and paths[0].is_file(),
            'native VAE cache methods have different or unavailable source')
    digest=hashlib.sha256(paths[0].read_bytes()).hexdigest()
    require(digest==VAE_SHA,'native VAE implementation bytes differ')
    return dict(schema='native_wan_vae_decoder_owner_v1',source_sha256=digest,
                owner_class=type(owner).__name__,snapshot_fields=list(FIELDS))


def state_values(owner):
    count=getattr(owner,'_conv_num',None);index=getattr(owner,'_conv_idx',None)
    features=getattr(owner,'_feat_map',None)
    require(type(count) is int and 0<count<=512 and type(index) is list and len(index)==1
        and type(index[0]) is int and 0<=index[0]<=count
        and type(features) is list and len(features)==count,'native VAE decoder cache layout differs')
    return {k:getattr(owner,k) for k in FIELDS}


def clone_state(values,torch,*,max_bytes):
    """Copy only bounded feature state, preserving repeated object identities."""
    require(type(max_bytes) is int and max_bytes>=0,'invalid decoder snapshot budget')
    tensors={};visiting=set();nodes=0
    def inspect_value(value,depth=0):
        nonlocal nodes
        nodes+=1;require(nodes<=4096 and depth<=8,'decoder cache is not a bounded acyclic tree')
        if isinstance(value,torch.Tensor):
            require(not value.requires_grad,'decoder snapshots are inference-only')
            tensors[id(value)]=value;return
        if value is None or type(value) is int or (type(value) is str and value=='Rep'):return
        require(type(value) in (list,tuple,dict),'unsupported native decoder cache value')
        require(id(value) not in visiting,'cyclic decoder cache')
        visiting.add(id(value))
        if type(value) is dict:require(all(type(k) is str for k in value),'invalid decoder cache field')
        for item in (value.values() if type(value) is dict else value):inspect_value(item,depth+1)
        visiting.remove(id(value))
    inspect_value(values)
    size=sum(x.numel()*x.element_size() for x in tensors.values())
    require(size<=max_bytes,'decoder snapshot exceeds the declared memory budget')
    memo={}
    def clone(value):
        if id(value) in memo:return memo[id(value)]
        if isinstance(value,torch.Tensor):result=value.detach().clone()
        elif type(value) is dict:result={k:clone(v) for k,v in value.items()}
        elif type(value) is list:result=[clone(v) for v in value]
        elif type(value) is tuple:result=tuple(clone(v) for v in value)
        else:return value
        memo[id(value)]=result;return result
    return clone(values),dict(tensor_bytes=size,tensor_count=len(tensors),fields=list(FIELDS))


@dataclass
class DecoderSnapshot:
    values:dict
    accounting:dict

    @classmethod
    def capture(cls,owner,torch,*,max_bytes):
        values,counts=clone_state(state_values(owner),torch,max_bytes=max_bytes)
        return cls(values,counts)

    def restore(self,owner,torch,*,max_bytes):
        # Keep the saved prefix immutable even if selected decoding fails and
        # mutates the restored cache before the worker poisons itself.
        values,counts=clone_state(self.values,torch,max_bytes=max_bytes)
        for key in FIELDS:setattr(owner,key,values[key])
        state_values(owner)
        return counts


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return type(value) is str and len(value)==64 and set(value)<=set('0123456789abcdef')


class AcceptedVaeTransaction:
    """Exactly one proposal → one selected-output commit at each block.

    The caller verifies file hashes and the reward-derived decision. Rejection
    restores the accepted prefix, then decodes the actual selected Target latent.
    Accepting the identical Draft requires no additional decoder call.
    """
    def __init__(self,*,owner,torch,session_id,max_snapshot_bytes=16*1024**3):
        require(is_sha(session_id),'decoder transaction requires a bound session identity')
        require(type(max_snapshot_bytes) is int and max_snapshot_bytes>=0,
                'invalid decoder snapshot budget')
        state_values(owner)
        self.owner,self.torch,self.session_id=owner,torch,session_id
        self.max_bytes=max_snapshot_bytes
        self.pending=None;self.next_index=0;self.failed=False;self.closed=False
        self.head=sha(dict(schema=SCHEMA,session_id=session_id,accepted=[]))

    def available(self):
        require(not self.failed and not self.closed,'decoder transaction is failed or closed')

    def probe(self,*,chunk_index,candidate_sha256,decode_and_score):
        self.available()
        require(type(chunk_index) is int and chunk_index==self.next_index and self.pending is None
            and is_sha(candidate_sha256) and callable(decode_and_score),'decoder proposal order or binding differs')
        snapshot=DecoderSnapshot.capture(self.owner,self.torch,max_bytes=self.max_bytes)
        token=sha(dict(schema=SCHEMA,session_id=self.session_id,chunk_index=chunk_index,
                       candidate_sha256=candidate_sha256,accepted_head=self.head))
        try:
            result=decode_and_score()
            state_values(self.owner)
        except BaseException:
            self.failed=True;snapshot.restore(self.owner,self.torch,max_bytes=self.max_bytes);raise
        self.pending=dict(chunk_index=chunk_index,candidate_sha256=candidate_sha256,
                          token=token,snapshot=snapshot)
        return result,dict(schema=SCHEMA,pending_token=token,chunk_index=chunk_index,
            candidate_sha256=candidate_sha256,accepted_prefix_head_sha256=self.head,
            snapshot_accounting=dict(snapshot.accounting),not_yet_committed=True)

    def commit(self,*,pending_token,accepted,selected_sha256,decode_selected=None):
        self.available();pending=self.pending
        require(pending is not None and pending_token==pending['token'] and type(accepted) is bool
            and is_sha(selected_sha256),'decoder commit capability or selected binding differs')
        require((accepted and selected_sha256==pending['candidate_sha256'] and decode_selected is None)
            or (not accepted and callable(decode_selected)),'decoder accept/reject operation differs')
        snapshot=pending['snapshot'];restored=0;selected_result=None
        try:
            if not accepted:
                restored=snapshot.restore(self.owner,self.torch,max_bytes=self.max_bytes)['tensor_bytes']
                selected_result=decode_selected()
            state_values(self.owner)
        except BaseException:
            self.failed=True;snapshot.restore(self.owner,self.torch,max_bytes=self.max_bytes);raise
        previous=self.head
        self.head=sha(dict(schema=SCHEMA,previous=previous,chunk_index=self.next_index,selected_sha256=selected_sha256))
        receipt=dict(schema=SCHEMA,session_id=self.session_id,chunk_index=self.next_index,
            pending_token=pending_token,candidate_sha256=pending['candidate_sha256'],
            selected_sha256=selected_sha256,accepted_draft=accepted,
            operation='keep_scored_draft_cache' if accepted else 'restore_prefix_then_decode_selected_target',
            snapshot_accounting=dict(snapshot.accounting),restore_tensor_bytes=restored,
            selected_decode_calls=int(not accepted),accepted_prefix_head_before=previous,
            accepted_prefix_head_after=self.head,committed_blocks=self.next_index+1,
            diffusion_kv_untouched=True)
        self.next_index+=1;self.pending=None
        return selected_result,receipt

    def finish(self,*,expected_chunks):
        self.available()
        require(type(expected_chunks) is int and self.next_index==expected_chunks and self.pending is None,
                'decoder transaction finished before selected-history commits completed')
        self.closed=True
        return dict(schema=SCHEMA,committed_blocks=self.next_index,accepted_prefix_head_sha256=self.head)
