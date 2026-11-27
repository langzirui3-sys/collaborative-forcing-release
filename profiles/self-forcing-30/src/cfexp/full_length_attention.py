"""Full-length single-sample FA2 fast path; currently diagnostic only."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import tarfile
import warnings

from .rope_coefficients import ARCHIVE_SHA

ATTENTION_SHA='02ad719a76e6bdec79d18fc62b094ef2ad7c28b2d593c9f395ae1226041b7c9d'
FLASH_FUNCTION_SHA='a3cf8df42b733dfee0b147a3528a7d270cbe2097a32725eebf39a0e09f521e82'
_DEFAULT=object()


def load_native(torch, flash_attn, archive_path: Path):
    if hashlib.sha256(archive_path.read_bytes()).hexdigest()!=ARCHIVE_SHA:
        raise ValueError('native SF source archive differs')
    with tarfile.open(archive_path) as archive:
        members=[m for m in archive.getmembers() if m.name.endswith('/wan/modules/attention.py')]
        if len(members)!=1 or not members[0].isfile():
            raise ValueError('native attention module missing or ambiguous')
        source=archive.extractfile(members[0]).read()
    if hashlib.sha256(source).hexdigest()!=ATTENTION_SHA:
        raise ValueError('native attention source differs')
    text=source.decode();nodes=[n for n in ast.parse(text).body
                               if isinstance(n,ast.FunctionDef) and n.name=='flash_attention']
    if len(nodes)!=1 or nodes[0].decorator_list:
        raise ValueError('native flash attention function missing or decorated')
    function=ast.get_source_segment(text,nodes[0])
    if hashlib.sha256(function.encode()).hexdigest()!=FLASH_FUNCTION_SHA:
        raise ValueError('native flash attention function differs')
    scope={'torch':torch,'flash_attn':flash_attn,'warnings':warnings,
           'FLASH_ATTN_2_AVAILABLE':True,'FLASH_ATTN_3_AVAILABLE':False}
    exec(compile(function,'<pinned-native-flash-attention>','exec'),scope)
    return scope['flash_attention']


class FullLengthAttention:
    """Skip varlen length setup only for the exact unmasked BF16 SF case.

    This stores no activations or KV and performs no tensor conversion. All
    unsupported arguments retain the original function, including validation.
    No generation worker installs this diagnostic helper.
    """
    def __init__(self,native,dense,torch):
        self.native,self.dense,self.torch=native,dense,torch
        self.calls=self.dense_calls=self.fallbacks=0

    def evidence(self):
        return {'calls':self.calls,'dense_calls':self.dense_calls,'fallbacks':self.fallbacks}

    def __call__(self,q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,
                 q_scale=None,causal=False,window_size=(-1,-1),deterministic=False,
                 dtype=_DEFAULT,version=None):
        t=self.torch;self.calls+=1
        dtype=t.bfloat16 if dtype is _DEFAULT else dtype
        eligible=(q_lens is None and k_lens is None and type(dropout_p) in (int,float)
            and dropout_p==0 and softmax_scale is None and q_scale is None
            and causal is False and window_size==(-1,-1) and deterministic is False
            and dtype==t.bfloat16 and (version is None or type(version) is int and version==2)
            and not t.is_grad_enabled()
            and all(isinstance(x,t.Tensor) and x.ndim==4 and x.device.type=='cuda'
                    and x.dtype==t.bfloat16 and not x.requires_grad and x.stride(-1)==1
                    for x in (q,k,v)))
        if eligible:
            eligible=(q.shape[0]==k.shape[0]==v.shape[0]==1 and k.shape==v.shape
                and tuple(q.shape[2:])==tuple(k.shape[2:])==(12,128)
                and q.shape[1]>0 and k.shape[1]>0 and q.device==k.device==v.device)
        if eligible:
            self.dense_calls+=1
            return self.dense(q,k,v,dropout_p=0.,softmax_scale=None,causal=False,
                              window_size=(-1,-1),deterministic=False)
        self.fallbacks+=1
        return self.native(q,k,v,q_lens=q_lens,k_lens=k_lens,dropout_p=dropout_p,
            softmax_scale=softmax_scale,q_scale=q_scale,causal=causal,window_size=window_size,
            deterministic=deterministic,dtype=dtype,version=version)
