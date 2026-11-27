"""Opt-in production boundary for the separately verified native Rolling undo path."""
from __future__ import annotations

import os

from .rolling_transaction_proof import FAST_SOURCE, FAST_INDEX, FAST_RESULT, verify_fast

ENV = 'CFEXP_ROLLING_PREFIX_TRANSACTIONS'
PROOF_ENV = 'CFEXP_ROLLING_PREFIX_EXECUTION_PROOF_RUN'
POLICY = 'single_active_branch_clean_prefix_transaction_v1'
BINDING = dict(source=FAST_SOURCE,artifact_index_sha256=FAST_INDEX,result_sha256=FAST_RESULT)


def require(value,message):
    if not value:
        raise ValueError(message)


def enabled():
    value=os.environ.get(ENV,'0')
    require(value in ('0','1'),ENV+' must be 0 or 1')
    return value=='1'


def load_proof(*,repository_root):
    if not enabled():
        return None
    require(os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS','0')=='1'
            and os.environ.get('CFEXP_ROLLING_HOST_WINDOW','0')=='1'
            and os.environ.get('CFEXP_ROLLING_CPU_HISTORY_IDENTITY','0')=='0',
            'Rolling prefix execution requires its verified host metadata and GPU history configuration')
    require(bool(os.environ.get(PROOF_ENV)),'Rolling prefix execution needs its original sealed proof')
    proof=verify_fast(os.environ[PROOF_ENV],repository_root=repository_root)
    require({k:proof[k] for k in BINDING}==BINDING,'Rolling prefix proof binding differs')
    return proof


def validate_state(state,*,index,previous):
    """Ours B4 route only: one rejected candidate, then four accepted Target blocks."""
    missing=index if previous is None else index-previous-1
    reused=previous is not None and missing==0
    require(type(state.get('chunk_index')) is int and state['chunk_index']==index
            and state.get('state_policy')==POLICY
            and state.get('prefix_transaction_proof')==BINDING
            and state.get('reused_rolling_state') is reused
            and state.get('discarded_future_state') is (previous is not None and not reused),
            'Rolling prefix state owner, branch or proof differs')
    from .rolling_host_cursors import validate_evidence
    cursor=state.get('host_cursor_execution');validate_evidence(cursor)
    require('cpu_history_identity' not in state and all(
                row['global_end_index']==(index+1)*4680
                and row['local_end_index']==min((index+1)*4680,37440)
                for row in cursor['layers']), 'Rolling prefix actual cursor/history configuration differs')
    tx=state['prefix_transaction']
    require(tx['schema']=='rolling_native_prefix_transaction_v1'
            and type(state['replayed_history_blocks']) is int
            and tx['checkpoint_undo_bytes']==(862617600 if index>=8 else 0),
            'Rolling prefix checkpoint geometry differs')
    if previous is None:
        require(tx['action']=='full_replay' and tx['retained_prefix_blocks']==0
                and tx['clean_suffix_blocks']==0 and tx['full_history_replay_blocks']==index
                and tx['rollback'] is None and state['replayed_history_blocks']==index,
                'Rolling prefix initialization differs')
    elif reused:
        require(tx['action']=='continued' and tx['retained_prefix_blocks']==index
                and tx['clean_suffix_blocks']==tx['full_history_replay_blocks']==0
                and tx['rollback'] is None and state['replayed_history_blocks']==0,
                'Rolling prefix ordinary continuation differs')
    else:
        prefix=previous+1
        undo=tx['rollback']
        require(missing==4 and tx['action']=='retained_after_rollback'
                and tx['retained_prefix_blocks']==prefix
                and tx['clean_suffix_blocks']==state['replayed_history_blocks']==4
                and tx['full_history_replay_blocks']==0
                and undo['schema']=='rolling_native_prefix_rollback_v1'
                and undo['prefix_blocks']==prefix
                and undo['evicted_tokens_per_layer']==[4680 if prefix>=8 else 0]*30
                and undo['copied_undo_bytes']==(862617600 if prefix>=8 else 0)
                and undo['unused_slots_restored'] is False,
                'Rolling prefix replacement silently replayed or retained a wrong branch')
