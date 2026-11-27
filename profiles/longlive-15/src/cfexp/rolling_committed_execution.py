"""Explicit keep-current-Draft B4 execution bound to the original trained gate."""
from __future__ import annotations

import os

from .rolling_committed_proof import SOURCE, INDEX, RESULT, verify

ENV = 'CFEXP_ROLLING_COMMITTED_PREFIX'
PROOF_ENV = 'CFEXP_ROLLING_COMMITTED_PREFIX_PROOF_RUN'
POLICY = 'single_active_branch_committed_clean_prefix_v1'
BINDING = dict(source=SOURCE,artifact_index_sha256=INDEX,result_sha256=RESULT)


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
    require(os.environ.get('CFEXP_ROLLING_PREFIX_TRANSACTIONS','0')=='0'
            and os.environ.get('CFEXP_ROLLING_HOST_KV_CURSORS','0')=='1'
            and os.environ.get('CFEXP_ROLLING_HOST_WINDOW','0')=='1'
            and os.environ.get('CFEXP_ROLLING_CPU_HISTORY_IDENTITY','0')=='0',
            'committed prefix requires its verified native configuration and no rollback')
    require(bool(os.environ.get(PROOF_ENV)),'committed prefix needs its original sealed proof')
    proof=verify(os.environ[PROOF_ENV],repository_root=repository_root)
    require({k:proof[k] for k in BINDING}==BINDING,'committed prefix proof binding differs')
    return proof


def validate_state(state,*,index,previous):
    initial=previous is None
    missing=index if initial else index-previous-1
    reused=not initial and missing==0
    require(type(state.get('chunk_index')) is int and state['chunk_index']==index
            and state.get('state_policy')==POLICY
            and state.get('committed_prefix_proof')==BINDING
            and state.get('reused_rolling_state') is reused
            and state.get('discarded_future_state') is (not initial and not reused)
            and type(state.get('replayed_history_blocks')) is int,
            'committed prefix owner, branch or proof differs')
    from .rolling_host_cursors import validate_evidence
    cursor=state.get('host_cursor_execution');validate_evidence(cursor)
    require('cpu_history_identity' not in state and 'prefix_transaction' not in state
            and all(row['global_end_index']==(index+1)*4680
                    and row['local_end_index']==min((index+1)*4680,37440)
                    for row in cursor['layers']), 'committed prefix native cursor/history differs')
    require((index in (0,1) if initial else missing in (0,4)),
            'committed prefix requires its native root or a four-Target suffix')
    require(state.get('committed_prefix') == dict(schema='rolling_committed_clean_prefix_v1',
            action=('initial_clean_history' if initial else
                    'continued_native_window' if reused else 'accepted_prefix_and_target_suffix'),
            retained_prefix_blocks=0 if initial else previous+1,
            clean_suffix_blocks=4 if not initial and not reused else 0,
            full_history_replay_blocks=index if initial else 0,
            rollback_performed=False,checkpoint_undo_bytes=0,
            last_returned_draft_retained=not initial)
            and state['replayed_history_blocks']==(index if initial else missing),
            'committed prefix replay/rollback/suffix differs')
