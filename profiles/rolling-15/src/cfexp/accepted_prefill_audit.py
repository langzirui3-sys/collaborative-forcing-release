"""Validate actual prefill receipts without replacing the generation trace.

Receipt readers own file/index verification. This module joins independently
recorded native clean forwards solely for operation-order parity comparison;
the physical generation trace remains unmodified in each checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .accepted_prefill import object_sha,require
from .correction_state import LatentHistory


def canonical_trace(diagnostics, *, history, prompt, read_receipt, burst_length=4):
    """Return validated execution order and receipt paths for one Draft call."""
    from .rope_worker import validate_execution as validate_rope
    from .full_attention_worker import validate_execution as validate_attention
    e=diagnostics.get('accepted_prefill_execution',{})
    trace=diagnostics['forward_trace'];reuse=diagnostics['kv_reuse'];i=history.next_chunk_index
    prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
    require(e.get('schema')=='accepted_prefill_generation_v1'
        and e.get('chunk_index')==i and e.get('prompt_sha256')==prompt_sha
        and e.get('actual_generate_trace_sha256')==object_sha(trace),
        'accepted prefill generation identity or physical trace differs')
    require(type(burst_length) is int and burst_length in (2, 4), 'invalid Self-Forcing prefill burst length')
    prefills = burst_length - 1
    count=e.get('prefill_count');bindings=e.get('receipts')
    require(type(count) is int and count in (0, prefills) and isinstance(bindings,list)
        and len(bindings)==count,'accepted prefill receipt count differs')
    roles=[s.producer_role for s in history.segments]
    handoff = (i > burst_length and roles[-burst_length:] == ['target'] * burst_length
               and roles[-burst_length - 1] == 'draft')
    require((count == prefills) == handoff, 'accepted prefill must cover exactly the configured Target handoff')
    require(reuse['prompt_sha256']==prompt_sha
        and len(reuse['history_tensor_sha256'])==len(reuse['history_kv_widths'])==i,
        'accepted prefill consumer history binding differs')
    prefix=[];paths=[];ids=set();last_end=None
    for offset,binding in enumerate(bindings):
        require(isinstance(binding,dict) and set(binding)=={'path','sha256','nbytes'},
                'invalid accepted prefill file binding')
        r=read_receipt(binding);n=i-prefills+offset
        h=LatentHistory.from_dict(r['history']);native=r['native']
        require(r.get('schema')=='accepted_target_prefill_receipt_v1'
            and r.get('prompt_sha256')==prompt_sha and r.get('next_draft_chunk_index')==i
            and h==history.prefix_before(n),'prefill receipt belongs to another accepted history')
        ident=object_sha({'request_id':r['request_id'],
            'history_manifest_sha256':h.manifest_sha256,'next_draft_chunk_index':i,
            'prompt_sha256':prompt_sha})
        require(r['prefill_id']==ident and binding['path']=='provenance/accepted-target-prefill/'+ident+'.json'
            and ident not in ids,'duplicate or misidentified accepted prefill receipt')
        ids.add(ident);paths.append(binding['path'])
        require(native.get('schema')=='native_accepted_prefix_prefill_v1'
            and native.get('reused_chunks')==n-1 and native.get('replayed_chunks')==1
            and native.get('next_chunk_index')==n
            and native.get('scope_sha256')==reuse['scope_sha256']
            and native.get('history_binding')==reuse['history_tensor_sha256'][:n]
            and native.get('history_kv_widths')==reuse['history_kv_widths'][:n],
            'accepted prefill native cache prefix or model scope differs')
        expected_widths=[3 if s.producer_role=='draft' else 1 for s in h.segments]
        require(native['history_kv_widths']==expected_widths
            and native.get('noise_created') is False
            and native.get('scheduler_updates')==native.get('generated_chunks')==0
            and native.get('branch_snapshot_clone_bytes')==0
            and native.get('native_payload_reused') is True
            and type(native.get('native_allocation_bytes')) is int and native['native_allocation_bytes']>0
            and all(native.get(k) is True for k in ('torch_cpu_rng_preserved',
                'torch_cuda_rng_preserved','cuda_stream_completed')),
            'accepted prefill changed native construction, RNG or allocation semantics')
        t=native.get('started_monotonic');end=native.get('completed_monotonic')
        ms=native.get('cuda_stream_elapsed_ms_not_kernel_busy')
        require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (t,end,ms))
            and end>=t and (last_end is None or t>=last_end),
            'accepted prefill completion order or CUDA timing differs')
        last_end=end
        physical=[{'timestep':0.,'batch_timesteps_equal':True,'current_start':(n-1)*4680}]
        require(native['forward_trace']==physical,'accepted prefill performed something other than one native clean forward')
        validate_rope(native['rope_execution_evidence']);validate_attention(native['full_attention_execution_evidence'])
        for key,fields in (('rope_execution_evidence',('probe_gate_sha256','frequency_initialization_sha256')),
                           ('full_attention_execution_evidence',('probe_gate_sha256',))):
            require(all(native[key][f]==diagnostics[key][f] for f in fields),
                    'accepted prefill helper belongs to another gate or frequency source')
        prefix.extend(physical)
    if count:
        require(reuse['reused_chunks']==i-1 and reuse['replayed_chunks']==1,
                'accepted prefill was not reused at the next Draft block')
    require(e.get('canonical_native_trace_sha256')==object_sha(prefix+trace),
            'accepted prefill canonical operation order binding differs')
    return prefix+trace,paths


def local_receipt_reader(root):
    """Verify the immutable worker-written file before terminal sealing."""
    from .correction_worker import _input_path,_sha256_file
    def read(binding):
        _,path=_input_path(Path(root),binding['path'],label='accepted prefill receipt')
        require(_sha256_file(path)==(binding['sha256'],binding['nbytes']),
                'accepted prefill receipt bytes changed')
        return json.loads(path.read_text())
    return read


def audit_sealed_execution(frozen,marker,data,*,target,chunk_count=30,budget=4,burst_length=4):
    """Require indexed, exactly-once prefix receipts and every actual Draft call."""
    from .insight2_run import _load_audited_checkpoint
    require((burst_length, budget) in ((4, 4), (2, 6), (2, 8)),
            'unsupported Self-Forcing accepted prefill coverage')
    prefills = (budget // burst_length) * (burst_length - 1)
    prefix='attempts/'+marker['attempt_id']+'/'
    prompts={r['prompt_id']:r['prompt'] for r in data['rows']}
    counts={p:0 for p in prompts};receipt_counts={p:0 for p in prompts};used=set();seen=set()
    for name in frozen.entries:
        if not name.startswith(prefix) or '/state/rollouts/' not in name or not name.endswith('.json'):continue
        checkpoint,history,context,_,_=_load_audited_checkpoint(
            frozen.file(name,label='accepted prefill checkpoint'),label='accepted prefill checkpoint')
        if context.role!='draft':continue
        require(not target,'pure Target executed accepted-prefix Draft work')
        cell=name.split('/state/rollouts/',1)[0];pid=context.noise.prompt_id
        require(pid in prompts,'extra accepted prefill prompt')
        identity=(pid,context.next_chunk_index)
        require(identity not in seen,'duplicate accepted prefill generation position');seen.add(identity)
        def read(binding):
            relative=cell+'/'+binding['path']
            require(relative not in used,'prefill receipt used by more than one Draft generation')
            path=frozen.file(relative,label='accepted prefill receipt')
            require(frozen.entries[relative]==(binding['nbytes'],binding['sha256']),
                    'accepted prefill receipt is not bound to the sealed index')
            used.add(relative)
            return json.loads(path.read_text())
        canonical,paths=canonical_trace(checkpoint['generated']['diagnostics'],
            history=history.prefix_before(context.next_chunk_index),prompt=prompts[pid],
            read_receipt=read, burst_length=burst_length)
        counts[pid]+=1;receipt_counts[pid]+=len(paths)
    actual={n for n in frozen.entries if n.startswith(prefix) and '/provenance/accepted-target-prefill/' in n}
    require(actual==used,'orphan, duplicate or unindexed accepted prefill receipts')
    require(all(n == (0 if target else chunk_count - 1 - budget) for n in counts.values())
        and all(n == (0 if target else prefills) for n in receipt_counts.values()),
        'accepted prefill coverage differs from all frozen P8 prompts')
    return {'schema':'sealed_accepted_target_prefill_audit_v1','draft_generation_calls':sum(counts.values()),
        'separate_native_clean_prefix_calls':len(used),'exactly_once_indexed_receipts_verified':True,
        'canonical_trace_comparison_is_separate_from_physical_generation_trace':True}
