"""Read physical integer-index reuse receipts from immutable run artifacts."""
from __future__ import annotations

import hashlib
import json
import math


def require(value, message):
    if not value:
        raise ValueError(message)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_witness(diagnostics):
    descriptor = diagnostics['kv_witness']
    sha = diagnostics['kv_witness_sha256']
    require(object_sha(descriptor) == sha, 'index audit witness digest differs')
    constants = set()
    allocations = hits = samples = 0
    for layer in descriptor['layers'] + descriptor['cross_attention_layers']:
        for name in ('k', 'v'):
            record = layer[name]
            if not record['used_tokens']:
                continue
            samples += 1
            shape = list(record['shape'])
            shape[1] = record['used_tokens']
            size = math.prod(shape)
            key = (size, min(64, size))
            if key in constants:
                hits += 1
            else:
                allocations += 1
                if len(constants) < 64:
                    constants.add(key)
    e = diagnostics['kv_witness_index_reuse']
    expected = dict(schema='call_local_witness_index_execution_v1', enabled=True,
        sample_tensors=samples, allocations=allocations, hits=hits,
        resident_entries=len(constants), witness_sha256=sha, retained_model_values=False)
    require(e == expected, 'physical witness index allocation/reuse evidence differs')
    return samples, allocations, hits


def validate_rank(e):
    from .self_forcing_frame_adapter import deterministic_token_indices
    values = deterministic_token_indices(4680)
    sha = hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
    require(e.get('schema') == 'native_rank_index_execution_v1' and e.get('enabled') is True
        and e.get('counts') == dict(calls=1, hits=1, misses=0, allocations=0)
        and e.get('capacity') == 8 and e.get('resident_entries') == 1
        and e.get('resident_index_bytes') == 2048
        and e.get('constants') == [dict(token_count=4680, sample_count=256, sha256=sha)]
        and e.get('retained_model_values') is False
        and str(e.get('device', '')).startswith('cuda:'),
        'native rank index constants were not physically reused exactly once')


def audit_sealed_execution(frozen, marker, data, *, target, chunk_count=30, budget=4):
    prefix = 'attempts/' + marker['attempt_id'] + '/'
    counts = {row['prompt_id']: 0 for row in data['rows']}
    samples = allocations = hits = 0
    if target:
        require(len(counts) == len(data['rows']) and
                all(len(row['native_generation']['timing']['per_chunk']) == chunk_count
                    for row in data['rows']), 'Target index audit chunk coverage differs')
        records = [item for row in data['rows']
            for item in row['native_generation']['timing']['per_chunk']]
        for record in records:
            a, b, c = validate_witness(record)
            samples += a; allocations += b; hits += c
        rank_calls = 0
    else:
        seen = set()
        for name in frozen.entries:
            if not name.startswith(prefix) or '/state/rollouts/' not in name or not name.endswith('.json'):
                continue
            from .insight2_run import _load_audited_checkpoint
            checkpoint, history, context, segment, _ = _load_audited_checkpoint(
                frozen.file(name, label='index reuse checkpoint'), label='index reuse checkpoint')
            generated = checkpoint['generated']
            if generated['context']['role'] != 'draft':
                continue
            pid = context.noise.prompt_id
            key = (pid, context.next_chunk_index)
            require(pid in counts and key not in seen, 'index audit prompt or position differs')
            seen.add(key); counts[pid] += 1
            diagnostics = generated['diagnostics']
            validate_rank(diagnostics['rank_index_reuse'])
            a, b, c = validate_witness(diagnostics)
            samples += a; allocations += b; hits += c
        require(all(n == chunk_count - 1 - budget for n in counts.values()),
                'Draft index audit chunk coverage differs')
        rank_calls = sum(counts.values())
    return dict(schema='sealed_native_index_constants_audit_v1',
        rank_calls=rank_calls, sample_tensors=samples, allocations=allocations, hits=hits,
        exact_descriptor_bindings_verified=True, retained_model_values=False)
