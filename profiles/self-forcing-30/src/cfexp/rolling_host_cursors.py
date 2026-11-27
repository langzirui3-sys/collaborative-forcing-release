"""Move only Rolling's integer cache metadata; retain its native K/V tensors."""
import os

ENV = 'CFEXP_ROLLING_HOST_KV_CURSORS'
SCHEMA = 'rolling_native_host_kv_cursors_v1'
FIELDS = ('global_end_index', 'local_end_index')


def enabled():
    value = os.environ.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError(ENV + ' must be 0 or 1')
    return value == '1'


def initialize(caches, *, torch):
    """Called immediately after official zero-cache allocation, before any use."""
    if not isinstance(caches, list) or not caches:
        raise ValueError('missing native Rolling cache layers')
    replacements = []
    for layer in caches:
        k, v = layer['k'], layer['v']
        if k.ndim != 4 or v.shape != k.shape or v.device != k.device or v.dtype != k.dtype:
            raise ValueError('native Rolling K/V geometry differs')
        row = {}
        for name in FIELDS:
            cursor = layer[name]
            if (not isinstance(cursor, torch.Tensor) or cursor.dtype != torch.int64
                    or tuple(cursor.shape) != (1,) or cursor.requires_grad
                    or cursor.device != k.device or int(cursor.item()) != 0):
                raise ValueError('expected untouched native Rolling int64[1] cursor')
            row[name] = cursor.detach().to(device='cpu').clone()
        replacements.append(row)
    # No K/V reference is replaced and no attention/update function is patched.
    for layer, row in zip(caches, replacements):
        layer.update(row)


def evidence(caches, *, torch):
    values = []
    for layer in caches:
        row = {}
        for name in FIELDS:
            cursor = layer[name]
            if (cursor.device.type != 'cpu' or cursor.dtype != torch.int64
                    or tuple(cursor.shape) != (1,) or cursor.requires_grad):
                raise ValueError('native Rolling host cursor placement changed')
            row[name] = int(cursor.item())
        values.append(row)
    return {'schema': SCHEMA, 'cursor_device': 'cpu', 'cursor_dtype': 'int64',
            'cursor_shape': [1], 'native_kv_storage_preserved': True,
            'native_attention_and_updates_preserved': True, 'layers': values}


def validate_evidence(proof):
    if (not isinstance(proof, dict) or proof.get('schema') != SCHEMA
            or proof.get('cursor_device') != 'cpu' or proof.get('cursor_dtype') != 'int64'
            or proof.get('cursor_shape') != [1]
            or proof.get('native_kv_storage_preserved') is not True
            or proof.get('native_attention_and_updates_preserved') is not True
            or not isinstance(proof.get('layers'), list) or len(proof['layers']) != 30):
        raise ValueError('missing native Rolling host cursor evidence')
    for row in proof['layers']:
        if (not isinstance(row,dict) or set(row) != set(FIELDS)
                or any(type(row[x]) is not int or row[x] < 0 for x in FIELDS)):
            raise ValueError('invalid native Rolling cursor values')


def audit_checkpoints(campaign, key, chunk_count):
    import json
    from .erank_acceptance_pilot import bound_file
    rows=[]
    for index in range(chunk_count):
        path=campaign._checkpoint_path('rollouts',key,index)
        binding=bound_file(campaign.run_dir,path)
        generated=json.loads(path.read_text())['generated']
        expected_role='target' if index in (0,8,9,10,11) else 'draft'
        if (generated['context']['role'] != expected_role
                or generated['context']['next_chunk_index'] != index):
            raise ValueError('Rolling cursor checkpoint role or position differs')
        if generated['context']['role']=='draft':
            proof=generated.get('diagnostics',{}).get('rolling_state',{}).get('host_cursor_execution')
            validate_evidence(proof)
            rows.append({'chunk_index':index,'execution':proof,'checkpoint':binding})
    if len(rows)!=chunk_count-5:
        raise ValueError('Rolling fixed-route Draft receipt coverage differs')
    return {'schema':'rolling_host_cursor_campaign_audit_v1','draft_chunks':len(rows),'rows':rows}
