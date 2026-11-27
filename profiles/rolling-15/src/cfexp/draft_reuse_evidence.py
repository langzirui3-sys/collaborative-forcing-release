"""Validate honest incremental traces and worker-bound history identities."""
from collections.abc import Mapping, Sequence
import re


def validate_branch_anchor(value, *, history_chunks, reused_chunks, kv_branch_action=None):
    keys = {'schema', 'action', 'anchor_chunks', 'anchor_clone_count', 'anchor_clone_bytes',
            'anchor_replaced', 'anchor_consumed', 'anchor_miss_reason', 'anchor_retained',
            'resident_bytes', 'peak_resident_bytes', 'state_count'}
    if not isinstance(value, Mapping) or set(value) != keys or value['schema'] != 'draft_kv_branch_anchor_v1':
        raise ValueError('invalid Draft KV branch anchor evidence fields')
    action = value['action']
    if action not in ('none', 'preserve', 'consume'):
        raise ValueError('invalid Draft KV branch anchor action')
    if kv_branch_action is not None and action != kv_branch_action:
        raise ValueError('Draft KV anchor action differs from the request')
    for key in ('anchor_replaced', 'anchor_consumed', 'anchor_retained'):
        if type(value[key]) is not bool:
            raise ValueError('Draft KV anchor flags must be booleans')
    for key in ('anchor_clone_count', 'anchor_clone_bytes', 'resident_bytes',
                'peak_resident_bytes', 'state_count'):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('Draft KV anchor counts must be nonnegative integers')
    chunks = value['anchor_chunks']
    if chunks is not None and (type(chunks) is not int or chunks < 0):
        raise ValueError('Draft KV anchor history length is invalid')
    if (value['state_count'] not in (1, 2) or value['resident_bytes'] <= 0
            or value['peak_resident_bytes'] < value['resident_bytes']):
        raise ValueError('Draft KV anchor resident memory evidence is invalid')
    if action == 'preserve':
        if not (chunks == history_chunks and value['anchor_clone_count'] == 1
                and value['anchor_clone_bytes'] > 0 and value['anchor_retained']
                and not value['anchor_consumed'] and value['anchor_miss_reason'] is None
                and value['state_count'] == 2
                and value['resident_bytes'] == 2 * value['anchor_clone_bytes']):
            raise ValueError('Draft KV preserved anchor evidence is inconsistent')
    else:
        if value['anchor_clone_count'] or value['anchor_clone_bytes'] or value['anchor_replaced']:
            raise ValueError('Draft KV clone evidence requires preserve action')
        if action == 'consume':
            if value['anchor_retained']:
                raise ValueError('Draft KV consumed anchor must not remain pinned')
            if value['anchor_consumed']:
                if chunks != reused_chunks or value['anchor_miss_reason'] is not None:
                    raise ValueError('Draft KV consumed anchor differs from the reused prefix')
            elif chunks is not None or reused_chunks != 0 or value['anchor_miss_reason'] != 'absent':
                raise ValueError('Draft KV absent anchor must report full-prefix replay')
        elif (value['anchor_consumed'] or value['anchor_miss_reason'] is not None
              or value['anchor_retained'] != (chunks is not None)):
            raise ValueError('Draft KV ordinary continuation changed the pinned anchor')


def validate_reuse_evidence(value, *, history_chunks, history_file_sha256=None, prompt_sha256=None,
                            kv_branch_action=None, history_kv_widths=None):
    if kv_branch_action is not None and kv_branch_action not in ('none', 'preserve', 'consume'):
        raise ValueError('invalid requested Draft KV branch action')
    if value is None:
        if history_kv_widths is not None:
            raise ValueError('Draft KV construction evidence is missing')
        if kv_branch_action not in (None, 'none'):
            raise ValueError('Draft KV branch action is missing reuse evidence')
        return 0
    keys = {'schema','reused_chunks','replayed_chunks','scope_sha256','history_tensor_sha256',
            'history_file_sha256','prompt_sha256'}
    if isinstance(value, Mapping) and value.get('schema') == 'draft_kv_reuse_v2':
        keys = keys | {'history_kv_widths'}
        widths = value.get('history_kv_widths')
        if (not isinstance(widths, list) or len(widths) != history_chunks
                or any(type(w) is not int or w not in (1, 3) for w in widths)):
            raise ValueError('invalid Draft history KV widths')
        if history_kv_widths is not None and widths != list(history_kv_widths):
            raise ValueError('Draft KV construction differs from requested history')
    elif history_kv_widths is not None:
        raise ValueError('Draft KV construction evidence is missing')
    if (not isinstance(value, Mapping) or set(value) not in (keys, keys | {'branch_anchor'})
            or value['schema'] not in ('draft_kv_reuse_v1', 'draft_kv_reuse_v2')):
        raise ValueError('invalid Draft KV reuse evidence fields')
    reused, replayed = value['reused_chunks'], value['replayed_chunks']
    if type(reused) is not int or type(replayed) is not int or not 0 <= reused <= history_chunks or replayed != history_chunks-reused:
        raise ValueError('Draft reuse/replay counts disagree with history')
    def digest(x):
        return isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None
    if not digest(value['scope_sha256']) or not digest(value['prompt_sha256']):
        raise ValueError('Draft reuse scope or prompt digest invalid')
    for key in ('history_tensor_sha256','history_file_sha256'):
        items=value[key]
        if not isinstance(items,list) or len(items)!=history_chunks or not all(digest(x) for x in items):
            raise ValueError('Draft reuse history binding invalid')
    if history_file_sha256 is not None and value['history_file_sha256'] != list(history_file_sha256):
        raise ValueError('Draft reuse belongs to a different latent history')
    if prompt_sha256 is not None and value['prompt_sha256'] != prompt_sha256:
        raise ValueError('Draft reuse belongs to a different prompt')
    if 'branch_anchor' in value:
        validate_branch_anchor(value['branch_anchor'], history_chunks=history_chunks,
                               reused_chunks=reused, kv_branch_action=kv_branch_action)
    elif kv_branch_action not in (None, 'none'):
        raise ValueError('Draft KV branch action is missing anchor evidence')
    return reused
