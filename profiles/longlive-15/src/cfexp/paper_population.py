"""Explicit population/seed selection for the paper's 128 x 3 replication.

The historical 24-prompt runs retain their original inputs and seed.  This
mode requires a separately frozen, tracked population file; it never pads a
development cohort or treats repeated routing draws as generation seeds.
"""
import hashlib
import json
import os
from pathlib import Path

ENV = 'CFEXP_PAPER_REPRODUCTION'
SEED_ENV = 'CFEXP_GENERATION_SEED'
C10_ENV = 'CFEXP_PAPER_C10_CALIBRATION'
SEEDS = (20260911, 20260912, 20260913)
CALIBRATION_SEED = 20260911
PROTOCOL = 'experiments/paper-128x3-v1.protocol.json'
SOURCE_FILE = 'experiments/frozen-vbench-long-128-v1.source.json'
SOURCE_SHA256 = '8a2f89312402c2688cb00074194d748e264dd47e5a2f2e6fe010ca2afb0d9d94'
MODELS = ('krea_self_forcing', 'krea_rolling_forcing', 'krea_longlive')


def _text_identity(value):
    return ' '.join(value.casefold().split())


def _calibration_definition(root, data, evaluation_texts):
    reference = data.get('calibration_protocol')
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('paper population requires an explicit independent calibration binding')
    relative = reference['path']
    if (not isinstance(relative, str) or '\\' in relative or '\x00' in relative
            or Path(relative).is_absolute() or '..' in Path(relative).parts
            or str(Path(relative)) != relative or not relative.startswith('experiments/')):
        raise ValueError('unsafe paper calibration protocol path')
    path = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('paper calibration protocol may not traverse symlinks')
    if not path.is_file():
        raise ValueError('missing paper calibration protocol')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('paper calibration protocol digest differs')
    selected = _strict_json(path)['prompt_sources']['official_vbench2_long']['selected']
    calibration = [x for x in selected if x['split'] == 'calibration']
    texts = {_text_identity(x['prompt']) for x in calibration}
    if (len(calibration) != 10 or len(texts) != 10
            or len({x['prompt_id'] for x in calibration}) != 10
            or texts & {_text_identity(text) for text in evaluation_texts}):
        raise ValueError('paper evaluation overlaps or changes the frozen C10 control')
    return calibration, dict(reference)


def assert_calibration_compatible(root, protocol, *, calibration_run=False):
    """Bind the actual resolved campaign to paper128's independently frozen C10.

    Calibration smoke remains a separate 120-block, one-seed phase. Its flag
    never enables generation-population relabeling or imports a legacy receipt.
    """
    flag = os.environ.get(C10_ENV, '0')
    if flag not in ('0', '1') or (flag == '1' and (enabled() or not calibration_run)):
        raise ValueError('invalid independent paper calibration phase')
    if not enabled() and flag == '0':
        return None
    root = Path(root).resolve(strict=True)
    data = _strict_json(root / PROTOCOL)
    source_path = root / SOURCE_FILE
    if (data.get('schema') != 'paper_reproduction_population_v1'
            or data.get('status') != 'frozen'
            or data.get('source_manifest_sha256') != SOURCE_SHA256
            or source_path.is_symlink()
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA256):
        raise ValueError('paper calibration is not bound to the original evaluation source')
    source = _strict_json(source_path)
    expected, reference = _calibration_definition(
        root, data, [x['prompt'] for x in source['cases']])
    fields = ('prompt_id', 'prompt', 'source_index_zero_based', 'dimensions', 'selection_sha256')
    actual = tuple(tuple(row[field] for field in fields) for row in protocol.calibration_prompts)
    frozen = tuple(tuple(row[field] for field in fields) for row in expected)
    if actual != frozen:
        raise ValueError('active campaign uses a different C10 than the paper population')
    return dict(schema='paper128_calibration_binding_v1', protocol=reference,
        evaluation_source_sha256=SOURCE_SHA256, seed=CALIBRATION_SEED,
        prompt_ids=[x['prompt_id'] for x in expected], mode='calibration' if flag == '1' else 'evaluation')


def enabled(env=None):
    env = os.environ if env is None else env
    flag = env.get(ENV, '0')
    if flag not in ('0', '1'):
        raise ValueError('invalid paper replication switch')
    return flag == '1'


def generation_seed(env=None):
    env = os.environ if env is None else env
    if not enabled(env):
        if SEED_ENV in env:
            raise ValueError('a new generation seed requires explicit paper replication')
        return CALIBRATION_SEED
    raw = env.get(SEED_ENV)
    if raw not in tuple(map(str, SEEDS)):
        raise ValueError('paper replication requires one of its three explicit seeds')
    return int(raw)


def population_label(legacy):
    return 'paper128_three_seed_replication' if enabled() else legacy


def validate_execution(*, single_gpu, seconds, full_population=True):
    if enabled():
        generation_seed()
        if single_gpu is not True or seconds not in (15, 30) or full_population is not True:
            raise ValueError('paper replication requires its full-population 15/30s single-GPU driver')


def _strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate population field')
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs)


def _pointer(source, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('missing original source field pointer')
    value = source
    for key in pointer[1:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not key.isdecimal() or str(int(key)) != key:
                raise ValueError('invalid original source list position')
            value = value[int(key)]
        else:
            value = value[key]
    return value


def load(root, *, model_pair):
    if not enabled() or model_pair not in MODELS:
        raise ValueError('unsupported paper population/model')
    root = Path(root).resolve(strict=True)
    path = root / PROTOCOL
    if path.is_symlink() or not path.is_file():
        raise ValueError('the original 128-prompt population has not been frozen')
    raw = path.read_bytes()
    data = _strict_json(path)
    if (data.get('schema') != 'paper_reproduction_population_v1'
            or data.get('status') != 'frozen' or data.get('prompt_count') != 128
            or data.get('seeds') != list(SEEDS)
            or data.get('source_manifest_sha256') != SOURCE_SHA256):
        raise ValueError('paper population identity, count, seeds or source differs')
    source_path = root / SOURCE_FILE
    if (source_path.is_symlink() or not source_path.is_file()
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA256):
        raise ValueError('original source bytes do not match the recorded 128-prompt input')
    source = _strict_json(source_path)
    prompts = data.get('prompts', [])
    if (len(prompts) != 128 or len({x['id'] for x in prompts}) != 128
            or len({x['text'].strip() for x in prompts}) != 128
            or any(not x['id'] or not x['text'].strip() for x in prompts)
            or [x['source_selection_ordinal'] for x in prompts] != list(range(1, 129))):
        raise ValueError('missing, duplicate or reordered original prompt cases')
    counts = {}
    for prompt in prompts:
        for field in ('id', 'text', 'assigned_dimension', 'source_selection_ordinal'):
            if _pointer(source, prompt['source_fields'][field]) != prompt[field]:
                raise ValueError('normalized case differs from the original source')
        dim = prompt['assigned_dimension']
        if not isinstance(dim, str) or not dim:
            raise ValueError('missing source dimension')
        counts[dim] = counts.get(dim, 0) + 1
    if len(counts) != 16 or set(counts.values()) != {8}:
        raise ValueError('the original 16-by-8 population balance differs')
    _, calibration_reference = _calibration_definition(root, data, [p['text'] for p in prompts])
    seed = generation_seed()
    index = os.environ.get('CFEXP_PAPER_PROMPT_SHARD_INDEX', '0')
    count = os.environ.get('CFEXP_PAPER_PROMPT_SHARD_COUNT', '1')
    if not index.isdecimal() or not count.isdecimal():
        raise ValueError('invalid paper prompt shard')
    index, count = int(index), int(count)
    if not 1 <= count <= 128 or not 0 <= index < count:
        raise ValueError('paper prompt shard is out of range')
    selected = prompts[index::count]
    binding = dict(schema='paper128_replication_binding_v1', protocol=PROTOCOL,
        sha256=hashlib.sha256(raw).hexdigest(), source_manifest_sha256=SOURCE_SHA256,
        model_pair=model_pair, seed=seed, generation_seeds=list(SEEDS),
        full_population_count=128, full_case_count=384, shard_index=index,
        shard_count=count, selected_prompt_ids=[p['id'] for p in selected],
        selected_case_count=len(selected), calibration_seed=CALIBRATION_SEED)
    binding['calibration_protocol'] = calibration_reference['path']
    binding['calibration_protocol_sha256'] = calibration_reference['sha256']
    return [{'id': p['id'], 'text': p['text']} for p in selected], binding
