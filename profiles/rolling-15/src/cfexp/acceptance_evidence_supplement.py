"""Publish omitted decode JSON in a new run, without changing a source seal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from .main_table_baseline_campaign import _run_dir, _write_new_json
from .main_table_merge import _load_frozen_index


def supplement(sources: list[tuple[Path, str]], output: Path) -> dict:
    if not sources or len({str(p.resolve()) for p, _ in sources}) != len(sources):
        raise ValueError('expected distinct sealed sources')
    if (output / 'result.json').exists() or (output / 'sources').exists():
        raise ValueError('supplement requires a fresh output directory')
    for parent in (output, *output.parents):
        if any((parent / name).exists() for name in ('_SUCCESS.json','_FAILED.json','artifact-index.json')):
            raise ValueError('supplement output is inside a sealed run')
    records = []
    for root, expected_index in sources:
        root = root.resolve(strict=True)
        if output.resolve().is_relative_to(root):
            raise ValueError('supplement cannot write into its source')
        index_bytes = (root / 'artifact-index.json').read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != expected_index:
            raise ValueError('source index differs from explicit pin')
        frozen, marker = _load_frozen_index(root)
        prefix = 'attempts/' + marker['attempt_id'] + '/'
        result_path = frozen.file(prefix+'result.json', label='source result')
        result_bytes = result_path.read_bytes()
        data = json.loads(result_bytes)
        if data['code_revision'] != marker['git_sha'] or data['status'] != 'succeeded':
            raise ValueError('source result and seal disagree')
        marker_bytes = (root / '_SUCCESS.json').read_bytes()
        destination = output / 'sources' / marker['run_id']
        destination.mkdir(parents=True, exist_ok=False)
        for name, raw in [('source-artifact-index.json', index_bytes), ('source-success.json', marker_bytes),
                          ('source-result.json', result_bytes)]:
            (destination / name).write_bytes(raw)
        evidence = []
        seen = set()
        for row in data['rows']:
            binding = row['decode_evidence']
            relative = binding['path']
            parts = PurePosixPath(relative)
            if (str(parts) != relative or parts.is_absolute() or '..' in parts.parts
                    or len(parts.parts) != 4 or parts.parts[0] != 'cells'
                    or parts.parts[2] != 'decoded' or parts.suffix != '.json'
                    or row['prompt_id'] in seen):
                raise ValueError('unexpected decode evidence path or duplicate prompt')
            seen.add(row['prompt_id'])
            source = root / prefix
            for part in parts.parts:
                source = source / part
                if source.is_symlink():
                    raise ValueError('decode evidence traverses a symlink')
            raw = source.read_bytes()
            if len(raw) != binding['nbytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
                raise ValueError('decode evidence differs from sealed result binding')
            json.loads(raw)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(raw)
            evidence.append({'prompt_id':row['prompt_id'], 'original_binding':binding,
                'supplement_path':target.relative_to(output).as_posix(),
                'was_in_original_index':prefix+relative in frozen.entries})
        if (root/'artifact-index.json').read_bytes() != index_bytes or (root/'_SUCCESS.json').read_bytes() != marker_bytes:
            raise ValueError('source seal changed while reading')
        records.append({'run_id':marker['run_id'], 'source_code_revision':marker['git_sha'],
            'source_index_sha256':expected_index,
            'source_marker_sha256':hashlib.sha256(marker_bytes).hexdigest(),
            'source_result_sha256':hashlib.sha256(result_bytes).hexdigest(),
            'source_destination':marker['artifact_publication']['destination'], 'evidence':evidence})
    result = {'schema':'cfexp_acceptance_decode_supplement_v1', 'status':'succeeded',
              'source_seals_modified':False, 'regenerated':False, 'sources':records}
    _write_new_json(output/'result.json', result)
    return result


def supplementary_decode(supplement_root: Path, source_marker: dict, source_result_sha: str,
                         prompt_id: str, binding: dict) -> Path:
    """Accept only a new sealed supplement bound to this original sealed result."""
    frozen, marker = _load_frozen_index(supplement_root)
    prefix = 'attempts/' + marker['attempt_id'] + '/'
    payload = json.loads(frozen.file(prefix+'result.json', label='supplement result').read_text())
    if (payload.get('schema') != 'cfexp_acceptance_decode_supplement_v1' or payload.get('status') != 'succeeded'
            or payload.get('source_seals_modified') is not False or payload.get('regenerated') is not False):
        raise ValueError('invalid evidence supplement')
    sources = [s for s in payload['sources'] if s['run_id'] == source_marker['run_id']]
    if len(sources) != 1:
        raise ValueError('supplement source is absent or ambiguous')
    source = sources[0]
    if (source['source_index_sha256'] != source_marker['artifact_index_sha256']
            or source['source_result_sha256'] != source_result_sha
            or source['source_code_revision'] != source_marker['git_sha']):
        raise ValueError('supplement belongs to a different sealed source')
    copied_marker = frozen.file(prefix+'sources/'+source_marker['run_id']+'/source-success.json',
                                label='original source marker copy')
    if json.loads(copied_marker.read_text()) != source_marker:
        raise ValueError('supplement original marker differs')
    matches = [r for r in source['evidence'] if r['prompt_id'] == prompt_id]
    if len(matches) != 1 or matches[0]['original_binding'] != binding:
        raise ValueError('supplement decode binding differs')
    relative = prefix + matches[0]['supplement_path']
    path = frozen.file(relative, label='supplementary decode evidence')
    if frozen.entries[relative] != (binding['nbytes'], binding['sha256']):
        raise ValueError('supplement file differs from original decode binding')
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', nargs=2, action='append', required=True, metavar=('RUN_DIR','INDEX_SHA256'))
    a=p.parse_args(argv)
    result=supplement([(Path(root), sha) for root,sha in a.source], _run_dir())
    print(json.dumps({'sources':len(result['sources']), 'decode_files':sum(len(x['evidence']) for x in result['sources'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
