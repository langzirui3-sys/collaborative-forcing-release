"""Materialize pinned Rolling Forcing assets inside one formal run only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from .rolling_forcing_native import SOURCE_COMMIT, _check_file
from .rolling_forcing_stream import RollingForcingError

SPEC_SHA256 = 'b1dbc91174fc0f4c3fe1ad0eb74c1103244538e9375fe301ff7625e4e6fe87bd'


def _contained_file(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise RollingForcingError('unsafe native asset path')
    candidate = root / relative
    for part in [candidate, *candidate.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise RollingForcingError('native asset path contains a symlink')
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RollingForcingError('native asset escaped its root')
    return resolved


def materialize(*, run_dir, repository_root, model_root):
    """Verify every byte before extracting code; do not import upstream here."""
    from .longlive_adapter import _safe_extract_tar
    from .self_forcing_frame_adapter import _safe_extract_wheel

    run_dir = Path(run_dir).resolve(strict=True)
    environment = os.environ.get('CFEXP_RUN_DIR')
    if not environment or Path(environment).resolve(strict=True) != run_dir:
        raise RollingForcingError('native assets require the assigned CFEXP_RUN_DIR')
    repository_root = Path(repository_root).resolve(strict=True)
    if run_dir.is_relative_to(repository_root):
        raise RollingForcingError('native runtime must not be materialized in repository')
    spec_path = _contained_file(repository_root, 'experiments/erank3-rolling-native-v1.artifacts.json')
    _check_file(spec_path, SPEC_SHA256)
    spec = json.loads(spec_path.read_text())
    if spec['schema'] != 'rolling_native_artifacts_v1' or spec['source_commit'] != SOURCE_COMMIT:
        raise RollingForcingError('native asset identity mismatch')
    artifacts, records = {}, []
    for item in spec['artifacts']:
        if item['root'] != 'model' or item['name'] in artifacts:
            raise RollingForcingError('invalid native asset binding')
        path = _contained_file(model_root, item['path'])
        _check_file(path, item['sha256'])
        artifacts[item['name']] = path
        records.append({**item, 'resolved_path': str(path), 'size': path.stat().st_size})
    scratch = run_dir / 'tmp' / 'rolling-native'
    if scratch.exists() or scratch.is_symlink() or scratch.parent.is_symlink():
        raise RollingForcingError('native scratch is already present or redirected')
    _safe_extract_tar(artifacts['rolling_source'], scratch)
    source = scratch / f'Causal-Forcing-{SOURCE_COMMIT}' / 'long_video'
    for relative in ('pipeline/rolling_forcing_inference.py', 'utils/wan_wrapper.py',
                     'configs/default_config.yaml', 'configs/rolling_forcing_dmd.yaml'):
        _contained_file(source, relative)
    for mapping in spec['runtime_files']:
        destination = source / mapping['destination']
        if destination.exists() or destination.is_symlink():
            raise RollingForcingError('native runtime destination already exists')
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = artifacts[mapping['artifact']]
        try:
            os.link(original, destination)
        except OSError:
            shutil.copy2(original, destination)
        item = next(x for x in spec['artifacts'] if x['name'] == mapping['artifact'])
        _check_file(destination, item['sha256'])
    _safe_extract_wheel(artifacts['flash_attn_wheel'], source / '_runtime_site')
    return source, artifacts, records
