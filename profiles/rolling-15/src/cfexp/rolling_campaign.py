"""Frozen native assets and common-noise preparation for Rolling campaigns."""
import json
from pathlib import Path

from .correction_runtime import DeterministicNoiseProvider
from .rolling_forcing_assets import SPEC_SHA256, _contained_file
from .rolling_forcing_native import SOURCE_COMMIT, _check_file
from .rolling_forcing_noise import seal_schedule
from .rolling_forcing_stream import RollingForcingError


def verify_draft_spec(path, *, model_root, repo_root):
    expected = _contained_file(repo_root, 'experiments/erank3-rolling-native-v1.artifacts.json')
    if Path(path).resolve(strict=True) != expected:
        raise RollingForcingError('Rolling campaign requires its pinned native specification')
    _check_file(expected, SPEC_SHA256)
    spec = json.loads(expected.read_text())
    if spec['schema'] != 'rolling_native_artifacts_v1' or spec['source_commit'] != SOURCE_COMMIT:
        raise RollingForcingError('Rolling campaign source identity mismatch')
    records = []
    for item in spec['artifacts']:
        if item['root'] != 'model':
            raise RollingForcingError('Rolling asset must belong to model root')
        asset = _contained_file(model_root, item['path'])
        _check_file(asset, item['sha256'])
        records.append(dict(name=item['name'], resolved_path=str(asset),
                            sha256=item['sha256'], size=asset.stat().st_size))
    return spec, records, 'cfexp.rolling_policy_worker'


def prepare_launch(*, run_dir, repo_root, model_root, prompts, seeds, chunk_count,
                   provider=None):
    """Seal all lookahead noise before worker startup; never infer future RNG."""
    root = Path(run_dir).resolve(strict=True)
    provider = provider or DeterministicNoiseProvider(run_dir=root)
    if provider.run_dir.resolve(strict=True) != root:
        raise RollingForcingError('Rolling noise provider belongs to another run')
    identities = set()
    bindings = []
    for prompt in prompts:
        for seed in seeds:
            identity = (prompt['prompt_id'], seed)
            if identity in identities:
                raise RollingForcingError('duplicate Rolling campaign prompt/seed')
            identities.add(identity)
            receipt = seal_schedule(provider=provider, prompt_id=prompt['prompt_id'],
                prompt=prompt['prompt'], trajectory_seed=seed, chunk_count=chunk_count)
            bindings.append(dict(run_dir=str(root), relative_path=receipt['path'],
                sha256=receipt['sha256'], prompt_id=prompt['prompt_id'], prompt=prompt['prompt'],
                trajectory_seed=seed, namespace=provider.namespace, chunk_count=chunk_count))
    if not bindings:
        raise RollingForcingError('Rolling campaign has no trajectories')
    return dict(repository_root=str(Path(repo_root).resolve(strict=True)),
        model_root=str(Path(model_root).resolve(strict=True)),
        musiq_relative_path='metrics/musiq/musiq_spaq_ckpt-358bb6af.pth', noise_schedules=bindings)
