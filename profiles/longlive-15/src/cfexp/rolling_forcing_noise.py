"""Bind Rolling's lookahead noise to the common per-chunk noise provider.

No alternate RNG or future-seed inference is allowed. This preparation utility
is not yet wired into the formal policy worker; consumers must bind the returned
manifest digest in their immutable worker configuration.
"""
from __future__ import annotations

import hashlib
import json

from .correction_state import InitialNoiseDescriptor
from .correction_runtime import NOISE_SHAPE, _write_immutable_json
from .rolling_forcing_assets import _contained_file
from .rolling_forcing_stream import RollingForcingError

SCHEMA = 'rolling_shared_noise_schedule_v1'


def _require(value, message):
    if not value:
        raise RollingForcingError(message)


def seal_schedule(*, provider, prompt_id, prompt, trajectory_seed, chunk_count):
    _require(isinstance(prompt, str) and bool(prompt.strip()), 'empty schedule prompt')
    _require(type(chunk_count) is int and chunk_count in (16,30,60,120),
             'unsupported schedule chunk count')
    # Provider validation runs before prompt_id is used to construct a path.
    chunks = []
    for index in range(chunk_count):
        item = provider.materialize(prompt_id=prompt_id,
            trajectory_seed=trajectory_seed, chunk_index=index)
        chunks.append({'path':item.tensor_path, 'noise':item.descriptor.to_dict()})
    payload = dict(schema=SCHEMA, prompt_id=prompt_id, prompt=prompt,
        trajectory_seed=trajectory_seed, namespace=provider.namespace,
        chunk_count=chunk_count, chunks=chunks)
    relative = f'provenance/rolling-noise/{prompt_id}-seed-{trajectory_seed}.json'
    target = provider.run_dir / relative
    for part in [target, *target.parents]:
        if part == provider.run_dir:
            break
        _require(not part.is_symlink(), "schedule destination contains a symlink")
    _write_immutable_json(target, payload)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    verify_schedule(run_dir=provider.run_dir, relative_path=relative,
        sha256=digest, prompt=prompt, prompt_id=prompt_id,
        trajectory_seed=trajectory_seed, namespace=provider.namespace,
        chunk_count=chunk_count)
    return {'path':relative, 'sha256':digest}


def verify_schedule(*, run_dir, relative_path, sha256, prompt, prompt_id,
                    trajectory_seed, namespace, chunk_count):
    path = _contained_file(run_dir, relative_path)
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == sha256, 'noise schedule digest mismatch')
    payload = json.loads(raw)
    _require(set(payload) == {'schema','prompt_id','prompt','trajectory_seed',
                             'namespace','chunk_count','chunks'}, 'invalid schedule fields')
    _require(payload['schema'] == SCHEMA, 'invalid schedule schema')
    for key, expected in [('prompt',prompt), ('prompt_id',prompt_id),
                          ('trajectory_seed',trajectory_seed), ('namespace',namespace),
                          ('chunk_count',chunk_count)]:
        _require(payload[key] == expected, f'schedule {key} mismatch')
    _require(type(chunk_count) is int and chunk_count in (16,30,60,120),
             'unsupported schedule chunk count')
    _require(isinstance(payload['chunks'], list) and len(payload['chunks']) == chunk_count,
             'incomplete future noise schedule')
    paths = []
    for index, row in enumerate(payload['chunks']):
        _require(set(row) == {'path','noise'}, 'invalid schedule chunk fields')
        noise = InitialNoiseDescriptor.from_dict(row['noise'])
        _require((noise.prompt_id,noise.trajectory_seed,noise.namespace,noise.chunk_index)
                 == (prompt_id,trajectory_seed,namespace,index), 'noise trajectory or order mismatch')
        _require(noise.shape == NOISE_SHAPE and noise.dtype == 'bfloat16',
                 'noise geometry differs from common model contract')
        tensor_path = _contained_file(run_dir, row['path'])
        data = tensor_path.read_bytes()
        _require(len(data) == noise.tensor_nbytes and
                 hashlib.sha256(data).hexdigest() == noise.tensor_sha256,
                 'bound noise bytes changed')
        _require(tensor_path not in paths, 'duplicate noise file')
        paths.append(tensor_path)
    return payload, tuple(paths)


def load_schedule(*, torch, load_file, device, **binding):
    """Load only verified bytes and check the actual tensor before concatenation."""
    payload, paths = verify_schedule(**binding)
    tensors = []
    for row, path in zip(payload['chunks'], paths):
        # Read once more into owned bytes; verify these exact bytes, so a change
        # between verification and decoding cannot alter the loaded schedule.
        data = path.read_bytes()
        _require(hashlib.sha256(data).hexdigest() == row['noise']['tensor_sha256'],
                 'noise changed before tensor load')
        values = load_file(data)
        _require(set(values) == {'noise'}, 'unexpected noise tensor keys')
        value = values['noise']
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == NOISE_SHAPE
                 and value.dtype == torch.bfloat16, 'loaded noise geometry mismatch')
        _require(bool(torch.isfinite(value).all()), 'nonfinite bound noise')
        tensors.append(value)
    return torch.cat(tensors, dim=1).to(device=device), payload['trajectory_seed']
