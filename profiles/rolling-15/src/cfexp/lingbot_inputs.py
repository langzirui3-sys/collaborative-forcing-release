"""Deterministic, hash-bound LingBot camera extension without NumPy dependency."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from .lingbot_geometry import require_horizon

POLICY = 'preserve_prefix_hold_last_camera_zero_keys_v1'


def read_npy(data: bytes) -> tuple[dict, bytes]:
    if data[:8] not in (b'\x93NUMPY\x01\x00', b'\x93NUMPY\x02\x00'):
        raise ValueError('unsupported NPY version')
    width = 2 if data[6] == 1 else 4
    size = int.from_bytes(data[8:8 + width], 'little')
    if not 0 < size <= 65536:
        raise ValueError('invalid NPY header size')
    start = 8 + width + size
    header = ast.literal_eval(data[8 + width:start].decode('latin1').strip())
    if (set(header) != {'descr', 'fortran_order', 'shape'}
            or header['descr'] not in ('<f4', '<i4')
            or header['fortran_order'] is not False
            or not isinstance(header['shape'], tuple)
            or not header['shape']
            or any(type(n) is not int or n <= 0 for n in header['shape'])):
        raise ValueError('unsupported camera array layout')
    payload = data[start:]
    if len(payload) != math.prod(header['shape']) * 4:
        raise ValueError('NPY payload size mismatch')
    if header['descr'] == '<f4' and any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)):
        raise ValueError('nonfinite camera array')
    return header, payload


def resize_npy(data: bytes, frames: int, *, hold_last: bool) -> bytes:
    if type(frames) is not int or frames <= 0:
        raise ValueError('frames must be positive integer')
    header, payload = read_npy(data)
    rows = header['shape'][0]
    row_bytes = math.prod(header['shape'][1:]) * 4
    prefix = payload[:frames * row_bytes]
    if frames > rows:
        row = payload[-row_bytes:] if hold_last else bytes(row_bytes)
        prefix += row * (frames - rows)
    header['shape'] = (frames, *header['shape'][1:])
    encoded = repr(header).encode('latin1')
    encoded += b' ' * ((-(10 + len(encoded) + 1)) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + len(encoded).to_bytes(2, 'little') + encoded + prefix


def prepare_inputs(source_root: Path, protocol_path: Path, destination: Path, duration: int) -> dict:
    """Write a new, run-local input tree; never overwrite an existing destination."""
    run_root = Path(os.environ['CFEXP_RUN_DIR']).resolve()
    destination = destination.resolve()
    if destination == run_root or not destination.is_relative_to(run_root):
        raise ValueError('prepared inputs must be below CFEXP_RUN_DIR')
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    names = {m['example_directory'] for m in protocol['mappings']}
    staged = {}
    originals = {}
    for asset in protocol['assets']:
        relative = Path(asset['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe asset path')
        data = (source_root / relative).read_bytes()
        if len(data) != asset['bytes'] or hashlib.sha256(data).hexdigest() != asset['sha256']:
            raise ValueError('source asset hash mismatch: ' + asset['path'])
        originals[asset['path']] = data
        if relative.suffix == '.npy':
            header, _ = read_npy(data)
            if list(header['shape']) != asset['array_shape'] or header['descr'] != asset['array_dtype']:
                raise ValueError('source array metadata mismatch')
    conditions = []
    for name in sorted(names):
        pose_header, _ = read_npy(originals[name + '/poses.npy'])
        source_frames = pose_header['shape'][0]
        # Native duration geometry is computed independently of available source rows.
        contract = require_horizon(duration, duration * 16 + 1)
        frames = contract['requested_rgb_frames']
        contract = require_horizon(duration, frames)
        action_transforms = []
        absent_optional = []
        for filename in ('poses.npy', 'intrinsics.npy', 'action.npy', 'wasd_action.npy', 'ijkl_action.npy'):
            path = name + '/' + filename
            if filename == 'ijkl_action.npy' and path not in originals:
                absent_optional.append(filename)
                continue
            header, _ = read_npy(originals[path])
            camera = filename in ('poses.npy', 'intrinsics.npy')
            if camera and header['shape'][0] != source_frames:
                raise ValueError('unaligned source camera rows')
            data = originals[path]
            if not camera:
                # Some official action files outlive their camera trajectory.
                # Discard those extra keys before zero-padding the held camera.
                data = resize_npy(data, source_frames, hold_last=False)
                action_transforms.append({'file':filename,
                    'source_rows':header['shape'][0],
                    'discarded_rows_after_camera':max(0,header['shape'][0]-source_frames),
                    'zero_rows_before_camera_end':max(0,source_frames-header['shape'][0])})
            staged[path] = resize_npy(data, frames, hold_last=camera)
        staged[name + '/image.jpg'] = originals[name + '/image.jpg']
        conditions.append({'example_directory': name, 'source_frames': source_frames,
                           'padded_frames': max(0, frames - source_frames),
                           'action_transforms':action_transforms,
                           'absent_optional_action_files':absent_optional,
                           'padding_start_frame_zero_based': source_frames if frames > source_frames else None,
                           'temporal_contract': contract})
    receipt = {'schema': 'lingbot_prepared_inputs_v1', 'policy': POLICY,
               'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
               'nominal_duration_seconds': duration, 'conditions': conditions,
               'mappings': protocol['mappings'], 'prompt_source': protocol['prompt_source'],
               'report_note': 'LingBot runs at native 16 FPS; short official trajectories hold the last camera pose and intrinsics, with zero padded key actions. Text prompts remain unchanged.',
               'artifacts': [{'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                             for path, data in sorted(staged.items())]}
    destination.mkdir(parents=True, exist_ok=False)
    for path, data in staged.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    with (destination / 'input-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--duration', type=int, choices=(15, 30, 60), required=True)
    args = parser.parse_args()
    prepare_inputs(args.source_root, args.protocol, args.destination, args.duration)
