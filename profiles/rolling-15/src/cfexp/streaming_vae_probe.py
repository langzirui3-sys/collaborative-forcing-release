"""Native cached VAE parity and stage costs on a sealed latent trajectory."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tarfile
import time

VAE_SHA = 'a4c1d9362f25a1bef463ac51caed4097317c007d4d87887b02a3a733fdbc48d5'
WRAPPER_SHA = 'da919629bac4472cd9072223f1ab03200de276e202ab0d9b91e6789d500ff8fa'
ARCHIVE_SHA = '2ec791e1224511b4b62971a0c236a918d960bdba38676957cfc090b1f85954bf'
WEIGHTS_SHA = '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'


def normalization(source):
    """Read the exact native wrapper constants without executing the wrapper."""
    if hashlib.sha256(source).hexdigest() != WRAPPER_SHA:
        raise ValueError('native wrapper source differs from frozen version')
    tree = ast.parse(source.decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanVAEWrapper')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    values = {}
    for node in init.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('mean', 'std'):
                if name in values:
                    raise ValueError('duplicate normalization constants')
                values[name] = ast.literal_eval(node.value)
    if (set(values) != {'mean', 'std'} or any(len(v) != 16 for v in values.values())
            or any(type(x) not in (int, float) or not math.isfinite(x) for v in values.values() for x in v)
            or any(x <= 0 for x in values['std'])):
        raise ValueError('invalid native VAE normalization')
    return values


def to_pixels(torch, video):
    # Same float/clamp/layout/quantization order as the frozen wrapper + adapter.
    return (video.float().clamp(-1, 1)[0].permute(1, 0, 2, 3)
            .add(1.0).mul(127.5).round().clamp(0, 255)
            .to(device='cpu', dtype=torch.uint8).contiguous())


def decode_history(torch, model, chunks, scale, *, device, streamed):
    """Only committed chunks; native VAE owns its separate causal feature cache."""
    model.clear_cache()
    try:
        if not streamed:
            latent = torch.cat(chunks, dim=1).to(device=device, dtype=torch.bfloat16).contiguous()
            return to_pixels(torch, model.decode(latent.permute(0, 2, 1, 3, 4), scale))
        pixels = []
        for chunk in chunks:
            latent = chunk.to(device=device, dtype=torch.bfloat16).contiguous()
            pixels.append(to_pixels(torch, model.cached_decode(latent.permute(0, 2, 1, 3, 4), scale)))
        return torch.cat(pixels, dim=0)
    finally:
        model.clear_cache()


def run(spec_path, source_config, *, render_controls=False, async_render=False):
    from .main_table_baseline_campaign import (_run_dir, _model_root, _visible_devices,
        _sha256_file, _write_new_json, _gpu_uuid, _gpu_compute_pids)
    from .main_table_ours_campaign import _assert_idle_pair
    from .main_table_merge import _load_frozen_index
    from .correction_state import LatentHistory
    from .stage_timing import _unsealed_directory
    from .metadata import collect_git_metadata, find_git_root
    from .video_only_decode import pixel_witness
    from .video_encoding_policy import policy, SERIAL, writer_options, verify_video
    root = _unsealed_directory(_run_dir())
    if policy() != SERIAL:
        raise ValueError('parity diagnostic requires the frozen serial encoder')
    source = json.loads(source_config.read_text())
    if set(source) != {'schema', 'source_run', 'artifact_index_sha256'} or source['schema'] != 'streaming_vae_source_v1':
        raise ValueError('invalid frozen source configuration')
    source_root = Path(source['source_run'])
    if source_root.is_symlink() or not source_root.is_absolute() or source_root.resolve(strict=True) != source_root:
        raise ValueError('source run must be an absolute canonical directory')
    if _sha256_file(source_root / 'artifact-index.json')[0] != source['artifact_index_sha256']:
        raise ValueError('source index differs from frozen binding')
    index, marker = _load_frozen_index(source_root)
    index.verify_all()
    prefix = 'attempts/' + marker['attempt_id'] + '/'
    route = json.loads(index.file(prefix + 'raw/fixed-route.json', label='source route').read_text())
    history = LatentHistory.from_dict(route['history'])
    if history.next_chunk_index != 30:
        raise ValueError('parity diagnostic requires the complete 30-chunk source')
    source_attempt = source_root / 'attempts' / marker['attempt_id']
    # These raw tensors are excluded as standalone index entries. The indexed
    # history binds every actual file path, size and hash, which must still verify.
    history.verify_files(source_attempt)
    spec = json.loads(spec_path.read_text())
    model_root = _model_root()
    def artifact(name, expected):
        row = next(x for x in spec['artifacts'] if x['name'] == name)
        path = (model_root / row['path']).resolve(strict=True)
        if (row['root'] != 'model' or row['sha256'] != expected
                or not path.is_relative_to(model_root) or _sha256_file(path)[0] != expected):
            raise ValueError('model artifact differs from frozen binding: ' + name)
        return path
    archive = artifact('official_self_forcing_source_archive', ARCHIVE_SHA)
    checkpoint = artifact('wan_vae_checkpoint', WEIGHTS_SHA)
    with tarfile.open(archive) as tar:
        def member(suffix):
            names = [n for n in tar.getnames() if n.endswith(suffix)]
            if len(names) != 1:
                raise ValueError('ambiguous frozen source member')
            return tar.extractfile(names[0]).read()
        vae_bytes = member('/wan/modules/vae.py')
        constants = normalization(member('/utils/wan_wrapper.py'))
    if hashlib.sha256(vae_bytes).hexdigest() != VAE_SHA:
        raise ValueError('native VAE source differs from frozen version')
    path = root / 'tmp' / 'native-vae.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(vae_bytes)
    module_spec = importlib.util.spec_from_file_location('cfexp_frozen_sf_vae_probe', path)
    native = importlib.util.module_from_spec(module_spec)
    devices = _visible_devices()
    _assert_idle_pair(*devices)
    module_spec.loader.exec_module(native)
    import torch
    from safetensors.torch import load_file
    from torchvision.io import write_video
    chunks = []
    for segment in history.segments:
        tensors = load_file(str(source_attempt / segment.latent_path), device='cpu')
        latent = tensors.get('latent')
        if (set(tensors) != {'latent'} or latent is None or tuple(latent.shape) != (1, 3, 16, 60, 104)
                or latent.dtype != torch.bfloat16 or not bool(torch.isfinite(latent).all())):
            raise ValueError('source chunk violates the frozen latent contract')
        chunks.append(latent)
    result = {'schema': 'streaming_vae_parity_probe_v1', 'source': source,
              'source_marker': marker, 'history_sha256': history.manifest_sha256,
              'history_head_sha256': history.head_sha256, 'native_vae_sha256': VAE_SHA,
              'native_wrapper_sha256': WRAPPER_SHA, 'native_checkpoint_sha256': WEIGHTS_SHA,
              'torch_version': str(torch.__version__), 'cudnn_version': torch.backends.cudnn.version(),
              'git': collect_git_metadata(find_git_root(Path.cwd())),
              'native_generation_changed': False, 'overlap_performed': False,
              'formal_video_speedup': False, 'quality_measured': False,
              'render_controls_performed': render_controls,
              'async_decode_encode_performed': async_render, 'rows': []}
    def exclusive():
        if any(not _gpu_compute_pids(_gpu_uuid(d)) <= {os.getpid()} for d in devices):
            raise ValueError('selected GPU acquired an unrelated compute process')
    for ordinal in range(2):
        exclusive()
        with torch.cuda.device(ordinal), torch.inference_mode():
            started = time.perf_counter()
            # Match pipeline.to(BF16) followed by vae.to(device), without loading
            # either generative model or substituting the different Krea decoder.
            model = native._video_vae(pretrained_path=str(checkpoint), z_dim=16).eval().requires_grad_(False)
            model.to(dtype=torch.bfloat16)
            model.to(device='cuda')
            scale = [torch.tensor(constants['mean'], dtype=torch.float32).to('cuda', torch.bfloat16),
                     1.0 / torch.tensor(constants['std'], dtype=torch.float32).to('cuda', torch.bfloat16)]
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - started
            decode_history(torch, model, chunks[:1], scale, device='cuda', streamed=False)
            frames, records = {}, {}
            order = (False, True) if ordinal == 0 else (True, False)
            for streamed in order:
                exclusive()
                name = 'streamed' if streamed else 'whole'
                torch.cuda.synchronize()
                started = time.perf_counter()
                frames[name] = decode_history(torch, model, chunks, scale, device='cuda', streamed=streamed)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if tuple(frames[name].shape) != (357, 3, 480, 832):
                    raise ValueError('decoded frame geometry differs from frozen native output')
                pixels = pixel_witness(frames[name])
                video = root / 'videos' / ('gpu-' + str(ordinal) + '-' + name + '.mp4')
                video.parent.mkdir(exist_ok=True)
                started = time.perf_counter()
                encoded = frames[name].permute(0, 2, 3, 1).contiguous()
                write_video(str(video), encoded, fps=24, **writer_options())
                verify_video(video)
                encode_seconds = time.perf_counter() - started
                digest, size = _sha256_file(video)
                records[name] = {'decode_wall_seconds': elapsed, 'encode_wall_seconds': encode_seconds,
                                 'pixels': pixels, 'video_sha256': digest, 'video_nbytes': size,
                                 'video_path': video.relative_to(root).as_posix()}
                del encoded
            controls = {}
            if render_controls:
                from .streaming_video import StreamingVideoWriter, decoded_pixel_witness, LOSSLESS_CONTROL
                expected_pixels = records['whole']['pixels']['raw_sha256']
                variants = [('serial_batch', SERIAL, False), ('serial_chunks', SERIAL, True),
                            ('lossless_batch', LOSSLESS_CONTROL, False), ('lossless_chunks', LOSSLESS_CONTROL, True)]
                if ordinal == 1:
                    variants.reverse()
                for label, selected_policy, chunked in variants:
                    exclusive()
                    relative = 'videos/gpu-' + str(ordinal) + '-' + label + '.mp4'
                    started = time.perf_counter()
                    writer = StreamingVideoWriter(root, relative, expected_frames=357, policy=selected_policy)
                    data = frames['streamed' if chunked else 'whole']
                    try:
                        if chunked:
                            writer.push(data[:9])
                            for offset in range(9, 357, 12):
                                writer.push(data[offset:offset+12])
                        else:
                            writer.push(data)
                        evidence = writer.finish()
                    finally:
                        writer.abort()
                    encode_seconds = time.perf_counter() - started
                    witness = decoded_pixel_witness(root / relative)
                    controls[label] = {'video': evidence, 'encode_wall_seconds': encode_seconds,
                        'decoded_pixels': witness, 'encoder_input_matches_native_pixels':
                            evidence['input_rgb_tchw_sha256'] == expected_pixels,
                        'decoded_pixels_match_native_pixels': witness['raw_sha256'] == expected_pixels}
                for name in records:
                    records[name]['decoded_pixels'] = decoded_pixel_witness(root / records[name]['video_path'])
            async_records = {}
            if async_render:
                from .async_video import AsyncVideoWriter
                from .streaming_video import decoded_pixel_witness
                # Preserve frozen primary ABR settings. This measures only
                # overlap of native GPU decode with CPU encode, not generation.
                for asynchronous in ((False, True) if ordinal == 0 else (True, False)):
                    from .streaming_video import StreamingVideoWriter
                    exclusive()
                    label = 'async' if asynchronous else 'sync'
                    relative = 'videos/gpu-' + str(ordinal) + '-' + label + '-decode-encode.mp4'
                    factory = AsyncVideoWriter if asynchronous else StreamingVideoWriter
                    model.clear_cache()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    writer = factory(root, relative, expected_frames=357)
                    try:
                        for chunk in chunks:
                            latent = chunk.to(device='cuda', dtype=torch.bfloat16).contiguous()
                            pixels = to_pixels(torch, model.cached_decode(latent.permute(0, 2, 1, 3, 4), scale))
                            writer.push(pixels)
                        evidence = writer.finish()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                    finally:
                        writer.abort()
                        model.clear_cache()
                    async_records[label] = {'decode_encode_ready_wall_seconds': elapsed,
                        'video': evidence,
                        'encoder_input_matches_native_pixels': evidence['input_rgb_tchw_sha256'] == records['whole']['pixels']['raw_sha256'],
                        'decoded_pixels': decoded_pixel_witness(root / relative)}
            difference = (frames['whole'].to(torch.int16) - frames['streamed'].to(torch.int16)).abs()
            row = {'gpu_uuid': _gpu_uuid(devices[ordinal]), 'gpu_name': torch.cuda.get_device_name(),
                   'order': ['streamed' if x else 'whole' for x in order], 'model_load_seconds': load_seconds,
                   'records': records, 'max_pixel_difference': int(difference.max()),
                   'changed_pixel_components': int(torch.count_nonzero(difference)),
                   'pixels_bitwise_equal': bool(torch.equal(frames['whole'], frames['streamed'])),
                   'encoded_bytes_equal': records['whole']['video_sha256'] == records['streamed']['video_sha256']}
            if render_controls:
                row['render_controls'] = controls
                row['lossless_render_gate_passed'] = row['pixels_bitwise_equal'] and all(
                    c['encoder_input_matches_native_pixels'] and
                    (not name.startswith('lossless') or c['decoded_pixels_match_native_pixels'])
                    for name, c in controls.items())
            if async_render:
                row['async_decode_encode'] = async_records
                row['async_input_gate_passed'] = all(c['encoder_input_matches_native_pixels'] for c in async_records.values())
                row['decode_encode_speedup_only'] = async_records['sync']['decode_encode_ready_wall_seconds'] / async_records['async']['decode_encode_ready_wall_seconds']
            exclusive()
            result['rows'].append(row)
            _write_new_json(root / 'provenance' / ('streaming-vae-gpu-' + str(ordinal) + '.json'), row)
            print(json.dumps(row), flush=True)
            del difference, frames, model
            gc.collect()
            torch.cuda.empty_cache()
    result['parity_passed'] = all(x['pixels_bitwise_equal'] and x['encoded_bytes_equal'] for x in result['rows'])
    if render_controls:
        result['lossless_render_gate_passed'] = all(x['lossless_render_gate_passed'] for x in result['rows'])
        result['gate_scope'] = ('native VAE pixel equivalence, ordered streaming encoder inputs, '
            'lossless RGB round trip and presentation sequence; no claim of frozen lossy ABR repeatability')
    if async_render:
        result['async_input_gate_passed'] = all(x['async_input_gate_passed'] for x in result['rows'])
        result['async_gate_scope'] = 'native decoder input/output ordering and joined CPU encoding; no generation overlap or model speedup claim'
    _write_new_json(root / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-artifact-spec', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--render-controls', action='store_true')
    parser.add_argument('--async-render', action='store_true')
    args = parser.parse_args(argv)
    run(args.draft_artifact_spec, args.source_config, render_controls=args.render_controls,
        async_render=args.async_render)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
