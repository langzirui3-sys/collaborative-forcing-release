"""Ordered causal rendering of committed latents, isolated from model KV state."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import queue
import tarfile
import threading
import time
import os
import tempfile

from .async_video import AsyncVideoWriter
from .correction_state import LatentHistory
from .stage_timing import _unsealed_directory
from .streaming_vae_probe import (ARCHIVE_SHA, VAE_SHA, WEIGHTS_SHA, normalization, to_pixels)

ENV = 'CFEXP_ACCEPTED_VIDEO_STREAM'
PRIORITY_ENV = 'CFEXP_RENDER_TARGET_PRIORITY'
STREAM_CHUNK_COUNTS = (30, 60, 120)
TIMING_SCOPE = 'warm_cached_conditioning_generation_accepted_stream_and_final_flush_v1'
SF_SOURCE = 'Self-Forcing-33593df3e81fa3ec10239271dd2c100facac6de1'
SF_CODELOAD_SHA = '9c8ea88fccca61b6421035add7605cb3a96a939b384dad7d9a9e5ff32beeae26'


def stage_native_source(root, source):
    """Atomically share only the hash-bound source, never decoder state."""
    if hashlib.sha256(source).hexdigest() != VAE_SHA:
        raise ValueError('stream decoder source digest differs')
    directory = root/'tmp'
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('stream decoder source directory is a symlink')
    path = directory/'accepted-stream-native-vae.py'
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix='.accepted-stream-native-vae-',dir=directory)
        try:
            with os.fdopen(fd,'wb') as output:
                output.write(source);output.flush();os.fsync(output.fileno())
            os.chmod(temporary,0o400)
            try:
                os.link(temporary,path)
            except FileExistsError:
                pass  # Another worker published its complete immutable copy.
        finally:
            os.unlink(temporary)
    if (path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid()
            or path.stat().st_mode&0o022 or path.read_bytes() != source):
        raise ValueError('existing stream decoder source differs')
    return path


class NativeStreamDecoder:
    """Separate coupled native VAE state on the existing renderer worker's GPU."""
    def __init__(self, run_dir, *, native_model=None, repository=None):
        from .main_table_baseline_campaign import _model_root, _sha256_file
        root, model_root = _unsealed_directory(Path(run_dir)), _model_root()
        native_identity = None
        if native_model is not None:
            from .native_vae_component import components
            if repository is None:
                raise ValueError('coupled native decoder requires its explicit repository binding')
            source, constants, weights, native_identity = components(native_model,
                repository=Path(repository), model_root=model_root)
        if native_model is None:
            archive = model_root/'source'/f'{SF_SOURCE}.tar.gz'
            archive_sha = ARCHIVE_SHA
            if not archive.exists() and not archive.is_symlink():
                archive = model_root/'source'/f'{SF_SOURCE}.github-codeload.tar.gz'
                archive_sha = SF_CODELOAD_SHA
            weights = model_root/'hf/Wan-AI/Wan2.1-T2V-1.3B/37ec512624d61f7aa208f7ea8140a131f93afc9a/Wan2.1_VAE.pth'
        for path, expected in (((archive, archive_sha), (weights, WEIGHTS_SHA))
                               if native_model is None else ((weights, WEIGHTS_SHA),)):
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(model_root) or _sha256_file(path)[0] != expected:
                raise ValueError('stream decoder artifact differs from frozen native source')
        if native_model is None:
            with tarfile.open(archive) as tar:
                def member(suffix):
                    names = [n for n in tar.getnames() if n.endswith(suffix)]
                    if len(names) != 1:
                        raise ValueError('ambiguous native stream decoder source')
                    return tar.extractfile(names[0]).read()
                source = member('/wan/modules/vae.py')
                constants = normalization(member('/utils/wan_wrapper.py'))
        path = stage_native_source(root,source)
        spec = importlib.util.spec_from_file_location('cfexp_accepted_native_vae', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch
        self.torch = torch
        # Loading an additional VAE initializes tensors before loading weights.
        # Preserve both generators so this cannot perturb the generation tape.
        with torch.random.fork_rng(devices=[0]), torch.inference_mode(), torch.cuda.device(0):
            self.model = module._video_vae(pretrained_path=str(weights), z_dim=16).eval().requires_grad_(False)
            self.model.to(dtype=torch.bfloat16).to(device='cuda:0')
            self.scale = [torch.tensor(constants['mean'], dtype=torch.float32).to('cuda:0', torch.bfloat16),
                          1.0 / torch.tensor(constants['std'], dtype=torch.float32).to('cuda:0', torch.bfloat16)]
            # Warm only this decoder. No generation or intervention is performed.
            self.decode(torch.zeros((1, 3, 16, 60, 104), dtype=torch.bfloat16))
            self.clear()
        self.evidence = {'native_vae_sha256': VAE_SHA, 'checkpoint_sha256': WEIGHTS_SHA,
                         'separate_causal_decoder_state': True, 'gpu': torch.cuda.get_device_name(0)}
        if native_identity is not None:
            self.evidence['coupled_native_identity'] = native_identity

    def clear(self):
        self.model.clear_cache()

    def decode(self, latent):
        torch = self.torch
        with torch.inference_mode(), torch.cuda.device(0):
            if tuple(latent.shape) != (1, 3, 16, 60, 104) or latent.dtype != torch.bfloat16:
                raise ValueError('committed stream latent shape/dtype differs')
            latent = latent.to('cuda:0', dtype=torch.bfloat16).contiguous()
            return to_pixels(torch, self.model.cached_decode(latent.permute(0, 2, 1, 3, 4), self.scale))


class AcceptedStream:
    """Only append an extension of the exact accepted prefix; never rewind."""
    def __init__(self, root, *, history, output_path, chunk_count, decoder, gpu_lock,
                 writer_factory=AsyncVideoWriter, load_latent=None, priority=False):
        if type(chunk_count) is not int or chunk_count not in STREAM_CHUNK_COUNTS or type(priority) is not bool:
            raise ValueError('accepted stream requires a supported duration profile')
        self.root = _unsealed_directory(Path(root))
        if history.next_chunk_index != 0:
            raise ValueError('accepted stream must start from the empty bound prefix')
        self.history, self.count = history, chunk_count
        self.decoder, self.gpu_lock = decoder, gpu_lock
        self.writer = writer_factory(self.root, output_path, expected_frames=chunk_count*12-3)
        if load_latent is None:
            from safetensors.torch import load_file
            def load_latent(path):
                values = load_file(str(path), device='cpu')
                if set(values) != {'latent'}:
                    raise ValueError('committed stream tensor keys differ')
                return values['latent']
        self.load_latent = load_latent
        # A paused four-Target burst must never deadlock behind a full queue.
        # Queue only small immutable segment descriptors, never GPU KV or pixels.
        # The fixed horizon bounds the number of descriptors even if rendering lags.
        self.priority = priority
        self.queue = queue.Queue(maxsize=chunk_count if priority else 2)
        self.render_allowed = threading.Event()
        self.render_allowed.set()
        self.cancelled, self.ready, self.done = (threading.Event() for _ in range(3))
        self.error, self.result, self.closed = None, None, False
        self.records = []
        self.thread = threading.Thread(target=self._render, name='cfexp-accepted-render', daemon=False)
        self.thread.start()
        self.ready.wait()
        try:
            self._check()
        except BaseException:
            self.abort()
            raise

    def _render(self):
        try:
            with self.gpu_lock:
                self.decoder.clear()
            self.ready.set()
            while not self.cancelled.is_set():
                try:
                    segment = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if segment is None:
                    self.result = self.writer.finish()
                    return
                started = time.perf_counter()
                segment.verify_file(self.root)
                latent = self.load_latent(self.root/segment.latent_path)
                segment.verify_file(self.root)
                waiting = time.perf_counter()
                while not self.cancelled.is_set():
                    if not self.render_allowed.wait(0.05):
                        continue
                    with self.gpu_lock:
                        if not self.render_allowed.is_set():
                            continue
                        acquired = time.perf_counter()
                        pixels = self.decoder.decode(latent)
                        decoded = time.perf_counter()
                    break
                else:
                    return
                expected = 9 if segment.chunk_index == 0 else 12
                if tuple(pixels.shape) != (expected, 3, 480, 832):
                    raise ValueError('native causal decoder frame position drifted')
                self.writer.push(pixels)
                self.records.append({'chunk_index': segment.chunk_index,
                    'segment_sha256': segment.segment_sha256,
                    'latent_sha256': segment.latent_sha256,
                    'frame_start': 0 if segment.chunk_index == 0 else segment.chunk_index*12-3,
                    'frame_count': expected, 'input_wall_seconds': waiting-started,
                    'gpu_lock_wait_seconds': acquired-waiting,
                    'decode_wall_seconds': decoded-acquired,
                    'encode_enqueue_wall_seconds': time.perf_counter()-decoded})
        except BaseException as error:
            self.error = error
        finally:
            try:
                self.writer.abort()
                with self.gpu_lock:
                    self.decoder.clear()
            except BaseException as error:
                if self.error is None:
                    self.error = error
            self.ready.set()
            self.done.set()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('accepted stream rendering failed') from self.error
        if self.closed or self.cancelled.is_set() or self.done.is_set():
            raise ValueError('accepted stream is closed')

    def _put(self, value):
        while True:
            self._check()
            try:
                self.queue.put(value, timeout=0.05)
                return
            except queue.Full:
                pass

    def append(self, history):
        _unsealed_directory(self.root)
        self._check()
        n = self.history.next_chunk_index
        if (history.next_chunk_index != n+1 or n >= self.count
                or history.prefix_before(n) != self.history):
            raise ValueError('stream input is not the next accepted history extension')
        self._put(history.segments[-1])
        self.history = history

    def set_render_allowed(self, allowed):
        self._check()
        if not self.priority or type(allowed) is not bool:
            raise ValueError('role-aware renderer priority is not enabled')
        if allowed:
            self.render_allowed.set()
        else:
            self.render_allowed.clear()
            # Acknowledge pause only after the current GPU batch has finished.
            # The renderer rechecks permission after acquiring this same lock.
            with self.gpu_lock:
                pass

    def finish(self, history):
        self._check()
        if history != self.history or history.next_chunk_index != self.count:
            raise ValueError('cannot finish an incomplete or different accepted history')
        self.render_allowed.set()
        self._put(None)
        self.thread.join()
        self.closed = True
        if self.error is not None:
            raise RuntimeError('accepted stream rendering failed') from self.error
        if self.result is None or len(self.records) != self.count:
            raise RuntimeError('accepted stream did not drain every committed chunk')
        video = self.result
        frames = self.count*12-3
        return {'video': video, 'pixels': {'layout': 'TCHW', 'dtype': 'uint8',
                'shape': [frames, 3, 480, 832], 'raw_sha256': video['input_rgb_tchw_sha256'],
                'raw_nbytes': frames*3*480*832}, 'history_manifest_sha256': history.manifest_sha256,
                'history_head_sha256': history.head_sha256, 'latent_chunk_count': self.count,
                'chunks': self.records, 'decoder': self.decoder.evidence,
                'role_aware_priority':self.priority,
                'latent_descriptor_queue_bound':self.queue.maxsize,
                'decoder_and_encoder_joined': not self.thread.is_alive()}

    def abort(self):
        self.cancelled.set()
        self.render_allowed.set()
        self.thread.join()
        self.writer.abort()
        self.closed = True
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
