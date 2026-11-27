"""Bounded frame-batch encoder with a frozen, explicit rendering policy.

No generation or routing decisions occur here. The caller may append only
frames from committed latents, in order, and must finish before run sealing.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path

from .video_encoding_policy import SERIAL, verify_video

LOSSLESS_CONTROL = 'lossless_rgb_control_v1'


class StreamingVideoWriter:
    def __init__(self, run_dir, relative_path, *, expected_frames, width=832, height=480,
                 fps=24, policy=SERIAL):
        from .stage_timing import _unsealed_directory
        from .correction_worker import _output_path, _temporary_sibling
        if (any(type(x) is not int or x <= 0 for x in (expected_frames, width, height, fps))
                or policy not in (SERIAL, LOSSLESS_CONTROL)):
            raise ValueError('invalid streaming video geometry or policy')
        self.root = _unsealed_directory(Path(run_dir))
        self.relative, self.path = _output_path(self.root, relative_path, label='streaming video')
        self.temporary = _temporary_sibling(self.path)
        self.expected_frames, self.width, self.height, self.fps = expected_frames, width, height, fps
        self.policy, self.frames, self.state = policy, 0, 'open'
        self.pixel_digest = hashlib.sha256()
        self.container = None
        try:
            import av
            if av.__version__ != '13.1.0':
                raise ValueError('streaming encoder requires frozen PyAV 13.1.0')
            self.av = av
            self.container = av.open(str(self.temporary), 'w')
            codec = 'libx264' if policy == SERIAL else 'libx264rgb'
            self.stream = self.container.add_stream(codec, rate=fps)
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = 'yuv420p' if policy == SERIAL else 'rgb24'
            self.stream.options = ({'threads': '1'} if policy == SERIAL else
                                   {'threads': '1', 'crf': '0', 'preset': 'ultrafast'})
        except BaseException:
            self.abort()
            raise

    def push(self, frames):
        import torch
        from .stage_timing import _unsealed_directory
        _unsealed_directory(self.root)
        if self.state != 'open':
            raise ValueError('streaming encoder is not open')
        if (not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8
                or frames.device.type != 'cpu' or frames.ndim != 4
                or tuple(frames.shape[1:]) != (3, self.height, self.width)
                or frames.shape[0] < 1 or self.frames + frames.shape[0] > self.expected_frames):
            raise ValueError('streaming encoder requires bounded CPU uint8 TCHW frame batches')
        pixels = frames.detach().contiguous()
        self.pixel_digest.update(memoryview(pixels.numpy()).cast('B'))
        images = pixels.permute(0, 2, 3, 1).contiguous().numpy()
        for image in images:
            frame = self.av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pict_type = 'NONE'
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.frames += 1

    def finish(self):
        from .stage_timing import _unsealed_directory
        from .correction_worker import _sha256_file, _StagedOutput, _publish_staged
        _unsealed_directory(self.root)
        if self.state != 'open' or self.frames != self.expected_frames:
            raise ValueError('cannot finish an incomplete or closed video stream')
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            self.container = None
            if self.policy == SERIAL:
                verify_video(self.temporary, SERIAL)
            digest, size = _sha256_file(self.temporary)
            result = {'path': self.relative, 'sha256': digest, 'nbytes': size,
                      'fps': self.fps, 'frames': self.frames, 'encoding_policy': self.policy,
                      'input_rgb_tchw_sha256': self.pixel_digest.hexdigest()}
            _unsealed_directory(self.root)
            _publish_staged([_StagedOutput(relative_path=self.relative, target=self.path,
                temporary=self.temporary, metadata=result)])
            self.state = 'finished'
            return result
        except BaseException:
            self.abort()
            raise

    def abort(self):
        if self.state == 'finished':
            return
        from .stage_timing import _unsealed_directory
        _unsealed_directory(self.root)
        if self.container is not None:
            self.container.close()
            self.container = None
        self.temporary.unlink(missing_ok=True)
        self.state = 'aborted'


def decoded_pixel_witness(path, *, width=832, height=480, frames=357, fps=24):
    """Decode independently; hash canonical TCHW bytes and exact presentation times."""
    import av
    if av.__version__ != '13.1.0':
        raise ValueError('decoded pixel witness requires frozen PyAV 13.1.0')
    digest, count = hashlib.sha256(), 0
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
            raise ValueError('expected one video stream without audio')
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if stream.width != width or stream.height != height or stream.average_rate != fps:
            raise ValueError('decoded stream metadata differs from expected geometry')
        for frame in container.decode(video=0):
            if frame.pts is None or frame.pts * frame.time_base != Fraction(count, fps):
                raise ValueError('decoded frame presentation sequence differs')
            image = frame.to_ndarray(format='rgb24')
            if image.shape != (height, width, 3) or str(image.dtype) != 'uint8':
                raise ValueError('decoded frame geometry differs')
            digest.update(memoryview(image.transpose(2, 0, 1).copy()).cast('B'))
            count += 1
    if count != frames:
        raise ValueError('decoded frame count differs')
    return {'shape': [frames, 3, height, width], 'dtype': 'uint8', 'layout': 'TCHW',
            'raw_sha256': digest.hexdigest(), 'frames': count, 'fps': fps,
            'presentation_sequence_verified': True}
