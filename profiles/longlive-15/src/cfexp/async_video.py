"""Bounded CPU encoding behind an ordered stream of committed video frames.

This overlaps encoding only. GPU decode/generation scheduling is deliberately
owned by the caller. A successful finish joins the encoder before publication
or run sealing; cancelling cannot leave a thread writing to the run directory.
"""
from __future__ import annotations

from pathlib import Path
import queue
import threading

from .streaming_video import StreamingVideoWriter


class AsyncVideoWriter:
    def __init__(self, run_dir, relative_path, *, expected_frames, width=832,
                 height=480, fps=24, max_batch_frames=12, queue_batches=2,
                 writer_factory=StreamingVideoWriter, **writer_options):
        from .stage_timing import _unsealed_directory
        if any(type(v) is not int or v <= 0 for v in
               (expected_frames, width, height, fps, max_batch_frames, queue_batches)):
            raise ValueError('invalid asynchronous encoder bounds')
        self.root = _unsealed_directory(Path(run_dir))
        self.expected, self.height, self.width = expected_frames, height, width
        self.max_batch_frames, self.frames = max_batch_frames, 0
        self.queue = queue.Queue(maxsize=queue_batches)
        self.cancelled, self.ready, self.done = (threading.Event() for _ in range(3))
        self.error, self.result, self.closed = None, None, False

        def encode():
            writer = None
            try:
                writer = writer_factory(self.root, relative_path,
                    expected_frames=expected_frames, width=width, height=height,
                    fps=fps, **writer_options)
                self.ready.set()
                while not self.cancelled.is_set():
                    try:
                        value = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if value is None:
                        self.result = writer.finish()
                        return
                    writer.push(value)
            except BaseException as error:
                self.error = error
            finally:
                if writer is not None:
                    try:
                        writer.abort()
                    except BaseException as error:
                        if self.error is None:
                            self.error = error
                self.ready.set()
                self.done.set()

        self.thread = threading.Thread(target=encode, name='cfexp-video-encoder', daemon=False)
        self.thread.start()
        self.ready.wait()
        try:
            self._check()
        except BaseException:
            self.abort()
            raise

    def _check(self):
        if self.error is not None:
            raise RuntimeError('asynchronous video encoder failed') from self.error
        if self.closed or self.cancelled.is_set() or self.done.is_set():
            raise ValueError('asynchronous video encoder is closed')

    def _put(self, value):
        while True:
            self._check()
            try:
                self.queue.put(value, timeout=0.05)
                return
            except queue.Full:
                continue

    def push(self, frames):
        import torch
        from .stage_timing import _unsealed_directory
        _unsealed_directory(self.root)
        self._check()
        if (not isinstance(frames, torch.Tensor) or frames.device.type != 'cpu'
                or frames.dtype != torch.uint8 or frames.ndim != 4
                or tuple(frames.shape[1:]) != (3, self.height, self.width)
                or not 1 <= frames.shape[0] <= self.max_batch_frames
                or self.frames + frames.shape[0] > self.expected):
            raise ValueError('invalid committed CPU frame batch')
        # The producer may release/reuse its buffer immediately after push.
        # Retain at most the bounded queue plus one producer and one consumer batch.
        owned = frames.detach().clone(memory_format=torch.contiguous_format)
        self._put(owned)
        self.frames += owned.shape[0]

    def finish(self):
        self._check()
        if self.frames != self.expected:
            raise ValueError('cannot finish an incomplete asynchronous video')
        self._put(None)
        self.thread.join()
        self.closed = True
        if self.error is not None:
            raise RuntimeError('asynchronous video encoder failed') from self.error
        if self.result is None:
            raise RuntimeError('encoder terminated without a complete video')
        return self.result

    def abort(self):
        self.cancelled.set()
        self.thread.join()
        self.closed = True
        # Release queued owned tensors after the consumer has stopped.
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
