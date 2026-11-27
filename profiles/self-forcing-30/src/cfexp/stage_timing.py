"""Opt-in request-local timing; no torch import or synchronization when disabled.

CPU and CUDA spans are inclusive. CUDA event elapsed time is stream elapsed
time (including gaps), not a claim of kernel busy time. Request completion
waits only for the last recorded event, never once per stage. Sidecars are
diagnostics, separate from semantic generation/provenance payloads.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterator, Mapping
import uuid

SCHEMA = "stage_timing_v1"
ENV = "CFEXP_STAGE_TIMING"
_ACTIVE: ContextVar[Any] = ContextVar("cfexp_stage_timing", default=None)
_FILES: dict[tuple[str, str, int], Path] = {}
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")


def enabled() -> bool:
    value = os.environ.get(ENV)
    if value not in (None, "", "0", "1"):
        raise ValueError(f"{ENV} must be unset, 0, or 1")
    return value == "1"


def current_timing() -> Any:
    return _ACTIVE.get()


def _finite(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("stage timing clock produced an invalid duration")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _unsealed_directory(path: Path) -> Path:
    raw = path.absolute()
    if any(parent.is_symlink() for parent in (raw, *raw.parents)):
        raise ValueError("stage timing directory may not traverse a symlink")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("stage timing requires an existing run directory")
    # Workers receive an attempt directory. Check its run ancestors too.
    for parent in (root, *root.parents):
        if any((parent / name).exists() for name in
               ("_SUCCESS.json", "_FAILED.json", "artifact-index.json")):
            raise ValueError("cannot write stage timing below a sealed run")
    return root


def write_record(run_dir: Path, record: Mapping[str, Any]) -> None:
    root = _unsealed_directory(run_dir)
    source = record["source"]
    if not isinstance(source, str) or _NAME.fullmatch(source) is None:
        raise ValueError("invalid stage timing source")
    directory = root
    for part in ("provenance", "stage-timing"):
        directory = directory / part
        if directory.is_symlink():
            raise ValueError("stage timing output directory may not be a symlink")
        directory.mkdir(exist_ok=True)
    key = (str(root), source, os.getpid())
    path = _FILES.get(key)
    if path is None:
        path = directory / f"{source}-{os.getpid()}-{uuid.uuid4().hex}.jsonl"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    else:
        flags = os.O_WRONLY | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    raw = (json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "ab") as output:
            output.write(raw)
            output.flush()
    except BaseException:
        raise
    else:
        _FILES[key] = path


class TimingSession:
    def __init__(self, *, source: str, metadata: Mapping[str, Any],
                 clock: Any = time.perf_counter) -> None:
        self.source = source
        self.metadata = _json_copy(dict(metadata))
        self.record_id = uuid.uuid4().hex
        self.clock = clock
        self.started = clock()
        self.spans: list[dict[str, Any]] = []
        self.not_applicable: list[dict[str, str]] = []
        self._stack: list[int] = []
        self._cuda: Any = None
        self._stream: Any = None
        self._events: list[tuple[int, Any, Any]] = []
        self._open_events: dict[int, Any] = {}
        self._finished = False

    def attach_cuda(self, torch: Any, device: Any) -> None:
        if self._cuda is not None:
            return
        self._cuda = torch.cuda
        self._stream = torch.cuda.current_stream(device=device)

    def start_span(self, name: str, *, detail: Any = None, cuda: bool = False) -> int:
        if self._finished:
            raise ValueError("timing session is finished")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ValueError("invalid stage timing span name")
        span_id = len(self.spans)
        if span_id >= 100000:
            raise ValueError("too many stage timing spans in one request")
        start = _finite(self.clock() - self.started)
        self.spans.append({"span_id": span_id,
                           "parent_span_id": self._stack[-1] if self._stack else None,
                           "name": name, "detail": _json_copy(detail),
                           "start_offset_seconds": start, "end_offset_seconds": None,
                           "wall_seconds": None, "cuda_seconds": None,
                           "status": "completed"})
        self._stack.append(span_id)
        if cuda and self._cuda is not None:
            event = self._cuda.Event(enable_timing=True)
            event.record(self._stream)
            self._open_events[span_id] = event
        return span_id

    def end_span(self, handle: int, *, status: str = "completed") -> None:
        if not self._stack or self._stack[-1] != handle:
            raise ValueError("timing spans must close in nesting order")
        if status not in ("completed", "error", "interrupted"):
            raise ValueError("invalid stage timing span status")
        start_event = self._open_events.pop(handle, None)
        if start_event is not None:
            end_event = self._cuda.Event(enable_timing=True)
            end_event.record(self._stream)
            self._events.append((handle, start_event, end_event))
        self._stack.pop()
        span = self.spans[handle]
        if status == "error":
            # Forward hooks close spans while the probe exception is active.
            # Only this exact control-flow exception is an intentional stop.
            from .early_draft import DraftProbeStop
            if (type(sys.exc_info()[1]) is DraftProbeStop and span["name"] == "denoise"
                    and span["detail"] in ("forward_0", "forward_1")
                    and self.source == "worker-draft"
                    and self.metadata.get("action") == "generate_early"):
                status = "interrupted"
                span["interruption"] = "DraftProbeStop"
        end = _finite(self.clock() - self.started)
        span.update(end_offset_seconds=end,
                    wall_seconds=_finite(end - span["start_offset_seconds"]), status=status)

    def note_not_applicable(self, name: str, reason: str) -> None:
        if _NAME.fullmatch(name) is None or not reason:
            raise ValueError("invalid stage timing not-applicable annotation")
        item = {"name": name, "reason": reason}
        if item not in self.not_applicable:
            self.not_applicable.append(item)

    def finish(self, *, status: str = "completed") -> dict[str, Any]:
        if self._finished:
            raise ValueError("timing session is already finished")
        if status not in ("completed", "error", "interrupted"):
            raise ValueError("invalid timing request status")
        if self._stack and status == "completed":
            raise ValueError("unclosed timing spans")
        while self._stack:
            self.end_span(self._stack[-1], status="error")
        sync_wall = 0.0
        if self._events:
            start = self.clock()
            self._events[-1][2].synchronize()
            sync_wall = _finite(self.clock() - start)
            for handle, start_event, end_event in self._events:
                self.spans[handle]["cuda_seconds"] = _finite(
                    float(start_event.elapsed_time(end_event)) / 1000.0)
        self._finished = True
        return {"schema": SCHEMA, "record_id": self.record_id,
                "source": self.source, "pid": os.getpid(), "metadata": self.metadata,
                "status": status, "wall_seconds": _finite(self.clock() - self.started),
                "cuda_sync_wall_seconds": sync_wall, "spans": self.spans,
                "not_applicable": self.not_applicable}


@contextmanager
def timing_session(run_dir: str | Path, source: str,
                   metadata: Mapping[str, Any]) -> Iterator[TimingSession | None]:
    if not enabled():
        yield None
        return
    session = TimingSession(source=source, metadata=metadata)
    _unsealed_directory(Path(run_dir))
    token = _ACTIVE.set(session)
    status = "completed"
    failure: BaseException | None = None
    try:
        yield session
    except BaseException as error:
        status = "error"
        failure = error
        raise
    finally:
        _ACTIVE.reset(token)
        try:
            write_record(Path(run_dir), session.finish(status=status))
        except BaseException as diagnostic_error:
            if failure is None:
                raise
            failure.add_note(
                "Stage timing record incomplete: " + type(diagnostic_error).__name__
            )


@contextmanager
def stage(name: str, *, detail: Any = None, cuda: bool = False) -> Iterator[None]:
    session = current_timing()
    if session is None:
        yield
        return
    handle = session.start_span(name, detail=detail, cuda=cuda)
    status = "completed"
    failure: BaseException | None = None
    try:
        yield
    except BaseException as error:
        status = "error"
        failure = error
        raise
    finally:
        try:
            session.end_span(handle, status=status)
        except BaseException as diagnostic_error:
            if failure is None:
                raise
            failure.add_note(
                "Stage timing span incomplete: " + type(diagnostic_error).__name__
            )
