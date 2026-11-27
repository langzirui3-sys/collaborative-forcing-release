"""Reap helpers owned by a process at its terminal shutdown boundary.

This is for an experiment/worker ``main`` finally block, never for closing one
model in a process that will continue doing work.  It neither imports torch nor
signals a process group.  Existing library shutdown hooks run before the owner
exits, and their child PIDs must actually have been waited/reaped.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from typing import Any, Callable, TextIO


class RuntimeHelperShutdownError(RuntimeError):
    """An owned library helper did not shut down and reap within the deadline."""


def shutdown_owned_runtime_helpers(
    *,
    timeout_seconds: float = 10.0,
    label: str = "controller",
    log_stream: TextIO | None = None,
) -> list[dict[str, Any]]:
    """Finalize already-loaded helper libraries with bounded, checked reaping.

    ``waitid(..., WNOWAIT)`` proves ownership without consuming child status on
    Linux.  The portable waitpid fallback preserves consumed status in the
    library's process object.  Inherited tracker descriptors (whose local ``_pid`` is
    None) are left alone.  A timeout/error fails closed; the enclosing formal
    runner remains responsible for detecting and terminating any residuals.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("helper shutdown timeout must be finite and positive")
    stream = sys.stderr if log_stream is None else log_stream
    records: list[dict[str, Any]] = []
    owner_pid = os.getpid()
    deadline = time.monotonic() + float(timeout_seconds)

    def emit(kind: str, pid: int | None, phase: str, state: str) -> None:
        record = {
            "schema": "runtime_helper_shutdown_v1",
            "label": label,
            "owner_pid": owner_pid,
            "helper_type": kind,
            "helper_pid": pid,
            "phase": phase,
            "state": state,
        }
        records.append(record)
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()

    def probe_child(kind: str, pid: int, process: Any | None = None) -> bool:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeHelperShutdownError("helper has an invalid PID")
        try:
            waitid = getattr(os, "waitid", None)
            if callable(waitid) and hasattr(os, "WNOWAIT"):
                status = waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                emit(kind, pid, "before", "owned_running" if status is None else "owned_exited")
                return True
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                emit(kind, pid, "before", "already_absent")
                return False
            # A live PID that is not our child is never ours to stop.  In
            # particular this prevents acting on inherited/stale helper data.
            emit(kind, pid, "failure", "not_owned_child")
            raise RuntimeHelperShutdownError("helper PID is not an owned child")
        if waited_pid == pid:
            # Python on macOS lacks waitid.  If its fallback wins the race
            # with library poll/wait, give the status back to that exact
            # process object so its shutdown hook still sees the true code.
            if process is not None:
                popen = getattr(process, "_popen", process)
                if popen is None or not hasattr(popen, "returncode"):
                    raise RuntimeHelperShutdownError("cannot preserve reaped helper exit status")
                popen.returncode = os.waitstatus_to_exitcode(status)
            emit(kind, pid, "before", "exited_and_reaped")
            return False
        emit(kind, pid, "before", "owned_running")
        return True

    def bounded_call(name: str, callback: Callable[[], Any]) -> None:
        outcome: list[BaseException] = []

        def invoke() -> None:
            try:
                callback()
            except BaseException as error:
                outcome.append(error)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeHelperShutdownError("runtime helper shutdown deadline exceeded")
        thread = threading.Thread(target=invoke, name=name, daemon=True)
        thread.start()
        thread.join(remaining)
        if thread.is_alive():
            emit(name, None, "failure", "timeout")
            raise RuntimeHelperShutdownError(f"{name} exceeded shutdown deadline")
        if outcome:
            emit(name, None, "failure", type(outcome[0]).__name__)
            raise RuntimeHelperShutdownError(f"{name} shutdown failed") from outcome[0]

    def verify_reaped(kind: str, pid: int) -> None:
        try:
            waited_pid, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            emit(kind, pid, "after", "reaped")
            return
        if waited_pid == pid:
            emit(kind, pid, "after", "exited_and_reaped")
            return
        emit(kind, pid, "failure", "still_running")
        raise RuntimeHelperShutdownError(f"{kind} returned without reaping its child")

    # Torch 2.5 warms these pools on import when Triton is present, even for
    # callers that never request torch.compile.  Only inspect a loaded module.
    compiler = sys.modules.get("torch._inductor.async_compile")
    if compiler is not None:
        pools = tuple(getattr(compiler, "_pool_set", ()))
        children: list[tuple[str, int]] = []
        for pool in pools:
            process = getattr(pool, "process", None)
            if process is not None:
                kind = "torch_inductor_subprocess"
                pid = getattr(process, "pid", None)
                if probe_child(kind, pid, process):
                    children.append((kind, pid))
            elif hasattr(pool, "_processes"):
                for process in (getattr(pool, "_processes", None) or {}).values():
                    kind = "torch_inductor_pool_worker"
                    pid = getattr(process, "pid", None)
                    if probe_child(kind, pid, process):
                        children.append((kind, pid))
            else:
                raise RuntimeHelperShutdownError("unrecognized loaded compiler pool")
        if pools:
            shutdown = getattr(compiler, "shutdown_compile_workers", None)
            if not callable(shutdown):
                raise RuntimeHelperShutdownError("loaded compiler lacks its shutdown hook")
            bounded_call("torch_inductor_shutdown", shutdown)
            for kind, pid in children:
                verify_reaped(kind, pid)

    # Run multiprocessing's normal terminal finalizers while its tracker is
    # still alive, so SemLock/shared-memory finalizers cannot restart a tracker
    # after we stop it.  This also joins registered multiprocessing children.
    multiprocessing = sys.modules.get("multiprocessing")
    util = sys.modules.get("multiprocessing.util")
    mp_children: list[int] = []
    if multiprocessing is not None:
        active_children = getattr(multiprocessing, "active_children", None)
        if callable(active_children):
            for child in active_children():
                pid = getattr(child, "pid", None)
                if probe_child("multiprocessing_child", pid, child):
                    mp_children.append(pid)
    if util is not None:
        finalize = getattr(util, "_exit_function", None)
        if not callable(finalize):
            raise RuntimeHelperShutdownError("loaded multiprocessing lacks its exit hook")
        bounded_call("multiprocessing_finalize", finalize)
    for pid in mp_children:
        verify_reaped("multiprocessing_child", pid)

    module = sys.modules.get("multiprocessing.resource_tracker")
    tracker = getattr(module, "_resource_tracker", None) if module is not None else None
    pid = getattr(tracker, "_pid", None)
    if pid is not None and probe_child("multiprocessing_resource_tracker", pid):
        stop = getattr(tracker, "_stop", None)
        if not callable(stop):
            raise RuntimeHelperShutdownError("owned resource tracker lacks its stop hook")
        bounded_call("multiprocessing_resource_tracker_stop", stop)
        verify_reaped("multiprocessing_resource_tracker", pid)
    return records
