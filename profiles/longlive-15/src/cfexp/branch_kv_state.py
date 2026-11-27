"""Bounded, exclusively owned prefix states for incremental generation.

This module owns no model implementation. A backend supplies the native KV,
position and cursor together as one payload. Checkout transfers ownership:
mutating a checked-out payload cannot corrupt any retained branch. Uncommitted
or failed leases are discarded, never returned to the cache. Missing ancestors
require explicit replay by the backend, not an approximate prefix match.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import re
from typing import Any, Callable

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _payload_resources(payload: Any) -> set[tuple[Any, ...]]:
    """Identify mutable containers and tensor storage, including tensor views.

    No torch import or device operation is needed. Native backends may not
    retain two states that merely wrap the same mutable tensor storage.
    """
    result: set[tuple[Any, ...]] = set()
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            result.add(("object", id(value)))
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            if isinstance(value, list):
                result.add(("object", id(value)))
            for item in value:
                visit(item)
        elif callable(getattr(value, "untyped_storage", None)):
            result.add(("object", id(value)))
            storage = value.untyped_storage()
            if storage.nbytes():
                result.add(("storage", str(value.device), storage.data_ptr()))

    visit(payload)
    return result


def payloads_alias(left: Any, right: Any) -> bool:
    return bool(_payload_resources(left) & _payload_resources(right))


@dataclass(frozen=True)
class PrefixBinding:
    # Scope must bind model artifacts, prompt/conditioning, precision, geometry
    # and attention configuration. Entries bind exact serialized latent chunks.
    scope_sha256: str
    chunk_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_sha256, tuple):
            raise ValueError("chunk digests must be an immutable tuple")
        for value in (self.scope_sha256, *self.chunk_sha256):
            if not isinstance(value, str) or not _DIGEST.fullmatch(value):
                raise ValueError("prefix bindings require lowercase SHA256 digests")

    def is_prefix_of(self, other: PrefixBinding) -> bool:
        return (self.scope_sha256 == other.scope_sha256
                and len(self.chunk_sha256) <= len(other.chunk_sha256)
                and other.chunk_sha256[:len(self.chunk_sha256)] == self.chunk_sha256)


@dataclass(frozen=True)
class NativeBranchState:
    binding: PrefixBinding
    payload: Any
    next_chunk: int
    next_token: int
    resident_bytes: int

    def validate(self, tokens_per_chunk: int) -> None:
        if type(self.next_chunk) is not int or self.next_chunk != len(self.binding.chunk_sha256):
            raise ValueError("KV cursor does not match its bound history")
        if type(self.next_token) is not int or self.next_token != self.next_chunk * tokens_per_chunk:
            raise ValueError("KV absolute position does not match its bound history")
        if type(self.resident_bytes) is not int or self.resident_bytes <= 0:
            raise ValueError("native state must report its positive resident size")
        if self.payload is None:
            raise ValueError("native state payload is required")


class BranchKVCache:
    """Two retained branches by default; no implicit tensor clone or disk state.

    The backend must stop referring to a payload after committing it. The cache
    is deliberately serial: a second checkout while a lease is live fails. The
    memory budget includes the leased state, with eviction before admission.
    A cache miss does not manufacture a state or change the requested history.
    """

    def __init__(self, *, max_bytes: int, max_states: int = 2, tokens_per_chunk: int = 4680):
        for value in (max_bytes, max_states, tokens_per_chunk):
            if type(value) is not int or value <= 0:
                raise ValueError("cache limits must be positive integers")
        self.max_bytes = max_bytes
        self.max_states = max_states
        self.tokens_per_chunk = tokens_per_chunk
        self._states: OrderedDict[PrefixBinding, NativeBranchState] = OrderedDict()
        self._anchor: NativeBranchState | None = None
        self._lease: BranchLease | None = None
        self.peak_resident_bytes = 0

    @property
    def retained_bindings(self) -> tuple[PrefixBinding, ...]:
        return tuple(self._states)

    @property
    def resident_bytes(self) -> int:
        leased = self._lease._active_bytes if self._lease else 0
        anchor = self._anchor.resident_bytes if self._anchor else 0
        return sum(state.resident_bytes for state in self._states.values()) + leased + anchor

    @property
    def state_count(self) -> int:
        return len(self._states) + int(self._anchor is not None) + int(
            self._lease is not None and self._lease._active_bytes > 0)

    @property
    def pinned_binding(self) -> PrefixBinding | None:
        return self._anchor.binding if self._anchor is not None else None

    def _observe_memory(self) -> None:
        if self.resident_bytes > self.max_bytes or self.state_count > self.max_states:
            raise ValueError("native state exceeds the cache memory budget")
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    def checkout(self, requested: PrefixBinding, *, consume_anchor: bool = False) -> BranchLease:
        if not isinstance(requested, PrefixBinding):
            raise ValueError("requested prefix must be bound")
        if self._lease is not None:
            raise RuntimeError("a native KV lease is already active")
        if type(consume_anchor) is not bool:
            raise ValueError("consume_anchor must be a boolean")
        if consume_anchor:
            # An absent pin after process restart honestly rebuilds the prefix.
            # A nonmatching pin is a controller error, never a silent fallback.
            state = self._anchor
            if state is not None and not state.binding.is_prefix_of(requested):
                raise ValueError("pinned anchor belongs to a different scope or history")
            self._anchor = None
        else:
            # Ordinary continuation must never consume the immutable anchor.
            matches = [key for key in self._states if key.is_prefix_of(requested)]
            key = max(matches, key=lambda x: len(x.chunk_sha256)) if matches else None
            state = self._states.pop(key) if key is not None else None
        lease = BranchLease(self, requested, state)
        self._lease = lease
        return lease

    def clear(self) -> None:
        if self._lease is not None:
            raise RuntimeError("cannot clear while a native KV lease is active")
        self._states.clear()
        self._anchor = None


class BranchLease:
    def __init__(self, owner: BranchKVCache, requested: PrefixBinding,
                 state: NativeBranchState | None):
        self._owner = owner
        self.requested = requested
        self.state = state
        self._active_bytes = state.resident_bytes if state is not None else 0
        self._closed = False

    @property
    def replay_from_chunk(self) -> int:
        return self.state.next_chunk if self.state is not None else 0

    def __enter__(self) -> BranchLease:
        if self._closed:
            raise RuntimeError("lease is closed")
        return self

    def reserve_allocation(self, size: int) -> None:
        """Evict before a backend allocates a new state on a cache miss."""
        if self._closed or self._owner._lease is not self or self.state is not None:
            raise RuntimeError("allocation reservation requires an active cache miss")
        if type(size) is not int or not 0 < size <= self._owner.max_bytes:
            raise ValueError("allocation exceeds the native cache budget")
        states = self._owner._states
        anchor = self._owner._anchor
        anchor_bytes = anchor.resident_bytes if anchor is not None else 0
        if anchor_bytes + size > self._owner.max_bytes or int(anchor is not None) + 1 > self._owner.max_states:
            raise ValueError("allocation cannot coexist with the pinned anchor")
        while states and (len(states) + int(anchor is not None) >= self._owner.max_states
                          or sum(s.resident_bytes for s in states.values()) + anchor_bytes + size > self._owner.max_bytes):
            states.popitem(last=False)

    def note_allocated(self, size: int) -> None:
        """Account for the actual new buffer set, after pre-allocation eviction."""
        if self._closed or self._owner._lease is not self or self.state is not None:
            raise RuntimeError("new allocation requires an active cache miss")
        if type(size) is not int or size <= 0:
            raise ValueError("native allocation size must be positive")
        self._active_bytes = size
        self._owner._observe_memory()

    def preserve_input(self, state: NativeBranchState,
                       clone_payload: Callable[[Any], Any]) -> bool:
        """Clone the exact replayed input once, evicting before copying.

        The active state may already have replayed a suffix since checkout.
        Its cloned input prefix lives outside ordinary longest-prefix lookup.
        At most the active payload and this one anchor survive this operation.
        """
        if self._closed or self._owner._lease is not self:
            raise RuntimeError("preserve requires an active native lease")
        state.validate(self._owner.tokens_per_chunk)
        if state.binding != self.requested:
            raise ValueError("anchor must bind the exact requested input history")
        if state.resident_bytes != self._active_bytes:
            raise ValueError("anchor size differs from its active allocation")
        if self._owner.max_states < 2 or 2 * state.resident_bytes > self._owner.max_bytes:
            raise ValueError("two native states exceed the anchor memory budget")
        replaced = self._owner._anchor is not None
        self._owner._anchor = None
        self._owner._states.clear()
        copied = clone_payload(state.payload)
        if payloads_alias(state.payload, copied):
            raise ValueError("anchor clone aliases active native storage")
        self._owner._anchor = NativeBranchState(
            state.binding, copied, state.next_chunk, state.next_token, state.resident_bytes)
        self._owner._observe_memory()
        return replaced

    def commit(self, result: NativeBranchState) -> None:
        self._commit(result, append_generated_chunk=True)

    def commit_prefill(self, result: NativeBranchState) -> None:
        """Retain the exact requested accepted prefix without generating a chunk.

        Only an existing exclusively leased allocation may be advanced. This
        path cannot allocate a replacement payload or coexist with an anchor.
        Ordinary ``commit`` still requires exactly one generated append.
        """
        if (self.state is None or result.payload is not self.state.payload
                or result.resident_bytes != self.state.resident_bytes
                or self._owner._anchor is not None):
            raise ValueError('prefill requires the existing unpinned native payload')
        if result.binding != self.requested:
            raise ValueError('prefill must bind the exact requested accepted prefix')
        self._commit(result, append_generated_chunk=False)

    def _commit(self, result: NativeBranchState, *, append_generated_chunk: bool) -> None:
        if self._closed or self._owner._lease is not self:
            raise RuntimeError("lease is not active")
        result.validate(self._owner.tokens_per_chunk)
        if not self.requested.is_prefix_of(result.binding):
            raise ValueError("result KV belongs to a different branch")
        # One request may replay history and append exactly one generated chunk.
        if result.next_chunk != len(self.requested.chunk_sha256) + int(append_generated_chunk):
            raise ValueError("result must append exactly one chunk" if append_generated_chunk
                             else 'prefill must not append a generated chunk')
        if result.resident_bytes > self._owner.max_bytes:
            raise ValueError("native state exceeds the cache memory budget")
        states = self._owner._states
        # Reject aliasing before changing the retained collection.
        retained = [*states.values()]
        if self._owner._anchor is not None:
            retained.append(self._owner._anchor)
        if any(payloads_alias(s.payload, result.payload) for s in retained):
            raise ValueError("native payload aliases another retained branch")
        states.pop(result.binding, None)
        anchor = self._owner._anchor
        anchor_bytes = anchor.resident_bytes if anchor is not None else 0
        if anchor_bytes + result.resident_bytes > self._owner.max_bytes:
            raise ValueError("result cannot coexist with the pinned anchor")
        while states and (len(states) + int(anchor is not None) >= self._owner.max_states
                          or sum(s.resident_bytes for s in states.values())
                          + anchor_bytes + result.resident_bytes > self._owner.max_bytes):
            states.popitem(last=False)
        states[result.binding] = result
        self._finish()
        self._owner._observe_memory()

    def _finish(self) -> None:
        self.state = None
        self._active_bytes = 0
        self._closed = True
        self._owner._lease = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._closed:
            # The checked-out KV may have been partially overwritten by a
            # failed model forward. It is unsafe to restore that state.
            self._finish()
