"""Optional bounded diagnostics for high-volume Nuzlocke observations."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .events import Event


class BoundedEventTrace:
    """Keep only the most recent diagnostic events in memory.

    This is opt-in and deliberately separate from the durable event store.
    A caller may provide its own writer if a rotating on-disk trace is wanted.
    """

    def __init__(self, max_events: int = 1024) -> None:
        """Create a bounded in-memory trace with the requested capacity."""

        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._events: deque[Event] = deque(maxlen=max_events)

    def __call__(self, event: Event, _session_id: str) -> None:
        """Record one event, discarding the oldest event when full."""

        self._events.append(event)

    def events(self) -> tuple[Event, ...]:
        """Return a stable tuple of the currently retained events."""

        return tuple(self._events)

    def __iter__(self) -> Iterable[Event]:
        """Iterate over retained events from oldest to newest."""

        return iter(self._events)
