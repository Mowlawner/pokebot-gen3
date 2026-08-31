"""Persistence policy for events produced from live Nuzlocke observations.

The observer intentionally reports every useful transition.  This module is
the boundary that decides whether a transition belongs in durable campaign
history, an optional bounded diagnostic trace, or live runtime state only.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Callable

from .events import (
    BattleEnded,
    BattleStarted,
    Event,
    GameStateChanged,
    MapChanged,
    NuzlockeStarted,
    PartyChanged,
    PokemonCaptured,
    PokemonFainted,
    StorageChanged,
    WhiteoutOccurred,
)


class PersistenceClass(str, Enum):
    """Destination class for an observed event in runtime persistence."""

    DURABLE = "durable"
    EPHEMERAL = "ephemeral"
    DIAGNOSTIC = "diagnostic"


# These events are consumed by the durable campaign/rules projections.  Map
# and game-state transitions remain available to the runtime projection, but
# replaying their entire observation history is not needed for Nuzlocke state.
_DURABLE_TYPES = (
    BattleEnded,
    BattleStarted,
    NuzlockeStarted,
    PokemonCaptured,
    PokemonFainted,
    StorageChanged,
    WhiteoutOccurred,
)


def classify_event(event: Event) -> PersistenceClass:
    """Classify an observed event by its campaign persistence semantics."""
    if isinstance(event, PartyChanged):
        # Composition, identity, and ordering can affect campaign state.
        # Level/status/move/held-item refreshes are still delivered to live
        # consumers but are not represented by the durable campaign model.
        if event.entered_party_indices or event.left_party_indices or event.reordered:
            return PersistenceClass.DURABLE
        return PersistenceClass.EPHEMERAL
    if isinstance(event, _DURABLE_TYPES):
        return PersistenceClass.DURABLE
    if isinstance(event, (MapChanged, GameStateChanged)):
        return PersistenceClass.EPHEMERAL
    # Unknown event types are retained in memory until a consumer explicitly
    # assigns them a policy; they must not silently become durable history.
    return PersistenceClass.EPHEMERAL


class EventStatistics:
    """Low-overhead counts for development diagnostics, not per-frame output."""

    def __init__(self) -> None:
        """Create empty counters for observed event classifications."""

        self._counts: Counter[tuple[PersistenceClass, str]] = Counter()

    def record(self, event: Event, persistence_class: PersistenceClass | None = None) -> None:
        """Increment the counter for an event and its persistence class."""

        classification = persistence_class or classify_event(event)
        self._counts[(classification, type(event).__name__)] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return event counts grouped by persistence class and type."""

        result: dict[str, dict[str, int]] = {classification.value: {} for classification in PersistenceClass}
        for (classification, event_type), count in self._counts.items():
            result[classification.value][event_type] = count
        return result


EventSink = Callable[[Event, str], object]
