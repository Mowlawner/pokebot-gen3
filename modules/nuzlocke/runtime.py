"""Application-level delivery for normalized Nuzlocke observations.

This module is the boundary between the frame loop and the domain observer.
It reads no emulator state itself: callers provide a snapshot (or a snapshot
provider), and consumers receive immutable events in frame order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from uuid import uuid4
from typing import TYPE_CHECKING

from .events import BattleStarted, Event, NuzlockeEventObserver, NuzlockeStarted
from .policy import EventSink, EventStatistics, PersistenceClass, classify_event
from .snapshots import NuzlockeSnapshot, get_nuzlocke_snapshot
from .campaign_state import Fact, derive_campaign_facts
from .projection import CampaignProjection
from .rules import NuzlockeRulesProjection
from modules.console import diagnostic_print

if TYPE_CHECKING:
    from typing import Deque


class NuzlockeRuntime:
    """Feed one snapshot per application frame and expose each event once."""

    def __init__(
        self,
        snapshot_provider: Callable[[], NuzlockeSnapshot] = get_nuzlocke_snapshot,
        event_sink: EventSink | None = None,
        diagnostic_sink: EventSink | None = None,
        session_id: str | None = None,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._event_sink = event_sink
        self._diagnostic_sink = diagnostic_sink
        self._session_id = session_id or str(uuid4())
        self._event_sequence = 0
        self._observer = NuzlockeEventObserver()
        self._events: Deque[Event] = deque()
        self._subscribers: list[Callable[[Event], None]] = []
        self._last_frame: int | None = None
        self._campaign_projection = CampaignProjection()
        self._rules_projection = NuzlockeRulesProjection(encounters_active=False)
        self._event_statistics = EventStatistics()

    @property
    def observed_projection(self) -> CampaignProjection:
        return self._campaign_projection

    @property
    def rules_projection(self) -> NuzlockeRulesProjection:
        return self._rules_projection

    def capture_target_for(
        self,
        location: tuple[int, int] | None,
        *,
        is_wild: bool,
        is_trainer: bool,
    ) -> bool:
        """Return whether this already-observed battle is the legal target.

        This is intentionally a read-only view of the rules projection.  It
        never claims an encounter; the observer claims it when BattleStarted
        is delivered.  Trainers and location-less battles can never opt in.
        """
        projection = self._rules_projection
        diagnostic_print(
            lambda: (
                "NUZLOCKE_CAPTURE_TARGET_TRACE: "
                f"runtime_id={id(self)} location_arg={location!r} "
                f"is_wild={is_wild} is_trainer={is_trainer} "
                f"projection_id={id(projection)} "
                f"filter_not_wild={not is_wild} filter_trainer={is_trainer} "
                f"filter_location_none={location is None}"
            ),
            trace=True,
        )
        if not is_wild or is_trainer or location is None:
            if is_wild and not is_trainer:
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_CAPTURE_TARGET: "
                        f"location={location!r} is_wild={is_wild} is_trainer={is_trainer} "
                        "projection_encounter_found=False status=None eligible=None returned=False"
                    ),
                    trace=True,
                )
            return False
        encounters = projection.state.encounters
        registered_locations = tuple(item.location for item in encounters)
        encounter = projection.state.encounter_for(location)
        registered_location = next(
            (item.location for item in encounters if item.location == location),
            None,
        )
        locations_equal = registered_location == location
        result = encounter.eligible and encounter.status == "pending"
        diagnostic_print(
            lambda: (
                "NUZLOCKE_CAPTURE_TARGET_TRACE: "
                f"runtime_id={id(self)} location_arg={location!r} "
                f"is_wild={is_wild} is_trainer={is_trainer} "
                f"projection_id={id(projection)} "
                f"projection_encounter_lookup={encounter!r} "
                f"registered_locations={registered_locations!r} "
                f"registered_location={registered_location!r} "
                f"locations_equal={locations_equal} "
                f"status={encounter.status!r} eligible={encounter.eligible!r} "
                f"status_pending={encounter.status == 'pending'} "
                f"final_return={result}"
            ),
            trace=True,
        )
        return result

    def update(self, snapshot: NuzlockeSnapshot | None = None) -> tuple[Event, ...]:
        """Process the current frame and return events generated by it.

        A backwards frame count denotes a reset or save-state load.  The
        observer is replaced and the current snapshot becomes a new baseline,
        so the reset cannot manufacture transitions between unrelated states.
        """
        trace = getattr(__import__("modules.context", fromlist=["context"]), "context", None)
        trace = getattr(trace, "stutter_trace", None)
        observe_started = trace.now() if trace is not None else 0
        current = self._snapshot_provider() if snapshot is None else snapshot
        if trace is not None:
            trace.duration("nuzlocke_observer_duration_ms", observe_started)
        if self._last_frame is not None and current.frame < self._last_frame:
            self._observer = NuzlockeEventObserver()
            self._campaign_projection = CampaignProjection()
            self._rules_projection = NuzlockeRulesProjection(encounters_active=False)
            self._events.clear()
            self._session_id = str(uuid4())
            self._event_sequence = 0

        self._last_frame = current.frame
        inventory_fact = Fact.known(current.inventory) if current.inventory_available else Fact.unavailable()
        campaign_facts = derive_campaign_facts(current, inventory_fact, Fact.unavailable())
        # Pokédex receipt is the Emerald campaign fact that activates the
        # Nuzlocke encounter rule.  Inventory remains a capture/readiness
        # concern and must not affect whether a location is consumed.
        encounter_eligible = campaign_facts.pokedex_received.is_known and campaign_facts.pokedex_received.value is True
        observer_started = trace.now() if trace is not None else 0
        events = self._observer.observe(current)
        if trace is not None:
            trace.duration("nuzlocke_observer_duration_ms", observer_started)
        for event in events:
            self._event_sequence += 1
            self._campaign_projection.apply(event, session_id=self._session_id, sequence=self._event_sequence)
            self._rules_projection.apply(
                event,
                sequence=self._event_sequence,
                encounter_eligible=encounter_eligible,
            )
            if isinstance(event, BattleStarted):
                encounter = next(
                    (item for item in self._rules_projection.state.encounters if item.location == event.location),
                    None,
                )
                diagnostic_print(
                    lambda event=event, encounter=encounter: (
                        "NUZLOCKE_BATTLE_STARTED: "
                        f"runtime_id={id(self)} projection_id={id(self._rules_projection)} "
                        f"frame={event.frame} location={event.location!r} "
                        f"is_wild={event.is_wild} is_trainer={event.is_trainer} "
                        f"pokedex_received_status={campaign_facts.pokedex_received.status.value} "
                        f"pokedex_received_value={campaign_facts.pokedex_received.value!r} "
                        f"encounter_eligible={encounter_eligible} emitted=True "
                        f"projection_encounter_status={getattr(encounter, 'status', None)!r} "
                        f"projection_encounter_eligible={getattr(encounter, 'eligible', None)!r} "
                        f"projection_pokemon_identity={getattr(encounter, 'pokemon_identity', None)!r}"
                    ),
                    trace=True,
                )
            persistence_class = classify_event(event)
            self._event_statistics.record(event, persistence_class)
            if persistence_class is PersistenceClass.DURABLE and self._event_sink is not None:
                self._event_sink(event, self._session_id)
            elif persistence_class is PersistenceClass.DIAGNOSTIC and self._diagnostic_sink is not None:
                self._diagnostic_sink(event, self._session_id)
            for subscriber in tuple(self._subscribers):
                subscriber(event)
        self._events.extend(events)
        return events

    def event_statistics(self) -> dict[str, dict[str, int]]:
        """Return generated-event counts grouped by persistence class/type."""
        return self._event_statistics.snapshot()

    @property
    def session_id(self) -> str:
        """Identifier for this emulator timeline, including reset boundaries."""
        return self._session_id

    def subscribe(self, consumer: Callable[[Event], None]) -> Callable[[], None]:
        """Register a synchronous consumer and return its unsubscribe action."""
        self._subscribers.append(consumer)

        def unsubscribe() -> None:
            if consumer in self._subscribers:
                self._subscribers.remove(consumer)

        return unsubscribe

    def drain_events(self) -> tuple[Event, ...]:
        """Return queued events in order, removing them from the runtime."""
        events = tuple(self._events)
        self._events.clear()
        return events

    def mark_nuzlocke_started(self, frame: int | None = None) -> NuzlockeStarted | None:
        """Record Nuzbot's campaign boundary without inventing gameplay."""
        if self._campaign_projection.state.nuzlocke_started:
            return None
        event = NuzlockeStarted(self._last_frame if frame is None else frame)
        self._event_sequence += 1
        self._campaign_projection.apply(event, session_id=self._session_id, sequence=self._event_sequence)
        self._rules_projection.apply(event, sequence=self._event_sequence)
        if self._event_sink is not None:
            self._event_sink(event, self._session_id)
        self._event_statistics.record(event, PersistenceClass.DURABLE)
        for subscriber in tuple(self._subscribers):
            subscriber(event)
        self._events.append(event)
        return event
