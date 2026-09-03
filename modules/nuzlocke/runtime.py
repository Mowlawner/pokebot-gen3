"""Application-level delivery for normalized Nuzlocke observations.

This module is the boundary between the frame loop and the domain observer.
It reads no emulator state itself: callers provide a snapshot (or a snapshot
provider), and consumers receive immutable events in frame order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import uuid4
from typing import TYPE_CHECKING

from .events import BattleStarted, Event, NuzlockeEventObserver, NuzlockeStarted
from .policy import EventSink, EventStatistics, PersistenceClass, classify_event
from .snapshots import NuzlockeSnapshot, get_nuzlocke_snapshot
from .campaign_state import Fact, derive_campaign_facts
from .projection import CampaignProjection
from .rules import (
    CAPTURED,
    FAINTED,
    LOST,
    UNKNOWN,
    NuzlockeRulesProjection,
    species_clause_conflict_reason,
)
from .rule_config import CampaignRulesConfig, CampaignRuleId
from .persistence import JsonEventStore
from modules.console import diagnostic_print

if TYPE_CHECKING:
    from typing import Deque


@dataclass(frozen=True, slots=True)
class CaptureEligibility:
    """Explain whether a wild battle is authorized to use a Poké Ball."""

    eligible: bool
    reason: str
    location: tuple[int, int] | None = None
    species: tuple[str, ...] = ()
    encounter_status: str | None = None


class NuzlockeRuntime:
    """Feed one snapshot per application frame and expose each event once."""

    def __init__(
        self,
        snapshot_provider: Callable[[], NuzlockeSnapshot] = get_nuzlocke_snapshot,
        event_sink: EventSink | None = None,
        diagnostic_sink: EventSink | None = None,
        session_id: str | None = None,
        rule_config: CampaignRulesConfig | None = None,
        event_store: JsonEventStore | None = None,
        defer_event_persistence: bool = False,
    ) -> None:
        """Create a snapshot-driven runtime with optional durable event sinks.

        ``defer_event_persistence`` keeps durable events in memory until the
        caller reports a successful save boundary.  The projections still
        consume every event immediately; only the external sink is deferred.
        This lets the emulator save and the event history advance together,
        while preserving immediate sink delivery for lightweight embedders and
        existing callers by default.
        """

        self._snapshot_provider = snapshot_provider
        self._event_sink = event_sink
        self._diagnostic_sink = diagnostic_sink
        self._session_id = session_id or str(uuid4())
        self._rule_config = rule_config or CampaignRulesConfig()
        self._event_sequence = 0
        self._observer = NuzlockeEventObserver()
        self._events: Deque[Event] = deque()
        self._subscribers: list[Callable[[Event], None]] = []
        self._last_frame: int | None = None
        self._campaign_projection = CampaignProjection()
        self._rules_projection = NuzlockeRulesProjection(encounters_active=False, rule_config=self._rule_config)
        self._event_statistics = EventStatistics()
        self._event_store = event_store
        self._defer_event_persistence = defer_event_persistence
        self._pending_durable_events: list[tuple[Event, str]] = []
        self._campaign_history_checked = event_store is None
        self._campaign_history_compatible = True
        # Do not replay durable history until a complete ROM-backed campaign
        # observation has established that the event store belongs to the
        # loaded save.  In particular, constructing the runtime must not let a
        # stale NuzlockeStarted/projection state influence the first campaign
        # selection.
        self._campaign_history_hydrated = event_store is None
        self._latest_snapshot: NuzlockeSnapshot | None = None
        self._latest_campaign_facts = None

    def _hydrate(self, event_store: JsonEventStore | None) -> None:
        """Replay durable history into the projections before live updates."""
        if event_store is None:
            return
        for record in event_store.iter_records():
            self._campaign_projection.apply_record(record)
            self._rules_projection.apply_record(record)
        self._event_sequence = event_store.last_sequence()
        self._campaign_history_hydrated = True

    @property
    def observed_projection(self) -> CampaignProjection:
        """Return the read-only projection of observed campaign transitions."""

        return self._campaign_projection

    @property
    def rules_projection(self) -> NuzlockeRulesProjection:
        """Return the current reduced Nuzlocke legality projection."""

        return self._rules_projection

    @property
    def rule_config(self) -> CampaignRulesConfig:
        """Return the immutable rule configuration used by this runtime."""

        return self._rule_config

    @property
    def campaign_history_compatible(self) -> bool:
        """Whether durable history was accepted for the current ROM save."""
        return self._campaign_history_compatible

    @property
    def campaign_history_ready(self) -> bool:
        """Whether the current ROM has authorized projection hydration.

        A runtime backed by an event store starts unready and becomes ready
        only after the first complete campaign observation has either accepted
        or quarantined that store.  This lets campaign consumers distinguish a
        fresh, empty projection from a projection that has not yet been
        reconciled with the loaded save.
        """
        return self._campaign_history_checked and self._campaign_history_hydrated

    @property
    def latest_snapshot(self) -> NuzlockeSnapshot | None:
        """Return the snapshot materialized by the most recent update.

        The frame loop updates the runtime before campaign planning.  Exposing
        that immutable observation lets planners and readiness diagnostics
        share the same ROM read instead of independently rereading save blocks
        while an emulator frame is still being assembled.
        """
        return self._latest_snapshot

    @staticmethod
    def _campaign_provenance_observation(snapshot: NuzlockeSnapshot, facts) -> dict:
        """Build the save-backed provenance view used to scope event history."""

        identity = None
        for pokemon in sorted(snapshot.party, key=lambda item: item.party_index):
            if pokemon.identity is not None:
                identity = [
                    pokemon.identity.personality_value,
                    pokemon.identity.original_trainer_id,
                    pokemon.identity.original_trainer_secret_id,
                ]
                break
        return {
            "game_id": snapshot.game_id,
            "lifecycle": snapshot.campaign_observation.lifecycle.value,
            "facts": {
                name: getattr(facts, name).value
                for name in (
                    "new_game_setup_complete",
                    "wall_clock_set",
                    "rival_met",
                    "birch_rescued",
                    "starter_obtained",
                    "intro_rival_battle_complete",
                    "pokedex_received",
                    "pokeballs_available",
                    "pokeballs_ready",
                    "pokeballs_received",
                    "visited_petalburg",
                    "petalburg_wally_scene_complete",
                    "petalburg_woods_scene_complete",
                    "devon_goods_stolen",
                    "devon_goods_reported",
                    "devon_goods_recovered",
                    "devon_goods_returned",
                    "devon_goods_delivered",
                    "devon_corp_3f_scene_complete",
                    "visited_rustboro",
                    "first_badge_obtained",
                )
                if getattr(facts, name).is_known
            },
            "stable_identity": identity,
        }

    @staticmethod
    def _campaign_observation_is_complete(facts) -> bool:
        """Return whether the ROM facts are sufficient to scope history.

        ``CampaignObservationSnapshot.available`` means that the observation
        read path completed, but it does not by itself prove that every
        named field was present.  Provenance reconciliation must be able to
        observe both sides of each opening milestone; otherwise an absent
        field could make stale high-water history appear compatible.
        """
        return all(
            getattr(facts, name).is_known
            for name in (
                "new_game_setup_complete",
                "wall_clock_set",
                "rival_met",
                "birch_rescued",
                "starter_obtained",
                "intro_rival_battle_complete",
                "pokedex_received",
            )
        )

    def _check_campaign_history(self, snapshot: NuzlockeSnapshot, facts) -> bool:
        """Authorize durable history only after checking the loaded ROM save."""
        if self._event_store is None:
            return True
        if self._campaign_history_checked:
            # Keep the sidecar's monotonic high-water facts current after the
            # initial compatibility decision, but never let a transiently
            # older map/script observation quarantine an already-active run.
            if snapshot.campaign_observation.available:
                observation = self._campaign_provenance_observation(snapshot, facts)
                self._event_store.campaign_history_compatible(
                    observation,
                    enforce=False,
                    persist=not self._defer_event_persistence,
                )
            return True
        # During boot, campaign RAM can be temporarily unavailable. Defer the
        # decision until the first complete observation rather than treating
        # an incomplete frame as a new run.
        if not snapshot.campaign_observation.available or not self._campaign_observation_is_complete(facts):
            return False
        observation = self._campaign_provenance_observation(snapshot, facts)
        # Enforce compatibility only at the first complete observation of a
        # runtime timeline.  Later complete observations still advance the
        # provenance high-water mark, but must not quarantine the active run
        # when a map-local/script boundary briefly exposes an older value.
        # A real emulator reset creates a new runtime below and therefore
        # performs the strict check again against the newly loaded save.
        compatible = self._event_store.campaign_history_compatible(
            observation,
            enforce=True,
            persist=not self._defer_event_persistence,
        )
        self._campaign_history_checked = True
        self._campaign_history_compatible = compatible
        if compatible:
            self._hydrate(self._event_store)
            return True

        archived = self._event_store.quarantine()
        self._observer = NuzlockeEventObserver()
        self._campaign_projection = CampaignProjection()
        self._rules_projection = NuzlockeRulesProjection(
            encounters_active=False,
            rule_config=self._rule_config,
        )
        self._events.clear()
        self._event_sequence = 0
        diagnostic_print(
            lambda: (
                "NUZLOCKE_HISTORY_QUARANTINED: "
                f"event_store={self._event_store.path!s} archived={archived!s} "
                "reason='loaded ROM save contradicts durable campaign provenance'"
            ),
            trace=True,
        )
        self._campaign_history_hydrated = True
        return True

    def capture_eligibility_for(
        self,
        location: tuple[int, int] | None,
        *,
        is_wild: bool,
        is_trainer: bool,
        species: tuple[str, ...] | None = None,
    ) -> CaptureEligibility:
        """Return a structured, read-only capture authorization decision.

        This never claims an encounter; the observer claims it when
        ``BattleStarted`` is delivered.  The result is also the hard safety
        boundary used by the Nuzlocke capture strategy before it can select a
        ball, so an ineligible encounter cannot be captured by a policy race.
        """
        projection = self._rules_projection
        if species is None:
            snapshot_battle = getattr(self._latest_snapshot, "battle", None)
            species = tuple(
                getattr(pokemon, "species", "")
                for pokemon in getattr(snapshot_battle, "opponent_active", ())
                if getattr(pokemon, "species", None)
            )
        species = tuple(species or ())

        def result(eligible: bool, reason: str, status: str | None = None) -> CaptureEligibility:
            decision = CaptureEligibility(eligible, reason, location, species, status)
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_CAPTURE_ELIGIBILITY: "
                    f"location={location!r} species={species!r} eligible={eligible!r} "
                    f"reason={reason!r} encounter_status={status!r}"
                ),
                trace=True,
            )
            return decision

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
            reason = "trainer battle is never capture-eligible" if is_trainer else (
                "battle is not wild" if not is_wild else "encounter location is unavailable"
            )
            return result(False, reason)

        facts = self._latest_campaign_facts
        pokedex = getattr(facts, "pokedex_received", None)
        if pokedex is None or not getattr(pokedex, "is_known", False):
            return result(False, "pokedex receipt is unknown")
        if pokedex.value is not True:
            return result(False, "previous encounters were before pokedex receipt")
        if not projection.rule_config.is_enabled(CampaignRuleId.ONE_ENCOUNTER_PER_AREA):
            return result(False, "one-encounter-per-area rule is disabled")

        if projection.rule_config.is_enabled(CampaignRuleId.SPECIES_CLAUSE) and species:
            conflict = species_clause_conflict_reason(projection.state, species)
            if conflict is not None:
                return result(False, conflict)

        encounters = tuple(item for item in projection.state.encounters if item.location == location)
        encounter = next((item for item in encounters if item.eligible), None)
        if encounter is None:
            if encounters:
                return result(False, "encounter was recorded as ineligible", encounters[-1].status)
            return result(False, "wild encounter has not been registered by the observer")
        if encounter.status == "pending":
            return result(True, "first eligible encounter is pending", encounter.status)
        if encounter.status in (LOST, FAINTED):
            return result(False, "failed to catch first eligible encounter", encounter.status)
        if encounter.status == CAPTURED:
            return result(False, "area encounter already resolved", encounter.status)
        if encounter.status == UNKNOWN:
            return result(False, "first eligible encounter outcome is unknown", encounter.status)
        return result(False, "area encounter is not pending", encounter.status)

    def capture_target_for(
        self,
        location: tuple[int, int] | None,
        *,
        is_wild: bool,
        is_trainer: bool,
        species: tuple[str, ...] | None = None,
    ) -> bool:
        """Compatibility boolean wrapper around :meth:`capture_eligibility_for`."""

        return self.capture_eligibility_for(
            location,
            is_wild=is_wild,
            is_trainer=is_trainer,
            species=species,
        ).eligible

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
        self._latest_snapshot = current
        if trace is not None:
            trace.duration("nuzlocke_observer_duration_ms", observe_started)
        if self._last_frame is not None and current.frame < self._last_frame:
            # A save-state load starts a new emulator timeline, but it does
            # not erase the campaign's durable history.  Recreate only the
            # frame observer and hydrate projections from the event store so
            # encounter/death/campaign facts remain available after reset.
            self._observer = NuzlockeEventObserver()
            self._campaign_projection = CampaignProjection()
            self._rules_projection = NuzlockeRulesProjection(encounters_active=False, rule_config=self._rule_config)
            self._events.clear()
            self._session_id = str(uuid4())
            self._event_sequence = 0
            self._campaign_history_checked = False
            self._campaign_history_compatible = True
            self._campaign_history_hydrated = self._event_store is None
            # Events produced after the last durable boundary belong to the
            # discarded emulator timeline and must not be committed after a
            # save-state load/reset.
            self._pending_durable_events.clear()
            self._latest_snapshot = None
            self._latest_campaign_facts = None

        self._last_frame = current.frame
        inventory_fact = Fact.known(current.inventory) if current.inventory_available else Fact.unavailable()
        campaign_facts = derive_campaign_facts(
            current,
            inventory_fact,
            Fact.unavailable(),
            pokeball_policy=self._rule_config.pokeball_policy,
        )
        self._latest_campaign_facts = campaign_facts
        if not self._check_campaign_history(current, campaign_facts):
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_HISTORY_WAITING: "
                    "campaign observation is incomplete; durable history and live events remain hidden"
                ),
                trace=True,
            )
            return ()
        # Pokédex receipt is the Emerald campaign fact that activates the
        # Nuzlocke encounter rule.  Inventory remains a capture/readiness
        # concern and must not affect whether a location is consumed.
        encounter_eligible = (
            self._rule_config.is_enabled(CampaignRuleId.ONE_ENCOUNTER_PER_AREA)
            and campaign_facts.pokedex_received.is_known
            and campaign_facts.pokedex_received.value is True
        )
        observer_started = trace.now() if trace is not None else 0
        events = self._observer.observe(current)
        if trace is not None:
            trace.duration("nuzlocke_observer_duration_ms", observer_started)
        for event in events:
            if isinstance(event, BattleStarted) and event.encounter_eligible is None:
                # Enrich the observer event before it reaches either the
                # projections or the durable sink.  The eligibility decision
                # is a fact of the battle-start frame, not a replay-time
                # inference from a later snapshot.
                event_encounter_eligible = encounter_eligible
                if event_encounter_eligible and event.is_wild and not event.is_trainer and event.location is not None:
                    # The global Pokédex boundary enables encounter rules,
                    # but it does not make a second encounter in an already
                    # resolved area eligible.  Repeat battles remain valid
                    # battles to fight; only their capture target is barred.
                    event_encounter_eligible = (
                        self._rules_projection.state.encounter_for(event.location).status == "none"
                    )
                if (
                    event_encounter_eligible
                    and event.is_wild
                    and not event.is_trainer
                    and self._rule_config.is_enabled(CampaignRuleId.SPECIES_CLAUSE)
                    and species_clause_conflict_reason(
                        self._rules_projection.state, event.opponent_species
                    ) is not None
                ):
                    # A duplicate species is still a real battle, but it is
                    # not the route's eligible encounter under Species Clause.
                    event_encounter_eligible = False
                event = replace(event, encounter_eligible=event_encounter_eligible)
                encounter_eligible_for_event = event_encounter_eligible
            else:
                encounter_eligible_for_event = encounter_eligible
            self._event_sequence += 1
            self._campaign_projection.apply(event, session_id=self._session_id, sequence=self._event_sequence)
            self._rules_projection.apply(
                event,
                sequence=self._event_sequence,
                encounter_eligible=encounter_eligible_for_event,
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
                        # Keep the global Pokédex/rule gate separate from the
                        # final decision attached to this event.  The latter
                        # also includes area and Species Clause checks.
                        f"encounter_eligible={event.encounter_eligible!r} "
                        f"global_encounter_gate={encounter_eligible!r} emitted=True "
                        f"projection_encounter_status={getattr(encounter, 'status', None)!r} "
                        f"projection_encounter_eligible={getattr(encounter, 'eligible', None)!r} "
                        f"projection_encounter_frame={getattr(encounter, 'frame', None)!r} "
                        f"projection_pokemon_identity={getattr(encounter, 'pokemon_identity', None)!r}"
                    ),
                    trace=True,
                )
            persistence_class = classify_event(event)
            self._event_statistics.record(event, persistence_class)
            if persistence_class is PersistenceClass.DURABLE and self._event_sink is not None:
                self._persist_or_defer(event)
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
            """Remove the consumer if it is still subscribed."""

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
            self._persist_or_defer(event)
        self._event_statistics.record(event, PersistenceClass.DURABLE)
        for subscriber in tuple(self._subscribers):
            subscriber(event)
        self._events.append(event)
        return event

    def _persist_or_defer(self, event: Event) -> None:
        """Deliver a durable event now, or hold it for a save boundary."""

        if self._defer_event_persistence:
            self._pending_durable_events.append((event, self._session_id))
            return
        self._event_sink(event, self._session_id)

    @property
    def pending_durable_event_count(self) -> int:
        """Return the number of durable events awaiting a save boundary."""

        return len(self._pending_durable_events)

    def commit_pending_events(self, boundary: str = "save") -> int:
        """Persist events observed since the previous successful save.

        The event sink is called in observer order.  Events are removed only
        after the sink accepts them, so a failed write leaves the uncommitted
        suffix available for a later explicit save boundary.
        """

        if not self._defer_event_persistence or self._event_sink is None:
            return 0
        committed = 0
        while self._pending_durable_events:
            event, session_id = self._pending_durable_events[0]
            self._event_sink(event, session_id)
            self._pending_durable_events.pop(0)
            committed += 1
        if (
            self._event_store is not None
            and self._latest_snapshot is not None
            and self._latest_campaign_facts is not None
        ):
            # Provenance is intentionally committed at the same boundary as
            # the event batch.  Without this, milestone observations would
            # create profile writes on ordinary frames even when event
            # persistence is deferred.
            observation = self._campaign_provenance_observation(
                self._latest_snapshot,
                self._latest_campaign_facts,
            )
            self._event_store.campaign_history_compatible(
                observation,
                enforce=False,
                persist=True,
            )
        diagnostic_print(
            lambda: (
                "NUZLOCKE_EVENT_COMMIT: "
                f"boundary={boundary!r} committed={committed} "
                f"runtime_id={id(self)!r} session_id={self._session_id!r}"
            ),
            trace=True,
        )
        return committed
