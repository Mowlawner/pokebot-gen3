"""Deterministic, read-only Nuzlocke rules reduction.

This module consumes immutable observation events.  It does not inspect the
emulator, alter events, or persist campaign state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Any

from .events import (
    BattleEnded,
    BattleStarted,
    Event,
    GameStateChanged,
    MapChanged,
    PartyChanged,
    StorageChanged,
    PokemonCaptured,
    PokemonFainted,
    WhiteoutOccurred,
    NuzlockeStarted,
)
from .identity import PokemonIdentity
from .rule_config import CampaignRuleId, CampaignRulesConfig
from .persistence import JsonEventStore, deserialize_event

NO_ENCOUNTER = "none"
PENDING = "pending"
CAPTURED = "captured"
FAINTED = "fainted"
LOST = "lost_without_capture"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LocationEncounter:
    """Reduced first-encounter state for one map location."""

    location: tuple[int, int]
    status: str = NO_ENCOUNTER
    pokemon_identity: PokemonIdentity | None = None
    frame: int | None = None
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class RuleViolation:
    """A rule breach observed at a location and frame."""

    reason: str
    location: tuple[int, int] | None
    frame: int


@dataclass(frozen=True, slots=True)
class RuleAssessment:
    """A pure, rule-local view of reduced campaign legality."""

    rule_id: CampaignRuleId
    legal: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NuzlockeCampaignState:
    """Immutable legality state produced by the enabled Nuzlocke rules."""

    encounters: tuple[LocationEncounter, ...] = ()
    dead_pokemon: tuple[PokemonIdentity, ...] = ()
    unknown_faints: tuple[PokemonFainted, ...] = ()
    violations: tuple[RuleViolation, ...] = ()
    run_lost: bool = False
    last_event_sequence: int = 0

    @property
    def legal(self) -> bool:
        """Return whether no rule violation has been reduced so far."""

        return not self.violations

    @property
    def first_encounters(self) -> tuple[LocationEncounter, ...]:
        """Return encounters eligible for the run's first-encounter rule."""

        return tuple(e for e in self.encounters if e.eligible)

    @property
    def unresolved_encounters(self) -> tuple[LocationEncounter, ...]:
        """Return eligible encounters whose outcomes are not settled."""

        return tuple(e for e in self.first_encounters if e.status in (PENDING, UNKNOWN))

    @property
    def alive_pokemon(self) -> tuple[PokemonIdentity, ...]:
        """Return captured first-encounter Pokémon not marked dead."""

        return tuple(
            e.pokemon_identity
            for e in self.first_encounters
            if e.status == CAPTURED and e.pokemon_identity is not None and e.pokemon_identity not in self.dead_pokemon
        )

    @property
    def caught_first_encounters(self) -> tuple[PokemonIdentity, ...]:
        """Return identities captured from eligible first encounters."""

        return tuple(
            e.pokemon_identity for e in self.first_encounters if e.status == CAPTURED and e.pokemon_identity is not None
        )

    def encounter_for(self, location: tuple[int, int]) -> LocationEncounter:
        """Return the first encounter at a location, or an explicit empty view."""
        return next((e for e in self.first_encounters if e.location == location), LocationEncounter(location))


class FaintingRule:
    """Reduce permanent fainting and whiteout events when enabled."""

    rule_id = CampaignRuleId.FAINTING

    def apply(
        self,
        state: NuzlockeCampaignState,
        event: Event,
        *,
        encounter_eligible: bool,
        active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]],
    ) -> NuzlockeCampaignState:
        """Reduce faint and whiteout events into permanent run state."""

        if isinstance(event, PokemonFainted):
            if event.identity is None:
                return replace(state, unknown_faints=state.unknown_faints + (event,))
            if event.identity not in state.dead_pokemon:
                return replace(state, dead_pokemon=state.dead_pokemon + (event.identity,))
        elif isinstance(event, WhiteoutOccurred):
            return replace(state, run_lost=True)
        return state

    def evaluate(self, state: NuzlockeCampaignState) -> RuleAssessment:
        """Assess whether the reduced run has suffered a whiteout."""

        return RuleAssessment(
            self.rule_id,
            not state.run_lost,
            ("whiteout occurred",) if state.run_lost else (),
        )

    def constrain(self, state: NuzlockeCampaignState, candidate):
        """Leave candidate selection unchanged for this observation rule."""

        return candidate


class OneEncounterPerAreaRule:
    """Reduce first-wild-encounter ownership without changing event history."""

    rule_id = CampaignRuleId.ONE_ENCOUNTER_PER_AREA

    def apply(
        self,
        state: NuzlockeCampaignState,
        event: Event,
        *,
        encounter_eligible: bool,
        active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]],
    ) -> NuzlockeCampaignState:
        """Reduce eligible wild battles into first-encounter outcomes."""

        if isinstance(event, BattleStarted):
            if not (encounter_eligible and event.is_wild and not event.is_trainer and event.location is not None):
                return state
            location = event.location
            existing = next((e for e in state.encounters if e.location == location), None)
            if existing is None:
                identity = event.opponent_pokemon_identities[0] if len(event.opponent_pokemon_identities) == 1 else None
                state = replace(
                    state, encounters=state.encounters + (LocationEncounter(location, PENDING, identity, event.frame),)
                )
            elif existing.status not in (PENDING, UNKNOWN):
                identity = event.opponent_pokemon_identities[0] if len(event.opponent_pokemon_identities) == 1 else None
                # A repeat wild battle is still legal to fight.  The
                # one-encounter rule constrains capture eligibility, not the
                # player's ability to battle or escape incidental encounters.
                # Keep an explicit ineligible history record so later
                # capture handling cannot mistake it for a first encounter.
                state = replace(
                    state,
                    encounters=state.encounters + (LocationEncounter(location, UNKNOWN, identity, event.frame, False),),
                )
            active_wild[location] = event.opponent_pokemon_identities
        elif isinstance(event, PokemonCaptured):
            locations = ((event.location, ()),) if event.location is not None else tuple(active_wild.items())
            for location, opponent_ids in locations:
                if not opponent_ids or event.identity in opponent_ids:
                    current = next((item for item in state.encounters if item.location == location), None)
                    # Capture events from an incidental repeat must not
                    # mutate the already-resolved first encounter.  Only a
                    # pending eligible battle owns a capture outcome.
                    if current is not None and current.status == PENDING:
                        return self._resolve(state, location, CAPTURED, event.identity)
        elif isinstance(event, BattleEnded) and event.is_wild and not event.is_trainer and event.location is not None:
            current = next((e for e in state.encounters if e.location == event.location), None)
            if current is not None and current.status == PENDING:
                outcome = event.outcome.lower()
                status = (
                    UNKNOWN
                    if outcome in {"unknown", "incomplete", "in progress", "inprogress", ""}
                    else (FAINTED if outcome in {"fainted", "lost", "whiteout", "opponent fainted"} else LOST)
                )
                state = self._resolve(state, event.location, status, current.pokemon_identity)
            active_wild.pop(event.location, None)
        return state

    def evaluate(self, state: NuzlockeCampaignState) -> RuleAssessment:
        """Assess whether the encounter history contains a repeat encounter."""

        reasons = tuple(violation.reason for violation in state.violations)
        return RuleAssessment(self.rule_id, not reasons, reasons)

    def constrain(self, state: NuzlockeCampaignState, candidate):
        """Leave candidate selection unchanged for this observation rule."""

        return candidate

    @staticmethod
    def _resolve(
        state: NuzlockeCampaignState, location: tuple[int, int], status: str, identity: PokemonIdentity | None
    ) -> NuzlockeCampaignState:
        """Return state with the eligible encounter at ``location`` resolved."""

        return replace(
            state,
            encounters=tuple(
                (
                    replace(item, status=status, pokemon_identity=identity or item.pokemon_identity)
                    if item.location == location and item.eligible
                    else item
                )
                for item in state.encounters
            ),
        )


class NuzlockeRulesProjection:
    """Reduce an ordered event stream into immutable Nuzlocke state."""

    def __init__(
        self,
        *,
        encounters_active: bool = True,
        rule_config: CampaignRulesConfig | None = None,
    ) -> None:
        """Create a reducer with the configured rule set and encounter gate."""

        self._state = NuzlockeCampaignState()
        self._seen: set[tuple[Any, ...]] = set()
        self._active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]] = {}
        self._encounters_active = encounters_active
        self._rule_config = rule_config or CampaignRulesConfig()
        self._rules = tuple(
            rule for rule in (OneEncounterPerAreaRule(), FaintingRule()) if self._rule_config.is_enabled(rule.rule_id)
        )

    @property
    def rule_config(self) -> CampaignRulesConfig:
        """Return the immutable run configuration used by this reduction."""
        return self._rule_config

    @property
    def rule_assessments(self) -> tuple[RuleAssessment, ...]:
        """Return each enabled rule's current pure assessment."""

        return tuple(rule.evaluate(self._state) for rule in self._rules)

    def set_encounters_active(self, active: bool) -> None:
        """Enable the baseline rules boundary once Poké Balls exist."""
        self._encounters_active = self._encounters_active or active

    @property
    def state(self) -> NuzlockeCampaignState:
        """Return the current immutable reduced state."""

        return self._state

    def apply(
        self,
        event: Event,
        *,
        sequence: int | None = None,
        event_id: str | None = None,
        encounter_eligible: bool | None = None,
    ) -> None:
        """Apply one ordered event, ignoring only an already-seen event key."""

        if not isinstance(
            event,
            (
                BattleStarted,
                BattleEnded,
                PokemonCaptured,
                PokemonFainted,
                WhiteoutOccurred,
                GameStateChanged,
                MapChanged,
                PartyChanged,
                StorageChanged,
                NuzlockeStarted,
            ),
        ):
            raise TypeError(f"Unsupported Nuzlocke rules event: {type(event).__name__}")
        key = (event_id,) if event_id is not None else (type(event).__name__, event)
        if key in self._seen:
            return
        expected = self._state.last_event_sequence + 1
        actual = expected if sequence is None else sequence
        if actual != expected:
            raise ValueError(f"Event sequence out of order: expected {expected}, got {actual}")
        self._seen.add(key)
        state = self._state
        eligible_now = self._encounters_active if encounter_eligible is None else encounter_eligible
        if isinstance(event, BattleStarted) and event.encounter_eligible is not None:
            # A persisted BattleStarted owns the eligibility decision made at
            # its observation boundary.  This takes precedence over the
            # caller's current runtime state during replay.
            eligible_now = event.encounter_eligible
        for rule in self._rules:
            state = rule.apply(
                state,
                event,
                encounter_eligible=eligible_now,
                active_wild=self._active_wild,
            )
        self._state = replace(state, last_event_sequence=actual)
        if isinstance(event, NuzlockeStarted):
            # Older persisted BattleStarted records predate the explicit
            # encounter_eligible field.  The durable campaign boundary is
            # enough to restore the historical rule activation point for
            # those records; newer records still use their own frame-local
            # eligibility decision above.
            self._encounters_active = True

    def apply_record(self, record: dict[str, Any]) -> None:
        """Validate and apply one persisted event-store record."""

        required = {"event_id", "session_id", "sequence", "frame", "type", "payload"}
        if set(record) != required:
            raise ValueError("Malformed event record")
        event = deserialize_event({"type": record["type"], "payload": record["payload"]})
        if record["frame"] != event.frame:
            raise ValueError("Event record frame does not match payload")
        self.apply(event, sequence=record["sequence"], event_id=record["event_id"])


def reduce_rules(events: Iterable[Event]) -> NuzlockeCampaignState:
    """Reduce an in-memory event iterable into Nuzlocke campaign state."""

    projection = NuzlockeRulesProjection()
    for event in events:
        projection.apply(event)
    return projection.state


def load_rules(event_store: JsonEventStore) -> NuzlockeCampaignState:
    """Replay all records from an event store through the rules projection."""

    projection = NuzlockeRulesProjection()
    for record in event_store.iter_records():
        projection.apply_record(record)
    return projection.state
