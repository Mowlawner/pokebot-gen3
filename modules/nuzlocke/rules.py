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
from .persistence import JsonEventStore, deserialize_event

NO_ENCOUNTER = "none"
PENDING = "pending"
CAPTURED = "captured"
FAINTED = "fainted"
LOST = "lost_without_capture"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class LocationEncounter:
    location: tuple[int, int]
    status: str = NO_ENCOUNTER
    pokemon_identity: PokemonIdentity | None = None
    frame: int | None = None
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class RuleViolation:
    reason: str
    location: tuple[int, int] | None
    frame: int


@dataclass(frozen=True, slots=True)
class NuzlockeCampaignState:
    encounters: tuple[LocationEncounter, ...] = ()
    dead_pokemon: tuple[PokemonIdentity, ...] = ()
    unknown_faints: tuple[PokemonFainted, ...] = ()
    violations: tuple[RuleViolation, ...] = ()
    run_lost: bool = False
    last_event_sequence: int = 0

    @property
    def legal(self) -> bool:
        return not self.violations

    @property
    def first_encounters(self) -> tuple[LocationEncounter, ...]:
        return tuple(e for e in self.encounters if e.eligible)

    @property
    def unresolved_encounters(self) -> tuple[LocationEncounter, ...]:
        return tuple(e for e in self.first_encounters if e.status in (PENDING, UNKNOWN))

    @property
    def alive_pokemon(self) -> tuple[PokemonIdentity, ...]:
        return tuple(
            e.pokemon_identity
            for e in self.first_encounters
            if e.status == CAPTURED and e.pokemon_identity is not None and e.pokemon_identity not in self.dead_pokemon
        )

    @property
    def caught_first_encounters(self) -> tuple[PokemonIdentity, ...]:
        return tuple(
            e.pokemon_identity for e in self.first_encounters if e.status == CAPTURED and e.pokemon_identity is not None
        )

    def encounter_for(self, location: tuple[int, int]) -> LocationEncounter:
        """Return the first encounter at a location, or an explicit empty view."""
        return next((e for e in self.first_encounters if e.location == location), LocationEncounter(location))


class NuzlockeRulesProjection:
    """Reduce an ordered event stream into immutable Nuzlocke state."""

    def __init__(self, *, encounters_active: bool = True) -> None:
        self._state = NuzlockeCampaignState()
        self._seen: set[tuple[Any, ...]] = set()
        self._active_wild: dict[tuple[int, int], tuple[PokemonIdentity, ...]] = {}
        self._encounters_active = encounters_active

    def set_encounters_active(self, active: bool) -> None:
        """Enable the baseline rules boundary once Poké Balls exist."""
        self._encounters_active = self._encounters_active or active

    @property
    def state(self) -> NuzlockeCampaignState:
        return self._state

    def apply(
        self,
        event: Event,
        *,
        sequence: int | None = None,
        event_id: str | None = None,
        encounter_eligible: bool | None = None,
    ) -> None:
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
        if isinstance(event, PokemonFainted):
            if event.identity is None:
                state = replace(state, unknown_faints=state.unknown_faints + (event,))
            elif event.identity not in state.dead_pokemon:
                state = replace(state, dead_pokemon=state.dead_pokemon + (event.identity,))
        elif isinstance(event, WhiteoutOccurred):
            state = replace(state, run_lost=True)
        elif isinstance(event, NuzlockeStarted):
            # The campaign marker is reduced here only to keep the shared
            # event sequence replayable; encounter activation remains driven
            # by the campaign's observed Pokédex-received fact in runtime.
            pass
        elif isinstance(event, BattleStarted):
            eligible_now = self._encounters_active if encounter_eligible is None else encounter_eligible
            if eligible_now and event.is_wild and not event.is_trainer and event.location is not None:
                location = event.location
                existing = next((e for e in state.encounters if e.location == location), None)
                if existing is None:
                    identity = (
                        event.opponent_pokemon_identities[0] if len(event.opponent_pokemon_identities) == 1 else None
                    )
                    encounter = LocationEncounter(location, PENDING, identity, event.frame)
                    state = replace(state, encounters=state.encounters + (encounter,))
                elif existing.status not in (PENDING, UNKNOWN):
                    violation = RuleViolation(
                        "wild encounter after location's first encounter was resolved", location, event.frame
                    )
                    identity = (
                        event.opponent_pokemon_identities[0] if len(event.opponent_pokemon_identities) == 1 else None
                    )
                    ineligible = LocationEncounter(location, UNKNOWN, identity, event.frame, False)
                    state = replace(
                        state, encounters=state.encounters + (ineligible,), violations=state.violations + (violation,)
                    )
                self._active_wild[location] = event.opponent_pokemon_identities
        elif isinstance(event, PokemonCaptured):
            locations = ((event.location, ()),) if event.location is not None else tuple(self._active_wild.items())
            for location, opponent_ids in locations:
                if not opponent_ids or event.identity in opponent_ids:
                    state = self._resolve(state, location, CAPTURED, event.identity)
                    break
        elif isinstance(event, BattleEnded):
            if event.is_wild and not event.is_trainer and event.location is not None:
                location = event.location
                current = next((e for e in state.encounters if e.location == location), None)
                if current is not None and current.status == PENDING:
                    outcome = event.outcome.lower()
                    if outcome in {"unknown", "incomplete", "in progress", "inprogress", ""}:
                        status = UNKNOWN
                    else:
                        status = FAINTED if outcome in {"fainted", "lost", "whiteout", "opponent fainted"} else LOST
                    state = self._resolve(state, location, status, current.pokemon_identity)
                self._active_wild.pop(location, None)
        self._state = replace(state, last_event_sequence=actual)

    @staticmethod
    def _resolve(
        state: NuzlockeCampaignState, location: tuple[int, int], status: str, identity: PokemonIdentity | None
    ) -> NuzlockeCampaignState:
        encounters = tuple(
            (
                replace(e, status=status, pokemon_identity=identity or e.pokemon_identity)
                if e.location == location and e.eligible
                else e
            )
            for e in state.encounters
        )
        return replace(state, encounters=encounters)

    def apply_record(self, record: dict[str, Any]) -> None:
        required = {"event_id", "session_id", "sequence", "frame", "type", "payload"}
        if set(record) != required:
            raise ValueError("Malformed event record")
        event = deserialize_event({"type": record["type"], "payload": record["payload"]})
        if record["frame"] != event.frame:
            raise ValueError("Event record frame does not match payload")
        self.apply(event, sequence=record["sequence"], event_id=record["event_id"])


def reduce_rules(events: Iterable[Event]) -> NuzlockeCampaignState:
    projection = NuzlockeRulesProjection()
    for event in events:
        projection.apply(event)
    return projection.state


def load_rules(event_store: JsonEventStore) -> NuzlockeCampaignState:
    projection = NuzlockeRulesProjection()
    for record in event_store.iter_records():
        projection.apply_record(record)
    return projection.state
