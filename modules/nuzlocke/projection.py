"""Read-only reconstruction of observed Nuzlocke campaign facts.

Events are facts emitted by the observer; this module only reduces those
facts into a compact, immutable view.  It deliberately does not apply any
Nuzlocke legality or death rules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .events import (
    BattleEnded,
    BattleStarted,
    Event,
    GameStateChanged,
    MapChanged,
    PartyChanged,
    PokemonCaptured,
    PokemonFainted,
    StorageChanged,
    WhiteoutOccurred,
)
from .persistence import JsonEventStore, deserialize_event, serialize_event
from .identity import PokemonIdentity


@dataclass(frozen=True, slots=True)
class ObservedPokemon:
    """Details observed for a Pokémon faint; this is not a death record."""

    personality_value: int
    species: str
    nickname: str
    last_faint_frame: int
    faint_count: int
    identity: PokemonIdentity | None = None


@dataclass(frozen=True, slots=True)
class ObservedBattle:
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool
    started_frame: int


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    sequence: int
    event_type: str
    session_id: str | None
    frame: int


@dataclass(frozen=True, slots=True)
class ObservedCampaignState:
    current_session_id: str | None = None
    last_event_sequence: int = 0
    last_observed_frame: int | None = None
    known_session_ids: tuple[str, ...] = ()
    current_map: tuple[int, int] | None = None
    map_history: tuple[ObservedEvent, ...] = ()
    current_game_state: Any = None
    last_game_state_change: ObservedEvent | None = None
    active_battle: ObservedBattle | None = None
    last_battle_end: BattleEnded | None = None
    party_indices: tuple[int, ...] | None = None
    party_identities: tuple[PokemonIdentity, ...] | None = None
    pc_locations: tuple[tuple[PokemonIdentity, int, int], ...] | None = None
    last_party_change: ObservedEvent | None = None
    observed_pokemon: tuple[ObservedPokemon, ...] = ()
    observed_faints: tuple[PokemonFainted, ...] = ()
    observed_whiteouts: tuple[ObservedEvent, ...] = ()
    observed_captures: tuple[PokemonCaptured, ...] = ()


def _event_key(event: Event, session_id: str | None) -> str:
    payload = json.dumps(serialize_event(event), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{session_id}\0{event.frame}\0{payload}".encode()).hexdigest()


class CampaignProjection:
    """Deterministically reduce ordered events into observed campaign state.

    ``apply`` accepts event objects for simple replay.  ``apply_record`` is
    preferred for persisted events because it retains session and global
    sequence metadata from :class:`JsonEventStore`.

    ``PartyChanged`` does not carry Pokémon identities or a complete party
    snapshot.  Consequently ``party_indices`` is ``None`` until a reliable
    index set can be assembled from deltas; faint observations are retained
    separately and are not treated as deaths.
    """

    def __init__(self) -> None:
        self._state = ObservedCampaignState()
        self._seen: set[str] = set()

    @property
    def state(self) -> ObservedCampaignState:
        return self._state

    def apply(
        self,
        event: Event,
        *,
        session_id: str | None = None,
        sequence: int | None = None,
        event_id: str | None = None,
    ) -> None:
        if not isinstance(
            event,
            (
                BattleStarted,
                BattleEnded,
                MapChanged,
                PartyChanged,
                PokemonFainted,
                PokemonCaptured,
                StorageChanged,
                WhiteoutOccurred,
                GameStateChanged,
            ),
        ):
            raise TypeError(f"Unsupported Nuzlocke event: {type(event).__name__}")
        key = event_id or _event_key(event, session_id)
        if key in self._seen:
            return
        expected = self._state.last_event_sequence + 1
        actual = expected if sequence is None else sequence
        if actual != expected:
            raise ValueError(f"Event sequence out of order: expected {expected}, got {actual}")
        self._seen.add(key)
        meta = ObservedEvent(actual, type(event).__name__, session_id, event.frame)
        state = self._state
        sessions = state.known_session_ids
        if session_id is not None and session_id not in sessions:
            sessions = sessions + (session_id,)
        changes = dict(
            last_event_sequence=actual,
            last_observed_frame=event.frame,
            current_session_id=(session_id if session_id is not None else state.current_session_id),
            known_session_ids=sessions,
        )
        if isinstance(event, MapChanged):
            changes["current_map"] = event.new_map
            changes["map_history"] = state.map_history + (meta,)
        elif isinstance(event, GameStateChanged):
            changes["current_game_state"] = event.new_state
            changes["last_game_state_change"] = meta
        elif isinstance(event, BattleStarted):
            changes["active_battle"] = ObservedBattle(
                event.battle_type,
                event.is_trainer,
                event.is_wild,
                event.is_double,
                event.frame,
            )
        elif isinstance(event, BattleEnded):
            changes["active_battle"] = None
            changes["last_battle_end"] = event
        elif isinstance(event, PartyChanged):
            indices = set(state.party_indices or ())
            if state.party_indices is not None or event.entered_party_indices:
                indices.update(event.entered_party_indices)
                indices.difference_update(event.left_party_indices)
                changes["party_indices"] = tuple(sorted(indices))
            changes["last_party_change"] = meta
            if event.entered_identities or event.left_identities or event.changed_identities:
                identities = list(state.party_identities or ())
                identities.extend(event.entered_identities)
                identities = [identity for identity in identities if identity not in event.left_identities]
                changes["party_identities"] = tuple(identities)
        elif isinstance(event, PokemonFainted):
            changes["observed_faints"] = state.observed_faints + (event,)
            old = next(
                (
                    p
                    for p in state.observed_pokemon
                    if (event.identity is not None and p.identity == event.identity)
                    or (event.identity is None and p.personality_value == event.personality_value)
                ),
                None,
            )
            replacement = ObservedPokemon(
                event.personality_value,
                event.species,
                event.nickname,
                event.frame,
                (old.faint_count + 1) if old else 1,
                event.identity,
            )
            changes["observed_pokemon"] = (
                tuple(
                    (
                        replacement
                        if (
                            (event.identity is not None and p.identity == event.identity)
                            or (event.identity is None and p.personality_value == event.personality_value)
                        )
                        else p
                    )
                    for p in state.observed_pokemon
                )
                if old
                else state.observed_pokemon + (replacement,)
            )
        elif isinstance(event, PokemonCaptured):
            changes["observed_captures"] = state.observed_captures + (event,)
        elif isinstance(event, StorageChanged):
            locations = dict((identity, (box, slot)) for identity, box, slot in (state.pc_locations or ()))
            for location in event.left:
                locations.pop(location.identity, None)
            for location in event.entered:
                locations[location.identity] = (location.box, location.slot)
            for old_location, new_location in event.moved:
                locations[new_location.identity] = (new_location.box, new_location.slot)
            changes["pc_locations"] = tuple((identity, box, slot) for identity, (box, slot) in locations.items())
        elif isinstance(event, WhiteoutOccurred):
            changes["observed_whiteouts"] = state.observed_whiteouts + (meta,)
        self._state = replace(state, **changes)

    def apply_record(self, record: dict[str, Any]) -> None:
        required = {"event_id", "session_id", "sequence", "frame", "type", "payload"}
        if set(record) != required:
            raise ValueError("Malformed event record")
        event = deserialize_event({"type": record["type"], "payload": record["payload"]})
        if record["frame"] != event.frame:
            raise ValueError("Event record frame does not match payload")
        self.apply(
            event,
            session_id=record["session_id"],
            sequence=record["sequence"],
            event_id=record["event_id"],
        )


def reduce_events(events: Iterable[Event]) -> ObservedCampaignState:
    projection = CampaignProjection()
    for event in events:
        projection.apply(event)
    return projection.state


def load_campaign_projection(event_store: JsonEventStore) -> ObservedCampaignState:
    projection = CampaignProjection()
    for record in event_store.iter_records():
        projection.apply_record(record)
    return projection.state
