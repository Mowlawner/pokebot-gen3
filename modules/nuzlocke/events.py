"""Observable transitions derived from :mod:`modules.nuzlocke.snapshots`.

This module deliberately contains no Nuzlocke rules or persistence.  The
observer only compares its most recent snapshot with the current one.

Battle boundaries use the snapshot's battle presence, which corresponds to
the existing BattleListener's battle-state lifecycle.  Whiteout uses the
stable ``GameState.WHITEOUT`` boundary.  The existing PoisonListener has a
more precise script hook, but it is not part of a snapshot; consequently the
``poison`` cause is only reported for a party HP transition to zero while the
snapshot is in the overworld and still reports a poison status.  Other faint
causes remain ``battle`` or ``unknown``.  This observer is therefore not a
replacement for those listeners when exact callback timing is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .snapshots import NuzlockeSnapshot, PartyPokemonSnapshot
from .identity import PokemonIdentity


@dataclass(frozen=True, slots=True)
class BattleStarted:
    frame: int
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool
    own_pokemon_identities: tuple[PokemonIdentity, ...] = ()


@dataclass(frozen=True, slots=True)
class BattleEnded:
    frame: int
    outcome: str
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool
    own_pokemon_identities: tuple[PokemonIdentity, ...] = ()


@dataclass(frozen=True, slots=True)
class MapChanged:
    frame: int
    old_map: tuple[int, int] | None
    new_map: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class PartyChanged:
    frame: int
    entered_party_indices: tuple[int, ...]
    left_party_indices: tuple[int, ...]
    reordered: bool
    changed_party_indices: tuple[int, ...]
    entered_identities: tuple[PokemonIdentity, ...] = ()
    left_identities: tuple[PokemonIdentity, ...] = ()
    changed_identities: tuple[PokemonIdentity, ...] = ()


@dataclass(frozen=True, slots=True)
class PokemonFainted:
    frame: int
    party_index: int
    species: str
    nickname: str
    personality_value: int
    cause: str
    identity: PokemonIdentity | None = None


@dataclass(frozen=True, slots=True)
class PokemonStorageLocation:
    identity: PokemonIdentity
    box: int
    slot: int


@dataclass(frozen=True, slots=True)
class StorageChanged:
    frame: int
    entered: tuple[PokemonStorageLocation, ...]
    left: tuple[PokemonStorageLocation, ...]
    moved: tuple[tuple[PokemonStorageLocation, PokemonStorageLocation], ...] = ()


@dataclass(frozen=True, slots=True)
class WhiteoutOccurred:
    frame: int


@dataclass(frozen=True, slots=True)
class GameStateChanged:
    frame: int
    old_state: Any
    new_state: Any


Event = (
    BattleStarted
    | BattleEnded
    | MapChanged
    | PartyChanged
    | PokemonFainted
    | StorageChanged
    | WhiteoutOccurred
    | GameStateChanged
)


def _identity(pokemon: PartyPokemonSnapshot) -> tuple[Any, ...]:
    """Use stable identity, retaining a legacy fallback for old snapshots."""
    if pokemon.identity is not None:
        return pokemon.identity
    return (
        pokemon.personality_value,
        pokemon.original_trainer_id,
        pokemon.original_trainer_secret_id,
        pokemon.species,
        pokemon.nickname,
        pokemon.egg,
    )


def _party_state(pokemon: PartyPokemonSnapshot) -> tuple[Any, ...]:
    # HP and fainted are intentionally excluded: battle damage is not a party
    # composition change and must not produce an event every frame.
    return (
        _identity(pokemon),
        pokemon.party_index,
        pokemon.level,
        pokemon.max_hp,
        pokemon.status,
        pokemon.moves,
        pokemon.held_item,
    )


def _map(snapshot: NuzlockeSnapshot) -> tuple[int, int] | None:
    if snapshot.player.map_group is None or snapshot.player.map_number is None:
        return None
    return snapshot.player.map_group, snapshot.player.map_number


class NuzlockeEventObserver:
    """Convert a sequence of normalized snapshots into one-shot transitions."""

    def __init__(self) -> None:
        self._previous: NuzlockeSnapshot | None = None
        self._previous_battle: NuzlockeSnapshot | None = None
        self._previous_ready_battle: NuzlockeSnapshot | None = None
        self._previous_map: NuzlockeSnapshot | None = None
        self._previous_game_state: NuzlockeSnapshot | None = None
        self._previous_party: NuzlockeSnapshot | None = None
        self._fainted: set[tuple[Any, ...]] = set()

    def observe(self, snapshot: NuzlockeSnapshot) -> tuple[Event, ...]:
        previous = self._previous
        self._previous = snapshot
        if previous is None:
            self._remember_available(snapshot)
            return ()

        events: list[Event] = []
        previous_battle = self._previous_battle
        if (
            snapshot.battle_available
            and snapshot.battle is not None
            and snapshot.battle.ready
            and self._previous_ready_battle is None
            and (
                previous_battle is None
                or previous_battle.battle is None
                or not previous_battle.battle.ready
            )
        ):
            battle = snapshot.battle
            events.append(BattleStarted(snapshot.frame, battle.battle_type, battle.is_trainer, battle.is_wild, battle.is_double,
                                        tuple(p.identity for p in battle.own_active if p.identity is not None)))
        elif (
            snapshot.battle_available
            and snapshot.battle is None
            and self._previous_ready_battle is not None
        ):
            battle = self._previous_ready_battle.battle
            events.append(BattleEnded(snapshot.frame, battle.outcome, battle.battle_type, battle.is_trainer, battle.is_wild, battle.is_double,
                                      tuple(p.identity for p in battle.own_active if p.identity is not None)))

        previous_map = self._previous_map
        old_map, new_map = (_map(previous_map) if previous_map else None), _map(snapshot)
        if snapshot.player_available and previous_map is not None and old_map != new_map:
            events.append(MapChanged(snapshot.frame, old_map, new_map))

        previous_state = self._previous_game_state
        if snapshot.game_state_available and previous_state is not None and previous_state.game_state != snapshot.game_state:
            events.append(GameStateChanged(snapshot.frame, previous_state.game_state, snapshot.game_state))
        if snapshot.game_state_available and previous_state is not None and previous_state.game_state != snapshot.game_state and getattr(snapshot.game_state, "name", None) == "WHITEOUT":
            events.append(WhiteoutOccurred(snapshot.frame))

        if snapshot.party_available and self._previous_party is not None:
            events.extend(self._party_events(self._previous_party, snapshot))
        if snapshot.pc_available and previous is not None and previous.pc_available:
            events.extend(self._storage_events(previous, snapshot))
        self._remember_available(snapshot)
        return tuple(events)

    def _remember_available(self, snapshot: NuzlockeSnapshot) -> None:
        if snapshot.battle_available:
            self._previous_battle = snapshot
            if snapshot.battle is not None and snapshot.battle.ready:
                self._previous_ready_battle = snapshot
            elif snapshot.battle is None:
                self._previous_ready_battle = None
        if snapshot.player_available:
            self._previous_map = snapshot
        if snapshot.game_state_available:
            self._previous_game_state = snapshot
        if snapshot.party_available:
            self._previous_party = snapshot

    def _party_events(self, previous: NuzlockeSnapshot, snapshot: NuzlockeSnapshot) -> list[Event]:
        old = previous.party
        new = snapshot.party
        old_keys = tuple(_identity(p) for p in old)
        new_keys = tuple(_identity(p) for p in new)
        old_by_key = {key: p for key, p in zip(old_keys, old)}
        new_by_key = {key: p for key, p in zip(new_keys, new)}
        entered = tuple(p.party_index for key, p in zip(new_keys, new) if key not in old_by_key)
        left = tuple(p.party_index for key, p in zip(old_keys, old) if key not in new_by_key)
        changed = tuple(
            new_by_key[key].party_index
            for key in new_keys
            if key in old_by_key and _party_state(old_by_key[key]) != _party_state(new_by_key[key])
        )
        reordered = len(old_keys) == len(new_keys) and set(old_keys) == set(new_keys) and old_keys != new_keys

        if entered or left or changed or reordered:
            entered_identities = tuple(p.identity for key, p in zip(new_keys, new) if key not in old_by_key and p.identity is not None)
            left_identities = tuple(p.identity for key, p in zip(old_keys, old) if key not in new_by_key and p.identity is not None)
            changed_identities = tuple(new_by_key[key].identity for key in new_keys
                                       if key in old_by_key and new_by_key[key].identity is not None
                                       and _party_state(old_by_key[key]) != _party_state(new_by_key[key]))
            event = PartyChanged(snapshot.frame, entered, left, reordered, changed,
                                 entered_identities, left_identities, changed_identities)
        else:
            event = None

        faint_events: list[Event] = []
        self._fainted.intersection_update(new_keys)
        for pokemon in new:
            key = _identity(pokemon)
            old_pokemon = old_by_key.get(key)
            # A newly observed party member may already have 0 HP (for
            # example, while party data is being populated).  Without a
            # prior member snapshot that is not enough evidence of a faint.
            if pokemon.fainted and old_pokemon is not None and not old_pokemon.fainted and key not in self._fainted:
                cause = "battle" if snapshot.battle is not None or previous.battle is not None else "unknown"
                if snapshot.battle is None and getattr(snapshot.game_state, "name", None) == "OVERWORLD" and pokemon.status in {"poison", "bad_poison"}:
                    cause = "poison"
                faint_events.append(PokemonFainted(snapshot.frame, pokemon.party_index, pokemon.species, pokemon.nickname,
                                                   pokemon.personality_value, cause, pokemon.identity))
                self._fainted.add(key)
            elif not pokemon.fainted:
                self._fainted.discard(key)

        return ([event] if event is not None else []) + faint_events

    @staticmethod
    def _storage_events(previous: NuzlockeSnapshot, snapshot: NuzlockeSnapshot) -> list[Event]:
        def locations(value: NuzlockeSnapshot) -> dict[PokemonIdentity, PokemonStorageLocation]:
            return {
                item.pokemon.identity: PokemonStorageLocation(item.pokemon.identity, item.box, item.slot)
                for item in value.pc.pokemon
                if item.pokemon.identity is not None
            }

        old, new = locations(previous), locations(snapshot)
        entered = tuple(new[identity] for identity in new.keys() - old.keys())
        left = tuple(old[identity] for identity in old.keys() - new.keys())
        moved = tuple((old[identity], new[identity]) for identity in old.keys() & new.keys()
                      if old[identity] != new[identity])
        return [StorageChanged(snapshot.frame, entered, left, moved)] if entered or left or moved else []
