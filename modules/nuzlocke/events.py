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


@dataclass(frozen=True, slots=True)
class BattleStarted:
    frame: int
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool


@dataclass(frozen=True, slots=True)
class BattleEnded:
    frame: int
    outcome: str
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool


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


@dataclass(frozen=True, slots=True)
class PokemonFainted:
    frame: int
    party_index: int
    species: str
    nickname: str
    personality_value: int
    cause: str


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
    | WhiteoutOccurred
    | GameStateChanged
)


def _identity(pokemon: PartyPokemonSnapshot) -> tuple[Any, ...]:
    """Best available current-game identity; this is not a persistent ID."""
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
        self._fainted: set[tuple[Any, ...]] = set()

    def observe(self, snapshot: NuzlockeSnapshot) -> tuple[Event, ...]:
        previous = self._previous
        self._previous = snapshot
        if previous is None:
            return ()

        events: list[Event] = []
        if previous.battle is None and snapshot.battle is not None:
            battle = snapshot.battle
            events.append(BattleStarted(snapshot.frame, battle.battle_type, battle.is_trainer, battle.is_wild, battle.is_double))
        elif previous.battle is not None and snapshot.battle is None:
            battle = previous.battle
            events.append(BattleEnded(snapshot.frame, battle.outcome, battle.battle_type, battle.is_trainer, battle.is_wild, battle.is_double))

        old_map, new_map = _map(previous), _map(snapshot)
        if old_map != new_map:
            events.append(MapChanged(snapshot.frame, old_map, new_map))

        if previous.game_state != snapshot.game_state:
            events.append(GameStateChanged(snapshot.frame, previous.game_state, snapshot.game_state))
        if previous.game_state != snapshot.game_state and getattr(snapshot.game_state, "name", None) == "WHITEOUT":
            events.append(WhiteoutOccurred(snapshot.frame))

        events.extend(self._party_events(previous, snapshot))
        return tuple(events)

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
            event = PartyChanged(snapshot.frame, entered, left, reordered, changed)
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
                faint_events.append(PokemonFainted(snapshot.frame, pokemon.party_index, pokemon.species, pokemon.nickname, pokemon.personality_value, cause))
                self._fainted.add(key)
            elif not pokemon.fainted:
                self._fainted.discard(key)

        return ([event] if event is not None else []) + faint_events
