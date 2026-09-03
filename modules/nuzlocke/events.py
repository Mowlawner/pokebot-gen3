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
    """Observation of a battle becoming fully identifiable in the ROM."""

    frame: int
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool
    own_pokemon_identities: tuple[PokemonIdentity, ...] = ()
    opponent_pokemon_identities: tuple[PokemonIdentity, ...] = ()
    location: tuple[int, int] | None = None
    # ``None`` preserves compatibility with events written before the
    # runtime began recording the campaign eligibility boundary.  New live
    # events carry the decision made from the same frame's campaign facts so
    # rules replay does not have to reconstruct emulator state.
    encounter_eligible: bool | None = None
    # Species is kept separately from stable Pokémon identity because a wild
    # battler's identity is not necessarily the identity of the Pokémon that
    # is eventually added to the party.
    opponent_species: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BattleEnded:
    """Observation of a battle leaving the active battle lifecycle."""

    frame: int
    outcome: str
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool
    own_pokemon_identities: tuple[PokemonIdentity, ...] = ()
    opponent_pokemon_identities: tuple[PokemonIdentity, ...] = ()
    location: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class PokemonCaptured:
    """Generic observation that a Pokémon was captured.

    Capture detection is game/application specific.  Producers must emit this
    only when they have a reliable capture signal; the rules layer never
    guesses from party or PC movement.
    """

    frame: int
    identity: PokemonIdentity
    location: tuple[int, int] | None = None
    species: str | None = None


@dataclass(frozen=True, slots=True)
class MapChanged:
    """Observation of a player map transition."""

    frame: int
    old_map: tuple[int, int] | None
    new_map: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class PartyChanged:
    """Observation of party membership, order, or relevant member changes."""

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
    """Observation of a party Pokémon transitioning to zero HP."""

    frame: int
    party_index: int
    species: str
    nickname: str
    personality_value: int
    cause: str
    identity: PokemonIdentity | None = None


@dataclass(frozen=True, slots=True)
class PokemonStorageLocation:
    """Observed box and slot for a stable Pokémon identity."""

    identity: PokemonIdentity
    box: int
    slot: int


@dataclass(frozen=True, slots=True)
class StorageChanged:
    """Observation of Pokémon entering, leaving, or moving within storage."""

    frame: int
    entered: tuple[PokemonStorageLocation, ...]
    left: tuple[PokemonStorageLocation, ...]
    moved: tuple[tuple[PokemonStorageLocation, PokemonStorageLocation], ...] = ()


@dataclass(frozen=True, slots=True)
class WhiteoutOccurred:
    """Observation that the ROM entered its whiteout game state."""

    frame: int


@dataclass(frozen=True, slots=True)
class GameStateChanged:
    """Observation of a change in the emulator's game-state enum."""

    frame: int
    old_state: Any
    new_state: Any


@dataclass(frozen=True, slots=True)
class NuzlockeStarted:
    """Nuzbot-owned campaign boundary; this is not an emulator observation."""

    frame: int


Event = (
    BattleStarted
    | BattleEnded
    | PokemonCaptured
    | MapChanged
    | PartyChanged
    | PokemonFainted
    | StorageChanged
    | WhiteoutOccurred
    | GameStateChanged
    | NuzlockeStarted
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
    """Return party fields whose changes warrant a PartyChanged event."""

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
    """Return a snapshot's map identity when both map components are known."""

    if snapshot.player.map_group is None or snapshot.player.map_number is None:
        return None
    return snapshot.player.map_group, snapshot.player.map_number


class NuzlockeEventObserver:
    """Convert a sequence of normalized snapshots into one-shot transitions."""

    def __init__(self) -> None:
        """Create an observer with no prior snapshot baseline."""

        self._previous: NuzlockeSnapshot | None = None
        self._previous_battle: NuzlockeSnapshot | None = None
        self._previous_ready_battle: NuzlockeSnapshot | None = None
        self._previous_map: NuzlockeSnapshot | None = None
        self._previous_game_state: NuzlockeSnapshot | None = None
        self._previous_party: NuzlockeSnapshot | None = None
        self._fainted: set[tuple[Any, ...]] = set()
        self._storage_location_cache: dict[int, tuple[object, dict[PokemonIdentity, PokemonStorageLocation]]] = {}

    @staticmethod
    def _battle_is_ready(snapshot: NuzlockeSnapshot) -> bool:
        """Return whether the battle boundary has usable ROM identity data.

        During Emerald's ``BATTLE_STARTING`` transition, battler buffers can
        already contain enough data for ``BattleSnapshot.ready`` to be true
        while the battle type is still zeroed.  Such a frame is a baseline,
        not a legal encounter boundary; accepting it would claim a location
        with stale opponent identity data.
        """
        return bool(
            snapshot.battle_available
            and snapshot.battle is not None
            and snapshot.battle.ready
            and snapshot.battle.battle_type
        )

    def observe(self, snapshot: NuzlockeSnapshot) -> tuple[Event, ...]:
        """Compare one snapshot with the prior baseline and emit transitions."""

        previous = self._previous
        self._previous = snapshot
        if previous is None:
            self._remember_available(snapshot)
            # A process can be restored from a save-state while the ROM is
            # already in a fully materialized battle.  Treat that first
            # complete battle observation as the start boundary; otherwise
            # the observer can only emit BattleEnded and the campaign rules
            # never get a chance to claim a legal encounter.  An incomplete
            # BATTLE_STARTING snapshot still remains a baseline and will be
            # promoted by the normal ready transition below.
            if self._battle_is_ready(snapshot):
                battle = snapshot.battle
                return (
                    BattleStarted(
                        snapshot.frame,
                        battle.battle_type,
                        battle.is_trainer,
                        battle.is_wild,
                        battle.is_double,
                        tuple(p.identity for p in battle.own_active if p.identity is not None),
                        tuple(p.identity for p in battle.opponent_active if p.identity is not None),
                        _map(snapshot),
                        opponent_species=tuple(p.species for p in battle.opponent_active),
                    ),
                )
            return ()

        events: list[Event] = []
        previous_battle = self._previous_battle
        if (
            self._battle_is_ready(snapshot)
            and self._previous_ready_battle is None
            and (
                previous_battle is None or previous_battle.battle is None or not self._battle_is_ready(previous_battle)
            )
        ):
            battle = snapshot.battle
            events.append(
                BattleStarted(
                    snapshot.frame,
                    battle.battle_type,
                    battle.is_trainer,
                    battle.is_wild,
                    battle.is_double,
                    tuple(p.identity for p in battle.own_active if p.identity is not None),
                    tuple(p.identity for p in battle.opponent_active if p.identity is not None),
                    _map(snapshot),
                    opponent_species=tuple(p.species for p in battle.opponent_active),
                )
            )
        elif snapshot.battle_available and snapshot.battle is None and self._previous_ready_battle is not None:
            battle = self._previous_ready_battle.battle
            if (
                battle is not None
                and battle.is_wild
                and not battle.is_trainer
                and battle.outcome.lower() in {"caught", "captured"}
                and len(battle.opponent_active) == 1
                and battle.opponent_active[0].identity is not None
            ):
                events.append(
                    PokemonCaptured(
                        snapshot.frame,
                        battle.opponent_active[0].identity,
                        _map(self._previous_ready_battle),
                        battle.opponent_active[0].species,
                    )
                )
            events.append(
                BattleEnded(
                    snapshot.frame,
                    battle.outcome,
                    battle.battle_type,
                    battle.is_trainer,
                    battle.is_wild,
                    battle.is_double,
                    tuple(p.identity for p in battle.own_active if p.identity is not None),
                    tuple(p.identity for p in battle.opponent_active if p.identity is not None),
                    _map(self._previous_ready_battle),
                )
            )

        previous_map = self._previous_map
        old_map, new_map = (
            (_map(previous_map) if previous_map else None),
            _map(snapshot),
        )
        if snapshot.player_available and previous_map is not None and old_map != new_map:
            events.append(MapChanged(snapshot.frame, old_map, new_map))

        previous_state = self._previous_game_state
        if (
            snapshot.game_state_available
            and previous_state is not None
            and previous_state.game_state != snapshot.game_state
        ):
            events.append(GameStateChanged(snapshot.frame, previous_state.game_state, snapshot.game_state))
        if (
            snapshot.game_state_available
            and previous_state is not None
            and previous_state.game_state != snapshot.game_state
            and getattr(snapshot.game_state, "name", None) == "WHITEOUT"
        ):
            events.append(WhiteoutOccurred(snapshot.frame))

        if snapshot.party_available and self._previous_party is not None:
            events.extend(self._party_events(self._previous_party, snapshot))
        if snapshot.pc_available and previous is not None and previous.pc_available:
            events.extend(self._storage_events(previous, snapshot))
        self._remember_available(snapshot)
        return tuple(events)

    def _remember_available(self, snapshot: NuzlockeSnapshot) -> None:
        """Update only the observation baselines available in this snapshot."""

        if snapshot.battle_available:
            self._previous_battle = snapshot
            if self._battle_is_ready(snapshot):
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
        """Derive party composition and faint transitions between snapshots."""

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
            entered_identities = tuple(
                p.identity for key, p in zip(new_keys, new) if key not in old_by_key and p.identity is not None
            )
            left_identities = tuple(
                p.identity for key, p in zip(old_keys, old) if key not in new_by_key and p.identity is not None
            )
            changed_identities = tuple(
                new_by_key[key].identity
                for key in new_keys
                if key in old_by_key
                and new_by_key[key].identity is not None
                and _party_state(old_by_key[key]) != _party_state(new_by_key[key])
            )
            event = PartyChanged(
                snapshot.frame,
                entered,
                left,
                reordered,
                changed,
                entered_identities,
                left_identities,
                changed_identities,
            )
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
                if (
                    snapshot.battle is None
                    and getattr(snapshot.game_state, "name", None) == "OVERWORLD"
                    and pokemon.status in {"poison", "bad_poison"}
                ):
                    cause = "poison"
                faint_events.append(
                    PokemonFainted(
                        snapshot.frame,
                        pokemon.party_index,
                        pokemon.species,
                        pokemon.nickname,
                        pokemon.personality_value,
                        cause,
                        pokemon.identity,
                    )
                )
                self._fainted.add(key)
            elif not pokemon.fainted:
                self._fainted.discard(key)

        return ([event] if event is not None else []) + faint_events

    def _storage_locations(self, value: NuzlockeSnapshot) -> dict[PokemonIdentity, PokemonStorageLocation]:
        """Index readable storage identities, reusing the snapshot-local cache."""

        source = value.pc
        key = id(source)
        cached = self._storage_location_cache.get(key)
        if cached is not None and cached[0] is source:
            return cached[1]
        from modules.context import context

        trace = getattr(context, "stutter_trace", None)
        started = trace.now() if trace is not None else 0
        locations = {
            item.pokemon.identity: PokemonStorageLocation(item.pokemon.identity, item.box, item.slot)
            for item in source.pokemon
            if item.pokemon.identity is not None
        }
        if trace is not None:
            trace.duration("nuzlocke_storage_event_index_duration_ms", started)
        self._storage_location_cache[key] = (source, locations)
        return locations

    def _storage_events(self, previous: NuzlockeSnapshot, snapshot: NuzlockeSnapshot) -> list[Event]:
        """Derive storage entry, exit, and movement transitions."""

        old, new = self._storage_locations(previous), self._storage_locations(snapshot)
        entered = tuple(new[identity] for identity in new.keys() - old.keys())
        left = tuple(old[identity] for identity in old.keys() - new.keys())
        moved = tuple(
            (old[identity], new[identity]) for identity in old.keys() & new.keys() if old[identity] != new[identity]
        )
        return [StorageChanged(snapshot.frame, entered, left, moved)] if entered or left or moved else []
