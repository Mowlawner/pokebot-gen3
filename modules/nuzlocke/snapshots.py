"""Immutable, point-in-time observations of state already exposed by pokebot.

This module is deliberately a read-only boundary.  It contains no Nuzlocke
legality, persistent identity, or irreversible-event recording; those belong
to later domain/event layers.  The upstream readers are frame-cache-backed,
so this is a point-in-time copy: it can reflect the reader cache's normal
within-frame freshness limitations, but it never adds another snapshot cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.memory import GameState


@dataclass(frozen=True, slots=True)
class MoveSnapshot:
    name: str
    pp: int
    total_pp: int
    pp_ups: int


@dataclass(frozen=True, slots=True)
class PokemonSnapshot:
    species: str
    nickname: str
    level: int
    current_hp: int
    max_hp: int
    status: str
    personality_value: int
    original_trainer_id: int
    original_trainer_secret_id: int
    original_trainer_name: str
    moves: tuple[MoveSnapshot, ...]
    held_item: str | None
    fainted: bool
    egg: bool


@dataclass(frozen=True, slots=True)
class PartyPokemonSnapshot(PokemonSnapshot):
    party_index: int


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    name: str | None
    map_group: int | None
    map_number: int | None
    map_name: str | None
    coordinates: tuple[int, int] | None
    facing: str | None
    controllable: bool


@dataclass(frozen=True, slots=True)
class ItemQuantity:
    name: str
    quantity: int


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    items: tuple[ItemQuantity, ...]
    poke_balls: tuple[ItemQuantity, ...]
    key_items: tuple[ItemQuantity, ...]


@dataclass(frozen=True, slots=True)
class StoragePokemonSnapshot:
    box: int
    slot: int
    pokemon: PokemonSnapshot


@dataclass(frozen=True, slots=True)
class StorageSnapshot:
    active_box: int
    pokemon: tuple[StoragePokemonSnapshot, ...]


@dataclass(frozen=True, slots=True)
class BattlePokemonSnapshot:
    party_index: int
    species: str
    current_hp: int
    max_hp: int
    status: str
    fainted: bool
    egg: bool
    moves: tuple[MoveSnapshot, ...]


@dataclass(frozen=True, slots=True)
class BattleSnapshot:
    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool
    own_active: tuple[BattlePokemonSnapshot, ...]
    opponent_active: tuple[BattlePokemonSnapshot, ...]
    outcome: str


@dataclass(frozen=True, slots=True)
class NamedFlag:
    name: str
    value: bool


@dataclass(frozen=True, slots=True)
class ProgressionSnapshot:
    badges: tuple[NamedFlag, ...]


@dataclass(frozen=True, slots=True)
class NuzlockeSnapshot:
    frame: int
    game_id: str | None
    game_state: "GameState"
    player: PlayerSnapshot
    party: tuple[PartyPokemonSnapshot, ...]
    inventory: InventorySnapshot
    battle: BattleSnapshot | None
    pc: StorageSnapshot
    progression: ProgressionSnapshot


def _moves(pokemon) -> tuple[MoveSnapshot, ...]:
    return tuple(
        MoveSnapshot(move.move.name, move.pp, move.total_pp, move.pp_ups)
        for move in pokemon.moves
        if move is not None
    )


def _pokemon(pokemon) -> PokemonSnapshot:
    trainer = pokemon.original_trainer
    return PokemonSnapshot(
        species=pokemon.species.name,
        nickname=pokemon.nickname,
        level=pokemon.level,
        current_hp=pokemon.current_hp,
        max_hp=pokemon.total_hp,
        status=pokemon.status_condition.value,
        personality_value=pokemon.personality_value,
        original_trainer_id=trainer.id,
        original_trainer_secret_id=trainer.secret_id,
        original_trainer_name=trainer.name,
        moves=_moves(pokemon),
        held_item=pokemon.held_item.name if pokemon.held_item is not None else None,
        fainted=pokemon.current_hp == 0,
        egg=pokemon.is_egg,
    )


def _battle_pokemon(pokemon) -> BattlePokemonSnapshot:
    return BattlePokemonSnapshot(
        party_index=pokemon.party_index,
        species=pokemon.species.name,
        current_hp=pokemon.current_hp,
        max_hp=pokemon.total_hp,
        # BattlePokemon exposes battle HP and moves, but not the permanent
        # party status property.  The battle-state model intentionally keeps
        # that distinction, so do not infer it here.
        status="unknown",
        fainted=pokemon.is_fainted,
        egg=pokemon.is_egg,
        moves=_moves(pokemon),
    )


def _items(slots) -> tuple[ItemQuantity, ...]:
    return tuple(ItemQuantity(slot.item.name, slot.quantity) for slot in slots)


def _player() -> PlayerSnapshot:
    from modules.player import get_player, get_player_avatar, get_player_location, player_avatar_is_controllable

    player = get_player()
    avatar = get_player_avatar()
    if avatar is None:
        return PlayerSnapshot(player.name if player else None, None, None, None, None, None, False)
    location, coordinates = get_player_location()
    group, number = avatar.map_group_and_number
    return PlayerSnapshot(
        name=player.name if player else None,
        map_group=group,
        map_number=number,
        map_name=getattr(location, "name", None),
        coordinates=tuple(coordinates),
        facing=avatar.facing_direction,
        controllable=player_avatar_is_controllable(),
    )


def _battle(game_state: GameState) -> BattleSnapshot | None:
    if getattr(game_state, "name", None) not in {"BATTLE", "BATTLE_STARTING", "BATTLE_ENDING"}:
        return None
    from modules.battle_state import BattleState, BattleType, get_battle_state, get_last_battle_outcome
    from modules.memory import GameState

    if game_state not in (GameState.BATTLE, GameState.BATTLE_STARTING, GameState.BATTLE_ENDING):
        return None
    state: BattleState = get_battle_state()
    battle_type = state.type
    try:
        outcome = get_last_battle_outcome().name
    except (RuntimeError, ValueError):
        outcome = "Unknown"
    return BattleSnapshot(
        battle_type=tuple(flag.name for flag in BattleType if flag in battle_type),
        is_trainer=state.is_trainer_battle,
        is_wild=not state.is_trainer_battle,
        is_double=state.is_double_battle,
        own_active=tuple(_battle_pokemon(p) for p in state.own_side.active_battlers),
        opponent_active=tuple(_battle_pokemon(p) for p in state.opponent.active_battlers),
        outcome=outcome,
    )


def get_nuzlocke_snapshot() -> NuzlockeSnapshot:
    """Read current observable state without performing any emulator action."""
    from modules.context import context
    from modules.items import get_item_bag
    from modules.memory import get_event_flag, get_game_state
    from modules.pokemon_party import get_party
    from modules.pokemon_storage import get_pokemon_storage

    game_state = get_game_state()
    party = get_party()
    bag = get_item_bag()
    storage = get_pokemon_storage()
    return NuzlockeSnapshot(
        frame=context.emulator.get_frame_count(),
        game_id=getattr(context.rom, "game_name", None),
        game_state=game_state,
        player=_player(),
        party=tuple(
            PartyPokemonSnapshot(
                **{field: getattr(snapshot, field) for field in PokemonSnapshot.__dataclass_fields__},
                party_index=p.index,
            )
            for p in party
            for snapshot in (_pokemon(p),)
        ),
        inventory=InventorySnapshot(_items(bag.items), _items(bag.poke_balls), _items(bag.key_items)),
        battle=_battle(game_state),
        pc=StorageSnapshot(
            active_box=storage.active_box_index,
            pokemon=tuple(
                StoragePokemonSnapshot(box.number, slot.slot_index, _pokemon(slot.pokemon))
                for box in storage.boxes
                for slot in box.slots
            ),
        ),
        progression=ProgressionSnapshot(
            tuple(NamedFlag(f"BADGE{i:02d}_GET", get_event_flag(f"BADGE{i:02d}_GET")) for i in range(1, 9))
        ),
    )
