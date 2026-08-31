"""Immutable, point-in-time observations of state already exposed by pokebot.

This module is deliberately a read-only boundary.  It contains no Nuzlocke
legality, persistent identity, or irreversible-event recording; those belong
to later domain/event layers.  The upstream readers are frame-cache-backed,
so this is a point-in-time copy: it can reflect the reader cache's normal
within-frame freshness limitations, but it never adds another snapshot cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import TYPE_CHECKING

from .identity import PokemonIdentity

# ``get_pokemon_storage`` already reuses an unchanged raw-storage object, but
# the Nuzlocke snapshot used to materialize all readable PC slots again every
# frame.  Keep the derived immutable representation alongside that source
# object.  A new PokemonStorage object is produced when the raw block changes,
# so this cache naturally refreshes without weakening storage observations.
_last_storage_source: object | None = None
_last_storage_snapshot: "StorageSnapshot | None" = None

if TYPE_CHECKING:
    from modules.memory import GameState


@dataclass(frozen=True, slots=True)
class MoveSnapshot:
    """Immutable move data exposed by a Pokémon observation."""

    name: str
    pp: int
    total_pp: int
    pp_ups: int


@dataclass(frozen=True, slots=True)
class PokemonSnapshot:
    """Immutable identity, health, move, and ownership data for a Pokémon."""

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
    identity: PokemonIdentity | None = dataclass_field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class PartyPokemonSnapshot(PokemonSnapshot):
    """A Pokémon snapshot augmented with its current party slot."""

    party_index: int


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    """Immutable player position and overworld-control observation."""

    name: str | None
    map_group: int | None
    map_number: int | None
    map_name: str | None
    coordinates: tuple[int, int] | None
    facing: str | None
    controllable: bool


@dataclass(frozen=True, slots=True)
class ItemQuantity:
    """An observed item name and quantity from one inventory pocket."""

    name: str
    quantity: int


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    """Immutable view of the item pockets relevant to campaign policy."""

    items: tuple[ItemQuantity, ...]
    poke_balls: tuple[ItemQuantity, ...]
    key_items: tuple[ItemQuantity, ...]


@dataclass(frozen=True, slots=True)
class StoragePokemonSnapshot:
    """A normalized stored Pokémon together with its box and slot."""

    box: int
    slot: int
    pokemon: PokemonSnapshot


@dataclass(frozen=True, slots=True)
class StorageSnapshot:
    """Immutable view of readable Pokémon currently held in storage."""

    active_box: int
    pokemon: tuple[StoragePokemonSnapshot, ...]


@dataclass(frozen=True, slots=True)
class BattlePokemonSnapshot:
    """Immutable battle-side data for one active battler."""

    party_index: int
    species: str
    current_hp: int
    max_hp: int
    status: str
    # BattlePokemon does not expose a direct fainted flag.  Keep this
    # observational field nullable instead of inferring party state here.
    fainted: bool | None
    egg: bool
    moves: tuple[MoveSnapshot, ...]
    identity: PokemonIdentity | None = None


@dataclass(frozen=True, slots=True)
class BattleSnapshot:
    """Immutable observation of the current battle boundary and battlers."""

    battle_type: tuple[str, ...]
    is_trainer: bool
    is_wild: bool
    is_double: bool
    own_active: tuple[BattlePokemonSnapshot, ...]
    opponent_active: tuple[BattlePokemonSnapshot, ...]
    outcome: str
    ready: bool = True


@dataclass(frozen=True, slots=True)
class NamedFlag:
    """A named ROM event flag and its observed boolean value."""

    name: str
    value: bool


@dataclass(frozen=True, slots=True)
class NamedVariable:
    """A named ROM event variable and its observed integer value."""

    name: str
    value: int


class CampaignObservationLifecycle(Enum):
    """Lifecycle of the save-backed campaign observation.

    ``FRESH_START`` is a real emulator observation at the title/main menu:
    the ROM has not loaded a save or started a new game, so bytes exposed by
    save-block readers are not campaign facts.  It is deliberately distinct
    from ``UNAVAILABLE`` so campaign execution can still mount the opening
    capability without treating unavailable facts as false or true.
    """

    UNAVAILABLE = "unavailable"
    FRESH_START = "fresh_start"
    ACTIVE = "active"


# These ROM callbacks can occur before ``game_has_started()`` becomes true.
# Their save-block reads are therefore not campaign facts, but they are still
# part of the executable new-game flow and must keep the opening capability
# mounted.
_FRESH_START_GAME_STATE_NAMES = frozenset(
    {
        "TITLE_SCREEN",
        "MAIN_MENU",
        "OPTIONS_MENU",
        "NAMING_SCREEN",
        "CHOOSE_STARTER",
    }
)


@dataclass(frozen=True, slots=True)
class CampaignObservationSnapshot:
    """Raw, save-backed inputs used by the pure campaign-fact reducer."""

    flags: tuple[NamedFlag, ...] = ()
    variables: tuple[NamedVariable, ...] = ()
    text_speed: int | None = None
    available: bool = False
    lifecycle: CampaignObservationLifecycle = CampaignObservationLifecycle.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class ProgressionSnapshot:
    """Observed badge flags used by campaign progression."""

    badges: tuple[NamedFlag, ...]


@dataclass(frozen=True, slots=True)
class NuzlockeSnapshot:
    """Complete immutable observation passed to Nuzlocke reducers."""

    frame: int
    game_id: str | None
    game_state: "GameState | None"
    player: PlayerSnapshot
    party: tuple[PartyPokemonSnapshot, ...]
    inventory: InventorySnapshot
    battle: BattleSnapshot | None
    pc: StorageSnapshot
    progression: ProgressionSnapshot
    # Collection fields retain their normal empty-tuple representation for
    # compatibility, while these flags distinguish empty data from data that
    # could not be read in the current game lifecycle state.
    game_state_available: bool = True
    player_available: bool = True
    party_available: bool = True
    inventory_available: bool = True
    pc_available: bool = True
    battle_available: bool = True
    campaign_observation: CampaignObservationSnapshot = dataclass_field(default_factory=CampaignObservationSnapshot)


def _moves(pokemon) -> tuple[MoveSnapshot, ...]:
    """Normalize the readable move slots of a runtime Pokémon object."""

    return tuple(
        MoveSnapshot(move.move.name, move.pp, move.total_pp, move.pp_ups) for move in pokemon.moves if move is not None
    )


def _pokemon(pokemon) -> PokemonSnapshot:
    """Normalize a party or storage Pokémon without mutating the source."""

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
        identity=PokemonIdentity.from_pokemon(pokemon),
    )


def _storage_pokemon_is_readable(pokemon) -> bool:
    """Return whether a storage Pokémon can safely be normalized.

    PokemonStorage normally applies parse_pokemon() before exposing a slot.
    Keep this narrow guard at the passive snapshot boundary as storage data can
    still be observed while the emulator is updating it.
    """
    try:
        return not pokemon.is_empty and pokemon.is_valid
    except (IndexError, TypeError, ValueError):
        return False


def _battle_pokemon(pokemon) -> BattlePokemonSnapshot:
    """Normalize one runtime battle battler into an immutable snapshot."""

    return BattlePokemonSnapshot(
        party_index=pokemon.party_index,
        species=pokemon.species.name,
        current_hp=pokemon.current_hp,
        max_hp=pokemon.total_hp,
        # BattlePokemon exposes its permanent battle status, but not a direct
        # fainted flag.  Keep fainted unknown rather than inferring it from a
        # different Pokémon model or from a transitional battle frame.
        status=pokemon.status_permanent.value,
        fainted=None,
        egg=pokemon.is_egg,
        moves=_moves(pokemon),
        identity=PokemonIdentity.from_battle_pokemon(pokemon),
    )


def _items(slots) -> tuple[ItemQuantity, ...]:
    """Normalize item slots while preserving their observed quantities."""

    return tuple(ItemQuantity(slot.item.name, slot.quantity) for slot in slots)


def _player() -> tuple[PlayerSnapshot, bool]:
    """Read the player projection and whether its avatar is available."""

    from modules.player import (
        get_player,
        get_player_avatar,
        get_player_location,
        player_avatar_is_controllable,
    )

    player = get_player()
    avatar = get_player_avatar()
    if avatar is None:
        return (
            PlayerSnapshot(player.name if player else None, None, None, None, None, None, False),
            False,
        )
    location, coordinates = get_player_location()
    group, number = avatar.map_group_and_number
    return (
        PlayerSnapshot(
            name=player.name if player else None,
            map_group=group,
            map_number=number,
            map_name=getattr(location, "name", None),
            coordinates=tuple(coordinates),
            facing=avatar.facing_direction,
            controllable=player_avatar_is_controllable(),
        ),
        True,
    )


def _battle(game_state: GameState) -> BattleSnapshot | None:
    """Read battle state only while the runtime reports a battle lifecycle."""

    if getattr(game_state, "name", None) not in {
        "BATTLE",
        "BATTLE_STARTING",
        "BATTLE_ENDING",
    }:
        return None
    from modules.battle_state import (
        BattleState,
        BattleType,
        get_battle_state,
        get_last_battle_outcome,
    )
    from modules.memory import GameState

    if game_state not in (
        GameState.BATTLE,
        GameState.BATTLE_STARTING,
        GameState.BATTLE_ENDING,
    ):
        return None
    state: BattleState | None = get_battle_state()
    if state is None:
        return BattleSnapshot((), False, False, False, (), (), "Unknown", ready=False)
    battle_type = state.type
    try:
        outcome = get_last_battle_outcome().name
    except (RuntimeError, ValueError):
        outcome = "Unknown"
    battling_pokemon = state.battling_pokemon
    battle_ready = len(battling_pokemon) >= 2
    own_active = tuple(_battle_pokemon(p) for p in state.own_side.active_battlers) if battle_ready else ()
    opponent_active = tuple(_battle_pokemon(p) for p in state.opponent.active_battlers) if battle_ready else ()
    battle_ready = battle_ready and bool(own_active) and bool(opponent_active)
    return BattleSnapshot(
        battle_type=tuple(flag.name for flag in BattleType if flag in battle_type),
        is_trainer=state.is_trainer_battle,
        is_wild=not state.is_trainer_battle,
        is_double=state.is_double_battle,
        own_active=own_active,
        opponent_active=opponent_active,
        outcome=outcome,
        ready=battle_ready,
    )


def _storage_snapshot(storage) -> StorageSnapshot:
    """Build or reuse the normalized snapshot for the current storage object."""

    global _last_storage_source, _last_storage_snapshot

    from modules.context import context
    from modules.profiler import count as profile_count

    if storage is _last_storage_source and _last_storage_snapshot is not None:
        profile_count("nuzlocke_storage_snapshot_cache_hits")
        trace = getattr(context, "stutter_trace", None)
        if trace is not None:
            trace.mark("storage_snapshot_cache_hit", True)
        return _last_storage_snapshot

    trace = getattr(context, "stutter_trace", None)
    started = trace.now() if trace is not None else 0
    profile_count("nuzlocke_storage_snapshot_builds")
    readable_slots = 0
    boxes = []
    for box in (storage.boxes if storage is not None else ()):
        slots = []
        for slot in box.slots:
            if not _storage_pokemon_is_readable(slot.pokemon):
                continue
            readable_slots += 1
            slots.append(StoragePokemonSnapshot(box.number, slot.slot_index, _pokemon(slot.pokemon)))
        boxes.extend(slots)
    profile_count("nuzlocke_storage_snapshot_slots", readable_slots)
    result = StorageSnapshot(
        active_box=storage.active_box_index if storage is not None else 0,
        pokemon=tuple(boxes),
    )
    _last_storage_source = storage
    _last_storage_snapshot = result
    if trace is not None:
        trace.duration("storage_snapshot_duration_ms", started)
    return result


def get_nuzlocke_snapshot() -> NuzlockeSnapshot:
    """Read current observable state without performing any emulator action."""
    from modules.context import context
    from modules.items import get_item_bag
    from modules.memory import game_has_started, get_event_flag, get_game_state
    from modules.pokemon_party import get_party
    from modules.pokemon_storage import get_pokemon_storage

    trace = getattr(context, "stutter_trace", None)
    stage = trace.now() if trace is not None else 0
    game_state = get_game_state()
    if trace is not None:
        trace.duration("nuzlocke_game_state_duration_ms", stage)
    stage = trace.now() if trace is not None else 0
    party = get_party()
    if trace is not None:
        trace.duration("nuzlocke_party_duration_ms", stage)
    stage = trace.now() if trace is not None else 0
    bag = get_item_bag()
    if trace is not None:
        trace.duration("nuzlocke_bag_duration_ms", stage)
    stage = trace.now() if trace is not None else 0
    storage = get_pokemon_storage()
    if trace is not None:
        trace.duration("nuzlocke_storage_access_duration_ms", stage)
    stage = trace.now() if trace is not None else 0
    player, player_available = _player()
    if trace is not None:
        trace.duration("nuzlocke_player_duration_ms", stage)
    stage = trace.now() if trace is not None else 0
    battle = _battle(game_state)
    if trace is not None:
        trace.duration("nuzlocke_battle_duration_ms", stage)
    stage = trace.now() if trace is not None else 0
    progression = tuple(NamedFlag(f"BADGE{i:02d}_GET", get_event_flag(f"BADGE{i:02d}_GET")) for i in range(1, 9))
    campaign_observation = CampaignObservationSnapshot()
    try:
        from modules.memory import get_event_var, get_save_block, unpack_uint16

        campaign_flags = tuple(
            NamedFlag(name, get_event_flag(name))
            for name in (
                "SET_WALL_CLOCK",
                "RESCUED_BIRCH",
                "DEFEATED_RIVAL_ROUTE103",
                "HIDE_ROUTE_103_RIVAL",
                "SYS_POKEMON_GET",
                "SYS_POKEDEX_GET",
                "RECEIVED_POKEDEX_FROM_BIRCH",
                "VISITED_PETALBURG_CITY",
                "DEVON_GOODS_STOLEN",
                "RECOVERED_DEVON_GOODS",
                "RETURNED_DEVON_GOODS",
                "DELIVERED_DEVON_GOODS",
                "INTERACTED_WITH_DEVON_EMPLOYEE_GOODS_STOLEN",
                "VISITED_RUSTBORO_CITY",
                "DEFEATED_RUSTBORO_GYM",
            )
        )
        campaign_variables = tuple(
            NamedVariable(name, get_event_var(name))
            for name in (
                "LITTLEROOT_INTRO_STATE",
                "LITTLEROOT_RIVAL_STATE",
                "BIRCH_LAB_STATE",
                "PETALBURG_CITY_STATE",
                "PETALBURG_GYM_STATE",
                "PETALBURG_WOODS_STATE",
                "RUSTBORO_CITY_STATE",
                "RUSTURF_TUNNEL_STATE",
                "DEVON_CORP_3F_STATE",
            )
        )
        text_speed = unpack_uint16(get_save_block(2, offset=0x14, size=2)) & 0x07
        # ``get_game_state()`` is also non-None on the title and main-menu
        # callbacks.  Those callbacks can expose an uninitialised save block
        # (typically all 0xff after mGBA opens an empty save file), which
        # would make every raw flag appear true.  Only treat save-backed
        # campaign values as observations after the ROM reports that a save
        # has been loaded or a new game has actually started.
        save_backed = game_has_started()
        if save_backed:
            lifecycle = CampaignObservationLifecycle.ACTIVE
        elif getattr(game_state, "name", None) in _FRESH_START_GAME_STATE_NAMES:
            # mGBA exposes an all-0xff placeholder save after opening a
            # profile with no game save.  Keep the title/menu usable by the
            # opening executor, but never expose those bytes as facts.
            lifecycle = CampaignObservationLifecycle.FRESH_START
        else:
            lifecycle = CampaignObservationLifecycle.UNAVAILABLE
        campaign_observation = CampaignObservationSnapshot(
            campaign_flags,
            campaign_variables,
            text_speed,
            save_backed,
            lifecycle,
        )
    except (AttributeError, ImportError, KeyError, IndexError, RuntimeError, TypeError, ValueError):
        pass
    if trace is not None:
        trace.duration("nuzlocke_progression_duration_ms", stage)
    return NuzlockeSnapshot(
        frame=context.emulator.get_frame_count(),
        game_id=getattr(context.rom, "game_name", None),
        game_state=game_state,
        player=player,
        party=tuple(
            PartyPokemonSnapshot(
                **{field: getattr(snapshot, field) for field in PokemonSnapshot.__dataclass_fields__},
                party_index=p.index,
            )
            for p in party or ()
            for snapshot in (_pokemon(p),)
        ),
        inventory=InventorySnapshot(
            _items(bag.items) if bag is not None else (),
            _items(bag.poke_balls) if bag is not None else (),
            _items(bag.key_items) if bag is not None else (),
        ),
        battle=battle,
        pc=_storage_snapshot(storage),
        progression=ProgressionSnapshot(progression),
        game_state_available=game_state is not None,
        player_available=player_available,
        party_available=party is not None,
        inventory_available=bag is not None,
        pc_available=storage is not None,
        battle_available=game_state is not None,
        campaign_observation=campaign_observation,
    )
