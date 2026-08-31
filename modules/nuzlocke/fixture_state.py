"""Read-only bridge from a fixture's persisted events to CampaignState."""

from pathlib import Path

from . import snapshots as snapshot_module
from .campaign_state import CampaignState
from .persistence import JsonEventStore
from .projection import load_campaign_projection
from .rules import load_rules
from .snapshots import (
    CampaignObservationLifecycle,
    CampaignObservationSnapshot,
    InventorySnapshot,
    ItemQuantity,
    NamedFlag,
    NamedVariable,
    NuzlockeSnapshot,
    PartyPokemonSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    StoragePokemonSnapshot,
    StorageSnapshot,
)


class FixtureStateUnavailable(RuntimeError):
    """The fixture lacks an offline source for a required observation."""


def load_offline_normalized_snapshot(profile: str | Path) -> NuzlockeSnapshot:
    """Build the existing normalized snapshot from a raw, validated save.

    This is deliberately a fixture adapter, not a second save parser.  The
    existing ``SaveData`` parser remains responsible for section validation and
    all game-data decoding.  Values that only exist in emulator memory remain
    unavailable in the returned snapshot.
    """
    from modules.context import context
    from modules.game import set_rom
    from modules.profiles import load_profile
    from modules.save_data import get_save_data
    from modules.memory import unpack_uint16

    profile_path = Path(profile)
    profile_object = load_profile(profile_path)
    save_path = profile_path / "current_save.sav"
    if not save_path.is_file():
        raise FixtureStateUnavailable(f"offline save is missing: {save_path}")

    previous_profile = context.profile
    previous_emulator = context.emulator
    # get_save_data() is the existing validation/parser boundary.  Its
    # emulator dependency is limited to read_save_data(), so this adapter
    # supplies immutable file bytes without starting an emulator.
    context.profile = profile_object
    set_rom(profile_object.rom)
    raw_save = save_path.read_bytes()
    read_only_source = type("ReadOnlySaveSource", (), {"read_save_data": lambda self: raw_save})()
    context.emulator = read_only_source
    save_data = get_save_data()

    if save_data is None:
        context.profile = previous_profile
        context.emulator = previous_emulator
        raise FixtureStateUnavailable(f"save is not a valid Emerald save: {save_path}")

    player = save_data.get_player()
    party = tuple(
        PartyPokemonSnapshot(
            **{field: getattr(snapshot, field) for field in snapshot_module.PokemonSnapshot.__dataclass_fields__},
            party_index=pokemon.index,
        )
        for pokemon in save_data.get_party()
        for snapshot in (snapshot_module._pokemon(pokemon),)
    )

    bag = save_data.get_item_bag()
    inventory = InventorySnapshot(
        tuple(ItemQuantity(slot.item.name, slot.quantity) for slot in bag.items),
        tuple(ItemQuantity(slot.item.name, slot.quantity) for slot in bag.poke_balls),
        tuple(ItemQuantity(slot.item.name, slot.quantity) for slot in bag.key_items),
    )

    storage_pokemon = tuple(
        StoragePokemonSnapshot(box.number, slot.slot_index, snapshot_module._pokemon(slot.pokemon))
        for box in save_data.get_pokemon_storage().boxes
        for slot in box.slots
        if snapshot_module._storage_pokemon_is_readable(slot.pokemon)
    )
    storage = StorageSnapshot(save_data.get_pokemon_storage().active_box_index, storage_pokemon)

    flag_names = (
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
    variable_names = (
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
    badges = tuple(NamedFlag(f"BADGE{i:02d}_GET", save_data.get_event_flag(f"BADGE{i:02d}_GET")) for i in range(1, 9))
    campaign_flags = tuple(NamedFlag(name, save_data.get_event_flag(name)) for name in flag_names)
    campaign_variables = tuple(NamedVariable(name, save_data.get_event_var(name)) for name in variable_names)
    text_speed = unpack_uint16(save_data.get_save_block(2, offset=0x14, size=2)) & 0x07
    map_group, map_number = save_data.get_map_group_and_number()
    coordinates = save_data.get_map_local_coordinates()
    try:
        from modules.map_data import get_map_enum

        canonical_map_name = get_map_enum((map_group, map_number)).name
    except (KeyError, RuntimeError, TypeError, ValueError):
        canonical_map_name = None

    result = NuzlockeSnapshot(
        frame=0,
        game_id=profile_object.rom.game_name,
        game_state=None,
        player=PlayerSnapshot(player.name, map_group, map_number, canonical_map_name, coordinates, None, False),
        party=party,
        inventory=inventory,
        battle=None,
        pc=storage,
        progression=ProgressionSnapshot(badges),
        game_state_available=False,
        player_available=True,
        party_available=True,
        inventory_available=True,
        pc_available=True,
        battle_available=False,
        campaign_observation=CampaignObservationSnapshot(
            campaign_flags,
            campaign_variables,
            text_speed,
            available=True,
            lifecycle=CampaignObservationLifecycle.ACTIVE,
        ),
    )
    context.profile = previous_profile
    context.emulator = previous_emulator
    return result


def audit_fixture(profile: str | Path, *, event_file: str = "nuzlocke_events.json") -> dict[str, object]:
    """Report offline fixture provenance without attempting state materialization."""
    profile_path = Path(profile)
    store = JsonEventStore(profile_path / event_file)
    records = tuple(store.iter_records())
    projection = load_campaign_projection(store)
    rules = load_rules(store)
    try:
        snapshot = load_offline_normalized_snapshot(profile_path)
        state_status = "available"
        missing = (
            "emulator frame",
            "game state",
            "current battle",
            "facing",
            "controllability",
        )
        save_map = (snapshot.player.map_group, snapshot.player.map_number)
    except FixtureStateUnavailable:
        state_status = "unavailable"
        missing = (
            "current map",
            "player coordinates",
            "party",
            "inventory",
            "save-backed campaign observation",
        )
        save_map = None
    return {
        "state_status": state_status,
        "missing": missing,
        "save_map": save_map,
        "persisted_event_count": len(records),
        "known_session_ids": projection.known_session_ids,
        "nuzlocke_started": projection.nuzlocke_started,
        "encounters": tuple(
            {
                "location": encounter.location,
                "status": encounter.status,
                "eligible": encounter.eligible,
            }
            for encounter in rules.encounters
        ),
        "rules_legal": rules.legal,
    }


def load_campaign_state_from_fixture(
    profile: str | Path,
    *,
    snapshot: NuzlockeSnapshot | None = None,
    event_file: str = "nuzlocke_events.json",
) -> CampaignState:
    """Materialize state without writes or emulator access when a snapshot exists.

    Persisted events provide campaign/rules projections.  Current map,
    coordinates, party, inventory, and save-backed campaign flags still come
    from the supplied normalized snapshot, which is normally produced by the
    existing live observation layer.
    """
    if snapshot is None:
        snapshot = load_offline_normalized_snapshot(profile)
    profile_path = Path(profile)
    store = JsonEventStore(profile_path / event_file)
    canonical_area = snapshot.player.map_name
    return CampaignState.from_runtime_state(
        snapshot=snapshot,
        observed_projection=load_campaign_projection(store),
        rules_projection=load_rules(store),
        canonical_area=canonical_area,
    )
