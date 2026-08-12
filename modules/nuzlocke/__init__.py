"""Read-only normalized observations for future Nuzlocke domain code."""

from .snapshots import (
    BattleSnapshot,
    BattlePokemonSnapshot,
    ItemQuantity,
    InventorySnapshot,
    MoveSnapshot,
    NamedFlag,
    NuzlockeSnapshot,
    PartyPokemonSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    PokemonSnapshot,
    StorageSnapshot,
    StoragePokemonSnapshot,
    get_nuzlocke_snapshot,
)
from .identity import PokemonIdentity
from .events import PokemonStorageLocation, StorageChanged
from .runtime import NuzlockeRuntime
from .persistence import (
    EventStoreCorruptionError,
    EventStoreError,
    JsonEventStore,
    deserialize_event,
    serialize_event,
)
from .projection import (
    CampaignProjection,
    ObservedBattle,
    ObservedCampaignState,
    ObservedEvent,
    ObservedPokemon,
    load_campaign_projection,
    reduce_events,
)

__all__ = [
    "BattleSnapshot",
    "BattlePokemonSnapshot",
    "ItemQuantity",
    "InventorySnapshot",
    "MoveSnapshot",
    "NamedFlag",
    "NuzlockeSnapshot",
    "PartyPokemonSnapshot",
    "PlayerSnapshot",
    "ProgressionSnapshot",
    "PokemonSnapshot",
    "StorageSnapshot",
    "StoragePokemonSnapshot",
    "PokemonIdentity",
    "PokemonStorageLocation",
    "StorageChanged",
    "get_nuzlocke_snapshot",
    "NuzlockeRuntime",
    "EventStoreError",
    "EventStoreCorruptionError",
    "JsonEventStore",
    "serialize_event",
    "deserialize_event",
    "CampaignProjection",
    "ObservedBattle",
    "ObservedCampaignState",
    "ObservedEvent",
    "ObservedPokemon",
    "load_campaign_projection",
    "reduce_events",
]
