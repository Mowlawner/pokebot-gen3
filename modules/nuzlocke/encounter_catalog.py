"""World-owned encounter opportunity catalog.

The ROM supplies existence; the Nuzlocke projection supplies campaign state.
"""

from dataclasses import dataclass
from functools import lru_cache

from modules.context import context
from modules.map import get_maps_with_wild_encounters
from modules.map import get_wild_encounters_for_map

from .campaign_state import CampaignState
from .capture_policy import EncounterCandidate, EncounterMethod


@dataclass(frozen=True, slots=True)
class EncounterOpportunity:
    """World encounter location overlaid with observed Nuzlocke state."""

    location: tuple[int, int]
    observed: bool
    consumed: bool
    eligible: bool


_world_encounter_locations_cache: dict[tuple[int, str], tuple[tuple[int, int], ...]] = {}


# Encounter acquisition is intentionally conservative while the campaign
# runtime only has a land-encounter executor.  The ROM catalog remains broad;
# this boundary controls which methods may be mounted by campaign planning.
CAMPAIGN_EXECUTABLE_ENCOUNTER_METHODS = frozenset({EncounterMethod.LAND})


def _party_has_move(state: CampaignState, move_name: str) -> bool:
    """Return whether an observed party member knows a field move."""

    if not state.party.is_known:
        return False
    return any(
        not getattr(member, "egg", getattr(member, "is_egg", False))
        and any(getattr(move, "name", None) == move_name for move in member.moves)
        for member in state.party.value or ()
    )


def _badge_is_observed(state: CampaignState, badge_name: str) -> bool:
    """Return whether an observed badge flag is set."""

    badge = state.has_badge(badge_name)
    return badge.is_known and badge.value is True


def available_encounter_methods(state: CampaignState) -> frozenset[EncounterMethod]:
    """Return methods the current save can perform in principle.

    This is deliberately separate from the campaign executor support set. A
    Surf-capable save should remain visible to future catalog consumers even
    while the campaign encounter executor supports land only.
    """

    methods = {EncounterMethod.LAND}
    if _badge_is_observed(state, "BADGE05_GET") and _party_has_move(state, "Surf"):
        methods.add(EncounterMethod.SURF)
    if _badge_is_observed(state, "BADGE03_GET") and _party_has_move(state, "Rock Smash"):
        methods.add(EncounterMethod.ROCK_SMASH)

    if state.inventory.is_known:
        for item_name, method in (
            ("Old Rod", EncounterMethod.FISHING_OLD_ROD),
            ("Good Rod", EncounterMethod.FISHING_GOOD_ROD),
            ("Super Rod", EncounterMethod.FISHING_SUPER_ROD),
        ):
            if state.item_quantity(item_name).value > 0:
                methods.add(method)
    return frozenset(methods)


def campaign_encounter_methods(state: CampaignState) -> frozenset[EncounterMethod]:
    """Return methods that are both save-usable and campaign-executable."""

    return available_encounter_methods(state) & CAMPAIGN_EXECUTABLE_ENCOUNTER_METHODS


def _world_encounter_locations() -> tuple[tuple[int, int], ...]:
    """Return the static encounter-map catalog for the active ROM."""

    game_key = (id(context.rom), context.rom.id)
    locations = _world_encounter_locations_cache.get(game_key)
    if locations is None:
        locations = get_maps_with_wild_encounters()
        _world_encounter_locations_cache[game_key] = locations
    return locations


def _encounter_projection_signature(state: CampaignState) -> tuple[object, tuple[tuple[object, ...], ...]]:
    """Return only encounter facts that affect opportunity eligibility."""

    if not state.encounters.is_known:
        return (state.encounters.status.value, ())
    records = tuple(
        sorted(
            (
                encounter.location,
                encounter.status,
                encounter.eligible,
            )
            for encounter in state.encounters.value or ()
        )
    )
    return (state.encounters.status.value, records)


@lru_cache(maxsize=32)
def _cached_encounter_opportunities(
    world_locations: tuple[tuple[int, int], ...],
    encounter_signature: tuple[object, tuple[tuple[object, ...], ...]],
) -> tuple[EncounterOpportunity, ...]:
    """Overlay one immutable encounter projection onto the static catalog."""

    status, records = encounter_signature
    observed = {record[0]: (record[1], record[2]) for record in records}
    result = []
    for location in sorted(set(world_locations)):
        record = observed.get(location)
        if status != "known" or record is None:
            result.append(EncounterOpportunity(location, False, False, True))
            continue
        encounter_status, eligible = record
        result.append(
            EncounterOpportunity(
                location,
                True,
                encounter_status != "none",
                eligible,
            )
        )
    return tuple(result)


def encounter_candidates_for_location(
    location: tuple[int, int],
    *,
    pending: bool = True,
    reachable: bool = True,
    available_methods: frozenset[EncounterMethod] | None = None,
) -> tuple[EncounterCandidate, ...]:
    """Convert ROM encounter tables into policy candidates for one area.

    The returned candidates describe possible species, not a guaranteed
    result. Legality and route reachability remain inputs from the caller.
    When ``available_methods`` is supplied, tables requiring unavailable
    capabilities are omitted; the unfiltered catalog remains available for
    future-capability planning.
    """

    encounters = get_wild_encounters_for_map(*location)
    if encounters is None:
        return ()
    tables = (
        (EncounterMethod.LAND, encounters.land_encounters),
        (EncounterMethod.SURF, encounters.surf_encounters),
        (EncounterMethod.ROCK_SMASH, encounters.rock_smash_encounters),
        (EncounterMethod.FISHING_OLD_ROD, encounters.old_rod_encounters),
        (EncounterMethod.FISHING_GOOD_ROD, encounters.good_rod_encounters),
        (EncounterMethod.FISHING_SUPER_ROD, encounters.super_rod_encounters),
    )
    result = []
    for method, entries in tables:
        method_is_available = (
            available_methods is None
            or method in available_methods
            or (
                method
                in {
                    EncounterMethod.FISHING_OLD_ROD,
                    EncounterMethod.FISHING_GOOD_ROD,
                    EncounterMethod.FISHING_SUPER_ROD,
                }
                and EncounterMethod.FISHING in available_methods
            )
        )
        if not entries or not method_is_available:
            continue
        species = tuple(dict.fromkeys(entry.species.name for entry in entries))
        types = tuple(dict.fromkeys(type_.name.lower() for entry in entries for type_ in entry.species.types))
        result.append(EncounterCandidate(location, method, species, types, reachable=reachable, pending=pending))
    return tuple(result)


def encounter_opportunities(
    state: CampaignState,
    locations: tuple[tuple[int, int], ...] | None = None,
) -> tuple[EncounterOpportunity, ...]:
    """Overlay observed Nuzlocke facts on the authoritative world catalog."""
    world_locations = _world_encounter_locations() if locations is None else tuple(locations)
    return _cached_encounter_opportunities(world_locations, _encounter_projection_signature(state))
