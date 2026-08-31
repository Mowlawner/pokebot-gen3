"""World-owned encounter opportunity catalog.

The ROM supplies existence; the Nuzlocke projection supplies campaign state.
"""

from dataclasses import dataclass

from modules.map import get_maps_with_wild_encounters

from .campaign_state import CampaignState


@dataclass(frozen=True, slots=True)
class EncounterOpportunity:
    """World encounter location overlaid with observed Nuzlocke state."""

    location: tuple[int, int]
    observed: bool
    consumed: bool
    eligible: bool


def encounter_opportunities(
    state: CampaignState,
    locations: tuple[tuple[int, int], ...] | None = None,
) -> tuple[EncounterOpportunity, ...]:
    """Overlay observed Nuzlocke facts on the authoritative world catalog."""
    world_locations = get_maps_with_wild_encounters() if locations is None else locations
    result = []
    for location in sorted(set(world_locations)):
        observed = state.encounter_for(location)
        if not observed.is_known:
            result.append(EncounterOpportunity(location, False, False, True))
            continue
        encounter = observed.value
        result.append(
            EncounterOpportunity(
                location,
                any(item.location == location for item in state.encounters.value or ()),
                encounter.status != "none",
                encounter.eligible,
            )
        )
    return tuple(result)
