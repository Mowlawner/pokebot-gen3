"""Campaign field-lead context and route-envelope construction.

The battle strategy decides which party member should lead, but it should not
have to discover campaign intent or reconstruct the world route itself.  This
module is the campaign-side adapter that supplies that information at stable
event boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.map import get_wild_encounters_for_map
from modules.player import get_player_avatar
from modules.world_navigation import MapId, WorldMapGraph, get_world_map_graph

from .campaign_state import CampaignState
from .encounter_catalog import EncounterMethod, campaign_encounter_methods


@dataclass(frozen=True, slots=True)
class FieldEncounterThreat:
    """One weighted species threat in a campaign lead envelope."""

    location: MapId
    species: object
    weight: int
    min_level: int
    max_level: int
    method: EncounterMethod


@dataclass(frozen=True, slots=True)
class CampaignFieldLeadContext:
    """Immutable campaign context supplied to the field-lead policy."""

    objective_id: str | None
    current_map: MapId | None
    objective_map: MapId | None
    route_maps: tuple[MapId, ...]
    threats: tuple[FieldEncounterThreat, ...]
    encounter_level_ceiling: int | None
    route_cost: int | None
    source: str

    @property
    def has_encounter_envelope(self) -> bool:
        return bool(self.threats)


@dataclass(frozen=True, slots=True)
class CampaignFieldLeadDecision:
    """Auditable result returned by the campaign field-lead policy."""

    selected_index: int | None
    reason: str
    objective_id: str | None
    envelope_source: str
    route_maps: tuple[MapId, ...]


def _map_id(value) -> MapId | None:
    """Return a map identifier, rejecting coordinate-like or malformed data."""

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(component, int) for component in value)
    ):
        return value
    return None


def objective_map(objective) -> MapId | None:
    """Extract the map-level destination from a campaign objective."""

    destination = getattr(objective, "destination", None)
    direct = _map_id(destination)
    if direct is not None:
        return direct

    target = getattr(getattr(objective, "tactical_target", None), "target", None)
    return _map_id(getattr(target, "location", None))


def _route_corridor(
    current_map: MapId | None,
    target_map: MapId | None,
    graph: WorldMapGraph,
) -> tuple[tuple[MapId, ...], int | None]:
    """Return the bounded map corridor between the current and target maps.

    The static graph is used as a route estimator, not as story authority. A
    mounted campaign objective is the story/capability gate; the corridor is
    merely the small set of maps worth consulting for future wild encounters.
    """

    if current_map is None:
        return (), None
    if target_map is None or current_map == target_map:
        return (current_map,), 0

    try:
        forward = dict(graph.map_costs(current_map))
        reverse = dict(graph.map_costs_to(target_map))
        direct = forward.get(target_map)
        if direct is None:
            return (current_map,), None

        # Match the campaign encounter frontier's small-detour policy. The
        # shortest route is always included; nearby alternatives are included
        # only when they remain in the same bounded corridor.
        detour_limit = max(10, direct * 0.25)
        corridor = tuple(
            map_id
            for map_id, cost_to_map in sorted(forward.items(), key=lambda item: (item[1], item[0]))
            if (
                map_id in reverse
                and max(0, cost_to_map + reverse[map_id] - direct) <= detour_limit
            )
        )
        shortest = graph.route(current_map, target_map).maps
        return tuple(dict.fromkeys((*shortest, *corridor))), direct
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        # A missing static route must not prevent the campaign from using the
        # ordinary lowest-level fallback.
        return (current_map,), None


def _land_encounter_method(methods: frozenset[EncounterMethod]) -> bool:
    """Return whether the campaign can currently execute land encounters."""

    return EncounterMethod.LAND in methods


def _threats_for_maps(
    maps: tuple[MapId, ...],
    methods: frozenset[EncounterMethod],
) -> tuple[FieldEncounterThreat, ...]:
    if not _land_encounter_method(methods):
        return ()

    threats: list[FieldEncounterThreat] = []
    for location in maps:
        try:
            encounter_list = get_wild_encounters_for_map(*location)
            encounters = () if encounter_list is None else tuple(encounter_list.land_encounters)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError, IndexError):
            continue
        for encounter in encounters:
            species = getattr(encounter, "species", None)
            if species is None:
                continue
            try:
                min_level = int(getattr(encounter, "min_level"))
                max_level = int(getattr(encounter, "max_level"))
                weight = max(1, int(getattr(encounter, "encounter_rate", 1)))
            except (AttributeError, TypeError, ValueError):
                continue
            threats.append(
                FieldEncounterThreat(
                    location=location,
                    species=species,
                    weight=weight,
                    min_level=min_level,
                    max_level=max_level,
                    method=EncounterMethod.LAND,
                )
            )
    return tuple(threats)


def build_campaign_field_lead_context(
    objective,
    *,
    campaign_state: CampaignState | None = None,
    graph: WorldMapGraph | None = None,
) -> CampaignFieldLeadContext:
    """Build the field policy input for the currently mounted objective."""

    try:
        current_map = _map_id(get_player_avatar().map_group_and_number)
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        current_map = None
    target_map = objective_map(objective)
    objective_id = getattr(objective, "objective_id", None)

    route_graph = graph
    if route_graph is None:
        try:
            route_graph = get_world_map_graph()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            route_graph = None

    if route_graph is None:
        route_maps, route_cost = ((current_map,) if current_map is not None else ()), None
    else:
        route_maps, route_cost = _route_corridor(current_map, target_map, route_graph)

    methods = frozenset({EncounterMethod.LAND})
    if campaign_state is not None:
        try:
            methods = campaign_encounter_methods(campaign_state)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            methods = frozenset({EncounterMethod.LAND})

    threats = _threats_for_maps(route_maps, methods)
    level_ceiling = max((threat.max_level for threat in threats), default=None)
    if threats:
        source = "objective_route_corridor"
    elif current_map is not None and target_map is not None:
        source = "objective_route_without_encounters"
    else:
        source = "current_map_only"

    return CampaignFieldLeadContext(
        objective_id=objective_id,
        current_map=current_map,
        objective_map=target_map,
        route_maps=route_maps,
        threats=threats,
        encounter_level_ceiling=level_ceiling,
        route_cost=route_cost,
        source=source,
    )
