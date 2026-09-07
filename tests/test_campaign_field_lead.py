from types import SimpleNamespace
from unittest.mock import patch

from modules.nuzlocke.encounter_catalog import EncounterMethod
from modules.nuzlocke.field_lead import (
    CampaignFieldLeadContext,
    build_campaign_field_lead_context,
)
from modules.world_navigation import WorldEdge, WorldMapGraph


def test_campaign_field_context_uses_the_mounted_objective_route_when_current_map_has_no_encounters():
    current_map = (0, 10)
    route_map = (0, 17)
    graph = WorldMapGraph(
        (
            WorldEdge(
                current_map,
                route_map,
                "connection",
                ((1, 1),),
                ((1, 1),),
            ),
        )
    )
    objective = SimpleNamespace(objective_id="obtain_encounter:0:17", destination=route_map)
    wild = SimpleNamespace(
        land_encounters=(
            SimpleNamespace(
                species=SimpleNamespace(name="Poochyena"),
                min_level=2,
                max_level=4,
                encounter_rate=20,
            ),
        ),
    )

    with (
        patch(
            "modules.nuzlocke.field_lead.get_player_avatar",
            return_value=SimpleNamespace(map_group_and_number=current_map),
        ),
        patch("modules.nuzlocke.field_lead.get_wild_encounters_for_map", side_effect=(None, wild)),
    ):
        result = build_campaign_field_lead_context(objective, graph=graph)

    assert result.current_map == current_map
    assert result.objective_map == route_map
    assert result.route_maps == (current_map, route_map)
    assert result.source == "objective_route_corridor"
    assert result.encounter_level_ceiling == 4
    assert len(result.threats) == 1
    assert result.threats[0].method is EncounterMethod.LAND


def test_campaign_field_context_without_a_route_keeps_a_safe_empty_envelope():
    context = CampaignFieldLeadContext(
        objective_id="receive_pokedex",
        current_map=(0, 16),
        objective_map=(0, 9),
        route_maps=((0, 16),),
        threats=(),
        encounter_level_ceiling=None,
        route_cost=None,
        source="objective_route_without_encounters",
    )

    assert not context.has_encounter_envelope
