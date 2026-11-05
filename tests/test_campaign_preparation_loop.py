from types import SimpleNamespace
from unittest.mock import patch

from modules.map_data import MapRSE, PokemonCenter
from modules.nuzlocke.campaign_preparation_loop import CampaignPreparationLoopController


def test_campaign_preparation_mount_does_not_recover_a_healthy_party():
    encounter_spot = SimpleNamespace(
        local_position=(3, 3),
        has_encounters=True,
        is_surfable=False,
    )
    activity_calls = []
    recovery_calls = []

    def recovery_handler(center):
        recovery_calls.append(center)
        yield "unexpected-recovery"

    def activity_handler():
        activity_calls.append(True)

    with (
        patch(
            "modules.nuzlocke.campaign_preparation_loop.get_map_data_for_current_position",
            return_value=encounter_spot,
        ),
        patch(
            "modules.nuzlocke.campaign_preparation_loop.find_closest_pokemon_center",
            return_value=PokemonCenter.OldaleTown,
        ),
        patch(
            "modules.nuzlocke.campaign_preparation_loop.get_map_enum",
            return_value=MapRSE.ROUTE101,
        ),
        patch(
            "modules.nuzlocke.campaign_preparation_loop.navigate_to",
            return_value=iter(()),
        ) as navigate,
        patch(
            "modules.nuzlocke.campaign_preparation_loop.apply_white_flute_if_available",
            return_value=iter(()),
        ),
        patch(
            "modules.nuzlocke.campaign_preparation_loop.heal_in_pokemon_center",
            side_effect=AssertionError("campaign preparation must not heal on mount"),
        ),
    ):
        controller = CampaignPreparationLoopController(
            recovery_handler=recovery_handler,
            activity_handler=activity_handler,
        )
        result = list(controller.run(stop_condition=lambda: bool(activity_calls)))

    assert result == [None]
    assert recovery_calls == []
    navigate.assert_called_once_with(MapRSE.ROUTE101, encounter_spot.local_position)
