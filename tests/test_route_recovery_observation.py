import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.map_data import MapRSE, PokemonCenter
from modules.map_path import PathFindingError
from modules.nuzlocke.resource_policy import RouteRecovery
from modules.nuzlocke.resource_runtime import observe_route_recovery


class RouteRecoveryObservationTests(unittest.TestCase):
    def test_valid_location_and_center_report_route(self):
        with patch(
            "modules.nuzlocke.resource_runtime.get_player_location", return_value=(MapRSE.ROUTE101, (3, 4))
        ), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.OldaleTown
        ), patch(
            "modules.nuzlocke.resource_runtime.calculate_path", return_value=[object()]
        ):
            result = observe_route_recovery()
        self.assertEqual(result.distance_to_center, 1)
        self.assertTrue(result.center_available)
        self.assertTrue(result.observation_available)

    def test_unavailable_player_location_is_explicit_and_safe(self):
        with patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=None):
            result = observe_route_recovery()
        self.assertFalse(result.observation_available)
        self.assertEqual(result.observation_error, "player_location_unavailable")
        self.assertFalse(result.center_available)

    def test_transient_location_error_is_not_silently_known_route(self):
        with patch(
            "modules.nuzlocke.resource_runtime.get_player_location",
            side_effect=RuntimeError("avatar is between maps"),
        ):
            result = observe_route_recovery()
        self.assertFalse(result.observation_available)
        self.assertIn("between maps", result.observation_error)

    def test_malformed_center_is_explicitly_unavailable(self):
        with patch(
            "modules.nuzlocke.resource_runtime.get_player_location", return_value=(MapRSE.ROUTE101, (3, 4))
        ), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center",
            return_value=type("Center", (), {"value": None})(),
        ):
            result = observe_route_recovery()
        self.assertFalse(result.observation_available)
        self.assertEqual(result.observation_error, "center_location_unavailable")

    def test_pathfinding_failure_with_valid_observation_is_known_unavailable_route(self):
        with patch(
            "modules.nuzlocke.resource_runtime.get_player_location", return_value=(MapRSE.ROUTE101, (3, 4))
        ), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.OldaleTown
        ), patch(
            "modules.nuzlocke.resource_runtime.calculate_path", side_effect=RuntimeError("unexpected path bug")
        ):
            with self.assertRaises(RuntimeError):
                observe_route_recovery()

    def test_cross_map_recovery_uses_world_navigation_when_legacy_pathfinder_cannot(self):
        location = (MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB, (6, 5))
        plan = SimpleNamespace(metrics=SimpleNamespace(total_route_cost=84), destination=(MapRSE.OLDALE_TOWN, (6, 16)))
        with patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.OldaleTown
        ), patch(
            "modules.nuzlocke.resource_runtime.calculate_path",
            side_effect=PathFindingError("warps are not supported"),
        ), patch(
            "modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(plan, object())
        ) as world_plan, patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=object()
        ):
            result = observe_route_recovery()
        self.assertEqual(result.distance_to_center, 84)
        self.assertTrue(result.center_available)
        world_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
