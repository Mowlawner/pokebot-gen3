import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.map_data import MapRSE, PokemonCenter
from modules.map_path import Direction, PathFindingError
from modules.modes import BotModeError
from modules.navigation import NavigationAction, NavigationActionType, NavigationPlan
from modules.nuzlocke.resource_policy import RouteRecovery
from modules.nuzlocke.resource_runtime import observe_route_recovery
from modules.nuzlocke.emerald_healing_catalog import HealingSourceRSE
from modules.modes.util.map import find_closest_pokemon_center, pokemon_center_candidates


class RouteRecoveryObservationTests(unittest.TestCase):
    def test_indoor_map_uses_registered_outdoor_parent_for_recovery(self):
        self.assertEqual(
            pokemon_center_candidates(MapRSE.PETALBURG_CITY_GYM),
            (PokemonCenter.PetalburgCity,),
        )

    def test_indoor_map_retains_center_when_legacy_pathfinder_cannot_cross_warp(self):
        location = (MapRSE.PETALBURG_CITY_GYM, (4, 4))
        with patch(
            "modules.modes.util.map.calculate_path",
            side_effect=PathFindingError("indoor and outdoor maps are not connected"),
        ):
            center = find_closest_pokemon_center(location)
        self.assertIs(center, PokemonCenter.PetalburgCity)

    def test_valid_location_and_center_report_route(self):
        plan = SimpleNamespace(metrics=SimpleNamespace(total_route_cost=1), destination=PokemonCenter.OldaleTown.value)
        with patch(
            "modules.nuzlocke.resource_runtime.get_player_location", return_value=(MapRSE.ROUTE101, (3, 4))
        ), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.OldaleTown
        ), patch(
            "modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(plan, object())
        ), patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=object()
        ):
            result = observe_route_recovery()
        self.assertEqual(result.distance_to_center, 1)
        self.assertTrue(result.center_available)
        self.assertTrue(result.observation_available)

    def test_valid_location_retains_world_plan_as_executable_route(self):
        location = (MapRSE.ROUTE101, (3, 4))
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            location,
            (MapRSE.ROUTE101, (4, 4)),
        )
        plan = NavigationPlan((action,), (MapRSE.ROUTE101, (4, 4)), SimpleNamespace(total_route_cost=1))
        with patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.OldaleTown
        ), patch("modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(plan, object())
        ), patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=object()
        ):
            result = observe_route_recovery()
        self.assertIsInstance(result.route, NavigationPlan)
        self.assertEqual(result.route.actions[0].action_type, NavigationActionType.MOVE)
        self.assertEqual(result.route.actions[0].source, location)
        self.assertEqual(result.route.actions[0].destination, (MapRSE.ROUTE101, (4, 4)))

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
            "modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=RuntimeError("unexpected path bug"),
        ), patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=object()
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

    def test_woods_uses_cataloged_petalburg_center_when_legacy_table_has_no_entry(self):
        location = (MapRSE.PETALBURG_WOODS, (21, 34))
        plan = SimpleNamespace(
            metrics=SimpleNamespace(total_route_cost=37),
            destination=(MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value, (6, 8)),
            forced_trainer_exposure=False,
        )
        graph = SimpleNamespace(
            route=lambda _source, _target: SimpleNamespace(estimated_cost=10),
        )
        with patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
        ), patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center",
            side_effect=BotModeError("no legacy entry"),
        ), patch(
            "modules.nuzlocke.resource_runtime.emerald_healing_sources",
            return_value=(HealingSourceRSE.PETALBURG_POKEMON_CENTER,),
        ), patch(
            "modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=graph
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(plan, object())
        ) as world_plan:
            result = observe_route_recovery()

        self.assertTrue(result.center_available)
        self.assertEqual(result.center_location, HealingSourceRSE.PETALBURG_POKEMON_CENTER.outdoor_location)
        self.assertEqual(result.distance_to_center, 37)
        self.assertIs(result.route, plan)
        self.assertEqual(
            result.navigation_goal.target.destination_map,
            MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value,
        )
        world_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
