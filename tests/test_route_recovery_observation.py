import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.map_data import MapRSE, PokemonCenter
from modules.map_path import Direction, PathFindingError
from modules.modes import BotModeError
from modules.goals import EncounterMode
from modules.navigation import (
    NavigationAction,
    NavigationActionType,
    NavigationPlan,
    NavigationSearchLimitExceeded,
)
from modules.overworld import WorldTransition
from modules.nuzlocke.resource_policy import RouteRecovery
from modules.nuzlocke.resource_runtime import (
    RecoveryPlanningError,
    RouteRecoveryRequest,
    _cataloged_healing_warp,
    observe_route_recovery,
    plan_route_recovery,
    prepare_route_recovery,
)
from modules.nuzlocke.emerald_healing_catalog import HealingSourceRSE
from modules.modes.util.map import find_closest_pokemon_center, pokemon_center_candidates


class RouteRecoveryObservationTests(unittest.TestCase):
    def test_cataloged_healing_warp_normalizes_enum_map_cache_keys(self):
        door = WorldTransition(
            (MapRSE.OLDALE_TOWN.value, (6, 16)),
            (MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value, (7, 8)),
        )
        source = SimpleNamespace(
            interior_map=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F,
            outdoor_location=(MapRSE.OLDALE_TOWN, (6, 16)),
        )
        with patch(
            "modules.nuzlocke.resource_runtime.static_map_transitions",
            return_value=(door,),
        ) as transitions:
            self.assertIs(_cataloged_healing_warp(source.outdoor_location, source), door)

        transitions.assert_called_once_with(MapRSE.OLDALE_TOWN.value)

    def test_recovery_snapshot_includes_candidate_outdoor_and_interior_maps(self):
        location = (MapRSE.ROUTE103.value, (14, 11))
        source = SimpleNamespace(
            source_id="pokemon_center:oldale",
            outdoor_location=(MapRSE.OLDALE_TOWN, (6, 16)),
            interior_map=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F,
        )
        door = WorldTransition(
            (MapRSE.OLDALE_TOWN.value, (6, 16)),
            (MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value, (7, 8)),
        )
        graph = SimpleNamespace(
            route=lambda _source, _target: SimpleNamespace(
                maps=(MapRSE.ROUTE103.value, MapRSE.OLDALE_TOWN.value),
                estimated_cost=43,
            ),
        )
        with (
            patch(
                "modules.nuzlocke.resource_runtime.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location),
            patch("modules.nuzlocke.resource_runtime.pokemon_center_candidates", return_value=(PokemonCenter.OldaleTown,)),
            patch("modules.nuzlocke.resource_runtime.emerald_healing_sources", return_value=(source,)),
            patch("modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()),
            patch("modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()),
            patch("modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=graph),
            patch("modules.nuzlocke.resource_runtime.static_map_transitions", return_value=(door,)),
            patch("modules.nuzlocke.resource_runtime.prewarm_static_map_observation", return_value=()),
            patch("modules.nuzlocke.resource_runtime.static_navigation_tile_snapshot", return_value=()),
        ):
            request = prepare_route_recovery(candidate_limit=1)

        catalog_maps = {map_id for map_id, _tiles in request.static_tile_catalog}
        self.assertIn(MapRSE.ROUTE103.value, catalog_maps)
        self.assertIn(MapRSE.OLDALE_TOWN.value, catalog_maps)
        self.assertIn(MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value, catalog_maps)

    def test_route_102_recovery_keeps_both_nearby_centers_as_candidates(self):
        location = (MapRSE.ROUTE102, (16, 20))
        graph = SimpleNamespace(route=lambda _source, _target: SimpleNamespace(estimated_cost=10))
        with patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
        ), patch(
            "modules.nuzlocke.resource_runtime.get_player_location", return_value=location
        ), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.PetalburgCity
        ), patch(
            "modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=graph
        ):
            request = prepare_route_recovery()

        self.assertEqual(
            {source.source_id for _, source in request.ordered_sources},
            {"pokemon_center:oldale", "pokemon_center:petalburg"},
        )
        self.assertNotIn("pokemon_center:slateport", {source.source_id for _, source in request.ordered_sources})

    def test_route_104_recovery_normalizes_catalog_maps_and_prefers_petalburg(self):
        location = (MapRSE.ROUTE104, (8, 18))
        costs = {
            MapRSE.PETALBURG_CITY.value: 12,
            MapRSE.RUSTBORO_CITY.value: 31,
        }
        graph = SimpleNamespace(
            estimate_location_cost=lambda _start, target: costs[target[0]],
        )
        with patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
        ), patch(
            "modules.nuzlocke.resource_runtime.get_player_location", return_value=location
        ), patch(
            "modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=graph
        ):
            request = prepare_route_recovery()

        ordered = [source.source_id for _, source in request.ordered_sources]
        assert ordered[:2] == ["pokemon_center:petalburg", "pokemon_center:rustboro"]
        assert "pokemon_center:oldale" not in ordered

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

    def test_live_emerald_recovery_uses_catalog_and_world_plan(self):
        location = (MapRSE.ROUTE103, (9, 3))
        plan = SimpleNamespace(metrics=SimpleNamespace(total_route_cost=12), destination=(MapRSE.OLDALE_TOWN, (6, 16)))
        with patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(
                rom=SimpleNamespace(is_rse=True),
                testing=False,
                stutter_trace=None,
            ),
        ), patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location), patch(
            "modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(plan, object())
        ) as world_plan:
            result = observe_route_recovery(candidate_limit=1)

        world_plan.assert_called_once()
        self.assertTrue(result.center_available)
        self.assertIsNotNone(result.route)
        self.assertEqual(result.distance_to_center, 12)
        self.assertEqual(
            result.navigation_goal.target.destination_map,
            MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
        )

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

    def test_transient_keyerror_in_overworld_observation_is_unavailable(self):
        with patch(
            "modules.nuzlocke.resource_runtime.get_player_location",
            return_value=(MapRSE.ROUTE101, (3, 4)),
        ), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center",
            return_value=PokemonCenter.OldaleTown,
        ), patch(
            "modules.nuzlocke.resource_runtime.perceive_overworld",
            side_effect=KeyError("map object table"),
        ):
            result = observe_route_recovery()

        self.assertFalse(result.observation_available)
        self.assertIn("KeyError", result.observation_error)
        self.assertFalse(result.center_available)

    def test_malformed_center_is_explicitly_unavailable(self):
        with patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(rom=SimpleNamespace(is_rse=False), stutter_trace=None),
        ), patch(
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

    def test_cross_map_recovery_uses_world_navigation(self):
        location = (MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB, (6, 5))
        plan = SimpleNamespace(metrics=SimpleNamespace(total_route_cost=84), destination=(MapRSE.OLDALE_TOWN, (6, 16)))
        with patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center", return_value=PokemonCenter.OldaleTown
        ), patch("modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(plan, object())
        ) as world_plan, patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=object()
        ):
            result = observe_route_recovery(candidate_limit=1)
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

    def test_recovery_candidate_search_is_bounded_and_deterministic(self):
        location = (MapRSE.ROUTE101, (3, 4))
        sources = tuple(
            SimpleNamespace(
                source_id=source_id,
                outdoor_location=(MapRSE.ROUTE101, coordinates),
                interior_map=interior_map,
            )
            for source_id, coordinates, interior_map in (
                ("candidate:first", (1, 1), MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F),
                ("candidate:second", (2, 1), MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F),
                ("candidate:third", (3, 1), MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F),
            )
        )
        estimated_costs = {
            MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value: 10,
            MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value: 20,
            MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F.value: 30,
        }
        graph = SimpleNamespace(
            route=lambda _source, target: SimpleNamespace(estimated_cost=estimated_costs[target]),
        )
        plans = (
            NavigationSearchLimitExceeded("candidate search budget exhausted"),
            (
                SimpleNamespace(
                    metrics=SimpleNamespace(encounter_opportunities=1, total_route_cost=20),
                    destination=(MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value, (6, 8)),
                    forced_trainer_exposure=False,
                ),
                object(),
            ),
            (
                SimpleNamespace(
                    metrics=SimpleNamespace(encounter_opportunities=1, total_route_cost=20),
                    destination=(MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F.value, (6, 8)),
                    forced_trainer_exposure=False,
                ),
                object(),
            ),
        )

        with patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
        ), patch("modules.nuzlocke.resource_runtime.get_player_location", return_value=location), patch(
            "modules.nuzlocke.resource_runtime.emerald_healing_sources", return_value=sources
        ), patch("modules.nuzlocke.resource_runtime.perceive_overworld", return_value=object()), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch("modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=graph), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", side_effect=plans
        ) as world_plan:
            result = observe_route_recovery(candidate_limit=3)

        self.assertTrue(result.center_available)
        self.assertEqual(result.center_location, sources[1].outdoor_location)
        self.assertEqual(result.distance_to_center, 20)
        self.assertEqual(world_plan.call_args_list[0].kwargs["cost_ceiling"], None)
        self.assertEqual(world_plan.call_args_list[1].kwargs["cost_ceiling"], None)
        self.assertEqual(world_plan.call_args_list[2].kwargs["cost_ceiling"], 20)

    def test_zero_exposure_nearby_recovery_does_not_search_later_centers(self):
        first = SimpleNamespace(
            source_id="pokemon_center:petalburg",
            outdoor_location=(MapRSE.PETALBURG_CITY, (8, 18)),
            interior_map=MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
        )
        second = SimpleNamespace(
            source_id="pokemon_center:rustboro",
            outdoor_location=(MapRSE.RUSTBORO_CITY, (8, 18)),
            interior_map=MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F,
        )
        first_plan = SimpleNamespace(
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=18),
            destination=(MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value, (6, 8)),
            forced_trainer_exposure=False,
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE104, (8, 18)),
            world=object(),
            graph=object(),
            ordered_sources=((0, first), (1, second)),
            center_location=first.outdoor_location,
            candidate_limit=2,
            critical=False,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(first_plan, object()),
        ) as world_plan:
            result = plan_route_recovery(request)

        self.assertEqual(world_plan.call_count, 2)
        self.assertEqual(result.center_location, first.outdoor_location)

    def test_recovery_plan_uses_the_cataloged_center_door_transition(self):
        source = SimpleNamespace(
            source_id="pokemon_center:rustboro",
            outdoor_location=(MapRSE.RUSTBORO_CITY, (16, 38)),
            interior_map=MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F,
        )
        door = WorldTransition(
            source.outdoor_location,
            (MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F, (6, 8)),
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE116, (37, 13)),
            world=object(),
            graph=object(),
            ordered_sources=((0, source),),
            center_location=source.outdoor_location,
            candidate_limit=1,
            healing_warps=((0, door),),
        )
        plan = SimpleNamespace(
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=12),
            destination=door.destination,
            forced_trainer_exposure=False,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(plan, object()),
        ) as world_plan:
            result = plan_route_recovery(request)

        self.assertIs(result.navigation_goal.target.warp, door)
        self.assertEqual(world_plan.call_args.args[2].target.destination, door.destination)
        self.assertEqual(world_plan.call_args.args[2].target.destination_map, door.destination[0].value)

    def test_critical_recovery_prefers_fewer_encounters_over_a_shorter_exposed_route(self):
        first = SimpleNamespace(
            source_id="pokemon_center:near",
            outdoor_location=(MapRSE.ROUTE104, (8, 18)),
            interior_map=MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
        )
        second = SimpleNamespace(
            source_id="pokemon_center:safer",
            outdoor_location=(MapRSE.ROUTE104, (20, 18)),
            interior_map=MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F,
        )
        plans = (
            (
                SimpleNamespace(
                    metrics=SimpleNamespace(encounter_opportunities=2, total_route_cost=10),
                    destination=(MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value, (6, 8)),
                    forced_trainer_exposure=False,
                ),
                object(),
            ),
            (
                SimpleNamespace(
                    metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=30),
                    destination=(MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F.value, (6, 8)),
                    forced_trainer_exposure=False,
                ),
                object(),
            ),
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE104, (8, 18)),
            world=object(),
            graph=object(),
            ordered_sources=((0, first), (1, second)),
            center_location=first.outdoor_location,
            candidate_limit=2,
            critical=True,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=plans,
        ):
            result = plan_route_recovery(request)

        self.assertEqual(result.center_location, second.outdoor_location)
        self.assertEqual(result.encounter_opportunities, 0)

    def test_recovery_keeps_completed_incumbent_when_later_candidate_times_out(self):
        first = SimpleNamespace(
            source_id="pokemon_center:petalburg",
            outdoor_location=(MapRSE.PETALBURG_CITY, (8, 18)),
            interior_map=MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
        )
        second = SimpleNamespace(
            source_id="pokemon_center:rustboro",
            outdoor_location=(MapRSE.RUSTBORO_CITY, (16, 38)),
            interior_map=MapRSE.RUSTBORO_CITY_POKEMON_CENTER_1F,
        )
        first_plan = SimpleNamespace(
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=24),
            destination=(MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value, (6, 8)),
            forced_trainer_exposure=False,
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE104, (8, 18)),
            world=object(),
            graph=object(),
            ordered_sources=((0, first), (1, second)),
            center_location=first.outdoor_location,
            candidate_limit=2,
            critical=False,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=((first_plan, object()), TimeoutError("candidate budget exhausted")),
        ):
            result = plan_route_recovery(request)

        self.assertEqual(result.center_location, first.outdoor_location)
        self.assertTrue(result.calculation_timed_out)
        self.assertEqual(
            [(item.source_id, item.outcome) for item in result.candidate_diagnostics],
            [("pokemon_center:petalburg", "success"), ("pokemon_center:rustboro", "timed_out")],
        )

    def test_opportunistic_recovery_uses_weighted_route_cost_for_center_selection(self):
        nearby = SimpleNamespace(
            source_id="pokemon_center:petalburg",
            outdoor_location=(MapRSE.PETALBURG_CITY, (8, 18)),
            interior_map=MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
        )
        distant = SimpleNamespace(
            source_id="pokemon_center:oldale",
            outdoor_location=(MapRSE.OLDALE_TOWN, (6, 16)),
            interior_map=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F,
        )
        plans = (
            (
                SimpleNamespace(
                    metrics=SimpleNamespace(encounter_opportunities=7, total_route_cost=93),
                    destination=(MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value, (6, 8)),
                    forced_trainer_exposure=False,
                ),
                object(),
            ),
            (
                SimpleNamespace(
                    metrics=SimpleNamespace(encounter_opportunities=1, total_route_cost=96),
                    destination=(MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value, (6, 8)),
                    forced_trainer_exposure=False,
                ),
                object(),
            ),
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE104, (8, 18)),
            world=object(),
            graph=object(),
            ordered_sources=((0, nearby), (1, distant)),
            center_location=nearby.outdoor_location,
            candidate_limit=2,
            critical=False,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=plans,
        ) as world_plan:
            result = plan_route_recovery(request)

        self.assertEqual(result.center_location, nearby.outdoor_location)
        self.assertEqual(result.encounter_opportunities, 7)
        self.assertEqual(
            [item.args[2].encounter_mode for item in world_plan.call_args_list],
            [EncounterMode.MOSTLY_AVOID, EncounterMode.MOSTLY_AVOID],
        )
        self.assertEqual(
            [(item.distance, item.encounter_opportunities) for item in result.candidate_diagnostics],
            [(93, 7), (96, 1)],
        )
        self.assertIsNone(world_plan.call_args_list[0].kwargs["cost_ceiling"])
        self.assertEqual(world_plan.call_args_list[1].kwargs["cost_ceiling"], 93)

    def test_critical_recovery_retries_search_exhaustion_with_mostly_avoid(self):
        source = SimpleNamespace(
            source_id="pokemon_center:petalburg",
            outdoor_location=(MapRSE.PETALBURG_CITY, (8, 18)),
            interior_map=MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
        )
        plan = SimpleNamespace(
            metrics=SimpleNamespace(encounter_opportunities=1, total_route_cost=24),
            destination=(MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value, (6, 8)),
            forced_trainer_exposure=False,
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE104, (8, 18)),
            world=object(),
            graph=object(),
            ordered_sources=((0, source),),
            center_location=source.outdoor_location,
            candidate_limit=1,
            critical=True,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=(NavigationSearchLimitExceeded("strict budget exhausted"), (plan, object())),
        ) as world_plan:
            result = plan_route_recovery(request)

        self.assertEqual(world_plan.call_count, 2)
        strict_goal = world_plan.call_args_list[0].args[2]
        emergency_goal = world_plan.call_args_list[1].args[2]
        self.assertEqual(strict_goal.encounter_mode, EncounterMode.AVOID)
        self.assertEqual(emergency_goal.encounter_mode, EncounterMode.MOSTLY_AVOID)
        self.assertIs(result.route, plan)
        self.assertEqual(result.candidate_diagnostics[0].reason, "executable trainer-safe route after encounter-risk fallback")
        self.assertEqual(world_plan.call_args_list[1].kwargs["cost_ceiling"], None)

    def test_noncritical_recovery_does_not_use_emergency_search_after_budget_exhaustion(self):
        source = SimpleNamespace(
            source_id="pokemon_center:petalburg",
            outdoor_location=(MapRSE.PETALBURG_CITY, (8, 18)),
            interior_map=MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE104, (8, 18)),
            world=object(),
            graph=object(),
            ordered_sources=((0, source),),
            center_location=source.outdoor_location,
            candidate_limit=1,
            critical=False,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=NavigationSearchLimitExceeded("candidate budget exhausted"),
        ) as world_plan:
            with self.assertRaises(RecoveryPlanningError):
                plan_route_recovery(request)

        world_plan.assert_called_once()

    def test_failed_recovery_plan_retains_diagnostics_for_every_candidate(self):
        sources = tuple(
            SimpleNamespace(
                source_id=source_id,
                outdoor_location=(MapRSE.ROUTE104, (index, 18)),
                interior_map=MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F,
            )
            for index, source_id in enumerate(("pokemon_center:petalburg", "pokemon_center:oldale"))
        )
        request = RouteRecoveryRequest(
            location=(MapRSE.ROUTE104, (8, 18)),
            world=object(),
            graph=object(),
            ordered_sources=tuple(enumerate(sources)),
            center_location=sources[0].outdoor_location,
            candidate_limit=2,
            critical=True,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=(PathFindingError("blocked by trainer"), PathFindingError("no safe exit")),
        ):
            with self.assertRaises(RecoveryPlanningError) as raised:
                plan_route_recovery(request)

        self.assertEqual(
            [(item.source_id, item.outcome, item.reason) for item in raised.exception.candidate_diagnostics],
            [
                ("pokemon_center:petalburg", "failed", "PathFindingError: blocked by trainer"),
                ("pokemon_center:oldale", "failed", "PathFindingError: no safe exit"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
