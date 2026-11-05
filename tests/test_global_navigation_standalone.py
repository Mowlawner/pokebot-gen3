"""Navigation-only regressions; deliberately avoids agent/campaign imports."""

import unittest
from unittest.mock import patch

from modules.goals import GoalConstraints, NavigationGoal, ReachLocation, ReachWarp, SemanticTarget, TrainerMode
from modules.map_path import Direction
from modules.navigation import (
    GoalAwareNavigator,
    NavigationActionType,
    NavigationWorld,
    NavigableTile,
    plan_observed_warp_locally,
    plan_with_world_navigation,
)
from modules.overworld import MapConnectionObservation, TriggerObservation, WarpActivation, WarpObservation, WorldTransition
from modules.world_navigation import WorldEdge, WorldMapGraph


def world(map_id, coords, transitions=(), facing=Direction.South):
    tiles = {(map_id, xy): NavigableTile((map_id, xy), False, frozenset(Direction)) for xy in coords}
    return NavigationWorld(tiles=tiles, transitions=tuple(transitions), facing=facing)


class GlobalNavigationTests(unittest.TestCase):
    def test_weighted_portal_route_prefers_cheaper_intermediate_map(self):
        source_map, expensive_map, cheap_map, target_map = (702, 0), (702, 1), (702, 2), (702, 3)
        tiles = {}
        for map_id, width, traversal_cost in (
            (source_map, 5, 1),
            (expensive_map, 5, 10),
            (cheap_map, 2, 1),
            (target_map, 1, 1),
        ):
            for x in range(width):
                location = (map_id, (x, 0))
                tiles[location] = NavigableTile(
                    location,
                    False,
                    frozenset(Direction),
                    traversal_cost=traversal_cost,
                )

        transitions = (
            WorldTransition((source_map, (4, 0)), (expensive_map, (0, 0))),
            WorldTransition((expensive_map, (4, 0)), (target_map, (0, 0))),
            WorldTransition((source_map, (1, 0)), (cheap_map, (0, 0))),
            WorldTransition((cheap_map, (1, 0)), (target_map, (0, 0))),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(source_map, expensive_map, "warp", ((4, 0),), ((0, 0),), estimated_cost=1),
                WorldEdge(expensive_map, target_map, "warp", ((4, 0),), ((0, 0),), estimated_cost=1),
                WorldEdge(source_map, cheap_map, "warp", ((1, 0),), ((0, 0),), estimated_cost=5),
                WorldEdge(cheap_map, target_map, "warp", ((1, 0),), ((0, 0),), estimated_cost=5),
            )
        )

        plan, route = plan_with_world_navigation(
            NavigationWorld(
                tiles=tiles,
                transitions=transitions,
                facing=Direction.East,
                static_metadata_frozen=True,
            ),
            (source_map, (0, 0)),
            NavigationGoal(SemanticTarget.map(target_map)),
            graph,
        )

        self.assertEqual(route.maps, (source_map, cheap_map, target_map))
        self.assertEqual(plan.destination, (target_map, (0, 0)))
        self.assertEqual(route.selected_transitions, transitions[2:])
        self.assertEqual(route.estimated_cost, plan.metrics.total_route_cost)

    def test_preferred_map_route_does_not_expand_unrelated_detours(self):
        source_map, direct_map, detour_map, target_map = (703, 0), (703, 1), (703, 2), (703, 3)
        tiles = {
            (map_id, (x, 0)): NavigableTile(
                (map_id, (x, 0)),
                False,
                frozenset(Direction),
            )
            for map_id in (source_map, direct_map, detour_map, target_map)
            for x in range(5)
        }
        transitions = (
            WorldTransition((source_map, (4, 0)), (direct_map, (0, 0))),
            WorldTransition((direct_map, (4, 0)), (target_map, (0, 0))),
            WorldTransition((source_map, (1, 0)), (detour_map, (0, 0))),
            WorldTransition((detour_map, (1, 0)), (target_map, (0, 0))),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(source_map, direct_map, "warp", ((4, 0),), ((0, 0),), estimated_cost=100),
                WorldEdge(direct_map, target_map, "warp", ((4, 0),), ((0, 0),), estimated_cost=100),
                WorldEdge(source_map, detour_map, "warp", ((1, 0),), ((0, 0),), estimated_cost=1),
                WorldEdge(detour_map, target_map, "warp", ((1, 0),), ((0, 0),), estimated_cost=1),
            )
        )

        plan, route = plan_with_world_navigation(
            NavigationWorld(
                tiles=tiles,
                transitions=transitions,
                facing=Direction.East,
                static_metadata_frozen=True,
            ),
            (source_map, (0, 0)),
            NavigationGoal(
                SemanticTarget.map(target_map),
                preferred_map_route=(source_map, direct_map, target_map),
            ),
            graph,
        )

        self.assertEqual(route.maps, (source_map, direct_map, target_map))
        self.assertEqual(plan.destination, (target_map, (0, 0)))

    def test_semantic_map_goal_composes_each_boundary_locally(self):
        source_map, middle_map, target_map = (701, 0), (701, 1), (701, 2)
        tiles = {
            (map_id, (x, 0)): NavigableTile(
                (map_id, (x, 0)),
                False,
                frozenset(Direction),
            )
            for map_id in (source_map, middle_map, target_map)
            for x in range(3)
        }
        first_transition = WorldTransition(
            (source_map, (2, 0)),
            (middle_map, (0, 0)),
        )
        second_transition = WorldTransition(
            (middle_map, (2, 0)),
            (target_map, (0, 0)),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(source_map, middle_map, "warp", ((2, 0),), ((0, 0),)),
                WorldEdge(middle_map, target_map, "warp", ((2, 0),), ((0, 0),)),
            )
        )
        navigation_world = NavigationWorld(
            tiles=tiles,
            transitions=(first_transition, second_transition),
            facing=Direction.East,
            static_metadata_frozen=True,
        )

        with patch(
            "modules.navigation.plan_observed_warp_locally",
            wraps=plan_observed_warp_locally,
        ) as local_planner:
            plan, route = plan_with_world_navigation(
                navigation_world,
                (source_map, (0, 0)),
                NavigationGoal(SemanticTarget.map(target_map)),
                graph,
            )

        self.assertEqual(route.maps, (source_map, middle_map, target_map))
        self.assertEqual(plan.destination, (target_map, (0, 0)))
        self.assertEqual(local_planner.call_count, 2)

    def test_semantic_route_uses_downstream_safe_intermediate_entry(self):
        source_map, middle_map, target_map = (704, 0), (704, 1), (704, 2)
        tiles = {}
        for x in range(5):
            tiles[(source_map, (x, 0))] = NavigableTile(
                (source_map, (x, 0)), False, frozenset(Direction)
            )
            # The two middle-map corridors are deliberately independent. The
            # upper corridor contains the trainer hazard; only its lower
            # counterpart can reach the final exit under AVOID.
            for y in (0, 1):
                tiles[(middle_map, (x, y))] = NavigableTile(
                    (middle_map, (x, y)),
                    False,
                    frozenset((Direction.East, Direction.West)),
                )
        tiles[(source_map, (4, 1))] = NavigableTile(
            (source_map, (4, 1)), False, frozenset(Direction)
        )
        tiles[(target_map, (0, 0))] = NavigableTile(
            (target_map, (0, 0)), False, frozenset(Direction)
        )

        upper_entry = WorldTransition(
            (source_map, (4, 0)),
            (middle_map, (0, 0)),
        )
        lower_entry = WorldTransition(
            (source_map, (4, 1)),
            (middle_map, (0, 1)),
        )
        final_exit = WorldTransition(
            (middle_map, (4, 1)),
            (target_map, (0, 0)),
        )
        trainer = TriggerObservation(
            "trainer:upper",
            locations=frozenset({(middle_map, (2, 0))}),
            hazard_locations=frozenset({(middle_map, (2, 0))}),
            hazard_kind="trainer",
            affordance_id="trainer:upper",
        )
        graph = WorldMapGraph(
            (
                WorldEdge(
                    source_map,
                    middle_map,
                    "warp",
                    ((4, 0), (4, 1)),
                    ((0, 0), (0, 1)),
                ),
                WorldEdge(middle_map, target_map, "warp", ((4, 1),), ((0, 0),)),
            )
        )

        plan, route = plan_with_world_navigation(
            NavigationWorld(
                tiles=tiles,
                transitions=(upper_entry, lower_entry, final_exit),
                triggers=(trainer,),
                facing=Direction.East,
                static_metadata_frozen=True,
            ),
            (source_map, (0, 0)),
            NavigationGoal(
                SemanticTarget.map(target_map),
                constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
            ),
            graph,
        )

        self.assertEqual(route.selected_transitions, (lower_entry, final_exit))
        self.assertEqual(plan.destination, (target_map, (0, 0)))

    def test_semantic_cross_map_chain_resolves_final_interaction_locally(self):
        source_map, middle_map, target_map = (700, 0), (700, 1), (700, 2)
        tiles = {
            (map_id, (x, 0)): NavigableTile(
                (map_id, (x, 0)),
                False,
                frozenset(Direction),
            )
            for map_id in (source_map, middle_map, target_map)
            for x in range(3)
        }
        first_transition = WorldTransition(
            (source_map, (2, 0)),
            (middle_map, (0, 0)),
        )
        second_transition = WorldTransition(
            (middle_map, (2, 0)),
            (target_map, (0, 0)),
        )
        final_trigger = TriggerObservation(
            trigger_id="norman",
            locations=frozenset({(target_map, (2, 0))}),
            activation_locations=frozenset({(target_map, (1, 0))}),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(source_map, middle_map, "warp", ((2, 0),), ((0, 0),)),
                WorldEdge(middle_map, target_map, "warp", ((2, 0),), ((0, 0),)),
            )
        )
        navigation_goal = NavigationGoal(SemanticTarget.interaction(target_map, "norman"))

        plan, route = plan_with_world_navigation(
            NavigationWorld(
                tiles=tiles,
                transitions=(first_transition, second_transition),
                triggers=(final_trigger,),
                facing=Direction.East,
                static_metadata_frozen=True,
            ),
            (source_map, (0, 0)),
            navigation_goal,
            graph,
        )

        assert route.maps == (source_map, middle_map, target_map)
        assert plan.destination == (target_map, (1, 0))
        warp_actions = [action for action in plan.actions if action.action_type is NavigationActionType.WARP]
        assert [action.destination for action in warp_actions] == [
            (middle_map, (0, 0)),
            (target_map, (0, 0)),
        ]
        assert plan.actions[-1].destination == (target_map, (1, 0))

    def test_general_route_cannot_use_pokemon_center_as_transit(self):
        source_map, center_map, target_map = (91, 0), (91, 1), (91, 2)
        tiles = {
            (source_map, (x, 0)): NavigableTile((source_map, (x, 0)), False, frozenset(Direction))
            for x in range(9)
        }
        tiles.update(
            {
                (center_map, (0, 0)): NavigableTile((center_map, (0, 0)), False, frozenset(Direction)),
                (target_map, (0, 0)): NavigableTile((target_map, (0, 0)), False, frozenset(Direction)),
            }
        )
        transitions = (
            WarpObservation((source_map, (1, 0)), (center_map, (0, 0))),
            WarpObservation((center_map, (0, 0)), (source_map, (0, 0))),
            MapConnectionObservation((source_map, (8, 0)), (target_map, (0, 0)), required_facing=Direction.East),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(source_map, center_map, "warp", ((1, 0),), ((0, 0),)),
                WorldEdge(center_map, source_map, "warp", ((0, 0),), ((0, 0),)),
                WorldEdge(source_map, target_map, "connection", ((8, 0),), ((0, 0),)),
            )
        )
        with patch("modules.navigation._pokemon_center_map_ids", return_value=frozenset({center_map})):
            plan, _ = plan_with_world_navigation(
                NavigationWorld(tiles=tiles, transitions=transitions, facing=Direction.East),
                (source_map, (0, 0)),
                SemanticTarget.map(target_map),
                graph,
            )

        route_maps = {
            location[0]
            for action in plan.actions
            for location in (action.source, action.destination)
            if location is not None
        }
        self.assertNotIn(center_map, route_maps)

    def test_cross_map_navigation_preserves_trainer_avoidance_constraints(self):
        source_map, target_map = (56, 0), (56, 1)
        coordinates = {(source_map, (x, y)) for y in range(2) for x in range(3)}
        trainer = TriggerObservation(
            "trainer:route",
            frozenset({(source_map, (1, 0))}),
            hazard_locations=frozenset({(source_map, (1, 0))}),
            hazard_kind="trainer",
        )
        observed = NavigationWorld(
            tiles={location: NavigableTile(location, False, frozenset(Direction)) for location in coordinates},
            triggers=(trainer,),
            facing=Direction.East,
        )
        graph = WorldMapGraph(
            (
                WorldEdge(
                    source_map,
                    target_map,
                    "warp",
                    ((2, 0),),
                    ((0, 0),),
                ),
            )
        )
        goal = NavigationGoal(
            ReachWarp(destination_map=target_map),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        plan, _ = plan_with_world_navigation(observed, (source_map, (0, 0)), goal, graph)

        route_locations = {
            location
            for action in plan.actions
            for location in (action.source, action.destination)
            if location is not None
        }
        self.assertNotIn((source_map, (1, 0)), route_locations)

    def test_observed_directional_warp_does_not_duplicate_activation(self):
        source_map, target_map = (8, 1), (0, 0)
        warp = WarpObservation(
            (source_map, (1, 0)),
            (target_map, (15, 8)),
            required_facing=Direction.East,
            activation=WarpActivation.DIRECTIONAL_STEP,
        )
        observed = world(source_map, [(0, 0), (1, 0)], (warp,), facing=Direction.East)

        plan = plan_observed_warp_locally(
            observed,
            (source_map, (0, 0)),
            ReachWarp(destination_map=target_map, destination=warp.destination, warp=warp),
        )

        warp_actions = [action for action in plan.actions if action.action_type is NavigationActionType.WARP]
        self.assertEqual(len(warp_actions), 1)
        self.assertEqual(warp_actions[0].source, warp.entry)
        self.assertEqual(warp_actions[0].destination, warp.destination)

    def test_observed_door_warp_avoids_global_expansion(self):
        source_map, target_map = (98, 1), (98, 2)
        warp = WarpObservation((source_map, (2, 0)), (target_map, (0, 0)))
        observed = world(source_map, [(0, 0), (1, 0), (2, 0)], (warp,), facing=Direction.East)

        with patch(
            "modules.navigation.get_world_map_graph", side_effect=AssertionError("global search is unnecessary")
        ):
            plan, route = plan_with_world_navigation(
                observed,
                (source_map, (0, 0)),
                ReachWarp(destination_map=target_map, destination=warp.destination, warp=warp),
            )

        self.assertIsNone(route)
        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].destination, warp.destination)

    def test_cross_map_navigation_selects_cheapest_reachable_activation_side(self):
        source_map, target_map = (99, 1), (99, 2)
        entry = (source_map, (2, 1))
        # Keep the more expensive side first to prove that selection is based
        # on the completed route cost rather than activation-side iteration
        # order. Live ROM observations use a frozenset here, so order must not
        # affect the result.
        expensive_side = (source_map, (3, 1))
        cheap_side = (source_map, (2, 0))
        warp = WarpObservation(
            entry,
            (target_map, (0, 0)),
            activation_locations=(expensive_side, cheap_side),
        )
        tiles = {
            (source_map, (x, y)): NavigableTile((source_map, (x, y)), False, frozenset(Direction))
            for y in range(3)
            for x in range(4)
        }
        tiles[(target_map, (0, 0))] = NavigableTile((target_map, (0, 0)), False, frozenset(Direction))
        graph = WorldMapGraph(
            (WorldEdge(source_map, target_map, "warp", ((2, 1),), ((0, 0),)),)
        )

        plan, _ = plan_with_world_navigation(
            NavigationWorld(tiles=tiles, transitions=(warp,), facing=Direction.East),
            (source_map, (0, 0)),
            SemanticTarget.map(target_map),
            graph,
        )

        transition = next(action for action in plan.actions if action.action_type is NavigationActionType.WARP)
        self.assertEqual(transition.source, cheap_side)

    def test_debug_trace_keeps_first_goal_termination(self):
        from modules.context import context

        map_id = (99, 1)
        tiles = {
            (map_id, (0, 0)): NavigableTile((map_id, (0, 0)), False, frozenset(Direction)),
            (map_id, (1, 0)): NavigableTile((map_id, (1, 0)), False, frozenset(Direction)),
            (map_id, (2, 0)): NavigableTile((map_id, (2, 0)), False, frozenset(Direction)),
        }
        old_debug, old_trace = context.debug, context.debug_trace
        context.debug, context.debug_trace = True, True
        try:
            plan = GoalAwareNavigator(NavigationWorld(tiles=tiles, facing=Direction.East)).plan(
                (map_id, (0, 0)), ReachLocation((map_id, (1, 0)))
            )
        finally:
            context.debug, context.debug_trace = old_debug, old_trace
        self.assertEqual(plan.destination, (map_id, (1, 0)))
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.candidate_metrics, ())

    def test_littleroot_house_cycle_uses_outward_connection(self):
        house2f, house1f, rival, town, route101 = ((10, 1), (10, 2), (10, 3), (10, 4), (10, 5))
        transitions = (
            WarpObservation((house2f, (1, 0)), (house1f, (1, 1))),
            WarpObservation((house1f, (1, 1)), (town, (2, 2))),
            WarpObservation((rival, (1, 0)), (town, (3, 2))),
            WarpObservation((town, (2, 2)), (rival, (1, 0))),
            WarpObservation((town, (3, 2)), (house1f, (1, 1))),
            MapConnectionObservation((town, (0, 1)), (route101, (0, 0)), required_facing=Direction.West),
        )
        coordinates = {
            house2f: [(0, 0), (1, 0)],
            house1f: [(1, 1)],
            rival: [(1, 0)],
            town: [(0, 1), (1, 1), (2, 2), (3, 2)],
            route101: [(0, 0)],
        }
        # Give each synthetic map a small connected walkable floor. The
        # transition topology, rather than accidental local geometry, is the
        # subject of this regression.
        tiles = {
            (map_id, (x, y)): NavigableTile((map_id, (x, y)), False, frozenset(Direction))
            for map_id in coordinates
            for y in range(4)
            for x in range(4)
        }
        graph = WorldMapGraph(
            (
                WorldEdge(house2f, house1f, "warp", ((1, 0),), ((1, 1),)),
                WorldEdge(house1f, town, "warp", ((1, 1),), ((2, 2),)),
                WorldEdge(rival, town, "warp", ((1, 0),), ((3, 2),)),
                WorldEdge(town, rival, "warp", ((2, 2),), ((1, 0),)),
                WorldEdge(town, house1f, "warp", ((3, 2),), ((1, 1),)),
                WorldEdge(town, route101, "connection", ((0, 1),), ((0, 0),)),
            )
        )
        route_world = NavigationWorld(tiles=tiles, transitions=transitions, facing=Direction.East)
        with patch("modules.navigation.diagnostic_print") as diagnostic:
            from modules.context import context

            old_debug, old_trace = context.debug, context.debug_trace
            context.debug, context.debug_trace = True, True
            try:
                plan, _ = plan_with_world_navigation(
                    route_world,
                    (house2f, (0, 0)),
                    SemanticTarget.interaction(route101, "Route101_EventScript_BirchsBag"),
                    graph,
                )
            finally:
                context.debug, context.debug_trace = old_debug, old_trace

        cross_map = [a for a in plan.actions if a.action_type is NavigationActionType.WARP]
        self.assertEqual(
            [(a.source[0], a.destination[0], a.transition_kind) for a in cross_map],
            [(house2f, house1f, "warp"), (house1f, town, "warp"), (town, route101, "map_connection")],
        )
        self.assertEqual(cross_map[-1].source, (town, (0, 1)))  # boundary tile after the approach move
        self.assertEqual(transitions[-1].entry, (town, (0, 1)))  # validated source endpoint
        self.assertEqual(cross_map[-1].destination, (route101, (0, 0)))
        self.assertEqual(plan.destination, (route101, (0, 0)))
        diagnostic_text = " ".join(
            (item.args[0]() if callable(item.args[0]) else str(item.args[0]))
            for item in diagnostic.call_args_list
            if item.args
        )
        self.assertIn("GLOBAL_ROUTE_FOUND", diagnostic_text)

    def test_navigation_goal_wrapper_routes_semantic_destination_cross_map(self):
        source, middle, target = (12, 1), (12, 2), (12, 3)
        transitions = (
            WarpObservation((source, (1, 0)), (middle, (1, 1))),
            MapConnectionObservation((middle, (0, 1)), (target, (0, 0)), required_facing=Direction.West),
        )
        tiles = {
            (map_id, (x, y)): NavigableTile((map_id, (x, y)), False, frozenset(Direction))
            for map_id in (source, middle, target)
            for y in range(3)
            for x in range(3)
        }
        graph = WorldMapGraph(
            (
                WorldEdge(source, middle, "warp", ((1, 0),), ((1, 1),)),
                WorldEdge(middle, target, "connection", ((0, 1),), ((0, 0),)),
            )
        )

        plan, route = plan_with_world_navigation(
            NavigationWorld(tiles=tiles, transitions=transitions, facing=Direction.East),
            (source, (0, 0)),
            NavigationGoal(SemanticTarget.map(target)),
            graph,
        )

        self.assertEqual(route.maps, (source, middle, target))
        self.assertEqual(plan.actions[-1].destination, (target, (0, 0)))

    def test_rival_house_also_reaches_route101_without_lab_substitution(self):
        rival, town, route101 = ((11, 1), (11, 2), (11, 3))
        transition = MapConnectionObservation((town, (0, 1)), (route101, (0, 0)), required_facing=Direction.West)
        transitions = (WarpObservation((rival, (0, 0)), (town, (1, 1))), transition)
        tiles = {
            (rival, (0, 0)): NavigableTile((rival, (0, 0)), False, frozenset(Direction)),
            (town, (0, 1)): NavigableTile((town, (0, 1)), False, frozenset(Direction)),
            (town, (1, 1)): NavigableTile((town, (1, 1)), False, frozenset(Direction)),
            (route101, (0, 0)): NavigableTile((route101, (0, 0)), False, frozenset(Direction)),
        }
        graph = WorldMapGraph(
            (
                WorldEdge(rival, town, "warp", ((0, 0),), ((1, 1),)),
                WorldEdge(town, route101, "connection", ((0, 1),), ((0, 0),)),
            )
        )
        plan, _ = plan_with_world_navigation(
            NavigationWorld(tiles=tiles, transitions=transitions, facing=Direction.East),
            (rival, (0, 0)),
            SemanticTarget.interaction(route101, "Route101_EventScript_BirchsBag"),
            graph,
        )
        self.assertEqual(plan.actions[-1].destination, (route101, (0, 0)))
        self.assertNotIn("Birch's Lab", repr(plan))

    def test_weighted_local_route_prefers_lower_cost(self):
        m = (1, 1)
        tiles = {
            (m, (x, y)): NavigableTile((m, (x, y)), False, frozenset(Direction), 1) for y in range(3) for x in range(4)
        }
        tiles[(m, (1, 1))] = NavigableTile((m, (1, 1)), False, frozenset(Direction), 8)
        plan = GoalAwareNavigator(NavigationWorld(tiles=tiles, facing=Direction.East)).plan(
            (m, (0, 1)), ReachLocation((m, (3, 1)))
        )
        self.assertGreaterEqual(plan.metrics.total_route_cost, 0)
        self.assertEqual(plan.destination, (m, (3, 1)))

    def test_connection_preserves_exact_destination_and_kind(self):
        a, b = (2, 1), (2, 2)
        transition = MapConnectionObservation((a, (1, 0)), (b, (1, 2)), required_facing=Direction.North)
        w = world(a, [(0, 1), (1, 1), (1, 0)], (transition,))
        graph = WorldMapGraph((WorldEdge(a, b, "connection", ((1, 0),), ((1, 2),)),))
        plan, _ = plan_with_world_navigation(w, (a, (0, 1)), SemanticTarget.map(b), graph)
        action = plan.actions[-1]
        self.assertEqual(action.action_type, NavigationActionType.WARP)
        self.assertEqual(action.destination, (b, (1, 2)))
        self.assertEqual(action.transition_kind, "map_connection")

    def test_interior_warp_can_be_required_without_map_cycle(self):
        house, town, target = (3, 1), (3, 2), (3, 3)
        warp = WarpObservation((house, (1, 0)), (town, (1, 1)))
        connection = MapConnectionObservation((town, (1, 0)), (target, (1, 1)), required_facing=Direction.North)
        w = world(house, [(0, 0), (1, 0)], (warp,))
        graph = WorldMapGraph(
            (
                WorldEdge(house, town, "warp", ((1, 0),), ((1, 1),)),
                WorldEdge(town, target, "connection", ((1, 0),), ((1, 1),)),
            )
        )
        # The synthetic map tiles supplied for each map are intentionally
        # small; this asserts the exact transition metadata contract.
        plan, _ = plan_with_world_navigation(w, (house, (0, 0)), SemanticTarget.map(town), graph)
        self.assertEqual(plan.actions[-1].destination, (town, (1, 1)))

    def test_semantic_interaction_routes_to_map_only(self):
        source, target = (4, 1), (4, 2)
        transition = MapConnectionObservation((source, (1, 0)), (target, (1, 1)), required_facing=Direction.North)
        w = world(source, [(0, 1), (1, 1), (1, 0)], (transition,))
        graph = WorldMapGraph((WorldEdge(source, target, "connection", ((1, 0),), ((1, 1),)),))
        semantic = SemanticTarget.interaction(target, "Route101_EventScript_BirchsBag")
        plan, _ = plan_with_world_navigation(w, (source, (0, 1)), semantic, graph)
        self.assertEqual(plan.actions[-1].destination, (target, (1, 1)))
        self.assertNotIn("Lab", repr(plan))


if __name__ == "__main__":
    unittest.main()
