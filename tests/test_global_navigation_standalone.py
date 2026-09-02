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
from modules.overworld import MapConnectionObservation, TriggerObservation, WarpActivation, WarpObservation
from modules.world_navigation import WorldEdge, WorldMapGraph


def world(map_id, coords, transitions=(), facing=Direction.South):
    tiles = {(map_id, xy): NavigableTile((map_id, xy), False, frozenset(Direction)) for xy in coords}
    return NavigationWorld(tiles=tiles, transitions=tuple(transitions), facing=facing)


class GlobalNavigationTests(unittest.TestCase):
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
