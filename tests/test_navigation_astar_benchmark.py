"""Deterministic Dijkstra/A* equivalence and small benchmark fixtures.

These fixtures model the same observable contracts as the live planner.  ROM
and emulator timing are intentionally not assumed here; the benchmark output
is useful for algorithmic node counts, while wall-clock results are reported
as local Python measurements only.
"""

import statistics
import time
import unittest

from modules.context import context
from modules.goals import ActivateTrigger, EncounterMode, NavigationGoal, ReachLocation, ReachWarp
from modules.map_path import Direction
from modules.navigation import (
    GoalAwareNavigator,
    NavigationWorld,
    NavigableTile,
    NavigationMetrics,
    navigation_candidate_key,
    plan_with_world_navigation,
)
from modules.overworld import TriggerObservation, WarpObservation
from modules.profiler import (
    clear_cross_map_goal_evaluations,
    clear_pathfinding_searches,
    cross_map_goal_evaluations,
    pathfinding_searches,
)
from modules.world_navigation import WorldEdge, WorldMapGraph

MAP = (990, 0)


def grid(width, height, grass=(), blocked=(), facing=Direction.East):
    grass = set(grass)
    blocked = set(blocked)
    return NavigationWorld(
        tiles={
            (MAP, (x, y)): NavigableTile(
                (MAP, (x, y)),
                (x, y) in blocked,
                frozenset(Direction),
                2 if (x, y) in grass else 1,
                (x, y) in grass,
            )
            for y in range(height)
            for x in range(width)
        },
        facing=facing,
    )


class TestAStarComparison(unittest.TestCase):
    def compare(self, world, start, goal):
        dijkstra = GoalAwareNavigator(world).plan(start, goal, algorithm="dijkstra")
        astar = GoalAwareNavigator(world).plan(start, goal, algorithm="astar")
        self.assertEqual(dijkstra.actions, astar.actions)
        self.assertEqual(dijkstra.metrics.encounter_opportunities, astar.metrics.encounter_opportunities)
        self.assertEqual(dijkstra.metrics.total_route_cost, astar.metrics.total_route_cost)
        return dijkstra, astar

    def test_named_route_fixtures_preserve_optimal_results(self):
        fixtures = (
            ("littleroot_route101", grid(12, 3), (MAP, (0, 1)), ReachLocation((MAP, (11, 1)))),
            (
                "first_route101_grass_patch",
                grid(8, 3, grass={(2, 1), (3, 1)}, facing=Direction.East),
                (MAP, (0, 1)),
                ReachLocation((MAP, (7, 1))),
            ),
            (
                "final_grass_patch",
                grid(8, 4, grass={(x, 2) for x in range(2, 6)}, facing=Direction.East),
                (MAP, (0, 2)),
                ReachLocation((MAP, (7, 2))),
            ),
            (
                "may_from_multiple_sides",
                NavigationWorld(
                    tiles={
                        (MAP, (x, y)): NavigableTile((MAP, (x, y)), allowed_directions=frozenset(Direction))
                        for x in range(3)
                        for y in range(3)
                    },
                    triggers=(
                        TriggerObservation(
                            "may",
                            frozenset({(MAP, (1, 1))}),
                            frozenset({(MAP, (1, 0)), (MAP, (2, 1)), (MAP, (1, 2)), (MAP, (0, 1))}),
                            activation_requirements=(
                                ((MAP, (1, 0)), Direction.South),
                                ((MAP, (2, 1)), Direction.West),
                                ((MAP, (1, 2)), Direction.North),
                                ((MAP, (0, 1)), Direction.East),
                            ),
                        ),
                    ),
                    facing=Direction.North,
                ),
                (MAP, (0, 0)),
                ActivateTrigger("may"),
            ),
            (
                "facing_constrained_interaction",
                NavigationWorld(
                    tiles={
                        (MAP, (x, 0)): NavigableTile((MAP, (x, 0)), allowed_directions=frozenset(Direction))
                        for x in range(4)
                    },
                    triggers=(
                        TriggerObservation(
                            "npc",
                            frozenset({(MAP, (3, 0))}),
                            frozenset({(MAP, (2, 0))}),
                            activation_requirements=(((MAP, (2, 0)), Direction.East),),
                        ),
                    ),
                    facing=Direction.North,
                ),
                (MAP, (0, 0)),
                ActivateTrigger("npc"),
            ),
            (
                "seek_grass_route",
                grid(8, 3, grass={(x, 1) for x in range(1, 7)}, facing=Direction.East),
                (MAP, (0, 1)),
                NavigationGoal(ReachLocation((MAP, (7, 1))), encounter_mode=EncounterMode.SEEK),
            ),
        )
        for name, world, start, goal in fixtures:
            with self.subTest(name=name):
                self.compare(world, start, goal)

    def test_profile_records_search_shape(self):
        previous = getattr(context, "debug_profile", False)
        context.debug_profile = True
        try:
            clear_pathfinding_searches()
            world = grid(10, 5, grass={(4, 2), (5, 2)})
            GoalAwareNavigator(world).plan((MAP, (0, 2)), ReachLocation((MAP, (9, 2))), algorithm="astar")
            record = pathfinding_searches()[-1]
            for field in (
                "algorithm",
                "map",
                "start_state",
                "goal_type",
                "nodes_expanded",
                "nodes_generated",
                "max_frontier_size",
                "duration_ms",
                "path_found",
                "final_encounter_opportunity_count",
                "final_route_cost",
                "candidate_searches_for_goal",
            ):
                self.assertIn(field, record)
            self.assertEqual(record["algorithm"], "astar")
            self.assertTrue(record["path_found"])
        finally:
            context.debug_profile = previous

    def test_cross_map_candidate_fixture_records_each_search(self):
        source = (1, 0)
        target = (2, 0)
        world = NavigationWorld(
            tiles={(source, (x, 0)): NavigableTile((source, (x, 0))) for x in range(5)},
            warps=tuple(WarpObservation((source, (x, 0)), (target, (0, 0))) for x in (1, 3)),
            facing=Direction.East,
        )
        graph = WorldMapGraph((WorldEdge(source, target, "warp", ((1, 0), (3, 0)), ((0, 0), (0, 0))),))
        previous = getattr(context, "debug_profile", False)
        context.debug_profile = True
        try:
            clear_pathfinding_searches()
            clear_cross_map_goal_evaluations()
            plan_with_world_navigation(
                world, (source, (0, 0)), ReachLocation((target, (0, 0))), graph, algorithm="astar"
            )
            records = pathfinding_searches()
            # Semantic cross-map goals are now solved by one global frontier;
            # the planner emits a normal search record rather than the old
            # per-transition candidate audit.
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["candidate_searches_for_goal"], 1)
            self.assertEqual(records[0]["goal_type"], "SemanticTarget")
            self.assertTrue(records[0]["path_found"])
            self.assertEqual(cross_map_goal_evaluations(), ())
        finally:
            context.debug_profile = previous

    def test_normal_candidate_selection_is_opportunity_first(self):
        encounter_heavy = NavigationMetrics(1, 1, 1, 1, 0, 0, Direction.East, 1)
        longer_clear = NavigationMetrics(5, 5, 0, 0, 0, 0, Direction.East, 8)
        self.assertLess(
            navigation_candidate_key(longer_clear, EncounterMode.NORMAL, "B"),
            navigation_candidate_key(encounter_heavy, EncounterMode.NORMAL, "A"),
        )

    def test_shared_frontier_benchmark_reduces_searches_and_nodes(self):
        source = (11, 0)
        target = (12, 0)
        world = NavigationWorld(
            tiles={(source, (x, 0)): NavigableTile((source, (x, 0))) for x in range(10)},
            warps=tuple(WarpObservation((source, (x, 0)), (target, (0, 0))) for x in (2, 5, 8)),
            facing=Direction.East,
        )
        graph = WorldMapGraph((WorldEdge(source, target, "warp", ((2, 0), (5, 0), (8, 0)), ((0, 0),) * 3),))
        activation_targets = tuple((source, (x, 0)) for x in (1, 3, 4, 6, 7, 9))
        previous = getattr(context, "debug_profile", False)
        context.debug_profile = True
        try:
            clear_pathfinding_searches()
            for activation_target in activation_targets:
                GoalAwareNavigator(world).plan((source, (0, 0)), ReachLocation(activation_target))
            old_records = pathfinding_searches()
            old_nodes = sum(record["nodes_expanded"] for record in old_records)

            clear_pathfinding_searches()
            plan_with_world_navigation(world, (source, (0, 0)), ReachLocation((target, (0, 0))), graph)
            new_records = pathfinding_searches()
            new_nodes = sum(record["nodes_expanded"] for record in new_records)
            print(
                "shared-frontier benchmark: "
                f"old searches={len(old_records)}, old expanded={old_nodes}; "
                f"new searches={len(new_records)}, new expanded={new_nodes}; "
                f"new pathfinding_ms={sum(record['duration_ms'] for record in new_records):.3f}"
            )
            self.assertGreater(len(old_records), len(new_records))
            self.assertGreater(old_nodes, new_nodes)
        finally:
            context.debug_profile = previous

    def test_escalator_and_cross_map_goal_preserve_activation_source(self):
        source = (3, 0)
        target = (4, 0)
        escalator = WarpObservation(
            (source, (3, 0)),
            (target, (0, 0)),
            required_facing=Direction.West,
            activation_locations=frozenset({(source, (4, 0))}),
            activation_direction=Direction.West,
        )
        world = NavigationWorld(
            tiles={(source, (x, 0)): NavigableTile((source, (x, 0))) for x in range(5)},
            warps=(escalator,),
            facing=Direction.East,
        )
        graph = WorldMapGraph((WorldEdge(source, target, "warp", ((3, 0),), ((0, 0),)),))
        dijkstra, _ = plan_with_world_navigation(
            world, (source, (0, 0)), ReachLocation((target, (0, 0))), graph, algorithm="dijkstra"
        )
        astar, _ = plan_with_world_navigation(
            world, (source, (0, 0)), ReachLocation((target, (0, 0))), graph, algorithm="astar"
        )
        self.assertEqual(dijkstra.actions, astar.actions)
        self.assertEqual(dijkstra.actions[-1].direction, Direction.West)
        self.assertEqual(dijkstra.actions[-1].source, (source, (3, 0)))


if __name__ == "__main__":
    unittest.main()
