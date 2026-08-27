"""Synthetic regression tests for executable ReachWarp transitions."""

import unittest

from modules.goals import ReachWarp, SemanticTarget
from modules.map_path import Direction
from modules.navigation import GoalAwareNavigator, NavigationActionType, NavigationWorld, NavigableTile
from modules.overworld import MapConnectionObservation, WarpActivation, WarpObservation


def tiles(*locations):
    return {location: NavigableTile(location, False, frozenset(Direction)) for location in locations}


class ReachWarpActivationTests(unittest.TestCase):
    def test_grouped_warp_goal_does_not_match_unrelated_transition(self):
        source, target, unrelated = (7, 1), (8, 1), (9, 1)
        selected = WarpObservation((source, (1, 0)), (target, (0, 0)))
        other = WarpObservation((source, (0, 0)), (unrelated, (0, 0)))
        world = NavigationWorld(
            tiles((source, (0, 0)), (source, (1, 0))),
            transitions=(selected, other),
            facing=Direction.East,
        )
        goal = ReachWarp(destination_map=target, warps=(selected,))
        self.assertFalse(GoalAwareNavigator(world).satisfies((source, (0, 0)), Direction.East, goal))

    def test_entry_is_not_terminal_without_explicit_warp_action(self):
        source, destination = (1, 1), (2, 1)
        warp = WarpObservation((source, (1, 0)), (destination, (0, 0)))
        world = NavigationWorld(tiles((source, (0, 0)), (source, (1, 0))), transitions=(warp,), facing=Direction.East)
        plan = GoalAwareNavigator(world).plan(
            (source, (0, 0)), ReachWarp(destination_map=destination, destination=warp.destination, warp=warp)
        )
        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].source, warp.entry)
        self.assertEqual(plan.actions[-1].destination, warp.destination)
        self.assertEqual(plan.destination, warp.destination)

    def test_directional_warp_preserves_activation_direction(self):
        source, destination = (3, 1), (4, 1)
        warp = WarpObservation(
            (source, (1, 0)),
            (destination, (0, 0)),
            required_facing=Direction.South,
            activation=WarpActivation.DIRECTIONAL_STEP,
            activation_direction=Direction.South,
        )
        world = NavigationWorld(tiles((source, (1, 0))), transitions=(warp,), facing=Direction.South)
        plan = GoalAwareNavigator(world).plan(
            (source, (1, 0)), ReachWarp(destination_map=destination, destination=warp.destination, warp=warp)
        )
        self.assertEqual(plan.actions[-1].direction, Direction.South)

    def test_map_connection_uses_approach_as_warp_source(self):
        source, destination = (5, 1), (6, 1)
        connection = MapConnectionObservation((source, (2, 0)), (destination, (1, 1)), required_facing=Direction.North)
        approach = (source, (2, 1))
        world = NavigationWorld(tiles(approach), transitions=(connection,), facing=Direction.North)
        plan = GoalAwareNavigator(world).plan(
            approach, ReachWarp(destination_map=destination, destination=connection.destination, warp=connection)
        )
        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].source, approach)
        self.assertEqual(plan.actions[-1].destination, connection.destination)

    def test_semantic_map_target_does_not_become_reach_warp(self):
        target = SemanticTarget.interaction((101, 1), "Route101_EventScript_BirchsBag")
        self.assertIsNone(target.location)
        self.assertEqual(target.target_map, (101, 1))


if __name__ == "__main__":
    unittest.main()
