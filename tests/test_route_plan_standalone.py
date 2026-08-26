"""Synthetic Phase 1 RoutePlan checks; no emulator or campaign imports."""

import unittest

from modules.goals import SemanticTarget
from modules.map_path import Direction
from modules.navigation import (
    NavigationAction,
    NavigationActionType,
    NavigationPlan,
    RoutePlan,
)


def action(kind, source, destination, direction=Direction.East, transition_kind=None):
    return NavigationAction(kind, direction, source, destination, transition_kind)


class RoutePlanStandaloneTests(unittest.TestCase):
    def test_reuses_actions_and_preserves_metrics(self):
        m = (1, 1)
        actions = tuple(action(NavigationActionType.MOVE, (m, (x, 0)), (m, (x + 1, 0))) for x in range(3))
        plan = NavigationPlan(actions, (m, (3, 0)))
        route = RoutePlan.from_navigation_plan(plan)
        self.assertIs(route.actions, actions)
        self.assertEqual(route.current_checkpoint, 0)
        self.assertEqual(len(route.checkpoints), 1)
        route.current_checkpoint = 1
        self.assertEqual(route.actions, actions)

    def test_transition_is_a_hard_boundary(self):
        a, b = (2, 1), (2, 2)
        actions = (
            action(NavigationActionType.MOVE, (a, (0, 0)), (a, (1, 0))),
            action(NavigationActionType.WARP, (a, (1, 0)), (b, (0, 2)), transition_kind="map_connection"),
            action(NavigationActionType.MOVE, (b, (0, 2)), (b, (1, 2))),
        )
        route = RoutePlan.from_navigation_plan(NavigationPlan(actions, (b, (1, 2))))
        self.assertEqual(
            [c.kind for c in route.checkpoints], ["transition_precondition", "transition_postcondition", "target_map"]
        )
        self.assertEqual(route.checkpoints[0].expected_coordinate, (1, 0))
        self.assertEqual(route.checkpoints[1].expected_coordinate, (0, 2))
        self.assertEqual(route.checkpoints[1].transition_kind, "map_connection")

    def test_unresolved_interaction_is_map_only(self):
        target_map = (3, 1)
        target = SemanticTarget.interaction(target_map, "Route101_EventScript_BirchsBag")
        plan = NavigationPlan((), (target_map, (9, 9)))
        route = RoutePlan.from_navigation_plan(plan, target)
        self.assertEqual(route.checkpoints[-1].kind, "target_map")
        self.assertIsNone(route.checkpoints[-1].expected_coordinate)
        self.assertEqual(route.checkpoints[-1].expected_map, target_map)

    def test_mismatch_data_is_retained_by_checkpoint(self):
        a, b = (4, 1), (4, 2)
        transition = action(NavigationActionType.WARP, (a, (2, 0)), (b, (3, 4)), transition_kind="warp")
        checkpoint = RoutePlan.from_navigation_plan(NavigationPlan((transition,), transition.destination)).checkpoints[
            1
        ]
        self.assertEqual(checkpoint.expected_map, b)
        self.assertEqual(checkpoint.expected_coordinate, (3, 4))


if __name__ == "__main__":
    unittest.main()
