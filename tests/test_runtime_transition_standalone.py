"""Session-local transition reconciliation tests; no emulator/campaign imports."""

import unittest

from modules.goals import SemanticTarget
from modules.navigation import (
    GoalAwareNavigator,
    NavigationActionType,
    NavigationWorld,
    NavigableTile,
    clear_runtime_transition_observations,
    effective_transition,
    record_runtime_transition_observation,
    runtime_transition_observation,
)
from modules.goals import ReachWarp
from modules.overworld import WarpActivation, WarpObservation, MapConnectionObservation
from modules.map_path import Direction


class RuntimeTransitionTests(unittest.TestCase):
    def setUp(self):
        clear_runtime_transition_observations()

    def tearDown(self):
        clear_runtime_transition_observations()

    def test_matching_destination_keeps_effective_route(self):
        transition = WarpObservation(((1, 1), (0, 0)), ((2, 1), (1, 1)))
        record_runtime_transition_observation(
            transition.entry, transition.kind, transition.destination, transition.destination
        )
        self.assertEqual(effective_transition(transition).destination, ((2, 1), (1, 1)))

    def test_mismatch_overrides_without_mutating_static_transition(self):
        transition = WarpObservation(((1, 1), (0, 0)), ((2, 1), (1, 1)))
        actual = ((3, 1), (4, 2))
        record_runtime_transition_observation(transition.entry, transition.kind, transition.destination, actual)
        self.assertEqual(transition.destination, ((2, 1), (1, 1)))
        self.assertEqual(effective_transition(transition).destination, actual)
        self.assertEqual(
            runtime_transition_observation(transition.entry, transition.kind).predicted_destination,
            transition.destination,
        )

    def test_identity_distinguishes_multiple_exits(self):
        first = WarpObservation(((1, 1), (0, 0)), ((2, 1), (1, 1)))
        second = WarpObservation(((1, 1), (3, 0)), ((4, 1), (1, 1)))
        record_runtime_transition_observation(first.entry, first.kind, first.destination, ((5, 1), (2, 2)))
        self.assertEqual(effective_transition(first).destination, ((5, 1), (2, 2)))
        self.assertEqual(effective_transition(second).destination, second.destination)

    def test_map_connection_uses_approach_identity(self):
        static = MapConnectionObservation(((1, 1), (5, 0)), ((2, 1), (1, 1)), required_facing=Direction.North)
        approach = ((1, 1), (5, 1))
        actual = ((3, 1), (2, 2))
        record_runtime_transition_observation(approach, static.kind, static.destination, actual)
        self.assertEqual(effective_transition(static).destination, actual)

    def test_override_survives_replanning_and_is_session_scoped(self):
        transition = WarpObservation(((1, 1), (0, 0)), ((2, 1), (1, 1)))
        observed = ((3, 1), (8, 3))
        record_runtime_transition_observation(transition.entry, transition.kind, transition.destination, observed)
        self.assertEqual(effective_transition(transition).destination, observed)
        clear_runtime_transition_observations()
        self.assertEqual(effective_transition(transition).destination, transition.destination)

    def test_override_preserves_arrow_warp_activation_metadata(self):
        transition = WarpObservation(
            ((1, 1), (4, 5)),
            ((2, 1), (8, 9)),
            required_facing=Direction.North,
            activation_direction=Direction.North,
            activation=WarpActivation.DIRECTIONAL_STEP,
        )
        observed = ((3, 1), (2, 2))
        record_runtime_transition_observation(transition.entry, transition.kind, transition.destination, observed)

        reconciled = effective_transition(transition)

        self.assertIsInstance(reconciled, WarpObservation)
        self.assertEqual(reconciled.destination, observed)
        self.assertEqual(reconciled.required_facing, Direction.North)
        self.assertEqual(reconciled.activation, WarpActivation.DIRECTIONAL_STEP)

    def test_reconciled_arrow_warp_still_dispatches_required_direction(self):
        transition = WarpObservation(
            ((1, 1), (4, 5)),
            ((2, 1), (8, 9)),
            activation_direction=Direction.North,
            activation=WarpActivation.DIRECTIONAL_STEP,
        )
        observed = transition.destination
        record_runtime_transition_observation(transition.entry, transition.kind, transition.destination, observed)
        reconciled = effective_transition(transition)
        world = NavigationWorld(
            {transition.entry: NavigableTile(transition.entry)},
            transitions=(reconciled,),
            facing=Direction.South,
        )

        plan = GoalAwareNavigator(world).plan(
            transition.entry,
            ReachWarp(destination_map=observed[0], destination=observed, warp=transition),
        )

        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].direction, Direction.North)

    def test_unresolved_interaction_remains_semantic_target(self):
        target = SemanticTarget.interaction((101, 1), "Route101_EventScript_BirchsBag")
        self.assertIsNone(target.location)
        self.assertEqual(target.target_map, (101, 1))


if __name__ == "__main__":
    unittest.main()
