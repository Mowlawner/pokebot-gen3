from dataclasses import replace
import unittest
from unittest.mock import patch

from modules.agent_control import (
    ActionDecision,
    ActionResultType,
    AgentAction,
    AgentActionExecutor,
    AgentActionType,
    AgentControlLoop,
    AgentObservation,
)
from modules.interaction_state import InteractionObservation
from modules.map_path import Direction
from modules.memory import GameState
from modules.navigation import NavigationAction, NavigationActionType
from modules.overworld import OverworldObservation, TileObservation


class _Emulator:
    def __init__(self):
        self.fresh = []
        self.pressed = []
        self.held = []
        self.released = []
        self.active = set()
        self.events = []

    def press_button_fresh(self, button):
        self.fresh.append(button)
        self.events.append(("fresh", button, frozenset(self.active)))

    def press_button(self, button):
        self.pressed.append(button)
        self.events.append(("press", button, frozenset(self.active)))

    def hold_button(self, button):
        self.held.append(button)
        self.active.add(button)
        self.events.append(("hold", button, frozenset(self.active)))

    def release_button(self, button):
        self.released.append(button)
        self.active.discard(button)


def observation(map_id=(1, 0), position=(8, 8), controllable=True, *, valid_tile=False):
    return AgentObservation(
        InteractionObservation(GameState.OVERWORLD, controllable=controllable),
        OverworldObservation(
            map_id=map_id,
            player_coordinates=position,
            facing=Direction.South,
            controllable=controllable,
            tiles=(TileObservation((map_id, position), False, frozenset(Direction)),) if valid_tile else (),
            warps=(),
            objects=(),
            triggers=(),
        ),
    )


def transition(kind="warp", destination=((0, 9), (1, 1))):
    return NavigationAction(
        NavigationActionType.WARP,
        Direction.South,
        ((1, 0), (8, 8)),
        destination,
        transition_kind=kind,
    )


class TestTransitionExecution(unittest.TestCase):
    def test_transition_signal_prevents_premature_watchdog(self):
        action = transition()
        base = observation()
        signaled = replace(
            base,
            overworld=replace(
                base.overworld,
                transition_in_progress=True,
                transition_signals=frozenset({"task:Task_WarpAndLoadMap"}),
            ),
        )
        loop, emulator, selected = self.make_loop((base, signaled), action)
        loop._pending_transition_watchdog_limit = 1
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ):
            loop.step()
            _, decision, result = loop.step()

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(result.result_type, ActionResultType.WAITING)
        self.assertIsNotNone(loop._pending_transition)

    def test_rom_control_releases_bot_direction_before_control_returns(self):
        emulator = _Emulator()
        emulator.active.add("Down")
        source = iter([observation(controllable=False), observation(position=(5, 9), controllable=True)])
        loop = AgentControlLoop(lambda: next(source), AgentActionExecutor())
        resumed = ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=resumed
        ):
            loop.step()
            loop.step()
        self.assertNotIn("Down", emulator.active)
        self.assertIn("Down", emulator.released)

    def make_loop(self, observations, action):
        emulator = _Emulator()
        source = iter(observations)
        selected = ActionDecision(
            # No evaluation means the test is only about execution lifecycle.
            action=AgentAction(
                AgentActionType.NAVIGATE_TOWARD_GOAL,
                direction=action.direction,
                navigation=action,
            )
        )
        loop = AgentControlLoop(lambda: next(source), AgentActionExecutor())
        return loop, emulator, selected

    def test_directional_dispatch_is_fresh_once_then_continues(self):
        action = transition()
        loop, emulator, selected = self.make_loop([observation()] * 4, action)
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ):
            loop.step()
            loop.step()
            loop.step()
        self.assertEqual(emulator.fresh, ["Down"])
        self.assertEqual(emulator.held, ["Down", "Down"])
        self.assertIsNotNone(loop._pending_transition)

    def test_player_motion_does_not_replan_before_destination(self):
        action = transition()
        loop, emulator, selected = self.make_loop(
            [observation(), observation(position=(8, 9)), observation(position=(8, 9))], action
        )
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ) as select:
            loop.step()
            loop.step()
            loop.step()
        self.assertEqual(select.call_count, 1)
        self.assertTrue(loop._pending_transition.moved)

    def test_destination_observation_clears_pending_transition(self):
        action = transition()
        loop, emulator, selected = self.make_loop([observation(), observation(map_id=(0, 9), valid_tile=True)], action)
        resumed = ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", side_effect=(selected, resumed)
        ):
            loop.step()
            loop.step()
        self.assertIsNone(loop._pending_transition)
        self.assertEqual(emulator.released, ["Down"])

    def test_transient_destination_coordinate_waits_without_runtime_override(self):
        action = transition(destination=((0, 9), (5, 8)))
        transient = observation(map_id=(0, 9), position=(19, -1))
        arrived = observation(map_id=(0, 9), position=(5, 8), valid_tile=True)
        loop, emulator, selected = self.make_loop([observation(), transient, arrived], action)
        resumed = ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", side_effect=(selected, resumed)
        ), patch("modules.agent_control.record_runtime_transition_observation") as record:
            loop.step()
            settling = loop.step()
            self.assertIsNotNone(loop._pending_transition)
            self.assertEqual(settling[2].result_type, ActionResultType.WAITING)
            self.assertEqual(settling[1].action.reason, "transition destination coordinate settling")
            loop.step()

        self.assertIsNone(loop._pending_transition)
        record.assert_not_called()
        self.assertEqual(emulator.released, ["Down", "Down"])

    def test_destination_releases_warp_direction_before_first_tactical_input(self):
        action = transition(destination=((0, 9), (5, 8)))
        first_tactical_move = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            ((0, 9), (5, 10)),
            ((0, 9), (6, 10)),
        )
        resumed = ActionDecision(
            AgentAction(
                AgentActionType.NAVIGATE_TOWARD_GOAL,
                direction=Direction.East,
                navigation=first_tactical_move,
            )
        )
        destination = observation(map_id=(0, 9), position=(5, 10), valid_tile=True)
        loop, emulator, selected = self.make_loop([observation(), destination, destination], action)
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", side_effect=(selected, resumed)
        ):
            loop.step()
            loop.step()
            loop.step()

        self.assertEqual(emulator.released, ["Down"])
        self.assertEqual(emulator.pressed, ["Right"])
        right_event = next(event for event in emulator.events if event[:2] == ("press", "Right"))
        self.assertNotIn("Down", right_event[2])
        self.assertEqual(first_tactical_move.source, ((0, 9), (5, 10)))

    def test_transition_release_does_not_clear_unrelated_held_input(self):
        action = transition()
        loop, emulator, selected = self.make_loop([observation(), observation(map_id=(0, 9))], action)
        emulator.active.add("A")
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action",
            side_effect=(selected, ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))),
        ):
            loop.step()
            loop.step()
        self.assertEqual(emulator.active, {"A"})

    def test_stationary_transition_eventually_reports_blocked(self):
        action = transition()
        loop, emulator, selected = self.make_loop([observation()] * 11, action)
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ):
            loop.step()
            results = [loop.step()[2] for _ in range(10)]
        failures = [result for result in results if result.result_type is ActionResultType.UNREACHABLE]
        self.assertEqual(failures, [])
        self.assertEqual(emulator.released, [])

    def test_destination_mismatch_releases_map_connection_direction(self):
        action = transition("map_connection", destination=((0, 9), (1, 1)))
        loop, emulator, selected = self.make_loop([observation(), observation(map_id=(2, 3))], action)
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action",
            side_effect=(selected, ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))),
        ):
            loop.step()
            loop.step()

        self.assertIsNone(loop._pending_transition)
        self.assertEqual(emulator.released, ["Down"])

    def test_map_connection_uses_same_pending_lifecycle(self):
        action = transition("map_connection")
        loop, emulator, selected = self.make_loop([observation(), observation()] * 2, action)
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ):
            loop.step()
            loop.step()
        self.assertEqual(emulator.fresh, ["Down"])
        self.assertEqual(emulator.held, ["Down"])
        self.assertEqual(loop._pending_transition.transition_kind, "map_connection")

    def test_map_connection_releases_direction_at_destination_without_changing_pending_behavior(self):
        action = transition("map_connection")
        loop, emulator, selected = self.make_loop([observation(), observation(map_id=(0, 9), valid_tile=True)], action)
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action",
            side_effect=(selected, ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))),
        ):
            loop.step()
            self.assertIsNotNone(loop._pending_transition)
            loop.step()
        self.assertIsNone(loop._pending_transition)
        self.assertEqual(emulator.released, ["Down"])

    def test_evicted_warp_loop_releases_only_its_pending_direction(self):
        action = transition()
        loop, emulator, selected = self.make_loop([observation()], action)
        runner = loop.run()
        emulator.active.add("A")
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ):
            next(runner)
            runner.close()
            runner.close()

        self.assertEqual(emulator.released, ["Down"])
        self.assertEqual(emulator.active, {"A"})
        self.assertIsNone(loop._pending_transition)

    def test_evicted_map_connection_releases_down_before_new_right_input(self):
        action = transition("map_connection")
        loop, emulator, selected = self.make_loop([observation()], action)
        runner = loop.run()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ):
            next(runner)
            runner.close()
            AgentActionExecutor().execute(
                AgentAction(AgentActionType.MOVE, direction=Direction.East),
                observation(map_id=(0, 9), position=(5, 10)),
            )

        self.assertEqual(emulator.released, ["Down"])
        right_event = next(event for event in emulator.events if event[:2] == ("press", "Right"))
        self.assertNotIn("Down", right_event[2])

    def test_unknown_destination_is_confirmed_by_any_map_change(self):
        action = transition(destination=None)
        loop, emulator, selected = self.make_loop([observation(), observation(map_id=(2, 3))], action)
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action",
            side_effect=(selected, ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE))),
        ):
            loop.step()
            loop.step()
        self.assertIsNone(loop._pending_transition)


if __name__ == "__main__":
    unittest.main()
