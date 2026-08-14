from unittest import TestCase
from unittest.mock import Mock, patch

from modules.agent_control import (
    ActionResultType,
    AgentActionExecutor,
    AgentActionType,
    AgentControlLoop,
    AgentObservation,
    GoalStatus,
    available_actions,
    select_action,
)
from modules.goals import GoalConstraints, NavigationGoal, ReachLocation
from modules.interaction_state import InteractionObservation
from modules.map_path import Direction
from modules.memory import GameState
from modules.overworld import OverworldObservation, TileObservation, TriggerObservation


MAP = ("test", 0)


def overworld(coordinates, start=(0, 0), *, triggers=(), controllable=True):
    all_directions = frozenset(Direction)
    return OverworldObservation(
        map_id=MAP,
        player_coordinates=start,
        facing=Direction.South,
        controllable=controllable,
        tiles=tuple(
            TileObservation((MAP, coordinate), False, all_directions)
            for coordinate in coordinates
        ),
        warps=(),
        objects=(),
        triggers=tuple(triggers),
    )


def observation(state, *, dialogue_waiting=False, choices=(), menu=(), world=None, goal=None, metadata=None):
    return AgentObservation(
        InteractionObservation(
            state,
            dialogue_waiting=dialogue_waiting,
            choice_options=tuple(choices),
            menu_options=tuple(menu),
            metadata=metadata or {},
        ),
        overworld=world,
        goal=goal,
    )


class AgentActionSelectionTests(TestCase):
    def test_dialogue_produces_advance_action_only_when_ready(self):
        ready = observation(GameState.OVERWORLD, dialogue_waiting=True)
        not_ready = observation(GameState.OVERWORLD, dialogue_waiting=False)
        self.assertEqual(available_actions(ready)[0].action_type, AgentActionType.ADVANCE_DIALOGUE)
        self.assertEqual(available_actions(not_ready)[0].action_type, AgentActionType.WAIT_REOBSERVE)

    def test_choice_exposes_each_selectable_option(self):
        actions = available_actions(observation(GameState.OVERWORLD, choices=("YES", "NO")))
        self.assertEqual([action.option for action in actions], ["YES", "NO"])
        self.assertTrue(all(action.action_type is AgentActionType.CHOOSE_DIALOGUE_OPTION for action in actions))

    def test_menu_does_not_expose_overworld_movement(self):
        actions = available_actions(observation(GameState.MAIN_MENU, menu=("SAVE", "EXIT")))
        self.assertEqual([action.action_type for action in actions], [
            AgentActionType.NAVIGATE_MENU, AgentActionType.NAVIGATE_MENU
        ])
        self.assertNotIn(AgentActionType.MOVE, [action.action_type for action in actions])

    def test_battle_delegates_to_battle_system(self):
        action = available_actions(observation(GameState.BATTLE))[0]
        self.assertEqual(action.action_type, AgentActionType.DELEGATE_BATTLE)

    def test_overworld_selects_first_navigation_step_for_reachable_goal(self):
        world = overworld({(0, 0), (1, 0), (2, 0)})
        decision = select_action(observation(
            GameState.OVERWORLD,
            world=world,
            goal=ReachLocation((MAP, (2, 0))),
        ))
        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(decision.action.direction, Direction.East)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.REACHABLE)

    def test_completed_goal_waits_without_movement(self):
        world = overworld({(0, 0)})
        decision = select_action(observation(
            GameState.OVERWORLD,
            world=world,
            goal=ReachLocation((MAP, (0, 0))),
        ))
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)

    def test_unreachable_goal_waits_and_reports_unreachable(self):
        world = overworld({(0, 0), (1, 0)}, start=(0, 0))
        decision = select_action(observation(
            GameState.OVERWORLD,
            world=world,
            goal=ReachLocation((MAP, (5, 5))),
        ))
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.UNREACHABLE)

    def test_avoidance_constraint_selects_alternate_route(self):
        bad = TriggerObservation("bad", frozenset({(MAP, (1, 0))}), frozenset({(MAP, (1, 0))}))
        world = overworld({(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)}, triggers=(bad,))
        goal = NavigationGoal(
            ReachLocation((MAP, (2, 0))),
            GoalConstraints(avoid_trigger_ids=frozenset({"bad"})),
        )
        decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))
        self.assertEqual(decision.action.direction, Direction.South)

    def test_special_and_unknown_states_wait(self):
        for state in (GameState.CHANGE_MAP, GameState.UNKNOWN):
            with self.subTest(state=state):
                action = select_action(observation(state)).action
                self.assertEqual(action.action_type, AgentActionType.WAIT_REOBSERVE)

    def test_selection_is_deterministic(self):
        world = overworld({(0, 0), (1, 0), (0, 1), (1, 1)})
        observation_value = observation(
            GameState.OVERWORLD, world=world, goal=ReachLocation((MAP, (1, 1)))
        )
        self.assertEqual(select_action(observation_value), select_action(observation_value))


class AgentExecutionTests(TestCase):
    def test_control_loop_executes_selected_movement_and_reobserves(self):
        world = overworld({(0, 0), (1, 0)})
        observe = Mock(return_value=observation(
            GameState.OVERWORLD, world=world, goal=ReachLocation((MAP, (1, 0)))
        ))
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(observe)
            observed, decision, result = loop.step()
        self.assertEqual(observed.overworld.player_coordinates, (0, 0))
        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        emulator.press_button.assert_called_once_with("Right")
        observe.assert_called_once_with()

    def test_executor_rejects_input_for_unknown_state(self):
        action = available_actions(observation(GameState.UNKNOWN))[0]
        result = AgentActionExecutor().execute(action, observation(GameState.UNKNOWN))
        self.assertEqual(result.result_type, ActionResultType.WAITING)
