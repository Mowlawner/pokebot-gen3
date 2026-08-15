from unittest import TestCase

from modules.agent_control import (
    ActionResult,
    ActionResultType,
    AgentAction,
    AgentActionType,
    AgentControlLoop,
    AgentObservation,
    GoalStatus,
    select_action,
)
from modules.goals import ActivateTrigger, ReachLocation
from modules.interaction_state import InteractionObservation
from modules.map_path import Direction
from modules.memory import GameState
from modules.overworld import OverworldObservation, TileObservation, TriggerObservation


MAP = ("test", 0)


def world(coordinates, start):
    return OverworldObservation(
        map_id=MAP,
        player_coordinates=start,
        facing=Direction.East,
        controllable=True,
        tiles=tuple(
            TileObservation((MAP, coordinate), False, frozenset(Direction))
            for coordinate in coordinates
        ),
        warps=(),
        objects=(),
        triggers=(),
    )


class RecordingExecutor:
    def __init__(self):
        self.actions = []

    def execute(self, action, observation):
        self.actions.append((action, observation))
        if action.action_type is AgentActionType.DELEGATE_BATTLE:
            return ActionResult(ActionResultType.DELEGATED, action)
        return ActionResult(ActionResultType.EXECUTED, action)


class AgentPilotSequenceTests(TestCase):
    def test_semantic_lab_to_rival_battle_sequence(self):
        rival_trigger = TriggerObservation(
            "rival", frozenset({(MAP, (2, 0))}), frozenset({(MAP, (2, 0))}), "object_interaction"
        )
        route_world = lambda start: OverworldObservation(
            map_id=MAP,
            player_coordinates=start,
            facing=Direction.East,
            controllable=True,
            tiles=tuple(
                TileObservation((MAP, coordinate), False, frozenset(Direction))
                for coordinate in ((0, 0), (1, 0), (2, 0))
            ),
            warps=(), objects=(), triggers=(rival_trigger,),
        )
        goal = ActivateTrigger("rival")
        observations = iter((
            AgentObservation(InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True), goal=goal),
            AgentObservation(InteractionObservation(GameState.OVERWORLD), route_world((0, 0))),
            AgentObservation(InteractionObservation(GameState.BATTLE)),
            AgentObservation(InteractionObservation(GameState.OVERWORLD), route_world((1, 0))),
            AgentObservation(InteractionObservation(GameState.OVERWORLD), route_world((2, 0))),
            AgentObservation(InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True)),
            AgentObservation(InteractionObservation(GameState.BATTLE)),
        ))
        executor = RecordingExecutor()
        loop = AgentControlLoop(lambda: next(observations), executor, goal=goal)

        results = [loop.step() for _ in range(7)]

        self.assertEqual(
            [decision.action.action_type for _, decision, _ in results],
            [AgentActionType.ADVANCE_DIALOGUE, AgentActionType.NAVIGATE_TOWARD_GOAL,
             AgentActionType.DELEGATE_BATTLE, AgentActionType.NAVIGATE_TOWARD_GOAL,
             AgentActionType.INTERACT, AgentActionType.ADVANCE_DIALOGUE,
             AgentActionType.DELEGATE_BATTLE],
        )

    def test_goal_survives_wild_battle_and_navigation_resumes(self):
        goal = ReachLocation((MAP, (2, 0)))
        observations = iter((
            AgentObservation(
                InteractionObservation(GameState.OVERWORLD),
                world({(0, 0), (1, 0), (2, 0)}, (0, 0)),
                goal,
            ),
            AgentObservation(InteractionObservation(GameState.BATTLE)),
            AgentObservation(
                InteractionObservation(GameState.OVERWORLD),
                world({(0, 0), (1, 0), (2, 0)}, (1, 0)),
            ),
        ))
        executor = RecordingExecutor()
        logs = []
        loop = AgentControlLoop(lambda: next(observations), executor, logger=logs.append)

        first = loop.step()
        battle = loop.step()
        resumed = loop.step()

        self.assertEqual(first[1].action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(battle[1].action.action_type, AgentActionType.DELEGATE_BATTLE)
        self.assertEqual(resumed[0].goal, goal)
        self.assertEqual(resumed[1].action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(resumed[1].action.direction, Direction.East)
        self.assertEqual(
            [action.action_type for action, _ in executor.actions],
            [AgentActionType.NAVIGATE_TOWARD_GOAL, AgentActionType.DELEGATE_BATTLE,
             AgentActionType.NAVIGATE_TOWARD_GOAL],
        )
        self.assertTrue(any("BATTLE_RETURN" in message for message in logs))

    def test_reaching_interaction_position_is_a_completed_tactical_goal(self):
        trigger = TriggerObservation(
            "rival", frozenset({(MAP, (1, 0))}), frozenset({(MAP, (0, 0))}), "object_interaction"
        )
        observation = AgentObservation(
            InteractionObservation(GameState.OVERWORLD),
            OverworldObservation(
                map_id=MAP,
                player_coordinates=(0, 0),
                facing=Direction.East,
                controllable=True,
                tiles=(
                    TileObservation((MAP, (0, 0)), False, frozenset(Direction)),
                    TileObservation((MAP, (1, 0)), False, frozenset(Direction)),
                ),
                warps=(), objects=(), triggers=(trigger,),
            ),
            ActivateTrigger("rival"),
        )
        decision = select_action(observation)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.REACHABLE)
        self.assertEqual(decision.action.action_type, AgentActionType.INTERACT)

    def test_battle_action_is_not_replaced_by_overworld_input(self):
        decision = select_action(AgentObservation(InteractionObservation(GameState.BATTLE)))
        self.assertEqual(decision.action, AgentAction(
            AgentActionType.DELEGATE_BATTLE, reason="BattleListener owns battle execution"
        ))
