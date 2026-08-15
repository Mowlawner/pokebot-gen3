from unittest import TestCase
from unittest.mock import Mock, call, patch

from modules.agent_control import (
    ActionResult,
    ActionResultType,
    AgentActionExecutor,
    AgentActionType,
    AgentControlLoop,
    AgentObservation,
    GoalEvaluation,
    GoalStatus,
    available_actions,
    evaluate_goal,
    prewarm_warp_destination,
    select_action,
)
from modules.goals import GoalConstraints, NavigationGoal, ReachLocation
from modules.interaction_state import InteractionObservation
from modules.map_path import Direction
from modules.memory import GameState
from modules.navigation import NavigationAction, NavigationActionType
from modules.overworld import (
    MovementState,
    OverworldObservation,
    TileObservation,
    TriggerObservation,
    WarpObservation,
)
from modules.world_navigation import WorldEdge, WorldMapGraph

MAP = ("test", 0)


def overworld(
    coordinates, start=(0, 0), *, triggers=(), controllable=True, facing=Direction.South, movement_state=None
):
    all_directions = frozenset(Direction)
    return OverworldObservation(
        map_id=MAP,
        player_coordinates=start,
        facing=facing,
        controllable=controllable,
        tiles=tuple(TileObservation((MAP, coordinate), False, all_directions) for coordinate in coordinates),
        warps=(),
        objects=(),
        triggers=tuple(triggers),
        movement_state=movement_state,
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
    def test_known_warp_prewarms_only_its_destination(self):
        destination = ("destination", 0)
        world = OverworldObservation(
            map_id=MAP,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=(TileObservation((MAP, (0, 0)), False, frozenset(Direction)),),
            warps=(WarpObservation((MAP, (0, 0)), (destination, (2, 3))),),
            objects=(),
            triggers=(),
        )
        navigation = NavigationAction(
            NavigationActionType.WARP,
            Direction.East,
            (MAP, (0, 0)),
            (destination, (2, 3)),
        )
        with patch(
            "modules.agent_control.prewarm_static_map_observation", return_value=("static tiles",)
        ) as static_prewarm, patch("modules.agent_control.prewarm_navigation_tiles") as navigation_prewarm:
            self.assertTrue(prewarm_warp_destination(observation(GameState.OVERWORLD, world=world), navigation))
        static_prewarm.assert_called_once_with(destination)
        navigation_prewarm.assert_called_once_with(destination, ("static tiles",))

    def test_unknown_warp_destination_is_not_prewarmed(self):
        destination = ("destination", 0)
        world = overworld({(0, 0)})
        navigation = NavigationAction(
            NavigationActionType.WARP,
            Direction.East,
            (MAP, (0, 0)),
            (destination, (2, 3)),
        )
        with patch("modules.agent_control.prewarm_static_map_observation") as static_prewarm, patch(
            "modules.agent_control.prewarm_navigation_tiles"
        ) as navigation_prewarm:
            self.assertFalse(prewarm_warp_destination(observation(GameState.OVERWORLD, world=world), navigation))
        static_prewarm.assert_not_called()
        navigation_prewarm.assert_not_called()

    def test_dynamic_destination_state_still_comes_from_normal_observation(self):
        destination = ("destination", 0)
        world = OverworldObservation(
            map_id=MAP,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=(TileObservation((MAP, (0, 0)), False, frozenset(Direction)),),
            warps=(WarpObservation((MAP, (0, 0)), (destination, (0, 0))),),
            objects=(),
            triggers=(),
        )
        navigation = NavigationAction(
            NavigationActionType.WARP,
            Direction.East,
            (MAP, (0, 0)),
            (destination, (0, 0)),
        )
        with patch("modules.agent_control.prewarm_static_map_observation", return_value=()) as static_prewarm, patch(
            "modules.agent_control.prewarm_navigation_tiles"
        ):
            prewarm_warp_destination(observation(GameState.OVERWORLD, world=world), navigation)
        self.assertEqual(world.objects, ())
        static_prewarm.assert_called_once_with(destination)

    def test_safe_batch_groups_straight_and_turning_moves(self):
        world = overworld({(x, y) for x, y in ((0, 0), (1, 0), (2, 0), (2, 1))})
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = (
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (0, 0)), (MAP, (1, 0))),
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (1, 0)), (MAP, (2, 0))),
            NavigationAction(NavigationActionType.MOVE, Direction.South, (MAP, (2, 0)), (MAP, (2, 1))),
        )
        batch = loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))
        self.assertEqual([action.direction for action in batch], [Direction.East, Direction.East, Direction.South])

    def test_safe_batch_stops_before_warp_and_at_interaction_checkpoint(self):
        trigger = TriggerObservation(
            "talk",
            frozenset({(MAP, (3, 0))}),
            activation_locations=frozenset({(MAP, (2, 0))}),
        )
        world = overworld({(x, 0) for x in range(5)}, triggers=(trigger,))
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = tuple(
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (x, 0)), (MAP, (x + 1, 0)))
            for x in range(4)
        )
        batch = loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[-1].destination, (MAP, (2, 0)))

    def test_batch_interrupts_on_non_overworld_state(self):
        from modules.agent_control import _MovementBatch

        loop = AgentControlLoop(lambda: None)
        loop._movement_batch = _MovementBatch(())
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.observe_interaction", return_value=InteractionObservation(GameState.BATTLE)
        ):
            self.assertFalse(loop._advance_movement_batch())
        emulator.reset_held_buttons.assert_called_once_with()

    def test_safe_batch_ends_before_warp_action(self):
        source = (0, 0)
        destination = (0, 1)
        world = OverworldObservation(
            map_id=source,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=tuple(TileObservation((source, (x, 0)), False, frozenset(Direction)) for x in range(2)),
            warps=(WarpObservation((source, (1, 0)), (destination, (0, 0))),),
            objects=(),
            triggers=(),
        )
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = (
            NavigationAction(NavigationActionType.MOVE, Direction.East, (source, (0, 0)), (source, (1, 0))),
            NavigationAction(NavigationActionType.WARP, Direction.East, (source, (1, 0)), (destination, (0, 0))),
        )
        self.assertEqual(len(loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))), 1)

    def test_cached_world_signature_ignores_static_tile_topology(self):
        first = observation(
            GameState.OVERWORLD,
            world=overworld({(0, 0), (1, 0)}),
        )
        changed_static_topology = observation(
            GameState.OVERWORLD,
            world=OverworldObservation(
                map_id=MAP,
                player_coordinates=(0, 0),
                facing=Direction.South,
                controllable=True,
                tiles=(
                    TileObservation((MAP, (0, 0)), False, frozenset({Direction.North})),
                    TileObservation((MAP, (1, 0)), False, frozenset({Direction.West})),
                ),
                warps=(),
                objects=(),
                triggers=(),
            ),
        )

        self.assertEqual(
            AgentControlLoop._world_signature(first),
            AgentControlLoop._world_signature(changed_static_topology),
        )

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
        self.assertEqual(
            [action.action_type for action in actions], [AgentActionType.NAVIGATE_MENU, AgentActionType.NAVIGATE_MENU]
        )
        self.assertNotIn(AgentActionType.MOVE, [action.action_type for action in actions])

    def test_battle_delegates_to_battle_system(self):
        action = available_actions(observation(GameState.BATTLE))[0]
        self.assertEqual(action.action_type, AgentActionType.DELEGATE_BATTLE)

    def test_overworld_selects_first_navigation_step_for_reachable_goal(self):
        world = overworld({(0, 0), (1, 0), (2, 0)})
        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=ReachLocation((MAP, (2, 0))),
            )
        )
        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(decision.action.direction, Direction.East)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.REACHABLE)

    def test_completed_goal_waits_without_movement(self):
        world = overworld({(0, 0)})
        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=ReachLocation((MAP, (0, 0))),
            )
        )
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)

    def test_unreachable_goal_waits_and_reports_unreachable(self):
        world = overworld({(0, 0), (1, 0)}, start=(0, 0))
        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=ReachLocation((MAP, (5, 5))),
            )
        )
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
        observation_value = observation(GameState.OVERWORLD, world=world, goal=ReachLocation((MAP, (1, 1))))
        self.assertEqual(select_action(observation_value), select_action(observation_value))


class AgentExecutionTests(TestCase):
    def test_post_warp_settling_defers_goal_evaluation_until_controllable(self):
        source_map = (0, 0)
        target_map = (0, 1)
        goal = ReachLocation((target_map, (0, 0)))

        def target_world(controllable):
            return OverworldObservation(
                map_id=target_map,
                player_coordinates=(0, 0),
                facing=Direction.East,
                controllable=controllable,
                tiles=(TileObservation((target_map, (0, 0)), False, frozenset(Direction)),),
                warps=(),
                objects=(),
                triggers=(),
            )

        observations = iter(
            (
                observation(GameState.OVERWORLD, world=target_world(False), goal=goal),
                observation(GameState.OVERWORLD, world=target_world(True), goal=goal),
            )
        )
        loop = AgentControlLoop(lambda: next(observations))
        loop._expected_world_transition = ((source_map, (1, 0)), (target_map, (0, 0)))
        loop._cached_evaluation = GoalEvaluation(GoalStatus.REACHABLE)

        with patch("modules.agent_control.evaluate_goal", return_value=GoalEvaluation(GoalStatus.COMPLETE)) as evaluate:
            settling = loop.step()
            resumed = loop.step()

        self.assertEqual(settling[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertIsNone(loop._cached_evaluation)
        self.assertEqual(resumed[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        evaluate.assert_called_once()
        self.assertFalse(loop._warp_settling)

    def test_control_loop_retries_move_after_opposite_direction_turn(self):
        observations = iter(
            (
                observation(
                    GameState.OVERWORLD,
                    world=overworld(
                        {(6, 5), (6, 6)},
                        start=(6, 5),
                        facing=Direction.North,
                    ),
                    goal=ReachLocation((MAP, (6, 6))),
                ),
                observation(
                    GameState.OVERWORLD,
                    world=overworld(
                        {(6, 5), (6, 6)},
                        start=(6, 5),
                        facing=Direction.South,
                        movement_state=MovementState.STANDING,
                    ),
                    goal=ReachLocation((MAP, (6, 6))),
                ),
                observation(
                    GameState.OVERWORLD,
                    world=overworld(
                        {(6, 5), (6, 6)},
                        start=(6, 6),
                        facing=Direction.South,
                        movement_state=MovementState.STANDING,
                    ),
                    goal=ReachLocation((MAP, (6, 6))),
                ),
            )
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(lambda: next(observations))
            first = loop.step()
            second = loop.step()
            arrived = loop.step()

        self.assertEqual(first[1].action.direction, Direction.South)
        self.assertEqual(second[1].action.direction, Direction.South)
        self.assertEqual(arrived[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(emulator.press_button.call_args_list, [call("Down"), call("Down")])

    def test_control_loop_executes_selected_movement_and_reobserves(self):
        world = overworld({(0, 0), (1, 0)})
        observe = Mock(return_value=observation(GameState.OVERWORLD, world=world, goal=ReachLocation((MAP, (1, 0)))))
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(observe)
            observed, decision, result = loop.step()
        self.assertEqual(observed.overworld.player_coordinates, (0, 0))
        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        emulator.press_button.assert_called_once_with("Right")
        observe.assert_called_once_with()

    def test_control_loop_reuses_plan_when_observed_position_advances(self):
        observations = iter(
            (
                observation(
                    GameState.OVERWORLD, world=overworld({(0, 0), (1, 0), (2, 0)}), goal=ReachLocation((MAP, (2, 0)))
                ),
                observation(GameState.OVERWORLD, world=overworld({(0, 0), (1, 0), (2, 0)}, start=(1, 0))),
            )
        )
        executor = Mock()
        executor.execute.side_effect = lambda action, _: ActionResult(ActionResultType.EXECUTED, action)
        with patch("modules.agent_control.context.emulator", Mock()), patch(
            "modules.agent_control.evaluate_goal", wraps=evaluate_goal
        ) as evaluate:
            loop = AgentControlLoop(lambda: next(observations), executor)
            first = loop.step()
            second = loop.step()

        self.assertEqual(first[1].action.direction, Direction.East)
        self.assertEqual(second[1].action.direction, Direction.East)
        self.assertEqual(evaluate.call_count, 1)

    def test_control_loop_waits_for_move_destination_before_advancing_cached_plan(self):
        observations = iter(
            (
                observation(
                    GameState.OVERWORLD,
                    world=overworld({(0, 0), (1, 0), (2, 0)}),
                    goal=ReachLocation((MAP, (2, 0))),
                ),
                observation(
                    GameState.OVERWORLD,
                    world=overworld({(0, 0), (1, 0), (2, 0)}),
                    goal=ReachLocation((MAP, (2, 0))),
                ),
                observation(
                    GameState.OVERWORLD,
                    world=overworld({(0, 0), (1, 0), (2, 0)}, start=(1, 0)),
                    goal=ReachLocation((MAP, (2, 0))),
                ),
            )
        )
        emulator = Mock()
        messages = []
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(lambda: next(observations), logger=messages.append)
            first = loop.step()
            settling = loop.step()
            arrived = loop.step()

        self.assertEqual(first[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(first[1].action.navigation.source, (MAP, (0, 0)))
        self.assertEqual(first[1].action.navigation.destination, (MAP, (1, 0)))
        self.assertEqual(settling[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(settling[2].result_type, ActionResultType.WAITING)
        self.assertEqual(arrived[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(arrived[1].action.navigation.source, (MAP, (1, 0)))
        self.assertEqual(emulator.press_button.call_args_list, [call("Right"), call("Right")])
        self.assertFalse(any("plan divergence" in message for message in messages))

    def test_cached_warp_survives_dynamic_world_change_and_waits_for_destination(self):
        source_map = (0, 0)
        target_map = (0, 1)
        goal = ReachLocation((target_map, (0, 0)))

        def source_world(coordinates, dynamic=False):
            return OverworldObservation(
                map_id=source_map,
                player_coordinates=coordinates,
                facing=Direction.East,
                controllable=True,
                tiles=tuple(
                    TileObservation((source_map, coordinate), False, frozenset(Direction))
                    for coordinate in ((0, 0), (1, 0))
                ),
                warps=(
                    WarpObservation(
                        (source_map, (1, 0)),
                        (target_map, (0, 0)),
                        Direction.East,
                    ),
                ),
                objects=(),
                triggers=(TriggerObservation("dynamic", frozenset()),) if dynamic else (),
            )

        target_world = OverworldObservation(
            map_id=target_map,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=(TileObservation((target_map, (0, 0)), False, frozenset(Direction)),),
            warps=(),
            objects=(),
            triggers=(),
        )
        observations = iter(
            (
                observation(GameState.OVERWORLD, world=source_world((0, 0)), goal=goal),
                observation(GameState.OVERWORLD, world=source_world((0, 0)), goal=goal),
                observation(GameState.OVERWORLD, world=source_world((1, 0), dynamic=True), goal=goal),
                observation(GameState.OVERWORLD, world=source_world((1, 0), dynamic=True), goal=goal),
                observation(GameState.OVERWORLD, world=source_world((1, 0), dynamic=True), goal=goal),
                observation(GameState.OVERWORLD, world=target_world, goal=goal),
            )
        )
        graph = WorldMapGraph(
            (
                WorldEdge(
                    source_map=source_map,
                    destination_map=target_map,
                    kind="warp",
                    source_coordinates=((1, 0),),
                    destination_coordinates=((0, 0),),
                ),
            )
        )
        emulator = Mock()
        with patch("modules.navigation.get_world_map_graph", return_value=graph), patch(
            "modules.agent_control.context.emulator", emulator
        ):
            loop = AgentControlLoop(lambda: next(observations))
            first = loop.step()
            settling = loop.step()
            warp = loop.step()
            waiting = loop.step()
            waiting_again = loop.step()
            resumed = loop.step()

        self.assertEqual(first[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(settling[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(warp[1].action.navigation.action_type, NavigationActionType.WARP)
        self.assertEqual(warp[1].action.navigation.source, (source_map, (1, 0)))
        self.assertEqual(warp[1].action.direction, Direction.East)
        self.assertEqual(waiting[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(waiting_again[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(resumed[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(resumed[1].goal_evaluation.status, GoalStatus.COMPLETE)
        self.assertIsNone(loop._expected_world_transition)
        self.assertEqual(emulator.press_button.call_args_list, [call("Right"), call("Right")])

    def test_warp_activation_waits_for_standing_on_warp_source(self):
        source_map = (0, 0)
        target_map = (0, 1)
        goal = ReachLocation((target_map, (0, 0)))

        def source_world(coordinates, movement_state):
            return OverworldObservation(
                map_id=source_map,
                player_coordinates=coordinates,
                facing=Direction.East,
                controllable=True,
                tiles=tuple(
                    TileObservation((source_map, coordinate), False, frozenset(Direction))
                    for coordinate in ((0, 0), (1, 0))
                ),
                warps=(
                    WarpObservation(
                        (source_map, (1, 0)),
                        (target_map, (0, 0)),
                        Direction.East,
                    ),
                ),
                objects=(),
                triggers=(),
                movement_state=movement_state,
            )

        target_world = OverworldObservation(
            map_id=target_map,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=(TileObservation((target_map, (0, 0)), False, frozenset(Direction)),),
            warps=(),
            objects=(),
            triggers=(),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(
                    source_map=source_map,
                    destination_map=target_map,
                    kind="warp",
                    source_coordinates=((1, 0),),
                    destination_coordinates=((0, 0),),
                ),
            )
        )
        observations = iter(
            (
                observation(GameState.OVERWORLD, world=source_world((0, 0), MovementState.STANDING), goal=goal),
                observation(GameState.OVERWORLD, world=source_world((1, 0), MovementState.MOVING), goal=goal),
                observation(GameState.OVERWORLD, world=source_world((1, 0), MovementState.STANDING), goal=goal),
                observation(GameState.OVERWORLD, world=target_world, goal=goal),
            )
        )
        emulator = Mock()
        with patch("modules.navigation.get_world_map_graph", return_value=graph), patch(
            "modules.agent_control.context.emulator", emulator
        ):
            loop = AgentControlLoop(lambda: next(observations))
            move = loop.step()
            deferred = loop.step()
            deferred_transition = loop._expected_world_transition
            deferred_inputs = list(emulator.press_button.call_args_list)
            activated = loop.step()
            completed = loop.step()

        self.assertEqual(move[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(deferred[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(deferred[2].result_type, ActionResultType.WAITING)
        self.assertIsNone(deferred_transition)
        self.assertEqual(deferred_inputs, [call("Right")])
        self.assertIsNone(loop._expected_world_transition)
        self.assertEqual(activated[1].action.navigation.action_type, NavigationActionType.WARP)
        self.assertEqual(activated[2].result_type, ActionResultType.EXECUTED)
        self.assertEqual(completed[1].goal_evaluation.status, GoalStatus.COMPLETE)
        self.assertEqual(emulator.press_button.call_args_list, [call("Right"), call("Right")])

    def test_executor_rejects_input_for_unknown_state(self):
        action = available_actions(observation(GameState.UNKNOWN))[0]
        result = AgentActionExecutor().execute(action, observation(GameState.UNKNOWN))
        self.assertEqual(result.result_type, ActionResultType.WAITING)
