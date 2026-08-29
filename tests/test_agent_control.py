from unittest import TestCase
from unittest.mock import Mock, call, patch
from dataclasses import replace
from types import SimpleNamespace

import modules.agent_control as agent_control
from modules.agent_control import (
    ActionResult,
    ActionResultType,
    AgentAction,
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
    observe_agent,
)
from modules.goals import GoalConstraints, NavigationGoal, ReachLocation, ReachWarp
from modules.interaction_state import InteractionObservation, InteractionPhase
from modules.map_path import Direction
from modules.memory import GameState
from modules.navigation import NavigationAction, NavigationActionType, NavigationPlan, NavigationError
from modules.overworld import (
    MovementState,
    OverworldObservation,
    TileObservation,
    TriggerObservation,
    WarpObservation,
    WarpActivation,
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


def observation(
    state,
    *,
    dialogue_waiting=False,
    choices=(),
    menu=(),
    world=None,
    goal=None,
    metadata=None,
    controllable=True,
    field_message_lifecycle_active=False,
    field_message_advance_ready=False,
):
    return AgentObservation(
        InteractionObservation(
            state,
            dialogue_waiting=dialogue_waiting,
            choice_options=tuple(choices),
            menu_options=tuple(menu),
            metadata=metadata or {},
            controllable=controllable,
            field_message_lifecycle_active=field_message_lifecycle_active,
            field_message_advance_ready=field_message_advance_ready,
            interaction_phase=(
                InteractionPhase.FIELD_MESSAGE_INPUT_WAIT
                if dialogue_waiting or field_message_advance_ready
                else InteractionPhase.NONE
            ),
        ),
        overworld=world,
        goal=goal,
    )


class AgentActionSelectionTests(TestCase):
    def test_script_owned_controllable_overworld_waits_instead_of_reinteracting(self):
        world = overworld({(0, 0)}, controllable=True)
        observed = observation(
            GameState.OVERWORLD,
            world=world,
            goal=ReachLocation((MAP, (0, 0))),
        )
        observed = replace(observed, interaction=InteractionObservation(
            GameState.OVERWORLD,
            controllable=True,
            script_active=True,
            script_function="SomeInteraction",
            native_function="WaitForEffect",
        ))
        decision = select_action(observed)
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)

    def test_observed_warp_uses_local_plan_when_global_route_is_unreachable(self):
        destination = (("destination", 0), (10, 19))
        warp = WarpObservation((MAP, (1, 0)), destination, required_facing=Direction.East)
        world = OverworldObservation(
            map_id=MAP,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=tuple(
                TileObservation(
                    (MAP, coordinate), False, frozenset(Direction), warp=warp if coordinate == (1, 0) else None
                )
                for coordinate in ((0, 0), (1, 0))
            ),
            warps=(warp,),
            objects=(),
            triggers=(),
        )
        goal = ReachWarp(destination_map=destination[0], destination=destination, warp=warp)
        with patch("modules.navigation.get_world_map_graph", return_value=WorldMapGraph(())):
            decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(decision.goal_evaluation.status, GoalStatus.REACHABLE)
        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(decision.action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(decision.action.direction, Direction.East)
        self.assertIn(
            "global route unavailable; using observed local transition", decision.goal_evaluation.world_diagnostics
        )

    def test_arrival_at_directional_observed_warp_emits_activation_input(self):
        destination = (("destination", 0), (10, 19))
        warp = WarpObservation(
            (MAP, (1, 0)),
            destination,
            required_facing=Direction.East,
            activation=WarpActivation.DIRECTIONAL_STEP,
        )
        world = OverworldObservation(
            MAP,
            (1, 0),
            Direction.East,
            True,
            (TileObservation((MAP, (1, 0)), False, frozenset(Direction), warp=warp),),
            (warp,),
            (),
            (),
        )
        goal = ReachWarp(destination_map=destination[0], destination=destination, warp=warp)
        with patch("modules.navigation.get_world_map_graph", return_value=WorldMapGraph(())):
            decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(decision.goal_evaluation.status, GoalStatus.REACHABLE)
        self.assertEqual(decision.action.navigation.action_type, NavigationActionType.WARP)
        self.assertEqual(decision.action.direction, Direction.East)
        self.assertEqual(decision.action.navigation.source, (MAP, (1, 0)))

    def test_local_directional_warp_requires_reobserved_activation_and_destination(self):
        destination = (("destination", 0), (10, 19))
        warp = WarpObservation(
            (MAP, (1, 0)),
            destination,
            required_facing=Direction.East,
            activation=WarpActivation.DIRECTIONAL_STEP,
        )

        def source_world(coordinates):
            return OverworldObservation(
                MAP,
                coordinates,
                Direction.East,
                True,
                tuple(
                    TileObservation(
                        (MAP, coordinate), False, frozenset(Direction), warp=warp if coordinate == (1, 0) else None
                    )
                    for coordinate in ((0, 0), (1, 0))
                ),
                (warp,),
                (),
                (),
            )

        target_world = OverworldObservation(
            destination[0],
            destination[1],
            Direction.South,
            True,
            (TileObservation(destination, False, frozenset(Direction)),),
            (),
            (),
            (),
        )
        observations = iter((source_world((0, 0)), source_world((1, 0)), target_world))
        goal = ReachWarp(destination_map=destination[0], destination=destination, warp=warp)
        emulator = Mock()
        loop = AgentControlLoop(
            lambda: AgentObservation(
                InteractionObservation(GameState.OVERWORLD, controllable=True),
                overworld=next(observations),
                goal=goal,
            )
        )
        with patch("modules.navigation.get_world_map_graph", return_value=WorldMapGraph(())), patch(
            "modules.agent_control.context.emulator", emulator
        ):
            movement = loop.step()
            activation = loop.step()
            arrived = loop.step()

        self.assertEqual(movement[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(activation[1].action.navigation.action_type, NavigationActionType.WARP)
        self.assertEqual(activation[1].action.direction, Direction.East)
        self.assertEqual(arrived[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertIsNone(loop._expected_world_transition)
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False), call("Right", run=False, fresh=True)])

    def test_failed_local_directional_activation_is_not_repeated_forever(self):
        destination = (("destination", 0), (10, 19))
        warp = WarpObservation(
            (MAP, (1, 0)),
            destination,
            required_facing=Direction.East,
            activation=WarpActivation.DIRECTIONAL_STEP,
        )
        world = OverworldObservation(
            MAP,
            (1, 0),
            Direction.East,
            True,
            (TileObservation((MAP, (1, 0)), False, frozenset(Direction), warp=warp),),
            (warp,),
            (),
            (),
        )
        goal = ReachWarp(destination_map=destination[0], destination=destination, warp=warp)
        observations = iter((world,) * 12)
        emulator = Mock()
        loop = AgentControlLoop(
            lambda: AgentObservation(
                InteractionObservation(GameState.OVERWORLD, controllable=True),
                overworld=next(observations),
                goal=goal,
            )
        )
        with patch("modules.navigation.get_world_map_graph", return_value=WorldMapGraph(())), patch(
            "modules.agent_control.context.emulator", emulator
        ):
            results = [loop.step() for _ in range(12)]

        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=True)])
        self.assertTrue(all(result[2].result_type is not ActionResultType.UNREACHABLE for result in results))

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

    def test_goal_does_not_reinteract_while_avatar_is_not_controllable(self):
        from modules.goals import ActivateTrigger

        world = overworld(
            {(0, 0)},
            triggers=(
                TriggerObservation(
                    "early_pokeballs", frozenset({(MAP, (1, 0))}), activation_locations=frozenset({(MAP, (0, 0))})
                ),
            ),
            controllable=False,
        )
        decision = select_action(
            observation(GameState.OVERWORLD, world=world, goal=ActivateTrigger("early_pokeballs"), controllable=False)
        )
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)

    def test_uncontrollable_field_dialogue_transition_advances_once(self):
        from modules.goals import ActivateTrigger

        world = overworld({(0, 0)}, controllable=False)
        transition = observation(
            GameState.OVERWORLD,
            world=world,
            goal=ActivateTrigger("early_pokeballs"),
            controllable=False,
            field_message_lifecycle_active=True,
            field_message_advance_ready=True,
        )
        ended = observation(
            GameState.OVERWORLD,
            world=world,
            goal=ActivateTrigger("early_pokeballs"),
            controllable=False,
        )
        emulator = Mock()
        observations = iter((transition, transition, ended))
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(lambda: next(observations), goal=ActivateTrigger("early_pokeballs"))
            first = loop.step()
            second = loop.step()
            third = loop.step()

        self.assertEqual(first[1].action.action_type, AgentActionType.ADVANCE_DIALOGUE)
        self.assertEqual(first[2].result_type, ActionResultType.EXECUTED)
        self.assertEqual(second[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(third[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        emulator.press_button.assert_called_once_with("A")

    def test_controllable_dialogue_remains_unchanged(self):
        emulator = Mock()
        dialogue = observation(GameState.OVERWORLD, dialogue_waiting=True, controllable=True)
        with patch("modules.agent_control.context.emulator", emulator):
            result = AgentActionExecutor().execute(AgentAction(AgentActionType.ADVANCE_DIALOGUE), dialogue)
        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        emulator.press_button.assert_called_once_with("A")

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

    def _movement_batch_loop(self, *, coordinates=(0, 0), destination=(1, 0)):
        from modules.agent_control import _MovementBatch

        loop = AgentControlLoop(lambda: None)
        loop._movement_batch = _MovementBatch(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, destination),
                ),
            )
        )
        avatar = SimpleNamespace(
            map_group_and_number=MAP,
            local_coordinates=coordinates,
            facing_direction="Right",
        )
        return loop, avatar

    def test_unchanged_batch_uses_fast_path_without_full_interaction_observation(self):
        loop, avatar = self._movement_batch_loop()
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ), patch(
            "modules.agent_control.observe_interaction", side_effect=AssertionError("full observation")
        ), patch(
            "modules.agent_control.evaluate_goal", side_effect=AssertionError("goal evaluation")
        ), patch(
            "modules.agent_control.plan_with_world_navigation", side_effect=AssertionError("pathfinding")
        ):
            self.assertTrue(loop._advance_movement_batch())
        emulator.hold_button.assert_called_once_with("Right")

    def test_fast_path_turns_without_debug_diagnostics(self):
        loop, avatar = self._movement_batch_loop()
        avatar.facing_direction = "Up"
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.context.debug", False
        ), patch("modules.agent_control.context.debug_trace", False), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ):
            self.assertTrue(loop._advance_movement_batch())

        emulator.press_direction.assert_called_once_with("Right", run=False)

    def test_batch_position_mismatch_falls_back(self):
        loop, avatar = self._movement_batch_loop(coordinates=(9, 9))
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ):
            self.assertFalse(loop._advance_movement_batch())
        self.assertIsNone(loop._movement_batch)
        emulator.reset_held_buttons.assert_called_once_with()

    def test_batch_dialogue_falls_back(self):
        loop, avatar = self._movement_batch_loop()
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=True), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ):
            self.assertFalse(loop._advance_movement_batch())
        self.assertIsNone(loop._movement_batch)

    def test_periodic_reobserve_invalidates_runtime_cache(self):
        loop = AgentControlLoop(lambda: None)
        loop._last_observation_frame = 100
        emulator = Mock()
        emulator.get_frame_count.return_value = 190
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.state_cache.invalidate_runtime_observations"
        ) as invalidate:
            loop._force_periodic_reobserve()
        invalidate.assert_called_once_with()

    def test_universal_observation_watchdog_invalidates_at_frame_boundary(self):
        emulator = Mock()
        emulator.get_frame_count.return_value = 190
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control._last_universal_observation_frame", 100
        ), patch("modules.agent_control.state_cache.invalidate_runtime_observations") as invalidate:
            agent_control._force_universal_reobserve()
        invalidate.assert_called_once_with()

    def test_universal_watchdog_perceives_settled_warp_during_unknown_callback(self):
        settled_interior = overworld({(0, 0)})
        unknown = InteractionObservation(GameState.UNKNOWN)
        with patch("modules.agent_control._force_universal_reobserve", return_value=True), patch(
            "modules.agent_control.observe_interaction", return_value=unknown
        ), patch("modules.agent_control.perceive_overworld", return_value=settled_interior) as perceive:
            result = observe_agent()
        self.assertIs(result.overworld, settled_interior)
        perceive.assert_called_once_with()

    def test_required_overworld_observes_map_during_change_map_callback(self):
        settled_interior = overworld({(0, 0)})
        changing_map = InteractionObservation(GameState.CHANGE_MAP)
        with patch("modules.agent_control._force_universal_reobserve", return_value=False), patch(
            "modules.agent_control.observe_interaction", return_value=changing_map
        ), patch("modules.agent_control.perceive_overworld", return_value=settled_interior) as perceive:
            result = observe_agent(require_overworld=True)
        self.assertIs(result.overworld, settled_interior)
        perceive.assert_called_once_with()

    def test_cached_batch_forces_reobserve_at_periodic_boundary(self):
        loop, avatar = self._movement_batch_loop()
        loop._last_observation_frame = 100
        emulator = Mock()
        emulator.get_frame_count.return_value = 190
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.state_cache.invalidate_runtime_observations"
        ) as invalidate:
            self.assertFalse(loop._advance_movement_batch())
        invalidate.assert_called_once_with()
        self.assertIsNone(loop._movement_batch)

    def test_batch_map_transition_falls_back(self):
        loop, avatar = self._movement_batch_loop()
        avatar.map_group_and_number = ("other", 0)
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ):
            self.assertFalse(loop._advance_movement_batch())
        self.assertIsNone(loop._movement_batch)

    def test_batch_blocked_movement_falls_back_after_existing_timeout(self):
        loop, avatar = self._movement_batch_loop()
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ):
            self.assertTrue(all(loop._advance_movement_batch() for _ in range(24)))
            self.assertFalse(loop._advance_movement_batch())
        self.assertIsNone(loop._movement_batch)

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
        self.assertIsNotNone(loop._cached_evaluation)
        self.assertEqual(resumed[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(evaluate.call_count, 2)
        self.assertFalse(loop._warp_settling)

    def test_post_warp_settling_defers_invalid_destination_coordinate(self):
        source_map = (0, 0)
        target_map = (0, 1)
        goal = ReachLocation((target_map, (0, 0)))

        def target_world(coordinates):
            return OverworldObservation(
                map_id=target_map,
                player_coordinates=coordinates,
                facing=Direction.East,
                controllable=True,
                tiles=(TileObservation((target_map, (0, 0)), False, frozenset(Direction)),),
                warps=(),
                objects=(),
                triggers=(),
            )

        observations = iter(
            (
                observation(GameState.OVERWORLD, world=target_world((9, 22)), goal=goal),
                observation(GameState.OVERWORLD, world=target_world((0, 0)), goal=goal),
            )
        )
        loop = AgentControlLoop(lambda: next(observations))
        loop._expected_world_transition = ((source_map, (1, 0)), (target_map, (0, 0)))
        loop._cached_evaluation = GoalEvaluation(GoalStatus.REACHABLE)

        with patch("modules.agent_control.evaluate_goal", return_value=GoalEvaluation(GoalStatus.COMPLETE)) as evaluate:
            transient = loop.step()
            arrived = loop.step()

        self.assertEqual(transient[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(transient[1].action.reason, "")
        self.assertEqual(arrived[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(evaluate.call_count, 2)

    def test_exhausted_cached_reach_location_uses_semantic_completion(self):
        goal = ReachLocation((MAP, (0, 0)))
        world = overworld({(0, 0)}, start=(0, 0))
        observed = observation(GameState.OVERWORLD, world=world, goal=goal)
        semantic_complete = GoalEvaluation(GoalStatus.COMPLETE, reason="goal position reached")
        loop = AgentControlLoop(lambda: observed, goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((), (MAP, (0, 0))),
        )
        with patch("modules.agent_control.evaluate_goal", return_value=semantic_complete) as evaluate:
            result = loop.step()
            self.assertRaises(StopIteration, next, loop.run())
        self.assertEqual(result[2].result_type, ActionResultType.GOAL_COMPLETE)
        self.assertIsNotNone(loop._cached_evaluation)
        self.assertGreaterEqual(evaluate.call_count, 1)

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
        self.assertEqual(emulator.press_direction.call_args_list, [call("Down", run=False, fresh=False)] * 2)

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
        emulator.press_direction.assert_called_once_with("Right", run=False, fresh=False)
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
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False)] * 2)
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
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False), call("Right", run=False, fresh=True)])

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
            deferred_inputs = list(emulator.mock_calls)
            activated = loop.step()
            completed = loop.step()

        self.assertEqual(move[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(deferred[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(deferred[2].result_type, ActionResultType.WAITING)
        self.assertIsNone(deferred_transition)
        self.assertIn(call.press_direction("Right", run=False, fresh=False), deferred_inputs)
        self.assertIsNone(loop._expected_world_transition)
        self.assertEqual(activated[1].action.navigation.action_type, NavigationActionType.WARP)
        self.assertEqual(activated[2].result_type, ActionResultType.EXECUTED)
        self.assertEqual(completed[1].goal_evaluation.status, GoalStatus.COMPLETE)
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False), call("Right", run=False, fresh=True)])

    def test_executor_rejects_input_for_unknown_state(self):
        action = available_actions(observation(GameState.UNKNOWN))[0]
        result = AgentActionExecutor().execute(action, observation(GameState.UNKNOWN))
        self.assertEqual(result.result_type, ActionResultType.WAITING)

    def test_run_propagates_unreachable_instead_of_reporting_completion(self):
        loop = AgentControlLoop(lambda: observation(GameState.UNKNOWN))
        action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason="blocked")
        with patch.object(
            loop,
            "step",
            return_value=(None, None, ActionResult(ActionResultType.UNREACHABLE, action, "no route")),
        ):
            with self.assertRaises(NavigationError):
                next(loop.run())
