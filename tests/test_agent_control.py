from unittest import TestCase
from unittest.mock import Mock, call, patch
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import modules.agent_control as agent_control
from modules.agent_control import (
    ActionResult,
    ActionResultType,
    AgentAction,
    AgentActionExecutor,
    AgentActionType,
    ActionDecision,
    AgentControlLoop,
    AgentObservation,
    GoalEvaluation,
    GoalStatus,
    available_actions,
    evaluate_goal,
    prewarm_warp_destination,
    select_action,
    observe_agent,
    notify_battle_ended,
    _navigation_request_key,
    _SingleFlightNavigationPlanner,
)
from modules.goals import (
    EngageTrainer,
    GoalConstraints,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    TrainerEngagementMode,
    TrainerMode,
)
from modules.interaction_state import InteractionObservation, InteractionPhase
from modules.map_path import Direction
from modules.memory import GameState
from modules.navigation import NavigationAction, NavigationActionType, NavigationPlan, NavigationError
from modules.navigation import NavigationSearchCancelled
from modules.overworld import (
    MovementState,
    ObjectObservation,
    OverworldObservation,
    TileObservation,
    TriggerObservation,
    WarpObservation,
    WarpActivation,
)
from modules.world_navigation import WorldEdge, WorldMapGraph

MAP = ("test", 0)


def test_tactical_navigation_request_key_ignores_transient_occupancy():
    state = {
        "blocked": ((0, 0),),
        "transitions": (("warp", (MAP, (2, 0)), ("other", 0), None),),
    }
    world = SimpleNamespace(revision_key=lambda: (state["blocked"], state["transitions"]))
    goal = ReachLocation((MAP, (3, 0)))

    first = _navigation_request_key((MAP, (0, 0)), goal, world)
    state["blocked"] = ((1, 0),)
    second = _navigation_request_key((MAP, (0, 0)), goal, world)
    assert first == second

    state["transitions"] = (("warp", (MAP, (1, 0)), ("other", 0), None),)
    third = _navigation_request_key((MAP, (0, 0)), goal, world)
    assert third != second


def test_failed_tactical_search_reopens_when_trainer_state_changes():
    planner = _SingleFlightNavigationPlanner()
    calls = []
    key = ((MAP, (0, 0)), "goal", ("topology",))
    unchanged_failure = (key, ("trainer", False))
    reopened_failure = (key, ("trainer", True))

    def work(_cancel_check):
        calls.append(True)
        raise NavigationError("trainer blocks the route")

    try:
        result = planner.poll(key, work, failure_key=unchanged_failure)
        for _ in range(100):
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
            result = planner.poll(key, work, failure_key=unchanged_failure)
        assert result[0] is None
        assert len(calls) == 1

        # Identical failure state is cached, while a changed trainer state
        # reopens the same structural route request.
        assert planner.poll(key, work, failure_key=unchanged_failure) == result
        result = planner.poll(key, work, failure_key=reopened_failure)
        for _ in range(100):
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
            result = planner.poll(key, work, failure_key=reopened_failure)
        assert result[0] is None
        assert len(calls) == 2
    finally:
        planner._executor.shutdown(wait=True)


def test_tactical_search_accepts_result_when_only_failure_occupancy_key_changes():
    planner = _SingleFlightNavigationPlanner()
    started = threading.Event()
    release = threading.Event()
    calls = []
    key = ((MAP, (0, 0)), "goal", ("topology",))

    def work(_cancel_check):
        calls.append(True)
        started.set()
        release.wait(1)
        return object(), None

    try:
        assert planner.poll(key, work, failure_key=(key, ((0, 0),))) is agent_control._NAVIGATION_PLAN_PENDING
        assert started.wait(1)
        release.set()
        result = None
        for _ in range(100):
            result = planner.poll(key, work, failure_key=(key, ((1, 0),)))
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
        assert result[2] is None
        assert result[0] is not None
        assert calls == [True]
    finally:
        planner._executor.shutdown(wait=True)


def test_tactical_search_cancels_superseded_request_before_submitting_new_tile():
    planner = _SingleFlightNavigationPlanner()
    started = threading.Event()
    calls = []
    first_key = (MAP, (0, 0), "goal")
    second_key = (MAP, (1, 0), "goal")

    def work(cancel_check):
        calls.append(len(calls) + 1)
        started.set()
        if len(calls) == 1:
            while not cancel_check():
                time.sleep(0.001)
        return object(), None

    try:
        assert planner.poll(first_key, work) is agent_control._NAVIGATION_PLAN_PENDING
        assert started.wait(1)
        with patch("modules.agent_control.print_navigation_cancellation") as cancellation_log:
            assert planner.poll(second_key, work) is agent_control._NAVIGATION_PLAN_PENDING
            requested = [
                call.kwargs
                for call in cancellation_log.call_args_list
                if call.kwargs.get("phase") == "requested"
            ]
            assert requested
            assert requested[0]["reason"] == "newer navigation request superseded the active request"
            assert requested[0]["changed"] == ("goal",)
        result = None
        for _ in range(100):
            result = planner.poll(second_key, work)
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
        assert result[0] is not None
        assert result[2] is None
        assert calls == [1, 2]
    finally:
        planner._executor.shutdown(wait=True)


def test_tactical_search_coalesces_start_only_changes_without_cancelling_worker():
    planner = _SingleFlightNavigationPlanner()
    started = threading.Event()
    release = threading.Event()
    calls = []
    first_key = ((MAP, (0, 0)), "goal", ("topology",))
    second_key = ((MAP, (1, 0)), "goal", ("topology",))

    def work(_cancel_check):
        calls.append(len(calls) + 1)
        started.set()
        if len(calls) == 1:
            release.wait(1)
        return object(), None

    try:
        assert planner.poll(first_key, work) is agent_control._NAVIGATION_PLAN_PENDING
        assert started.wait(1)
        with patch("modules.agent_control.print_navigation_cancellation") as cancellation_log:
            assert planner.poll(second_key, work) is agent_control._NAVIGATION_PLAN_PENDING
            requested = [
                call.kwargs
                for call in cancellation_log.call_args_list
                if call.kwargs.get("phase") == "requested"
            ]
            assert requested == []

        release.set()
        result = None
        for _ in range(100):
            result = planner.poll(second_key, work)
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
        assert result[0] is not None
        assert result[2] is None
        assert calls == [1, 2]
    finally:
        release.set()
        planner._executor.shutdown(wait=True)


def test_tactical_search_has_a_terminal_timeout_for_a_stable_request():
    planner = _SingleFlightNavigationPlanner(search_timeout_seconds=0.01)
    started = threading.Event()

    def work(cancel_check):
        started.set()
        while True:
            cancel_check()
            time.sleep(0.001)

    try:
        key = (MAP, (0, 0), "goal")
        assert planner.poll(key, work) is agent_control._NAVIGATION_PLAN_PENDING
        assert started.wait(1)
        result = None
        for _ in range(100):
            result = planner.poll(key, work)
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
        assert result[0] is None
        assert isinstance(result[2], NavigationSearchCancelled)
        assert "10" not in str(result[2])
        assert "second budget" in str(result[2])
    finally:
        planner._executor.shutdown(wait=True)


def test_tactical_search_does_not_cache_a_superseded_cancellation():
    planner = _SingleFlightNavigationPlanner()
    started = threading.Event()
    calls = []
    owner = object()
    key = (MAP, (0, 0), "woods-encounter")

    def work(cancel_check):
        calls.append(len(calls) + 1)
        started.set()
        if len(calls) == 1:
            while not cancel_check():
                time.sleep(0.001)
        return object(), None

    try:
        assert planner.poll(key, work, owner=owner) is agent_control._NAVIGATION_PLAN_PENDING
        assert started.wait(1)
        assert planner.cancel(owner, reason="loop disposed", owner_label="replacement-loop")

        cancelled = None
        for _ in range(100):
            cancelled = planner.poll(key, work, owner=owner)
            if cancelled is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
        assert isinstance(cancelled[2], NavigationSearchCancelled)

        # The cancelled result must not poison the exact request. A later
        # owner can submit the same Woods route and receive a real plan.
        result = planner.poll(key, work, owner=owner)
        assert result is agent_control._NAVIGATION_PLAN_PENDING
        for _ in range(100):
            result = planner.poll(key, work, owner=owner)
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
        assert result[0] is not None
        assert result[2] is None
        assert calls == [1, 2]
    finally:
        planner._executor.shutdown(wait=True)


def test_tactical_search_does_not_outlive_its_navigation_owner():
    planner = _SingleFlightNavigationPlanner()
    started = threading.Event()
    calls = []

    def work(cancel_check):
        calls.append(len(calls) + 1)
        started.set()
        if len(calls) == 1:
            while not cancel_check():
                time.sleep(0.001)
        return object(), None

    owner = object()
    replacement = object()
    try:
        assert planner.poll(("owner", 1), work, owner=owner, owner_label="old-loop") is agent_control._NAVIGATION_PLAN_PENDING
        assert started.wait(1)
        with patch("modules.agent_control.print_navigation_cancellation") as cancellation_log:
            assert planner.cancel(owner, reason="loop disposed", owner_label="new-loop")
            assert any(
                item.kwargs.get("reason") == "loop disposed"
                for item in cancellation_log.call_args_list
            )

        result = agent_control._NAVIGATION_PLAN_PENDING
        for _ in range(100):
            result = planner.poll(
                ("replacement", 1),
                work,
                owner=replacement,
                owner_label="new-loop",
            )
            if result is not agent_control._NAVIGATION_PLAN_PENDING:
                break
            time.sleep(0.001)
        assert result[0] is not None
        assert calls == [1, 2]
    finally:
        planner._executor.shutdown(wait=True)


def test_tactical_request_key_uses_semantic_transition_identity():
    world = SimpleNamespace(revision_key=lambda: ((), (("warp", (MAP, (1, 1)), ("other", 0), None),)))
    first_transition = SimpleNamespace(
        kind="warp",
        entry=(MAP, (1, 1)),
        destination=(("other", 0), (2, 2)),
        required_facing=None,
        activation=None,
        activation_locations=(),
        activation_direction=None,
    )
    second_transition = SimpleNamespace(
        kind="warp",
        entry=(MAP, (1, 1)),
        destination=(("other", 0), (2, 2)),
        required_facing=None,
        activation=None,
        activation_locations=(),
        activation_direction=None,
    )
    first_goal = NavigationGoal(
        ReachWarp(destination_map=("other", 0), destination=(("other", 0), (2, 2)), warp=first_transition)
    )
    second_goal = NavigationGoal(
        ReachWarp(destination_map=("other", 0), destination=(("other", 0), (2, 2)), warp=second_transition)
    )

    assert _navigation_request_key((MAP, (0, 0)), first_goal, world) == _navigation_request_key(
        (MAP, (0, 0)), second_goal, world
    )


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
    field_message_render_rescue_available=False,
    interaction_phase=None,
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
            field_message_render_rescue_available=field_message_render_rescue_available,
            interaction_phase=(
                interaction_phase
                if interaction_phase is not None
                else (
                    InteractionPhase.FIELD_MESSAGE_INPUT_WAIT
                    if dialogue_waiting or field_message_advance_ready
                    else InteractionPhase.NONE
                )
            ),
        ),
        overworld=world,
        goal=goal,
    )


class AgentActionSelectionTests(TestCase):
    def test_forced_trainer_exposure_is_not_dispatched_for_avoidance_goal(self):
        world = overworld({(0, 0), (1, 0), (2, 0)}, start=(0, 0))
        goal = NavigationGoal(
            ReachLocation((MAP, (2, 1))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        forced = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, (1, 0)),
                ),
            ),
            (MAP, (2, 0)),
            forced_trainer_exposure=True,
        )
        with patch("modules.agent_control.plan_with_world_navigation", return_value=(forced, None)):
            decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(decision.goal_evaluation.status, GoalStatus.UNREACHABLE)
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertIn("trainer-free", decision.action.reason)

    def test_forced_trainer_exposure_can_be_dispatched_when_policy_allows_it(self):
        world = overworld({(0, 0), (1, 0), (2, 0)}, start=(0, 0))
        goal = NavigationGoal(
            ReachLocation((MAP, (2, 1))),
            constraints=GoalConstraints(
                trainer_mode=TrainerMode.AVOID,
                allow_forced_trainer_exposure=True,
            ),
        )
        forced = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, (1, 0)),
                ),
            ),
            (MAP, (2, 0)),
            forced_trainer_exposure=True,
        )
        with patch("modules.agent_control.plan_with_world_navigation", return_value=(forced, None)):
            decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(decision.goal_evaluation.status, GoalStatus.REACHABLE)
        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)

    def test_goal_evaluation_does_not_collect_readiness_diagnostics_without_trace(self):
        world = overworld({(0, 0), (1, 0)}, start=(0, 0))
        goal = ReachLocation((MAP, (1, 0)))
        with patch("modules.agent_control.context.debug", True), patch(
            "modules.agent_control.context.debug_trace", False
        ), patch("modules.nuzlocke.snapshots.get_nuzlocke_snapshot") as snapshot:
            result = evaluate_goal(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(result.status, GoalStatus.REACHABLE)
        snapshot.assert_not_called()

    def test_semantic_map_goal_completes_when_already_on_target_map(self):
        world = overworld({(0, 0)}, start=(0, 0))
        goal = NavigationGoal(SemanticTarget.map(MAP))

        decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)
        self.assertEqual(decision.goal_evaluation.reason, "goal position reached")

    def test_reach_warp_completes_when_destination_map_is_already_observed(self):
        destination_map = ("destination", 0)
        world = OverworldObservation(
            map_id=destination_map,
            player_coordinates=(0, 0),
            facing=Direction.South,
            controllable=True,
            tiles=(TileObservation((destination_map, (0, 0)), False, frozenset(Direction)),),
            warps=(),
            objects=(),
            triggers=(),
        )
        goal = ReachWarp(destination_map=destination_map)

        decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)
        self.assertEqual(decision.goal_evaluation.reason, "destination map reached")

    def test_pending_navigation_does_not_delay_map_goal_completion(self):
        world = overworld({(0, 0)}, start=(0, 0))
        goal = NavigationGoal(SemanticTarget.map(MAP))

        decision = select_action(
            observation(GameState.OVERWORLD, world=world, goal=goal),
            navigation_result=agent_control._NAVIGATION_PLAN_PENDING,
        )

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.action.reason, "destination map reached")
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)

    def test_exact_staging_location_completes_without_starting_background_search(self):
        world = overworld({(0, 0)}, start=(0, 0))
        goal = NavigationGoal(ReachLocation((MAP, (0, 0))))
        loop = AgentControlLoop(
            lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
            goal=goal,
        )

        with patch.object(
            agent_control._background_navigation_planner,
            "poll",
            side_effect=AssertionError("completed staging route started another search"),
        ):
            _, decision, result = loop.step()

        self.assertEqual(result.result_type, ActionResultType.GOAL_COMPLETE)
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.action.reason, "goal position reached")
        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)

    def test_invalid_destination_map_coordinate_waits_for_transition_to_settle(self):
        world = overworld({(0, 0)}, start=(0, -1))
        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=ReachLocation((MAP, (0, 0))),
            )
        )

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(
            decision.action.reason,
            "avatar coordinate is not present in current overworld topology; waiting for transition to settle",
        )

    def test_script_owned_controllable_overworld_waits_instead_of_reinteracting(self):
        world = overworld({(0, 0)}, controllable=True)
        observed = observation(
            GameState.OVERWORLD,
            world=world,
            goal=ReachLocation((MAP, (0, 0))),
        )
        observed = replace(
            observed,
            interaction=InteractionObservation(
                GameState.OVERWORLD,
                controllable=True,
                script_active=True,
                script_function="SomeInteraction",
                native_function="WaitForEffect",
            ),
        )
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
            "observed transition planned locally; global route not consulted",
            decision.goal_evaluation.world_diagnostics,
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
        self.assertEqual(
            emulator.press_direction.call_args_list,
            [call("Right", run=False, fresh=False), call("Right", run=False, fresh=True)],
        )

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

    def test_automatic_coordinate_trigger_waits_for_rom_scene_instead_of_pressing_a(self):
        from modules.goals import ActivateTrigger

        trigger = TriggerObservation(
            "coord:0:152:0",
            frozenset({(MAP, (0, 0))}),
            activation_locations=frozenset({(MAP, (0, 0))}),
            requires_input=False,
        )
        world = overworld({(0, 0)}, triggers=(trigger,))
        decision = select_action(observation(GameState.OVERWORLD, world=world, goal=ActivateTrigger("coord:0:152:0")))

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.action.reason, "automatic coordinate trigger fires on tile entry")

    def test_reached_trainer_sight_line_waits_for_battle_handoff(self):
        trainer_id = "trainer:('test', 0):1"
        trigger = TriggerObservation(
            "trainer:1",
            frozenset({(MAP, (0, 1))}),
            kind="trainer",
            affordance_id=trainer_id,
            hazard_locations=frozenset({(MAP, (0, 0))}),
            engagement_locations=frozenset({(MAP, (0, 0))}),
            hazard_kind="trainer",
        )
        world = overworld({(0, 0)}, triggers=(trigger,))

        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=NavigationGoal(EngageTrainer(trainer_id)),
            )
        )

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.action.reason, "waiting for trainer sight-line activation")

    def test_campaign_cleared_trainer_goal_completes_without_live_object(self):
        trainer_id = "trainer:('test', 0):1"
        goal = NavigationGoal(EngageTrainer(trainer_id))
        world = overworld({(0, 0)})

        with patch.object(
            agent_control.context,
            "campaign_defeated_trainer_ids",
            {("trainer_affordance_id", trainer_id)},
        ):
            decision = select_action(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)
        self.assertEqual(decision.goal_evaluation.reason, "campaign trainer clearance recorded")
        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)

    def test_unchanged_automatic_trainer_wait_is_bounded(self):
        trainer_id = "trainer:('test', 0):1"
        world = overworld({(0, 0)})
        observed = observation(
            GameState.OVERWORLD,
            world=world,
            goal=NavigationGoal(EngageTrainer(trainer_id)),
        )
        loop = AgentControlLoop(lambda: None, goal=NavigationGoal(EngageTrainer(trainer_id)))
        loop._automatic_trainer_wait_limit = 2
        decision = ActionDecision(
            AgentAction(AgentActionType.WAIT_REOBSERVE, reason="waiting for trainer sight-line activation"),
            GoalEvaluation(
                GoalStatus.REACHABLE,
                plan=NavigationPlan((), (MAP, (0, 0))),
                engagement_plan=SimpleNamespace(
                    mode=TrainerEngagementMode.SIGHTLINE,
                    trigger_id="trainer:1",
                ),
            ),
        )

        self.assertIsNone(loop._apply_automatic_trainer_wait_watchdog(observed, decision))
        watchdog_result = loop._apply_automatic_trainer_wait_watchdog(observed, decision)
        self.assertIsNotNone(watchdog_result)
        self.assertEqual(watchdog_result[0].action.reason, "automatic_trainer_wait_watchdog")
        self.assertEqual(watchdog_result[1].result_type, ActionResultType.WAITING)
        self.assertRaises(
            NavigationError,
            lambda: (
                loop._apply_automatic_trainer_wait_watchdog(observed, decision),
                loop._apply_automatic_trainer_wait_watchdog(observed, decision),
            ),
        )

    def test_interaction_trainer_faces_and_presses_a(self):
        trainer_id = "trainer:('test', 0):7"
        trigger = TriggerObservation(
            "object:7:GymLeaderScript",
            frozenset({(MAP, (1, 0))}),
            activation_locations=frozenset({(MAP, (0, 0))}),
            activation_requirements=(((MAP, (0, 0)), Direction.East),),
            kind="semantic_object",
            affordance_id=trainer_id,
            hazard_kind="trainer",
        )
        world = overworld(
            {(0, 0), (1, 0)},
            triggers=(trigger,),
            facing=Direction.East,
        )

        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=NavigationGoal(EngageTrainer(trainer_id)),
            )
        )

        self.assertEqual(decision.action.action_type, AgentActionType.INTERACT)
        self.assertEqual(decision.action.option, trigger.trigger_id)
        self.assertEqual(decision.goal_evaluation.engagement_plan.mode, TrainerEngagementMode.INTERACTION)

    def test_interaction_trainer_turns_before_pressing_a(self):
        trainer_id = "trainer:('test', 0):8"
        trigger = TriggerObservation(
            "object:8:GymLeaderScript",
            frozenset({(MAP, (1, 0))}),
            activation_locations=frozenset({(MAP, (0, 0))}),
            activation_requirements=(((MAP, (0, 0)), Direction.East),),
            kind="semantic_object",
            affordance_id=trainer_id,
            hazard_kind="trainer",
        )
        world = overworld(
            {(0, 0), (1, 0)},
            triggers=(trigger,),
            facing=Direction.North,
        )

        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=NavigationGoal(EngageTrainer(trainer_id)),
            )
        )

        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(decision.action.navigation.action_type.name, "TURN")
        self.assertEqual(decision.goal_evaluation.engagement_plan.mode, TrainerEngagementMode.INTERACTION)

    def test_unknown_trainer_geometry_never_becomes_an_interaction(self):
        trainer_id = "trainer:('test', 0):9"
        trigger = TriggerObservation(
            "static_object:9:TrainerScript",
            frozenset({(MAP, (1, 0))}),
            kind="trainer_hazard",
            affordance_id=trainer_id,
            hazard_kind="trainer",
            requires_input=False,
        )
        world = overworld({(0, 0), (1, 0)}, triggers=(trigger,))

        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=world,
                goal=NavigationGoal(EngageTrainer(trainer_id)),
            )
        )

        self.assertNotEqual(decision.action.action_type, AgentActionType.INTERACT)
        self.assertEqual(decision.goal_evaluation.engagement_plan.mode, TrainerEngagementMode.UNKNOWN)

    def test_reached_interaction_position_returns_when_rom_starts_script(self):
        trigger = TriggerObservation(
            "nurse",
            frozenset({(MAP, (0, 0))}),
            activation_locations=frozenset({(MAP, (0, 0))}),
            requires_input=False,
        )
        world = overworld({(0, 0)}, triggers=(trigger,))
        goal = ReachInteractionPosition("nurse")
        reached = observation(GameState.OVERWORLD, world=world, goal=goal)
        loop = AgentControlLoop(lambda: None, goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((), (MAP, (0, 0))),
        )
        loop._cached_world_signature = loop._world_signature(reached)

        positioning_decision = loop._cached_decision(reached)
        self.assertEqual(positioning_decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertTrue(loop._interaction_position_ready)

        script_started = AgentObservation(
            InteractionObservation(
                GameState.UNKNOWN,
                controllable=True,
                script_active=True,
                script_function="Common_EventScript_PkmnCenterNurse",
            ),
            goal=goal,
        )
        loop._observe = lambda: script_started
        with patch("modules.agent_control.context.emulator", Mock()):
            _, decision, result = loop.step()

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(result.result_type, ActionResultType.GOAL_COMPLETE)
        self.assertFalse(loop._interaction_position_ready)

    def test_route_completion_hands_off_when_automatic_script_starts_before_readiness_observation(self):
        """A cached route may reach an automatic nurse tile without an intervening overworld frame."""
        trigger = TriggerObservation(
            "nurse",
            frozenset({(MAP, (0, 0))}),
            activation_locations=frozenset({(MAP, (0, 0))}),
            requires_input=False,
        )
        goal = ReachInteractionPosition("nurse")
        loop = AgentControlLoop(lambda: None, goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((), (MAP, (0, 0))),
        )
        loop._cached_actions = ()
        loop._cached_action_index = 0

        script_started = AgentObservation(
            InteractionObservation(
                GameState.UNKNOWN,
                controllable=True,
                script_active=True,
                script_function="Common_EventScript_PkmnCenterNurse",
            ),
            goal=goal,
        )
        loop._observe = lambda: script_started
        with patch("modules.agent_control.context.emulator", Mock()):
            _, decision, result = loop.step()

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(result.result_type, ActionResultType.GOAL_COMPLETE)
        self.assertIn("route completion", result.message)

    def test_completed_automatic_trigger_is_complete_when_rom_condition_turns_off(self):
        from modules.goals import ActivateTrigger

        trigger = TriggerObservation(
            "coord:0:152:0",
            frozenset({(MAP, (0, 0))}),
            activation_locations=frozenset({(MAP, (0, 0))}),
            condition_active=False,
            requires_input=False,
        )
        decision = select_action(
            observation(
                GameState.OVERWORLD,
                world=overworld({(0, 0)}, triggers=(trigger,)),
                goal=ActivateTrigger("coord:0:152:0"),
            )
        )

        self.assertEqual(decision.goal_evaluation.status, GoalStatus.COMPLETE)
        self.assertIn("ROM scene already completed", decision.goal_evaluation.reason)

    def test_cached_automatic_trigger_is_invalidated_after_rom_condition_turns_off(self):
        from modules.goals import ActivateTrigger

        goal = ActivateTrigger("coord:0:152:0")
        trigger = TriggerObservation(
            "coord:0:152:0",
            frozenset({(MAP, (0, 0))}),
            activation_locations=frozenset({(MAP, (0, 0))}),
            condition_active=False,
            requires_input=False,
        )
        world = overworld({(0, 0)}, triggers=(trigger,))
        loop = AgentControlLoop(lambda: None, goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(GoalStatus.REACHABLE)

        with patch.object(agent_control.context, "emulator", Mock()):
            decision = loop._cached_decision(observation(GameState.OVERWORLD, world=world, goal=goal))

        self.assertIsNone(decision)
        self.assertIsNone(loop._cached_evaluation)
        self.assertIsNone(loop._cached_goal)

    def test_cached_static_object_target_interacts_without_runtime_spawn(self):
        from modules.goals import ActivateTrigger

        goal = ActivateTrigger("rival")
        trigger = TriggerObservation(
            "rival",
            frozenset(),
            activation_locations=frozenset({(MAP, (0, 0))}),
            kind="semantic_object",
        )
        world = overworld({(0, 0)}, triggers=(trigger,))
        observed = observation(GameState.OVERWORLD, world=world, goal=goal)
        loop = AgentControlLoop(lambda: observed, goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((), (MAP, (0, 0))),
        )
        loop._cached_world_signature = loop._world_signature(observed)

        decision = loop._cached_decision(observed)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action.action_type, AgentActionType.INTERACT)
        self.assertEqual(decision.action.option, "rival")

    def test_cached_cross_map_trainer_route_reobserves_unknown_engagement_geometry(self):
        trainer_id = "trainer:('route102', 0):2"
        goal = NavigationGoal(EngageTrainer(trainer_id))
        world = overworld({(0, 0)}, start=(0, 0))
        observed = observation(GameState.OVERWORLD, world=world, goal=goal)
        loop = AgentControlLoop(lambda: observed, goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((), (MAP, (0, 0))),
            engagement_plan=agent_control.TrainerEngagementPlan(
                trainer_id,
                "trainer-trigger",
                TrainerEngagementMode.UNKNOWN,
            ),
        )
        loop._cached_world_signature = loop._world_signature(observed)

        with patch.object(agent_control.context, "emulator", Mock()):
            decision = loop._cached_decision(observed)

        self.assertIsNone(decision)
        self.assertIsNone(loop._cached_evaluation)
        self.assertIsNone(loop._cached_goal)

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

    def test_render_rescue_can_follow_a_previous_dialogue_input(self):
        from modules.goals import ActivateTrigger

        world = overworld({(0, 0)}, controllable=False)
        ready = observation(
            GameState.OVERWORLD,
            world=world,
            goal=ActivateTrigger("scene"),
            controllable=False,
            field_message_lifecycle_active=True,
            field_message_advance_ready=True,
        )
        rendering = observation(
            GameState.OVERWORLD,
            world=world,
            goal=ActivateTrigger("scene"),
            controllable=False,
            field_message_lifecycle_active=True,
            field_message_render_rescue_available=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        emulator = Mock()
        observations = iter((ready, rendering))
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(lambda: next(observations), goal=ActivateTrigger("scene"))
            first = loop.step()
            second = loop.step()

        self.assertEqual(first[1].action.action_type, AgentActionType.ADVANCE_DIALOGUE)
        self.assertEqual(second[1].action.action_type, AgentActionType.ACCELERATE_DIALOGUE_RENDER)
        self.assertEqual(second[2].result_type, ActionResultType.EXECUTED)

    def test_controllable_dialogue_remains_unchanged(self):
        emulator = Mock()
        dialogue = observation(GameState.OVERWORLD, dialogue_waiting=True, controllable=True)
        with patch("modules.agent_control.context.emulator", emulator):
            result = AgentActionExecutor().execute(AgentAction(AgentActionType.ADVANCE_DIALOGUE), dialogue)
        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        emulator.press_button.assert_called_once_with("A")

    def test_render_rescue_uses_a_fresh_b_pulse(self):
        class FreshEmulator:
            press_button_fresh = Mock()
            hold_button = Mock()

        emulator = FreshEmulator()
        rendering = observation(
            GameState.OVERWORLD,
            field_message_lifecycle_active=True,
            field_message_render_rescue_available=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        action = AgentAction(AgentActionType.ACCELERATE_DIALOGUE_RENDER)
        with patch("modules.agent_control.context.emulator", emulator):
            result = AgentActionExecutor().execute(action, rendering)

        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        emulator.press_button_fresh.assert_called_once_with("B")
        emulator.hold_button.assert_called_once_with("B")

    def test_render_rescue_releases_held_b_before_dialogue_input(self):
        class FreshEmulator:
            press_button_fresh = Mock()
            hold_button = Mock()
            release_button = Mock()
            is_button_held = Mock(return_value=False)

        emulator = FreshEmulator()
        rendering = observation(
            GameState.OVERWORLD,
            field_message_lifecycle_active=True,
            field_message_render_rescue_available=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        ready = replace(
            rendering,
            interaction=replace(
                rendering.interaction,
                field_message_render_rescue_available=False,
                interaction_phase=InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
            ),
        )
        executor = AgentActionExecutor()
        with patch("modules.agent_control.context.emulator", emulator):
            executor.execute(AgentAction(AgentActionType.ACCELERATE_DIALOGUE_RENDER), rendering)
            executor.execute(AgentAction(AgentActionType.ADVANCE_DIALOGUE), ready)

        emulator.release_button.assert_called_once_with("B")
        self.assertEqual(emulator.press_button_fresh.call_args_list, [call("B"), call("A")])

    def test_render_rescue_sends_one_fresh_b_then_holds_until_release(self):
        class StatefulEmulator:
            def __init__(self):
                self.held = set()
                self.fresh_presses = []
                self.holds = []
                self.releases = []

            def press_button_fresh(self, button):
                self.fresh_presses.append(button)

            def hold_button(self, button):
                self.holds.append(button)
                self.held.add(button)

            def is_button_held(self, button):
                return button in self.held

            def release_button(self, button):
                self.releases.append(button)
                self.held.discard(button)

        emulator = StatefulEmulator()
        rendering = observation(
            GameState.OVERWORLD,
            field_message_lifecycle_active=True,
            field_message_render_rescue_available=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        ready = replace(
            rendering,
            interaction=replace(
                rendering.interaction,
                field_message_render_rescue_available=False,
                interaction_phase=InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
            ),
        )
        executor = AgentActionExecutor()
        with patch("modules.agent_control.context.emulator", emulator):
            executor.execute(AgentAction(AgentActionType.ACCELERATE_DIALOGUE_RENDER), rendering)
            executor.execute(AgentAction(AgentActionType.ACCELERATE_DIALOGUE_RENDER), rendering)
            executor.execute(AgentAction(AgentActionType.ADVANCE_DIALOGUE), ready)

        self.assertEqual(emulator.fresh_presses, ["B", "A"])
        self.assertEqual(emulator.holds, ["B", "B"])
        self.assertEqual(emulator.releases, ["B"])

    def test_plan_invalidation_preserves_executor_owned_render_b(self):
        class StatefulEmulator:
            def __init__(self):
                self.held = set()
                self.resets = 0
                self.holds = []

            def reset_held_buttons(self):
                self.resets += 1
                self.held.clear()

            def hold_button(self, button):
                self.holds.append(button)
                self.held.add(button)

        emulator = StatefulEmulator()
        executor = AgentActionExecutor()
        executor._field_message_render_b_held = True
        loop = AgentControlLoop(lambda: None, executor=executor)

        with patch("modules.agent_control.context.emulator", emulator):
            loop._invalidate_plan("non_overworld_interaction")

        self.assertEqual(emulator.resets, 1)
        self.assertEqual(emulator.holds, ["B"])
        self.assertEqual(emulator.held, {"B"})

    def test_render_rescue_rejects_stale_or_wrong_phase_observations(self):
        emulator = Mock()
        rendering = observation(
            GameState.OVERWORLD,
            field_message_lifecycle_active=True,
            field_message_render_rescue_available=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        action = AgentAction(AgentActionType.ACCELERATE_DIALOGUE_RENDER)
        not_rendering = replace(
            rendering,
            interaction=replace(
                rendering.interaction,
                interaction_phase=InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
            ),
        )
        with patch("modules.agent_control.context.emulator", emulator):
            result = AgentActionExecutor().execute(action, not_rendering)

        self.assertEqual(result.result_type, ActionResultType.WAITING)
        emulator.press_button.assert_not_called()

    def test_safe_batch_stops_before_direction_change(self):
        world = overworld(
            {(x, y) for x, y in ((0, 0), (1, 0), (2, 0), (2, 1))},
            facing=Direction.East,
        )
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = (
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (0, 0)), (MAP, (1, 0))),
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (1, 0)), (MAP, (2, 0))),
            NavigationAction(NavigationActionType.MOVE, Direction.South, (MAP, (2, 0)), (MAP, (2, 1))),
        )
        batch = loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))
        self.assertEqual([action.direction for action in batch], [Direction.East, Direction.East])

    def test_safe_batch_defers_first_turn_to_normal_movement_path(self):
        world = overworld(
            {(0, 0), (1, 0), (2, 0)},
            facing=Direction.North,
        )
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = (
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (0, 0)), (MAP, (1, 0))),
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (1, 0)), (MAP, (2, 0))),
        )

        batch = loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))

        self.assertEqual(batch, ())

    def test_safe_batch_stops_before_warp_and_at_interaction_checkpoint(self):
        trigger = TriggerObservation(
            "talk",
            frozenset({(MAP, (3, 0))}),
            activation_locations=frozenset({(MAP, (2, 0))}),
        )
        world = overworld({(x, 0) for x in range(5)}, facing=Direction.East, triggers=(trigger,))
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = tuple(
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (x, 0)), (MAP, (x + 1, 0)))
            for x in range(4)
        )
        batch = loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[-1].destination, (MAP, (2, 0)))

    def test_safe_batch_stops_before_occupied_object_tile(self):
        world = replace(
            overworld({(0, 0), (1, 0), (2, 0), (3, 0)}, facing=Direction.East),
            objects=(
                ObjectObservation(
                    local_id=9,
                    location=(MAP, (2, 0)),
                    trainer_type="Normal",
                    trainer_defeated=True,
                ),
            ),
        )
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = tuple(
            NavigationAction(
                NavigationActionType.MOVE,
                Direction.East,
                (MAP, (x, 0)),
                (MAP, (x + 1, 0)),
            )
            for x in range(3)
        )

        batch = loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))

        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].destination, (MAP, (1, 0)))

    def test_safe_batch_does_not_checkpoint_adjacent_to_irrelevant_object(self):
        world = replace(
            overworld({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}, facing=Direction.East),
            objects=(ObjectObservation(local_id=9, location=(MAP, (3, 0)), trainer_type=None),),
        )
        loop = AgentControlLoop(lambda: None)
        loop._cached_actions = tuple(
            NavigationAction(
                NavigationActionType.MOVE,
                Direction.East,
                (MAP, (x, 0)),
                (MAP, (x + 1, 0)),
            )
            for x in range(4)
        )

        batch = loop._safe_movement_batch(observation(GameState.OVERWORLD, world=world))

        self.assertEqual([action.destination for action in batch], [(MAP, (1, 0)), (MAP, (2, 0))])

    def test_cached_route_uses_safe_local_detour_for_immediate_object(self):
        goal = ReachLocation((MAP, (2, 0)))
        route = (
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (0, 0)), (MAP, (1, 0))),
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (1, 0)), (MAP, (2, 0))),
        )
        clear_world = observation(
            GameState.OVERWORLD,
            world=overworld({(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)}, facing=Direction.East),
            goal=goal,
        )
        occupied_world = replace(
            clear_world.overworld,
            objects=(ObjectObservation(local_id=4, location=(MAP, (1, 0)), trainer_type=None),),
        )
        occupied = replace(clear_world, overworld=occupied_world)
        loop = AgentControlLoop(lambda: occupied, goal=goal, use_movement_batch=False)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan(route, route[-1].destination),
        )
        loop._cached_actions = route
        loop._cached_world_signature = loop._world_signature(clear_world)

        decision = loop._cached_decision(occupied)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(decision.action.direction, Direction.South)
        self.assertEqual(decision.action.reason, "local detour around current route obstacle")
        self.assertEqual(loop._cached_actions[-1].destination, (MAP, (2, 0)))
        self.assertGreaterEqual(len(loop._cached_actions), 4)

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
        # These tests use a lightweight emulator double.  The live-object
        # reader is exercised explicitly by the collision regression below.
        loop._live_object_coordinates = lambda map_id: frozenset()
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

    def test_running_batch_holds_direction_and_run_button_each_frame(self):
        from modules.agent_control import _MovementBatch

        loop = AgentControlLoop(lambda: None)
        loop._movement_batch = _MovementBatch(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, (1, 0)),
                    run=True,
                ),
            )
        )
        avatar = SimpleNamespace(
            map_group_and_number=MAP,
            local_coordinates=(0, 0),
            facing_direction="Right",
        )
        loop._live_object_coordinates = lambda map_id: frozenset()
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ):
            self.assertTrue(loop._advance_movement_batch())

        self.assertEqual(emulator.hold_button.call_args_list, [call("Right"), call("B")])
        emulator.press_direction.assert_not_called()

    def test_batch_accepts_intermediate_coordinate_during_forced_movement(self):
        from modules.agent_control import _MovementBatch

        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.South,
            (MAP, (0, 0)),
            (MAP, (0, 2)),
        )
        loop = AgentControlLoop(lambda: None)
        loop._movement_batch = _MovementBatch((action,))
        loop._live_object_coordinates = lambda map_id: frozenset()
        avatar = SimpleNamespace(
            map_group_and_number=MAP,
            local_coordinates=(0, 1),
            facing_direction="Down",
        )
        emulator = Mock()

        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ):
            self.assertTrue(loop._advance_movement_batch())

        emulator.hold_button.assert_called_once_with("Down")
        self.assertIsNotNone(loop._movement_batch)

    def test_in_flight_move_accepts_intermediate_coordinate_without_replanning(self):
        goal = ReachLocation((MAP, (0, 3)))
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.South,
            (MAP, (0, 0)),
            (MAP, (0, 2)),
        )
        world = overworld(
            {(0, 0), (0, 1), (0, 2), (0, 3)},
            start=(0, 1),
            facing=Direction.South,
            movement_state=MovementState.MOVING,
        )
        loop = AgentControlLoop(
            lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
            goal=goal,
            use_movement_batch=False,
        )
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((action,), action.destination),
        )
        loop._cached_actions = (action,)
        loop._cached_world_signature = loop._world_signature(observation(GameState.OVERWORLD, world=world, goal=goal))
        loop._in_flight_move = action
        loop._in_flight_move_initial_facing = Direction.South

        _, decision, result = loop.step()

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(result.result_type, ActionResultType.WAITING)
        self.assertIsNotNone(loop._cached_evaluation)
        self.assertIs(loop._in_flight_move, action)

    def test_batch_rechecks_live_object_before_dispatch(self):
        """A newly occupied route tile must cancel the fast path first."""

        loop, avatar = self._movement_batch_loop()
        del loop._live_object_coordinates
        object_event = SimpleNamespace(
            map_group_and_number=MAP,
            current_coords=(1, 0),
            flags=("active",),
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch("modules.agent_control.is_field_message_waiting_for_input", return_value=False), patch(
            "modules.agent_control.get_player_avatar", return_value=avatar
        ), patch(
            "modules.agent_control.get_map_objects", return_value=(object_event,)
        ), patch(
            "modules.agent_control.state_cache.invalidate_runtime_observations"
        ) as invalidate:
            self.assertFalse(loop._advance_movement_batch())

        invalidate.assert_called_once_with()
        self.assertIsNone(loop._movement_batch)
        emulator.hold_button.assert_not_called()
        emulator.press_direction.assert_not_called()
        emulator.reset_held_buttons.assert_called_once_with()

    def test_live_object_coordinates_reads_only_current_same_map_non_player_objects(self):
        current_map = MAP
        objects = (
            SimpleNamespace(
                map_group_and_number=current_map,
                current_coords=(1, 0),
                flags=("active",),
            ),
            SimpleNamespace(
                map_group_and_number=current_map,
                current_coords=(2, 0),
                flags=("active", "isPlayer"),
            ),
            SimpleNamespace(
                map_group_and_number=("other", 0),
                current_coords=(3, 0),
                flags=("active",),
            ),
        )

        with patch("modules.agent_control.get_map_objects", return_value=objects):
            occupied = AgentControlLoop._live_object_coordinates(current_map)

        self.assertEqual(occupied, frozenset({(1, 0)}))

    def test_fast_path_turns_without_debug_diagnostics(self):
        loop, avatar = self._movement_batch_loop()
        avatar.facing_direction = "Up"
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.context.debug", False
        ), patch("modules.agent_control.context.debug_trace", False), patch(
            "modules.agent_control.get_game_state", return_value=GameState.OVERWORLD
        ), patch(
            "modules.agent_control.is_field_message_waiting_for_input", return_value=False
        ), patch(
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

    def test_observe_agent_reuses_supplied_overworld_observation(self):
        settled_overworld = overworld({(0, 0)})
        interaction = InteractionObservation(GameState.OVERWORLD, controllable=True)
        with patch("modules.agent_control._force_universal_reobserve", return_value=False), patch(
            "modules.agent_control.observe_interaction", return_value=interaction
        ), patch("modules.agent_control.perceive_overworld") as perceive:
            result = observe_agent(overworld_observation=settled_overworld)

        self.assertIs(result.overworld, settled_overworld)
        perceive.assert_not_called()

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

    def test_world_signature_extra_field_invalidates_conservatively(self):
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (MAP, (0, 0)),
            (MAP, (1, 0)),
        )
        previous = (MAP, (), (), (), (), (), ())
        current = previous + (("new_signature_field",),)

        affects_route, detail = AgentControlLoop._signature_change_affects_remaining_route(
            previous,
            current,
            (action,),
        )

        self.assertTrue(affects_route)
        self.assertIn("field_7", detail)

    def test_world_signature_binding_field_is_named_and_invalidates(self):
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (MAP, (0, 0)),
            (MAP, (1, 0)),
        )
        previous = (MAP, (), (), (), (), (), ())
        current = (MAP, (), (), (), (), (), (("binding",),))

        affects_route, detail = AgentControlLoop._signature_change_affects_remaining_route(
            previous,
            current,
            (action,),
        )

        self.assertTrue(affects_route)
        self.assertIn("bindings", detail)

    def test_world_signature_dynamic_blocker_on_another_map_is_irrelevant(self):
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (("current", 0), (0, 0)),
            (("current", 0), (1, 0)),
        )
        previous = (("current", 0), (), (), ((9, 9),), (), (), ())
        current = (("current", 0), (), (), ((9, 8),), (), (), ())

        affects_route, detail = AgentControlLoop._signature_change_affects_remaining_route(
            previous,
            current,
            (action,),
        )

        self.assertFalse(affects_route)
        self.assertIn("relevant=none", detail)

    def test_world_signature_future_dynamic_blocker_defers_until_immediate_action(self):
        actions = (
            NavigationAction(
                NavigationActionType.MOVE,
                Direction.East,
                (MAP, (0, 0)),
                (MAP, (1, 0)),
            ),
            NavigationAction(
                NavigationActionType.MOVE,
                Direction.East,
                (MAP, (1, 0)),
                (MAP, (2, 0)),
            ),
        )
        previous = (MAP, (), (), (), (), (), ())
        current = (MAP, (), (), ((2, 0),), (), (), ())

        affects_route, detail = AgentControlLoop._signature_change_affects_remaining_route(
            previous,
            current,
            actions,
        )

        self.assertFalse(affects_route)
        self.assertIn("route_locations=3", detail)
        self.assertIn("relevant=none", detail)

    def test_world_signature_vacated_next_tile_does_not_invalidate(self):
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (MAP, (0, 0)),
            (MAP, (1, 0)),
        )
        previous = (MAP, (), (), ((1, 0),), (), (), ())
        current = (MAP, (), (), (), (), (), ())

        affects_route, detail = AgentControlLoop._signature_change_affects_remaining_route(
            previous,
            current,
            (action,),
        )

        self.assertFalse(affects_route)
        self.assertIn("relevant=none", detail)

    def test_world_signature_new_generic_runtime_object_trigger_is_deferred(self):
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (MAP, (0, 0)),
            (MAP, (1, 0)),
        )
        previous = (MAP, (), (), (), (), (), ())
        current = (
            MAP,
            (),
            (
                (
                    "object:4:Route104_EventScript_Boy1",
                    ((MAP, (3, 0)),),
                    ((MAP, (2, 0)),),
                    (),
                    None,
                    None,
                    True,
                    (),
                    None,
                    (),
                ),
            ),
            ((2, 0),),
            (),
            ((4, (MAP, (2, 0)), "Route104_EventScript_Boy1"),),
            (),
        )

        affects_route, detail = AgentControlLoop._signature_change_affects_remaining_route(
            previous,
            current,
            (action,),
        )

        self.assertFalse(affects_route)
        self.assertIn("relevant=none", detail)

    def test_world_signature_new_trainer_hazard_on_remaining_route_invalidates(self):
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (MAP, (0, 0)),
            (MAP, (1, 0)),
        )
        previous = (MAP, (), (), (), (), (), ())
        current = (
            MAP,
            (),
            (
                (
                    "static_object:28:Billy",
                    (),
                    (),
                    (),
                    None,
                    None,
                    False,
                    ((MAP, (1, 0)),),
                    "trainer",
                    (),
                ),
            ),
            (),
            (),
            (),
            (),
        )

        affects_route, detail = AgentControlLoop._signature_change_affects_remaining_route(
            previous,
            current,
            (action,),
        )

        self.assertTrue(affects_route)
        self.assertIn("route_trigger", detail)

    def test_completed_route_updates_signature_without_world_change_invalidation(self):
        goal = ReachLocation((MAP, (0, 0)))
        first = observation(GameState.OVERWORLD, world=overworld({(0, 0)}), goal=goal)
        changed_world = replace(
            first.overworld,
            dynamic_blocked_coordinates=frozenset({(1, 1)}),
            triggers=(TriggerObservation("new-trigger", frozenset()),),
        )
        changed = replace(first, overworld=changed_world)
        loop = AgentControlLoop(lambda: changed, goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((), (MAP, (0, 0))),
        )
        loop._cached_actions = (
            NavigationAction(NavigationActionType.MOVE, Direction.East, (MAP, (0, 0)), (MAP, (0, 0))),
        )
        loop._cached_action_index = 1
        loop._route_plan = SimpleNamespace(route_id=1)
        loop._cached_world_signature = loop._world_signature(first)

        with patch("modules.agent_control.print_navigation_invalidation") as invalidation:
            decision = loop._cached_decision(changed)

        self.assertIsNone(decision)
        self.assertEqual(invalidation.call_args.kwargs["reason"], "cached goal position reached")

    def test_dialogue_produces_advance_action_only_when_ready(self):
        ready = observation(GameState.OVERWORLD, dialogue_waiting=True)
        not_ready = observation(GameState.OVERWORLD, dialogue_waiting=False)
        rendering = observation(
            GameState.OVERWORLD,
            field_message_lifecycle_active=True,
            field_message_render_rescue_available=True,
            interaction_phase=InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
        )
        self.assertEqual(available_actions(ready)[0].action_type, AgentActionType.ADVANCE_DIALOGUE)
        self.assertEqual(available_actions(not_ready)[0].action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(available_actions(rendering)[0].action_type, AgentActionType.ACCELERATE_DIALOGUE_RENDER)

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
    def test_choose_yes_moves_up_from_no_with_canonical_button_name(self):
        emulator = Mock()
        prompt = observation(GameState.OVERWORLD, choices=("YES", "NO"))
        prompt = replace(
            prompt,
            interaction=replace(
                prompt.interaction,
                choice_menu_active=True,
                choice_menu_input_ready=True,
                choice_selected="NO",
                interaction_phase=InteractionPhase.CHOICE_MENU_INPUT_WAIT,
            ),
        )

        with patch("modules.agent_control.context.emulator", emulator):
            result = AgentActionExecutor().execute(
                AgentAction(AgentActionType.CHOOSE_DIALOGUE_OPTION, option="YES"),
                prompt,
            )

        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        emulator.press_button_fresh.assert_called_once_with(Direction.North.button_name)

    def test_battle_return_discards_suspended_movement_route(self):
        from modules.agent_control import _MovementBatch

        goal = ReachLocation((MAP, (2, 0)))
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (MAP, (0, 0)),
            (MAP, (1, 0)),
        )
        world = overworld({(0, 0), (1, 0), (2, 0)}, start=(0, 0), facing=Direction.East)
        loop = AgentControlLoop(lambda: observation(GameState.OVERWORLD, world=world, goal=goal), goal=goal)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((action,), action.destination),
        )
        loop._cached_actions = (action,)
        loop._cached_world_signature = loop._world_signature(observation(GameState.OVERWORLD, world=world, goal=goal))
        loop._movement_batch = _MovementBatch((action,))
        emulator = Mock()

        with patch("modules.agent_control.context.emulator", emulator):
            notify_battle_ended()
            self.assertFalse(loop._advance_movement_batch())

        self.assertIsNone(loop._movement_batch)
        self.assertIsNone(loop._cached_evaluation)
        emulator.hold_button.assert_not_called()
        self.assertEqual(emulator.reset_held_buttons.call_count, 2)

    def test_battle_end_releases_input_before_a_new_recovery_loop_is_mounted(self):
        emulator = Mock()

        with patch("modules.agent_control.context.emulator", emulator):
            notify_battle_ended()

        emulator.reset_held_buttons.assert_called_once_with()

    def test_wild_battle_return_can_refresh_runtime_without_discarding_route(self):
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            before = agent_control._battle_end_generation
            notify_battle_ended(invalidate_navigation=False)

        self.assertEqual(agent_control._battle_end_generation, before)
        emulator.reset_held_buttons.assert_called_once_with()

    def test_battle_return_replans_before_input_into_defeated_trainer(self):
        """A post-battle trainer tile must be rejected before the next input."""

        goal = ReachLocation((MAP, (2, 0)))
        world = replace(
            overworld(
                {(0, 1), (1, 1), (2, 1), (1, 0), (2, 0)},
                start=(1, 1),
                facing=Direction.North,
            ),
            # Exercise the runtime object source directly. This mirrors the
            # battle-return state where the defeated trainer remains visible
            # on the tile immediately north of the player.
            objects=(
                ObjectObservation(
                    local_id=9,
                    location=(MAP, (1, 0)),
                    trainer_type="Normal",
                    trainer_defeated=True,
                ),
            ),
        )
        stale_route = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.North,
                    (MAP, (1, 1)),
                    (MAP, (1, 0)),
                ),
            ),
            (MAP, (2, 0)),
        )
        emulator = Mock()
        loop = AgentControlLoop(
            lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
            goal=goal,
            navigation_plan=stale_route,
        )

        with patch("modules.agent_control.context.emulator", emulator):
            notify_battle_ended()
            _, decision, result = loop.step()

        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        self.assertEqual(decision.action.direction, Direction.East)
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False)])
        self.assertNotIn(call("Up", run=False, fresh=False), emulator.press_direction.call_args_list)

    def test_battle_return_stale_moving_state_is_bounded_before_replanning(self):
        """A frozen post-battle movement flag must not suppress recovery."""

        goal = ReachLocation((MAP, (2, 1)))
        world = replace(
            overworld(
                {(0, 1), (1, 1), (2, 1), (1, 0)},
                start=(1, 1),
                facing=Direction.North,
                movement_state=MovementState.MOVING,
            ),
            objects=(
                ObjectObservation(
                    local_id=9,
                    location=(MAP, (1, 0)),
                    trainer_type="Normal",
                    trainer_defeated=True,
                ),
            ),
        )
        emulator = Mock()

        with patch("modules.agent_control.context.emulator", emulator):
            emulator.get_frame_count.return_value = 100
            notify_battle_ended()
            # Recovery constructs its loop after the battle notification, so
            # generation-edge detection alone cannot identify this boundary.
            loop = AgentControlLoop(
                lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
                goal=goal,
                use_movement_batch=False,
            )
            results = [loop.step() for _ in range(loop._post_battle_stale_movement_limit)]

        self.assertTrue(all(result[1].action.action_type is AgentActionType.WAIT_REOBSERVE for result in results[:-1]))
        self.assertEqual(results[-1][1].action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(results[-1][1].action.direction, Direction.East)
        self.assertEqual(results[-1][2].result_type, ActionResultType.EXECUTED)
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False)])

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
                movement_state=MovementState.STANDING,
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

    def test_null_completed_navigation_plan_is_reported_as_unreachable(self):
        goal = ReachLocation((MAP, (1, 0)))
        world = overworld({(0, 0), (1, 0)}, start=(0, 0))
        result = evaluate_goal(
            observation(GameState.OVERWORLD, world=world, goal=goal),
            navigation_result=(None, None, None),
        )

        self.assertEqual(result.status, GoalStatus.UNREACHABLE)
        self.assertIn("navigation planner returned no plan", result.reason)

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

    def test_supplied_route_is_replanned_when_live_destination_is_occupied(self):
        goal = ReachLocation((MAP, (2, 0)))
        world = replace(
            overworld(
                {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
                start=(0, 0),
            ),
            dynamic_blocked_coordinates=frozenset({(1, 0)}),
        )
        stale_route = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, (1, 0)),
                ),
            ),
            (MAP, (2, 0)),
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(
                lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
                goal=goal,
                navigation_plan=stale_route,
            )
            _, decision, result = loop.step()

        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        self.assertEqual(decision.action.direction, Direction.South)
        self.assertEqual(emulator.press_direction.call_args_list, [call("Down", run=False, fresh=False)])
        self.assertIsNone(loop._initial_navigation_plan)
        self.assertIsNotNone(loop._cached_evaluation)

    def test_supplied_route_is_discarded_when_its_origin_is_not_observed(self):
        goal = ReachLocation((MAP, (2, 0)))
        world = overworld(
            {(0, 0), (1, 0), (2, 0)},
            start=(1, 0),
        )
        stale_route = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, (1, 0)),
                ),
            ),
            (MAP, (2, 0)),
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(
                lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
                goal=goal,
                navigation_plan=stale_route,
            )
            _, decision, result = loop.step()

        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        self.assertEqual(decision.action.direction, Direction.East)
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False)])
        self.assertIsNone(loop._initial_navigation_plan)
        self.assertIsNotNone(loop._cached_evaluation)

    def test_supplied_avoidance_route_is_replanned_when_it_crosses_trainer_hazard(self):
        goal = NavigationGoal(
            ReachLocation((MAP, (2, 0))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        trainer = TriggerObservation(
            "trainer:route",
            frozenset({(MAP, (1, 0))}),
            hazard_locations=frozenset({(MAP, (1, 0))}),
            hazard_kind="trainer",
        )
        world = overworld(
            {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
            start=(0, 0),
            triggers=(trainer,),
        )
        stale_route = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, (1, 0)),
                ),
            ),
            (MAP, (2, 0)),
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(
                lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
                goal=goal,
                navigation_plan=stale_route,
            )
            _, decision, result = loop.step()

        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        self.assertEqual(decision.action.direction, Direction.South)
        self.assertIsNone(loop._initial_navigation_plan)
        self.assertNotIn(call("Right", run=False, fresh=False), emulator.press_direction.call_args_list)

    def test_cached_route_is_replanned_when_trainer_approach_appears(self):
        goal = NavigationGoal(
            ReachLocation((MAP, (2, 0))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        clear = overworld(
            {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
            start=(0, 0),
        )
        trainer = TriggerObservation(
            "trainer:approach",
            locations=frozenset({(MAP, (1, 1))}),
            activation_locations=frozenset({(MAP, (1, 0))}),
            hazard_kind="trainer",
        )
        trainer_visible = replace(clear, triggers=(trainer,), movement_state=MovementState.STANDING)
        observations = iter(
            (
                observation(GameState.OVERWORLD, world=clear, goal=goal),
                observation(GameState.OVERWORLD, world=trainer_visible, goal=goal),
            )
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(
                lambda: next(observations),
                goal=goal,
                use_movement_batch=False,
            )
            first = loop.step()
            second = loop.step()

        self.assertEqual(first[1].action.direction, Direction.East)
        self.assertEqual(second[1].action.direction, Direction.South)
        self.assertEqual(
            emulator.press_direction.call_args_list,
            [call("Right", run=False, fresh=False), call("Down", run=False, fresh=False)],
        )

    def test_runtime_object_blocks_route_even_without_normalized_occupancy(self):
        goal = ReachLocation((MAP, (2, 0)))
        world = replace(
            overworld(
                {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
                start=(0, 0),
            ),
            objects=(
                ObjectObservation(
                    local_id=9,
                    location=(MAP, (1, 0)),
                    trainer_type="Normal",
                    trainer_defeated=True,
                ),
            ),
        )
        stale_route = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (MAP, (0, 0)),
                    (MAP, (1, 0)),
                ),
            ),
            (MAP, (2, 0)),
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(
                lambda: observation(GameState.OVERWORLD, world=world, goal=goal),
                goal=goal,
                navigation_plan=stale_route,
            )
            _, decision, result = loop.step()

        self.assertEqual(result.result_type, ActionResultType.EXECUTED)
        self.assertEqual(decision.action.direction, Direction.South)
        self.assertEqual(emulator.press_direction.call_args_list, [call("Down", run=False, fresh=False)])

    def test_selected_blocked_route_action_is_not_sent_to_emulator(self):
        goal = ReachLocation((MAP, (2, 0)))
        world = replace(
            overworld({(0, 0), (1, 0), (2, 0)}, start=(0, 0)),
            dynamic_blocked_coordinates=frozenset({(1, 0)}),
        )
        blocked_action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.East,
            (MAP, (0, 0)),
            (MAP, (1, 0)),
        )
        selected = ActionDecision(
            AgentAction(
                AgentActionType.NAVIGATE_TOWARD_GOAL,
                direction=Direction.East,
                navigation=blocked_action,
            ),
            GoalEvaluation(GoalStatus.REACHABLE, NavigationPlan((blocked_action,), (MAP, (2, 0)))),
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator), patch(
            "modules.agent_control.select_action", return_value=selected
        ):
            loop = AgentControlLoop(lambda: observation(GameState.OVERWORLD, world=world, goal=goal), goal=goal)
            _, decision, result = loop.step()

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(result.result_type, ActionResultType.WAITING)
        emulator.press_direction.assert_not_called()

    def test_failed_move_remembers_unreported_blocker_before_replanning(self):
        """Do not retry an unseen occupied tile forever after a failed step."""

        goal = ReachLocation((MAP, (2, 0)))
        world = overworld(
            {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)},
            start=(0, 0),
            facing=Direction.East,
            movement_state=MovementState.STANDING,
        )
        observations = iter(
            (
                observation(GameState.OVERWORLD, world=world, goal=goal),
                observation(GameState.OVERWORLD, world=world, goal=goal),
                observation(GameState.OVERWORLD, world=world, goal=goal),
            )
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(
                lambda: next(observations),
                goal=goal,
                use_movement_batch=False,
            )
            first = loop.step()
            second = loop.step()
            third = loop.step()

        self.assertEqual(first[1].action.direction, Direction.East)
        self.assertEqual(second[1].action.direction, Direction.East)
        self.assertEqual(third[1].action.direction, Direction.South, (third, loop._locally_blocked_destinations))
        self.assertIn((MAP, (1, 0)), loop._locally_blocked_destinations)
        self.assertEqual(
            [call_args.args[0] for call_args in emulator.press_direction.call_args_list],
            ["Right", "Right", "Down"],
        )

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

    def test_control_loop_waits_when_destination_coordinate_is_still_moving(self):
        goal = ReachLocation((MAP, (2, 0)))
        world = overworld(
            {(0, 0), (1, 0), (2, 0)},
            start=(0, 0),
            facing=Direction.East,
            movement_state=MovementState.STANDING,
        )
        moving_at_destination = overworld(
            {(0, 0), (1, 0), (2, 0)},
            start=(1, 0),
            facing=Direction.East,
            movement_state=MovementState.MOVING,
        )
        standing_at_destination = overworld(
            {(0, 0), (1, 0), (2, 0)},
            start=(1, 0),
            facing=Direction.East,
            movement_state=MovementState.STANDING,
        )
        observations = iter(
            (
                observation(GameState.OVERWORLD, world=world, goal=goal),
                observation(GameState.OVERWORLD, world=moving_at_destination, goal=goal),
                observation(GameState.OVERWORLD, world=standing_at_destination, goal=goal),
            )
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(lambda: next(observations), use_movement_batch=False)
            first = loop.step()
            settling = loop.step()
            resumed = loop.step()

        self.assertEqual(first[1].action.navigation.destination, (MAP, (1, 0)))
        self.assertEqual(settling[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(settling[2].result_type, ActionResultType.WAITING)
        self.assertEqual(resumed[1].action.navigation.destination, (MAP, (2, 0)))
        self.assertEqual(emulator.press_direction.call_args_list, [call("Right", run=False, fresh=False)] * 2)

    def test_stale_movement_at_interaction_tile_eventually_hands_off_to_interaction(self):
        from modules.goals import ActivateTrigger

        goal = ActivateTrigger("rival")
        trigger = TriggerObservation(
            "rival",
            frozenset({(MAP, (2, 0))}),
            activation_locations=frozenset({(MAP, (1, 0))}),
            activation_requirements=(((MAP, (1, 0)), Direction.East),),
        )
        world = overworld(
            {(0, 0), (1, 0), (2, 0)},
            start=(1, 0),
            facing=Direction.East,
            triggers=(trigger,),
            movement_state=MovementState.MOVING,
        )
        observed = observation(GameState.OVERWORLD, world=world, goal=goal)
        loop = AgentControlLoop(lambda: observed, goal=goal, use_movement_batch=False)
        loop._cached_goal = goal
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((), (MAP, (1, 0))),
        )
        loop._cached_world_signature = loop._world_signature(observed)
        emulator = Mock()

        with patch("modules.agent_control.context.emulator", emulator):
            results = [loop.step() for _ in range(loop._stale_interaction_movement_limit)]

        self.assertTrue(
            all(result[1].action.action_type is AgentActionType.WAIT_REOBSERVE for result in results[:-1])
        )
        self.assertEqual(results[-1][1].action.action_type, AgentActionType.INTERACT)
        self.assertEqual(results[-1][1].action.option, "rival")
        emulator.reset_held_buttons.assert_called_once_with()

    def test_stale_movement_on_new_interaction_goal_allows_initial_route_planning(self):
        from modules.goals import ActivateTrigger

        goal = ActivateTrigger("rival")
        trigger = TriggerObservation(
            "rival",
            frozenset({(MAP, (3, 0))}),
            activation_locations=frozenset({(MAP, (2, 0))}),
            activation_requirements=(((MAP, (2, 0)), Direction.East),),
        )
        world = overworld(
            {(0, 0), (1, 0), (2, 0), (3, 0)},
            start=(1, 0),
            facing=Direction.East,
            triggers=(trigger,),
            movement_state=MovementState.MOVING,
        )
        observed = observation(GameState.OVERWORLD, world=world, goal=goal)
        loop = AgentControlLoop(lambda: observed, goal=goal, use_movement_batch=False)
        emulator = Mock()

        with patch("modules.agent_control.context.emulator", emulator):
            results = [loop.step() for _ in range(loop._stale_interaction_movement_limit)]

        self.assertTrue(
            all(result[1].action.action_type is AgentActionType.WAIT_REOBSERVE for result in results[:-1])
        )
        self.assertEqual(results[-1][1].action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(results[-1][1].action.navigation.destination, (MAP, (2, 0)))
        emulator.reset_held_buttons.assert_called_once_with()

    def test_stale_movement_during_interaction_route_resumes_intermediate_navigation(self):
        from modules.goals import ActivateTrigger

        goal = ActivateTrigger("rival")
        trigger = TriggerObservation(
            "rival",
            frozenset({(MAP, (2, 0))}),
            activation_locations=frozenset({(MAP, (1, 0))}),
            activation_requirements=(((MAP, (1, 0)), Direction.East),),
        )
        world = overworld(
            {(0, 0), (1, 0), (1, 1), (2, 0)},
            start=(1, 1),
            facing=Direction.North,
            triggers=(trigger,),
            movement_state=MovementState.MOVING,
        )
        observed = observation(GameState.OVERWORLD, world=world, goal=goal)
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.North,
            (MAP, (1, 1)),
            (MAP, (1, 0)),
        )
        loop = AgentControlLoop(lambda: observed, goal=goal, use_movement_batch=False)
        loop._cached_goal = goal
        loop._cached_actions = (action,)
        loop._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan=NavigationPlan((action,), (MAP, (1, 0))),
        )
        loop._cached_world_signature = loop._world_signature(observed)
        emulator = Mock()

        with patch("modules.agent_control.context.emulator", emulator):
            results = [loop.step() for _ in range(loop._stale_interaction_movement_limit)]

        self.assertTrue(
            all(result[1].action.action_type is AgentActionType.WAIT_REOBSERVE for result in results[:-1])
        )
        self.assertEqual(results[-1][1].action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(results[-1][1].action.navigation.destination, (MAP, (1, 0)))
        emulator.reset_held_buttons.assert_called_once_with()

    def test_control_loop_waits_before_selecting_navigation_while_avatar_is_moving(self):
        goal = ReachLocation((MAP, (2, 0)))
        moving_world = overworld(
            {(0, 0), (1, 0), (2, 0)},
            start=(1, 0),
            facing=Direction.East,
            movement_state=MovementState.MOVING,
        )
        emulator = Mock()
        with patch("modules.agent_control.context.emulator", emulator):
            loop = AgentControlLoop(
                lambda: observation(GameState.OVERWORLD, world=moving_world, goal=goal),
                goal=goal,
                use_movement_batch=False,
            )
            _, decision, result = loop.step()

        self.assertEqual(decision.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(decision.action.reason, "waiting for avatar movement to settle")
        self.assertEqual(result.result_type, ActionResultType.WAITING)
        emulator.press_direction.assert_not_called()

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
        self.assertEqual(
            emulator.press_direction.call_args_list,
            [call("Right", run=False, fresh=False), call("Right", run=False, fresh=True)],
        )

    def test_implicit_step_on_warp_resumes_after_change_map_boundary(self):
        """A route that ends on a step-on warp tile has no owned WARP dispatch."""

        source_map = (0, 0)
        target_map = (0, 1)
        destination = (target_map, (0, 0))
        warp = WarpObservation((source_map, (1, 0)), destination, required_facing=None)
        goal = ReachWarp(destination_map=target_map, destination=destination, warp=warp)
        plan = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (source_map, (0, 0)),
                    (source_map, (1, 0)),
                ),
            ),
            destination,
        )

        def source_world(coordinates):
            return OverworldObservation(
                map_id=source_map,
                player_coordinates=coordinates,
                facing=Direction.East,
                controllable=True,
                movement_state=MovementState.STANDING,
                tiles=tuple(
                    TileObservation((source_map, coordinate), False, frozenset(Direction))
                    for coordinate in ((0, 0), (1, 0))
                ),
                warps=(warp,),
                objects=(),
                triggers=(),
            )

        def destination_world(controllable):
            return OverworldObservation(
                map_id=target_map,
                player_coordinates=(0, 0),
                facing=Direction.East,
                controllable=controllable,
                movement_state=MovementState.STANDING,
                tiles=(TileObservation(destination, False, frozenset(Direction)),),
                warps=(),
                objects=(),
                triggers=(),
            )

        observations = iter(
            (
                observation(GameState.OVERWORLD, world=source_world((0, 0)), goal=goal),
                observation(GameState.CHANGE_MAP, world=None, goal=goal, controllable=False),
                observation(GameState.CHANGE_MAP, world=destination_world(False), goal=goal, controllable=False),
                observation(GameState.OVERWORLD, world=destination_world(True), goal=goal),
            )
        )
        emulator = Mock()
        loop = AgentControlLoop(lambda: next(observations), goal=goal, navigation_plan=plan)

        with patch("modules.agent_control.context.emulator", emulator):
            movement = loop.step()
            source_boundary = loop.step()
            destination_boundary = loop.step()
            resumed = loop.step()

        self.assertEqual(movement[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(source_boundary[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(destination_boundary[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(resumed[2].result_type, ActionResultType.GOAL_COMPLETE)
        self.assertFalse(loop.navigation_transition_active)

    def test_warp_route_survives_non_overworld_boundary_before_destination_action(self):
        """A script boundary after warp activation must preserve resumption."""

        source_map = (0, 0)
        target_map = (0, 1)
        destination = (target_map, (0, 0))
        warp = WarpObservation(
            (source_map, (1, 0)),
            destination,
            required_facing=Direction.East,
        )
        goal = ReachWarp(destination_map=target_map, destination=destination, warp=warp)
        plan = NavigationPlan(
            (
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (source_map, (0, 0)),
                    (source_map, (1, 0)),
                ),
                NavigationAction(
                    NavigationActionType.WARP,
                    Direction.East,
                    (source_map, (1, 0)),
                    destination,
                    transition_kind="warp",
                ),
            ),
            destination,
        )

        def source_world(coordinates, controllable=True):
            return OverworldObservation(
                map_id=source_map,
                player_coordinates=coordinates,
                facing=Direction.East,
                controllable=controllable,
                movement_state=MovementState.STANDING,
                tiles=tuple(
                    TileObservation((source_map, coordinate), False, frozenset(Direction))
                    for coordinate in ((0, 0), (1, 0))
                ),
                warps=(warp,),
                objects=(),
                triggers=(),
            )

        def destination_world(controllable):
            return OverworldObservation(
                map_id=target_map,
                player_coordinates=(0, 0),
                facing=Direction.East,
                controllable=controllable,
                movement_state=MovementState.STANDING,
                tiles=(TileObservation(destination, False, frozenset(Direction)),),
                warps=(),
                objects=(),
                triggers=(),
            )

        observations = iter(
            (
                observation(GameState.OVERWORLD, world=source_world((0, 0)), goal=goal),
                observation(GameState.OVERWORLD, world=source_world((1, 0)), goal=goal),
                observation(
                    GameState.CHANGE_MAP,
                    world=None,
                    goal=goal,
                    controllable=False,
                ),
                observation(
                    GameState.CHANGE_MAP,
                    world=destination_world(False),
                    goal=goal,
                    controllable=False,
                ),
                observation(GameState.OVERWORLD, world=destination_world(True), goal=goal),
            )
        )
        emulator = Mock()
        loop = AgentControlLoop(lambda: next(observations), goal=goal, navigation_plan=plan)

        with patch("modules.agent_control.context.emulator", emulator):
            movement = loop.step()
            activation = loop.step()
            source_boundary = loop.step()
            destination_boundary = loop.step()
            cached_actions_at_destination = loop._cached_action_index
            resumed = loop.step()

        self.assertEqual(movement[1].action.navigation.action_type, NavigationActionType.MOVE)
        self.assertEqual(activation[1].action.navigation.action_type, NavigationActionType.WARP)
        self.assertEqual(source_boundary[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(destination_boundary[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(resumed[2].result_type, ActionResultType.GOAL_COMPLETE)
        self.assertTrue(loop.navigation_transition_active is False)
        self.assertEqual(cached_actions_at_destination, len(plan.actions))
        self.assertEqual(
            emulator.press_direction.call_args_list,
            [call("Right", run=False, fresh=False), call("Right", run=False, fresh=True)],
        )

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
        self.assertEqual(
            emulator.press_direction.call_args_list,
            [call("Right", run=False, fresh=False), call("Right", run=False, fresh=True)],
        )

    def test_executor_rejects_input_for_unknown_state(self):
        action = available_actions(observation(GameState.UNKNOWN))[0]
        result = AgentActionExecutor().execute(action, observation(GameState.UNKNOWN))
        self.assertEqual(result.result_type, ActionResultType.WAITING)

    def test_run_propagates_unreachable_instead_of_reporting_completion(self):
        loop = AgentControlLoop(lambda: observation(GameState.UNKNOWN))
        action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason="blocked")
        terminal = ActionResult(ActionResultType.UNREACHABLE, action, "no route")
        with patch.object(
            loop,
            "step",
            side_effect=((None, None, terminal), (None, None, terminal)),
        ):
            with self.assertRaises(NavigationError):
                list(loop.run())

    def test_run_registers_active_loop_while_suspended(self):
        loop = AgentControlLoop(lambda: observation(GameState.UNKNOWN))
        action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason="transition")
        waiting = ActionResult(ActionResultType.WAITING, action, "transition")
        previous = object()

        with patch.object(agent_control.context, "active_agent_control_loop", previous), patch.object(
            loop,
            "step",
            side_effect=((None, None, waiting),),
        ):
            runner = loop.run()
            next(runner)
            self.assertIs(agent_control.context.active_agent_control_loop, loop)
            runner.close()
            self.assertIs(agent_control.context.active_agent_control_loop, previous)

    def test_run_retries_terminal_navigation_once_after_twenty_frames(self):
        loop = AgentControlLoop(lambda: observation(GameState.UNKNOWN))
        action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason="stale route")
        terminal = ActionResult(ActionResultType.UNREACHABLE, action, "stale route")
        complete = ActionResult(ActionResultType.GOAL_COMPLETE, action, "recovered")
        emulator = Mock()

        with patch.object(loop, "step", side_effect=((None, None, terminal), (None, None, complete))) as step, patch(
            "modules.agent_control.context.emulator", emulator
        ):
            yielded = list(loop.run())

        assert len(yielded) == 20
        assert step.call_count == 2
        assert emulator.reset_held_buttons.call_count >= 2

    def test_run_makes_second_terminal_navigation_failure_fatal(self):
        loop = AgentControlLoop(lambda: observation(GameState.UNKNOWN))
        action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason="unreachable")
        terminal = ActionResult(ActionResultType.UNREACHABLE, action, "unreachable")
        emulator = Mock()

        with patch.object(loop, "step", side_effect=((None, None, terminal), (None, None, terminal))) as step, patch(
            "modules.agent_control.context.emulator", emulator
        ):
            with self.assertRaises(NavigationError):
                list(loop.run())

        assert step.call_count == 2
        assert emulator.reset_held_buttons.call_count >= 2

    def test_preferred_map_route_retains_controller_for_a_second_recovery(self):
        goal = NavigationGoal(
            SemanticTarget.map((0, 18)),
            preferred_map_route=((0, 9), (0, 16), (0, 10), (0, 18)),
        )
        loop = AgentControlLoop(lambda: observation(GameState.UNKNOWN), goal=goal)
        action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason="stale route")
        terminal = ActionResult(ActionResultType.UNREACHABLE, action, "stale route")
        emulator = Mock()

        with patch.object(
            loop,
            "step",
            side_effect=((None, None, terminal), (None, None, terminal), (None, None, terminal)),
        ) as step, patch("modules.agent_control.context.emulator", emulator):
            with self.assertRaises(NavigationError):
                list(loop.run())

        assert step.call_count == 3
        assert emulator.reset_held_buttons.call_count >= 3

    def test_run_retries_navigation_error_raised_by_movement_batch_once(self):
        loop = AgentControlLoop(lambda: observation(GameState.UNKNOWN))
        loop._movement_batch = object()
        action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason="movement batch")
        complete = ActionResult(ActionResultType.GOAL_COMPLETE, action, "recovered")
        emulator = Mock()

        with patch.object(
            loop,
            "_advance_movement_batch",
            side_effect=(NavigationError("movement batch"), False),
        ) as advance, patch.object(loop, "step", return_value=(None, None, complete)) as step, patch(
            "modules.agent_control.context.emulator", emulator
        ):
            yielded = list(loop.run())

        assert len(yielded) == 20
        assert advance.call_count == 1
        step.assert_called_once_with()
        assert emulator.reset_held_buttons.call_count >= 2


class NavigationWorldSummaryTests(TestCase):
    def test_summary_reports_hazards_and_whether_the_start_is_inside_one(self):
        from modules.agent_control import _navigation_world_summary

        map_id = (0, 19)
        world = SimpleNamespace(
            tiles=tuple(range(12)),
            triggers=(
                SimpleNamespace(
                    affordance_id="trainer:(0, 19):26",
                    hazard_locations=frozenset({(map_id, (11, 41)), (map_id, (11, 42))}),
                ),
                SimpleNamespace(affordance_id="not_a_trainer", hazard_locations=frozenset()),
            ),
        )
        goal = NavigationGoal(
            ReachLocation((map_id, (1, 1))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        inside = _navigation_world_summary(world, (map_id, (11, 41)), goal)
        outside = _navigation_world_summary(world, (map_id, (10, 45)), goal)

        self.assertIn("trainer_mode=AVOID", inside)
        self.assertIn("hazard_tiles=2", inside)
        self.assertIn("trainer:(0, 19):26:2", inside)
        self.assertIn("start_inside_hazard_of=['trainer:(0, 19):26']", inside)
        self.assertIn("start_inside_hazard_of=[]", outside)
        self.assertNotIn("not_a_trainer", inside)

    def test_summary_never_raises(self):
        from modules.agent_control import _navigation_world_summary

        # A hostile world must not break navigation diagnostics.
        broken = SimpleNamespace(triggers=object())
        summary = _navigation_world_summary(broken, (0, 0), None)

        self.assertIn("world_summary_unavailable", summary)


class SlowSearchDumpTests(TestCase):
    def test_dump_is_disabled_without_the_environment_variable(self):
        from modules.agent_control import _dump_slow_navigation_search

        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("POKEBOT_DUMP_SLOW_SEARCH", None)
            with patch("modules.agent_control.print_always") as printed:
                _dump_slow_navigation_search(object(), (0, 0), object(), 99999.0)
        printed.assert_not_called()

    def test_dump_skips_fast_failures_and_writes_slow_ones(self):
        import os
        import pickle
        import tempfile

        from modules.agent_control import _dump_slow_navigation_search

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"POKEBOT_DUMP_SLOW_SEARCH": directory}
        ), patch("modules.agent_control.print_always"):
            _dump_slow_navigation_search({"tiles": 1}, ((0, 1), (2, 3)), ("goal",), 500.0)
            self.assertEqual(os.listdir(directory), [])
            _dump_slow_navigation_search({"tiles": 1}, ((0, 1), (2, 3)), ("goal",), 12000.0)
            files = os.listdir(directory)
            self.assertEqual(len(files), 1)
            with open(os.path.join(directory, files[0]), "rb") as handle:
                payload = pickle.load(handle)
        self.assertEqual(payload["start"], ((0, 1), (2, 3)))
        self.assertEqual(payload["goal"], ("goal",))

    def test_dump_failure_never_raises(self):
        from modules.agent_control import _dump_slow_navigation_search

        with patch.dict("os.environ", {"POKEBOT_DUMP_SLOW_SEARCH": "/proc/definitely/not/writable"}), patch(
            "modules.agent_control.print_always"
        ):
            _dump_slow_navigation_search(lambda: None, (0, 0), None, 99999.0)  # unpicklable


class TrainerHazardValidatorTests(TestCase):
    """The final input gate must agree with the planner's trainer-hazard model."""

    MAP = (0, 19)

    def _cindy(self, *, hazards, activation):
        return SimpleNamespace(
            hazard_kind="trainer",
            kind="trainer_hazard",
            hazard_locations=frozenset((self.MAP, tile) for tile in hazards),
            activation_locations=frozenset((self.MAP, tile) for tile in activation),
        )

    def test_adjacent_interaction_tile_is_not_a_hazard_when_sight_lines_are_known(self):
        # Cindy at (11, 44) faces Up/Right: the tile behind her, (11, 45), is
        # only an interaction position and must not close the corridor.
        world = SimpleNamespace(
            map_id=self.MAP,
            triggers=(
                self._cindy(
                    hazards=((11, 41), (11, 42), (11, 43), (12, 44), (13, 44), (14, 44)),
                    activation=((10, 44), (11, 43), (11, 45), (12, 44)),
                ),
            ),
        )

        hazards = agent_control.AgentControlLoop._trainer_hazard_coordinates(world)

        self.assertNotIn((self.MAP, (11, 45)), hazards)
        self.assertNotIn((self.MAP, (10, 44)), hazards)
        self.assertIn((self.MAP, (11, 43)), hazards)
        self.assertIn((self.MAP, (12, 44)), hazards)

    def test_activation_tiles_stay_hazardous_when_sight_line_geometry_is_unknown(self):
        world = SimpleNamespace(
            map_id=self.MAP,
            triggers=(self._cindy(hazards=(), activation=((11, 45), (10, 44))),),
        )

        hazards = agent_control.AgentControlLoop._trainer_hazard_coordinates(world)

        self.assertEqual(hazards, {(self.MAP, (11, 45)), (self.MAP, (10, 44))})

    def test_planned_step_onto_an_interaction_tile_is_not_blocked_under_avoid(self):
        world = SimpleNamespace(
            map_id=self.MAP,
            triggers=(
                self._cindy(
                    hazards=((11, 41), (11, 42), (11, 43), (12, 44), (13, 44), (14, 44)),
                    activation=((10, 44), (11, 43), (11, 45), (12, 44)),
                ),
            ),
        )
        action = NavigationAction(
            NavigationActionType.MOVE,
            Direction.West,
            (self.MAP, (12, 45)),
            (self.MAP, (11, 45)),
        )
        goal = NavigationGoal(
            ReachLocation((self.MAP, (10, 38))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        loop = agent_control.AgentControlLoop(lambda: None, goal=goal)
        observation = SimpleNamespace(overworld=world)

        with patch.object(
            agent_control.AgentControlLoop, "_blocked_cached_action", return_value=None
        ):
            blocked = loop._blocked_action(observation, action)

        self.assertIsNone(blocked)
