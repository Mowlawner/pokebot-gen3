import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import call, patch

from modules.context import context
from modules.map_data import MapRSE
from modules.goals import (
    EngageTrainer,
    GoalConstraints,
    Goal,
    NavigationGoal,
    ReachLocation,
    SemanticTarget,
    TrainerEngagementMode,
    TrainerEngagementPlan,
    TrainerMode,
)
from modules.map_path import Direction
from modules.navigation import IntermediateRouteAnalysis, NavigableTile, NavigationWorld, RouteAnalysis
from modules.overworld import TriggerObservation
from modules.nuzlocke.campaign_controller import (
    CampaignController,
    CampaignControllerStatus,
    _PREPARATION_LEVEL_CAP_UNAVAILABLE,
    _actionable_scripted_interaction_is_active,
    _campaign_battle_transition_active,
    _campaign_target_for_execution,
    _retention_order_allows,
    _story_progression_owns_restock_boundary,
    _trainer_preparation_ceiling_reached,
)
from modules.nuzlocke.campaign_status import CampaignStatus
from modules.nuzlocke.campaign_execution import (
    CampaignExecutionResult,
    CampaignExecutionStatus,
    adapt_campaign_execution,
)
from modules.nuzlocke.campaign_controller import CampaignTrainerLeadFailure
from modules.menuing import MenuRotationFailure
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    ObjectiveSelection,
    ObjectiveStatus,
    initial_emerald_campaign,
    plan_campaign,
    restock_recovery_items_objective,
    select_campaign_objective,
    ultimate_emerald_campaign_goal,
)
from modules.nuzlocke.campaign_state import Fact, FactStatus, RunStatus
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    PartyReadinessMember,
    ProgressionReadinessDiagnostic,
    ReadinessDecision,
    ReadinessReason,
)
from modules.trainer_checkpoints import ROUTE102_FIRST_POOCHYENA
from modules.interaction_state import InteractionObservation, InteractionPhase
from modules.memory import GameState

from tests.test_nuzlocke_campaign_objectives import CampaignObjectiveTests


class CampaignControllerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CampaignObjectiveTests()
        self.fixture.setUp()
        self.current = self.fixture.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
            )
        )
        self.created: list[Goal] = []
        self.executed = 0

    def tearDown(self):
        self.fixture.tearDown()

    def state(self, **changes):
        self.current = replace(self.current, **changes)

    def factory(self, goal):
        self.created.append(goal)

        def tactical():
            self.executed += 1
            yield
            while True:
                yield

        return tactical()

    def controller(self):
        return CampaignController(lambda: self.current, tactical_loop_factory=self.factory)

    def test_trainer_staging_selects_reachable_side_of_sightline(self):
        map_id = MapRSE.ROUTE102.value
        # Calvin's sightline forms a trainer-avoidance wall.  The left-side
        # staging tiles are valid geometry, but the current position can only
        # reach the right-side candidates without entering the hazard.
        coordinates = {
            (x, y)
            for x in range(34, 40)
            for y in range(6, 18)
        }
        coordinates.update(
            {
                (32, 15),
                (32, 16),
                (32, 17),
                (33, 15),
                (33, 16),
                (33, 17),
                (34, 15),
                (34, 16),
                (34, 17),
            }
        )
        coordinates.remove((34, 14))
        tiles = {
            (map_id, coordinate): NavigableTile(
                (map_id, coordinate),
                allowed_directions=frozenset(Direction),
            )
            for coordinate in coordinates
        }
        # The occupied trainer tile is present in the geometry but cannot be
        # entered by the live navigation world.
        tiles[(map_id, (33, 14))] = NavigableTile(
            (map_id, (33, 14)),
            blocked=True,
            allowed_directions=frozenset(Direction),
        )
        exposure = frozenset((map_id, coordinate) for coordinate in ((33, 15), (33, 16), (33, 17)))
        trigger = TriggerObservation(
            "calvin",
            locations=frozenset({(map_id, (33, 14))}),
            affordance_id="calvin",
            hazard_kind="trainer",
            hazard_locations=exposure,
            engagement_locations=exposure,
        )
        world = NavigationWorld(tiles=tiles, triggers=(trigger,))
        engagement_plan = TrainerEngagementPlan(
            "calvin",
            "calvin",
            TrainerEngagementMode.SIGHTLINE,
            engagement_locations=exposure,
        )

        selected = CampaignController._trainer_staging_location(
            world,
            "calvin",
            engagement_plan,
            start=(map_id, (39, 6)),
            constraints=GoalConstraints(
                trainer_mode=TrainerMode.AVOID,
                avoid_trainer_ids=frozenset({"calvin"}),
            ),
        )

        self.assertEqual(selected, (map_id, (34, 15)))

    def test_trainer_preparation_ceiling_stops_one_level_above_known_roster(self):
        readiness = SimpleNamespace(
            trainer_checkpoint=SimpleNamespace(party=(SimpleNamespace(level=5),)),
        )
        party = (SimpleNamespace(is_egg=False, current_hp=20, level=6),)

        with patch("modules.pokemon_party.get_party", return_value=party):
            self.assertEqual(_trainer_preparation_ceiling_reached(readiness), (True, 6, 6))

    def test_readiness_interaction_probe_does_not_perceive_overworld(self):
        """A transient map boundary must not trigger a nested full map read."""

        interaction = SimpleNamespace(game_state=GameState.OVERWORLD)
        with patch(
            "modules.nuzlocke.campaign_controller.observe_interaction",
            return_value=interaction,
        ), patch(
            "modules.nuzlocke.campaign_controller.observe_agent",
            side_effect=AssertionError("readiness probe must not scan the overworld"),
        ):
            self.assertFalse(_actionable_scripted_interaction_is_active())

    def test_readiness_interaction_probe_ignores_stale_script_in_manual_mode(self):
        """A shared test emulator cannot grant campaign ownership in Manual."""

        interaction = SimpleNamespace(game_state=GameState.OVERWORLD)
        with patch.object(context, "_current_bot_mode", "Manual"), patch(
            "modules.nuzlocke.campaign_controller.observe_interaction",
            return_value=interaction,
        ) as observe:
            self.assertFalse(_actionable_scripted_interaction_is_active())
        observe.assert_not_called()

    def test_readiness_interaction_probe_preserves_non_actionable_script_handoff(self):
        """The tactical loop must survive a script's native wait before dialogue."""

        interaction = InteractionObservation(
            game_state=GameState.OVERWORLD,
            script_active=True,
            interaction_phase=InteractionPhase.SCRIPT_NATIVE_WAIT,
        )
        with patch.object(context, "_current_bot_mode", "Campaign Progression"), patch(
            "modules.nuzlocke.campaign_controller.observe_interaction",
            return_value=interaction,
        ):
            self.assertTrue(_actionable_scripted_interaction_is_active())

    def test_battle_transition_tasks_own_campaign_boundary_before_game_state_changes(self):
        with patch(
            "modules.nuzlocke.campaign_controller.get_game_state",
            return_value=GameState.OVERWORLD,
        ), patch(
            "modules.nuzlocke.campaign_controller.get_tasks",
            return_value=(SimpleNamespace(symbol="Task_BattleTransition_Intro"),),
        ):
            self.assertEqual(
                _campaign_battle_transition_active(),
                (True, "tasks:Task_BattleTransition_Intro"),
            )

    def test_campaign_does_not_refresh_or_advance_during_battle_transition(self):
        controller = self.controller()
        controller.status = CampaignControllerStatus.READY
        controller.current_objective_id = "complete_intro_rival"
        advanced = []

        def tactical_loop():
            advanced.append(True)
            yield

        controller._tactical_loop = tactical_loop()
        controller._tactical_loop_objective_id = controller.current_objective_id
        controller._refresh_required = True

        with patch(
            "modules.nuzlocke.campaign_controller.get_game_state",
            return_value=GameState.BATTLE_STARTING,
        ), patch(
            "modules.nuzlocke.campaign_controller.get_tasks",
            return_value=(),
        ):
            state = controller.step()

        self.assertEqual(state.status, CampaignControllerStatus.READY)
        self.assertEqual(advanced, [])
        self.assertTrue(controller._refresh_required)

    def test_default_tactical_loop_adopts_matching_recovery_continuation_route(self):
        objective_id = "complete_petalburg_wally"
        goal = NavigationGoal(SemanticTarget.map((0, 17)))
        continuation = SimpleNamespace(
            actions=(SimpleNamespace(source=((0, 10), (6, 16))),),
            destination=((0, 17), (49, 10)),
        )
        controller = CampaignController(lambda: self.current)
        controller.current_objective_id = objective_id
        fake_loop = SimpleNamespace(run=lambda: iter(()))

        with patch.object(context, "campaign_continuation_route", (objective_id, continuation)), patch(
            "modules.nuzlocke.campaign_controller.AgentControlLoop",
            return_value=fake_loop,
        ) as agent_loop:
            controller._default_tactical_loop(goal)

        agent_loop.assert_called_once()
        self.assertIs(agent_loop.call_args.kwargs["navigation_plan"], continuation)
        self.assertIsNone(context.campaign_continuation_route)

    def test_default_tactical_loop_replans_promoted_trainer_goal_after_recovery(self):
        objective_id = "obtain_encounter:0:17"
        goal = NavigationGoal(EngageTrainer("trainer:(0, 17):2"))
        continuation = SimpleNamespace(
            actions=(SimpleNamespace(source=((0, 10), (6, 16))),),
            destination=((0, 17), (49, 10)),
        )
        controller = CampaignController(lambda: self.current)
        controller.current_objective_id = objective_id
        fake_loop = SimpleNamespace(run=lambda: iter(()))

        with patch.object(context, "campaign_continuation_route", (objective_id, continuation)), patch(
            "modules.nuzlocke.campaign_controller.AgentControlLoop",
            return_value=fake_loop,
        ) as agent_loop:
            controller._default_tactical_loop(goal)

        # Trainer exposure now owns a staging generator. It discards the
        # recovery continuation immediately and creates the final navigation
        # loop only after the staging tile and readiness-selected lead are
        # observed.
        agent_loop.assert_not_called()
        self.assertIsNone(context.campaign_continuation_route)

    def test_trainer_exposure_stages_before_sightline_and_rotates_lead_before_engaging(self):
        trainer_id = "trainer:(0, 17):2"
        target = ((0, 17), (5, 5))
        staging = ((0, 17), (4, 5))
        world = SimpleNamespace(
            triggers=(
                SimpleNamespace(
                    affordance_id=trainer_id,
                    hazard_locations=frozenset({target}),
                ),
            ),
            tiles={
                staging: SimpleNamespace(blocked=False),
                target: SimpleNamespace(blocked=False),
            },
            neighbors=lambda location: (("East", target, False),) if location == staging else (),
        )
        engagement_plan = SimpleNamespace(
            mode=TrainerEngagementMode.SIGHTLINE,
            engagement_locations=frozenset({target}),
            reason="fixture sight line",
        )
        observation = SimpleNamespace(
            overworld=SimpleNamespace(
                map_id=(0, 17),
                player_coordinates=staging[1],
                controllable=True,
            )
        )
        lead_verified = []
        ensure_calls = []

        def rotate_lead(*args, **kwargs):
            ensure_calls.append((args, kwargs))
            lead_verified.append(True)
            yield "lead-rotation"
            return True

        class EngagementLoop:
            def run(self):
                assert lead_verified == [True]
                return iter(("engaged",))

        controller = self.controller()
        previous_readiness = getattr(context, "campaign_trainer_readiness", None)
        context.campaign_trainer_readiness = SimpleNamespace(lead_party_index=1)
        try:
            with (
                patch.object(controller, "_trainer_map_for_goal", return_value=(0, 17)),
                patch(
                    "modules.nuzlocke.campaign_controller.observe_agent",
                    return_value=observation,
                ),
                patch(
                    "modules.nuzlocke.campaign_controller.NavigationWorld.from_overworld",
                    return_value=world,
                ),
                patch(
                    "modules.nuzlocke.campaign_controller.resolve_trainer_engagement_plan",
                    return_value=engagement_plan,
                ),
                patch(
                    "modules.nuzlocke.campaign_controller.ensure_campaign_trainer_lead",
                    new=rotate_lead,
                ),
                patch(
                    "modules.nuzlocke.campaign_controller.AgentControlLoop",
                    return_value=EngagementLoop(),
                ) as agent_loop,
                patch("modules.nuzlocke.campaign_controller.diagnostic_print"),
            ):
                events = list(controller._trainer_exposure_tactical_loop(NavigationGoal(EngageTrainer(trainer_id))))
        finally:
            context.campaign_trainer_readiness = previous_readiness

        assert events == ["lead-rotation", "engaged"]
        assert ensure_calls == [
            (
                (1,),
                {
                    "trainer_id": trainer_id,
                    "reason": "final trainer exposure staging",
                },
            )
        ]
        agent_loop.assert_called_once()
        assert agent_loop.call_args.kwargs["goal"].target == EngageTrainer(trainer_id)

    def test_trainer_staging_handoff_does_not_rerank_from_the_staging_tile(self):
        trainer_id = "trainer:(0, 17):2"
        start = ((0, 17), (6, 5))
        staging = ((0, 17), (4, 5))
        engagement_plan = SimpleNamespace(
            mode=TrainerEngagementMode.SIGHTLINE,
            engagement_locations=frozenset({((0, 17), (5, 5))}),
            reason="fixture sight line",
        )
        world = SimpleNamespace(
            revision_key=lambda: ((), ()),
            triggers=(),
            tiles={
                start: SimpleNamespace(blocked=False),
                staging: SimpleNamespace(blocked=False),
            },
        )
        observations = iter(
            (
                SimpleNamespace(
                    overworld=SimpleNamespace(
                        map_id=(0, 17), player_coordinates=start[1], controllable=True
                    )
                ),
                SimpleNamespace(
                    overworld=SimpleNamespace(
                        map_id=(0, 17), player_coordinates=staging[1], controllable=True
                    )
                ),
            )
        )

        class StageLoop:
            def run(self):
                yield "stage"

        class EngagementLoop:
            def run(self):
                yield "engaged"

        def rotate_lead(*_args, **_kwargs):
            yield "lead"
            return True

        controller = self.controller()
        previous_readiness = getattr(context, "campaign_trainer_readiness", None)
        context.campaign_trainer_readiness = SimpleNamespace(lead_party_index=1)
        try:
            with (
                patch.object(controller, "_trainer_map_for_goal", return_value=(0, 17)),
                patch(
                    "modules.nuzlocke.campaign_controller.observe_agent",
                    side_effect=lambda: next(observations),
                ),
                patch(
                    "modules.nuzlocke.campaign_controller.NavigationWorld.from_overworld",
                    return_value=world,
                ),
                patch(
                    "modules.nuzlocke.campaign_controller.resolve_trainer_engagement_plan",
                    return_value=engagement_plan,
                ),
                patch.object(controller, "_trainer_staging_location", return_value=staging) as choose_staging,
                patch(
                    "modules.nuzlocke.campaign_controller.ensure_campaign_trainer_lead",
                    new=rotate_lead,
                ),
                patch(
                    "modules.nuzlocke.campaign_controller.AgentControlLoop",
                    side_effect=(StageLoop(), EngagementLoop()),
                ),
                patch("modules.nuzlocke.campaign_controller.diagnostic_print"),
            ):
                events = list(
                    controller._trainer_exposure_tactical_loop(
                        NavigationGoal(EngageTrainer(trainer_id))
                    )
                )
        finally:
            context.campaign_trainer_readiness = previous_readiness

        assert events == ["stage", "lead", "engaged"]
        choose_staging.assert_called_once()

    def test_trainer_lead_failure_blocks_campaign_with_structured_reason(self):
        objective = self.fixture.objective("route102_trainer", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "trainer route")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "trainer route",
            tactical_goal=Goal(),
        )
        failure = CampaignTrainerLeadFailure(
            "calvin",
            MenuRotationFailure("select_second_pokemon", "replacement cursor did not reach target"),
        )

        def failing_loop():
            raise failure
            yield

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            tactical_loop_factory=lambda _goal: failing_loop(),
        )
        controller.current_objective_id = objective.objective_id
        controller.current_tactical_goal = execution.tactical_goal
        controller.last_selection = selection
        controller.last_execution = execution
        controller._tactical_loop = failing_loop()
        controller._tactical_loop_objective_id = objective.objective_id
        controller._refresh_required = False
        controller.status = CampaignControllerStatus.READY

        blocked = controller.step()

        self.assertEqual(blocked.execution_phase, "BLOCKED")
        self.assertEqual(blocked.status, CampaignControllerStatus.BLOCKED)
        self.assertIn("replacement cursor", blocked.reason)
        self.assertEqual(context.campaign_status.intent, "Trainer lead rotation failed; manual review required")

    def test_promoted_trainer_goal_publishes_trainer_target(self):
        objective = SimpleNamespace(destination=((0, 17), (49, 10)))
        goal = NavigationGoal(EngageTrainer("trainer:(0, 17):2"))

        self.assertEqual(
            _campaign_target_for_execution(objective, goal),
            SemanticTarget.interaction((0, 17), "trainer:(0, 17):2"),
        )

    def test_healthy_underprepared_known_trainer_mounts_bounded_preparation(self):
        objective = self.fixture.objective("route102_trainer", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "known trainer route")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "trainer route",
            tactical_goal=Goal(),
        )
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=(0, 17),
            navigation_goal=Goal(),
            current_map=(0, 17),
            current_coordinates=(4, 4),
            game_state="OVERWORLD",
            campaign_mode="Campaign Progression",
            party=(PartyReadinessMember(0, "Treecko", 5, 20, 1.0, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=None,
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.PREPARE,
            readiness_reason=ReadinessReason.TRAINER_BATTLE_UNREADY,
            trainer_checkpoint=ROUTE102_FIRST_POOCHYENA,
            trainer_engagement_required=True,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.execute_campaign_preparation",
            return_value=iter(()),
        ) as prepare, patch.object(
            CampaignController,
            "_active_preparation_level_cap",
            return_value=6,
        ):
            controller = CampaignController(
                lambda: self.current,
                selector=lambda _: selection,
                adapter=lambda _: execution,
                readiness_provider=lambda *_: readiness,
            )

            mounted = controller.step()
            self.assertEqual(mounted.execution_phase, "PREPARATION", mounted.reason)
            prepare.assert_called_once_with((0, 17), target_level=5)
            completed = controller.step()
            self.assertEqual(completed.execution_phase, "CAMPAIGN")
            self.assertEqual(completed.status, CampaignControllerStatus.READY)
            escalated = controller.step()
            self.assertEqual(escalated.execution_phase, "PREPARATION")
            prepare.assert_has_calls(
                [
                    call((0, 17), target_level=5),
                    call((0, 17), target_level=6),
                ]
            )
            completed = controller.step()
            self.assertEqual(completed.execution_phase, "CAMPAIGN")
            blocked = controller.step()
            self.assertEqual(blocked.execution_phase, "BLOCKED")
            self.assertEqual(blocked.status, CampaignControllerStatus.BLOCKED)

    def test_scripted_trainer_capability_keeps_ownership_during_preparation(self):
        objective = next(
            item for item in initial_emerald_campaign() if item.objective_id == "complete_petalburg_woods"
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "scripted trainer gate")
        scripted_capability = lambda: iter(())
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "scripted trainer capability",
            tactical_goal=None,
            capability=scripted_capability,
        )
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=MapRSE.PETALBURG_WOODS.value,
            navigation_goal=None,
            current_map=MapRSE.PETALBURG_WOODS.value,
            current_coordinates=(8, 8),
            game_state="OVERWORLD",
            campaign_mode="Campaign Progression",
            party=(PartyReadinessMember(0, "Treecko", 5, 20, 1.0, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=None,
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.NOT_APPLICABLE,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.PREPARE,
            readiness_reason=ReadinessReason.TRAINER_BATTLE_UNREADY,
            trainer_checkpoint=SimpleNamespace(
                checkpoint_id="petalburg_woods:aqua_grunt",
                map_id=MapRSE.PETALBURG_WOODS.value,
                party=(SimpleNamespace(level=5),),
            ),
            trainer_engagement_required=True,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.execute_campaign_preparation",
            return_value=iter(()),
        ) as prepare, patch.object(
            CampaignController,
            "_active_preparation_level_cap",
            return_value=6,
        ):
            controller = CampaignController(
                lambda: self.current,
                selector=lambda _: selection,
                adapter=lambda _: execution,
                readiness_provider=lambda *_: readiness,
            )

            mounted = controller.step()
            self.assertEqual(mounted.execution_phase, "PREPARATION")
            self.assertIs(controller.last_execution.capability, scripted_capability)
            completed = controller.step()

        self.assertEqual(completed.execution_phase, "CAMPAIGN")
        self.assertIs(controller.last_execution.capability, scripted_capability)
        prepare.assert_called_once_with(MapRSE.PETALBURG_WOODS.value, target_level=5)

    def test_trainer_preparation_defers_when_active_cap_is_unavailable(self):
        objective = self.fixture.objective("route102_trainer", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "known trainer route")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "trainer route",
            tactical_goal=Goal(),
        )
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=(0, 17),
            navigation_goal=Goal(),
            current_map=(0, 17),
            current_coordinates=(4, 4),
            game_state="OVERWORLD",
            campaign_mode="Campaign Progression",
            party=(PartyReadinessMember(0, "Treecko", 5, 20, 1.0, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=None,
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.PREPARE,
            readiness_reason=ReadinessReason.TRAINER_BATTLE_UNREADY,
            trainer_checkpoint=ROUTE102_FIRST_POOCHYENA,
            trainer_engagement_required=True,
        )
        with patch(
            "modules.nuzlocke.resource_runtime.execute_campaign_preparation",
            return_value=iter(()),
        ) as prepare, patch.object(
            CampaignController,
            "_active_preparation_level_cap",
            return_value=_PREPARATION_LEVEL_CAP_UNAVAILABLE,
        ):
            controller = CampaignController(
                lambda: self.current,
                selector=lambda _: selection,
                adapter=lambda _: execution,
                readiness_provider=lambda *_: readiness,
            )

            deferred = controller.step()

        self.assertEqual(deferred.status, CampaignControllerStatus.UNKNOWN)
        self.assertEqual(deferred.execution_phase, "CAMPAIGN")
        prepare.assert_not_called()

    def test_restock_interrupt_can_replace_mounted_progression(self):
        self.assertFalse(_retention_order_allows("complete_petalburg_wally", "restock_pokeballs"))
        self.assertFalse(_retention_order_allows("obtain_encounter:0:17", "restock_pokeballs"))

    def test_active_story_boundary_retains_ownership_over_restock(self):
        story = next(
            item for item in initial_emerald_campaign() if item.objective_id == "complete_petalburg_wally"
        )
        with patch(
            "modules.nuzlocke.campaign_controller._actionable_scripted_interaction_is_active",
            return_value=True,
        ):
            self.assertTrue(
                _story_progression_owns_restock_boundary(
                    story,
                    refresh_reason="capability_boundary",
                )
            )

    def test_encounter_target_change_replaces_the_old_spatial_owner(self):
        self.assertFalse(_retention_order_allows("obtain_encounter:0:29", "obtain_encounter:0:30"))

    def test_shared_restock_objectives_keep_one_live_shop_transaction(self):
        self.assertTrue(_retention_order_allows("restock_recovery_items", "restock_pokeballs"))
        self.assertTrue(_retention_order_allows("restock_pokeballs", "restock_recovery_items"))

        recovery_complete = [False]
        recovery = CampaignObjective(
            "restock_recovery_items",
            "Restock recovery items",
            (),
            CampaignPredicate(
                "recovery_complete",
                "recovery restock is complete",
                lambda _: Fact.known(recovery_complete[0]),
            ),
        )
        pokeballs = self.fixture.objective(
            "restock_pokeballs", self.fixture.predicate("balls_complete", False)
        )
        selections = iter(
            (
                ObjectiveSelection(recovery, ObjectiveStatus.READY, "recovery restock"),
                ObjectiveSelection(pokeballs, ObjectiveStatus.READY, "Poké Ball restock became visible"),
            )
        )
        advanced = []

        def capability():
            advanced.append("recovery")
            yield
            advanced.append("recovery-finished")
            yield

        execution = CampaignExecutionResult(
            recovery,
            CampaignExecutionStatus.READY,
            "shared restock capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        recovery_complete[0] = True
        controller.request_refresh("recovery_inventory_changed")
        result = controller.step()

        self.assertEqual(result.objective_id, "restock_recovery_items")
        self.assertEqual(advanced, ["recovery", "recovery-finished"])

    def test_mounted_restock_is_not_replaced_by_encounter_before_shop_exit(self):
        recovery_complete = [True]
        recovery = CampaignObjective(
            "restock_recovery_items",
            "Restock recovery items",
            (),
            CampaignPredicate(
                "recovery_complete",
                "recovery restock is complete",
                lambda _: Fact.known(recovery_complete[0]),
            ),
        )
        encounter = self.fixture.objective(
            "obtain_encounter:0:17",
            self.fixture.predicate("encounter_complete", False),
        )
        selections = iter(
            (
                ObjectiveSelection(recovery, ObjectiveStatus.READY, "recovery restock"),
                ObjectiveSelection(encounter, ObjectiveStatus.READY, "strategic encounter"),
            )
        )
        advanced = []

        def recovery_capability():
            advanced.append("recovery")
            while True:
                yield

        def encounter_capability():
            advanced.append("encounter")
            while True:
                yield

        def adapter(selection):
            capability = recovery_capability if selection.objective.objective_id == recovery.objective_id else encounter_capability
            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "test capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        with patch(
            "modules.nuzlocke.campaign_controller._retention_order_allows",
            return_value=False,
        ):
            controller.step()
            controller.request_refresh("inventory_updated_before_shop_exit")
            result = controller.step()

        self.assertEqual(result.objective_id, recovery.objective_id)
        self.assertEqual(controller._tactical_loop_objective_id, recovery.objective_id)
        self.assertEqual(advanced, ["recovery"])

    def test_mart_restock_finishes_before_opportunistic_recovery(self):
        objective = restock_recovery_items_objective()
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "recovery restock")
        recovery_starts = []
        mounted = []

        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
            )
        )

        def restock_capability():
            mounted.append(objective.objective_id)
            while True:
                yield

        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=None,
            navigation_goal=None,
            current_map=(2, 4),
            current_coordinates=(3, 3),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(PartyReadinessMember(0, "Treecko", 7, 23, 0.6, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=SimpleNamespace(
                center_available=True,
                safe_to_reach_center=True,
                center_location=((0, 10), (6, 16)),
                route=(),
                navigation_goal=Goal(),
                healing_source=None,
                calculation_pending=False,
            ),
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.OPPORTUNISTIC_RECOVERY,
        )
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "restock capability",
            capability=restock_capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: readiness,
            recovery_factory=lambda stop: recovery_starts.append(stop) or iter(()),
        )

        result = controller.step()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.execution_phase, "CAMPAIGN")
        self.assertEqual(mounted, [objective.objective_id])
        self.assertEqual(recovery_starts, [])
        self.assertIsNotNone(controller._tactical_loop)

    def test_ready_intro_battle_mounts_existing_tactical_goal(self):
        controller = self.controller()
        result = controller.refresh()
        self.assertEqual(result.objective_id, "complete_intro_rival")
        self.assertEqual(result.status, CampaignControllerStatus.READY)

    def test_confirmed_pokedex_does_not_regress_on_transient_campaign_read(self):
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_available=True,
                pokeballs_ready=True,
            )
        )
        confirmed = self.current
        controller = self.controller()
        controller._stabilize_campaign_facts(confirmed)
        controller._stabilize_campaign_facts(confirmed)

        transient = replace(
            confirmed,
            campaign_facts=replace(confirmed.campaign_facts, pokedex_received=Fact.known(False)),
        )
        stabilized = controller._stabilize_campaign_facts(transient)

        self.assertTrue(stabilized.campaign_facts.pokedex_received.value)

    def test_birch_rescue_does_not_advance_on_one_transient_flag_read(self):
        before = replace(
            self.current,
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=False,
            ),
        )
        controller = CampaignController(lambda: self.current, tactical_loop_factory=self.factory)
        controller._stabilize_campaign_facts(before)

        transient = replace(
            before,
            campaign_facts=replace(before.campaign_facts, birch_rescued=Fact.known(True)),
        )
        first = controller._stabilize_campaign_facts(transient)
        self.assertEqual(first.campaign_facts.birch_rescued.status, FactStatus.UNKNOWN)
        self.assertEqual(select_campaign_objective(first).objective.objective_id, "rescue_birch")

        cleared = controller._stabilize_campaign_facts(before)
        self.assertEqual(cleared.campaign_facts.birch_rescued.value, False)

        second_transient = controller._stabilize_campaign_facts(transient)
        self.assertEqual(second_transient.campaign_facts.birch_rescued.status, FactStatus.UNKNOWN)

        confirmed = controller._stabilize_campaign_facts(transient)
        self.assertTrue(confirmed.campaign_facts.birch_rescued.value)
        self.assertEqual(select_campaign_objective(confirmed).objective.objective_id, "obtain_starter")

    def test_late_campaign_fact_is_suppressed_until_coherent_and_repeated(self):
        transient = replace(
            self.current,
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                first_badge_obtained=True,
            ),
        )
        controller = self.controller()

        first = controller._stabilize_campaign_facts(transient)
        second = controller._stabilize_campaign_facts(transient)

        self.assertEqual(first.campaign_facts.first_badge_obtained.status, FactStatus.UNKNOWN)
        self.assertEqual(second.campaign_facts.first_badge_obtained.status, FactStatus.UNKNOWN)
        self.assertNotIn("first_badge_obtained", controller._confirmed_campaign_facts)

        coherent = replace(
            transient,
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                petalburg_wally_scene_complete=True,
                petalburg_woods_scene_complete=True,
                visited_rustboro=True,
                first_badge_obtained=True,
            ),
        )
        controller._stabilize_campaign_facts(coherent)
        confirmed = controller._stabilize_campaign_facts(coherent)

        self.assertTrue(confirmed.campaign_facts.first_badge_obtained.value)
        self.assertIn("first_badge_obtained", controller._confirmed_campaign_facts)

    def test_campaign_high_water_resets_when_observed_session_changes(self):
        confirmed = replace(
            self.current,
            session_id="session-after-progress",
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
            ),
        )
        controller = self.controller()
        controller._stabilize_campaign_facts(confirmed)

        earlier_save = replace(
            confirmed,
            session_id="session-before-progress",
            campaign_facts=replace(confirmed.campaign_facts, pokedex_received=Fact.known(False)),
        )
        stabilized = controller._stabilize_campaign_facts(earlier_save)

        self.assertFalse(stabilized.campaign_facts.pokedex_received.value)

    def test_task_diagnostics_are_cached_with_map_level_planning_facts(self):
        objective = self.fixture.objective("diagnostic", self.fixture.predicate("diagnostic_done", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "diagnostic")
        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "diagnostic", tactical_goal=Goal()
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )
        with patch.object(context, "debug", True), patch.object(context, "debug_trace", True), patch(
            "modules.nuzlocke.campaign_controller.campaign_task_diagnostics", return_value=()
        ) as diagnostics:
            controller.refresh()
            controller.refresh()
            controller.refresh()

        self.assertEqual(diagnostics.call_count, 2)
        diagnostics.assert_called_with(self.current)

    def test_campaign_frontier_is_published_without_debug_logging(self):
        controller = self.controller()
        with patch.object(context, "debug", False), patch(
            "modules.nuzlocke.campaign_controller.print_campaign_frontier"
        ) as frontier:
            controller.refresh()
            controller.refresh()
            controller.refresh()

        self.assertEqual(frontier.call_count, 2)

    def test_campaign_frontier_is_suppressed_across_controller_remounts_in_one_runtime(self):
        runtime = SimpleNamespace()
        first = self.controller()
        second = self.controller()
        with patch.object(context, "nuzlocke_runtime", runtime), patch.object(
            context, "campaign_frontier_diagnostics_key", None
        ), patch.object(context, "debug", False), patch(
            "modules.nuzlocke.campaign_controller.print_campaign_frontier"
        ) as frontier:
            first.refresh()
            second.refresh()

        self.assertEqual(frontier.call_count, 1)

    def test_restock_frontier_includes_the_baseline_item_plan(self):
        state = self.fixture.state(area="ROUTE101", balls=1)
        selection = ObjectiveSelection(
            restock_recovery_items_objective(),
            ObjectiveStatus.READY,
            "selected required recovery-item restock task",
        )
        controller = self.controller()
        with patch(
            "modules.nuzlocke.campaign_controller.campaign_frontier_summary",
            return_value=((), ()),
        ), patch("modules.nuzlocke.campaign_controller.print_campaign_frontier") as frontier:
            controller._emit_campaign_task_diagnostics(state, selection)

        self.assertEqual(
            frontier.call_args.kwargs["selected_detail"],
            "4x Antidote, 5x Potion, 9x Poké Ball (policy targets; wallet/cash-floor limits may reduce purchase)",
        )

    def test_completed_intro_battle_reselects_next_declarative_objective(self):
        controller = self.controller()
        controller.refresh()
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
            )
        )
        controller.refresh()
        result = controller.refresh()
        self.assertEqual(result.selection.objective.objective_id, "receive_pokedex")
        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.objective_id, "receive_pokedex")

    def test_blocked_and_unknown_do_not_execute(self):
        for status in (ObjectiveStatus.BLOCKED, ObjectiveStatus.UNKNOWN):
            selection = ObjectiveSelection(None, status, status.value)
            controller = CampaignController(
                lambda: self.current, tactical_loop_factory=self.factory, selector=lambda _: selection
            )
            result = controller.step()
            self.assertEqual(result.status.value, status.value)
            self.assertEqual(self.executed, 0)
            self.assertEqual(controller.step().status.value, status.value)
            self.assertEqual(self.executed, 0)

    def test_transient_unknown_selection_retries_after_post_battle_boundary(self):
        objective = self.fixture.objective("receive_pokedex", self.fixture.predicate("pokedex_done", False))
        unknown = ObjectiveSelection(
            objective,
            ObjectiveStatus.UNKNOWN,
            "prerequisite campaign_fact:intro_rival_battle_complete is unknown",
        )
        ready = ObjectiveSelection(objective, ObjectiveStatus.READY, "all prerequisites are satisfied")
        selections = iter((unknown, ready))

        def capability():
            while True:
                yield

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "pokedex capability",
            capability=capability,
        )

        def adapt(selection):
            if selection.status is ObjectiveStatus.UNKNOWN:
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.UNKNOWN,
                    selection.reason,
                )
            return execution

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapt,
        )

        first = controller.step()
        second = controller.step()

        self.assertEqual(first.status, CampaignControllerStatus.UNKNOWN)
        self.assertEqual(first.execution_phase, "CAMPAIGN")
        self.assertEqual(first.objective_id, "receive_pokedex")
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertEqual(second.objective_id, "receive_pokedex")
        self.assertEqual(second.execution_phase, "CAMPAIGN")
        self.assertIsNotNone(controller._tactical_loop)

    def test_blocked_selection_remains_autonomous_and_retries(self):
        objective = self.fixture.objective("blocked", self.fixture.predicate("blocked_done", False))
        blocked = ObjectiveSelection(objective, ObjectiveStatus.BLOCKED, "prerequisite is false")
        ready = ObjectiveSelection(objective, ObjectiveStatus.READY, "prerequisite became available")
        selections = iter((blocked, ready))

        def capability():
            while True:
                yield

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "blocked capability",
            capability=capability,
        )

        def adapt(selection):
            if selection.status is ObjectiveStatus.BLOCKED:
                return CampaignExecutionResult(
                    objective,
                    CampaignExecutionStatus.BLOCKED,
                    selection.reason,
                )
            return execution

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapt,
        )

        first = controller.step()
        second = controller.step()

        self.assertEqual(first.status, CampaignControllerStatus.BLOCKED)
        self.assertEqual(first.execution_phase, "CAMPAIGN")
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertIsNotNone(controller._tactical_loop)

    def test_refresh_exception_is_retried_without_unwinding_campaign_mode(self):
        calls = []
        objective = self.fixture.objective("retry", self.fixture.predicate("retry_done", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "retry")

        def state_provider():
            calls.append("state")
            if len(calls) == 1:
                raise RuntimeError("transient post-battle observation")
            return self.current

        def capability():
            calls.append("capability")
            yield

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "retry capability",
            capability=capability,
        )
        controller = CampaignController(
            state_provider,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )

        first = controller.step()
        second = controller.step()

        self.assertEqual(first.status, CampaignControllerStatus.UNKNOWN)
        self.assertIn("will re-evaluate", first.reason)
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertEqual(calls, ["state", "state", "capability"])

    def test_failed_selection_is_terminal(self):
        selection = ObjectiveSelection(None, ObjectiveStatus.FAILED, "failed")
        controller = CampaignController(lambda: self.current, selector=lambda _: selection)
        self.assertEqual(controller.step().status, CampaignControllerStatus.FAILED)
        self.assertEqual(controller.step().status, CampaignControllerStatus.FAILED)

    def test_unsupported_execution_is_safe(self):
        selection = ObjectiveSelection(None, ObjectiveStatus.READY, "bad")
        unsupported = CampaignExecutionResult(None, CampaignExecutionStatus.UNSUPPORTED, "unsupported")
        controller = CampaignController(
            lambda: self.current, selector=lambda _: selection, adapter=lambda _: unsupported
        )
        result = controller.step()
        self.assertEqual(result.status, CampaignControllerStatus.FAILED)
        self.assertEqual(self.executed, 0)

        malformed = CampaignExecutionResult(None, CampaignExecutionStatus.READY, "missing goal")
        controller = CampaignController(lambda: self.current, selector=lambda _: selection, adapter=lambda _: malformed)
        self.assertEqual(controller.step().status, CampaignControllerStatus.FAILED)

    def test_active_objective_is_not_reissued(self):
        controller = self.controller()
        controller.step()
        controller.step()
        # Introductory-rival execution is now mounted through the resource
        # capability boundary; the legacy tactical-loop factory is not used.
        self.assertEqual(len(self.created), 0)
        self.assertEqual(self.executed, 0)

    def test_mounted_tactical_loop_skips_campaign_refresh_until_requested(self):
        objective = self.fixture.objective("stable", self.fixture.predicate("stable_done", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "stable")
        selector_calls = []

        def select(_state):
            selector_calls.append(1)
            return selection

        def capability():
            while True:
                yield

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "stable capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=select,
            adapter=lambda _: execution,
        )

        controller.step()
        controller.step()
        controller.step()
        self.assertEqual(len(selector_calls), 1)

        controller.request_refresh("map_changed")
        controller.step()
        self.assertEqual(len(selector_calls), 2)

    def test_navigation_transition_refresh_preserves_mounted_loop(self):
        """A map-loading frame cannot tear down the loop settling its warp."""

        class TransitionOwner:
            navigation_transition_active = True

        selector_calls = []
        controller = CampaignController(
            lambda: (_ for _ in ()).throw(AssertionError("campaign state must wait for the warp")),
            selector=lambda _: selector_calls.append(True),
        )
        mounted = TransitionOwner()
        controller._tactical_loop = mounted
        controller._tactical_loop_objective_id = "reach_route_102"
        controller.current_objective_id = "reach_route_102"
        controller.status = CampaignControllerStatus.READY

        result = controller.refresh()

        self.assertEqual(result, controller.state)
        self.assertIs(controller._tactical_loop, mounted)
        self.assertTrue(controller._refresh_required)
        self.assertEqual(controller._refresh_required_reason, "navigation_transition_settling")
        self.assertEqual(selector_calls, [])
        self.assertEqual(controller.transition_reason, "navigation transition settling; preserving tactical loop")

    def test_nested_agent_navigation_transition_is_visible_to_campaign_refresh(self):
        """Capabilities expose their inner AgentControlLoop through context."""

        active_loop = SimpleNamespace(navigation_transition_active=True)
        selector_calls = []
        controller = CampaignController(
            lambda: (_ for _ in ()).throw(AssertionError("campaign state must wait for the warp")),
            selector=lambda _: selector_calls.append(True),
        )
        controller._tactical_loop = iter(())
        controller._tactical_loop_objective_id = "reach_route_102"
        controller.current_objective_id = "reach_route_102"
        controller.status = CampaignControllerStatus.READY

        with patch.object(context, "active_agent_control_loop", active_loop):
            controller.refresh()

        self.assertEqual(selector_calls, [])
        self.assertTrue(controller._refresh_required)
        self.assertEqual(controller._refresh_required_reason, "navigation_transition_settling")

    def test_readiness_recheck_requests_campaign_refresh(self):
        objective = self.fixture.objective("stable", self.fixture.predicate("stable_done", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "stable")
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "stable capability",
                capability=lambda: iter((None,)),
            ),
        )

        controller.step()
        self.assertFalse(controller._refresh_required)
        controller.request_readiness_recheck("battle_ended")
        self.assertTrue(controller._refresh_required)

    def test_post_pokedex_targetless_capability_rechecks_readiness_once(self):
        objective = next(item for item in initial_emerald_campaign() if item.objective_id == "complete_petalburg_wally")
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_ready=True,
            )
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "route traversal")
        readiness_calls = []
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=objective.destination,
            navigation_goal=None,
            current_map=(0, 18),
            current_coordinates=(10, 3),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(PartyReadinessMember(0, "Treecko", 1, 10, 1.0, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=False,
            recovery=None,
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.UNAVAILABLE,
            navigation_availability=Availability.NOT_APPLICABLE,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
        )

        def readiness_provider(*_):
            readiness_calls.append(1)
            return readiness

        def capability():
            while True:
                yield

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "route capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=readiness_provider,
        )

        controller.step()
        controller.request_readiness_recheck("battle_ended")
        controller.step()
        controller.step()

        self.assertEqual(len(readiness_calls), 2)
        self.assertFalse(controller._readiness_recheck_pending)

    def test_opening_targetless_capability_discards_inapplicable_recheck(self):
        objective = self.fixture.objective("opening", self.fixture.predicate("opening_done", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "opening")

        def capability():
            while True:
                yield

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "opening capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )

        controller.step()
        controller.request_readiness_recheck("battle_ended")
        controller.step()
        controller.step()

        self.assertFalse(controller._readiness_recheck_pending)
        self.assertFalse(controller._refresh_required)

    def test_deferred_no_loop_state_does_not_refresh_every_frame(self):
        objective = self.fixture.objective("deferred", self.fixture.predicate("deferred_done", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "deferred")
        selector_calls = []

        def select(_state):
            selector_calls.append(1)
            return selection

        controller = CampaignController(
            lambda: self.current,
            selector=select,
            adapter=lambda _: CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.UNKNOWN,
                "emulator boundary",
            ),
        )

        controller.step()
        controller.step()
        self.assertEqual(len(selector_calls), 1)

        controller.request_refresh("battle_returned")
        controller.step()
        self.assertEqual(len(selector_calls), 2)

    def test_capability_remains_active_until_authoritative_fact_changes(self):
        calls = []

        def capability():
            calls.append("mounted")
            yield
            yield

        objective = self.fixture.objective("multi_frame", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )
        controller.step()
        controller.step()
        self.assertEqual(calls, ["mounted"])

    def test_completed_safety_fact_does_not_replace_running_capability(self):
        healed = [False]
        safety = CampaignObjective(
            "HEAL_PARTY",
            "Restore the party",
            (),
            CampaignPredicate("healed", "healed", lambda _: Fact.known(healed[0])),
            task_kind="safety",
        )
        parent = self.fixture.objective("parent", self.fixture.predicate("parent_done", False))
        selections = iter(
            (
                ObjectiveSelection(safety, ObjectiveStatus.READY, "safety ready"),
                ObjectiveSelection(parent, ObjectiveStatus.READY, "parent ready"),
            )
        )
        advanced = []

        def adapter(selection):
            def capability():
                advanced.append(selection.objective.objective_id)
                yield
                if selection.objective is safety:
                    healed[0] = True
                    advanced.append("safety-finished")
                    return
                advanced.append("parent-ran")
                yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        controller.step()
        # The resource is restored before the interaction capability returns.
        healed[0] = True
        # The safety generator has not yet returned, so it must receive the
        # next controller step even though its completion fact is now true.
        controller.step()
        self.assertEqual(advanced, ["HEAL_PARTY", "safety-finished"])

        controller.step()
        self.assertEqual(advanced, ["HEAL_PARTY", "safety-finished", "parent"])

    def test_active_capability_owns_transient_planner_regression(self):
        first = self.fixture.objective("active", self.fixture.predicate("active_done", False))
        second = self.fixture.objective("transient", self.fixture.predicate("transient_done", False))
        selections = iter(
            (
                ObjectiveSelection(first, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(second, ObjectiveStatus.READY, "transient planner result"),
            )
        )
        advanced = []

        def capability():
            while True:
                advanced.append("active")
                yield

        execution = CampaignExecutionResult(
            first,
            CampaignExecutionStatus.READY,
            "active capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "active")
        self.assertEqual(advanced, ["active", "active"])

    def test_completed_capability_owns_downstream_selection_until_it_returns(self):
        completed = [False]
        first = CampaignObjective(
            "first_capability",
            "First capability",
            (),
            CampaignPredicate("first_done", "first is complete", lambda _: Fact.known(completed[0])),
        )
        second = self.fixture.objective("second_capability", self.fixture.predicate("second_done", False))
        selections = iter(
            (
                ObjectiveSelection(first, ObjectiveStatus.READY, "first"),
                ObjectiveSelection(second, ObjectiveStatus.READY, "second"),
            )
        )
        advanced = []

        def capability():
            advanced.append("first")
            yield
            while True:
                advanced.append("first")
                yield

        execution = CampaignExecutionResult(
            first,
            CampaignExecutionStatus.READY,
            "first capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        completed[0] = True
        controller.request_refresh("completion_boundary")
        result = controller.step()

        self.assertEqual(result.objective_id, "first_capability")
        self.assertEqual(result.selection.objective.objective_id, "first_capability")
        self.assertEqual(advanced, ["first", "first"])

    def test_canonical_later_objective_owns_transient_earlier_selection(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        active = objectives["prepare_roxanne"]
        earlier = objectives["set_text_speed"]
        self.state(campaign_facts=self.fixture.facts(visited_rustboro=True))
        selections = iter(
            (
                ObjectiveSelection(active, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "transient planner result"),
            )
        )
        advanced = []

        def capability():
            while True:
                advanced.append("prepare_roxanne")
                yield

        execution = CampaignExecutionResult(
            active,
            CampaignExecutionStatus.READY,
            "active capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "prepare_roxanne")
        self.assertEqual(advanced, ["prepare_roxanne", "prepare_roxanne"])

    def test_later_active_objective_survives_unavailable_battle_boundary(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        active = objectives["prepare_roxanne"]
        earlier = objectives["set_text_speed"]
        # The live battle-start snapshot has no field map or coordinates. Its
        # save-backed facts can consequently make the active prerequisite look
        # false for one frame even though no coherent save regression exists.
        self.state(
            raw_map=Fact.unknown(),
            coordinates=Fact.unknown(),
            campaign_facts=self.fixture.facts(),
        )
        selections = iter(
            (
                ObjectiveSelection(active, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "battle boundary"),
            )
        )
        advanced = []

        def capability():
            while True:
                advanced.append("prepare_roxanne")
                yield

        execution = CampaignExecutionResult(
            active,
            CampaignExecutionStatus.READY,
            "active capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "prepare_roxanne")
        self.assertEqual(advanced, ["prepare_roxanne", "prepare_roxanne"])

    def test_deferred_readiness_owns_objective_across_planner_handoff(self):
        pending = self.fixture.objective("pending_readiness", self.fixture.predicate("done", False))
        transient = self.fixture.objective("transient_handoff", self.fixture.predicate("done", False))
        selections = iter(
            (
                ObjectiveSelection(pending, ObjectiveStatus.READY, "pending readiness"),
                ObjectiveSelection(transient, ObjectiveStatus.READY, "transition observation"),
            )
        )
        adapted = []
        readiness_calls = []
        recovery_starts = []

        def readiness_provider(*_):
            readiness_calls.append(1)
            recovering = len(readiness_calls) == 2
            return ProgressionReadinessDiagnostic(
                objective_id="pending_readiness",
                objective_status="ready",
                destination=None,
                navigation_goal=Goal(),
                current_map=(1, 2),
                current_coordinates=(3, 4),
                game_state="OVERWORLD" if recovering else "BATTLE",
                campaign_mode="test",
                party=(PartyReadinessMember(0, "Treecko", 1, 10, 0.1, False, True, "none"),),
                has_usable_pokemon=True,
                healing_available=True,
                recovery=None,
                trainers=(),
                party_availability=Availability.KNOWN,
                trainer_availability=Availability.KNOWN,
                recovery_availability=Availability.KNOWN,
                navigation_availability=Availability.KNOWN,
                overworld_availability=Availability.KNOWN,
                resource_availability=Availability.KNOWN,
                route_analysis=(
                    RouteAnalysis(
                        Goal(),
                        "normal-route",
                        40,
                        (
                            IntermediateRouteAnalysis(
                                ReachLocation(("OLDale", (5, 6))),
                                48,
                                8,
                                True,
                                first_route=SimpleNamespace(
                                    metrics=SimpleNamespace(total_route_cost=8),
                                ),
                            ),
                        ),
                    )
                    if recovering
                    else None
                ),
            )

        def adapter(selection):
            adapted.append(selection.objective.objective_id)
            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "ready",
                tactical_goal=Goal(),
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
            readiness_provider=readiness_provider,
            recovery_factory=lambda _: iter((recovery_starts.append("pending_readiness"),)),
        )

        first = controller.step()
        controller.request_readiness_recheck("test_observation_boundary")
        second = controller.step()

        self.assertEqual(first.status, CampaignControllerStatus.READY)
        self.assertIn("readiness deferred", first.reason)
        self.assertEqual(second.execution_phase, "RECOVERY")
        self.assertEqual(adapted, ["pending_readiness", "pending_readiness"])
        self.assertEqual(recovery_starts, ["pending_readiness"])
        self.assertEqual(controller.current_objective_id, "pending_readiness")

    def test_targetless_capability_mounts_across_unknown_post_battle_readiness(self):
        objective = next(item for item in initial_emerald_campaign() if item.objective_id == "receive_pokedex")
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "pokedex frontier")
        mounted = []

        def capability():
            mounted.append(objective.objective_id)
            yield

        unknown = ProgressionReadinessDiagnostic(
            objective_id="receive_pokedex",
            objective_status="ready",
            destination=None,
            navigation_goal=None,
            current_map=(0, 18),
            current_coordinates=None,
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(),
            has_usable_pokemon=None,
            healing_available=None,
            recovery=None,
            trainers=(),
            party_availability=Availability.UNKNOWN,
            trainer_availability=Availability.UNKNOWN,
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.NOT_APPLICABLE,
            overworld_availability=Availability.UNKNOWN,
            resource_availability=Availability.UNKNOWN,
            readiness_decision=None,
            readiness_reason=None,
        )
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "pokedex capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: unknown,
        )

        result = controller.step()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.objective_id, "receive_pokedex")
        self.assertEqual(mounted, ["receive_pokedex"])
        self.assertIsNotNone(controller._tactical_loop)

    def test_targetless_recovery_capability_waits_for_unknown_route_readiness(self):
        objective = replace(restock_recovery_items_objective(), destination=(0, 17))
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
            )
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "recovery readiness")
        mounted = []

        def capability():
            mounted.append(objective.objective_id)
            yield

        unknown = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=None,
            navigation_goal=None,
            current_map=(0, 17),
            current_coordinates=(39, 6),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(PartyReadinessMember(0, "Torchic", 5, 10, 0.5, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=SimpleNamespace(
                center_available=True,
                safe_to_reach_center=True,
                route=None,
                calculation_pending=False,
            ),
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.UNKNOWN,
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.NOT_APPLICABLE,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.UNKNOWN,
            readiness_reason=ReadinessReason.OPPORTUNISTIC_ROUTE_UNAVAILABLE,
        )
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "targetless recovery capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: unknown,
        )

        result = controller.step()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(mounted, [])
        self.assertIsNone(controller._tactical_loop)
        self.assertIn("unresolved", result.reason)

    def test_dialogue_boundary_mounts_tactical_capability_when_readiness_is_unknown(self):
        goal = NavigationGoal(SemanticTarget.map((0, 17)))
        objective = CampaignObjective(
            "obtain_encounter:0:17",
            "Obtain the first encounter at (0, 17)",
            (),
            self.fixture.predicate("encounter_done", False),
            destination=(0, 17),
            tactical_target=goal,
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "encounter frontier")
        mounted = []

        def capability():
            mounted.append(objective.objective_id)
            while True:
                yield

        unknown = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=objective.destination,
            navigation_goal=goal,
            current_map=(0, 18),
            current_coordinates=(10, 3),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(),
            has_usable_pokemon=None,
            healing_available=None,
            recovery=None,
            trainers=(),
            party_availability=Availability.UNKNOWN,
            trainer_availability=Availability.UNKNOWN,
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.UNKNOWN,
            resource_availability=Availability.UNKNOWN,
        )
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "encounter capability",
            tactical_goal=goal,
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: unknown,
        )

        with patch(
            "modules.nuzlocke.campaign_controller._actionable_scripted_interaction_is_active",
            return_value=True,
        ):
            result = controller.step()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(mounted, [objective.objective_id])
        self.assertIsNotNone(controller._tactical_loop)

    def test_tactical_capability_remounts_after_unknown_readiness_without_loop(self):
        objective = CampaignObjective(
            "calvin_route",
            "Navigate to Calvin",
            (),
            self.fixture.predicate("calvin_done", False),
            destination=(0, 17),
            tactical_target=NavigationGoal(SemanticTarget.map((0, 17))),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "Calvin route")
        mounted = []

        def capability():
            mounted.append(objective.objective_id)
            while True:
                yield

        def readiness(decision, reason, overworld):
            return ProgressionReadinessDiagnostic(
                objective_id=objective.objective_id,
                objective_status="ready",
                destination=objective.destination,
                navigation_goal=objective.tactical_target,
                current_map=(0, 10),
                current_coordinates=(14, 7),
                game_state="OVERWORLD",
                campaign_mode="test",
                party=(),
                has_usable_pokemon=True,
                healing_available=True,
                recovery=None,
                trainers=(),
                party_availability=Availability.KNOWN,
                trainer_availability=Availability.KNOWN,
                recovery_availability=Availability.KNOWN,
                navigation_availability=Availability.KNOWN,
                overworld_availability=overworld,
                resource_availability=Availability.KNOWN,
                readiness_decision=decision,
                readiness_reason=reason,
            )

        observations = iter(
            (
                readiness(
                    ReadinessDecision.UNKNOWN,
                    ReadinessReason.OVERWORLD_UNAVAILABLE,
                    Availability.UNKNOWN,
                ),
                readiness(
                    ReadinessDecision.UNKNOWN,
                    ReadinessReason.OVERWORLD_UNAVAILABLE,
                    Availability.UNKNOWN,
                ),
                readiness(
                    ReadinessDecision.CONTINUE,
                    ReadinessReason.PARTY_HEALTHY,
                    Availability.KNOWN,
                ),
            )
        )
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "Calvin route capability",
            tactical_goal=objective.tactical_target,
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: next(observations),
        )

        first = controller.step()
        second = controller.step()

        self.assertEqual(first.status, CampaignControllerStatus.READY)
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertIsNone(controller._tactical_loop)

        third = controller.step()

        self.assertEqual(third.status, CampaignControllerStatus.READY)
        self.assertEqual(mounted, [objective.objective_id])
        self.assertIsNotNone(controller._tactical_loop)

    def test_promoted_trainer_route_remounts_after_unknown_readiness_without_capability(self):
        """A Center script must not strand a promoted encounter route."""

        objective = CampaignObjective(
            "obtain_encounter:0:17",
            "Obtain the first encounter at Route 102",
            (),
            self.fixture.predicate("route102_done", False),
            destination=(0, 17),
            tactical_target=NavigationGoal(SemanticTarget.map((0, 17))),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "Route 102 encounter")
        trainer_goal = NavigationGoal(EngageTrainer("trainer:(0, 17):2"))
        observations = iter(
            (
                ProgressionReadinessDiagnostic(
                    objective_id=objective.objective_id,
                    objective_status="ready",
                    destination=objective.destination,
                    navigation_goal=trainer_goal,
                    current_map=(2, 2),
                    current_coordinates=(7, 4),
                    game_state="OVERWORLD",
                    campaign_mode="test",
                    party=(),
                    has_usable_pokemon=True,
                    healing_available=True,
                    recovery=None,
                    trainers=(),
                    party_availability=Availability.KNOWN,
                    trainer_availability=Availability.KNOWN,
                    recovery_availability=Availability.KNOWN,
                    navigation_availability=Availability.KNOWN,
                    overworld_availability=Availability.UNKNOWN,
                    resource_availability=Availability.KNOWN,
                    readiness_decision=ReadinessDecision.UNKNOWN,
                    readiness_reason=ReadinessReason.OVERWORLD_UNAVAILABLE,
                ),
                ProgressionReadinessDiagnostic(
                    objective_id=objective.objective_id,
                    objective_status="ready",
                    destination=objective.destination,
                    navigation_goal=trainer_goal,
                    current_map=(2, 2),
                    current_coordinates=(7, 4),
                    game_state="OVERWORLD",
                    campaign_mode="test",
                    party=(),
                    has_usable_pokemon=True,
                    healing_available=True,
                    recovery=None,
                    trainers=(),
                    party_availability=Availability.KNOWN,
                    trainer_availability=Availability.KNOWN,
                    recovery_availability=Availability.KNOWN,
                    navigation_availability=Availability.KNOWN,
                    overworld_availability=Availability.KNOWN,
                    resource_availability=Availability.KNOWN,
                    readiness_decision=ReadinessDecision.CONTINUE,
                    readiness_reason=ReadinessReason.PARTY_HEALTHY,
                ),
            )
        )
        mounted = []

        def tactical_loop_factory(goal):
            mounted.append(goal)

            def loop():
                while True:
                    yield

            return loop()

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "encounter capability",
            tactical_goal=objective.tactical_target,
            capability=lambda: iter(()),
        )
        controller = CampaignController(
            lambda: self.current,
            tactical_loop_factory=tactical_loop_factory,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: next(observations),
        )

        first = controller.step()
        self.assertEqual(first.status, CampaignControllerStatus.READY)
        self.assertIsNone(controller._tactical_loop)
        self.assertTrue(controller._refresh_required)
        self.assertIsNone(controller.last_execution.capability)

        second = controller.step()
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertEqual(mounted, [trainer_goal])
        self.assertIsNotNone(controller._tactical_loop)

    def test_unresolved_trainer_roster_blocks_capability_remount_until_recheck(self):
        objective = CampaignObjective(
            "obtain_encounter:0:17",
            "Obtain the first encounter at Route 102",
            (),
            self.fixture.predicate("route102_done", False),
            destination=(0, 17),
            tactical_target=NavigationGoal(SemanticTarget.map((0, 17))),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "Route 102 encounter")
        trainer_goal = NavigationGoal(EngageTrainer("trainer:(0, 17):2"))
        mounted = []

        def capability():
            mounted.append("capability")
            while True:
                yield

        def tactical_loop_factory(goal):
            mounted.append(goal)

            def loop():
                while True:
                    yield

            return loop()

        def diagnostic(reason, decision, *, pending=False):
            return ProgressionReadinessDiagnostic(
                objective_id=objective.objective_id,
                objective_status="ready",
                destination=objective.destination,
                navigation_goal=trainer_goal,
                current_map=(0, 10),
                current_coordinates=(14, 7),
                game_state="OVERWORLD",
                campaign_mode="test",
                party=(),
                has_usable_pokemon=True,
                healing_available=True,
                recovery=None,
                trainers=(),
                party_availability=Availability.KNOWN,
                trainer_availability=Availability.KNOWN,
                recovery_availability=Availability.KNOWN,
                navigation_availability=Availability.KNOWN,
                overworld_availability=Availability.KNOWN,
                resource_availability=Availability.KNOWN,
                readiness_decision=decision,
                readiness_reason=reason,
                route_analysis_pending=pending,
            )

        observations = iter(
            (
                diagnostic(ReadinessReason.TRAINER_BATTLE_UNKNOWN, ReadinessDecision.UNKNOWN, pending=True),
                diagnostic(ReadinessReason.PARTY_HEALTHY, ReadinessDecision.CONTINUE),
            )
        )
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "encounter capability",
            tactical_goal=objective.tactical_target,
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            tactical_loop_factory=tactical_loop_factory,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: next(observations),
        )

        first = controller.step()
        self.assertEqual(first.status, CampaignControllerStatus.READY)
        self.assertIsNone(controller._tactical_loop)
        self.assertEqual(mounted, [])
        self.assertIn("recovery route calculation pending", first.reason)

        controller.request_readiness_recheck("trainer_facts_available")
        second = controller.step()
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertEqual(mounted, [trainer_goal])
        self.assertIsNotNone(controller._tactical_loop)

    def test_pending_recovery_blocks_activate_trigger_for_mandatory_trainer(self):
        objective = next(
            item for item in initial_emerald_campaign() if item.objective_id == "complete_intro_rival"
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "introductory rival")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "introductory rival",
            tactical_goal=objective.tactical_target,
            capability=lambda: iter(()),
        )
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=objective.destination,
            navigation_goal=objective.tactical_target,
            current_map=(0, 18),
            current_coordinates=(8, 6),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(PartyReadinessMember(0, "Mudkip", 15, 22, 15 / 22, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=None,
            recovery=SimpleNamespace(calculation_pending=True),
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.CONTINUE,
            readiness_reason=ReadinessReason.PARTY_HEALTHY,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: readiness,
        )

        result = controller.step()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertIsNone(controller._tactical_loop)
        self.assertEqual(controller._execution_phase, "RECOVERY_WAIT")
        self.assertIn("recovery route calculation pending", result.reason)

    def test_blocked_encounter_stage_keeps_polling_pending_recovery(self):
        objective = CampaignObjective(
            "obtain_encounter:0:19",
            "Obtain the first encounter at Route 104",
            (),
            self.fixture.predicate("encounter_done", False),
            destination=MapRSE.ROUTE104.value,
            tactical_target=Goal(),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "Route 104 encounter")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "encounter capability",
            tactical_goal=Goal(),
            capability=lambda: iter(()),
        )
        stage = SimpleNamespace(
            blocked=True,
            current_map=MapRSE.ROUTE104.value,
            next_map=MapRSE.PETALBURG_WOODS.value,
            world_signature=("route104", 1),
            known_hazard_ids=("trainer:route104:28",),
            unresolved_hazard_ids=(),
            reason="stage_unavailable",
        )

        def diagnostic(*, pending):
            return ProgressionReadinessDiagnostic(
                objective_id=objective.objective_id,
                objective_status="ready",
                destination=objective.destination,
                navigation_goal=Goal(),
                current_map=MapRSE.ROUTE104.value,
                current_coordinates=(39, 63),
                game_state="OVERWORLD",
                campaign_mode="test",
                party=(),
                has_usable_pokemon=True,
                healing_available=True,
                recovery=SimpleNamespace(calculation_pending=pending),
                trainers=(),
                party_availability=Availability.KNOWN,
                trainer_availability=Availability.KNOWN,
                recovery_availability=Availability.UNKNOWN if pending else Availability.KNOWN,
                navigation_availability=Availability.KNOWN,
                overworld_availability=Availability.KNOWN,
                resource_availability=Availability.KNOWN,
                readiness_decision=ReadinessDecision.CONTINUE,
                readiness_reason=ReadinessReason.PARTY_HEALTHY,
            )

        observations = iter((diagnostic(pending=True), diagnostic(pending=True)))
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: next(observations),
        )

        previous_stage = getattr(context, "campaign_encounter_stage", None)
        context.campaign_encounter_stage = stage
        try:
            first = controller.step()
            second = controller.step()
        finally:
            context.campaign_encounter_stage = previous_stage

        self.assertEqual(first.execution_phase, "RECOVERY_WAIT")
        self.assertEqual(second.execution_phase, "RECOVERY_WAIT")
        self.assertEqual(controller._refresh_required_reason, "encounter_stage_recovery_pending")
        self.assertIsNone(controller._tactical_loop)

    def test_blocked_encounter_stage_mounts_completed_recovery_route(self):
        """A completed safe Center route must own a blocked encounter handoff."""

        objective = CampaignObjective(
            "obtain_encounter:0:19",
            "Obtain the first encounter at Route 104",
            (),
            self.fixture.predicate("encounter_done", False),
            destination=MapRSE.ROUTE104.value,
            tactical_target=Goal(),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "Route 104 encounter")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "encounter capability",
            tactical_goal=Goal(),
            capability=lambda: iter(()),
        )
        route = SimpleNamespace(actions=("up", "left"), metrics=SimpleNamespace(total_route_cost=2))
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=objective.destination,
            navigation_goal=Goal(),
            current_map=MapRSE.ROUTE104.value,
            current_coordinates=(39, 63),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=SimpleNamespace(
                center_available=True,
                safe_to_reach_center=True,
                route=route,
                route_origin=(MapRSE.ROUTE104.value, (39, 63)),
                center_location=(MapRSE.PETALBURG_CITY.value, (20, 16)),
                calculation_pending=False,
            ),
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            # The worker can publish a completed executable route one
            # observation before the aggregate availability field catches up.
            # This is the state recorded by the Route 104 stall log.
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.UNKNOWN,
            readiness_reason=ReadinessReason.TRAINER_HAZARD_UNKNOWN,
        )
        stage = SimpleNamespace(
            blocked=True,
            current_map=MapRSE.ROUTE104.value,
            next_map=MapRSE.PETALBURG_WOODS.value,
            world_signature=("route104", 1),
            known_hazard_ids=("trainer:route104:28",),
            unresolved_hazard_ids=(),
            reason="stage_unavailable",
        )
        mounted = []

        def recovery_factory(stop):
            mounted.append(stop)
            return iter(())

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: readiness,
            recovery_factory=recovery_factory,
        )

        previous_stage = getattr(context, "campaign_encounter_stage", None)
        context.campaign_encounter_stage = stage
        try:
            result = controller.step()
        finally:
            context.campaign_encounter_stage = previous_stage

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.execution_phase, "RECOVERY")
        self.assertEqual(len(mounted), 1)
        self.assertIs(mounted[0].route, route)

    def test_blocked_encounter_stage_retries_with_bounded_backoff(self):
        objective = CampaignObjective(
            "obtain_encounter:0:19",
            "Obtain the first encounter at Route 104",
            (),
            self.fixture.predicate("encounter_done", False),
            destination=MapRSE.ROUTE104.value,
            tactical_target=Goal(),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "Route 104 encounter")
        closed = []

        def capability():
            try:
                yield
            finally:
                closed.append(True)

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "encounter capability",
            tactical_goal=Goal(),
            capability=capability,
        )
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=objective.destination,
            navigation_goal=Goal(),
            current_map=MapRSE.ROUTE104.value,
            current_coordinates=(39, 63),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=SimpleNamespace(calculation_pending=False),
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.CONTINUE,
            readiness_reason=ReadinessReason.PARTY_HEALTHY,
        )
        stage = SimpleNamespace(
            blocked=True,
            current_map=MapRSE.ROUTE104.value,
            next_map=MapRSE.PETALBURG_WOODS.value,
            world_signature=("route104", 1),
            known_hazard_ids=("trainer:route104:28",),
            unresolved_hazard_ids=(),
            reason="stage_unavailable",
        )
        calls = []

        def provider(*_):
            calls.append(1)
            return readiness

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=provider,
        )
        mounted_loop = SimpleNamespace(close=lambda: closed.append(True))
        controller._tactical_loop = mounted_loop
        controller._tactical_loop_objective_id = objective.objective_id
        previous_stage = getattr(context, "campaign_encounter_stage", None)
        context.campaign_encounter_stage = stage
        try:
            first = controller.step()
            second = controller.step()
            third = controller.step()
            fourth = controller.step()
        finally:
            context.campaign_encounter_stage = previous_stage

        self.assertEqual(first.status, CampaignControllerStatus.UNKNOWN)
        self.assertEqual(len(calls), 2)
        self.assertEqual(second.status, CampaignControllerStatus.UNKNOWN)
        self.assertEqual(third.status, CampaignControllerStatus.UNKNOWN)
        self.assertEqual(fourth.status, CampaignControllerStatus.UNKNOWN)
        self.assertEqual(controller._encounter_stage_retry_failures, 2)
        self.assertEqual(controller._encounter_stage_retry_delay, 4)
        self.assertTrue(controller._refresh_required)
        self.assertEqual(closed, [True])

    def test_trainer_recovery_promotion_recomposes_recovery_stop(self):
        """A late trainer precondition promotion must carry an executable stop."""

        objective = self.fixture.objective("trainer_recovery", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "trainer recovery")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "trainer route",
            tactical_goal=EngageTrainer("calvin"),
        )
        recovery_route = SimpleNamespace(
            actions=(),
            metrics=SimpleNamespace(total_route_cost=11),
        )
        destination = ((0, 17), (6, 16))
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=None,
            navigation_goal=EngageTrainer("calvin"),
            current_map=(0, 17),
            current_coordinates=(11, 14),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(PartyReadinessMember(0, "Treecko", 7, 10, 0.4, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=SimpleNamespace(
                center_available=True,
                safe_to_reach_center=True,
                center_location=destination,
                route=recovery_route,
                navigation_goal=ReachLocation(destination),
                healing_source=None,
                distance_to_center=10,
                calculation_pending=False,
            ),
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.CONTINUE,
            readiness_reason=ReadinessReason.PARTY_HEALTHY,
            trainer_readiness=SimpleNamespace(required_hp_ratio=0.5),
        )
        recovery_starts = []

        def recovery_factory(stop):
            recovery_starts.append(stop)
            yield

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: readiness,
            recovery_factory=recovery_factory,
        )

        result = controller.step()

        self.assertEqual(result.execution_phase, "RECOVERY")
        self.assertEqual(len(recovery_starts), 1)
        self.assertIs(recovery_starts[0].route, recovery_route)
        self.assertEqual(recovery_starts[0].destination, destination)
        self.assertEqual(controller.state.plan.readiness_decision, ReadinessDecision.RECOVER)

    def test_hurt_unready_trainer_uses_safe_center_route_beyond_opportunistic_detour(self):
        """A mandatory trainer cannot bypass healing solely because the route is long."""

        objective = self.fixture.objective("trainer_recovery_long_route", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "trainer recovery")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "trainer route",
            tactical_goal=EngageTrainer("calvin"),
        )
        recovery_route = SimpleNamespace(
            actions=(),
            metrics=SimpleNamespace(total_route_cost=54),
        )
        destination = ((0, 17), (6, 16))
        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=None,
            navigation_goal=EngageTrainer("calvin"),
            current_map=(0, 17),
            current_coordinates=(11, 14),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(PartyReadinessMember(0, "Treecko", 11, 20, 0.55, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=SimpleNamespace(
                center_available=True,
                safe_to_reach_center=True,
                center_location=destination,
                route=recovery_route,
                navigation_goal=ReachLocation(destination),
                healing_source=None,
                distance_to_center=54,
                calculation_pending=False,
            ),
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.PREPARE,
            readiness_reason=ReadinessReason.TRAINER_BATTLE_UNREADY,
            trainer_readiness=SimpleNamespace(
                required_hp_ratio=0.5,
                status=SimpleNamespace(value="prepare"),
            ),
        )
        recovery_starts = []

        def recovery_factory(stop):
            recovery_starts.append(stop)
            yield

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: readiness,
            recovery_factory=recovery_factory,
        )

        result = controller.step()

        self.assertEqual(result.execution_phase, "RECOVERY")
        self.assertEqual(len(recovery_starts), 1)
        self.assertIs(recovery_starts[0].route, recovery_route)
        self.assertEqual(recovery_starts[0].destination, destination)

    def test_deferred_readiness_publishes_new_execution_status(self):
        objective = CampaignObjective(
            "next_objective",
            "Next Objective",
            (),
            self.fixture.predicate("next_done", False),
            destination=(0, 17),
            tactical_target=NavigationGoal(SemanticTarget.map((0, 17))),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "next frontier")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "next capability",
            tactical_goal=objective.tactical_target,
            capability=lambda: iter(()),
        )
        unknown = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=objective.destination,
            navigation_goal=objective.tactical_target,
            current_map=(0, 18),
            current_coordinates=(10, 3),
            game_state="BATTLE",
            campaign_mode="test",
            party=(),
            has_usable_pokemon=None,
            healing_available=None,
            recovery=None,
            trainers=(),
            party_availability=Availability.UNKNOWN,
            trainer_availability=Availability.UNKNOWN,
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.UNKNOWN,
            resource_availability=Availability.UNKNOWN,
        )
        previous_status = context.campaign_status
        try:
            context.campaign_status = CampaignStatus(
                "Wally's Tutorial",
                SemanticTarget.map((8, 1)),
                "Waiting / target temporarily unavailable",
            )
            controller = CampaignController(
                lambda: self.current,
                selector=lambda _: selection,
                adapter=lambda _: execution,
                readiness_provider=lambda *_: unknown,
            )

            result = controller.step()

            self.assertEqual(result.status, CampaignControllerStatus.READY)
            self.assertEqual(context.campaign_status.objective, "Next Objective")
            self.assertEqual(context.campaign_status.target, SemanticTarget.map((0, 17)))
            self.assertEqual(context.campaign_status.intent, "Waiting / readiness temporarily unavailable")
        finally:
            context.campaign_status = previous_status

    def test_deferred_readiness_does_not_advance_stale_previous_capability(self):
        pokedex = next(item for item in initial_emerald_campaign() if item.objective_id == "receive_pokedex")
        next_objective = next(item for item in initial_emerald_campaign() if item.objective_id == "complete_intro_rival")
        selections = iter(
            (
                ObjectiveSelection(pokedex, ObjectiveStatus.READY, "pokedex frontier"),
                ObjectiveSelection(next_objective, ObjectiveStatus.READY, "next frontier"),
            )
        )
        advanced = []

        def adapter(selection):
            def capability():
                advanced.append(selection.objective.objective_id)
                while True:
                    yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
                tactical_goal=selection.objective.tactical_target,
            )

        unknown = ProgressionReadinessDiagnostic(
            objective_id="complete_intro_rival",
            objective_status="ready",
            destination=next_objective.destination,
            navigation_goal=next_objective.tactical_target,
            current_map=(1, 4),
            current_coordinates=(6, 5),
            game_state="BATTLE",
            campaign_mode="test",
            party=(),
            has_usable_pokemon=None,
            healing_available=None,
            recovery=None,
            trainers=(),
            party_availability=Availability.UNKNOWN,
            trainer_availability=Availability.UNKNOWN,
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.UNKNOWN,
            resource_availability=Availability.UNKNOWN,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
            readiness_provider=lambda *_: unknown,
        )

        controller.step()
        controller.request_refresh("script_boundary")
        result = controller.step()

        self.assertEqual(advanced, ["receive_pokedex"])
        self.assertEqual(result.objective_id, "complete_intro_rival")
        self.assertIsNone(controller._tactical_loop)
        self.assertTrue(controller._refresh_required)

    def test_later_active_objective_releases_on_coherent_save_regression(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        active = objectives["prepare_roxanne"]
        earlier = objectives["set_text_speed"]
        self.state(campaign_facts=self.fixture.facts())
        selections = iter(
            (
                ObjectiveSelection(active, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "coherent regression"),
            )
        )

        def adapter(selection):
            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=lambda: iter(("advanced",)),
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        controller.step()
        controller.request_refresh("test_selection_boundary")
        result = controller.step()

        self.assertEqual(result.objective_id, "set_text_speed")

    def test_canonical_earlier_objective_cannot_retain_over_later_selection(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        earlier = objectives["set_text_speed"]
        later = objectives["prepare_roxanne"]
        selections = iter(
            (
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(later, ObjectiveStatus.READY, "later planner result"),
            )
        )
        mounted = []

        def adapter(selection):
            def capability():
                mounted.append(selection.objective.objective_id)
                yield
                while True:
                    yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        controller.step()
        controller.request_refresh("test_selection_boundary")
        result = controller.step()

        self.assertEqual(result.objective_id, "prepare_roxanne")
        self.assertEqual(mounted, ["set_text_speed", "prepare_roxanne"])

    def test_capability_only_execution_advances_to_input_boundary(self):
        objective = self.fixture.objective("dialogue", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "actionable dialogue")
        emitted = []

        def capability():
            emitted.append("ADVANCE_DIALOGUE")
            yield

        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability-only dialogue", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )

        state = controller.step()

        self.assertEqual(state.status, CampaignControllerStatus.READY)
        self.assertIs(controller.last_execution.capability, capability)
        self.assertEqual(emitted, ["ADVANCE_DIALOGUE"])
        self.assertIsNone(state.tactical_goal)

    def test_capability_boundary_is_not_failure(self):
        objective = self.fixture.objective("boundary", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")

        def capability():
            if False:
                yield

        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )
        state = controller.step()
        self.assertEqual(state.status, CampaignControllerStatus.READY)
        self.assertIn("capability boundary", state.reason)

    def test_capability_boundary_retains_incomplete_story_owner(self):
        story = self.fixture.objective("story_scene", self.fixture.predicate("story_done", False))
        encounter = self.fixture.objective("obtain_encounter:Route 103", self.fixture.predicate("encounter_done", False))
        selections = iter(
            (
                ObjectiveSelection(story, ObjectiveStatus.READY, "story scene"),
                ObjectiveSelection(encounter, ObjectiveStatus.READY, "optional encounter"),
            )
        )

        def adapter(selection):
            def capability():
                if False:
                    yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        first = controller.step()
        self.assertIn("capability boundary", first.reason)

        second = controller.step()

        self.assertEqual(second.objective_id, "story_scene")
        self.assertEqual(second.selection.objective.objective_id, "story_scene")

    def test_battle_end_recheck_retains_story_capability_owner(self):
        story = next(
            item for item in initial_emerald_campaign() if item.objective_id == "complete_petalburg_wally"
        )
        encounter = self.fixture.objective(
            "obtain_encounter:Route 103", self.fixture.predicate("encounter_done", False)
        )
        self.current = replace(
            self.current,
            campaign_facts=self.fixture.facts(
                pokedex_received=True,
                pokeballs_received=True,
                pokeballs_ready=True,
            ),
        )
        selections = iter(
            (
                ObjectiveSelection(story, ObjectiveStatus.READY, "story scene"),
                ObjectiveSelection(encounter, ObjectiveStatus.READY, "optional encounter"),
            )
        )

        def adapter(selection):
            def capability():
                if False:
                    yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        first = controller.step()
        self.assertIn("capability boundary", first.reason)
        controller.request_readiness_recheck("battle_ended")

        second = controller.step()

        self.assertEqual(second.objective_id, "complete_petalburg_wally")
        self.assertEqual(second.selection.objective.objective_id, "complete_petalburg_wally")

    def test_live_capability_completion_latch_fences_stale_story_projection(self):
        story_done = [False]
        story = CampaignObjective(
            "complete_petalburg_wally",
            "Wally tutorial",
            (),
            CampaignPredicate("story_done", "story is complete", lambda _: Fact.known(story_done[0])),
        )
        encounter = CampaignObjective(
            "obtain_encounter:Route 103",
            "Route 103 encounter",
            (),
            CampaignPredicate("encounter_done", "encounter is complete", lambda _: Fact.known(False)),
        )
        selections = [ObjectiveSelection(story, ObjectiveStatus.READY, "story"), ObjectiveSelection(encounter, ObjectiveStatus.READY, "encounter")]

        def select(_state):
            return selections[1] if story_done[0] else selections[0]

        def adapter(selection):
            def capability():
                if False:
                    yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(lambda: self.current, selector=select, adapter=adapter)
        with patch.object(context, "campaign_capability_completion_latch", "complete_petalburg_wally"):
            first = controller.step()
            self.assertEqual(first.objective_id, "complete_petalburg_wally")
            completed = controller.step()
            self.assertIn("capability completed", completed.reason)
            waiting = controller.step()
            self.assertEqual(waiting.objective_id, "complete_petalburg_wally")
            self.assertEqual(waiting.selection.objective.objective_id, "complete_petalburg_wally")

            story_done[0] = True
            released = controller.step()

        self.assertEqual(released.objective_id, "obtain_encounter:Route 103")

    def test_capability_return_releases_objective_when_completion_is_authoritative(self):
        completed = [False]
        objective = self.fixture.objective(
            "completed_capability",
            CampaignPredicate("completed", "completed", lambda _: Fact.known(completed[0])),
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")

        def capability():
            yield
            completed[0] = True

        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability", capability=capability
        )
        next_objective = self.fixture.objective("next_capability", self.fixture.predicate("next", False))
        next_selection = ObjectiveSelection(next_objective, ObjectiveStatus.READY, "next")
        selections = iter((selection, next_selection))
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda selected: (
                execution
                if selected.objective is objective
                else CampaignExecutionResult(
                    selected.objective,
                    CampaignExecutionStatus.READY,
                    "next capability",
                    capability=lambda: iter(()),
                )
            ),
        )

        controller.step()
        state = controller.step()

        self.assertEqual(state.status, CampaignControllerStatus.READY)
        self.assertIn("capability completed", state.reason)
        self.assertIsNone(controller._tactical_loop)
        self.assertTrue(controller._refresh_required)

        controller.step()

        self.assertEqual(controller.last_selection.objective.objective_id, "next_capability")

    def test_completed_rival_capability_hands_off_to_pokedex_frontier(self):
        mounted = []

        def adapter(selection):
            objective_id = selection.objective.objective_id

            def capability():
                mounted.append(objective_id)
                if objective_id == "complete_intro_rival":
                    return
                yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda state: plan_campaign(state, goal=ultimate_emerald_campaign_goal()),
            adapter=adapter,
        )

        first = controller.step()
        self.assertIn("capability boundary", first.reason)
        self.assertEqual(mounted, ["complete_intro_rival"])

        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
            )
        )
        second = controller.step()

        self.assertEqual(second.objective_id, "receive_pokedex")
        self.assertEqual(mounted, ["complete_intro_rival", "receive_pokedex"])

    def test_failed_capability_is_cleared_and_can_be_rebuilt(self):
        objective = self.fixture.objective("recoverable", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        attempts = []

        def capability():
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise RuntimeError("temporary recovery lookup failure")
            yield

        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )
        first = controller.step()
        self.assertEqual(first.status, CampaignControllerStatus.UNKNOWN)
        self.assertIsNone(controller._tactical_loop)
        second = controller.step()
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertEqual(attempts, [0, 1])

    def test_preparation_setup_failure_does_not_escalate_level_target(self):
        controller = CampaignController(lambda: self.current)
        controller._execution_phase = "PREPARATION"
        controller.status = CampaignControllerStatus.READY
        controller._trainer_preparation_checkpoint_key = ("complete_petalburg_wally", "route102", (0, 18))
        controller._trainer_preparation_last_target = 6

        def failing_preparation():
            raise RuntimeError("recovery path setup failed")
            yield

        controller._tactical_loop = failing_preparation()

        result = controller.step()

        self.assertEqual(result.status, CampaignControllerStatus.UNKNOWN)
        self.assertIsNone(controller._trainer_preparation_checkpoint_key)
        self.assertIsNone(controller._trainer_preparation_last_target)
        self.assertEqual(controller._refresh_required_reason, "trainer_preparation_exception")

    def test_transient_keyerror_in_recovery_remounts_bounded_retry(self):
        objective = self.fixture.objective("recoverable", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "capability",
            tactical_goal=Goal(),
        )
        attempts = []

        def recovery(_stop):
            attempt = len(attempts)
            attempts.append(attempt)
            if attempt == 0:

                def failed_before_first_frame():
                    raise KeyError("transient map table")
                    if False:
                        yield

                return failed_before_first_frame()
            return iter((None,))

        readiness = ProgressionReadinessDiagnostic(
            objective_id=objective.objective_id,
            objective_status="ready",
            destination=None,
            navigation_goal=Goal(),
            current_map=(1, 2),
            current_coordinates=(3, 4),
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(PartyReadinessMember(0, "Treecko", 1, 10, 0.1, False, True, "none"),),
            has_usable_pokemon=True,
            healing_available=True,
            recovery=None,
            trainers=(),
            party_availability=Availability.KNOWN,
            trainer_availability=Availability.KNOWN,
            recovery_availability=Availability.KNOWN,
            navigation_availability=Availability.KNOWN,
            overworld_availability=Availability.KNOWN,
            resource_availability=Availability.KNOWN,
            readiness_decision=ReadinessDecision.RECOVER,
            readiness_reason=ReadinessReason.CRITICAL_PARTY_HP,
            route_analysis=RouteAnalysis(
                Goal(),
                "normal-route",
                40,
                (
                    IntermediateRouteAnalysis(
                        ReachLocation(("Oldale", (5, 6))),
                        48,
                        8,
                        True,
                        first_route=SimpleNamespace(metrics=SimpleNamespace(total_route_cost=8)),
                    ),
                ),
            ),
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: readiness,
            recovery_factory=recovery,
        )

        retry = controller.step()
        self.assertEqual(retry.status, CampaignControllerStatus.READY)
        self.assertEqual(retry.execution_phase, "RECOVERY")
        self.assertIn("transient KeyError", retry.reason)
        self.assertEqual(attempts, [0, 1])

        controller.step()
        completed = controller.step()
        self.assertEqual(completed.recovery_status, "COMPLETED")
        self.assertEqual(controller._recovery_keyerror_retries, 0)

    def test_identical_recovery_failures_escalate_after_three_attempts(self):
        controller = CampaignController(lambda: self.current)
        error = RuntimeError("no safe cataloged healing route")

        self.assertFalse(controller._record_recovery_failure(error))
        self.assertFalse(controller._record_recovery_failure(error))
        with patch("modules.nuzlocke.campaign_controller.print_recovery_failure") as print_failure:
            self.assertTrue(controller._record_recovery_failure(error))

        self.assertEqual(controller.status, CampaignControllerStatus.BLOCKED)
        self.assertEqual(controller._execution_phase, "BLOCKED")
        self.assertEqual(controller._recovery_status, "ESCALATED")
        print_failure.assert_called_once()
        self.assertTrue(print_failure.call_args.kwargs["escalated"])
        self.assertEqual(print_failure.call_args.kwargs["repeated"], 3)

    def test_completion_clears_goal_before_unsupported_next_objective(self):
        controller = self.controller()
        controller.refresh()
        old_goal = controller.current_tactical_goal
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
            )
        )
        controller.refresh()
        result = controller.refresh()
        self.assertNotEqual(old_goal, None)
        self.assertIsNone(result.tactical_goal)
        self.assertEqual(result.selection.objective.objective_id, "receive_pokedex")
        self.assertEqual(self.created.__len__(), 0)

    def test_pokedex_boundary_requires_no_controller_handoff(self):
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_ready=True,
            )
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda state: plan_campaign(state, goal=ultimate_emerald_campaign_goal()),
            adapter=adapt_campaign_execution,
            tactical_loop_factory=self.factory,
        )
        controller.refresh()
        result = controller.refresh()
        self.assertEqual(result.selection.objective.objective_id, "complete_petalburg_wally")
        self.assertEqual(result.status, CampaignControllerStatus.READY)

    def test_campaign_snapshot_deduplicates_semantic_state_across_frames(self):
        controller = CampaignController(lambda: self.current)
        party = (
            SimpleNamespace(
                species="Treecko",
                level=7,
                current_hp=20,
                max_hp=20,
                status="none",
            ),
        )
        player = SimpleNamespace(map_group=0, map_number=17, map_name="ROUTE102")
        runtime = SimpleNamespace(latest_snapshot=SimpleNamespace(frame=101, player=player, party=party))
        status = CampaignStatus("Reach Petalburg", None, "Waiting")

        with (
            patch.object(context, "nuzlocke_runtime", runtime),
            patch.object(context, "campaign_trainer_checkpoint", None),
            patch("modules.nuzlocke.campaign_controller.print_campaign_snapshot") as print_snapshot,
        ):
            controller._emit_campaign_snapshot(status)
            runtime.latest_snapshot = SimpleNamespace(frame=102, player=player, party=party)
            controller._emit_campaign_snapshot(status)
            runtime.latest_snapshot = SimpleNamespace(
                frame=103,
                player=SimpleNamespace(map_group=0, map_number=18, map_name="ROUTE103"),
                party=party,
            )
            controller._emit_campaign_snapshot(status)

        self.assertEqual(print_snapshot.call_count, 2)

    def test_petalburg_progression_is_gated_by_pokedex_receipt(self):
        objective = next(item for item in initial_emerald_campaign() if item.objective_id == "complete_petalburg_wally")
        facts = self.fixture.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=False,
            pokeballs_available=True,
            pokeballs_ready=True,
        )
        state = self.fixture.state(campaign_facts=facts)
        selection = select_campaign_objective(state, (objective,))
        self.assertEqual(selection.status, ObjectiveStatus.BLOCKED)

    def test_controller_mounts_preparation_after_pokedex_boundary(self):
        self.current = replace(
            self.current,
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_available=True,
                pokeballs_ready=True,
            ),
            party=Fact.known((replace(self.current.party.value[0], current_hp=19),)),
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda state: plan_campaign(state, goal=ultimate_emerald_campaign_goal()),
            adapter=adapt_campaign_execution,
        )

        controller.refresh()
        result = controller.refresh()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.objective_id, "complete_petalburg_wally")
        self.assertEqual(result.execution.execution_id, "complete_petalburg_wally")
        self.assertIsNotNone(result.execution.capability)


from modules.map_data import MapRSE

if __name__ == "__main__":
    unittest.main()


class ControllerDiagnosticsTests(unittest.TestCase):
    def test_repeated_identical_exceptions_are_rate_limited(self):
        from modules.nuzlocke.campaign_controller import _ControllerExceptionReporter

        reporter = _ControllerExceptionReporter()
        error = RuntimeError("stale route")
        reported = [reporter.should_report("trainer_preparation", error)[0] for _ in range(12)]

        self.assertEqual([index + 1 for index, value in enumerate(reported) if value], [1, 2, 3, 10])
        # A different kind or message is counted independently.
        self.assertTrue(reporter.should_report("refresh", error)[0])
        self.assertTrue(reporter.should_report("trainer_preparation", RuntimeError("other"))[0])

    def test_exception_report_goes_to_the_normal_log_with_a_location(self):
        from unittest.mock import patch

        from modules.nuzlocke import campaign_controller as controller

        logs = []
        flags = []

        def capture(message, *args, **kwargs):
            logs.append(message())
            flags.append(kwargs.get("trace"))

        def boom():
            raise ValueError("no executable capability")

        try:
            boom()
        except ValueError as error:
            with patch.object(controller, "diagnostic_print", side_effect=capture), patch.object(
                controller, "_CONTROLLER_EXCEPTIONS", controller._ControllerExceptionReporter()
            ):
                controller._report_controller_exception("trainer_preparation", error, objective="x")

        self.assertEqual(flags, [False])
        self.assertIn("CAMPAIGN_CONTROLLER_EXCEPTION", logs[0])
        self.assertIn("no executable capability", logs[0])
        self.assertIn("boom:", logs[0])

    def test_idle_watchdog_reports_once_per_period_and_resets_on_change(self):
        from modules.nuzlocke.campaign_controller import _ControllerIdleWatchdog

        watchdog = _ControllerIdleWatchdog(first_report_after=3, repeat_every=5)
        state = ("CAMPAIGN", "unknown", "waiting", "obj")
        results = [watchdog.observe(state, idle=True) for _ in range(9)]

        self.assertEqual(results, [None, None, 3, None, None, None, None, 8, None])
        # A mounted capability, or a changed state, restarts the count.
        self.assertIsNone(watchdog.observe(state, idle=False))
        self.assertEqual([watchdog.observe(state, idle=True) for _ in range(3)], [None, None, 3])
        self.assertIsNone(watchdog.observe(("CAMPAIGN", "ready", "other", "obj"), idle=True))
