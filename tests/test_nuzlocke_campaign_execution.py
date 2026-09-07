import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.goals import (
    ActivateTrigger,
    EngageTrainer,
    GoalConstraints,
    NavigationGoal,
    SemanticTarget,
    SemanticTargetKind,
    TrainerMode,
)
from modules.map_data import MapRSE
from modules.nuzlocke.campaign_execution import (
    CampaignExecutionStatus,
    adapt_campaign_execution,
)
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    ObjectiveSelection,
    ObjectiveStatus,
    initial_emerald_campaign,
    restock_pokeballs_objective,
    restock_recovery_items_objective,
)
from modules.nuzlocke.campaign_state import Fact


class CampaignExecutionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}

    @staticmethod
    def ready(objective):
        return ObjectiveSelection(objective, ObjectiveStatus.READY, "ready for test")

    def test_introductory_rival_translates_to_existing_trigger_goal(self):
        result = adapt_campaign_execution(self.ready(self.objectives["complete_intro_rival"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "intro_rival")
        self.assertEqual(result.tactical_goal, ActivateTrigger("introductory_rival"))
        self.assertEqual(result.capability.tactical_goal, ActivateTrigger("introductory_rival"))

    def test_birch_handoff_objective_mounts_observation_capability(self):
        result = adapt_campaign_execution(self.ready(self.objectives["receive_pokedex"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "receive_pokedex")
        self.assertIsNone(result.tactical_goal)
        self.assertIsNotNone(result.capability)

    def test_birch_handoff_objective_targets_birch_lab(self):
        from modules.nuzlocke.emerald_capabilities import _semantic_target_for_objective

        target = _semantic_target_for_objective("receive_pokedex")
        self.assertEqual(
            target,
            SemanticTarget.map(MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value),
        )

    def test_pokeball_restock_translates_to_mart_capability(self):
        objective = restock_pokeballs_objective()
        result = adapt_campaign_execution(self.ready(objective))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "restock_pokeballs")
        self.assertIsNotNone(result.capability)

    def test_recovery_restock_translates_to_mart_capability(self):
        objective = restock_recovery_items_objective()
        result = adapt_campaign_execution(self.ready(objective))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "restock_recovery_items")
        self.assertIsNotNone(result.capability)

    def test_opening_objectives_mount_capabilities(self):
        supported = {
            "set_text_speed",
            "complete_new_game_setup",
            "set_wall_clock",
            "meet_rival",
            "rescue_birch",
            "obtain_starter",
            "receive_pokedex",
            "reach_petalburg",
            "complete_petalburg_wally",
            "complete_petalburg_woods",
            "complete_rustboro_goods_stolen",
            "report_devon_goods",
            "recover_devon_goods",
            "return_devon_goods",
            "meet_mr_stone",
            "reach_rustboro",
            "prepare_roxanne",
            "defeat_roxanne",
            "complete_intro_rival",
        }
        for objective_id, objective in self.objectives.items():
            if objective_id in supported:
                result = adapt_campaign_execution(self.ready(objective))
                self.assertEqual(result.status, CampaignExecutionStatus.READY, objective_id)
                if objective_id != "complete_intro_rival":
                    self.assertIsNotNone(result.capability, objective_id)
                continue
            result = adapt_campaign_execution(self.ready(objective))
            self.assertEqual(
                result.status,
                CampaignExecutionStatus.READY if objective_id in supported else CampaignExecutionStatus.UNSUPPORTED,
                objective_id,
            )
            self.assertEqual(result.execution_id, objective.execution_id, objective_id)

    def test_party_restoration_mounts_healing_capability(self):
        from modules.nuzlocke.campaign_objectives import heal_party_objective

        objective = heal_party_objective()
        result = adapt_campaign_execution(self.ready(objective))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "heal_party")
        self.assertIsNotNone(result.capability)

    def test_receive_pokedex_navigates_to_birch_lab(self):
        result = adapt_campaign_execution(self.ready(self.objectives["receive_pokedex"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertIsNone(result.tactical_goal)
        self.assertIsNotNone(result.capability)

    def test_reach_petalburg_mounts_shared_emerald_navigation_capability(self):
        result = adapt_campaign_execution(self.ready(self.objectives["reach_petalburg"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "reach_petalburg")
        self.assertIsNone(result.tactical_goal)
        self.assertIsNotNone(result.capability)

    def test_reach_petalburg_resolves_to_map_semantic_target(self):
        from modules.nuzlocke.emerald_capabilities import _semantic_target_for_objective

        target = _semantic_target_for_objective("reach_petalburg")
        self.assertIsNotNone(target)
        self.assertEqual(target.kind, SemanticTargetKind.MAP)
        self.assertEqual(target.target_map, MapRSE.PETALBURG_CITY.value)

    def test_petalburg_wally_targets_norman_in_the_gym(self):
        from modules.nuzlocke.emerald_capabilities import _semantic_target_for_objective

        result = adapt_campaign_execution(self.ready(self.objectives["complete_petalburg_wally"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertIsNotNone(result.capability)
        target = _semantic_target_for_objective("complete_petalburg_wally")
        self.assertIsNotNone(target)
        self.assertEqual(target.kind, SemanticTargetKind.INTERACTION)
        self.assertEqual(target.target_map, MapRSE.PETALBURG_CITY_GYM.value)
        self.assertEqual(target.interaction_id, "PetalburgCity_Gym_EventScript_Norman")

    def test_first_badge_capability_is_registry_owned_and_readiness_aware(self):
        result = adapt_campaign_execution(self.ready(self.objectives["defeat_roxanne"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertIsNotNone(result.capability)
        self.assertEqual(
            result.tactical_goal,
            NavigationGoal(
                SemanticTarget.interaction(MapRSE.RUSTBORO_CITY_GYM.value, interaction_id="roxanne"),
                constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
            ),
        )

    def test_preparation_capability_uses_registry_owned_target_level(self):
        with (
            patch(
                "modules.nuzlocke.resource_runtime.execute_campaign_preparation",
                return_value=iter(("training",)),
            ) as execute,
            patch("modules.nuzlocke.resource_runtime.get_party", return_value=()),
        ):
            result = adapt_campaign_execution(self.ready(self.objectives["prepare_roxanne"]))
            self.assertEqual(result.status, CampaignExecutionStatus.READY)
            self.assertIsNotNone(result.capability)
            self.assertEqual(list(result.capability()), ["training"])
        execute.assert_called_once_with(MapRSE.ROUTE116.value, target_level=14)

    def test_selection_statuses_are_preserved_without_translation(self):
        objective = self.objectives["set_text_speed"]
        for objective_status, execution_status in (
            (ObjectiveStatus.BLOCKED, CampaignExecutionStatus.BLOCKED),
            (ObjectiveStatus.UNKNOWN, CampaignExecutionStatus.UNKNOWN),
            (ObjectiveStatus.FAILED, CampaignExecutionStatus.FAILED),
        ):
            result = adapt_campaign_execution(ObjectiveSelection(objective, objective_status, "test reason"))
            self.assertEqual(result.status, execution_status)
            self.assertEqual(result.reason, "test reason")
            self.assertIsNone(result.tactical_goal)

    def test_completed_selection_returns_complete(self):
        result = adapt_campaign_execution(ObjectiveSelection(None, ObjectiveStatus.COMPLETE, "all complete"))
        self.assertEqual(result.status, CampaignExecutionStatus.COMPLETE)
        self.assertEqual(result.reason, "all complete")
        self.assertIsNone(result.objective)
        self.assertIsNone(result.tactical_goal)

    def test_unknown_objective_id_is_unsupported(self):
        predicate = CampaignPredicate("never", "never", lambda _: Fact.known(False))
        objective = CampaignObjective("not_known", "Unknown", (), predicate, execution_id="unknown")
        result = adapt_campaign_execution(self.ready(objective))
        self.assertEqual(result.status, CampaignExecutionStatus.UNSUPPORTED)
        self.assertIn("unknown campaign objective ID", result.reason)
        self.assertEqual(result.execution_id, "unknown")

    def test_malformed_selection_fails_safely(self):
        result = adapt_campaign_execution(ObjectiveSelection("not-an-objective", ObjectiveStatus.READY, "bad"))
        self.assertEqual(result.status, CampaignExecutionStatus.UNSUPPORTED)
        self.assertEqual(result.reason, "malformed ready objective")

        result = adapt_campaign_execution(None)
        self.assertEqual(result.status, CampaignExecutionStatus.UNSUPPORTED)
        self.assertEqual(result.reason, "malformed objective selection")

    def test_known_objective_with_invalid_tactical_target_is_unsupported(self):
        objective = CampaignObjective(
            "complete_intro_rival",
            "Oldale",
            (),
            self.objectives["complete_intro_rival"].completion,
            execution_id="intro_rival",
            tactical_target=ActivateTrigger("wrong"),
        )
        result = adapt_campaign_execution(self.ready(objective))
        self.assertEqual(result.status, CampaignExecutionStatus.UNSUPPORTED)
        self.assertIn("valid introductory rival goal", result.reason)

    def test_trainer_preparation_mounts_targeted_navigation(self):
        predicate = CampaignPredicate("never", "never", lambda _: Fact.known(False))
        objective = CampaignObjective(
            "prepare_trainer:trainer-a",
            "Train against trainer-a",
            (),
            predicate,
            execution_id="prepare_trainer:trainer-a",
            tactical_target=NavigationGoal(
                EngageTrainer("trainer-a"),
                constraints=GoalConstraints(trainer_mode=TrainerMode.ENGAGE),
            ),
        )
        result = adapt_campaign_execution(self.ready(objective))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.tactical_goal, objective.tactical_target)

    def test_translation_is_deterministic_and_does_not_mutate_inputs(self):
        objective = self.objectives["complete_intro_rival"]
        selection = self.ready(objective)
        before_objective = objective
        before_selection = selection
        first = adapt_campaign_execution(selection)
        second = adapt_campaign_execution(selection)
        self.assertEqual(first, second)
        self.assertIs(selection.objective, before_objective)
        self.assertEqual(selection, before_selection)

    def test_adapter_has_no_execution_side_effects(self):
        # The adapter is intentionally importable and callable without an
        # initialized emulator, profile, persistence store, or runtime.
        result = adapt_campaign_execution(self.ready(self.objectives["set_text_speed"]))
        self.assertIsNotNone(result)

    def test_campaign_mode_delegates_scripted_battle_to_existing_fight_controller(self):
        from modules.modes import BattleAction
        from modules.modes.campaign import CampaignProgressionMode

        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        self.assertIs(mode.on_battle_started(None), BattleAction.Fight)

    def test_campaign_mode_hands_legal_wild_encounter_to_catch_strategy(self):
        from modules.modes import BattleAction
        from modules.modes.campaign import CampaignProgressionMode

        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        objective = SimpleNamespace(objective_id="prepare_roxanne", resource_policy=None)
        mode.controller = SimpleNamespace(last_selection=SimpleNamespace(objective=objective))
        runtime = SimpleNamespace(
            capture_target_for=lambda location, *, is_wild, is_trainer: (
                location == (0, 17) and is_wild and not is_trainer
            )
        )
        avatar = SimpleNamespace(map_group_and_number=(0, 17))
        fake_context = SimpleNamespace(nuzlocke_runtime=runtime)

        with (
            patch("modules.modes.campaign.context", fake_context),
            patch("modules.modes.campaign.get_player_avatar", return_value=avatar),
            patch("modules.modes.campaign.get_party", return_value=()),
            patch(
                "modules.modes.campaign.runtime_campaign_state", return_value=SimpleNamespace(campaign_facts=object())
            ),
            patch("modules.modes.campaign.evaluate_battle_entry", return_value=SimpleNamespace(allowed=True)),
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            self.assertIs(mode.on_battle_started(object()), BattleAction.Catch)

    def test_campaign_mode_selects_nuzlocke_strategy_for_preparation_and_roxanne(self):
        from modules.modes.campaign import CampaignProgressionMode

        for objective_id, strategy_name in (
            ("complete_intro_rival", "EmeraldIntroRivalBattleStrategy"),
            ("prepare_roxanne", "NuzlockeLevelBalancingBattleStrategy"),
            ("defeat_roxanne", "RoxanneBattleStrategy"),
        ):
            mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
            objective = SimpleNamespace(objective_id=objective_id, resource_policy=None)
            mode.controller = SimpleNamespace(last_selection=SimpleNamespace(objective=objective))
            selected = object()
            with (
                patch("modules.modes.campaign.get_party", return_value=()),
                patch(
                    "modules.modes.campaign.runtime_campaign_state",
                    return_value=SimpleNamespace(campaign_facts=object()),
                ),
                patch(
                    "modules.modes.campaign.evaluate_battle_entry",
                    return_value=SimpleNamespace(allowed=True),
                ),
                patch(f"modules.modes.campaign.{strategy_name}", return_value=selected) as factory,
            ):
                self.assertIs(mode.on_battle_started(None), selected)
            factory.assert_called_once_with()

    def test_campaign_mode_balances_ordinary_campaign_battles(self):
        from modules.modes.campaign import CampaignProgressionMode

        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        objective = SimpleNamespace(objective_id="reach_petalburg", resource_policy=None)
        mode.controller = SimpleNamespace(last_selection=SimpleNamespace(objective=objective))
        selected = object()
        with (
            patch("modules.modes.campaign.get_party", return_value=()),
            patch(
                "modules.modes.campaign.runtime_campaign_state", return_value=SimpleNamespace(campaign_facts=object())
            ),
            patch("modules.modes.campaign.evaluate_battle_entry", return_value=SimpleNamespace(allowed=True)),
            patch("modules.modes.campaign.NuzlockeLevelBalancingBattleStrategy", return_value=selected) as factory,
        ):
            self.assertIs(mode.on_battle_started(None), selected)
        factory.assert_called_once_with()

    def test_battle_end_requests_a_fresh_readiness_observation(self):
        from modules.modes.campaign import CampaignProgressionMode
        from unittest.mock import Mock

        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode.controller = SimpleNamespace(request_readiness_recheck=Mock())
        mode._readiness_scheduler = SimpleNamespace(invalidate=Mock())

        with patch("modules.modes.campaign.diagnostic_print"):
            mode.on_battle_ended(object())

        mode.controller.request_readiness_recheck.assert_called_once_with("battle_ended")
        mode._readiness_scheduler.invalidate.assert_not_called()

    def test_overworld_execution_cache_retains_tactical_loop_between_observations(self):
        from modules.goals import SemanticTarget
        from modules.nuzlocke import emerald_capabilities as capabilities

        target = SemanticTarget.interaction((1, 3), "rival")
        world = SimpleNamespace(
            map_id=(1, 3),
            player_coordinates=(2, 2),
            controllable=True,
            facing=None,
            tiles=(),
            bindings=(),
            dynamic_blocked_coordinates=frozenset(),
            triggers=(
                SimpleNamespace(
                    trigger_id="object:2:rival",
                    affordance_id="rival",
                    script_symbol="rival",
                    condition_active=True,
                    locations=frozenset({((1, 3), (2, 2))}),
                    activation_locations=frozenset({((1, 3), (2, 2))}),
                ),
            ),
            warps=(),
            objects=(),
        )
        created = []

        class FakeLoop:
            def run(self):
                while True:
                    yield

        def fake_constructor(*args, **kwargs):
            created.append(1)
            return FakeLoop()

        cache = {}
        with (
            patch.object(capabilities, "perceive_overworld", return_value=world),
            patch.object(capabilities, "AgentControlLoop", side_effect=fake_constructor),
        ):
            progression = capabilities.observation_driven_overworld_progression(
                semantic_target=target, execution_cache=cache
            )
            next(progression)
            next(progression)

        self.assertEqual(len(created), 1)

    def test_tactical_loop_failure_releases_cached_execution(self):
        from modules.goals import ReachWarp, SemanticTarget
        from modules.nuzlocke import emerald_capabilities as capabilities

        target = SemanticTarget.map((1, 0))
        world = SimpleNamespace(
            map_id=(1, 1),
            player_coordinates=(7, 2),
            controllable=True,
            facing=None,
            tiles=(),
            bindings=(),
            dynamic_blocked_coordinates=frozenset(),
            triggers=(),
            warps=(
                SimpleNamespace(
                    entry=((1, 1), (7, 1)),
                    destination=((1, 0), (8, 2)),
                ),
            ),
            objects=(),
        )
        created = []

        class FakeLoop:
            def run(self):
                created.append(1)
                yield
                return

        def fake_constructor(*args, **kwargs):
            return FakeLoop()

        cache = {}
        with (
            patch.object(capabilities, "perceive_overworld", return_value=world),
            patch.object(capabilities, "AgentControlLoop", side_effect=fake_constructor),
            patch.object(
                capabilities,
                "_observed_exit_goal",
                return_value=ReachWarp(destination_map=(1, 0), destination=((1, 0), (8, 2)), warp=world.warps[0]),
            ),
        ):
            progression = capabilities.observation_driven_overworld_progression(
                semantic_target=target, execution_cache=cache
            )
            next(progression)
            next(progression)
            next(progression)

        self.assertEqual(len(created), 2)

    def test_blocking_boundary_gate_owns_execution_before_exit_selection(self):
        from modules.goals import ActivateTrigger, SemanticTarget
        from modules.nuzlocke import emerald_capabilities as capabilities
        from modules.overworld import MapConnectionObservation

        target = SemanticTarget.map((1, 0))
        boundary = MapConnectionObservation(
            entry=((1, 1), (4, 0)),
            destination=((1, 0), (0, 0)),
            required_facing=capabilities.Direction.East,
        )
        gate = SimpleNamespace(
            trigger_id="city_gatekeeper",
            locations=frozenset({((1, 1), (3, 0))}),
            activation_locations=frozenset({((1, 1), (2, 0))}),
            condition_active=True,
        )
        world = SimpleNamespace(
            map_id=(1, 1),
            player_coordinates=(2, 0),
            controllable=True,
            facing=None,
            tiles=(),
            bindings=(),
            dynamic_blocked_coordinates=frozenset({(3, 0)}),
            triggers=(gate,),
            transitions=(boundary,),
            warps=(),
            objects=(),
        )
        created = []

        class FakeLoop:
            def run(self):
                while True:
                    yield

        def fake_constructor(*args, **kwargs):
            created.append(kwargs.get("goal"))
            return FakeLoop()

        cache = {}
        with (
            patch.object(capabilities, "perceive_overworld", return_value=world),
            patch.object(capabilities, "AgentControlLoop", side_effect=fake_constructor),
            patch.object(capabilities, "GoalAwareNavigator"),
        ):
            progression = capabilities.observation_driven_overworld_progression(
                semantic_target=target, execution_cache=cache
            )
            next(progression)

        self.assertEqual(created, [ActivateTrigger("city_gatekeeper")])

    def test_petalburg_rom_gym_coordinate_trigger_owns_execution_before_east_exit(self):
        from modules.goals import ActivateTrigger, SemanticTarget
        from modules.nuzlocke import emerald_capabilities as capabilities
        from modules.overworld import MapConnectionObservation, WarpObservation

        target = SemanticTarget.interaction(
            MapRSE.PETALBURG_CITY_GYM.value,
            "PetalburgCity_Gym_EventScript_Norman",
        )
        east = MapConnectionObservation(
            entry=(MapRSE.PETALBURG_CITY.value, (19, 12)),
            destination=(MapRSE.ROUTE102.value, (0, 12)),
            required_facing=capabilities.Direction.East,
        )
        gym = WarpObservation(
            entry=(MapRSE.PETALBURG_CITY.value, (15, 8)),
            destination=(MapRSE.PETALBURG_CITY_GYM.value, (8, 1)),
        )
        gate = SimpleNamespace(
            trigger_id="coord:0:PETALBURG_CITY_STATE:0",
            locations=frozenset({(MapRSE.PETALBURG_CITY.value, (8, 12))}),
            activation_locations=frozenset({(MapRSE.PETALBURG_CITY.value, (8, 12))}),
            script_symbol="PetalburgCity_EventScript_ShowGymToPlayer2",
            affordance_id="PetalburgCity_EventScript_ShowGymToPlayer2",
            condition_active=True,
            currently_actionable=True,
            requires_input=False,
        )
        world = SimpleNamespace(
            map_id=MapRSE.PETALBURG_CITY.value,
            player_coordinates=(8, 12),
            controllable=True,
            facing=None,
            tiles=(),
            bindings=(),
            dynamic_blocked_coordinates=frozenset({(8, 13)}),
            triggers=(gate,),
            transitions=(east, gym),
            warps=(east, gym),
            objects=(),
        )
        created = []

        class FakeLoop:
            def run(self):
                while True:
                    yield

        def fake_constructor(*args, **kwargs):
            created.append(kwargs.get("goal"))
            return FakeLoop()

        cache = {}
        with (
            patch.object(capabilities, "perceive_overworld", return_value=world),
            patch.object(capabilities, "AgentControlLoop", side_effect=fake_constructor),
            patch.object(
                capabilities,
                "_observed_exit_goal",
                side_effect=AssertionError("the east exit must not be selected on the ROM gate tile"),
            ),
        ):
            progression = capabilities.observation_driven_overworld_progression(
                semantic_target=target, execution_cache=cache
            )
            next(progression)

        self.assertEqual(created, [ActivateTrigger(gate.trigger_id)])

    def test_petalburg_rom_gym_coordinate_trigger_requires_known_active_condition(self):
        from modules.nuzlocke import emerald_capabilities as capabilities

        target = SemanticTarget.interaction(
            MapRSE.PETALBURG_CITY_GYM.value,
            "PetalburgCity_Gym_EventScript_Norman",
        )
        gate = SimpleNamespace(
            trigger_id="coord:0:PETALBURG_CITY_STATE:0",
            activation_locations=frozenset({(MapRSE.PETALBURG_CITY.value, (8, 12))}),
            script_symbol="PetalburgCity_EventScript_ShowGymToPlayer2",
            affordance_id="PetalburgCity_EventScript_ShowGymToPlayer2",
            condition_active=None,
            currently_actionable=True,
        )
        world = SimpleNamespace(
            map_id=MapRSE.PETALBURG_CITY.value,
            player_coordinates=(8, 12),
            triggers=(gate,),
        )

        self.assertIsNone(capabilities._observed_petalburg_wally_gate_goal(world, target))

    def test_petalburg_rom_gym_unsuffixed_coordinate_trigger_owns_execution(self):
        from modules.nuzlocke import emerald_capabilities as capabilities

        target = SemanticTarget.interaction(
            MapRSE.PETALBURG_CITY_GYM.value,
            "PetalburgCity_Gym_EventScript_Norman",
        )
        gate = SimpleNamespace(
            trigger_id="coord:4:PETALBURG_CITY_STATE:0",
            activation_locations=frozenset({(MapRSE.PETALBURG_CITY.value, (8, 12))}),
            script_symbol="PetalburgCity_EventScript_ShowGymToPlayer",
            affordance_id="PetalburgCity_EventScript_ShowGymToPlayer",
            condition_active=True,
            currently_actionable=True,
        )
        world = SimpleNamespace(
            map_id=MapRSE.PETALBURG_CITY.value,
            player_coordinates=(8, 12),
            triggers=(gate,),
        )

        self.assertEqual(
            capabilities._observed_petalburg_wally_gate_goal(world, target),
            ActivateTrigger(gate.trigger_id),
        )

    def test_agent_loop_ends_at_terminal_tactical_result(self):
        from modules.agent_control import (
            ActionResult,
            ActionResultType,
            AgentAction,
            AgentActionType,
            AgentControlLoop,
        )
        from modules.navigation import NavigationError

        loop = AgentControlLoop(lambda: None)
        terminal = ActionResult(
            ActionResultType.UNREACHABLE,
            AgentAction(AgentActionType.WAIT_REOBSERVE),
            "warp activation failed",
        )
        with patch.object(loop, "step", side_effect=((None, None, terminal), (None, None, terminal))):
            with self.assertRaises(NavigationError):
                list(loop.run())


if __name__ == "__main__":
    unittest.main()
