import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.goals import ActivateTrigger, EARLY_POKEBALL_TRIGGER_ID, EngageTrainer, GoalConstraints, NavigationGoal, SemanticTargetKind, TrainerMode
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

    def test_pokeball_objective_translates_to_professor_birch_goal(self):
        result = adapt_campaign_execution(self.ready(self.objectives["receive_pokeballs"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "receive_pokeballs")
        self.assertEqual(result.tactical_goal, ActivateTrigger(EARLY_POKEBALL_TRIGGER_ID))
        self.assertIn("Professor Birch", result.reason)

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
            "complete_intro_rival",
            "receive_pokeballs",
        }
        for objective_id, objective in self.objectives.items():
            if objective_id in supported:
                result = adapt_campaign_execution(self.ready(objective))
                self.assertEqual(result.status, CampaignExecutionStatus.READY, objective_id)
                if objective_id not in {"complete_intro_rival", "receive_pokeballs"}:
                    self.assertIsNotNone(result.capability, objective_id)
                continue
            result = adapt_campaign_execution(self.ready(objective))
            self.assertEqual(result.status, CampaignExecutionStatus.UNSUPPORTED, objective_id)
            self.assertEqual(result.execution_id, objective.execution_id, objective_id)

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
        with patch.object(capabilities, "perceive_overworld", return_value=world), patch.object(
            capabilities, "AgentControlLoop", side_effect=fake_constructor
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
        with patch.object(capabilities, "perceive_overworld", return_value=world), patch.object(
            capabilities, "AgentControlLoop", side_effect=fake_constructor
        ), patch.object(
            capabilities,
            "_observed_exit_goal",
            return_value=ReachWarp(destination_map=(1, 0), destination=((1, 0), (8, 2)), warp=world.warps[0]),
        ):
            progression = capabilities.observation_driven_overworld_progression(
                semantic_target=target, execution_cache=cache
            )
            next(progression)
            next(progression)
            next(progression)

        self.assertEqual(len(created), 2)

    def test_agent_loop_ends_at_terminal_tactical_result(self):
        from modules.agent_control import (
            ActionResult,
            ActionResultType,
            AgentAction,
            AgentActionType,
            AgentControlLoop,
        )

        loop = AgentControlLoop(lambda: None)
        terminal = ActionResult(
            ActionResultType.UNREACHABLE,
            AgentAction(AgentActionType.WAIT_REOBSERVE),
            "warp activation failed",
        )
        with patch.object(loop, "step", return_value=(None, None, terminal)):
            with self.assertRaises(StopIteration):
                next(loop.run())


if __name__ == "__main__":
    unittest.main()
