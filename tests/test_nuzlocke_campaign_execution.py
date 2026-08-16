import unittest

from modules.goals import ActivateTrigger, ReachWarp, EARLY_POKEBALL_TRIGGER_ID
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

    def test_reach_oldale_translates_to_existing_navigation_goal(self):
        result = adapt_campaign_execution(self.ready(self.objectives["reach_oldale"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "reach_oldale")
        self.assertEqual(result.tactical_goal, ReachWarp(destination_map=MapRSE.OLDALE_TOWN.value))

    def test_introductory_rival_translates_to_existing_trigger_goal(self):
        result = adapt_campaign_execution(self.ready(self.objectives["complete_intro_rival"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "intro_rival")
        self.assertEqual(result.tactical_goal, ActivateTrigger("introductory_rival"))

    def test_pokeball_objective_translates_to_professor_birch_goal(self):
        result = adapt_campaign_execution(self.ready(self.objectives["confirm_early_pokeballs"]))
        self.assertEqual(result.status, CampaignExecutionStatus.READY)
        self.assertEqual(result.execution_id, "obtain_early_pokeballs")
        self.assertEqual(result.tactical_goal, ActivateTrigger(EARLY_POKEBALL_TRIGGER_ID))
        self.assertIn("Professor Birch", result.reason)

    def test_selection_statuses_are_preserved_without_translation(self):
        objective = self.objectives["reach_oldale"]
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
            "reach_oldale",
            "Oldale",
            (),
            self.objectives["reach_oldale"].completion,
            execution_id="reach_oldale",
            tactical_target=ActivateTrigger("wrong"),
        )
        result = adapt_campaign_execution(self.ready(objective))
        self.assertEqual(result.status, CampaignExecutionStatus.UNSUPPORTED)
        self.assertIn("valid Oldale navigation goal", result.reason)

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
        result = adapt_campaign_execution(self.ready(self.objectives["reach_oldale"]))
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
