import unittest
from dataclasses import replace

from modules.goals import Goal
from modules.nuzlocke.campaign_controller import CampaignController, CampaignControllerStatus
from modules.nuzlocke.campaign_execution import CampaignExecutionResult, CampaignExecutionStatus
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    ObjectiveSelection,
    ObjectiveStatus,
    initial_emerald_campaign,
    select_campaign_objective,
)
from modules.nuzlocke.campaign_state import Fact, RunStatus

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

    def test_ready_intro_battle_mounts_existing_tactical_goal(self):
        controller = self.controller()
        result = controller.refresh()
        self.assertEqual(result.objective_id, "complete_intro_rival")
        self.assertEqual(result.status, CampaignControllerStatus.READY)

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
        self.assertEqual(len(self.created), 1)
        self.assertEqual(self.executed, 1)

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
        result = controller.refresh()
        self.assertNotEqual(old_goal, None)
        self.assertIsNotNone(result.tactical_goal)
        self.assertEqual(result.selection.objective.objective_id, "receive_pokedex")
        self.assertEqual(self.created.__len__(), 1)

    def test_start_nuzlocke_is_internal_and_advances_without_tactical_goal(self):
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
            )
        )
        started = self.fixture.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_available=True,
            nuzlocke_started=True,
        )
        next_objective = CampaignObjective(
            "next_milestone", "next milestone", (), self.fixture.predicate("done", False)
        )
        selections = []

        def select(state):
            if state.campaign_facts.nuzlocke_started.value:
                result = select_campaign_objective(state, (next_objective,))
            else:
                result = select_campaign_objective(state)
            selections.append(result)
            return result

        def mark_boundary(_):
            self.state(campaign_facts=started)
            return True

        controller = CampaignController(
            lambda: self.current,
            selector=select,
            campaign_boundary_handler=mark_boundary,
            adapter=lambda selection: CampaignExecutionResult(
                selection.objective, CampaignExecutionStatus.COMPLETE, "next milestone is outside this test"
            ),
            tactical_loop_factory=self.factory,
        )
        result = controller.refresh()
        self.assertEqual(selections[0].objective.objective_id, "start_nuzlocke")
        self.assertEqual(selections[1].objective.objective_id, "next_milestone")
        self.assertEqual(result.selection.objective.objective_id, "next_milestone")
        self.assertIsNone(result.tactical_goal)
        self.assertEqual(self.created, [])

    def test_start_nuzlocke_is_not_ready_when_pokeballs_are_unavailable(self):
        objective = initial_emerald_campaign()[-1]
        facts = self.fixture.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
        )
        facts = replace(facts, pokeballs_available=Fact.unavailable())
        state = self.fixture.state(campaign_facts=facts)
        selection = select_campaign_objective(state, (objective,))
        self.assertEqual(selection.status, ObjectiveStatus.UNKNOWN)


from modules.map_data import MapRSE

if __name__ == "__main__":
    unittest.main()
