import unittest
from types import SimpleNamespace

from modules.goals import Goal
from modules.nuzlocke.campaign_controller import CampaignController, CampaignControllerStatus
from modules.nuzlocke.campaign_execution import CampaignExecutionResult, CampaignExecutionStatus
from modules.nuzlocke.campaign_objectives import CampaignObjective, ObjectiveSelection, ObjectiveStatus
from modules.nuzlocke.campaign_state import Fact, FactStatus
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    ProgressionReadinessDiagnostic,
    ReadinessDecision,
    ReadinessReason,
    TrainerHazardObservation,
)


def readiness(decision=ReadinessDecision.RECOVER, game_state="OVERWORLD"):
    trainer_availability = Availability.UNAVAILABLE if decision is ReadinessDecision.UNKNOWN else Availability.KNOWN
    return ProgressionReadinessDiagnostic(
        "reach_generic",
        "ready",
        None,
        Goal(),
        (1, 2),
        (3, 4),
        game_state,
        "test",
        (SimpleNamespace(usable=True, hp_ratio=1.0 if decision is ReadinessDecision.CONTINUE else 0.1, fainted=False),),
        True,
        None,
        None,
        (TrainerHazardObservation(1, "Normal", 4, False, 3, "nearby", None),),
        "NOT_EVALUATED",
        Availability.KNOWN,
        trainer_availability,
        Availability.KNOWN,
        Availability.KNOWN,
        overworld_availability=(Availability.UNKNOWN if decision is ReadinessDecision.UNKNOWN else Availability.KNOWN),
        readiness_decision=decision,
        readiness_reason=(
            ReadinessReason.OVERWORLD_UNAVAILABLE
            if decision is ReadinessDecision.UNKNOWN
            else ReadinessReason.CRITICAL_PARTY_HP
        ),
    )


class CampaignRecoveryIntegrationTests(unittest.TestCase):
    def _safety_fixture(self, restored=False):
        state = SimpleNamespace(party=Fact.known(SimpleNamespace(restored=restored)))
        parent = CampaignObjective(
            "complete_intro_rival", "parent", (), SimpleNamespace(evaluate=lambda _: Fact.known(False))
        )
        safety = CampaignObjective(
            "SAFETY",
            "safety",
            (),
            SimpleNamespace(evaluate=lambda value: Fact.known(value.party.value.restored)),
            task_kind="safety",
        )
        parent_selection = ObjectiveSelection(parent, ObjectiveStatus.READY, "parent")
        safety_selection = ObjectiveSelection(safety, ObjectiveStatus.READY, "safety")
        steps = []

        def adapter(selection):
            if selection.objective is safety:
                return CampaignExecutionResult(
                    safety, CampaignExecutionStatus.READY, "safety", capability=lambda: iter((steps.append("safety"),))
                )
            return CampaignExecutionResult(parent, CampaignExecutionStatus.READY, "parent", tactical_goal=Goal())

        controller = CampaignController(
            lambda: state,
            selector=lambda _: parent_selection,
            adapter=adapter,
            tactical_loop_factory=lambda _: iter((steps.append("parent"),)),
        )
        controller.last_selection = safety_selection
        controller.current_objective_id = safety.objective_id
        controller._tactical_loop = None
        controller._tactical_loop_objective_id = safety.objective_id
        controller.interrupted_objective_id = parent.objective_id
        return controller, state, steps

    def test_active_safety_objective_cannot_be_replaced(self):
        controller, state, steps = self._safety_fixture()
        controller.step()
        self.assertEqual(controller.current_objective_id, "SAFETY")
        self.assertEqual(controller.interrupted_objective_id, "complete_intro_rival")
        self.assertEqual(steps, ["safety"])

    def test_completed_safety_objective_releases_parent(self):
        controller, state, steps = self._safety_fixture(restored=True)
        controller.step()
        self.assertEqual(controller.current_objective_id, "complete_intro_rival")
        self.assertIsNone(controller.interrupted_objective_id)
        self.assertEqual(steps, ["parent"])

    def test_recovery_suspends_and_resumes_same_objective_once(self):
        objective = CampaignObjective("reach_generic", "generic", (), SimpleNamespace(evaluate=lambda _: None))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        goal = Goal()
        execution = CampaignExecutionResult(objective, CampaignExecutionStatus.READY, "ready", tactical_goal=goal)
        recovery_starts = []
        tactical_starts = []

        def recovery(_):
            recovery_starts.append(1)
            yield "recovery"

        def tactical(_):
            tactical_starts.append(1)
            yield "campaign"

        controller = CampaignController(
            lambda: object(),
            selector=lambda _: selection,
            adapter=lambda _: execution,
            tactical_loop_factory=tactical,
            readiness_provider=lambda *_: readiness(),
            recovery_factory=recovery,
        )
        self.assertEqual(controller.step().execution_phase, "RECOVERY")
        self.assertEqual(len(recovery_starts), 1)
        self.assertEqual(controller.current_objective_id, "reach_generic")
        controller._readiness_provider = lambda *_: readiness(ReadinessDecision.CONTINUE)
        controller.step()  # recovery completes
        self.assertEqual(len(recovery_starts), 1)
        self.assertEqual(controller.state.recovery_status, "COMPLETED")
        controller.step()  # remounts the original campaign objective
        self.assertEqual(len(tactical_starts), 1)
        self.assertEqual(controller.current_objective_id, "reach_generic")

    def test_unknown_defers_without_recovery_or_losing_campaign_ownership(self):
        objective = CampaignObjective("reach_generic", "generic", (), SimpleNamespace(evaluate=lambda _: None))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        execution = CampaignExecutionResult(objective, CampaignExecutionStatus.READY, "ready", tactical_goal=Goal())
        starts = []
        controller = CampaignController(
            lambda: object(),
            selector=lambda _: selection,
            adapter=lambda _: execution,
            tactical_loop_factory=lambda _: iter((starts.append("campaign"),)),
            readiness_provider=lambda *_: readiness(ReadinessDecision.UNKNOWN),
            recovery_factory=lambda _: iter((starts.append("recovery"),)),
        )
        result = controller.step()
        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.execution_phase, "CAMPAIGN")
        self.assertIn("OVERWORLD_UNAVAILABLE", result.reason)
        self.assertEqual(starts, [])
        controller._readiness_provider = lambda *_: readiness(ReadinessDecision.CONTINUE)
        controller.step()
        self.assertEqual(starts, ["campaign"])

    def test_critical_unknown_game_state_does_not_mount_recovery(self):
        objective = CampaignObjective("reach_generic", "generic", (), SimpleNamespace(evaluate=lambda _: None))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        execution = CampaignExecutionResult(objective, CampaignExecutionStatus.READY, "ready", tactical_goal=Goal())
        starts = []
        controller = CampaignController(
            lambda: object(),
            selector=lambda _: selection,
            adapter=lambda _: execution,
            tactical_loop_factory=lambda _: iter((starts.append("campaign"),)),
            readiness_provider=lambda *_: readiness(ReadinessDecision.RECOVER, game_state="UNKNOWN"),
            recovery_factory=lambda _: iter((starts.append("recovery"),)),
        )
        result = controller.step()
        self.assertEqual(result.execution_phase, "CAMPAIGN")
        self.assertIn("OVERWORLD_UNAVAILABLE", result.reason)
        self.assertEqual(starts, [])


if __name__ == "__main__":
    unittest.main()
