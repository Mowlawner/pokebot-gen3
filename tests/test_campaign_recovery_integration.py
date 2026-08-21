import unittest
from types import SimpleNamespace

from modules.goals import Goal
from modules.nuzlocke.campaign_controller import CampaignController, CampaignControllerStatus
from modules.nuzlocke.campaign_execution import CampaignExecutionResult, CampaignExecutionStatus
from modules.nuzlocke.campaign_objectives import CampaignObjective, ObjectiveSelection, ObjectiveStatus
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    ProgressionReadinessDiagnostic,
    ReadinessDecision,
    ReadinessReason,
    TrainerHazardObservation,
)


def readiness(decision=ReadinessDecision.RECOVER):
    trainer_availability = Availability.UNAVAILABLE if decision is ReadinessDecision.UNKNOWN else Availability.KNOWN
    return ProgressionReadinessDiagnostic(
        "reach_generic",
        "ready",
        None,
        Goal(),
        (1, 2),
        (3, 4),
        "OVERWORLD",
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
        decision,
        (
            ReadinessReason.TRAINER_HAZARD_UNKNOWN
            if decision is ReadinessDecision.UNKNOWN
            else ReadinessReason.CRITICAL_PARTY_HP_BEFORE_TRAINER
        ),
    )


class CampaignRecoveryIntegrationTests(unittest.TestCase):
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

    def test_unknown_blocks_without_recovery_or_tactical_execution(self):
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
        self.assertEqual(result.status, CampaignControllerStatus.BLOCKED)
        self.assertIn("TRAINER_HAZARD_UNKNOWN", result.reason)
        self.assertEqual(starts, [])


if __name__ == "__main__":
    unittest.main()
