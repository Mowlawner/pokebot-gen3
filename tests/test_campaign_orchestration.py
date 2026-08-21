import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.modes.campaign import CampaignProgressionMode
from modules.nuzlocke.campaign_controller import CampaignController
from modules.nuzlocke.campaign_execution import CampaignExecutionResult, CampaignExecutionStatus
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    ObjectiveSelection,
    ObjectiveStatus,
)
from modules.nuzlocke.campaign_state import Fact


class CampaignOrchestrationTests(unittest.TestCase):
    def test_controller_is_constructed_during_mode_initialization(self):
        with patch("modules.modes.campaign.CampaignController") as controller_factory:
            mode = CampaignProgressionMode()
        controller_factory.assert_called_once()
        self.assertIs(mode.controller, controller_factory.return_value)

    def test_run_delegates_startup_observation_to_controller(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        fake_controller = SimpleNamespace(
            step=lambda: SimpleNamespace(
                status=SimpleNamespace(value="ready"),
                objective_id="set_text_speed",
                reason="resolved executable frontier",
            )
        )
        mode.controller = fake_controller
        mode._readiness_evaluated = False

        loop = mode.run()
        next(loop)
        self.assertEqual(mode.controller, fake_controller)

    def test_run_retries_transient_boot_observation_without_releasing_controller(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        statuses = iter(("unknown", "ready"))
        steps = []

        def step():
            status = next(statuses)
            steps.append(status)
            return SimpleNamespace(
                status=SimpleNamespace(value=status),
                objective_id=None if status == "unknown" else "set_text_speed",
                reason="campaign observation unavailable" if status == "unknown" else "ready",
            )

        controller = SimpleNamespace(step=step)
        mode.controller = controller
        mode._readiness_evaluated = False
        loop = mode.run()
        next(loop)
        next(loop)
        self.assertEqual(steps, ["unknown", "ready"])
        self.assertIs(mode.controller, controller)

    def test_battle_callback_remains_execution_policy(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_scheduler = SimpleNamespace(invalidate=lambda *_: None)
        from modules.modes import BattleAction

        self.assertIs(mode.on_battle_started(None), BattleAction.Fight)

    def test_startup_capability_is_not_blocked_by_overworld_readiness(self):
        objective = CampaignObjective(
            "set_text_speed",
            "set text speed",
            (),
            CampaignPredicate("text_speed_fast", "text speed is fast", lambda _: Fact.known(False)),
        )
        advanced = []

        def capability():
            advanced.append("first startup executor step")
            yield

        controller = CampaignController(
            lambda: object(),
            selector=lambda _: ObjectiveSelection(objective, ObjectiveStatus.READY, "startup"),
            adapter=lambda _: CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "startup capability",
                capability=capability,
            ),
            readiness_provider=lambda *_: object(),
        )
        with patch(
            "modules.nuzlocke.campaign_controller.evaluate_progression_readiness",
            side_effect=AssertionError("startup capability must not require overworld readiness"),
        ):
            state = controller.step()

        self.assertEqual(state.status.value, "ready")
        self.assertEqual(advanced, ["first startup executor step"])


if __name__ == "__main__":
    unittest.main()
