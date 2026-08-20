import unittest
from unittest.mock import patch

from modules.goals import ActivateTrigger
from modules.nuzlocke.resource_policy import PartyResource, ResourceDecision, ResourceSnapshot
from modules.nuzlocke.resource_runtime import CampaignCapability


class CampaignCapabilityTests(unittest.TestCase):
    def test_continue_delegates_original_tactical_goal_without_center_lookup(self):
        goal = ActivateTrigger("introductory_rival")
        delegated = []

        def delegate(received_goal):
            delegated.append(received_goal)
            yield "tactical"

        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "none", False),),
            bag_healing_items=(),
            pc_healing_items=(),
        )
        capability = CampaignCapability("complete_intro_rival", object(), goal, delegate)

        with patch("modules.nuzlocke.resource_runtime.observe_resource_snapshot", return_value=snapshot), patch(
            "modules.nuzlocke.resource_runtime.assess_campaign_resources",
            return_value=ResourceDecision.CONTINUE,
        ), patch("modules.nuzlocke.resource_runtime.observe_route_recovery") as find_center:
            self.assertEqual(next(capability()), "tactical")

        self.assertEqual(delegated, [goal])
        find_center.assert_not_called()

    def test_recovery_then_continue_delegates_same_tactical_goal(self):
        goal = ActivateTrigger("introductory_rival")
        delegated = []

        def delegate(received_goal):
            delegated.append(received_goal)
            yield "tactical"

        snapshot = ResourceSnapshot(
            party=(PartyResource(5, 20, "none", False),),
            bag_healing_items=(),
            pc_healing_items=(),
        )
        capability = CampaignCapability("complete_intro_rival", object(), goal, delegate)
        decisions = iter((ResourceDecision.PRESERVE_RESOURCES, ResourceDecision.RECOVER_AT_CENTER))
        route = type("Route", (), {"center_available": True})()

        def recovery():
            yield "recovery"

        with patch("modules.nuzlocke.resource_runtime.observe_resource_snapshot", return_value=snapshot), patch(
            "modules.nuzlocke.resource_runtime.assess_campaign_resources",
            side_effect=lambda *args: next(decisions),
        ), patch("modules.nuzlocke.resource_runtime.observe_route_recovery", return_value=route), patch(
            "modules.nuzlocke.resource_runtime.recover_at_nearest_center", side_effect=recovery
        ):
            self.assertEqual(list(capability()), ["recovery", "tactical"])

        self.assertEqual(delegated, [goal])


if __name__ == "__main__":
    unittest.main()
