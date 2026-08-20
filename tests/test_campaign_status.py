import unittest

from modules.goals import SemanticTarget
from modules.nuzlocke.campaign_status import CampaignStatus, format_campaign_status


class CampaignStatusTests(unittest.TestCase):
    def test_meaningful_target_is_visible(self):
        text = format_campaign_status(
            CampaignStatus("Meet Rival", SemanticTarget.map((0, 16)), "Navigate")
        )
        self.assertIn("Objective: Meet Rival", text)
        self.assertIn("Target: map (0, 16)", text)
        self.assertIn("Intent: Navigate", text)

    def test_targetless_state_is_explicit(self):
        text = format_campaign_status(CampaignStatus("Meet Rival", None, "Targetless fallback"))
        self.assertIn("Target: NONE", text)
        self.assertIn("Targetless fallback", text)

    def test_no_objective_and_complete_are_explicit(self):
        self.assertEqual(
            format_campaign_status(None),
            "Objective: NONE\nTarget: NONE\nIntent: No semantic objective",
        )
        self.assertEqual(format_campaign_status(CampaignStatus.complete_status()), "Campaign: COMPLETE")


if __name__ == "__main__":
    unittest.main()
