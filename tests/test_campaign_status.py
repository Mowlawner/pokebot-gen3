import unittest

from modules.goals import SemanticTarget
from modules.nuzlocke.campaign_status import CampaignStatus, format_campaign_status, recovery_status


class CampaignStatusTests(unittest.TestCase):
    def test_meaningful_target_is_visible(self):
        text = format_campaign_status(CampaignStatus("Meet Rival", SemanticTarget.map((0, 16)), "Navigate"))
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

    def test_recovery_uses_the_campaign_projection_format(self):
        text = format_campaign_status(recovery_status(SemanticTarget.map((1, 2)), "Navigate to nurse"))
        self.assertEqual(
            text,
            "Objective: Recover Party\nTarget: map (1, 2) (Littleroot Town Mays House 1F)\nIntent: Navigate to nurse",
        )


if __name__ == "__main__":
    unittest.main()
