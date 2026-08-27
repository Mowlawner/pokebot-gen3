import unittest

from types import SimpleNamespace

from modules.nuzlocke.campaign_state import CampaignFacts, Fact, FactStatus
from modules.nuzlocke.level_cap import assess_level_cap, evaluate_battle_entry
from modules.nuzlocke.rule_config import CampaignRulesConfig


def facts(first_badge=False):
    values = [Fact.unavailable() for _ in range(15)]
    values[-1] = Fact.known(first_badge)
    return CampaignFacts(*values)


class LevelCapTests(unittest.TestCase):
    def test_roxanne_cap_rejects_over_cap_party_member(self):
        assessment = assess_level_cap(
            facts(), [SimpleNamespace(level=16, party_index=2)], objective_id="defeat_roxanne"
        )

        self.assertTrue(assessment.active_boss)
        self.assertEqual(assessment.level_cap, 15)
        self.assertEqual(assessment.illegal_party_indices, (2,))
        self.assertFalse(assessment.legal)

    def test_active_cap_applies_to_any_campaign_battle_entry(self):
        decision = evaluate_battle_entry(facts(), [SimpleNamespace(level=99)], objective_id="reach_rustboro")

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.assessment.status, FactStatus.KNOWN)

    def test_unknown_completion_is_conservative(self):
        values = [Fact.unavailable() for _ in range(15)]
        decision = evaluate_battle_entry(
            CampaignFacts(*values), [SimpleNamespace(level=1)], objective_id="defeat_roxanne"
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.assessment.status, FactStatus.UNAVAILABLE)

    def test_defeated_roxanne_advances_past_implemented_cap(self):
        decision = evaluate_battle_entry(facts(True), [SimpleNamespace(level=99)], objective_id="defeat_roxanne")

        self.assertTrue(decision.allowed)
        self.assertIsNone(decision.assessment.active_boss)

    def test_disabled_level_cap_allows_over_cap_party(self):
        decision = evaluate_battle_entry(
            facts(),
            [SimpleNamespace(level=99)],
            objective_id="defeat_roxanne",
            rule_config=CampaignRulesConfig.unrestricted(),
        )

        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
