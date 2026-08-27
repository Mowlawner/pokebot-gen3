import unittest

from modules.nuzlocke.campaign_state import CampaignFacts, CampaignState, Fact
from modules.nuzlocke.emerald_campaign_registry import emerald_objectives
from modules.nuzlocke.campaign_objectives import campaign_fact


class EmeraldRegistryLifecycleTests(unittest.TestCase):
    def test_registry_contains_documented_intro_and_gym_milestones(self):
        names = {item.completion_fact for item in emerald_objectives()}
        self.assertTrue({"rival_met", "pokedex_received", "first_badge_obtained"} <= names)

    def test_registered_fact_predicate_delegates_to_campaign_state(self):
        values = [Fact.unavailable() for _ in range(15)]
        values[3] = Fact.known(True)  # rival_met
        facts = CampaignFacts(*values)
        state = CampaignState(
            *(Fact.unknown() for _ in range(14)),
            None,
            (),
            campaign_facts=facts,
        )
        self.assertTrue(campaign_fact("rival_met").evaluate(state).value)


if __name__ == "__main__":
    unittest.main()
