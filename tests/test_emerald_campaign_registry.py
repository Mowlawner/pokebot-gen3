import unittest

from modules.nuzlocke.campaign_state import CampaignFacts, Fact, FactStatus
from modules.nuzlocke.emerald_campaign_registry import active_emerald_boss, emerald_bosses


def facts(*, first_badge=Fact.known(False)):
    values = [Fact.unavailable() for _ in range(15)]
    values[-1] = first_badge
    return CampaignFacts(*values)


class EmeraldCampaignRegistryTests(unittest.TestCase):
    def test_implemented_boss_data_is_ordered_and_rom_owned(self):
        bosses = emerald_bosses()

        self.assertEqual(len(bosses), 1)
        self.assertEqual(bosses[0].boss_id, "roxanne")
        self.assertEqual(bosses[0].level_cap, 15)

    def test_active_boss_uses_authoritative_completion_fact(self):
        progression = active_emerald_boss(facts())

        self.assertEqual(progression.status, FactStatus.KNOWN)
        self.assertEqual(progression.active_boss.boss_id, "roxanne")

    def test_unavailable_completion_does_not_guess_boss_progression(self):
        progression = active_emerald_boss(facts(first_badge=Fact.unavailable()))

        self.assertEqual(progression.status, FactStatus.UNAVAILABLE)
        self.assertIsNone(progression.active_boss)

    def test_completed_implemented_slice_has_no_active_boss(self):
        progression = active_emerald_boss(facts(first_badge=Fact.known(True)))

        self.assertEqual(progression.status, FactStatus.KNOWN)
        self.assertIsNone(progression.active_boss)


if __name__ == "__main__":
    unittest.main()
