import unittest

from modules.nuzlocke.capture_policy import (
    CapturePolicyContext,
    EncounterCandidate,
    EncounterMethod,
    choose_encounter,
)


class CapturePolicyTests(unittest.TestCase):
    def test_default_policy_pursues_a_legal_encounter(self):
        result = choose_encounter(
            (EncounterCandidate((0, 1), EncounterMethod.LAND, ("poochyena",), ("dark",)),),
            CapturePolicyContext(),
        )
        self.assertTrue(result.pursue_now)
        self.assertEqual(result.candidate.area, (0, 1))

    def test_immediate_encounter_is_scored_for_boss_and_team_coverage(self):
        result = choose_encounter(
            (
                EncounterCandidate((0, 1), EncounterMethod.LAND, type_pool=("normal",)),
                EncounterCandidate((0, 2), EncounterMethod.LAND, type_pool=("grass",)),
            ),
            CapturePolicyContext(boss_weakness_types=frozenset({"grass"})),
        )
        self.assertEqual(result.candidate.area, (0, 2))
        self.assertTrue(result.pursue_now)

    def test_superior_reachable_future_method_can_be_deferred_to(self):
        result = choose_encounter(
            (
                EncounterCandidate((0, 1), EncounterMethod.LAND, type_pool=("normal",), revisit_likelihood=1.0),
                EncounterCandidate(
                    (0, 1), EncounterMethod.SURF, type_pool=("water",), future_only=True, future_value=3
                ),
            ),
            CapturePolicyContext(
                boss_weakness_types=frozenset({"water"}),
                available_future_methods=frozenset({EncounterMethod.SURF}),
            ),
        )
        self.assertFalse(result.pursue_now)
        self.assertEqual(result.candidate.method, EncounterMethod.SURF)

    def test_future_opportunity_does_not_override_an_area_unlikely_to_be_revisited(self):
        result = choose_encounter(
            (
                EncounterCandidate((0, 1), EncounterMethod.LAND, type_pool=("normal",), revisit_likelihood=0.0),
                EncounterCandidate(
                    (0, 1), EncounterMethod.SURF, type_pool=("water",), future_only=True, future_value=3
                ),
            ),
            CapturePolicyContext(
                boss_weakness_types=frozenset({"water"}),
                available_future_methods=frozenset({EncounterMethod.SURF}),
            ),
        )
        self.assertTrue(result.pursue_now)
        self.assertEqual(result.candidate.method, EncounterMethod.LAND)


if __name__ == "__main__":
    unittest.main()
