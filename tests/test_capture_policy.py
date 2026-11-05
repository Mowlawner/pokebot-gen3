import unittest

from modules.nuzlocke.capture_policy import (
    CapturePolicyContext,
    EncounterPortfolioContext,
    EncounterCandidate,
    EncounterMethod,
    choose_encounter_portfolio,
    choose_encounter,
    conditioned_pool_certainty,
    conditioned_species_rates,
    score_encounter,
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

    def test_species_clause_conditions_common_duplicate_outcomes(self):
        candidate = EncounterCandidate(
            (0, 18),
            EncounterMethod.LAND,
            species_pool=("Poochyena", "Ralts"),
            type_pool=("dark", "psychic"),
            species_rates=(("Poochyena", 70), ("Ralts", 30)),
            species_types=(("Poochyena", ("dark",)), ("Ralts", ("psychic",))),
        )
        context = CapturePolicyContext(
            boss_weakness_types=frozenset({"psychic"}),
            species_clause_enabled=True,
            captured_species=frozenset({"Poochyena"}),
            open_party_slots=1,
        )

        self.assertEqual(conditioned_species_rates(candidate, context), (("Ralts", 1.0),))
        self.assertGreater(score_encounter(candidate, context), 0.0)

    def test_full_party_removes_capacity_value_from_encounter_score(self):
        candidate = EncounterCandidate(
            (0, 17),
            EncounterMethod.LAND,
            species_pool=("Poochyena",),
            type_pool=("dark",),
        )
        empty_slot = CapturePolicyContext(open_party_slots=1)
        full_party = CapturePolicyContext(open_party_slots=0)

        self.assertGreater(score_encounter(candidate, empty_slot), score_encounter(candidate, full_party))

    def test_species_clause_certainty_values_a_guaranteed_remaining_species(self):
        guaranteed = EncounterCandidate(
            (0, 16),
            EncounterMethod.LAND,
            species_pool=("Poochyena", "Wurmple"),
            species_rates=(("Poochyena", 45), ("Wurmple", 45)),
            species_types=(("Poochyena", ("dark",)), ("Wurmple", ("bug",))),
        )
        mixed = EncounterCandidate(
            (0, 18),
            EncounterMethod.LAND,
            species_pool=("Poochyena", "Wurmple", "Ralts"),
            species_rates=(("Poochyena", 45), ("Wurmple", 10), ("Ralts", 45)),
        )
        context = CapturePolicyContext(
            species_clause_enabled=True,
            captured_species=frozenset({"Poochyena", "Zigzagoon"}),
            open_party_slots=1,
        )

        self.assertEqual(conditioned_pool_certainty(guaranteed, context), 1.0)
        self.assertLess(conditioned_pool_certainty(mixed, context), 1.0)
        self.assertGreater(score_encounter(guaranteed, context), score_encounter(mixed, context))

    def test_guaranteed_species_clause_result_can_justify_a_bounded_detour(self):
        guaranteed = EncounterCandidate(
            (0, 16),
            EncounterMethod.LAND,
            species_pool=("Poochyena", "Wurmple"),
            species_rates=(("Poochyena", 70), ("Wurmple", 30)),
            species_types=(("Poochyena", ("dark",)), ("Wurmple", ("bug",))),
            detour_cost=60,
        )
        nearby = EncounterCandidate(
            (0, 18),
            EncounterMethod.LAND,
            species_pool=("Ralts", "Zigzagoon"),
            species_rates=(("Ralts", 50), ("Zigzagoon", 50)),
            detour_cost=0,
        )
        context = CapturePolicyContext(
            species_clause_enabled=True,
            captured_species=frozenset({"Poochyena"}),
            open_party_slots=1,
        )

        self.assertEqual(conditioned_pool_certainty(guaranteed, context), 1.0)
        self.assertGreater(score_encounter(guaranteed, context), score_encounter(nearby, context))

    def test_portfolio_preserves_full_order_when_capacity_bounds_selection(self):
        candidates = (
            EncounterCandidate((0, 1), EncounterMethod.LAND, species_pool=("Ralts",), future_value=3),
            EncounterCandidate((0, 2), EncounterMethod.LAND, species_pool=("Wurmple",), future_value=2),
            EncounterCandidate((0, 3), EncounterMethod.LAND, species_pool=("Zigzagoon",), future_value=1),
        )

        result = choose_encounter_portfolio(
            candidates,
            EncounterPortfolioContext(
                capture=CapturePolicyContext(open_party_slots=1),
                max_encounters=1,
            ),
        )

        self.assertEqual(result.candidates, (candidates[0],))
        self.assertEqual(result.ordered_candidates, candidates)


if __name__ == "__main__":
    unittest.main()
