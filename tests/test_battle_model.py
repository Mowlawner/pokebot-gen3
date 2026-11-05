import unittest

from modules.battle_model import move_outcomes


class MoveOutcomeTests(unittest.TestCase):
    def test_probabilities_always_sum_to_one(self):
        for args in ((3, 4, 1.0), (5, 7, 0.95), (10, 10, 0.7), (1, 20, 1.0), (0, 0, 1.0)):
            with self.subTest(args=args):
                total = sum(outcome.probability for outcome in move_outcomes(*args))
                self.assertAlmostEqual(total, 1.0)

    def test_a_status_move_is_one_zero_damage_outcome(self):
        outcomes = move_outcomes(0, 0, 1.0)
        self.assertEqual([(o.probability, o.damage) for o in outcomes], [(1.0, 0)])

    def test_misses_deal_no_damage_and_scale_with_accuracy(self):
        misses = [o for o in move_outcomes(4, 6, 0.75) if o.damage == 0]
        self.assertEqual(len(misses), 1)
        self.assertAlmostEqual(misses[0].probability, 0.25)

    def test_damage_rolls_span_the_range_and_a_crit_exceeds_the_maximum(self):
        outcomes = move_outcomes(5, 7, 1.0, critical_chance=0.1)
        normal = sorted({o.damage for o in outcomes if not o.critical})
        critical = [o for o in outcomes if o.critical]
        self.assertEqual(normal, [5, 6, 7])
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0].damage, 14)  # double the maximum by default
        self.assertAlmostEqual(critical[0].probability, 0.1)

    def test_an_explicit_critical_damage_is_used_and_never_below_the_maximum(self):
        assert [o.damage for o in move_outcomes(5, 7, 1.0, critical_damage=11) if o.critical] == [11]
        assert [o.damage for o in move_outcomes(5, 7, 1.0, critical_damage=2) if o.critical] == [7]

    def test_fixed_damage_collapses_to_a_single_roll(self):
        normal = [o for o in move_outcomes(6, 6, 1.0, critical_chance=0.0) if not o.critical]
        self.assertEqual([o.damage for o in normal], [6])

    def test_out_of_range_inputs_are_clamped(self):
        outcomes = move_outcomes(9, 4, 5.0, critical_chance=-1.0)
        self.assertAlmostEqual(sum(o.probability for o in outcomes), 1.0)
        self.assertTrue(all(o.damage >= 0 for o in outcomes))


if __name__ == "__main__":
    unittest.main()
