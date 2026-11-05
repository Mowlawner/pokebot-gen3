import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_readiness import BattleReadinessStatus
from modules.nuzlocke.trainer_preparation import (
    TrainerSource,
    build_exp_budget,
    projected_trainer_experience,
    select_trainer_source,
)


class TrainerPreparationTests(unittest.TestCase):
    def setUp(self):
        growth = SimpleNamespace(get_experience_needed_for_level=lambda level: 1000)
        self.party = (
            SimpleNamespace(
                is_egg=False,
                current_hp=20,
                total_hp=20,
                level=4,
                total_exp=100,
                species=SimpleNamespace(level_up_type=growth),
            ),
        )
        self.checkpoint = SimpleNamespace(
            party=(SimpleNamespace(species="Poochyena", level=5),),
        )

    def test_projected_trainer_exp_uses_species_yield_and_multiplier(self):
        species = SimpleNamespace(base_experience_yield=50)
        with patch("modules.nuzlocke.trainer_preparation.get_species_by_name", return_value=species):
            self.assertEqual(projected_trainer_experience(self.checkpoint), 52)

    def test_select_prefers_higher_safe_win_probability(self):
        less_safe = SimpleNamespace(
            status=BattleReadinessStatus.READY,
            win_probability=0.8,
            catastrophic_probability=0.05,
        )
        safer = SimpleNamespace(
            status=BattleReadinessStatus.READY,
            win_probability=0.9,
            catastrophic_probability=0.02,
        )
        sources = (
            TrainerSource("near", self.checkpoint, distance=1, route_simplification=100),
            TrainerSource("safe", self.checkpoint, distance=20, route_simplification=0),
        )
        with patch("modules.nuzlocke.trainer_preparation.get_species_by_name", return_value=SimpleNamespace(base_experience_yield=50)), patch(
            "modules.nuzlocke.trainer_preparation.assess_live_trainer_checkpoint",
            side_effect=(less_safe, safer),
        ):
            selected, evaluations = select_trainer_source(sources, self.party, level_cap=5)

        self.assertEqual(selected.source.trainer_id, "safe")
        self.assertEqual(len(evaluations), 2)

    def test_exp_budget_tracks_unavoidable_reward_and_remaining_deficit(self):
        with patch(
            "modules.nuzlocke.trainer_preparation.get_species_by_name",
            return_value=SimpleNamespace(base_experience_yield=50),
        ):
            budget = build_exp_budget(
                self.party,
                target_level=5,
                level_cap=5,
                unavoidable_checkpoints=(self.checkpoint,),
            )

        self.assertEqual(budget.unavoidable_exp, 52)
        self.assertEqual(budget.required_exp, 900)
        self.assertEqual(budget.remaining_deficit, 848)
        self.assertTrue(budget.mandatory_exp_is_legal)

    def test_rejects_reward_that_cannot_fit_below_hard_cap(self):
        ready = SimpleNamespace(
            status=BattleReadinessStatus.READY,
            win_probability=1.0,
            catastrophic_probability=0.0,
        )
        source = TrainerSource("too_much", self.checkpoint)
        with patch("modules.nuzlocke.trainer_preparation.get_species_by_name", return_value=SimpleNamespace(base_experience_yield=1000)), patch(
            "modules.nuzlocke.trainer_preparation.assess_live_trainer_checkpoint",
            return_value=ready,
        ):
            selected, evaluations = select_trainer_source((source,), self.party, level_cap=5)

        self.assertIsNone(selected)
        self.assertEqual(evaluations[0].cap_safe, False)
        self.assertIn("absorb", evaluations[0].rejection_reason)


if __name__ == "__main__":
    unittest.main()
