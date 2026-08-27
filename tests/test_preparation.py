import unittest

from modules.nuzlocke.emerald_campaign_registry import emerald_bosses
from modules.nuzlocke.preparation import (
    PreparationInput,
    PreparationStrategy,
    RiskTolerance,
    decide_preparation,
)
from modules.goals import EncounterMode, TrainerMode
from modules.overworld import TriggerObservation
from modules.nuzlocke.preparation import discover_trainer_ids


class PreparationTests(unittest.TestCase):
    boss = emerald_bosses()[0]

    def test_seeded_target_and_area_are_reproducible(self):
        request = PreparationInput(self.boss, (5, 6), wild_areas=((0, 17), (0, 18)), seed=12)

        first = decide_preparation(request)
        second = decide_preparation(request)

        self.assertEqual(first, second)
        self.assertEqual(first.strategy, PreparationStrategy.WILD_FIRST)
        self.assertIn(first.target_level, (7, 8))
        self.assertIn(first.wild_area, request.wild_areas)

    def test_target_never_exceeds_active_cap(self):
        decision = decide_preparation(PreparationInput(self.boss, (5,), active_cap=7, seed=99))

        self.assertLessEqual(decision.target_level, 7)

    def test_no_training_when_party_meets_target(self):
        decision = decide_preparation(PreparationInput(self.boss, (8,), seed=1))

        self.assertFalse(decision.required)
        self.assertIsNone(decision.strategy)

    def test_missing_party_is_unknown(self):
        decision = decide_preparation(PreparationInput(self.boss, (), seed=1))

        self.assertEqual(decision.status.value, "unknown")
        self.assertIn("unavailable", decision.reason)

    def test_no_boss_is_not_a_preparation_requirement(self):
        decision = decide_preparation(PreparationInput(None, (5,), risk_tolerance=RiskTolerance.CONSERVATIVE))

        self.assertFalse(decision.required)
        self.assertIn("no active boss", decision.reason)

    def test_trainer_preparation_selects_target_and_engage_policy(self):
        decision = decide_preparation(PreparationInput(self.boss, (5,), trainer_ids=("trainer-1", "trainer-2"), seed=4))

        self.assertEqual(decision.strategy, PreparationStrategy.TRAINERS_FIRST)
        self.assertIn(decision.trainer_id, ("trainer-1", "trainer-2"))
        self.assertIs(decision.trainer_mode, TrainerMode.ENGAGE)
        self.assertEqual(decision.navigation_goal().target.trainer_id, decision.trainer_id)

    def test_wild_preparation_emits_seek_and_ignore_trainer_policy(self):
        decision = decide_preparation(PreparationInput(self.boss, (5,), wild_areas=((0, 17),), seed=2))
        goal = decision.navigation_goal()

        self.assertIs(decision.encounter_mode, EncounterMode.SEEK)
        self.assertIs(decision.trainer_mode, TrainerMode.IGNORE)
        self.assertIs(goal.encounter_mode, EncounterMode.SEEK)

    def test_discovery_returns_sorted_actionable_trainer_affordances(self):
        observation = type(
            "Observation",
            (),
            {
                "triggers": (
                    TriggerObservation(
                        "b", frozenset(), affordance_id="trainer-b", hazard_locations=frozenset({(0, (1, 1))})
                    ),
                    TriggerObservation(
                        "a", frozenset(), affordance_id="trainer-a", hazard_locations=frozenset({(0, (2, 2))})
                    ),
                    TriggerObservation(
                        "x",
                        frozenset(),
                        affordance_id="not-actionable",
                        hazard_locations=frozenset({(0, (3, 3))}),
                        currently_actionable=False,
                    ),
                )
            },
        )()

        self.assertEqual(discover_trainer_ids(observation), ("trainer-a", "trainer-b"))


if __name__ == "__main__":
    unittest.main()
