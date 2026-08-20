import unittest

from modules.semantic_choices import (
    ChoiceConsequence,
    ConsequenceKnowledge,
    DialogueChoice,
    choose_choice_for_outcome,
)


class SemanticChoiceTests(unittest.TestCase):
    def choice(self, consequences, selected="NO", options=("YES", "NO")):
        return DialogueChoice("question", options, selected, consequences=tuple(consequences))

    def choose(self, choice, facts=None, desired=None):
        return choose_choice_for_outcome(
            choice, current_facts=facts or {"complete": False}, desired_facts=desired or {"complete": True}
        )

    def test_option_a_completes_objective(self):
        self.assertEqual(
            self.choose(self.choice((ChoiceConsequence("YES", ConsequenceKnowledge.KNOWN, {"complete": True}),))), "YES"
        )

    def test_option_b_completes_objective(self):
        self.assertEqual(
            self.choose(self.choice((ChoiceConsequence("NO", ConsequenceKnowledge.KNOWN, {"complete": True}),))), "NO"
        )

    def test_different_valid_consequences_follow_requested_outcome(self):
        choice = self.choice(
            (
                ChoiceConsequence("YES", ConsequenceKnowledge.KNOWN, {"healed": True}),
                ChoiceConsequence("NO", ConsequenceKnowledge.KNOWN, {"left": True}),
            )
        )
        self.assertEqual(self.choose(choice, {"healed": False, "left": False}, {"left": True}), "NO")

    def test_unknown_consequence_is_not_guessed(self):
        self.assertIsNone(self.choose(self.choice((ChoiceConsequence("YES"),))))

    def test_no_known_consequence_is_not_guessed(self):
        choice = self.choice((ChoiceConsequence("YES", ConsequenceKnowledge.PARTIALLY_KNOWN), ChoiceConsequence("NO")))
        self.assertIsNone(self.choose(choice))

    def test_selected_desired_option_is_confirmed_without_movement(self):
        choice = self.choice(
            (ChoiceConsequence("YES", ConsequenceKnowledge.KNOWN, {"complete": True}),), selected="YES"
        )
        self.assertEqual(self.choose(choice), "YES")

    def test_unavailable_desired_option_is_not_fabricated(self):
        choice = self.choice(
            (ChoiceConsequence("NO", ConsequenceKnowledge.KNOWN, {"complete": False}),), options=("NO",)
        )
        self.assertIsNone(self.choose(choice))

    def test_completed_objective_requires_no_choice(self):
        choice = self.choice((ChoiceConsequence("YES", ConsequenceKnowledge.KNOWN, {"complete": True}),))
        self.assertIsNone(self.choose(choice, {"complete": True}))


if __name__ == "__main__":
    unittest.main()
