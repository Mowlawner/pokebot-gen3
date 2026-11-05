import unittest

from modules.battle_kind import BattleKind, classify_battle


class BattleKindTests(unittest.TestCase):
    def kind(self, **facts):
        defaults = dict(objective="defeat", is_trainer=False)
        defaults.update(facts)
        return classify_battle(**defaults).kind

    def test_first_battle_is_a_scripted_defeat_even_when_capture_looks_available(self):
        self.assertIs(
            self.kind(objective="capture", first_battle=True, capture_target=True, capture_available=True),
            BattleKind.SCRIPTED_DEFEAT,
        )

    def test_trainer_battles_are_never_captures(self):
        self.assertIs(self.kind(is_trainer=True, objective="capture"), BattleKind.TRAINER)

    def test_capture_objective_with_a_ball_is_a_wild_capture(self):
        result = classify_battle(
            objective="capture", is_trainer=False, capture_target=True, capture_available=True
        )
        self.assertIs(result.kind, BattleKind.WILD_CAPTURE)
        self.assertIn("ball", result.reason)

    def test_capture_target_without_a_ball_is_a_defeat_with_that_reason(self):
        result = classify_battle(
            objective="defeat", is_trainer=False, capture_target=True, capture_available=False
        )
        self.assertIs(result.kind, BattleKind.WILD_DEFEAT)
        self.assertIn("no usable ball", result.reason)

    def test_avoid_and_transit_intents_flee(self):
        self.assertIs(self.kind(objective="avoid"), BattleKind.WILD_FLEE)
        self.assertIs(self.kind(objective="transit"), BattleKind.WILD_FLEE)

    def test_preparation_is_wild_training(self):
        self.assertIs(self.kind(objective="training"), BattleKind.WILD_TRAINING)

    def test_unrecognized_wild_objectives_default_to_defeat(self):
        self.assertIs(self.kind(objective="auto", intent="auto"), BattleKind.WILD_DEFEAT)
        self.assertIs(self.kind(objective=None), BattleKind.WILD_DEFEAT)


if __name__ == "__main__":
    unittest.main()


class BattleKindLoggingTests(unittest.TestCase):
    def setUp(self):
        from modules import battle_planner

        self.module = battle_planner
        self.module._LAST_BATTLE_KIND_LOG = None

    def tearDown(self):
        self.module._LAST_BATTLE_KIND_LOG = None

    def _context(self, **changes):
        from types import SimpleNamespace

        values = dict(
            battle_objective="capture",
            capture_target=True,
            capture_available=True,
            opponent_name="Wurmple",
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_the_kind_is_logged_once_per_battle_even_across_turns(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        logs = []
        battle = SimpleNamespace(is_trainer_battle=False, type=0)
        knowledge = SimpleNamespace(battle_id="battle-1")
        with patch.object(self.module, "print_always", side_effect=lambda m: logs.append(m())):
            for _ in range(3):
                self.module._log_battle_kind(battle, knowledge, self._context())

        self.assertEqual(len(logs), 1)
        self.assertIn("BATTLE_KIND", logs[0])
        self.assertIn("wild_capture", logs[0])
        self.assertIn("Wurmple", logs[0])

    def test_a_change_of_kind_or_battle_is_logged_again(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        logs = []
        battle = SimpleNamespace(is_trainer_battle=False, type=0)
        with patch.object(self.module, "print_always", side_effect=lambda m: logs.append(m())):
            self.module._log_battle_kind(battle, SimpleNamespace(battle_id="a"), self._context())
            # The same battle loses its last ball and stops being a capture.
            self.module._log_battle_kind(
                battle,
                SimpleNamespace(battle_id="a"),
                self._context(battle_objective="defeat", capture_available=False),
            )
            # A new battle with the same facts is a new record.
            self.module._log_battle_kind(battle, SimpleNamespace(battle_id="b"), self._context())

        self.assertEqual(len(logs), 3)
        self.assertIn("wild_defeat", logs[1])
        self.assertIn("no usable ball", logs[1])
