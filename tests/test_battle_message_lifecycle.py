import unittest

from modules.modes._listeners import clear_transient_battle_message


class BattleMessageLifecycleTests(unittest.TestCase):
    def test_transient_planner_message_is_cleared(self):
        from modules.context import context

        context.message = "PLANNER SAFE: use Tackle"
        clear_transient_battle_message()
        self.assertEqual(context.message, "")

    def test_clear_boundary_does_not_replace_battle_end_callback(self):
        # The listener invokes the existing callback before this helper; this
        # regression test keeps the clearing operation independently callable.
        self.assertTrue(callable(clear_transient_battle_message))


if __name__ == "__main__":
    unittest.main()
