import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.modes._interface import BotModeError
from modules.memory import GameState


class EmeraldFieldItemContextTests(unittest.TestCase):
    def test_field_item_context_accepts_all_emerald_task_variants(self):
        from modules.modes.util import items

        for task_name in items._EMERALD_FIELD_ITEM_CONTEXT_TASKS:
            with self.subTest(task_name=task_name):
                active = True

                def task_is_active(name):
                    return active and name == task_name

                def press_button(button):
                    nonlocal active
                    self.assertEqual(button, "A")
                    active = False

                emulator = SimpleNamespace(press_button=Mock(side_effect=press_button))
                generator = items._wait_for_emerald_field_item_context()
                with patch.object(items, "task_is_active", side_effect=task_is_active), patch.object(
                    items, "context", SimpleNamespace(emulator=emulator)
                ):
                    next(generator)
                    next(generator)
                    with self.assertRaises(StopIteration):
                        next(generator)

                emulator.press_button.assert_called_once_with("A")

    def test_field_item_context_timeout_closes_the_bag(self):
        from modules.modes.util import items

        emulator = SimpleNamespace(press_button=Mock())
        generator = items._wait_for_emerald_field_item_context()
        with (
            patch.object(items, "_EMERALD_FIELD_ITEM_CONTEXT_TIMEOUT", 2),
            patch.object(items, "task_is_active", return_value=False),
            patch.object(items, "get_game_state", return_value=GameState.BAG_MENU),
            patch.object(items, "context", SimpleNamespace(emulator=emulator)),
        ):
            with self.assertRaisesRegex(BotModeError, "context menu did not start"):
                list(generator)

        self.assertEqual(emulator.press_button.call_args_list[0].args, ("A",))
        self.assertEqual(emulator.press_button.call_args_list[-1].args, ("B",))


if __name__ == "__main__":
    unittest.main()
