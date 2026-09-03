import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class _ShopEmulator:
    def __init__(self):
        self.fresh_presses = []
        self.presses = []

    def press_button_fresh(self, button):
        self.fresh_presses.append(button)

    def press_button(self, button):
        self.presses.append(button)


class BuyInShopTests(unittest.TestCase):
    def test_buy_menu_open_uses_a_fresh_edge_after_clerk_interaction(self):
        from modules.modes.util import higher_level_actions as actions

        ball = SimpleNamespace(name="Poké Ball", price=200)
        emulator = _ShopEmulator()
        return_task_checks = 0

        def task_is_active(task_name):
            nonlocal return_task_checks
            if task_name == "Task_ReturnToShopMenu":
                return_task_checks += 1
                return return_task_checks == 1
            return task_name in {
                "Task_ShopMenu",
                "Task_BuyHowManyDialogueHandleInput",
                "Task_BuyMenu",
            }

        quantity_task = SimpleNamespace(data_value=lambda index: 1)
        fake_context = SimpleNamespace(
            emulator=emulator,
            rom=SimpleNamespace(is_rs=False),
        )
        with (
            patch.object(actions, "context", fake_context),
            patch.object(actions, "task_is_active", side_effect=task_is_active),
            patch.object(actions, "get_mart_buyable_items", return_value=[ball]),
            patch.object(actions, "get_mart_main_menu_scroll_position", return_value=0),
            patch.object(actions, "get_mart_buy_menu_scroll_position", return_value=0),
            patch.object(actions, "get_game_state_symbol", return_value="CB2_BUYMENU"),
            patch.object(actions, "get_task", return_value=quantity_task),
            patch.object(actions, "get_player", return_value=SimpleNamespace(money=200)),
        ):
            list(actions.buy_in_shop([(ball, 1)]))

        self.assertEqual(emulator.fresh_presses, ["A"])
        self.assertNotIn("A", emulator.presses)

    def test_stalled_buy_callback_is_bounded_and_reports_runtime_state(self):
        from modules.modes.util import higher_level_actions as actions

        ball = SimpleNamespace(name="Poké Ball", price=200)
        emulator = _ShopEmulator()
        fake_context = SimpleNamespace(
            emulator=emulator,
            rom=SimpleNamespace(is_rs=False),
        )
        with (
            patch.object(actions, "context", fake_context),
            patch.object(actions, "task_is_active", return_value=True),
            patch.object(actions, "get_mart_buyable_items", return_value=[ball]),
            patch.object(actions, "get_mart_main_menu_scroll_position", return_value=0),
            patch.object(actions, "get_game_state_symbol", return_value="CB2_OTHER"),
            patch.object(actions, "get_player", return_value=SimpleNamespace(money=200)),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out.*CB2_BUYMENU"):
                list(actions.buy_in_shop([(ball, 1)]))

        self.assertEqual(emulator.fresh_presses, ["A"])


if __name__ == "__main__":
    unittest.main()
