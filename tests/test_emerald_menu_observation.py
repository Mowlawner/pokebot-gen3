import types
import unittest
from unittest.mock import patch

from modules.memory import GameState
from modules.nuzlocke.emerald_menu_observation import (
    EmeraldMainMenuItem,
    EmeraldMenuKind,
    EmeraldOptionsItem,
    observe_emerald_menu,
)


class EmeraldMenuObservationTests(unittest.TestCase):
    @staticmethod
    def task(*values):
        return types.SimpleNamespace(data_value=lambda index: values[index])

    def test_main_menu_uses_data_one_for_cursor_and_data_zero_for_menu_variant(self):
        with (
            patch("modules.nuzlocke.emerald_menu_observation.get_game_state", return_value=GameState.MAIN_MENU),
            patch(
                "modules.nuzlocke.emerald_menu_observation.get_task",
                side_effect=lambda name: self.task(0, 1) if name == "Task_HandleMainMenuInput" else None,
            ),
        ):
            observation = observe_emerald_menu()

        self.assertEqual(observation.menu_kind, EmeraldMenuKind.MAIN_MENU)
        self.assertEqual(observation.cursor_index, 1)
        self.assertEqual(observation.raw_menu_type, 0)
        self.assertIs(observation.selected_item, EmeraldMainMenuItem.OPTIONS)
        self.assertTrue(observation.input_ready)

    def test_options_menu_uses_data_zero_for_selection(self):
        with (
            patch("modules.nuzlocke.emerald_menu_observation.get_game_state", return_value=GameState.OPTIONS_MENU),
            patch(
                "modules.nuzlocke.emerald_menu_observation.get_task",
                side_effect=lambda name: (
                    self.task(0, 2, 1, 0, 0, 0, 3) if name == "Task_OptionMenuProcessInput" else None
                ),
            ),
        ):
            observation = observe_emerald_menu()

        self.assertEqual(observation.menu_kind, EmeraldMenuKind.OPTIONS_MENU)
        self.assertEqual(observation.cursor_index, 0)
        self.assertIs(observation.selected_item, EmeraldOptionsItem.TEXT_SPEED)
        self.assertEqual(observation.raw_option_values, (2, 1, 0, 0, 0, 3))
        self.assertTrue(observation.input_ready)

    def test_options_fade_has_cursor_but_is_not_input_ready(self):
        with (
            patch("modules.nuzlocke.emerald_menu_observation.get_game_state", return_value=GameState.OPTIONS_MENU),
            patch(
                "modules.nuzlocke.emerald_menu_observation.get_task",
                side_effect=lambda name: self.task(5, 0, 0, 0, 0, 0, 0) if name == "Task_OptionMenuFadeIn" else None,
            ),
        ):
            observation = observe_emerald_menu()

        self.assertIs(observation.selected_item, EmeraldOptionsItem.FRAME_TYPE)
        self.assertFalse(observation.input_ready)

    def test_unknown_cursor_is_preserved_without_inventing_selected_item(self):
        with (
            patch("modules.nuzlocke.emerald_menu_observation.get_game_state", return_value=GameState.OPTIONS_MENU),
            patch(
                "modules.nuzlocke.emerald_menu_observation.get_task",
                side_effect=lambda name: (
                    self.task(99, 0, 0, 0, 0, 0, 0) if name == "Task_OptionMenuProcessInput" else None
                ),
            ),
        ):
            observation = observe_emerald_menu()

        self.assertEqual(observation.raw_cursor_value, 99)
        self.assertIsNone(observation.selected_item)


if __name__ == "__main__":
    unittest.main()
