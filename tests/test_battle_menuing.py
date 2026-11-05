from unittest.mock import Mock, patch

from modules.battle_menuing import _menu_cursor_path, scroll_to_move
from modules.context import context


def test_three_move_cursor_routes_bottom_left_to_top_right_without_empty_slot():
    assert _menu_cursor_path(2, 1, (0, 1, 2)) == ("Up", "Right")


def test_three_move_cursor_preserves_safe_top_right_to_bottom_left_route():
    assert _menu_cursor_path(1, 2, (0, 1, 2)) == ("Left", "Down")


def test_scroll_to_move_applies_the_safe_three_move_route():
    cursor = [2]
    emulator = Mock()

    def press_button(button):
        if button == "Up" and cursor[0] == 2:
            cursor[0] = 0
        elif button == "Right" and cursor[0] == 0:
            cursor[0] = 1

    emulator.press_button.side_effect = press_button
    previous_emulator = context.emulator
    context.emulator = emulator
    try:
        with patch(
            "modules.battle_menuing.read_symbol",
            side_effect=lambda _symbol, size=4: bytes((cursor[0], 0, 0, 0)),
        ):
            list(scroll_to_move(1, valid_indices=(0, 1, 2)))
    finally:
        context.emulator = previous_emulator

    assert [call.args[0] for call in emulator.press_button.call_args_list] == ["Up", "Right"]
