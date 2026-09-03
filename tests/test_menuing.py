from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from modules.memory import GameState
from modules.menuing import RotatePokemon


def test_rotate_pokemon_exit_closes_reopened_start_menu_with_fresh_b():
    navigator = RotatePokemon.__new__(RotatePokemon)
    emulator = SimpleNamespace(press_button=Mock(), press_button_fresh=Mock())
    state = {
        "game_state": GameState.PARTY_MENU,
        "party_task": "Task_HandleChooseMonInput",
        "party_b_presses": 0,
        "start_task": None,
        "start_open": False,
    }

    def get_game_state():
        if state["party_task"] == "Task_ClosePartyMenuAndSetCB2":
            state["party_task"] = None
            state["game_state"] = GameState.OVERWORLD
            state["start_task"] = "Task_ShowStartMenu"
            state["start_open"] = True
        return state["game_state"]

    def task_is_active(task):
        return task in (state["party_task"], state["start_task"])

    def parse_start_menu():
        return {"open": state["start_open"]}

    def press_button_fresh(_button):
        if state["party_task"] == "Task_HandleChooseMonInput":
            state["party_b_presses"] += 1
            if state["party_b_presses"] >= 2:
                state["party_task"] = "Task_ClosePartyMenuAndSetCB2"
        elif state["start_task"] == "Task_ShowStartMenu":
            state["start_open"] = False
            state["start_task"] = None

    with (
        patch("modules.menuing.context", SimpleNamespace(emulator=emulator)),
        patch("modules.menuing.get_game_state", side_effect=get_game_state),
        patch("modules.menuing.task_is_active", side_effect=task_is_active),
        patch("modules.menuing.parse_start_menu", side_effect=parse_start_menu),
    ):
        emulator.press_button_fresh.side_effect = press_button_fresh
        exit_controller = navigator.exit_to_overworld()
        for _ in range(10):
            try:
                next(exit_controller)
            except StopIteration:
                break
        else:
            pytest.fail("party rotation cleanup did not finish")

    assert emulator.press_button_fresh.call_count == 3
    assert emulator.press_button_fresh.call_args_list == [(("B",),), (("B",),), (("B",),)]
    emulator.press_button.assert_not_called()
