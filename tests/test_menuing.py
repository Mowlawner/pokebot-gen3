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


def test_rotate_pokemon_replacement_cursor_timeout_is_bounded():
    navigator = RotatePokemon.__new__(RotatePokemon)
    navigator.party = (object(), object())
    navigator._first_party_index = 1
    navigator._second_party_index = 0
    navigator.completed = False
    navigator.failure = None
    navigator.last_input = None
    navigator._diagnostic_phase = None

    emulator = SimpleNamespace(press_button=Mock())
    fake_context = SimpleNamespace(emulator=emulator, bot_mode="Campaign Progression")

    def drive(generator):
        while True:
            try:
                next(generator)
            except StopIteration as result:
                return result.value

    with (
        patch("modules.menuing.context", fake_context),
        patch(
            "modules.menuing.get_party_menu_cursor_pos",
            return_value={"slot_id_2": 1},
        ),
        patch("modules.menuing.diagnostic_print"),
    ):
        result = drive(navigator.select_replacement())

    assert result is False
    assert navigator.failure is not None
    assert navigator.failure.phase == "select_second_pokemon"
    assert "cursor" in navigator.failure.reason
    assert emulator.press_button.call_count == 180


def test_rotate_pokemon_reports_success_after_all_menu_phases():
    navigator = RotatePokemon.__new__(RotatePokemon)
    navigator.party = (object(), object())
    navigator._first_party_index = 1
    navigator._second_party_index = 0
    navigator.completed = False
    navigator.failure = None
    navigator.last_input = None
    navigator._diagnostic_phase = None
    navigator._diagnostic_state = lambda: (GameState.OVERWORLD, (), None)

    fake_context = SimpleNamespace(bot_mode="Campaign Progression")

    def frames():
        yield

    with (
        patch("modules.menuing.context", fake_context),
        patch("modules.menuing.diagnostic_print"),
        patch("modules.menuing.StartMenuNavigator") as start_menu,
        patch("modules.menuing.PokemonPartyMenuNavigator") as party_menu,
        patch.object(navigator, "select_replacement", return_value=frames()),
        patch.object(navigator, "exit_to_overworld", return_value=frames()),
    ):
        start_menu.return_value.step.return_value = frames()
        party_menu.return_value.step.return_value = frames()
        controller = navigator.step()
        for _ in range(20):
            try:
                next(controller)
            except StopIteration:
                break
        else:
            pytest.fail("successful lead rotation did not finish")

    assert navigator.completed is True
    assert navigator.failure is None
