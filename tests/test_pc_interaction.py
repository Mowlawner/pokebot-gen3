from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules.modes import BotModeError
from modules.memory import GameState
from modules.modes.util.pc_interaction import _open_pokemon_storage_menu, _validate_party_deposit


class DummyParty(list):
    @property
    def non_eggs(self):
        return [pokemon for pokemon in self if not pokemon.is_egg]

    @property
    def non_fainted_pokemon(self):
        return [pokemon for pokemon in self.non_eggs if pokemon.current_hp > 0]


def pokemon(current_hp, *, is_egg=False):
    return SimpleNamespace(current_hp=current_hp, is_egg=is_egg)


def test_fainted_party_member_can_be_deposited_when_one_living_member_remains():
    living = pokemon(10)
    fainted = pokemon(0)

    _validate_party_deposit(DummyParty([living, fainted]), fainted)


def test_only_living_party_member_cannot_be_deposited():
    living = pokemon(10)
    fainted = pokemon(0)

    with pytest.raises(BotModeError, match="only living"):
        _validate_party_deposit(DummyParty([living, fainted]), living)


def test_storage_open_waits_for_callback_transition_after_task_becomes_ready():
    task = SimpleNamespace(data_value=lambda index: 2 if index == 0 else 0)
    states = iter((GameState.OVERWORLD, GameState.OVERWORLD, GameState.POKE_STORAGE))

    with (
        patch(
            "modules.modes.util.pc_interaction.context",
            SimpleNamespace(
                rom=SimpleNamespace(is_rs=False, is_frlg=False),
                emulator=SimpleNamespace(press_button=MagicMock()),
            ),
        ),
        patch(
            "modules.modes.util.pc_interaction.get_global_script_context",
            return_value=SimpleNamespace(is_active=False, stack=()),
        ),
        patch("modules.modes.util.pc_interaction.get_game_state", side_effect=lambda: next(states)),
        patch("modules.modes.util.pc_interaction.get_task", return_value=task),
        patch("modules.modes.util.pc_interaction._get_menu_cursor", return_value=SimpleNamespace(cursor_position=0)),
        patch("modules.modes.util.pc_interaction._get_storage_menu", return_value=[]),
        patch("modules.modes.util.pc_interaction.wait_until_task_is_active", return_value=iter(())),
    ):
        list(_open_pokemon_storage_menu(0))


def test_storage_open_accepts_live_menu_when_callback_state_is_stale():
    task = SimpleNamespace(data_value=lambda index: 2 if index == 0 else 0)

    with (
        patch(
            "modules.modes.util.pc_interaction.context",
            SimpleNamespace(
                rom=SimpleNamespace(is_rs=False, is_frlg=False),
                emulator=SimpleNamespace(press_button=MagicMock()),
            ),
        ),
        patch(
            "modules.modes.util.pc_interaction.get_global_script_context",
            return_value=SimpleNamespace(is_active=False, stack=()),
        ),
        patch("modules.modes.util.pc_interaction.get_game_state", return_value=GameState.OVERWORLD),
        patch("modules.modes.util.pc_interaction.get_game_state_symbol", return_value="CB2_OVERWORLD"),
        patch("modules.modes.util.pc_interaction.get_task", return_value=task),
        patch("modules.modes.util.pc_interaction._get_menu_cursor", return_value=SimpleNamespace(cursor_position=0)),
        patch("modules.modes.util.pc_interaction._get_storage_menu", return_value=[(1, "Withdraw Pokemon")]),
        patch("modules.modes.util.pc_interaction.wait_until_task_is_active", return_value=iter(())),
    ):
        list(_open_pokemon_storage_menu(0))


def test_storage_open_selects_pokemon_when_pc_script_has_finished():
    cursor = [0]
    pc_task = SimpleNamespace(data_value=lambda index: {0: 2, 1: cursor[0], 2: 0}[index])
    storage_task = SimpleNamespace(data_value=lambda index: 0)
    press_button = MagicMock(side_effect=lambda button: cursor.__setitem__(0, 1) if button == "Down" else None)
    wait_for_storage_task = MagicMock(return_value=iter(()))

    with (
        patch(
            "modules.modes.util.pc_interaction.context",
            SimpleNamespace(
                rom=SimpleNamespace(is_rs=False, is_frlg=False),
                emulator=SimpleNamespace(press_button=press_button),
            ),
        ),
        patch(
            "modules.modes.util.pc_interaction.get_global_script_context",
            return_value=SimpleNamespace(is_active=False, stack=()),
        ),
        patch("modules.modes.util.pc_interaction.get_game_state", return_value=GameState.OVERWORLD),
        patch(
            "modules.modes.util.pc_interaction.get_task",
            side_effect=lambda name: pc_task if name == "Task_PCMainMenu" else storage_task,
        ),
        patch(
            "modules.modes.util.pc_interaction._get_storage_menu",
            side_effect=[[], [(1, "Withdraw Pokemon")]],
        ),
        patch(
            "modules.modes.util.pc_interaction.wait_until_task_is_active",
            wait_for_storage_task,
        ),
    ):
        list(_open_pokemon_storage_menu(1))

    press_button.assert_called_once_with("Down")
    wait_for_storage_task.assert_called_once_with("Task_PokeStorageMain", "A")
