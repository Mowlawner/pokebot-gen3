from types import SimpleNamespace
from unittest.mock import patch

from modules.nuzlocke.emerald_confirmation import (
    EmeraldConfirmationChoice,
    EmeraldConfirmationContext,
    observe_emerald_confirmation,
)


def test_starter_nickname_confirmation_uses_current_script_when_stack_is_empty():
    task = SimpleNamespace(symbol="Task_HandleYesNoInput")
    script = SimpleNamespace(
        stack=(),
        script_function_name="LittlerootTown_ProfessorBirchsLab_EventScript_GiveStarterEvent",
    )
    with (
        patch("modules.nuzlocke.emerald_confirmation.get_tasks", return_value=(task,)),
        patch("modules.nuzlocke.emerald_confirmation.get_global_script_context", return_value=script),
        patch("modules.nuzlocke.emerald_confirmation.read_symbol", return_value=bytes((1,))),
    ):
        observation = observe_emerald_confirmation()

    assert observation is not None
    assert observation.context is EmeraldConfirmationContext.POKEMON_NICKNAME
    assert observation.selected is EmeraldConfirmationChoice.NO
    assert observation.input_ready is True
