"""Task-backed observations of Emerald's two-row confirmation menus.

Emerald's normal yes/no menus share ``menu.c``'s ``sMenu`` state.  The
selected row is ``struct Menu.cursorPos`` (offset 2), while the active task
identifies both whether the menu is ready for input and, when available, the
semantic owner of the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from modules.memory import read_symbol
from modules.tasks import get_tasks


class EmeraldConfirmationChoice(Enum):
    YES = auto()
    NO = auto()


class EmeraldConfirmationContext(Enum):
    PLAYER_NAME = auto()
    POKEMON_NICKNAME = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class EmeraldConfirmationObservation:
    active: bool
    selected: EmeraldConfirmationChoice | None
    input_ready: bool
    context: EmeraldConfirmationContext = EmeraldConfirmationContext.UNKNOWN
    task_name: str | None = None


_PLAYER_NAME_TASKS = {
    "Task_NewGameBirchSpeech_CreateNameYesNo",
    "Task_NewGameBirchSpeech_ProcessNameYesNoMenu",
}
_READY_TASKS = {
    "Task_NewGameBirchSpeech_ProcessNameYesNoMenu",
    "Task_HandleYesNoInput",
}


def _active_confirmation_task() -> str | None:
    try:
        for task in get_tasks() or ():
            name = task.symbol
            # This deliberately recognizes task-owned menu handlers, not
            # text or pixels.  New prompt implementations remain observable
            # as UNKNOWN until their handler is classified here.
            if name in _PLAYER_NAME_TASKS or (
                name.startswith("Task_")
                and "YesNo" in name
                and ("Process" in name or "Handle" in name or "Create" in name)
            ):
                return name
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return None
    return None


def observe_emerald_confirmation() -> EmeraldConfirmationObservation | None:
    """Observe an active Emerald yes/no task, or ``None`` outside one."""
    task_name = _active_confirmation_task()
    if task_name is None:
        return None

    selected = None
    try:
        cursor = read_symbol("sMenu", offset=0x02, size=1)[0]
        selected = (EmeraldConfirmationChoice.YES, EmeraldConfirmationChoice.NO)[cursor]
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass

    context = (
        EmeraldConfirmationContext.PLAYER_NAME
        if task_name in _PLAYER_NAME_TASKS
        else EmeraldConfirmationContext.UNKNOWN
    )
    input_ready = task_name in _READY_TASKS or (
        ("Process" in task_name or "Handle" in task_name) and "Create" not in task_name
    )
    return EmeraldConfirmationObservation(
        active=True,
        selected=selected,
        input_ready=input_ready,
        context=context,
        task_name=task_name,
    )


__all__ = [
    "EmeraldConfirmationChoice",
    "EmeraldConfirmationContext",
    "EmeraldConfirmationObservation",
    "observe_emerald_confirmation",
]
