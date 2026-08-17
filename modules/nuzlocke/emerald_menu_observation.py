"""Read-only observations of Pokémon Emerald's title/menu tasks.

The offsets in this module mirror the Emerald decomp task definitions:

* ``main_menu.c``: ``tMenuType = data[0]``, ``tCurrItem = data[1]``
* ``option_menu.c``: ``tMenuSelection = data[0]`` and option values in
  ``data[1:7]``

These are task-owned values, not inferred cursor positions.  A menu can be
active while its input task is not yet ready (for example during fade-in), so
``input_ready`` is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from modules.memory import GameState, get_game_state
from modules.tasks import get_task


class EmeraldMenuKind(Enum):
    MAIN_MENU = auto()
    OPTIONS_MENU = auto()


class EmeraldMainMenuItem(Enum):
    CONTINUE = auto()
    NEW_GAME = auto()
    OPTIONS = auto()
    MYSTERY_GIFT = auto()
    MYSTERY_EVENTS = auto()


class EmeraldOptionsItem(Enum):
    TEXT_SPEED = auto()
    BATTLE_SCENE = auto()
    BATTLE_STYLE = auto()
    SOUND = auto()
    BUTTON_MODE = auto()
    FRAME_TYPE = auto()
    CANCEL = auto()


@dataclass(frozen=True, slots=True)
class EmeraldMenuObservation:
    menu_kind: EmeraldMenuKind
    cursor_index: int | None
    selected_item: EmeraldMainMenuItem | EmeraldOptionsItem | None
    input_ready: bool
    task_name: str | None
    raw_cursor_value: int | None
    raw_menu_type: int | None = None
    raw_option_values: tuple[int, ...] | None = None


_MAIN_MENU_TASKS = (
    "Task_MainMenuCheckSaveFile",
    "Task_MainMenuCheckBattery",
    "Task_DisplayMainMenu",
    "Task_HighlightSelectedMainMenuItem",
    "Task_HandleMainMenuInput",
    "Task_HandleMainMenuAPressed",
)
_OPTIONS_MENU_TASKS = (
    "Task_OptionMenuFadeIn",
    "Task_OptionMenuProcessInput",
    "Task_OptionMenuSave",
    "Task_OptionMenuFadeOut",
)


def _first_task(names: tuple[str, ...]):
    for name in names:
        task = get_task(name)
        if task is not None:
            return name, task
    return None, None


def _main_item(menu_type: int | None, cursor: int | None) -> EmeraldMainMenuItem | None:
    if menu_type is None or cursor is None:
        return None
    items = {
        0: (EmeraldMainMenuItem.NEW_GAME, EmeraldMainMenuItem.OPTIONS),
        1: (EmeraldMainMenuItem.CONTINUE, EmeraldMainMenuItem.NEW_GAME, EmeraldMainMenuItem.OPTIONS),
        2: (
            EmeraldMainMenuItem.CONTINUE,
            EmeraldMainMenuItem.NEW_GAME,
            EmeraldMainMenuItem.MYSTERY_GIFT,
            EmeraldMainMenuItem.OPTIONS,
        ),
        3: (
            EmeraldMainMenuItem.CONTINUE,
            EmeraldMainMenuItem.NEW_GAME,
            EmeraldMainMenuItem.MYSTERY_GIFT,
            EmeraldMainMenuItem.MYSTERY_EVENTS,
            EmeraldMainMenuItem.OPTIONS,
        ),
    }.get(menu_type)
    return items[cursor] if items is not None and 0 <= cursor < len(items) else None


def observe_emerald_menu() -> EmeraldMenuObservation | None:
    """Return the currently observable Emerald menu, if one is active."""
    game_state = get_game_state()
    if game_state == GameState.MAIN_MENU:
        task_name, task = _first_task(_MAIN_MENU_TASKS)
        menu_type = task.data_value(0) if task is not None else None
        cursor = task.data_value(1) if task is not None else None
        return EmeraldMenuObservation(
            EmeraldMenuKind.MAIN_MENU,
            cursor,
            _main_item(menu_type, cursor),
            task_name == "Task_HandleMainMenuInput",
            task_name,
            cursor,
            menu_type,
        )
    if game_state == GameState.OPTIONS_MENU:
        task_name, task = _first_task(_OPTIONS_MENU_TASKS)
        cursor = task.data_value(0) if task is not None else None
        item = None
        if cursor is not None and 0 <= cursor < len(EmeraldOptionsItem):
            item = tuple(EmeraldOptionsItem)[cursor]
        return EmeraldMenuObservation(
            EmeraldMenuKind.OPTIONS_MENU,
            cursor,
            item,
            task_name == "Task_OptionMenuProcessInput",
            task_name,
            cursor,
            None,
            tuple(task.data_value(index) for index in range(1, 7)) if task is not None else None,
        )
    return None


__all__ = [
    "EmeraldMainMenuItem",
    "EmeraldMenuKind",
    "EmeraldMenuObservation",
    "EmeraldOptionsItem",
    "observe_emerald_menu",
]
