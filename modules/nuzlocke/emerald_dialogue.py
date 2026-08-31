"""Frame-local Emerald field-dialogue observation and execution.

This module is campaign-neutral.  It intentionally does not depend on the
opening phase machine or Birch-specific task names.
"""

from __future__ import annotations

from typing import Generator

from modules.context import context
from modules.console import diagnostic_print
from modules.memory import get_game_state, GameState
from modules.tasks import (
    get_global_script_context,
    get_task,
    get_tasks,
    is_emerald_field_dialogue_advanceable,
)
from modules.text_printer import get_text_printer


def dialogue_state_snapshot() -> tuple:
    """Capture the current field-message runtime state without phase data."""
    task_active = False
    task_state = None
    script_active = None
    native = None
    script = None
    waiting = None
    printer_active = None
    printer_state = None
    active_tasks = ()
    native_pointer = None
    bytecode_pointer = None
    try:
        task = get_task("Task_DrawFieldMessage")
        task_active = task is not None
        if task is not None:
            task_state = task.data_value(0)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    try:
        script_context = get_global_script_context()
        if script_context is not None:
            script_active = script_context.is_active
            native = script_context.native_function_name
            script = script_context.script_function_name
            native_pointer = getattr(script_context, "native_pointer", None)
            bytecode_pointer = getattr(script_context, "bytecode_pointer", None)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    try:
        from modules.tasks import is_waiting_for_input

        waiting = is_waiting_for_input()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    try:
        printer = get_text_printer()
        printer_active = printer.active
        printer_state = printer.raw_state
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    try:
        active_tasks = tuple(task.symbol for task in get_tasks())
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    message_visible = native == "IsFieldMessageBoxHidden" or bool(printer_active)
    return (
        task_active,
        task_state,
        script_active,
        native,
        script,
        waiting,
        None,
        message_visible,
        printer_state,
        active_tasks,
        native_pointer,
        bytecode_pointer,
    )


def observe_dialogue(
    field_message_lifecycle_active: bool = False,
    *,
    state: tuple | None = None,
    advanceable_predicate=None,
) -> tuple[bool, bool]:
    """Return ``(actionable, lifecycle_active)`` for the current frame."""
    state = dialogue_state_snapshot() if state is None else state
    lifecycle = field_message_lifecycle_active or state[0]
    birch_task = next(
        (name for name in ((state[9] if len(state) > 9 else ()) or ()) if name.startswith("Task_NewGameBirchSpeech")),
        None,
    )
    predicate = advanceable_predicate or is_emerald_field_dialogue_advanceable
    actionable = predicate(
        task_active=state[0],
        task_name=birch_task,
        script_active=state[2],
        native_function_name=state[3],
        script_function_name=state[4],
        input_waiting=state[5],
        field_message_lifecycle_active=lifecycle,
    )
    # The native handoff is independently authoritative once the ROM has
    # stopped the draw task.  Keep this in the shared observer so callers do
    # not need a second, subtly different standard-message predicate.
    if not actionable:
        actionable = bool(state[2] and state[3] == "WaitForAorBPress" and state[4] == "Std_MsgboxDefault" and state[5])
    if actionable:
        lifecycle = True
    if (
        lifecycle
        and not state[0]
        and not (state[3] == "WaitForAorBPress" and state[4] == "Std_MsgboxDefault" and state[5])
        and not state[7]
    ):
        lifecycle = False
    diagnostic_print(
        lambda: (
            "EMERALD_DIALOGUE_OBSERVATION: "
            f"frame={getattr(context, 'frame', None)!r} "
            f"task_active={state[0]!r} task_state={state[1]!r} "
            f"birch_task={birch_task!r} active_tasks={state[9]!r} "
            f"script_active={state[2]!r} native={state[3]!r} script={state[4]!r} "
            f"input_waiting={state[5]!r} message_visible={state[7]!r} "
            f"printer_state={state[8]!r} lifecycle_in={field_message_lifecycle_active!r} "
            f"actionable={actionable!r} lifecycle_out={lifecycle!r}"
        ),
        trace=True,
    )
    return actionable, lifecycle


def advance_dialogue(
    field_message_lifecycle_active: bool = False,
    *,
    actionable: bool | None = None,
) -> Generator:
    """Emit one fresh A edge when the current field message accepts input."""
    if actionable is None:
        actionable, _ = observe_dialogue(field_message_lifecycle_active)
    if actionable:
        press_button_fresh = getattr(context.emulator, "press_button_fresh", None)
        if callable(press_button_fresh):
            press_button_fresh("A")
        else:
            context.emulator.press_button("A")
        diagnostic_print(
            lambda: ("EMERALD_DIALOGUE_EXECUTION: " f"frame={getattr(context, 'frame', None)!r} button='A' fresh=True"),
            trace=True,
        )
    else:
        diagnostic_print(
            lambda: ("EMERALD_DIALOGUE_EXECUTION: " f"frame={getattr(context, 'frame', None)!r} action='WAIT'"),
            trace=True,
        )
    yield


__all__ = ["advance_dialogue", "dialogue_state_snapshot", "observe_dialogue"]
