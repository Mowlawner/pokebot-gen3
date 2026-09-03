"""
'Tasks' are the game's implementation of coroutines. This files contains
abstractions and utilities for working with them.
"""

from functools import cached_property
from typing import Iterator

from modules.context import context
from modules.game import get_symbol_name_before
from modules.memory import get_symbol_name, read_symbol, unpack_uint16, unpack_uint32
from modules.state_cache import state_cache
from modules.text_printer import get_text_printer
from modules.profiler import count as profile_count, profiled


class Task:
    def __init__(self, data: bytes):
        self._data = data

    def __eq__(self, other):
        return other._data == self._data if isinstance(other, Task) else NotImplemented

    def __ne__(self, other):
        return other._data != self._data if isinstance(other, Task) else NotImplemented

    @property
    def function_pointer(self) -> int:
        return unpack_uint32(self._data[:4]) - 1

    @cached_property
    def symbol(self) -> str:
        symbol = get_symbol_name(self.function_pointer, True)
        return hex(self.function_pointer) if symbol == "" else symbol

    @property
    def priority(self) -> int:
        return self._data[7]

    @property
    def data(self) -> bytes:
        return self._data[8:]

    def data_value(self, index: int) -> int:
        return unpack_uint16(self.data[(index * 2) : ((index + 1) * 2)])


class TaskList:
    def __init__(self, data: bytes):
        self._data = data

    def __eq__(self, other):
        if isinstance(other, TaskList):
            return other._data == self._data
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, TaskList):
            return other._data != self._data
        else:
            return NotImplemented

    def __iter__(self) -> Iterator[Task]:
        yield from self._dict.values()

    def __contains__(self, task_name: str):
        return task_name.lower() in self._dict

    def __getitem__(self, item: str) -> Task | None:
        return self._dict.get(item.lower(), None)

    @cached_property
    def _dict(self) -> dict[str, Task]:
        tasks: dict[str, Task] = {}
        for index in range(16):
            task_data = self._data[index * 40 : (index + 1) * 40]
            # offset 4 is `is_active` and offsets 0 through 3 are the function pointer
            if task_data[4] != 0 and task_data[:4] != b"\x00\x00\x00\x00":
                task = Task(task_data)
                if task.symbol != "TaskDummy":
                    tasks[task.symbol.lower()] = task
        return tasks


class ScriptContext:
    def __init__(self, data: bytes):
        self._data = data

    def __eq__(self, other):
        if isinstance(other, ScriptContext):
            return other._data == self._data
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, ScriptContext):
            return other._data != self._data
        else:
            return NotImplemented

    @property
    def is_active(self) -> bool:
        return self._data[1] != 0

    @property
    def mode(self) -> str:
        match self._data[1]:
            case 0:
                return "Stopped"
            case 1:
                return "Bytecode"
            case 2:
                return "Native"
            case _:
                return ""

    @property
    def stack_depth(self) -> int:
        return self._data[0]

    @property
    def comparison_result(self) -> int:
        return self._data[2]

    @property
    def native_pointer(self):
        return unpack_uint32(self._data[4:8])

    @property
    def native_function_name(self):
        return get_symbol_name_before(self.native_pointer, pretty_name=True)

    @property
    def bytecode_pointer(self):
        return unpack_uint32(self._data[8:12])

    @property
    def script_function_name(self):
        return get_symbol_name_before(self.bytecode_pointer, pretty_name=True)

    @property
    def stack_pointers(self) -> list[int]:
        result = []
        for index in range(self.stack_depth):
            offset = 12 + (index * 4)
            pointer = unpack_uint32(self._data[offset : offset + 4])
            result.append(pointer)
        result.append(self.bytecode_pointer)
        return result

    @cached_property
    def stack(self) -> list[str]:
        return [get_symbol_name_before(pointer, pretty_name=True) for pointer in self.stack_pointers]

    @property
    def data(self) -> tuple[int, int, int, int]:
        return (
            unpack_uint32(self._data[100:104]),
            unpack_uint32(self._data[104:108]),
            unpack_uint32(self._data[108:112]),
            unpack_uint32(self._data[112:116]),
        )


@profiled("task_list_total", "task_list_calls")
def get_tasks() -> TaskList:
    if state_cache.tasks.age_in_frames == 0 and state_cache.tasks.value is not None:
        profile_count("task_list_cached_calls")
        return state_cache.tasks.value

    profile_count("task_list_fresh_calls")
    task_list = TaskList(read_symbol("gTasks"))
    state_cache.tasks = task_list
    return task_list


def get_task(task_name: str) -> Task | None:
    try:
        return get_tasks()[task_name]
    except KeyError:
        # During an Emerald script/task handoff the function-pointer symbol
        # table can be incomplete for one frame.  A missing task is the safe
        # interpretation at this boundary; callers will retry on the next
        # observation instead of aborting a campaign capability.
        return None


@profiled("task_state_check_total", "task_state_checks")
def task_is_active(task_name: str) -> bool:
    try:
        return task_name in get_tasks()
    except KeyError:
        # Keep the membership helper consistent with get_task() across the
        # same transient task-table replacement window.
        return False


def get_global_script_context() -> ScriptContext:
    if state_cache.global_script_context.age_in_frames == 0:
        return state_cache.global_script_context.value

    ctx = ScriptContext(read_symbol("sScriptContext1" if context.rom.is_rs else "sGlobalScriptContext"))
    state_cache.global_script_context = ctx
    return ctx


def get_immediate_script_context() -> ScriptContext:
    if state_cache.immediate_script_context.age_in_frames == 0:
        return state_cache.immediate_script_context.value

    ctx = ScriptContext(read_symbol("sScriptContext2" if context.rom.is_rs else "sImmediateScriptContext"))
    state_cache.immediate_script_context = ctx
    return ctx


def is_waiting_for_input() -> bool:
    """
    :return: Whether the game is currently waiting for the A or B button to be pressed in order
             to advance some dialogue or scripted event.
    """
    script_context = get_global_script_context()
    if script_context is not None and script_context.native_function_name == "WaitForAorBPress":
        return True

    if context.rom.is_rs:
        if task_is_active("Task_FieldMessageBox"):
            text_printer_state = read_symbol("gFieldMessageBoxWindow", offset=0x16, size=1)[0]
            return text_printer_state in (8, 9)
    else:
        text_printer_data = read_symbol("sTextPrinters", offset=0x1B, size=2)
        text_printer_is_active = text_printer_data[0]
        text_printer_state = text_printer_data[1]
        # ``and`` otherwise returns the integer printer-active flag when it is
        # zero.  Keep this helper's documented bool contract at the boundary.
        # Emerald keeps the printer active while it is stopped at either a
        # ``\\p`` pause (state 1) or a page clear/scroll prompt (states 2/3).
        # The field-message draw task owns all three states and the script
        # resumes only after an A/B edge.  State 0 is ordinary character
        # rendering and must remain non-actionable.
        return bool(text_printer_is_active and text_printer_state in (1, 2, 3))


def is_field_message_waiting_for_input(field_message_lifecycle_active: bool = False) -> bool:
    """Whether Emerald's ordinary field-message printer is awaiting A/B.

    ``is_waiting_for_input`` intentionally also recognizes the script native
    ``WaitForAorBPress``.  That native is useful for many script operations,
    but is not by itself proof that an ordinary field message is on screen.
    Once a caller has observed the field-message task, the caller can keep
    that lifecycle active while Emerald transitions to ``WaitForAorBPress``.
    During that transition the task and printer may both be inactive, so the
    native wait is the authoritative input signal.
    """
    try:
        if context.rom.is_rs:
            return False
        if not is_waiting_for_input():
            return False
        if task_is_active("Task_DrawFieldMessage"):
            return True
        script_context = get_global_script_context()
        if script_context is None or script_context.native_function_name != "WaitForAorBPress":
            return False
        return field_message_lifecycle_active or get_text_printer().active
    except (AttributeError, KeyError, RuntimeError, ValueError, TypeError, IndexError):
        # Emulator state is legitimately incomplete for a frame during a map
        # transition.  Callers should retry, not switch modes or raise.
        return False


def is_field_message_task_waiting_for_input() -> bool:
    """Whether Emerald's draw task and printer are at the input boundary.

    ``IsFieldMessageBoxHidden`` can remain the script native for one or more
    frames while the printer is paused, before the script reaches its
    ``WaitForAorBPress`` handoff.  Task state 2 plus printer states Wait,
    Clear, or ScrollStart is the ROM-level distinction from ordinary
    character rendering.
    """
    try:
        if context.rom.is_rs or not task_is_active("Task_DrawFieldMessage"):
            return False
        task = get_task("Task_DrawFieldMessage")
        printer = get_text_printer()
        return (
            task is not None
            and task.data_value(0) == 2
            and printer.active
            and printer.raw_state in ("Wait", "Clear", "ScrollStart")
        )
    except (AttributeError, KeyError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def is_emerald_field_dialogue_advanceable(
    *,
    task_active: bool,
    task_name: str | None = None,
    script_active: bool | None,
    native_function_name: str | None,
    script_function_name: str | None,
    input_waiting: bool | None,
    field_message_lifecycle_active: bool = False,
) -> bool:
    """Return Emerald's authoritative ordinary-message advance predicate.

    Emerald exposes the same message lifecycle through two adjacent runtime
    states: the field-message task/printer, then the script native
    ``WaitForAorBPress`` after the draw task has gone away.  The latter is only
    dialogue when the script is ``Std_MsgboxDefault`` and the sampled runtime
    state says input is ready.  This is the low-level predicate shared by the
    legacy opening handler and observation-driven campaign capabilities.
    """
    if is_field_message_waiting_for_input(field_message_lifecycle_active or task_active):
        return True
    # New Game Birch's opening is owned by main_menu.c rather than the field
    # script engine.  Its ``Task_NewGameBirchSpeech_*`` task drives the same
    # text-printer pause control code (\\p), but there is no field-message task
    # and no WaitForAorBPress native callback.
    if task_name and task_name.startswith("Task_NewGameBirchSpeech") and input_waiting:
        return True
    return bool(
        script_active
        and native_function_name == "WaitForAorBPress"
        and script_function_name == "Std_MsgboxDefault"
        and input_waiting
    )
