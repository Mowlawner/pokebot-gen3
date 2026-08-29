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
from modules.tasks import get_global_script_context, get_tasks
from modules.semantic_choices import ChoiceConsequence, ConsequenceKnowledge, DialogueChoice


class EmeraldConfirmationChoice(Enum):
    YES = auto()
    NO = auto()


class EmeraldConfirmationContext(Enum):
    PLAYER_NAME = auto()
    POKEMON_NICKNAME = auto()
    GO_SEE_RIVAL = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class EmeraldConfirmationObservation:
    active: bool
    selected: EmeraldConfirmationChoice | None
    input_ready: bool
    context: EmeraldConfirmationContext = EmeraldConfirmationContext.UNKNOWN
    task_name: str | None = None
    question_text: str | None = None
    options: tuple[str, ...] = ("YES", "NO")
    source_affordance: str | None = None
    script_identity: str | None = None
    consequences: tuple[ChoiceConsequence, ...] = ()

    @property
    def dialogue_choice(self) -> DialogueChoice:
        return DialogueChoice(
            question_text=self.question_text,
            options=self.options,
            selected_option=self.selected.name if self.selected is not None else None,
            source_affordance=self.source_affordance,
            script_identity=self.script_identity,
            consequences=self.consequences,
        )


_PLAYER_NAME_TASKS = {
    "Task_NewGameBirchSpeech_CreateNameYesNo",
    "Task_NewGameBirchSpeech_ProcessNameYesNoMenu",
}
_CLOCK_CONFIRM_TASKS = {"Task_SetClock_HandleConfirmInput"}
_READY_TASKS = {
    "Task_NewGameBirchSpeech_ProcessNameYesNoMenu",
    "Task_HandleYesNoInput",
}
_STARTER_NICKNAME_SCRIPT = "LittlerootTown_ProfessorBirchsLab_EventScript_GiveStarterEvent"
_GO_SEE_RIVAL_SCRIPT = "LittlerootTown_ProfessorBirchsLab_EventScript_GoSeeRival"


def _active_confirmation_task() -> str | None:
    try:
        for task in get_tasks() or ():
            name = task.symbol
            # This deliberately recognizes task-owned menu handlers, not
            # text or pixels.  New prompt implementations remain observable
            # as UNKNOWN until their handler is classified here.
            if (
                name in _PLAYER_NAME_TASKS
                or name in _CLOCK_CONFIRM_TASKS
                or (
                    name.startswith("Task_")
                    and "YesNo" in name
                    and ("Process" in name or "Handle" in name or "Create" in name)
                )
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
    if task_name == "Task_HandleYesNoInput":
        try:
            script = get_global_script_context()
            stack = tuple(script.stack or ()) if script is not None else ()
            # At the menu boundary the active script can be represented either
            # as the current function or as an entry in the return stack.
            # Treat both forms as the same observed ROM ownership; otherwise
            # this real prompt becomes UNKNOWN and the dispatcher waits.
            script_function = getattr(script, "script_function_name", None) if script is not None else None
            if _STARTER_NICKNAME_SCRIPT in stack or script_function == _STARTER_NICKNAME_SCRIPT:
                context = EmeraldConfirmationContext.POKEMON_NICKNAME
            elif _GO_SEE_RIVAL_SCRIPT in stack or script_function == _GO_SEE_RIVAL_SCRIPT:
                context = EmeraldConfirmationContext.GO_SEE_RIVAL
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
    input_ready = task_name in _READY_TASKS or (
        ("Process" in task_name or "Handle" in task_name) and "Create" not in task_name
    )
    script_identity = None
    question_text = None
    consequences = ()
    if task_name in _CLOCK_CONFIRM_TASKS:
        # This is ROM metadata, not policy: the script's two continuations
        # have been verified against the live ROM.  Policy still chooses by
        # desired fact, and remains conservative for unknown scripts.
        script_identity = "PlayersHouse_2F_EventScript_SetWallClock"
        question_text = "Is this the correct time?"
        consequences = (
            ChoiceConsequence(
                "YES",
                ConsequenceKnowledge.KNOWN,
                {"wall_clock_set": True},
                True,
                "accept displayed time and finish clock initialization",
            ),
            ChoiceConsequence(
                "NO",
                ConsequenceKnowledge.KNOWN,
                {"wall_clock_set": False},
                False,
                "return to clock configuration",
            ),
        )
    elif context is EmeraldConfirmationContext.GO_SEE_RIVAL:
        script_identity = _GO_SEE_RIVAL_SCRIPT
        question_text = "Go see your rival?"
    return EmeraldConfirmationObservation(
        active=True,
        selected=selected,
        input_ready=input_ready,
        context=context,
        task_name=task_name,
        question_text=question_text,
        source_affordance=script_identity,
        script_identity=script_identity,
        consequences=consequences,
    )


__all__ = [
    "EmeraldConfirmationChoice",
    "EmeraldConfirmationContext",
    "EmeraldConfirmationObservation",
    "observe_emerald_confirmation",
]
