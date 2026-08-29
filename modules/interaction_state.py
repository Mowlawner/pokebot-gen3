"""State-aware, read-only description of the game's current interaction.

This module deliberately does not execute input.  It provides a stable boundary
between emulator-specific observations and future goal/action controllers.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping

from modules.memory import GameState, get_game_state, read_symbol
from modules.context import context
from modules.player import player_avatar_is_controllable
from modules.tasks import (
    get_global_script_context,
    get_task,
    get_tasks,
    is_field_message_waiting_for_input,
    is_field_message_task_waiting_for_input,
    is_waiting_for_input,
    task_is_active,
)
from modules.profiler import count, now, timing
from modules.console import diagnostic_print


class InteractionType(Enum):
    DIALOGUE = auto()
    CHOICE = auto()
    MENU = auto()
    BATTLE = auto()
    OVERWORLD = auto()
    SPECIAL_INTERACTION = auto()
    UNKNOWN = auto()


class InteractionPhase(Enum):
    """Semantic wait state derived from Emerald script/task ownership."""

    NONE = auto()
    FIELD_MESSAGE_RENDER_WAIT = auto()
    FIELD_MESSAGE_INPUT_WAIT = auto()
    CHOICE_MENU_INPUT_WAIT = auto()
    SCRIPT_NATIVE_WAIT = auto()


@dataclass(frozen=True)
class InteractionObservation:
    """Inputs to :func:`classify_interaction` from one emulator frame."""

    game_state: GameState | Any
    dialogue_waiting: bool = False
    choice_options: tuple[str, ...] = ()
    menu_options: tuple[str, ...] = ()
    special_interaction: str | None = None
    controllable: bool = False
    field_message_lifecycle_active: bool = False
    field_message_advance_ready: bool = False
    script_active: bool = False
    script_function: str | None = None
    native_function: str | None = None
    interaction_phase: InteractionPhase = InteractionPhase.NONE
    choice_menu_active: bool = False
    choice_menu_input_ready: bool = False
    choice_selected: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


_field_message_lifecycle_active = False
_field_message_advance_ready = False

_FIELD_MESSAGE_TRANSITION_NATIVES = frozenset({"IsFieldMessageBoxHidden"})


def _observe_field_message_waiting(state: GameState | Any) -> bool:
    """Carry Emerald's field-message lifecycle across script native waits."""
    global _field_message_lifecycle_active, _field_message_advance_ready
    if state is not GameState.OVERWORLD:
        # During a map/script boundary the coarse game-state reader can
        # report UNKNOWN even though Emerald's field script is at the
        # message input native.  Preserve the actionable dialogue signal
        # from the script context instead of classifying it as an inert
        # unknown state.
        try:
            script_context = get_global_script_context()
            input_waiting = is_waiting_for_input()
            if (
                script_context is not None
                and getattr(script_context, "is_active", False)
                and getattr(script_context, "native_function_name", None) == "WaitForAorBPress"
                and getattr(script_context, "script_function_name", None) == "Std_MsgboxDefault"
                and input_waiting
            ):
                _field_message_lifecycle_active = True
                _field_message_advance_ready = False
                return True
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        _field_message_lifecycle_active = False
        _field_message_advance_ready = False
        return False
    try:
        draw_task_active = task_is_active("Task_DrawFieldMessage")
        script_context = get_global_script_context()
        # The native pointer can remain stale for one or more frames after
        # Emerald stops the global script.  It must not keep an old message
        # lifecycle actionable or cause the next A to be dispatched to the
        # field object in front of the player.
        if script_context is not None and not getattr(script_context, "is_active", False) and not draw_task_active:
            _field_message_lifecycle_active = False
            _field_message_advance_ready = False
            return False
        waiting = is_field_message_waiting_for_input(_field_message_lifecycle_active or draw_task_active)
        # Keep the native name available even when the higher-level waiting
        # predicate is true. The hidden-box callback is a transition inside a
        # field-message lifecycle, not proof that the message accepts input.
        native_name = script_context.native_function_name if script_context is not None else None
        # A field-message task can report input-waiting while its native
        # message box is still hidden. Preserve the lifecycle for ownership,
        # but do not expose that transition as actionable dialogue.
        task_input_ready = is_field_message_task_waiting_for_input()
        hidden_transition = (
            waiting
            and native_name == "IsFieldMessageBoxHidden"
            and _field_message_lifecycle_active
            and not draw_task_active
            and not task_input_ready
        )
        if hidden_transition:
            waiting = False
            native_name = script_context.native_function_name
        if draw_task_active or waiting:
            _field_message_lifecycle_active = True
        else:
            if script_context is None or native_name not in ("WaitForAorBPress", *_FIELD_MESSAGE_TRANSITION_NATIVES):
                _field_message_lifecycle_active = False
        _field_message_advance_ready = (
            _field_message_lifecycle_active
            and not hidden_transition
            and not waiting
            and native_name in _FIELD_MESSAGE_TRANSITION_NATIVES
            and (task_input_ready or draw_task_active)
        )
        if (
            task_input_ready
            and script_context is not None
            and script_context.native_function_name == "IsFieldMessageBoxHidden"
        ):
            _field_message_advance_ready = True
        return waiting
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        _field_message_advance_ready = False
        return False


def classify_interaction(observation: InteractionObservation) -> InteractionType:
    """Classify the most actionable interaction, with specific states first."""

    if observation.game_state in (GameState.BATTLE, GameState.BATTLE_STARTING, GameState.BATTLE_ENDING):
        return InteractionType.BATTLE
    if observation.choice_options or observation.game_state is GameState.CHOOSE_STARTER:
        return InteractionType.CHOICE
    if observation.interaction_phase is InteractionPhase.CHOICE_MENU_INPUT_WAIT:
        return InteractionType.CHOICE
    if observation.interaction_phase is InteractionPhase.SCRIPT_NATIVE_WAIT:
        return InteractionType.UNKNOWN
    if observation.menu_options or observation.game_state in {
        GameState.BAG_MENU,
        GameState.PARTY_MENU,
        GameState.MAIN_MENU,
        GameState.OPTIONS_MENU,
        GameState.POKE_STORAGE,
        GameState.POKEMON_SUMMARY_SCREEN,
    }:
        return InteractionType.MENU
    if observation.special_interaction is not None or observation.game_state in {
        GameState.CHANGE_MAP,
        GameState.NAMING_SCREEN,
        GameState.EVOLUTION,
        GameState.EGG_HATCH,
        GameState.WHITEOUT,
        GameState.QUEST_LOG,
    }:
        return InteractionType.SPECIAL_INTERACTION
    if observation.dialogue_waiting or observation.field_message_lifecycle_active:
        return InteractionType.DIALOGUE
    if observation.game_state is GameState.OVERWORLD:
        return InteractionType.OVERWORLD
    return InteractionType.UNKNOWN


def observe_interaction(
    *,
    choice_options: tuple[str, ...] = (),
    menu_options: tuple[str, ...] = (),
    special_interaction: str | None = None,
) -> InteractionObservation:
    """Build an observation using the existing emulator state APIs.

    Menu parsers and choice-specific readers remain controller-owned for now;
    callers can enrich this observation with their parsed options.
    """

    global _field_message_advance_ready
    interaction_start = now()
    state_start = now()
    state = get_game_state()
    timing("agent_interaction_game_state_read", state_start)
    count("interaction_game_state_reads")
    dialogue_start = now()
    dialogue_waiting = _observe_field_message_waiting(state)
    timing("agent_interaction_dialogue_check", dialogue_start)
    count("interaction_dialogue_checks")
    controllable_start = now()
    controllable = player_avatar_is_controllable()
    script_context = get_global_script_context()
    script_active = bool(script_context is not None and getattr(script_context, "is_active", False))
    native_function = getattr(script_context, "native_function_name", None) if script_context is not None else None
    # Keep the common boundary independent of the nuzlocke package initializer
    # (which imports the campaign controller and would create a cycle here).
    # Task_HandleYesNoInput is Emerald's generic field Yes/No owner; the
    # confirmation observer uses the same task-backed source and additionally
    # supplies cursor/consequence details to callers that need them.
    from modules.nuzlocke.emerald_confirmation import observe_emerald_confirmation

    confirmation = observe_emerald_confirmation()
    choice_menu_active = bool(confirmation is not None and confirmation.active)
    choice_menu_input_ready = bool(confirmation is not None and confirmation.input_ready)
    choice_selected = confirmation.selected.name if confirmation is not None and confirmation.selected else None
    observed_choice_options = choice_options or (confirmation.options if confirmation is not None else ())
    # ``Std_MsgboxYesNo`` first displays an ordinary field message, then
    # installs the Yes/No menu.  Its script identity alone is therefore not
    # an actionable choice boundary: classifying the initial message as a
    # choice would suppress the A input that advances it and prevents the
    # menu task from ever being created.  The generic confirmation observer
    # recognizes ``Task_HandleYesNoInput`` once it exists.  Retain this
    # fallback only for the verified task/printer boundary used by field
    # prompts that expose their ready state through ``Task_DrawFieldMessage``.
    if (
        confirmation is None
        and script_context is not None
        and getattr(script_context, "script_function_name", None) == "Std_MsgboxYesNo"
        and script_active
    ):
        choice_menu_input_ready = is_field_message_task_waiting_for_input()
        if choice_menu_input_ready:
            choice_menu_active = True
            observed_choice_options = choice_options or ("YES", "NO")
            try:
                cursor = read_symbol("sMenu", offset=0x02, size=1)[0]
                choice_selected = ("YES", "NO")[cursor]
            except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
                choice_selected = None
    if choice_menu_active:
        interaction_phase = InteractionPhase.CHOICE_MENU_INPUT_WAIT
    elif (_field_message_advance_ready and script_active and native_function == "IsFieldMessageBoxHidden") or (
        script_active and dialogue_waiting and native_function == "WaitForAorBPress"
    ) or (
        script_active
        and dialogue_waiting
        and native_function == "IsFieldMessageBoxHidden"
    ):
        interaction_phase = InteractionPhase.FIELD_MESSAGE_INPUT_WAIT
    elif _field_message_lifecycle_active and native_function == "IsFieldMessageBoxHidden":
        interaction_phase = InteractionPhase.FIELD_MESSAGE_RENDER_WAIT
    elif script_active:
        interaction_phase = InteractionPhase.SCRIPT_NATIVE_WAIT
    else:
        interaction_phase = InteractionPhase.NONE
    field_message_task_state = None
    field_message_task_active = False
    printer_active = None
    printer_state = None
    try:
        field_message_task = get_task("Task_DrawFieldMessage")
        field_message_task_active = field_message_task is not None
        if field_message_task is not None:
            field_message_task_state = field_message_task.data_value(0)
        if context.rom.is_emerald:
            printer_data = read_symbol("sTextPrinters", offset=0x1B, size=2)
            printer_active, printer_state = printer_data
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    if script_context is not None and getattr(script_context, "script_function_name", None) == "LittlerootTown_ProfessorBirchsLab_EventScript_GiveStarterEvent":
        # GiveStarterEvent uses IsFieldMessageBoxHidden while its printer is
        # still active.  A is intentionally accepted in this state to finish
        # or accelerate the message; the generic hidden-box debounce would
        # otherwise leave the campaign waiting forever on the first lab line.
        if field_message_task_active and printer_active:
            _field_message_advance_ready = True
            interaction_phase = InteractionPhase.FIELD_MESSAGE_INPUT_WAIT
        diagnostic_print(
            lambda: (
                "LAB_DIALOGUE_GATE: "
                f"frame={getattr(context, 'frame', None)!r} "
                f"native={native_function!r} script_active={script_active!r} "
                f"task_active={field_message_task_active!r} task_state={field_message_task_state!r} "
                f"printer_active={printer_active!r} printer_state={printer_state!r} "
                f"waiting_for_input={is_waiting_for_input()!r} "
                f"dialogue_waiting={dialogue_waiting!r} lifecycle={_field_message_lifecycle_active!r} "
                f"advance_ready={_field_message_advance_ready!r} phase={interaction_phase.name!r}"
            ),
            trace=True,
        )
    trace = getattr(context, "stutter_trace", None)
    if trace is not None and callable(getattr(trace, "mark", None)):
        trace.mark("interaction_game_state", getattr(state, "name", repr(state)))
        trace.mark("interaction_dialogue_waiting", dialogue_waiting)
        trace.mark("interaction_field_message_lifecycle", _field_message_lifecycle_active)
        trace.mark("interaction_field_message_ready", _field_message_advance_ready)
        trace.mark("interaction_controllable", controllable)
        trace.mark("interaction_phase", interaction_phase.name)
        trace.mark("interaction_choice_active", choice_menu_active)
        trace.mark("interaction_choice_ready", choice_menu_input_ready)
        trace.mark("interaction_choice_selected", choice_selected)
    timing("agent_interaction_controllability_check", controllable_start)
    count("interaction_controllability_checks")
    timing("agent_interaction_observation", interaction_start)
    count("interaction_observations")
    return InteractionObservation(
        game_state=state,
        dialogue_waiting=dialogue_waiting,
        choice_options=observed_choice_options,
        menu_options=menu_options,
        special_interaction=special_interaction,
        controllable=controllable,
        field_message_lifecycle_active=_field_message_lifecycle_active,
        field_message_advance_ready=_field_message_advance_ready,
        script_active=script_active,
        script_function=(getattr(script_context, "script_function_name", None) if script_context is not None else None),
        native_function=native_function,
        interaction_phase=interaction_phase,
        choice_menu_active=choice_menu_active,
        choice_menu_input_ready=choice_menu_input_ready,
        choice_selected=choice_selected,
        metadata={
            "field_message_task_active": field_message_task_active,
            "field_message_task_state": field_message_task_state,
            "text_printer_active": printer_active,
            "text_printer_state": printer_state,
        },
    )
