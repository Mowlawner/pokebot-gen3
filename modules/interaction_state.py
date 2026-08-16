"""State-aware, read-only description of the game's current interaction.

This module deliberately does not execute input.  It provides a stable boundary
between emulator-specific observations and future goal/action controllers.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping

from modules.memory import GameState, get_game_state
from modules.player import player_avatar_is_controllable
from modules.tasks import get_global_script_context, is_field_message_waiting_for_input, task_is_active
from modules.profiler import count, now, timing


class InteractionType(Enum):
    DIALOGUE = auto()
    CHOICE = auto()
    MENU = auto()
    BATTLE = auto()
    OVERWORLD = auto()
    SPECIAL_INTERACTION = auto()
    UNKNOWN = auto()


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
    metadata: Mapping[str, Any] = field(default_factory=dict)


_field_message_lifecycle_active = False
_field_message_advance_ready = False

_FIELD_MESSAGE_TRANSITION_NATIVES = frozenset({"IsFieldMessageBoxHidden"})


def _observe_field_message_waiting(state: GameState | Any) -> bool:
    """Carry Emerald's field-message lifecycle across script native waits."""
    global _field_message_lifecycle_active, _field_message_advance_ready
    if state is not GameState.OVERWORLD:
        _field_message_lifecycle_active = False
        _field_message_advance_ready = False
        return False
    try:
        draw_task_active = task_is_active("Task_DrawFieldMessage")
        waiting = is_field_message_waiting_for_input(_field_message_lifecycle_active or draw_task_active)
        script_context = None
        native_name = None
        if not waiting:
            script_context = get_global_script_context()
            native_name = script_context.native_function_name if script_context is not None else None
        if draw_task_active or waiting:
            _field_message_lifecycle_active = True
        else:
            if script_context is None or native_name not in ("WaitForAorBPress", *_FIELD_MESSAGE_TRANSITION_NATIVES):
                _field_message_lifecycle_active = False
        _field_message_advance_ready = (
            _field_message_lifecycle_active and not waiting and native_name in _FIELD_MESSAGE_TRANSITION_NATIVES
        )
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
    if observation.dialogue_waiting:
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
    timing("agent_interaction_controllability_check", controllable_start)
    count("interaction_controllability_checks")
    timing("agent_interaction_observation", interaction_start)
    count("interaction_observations")
    return InteractionObservation(
        game_state=state,
        dialogue_waiting=dialogue_waiting,
        choice_options=choice_options,
        menu_options=menu_options,
        special_interaction=special_interaction,
        controllable=controllable,
        field_message_lifecycle_active=_field_message_lifecycle_active,
        field_message_advance_ready=_field_message_advance_ready,
    )
