"""State-aware, read-only description of the game's current interaction.

This module deliberately does not execute input.  It provides a stable boundary
between emulator-specific observations and future goal/action controllers.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping

from modules.memory import GameState, get_game_state
from modules.player import player_avatar_is_controllable
from modules.tasks import is_field_message_waiting_for_input


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
    metadata: Mapping[str, Any] = field(default_factory=dict)


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

    state = get_game_state()
    return InteractionObservation(
        game_state=state,
        dialogue_waiting=state is GameState.OVERWORLD and is_field_message_waiting_for_input(),
        choice_options=choice_options,
        menu_options=menu_options,
        special_interaction=special_interaction,
        controllable=player_avatar_is_controllable(),
    )
