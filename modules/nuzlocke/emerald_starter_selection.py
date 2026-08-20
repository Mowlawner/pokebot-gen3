"""ROM-backed observation of Emerald's Birch-bag starter chooser."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from modules.context import context
from modules.map_data import MapRSE
from modules.memory import GameState, get_game_state
from modules.player import get_player_avatar
from modules.tasks import get_task
from typing import Protocol


HOENN_STARTER_CHOICES = ("Treecko", "Torchic", "Mudkip")


class StarterRandomSource(Protocol):
    def choice(self, sequence): ...


class EmeraldStarterSelectionPhase(Enum):
    CHOOSING = auto()
    ANIMATING = auto()
    CONFIRMING = auto()


@dataclass(frozen=True, slots=True)
class EmeraldStarterSelectionObservation:
    choices: tuple[str, str, str]
    selected_index: int
    phase: EmeraldStarterSelectionPhase
    input_ready: bool

    @property
    def selected_species(self) -> str:
        return self.choices[self.selected_index]


def configured_emerald_starter() -> str:
    """Return the configured Hoenn starter, rejecting unsafe values clearly."""
    configured = getattr(getattr(getattr(context, "config", None), "start_game", None), "starter", None)
    if configured not in (*HOENN_STARTER_CHOICES, "Random"):
        raise ValueError(
            "start_game.starter must be Treecko, Torchic, Mudkip, or Random "
            f"for Emerald campaign progression; got {configured!r}"
        )
    return configured


def resolve_emerald_starter(configured: str, rng: StarterRandomSource) -> str:
    """Resolve one concrete starter for one selection session."""
    if configured == "Random":
        return rng.choice(HOENN_STARTER_CHOICES)
    if configured not in HOENN_STARTER_CHOICES:
        raise ValueError(f"Invalid configured Emerald starter: {configured!r}")
    return configured


def observe_emerald_starter_selection() -> EmeraldStarterSelectionObservation | None:
    """Observe only the ROM's active starter chooser and its current cursor."""
    try:
        if get_game_state() is not GameState.CHOOSE_STARTER:
            return None
        if get_player_avatar().map_group_and_number != MapRSE.ROUTE101.value:
            return None
        for task_name, phase, ready in (
            ("Task_HandleStarterChooseInput", EmeraldStarterSelectionPhase.CHOOSING, True),
            ("Task_HandleConfirmStarterInput", EmeraldStarterSelectionPhase.CONFIRMING, True),
            ("Task_StarterChoose", EmeraldStarterSelectionPhase.ANIMATING, False),
            ("Task_MoveStarterChooseCursor", EmeraldStarterSelectionPhase.ANIMATING, False),
            ("Task_CreateStarterLabel", EmeraldStarterSelectionPhase.ANIMATING, False),
            ("Task_WaitForStarterSprite", EmeraldStarterSelectionPhase.ANIMATING, False),
            ("Task_AskConfirmStarter", EmeraldStarterSelectionPhase.ANIMATING, False),
        ):
            task = get_task(task_name)
            if task is None:
                continue
            selected = task.data_value(0)
            if selected not in range(len(HOENN_STARTER_CHOICES)):
                return None
            return EmeraldStarterSelectionObservation(HOENN_STARTER_CHOICES, selected, phase, ready)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
        return None
    return None


__all__ = [
    "HOENN_STARTER_CHOICES",
    "EmeraldStarterSelectionObservation",
    "EmeraldStarterSelectionPhase",
    "configured_emerald_starter",
    "resolve_emerald_starter",
    "observe_emerald_starter_selection",
]
