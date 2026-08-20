"""Immutable, frame-local Emerald observations.

This is the boundary between ROM perception and campaign policy.  Nothing in
this module represents an execution phase or an in-flight transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.memory import GameState
from modules.overworld import OverworldObservation
from modules.goals import SemanticTarget

from .emerald_confirmation import EmeraldConfirmationObservation
from .emerald_menu_observation import EmeraldMenuObservation
from .emerald_naming import EmeraldNamingObservation
from .emerald_starter_selection import EmeraldStarterSelectionObservation


@dataclass(frozen=True, slots=True)
class EmeraldObservation:
    """All semantic inputs available to the Emerald action dispatcher."""

    game_state: GameState | None
    ui_mode: str
    map_id: tuple[int, int] | None
    coordinates: tuple[int, int] | None
    controllable: bool
    dialogue_actionable: bool
    dialogue_lifecycle_active: bool
    menu: EmeraldMenuObservation | None
    naming: EmeraldNamingObservation | None
    confirmation: EmeraldConfirmationObservation | None
    gender_task: str | None
    rom_owned_movement: bool
    starter_selection: EmeraldStarterSelectionObservation | None = None
    title_actionable: bool = False
    startup_tasks: tuple[str, ...] = ()
    native_task_state: tuple[str | None, str | None, bool | None] = (None, None, None)
    script_stack: tuple[str, ...] = ()
    # (task, hand_angle, hour, minute, cursor), all read from the current ROM frame.
    clock_interaction: tuple[str, int | None, int | None, int | None, int | None] | None = None
    # The player tile/facing required to activate the observed wall-clock
    # event. This is a derived affordance, not an execution transaction.
    clock_target: tuple[tuple[int, int], str] | None = None
    objective_id: str | None = None
    semantic_target: SemanticTarget | None = None
    campaign_facts: tuple[tuple[str, object], ...] = ()
    overworld: OverworldObservation | None = None

    @property
    def actionable_dialogue(self) -> bool:
        """Name used by policy code; kept distinct from lifecycle activity."""
        return self.dialogue_actionable


__all__ = ["EmeraldObservation"]
