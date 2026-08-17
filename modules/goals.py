"""Small, composable goal descriptions for lower-level controllers."""

from dataclasses import dataclass, field

from modules.overworld import Location, MapId, WarpObservation
from enum import Enum, auto


class EncounterMode(Enum):
    NORMAL = auto()
    SEEK = auto()


INTRODUCTORY_RIVAL_TRIGGER_ID = "introductory_rival"
EARLY_POKEBALL_TRIGGER_ID = "early_pokeballs"


class Goal:
    """Marker base class for objectives supplied by a higher-level planner."""


@dataclass(frozen=True)
class ReachLocation(Goal):
    location: Location


@dataclass(frozen=True)
class ReachWarp(Goal):
    destination_map: MapId | None = None
    destination: Location | None = None
    # The destination map is only a routing hint; preserve the observed
    # source record so equal-destination warps remain distinguishable.
    warp: WarpObservation | None = None


@dataclass(frozen=True)
class ActivateTrigger(Goal):
    trigger_id: str


def introductory_rival_goal() -> ActivateTrigger:
    """Return the generic trigger goal used by the Emerald pilot."""
    return ActivateTrigger(INTRODUCTORY_RIVAL_TRIGGER_ID)


def early_pokeball_goal() -> ActivateTrigger:
    """Return the Emerald Professor Birch Poké Ball interaction goal."""
    return ActivateTrigger(EARLY_POKEBALL_TRIGGER_ID)


@dataclass(frozen=True)
class ReachInteractionPosition(Goal):
    trigger_id: str


@dataclass(frozen=True)
class GoalConstraints:
    avoid_trigger_ids: frozenset[str] = frozenset()
    avoid_locations: frozenset[Location] = frozenset()


@dataclass(frozen=True)
class NavigationGoal(Goal):
    target: Goal
    constraints: GoalConstraints = field(default_factory=GoalConstraints)
    encounter_mode: EncounterMode = EncounterMode.NORMAL
