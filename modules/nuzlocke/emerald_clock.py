"""ROM-backed Emerald clock observations and pure clock policy helpers.

This module is shared by the campaign capability and the legacy opening mode
without importing either one.  The campaign may ask for a clock affordance or
an input direction, but it never needs to know which opening phase produced
the current observation.
"""

from __future__ import annotations

from datetime import datetime
import random

from modules.config.schemas_v1 import WallClockTimeMode
from modules.context import context
from modules.map import get_map_data_for_current_position
from modules.map_data import MapRSE
from modules.memory import get_event_flag, get_game_state, GameState
from modules.player import get_player, get_player_avatar
from modules.start_game import RandomSource
from modules.tasks import get_task, task_is_active


_CLOCK_TASKS = (
    "Task_SetClock_WaitFadeIn",
    "Task_SetClock_HandleInput",
    "Task_SetClock_AskConfirm",
    "Task_SetClock_HandleConfirmInput",
    "Task_SetClock_Confirmed",
    "Task_SetClock_Exit",
)
_CLOCK_HOUR_MIN = 0
_CLOCK_HOUR_MAX = 23
_CLOCK_MINUTE_MIN = 0
_CLOCK_MINUTE_MAX = 59
_CLOCK_MINUTES_PER_HOUR = 60
_CLOCK_PERIOD_MINUTES = (_CLOCK_HOUR_MAX + 1) * _CLOCK_MINUTES_PER_HOUR

_EMERALD_WALL_CLOCK_INTERACTIONS = {
    # Brendan's house 2F: the player stands below the clock and faces up.
    MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value: ((5, 2), "Up"),
}
_EMERALD_PLAYER_HOUSE_CLOCK_INTERACTIONS = {
    # May's house has a different layout.
    "male": ((5, 2), "Up"),
    "female": ((3, 2), "Up"),
}


def _current_map_id() -> tuple[int, int] | None:
    """Read the current map identity, returning none during transitions."""

    try:
        location = get_map_data_for_current_position()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None
    return None if location is None else location.map_group_and_number


def _opening_player_gender(player_gender: object | None = None) -> str:
    """Resolve protagonist gender from an explicit value or ROM player data."""

    value = getattr(player_gender, "value", player_gender)
    if value in ("male", "female"):
        return value
    try:
        value = get_player().gender
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        value = "male"
    return value if value in ("male", "female") else "male"


def _littleroot_house_maps(
    player_gender: object | None = None,
) -> tuple[MapRSE, MapRSE, MapRSE, MapRSE]:
    """Return player/rival 1F and 2F maps for the selected protagonist."""
    if _opening_player_gender(player_gender) == "female":
        return (
            MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F,
            MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F,
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F,
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
        )
    return (
        MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F,
        MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
        MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F,
        MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F,
    )


def player_house_map(floor: int, player_gender: object | None = None) -> MapRSE:
    """Return the player's Littleroot house map for a one-based floor."""

    return _littleroot_house_maps(player_gender)[floor - 1]


def rival_house_map(floor: int, player_gender: object | None = None) -> MapRSE:
    """Return the rival's Littleroot house map for a one-based floor."""

    return _littleroot_house_maps(player_gender)[floor + 1]


def active_clock_task() -> str | None:
    """Return the active Emerald wall-clock task, if one exists."""
    try:
        return next((task for task in _CLOCK_TASKS if task_is_active(task)), None)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return None


def clock_input_direction(
    current_hour: int,
    current_minute: int,
    target_hour: int,
    target_minute: int,
) -> str:
    """Choose the shortest direction around Emerald's 24-hour clock."""
    current = current_hour * _CLOCK_MINUTES_PER_HOUR + current_minute
    target = target_hour * _CLOCK_MINUTES_PER_HOUR + target_minute
    forward = (target - current) % _CLOCK_PERIOD_MINUTES
    return "Right" if forward <= _CLOCK_PERIOD_MINUTES // 2 else "Left"


def clock_time_mode() -> WallClockTimeMode:
    """Return the configured wall-clock selection mode with a safe fallback."""

    config = getattr(context, "config", None)
    start_game = getattr(config, "start_game", None)
    configured = getattr(start_game, "clock_time_mode", WallClockTimeMode.SYSTEM_TIME)
    try:
        return configured if isinstance(configured, WallClockTimeMode) else WallClockTimeMode(configured)
    except (TypeError, ValueError):
        return WallClockTimeMode.SYSTEM_TIME


def emerald_clock_time(
    mode: WallClockTimeMode,
    *,
    now: datetime | None = None,
    rng: RandomSource | None = None,
) -> tuple[int, int]:
    """Resolve one target in Emerald's internal 24-hour representation."""
    if mode is WallClockTimeMode.RANDOM:
        if rng is None:
            rng = random.Random()
        return (
            rng.randint(_CLOCK_HOUR_MIN, _CLOCK_HOUR_MAX),
            rng.randint(_CLOCK_MINUTE_MIN, _CLOCK_MINUTE_MAX),
        )
    snapshot = datetime.now() if now is None else now
    return snapshot.hour, snapshot.minute


def wall_clock_interaction(
    map_id: tuple[int, int] | None = None,
    player_gender: object | None = None,
) -> tuple[tuple[int, int], str] | None:
    """Return the current player's physical wall-clock approach affordance."""
    if map_id is None:
        map_id = _current_map_id()
    gender = _opening_player_gender(player_gender)
    if map_id != player_house_map(2, gender).value:
        return None

    target = _EMERALD_WALL_CLOCK_INTERACTIONS.get(map_id)
    if target is not None:
        return target

    # Resolve the alternate house layout from the map's own BG-event metadata.
    try:
        location = get_map_data_for_current_position()
        if location is not None:
            clock = next(
                (
                    event
                    for event in location.bg_events
                    if event.kind == "Script"
                    and event.script_symbol is not None
                    and event.script_symbol.endswith("_EventScript_WallClock")
                ),
                None,
            )
            if clock is not None:
                offsets = {
                    "Up": ((0, 1), "Up"),
                    "Down": ((0, -1), "Down"),
                    "Left": ((1, 0), "Left"),
                    "Right": ((-1, 0), "Right"),
                }
                offset_and_direction = offsets.get(clock.player_facing_direction)
                if offset_and_direction is not None:
                    (offset_x, offset_y), direction = offset_and_direction
                    clock_x, clock_y = clock.local_coordinates
                    return (clock_x + offset_x, clock_y + offset_y), direction
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass

    return _EMERALD_PLAYER_HOUSE_CLOCK_INTERACTIONS.get(gender)


__all__ = [
    "active_clock_task",
    "clock_input_direction",
    "clock_time_mode",
    "emerald_clock_time",
    "player_house_map",
    "rival_house_map",
    "wall_clock_interaction",
]
