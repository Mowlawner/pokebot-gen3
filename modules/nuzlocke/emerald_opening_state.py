"""Compatibility labels for Emerald's legacy opening mode.

The campaign executor does not use these labels to make decisions.  They
remain in a small shared module so the legacy opening mode and older callers
can refer to the same enum without making campaign code import that mode.
"""

from enum import Enum, auto


class OpeningSequenceState(Enum):
    TITLE = auto()
    MAIN_MENU = auto()
    PLAYER_NAMING = auto()
    TRUCK = auto()
    LITTLEROOT_TOWN = auto()
    POST_CLOCK_TOWN = auto()
    PLAYER_HOUSE_2F = auto()
    CLOCK_SETTING = auto()
    PLAYER_HOUSE_1F = auto()
    PLAYER_HOUSE_1F_POST_CLOCK_ARRIVAL = auto()
    BIRCH_HOUSE_1F = auto()
    BIRCH_HOUSE_2F = auto()
    BIRCH_POKEBALL = auto()
    MAY_SEQUENCE = auto()
    ROUTE_101 = auto()
    STARTER_SELECTION = auto()
    SCRIPTED_INTRO = auto()
    UNKNOWN = auto()
    OPTIONS_MENU = auto()
    COMPLETE = auto()


__all__ = ["OpeningSequenceState"]
