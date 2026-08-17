"""Authoritative observations of Emerald's generic naming screen.

Emerald stores the ``DoNamingScreen`` template number in the live
``sNamingScreen`` object.  This is the ROM's context, unlike the generic
``GameState.NAMING_SCREEN`` classification which has no target information.
The offsets below are derived from ``struct NamingScreenData`` in
pokeemerald's ``src/naming_screen.c``. The GBA build aligns the pointer
member after ``keyRepeatStartDelayCopy`` to four bytes, adding two padding
bytes; the template number is therefore at 0x1e2c, not the packed offset
0x1e2a.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from modules.keyboard import get_naming_screen_data
from modules.memory import GameState, get_game_state, read_symbol, unpack_uint32


class EmeraldNamingTarget(Enum):
    PLAYER_NAME = auto()
    POKEMON_NICKNAME = auto()
    CAUGHT_POKEMON_NICKNAME = auto()
    BOX_NAME = auto()
    WALDA_PHRASE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class EmeraldNamingObservation:
    target: EmeraldNamingTarget
    template_number: int | None
    screen_pointer: int | None
    keyboard_ready: bool


# NamingScreenData: tilemap[0x1800], text[0x10], tiles[0x600], then fields.
_TEMPLATE_NUMBER_OFFSET = 0x1E2C
_TEMPLATES = {
    0: EmeraldNamingTarget.PLAYER_NAME,
    1: EmeraldNamingTarget.BOX_NAME,
    2: EmeraldNamingTarget.CAUGHT_POKEMON_NICKNAME,
    3: EmeraldNamingTarget.POKEMON_NICKNAME,
    4: EmeraldNamingTarget.WALDA_PHRASE,
}


def observe_emerald_naming() -> EmeraldNamingObservation | None:
    """Observe the active naming target, or ``None`` outside the naming UI."""
    if get_game_state() is not GameState.NAMING_SCREEN:
        return None
    try:
        pointer = unpack_uint32(read_symbol("sNamingScreen", size=4))
        if pointer == 0:
            return EmeraldNamingObservation(EmeraldNamingTarget.UNKNOWN, None, None, False)
        value = _read_dynamic_byte(pointer, _TEMPLATE_NUMBER_OFFSET)
        return EmeraldNamingObservation(
            _TEMPLATES.get(value, EmeraldNamingTarget.UNKNOWN),
            value,
            pointer,
            get_naming_screen_data() is not None,
        )
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return EmeraldNamingObservation(EmeraldNamingTarget.UNKNOWN, None, None, False)


def _read_dynamic_byte(pointer: int, offset: int) -> int:
    # Kept separate so tests can assert the exact decomp-derived address.
    from modules.context import context

    return context.emulator.read_bytes(pointer + offset, 1)[0]


__all__ = ["EmeraldNamingObservation", "EmeraldNamingTarget", "observe_emerald_naming"]
