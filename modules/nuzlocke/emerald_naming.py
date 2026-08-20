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
from modules.memory import unpack_uint16
from modules.pokemon import get_species_by_index


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
    species_id: int | None = None
    species_name: str | None = None
    pokemon_gender: str | None = None
    personality_value: int | None = None


# NamingScreenData: tilemap[0x1800], text[0x10], tiles[0x600], then fields.
_TEMPLATE_NUMBER_OFFSET = 0x1E2C
_MON_SPECIES_OFFSET = 0x1E34
_MON_GENDER_OFFSET = 0x1E36
_MON_PERSONALITY_OFFSET = 0x1E38
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
        target = _TEMPLATES.get(value, EmeraldNamingTarget.UNKNOWN)
        species_id = None
        species_name = None
        pokemon_gender = None
        personality = None
        if target in {EmeraldNamingTarget.POKEMON_NICKNAME, EmeraldNamingTarget.CAUGHT_POKEMON_NICKNAME}:
            species_id = unpack_uint16(_read_dynamic_bytes(pointer, _MON_SPECIES_OFFSET, 2))
            gender_value = unpack_uint16(_read_dynamic_bytes(pointer, _MON_GENDER_OFFSET, 2))
            personality = unpack_uint32(_read_dynamic_bytes(pointer, _MON_PERSONALITY_OFFSET, 4))
            species_name = get_species_by_index(species_id).name
            pokemon_gender = {0: "male", 254: "female", 255: None}.get(gender_value)
        return EmeraldNamingObservation(
            target,
            value,
            pointer,
            get_naming_screen_data() is not None,
            species_id,
            species_name,
            pokemon_gender,
            personality,
        )
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return EmeraldNamingObservation(EmeraldNamingTarget.UNKNOWN, None, None, False)


def _read_dynamic_byte(pointer: int, offset: int) -> int:
    # Kept separate so tests can assert the exact decomp-derived address.
    from modules.context import context

    return context.emulator.read_bytes(pointer + offset, 1)[0]


def _read_dynamic_bytes(pointer: int, offset: int, size: int) -> bytes:
    from modules.context import context

    return context.emulator.read_bytes(pointer + offset, size)


__all__ = ["EmeraldNamingObservation", "EmeraldNamingTarget", "observe_emerald_naming"]
