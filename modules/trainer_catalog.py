"""ROM-backed Generation III trainer and trainer-party data.

The campaign planner must be able to inspect trainer parties without relying
on a hand-maintained list.  This module is deliberately an adapter around the
ROM layout: parsing is pure, live reads are lazy, and unsupported or malformed
layouts return no record rather than an invented party.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable


TRAINER_RECORD_SIZE = 40
TRAINER_PARTY_FLAGS_MASK = 0x03


@dataclass(frozen=True, slots=True)
class CatalogPokemon:
    species_index: int
    level: int
    held_item_index: int = 0
    moves: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogTrainer:
    trainer_id: int
    name: str
    trainer_class: int
    encounter_music: int
    gender: int
    party_flags: int
    ai_flags: int
    party: tuple[CatalogPokemon, ...]

    @property
    def party_size(self) -> int:
        return len(self.party)


def _u16(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise ValueError("u16 read is outside the supplied ROM bytes")
    return int.from_bytes(data[offset : offset + 2], "little")


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ValueError("u32 read is outside the supplied ROM bytes")
    return int.from_bytes(data[offset : offset + 4], "little")


def _party_entry_size(party_flags: int) -> int:
    """Return the verified Gen III party-entry size for the four layouts.

    The low two trainer-party flags are independent bits: bit 0 selects a
    custom moveset and bit 1 selects a held item.  The default no-item record
    is eight bytes in the supported ROMs; custom moves add four two-byte move
    IDs, and a held item adds one two-byte field.

    In particular, flag 2 means *held item plus default moves*, not custom
    moves.  Treating it as a 14-byte record makes the held-item field and the
    next party bytes look like move IDs.
    """

    layout = party_flags & TRAINER_PARTY_FLAGS_MASK
    return {
        0: 8,   # default moves, no item
        1: 14,  # custom moves, no item
        2: 8,   # default moves, held item
        3: 16,  # custom moves, held item
    }[layout]


def _party_entry_sizes(party_flags: int) -> tuple[int, ...]:
    """Return the primary and compatible party layouts for one flag value.

    ROM revisions and older adapters have disagreed about whether the item
    and custom-move bits are represented by the low flag values in the same
    order. Keep the active adapter's verified layout first, then try the
    other valid Gen III record sizes if that parse is structurally invalid.
    """

    primary = _party_entry_size(party_flags)
    layout = party_flags & TRAINER_PARTY_FLAGS_MASK
    candidates = [primary]
    if layout in (1, 2):
        candidates.extend((14 if primary == 8 else 8,))
    candidates.extend((6, 8, 14, 16))
    return tuple(dict.fromkeys(candidates))


def parse_trainer_catalog(
    trainer_table: bytes,
    *,
    party_reader: Callable[[int, int], bytes],
    name_decoder: Callable[[bytes], str] | None = None,
    max_party_size: int = 6,
) -> tuple[CatalogTrainer | None, ...]:
    """Parse a complete ``gTrainers`` table from supplied ROM bytes.

    The table layout used by the supported R/S/E and FR/LG symbols is a
    40-byte record.  The party pointer and count are checked against the GBA
    ROM address range and every party read is bounds-checked by the caller.
    """

    if not trainer_table or len(trainer_table) % TRAINER_RECORD_SIZE:
        return ()
    if max_party_size < 1 or max_party_size > 6:
        raise ValueError("max_party_size must be between 1 and 6")
    decoder = name_decoder or (lambda value: value.split(b"\xff", 1)[0].decode("ascii", "replace"))
    # Preserve numeric IDs even when one malformed record is rejected.  A
    # compacted tuple would silently change every later trainer's identity.
    records: list[CatalogTrainer | None] = [None] * (len(trainer_table) // TRAINER_RECORD_SIZE)
    for trainer_id, start in enumerate(range(0, len(trainer_table), TRAINER_RECORD_SIZE)):
        record = trainer_table[start : start + TRAINER_RECORD_SIZE]
        try:
            party_flags = record[0]
            trainer_class = record[1]
            encounter_music = record[2]
            gender = record[3]
            # Trainer names occupy 24 bytes in the supported ROM table.  Most
            # names terminate much earlier, but retaining the complete field
            # matters for catalog display and long localized names.
            name = decoder(record[4:28])
            ai_flags = _u32(record, 28)
            party_count = _u32(record, 32)
            party_pointer = _u32(record, 36)
            if party_count == 0:
                party = ()
            elif party_count > max_party_size or not (0x08000000 <= party_pointer < 0x0A000000):
                continue
            else:
                party = None
                for entry_size in _party_entry_sizes(party_flags):
                    raw_party = party_reader(party_pointer, party_count * entry_size)
                    if len(raw_party) != party_count * entry_size:
                        continue
                    parsed: list[CatalogPokemon] = []
                    layout = party_flags & TRAINER_PARTY_FLAGS_MASK
                    has_custom_moves = entry_size in (14, 16)
                    has_held_item = layout in (2, 3) and entry_size in (8, 16)
                    moves_offset = 8 if has_held_item else 6
                    for index in range(party_count):
                        entry = raw_party[index * entry_size : (index + 1) * entry_size]
                        level = _u16(entry, 2)
                        species_index = _u16(entry, 4)
                        held_item = _u16(entry, 6) if has_held_item else 0
                        moves = (
                            tuple(_u16(entry, moves_offset + 2 * move_index) for move_index in range(4))
                            if has_custom_moves
                            else ()
                        )
                        if level < 1 or species_index < 1:
                            parsed = []
                            break
                        parsed.append(CatalogPokemon(species_index, level, held_item, moves))
                    if len(parsed) == party_count:
                        party = tuple(parsed)
                        break
                if party is None:
                    continue
            records[trainer_id] = (
                CatalogTrainer(
                    trainer_id,
                    name,
                    trainer_class,
                    encounter_music,
                    gender,
                    party_flags,
                    ai_flags,
                    party,
                )
            )
        except (IndexError, TypeError, ValueError):
            continue
    return tuple(records)


def resolve_trainerbattle_id(script_bytes: bytes, *, catalog_size: int | None = None) -> int | None:
    """Resolve a direct ``trainerbattle`` script command conservatively.

    Emerald's command is ``0x5c``, followed by a one-byte battle type and a
    little-endian trainer ID. A few tests and older adapters provide the
    compact ``0x5c`` + trainer-ID form, so retain that form as a fallback when
    the full command is not available. Scripts whose first command is not
    ``0x5c`` are intentionally left unresolved; scanning arbitrary script
    bytes could mistake text or command operands for a trainer ID.
    """

    if len(script_bytes) < 3 or script_bytes[0] != 0x5C:
        return None

    # The live Gen III script layout is opcode, battle type, trainer ID.
    # Prefer it whenever the candidate is present and catalog-valid. This
    # matters when the compact candidate formed from bytes 1:3 also happens
    # to look like a valid catalog index.
    if len(script_bytes) >= 4:
        trainer_id = int.from_bytes(script_bytes[2:4], "little")
        if catalog_size is None or 0 <= trainer_id < catalog_size:
            return trainer_id

    # Compatibility with the compact representation used by older callers
    # and fixtures. It is only accepted when it is catalog-valid, so a live
    # command such as ``5c 03 17 02`` cannot be misread as trainer 0x1703.
    trainer_id = int.from_bytes(script_bytes[1:3], "little")
    if catalog_size is not None and not 0 <= trainer_id < catalog_size:
        return None
    return trainer_id


def resolve_trainerbattle_id_from_script(
    script_bytes: bytes, *, catalog_size: int | None = None
) -> int | None:
    """Resolve a trainer battle from a short wrapper script.

    Object events often point at a wrapper that locks/faces the player or
    calls a shared script before executing ``trainerbattle``.  The strict
    resolver above intentionally requires the command at offset zero for
    compact fixtures and direct script reads.  This adapter scans only the
    bounded command window supplied by the caller and delegates each
    candidate to that strict resolver, so arbitrary ROM bytes are never
    interpreted as a trainer command.
    """

    raw = bytes(script_bytes or ())
    for offset, opcode in enumerate(raw):
        if opcode != 0x5C:
            continue
        trainer_id = resolve_trainerbattle_id(raw[offset:], catalog_size=catalog_size)
        if trainer_id is not None:
            return trainer_id
    return None


def _live_read(address: int, size: int) -> bytes:
    from modules.context import context

    emulator = getattr(context, "emulator", None)
    reader = getattr(emulator, "read_bytes", None)
    if not callable(reader):
        raise RuntimeError("the active emulator cannot read ROM bytes")
    try:
        return bytes(reader(address, size))
    except TypeError:
        return bytes(reader(address, length=size))


@lru_cache(maxsize=4)
def _catalog_for_rom(rom_id: str) -> tuple[CatalogTrainer, ...]:
    from modules.game import decode_string
    from modules.memory import read_symbol

    def decode_name(value: bytes) -> str:
        try:
            return decode_string(value)
        except (IndexError, RuntimeError, TypeError, ValueError):
            return value.split(b"\xff", 1)[0].decode("ascii", "replace")

    table = read_symbol("gTrainers")
    return parse_trainer_catalog(
        table,
        party_reader=_live_read,
        name_decoder=decode_name,
    )


def active_trainer_catalog() -> tuple[CatalogTrainer, ...]:
    """Return the active ROM's trainer catalog, or an empty catalog safely."""

    from modules.context import context

    rom = getattr(context, "rom", None)
    if rom is None:
        return ()
    try:
        return _catalog_for_rom(getattr(rom, "id", repr(rom)))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        from modules.console import diagnostic_print

        diagnostic_print(
            lambda: (
                "TRAINER_CATALOG_LIVE: "
                f"result='exception' rom={getattr(rom, 'id', repr(rom))!r} "
                f"exception_type={type(error).__name__!r} exception={error!r}"
            ),
            trace=True,
            prefix="TRAINER_CATALOG_LIVE",
        )
        return ()


def invalidate_trainer_catalog_cache() -> None:
    _catalog_for_rom.cache_clear()


__all__ = [
    "CatalogPokemon",
    "CatalogTrainer",
    "TRAINER_RECORD_SIZE",
    "active_trainer_catalog",
    "invalidate_trainer_catalog_cache",
    "parse_trainer_catalog",
    "resolve_trainerbattle_id",
    "resolve_trainerbattle_id_from_script",
]
