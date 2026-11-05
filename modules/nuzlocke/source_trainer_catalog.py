"""Static trainer roster extraction from a pret-style Gen III source tree."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .gate_discovery import SourceLocation


_TRAINER_BLOCK_RE = re.compile(r"\[\s*(TRAINER_[A-Za-z0-9_]+)\s*\]\s*=\s*\{")
_PARTY_BLOCK_RE = re.compile(
    r"static\s+const\s+struct\s+TrainerMon[A-Za-z0-9_]*\s+(sParty_[A-Za-z0-9_]+)\[\]\s*=\s*\{"
)
_TRAINER_CONSTANT_RE = re.compile(r"^\s*#define\s+(TRAINER_[A-Za-z0-9_]+)\s+([0-9]+)\s*$", re.MULTILINE)
_FIELD_RE = re.compile(r"\.(?P<field>lvl|species|moves)\s*=\s*(?P<value>[^,}\n]+|\{[^}]*\})")


@dataclass(frozen=True, slots=True)
class SourceTrainerPokemon:
    species: str
    level: str
    moves: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceTrainerRecord:
    trainer_symbol: str
    trainer_id: int | None
    display_name: str | None
    party_symbol: str | None
    party: tuple[SourceTrainerPokemon, ...]
    source: SourceLocation
    party_source: SourceLocation | None = None
    double_battle: bool | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["party"] = [asdict(pokemon) for pokemon in self.party]
        return result


def _matching_brace(text: str, opening: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(text)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _parse_party_blocks(path: Path, root: Path) -> dict[str, tuple[tuple[SourceTrainerPokemon, ...], SourceLocation]]:
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(root).as_posix()
    parties: dict[str, tuple[tuple[SourceTrainerPokemon, ...], SourceLocation]] = {}
    for match in _PARTY_BLOCK_RE.finditer(text):
        opening = text.find("{", match.start(), match.end())
        closing = _matching_brace(text, opening)
        body = text[opening + 1 : closing]
        level_matches = list(re.finditer(r"\.lvl\s*=\s*([^,}\n]+)", body))
        pokemon: list[SourceTrainerPokemon] = []
        for index, level_match in enumerate(level_matches):
            end = level_matches[index + 1].start() if index + 1 < len(level_matches) else len(body)
            entry = body[level_match.start() : end]
            species_match = re.search(r"\.species\s*=\s*([^,}\n]+)", entry)
            if species_match is None:
                continue
            moves_match = re.search(r"\.moves\s*=\s*\{([^}]*)\}", entry, re.DOTALL)
            moves = ()
            if moves_match:
                moves = tuple(
                    move.strip()
                    for move in moves_match.group(1).split(",")
                    if move.strip()
                )
            pokemon.append(
                SourceTrainerPokemon(
                    species=species_match.group(1).strip(),
                    level=level_match.group(1).strip(),
                    moves=moves,
                )
            )
        parties[match.group(1)] = (
            tuple(pokemon),
            SourceLocation(relative, _line_number(text, match.start()), match.group(1)),
        )
    return parties


def parse_source_trainer_catalog(source_root: str | Path) -> tuple[SourceTrainerRecord, ...]:
    """Parse source trainer identities and rosters in stable symbol order."""

    root = Path(source_root).expanduser().resolve()
    constants_path = root / "include" / "constants" / "opponents.h"
    trainers_path = root / "src" / "data" / "trainers.h"
    parties_path = root / "src" / "data" / "trainer_parties.h"
    if not trainers_path.exists():
        return ()
    constants_text = constants_path.read_text(encoding="utf-8") if constants_path.exists() else ""
    trainer_ids = {symbol: int(value) for symbol, value in _TRAINER_CONSTANT_RE.findall(constants_text)}
    parties = _parse_party_blocks(parties_path, root) if parties_path.exists() else {}
    trainers_text = trainers_path.read_text(encoding="utf-8")
    relative = trainers_path.relative_to(root).as_posix()
    records: list[SourceTrainerRecord] = []
    for match in _TRAINER_BLOCK_RE.finditer(trainers_text):
        opening = trainers_text.find("{", match.start(), match.end())
        closing = _matching_brace(trainers_text, opening)
        block = trainers_text[opening + 1 : closing]
        name_match = re.search(r"\.trainerName\s*=\s*_\(\"([^\"]*)\"\)", block)
        party_match = re.search(r"\.party\s*=\s*[A-Z_]+\((sParty_[A-Za-z0-9_]+)\)", block)
        double_match = re.search(r"\.doubleBattle\s*=\s*(TRUE|FALSE)", block)
        party_symbol = party_match.group(1) if party_match else None
        party, party_source = parties.get(party_symbol, ((), None)) if party_symbol else ((), None)
        records.append(
            SourceTrainerRecord(
                trainer_symbol=match.group(1),
                trainer_id=trainer_ids.get(match.group(1)),
                display_name=name_match.group(1) if name_match else None,
                party_symbol=party_symbol,
                party=party,
                source=SourceLocation(relative, _line_number(trainers_text, match.start()), match.group(1)),
                party_source=party_source,
                double_battle=(double_match.group(1) == "TRUE") if double_match else None,
            )
        )
    return tuple(sorted(records, key=lambda record: record.trainer_symbol))


def trainer_record_index(records: tuple[SourceTrainerRecord, ...]) -> Mapping[str, SourceTrainerRecord]:
    return {record.trainer_symbol: record for record in records}


def trainer_metadata(record: SourceTrainerRecord | None) -> dict[str, str]:
    """Return compact metadata suitable for a GateCandidate record."""

    if record is None:
        return {"roster_resolution": "unknown"}
    return {
        "roster_resolution": "resolved" if record.party else "unresolved",
        "trainer_id": str(record.trainer_id) if record.trainer_id is not None else "unknown",
        "trainer_name": record.display_name or "unknown",
        "party_symbol": record.party_symbol or "unknown",
        "party": json.dumps([asdict(pokemon) for pokemon in record.party], sort_keys=True, separators=(",", ":")),
        "double_battle": str(record.double_battle).lower() if record.double_battle is not None else "unknown",
        "trainer_source": record.source.path + ":" + str(record.source.line),
        "party_source": (
            record.party_source.path + ":" + str(record.party_source.line)
            if record.party_source is not None
            else "unknown"
        ),
    }


__all__ = [
    "SourceTrainerPokemon",
    "SourceTrainerRecord",
    "parse_source_trainer_catalog",
    "trainer_metadata",
    "trainer_record_index",
]
