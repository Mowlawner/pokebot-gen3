"""Source-first discovery of campaign gate candidates.

This module deliberately stops at evidence collection.  It does not decide
that a flag is a campaign fact, and it does not modify the Emerald registry.
The source tree is treated as an immutable input and the report is canonical
JSON, so a later route-analysis or live-observation layer can consume it
without repeating source parsing.

The first adapter targets the current ``pret/pokeemerald`` source layout:
``data/maps/*/map.json`` and the corresponding ``scripts.inc`` files.  The
parser recognizes common event macros and records unsupported comparisons as
unknown candidates instead of treating them as open or closed gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
SCANNER_VERSION = "emerald-source-1"

_LABEL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*::?\s*$")
_COMMAND_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\s+(.*?))?\s*$")
_MAP_DIR_RE = re.compile(r"^data/maps/([^/]+)/map\.json$")


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A stable source reference retained in every candidate."""

    path: str
    line: int
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class ProducerAction:
    """A script action that may satisfy a candidate guard."""

    action: str
    symbol: str
    value: str | None
    source: SourceLocation


@dataclass(frozen=True, slots=True)
class RouteEvidence:
    """Reserved route-analysis fields, unknown until graph analysis runs."""

    blocked_routes: tuple[str, ...] = ()
    alternate_routes: tuple[str, ...] = ()
    cut_set_confidence: str = "unknown"
    notes: tuple[str, ...] = ()
    # Evidence owners may attach stable observation references here without
    # making the source scanner guess how a script changes map traversal.
    evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class GateCandidate:
    """Immutable, source-backed evidence for a possible campaign gate."""

    candidate_id: str
    game: str
    source_revision: str
    kind: str
    source: SourceLocation
    guard_expression: str | None = None
    guard_symbol: str | None = None
    guard_value: str | None = None
    affected_maps: tuple[str, ...] = ()
    affected_objects: tuple[str, ...] = ()
    producer_actions: tuple[ProducerAction, ...] = ()
    current_state: str = "unknown"
    route_evidence: RouteEvidence = field(default_factory=RouteEvidence)
    semantic_confidence: str = "low"
    verification_status: str = "candidate"
    diagnostics: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        """Return JSON-ready data with tuples normalized to lists."""

        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class GateDiscoveryReport:
    """Canonical output of one source scan."""

    schema_version: int
    scanner_version: str
    game: str
    source_root: str
    source_revision: str
    candidates: tuple[GateCandidate, ...]
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "scanner_version": self.scanner_version,
            "game": self.game,
            "source_root": self.source_root,
            "source_revision": self.source_revision,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "diagnostics": list(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _ScriptCommand:
    name: str
    arguments: tuple[str, ...]
    raw: str
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class _MapInput:
    map_id: str
    map_name: str
    json_path: Path
    json_relative_path: str
    raw_json: str
    data: Mapping
    scripts: tuple[_ScriptCommand, ...]
    labels: Mapping[str, tuple[_ScriptCommand, ...]]


def _json_value(value):
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _canonical(value) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def _split_arguments(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    # Script arguments are identifiers and literals in the current adapter.
    # Keeping commas inside quoted strings intact costs little and makes the
    # fixture parser useful for future text-bearing commands.
    result: list[str] = []
    current: list[str] = []
    quoted = False
    for char in text:
        if char == '"':
            quoted = not quoted
        if char == "," and not quoted:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    result.append("".join(current).strip())
    return tuple(argument for argument in result if argument)


def _strip_comment(line: str) -> str:
    return line.split("@", 1)[0].rstrip()


def _parse_script(path: Path, root: Path) -> tuple[tuple[_ScriptCommand, ...], dict[str, tuple[_ScriptCommand, ...]]]:
    relative = path.relative_to(root).as_posix()
    all_commands: list[_ScriptCommand] = []
    by_label: dict[str, list[_ScriptCommand]] = {}
    current_label: str | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        label_match = _LABEL_RE.match(line)
        if label_match:
            current_label = label_match.group(1)
            by_label.setdefault(current_label, [])
            continue
        command_match = _COMMAND_RE.match(line)
        if not command_match:
            continue
        command = _ScriptCommand(
            name=command_match.group(1),
            arguments=_split_arguments(command_match.group(2)),
            raw=line,
            location=SourceLocation(relative, line_number, current_label),
        )
        all_commands.append(command)
        if current_label is not None:
            by_label[current_label].append(command)
    return tuple(all_commands), {label: tuple(commands) for label, commands in by_label.items()}


def _source_line(raw: str, value: str, start: int = 0) -> int:
    index = raw.find(value, start)
    return raw.count("\n", 0, index if index >= 0 else len(raw)) + 1


def _git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"
    revision = completed.stdout.strip()
    return revision or "unversioned"


def _map_inputs(source_root: Path) -> tuple[_MapInput, ...]:
    inputs: list[_MapInput] = []
    for json_path in sorted(source_root.glob("data/maps/*/map.json"), key=lambda path: path.as_posix()):
        relative = json_path.relative_to(source_root).as_posix()
        raw = json_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        scripts_path = json_path.with_name("scripts.inc")
        if scripts_path.exists():
            commands, labels = _parse_script(scripts_path, source_root)
        else:
            commands, labels = (), {}
        inputs.append(
            _MapInput(
                map_id=str(data.get("id", json_path.parent.name)),
                map_name=str(data.get("name", json_path.parent.name)),
                json_path=json_path,
                json_relative_path=relative,
                raw_json=raw,
                data=data,
                scripts=commands,
                labels=labels,
            )
        )
    return tuple(inputs)


def _map_for_script(path: str) -> str | None:
    match = _MAP_DIR_RE.match(path)
    return match.group(1) if match else None


def _guard_producers(
    inputs: Sequence[_MapInput],
    extra_commands: Sequence[_ScriptCommand] = (),
) -> dict[str, tuple[ProducerAction, ...]]:
    producers: dict[str, list[ProducerAction]] = {}
    commands = [command for map_input in inputs for command in map_input.scripts]
    commands.extend(extra_commands)
    for command in commands:
            symbol: str | None = None
            value: str | None = None
            action: str | None = None
            if command.name in {"setflag", "clearflag"} and command.arguments:
                symbol, action = command.arguments[0], command.name
            elif command.name in {"setvar", "addvar", "subvar", "copyvar"} and command.arguments:
                symbol, action = command.arguments[0], command.name
                if len(command.arguments) > 1:
                    value = command.arguments[1]
            if symbol and action:
                producers.setdefault(symbol, []).append(
                    ProducerAction(action, command.location.symbol or "<anonymous>", value, command.location)
                )
    return {
        symbol: tuple(sorted(actions, key=lambda action: _canonical(action)))
        for symbol, actions in sorted(producers.items())
    }


def _shared_script_commands(root: Path) -> tuple[_ScriptCommand, ...]:
    """Read common script files so their writes can satisfy map guards."""

    commands: list[_ScriptCommand] = []
    for path in sorted((root / "data" / "scripts").rglob("*.inc"), key=lambda item: item.as_posix()):
        parsed, _ = _parse_script(path, root)
        commands.extend(parsed)
    return tuple(commands)


def _producers_for(
    producers: Mapping[str, tuple[ProducerAction, ...]],
    symbol: str,
    map_input: _MapInput,
) -> tuple[ProducerAction, ...]:
    """Resolve producers while respecting per-map temporary variable scope."""

    actions = producers.get(symbol, ())
    if symbol.startswith(("VAR_TEMP_", "VAR_0x800", "VAR_RESULT", "FLAG_TEMP_")):
        prefix = f"data/maps/{map_input.map_name}/"
        actions = tuple(action for action in actions if action.source.path.startswith(prefix))
    return actions


def _candidate_id(
    *,
    game: str,
    revision: str,
    kind: str,
    source: SourceLocation,
    guard_expression: str | None,
    metadata: tuple[tuple[str, str], ...],
) -> str:
    payload = {
        "game": game,
        "revision": revision,
        "kind": kind,
        "source": asdict(source),
        "guard_expression": guard_expression,
        "metadata": metadata,
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:20]
    return f"{game}:{kind}:{digest}"


def _make_candidate(
    *,
    game: str,
    revision: str,
    kind: str,
    source: SourceLocation,
    guard_expression: str | None = None,
    guard_symbol: str | None = None,
    guard_value: str | None = None,
    affected_maps: Iterable[str] = (),
    affected_objects: Iterable[str] = (),
    producer_actions: Iterable[ProducerAction] = (),
    semantic_confidence: str = "low",
    diagnostics: Iterable[str] = (),
    metadata: Mapping[str, str] | None = None,
) -> GateCandidate:
    normalized_metadata = tuple(sorted((str(key), str(value)) for key, value in (metadata or {}).items()))
    normalized_maps = tuple(sorted(set(affected_maps)))
    normalized_objects = tuple(sorted(set(affected_objects)))
    normalized_producers = tuple(sorted(set(producer_actions), key=lambda producer: _canonical(producer)))
    normalized_diagnostics = tuple(sorted(set(diagnostics)))
    candidate_id = _candidate_id(
        game=game,
        revision=revision,
        kind=kind,
        source=source,
        guard_expression=guard_expression,
        metadata=normalized_metadata,
    )
    return GateCandidate(
        candidate_id=candidate_id,
        game=game,
        source_revision=revision,
        kind=kind,
        source=source,
        guard_expression=guard_expression,
        guard_symbol=guard_symbol,
        guard_value=guard_value,
        affected_maps=normalized_maps,
        affected_objects=normalized_objects,
        producer_actions=normalized_producers,
        semantic_confidence=semantic_confidence,
        diagnostics=normalized_diagnostics,
        metadata=normalized_metadata,
    )


def _object_line(raw: str, object_index: int, key: str, value: str) -> int:
    # JSON is formatted one property per line in pokeemerald.  The fallback is
    # still deterministic for compact fixture JSON.
    object_starts = [match.start() for match in re.finditer(r"\{", raw)]
    start = object_starts[object_index] if object_index < len(object_starts) else 0
    return _source_line(raw, f'"{key}": "{value}"', start)


def _script_candidates(
    map_input: _MapInput,
    *,
    game: str,
    revision: str,
    producers: Mapping[str, tuple[ProducerAction, ...]],
    trainer_details: Mapping[str, Mapping[str, str]],
) -> list[GateCandidate]:
    candidates: list[GateCandidate] = []
    map_id = map_input.map_id
    supported_guards = {
        "checkflag": (0, "flag_read"),
        "goto_if_set": (0, "flag_set"),
        "goto_if_unset": (0, "flag_unset"),
        "call_if_set": (0, "flag_set"),
        "call_if_unset": (0, "flag_unset"),
        "goto_if_eq": (0, "value_comparison"),
        "goto_if_ne": (0, "value_comparison"),
        "goto_if_lt": (0, "value_comparison"),
        "goto_if_le": (0, "value_comparison"),
        "goto_if_gt": (0, "value_comparison"),
        "goto_if_ge": (0, "value_comparison"),
        "call_if_eq": (0, "value_comparison"),
        "call_if_ne": (0, "value_comparison"),
        "call_if_lt": (0, "value_comparison"),
        "call_if_le": (0, "value_comparison"),
        "call_if_gt": (0, "value_comparison"),
        "call_if_ge": (0, "value_comparison"),
        "map_script_2": (0, "map_script_value"),
        "goto_if_defeated": (0, "trainer_defeated"),
        "goto_if_not_defeated": (0, "trainer_not_defeated"),
    }
    unsupported = {"compare", "compare_var_to_value", "compare_var_to_var", "specialvar", "vgoto_if"}
    for command in map_input.scripts:
        if command.name in supported_guards and command.arguments:
            symbol = command.arguments[0]
            value = command.arguments[1] if len(command.arguments) > 1 else None
            guard_kind = supported_guards[command.name][1]
            expression = command.raw
            candidates.append(
                _make_candidate(
                    game=game,
                    revision=revision,
                    kind="script_event",
                    source=command.location,
                    guard_expression=expression,
                    guard_symbol=symbol,
                    guard_value=value,
                    affected_maps=(map_id,),
                    producer_actions=_producers_for(producers, symbol, map_input),
                    semantic_confidence="medium" if guard_kind in {"flag_set", "flag_unset", "map_script_value"} else "low",
                    diagnostics=("script_guard_effect_requires_route_analysis",),
                    metadata={"command": command.name, "guard_kind": guard_kind},
                )
            )
        elif command.name in unsupported:
            candidates.append(
                _make_candidate(
                    game=game,
                    revision=revision,
                    kind="unknown_expression",
                    source=command.location,
                    guard_expression=command.raw,
                    affected_maps=(map_id,),
                    semantic_confidence="low",
                    diagnostics=("unsupported_or_computed_script_expression",),
                    metadata={"command": command.name},
                )
            )
        if command.name.startswith("trainerbattle") and command.arguments:
            metadata = {
                "command": command.name,
                "trainer": command.arguments[0],
                "clearance_state": "unknown",
            }
            metadata.update(trainer_details.get(command.arguments[0], {"roster_resolution": "unknown"}))
            candidates.append(
                _make_candidate(
                    game=game,
                    revision=revision,
                    kind="trainer_checkpoint",
                    source=command.location,
                    affected_maps=(map_id,),
                    semantic_confidence="medium",
                    diagnostics=("mandatory_route_status_requires_geometry_and_route_analysis",),
                    metadata=metadata,
                )
            )
    return candidates


def _map_event_candidates(
    map_input: _MapInput,
    *,
    game: str,
    revision: str,
    producers: Mapping[str, tuple[ProducerAction, ...]],
    trainer_details: Mapping[str, Mapping[str, str]],
) -> list[GateCandidate]:
    candidates: list[GateCandidate] = []
    data = map_input.data
    raw = map_input.raw_json
    map_id = map_input.map_id
    for index, event in enumerate(data.get("object_events") or ()):
        trainer_type = str(event.get("trainer_type", "TRAINER_TYPE_NONE"))
        script = str(event.get("script", "0x0"))
        if trainer_type != "TRAINER_TYPE_NONE":
            local_id = str(event.get("local_id", index + 1))
            trainer_command = next(
                (
                    command
                    for command in map_input.labels.get(script, ())
                    if command.name.startswith("trainerbattle") and command.arguments
                ),
                None,
            )
            source = trainer_command.location if trainer_command else SourceLocation(
                map_input.json_relative_path,
                _object_line(raw, index, "trainer_type", trainer_type),
                script if script not in {"0", "0x0"} else None,
            )
            metadata = {
                "affordance": "object_event",
                "object_index": str(index),
                "local_id": local_id,
                "trainer_type": trainer_type,
                "script": script,
                "x": str(event.get("x", "")),
                "y": str(event.get("y", "")),
                "movement_type": str(event.get("movement_type", "unknown")),
                "movement_range_x": str(event.get("movement_range_x", "unknown")),
                "movement_range_y": str(event.get("movement_range_y", "unknown")),
                "sight_or_berry_tree_id": str(event.get("trainer_sight_or_berry_tree_id", "unknown")),
                "defeated_state": "engine_trainer_flag",
            }
            if trainer_command:
                metadata.update(
                    {
                        "command": trainer_command.name,
                        "trainer": trainer_command.arguments[0],
                    }
                )
                metadata.update(trainer_details.get(trainer_command.arguments[0], {"roster_resolution": "unknown"}))
                diagnostics = ("mandatory_route_status_requires_geometry_and_route_analysis",)
                confidence = "medium"
            else:
                metadata["trainer"] = "unknown"
                diagnostics = ("trainer_command_not_resolved_from_object_script",)
                confidence = "low"
            candidates.append(
                _make_candidate(
                    game=game,
                    revision=revision,
                    kind="trainer_checkpoint",
                    source=source,
                    affected_maps=(map_id,),
                    affected_objects=(f"{map_id}:{local_id}",),
                    semantic_confidence=confidence,
                    diagnostics=diagnostics,
                    metadata=metadata,
                )
            )
        flag = str(event.get("flag", "0"))
        if flag in {"0", "0x0", "FLAG_NONE"}:
            continue
        local_id = str(event.get("local_id", index + 1))
        source = SourceLocation(
            map_input.json_relative_path,
            _object_line(raw, index, "flag", flag),
            script if script not in {"0", "0x0"} else None,
        )
        candidates.append(
            _make_candidate(
                game=game,
                revision=revision,
                kind="object_visibility",
                source=source,
                guard_expression=f"{flag} is set",
                guard_symbol=flag,
                affected_maps=(map_id,),
                affected_objects=(f"{map_id}:{local_id}",),
                producer_actions=_producers_for(producers, flag, map_input),
                semantic_confidence="medium",
                diagnostics=("object_visibility_effect_requires_route_analysis",),
                metadata={
                    "visibility": "hidden_when_set",
                    "object_index": str(index),
                    "local_id": local_id,
                    "script": script,
                },
            )
        )
    for index, event in enumerate(data.get("coord_events") or ()):
        variable = str(event.get("var", ""))
        value = str(event.get("var_value", ""))
        if not variable:
            continue
        script = str(event.get("script", "0x0"))
        source = SourceLocation(
            map_input.json_relative_path,
            _source_line(raw, f'"var": "{variable}"'),
            script if script not in {"0", "0x0"} else None,
        )
        candidates.append(
            _make_candidate(
                game=game,
                revision=revision,
                kind="script_event",
                source=source,
                guard_expression=f"{variable} == {value}",
                guard_symbol=variable,
                guard_value=value,
                affected_maps=(map_id,),
                affected_objects=(f"{map_id}:coord:{index}",),
                producer_actions=_producers_for(producers, variable, map_input),
                semantic_confidence="medium",
                diagnostics=("coordinate_trigger_effect_requires_route_analysis",),
                metadata={
                    "event_type": "coordinate_trigger",
                    "index": str(index),
                    "script": script,
                    "x": str(event.get("x", "")),
                    "y": str(event.get("y", "")),
                },
            )
        )
    return candidates


def scan_source_tree(
    source_root: str | Path,
    *,
    game: str = "emerald",
    source_revision: str | None = None,
) -> GateDiscoveryReport:
    """Scan a pinned source tree and return a deterministic evidence report."""

    root = Path(source_root).expanduser().resolve()
    revision = source_revision or _git_revision(root)
    inputs = _map_inputs(root)
    producers = _guard_producers(inputs, _shared_script_commands(root))
    from .source_trainer_catalog import parse_source_trainer_catalog, trainer_metadata

    trainer_details = {
        record.trainer_symbol: trainer_metadata(record)
        for record in parse_source_trainer_catalog(root)
    }
    candidates: list[GateCandidate] = []
    for map_input in inputs:
        candidates.extend(
            _map_event_candidates(
                map_input,
                game=game,
                revision=revision,
                producers=producers,
                trainer_details=trainer_details,
            )
        )
        candidates.extend(
            _script_candidates(
                map_input,
                game=game,
                revision=revision,
                producers=producers,
                trainer_details=trainer_details,
            )
        )
    candidates.sort(key=lambda candidate: candidate.candidate_id)
    diagnostics: list[str] = []
    if not inputs:
        diagnostics.append("no_map_json_files_found_under_data/maps")
    return GateDiscoveryReport(
        schema_version=SCHEMA_VERSION,
        scanner_version=SCANNER_VERSION,
        game=game,
        source_root=str(root),
        source_revision=revision,
        candidates=tuple(candidates),
        diagnostics=tuple(diagnostics),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a deterministic source-first campaign gate report")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--game", default="emerald")
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = scan_source_tree(args.source_root, game=args.game, source_revision=args.source_revision)
    serialized = report.to_json()
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
