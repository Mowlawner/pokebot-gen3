"""Durable storage for immutable Nuzlocke campaign events.

The runtime persistence policy decides which observed events reach this store.
New stores use a small
header followed by one JSON record per line. Each accepted event is appended,
flushed, and synced before ``append`` returns. Legacy schema-1 JSON documents
remain immutable and receive new records in a sibling append log, so opening a
large existing campaign never makes the next emulator frame rewrite history.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
import uuid
from collections import Counter
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from types import UnionType
from typing import Any, Iterable, Union, get_args, get_origin, get_type_hints

from .events import (
    BattleEnded,
    BattleStarted,
    PokemonCaptured,
    Event,
    GameStateChanged,
    MapChanged,
    PartyChanged,
    PokemonFainted,
    StorageChanged,
    WhiteoutOccurred,
    NuzlockeStarted,
)
from .identity import PokemonIdentity
from .policy import PersistenceClass

SCHEMA_VERSION = 1
LINE_SCHEMA_VERSION = 2
PROVENANCE_SCHEMA_VERSION = 1
_PROVENANCE_CAMPAIGN_FACTS = (
    "new_game_setup_complete",
    "wall_clock_set",
    "rival_met",
    "birch_rescued",
    "starter_obtained",
    "intro_rival_battle_complete",
    "pokedex_received",
    "pokeballs_available",
    "pokeballs_ready",
    "visited_petalburg",
    "petalburg_wally_scene_complete",
    "devon_goods_recovered",
    "visited_rustboro",
    "first_badge_obtained",
)
_EVENT_TYPES = {
    cls.__name__: cls
    for cls in (
        BattleEnded,
        BattleStarted,
        PokemonCaptured,
        GameStateChanged,
        MapChanged,
        PartyChanged,
        PokemonFainted,
        WhiteoutOccurred,
        StorageChanged,
        NuzlockeStarted,
    )
}


class EventStoreError(Exception):
    """Base class for event-store failures."""


class EventStoreCorruptionError(EventStoreError):
    """The existing event log is missing, malformed, or has an unknown schema."""


def _encode(value: Any) -> Any:
    """Convert supported event values into JSON-compatible primitives."""

    if isinstance(value, Enum):
        cls = type(value)
        return {
            "__enum__": f"{cls.__module__}:{cls.__qualname__}",
            "name": value.name,
            "value": _encode(value.value),
        }
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def _decode(value: Any) -> Any:
    """Decode JSON primitives and compatibility enum/identity representations."""

    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if isinstance(value, dict) and "__enum__" in value:
        module_name, qualname = value["__enum__"].split(":", 1)
        current: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            current = getattr(current, part)
        return current[value["name"]]
    if isinstance(value, dict):
        if set(value) == {"personality_value", "original_trainer_id", "original_trainer_secret_id"}:
            return PokemonIdentity(
                value["personality_value"], value["original_trainer_id"], value["original_trainer_secret_id"]
            )
        return {key: _decode(item) for key, item in value.items()}
    return value


def _decode_typed(value: Any, annotation: Any) -> Any:
    """Decode a payload using the event field's declared nested type.

    The original decoder handled primitive containers and identities, but
    persisted dataclass values inside a tuple remained plain dictionaries.
    That made a valid ``StorageChanged`` record fail only when a process was
    restarted and the projection accessed ``location.box``.  Keep the
    untyped decoder as the compatibility fallback and recursively materialize
    the small typed dataclass/container shapes used by events.
    """
    if value is None or annotation is Any:
        return _decode(value)

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _decode_typed(value, candidate)
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
        return _decode(value)

    if origin is tuple:
        decoded = _decode(value)
        if not isinstance(decoded, tuple):
            return decoded
        if not args:
            return decoded
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode_typed(item, args[0]) for item in decoded)
        return tuple(
            _decode_typed(item, args[index]) if index < len(args) else _decode(item)
            for index, item in enumerate(decoded)
        )

    if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
        hints = get_type_hints(annotation)
        return annotation(
            **{
                field.name: _decode_typed(value[field.name], hints.get(field.name, field.type))
                for field in fields(annotation)
                if field.name in value
            }
        )

    return _decode(value)


def serialize_event(event: Event) -> dict[str, Any]:
    """Return an explicit primitive representation of an immutable event."""
    if type(event).__name__ not in _EVENT_TYPES or not is_dataclass(event):
        raise TypeError(f"Unsupported Nuzlocke event type: {type(event).__name__}")
    return {"type": type(event).__name__, "payload": _encode(event)}


def deserialize_event(data: dict[str, Any]) -> Event:
    """Reconstruct an event without accessing the emulator."""
    if not isinstance(data, dict) or set(data) != {"type", "payload"}:
        raise EventStoreCorruptionError("Invalid event representation")
    event_type = _EVENT_TYPES.get(data["type"])
    if event_type is None or not isinstance(data["payload"], dict):
        raise EventStoreCorruptionError("Unknown or malformed event type")
    try:
        arguments = {}
        type_hints = get_type_hints(event_type)
        for field in fields(event_type):
            if field.name in data["payload"]:
                arguments[field.name] = _decode_typed(
                    data["payload"][field.name],
                    type_hints.get(field.name, field.type),
                )
            elif field.default is MISSING and field.default_factory is MISSING:
                raise KeyError(field.name)
        return event_type(**arguments)
    except (KeyError, TypeError, ValueError, ImportError, AttributeError, NameError) as error:
        raise EventStoreCorruptionError("Invalid event payload") from error


class JsonEventStore:
    """Versioned, inspectable, synchronously durable event log.

    ``session_id`` identifies the runtime/emulator timeline, not a permanent
    Nuzlocke campaign.  A runtime may pass its session ID to ``append``;
    otherwise this store's own ID is used.  Event IDs are hashes of session,
    frame, type, and canonical payload, making duplicate delivery idempotent.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str | None = None,
        create_on_open: bool = True,
    ) -> None:
        """Open an existing store or create a durable line-log at ``path``.

        ``create_on_open=False`` is used by live emulator sessions whose
        profile should remain untouched until an explicit save boundary.  In
        that mode the line-log header is created lazily by the first append.
        Existing callers retain the historical eager behavior by default.
        """

        self.path = Path(path)
        self.session_id = session_id or str(uuid.uuid4())
        self._records: list[dict[str, Any]] = []
        self._ids: set[str] = set()
        self._legacy = False
        self._append_path = self.path.with_name(self.path.name + ".append")
        self._provenance_path = self.path.with_name(self.path.name + ".provenance")
        self._write_lock = RLock()
        self._initialized = False
        if self.path.exists():
            self._load()
            self._initialized = True
        elif create_on_open:
            self._write_header()
            self._initialized = True

    def _load(self) -> None:
        """Load either the current line format or the legacy JSON format."""

        try:
            # ``utf-8-sig`` accepts ordinary UTF-8 and consumes a leading BOM.
            # Some editors/exporters emit the BOM even though JSON itself does
            # not treat it as whitespace.
            with self.path.open("r", encoding="utf-8-sig") as handle:
                first_line = handle.readline()
        except OSError as error:
            raise EventStoreCorruptionError(f"Could not load event store: {self.path}") from error
        if first_line.lstrip().startswith("{"):
            try:
                first = json.loads(first_line)
            except json.JSONDecodeError:
                first = None
            if isinstance(first, dict) and first.get("schema_version") == LINE_SCHEMA_VERSION:
                self._load_lines(self.path)
                return
            if isinstance(first, dict) and {"event_id", "sequence", "type", "payload"}.issubset(first):
                sequence = first.get("sequence")
                raise EventStoreCorruptionError(
                    f"Missing schema-{LINE_SCHEMA_VERSION} header in {self.path}; "
                    f"first event record has sequence {sequence!r}"
                )
        self._legacy = True
        try:
            document = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if document.get("schema_version") != SCHEMA_VERSION or not isinstance(document.get("events"), list):
                raise EventStoreCorruptionError("Unsupported or missing event-store schema version")
            records = document["events"]
            for index, record in enumerate(records, start=1):
                try:
                    self._validate_record(record)
                except EventStoreCorruptionError as error:
                    raise EventStoreCorruptionError(
                        f"Malformed legacy event record {index} in {self.path}: {error}"
                    ) from error
            expected = list(range(1, len(records) + 1))
            actual = [record["sequence"] for record in records]
            if actual != expected:
                first_bad = next((index for index, (got, want) in enumerate(zip(actual, expected)) if got != want), 0)
                raise EventStoreCorruptionError(
                    f"Legacy event sequence is not contiguous at record {first_bad + 1}: "
                    f"expected {expected[first_bad]}, got {actual[first_bad]}"
                )
            self._records = records
            self._ids = {record["event_id"] for record in records}
        except EventStoreCorruptionError:
            raise
        except (
            OSError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as error:
            raise EventStoreCorruptionError(f"Could not load event store {self.path}: {error}") from error

        if self._append_path.exists():
            self._load_lines(self._append_path, allow_missing_header=True)
        self._validate_sequence()

    def _load_lines(self, path: Path, *, allow_missing_header: bool = False) -> None:
        """Read validated line records, recovering only a final partial line."""

        try:
            with path.open("rb") as handle:
                raw_lines = handle.readlines()
            if not raw_lines:
                if allow_missing_header:
                    return
                raise EventStoreCorruptionError("Empty event store")
            start = 0
            if not allow_missing_header:
                header = json.loads(raw_lines[0].decode("utf-8-sig"))
                if header != {"schema_version": LINE_SCHEMA_VERSION}:
                    raise EventStoreCorruptionError("Unsupported event-line schema")
                start = 1
            for index, raw_line in enumerate(raw_lines[start:], start=start):
                if not raw_line.endswith(b"\n"):
                    if index == len(raw_lines) - 1:
                        self._truncate_partial_line(path, sum(len(line) for line in raw_lines[:index]))
                        break
                    raise EventStoreCorruptionError("Truncated event record")
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    if index == len(raw_lines) - 1:
                        self._truncate_partial_line(path, sum(len(line) for line in raw_lines[:index]))
                        break
                    raise EventStoreCorruptionError(
                        f"Malformed event record at line {index + 1} in {path}: {error}"
                    ) from error
                try:
                    self._validate_record(record)
                except EventStoreCorruptionError as error:
                    if index == len(raw_lines) - 1:
                        self._truncate_partial_line(path, sum(len(line) for line in raw_lines[:index]))
                        break
                    raise EventStoreCorruptionError(
                        f"Malformed event record at line {index + 1} in {path}: {error}"
                    ) from error
                self._records.append(record)
                self._ids.add(record["event_id"])
            self._validate_sequence()
        except EventStoreCorruptionError:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EventStoreCorruptionError(f"Could not load event store {path}: {error}") from error

    @staticmethod
    def _truncate_partial_line(path: Path, size: int) -> None:
        """Truncate a recoverable incomplete final record and sync the file."""

        try:
            with path.open("r+b") as handle:
                handle.truncate(size)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise EventStoreCorruptionError(f"Could not recover event store: {path}") from error

    def _validate_sequence(self) -> None:
        """Require all loaded records to have contiguous one-based sequences."""

        actual = [record["sequence"] for record in self._records]
        expected = list(range(1, len(actual) + 1))
        if actual != expected:
            first_bad = next((index for index, (got, want) in enumerate(zip(actual, expected)) if got != want), 0)
            raise EventStoreCorruptionError(
                f"Event sequence is not contiguous at record {first_bad + 1}: "
                f"expected {expected[first_bad]}, got {actual[first_bad]}"
            )

    def _write_header(self) -> None:
        """Create and durably publish the current line-log schema header."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write((json.dumps({"schema_version": LINE_SCHEMA_VERSION}) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._fsync_directory()
        except OSError as error:
            raise EventStoreError(f"Could not create event store: {self.path}") from error
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_record(record: Any) -> None:
        """Validate record shape and payload before accepting it."""

        required = {"event_id", "session_id", "sequence", "frame", "type", "payload"}
        if not isinstance(record, dict) or set(record) != required:
            raise EventStoreCorruptionError("Malformed event record")
        if not isinstance(record["sequence"], int) or record["sequence"] < 1 or not isinstance(record["frame"], int):
            raise EventStoreCorruptionError("Malformed event sequence or frame")
        deserialize_event({"type": record["type"], "payload": record["payload"]})

    def append(self, event: Event, session_id: str | None = None) -> bool:
        """Append one event; return ``False`` when it was already recorded."""
        session = session_id or self.session_id
        serialized = serialize_event(event)
        canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
        event_id = hashlib.sha256(f"{session}\0{event.frame}\0{canonical}".encode()).hexdigest()
        with self._write_lock:
            if event_id in self._ids:
                return False
            if not self._initialized:
                self._write_header()
                self._initialized = True
            record = {
                "event_id": event_id,
                "session_id": session,
                "sequence": len(self._records) + 1,
                "frame": event.frame,
                "type": serialized["type"],
                "payload": serialized["payload"],
            }
            self._append_record(record)
            self._records.append(record)
            self._ids.add(event_id)
            return True

    def campaign_history_compatible(
        self,
        observation: dict[str, Any],
        *,
        enforce: bool = True,
        persist: bool = True,
    ) -> bool:
        """Check whether durable history belongs to the observed ROM save.

        Event records are intentionally gameplay history, not campaign
        completion facts.  A save-state can be replaced independently of the
        profile's event log, so the runtime uses this small sidecar manifest
        to compare monotonic ROM milestones and a stable party identity before
        hydrating history into the current timeline.

        Missing provenance is supported for old profiles.  An explicit
        ``NuzlockeStarted`` record supplies the opening high-water mark for
        those stores, which is enough to reject the common stale-store case:
        an old post-opening log paired with a pre-clock save state.

        ``persist=False`` performs the same compatibility check without
        writing the provenance sidecar.  Live runtimes use this until a game
        save or emulator save state establishes a durability boundary.
        """
        # A title/menu observation may contain placeholder save bytes, but it
        # is not a save identity and must not create or advance provenance.
        # Runtime normally defers this call until ACTIVE; keeping the guard
        # here protects direct callers and makes the persistence boundary
        # explicit as well.
        if observation.get("lifecycle") == "fresh_start":
            return True
        current = observation.get("facts")
        if not isinstance(current, dict):
            return True
        manifest = self._read_provenance()
        if manifest is None:
            manifest = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "game_id": observation.get("game_id"),
                "high_water_facts": self._inferred_high_water_facts(),
                "stable_identity": self._historical_stable_identity(),
            }

        inferred = self._inferred_high_water_facts()
        high_water = dict(manifest.get("high_water_facts") or {})
        for name, value in inferred.items():
            if value is True:
                high_water[name] = True
        if enforce and manifest.get("game_id") and observation.get("game_id"):
            if manifest["game_id"] != observation["game_id"]:
                return False
        for name, expected in high_water.items():
            if enforce and expected is True and current.get(name) is False:
                return False

        historical_identity = manifest.get("stable_identity")
        current_identity = observation.get("stable_identity")
        if enforce and historical_identity and current_identity and historical_identity != current_identity:
            return False

        changed = False
        if not manifest.get("game_id") and observation.get("game_id"):
            manifest["game_id"] = observation["game_id"]
            changed = True
        for name in _PROVENANCE_CAMPAIGN_FACTS:
            if current.get(name) is True and high_water.get(name) is not True:
                high_water[name] = True
                changed = True
        if not historical_identity and current_identity:
            manifest["stable_identity"] = current_identity
            changed = True
        manifest["schema_version"] = PROVENANCE_SCHEMA_VERSION
        manifest["high_water_facts"] = high_water
        if persist and (changed or not self._provenance_path.exists()):
            self._write_provenance(manifest)
        return True

    def quarantine(self) -> Path | None:
        """Archive incompatible history and start a fresh event log.

        The old files are renamed, never deleted.  This lets a user inspect
        or restore a prior timeline while keeping the profile's canonical
        event-store path aligned with the newly observed save.
        """
        with self._write_lock:
            archived: Path | None = None
            if self.path.exists():
                suffix = f".stale-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
                archived = self.path.with_name(self.path.name + suffix)
                os.replace(self.path, archived)
                if self._append_path.exists():
                    os.replace(self._append_path, archived.with_name(archived.name + ".append"))
                if self._provenance_path.exists():
                    os.replace(self._provenance_path, archived.with_name(archived.name + ".provenance"))
            self._records = []
            self._ids = set()
            self._legacy = False
            self._write_header()
            self._initialized = True
            return archived

    def _read_provenance(self) -> dict[str, Any] | None:
        """Read the optional save-compatibility manifest."""

        if not self._provenance_path.exists():
            return None
        try:
            value = json.loads(self._provenance_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EventStoreCorruptionError(
                f"Could not load event-store provenance {self._provenance_path}: {error}"
            ) from error
        if not isinstance(value, dict) or value.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
            raise EventStoreCorruptionError("Unsupported event-store provenance schema")
        return value

    def _write_provenance(self, value: dict[str, Any]) -> None:
        """Atomically write and sync the save-compatibility manifest."""

        temporary = self._provenance_path.with_name(self._provenance_path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, self._provenance_path)
            self._fsync_directory()
        except OSError as error:
            raise EventStoreError(f"Could not write event-store provenance: {self._provenance_path}") from error
        finally:
            temporary.unlink(missing_ok=True)

    def _historical_stable_identity(self) -> list[int] | None:
        """Find the first durable party identity in old event payloads."""
        for event in self.iter_events():
            candidates = []
            for name in (
                "identity",
                "own_pokemon_identities",
                "entered_identities",
                "changed_identities",
            ):
                value = getattr(event, name, None)
                if isinstance(value, tuple):
                    candidates.extend(value)
                elif value is not None:
                    candidates.append(value)
            for identity in candidates:
                if isinstance(identity, PokemonIdentity):
                    return [
                        identity.personality_value,
                        identity.original_trainer_id,
                        identity.original_trainer_secret_id,
                    ]
        return None

    def _inferred_high_water_facts(self) -> dict[str, bool]:
        """Recover conservative milestones from pre-provenance event logs."""
        if not any(record["type"] == "NuzlockeStarted" for record in self._records):
            return {}
        facts = {name: True for name in _PROVENANCE_CAMPAIGN_FACTS[:7]}
        # A Wally tutorial battle is not the completion boundary: Emerald
        # sets PETALBURG_CITY_STATE=3 before warping back to the gym, then
        # runs a separate return script before PETALBURG_GYM_STATE=2.  Older
        # event stores must not infer the milestone from the battle alone.
        return facts

    def append_many(self, events: Iterable[Event], session_id: str | None = None) -> int:
        """Append an iterable and return the number of newly stored events."""

        count = 0
        for event in events:
            count += self.append(event, session_id)
        return count

    def iter_events(self) -> tuple[Event, ...]:
        """Return all stored records reconstructed as immutable events."""

        return tuple(deserialize_event({"type": r["type"], "payload": r["payload"]}) for r in self._records)

    def iter_records(self) -> tuple[dict[str, Any], ...]:
        """Return shallow copies of all validated persisted records."""

        return tuple(dict(record) for record in self._records)

    def event_statistics(self) -> dict[str, dict[str, int]]:
        """Return counts for events actually present in durable history."""
        counts: Counter[str] = Counter(record["type"] for record in self._records)
        return {
            PersistenceClass.DURABLE.value: dict(counts),
            PersistenceClass.EPHEMERAL.value: {},
            PersistenceClass.DIAGNOSTIC.value: {},
        }

    def last_sequence(self) -> int:
        """Return the current final event sequence, or zero for an empty store."""

        return len(self._records)

    def flush(self) -> None:
        """Retain the legacy API; every successful append is already synced."""
        return None

    def _append_record(self, record: dict[str, Any]) -> None:
        """Append and fsync one serialized record to the active log file."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        target = self._append_path if self._legacy else self.path
        line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(line)
                while view:
                    view = view[os.write(descriptor, view) :]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._fsync_directory()
        except OSError as error:
            raise EventStoreError(f"Could not append event record: {target}") from error

    def _fsync_directory(self) -> None:
        """Best-effort sync the parent directory after a file replacement."""

        try:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass

    def __call__(self, event: Event, session_id: str) -> None:
        """Adapt the store to the runtime event-sink callback contract."""

        self.append(event, session_id)
