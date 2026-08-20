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
import uuid
from collections import Counter
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

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
        for field in fields(event_type):
            if field.name in data["payload"]:
                arguments[field.name] = _decode(data["payload"][field.name])
            elif field.default is MISSING and field.default_factory is MISSING:
                raise KeyError(field.name)
        return event_type(**arguments)
    except (KeyError, TypeError, ValueError, ImportError, AttributeError) as error:
        raise EventStoreCorruptionError("Invalid event payload") from error


class JsonEventStore:
    """Versioned, inspectable, synchronously durable event log.

    ``session_id`` identifies the runtime/emulator timeline, not a permanent
    Nuzlocke campaign.  A runtime may pass its session ID to ``append``;
    otherwise this store's own ID is used.  Event IDs are hashes of session,
    frame, type, and canonical payload, making duplicate delivery idempotent.
    """

    def __init__(self, path: str | Path, *, session_id: str | None = None) -> None:
        self.path = Path(path)
        self.session_id = session_id or str(uuid.uuid4())
        self._records: list[dict[str, Any]] = []
        self._ids: set[str] = set()
        self._legacy = False
        self._append_path = self.path.with_name(self.path.name + ".append")
        self._write_lock = RLock()
        if self.path.exists():
            self._load()
        else:
            self._write_header()

    def _load(self) -> None:
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
        try:
            with path.open("r+b") as handle:
                handle.truncate(size)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise EventStoreCorruptionError(f"Could not recover event store: {path}") from error

    def _validate_sequence(self) -> None:
        actual = [record["sequence"] for record in self._records]
        expected = list(range(1, len(actual) + 1))
        if actual != expected:
            first_bad = next((index for index, (got, want) in enumerate(zip(actual, expected)) if got != want), 0)
            raise EventStoreCorruptionError(
                f"Event sequence is not contiguous at record {first_bad + 1}: "
                f"expected {expected[first_bad]}, got {actual[first_bad]}"
            )

    def _write_header(self) -> None:
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

    def append_many(self, events: Iterable[Event], session_id: str | None = None) -> int:
        count = 0
        for event in events:
            count += self.append(event, session_id)
        return count

    def iter_events(self) -> tuple[Event, ...]:
        return tuple(deserialize_event({"type": r["type"], "payload": r["payload"]}) for r in self._records)

    def iter_records(self) -> tuple[dict[str, Any], ...]:
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
        return len(self._records)

    def flush(self) -> None:
        """Retain the legacy API; every successful append is already synced."""
        return None

    def _append_record(self, record: dict[str, Any]) -> None:
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
        try:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass

    def __call__(self, event: Event, session_id: str) -> None:
        self.append(event, session_id)
