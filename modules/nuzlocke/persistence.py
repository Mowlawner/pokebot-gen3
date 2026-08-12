"""Durable storage for immutable Nuzlocke events.

This module stores observations, not campaign state.  The JSON document is a
versioned append log.  Writes replace the previous document only after the
complete new document has been flushed and synced, so a truncated write cannot
look like a valid state.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import uuid
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .events import (
    BattleEnded,
    BattleStarted,
    Event,
    GameStateChanged,
    MapChanged,
    PartyChanged,
    PokemonFainted,
    StorageChanged,
    WhiteoutOccurred,
)

SCHEMA_VERSION = 1
_EVENT_TYPES = {
    cls.__name__: cls
    for cls in (
        BattleEnded,
        BattleStarted,
        GameStateChanged,
        MapChanged,
        PartyChanged,
        PokemonFainted,
        WhiteoutOccurred,
        StorageChanged,
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
        return {
            field.name: _encode(getattr(value, field.name)) for field in fields(value)
        }
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
    """Versioned, atomic, inspectable event log.

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
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if document.get("schema_version") != SCHEMA_VERSION or not isinstance(
                document.get("events"), list
            ):
                raise EventStoreCorruptionError(
                    "Unsupported or missing event-store schema version"
                )
            records = document["events"]
            for record in records:
                self._validate_record(record)
            if [record["sequence"] for record in records] != list(
                range(1, len(records) + 1)
            ):
                raise EventStoreCorruptionError("Event sequence is not contiguous")
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
            raise EventStoreCorruptionError(
                f"Could not load event store: {self.path}"
            ) from error

    @staticmethod
    def _validate_record(record: Any) -> None:
        required = {"event_id", "session_id", "sequence", "frame", "type", "payload"}
        if not isinstance(record, dict) or set(record) != required:
            raise EventStoreCorruptionError("Malformed event record")
        if (
            not isinstance(record["sequence"], int)
            or record["sequence"] < 1
            or not isinstance(record["frame"], int)
        ):
            raise EventStoreCorruptionError("Malformed event sequence or frame")
        deserialize_event({"type": record["type"], "payload": record["payload"]})

    def append(self, event: Event, session_id: str | None = None) -> bool:
        """Append one event; return ``False`` when it was already recorded."""
        session = session_id or self.session_id
        serialized = serialize_event(event)
        canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
        event_id = hashlib.sha256(
            f"{session}\0{event.frame}\0{canonical}".encode()
        ).hexdigest()
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
        self._records.append(record)
        self._ids.add(event_id)
        self.flush()
        return True

    def append_many(
        self, events: Iterable[Event], session_id: str | None = None
    ) -> int:
        count = 0
        for event in events:
            count += self.append(event, session_id)
        return count

    def iter_events(self) -> tuple[Event, ...]:
        return tuple(
            deserialize_event({"type": r["type"], "payload": r["payload"]})
            for r in self._records
        )

    def iter_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    def last_sequence(self) -> int:
        return len(self._records)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        document = (
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "events": self._records},
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(document)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        except OSError as error:
            raise EventStoreError(
                f"Could not atomically write event store: {self.path}"
            ) from error
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def __call__(self, event: Event, session_id: str) -> None:
        self.append(event, session_id)
