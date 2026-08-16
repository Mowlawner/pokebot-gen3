import json
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

from modules.nuzlocke.events import MapChanged
from modules.nuzlocke.persistence import (
    EventStoreCorruptionError,
    JsonEventStore,
    deserialize_event,
    serialize_event,
)
import modules.nuzlocke.persistence as persistence
from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.snapshots import (
    InventorySnapshot,
    NuzlockeSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    StorageSnapshot,
)


class State(Enum):
    OVERWORLD = 1


def snapshot(frame, map_number=2):
    return NuzlockeSnapshot(
        frame,
        "test",
        State.OVERWORLD,
        PlayerSnapshot("May", 1, map_number, "MAP", (1, 1), "Down", True),
        (),
        InventorySnapshot((), (), ()),
        None,
        StorageSnapshot(0, ()),
        ProgressionSnapshot(()),
    )


class TestNuzlockePersistence(unittest.TestCase):
    def test_reload_order_and_duplicate_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            event = MapChanged(1, (1, 2), (1, 3))
            store = JsonEventStore(path, session_id="session-a")
            self.assertTrue(store.append(event))
            self.assertFalse(store.append(event))
            self.assertEqual(store.append_many((MapChanged(2, (1, 3), (1, 4)),)), 1)
            reloaded = JsonEventStore(path)
            self.assertEqual(reloaded.iter_events(), (event, MapChanged(2, (1, 3), (1, 4))))
            self.assertEqual(reloaded.last_sequence(), 2)

    def test_atomic_replace_runs_after_temporary_handle_is_closed(self):
        """The real replacement must occur outside the temporary-file context."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path, session_id="session-a")
            event = MapChanged(1, (1, 2), (1, 3))
            real_replace = persistence.os.replace

            def replace(source, destination):
                # On Windows this real replace raises if the writer still owns
                # source.  Keeping the callback real also exercises the
                # destination replacement path rather than merely mocking it.
                real_replace(source, destination)

            with patch.object(persistence.os, "replace", side_effect=replace):
                self.assertTrue(store.append(event))

            self.assertEqual(JsonEventStore(path).iter_events(), (event,))

    def test_new_store_is_line_appended_and_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path, session_id="session-a")
            header_size = path.stat().st_size
            event = MapChanged(1, (1, 2), (1, 3))
            self.assertTrue(store.append(event))
            self.assertGreater(path.stat().st_size, header_size)
            self.assertEqual(len(path.read_bytes().splitlines()), 2)
            self.assertEqual(JsonEventStore(path).iter_events(), (event,))

    def test_legacy_store_uses_append_sidecar_without_rewriting_base(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            first = MapChanged(1, (1, 2), (1, 3))
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "events": [
                            {
                                "event_id": "legacy",
                                "session_id": "old",
                                "sequence": 1,
                                "frame": first.frame,
                                "type": "MapChanged",
                                "payload": serialize_event(first)["payload"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            original = path.read_bytes()
            second = MapChanged(2, (1, 3), (1, 4))
            store = JsonEventStore(path)
            self.assertTrue(store.append(second, "new"))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(JsonEventStore(path).iter_events(), (first, second))
            self.assertTrue(store._append_path.exists())

    def test_truncated_final_record_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            event = MapChanged(1, (1, 2), (1, 3))
            store.append(event)
            with path.open("ab") as handle:
                handle.write(b'{"event_id":"partial"')
            recovered = JsonEventStore(path)
            self.assertEqual(recovered.iter_events(), (event,))
            self.assertFalse(path.read_bytes().endswith(b"partial"))

    def test_malformed_final_record_is_recovered_but_middle_corruption_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            event = MapChanged(1, (1, 2), (1, 3))
            store.append(event)
            with path.open("ab") as handle:
                handle.write(b'{"not":"an event"}\n')
            self.assertEqual(JsonEventStore(path).iter_events(), (event,))

            path.write_bytes(
                b'{"schema_version":2}\n{"not":"an event"}\n'
                + json.dumps(
                    {
                        "event_id": "later",
                        "session_id": "s",
                        "sequence": 2,
                        "frame": 2,
                        "type": "MapChanged",
                        "payload": serialize_event(MapChanged(2, (1, 3), (1, 4)))["payload"],
                    }
                ).encode()
                + b"\n"
            )
            with self.assertRaises(EventStoreCorruptionError):
                JsonEventStore(path)

    def test_append_failure_does_not_mark_event_committed_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            event = MapChanged(1, (1, 2), (1, 3))
            with patch.object(persistence.os, "fsync", side_effect=OSError("disk full")):
                with self.assertRaises(persistence.EventStoreError):
                    store.append(event)
            self.assertEqual(store.iter_events(), ())

    def test_sessions_allow_repeated_frames_and_runtime_sink(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json")
            runtime = NuzlockeRuntime(event_sink=store)
            runtime.update(snapshot(1))
            runtime.update(snapshot(2, 3))
            first_session = store.iter_records()[0]["session_id"]
            store.append(MapChanged(2, (1, 2), (1, 3)), "session-b")
            records = store.iter_records()
            self.assertEqual([record["frame"] for record in records], [2, 2])
            self.assertNotEqual(first_session, records[1]["session_id"])

    def test_schema_and_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(json.dumps({"schema_version": 999, "events": []}), encoding="utf-8")
            with self.assertRaises(EventStoreCorruptionError):
                JsonEventStore(path)
            path.write_text('{"schema_version": 1, "events": [', encoding="utf-8")
            with self.assertRaises(EventStoreCorruptionError):
                JsonEventStore(path)

    def test_explicit_serialization_round_trip(self):
        event = MapChanged(7, (1, 2), None)
        self.assertEqual(deserialize_event(serialize_event(event)), event)

    def test_runtime_reset_creates_new_session(self):
        runtime = NuzlockeRuntime()
        runtime.update(snapshot(100))
        old_session = runtime.session_id
        runtime.update(snapshot(1, 4))
        self.assertNotEqual(runtime.session_id, old_session)


if __name__ == "__main__":
    unittest.main()
