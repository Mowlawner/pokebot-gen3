import json
import tempfile
import unittest
from enum import Enum
from pathlib import Path

from modules.nuzlocke.events import MapChanged
from modules.nuzlocke.persistence import (
    EventStoreCorruptionError,
    JsonEventStore,
    deserialize_event,
    serialize_event,
)
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
