import tempfile
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

from modules.nuzlocke.diagnostics import BoundedEventTrace
from modules.nuzlocke.events import (
    BattleStarted,
    GameStateChanged,
    MapChanged,
    PartyChanged,
    PokemonFainted,
    WhiteoutOccurred,
)
from modules.nuzlocke.persistence import EventStoreCorruptionError, JsonEventStore
from modules.nuzlocke.policy import PersistenceClass, classify_event
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
    BATTLE = 2


def snapshot(frame: int, map_number: int = 2, state: State = State.OVERWORLD) -> NuzlockeSnapshot:
    return NuzlockeSnapshot(
        frame,
        "test",
        state,
        PlayerSnapshot("May", 1, map_number, "MAP", (1, 1), "Down", True),
        (),
        InventorySnapshot((), (), ()),
        None,
        StorageSnapshot(0, ()),
        ProgressionSnapshot(()),
    )


class TestNuzlockePolicy(unittest.TestCase):
    def test_observations_are_ephemeral_but_critical_events_are_durable(self):
        self.assertEqual(classify_event(MapChanged(1, None, (1, 2))), PersistenceClass.EPHEMERAL)
        self.assertEqual(classify_event(GameStateChanged(1, State.OVERWORLD, State.BATTLE)), PersistenceClass.EPHEMERAL)
        self.assertEqual(classify_event(BattleStarted(1, ("WILD",), False, True, False)), PersistenceClass.DURABLE)
        self.assertEqual(classify_event(PokemonFainted(2, 0, "Poochyena", "Pup", 1, "battle")), PersistenceClass.DURABLE)
        self.assertEqual(classify_event(WhiteoutOccurred(3)), PersistenceClass.DURABLE)
        self.assertEqual(classify_event(PartyChanged(4, (0,), (), False, ())), PersistenceClass.DURABLE)
        self.assertEqual(classify_event(PartyChanged(5, (), (), False, (0,))), PersistenceClass.EPHEMERAL)

    def test_runtime_keeps_observations_but_does_not_write_them(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json")
            runtime = NuzlockeRuntime(event_sink=store)
            runtime.update(snapshot(1))
            events = runtime.update(snapshot(2, map_number=3, state=State.BATTLE))

            self.assertEqual([type(event).__name__ for event in events], ["MapChanged", "GameStateChanged"])
            self.assertEqual(store.last_sequence(), 0)
            self.assertEqual(runtime.drain_events(), events)
            self.assertEqual(runtime.event_statistics()["ephemeral"], {"MapChanged": 1, "GameStateChanged": 1})

    def test_bounded_diagnostic_trace_does_not_affect_durable_store(self):
        trace = BoundedEventTrace(max_events=2)
        trace(MapChanged(1, None, (1, 2)), "run")
        trace(MapChanged(2, (1, 2), (1, 3)), "run")
        trace(MapChanged(3, (1, 3), (1, 4)), "run")
        self.assertEqual([event.frame for event in trace.events()], [2, 3])

    def test_headerless_line_store_reports_missing_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(
                '{"event_id":"old","sequence":224284,"frame":1,"type":"MapChanged",'
                '"payload":{"frame":1,"old_map":null,"new_map":[1,2]}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EventStoreCorruptionError, "Missing schema-2 header.*224284"):
                JsonEventStore(path)

    def test_event_file_alone_is_not_a_selectable_profile(self):
        from modules.profiles import profile_directory_exists

        with tempfile.TemporaryDirectory() as directory:
            profiles_directory = Path(directory)
            (profiles_directory / "deleted").write_text("event data", encoding="utf-8")
            with patch("modules.profiles.PROFILES_DIRECTORY", profiles_directory):
                self.assertFalse(profile_directory_exists("deleted"))
                self.assertFalse(profile_directory_exists("missing"))


if __name__ == "__main__":
    unittest.main()
