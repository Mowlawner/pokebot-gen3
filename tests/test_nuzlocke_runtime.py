import unittest
from dataclasses import replace
from enum import Enum

from modules.nuzlocke.events import MapChanged
from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.snapshots import (
    BattleSnapshot,
    InventorySnapshot,
    NuzlockeSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    StorageSnapshot,
)


class State(Enum):
    OVERWORLD = 1
    BATTLE = 2


def snapshot(frame: int, map_number: int = 2) -> NuzlockeSnapshot:
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


class TestNuzlockeRuntime(unittest.TestCase):
    def test_unavailable_startup_snapshot_is_safe(self):
        runtime = NuzlockeRuntime()
        startup = replace(
            snapshot(1),
            game_state=None,
            player_available=False,
            party_available=False,
            inventory_available=False,
            pc_available=False,
            battle_available=False,
            game_state_available=False,
        )
        self.assertEqual(runtime.update(startup), ())

    def test_provider_is_called_once_per_update_and_events_are_delivered_once(self):
        snapshots = iter((snapshot(1), snapshot(2, 3), snapshot(3, 3)))
        calls = 0

        def provider():
            nonlocal calls
            calls += 1
            return next(snapshots)

        runtime = NuzlockeRuntime(provider)
        received = []
        runtime.subscribe(received.append)

        self.assertEqual(runtime.update(), ())
        events = runtime.update()
        self.assertEqual(calls, 2)
        self.assertEqual(events, tuple(received))
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], MapChanged)
        self.assertEqual(runtime.drain_events(), events)
        self.assertEqual(runtime.drain_events(), ())
        self.assertEqual(runtime.update(), ())
        self.assertEqual(received, list(events))

    def test_multiple_events_keep_observer_order(self):
        runtime = NuzlockeRuntime()
        first = snapshot(1)
        second = NuzlockeSnapshot(
            2,
            "test",
            State.BATTLE,
            PlayerSnapshot("May", 1, 3, "MAP", (1, 1), "Down", True),
            (),
            first.inventory,
            BattleSnapshot(("WILD",), False, True, False, (), (), "InProgress"),
            first.pc,
            first.progression,
        )
        runtime.update(first)
        self.assertEqual(
            [type(event).__name__ for event in runtime.update(second)],
            ["BattleStarted", "MapChanged", "GameStateChanged"],
        )

    def test_frame_reset_rebaselines_without_false_events(self):
        runtime = NuzlockeRuntime()
        runtime.update(snapshot(100))
        self.assertEqual(runtime.update(snapshot(1, 9)), ())
        self.assertEqual(runtime.update(snapshot(2, 9)), ())

    def test_supplied_snapshot_is_not_mutated(self):
        runtime = NuzlockeRuntime()
        supplied = snapshot(1)
        runtime.update(supplied)
        self.assertEqual(supplied, snapshot(1))


if __name__ == "__main__":
    unittest.main()
