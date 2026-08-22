import unittest
from dataclasses import replace
from enum import Enum
import tempfile
from pathlib import Path

from modules.nuzlocke.events import MapChanged
from modules.nuzlocke.persistence import JsonEventStore
from modules.nuzlocke.projection import load_campaign_projection
from modules.nuzlocke.runtime import NuzlockeRuntime
from modules.nuzlocke.snapshots import (
    BattleSnapshot,
    InventorySnapshot,
    NuzlockeSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    StorageSnapshot,
    ItemQuantity,
    CampaignObservationSnapshot,
    NamedFlag,
)


class State(Enum):
    OVERWORLD = 1
    BATTLE = 2


def snapshot(frame: int, map_number: int = 2, *, pokedex_received: bool | None = None) -> NuzlockeSnapshot:
    observation = CampaignObservationSnapshot()
    if pokedex_received is not None:
        observation = CampaignObservationSnapshot(
            flags=(NamedFlag("RECEIVED_POKEDEX_FROM_BIRCH", pokedex_received),),
            available=True,
        )
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
        campaign_observation=observation,
    )


class TestNuzlockeRuntime(unittest.TestCase):
    @staticmethod
    def wild_battle(frame, map_number=2, inventory=None, inventory_available=True, pokedex_received=None):
        inventory = inventory or InventorySnapshot((), (), ())
        observation = CampaignObservationSnapshot()
        if pokedex_received is not None:
            observation = CampaignObservationSnapshot(
                flags=(NamedFlag("RECEIVED_POKEDEX_FROM_BIRCH", pokedex_received),),
                available=True,
            )
        return NuzlockeSnapshot(
            frame,
            "test",
            State.BATTLE,
            PlayerSnapshot("May", 1, map_number, "MAP", (1, 1), "Down", True),
            (),
            inventory,
            BattleSnapshot(("WILD",), False, True, False, (), (), "InProgress"),
            StorageSnapshot(0, ()),
            ProgressionSnapshot(()),
            inventory_available=inventory_available,
            campaign_observation=observation,
        )

    def test_pre_pokeball_wild_battle_does_not_consume_location(self):
        runtime = NuzlockeRuntime()
        runtime.update(snapshot(1, pokedex_received=False))
        runtime.update(self.wild_battle(2))
        self.assertEqual(runtime.rules_projection.state.encounters, ())
        self.assertFalse(runtime.capture_target_for((1, 2), is_wild=True, is_trainer=False))

        balls = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())
        runtime.update(replace(snapshot(3, pokedex_received=True), inventory=balls))
        runtime.update(self.wild_battle(4, inventory=balls, pokedex_received=True))
        encounters = runtime.rules_projection.state.encounters
        self.assertEqual(len(encounters), 1)
        self.assertEqual(encounters[0].location, (1, 2))
        self.assertTrue(runtime.capture_target_for((1, 2), is_wild=True, is_trainer=False))

    def test_pokedex_receipt_not_ball_quantity_makes_wild_battle_eligible(self):
        runtime = NuzlockeRuntime()
        runtime.update(snapshot(1, map_number=3, pokedex_received=True))
        runtime.update(
            self.wild_battle(2, map_number=3, inventory=InventorySnapshot((), (), ()), pokedex_received=True)
        )
        self.assertEqual(len(runtime.rules_projection.state.encounters), 1)

    def test_first_legal_wild_battle_designates_capture_target(self):
        runtime = NuzlockeRuntime()
        inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 5),), ())
        runtime.update(replace(snapshot(1, pokedex_received=True), inventory=inventory, inventory_available=True))
        battle = NuzlockeSnapshot(
            2,
            "test",
            State.BATTLE,
            PlayerSnapshot("May", 1, 2, "MAP", (1, 1), "Down", True),
            (),
            inventory,
            BattleSnapshot(("WILD",), False, True, False, (), (), "InProgress"),
            StorageSnapshot(0, ()),
            ProgressionSnapshot(()),
            campaign_observation=CampaignObservationSnapshot(
                flags=(NamedFlag("RECEIVED_POKEDEX_FROM_BIRCH", True),), available=True
            ),
        )
        runtime.update(battle)
        self.assertTrue(runtime.capture_target_for((1, 2), is_wild=True, is_trainer=False))
        self.assertFalse(runtime.capture_target_for((1, 2), is_wild=True, is_trainer=True))
        self.assertFalse(runtime.capture_target_for((1, 3), is_wild=True, is_trainer=False))

    def test_capture_target_query_does_not_advance_encounter_state(self):
        runtime = NuzlockeRuntime()
        inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())
        runtime.update(replace(snapshot(1, pokedex_received=True), inventory=inventory))
        self.assertFalse(runtime.capture_target_for((1, 2), is_wild=True, is_trainer=False))
        self.assertEqual(runtime.rules_projection.state.encounters, ())

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

    def test_campaign_boundary_is_explicit_and_does_not_depend_on_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json", session_id="run")
            runtime = NuzlockeRuntime(event_sink=store)
            runtime.update(snapshot(1))
            self.assertFalse(runtime.observed_projection.state.nuzlocke_started)
            event = runtime.mark_nuzlocke_started()
            self.assertEqual(event.frame, 1)
            self.assertTrue(runtime.observed_projection.state.nuzlocke_started)
            self.assertIsNone(runtime.mark_nuzlocke_started())
            self.assertEqual(len(store.iter_events()), 1)
            self.assertTrue(load_campaign_projection(JsonEventStore(store.path)).nuzlocke_started)
            from modules.nuzlocke.campaign_state import CampaignState

            state = CampaignState.from_runtime_state(
                snapshot=snapshot(1),
                observed_projection=runtime.observed_projection,
                rules_projection=runtime.rules_projection,
            )
            self.assertTrue(state.campaign_facts.nuzlocke_started.value)


if __name__ == "__main__":
    unittest.main()
