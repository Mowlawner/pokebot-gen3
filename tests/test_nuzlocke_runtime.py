import unittest
from dataclasses import replace
from enum import Enum
import json
import tempfile
from pathlib import Path

from modules.nuzlocke.events import BattleStarted, MapChanged, NuzlockeStarted
from modules.nuzlocke.campaign_objectives import ObjectiveStatus, plan_campaign, select_campaign_objective
from modules.nuzlocke.campaign_state import CampaignState, derive_campaign_facts, Fact
from modules.nuzlocke.persistence import JsonEventStore
from modules.nuzlocke.projection import load_campaign_projection
from modules.nuzlocke.rules import NuzlockeRulesProjection
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
    CampaignObservationLifecycle,
    NamedFlag,
    NamedVariable,
)


class State(Enum):
    OVERWORLD = 1
    BATTLE = 2
    TITLE_SCREEN = 3
    NAMING_SCREEN = 4


def complete_campaign_observation(*, pokedex_received: bool) -> CampaignObservationSnapshot:
    """Build the complete opening read used by event-store tests."""
    return CampaignObservationSnapshot(
        flags=(
            NamedFlag("SET_WALL_CLOCK", True),
            NamedFlag("RESCUED_BIRCH", True),
            NamedFlag("DEFEATED_RIVAL_ROUTE103", True),
            NamedFlag("SYS_POKEMON_GET", True),
            NamedFlag("SYS_POKEDEX_GET", pokedex_received),
            NamedFlag("RECEIVED_POKEDEX_FROM_BIRCH", pokedex_received),
        ),
        variables=(
            NamedVariable("LITTLEROOT_INTRO_STATE", 7),
            NamedVariable("LITTLEROOT_RIVAL_STATE", 4),
            NamedVariable("BIRCH_LAB_STATE", 5),
        ),
        text_speed=2,
        available=True,
    )


def snapshot(frame: int, map_number: int = 2, *, pokedex_received: bool | None = None) -> NuzlockeSnapshot:
    observation = CampaignObservationSnapshot()
    if pokedex_received is not None:
        observation = complete_campaign_observation(pokedex_received=pokedex_received)
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


def opening_snapshot(frame: int, *, wall_clock: bool = True) -> NuzlockeSnapshot:
    """Build a complete save-backed opening observation for campaign tests."""
    observation = CampaignObservationSnapshot(
        flags=(
            NamedFlag("SET_WALL_CLOCK", wall_clock),
            NamedFlag("RESCUED_BIRCH", True),
            NamedFlag("DEFEATED_RIVAL_ROUTE103", True),
            NamedFlag("SYS_POKEMON_GET", True),
            NamedFlag("SYS_POKEDEX_GET", True),
            NamedFlag("RECEIVED_POKEDEX_FROM_BIRCH", True),
            NamedFlag("VISITED_PETALBURG_CITY", False),
            NamedFlag("RECOVERED_DEVON_GOODS", False),
            NamedFlag("VISITED_RUSTBORO_CITY", False),
            NamedFlag("DEFEATED_RUSTBORO_GYM", False),
        ),
        variables=(
            NamedVariable("LITTLEROOT_INTRO_STATE", 7),
            NamedVariable("LITTLEROOT_RIVAL_STATE", 4),
            NamedVariable("BIRCH_LAB_STATE", 5),
            NamedVariable("PETALBURG_CITY_STATE", 0),
        ),
        text_speed=2,
        available=True,
    )
    return replace(snapshot(frame), campaign_observation=observation)


class TestNuzlockeRuntime(unittest.TestCase):
    @staticmethod
    def fresh_start_snapshot(frame: int = 1) -> NuzlockeSnapshot:
        """Represent mGBA's title screen with an uninitialized save buffer."""
        return NuzlockeSnapshot(
            frame,
            "test",
            State.TITLE_SCREEN,
            PlayerSnapshot(None, None, None, None, None, None, False),
            (),
            InventorySnapshot((), (), ()),
            None,
            StorageSnapshot(0, ()),
            ProgressionSnapshot(()),
            game_state_available=True,
            player_available=False,
            party_available=False,
            inventory_available=False,
            pc_available=False,
            battle_available=False,
            campaign_observation=CampaignObservationSnapshot(
                flags=(NamedFlag("DEFEATED_RUSTBORO_GYM", True),),
                variables=(NamedVariable("BIRCH_LAB_STATE", 0xFFFF),),
                text_speed=7,
                available=False,
                lifecycle=CampaignObservationLifecycle.FRESH_START,
            ),
        )

    def test_fresh_start_cannot_report_campaign_complete(self):
        snapshot = self.fresh_start_snapshot()
        facts = derive_campaign_facts(snapshot, Fact.unavailable(), Fact.unavailable())
        self.assertEqual(facts.first_badge_obtained.status.value, "unavailable")

        state = CampaignState.from_runtime_state(
            snapshot=snapshot,
            observed_projection=None,
            rules_projection=None,
        )

        selection = plan_campaign(state)

        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "set_text_speed")

    def test_naming_screen_fresh_start_keeps_opening_planner_ready(self):
        startup = replace(self.fresh_start_snapshot(), game_state=State.NAMING_SCREEN)
        state = CampaignState.from_runtime_state(
            snapshot=startup,
            observed_projection=None,
            rules_projection=None,
        )

        selection = plan_campaign(state)

        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "set_text_speed")

    def test_fresh_start_does_not_hydrate_or_write_durable_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            store.append(NuzlockeStarted(1))
            runtime = NuzlockeRuntime(event_sink=store, event_store=store)

            self.assertEqual(runtime.update(self.fresh_start_snapshot()), ())

            self.assertFalse(runtime.campaign_history_ready)
            self.assertFalse(runtime.observed_projection.state.nuzlocke_started)
            self.assertFalse(path.with_name(path.name + ".provenance").exists())
            self.assertEqual(len(tuple(path.parent.glob("events.json.stale-*"))), 0)
    @staticmethod
    def wild_battle(frame, map_number=2, inventory=None, inventory_available=True, pokedex_received=None):
        inventory = inventory or InventorySnapshot((), (), ())
        observation = CampaignObservationSnapshot()
        if pokedex_received is not None:
            observation = complete_campaign_observation(pokedex_received=pokedex_received)
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

    def test_repeat_wild_battle_is_ineligible_at_battle_start(self):
        """A random repeat after the first encounter must not invalidate the run."""
        runtime = NuzlockeRuntime()
        inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())
        runtime.update(replace(snapshot(1, pokedex_received=True), inventory=inventory))

        first = runtime.update(
            self.wild_battle(2, inventory=inventory, pokedex_received=True)
        )
        next(event for event in first if isinstance(event, BattleStarted))
        self.assertTrue(runtime.capture_target_for((1, 2), is_wild=True, is_trainer=False))
        # The first battle resolves as a non-capture outcome for this focused
        # boundary test; a real capture is resolved by PokemonCaptured.
        runtime.update(snapshot(3, pokedex_received=True))
        self.assertNotEqual(runtime.rules_projection.state.encounter_for((1, 2)).status, "none")

        repeat = runtime.update(
            self.wild_battle(4, inventory=inventory, pokedex_received=True)
        )
        next(event for event in repeat if isinstance(event, BattleStarted))
        self.assertTrue(runtime.rules_projection.state.legal)
        self.assertEqual(len(runtime.rules_projection.state.violations), 0)

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

    def test_deferred_durable_events_wait_for_a_save_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json")
            runtime = NuzlockeRuntime(event_sink=store, defer_event_persistence=True)
            inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())

            runtime.update(replace(snapshot(1, pokedex_received=True), inventory=inventory))
            events = runtime.update(self.wild_battle(2, inventory=inventory, pokedex_received=True))

            self.assertEqual(type(events[0]).__name__, "BattleStarted")
            self.assertEqual(store.last_sequence(), 0)
            self.assertEqual(runtime.pending_durable_event_count, 1)

            self.assertEqual(runtime.commit_pending_events("manual_save_state"), 1)
            self.assertEqual(store.last_sequence(), 1)
            self.assertEqual(runtime.pending_durable_event_count, 0)
            self.assertEqual(runtime.commit_pending_events("manual_save_state"), 0)

    def test_deferred_events_from_a_discarded_timeline_are_not_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json")
            runtime = NuzlockeRuntime(event_sink=store, defer_event_persistence=True)
            inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())

            runtime.update(replace(snapshot(100, pokedex_received=True), inventory=inventory))
            runtime.update(self.wild_battle(101, inventory=inventory, pokedex_received=True))
            self.assertEqual(runtime.pending_durable_event_count, 1)

            runtime.update(replace(snapshot(1, pokedex_received=True), inventory=inventory))

            self.assertEqual(runtime.pending_durable_event_count, 0)
            self.assertEqual(store.last_sequence(), 0)

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
            runtime.update(opening_snapshot(1))
            self.assertFalse(runtime.observed_projection.state.nuzlocke_started)
            event = runtime.mark_nuzlocke_started()
            self.assertEqual(event.frame, 1)
            self.assertTrue(runtime.observed_projection.state.nuzlocke_started)
            self.assertIsNone(runtime.mark_nuzlocke_started())
            self.assertEqual(len(store.iter_events()), 1)
            self.assertTrue(load_campaign_projection(JsonEventStore(store.path)).nuzlocke_started)
            from modules.nuzlocke.campaign_state import CampaignState

            state = CampaignState.from_runtime_state(
                snapshot=opening_snapshot(1),
                observed_projection=runtime.observed_projection,
                rules_projection=runtime.rules_projection,
            )
            self.assertTrue(state.campaign_facts.nuzlocke_started.value)

    def test_stale_started_event_cannot_advance_a_save_before_wall_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json")
            store.append(NuzlockeStarted(1))

            state = CampaignState.from_runtime_state(
                snapshot=opening_snapshot(2, wall_clock=False),
                observed_projection=load_campaign_projection(JsonEventStore(store.path)),
                rules_projection=NuzlockeRulesProjection(),
            )

            self.assertEqual(state.campaign_facts.nuzlocke_started.value, True)
            selection = select_campaign_objective(state)
            self.assertEqual(selection.status, ObjectiveStatus.READY)
            self.assertEqual(selection.objective.objective_id, "set_wall_clock")

    def test_incompatible_event_history_is_quarantined_before_projection_hydration(self):
        """A stale post-opening log cannot drive a pre-clock save forward."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            store.append(NuzlockeStarted(1))

            runtime = NuzlockeRuntime(event_sink=store, event_store=store)
            self.assertFalse(runtime.campaign_history_ready)
            self.assertFalse(runtime.observed_projection.state.nuzlocke_started)
            runtime.update(opening_snapshot(2, wall_clock=False))

            self.assertFalse(runtime.campaign_history_compatible)
            self.assertEqual(store.last_sequence(), 0)
            self.assertFalse(runtime.observed_projection.state.nuzlocke_started)
            self.assertEqual(runtime.rules_projection.state.encounters, ())
            self.assertEqual(len(tuple(path.parent.glob("events.json.stale-*"))), 1)

            state = CampaignState.from_runtime_state(
                snapshot=opening_snapshot(2, wall_clock=False),
                observed_projection=runtime.observed_projection,
                rules_projection=runtime.rules_projection,
            )
            selection = select_campaign_objective(state)
            self.assertEqual(selection.status, ObjectiveStatus.READY)
            self.assertEqual(selection.objective.objective_id, "set_wall_clock")

    def test_partial_campaign_observation_cannot_authorize_history_hydration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            store.append(NuzlockeStarted(1))

            # ``available`` alone is not enough: this observation only
            # exposes one campaign flag and cannot establish the opening
            # milestone boundary against stale history.
            partial = replace(
                snapshot(2, pokedex_received=True),
                campaign_observation=CampaignObservationSnapshot(
                    flags=(NamedFlag("RECEIVED_POKEDEX_FROM_BIRCH", True),),
                    available=True,
                ),
            )
            runtime = NuzlockeRuntime(event_sink=store, event_store=store)
            self.assertEqual(runtime.update(partial), ())

            self.assertFalse(runtime.campaign_history_ready)
            self.assertTrue(runtime.campaign_history_compatible)
            self.assertFalse(runtime.observed_projection.state.nuzlocke_started)
            self.assertEqual(len(tuple(path.parent.glob("events.json.stale-*"))), 0)

            # A complete ROM-backed observation can now make the strict
            # compatibility decision and quarantine the stale store.
            runtime.update(opening_snapshot(3, wall_clock=False))
            self.assertFalse(runtime.campaign_history_compatible)
            self.assertTrue(runtime.campaign_history_ready)
            self.assertEqual(len(tuple(path.parent.glob("events.json.stale-*"))), 1)

    def test_provenance_tracks_later_milestones_and_rejects_older_same_party_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            runtime = NuzlockeRuntime(event_sink=store, event_store=store)
            runtime.update(opening_snapshot(1))
            runtime.mark_nuzlocke_started()

            later_flags = tuple(
                replace(flag, value=True) if flag.name == "DEFEATED_RUSTBORO_GYM" else flag
                for flag in opening_snapshot(2).campaign_observation.flags
            )
            later = replace(
                opening_snapshot(2),
                campaign_observation=replace(opening_snapshot(2).campaign_observation, flags=later_flags),
            )
            runtime.update(later)

            provenance = json.loads(path.with_name(path.name + ".provenance").read_text(encoding="utf-8"))
            self.assertTrue(provenance["high_water_facts"]["first_badge_obtained"])

            resumed = NuzlockeRuntime(event_store=JsonEventStore(path))
            resumed.update(opening_snapshot(3))
            self.assertFalse(resumed.campaign_history_compatible)
            self.assertEqual(store.last_sequence(), 1)
            # The event log and its provenance sidecar are quarantined
            # independently but share the same stale prefix.
            self.assertEqual(len(tuple(path.parent.glob("events.json.stale-*"))), 2)

    def test_provenance_update_does_not_quarantine_active_runtime_on_older_observation(self):
        """A transient older ROM observation must not reset live history."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            store = JsonEventStore(path)
            runtime = NuzlockeRuntime(event_sink=store, event_store=store)
            runtime.update(opening_snapshot(1))

            later_flags = tuple(
                replace(flag, value=True) if flag.name == "DEFEATED_RUSTBORO_GYM" else flag
                for flag in opening_snapshot(2).campaign_observation.flags
            )
            later = replace(
                opening_snapshot(2),
                campaign_observation=replace(opening_snapshot(2).campaign_observation, flags=later_flags),
            )
            runtime.update(later)
            self.assertTrue(
                json.loads(path.with_name(path.name + ".provenance").read_text(encoding="utf-8"))["high_water_facts"][
                    "first_badge_obtained"
                ]
            )

            runtime.update(opening_snapshot(3))

            self.assertTrue(runtime.campaign_history_compatible)
            self.assertEqual(len(tuple(path.parent.glob("events.json.stale-*"))), 0)
            self.assertEqual(store.last_sequence(), 0)

    def test_runtime_hydrates_campaign_and_rules_projections_from_event_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json", session_id="run-a")
            first = NuzlockeRuntime(event_sink=store, event_store=store)
            inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())
            first.update(replace(snapshot(1, pokedex_received=True), inventory=inventory))
            events = first.update(
                self.wild_battle(2, inventory=inventory, pokedex_received=True)
            )
            self.assertEqual(type(events[0]).__name__, "BattleStarted")
            self.assertEqual(store.last_sequence(), 1)

            restored = NuzlockeRuntime(event_store=JsonEventStore(store.path))
            self.assertFalse(restored.campaign_history_ready)
            restored.update(replace(snapshot(3, pokedex_received=True), inventory=inventory))
            self.assertEqual(restored.rules_projection.state.last_event_sequence, 1)
            self.assertEqual(len(restored.rules_projection.state.encounters), 1)
            self.assertTrue(restored.capture_target_for((1, 2), is_wild=True, is_trainer=False))

    def test_restart_rebases_runtime_sequence_after_ephemeral_events(self):
        """Durable replay uses store sequence, not the live ephemeral count."""
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json", session_id="run-a")
            first = NuzlockeRuntime(event_sink=store, event_store=store)
            inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())

            first.update(replace(snapshot(1, pokedex_received=True), inventory=inventory))
            first.update(replace(snapshot(2, map_number=3, pokedex_received=True), inventory=inventory))
            first_events = first.update(self.wild_battle(3, inventory=inventory, pokedex_received=True))

            self.assertEqual(type(first_events[0]).__name__, "BattleStarted")
            # The battle frame also reports the return to map 2 and the
            # overworld-to-battle state change; both are ephemeral.
            self.assertEqual(first.observed_projection.state.last_event_sequence, 4)
            self.assertEqual(store.last_sequence(), 1)

            restored = NuzlockeRuntime(event_sink=store, event_store=JsonEventStore(store.path))
            restored.update(replace(snapshot(4, pokedex_received=True), inventory=inventory))
            restored_events = restored.update(self.wild_battle(5, inventory=inventory, pokedex_received=True))

            self.assertEqual(type(restored_events[0]).__name__, "BattleStarted")
            # The resumed battle adds one live GameStateChanged event after
            # the replayed durable record; it is not added to the store.
            self.assertEqual(restored.observed_projection.state.last_event_sequence, 3)
            self.assertEqual(restored.rules_projection.state.last_event_sequence, 3)
            self.assertEqual(store.last_sequence(), 2)
            self.assertEqual([record["sequence"] for record in store.iter_records()], [1, 2])

    def test_legacy_replay_restores_encounter_activation_after_campaign_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json")
            store.append(BattleStarted(1, ("WILD",), False, True, False, (), (), (1, 2)))
            store.append(NuzlockeStarted(2))
            store.append(BattleStarted(3, ("WILD",), False, True, False, (), (), (1, 2)))

            restored = NuzlockeRuntime(event_store=store)
            restored.update(snapshot(4, pokedex_received=True))

            self.assertEqual(len(restored.rules_projection.state.encounters), 1)
            self.assertEqual(restored.rules_projection.state.encounters[0].frame, 3)

    def test_runtime_reset_rehydrates_history_instead_of_erasing_nuzlocke_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JsonEventStore(Path(directory) / "events.json", session_id="run-a")
            runtime = NuzlockeRuntime(event_sink=store, event_store=store)
            inventory = InventorySnapshot((), (ItemQuantity("Poké Ball", 1),), ())
            runtime.update(replace(snapshot(100, pokedex_received=True), inventory=inventory))
            runtime.update(self.wild_battle(101, inventory=inventory, pokedex_received=True))
            old_session = runtime.session_id

            self.assertEqual(runtime.update(replace(snapshot(1, pokedex_received=True), inventory=inventory)), ())
            self.assertNotEqual(runtime.session_id, old_session)
            self.assertEqual(len(runtime.rules_projection.state.encounters), 1)
            self.assertEqual(runtime.observed_projection.state.last_event_sequence, 1)


if __name__ == "__main__":
    unittest.main()
