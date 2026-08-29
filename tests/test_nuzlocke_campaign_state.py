import unittest
from enum import Enum

from modules.nuzlocke.campaign_state import CampaignState, FactStatus, RunStatus
from modules.nuzlocke.events import (
    BattleEnded,
    BattleStarted,
    MapChanged,
    PokemonCaptured,
    PokemonFainted,
    WhiteoutOccurred,
)
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.projection import CampaignProjection
from modules.nuzlocke.rules import NuzlockeRulesProjection, CAPTURED
from modules.nuzlocke.snapshots import (
    InventorySnapshot,
    ItemQuantity,
    NamedFlag,
    NuzlockeSnapshot,
    PartyPokemonSnapshot,
    PlayerSnapshot,
    ProgressionSnapshot,
    StoragePokemonSnapshot,
    StorageSnapshot,
)


class FixtureGameState(Enum):
    OVERWORLD = 1


class CampaignStateTests(unittest.TestCase):
    identity = PokemonIdentity(10, 20, 30)

    def snapshot(self, *, available=True, map_id=(1, 2), canonical=None):
        party_pokemon = PartyPokemonSnapshot(
            species="Treecko",
            nickname="Sprig",
            level=10,
            current_hp=20,
            max_hp=20,
            status="none",
            personality_value=10,
            original_trainer_id=20,
            original_trainer_secret_id=30,
            original_trainer_name="Brendan",
            moves=(),
            held_item=None,
            fainted=False,
            egg=False,
            identity=self.identity,
            party_index=0,
        )
        storage_pokemon = StoragePokemonSnapshot(0, 1, party_pokemon)
        return (
            NuzlockeSnapshot(
                frame=20,
                game_id="Pokemon Emerald",
                game_state=FixtureGameState.OVERWORLD if available else None,
                player=PlayerSnapshot("Brendan", map_id[0], map_id[1], "Raw Map", (4, 5), "Down", available),
                party=(party_pokemon,),
                inventory=InventorySnapshot(
                    (ItemQuantity("Potion", 2),),
                    (ItemQuantity("Poke Ball", 5),),
                    (ItemQuantity("Running Shoes", 1),),
                ),
                battle=None,
                pc=StorageSnapshot(0, (storage_pokemon,)),
                progression=ProgressionSnapshot((NamedFlag("BADGE01_GET", True), NamedFlag("BADGE02_GET", False))),
                game_state_available=available,
                player_available=available,
                party_available=available,
                inventory_available=available,
                pc_available=available,
                battle_available=available,
            ),
            canonical,
        )

    def projections(self):
        observed = CampaignProjection()
        observed.apply(MapChanged(1, (1, 1), (1, 2)), session_id="session-a", sequence=1)
        battle_end = BattleEnded(2, "Won", ("TRAINER",), True, False, False, (), (), (1, 2))
        observed.apply(battle_end, session_id="session-a", sequence=2)

        rules = NuzlockeRulesProjection()
        rules.apply(
            BattleStarted(1, ("WILD",), False, True, False, (), (self.identity,), (1, 2)),
            sequence=1,
        )
        rules.apply(PokemonCaptured(2, self.identity, (1, 2)), sequence=2)
        rules.apply(
            BattleEnded(3, "Caught", ("WILD",), False, True, False, (), (self.identity,), (1, 2)),
            sequence=3,
        )
        return observed, rules

    def test_complete_state_and_predicates(self):
        snapshot, _ = self.snapshot()
        observed, rules = self.projections()
        state = CampaignState.from_runtime_state(
            snapshot=snapshot,
            observed_projection=observed,
            rules_projection=rules,
            canonical_area="ROUTE_101",
        )

        self.assertEqual(state.raw_map.value, (1, 2))
        self.assertEqual(state.canonical_area.value, "ROUTE_101")
        self.assertEqual(state.coordinates.value, (4, 5))
        self.assertEqual(state.has_badge("BADGE01_GET").value, True)
        self.assertEqual(state.has_badge("BADGE02_GET").value, False)
        self.assertEqual(state.item_quantity("Poke Ball").value, 5)
        self.assertEqual(state.key_item_quantity("Running Shoes").value, 1)
        self.assertEqual(state.party.value[0].species, "Treecko")
        self.assertEqual(state.storage.value[0].pokemon.species, "Treecko")
        self.assertEqual(state.pokemon_is_dead(self.identity).value, False)
        self.assertEqual(state.alive_pokemon.value, frozenset({self.identity}))
        self.assertEqual(state.current_encounter.value.status, CAPTURED)
        self.assertEqual(state.run_status.value, RunStatus.ACTIVE)
        self.assertEqual(state.run_is_legal().value, True)
        self.assertEqual(state.last_completed_battle.value.outcome, "Won")
        self.assertEqual(state.session_id, "session-a")

    def test_raw_and_canonical_locations_are_distinct_and_canonical_is_explicit(self):
        snapshot, _ = self.snapshot()
        state = CampaignState.from_runtime_state(snapshot=snapshot)
        self.assertEqual(state.raw_map.value, (1, 2))
        self.assertEqual(state.canonical_area.status, FactStatus.UNKNOWN)
        self.assertIsNone(state.canonical_area.value)

    def test_unknown_data_does_not_become_false_or_empty(self):
        snapshot, _ = self.snapshot(available=False)
        state = CampaignState.from_runtime_state(snapshot=snapshot)
        self.assertEqual(state.raw_map.status, FactStatus.UNKNOWN)
        self.assertEqual(state.badges.status, FactStatus.UNKNOWN)
        self.assertEqual(state.inventory.status, FactStatus.UNKNOWN)
        self.assertEqual(state.party.status, FactStatus.UNKNOWN)
        self.assertEqual(state.item_quantity("Poke Ball").status, FactStatus.UNKNOWN)
        self.assertEqual(state.run_status.status, FactStatus.UNKNOWN)
        self.assertEqual(state.run_is_lost().status, FactStatus.UNKNOWN)

    def test_unavailable_story_flags_are_explicit(self):
        snapshot, _ = self.snapshot()
        state = CampaignState.from_runtime_state(snapshot=snapshot)
        self.assertEqual(state.story_flags.status, FactStatus.UNAVAILABLE)

    def test_lost_run_is_distinct(self):
        snapshot, _ = self.snapshot()
        rules = NuzlockeRulesProjection()
        rules.apply(WhiteoutOccurred(1), sequence=1)
        state = CampaignState.from_runtime_state(snapshot=snapshot, rules_projection=rules)
        self.assertEqual(state.run_status.value, RunStatus.LOST)
        self.assertEqual(state.run_is_lost().value, True)
        self.assertEqual(state.run_is_legal().value, False)

    def test_minimal_state_and_missing_projections(self):
        snapshot, _ = self.snapshot()
        state = CampaignState.from_runtime_state(snapshot=snapshot)
        self.assertEqual(state.run_status.status, FactStatus.UNKNOWN)
        self.assertEqual(state.encounters.status, FactStatus.UNKNOWN)
        self.assertEqual(state.last_completed_battle.status, FactStatus.UNAVAILABLE)
        self.assertEqual(state.known_session_ids, ())

    def test_replay_and_direct_projection_produce_equivalent_state(self):
        snapshot, _ = self.snapshot()
        observed, rules = self.projections()
        replayed_observed = CampaignProjection()
        for event, sequence in (
            (MapChanged(1, (1, 1), (1, 2)), 1),
            (BattleEnded(2, "Won", ("TRAINER",), True, False, False, (), (), (1, 2)), 2),
        ):
            replayed_observed.apply(event, session_id="session-a", sequence=sequence)
        replayed_rules = NuzlockeRulesProjection()
        for event, sequence in (
            (BattleStarted(1, ("WILD",), False, True, False, (), (self.identity,), (1, 2)), 1),
            (PokemonCaptured(2, self.identity, (1, 2)), 2),
            (BattleEnded(3, "Caught", ("WILD",), False, True, False, (), (self.identity,), (1, 2)), 3),
        ):
            replayed_rules.apply(event, sequence=sequence)
        direct = CampaignState.from_runtime_state(
            snapshot=snapshot, observed_projection=observed, rules_projection=rules
        )
        replayed = CampaignState.from_runtime_state(
            snapshot=snapshot, observed_projection=replayed_observed, rules_projection=replayed_rules
        )
        self.assertEqual(direct.raw_map, replayed.raw_map)
        self.assertEqual(direct.encounters, replayed.encounters)
        self.assertEqual(direct.last_completed_battle, replayed.last_completed_battle)

    def test_inputs_are_not_mutated_and_state_is_read_only(self):
        snapshot, _ = self.snapshot()
        observed, rules = self.projections()
        observed_before = observed.state
        rules_before = rules.state
        state = CampaignState.from_runtime_state(
            snapshot=snapshot, observed_projection=observed, rules_projection=rules
        )
        self.assertEqual(observed.state, observed_before)
        self.assertEqual(rules.state, rules_before)
        with self.assertRaises(AttributeError):
            state.raw_map = state.raw_map

    def test_multiple_sessions_are_visible_without_being_reinterpreted(self):
        snapshot, _ = self.snapshot()
        observed = CampaignProjection()
        observed.apply(MapChanged(1, (1, 1), (1, 2)), session_id="session-a", sequence=1)
        observed.apply(MapChanged(2, (1, 2), (1, 3)), session_id="session-b", sequence=2)
        state = CampaignState.from_runtime_state(snapshot=snapshot, observed_projection=observed)
        self.assertEqual(state.session_id, "session-b")
        self.assertEqual(state.known_session_ids, ("session-a", "session-b"))


if __name__ == "__main__":
    unittest.main()
