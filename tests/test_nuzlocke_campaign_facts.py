import unittest
from dataclasses import replace

from modules.nuzlocke.campaign_objectives import ObjectiveStatus, initial_emerald_campaign, select_campaign_objective
from modules.nuzlocke.campaign_state import CampaignState, FactStatus
from modules.nuzlocke.rules import NuzlockeRulesProjection
from modules.nuzlocke.events import BattleStarted, NuzlockeStarted
from modules.nuzlocke.projection import CampaignProjection, reduce_events
from modules.nuzlocke.snapshots import CampaignObservationSnapshot, NamedFlag, NamedVariable
from tests.test_nuzlocke_campaign_state import CampaignStateTests
from tests.utility import BotTestCase, with_frame_timeout, with_save_state


class CampaignFactsTests(unittest.TestCase):
    def state(self, *, flags=(), variables=(), text_speed=2, balls=0, available=True):
        fixture = CampaignStateTests()
        snapshot, _ = fixture.snapshot()
        snapshot = replace(
            snapshot,
            inventory=replace(
                snapshot.inventory, poke_balls=(snapshot.inventory.poke_balls[0].__class__("Poke Ball", balls),)
            ),
            campaign_observation=CampaignObservationSnapshot(flags, variables, text_speed, available),
        )
        return CampaignState.from_runtime_state(snapshot=snapshot, rules_projection=NuzlockeRulesProjection())

    def test_emerald_flags_and_variables_reduce_to_semantic_facts(self):
        flags = tuple(
            NamedFlag(name, True)
            for name in (
                "SET_WALL_CLOCK",
                "RESCUED_BIRCH",
                "DEFEATED_RIVAL_ROUTE103",
                "SYS_POKEMON_GET",
                "SYS_POKEDEX_GET",
                "RECEIVED_POKEDEX_FROM_BIRCH",
            )
        )
        variables = (
            NamedVariable("LITTLEROOT_INTRO_STATE", 3),
            NamedVariable("LITTLEROOT_RIVAL_STATE", 3),
            NamedVariable("BIRCH_LAB_STATE", 2),
        )
        facts = self.state(flags=flags, variables=variables, balls=5).campaign_facts
        self.assertTrue(
            all(
                facts[name].value
                for name in (
                    "text_speed_fast",
                    "new_game_setup_complete",
                    "wall_clock_set",
                    "rival_met",
                    "birch_rescued",
                    "starter_obtained",
                    "intro_rival_battle_complete",
                    "pokedex_received",
                    "pokeballs_available",
                )
            )
        )
        self.assertFalse(facts.nuzlocke_started.value)

    def test_unknown_observation_is_not_false(self):
        facts = self.state(available=False).campaign_facts
        self.assertEqual(facts.wall_clock_set.status, FactStatus.UNAVAILABLE)
        self.assertEqual(facts.rival_met.status, FactStatus.UNAVAILABLE)

    def test_selector_is_ordered_and_does_not_recurse(self):
        state = self.state(
            flags=(NamedFlag("SET_WALL_CLOCK", True),),
            variables=(NamedVariable("LITTLEROOT_INTRO_STATE", 3),),
        )
        selection = select_campaign_objective(state)
        self.assertEqual(selection.objective.objective_id, "meet_rival")
        self.assertEqual(selection.status, ObjectiveStatus.UNKNOWN)
        self.assertEqual(
            tuple(o.objective_id for o in initial_emerald_campaign()),
            (
                "set_text_speed",
                "complete_new_game_setup",
                "set_wall_clock",
                "meet_rival",
                "rescue_birch",
                "obtain_starter",
                "complete_intro_rival",
                "receive_pokedex",
                "receive_pokeballs",
                "start_nuzlocke",
            ),
        )

    def test_encounters_are_inactive_before_balls(self):
        projection = NuzlockeRulesProjection(encounters_active=False)
        projection.apply(BattleStarted(1, ("WILD",), False, True, False, (), (), (1, 2)), sequence=1)
        self.assertEqual(projection.state.encounters, ())
        projection.set_encounters_active(True)
        projection.apply(BattleStarted(2, ("WILD",), False, True, False, (), (), (1, 2)), sequence=2)
        self.assertEqual(len(projection.state.encounters), 1)

    def test_campaign_boundary_replays_deterministically(self):
        events = (NuzlockeStarted(10),)
        self.assertEqual(reduce_events(events), reduce_events(events))
        self.assertTrue(reduce_events(events).nuzlocke_started)

    def test_unavailable_facts_do_not_start_nuzlocke(self):
        facts = self.state(available=False).campaign_facts
        self.assertFalse(facts.pokeballs_available.value)
        self.assertEqual(facts.nuzlocke_started.status, FactStatus.UNAVAILABLE)


class CampaignFactsEmulatorTests(BotTestCase):
    @with_save_state("emerald/in_tall_grass_after_receiving_pokeballs.ss1")
    @with_frame_timeout(1)
    def test_retained_pokeball_state_sets_campaign_boundary(self):
        from modules.nuzlocke.campaign_state import CampaignState
        from modules.nuzlocke.rules import NuzlockeRulesProjection
        from modules.nuzlocke.snapshots import get_nuzlocke_snapshot

        state = CampaignState.from_runtime_state(
            snapshot=get_nuzlocke_snapshot(),
            rules_projection=NuzlockeRulesProjection(),
        )
        self.assertEqual(state.campaign_facts.pokeballs_available.value, True)
        self.assertEqual(state.campaign_facts.nuzlocke_started.status, FactStatus.UNAVAILABLE)
        if False:
            yield


if __name__ == "__main__":
    unittest.main()
