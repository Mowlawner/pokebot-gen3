import unittest
from dataclasses import replace

from modules.map_data import MapRSE
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    ObjectiveStatus,
    current_area_is,
    has_item,
    initial_emerald_campaign,
    select_campaign_objective,
)
from modules.nuzlocke.campaign_state import CampaignFacts, CampaignState, Fact, FactStatus, RunStatus
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.snapshots import (
    InventorySnapshot,
    ItemQuantity,
    NamedFlag,
    PartyPokemonSnapshot,
    StorageSnapshot,
)


class CampaignObjectiveTests(unittest.TestCase):
    identity = PokemonIdentity(10, 20, 30)

    def state(
        self,
        *,
        area="LITTLEROOT_TOWN",
        balls=0,
        party_usable=True,
        run_status=RunStatus.ACTIVE,
        legal=True,
        canonical_available=True,
        inventory_available=True,
        battle=None,
        campaign_facts=None,
    ):
        party = ()
        if party_usable:
            party = (
                PartyPokemonSnapshot(
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
                ),
            )
        if campaign_facts is None:
            campaign_facts = CampaignFacts(
                *(
                    Fact.known(value)
                    for value in (False, False, False, False, False, False, False, False, balls > 0, False)
                )
            )
        return CampaignState(
            raw_map=Fact.known(MapRSE.OLDALE_TOWN.value),
            canonical_area=Fact.known(area) if canonical_available else Fact.unknown(),
            coordinates=Fact.known((1, 1)),
            badges=Fact.known((NamedFlag("BADGE01_GET", False),)),
            inventory=(
                Fact.known(InventorySnapshot((), (ItemQuantity("Poke Ball", balls),), ()))
                if inventory_available
                else Fact.unknown()
            ),
            party=Fact.known(party),
            storage=Fact.known(StorageSnapshot(0, ())),
            story_flags=Fact.unavailable(),
            encounters=Fact.known(()),
            run_status=Fact.known(run_status),
            dead_pokemon=Fact.known(frozenset()),
            alive_pokemon=Fact.known(frozenset({self.identity}) if party_usable else frozenset()),
            rules_legal=Fact.known(legal),
            last_completed_battle=Fact.known(battle),
            session_id="test-session",
            known_session_ids=("test-session",),
            campaign_facts=campaign_facts,
        )

    @staticmethod
    def facts(**values):
        names = (
            "text_speed_fast",
            "new_game_setup_complete",
            "wall_clock_set",
            "rival_met",
            "birch_rescued",
            "starter_obtained",
            "intro_rival_battle_complete",
            "pokedex_received",
            "pokeballs_available",
            "nuzlocke_started",
        )
        return CampaignFacts(*(Fact.known(values.get(name, False)) for name in names))

    @staticmethod
    def objective(objective_id, completion, prerequisites=(), failure=None):
        return CampaignObjective(
            objective_id=objective_id,
            description=objective_id,
            prerequisites=tuple(prerequisites),
            completion=completion,
            failure=failure,
        )

    @staticmethod
    def predicate(predicate_id, value):
        return CampaignPredicate(
            predicate_id,
            predicate_id,
            lambda _: value if isinstance(value, Fact) else Fact.known(value),
        )

    def test_initial_state_selects_first_objective(self):
        selection = select_campaign_objective(self.state())
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "set_text_speed")

    def test_completed_text_speed_selects_setup_objective(self):
        selection = select_campaign_objective(self.state(campaign_facts=self.facts(text_speed_fast=True)))
        self.assertEqual(selection.objective.objective_id, "complete_new_game_setup")
        self.assertEqual(selection.status, ObjectiveStatus.READY)

    def test_prerequisite_chain_selects_intro_battle(self):
        selection = select_campaign_objective(
            self.state(
                campaign_facts=self.facts(
                    text_speed_fast=True,
                    new_game_setup_complete=True,
                    wall_clock_set=True,
                    rival_met=True,
                    birch_rescued=True,
                    starter_obtained=True,
                )
            )
        )
        self.assertEqual(selection.objective.objective_id, "complete_intro_rival")
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.execution_id, "intro_rival")

    def test_prerequisite_satisfied_is_ready(self):
        objective = self.objective("ready", self.predicate("complete", False), (self.predicate("ready", True),))
        selection = select_campaign_objective(self.state(), (objective,))
        self.assertEqual(selection.status, ObjectiveStatus.READY)

    def test_false_prerequisite_is_blocked(self):
        objective = self.objective("blocked", self.predicate("complete", False), (self.predicate("ready", False),))
        selection = select_campaign_objective(self.state(), (objective,))
        self.assertEqual(selection.status, ObjectiveStatus.BLOCKED)
        self.assertIn("ready", selection.reason)

    def test_unknown_completion_is_unknown(self):
        unknown = CampaignPredicate("missing", "missing", lambda _: Fact.unknown())
        objective = self.objective("unknown", unknown)
        selection = select_campaign_objective(self.state(), (objective,))
        self.assertEqual(selection.status, ObjectiveStatus.UNKNOWN)

    def test_unavailable_campaign_fact_is_unknown_not_blocked(self):
        facts = self.facts()
        facts = replace(facts, text_speed_fast=Fact.unknown())
        selection = select_campaign_objective(self.state(campaign_facts=facts))
        self.assertEqual(selection.objective.objective_id, "set_text_speed")
        self.assertEqual(selection.status, ObjectiveStatus.UNKNOWN)

    def test_completed_objective_is_skipped(self):
        first = self.objective("first", self.predicate("complete", True))
        second = self.objective("second", self.predicate("complete", False))
        selection = select_campaign_objective(self.state(), (first, second))
        self.assertEqual(selection.objective.objective_id, "second")

    def test_blocked_objective_is_not_skipped(self):
        first = self.objective("first", self.predicate("complete", False), (self.predicate("blocked", False),))
        second = self.objective("second", self.predicate("complete", False))
        selection = select_campaign_objective(self.state(), (first, second))
        self.assertEqual(selection.objective.objective_id, "first")
        self.assertEqual(selection.status, ObjectiveStatus.BLOCKED)

    def test_unknown_objective_is_not_skipped(self):
        first = self.objective("first", self.predicate("unknown", Fact.unknown()))
        second = self.objective("second", self.predicate("complete", False))
        selection = select_campaign_objective(self.state(), (first, second))
        self.assertEqual(selection.objective.objective_id, "first")
        self.assertEqual(selection.status, ObjectiveStatus.UNKNOWN)

    def test_failed_objective_stops_progression(self):
        first = self.objective("first", self.predicate("complete", False), failure=self.predicate("failed", True))
        second = self.objective("second", self.predicate("complete", False))
        selection = select_campaign_objective(self.state(), (first, second))
        self.assertEqual(selection.objective.objective_id, "first")
        self.assertEqual(selection.status, ObjectiveStatus.FAILED)

    def test_lost_run_prevents_ready_objectives(self):
        selection = select_campaign_objective(self.state(run_status=RunStatus.LOST, area="OLDALE_TOWN", balls=5))
        self.assertEqual(selection.status, ObjectiveStatus.FAILED)
        self.assertIsNone(selection.objective)

    def test_illegal_run_prevents_ready_objectives(self):
        selection = select_campaign_objective(self.state(legal=False))
        self.assertEqual(selection.status, ObjectiveStatus.FAILED)

    def test_unknown_run_status_prevents_guessing(self):
        state = self.state()
        state = replace(state, run_status=Fact.unknown())
        selection = select_campaign_objective(state)
        self.assertEqual(selection.status, ObjectiveStatus.UNKNOWN)

    def test_completed_intro_battle_selects_pokedex_objective(self):
        selection = select_campaign_objective(
            self.state(
                campaign_facts=self.facts(
                    text_speed_fast=True,
                    new_game_setup_complete=True,
                    wall_clock_set=True,
                    rival_met=True,
                    birch_rescued=True,
                    starter_obtained=True,
                    intro_rival_battle_complete=True,
                )
            )
        )
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "receive_pokedex")

    def test_newly_acquired_pokeballs_select_start_nuzlocke(self):
        state = self.state(
            balls=5,
            campaign_facts=self.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_available=True,
            ),
        )
        selection = select_campaign_objective(state)
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "start_nuzlocke")

    def test_stable_ids_and_deterministic_definition(self):
        first = initial_emerald_campaign()
        second = initial_emerald_campaign()
        self.assertEqual(
            tuple(objective.objective_id for objective in first),
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
        self.assertEqual(
            [(objective.objective_id, objective.description, objective.execution_id) for objective in first],
            [(objective.objective_id, objective.description, objective.execution_id) for objective in second],
        )
        self.assertEqual(
            [tuple(predicate.predicate_id for predicate in objective.prerequisites) for objective in first],
            [
                (),
                ("campaign_fact:text_speed_fast",),
                ("campaign_fact:new_game_setup_complete",),
                ("campaign_fact:wall_clock_set",),
                ("campaign_fact:rival_met",),
                ("campaign_fact:birch_rescued",),
                ("campaign_fact:starter_obtained",),
                ("campaign_fact:intro_rival_battle_complete",),
                ("campaign_fact:pokedex_received",),
                ("campaign_fact:pokeballs_available",),
            ],
        )

    def test_empty_definition_and_all_complete(self):
        empty = select_campaign_objective(self.state(), ())
        self.assertEqual(empty.status, ObjectiveStatus.COMPLETE)
        self.assertIsNone(empty.objective)
        complete = self.objective("complete", self.predicate("complete", True))
        self.assertEqual(select_campaign_objective(self.state(), (complete,)).status, ObjectiveStatus.COMPLETE)

    def test_selector_does_not_mutate_campaign_state(self):
        state = self.state()
        before = state
        select_campaign_objective(state)
        self.assertEqual(state, before)

    def test_predicates_preserve_unknown_and_use_existing_facts(self):
        self.assertEqual(current_area_is("OLDALE_TOWN").evaluate(self.state()).value, False)
        self.assertEqual(current_area_is("OLDALE_TOWN").evaluate(self.state(area="OLDALE_TOWN")).value, True)
        self.assertEqual(has_item("Poke Ball").evaluate(self.state(area="OLDALE_TOWN", balls=0)).value, False)
        self.assertEqual(
            current_area_is("OLDALE_TOWN").evaluate(self.state(canonical_available=False)).status,
            FactStatus.UNKNOWN,
        )


if __name__ == "__main__":
    unittest.main()
