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
from modules.nuzlocke.campaign_state import CampaignState, Fact, FactStatus, RunStatus
from modules.nuzlocke.events import BattleEnded
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
        )

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
        self.assertEqual(selection.objective.objective_id, "reach_oldale")

    def test_completed_first_objective_selects_pokeball_objective(self):
        selection = select_campaign_objective(self.state(area="OLDALE_TOWN"))
        self.assertEqual(selection.objective.objective_id, "confirm_early_pokeballs")
        self.assertEqual(selection.status, ObjectiveStatus.READY)

    def test_satisfied_pokeballs_selects_rival_objective(self):
        selection = select_campaign_objective(self.state(area="OLDALE_TOWN", balls=5))
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

    def test_unavailable_inventory_is_unknown_not_blocked(self):
        selection = select_campaign_objective(self.state(area="OLDALE_TOWN", inventory_available=False))
        self.assertEqual(selection.objective.objective_id, "confirm_early_pokeballs")
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

    def test_completed_rival_advances_past_initial_campaign(self):
        rival_battle = BattleEnded(10, "Won", ("TRAINER",), True, False, False, (), (), MapRSE.ROUTE103.value)
        selection = select_campaign_objective(self.state(area="OLDALE_TOWN", balls=5, battle=rival_battle))
        self.assertEqual(selection.status, ObjectiveStatus.COMPLETE)
        self.assertIsNone(selection.objective)

    def test_stable_ids_and_deterministic_definition(self):
        first = initial_emerald_campaign()
        second = initial_emerald_campaign()
        self.assertEqual(
            tuple(objective.objective_id for objective in first),
            ("reach_oldale", "confirm_early_pokeballs", "complete_intro_rival"),
        )
        self.assertEqual(
            [(objective.objective_id, objective.description, objective.execution_id) for objective in first],
            [(objective.objective_id, objective.description, objective.execution_id) for objective in second],
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
