import unittest
from dataclasses import replace

from modules.nuzlocke.campaign_objectives import ObjectiveStatus, initial_emerald_campaign, select_campaign_objective
from modules.nuzlocke.campaign_state import CampaignState, FactStatus
from modules.nuzlocke.rules import NuzlockeRulesProjection
from modules.nuzlocke.events import BattleStarted, NuzlockeStarted
from modules.nuzlocke.projection import CampaignProjection, reduce_events
from modules.nuzlocke.snapshots import CampaignObservationSnapshot, NamedFlag, NamedVariable
from modules.nuzlocke.resource_policy import PokeballRestockPolicy
from tests.test_nuzlocke_campaign_state import CampaignStateTests
from tests.utility import BotTestCase, with_frame_timeout, with_save_state


class CampaignFactsTests(unittest.TestCase):
    def state(self, *, flags=(), variables=(), text_speed=2, balls=0, available=True, pokeball_policy=None):
        fixture = CampaignStateTests()
        snapshot, _ = fixture.snapshot()
        snapshot = replace(
            snapshot,
            inventory=replace(
                snapshot.inventory, poke_balls=(snapshot.inventory.poke_balls[0].__class__("Poke Ball", balls),)
            ),
            campaign_observation=CampaignObservationSnapshot(flags, variables, text_speed, available),
        )
        return CampaignState.from_runtime_state(
            snapshot=snapshot,
            rules_projection=NuzlockeRulesProjection(),
            pokeball_policy=pokeball_policy,
        )

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
            NamedVariable("BIRCH_LAB_STATE", 3),
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
        self.assertTrue(facts.nuzlocke_started.value)
        self.assertFalse(facts.pokeballs_ready.value)

    def test_starter_is_not_complete_during_lab_nickname_sequence(self):
        facts = self.state(
            flags=(NamedFlag("SYS_POKEMON_GET", True),),
            variables=(NamedVariable("BIRCH_LAB_STATE", 2),),
        ).campaign_facts
        self.assertFalse(facts.starter_obtained.value)

    def test_pokeball_receipt_survives_inventory_depletion(self):
        facts = self.state(
            balls=0,
            variables=(NamedVariable("BIRCH_LAB_STATE", 5),),
        ).campaign_facts

        self.assertTrue(facts.pokeballs_received.value)
        self.assertTrue(facts.pokeballs_ready.value)
        self.assertFalse(facts.pokeballs_available.value)
        self.assertFalse(facts.pokeballs_sufficient.value)

    def test_pokeball_sufficiency_uses_configured_lower_threshold(self):
        policy = PokeballRestockPolicy(lower_threshold=3, upper_target=10)

        at_threshold = self.state(balls=3, pokeball_policy=policy).campaign_facts
        below_threshold = self.state(balls=2, pokeball_policy=policy).campaign_facts

        self.assertTrue(at_threshold.pokeballs_sufficient.value)
        self.assertFalse(below_threshold.pokeballs_sufficient.value)

    def test_intro_rival_completion_accepts_post_scene_hide_flag(self):
        facts = self.state(
            flags=(
                NamedFlag("DEFEATED_RIVAL_ROUTE103", False),
                NamedFlag("HIDE_ROUTE_103_RIVAL", True),
            )
        ).campaign_facts

        self.assertTrue(facts.intro_rival_battle_complete.value)

    def test_selector_advances_to_pokedex_after_rival_is_hidden(self):
        state = self.state(
            flags=tuple(
                NamedFlag(name, value)
                for name, value in (
                    ("SET_WALL_CLOCK", True),
                    ("RESCUED_BIRCH", True),
                    ("DEFEATED_RIVAL_ROUTE103", False),
                    ("HIDE_ROUTE_103_RIVAL", True),
                    ("SYS_POKEMON_GET", True),
                    ("SYS_POKEDEX_GET", False),
                    ("RECEIVED_POKEDEX_FROM_BIRCH", False),
                )
            ),
            variables=(
                NamedVariable("LITTLEROOT_INTRO_STATE", 3),
                NamedVariable("LITTLEROOT_RIVAL_STATE", 3),
                NamedVariable("BIRCH_LAB_STATE", 3),
            ),
            balls=5,
        )

        selection = select_campaign_objective(state)

        self.assertEqual(selection.objective.objective_id, "receive_pokedex")
        self.assertEqual(selection.status, ObjectiveStatus.READY)

    def test_intro_rival_completion_is_unknown_when_hide_signal_is_unavailable(self):
        facts = self.state(flags=(NamedFlag("DEFEATED_RIVAL_ROUTE103", False),)).campaign_facts

        self.assertEqual(facts.intro_rival_battle_complete.status, FactStatus.UNKNOWN)

    def test_intro_rival_completion_is_known_false_when_both_flags_are_false(self):
        facts = self.state(
            flags=(
                NamedFlag("DEFEATED_RIVAL_ROUTE103", False),
                NamedFlag("HIDE_ROUTE_103_RIVAL", False),
            )
        ).campaign_facts

        self.assertEqual(facts.intro_rival_battle_complete.status, FactStatus.KNOWN)
        self.assertFalse(facts.intro_rival_battle_complete.value)

    def test_pokedex_completion_accepts_either_authoritative_rom_flag(self):
        for flag_name in ("RECEIVED_POKEDEX_FROM_BIRCH", "SYS_POKEDEX_GET"):
            flags = tuple(
                NamedFlag(name, name == flag_name) for name in ("RECEIVED_POKEDEX_FROM_BIRCH", "SYS_POKEDEX_GET")
            )
            facts = self.state(flags=flags).campaign_facts
            self.assertTrue(facts.pokedex_received.value, flag_name)

    def test_unknown_observation_is_not_false(self):
        facts = self.state(available=False).campaign_facts
        self.assertEqual(facts.wall_clock_set.status, FactStatus.UNAVAILABLE)
        self.assertEqual(facts.rival_met.status, FactStatus.UNAVAILABLE)

    def test_new_emerald_story_facts_preserve_unknown_and_unavailable_states(self):
        names = (
            "petalburg_wally_scene_complete",
            "petalburg_woods_scene_complete",
            "devon_goods_stolen",
            "devon_goods_reported",
            "devon_goods_recovered",
            "devon_goods_returned",
            "devon_goods_delivered",
            "rustboro_city_state",
            "rusturf_tunnel_state",
            "devon_corp_3f_state",
            "devon_corp_3f_scene_complete",
            "roxanne_available",
        )

        unknown = self.state().campaign_facts
        unavailable = self.state(available=False).campaign_facts
        for name in names:
            self.assertEqual(unknown[name].status, FactStatus.UNKNOWN, name)
            self.assertEqual(unavailable[name].status, FactStatus.UNAVAILABLE, name)

    def test_fresh_initialized_save_observes_clock_as_not_set(self):
        facts = self.state(flags=(NamedFlag("SET_WALL_CLOCK", False),)).campaign_facts
        self.assertEqual(facts.wall_clock_set.status, FactStatus.KNOWN)
        self.assertFalse(facts.wall_clock_set.value)

    def test_woods_scene_and_devon_goods_recovery_are_distinct_rom_facts(self):
        before = self.state(
            flags=(NamedFlag("RECOVERED_DEVON_GOODS", False),),
            variables=(NamedVariable("PETALBURG_WOODS_STATE", 0),),
        ).campaign_facts
        after = self.state(
            flags=(NamedFlag("RECOVERED_DEVON_GOODS", False),),
            variables=(NamedVariable("PETALBURG_WOODS_STATE", 1),),
        ).campaign_facts

        self.assertFalse(before.petalburg_woods_scene_complete.value)
        self.assertTrue(after.petalburg_woods_scene_complete.value)
        self.assertFalse(after.devon_goods_recovered.value)

    def test_devon_goods_progression_reads_each_authoritative_flag_and_state(self):
        facts = self.state(
            flags=tuple(
                NamedFlag(name, True)
                for name in (
                    "DEVON_GOODS_STOLEN",
                    "INTERACTED_WITH_DEVON_EMPLOYEE_GOODS_STOLEN",
                    "RECOVERED_DEVON_GOODS",
                    "RETURNED_DEVON_GOODS",
                    "DELIVERED_DEVON_GOODS",
                    "VISITED_RUSTBORO_CITY",
                )
            )
            + (NamedFlag("DEFEATED_RUSTBORO_GYM", False),),
            variables=(
                NamedVariable("RUSTBORO_CITY_STATE", 5),
                NamedVariable("RUSTURF_TUNNEL_STATE", 2),
                NamedVariable("DEVON_CORP_3F_STATE", 1),
            ),
        ).campaign_facts

        self.assertTrue(facts.devon_goods_stolen.value)
        self.assertTrue(facts.devon_goods_reported.value)
        self.assertTrue(facts.devon_goods_recovered.value)
        self.assertTrue(facts.devon_goods_returned.value)
        self.assertTrue(facts.devon_goods_delivered.value)
        self.assertEqual(facts.rustboro_city_state.value, 5)
        self.assertEqual(facts.rusturf_tunnel_state.value, 2)
        self.assertEqual(facts.devon_corp_3f_state.value, 1)
        self.assertTrue(facts.devon_corp_3f_scene_complete.value)
        self.assertTrue(facts.roxanne_available.value)

    def test_roxanne_availability_requires_rustboro_and_no_badge(self):
        not_reached = self.state(
            flags=(
                NamedFlag("VISITED_RUSTBORO_CITY", False),
                NamedFlag("DEFEATED_RUSTBORO_GYM", False),
            )
        ).campaign_facts
        already_defeated = self.state(
            flags=(
                NamedFlag("VISITED_RUSTBORO_CITY", True),
                NamedFlag("DEFEATED_RUSTBORO_GYM", True),
            )
        ).campaign_facts
        missing_badge_observation = self.state(flags=(NamedFlag("VISITED_RUSTBORO_CITY", True),)).campaign_facts

        self.assertFalse(not_reached.roxanne_available.value)
        self.assertFalse(already_defeated.roxanne_available.value)
        self.assertEqual(missing_badge_observation.roxanne_available.status, FactStatus.UNKNOWN)

    def test_petalburg_wally_completion_waits_for_gym_return_script(self):
        returning_to_gym = self.state(
            variables=(
                NamedVariable("PETALBURG_CITY_STATE", 3),
                NamedVariable("PETALBURG_GYM_STATE", 1),
            )
        ).campaign_facts
        completed = self.state(
            variables=(
                NamedVariable("PETALBURG_CITY_STATE", 3),
                NamedVariable("PETALBURG_GYM_STATE", 2),
            )
        ).campaign_facts

        self.assertFalse(returning_to_gym.petalburg_wally_scene_complete.value)
        self.assertTrue(completed.petalburg_wally_scene_complete.value)

    def test_petalburg_wally_completion_is_unknown_without_gym_state(self):
        facts = self.state(variables=(NamedVariable("PETALBURG_CITY_STATE", 3),)).campaign_facts

        self.assertEqual(facts.petalburg_wally_scene_complete.status, FactStatus.UNKNOWN)

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
                "reach_petalburg",
                "complete_petalburg_wally",
                "complete_petalburg_woods",
                "reach_rustboro",
                "complete_rustboro_goods_stolen",
                "report_devon_goods",
                "recover_devon_goods",
                "return_devon_goods",
                "meet_mr_stone",
                "prepare_roxanne",
                "defeat_roxanne",
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
