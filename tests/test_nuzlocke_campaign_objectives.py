import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from modules.map_data import MapRSE
import modules.nuzlocke.campaign_objectives as campaign_objectives_module
import modules.nuzlocke.encounter_catalog as encounter_catalog_module
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignGoal,
    CampaignPredicate,
    ObjectiveStatus,
    current_area_is,
    has_item,
    initial_emerald_campaign,
    party_meets_level_target,
    party_fully_restored,
    select_campaign_objective,
    available_campaign_tasks,
    select_available_campaign_task,
    plan_campaign,
    campaign_objective_producers,
    ultimate_emerald_campaign_goal,
    encounter_task,
    measure_route_context,
    RouteRelation,
    campaign_task_diagnostics,
    EncounterEvaluationPolicy,
    EncounterClassification,
    EncounterRecommendation,
    EncounterEvaluation,
    evaluate_encounter_opportunity,
)
from modules.nuzlocke.campaign_state import CampaignFacts, CampaignState, Fact, FactStatus, RunStatus
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.rules import LocationEncounter
from modules.nuzlocke.encounter_catalog import encounter_opportunities
from modules.nuzlocke.capture_policy import EncounterCandidate, EncounterMethod
from modules.nuzlocke.campaign_reachability import (
    CampaignReachabilityStatus,
    assess_campaign_reachability,
)
from modules.nuzlocke.campaign_access import CampaignAccessRequirement
from modules.nuzlocke.emerald_campaign_registry import emerald_map_access_requirements
from modules.nuzlocke.campaign_simulation import CampaignCheckpoint, simulate_checkpoint
from modules.world_navigation import WorldEdge, WorldMapGraph
from modules.nuzlocke.snapshots import (
    InventorySnapshot,
    ItemQuantity,
    NamedFlag,
    PartyPokemonSnapshot,
    StorageSnapshot,
)


class CampaignObjectiveTests(unittest.TestCase):
    identity = PokemonIdentity(10, 20, 30)

    def setUp(self):
        # These are pure planner tests. Prevent a ROM left by an emulator-tier
        # test from turning the static world catalog into an implicit extra
        # fixture and changing the required-objective assertions.
        self._previous_objectives_context = campaign_objectives_module.context
        self._previous_catalog_context = encounter_catalog_module.context
        campaign_objectives_module.context = SimpleNamespace(rom=None)
        encounter_catalog_module.context = SimpleNamespace(rom=None)

    def tearDown(self):
        campaign_objectives_module.context = self._previous_objectives_context
        encounter_catalog_module.context = self._previous_catalog_context

    def test_available_task_discovery_filters_prerequisites_and_keeps_optional_tasks(self):
        first = CampaignObjective(
            "required",
            "required",
            (CampaignPredicate("ready", "ready", lambda _: Fact.known(True)),),
            CampaignPredicate("done", "done", lambda _: Fact.known(False)),
            task_kind="required",
        )
        optional = encounter_task(MapRSE.ROUTE102.value)
        tasks = available_campaign_tasks(
            replace(
                self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
                encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "none"),)),
            ),
            (first, optional),
        )
        self.assertEqual(tuple(task.objective_id for task in tasks), ("required", optional.objective_id))

    def test_encounter_tasks_require_pokedex_before_observed_or_catalog_discovery(self):
        state = replace(
            self.state(),
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "none"),)),
        )
        before = available_campaign_tasks(state)
        self.assertNotIn("obtain_encounter:0:17", tuple(task.objective_id for task in before))

        after = replace(state, campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        discovered = available_campaign_tasks(after)
        self.assertIn("obtain_encounter:0:17", tuple(task.objective_id for task in discovered))

    def test_consumed_encounter_is_not_available(self):
        optional = encounter_task(MapRSE.OLDALE_TOWN.value)
        consumed = self.state()
        consumed = replace(consumed, encounters=Fact.known((LocationEncounter(MapRSE.OLDALE_TOWN.value, "captured"),)))
        self.assertEqual(available_campaign_tasks(consumed, (optional,)), ())

    def test_default_discovery_uses_only_unconsumed_observed_encounter_locations(self):
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            encounters=Fact.known(
                (
                    LocationEncounter(MapRSE.ROUTE102.value, "none"),
                    LocationEncounter(MapRSE.ROUTE103.value, "captured"),
                )
            ),
        )
        tasks = available_campaign_tasks(state)
        self.assertIn("obtain_encounter:0:17", tuple(task.objective_id for task in tasks))
        self.assertNotIn("obtain_encounter:0:18", tuple(task.objective_id for task in tasks))

    def test_world_catalog_preserves_unobserved_and_consumed_state(self):
        state = replace(
            self.state(),
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE103.value, "captured"),)),
        )
        opportunities = encounter_opportunities(state, (MapRSE.ROUTE102.value, MapRSE.ROUTE103.value))
        self.assertEqual(opportunities[0].observed, False)
        self.assertEqual(opportunities[0].consumed, False)
        self.assertEqual(opportunities[1].observed, True)
        self.assertEqual(opportunities[1].consumed, True)

    def test_world_catalog_keeps_ineligible_repeat_area_open(self):
        state = replace(
            self.state(),
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "unknown", eligible=False),)),
        )

        opportunity = encounter_opportunities(state, (MapRSE.ROUTE102.value,))[0]

        self.assertTrue(opportunity.observed)
        self.assertFalse(opportunity.consumed)
        self.assertTrue(opportunity.eligible)

    def test_world_catalog_is_reused_until_encounter_projection_changes(self):
        state = self.state()
        locations = (MapRSE.ROUTE102.value, MapRSE.ROUTE103.value)
        first = encounter_opportunities(state, locations)
        second = encounter_opportunities(state, locations)
        self.assertIs(first, second)

        changed = replace(
            state,
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "captured"),)),
        )
        third = encounter_opportunities(changed, locations)
        self.assertIsNot(first, third)
        self.assertTrue(third[0].consumed)

    def test_default_discovery_omits_location_with_only_unavailable_encounter_tables(self):
        state = self.state()
        with (
            patch(
                "modules.nuzlocke.encounter_catalog._world_encounter_locations",
                return_value=(MapRSE.PETALBURG_CITY.value,),
            ),
            patch(
                "modules.nuzlocke.campaign_objectives.encounter_candidates_for_location",
                return_value=(),
            ) as candidates,
        ):
            tasks = available_campaign_tasks(state)

        self.assertNotIn(
            "obtain_encounter:0:0",
            tuple(task.objective_id for task in tasks),
        )
        candidates.assert_called()
        self.assertEqual(
            candidates.call_args.kwargs["available_methods"],
            frozenset({EncounterMethod.LAND}),
        )

    def test_default_discovery_keeps_large_detour_encounters_visible(self):
        state = self.state(
            campaign_facts=self.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_ready=True,
            )
        )
        near = (0, 16)
        far = (25, 25)
        graph = WorldMapGraph(
            (
                WorldEdge(
                    MapRSE.OLDALE_TOWN.value,
                    MapRSE.PETALBURG_CITY_GYM.value,
                    "connection",
                    (),
                    (),
                    estimated_cost=10,
                ),
                WorldEdge(MapRSE.OLDALE_TOWN.value, near, "connection", (), (), estimated_cost=4),
                WorldEdge(near, MapRSE.PETALBURG_CITY_GYM.value, "connection", (), (), estimated_cost=6),
                WorldEdge(MapRSE.OLDALE_TOWN.value, far, "connection", (), (), estimated_cost=30),
                WorldEdge(far, MapRSE.PETALBURG_CITY_GYM.value, "connection", (), (), estimated_cost=30),
            )
        )
        opportunities = (
            campaign_objectives_module.EncounterOpportunity(near, False, False, True),
            campaign_objectives_module.EncounterOpportunity(far, False, False, True),
        )
        with (
            patch.object(campaign_objectives_module, "encounter_opportunities", return_value=opportunities),
            patch.object(campaign_objectives_module, "_has_current_campaign_encounter", return_value=True),
            patch.object(campaign_objectives_module, "get_world_map_graph", return_value=graph),
        ):
            tasks = available_campaign_tasks(state)

        task_ids = tuple(task.objective_id for task in tasks)
        self.assertIn("obtain_encounter:0:16", task_ids)
        self.assertIn("obtain_encounter:25:25", task_ids)
        far_task = next(task for task in tasks if task.objective_id == "obtain_encounter:25:25")
        self.assertEqual(far_task.encounter_evaluation.classification, EncounterClassification.LARGE_DETOUR)
        self.assertEqual(far_task.encounter_evaluation.recommendation, EncounterRecommendation.DEFER)

    def test_encounter_diagnostics_report_projection_consumption(self):
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "none"),)),
        )
        row = next(row for row in campaign_task_diagnostics(state) if row["task_id"] == "obtain_encounter:0:17")
        self.assertEqual(row["encounter"], {"location": MapRSE.ROUTE102.value, "eligible": True, "consumed": False})

    def test_encounter_evaluation_classifies_route_costs_without_mutation(self):
        current, progression = MapRSE.LITTLEROOT_TOWN.value, MapRSE.PETALBURG_CITY.value
        edges = (
            WorldEdge(current, progression, "connection", (), (), estimated_cost=10),
            WorldEdge(current, MapRSE.ROUTE102.value, "connection", (), (), estimated_cost=4),
            WorldEdge(MapRSE.ROUTE102.value, progression, "connection", (), (), estimated_cost=6),
            WorldEdge(current, MapRSE.ROUTE103.value, "connection", (), (), estimated_cost=10),
            WorldEdge(MapRSE.ROUTE103.value, progression, "connection", (), (), estimated_cost=10),
        )
        state = replace(self.state(), raw_map=Fact.known(current))
        state = replace(state, campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        before = state
        on_route = evaluate_encounter_opportunity(
            state,
            encounter_opportunities(state, (MapRSE.ROUTE102.value,))[0],
            progression,
            graph=WorldMapGraph(edges),
        )
        detour = evaluate_encounter_opportunity(
            state,
            encounter_opportunities(state, (MapRSE.ROUTE103.value,))[0],
            progression,
            policy=EncounterEvaluationPolicy(small_detour_cost=2, small_detour_ratio=0.1),
            graph=WorldMapGraph(edges),
        )
        self.assertEqual(on_route.classification, EncounterClassification.ON_ROUTE)
        self.assertEqual(on_route.recommendation, EncounterRecommendation.PREFER)
        self.assertEqual(detour.classification, EncounterClassification.LARGE_DETOUR)
        self.assertEqual(detour.recommendation, EncounterRecommendation.DEFER)
        self.assertEqual(state, before)

    def test_route101_is_zero_detour_from_littleroot_when_it_is_on_the_direct_corridor(self):
        current = MapRSE.LITTLEROOT_TOWN.value
        route101 = MapRSE.ROUTE101.value
        oldale = MapRSE.OLDALE_TOWN.value
        route102 = MapRSE.ROUTE102.value
        progression = MapRSE.PETALBURG_CITY.value
        graph = WorldMapGraph(
            (
                # Deliberately make each corridor edge expensive. Membership
                # in the direct ROM route, rather than edge arithmetic, is
                # the fact that defines a zero campaign detour.
                WorldEdge(current, route101, "connection", (), (), estimated_cost=40),
                WorldEdge(route101, oldale, "connection", (), (), estimated_cost=40),
                WorldEdge(oldale, route102, "connection", (), (), estimated_cost=2),
                WorldEdge(route102, progression, "connection", (), (), estimated_cost=2),
            )
        )
        state = replace(self.state(), raw_map=Fact.known(current))
        evaluation = evaluate_encounter_opportunity(
            state,
            campaign_objectives_module.EncounterOpportunity(route101, False, False, True),
            progression,
            graph=graph,
        )

        self.assertEqual(evaluation.classification, EncounterClassification.ON_ROUTE)
        self.assertEqual(evaluation.detour_cost, 0)

    def test_unreachable_encounter_is_unavailable(self):
        state = self.state()
        evaluation = evaluate_encounter_opportunity(
            state,
            encounter_opportunities(state, (MapRSE.ROUTE103.value,))[0],
            MapRSE.PETALBURG_CITY.value,
            graph=WorldMapGraph(()),
        )
        self.assertEqual(evaluation.classification, EncounterClassification.UNREACHABLE)
        self.assertEqual(evaluation.recommendation, EncounterRecommendation.UNAVAILABLE)

    def test_checkpoint_simulation_is_read_only_and_uses_real_selector(self):
        state = self.state(campaign_facts=self.facts(intro_rival_battle_complete=True))
        before = state
        report = simulate_checkpoint(CampaignCheckpoint("after_rival", state))
        self.assertEqual(report.status, "observed")
        self.assertEqual(report.primary_task, select_available_campaign_task(state).objective.objective_id)
        self.assertTrue(report.tasks)
        self.assertEqual(state, before)

    def test_unavailable_checkpoint_is_explicit(self):
        report = simulate_checkpoint(CampaignCheckpoint.unavailable("before_roxanne", "no authoritative state"))
        self.assertEqual(report.status, "unavailable")
        self.assertIn("authoritative", report.reason)

    def test_selector_is_deterministic_and_required_precedes_optional(self):
        required = CampaignObjective(
            "required", "required", (), CampaignPredicate("done", "done", lambda _: Fact.known(False)), priority=0
        )
        optional = encounter_task(MapRSE.OLDALE_TOWN.value)
        selection = select_available_campaign_task(self.state(), (optional, required))
        self.assertEqual(selection.objective.objective_id, "required")

    def test_open_party_slot_prefers_nearest_legal_encounter_before_progression(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY.value,
        )

        def evaluated(location, encounter_cost):
            return EncounterEvaluation(
                task_id=f"obtain_encounter:{location[0]}:{location[1]}",
                location=location,
                progression_destination=MapRSE.PETALBURG_CITY.value,
                classification=EncounterClassification.ON_ROUTE,
                recommendation=EncounterRecommendation.PREFER,
                direct_progression_cost=10,
                encounter_cost=encounter_cost,
                encounter_to_progression_cost=10 - encounter_cost,
                via_encounter_cost=10,
                detour_cost=0,
                detour_ratio=0.0,
            )

        route_101 = replace(
            encounter_task(MapRSE.ROUTE101.value),
            encounter_evaluation=evaluated(MapRSE.ROUTE101.value, 3),
        )
        route_102 = replace(
            encounter_task(MapRSE.ROUTE102.value),
            encounter_evaluation=evaluated(MapRSE.ROUTE102.value, 6),
        )
        state = self.state(
            campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True),
        )
        candidate = EncounterCandidate(MapRSE.ROUTE101.value, EncounterMethod.LAND)
        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            return_value=(candidate,),
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, route_102, route_101), state
            )

        self.assertEqual(selection.objective.objective_id, route_101.objective_id)
        self.assertEqual(selection.reason, "selected nearest on-route encounter while party has an open slot")

    def test_current_corridor_encounter_and_next_map_share_one_portfolio(self):
        required = replace(
            self.objective("complete_petalburg_woods", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_WOODS.value,
        )

        def on_route(location, route_cost):
            return replace(
                encounter_task(location),
                encounter_evaluation=EncounterEvaluation(
                    task_id=f"encounter:{location!r}",
                    location=location,
                    progression_destination=MapRSE.PETALBURG_WOODS.value,
                    classification=EncounterClassification.ON_ROUTE,
                    recommendation=EncounterRecommendation.PREFER,
                    direct_progression_cost=20,
                    encounter_cost=route_cost,
                    encounter_to_progression_cost=20 - route_cost,
                    via_encounter_cost=20,
                    detour_cost=0,
                    detour_ratio=0.0,
                ),
            )

        route104 = on_route(MapRSE.ROUTE104.value, 0)
        woods = on_route(MapRSE.PETALBURG_WOODS.value, 10)
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            raw_map=Fact.known(MapRSE.ROUTE104.value),
        )
        candidates = {
            MapRSE.ROUTE104.value: (EncounterCandidate(MapRSE.ROUTE104.value, EncounterMethod.LAND, ("Taillow",)),),
            MapRSE.PETALBURG_WOODS.value: (
                EncounterCandidate(MapRSE.PETALBURG_WOODS.value, EncounterMethod.LAND, ("Shroomish",)),
            ),
        }
        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            side_effect=lambda _state, location: candidates[location],
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, route104, woods), state
            )

        self.assertEqual(selection.objective.objective_id, route104.objective_id)
        self.assertTrue(any(row["task_id"] == woods.objective_id for row in selection.encounter_portfolio))

    def test_early_petalburg_itinerary_keeps_route_101_103_and_102_encounters(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY.value,
        )

        def deferred_encounter(location):
            return replace(
                encounter_task(location),
                encounter_evaluation=EncounterEvaluation(
                    task_id=f"obtain_encounter:{location[0]}:{location[1]}",
                    location=location,
                    progression_destination=MapRSE.PETALBURG_CITY.value,
                    classification=EncounterClassification.LARGE_DETOUR,
                    recommendation=EncounterRecommendation.DEFER,
                    direct_progression_cost=10,
                    encounter_cost=20,
                    encounter_to_progression_cost=10,
                    via_encounter_cost=30,
                    detour_cost=20,
                    detour_ratio=2.0,
                ),
            )

        route_101 = deferred_encounter(MapRSE.ROUTE101.value)
        route_103 = deferred_encounter(MapRSE.ROUTE103.value)
        route_102 = deferred_encounter(MapRSE.ROUTE102.value)
        state = self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        candidates = {
            location: (EncounterCandidate(location, EncounterMethod.LAND, ("Poochyena",), ("dark",)),)
            for location in (MapRSE.ROUTE101.value, MapRSE.ROUTE103.value, MapRSE.ROUTE102.value)
        }

        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            side_effect=lambda _state, location: candidates[location],
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, route_101, route_103, route_102), state
            )
            self.assertEqual(selection.objective.objective_id, route_101.objective_id)

            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, route_103, route_102), state
            )
            self.assertEqual(selection.objective.objective_id, route_103.objective_id)

            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, route_102), state
            )
            self.assertEqual(selection.objective.objective_id, route_102.objective_id)

    def test_large_detour_competes_with_preferred_encounter_in_one_portfolio(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY_GYM.value,
        )

        def evaluated(location, classification, recommendation, detour_cost):
            return replace(
                encounter_task(location),
                encounter_evaluation=EncounterEvaluation(
                    task_id=f"obtain_encounter:{location[0]}:{location[1]}",
                    location=location,
                    progression_destination=MapRSE.PETALBURG_CITY_GYM.value,
                    classification=classification,
                    recommendation=recommendation,
                    direct_progression_cost=10,
                    encounter_cost=10 + detour_cost,
                    encounter_to_progression_cost=10,
                    via_encounter_cost=10 + detour_cost,
                    detour_cost=detour_cost,
                    detour_ratio=detour_cost / 10,
                ),
            )

        preferred = evaluated(
            MapRSE.ROUTE102.value,
            EncounterClassification.ON_ROUTE,
            EncounterRecommendation.PREFER,
            0,
        )
        deferred = evaluated(
            MapRSE.ROUTE103.value,
            EncounterClassification.LARGE_DETOUR,
            EncounterRecommendation.DEFER,
            20,
        )
        state = self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        candidates = {
            preferred.destination: (
                EncounterCandidate(preferred.destination, EncounterMethod.LAND, ("Zigzagoon",)),
            ),
            deferred.destination: (
                EncounterCandidate(
                    deferred.destination,
                    EncounterMethod.LAND,
                    ("Lotad",),
                    ("water",),
                    future_value=20.0,
                ),
            ),
        }
        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            side_effect=lambda _state, location: candidates[location],
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, preferred, deferred), state
            )

        self.assertEqual(selection.objective.objective_id, deferred.objective_id)
        self.assertEqual(selection.encounter_portfolio[0]["task_id"], deferred.objective_id)
        self.assertEqual(
            {row["task_id"] for row in selection.encounter_portfolio},
            {preferred.objective_id, deferred.objective_id},
        )

    def test_early_petalburg_itinerary_catches_route_102_before_calvin_when_already_there(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY.value,
        )
        route_101 = replace(encounter_task(MapRSE.ROUTE101.value), encounter_evaluation=replace(
            EncounterEvaluation(
                task_id="route101",
                location=MapRSE.ROUTE101.value,
                progression_destination=MapRSE.PETALBURG_CITY.value,
                classification=EncounterClassification.LARGE_DETOUR,
                recommendation=EncounterRecommendation.DEFER,
                direct_progression_cost=10,
                encounter_cost=20,
                encounter_to_progression_cost=10,
                via_encounter_cost=30,
                detour_cost=20,
                detour_ratio=2.0,
            )
        ))
        route_102 = replace(route_101, objective_id="obtain_encounter:0:17", destination=MapRSE.ROUTE102.value)
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            raw_map=Fact.known(MapRSE.ROUTE102.value),
        )
        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            return_value=(EncounterCandidate(MapRSE.ROUTE102.value, EncounterMethod.LAND, ("Poochyena",), ("dark",)),),
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, route_101, route_102), state
            )

        self.assertEqual(selection.objective.objective_id, route_102.objective_id)

    def test_trainer_preparation_can_promote_a_large_encounter_detour(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY.value,
        )
        detour = replace(
            encounter_task(MapRSE.ROUTE103.value),
            encounter_evaluation=EncounterEvaluation(
                task_id="obtain_encounter:0:18",
                location=MapRSE.ROUTE103.value,
                progression_destination=MapRSE.PETALBURG_CITY.value,
                classification=EncounterClassification.LARGE_DETOUR,
                recommendation=EncounterRecommendation.DEFER,
                direct_progression_cost=10,
                encounter_cost=20,
                encounter_to_progression_cost=10,
                via_encounter_cost=30,
                detour_cost=20,
                detour_ratio=2.0,
                trainer_preparation_gain=1.0,
                trainer_checkpoint_id="route102:first_poochyena",
            ),
        )
        state = self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        candidate = EncounterCandidate(MapRSE.ROUTE103.value, EncounterMethod.LAND)
        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            return_value=(candidate,),
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, detour), state
            )

        self.assertEqual(selection.objective.objective_id, detour.objective_id)
        self.assertIn("route102:first_poochyena", selection.reason)

    def test_level_target_completion_uses_all_living_non_egg_party_members(self):
        state = self.state()
        predicate = party_meets_level_target(14)
        self.assertFalse(predicate.evaluate(state).value)

        ready_party = tuple(replace(pokemon, level=14) for pokemon in state.party.value)
        self.assertTrue(predicate.evaluate(replace(state, party=Fact.known(ready_party))).value)

    def test_level_target_completion_excludes_dead_members_but_requires_a_living_member(self):
        state = self.state()
        living = replace(state.party.value[0], level=14, party_index=1)
        dead = replace(state.party.value[0], level=1, party_index=0)
        state = replace(
            state,
            party=Fact.known((dead, living)),
            dead_pokemon=Fact.known(frozenset({self.identity})),
        )
        # Both fixtures share the same identity, so the dead projection must
        # exclude both if identity is the only available discriminator.
        self.assertFalse(party_meets_level_target(14).evaluate(state).value)

        distinct_living = replace(living, identity=PokemonIdentity(11, 21, 31))
        state = replace(state, party=Fact.known((dead, distinct_living)))
        self.assertTrue(party_meets_level_target(14).evaluate(state).value)

    def test_route_context_distinguishes_aligned_and_detour_encounters(self):
        current = MapRSE.LITTLEROOT_TOWN.value
        progression = MapRSE.PETALBURG_CITY.value
        aligned = MapRSE.ROUTE102.value
        detour = MapRSE.ROUTE103.value
        edges = (
            WorldEdge(current, aligned, "connection", (), (), estimated_cost=1),
            WorldEdge(aligned, progression, "connection", (), (), estimated_cost=1),
            WorldEdge(current, progression, "connection", (), (), estimated_cost=2),
            WorldEdge(current, detour, "connection", (), (), estimated_cost=3),
            WorldEdge(detour, progression, "connection", (), (), estimated_cost=3),
        )
        route_state = replace(self.state(), raw_map=Fact.known(current))
        aligned_context = measure_route_context(
            route_state, encounter_task(aligned), progression, graph=WorldMapGraph(edges)
        )
        detour_context = measure_route_context(
            route_state, encounter_task(detour), progression, graph=WorldMapGraph(edges)
        )
        self.assertEqual(aligned_context.relation, RouteRelation.ON_ROUTE)
        self.assertEqual(aligned_context.detour_distance, 0)
        self.assertEqual(detour_context.relation, RouteRelation.DETOUR)
        self.assertGreater(detour_context.detour_distance, aligned_context.detour_distance)

    def test_unreachable_route_is_structured(self):
        task = encounter_task(MapRSE.ROUTE103.value)
        context = measure_route_context(
            replace(self.state(), raw_map=Fact.known(MapRSE.LITTLEROOT_TOWN.value)),
            task,
            MapRSE.PETALBURG_CITY.value,
            graph=WorldMapGraph(()),
        )
        self.assertEqual(context.relation, RouteRelation.UNREACHABLE)

    def test_non_spatial_task_has_no_route_context(self):
        task = CampaignObjective("story", "story", (), CampaignPredicate("done", "done", lambda _: Fact.known(False)))
        self.assertIsNone(available_campaign_tasks(self.state(), (task,))[0].route_context)

    def test_ineligible_repeat_history_does_not_hide_open_encounter(self):
        task = encounter_task(MapRSE.ROUTE102.value)
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "unknown", eligible=False),)),
        )

        available = available_campaign_tasks(state, (task,))

        self.assertEqual(tuple(item.objective_id for item in available), (task.objective_id,))
        self.assertEqual(state.encounter_for(MapRSE.ROUTE102.value).value.status, "none")

    def test_deferred_encounter_is_fallback_when_no_required_task_is_executable(self):
        task = replace(
            encounter_task(MapRSE.ROUTE102.value),
            encounter_evaluation=campaign_objectives_module.EncounterEvaluation(
                task_id="obtain_encounter:0:17",
                location=MapRSE.ROUTE102.value,
                progression_destination=MapRSE.PETALBURG_CITY.value,
                classification=EncounterClassification.LARGE_DETOUR,
                recommendation=EncounterRecommendation.DEFER,
                direct_progression_cost=10,
                encounter_cost=20,
                encounter_to_progression_cost=20,
                via_encounter_cost=40,
                detour_cost=30,
                detour_ratio=3.0,
            ),
        )

        selection = campaign_objectives_module._select_available_campaign_tasks((task,), self.state())

        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, task.objective_id)
        self.assertIn("deferred encounter", selection.reason)

    def test_task_diagnostics_are_structured_and_non_mutating(self):
        task = encounter_task(MapRSE.ROUTE102.value)
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "none"),)),
        )
        before = state
        rows = campaign_task_diagnostics(state, (task,))
        self.assertEqual(rows[0]["task_id"], task.objective_id)
        self.assertIn("route_relation", rows[0])
        self.assertIn("selection_diagnostic", rows[0])
        self.assertEqual(rows[0]["selection_diagnostic"]["task_id"], task.objective_id)
        self.assertEqual(state, before)
        self.assertIsNone(select_available_campaign_task(state, (task,)).objective)

    def test_required_selection_explains_skipped_encounter_tradeoff(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY.value,
        )
        encounter = replace(
            encounter_task(MapRSE.ROUTE102.value),
            encounter_evaluation=EncounterEvaluation(
                task_id="obtain_encounter:0:17",
                location=MapRSE.ROUTE102.value,
                progression_destination=MapRSE.PETALBURG_CITY.value,
                classification=EncounterClassification.LARGE_DETOUR,
                recommendation=EncounterRecommendation.DEFER,
                direct_progression_cost=10,
                encounter_cost=12,
                encounter_to_progression_cost=10,
                via_encounter_cost=22,
                detour_cost=12,
                detour_ratio=1.2,
                trainer_preparation_gain=0.25,
                trainer_checkpoint_id="route102:first_poochyena",
            ),
        )

        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            return_value=(),
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks(
                (required, encounter), self.state()
            )

        self.assertEqual(selection.objective.objective_id, required.objective_id)
        detail = selection.selection_diagnostics[0]
        self.assertFalse(detail["selected"])
        self.assertIn("required progression", detail["reason"])
        self.assertEqual(detail["detour_cost"], 12)
        self.assertEqual(detail["trainer_preparation_gain"], 0.25)

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
                    for value in (
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        balls > 0,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    )
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
            "pokeballs_ready",
            "nuzlocke_started",
            "visited_petalburg",
            "devon_goods_recovered",
            "visited_rustboro",
            "first_badge_obtained",
            "petalburg_wally_scene_complete",
            "petalburg_woods_scene_complete",
            "devon_goods_returned",
            "devon_goods_delivered",
            "devon_goods_reported",
            "devon_goods_stolen",
            "rustboro_city_state",
            "rusturf_tunnel_state",
            "devon_corp_3f_state",
            "devon_corp_3f_scene_complete",
            "roxanne_available",
            "pokeballs_received",
            "pokeballs_sufficient",
        )
        return CampaignFacts(
            *(
                Fact.known(
                    values.get(
                        name,
                        (
                            values.get("pokedex_received", False)
                            if name == "nuzlocke_started"
                            else (
                                values.get("pokeballs_ready", False)
                                if name in {"pokeballs_received", "pokeballs_sufficient"}
                                else False
                            )
                        ),
                    )
                )
                for name in names
            )
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
        self.assertEqual(selection.objective.objective_id, "set_text_speed")

    def test_planner_preserves_available_task_selection(self):
        state = self.state(campaign_facts=self.facts(text_speed_fast=True))
        planned = plan_campaign(
            state,
            initial_emerald_campaign()[:2],
            goal=CampaignGoal("startup", "startup", self.predicate("campaign_fact:new_game_setup_complete", False)),
        )
        self.assertEqual(planned.status, ObjectiveStatus.READY)
        self.assertEqual(planned.objective.objective_id, "complete_new_game_setup")

    def test_planner_selects_startup_objective_from_facts_without_game_phase(self):
        goal = CampaignGoal(
            "startup",
            "startup",
            self.predicate("campaign_fact:text_speed_fast", False),
        )
        planned = plan_campaign(self.state(campaign_facts=self.facts()), initial_emerald_campaign()[:1], goal=goal)
        self.assertEqual(planned.status, ObjectiveStatus.READY)
        self.assertEqual(planned.objective.objective_id, "set_text_speed")

    def test_completion_predicates_derive_producer_relationships(self):
        objectives = initial_emerald_campaign()
        producers = campaign_objective_producers(objectives)
        starter = producers["campaign_fact:starter_obtained"]
        self.assertEqual(tuple(item.objective_id for item in starter), ("obtain_starter",))
        rival = producers["campaign_fact:intro_rival_battle_complete"]
        self.assertEqual(tuple(item.objective_id for item in rival), ("complete_intro_rival",))

    def test_pokedex_receipt_is_the_campaign_nuzlocke_boundary(self):
        state = self.state(
            campaign_facts=self.facts(
                pokedex_received=True,
                pokeballs_available=True,
                pokeballs_ready=True,
            )
        )

        self.assertTrue(state.campaign_facts.nuzlocke_started.value)
        planned = plan_campaign(state, goal=ultimate_emerald_campaign_goal())
        self.assertEqual(planned.status, ObjectiveStatus.READY)
        self.assertEqual(planned.objective.objective_id, "complete_petalburg_wally")

    def test_introductory_rival_accepts_a_usable_partially_damaged_party(self):
        objective = next(item for item in initial_emerald_campaign() if item.objective_id == "complete_intro_rival")
        state = self.state(
            campaign_facts=self.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
            ),
        )
        damaged_party = replace(state.party.value[0], current_hp=19)
        state = replace(state, party=Fact.known((damaged_party,)))

        planned = plan_campaign(
            state,
            goal=CampaignGoal(
                "intro_rival",
                "complete the introductory rival battle",
                objective.completion,
            ),
        )

        self.assertEqual(planned.status, ObjectiveStatus.READY)
        self.assertEqual(planned.objective.objective_id, "complete_intro_rival")

    def test_planner_resolves_false_prerequisite_to_existing_producer(self):
        prerequisite = self.predicate("ready", False)
        producer = replace(
            self.objective("produce_ready", self.predicate("ready", False)),
            task_kind="optional",
        )
        blocked = self.objective("blocked", self.predicate("blocked_done", False), (prerequisite,))
        goal = CampaignGoal("blocked_goal", "blocked goal", blocked.completion)
        planned = plan_campaign(self.state(), (blocked, producer), goal=goal)
        self.assertEqual(planned.status, ObjectiveStatus.READY)
        self.assertEqual(planned.objective.objective_id, "produce_ready")
        self.assertIn("resolved executable frontier", planned.reason)

    def test_planner_preserves_unknown_observation(self):
        facts = self.facts()
        facts = replace(facts, text_speed_fast=Fact.unknown())
        goal = CampaignGoal("startup", "startup", self.predicate("campaign_fact:text_speed_fast", Fact.unknown()))
        planned = plan_campaign(self.state(campaign_facts=facts), initial_emerald_campaign()[:1], goal=goal)
        self.assertEqual(planned.status, ObjectiveStatus.UNKNOWN)
        self.assertIsNone(planned.objective)

    def test_dependency_planner_rejects_lost_run_before_mounting_campaign_work(self):
        goal = ultimate_emerald_campaign_goal()
        planned = plan_campaign(
            self.state(run_status=RunStatus.LOST, area="RUSTBORO_CITY", balls=5),
            goal=goal,
        )
        self.assertEqual(planned.status, ObjectiveStatus.FAILED)
        self.assertIsNone(planned.objective)
        self.assertEqual(planned.reason, "run is lost")

    def test_ultimate_goal_is_explicitly_represented_by_current_slice_endpoint(self):
        goal = ultimate_emerald_campaign_goal()
        self.assertEqual(goal.goal_id, "beat_elite_four")
        self.assertEqual(goal.completion.predicate_id, "campaign_fact:first_badge_obtained")
        self.assertEqual(goal.implemented_through, "defeat_roxanne")

    def test_planner_resolves_multi_level_chain_without_order_semantics(self):
        goal_predicate = self.predicate("goal_done", False)
        fact_b = self.predicate("fact_b", False)
        fact_c = self.predicate("fact_c", False)
        goal_producer = self.objective("objective_a", goal_predicate, (fact_b,))
        producer_b = self.objective("objective_b", fact_b, (fact_c,))
        executable = self.objective("objective_c", fact_c)
        goal = CampaignGoal("test_goal", "test goal", goal_predicate)

        planned = plan_campaign(
            self.state(),
            (executable, goal_producer, producer_b),
            goal=goal,
        )
        self.assertEqual(planned.status, ObjectiveStatus.READY)
        self.assertEqual(planned.objective.objective_id, "objective_c")

    def test_planner_is_independent_of_reversed_objective_order(self):
        goal_predicate = self.predicate("goal_done", False)
        prerequisite = self.predicate("prerequisite", False)
        goal_producer = self.objective("goal_producer", goal_predicate, (prerequisite,))
        producer = self.objective("producer", prerequisite)
        goal = CampaignGoal("test_goal", "test goal", goal_predicate)

        forward = plan_campaign(self.state(), (goal_producer, producer), goal=goal)
        reverse = plan_campaign(self.state(), (producer, goal_producer), goal=goal)
        self.assertEqual(forward.objective.objective_id, "producer")
        self.assertEqual(reverse.objective.objective_id, "producer")

    def test_planner_skips_completed_and_failed_producers(self):
        goal_predicate = self.predicate("goal_done", False)
        prerequisite = self.predicate("prerequisite", False)
        failed = self.objective("a_failed_producer", prerequisite, failure=self.predicate("failed", True))
        executable = self.objective("z_executable_producer", prerequisite)
        goal_producer = self.objective("goal_producer", goal_predicate, (prerequisite,))
        goal = CampaignGoal("test_goal", "test goal", goal_predicate)

        planned = plan_campaign(self.state(), (failed, goal_producer, executable), goal=goal)
        self.assertEqual(planned.status, ObjectiveStatus.READY)
        self.assertEqual(planned.objective.objective_id, "z_executable_producer")

    def test_planner_reports_missing_producer_as_blocked(self):
        goal_predicate = self.predicate("goal_done", False)
        missing = self.predicate("missing_fact", False)
        goal_producer = self.objective("goal_producer", goal_predicate, (missing,))
        goal = CampaignGoal("test_goal", "test goal", goal_predicate)

        planned = plan_campaign(self.state(), (goal_producer,), goal=goal)
        self.assertEqual(planned.status, ObjectiveStatus.BLOCKED)
        self.assertIn("no producer", planned.reason)

    def test_planner_reports_unknown_dependency_without_executing_a_producer(self):
        goal_predicate = self.predicate("goal_done", False)
        unknown = self.predicate("unknown_fact", Fact.unknown())
        goal_producer = self.objective("goal_producer", goal_predicate, (unknown,))
        goal = CampaignGoal("test_goal", "test goal", goal_predicate)

        planned = plan_campaign(self.state(), (goal_producer,), goal=goal)
        self.assertEqual(planned.status, ObjectiveStatus.UNKNOWN)
        self.assertIn("unknown", planned.reason)

    def test_planner_reports_complete_ultimate_goal(self):
        goal = CampaignGoal("test_goal", "test goal", self.predicate("goal_done", True))
        planned = plan_campaign(self.state(), (), goal=goal)
        self.assertEqual(planned.status, ObjectiveStatus.COMPLETE)
        self.assertIsNone(planned.objective)

    def test_completed_text_speed_selects_setup_objective(self):
        selection = select_campaign_objective(self.state(campaign_facts=self.facts(text_speed_fast=True)))
        self.assertEqual(selection.objective.objective_id, "complete_new_game_setup")
        self.assertEqual(selection.status, ObjectiveStatus.READY)

    def test_rescue_birch_requires_authoritative_flag_and_then_advances(self):
        before = self.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
        )
        selection = select_campaign_objective(self.state(campaign_facts=before))
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "rescue_birch")

        after = replace(before, birch_rescued=Fact.known(True))
        selection = select_campaign_objective(self.state(campaign_facts=after))
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "obtain_starter")

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

    def test_newly_acquired_pokeballs_select_petalburg_progression(self):
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
                pokeballs_ready=True,
            ),
        )
        selection = select_campaign_objective(state)
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "complete_petalburg_wally")

    def test_low_ball_inventory_without_a_planned_encounter_allows_progression(self):
        facts = self.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_available=True,
            pokeballs_ready=True,
        )
        facts = replace(
            facts,
            pokeballs_received=Fact.known(True),
            pokeballs_sufficient=Fact.known(False),
        )
        state = self.state(balls=1, campaign_facts=facts)

        selection = select_available_campaign_task(state)

        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "complete_petalburg_wally")

    def test_dependency_aware_planner_allows_progression_without_a_planned_encounter(self):
        facts = replace(
            self.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_available=True,
                pokeballs_ready=True,
            ),
            pokeballs_received=Fact.known(True),
            pokeballs_sufficient=Fact.known(False),
        )

        selection = plan_campaign(self.state(balls=1, campaign_facts=facts))

        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "complete_petalburg_wally")

    def test_restock_uses_nonzero_live_inventory_when_receipt_fact_is_unknown(self):
        facts = replace(
            self.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_ready=True,
            ),
            pokeballs_received=Fact.unknown(),
            pokeballs_sufficient=Fact.known(False),
        )

        for balls in (1, 3, 4):
            selection = select_available_campaign_task(self.state(balls=balls, campaign_facts=facts))
            self.assertTrue(
                selection.objective is None
                or selection.objective.objective_id != "restock_pokeballs",
                balls,
            )

        sufficient_facts = replace(facts, pokeballs_sufficient=Fact.known(True))
        sufficient_selection = select_available_campaign_task(self.state(balls=5, campaign_facts=sufficient_facts))
        self.assertTrue(
            sufficient_selection.objective is None
            or sufficient_selection.objective.objective_id != "restock_pokeballs"
        )

    def test_zero_ball_restock_requires_opening_receipt_and_planned_encounter(self):
        facts = self.facts(pokedex_received=True, pokeballs_ready=True)
        unknown_receipt = replace(
            facts,
            pokeballs_received=Fact.unknown(),
            pokeballs_sufficient=Fact.known(False),
        )
        self.assertNotEqual(
            select_available_campaign_task(self.state(balls=0, campaign_facts=unknown_receipt)).objective.objective_id,
            "restock_pokeballs",
        )

        received = replace(unknown_receipt, pokeballs_received=Fact.known(True))
        selection = select_available_campaign_task(self.state(balls=0, campaign_facts=received))
        self.assertTrue(
            selection.objective is None
            or selection.objective.objective_id != "restock_pokeballs"
        )

    def test_low_ball_reserve_suppresses_encounters_until_restock_is_selected(self):
        facts = replace(
            self.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
                pokedex_received=True,
                pokeballs_available=True,
                pokeballs_ready=True,
            ),
            pokeballs_received=Fact.known(True),
            pokeballs_sufficient=Fact.known(False),
        )
        state = replace(
            self.state(balls=3, campaign_facts=facts),
            encounters=Fact.known((LocationEncounter(MapRSE.ROUTE102.value, "none"),)),
        )

        graph = WorldMapGraph(
            (
                WorldEdge(MapRSE.OLDALE_TOWN.value, MapRSE.ROUTE102.value, "connection", (), (), estimated_cost=4),
                WorldEdge(MapRSE.ROUTE102.value, MapRSE.PETALBURG_CITY_GYM.value, "connection", (), (), estimated_cost=6),
                WorldEdge(MapRSE.OLDALE_TOWN.value, MapRSE.PETALBURG_CITY_GYM.value, "connection", (), (), estimated_cost=10),
            )
        )
        opportunity = campaign_objectives_module.EncounterOpportunity(
            MapRSE.ROUTE102.value, True, False, True
        )
        with (
            patch.object(campaign_objectives_module, "encounter_opportunities", return_value=(opportunity,)),
            patch.object(campaign_objectives_module, "get_world_map_graph", return_value=graph),
        ):
            available = available_campaign_tasks(state)
            selection = select_available_campaign_task(state)
            planned = plan_campaign(state)
        task_ids = tuple(task.objective_id for task in available)

        self.assertIn("restock_pokeballs", task_ids)
        self.assertNotIn("obtain_encounter:0:17", task_ids)
        self.assertEqual(selection.objective.objective_id, "restock_pokeballs")
        self.assertEqual(planned.objective.objective_id, "restock_pokeballs")

    def test_sufficient_ball_inventory_does_not_expose_restock(self):
        facts = replace(
            self.facts(pokeballs_ready=True),
            pokeballs_received=Fact.known(True),
            pokeballs_sufficient=Fact.known(True),
        )
        available = available_campaign_tasks(self.state(balls=5, campaign_facts=facts))

        self.assertNotIn("restock_pokeballs", tuple(task.objective_id for task in available))

    def test_pokeballs_available_without_ready_blocks_nuzlocke(self):
        # Test that pokeballs_available=True but pokeballs_ready=False keeps
        # Birch's combined handoff active (i.e., not complete).
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
                pokeballs_ready=False,
            ),
        )
        selection = select_campaign_objective(state)
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "receive_pokedex")

    def test_medicine_restock_waits_for_birch_pokeball_handoff(self):
        facts = self.facts(
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_received=False,
            pokeballs_ready=False,
        )
        state = self.state(area="ROUTE102", campaign_facts=facts)

        available = available_campaign_tasks(state)
        self.assertNotIn("restock_recovery_items", tuple(task.objective_id for task in available))
        self.assertEqual(plan_campaign(state).objective.objective_id, "receive_pokedex")
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
                "complete_petalburg_wally",
                "complete_petalburg_woods",
                "complete_rustboro_goods_stolen",
                "report_devon_goods",
                "recover_devon_goods",
                "return_devon_goods",
                "meet_mr_stone",
                "prepare_roxanne",
                "defeat_roxanne",
            ),
        )

    def test_post_pokeball_state_selects_petalburg_progression(self):
        # Keep this selector test independent of ignored, mutable emulator
        # profiles. The ROM-backed fixture is covered by live validation.
        state = self.state(
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
                pokeballs_ready=True,
                nuzlocke_started=True,
            )
        )
        selection = select_available_campaign_task(state)
        self.assertEqual(selection.status, ObjectiveStatus.READY)
        self.assertEqual(selection.objective.objective_id, "complete_petalburg_wally")
        self.assertEqual(selection.objective.task_kind, "required")
        self.assertEqual(selection.objective.destination, MapRSE.PETALBURG_CITY_GYM.value)

    def test_post_start_objectives_remain_ordered(self):
        complete = dict(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_available=True,
            pokeballs_ready=True,
            nuzlocke_started=True,
        )
        state = self.state(campaign_facts=self.facts(**complete))
        self.assertEqual(select_available_campaign_task(state).objective.objective_id, "complete_petalburg_wally")
        state = self.state(campaign_facts=self.facts(**complete, visited_petalburg=True))
        self.assertEqual(select_available_campaign_task(state).objective.objective_id, "complete_petalburg_wally")
        state = self.state(
            campaign_facts=self.facts(**complete, visited_petalburg=True, petalburg_wally_scene_complete=True)
        )
        self.assertEqual(select_available_campaign_task(state).objective.objective_id, "complete_petalburg_woods")
        state = self.state(
            campaign_facts=self.facts(
                **complete,
                visited_petalburg=True,
                petalburg_wally_scene_complete=True,
                petalburg_woods_scene_complete=True,
            )
        )
        self.assertEqual(
            select_available_campaign_task(state).objective.objective_id,
            "complete_rustboro_goods_stolen",
        )
        state = self.state(
            campaign_facts=self.facts(
                **complete,
                visited_petalburg=True,
                petalburg_wally_scene_complete=True,
                petalburg_woods_scene_complete=True,
                devon_goods_recovered=True,
            )
        )
        self.assertEqual(
            select_available_campaign_task(state).objective.objective_id,
            "complete_rustboro_goods_stolen",
        )
        state = self.state(
            campaign_facts=self.facts(
                **complete,
                visited_petalburg=True,
                petalburg_wally_scene_complete=True,
                petalburg_woods_scene_complete=True,
                devon_goods_recovered=True,
                visited_rustboro=True,
                devon_goods_stolen=True,
            )
        )
        self.assertEqual(select_available_campaign_task(state).objective.objective_id, "prepare_roxanne")

    def test_recovery_restock_interrupt_precedes_campaign_progression(self):
        complete = self.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_available=True,
            pokeballs_ready=True,
            nuzlocke_started=True,
        )
        state = self.state(area="ROUTE101", balls=5, campaign_facts=complete)
        state = replace(
            state,
            inventory=Fact.known(
                InventorySnapshot(
                    (),
                    (
                        ItemQuantity("Poke Ball", 5),
                        ItemQuantity("Antidote", 0),
                        ItemQuantity("Potion", 1),
                    ),
                    (),
                )
            ),
        )

        selection = select_available_campaign_task(state)

        self.assertEqual(selection.objective.objective_id, "restock_recovery_items")
        self.assertEqual(selection.reason, "selected required recovery-item restock task")
        self.assertEqual(plan_campaign(state).objective.objective_id, "restock_recovery_items")

    def test_unfundable_recovery_restock_defers_on_current_map(self):
        complete = self.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_available=True,
            pokeballs_ready=True,
            nuzlocke_started=True,
        )
        state = replace(
            self.state(area="ROUTE101", balls=5, campaign_facts=complete),
            inventory=Fact.known(
                InventorySnapshot(
                    (),
                    (
                        ItemQuantity("Poke Ball", 5),
                        ItemQuantity("Antidote", 0),
                        ItemQuantity("Potion", 1),
                    ),
                    (),
                )
            ),
        )
        campaign_objectives_module.context.campaign_restock_deferral = (
            "restock_recovery_items",
            MapRSE.OLDALE_TOWN.value,
        )

        selection = select_available_campaign_task(state)

        self.assertEqual(selection.objective.objective_id, "complete_petalburg_wally")
        self.assertEqual(plan_campaign(state).objective.objective_id, "complete_petalburg_wally")

    def test_unfundable_restock_deferral_survives_unknown_map_observation(self):
        complete = self.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_available=True,
            pokeballs_ready=True,
            nuzlocke_started=True,
        )
        state = replace(
            self.state(area="ROUTE101", balls=5, campaign_facts=complete),
            raw_map=Fact.unknown(),
            inventory=Fact.known(
                InventorySnapshot(
                    (),
                    (
                        ItemQuantity("Poke Ball", 5),
                        ItemQuantity("Antidote", 0),
                        ItemQuantity("Potion", 1),
                    ),
                    (),
                )
            ),
        )
        campaign_objectives_module.context.campaign_restock_deferral = (
            "restock_recovery_items",
            MapRSE.OLDALE_TOWN.value,
        )

        selection = select_available_campaign_task(state)

        self.assertEqual(selection.objective.objective_id, "complete_petalburg_wally")
        self.assertEqual(
            campaign_objectives_module.context.campaign_restock_deferral,
            ("restock_recovery_items", MapRSE.OLDALE_TOWN.value),
        )

    def test_active_wally_scene_keeps_restock_from_preempting_story_progression(self):
        complete = self.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=True,
            pokeballs_available=True,
            pokeballs_ready=True,
            nuzlocke_started=True,
        )
        state = replace(
            self.state(area="PETALBURG_CITY_GYM", balls=5, campaign_facts=complete),
            raw_map=Fact.known(MapRSE.PETALBURG_CITY_GYM.value),
            inventory=Fact.known(
                InventorySnapshot(
                    (),
                    (
                        ItemQuantity("Poke Ball", 5),
                        ItemQuantity("Antidote", 0),
                        ItemQuantity("Potion", 1),
                    ),
                    (),
                )
            ),
        )

        selection = select_available_campaign_task(state)

        self.assertEqual(selection.objective.objective_id, "complete_petalburg_wally")

    def test_first_badge_fixture_advances_from_new_run_to_authoritative_gym_flag(self):
        """Exercise the complete declarative slice through DEFEATED_RUSTBORO_GYM."""
        stages = [((), "set_text_speed")]
        completed: list[str] = []
        for fact, next_objective in (
            ("text_speed_fast", "complete_new_game_setup"),
            ("new_game_setup_complete", "set_wall_clock"),
            ("wall_clock_set", "meet_rival"),
            ("rival_met", "rescue_birch"),
            ("birch_rescued", "obtain_starter"),
            ("starter_obtained", "complete_intro_rival"),
            ("intro_rival_battle_complete", "receive_pokedex"),
            ("pokedex_received", "receive_pokedex"),
        ):
            completed.append(fact)
            stages.append((tuple(completed), next_objective))
        completed.extend(("pokeballs_available", "pokeballs_received"))
        for fact, next_objective in (
            ("pokeballs_ready", "complete_petalburg_wally"),
            ("petalburg_wally_scene_complete", "complete_petalburg_woods"),
            ("petalburg_woods_scene_complete", "complete_rustboro_goods_stolen"),
            ("devon_goods_stolen", "recover_devon_goods"),
            ("devon_goods_recovered", "prepare_roxanne"),
            ("prepare_roxanne", "defeat_roxanne"),
            ("first_badge_obtained", None),
        ):
            completed.append(fact)
            stages.append((tuple(completed), next_objective))

        for completed, expected_id in stages:
            state = self.state(
                balls=5 if "pokeballs_available" in completed else 0,
                campaign_facts=self.facts(**{name: True for name in completed}),
            )
            if "prepare_roxanne" in completed:
                state = replace(
                    state,
                    party=Fact.known(tuple(replace(pokemon, level=14) for pokemon in state.party.value)),
                )
            selection = plan_campaign(state, goal=ultimate_emerald_campaign_goal())
            if expected_id is None:
                self.assertEqual(selection.status, ObjectiveStatus.COMPLETE)
                self.assertIsNone(selection.objective)
            else:
                self.assertEqual(selection.status, ObjectiveStatus.READY, expected_id)
                self.assertEqual(selection.objective.objective_id, expected_id)

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

    def test_reachability_keeps_current_checkpoint_map_open_but_blocks_beyond_it(self):
        graph = WorldMapGraph(
            (
                WorldEdge(
                    MapRSE.OLDALE_TOWN.value,
                    MapRSE.ROUTE102.value,
                    "connection",
                    (),
                    (),
                    estimated_cost=1,
                ),
                WorldEdge(
                    MapRSE.ROUTE102.value,
                    MapRSE.PETALBURG_CITY.value,
                    "connection",
                    (),
                    (),
                    estimated_cost=1,
                ),
                WorldEdge(
                    MapRSE.PETALBURG_CITY.value,
                    MapRSE.ROUTE111.value,
                    "connection",
                    (),
                    (),
                    estimated_cost=1,
                ),
            )
        )
        state = self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))

        current_checkpoint = assess_campaign_reachability(state, MapRSE.ROUTE102.value, graph=graph)
        future_route = assess_campaign_reachability(state, MapRSE.ROUTE111.value, graph=graph)

        self.assertEqual(current_checkpoint.status, CampaignReachabilityStatus.REACHABLE_NOW)
        self.assertEqual(future_route.status, CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION)
        self.assertIn("route102:first_poochyena", future_route.blockers[0])
        self.assertFalse(future_route.executable_now)

        cleared_state = replace(
            state,
            campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True, visited_petalburg=True),
            badges=Fact.known(
                (
                    NamedFlag("BADGE01_GET", True),
                    NamedFlag("BADGE02_GET", True),
                )
            ),
        )
        self.assertEqual(
            assess_campaign_reachability(cleared_state, MapRSE.ROUTE111.value, graph=graph).status,
            CampaignReachabilityStatus.REACHABLE_NOW,
        )

    def test_oldale_west_exit_requires_the_pokedex_adventure_flag_boundary(self):
        graph = WorldMapGraph(
            (
                WorldEdge(
                    MapRSE.OLDALE_TOWN.value,
                    MapRSE.ROUTE102.value,
                    "connection",
                    (),
                    (),
                    estimated_cost=1,
                ),
            )
        )
        requirements = emerald_map_access_requirements()
        before_pokedex = self.state(campaign_facts=self.facts(pokedex_received=False))
        blocked = assess_campaign_reachability(
            before_pokedex,
            MapRSE.ROUTE102.value,
            graph=graph,
            access_requirements=requirements,
        )

        self.assertEqual(blocked.status, CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION)
        self.assertEqual(blocked.access_requirement_ids, ("route102:oldale_adventure_started",))
        self.assertIn("route102:oldale_adventure_started: missing campaign fact pokedex_received", blocked.blockers)

        after_pokedex = replace(
            before_pokedex,
            campaign_facts=self.facts(pokedex_received=True),
        )
        opened = assess_campaign_reachability(
            after_pokedex,
            MapRSE.ROUTE102.value,
            graph=graph,
            access_requirements=requirements,
        )
        self.assertEqual(opened.status, CampaignReachabilityStatus.REACHABLE_NOW)

    def test_current_map_trainer_does_not_block_unrelated_route101_or_route103(self):
        graph = WorldMapGraph(
            (
                WorldEdge(MapRSE.ROUTE102.value, MapRSE.ROUTE101.value, "connection", (), (), estimated_cost=1),
                WorldEdge(MapRSE.ROUTE102.value, MapRSE.ROUTE103.value, "connection", (), (), estimated_cost=1),
            )
        )
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            raw_map=Fact.known(MapRSE.ROUTE102.value),
        )

        for destination in (MapRSE.ROUTE101.value, MapRSE.ROUTE103.value):
            result = assess_campaign_reachability(state, destination, graph=graph)
            self.assertEqual(result.status, CampaignReachabilityStatus.REACHABLE_NOW)
            self.assertEqual(result.blockers, ())

    def test_emerald_registry_blocks_future_encounter_corridors_before_story_opening(self):
        graph = WorldMapGraph(
            (
                WorldEdge(MapRSE.OLDALE_TOWN.value, MapRSE.ROUTE110.value, "connection", (), (), estimated_cost=1),
                WorldEdge(MapRSE.OLDALE_TOWN.value, MapRSE.ROUTE116.value, "connection", (), (), estimated_cost=1),
            )
        )
        state = self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        requirements = emerald_map_access_requirements()

        route110 = assess_campaign_reachability(
            state,
            MapRSE.ROUTE110.value,
            graph=graph,
            access_requirements=requirements,
        )
        route116 = assess_campaign_reachability(
            state,
            MapRSE.ROUTE116.value,
            graph=graph,
            access_requirements=requirements,
        )

        self.assertEqual(route110.status, CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION)
        self.assertIn("route110:rustboro_corridor: missing campaign fact visited_rustboro", route110.blockers)
        self.assertEqual(route116.status, CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION)
        self.assertIn("route116:rustboro: missing campaign fact visited_rustboro", route116.blockers)

        opened = replace(
            state,
            campaign_facts=self.facts(
                pokedex_received=True,
                pokeballs_ready=True,
                petalburg_wally_scene_complete=True,
                petalburg_woods_scene_complete=True,
                visited_rustboro=True,
            ),
        )
        self.assertEqual(
            assess_campaign_reachability(
                opened,
                MapRSE.ROUTE110.value,
                graph=graph,
                access_requirements=requirements,
            ).status,
            CampaignReachabilityStatus.REACHABLE_NOW,
        )

    def test_route111_requires_both_badges_even_after_early_trainer_clearance(self):
        graph = WorldMapGraph(
            (
                WorldEdge(MapRSE.OLDALE_TOWN.value, MapRSE.ROUTE102.value, "connection", (), (), estimated_cost=1),
                WorldEdge(MapRSE.ROUTE102.value, MapRSE.PETALBURG_CITY.value, "connection", (), (), estimated_cost=1),
                WorldEdge(MapRSE.PETALBURG_CITY.value, MapRSE.ROUTE111.value, "connection", (), (), estimated_cost=1),
            )
        )
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True, visited_petalburg=True)),
            badges=Fact.known(
                (
                    NamedFlag("BADGE01_GET", True),
                    NamedFlag("BADGE02_GET", False),
                )
            ),
        )

        result = assess_campaign_reachability(state, MapRSE.ROUTE111.value, graph=graph)

        self.assertEqual(result.status, CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION)
        self.assertIn("route111:two_badges: missing badge BADGE02_GET", result.blockers)
        self.assertEqual(result.access_requirement_ids, ("route111:two_badges",))

    def test_unknown_badge_does_not_open_future_map(self):
        graph = WorldMapGraph(
            (
                WorldEdge(MapRSE.OLDALE_TOWN.value, MapRSE.ROUTE111.value, "connection", (), (), estimated_cost=1),
            )
        )
        state = replace(
            self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True)),
            badges=Fact.unknown(),
        )

        result = assess_campaign_reachability(state, MapRSE.ROUTE111.value, graph=graph)

        self.assertEqual(result.status, CampaignReachabilityStatus.UNKNOWN)
        self.assertIn("route111:two_badges: BADGE01_GET is unresolved", result.blockers)

    def test_new_mauville_requires_surf_and_basement_key(self):
        graph = WorldMapGraph(
            (
                WorldEdge(MapRSE.OLDALE_TOWN.value, MapRSE.NEW_MAUVILLE_ENTRANCE.value, "connection", (), (), estimated_cost=1),
                WorldEdge(MapRSE.NEW_MAUVILLE_ENTRANCE.value, MapRSE.NEW_MAUVILLE_INSIDE.value, "warp", (), (), estimated_cost=1),
            )
        )
        requirements = (
            CampaignAccessRequirement(
                "newmauville:surf",
                MapRSE.NEW_MAUVILLE_ENTRANCE.value,
                required_capabilities=("surf_available",),
            ),
            CampaignAccessRequirement(
                "newmauville:basement_key",
                MapRSE.NEW_MAUVILLE_INSIDE.value,
                required_capabilities=("surf_available", "basement_key_obtained"),
            ),
        )
        state = self.state(
            campaign_facts=replace(
                self.facts(pokedex_received=True, pokeballs_ready=True),
                surf_available=Fact.known(False),
                basement_key_obtained=Fact.known(False),
            )
        )

        blocked = assess_campaign_reachability(
            state,
            MapRSE.NEW_MAUVILLE_INSIDE.value,
            graph=graph,
            access_requirements=requirements,
        )

        self.assertEqual(blocked.status, CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION)
        self.assertIn("newmauville:surf: missing capability surf_available", blocked.blockers)
        self.assertIn("newmauville:basement_key: missing capability basement_key_obtained", blocked.blockers)

        ready_state = replace(
            state,
            campaign_facts=replace(
                state.campaign_facts,
                surf_available=Fact.known(True),
                basement_key_obtained=Fact.known(True),
            ),
        )
        ready = assess_campaign_reachability(
            ready_state,
            MapRSE.NEW_MAUVILLE_INSIDE.value,
            graph=graph,
            access_requirements=requirements,
        )
        self.assertEqual(ready.status, CampaignReachabilityStatus.REACHABLE_NOW)

    def test_frontier_ids_hide_deferred_encounters_but_keep_them_in_policy_input(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY.value,
        )
        future = replace(
            encounter_task(MapRSE.NEW_MAUVILLE_INSIDE.value),
            encounter_evaluation=EncounterEvaluation(
                task_id="obtain_encounter:newmauville",
                location=MapRSE.NEW_MAUVILLE_INSIDE.value,
                progression_destination=MapRSE.PETALBURG_CITY.value,
                classification=EncounterClassification.UNREACHABLE,
                recommendation=EncounterRecommendation.UNAVAILABLE,
                direct_progression_cost=10,
                encounter_cost=20,
                encounter_to_progression_cost=10,
                via_encounter_cost=30,
                detour_cost=20,
                detour_ratio=2.0,
                reachability_status=CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION,
                reachability_blockers=("newmauville:basement_key: missing capability basement_key_obtained",),
            ),
        )
        state = self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        with patch.object(
            campaign_objectives_module,
            "available_campaign_tasks",
            return_value=(required, future),
        ):
            selection = select_available_campaign_task(state)

        self.assertEqual(selection.objective.objective_id, required.objective_id)
        self.assertEqual(selection.frontier_objective_ids, (required.objective_id,))

    def test_inaccessible_encounter_cannot_take_ownership_from_required_progression(self):
        required = replace(
            self.objective("complete_petalburg_wally", self.predicate("reached", False)),
            destination=MapRSE.PETALBURG_CITY.value,
        )
        future = replace(
            encounter_task(MapRSE.ROUTE111.value),
            encounter_evaluation=EncounterEvaluation(
                task_id="obtain_encounter:route111",
                location=MapRSE.ROUTE111.value,
                progression_destination=MapRSE.PETALBURG_CITY.value,
                classification=EncounterClassification.UNREACHABLE,
                recommendation=EncounterRecommendation.UNAVAILABLE,
                direct_progression_cost=10,
                encounter_cost=20,
                encounter_to_progression_cost=10,
                via_encounter_cost=30,
                detour_cost=20,
                detour_ratio=2.0,
                reachability_status=CampaignReachabilityStatus.REACHABLE_AFTER_PREPARATION,
                reachability_blockers=("route102:first_poochyena: mandatory trainer preparation required",),
            ),
        )
        state = self.state(campaign_facts=self.facts(pokedex_received=True, pokeballs_ready=True))
        candidate = EncounterCandidate(MapRSE.ROUTE111.value, EncounterMethod.LAND)
        with patch.object(
            campaign_objectives_module,
            "_current_campaign_encounter_candidates",
            return_value=(candidate,),
        ):
            selection = campaign_objectives_module._select_available_campaign_tasks((required, future), state)

        self.assertEqual(selection.objective.objective_id, required.objective_id)


if __name__ == "__main__":
    unittest.main()
