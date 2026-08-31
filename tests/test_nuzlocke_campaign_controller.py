import unittest
from dataclasses import replace
from types import SimpleNamespace

from modules.goals import Goal, ReachLocation
from modules.navigation import IntermediateRouteAnalysis, RouteAnalysis
from modules.nuzlocke.campaign_controller import CampaignController, CampaignControllerStatus
from modules.nuzlocke.campaign_execution import CampaignExecutionResult, CampaignExecutionStatus, adapt_campaign_execution
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    ObjectiveSelection,
    ObjectiveStatus,
    initial_emerald_campaign,
    plan_campaign,
    select_campaign_objective,
    ultimate_emerald_campaign_goal,
)
from modules.nuzlocke.campaign_state import Fact, RunStatus
from modules.nuzlocke.readiness_diagnostics import Availability, PartyReadinessMember, ProgressionReadinessDiagnostic

from tests.test_nuzlocke_campaign_objectives import CampaignObjectiveTests


class CampaignControllerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = CampaignObjectiveTests()
        self.fixture.setUp()
        self.current = self.fixture.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
            )
        )
        self.created: list[Goal] = []
        self.executed = 0

    def state(self, **changes):
        self.current = replace(self.current, **changes)

    def factory(self, goal):
        self.created.append(goal)

        def tactical():
            self.executed += 1
            yield
            while True:
                yield

        return tactical()

    def controller(self):
        return CampaignController(lambda: self.current, tactical_loop_factory=self.factory)

    def test_ready_intro_battle_mounts_existing_tactical_goal(self):
        controller = self.controller()
        result = controller.refresh()
        self.assertEqual(result.objective_id, "complete_intro_rival")
        self.assertEqual(result.status, CampaignControllerStatus.READY)

    def test_completed_intro_battle_reselects_next_declarative_objective(self):
        controller = self.controller()
        controller.refresh()
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
            )
        )
        result = controller.refresh()
        self.assertEqual(result.selection.objective.objective_id, "receive_pokedex")
        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.objective_id, "receive_pokedex")

    def test_blocked_and_unknown_do_not_execute(self):
        for status in (ObjectiveStatus.BLOCKED, ObjectiveStatus.UNKNOWN):
            selection = ObjectiveSelection(None, status, status.value)
            controller = CampaignController(
                lambda: self.current, tactical_loop_factory=self.factory, selector=lambda _: selection
            )
            result = controller.step()
            self.assertEqual(result.status.value, status.value)
            self.assertEqual(self.executed, 0)
            self.assertEqual(controller.step().status.value, status.value)
            self.assertEqual(self.executed, 0)

    def test_refresh_exception_is_retried_without_unwinding_campaign_mode(self):
        calls = []
        objective = self.fixture.objective("retry", self.fixture.predicate("retry_done", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "retry")

        def state_provider():
            calls.append("state")
            if len(calls) == 1:
                raise RuntimeError("transient post-battle observation")
            return self.current

        def capability():
            calls.append("capability")
            yield

        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "retry capability",
            capability=capability,
        )
        controller = CampaignController(
            state_provider,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )

        first = controller.step()
        second = controller.step()

        self.assertEqual(first.status, CampaignControllerStatus.UNKNOWN)
        self.assertIn("will re-evaluate", first.reason)
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertEqual(calls, ["state", "state", "capability"])

    def test_failed_selection_is_terminal(self):
        selection = ObjectiveSelection(None, ObjectiveStatus.FAILED, "failed")
        controller = CampaignController(lambda: self.current, selector=lambda _: selection)
        self.assertEqual(controller.step().status, CampaignControllerStatus.FAILED)
        self.assertEqual(controller.step().status, CampaignControllerStatus.FAILED)

    def test_unsupported_execution_is_safe(self):
        selection = ObjectiveSelection(None, ObjectiveStatus.READY, "bad")
        unsupported = CampaignExecutionResult(None, CampaignExecutionStatus.UNSUPPORTED, "unsupported")
        controller = CampaignController(
            lambda: self.current, selector=lambda _: selection, adapter=lambda _: unsupported
        )
        result = controller.step()
        self.assertEqual(result.status, CampaignControllerStatus.FAILED)
        self.assertEqual(self.executed, 0)

        malformed = CampaignExecutionResult(None, CampaignExecutionStatus.READY, "missing goal")
        controller = CampaignController(lambda: self.current, selector=lambda _: selection, adapter=lambda _: malformed)
        self.assertEqual(controller.step().status, CampaignControllerStatus.FAILED)

    def test_active_objective_is_not_reissued(self):
        controller = self.controller()
        controller.step()
        controller.step()
        # Introductory-rival execution is now mounted through the resource
        # capability boundary; the legacy tactical-loop factory is not used.
        self.assertEqual(len(self.created), 0)
        self.assertEqual(self.executed, 0)

    def test_capability_remains_active_until_authoritative_fact_changes(self):
        calls = []

        def capability():
            calls.append("mounted")
            yield
            yield

        objective = self.fixture.objective("multi_frame", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )
        controller.step()
        controller.step()
        self.assertEqual(calls, ["mounted"])

    def test_completed_safety_fact_does_not_replace_running_capability(self):
        healed = [False]
        safety = CampaignObjective(
            "HEAL_PARTY",
            "Restore the party",
            (),
            CampaignPredicate("healed", "healed", lambda _: Fact.known(healed[0])),
            task_kind="safety",
        )
        parent = self.fixture.objective("parent", self.fixture.predicate("parent_done", False))
        selections = iter(
            (
                ObjectiveSelection(safety, ObjectiveStatus.READY, "safety ready"),
                ObjectiveSelection(parent, ObjectiveStatus.READY, "parent ready"),
            )
        )
        advanced = []

        def adapter(selection):
            def capability():
                advanced.append(selection.objective.objective_id)
                yield
                if selection.objective is safety:
                    healed[0] = True
                    advanced.append("safety-finished")
                    return
                advanced.append("parent-ran")
                yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        controller.step()
        # The resource is restored before the interaction capability returns.
        healed[0] = True
        # The safety generator has not yet returned, so it must receive the
        # next controller step even though its completion fact is now true.
        controller.step()
        self.assertEqual(advanced, ["HEAL_PARTY", "safety-finished"])

        controller.step()
        self.assertEqual(advanced, ["HEAL_PARTY", "safety-finished", "parent"])

    def test_active_capability_owns_transient_planner_regression(self):
        first = self.fixture.objective("active", self.fixture.predicate("active_done", False))
        second = self.fixture.objective("transient", self.fixture.predicate("transient_done", False))
        selections = iter(
            (
                ObjectiveSelection(first, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(second, ObjectiveStatus.READY, "transient planner result"),
            )
        )
        advanced = []

        def capability():
            while True:
                advanced.append("active")
                yield

        execution = CampaignExecutionResult(
            first,
            CampaignExecutionStatus.READY,
            "active capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "active")
        self.assertEqual(advanced, ["active", "active"])

    def test_canonical_later_objective_owns_transient_earlier_selection(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        active = objectives["prepare_roxanne"]
        earlier = objectives["set_text_speed"]
        self.state(campaign_facts=self.fixture.facts(visited_rustboro=True))
        selections = iter(
            (
                ObjectiveSelection(active, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "transient planner result"),
            )
        )
        advanced = []

        def capability():
            while True:
                advanced.append("prepare_roxanne")
                yield

        execution = CampaignExecutionResult(
            active,
            CampaignExecutionStatus.READY,
            "active capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "prepare_roxanne")
        self.assertEqual(advanced, ["prepare_roxanne", "prepare_roxanne"])

    def test_later_active_objective_survives_unavailable_battle_boundary(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        active = objectives["prepare_roxanne"]
        earlier = objectives["set_text_speed"]
        # The live battle-start snapshot has no field map or coordinates. Its
        # save-backed facts can consequently make the active prerequisite look
        # false for one frame even though no coherent save regression exists.
        self.state(
            raw_map=Fact.unknown(),
            coordinates=Fact.unknown(),
            campaign_facts=self.fixture.facts(),
        )
        selections = iter(
            (
                ObjectiveSelection(active, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "battle boundary"),
            )
        )
        advanced = []

        def capability():
            while True:
                advanced.append("prepare_roxanne")
                yield

        execution = CampaignExecutionResult(
            active,
            CampaignExecutionStatus.READY,
            "active capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=lambda _: execution,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "prepare_roxanne")
        self.assertEqual(advanced, ["prepare_roxanne", "prepare_roxanne"])

    def test_deferred_readiness_owns_objective_across_planner_handoff(self):
        pending = self.fixture.objective("pending_readiness", self.fixture.predicate("done", False))
        transient = self.fixture.objective("transient_handoff", self.fixture.predicate("done", False))
        selections = iter(
            (
                ObjectiveSelection(pending, ObjectiveStatus.READY, "pending readiness"),
                ObjectiveSelection(transient, ObjectiveStatus.READY, "transition observation"),
            )
        )
        adapted = []
        readiness_calls = []
        recovery_starts = []

        def readiness_provider(*_):
            readiness_calls.append(1)
            recovering = len(readiness_calls) == 2
            return ProgressionReadinessDiagnostic(
                objective_id="pending_readiness",
                objective_status="ready",
                destination=None,
                navigation_goal=Goal(),
                current_map=(1, 2),
                current_coordinates=(3, 4),
                game_state="OVERWORLD" if recovering else "BATTLE",
                campaign_mode="test",
                party=(PartyReadinessMember(0, "Treecko", 1, 10, 0.1, False, True, "none"),),
                has_usable_pokemon=True,
                healing_available=True,
                recovery=None,
                trainers=(),
                party_availability=Availability.KNOWN,
                trainer_availability=Availability.KNOWN,
                recovery_availability=Availability.KNOWN,
                navigation_availability=Availability.KNOWN,
                overworld_availability=Availability.KNOWN,
                resource_availability=Availability.KNOWN,
                route_analysis=(
                    RouteAnalysis(
                        Goal(),
                        "normal-route",
                        40,
                        (
                            IntermediateRouteAnalysis(
                                ReachLocation(("OLDale", (5, 6))),
                                48,
                                8,
                                True,
                                first_route=SimpleNamespace(
                                    metrics=SimpleNamespace(total_route_cost=8),
                                ),
                            ),
                        ),
                    )
                    if recovering
                    else None
                ),
            )

        def adapter(selection):
            adapted.append(selection.objective.objective_id)
            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "ready",
                tactical_goal=Goal(),
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
            readiness_provider=readiness_provider,
            recovery_factory=lambda _: iter((recovery_starts.append("pending_readiness"),)),
        )

        first = controller.step()
        second = controller.step()

        self.assertEqual(first.status, CampaignControllerStatus.READY)
        self.assertIn("readiness deferred", first.reason)
        self.assertEqual(second.execution_phase, "RECOVERY")
        self.assertEqual(adapted, ["pending_readiness", "pending_readiness"])
        self.assertEqual(recovery_starts, ["pending_readiness"])
        self.assertEqual(controller.current_objective_id, "pending_readiness")

    def test_targetless_capability_mounts_across_unknown_post_battle_readiness(self):
        objective = next(
            item for item in initial_emerald_campaign() if item.objective_id == "receive_pokedex"
        )
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "pokedex frontier")
        mounted = []

        def capability():
            mounted.append(objective.objective_id)
            yield

        unknown = ProgressionReadinessDiagnostic(
            objective_id="receive_pokedex",
            objective_status="ready",
            destination=None,
            navigation_goal=None,
            current_map=(0, 18),
            current_coordinates=None,
            game_state="OVERWORLD",
            campaign_mode="test",
            party=(),
            has_usable_pokemon=None,
            healing_available=None,
            recovery=None,
            trainers=(),
            party_availability=Availability.UNKNOWN,
            trainer_availability=Availability.UNKNOWN,
            recovery_availability=Availability.UNKNOWN,
            navigation_availability=Availability.NOT_APPLICABLE,
            overworld_availability=Availability.UNKNOWN,
            resource_availability=Availability.UNKNOWN,
            readiness_decision=None,
            readiness_reason=None,
        )
        execution = CampaignExecutionResult(
            objective,
            CampaignExecutionStatus.READY,
            "pokedex capability",
            capability=capability,
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
            readiness_provider=lambda *_: unknown,
        )

        result = controller.step()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.objective_id, "receive_pokedex")
        self.assertEqual(mounted, ["receive_pokedex"])
        self.assertIsNotNone(controller._tactical_loop)

    def test_later_active_objective_releases_on_coherent_save_regression(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        active = objectives["prepare_roxanne"]
        earlier = objectives["set_text_speed"]
        self.state(campaign_facts=self.fixture.facts())
        selections = iter(
            (
                ObjectiveSelection(active, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "coherent regression"),
            )
        )

        def adapter(selection):
            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=lambda: iter(("advanced",)),
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "set_text_speed")

    def test_canonical_earlier_objective_cannot_retain_over_later_selection(self):
        objectives = {objective.objective_id: objective for objective in initial_emerald_campaign()}
        earlier = objectives["set_text_speed"]
        later = objectives["prepare_roxanne"]
        selections = iter(
            (
                ObjectiveSelection(earlier, ObjectiveStatus.READY, "active"),
                ObjectiveSelection(later, ObjectiveStatus.READY, "later planner result"),
            )
        )
        mounted = []

        def adapter(selection):
            def capability():
                mounted.append(selection.objective.objective_id)
                yield
                while True:
                    yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: next(selections),
            adapter=adapter,
        )

        controller.step()
        result = controller.step()

        self.assertEqual(result.objective_id, "prepare_roxanne")
        self.assertEqual(mounted, ["set_text_speed", "prepare_roxanne"])

    def test_capability_only_execution_advances_to_input_boundary(self):
        objective = self.fixture.objective("dialogue", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "actionable dialogue")
        emitted = []

        def capability():
            emitted.append("ADVANCE_DIALOGUE")
            yield

        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability-only dialogue", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )

        state = controller.step()

        self.assertEqual(state.status, CampaignControllerStatus.READY)
        self.assertIs(controller.last_execution.capability, capability)
        self.assertEqual(emitted, ["ADVANCE_DIALOGUE"])
        self.assertIsNone(state.tactical_goal)

    def test_capability_boundary_is_not_failure(self):
        objective = self.fixture.objective("boundary", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")

        def capability():
            if False:
                yield

        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )
        state = controller.step()
        self.assertEqual(state.status, CampaignControllerStatus.READY)
        self.assertIn("capability boundary", state.reason)

    def test_completed_rival_capability_hands_off_to_pokedex_frontier(self):
        mounted = []

        def adapter(selection):
            objective_id = selection.objective.objective_id

            def capability():
                mounted.append(objective_id)
                if objective_id == "complete_intro_rival":
                    return
                yield

            return CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.READY,
                "capability",
                capability=capability,
            )

        controller = CampaignController(
            lambda: self.current,
            selector=lambda state: plan_campaign(state, goal=ultimate_emerald_campaign_goal()),
            adapter=adapter,
        )

        first = controller.step()
        self.assertIn("capability boundary", first.reason)
        self.assertEqual(mounted, ["complete_intro_rival"])

        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
            )
        )
        second = controller.step()

        self.assertEqual(second.objective_id, "receive_pokedex")
        self.assertEqual(mounted, ["complete_intro_rival", "receive_pokedex"])

    def test_failed_capability_is_cleared_and_can_be_rebuilt(self):
        objective = self.fixture.objective("recoverable", self.fixture.predicate("complete", False))
        selection = ObjectiveSelection(objective, ObjectiveStatus.READY, "ready")
        attempts = []

        def capability():
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise RuntimeError("temporary recovery lookup failure")
            yield

        execution = CampaignExecutionResult(
            objective, CampaignExecutionStatus.READY, "capability", capability=capability
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda _: selection,
            adapter=lambda _: execution,
        )
        first = controller.step()
        self.assertEqual(first.status, CampaignControllerStatus.UNKNOWN)
        self.assertIsNone(controller._tactical_loop)
        second = controller.step()
        self.assertEqual(second.status, CampaignControllerStatus.READY)
        self.assertEqual(attempts, [0, 1])

    def test_completion_clears_goal_before_unsupported_next_objective(self):
        controller = self.controller()
        controller.refresh()
        old_goal = controller.current_tactical_goal
        self.state(
            campaign_facts=self.fixture.facts(
                text_speed_fast=True,
                new_game_setup_complete=True,
                wall_clock_set=True,
                rival_met=True,
                birch_rescued=True,
                starter_obtained=True,
                intro_rival_battle_complete=True,
            )
        )
        result = controller.refresh()
        self.assertNotEqual(old_goal, None)
        self.assertIsNone(result.tactical_goal)
        self.assertEqual(result.selection.objective.objective_id, "receive_pokedex")
        self.assertEqual(self.created.__len__(), 0)

    def test_pokedex_boundary_requires_no_controller_handoff(self):
        self.state(
            campaign_facts=self.fixture.facts(
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
        controller = CampaignController(
            lambda: self.current,
            selector=lambda state: plan_campaign(state, goal=ultimate_emerald_campaign_goal()),
            adapter=adapt_campaign_execution,
            tactical_loop_factory=self.factory,
        )
        result = controller.refresh()
        self.assertEqual(result.selection.objective.objective_id, "reach_petalburg")
        self.assertEqual(result.status, CampaignControllerStatus.READY)

    def test_petalburg_progression_is_gated_by_pokedex_receipt(self):
        objective = next(item for item in initial_emerald_campaign() if item.objective_id == "reach_petalburg")
        facts = self.fixture.facts(
            text_speed_fast=True,
            new_game_setup_complete=True,
            wall_clock_set=True,
            rival_met=True,
            birch_rescued=True,
            starter_obtained=True,
            intro_rival_battle_complete=True,
            pokedex_received=False,
            pokeballs_available=True,
            pokeballs_ready=True,
        )
        state = self.fixture.state(campaign_facts=facts)
        selection = select_campaign_objective(state, (objective,))
        self.assertEqual(selection.status, ObjectiveStatus.BLOCKED)

    def test_controller_mounts_preparation_after_pokedex_boundary(self):
        self.current = replace(
            self.current,
            campaign_facts=self.fixture.facts(
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
            party=Fact.known((replace(self.current.party.value[0], current_hp=19),)),
        )
        controller = CampaignController(
            lambda: self.current,
            selector=lambda state: plan_campaign(state, goal=ultimate_emerald_campaign_goal()),
            adapter=adapt_campaign_execution,
        )

        result = controller.refresh()

        self.assertEqual(result.status, CampaignControllerStatus.READY)
        self.assertEqual(result.objective_id, "reach_petalburg")
        self.assertEqual(result.execution.execution_id, "reach_petalburg")
        self.assertIsNotNone(result.execution.capability)


from modules.map_data import MapRSE

if __name__ == "__main__":
    unittest.main()
