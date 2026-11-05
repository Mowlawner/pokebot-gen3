import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
import modules.modes.campaign as campaign_module

from modules.modes.campaign import (
    CampaignProgressionMode,
    _mandatory_trainer_on_outbound_map,
    _known_trainer_checkpoint_on_route,
    _encounter_stage_route_risks,
    _trainer_risk_selected_for_engagement,
    _trainer_checkpoints_for_battle_start,
    _overworld_position_is_coherent,
)
from modules.map_data import MapRSE, PokemonCenter
from modules.goals import (
    EncounterMode,
    EngageTrainer,
    GoalConstraints,
    NavigationGoal,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    TrainerMode,
)
from modules.navigation import (
    NavigationAction,
    NavigationActionType,
    NavigationPlan,
    TrainerAvoidability,
    TrainerRouteRisk,
)
from modules.nuzlocke.readiness_diagnostics import CampaignReadinessPolicy
from modules.nuzlocke.resource_policy import (
    PartyResource,
    ResourceObservationStatus,
    RouteRecovery,
    ResourceSnapshot,
)
from modules.overworld import OverworldObservationResult, OverworldObservationStatus, TriggerObservation
from modules.nuzlocke.campaign_controller import CampaignController
from modules.nuzlocke.campaign_execution import CampaignExecutionResult, CampaignExecutionStatus
from modules.nuzlocke.campaign_objectives import (
    CampaignObjective,
    CampaignPredicate,
    ObjectiveSelection,
    ObjectiveStatus,
    plan_campaign,
)
from modules.nuzlocke.campaign_state import Fact, RunStatus
from modules.memory import GameState
from modules.battle_state import BattleOutcome
from modules.nuzlocke.readiness_diagnostics import Availability


class CampaignOrchestrationTests(unittest.TestCase):
    def test_blocked_encounter_stage_remains_unknown_until_forced_exposure_is_proven(self):
        evidence = SimpleNamespace(
            blocked=True,
            current_map=MapRSE.ROUTE104.value,
            known_hazard_ids=(f"trainer:{MapRSE.ROUTE104.value}:3",),
            unresolved_hazard_ids=(),
        )
        with patch.object(
            campaign_module,
            "context",
            SimpleNamespace(campaign_encounter_stage=evidence),
        ):
            risks = _encounter_stage_route_risks(
                SimpleNamespace(
                    map_id=MapRSE.ROUTE104.value,
                ),
                SimpleNamespace(objective_id="obtain_encounter:0:19"),
            )

        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0].trainer_id, f"trainer:{MapRSE.ROUTE104.value}:3")
        self.assertIs(risks[0].avoidability, TrainerAvoidability.UNKNOWN)
        self.assertEqual(risks[0].confidence, "unknown")

    def test_forced_encounter_stage_can_claim_known_unavoidable_route_risk(self):
        evidence = SimpleNamespace(
            blocked=False,
            current_map=MapRSE.ROUTE104.value,
            known_hazard_ids=(f"trainer:{MapRSE.ROUTE104.value}:3",),
            unresolved_hazard_ids=(),
            reason="forced_trainer_exposure:avoid_unavailable_ignore_available:candidate=0",
        )
        with patch.object(
            campaign_module,
            "context",
            SimpleNamespace(campaign_encounter_stage=evidence),
        ):
            risks = _encounter_stage_route_risks(
                SimpleNamespace(map_id=MapRSE.ROUTE104.value),
                SimpleNamespace(objective_id="obtain_encounter:0:19"),
            )

        self.assertEqual(len(risks), 1)
        self.assertIs(risks[0].avoidability, TrainerAvoidability.UNAVOIDABLE)
        self.assertEqual(risks[0].confidence, "known")

    def test_encounter_stage_reuses_checkpoint_when_live_script_disappears(self):
        checkpoint = SimpleNamespace(
            checkpoint_id="route104:cindy",
            trainer_affordance_id="trainer:cindy",
        )
        evidence = SimpleNamespace(known_hazard_checkpoints=(checkpoint,))
        with patch.object(campaign_module, "trainer_checkpoint_for_source", return_value=None):
            resolved = campaign_module._encounter_stage_checkpoint_for_id(
                SimpleNamespace(objects=(), triggers=()),
                "trainer:cindy",
                evidence,
            )

        self.assertIs(resolved, checkpoint)

    def test_trainer_risk_threshold_promotes_only_high_cost_routes(self):
        route = SimpleNamespace(
            trainer_risks=(
                TrainerRouteRisk("trainer:cheap", TrainerAvoidability.COSTLY),
                TrainerRouteRisk("trainer:calvin", TrainerAvoidability.PRACTICALLY_UNAVOIDABLE),
            )
        )

        self.assertEqual(
            _trainer_risk_selected_for_engagement(route, CampaignReadinessPolicy()),
            "trainer:calvin",
        )
        self.assertIsNone(
            _trainer_risk_selected_for_engagement(
                SimpleNamespace(
                    trainer_risks=(TrainerRouteRisk("trainer:cheap", TrainerAvoidability.COSTLY),)
                ),
                CampaignReadinessPolicy(),
            )
        )

    def test_trainer_risk_threshold_can_be_lowered_to_prepare_costly_route(self):
        risk = TrainerRouteRisk("trainer:costly", TrainerAvoidability.COSTLY)
        policy = CampaignReadinessPolicy(trainer_risk_prepare_at="costly")

        self.assertEqual(
            _trainer_risk_selected_for_engagement(SimpleNamespace(trainer_risks=(risk,)), policy),
            "trainer:costly",
        )

    def test_route_trainer_risks_publish_rom_backed_facts(self):
        visible = (SimpleNamespace(checkpoint_id="visible", map_id=(0, 1)),)
        resolved = SimpleNamespace(
            checkpoint_id="risk",
            map_id=(0, 2),
            trainer_affordance_id="trainer:risk",
            display_name="Route Trainer",
            source="active_rom",
        )
        route = SimpleNamespace(
            trainer_risks=(SimpleNamespace(trainer_id="trainer:risk"),),
        )
        overworld = SimpleNamespace(objects=(), triggers=())
        with (
            patch.object(campaign_module, "_trainer_script_for_id", return_value="TrainerScript"),
            patch.object(campaign_module, "trainer_checkpoint_for_source", return_value=resolved),
        ):
            facts = campaign_module._merge_route_trainer_facts(overworld, visible, route)

        self.assertEqual(tuple(fact.checkpoint_id for fact in facts), ("visible", "risk"))

    def test_route_trainer_facts_drop_visible_trainers_outside_active_route(self):
        visible = (
            SimpleNamespace(
                checkpoint_id="stale",
                trainer_affordance_id="trainer:other",
                map_id=MapRSE.ROUTE103.value,
            ),
        )
        resolved = SimpleNamespace(
            checkpoint_id="risk",
            trainer_affordance_id="trainer:risk",
            map_id=MapRSE.ROUTE102.value,
            display_name="Route Trainer",
            source="active_rom",
        )
        route = NavigationPlan(
            actions=(
            NavigationAction(
                    NavigationActionType.MOVE,
                    direction=object(),
                    source=(MapRSE.ROUTE102.value, (1, 1)),
                    destination=(MapRSE.ROUTE102.value, (1, 2)),
                ),
            ),
            destination=(MapRSE.PETALBURG_CITY.value, (1, 1)),
        )
        route_analysis = SimpleNamespace(
            normal_route=route,
            trainer_risks=(SimpleNamespace(trainer_id="trainer:risk"),),
        )
        overworld = SimpleNamespace(map_id=MapRSE.ROUTE102.value, objects=(), triggers=())

        with (
            patch.object(campaign_module, "_trainer_script_for_id", return_value="TrainerScript"),
            patch.object(campaign_module, "trainer_checkpoint_for_source", return_value=resolved),
        ):
            facts = campaign_module._merge_route_trainer_facts(overworld, visible, route_analysis)

        self.assertEqual(tuple(fact.checkpoint_id for fact in facts), ("risk",))

    def test_defeated_trainer_is_removed_from_frontier_aliases(self):
        checkpoint = SimpleNamespace(
            checkpoint_id="rom:emerald:318",
            trainer_affordance_id="rom_trainer:318",
            rom_trainer_id=318,
            map_id=MapRSE.ROUTE102.value,
            display_name="Calvin",
        )
        with (
            patch.object(campaign_module.context, "campaign_trainer_facts", (checkpoint,)),
            patch.object(campaign_module.context, "campaign_defeated_trainer_ids", set()),
            patch.object(campaign_module.context, "campaign_trainer_checkpoint", checkpoint),
        ):
            campaign_module._remember_defeated_trainer(checkpoint)
            self.assertEqual(campaign_module.context.campaign_trainer_facts, ())
            self.assertIsNone(campaign_module.context.campaign_trainer_checkpoint)
            self.assertIn(("rom_trainer_id", 318), campaign_module.context.campaign_defeated_trainer_ids)

    def test_unresolved_trainer_battle_does_not_defeat_previous_checkpoint(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._active_battle_wild = False
        mode.controller = SimpleNamespace(
            last_selection=None,
            request_readiness_recheck=Mock(),
        )
        previous_active = campaign_module.context.campaign_active_trainer_checkpoint
        previous_checkpoint = campaign_module.context.campaign_trainer_checkpoint
        stale_checkpoint = SimpleNamespace(checkpoint_id="stale")
        campaign_module.context.campaign_active_trainer_checkpoint = None
        campaign_module.context.campaign_trainer_checkpoint = stale_checkpoint
        try:
            with (
                patch.object(campaign_module, "_remember_defeated_trainer") as remember,
                patch.object(campaign_module, "notify_battle_ended"),
                patch.object(campaign_module, "diagnostic_print"),
                patch.object(campaign_module, "runtime_campaign_state", return_value=SimpleNamespace(campaign_facts=())),
                patch.object(campaign_module, "get_party", return_value=()),
            ):
                mode.on_battle_ended(BattleOutcome.Won)

            remember.assert_not_called()
        finally:
            campaign_module.context.campaign_active_trainer_checkpoint = previous_active
            campaign_module.context.campaign_trainer_checkpoint = previous_checkpoint

    def test_battle_start_latches_route_and_live_rom_trainer_aliases(self):
        planned = SimpleNamespace(
            checkpoint_id="trainer:(0, 19):28",
            trainer_affordance_id="trainer:(0, 19):28",
            map_id=MapRSE.ROUTE104.value,
            display_name="Billy",
            rom_trainer_id=None,
        )
        live = SimpleNamespace(
            checkpoint_id="rom:BPEE0:319",
            trainer_affordance_id="rom_trainer:319",
            map_id=MapRSE.ROUTE104.value,
            display_name="Billy",
            rom_trainer_id=319,
        )
        with (
            patch.object(campaign_module.context, "campaign_trainer_checkpoint", planned),
            patch.object(campaign_module.context, "campaign_trainer_facts", (planned,)),
            patch.object(campaign_module, "get_battle_state", return_value=SimpleNamespace(
                is_trainer_battle=True,
                opponent=SimpleNamespace(active_battler=object()),
            )),
            patch.object(campaign_module, "trainer_checkpoint_for_battle", return_value=live),
        ):
            latched = _trainer_checkpoints_for_battle_start()

        self.assertEqual(
            tuple(checkpoint.checkpoint_id for checkpoint in latched),
            ("trainer:(0, 19):28", "rom:BPEE0:319"),
        )

    def test_won_battle_prunes_every_latched_trainer_alias(self):
        route_checkpoint = SimpleNamespace(
            checkpoint_id="trainer:(0, 19):28",
            trainer_affordance_id="trainer:(0, 19):28",
            map_id=MapRSE.ROUTE104.value,
            display_name="Billy",
        )
        rom_checkpoint = SimpleNamespace(
            checkpoint_id="rom:BPEE0:319",
            trainer_affordance_id="rom_trainer:319",
            rom_trainer_id=319,
            map_id=MapRSE.ROUTE104.value,
            display_name="Billy",
        )
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._active_battle_wild = False
        mode._active_trainer_checkpoints = (route_checkpoint, rom_checkpoint)
        mode.controller = SimpleNamespace(
            last_selection=None,
            request_readiness_recheck=Mock(),
        )
        with (
            patch.object(campaign_module, "_remember_defeated_trainer") as remember,
            patch.object(campaign_module, "notify_battle_ended"),
            patch.object(campaign_module, "diagnostic_print"),
            patch.object(campaign_module, "runtime_campaign_state", return_value=SimpleNamespace(campaign_facts=())),
            patch.object(campaign_module, "get_party", return_value=()),
        ):
            mode.on_battle_ended(BattleOutcome.Won)

        self.assertEqual(remember.call_args_list, [call(route_checkpoint), call(rom_checkpoint)])
        self.assertEqual(mode._active_trainer_checkpoints, ())

    def test_route_analysis_key_changes_when_trainer_hazard_geometry_changes(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        world = SimpleNamespace(
            facing=None,
            running_shoes=False,
            player_elevation=None,
            surfing=False,
            revision_key=lambda: ((), ()),
            triggers=(SimpleNamespace(
                trigger_id="static:trainer",
                affordance_id="trainer:1",
                hazard_kind="trainer",
                condition_active=True,
                hazard_locations=frozenset({((1, 2), (4, 4))}),
                engagement_locations=frozenset(),
            ),),
        )
        changed = SimpleNamespace(
            facing=None,
            running_shoes=False,
            player_elevation=None,
            surfing=False,
            revision_key=lambda: ((), ()),
            triggers=(SimpleNamespace(
                trigger_id="static:trainer",
                affordance_id="trainer:1",
                hazard_kind="trainer",
                condition_active=True,
                hazard_locations=frozenset({((1, 2), (4, 5))}),
                engagement_locations=frozenset(),
            ),),
        )

        target = object()
        first = mode._route_analysis_key(world, ((1, 2), (1, 1)), target, ())
        second = mode._route_analysis_key(changed, ((1, 2), (1, 1)), target, ())

        self.assertNotEqual(first, second)

    def test_route_analysis_key_ignores_transient_engagement_positions(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        world = SimpleNamespace(
            facing=None,
            running_shoes=False,
            player_elevation=None,
            surfing=False,
            revision_key=lambda: ((), ()),
            triggers=(SimpleNamespace(
                trigger_id="static:trainer",
                affordance_id="trainer:1",
                hazard_kind="trainer",
                condition_active=True,
                trainer_defeated=False,
                hazard_locations=frozenset({((1, 2), (4, 4))}),
                engagement_locations=frozenset({((1, 2), (4, 5))}),
            ),),
        )
        changed = SimpleNamespace(
            facing=None,
            running_shoes=False,
            player_elevation=None,
            surfing=False,
            revision_key=lambda: ((), ()),
            triggers=(SimpleNamespace(
                trigger_id="static:trainer",
                affordance_id="trainer:1",
                hazard_kind="trainer",
                condition_active=True,
                trainer_defeated=False,
                hazard_locations=frozenset({((1, 2), (4, 4))}),
                engagement_locations=frozenset({((1, 2), (4, 6))}),
            ),),
        )

        target = object()
        self.assertEqual(
            mode._route_analysis_key(world, ((1, 2), (1, 1)), target, ()),
            mode._route_analysis_key(changed, ((1, 2), (1, 1)), target, ()),
        )

    def test_advisory_route_timeout_uses_bounded_retry_cooldown(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)

        class Planner:
            def __init__(self):
                self.calls = 0

            def poll(self, key, operation, **_kwargs):
                self.calls += 1
                return "error", None, TimeoutError("advisory timeout")

            def discard(self, key):
                pass

        mode._background_route_planner = Planner()
        world = SimpleNamespace(
            facing=None,
            running_shoes=False,
            player_elevation=None,
            surfing=False,
            revision_key=lambda: ((), ()),
            triggers=(),
        )
        goal = object()
        with patch.object(campaign_module, "navigation_background_search_active", return_value=False):
            first = mode._analyze_route_async(world, object(), ((1, 2), (1, 1)), goal, ())
            second = mode._analyze_route_async(world, object(), ((1, 2), (1, 1)), goal, ())

        self.assertEqual(first, (None, False, True))
        self.assertEqual(second, (None, True, False))
        self.assertEqual(mode._background_route_planner.calls, 1)

    def test_route_analysis_coalesces_moving_start_while_same_envelope_is_pending(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)

        class Planner:
            def __init__(self):
                self.calls = []

            def poll(self, key, operation, **_kwargs):
                self.calls.append(key)
                return "pending", None, None

            def discard(self, key):
                pass

        planner = Planner()
        mode._background_route_planner = planner
        world = SimpleNamespace(
            facing=None,
            running_shoes=False,
            player_elevation=None,
            surfing=False,
            revision_key=lambda: ((), ()),
            triggers=(),
        )
        goal = object()

        first = mode._analyze_route_async(world, object(), ((1, 2), (1, 1)), goal, ())
        second = mode._analyze_route_async(world, object(), ((1, 2), (1, 2)), goal, ())

        self.assertEqual(first[:2], (None, True))
        self.assertEqual(second[:2], (None, True))
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(planner.calls[0], planner.calls[1])

    def test_campaign_selection_cache_replans_when_cached_objective_is_complete(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        completed = SimpleNamespace(value=True)
        objective = CampaignObjective(
            "obtain_encounter:0:18",
            "Obtain the Route 103 encounter",
            (),
            CampaignPredicate("encounter_done", "encounter is captured", lambda _: Fact.known(completed.value)),
        )
        cached = ObjectiveSelection(objective, ObjectiveStatus.READY, "cached encounter")
        replanned = ObjectiveSelection(None, ObjectiveStatus.COMPLETE, "frontier advanced")
        mode._campaign_selection_key = ("unchanged",)
        mode._campaign_selection = cached
        state = SimpleNamespace()

        with (
            patch("modules.modes.campaign.campaign_planning_signature", return_value=("unchanged",)),
            patch("modules.modes.campaign.plan_campaign", return_value=replanned) as planner,
        ):
            result = mode._select_campaign(state)

        self.assertIs(result, replanned)
        planner.assert_called_once_with(state)
        self.assertEqual(mode._campaign_selection_key, ("unchanged",))

    def test_campaign_selection_cache_replans_after_restock_deferral(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        objective = CampaignObjective(
            "restock_recovery_items",
            "Restock recovery items",
            (),
            CampaignPredicate("not_done", "restock remains pending", lambda _: Fact.known(False)),
        )
        cached = ObjectiveSelection(objective, ObjectiveStatus.READY, "cached restock")
        replanned = ObjectiveSelection(None, ObjectiveStatus.READY, "progression resumed")
        mode._campaign_selection_key = ("unchanged",)
        mode._campaign_selection = cached
        state = SimpleNamespace()
        deferral = ("restock_recovery_items", MapRSE.OLDALE_TOWN.value)

        campaign_module.context.campaign_restock_deferral = deferral
        try:
            with (
                patch("modules.modes.campaign.campaign_planning_signature", return_value=("unchanged",)),
                patch("modules.modes.campaign.plan_campaign", return_value=replanned) as planner,
            ):
                result = mode._select_campaign(state)
        finally:
            campaign_module.context.campaign_restock_deferral = None

        self.assertIs(result, replanned)
        planner.assert_called_once_with(state)
        self.assertEqual(
            mode._campaign_selection_key,
            (("unchanged",), ("campaign_restock_deferral", deferral)),
        )

    def test_route_trainer_handoff_uses_static_hazard_without_runtime_object(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        hazard = (MapRSE.ROUTE102.value, (8, 6))
        trigger = TriggerObservation(
            trigger_id="static_object:1:Route102_EventScript_Youngster",
            locations=frozenset({(MapRSE.ROUTE102.value, (8, 5))}),
            affordance_id=trainer_id,
            hazard_locations=frozenset({hazard}),
            hazard_kind="trainer",
        )
        route = NavigationPlan(
            actions=(
                NavigationAction(
                    NavigationActionType.MOVE,
                    direction=object(),
                    source=(MapRSE.ROUTE102.value, (8, 5)),
                    destination=hazard,
                ),
            ),
            destination=hazard,
            forced_trainer_exposure=False,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(trigger,),
        )
        route_analysis = SimpleNamespace(normal_route=route)
        goal = NavigationGoal(
            SemanticTarget.map(MapRSE.PETALBURG_CITY.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        checkpoint = object()
        with (
            patch("modules.modes.campaign._trainer_script_for_id", return_value="Route102_EventScript_Youngster"),
            patch("modules.modes.campaign.trainer_checkpoint_for_source", return_value=checkpoint),
        ):
            result = _known_trainer_checkpoint_on_route(overworld, goal, route_analysis)

        self.assertEqual(result, trainer_id)

    def test_mandatory_outbound_trainer_fallback_survives_missing_route_analysis(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(SimpleNamespace(affordance_id=trainer_id, hazard_kind="trainer"),),
        )
        goal = NavigationGoal(
            SemanticTarget.map(MapRSE.PETALBURG_CITY.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        self.assertEqual(
            _mandatory_trainer_on_outbound_map(overworld, goal, None),
            trainer_id,
        )

    def test_mandatory_outbound_trainer_fallback_survives_missing_dynamic_geometry(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        # The runtime object table and static trigger geometry can both be
        # unavailable during a map transition. The registered mandatory
        # checkpoint must still gate the outbound route.
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(),
        )
        goal = NavigationGoal(
            SemanticTarget.map(MapRSE.ROUTE104.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        self.assertEqual(
            _mandatory_trainer_on_outbound_map(overworld, goal, None),
            trainer_id,
        )

    def test_mandatory_outbound_trainer_fallback_drops_static_defeated_checkpoint(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(SimpleNamespace(affordance_id=trainer_id, trainer_defeated=True),),
        )
        goal = NavigationGoal(
            SemanticTarget.map(MapRSE.PETALBURG_CITY.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        self.assertIsNone(_mandatory_trainer_on_outbound_map(overworld, goal, None))

    def test_mandatory_destination_trainer_fallback_covers_initial_route_entry(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        overworld = SimpleNamespace(
            map_id=MapRSE.OLDALE_TOWN.value,
            objects=(),
            triggers=(),
        )
        goal = NavigationGoal(
            SemanticTarget.map(MapRSE.ROUTE102.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        self.assertEqual(
            _mandatory_trainer_on_outbound_map(overworld, goal, None),
            trainer_id,
        )

    def test_mandatory_trainer_corridor_fallback_covers_intermediate_map(self):
        """A Gym interaction must promote Calvin before leaving Oldale."""

        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        overworld = SimpleNamespace(
            map_id=MapRSE.OLDALE_TOWN.value,
            objects=(),
            triggers=(),
        )
        goal = NavigationGoal(
            SemanticTarget.interaction(
                MapRSE.PETALBURG_CITY_GYM.value,
                interaction_id="PetalburgCity_Gym_EventScript_Norman",
            ),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        graph = SimpleNamespace(
            route=lambda _source, _target: SimpleNamespace(
                maps=(
                    MapRSE.OLDALE_TOWN.value,
                    MapRSE.ROUTE102.value,
                    MapRSE.PETALBURG_CITY.value,
                    MapRSE.PETALBURG_CITY_GYM.value,
                )
            )
        )

        with patch("modules.modes.campaign.get_world_map_graph", return_value=graph):
            self.assertEqual(
                _mandatory_trainer_on_outbound_map(overworld, goal, None),
                trainer_id,
            )

    def test_mandatory_outbound_trainer_fallback_extracts_reach_location_map(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(),
        )
        goal = NavigationGoal(
            ReachLocation((MapRSE.PETALBURG_CITY.value, (1, 1))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        self.assertEqual(
            _mandatory_trainer_on_outbound_map(overworld, goal, None),
            trainer_id,
        )

    def test_mandatory_outbound_trainer_fallback_extracts_reach_warp_destination(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(),
        )
        goal = NavigationGoal(
            ReachWarp(destination=(MapRSE.PETALBURG_CITY.value, (1, 1))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        self.assertEqual(
            _mandatory_trainer_on_outbound_map(overworld, goal, None),
            trainer_id,
        )

    def test_wally_interaction_readiness_goal_exposes_outbound_calvin(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(),
        )
        goal = NavigationGoal(
            SemanticTarget.interaction(
                MapRSE.PETALBURG_CITY_GYM.value,
                interaction_id="PetalburgCity_Gym_EventScript_Norman",
            ),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
            encounter_mode=EncounterMode.AVOID,
        )

        self.assertEqual(
            _mandatory_trainer_on_outbound_map(overworld, goal, None),
            trainer_id,
        )

    def test_forced_route_exposure_uses_registered_mandatory_checkpoint_when_path_is_compressed(self):
        trainer_id = f"trainer:{MapRSE.ROUTE102.value}:2"
        trigger = TriggerObservation(
            trigger_id="static_object:2:Route102_EventScript_Calvin",
            locations=frozenset(),
            affordance_id=trainer_id,
            hazard_locations=frozenset(),
            hazard_kind="trainer",
        )
        route = NavigationPlan(actions=(), destination=(MapRSE.PETALBURG_CITY.value, (1, 1)), forced_trainer_exposure=True)
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            objects=(),
            triggers=(trigger,),
        )
        route_analysis = SimpleNamespace(normal_route=route)
        goal = NavigationGoal(
            SemanticTarget.map(MapRSE.PETALBURG_CITY.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        with patch("modules.modes.campaign.trainer_checkpoint_for_source", return_value=object()):
            result = _known_trainer_checkpoint_on_route(overworld, goal, route_analysis)

        self.assertEqual(result, trainer_id)

    def test_map_readiness_analyzes_normal_route_without_recovery_detour(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=True,
            player=SimpleNamespace(
                map_group=MapRSE.ROUTE102.value[0],
                map_number=MapRSE.ROUTE102.value[1],
                coordinates=(4, 4),
            ),
        )
        resources = ResourceSnapshot(
            (PartyResource(20, 20),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            player_coordinates=(4, 4),
            controllable=True,
            transition_in_progress=False,
            objects=(),
            triggers=(),
        )
        objective = SimpleNamespace(
            objective_id="complete_petalburg_wally",
            destination=MapRSE.PETALBURG_CITY.value,
            resource_policy=SimpleNamespace(minimum_hp_ratio=0.5),
        )
        goal = NavigationGoal(
            SemanticTarget.map(MapRSE.PETALBURG_CITY.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        route_analysis = SimpleNamespace(normal_cost=10, normal_route=None, candidates=())

        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.modes.campaign.perceive_overworld", return_value=overworld),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.NavigationWorld.from_overworld", return_value=object()),
            patch("modules.modes.campaign.get_world_map_graph", return_value=object()),
            patch.object(mode, "_analyze_route_async", return_value=(route_analysis, False, False)) as analyze,
        ):
            diagnostic = mode._readiness_input(objective, None)

            # Encounter capabilities are targetless at the execution adapter,
            # but they still carry a declared destination. Their route must be
            # inspected for mandatory trainer sight lines as well.
            objective.objective_id = "obtain_encounter:0:19"
            mode._readiness_input(objective, None)

            scripted_objective = SimpleNamespace(
                objective_id="complete_petalburg_woods",
                destination=MapRSE.PETALBURG_WOODS.value,
                resource_policy=SimpleNamespace(mandatory_battle=True),
            )
            scripted = mode._readiness_input(scripted_objective, None)

        # Each map-owning capability exposes a declared destination, so each
        # route owner receives one trainer analysis.
        assert analyze.call_count == 3
        assert analyze.call_args.args[2] == (MapRSE.ROUTE102.value, (4, 4))
        assert analyze.call_args.args[4] == ()
        assert diagnostic.route_analysis is route_analysis
        assert isinstance(diagnostic.navigation_goal.target, EngageTrainer)
        assert diagnostic.navigation_goal.target.trainer_id == f"trainer:{MapRSE.ROUTE102.value}:2"
        assert diagnostic.trainer_checkpoint is not None
        assert diagnostic.trainer_checkpoint.checkpoint_id == "route102:first_poochyena"
        assert diagnostic.trainer_engagement_required is True
        assert not isinstance(scripted.navigation_goal.target, EngageTrainer)
        assert scripted.navigation_goal.target.target_map == MapRSE.PETALBURG_WOODS.value
        assert scripted.trainer_checkpoint.checkpoint_id == "route102:first_poochyena"
        assert scripted.trainer_engagement_required is True

    def test_campaign_owns_evolution_decision_and_delays_pre_evolution_move(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        move = SimpleNamespace(name="Future Strike", base_power=60)
        current_species = SimpleNamespace(
            name="PreForm",
            evolutions=(SimpleNamespace(method="level", target_species_index=2),),
            learnset=SimpleNamespace(level_up=(SimpleNamespace(level=10, move=move),)),
        )
        target_species = SimpleNamespace(name="Evolved", learnset=SimpleNamespace(level_up=()))
        pokemon = SimpleNamespace(name="Candidate", species=current_species, level=7)
        isolated_context = SimpleNamespace(
            config=SimpleNamespace(battle=SimpleNamespace(banned_moves=())),
            message=None,
        )
        with (
            patch("modules.modes.campaign.context", isolated_context),
            patch("modules.modes.campaign.get_species_by_index", return_value=target_species),
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            self.assertTrue(mode.owns_evolution_policy())
            self.assertFalse(mode.on_pokemon_evolving_after_battle(pokemon, 0))
        self.assertIn("Future Strike", isolated_context.message)

    def test_campaign_allows_evolution_when_no_unique_future_move_is_lost(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        move = SimpleNamespace(name="Shared Move", base_power=60)
        current_species = SimpleNamespace(
            name="PreForm",
            evolutions=(SimpleNamespace(method="level", target_species_index=2),),
            learnset=SimpleNamespace(level_up=(SimpleNamespace(level=10, move=move),)),
        )
        target_species = SimpleNamespace(
            name="Evolved",
            learnset=SimpleNamespace(level_up=(SimpleNamespace(level=12, move=move),)),
        )
        pokemon = SimpleNamespace(name="Candidate", species=current_species, level=7)
        isolated_context = SimpleNamespace(
            config=SimpleNamespace(battle=SimpleNamespace(banned_moves=())),
            message=None,
        )
        with (
            patch("modules.modes.campaign.context", isolated_context),
            patch("modules.modes.campaign.get_species_by_index", return_value=target_species),
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            self.assertTrue(mode.on_pokemon_evolving_after_battle(pokemon, 0))

    def test_readiness_rejects_destination_header_with_out_of_bounds_avatar_coordinate(self):
        class FakeOverworld:
            player_coordinates = (19, -1)

            @staticmethod
            def tile_at(coordinates):
                return None if coordinates == (19, -1) else object()

        self.assertFalse(_overworld_position_is_coherent(FakeOverworld()))

    def test_readiness_accepts_a_coherent_avatar_coordinate(self):
        class FakeOverworld:
            player_coordinates = (19, 59)

            @staticmethod
            def tile_at(coordinates):
                return object() if coordinates == (19, 59) else None

        self.assertTrue(_overworld_position_is_coherent(FakeOverworld()))

    def test_readiness_recovery_analysis_is_bounded_to_local_or_selected_source(self):
        current = (MapRSE.RUSTBORO_CITY, (19, 59))
        local = CampaignProgressionMode._recovery_candidate_goals(
            current,
            RouteRecovery(center_available=True, center_location=(MapRSE.RUSTBORO_CITY, (16, 38))),
            is_rse=True,
        )
        self.assertEqual(local, (ReachLocation((MapRSE.RUSTBORO_CITY, (16, 38))),))

        with (
            patch(
                "modules.modes.campaign.emerald_healing_sources_for_map",
                return_value=(),
            ),
            patch(
                "modules.modes.campaign.pokemon_center_candidates",
                return_value=(PokemonCenter.RustboroCity,),
            ),
        ):
            selected = CampaignProgressionMode._recovery_candidate_goals(
                (MapRSE.ROUTE104, (1, 1)),
                RouteRecovery(center_available=True, center_location=(MapRSE.RUSTBORO_CITY, (16, 38))),
                is_rse=True,
            )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].location, (MapRSE.RUSTBORO_CITY, (16, 38)))

    def test_readiness_does_not_analyze_recovery_during_unavailable_overworld(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=None,
            party=(),
            party_available=False,
            player_available=False,
        )
        resources = ResourceSnapshot(
            observation_status=ResourceObservationStatus.VALID,
        )
        objective = SimpleNamespace(
            objective_id="test_objective",
            resource_policy=SimpleNamespace(minimum_hp_ratio=0.5),
        )
        with (
            patch(
                "modules.modes.campaign.perceive_overworld",
                return_value=OverworldObservationResult(
                    OverworldObservationStatus.UNAVAILABLE,
                    reason="player_avatar_unavailable",
                ),
            ),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch("modules.modes.campaign.get_game_state", return_value=None),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery") as observe_recovery,
            patch("modules.modes.campaign.build_progression_readiness_diagnostic", return_value=object()),
        ):
            mode._readiness_input(objective, None)

        observe_recovery.assert_not_called()

    def test_readiness_does_not_analyze_recovery_for_transient_empty_snapshot_party(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=True,
            player=SimpleNamespace(map_group=0, map_number=18, coordinates=(10, 3)),
        )
        resources = ResourceSnapshot(
            (PartyResource(20, 23),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=(0, 18),
            player_coordinates=(10, 3),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        objective = SimpleNamespace(
            objective_id="receive_pokedex",
            destination=(0, 7),
            resource_policy=SimpleNamespace(minimum_hp_ratio=0.5),
        )
        with (
            patch(
                "modules.modes.campaign.perceive_overworld",
                return_value=overworld,
            ),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery") as observe_recovery,
        ):
            diagnostic = mode._readiness_input(objective, None)

        observe_recovery.assert_not_called()
        assert diagnostic.party_availability is Availability.KNOWN
        assert diagnostic.lowest_hp_ratio == 20 / 23

    def test_readiness_route_analysis_uses_coherent_overworld_when_snapshot_player_is_stale(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=False,
            player=SimpleNamespace(map_group=8, map_number=1, coordinates=(4, 108)),
        )
        resources = ResourceSnapshot(
            (PartyResource(5, 10),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.PETALBURG_CITY_GYM,
            player_coordinates=(4, 108),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        objective = SimpleNamespace(
            objective_id="obtain_encounter:24:11",
            destination=(24, 11),
            resource_policy=SimpleNamespace(minimum_hp_ratio=0.5),
        )
        goal = SimpleNamespace()
        route_analysis = SimpleNamespace(normal_cost=10, candidates=())
        analyzer = SimpleNamespace(analyze=Mock(return_value=route_analysis))
        recovery = RouteRecovery(
            center_available=True,
            center_location=(MapRSE.PETALBURG_CITY, (20, 16)),
            distance_to_center=10,
            safe_to_reach_center=True,
        )
        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch(
                "modules.modes.campaign.perceive_overworld",
                return_value=overworld,
            ),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch(
                "modules.modes.campaign.get_nuzlocke_snapshot",
                return_value=snapshot,
            ),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch(
                "modules.modes.campaign.observe_resource_snapshot",
                return_value=resources,
            ),
            patch(
                "modules.modes.campaign.observe_route_recovery",
                return_value=recovery,
            ),
            patch(
                "modules.modes.campaign.emerald_healing_sources_for_map",
                return_value=(),
            ),
            patch(
                "modules.modes.campaign.NavigationWorld.from_overworld",
                return_value=object(),
            ),
            patch(
                "modules.modes.campaign.get_world_map_graph",
                return_value=object(),
            ),
            patch(
                "modules.modes.campaign.RouteCostAnalyzer",
                return_value=analyzer,
            ),
        ):
            diagnostic = mode._readiness_input(objective, goal)

        analyzer.analyze.assert_called_once()
        assert diagnostic.route_analysis is route_analysis

    def test_critical_readiness_skips_detour_route_analysis(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=False,
            player=SimpleNamespace(map_group=0, map_number=18, coordinates=(10, 3)),
        )
        resources = ResourceSnapshot(
            (PartyResource(1, 10),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE102.value,
            player_coordinates=(10, 3),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        objective = SimpleNamespace(
            objective_id="complete_petalburg_wally",
            destination=MapRSE.PETALBURG_CITY.value,
            resource_policy=SimpleNamespace(minimum_hp_ratio=0.5),
        )
        recovery = RouteRecovery(
            center_available=True,
            center_location=(MapRSE.OLDALE_TOWN, (6, 6)),
            distance_to_center=4,
            safe_to_reach_center=True,
            route=object(),
        )
        analyzer = Mock()
        analyzer.analyze.return_value = SimpleNamespace(normal_route=None, normal_cost=10, candidates=())
        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.modes.campaign.perceive_overworld", return_value=overworld),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery", return_value=recovery),
            patch("modules.modes.campaign.NavigationWorld.from_overworld", return_value=object()),
            patch("modules.modes.campaign.get_world_map_graph", return_value=object()),
            patch("modules.modes.campaign.RouteCostAnalyzer", return_value=analyzer),
        ):
            diagnostic = mode._readiness_input(objective, SimpleNamespace())

        analyzer.analyze.assert_not_called()
        assert diagnostic.route_analysis is None

    def test_completed_targetful_recovery_route_runs_detour_analysis(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=GameState.OVERWORLD,
            party=(),
            party_available=False,
            player_available=False,
            player=SimpleNamespace(map_group=MapRSE.ROUTE104.value[0], map_number=MapRSE.ROUTE104.value[1], coordinates=(8, 18)),
        )
        resources = ResourceSnapshot(
            (PartyResource(7, 10),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE104.value,
            player_coordinates=(8, 18),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        objective = SimpleNamespace(
            objective_id="obtain_encounter:0:19",
            destination=MapRSE.ROUTE104.value,
            resource_policy=SimpleNamespace(minimum_hp_ratio=0.5),
        )
        recovery = RouteRecovery(
            center_available=True,
            center_location=(MapRSE.PETALBURG_CITY, (20, 16)),
            distance_to_center=30,
            safe_to_reach_center=True,
            route=object(),
        )
        analyzer = Mock()
        analyzer.analyze.return_value = SimpleNamespace(normal_route=None, normal_cost=10, candidates=())
        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.modes.campaign.perceive_overworld", return_value=overworld),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery", return_value=recovery),
            patch("modules.modes.campaign.NavigationWorld.from_overworld", return_value=object()),
            patch("modules.modes.campaign.get_world_map_graph", return_value=object()),
            patch("modules.modes.campaign.RouteCostAnalyzer", return_value=analyzer),
        ):
            diagnostic = mode._readiness_input(objective, SimpleNamespace())

        analyzer.analyze.assert_called_once()
        self.assertIs(diagnostic.route_analysis, analyzer.analyze.return_value)

    def test_targetless_milestone_projects_declared_destination_for_readiness(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=False,
            player=SimpleNamespace(
                map_group=MapRSE.PETALBURG_CITY.value[0],
                map_number=MapRSE.PETALBURG_CITY.value[1],
                coordinates=(20, 16),
            ),
        )
        resources = ResourceSnapshot(
            (PartyResource(5, 10),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.PETALBURG_CITY.value,
            player_coordinates=(20, 16),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        objective = SimpleNamespace(
            objective_id="complete_petalburg_wally",
            destination=MapRSE.PETALBURG_CITY_GYM.value,
            resource_policy=SimpleNamespace(minimum_hp_ratio=0.5),
        )
        route_analysis = SimpleNamespace(normal_cost=0, candidates=())
        analyzer = SimpleNamespace(analyze=Mock(return_value=route_analysis))
        recovery = RouteRecovery(
            center_available=True,
            center_location=(MapRSE.PETALBURG_CITY, (20, 16)),
            distance_to_center=2,
            safe_to_reach_center=True,
        )
        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.modes.campaign.perceive_overworld", return_value=overworld),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery", return_value=recovery),
            patch("modules.modes.campaign.NavigationWorld.from_overworld", return_value=object()),
            patch("modules.modes.campaign.get_world_map_graph", return_value=object()),
            patch("modules.modes.campaign.RouteCostAnalyzer", return_value=analyzer),
        ):
            diagnostic = mode._readiness_input(objective, None)

        analyzer.analyze.assert_called_once()
        self.assertEqual(diagnostic.navigation_goal.target.target_map, MapRSE.PETALBURG_CITY_GYM.value)
        self.assertIs(diagnostic.route_analysis, route_analysis)

    def test_targetless_objective_uses_universal_readiness(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=True,
            player=SimpleNamespace(map_group=0, map_number=18, coordinates=(10, 3)),
        )
        resources = ResourceSnapshot(
            (PartyResource(8, 25),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE103,
            player_coordinates=(10, 3),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        recovery = RouteRecovery(
            center_available=True,
            center_location=(MapRSE.OLDALE_TOWN, (6, 6)),
            distance_to_center=38,
            safe_to_reach_center=True,
        )
        analyzer = SimpleNamespace(analyze=Mock(return_value=SimpleNamespace(normal_cost=10, candidates=())))
        objective = SimpleNamespace(
            objective_id="receive_pokedex",
            destination=MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value,
            resource_policy=None,
        )

        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.modes.campaign.perceive_overworld", return_value=overworld),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery", return_value=recovery) as observe_recovery,
            patch("modules.modes.campaign.NavigationWorld.from_overworld", return_value=object()),
            patch("modules.modes.campaign.get_world_map_graph", return_value=object()),
            patch("modules.modes.campaign.RouteCostAnalyzer", return_value=analyzer),
        ):
            diagnostic = mode._readiness_input(objective, None)

        observe_recovery.assert_called_once_with(candidate_limit=None)
        analyzer.analyze.assert_not_called()
        self.assertEqual(diagnostic.lowest_hp_ratio, 8 / 25)
        self.assertIs(diagnostic.recovery, recovery)

    def test_universal_readiness_applies_to_restock_objective(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=True,
            player=SimpleNamespace(map_group=0, map_number=10, coordinates=(14, 6)),
        )
        resources = ResourceSnapshot(
            (PartyResource(8, 25),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.OLDALE_TOWN,
            player_coordinates=(14, 6),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        objective = SimpleNamespace(objective_id="restock_pokeballs", resource_policy=None)

        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.modes.campaign.perceive_overworld", return_value=overworld),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery") as observe_recovery,
        ):
            diagnostic = mode._readiness_input(objective, None)

        observe_recovery.assert_called_once_with(candidate_limit=None)
        self.assertEqual(diagnostic.lowest_hp_ratio, 8 / 25)

    def test_universal_readiness_survives_transient_rival_selection(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_evaluated = False
        snapshot = SimpleNamespace(
            frame=1,
            game_state=SimpleNamespace(name="OVERWORLD"),
            party=(),
            party_available=False,
            player_available=True,
            player=SimpleNamespace(map_group=0, map_number=18, coordinates=(9, 3)),
        )
        resources = ResourceSnapshot(
            (PartyResource(8, 25),),
            observation_status=ResourceObservationStatus.VALID,
        )
        overworld = SimpleNamespace(
            map_id=MapRSE.ROUTE103,
            player_coordinates=(9, 3),
            controllable=True,
            transition_in_progress=False,
            objects=(),
        )
        objective = SimpleNamespace(
            objective_id="complete_intro_rival",
            resource_policy=None,
        )

        with (
            patch(
                "modules.modes.campaign.context",
                SimpleNamespace(rom=SimpleNamespace(is_rse=True), stutter_trace=None),
            ),
            patch("modules.modes.campaign.perceive_overworld", return_value=overworld),
            patch("modules.modes.campaign.publish_shared_overworld_observation"),
            patch("modules.modes.campaign.get_nuzlocke_snapshot", return_value=snapshot),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.modes.campaign.observe_interaction",
                return_value=SimpleNamespace(
                    interaction_phase=SimpleNamespace(),
                    script_active=False,
                    dialogue_waiting=False,
                    field_message_lifecycle_active=False,
                    native_function=None,
                ),
            ),
            patch("modules.modes.campaign.observe_resource_snapshot", return_value=resources),
            patch("modules.modes.campaign.observe_route_recovery") as observe_recovery,
        ):
            diagnostic = mode._readiness_input(objective, None)

        observe_recovery.assert_called_once_with(candidate_limit=None)
        self.assertEqual(diagnostic.lowest_hp_ratio, 8 / 25)

    def test_battle_end_requests_universal_readiness_recheck(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        objective = SimpleNamespace(objective_id="complete_intro_rival", resource_policy=None)
        mode._active_battle_wild = False
        mode.controller = SimpleNamespace(
            last_selection=SimpleNamespace(objective=objective),
            request_readiness_recheck=Mock(),
        )

        with patch("modules.modes.campaign.notify_battle_ended"), patch("modules.modes.campaign.diagnostic_print"):
            mode.on_battle_ended(object())

        mode.controller.request_readiness_recheck.assert_called_once_with("battle_ended")

    def test_controller_is_constructed_during_mode_initialization(self):
        with patch("modules.modes.campaign.CampaignController") as controller_factory:
            mode = CampaignProgressionMode()
        controller_factory.assert_called_once()
        self.assertIs(mode.controller, controller_factory.return_value)

    def test_cheap_readiness_context_excludes_walking_coordinates(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        avatar = SimpleNamespace(map_group_and_number=(1, 2), local_coordinates=(3, 4))
        with (
            patch("modules.modes.campaign.get_player_avatar", return_value=avatar),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
        ):
            self.assertEqual(mode._cheap_readiness_context(), ("OVERWORLD", (1, 2), None, None, None))

    def test_encounter_stage_change_is_a_readiness_boundary(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        avatar = SimpleNamespace(map_group_and_number=(1, 2), local_coordinates=(3, 4))
        evidence = SimpleNamespace(
            current_map=(1, 2),
            next_map=(1, 3),
            world_signature=("world", 1),
            readiness=None,
            blocked=True,
            known_hazard_ids=("trainer:one",),
            unresolved_hazard_ids=(),
            known_hazard_checkpoints=(SimpleNamespace(checkpoint_id="checkpoint:one"),),
        )
        with (
            patch("modules.modes.campaign.context", SimpleNamespace(campaign_encounter_stage=evidence)),
            patch("modules.modes.campaign.get_player_avatar", return_value=avatar),
            patch("modules.modes.campaign.get_game_state", return_value=GameState.OVERWORLD),
        ):
            key = mode._cheap_readiness_context()

        self.assertEqual(key[-1], ((1, 2), (1, 3), ("world", 1), None, True, ("trainer:one",), (), ("checkpoint:one",)))

    def test_pending_recovery_search_is_not_reprepared_for_same_map_coordinate_change(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)

        class Planner:
            def __init__(self):
                self.polled = []

            def is_pending(self, key):
                return key == ("recovery", "prepared")

            def poll(self, key, operation, **_kwargs):
                self.polled.append(key)
                return "pending", None, None

        planner = Planner()
        mode._background_route_planner = planner
        mode._prepared_recovery_request = SimpleNamespace(key="prepared", critical=True)
        mode._prepared_recovery_anchor = (0, 19, (11, 39))

        with (
            patch.object(CampaignProgressionMode, "_recovery_anchor", return_value=(0, 19, (12, 39))),
            patch("modules.modes.campaign.prepare_route_recovery") as prepare,
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            result = mode._observe_route_recovery_async(critical=True)

        self.assertTrue(result.calculation_pending)
        prepare.assert_not_called()
        self.assertEqual(planner.polled, [("recovery", "prepared")])

    def test_ball_inventory_change_requests_campaign_refresh(self):
        refresh_reasons = []
        controller = SimpleNamespace(
            request_readiness_recheck=lambda reason: refresh_reasons.append(reason),
        )
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode.controller = controller
        mode._campaign_boundary_context_seen = True
        mode._last_campaign_boundary_context = ("OVERWORLD", (1, 2), True, 5)

        with patch.object(
            CampaignProgressionMode,
            "_cheap_readiness_context",
            return_value=("OVERWORLD", (1, 2), True, 4),
        ):
            mode._observe_campaign_boundary(controller)

        self.assertEqual(refresh_reasons, ["pokeball_inventory_changed"])

    def test_run_delegates_startup_observation_to_controller(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        fake_controller = SimpleNamespace(
            step=lambda: SimpleNamespace(
                status=SimpleNamespace(value="ready"),
                objective_id="set_text_speed",
                reason="resolved executable frontier",
            )
        )
        mode.controller = fake_controller
        mode._readiness_evaluated = False

        loop = mode.run()
        next(loop)
        self.assertEqual(mode.controller, fake_controller)

    def test_run_retries_transient_boot_observation_without_releasing_controller(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        statuses = iter(("unknown", "ready"))
        steps = []

        def step():
            status = next(statuses)
            steps.append(status)
            return SimpleNamespace(
                status=SimpleNamespace(value=status),
                objective_id=None if status == "unknown" else "set_text_speed",
                reason="campaign observation unavailable" if status == "unknown" else "ready",
            )

        controller = SimpleNamespace(step=step)
        mode.controller = controller
        mode._readiness_evaluated = False
        loop = mode.run()
        next(loop)
        next(loop)
        self.assertEqual(steps, ["unknown", "ready"])
        self.assertIs(mode.controller, controller)

    def test_run_requests_readiness_refresh_on_map_boundary(self):
        refresh_reasons = []
        controller = SimpleNamespace(
            request_readiness_recheck=lambda reason: refresh_reasons.append(reason),
            step=lambda: SimpleNamespace(
                status=SimpleNamespace(value="ready"),
                objective_id="complete_petalburg_wally",
                reason="ready",
            ),
        )
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode.controller = controller
        mode._campaign_boundary_context_seen = False
        mode._last_campaign_boundary_context = None

        with patch.object(
            CampaignProgressionMode,
            "_cheap_readiness_context",
            side_effect=[("OVERWORLD", (1, 2), True), ("OVERWORLD", (1, 3), True)],
        ):
            loop = mode.run()
            next(loop)
            next(loop)

        self.assertEqual(refresh_reasons, ["map_changed"])

    def test_trainer_spotted_requests_readiness_refresh(self):
        reasons = []
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode.controller = SimpleNamespace(request_readiness_recheck=lambda reason: reasons.append(reason))

        mode.on_spotted_by_trainer()

        self.assertEqual(reasons, ["trainer_spotted"])

    def test_wild_battle_end_retains_navigation_route(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._active_battle_wild = True
        mode._prepared_recovery_request = object()
        mode._prepared_recovery_anchor = (1, 2)
        mode.controller = SimpleNamespace(request_readiness_recheck=Mock())

        with (
            patch("modules.modes.campaign.notify_battle_ended") as notify,
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            mode.on_battle_ended(object())

        notify.assert_called_once_with(invalidate_navigation=False)
        mode.controller.request_readiness_recheck.assert_called_once_with("battle_ended")
        self.assertIsNone(mode._active_battle_wild)
        self.assertIsNotNone(mode._prepared_recovery_request)

    def test_trainer_battle_end_invalidates_navigation_route(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._active_battle_wild = False
        mode._prepared_recovery_request = object()
        mode._prepared_recovery_anchor = (1, 2)
        mode.controller = SimpleNamespace(request_readiness_recheck=Mock())

        with (
            patch("modules.modes.campaign.notify_battle_ended") as notify,
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            mode.on_battle_ended(object())

        notify.assert_called_once_with(invalidate_navigation=True)
        self.assertIsNone(mode._prepared_recovery_request)

    def test_battle_callback_returns_explicit_strategy_without_legacy_fight_action(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_scheduler = SimpleNamespace(invalidate=lambda *_: None)

        selected = object()
        with patch(
            "modules.modes.campaign.NuzlockeLevelBalancingBattleStrategy", return_value=selected
        ) as factory:
            self.assertIs(mode.on_battle_started(None), selected)
        factory.assert_called_once_with()

    def test_startup_capability_is_not_blocked_by_overworld_readiness(self):
        objective = CampaignObjective(
            "set_text_speed",
            "set text speed",
            (),
            CampaignPredicate("text_speed_fast", "text speed is fast", lambda _: Fact.known(False)),
        )
        advanced = []

        def capability():
            advanced.append("first startup executor step")
            yield

        controller = CampaignController(
            lambda: object(),
            selector=lambda _: ObjectiveSelection(objective, ObjectiveStatus.READY, "startup"),
            adapter=lambda _: CampaignExecutionResult(
                objective,
                CampaignExecutionStatus.READY,
                "startup capability",
                capability=capability,
            ),
            readiness_provider=lambda *_: object(),
        )
        with patch(
            "modules.nuzlocke.campaign_controller.build_campaign_plan",
            side_effect=AssertionError("startup capability must not require overworld readiness"),
        ):
            state = controller.step()

        self.assertEqual(state.status.value, "ready")
        self.assertEqual(advanced, ["first startup executor step"])

    def test_lost_run_does_not_mount_campaign_tactical_loop(self):
        state = SimpleNamespace(run_status=Fact.known(RunStatus.LOST))

        controller = CampaignController(
            lambda: state,
            selector=plan_campaign,
            adapter=lambda selection: CampaignExecutionResult(
                selection.objective,
                CampaignExecutionStatus.FAILED,
                selection.reason,
            ),
        )

        result = controller.step()

        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.reason, "run is lost")
        self.assertIsNone(controller.current_objective_id)


if __name__ == "__main__":
    unittest.main()
