import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.modes.campaign import CampaignProgressionMode, _overworld_position_is_coherent
from modules.map_data import MapRSE, PokemonCenter
from modules.goals import ReachLocation
from modules.nuzlocke.resource_policy import (
    PartyResource,
    ResourceObservationStatus,
    RouteRecovery,
    ResourceSnapshot,
)
from modules.overworld import OverworldObservationResult, OverworldObservationStatus
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
from modules.nuzlocke.readiness_diagnostics import Availability


class CampaignOrchestrationTests(unittest.TestCase):
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
            objective_id="reach_petalburg",
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
            patch("modules.modes.campaign.RouteCostAnalyzer", return_value=analyzer),
        ):
            diagnostic = mode._readiness_input(objective, SimpleNamespace())

        analyzer.analyze.assert_not_called()
        assert diagnostic.route_analysis is None

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

        analyzer.analyze.assert_not_called()
        self.assertEqual(diagnostic.navigation_goal.target.target_map, MapRSE.PETALBURG_CITY_GYM.value)
        self.assertIsNone(diagnostic.route_analysis)

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
            self.assertEqual(mode._cheap_readiness_context(), ("OVERWORLD", (1, 2), None, None))

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
                objective_id="reach_oldale",
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
        mode.controller = SimpleNamespace(request_readiness_recheck=Mock())

        with (
            patch("modules.modes.campaign.notify_battle_ended") as notify,
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            mode.on_battle_ended(object())

        notify.assert_called_once_with(invalidate_navigation=False)
        mode.controller.request_readiness_recheck.assert_called_once_with("battle_ended")
        self.assertIsNone(mode._active_battle_wild)

    def test_trainer_battle_end_invalidates_navigation_route(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._active_battle_wild = False
        mode.controller = SimpleNamespace(request_readiness_recheck=Mock())

        with (
            patch("modules.modes.campaign.notify_battle_ended") as notify,
            patch("modules.modes.campaign.diagnostic_print"),
        ):
            mode.on_battle_ended(object())

        notify.assert_called_once_with(invalidate_navigation=True)

    def test_battle_callback_remains_execution_policy(self):
        mode = CampaignProgressionMode.__new__(CampaignProgressionMode)
        mode._readiness_scheduler = SimpleNamespace(invalidate=lambda *_: None)
        from modules.modes import BattleAction

        self.assertIs(mode.on_battle_started(None), BattleAction.Fight)

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
