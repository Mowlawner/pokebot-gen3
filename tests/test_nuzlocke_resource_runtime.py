import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.goals import ActivateTrigger, GoalConstraints, NavigationGoal, ReachLocation, ReachWarp, TrainerMode
from modules.context import context
from modules.nuzlocke.resource_policy import (
    HealingResource,
    PartyResource,
    ResourceDecision,
    ResourceSnapshot,
    RouteRecovery,
)
from modules.nuzlocke.resource_runtime import (
    CampaignCapability,
    FAINTED_POKEMON_BOX,
    HealingSource,
    HealingSourceType,
    discover_healing_source,
    execute_campaign_encounter,
    execute_campaign_recovery,
    execute_campaign_preparation,
    execute_heal_party,
    _preparation_navigation_goal,
    recover_at_nearest_center,
)
from modules.nuzlocke.resource_runtime import _execute_healing_source_interaction
from modules.nuzlocke.resource_runtime import _safe_capture_box_index
from modules.nuzlocke.resource_runtime import _navigate_to_known_center_pc
from modules.nuzlocke.resource_runtime import _finalize_fainted_party_cleanup
from modules.nuzlocke.resource_runtime import _wait_for_center_interior
from modules.nuzlocke.resource_runtime import execute_planned_recovery, RecoveryTrainerSafetyError
from modules.nuzlocke.resource_runtime import _execute_preparation_recovery
from modules.agent_control import AgentAction, AgentActionType, ActionDecision
from modules.agent_control import AgentObservation
from modules.interaction_state import InteractionObservation
from modules.memory import GameState
from modules.map_data import MapRSE, PokemonCenter
from modules.map_path import Direction
from modules.modes.util.pokecenter_loop import PokecenterLoopController
from modules.overworld import MovementState, OverworldObservation, TileObservation, TriggerObservation
from modules.nuzlocke.resource_runtime import _encounter_roll_direction
from modules.nuzlocke.resource_runtime import _encounter_stage_mandatory_hazards
from modules.nuzlocke.resource_runtime import _encounter_stage_route_cut_hazards
from modules.nuzlocke.resource_runtime import _encounter_stage_trainer_mode
from modules.nuzlocke.resource_runtime import _best_encounter_stage_probe
from modules.nuzlocke.resource_runtime import _encounter_stage_trainer_evidence
from modules.nuzlocke.resource_runtime import _cached_encounter_stage_plan
from modules.nuzlocke.resource_runtime import _mandatory_checkpoint_trigger_ids
from modules.nuzlocke.resource_runtime import _mandatory_trainer_hazard_state
from modules.navigation import NavigationAction, NavigationActionType, NavigationError, NavigationPlan
from modules.world_navigation import WorldEdge, WorldMapGraph


class CampaignCapabilityTests(unittest.TestCase):
    @staticmethod
    def _stage_candidate(*trainer_ids, status="found"):
        return SimpleNamespace(status=status, trainer_ids=tuple(trainer_ids))

    @staticmethod
    def _trainer_object(trainer_id, **changes):
        values = dict(
            trainer_id=trainer_id,
            local_id=1,
            location=((0, 19), (5, 5)),
            trainer_type="Normal",
            trainer_range=3,
            facing="Up",
            movement_type="FACE_UP_AND_RIGHT",
            movement_range=(0, 0),
            movement_origin=((0, 19), (5, 5)),
            trainer_defeated=False,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def _exact_overworld(self, *trainer_ids, **changes):
        return SimpleNamespace(
            objects=tuple(self._trainer_object(trainer_id, **changes) for trainer_id in trainer_ids),
            triggers=(),
        )

    def test_route_cut_promotes_trainers_common_to_all_permissive_exits(self):
        from modules.nuzlocke import resource_runtime as runtime

        overworld = self._exact_overworld("trainer:cindy", "trainer:billy")
        candidates = (
            self._stage_candidate("trainer:cindy", "trainer:billy"),
            self._stage_candidate("trainer:cindy", "trainer:billy"),
        )

        with patch.object(
            runtime,
            "trainer_checkpoint_for_source",
            side_effect=lambda trainer_id, **_: SimpleNamespace(trainer_affordance_id=trainer_id),
        ):
            known, unresolved = _encounter_stage_route_cut_hazards(
                overworld,
                safe_candidates=(),
                permissive_candidates=candidates,
            )

        self.assertEqual(known, ("trainer:billy", "trainer:cindy"))
        self.assertEqual(unresolved, ())

    def _route_cut(self, overworld, *trainer_ids):
        from modules.nuzlocke import resource_runtime as runtime

        candidates = (self._stage_candidate(*trainer_ids), self._stage_candidate(*trainer_ids))
        with patch.object(
            runtime,
            "trainer_checkpoint_for_source",
            side_effect=lambda trainer_id, **_: SimpleNamespace(trainer_affordance_id=trainer_id),
        ):
            return _encounter_stage_route_cut_hazards(
                overworld, safe_candidates=(), permissive_candidates=candidates
            )

    def test_route_cut_does_not_promote_a_trainer_whose_facing_is_a_guess(self):
        # No FACE_* template and no readable facing: the hazard area is all four
        # directions, so a failed search proves nothing about this trainer.
        overworld = self._exact_overworld("trainer:billy", movement_type="NONE", facing=None)

        known, unresolved = self._route_cut(overworld, "trainer:billy")

        self.assertEqual(known, ())
        self.assertEqual(unresolved, ("trainer:billy",))

    def test_route_cut_does_not_promote_a_fixed_axis_walker_with_an_over_covered_hazard(self):
        overworld = self._exact_overworld("trainer:walker", movement_type="WALK_UP_AND_DOWN")

        known, unresolved = self._route_cut(overworld, "trainer:walker")

        self.assertEqual(known, ())
        self.assertEqual(unresolved, ("trainer:walker",))

    def test_route_cut_promotes_a_walk_sequence_trainer_with_exact_geometry(self):
        overworld = self._exact_overworld(
            "trainer:billy", movement_type="WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT", movement_range=(6, 5)
        )

        known, unresolved = self._route_cut(overworld, "trainer:billy")

        self.assertEqual(known, ("trainer:billy",))
        self.assertEqual(unresolved, ())

    def test_route_cut_does_not_promote_a_trainer_with_no_observed_geometry(self):
        known, unresolved = self._route_cut(SimpleNamespace(objects=(), triggers=()), "trainer:ghost")

        self.assertEqual(known, ())
        self.assertEqual(unresolved, ("trainer:ghost",))

    def test_route_cut_keeps_only_the_exact_trainer_when_the_cut_is_mixed(self):
        overworld = SimpleNamespace(
            objects=(
                self._trainer_object("trainer:cindy"),
                self._trainer_object("trainer:blind", movement_type="NONE", facing=None),
            ),
            triggers=(),
        )

        known, unresolved = self._route_cut(overworld, "trainer:cindy", "trainer:blind")

        self.assertEqual(known, ("trainer:cindy",))
        self.assertEqual(unresolved, ("trainer:blind",))

    def test_route_cut_promotes_a_trainer_whose_live_facing_is_known(self):
        overworld = self._exact_overworld("trainer:sam", movement_type="NONE", facing="Down")

        known, unresolved = self._route_cut(overworld, "trainer:sam")

        self.assertEqual(known, ("trainer:sam",))
        self.assertEqual(unresolved, ())

    def test_encounter_stage_prefers_fewer_grass_opportunities_before_route_cost(self):
        first = SimpleNamespace(
            index=0,
            status="found",
            plan=SimpleNamespace(
                metrics=SimpleNamespace(
                    encounter_opportunities=4,
                    total_route_cost=8,
                    movement_actions=8,
                )
            ),
        )
        safe_grass_corridor = SimpleNamespace(
            index=1,
            status="found",
            plan=SimpleNamespace(
                metrics=SimpleNamespace(
                    encounter_opportunities=1,
                    total_route_cost=14,
                    movement_actions=14,
                )
            ),
        )
        batch = SimpleNamespace(probes=(first, safe_grass_corridor))

        selected = _best_encounter_stage_probe(batch)

        self.assertIs(selected, safe_grass_corridor)

    def test_route_cut_does_not_promote_a_trainer_present_on_one_exit(self):
        from modules.nuzlocke import resource_runtime as runtime

        overworld = self._exact_overworld("trainer:cindy", "trainer:billy")
        candidates = (
            self._stage_candidate("trainer:cindy", "trainer:billy"),
            self._stage_candidate("trainer:billy"),
        )

        with patch.object(
            runtime,
            "trainer_checkpoint_for_source",
            side_effect=lambda trainer_id, **_: SimpleNamespace(trainer_affordance_id=trainer_id),
        ):
            known, unresolved = _encounter_stage_route_cut_hazards(
                overworld,
                safe_candidates=(),
                permissive_candidates=candidates,
            )

        self.assertEqual(known, ("trainer:billy",))
        self.assertEqual(unresolved, ())

    def test_safe_candidate_suppresses_route_cut_promotion(self):
        from modules.nuzlocke import resource_runtime as runtime

        overworld = SimpleNamespace(objects=(), triggers=())
        with patch.object(
            runtime,
            "trainer_checkpoint_for_source",
            return_value=SimpleNamespace(trainer_affordance_id="trainer:cindy"),
        ):
            known, unresolved = _encounter_stage_route_cut_hazards(
                overworld,
                safe_candidates=(self._stage_candidate(),),
                permissive_candidates=(self._stage_candidate("trainer:cindy"),),
            )

        self.assertEqual(known, ())
        self.assertEqual(unresolved, ())

    def test_route_cut_known_checkpoint_enters_mandatory_hazards(self):
        from modules.nuzlocke import resource_runtime as runtime

        overworld = SimpleNamespace(
            map_id=(0, 19),
            triggers=(),
            objects=(self._trainer_object("trainer:cindy"),),
        )
        candidates = (
            self._stage_candidate("trainer:cindy"),
            self._stage_candidate("trainer:cindy"),
        )
        with patch.object(
            runtime,
            "trainer_checkpoint_for_source",
            return_value=SimpleNamespace(trainer_affordance_id="trainer:cindy"),
        ):
            hazards = _encounter_stage_mandatory_hazards(
                overworld,
                safe_candidates=(),
                permissive_candidates=candidates,
            )

        self.assertEqual(hazards, ("trainer:cindy",))

    def test_encounter_stage_preserves_resolved_hazard_checkpoint_for_readiness(self):
        from modules.nuzlocke import resource_runtime as runtime

        checkpoint = SimpleNamespace(
            checkpoint_id="route104:cindy",
            trainer_affordance_id="trainer:cindy",
        )
        candidates = (self._stage_candidate("trainer:cindy"),)
        with (
            patch.object(
                runtime,
                "_encounter_stage_candidate_evidence",
                side_effect=[(), candidates],
            ),
            patch.object(runtime, "trainer_checkpoint_for_source", return_value=checkpoint),
        ):
            evidence = runtime._publish_encounter_stage_evidence(
                SimpleNamespace(
                    map_id=(0, 19),
                    triggers=(),
                    objects=(self._trainer_object("trainer:cindy"),),
                ),
                (24, 11),
                SimpleNamespace(signature=("safe",), shared_search=False),
                SimpleNamespace(signature=("permissive",), shared_search=False),
                selected_candidate=0,
                readiness=None,
                reason="stage_unavailable:trainer_exposure_readiness_unknown",
                blocked=True,
            )

        self.assertEqual(evidence.known_hazard_ids, ("trainer:cindy",))
        self.assertEqual(evidence.known_hazard_checkpoints, (checkpoint,))

    def test_encounter_stage_records_trainer_geometry_for_live_diagnostics(self):
        trainer = SimpleNamespace(
            trainer_id="trainer:(0, 19):28",
            local_id=28,
            location=((0, 19), (18, 67)),
            trainer_type="Normal",
            trainer_range=3,
            facing="South",
            movement_type="WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT",
            movement_range=(6, 5),
            movement_origin=((0, 19), (18, 67)),
            trainer_defeated=False,
        )
        trigger = SimpleNamespace(
            hazard_kind="trainer",
            affordance_id="trainer:(0, 19):28",
            trigger_id="static_object:28:Route104_EventScript_Billy",
            locations=frozenset({((0, 19), (18, 67))}),
            hazard_locations=frozenset({((0, 19), (18, 66))}),
            engagement_locations=frozenset({((0, 19), (18, 66))}),
            activation_locations=frozenset({((0, 19), (18, 68))}),
            trainer_defeated=False,
        )

        evidence = _encounter_stage_trainer_evidence(
            SimpleNamespace(objects=(trainer,), triggers=(trigger,))
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].trainer_id, "trainer:(0, 19):28")
        self.assertEqual(evidence[0].movement_type, "WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT")
        self.assertEqual(evidence[0].movement_range, (6, 5))
        self.assertEqual(evidence[0].hazard_locations, (((0, 19), (18, 66)),))
        self.assertEqual(evidence[0].activation_locations, (((0, 19), (18, 68)),))

    def test_encounter_stage_lookahead_selects_downstream_safe_exit(self):
        from modules.nuzlocke import resource_runtime as runtime

        transitions = (
            SimpleNamespace(
                kind="map_connection",
                entry=((0, 0), (39, 62)),
                destination=((0, 19), (39, 62)),
                required_facing=Direction.West,
            ),
            SimpleNamespace(
                kind="map_connection",
                entry=((0, 0), (39, 63)),
                destination=((0, 19), (39, 63)),
                required_facing=Direction.West,
            ),
        )
        overworld = OverworldObservation(
            map_id=(0, 0),
            player_coordinates=(15, 9),
            facing=Direction.West,
            controllable=True,
            tiles=(),
            warps=(),
            objects=(),
            triggers=(),
            transitions=transitions,
        )
        selected = SimpleNamespace(selected_transitions=(transitions[1],))

        with (
            patch.object(runtime.NavigationWorld, "from_overworld", return_value=object()),
            patch.object(runtime, "plan_with_world_navigation", return_value=(object(), selected)),
        ):
            self.assertEqual(
                runtime._encounter_stage_lookahead_candidate(
                    overworld,
                    transitions,
                    (0, 19),
                    (24, 11),
                    object(),
                ),
                1,
            )

    def test_encounter_stage_uses_lookahead_exit_before_local_cost(self):
        from modules.nuzlocke import resource_runtime as runtime

        transitions = (
            SimpleNamespace(kind="warp", entry=((0, 1), (1, 0)), destination=((0, 2), (1, 1))),
            SimpleNamespace(kind="warp", entry=((0, 1), (2, 0)), destination=((0, 2), (2, 1))),
        )
        overworld = SimpleNamespace(
            map_id=(0, 1),
            player_coordinates=(0, 0),
            transitions=transitions,
            triggers=(),
            objects=(),
        )
        found = SimpleNamespace(
            status="found",
            index=0,
            transition=transitions[0],
            approach=None,
            plan=SimpleNamespace(
                metrics=SimpleNamespace(
                    encounter_opportunities=0,
                    total_route_cost=1,
                    movement_actions=1,
                )
            ),
        )
        lookahead = SimpleNamespace(
            status="found",
            index=1,
            transition=transitions[1],
            approach=None,
            plan=SimpleNamespace(
                metrics=SimpleNamespace(
                    encounter_opportunities=0,
                    total_route_cost=9,
                    movement_actions=9,
                )
            ),
        )
        batch = SimpleNamespace(probes=(found, lookahead), shared_search=False, signature=("lookahead",))

        with (
            patch.object(runtime, "_encounter_stage_world_signature", return_value=("lookahead",)),
            patch.object(runtime, "_encounter_stage_probe_batch", return_value=batch),
            patch.object(runtime, "_encounter_stage_lookahead_candidate", return_value=1),
            patch.object(runtime, "_publish_encounter_stage_evidence"),
            patch.object(runtime, "_encounter_stage_battle_ready", return_value=None),
        ):
            transition, mode, reason = runtime._encounter_stage_trainer_mode(
                overworld,
                transitions,
                (0, 2),
                final_map=(0, 3),
                graph=object(),
            )

        self.assertIs(transition, transitions[1])
        self.assertIs(mode, TrainerMode.AVOID)
        self.assertIn("lookahead_candidate=1", reason)

    def test_encounter_stage_reuses_the_selected_probe_plan_for_execution(self):
        from modules.nuzlocke import resource_runtime as runtime

        transition = SimpleNamespace(
            kind="map_connection",
            entry=((0, 0), (39, 63)),
            destination=((0, 19), (39, 63)),
            required_facing=Direction.West,
        )
        plan = NavigationPlan(
            actions=(
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.West,
                    ((0, 0), (2, 13)),
                    ((0, 0), (1, 13)),
                ),
            ),
            destination=transition.entry,
        )
        batch = runtime._EncounterStageBatch(
            signature=("route104",),
            trainer_mode=TrainerMode.AVOID,
            probes=(
                runtime._EncounterStageProbe(
                    1,
                    transition,
                    ((0, 0), (1, 13)),
                    "found",
                    plan=plan,
                ),
            ),
            shared_search=True,
        )
        overworld = SimpleNamespace(
            map_id=(0, 0),
            player_coordinates=(25, 13),
            transitions=(transition,),
        )

        with patch.object(runtime, "_encounter_stage_world_signature", return_value=("route104",)), patch.object(
            runtime,
            "_encounter_stage_probe_cache_get",
            return_value=batch,
        ):
            reused = _cached_encounter_stage_plan(
                overworld,
                (transition,),
                (0, 19),
                transition,
                TrainerMode.AVOID,
            )

        self.assertIs(reused, plan)

    def test_encounter_stage_geometry_uses_static_trainer_template_before_spawn(self):
        from modules.nuzlocke import resource_runtime as runtime

        static_trainer = SimpleNamespace(
            trainer_id="trainer:(0, 19):28",
            local_id=28,
            location=((0, 19), (18, 67)),
            trainer_type="Normal",
            trainer_range=3,
            facing="South",
            movement_type="WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT",
            movement_range=(6, 5),
            movement_origin=((0, 19), (18, 67)),
            trainer_defeated=None,
        )

        with (
            patch.object(runtime, "get_map_metadata", return_value=object()),
            patch.object(runtime, "static_trainer_observations", return_value=(static_trainer,)),
        ):
            evidence = _encounter_stage_trainer_evidence(
                SimpleNamespace(map_id=(0, 19), objects=(), triggers=())
            )

        self.assertEqual(evidence[0].movement_type, "WALK_SEQUENCE_DOWN_RIGHT_UP_LEFT")
        self.assertEqual(evidence[0].movement_origin, ((0, 19), (18, 67)))
        self.assertIsNone(evidence[0].defeated)

    def test_encounter_stage_promotes_only_a_known_mandatory_unavoidable_trainer(self):
        from modules.nuzlocke import resource_runtime as runtime

        transitions = (
            SimpleNamespace(
                kind="warp",
                entry=((0, 1), (1, 0)),
                destination=((0, 2), (1, 1)),
            ),
            SimpleNamespace(
                kind="warp",
                entry=((0, 1), (2, 0)),
                destination=((0, 2), (2, 1)),
            ),
        )
        overworld = SimpleNamespace(
            map_id=(0, 1),
            player_coordinates=(0, 0),
            triggers=(),
        )
        found = SimpleNamespace(
            metrics=SimpleNamespace(total_route_cost=4),
            forced_trainer_exposure=False,
        )

        def probe(_overworld, _transition, _next_map, trainer_mode):
            if trainer_mode is TrainerMode.AVOID:
                raise NavigationError("trainer hazard blocks candidate")
            return found

        runtime._encounter_stage_probe_cache.clear()
        with (
            patch.object(runtime, "_encounter_stage_world_signature", return_value=("fixture",)),
            patch.object(runtime, "_probe_encounter_stage_route", side_effect=probe) as probe_route,
            patch.object(runtime, "_encounter_stage_mandatory_hazards", return_value=("trainer:mandatory",)),
            patch.object(context, "campaign_trainer_readiness", SimpleNamespace(ready=True)),
        ):
            transition, mode, reason = _encounter_stage_trainer_mode(overworld, transitions, (0, 2))
            cached_transition, cached_mode, _ = _encounter_stage_trainer_mode(
                overworld, transitions, (0, 2)
            )

        self.assertIs(transition, transitions[0])
        self.assertIs(cached_transition, transitions[0])
        self.assertIs(mode, TrainerMode.ENGAGE)
        self.assertIs(cached_mode, TrainerMode.ENGAGE)
        self.assertIn("forced_trainer_exposure", reason)
        # One AVOID and one IGNORE batch were computed; the second request was
        # served from the complete topology-signature cache.
        self.assertEqual(probe_route.call_count, 4)

    def test_encounter_stage_returns_to_frontier_when_geometry_is_unavailable(self):
        from modules.nuzlocke import resource_runtime as runtime

        transition = SimpleNamespace(
            kind="warp",
            entry=((0, 1), (1, 0)),
            destination=((0, 2), (1, 1)),
        )
        overworld = SimpleNamespace(map_id=(0, 1), player_coordinates=(0, 0), triggers=())
        runtime._encounter_stage_probe_cache.clear()
        with (
            patch.object(runtime, "_encounter_stage_world_signature", return_value=("unreachable",)),
            patch.object(
                runtime,
                "_probe_encounter_stage_route",
                side_effect=NavigationError("no route"),
            ),
        ):
            selected, mode, reason = _encounter_stage_trainer_mode(overworld, (transition,), (0, 2))

        self.assertIsNone(selected)
        self.assertIs(mode, TrainerMode.AVOID)
        self.assertIn("avoid_and_ignore_unreachable", reason)

    def test_bag_recovery_does_not_select_a_potion_for_a_full_party(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "none"),),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
        )

        self.assertIsNone(runtime._select_bag_recovery_item(snapshot))

    def test_bag_recovery_still_selects_hp_item_for_poison_bridge(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "poisoned"),),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
        )

        self.assertEqual(runtime._select_bag_recovery_item(snapshot).name, "Potion")

    def test_bag_recovery_prefers_the_smallest_effective_hp_medicine(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(75, 100, "none"),),
            bag_healing_items=(
                HealingResource("Potion", 1, 20),
                HealingResource("Super Potion", 1, 50),
            ),
        )

        self.assertEqual(runtime._select_bag_recovery_item(snapshot).name, "Potion")

    def test_potion_is_not_treated_as_a_status_cure(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "paralysed"),),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
        )

        self.assertIsNone(runtime._select_bag_recovery_item(snapshot))

    def test_targeted_status_cure_beats_full_restore_for_tiny_hp_deficit(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(99, 100, "poisoned"),),
            bag_healing_items=(
                HealingResource("Full Restore", 1, 255),
                HealingResource("Antidote", 1, 0),
            ),
        )

        self.assertEqual(runtime._select_bag_recovery_item(snapshot).name, "Antidote")

    def test_full_restore_is_status_fallback_when_no_targeted_cure_exists(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(99, 100, "poisoned"),),
            bag_healing_items=(HealingResource("Full Restore", 1, 255),),
        )

        self.assertEqual(runtime._select_bag_recovery_item(snapshot).name, "Full Restore")

    def test_one_point_safe_hp_deficit_does_not_consume_full_restore(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(99, 100, "none"),),
            bag_healing_items=(HealingResource("Full Restore", 1, 255),),
        )

        self.assertIsNone(runtime._select_bag_recovery_item(snapshot))

    def test_unavailable_recovery_route_does_not_open_bag_for_a_noop(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "none"),),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
        )
        stop = SimpleNamespace(route=None, destination=None)
        with patch.object(runtime, "observe_resource_snapshot", return_value=snapshot), patch.object(
            runtime, "get_game_state", return_value=GameState.OVERWORLD
        ), patch.object(runtime, "_use_field_recovery_item", return_value=iter(())) as use_item:
            with self.assertRaisesRegex(RuntimeError, "no usable campaign recovery"):
                list(runtime.execute_campaign_recovery(stop))

        use_item.assert_not_called()

    def test_fainted_party_recovery_never_uses_a_bag_item_as_a_center_substitute(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(0, 20, "none", fainted=True),),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
        )
        stop = SimpleNamespace(
            route=None,
            destination=None,
            reason="FAINTED_PARTY_MEMBER",
        )
        with patch.object(runtime, "observe_resource_snapshot", return_value=snapshot), patch.object(
            runtime, "get_game_state", return_value=GameState.OVERWORLD
        ), patch.object(runtime, "_use_field_recovery_item", return_value=iter(())) as use_item:
            with self.assertRaisesRegex(RuntimeError, "mandatory Center recovery"):
                list(runtime.execute_campaign_recovery(stop))

        use_item.assert_not_called()

    def test_center_recovery_stop_never_uses_a_potion_when_its_route_is_missing(self):
        from modules.nuzlocke import resource_runtime as runtime

        snapshot = ResourceSnapshot(
            party=(PartyResource(5, 20, "none"),),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
        )
        stop = SimpleNamespace(
            route=None,
            destination=(MapRSE.ROUTE102, (39, 6)),
            reason="OPPORTUNISTIC_RECOVERY",
            execution_kind="center_route",
        )
        with patch.object(runtime, "observe_resource_snapshot", return_value=snapshot), patch.object(
            runtime, "get_game_state", return_value=GameState.OVERWORLD
        ), patch.object(runtime, "_use_field_recovery_item", return_value=iter(())) as use_item:
            with self.assertRaisesRegex(RuntimeError, "planned Center recovery has no executable route"):
                list(runtime.execute_campaign_recovery(stop))

        use_item.assert_not_called()

    def test_restored_party_keeps_selected_center_visit_for_stored_medicine(self):
        from modules.nuzlocke import resource_runtime as runtime

        nurse = SimpleNamespace(
            source_id="nurse",
            interaction_trigger_id="object:nurse",
            source_type=HealingSourceType.POKEMON_CENTER_NURSE,
        )
        interior = SimpleNamespace(
            overworld=SimpleNamespace(map_id=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value),
            interaction_type=SimpleNamespace(name="OVERWORLD"),
            interaction=SimpleNamespace(controllable=True),
        )
        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "none"),),
            pc_healing_items=(HealingResource("Potion", 1, 20, location="pc"),),
        )
        navigation_goals = []

        class FakeLoop:
            def __init__(self, *args, **kwargs):
                navigation_goals.append(kwargs["goal"])

            def run(self):
                return iter(())

        with (
            patch.object(runtime, "party_is_restored", return_value=True),
            patch.object(runtime, "observe_resource_snapshot", return_value=snapshot),
            patch.object(runtime, "AgentControlLoop", FakeLoop),
            patch.object(runtime, "_wait_for_center_interior", return_value=iter(())),
            patch.object(runtime, "observe_agent", return_value=interior),
            patch.object(runtime, "discover_healing_source", return_value=nurse),
            patch.object(runtime, "_execute_healing_source_interaction", return_value=iter(())),
            patch.object(runtime, "_withdraw_pc_healing_item_at_current_location", return_value=iter(("withdraw",))),
            patch.object(runtime, "_deposit_fainted_party_at_current_location", return_value=iter(())),
        ):
            self.assertEqual(list(runtime.execute_planned_recovery(PokemonCenter.OldaleTown.value)), ["withdraw"])

        self.assertEqual(len(navigation_goals), 2)

    def test_safe_capture_box_skips_graveyard_box(self):
        storage = SimpleNamespace(
            active_box_index=FAINTED_POKEMON_BOX,
            boxes=tuple(SimpleNamespace(first_empty_slot_index=0) for _ in range(14)),
        )

        self.assertEqual(_safe_capture_box_index(storage), 0)

    def test_safe_capture_box_prefers_current_box_when_it_is_safe(self):
        storage = SimpleNamespace(
            active_box_index=3,
            boxes=tuple(SimpleNamespace(first_empty_slot_index=0) for _ in range(14)),
        )

        self.assertEqual(_safe_capture_box_index(storage), 3)

    def test_safe_capture_box_returns_none_when_only_graveyard_has_room(self):
        storage = SimpleNamespace(
            active_box_index=FAINTED_POKEMON_BOX,
            boxes=tuple(
                SimpleNamespace(first_empty_slot_index=0 if index == FAINTED_POKEMON_BOX else None)
                for index in range(14)
            ),
        )

        self.assertIsNone(_safe_capture_box_index(storage))

    def test_center_pc_geometry_fallback_rechecks_corrected_facing(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value
        first = SimpleNamespace(
            overworld=SimpleNamespace(map_id=map_id, player_coordinates=(10, 2), facing=Direction.South)
        )
        corrected = SimpleNamespace(
            overworld=SimpleNamespace(map_id=map_id, player_coordinates=(10, 2), facing=Direction.North)
        )
        emulator = SimpleNamespace(press_direction=Mock())
        approach = ((map_id, (10, 2)), Direction.North)
        avatar = SimpleNamespace(
            map_group_and_number=map_id,
            local_coordinates=(10, 2),
            facing_direction="Up",
            map_location_in_front=SimpleNamespace(tile_type="PC"),
        )

        class FakeLoop:
            def run(self):
                return iter(())

        with (
            patch.object(runtime, "_known_center_pc_approach", return_value=approach),
            patch.object(runtime, "AgentControlLoop", return_value=FakeLoop()),
            patch.object(runtime, "observe_agent", side_effect=(first, corrected)),
            patch.object(runtime, "get_player_avatar", return_value=avatar),
            patch.object(context, "emulator", emulator),
        ):
            self.assertEqual(list(_navigate_to_known_center_pc(first, purpose="test")), [None])

        emulator.press_direction.assert_called_once_with(Direction.North.button_name, run=False, fresh=True)
        self.assertEqual(context.campaign_recovery_pc_target, approach[0])

    def test_center_pc_geometry_fallback_rejects_failed_facing_verification(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value
        first = SimpleNamespace(
            overworld=SimpleNamespace(map_id=map_id, player_coordinates=(10, 2), facing=Direction.South)
        )
        still_wrong = SimpleNamespace(
            overworld=SimpleNamespace(map_id=map_id, player_coordinates=(10, 2), facing=Direction.South)
        )
        approach = ((map_id, (10, 2)), Direction.North)

        class FakeLoop:
            def run(self):
                return iter(())

        with (
            patch.object(runtime, "_known_center_pc_approach", return_value=approach),
            patch.object(runtime, "AgentControlLoop", return_value=FakeLoop()),
            patch.object(runtime, "observe_agent", side_effect=(first, still_wrong)),
            patch.object(context, "emulator", SimpleNamespace(press_direction=Mock())),
        ):
            with self.assertRaisesRegex(RuntimeError, "required approach and facing"):
                list(_navigate_to_known_center_pc(first, purpose="test"))

    def test_center_pc_geometry_fallback_derives_the_pc_approach_from_map_tiles(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value
        observation = SimpleNamespace(
            overworld=SimpleNamespace(map_id=map_id, player_coordinates=(7, 8))
        )
        source = SimpleNamespace(
            source_id="pokemon_center:oldale",
            center_pc_approach=((map_id, (7, 8)), Direction.North),
        )
        tiles = [
            SimpleNamespace(local_position=(10, 1), tile_type="PC", collision=1),
            SimpleNamespace(local_position=(10, 2), tile_type="Normal", collision=0),
        ]
        with (
            patch.object(runtime, "emerald_healing_sources", return_value=(source,)),
            patch.object(runtime, "get_map_data", return_value=object()),
            patch.object(runtime, "get_map_all_tiles", return_value=tiles),
        ):
            self.assertEqual(
                runtime._known_center_pc_approach(observation),
                ((map_id, (10, 2)), Direction.North),
            )

    def test_center_pc_geometry_fallback_rejects_a_non_pc_front_tile(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value
        observation = SimpleNamespace(
            overworld=SimpleNamespace(map_id=map_id, player_coordinates=(10, 2), facing=Direction.North)
        )
        approach = ((map_id, (10, 2)), Direction.North)
        avatar = SimpleNamespace(
            map_group_and_number=map_id,
            local_coordinates=(10, 2),
            facing_direction="Up",
            map_location_in_front=SimpleNamespace(tile_type="Normal"),
        )

        class FakeLoop:
            def run(self):
                return iter(())

        with (
            patch.object(runtime, "_known_center_pc_approach", return_value=approach),
            patch.object(runtime, "AgentControlLoop", return_value=FakeLoop()),
            patch.object(runtime, "observe_agent", return_value=observation),
            patch.object(runtime, "get_player_avatar", return_value=avatar),
            patch.object(context, "emulator", SimpleNamespace(press_direction=Mock())),
        ):
            with self.assertRaisesRegex(RuntimeError, "not executable"):
                list(_navigate_to_known_center_pc(observation, purpose="test"))

    def test_fainted_cleanup_clears_authoritative_pending_identity_after_storage(self):
        from modules.nuzlocke import resource_runtime as runtime

        identity = object()
        pokemon = SimpleNamespace(name="Wurmple")
        mode = SimpleNamespace(_pending_fainted_identities={identity})
        storage = SimpleNamespace(active_box_index=0)
        with (
            patch.object(runtime, "_remembered_fainted_party_members", return_value=()),
            patch.object(runtime, "get_pokemon_storage", return_value=storage),
            patch.object(runtime.PokemonIdentity, "from_pokemon", return_value=identity),
            patch.object(context, "bot_mode_instance", mode),
        ):
            _finalize_fainted_party_cleanup((pokemon,))

        self.assertEqual(mode._pending_fainted_identities, set())

    @staticmethod
    def _encounter_world(*, tiles, facing=Direction.North, current=(1, 1)):
        map_id = MapRSE.ROUTE102.value
        observations = tuple(
            TileObservation(
                (map_id, coordinates),
                blocked=False,
                walkable_neighbors=frozenset(Direction),
                has_encounters=True,
            )
            for coordinates in tiles
        )
        return OverworldObservation(
            map_id=map_id,
            player_coordinates=current,
            facing=facing,
            controllable=True,
            tiles=observations,
            warps=(),
            objects=(),
            triggers=(),
            movement_state=MovementState.STANDING,
        )

    def test_encounter_roll_turns_at_grass_boundary_before_moving(self):
        overworld = self._encounter_world(tiles=((1, 1), (2, 1)))

        direction, reason = _encounter_roll_direction(overworld)

        self.assertEqual(direction, Direction.South)
        self.assertEqual(reason, "turn_in_place_on_blocked_edge")

    def test_encounter_roll_moves_when_grass_tile_has_no_blocked_edge(self):
        overworld = self._encounter_world(
            tiles=((1, 1), (1, 0), (2, 1), (1, 2), (0, 1)),
        )

        direction, reason = _encounter_roll_direction(overworld)

        self.assertEqual(direction, Direction.North)
        self.assertEqual(reason, "move_within_encounter_terrain")

    def test_encounter_roll_escapes_an_isolated_grass_tile_on_a_walkable_path(self):
        map_id = MapRSE.ROUTE102.value
        coordinates = ((1, 1), (1, 0), (2, 1), (1, 2), (0, 1))
        observations = tuple(
            TileObservation(
                (map_id, location),
                blocked=False,
                walkable_neighbors=frozenset(Direction),
                has_encounters=location == (1, 1),
            )
            for location in coordinates
        )
        overworld = OverworldObservation(
            map_id=map_id,
            player_coordinates=(1, 1),
            facing=Direction.North,
            controllable=True,
            tiles=observations,
            warps=(),
            objects=(),
            triggers=(),
            movement_state=MovementState.STANDING,
        )

        direction, reason = _encounter_roll_direction(overworld)

        self.assertEqual(direction, Direction.East)
        self.assertEqual(reason, "step_off_encounter_terrain")

    def test_encounter_roll_turns_into_blocked_edge_when_tile_isolated(self):
        overworld = self._encounter_world(tiles=((1, 1),), facing=Direction.East)

        direction, reason = _encounter_roll_direction(overworld)

        self.assertEqual(direction, Direction.North)
        self.assertEqual(reason, "turn_in_place_on_blocked_edge")

    def test_campaign_encounter_does_not_finish_at_grass(self):
        overworld = self._encounter_world(tiles=((1, 1), (2, 1)))
        battle = AgentObservation(InteractionObservation(GameState.BATTLE))
        observations = iter(
            (
                AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True), overworld),
                AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True), overworld),
                AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True), overworld),
                battle,
            )
        )
        emulator = SimpleNamespace(
            reset_held_buttons=lambda: None,
            press_direction=Mock(),
        )

        class FakeNavigationLoop:
            def run(self):
                return iter(())

        with patch(
            "modules.nuzlocke.resource_runtime._preparation_training_location",
            return_value=(MapRSE.ROUTE102.value, ((1, 1),)),
        ), patch(
            "modules.nuzlocke.resource_runtime._preparation_navigation_goal",
            return_value=(ReachLocation((MapRSE.ROUTE102.value, (1, 1))), None),
        ) as select_goal, patch(
            "modules.nuzlocke.resource_runtime.get_world_map_graph",
            return_value=SimpleNamespace(),
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop",
            return_value=FakeNavigationLoop(),
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            side_effect=observations,
        ), patch(
            "modules.nuzlocke.resource_runtime._preparation_encounter_resolved",
            return_value=False,
        ), patch(
            "modules.nuzlocke.resource_runtime.context",
            SimpleNamespace(emulator=emulator, nuzlocke_runtime=None),
        ):
            execution = execute_campaign_encounter(MapRSE.ROUTE102.value)
            next(execution)
            self.assertIs(select_goal.call_args.kwargs["trainer_mode"], TrainerMode.AVOID)
            emulator.press_direction.assert_called_once_with("Down", run=False, fresh=True)
            with self.assertRaises(StopIteration):
                next(execution)

    def _drive_encounter(self, observe, *, goal_result, limit=200, **patches):
        """Run execute_campaign_encounter against fakes; return (steps, finished, logs)."""

        from modules.nuzlocke import resource_runtime as runtime

        logs = []

        def capture(message, *args, **kwargs):
            logs.append(message() if callable(message) else str(message))

        class FakeLoop:
            def __init__(self, *args, **kwargs):
                self.goal = kwargs.get("goal")

            def run(self):
                return iter(())

        with (
            patch.object(runtime, "_preparation_training_location",
                         return_value=(MapRSE.ROUTE102.value, ((1, 1),))),
            patch.object(runtime, "get_world_map_graph", return_value=SimpleNamespace()),
            patch.object(runtime, "observe_agent", side_effect=observe),
            patch.object(runtime, "_preparation_navigation_goal", return_value=goal_result),
            patch.object(runtime, "AgentControlLoop", side_effect=FakeLoop),
            patch.object(runtime, "diagnostic_print", side_effect=capture),
            patch.object(runtime, "print_always", side_effect=capture),
            patch.object(runtime, "_encounter_stage_trainer_mode") as stage_mode,
            patch.object(runtime, "_rom_backed_encounter_route") as route,
            patch.object(runtime, "static_map_transitions") as static,
            patch.object(runtime, "_cached_encounter_stage_plan", return_value=None),
            patch.object(runtime, "_acquire_campaign_encounter", **patches.get("acquire", {"return_value": iter(())})) as acquire,
            patch.object(runtime, "_ENCOUNTER_TRAINER_GATE_WAIT_LIMIT", 3),
            patch.object(runtime, "_ENCOUNTER_POST_TRANSITION_UNSETTLED_LIMIT", 3),
        ):
            warp = SimpleNamespace(destination=(MapRSE.ROUTE102.value, (1, 1)), kind="warp")
            static.return_value = (warp,)
            route.return_value = SimpleNamespace(maps=(MapRSE.ROUTE104.value, MapRSE.ROUTE102.value))
            stage_mode.return_value = (warp, TrainerMode.AVOID, "safe")
            execution = execute_campaign_encounter(MapRSE.ROUTE102.value)
            steps = 0
            finished = False
            try:
                while steps < limit:
                    next(execution)
                    steps += 1
            except StopIteration:
                finished = True
            return steps, finished, logs, acquire

    def test_encounter_resumes_with_a_destination_action_after_the_stage_warp(self):
        arrived = self._encounter_world(tiles=((1, 1), (2, 1)))
        origin = replace(arrived, map_id=MapRSE.ROUTE104.value)
        controllable = lambda world: AgentObservation(
            InteractionObservation(GameState.OVERWORLD, controllable=True), world
        )
        observations = iter((controllable(origin),))

        def observe(*_args, **_kwargs):
            return next(observations, controllable(arrived))

        goal = NavigationGoal(
            ReachLocation((MapRSE.ROUTE102.value, (2, 1))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        steps, finished, logs, acquire = self._drive_encounter(observe, goal_result=(goal, None))

        self.assertTrue(finished)
        acquire.assert_called_once_with(MapRSE.ROUTE102.value)

    def test_encounter_trainer_gate_wait_is_visible_and_bounded(self):
        world = self._encounter_world(tiles=((1, 1),))
        observation = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True), world)

        steps, finished, logs, acquire = self._drive_encounter(
            lambda *_a, **_k: observation,
            goal_result=(None, "trainer:(0, 17):2: mandatory trainer is absent from the hazard observation"),
        )

        self.assertTrue(finished, "capability must hand back to the frontier instead of waiting forever")
        acquire.assert_not_called()
        gate_logs = [line for line in logs if "CAMPAIGN_ENCOUNTER_TRAINER_GATE" in line]
        self.assertTrue(any("action='reobserve'" in line for line in gate_logs))
        self.assertTrue(any("action='return_to_frontier'" in line for line in gate_logs))

    def test_encounter_unsettled_destination_observation_is_visible_and_bounded(self):
        arrived = self._encounter_world(tiles=((1, 1),))
        elsewhere = replace(arrived, map_id=MapRSE.ROUTE104.value)
        good = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True), arrived)
        bad = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True), elsewhere)
        calls = iter((good,))

        def observe(*_args, **_kwargs):
            return next(calls, bad)

        with self.assertRaisesRegex(Exception, "did not settle"):
            self._drive_encounter(observe, goal_result=(None, "unused"))

    def test_encounter_navigation_waits_for_mandatory_trainer_geometry(self):
        overworld = self._encounter_world(tiles=((1, 1),))

        selected, error = _preparation_navigation_goal(
            overworld,
            MapRSE.ROUTE102.value,
            ((1, 1),),
            trainer_mode=TrainerMode.AVOID,
        )

        self.assertIsNone(selected)
        self.assertIn("mandatory trainer is absent", error)

    def test_woods_static_researcher_geometry_releases_encounter_gate(self):
        map_id = MapRSE.PETALBURG_WOODS.value
        overworld = SimpleNamespace(
            map_id=map_id,
            player_coordinates=(5, 5),
            objects=(),
            triggers=(
                TriggerObservation(
                    trigger_id="devon_goods_researcher",
                    locations=frozenset(),
                    activation_locations=frozenset({(map_id, (12, 7))}),
                    affordance_id="devon_goods_researcher",
                    hazard_locations=frozenset({(map_id, (12, 7))}),
                    hazard_kind="trainer",
                    trainer_defeated=None,
                ),
            ),
        )

        self.assertIsNone(_mandatory_trainer_hazard_state(overworld, map_id))

    def test_woods_coordinate_scene_geometry_releases_encounter_gate(self):
        map_id = MapRSE.PETALBURG_WOODS.value
        overworld = SimpleNamespace(
            map_id=map_id,
            player_coordinates=(17, 38),
            objects=(),
            triggers=(
                TriggerObservation(
                    trigger_id="coord:0:152:0",
                    locations=frozenset({(map_id, (26, 23))}),
                    activation_locations=frozenset({(map_id, (26, 23))}),
                    script_symbol="PetalburgWoods_EventScript_DevonResearcherLeft",
                    affordance_id="PetalburgWoods_EventScript_DevonResearcherLeft",
                    kind="trigger",
                ),
            ),
        )

        # Hermetic: without this patch the helper also reconstructs the ROM's
        # right-hand scene event whenever real map metadata happens to be loaded,
        # making the result depend on test order.
        from modules.nuzlocke import resource_runtime as runtime

        with patch.object(runtime, "get_map_metadata", return_value=SimpleNamespace(coord_events=())):
            self.assertEqual(
                _mandatory_checkpoint_trigger_ids(overworld, map_id),
                frozenset({"coord:0:152:0"}),
            )
            self.assertIsNone(_mandatory_trainer_hazard_state(overworld, map_id))

    def test_woods_both_researcher_scene_events_are_avoided_when_only_one_is_observed(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.PETALBURG_WOODS.value
        events = tuple(
            SimpleNamespace(
                type="script",
                local_coordinates=coordinates,
                elevation=3,
                trigger_var_number=152,
                trigger_value=0,
                script_symbol=symbol,
            )
            for coordinates, symbol in (
                ((26, 23), "PetalburgWoods_EventScript_DevonResearcherLeft"),
                ((27, 23), "PetalburgWoods_EventScript_DevonResearcherRight"),
            )
        )
        overworld = SimpleNamespace(
            map_id=map_id,
            player_coordinates=(17, 38),
            objects=(),
            triggers=(
                TriggerObservation(
                    trigger_id="coord:0:152:0",
                    locations=frozenset({(map_id, (26, 23))}),
                    activation_locations=frozenset({(map_id, (26, 23))}),
                    script_symbol="PetalburgWoods_EventScript_DevonResearcherLeft",
                    affordance_id="PetalburgWoods_EventScript_DevonResearcherLeft",
                    kind="trigger",
                ),
            ),
        )
        with patch.object(runtime, "get_map_metadata", return_value=SimpleNamespace(coord_events=events)):
            self.assertEqual(
                _mandatory_checkpoint_trigger_ids(overworld, map_id),
                frozenset({"coord:0:152:0", "coord:1:152:0"}),
            )

    def test_woods_coordinate_scene_is_reconstructed_from_static_map_metadata(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.PETALBURG_WOODS.value
        event = SimpleNamespace(
            type="script",
            local_coordinates=(26, 23),
            elevation=3,
            trigger_var_number=152,
            trigger_value=0,
            script_symbol="PetalburgWoods_EventScript_DevonResearcherLeft",
        )
        overworld = SimpleNamespace(
            map_id=map_id,
            player_coordinates=(17, 38),
            objects=(),
            triggers=(),
        )
        with patch.object(runtime, "get_map_metadata", return_value=SimpleNamespace(coord_events=(event,))):
            self.assertEqual(
                _mandatory_checkpoint_trigger_ids(overworld, map_id),
                frozenset({"coord:0:152:0"}),
            )
            self.assertIsNone(_mandatory_trainer_hazard_state(overworld, map_id))

    def test_woods_encounter_route_avoids_coordinate_scene_triggers(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.PETALBURG_WOODS.value
        trigger = TriggerObservation(
            trigger_id="coord:0:152:0",
            locations=frozenset({(map_id, (26, 23))}),
            activation_locations=frozenset({(map_id, (26, 23))}),
            script_symbol="PetalburgWoods_EventScript_DevonResearcherLeft",
            affordance_id="PetalburgWoods_EventScript_DevonResearcherLeft",
            kind="trigger",
        )
        overworld = SimpleNamespace(
            map_id=map_id,
            player_coordinates=(17, 38),
            objects=(),
            triggers=(trigger,),
        )
        planned = SimpleNamespace(forced_trainer_exposure=False)
        with (
            patch.object(runtime, "get_map_metadata", return_value=SimpleNamespace(coord_events=())),
            patch.object(runtime.NavigationWorld, "from_overworld", return_value=object()),
            patch.object(runtime, "_reachable_local_positions", return_value=frozenset({(18, 38)})),
            patch.object(runtime, "plan_with_world_navigation", return_value=(planned, None)) as plan,
        ):
            selected, error = _preparation_navigation_goal(
                overworld,
                map_id,
                ((18, 38),),
                trainer_mode=TrainerMode.AVOID,
            )

        self.assertIsNone(error)
        self.assertIsNotNone(selected)
        self.assertEqual(
            selected.constraints.avoid_trigger_ids,
            frozenset({"coord:0:152:0"}),
        )
        self.assertEqual(plan.call_args.args[2].constraints.avoid_trigger_ids, frozenset({"coord:0:152:0"}))

    def test_encounter_navigation_rejects_forced_trainer_exposure(self):
        from modules.navigation import NavigationPlan

        map_id = MapRSE.ROUTE102.value
        trainer_id = f"trainer:{map_id}:2"
        overworld = self._encounter_world(tiles=((1, 1),))
        overworld = replace(
            overworld,
            triggers=(
                    TriggerObservation(
                        trigger_id="static_object:2:Route102_EventScript_Calvin",
                        locations=frozenset({(map_id, (2, 1))}),
                        affordance_id=trainer_id,
                        hazard_locations=frozenset({(map_id, (2, 1))}),
                        hazard_kind="trainer",
                        engagement_locations=frozenset({(map_id, (2, 1))}),
                    ),
                ),
        )
        forced = NavigationPlan(actions=(), destination=(map_id, (1, 1)), forced_trainer_exposure=True)
        with patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(forced, None),
        ):
            selected, error = _preparation_navigation_goal(
                overworld,
                map_id,
                ((1, 1),),
                trainer_mode=TrainerMode.AVOID,
            )

        self.assertIsNone(selected)
        self.assertIn("safe encounter route is unavailable", error)

    def test_campaign_encounter_advances_dialogue_before_overworld_is_available(self):
        dialogue = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True)
        )
        action = ActionDecision(AgentAction(AgentActionType.ADVANCE_DIALOGUE))
        executor = Mock()

        with (
            patch(
                "modules.nuzlocke.resource_runtime._preparation_training_location",
                return_value=(MapRSE.ROUTE102.value, ((1, 1),)),
            ),
            patch("modules.nuzlocke.resource_runtime.get_world_map_graph", return_value=SimpleNamespace()),
            patch("modules.nuzlocke.resource_runtime.observe_agent", return_value=dialogue),
            patch("modules.nuzlocke.resource_runtime.select_action", return_value=action),
            patch("modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor),
        ):
            execution = execute_campaign_encounter(MapRSE.ROUTE102.value)
            next(execution)

        executor.execute.assert_called_once_with(action.action, dialogue)

    def test_encounter_route_skips_coarse_graph_edge_without_rom_transition(self):
        source = (0, 0)
        stale = (0, 1)
        alternate = (0, 2)
        target = (0, 3)
        graph = WorldMapGraph(
            (
                WorldEdge(source, stale, "warp", ((1, 0),), ((0, 0),)),
                WorldEdge(source, alternate, "warp", ((2, 0),), ((0, 0),)),
                WorldEdge(stale, target, "warp", ((1, 0),), ((0, 0),)),
                WorldEdge(alternate, target, "warp", ((1, 0),), ((0, 0),)),
            )
        )

        def transitions_for(map_id):
            destinations = {
                source: (alternate,),
                stale: (target,),
                alternate: (target,),
            }
            return tuple(
                SimpleNamespace(kind="warp", destination=(destination, (0, 0)))
                for destination in destinations.get(map_id, ())
            )

        from modules.nuzlocke import resource_runtime as runtime

        with patch.object(runtime, "static_map_transitions", side_effect=transitions_for):
            route = runtime._rom_backed_encounter_route(graph, source, target)

        self.assertEqual(route.maps, (source, alternate, target))

    def test_field_status_item_targets_the_party_member_with_that_status(self):
        from modules.nuzlocke import resource_runtime as runtime

        item = HealingResource("Antidote", 1, 0)
        before = ResourceSnapshot(
            party=(
                PartyResource(12, 20, "poisoned"),
                PartyResource(8, 20, "none"),
            ),
            bag_healing_items=(item,),
        )
        after = ResourceSnapshot(
            party=(
                PartyResource(12, 20, "none"),
                PartyResource(8, 20, "none"),
            ),
            bag_healing_items=(HealingResource("Antidote", 0, 0),),
        )
        party = (
            SimpleNamespace(
                index=0,
                current_hp=12,
                total_hp=20,
                status_condition=SimpleNamespace(value="poisoned"),
                is_egg=False,
            ),
            SimpleNamespace(
                index=1,
                current_hp=8,
                total_hp=20,
                status_condition=SimpleNamespace(value="none"),
                is_egg=False,
            ),
        )
        emulator = SimpleNamespace(press_button=Mock(), press_button_fresh=Mock())
        with (
            patch.object(runtime, "get_party", return_value=party),
            patch.object(runtime, "observe_resource_snapshot", side_effect=(before, after)),
            patch.object(runtime, "get_item_by_name", return_value=item),
            patch.object(runtime, "use_item_from_bag", return_value=iter(())) as use_item,
            patch.object(runtime, "scroll_to_party_menu_index", return_value=iter(())) as scroll,
            patch.object(
                runtime,
                "get_game_state",
                side_effect=(
                    GameState.PARTY_MENU,
                    GameState.PARTY_MENU,
                    GameState.OVERWORLD,
                    GameState.OVERWORLD,
                    GameState.OVERWORLD,
                ),
            ),
            patch.object(runtime, "_close_field_recovery_menus", return_value=iter(())),
            patch.object(runtime, "context", SimpleNamespace(emulator=emulator)),
        ):
            list(runtime._use_field_recovery_item(item))

        use_item.assert_called_once_with(item, wait_for_start_menu_to_reappear=False)
        scroll.assert_called_once_with(0)

    def test_field_recovery_retries_fresh_b_until_bag_and_start_menu_close(self):
        from modules.nuzlocke import resource_runtime as runtime

        item = HealingResource("Antidote", 1, 0)
        before = ResourceSnapshot(
            party=(PartyResource(12, 20, "poisoned"),),
            bag_healing_items=(item,),
        )
        after = ResourceSnapshot(
            party=(PartyResource(12, 20, "none"),),
            bag_healing_items=(HealingResource("Antidote", 0, 0),),
        )
        party = (
            SimpleNamespace(
                index=0,
                current_hp=12,
                total_hp=20,
                status_condition=SimpleNamespace(value="poisoned"),
                is_egg=False,
            ),
        )
        states = iter(
            (
                GameState.PARTY_MENU,
                GameState.PARTY_MENU,
                GameState.BAG_MENU,
                GameState.BAG_MENU,
                GameState.BAG_MENU,
                GameState.BAG_MENU,
                GameState.OVERWORLD,
                GameState.OVERWORLD,
                GameState.OVERWORLD,
                GameState.OVERWORLD,
                GameState.OVERWORLD,
            )
        )
        current_state = [None]
        start_menu_open = [False]
        start_menu_appeared = [False]
        overworld_frames = [0]
        emulator = SimpleNamespace(press_button=Mock(), press_button_fresh=Mock())

        def get_state():
            current_state[0] = next(states)
            if current_state[0] is GameState.OVERWORLD:
                overworld_frames[0] += 1
                if overworld_frames[0] >= 3 and not start_menu_appeared[0]:
                    start_menu_open[0] = True
                    start_menu_appeared[0] = True
            return current_state[0]

        def task_is_active(task_name):
            if task_name == "Task_BagMenu_HandleInput":
                return current_state[0] is GameState.BAG_MENU
            if task_name == "Task_ShowStartMenu":
                return current_state[0] is GameState.OVERWORLD and start_menu_open[0]
            return False

        def press_fresh(button):
            if button == "B" and current_state[0] is GameState.OVERWORLD:
                start_menu_open[0] = False

        emulator.press_button_fresh.side_effect = press_fresh
        with (
            patch.object(runtime, "get_party", return_value=party),
            patch.object(runtime, "observe_resource_snapshot", side_effect=(before, after)),
            patch.object(runtime, "get_item_by_name", return_value=item),
            patch.object(runtime, "use_item_from_bag", return_value=iter(())),
            patch.object(runtime, "scroll_to_party_menu_index", return_value=iter(())),
            patch.object(runtime, "get_game_state", side_effect=get_state),
            patch.object(runtime, "task_is_active", side_effect=task_is_active),
            patch.object(runtime, "parse_start_menu", return_value={"open": False}),
            patch.object(runtime, "context", SimpleNamespace(emulator=emulator)),
        ):
            list(runtime._use_field_recovery_item(item))

        assert emulator.press_button_fresh.call_args_list == [
            (('A',),),
            (('B',),),
            (('B',),),
            (('B',),),
        ]
        emulator.press_button.assert_not_called()

    def test_pc_withdrawal_prefers_a_status_cure_for_the_current_party(self):
        from modules.nuzlocke import resource_runtime as runtime

        item = HealingResource("Antidote", 2, 0, location="pc")
        snapshot = ResourceSnapshot(
            party=(PartyResource(12, 20, "poisoned"),),
            pc_healing_items=(item, HealingResource("Potion", 2, 20, location="pc")),
        )
        after = ResourceSnapshot(
            party=(PartyResource(12, 20, "poisoned"),),
            bag_healing_items=(HealingResource("Antidote", 1, 0),),
            pc_healing_items=(
                HealingResource("Antidote", 1, 0, location="pc"),
                HealingResource("Potion", 2, 20, location="pc"),
            ),
        )
        with (
            patch.object(runtime, "observe_resource_snapshot", side_effect=(snapshot, after)),
            patch.object(runtime, "get_item_by_name", return_value=item),
            patch.object(runtime, "interact_with_pc", return_value=iter(("withdraw",))) as interact,
        ):
            self.assertEqual(list(runtime.withdraw_best_pc_healing_item()), ["withdraw"])

        action = interact.call_args.args[0][0]
        self.assertEqual(action.item, item)
        self.assertEqual(action.quantity, 1)

    def test_pc_withdrawal_can_collect_hp_medicine_before_starter_is_in_party(self):
        from modules.nuzlocke import resource_runtime as runtime

        item = HealingResource("Potion", 1, 20, location="pc")
        snapshot = ResourceSnapshot(party=(), pc_healing_items=(item,))
        after = ResourceSnapshot(
            party=(),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
            pc_healing_items=(),
        )
        with (
            patch.object(runtime, "observe_resource_snapshot", side_effect=(snapshot, after)),
            patch.object(runtime, "get_item_by_name", return_value=item),
            patch.object(runtime, "interact_with_pc", return_value=iter(("withdraw",))) as interact,
        ):
            self.assertEqual(list(runtime.withdraw_best_pc_healing_item()), ["withdraw"])

        action = interact.call_args.args[0][0]
        self.assertEqual(action.item, item)
        self.assertEqual(action.quantity, 1)

    def test_pc_withdrawal_reports_empty_storage_instead_of_succeeding_silently(self):
        from modules.nuzlocke import resource_runtime as runtime

        previous_status = getattr(context, "campaign_recovery_pc_status", None)
        try:
            with (
                patch.object(runtime, "observe_resource_snapshot", return_value=ResourceSnapshot(party=())),
                patch.object(runtime, "diagnostic_print") as diagnostic,
            ):
                self.assertEqual(list(runtime.withdraw_best_pc_healing_item()), [])

            self.assertEqual(context.campaign_recovery_pc_status, "no_stock")
            rendered = [
                call.args[0]() if callable(call.args[0]) else call.args[0]
                for call in diagnostic.call_args_list
            ]
            self.assertTrue(any("status='no_stock'" in message for message in rendered))
        finally:
            context.campaign_recovery_pc_status = previous_status

    def test_center_recovery_reaches_the_observed_item_pc_before_withdrawing(self):
        from modules.nuzlocke import resource_runtime as runtime

        item = HealingResource("Potion", 1, 20, location="pc")
        after = ResourceSnapshot(
            party=(PartyResource(20, 20, "none"),),
            bag_healing_items=(HealingResource("Potion", 1, 20),),
            pc_healing_items=(),
        )
        map_id = MapRSE.PETALBURG_CITY_POKEMON_CENTER_1F.value
        trigger = SimpleNamespace(
            trigger_id="object:4:EventScript_PC",
            kind="object_interaction",
            affordance_id="EventScript_PC",
            activation_locations=frozenset({(map_id, (10, 2))}),
            activation_requirements=(((map_id, (10, 2)), Direction.North),),
        )
        initial = SimpleNamespace(
            overworld=SimpleNamespace(
                map_id=map_id,
                player_coordinates=(7, 8),
                facing=Direction.South,
                triggers=(trigger,),
            )
        )
        targeted = SimpleNamespace(
            overworld=SimpleNamespace(
                map_id=map_id,
                player_coordinates=(10, 2),
                facing=Direction.South,
                triggers=(trigger,),
            )
        )
        emulator = SimpleNamespace(press_direction=Mock(), press_button=Mock())
        navigation_goals = []

        class FakeNavigationLoop:
            def __init__(self, _observe, *, goal, **_kwargs):
                navigation_goals.append(goal)

            def run(self):
                return iter(())

        with (
            patch.object(
                runtime,
                "observe_resource_snapshot",
                side_effect=(
                    ResourceSnapshot(pc_healing_items=(item,)),
                    ResourceSnapshot(pc_healing_items=(item,)),
                    after,
                ),
            ),
            patch.object(runtime, "observe_agent", side_effect=(initial, targeted)),
            patch.object(runtime, "AgentControlLoop", FakeNavigationLoop),
            patch.object(runtime, "get_game_state", return_value=GameState.OVERWORLD),
            patch.object(runtime, "get_item_by_name", return_value=item),
            patch.object(runtime, "interact_with_pc", return_value=iter(())) as interact,
            patch.object(runtime, "context", SimpleNamespace(emulator=emulator)),
        ):
            self.assertEqual(list(runtime._withdraw_pc_healing_item_at_current_location()), [None])

        self.assertIsInstance(navigation_goals[0], NavigationGoal)
        self.assertEqual(navigation_goals[0].target.location, (map_id, (10, 2)))
        emulator.press_direction.assert_called_once_with("Up", run=False, fresh=True)
        action = interact.call_args.args[0][0]
        self.assertEqual(action.item, item)
        self.assertEqual(action.quantity, 1)

    def test_preparation_center_loop_uses_observation_driven_recovery_handler(self):
        recovery_calls = []

        def recovery_handler(center):
            recovery_calls.append(center)
            yield "door-aware-recovery"

        def stop_condition():
            return bool(recovery_calls)

        encounter_spot = SimpleNamespace(local_position=(3, 3))
        with patch(
            "modules.modes.util.pokecenter_loop.get_map_data_for_current_position",
            return_value=encounter_spot,
        ), patch(
            "modules.modes.util.pokecenter_loop.find_closest_pokemon_center",
            return_value=PokemonCenter.OldaleTown,
        ), patch(
            "modules.modes.util.pokecenter_loop.navigate_to",
            return_value=iter(()),
        ), patch(
            "modules.modes.util.pokecenter_loop.get_map_enum",
            return_value=MapRSE.ROUTE101,
        ), patch(
            "modules.modes.util.pokecenter_loop.apply_white_flute_if_available",
            return_value=iter(()),
        ), patch(
            "modules.modes.util.pokecenter_loop.spin",
            side_effect=lambda *, stop_condition: iter(()),
        ), patch(
            "modules.modes.util.pokecenter_loop.heal_in_pokemon_center",
            side_effect=AssertionError("legacy blocked-door recovery was used"),
        ):
            controller = PokecenterLoopController(recovery_handler=recovery_handler)
            self.assertEqual(list(controller.run(stop_condition=stop_condition)), ["door-aware-recovery"])

        self.assertEqual(recovery_calls, [PokemonCenter.OldaleTown])

    def test_preparation_center_loop_uses_injected_center_finder(self):
        recovery_calls = []
        selected_locations = []

        def center_finder(location):
            selected_locations.append(location)
            return PokemonCenter.OldaleTown

        def recovery_handler(center):
            recovery_calls.append(center)
            yield "campaign-recovery"

        encounter_spot = SimpleNamespace(
            local_position=(3, 3),
            has_encounters=True,
            is_surfable=False,
        )
        with patch(
            "modules.modes.util.pokecenter_loop.get_map_data_for_current_position",
            return_value=encounter_spot,
        ), patch(
            "modules.modes.util.pokecenter_loop.get_effective_encounter_rates_for_current_map",
            return_value=SimpleNamespace(land_encounters=(object(),), surf_encounters=()),
        ), patch(
            "modules.modes.util.pokecenter_loop.find_closest_pokemon_center",
            side_effect=AssertionError("preparation must use its injected Center selector"),
        ), patch(
            "modules.modes.util.pokecenter_loop.navigate_to",
            return_value=iter(()),
        ), patch(
            "modules.modes.util.pokecenter_loop.get_map_enum",
            return_value=MapRSE.ROUTE101,
        ), patch(
            "modules.modes.util.pokecenter_loop.apply_white_flute_if_available",
            return_value=iter(()),
        ), patch(
            "modules.modes.util.pokecenter_loop.spin",
            side_effect=lambda *, stop_condition: iter(()),
        ):
            controller = PokecenterLoopController(
                recovery_handler=recovery_handler,
                center_finder=center_finder,
            )
            controller.verify_on_start()
            self.assertEqual(list(controller.run(stop_condition=lambda: bool(recovery_calls))), ["campaign-recovery"])

        self.assertEqual(recovery_calls, [PokemonCenter.OldaleTown])
        self.assertEqual(selected_locations, [encounter_spot, encounter_spot])

    def test_preparation_recovery_exits_center_before_resuming_grass_navigation(self):
        interior = MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F
        with patch(
            "modules.nuzlocke.resource_runtime.execute_planned_recovery",
            return_value=iter(("healed",)),
        ), patch(
            "modules.nuzlocke.resource_runtime.get_player_location",
            return_value=(interior, (7, 4)),
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            return_value=SimpleNamespace(
                overworld=SimpleNamespace(map_id=interior, controllable=True),
            ),
        ), patch(
            "modules.nuzlocke.resource_runtime.navigate_to",
            return_value=iter(("exited",)),
        ) as navigate:
            self.assertEqual(
                list(_execute_preparation_recovery(PokemonCenter.OldaleTown)),
                ["healed", "exited"],
            )

        navigate.assert_called_once_with(interior, (7, 8))

    def test_preparation_stops_without_navigation_when_target_is_already_met(self):
        party = (SimpleNamespace(is_egg=False, level=14),)
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=party), patch(
            "modules.nuzlocke.resource_runtime._preparation_training_location"
        ) as training_location:
            self.assertEqual(list(execute_campaign_preparation((0, 18), target_level=14)), [])

        training_location.assert_not_called()

    def test_preparation_relays_battle_end_to_pokecenter_loop_before_stopping(self):
        party = (SimpleNamespace(is_egg=False, level=1),)

        class FakeNavigationLoop:
            def run(self):
                return iter(())

        class FakePokecenterLoop:
            def __init__(self, **_kwargs):
                from unittest.mock import Mock

                self.on_battle_ended = Mock()
                self.verify_on_start = Mock()
                self.battle_strategy = None

            def run(self, *, stop_condition):
                stop_condition()
                return iter(())

        pokecenter_loop = FakePokecenterLoop()
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=party), patch(
            "modules.nuzlocke.resource_runtime._preparation_training_location",
            return_value=((0, 16), ((1, 1),)),
        ), patch("modules.nuzlocke.resource_runtime.AgentControlLoop", return_value=FakeNavigationLoop()), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            return_value=SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 16), player_coordinates=(1, 1))),
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld",
            return_value=object(),
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(object(), None),
        ), patch(
            "modules.nuzlocke.resource_runtime.pokemon_center_candidates",
            return_value=(PokemonCenter.OldaleTown,),
        ), patch(
            "modules.nuzlocke.resource_runtime.CampaignPreparationLoopController", return_value=pokecenter_loop
        ):
            self.assertEqual(list(execute_campaign_preparation((0, 16), target_level=14)), [])

        pokecenter_loop.verify_on_start.assert_called_once_with()
        pokecenter_loop.on_battle_ended.assert_called_once_with()

    def test_preparation_publishes_a_target_for_the_trainer_safe_grass_goal(self):
        from modules.goals import ReachLocation, TrainerMode
        from modules.nuzlocke import resource_runtime as runtime

        goal = NavigationGoal(
            ReachLocation((MapRSE.ROUTE103.value, (11, 14))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        fake_context = SimpleNamespace(campaign_preparation_target=None, campaign_status=None)
        with patch.object(runtime, "context", fake_context):
            target = runtime._publish_preparation_target(
                goal, SimpleNamespace(map_id=MapRSE.ROUTE103.value, objects=(), triggers=())
            )

        self.assertIsNotNone(target)
        self.assertTrue(fake_context.campaign_preparation_target)

    def test_preparation_logs_when_a_goal_cannot_be_described(self):
        from modules.nuzlocke import resource_runtime as runtime

        logs = []
        fake_context = SimpleNamespace(campaign_preparation_target=None, campaign_status=None)
        with patch.object(runtime, "context", fake_context), patch.object(
            runtime, "print_always", side_effect=lambda message: logs.append(message())
        ):
            result = runtime._publish_preparation_target(object(), SimpleNamespace(objects=(), triggers=()))

        self.assertIsNone(result)
        self.assertTrue(any("CAMPAIGN_PREPARATION_TARGET_UNDESCRIBED" in line for line in logs))

    def test_preparation_grass_goal_defaults_to_trainer_avoidance(self):
        from modules.goals import ReachLocation, TrainerMode

        training_map = (0, 16)
        overworld = SimpleNamespace(map_id=training_map, player_coordinates=(1, 1), objects=(), triggers=())
        with patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(SimpleNamespace(forced_trainer_exposure=False), None),
        ):
            selected, error = _preparation_navigation_goal(overworld, training_map, ((8, 8),))

        self.assertIsNone(error)
        self.assertIsInstance(selected, NavigationGoal)
        self.assertIsInstance(selected.target, ReachLocation)
        self.assertEqual(selected.constraints.trainer_mode, TrainerMode.AVOID)
        self.assertFalse(selected.constraints.allow_forced_trainer_exposure)

    def test_preparation_safe_route_failure_does_not_fall_back_to_permissive_grass(self):
        from modules.navigation import NavigationError

        training_map = (0, 16)
        overworld = SimpleNamespace(map_id=training_map, player_coordinates=(1, 1), objects=(), triggers=())
        with patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            side_effect=NavigationError("trainer-safe route not found"),
        ):
            selected, error = _preparation_navigation_goal(overworld, training_map, ((8, 8),))

        self.assertIsNone(selected)
        self.assertIsNotNone(error)

    def test_preparation_rejects_grass_route_that_requires_trainer_exposure(self):
        training_map = (0, 16)
        overworld = SimpleNamespace(map_id=training_map, player_coordinates=(1, 1), objects=(), triggers=())
        with patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(SimpleNamespace(forced_trainer_exposure=True), None),
        ):
            selected, error = _preparation_navigation_goal(overworld, training_map, ((8, 8),))

        self.assertIsNone(selected)
        self.assertIsNotNone(error)

    def test_preparation_map_route_keeps_trainer_avoidance(self):
        from modules.goals import TrainerMode

        party = (SimpleNamespace(is_egg=False, level=1),)
        goals = []

        class FakeNavigationLoop:
            def run(self):
                return iter(())

        class FakePokecenterLoop:
            def __init__(self, **_kwargs):
                self.battle_strategy = None

            def on_battle_ended(self):
                return None

            def verify_on_start(self):
                return None

            def run(self, *, stop_condition):
                stop_condition()
                return iter(())

        def make_loop(_observe, goal=None, **_kwargs):
            goals.append(goal)
            return FakeNavigationLoop()

        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=party), patch(
            "modules.nuzlocke.resource_runtime._preparation_training_location",
            return_value=((0, 16), ((1, 1),)),
        ), patch("modules.nuzlocke.resource_runtime.AgentControlLoop", side_effect=make_loop), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            return_value=SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 16), player_coordinates=(1, 1))),
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(SimpleNamespace(forced_trainer_exposure=False), None),
        ), patch(
            "modules.nuzlocke.resource_runtime.CampaignPreparationLoopController",
            return_value=FakePokecenterLoop(),
        ):
            list(execute_campaign_preparation((0, 16), target_level=14))

        self.assertGreaterEqual(len(goals), 2)
        for goal in goals[:2]:
            self.assertEqual(goal.constraints.trainer_mode, TrainerMode.AVOID)

    def _run_preparation_with_failing_global_route(self, *, staged_result):
        from modules.nuzlocke import resource_runtime as runtime
        from modules.navigation import NavigationError

        party = (SimpleNamespace(is_egg=False, level=1),)
        staged_calls = []
        logs = []
        loops = []

        class FakeLoop:
            def __init__(self, fails):
                self.fails = fails

            def run(self):
                if self.fails:

                    def failed():
                        raise NavigationError(
                            "preferred map route is currently unreachable: (a, b, c)"
                        )
                        yield

                    return failed()
                return iter(())

        class FakePokecenterLoop:
            def __init__(self, **_kwargs):
                self.battle_strategy = None

            def on_battle_ended(self):
                return None

            def verify_on_start(self):
                return None

            def run(self, *, stop_condition):
                stop_condition()
                return iter(())

        def make_loop(_observe, goal=None, **_kwargs):
            loops.append(goal)
            # Only the first (global, map-level) route fails.
            return FakeLoop(fails=len(loops) == 1)

        def fake_travel(map_id, graph):
            staged_calls.append(map_id)
            return staged_result
            yield

        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=party), patch.object(
            runtime, "_preparation_training_location", return_value=((0, 16), ((1, 1),))
        ), patch.object(runtime, "AgentControlLoop", side_effect=make_loop), patch.object(
            runtime,
            "observe_agent",
            return_value=SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 16), player_coordinates=(1, 1))),
        ), patch.object(runtime, "get_world_map_graph", return_value=SimpleNamespace()), patch.object(
            runtime, "_travel_to_map_staged", side_effect=fake_travel
        ), patch.object(
            runtime.NavigationWorld, "from_overworld", return_value=object()
        ), patch.object(
            runtime,
            "plan_with_world_navigation",
            return_value=(SimpleNamespace(forced_trainer_exposure=False), None),
        ), patch.object(
            runtime, "CampaignPreparationLoopController", return_value=FakePokecenterLoop()
        ), patch.object(
            runtime, "print_always", side_effect=lambda message: logs.append(message())
        ):
            try:
                list(execute_campaign_preparation((0, 16), target_level=14))
                error = None
            except Exception as caught:  # noqa: BLE001 - asserted by the callers
                error = caught
        return staged_calls, logs, loops, error

    def test_preparation_falls_back_to_staged_travel_when_the_global_route_fails(self):
        from modules.goals import TrainerMode

        staged_calls, logs, loops, error = self._run_preparation_with_failing_global_route(staged_result=True)

        self.assertIsNone(error)
        self.assertEqual(staged_calls, [(0, 16)])
        self.assertTrue(any("CAMPAIGN_PREPARATION_MAP_ROUTE" in line for line in logs))
        # The failed global attempt was itself trainer-avoiding, and so was
        # everything after it: the fallback is not a permissive route.
        for goal in loops:
            self.assertEqual(goal.constraints.trainer_mode, TrainerMode.AVOID)

    def test_preparation_surfaces_a_staged_travel_failure_instead_of_going_permissive(self):
        from modules.modes._interface import BotModeError

        staged_calls, _logs, loops, error = self._run_preparation_with_failing_global_route(staged_result=False)

        self.assertEqual(staged_calls, [(0, 16)])
        self.assertIsInstance(error, BotModeError)
        self.assertIn("no trainer-safe staged route", str(error))
        # No further navigation was attempted after the staged route failed.
        self.assertEqual(len(loops), 1)

    def test_preparation_can_select_a_reachable_trainer_when_grass_is_blocked(self):
        from modules.goals import EngageTrainer, ReachLocation, TrainerMode
        from modules.navigation import NavigationError
        from modules.overworld import ObjectObservation

        training_map = (0, 16)
        trainer = ObjectObservation(
            local_id=3,
            location=(training_map, (5, 5)),
            trainer_type="Normal",
            trainer_range=3,
            trainer_defeated=False,
        )
        overworld = SimpleNamespace(
            map_id=training_map,
            player_coordinates=(1, 1),
            objects=(trainer,),
        )

        def plan(_world, _start, goal):
            # The trainer-safe grass probe wraps its ReachLocation in an AVOID
            # NavigationGoal; the blocked-grass fixture rejects either shape.
            if isinstance(getattr(goal, "target", goal), ReachLocation):
                raise NavigationError("grass component is blocked")
            self.assertIsInstance(goal.target, EngageTrainer)
            return object(), None

        with patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld",
            return_value=object(),
        ), patch("modules.nuzlocke.resource_runtime.plan_with_world_navigation", side_effect=plan):
            selected, error = _preparation_navigation_goal(overworld, training_map, ((8, 8),))

        self.assertIsNone(error)
        self.assertIsInstance(selected, NavigationGoal)
        self.assertIsInstance(selected.target, EngageTrainer)
        self.assertEqual(selected.target.trainer_id, trainer.trainer_id)
        self.assertEqual(selected.constraints.trainer_mode, TrainerMode.ENGAGE)

    def test_preparation_rejects_trainer_behind_static_blocker(self):
        training_map = (0, 31)
        trainer_id = f"trainer:{training_map}:16"
        tiles = tuple(
            TileObservation(
                (training_map, (x, 0)),
                blocked=False,
                walkable_neighbors=frozenset(Direction),
            )
            for x in range(5)
        )
        overworld = OverworldObservation(
            map_id=training_map,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=tiles,
            warps=(),
            objects=(),
            triggers=(
                TriggerObservation(
                    trigger_id="static_object:16:Route116_EventScript_Janice",
                    locations=frozenset({(training_map, (4, 0))}),
                    affordance_id=trainer_id,
                    kind="trainer_hazard",
                    hazard_kind="trainer",
                    engagement_locations=frozenset({(training_map, (4, 0))}),
                ),
            ),
            static_blocked_coordinates=frozenset({(2, 0)}),
        )

        selected, error = _preparation_navigation_goal(overworld, training_map, ((4, 0),))

        self.assertIsNone(selected)
        self.assertIsNotNone(error)

    def test_preparation_uses_static_trainer_trigger_after_runtime_object_disappears(self):
        from modules.goals import EngageTrainer, ReachLocation, TrainerMode
        from modules.navigation import NavigationError
        from modules.overworld import TriggerObservation

        training_map = (0, 31)
        trainer_id = f"trainer:{training_map}:6"
        overworld = SimpleNamespace(
            map_id=training_map,
            player_coordinates=(11, 17),
            objects=(),
            triggers=(
                TriggerObservation(
                    trigger_id="static_object:6:Route116_EventScript_Trainer",
                    locations=frozenset({(training_map, (12, 17))}),
                    kind="trainer_hazard",
                    affordance_id=trainer_id,
                    hazard_locations=frozenset({(training_map, (11, 17))}),
                    hazard_kind="trainer",
                    engagement_locations=frozenset({(training_map, (11, 17))}),
                ),
            ),
        )

        def plan(_world, _start, goal):
            # The trainer-safe grass probe wraps its ReachLocation in an AVOID
            # NavigationGoal; the blocked-grass fixture rejects either shape.
            if isinstance(getattr(goal, "target", goal), ReachLocation):
                raise NavigationError("grass component is blocked")
            self.assertIsInstance(goal.target, EngageTrainer)
            return object(), None

        with patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld",
            return_value=object(),
        ), patch("modules.nuzlocke.resource_runtime.plan_with_world_navigation", side_effect=plan):
            selected, error = _preparation_navigation_goal(overworld, training_map, ((8, 8),))

        self.assertIsNone(error)
        self.assertIsInstance(selected, NavigationGoal)
        self.assertEqual(selected.target.trainer_id, trainer_id)
        self.assertEqual(selected.constraints.trainer_mode, TrainerMode.ENGAGE)

    def test_preparation_can_reenter_grass_after_area_encounter_is_resolved(self):
        from modules.goals import EngageTrainer, NavigationGoal, ReachLocation
        from modules.overworld import ObjectObservation

        training_map = (0, 18)
        trainer = ObjectObservation(
            local_id=3,
            location=(training_map, (5, 5)),
            trainer_type="Normal",
            trainer_range=3,
            trainer_defeated=False,
        )
        party = [SimpleNamespace(is_egg=False, level=1)]
        runtime = SimpleNamespace(
            rules_projection=SimpleNamespace(
                state=SimpleNamespace(
                    encounters=(SimpleNamespace(location=training_map, status="captured", eligible=True),)
                )
            )
        )
        observations = iter(
            (
                SimpleNamespace(
                    overworld=SimpleNamespace(
                        map_id=training_map,
                        player_coordinates=(1, 1),
                        objects=(trainer,),
                    )
                ),
                SimpleNamespace(
                    overworld=SimpleNamespace(
                        map_id=training_map,
                        player_coordinates=(5, 4),
                        objects=(trainer,),
                    )
                ),
            )
        )

        class FakeNavigationLoop:
            def __init__(self, goal):
                self.goal = goal

            def run(self):
                if isinstance(self.goal, NavigationGoal) and isinstance(self.goal.target, (EngageTrainer, ReachLocation)):
                    party[0].level = 14
                return iter(())

        class FakePokecenterLoop:
            def __init__(self, **_kwargs):
                self.battle_strategy = None

            def verify_on_start(self):
                return None

            def on_battle_ended(self):
                return None

            def run(self, *, stop_condition):
                stop_condition()
                return iter(())

        with patch("modules.nuzlocke.resource_runtime.context", SimpleNamespace(nuzlocke_runtime=runtime)), patch(
            "modules.nuzlocke.resource_runtime.get_party", return_value=party
        ), patch("modules.nuzlocke.resource_runtime.PokemonIdentity.from_pokemon", return_value=object()), patch(
            "modules.nuzlocke.resource_runtime._preparation_training_location",
            return_value=(training_map, ((8, 8),)),
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop",
            side_effect=lambda _factory, goal: FakeNavigationLoop(goal),
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=observations
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(object(), None)
        ), patch(
            "modules.nuzlocke.resource_runtime.CampaignPreparationLoopController",
            return_value=FakePokecenterLoop(),
        ):
            self.assertEqual(list(execute_campaign_preparation(training_map, target_level=14)), [])

    def test_preparation_reobserves_after_navigation_failure(self):
        from modules.navigation import NavigationError

        party = (SimpleNamespace(is_egg=False, level=1),)

        class FakeNavigationLoop:
            def __init__(self, failure=False):
                self.failure = failure

            def run(self):
                if self.failure:

                    def failed():
                        raise NavigationError("battle displaced the avatar")
                        yield

                    return failed()
                return iter(())

        class FakePokecenterLoop:
            def __init__(self, **_kwargs):
                self.battle_strategy = None

            def on_battle_ended(self):
                return None

            def verify_on_start(self):
                return None

            def run(self, *, stop_condition):
                stop_condition()
                return iter(())

        observations = [
            SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 16), player_coordinates=(1, 1))),
            SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 16), player_coordinates=(2, 2))),
        ]
        loops = iter((FakeNavigationLoop(failure=False), FakeNavigationLoop(failure=True), FakeNavigationLoop()))
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=party), patch(
            "modules.nuzlocke.resource_runtime._preparation_training_location",
            return_value=((0, 16), ((3, 3),)),
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop", side_effect=lambda *_args, **_kwargs: next(loops)
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=observations
        ) as observe, patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(object(), None)
        ), patch(
            "modules.nuzlocke.resource_runtime.CampaignPreparationLoopController", return_value=FakePokecenterLoop()
        ):
            self.assertEqual(list(execute_campaign_preparation((0, 16), target_level=14)), [])

        self.assertEqual(observe.call_count, 2)

    def test_planned_recovery_navigates_to_observed_nurse_before_interacting(self):
        nurse = type(
            "Nurse",
            (),
            {
                "source_id": "nurse",
                "interaction_trigger_id": "object:nurse",
            },
        )()
        interior = type(
            "Observation",
            (),
            {
                "overworld": type("World", (), {})(),
                "interaction_type": type("InteractionType", (), {"name": "OVERWORLD"})(),
                "interaction": type("Interaction", (), {"controllable": True})(),
            },
        )()
        navigation_goals = []

        class FakeLoop:
            def __init__(self, *, goal):
                navigation_goals.append(goal)

            def run(self):
                return iter(())

        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, True]), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop",
            side_effect=lambda _factory, goal, **_kwargs: FakeLoop(goal=goal),
        ), patch("modules.nuzlocke.resource_runtime.observe_agent", return_value=interior), patch(
            "modules.nuzlocke.resource_runtime.discover_healing_source", return_value=nurse
        ), patch(
            "modules.nuzlocke.resource_runtime._execute_healing_source_interaction", return_value=iter(())
        ), patch(
            "modules.nuzlocke.resource_runtime._wait_for_center_interior", return_value=iter(())
        ):
            list(execute_planned_recovery(PokemonCenter.OldaleTown.value))

        self.assertEqual(len(navigation_goals), 2)
        self.assertIsInstance(navigation_goals[0], NavigationGoal)
        self.assertIsInstance(navigation_goals[0].target, ReachWarp)
        self.assertEqual(navigation_goals[0].target.destination_map, MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value)
        self.assertIs(navigation_goals[0].constraints.trainer_mode, TrainerMode.AVOID)
        self.assertEqual(navigation_goals[1].trigger_id, "object:nurse")

    def test_planned_recovery_resumes_at_pc_cleanup_after_healing_confirmation(self):
        from modules.nuzlocke import resource_runtime as runtime

        destination = PokemonCenter.OldaleTown.value
        previous_phase = getattr(context, "campaign_recovery_phase", None)
        context.campaign_recovery_phase = ("cleanup", destination)
        try:
            with patch.object(runtime, "_execute_recovery_pc_cleanup", return_value=iter(("cleanup",))) as cleanup:
                self.assertEqual(list(execute_planned_recovery(destination)), ["cleanup"])
            cleanup.assert_called_once_with()
            self.assertIsNone(context.campaign_recovery_phase)
        finally:
            context.campaign_recovery_phase = previous_phase

    def test_planned_recovery_adopts_planner_route_for_outdoor_leg(self):
        planned_route = object()
        nurse = type(
            "Nurse",
            (),
            {"source_id": "nurse", "interaction_trigger_id": "object:nurse"},
        )()
        interior = SimpleNamespace(
            overworld=SimpleNamespace(map_id=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value),
            interaction_type=SimpleNamespace(name="OVERWORLD"),
            interaction=SimpleNamespace(controllable=True),
        )
        loop_calls = []

        class FakeLoop:
            def __init__(self, *args, **kwargs):
                loop_calls.append((args, kwargs))

            def run(self):
                return iter(())

        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, True]), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop", side_effect=FakeLoop
        ), patch("modules.nuzlocke.resource_runtime._wait_for_center_interior", return_value=iter(())), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=interior
        ), patch(
            "modules.nuzlocke.resource_runtime.discover_healing_source", return_value=nurse
        ), patch(
            "modules.nuzlocke.resource_runtime._execute_healing_source_interaction", return_value=iter(())
        ):
            list(execute_planned_recovery(PokemonCenter.OldaleTown.value, planned_route=planned_route))

        self.assertEqual(len(loop_calls), 2)
        self.assertIs(loop_calls[0][1]["navigation_plan"], planned_route)
        self.assertIsInstance(loop_calls[0][1]["goal"], NavigationGoal)
        self.assertIsInstance(loop_calls[0][1]["goal"].target, ReachLocation)
        self.assertIs(loop_calls[0][1]["goal"].constraints.trainer_mode, TrainerMode.AVOID)
        self.assertNotIn("navigation_plan", loop_calls[1][1])

    def test_planned_recovery_rejects_route_crossing_explicitly_unready_trainer(self):
        """A stale recovery route cannot turn a readiness block into exposure."""

        route_map = MapRSE.ROUTE104.value
        hazard = (route_map, (12, 42))
        destination = PokemonCenter.OldaleTown.value
        checkpoint = SimpleNamespace(
            checkpoint_id="rom:BPEE0:319",
            trainer_affordance_id="trainer:(0, 19):28",
            rom_trainer_id=319,
        )
        route = NavigationPlan(
            actions=(
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (route_map, (11, 42)),
                    hazard,
                ),
            ),
            destination=hazard,
        )
        goal = NavigationGoal(
            ReachLocation(destination),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        diagnostic = SimpleNamespace(
            trainer_checkpoint=checkpoint,
            trainer_risks=(
                SimpleNamespace(
                    trainer_id=checkpoint.trainer_affordance_id,
                    hazard_locations=frozenset({hazard}),
                ),
            ),
        )
        previous = (
            context.campaign_trainer_checkpoint,
            context.campaign_trainer_readiness,
            context.campaign_readiness_diagnostic,
        )
        context.campaign_trainer_checkpoint = checkpoint
        context.campaign_trainer_readiness = SimpleNamespace(status=SimpleNamespace(value="prepare"))
        context.campaign_readiness_diagnostic = diagnostic
        try:
            with patch("modules.nuzlocke.resource_runtime.party_is_restored", return_value=False), patch(
                "modules.nuzlocke.resource_runtime.AgentControlLoop"
            ) as control_loop:
                with self.assertRaisesRegex(RecoveryTrainerSafetyError, "explicitly unready trainer"):
                    list(
                        execute_planned_recovery(
                            destination,
                            planned_source=SimpleNamespace(
                                interior_map=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F
                            ),
                            planned_route=route,
                            planned_goal=goal,
                        )
                    )
            control_loop.assert_not_called()
        finally:
            (
                context.campaign_trainer_checkpoint,
                context.campaign_trainer_readiness,
                context.campaign_readiness_diagnostic,
            ) = previous

    def _recovery_guard_fixture(self, *, route_tiles, hazard_tiles, status=None, goal=None):
        from modules.nuzlocke.resource_runtime import _recovery_trainer_safety_error

        route_map = MapRSE.ROUTE104.value
        route = NavigationPlan(
            actions=tuple(
                NavigationAction(
                    NavigationActionType.MOVE,
                    Direction.East,
                    (route_map, source),
                    (route_map, destination),
                )
                for source, destination in zip(route_tiles, route_tiles[1:])
            ),
            destination=(route_map, route_tiles[-1]),
        )
        goal = goal or NavigationGoal(
            ReachLocation(PokemonCenter.OldaleTown.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        diagnostic = SimpleNamespace(
            trainer_checkpoint=None,
            trainer_risks=(
                SimpleNamespace(
                    trainer_id="trainer:(0, 19):26",
                    hazard_locations=frozenset((route_map, tile) for tile in hazard_tiles),
                ),
            ),
        )
        readiness = SimpleNamespace(status=SimpleNamespace(value=status)) if status else None
        previous = (
            context.campaign_trainer_checkpoint,
            context.campaign_trainer_readiness,
            context.campaign_readiness_diagnostic,
        )
        context.campaign_trainer_checkpoint = None
        context.campaign_trainer_readiness = readiness
        context.campaign_readiness_diagnostic = diagnostic
        self.addCleanup(
            lambda: (
                setattr(context, "campaign_trainer_checkpoint", previous[0]),
                setattr(context, "campaign_trainer_readiness", previous[1]),
                setattr(context, "campaign_readiness_diagnostic", previous[2]),
            )
        )
        return _recovery_trainer_safety_error(route, goal)

    def test_recovery_guard_rejects_any_undefeated_trainer_on_the_route(self):
        error = self._recovery_guard_fixture(
            route_tiles=((11, 42), (12, 42), (13, 42)),
            hazard_tiles=((12, 42),),
            status=None,
        )

        self.assertIsNotNone(error)
        self.assertIn("undefeated trainer", error)

    def test_recovery_guard_accepts_route_clear_of_every_trainer_hazard(self):
        self.assertIsNone(
            self._recovery_guard_fixture(
                route_tiles=((11, 42), (12, 42), (13, 42)),
                hazard_tiles=((12, 44),),
                status=None,
            )
        )

    def test_recovery_guard_rejects_non_avoid_and_forced_exposure_goals(self):
        permissive = NavigationGoal(
            ReachLocation(PokemonCenter.OldaleTown.value),
            constraints=GoalConstraints(trainer_mode=TrainerMode.IGNORE),
        )
        forced = NavigationGoal(
            ReachLocation(PokemonCenter.OldaleTown.value),
            constraints=GoalConstraints(
                trainer_mode=TrainerMode.AVOID,
                allow_forced_trainer_exposure=True,
            ),
        )
        for goal in (permissive, forced):
            with self.subTest(goal=goal):
                self.assertIsNotNone(
                    self._recovery_guard_fixture(
                        route_tiles=((11, 42), (13, 42)),
                        hazard_tiles=(),
                        goal=goal,
                    )
                )

    def test_planned_recovery_preserves_exact_center_warp_goal(self):
        planned_route = object()
        center_warp = object()
        planned_goal = NavigationGoal(
            ReachWarp(
                destination_map=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
                destination=(MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value, (7, 8)),
                warp=center_warp,
            ),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        nurse = type(
            "Nurse",
            (),
            {"source_id": "nurse", "interaction_trigger_id": "object:nurse"},
        )()
        interior = SimpleNamespace(
            overworld=SimpleNamespace(map_id=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value),
            interaction_type=SimpleNamespace(name="OVERWORLD"),
            interaction=SimpleNamespace(controllable=True),
        )
        loop_calls = []

        class FakeLoop:
            def __init__(self, *args, **kwargs):
                loop_calls.append((args, kwargs))

            def run(self):
                return iter(())

        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, True]), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop", side_effect=FakeLoop
        ), patch("modules.nuzlocke.resource_runtime._wait_for_center_interior", return_value=iter(())), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=interior
        ), patch(
            "modules.nuzlocke.resource_runtime.discover_healing_source", return_value=nurse
        ), patch(
            "modules.nuzlocke.resource_runtime._execute_healing_source_interaction", return_value=iter(())
        ):
            list(
                execute_planned_recovery(
                    PokemonCenter.OldaleTown.value,
                    planned_route=planned_route,
                    planned_goal=planned_goal,
                )
            )

        self.assertIs(loop_calls[0][1]["navigation_plan"], planned_route)
        self.assertIs(loop_calls[0][1]["goal"], planned_goal)
        self.assertIs(loop_calls[0][1]["goal"].target.warp, center_warp)

    def test_heal_party_uses_cataloged_door_warp_for_center_destination(self):
        outdoor = SimpleNamespace(
            overworld=SimpleNamespace(map_id=MapRSE.OLDALE_TOWN.value, player_coordinates=(6, 17)),
            interaction_type=SimpleNamespace(name="OVERWORLD"),
            interaction=SimpleNamespace(controllable=True),
        )
        interior = SimpleNamespace(
            overworld=SimpleNamespace(
                map_id=MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
                player_coordinates=(7, 8),
            ),
            interaction_type=SimpleNamespace(name="OVERWORLD"),
            interaction=SimpleNamespace(controllable=True),
        )
        destination_source = HealingSource(
            "nearest_full_party_source",
            HealingSourceType.FULL_PARTY_PROVIDER,
            PokemonCenter.OldaleTown.value,
            PokemonCenter.OldaleTown,
            interaction_trigger_id=None,
        )
        nurse = HealingSource(
            "nurse",
            HealingSourceType.POKEMON_CENTER_NURSE,
            ((2, 2), (7, 2)),
            "object:nurse",
            interaction_trigger_id="object:nurse",
        )
        goals = []

        class FakeLoop:
            def __init__(self, *, goal):
                goals.append(goal)

            def run(self):
                return iter(())

        with patch("modules.nuzlocke.resource_runtime.party_is_restored", return_value=False), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=[outdoor, interior]
        ), patch(
            "modules.nuzlocke.resource_runtime.discover_healing_source",
            side_effect=[None, destination_source, nurse],
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop",
            side_effect=lambda _factory, goal: FakeLoop(goal=goal),
        ), patch(
            "modules.nuzlocke.resource_runtime._execute_healing_source_interaction", return_value=iter(())
        ):
            list(execute_heal_party())

        self.assertIsInstance(goals[0], NavigationGoal)
        self.assertIsInstance(goals[0].target, ReachWarp)
        self.assertEqual(goals[0].target.destination_map, MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value)

    def test_observed_center_nurse_becomes_healing_affordance(self):
        trigger = type(
            "Trigger",
            (),
            {
                "kind": "object_interaction",
                "affordance_id": "OldaleTown_PokemonCenter_1F_EventScript_Nurse",
                "script_symbol": "OldaleTown_PokemonCenter_1F_EventScript_Nurse",
                "activation_locations": frozenset({((2, 2), (7, 4))}),
                "trigger_id": "object:1:OldaleTown_PokemonCenter_1F_EventScript_Nurse",
            },
        )()
        observation = type(
            "Observation",
            (),
            {
                "overworld": type("World", (), {"triggers": (trigger,)})(),
            },
        )()
        source = discover_healing_source(observation=observation)
        self.assertEqual(source.source_type, HealingSourceType.POKEMON_CENTER_NURSE)
        self.assertEqual(source.interaction_trigger_id, trigger.trigger_id)
        self.assertEqual(source.interaction_target, trigger.trigger_id)

    def test_observed_non_center_full_party_sources_are_healing_affordances(self):
        for script in (
            "PlayersHouse_1F_EventScript_Mom",
            "Route111_OldLadysRestStop_EventScript_OldLady",
        ):
            trigger = type(
                "Trigger",
                (),
                {
                    "kind": "object_interaction",
                    "affordance_id": script,
                    "script_symbol": script,
                    "activation_locations": frozenset({((1, 1), (4, 4))}),
                    "trigger_id": f"object:1:{script}",
                },
            )()
            observation = type("Observation", (), {"overworld": type("World", (), {"triggers": (trigger,)})()})()
            source = discover_healing_source(observation=observation)
            self.assertIsNotNone(source)
            self.assertEqual(source.source_type, HealingSourceType.FULL_PARTY_PROVIDER)

    def test_healing_source_dialogue_uses_common_action_and_fresh_completion(self):
        source = HealingSource("nurse", HealingSourceType.FULL_PARTY_PROVIDER, None, None)
        dialogue = AgentObservation(InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True))
        overworld = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        actions = [
            ActionDecision(AgentAction(AgentActionType.ADVANCE_DIALOGUE)),
            ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE)),
        ]
        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, True]), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=[dialogue, overworld]
        ), patch("modules.nuzlocke.resource_runtime.select_action", side_effect=actions), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

        # The common selector owns dialogue; no Center helper is involved.
        self.assertEqual(
            [decision.action.action_type for decision in actions],
            [AgentActionType.ADVANCE_DIALOGUE, AgentActionType.WAIT_REOBSERVE],
        )

    def test_healing_source_does_not_claim_success_after_interaction(self):
        source = HealingSource("nurse", HealingSourceType.FULL_PARTY_PROVIDER, None, None)
        overworld = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False] * 40), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=overworld
        ), patch("modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor):
            with self.assertRaisesRegex(RuntimeError, "HEALING_NOT_CONFIRMED"):
                list(_execute_healing_source_interaction(source))

    def test_healing_source_retains_ownership_during_delayed_interaction_start(self):
        source = HealingSource("nurse", HealingSourceType.POKEMON_CENTER_NURSE, None, None)
        overworld = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        dialogue = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True, controllable=False)
        )
        finished = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        with patch(
            "modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, False, False, True]
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=[overworld, overworld, dialogue, finished]
        ), patch(
            "modules.nuzlocke.resource_runtime.select_action",
            return_value=ActionDecision(AgentAction(AgentActionType.ADVANCE_DIALOGUE)),
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

    def test_healing_source_waits_for_stable_standing_before_interacting(self):
        source = HealingSource("nurse", HealingSourceType.POKEMON_CENTER_NURSE, None, None)
        moving_world = type(
            "World",
            (),
            {"map_id": "center", "movement_state": MovementState.MOVING, "controllable": True},
        )()
        standing_world = type(
            "World",
            (),
            {"map_id": "center", "movement_state": MovementState.STANDING, "controllable": True},
        )()
        moving = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, controllable=True),
            overworld=moving_world,
        )
        standing = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, controllable=True),
            overworld=standing_world,
        )
        finished = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, controllable=True),
            overworld=standing_world,
        )
        actions = []
        executor = type("Executor", (), {"execute": lambda self, action, observed: actions.append(action)})()
        with patch(
            "modules.nuzlocke.resource_runtime.party_is_restored",
            side_effect=[False, False, False, True],
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            side_effect=[moving, standing, standing, finished],
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

        self.assertEqual([action.action_type for action in actions], [AgentActionType.INTERACT])

    def test_healing_source_retains_planned_source_when_counter_trigger_is_transiently_hidden(self):
        source = HealingSource(
            "nurse",
            HealingSourceType.POKEMON_CENTER_NURSE,
            None,
            "object:nurse",
            interaction_trigger_id="object:nurse",
        )
        world = type(
            "World",
            (),
            {
                "map_id": "center",
                "movement_state": MovementState.STANDING,
                "controllable": True,
                "triggers": (),
            },
        )()
        standing = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, controllable=True),
            overworld=world,
        )
        actions = []
        executor = type("Executor", (), {"execute": lambda self, action, observed: actions.append(action)})()
        with patch(
            "modules.nuzlocke.resource_runtime.party_is_restored",
            side_effect=[False, False, True],
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            side_effect=[standing, standing, standing],
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

        self.assertEqual([action.action_type for action in actions], [AgentActionType.INTERACT])

    def test_script_owned_controllable_interaction_remains_owned(self):
        source = HealingSource("source", HealingSourceType.POKEMON_CENTER_NURSE, None, None)
        script_owned = AgentObservation(
            InteractionObservation(
                GameState.OVERWORLD,
                controllable=True,
                script_active=True,
                script_function="SomeInteraction",
                native_function="WaitForEffect",
            )
        )
        actions = []
        executor = type("Executor", (), {"execute": lambda self, action, observed: actions.append(action)})()
        finished = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        with patch(
            "modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, False, True, True]
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            side_effect=[script_owned, script_owned, script_owned, finished],
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))
        # The active script preserves ownership without emitting another
        # interaction or falsely reporting HEALING_NOT_CONFIRMED.
        self.assertEqual(actions, [])

    def test_ready_confirmation_precedes_generic_script_ownership(self):
        source = HealingSource("source", HealingSourceType.POKEMON_CENTER_NURSE, None, None)
        confirmation = AgentObservation(
            InteractionObservation(
                GameState.OVERWORLD,
                controllable=True,
                dialogue_waiting=True,
                script_active=True,
                choice_options=("YES", "NO"),
                choice_menu_active=True,
                choice_menu_input_ready=True,
            )
        )
        executor = type("Executor", (), {"execute": lambda self, action, observed: None})()
        decisions = [ActionDecision(AgentAction(AgentActionType.CHOOSE_DIALOGUE_OPTION, option="YES"))]
        finished = AgentObservation(InteractionObservation(GameState.OVERWORLD, controllable=True))
        with patch("modules.nuzlocke.resource_runtime.party_is_restored", side_effect=[False, True]), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=[confirmation, finished]
        ), patch("modules.nuzlocke.resource_runtime.select_action", side_effect=decisions), patch(
            "modules.nuzlocke.resource_runtime.AgentActionExecutor", return_value=executor
        ):
            list(_execute_healing_source_interaction(source))

        self.assertEqual(decisions[0].action.action_type, AgentActionType.CHOOSE_DIALOGUE_OPTION)

    def test_campaign_recovery_executes_the_selected_observation_driven_source(self):
        destination = PokemonCenter.OldaleTown.value
        planned_route = object()
        with patch(
            "modules.nuzlocke.resource_runtime.observe_route_recovery",
            return_value=RouteRecovery(
                center_available=True,
                center_location=destination,
                safe_to_reach_center=True,
                route=planned_route,
            ),
        ), patch(
            "modules.nuzlocke.resource_runtime.execute_planned_recovery",
            return_value=iter(("planned-recovery",)),
        ) as execute:
            self.assertEqual(list(execute_campaign_recovery()), ["planned-recovery"])
        self.assertEqual(execute.call_args.args[0], destination)
        self.assertEqual(execute.call_args.args[1].source_id, "pokemon_center:oldale")
        self.assertIs(execute.call_args.kwargs["planned_route"], planned_route)

    def test_campaign_recovery_rejects_a_route_without_an_executable_source(self):
        with patch(
            "modules.nuzlocke.resource_runtime.observe_route_recovery",
            return_value=RouteRecovery(center_available=True, safe_to_reach_center=True),
        ), patch(
            "modules.nuzlocke.resource_runtime.find_closest_pokemon_center",
            side_effect=AssertionError("campaign recovery must not reopen Center selection"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no executable healing source"):
                list(execute_campaign_recovery())

    def test_recovery_failure_reports_observed_execution_fields(self):
        from modules.nuzlocke import resource_runtime as runtime

        destination = PokemonCenter.OldaleTown.value
        stop = SimpleNamespace(destination=destination, reason="test")
        status = SimpleNamespace(target=SimpleNamespace(location=(MapRSE.ROUTE103.value, (14, 11))))
        previous = (
            getattr(context, "campaign_recovery_phase", None),
            getattr(context, "campaign_recovery_observed_triggers", None),
            getattr(context, "campaign_recovery_pc_target", None),
            getattr(context, "campaign_status", None),
        )
        context.campaign_recovery_phase = ("cleanup", destination)
        context.campaign_recovery_observed_triggers = ("object:pc",)
        context.campaign_recovery_pc_target = (MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value, (10, 2))
        context.campaign_status = status
        try:
            with (
                patch.object(runtime, "_execute_campaign_recovery_impl", side_effect=RuntimeError("test failure")),
                patch.object(runtime, "print_recovery_failure") as failure,
                patch.object(runtime, "get_game_state", return_value=GameState.POKE_STORAGE) as game_state,
            ):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    list(execute_campaign_recovery(stop))

            fields = failure.call_args.kwargs
            self.assertEqual(fields["phase"], str(("cleanup", destination)))
            self.assertEqual(fields["location"], (MapRSE.ROUTE103.value, (14, 11)))
            self.assertEqual(fields["game_state"], "POKE_STORAGE")
            self.assertEqual(fields["observed_triggers"], ("object:pc",))
            self.assertEqual(fields["target_pc"], context.campaign_recovery_pc_target)
            game_state.assert_called_once_with()
        finally:
            (
                context.campaign_recovery_phase,
                context.campaign_recovery_observed_triggers,
                context.campaign_recovery_pc_target,
                context.campaign_status,
            ) = previous

    def test_nearest_center_compatibility_entry_point_uses_planned_recovery(self):
        with patch(
            "modules.nuzlocke.resource_runtime.execute_planned_recovery",
            return_value=iter(("planned-recovery",)),
        ) as execute:
            self.assertEqual(
                list(recover_at_nearest_center(selected_center=PokemonCenter.OldaleTown)),
                ["planned-recovery"],
            )

        self.assertEqual(execute.call_args.args[0], PokemonCenter.OldaleTown.value)
        self.assertEqual(execute.call_args.args[1].source_id, "pokemon_center:oldale")

    def test_recovery_handoff_waits_for_stable_center_interior(self):
        transient = type(
            "Observation",
            (),
            {
                "overworld": type(
                    "World",
                    (),
                    {"map_id": MapRSE.OLDALE_TOWN.value, "controllable": True, "map_identity_source": "live_header"},
                )()
            },
        )()
        stable = type(
            "Observation",
            (),
            {
                "overworld": type(
                    "World",
                    (),
                    {
                        "map_id": MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
                        "controllable": True,
                        "map_identity_source": "live_header",
                    },
                )()
            },
        )()
        with patch("modules.nuzlocke.resource_runtime.observe_agent", side_effect=[transient, stable]):
            self.assertEqual(list(_wait_for_center_interior(PokemonCenter.OldaleTown)), [None])

    def test_recovery_handoff_warms_center_interior_before_observing_warp(self):
        interior = type(
            "Observation",
            (),
            {
                "overworld": type(
                    "World",
                    (),
                    {
                        "map_id": MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
                        "controllable": True,
                        "map_identity_source": "live_header",
                    },
                )()
            },
        )()
        with patch("modules.nuzlocke.resource_runtime.get_map_metadata") as warm, patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=interior
        ):
            self.assertEqual(list(_wait_for_center_interior(PokemonCenter.OldaleTown)), [])
        warm.assert_called_once_with(MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value)

    def test_recovery_handoff_rejects_save_block_only_interior_identity(self):
        stale_interior = type(
            "Observation",
            (),
            {
                "overworld": type(
                    "World",
                    (),
                    {
                        "map_id": MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
                        "controllable": True,
                        "map_identity_source": "save_block",
                        "save_block_map_id": MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
                        "live_map_candidates": (),
                    },
                )()
            },
        )()
        with patch("modules.nuzlocke.resource_runtime.get_map_metadata"), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", return_value=stale_interior
        ) as observe, patch("modules.nuzlocke.resource_runtime._MAP_IDENTITY_RESOLUTION_TIMEOUT", 2):
            with self.assertRaisesRegex(RuntimeError, "map identity did not resolve"):
                list(_wait_for_center_interior(PokemonCenter.OldaleTown))
        self.assertEqual(observe.call_count, 2)
        self.assertEqual(observe.call_args.kwargs, {"require_overworld": True})

    def test_recovery_handoff_reobserves_stale_frozen_exterior(self):
        stale = type(
            "Observation",
            (),
            {
                "overworld": type(
                    "World",
                    (),
                    {
                        "map_id": MapRSE.OLDALE_TOWN.value,
                        "player_coordinates": (6, 17),
                        "controllable": False,
                        "map_identity_source": "save_block",
                    },
                )()
            },
        )()
        interior = type(
            "Observation",
            (),
            {
                "overworld": type(
                    "World",
                    (),
                    {
                        "map_id": MapRSE.OLDALE_TOWN_POKEMON_CENTER_1F.value,
                        "player_coordinates": (7, 8),
                        "controllable": True,
                        "map_identity_source": "live_header",
                    },
                )()
            },
        )()
        with patch("modules.nuzlocke.resource_runtime.get_map_metadata"), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=[stale] * 91 + [interior]
        ):
            self.assertEqual(len(list(_wait_for_center_interior(PokemonCenter.OldaleTown))), 91)

    def test_capability_delegates_original_tactical_goal_without_readiness_preflight(self):
        goal = ActivateTrigger("introductory_rival")
        delegated = []

        def delegate(received_goal):
            delegated.append(received_goal)
            yield "tactical"

        snapshot = ResourceSnapshot(
            party=(PartyResource(20, 20, "none", False),),
            bag_healing_items=(),
            pc_healing_items=(),
        )
        capability = CampaignCapability("complete_intro_rival", object(), goal, delegate)

        with patch(
            "modules.nuzlocke.resource_runtime.observe_resource_snapshot", return_value=snapshot
        ) as observe, patch("modules.nuzlocke.resource_runtime.assess_campaign_resources") as assess, patch(
            "modules.nuzlocke.resource_runtime.observe_route_recovery"
        ) as find_center:
            self.assertEqual(next(capability()), "tactical")

        self.assertEqual(delegated, [goal])
        observe.assert_not_called()
        assess.assert_not_called()
        find_center.assert_not_called()

    def test_capability_does_not_mount_recovery_from_internal_resource_policy(self):
        goal = ActivateTrigger("introductory_rival")
        delegated = []

        def delegate(received_goal):
            delegated.append(received_goal)
            yield "tactical"

        snapshot = ResourceSnapshot(
            party=(PartyResource(5, 20, "none", False),),
            bag_healing_items=(),
            pc_healing_items=(),
        )
        capability = CampaignCapability("complete_intro_rival", object(), goal, delegate)
        with patch(
            "modules.nuzlocke.resource_runtime.observe_resource_snapshot", return_value=snapshot
        ) as observe, patch("modules.nuzlocke.resource_runtime.assess_campaign_resources") as assess:
            self.assertEqual(list(capability()), ["tactical"])

        self.assertEqual(delegated, [goal])
        observe.assert_not_called()
        assess.assert_not_called()



class PreparationEarlyStopTests(unittest.TestCase):
    """Preparation ends once the trainer line is ready, without a full-party level target."""

    @staticmethod
    def _member(level, hp=26, max_hp=26, name="Torchic"):
        return SimpleNamespace(
            is_egg=False,
            level=level,
            species=SimpleNamespace(name=name),
            total_hp=max_hp,
            current_hp=hp,
        )

    def _run(self, party, ready_check, *, on_run=None):
        from modules.nuzlocke import resource_runtime as runtime

        stop_results = []
        logs = []

        class FakeNavigationLoop:
            def run(self):
                return iter(())

        class FakePokecenterLoop:
            def __init__(self, **_kwargs):
                self.battle_strategy = None

            def on_battle_ended(self):
                return None

            def verify_on_start(self):
                return None

            def run(self, *, stop_condition):
                if on_run is not None:
                    on_run(stop_condition, stop_results)
                else:
                    stop_results.append(stop_condition())
                return iter(())

        with patch.object(runtime, "get_party", side_effect=lambda: party), patch.object(
            runtime, "_preparation_training_location", return_value=((0, 16), ((1, 1),))
        ), patch.object(runtime, "AgentControlLoop", side_effect=lambda *a, **k: FakeNavigationLoop()), patch.object(
            runtime,
            "observe_agent",
            return_value=SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 16), player_coordinates=(1, 1))),
        ), patch.object(runtime.NavigationWorld, "from_overworld", return_value=object()), patch.object(
            runtime,
            "plan_with_world_navigation",
            return_value=(SimpleNamespace(forced_trainer_exposure=False), None),
        ), patch.object(
            runtime, "CampaignPreparationLoopController", return_value=FakePokecenterLoop()
        ), patch.object(
            runtime, "print_always", side_effect=lambda message: logs.append(message())
        ):
            list(execute_campaign_preparation((0, 16), target_level=9, ready_check=ready_check))
        return stop_results, logs

    def test_a_ready_line_ends_preparation_before_the_whole_party_reaches_the_target(self):
        party = [self._member(8), self._member(3, name="Wurmple"), self._member(3, name="Wingull")]
        ready_check = Mock(return_value=True)

        stop_results, logs = self._run(party, ready_check)

        # The very first check already ends the capability: nothing was trained.
        self.assertEqual(stop_results, [])
        ready_check.assert_called_once_with()
        self.assertTrue(any("CAMPAIGN_PREPARATION_READY" in line for line in logs))

    def test_without_a_ready_check_the_full_party_target_still_applies(self):
        party = [self._member(8), self._member(3, name="Wurmple")]

        stop_results, logs = self._run(party, None)

        self.assertEqual(stop_results, [False])
        self.assertEqual(logs, [])

    def test_a_line_that_is_not_ready_keeps_training(self):
        party = [self._member(8), self._member(3, name="Wurmple")]

        stop_results, _ = self._run(party, Mock(return_value=False))

        self.assertEqual(stop_results, [False])

    def test_readiness_is_only_rechecked_when_the_party_changes(self):
        party = [self._member(8), self._member(3, name="Wurmple")]
        ready_check = Mock(return_value=False)

        def on_run(stop_condition, results):
            results.extend(stop_condition() for _ in range(5))  # nothing changed
            party[1].level = 4  # a level-up changes the readiness-relevant facts
            results.extend(stop_condition() for _ in range(5))

        self._run(party, ready_check, on_run=on_run)

        # Once at the start, once after the level-up; not on every call.
        self.assertEqual(ready_check.call_count, 2)

    def test_dropping_below_half_hp_and_healing_each_trigger_a_recheck(self):
        party = [self._member(8, hp=26), self._member(3, name="Wurmple")]
        ready_check = Mock(side_effect=[False, False, True])

        def on_run(stop_condition, results):
            results.append(stop_condition())  # unchanged since the opening check
            party[0].current_hp = 10  # below half: not eligible
            results.append(stop_condition())
            party[0].current_hp = 26  # healed: eligible again
            results.append(stop_condition())

        stop_results, _ = self._run(party, ready_check, on_run=on_run)

        self.assertEqual(ready_check.call_count, 3)
        self.assertEqual(stop_results, [False, False, True])

    def test_a_failing_readiness_check_never_ends_preparation(self):
        party = [self._member(8), self._member(3, name="Wurmple")]

        stop_results, _ = self._run(party, Mock(side_effect=RuntimeError("stats unavailable")))

        self.assertEqual(stop_results, [False])


class TrainerCheckpointReadyCheckTests(unittest.TestCase):
    def _check(self, status):
        from modules.nuzlocke import resource_runtime as runtime

        readiness = SimpleNamespace(status=SimpleNamespace(value=status))
        with patch.object(runtime, "assess_live_trainer_checkpoint", return_value=readiness) as assess, patch.object(
            runtime, "get_party", return_value=["party"]
        ):
            return runtime.trainer_checkpoint_ready_check("checkpoint")(), assess

    def test_only_a_ready_status_counts(self):
        ready, assess = self._check("ready")

        self.assertTrue(ready)
        assess.assert_called_once_with("checkpoint", ["party"])
        for status in ("prepare", "unknown", None):
            with self.subTest(status=status):
                self.assertFalse(self._check(status)[0])


if __name__ == "__main__":
    unittest.main()


class PreparationFailureExplanationTests(unittest.TestCase):
    """A failed preparation must say which stage failed and why."""

    @staticmethod
    def _evaluation(trainer_id, *, selected=False, reason="win probability below threshold"):
        return SimpleNamespace(
            source=SimpleNamespace(trainer_id=trainer_id),
            win_probability=0.5,
            catastrophic_probability=0.2,
            projected_exp=120,
            recipient_headroom=3,
            cap_safe=False,
            selected=selected,
            rejection_reason=None if selected else reason,
        )

    def _describe(self, **changes):
        from collections import Counter
        from modules.nuzlocke.resource_runtime import _describe_preparation_failure

        values = dict(
            allow_wild=True,
            grass_candidates=12,
            grass_reachable=7,
            grass_rejections=Counter(trainer_safe_route_unavailable=7),
            trainers_observed=2,
            trainers_without_checkpoint=(),
            evaluations=(self._evaluation("trainer:A"),),
            level_cap=15,
            target_level=10,
        )
        values.update(changes)
        return _describe_preparation_failure(**values)

    def test_grass_stage_names_the_counts_and_rejection_reasons(self):
        text = self._describe()

        self.assertIn("grass: 12 tiles, 7 reachable, rejected [trainer_safe_route_unavailable=7]", text)
        self.assertIn("target_level=10 level_cap=15", text)

    def test_grass_stage_says_when_the_area_encounter_disables_wild_training(self):
        text = self._describe(allow_wild=False)

        self.assertIn("grass: not used, the first encounter of this area is already resolved", text)

    def test_grass_stage_says_when_no_tiles_are_known(self):
        self.assertIn("no encounter tiles known", self._describe(grass_candidates=0))

    def test_each_trainer_reports_its_numbers_and_rejection_reason(self):
        text = self._describe(
            evaluations=(
                self._evaluation("trainer:A", reason="catastrophic probability above limit"),
                self._evaluation("trainer:B", reason="no recipient has cap headroom"),
            ),
            trainers_without_checkpoint=("trainer:C",),
            trainers_observed=3,
        )

        self.assertIn("trainer:A (win=50% catastrophic=20% exp=120 headroom=3 cap_safe=False)", text)
        self.assertIn("rejected: catastrophic probability above limit", text)
        self.assertIn("rejected: no recipient has cap headroom", text)
        self.assertIn("no ROM checkpoint, not evaluated: trainer:C", text)

    def test_no_observed_trainer_is_stated(self):
        text = self._describe(trainers_observed=0, evaluations=())

        self.assertIn("no undefeated trainer observed on this map", text)

    def test_the_explanation_is_logged_once_per_distinct_text(self):
        from modules.nuzlocke import resource_runtime as runtime

        runtime._last_preparation_no_line_signature = None
        with patch.object(runtime, "print_always") as printed:
            runtime._report_preparation_no_line((0, 19), "same")
            runtime._report_preparation_no_line((0, 19), "same")
            runtime._report_preparation_no_line((0, 19), "different")

        self.assertEqual(printed.call_count, 2)

    def test_a_resolved_area_with_no_trainers_explains_itself_instead_of_returning_none(self):
        from modules.nuzlocke import resource_runtime as runtime

        map_id = MapRSE.PETALBURG_WOODS.value
        overworld = SimpleNamespace(
            map_id=map_id,
            player_coordinates=(7, 35),
            objects=(),
            triggers=(),
        )
        runtime._last_preparation_no_line_signature = None
        with (
            patch.object(runtime.NavigationWorld, "from_overworld", return_value=object()),
            patch.object(runtime, "print_always") as printed,
        ):
            goal, error = _preparation_navigation_goal(overworld, map_id, ((18, 38),), allow_wild=False)

        self.assertIsNone(goal)
        self.assertIn("first encounter of this area is already resolved", str(error))
        self.assertIn("no undefeated trainer observed", str(error))
        printed.assert_called_once()
        self.assertIn("CAMPAIGN_PREPARATION_NO_LINE", printed.call_args.args[0]())
