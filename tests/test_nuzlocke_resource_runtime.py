import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.goals import ActivateTrigger, NavigationGoal, ReachLocation, ReachWarp, TrainerMode
from modules.nuzlocke.resource_policy import (
    HealingResource,
    PartyResource,
    ResourceDecision,
    ResourceSnapshot,
    RouteRecovery,
)
from modules.nuzlocke.resource_runtime import (
    CampaignCapability,
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
from modules.nuzlocke.resource_runtime import _wait_for_center_interior
from modules.nuzlocke.resource_runtime import execute_planned_recovery
from modules.agent_control import AgentAction, AgentActionType, ActionDecision
from modules.agent_control import AgentObservation
from modules.interaction_state import InteractionObservation
from modules.memory import GameState
from modules.map_data import MapRSE, PokemonCenter
from modules.map_path import Direction
from modules.modes.util.pokecenter_loop import PokecenterLoopController
from modules.overworld import MovementState, OverworldObservation, TileObservation
from modules.nuzlocke.resource_runtime import _encounter_roll_direction


class CampaignCapabilityTests(unittest.TestCase):
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

    def test_encounter_roll_prefers_adjacent_encounter_terrain(self):
        overworld = self._encounter_world(tiles=((1, 1), (2, 1)))

        direction, reason = _encounter_roll_direction(overworld)

        self.assertEqual(direction, Direction.East)
        self.assertEqual(reason, "move_within_encounter_terrain")

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
        ), patch(
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
            emulator.press_direction.assert_called_once_with("Right", run=False, fresh=True)
            with self.assertRaises(StopIteration):
                next(execution)

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
        emulator = SimpleNamespace(press_button=Mock())
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
            patch.object(runtime, "context", SimpleNamespace(emulator=emulator)),
        ):
            list(runtime._use_field_recovery_item(item))

        use_item.assert_called_once_with(item, wait_for_start_menu_to_reappear=False)
        scroll.assert_called_once_with(0)

    def test_pc_withdrawal_prefers_a_status_cure_for_the_current_party(self):
        from modules.nuzlocke import resource_runtime as runtime

        item = HealingResource("Antidote", 2, 0, location="pc")
        snapshot = ResourceSnapshot(
            party=(PartyResource(12, 20, "poisoned"),),
            pc_healing_items=(item, HealingResource("Potion", 2, 20, location="pc")),
        )
        with (
            patch.object(runtime, "observe_resource_snapshot", return_value=snapshot),
            patch.object(runtime, "get_item_by_name", return_value=item),
            patch.object(runtime, "interact_with_pc", return_value=iter(("withdraw",))) as interact,
        ):
            self.assertEqual(list(runtime.withdraw_best_pc_healing_item()), ["withdraw"])

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
            return_value=((0, 18), ((1, 1),)),
        ), patch("modules.nuzlocke.resource_runtime.AgentControlLoop", return_value=FakeNavigationLoop()), patch(
            "modules.nuzlocke.resource_runtime.observe_agent",
            return_value=SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 18), player_coordinates=(1, 1))),
        ), patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld",
            return_value=object(),
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation",
            return_value=(object(), None),
        ), patch(
            "modules.modes.util.pokecenter_loop.PokecenterLoopController", return_value=pokecenter_loop
        ):
            self.assertEqual(list(execute_campaign_preparation((0, 18), target_level=14)), [])

        pokecenter_loop.verify_on_start.assert_called_once_with()
        pokecenter_loop.on_battle_ended.assert_called_once_with()

    def test_preparation_can_select_a_reachable_trainer_when_grass_is_blocked(self):
        from modules.goals import EngageTrainer, ReachLocation, TrainerMode
        from modules.navigation import NavigationError
        from modules.overworld import ObjectObservation

        training_map = (0, 18)
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
            if isinstance(goal, ReachLocation):
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

    def test_preparation_requires_trainers_after_area_encounter_is_resolved(self):
        from modules.goals import EngageTrainer, NavigationGoal
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
                if isinstance(self.goal, NavigationGoal) and isinstance(self.goal.target, EngageTrainer):
                    party[0].level = 14
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
            "modules.modes.util.pokecenter_loop.PokecenterLoopController",
            side_effect=AssertionError("consumed encounter must not re-enter grass loop"),
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
            SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 18), player_coordinates=(1, 1))),
            SimpleNamespace(overworld=SimpleNamespace(map_id=(0, 18), player_coordinates=(2, 2))),
        ]
        loops = iter((FakeNavigationLoop(failure=False), FakeNavigationLoop(failure=True), FakeNavigationLoop()))
        with patch("modules.nuzlocke.resource_runtime.get_party", return_value=party), patch(
            "modules.nuzlocke.resource_runtime._preparation_training_location",
            return_value=((0, 18), ((3, 3),)),
        ), patch(
            "modules.nuzlocke.resource_runtime.AgentControlLoop", side_effect=lambda *_args, **_kwargs: next(loops)
        ), patch(
            "modules.nuzlocke.resource_runtime.observe_agent", side_effect=observations
        ) as observe, patch(
            "modules.nuzlocke.resource_runtime.NavigationWorld.from_overworld", return_value=object()
        ), patch(
            "modules.nuzlocke.resource_runtime.plan_with_world_navigation", return_value=(object(), None)
        ), patch(
            "modules.modes.util.pokecenter_loop.PokecenterLoopController", return_value=FakePokecenterLoop()
        ):
            self.assertEqual(list(execute_campaign_preparation((0, 18), target_level=14)), [])

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


if __name__ == "__main__":
    unittest.main()
