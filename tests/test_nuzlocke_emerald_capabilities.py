import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.modes.opening import OpeningSequenceState
from modules.memory import GameState
from modules.nuzlocke.emerald_capabilities import (
    EmeraldCampaignAction,
    EmeraldCampaignObservation,
    choose_emerald_campaign_action,
    choose_emerald_observation_action,
    observation_driven_emerald_campaign,
    _observed_exit_goal,
    _observed_interaction_goal,
    _observed_local_destination_goal,
    _semantic_target_for_objective,
)
from modules.goals import ActivateTrigger, SemanticTarget, SemanticTargetKind
from modules.map_path import Direction
from modules.map_data import MapRSE
from modules.overworld import (
    MapConnectionObservation,
    OverworldObservation,
    TileObservation,
    TriggerObservation,
    WarpObservation,
)
from modules.navigation import TransitionRelevance, classify_transition_relevance
from modules.nuzlocke.emerald_confirmation import (
    EmeraldConfirmationChoice,
    EmeraldConfirmationContext,
    EmeraldConfirmationObservation,
)
from modules.nuzlocke.emerald_observation import EmeraldObservation
from modules.nuzlocke.emerald_menu_observation import (
    EmeraldMainMenuItem,
    EmeraldMenuKind,
    EmeraldMenuObservation,
    EmeraldOptionsItem,
)


class EmeraldCampaignCapabilityTests(unittest.TestCase):
    def observation(
        self, state, *, fast=False, setup=False, dialogue=False, gender_task=None, rom_owned_movement=False
    ):
        return EmeraldCampaignObservation(
            state,
            text_speed_fast=fast,
            new_game_setup_complete=setup,
            dialogue_waiting=dialogue,
            gender_task=gender_task,
            rom_owned_movement=rom_owned_movement,
        )

    def test_options_with_fast_text_speed_exits_without_opening_flow_state(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.OPTIONS_MENU, fast=True))
        self.assertIs(action, EmeraldCampaignAction.EXIT_OPTIONS)

    def test_main_menu_with_fast_text_speed_starts_new_game(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.MAIN_MENU, fast=True))
        self.assertIs(action, EmeraldCampaignAction.START_NEW_GAME)

    def test_main_menu_without_fast_text_speed_enters_options(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.MAIN_MENU))
        self.assertIs(action, EmeraldCampaignAction.ENTER_OPTIONS)

    def test_actionable_menu_precedes_rom_owned_movement_yield(self):
        action = choose_emerald_campaign_action(
            self.observation(
                OpeningSequenceState.MAIN_MENU,
                rom_owned_movement=True,
            )
        )
        self.assertIs(action, EmeraldCampaignAction.ENTER_OPTIONS)

    def test_main_menu_input_remains_actionable_when_avatar_is_uncontrollable(self):
        with (
            patch("modules.nuzlocke.emerald_capabilities.player_avatar_is_controllable", return_value=False),
            patch("modules.nuzlocke.emerald_capabilities.get_tasks", return_value=(SimpleNamespace(),)),
        ):
            action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.MAIN_MENU, fast=False))
        self.assertIs(action, EmeraldCampaignAction.ENTER_OPTIONS)

    def test_immediate_states_take_priority_over_campaign_menu_action(self):
        dialogue = choose_emerald_campaign_action(
            self.observation(OpeningSequenceState.MAIN_MENU, fast=True, dialogue=True)
        )
        gender = choose_emerald_campaign_action(
            self.observation(
                OpeningSequenceState.MAIN_MENU,
                fast=True,
                gender_task="Task_NewGameBirchSpeech_ChooseGender",
            )
        )
        self.assertIs(dialogue, EmeraldCampaignAction.ADVANCE_DIALOGUE)
        self.assertIs(gender, EmeraldCampaignAction.CHOOSE_GENDER)

    def test_rom_owned_movement_yields_before_navigation(self):
        action = choose_emerald_campaign_action(
            self.observation(OpeningSequenceState.SCRIPTED_INTRO, rom_owned_movement=True)
        )
        self.assertIs(action, EmeraldCampaignAction.WAIT)

    def test_rom_owned_movement_still_advances_actionable_dialogue(self):
        action = choose_emerald_campaign_action(
            self.observation(
                OpeningSequenceState.SCRIPTED_INTRO,
                dialogue=True,
                rom_owned_movement=True,
            )
        )
        self.assertIs(action, EmeraldCampaignAction.ADVANCE_DIALOGUE)

    def test_inactive_rom_movement_preserves_overworld_navigation(self):
        with (
            patch("modules.nuzlocke.emerald_capabilities.get_game_state", return_value=GameState.OVERWORLD),
            patch("modules.nuzlocke.emerald_capabilities.player_avatar_is_controllable", return_value=True),
        ):
            action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.SCRIPTED_INTRO))
        self.assertIs(action, EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD)

    def test_rom_owned_movement_does_not_plan_or_issue_input(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        observation = self.observation(OpeningSequenceState.SCRIPTED_INTRO, rom_owned_movement=True)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", return_value=observation),
            patch("modules.nuzlocke.emerald_capabilities.perceive_overworld") as perceive_overworld,
            patch("modules.nuzlocke.emerald_capabilities._observed_exit_goal") as observed_exit_goal,
        ):
            campaign = observation_driven_emerald_campaign()
            next(campaign)

        perceive_overworld.assert_not_called()
        observed_exit_goal.assert_not_called()
        emulator.press_button.assert_not_called()

    def test_observed_ui_interrupt_abandons_overworld_transaction(self):
        from modules.nuzlocke.emerald_capabilities import observation_driven_overworld_progression

        with patch("modules.nuzlocke.emerald_capabilities.perceive_overworld") as perceive_overworld:
            progression = observation_driven_overworld_progression(lambda: True)
            next(progression)

        perceive_overworld.assert_not_called()

    def test_map_change_eviction_closes_cached_tactical_loop(self):
        from modules.nuzlocke.emerald_capabilities import observation_driven_overworld_progression

        cached_loop = Mock()
        current = OverworldObservation((2, 0), (1, 1), Direction.East, True, (), (), (), ())
        cache = {"overworld": ((None, (1, 0)), cached_loop)}
        with (
            patch("modules.nuzlocke.emerald_capabilities.perceive_overworld", return_value=current),
            patch("modules.nuzlocke.emerald_capabilities._observed_exit_goal", return_value=None),
        ):
            next(observation_driven_overworld_progression(execution_cache=cache))

        cached_loop.close.assert_called_once_with()
        self.assertNotIn("overworld", cache)

    def test_post_gender_name_prompt_is_a_single_transition_input(self):
        action = choose_emerald_campaign_action(
            self.observation(
                OpeningSequenceState.SCRIPTED_INTRO,
                gender_task="Task_NewGameBirchSpeech_WaitPressBeforeNameChoice",
            )
        )
        self.assertIs(action, EmeraldCampaignAction.ADVANCE_NAME_PROMPT)

    def test_configured_gender_selection_issues_one_input(self):
        from modules.nuzlocke.emerald_capabilities import _gender_input

        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_gender="male")),
        )
        task = SimpleNamespace(data_value=lambda index: 0)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities.get_task", return_value=task),
        ):
            list(_gender_input("male"))
        emulator.press_button.assert_called_once_with("A")

    def test_completed_setup_does_not_start_new_game_again(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.MAIN_MENU, fast=True, setup=True))
        self.assertIs(action, EmeraldCampaignAction.WAIT)

    def test_observed_exit_preserves_one_specific_warp_with_equal_destination(self):
        from modules.map_path import Direction
        from modules.overworld import OverworldObservation, TileObservation, WarpObservation
        from modules.navigation import GoalAwareNavigator, NavigationWorld

        source = ((25, 40), (4, 1))
        destination = ((0, 0), (10, 19))
        first = WarpObservation(((25, 40), (4, 1)), destination, required_facing=Direction.East)
        second = WarpObservation(((25, 40), (4, 2)), destination, required_facing=Direction.East)
        third = WarpObservation(((25, 40), (4, 3)), destination, required_facing=Direction.East)
        world = OverworldObservation(
            source[0],
            source[1],
            Direction.East,
            True,
            tuple(
                TileObservation(((25, 40), (x, y)), False, frozenset(Direction)) for x, y in ((4, 1), (4, 2), (4, 3))
            ),
            (first, second, third),
            (),
            (),
        )
        navigator = GoalAwareNavigator(NavigationWorld.from_overworld(world))
        goal = _observed_exit_goal(world, navigator)
        self.assertIsNotNone(goal)
        self.assertIs(goal.warp, first)
        self.assertEqual(goal.destination, destination)

    def test_observed_exit_requires_controllability(self):
        from modules.overworld import OverworldObservation
        from modules.navigation import GoalAwareNavigator, NavigationWorld

        world = OverworldObservation((25, 40), (4, 1), None, False, (), (), (), ())
        navigator = GoalAwareNavigator(NavigationWorld.from_overworld(world))
        self.assertIsNone(_observed_exit_goal(world, navigator))

    def test_transition_relevance_precedes_local_cost(self):
        from modules.map_path import Direction
        from modules.overworld import OverworldObservation, TileObservation, WarpObservation

        source_map = (25, 40)
        cheap_unrelated = WarpObservation((source_map, (2, 8)), ((1, 1), (0, 0)))
        expensive_relevant = WarpObservation((source_map, (8, 1)), ((9, 9), (0, 0)))
        world = OverworldObservation(
            source_map,
            (2, 7),
            Direction.South,
            True,
            tuple(
                TileObservation((source_map, coordinate), False, frozenset(Direction))
                for coordinate in ((2, 7), (2, 8), (8, 1))
            ),
            (cheap_unrelated, expensive_relevant),
            (),
            (),
        )
        navigator = Mock()
        navigator.plan.side_effect = lambda start, goal, algorithm: SimpleNamespace(
            metrics=SimpleNamespace(
                encounter_opportunities=0,
                total_route_cost=1 if goal.warp is cheap_unrelated else 100,
                movement_actions=1,
            )
        )
        target = SemanticTarget.map((9, 9))
        with patch(
            "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
            side_effect=lambda warp, semantic_target: (
                TransitionRelevance.IRRELEVANT if warp is cheap_unrelated else TransitionRelevance.RELEVANT
            ),
        ):
            selected = _observed_exit_goal(world, navigator, target)

        self.assertIs(selected.warp, expensive_relevant)

    def test_semantic_map_target_prefers_progression_connection_over_interior_warp(self):
        """A globally relevant house warp must not beat the next boundary edge."""
        from modules.world_navigation import WorldRoute

        oldale, route102, house, petalburg = (1, 1), (1, 2), (2, 0), (1, 3)
        progression = MapConnectionObservation((oldale, (8, 0)), (route102, (0, 20)), required_facing=Direction.North)
        interior = WarpObservation((oldale, (1, 1)), (house, (3, 3)))
        world = SimpleNamespace(
            map_id=oldale,
            player_coordinates=(1, 0),
            controllable=True,
            transitions=(progression, interior),
            warps=(),
        )
        navigator = Mock()
        navigator.plan.side_effect = lambda start, goal, algorithm: SimpleNamespace(
            destination=goal.warp.entry,
            metrics=SimpleNamespace(
                encounter_opportunities=0,
                total_route_cost=20 if goal.warp is progression else 1,
                movement_actions=20 if goal.warp is progression else 1,
            ),
        )

        def downstream(transition, target, graph=None):
            destination = transition.destination[0]
            return WorldRoute((destination, petalburg), (), 100 if destination == route102 else 1)

        with (
            patch(
                "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
                return_value=TransitionRelevance.RELEVANT,
            ),
            patch("modules.nuzlocke.emerald_capabilities.transition_world_route", side_effect=downstream),
        ):
            selected = _observed_exit_goal(world, navigator, SemanticTarget.map(petalburg))

        self.assertIs(selected.warp, progression)
        self.assertEqual(selected.destination_map, route102)

    def test_semantic_map_target_allows_interior_warp_when_no_connection_exists(self):
        house, petalburg = (2, 0), (1, 3)
        interior = WarpObservation(((1, 1), (1, 1)), (house, (3, 3)))
        world = SimpleNamespace(
            map_id=(1, 1),
            player_coordinates=(1, 0),
            controllable=True,
            transitions=(interior,),
            warps=(),
        )
        navigator = Mock()
        navigator.plan.return_value = SimpleNamespace(
            destination=interior.entry,
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=1, movement_actions=1),
        )
        with (
            patch(
                "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
                return_value=TransitionRelevance.RELEVANT,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities.transition_world_route",
                return_value=SimpleNamespace(estimated_cost=1),
            ),
        ):
            selected = _observed_exit_goal(world, navigator, SemanticTarget.map(petalburg))
        self.assertIs(selected.warp, interior)

    def test_explicit_interaction_target_allows_interior_warp_with_connection(self):
        from modules.world_navigation import WorldRoute

        oldale, route102, house, target_map = (1, 1), (1, 2), (2, 0), (1, 3)
        progression = MapConnectionObservation((oldale, (8, 0)), (route102, (0, 20)), required_facing=Direction.North)
        interior = WarpObservation((oldale, (1, 1)), (house, (3, 3)))
        world = SimpleNamespace(
            map_id=oldale,
            player_coordinates=(1, 0),
            controllable=True,
            transitions=(progression, interior),
            warps=(),
        )
        navigator = Mock()
        navigator.plan.side_effect = lambda start, goal, algorithm: SimpleNamespace(
            destination=goal.warp.entry,
            metrics=SimpleNamespace(
                encounter_opportunities=0, total_route_cost=1 if goal.warp is interior else 20, movement_actions=1
            ),
        )
        with (
            patch(
                "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
                return_value=TransitionRelevance.RELEVANT,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities.transition_world_route",
                return_value=WorldRoute((house, target_map), (), 1),
            ),
        ):
            selected = _observed_exit_goal(world, navigator, SemanticTarget.interaction(house, "enter_house"))
        self.assertIs(selected.warp, interior)

    def test_recent_transition_prevents_immediate_cheap_reversal(self):
        from modules.world_navigation import WorldRoute

        current_map, previous_map, forward_map = (50, 1), (50, 0), (50, 2)
        reverse = MapConnectionObservation(
            (current_map, (1, 0)), (previous_map, (1, 4)), required_facing=Direction.North
        )
        forward = MapConnectionObservation(
            (current_map, (3, 0)), (forward_map, (3, 4)), required_facing=Direction.North
        )
        world = SimpleNamespace(
            map_id=current_map,
            player_coordinates=(2, 2),
            controllable=True,
            transitions=(reverse, forward),
            warps=(),
        )
        navigator = Mock()
        navigator.plan.side_effect = lambda start, goal, algorithm: SimpleNamespace(
            metrics=SimpleNamespace(
                encounter_opportunities=0,
                total_route_cost=1 if goal.warp is reverse else 44,
                movement_actions=1,
            )
        )

        def downstream(transition, target, graph=None):
            return WorldRoute((transition.destination[0], (0, 99)), (), 40 if transition is reverse else 20)

        with (
            patch(
                "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
                return_value=TransitionRelevance.RELEVANT,
            ),
            patch("modules.nuzlocke.emerald_capabilities.transition_world_route", side_effect=downstream),
        ):
            selected = _observed_exit_goal(
                world,
                navigator,
                SemanticTarget.map(forward_map),
                {"previous_transition": (previous_map, current_map)},
            )

        self.assertIs(selected.warp, forward)

    def test_recent_transition_does_not_permanently_forbid_required_backtracking(self):
        from modules.world_navigation import WorldRoute

        current_map, previous_map = (51, 1), (51, 0)
        reverse = MapConnectionObservation(
            (current_map, (1, 0)), (previous_map, (1, 4)), required_facing=Direction.North
        )
        world = SimpleNamespace(
            map_id=current_map,
            player_coordinates=(1, 2),
            controllable=True,
            transitions=(reverse,),
            warps=(),
        )
        navigator = Mock()
        navigator.plan.return_value = SimpleNamespace(
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=1, movement_actions=1)
        )
        with (
            patch(
                "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
                return_value=TransitionRelevance.RELEVANT,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities.transition_world_route",
                return_value=WorldRoute((previous_map,), (), 0),
            ),
        ):
            selected = _observed_exit_goal(
                world,
                navigator,
                SemanticTarget.map(previous_map),
                {"previous_transition": (previous_map, current_map)},
            )

        self.assertIs(selected.warp, reverse)

    def test_goal_route_quality_prefers_staircase_over_cheap_cyclic_exit(self):
        """The autonomous 1F boundary must choose forward world progress."""
        from modules.map_path import Direction
        from modules.overworld import OverworldObservation, TileObservation, WarpObservation
        from modules.world_navigation import WorldEdge, WorldMapGraph

        house_1f, outside, house_2f = (1, 2), (1, 1), (1, 3)
        staircase = WarpObservation((house_1f, (2, 2)), (house_2f, (2, 7)))
        door = WarpObservation((house_1f, (2, 8)), (outside, (10, 5)))
        world = OverworldObservation(
            house_1f,
            (2, 7),
            Direction.South,
            True,
            tuple(
                TileObservation((house_1f, coordinate), False, frozenset(Direction))
                for coordinate in ((2, 7), (2, 8), (2, 2))
            ),
            (staircase, door),
            (),
            (),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(house_1f, house_2f, "warp", ((2, 2),), ((2, 7),)),
                WorldEdge(house_1f, outside, "warp", ((2, 8),), ((10, 5),)),
                WorldEdge(outside, house_1f, "warp", ((10, 5),), ((2, 8),)),
            )
        )
        navigator = Mock()
        navigator.plan.side_effect = lambda start, goal, algorithm: SimpleNamespace(
            metrics=SimpleNamespace(
                encounter_opportunities=0,
                total_route_cost=5 if goal.warp is staircase else 2,
                movement_actions=5 if goal.warp is staircase else 2,
            )
        )

        with patch("modules.navigation.get_world_map_graph", return_value=graph):
            selected = _observed_exit_goal(world, navigator, SemanticTarget.interaction(house_2f, "wall-clock"))

        self.assertIs(selected.warp, staircase)

    def test_semantic_exit_selection_checks_local_reachability_before_downstream_cost(self):
        from modules.map_path import Direction
        from modules.navigation import NavigationError
        from modules.overworld import OverworldObservation, TileObservation, WarpObservation
        from modules.world_navigation import WorldRoute

        source = (25, 40)
        locally_blocked = WarpObservation((source, (2, 8)), ((1, 1), (0, 0)))
        locally_reachable = WarpObservation((source, (8, 1)), ((9, 9), (0, 0)))
        world = OverworldObservation(
            source,
            (2, 7),
            Direction.South,
            True,
            tuple(
                TileObservation((source, coordinate), False, frozenset(Direction))
                for coordinate in ((2, 7), (2, 8), (8, 1))
            ),
            (locally_blocked, locally_reachable),
            (),
            (),
        )
        navigator = Mock()

        def plan(_start, goal, algorithm):
            if goal.warp is locally_blocked:
                raise NavigationError("local route unavailable")
            return SimpleNamespace(
                metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=5, movement_actions=5)
            )

        navigator.plan.side_effect = plan
        target = SemanticTarget.map((9, 9))
        with (
            patch(
                "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
                return_value=TransitionRelevance.RELEVANT,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities.transition_world_route",
                side_effect=lambda warp, _target, _graph: WorldRoute(
                    (source,), (), 1 if warp is locally_blocked else 20
                ),
            ),
        ):
            selected = _observed_exit_goal(world, navigator, target)
        self.assertIs(selected.warp, locally_reachable)

    def test_semantic_target_with_no_known_relevant_exit_does_not_escape_unrelated(self):
        from modules.map_path import Direction
        from modules.overworld import OverworldObservation, TileObservation, WarpObservation

        source_map = (25, 40)
        warp = WarpObservation((source_map, (2, 8)), ((1, 1), (0, 0)))
        world = OverworldObservation(
            source_map,
            (2, 7),
            Direction.South,
            True,
            (
                TileObservation((source_map, (2, 7)), False, frozenset(Direction)),
                TileObservation((source_map, (2, 8)), False, frozenset(Direction)),
            ),
            (warp,),
            (),
            (),
        )
        navigator = Mock()
        with patch(
            "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
            return_value=TransitionRelevance.IRRELEVANT,
        ):
            self.assertIsNone(_observed_exit_goal(world, navigator, SemanticTarget.map((9, 9))))

    def test_semantic_exit_selection_includes_observed_map_connection(self):
        """Live campaign selection must pass a boundary transition to tactics."""
        from modules.navigation import GoalAwareNavigator
        from modules.world_navigation import WorldEdge, WorldMapGraph

        source_map, route101 = (0, 9), (0, 16)
        connection = MapConnectionObservation((source_map, (3, 3)), (route101, (3, 0)), required_facing=Direction.South)
        world = OverworldObservation(
            source_map,
            (3, 2),
            Direction.South,
            True,
            tuple(
                TileObservation((source_map, coordinate), False, frozenset(Direction))
                for coordinate in ((3, 2), (3, 3))
            ),
            (),
            (),
            (),
            transitions=(connection,),
        )
        graph = WorldMapGraph((WorldEdge(source_map, route101, "connection", ((3, 3),), ((3, 0),)),))
        navigator = Mock()
        navigator.plan.return_value = SimpleNamespace(
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=1, movement_actions=1)
        )

        with patch("modules.navigation.get_world_map_graph", return_value=graph):
            selected = _observed_exit_goal(world, navigator, SemanticTarget.map(route101))

        self.assertIsNotNone(selected)
        self.assertIs(selected.warp, connection)
        self.assertEqual(selected.warp.kind, "map_connection")

    def test_aligned_map_connection_is_one_search_and_selects_cheapest_member(self):
        from modules.navigation import GoalAwareNavigator, NavigationWorld
        from modules.overworld import WarpObservation
        from modules.world_navigation import WorldEdge, WorldMapGraph

        source_map, target_map, detour_map = (50, 0), (50, 1), (50, 2)
        connections = tuple(
            MapConnectionObservation(
                (source_map, (x, 0)),
                (target_map, (x + 2, 9)),
                required_facing=Direction.North,
            )
            for x in range(10)
        )
        ordinary_warp = WarpObservation((source_map, (0, 3)), (detour_map, (0, 0)))
        world = OverworldObservation(
            source_map,
            (7, 2),
            Direction.North,
            True,
            tuple(TileObservation((source_map, (x, y)), False, frozenset(Direction)) for x, y in ((7, 1), (7, 2))),
            (ordinary_warp,),
            (),
            (),
            transitions=(ordinary_warp,) + connections,
        )
        navigator = GoalAwareNavigator(NavigationWorld.from_overworld(world))
        original_plan = navigator.plan
        calls = []

        def counted_plan(start, goal, *, algorithm):
            calls.append(goal)
            return original_plan(start, goal, algorithm=algorithm)

        navigator.plan = counted_plan
        graph = WorldMapGraph(
            (
                WorldEdge(
                    source_map,
                    target_map,
                    "connection",
                    tuple(connection.entry[1] for connection in connections),
                    tuple(connection.destination[1] for connection in connections),
                    estimated_cost=1,
                ),
                WorldEdge(source_map, detour_map, "warp", ((0, 3),), ((0, 0),), estimated_cost=50),
                WorldEdge(detour_map, target_map, "warp", ((0, 0),), ((0, 0),), estimated_cost=50),
            )
        )
        with patch("modules.navigation.get_world_map_graph", return_value=graph):
            selected = _observed_exit_goal(world, navigator, SemanticTarget.map(target_map))

        grouped_calls = [goal for goal in calls if goal.warps]
        ordinary_calls = [goal for goal in calls if not goal.warps]
        self.assertEqual(len(grouped_calls), 1)
        self.assertEqual(len(grouped_calls[0].warps), 10)
        self.assertEqual(len(ordinary_calls), 1)
        self.assertIs(ordinary_calls[0].warp, ordinary_warp)
        self.assertIs(selected.warp, connections[7])

    def test_meet_rival_print_tiles(self):
        from modules.navigation import NavigationWorld

        # This test is just for debugging
        # In a real environment, we would use the emulator context
        # But here we can mock the world or inspect the environment
        print("DEBUG: Inspecting tiles around (5,9)")
        # We need to get the world. The test has access to world if I pass it?
        # Actually I can just construct the world from the map data
        # But this is hard.
        pass

        metadata = SimpleNamespace(
            objects=(SimpleNamespace(script_symbol="LittlerootTown_MaysHouse_2F_EventScript_RivalsPokeBall"),),
            bg_events=(),
        )
        with (
            patch(
                "modules.nuzlocke.emerald_capabilities._player_house_map",
                return_value=MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities._rival_house_map",
                return_value=MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
            ),
            patch("modules.nuzlocke.emerald_capabilities._opening_gender", return_value="female"),
            patch("modules.nuzlocke.emerald_capabilities.get_map_metadata", return_value=metadata),
        ):
            target = _semantic_target_for_objective("meet_rival")
        self.assertEqual(target.kind, SemanticTargetKind.INTERACTION)
        self.assertEqual(target.target_map, MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value)
        self.assertEqual(target.interaction_id, "LittlerootTown_MaysHouse_2F_EventScript_RivalsPokeBall")

    def test_rescue_birch_targets_route_101_birch_bag_interaction(self):
        target = _semantic_target_for_objective("rescue_birch")
        self.assertEqual(target.kind, SemanticTargetKind.INTERACTION)
        self.assertEqual(target.target_map, MapRSE.ROUTE101.value)
        self.assertEqual(target.interaction_id, "Route101_EventScript_BirchsBag")

    def test_receive_pokedex_targets_birch_lab_map(self):
        target = _semantic_target_for_objective("receive_pokedex")
        self.assertEqual(target.kind, SemanticTargetKind.MAP)
        self.assertEqual(
            target.target_map,
            MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value,
        )
        self.assertIsNone(target.interaction_id)

    def test_receive_pokedex_world_route_returns_through_littleroot(self):
        from modules.world_navigation import WorldEdge, WorldMapGraph

        route103 = MapRSE.ROUTE103.value
        oldale = MapRSE.OLDALE_TOWN.value
        route101 = MapRSE.ROUTE101.value
        littleroot = MapRSE.LITTLEROOT_TOWN.value
        lab = MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value
        graph = WorldMapGraph(
            (
                WorldEdge(route103, oldale, "connection", ((9, 21),), ((9, 0),), estimated_cost=22),
                WorldEdge(oldale, route101, "connection", ((9, 19),), ((9, 0),), estimated_cost=20),
                WorldEdge(route101, littleroot, "connection", ((9, 28),), ((9, 0),), estimated_cost=30),
                WorldEdge(littleroot, lab, "warp", ((10, 8),), ((6, 7),), estimated_cost=10),
            )
        )

        self.assertEqual(
            graph.route(route103, lab).maps,
            (route103, oldale, route101, littleroot, lab),
        )

    def test_receive_pokedex_littleroot_local_target_selects_lab_warp_only(self):
        littleroot = MapRSE.LITTLEROOT_TOWN.value
        route101 = MapRSE.ROUTE101.value
        lab = MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value
        route101_exit = MapConnectionObservation(
            (littleroot, (10, 1)), (route101, (10, 28)), required_facing=Direction.North
        )
        lab_entry = WarpObservation((littleroot, (10, 8)), (lab, (6, 7)))
        world = OverworldObservation(
            littleroot,
            (10, 4),
            Direction.South,
            True,
            tuple(
                TileObservation((littleroot, coordinate), False, frozenset(Direction))
                for coordinate in ((10, 1), (10, 4), (10, 8))
            ),
            (lab_entry,),
            (),
            (),
            transitions=(route101_exit, lab_entry),
        )
        navigator = Mock()
        navigator.plan.return_value = SimpleNamespace(
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=4, movement_actions=4)
        )

        selected = _observed_local_destination_goal(world, navigator, lab)

        self.assertIsNotNone(selected)
        self.assertIs(selected.warp, lab_entry)
        self.assertEqual(selected.destination_map, lab)
        self.assertEqual(navigator.plan.call_count, 1)

    def test_receive_pokedex_littleroot_local_target_does_not_use_route101_fallback(self):
        littleroot = MapRSE.LITTLEROOT_TOWN.value
        route101 = MapRSE.ROUTE101.value
        route101_exit = MapConnectionObservation(
            (littleroot, (10, 1)), (route101, (10, 28)), required_facing=Direction.North
        )
        world = OverworldObservation(
            littleroot,
            (10, 4),
            Direction.South,
            True,
            (TileObservation((littleroot, (10, 4)), False, frozenset(Direction)),),
            (),
            (),
            (),
            transitions=(route101_exit,),
        )

        self.assertIsNone(
            _observed_local_destination_goal(world, Mock(), MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value)
        )

    def test_receive_pokedex_waits_for_rom_owned_lab_event_after_entry(self):
        action = choose_emerald_observation_action(
            EmeraldObservation(
                game_state=GameState.OVERWORLD,
                ui_mode="OVERWORLD",
                title_actionable=False,
                startup_tasks=(),
                native_task_state=(None, None, None),
                script_stack=(),
                map_id=MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value,
                coordinates=(6, 7),
                controllable=False,
                dialogue_actionable=False,
                dialogue_lifecycle_active=False,
                menu=None,
                naming=None,
                confirmation=None,
                starter_selection=None,
                gender_task=None,
                rom_owned_movement=True,
                clock_interaction=None,
                clock_target=None,
                campaign_facts=(),
                overworld=None,
                objective_id="receive_pokedex",
                semantic_target=_semantic_target_for_objective("receive_pokedex"),
            )
        )
        self.assertIs(action, EmeraldCampaignAction.WAIT)

    def test_lab_target_prevents_oldale_return_to_route103(self):
        from modules.world_navigation import WorldEdge, WorldMapGraph

        route103 = MapRSE.ROUTE103.value
        oldale = MapRSE.OLDALE_TOWN.value
        route101 = MapRSE.ROUTE101.value
        littleroot = MapRSE.LITTLEROOT_TOWN.value
        lab = MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value
        north = MapConnectionObservation((oldale, (9, 0)), (route103, (9, 21)), required_facing=Direction.North)
        south = MapConnectionObservation((oldale, (9, 19)), (route101, (9, 0)), required_facing=Direction.South)
        world = OverworldObservation(
            oldale,
            (9, 0),
            Direction.South,
            True,
            (TileObservation((oldale, (9, 0)), False, frozenset(Direction)),),
            (),
            (),
            (),
            transitions=(north, south),
        )
        graph = WorldMapGraph(
            (
                WorldEdge(oldale, route103, "connection", ((9, 0),), ((9, 21),), estimated_cost=22),
                WorldEdge(route103, oldale, "connection", ((9, 21),), ((9, 0),), estimated_cost=22),
                WorldEdge(oldale, route101, "connection", ((9, 19),), ((9, 0),), estimated_cost=20),
                WorldEdge(route101, littleroot, "connection", ((9, 28),), ((9, 0),), estimated_cost=30),
                WorldEdge(littleroot, lab, "warp", ((10, 8),), ((6, 7),), estimated_cost=10),
            )
        )
        navigator = Mock()
        navigator.plan.side_effect = lambda _start, goal, algorithm: SimpleNamespace(
            destination=goal.warp.entry,
            metrics=SimpleNamespace(
                encounter_opportunities=0,
                total_route_cost=0 if goal.warp is north else 19,
                movement_actions=0 if goal.warp is north else 19,
            ),
        )

        with patch("modules.navigation.get_world_map_graph", return_value=graph):
            selected = _observed_exit_goal(world, navigator, _semantic_target_for_objective("receive_pokedex"))

        self.assertIs(selected.warp, south)

    def test_lab_target_selects_oldale_from_route103(self):
        route103 = MapRSE.ROUTE103.value
        oldale = MapRSE.OLDALE_TOWN.value
        connection = MapConnectionObservation((route103, (9, 21)), (oldale, (9, 0)), required_facing=Direction.South)
        world = OverworldObservation(
            route103,
            (9, 20),
            Direction.South,
            True,
            (TileObservation((route103, (9, 20)), False, frozenset(Direction)),),
            (),
            (),
            (),
            transitions=(connection,),
        )
        navigator = Mock()
        navigator.plan.return_value = SimpleNamespace(
            destination=connection.entry,
            metrics=SimpleNamespace(encounter_opportunities=0, total_route_cost=0, movement_actions=0),
        )
        with (
            patch(
                "modules.nuzlocke.emerald_capabilities.classify_transition_relevance",
                return_value=TransitionRelevance.RELEVANT,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities.transition_world_route",
                return_value=SimpleNamespace(estimated_cost=80),
            ),
        ):
            selected = _observed_exit_goal(world, navigator, _semantic_target_for_objective("receive_pokedex"))

        self.assertIs(selected.warp, connection)

    def test_unrelated_targetless_objective_remains_targetless(self):
        self.assertIsNone(_semantic_target_for_objective("obtain_starter"))

    def test_rescue_birch_bag_target_uses_existing_affordance_and_interaction_planning(self):
        from modules.navigation import GoalAwareNavigator, NavigationWorld

        map_id = MapRSE.ROUTE101.value
        bag_location = (map_id, (7, 14))
        activation_location = (map_id, (7, 15))
        trigger = TriggerObservation(
            "object:3:Route101_EventScript_BirchsBag",
            frozenset({bag_location}),
            frozenset({activation_location}),
            kind="object_interaction",
            script_symbol="Route101_EventScript_BirchsBag",
            affordance_id="Route101_EventScript_BirchsBag",
            activation_requirements=((activation_location, Direction.North),),
        )
        world = OverworldObservation(
            map_id,
            (7, 16),
            Direction.North,
            True,
            tuple(
                TileObservation((map_id, coordinate), False, frozenset(Direction))
                for coordinate in ((7, 14), (7, 15), (7, 16))
            ),
            (),
            (),
            (trigger,),
        )

        target = _semantic_target_for_objective("rescue_birch")
        goal = _observed_interaction_goal(world, target)
        plan = GoalAwareNavigator(NavigationWorld.from_overworld(world)).plan(
            (map_id, (7, 16)),
            goal,
        )

        self.assertEqual(goal, ActivateTrigger(trigger.trigger_id))
        self.assertEqual(plan.destination, activation_location)
        self.assertTrue(
            GoalAwareNavigator(NavigationWorld.from_overworld(world)).satisfies(
                plan.destination,
                Direction.North,
                goal,
            )
        )
        self.assertFalse(
            GoalAwareNavigator(NavigationWorld.from_overworld(world)).satisfies(
                plan.destination,
                Direction.South,
                goal,
            )
        )

    def test_wall_clock_campaign_target_is_a_world_interaction_target(self):
        house = SimpleNamespace(value=(1, 3))
        with patch("modules.nuzlocke.emerald_capabilities._player_house_map", return_value=house):
            target = _semantic_target_for_objective("set_wall_clock")

        self.assertEqual(target.target_map, (1, 3))
        self.assertEqual(target.interaction_id, "wall_clock")

    def test_current_map_semantic_interaction_resolves_to_observed_affordance(self):
        map_id = (1, 3)
        trigger = TriggerObservation(
            "bg:clock",
            frozenset({(map_id, (2, 2))}),
            frozenset({(map_id, (2, 3))}),
            kind="bg_script",
            script_symbol="MaysHouse_EventScript_WallClock",
            affordance_id="MaysHouse_EventScript_WallClock",
            activation_requirements=(((map_id, (2, 3)), Direction.North),),
        )
        world = OverworldObservation(
            map_id,
            (2, 3),
            Direction.North,
            True,
            tuple(
                TileObservation((map_id, coordinate), False, frozenset(Direction)) for coordinate in ((2, 2), (2, 3))
            ),
            (),
            (),
            (trigger,),
        )

        goal = _observed_interaction_goal(
            world,
            SemanticTarget.interaction(map_id, "MaysHouse_EventScript_WallClock"),
        )

        self.assertEqual(goal, ActivateTrigger("bg:clock"))

        hidden = replace(trigger, condition_active=False)
        self.assertIsNone(
            _observed_interaction_goal(
                replace(world, triggers=(hidden,)),
                SemanticTarget.interaction(map_id, "MaysHouse_EventScript_WallClock"),
            )
        )

    def test_transition_classifier_exposes_relevance_and_blocking(self):
        from modules.overworld import WarpObservation
        from modules.world_navigation import WorldMapGraph, WorldEdge

        graph = WorldMapGraph((WorldEdge((1, 1), (9, 9), "warp", ((0, 0),), ((0, 0),)),))
        target = SemanticTarget.map((9, 9))
        relevant = WarpObservation(((0, 0), (0, 0)), ((1, 1), (0, 0)))
        unrelated = WarpObservation(((0, 0), (1, 0)), ((4, 4), (0, 0)))
        self.assertIs(
            classify_transition_relevance(relevant, target, graph),
            TransitionRelevance.RELEVANT,
        )
        self.assertIs(
            classify_transition_relevance(unrelated, target, graph),
            TransitionRelevance.IRRELEVANT,
        )
        self.assertIs(
            classify_transition_relevance(relevant, target, graph, currently_reachable=False),
            TransitionRelevance.BLOCKED,
        )
        self.assertIs(
            classify_transition_relevance(relevant, None, graph),
            TransitionRelevance.UNKNOWN,
        )


class EmeraldCampaignLifecycleTests(unittest.TestCase):
    def observation(
        self,
        state,
        *,
        fast=False,
        setup=False,
        dialogue=False,
        cursor=None,
        menu=None,
        option_value=None,
        gender_task=None,
    ):
        if menu is None and state is OpeningSequenceState.MAIN_MENU and cursor is not None:
            items = (EmeraldMainMenuItem.NEW_GAME, EmeraldMainMenuItem.OPTIONS)
            menu = EmeraldMenuObservation(
                EmeraldMenuKind.MAIN_MENU,
                cursor,
                items[cursor] if 0 <= cursor < len(items) else None,
                True,
                "Task_HandleMainMenuInput",
                cursor,
                0,
            )
        if menu is None and state is OpeningSequenceState.OPTIONS_MENU:
            row = 0 if cursor is None else cursor
            value = 2 if fast else (1 if option_value is None else option_value)
            menu = EmeraldMenuObservation(
                EmeraldMenuKind.OPTIONS_MENU,
                row,
                tuple(EmeraldOptionsItem)[row] if 0 <= row < len(EmeraldOptionsItem) else None,
                True,
                "Task_OptionMenuProcessInput",
                row,
                None,
                (value, 0, 0, 0, 0, 0),
            )
        return EmeraldCampaignObservation(
            state,
            fast,
            setup,
            dialogue,
            gender_task,
            cursor,
            False,
            menu,
        )

    def run_generator(self, observations, *, task_active=True, dialogue_step=None):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        generator = None
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=observations),
            patch("modules.nuzlocke.emerald_capabilities._advance_scripted_input", side_effect=dialogue_step),
        ):
            generator = observation_driven_emerald_campaign()
            for _ in observations:
                next(generator)
        return emulator.press_button

    def test_main_menu_options_is_two_observed_transactions(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=1),
                self.observation(OpeningSequenceState.OPTIONS_MENU),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Down", "A", "Right"])

    def test_options_changes_speed_reobserves_fact_then_exits(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.OPTIONS_MENU),
                self.observation(OpeningSequenceState.OPTIONS_MENU),
                self.observation(OpeningSequenceState.OPTIONS_MENU, fast=True),
                self.observation(OpeningSequenceState.MAIN_MENU, fast=True, setup=True, cursor=1),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Right", "B"])

    def test_options_moves_toward_text_speed_one_row_per_observation(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=3),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=2),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=1),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=0),
                self.observation(OpeningSequenceState.OPTIONS_MENU, cursor=0, fast=True),
            ]
        )
        self.assertEqual(
            [call.args[0] for call in presses.call_args_list],
            ["Up", "Up", "Up", "Right", "B"],
        )

    def test_options_exit_is_not_followed_by_reopening_options(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.OPTIONS_MENU, fast=True),
                self.observation(OpeningSequenceState.MAIN_MENU, fast=True, cursor=1),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["B", "Up"])
        self.assertNotIn("Down", [call.args[0] for call in presses.call_args_list])

    def test_dialogue_input_is_one_transaction_even_when_uncontrollable(self):
        # The executor only relies on the observation's dialogue predicate;
        # the legacy helper remains responsible for the native wait details.
        with patch(
            "modules.nuzlocke.emerald_capabilities._advance_scripted_input",
            return_value=iter([None]),
        ) as advance_input:
            emulator = SimpleNamespace(press_button=Mock())
            fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
            with (
                patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
                patch(
                    "modules.nuzlocke.emerald_capabilities._campaign_observation",
                    side_effect=[
                        self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                        self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
                    ],
                ),
            ):
                generator = observation_driven_emerald_campaign()
                next(generator)
                next(generator)
        advance_input.assert_called_once()

    def test_overworld_dialogue_wait_precedes_controllable_navigation(self):
        """A Mom-style native message wait owns the frame, despite control being returned."""
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        observations = [
            self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
            self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
        ]

        def advance_dialogue():
            fake_context.emulator.press_button("A")
            yield

        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._campaign_observation",
                side_effect=observations,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities.get_game_state",
                return_value=GameState.OVERWORLD,
            ),
            patch(
                "modules.nuzlocke.emerald_capabilities.player_avatar_is_controllable",
                return_value=True,
            ),
            patch("modules.nuzlocke.emerald_capabilities.perceive_overworld") as perceive_overworld,
            patch(
                "modules.nuzlocke.emerald_capabilities._advance_scripted_input",
                side_effect=lambda: advance_dialogue(),
            ) as advance_input,
        ):
            campaign = observation_driven_emerald_campaign()
            next(campaign)
            next(campaign)

        self.assertEqual([call.args[0] for call in emulator.press_button.call_args_list], ["A"])
        advance_input.assert_called_once()
        perceive_overworld.assert_not_called()

    def test_dialogue_transition_does_not_repeat_input(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
            ],
            dialogue_step=iter([iter([None])]),
        )
        self.assertEqual(presses.call_count, 0)

    def test_next_dialogue_message_is_reobserved_and_advanced_once(self):
        advance = Mock(return_value=iter([None]))
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._campaign_observation",
                side_effect=[
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                    self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                ],
            ),
            patch("modules.nuzlocke.emerald_capabilities._advance_scripted_input", advance),
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
            next(generator)
            next(generator)
        self.assertEqual(advance.call_count, 2)

    def test_dialogue_end_is_not_treated_as_dialogue(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=True),
                self.observation(OpeningSequenceState.SCRIPTED_INTRO, dialogue=False),
            ],
            dialogue_step=iter([iter([None])]),
        )
        self.assertEqual(presses.call_count, 0)

    def test_failed_menu_transition_does_not_continue_to_a_without_cursor_observation(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Down"])

    def test_unexpected_menu_transition_recalculates_from_new_observation(self):
        presses = self.run_generator(
            [
                self.observation(OpeningSequenceState.MAIN_MENU, cursor=0),
                self.observation(
                    OpeningSequenceState.MAIN_MENU,
                    cursor=2,
                    menu=EmeraldMenuObservation(
                        EmeraldMenuKind.MAIN_MENU,
                        2,
                        EmeraldMainMenuItem.OPTIONS,
                        True,
                        "Task_HandleMainMenuInput",
                        2,
                        1,
                    ),
                ),
                self.observation(
                    OpeningSequenceState.MAIN_MENU,
                    cursor=1,
                    menu=EmeraldMenuObservation(
                        EmeraldMenuKind.MAIN_MENU,
                        1,
                        EmeraldMainMenuItem.NEW_GAME,
                        True,
                        "Task_HandleMainMenuInput",
                        1,
                        1,
                    ),
                ),
            ]
        )
        self.assertEqual([call.args[0] for call in presses.call_args_list], ["Down", "A", "Down"])

    def test_native_standard_message_wait_is_observed_without_controllability(self):
        from modules.nuzlocke.emerald_capabilities import _campaign_observation

        facts = SimpleNamespace(
            text_speed_fast=SimpleNamespace(value=False),
            new_game_setup_complete=SimpleNamespace(value=False),
        )
        campaign_state = SimpleNamespace(campaign_facts=facts)
        native_message_wait = (False, None, True, "WaitForAorBPress", "Std_MsgboxDefault", True, False, False)
        with (
            patch("modules.nuzlocke.campaign_controller.runtime_campaign_state", return_value=campaign_state),
            patch(
                "modules.nuzlocke.emerald_capabilities.EmeraldOpeningCapability._dialogue_state_snapshot",
                return_value=native_message_wait,
            ),
            patch("modules.nuzlocke.emerald_capabilities.get_game_state", return_value=GameState.OVERWORLD),
            patch(
                "modules.nuzlocke.emerald_capabilities.get_opening_sequence_state",
                return_value=OpeningSequenceState.SCRIPTED_INTRO,
            ),
            patch("modules.nuzlocke.emerald_capabilities._active_gender_task", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities._main_menu_cursor", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities.is_field_message_waiting_for_input", return_value=False),
        ):
            observation = _campaign_observation()
        self.assertTrue(observation.dialogue_waiting)

    def test_birch_speech_printer_pause_is_observed_outside_overworld(self):
        from modules.nuzlocke.emerald_capabilities import _campaign_observation

        facts = SimpleNamespace(
            text_speed_fast=SimpleNamespace(value=False),
            new_game_setup_complete=SimpleNamespace(value=False),
        )
        campaign_state = SimpleNamespace(campaign_facts=facts)
        birch_message = (
            True,
            0,
            False,
            None,
            None,
            True,
            False,
            True,
            "Pause",
            ("Task_NewGameBirchSpeech_ThisIsAPokemon",),
            None,
            None,
            None,
            None,
        )
        with (
            patch("modules.nuzlocke.campaign_controller.runtime_campaign_state", return_value=campaign_state),
            patch(
                "modules.nuzlocke.emerald_capabilities.EmeraldOpeningCapability._dialogue_state_snapshot",
                return_value=birch_message,
            ),
            patch("modules.nuzlocke.emerald_capabilities.get_game_state", return_value=GameState.UNKNOWN),
            patch(
                "modules.nuzlocke.emerald_capabilities.get_opening_sequence_state",
                return_value=OpeningSequenceState.SCRIPTED_INTRO,
            ),
            patch("modules.nuzlocke.emerald_capabilities._active_gender_task", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities.observe_emerald_menu", return_value=None),
            patch("modules.nuzlocke.emerald_capabilities.is_emerald_field_dialogue_advanceable", return_value=True),
        ):
            observation = _campaign_observation()
        self.assertTrue(observation.dialogue_waiting)
        self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.ADVANCE_DIALOGUE)

    def test_birch_speech_runtime_shape_issues_one_input_then_yields(self):
        from modules.modes.opening import _advance_scripted_input, EmeraldOpeningCapability

        emulator = SimpleNamespace(press_button=Mock())
        opening_context = SimpleNamespace(
            rom=SimpleNamespace(is_emerald=True),
            emulator=emulator,
            debug=False,
        )
        birch_message = (
            True,
            0,
            False,
            None,
            None,
            True,
            False,
            True,
            "Pause",
            ("Task_NewGameBirchSpeech_ThisIsAPokemon",),
            None,
            None,
            None,
            None,
        )
        with (
            patch("modules.modes.opening.context", opening_context),
            patch.object(EmeraldOpeningCapability, "_dialogue_state_snapshot", return_value=birch_message),
            patch("modules.modes.opening.is_emerald_field_dialogue_advanceable", return_value=True),
        ):
            list(_advance_scripted_input())
        emulator.press_button.assert_called_once_with("A")

    def test_name_prompt_transition_issues_one_input_and_reobserves(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(emulator=emulator, config=SimpleNamespace())
        prompt = self.observation(
            OpeningSequenceState.SCRIPTED_INTRO,
            gender_task="Task_NewGameBirchSpeech_WaitPressBeforeNameChoice",
        )
        naming = self.observation(OpeningSequenceState.PLAYER_NAMING)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=[prompt, naming]),
            patch("modules.nuzlocke.emerald_capabilities._advance_name_prompt", return_value=iter([None])) as advance,
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
        advance.assert_called_once()

    def test_naming_screen_is_recognized_without_entering_name(self):
        action = choose_emerald_campaign_action(self.observation(OpeningSequenceState.PLAYER_NAMING))
        self.assertIs(action, EmeraldCampaignAction.WAIT)

    def test_player_naming_target_selects_reusable_name_capability(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        observation = self.observation(OpeningSequenceState.PLAYER_NAMING)
        observation = replace(
            observation,
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, True),
        )
        self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.ENTER_NAME)

    def test_naming_capability_waits_until_keyboard_is_ready(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        observation = self.observation(OpeningSequenceState.PLAYER_NAMING)
        observation = replace(
            observation,
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, False),
        )
        self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.WAIT)

    def test_player_confirmation_yes_and_no_are_observed_actions(self):
        for choice, expected in (
            (EmeraldConfirmationChoice.YES, EmeraldCampaignAction.CONFIRM_PLAYER_NAME),
            (EmeraldConfirmationChoice.NO, EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES),
        ):
            confirmation = EmeraldConfirmationObservation(
                True,
                choice,
                True,
                EmeraldConfirmationContext.PLAYER_NAME,
            )
            observation = replace(
                self.observation(OpeningSequenceState.SCRIPTED_INTRO),
                confirmation_observation=confirmation,
            )
            self.assertIs(choose_emerald_campaign_action(observation), expected)

    def test_unready_or_unknown_selected_confirmation_waits(self):
        for confirmation in (
            EmeraldConfirmationObservation(
                True,
                EmeraldConfirmationChoice.NO,
                False,
                EmeraldConfirmationContext.PLAYER_NAME,
            ),
            EmeraldConfirmationObservation(
                True,
                None,
                True,
                EmeraldConfirmationContext.PLAYER_NAME,
            ),
        ):
            observation = replace(
                self.observation(OpeningSequenceState.SCRIPTED_INTRO),
                confirmation_observation=confirmation,
            )
            self.assertIs(choose_emerald_campaign_action(observation), EmeraldCampaignAction.WAIT)

    def test_naming_capability_reobserves_from_transition_to_ready_keyboard(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        not_ready = replace(
            self.observation(OpeningSequenceState.PLAYER_NAMING),
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, False),
        )
        ready = replace(
            not_ready,
            naming_observation=EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 0x2010000, True),
        )
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=[not_ready, ready]),
            patch("modules.nuzlocke.emerald_capabilities._naming_input") as naming_input,
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
        naming_input.assert_called_once_with("ASH")

    def test_naming_input_reobserves_before_repeating(self):
        from modules.nuzlocke.emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget

        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        naming = EmeraldNamingObservation(EmeraldNamingTarget.PLAYER_NAME, 0, 1, True)
        first = self.observation(OpeningSequenceState.PLAYER_NAMING)
        first = replace(first, naming_observation=naming)
        same = replace(first, naming_observation=naming)
        ended = self.observation(OpeningSequenceState.SCRIPTED_INTRO)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch("modules.nuzlocke.emerald_capabilities._campaign_observation", side_effect=[first, same, ended]),
            patch("modules.nuzlocke.emerald_capabilities._naming_input") as naming_input,
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
            next(generator)
        naming_input.assert_called_once_with("ASH")

    def test_no_to_yes_then_confirm_uses_one_input_per_observation(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        no = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.NO,
            True,
            EmeraldConfirmationContext.PLAYER_NAME,
        )
        yes = replace(no, selected=EmeraldConfirmationChoice.YES)
        first = replace(self.observation(OpeningSequenceState.SCRIPTED_INTRO), confirmation_observation=no)
        second = replace(first, confirmation_observation=yes)
        exited = self.observation(OpeningSequenceState.SCRIPTED_INTRO)
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._campaign_observation",
                side_effect=[first, second, exited],
            ),
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            self.assertEqual(emulator.press_button.call_args_list, [unittest.mock.call("Up")])
            next(generator)
            self.assertEqual(
                emulator.press_button.call_args_list,
                [unittest.mock.call("Up"), unittest.mock.call("A")],
            )
            next(generator)
        self.assertEqual(emulator.press_button.call_count, 2)

    def test_confirmation_exit_accepts_real_birch_post_name_task(self):
        emulator = SimpleNamespace(press_button=Mock())
        fake_context = SimpleNamespace(
            emulator=emulator,
            config=SimpleNamespace(start_game=SimpleNamespace(player_name="ASH", player_gender="male")),
        )
        confirmation = EmeraldConfirmationObservation(
            True,
            EmeraldConfirmationChoice.YES,
            True,
            EmeraldConfirmationContext.PLAYER_NAME,
        )
        before = replace(
            self.observation(OpeningSequenceState.MAIN_MENU),
            confirmation_observation=confirmation,
        )
        after = self.observation(
            OpeningSequenceState.MAIN_MENU,
            gender_task="Task_NewGameBirchSpeech_SlidePlatformAway2",
        )
        with (
            patch("modules.nuzlocke.emerald_capabilities.context", fake_context),
            patch(
                "modules.nuzlocke.emerald_capabilities._campaign_observation",
                side_effect=[before, after],
            ),
        ):
            generator = observation_driven_emerald_campaign()
            next(generator)
            next(generator)
        emulator.press_button.assert_called_once_with("A")

    def test_unexpected_post_gender_state_waits_for_reobservation(self):
        action = choose_emerald_campaign_action(
            self.observation(OpeningSequenceState.SCRIPTED_INTRO, gender_task=None, dialogue=False)
        )
        self.assertIs(action, EmeraldCampaignAction.WAIT)


if __name__ == "__main__":
    unittest.main()
