from unittest import TestCase
from types import SimpleNamespace
from unittest.mock import patch

from modules.goals import ActivateTrigger, GoalConstraints, NavigationGoal, ReachLocation, ReachWarp
from modules.interaction_state import InteractionObservation, InteractionType, classify_interaction
from modules.map_path import Direction
from modules.memory import GameState
from modules.navigation import (
    GoalAwareNavigator,
    NavigationActionType,
    NavigationWorld,
    NavigableTile,
    NavigationError,
    prewarm_navigation_tiles,
)
from modules.overworld import (
    OverworldObservation,
    TileObservation,
    TriggerObservation,
    WarpObservation,
    prewarm_static_map_observation,
)


def world(coordinates, *, blocked=(), warps=(), triggers=()):
    blocked = set(blocked)
    return NavigationWorld(
        tiles={("test", coordinate): NavigableTile(("test", coordinate), coordinate in blocked)
               for coordinate in coordinates},
        warps=tuple(warps),
        triggers=tuple(triggers),
    )


class InteractionStateTests(TestCase):
    def test_classifies_representative_interactions(self):
        cases = (
            (InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True), InteractionType.DIALOGUE),
            (InteractionObservation(GameState.CHOOSE_STARTER), InteractionType.CHOICE),
            (InteractionObservation(GameState.BAG_MENU), InteractionType.MENU),
            (InteractionObservation(GameState.BATTLE), InteractionType.BATTLE),
            (InteractionObservation(GameState.OVERWORLD, controllable=True), InteractionType.OVERWORLD),
            (InteractionObservation(GameState.CHANGE_MAP), InteractionType.SPECIAL_INTERACTION),
            (InteractionObservation(GameState.UNKNOWN), InteractionType.UNKNOWN),
        )
        for observation, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(classify_interaction(observation), expected)

    def test_choice_options_take_precedence_over_dialogue(self):
        observation = InteractionObservation(GameState.OVERWORLD, dialogue_waiting=True, choice_options=("YES", "NO"))
        self.assertIs(classify_interaction(observation), InteractionType.CHOICE)


class GoalAwareNavigationTests(TestCase):
    def test_prewarm_static_observation_materializes_static_cache(self):
        map_id = ("static-prewarm", 0)
        path_tile = SimpleNamespace(
            local_coordinates=(0, 0),
            accessible_from_direction=[True, True, True, True],
            warps_to=None,
            traversal_cost=2,
        )
        path_map = SimpleNamespace(tiles=[path_tile])
        map_data = SimpleNamespace(map_size=(1, 1), warps=(), coord_events=(), bg_events=())
        with patch("modules.overworld._get_map_metadata", return_value=path_map), \
                patch("modules.overworld.get_map_metadata", return_value=map_data):
            tiles = prewarm_static_map_observation(map_id)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].traversal_cost, 2)
        self.assertEqual(tiles[0].location, (map_id, (0, 0)))

    def test_prewarm_navigation_tiles_reuses_static_index(self):
        map_id = ("prewarm", 0)
        tiles = tuple(
            TileObservation((map_id, coordinate), False, frozenset(Direction))
            for coordinate in ((0, 0), (1, 0))
        )
        prewarm_navigation_tiles(map_id, tiles)
        first = NavigationWorld.from_overworld(OverworldObservation(
            map_id=map_id, player_coordinates=(0, 0), facing=Direction.East,
            controllable=True, tiles=tiles, warps=(), objects=(), triggers=(),
        ))
        second = NavigationWorld.from_overworld(OverworldObservation(
            map_id=map_id, player_coordinates=(1, 0), facing=Direction.West,
            controllable=True, tiles=tiles, warps=(), objects=(), triggers=(),
        ))
        self.assertIs(first.tiles, second.tiles)

    def test_overworld_observation_adapts_to_navigation_world(self):
        location = (("test"), (0, 0))
        observation = OverworldObservation(
            map_id=("test", 0), player_coordinates=(0, 0), facing=Direction.East,
            controllable=True,
            tiles=(TileObservation(location, False, frozenset({Direction.East})),),
            warps=(), objects=(), triggers=(),
        )
        navigation_world = NavigationWorld.from_overworld(observation)
        self.assertIn(location, navigation_world.tiles)
        self.assertFalse(navigation_world.tiles[location].blocked)

    def test_navigation_reuses_static_tiles_and_overlays_dynamic_blocking(self):
        locations = (("test", (0, 0)), ("test", (1, 0)))
        tiles = tuple(TileObservation(location, False, frozenset(Direction)) for location in locations)
        observation = OverworldObservation(
            map_id=("test", 0), player_coordinates=(0, 0), facing=Direction.East,
            controllable=True, tiles=tiles, warps=(), objects=(), triggers=(),
        )
        first = NavigationWorld.from_overworld(observation)
        second = NavigationWorld.from_overworld(observation)
        self.assertIs(first.tiles, second.tiles)

        blocked = OverworldObservation(
            map_id=("test", 0), player_coordinates=(0, 0), facing=Direction.East,
            controllable=True, tiles=tiles, warps=(), objects=(), triggers=(),
            dynamic_blocked_coordinates=frozenset({(1, 0)}),
        )
        dynamic_world = NavigationWorld.from_overworld(blocked)
        self.assertTrue(dynamic_world.tiles[("test", (1, 0))].blocked)
        self.assertIs(dynamic_world.tiles[("test", (0, 0))], first.tiles[("test", (0, 0))])

    def test_prefers_shortest_matching_warp(self):
        start = (("test"), (0, 0))
        near = WarpObservation((("test"), (1, 0)), (("outside", 0), (0, 0)))
        far = WarpObservation((("test"), (0, 2)), (("outside", 0), (1, 0)))
        tiles = {(x, y) for x in range(3) for y in range(3)}
        plan = GoalAwareNavigator(world(tiles, warps=(far, near))).plan(
            start, ReachWarp(destination_map=("outside", 0))
        )
        self.assertEqual(plan.destination, (("test"), (1, 0)))
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].direction, Direction.East)

    def test_routes_to_trigger_activation_position(self):
        start = (("test"), (0, 1))
        trigger = TriggerObservation(
            "rival", frozenset({(("test"), (2, 1))}), frozenset({(("test"), (2, 0))}), "object_interaction"
        )
        plan = GoalAwareNavigator(world({(x, y) for x in range(3) for y in range(2)}, triggers=(trigger,))).plan(
            start, ActivateTrigger("rival")
        )
        self.assertEqual(plan.destination, (("test"), (2, 0)))
        self.assertEqual(len(plan.actions), 3)

    def test_avoids_explicitly_undesirable_trigger(self):
        start = (("test"), (0, 1))
        bad = TriggerObservation("bad", frozenset({(("test"), (1, 1))}), frozenset({(("test"), (1, 1))}))
        tiles = {(x, y) for x in range(3) for y in range(3)}
        goal = NavigationGoal(
            ReachLocation((("test"), (2, 1))),
            GoalConstraints(avoid_trigger_ids=frozenset({"bad"})),
        )
        plan = GoalAwareNavigator(world(tiles, triggers=(bad,))).plan(start, goal)
        self.assertNotIn((("test"), (1, 1)), [action.destination for action in plan.actions])
        self.assertEqual(len(plan.actions), 4)

    def test_unreachable_goal_is_explicit(self):
        with self.assertRaises(NavigationError):
            GoalAwareNavigator(world({(0, 0), (1, 0)}, blocked={(1, 0)})).plan(
                (("test"), (0, 0)), ReachLocation((("test"), (1, 0)))
            )
