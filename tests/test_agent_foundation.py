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
    WarpActivation,
    prewarm_static_map_observation,
)


def world(coordinates, *, blocked=(), warps=(), triggers=()):
    blocked = set(blocked)
    return NavigationWorld(
        tiles={
            ("test", coordinate): NavigableTile(("test", coordinate), coordinate in blocked)
            for coordinate in coordinates
        },
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
        with patch("modules.overworld._get_map_metadata", return_value=path_map), patch(
            "modules.overworld.get_map_metadata", return_value=map_data
        ):
            tiles = prewarm_static_map_observation(map_id)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].traversal_cost, 2)
        self.assertEqual(tiles[0].location, (map_id, (0, 0)))

    def test_rom_arrow_warp_is_directional_step_while_other_warps_step_on(self):
        map_id = ("warp-semantics", 0)
        destination = ("warp-semantics", 1)
        warp_data = SimpleNamespace(
            local_coordinates=(0, 0),
            destination_location=SimpleNamespace(
                map_group=destination[0], map_number=destination[1], local_position=(1, 1)
            ),
        )
        path_tile = SimpleNamespace(
            local_coordinates=(0, 0),
            accessible_from_direction=[True] * 4,
            warps_to=((destination[0], destination[1]), (1, 1), Direction.North),
            traversal_cost=1,
            has_encounters=False,
            tile_type="North Arrow Warp",
        )
        path_map = SimpleNamespace(tiles=[path_tile])
        map_data = SimpleNamespace(map_size=(1, 1), warps=(warp_data,), coord_events=(), bg_events=())
        with patch("modules.overworld._get_map_metadata", return_value=path_map), patch(
            "modules.overworld.get_map_metadata", return_value=map_data
        ):
            tiles = prewarm_static_map_observation(map_id)
        self.assertEqual(tiles[0].warp.activation, WarpActivation.DIRECTIONAL_STEP)
        self.assertEqual(tiles[0].warp.required_facing, Direction.North)

    def test_escalator_derives_one_valid_source_tile_from_rom_tile_behavior(self):
        map_id = ("escalator", 0)
        destination = ("escalator", 1)
        warp_data = SimpleNamespace(
            local_coordinates=(1, 1),
            destination_location=SimpleNamespace(
                map_group=destination[0], map_number=destination[1], local_position=(2, 2)
            ),
        )
        tiles = [
            SimpleNamespace(
                local_coordinates=(x, y),
                accessible_from_direction=[True] * 4,
                warps_to=None,
                traversal_cost=1,
                has_encounters=False,
                tile_type="Escalator Up" if (x, y) == (1, 1) else "Floor",
            )
            for y in range(3)
            for x in range(3)
        ]
        path_map = SimpleNamespace(tiles=tiles)
        map_data = SimpleNamespace(map_size=(3, 3), warps=(warp_data,), coord_events=(), bg_events=())
        with patch("modules.overworld._get_map_metadata", return_value=path_map), patch(
            "modules.overworld.get_map_metadata", return_value=map_data
        ):
            observed_tiles = prewarm_static_map_observation(map_id)
        warp = next(tile.warp for tile in observed_tiles if tile.warp is not None)
        self.assertEqual(warp.activation_locations, frozenset({(map_id, (2, 1))}))
        self.assertIs(warp.activation_direction, Direction.West)
        self.assertIs(warp.required_facing, Direction.West)

    def test_escalator_goal_requires_right_source_and_left_facing(self):
        map_id = ("escalator-goal", 0)
        entry = (map_id, (1, 1))
        source = (map_id, (2, 1))
        destination = (("escalator-goal", 1), (0, 0))
        warp = WarpObservation(
            entry,
            destination,
            required_facing=Direction.West,
            activation_locations=frozenset({source}),
            activation_direction=Direction.West,
        )
        world = NavigationWorld(
            tiles={(map_id, (x, y)): NavigableTile((map_id, (x, y))) for y in range(3) for x in range(3)},
            warps=(warp,),
            facing=Direction.North,
        )
        navigator = GoalAwareNavigator(world)

        self.assertFalse(
            navigator.satisfies((map_id, (1, 0)), Direction.South, ReachWarp(destination_map=destination[0]))
        )
        self.assertFalse(
            navigator.satisfies((map_id, (1, 2)), Direction.North, ReachWarp(destination_map=destination[0]))
        )
        self.assertTrue(navigator.satisfies(source, Direction.West, ReachWarp(destination_map=destination[0])))
        plan = navigator.plan((map_id, (1, 0)), ReachWarp(destination_map=destination[0]))
        self.assertEqual(plan.destination, source)
        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.TURN)
        self.assertEqual(plan.actions[-1].direction, Direction.West)

    def test_interaction_goal_requires_turn_and_never_finishes_on_adjacency_alone(self):
        map_id = ("npc-facing", 0)
        source = (map_id, (1, 1))
        trigger = TriggerObservation(
            "npc",
            frozenset({(map_id, (1, 0))}),
            frozenset({source}),
            "object_interaction",
            activation_requirements=((source, Direction.North),),
        )
        navigation_world = NavigationWorld(
            tiles={
                (map_id, (1, 1)): NavigableTile((map_id, (1, 1))),
                (map_id, (1, 0)): NavigableTile((map_id, (1, 0))),
            },
            triggers=(trigger,),
            facing=Direction.East,
        )
        plan = GoalAwareNavigator(navigation_world).plan(source, ActivateTrigger("npc"))
        self.assertEqual(
            [(action.action_type, action.direction) for action in plan.actions],
            [(NavigationActionType.TURN, Direction.North)],
        )
        self.assertTrue(GoalAwareNavigator(navigation_world).satisfies(source, Direction.North, ActivateTrigger("npc")))
        self.assertFalse(GoalAwareNavigator(navigation_world).satisfies(source, Direction.East, ActivateTrigger("npc")))

    def test_interaction_goal_requires_the_target_facing_from_each_side(self):
        map_id = ("npc-sides", 0)
        npc = (2, 2)
        requirements = {
            (2, 1): Direction.South,
            (3, 2): Direction.West,
            (2, 3): Direction.North,
            (1, 2): Direction.East,
        }
        for source_coordinates, required in requirements.items():
            source = (map_id, source_coordinates)
            trigger = TriggerObservation(
                "npc",
                frozenset({(map_id, npc)}),
                frozenset({source}),
                "object_interaction",
                activation_requirements=((source, required),),
            )
            world = NavigationWorld(
                tiles={source: NavigableTile(source), (map_id, npc): NavigableTile((map_id, npc))},
                triggers=(trigger,),
                facing=required.opposite(),
            )
            with self.subTest(source=source_coordinates):
                plan = GoalAwareNavigator(world).plan(source, ActivateTrigger("npc"))
                self.assertEqual(plan.actions[-1].action_type, NavigationActionType.TURN)
                self.assertEqual(plan.actions[-1].direction, required)

    def test_interaction_source_state_must_be_walkable(self):
        map_id = ("npc-source-validation", 0)
        trigger = TriggerObservation(
            "npc",
            frozenset({(map_id, (1, 1))}),
            frozenset({(map_id, (0, 1)), (map_id, (2, 1))}),
            "object_interaction",
            activation_requirements=(
                ((map_id, (0, 1)), Direction.East),
                ((map_id, (2, 1)), Direction.West),
            ),
        )
        world = NavigationWorld(
            tiles={
                (map_id, (0, 1)): NavigableTile((map_id, (0, 1)), blocked=True),
                (map_id, (1, 1)): NavigableTile((map_id, (1, 1))),
                (map_id, (2, 1)): NavigableTile((map_id, (2, 1))),
            },
            triggers=(trigger,),
            facing=Direction.South,
        )

        plan = GoalAwareNavigator(world).plan((map_id, (2, 1)), ActivateTrigger("npc"))

        self.assertEqual(plan.destination, (map_id, (2, 1)))
        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.TURN)
        self.assertEqual(plan.actions[-1].direction, Direction.West)

    def test_prewarm_navigation_tiles_reuses_static_index(self):
        map_id = ("prewarm", 0)
        tiles = tuple(
            TileObservation((map_id, coordinate), False, frozenset(Direction)) for coordinate in ((0, 0), (1, 0))
        )
        prewarm_navigation_tiles(map_id, tiles)
        first = NavigationWorld.from_overworld(
            OverworldObservation(
                map_id=map_id,
                player_coordinates=(0, 0),
                facing=Direction.East,
                controllable=True,
                tiles=tiles,
                warps=(),
                objects=(),
                triggers=(),
            )
        )
        second = NavigationWorld.from_overworld(
            OverworldObservation(
                map_id=map_id,
                player_coordinates=(1, 0),
                facing=Direction.West,
                controllable=True,
                tiles=tiles,
                warps=(),
                objects=(),
                triggers=(),
            )
        )
        self.assertIs(first.tiles, second.tiles)

    def test_overworld_observation_adapts_to_navigation_world(self):
        location = (("test"), (0, 0))
        observation = OverworldObservation(
            map_id=("test", 0),
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=(TileObservation(location, False, frozenset({Direction.East})),),
            warps=(),
            objects=(),
            triggers=(),
        )
        navigation_world = NavigationWorld.from_overworld(observation)
        self.assertIn(location, navigation_world.tiles)
        self.assertFalse(navigation_world.tiles[location].blocked)

    def test_navigation_reuses_static_tiles_and_overlays_dynamic_blocking(self):
        locations = (("test", (0, 0)), ("test", (1, 0)))
        tiles = tuple(TileObservation(location, False, frozenset(Direction)) for location in locations)
        observation = OverworldObservation(
            map_id=("test", 0),
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=tiles,
            warps=(),
            objects=(),
            triggers=(),
        )
        first = NavigationWorld.from_overworld(observation)
        second = NavigationWorld.from_overworld(observation)
        self.assertIs(first.tiles, second.tiles)

        blocked = OverworldObservation(
            map_id=("test", 0),
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=tiles,
            warps=(),
            objects=(),
            triggers=(),
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

    def test_reach_warp_preserves_source_identity_for_equal_destinations(self):
        destination = (("outside", 0), (9, 9))
        first = WarpObservation(("test", (1, 0)), destination)
        second = WarpObservation(("test", (1, 2)), destination)
        selected = ReachWarp(destination_map=("outside", 0), destination=destination, warp=second)
        navigator = GoalAwareNavigator(world({(x, y) for x in range(3) for y in range(3)}, warps=(first, second)))

        self.assertTrue(navigator.satisfies(("test", (1, 2)), None, selected))
        self.assertFalse(navigator.satisfies(("test", (1, 0)), None, selected))
        self.assertEqual(navigator.plan(("test", (0, 1)), selected).destination, ("test", (1, 2)))

    def test_cross_map_plan_restricts_candidates_to_selected_warp(self):
        source = "test"
        outside = ("outside", 0)
        first = WarpObservation((source, (1, 0)), (outside, (0, 0)))
        second = WarpObservation((source, (1, 2)), (outside, (2, 0)))
        selected = ReachWarp(destination_map=outside, destination=second.destination, warp=second)
        navigation_world = world({(x, y) for x in range(3) for y in range(3)}, warps=(first, second))
        from modules.navigation import plan_with_world_navigation
        from modules.world_navigation import WorldEdge, WorldMapGraph

        plan, _ = plan_with_world_navigation(
            navigation_world,
            (source, (0, 1)),
            selected,
            graph=WorldMapGraph((WorldEdge(source, outside, "warp", ((1, 0), (1, 2)), ((0, 0), (2, 0))),)),
        )
        self.assertEqual(plan.actions[-1].destination, (outside, (2, 0)))

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
