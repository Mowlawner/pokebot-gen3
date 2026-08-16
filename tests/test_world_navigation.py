import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.agent_control import (
    AgentActionExecutor,
    AgentActionType,
    AgentControlLoop,
    AgentObservation,
    GoalStatus,
    evaluate_goal,
    select_action,
)
from modules.goals import ActivateTrigger, NavigationGoal, ReachInteractionPosition, ReachLocation, EncounterMode
from modules.interaction_state import InteractionObservation
from modules.map_path import Direction
from modules.memory import GameState
from modules.navigation import (
    GoalAwareNavigator,
    NavigationAction,
    NavigationActionType,
    NavigationWorld,
    NavigableTile,
    plan_with_world_navigation,
)
from modules.overworld import OverworldObservation, TileObservation, TriggerObservation, WarpActivation
from modules.overworld import WarpObservation
from modules.trigger_bindings import BindingResolution, TriggerBinding
from modules.world_navigation import WorldEdge, WorldMapGraph


def edge(source, destination, source_coordinate=(0, 0), destination_coordinate=(0, 0), *, kind="warp"):
    return WorldEdge(
        source_map=source,
        destination_map=destination,
        kind=kind,
        source_coordinates=(source_coordinate,),
        destination_coordinates=(destination_coordinate,),
    )


class TestWorldMapGraph(unittest.TestCase):
    def test_simple_two_map_route(self):
        graph = WorldMapGraph((edge((0, 0), (0, 1)),))

        route = graph.route((0, 0), (0, 1))

        self.assertEqual(route.maps, ((0, 0), (0, 1)))
        self.assertEqual(route.edges[0].source_coordinates, ((0, 0),))

    def test_multi_map_route_and_shortest_branch(self):
        graph = WorldMapGraph(
            (
                edge((0, 0), (0, 1)),
                edge((0, 1), (0, 2)),
                edge((0, 0), (0, 3)),
                edge((0, 3), (0, 4)),
                edge((0, 4), (0, 2)),
            )
        )

        route = graph.route((0, 0), (0, 2))

        self.assertEqual(route.maps, ((0, 0), (0, 1), (0, 2)))

    def test_unreachable_and_current_map_routes(self):
        graph = WorldMapGraph((edge((0, 0), (0, 1)),))

        self.assertEqual(graph.route((0, 0), (0, 0)).edges, ())
        with self.assertRaisesRegex(RuntimeError, "No map route"):
            graph.route((0, 0), (9, 9))

    def test_multiple_outgoing_edges_retain_source_association(self):
        first = edge((0, 0), (0, 1), source_coordinate=(1, 2))
        second = edge((0, 0), (0, 2), source_coordinate=(8, 2))
        graph = WorldMapGraph((first, second))

        self.assertEqual(graph.outgoing((0, 0)), (first, second))
        self.assertEqual(graph.route((0, 0), (0, 2)).edges, (second,))

    def test_connection_edge_derives_only_boundary_pairs(self):
        source = SimpleNamespace(
            map_size=(4, 3),
            warps=[],
            connections=[
                SimpleNamespace(
                    destination_map_group=0,
                    destination_map_number=1,
                    direction="North",
                    offset=1,
                )
            ],
        )
        destination = SimpleNamespace(map_size=(3, 2), warps=[], connections=[])

        graph = WorldMapGraph.from_map_data({(0, 0): source, (0, 1): destination})

        connection = graph.outgoing((0, 0))[0]
        self.assertEqual(connection.kind, "connection")
        self.assertEqual(connection.source_coordinates, ((1, 0), (2, 0), (3, 0)))
        self.assertEqual(connection.destination_coordinates, ((0, 1), (1, 1), (2, 1)))


class TestWorldGoalIntegration(unittest.TestCase):
    def test_offscreen_static_target_is_navigable(self):
        map_id = (0, 18)
        tiles = tuple(TileObservation((map_id, (x, 0)), False, frozenset(Direction)) for x in range(3))
        binding = TriggerBinding("rival", map_id, "RivalScript", local_id=2)
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            triggers=(
                TriggerObservation(
                    "rival",
                    frozenset(),
                    frozenset(),
                    "semantic_object",
                    map_id,
                    frozenset({(map_id, (2, 0))}),
                ),
            ),
            bindings=(
                BindingResolution(
                    binding=binding,
                    static_match=True,
                    static_location=(map_id, (2, 0)),
                    static_available=True,
                ),
            ),
        )

        plan, _ = plan_with_world_navigation(world, (map_id, (0, 0)), ActivateTrigger("rival"))

        self.assertEqual(plan.destination, (map_id, (2, 0)))
        self.assertEqual(len(plan.actions), 2)


class TestWeightedNavigation(unittest.TestCase):
    MAP = (99, 0)

    @classmethod
    def world(cls, coordinates, grass=(), blocked=()):
        coordinates = set(coordinates)
        all_directions = frozenset(Direction)
        return NavigationWorld(
            tiles={
                (cls.MAP, coordinate): NavigableTile(
                    (cls.MAP, coordinate),
                    coordinate in blocked,
                    all_directions,
                    2 if coordinate in grass else 1,
                )
                for coordinate in coordinates
            }
        )

    def test_normal_terrain_has_baseline_cost(self):
        world = self.world({(0, 0)})
        self.assertEqual(world.tiles[(self.MAP, (0, 0))].traversal_cost, 1)

    def test_same_length_route_prefers_fewer_encounter_opportunities(self):
        coordinates = {(x, y) for y in range(3) for x in range(3)}
        world = NavigationWorld(
            tiles={
                (self.MAP, coordinate): NavigableTile(
                    (self.MAP, coordinate),
                    False,
                    frozenset(Direction),
                    1,
                    coordinate in {(0, 0), (1, 0), (1, 2)},
                )
                for coordinate in coordinates
                if coordinate != (1, 1)
            },
            facing=Direction.North,
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 1)), ReachLocation((self.MAP, (2, 1))))
        self.assertEqual(plan.metrics.encounter_opportunities, 1)
        self.assertEqual(plan.metrics.movement_actions, 4)

    def test_normal_prefers_fewer_opportunities_over_shorter_route(self):
        # The upper branch is shorter (4 moves) but crosses two encounter
        # tiles.  The lower branch takes 6 moves and has no exposure.
        coordinates = {(x, y) for y in range(3) for x in range(5)}
        grass = {(1, 0), (2, 0)}
        world = NavigationWorld(
            tiles={
                (self.MAP, coordinate): NavigableTile(
                    (self.MAP, coordinate), False, frozenset(Direction), 1, coordinate in grass
                )
                for coordinate in coordinates
            }
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachLocation((self.MAP, (4, 0))))

        self.assertEqual(plan.metrics.encounter_opportunities, 0)
        self.assertEqual(plan.metrics.movement_actions, 6)
        self.assertEqual(plan.metrics.ordinary_movement_steps, 6)

    def test_first_grass_patch_geometry_selects_lower_zero_exposure_branch(self):
        # This is the shape of the first patch decision: the direct upper
        # branch crosses two grass tiles, while the lower branch adds no
        # grass and has the same number of movement actions.
        coordinates = {
            (0, 1),
            (3, 1),
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
        }
        grass = {(1, 0), (2, 0)}
        world = NavigationWorld(
            tiles={
                (self.MAP, coordinate): NavigableTile(
                    (self.MAP, coordinate), False, frozenset(Direction), 1, coordinate in grass
                )
                for coordinate in coordinates
            },
            facing=Direction.East,
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 1)), ReachLocation((self.MAP, (3, 1))))

        self.assertEqual(plan.metrics.movement_actions, 5)
        self.assertEqual(plan.metrics.encounter_terrain_moves, 0)
        self.assertEqual(plan.metrics.turns_on_encounter_terrain, 0)
        self.assertEqual(plan.metrics.turns_on_non_encounter_terrain, 3)
        self.assertEqual(plan.metrics.encounter_opportunities, 0)
        self.assertEqual(plan.metrics.total_route_cost, 8)
        self.assertTrue(all(action.destination[1] not in grass for action in plan.actions))

        # Isolate the competing direct branch using the same map geometry so
        # this regression records why the branch is rejected, not just which
        # coordinates happen to be returned.
        upper_coordinates = {(0, 1), (3, 1), (0, 0), (1, 0), (2, 0), (3, 0)}
        upper_world = NavigationWorld(
            tiles={location: world.tiles[location] for location in world.tiles if location[1] in upper_coordinates},
            facing=Direction.East,
        )
        upper = GoalAwareNavigator(upper_world).plan((self.MAP, (0, 1)), ReachLocation((self.MAP, (3, 1))))
        self.assertEqual(upper.metrics.movement_actions, 5)
        self.assertEqual(upper.metrics.encounter_terrain_moves, 2)
        self.assertGreater(upper.metrics.encounter_opportunities, plan.metrics.encounter_opportunities)

    def test_first_grass_patch_seek_mode_does_not_apply_exposure_penalty(self):
        coordinates = {
            (0, 1),
            (3, 1),
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
        }
        grass = {(1, 0), (2, 0)}
        world = NavigationWorld(
            tiles={
                (self.MAP, coordinate): NavigableTile(
                    (self.MAP, coordinate), False, frozenset(Direction), 1, coordinate in grass
                )
                for coordinate in coordinates
            },
            facing=Direction.East,
        )
        plan = GoalAwareNavigator(world).plan(
            (self.MAP, (0, 1)),
            NavigationGoal(ReachLocation((self.MAP, (3, 1))), encounter_mode=EncounterMode.SEEK),
        )
        self.assertTrue(any(action.destination[1] in grass for action in plan.actions))

    def test_turns_on_encounter_terrain_are_reported_as_encounter_opportunities(self):
        world = NavigationWorld(
            tiles={
                (self.MAP, coordinate): NavigableTile(
                    (self.MAP, coordinate), False, frozenset(Direction), 1, coordinate == (0, 0)
                )
                for coordinate in ((0, 0), (1, 0), (1, 1))
            },
            facing=Direction.North,
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachLocation((self.MAP, (1, 1))))
        self.assertEqual(plan.metrics.encounter_opportunities, 1)
        self.assertEqual(plan.metrics.turns_on_encounter_terrain, 1)
        self.assertEqual(plan.metrics.turns_on_non_encounter_terrain, 1)

    def test_turning_on_grass_is_an_encounter_opportunity(self):
        world = NavigationWorld(
            tiles={
                (self.MAP, (0, 0)): NavigableTile((self.MAP, (0, 0)), False, frozenset(Direction), 1, True),
                (self.MAP, (0, 1)): NavigableTile((self.MAP, (0, 1)), False, frozenset(Direction), 1, False),
            },
            facing=Direction.East,
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachLocation((self.MAP, (0, 1))))
        self.assertEqual(plan.metrics.encounter_opportunities, 1)
        self.assertEqual(plan.metrics.turns_on_encounter_terrain, 1)

    def test_interaction_route_includes_turn_before_adjacent_move(self):
        trigger = TriggerObservation(
            "npc",
            frozenset({(self.MAP, (2, 0))}),
            frozenset({(self.MAP, (1, 0))}),
            activation_requirements=(((self.MAP, (1, 0)), Direction.East),),
        )
        world = NavigationWorld(
            tiles={
                (self.MAP, (x, 0)): NavigableTile((self.MAP, (x, 0)), False, frozenset(Direction)) for x in range(3)
            },
            triggers=(trigger,),
            facing=Direction.North,
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachInteractionPosition("npc"))
        self.assertEqual(plan.actions[0].action_type, NavigationActionType.TURN)
        self.assertEqual(plan.actions[0].source, plan.actions[0].destination)
        self.assertEqual(plan.actions[1].action_type, NavigationActionType.MOVE)
        self.assertEqual(plan.actions[1].direction, Direction.East)

    def test_semantic_interaction_route_preserves_required_facing(self):
        trigger = TriggerObservation(
            "rival",
            frozenset({(self.MAP, (1, 1))}),
            frozenset({(self.MAP, (1, 0)), (self.MAP, (2, 1)), (self.MAP, (1, 2)), (self.MAP, (0, 1))}),
            activation_requirements=(
                ((self.MAP, (1, 0)), Direction.South),
                ((self.MAP, (2, 1)), Direction.West),
                ((self.MAP, (1, 2)), Direction.North),
                ((self.MAP, (0, 1)), Direction.East),
            ),
        )
        world = NavigationWorld(
            tiles={(self.MAP, (x, y)): NavigableTile((self.MAP, (x, y))) for x in range(3) for y in range(3)},
            triggers=(trigger,),
            facing=Direction.East,
        )

        plan = GoalAwareNavigator(world).plan((self.MAP, (1, 2)), ActivateTrigger("rival"))

        self.assertEqual(plan.destination, (self.MAP, (1, 2)))
        self.assertEqual(
            plan.actions,
            (NavigationAction(NavigationActionType.TURN, Direction.North, (self.MAP, (1, 2)), (self.MAP, (1, 2))),),
        )

    def test_grass_route_is_avoided_when_normal_detour_is_cheaper(self):
        coordinates = {(x, y) for y in (0, 1) for x in range(5)}
        world = self.world(coordinates, grass={(1, 0), (2, 0), (3, 0)})
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachLocation((self.MAP, (4, 0))))
        self.assertTrue(all(action.destination[1][1] == 1 for action in plan.actions[1:-1]))

    def test_grass_is_used_when_it_is_the_only_route(self):
        coordinates = {(x, 0) for x in range(4)}
        world = self.world(coordinates, grass={(1, 0), (2, 0)})
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachLocation((self.MAP, (3, 0))))
        self.assertEqual([action.destination[1] for action in plan.actions], [(1, 0), (2, 0), (3, 0)])

    def test_large_normal_detour_beats_grass_route_only_when_cheaper(self):
        # The normal branch has eight steps; the grass branch has four steps,
        # so cost 2 still makes the grass branch the practical choice.
        path = (
            ((0, 0), Direction.East, 1),
            ((1, 0), Direction.East, 2),
            ((2, 0), Direction.East, 2),
            ((3, 0), Direction.East, 2),
            ((4, 0), frozenset({Direction.East, Direction.North}), 1),
            ((0, 1), Direction.South, 1),
            ((0, 2), Direction.South, 1),
            ((1, 2), Direction.East, 1),
            ((2, 2), Direction.East, 1),
            ((3, 2), Direction.East, 1),
            ((4, 2), Direction.East, 1),
            ((4, 1), Direction.South, 1),
        )
        world = NavigationWorld(
            tiles={
                (self.MAP, coordinate): NavigableTile(
                    (self.MAP, coordinate),
                    False,
                    direction if isinstance(direction, frozenset) else frozenset({direction}),
                    cost,
                )
                for coordinate, direction, cost in path
            }
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachLocation((self.MAP, (4, 0))))
        self.assertTrue(any(action.destination[1] in {(1, 0), (2, 0), (3, 0)} for action in plan.actions))

    def test_equal_cost_normal_tiles_retain_shortest_route(self):
        coordinates = {(x, y) for y in (0, 1) for x in range(3)}
        world = self.world(coordinates)
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), ReachLocation((self.MAP, (2, 0))))
        self.assertEqual([action.destination[1] for action in plan.actions], [(1, 0), (2, 0)])

    def test_dynamic_blocking_does_not_change_static_terrain_cost(self):
        location = (self.MAP, (1, 0))
        observation = OverworldObservation(
            self.MAP,
            (0, 0),
            Direction.East,
            True,
            (
                TileObservation((self.MAP, (0, 0)), False, frozenset(Direction)),
                TileObservation(location, False, frozenset(Direction), traversal_cost=2),
            ),
            (),
            (),
            (),
            dynamic_blocked_coordinates=frozenset({(1, 0)}),
        )
        world = NavigationWorld.from_overworld(observation)
        self.assertTrue(world.tiles[location].blocked)
        self.assertEqual(world.tiles[location].traversal_cost, 2)

    def test_static_target_reached_waits_for_runtime_spawn(self):
        map_id = (0, 18)
        location = (map_id, (0, 0))
        binding = TriggerBinding("rival", map_id, "RivalScript", local_id=2)
        observation = AgentObservation(
            InteractionObservation(GameState.OVERWORLD),
            overworld=OverworldObservation(
                map_id=map_id,
                player_coordinates=(0, 0),
                facing=Direction.South,
                controllable=True,
                tiles=(TileObservation(location, False, frozenset(Direction)),),
                warps=(),
                objects=(),
                triggers=(
                    TriggerObservation(
                        "rival",
                        frozenset(),
                        frozenset(),
                        "semantic_object",
                        map_id,
                        frozenset({location}),
                    ),
                ),
                bindings=(
                    BindingResolution(
                        binding=binding,
                        static_match=True,
                        static_location=location,
                        static_available=True,
                    ),
                ),
            ),
            goal=ActivateTrigger("rival"),
        )

        decision = evaluate_goal(observation)
        selected = select_action(observation)

        self.assertEqual(decision.status, GoalStatus.REACHABLE)
        self.assertEqual(selected.action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertNotEqual(selected.action.action_type, AgentActionType.INTERACT)

    def test_warp_action_is_the_step_onto_a_normal_warp_tile(self):
        source_map = (0, 0)
        target_map = (0, 1)
        tiles = tuple(TileObservation((source_map, (x, 0)), False, frozenset(Direction)) for x in range(3))
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            warps=(WarpObservation((source_map, (2, 0)), (target_map, (0, 0))),),
        )
        graph = WorldMapGraph((edge(source_map, target_map, source_coordinate=(2, 0)),))

        plan, _ = plan_with_world_navigation(
            world, (source_map, (0, 0)), ReachLocation((target_map, (0, 0))), graph=graph
        )

        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].source, (source_map, (1, 0)))
        self.assertEqual(plan.actions[-1].direction, Direction.East)
        self.assertEqual(world.warps[0].activation, WarpActivation.STEP_ON)

    def test_arrow_warp_steps_onto_tile_then_uses_required_facing(self):
        source_map = (0, 0)
        target_map = (0, 1)
        tiles = tuple(TileObservation((source_map, (x, 0)), False, frozenset(Direction)) for x in range(3))
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            warps=(WarpObservation((source_map, (2, 0)), (target_map, (0, 0)), Direction.North),),
        )
        graph = WorldMapGraph((edge(source_map, target_map, source_coordinate=(2, 0)),))

        plan, _ = plan_with_world_navigation(
            world, (source_map, (0, 0)), ReachLocation((target_map, (0, 0))), graph=graph
        )

        self.assertEqual(plan.actions[-2].action_type, NavigationActionType.MOVE)
        self.assertEqual(plan.actions[-2].destination, (source_map, (2, 0)))
        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].direction, Direction.North)

    def test_cross_map_semantic_goal_gets_world_transition_plan(self):
        current_map = (1, 4)
        target_map = (0, 18)
        tiles = tuple(TileObservation((current_map, (x, 0)), False, frozenset(Direction)) for x in range(3))
        overworld = OverworldObservation(
            map_id=current_map,
            player_coordinates=(0, 0),
            facing=Direction.East,
            controllable=True,
            tiles=tiles,
            warps=(),
            objects=(),
            triggers=(
                TriggerObservation(
                    "introductory_rival",
                    frozenset(),
                    frozenset(),
                    "semantic_object",
                    target_map=target_map,
                ),
            ),
        )
        observation = AgentObservation(
            InteractionObservation(GameState.OVERWORLD),
            overworld=overworld,
            goal=ActivateTrigger("introductory_rival"),
        )
        graph = WorldMapGraph((edge(current_map, target_map, source_coordinate=(2, 0)),))

        with patch("modules.navigation.get_world_map_graph", return_value=graph):
            evaluation = evaluate_goal(observation)

        self.assertEqual(evaluation.status, GoalStatus.REACHABLE)
        self.assertIsNotNone(evaluation.plan)
        self.assertEqual(evaluation.plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(evaluation.plan.actions[-1].destination, (target_map, (0, 0)))

    def test_replanning_uses_observed_destination_as_new_source(self):
        graph = WorldMapGraph(
            (
                edge((0, 0), (0, 1)),
                edge((0, 1), (0, 2)),
            )
        )

        first = graph.route((0, 0), (0, 2))
        second = graph.route(first.edges[0].destination_map, (0, 2))

        self.assertEqual(first.maps, ((0, 0), (0, 1), (0, 2)))
        self.assertEqual(second.maps, ((0, 1), (0, 2)))

    def test_transition_waits_for_and_verifies_destination_before_resuming_goal(self):
        source_map = (0, 0)
        target_map = (0, 1)
        goal = ActivateTrigger("introductory_rival")
        source_world = OverworldObservation(
            map_id=source_map,
            player_coordinates=(0, 0),
            facing=Direction.South,
            controllable=True,
            tiles=(TileObservation((source_map, (0, 0)), False, frozenset(Direction)),),
            warps=(),
            objects=(),
            triggers=(
                TriggerObservation(
                    "introductory_rival",
                    frozenset(),
                    frozenset(),
                    "semantic_object",
                    target_map=target_map,
                ),
            ),
        )
        target_world = OverworldObservation(
            map_id=target_map,
            player_coordinates=(0, 0),
            facing=Direction.South,
            controllable=True,
            tiles=(TileObservation((target_map, (0, 0)), False, frozenset(Direction)),),
            warps=(),
            objects=(),
            triggers=(
                TriggerObservation(
                    "introductory_rival",
                    frozenset({(target_map, (0, 0))}),
                    frozenset({(target_map, (0, 0))}),
                    "semantic_object",
                    target_map=target_map,
                ),
            ),
        )
        observations = iter((source_world, source_world, target_world))
        graph = WorldMapGraph((edge(source_map, target_map),))
        emulator = SimpleNamespace(press_button=lambda _: None)
        loop = AgentControlLoop(
            lambda: AgentObservation(
                InteractionObservation(GameState.OVERWORLD),
                overworld=next(observations),
                goal=goal,
            ),
            AgentActionExecutor(),
        )

        with patch("modules.navigation.get_world_map_graph", return_value=graph), patch(
            "modules.agent_control.context.emulator", emulator
        ):
            first = loop.step()
            settling = loop.step()
            resumed = loop.step()

        self.assertEqual(first[1].action.action_type, AgentActionType.NAVIGATE_TOWARD_GOAL)
        self.assertEqual(first[1].action.navigation.action_type, NavigationActionType.WARP)
        self.assertEqual(settling[1].action.action_type, AgentActionType.WAIT_REOBSERVE)
        self.assertEqual(resumed[1].action.action_type, AgentActionType.INTERACT)


if __name__ == "__main__":
    unittest.main()
