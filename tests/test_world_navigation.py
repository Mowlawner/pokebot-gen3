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
from modules.goals import (
    ActivateTrigger,
    EngageTrainer,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
    EncounterMode,
    GoalConstraints,
    TrainerMode,
    SemanticTarget,
)
from modules.interaction_state import InteractionObservation
from modules.map_path import Direction
from modules.memory import GameState
from modules.navigation import (
    GoalAwareNavigator,
    _global_navigation_world,
    _plan_exact_cross_map_warp_chain,
    NavigationAction,
    NavigationActionType,
    NavigationWorld,
    NavigableTile,
    map_connection_approach_position,
    plan_observed_warp_locally,
    plan_with_world_navigation,
    classify_transition_relevance,
    transition_world_route,
    TransitionRelevance,
    NavigationError,
    NavigationSearchLimitExceeded,
    prewarm_navigation_tiles,
    goal_target_map,
)
from modules.nuzlocke.emerald_capabilities import (
    NavigationLoopDetected,
    _record_completed_navigation_transition,
)
from modules.map_data import MapRSE
from modules.overworld import (
    MapConnectionObservation,
    OverworldObservation,
    TileObservation,
    TriggerObservation,
    WarpActivation,
    WorldTransition,
    _connection_endpoint_is_executable,
)
from modules.overworld import WarpObservation
from modules.trigger_bindings import BindingResolution, TriggerBinding
from modules.world_navigation import WorldEdge, WorldMapGraph, WorldRoute


def edge(source, destination, source_coordinate=(0, 0), destination_coordinate=(0, 0), *, kind="warp"):
    return WorldEdge(
        source_map=source,
        destination_map=destination,
        kind=kind,
        source_coordinates=(source_coordinate,),
        destination_coordinates=(destination_coordinate,),
    )


class TestWorldMapGraph(unittest.TestCase):
    def test_navigation_tile_cache_accepts_enum_and_tuple_map_ids(self):
        map_enum = MapRSE.ROUTE103
        map_value = map_enum.value
        tile = TileObservation((map_value, (0, 0)), False, frozenset(Direction))
        prewarm_navigation_tiles(map_enum, (tile,))
        observation = OverworldObservation(
            map_enum,
            (0, 0),
            Direction.East,
            True,
            (tile,),
            (),
            (),
            (),
        )

        world = NavigationWorld.from_overworld(observation)

        self.assertEqual(world.tiles[(map_value, (0, 0))].location, (map_value, (0, 0)))
        self.assertEqual(world.tiles[(map_enum, (0, 0))].location, (map_value, (0, 0)))

    def test_navigation_loop_detection_normalizes_enum_map_ids(self):
        first = SimpleNamespace(
            kind="warp",
            entry=(MapRSE.ROUTE103, (0, 0)),
            destination=(MapRSE.ROUTE102, (1, 1)),
        )
        second = SimpleNamespace(
            kind="warp",
            entry=(MapRSE.ROUTE102, (1, 1)),
            destination=(MapRSE.ROUTE103, (0, 0)),
        )
        target = SimpleNamespace(target_map=MapRSE.ROUTE110.value)
        progress = {}

        with self.assertRaises(NavigationLoopDetected):
            for transition in (first, second, first, second):
                _record_completed_navigation_transition(progress, transition, target)

    def test_frozen_global_world_does_not_read_static_metadata_in_worker(self):
        source_map = (0, 1)
        destination_map = (0, 2)
        source = (source_map, (1, 0))
        destination = (destination_map, (0, 0))
        transition = WorldTransition(source, destination)
        tile = NavigableTile(source, False, frozenset(Direction))
        world = NavigationWorld(
            tiles={source: tile},
            transitions=(transition,),
            static_tile_catalog=((source_map, ((source, tile),)),),
            static_transition_catalog=((source_map, (transition,)),),
            static_metadata_frozen=True,
        )
        graph = WorldMapGraph(
            (
                WorldEdge(
                    source_map,
                    destination_map,
                    "warp",
                    ((1, 0),),
                    ((0, 0),),
                ),
            )
        )

        with (
            patch("modules.navigation.static_map_transitions", side_effect=AssertionError("worker ROM read")),
            patch("modules.navigation.prewarm_static_map_observation", side_effect=AssertionError("worker ROM read")),
        ):
            enriched = _global_navigation_world(world, graph, enrich_maps=(source_map,))

        self.assertTrue(enriched.static_metadata_frozen)
        self.assertEqual(enriched.transitions[0].destination, destination)

    def test_global_overlay_uses_static_current_map_transition_as_fallback(self):
        source = (1, 4)
        destination = (0, 9)
        static_exit = WorldTransition(entry=(source, (1, 0)), destination=(destination, (2, 2)))
        observed = NavigationWorld(
            tiles={
                (source, (0, 0)): NavigableTile((source, (0, 0)), False, frozenset(Direction)),
            },
            warps=(),
            triggers=(),
            bindings=(),
            facing=Direction.East,
            transitions=(),
        )
        graph = WorldMapGraph((edge(source, destination, source_coordinate=(1, 0), destination_coordinate=(2, 2)),))

        with patch("modules.navigation.static_map_transitions", return_value=(static_exit,)):
            global_world = _global_navigation_world(observed, graph, enrich_maps=())

        self.assertIn(static_exit, global_world.transitions)

    def test_global_overlay_preserves_live_transition_over_static_duplicate(self):
        source = (1, 4)
        destination = (0, 9)
        live_exit = WorldTransition(entry=(source, (1, 0)), destination=(destination, (2, 2)))
        static_exit = WorldTransition(
            entry=(source, (1, 0)), destination=(destination, (2, 2)), required_facing=Direction.East
        )
        observed = NavigationWorld(
            tiles={(source, (0, 0)): NavigableTile((source, (0, 0)), False, frozenset(Direction))},
            warps=(),
            triggers=(),
            bindings=(),
            facing=Direction.East,
            transitions=(live_exit,),
        )
        graph = WorldMapGraph((edge(source, destination, source_coordinate=(1, 0), destination_coordinate=(2, 2)),))

        with patch("modules.navigation.static_map_transitions", return_value=(static_exit,)):
            global_world = _global_navigation_world(observed, graph, enrich_maps=())

        matching = tuple(t for t in global_world.transitions if t.entry == live_exit.entry)
        self.assertEqual(matching, (live_exit,))

    def test_global_overlay_preserves_live_door_when_static_entry_conflicts(self):
        source = (1, 4)
        destination = (0, 9)
        live_exit = WorldTransition(
            entry=(source, (6, 16)),
            destination=(destination, (7, 8)),
            activation_locations=frozenset({(source, (6, 17))}),
            activation_direction=Direction.North,
        )
        static_duplicate = WorldTransition(
            entry=(source, (6, 18)),
            destination=(destination, (7, 8)),
            activation_locations=frozenset({(source, (6, 17))}),
            activation_direction=Direction.South,
        )
        observed = NavigationWorld(
            tiles={(source, (6, 17)): NavigableTile((source, (6, 17)), False, frozenset(Direction))},
            warps=(),
            triggers=(),
            bindings=(),
            facing=Direction.North,
            transitions=(live_exit,),
        )
        graph = WorldMapGraph(
            (edge(source, destination, source_coordinate=(6, 18), destination_coordinate=(7, 8)),)
        )

        with patch("modules.navigation.static_map_transitions", return_value=(static_duplicate,)):
            global_world = _global_navigation_world(observed, graph, enrich_maps=())

        matching = tuple(
            transition
            for transition in global_world.transitions
            if transition.destination == static_duplicate.destination
            and (source, (6, 17)) in transition.activation_locations
        )
        self.assertEqual(matching, (live_exit,))

    def test_global_search_uses_live_door_facing_for_conflicting_static_entry(self):
        source = (1, 4)
        destination = (0, 9)
        live_exit = WorldTransition(
            entry=(source, (6, 16)),
            destination=(destination, (7, 8)),
            activation_locations=frozenset({(source, (6, 17))}),
            activation_direction=Direction.North,
        )
        static_duplicate = WorldTransition(
            entry=(source, (6, 18)),
            destination=(destination, (7, 8)),
            activation_locations=frozenset({(source, (6, 17))}),
            activation_direction=Direction.South,
        )
        source_tiles = {
            (source, coordinate): NavigableTile((source, coordinate), False, frozenset(Direction))
            for coordinate in ((6, 19), (6, 18), (6, 17))
        }
        observed = NavigationWorld(
            tiles=source_tiles,
            warps=(),
            triggers=(),
            bindings=(),
            facing=Direction.South,
            transitions=(live_exit,),
        )
        graph = WorldMapGraph(
            (edge(source, destination, source_coordinate=(6, 18), destination_coordinate=(7, 8)),)
        )

        with patch("modules.navigation.static_map_transitions", return_value=(static_duplicate,)):
            global_world = _global_navigation_world(observed, graph, enrich_maps=())

        plan = GoalAwareNavigator(global_world).plan(
            (source, (6, 19)),
            NavigationGoal(SemanticTarget.map(destination)),
        )
        warp_actions = tuple(action for action in plan.actions if action.action_type is NavigationActionType.WARP)
        self.assertEqual(len(warp_actions), 1)
        self.assertEqual(warp_actions[0].source, (source, (6, 17)))
        self.assertEqual(warp_actions[0].direction, Direction.North)

    def test_global_overlay_backfills_empty_current_trainer_geometry_from_rom(self):
        map_id = (90, 1)
        trainer_id = f"trainer:{map_id}:26"
        tiles = tuple(
            TileObservation((map_id, (x, y)), False, frozenset(Direction), elevation=3)
            for x in range(5)
            for y in range(5)
        )
        observed_trigger = TriggerObservation(
            "object:26:Route104_EventScript_LadyCindy",
            locations=frozenset({(map_id, (2, 4))}),
            affordance_id=trainer_id,
            hazard_kind="trainer",
            hazard_locations=frozenset(),
            trainer_defeated=False,
        )
        observed = NavigationWorld(
            tiles={tile.location: NavigableTile(tile.location, False, frozenset(Direction)) for tile in tiles},
            triggers=(observed_trigger,),
        )
        metadata = SimpleNamespace(
            map_size=(5, 5),
            objects=(
                SimpleNamespace(
                    kind="normal",
                    local_id=26,
                    local_coordinates=(2, 4),
                    trainer_type="Normal",
                    trainer_range=3,
                    is_trainer_defeated=False,
                    flag_id=0,
                    movement_type="FACE_UP",
                    script_symbol="Route104_EventScript_LadyCindy",
                ),
            ),
        )
        graph = WorldMapGraph(())

        with (
            patch("modules.map.get_map_metadata", return_value=metadata),
            patch("modules.navigation.prewarm_static_map_observation", return_value=tiles),
            patch("modules.navigation.static_map_transitions", return_value=()),
        ):
            enriched = _global_navigation_world(observed, graph)

        fallback = next(trigger for trigger in enriched.triggers if trigger.trigger_id.startswith("static_object:26:"))
        self.assertEqual(
            fallback.hazard_locations,
            frozenset((map_id, (2, y)) for y in (1, 2, 3)),
        )
        goal = NavigationGoal(
            ReachLocation((map_id, (0, 0))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        plan = GoalAwareNavigator(enriched).plan((map_id, (4, 4)), goal)
        destinations = {action.destination for action in plan.actions if action.destination is not None}
        self.assertTrue(destinations.isdisjoint(fallback.hazard_locations))

    def test_connection_endpoint_requires_static_passability_and_no_object(self):
        tile = SimpleNamespace(accessible_from_direction=[True, False, False, False])
        metadata = SimpleNamespace(objects=())
        self.assertTrue(_connection_endpoint_is_executable(metadata, {(0, 0): tile}, (0, 0)))
        blocked = SimpleNamespace(accessible_from_direction=[False, False, False, False])
        self.assertFalse(_connection_endpoint_is_executable(metadata, {(0, 0): blocked}, (0, 0)))
        tree = SimpleNamespace(local_coordinates=(0, 0))
        self.assertFalse(_connection_endpoint_is_executable(SimpleNamespace(objects=(tree,)), {(0, 0): tile}, (0, 0)))

    def test_map_connection_is_a_first_class_world_transition(self):
        connection = edge((0, 0), (0, 1), kind="connection")
        route = WorldMapGraph((connection,)).route((0, 0), (0, 1))

        self.assertEqual(route.edges[0].kind, "connection")
        self.assertEqual(route.edges[0].source_coordinates, ((0, 0),))

    def test_mixed_and_sequential_transition_route_uses_total_cost(self):
        source, middle, next_map, target = (1, 0), (1, 1), (1, 2), (1, 3)
        graph = WorldMapGraph(
            (
                WorldEdge(source, middle, "warp", ((0, 0),), ((0, 0),), estimated_cost=4),
                WorldEdge(middle, next_map, "connection", ((1, 0),), ((1, 2),), estimated_cost=2),
                WorldEdge(next_map, target, "warp", ((0, 2),), ((0, 0),), estimated_cost=3),
            )
        )

        route = graph.route(source, target)

        self.assertEqual(tuple(edge.kind for edge in route.edges), ("warp", "connection", "warp"))
        self.assertEqual(route.estimated_cost, 9)

    def test_unknown_generic_transition_is_not_given_a_destination(self):
        transition = WorldTransition(((2, 0), (0, 0)), None, kind="map_connection")

        self.assertIsNone(transition.destination)
        self.assertEqual(
            classify_transition_relevance(transition, SemanticTarget.map((2, 1))),
            TransitionRelevance.UNKNOWN,
        )

    def test_simple_two_map_route(self):
        graph = WorldMapGraph((edge((0, 0), (0, 1)),))

        route = graph.route((0, 0), (0, 1))

        self.assertEqual(route.maps, ((0, 0), (0, 1)))
        self.assertEqual(route.edges[0].source_coordinates, ((0, 0),))

    def test_location_estimate_includes_intermediate_map_traversal(self):
        graph = WorldMapGraph(
            (
                edge((0, 0), (0, 1), source_coordinate=(9, 1), destination_coordinate=(0, 5)),
                edge((0, 1), (0, 2), source_coordinate=(8, 5), destination_coordinate=(2, 0)),
            )
        )

        # 9 tiles to the first exit, one transition, 8 tiles across the
        # intermediate map, one transition, then two tiles to the destination.
        self.assertEqual(
            graph.estimate_location_cost(((0, 0), (0, 1)), ((0, 2), (4, 0))),
            21,
        )

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

    def test_emerald_littleroot_to_route101_connection_topology(self):
        littleroot = SimpleNamespace(
            map_size=(6, 4),
            warps=[],
            connections=[
                SimpleNamespace(destination_map_group=0, destination_map_number=16, direction="South", offset=0)
            ],
        )
        route101 = SimpleNamespace(map_size=(6, 5), warps=[], connections=[])

        graph = WorldMapGraph.from_map_data({(0, 9): littleroot, (0, 16): route101})

        route = graph.route((0, 9), (0, 16))
        self.assertEqual(route.maps, ((0, 9), (0, 16)))
        self.assertEqual(route.edges[0].kind, "connection")

    def test_route_uses_estimated_movement_cost_not_warp_count(self):
        source, target = (7, 0), (7, 3)
        direct = edge(source, target)
        direct = WorldEdge(**{**direct.__dict__, "estimated_cost": 50})
        graph = WorldMapGraph(
            (
                direct,
                WorldEdge(source, (7, 1), "warp", ((0, 0),), ((0, 0),), estimated_cost=5),
                WorldEdge((7, 1), (7, 2), "warp", ((0, 0),), ((0, 0),), estimated_cost=5),
                WorldEdge((7, 2), target, "warp", ((0, 0),), ((0, 0),), estimated_cost=5),
            )
        )

        route = graph.route(source, target)

        self.assertEqual(route.maps, (source, (7, 1), (7, 2), target))
        self.assertEqual(route.estimated_cost, 15)

    def test_interaction_route_prefers_interior_warps_over_outdoor_detour(self):
        town, house_1f, house_2f, route101 = (0, 9), (1, 2), (1, 3), (0, 16)
        graph = WorldMapGraph(
            (
                edge(town, house_1f, kind="warp"),
                edge(house_1f, house_2f, kind="warp"),
                edge(town, route101, kind="connection"),
                edge(route101, house_2f, kind="connection"),
            )
        )

        route = graph.route(town, house_2f, prefer_interior=True)

        self.assertEqual(route.maps, (town, house_1f, house_2f))

    def test_cyclic_detour_is_worse_than_direct_transition(self):
        current, target, detour = (8, 0), (8, 2), (8, 1)
        graph = WorldMapGraph(
            (
                edge(current, target),
                edge(detour, current),
                edge(current, target),
            )
        )
        semantic_target = SemanticTarget.map(target)
        direct = WarpObservation((current, (1, 0)), (target, (0, 0)))
        cyclic = WarpObservation((current, (2, 0)), (detour, (0, 0)))

        self.assertEqual(transition_world_route(direct, semantic_target, graph).estimated_cost, 0)
        self.assertGreater(transition_world_route(cyclic, semantic_target, graph).estimated_cost, 0)

    def test_unknown_transition_remains_unknown(self):
        transition = WarpObservation(((8, 0), (1, 0)), None)

        self.assertIs(
            classify_transition_relevance(transition, SemanticTarget.map((8, 2))),
            TransitionRelevance.UNKNOWN,
        )

    def test_step_on_warp_does_not_turn_forever_for_metadata_facing_hint(self):
        source_map = (8, 0)
        entry = (source_map, (1, 0))
        world = NavigationWorld(
            tiles={
                (source_map, (0, 0)): NavigableTile((source_map, (0, 0)), False, frozenset(Direction)),
                entry: NavigableTile(entry, False, frozenset(Direction)),
            },
            warps=(WarpObservation(entry, ((8, 1), (0, 0)), required_facing=Direction.North),),
            facing=Direction.East,
        )

        plan = GoalAwareNavigator(world).plan(
            (source_map, (0, 0)),
            ReachWarp(destination_map=(8, 1), warp=world.warps[0]),
        )

        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].source, entry)

    def test_multi_side_door_uses_selected_activation_side_without_global_direction(self):
        source_map = (8, 4)
        destination_map = (8, 5)
        entry = (source_map, (1, 1))
        north_source = (source_map, (1, 0))
        south_source = (source_map, (1, 2))
        door = WarpObservation(
            entry,
            (destination_map, (1, 1)),
            activation_locations=frozenset({north_source, south_source}),
        )
        world = NavigationWorld(
            tiles={
                location: NavigableTile(location, False, frozenset(Direction))
                for location in (north_source, entry, south_source)
            },
            transitions=(door,),
            facing=Direction.North,
        )

        plan = GoalAwareNavigator(world).plan(
            south_source,
            ReachWarp(destination_map=destination_map, warp=door),
        )

        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].source, south_source)
        self.assertEqual(plan.actions[-1].direction, Direction.North)


class TestWorldGoalIntegration(unittest.TestCase):
    def test_neighbors_index_matches_reference_transition_scan(self):
        source = (1, 4)
        tiles = {
            (source, (x, y)): NavigableTile((source, (x, y)), False, frozenset(Direction))
            for x, y in ((0, 0), (1, 0), (2, 0), (1, 1))
        }
        transitions = (
            WorldTransition((source, (2, 0)), ((0, 18), (1, 1)), kind="warp"),
            WorldTransition((source, (1, 0)), ((0, 19), (2, 2)), kind="map_connection"),
            WorldTransition((source, (1, 0)), ((0, 20), (3, 3)), kind="warp"),
        )
        world = NavigationWorld(tiles=tiles, transitions=transitions, facing=Direction.South)

        def reference(location):
            result = []
            for direction, coordinate in (
                (Direction.North, (location[1][0], location[1][1] - 1)),
                (Direction.East, (location[1][0] + 1, location[1][1])),
                (Direction.South, (location[1][0], location[1][1] + 1)),
                (Direction.West, (location[1][0] - 1, location[1][1])),
            ):
                neighbour = world.tiles.get((location[0], coordinate))
                if neighbour is not None and not neighbour.blocked:
                    result.append((direction, (location[0], coordinate), False))
            for transition in transitions:
                transition_source = transition.entry
                if transition.kind == "map_connection":
                    transition_source = map_connection_approach_position(transition)
                if transition_source == location and transition.destination is not None:
                    result.append((world._warp_direction(location, transition), transition.destination, True))
            return tuple(result)

        for location in ((source, (1, 0)), (source, (2, 0)), (source, (0, 0)), ((9, 9), (0, 0))):
            self.assertEqual(world.neighbors(location), reference(location))

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

    def test_cross_map_search_limit_is_preserved_by_world_diagnostics(self):
        destination_map = (99, 1)
        source = (self.MAP, (0, 0))
        world = NavigationWorld(
            tiles={
                source: NavigableTile(source, allowed_directions=frozenset(Direction)),
            }
        )
        graph = WorldMapGraph((edge(self.MAP, destination_map),))
        with (
            patch("modules.navigation._global_navigation_world", return_value=world),
            patch.object(
                GoalAwareNavigator,
                "plan",
                side_effect=NavigationSearchLimitExceeded("state search budget exhausted"),
            ),
        ):
            with self.assertRaises(NavigationSearchLimitExceeded) as raised:
                plan_with_world_navigation(
                    world,
                    source,
                    ReachLocation((destination_map, (0, 0))),
                    graph,
                )

        self.assertIn("current tile count=1", str(raised.exception))

    def test_bounded_search_reports_expansion_limit(self):
        observed = self.world([(x, 0) for x in range(8)])

        with self.assertRaises(NavigationSearchLimitExceeded):
            GoalAwareNavigator(observed).plan(
                (self.MAP, (0, 0)),
                ReachLocation((self.MAP, (7, 0))),
                max_expansions=2,
            )

    def test_bounded_search_reports_incumbent_cost_ceiling(self):
        observed = self.world([(x, 0) for x in range(5)])

        with self.assertRaises(NavigationSearchLimitExceeded):
            GoalAwareNavigator(observed).plan(
                (self.MAP, (0, 0)),
                ReachLocation((self.MAP, (4, 0))),
                cost_ceiling=(0, 3),
            )

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

    def test_forced_movement_lands_at_endpoint_without_intermediate_exposure(self):
        start = (self.MAP, (0, 0))
        ledge = (self.MAP, (1, 0))
        intermediate = (self.MAP, (2, 0))
        endpoint = (self.MAP, (3, 0))
        world = NavigationWorld(
            tiles={
                start: NavigableTile(start, allowed_directions=frozenset(Direction)),
                ledge: NavigableTile(
                    ledge,
                    allowed_directions=frozenset(Direction),
                    forced_movement_to={Direction.East: (endpoint, 2)},
                ),
                intermediate: NavigableTile(intermediate, allowed_directions=frozenset(Direction), has_encounters=True),
                endpoint: NavigableTile(endpoint, allowed_directions=frozenset(Direction)),
            }
        )

        self.assertEqual(world.neighbors(start), ((Direction.East, endpoint, False),))
        plan = GoalAwareNavigator(world).plan(start, ReachLocation(endpoint), algorithm="astar")
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].destination, endpoint)
        self.assertEqual(plan.metrics.encounter_opportunities, 0)

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

    def test_avoid_prefers_fewer_opportunities_over_shorter_route(self):
        # AVOID must retain the no-encounter route preference when candidate
        # plans are ranked (including cross-map activation candidates).  The
        # clear route is longer, but an encounter opportunity is never a
        # reason to choose the shorter route in AVOID mode.
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
        plan = GoalAwareNavigator(world).plan(
            (self.MAP, (0, 0)),
            NavigationGoal(ReachLocation((self.MAP, (4, 0))), encounter_mode=EncounterMode.AVOID),
        )

        self.assertEqual(plan.metrics.encounter_opportunities, 0)
        self.assertEqual(plan.metrics.movement_actions, 6)

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

    def test_mostly_avoid_trades_encounter_opportunities_for_distance(self):
        # Direct route crosses two grass tiles; the detour is longer but clear.
        coordinates = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
        world = NavigationWorld(
            tiles={
                (self.MAP, coordinate): NavigableTile(
                    (self.MAP, coordinate), False, frozenset(Direction), 1, coordinate in {(1, 0), (2, 0)}
                )
                for coordinate in coordinates
            },
            facing=Direction.East,
        )
        strict = GoalAwareNavigator(world).plan(
            (self.MAP, (0, 0)), NavigationGoal(ReachLocation((self.MAP, (2, 1))), encounter_mode=EncounterMode.AVOID)
        )
        weighted = GoalAwareNavigator(world).plan(
            (self.MAP, (0, 0)),
            NavigationGoal(
                ReachLocation((self.MAP, (2, 1))), encounter_mode=EncounterMode.MOSTLY_AVOID, encounter_penalty=0
            ),
        )
        self.assertEqual(strict.metrics.encounter_opportunities, 0)
        self.assertEqual(strict.metrics.movement_actions, weighted.metrics.movement_actions)

    def test_mostly_avoid_penalty_controls_grass_route_tradeoff(self):
        coordinates = (
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
        )
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
        weighted = GoalAwareNavigator(world).plan(
            (self.MAP, (0, 0)),
            NavigationGoal(
                ReachLocation((self.MAP, (3, 0))),
                encounter_mode=EncounterMode.MOSTLY_AVOID,
                encounter_penalty=3,
            ),
        )
        cheap_grass = GoalAwareNavigator(world).plan(
            (self.MAP, (0, 0)),
            NavigationGoal(
                ReachLocation((self.MAP, (3, 0))),
                encounter_mode=EncounterMode.MOSTLY_AVOID,
                encounter_penalty=1,
            ),
        )

        self.assertEqual(weighted.metrics.encounter_opportunities, 0)
        self.assertTrue(all(action.destination[1] not in grass for action in weighted.actions))
        self.assertEqual(cheap_grass.metrics.encounter_opportunities, 2)
        self.assertEqual(cheap_grass.metrics.total_route_cost, 3)
        self.assertGreater(weighted.metrics.total_route_cost, cheap_grass.metrics.total_route_cost)
        self.assertTrue(any(action.destination[1] in grass for action in cheap_grass.actions))

    def test_encounter_objective_is_translated_to_seek_navigation(self):
        from modules.nuzlocke.campaign_objectives import encounter_task

        task = encounter_task(self.MAP)
        self.assertEqual(task.tactical_target.encounter_mode, EncounterMode.SEEK)

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

    def test_land_navigation_does_not_treat_elevation_one_water_as_walkable(self):
        land = (self.MAP, (0, 0))
        water = (self.MAP, (1, 0))
        world = NavigationWorld(
            tiles={
                land: NavigableTile(land, allowed_directions=frozenset(Direction), elevation=3),
                water: NavigableTile(water, allowed_directions=frozenset(Direction), elevation=1),
            },
            facing=Direction.East,
        )

        self.assertEqual(world.neighbors(land), ())
        with self.assertRaisesRegex(NavigationError, "No legal route"):
            GoalAwareNavigator(world).plan(land, ReachLocation(water))

    def test_existing_surf_state_can_enter_elevation_one_water(self):
        land = (self.MAP, (0, 0))
        water = (self.MAP, (1, 0))
        world = NavigationWorld(
            tiles={
                land: NavigableTile(land, allowed_directions=frozenset(Direction), elevation=3),
                water: NavigableTile(water, allowed_directions=frozenset(Direction), elevation=1),
            },
            facing=Direction.East,
            surfing=True,
        )

        self.assertEqual(world.neighbors(land)[0][:2], (Direction.East, water))

    def test_static_target_reached_interacts_without_runtime_spawn(self):
        map_id = (0, 18)
        location = (map_id, (0, 0))
        binding = TriggerBinding("rival", map_id, "RivalScript", local_id=2)
        observation = AgentObservation(
            InteractionObservation(GameState.OVERWORLD, controllable=True),
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
                        frozenset({location}),
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
        self.assertEqual(selected.action.action_type, AgentActionType.INTERACT)
        self.assertEqual(selected.action.option, "rival")

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
        self.assertEqual(plan.actions[-1].source, (source_map, (2, 0)))
        self.assertEqual(plan.actions[-1].direction, Direction.South)
        self.assertEqual(world.warps[0].activation, WarpActivation.STEP_ON)

    def test_map_connection_action_crosses_boundary_with_direction_and_kind(self):
        source_map = (40, 0)
        target_map = (40, 1)
        boundary = (source_map, (1, 0))
        tiles = tuple(
            TileObservation((source_map, (x, y)), (x, y) == (1, 0), frozenset(Direction))
            for y in range(2)
            for x in range(2)
        )
        connection = MapConnectionObservation(boundary, (target_map, (1, 2)), required_facing=Direction.North)
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            transitions=(connection,),
        )
        graph = WorldMapGraph((edge(source_map, target_map, boundary[1], (1, 2), kind="connection"),))

        plan, _ = plan_with_world_navigation(
            world, (source_map, (0, 0)), ReachLocation((target_map, (1, 2))), graph=graph
        )

        self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[-1].transition_kind, "map_connection")
        self.assertEqual(plan.actions[-1].direction, Direction.North)
        self.assertEqual(plan.actions[-2].destination, boundary)
        self.assertEqual(plan.actions[-1].source, boundary)

    def test_map_connection_approach_geometry_is_generic_and_boundary_independent(self):
        cases = (
            (Direction.North, (3, 0), (3, 1)),
            (Direction.South, (3, 4), (3, 3)),
            (Direction.East, (4, 2), (3, 2)),
            (Direction.West, (0, 2), (1, 2)),
        )
        for direction, boundary, approach in cases:
            with self.subTest(direction=direction):
                self.assertEqual(map_connection_approach_position(boundary, direction), approach)

    def test_map_connection_directions_move_to_boundary_then_cross(self):
        source_map = (40, 2)
        target_map = (40, 3)
        cases = (
            (Direction.North, (2, 0), (2, 1), (2, 4)),
            (Direction.South, (2, 4), (2, 3), (2, 0)),
            (Direction.East, (4, 2), (3, 2), (0, 2)),
            (Direction.West, (0, 2), (1, 2), (4, 2)),
        )
        tiles = tuple(
            TileObservation((source_map, (x, y)), False, frozenset(Direction)) for y in range(5) for x in range(5)
        )
        for direction, boundary, approach, destination in cases:
            with self.subTest(direction=direction):
                connection = MapConnectionObservation(
                    (source_map, boundary),
                    (target_map, destination),
                    required_facing=direction,
                )
                world = NavigationWorld(
                    tiles={
                        tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors)
                        for tile in tiles
                    },
                    transitions=(connection,),
                )
                graph = WorldMapGraph((edge(source_map, target_map, boundary, destination, kind="connection"),))
                plan, _ = plan_with_world_navigation(
                    world,
                    (source_map, (2, 2)),
                    ReachLocation((target_map, destination)),
                    graph=graph,
                )
                self.assertEqual(plan.actions[-1].action_type, NavigationActionType.WARP)
                self.assertEqual(plan.actions[-2].action_type, NavigationActionType.MOVE)
                self.assertEqual(plan.actions[-2].source, (source_map, approach))
                self.assertEqual(plan.actions[-2].destination, (source_map, boundary))
                self.assertEqual(plan.actions[-1].source, (source_map, boundary))
                self.assertEqual(plan.actions[-1].direction, direction)
                self.assertEqual(plan.actions[-1].destination, (target_map, destination))

    def test_map_connection_offset_keeps_source_and_destination_alignment(self):
        source_map = (41, 2)
        target_map = (41, 3)
        boundary = (3, 0)
        destination = (1, 6)
        connection = MapConnectionObservation(
            (source_map, boundary),
            (target_map, destination),
            required_facing=Direction.North,
        )
        tiles = tuple(
            TileObservation((source_map, (x, y)), False, frozenset(Direction)) for y in range(5) for x in range(6)
        )
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            transitions=(connection,),
        )
        graph = WorldMapGraph((edge(source_map, target_map, boundary, destination, kind="connection"),))

        plan, _ = plan_with_world_navigation(
            world,
            (source_map, (3, 2)),
            ReachLocation((target_map, destination)),
            graph=graph,
        )

        self.assertEqual(plan.actions[-2].source, (source_map, (3, 1)))
        self.assertEqual(plan.actions[-1].source, (source_map, boundary))
        self.assertEqual(plan.actions[-1].destination, (target_map, destination))

    def test_littleroot_route101_connection_uses_reachable_north_approach(self):
        source_map = (0, 9)
        target_map = (0, 16)
        boundary = (5, 0)
        destination = (5, 19)
        connection = MapConnectionObservation(
            (source_map, boundary),
            (target_map, destination),
            required_facing=Direction.North,
        )
        tiles = tuple(
            TileObservation((source_map, (x, y)), (x, y) == boundary, frozenset(Direction))
            for y in range(20)
            for x in range(20)
        )
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            transitions=(connection,),
        )
        graph = WorldMapGraph((edge(source_map, target_map, boundary, destination, kind="connection"),))

        plan, _ = plan_with_world_navigation(
            world,
            (source_map, (5, 3)),
            ReachLocation((target_map, destination)),
            graph=graph,
        )

        self.assertEqual(plan.actions[-2].source, (source_map, (5, 1)))
        self.assertEqual(plan.actions[-2].destination, (source_map, boundary))
        self.assertEqual(plan.actions[-1].source, (source_map, boundary))
        self.assertEqual(tuple(action.direction for action in plan.actions[-2:]), (Direction.North, Direction.North))
        self.assertEqual(plan.actions[-1].direction, Direction.North)
        self.assertEqual(plan.actions[-1].destination, (target_map, destination))

    def test_reachable_map_connection_enters_existing_transition_ranking(self):
        from modules.nuzlocke.emerald_capabilities import _observed_exit_goal

        source_map = (43, 0)
        target_map = (43, 1)
        detour_map = (43, 2)
        connection = MapConnectionObservation(
            (source_map, (2, 0)),
            (target_map, (2, 4)),
            required_facing=Direction.North,
        )
        house = WarpObservation((source_map, (0, 2)), (detour_map, (0, 0)))
        tiles = tuple(
            TileObservation((source_map, (x, y)), (x, y) == (2, 0), frozenset(Direction))
            for y in range(5)
            for x in range(5)
        )
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            warps=(house,),
            transitions=(house, connection),
            facing=Direction.South,
        )
        graph = WorldMapGraph(
            (
                WorldEdge(source_map, target_map, "connection", ((2, 0),), ((2, 4),), estimated_cost=10),
                WorldEdge(source_map, detour_map, "warp", ((0, 2),), ((0, 0),), estimated_cost=1),
                WorldEdge(detour_map, target_map, "warp", ((0, 0),), ((0, 0),), estimated_cost=20),
            )
        )

        with patch("modules.navigation.get_world_map_graph", return_value=graph):
            selected = _observed_exit_goal(
                SimpleNamespace(
                    map_id=source_map,
                    player_coordinates=(2, 2),
                    controllable=True,
                    transitions=world.transitions,
                    warps=world.warps,
                ),
                GoalAwareNavigator(world),
                SemanticTarget.map(target_map),
            )

        self.assertIsNotNone(selected)
        self.assertIs(selected.warp, connection)

    def test_map_connection_does_not_require_blocked_boundary_tile(self):
        source_map = (41, 0)
        target_map = (41, 1)
        boundary = (source_map, (2, 0))
        tiles = tuple(
            TileObservation((source_map, (x, y)), (x, y) == boundary[1], frozenset(Direction))
            for y in range(3)
            for x in range(5)
        )
        connection = MapConnectionObservation(
            boundary,
            (target_map, (2, 4)),
            required_facing=Direction.North,
        )
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            transitions=(connection,),
        )
        graph = WorldMapGraph((edge(source_map, target_map, boundary[1], (2, 4), kind="connection"),))

        plan, _ = plan_with_world_navigation(
            world,
            (source_map, (2, 2)),
            ReachLocation((target_map, (2, 4))),
            graph=graph,
        )

        self.assertEqual(plan.actions[-2].source, (source_map, (2, 1)))
        self.assertEqual(plan.actions[-2].destination, boundary)
        self.assertEqual(plan.actions[-1].source, boundary)
        self.assertEqual(plan.actions[-1].direction, Direction.North)
        self.assertEqual(plan.actions[-1].transition_kind, "map_connection")

    def test_reach_warp_satisfies_map_connection_from_transitions(self):
        source_map = (42, 0)
        target_map = (42, 1)
        boundary = (source_map, (1, 0))
        approach = (source_map, (1, 1))
        connection = MapConnectionObservation(
            boundary,
            (target_map, (1, 4)),
            required_facing=Direction.North,
        )
        world = NavigationWorld(
            tiles={
                (source_map, (0, 1)): NavigableTile((source_map, (0, 1)), False, frozenset(Direction)),
                approach: NavigableTile(approach, False, frozenset(Direction)),
                boundary: NavigableTile(boundary, True, frozenset()),
            },
            transitions=(connection,),
            facing=Direction.East,
        )

        plan = GoalAwareNavigator(world).plan(
            (source_map, (0, 1)),
            ReachWarp(destination_map=target_map, destination=connection.destination, warp=connection),
        )

        self.assertEqual(plan.destination, connection.destination)

    def test_observed_map_connection_at_boundary_crosses_without_reverse_move(self):
        source_map = (42, 2)
        target_map = (42, 3)
        boundary = (source_map, (3, 0))
        connection = MapConnectionObservation(
            boundary,
            (target_map, (3, 4)),
            required_facing=Direction.North,
        )
        world = NavigationWorld(
            tiles={boundary: NavigableTile(boundary, True, frozenset())},
            transitions=(connection,),
            facing=Direction.North,
        )

        plan = plan_observed_warp_locally(
            world,
            boundary,
            ReachWarp(destination_map=target_map, destination=connection.destination, warp=connection),
        )

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[0].source, boundary)
        self.assertEqual(plan.actions[0].direction, Direction.North)

    def test_observed_step_on_warp_at_entry_still_emits_crossing_action(self):
        source_map = (42, 4)
        target_map = (42, 5)
        entry = (source_map, (2, 2))
        warp = WarpObservation(entry, (target_map, (2, 3)), activation=WarpActivation.STEP_ON)
        world = NavigationWorld(
            tiles={entry: NavigableTile(entry, False, frozenset(Direction))},
            warps=(warp,),
            facing=Direction.South,
        )

        plan = plan_observed_warp_locally(
            world,
            entry,
            ReachWarp(destination_map=target_map, destination=warp.destination, warp=warp),
        )

        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].action_type, NavigationActionType.WARP)
        self.assertEqual(plan.actions[0].source, entry)

    def test_world_route_step_on_from_south_emits_north_entry_input(self):
        """The cross-map planner must preserve STEP_ON approach geometry."""
        source_map = (0, 10)
        target_map = (0, 11)
        entry = (source_map, (2, 2))
        warp = WarpObservation(entry, (target_map, (1, 3)), activation=WarpActivation.STEP_ON)
        tiles = tuple(
            TileObservation((source_map, (x, y)), False, frozenset(Direction)) for y in range(5) for x in range(5)
        )
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            warps=(warp,),
            facing=Direction.North,
        )
        graph = WorldMapGraph((edge(source_map, target_map, source_coordinate=(2, 2), destination_coordinate=(1, 3)),))

        plan, _ = plan_with_world_navigation(
            world,
            (source_map, (2, 3)),
            ReachWarp(destination_map=target_map, destination=warp.destination, warp=warp),
            graph=graph,
        )

        transition = plan.actions[-1]
        self.assertEqual(transition.action_type, NavigationActionType.WARP)
        self.assertEqual(transition.source, entry)
        self.assertEqual(transition.direction, Direction.North)

    def test_arrow_warp_steps_onto_tile_then_uses_required_facing(self):
        source_map = (0, 0)
        target_map = (0, 1)
        tiles = tuple(TileObservation((source_map, (x, 0)), False, frozenset(Direction)) for x in range(3))
        world = NavigationWorld(
            tiles={
                tile.location: NavigableTile(tile.location, tile.blocked, tile.walkable_neighbors) for tile in tiles
            },
            warps=(
                WarpObservation(
                    (source_map, (2, 0)),
                    (target_map, (0, 0)),
                    Direction.North,
                    activation=WarpActivation.DIRECTIONAL_STEP,
                ),
            ),
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
                InteractionObservation(GameState.OVERWORLD, controllable=True),
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


class TestTrainerAvoidance(unittest.TestCase):
    MAP = (90, 0)

    @classmethod
    def _world(cls, coordinates, hazards=()):
        tiles = {
            (cls.MAP, coordinate): NavigableTile((cls.MAP, coordinate), allowed_directions=frozenset(Direction))
            for coordinate in coordinates
        }
        trainer = TriggerObservation(
            "trainer:1",
            locations=frozenset({(cls.MAP, (1, 0))}),
            kind="trainer",
            hazard_locations=frozenset((cls.MAP, coordinate) for coordinate in hazards),
            hazard_kind="trainer",
        )
        return NavigationWorld(tiles=tiles, triggers=(trainer,))

    def test_avoid_selects_longer_trainer_free_detour(self):
        coordinates = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
        world = self._world(coordinates, hazards=((1, 0),))
        goal = NavigationGoal(
            ReachLocation((self.MAP, (2, 0))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), goal)
        destinations = {action.destination for action in plan.actions if action.destination is not None}
        self.assertNotIn((self.MAP, (1, 0)), destinations)
        self.assertFalse(plan.forced_trainer_exposure)
        self.assertGreater(len(plan.actions), 2)

    def test_route_104_shape_avoids_lady_cindy_sight_line(self):
        route104 = MapRSE.ROUTE104.value
        coordinates = {
            (x, y)
            for x in range(10, 15)
            for y in range(40, 45)
        }
        tiles = {
            (route104, coordinate): NavigableTile(
                (route104, coordinate),
                allowed_directions=frozenset(Direction),
            )
            for coordinate in coordinates
        }
        cindy = TriggerObservation(
            "static_object:26:Route104_EventScript_LadyCindy",
            locations=frozenset({(route104, (11, 44))}),
            affordance_id=f"trainer:{route104}:26",
            kind="trainer_hazard",
            hazard_kind="trainer",
            hazard_locations=frozenset(
                (route104, coordinate) for coordinate in ((11, 41), (11, 42), (11, 43))
            ),
        )
        world = NavigationWorld(tiles=tiles, triggers=(cindy,))
        goal = NavigationGoal(
            ReachLocation((route104, (10, 40))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        plan = GoalAwareNavigator(world).plan((route104, (13, 43)), goal)

        destinations = {
            action.destination
            for action in plan.actions
            if action.destination is not None
        }
        self.assertTrue(plan.actions)
        self.assertTrue(destinations.isdisjoint(cindy.hazard_locations))
        self.assertFalse(plan.forced_trainer_exposure)

    def test_route_104_trainer_free_grass_corridor_avoids_cindy_and_billy(self):
        """Route 104's safe approach may cross grass to avoid both trainers."""

        route104 = MapRSE.ROUTE104.value
        woods = MapRSE.PETALBURG_WOODS.value
        # This fixture mirrors the real southern Route 104 approach shape:
        # the paved middle lane is watched by Cindy/Billy, while the upper
        # grass strip reaches the Woods boundary without entering either
        # sight line.  Keep the real map identities and a real Woods exit so
        # the test exercises the same cross-map goal contract.
        coordinates = {
            (x, y)
            for y in range(0, 5)
            for x in range(0, 7)
        }
        grass = {(x, 0) for x in range(1, 6)}
        tiles = {
            (route104, coordinate): NavigableTile(
                (route104, coordinate),
                allowed_directions=frozenset(Direction),
                has_encounters=coordinate in grass,
            )
            for coordinate in coordinates
        }
        cindy_hazards = frozenset(
            (route104, coordinate) for coordinate in ((2, 2), (2, 3), (2, 4))
        )
        billy_hazards = frozenset(
            (route104, coordinate) for coordinate in ((4, 2), (4, 3), (4, 4))
        )
        triggers = (
            TriggerObservation(
                "static_object:26:Route104_EventScript_LadyCindy",
                locations=frozenset({(route104, (2, 4))}),
                affordance_id=f"trainer:{route104}:26",
                kind="trainer_hazard",
                hazard_kind="trainer",
                hazard_locations=cindy_hazards,
            ),
            TriggerObservation(
                "static_object:28:Route104_EventScript_Billy",
                locations=frozenset({(route104, (4, 4))}),
                affordance_id=f"trainer:{route104}:28",
                kind="trainer_hazard",
                hazard_kind="trainer",
                hazard_locations=billy_hazards,
            ),
        )
        exit_transition = MapConnectionObservation(
            entry=(route104, (6, 0)),
            destination=(woods, (17, 38)),
            required_facing=Direction.East,
            activation_locations=frozenset({(route104, (5, 0))}),
        )
        world = NavigationWorld(tiles=tiles, transitions=(exit_transition,), triggers=triggers)
        goal = NavigationGoal(
            ReachWarp(destination_map=woods, warp=exit_transition),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
            encounter_mode=EncounterMode.AVOID,
        )

        plan = GoalAwareNavigator(world).plan((route104, (0, 2)), goal)

        destinations = {
            action.destination
            for action in plan.actions
            if action.destination is not None
        }
        self.assertTrue(destinations & {(route104, coordinate) for coordinate in grass})
        self.assertTrue(destinations.isdisjoint(cindy_hazards | billy_hazards))
        self.assertFalse(plan.forced_trainer_exposure)

    def test_route_104_avoid_does_not_complete_on_trainer_hazard_warp_approach(self):
        route104 = MapRSE.ROUTE104.value
        woods = MapRSE.PETALBURG_WOODS.value
        approach = (route104, (1, 0))
        boundary = (route104, (2, 0))
        tiles = {
            (route104, coordinate): NavigableTile(
                (route104, coordinate),
                allowed_directions=frozenset(Direction),
            )
            for coordinate in ((0, 0), (1, 0), (2, 0))
        }
        transition = MapConnectionObservation(
            entry=boundary,
            destination=(woods, (17, 38)),
            required_facing=Direction.East,
            activation_locations=frozenset({approach}),
        )
        trainer = TriggerObservation(
            "static_object:28:Route104_EventScript_Billy",
            locations=frozenset({(route104, (2, 1))}),
            affordance_id=f"trainer:{route104}:28",
            hazard_kind="trainer",
            hazard_locations=frozenset({approach}),
        )
        world = NavigationWorld(tiles=tiles, transitions=(transition,), triggers=(trainer,))
        goal = NavigationGoal(
            ReachWarp(destination_map=woods, warp=transition),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        with self.assertRaises(NavigationError):
            GoalAwareNavigator(world).plan((route104, (0, 0)), goal)

        local_world = NavigationWorld(tiles=tiles, triggers=(trainer,))
        self.assertEqual(
            GoalAwareNavigator(local_world).plan_many_locations(
                (route104, (0, 0)),
                (approach,),
                constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
                encounter_mode=EncounterMode.AVOID,
            ),
            {},
        )

    def test_avoid_rejects_forced_trainer_exposure_without_explicit_opt_in(self):
        world = self._world(((0, 0), (1, 0), (2, 0)), hazards=((1, 0),))
        goal = NavigationGoal(
            ReachLocation((self.MAP, (2, 0))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )
        with self.assertRaises(NavigationError):
            GoalAwareNavigator(world).plan((self.MAP, (0, 0)), goal)

    def test_avoid_treats_trainer_approach_positions_as_hazards(self):
        coordinates = (
            (0, 1),
            (1, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        )
        tiles = {
            (self.MAP, coordinate): NavigableTile((self.MAP, coordinate), allowed_directions=frozenset(Direction))
            for coordinate in coordinates
        }
        trainer = TriggerObservation(
            "trainer:approach",
            locations=frozenset({(self.MAP, (1, 0))}),
            activation_locations=frozenset({(self.MAP, (1, 1))}),
            hazard_kind="trainer",
        )
        world = NavigationWorld(tiles=tiles, triggers=(trainer,))
        goal = NavigationGoal(
            ReachLocation((self.MAP, (2, 1))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 1)), goal)

        destinations = {action.destination for action in plan.actions if action.destination is not None}
        self.assertNotIn((self.MAP, (1, 1)), destinations)
        self.assertGreater(len(plan.actions), 2)

    def test_avoid_allows_passing_known_trainer_activation_position(self):
        """Known sight lines constrain movement; adjacent A positions do not."""

        coordinates = {
            (x, y)
            for x in range(4)
            for y in range(2)
        }
        tiles = {
            (self.MAP, coordinate): NavigableTile(
                (self.MAP, coordinate),
                allowed_directions=frozenset(Direction),
            )
            for coordinate in coordinates
        }
        trainer = TriggerObservation(
            "trainer:known-envelope",
            locations=frozenset({(self.MAP, (1, 1))}),
            activation_locations=frozenset({(self.MAP, (1, 0))}),
            hazard_kind="trainer",
            hazard_locations=frozenset({(self.MAP, (2, 1))}),
        )
        world = NavigationWorld(tiles=tiles, triggers=(trainer,))
        goal = NavigationGoal(
            ReachLocation((self.MAP, (3, 0))),
            constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
        )

        plan = GoalAwareNavigator(world).plan((self.MAP, (0, 0)), goal)

        destinations = {action.destination for action in plan.actions if action.destination is not None}
        self.assertIn((self.MAP, (1, 0)), destinations)
        self.assertTrue(destinations.isdisjoint(trainer.hazard_locations))

    def test_engage_trainer_uses_facing_line_instead_of_broad_hazards(self):
        coordinates = {
            (19, 16),
            (20, 16),
            (21, 16),
            (22, 16),
            (19, 17),
            (20, 17),
            (21, 17),
            (22, 17),
        }
        tiles = {
            (self.MAP, coordinate): NavigableTile((self.MAP, coordinate), allowed_directions=frozenset(Direction))
            for coordinate in coordinates
        }
        trainer = TriggerObservation(
            "trainer:1",
            locations=frozenset({(self.MAP, (22, 16))}),
            kind="trainer",
            affordance_id="trainer:(90, 0):1",
            hazard_locations=frozenset({(self.MAP, (19, 16)), (self.MAP, (22, 17))}),
            engagement_locations=frozenset({(self.MAP, (22, 17))}),
            hazard_kind="trainer",
        )
        world = NavigationWorld(tiles=tiles, triggers=(trainer,))
        goal = NavigationGoal(EngageTrainer("trainer:(90, 0):1"))
        navigator = GoalAwareNavigator(world)

        self.assertFalse(navigator.satisfies((self.MAP, (19, 16)), Direction.East, goal.target))
        self.assertTrue(navigator.satisfies((self.MAP, (22, 17)), Direction.South, goal.target))
        plan = navigator.plan((self.MAP, (19, 16)), goal)

        self.assertTrue(plan.actions)
        self.assertEqual(plan.destination, (self.MAP, (22, 17)))

    def test_engage_trainer_routes_to_stable_map_before_trigger_is_observed(self):
        source_map = (1, 1)
        trainer_map = (1, 2)
        source = (source_map, (0, 0))
        transition = WorldTransition(
            entry=(source_map, (1, 0)),
            destination=(trainer_map, (0, 0)),
            activation_locations=(source,),
            activation_direction=Direction.East,
        )
        world = NavigationWorld(
            tiles={
                source: NavigableTile(source, False, frozenset(Direction)),
                (source_map, (1, 0)): NavigableTile(
                    (source_map, (1, 0)), False, frozenset(Direction)
                ),
            },
            transitions=(transition,),
            facing=Direction.East,
            static_metadata_frozen=True,
        )
        graph = WorldMapGraph(
            (
                edge(
                    source_map,
                    trainer_map,
                    source_coordinate=(1, 0),
                    destination_coordinate=(0, 0),
                ),
            )
        )
        goal = NavigationGoal(EngageTrainer("trainer:(1, 2):7"))

        self.assertEqual(goal_target_map(world, goal), trainer_map)
        plan, route = plan_with_world_navigation(world, source, goal, graph)

        self.assertEqual(route.maps, (source_map, trainer_map))
        self.assertEqual(plan.destination, (trainer_map, (0, 0)))
        self.assertTrue(any(action.action_type is NavigationActionType.WARP for action in plan.actions))


def test_exact_cross_map_warp_chain_composes_local_segments():
    source_map = (0, 31)
    city_map = (0, 3)
    interior_map = (11, 5)
    tiles = {
        (map_id, coordinate): NavigableTile(
            (map_id, coordinate),
            allowed_directions=frozenset(Direction),
        )
        for map_id in (source_map, city_map)
        for coordinate in ((0, 1), (1, 1), (2, 1))
    }
    connection = WorldTransition(
        (source_map, (2, 1)),
        (city_map, (0, 1)),
        required_facing=Direction.East,
        kind="map_connection",
    )
    center_door = WorldTransition((city_map, (1, 1)), (interior_map, (7, 8)))
    world = NavigationWorld(
        tiles=tiles,
        transitions=(connection, center_door),
        facing=Direction.East,
    )
    route = WorldRoute(
        (source_map, city_map, interior_map),
        (
            WorldEdge(
                source_map,
                city_map,
                "connection",
                ((2, 1),),
                ((0, 1),),
                required_facing=Direction.East,
            ),
            WorldEdge(city_map, interior_map, "warp", ((1, 1),), ((7, 8),)),
        ),
    )

    plan = _plan_exact_cross_map_warp_chain(
        world,
        (source_map, (0, 1)),
        NavigationGoal(ReachWarp(destination_map=interior_map, warp=center_door)),
        route,
    )

    assert plan is not None
    assert plan.destination == center_door.destination
    assert [action.action_type for action in plan.actions].count(NavigationActionType.WARP) == 2


def test_exact_cross_map_warp_chain_shares_local_search_for_connection_candidates():
    source_map = (0, 31)
    city_map = (0, 3)
    interior_map = (11, 5)
    tiles = {
        (map_id, coordinate): NavigableTile(
            (map_id, coordinate),
            allowed_directions=frozenset(Direction),
        )
        for map_id in (source_map, city_map)
        for coordinate in ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (0, 3), (1, 3), (2, 3))
    }
    near = WorldTransition(
        (source_map, (0, 1)),
        (city_map, (0, 1)),
        required_facing=Direction.West,
        kind="map_connection",
    )
    far = WorldTransition(
        (source_map, (0, 3)),
        (city_map, (0, 3)),
        required_facing=Direction.West,
        kind="map_connection",
    )
    center_door = WorldTransition((city_map, (1, 1)), (interior_map, (7, 8)))
    world = NavigationWorld(
        tiles=tiles,
        transitions=(far, near, center_door),
        facing=Direction.West,
    )
    local_world = NavigationWorld(tiles=tiles, facing=Direction.West)
    shared = GoalAwareNavigator(local_world).plan_many_locations(
        (source_map, (4, 1)),
        ((source_map, (1, 1)), (source_map, (1, 3))),
        target_groups=(((source_map, (1, 1)), (source_map, (1, 3))),),
    )
    assert shared
    center_local = NavigationWorld(tiles=tiles, transitions=(center_door,), facing=Direction.West)
    assert plan_observed_warp_locally(
        center_local,
        (city_map, (0, 1)),
        NavigationGoal(ReachWarp(destination_map=interior_map, warp=center_door)),
    )
    route = WorldRoute(
        (source_map, city_map, interior_map),
        (
            WorldEdge(
                source_map,
                city_map,
                "connection",
                ((0, 3), (0, 1)),
                ((0, 3), (0, 1)),
                required_facing=Direction.West,
            ),
            WorldEdge(city_map, interior_map, "warp", ((1, 1),), ((7, 8),)),
        ),
    )

    plan = _plan_exact_cross_map_warp_chain(
        world,
        (source_map, (4, 1)),
        NavigationGoal(ReachWarp(destination_map=interior_map, warp=center_door)),
        route,
    )

    assert plan is not None
    connection_warps = tuple(
        action for action in plan.actions if action.transition_kind == "map_connection"
    )
    assert connection_warps
    assert connection_warps[0].source == (source_map, (0, 1))


if __name__ == "__main__":
    unittest.main()
