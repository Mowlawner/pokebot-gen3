"""Explicit overworld perception models and the live emulator adapter."""

from dataclasses import dataclass, replace
from enum import Enum, auto
from modules.context import context
from modules.map import observe_live_map_identity, get_map_metadata, get_map_objects
from modules.game import get_event_var_name
from modules.memory import get_event_flag, get_event_flag_by_number, get_event_var_by_number, get_game_state_symbol
from modules.tasks import task_is_active
from modules.map_path import Direction, _get_map_metadata
from modules.player import get_player_avatar, player_avatar_is_controllable
from modules.trigger_bindings import BindingResolution, TRIGGER_BINDINGS, resolve_trigger_binding
from modules.console import profile_print
from modules.profiler import count, enabled, format_snapshot, now, timing
from modules.stutter_trace import traced
from modules.state_cache import state_cache

MapId = tuple[int, int]
Coordinate = tuple[int, int]
Location = tuple[MapId, Coordinate]


class MovementState(Enum):
    TURNING = auto()
    MOVING = auto()
    STANDING = auto()


# Emerald's MB_COUNTER value from include/constants/metatile_behaviors.h.
EMERALD_MB_COUNTER = 0x80


# Readiness and campaign capability execution run back-to-back during one
# application frame. Keep that frame's passive overworld observation available
# to the capability so it does not rebuild the same map/object/trigger model a
# second time before issuing its first action.
_shared_overworld_observation_frame = None
_shared_overworld_observation_emulator = None
_shared_overworld_observation_avatar_reader = None
_shared_overworld_observation = None


def _current_emulator_frame():
    """Return the live emulator frame used to scope passive observations."""

    emulator = getattr(context, "emulator", None)
    get_frame_count = getattr(emulator, "get_frame_count", None)
    if not callable(get_frame_count):
        return None
    try:
        frame = get_frame_count()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    # A test double without a concrete frame must not accidentally turn a
    # process-wide observation into a cache.  The real adapter returns an
    # int, and accepting bool here would be equally misleading.
    return frame if isinstance(frame, int) and not isinstance(frame, bool) else None


def invalidate_shared_overworld_observation() -> None:
    """Discard the passive world read after an explicit runtime invalidation."""

    global _shared_overworld_observation_frame, _shared_overworld_observation_emulator, _shared_overworld_observation_avatar_reader, _shared_overworld_observation
    _shared_overworld_observation_frame = None
    _shared_overworld_observation_emulator = None
    _shared_overworld_observation_avatar_reader = None
    _shared_overworld_observation = None


def publish_shared_overworld_observation(observation) -> None:
    """Publish a passive overworld read for another owner in this frame."""

    global _shared_overworld_observation_frame, _shared_overworld_observation_emulator, _shared_overworld_observation_avatar_reader, _shared_overworld_observation
    frame = _current_emulator_frame()
    if frame is None:
        return
    _shared_overworld_observation_frame = frame
    _shared_overworld_observation_emulator = getattr(context, "emulator", None)
    _shared_overworld_observation_avatar_reader = get_player_avatar
    _shared_overworld_observation = observation


def shared_overworld_observation_for_current_frame():
    """Return the published passive overworld read for the current frame."""

    frame = _current_emulator_frame()
    if frame is None:
        return None
    if (
        frame != _shared_overworld_observation_frame
        or getattr(context, "emulator", None) is not _shared_overworld_observation_emulator
        or get_player_avatar is not _shared_overworld_observation_avatar_reader
    ):
        return None
    return _shared_overworld_observation


def _running_shoes_received() -> bool:
    """Return the game-specific progression flag for running shoes."""
    rom = getattr(context, "rom", None)
    if rom is None:
        return False
    flag = "HIDE_PEWTER_CITY_RUNNING_SHOES_GUY" if rom.is_frlg else "RECEIVED_RUNNING_SHOES"
    try:
        return get_event_flag(flag)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False


class WarpActivation(Enum):
    STEP_ON = auto()
    DIRECTIONAL_STEP = auto()


class TransitionMechanism(Enum):
    ORDINARY_WARP = "ordinary_warp"
    MAP_CONNECTION = "map_connection"
    ARROW_WARP = "arrow_warp"
    DOOR_WARP = "door_warp"
    ESCALATOR_WARP = "escalator_warp"
    SCRIPTED = "scripted"


class TransitionActivationMode(Enum):
    ENTRY_TILE = "stepping_on_entry_tile"
    DIRECTIONAL_INPUT = "directional_input"
    FIELD_EFFECT = "field_effect"
    MAP_BOUNDARY = "map_boundary"
    SCRIPT = "script"


class TransitionGeometry(Enum):
    ENTRY_TILE = "entry_tile"
    APPROACH_TILE = "approach_tile"
    DESTINATION = "destination"


class OverworldObservationStatus(Enum):
    VALID = "valid"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class OverworldObservationResult:
    status: OverworldObservationStatus
    observation: "OverworldObservation | None" = None
    reason: str | None = None


@dataclass(frozen=True)
class WorldTransition:
    """A locally executable cross-map transition.

    The mechanism is deliberately retained: a map connection is a boundary
    crossing, while a warp is an entry-tile transition.  ``destination=None``
    is an unresolved dynamic transition, never an inferred destination.
    """

    entry: Location
    destination: Location | None
    required_facing: Direction | None = None
    activation: WarpActivation = WarpActivation.STEP_ON
    # A warp entry is not necessarily the tile from which the activating
    # input is issued.  ROM terrain such as escalators has one valid source
    # tile, even though the entry tile is geometrically reachable from more
    # than one side.
    activation_locations: frozenset[Location] = frozenset()
    activation_direction: Direction | None = None
    kind: str = "warp"
    mechanism: TransitionMechanism = TransitionMechanism.ORDINARY_WARP
    activation_mode: TransitionActivationMode = TransitionActivationMode.ENTRY_TILE
    geometry: TransitionGeometry = TransitionGeometry.ENTRY_TILE


@dataclass(frozen=True)
class WarpObservation(WorldTransition):
    """Compatibility name for ordinary ROM warp observations."""

    kind: str = "warp"

    @property
    def transition_type(self) -> TransitionMechanism:
        return self.mechanism


@dataclass(frozen=True)
class MapConnectionObservation(WorldTransition):
    """A boundary crossing derived from a ROM MapConnection."""

    kind: str = "map_connection"
    mechanism: TransitionMechanism = TransitionMechanism.MAP_CONNECTION
    activation_mode: TransitionActivationMode = TransitionActivationMode.MAP_BOUNDARY
    geometry: TransitionGeometry = TransitionGeometry.APPROACH_TILE


@dataclass(frozen=True)
class TriggerObservation:
    trigger_id: str
    locations: frozenset[Location]
    activation_locations: frozenset[Location] = frozenset()
    kind: str = "script"
    target_map: MapId | None = None
    navigation_locations: frozenset[Location] = frozenset()
    # Per activation-position orientation requirements.  This is a mapping
    # rather than one direction because an object can be approached from any
    # of its four adjacent tiles, with a different required facing at each.
    activation_requirements: tuple[tuple[Location, Direction], ...] = ()
    script_symbol: str | None = None
    condition_variable_number: int | None = None
    condition_variable: str | None = None
    condition_required_value: int | None = None
    condition_current_value: int | None = None
    condition_active: bool | None = None
    currently_actionable: bool | None = None
    # Stable semantic identity used to connect a campaign interaction target
    # to a currently observed affordance.  Navigation remains agnostic to the
    # consequence of activating it.
    affordance_id: str | None = None
    # Dynamic trainer sight-line hazard.  These are player positions that can
    # activate the trainer, distinct from the adjacent interaction positions.
    hazard_locations: frozenset[Location] = frozenset()
    hazard_kind: str | None = None
    # Coordinate scripts are dispatched by the ROM when the player enters
    # their tile. Object and background events, by contrast, require the
    # interaction button. Keep this execution distinction explicit so a
    # semantic campaign target cannot turn an automatic scene into repeated
    # A presses. These fields are appended to preserve the legacy positional
    # constructor order used by navigation fixtures.
    elevation: int | None = None
    requires_input: bool = True


@dataclass(frozen=True)
class ObjectObservation:
    local_id: int
    location: Location
    # ObjectEvent has no category/kind field.  Keep this nullable until a
    # domain-level category can be derived without guessing from graphics IDs.
    kind: str | None = None
    script: str = ""
    previous_location: Location | None = None
    facing: str | None = None
    movement_type: str | None = None
    movement_action: str | None = None
    script_controlled: bool | None = None
    visibility_flag_id: int | None = None
    trainer_type: str | None = None
    trainer_range: int | None = None
    trainer_defeated: bool | None = None
    interactable: bool = True
    elevation: int | None = None

    @property
    def trainer_id(self) -> str | None:
        if self.trainer_type in (None, "None"):
            return None
        return f"trainer:{self.location[0]}:{self.local_id}"


def resolve_object_activation_positions(
    object_observation: ObjectObservation,
    objects: tuple[ObjectObservation, ...] | list[ObjectObservation],
    tiles: tuple["TileObservation", ...] | list["TileObservation"],
    map_size: tuple[int, int],
) -> tuple[tuple[Location, Direction], ...]:
    """Return player states that Emerald's object resolver can activate.

    Emerald checks the tile in front of the player.  If that tile is a
    counter and contains no object, it checks one additional tile in the same
    direction.  Collision alone deliberately has no effect here.
    """
    x, y = object_observation.location[1]
    tile_by_location = {tile.location: tile for tile in tiles}
    object_locations = {(obj.location, obj.elevation) for obj in objects}
    candidates = (
        (Direction.North, (0, -1)),
        (Direction.East, (1, 0)),
        (Direction.South, (0, 1)),
        (Direction.West, (-1, 0)),
    )
    result: list[tuple[Location, Direction]] = []
    for facing, vector in candidates:
        ordinary = (x - vector[0], y - vector[1])
        counter = (x - vector[0], y - vector[1])
        counter_location = (object_observation.location[0], counter)
        counter_tile = tile_by_location.get(counter_location)
        counter_has_object = (counter_location, object_observation.elevation) in object_locations
        if counter_tile is not None and counter_tile.metatile_behavior == EMERALD_MB_COUNTER and not counter_has_object:
            activation = (ordinary[0] - vector[0], ordinary[1] - vector[1])
        else:
            activation = ordinary
        if 0 <= activation[0] < map_size[0] and 0 <= activation[1] < map_size[1]:
            # Emerald's elevation check applies to the object lookup, not to
            # the player's standing tile.  A counter-separated NPC can be on
            # a different tile-elevation layer from the player.
            result.append(((object_observation.location[0], activation), facing))
    return tuple(result)


def trainer_hazard_locations(
    trainer: ObjectObservation,
    tiles: tuple["TileObservation", ...] | list["TileObservation"],
    objects: tuple[ObjectObservation, ...] | list[ObjectObservation] = (),
) -> frozenset[Location]:
    """Return player locations visible to an undefeated trainer.

    This deliberately models only cardinal sight lines.  It is deterministic,
    independent of emulator state, and conservative when collision data is
    incomplete.
    """
    if trainer.trainer_type in (None, "None") or trainer.trainer_defeated is not False:
        return frozenset()
    tile_by_location = {tile.location: tile for tile in tiles}
    occupied = {obj.location for obj in objects if obj.location != trainer.location}
    directions = (
        tuple(Direction)
        if trainer.trainer_type == "See All Directions"
        else (Direction.from_string(trainer.facing or "South"),)
    )
    vectors = {Direction.North: (0, -1), Direction.East: (1, 0), Direction.South: (0, 1), Direction.West: (-1, 0)}
    x, y = trainer.location[1]
    result: set[Location] = set()
    for direction in directions:
        dx, dy = vectors[direction]
        for distance in range(1, (trainer.trainer_range or 0) + 1):
            location = (trainer.location[0], (x + dx * distance, y + dy * distance))
            tile = tile_by_location.get(location)
            if tile is None or tile.blocked or location in occupied:
                break
            result.add(location)
    return frozenset(result)


def static_trainer_observations(
    map_id: MapId,
    map_metadata,
    runtime_objects: tuple[ObjectObservation, ...] | list[ObjectObservation] = (),
) -> tuple[ObjectObservation, ...]:
    """Describe trainer hazards whose runtime object has not spawned yet.

    Emerald can populate ``gObjectEvents`` lazily as the camera approaches an
    object.  That is too late for avoidance: a route may already have crossed
    the trainer's sight line by the time the object becomes visible.  The map
    template is ROM-owned and supplies the trainer type, initial coordinate,
    range, direction, and defeat state, so it is a safe fallback for hazards.

    These observations are intentionally separate from the returned runtime
    object list.  A template is not evidence that the object currently blocks
    collision or that a semantic binding has a live object match.
    """
    runtime_ids = {object_observation.local_id for object_observation in runtime_objects}
    result: list[ObjectObservation] = []
    for template in getattr(map_metadata, "objects", ()):
        try:
            if getattr(template, "kind", "normal") != "normal":
                continue
            local_id = template.local_id
            trainer_type = template.trainer_type
            if local_id in runtime_ids or trainer_type in (None, "None", "???"):
                continue
            # A set template flag means the object is hidden.  If the flag
            # cannot be read at this boundary, retain the hazard: an unknown
            # visibility result must not silently expose the player to battle.
            flag_id = getattr(template, "flag_id", 0) or 0
            if flag_id:
                try:
                    if get_event_flag_by_number(flag_id):
                        continue
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                    pass
            defeated = template.is_trainer_defeated
            movement_type = str(getattr(template, "movement_type", ""))
            facing = None
            if trainer_type != "See All Directions":
                # FACE_DOWN_AND_UP and similar movement types begin in the
                # first listed direction; that is the object-event spawn
                # orientation used by the ROM before its first turn.
                direction_name = movement_type.removeprefix("FACE_").split("_")[0].title()
                if direction_name in {"Up", "Down", "Left", "Right"}:
                    facing = direction_name
            result.append(
                ObjectObservation(
                    local_id=local_id,
                    location=(map_id, template.local_coordinates),
                    script=getattr(template, "script_symbol", ""),
                    previous_location=(map_id, template.local_coordinates),
                    facing=facing,
                    movement_type=movement_type,
                    trainer_type=trainer_type,
                    trainer_range=template.trainer_range,
                    trainer_defeated=defeated,
                    interactable=True,
                    elevation=getattr(template, "elevation", None),
                )
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            # Partial map metadata is common during a map transition.  It is
            # better to omit one unresolved fallback than to make perception
            # fail and lose all of the other current-frame evidence.
            continue
    return tuple(result)


def evaluate_trigger_condition(trigger: TriggerObservation, current_value: int | None) -> bool | None:
    """Evaluate a normalized trigger condition without interpreting its script."""
    if trigger.condition_required_value is None:
        return True
    if current_value is None:
        return None
    return current_value == trigger.condition_required_value


@dataclass(frozen=True)
class TileObservation:
    location: Location
    blocked: bool
    walkable_neighbors: frozenset[Direction]
    warp: WarpObservation | None = None
    trigger_ids: frozenset[str] = frozenset()
    # Static map data. Dynamic object blocking is represented separately on
    # OverworldObservation and must not invalidate this value.
    traversal_cost: int = 1
    # ROM-derived encounter terrain property. This is deliberately separate
    # from traversal_cost: encounter generation is an action property.
    has_encounters: bool = False
    transition: WorldTransition | None = None
    metatile_behavior: int | None = None
    elevation: int | None = None
    cannot_run: bool = False
    # ROM forced movement (ledges, ice, currents): direction ->
    # (destination location, additional forced steps).
    forced_movement_to: dict[Direction, tuple[Location, int]] | None = None


@dataclass(frozen=True)
class OverworldObservation:
    map_id: MapId
    player_coordinates: Coordinate
    facing: Direction | None
    controllable: bool
    tiles: tuple[TileObservation, ...]
    warps: tuple[WarpObservation, ...]
    objects: tuple[ObjectObservation, ...]
    triggers: tuple[TriggerObservation, ...]
    bindings: tuple[BindingResolution, ...] = ()
    # ``player_elevation`` mirrors Emerald's PlayerGetElevation(), which is
    # the player object event's previous elevation and is what coordinate
    # events use for dispatch. The other fields support transition diagnosis.
    player_elevation: int | None = None
    player_current_elevation: int | None = None
    player_previous_coordinates: Coordinate | None = None
    movement_state: MovementState | None = None
    dynamic_blocked_coordinates: frozenset[Coordinate] = frozenset()
    transitions: tuple[WorldTransition, ...] = ()
    transition_in_progress: bool = False
    transition_signals: frozenset[str] = frozenset()
    running_shoes: bool = False
    map_identity_source: str = "save_block"
    save_block_map_id: MapId | None = None
    live_map_candidates: tuple[MapId, ...] = ()
    # True when the avatar is already in a water-traversal state. Navigation
    # uses this to distinguish an ordinary land avatar from one that can
    # legally enter elevation-1 water tiles.
    surfing: bool = False

    def tile_at(self, coordinates: Coordinate) -> TileObservation | None:
        tile = next((tile for tile in self.tiles if tile.location[1] == coordinates), None)
        if tile is not None and coordinates in self.dynamic_blocked_coordinates and not tile.blocked:
            return replace(tile, blocked=True)
        return tile


def _direction(value: str) -> Direction | None:
    try:
        return Direction.from_string(value)
    except (RuntimeError, AttributeError):
        return None


@dataclass(frozen=True)
class _StaticMapObservation:
    path_tiles: object
    map_data: object
    by_coordinate: dict[Coordinate, object]
    warps: tuple[WarpObservation, ...]
    transitions: tuple[WorldTransition, ...]
    tiles: tuple[TileObservation, ...]
    triggers: tuple[TriggerObservation, ...]


_static_map_observations: dict[MapId, _StaticMapObservation] = {}


def _connection_endpoint_is_executable(map_metadata, path_tiles, coordinate: Coordinate) -> bool:
    """Return whether a geometric connection endpoint is usable on a map."""
    tile = path_tiles.get(coordinate)
    if tile is None or not any(tile.accessible_from_direction):
        return False
    return not any(
        object_template.local_coordinates == coordinate for object_template in getattr(map_metadata, "objects", ())
    )


@traced("static_tile_preparation")
def prewarm_static_map_observation(map_id: MapId) -> tuple[TileObservation, ...]:
    """Populate and return static perception data for one known map.

    This deliberately does not read the player, object events, or interaction
    state.  It is safe to call before a warp because it only materializes data
    derived from the destination map's ROM metadata.
    """
    path_map = _get_map_metadata(map_id)
    path_tiles = path_map.tiles
    map_data = get_map_metadata(map_id)
    map_width, map_height = map_data.map_size
    static = _static_map_observations.get(map_id)
    if static is None or static.path_tiles is not path_tiles or static.map_data is not map_data:
        by_coordinate = {tile.local_coordinates: tile for tile in path_tiles}
        warps: list[WarpObservation] = []
        for warp in map_data.warps:
            destination = (
                (warp.destination_location.map_group, warp.destination_location.map_number),
                warp.destination_location.local_position,
            )
            path_tile = by_coordinate.get(warp.local_coordinates)
            required_facing = None
            activation = WarpActivation.STEP_ON
            activation_locations: frozenset[Location] = frozenset()
            activation_direction = None
            if path_tile is not None and path_tile.warps_to is not None:
                required_facing = path_tile.warps_to[2]
                if getattr(path_tile, "tile_type", "").endswith("Arrow Warp"):
                    activation = WarpActivation.DIRECTIONAL_STEP
            tile_type = getattr(path_tile, "tile_type", "") if path_tile is not None else ""
            x, y = warp.local_coordinates
            mechanism = TransitionMechanism.ORDINARY_WARP
            activation_mode = TransitionActivationMode.ENTRY_TILE
            if "Arrow Warp" in tile_type:
                mechanism = TransitionMechanism.ARROW_WARP
                activation_mode = TransitionActivationMode.DIRECTIONAL_INPUT
            elif "Door" in tile_type or "Exterior Door" in tile_type:
                mechanism = TransitionMechanism.DOOR_WARP
                activation_mode = TransitionActivationMode.FIELD_EFFECT
                # Exterior door tiles are collision-blocked ROM warp entries.
                # The field engine activates them from the adjacent walkable
                # tile, so the entry coordinate is not an executable source
                # state for navigation.  Keep the source geometry explicit;
                # this is especially important for recovery, whose catalog
                # destination is the Center door itself.
                direction_vectors = {
                    Direction.North: (0, -1),
                    Direction.East: (1, 0),
                    Direction.South: (0, 1),
                    Direction.West: (-1, 0),
                }
                static_object_locations = {
                    object_template.local_coordinates for object_template in getattr(map_data, "objects", ())
                }
                door_sources = []
                for direction, (dx, dy) in direction_vectors.items():
                    source = (x - dx, y - dy)
                    source_tile = by_coordinate.get(source)
                    if (
                        source_tile is not None
                        and any(source_tile.accessible_from_direction)
                        and source not in static_object_locations
                    ):
                        door_sources.append(((map_id, source), direction))
                activation_locations = frozenset(source for source, _ in door_sources)
                if len(door_sources) == 1:
                    activation_direction = door_sources[0][1]
            if tile_type in ("Escalator Up", "Escalator Down"):
                mechanism = TransitionMechanism.ESCALATOR_WARP
                activation_mode = TransitionActivationMode.FIELD_EFFECT
                # Up/Down describes the destination transition, not the
                # side from which the field interaction is started.  The
                # Gen III Pokémon Center escalator rail is activated from
                # its right-hand source tile with a LEFT input.  Keep this
                # in activation_locations so the navigator can represent
                # source-state semantics independently of the warp tile.
                activation_direction = Direction.West
                required_facing = Direction.West
                source = (x + 1, y)
                if source in by_coordinate:
                    activation_locations = frozenset({(map_id, source)})
            warps.append(
                WarpObservation(
                    (map_id, warp.local_coordinates),
                    destination,
                    required_facing=required_facing,
                    activation=activation,
                    activation_locations=activation_locations,
                    activation_direction=activation_direction,
                    mechanism=mechanism,
                    activation_mode=activation_mode,
                )
            )
        transitions: list[WorldTransition] = list(warps)
        # A connection is a boundary crossing, not a warp.  Keep its ROM
        # mechanism visible to tactical navigation while deriving the same
        # boundary coordinate pairs used by the world graph.
        for connection in getattr(map_data, "connections", ()):
            destination_map = (connection.destination_map_group, connection.destination_map_number)
            destination_data = get_map_metadata(destination_map)
            destination_path_tiles = {tile.local_coordinates: tile for tile in _get_map_metadata(destination_map).tiles}

            source_width, source_height = map_data.map_size
            destination_width, destination_height = destination_data.map_size
            pairs: list[tuple[Coordinate, Coordinate]] = []
            if connection.direction in ("North", "South"):
                first = max(0, connection.offset)
                last = min(source_width, connection.offset + destination_width)
                for source_x in range(first, last):
                    pairs.append(
                        (
                            (source_x, 0 if connection.direction == "North" else source_height - 1),
                            (
                                source_x - connection.offset,
                                destination_height - 1 if connection.direction == "North" else 0,
                            ),
                        )
                    )
            elif connection.direction in ("East", "West"):
                first = max(0, connection.offset)
                last = min(source_height, connection.offset + destination_height)
                for source_y in range(first, last):
                    pairs.append(
                        (
                            (source_width - 1 if connection.direction == "East" else 0, source_y),
                            (
                                0 if connection.direction == "East" else destination_width - 1,
                                source_y - connection.offset,
                            ),
                        )
                    )
            direction = {
                "North": Direction.North,
                "South": Direction.South,
                "East": Direction.East,
                "West": Direction.West,
            }.get(connection.direction)
            for source, destination in pairs:
                if not _connection_endpoint_is_executable(map_data, by_coordinate, source):
                    continue
                if not _connection_endpoint_is_executable(destination_data, destination_path_tiles, destination):
                    continue
                transitions.append(
                    MapConnectionObservation(
                        (map_id, source),
                        (destination_map, destination),
                        required_facing=direction,
                        activation=WarpActivation.STEP_ON,
                    )
                )
        transitions_tuple = tuple(transitions)
        warps_by_entry = {warp.entry[1]: warp for warp in warps}
        transitions_by_entry = {transition.entry[1]: transition for transition in transitions_tuple}
        static_trigger_ids_by_location: dict[Coordinate, set[str]] = {}
        for index, event in enumerate(map_data.coord_events):
            if event.type != "weather":
                static_trigger_ids_by_location.setdefault(event.local_coordinates, set()).add(
                    f"coord:{index}:{event.trigger_var_number}:{event.trigger_value}"
                )
        for index, event in enumerate(map_data.bg_events):
            script = event.script_symbol if event.kind == "Script" else ""
            static_trigger_ids_by_location.setdefault(event.local_coordinates, set()).add(
                f"bg:{index}:{event.kind}:{script}"
            )
        static_tiles: list[TileObservation] = []
        for coordinate, tile in by_coordinate.items():
            walkable = frozenset(
                direction for direction in Direction if tile.accessible_from_direction[direction.value]
            )
            static_tiles.append(
                TileObservation(
                    location=(map_id, coordinate),
                    blocked=not walkable and tile.warps_to is None,
                    walkable_neighbors=walkable,
                    warp=warps_by_entry.get(coordinate),
                    transition=transitions_by_entry.get(coordinate),
                    trigger_ids=frozenset(static_trigger_ids_by_location.get(coordinate, set())),
                    traversal_cost=getattr(tile, "traversal_cost", 1),
                    has_encounters=getattr(tile, "has_encounters", False),
                    metatile_behavior=getattr(tile, "metatile_behavior", None),
                    elevation=getattr(tile, "elevation", None),
                    cannot_run=getattr(tile, "cannot_run", False),
                    forced_movement_to=(
                        {
                            direction: ((forced[0], forced[1]), forced[2])
                            for direction, forced in (getattr(tile, "forced_movement_to", None) or {}).items()
                        }
                        or None
                    ),
                )
            )
        static_triggers: list[TriggerObservation] = []
        for index, event in enumerate(map_data.coord_events):
            if event.type != "weather":
                location = (map_id, event.local_coordinates)
                static_triggers.append(
                    TriggerObservation(
                        trigger_id=f"coord:{index}:{event.trigger_var_number}:{event.trigger_value}",
                        locations=frozenset({location}),
                        activation_locations=frozenset({location}),
                        kind=event.type,
                        elevation=event.elevation,
                        script_symbol=event.script_symbol if event.type == "script" else None,
                        condition_variable=get_event_var_name(event.trigger_var_number),
                        condition_variable_number=event.trigger_var_number,
                        condition_required_value=event.trigger_value,
                        affordance_id=event.script_symbol,
                        requires_input=False,
                    )
                )
        for index, event in enumerate(map_data.bg_events):
            location = (map_id, event.local_coordinates)
            activation_locations = frozenset()
            activation_requirements: tuple[tuple[Location, Direction], ...] = ()
            direction = None
            if event.player_facing_direction == "Any":
                activation_locations = frozenset(
                    (map_id, (candidate_x, candidate_y))
                    for candidate_x, candidate_y in (
                        (event.local_coordinates[0], event.local_coordinates[1] - 1),
                        (event.local_coordinates[0] + 1, event.local_coordinates[1]),
                        (event.local_coordinates[0], event.local_coordinates[1] + 1),
                        (event.local_coordinates[0] - 1, event.local_coordinates[1]),
                    )
                    if 0 <= candidate_x < map_width and 0 <= candidate_y < map_height
                )
                # Adjacent field-script interactions are activated while the
                # player faces the event tile.  ROM's ``Any`` means that all
                # adjacent approach sides are valid; it does not erase the
                # per-side facing affordance required by the field engine.
                activation_requirements = tuple(
                    (
                        source,
                        next(
                            direction
                            for direction, coordinate in (
                                (Direction.South, (event.local_coordinates[0], event.local_coordinates[1] - 1)),
                                (Direction.West, (event.local_coordinates[0] + 1, event.local_coordinates[1])),
                                (Direction.North, (event.local_coordinates[0], event.local_coordinates[1] + 1)),
                                (Direction.East, (event.local_coordinates[0] - 1, event.local_coordinates[1])),
                            )
                            if coordinate == source[1]
                        ),
                    )
                    for source in activation_locations
                )
            else:
                direction = _direction(event.player_facing_direction)
                if direction is not None:
                    offsets = {
                        Direction.North: (0, 1),
                        Direction.East: (-1, 0),
                        Direction.South: (0, -1),
                        Direction.West: (1, 0),
                    }
                    offset_x, offset_y = offsets[direction]
                    activation = (
                        event.local_coordinates[0] + offset_x,
                        event.local_coordinates[1] + offset_y,
                    )
                    if 0 <= activation[0] < map_width and 0 <= activation[1] < map_height:
                        activation_locations = frozenset({(map_id, activation)})
            script = event.script_symbol if event.kind == "Script" else ""
            static_triggers.append(
                TriggerObservation(
                    trigger_id=f"bg:{index}:{event.kind}:{script}",
                    locations=frozenset({location}),
                    activation_locations=activation_locations,
                    kind=f"bg_{event.kind.lower().replace(' ', '_')}",
                    activation_requirements=(
                        activation_requirements
                        if event.player_facing_direction == "Any"
                        else (
                            tuple(((map_id, activation), direction) for activation in activation_locations)
                            if direction is not None
                            else ()
                        )
                    ),
                    script_symbol=script if event.kind == "Script" else None,
                    affordance_id=script or None,
                )
            )
        tiles = tuple(static_tiles)
        static = _StaticMapObservation(
            path_tiles, map_data, by_coordinate, tuple(warps), transitions_tuple, tiles, tuple(static_triggers)
        )
        _static_map_observations[map_id] = static
    return static.tiles


def static_map_transitions(map_id: MapId) -> tuple[WorldTransition, ...]:
    """Return ROM-backed transitions for a lazily materialized map.

    Global navigation normally needs only the map graph.  Goals that end at a
    field-effect warp, however, must retain the executable activation geometry
    from the source map even when that map is reached later in the route.
    """
    prewarm_static_map_observation(map_id)
    return _static_map_observations[map_id].transitions


@traced("overworld_perception")
def perceive_overworld() -> OverworldObservation | OverworldObservationResult:
    """Read the current overworld once per emulator frame.

    Several owners can observe the same frame: readiness, the semantic
    campaign dispatcher, and the tactical controller.  The underlying read
    scans map metadata, runtime objects, and trigger bindings, so repeating it
    before the emulator advances can consume the entire 60 Hz budget.  A
    frame-scoped cache keeps those owners on one immutable passive snapshot.
    Explicit invalidation and a changed/reset frame counter still force a
    fresh read.
    """

    cached = shared_overworld_observation_for_current_frame()
    if cached is not None:
        trace = getattr(context, "stutter_trace", None)
        if trace is not None:
            trace.mark("overworld_perception_cache_hit", True)
        return cached

    observation = _perceive_overworld_uncached()
    publish_shared_overworld_observation(observation)
    return observation


def _perceive_overworld_uncached() -> OverworldObservation | OverworldObservationResult:
    """Read the current map and avatar without changing emulator state.

    Collision and directional access come from the existing PathMap metadata;
    active object positions are reported separately so a planner can choose to
    treat them as dynamic obstacles.
    """

    profiling = enabled()
    trace = getattr(context, "stutter_trace", None)
    profile_start = now()
    stage_start = now()
    # Map/coordinate observations are the authoritative boundary for
    # navigation.  Do not allow the per-frame avatar cache to hide a warp
    # when the emulator callback and the bot loop have crossed different
    # frame boundaries.
    state_cache.player_avatar.invalidate()
    avatar = get_player_avatar()
    if avatar is None:
        reason = "player_avatar_unavailable"
        if trace is not None:
            trace.mark("overworld_observation_status", OverworldObservationStatus.UNAVAILABLE.value)
            trace.mark("overworld_observation_reason", reason)
        return OverworldObservationResult(OverworldObservationStatus.UNAVAILABLE, reason=reason)
    try:
        save_block_map_id = avatar.map_group_and_number
        # The active map header changes with the rendered map.  Prefer it
        # when it is available because SaveBlock1 can still name the outdoor
        # door tile for a few (and, on some warp paths, many) frames after a
        # Pokémon Center interior has become visible.
        live_map = observe_live_map_identity()
        map_id = live_map.map_id or save_block_map_id
        player_coordinates = avatar.local_coordinates
        if trace is not None:
            # Preserve the evidence behind a map choice.  In particular, a
            # save-block fallback must remain distinguishable from a resolved
            # live header in a recovery/warp trace.
            trace.mark("overworld_save_block_map_id", save_block_map_id)
            trace.mark("overworld_live_map_id", live_map.map_id)
            trace.mark("overworld_live_map_source", live_map.source)
            trace.mark("overworld_live_map_candidates", live_map.candidates)
            trace.mark(
                "overworld_live_map_header_fingerprint",
                live_map.header_fingerprint[:32] if live_map.header_fingerprint is not None else None,
            )
        if (
            not isinstance(map_id, tuple)
            or len(map_id) != 2
            or any(value is None for value in map_id)
            or not isinstance(player_coordinates, tuple)
            or len(player_coordinates) != 2
            or any(value is None for value in player_coordinates)
        ):
            raise ValueError("player_map_or_coordinates_malformed")
    except ValueError as error:
        reason = str(error)
        if trace is not None:
            trace.mark("overworld_observation_status", OverworldObservationStatus.MALFORMED.value)
            trace.mark("overworld_observation_reason", reason)
        return OverworldObservationResult(OverworldObservationStatus.MALFORMED, reason=reason)
    timing("perception_player_avatar", stage_start)
    count("player_avatar_observations")
    stage_start = now()
    path_metadata_start = now()
    trace_path_metadata_start = trace.now() if trace is not None else 0
    if trace is not None:
        trace.call("map_metadata_path_lookup", _get_map_metadata, map_id).tiles
    else:
        _get_map_metadata(map_id).tiles
    if trace is not None:
        trace.duration("overworld_map_metadata_lookup_duration_ms", trace_path_metadata_start)
    timing("perception_path_map_lookup", path_metadata_start)
    map_data_start = now()
    map_data = (
        trace.call("map_metadata_rom_lookup", get_map_metadata, map_id)
        if trace is not None
        else get_map_metadata(map_id)
    )
    map_width, map_height = map_data.map_size
    timing("perception_rom_map_lookup", map_data_start)
    timing("perception_map_data_lookup", stage_start)
    count("map_data_lookups")

    static_start = now()
    trace_static_start = trace.now() if trace is not None else 0
    tiles = prewarm_static_map_observation(map_id)
    if trace is not None:
        trace.duration("overworld_static_tile_preparation_duration_ms", trace_static_start)
    timing("perception_static_map_observation", static_start)
    static = _static_map_observations[map_id]
    warps = static.warps

    # These are deliberately independent ROM-side signals.  A map can be
    # loading before its map id changes, and an ordinary warp can lock input
    # without exposing a stable task name.  Consumers use this as a lifecycle
    # hint and wait for destination confirmation instead of invalidating the
    # route that caused the transition.
    transition_signals: set[str] = set()
    controls_available = False
    try:
        controls_available = player_avatar_is_controllable()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    try:
        callback = get_game_state_symbol().upper()
        if "LOADMAP" in callback or callback in {"CB2_CHANGEMAP", "CB2_LOADMAP"}:
            transition_signals.add("map_loading_callback")
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    for task_name in (
        "Task_Warp",
        "Task_DoorWarp",
        "Task_EscalatorWarp",
        "Task_ArrowWarp",
        "Task_MapConnection",
        "Task_TransitionToMap",
        # The popup is created as part of map entry and hides itself on the
        # first accepted field input.  Until then, treat the destination as
        # settling so campaign navigation does not plan from a half-installed
        # map boundary.
        "Task_MapNamePopUpWindow",
    ):
        try:
            # Emerald's map-name banner can remain installed while the player
            # is already controllable (notably after the Route 104/Woods
            # boundary). It is presentation state, not a movement handoff;
            # treating it as a hard transition leaves the campaign waiting
            # forever without ever selecting the next exit.
            if task_name == "Task_MapNamePopUpWindow" and controls_available:
                continue
            if task_is_active(task_name):
                transition_signals.add(f"task:{task_name}")
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
    if not controls_available:
        transition_signals.add("player_controls_locked")
    if getattr(getattr(avatar, "tile_transition_state", None), "name", "NOT_MOVING") != "NOT_MOVING":
        transition_signals.add("avatar_tile_transition")

    triggers: list[TriggerObservation] = list(static.triggers)
    runtime_start = now()
    trace_runtime_start = trace.now() if trace is not None else 0

    def scan_runtime_objects():
        return tuple(
            ObjectObservation(
                local_id=object_event.local_id,
                location=(object_event.map_group_and_number, object_event.current_coords),
                # ObjectEvent itself does not expose either of these attributes.
                # Its map template owns the interaction script; template.kind is
                # only "normal"/"clone", not an object category, so do not expose
                # it as one here.
                script=object_event.object_event_template.script_symbol,
                previous_location=(object_event.map_group_and_number, object_event.previous_coords),
                facing=object_event.facing_direction,
                movement_type=getattr(object_event.object_event_template, "movement_type", None),
                movement_action=object_event.movement_action,
                script_controlled=any(
                    flag in object_event.flags for flag in ("heldMovementActive", "singleMovementActive", "frozen")
                ),
                visibility_flag_id=getattr(object_event.object_event_template, "flag_id", 0) or None,
                trainer_type=object_event.trainer_type,
                trainer_range=(
                    object_event.object_event_template.trainer_range if object_event.trainer_type != "None" else None
                ),
                trainer_defeated=(
                    object_event.object_event_template.is_trainer_defeated
                    if object_event.trainer_type != "None"
                    else None
                ),
                elevation=getattr(object_event.object_event_template, "elevation", None),
            )
            for object_event in get_map_objects()
            if object_event.map_group_and_number == map_id and "isPlayer" not in object_event.flags
        )

    objects = trace.call("runtime_object_scan", scan_runtime_objects) if trace is not None else scan_runtime_objects()
    # Keep runtime occupancy authoritative.  Static trainer templates below
    # contribute hazards only; they must not block a tile or claim a live
    # trigger binding before Emerald has spawned the object event.
    runtime_objects = objects
    static_trainers = static_trainer_observations(map_id, map_data, runtime_objects)
    object_locations = {object_observation.location[1] for object_observation in runtime_objects}

    # Conditions belong to the live observation, not the static cache.
    live_triggers: list[TriggerObservation] = []
    for trigger in triggers:
        if trigger.condition_variable_number is None or trigger.condition_required_value is None:
            live_triggers.append(trigger)
            continue
        try:
            current_value = get_event_var_by_number(trigger.condition_variable_number)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError, StopIteration):
            live_triggers.append(trigger)
            continue
        active = evaluate_trigger_condition(trigger, current_value)
        player_location = (map_id, player_coordinates)
        player_tile = next((tile for tile in tiles if tile.location == player_location), None)
        actionable = bool(
            active
            and player_location in trigger.activation_locations
            and player_coordinates not in object_locations
            and player_tile is not None
            and not player_tile.blocked
        )
        live_triggers.append(
            replace(
                trigger,
                condition_current_value=current_value,
                condition_active=active,
                currently_actionable=actionable,
            )
        )
    triggers = live_triggers

    # Object templates/scripts are the reliable data available for NPC
    # interactions.  Activation geometry follows Emerald's object resolver,
    # including its generic counter continuation rule.
    for object_observation in objects:
        activation_requirements = resolve_object_activation_positions(
            object_observation, objects, tiles, (map_width, map_height)
        )
        trainer_hazards = trainer_hazard_locations(object_observation, tiles, objects)
        activation_locations = frozenset(location for location, _ in activation_requirements)
        triggers.append(
            TriggerObservation(
                trigger_id=f"object:{object_observation.local_id}:{object_observation.script}",
                locations=frozenset({object_observation.location}),
                activation_locations=activation_locations,
                kind="object_interaction",
                activation_requirements=activation_requirements,
                affordance_id=object_observation.trainer_id
                or object_observation.script
                or f"object:{object_observation.local_id}",
                hazard_locations=trainer_hazards,
                hazard_kind="trainer" if trainer_hazards else None,
            )
        )

    # Object events can be absent from gObjectEvents until the camera nears
    # them.  Publish their ROM-derived sight lines immediately so a trainer-
    # avoiding route does not commit to an unsafe approach before that load.
    for trainer in static_trainers:
        triggers.append(
            TriggerObservation(
                trigger_id=f"static_object:{trainer.local_id}:{trainer.script}",
                locations=frozenset({trainer.location}),
                kind="trainer_hazard",
                affordance_id=trainer.trainer_id,
                hazard_locations=trainer_hazard_locations(
                    trainer,
                    tiles,
                    tuple(runtime_objects) + static_trainers,
                ),
                hazard_kind="trainer",
                # A template fallback is not a live interaction affordance.
                # It exists solely to constrain route selection.
                requires_input=False,
            )
        )

    timing("perception_active_runtime_objects", runtime_start)
    if trace is not None:
        trace.duration("overworld_runtime_object_scan_duration_ms", trace_runtime_start)
    count("runtime_object_scans")
    static_start = now()
    count("static_template_lookups")
    binding_start = now()
    trace_binding_start = trace.now() if trace is not None else 0

    def resolve_bindings():
        return tuple(
            resolve_trigger_binding(
                binding,
                runtime_objects,
                (map_width, map_height),
                static_objects=getattr(map_data, "objects", ()) if map_id == binding.map_id else (),
            )
            for binding in TRIGGER_BINDINGS
        )

    bindings = (
        trace.call("trigger_discovery_and_binding", resolve_bindings) if trace is not None else resolve_bindings()
    )
    timing("perception_static_object_template_lookup", static_start)
    timing("trigger_binding_resolution", binding_start)
    if trace is not None:
        trace.duration("overworld_trigger_binding_duration_ms", trace_binding_start)
    for resolution in bindings:
        binding = resolution.binding
        activation_requirements = tuple(
            (
                (map_id, (candidate_x, candidate_y)),
                {
                    (object_x, object_y - 1): Direction.South,
                    (object_x + 1, object_y): Direction.West,
                    (object_x, object_y + 1): Direction.North,
                    (object_x - 1, object_y): Direction.East,
                }[(candidate_x, candidate_y)],
            )
            for object_x, object_y in resolution.runtime_locations
            for candidate_x, candidate_y in (
                (object_x, object_y - 1),
                (object_x + 1, object_y),
                (object_x, object_y + 1),
                (object_x - 1, object_y),
            )
            if (candidate_x, candidate_y) in resolution.interaction_positions
        )
        triggers.append(
            TriggerObservation(
                trigger_id=binding.trigger_id,
                locations=frozenset((map_id, coordinates) for coordinates in resolution.runtime_locations),
                activation_locations=frozenset(
                    (map_id, coordinates) for coordinates in resolution.interaction_positions
                ),
                kind=f"semantic_{binding.event_type}",
                target_map=binding.map_id,
                navigation_locations=frozenset(
                    {resolution.static_location} if resolution.static_location is not None else set()
                ),
                activation_requirements=activation_requirements,
                affordance_id=binding.script_symbol,
            )
        )

    # Static topology is reused. Object positions remain a per-observation
    # overlay; they must never be folded into the cached tile objects.
    dynamic_blocked_coordinates = frozenset(object_locations)

    running_state = getattr(avatar, "running_state", None)
    tile_transition_state = getattr(avatar, "tile_transition_state", None)
    if getattr(running_state, "name", None) == "TURN_DIRECTION":
        movement_state = MovementState.TURNING
    elif (
        getattr(running_state, "name", None) == "MOVING"
        or getattr(tile_transition_state, "name", "NOT_MOVING") != "NOT_MOVING"
    ):
        movement_state = MovementState.MOVING
    else:
        movement_state = MovementState.STANDING

    if profiling:
        timing("perception_total", profile_start)
        profile_print(
            lambda: f"OVERWORLD_PROFILE: {format_snapshot()} "
            f"map={map_id!r} position={player_coordinates!r} "
            f"movement={movement_state.name!r}",
            every=60,
        )

    return OverworldObservation(
        map_id=map_id,
        player_coordinates=player_coordinates,
        facing=_direction(avatar.facing_direction),
        controllable=controls_available,
        tiles=static.tiles,
        warps=tuple(warps),
        transitions=static.transitions,
        objects=objects,
        triggers=tuple(triggers),
        bindings=bindings,
        player_elevation=getattr(avatar, "elevation", None),
        player_current_elevation=getattr(avatar, "current_elevation", None),
        player_previous_coordinates=getattr(avatar, "previous_coordinates", None),
        movement_state=movement_state,
        dynamic_blocked_coordinates=dynamic_blocked_coordinates,
        transition_in_progress=bool(transition_signals),
        transition_signals=frozenset(transition_signals),
        running_shoes=_running_shoes_received(),
        map_identity_source=live_map.source if live_map.map_id is not None else "save_block",
        save_block_map_id=save_block_map_id,
        live_map_candidates=live_map.candidates,
        surfing=bool(getattr(avatar, "is_in_water", False)),
    )
