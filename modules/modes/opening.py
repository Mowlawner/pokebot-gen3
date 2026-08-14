"""Emerald's fresh-game opening sequence."""

from enum import Enum, IntEnum, auto
from dataclasses import dataclass
from datetime import datetime
import random
from typing import Generator

from modules.console import console, diagnostic_print
from modules.context import context
from modules.keyboard import get_naming_screen_data, type_in_naming_screen
from modules.game_sprites import get_game_sprite_by_id
from modules.map import get_map_data_for_current_position, get_map_objects
from modules.map_data import MapRSE
from modules.memory import (
    GameState,
    get_event_flag,
    get_event_var,
    get_game_state,
    get_game_state_symbol,
    get_save_block,
    read_symbol,
    unpack_uint16,
    unpack_uint32,
)
from modules.modes.util.tasks_scripts import wait_for_fade_to_finish
from modules.modes.util.walking import ensure_facing_direction, navigate_to
from modules.player import (
    AvatarFlags,
    get_player,
    get_player_avatar,
    get_player_map_object,
    player_avatar_is_controllable,
    player_avatar_is_standing_still,
)
from modules.pokemon_party import get_party_size
from modules.text_printer import get_text_printer
from modules.config.schemas_v1 import WallClockTimeMode
from modules.start_game import RandomSource, resolve_start_game_initialization
from modules.tasks import (
    get_global_script_context,
    get_task,
    get_tasks,
    is_field_message_waiting_for_input,
    is_waiting_for_input,
    task_is_active,
)
from ._interface import BotMode, BotModeError

_starter_handoff_pending = False
_EMERALD_TEXT_SPEED_SAVE_BLOCK2_OFFSET = 0x14
_EMERALD_TEXT_SPEED_MASK = 0x07
_EMERALD_OPTIONS_TASKS = (
    "Task_OptionMenuProcessInput",
    "Task_OptionMenuFadeIn",
    "Task_OptionMenuSave",
    "Task_OptionMenuFadeOut",
)
_BIRCH_GENDER_TASKS = (
    "Task_NewGameBirchSpeech_ChooseGender",
    "Task_NewGameBirchSpeech_WaitToShowGenderMenu",
    "Task_NewGameBirchSpeech_SlideOutOldGenderSprite",
    "Task_NewGameBirchSpeech_SlideInNewGenderSprite",
)
_last_scripted_input_trace: tuple | None = None
_birch_gender_diagnostic_phase: str | None = None
_birch_gender_diagnostic_target: object | None = None


class EmeraldTextSpeed(IntEnum):
    """Values used by Emerald's Options-menu text-speed setting."""

    SLOW = 0
    MEDIUM = 1
    FAST = 2


def consume_starter_handoff() -> bool:
    global _starter_handoff_pending
    pending = _starter_handoff_pending
    _starter_handoff_pending = False
    return pending


class OpeningSequenceState(Enum):
    TITLE = auto()
    MAIN_MENU = auto()
    PLAYER_NAMING = auto()
    TRUCK = auto()
    LITTLEROOT_TOWN = auto()
    POST_CLOCK_TOWN = auto()
    PLAYER_HOUSE_2F = auto()
    CLOCK_SETTING = auto()
    PLAYER_HOUSE_1F = auto()
    PLAYER_HOUSE_1F_POST_CLOCK_ARRIVAL = auto()
    BIRCH_HOUSE_1F = auto()
    BIRCH_HOUSE_2F = auto()
    BIRCH_POKEBALL = auto()
    MAY_SEQUENCE = auto()
    ROUTE_101 = auto()
    STARTER_SELECTION = auto()
    SCRIPTED_INTRO = auto()
    UNKNOWN = auto()
    OPTIONS_MENU = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class OpeningDiagnostics:
    phase: OpeningSequenceState
    observed: OpeningSequenceState
    map_id: tuple[int, int] | None
    coordinates: tuple[int, int] | None
    game_state: GameState | None
    controllable: bool | None
    script_active: bool | None
    waiting_for_input: bool | None
    facing: str | None
    truck_flag: bool | None
    truck_sequence_active: bool | None
    active_script: str | None = None
    truck_task: str | None = None
    truck_task_state: int | None = None
    input_accepted: bool | None = None
    clock_task: str | None = None
    clock_hours: int | None = None
    clock_minutes: int | None = None


def get_opening_diagnostics(
    phase: OpeningSequenceState,
    observed: OpeningSequenceState,
) -> OpeningDiagnostics:
    """Snapshot the controller's decision inputs for debug/test visibility."""
    map_id = _current_map_id()
    coordinates = None
    facing = None
    try:
        avatar = get_player_avatar()
        coordinates = avatar.local_coordinates
        facing = avatar.facing_direction
        controllable = player_avatar_is_controllable()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        controllable = None

    try:
        script_context = get_global_script_context()
        script_active = None if script_context is None else script_context.is_active
    except (AttributeError, RuntimeError, ValueError, TypeError):
        script_active = None

    try:
        waiting = is_waiting_for_input()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        waiting = None

    try:
        game_state = get_game_state()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        game_state = None

    truck_flag = None
    truck_sequence_active = None
    active_script = None
    truck_task = None
    truck_task_state = None
    input_accepted = None
    clock_task = None
    clock_hours = None
    clock_minutes = None
    if phase is OpeningSequenceState.TRUCK or observed is OpeningSequenceState.TRUCK:
        truck_flag = _truck_has_left()
        truck_sequence_active = _truck_sequence_active()
        active_script = _active_script_name()
        truck_task, truck_task_state = _truck_task_snapshot()
        input_accepted = (
            game_state == GameState.OVERWORLD
            and controllable is True
            and not bool(script_active)
            and not bool(truck_sequence_active)
        )
    if phase is OpeningSequenceState.CLOCK_SETTING or observed is OpeningSequenceState.CLOCK_SETTING:
        clock_task = _active_clock_task()
        try:
            task = get_task(clock_task) if clock_task is not None else None
            if task is not None:
                clock_hours = task.data_value(2)
                clock_minutes = task.data_value(3)
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass

    return OpeningDiagnostics(
        phase,
        observed,
        map_id,
        coordinates,
        game_state,
        controllable,
        script_active,
        waiting,
        facing,
        truck_flag,
        truck_sequence_active,
        active_script,
        truck_task,
        truck_task_state,
        input_accepted,
        clock_task,
        clock_hours,
        clock_minutes,
    )


def _birch_gender_task_snapshot() -> tuple | None:
    """Collect read-only state while Emerald's Birch gender UI tasks exist.

    This deliberately reports raw task words and sprite metadata. It does not
    assign a meaning to any word or derive a gender from the saved Player.
    """
    try:
        tasks = get_tasks()
        present = [name for name in _BIRCH_GENDER_TASKS if name in tasks]
        if not present:
            return None

        task_data = []
        for name in _BIRCH_GENDER_TASKS:
            task = get_task(name)
            if task is None:
                task_data.append((name, False, None, ()))
                continue
            task_data.append(
                (
                    name,
                    True,
                    task.symbol,
                    tuple(task.data_value(i) for i in range(min(16, len(task.data) // 2))),
                )
            )

        script = get_global_script_context()
        script_state = (
            None
            if script is None
            else (
                script.is_active,
                script.mode,
                script.native_function_name,
                script.script_function_name,
                script.native_pointer,
                script.bytecode_pointer,
                tuple(script.stack),
            )
        )
        avatar = get_player_avatar()
        location = (avatar.map_group_and_number, avatar.local_coordinates)
        state = get_game_state()
        try:
            player_gender = get_player().gender
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
            player_gender = None

        sprites = []
        for sprite_id in range(64):
            sprite = get_game_sprite_by_id(sprite_id)
            if "in_use" in sprite.flags:
                sprites.append(
                    (
                        sprite_id,
                        sprite.coordinates,
                        sprite.secondary_coordinates,
                        tuple(sprite.data_value(i) for i in range(8)),
                        tuple(sprite.flags),
                    )
                )

        objects = []
        for obj in get_map_objects():
            objects.append((obj.local_id, obj.sprite_id, obj.graphics_id, obj.current_coords, tuple(obj.flags)))

        return (
            tuple(present),
            tuple(task_data),
            location,
            state.name,
            get_game_state_symbol(),
            player_gender,
            tuple(task.symbol for task in tasks),
            script_state,
            tuple(sprites),
            tuple(objects),
        )
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
        return ("unavailable",)


_CLOCK_TASKS = (
    "Task_SetClock_WaitFadeIn",
    "Task_SetClock_HandleInput",
    "Task_SetClock_AskConfirm",
    "Task_SetClock_HandleConfirmInput",
    "Task_SetClock_Confirmed",
    "Task_SetClock_Exit",
)
_CLOCK_HANDLE_INPUT = "Task_SetClock_HandleInput"
_CLOCK_HANDLE_CONFIRM_INPUT = "Task_SetClock_HandleConfirmInput"
_CLOCK_ASK_CONFIRM = "Task_SetClock_AskConfirm"
_CLOCK_HOUR_MIN = 0
_CLOCK_HOUR_MAX = 23
_CLOCK_MINUTE_MIN = 0
_CLOCK_MINUTE_MAX = 59
_CLOCK_MINUTES_PER_HOUR = 60
_CLOCK_PERIOD_MINUTES = (_CLOCK_HOUR_MAX + 1) * _CLOCK_MINUTES_PER_HOUR
_RIVAL_POKEBALL_SCRIPTS = {
    MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F.value: "LittlerootTown_MaysHouse_2F_EventScript_RivalsPokeBall",
    MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value: "LittlerootTown_BrendansHouse_2F_EventScript_RivalsPokeBall",
}
_STARTER_BAG_SCRIPT = "Route101_EventScript_BirchsBag"
# These are interaction positions, not the player's observed position.  In
# Emerald's Brendan house 2F (1, 1), the player starts at (7, 2) and must walk
# west to the clock interaction at (5, 2), then face Up.  Keep targets keyed by
# house map so a future randomized protagonist can supply a different target
# instead of silently reusing Brendan's layout.
_EMERALD_WALL_CLOCK_INTERACTIONS = {
    MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value: ((5, 2), "Up"),
}
_EMERALD_PLAYER_HOUSE_CLOCK_INTERACTIONS = {
    "male": ((5, 2), "Up"),
    "female": ((3, 2), "Up"),
}
_TRUCK_SEQUENCE_TASKS = (
    "Task_HandleTruckSequence",
    "Task_Truck1",
    "Task_Truck2",
    "Task_Truck3",
)


def _current_map_id() -> tuple[int, int] | None:
    try:
        location = get_map_data_for_current_position()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None
    return None if location is None else location.map_group_and_number


def _opening_player_gender(player_gender: object | None = None) -> str:
    """Return the gender used to assign Emerald's two Littleroot houses.

    During a fresh opening the save-backed Player may not exist yet, so the
    mode passes its already-resolved gender explicitly.  Direct state probes
    can use the saved Player when available and retain the historical male
    default during the pre-save transition.
    """
    value = getattr(player_gender, "value", player_gender)
    if value in ("male", "female"):
        return value
    try:
        value = get_player().gender
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        value = "male"
    return value if value in ("male", "female") else "male"


def _littleroot_house_maps(
    player_gender: object | None = None,
) -> tuple[MapRSE, MapRSE, MapRSE, MapRSE]:
    """Return player/rival 1F and 2F maps for the selected protagonist."""
    if _opening_player_gender(player_gender) == "female":
        return (
            MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F,
            MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F,
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F,
            MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
        )
    return (
        MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F,
        MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
        MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F,
        MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F,
    )


def _player_house_map(floor: int, player_gender: object | None = None) -> MapRSE:
    return _littleroot_house_maps(player_gender)[floor - 1]


def _rival_house_map(floor: int, player_gender: object | None = None) -> MapRSE:
    return _littleroot_house_maps(player_gender)[floor + 1]


def _clock_task_active() -> bool:
    try:
        return any(task_is_active(task) for task in _CLOCK_TASKS)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        # During emulator/map transitions the task table can be unavailable.
        return False


def _active_clock_task() -> str | None:
    """Return the currently running Emerald clock task callback, if any."""
    try:
        return next((task for task in _CLOCK_TASKS if task_is_active(task)), None)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return None


def _clock_setting_complete() -> bool:
    """Return true only after Emerald has finished the wall-clock task chain."""
    try:
        return get_event_flag("SET_WALL_CLOCK") and not _clock_task_active()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _clock_minutes(hour: int, minute: int) -> int:
    """Convert Emerald's internal 24-hour clock value to a position."""
    return hour * _CLOCK_MINUTES_PER_HOUR + minute


def _clock_input_direction(
    current_hour: int,
    current_minute: int,
    target_hour: int,
    target_minute: int,
) -> str:
    """Choose the shortest direction around Emerald's 24-hour clock."""
    forward = (
        _clock_minutes(target_hour, target_minute) - _clock_minutes(current_hour, current_minute)
    ) % _CLOCK_PERIOD_MINUTES
    return "Right" if forward <= _CLOCK_PERIOD_MINUTES // 2 else "Left"


def get_opening_sequence_state(player_gender: object | None = None) -> OpeningSequenceState:
    """Classify the currently observable opening state.

    This deliberately treats transitional frames as ``SCRIPTED_INTRO`` rather
    than as an error.  The controller keeps its own goal phase and retries the
    relevant observation on later frames.
    """
    try:
        game_state = get_game_state()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return OpeningSequenceState.UNKNOWN
    if game_state == GameState.CHOOSE_STARTER and _current_map_id() == MapRSE.ROUTE101.value:
        return OpeningSequenceState.STARTER_SELECTION
    if game_state == GameState.TITLE_SCREEN:
        return OpeningSequenceState.TITLE
    if game_state == GameState.MAIN_MENU:
        return OpeningSequenceState.MAIN_MENU
    if game_state == GameState.OPTIONS_MENU:
        return OpeningSequenceState.OPTIONS_MENU
    if game_state == GameState.NAMING_SCREEN:
        return OpeningSequenceState.PLAYER_NAMING
    if _clock_task_active():
        return OpeningSequenceState.CLOCK_SETTING

    map_id = _current_map_id()
    if map_id is None:
        return OpeningSequenceState.TRUCK
    if map_id == MapRSE.INSIDE_OF_TRUCK.value:
        return OpeningSequenceState.TRUCK
    if map_id == MapRSE.LITTLEROOT_TOWN.value:
        return OpeningSequenceState.LITTLEROOT_TOWN
    if map_id == _player_house_map(2, player_gender).value:
        return OpeningSequenceState.PLAYER_HOUSE_2F
    if map_id == _player_house_map(1, player_gender).value:
        return OpeningSequenceState.PLAYER_HOUSE_1F
    if map_id == _rival_house_map(1, player_gender).value:
        return OpeningSequenceState.BIRCH_HOUSE_1F
    if map_id == _rival_house_map(2, player_gender).value:
        return OpeningSequenceState.BIRCH_HOUSE_2F
    if map_id == MapRSE.ROUTE101.value:
        return OpeningSequenceState.ROUTE_101
    if _opening_complete():
        return OpeningSequenceState.COMPLETE
    return OpeningSequenceState.SCRIPTED_INTRO


def _opening_complete() -> bool:
    """Whether the starter opening has returned to Birch's lab.

    The starter is written to the party before the opening battle, and the
    battle's post-battle warp returns the player to this ROM-defined map. The
    combination avoids treating an arbitrary coordinate in the lab as the
    completion marker.
    """
    try:
        return (
            _current_map_id() == MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value
            and get_party_size() > 0
        )
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _truck_has_left() -> bool:
    try:
        return get_event_flag("HIDE_LITTLEROOT_TOWN_BRENDANS_HOUSE_TRUCK")
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return False


def _truck_sequence_active() -> bool:
    """Whether Emerald's ROM-owned truck animation/event is still running."""
    try:
        if any(task_is_active(task) for task in _TRUCK_SEQUENCE_TASKS):
            return True
        script_context = get_global_script_context()
        if script_context is None or not script_context.is_active:
            return False
        names = [script_context.script_function_name, script_context.native_function_name]
        names.extend(script_context.stack)
        return any(name and "Truck" in name for name in names)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        # A task/script table can be unavailable for a transition frame.  The
        # safe choice during startup is to wait for another observation.
        return True


def _active_script_name() -> str | None:
    try:
        script_context = get_global_script_context()
        if script_context is None or not script_context.is_active:
            return None
        names = [script_context.script_function_name, script_context.native_function_name]
        names.extend(script_context.stack)
        return next((name for name in names if name), None)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None


def _truck_task_snapshot() -> tuple[str | None, int | None]:
    """Return the ROM truck task and its state word, if the task is present."""
    try:
        for task_name in _TRUCK_SEQUENCE_TASKS:
            task = get_task(task_name)
            if task is not None:
                return task_name, task.data_value(0)
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    return None, None


def _normal_littleroot_overworld() -> bool:
    """Require the truck warp to have completed before leaving TRUCK phase."""
    try:
        return (
            _current_map_id() == MapRSE.LITTLEROOT_TOWN.value
            and _truck_has_left()
            and get_game_state() == GameState.OVERWORLD
            and player_avatar_is_controllable()
            and not _truck_sequence_active()
        )
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return False


def _littleroot_arrival_ready() -> bool:
    """Whether the post-truck arrival script has finished and walking is safe."""
    try:
        return (
            _current_map_id() == MapRSE.LITTLEROOT_TOWN.value
            and get_game_state() == GameState.OVERWORLD
            and player_avatar_is_controllable()
            and not _truck_sequence_active()
        )
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return False


def _player_house_1f_ready_for_navigation() -> bool:
    """Whether the initial first-floor arrival script has finished."""
    try:
        if get_game_state() != GameState.OVERWORLD or not player_avatar_is_controllable():
            return False
        if is_waiting_for_input():
            return False
        script_context = get_global_script_context()
        return script_context is None or not script_context.is_active
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _post_clock_2f_ready_for_stair_navigation() -> bool:
    """Whether the ROM has released control after the clock sequence.

    Setting the clock changes the event flag, but the wall-clock script does
    not itself change the map.  Mom's follow-up script can therefore finish
    while the player is still on 2F; the player must then walk to the stairs.
    """
    try:
        if get_game_state() != GameState.OVERWORLD or not player_avatar_is_controllable():
            return False
        if is_waiting_for_input():
            return False
        script_context = get_global_script_context()
        return script_context is None or not script_context.is_active
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _post_clock_1f_arrival_complete(player_gender: object | None = None) -> bool:
    """Whether Emerald finished the expected TV event after returning from 2F.

    The 1F map's ``OnFrame`` script starts the Petalburg Gym broadcast while
    ``VAR_LITTLEROOT_INTRO_STATE`` is 6.  The shared player-house script sets
    it to 7 only after the broadcast, its messages, and Mom's movement have
    completed.  This variable is the ROM's completion marker; task/script
    activity alone has a gap during map arrival.
    """
    try:
        if (
            _current_map_id() != _player_house_map(1, player_gender).value
            or get_event_var("LITTLEROOT_INTRO_STATE") != 7
            or get_game_state() != GameState.OVERWORLD
            or not player_avatar_is_controllable()
        ):
            return False
        script_context = get_global_script_context()
        return script_context is None or not script_context.is_active
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _may_sequence_event_complete() -> bool:
    """Whether the ROM has released control after the Rival event."""
    try:
        if get_game_state() != GameState.OVERWORLD or not player_avatar_is_controllable():
            return False
        if is_waiting_for_input():
            return False
        script_context = get_global_script_context()
        return script_context is None or not script_context.is_active
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _route101_ready_for_navigation() -> bool:
    """Whether the Route 101 rescue has released control for pathing onward.

    The ROM's Route 101 state variable is the event boundary: the rescue
    script sets it to 2 only after closing its final message and before
    releasing the player.  The field-message task is deliberately diagnostic
    only.  It can outlive ``closemessage`` in the emulator observation, so it
    is not evidence that the rescue event is still running.
    """
    try:
        return get_event_var("ROUTE101_STATE") >= 2 and player_avatar_is_controllable()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _route101_observation_stage(route_state: int | None, state: tuple) -> str:
    """Name only what the raw Route 101 snapshot proves.

    These labels are observational, not additional completion predicates.  In
    particular, a visible message is reported as such without claiming that
    it is the Birch rescue message or that it needs input from the bot.
    """
    message_visible = bool(state[7])
    script_active = state[2] is True
    controllable = state[6] is True
    if route_state == 0 and not script_active and not message_visible:
        return "A_arrival_before_rescue"
    if message_visible:
        return "C_message_visible"
    if route_state is not None and route_state < 2 and script_active:
        return "B_or_D_rescue_script_no_message"
    if route_state is not None and route_state >= 2 and controllable:
        return "E_or_F_rescue_complete_control_restored"
    return "unclassified"


def _warp_to(destination: MapRSE, *, expecting_script: bool = False) -> Generator:
    """Walk to the active map's warp leading to ``destination``.

    Warp coordinates are read from the ROM's map event data, avoiding brittle
    Littleroot exterior coordinates while still using the normal pathfinder.
    """
    try:
        if get_game_state() != GameState.OVERWORLD or not player_avatar_is_controllable():
            yield
            return
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        yield
        return

    try:
        location = get_map_data_for_current_position()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        yield
        return
    if location is None:
        yield
        return
    try:
        warp = next(
            (
                candidate
                for candidate in location.warps
                if candidate.destination_location.map_group_and_number == destination.value
            ),
            None,
        )
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        yield
        return
    if warp is None:
        yield
        return
    navigate_kwargs = {"avoid_scripted_events": False}
    if expecting_script:
        navigate_kwargs["expecting_script"] = True
    yield from navigate_to(location.map_group_and_number, warp.local_coordinates, **navigate_kwargs)


def _rival_pokeball_object_template():
    """Find the current house's Rival Poké Ball scripted object template."""
    location = get_map_data_for_current_position()
    if location is None:
        return None
    script_symbol = _RIVAL_POKEBALL_SCRIPTS.get(location.map_group_and_number)
    if script_symbol is None:
        return None
    return next(
        (obj for obj in location.objects if obj.script_symbol == script_symbol),
        None,
    )


def _rival_pokeball_interaction() -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return the tile and facing needed to interact with Rival's Poké Ball."""
    object_template = _rival_pokeball_object_template()
    if object_template is None:
        return None
    object_x, object_y = object_template.local_coordinates
    object_coordinates = (object_x, object_y)
    adjacent_tiles = {
        (object_x, object_y + 1): object_coordinates,
        (object_x, object_y - 1): object_coordinates,
        (object_x + 1, object_y): object_coordinates,
        (object_x - 1, object_y): object_coordinates,
    }
    try:
        player_coordinates = get_player_avatar().local_coordinates
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        player_coordinates = None
    if player_coordinates in adjacent_tiles:
        return player_coordinates, object_coordinates
    return (object_x, object_y + 1), object_coordinates


def _starter_bag_interaction() -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return the canonical tile below Route 101's scripted starter bag."""
    location = get_map_data_for_current_position()
    if location is None or location.map_group_and_number != MapRSE.ROUTE101.value:
        return None
    bag = next((obj for obj in location.objects if obj.script_symbol == _STARTER_BAG_SCRIPT), None)
    if bag is None:
        return None
    bag_x, bag_y = bag.local_coordinates
    bag_coordinates = (bag_x, bag_y)
    return (bag_x, bag_y + 1), bag_coordinates


def _wall_clock_interaction(
    map_id: tuple[int, int] | None = None,
    player_gender: object | None = None,
) -> tuple[tuple[int, int], str] | None:
    """Return the clock target for the selected player's 2F house.

    The two physical houses are semantically interchangeable during the
    opening, but their wall clocks are at different tiles.  Classify the
    physical map through the same gender-aware house resolver used by the
    opening state machine before resolving the layout-specific target.
    """
    if map_id is None:
        map_id = _current_map_id()
    gender = _opening_player_gender(player_gender)
    if map_id != _player_house_map(2, gender).value:
        return None

    target = _EMERALD_WALL_CLOCK_INTERACTIONS.get(map_id)
    if target is not None:
        return target

    # May's house has a different layout, so use its ROM event rather than
    # falling back to Brendan's interaction tile.
    location = get_map_data_for_current_position()
    if location is not None:
        clock = next(
            (
                event
                for event in location.bg_events
                if event.kind == "Script" and event.script_symbol.endswith("_EventScript_WallClock")
            ),
            None,
        )
        if clock is not None:
            offsets = {
                "Up": ((0, 1), "Up"),
                "Down": ((0, -1), "Down"),
                "Left": ((1, 0), "Left"),
                "Right": ((-1, 0), "Right"),
            }
            offset_and_direction = offsets.get(clock.player_facing_direction)
            if offset_and_direction is not None:
                (offset_x, offset_y), direction = offset_and_direction
                clock_x, clock_y = clock.local_coordinates
                return (clock_x + offset_x, clock_y + offset_y), direction

    # Some shipped Emerald symbol tables expose this event under an unnamed
    # or numeric script symbol.  The semantic house still identifies the
    # layout, so retain its ROM-known interaction tile when symbol lookup is
    # unavailable.
    return _EMERALD_PLAYER_HOUSE_CLOCK_INTERACTIONS.get(gender)


def _scripted_input_waiting() -> bool:
    """Backward-compatible field-message input predicate for phase helpers."""
    try:
        return is_field_message_waiting_for_input()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _report_birch_gender_input_attempt(
    *,
    path: str,
    button: str,
    observed: OpeningSequenceState | None = None,
    before_inputs: int | None = None,
    before_pending: tuple[int | None, int | None, int | None] | None = None,
) -> None:
    """Report a generic input immediately before it reaches the emulator.

    This is deliberately diagnostic-only.  It reports only attempted inputs
    while one of Birch's gender-menu tasks is present, so normal opening
    tracing remains quiet and the input call itself is not altered.
    """
    if not context.debug or not getattr(context, "debug_trace", False):
        return
    snapshot = _birch_gender_task_snapshot()
    if snapshot is None or snapshot == ("unavailable",):
        return
    state = None
    try:
        state = EmeraldOpeningMode._dialogue_state_snapshot()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        pass
    (
        present,
        task_data,
        _location,
        game_state,
        game_state_symbol,
        player_gender,
        active_tasks,
        _script,
        _sprites,
        _objects,
    ) = snapshot
    phase = observed.name if observed is not None else _birch_gender_diagnostic_phase
    native_state = None if state is None else (state[2], state[3], state[10])
    script_state = None if state is None else (state[2], state[4], state[11])
    field_message_state = (
        None
        if state is None
        else {
            "task_active": state[0],
            "task_data_0": state[1],
            "waiting": state[5],
            "visible": state[7],
            "printer": state[8],
        }
    )
    diagnostic_print(
        "[bold yellow]Birch gender input: "
        f"path={path} button={button} phase={phase} "
        f"active_gender_tasks={present} target_gender={_birch_gender_diagnostic_target} "
        f"player_gender={player_gender} "
        f"task_data_0_16={task_data} native_state={native_state} "
        f"script_state={script_state} field_message_state={field_message_state} "
        f"input_buffer_before={before_inputs} pending_inputs_before={before_pending} "
        f"game_state={game_state} game_state_symbol={game_state_symbol} "
        f"relevant_active_tasks={active_tasks}[/]",
        trace=True,
    )


def _report_opening_a_decision(
    *,
    source: str,
    reason: str,
    phase: OpeningSequenceState | None = None,
) -> None:
    """Trace the exact opening branch that decided to submit an A input."""
    if not context.debug or not getattr(context, "debug_trace", False):
        return
    try:
        active_tasks = [task.symbol for task in (get_tasks() or [])]
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        active_tasks = []
    try:
        player_gender = get_player().gender
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
        player_gender = None
    diagnostic_print(
        "[bold yellow]Opening A decision: "
        f"source={source} reason={reason!r} "
        f"phase={(phase.name if phase is not None else _birch_gender_diagnostic_phase)} "
        f"active_tasks={active_tasks} "
        f"target_gender={_birch_gender_diagnostic_target} player_gender={player_gender}[/]",
        trace=True,
    )


def _advance_scripted_input() -> Generator:
    """Advance only when the game reports that an input is currently wanted."""
    global _last_scripted_input_trace
    waiting_for_input = _scripted_input_waiting()
    waiting_reason = "is_field_message_waiting_for_input()" if waiting_for_input else None
    if not waiting_for_input:
        try:
            waiting_for_input = context.rom.is_emerald and task_is_active("Task_DrawFieldMessage")
            if waiting_for_input:
                waiting_reason = "Emerald Task_DrawFieldMessage fallback"
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
    if waiting_for_input:
        before_inputs = None
        before_pending = (None, None, None)
        after_inputs = None
        after_pending = (None, None, None)
        if context.debug and getattr(context, "debug_trace", False):
            try:
                before_inputs = context.emulator.get_inputs()
                before_pending = (
                    getattr(context.emulator, "_prev_pressed_inputs", None),
                    getattr(context.emulator, "_pressed_inputs", None),
                    getattr(context.emulator, "_held_inputs", None),
                )
            except (AttributeError, RuntimeError, TypeError):
                pass
            _report_birch_gender_input_attempt(
                path="_advance_scripted_input",
                button="A",
                before_inputs=before_inputs,
                before_pending=before_pending,
            )
        _report_opening_a_decision(
            source="_advance_scripted_input",
            reason=waiting_reason or "scripted input predicate returned true",
        )
        context.emulator.press_button("A")
        if context.debug and getattr(context, "debug_trace", False):
            try:
                after_inputs = context.emulator.get_inputs()
                after_pending = (
                    getattr(context.emulator, "_prev_pressed_inputs", None),
                    getattr(context.emulator, "_pressed_inputs", None),
                    getattr(context.emulator, "_held_inputs", None),
                )
            except (AttributeError, RuntimeError, TypeError):
                pass
            trace = (before_inputs, after_inputs, before_pending, after_pending)
            if trace != _last_scripted_input_trace:
                _last_scripted_input_trace = trace
                diagnostic_print(
                    "[dim]Opening scripted input: operation='press A' "
                    f"inputs_before={before_inputs} inputs_after={after_inputs} "
                    f"pending_before={before_pending} pending_after={after_pending}[/]",
                    trace=True,
                )
    yield


def _startup_dialogue_waiting(observed: OpeningSequenceState) -> bool:
    """Detect only ordinary Emerald field dialogue, never a generic script wait."""
    if observed in (
        OpeningSequenceState.PLAYER_NAMING,
        OpeningSequenceState.CLOCK_SETTING,
        OpeningSequenceState.STARTER_SELECTION,
    ):
        return False
    try:
        if get_game_state() != GameState.OVERWORLD:
            return False
        return context.rom.is_emerald and is_field_message_waiting_for_input()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _scripted_std_msgbox_waiting(state: tuple) -> bool:
    """Recognize Emerald's script-owned standard message input wait.

    ``Task_DrawFieldMessage`` is not present for every ROM message lifecycle.
    Require the script and native symbols together with the sampled input-wait
    state so an unrelated ``WaitForAorBPress`` native wait is not dialogue.
    """
    return state[2] is True and state[3] == "WaitForAorBPress" and state[4] == "Std_MsgboxDefault" and state[5] is True


def _birch_house_1f_ready_for_navigation() -> bool:
    """Whether the arrival event in Birch's house has released the player."""
    try:
        if get_game_state() != GameState.OVERWORLD or not player_avatar_is_controllable():
            return False
        script_context = get_global_script_context()
        return script_context is None or not script_context.is_active
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return False


def _enter_player_name(player_name: str) -> Generator:
    """Wait for the naming keyboard to exist, including pointer/task gaps."""
    while True:
        if get_game_state() != GameState.NAMING_SCREEN:
            return
        try:
            if get_naming_screen_data() is None:
                yield
                continue
            yield from type_in_naming_screen(player_name)
            return
        except (BotModeError, AttributeError, RuntimeError, ValueError, TypeError):
            # The naming callback can outlive the keyboard data by a frame.
            # This is a normal transition, not an automation failure.
            yield


def _start_game_configuration_values() -> tuple[str, str]:
    """Read Start Game settings while remaining compatible with minimal test contexts."""
    config = getattr(context, "config", None)
    start_game = getattr(config, "start_game", None)
    player_name = getattr(start_game, "player_name", "gibberish")
    player_gender = getattr(start_game, "player_gender", "random")
    if not isinstance(player_name, str):
        player_name = "gibberish"
    if not isinstance(player_gender, str):
        player_gender = "random"
    return player_name, player_gender


def _clock_time_mode() -> WallClockTimeMode:
    config = getattr(context, "config", None)
    start_game = getattr(config, "start_game", None)
    configured = getattr(start_game, "clock_time_mode", WallClockTimeMode.SYSTEM_TIME)
    try:
        return configured if isinstance(configured, WallClockTimeMode) else WallClockTimeMode(configured)
    except (TypeError, ValueError):
        return WallClockTimeMode.SYSTEM_TIME


def _emerald_clock_time(
    mode: WallClockTimeMode,
    *,
    now: datetime | None = None,
    rng: RandomSource | None = None,
) -> tuple[int, int]:
    """Resolve one target in Emerald's internal 24-hour task representation."""
    if mode is WallClockTimeMode.RANDOM:
        if rng is None:
            rng = random.Random()
        return (
            rng.randint(_CLOCK_HOUR_MIN, _CLOCK_HOUR_MAX),
            rng.randint(_CLOCK_MINUTE_MIN, _CLOCK_MINUTE_MAX),
        )

    snapshot = datetime.now() if now is None else now
    # The visible wall-clock UI is 12-hour, but the ROM task stores 0..23.
    return snapshot.hour, snapshot.minute


class EmeraldOpeningMode(BotMode):
    @staticmethod
    def name() -> str:
        return "Start New Game"

    @staticmethod
    def is_selectable() -> bool:
        return context.rom.is_emerald

    def __init__(self, rng: RandomSource | None = None):
        super().__init__()
        self._rng = random.Random() if rng is None else rng
        # Resolve policy once per mode run; do not regenerate on naming-screen frames.
        self._start_game_initialization = resolve_start_game_initialization(
            *_start_game_configuration_values(), self._rng
        )
        self._resolved_player_gender = self._start_game_initialization.gender
        self._resolved_player_name = self._start_game_initialization.name
        self.phase = OpeningSequenceState.TRUCK
        self._ball_coordinates: tuple[int, int] | None = None
        self._last_diagnostics: OpeningDiagnostics | None = None
        self._last_truck_decision: str | None = None
        self._last_truck_exit_warp: tuple[int, int] | None = None
        self._last_truck_exit_destination: tuple[int, int] | None = None
        self._last_truck_navigation_target: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._pending_house_warp_destination: MapRSE | None = None
        self._clock_interaction_started = False
        self._clock_target: tuple[int, int] | None = None
        self._last_clock_task: str | None = None
        self._clock_a_sent = False
        self._clock_confirm_yes_prepared = False
        self._last_dialogue_key: tuple | None = None
        self._dialogue_active = False
        self._dialogue_before_input: tuple | None = None
        self._dialogue_wait_input_state: tuple | None = None
        self._last_dialogue_detection_key: tuple | None = None
        self._last_route101_lifecycle_key: tuple | None = None
        self._last_phase_dispatch_key: tuple | None = None
        self._last_may_sequence_key: tuple | None = None
        self._last_route101_readiness_key: tuple | None = None
        self._last_route101_runtime_key: tuple | None = None
        self._initial_settings_configured = False
        self._initial_options_entered = False
        self._initial_options_cursor_positioned = False
        self._initial_menu_repositioned = False
        self._last_text_speed_configuration_key: tuple | None = None
        self._birch_gender_a_sent = False

    @staticmethod
    def _active_birch_gender_task() -> str | None:
        try:
            active_tasks = {task.symbol for task in (get_tasks() or [])}
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            return None
        # ChooseGender is the actionable state; the other tasks describe
        # setup/transition frames around the menu.
        for task_name in _BIRCH_GENDER_TASKS:
            if task_name in active_tasks:
                return task_name
        return None

    def _advance_birch_gender_selection(self, observed: OpeningSequenceState) -> Generator:
        """Give Birch's gender menu precedence over generic opening states."""
        task_name = self._active_birch_gender_task()
        if task_name is None:
            self._birch_gender_a_sent = False
            return False

        if task_name != "Task_NewGameBirchSpeech_ChooseGender":
            yield
            return True

        if self._birch_gender_a_sent:
            yield
            return True

        task = get_task(task_name)
        if task is None:
            yield
            return True

        cursor = task.data_value(6)
        target = 1 if getattr(self._resolved_player_gender, "value", self._resolved_player_gender) == "female" else 0
        if cursor == target:
            _report_opening_a_decision(
                source="EmeraldOpeningMode._advance_birch_gender_selection",
                reason=f"ChooseGender cursor {cursor} matches target {target}; confirm selection",
                phase=observed,
            )
            _report_birch_gender_input_attempt(
                path="_advance_birch_gender_selection",
                button="A",
                observed=observed,
            )
            context.emulator.press_button("A")
            self._birch_gender_a_sent = True
        else:
            button = "Down" if target > cursor else "Up"
            context.emulator.press_button(button)
        yield
        return True

    def run(self) -> Generator:
        if not context.rom.is_emerald:
            raise BotModeError("Start New Game only supports Pokémon Emerald.")

        while context.bot_mode != "Manual":
            observed = get_opening_sequence_state(self._resolved_player_gender)
            global _birch_gender_diagnostic_phase, _birch_gender_diagnostic_target
            _birch_gender_diagnostic_phase = observed.name
            _birch_gender_diagnostic_target = self._resolved_player_gender
            diagnostics = get_opening_diagnostics(self.phase, observed)
            self._report_diagnostics(diagnostics)
            self._report_dialogue_detection(observed)
            self._report_route101_lifecycle(observed)
            self._report_route101_runtime(observed)
            if observed is OpeningSequenceState.COMPLETE:
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=observed,
                    dialogue_action=False,
                    dispatch="complete",
                    decision="opening complete: starter received and returned to Birch's lab",
                )
                return
            if observed is OpeningSequenceState.STARTER_SELECTION:
                global _starter_handoff_pending
                _starter_handoff_pending = True
                context.bot_mode = "Starters"
                return

            birch_gender_action = yield from self._advance_birch_gender_selection(observed)
            if birch_gender_action:
                continue

            if observed == OpeningSequenceState.TITLE or observed == OpeningSequenceState.MAIN_MENU:
                if observed is OpeningSequenceState.TITLE:
                    _report_opening_a_decision(
                        source="EmeraldOpeningMode.run:title branch",
                        reason="TITLE state requires dismissing the title screen",
                        phase=observed,
                    )
                    context.emulator.press_button("A")
                    yield
                    continue

                if not self._initial_settings_configured:
                    yield from self._configure_initial_game_settings()
                    self._initial_settings_configured = True
                    continue

                # Returning from Options leaves Emerald's main-menu cursor on
                # Option. Move back to New Game before using the original
                # opening handoff. If FAST was already selected, the cursor
                # never left New Game and no repositioning is needed.
                if self._initial_options_entered and not self._initial_menu_repositioned:
                    while not task_is_active("Task_HandleMainMenuInput"):
                        yield
                    context.emulator.press_button("Up")
                    self._initial_menu_repositioned = True
                    yield
                    continue

                _report_opening_a_decision(
                    source="EmeraldOpeningMode.run:main-menu branch",
                    reason="main-menu cursor is positioned on New Game",
                    phase=observed,
                )
                context.emulator.press_button("A")
                yield
                continue
            if observed == OpeningSequenceState.PLAYER_NAMING:
                yield from _enter_player_name(self._resolved_player_name)
                continue
            if observed == OpeningSequenceState.CLOCK_SETTING:
                yield from self._set_clock()
                continue

            # Dialogue is controller-wide: it can begin on arrival, after a
            # warp, or while a map phase is otherwise stationary.
            dialogue_action = yield from self._advance_startup_dialogue(observed)
            if dialogue_action:
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=True,
                    dispatch="dialogue",
                )
                continue

            self._report_phase_dispatch(
                observed,
                phase_before=self.phase,
                resulting_phase=self.phase,
                dialogue_action=False,
                dispatch=self.phase.name,
            )
            phase_completed = yield from self._advance_phase(observed, diagnostics)
            if phase_completed:
                return

    @staticmethod
    def _message_speed_observation() -> tuple[int | None, int | None, int | None, str | None]:
        """Read temporary Options state alongside persisted SaveBlock2 state."""
        temporary_value = None
        candidate_symbol = None
        for task_name in _EMERALD_OPTIONS_TASKS:
            try:
                task = get_task(task_name)
                if task is not None:
                    temporary_value = task.data_value(1)
                    candidate_symbol = f"{task_name}.data[1]"
                    break
            except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
                pass

        try:
            persisted_bytes = get_save_block(
                2,
                offset=_EMERALD_TEXT_SPEED_SAVE_BLOCK2_OFFSET,
                size=2,
            )
            persisted_value = unpack_uint16(persisted_bytes) & _EMERALD_TEXT_SPEED_MASK
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
            persisted_value = None

        try:
            save_block_pointer = unpack_uint32(read_symbol("gSaveBlock2Ptr", size=4))
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
            save_block_pointer = None
        observed_value = temporary_value if temporary_value is not None else persisted_value
        if candidate_symbol is None and persisted_value is not None:
            candidate_symbol = "SaveBlock2.optionsTextSpeed"
        return observed_value, persisted_value, save_block_pointer, candidate_symbol

    @classmethod
    def _message_speed(cls) -> int | None:
        """Read Emerald's live Options-menu text-speed value.

        Emerald persists the Options values in SaveBlock2, but the open
        Options menu edits ``Task_OptionMenuProcessInput.data[1]`` first.
        The similarly named ``sTextSpeed`` symbol is part of recorded-battle
        state and is not the setting currently displayed by the menu.
        """
        return cls._message_speed_observation()[0]

    @staticmethod
    def _message_speed_name(raw_value: int | None) -> str:
        try:
            return EmeraldTextSpeed(raw_value).name
        except (TypeError, ValueError):
            return "UNKNOWN"

    def _report_text_speed_configuration(
        self,
        observation: tuple[int | None, int | None, int | None, str | None],
        decision: str,
    ) -> None:
        """Report setting observations only when the decision changes."""
        raw_value, persisted_value, save_block_pointer, candidate_symbol = observation
        key = (raw_value, persisted_value, save_block_pointer, candidate_symbol, decision)
        if key == self._last_text_speed_configuration_key:
            return
        self._last_text_speed_configuration_key = key
        diagnostic_print(
            lambda: (
                "[dim]Text speed configuration: "
                f"candidate_symbol={candidate_symbol!r} "
                f"observed_value={raw_value!r} "
                f"observed_name={self._message_speed_name(raw_value)} "
                f"save_block_value={persisted_value!r} "
                f"save_block2={None if save_block_pointer is None else hex(save_block_pointer)} "
                f"decision={decision}[/]"
            )
        )

    def _configure_initial_game_settings(self) -> Generator:
        """Ensure required settings before entering the fresh-game flow.

        Emerald's initial menu opens with New Game selected and its Options
        entry immediately below it. The Options screen opens on Text Speed,
        so the only setting needed here can be changed through the normal UI.
        This small generator is intentionally independent of the opening
        phases so future save-based initializers can reuse the same check and
        setting operation.
        """
        observation = self._message_speed_observation()
        if observation[0] == EmeraldTextSpeed.FAST:
            self._report_text_speed_configuration(observation, "complete: already FAST")
            return
        self._report_text_speed_configuration(observation, "wait: enter Options")

        while not task_is_active("Task_HandleMainMenuInput"):
            yield
        context.emulator.press_button("Down")
        self._initial_options_cursor_positioned = True
        self._report_text_speed_configuration(observation, "press Down: select Options")
        yield

        # The save-file check can briefly reclaim the main-menu task after a
        # cursor move.  Wait for the menu's input task to be active again so
        # this A press is an actual selection, not an ignored transition-frame
        # input.
        while not task_is_active("Task_HandleMainMenuInput"):
            self._report_text_speed_configuration(observation, "wait: main-menu input task")
            yield
        _report_opening_a_decision(
            source="EmeraldOpeningMode._configure_initial_game_settings",
            reason="main-menu input task active after selecting Options",
            phase=OpeningSequenceState.MAIN_MENU,
        )
        context.emulator.press_button("A")
        self._initial_options_entered = True
        self._report_text_speed_configuration(observation, "press A: enter Options")
        yield

        while not task_is_active("Task_OptionMenuProcessInput"):
            self._report_text_speed_configuration(observation, "wait: Options input task")
            yield

        while True:
            observation = self._message_speed_observation()
            if observation[0] == EmeraldTextSpeed.FAST:
                self._report_text_speed_configuration(observation, "complete: FAST observed")
                break
            self._report_text_speed_configuration(observation, "press Right: advance text speed")
            context.emulator.press_button("Right")
            yield

        context.emulator.press_button("B")
        self._report_text_speed_configuration(observation, "press B: save and exit Options")
        while get_game_state() != GameState.MAIN_MENU:
            self._report_text_speed_configuration(observation, "wait: return to main menu")
            yield

    def _advance_startup_dialogue(self, observed: OpeningSequenceState) -> Generator:
        """Advance one page of ordinary field dialogue and yield one frame."""
        # Route 101's rescue completion marker supersedes a stale native wait
        # only after the ROM has released the avatar.  Before that boundary,
        # a qualified Std_MsgboxDefault wait remains actionable dialogue.
        if observed is OpeningSequenceState.ROUTE_101 and _route101_ready_for_navigation():
            return False
        state = self._dialogue_state_snapshot()
        self._report_dialogue_handler(observed, "enter", state)
        if state[0] and not self._dialogue_active:
            # The draw task establishes the field-message lifecycle even
            # before the printer/native reaches an input state.
            self._dialogue_active = True
            self._report_dialogue(observed, "field message lifecycle started")
        field_message_waiting = self._dialogue_detection(observed, state)[0]
        script_waiting = _scripted_std_msgbox_waiting(state)
        if not script_waiting and self._dialogue_wait_input_state is not None:
            self._report_dialogue_handler(observed, "clear script-wait key after leaving native wait", state)
            self._dialogue_wait_input_state = None
        if field_message_waiting or script_waiting:
            if not self._dialogue_active:
                self._report_dialogue(observed, "detected field message waiting for input")
                self._dialogue_active = True
            # The native function and script PC identify the kind of wait,
            # not a particular page of text.  Emerald can expose the same
            # wait state for several consecutive pages, so input readiness is
            # the semantic signal that another dismissal is required.
            self._press_dialogue_input(state, observed)
            self._dialogue_before_input = state
            if script_waiting:
                self._dialogue_wait_input_state = state
            self._report_dialogue(observed, "input B")
            self._report_dialogue_handler(observed, "return after input B", state)
            yield True
            return
        if self._dialogue_active:
            state = self._dialogue_state_snapshot()
            if state[0]:
                self._report_dialogue(observed, "message task still active; waiting for script")
                self._report_dialogue_handler(observed, "return: task still active", state)
                yield True
                return
            if state[3] == "IsFieldMessageBoxHidden" or state[7]:
                self._report_dialogue(observed, "message box still visible; waiting for hide transition")
                self._report_dialogue_handler(observed, "return: message still visible", state)
                yield True
                return
            else:
                self._report_dialogue(observed, "message complete")
            self._dialogue_active = False
            self._dialogue_before_input = None
            self._dialogue_wait_input_state = None
            self._last_dialogue_key = None
            self._report_dialogue_handler(observed, "return: dialogue complete", state)
        else:
            self._report_dialogue_handler(observed, "return: no dialogue", state)
        return False

    def _report_route101_lifecycle(self, observed: OpeningSequenceState) -> None:
        """Report meaningful Route 101 message/control boundaries only."""
        if not context.debug or observed is not OpeningSequenceState.ROUTE_101:
            return
        state = self._dialogue_state_snapshot()
        detected, reason = self._dialogue_detection(observed, state)
        ready = _route101_ready_for_navigation()
        if ready:
            lifecycle = "route navigation ready; player controllable again"
        elif state[0] and not state[5]:
            lifecycle = "scripted field message active; not awaiting player input"
        elif state[2] and state[3] == "WaitForAorBPress" and not state[5]:
            lifecycle = "scripted event active; native wait but no input requested"
        elif state[2] and not state[0] and not state[7]:
            lifecycle = "scripted event active; no message"
        elif detected:
            lifecycle = "dialogue detected; actionable player input required"
        elif self._dialogue_active and not state[0] and not state[7]:
            lifecycle = "message/dialogue completed; awaiting control return"
        else:
            lifecycle = "scripted event transition"
        key = (
            lifecycle,
            state,
            detected,
            reason,
            ready,
        )
        if key == self._last_route101_lifecycle_key:
            return
        self._last_route101_lifecycle_key = key
        diagnostic_print(
            lambda: (
                "[dim]Opening Route 101 lifecycle: "
                f"state={lifecycle!r} detected={detected} reason={reason!r} "
                f"controllable={state[6]} game_message_task={state[0]} "
                f"message_task_state={state[1]!r} message_visible={state[7]} "
                f"printer_state={state[8]!r} waiting={state[5]} "
                f"script_active={state[2]} native={state[3]!r} "
                f"script={state[4]!r} active_tasks={state[9]!r} "
                f"route101_ready={ready}[/]"
            )
        )

    def _report_route101_readiness(self, observed: OpeningSequenceState) -> None:
        """Report each Route 101 gate input only when the snapshot changes."""
        if not context.debug or observed is not OpeningSequenceState.ROUTE_101:
            return
        state = self._dialogue_state_snapshot()
        try:
            route_state = get_event_var("ROUTE101_STATE")
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            route_state = None
        try:
            game_state = get_game_state()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            game_state = None
        ready = route_state is not None and route_state >= 2 and state[6] is True
        key = (self.phase, observed, route_state, game_state, state, ready)
        if key == getattr(self, "_last_route101_readiness_key", None):
            return
        self._last_route101_readiness_key = key
        diagnostic_print(
            lambda: (
                "[dim]Opening Route 101 readiness: "
                f"ready={ready} route101_state={route_state!r} "
                f"script_active={state[2]} native={state[3]!r} "
                f"native_ptr={state[10]!r} script={state[4]!r} "
                f"script_pc={state[11]!r} game_state={game_state!r} "
                f"controllable={state[6]} waiting={state[5]} "
                f"field_message_task_active={state[0]} "
                f"field_message_task_state={state[1]!r} "
                f"message_visible={state[7]} printer_state={state[8]!r} "
                f"active_tasks={state[9]!r} "
                "predicate=route101_state>=2 and controllable[/]"
            )
        )

    def _report_route101_runtime(self, observed: OpeningSequenceState) -> None:
        """Emit one complete Route 101 runtime snapshot per state change."""
        if not context.debug or observed is not OpeningSequenceState.ROUTE_101:
            return
        state = self._dialogue_state_snapshot()
        route_state_error = None
        try:
            route_state = get_event_var("ROUTE101_STATE")
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError) as error:
            route_state = None
            route_state_error = repr(error)
        controllable_error = None
        try:
            controllable = player_avatar_is_controllable()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError) as error:
            controllable = None
            controllable_error = repr(error)
        map_object_exists = None
        map_object_flags = None
        forced_move = None
        try:
            map_object = get_player_map_object()
            map_object_exists = map_object is not None
            map_object_flags = None if map_object is None else tuple(sorted(map_object.flags))
            forced_move = AvatarFlags.ForcedMove in get_player_avatar().flags
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError) as error:
            controllable_error = controllable_error or repr(error)
        try:
            avatar = get_player_avatar()
            coordinates = avatar.local_coordinates
            facing = avatar.facing_direction
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            coordinates = None
            facing = None
        try:
            map_id = _current_map_id()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            map_id = None
        try:
            game_state = get_game_state()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            game_state = None
        try:
            inputs = context.emulator.get_inputs()
        except (AttributeError, RuntimeError, TypeError):
            inputs = None
        input_state = (inputs, self._dialogue_pending_inputs())
        stage = _route101_observation_stage(route_state, state)
        key = (
            stage,
            route_state,
            route_state_error,
            controllable,
            controllable_error,
            map_object_exists,
            map_object_flags,
            forced_move,
            map_id,
            coordinates,
            facing,
            game_state,
            state,
            input_state,
        )
        if key == self._last_route101_runtime_key:
            return
        self._last_route101_runtime_key = key
        console.print(
            (
                lambda: (
                    "Opening Route 101 runtime source=working-tree/opening.py "
                    f"stage={stage!r} route101_state={route_state!r} "
                    f"route_state_error={route_state_error!r} "
                    f"map={map_id!r} coords={coordinates!r} facing={facing!r} "
                    f"game_state={game_state!r} "
                    f"controllable={controllable!r} "
                    f"controllable_snapshot={state[6]!r} "
                    f"controllable_error={controllable_error!r} "
                    f"map_object_exists={map_object_exists!r} "
                    f"map_object_flags={map_object_flags!r} forced_move={forced_move!r} "
                    f"script_active={state[2]} script={state[4]!r} "
                    f"native={state[3]!r} native_ptr={state[10]!r} "
                    f"script_pc={state[11]!r} field_task_active={state[0]} "
                    f"field_task_state={state[1]!r} message_visible={state[7]} "
                    f"printer_state={state[8]!r} waiting={state[5]} "
                    f"active_tasks={state[9]!r} inputs={input_state!r}"
                )
            )()
        )

    def _report_dialogue_handler(
        self,
        observed: OpeningSequenceState,
        event: str,
        state: tuple,
    ) -> None:
        """Trace handler entry/return decisions at the ROM wait boundary."""
        diagnostic_print(
            lambda: (
                "[dim]Opening dialogue handler: "
                f"event={event!r} observed={observed.name} "
                f"task_active={state[0]} waiting={state[5]} visible={state[7]} "
                f"native={state[3]!r} native_ptr={state[10]!r} script_pc={state[11]!r} "
                f"dialogue_active={self._dialogue_active} "
                f"before_input={self._dialogue_before_input!r} "
                f"wait_input_state={self._dialogue_wait_input_state!r}[/]"
            ),
            trace=True,
        )

    def _report_phase_dispatch(
        self,
        observed: OpeningSequenceState,
        *,
        phase_before: OpeningSequenceState,
        resulting_phase: OpeningSequenceState,
        dialogue_action: bool,
        dispatch: str,
        reconciled: bool = False,
        decision: str | None = None,
        navigation_target: tuple | None = None,
    ) -> None:
        """Trace controller dispatch separately from dialogue diagnostics."""
        if not context.debug:
            return
        key = (
            phase_before,
            observed,
            reconciled,
            resulting_phase,
            dialogue_action,
            dispatch,
            decision,
            navigation_target,
        )
        if key == self._last_phase_dispatch_key:
            return
        self._last_phase_dispatch_key = key
        console.print(
            "[dim]Opening phase dispatch: "
            f"phase={phase_before.name} observed={observed.name} "
            f"reconciled={reconciled} resulting_phase={resulting_phase.name} "
            f"dialogue_action={dialogue_action} dispatch={dispatch!r} "
            f"navigation_target={navigation_target!r} decision={decision!r}[/]"
        )

    def _report_may_sequence_state(
        self,
        observed: OpeningSequenceState,
        *,
        branch: str,
        reason: str,
        navigation_target: tuple | None = None,
    ) -> None:
        """Trace state changes while handing control back after Rival's event."""
        if not context.debug:
            return
        state = self._dialogue_state_snapshot()
        try:
            map_id = _current_map_id()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            map_id = None
        try:
            coordinates = get_player_avatar().local_coordinates
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            coordinates = None
        try:
            game_state = get_game_state()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            game_state = None
        event_complete = _may_sequence_event_complete()
        key = (
            self.phase,
            observed,
            map_id,
            coordinates,
            game_state,
            state[2],
            state[3],
            state[4],
            state[6],
            event_complete,
            branch,
            reason,
            navigation_target,
        )
        if key == self._last_may_sequence_key:
            return
        self._last_may_sequence_key = key
        console.print(
            "[dim]Opening MAY_SEQUENCE: "
            f"phase={self.phase.name} observed={observed.name} map={map_id} "
            f"coords={coordinates} game_state={game_state} controllable={state[6]} "
            f"script_active={state[2]} native={state[3]!r} script={state[4]!r} "
            f"rival_event_complete={event_complete} branch={branch!r} "
            f"reason={reason!r} navigation_target={navigation_target!r}[/]"
        )

    def _press_dialogue_input(self, state: tuple, observed: OpeningSequenceState | None = None) -> None:
        if not context.debug or not getattr(context, "debug_trace", False):
            context.emulator.press_button("B")
            return
        before_inputs = self._dialogue_emulator_inputs()
        before_pending = self._dialogue_pending_inputs()
        _report_birch_gender_input_attempt(
            path="_press_dialogue_input",
            button="B",
            observed=observed,
            before_inputs=before_inputs,
            before_pending=before_pending,
        )
        context.emulator.press_button("B")
        after_inputs = self._dialogue_emulator_inputs()
        after_pending = self._dialogue_pending_inputs()
        self._report_dialogue_input(
            state,
            "press B",
            before_inputs=before_inputs,
            after_inputs=after_inputs,
            before_pending=before_pending,
            after_pending=after_pending,
        )

    @staticmethod
    def _dialogue_emulator_inputs() -> int | None:
        try:
            return context.emulator.get_inputs()
        except (AttributeError, RuntimeError, TypeError):
            return None

    @staticmethod
    def _dialogue_pending_inputs() -> tuple[int | None, int | None, int | None]:
        try:
            emulator = context.emulator
            return (
                getattr(emulator, "_prev_pressed_inputs", None),
                getattr(emulator, "_pressed_inputs", None),
                getattr(emulator, "_held_inputs", None),
            )
        except (AttributeError, RuntimeError, TypeError):
            return None, None, None

    def _report_dialogue_input(
        self,
        state: tuple,
        operation: str,
        *,
        before_inputs: int | None = None,
        after_inputs: int | None = None,
        before_pending: tuple[int | None, int | None, int | None] | None = None,
        after_pending: tuple[int | None, int | None, int | None] | None = None,
    ) -> None:
        if not getattr(context, "debug_trace", False):
            return
        if before_inputs is None:
            before_inputs = self._dialogue_emulator_inputs()
        if after_inputs is None:
            after_inputs = self._dialogue_emulator_inputs()
        if before_pending is None:
            before_pending = self._dialogue_pending_inputs()
        if after_pending is None:
            after_pending = self._dialogue_pending_inputs()
        diagnostic_print(
            lambda: self._format_dialogue_input_trace(
                state,
                operation,
                before_inputs,
                after_inputs,
                before_pending,
                after_pending,
            ),
            trace=True,
        )

    def _format_dialogue_input_trace(
        self,
        state: tuple,
        operation: str,
        before_inputs: int | None,
        after_inputs: int | None,
        before_pending: tuple[int | None, int | None, int | None],
        after_pending: tuple[int | None, int | None, int | None],
    ) -> str:
        """Build the expensive before/after input trace only for tracing."""
        after_state = self._dialogue_state_snapshot()
        return (
            "[dim]Opening dialogue input: "
            f"operation={operation!r} requested=B "
            f"inputs_before={before_inputs} inputs_after={after_inputs} "
            f"pending_before={before_pending} pending_after={after_pending} "
            f"native_before={state[3]!r} script_before={state[4]!r} "
            f"native_ptr_before={state[10]!r} script_pc_before={state[11]!r} "
            f"native_after={after_state[3]!r} script_after={after_state[4]!r} "
            f"native_ptr_after={after_state[10]!r} script_pc_after={after_state[11]!r}[/]"
        )

    def _dialogue_detection(
        self,
        observed: OpeningSequenceState,
        state: tuple | None = None,
    ) -> tuple[bool, str]:
        """Return the reusable predicate result and a diagnostic reason."""
        if observed in (
            OpeningSequenceState.PLAYER_NAMING,
            OpeningSequenceState.CLOCK_SETTING,
            OpeningSequenceState.STARTER_SELECTION,
        ):
            return False, "specialized UI excluded"
        if state is None:
            state = self._dialogue_state_snapshot()
        try:
            game_state = get_game_state()
            if game_state != GameState.OVERWORLD:
                return False, f"game state is {game_state}, not OVERWORLD"
            if not context.rom.is_emerald:
                return False, "ROM is not Emerald"
            detected = is_field_message_waiting_for_input(self._dialogue_active or state[0])
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            return False, "task/script/message state unavailable"
        if detected:
            if state[0]:
                return True, "Task_DrawFieldMessage is active and input is waiting"
            if state[3] == "WaitForAorBPress" and state[7]:
                return True, "WaitForAorBPress with visible field message"
            return True, "field-message predicate reported actionable input"
        if _scripted_std_msgbox_waiting(state):
            return True, "Std_MsgboxDefault is waiting for A/B input"
        if not state[0] and state[3] == "WaitForAorBPress":
            return False, "WaitForAorBPress is outside an observed field-message lifecycle"
        if not state[0]:
            return False, "Task_DrawFieldMessage inactive"
        if not state[5]:
            return False, "field task active but input is not waiting"
        return False, "field-message predicate rejected current state"

    def _report_dialogue_detection(self, observed: OpeningSequenceState) -> None:
        """Emit state-change-only diagnostics for the cross-phase predicate."""
        if not context.debug:
            return
        state = self._dialogue_state_snapshot()
        detected, reason = self._dialogue_detection(observed, state)
        try:
            game_state = get_game_state()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            game_state = None
        handler_candidate = detected or (_scripted_std_msgbox_waiting(state))
        map_id = _current_map_id()
        coordinates = None
        try:
            coordinates = get_player_avatar().local_coordinates
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        key = (
            self.phase,
            observed,
            map_id,
            coordinates,
            state,
            detected,
            reason,
            handler_candidate,
        )
        if key == self._last_dialogue_detection_key:
            return
        self._last_dialogue_detection_key = key
        console.print(
            "[dim]Opening dialogue detection: "
            f"phase={self.phase.name} observed={observed.name} map={map_id} coords={coordinates} "
            f"game_state={game_state} "
            f"script_active={None if state is None else state[2]} "
            f"native={None if state is None else state[3]} "
            f"context_available={None if state is None else state[2] is not None} "
            f"native_ptr={None if state is None else state[10]} "
            f"script_pc={None if state is None else state[11]} "
            f"active_tasks={None if state is None else state[9]} "
            f"field_task_active={None if state is None else state[0]} "
            f"field_task_state={None if state is None else state[1]} "
            f"message_visible={None if state is None else state[7]} "
            f"printer_state={None if state is None else state[8]} "
            f"waiting={None if state is None else state[5]} "
            f"predicate={detected} reason={reason!r} "
            f"handler_entered={handler_candidate}[/]"
        )

    @staticmethod
    def _dialogue_state_snapshot() -> tuple:
        """Capture volatile state useful for diagnosing the final message edge."""
        task_active = False
        task_state = None
        script_active = None
        native = None
        script = None
        waiting = None
        controllable = None
        printer_active = None
        printer_state = None
        active_tasks = None
        native_pointer = None
        bytecode_pointer = None
        try:
            task = get_task("Task_DrawFieldMessage")
            task_active = task is not None
            if task is not None:
                task_state = task.data_value(0)
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        try:
            context_data = get_global_script_context()
            if context_data is not None:
                script_active = context_data.is_active
                native = context_data.native_function_name
                script = context_data.script_function_name
                native_pointer = context_data.native_pointer
                bytecode_pointer = context_data.bytecode_pointer
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        try:
            waiting = is_waiting_for_input()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        try:
            printer = get_text_printer()
            printer_active = printer.active
            printer_state = printer.raw_state
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        try:
            active_tasks = tuple(task.symbol for task in get_tasks())
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        try:
            controllable = player_avatar_is_controllable()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        # IsFieldMessageBoxHidden is the ROM's own closing/visibility wait;
        # unlike task lifetime, it is evidence that the field box has not yet
        # reached its hidden state.
        message_visible = native == "IsFieldMessageBoxHidden" or bool(printer_active)
        return (
            task_active,
            task_state,
            script_active,
            native,
            script,
            waiting,
            controllable,
            message_visible,
            printer_state,
            active_tasks,
            native_pointer,
            bytecode_pointer,
        )

    def _advance_phase(
        self,
        observed: OpeningSequenceState,
        diagnostics: OpeningDiagnostics | None = None,
    ) -> Generator:
        player_1f, player_2f = (
            _player_house_map(1, self._resolved_player_gender),
            _player_house_map(2, self._resolved_player_gender),
        )
        rival_1f, rival_2f = (
            _rival_house_map(1, self._resolved_player_gender),
            _rival_house_map(2, self._resolved_player_gender),
        )
        if self.phase is OpeningSequenceState.PLAYER_HOUSE_1F_POST_CLOCK_ARRIVAL:
            if not _post_clock_1f_arrival_complete(self._resolved_player_gender):
                self._last_truck_decision = "wait: expected post-clock 1F TV event"
                # Ordinary field dialogue is handled by run() before this
                # phase is advanced.  This yield covers the ROM's scripted
                # movement and the frame between map arrival and OnFrame.
                yield
                return
            self.phase = OpeningSequenceState.PLAYER_HOUSE_1F
            self._last_truck_decision = "complete: post-clock 1F TV event"
            yield
            return

        if self.phase is OpeningSequenceState.TRUCK:
            # The truck exit can start Mom's arrival script in the same frame in
            # which the ROM changes maps.  Once Littleroot is observable, the
            # truck waypoint is no longer meaningful, even while that script is
            # still moving/talking to the player.
            if observed is OpeningSequenceState.LITTLEROOT_TOWN:
                self._last_truck_navigation_target = None
                self._last_truck_exit_warp = None
                self._last_truck_exit_destination = None
                self._last_truck_decision = "complete: entered Littleroot; await arrival script"
                self.phase = OpeningSequenceState.LITTLEROOT_TOWN
                yield
                return
            if _normal_littleroot_overworld():
                self._last_truck_navigation_target = None
                self._last_truck_exit_warp = None
                self._last_truck_exit_destination = None
                self._last_truck_decision = "complete: normal Littleroot overworld"
                self.phase = OpeningSequenceState.LITTLEROOT_TOWN
            elif self._truck_can_use_exit(diagnostics):
                self._last_truck_navigation_target = (MapRSE.INSIDE_OF_TRUCK.value, (4, 1))
                self._last_truck_decision = "navigate: ROM truck exit tile (4,1); await game warp"
                yield from navigate_to(
                    MapRSE.INSIDE_OF_TRUCK.value,
                    (4, 1),
                    avoid_scripted_events=False,
                    expecting_script=True,
                )
            else:
                self._last_truck_decision = "wait: truck script/task or transitional state"
                yield
            return

        if self.phase is OpeningSequenceState.LITTLEROOT_TOWN:
            # Mom's forced movement/dialogue is expected here.  Let the ROM
            # script consume its own input sequence before starting navigation.
            if observed is OpeningSequenceState.PLAYER_HOUSE_1F:
                # The house warp may complete before the generator that started
                # it gets another controller observation.  Reconcile from the
                # observed map immediately so the arrival phase cannot remain
                # stale while the player is already inside the house.
                self.phase = OpeningSequenceState.PLAYER_HOUSE_1F
                self._last_truck_decision = "complete: entered player's house; navigate to stairs"
            elif not _littleroot_arrival_ready():
                yield from _advance_scripted_input()
                return
            else:
                yield from _warp_to(player_1f)
                self.phase = OpeningSequenceState.PLAYER_HOUSE_1F
                return

        if self.phase is OpeningSequenceState.PLAYER_HOUSE_1F:
            if self._pending_house_warp_destination is not None:
                destination = self._pending_house_warp_destination
                observed_map = _current_map_id()
                if observed_map != destination.value:
                    # The post-clock phase intentionally remains 1F while
                    # Mom's 2F sequence finishes.  A staircase navigation
                    # generator can return before the warp is observable, so
                    # do not turn its pending destination into a permanent
                    # wait state.  Once the ROM has released control, resume
                    # driving the same ROM-defined staircase warp.
                    if (
                        destination is player_1f
                        and observed_map == player_2f.value
                        and get_event_flag("SET_WALL_CLOCK")
                    ):
                        if not _post_clock_2f_ready_for_stair_navigation():
                            self._last_truck_decision = "wait: post-clock Mom sequence on player's house 2F"
                            yield
                            return
                        self._last_truck_decision = "navigate: ROM-defined staircase warp back to player's house 1F"
                        yield from _warp_to(destination)
                        return
                    self._last_truck_decision = f"wait: house warp to {destination.name}; observed map={observed_map}"
                    yield
                    return
                self._pending_house_warp_destination = None
                if destination is player_2f:
                    self.phase = OpeningSequenceState.PLAYER_HOUSE_2F
                elif destination is player_1f:
                    self.phase = OpeningSequenceState.PLAYER_HOUSE_1F_POST_CLOCK_ARRIVAL
                elif destination is MapRSE.LITTLEROOT_TOWN:
                    self.phase = OpeningSequenceState.POST_CLOCK_TOWN
                return

            if observed == OpeningSequenceState.PLAYER_HOUSE_1F and not get_event_flag("SET_WALL_CLOCK"):
                if not _player_house_1f_ready_for_navigation():
                    self._last_truck_decision = "wait: player's house 1F introductory dialogue"
                    yield from _advance_scripted_input()
                    return
                self._pending_house_warp_destination = player_2f
                self._last_truck_decision = "navigate: ROM-defined staircase warp to player's house 2F"
                yield from _warp_to(player_2f)
                return
            if observed == OpeningSequenceState.PLAYER_HOUSE_2F:
                if get_event_flag("SET_WALL_CLOCK"):
                    # The staircase is the only supported route back down;
                    # walking to it lets the map/path system handle the warp.
                    self._pending_house_warp_destination = player_1f
                    self._last_truck_decision = "navigate: ROM-defined staircase warp back to player's house 1F"
                    yield from _warp_to(player_1f)
                else:
                    self.phase = OpeningSequenceState.PLAYER_HOUSE_2F
            elif get_event_flag("SET_WALL_CLOCK") and observed == OpeningSequenceState.PLAYER_HOUSE_1F:
                self._pending_house_warp_destination = MapRSE.LITTLEROOT_TOWN
                self._last_truck_decision = "navigate: leave player's house after setting clock"
                yield from _warp_to(MapRSE.LITTLEROOT_TOWN)
            else:
                yield from _advance_scripted_input()
            return

        if self.phase is OpeningSequenceState.PLAYER_HOUSE_2F:
            self.phase = OpeningSequenceState.CLOCK_SETTING
            self._clock_interaction_started = False
            yield from self._start_clock_interaction()
            return

        if self.phase is OpeningSequenceState.CLOCK_SETTING:
            if not self._clock_interaction_started and not _clock_task_active():
                yield from self._start_clock_interaction()
                return
            # The clock interaction first displays a normal field message.
            # During that message the clock task is not active yet, so routing
            # directly to _set_clock() would wait for a fade forever even
            # though is_waiting_for_input() correctly reports True.
            if not _clock_task_active() and _scripted_input_waiting():
                yield from _advance_scripted_input()
                return
            yield from self._set_clock()
            # SET_WALL_CLOCK is written by the ROM before its final fade/exit
            # task has necessarily disappeared.  Keep ownership of the
            # opening until the complete task chain has ended.
            if _clock_setting_complete():
                self.phase = OpeningSequenceState.PLAYER_HOUSE_1F
            return

        if self.phase is OpeningSequenceState.POST_CLOCK_TOWN:
            if observed == OpeningSequenceState.LITTLEROOT_TOWN:
                # The rival's house starts a ROM-owned arrival event as the
                # warp completes.  Accept that expected interruption so the
                # generic dialogue handler can service it on the next frame.
                yield from _warp_to(rival_1f, expecting_script=True)
                self.phase = OpeningSequenceState.BIRCH_HOUSE_1F
            return

        if self.phase is OpeningSequenceState.BIRCH_HOUSE_1F:
            # The map transition can become observable one frame before the
            # staircase navigation generator returns.  Reconcile that
            # observed destination instead of treating it as an unrelated
            # scripted-input frame forever.
            if observed is OpeningSequenceState.BIRCH_HOUSE_2F:
                previous_phase = self.phase
                self.phase = OpeningSequenceState.BIRCH_HOUSE_2F
                self._report_phase_dispatch(
                    observed,
                    phase_before=previous_phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    reconciled=True,
                )
                return
            if observed is not OpeningSequenceState.BIRCH_HOUSE_1F:
                yield from _advance_scripted_input()
                return
            if not _birch_house_1f_ready_for_navigation():
                self._last_truck_decision = "wait: Birch's house arrival event"
                yield
                return
            yield from _warp_to(rival_2f)
            self.phase = OpeningSequenceState.BIRCH_HOUSE_2F
            return

        if self.phase is OpeningSequenceState.BIRCH_HOUSE_2F:
            if observed is not OpeningSequenceState.BIRCH_HOUSE_2F:
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    decision="wait: observed map is not Birch's 2F",
                )
                yield from _advance_scripted_input()
                return
            interaction = _rival_pokeball_interaction()
            if interaction is None:
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    decision="wait: Rival Poké Ball scripted object unavailable",
                )
                yield from _advance_scripted_input()
                return
            interaction_coordinates, interaction_facing = interaction
            self._ball_coordinates = interaction_coordinates
            if not self._can_navigate():
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    navigation_target=self._ball_coordinates,
                    decision="wait: player not controllable",
                )
                yield
                return
            self._last_truck_decision = "navigate: Poké Ball interaction"
            self._report_phase_dispatch(
                observed,
                phase_before=self.phase,
                resulting_phase=self.phase,
                dialogue_action=False,
                dispatch=self.phase.name,
                navigation_target=interaction_coordinates,
                decision=self._last_truck_decision,
            )
            yield from navigate_to(
                rival_2f,
                interaction_coordinates,
                avoid_scripted_events=False,
            )
            yield from ensure_facing_direction(interaction_facing)
            _report_opening_a_decision(
                source="EmeraldOpeningMode._advance_phase:May Poké Ball interaction",
                reason="navigation and facing completed at the interaction target",
                phase=observed,
            )
            context.emulator.press_button("A")
            yield
            self.phase = OpeningSequenceState.MAY_SEQUENCE
            return

        if self.phase is OpeningSequenceState.MAY_SEQUENCE:
            if observed in (OpeningSequenceState.LITTLEROOT_TOWN, OpeningSequenceState.ROUTE_101):
                self._report_may_sequence_state(
                    observed,
                    branch="transition to ROUTE_101",
                    reason="ROM has already moved player out of May's house",
                )
                self.phase = OpeningSequenceState.ROUTE_101
            elif observed in (
                OpeningSequenceState.BIRCH_HOUSE_2F,
                OpeningSequenceState.BIRCH_HOUSE_1F,
            ):
                if not _may_sequence_event_complete():
                    self._report_may_sequence_state(
                        observed,
                        branch="wait for Rival event",
                        reason="ROM movement/script has not released control",
                    )
                    yield
                    return
                if observed is OpeningSequenceState.BIRCH_HOUSE_2F:
                    self._report_may_sequence_state(
                        observed,
                        branch="navigate to May's House 1F warp",
                        reason="Rival event complete; descend from 2F",
                        navigation_target=rival_1f.value,
                    )
                    yield from _warp_to(
                        rival_1f,
                        expecting_script=True,
                    )
                else:
                    self._report_may_sequence_state(
                        observed,
                        branch="navigate to Littleroot Town warp",
                        reason="Rival event complete; leave May's house",
                        navigation_target=MapRSE.LITTLEROOT_TOWN.value,
                    )
                    yield from _warp_to(MapRSE.LITTLEROOT_TOWN, expecting_script=True)
                    self.phase = OpeningSequenceState.ROUTE_101
            else:
                self._report_may_sequence_state(
                    observed,
                    branch="wait for expected May-house map",
                    reason="observed map is outside post-Rival exit path",
                )
                yield
            return

        if self.phase is OpeningSequenceState.ROUTE_101:
            if observed not in (OpeningSequenceState.LITTLEROOT_TOWN, OpeningSequenceState.ROUTE_101):
                yield from _advance_scripted_input()
                return
            if observed is OpeningSequenceState.ROUTE_101:
                # Capture the gate inputs even while controllability itself is
                # the component keeping navigation paused.
                self._report_route101_readiness(observed)
            if not self._can_navigate():
                yield
                return
            if observed is OpeningSequenceState.LITTLEROOT_TOWN:
                # ROUTE_101 is the next objective, not proof that the player
                # has already reached that map.  The phase can be entered as
                # soon as Birch's house exit completes, while the physical
                # map observation remains Littleroot until this navigation
                # reaches the Route 101 entrance.
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    navigation_target=(10, 6),
                    decision="navigate: Route 101 entrance",
                )
                yield from navigate_to(
                    MapRSE.ROUTE101,
                    (10, 6),
                    avoid_encounters=False,
                    avoid_scripted_events=False,
                    expecting_script=True,
                )
                return
            # Only apply the Route 101 arrival-event gate after the player is
            # actually on Route 101.  Script/message state must never decide
            # whether to leave Littleroot Town.
            if not _route101_ready_for_navigation():
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    decision="wait: Route 101 arrival script/message still active",
                )
                yield
                return
            starter_interaction = _starter_bag_interaction()
            if starter_interaction is None:
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    decision="wait: Route 101 starter bag scripted object unavailable",
                )
                yield
                return
            starter_target, starter_bag = starter_interaction
            self._report_phase_dispatch(
                observed,
                phase_before=self.phase,
                resulting_phase=self.phase,
                dialogue_action=False,
                dispatch=self.phase.name,
                navigation_target=starter_target,
                decision="navigate: canonical starter bag handoff position",
            )
            yield from navigate_to(
                MapRSE.ROUTE101,
                starter_target,
                avoid_encounters=False,
                avoid_scripted_events=False,
                expecting_script=True,
            )
            try:
                reached_target = get_player_avatar().local_coordinates == starter_target
            except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
                reached_target = False
            if not reached_target:
                self._report_phase_dispatch(
                    observed,
                    phase_before=self.phase,
                    resulting_phase=self.phase,
                    dialogue_action=False,
                    dispatch=self.phase.name,
                    navigation_target=starter_target,
                    decision="wait: Route 101 navigation returned before target",
                )
                yield
                return
            yield from ensure_facing_direction(starter_bag)
            global _starter_handoff_pending
            _starter_handoff_pending = True
            self._report_phase_dispatch(
                observed,
                phase_before=OpeningSequenceState.ROUTE_101,
                resulting_phase=OpeningSequenceState.ROUTE_101,
                dialogue_action=False,
                dispatch="handoff",
                navigation_target=starter_target,
                decision="handoff: positioned and facing starter bag; opening complete",
            )
            context.bot_mode = "Starters"
            return True

        yield from _advance_scripted_input()

    def _start_clock_interaction(self) -> Generator:
        if not self._can_navigate():
            yield
            return
        house_map = _current_map_id()
        target = _wall_clock_interaction(house_map, self._resolved_player_gender)
        if house_map is None or target is None:
            self._last_truck_decision = f"wait: Emerald wall-clock target unavailable for map={house_map}"
            yield
            return
        coordinates, facing = target
        self._last_truck_decision = f"navigate: Emerald wall clock interaction at {coordinates}"
        yield from navigate_to(
            house_map,
            coordinates,
            avoid_scripted_events=False,
        )
        yield from ensure_facing_direction(facing)
        _report_opening_a_decision(
            source="EmeraldOpeningMode._start_clock_interaction",
            reason="navigation and facing completed at the wall clock",
            phase=OpeningSequenceState.CLOCK_SETTING,
        )
        context.emulator.press_button("A")
        self._clock_target = _emerald_clock_time(_clock_time_mode(), rng=self._rng)
        self._clock_interaction_started = True
        self._last_clock_task = None
        self._clock_a_sent = False
        self._clock_confirm_yes_prepared = False
        yield

    def _can_navigate(self) -> bool:
        try:
            return get_game_state() == GameState.OVERWORLD and player_avatar_is_controllable()
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return False

    def _truck_can_use_exit(self, diagnostics: OpeningDiagnostics | None = None) -> bool:
        try:
            if diagnostics is not None:
                # Use the same frame snapshot that produced the diagnostic.
                # Re-reading these volatile emulator values can turn a
                # controllable frame into a false transitional result.
                return (
                    diagnostics.map_id == MapRSE.INSIDE_OF_TRUCK.value
                    and diagnostics.game_state == GameState.OVERWORLD
                    and diagnostics.controllable is True
                    and diagnostics.script_active is False
                    and diagnostics.waiting_for_input is not None
                    and not diagnostics.waiting_for_input
                    and diagnostics.truck_sequence_active is False
                )
            return (
                _current_map_id() == MapRSE.INSIDE_OF_TRUCK.value
                and not _truck_sequence_active()
                and self._can_navigate()
            )
        except (AttributeError, RuntimeError, ValueError, TypeError):
            return False

    def _report_diagnostics(self, diagnostics: OpeningDiagnostics) -> None:
        if not context.debug:
            return
        if self._last_diagnostics is not None:
            previous_key = (
                self._last_diagnostics.phase,
                self._last_diagnostics.observed,
                self._last_diagnostics.map_id,
                self._last_diagnostics.game_state,
                self._last_diagnostics.controllable,
                self._last_diagnostics.script_active,
                self._last_diagnostics.waiting_for_input,
                self._last_diagnostics.facing,
                self._last_diagnostics.truck_flag,
                self._last_diagnostics.truck_sequence_active,
                self._last_diagnostics.active_script,
                self._last_diagnostics.truck_task,
                self._last_diagnostics.truck_task_state,
                self._last_diagnostics.input_accepted,
                self._last_truck_decision,
                self._last_truck_exit_warp,
                self._last_truck_exit_destination,
                self._last_truck_navigation_target,
                self._last_diagnostics.clock_task,
                self._last_diagnostics.clock_hours,
                self._last_diagnostics.clock_minutes,
            )
            current_key = (
                diagnostics.phase,
                diagnostics.observed,
                diagnostics.map_id,
                diagnostics.game_state,
                diagnostics.controllable,
                diagnostics.script_active,
                diagnostics.waiting_for_input,
                diagnostics.facing,
                diagnostics.truck_flag,
                diagnostics.truck_sequence_active,
                diagnostics.active_script,
                diagnostics.truck_task,
                diagnostics.truck_task_state,
                diagnostics.input_accepted,
                self._last_truck_decision,
                self._last_truck_exit_warp,
                self._last_truck_exit_destination,
                self._last_truck_navigation_target,
                diagnostics.clock_task,
                diagnostics.clock_hours,
                diagnostics.clock_minutes,
            )
            if current_key == previous_key:
                self._last_diagnostics = diagnostics
                return
        self._last_diagnostics = diagnostics
        console.print(
            "[dim]Opening startup: "
            f"phase={diagnostics.phase.name} observed={diagnostics.observed.name} "
            f"map={diagnostics.map_id} coords={diagnostics.coordinates} "
            f"game_state={diagnostics.game_state} controllable={diagnostics.controllable} "
            f"facing={diagnostics.facing} script_active={diagnostics.script_active} "
            f"waiting={diagnostics.waiting_for_input} truck_flag={diagnostics.truck_flag} "
            f"truck_sequence_active={diagnostics.truck_sequence_active} "
            f"active_script={diagnostics.active_script} "
            f"truck_task={diagnostics.truck_task} "
            f"truck_task_state={diagnostics.truck_task_state} "
            f"input_accepted={diagnostics.input_accepted} "
            f"clock_task={diagnostics.clock_task} "
            f"clock_time={None if diagnostics.clock_hours is None else f'{diagnostics.clock_hours:02d}:{diagnostics.clock_minutes:02d}'} "
            f"truck_navigation_target={self._last_truck_navigation_target} "
            f"decision={self._last_truck_decision}[/]"
        )

    def _report_dialogue(self, observed: OpeningSequenceState, event: str) -> None:
        """Print dialogue diagnostics only when the message/task state changes."""
        if not context.debug:
            return
        try:
            state = self._dialogue_state_snapshot()
            key = (self.phase, observed, event, state)
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            key = (self.phase, observed, event, None)
        if key == self._last_dialogue_key:
            return
        self._last_dialogue_key = key
        state = key[3]
        console.print(
            f"[dim]Opening startup dialogue: {event}; phase={self.phase.name} "
            f"task_active={None if state is None else state[0]} "
            f"task_state={None if state is None else state[1]} "
            f"script_active={None if state is None else state[2]} "
            f"native={None if state is None else state[3]} "
            f"script={None if state is None else state[4]} "
            f"native_ptr={None if state is None else state[10]} "
            f"script_pc={None if state is None else state[11]} "
            f"waiting={None if state is None else state[5]} "
            f"controllable={None if state is None else state[6]} "
            f"message_visible={None if state is None else state[7]} "
            f"printer_state={None if state is None else state[8]} "
            f"active_tasks={None if state is None else state[9]}[/]"
        )

    def _set_clock(self) -> Generator:
        """Drive Emerald's existing clock tasks toward the resolved target."""
        if self._clock_target is None:
            self._clock_target = _emerald_clock_time(_clock_time_mode(), rng=self._rng)
        target_hour, target_minute = self._clock_target
        clock_task = _active_clock_task()
        if clock_task is None:
            self._last_clock_task = None
            self._clock_a_sent = False
            self._last_truck_decision = "wait: clock task unavailable during transition"
            yield from wait_for_fade_to_finish()
            yield
            return

        if clock_task != self._last_clock_task:
            self._last_clock_task = clock_task
            self._clock_a_sent = False
            self._clock_confirm_yes_prepared = False

        if clock_task == _CLOCK_HANDLE_INPUT:
            task = get_task(clock_task)
            if task is None:
                self._last_truck_decision = "wait: clock selection task data unavailable"
            else:
                minute_hand_angle = task.data_value(0)
                hours = task.data_value(2)
                minutes = task.data_value(3)
                if minute_hand_angle % 6:
                    self._last_truck_decision = "wait: clock hand animation"
                elif (hours, minutes) == self._clock_target:
                    self._last_truck_decision = f"clock: confirm selected {target_hour:02d}:{target_minute:02d}"
                    if not self._clock_a_sent:
                        _report_opening_a_decision(
                            source="EmeraldOpeningMode._set_clock",
                            reason=f"clock hands reached the target time {target_hour:02d}:{target_minute:02d}",
                            phase=OpeningSequenceState.CLOCK_SETTING,
                        )
                        context.emulator.press_button("A")
                        self._clock_a_sent = True
                        self._last_truck_decision = f"clock input: A (confirm {target_hour:02d}:{target_minute:02d})"
                else:
                    button = _clock_input_direction(hours, minutes, target_hour, target_minute)
                    context.emulator.press_button(button)
                    self._clock_a_sent = False
                    self._last_truck_decision = (
                        f"clock input: {button} toward {target_hour:02d}:{target_minute:02d} "
                        f"(current={hours:02d}:{minutes:02d})"
                    )
        elif clock_task == _CLOCK_ASK_CONFIRM:
            self._last_truck_decision = "clock: waiting for confirmation menu"
        elif clock_task == _CLOCK_HANDLE_CONFIRM_INPUT:
            if not self._clock_confirm_yes_prepared:
                # Emerald creates this menu with initialCursorPos=1 (NO).
                # Move to YES before confirming; A on the initial cursor is
                # valid input but deliberately returns to HandleInput.
                context.emulator.press_button("Up")
                self._clock_confirm_yes_prepared = True
                self._last_truck_decision = "clock input: Up (select YES)"
            elif not self._clock_a_sent:
                _report_opening_a_decision(
                    source="EmeraldOpeningMode._set_clock",
                    reason="clock confirmation menu is prepared on YES",
                    phase=OpeningSequenceState.CLOCK_SETTING,
                )
                context.emulator.press_button("A")
                self._clock_a_sent = True
                self._last_truck_decision = "clock input: A (confirm YES)"
        else:
            self._last_truck_decision = f"clock: wait for task {clock_task}"
        yield
