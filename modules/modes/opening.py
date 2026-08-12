"""Emerald's fresh-game opening sequence."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Generator

from modules.console import console
from modules.context import context
from modules.keyboard import get_naming_screen_data, type_in_naming_screen
from modules.map import get_map_data_for_current_position
from modules.map_data import MapRSE
from modules.memory import GameState, get_event_flag, get_game_state
from modules.modes.util.tasks_scripts import wait_for_fade_to_finish
from modules.modes.util.walking import ensure_facing_direction, navigate_to
from modules.player import get_player_avatar, player_avatar_is_controllable, player_avatar_is_standing_still
from modules.tasks import get_global_script_context, get_task, is_waiting_for_input, task_is_active
from ._interface import BotMode, BotModeError

_starter_handoff_pending = False


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
    BIRCH_HOUSE_1F = auto()
    BIRCH_HOUSE_2F = auto()
    BIRCH_POKEBALL = auto()
    MAY_SEQUENCE = auto()
    ROUTE_101 = auto()
    STARTER_SELECTION = auto()
    SCRIPTED_INTRO = auto()
    UNKNOWN = auto()


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
    )


_CLOCK_TASKS = (
    "Task_SetClock_WaitFadeIn",
    "Task_SetClock_HandleInput",
    "Task_SetClock_AskConfirm",
    "Task_SetClock_HandleConfirmInput",
    "Task_SetClock_Confirmed",
    "Task_SetClock_Exit",
)
_EMERALD_ITEM_BALL_GRAPHICS_ID = 0x3C
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


def _clock_task_active() -> bool:
    try:
        return any(task_is_active(task) for task in _CLOCK_TASKS)
    except (AttributeError, RuntimeError, ValueError):
        # During emulator/map transitions the task table can be unavailable.
        return False


def get_opening_sequence_state() -> OpeningSequenceState:
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
    if map_id == MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F.value:
        return OpeningSequenceState.PLAYER_HOUSE_2F
    if map_id == MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F.value:
        return OpeningSequenceState.PLAYER_HOUSE_1F
    if map_id == MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F.value:
        return OpeningSequenceState.BIRCH_HOUSE_1F
    if map_id == MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F.value:
        return OpeningSequenceState.BIRCH_HOUSE_2F
    if map_id == MapRSE.ROUTE101.value:
        return OpeningSequenceState.ROUTE_101
    return OpeningSequenceState.SCRIPTED_INTRO


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


def _warp_to(destination: MapRSE) -> Generator:
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
    yield from navigate_to(location.map_group_and_number, warp.local_coordinates, avoid_scripted_events=False)


def _item_ball_coordinates() -> tuple[int, int] | None:
    """Find the item-ball object in the current rival-house map."""
    location = get_map_data_for_current_position()
    if location is None:
        return None
    # Emerald's object graphics table uses the item-ball graphic for the ball
    # on the floor.  Filtering templates instead of pressing A while walking
    # prevents accidentally talking to May or the television.
    ball = next(
        (obj for obj in location.objects if obj.graphics_id == _EMERALD_ITEM_BALL_GRAPHICS_ID),
        None,
    )
    return None if ball is None else ball.local_coordinates


def _advance_scripted_input() -> Generator:
    """Advance only when the game reports that an input is currently wanted."""
    waiting_for_input = False
    try:
        waiting_for_input = is_waiting_for_input()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        # Script/task tables can be unavailable for a transition frame.
        pass

    # Emerald's field dialogue is also held by Task_DrawFieldMessage while the
    # message box is displayed.  This is the same input strategy used by the
    # existing dialogue controllers and remains safe if the global script
    # context is temporarily None.
    if not waiting_for_input:
        try:
            waiting_for_input = context.rom.is_emerald and task_is_active("Task_DrawFieldMessage")
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass

    if waiting_for_input:
        context.emulator.press_button("A")
    yield


def _enter_player_name() -> Generator:
    """Wait for the naming keyboard to exist, including pointer/task gaps."""
    while True:
        if get_game_state() != GameState.NAMING_SCREEN:
            return
        try:
            if get_naming_screen_data() is None:
                yield
                continue
            yield from type_in_naming_screen("RED")
            return
        except (BotModeError, AttributeError, RuntimeError, ValueError, TypeError):
            # The naming callback can outlive the keyboard data by a frame.
            # This is a normal transition, not an automation failure.
            yield


class EmeraldOpeningMode(BotMode):
    @staticmethod
    def name() -> str:
        return "Start New Game"

    @staticmethod
    def is_selectable() -> bool:
        return context.rom.is_emerald

    def __init__(self):
        super().__init__()
        self.phase = OpeningSequenceState.TRUCK
        self._ball_coordinates: tuple[int, int] | None = None
        self._last_diagnostics: OpeningDiagnostics | None = None
        self._last_truck_decision: str | None = None
        self._last_truck_exit_warp: tuple[int, int] | None = None
        self._last_truck_exit_destination: tuple[int, int] | None = None
        self._last_truck_navigation_target: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._pending_house_warp_destination: MapRSE | None = None

    def run(self) -> Generator:
        if not context.rom.is_emerald:
            raise BotModeError("Start New Game only supports Pokémon Emerald.")

        while context.bot_mode != "Manual":
            observed = get_opening_sequence_state()
            diagnostics = get_opening_diagnostics(self.phase, observed)
            self._report_diagnostics(diagnostics)
            if observed is OpeningSequenceState.STARTER_SELECTION:
                global _starter_handoff_pending
                _starter_handoff_pending = True
                context.bot_mode = "Starters"
                return

            if observed == OpeningSequenceState.TITLE or observed == OpeningSequenceState.MAIN_MENU:
                context.emulator.press_button("A")
                yield
                continue
            if observed == OpeningSequenceState.PLAYER_NAMING:
                yield from _enter_player_name()
                continue
            if observed == OpeningSequenceState.CLOCK_SETTING:
                yield from self._set_clock()
                continue

            yield from self._advance_phase(observed, diagnostics)

    def _advance_phase(
        self,
        observed: OpeningSequenceState,
        diagnostics: OpeningDiagnostics | None = None,
    ) -> Generator:
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
                yield from _warp_to(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F)
                self.phase = OpeningSequenceState.PLAYER_HOUSE_1F
                return

        if self.phase is OpeningSequenceState.PLAYER_HOUSE_1F:
            if self._pending_house_warp_destination is not None:
                destination = self._pending_house_warp_destination
                observed_map = _current_map_id()
                if observed_map != destination.value:
                    self._last_truck_decision = (
                        f"wait: house warp to {destination.name}; observed map={observed_map}"
                    )
                    yield
                    return
                self._pending_house_warp_destination = None
                if destination is MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F:
                    self.phase = OpeningSequenceState.PLAYER_HOUSE_2F
                elif destination is MapRSE.LITTLEROOT_TOWN:
                    self.phase = OpeningSequenceState.POST_CLOCK_TOWN
                return

            if observed == OpeningSequenceState.PLAYER_HOUSE_1F and not get_event_flag("SET_WALL_CLOCK"):
                if not _player_house_1f_ready_for_navigation():
                    self._last_truck_decision = "wait: player's house 1F introductory dialogue"
                    yield from _advance_scripted_input()
                    return
                self._pending_house_warp_destination = MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F
                self._last_truck_decision = "navigate: ROM-defined staircase warp to player's house 2F"
                yield from _warp_to(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F)
                return
            if observed == OpeningSequenceState.PLAYER_HOUSE_2F:
                if get_event_flag("SET_WALL_CLOCK"):
                    # The staircase is the only supported route back down;
                    # walking to it lets the map/path system handle the warp.
                    self._pending_house_warp_destination = MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F
                    self._last_truck_decision = "navigate: ROM-defined staircase warp back to player's house 1F"
                    yield from _warp_to(MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_1F)
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
            if not self._can_navigate():
                yield
                return
            yield from navigate_to(
                MapRSE.LITTLEROOT_TOWN_BRENDANS_HOUSE_2F,
                (7, 2),
                avoid_scripted_events=False,
            )
            yield from ensure_facing_direction("Up")
            context.emulator.press_button("A")
            yield
            self.phase = OpeningSequenceState.CLOCK_SETTING
            return

        if self.phase is OpeningSequenceState.CLOCK_SETTING:
            yield from self._set_clock()
            if get_event_flag("SET_WALL_CLOCK"):
                self.phase = OpeningSequenceState.PLAYER_HOUSE_1F
            return

        if self.phase is OpeningSequenceState.POST_CLOCK_TOWN:
            if observed == OpeningSequenceState.LITTLEROOT_TOWN:
                yield from _warp_to(MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_1F)
                self.phase = OpeningSequenceState.BIRCH_HOUSE_1F
            return

        if self.phase is OpeningSequenceState.BIRCH_HOUSE_1F:
            if observed is not OpeningSequenceState.BIRCH_HOUSE_1F:
                yield from _advance_scripted_input()
                return
            yield from _warp_to(MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F)
            self.phase = OpeningSequenceState.BIRCH_HOUSE_2F
            return

        if self.phase is OpeningSequenceState.BIRCH_HOUSE_2F:
            if observed is not OpeningSequenceState.BIRCH_HOUSE_2F:
                yield from _advance_scripted_input()
                return
            self._ball_coordinates = _item_ball_coordinates()
            if self._ball_coordinates is None:
                yield from _advance_scripted_input()
                return
            if not self._can_navigate():
                yield
                return
            ball_x, ball_y = self._ball_coordinates
            yield from navigate_to(MapRSE.LITTLEROOT_TOWN_MAYS_HOUSE_2F, (ball_x, ball_y + 1))
            yield from ensure_facing_direction(self._ball_coordinates)
            context.emulator.press_button("A")
            yield
            self.phase = OpeningSequenceState.MAY_SEQUENCE
            return

        if self.phase is OpeningSequenceState.MAY_SEQUENCE:
            if observed in (OpeningSequenceState.LITTLEROOT_TOWN, OpeningSequenceState.ROUTE_101):
                self.phase = OpeningSequenceState.ROUTE_101
            else:
                yield from _advance_scripted_input()
            return

        if self.phase is OpeningSequenceState.ROUTE_101:
            if observed not in (OpeningSequenceState.LITTLEROOT_TOWN, OpeningSequenceState.ROUTE_101):
                yield from _advance_scripted_input()
                return
            if not self._can_navigate():
                yield
                return
            yield from navigate_to(
                MapRSE.ROUTE101,
                (10, 6),
                avoid_encounters=False,
                avoid_scripted_events=False,
                expecting_script=True,
            )
            yield
            return

        yield from _advance_scripted_input()

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
            f"truck_navigation_target={self._last_truck_navigation_target} "
            f"decision={self._last_truck_decision}[/]"
        )

    @staticmethod
    def _set_clock() -> Generator:
        if _clock_task_active():
            context.emulator.press_button("A")
        else:
            yield from wait_for_fade_to_finish()
        yield
