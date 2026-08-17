"""Emerald-specific campaign execution capabilities.

This module is the bridge between declarative campaign milestones and the
proven Emerald interaction machinery.  It is deliberately not a BotMode:
the campaign controller mounts its generator and re-observes facts every
frame.  The old selectable opening name remains only as a compatibility
alias for existing users/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import random
from typing import Iterator

from modules.context import context
from modules.agent_control import AgentControlLoop, observe_agent
from modules.goals import ReachWarp
from modules.navigation import GoalAwareNavigator, NavigationError, NavigationWorld
from modules.keyboard import type_in_naming_screen
from modules.memory import GameState, get_game_state
from modules.overworld import perceive_overworld
from modules.player import player_avatar_is_controllable
from modules.player import get_player
from modules.start_game import resolve_start_game_initialization
from modules.modes.opening import (
    OpeningSequenceState,
    _advance_scripted_input,
    _enter_player_name,
    _scripted_std_msgbox_waiting,
    EmeraldOpeningCapability,
    get_opening_sequence_state,
)
from modules.tasks import get_task, get_tasks, is_emerald_field_dialogue_advanceable, is_field_message_waiting_for_input
from .emerald_menu_observation import (
    EmeraldMainMenuItem,
    EmeraldMenuKind,
    EmeraldMenuObservation,
    EmeraldOptionsItem,
    observe_emerald_menu,
)
from .emerald_naming import EmeraldNamingObservation, EmeraldNamingTarget, observe_emerald_naming
from .emerald_confirmation import (
    EmeraldConfirmationChoice,
    EmeraldConfirmationContext,
    EmeraldConfirmationObservation,
    observe_emerald_confirmation,
)

_POST_PLAYER_NAME_TASKS = {
    "Task_NewGameBirchSpeech_SlidePlatformAway2",
    "Task_NewGameBirchSpeech_ReshowBirchLotad",
    "Task_NewGameBirchSpeech_WaitForSpriteFadeInAndTextPrinter",
    "Task_NewGameBirchSpeech_AreYouReady",
}


class EmeraldCampaignAction(Enum):
    WAIT = auto()
    ADVANCE_TITLE = auto()
    ENTER_OPTIONS = auto()
    ADVANCE_TEXT_SPEED = auto()
    EXIT_OPTIONS = auto()
    START_NEW_GAME = auto()
    ADVANCE_DIALOGUE = auto()
    ENTER_NAME = auto()
    MOVE_CONFIRMATION_TO_YES = auto()
    CONFIRM_PLAYER_NAME = auto()
    CHOOSE_GENDER = auto()
    ADVANCE_NAME_PROMPT = auto()
    ADVANCE_OBSERVED_OVERWORLD = auto()


@dataclass(frozen=True, slots=True)
class EmeraldCampaignObservation:
    """Small, current-frame view used by the deterministic campaign policy."""

    state: OpeningSequenceState
    text_speed_fast: bool
    new_game_setup_complete: bool
    dialogue_waiting: bool = False
    gender_task: str | None = None
    main_menu_cursor: int | None = None
    dialogue_lifecycle_active: bool = False
    menu_observation: EmeraldMenuObservation | None = None
    naming_observation: EmeraldNamingObservation | None = None
    confirmation_observation: EmeraldConfirmationObservation | None = None


def choose_emerald_campaign_action(observation: EmeraldCampaignObservation) -> EmeraldCampaignAction:
    """Choose one conservative action from the current observation only."""
    # Overworld avatar controllability is not a universal actionability test:
    # menus, naming, confirmations, and dialogue can all legitimately accept
    # input while the overworld avatar is not controllable.  Scripted waits
    # fall through to WAIT below because they expose no actionable observation.
    if observation.state is OpeningSequenceState.TITLE:
        return EmeraldCampaignAction.ADVANCE_TITLE
    if observation.state is OpeningSequenceState.OPTIONS_MENU:
        return (
            EmeraldCampaignAction.EXIT_OPTIONS
            if observation.text_speed_fast
            else EmeraldCampaignAction.ADVANCE_TEXT_SPEED
        )
    if observation.gender_task == "Task_NewGameBirchSpeech_ChooseGender":
        return EmeraldCampaignAction.CHOOSE_GENDER
    if observation.gender_task == "Task_NewGameBirchSpeech_WaitPressBeforeNameChoice":
        return EmeraldCampaignAction.ADVANCE_NAME_PROMPT
    confirmation = observation.confirmation_observation
    if confirmation is not None and confirmation.context is EmeraldConfirmationContext.PLAYER_NAME:
        if not confirmation.input_ready or confirmation.selected is None:
            return EmeraldCampaignAction.WAIT
        return (
            EmeraldCampaignAction.CONFIRM_PLAYER_NAME
            if confirmation.selected is EmeraldConfirmationChoice.YES
            else EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES
        )
    if observation.state is OpeningSequenceState.PLAYER_NAMING:
        if (
            observation.naming_observation is not None
            and observation.naming_observation.target is EmeraldNamingTarget.PLAYER_NAME
            and observation.naming_observation.keyboard_ready
        ):
            return EmeraldCampaignAction.ENTER_NAME
        return EmeraldCampaignAction.WAIT
    try:
        if get_game_state() is GameState.OVERWORLD and player_avatar_is_controllable():
            return EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    if observation.dialogue_waiting:
        return EmeraldCampaignAction.ADVANCE_DIALOGUE
    if observation.state is OpeningSequenceState.MAIN_MENU:
        if not observation.text_speed_fast:
            return EmeraldCampaignAction.ENTER_OPTIONS
        if not observation.new_game_setup_complete:
            return EmeraldCampaignAction.START_NEW_GAME
    return EmeraldCampaignAction.WAIT


def _observed_exit_goal(world: OverworldObservation, navigator: GoalAwareNavigator) -> ReachWarp | None:
    """Choose a tactical exit from the currently observed map.

    This is deliberately map/warp driven.  It does not identify the truck,
    house, stairs, or any other opening phase; the next observation decides
    again after the map changes.
    """
    if world is None or not world.controllable:
        return None
    exits = tuple(warp for warp in world.warps if warp.destination[0] != world.map_id)
    if not exits:
        return None

    # Sort by Manhattan distance to minimize the number of expensive pathfinding
    # calls.  A warp already under the player or very close is likely the
    # intended one.
    def distance(warp):
        return abs(warp.entry[1][0] - world.player_coordinates[0]) + abs(
            warp.entry[1][1] - world.player_coordinates[1]
        )

    candidates = sorted(exits, key=distance)
    ranked: list[tuple[tuple, object]] = []
    # Plan for at most the three closest candidates to avoid unthrottled spikes.
    for warp in candidates[:3]:
        target = ReachWarp(destination_map=warp.destination[0], destination=warp.destination, warp=warp)
        try:
            # Use A* for faster heuristic-guided search.
            plan = navigator.plan((world.map_id, world.player_coordinates), target, algorithm="astar")
        except NavigationError:
            continue
        metrics = plan.metrics
        ranked.append(
            (
                (
                    metrics.encounter_opportunities if metrics else float("inf"),
                    metrics.total_route_cost if metrics else float("inf"),
                    metrics.movement_actions if metrics else float("inf"),
                    warp.entry[1],
                ),
                warp,
            )
        )
    if not ranked:
        # Fall back to Manhattan-closest if no path was found (e.g. dynamic blocking).
        selected = candidates[0]
    else:
        selected = min(ranked, key=lambda item: item[0])[1]
    return ReachWarp(destination_map=selected.destination[0], destination=selected.destination, warp=selected)


def observation_driven_overworld_progression() -> Iterator[object]:
    """Advance through observed overworld affordances one input at a time.

    A non-controllable scripted transition naturally produces no goal and no
    input.  Once control is returned, the current map's observed exit becomes
    a tactical ``ReachWarp`` goal.  The loop is recreated after a map change,
    so it never carries an opening route across an asynchronous transition.
    """
    loop: Iterator[object] | None = None
    source_map: tuple[int, int] | None = None
    navigator: GoalAwareNavigator | None = None
    while True:
        try:
            current = perceive_overworld()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            current = None

        if loop is None:
            if current is None or not current.controllable:
                yield
                continue
            if navigator is None or navigator.world.facing is not None:
                # GoalAwareNavigator and its World are cheap to create but
                # we must provide current tiles.
                navigator = GoalAwareNavigator(NavigationWorld.from_overworld(current))
            goal = _observed_exit_goal(current, navigator)
            if goal is None:
                yield
                continue
            loop = AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()
            source_map = current.map_id
        try:
            next(loop)
        except StopIteration:
            loop = None
            source_map = None
            yield
            continue
        if current is None or current.map_id != source_map:
            loop = None
            source_map = None
        yield


def _start_game_values() -> tuple[str, str]:
    config = getattr(context, "config", None)
    start_game = getattr(config, "start_game", None)
    return (
        getattr(start_game, "player_name", "gibberish"),
        getattr(start_game, "player_gender", "random"),
    )


def _configured_name(target: EmeraldNamingTarget) -> str | None:
    """Select a policy only for its authoritative naming target."""
    config = getattr(context, "config", None)
    if target is EmeraldNamingTarget.PLAYER_NAME:
        return getattr(getattr(config, "start_game", None), "player_name", None)
    if target in (EmeraldNamingTarget.POKEMON_NICKNAME, EmeraldNamingTarget.CAUGHT_POKEMON_NICKNAME):
        return getattr(getattr(config, "naming", None), "pokemon_nickname", None)
    return None


def _resolve_player_campaign_initialization(rng: RandomSource | None = None) -> StartGameInitialization:
    name_config, gender_config = _start_game_values()
    if rng is None:
        # Use a session-stable RNG so that re-mounting the capability for
        # different objectives (e.g. set_wall_clock, meet_rival) produces
        # the same choice.
        session_id = getattr(context.nuzlocke_runtime, "session_id", None)
        rng = random.Random(session_id)
    return resolve_start_game_initialization(name_config, gender_config, rng)


def _resolve_player_campaign_name(rng: RandomSource | None = None) -> str | None:
    configured, configured_gender = _start_game_values()
    if not isinstance(configured, str):
        return None
    try:
        # SaveBlock2 is populated by Birch before the naming UI opens, so its
        # gender is the authoritative result of configured/random selection.
        gender = get_player().gender
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        gender = configured_gender if configured_gender in ("male", "female") else None
    if gender is None:
        return None
    try:
        if rng is None:
            session_id = getattr(context.nuzlocke_runtime, "session_id", None)
            rng = random.Random(session_id)
        return resolve_start_game_initialization(configured, gender, rng).name
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return None


def _naming_input(name: str) -> None:
    """Use the existing keyboard helper for exactly one input transaction."""
    keyboard = type_in_naming_screen(name)
    try:
        next(keyboard)
    except StopIteration:
        pass


def _active_gender_task() -> str | None:
    try:
        names = {task.symbol for task in (get_tasks() or [])}
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return None
    for name in (
        "Task_NewGameBirchSpeech_ChooseGender",
        "Task_NewGameBirchSpeech_WaitToShowGenderMenu",
        "Task_NewGameBirchSpeech_SlideOutOldGenderSprite",
        "Task_NewGameBirchSpeech_SlideInNewGenderSprite",
        "Task_NewGameBirchSpeech_WhatsYourName",
        "Task_NewGameBirchSpeech_WaitForWhatsYourNameToPrint",
        "Task_NewGameBirchSpeech_WaitPressBeforeNameChoice",
        "Task_NewGameBirchSpeech_StartNamingScreen",
        *_POST_PLAYER_NAME_TASKS,
    ):
        if name in names:
            return name
    return None


def _main_menu_cursor() -> int | None:
    """Compatibility accessor for callers that still need the old scalar."""
    observation = observe_emerald_menu()
    return observation.cursor_index if observation is not None else None


def _gender_input(target_gender: str) -> Iterator[object]:
    """Choose the target gender from the Observed gender-selection task."""
    task = get_task("Task_NewGameBirchSpeech_ChooseGender")
    if task is None:
        yield
        return
    cursor = task.data_value(6)
    target = 1 if target_gender == "female" else 0
    if cursor == target:
        context.emulator.press_button("A")
    else:
        context.emulator.press_button("Down" if target > cursor else "Up")
    yield


def _advance_name_prompt() -> Iterator[object]:
    """Cross Emerald's observed post-gender handoff into naming."""
    if get_task("Task_NewGameBirchSpeech_WaitPressBeforeNameChoice") is None:
        yield
        return
    context.emulator.press_button("A")
    yield


def _campaign_observation(field_message_lifecycle_active: bool = False) -> EmeraldCampaignObservation:
    from .campaign_controller import runtime_campaign_state

    state = runtime_campaign_state()
    _, configured_gender = _start_game_values()
    gender_task = _active_gender_task()
    try:
        dialogue_state = EmeraldOpeningCapability._dialogue_state_snapshot()
        birch_task = next(
            (
                name
                for name in ((dialogue_state[9] if len(dialogue_state) > 9 else ()) or ())
                if name.startswith("Task_NewGameBirchSpeech")
            ),
            None,
        )
        if dialogue_state[0]:
            field_message_lifecycle_active = True
        dialogue_waiting = (
            get_game_state() == GameState.OVERWORLD or birch_task is not None
        ) and is_emerald_field_dialogue_advanceable(
            task_active=dialogue_state[0],
            task_name=birch_task,
            script_active=dialogue_state[2],
            native_function_name=dialogue_state[3],
            script_function_name=dialogue_state[4],
            input_waiting=dialogue_state[5],
            field_message_lifecycle_active=field_message_lifecycle_active,
        )
        if dialogue_waiting:
            field_message_lifecycle_active = True
        # Do not let a completed message make every later native wait look
        # like dialogue.  The opening predicate uses the same lifecycle edge.
        if (
            field_message_lifecycle_active
            and not dialogue_state[0]
            and dialogue_state[3] != "WaitForAorBPress"
            and not dialogue_state[7]
        ):
            field_message_lifecycle_active = False
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        dialogue_waiting = False
    try:
        menu_observation = observe_emerald_menu()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        menu_observation = None
    try:
        naming_observation = observe_emerald_naming()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        naming_observation = None
    try:
        confirmation_observation = observe_emerald_confirmation()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        confirmation_observation = None
    observed_text_speed_fast = (
        menu_observation is not None
        and menu_observation.menu_kind is EmeraldMenuKind.OPTIONS_MENU
        and menu_observation.raw_option_values is not None
        and len(menu_observation.raw_option_values) > 0
        and menu_observation.raw_option_values[0] == 2
    )
    return EmeraldCampaignObservation(
        get_opening_sequence_state(configured_gender),
        observed_text_speed_fast or state.campaign_facts.text_speed_fast.value is True,
        state.campaign_facts.new_game_setup_complete.value is True,
        dialogue_waiting,
        gender_task,
        menu_observation.cursor_index if menu_observation is not None else None,
        field_message_lifecycle_active,
        menu_observation,
        naming_observation,
        confirmation_observation,
    )


def _main_menu_target_index(menu: EmeraldMenuObservation, target: EmeraldMainMenuItem) -> int | None:
    """Return the target's index for the observed save-dependent menu."""
    if menu.raw_menu_type is None:
        return None
    if target is EmeraldMainMenuItem.OPTIONS:
        return menu.raw_menu_type + 1
    if target is EmeraldMainMenuItem.NEW_GAME:
        return 0 if menu.raw_menu_type == 0 else 1
    return None


def _menu_button(observation: EmeraldCampaignObservation, action: EmeraldCampaignAction) -> str | None:
    """Choose one input from the current normalized menu observation."""
    menu = observation.menu_observation
    if menu is None or not menu.input_ready or menu.cursor_index is None:
        return None
    if menu.menu_kind is EmeraldMenuKind.MAIN_MENU:
        if not isinstance(menu.selected_item, EmeraldMainMenuItem):
            return None
        target = EmeraldMainMenuItem.OPTIONS if not observation.text_speed_fast else EmeraldMainMenuItem.NEW_GAME
        target_index = _main_menu_target_index(menu, target)
        if target_index is None or menu.cursor_index == target_index:
            return "A" if menu.cursor_index == target_index else None
        return "Down" if menu.cursor_index < target_index else "Up"
    if menu.menu_kind is EmeraldMenuKind.OPTIONS_MENU:
        if not isinstance(menu.selected_item, EmeraldOptionsItem):
            return None
        target_index = tuple(EmeraldOptionsItem).index(EmeraldOptionsItem.TEXT_SPEED)
        if action is EmeraldCampaignAction.ADVANCE_TEXT_SPEED:
            if menu.cursor_index != target_index:
                return "Down" if menu.cursor_index < target_index else "Up"
            return "Right"
        if action is EmeraldCampaignAction.EXIT_OPTIONS:
            return "B"
    return None


def observation_driven_emerald_campaign() -> Iterator[object]:
    """Execute one input transaction, then yield for a fresh observation.

    ``pending`` is only a wait guard for an asynchronous transition.  It is
    deliberately not a phase machine: if the observed state changes, the
    policy is run again immediately; if it does not, one quiet observation is
    allowed before the action is reconsidered for recovery.
    """
    field_message_lifecycle_active = False
    pending: EmeraldCampaignAction | None = None
    pending_signature: tuple | None = None
    pending_waited = False
    overworld_progression = observation_driven_overworld_progression()
    resolved_player_initialization: StartGameInitialization | None = None
    session_id = getattr(getattr(context, "nuzlocke_runtime", None), "session_id", None)
    rng = random.Random(session_id)
    while True:
        observation = _campaign_observation(field_message_lifecycle_active)
        field_message_lifecycle_active = observation.dialogue_lifecycle_active
        action = choose_emerald_campaign_action(observation)

        signature = (
            observation.state,
            observation.text_speed_fast,
            observation.new_game_setup_complete,
            observation.dialogue_waiting,
            observation.gender_task,
            observation.main_menu_cursor,
            observation.menu_observation,
            observation.naming_observation,
            observation.confirmation_observation,
        )
        if pending is not None:
            satisfied = (
                (
                    pending is EmeraldCampaignAction.ENTER_OPTIONS
                    and observation.state is OpeningSequenceState.OPTIONS_MENU
                )
                or (
                    pending is EmeraldCampaignAction.EXIT_OPTIONS
                    and observation.state is OpeningSequenceState.MAIN_MENU
                )
                or (
                    pending is EmeraldCampaignAction.ADVANCE_TITLE
                    and observation.state is not OpeningSequenceState.TITLE
                )
                or (pending is EmeraldCampaignAction.ADVANCE_TEXT_SPEED and observation.text_speed_fast)
                or (
                    pending is EmeraldCampaignAction.START_NEW_GAME
                    and observation.state is not OpeningSequenceState.MAIN_MENU
                )
                or (
                    pending is EmeraldCampaignAction.ENTER_NAME
                    and (
                        observation.naming_observation is None
                        or observation.naming_observation.target is not EmeraldNamingTarget.PLAYER_NAME
                    )
                )
                or (
                    pending is EmeraldCampaignAction.CONFIRM_PLAYER_NAME
                    and observation.confirmation_observation is None
                    and observation.naming_observation is None
                    and (
                        observation.state is OpeningSequenceState.SCRIPTED_INTRO
                        or (
                            observation.state is OpeningSequenceState.MAIN_MENU
                            and observation.gender_task in _POST_PLAYER_NAME_TASKS
                        )
                    )
                )
                or (pending is EmeraldCampaignAction.ADVANCE_DIALOGUE and not observation.dialogue_waiting)
                or (
                    pending is EmeraldCampaignAction.ADVANCE_NAME_PROMPT
                    and (
                        observation.state is OpeningSequenceState.PLAYER_NAMING
                        or observation.gender_task != pending_signature[4]
                    )
                )
            )
            if satisfied or signature != pending_signature:
                pending = None
                pending_signature = None
                pending_waited = False
            elif not pending_waited:
                pending_waited = True
                yield
                continue
            else:
                pending = None
                pending_signature = None
                pending_waited = False

        issued = False
        if action is EmeraldCampaignAction.ADVANCE_TITLE:
            context.emulator.press_button("A")
            issued = True
        elif action is EmeraldCampaignAction.ENTER_OPTIONS:
            button = _menu_button(observation, action)
            if button is not None:
                context.emulator.press_button(button)
                issued = True
        elif action is EmeraldCampaignAction.ADVANCE_TEXT_SPEED:
            button = _menu_button(observation, action)
            if button is not None:
                context.emulator.press_button(button)
                issued = True
        elif action is EmeraldCampaignAction.EXIT_OPTIONS:
            button = _menu_button(observation, action)
            if button is not None:
                context.emulator.press_button(button)
                issued = True
        elif action is EmeraldCampaignAction.START_NEW_GAME:
            button = _menu_button(observation, action)
            if button is not None:
                context.emulator.press_button(button)
                issued = True
        elif action is EmeraldCampaignAction.ADVANCE_DIALOGUE:
            yield from _advance_scripted_input()
            issued = True
        elif action is EmeraldCampaignAction.ENTER_NAME:
            if (
                observation.naming_observation is None
                or observation.naming_observation.target is not EmeraldNamingTarget.PLAYER_NAME
            ):
                yield
                continue
            if resolved_player_initialization is None:
                resolved_player_initialization = _resolve_player_campaign_initialization(rng)
            _naming_input(resolved_player_initialization.name)
            issued = True
        elif action is EmeraldCampaignAction.CHOOSE_GENDER:
            if resolved_player_initialization is None:
                resolved_player_initialization = _resolve_player_campaign_initialization(rng)
            yield from _gender_input(resolved_player_initialization.gender.value)
            issued = True
        elif action is EmeraldCampaignAction.ADVANCE_NAME_PROMPT:
            yield from _advance_name_prompt()
            issued = True
        elif action is EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES:
            confirmation = observation.confirmation_observation
            if (
                confirmation is None
                or not confirmation.input_ready
                or confirmation.selected is not EmeraldConfirmationChoice.NO
            ):
                yield
                continue
            context.emulator.press_button("Up")
            issued = True
        elif action is EmeraldCampaignAction.CONFIRM_PLAYER_NAME:
            confirmation = observation.confirmation_observation
            if (
                confirmation is None
                or not confirmation.input_ready
                or confirmation.selected is not EmeraldConfirmationChoice.YES
            ):
                yield
                continue
            context.emulator.press_button("A")
            issued = True
        elif action is EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD:
            # Tactical navigation owns the single-input/re-observation
            # transaction.  Campaign facts remain outside this loop and are
            # re-evaluated by CampaignController on the next frame.
            next(overworld_progression)
            yield
            continue
        else:
            yield
            continue

        if issued:
            pending = action
            pending_signature = signature
            pending_waited = False
        yield


def emerald_campaign_capability(objective_id: str) -> Iterator[object]:
    """Mount the observation-driven capability for an Emerald objective.

    Early objectives intentionally share a re-observing loop: a new generator
    can begin at any currently observable title/menu/dialogue state.
    """

    # Campaign objectives are executed by a current-observation loop. The
    # legacy opening state machine remains available for its direct callers,
    # but is no longer mounted by CampaignProgression.
    if objective_id in {"set_text_speed", "complete_new_game_setup"}:
        yield from observation_driven_emerald_campaign()
        return

    # Later objectives are migrated incrementally; retain the existing lower-
    # level opening machinery until their observation-driven capabilities are
    # extracted.
    from modules.modes.opening import EmeraldOpeningCapability

    session_id = getattr(getattr(context, "nuzlocke_runtime", None), "session_id", None)
    rng = random.Random(session_id)
    capability = EmeraldOpeningCapability(rng=rng, campaign_owned=True)
    yield from capability.run()
    if objective_id == "obtain_starter":
        # Reuse the proven Hoenn starter interaction directly as a capability
        # step. It does not receive control through context.bot_mode.
        from modules.modes.starters import run_rse_hoenn

        yield from run_rse_hoenn(lambda: None, reset_before_selection=False)
        # The opening state machine owns the post-selection dialogue and
        # reconciliation back to the campaign facts.
        yield from capability.run()


__all__ = [
    "EmeraldCampaignAction",
    "EmeraldCampaignObservation",
    "choose_emerald_campaign_action",
    "emerald_campaign_capability",
]
