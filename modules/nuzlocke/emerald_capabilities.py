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
from typing import Callable, Iterator

from modules.context import context
from modules.console import diagnostic_print
from modules.agent_control import AgentControlLoop, ActionResultType, observe_agent
from modules.goals import ActivateTrigger, ReachLocation, ReachWarp, SemanticTarget, SemanticTargetKind
from modules.navigation import (
    GoalAwareNavigator,
    NavigationError,
    NavigationWorld,
    TransitionRelevance,
    classify_transition_relevance,
    transition_approach_position,
    transition_world_route,
)
from modules.map_path import Direction
from modules.keyboard import type_in_naming_screen
from modules.map import get_map_metadata
from modules.map_data import MapRSE
from modules.memory import GameState, get_event_flag, get_event_var, get_game_state, get_save_block, unpack_uint16
from modules.overworld import perceive_overworld, WorldTransition
from modules.player import player_avatar_is_controllable, player_avatar_is_rom_owned_movement
from modules.player import get_player
from modules.start_game import resolve_start_game_initialization
from .emerald_clock import (
    _current_map_id,
    active_clock_task as _active_clock_task,
    clock_input_direction as _clock_input_direction,
    clock_time_mode as _clock_time_mode,
    emerald_clock_time as _emerald_clock_time,
    player_house_map as _player_house_map,
    rival_house_map as _rival_house_map,
    wall_clock_interaction as _wall_clock_interaction,
)
from .emerald_opening_state import OpeningSequenceState
from modules.tasks import (
    get_global_script_context,
    get_task,
    get_tasks,
    is_emerald_field_dialogue_advanceable,
    is_field_message_waiting_for_input,
)
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
from .emerald_dialogue import advance_dialogue, dialogue_state_snapshot, observe_dialogue


# Compatibility hooks for existing campaign tests and diagnostics.  These
# names now resolve to the extracted, phase-free dialogue implementation; they
# are not imports from the legacy opening mode.
_advance_scripted_input = advance_dialogue


class EmeraldOpeningCapability:
    _dialogue_state_snapshot = staticmethod(dialogue_state_snapshot)


def _uses_shared_runtime_context() -> bool:
    """Avoid leaking live sibling-module perception into synthetic frames."""
    from modules.context import context as shared_context

    return context is shared_context


def _compatibility_opening_state(player_gender: object | None = None) -> OpeningSequenceState:
    """Compatibility classifier for the legacy campaign observation adapter.

    The observation-driven campaign executor does not call this function. It
    remains here only for callers and tests that still consume the old
    ``EmeraldCampaignObservation`` shape while the legacy opening mode is
    being retired from the campaign path.
    """
    try:
        game_state = get_game_state()
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return OpeningSequenceState.UNKNOWN
    state_map = {
        GameState.TITLE_SCREEN: OpeningSequenceState.TITLE,
        GameState.MAIN_MENU: OpeningSequenceState.MAIN_MENU,
        GameState.OPTIONS_MENU: OpeningSequenceState.OPTIONS_MENU,
        GameState.NAMING_SCREEN: OpeningSequenceState.PLAYER_NAMING,
        GameState.CHOOSE_STARTER: OpeningSequenceState.STARTER_SELECTION,
    }
    if game_state in state_map:
        return state_map[game_state]
    if _active_clock_task() is not None:
        return OpeningSequenceState.CLOCK_SETTING
    map_id = _current_map_id()
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
    return OpeningSequenceState.SCRIPTED_INTRO


# Kept as a patchable compatibility name for older callers. The campaign
# executor does not call this phase classifier.
get_opening_sequence_state = _compatibility_opening_state
from .emerald_observation import EmeraldObservation
from .emerald_starter_selection import (
    EmeraldStarterSelectionPhase,
    configured_emerald_starter,
    observe_emerald_starter_selection,
    resolve_emerald_starter,
)
from .pokemon_naming import POKEMON_NICKNAME_MAX_LENGTH, generate_pokemon_nickname
from .campaign_status import CampaignStatus
from modules.semantic_choices import choose_choice_for_outcome

_POST_PLAYER_NAME_TASKS = {
    "Task_NewGameBirchSpeech_SlidePlatformAway2",
    "Task_NewGameBirchSpeech_ReshowBirchLotad",
    "Task_NewGameBirchSpeech_WaitForSpriteFadeInAndTextPrinter",
    "Task_NewGameBirchSpeech_AreYouReady",
}
_BIRCH_BAG_INTERACTION_ID = "Route101_EventScript_BirchsBag"
_GO_SEE_RIVAL_SCRIPT = "LittlerootTown_ProfessorBirchsLab_EventScript_GoSeeRival"


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
    INTERACT_CLOCK = auto()
    ADVANCE_CLOCK = auto()
    CONFIRM_CLOCK = auto()
    NAVIGATE_TO_CLOCK = auto()
    FACE_CLOCK = auto()
    ADVANCE_OBSERVED_OVERWORLD = auto()
    MOVE_STARTER_LEFT = auto()
    MOVE_STARTER_RIGHT = auto()
    CHOOSE_STARTER = auto()
    CONFIRM_STARTER = auto()
    CHOOSE_POKEMON_NICKNAME = auto()
    ENTER_POKEMON_NICKNAME = auto()
    CONFIRM_GO_SEE_RIVAL = auto()


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
    rom_owned_movement: bool = False


def choose_emerald_campaign_action(observation: EmeraldCampaignObservation) -> EmeraldCampaignAction:
    """Choose one conservative action from the current observation only."""
    if observation.dialogue_waiting:
        return EmeraldCampaignAction.ADVANCE_DIALOGUE
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
    if observation.state is OpeningSequenceState.MAIN_MENU:
        if not observation.text_speed_fast:
            return EmeraldCampaignAction.ENTER_OPTIONS
        if not observation.new_game_setup_complete:
            return EmeraldCampaignAction.START_NEW_GAME
    if observation.rom_owned_movement:
        return EmeraldCampaignAction.WAIT
    try:
        if get_game_state() is GameState.OVERWORLD and player_avatar_is_controllable():
            return EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return EmeraldCampaignAction.WAIT


def choose_emerald_observation_action(observation: EmeraldObservation) -> EmeraldCampaignAction:
    """Pure global priority dispatcher for one current ROM observation.

    This function intentionally has no campaign phase, previous action, or
    generator input.  Every call is a new ownership decision.
    """
    starter = observation.starter_selection
    if starter is not None:
        if not starter.input_ready:
            return EmeraldCampaignAction.WAIT
        if starter.phase is EmeraldStarterSelectionPhase.CONFIRMING:
            return EmeraldCampaignAction.CONFIRM_STARTER
        if starter.phase is EmeraldStarterSelectionPhase.CHOOSING:
            target = configured_emerald_starter()
            if target == "Random":
                # The executor resolves Random once per mounted capability.
                return EmeraldCampaignAction.WAIT
            target_index = starter.choices.index(target)
            if starter.selected_index < target_index:
                return EmeraldCampaignAction.MOVE_STARTER_RIGHT
            if starter.selected_index > target_index:
                return EmeraldCampaignAction.MOVE_STARTER_LEFT
            return EmeraldCampaignAction.CHOOSE_STARTER
    confirmation = observation.confirmation
    if confirmation is not None and confirmation.context is EmeraldConfirmationContext.POKEMON_NICKNAME:
        if not confirmation.input_ready or confirmation.selected is None:
            return EmeraldCampaignAction.WAIT
        return (
            EmeraldCampaignAction.CHOOSE_POKEMON_NICKNAME
            if confirmation.selected is EmeraldConfirmationChoice.YES
            else EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES
        )
    if confirmation is not None and confirmation.context is EmeraldConfirmationContext.GO_SEE_RIVAL:
        if not confirmation.input_ready or confirmation.selected is None:
            return EmeraldCampaignAction.WAIT
        return (
            EmeraldCampaignAction.CONFIRM_GO_SEE_RIVAL
            if confirmation.selected is EmeraldConfirmationChoice.YES
            else EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES
        )
    if observation.naming is not None and observation.naming.target in {
        EmeraldNamingTarget.POKEMON_NICKNAME,
        EmeraldNamingTarget.CAUGHT_POKEMON_NICKNAME,
    }:
        return (
            EmeraldCampaignAction.ENTER_POKEMON_NICKNAME
            if observation.naming.keyboard_ready
            else EmeraldCampaignAction.WAIT
        )
    # The field-message lifecycle can remain actionable for a frame while
    # GiveStarterEvent has already transferred ownership to DoNamingScreen.
    # Never consume that stale dialogue edge on the naming UI.
    if observation.game_state is GameState.NAMING_SCREEN:
        if observation.naming is not None and observation.naming.target is EmeraldNamingTarget.PLAYER_NAME:
            return EmeraldCampaignAction.ENTER_NAME if observation.naming.keyboard_ready else EmeraldCampaignAction.WAIT
        return EmeraldCampaignAction.WAIT
    if observation.actionable_dialogue:
        return EmeraldCampaignAction.ADVANCE_DIALOGUE
    if _GO_SEE_RIVAL_SCRIPT in observation.script_stack:
        # The ROM script retains ownership between its confirmation, messages,
        # and movement waits. Only actionable dialogue above may advance it.
        return EmeraldCampaignAction.WAIT
    if observation.game_state is GameState.TITLE_SCREEN and observation.title_actionable:
        return EmeraldCampaignAction.ADVANCE_TITLE
    if observation.gender_task == "Task_NewGameBirchSpeech_ChooseGender":
        return EmeraldCampaignAction.CHOOSE_GENDER
    if observation.gender_task == "Task_NewGameBirchSpeech_WaitPressBeforeNameChoice":
        return EmeraldCampaignAction.ADVANCE_NAME_PROMPT
    if observation.naming is not None and observation.naming.target is EmeraldNamingTarget.PLAYER_NAME:
        return EmeraldCampaignAction.ENTER_NAME if observation.naming.keyboard_ready else EmeraldCampaignAction.WAIT
    confirmation = observation.confirmation
    if confirmation is not None and confirmation.active:
        if not confirmation.input_ready or confirmation.selected is None:
            return EmeraldCampaignAction.WAIT
        if confirmation.consequences:
            desired = choose_choice_for_outcome(
                confirmation.dialogue_choice,
                current_facts=dict(observation.campaign_facts),
                desired_facts={"wall_clock_set": True},
            )
            if desired is None:
                return EmeraldCampaignAction.WAIT
            if confirmation.selected.name == desired:
                return EmeraldCampaignAction.CONFIRM_CLOCK
            # The current Emerald two-row executor can move from NO to YES;
            # it remains a tactical operation, while the choice above came
            # from the predicted campaign consequence.
            return EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES
        if confirmation.context is EmeraldConfirmationContext.PLAYER_NAME:
            return (
                EmeraldCampaignAction.CONFIRM_PLAYER_NAME
                if confirmation.selected is EmeraldConfirmationChoice.YES
                else EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES
            )
        return EmeraldCampaignAction.WAIT
    menu = observation.menu
    if menu is not None:
        if not menu.input_ready:
            return EmeraldCampaignAction.WAIT
        if menu.menu_kind is EmeraldMenuKind.OPTIONS_MENU:
            return (
                EmeraldCampaignAction.EXIT_OPTIONS
                if observation.game_state is GameState.OPTIONS_MENU
                and observation.campaign_facts
                and dict(observation.campaign_facts).get("text_speed_fast") is True
                else EmeraldCampaignAction.ADVANCE_TEXT_SPEED
            )
        if menu.menu_kind is EmeraldMenuKind.MAIN_MENU:
            setup_complete = dict(observation.campaign_facts).get("new_game_setup_complete") is True
            text_speed_fast = dict(observation.campaign_facts).get("text_speed_fast") is True
            if setup_complete:
                return EmeraldCampaignAction.WAIT
            target = EmeraldMainMenuItem.NEW_GAME if text_speed_fast else EmeraldMainMenuItem.OPTIONS
            if target is EmeraldMainMenuItem.OPTIONS:
                return EmeraldCampaignAction.ENTER_OPTIONS
            return EmeraldCampaignAction.START_NEW_GAME
    facts = dict(observation.campaign_facts)
    wall_clock_fact = facts.get("wall_clock_set")

    # SET_WALL_CLOCK is authoritative even if a stale/native clock task is
    # still visible for one frame.  A completed save must never be driven back
    # into the wall-clock interaction by an objective label or task residue.
    if observation.clock_interaction is not None:
        if wall_clock_fact is None:
            return EmeraldCampaignAction.WAIT
        if wall_clock_fact is False:
            task_name = observation.clock_interaction[0]
            if task_name == "Task_SetClock_HandleConfirmInput":
                return EmeraldCampaignAction.CONFIRM_CLOCK
            if task_name == "Task_SetClock_HandleInput":
                return EmeraldCampaignAction.ADVANCE_CLOCK
            return EmeraldCampaignAction.WAIT
    if observation.rom_owned_movement:
        return EmeraldCampaignAction.WAIT
    if (
        observation.semantic_target is not None
        and observation.overworld is not None
        and observation.semantic_target.target_map == getattr(observation.overworld, "map_id", None)
        and _observed_interaction_goal(observation.overworld, observation.semantic_target) is not None
    ):
        # Once the ROM affordance is observed, let the generic interaction
        # navigator activate it.  This is important for the wall clock: the
        # physical approach tile alone does not guarantee that the observed
        # BG event will accept a direct A press.
        return EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD
    if (
        observation.objective_id == "set_wall_clock"
        and wall_clock_fact is False
        and observation.clock_target is not None
        and observation.controllable
        and observation.overworld is not None
    ):
        target, facing = observation.clock_target
        if observation.coordinates == target:
            if observation.overworld.facing == facing:
                return EmeraldCampaignAction.INTERACT_CLOCK
            return EmeraldCampaignAction.FACE_CLOCK
        return EmeraldCampaignAction.NAVIGATE_TO_CLOCK
    if observation.semantic_target is not None and observation.overworld is not None:
        # Other interaction targets are resolved from the current affordance
        # observation.  The campaign does not own coordinates, facing, or
        # the interaction button.
        return EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD
    if observation.controllable and observation.overworld is not None:
        return EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD
    return EmeraldCampaignAction.WAIT


def _observed_exit_goal(
    world: OverworldObservation,
    navigator: GoalAwareNavigator,
    semantic_target: SemanticTarget | None = None,
    navigation_progress: dict | None = None,
) -> ReachWarp | None:
    """Choose a tactical exit from the currently observed map.

    This is deliberately map-transition driven.  It does not identify the truck,
    house, stairs, or any other opening phase; the next observation decides
    again after the map changes.
    """
    if world is None or not world.controllable:
        return None
    # Cross-map affordances have two sources in live perception: ordinary ROM
    # warps and boundary MapConnections.  The latter intentionally live in
    # ``transitions`` rather than ``warps``; dropping them here means the
    # world graph can know a route that tactical policy never considers.
    exits = tuple(
        transition
        for transition in (getattr(world, "transitions", ()) or world.warps)
        if transition.destination is not None
        and transition.destination[0] != world.map_id
        and not (
            transition.kind == "map_connection"
            and transition_approach_position(transition)
            in {(world.map_id, coordinate) for coordinate in getattr(world, "dynamic_blocked_coordinates", ())}
        )
    )
    if not exits:
        return None

    # Sort by Manhattan distance to minimize the number of expensive pathfinding
    # calls.  A warp already under the player or very close is likely the
    # intended one.
    def distance(warp):
        return abs(warp.entry[1][0] - world.player_coordinates[0]) + abs(warp.entry[1][1] - world.player_coordinates[1])

    graph = None
    route_graph = None
    planned_next_maps: frozenset | None = None
    if semantic_target is not None and semantic_target.target_map is not None:
        # Use the world route as a direction-of-travel constraint, not merely
        # as a score for each observed exit.  Without this, a reverse edge can
        # remain cheaper than the next edge after a map transition and the
        # frame-local tactical planner can bounce between the two maps.
        try:
            from modules.world_navigation import get_world_map_graph

            route_graph = get_world_map_graph()
            world_route = route_graph.route(world.map_id, semantic_target.target_map)
            if world_route.edges:
                planned_next_maps = frozenset(
                    edge.destination_map for edge in world_route.edges if edge.source_map == world.map_id
                )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            route_graph = None
    relevance = {
        transition: (
            classify_transition_relevance(transition, semantic_target, graph)
            if graph is not None
            else classify_transition_relevance(transition, semantic_target)
        )
        for transition in exits
    }
    trace = getattr(context, "stutter_trace", None)
    if trace is not None:
        route = None
        if semantic_target is not None and semantic_target.target_map is not None:
            try:
                from modules.world_navigation import get_world_map_graph

                route = (graph or get_world_map_graph()).route(world.map_id, semantic_target.target_map)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                route = None
        trace.mark("navigation_boundary", True)
        trace.mark("navigation_current_map", world.map_id)
        trace.mark("navigation_player_position", world.player_coordinates)
        trace.mark("navigation_controllable", world.controllable)
        trace.mark("navigation_target", repr(semantic_target))
        trace.mark(
            "navigation_world_route",
            (
                None
                if route is None
                else {
                    "maps": route.maps,
                    "edges": tuple(
                        {
                            "type": edge.kind,
                            "source": edge.source_map,
                            "source_positions": edge.source_coordinates,
                            "destination": edge.destination_map,
                            "destination_positions": edge.destination_coordinates,
                            "estimated_cost": edge.estimated_cost,
                        }
                        for edge in route.edges
                    ),
                    "estimated_cost": route.estimated_cost,
                }
            ),
        )
        trace.mark(
            "navigation_observed_transitions",
            tuple(
                {
                    "type": transition.kind.upper(),
                    "source_position": transition.entry,
                    "destination": transition.destination,
                    "local_reachability": "not_evaluated",
                    "relevance": relevance[transition].name,
                    "survives_filtering": semantic_target is None
                    or relevance[transition] is TransitionRelevance.RELEVANT,
                }
                for transition in exits
            ),
        )
    if semantic_target is not None:
        relevant = tuple(transition for transition in exits if relevance[transition] is TransitionRelevance.RELEVANT)
        # A semantic map destination normally advances through a boundary
        # connection.  Interior ROM warps can also be globally relevant (for
        # example, a house can return to its town and then reach the target),
        # but must not outrank an available progression connection merely
        # because their local activation tile is closer.  Retain warp
        # candidates as a fallback when no connection can advance the route;
        # this preserves necessary backtracking and maps whose only observed
        # exit is an interior transition.  Remote interaction targets follow
        # the same progression rule; only a warp directly entering the target
        # map is allowed to outrank a boundary connection.
        # A direct boundary connection to the semantic target map is the
        # correct progression edge even when the target is an interaction.
        # Otherwise interior warps can win on local cost and send the player
        # through the house cycle while still being graph-reachable.
        progression = tuple(
            transition
            for transition in relevant
            if transition.kind == "map_connection"
            and transition.destination is not None
            and semantic_target.target_map == transition.destination[0]
        )
        if progression:
            relevant = progression
        if relevant:
            exits = relevant
        else:
            # A target exists, but no known relevant route exists. Do not
            # silently turn an unrelated exit into a semantic route.
            return None

    candidates = sorted(exits, key=distance)
    if planned_next_maps:
        progressing = tuple(
            transition
            for transition in candidates
            if transition.destination is not None and transition.destination[0] in planned_next_maps
        )
        if progressing:
            candidates = progressing

    def connection_key(transition):
        """Identify the ROM connection strip represented by aligned entries."""
        if transition.kind != "map_connection" or transition.destination is None:
            return None
        source_x, source_y = transition.entry[1]
        destination_x, destination_y = transition.destination[1]
        direction = transition.required_facing
        # A connection's offset is the alignment delta on the axis parallel
        # to its boundary.  Keeping it in the key prevents distinct strips to
        # the same map/direction from being merged.
        offset = (
            destination_x - source_x if direction in (Direction.North, Direction.South) else destination_y - source_y
        )
        return (
            transition.entry[0],
            transition.destination[0],
            direction,
            offset,
            transition.activation,
            transition.activation_direction,
        )

    downstream_routes: dict[object, object] = {}
    if semantic_target is not None:
        goal_candidates: list[tuple[int, object, object]] = []
        for warp in candidates:
            try:
                downstream = transition_world_route(warp, semantic_target, graph)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                downstream = False
            if downstream is None:
                continue
            downstream_routes[warp] = downstream
            goal_candidates.append((downstream.estimated_cost if downstream is not False else 0, warp, downstream))
        if not goal_candidates:
            return None
        # Prefer a locally observed boundary connection whenever it has a
        # viable downstream route.  Interior warps are often physically
        # closer, but selecting one solely on that distance can strand the
        # campaign in a house loop instead of advancing toward its semantic
        # map target.
        progressing_connections = tuple(
            warp
            for warp in candidates
            if warp.kind == "map_connection" and downstream_routes.get(warp) not in (None, False)
        )
        if semantic_target.kind is SemanticTargetKind.MAP:
            direct_target_warps = tuple()
        else:
            direct_target_warps = tuple(
                warp
                for warp in candidates
                if warp.kind != "map_connection"
                and warp.destination is not None
                and (
                    warp.destination[0] == semantic_target.target_map
                    or (
                        downstream_routes.get(warp) not in (None, False)
                        and all(edge.kind == "warp" for edge in downstream_routes[warp].edges)
                    )
                )
            )
        if progressing_connections and not direct_target_warps:
            candidates = progressing_connections
            goal_candidates = [
                candidate for candidate in goal_candidates if candidate[1] in progressing_connections
            ]
        # Do not discard a transition solely because its static downstream
        # estimate is higher.  That estimate says nothing about whether the
        # transition's activation tile is locally reachable from the current
        # observation.  Local feasibility must be established before the
        # combined route score is compared.
    planning_groups: list[tuple[object, ...]] = []
    connection_groups: dict[tuple, list[object]] = {}
    for transition in candidates:
        key = connection_key(transition)
        if key is None:
            planning_groups.append((transition,))
        elif key not in connection_groups:
            group: list[object] = []
            connection_groups[key] = group
            planning_groups.append(group)
            group.append(transition)
        else:
            connection_groups[key].append(transition)
    ranked: list[tuple[tuple, object]] = []
    previous_transition = None if navigation_progress is None else navigation_progress.get("previous_transition")
    # Evaluate every observed candidate that could advance the semantic goal.
    # The world route is a quality discriminator; local path cost remains the
    # tie-breaker among equally good downstream routes.
    for group in planning_groups:
        representative = group[0]
        target = (
            ReachWarp(destination_map=representative.destination[0], warps=tuple(group))
            if len(group) > 1
            else ReachWarp(
                destination_map=representative.destination[0],
                destination=representative.destination,
                warp=representative,
            )
        )
        try:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PLAN_TRACE: phase=exit_candidate_begin "
                    f"frame={getattr(context, 'frame', None)!r} current={world.map_id, world.player_coordinates!r} "
                    f"goal={target!r}"
                ),
                trace=True,
            )
            # Use A* for faster heuristic-guided search.
            plan = navigator.plan((world.map_id, world.player_coordinates), target, algorithm="astar")
            diagnostic_print(lambda: ("CAMPAIGN_PLAN_TRACE: phase=exit_candidate_end " f"plan={plan!r}"), trace=True)
        except NavigationError:
            continue
        # Navigation providers are allowed to decline a candidate without a
        # plan.  Treat that the same as an unreachable local exit rather than
        # dereferencing a diagnostic-only None result.
        if plan is None:
            continue
        warp = representative
        if len(group) > 1:
            matching = tuple(
                transition
                for transition in group
                if plan.actions
                and plan.actions[-1].action_type.name == "WARP"
                and plan.actions[-1].destination == transition.destination
            )
            if not matching:
                continue
            warp = matching[0]
        metrics = plan.metrics
        downstream = downstream_routes.get(warp, False)
        downstream_cost = downstream.estimated_cost if downstream not in (None, False) else 0
        local_cost = metrics.total_route_cost if metrics else float("inf")
        total_route_cost = downstream_cost + local_cost
        immediate_reverse = (
            previous_transition is not None
            and warp.entry[0] == previous_transition[1]
            and warp.destination[0] == previous_transition[0]
            and not (semantic_target is not None and semantic_target.target_map == previous_transition[0])
        )
        ranked.append(
            (
                (
                    total_route_cost,
                    downstream_cost,
                    metrics.encounter_opportunities if metrics else float("inf"),
                    local_cost,
                    metrics.movement_actions if metrics else float("inf"),
                    warp.entry[1],
                ),
                warp,
                immediate_reverse,
            )
        )
    if trace is not None:
        trace.mark(
            "navigation_candidate_evaluations",
            tuple(
                {
                    "type": transition.kind.upper(),
                    "source_position": transition.entry,
                    "destination": transition.destination,
                    "local_reachability": any(candidate[1] is transition for candidate in ranked),
                    "local_movement_cost": next(
                        (candidate[0][3] for candidate in ranked if candidate[1] is transition), None
                    ),
                    "downstream_world_route_cost": (
                        downstream_routes[transition].estimated_cost if transition in downstream_routes else None
                    ),
                    "total_cost": next((candidate[0][0] for candidate in ranked if candidate[1] is transition), None),
                    "relevance": relevance[transition].name,
                    "survives_filtering": transition in candidates,
                    "selected": False,
                }
                for transition in exits
            ),
        )
    if not ranked:
        # Preserve targetless escape behavior. With a semantic target, a
        # locally blocked relevant transition is explicit rather than an
        # excuse to choose an unrelated exit.
        if semantic_target is not None:
            return None
        selected = candidates[0]
    else:
        # Prefer continuing through a relevant, locally reachable exit over
        # immediately reversing the transition just completed.  If every
        # viable exit is a reversal, retain the ordinary cost ordering so
        # legitimate backtracking remains possible.
        forward = tuple(item for item in ranked if not item[2])
        pool = forward or tuple(ranked)
        selected = min(pool, key=lambda item: item[0])[1]
    if trace is not None:
        trace.mark(
            "navigation_selected_transition",
            {
                "type": selected.kind.upper(),
                "source_position": selected.entry,
                "destination": selected.destination,
                "relevance": relevance[selected].name,
            },
        )
    return ReachWarp(destination_map=selected.destination[0], destination=selected.destination, warp=selected)


def observation_driven_overworld_progression(
    interrupt: Callable[[], bool] | None = None,
    semantic_target: SemanticTarget | None = None,
    execution_cache: dict | None = None,
    objective_id: str | None = None,
) -> Iterator[object]:
    """Advance through observed overworld affordances one input at a time.

    A non-controllable scripted transition naturally produces no goal and no
    input.  Once control is returned, the current map's observed exit becomes
    a tactical ``ReachWarp`` goal.  The loop is recreated after a map change,
    so it never carries an opening route across an asynchronous transition.
    """
    # The campaign loop is recreated only at semantic/map boundaries.  Keep
    # the tactical loop in the short-lived capability generator so its
    # bounded AgentControlLoop route cache survives the next observation.
    # This is execution state, not campaign history: dropping the capability
    # drops the cache as well.
    execution_cache = execution_cache if execution_cache is not None else {}
    while True:
        # Navigation is subordinate to all semantic UI actions.  The caller
        # supplies the current-frame interrupt predicate so a route generator
        # cannot consume an input after dialogue/menu/naming/confirmation has
        # become actionable.
        if interrupt is not None and interrupt():
            yield
            continue
        try:
            current = perceive_overworld()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            current = None

        if interrupt is not None and interrupt():
            yield
            continue
        progress = execution_cache.setdefault("navigation_progress", {})
        pending_transition = progress.get("pending_transition")
        if pending_transition is not None and current is not None:
            if current.map_id == pending_transition[1] and current.map_id != pending_transition[0]:
                progress["previous_transition"] = pending_transition
                progress.pop("pending_transition", None)
        # A tactical plan is deliberately local to this observation.  The
        # next frame gets a new navigator and cannot inherit route ownership.
        if current is None or not current.controllable:
            yield
            continue
        navigator = GoalAwareNavigator(NavigationWorld.from_overworld(current))
        cache_key = (semantic_target, current.map_id)
        cached = execution_cache.get("overworld")
        if cached is not None and cached[0] != cache_key:
            evicted = execution_cache.pop("overworld", None)
            close = getattr(evicted[1], "close", None) if evicted is not None else None
            if callable(close):
                close()
                cached = None
        if cached is not None:
            loop = cached[1]
            try:
                result = next(loop)
            except StopIteration:
                execution_cache.pop("overworld", None)
            else:
                # A tactical loop owns a concrete observed transition.  If
                # that transition becomes unreachable (dynamic NPCs and
                # scripts can change the local affordance), discard the loop
                # so this observation-driven layer can resolve a fresh exit
                # rather than replaying a stale ReachWarp forever.
                if (
                    isinstance(result, tuple)
                    and len(result) >= 3
                    and getattr(result[2], "result_type", None) is ActionResultType.UNREACHABLE
                ):
                    close = getattr(loop, "close", None)
                    if callable(close):
                        close()
                    execution_cache.pop("overworld", None)
            yield
            continue
        interaction_goal = _observed_interaction_goal(current, semantic_target)
        if interaction_goal is not None:
            _publish_navigation_intent(semantic_target, "INTERACT")
            loop = AgentControlLoop(lambda: observe_agent(goal=interaction_goal), goal=interaction_goal).run()
            execution_cache["overworld"] = (cache_key, loop)
            try:
                next(loop)
            except StopIteration:
                execution_cache.pop("overworld", None)
                yield
                continue
            yield
            continue
        if objective_id == "receive_pokedex" and current.map_id == MapRSE.LITTLEROOT_TOWN.value:
            # Emerald starts the Pokédex sequence from the lab map's ROM
            # on-frame script after the player enters. Select only the
            # observed Littleroot -> Birch's Lab warp here; do not ask the
            # global transition scorer whether Route 101 can return to lab.
            goal = _observed_local_destination_goal(
                current,
                navigator,
                MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value,
            )
            if goal is not None:
                _publish_navigation_intent(semantic_target, "LOCAL LAB WARP")
                loop = AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()
                execution_cache["overworld"] = (cache_key, loop)
                try:
                    next(loop)
                except StopIteration:
                    execution_cache.pop("overworld", None)
                    yield
                    continue
                yield
                continue
        if semantic_target is not None and semantic_target.target_map == current.map_id:
            # A current-map interaction target is meaningful only when its
            # affordance is currently observed. Do not fall back to a generic
            # map escape route.
            yield
            continue
        goal = _observed_exit_goal(current, navigator, semantic_target, progress)
        if goal is None:
            _publish_navigation_intent(semantic_target, "no viable local transition")
            yield
            continue
        _publish_navigation_intent(
            semantic_target,
            f"{getattr(goal.warp, 'kind', 'warp').upper()} → "
            f"{goal.destination[0] if goal.destination is not None else 'UNKNOWN'}",
        )
        progress = execution_cache.setdefault("navigation_progress", {})
        if goal.warp is not None and goal.destination is not None:
            progress["pending_transition"] = (goal.warp.entry[0], goal.destination[0])
        loop = AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()
        execution_cache["overworld"] = (cache_key, loop)
        try:
            next(loop)
        except StopIteration:
            execution_cache.pop("overworld", None)
            yield
            continue
        yield


def _observed_local_destination_goal(
    world: OverworldObservation,
    navigator: GoalAwareNavigator,
    destination_map: tuple[int, int],
) -> ReachWarp | None:
    """Choose an observed exit to one campaign-specific local destination."""
    exits = tuple(
        transition
        for transition in (getattr(world, "transitions", ()) or world.warps)
        if transition.destination is not None and transition.destination[0] == destination_map
    )
    candidates: list[tuple[tuple, object]] = []
    for transition in exits:
        goal = ReachWarp(
            destination_map=destination_map,
            destination=transition.destination,
            warp=transition,
        )
        try:
            plan = navigator.plan((world.map_id, world.player_coordinates), goal, algorithm="astar")
        except NavigationError:
            continue
        if plan is None:
            continue
        metrics = plan.metrics
        candidates.append(
            (
                (
                    metrics.encounter_opportunities if metrics else float("inf"),
                    metrics.total_route_cost if metrics else float("inf"),
                    metrics.movement_actions if metrics else float("inf"),
                    transition.entry[1],
                ),
                transition,
            )
        )
    if not candidates:
        return None
    selected = min(candidates, key=lambda item: item[0])[1]
    return ReachWarp(destination_map=destination_map, destination=selected.destination, warp=selected)


def _observed_interaction_goal(
    world: OverworldObservation,
    semantic_target: SemanticTarget | None,
) -> ActivateTrigger | None:
    """Resolve a current-map semantic interaction to its observed affordance."""
    if semantic_target is None or semantic_target.target_map != world.map_id:
        return None
    interaction_id = semantic_target.interaction_id
    if interaction_id is None:
        return None
    for trigger in world.triggers:
        identities = (trigger.trigger_id, trigger.affordance_id, trigger.script_symbol)
        matches = interaction_id in identities
        if interaction_id == "wall_clock":
            # Keep the campaign target stable while accepting each gender/map
            # specific ROM script at the observation boundary.
            matches = matches or any(
                isinstance(identity, str) and identity.endswith("_EventScript_WallClock")
                for identity in identities
            )
        if matches and trigger.condition_active is not False and trigger.activation_locations:
            return ActivateTrigger(trigger.trigger_id)
    return None


def _observed_boundary_interaction_goal(
    world: OverworldObservation,
    semantic_target: SemanticTarget | None,
) -> ActivateTrigger | None:
    """Resolve an observed NPC occupying the approach to a target connection.

    Some Emerald opening boundaries are script gates: the player must talk to
    the NPC standing on the connection approach tile before the connection can
    be traversed.  Keep this observation-driven and generic; the campaign
    target remains the eventual semantic interaction on the destination map.
    """
    if semantic_target is None or semantic_target.target_map is None:
        return None
    target_connections = tuple(
        transition
        for transition in (getattr(world, "transitions", ()) or world.warps)
        if transition.kind == "map_connection"
        and transition.destination is not None
        and transition.destination[0] == semantic_target.target_map
    )
    approach_tiles = {
        transition_approach_position(transition)
        for transition in target_connections
        if transition_approach_position(transition) is not None
    }
    blocked = set(getattr(world, "dynamic_blocked_coordinates", ()))
    blocked_locations = {(world.map_id, coordinate) for coordinate in blocked}
    occupied_approaches = approach_tiles & blocked_locations
    if not occupied_approaches:
        return None
    for trigger in world.triggers:
        if trigger.condition_active is False:
            continue
        if (
            trigger.locations & blocked_locations
            and (world.map_id, world.player_coordinates) in trigger.activation_locations
        ):
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_NAVIGATION_TRACE: boundary_interaction_required "
                    f"frame={getattr(context, 'frame', None)!r} map={world.map_id!r} "
                    f"player={world.player_coordinates!r} target_map={semantic_target.target_map!r} "
                    f"trigger={trigger.trigger_id!r} occupied_approaches={occupied_approaches!r}"
                ),
                trace=True,
            )
            return ActivateTrigger(trigger.trigger_id)
    return None


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
        session_id = getattr(getattr(context, "nuzlocke_runtime", None), "session_id", None)
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
            session_id = getattr(getattr(context, "nuzlocke_runtime", None), "session_id", None)
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
        # Keep one authority for field-message lifecycle classification.  The
        # old opening adapter used to duplicate this predicate here, which
        # allowed a stale native callback or printer snapshot to disagree
        # with the interaction observer used by execution.
        dialogue_state = EmeraldOpeningCapability._dialogue_state_snapshot()
        dialogue_waiting, field_message_lifecycle_active = observe_dialogue(
            field_message_lifecycle_active,
            state=dialogue_state,
            advanceable_predicate=is_emerald_field_dialogue_advanceable,
        )
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
    try:
        rom_owned_movement = player_avatar_is_rom_owned_movement()
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        rom_owned_movement = False
    observed_text_speed_fast = (
        menu_observation is not None
        and menu_observation.menu_kind is EmeraldMenuKind.OPTIONS_MENU
        and menu_observation.raw_option_values is not None
        and len(menu_observation.raw_option_values) > 0
        and menu_observation.raw_option_values[0] == 2
    )
    return EmeraldCampaignObservation(
        _compatibility_opening_state(configured_gender),
        observed_text_speed_fast or state.campaign_facts.text_speed_fast.value is True,
        state.campaign_facts.new_game_setup_complete.value is True,
        dialogue_waiting,
        gender_task,
        menu_observation.cursor_index if menu_observation is not None else None,
        field_message_lifecycle_active,
        menu_observation,
        naming_observation,
        confirmation_observation,
        rom_owned_movement,
    )


def _opening_gender() -> object | None:
    """Resolve gender policy from ROM state/configuration, never opening phase."""
    try:
        return get_player().gender
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return _start_game_values()[1]


def _emerald_observation(
    field_message_lifecycle_active: bool = False,
    objective_id: str | None = None,
) -> EmeraldObservation:
    """Build the new frame-local observation used by semantic execution.

    ``_campaign_observation`` remains as a legacy diagnostic/classification
    adapter for existing callers.  Its opening state is intentionally not
    copied into this structure and cannot influence the new dispatcher.
    """
    legacy = _campaign_observation(field_message_lifecycle_active)
    live_context = _uses_shared_runtime_context()
    if live_context:
        try:
            starter_selection = observe_emerald_starter_selection()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            starter_selection = None
    else:
        # Sibling observers retain the canonical context object even when a
        # caller replaces this module's action context with a test double.
        starter_selection = None
    # These observations own the frame; constructing a navigation world would
    # be both unnecessary work and an architectural leak of lower-priority
    # navigation into UI/script ownership.
    if not live_context or legacy.dialogue_waiting or legacy.rom_owned_movement:
        overworld = None
    else:
        try:
            overworld = perceive_overworld()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            overworld = None
    if live_context:
        try:
            game_state = get_game_state()
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            game_state = None
    else:
        game_state = {
            OpeningSequenceState.TITLE: GameState.TITLE_SCREEN,
            OpeningSequenceState.MAIN_MENU: GameState.MAIN_MENU,
            OpeningSequenceState.OPTIONS_MENU: GameState.OPTIONS_MENU,
            OpeningSequenceState.PLAYER_NAMING: GameState.NAMING_SCREEN,
            OpeningSequenceState.STARTER_SELECTION: GameState.CHOOSE_STARTER,
        }.get(legacy.state)
    if legacy.naming_observation is not None:
        naming = legacy.naming_observation
        diagnostic_print(
            lambda: (
                "EMERALD_NAMING_OBSERVATION: "
                f"frame={getattr(context, 'frame', None)!r} game_state={game_state!r} "
                f"target={naming.target.name!r} template={naming.template_number!r} "
                f"pointer={naming.screen_pointer!r} keyboard_ready={naming.keyboard_ready!r} "
                f"species={naming.species_name!r} species_id={naming.species_id!r} "
                f"gender={naming.pokemon_gender!r} personality={naming.personality_value!r}"
            ),
            trace=True,
        )
    # Menu/naming observations remain authoritative if the cheap game-state
    # read is temporarily unavailable during a transition.
    if game_state not in {
        GameState.TITLE_SCREEN,
        GameState.MAIN_MENU,
        GameState.OPTIONS_MENU,
        GameState.NAMING_SCREEN,
    }:
        if legacy.menu_observation is not None:
            game_state = (
                GameState.OPTIONS_MENU
                if legacy.menu_observation.menu_kind is EmeraldMenuKind.OPTIONS_MENU
                else GameState.MAIN_MENU
            )
        elif (
            legacy.naming_observation is not None
            and legacy.naming_observation.screen_pointer is not None
        ):
            game_state = GameState.NAMING_SCREEN
    if not live_context:
        script_stack = ()
        native_task_state = (None, None, None)
        startup_tasks = ()
    else:
        try:
            startup_tasks = tuple(task.symbol for task in (get_tasks() or ()))
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            startup_tasks = ()
        try:
            script = get_global_script_context()
            script_stack = (
                tuple(symbol for symbol in (getattr(script, "stack", ()) or ()) if isinstance(symbol, str))
                if script is not None
                else ()
            )
            native_task_state = (
                (
                    getattr(script, "native_function_name", None),
                    getattr(script, "script_function_name", None),
                    getattr(script, "is_active", None),
                )
                if script is not None
                else (None, None, None)
            )
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            script_stack = ()
            native_task_state = (None, None, None)
    if overworld is not None:
        map_id = overworld.map_id
        coordinates = overworld.player_coordinates
        controllable = overworld.controllable
    else:
        map_id = None
        coordinates = None
        if live_context:
            try:
                controllable = player_avatar_is_controllable()
            except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
                controllable = False
        else:
            controllable = False
    observed_text_speed_fast = _observed_text_speed_fast() if live_context else None
    facts = (
        ("text_speed_fast", legacy.text_speed_fast if observed_text_speed_fast is None else observed_text_speed_fast),
        ("new_game_setup_complete", legacy.new_game_setup_complete),
        # Emerald's SET_WALL_CLOCK event flag is the ROM-owned completion
        # fact.  Keep the semantic name identical to CampaignFacts so the
        # dispatcher cannot accidentally ignore the live observation.
        ("wall_clock_set", _safe_event_flag("SET_WALL_CLOCK")),
    )
    clock_interaction = None
    if live_context:
        try:
            task_name = _active_clock_task()
            task = get_task(task_name) if task_name is not None else None
            if task is not None:
                clock_interaction = (
                    task_name,
                    task.data_value(0),
                    task.data_value(2),
                    task.data_value(3),
                    task.data_value(1),
                )
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
    clock_target = None
    if objective_id == "set_wall_clock" and map_id is not None:
        try:
            clock_target = _wall_clock_interaction(map_id, _opening_gender())
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
            clock_target = None
    semantic_target = _semantic_target_for_objective(objective_id)
    diagnostic_print(
        lambda: (
            "CAMPAIGN_CAPABILITY_HANDOFF: "
            f"capability_id={id(legacy)!r} objective_id={objective_id!r} "
            f"semantic_target={semantic_target!r}"
        ),
        trace=True,
    )
    _publish_campaign_status(objective_id, semantic_target, legacy, overworld)
    trace = getattr(context, "stutter_trace", None)
    if trace is not None and callable(getattr(trace, "mark", None)):
        # These are targeted replay fields, not per-frame console output.
        # They make scripted-interaction stalls diagnosable without turning
        # the normal observation path into a verbose logger.
        trace.mark("campaign_objective_id", objective_id)
        trace.mark("semantic_target", repr(semantic_target))
        trace.mark("dialogue_actionable", legacy.dialogue_waiting)
        trace.mark("dialogue_lifecycle_active", legacy.dialogue_lifecycle_active)
        trace.mark("rom_owned_movement", legacy.rom_owned_movement)
        trace.mark("controllable", controllable)
        trace.mark("native_task_state", repr(native_task_state))
        trace.mark("script_stack", script_stack)
        trace.mark("startup_tasks", startup_tasks)
        trace.mark("confirmation", repr(legacy.confirmation_observation))
        try:
            trace.mark("littleroot_intro_state", get_event_var("LITTLEROOT_INTRO_STATE"))
            trace.mark("littleroot_rival_state", get_event_var("LITTLEROOT_RIVAL_STATE"))
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
            pass
        if overworld is not None:
            trace.mark("observed_objects", repr(overworld.objects))
            trace.mark("observed_triggers", repr(overworld.triggers))
    return EmeraldObservation(
        game_state=game_state,
        ui_mode=game_state.name if game_state is not None else "UNKNOWN",
        title_actionable=game_state is GameState.TITLE_SCREEN,
        startup_tasks=startup_tasks,
        native_task_state=native_task_state,
        script_stack=script_stack,
        map_id=map_id,
        coordinates=coordinates,
        controllable=controllable,
        dialogue_actionable=legacy.dialogue_waiting,
        dialogue_lifecycle_active=legacy.dialogue_lifecycle_active,
        menu=legacy.menu_observation,
        naming=legacy.naming_observation,
        confirmation=legacy.confirmation_observation,
        starter_selection=starter_selection,
        gender_task=legacy.gender_task,
        rom_owned_movement=legacy.rom_owned_movement,
        clock_interaction=clock_interaction,
        clock_target=clock_target,
        campaign_facts=facts,
        overworld=overworld,
        objective_id=objective_id,
        semantic_target=semantic_target,
    )


def _semantic_target_for_objective(objective_id: str | None) -> SemanticTarget | None:
    """Translate the current campaign objective into a ROM affordance."""
    diagnostic_print(
        lambda: ("CAMPAIGN_SEMANTIC_TARGET_CALL: " f"capability_id={id(objective_id)!r} objective_id={objective_id!r}"),
        trace=True,
    )
    if objective_id == "rescue_birch":
        result = SemanticTarget.interaction(MapRSE.ROUTE101.value, interaction_id=_BIRCH_BAG_INTERACTION_ID)
        diagnostic_print(
            lambda: f"CAMPAIGN_SEMANTIC_TARGET_RESULT: objective_id={objective_id!r} target={result!r}", trace=True
        )
        return result
    if objective_id == "receive_pokedex":
        # Returning to Birch's lab activates Emerald's ROM-owned Pokédex
        # sequence.  The campaign supplies only the destination map and keeps
        # completion authoritative through the existing Pokédex flags.
        result = SemanticTarget.map(MapRSE.LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB.value)
        diagnostic_print(
            lambda: f"CAMPAIGN_SEMANTIC_TARGET_RESULT: objective_id={objective_id!r} target={result!r}", trace=True
        )
        return result
    if objective_id == "reach_petalburg":
        result = SemanticTarget.map(MapRSE.PETALBURG_CITY.value)
        diagnostic_print(
            lambda: f"CAMPAIGN_SEMANTIC_TARGET_RESULT: objective_id={objective_id!r} target={result!r}", trace=True
        )
        return result
    if objective_id == "complete_intro_rival":
        # This is a Route 103 overworld trigger. The earlier ``meet_rival``
        # objective intentionally targets the rival's house, but it must not
        # be reused for the subsequent introductory battle.
        result = SemanticTarget.interaction(
            MapRSE.ROUTE103.value,
            interaction_id="Route103_EventScript_Rival",
        )
        diagnostic_print(
            lambda: f"CAMPAIGN_SEMANTIC_TARGET_RESULT: objective_id={objective_id!r} target={result!r}", trace=True
        )
        return result
    if objective_id not in {"set_wall_clock", "meet_rival"}:
        return None

    try:
        gender = _opening_gender()
        target_map = (
            _player_house_map(2, gender) if objective_id == "set_wall_clock" else _rival_house_map(2, gender)
        ).value
        interaction_id = "wall_clock" if objective_id == "set_wall_clock" else None
        try:
            if objective_id == "meet_rival":
                # Meeting the rival is the ROM's rival Poké Ball interaction,
                # not an instruction to visit an arbitrary upstairs map.
                # Resolve the script from the current house metadata so the
                # target remains gender/layout-aware and stateless.
                intro_state = get_event_var("LITTLEROOT_INTRO_STATE")
                if intro_state == 6:
                    # The post-clock 1F arrival has an active coordinate
                    # script before the rival-room interaction becomes the
                    # next affordance. Resolve it from ROM map metadata.
                    arrival_map = _player_house_map(1, gender).value
                    arrival_metadata = get_map_metadata(arrival_map)
                    arrival_script = next(
                        (
                            event.script_symbol
                            for event in arrival_metadata.coord_events
                            if event.script_symbol is not None
                            and event.script_symbol.endswith("_EventScript_GoSeeRoom")
                        ),
                        None,
                    )
                    if arrival_script is not None:
                        result = SemanticTarget.interaction(arrival_map, interaction_id=arrival_script)
                        diagnostic_print(
                            lambda: f"CAMPAIGN_SEMANTIC_TARGET_RESULT: objective_id={objective_id!r} target={result!r}",
                            trace=True,
                        )
                        return result
                interaction_id = next(
                    (
                        obj.script_symbol
                        for obj in get_map_metadata(target_map).objects
                        if obj.script_symbol is not None and obj.script_symbol.endswith("_EventScript_RivalsPokeBall")
                    ),
                    None,
                )
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            pass
        if interaction_id is None:
            return None
        result = SemanticTarget.interaction(target_map, interaction_id=interaction_id)
        diagnostic_print(
            lambda: f"CAMPAIGN_SEMANTIC_TARGET_RESULT: objective_id={objective_id!r} target={result!r}", trace=True
        )
        return result
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return None


def _publish_campaign_status(
    objective_id: str | None,
    target: SemanticTarget | None,
    observation: EmeraldCampaignObservation,
    overworld,
) -> None:
    """Publish the policy's current semantic state without affecting policy."""
    descriptions = {
        "set_text_speed": "Set Text Speed",
        "complete_new_game_setup": "Complete New Game Setup",
        "set_wall_clock": "Set Wall Clock",
        "meet_rival": "Meet Rival",
        "rescue_birch": "Rescue Professor Birch",
        "obtain_starter": "Obtain Starter Pokémon",
        "complete_intro_rival": "Complete Introductory Rival Battle",
        "receive_pokedex": "Receive Pokédex",
        "receive_pokeballs": "Receive Poké Balls",
        "start_nuzlocke": "Begin Nuzlocke",
    }
    label = descriptions.get(objective_id, objective_id) if objective_id is not None else None
    if label is None:
        status = CampaignStatus(None, None, "No semantic objective")
    elif objective_id == "obtain_starter":
        status = CampaignStatus(label, None, "Observe ROM starter-selection UI")
    elif target is None:
        status = CampaignStatus(label, None, "Targetless fallback")
    elif observation.dialogue_waiting or observation.rom_owned_movement:
        status = CampaignStatus(label, target, "Waiting / target temporarily unavailable")
    elif target.kind.name == "INTERACTION" and overworld is not None and target.target_map == overworld.map_id:
        status = CampaignStatus(label, target, "Interact")
    else:
        status = CampaignStatus(label, target, "Navigate")
    context.campaign_status = status


def _publish_navigation_intent(target: SemanticTarget | None, intent: str) -> None:
    """Update the GUI only when the selected semantic navigation changes."""
    status = getattr(context, "campaign_status", None)
    if status is None or target is None:
        return
    context.campaign_status = CampaignStatus(status.objective, target, intent)


def _safe_event_flag(name: str) -> bool | None:
    try:
        return bool(get_event_flag(name))
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
        return None


def _observed_text_speed_fast() -> bool | None:
    """Read the ROM-owned text-speed value without consulting opening phase."""
    try:
        for task_name in (
            "Task_OptionMenuProcessInput",
            "Task_OptionMenuFadeIn",
            "Task_OptionMenuSave",
            "Task_OptionMenuFadeOut",
        ):
            task = get_task(task_name)
            if task is not None:
                return task.data_value(1) == 2
        return (unpack_uint16(get_save_block(2, offset=0x14, size=2)) & 0x07) == 2
    except (AttributeError, RuntimeError, ValueError, TypeError, IndexError, KeyError):
        return None


def _main_menu_target_index(menu: EmeraldMenuObservation, target: EmeraldMainMenuItem) -> int | None:
    """Return the target's index for the observed save-dependent menu."""
    if menu.raw_menu_type is None:
        return None
    if target is EmeraldMainMenuItem.OPTIONS:
        return menu.raw_menu_type + 1
    if target is EmeraldMainMenuItem.NEW_GAME:
        return 0 if menu.raw_menu_type == 0 else 1
    return None


def _menu_button(
    observation: EmeraldCampaignObservation | EmeraldObservation, action: EmeraldCampaignAction
) -> str | None:
    """Choose one input from the current normalized menu observation."""
    menu = getattr(observation, "menu", None) or getattr(observation, "menu_observation", None)
    if menu is None or not menu.input_ready or menu.cursor_index is None:
        return None
    if menu.menu_kind is EmeraldMenuKind.MAIN_MENU:
        if not isinstance(menu.selected_item, EmeraldMainMenuItem):
            return None
        text_speed_fast = getattr(observation, "text_speed_fast", None)
        if text_speed_fast is None:
            text_speed_fast = dict(getattr(observation, "campaign_facts", ())).get("text_speed_fast", False)
        target = EmeraldMainMenuItem.OPTIONS if not text_speed_fast else EmeraldMainMenuItem.NEW_GAME
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


def _press_confirmation_button(button: str) -> None:
    """Emit a new input edge for a ROM-owned Yes/No handler.

    A confirmation task can be installed immediately after a dialogue A.
    ``press_button`` intentionally suppresses a repeated held input, whereas
    the new task needs its own JOY_NEW edge.  Keep this scoped to confirmed
    Yes/No observations; ordinary dialogue retains its normal lifecycle.
    """
    press_fresh = getattr(context.emulator, "press_button_fresh", None)
    if callable(press_fresh):
        press_fresh(button)
    else:
        context.emulator.press_button(button)


def observation_driven_emerald_campaign(objective_id: str | None = None) -> Iterator[object]:
    """Execute at most one bounded action, then yield for fresh perception."""
    field_message_lifecycle_active = False
    last_stable_map_id = None

    def higher_priority_actionable() -> bool:
        # Re-observe through the same normalized pipeline.  Avoid recursion by
        # checking only actions that outrank overworld movement.
        try:
            current = _emerald_observation(field_message_lifecycle_active, objective_id)
            return choose_emerald_observation_action(current) is not EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD
        except (AttributeError, RuntimeError, ValueError, TypeError, IndexError):
            return False

    resolved_player_initialization: StartGameInitialization | None = None
    session_id = getattr(getattr(context, "nuzlocke_runtime", None), "session_id", None)
    rng = random.Random(session_id)
    clock_target_time = None
    tactical_execution_cache: dict = {}
    resolved_starter = None
    pending_starter_signature = None
    pending_starter_quiet_frames = 0
    pending_nickname_confirmation = None
    pending_nickname_quiet_frames = 0
    pending_go_see_rival_confirmation = None
    pending_go_see_rival_quiet_frames = 0
    pending_dialogue = False
    pending_dialogue_quiet_frames = 0
    post_clock_dialogue_reset_done = False
    pending_menu_signature = None
    pending_menu_action = None
    awaiting_nickname_screen = False
    nickname = None
    nickname_input = None
    player_name_input = None
    if objective_id == "obtain_starter":
        configured = configured_emerald_starter()
        resolved_starter = resolve_emerald_starter(configured, rng)
    if objective_id == "set_wall_clock":
        clock_target_time = _emerald_clock_time(_clock_time_mode(), rng=rng)
    while True:
        observation = _emerald_observation(field_message_lifecycle_active, objective_id)
        # A field-message lifecycle belongs to the map/script that created
        # it. Do not carry the pre-warp printer state into the first stable
        # frame on the destination map; doing so makes the destination's
        # first dialogue intermittently look like a stale transition.
        if (
            observation.map_id is not None
            and last_stable_map_id is not None
            and observation.map_id != last_stable_map_id
        ):
            field_message_lifecycle_active = False
            observation = _emerald_observation(False, objective_id)
        if observation.map_id is not None:
            last_stable_map_id = observation.map_id
        field_message_lifecycle_active = observation.dialogue_lifecycle_active
        action = choose_emerald_observation_action(observation)
        diagnostic_print(
            lambda: (
                "EMERALD_CAMPAIGN_DECISION: "
                f"frame={getattr(context, 'frame', None)!r} objective={objective_id!r} "
                f"action={action.name!r} game_state={observation.game_state!r} "
                f"dialogue_actionable={observation.dialogue_actionable!r} "
                f"dialogue_lifecycle={observation.dialogue_lifecycle_active!r} "
                f"script={observation.native_task_state!r} tasks={observation.startup_tasks!r}"
            ),
            trace=True,
        )
        # The pure dispatcher cannot own session RNG. Resolve Random once per
        # mounted capability, then apply that stable choice to the observed
        # starter menu instead of waiting forever on the literal setting.
        starter = observation.starter_selection
        if (
            objective_id == "obtain_starter"
            and starter is not None
            and starter.phase is EmeraldStarterSelectionPhase.CHOOSING
            and starter.input_ready
            and resolved_starter is not None
        ):
            target_index = starter.choices.index(resolved_starter)
            if starter.selected_index < target_index:
                action = EmeraldCampaignAction.MOVE_STARTER_RIGHT
            elif starter.selected_index > target_index:
                action = EmeraldCampaignAction.MOVE_STARTER_LEFT
            else:
                action = EmeraldCampaignAction.CHOOSE_STARTER
        # A clock interaction starts a new ROM-owned dialogue lifecycle.  Do
        # not carry the previous field-message debounce state across that
        # boundary or the first clock prompt will be suppressed forever.
        if action is EmeraldCampaignAction.INTERACT_CLOCK or observation.clock_interaction is not None:
            pending_dialogue = False
            pending_dialogue_quiet_frames = 0
            post_clock_dialogue_reset_done = False
        elif (
            objective_id == "set_wall_clock"
            and observation.semantic_target is not None
            and observation.overworld is not None
            and observation.semantic_target.target_map == getattr(observation.overworld, "map_id", None)
            and _observed_interaction_goal(observation.overworld, observation.semantic_target) is not None
        ):
            # The generic observed-overworld interaction can activate the
            # clock affordance. Treat that as the start of a new dialogue
            # lifecycle too.
            pending_dialogue = False
            pending_dialogue_quiet_frames = 0
            post_clock_dialogue_reset_done = False
        elif (
            objective_id == "set_wall_clock"
            and action is EmeraldCampaignAction.ADVANCE_DIALOGUE
            and pending_dialogue
            and observation.map_id is None
            and not post_clock_dialogue_reset_done
        ):
            # The clock's ROM script temporarily removes the map projection
            # before exposing its first field prompt. This is a new dialogue
            # lifecycle even though the previous field message debounce flag
            # is still set.
            pending_dialogue = False
            pending_dialogue_quiet_frames = 0
            post_clock_dialogue_reset_done = True
        if objective_id == "set_wall_clock":
            diagnostic_print(
                lambda: (
                    "EMERALD_CLOCK_DECISION: "
                    f"frame={getattr(context, 'frame', None)!r} action={action.name!r} "
                    f"map={observation.map_id!r} coordinates={observation.coordinates!r} "
                    f"clock_target={observation.clock_target!r} facing="
                    f"{getattr(observation.overworld, 'facing', None)!r} "
                    f"clock_interaction={observation.clock_interaction!r} "
                    f"dialogue_actionable={observation.dialogue_actionable!r} "
                    f"dialogue_lifecycle={observation.dialogue_lifecycle_active!r} "
                    f"wall_clock_fact={dict(observation.campaign_facts).get('wall_clock_set')!r}"
                ),
                trace=True,
            )
        if not observation.dialogue_actionable:
            pending_dialogue = False
            pending_dialogue_quiet_frames = 0
        elif action is EmeraldCampaignAction.ADVANCE_DIALOGUE and pending_dialogue:
            # Normally the ROM changes the printer/task state after the fresh
            # edge.  Some Birch messages keep the same actionable observation
            # for several frames, however; permanently suppressing input here
            # strands the script.  Give the handoff two neutral frames, then
            # retry from the current observation if the ROM still wants A.
            if pending_dialogue_quiet_frames < 2:
                pending_dialogue_quiet_frames += 1
                action = EmeraldCampaignAction.WAIT
            else:
                pending_dialogue = False
                pending_dialogue_quiet_frames = 0
        if objective_id == "set_wall_clock":
            diagnostic_print(
                lambda: (
                    "EMERALD_CLOCK_DISPATCH: "
                    f"frame={getattr(context, 'frame', None)!r} action={action.name!r} "
                    f"pending_dialogue={pending_dialogue!r} "
                    f"dialogue_actionable={observation.dialogue_actionable!r}"
                ),
                trace=True,
            )

        menu_actions = {
            EmeraldCampaignAction.ENTER_OPTIONS,
            EmeraldCampaignAction.ADVANCE_TEXT_SPEED,
            EmeraldCampaignAction.EXIT_OPTIONS,
            EmeraldCampaignAction.START_NEW_GAME,
        }
        menu_signature = (
            observation.ui_mode,
            tuple(observation.campaign_facts),
            getattr(observation.menu, "cursor", None),
            repr(observation.menu),
        )
        if action in menu_actions:
            if pending_menu_signature == menu_signature and pending_menu_action is action:
                action = EmeraldCampaignAction.WAIT
            else:
                pending_menu_signature = None
                pending_menu_action = None
        else:
            pending_menu_signature = None
            pending_menu_action = None
        if observation.naming is not None:
            awaiting_nickname_screen = False
        elif awaiting_nickname_screen and observation.confirmation is None:
            # The accepted prompt transfers ownership to DoNamingScreen
            # asynchronously. Do not let an empty transition frame fall back
            # to overworld campaign navigation.
            action = EmeraldCampaignAction.WAIT
        starter = observation.starter_selection
        if (
            starter is not None
            and resolved_starter is not None
            and starter.phase is EmeraldStarterSelectionPhase.CHOOSING
        ):
            target_index = starter.choices.index(resolved_starter)
            if starter.selected_index < target_index:
                action = EmeraldCampaignAction.MOVE_STARTER_RIGHT
            elif starter.selected_index > target_index:
                action = EmeraldCampaignAction.MOVE_STARTER_LEFT
            else:
                action = EmeraldCampaignAction.CHOOSE_STARTER
        starter_signature = None if starter is None else (starter.phase, starter.selected_index, starter.input_ready)
        if starter_signature != pending_starter_signature:
            pending_starter_signature = None
            pending_starter_quiet_frames = 0
        if (
            action
            in {
                EmeraldCampaignAction.MOVE_STARTER_LEFT,
                EmeraldCampaignAction.MOVE_STARTER_RIGHT,
                EmeraldCampaignAction.CHOOSE_STARTER,
                EmeraldCampaignAction.CONFIRM_STARTER,
            }
            and pending_starter_signature == starter_signature
        ):
            if pending_starter_quiet_frames < 2:
                pending_starter_quiet_frames += 1
                action = EmeraldCampaignAction.WAIT
            else:
                # A fresh pulse can be dropped while Emerald hands the task
                # from animation to input. Retry only after two quiet frames.
                pending_starter_signature = None
                pending_starter_quiet_frames = 0

        nickname_confirmation_signature = None
        if (
            observation.confirmation is not None
            and observation.confirmation.context is EmeraldConfirmationContext.POKEMON_NICKNAME
        ):
            nickname_confirmation_signature = (
                observation.confirmation.task_name,
                observation.confirmation.selected,
                observation.confirmation.input_ready,
            )
        if nickname_confirmation_signature != pending_nickname_confirmation:
            pending_nickname_confirmation = None
            pending_nickname_quiet_frames = 0
        if (
            nickname_confirmation_signature is not None
            and action
            in {
                EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES,
                EmeraldCampaignAction.CHOOSE_POKEMON_NICKNAME,
            }
            and pending_nickname_confirmation == nickname_confirmation_signature
        ):
            if pending_nickname_quiet_frames < 2:
                pending_nickname_quiet_frames += 1
                action = EmeraldCampaignAction.WAIT
            else:
                # The first pulse can be dropped during the yes/no task's
                # input handoff. Retry only after two unchanged observations.
                pending_nickname_confirmation = None
                pending_nickname_quiet_frames = 0

        go_see_rival_signature = None
        if (
            observation.confirmation is not None
            and observation.confirmation.context is EmeraldConfirmationContext.GO_SEE_RIVAL
        ):
            go_see_rival_signature = (
                observation.confirmation.task_name,
                observation.confirmation.selected,
                observation.confirmation.input_ready,
                observation.confirmation.script_identity,
            )
        if go_see_rival_signature != pending_go_see_rival_confirmation:
            pending_go_see_rival_confirmation = None
            pending_go_see_rival_quiet_frames = 0
        if (
            go_see_rival_signature is not None
            and action
            in {
                EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES,
                EmeraldCampaignAction.CONFIRM_GO_SEE_RIVAL,
            }
            and pending_go_see_rival_confirmation == go_see_rival_signature
        ):
            if pending_go_see_rival_quiet_frames < 2:
                pending_go_see_rival_quiet_frames += 1
                action = EmeraldCampaignAction.WAIT
            else:
                pending_go_see_rival_confirmation = None
                pending_go_see_rival_quiet_frames = 0

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
            # The executor owns the same-frame dialogue observation.  Passing
            # only the lifecycle keeps this boundary compatible with the
            # legacy helper contract and avoids making a stale dispatcher
            # decision authoritative at execution time.
            yield from _advance_scripted_input(field_message_lifecycle_active)
            pending_dialogue = True
            pending_dialogue_quiet_frames = 0
            issued = True
        elif action is EmeraldCampaignAction.ENTER_NAME:
            if observation.naming is None or observation.naming.target is not EmeraldNamingTarget.PLAYER_NAME:
                yield
                continue
            if resolved_player_initialization is None:
                resolved_player_initialization = _resolve_player_campaign_initialization(rng)
            if player_name_input is None:
                player_name_input = type_in_naming_screen(resolved_player_initialization.name)
            try:
                next(player_name_input)
            except StopIteration:
                player_name_input = None
            yield
            continue
        elif action is EmeraldCampaignAction.CHOOSE_GENDER:
            if resolved_player_initialization is None:
                resolved_player_initialization = _resolve_player_campaign_initialization(rng)
            yield from _gender_input(resolved_player_initialization.gender.value)
            issued = True
        elif action is EmeraldCampaignAction.ADVANCE_NAME_PROMPT:
            yield from _advance_name_prompt()
            issued = True
        elif action is EmeraldCampaignAction.MOVE_CONFIRMATION_TO_YES:
            confirmation = observation.confirmation
            if (
                confirmation is None
                or not confirmation.input_ready
                or confirmation.selected is not EmeraldConfirmationChoice.NO
            ):
                yield
                continue
            _press_confirmation_button("Up")
            if confirmation.context is EmeraldConfirmationContext.POKEMON_NICKNAME:
                pending_nickname_confirmation = nickname_confirmation_signature
            elif confirmation.context is EmeraldConfirmationContext.GO_SEE_RIVAL:
                pending_go_see_rival_confirmation = go_see_rival_signature
                pending_go_see_rival_quiet_frames = 0
            issued = True
        elif action is EmeraldCampaignAction.CONFIRM_PLAYER_NAME:
            confirmation = observation.confirmation
            if (
                confirmation is None
                or not confirmation.input_ready
                or confirmation.selected is not EmeraldConfirmationChoice.YES
            ):
                yield
                continue
            _press_confirmation_button("A")
            issued = True
        elif action is EmeraldCampaignAction.CHOOSE_POKEMON_NICKNAME:
            confirmation = observation.confirmation
            if (
                confirmation is None
                or confirmation.context is not EmeraldConfirmationContext.POKEMON_NICKNAME
                or not confirmation.input_ready
                or confirmation.selected is not EmeraldConfirmationChoice.YES
            ):
                yield
                continue
            _press_confirmation_button("A")
            pending_nickname_confirmation = nickname_confirmation_signature
            pending_nickname_quiet_frames = 0
            awaiting_nickname_screen = True
            issued = True
        elif action is EmeraldCampaignAction.CONFIRM_GO_SEE_RIVAL:
            confirmation = observation.confirmation
            if (
                confirmation is None
                or confirmation.context is not EmeraldConfirmationContext.GO_SEE_RIVAL
                or not confirmation.input_ready
                or confirmation.selected is not EmeraldConfirmationChoice.YES
            ):
                yield
                continue
            _press_confirmation_button("A")
            pending_go_see_rival_confirmation = go_see_rival_signature
            pending_go_see_rival_quiet_frames = 0
            issued = True
        elif action is EmeraldCampaignAction.ENTER_POKEMON_NICKNAME:
            naming = observation.naming
            if naming is None or not naming.keyboard_ready or naming.species_name is None:
                diagnostic_print(
                    lambda: (
                        "EMERALD_NAMING_ACTION: action='ENTER_POKEMON_NICKNAME' "
                        f"result='WAIT' naming={naming!r}"
                    ),
                    trace=True,
                )
                yield
                continue
            if nickname is None:
                nickname = generate_pokemon_nickname(naming.species_name, naming.pokemon_gender)
                diagnostic_print(
                    lambda: (
                        "EMERALD_NAMING_ACTION: action='ENTER_POKEMON_NICKNAME' "
                        f"result='GENERATED' nickname={nickname!r} species={naming.species_name!r} "
                        f"gender={naming.pokemon_gender!r}"
                    ),
                    trace=True,
                )
            if nickname_input is None:
                nickname_input = type_in_naming_screen(nickname, max_length=POKEMON_NICKNAME_MAX_LENGTH)
                diagnostic_print(
                    lambda: "EMERALD_NAMING_ACTION: action='ENTER_POKEMON_NICKNAME' result='KEYBOARD_STARTED'",
                    trace=True,
                )
            try:
                next(nickname_input)
            except StopIteration:
                nickname_input = None
            yield
            continue
        elif action is EmeraldCampaignAction.FACE_CLOCK:
            context.emulator.press_button(observation.clock_target[1])
            issued = True
        elif action is EmeraldCampaignAction.INTERACT_CLOCK:
            # The clock script immediately opens a field message.  Use a
            # fresh edge here so the interaction A cannot be carried across
            # the script handoff and consume the first dialogue advance.
            press_button_fresh = getattr(context.emulator, "press_button_fresh", None)
            if callable(press_button_fresh):
                context.emulator.press_button_fresh("A")
            else:
                context.emulator.press_button("A")
            diagnostic_print(
                lambda: (
                    "EMERALD_CLOCK_INPUT: "
                    f"frame={getattr(context, 'frame', None)!r} action='INTERACT_CLOCK' "
                    "button='A' fresh=True emitted=True"
                ),
                trace=True,
            )
            issued = True
        elif action is EmeraldCampaignAction.NAVIGATE_TO_CLOCK:
            target = observation.clock_target
            if target is None or observation.map_id is None:
                yield
                continue
            loop = AgentControlLoop(
                lambda: observe_agent(goal=ReachLocation((observation.map_id, target[0]))),
                goal=ReachLocation((observation.map_id, target[0])),
            ).run()
            try:
                next(loop)
            except StopIteration:
                pass
            yield
            continue
        elif action is EmeraldCampaignAction.ADVANCE_CLOCK:
            interaction = observation.clock_interaction
            if interaction is None or clock_target_time is None:
                yield
                continue
            if len(interaction) >= 5:
                _, hand_angle, current_hour, current_minute, _ = interaction
            else:
                _, current_hour, current_minute = interaction
                hand_angle = 0
            if hand_angle is None or hand_angle % 6:
                yield
                continue
            if (current_hour, current_minute) == clock_target_time:
                context.emulator.press_button("A")
                clock_button = "A"
            else:
                clock_button = _clock_input_direction(
                        current_hour,
                        current_minute,
                        clock_target_time[0],
                        clock_target_time[1],
                    )
                context.emulator.press_button(clock_button)
            diagnostic_print(
                lambda: (
                    "EMERALD_CLOCK_INPUT: "
                    f"frame={getattr(context, 'frame', None)!r} action='ADVANCE_CLOCK' "
                    f"button={clock_button!r} interaction={interaction!r} emitted=True"
                ),
                trace=True,
            )
            issued = True
        elif action is EmeraldCampaignAction.CONFIRM_CLOCK:
            interaction = observation.clock_interaction
            cursor = interaction[4] if interaction is not None and len(interaction) > 4 else None
            # Emerald opens this confirmation on NO (cursor 1). The cursor is
            # observed from the current task; no "prepared" flag is retained.
            context.emulator.press_button("Up" if cursor == 1 else "A")
            issued = True
        elif action in {
            EmeraldCampaignAction.MOVE_STARTER_LEFT,
            EmeraldCampaignAction.MOVE_STARTER_RIGHT,
            EmeraldCampaignAction.CHOOSE_STARTER,
            EmeraldCampaignAction.CONFIRM_STARTER,
        }:
            starter = observation.starter_selection
            if starter is None or not starter.input_ready:
                yield
                continue
            button = {
                EmeraldCampaignAction.MOVE_STARTER_LEFT: "Left",
                EmeraldCampaignAction.MOVE_STARTER_RIGHT: "Right",
                EmeraldCampaignAction.CHOOSE_STARTER: "A",
                EmeraldCampaignAction.CONFIRM_STARTER: "A",
            }[action]
            context.emulator.press_button(button)
            pending_starter_signature = starter_signature
            pending_starter_quiet_frames = 0
            issued = True
        elif action is EmeraldCampaignAction.ADVANCE_OBSERVED_OVERWORLD:
            # Tactical navigation owns the single-input/re-observation
            # transaction.  Campaign facts remain outside this loop and are
            # re-evaluated by CampaignController on the next frame.
            # The target is read again from the same current observation that
            # selected this action; no route is retained across frames.
            next(
                observation_driven_overworld_progression(
                    higher_priority_actionable,
                    observation.semantic_target,
                    tactical_execution_cache,
                    objective_id,
                )
            )
            yield
            continue
        else:
            yield
            continue

        if issued and action in menu_actions:
            pending_menu_signature = menu_signature
            pending_menu_action = action
        yield


def emerald_campaign_capability(objective_id: str) -> Iterator[object]:
    """Mount the observation-driven capability for an Emerald objective.

    Every objective intentionally shares a re-observing loop: a new generator
    can begin at any currently observable title/menu/dialogue/overworld state.
    """

    # Campaign objectives are executed by a current-observation loop. The
    # legacy opening state machine remains available for its direct callers,
    # but is no longer mounted by CampaignProgression.
    # Every Emerald campaign objective uses the same frame-local executor.
    # The objective identifies the incomplete fact and supplies only policy
    # context; it never selects an opening phase or route.
    yield from observation_driven_emerald_campaign(objective_id)


__all__ = [
    "EmeraldCampaignAction",
    "EmeraldCampaignObservation",
    "EmeraldObservation",
    "choose_emerald_campaign_action",
    "choose_emerald_observation_action",
    "emerald_campaign_capability",
]
