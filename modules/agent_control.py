"""State-specific affordances, goal evaluation, and action execution.

The control loop is intentionally independent from ``main_loop`` and bot
modes.  A future mode or strategic planner can supply observations and goals,
then mount ``AgentControlLoop.run()`` on the existing controller stack when it
is ready to do so.
"""

from dataclasses import dataclass, replace
from enum import Enum, auto
from contextlib import nullcontext
from time import perf_counter_ns
from typing import Callable, Generator

from modules.context import context
from modules.console import diagnostic_print, profile_print
from modules.goals import (
    ActivateTrigger,
    EngageTrainer,
    Goal,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
)
from modules.interaction_state import (
    InteractionObservation,
    InteractionPhase,
    InteractionType,
    classify_interaction,
    is_emerald_field_message_rendering,
    observe_interaction,
)
from modules.map_path import Direction
from modules.map import get_event_flag, get_map_data, get_runtime_object_table
from modules.map_data import MapRSE, get_map_enum
from modules.game import get_event_flag_name
from modules.memory import GameState, get_game_state, get_save_block
from modules.navigation import (
    GoalAwareNavigator,
    NavigationAction,
    NavigationActionType,
    NavigationError,
    NavigationPlan,
    RoutePlan,
    RouteCheckpoint,
    record_runtime_transition_observation,
    NavigationWorld,
    WorldNavigationError,
    goal_target_map,
    plan_with_world_navigation,
    plan_observed_warp_locally,
    navigation_diagnostics,
    prewarm_navigation_tiles,
)
from modules.overworld import (
    MovementState,
    OverworldObservation,
    perceive_overworld,
    prewarm_static_map_observation,
)
from modules.player import get_player_avatar
from modules.profiler import count, invalidation, now, profiled, timing, format_snapshot
from modules.tasks import is_field_message_waiting_for_input
from modules.state_cache import state_cache
from modules.nuzlocke.readiness_diagnostics import (
    build_progression_readiness_diagnostic,
    evaluate_progression_readiness,
)
import json

UNIVERSAL_REOBSERVE_FRAMES = 90
_last_universal_observation_frame: int | None = None


def _force_universal_reobserve() -> bool:
    """Request a complete ROM-state observation at a fixed frame boundary.

    A map warp is allowed to expose a transient ``UNKNOWN`` or ``CHANGE_MAP``
    callback while the avatar/save-block map has already settled.  Cache
    invalidation alone cannot reveal that state when the normal observer
    consequently skips overworld perception, so callers use the returned
    value to include one authoritative overworld pass at this boundary.
    """
    global _last_universal_observation_frame
    emulator = getattr(context, "emulator", None)
    get_frame_count = getattr(emulator, "get_frame_count", None)
    if not callable(get_frame_count):
        return False
    try:
        frame = get_frame_count()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if not isinstance(frame, int):
        return False
    if (
        _last_universal_observation_frame is None
        or frame < _last_universal_observation_frame
        or frame - _last_universal_observation_frame >= UNIVERSAL_REOBSERVE_FRAMES
    ):
        if _last_universal_observation_frame is not None:
            state_cache.invalidate_runtime_observations()
        _last_universal_observation_frame = frame
        return True
    return False


class AgentActionType(Enum):
    ADVANCE_DIALOGUE = auto()
    ACCELERATE_DIALOGUE_RENDER = auto()
    CHOOSE_DIALOGUE_OPTION = auto()
    MOVE = auto()
    INTERACT = auto()
    NAVIGATE_MENU = auto()
    WAIT_REOBSERVE = auto()
    NAVIGATE_TOWARD_GOAL = auto()
    DELEGATE_BATTLE = auto()


@dataclass(frozen=True)
class AgentAction:
    action_type: AgentActionType
    option: str | None = None
    direction: Direction | None = None
    navigation: NavigationAction | None = None
    reason: str = ""


class GoalStatus(Enum):
    COMPLETE = auto()
    REACHABLE = auto()
    UNREACHABLE = auto()
    NOT_APPLICABLE = auto()


@dataclass(frozen=True)
class GoalEvaluation:
    status: GoalStatus
    plan: NavigationPlan | None = None
    reason: str = ""
    diagnostics: tuple[str, ...] = ()
    binding_diagnostics: tuple[str, ...] = ()
    world_diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentObservation:
    interaction: InteractionObservation
    overworld: OverworldObservation | None = None
    goal: Goal | None = None

    @property
    def interaction_type(self) -> InteractionType:
        return classify_interaction(self.interaction)


def prewarm_warp_destination(
    observation: AgentObservation,
    navigation: NavigationAction | None,
) -> bool:
    """Warm static caches for the observed destination of an imminent warp."""
    if (
        navigation is None
        or navigation.action_type is not NavigationActionType.WARP
        or observation.overworld is None
        or navigation.source[0] != observation.overworld.map_id
        or not any(
            transition.destination == navigation.destination
            for transition in (observation.overworld.transitions or observation.overworld.warps)
        )
    ):
        return False
    trace = getattr(context, "stutter_trace", None)
    trace_start = trace.now() if trace is not None else 0
    try:
        tiles = prewarm_static_map_observation(navigation.destination[0])
        prewarm_navigation_tiles(navigation.destination[0], tiles)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        # Cache preparation is an optimization and must never prevent the
        # existing warp dispatch from running.
        if trace is not None:
            trace.mark("warp_destination_prewarm_failed", True)
            trace.duration("warp_destination_prewarm_duration_ms", trace_start)
        return False
    if trace is not None:
        trace.mark("warp_destination_prewarm_used", True)
        trace.duration("warp_destination_prewarm_duration_ms", trace_start)
    return True


@profiled("agent_observation", "agent_observations")
def observe_agent(
    *,
    goal: Goal | None = None,
    choice_options: tuple[str, ...] = (),
    menu_options: tuple[str, ...] = (),
    special_interaction: str | None = None,
    require_overworld: bool = False,
    overworld_observation: OverworldObservation | None = None,
) -> AgentObservation:
    """Create one live observation.

    ``require_overworld`` is for callers waiting on a physical map change.
    Such callers must not let a transient UI/game-state classification decide
    whether the map is observed.
    """
    force_overworld_reobserve = _force_universal_reobserve()
    trace = getattr(context, "stutter_trace", None)
    span = trace.span("observation") if trace is not None else nullcontext()
    with span:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PLAN_TRACE: phase=observe_agent_begin "
                f"frame={getattr(context, 'frame', None)!r} goal={goal!r}"
            ),
            trace=True,
        )
        result = _observe_agent_instrumented(
            goal=goal,
            choice_options=choice_options,
            menu_options=menu_options,
            special_interaction=special_interaction,
            force_overworld_reobserve=force_overworld_reobserve,
            require_overworld=require_overworld,
            overworld_observation=overworld_observation,
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PLAN_TRACE: phase=observe_agent_end "
                f"frame={getattr(context, 'frame', None)!r} "
                f"map={result.overworld.map_id if result.overworld else None!r} "
                f"coordinates={result.overworld.player_coordinates if result.overworld else None!r} "
                f"facing={result.overworld.facing if result.overworld else None!r} goal={result.goal!r}"
                f" movement={getattr(getattr(result, 'overworld', None).movement_state, 'name', None) if result.overworld else None!r}"
                f" blocked={getattr(result.overworld, 'dynamic_blocked_coordinates', None) if result.overworld else None!r}"
                f" objects={getattr(result.overworld, 'objects', None) if result.overworld else None!r}"
            ),
            trace=True,
        )
        return result


def _observe_agent_instrumented(
    *,
    goal: Goal | None = None,
    choice_options: tuple[str, ...] = (),
    menu_options: tuple[str, ...] = (),
    special_interaction: str | None = None,
    force_overworld_reobserve: bool = False,
    require_overworld: bool = False,
    overworld_observation: OverworldObservation | None = None,
) -> AgentObservation:
    interaction_start = now()
    interaction = observe_interaction(
        choice_options=choice_options,
        menu_options=menu_options,
        special_interaction=special_interaction,
    )
    timing("agent_interaction_stage", interaction_start)
    count("agent_interaction_stages")
    classification_start = now()
    interaction_type = classify_interaction(interaction)
    timing("agent_interaction_classification", classification_start)
    count("agent_interaction_classifications")
    trace = getattr(context, "stutter_trace", None)
    perception_trace_start = trace.now() if trace is not None else 0
    perception_start = now()
    # A settled warp can briefly retain a non-overworld callback.  The
    # frame-based watchdog makes one bounded raw map/avatar read in that
    # state, so recovery can hand off from an outdoor door tile to the Center
    # interior even if normal classification has not caught up yet.  Menus,
    # battles, and scripted scenes remain excluded from this fallback.
    transient_map_state = interaction.game_state in (GameState.UNKNOWN, GameState.CHANGE_MAP)
    should_perceive_overworld = (
        require_overworld
        or interaction_type is InteractionType.OVERWORLD
        or (force_overworld_reobserve and transient_map_state)
    )
    if overworld_observation is not None:
        overworld = overworld_observation
    elif should_perceive_overworld:
        overworld = perceive_overworld()
    else:
        overworld = None
    timing("agent_overworld_perception", perception_start)
    count("agent_overworld_perceptions")
    if trace is not None:
        trace.mark("full_perception_ran", overworld is not None)
        if overworld is not None:
            trace.mark("map_id", overworld.map_id)
            trace.mark("player_position", overworld.player_coordinates)
        if trace.enabled:
            trace.mark("perception_duration_ms", round((trace.now() - perception_trace_start) / 1_000_000, 3))
    finalize_start = now()
    observation = AgentObservation(interaction=interaction, overworld=overworld, goal=goal)
    _publish_recovery_handoff_observation(observation)
    timing("agent_observation_finalize", finalize_start)
    count("agent_observation_finalizations")
    return observation


def _publish_recovery_handoff_observation(observation: AgentObservation) -> None:
    """Expose warp progress in the GUI while the route loop still owns it.

    A recovery route can physically cross a map boundary before its
    ``ReachLocation`` loop receives the observation that completes the goal.
    Publishing here makes that distinction visible without enabling tracing
    or changing navigation policy.
    """
    from modules.goals import ReachLocation, SemanticTarget
    from modules.nuzlocke.campaign_status import recovery_status

    if not isinstance(observation.goal, ReachLocation) or observation.overworld is None:
        return
    controller = getattr(getattr(context, "bot_mode_instance", None), "controller", None)
    if getattr(controller, "_execution_phase", None) != "RECOVERY":
        return
    status = getattr(context, "campaign_status", None)
    if getattr(status, "objective", None) != "Recover Party":
        return
    destination = observation.goal.location
    observed_map = observation.overworld.map_id
    if observed_map == getattr(destination[0], "value", destination[0]):
        context.campaign_status = recovery_status(
            SemanticTarget.at(destination),
            f"Waiting for Pokémon Center map (observed {observed_map!r} at "
            f"{observation.overworld.player_coordinates!r})",
        )
        return
    try:
        from modules.nuzlocke.emerald_healing_catalog import emerald_healing_source_for_destination

        source = emerald_healing_source_for_destination(destination)
        interior_map = source.interior_map if source is not None else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        interior_map = None
    interior_map_id = getattr(interior_map, "value", interior_map)
    if interior_map_id is not None and observed_map == interior_map_id:
        intent = (
            "Destination map observed; waiting for recovery handoff"
            if observation.overworld.controllable
            else "Waiting for avatar control"
        )
        context.campaign_status = recovery_status(
            SemanticTarget.map(interior_map_id),
            intent,
        )


@dataclass(frozen=True)
class ActionDecision:
    action: AgentAction
    goal_evaluation: GoalEvaluation | None = None


def _dialogue_actions(observation: AgentObservation) -> tuple[AgentAction, ...]:
    if observation.interaction.field_message_render_rescue_available:
        return (
            AgentAction(
                AgentActionType.ACCELERATE_DIALOGUE_RENDER,
                reason="fresh-B render rescue with held B",
            ),
        )
    if observation.interaction.interaction_phase is InteractionPhase.FIELD_MESSAGE_INPUT_WAIT:
        return (AgentAction(AgentActionType.ADVANCE_DIALOGUE, reason="dialogue is waiting for input"),)
    return (AgentAction(AgentActionType.WAIT_REOBSERVE, reason="dialogue is not ready"),)


def _choice_actions(observation: AgentObservation) -> tuple[AgentAction, ...]:
    if observation.interaction.choice_options:
        return tuple(
            AgentAction(AgentActionType.CHOOSE_DIALOGUE_OPTION, option=option)
            for option in observation.interaction.choice_options
        )
    return (AgentAction(AgentActionType.WAIT_REOBSERVE, reason="choice options are not available yet"),)


def _menu_actions(observation: AgentObservation) -> tuple[AgentAction, ...]:
    if observation.interaction.menu_options:
        return tuple(
            AgentAction(AgentActionType.NAVIGATE_MENU, option=option) for option in observation.interaction.menu_options
        )
    return (AgentAction(AgentActionType.WAIT_REOBSERVE, reason="menu options are not available yet"),)


def _battle_actions(_: AgentObservation) -> tuple[AgentAction, ...]:
    return (AgentAction(AgentActionType.DELEGATE_BATTLE, reason="BattleListener owns battle execution"),)


def _overworld_actions(observation: AgentObservation) -> tuple[AgentAction, ...]:
    if observation.overworld is None or not observation.overworld.controllable:
        return (AgentAction(AgentActionType.WAIT_REOBSERVE, reason="player avatar is not controllable"),)
    world = NavigationWorld.from_overworld(observation.overworld)
    start = (observation.overworld.map_id, observation.overworld.player_coordinates)
    world_diagnostics: tuple[str, ...] = ()
    actions: list[AgentAction] = []
    for direction, destination, is_warp in world.neighbors(start):
        if not is_warp:
            actions.append(AgentAction(AgentActionType.MOVE, direction=direction))
    for trigger in world.triggers:
        if start in trigger.activation_locations:
            actions.append(AgentAction(AgentActionType.INTERACT, option=trigger.trigger_id))
    return tuple(actions) or (AgentAction(AgentActionType.WAIT_REOBSERVE, reason="no legal overworld action"),)


def _safe_wait_actions(_: AgentObservation) -> tuple[AgentAction, ...]:
    return (AgentAction(AgentActionType.WAIT_REOBSERVE, reason="interaction state is not safely actionable"),)


def _goal_target(goal: Goal | None) -> Goal | None:
    """Return the semantic target when a goal carries navigation policy."""
    return goal.target if isinstance(goal, NavigationGoal) else goal


def _observed_trigger(observation: AgentObservation, trigger_id: str):
    """Return the live affordance for a trigger goal, when it is observed."""
    if observation.overworld is None:
        return None
    return next(
        (item for item in observation.overworld.triggers if item.trigger_id == trigger_id),
        None,
    )


def _trigger_condition_completed(observation: AgentObservation, trigger_id: str) -> bool:
    """Treat a ROM-gated trigger becoming inactive as completion.

    Coordinate scripts are dispatched by tile entry and usually turn their
    event variable off as their final step.  The trigger can therefore remain
    geometrically present after the scene has completed.  ``False`` is
    meaningful here; ``None`` still means that live condition state was not
    observed and must not be interpreted as completion.
    """
    trigger = _observed_trigger(observation, trigger_id)
    return trigger is not None and trigger.condition_active is False


def _trigger_requires_input(observation: AgentObservation, trigger_id: str) -> bool:
    """Return whether an observed trigger needs an interaction-button edge."""
    trigger = _observed_trigger(observation, trigger_id)
    return trigger is None or getattr(trigger, "requires_input", True)


def available_actions(observation: AgentObservation) -> tuple[AgentAction, ...]:
    """Return legal actions using a handler dedicated to the interaction state."""

    handlers = {
        InteractionType.DIALOGUE: _dialogue_actions,
        InteractionType.CHOICE: _choice_actions,
        InteractionType.MENU: _menu_actions,
        InteractionType.BATTLE: _battle_actions,
        InteractionType.OVERWORLD: _overworld_actions,
    }
    return handlers.get(observation.interaction_type, _safe_wait_actions)(observation)


@profiled("agent_goal_evaluation", "goal_evaluations")
def evaluate_goal(observation: AgentObservation) -> GoalEvaluation:
    """Evaluate a tactical goal against the current overworld observation."""

    trace = getattr(context, "stutter_trace", None)
    span = trace.span("goal_evaluation") if trace is not None else nullcontext()
    with span:
        return _evaluate_goal_instrumented(observation)


def _evaluate_goal_instrumented(observation: AgentObservation) -> GoalEvaluation:
    if observation.goal is None:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="no goal supplied")
    if observation.overworld is None or observation.interaction_type is not InteractionType.OVERWORLD:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="goal requires an overworld observation")

    goal_target = _goal_target(observation.goal)
    if isinstance(goal_target, ActivateTrigger) and _trigger_condition_completed(observation, goal_target.trigger_id):
        return GoalEvaluation(
            GoalStatus.COMPLETE,
            reason="trigger condition is inactive; ROM scene already completed",
        )

    # Diagnostics only: this deliberately does not gate, score, or alter the
    # navigation decision.  Keep the snapshot at the same overworld decision
    # boundary as the active goal evaluation so later telemetry can explain
    # exactly what was available before a trainer approach.
    try:
        from modules.nuzlocke.snapshots import get_nuzlocke_snapshot

        snapshot = get_nuzlocke_snapshot()
        controller = getattr(getattr(context, "bot_mode_instance", None), "controller", None)
        selection = getattr(controller, "last_selection", None)
        objective = getattr(selection, "objective", None)
        status = getattr(selection, "status", None)
        diagnostic = build_progression_readiness_diagnostic(
            snapshot,
            objective_id=getattr(objective, "objective_id", None),
            objective_status=getattr(status, "value", None),
            destination=getattr(objective, "destination", None),
            navigation_goal=observation.goal,
            campaign_mode=str(context.bot_mode),
            overworld=observation.overworld,
        )
        diagnostic = evaluate_progression_readiness(diagnostic)
        diagnostic_print(
            lambda: "CAMPAIGN_READINESS: " + json.dumps(diagnostic.as_dict(), sort_keys=True, default=str), trace=True
        )
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
        # Diagnostics must never make navigation unavailable.
        pass

    world_start = now()
    trace_world_start = getattr(context, "stutter_trace", None)
    trace_world_start_ns = trace_world_start.now() if trace_world_start is not None else 0
    world = NavigationWorld.from_overworld(observation.overworld)
    if trace_world_start is not None:
        trace_world_start.duration("navigation_world_construction_duration_ms", trace_world_start_ns)
    timing("goal_world_construction", world_start)
    count("goal_world_constructions")
    start = (observation.overworld.map_id, observation.overworld.player_coordinates)
    if isinstance(goal_target, ReachWarp):
        destination_map = goal_target.destination_map
        if destination_map is None and goal_target.destination is not None:
            destination_map = goal_target.destination[0]
        if destination_map == observation.overworld.map_id:
            # Recovery mounts a ReachWarp until the post-warp handoff is
            # observed. Once the destination map is visible, the transition
            # goal is complete even if the old warp record is still present
            # in the freshly observed world.
            return GoalEvaluation(
                GoalStatus.COMPLETE,
                reason="destination map reached",
            )
    # A map warp can expose the destination header before the avatar has a
    # valid local coordinate.  Treat that as a settling observation rather
    # than asking the world planner to search from a node that is absent from
    # the observed topology.  This matters especially to recovery, whose
    # ReachLocation loop is mounted immediately after a campaign warp.
    if start not in world.tiles:
        return GoalEvaluation(
            GoalStatus.UNREACHABLE,
            reason="avatar coordinate is not present in current overworld topology; waiting for transition to settle",
            diagnostics=navigation_diagnostics(world, start, observation.goal),
            world_diagnostics=("avatar coordinate is outside the observed map topology",),
        )
    diagnostics_enabled = bool(context.debug and getattr(context, "debug_trace", False))
    binding_diagnostics = ()
    world_diagnostics = ()
    if diagnostics_enabled and isinstance(_goal_target(observation.goal), (ActivateTrigger, ReachInteractionPosition)):
        resolution = next(
            (
                binding
                for binding in observation.overworld.bindings
                if binding.binding.trigger_id == _goal_target(observation.goal).trigger_id
            ),
            None,
        )
        if resolution is not None:
            binding_diagnostics = (
                f"trigger={resolution.binding.trigger_id!r}"
                f" map={resolution.binding.map_id!r}"
                f" script={resolution.binding.script_symbol!r}",
                f"runtime_match={resolution.runtime_match}" f" object_id={resolution.object_ids!r}",
                f"static_match={resolution.static_match}"
                f" static_available={resolution.static_available}"
                f" static_ambiguous={resolution.static_ambiguous}"
                f" static_location={resolution.static_location!r}",
                f"target_positions={resolution.interaction_positions!r}",
            )
    planning_start = now()
    diagnostic_print(
        lambda: (
            "CAMPAIGN_PLAN_TRACE: phase=evaluate_goal_begin "
            f"frame={getattr(context, 'frame', None)!r} start={start!r} goal={observation.goal!r}"
        ),
        trace=True,
    )
    trace_planning_start = trace_world_start.now() if trace_world_start is not None else 0
    try:
        diagnostic_print(lambda: "CAMPAIGN_PLAN_TRACE: phase=world_plan_begin", trace=True)
        plan, world_route = plan_with_world_navigation(world, start, observation.goal)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PLAN_TRACE: phase=world_plan_end "
                f"actions={len(plan.actions)} destination={plan.destination!r} "
                f"first={plan.actions[0] if plan.actions else None!r} "
                f"last={plan.actions[-1] if plan.actions else None!r}"
            ),
            trace=True,
        )
        target_map = goal_target_map(world, observation.goal)
        if (
            isinstance(observation.goal, ReachWarp)
            and observation.goal.warp is not None
            and target_map != start[0]
            and world_route is None
        ):
            world_diagnostics = ("observed transition planned locally; global route not consulted",)
        if diagnostics_enabled and world_route is not None and target_map is not None:
            world_diagnostics = (
                f"current_map={start[0]!r} target_map={target_map!r}",
                f"maps={world_route.maps!r}",
                f"edges={tuple((edge.kind, edge.source_map, edge.destination_map,
                                edge.source_coordinates) for edge in world_route.edges)!r}",
            )
        if diagnostics_enabled and plan.metrics is not None:
            world_diagnostics += (f"route_selected: {plan.metrics.summary()}",)
            if plan.candidate_metrics:
                world_diagnostics += (
                    "route_candidates: " + " | ".join(candidate.summary() for candidate in plan.candidate_metrics),
                )
    except (NavigationError, WorldNavigationError) as error:
        # A selected WarpObservation is direct runtime evidence of a local
        # exit.  If the world graph cannot prove a cross-map route, ask the
        # local navigator to prove only the observed movement/activation path.
        if isinstance(observation.goal, ReachWarp) and observation.goal.warp is not None:
            try:
                local_plan = plan_observed_warp_locally(world, start, observation.goal)
            except NavigationError:
                pass
            else:
                return GoalEvaluation(
                    GoalStatus.REACHABLE if local_plan.actions else GoalStatus.COMPLETE,
                    plan=local_plan,
                    reason="locally actionable observed warp; global route unavailable",
                    diagnostics=navigation_diagnostics(world, start, observation.goal),
                    binding_diagnostics=binding_diagnostics,
                    world_diagnostics=("global route unavailable; using observed local transition",),
                )
        timing("goal_planning", planning_start)
        count("goal_planning_attempts")
        return GoalEvaluation(
            GoalStatus.UNREACHABLE,
            reason=str(error),
            diagnostics=navigation_diagnostics(world, start, observation.goal),
            binding_diagnostics=binding_diagnostics,
            world_diagnostics=world_diagnostics,
        )
    timing("goal_planning", planning_start)
    if trace_world_start is not None:
        trace_world_start.duration("navigation_planning_duration_ms", trace_planning_start)
    count("goal_planning_attempts")

    if (
        isinstance(goal_target, (SemanticTarget, ReachLocation, ReachWarp, ReachInteractionPosition))
        and not plan.actions
    ):
        if isinstance(observation.goal, ReachWarp) and observation.goal.warp is not None:
            # A ReachLocation handoff may leave us standing on a step-on warp
            # entry.  Do not interpret that source tile as having crossed the
            # transition; recover an explicit local activation action.
            try:
                local_plan = plan_observed_warp_locally(world, start, observation.goal)
            except NavigationError:
                local_plan = None
            if local_plan is not None and local_plan.actions:
                return GoalEvaluation(
                    GoalStatus.REACHABLE,
                    plan=local_plan,
                    reason="warp entry observed; activation still required",
                    binding_diagnostics=binding_diagnostics,
                    world_diagnostics=world_diagnostics,
                )
        return GoalEvaluation(
            GoalStatus.COMPLETE,
            plan=plan,
            reason="goal position reached",
            binding_diagnostics=binding_diagnostics,
            world_diagnostics=world_diagnostics,
        )
    if isinstance(goal_target, EngageTrainer):
        trainer = next((obj for obj in observation.overworld.objects if obj.trainer_id == goal_target.trainer_id), None)
        if trainer is not None and trainer.trainer_defeated is True:
            return GoalEvaluation(
                GoalStatus.COMPLETE,
                plan=plan,
                reason="engaged trainer is defeated",
                world_diagnostics=(f"trainer_id={goal_target.trainer_id!r}", "trainer_defeated=True"),
            )
    if isinstance(goal_target, ActivateTrigger):
        activated = observation.interaction.metadata.get("activated_trigger_ids", ())
        if goal_target.trigger_id in activated:
            return GoalEvaluation(
                GoalStatus.COMPLETE,
                plan=plan,
                reason="trigger is already activated",
                binding_diagnostics=binding_diagnostics,
                world_diagnostics=world_diagnostics,
            )
    return GoalEvaluation(
        GoalStatus.REACHABLE,
        plan=plan,
        binding_diagnostics=binding_diagnostics,
        world_diagnostics=world_diagnostics,
    )


@profiled("agent_action_selection", "actions_selected")
def select_action(observation: AgentObservation) -> ActionDecision:
    """Select one deterministic action after state-specific affordance generation."""

    trace = getattr(context, "stutter_trace", None)
    span = trace.span("action_selection") if trace is not None else nullcontext()
    with span:
        return _select_action_instrumented(observation)


def _select_action_instrumented(observation: AgentObservation) -> ActionDecision:
    interaction_type = observation.interaction_type
    if interaction_type is not InteractionType.OVERWORLD:
        decision = ActionDecision(available_actions(observation)[0])
        trace = getattr(context, "stutter_trace", None)
        if trace is not None:
            trace.mark("selected_agent_action", decision.action.action_type.name)
            trace.mark("selected_agent_option", decision.action.option)
        return decision
    # A semantic trigger can remain geometrically reachable while its script
    # owns the avatar.  Do not send the goal's interaction input again until
    # the script has returned control to the player.
    if not observation.interaction.controllable:
        if observation.interaction.interaction_phase is InteractionPhase.FIELD_MESSAGE_INPUT_WAIT:
            return ActionDecision(
                AgentAction(
                    AgentActionType.ADVANCE_DIALOGUE,
                    reason="field dialogue transition requires input",
                )
            )
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason="player avatar is not controllable"))
    evaluation = evaluate_goal(observation)
    if evaluation.status is GoalStatus.COMPLETE:
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation)
    if evaluation.status is GoalStatus.UNREACHABLE:
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation)
    if evaluation.status is GoalStatus.NOT_APPLICABLE:
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation)
    if (
        isinstance(_goal_target(observation.goal), ActivateTrigger)
        and evaluation.status is GoalStatus.REACHABLE
        and evaluation.plan is not None
        and not evaluation.plan.actions
    ):
        if not _trigger_requires_input(observation, _goal_target(observation.goal).trigger_id):
            return ActionDecision(
                AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="automatic coordinate trigger fires on tile entry",
                ),
                evaluation,
            )
        return ActionDecision(
            AgentAction(
                AgentActionType.INTERACT,
                option=_goal_target(observation.goal).trigger_id,
                reason="goal activation position reached",
            ),
            evaluation,
        )
    if evaluation.status is GoalStatus.REACHABLE and evaluation.plan and evaluation.plan.actions:
        return ActionDecision(
            AgentAction(
                AgentActionType.NAVIGATE_TOWARD_GOAL,
                direction=evaluation.plan.actions[0].direction,
                navigation=evaluation.plan.actions[0],
                reason="first action in deterministic goal plan",
            ),
            evaluation,
        )
    return ActionDecision(available_actions(observation)[0], evaluation)


class ActionResultType(Enum):
    EXECUTED = auto()
    WAITING = auto()
    GOAL_COMPLETE = auto()
    UNREACHABLE = auto()
    DELEGATED = auto()
    UNSUPPORTED = auto()


@dataclass(frozen=True)
class ActionResult:
    result_type: ActionResultType
    action: AgentAction
    message: str = ""


@dataclass
class _PendingTransition:
    """Execution state for a selected cross-map action.

    This is deliberately controller-local.  Planning still chooses the
    action, but while the ROM is resolving it that choice remains
    authoritative and is not fed back through the planner each frame.
    """

    action: NavigationAction
    source: tuple[tuple[int, int], tuple[int, int]]
    activation_position: tuple[tuple[int, int], tuple[int, int]]
    direction: Direction
    destination: tuple[tuple[int, int], tuple[int, int]] | None
    transition_kind: str
    observations: int = 0
    moved: bool = False
    input_issued: bool = False
    last_position: tuple[tuple[int, int], tuple[int, int]] | None = None
    settling_observations: int = 0


@dataclass
class _MovementBatch:
    """A bounded suffix of a cached plan being driven frame-by-frame.

    This deliberately contains navigation actions, rather than raw buttons.  A
    warp (and any action after it) can therefore never be sent through this
    fast path.
    """

    actions: tuple[NavigationAction, ...]
    index: int = 0
    frames_waiting: int = 0

    @property
    def current(self) -> NavigationAction:
        return self.actions[self.index]


class AgentActionExecutor:
    """Translate semantic actions to input only after selection is complete."""

    def __init__(
        self, *, choose_option: Callable[[str], None] | None = None, navigate_menu: Callable[[str], None] | None = None
    ):
        self._choose_option = choose_option
        self._navigate_menu = navigate_menu
        self._field_message_render_b_held = False

    def _release_field_message_render_b(self) -> None:
        if not self._field_message_render_b_held:
            return
        release_button = getattr(context.emulator, "release_button", None)
        if callable(release_button):
            release_button("B")
        self._field_message_render_b_held = False

    @profiled("agent_action_execution", "actions_executed")
    def execute(self, action: AgentAction, observation: AgentObservation) -> ActionResult:
        trace = getattr(context, "stutter_trace", None)
        span = trace.span("action_construction_and_execution") if trace is not None else nullcontext()
        with span:
            return self._execute_instrumented(action, observation)

    def _execute_instrumented(self, action: AgentAction, observation: AgentObservation) -> ActionResult:
        diagnostic_print(
            lambda: (
                "CAMPAIGN_INTERACTION_TRACE: "
                f"phase=execute frame={getattr(context, 'frame', None)!r} "
                f"map={observation.overworld.map_id if observation.overworld else None!r} "
                f"coordinates={observation.overworld.player_coordinates if observation.overworld else None!r} "
                f"facing={observation.overworld.facing if observation.overworld else None!r} "
                f"player_elevation={getattr(observation.overworld, 'player_elevation', None) if observation.overworld else None!r} "
                f"player_current_elevation={getattr(observation.overworld, 'player_current_elevation', None) if observation.overworld else None!r} "
                f"player_previous_coordinates={getattr(observation.overworld, 'player_previous_coordinates', None) if observation.overworld else None!r} "
                f"action={action.action_type.name!r} option={action.option!r} "
                f"direction={getattr(action.direction, 'name', None)!r} navigation={action.navigation!r} "
                f"goal={observation.goal!r} interaction={observation.interaction!r}"
            ),
            trace=True,
        )
        if action.action_type is not AgentActionType.ACCELERATE_DIALOGUE_RENDER:
            # The render workaround holds B so the printer keeps its
            # speed-up flag active. Release it before any later dialogue,
            # choice, or overworld action can observe the handoff.
            render_b_was_owned = self._field_message_render_b_held
            self._release_field_message_render_b()
            if not render_b_was_owned and observation.interaction.field_message_lifecycle_active:
                release_button = getattr(context.emulator, "release_button", None)
                if callable(release_button):
                    # A recovery path may construct a short-lived executor
                    # for each observation. In that case the ownership bit
                    # is gone, but the emulator can still be holding the B
                    # installed by the preceding render action.
                    release_button("B")
        if action.action_type is AgentActionType.WAIT_REOBSERVE:
            return ActionResult(ActionResultType.WAITING, action, action.reason)
        if action.action_type is AgentActionType.DELEGATE_BATTLE:
            return ActionResult(ActionResultType.DELEGATED, action, action.reason)
        if observation.interaction_type in (InteractionType.UNKNOWN, InteractionType.SPECIAL_INTERACTION):
            return ActionResult(
                ActionResultType.WAITING,
                AgentAction(AgentActionType.WAIT_REOBSERVE, reason="unsafe interaction state"),
                "unsafe interaction state",
            )
        if context.emulator is None and action.action_type not in (
            AgentActionType.CHOOSE_DIALOGUE_OPTION,
            AgentActionType.NAVIGATE_MENU,
        ):
            return ActionResult(ActionResultType.UNSUPPORTED, action, "no emulator is attached")

        if action.action_type is AgentActionType.ACCELERATE_DIALOGUE_RENDER:
            if (
                observation.interaction.interaction_phase is not InteractionPhase.FIELD_MESSAGE_RENDER_WAIT
                or not observation.interaction.field_message_render_rescue_available
            ):
                return ActionResult(ActionResultType.WAITING, action, "field message render rescue is not available")
            is_button_held = getattr(context.emulator, "is_button_held", None)
            b_is_held = callable(is_button_held) and is_button_held("B") is True
            hold_button = getattr(context.emulator, "hold_button", None)
            operation = "hold_button(B)"
            if not b_is_held:
                press_button_fresh = getattr(type(context.emulator), "press_button_fresh", None)
                if callable(press_button_fresh):
                    context.emulator.press_button_fresh("B")
                else:
                    context.emulator.press_button("B")
                operation = "press_button_fresh(B)+hold_button(B)"
            if callable(hold_button):
                # One fresh edge enables Emerald's print-speed-up path; the
                # held B keeps it active until the printer reaches its ROM
                # backed input boundary. This remains safe for overworld
                # interaction because B is released before the next action.
                hold_button("B")
                self._field_message_render_b_held = True
            diagnostic_print(
                lambda: (
                    "AGENT_DIALOGUE_RENDER_RESCUE: action=ACCELERATE_DIALOGUE_RENDER "
                    f"pulses={observation.interaction.metadata.get('field_message_render_rescue_pulses')!r} "
                    f"operation={operation}"
                ),
                trace=True,
            )
        elif action.action_type is AgentActionType.ADVANCE_DIALOGUE:
            if observation.interaction.interaction_phase is not InteractionPhase.FIELD_MESSAGE_INPUT_WAIT:
                return ActionResult(ActionResultType.WAITING, action, "dialogue is not ready")
            diagnostic_print(
                lambda: (
                    "AGENT_DIALOGUE_INPUT: action=ADVANCE_DIALOGUE "
                    f"dialogue_waiting={observation.interaction.dialogue_waiting!r} "
                    f"field_message_advance_ready={observation.interaction.field_message_advance_ready!r} "
                    "operation=press_button(A)"
                ),
                trace=True,
            )
            # Dialogue dismissal must be a fresh input.  Otherwise the A
            # edge can survive the script-to-overworld handoff and be
            # consumed as a new object interaction on the first controllable
            # frame.
            press_button_fresh = getattr(type(context.emulator), "press_button_fresh", None)
            if callable(press_button_fresh):
                context.emulator.press_button_fresh("A")
            else:
                context.emulator.press_button("A")
        elif action.action_type is AgentActionType.CHOOSE_DIALOGUE_OPTION:
            if action.option not in observation.interaction.choice_options:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "choice is not available")
            if self._choose_option is not None:
                self._choose_option(action.option)
            elif observation.interaction.choice_menu_active and observation.interaction.choice_menu_input_ready:
                if action.option == "YES" and observation.interaction.choice_selected == "NO":
                    context.emulator.press_button_fresh("UP")
                else:
                    context.emulator.press_button_fresh("A")
            else:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "no choice controller is attached")
        elif action.action_type is AgentActionType.NAVIGATE_MENU:
            if action.option not in observation.interaction.menu_options:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "menu option is not available")
            if self._navigate_menu is None:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "no menu controller is attached")
            self._navigate_menu(action.option)
        elif action.action_type in (AgentActionType.MOVE, AgentActionType.NAVIGATE_TOWARD_GOAL):
            if action.direction is None:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "movement direction is missing")
            press_direction = getattr(context.emulator, "press_direction", None)
            press_button_fresh = getattr(type(context.emulator), "press_button_fresh", None)
            if callable(press_direction):
                press_direction(
                    action.direction.button_name,
                    run=bool(action.navigation is not None and action.navigation.run),
                    fresh=bool(
                        action.navigation is not None and action.navigation.action_type is NavigationActionType.WARP
                    ),
                )
                return ActionResult(ActionResultType.EXECUTED, action)
            if (
                action.navigation is not None
                and action.navigation.action_type is NavigationActionType.WARP
                and callable(press_button_fresh)
            ):
                context.emulator.press_button_fresh(action.direction.button_name)
            else:
                context.emulator.press_button(action.direction.button_name)
        elif action.action_type is AgentActionType.INTERACT:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_INTERACTION_TRACE: "
                    f"phase=press_input frame={getattr(context, 'frame', None)!r} "
                    f"button='A' trigger={action.option!r}"
                ),
                trace=True,
            )
            context.emulator.press_button("A")
        return ActionResult(ActionResultType.EXECUTED, action)


class AgentControlLoop:
    """A generator-compatible observe/select/execute loop for future mounting."""

    # Runtime structures can remain cached across a warp or while a field
    # message is being installed.  Keep the fast path bounded independently
    # of debug/tracing overhead so those transitions eventually get a full
    # perception pass in every mode.
    _periodic_reobserve_after = 90

    def __init__(
        self,
        observe: Callable[[], AgentObservation],
        executor: AgentActionExecutor | None = None,
        *,
        goal: Goal | None = None,
        navigation_plan: NavigationPlan | None = None,
        logger: Callable[[str], None] | None = None,
    ):
        self._observe = observe
        self._executor = executor or AgentActionExecutor()
        self._goal = goal
        # CampaignPlan may already have paid the cost of composing a route
        # through a recovery source.  Keep that route as an execution input;
        # the first stable observation seeds the normal route/checkpoint
        # machinery without running a second planner search.
        self._initial_navigation_plan = navigation_plan
        self._custom_logger = logger is not None
        self._logger = logger or (lambda message: diagnostic_print(message, trace=True))
        self._last_interaction_type: InteractionType | None = None
        self._battle_was_active = False
        self._last_navigation_diagnostics: tuple[str, ...] | None = None
        self._last_binding_diagnostics: tuple[str, ...] | None = None
        self._last_world_diagnostics: tuple[str, ...] | None = None
        self._expected_world_transition: (
            tuple[
                tuple[tuple[int, int], tuple[int, int]],
                tuple[tuple[int, int], tuple[int, int]],
            ]
            | None
        ) = None
        self._pending_transition: _PendingTransition | None = None
        self._warp_settling = False
        self._warp_wait_observations = 0
        self._pending_transition_watchdog_limit = 60
        self._blocked_warp: WarpObservation | None = None
        self._cached_evaluation: GoalEvaluation | None = None
        self._route_plan: RoutePlan | None = None
        self._cached_actions: tuple[NavigationAction, ...] = ()
        self._cached_action_index = 0
        self._in_flight_move: NavigationAction | None = None
        self._in_flight_move_initial_facing: Direction | None = None
        self._cached_goal: Goal | None = None
        self._cached_world_signature: tuple | None = None
        self._route103_stable_observations = 0
        self._route103_object_dumped_at: set[int] = set()
        self._last_world_transition_source: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._movement_batch: _MovementBatch | None = None
        self._fast_path_frames = 0
        self._last_observation_frame: int | None = None
        self._movement_blocked_retries = 0
        # Counts failed movement attempts for this mounted goal.  Unlike the
        # one-shot transient retry above, this survives route invalidation so
        # an unchanged plan cannot reset the failure budget.
        self._movement_failure_count = 0
        self._movement_failure_limit = 4
        self._dialogue_input_in_flight = False
        self._started_interaction_id: str | None = None
        self._interaction_start_waits = 0
        self._last_observed_location: Location | None = None
        self._previous_observed_location: Location | None = None

    def _seed_navigation_plan(self, observation: AgentObservation) -> None:
        """Install a planner-supplied route once its origin is observable."""

        plan = self._initial_navigation_plan
        if plan is None or self._cached_evaluation is not None or observation.overworld is None:
            return
        self._cached_evaluation = GoalEvaluation(
            GoalStatus.REACHABLE,
            plan,
            reason="campaign planner supplied route",
        )
        self._cached_actions = plan.actions
        self._cached_action_index = 0
        self._route_plan = RoutePlan.from_navigation_plan(plan, target=self._goal)
        self._cached_goal = self._goal
        self._cached_world_signature = self._world_signature(observation)
        self._initial_navigation_plan = None
        self._report(
            f"ROUTE_PLAN_ADOPTED route_id={self._route_plan.route_id} "
            f"actions={len(self._route_plan.actions)} source='campaign_planner'"
        )

    def _implicit_transition_boundary(self, observation: AgentObservation) -> bool:
        """Recognize a transition exposed between controller observations."""
        world = observation.overworld
        if world is None or self._previous_observed_location is None:
            return False
        current = (world.map_id, world.player_coordinates)
        previous = self._previous_observed_location
        if current == previous or current[0] != previous[0]:
            return False
        if abs(current[1][0] - previous[1][0]) + abs(current[1][1] - previous[1][1]) <= 1:
            return False
        return any(warp.destination == current for warp in world.warps if warp.destination is not None)

    def _safe_movement_batch(self, observation: AgentObservation) -> tuple[NavigationAction, ...]:
        """Return a short, same-map movement segment from the cached plan.

        Checkpoints are the end of the plan, a warp, and any tile known to be
        meaningful to a trigger/object interaction.  The bound also ensures
        that dynamic assumptions are periodically handed back to perception.
        """
        if observation.overworld is None or self._cached_action_index >= len(self._cached_actions):
            return ()
        world = observation.overworld
        trainer_avoidance_enabled = (
            getattr(getattr(self._goal, "constraints", None), "trainer_mode", None) is not None
            and getattr(self._goal.constraints.trainer_mode, "name", None) == "AVOID"
        )
        engagement_trainer_id = self._goal.trainer_id if isinstance(self._goal, EngageTrainer) else None
        checkpoints = {
            location
            for trigger in world.triggers
            for locations in (
                trigger.locations,
                trigger.activation_locations,
                trigger.navigation_locations,
                *((trigger.hazard_locations,) if trainer_avoidance_enabled else ()),
                *(
                    (trigger.hazard_locations,)
                    if engagement_trainer_id is not None and trigger.affordance_id == engagement_trainer_id
                    else ()
                ),
            )
            for location in locations
        }
        # Treat the tiles occupied by, and immediately surrounding, runtime
        # objects as checkpoints.  This keeps an NPC/trainer encounter in the
        # normal interaction path instead of carrying the player past it.
        for obj in world.objects:
            x, y = obj.location[1]
            checkpoints.update(
                {
                    obj.location,
                    (obj.location[0], (x - 1, y)),
                    (obj.location[0], (x + 1, y)),
                    (obj.location[0], (x, y - 1)),
                    (obj.location[0], (x, y + 1)),
                }
            )
        result: list[NavigationAction] = []
        for action in self._cached_actions[self._cached_action_index :]:
            if len(result) >= 8 or action.action_type is not NavigationActionType.MOVE:
                break
            if action.source[0] != world.map_id or action.destination[0] != world.map_id:
                break
            result.append(action)
            if action.destination in checkpoints:
                break
        return tuple(result)

    def _cancel_movement_batch(self, reason: str) -> None:
        trace = getattr(context, "stutter_trace", None)
        if trace is not None:
            trace.mark("route_invalidated", True)
            trace.mark("invalidation_reason", reason)
            trace.mark("state_transition", f"cached_route -> fallback: {reason}")
            if "map" in reason or "warp" in reason:
                trace.mark("map_warp_transition", True)
        count("cached_route_batch_interruptions")
        count(f"cached_route_batch_interruptions_{reason}")
        count("cached_route_invalidations")
        count(f"cached_route_invalidations_{reason}")
        if self._movement_batch is not None:
            self._report(f"MOVE_BATCH: interrupted reason={reason!r}")
        self._movement_batch = None
        self._fast_path_frames = 0
        self._in_flight_move = None
        self._in_flight_move_initial_facing = None
        reset_held_buttons = getattr(context.emulator, "reset_held_buttons", None)
        if callable(reset_held_buttons):
            reset_held_buttons()

    @profiled("cached_route_execution", "cached_route_frames")
    def _advance_movement_batch(self) -> bool:
        """Advance a batch for one emulator frame, returning whether it remains active.

        Only cheap state reads occur here.  Full perception, world signatures,
        goal evaluation, and pathfinding are deferred until the batch ends or
        an interruption is detected.
        """
        batch = self._movement_batch
        if batch is None:
            return False
        self._fast_path_frames += 1
        if self._force_periodic_reobserve():
            self._cancel_movement_batch("watchdog_reobserve")
            return False
        fast_path_start = now()
        trace = getattr(context, "stutter_trace", None)
        trace_fast_path_start = trace.now() if trace is not None else 0
        if trace is not None:
            trace.mark("cached_route_active", True)
            trace.mark("cached_route_actions_remaining", len(self._cached_actions) - self._cached_action_index)
        fast_path_completed = False
        try:
            # Do not call observe_interaction here.  It is the general-purpose
            # observation boundary and performs controllability, map-object,
            # task, and avatar checks intended for full planning decisions.
            # These reads are the minimum needed to fail closed while a known
            # same-map movement suffix is in flight.
            if get_game_state() is not GameState.OVERWORLD:
                count("cached_route_fast_path_fallback_frames")
                self._cancel_movement_batch("non_overworld_state")
                return False
            if is_emerald_field_message_rendering():
                count("cached_route_fast_path_fallback_frames")
                self._cancel_movement_batch("dialogue_rendering")
                return False
            if is_field_message_waiting_for_input():
                count("cached_route_fast_path_fallback_frames")
                self._cancel_movement_batch("dialogue_started")
                return False
            avatar = get_player_avatar()
            location = (avatar.map_group_and_number, avatar.local_coordinates)
            if trace is not None:
                trace.mark("map_id", location[0])
                trace.mark("player_position", location[1])
            action = batch.current
            if location[0] != action.source[0]:
                count("cached_route_fast_path_fallback_frames")
                self._cancel_movement_batch("map_transition")
                return False
            if location == action.destination:
                self._cached_action_index += 1
                batch.index += 1
                batch.frames_waiting = 0
                if batch.index >= len(batch.actions):
                    count("cached_route_batch_completions")
                    self._movement_batch = None
                    self._fast_path_frames = 0
                    self._in_flight_move = None
                    self._in_flight_move_initial_facing = None
                    reset_held_buttons = getattr(context.emulator, "reset_held_buttons", None)
                    if callable(reset_held_buttons):
                        reset_held_buttons()
                    return False
                action = batch.current
                reset_held_buttons = getattr(context.emulator, "reset_held_buttons", None)
                if callable(reset_held_buttons):
                    reset_held_buttons()
            elif location != action.source:
                count("cached_route_fast_path_fallback_frames")
                self._cancel_movement_batch("position_divergence")
                return False

            batch.frames_waiting += 1
            if batch.frames_waiting > 24:
                count("cached_route_fast_path_fallback_frames")
                self._movement_failure_count += 1
                self._cancel_movement_batch("movement_blocked")
                if self._movement_failure_count >= self._movement_failure_limit:
                    raise NavigationError(
                        "recovery/navigation movement made no progress " f"for {self._movement_failure_count} attempts"
                    )
                return False
            if avatar.facing_direction != action.direction.button_name:
                # Turning is execution, not diagnostics.  The previous
                # diagnostic guard made a cached route unable to take its
                # first differently-facing step unless --debug-trace was
                # enabled; a Pokémon Center door immediately north of the
                # avatar exposes that exact failure.
                if self._diagnostics_enabled():
                    self._report(
                        f"CACHED_ROUTE_DIRECTION_CORRECTION map={location[0]!r} "
                        f"source={action.source!r} destination={action.destination!r} "
                        f"observed_facing={avatar.facing_direction!r} "
                        f"planned_direction={action.direction.name!r}"
                    )
                press_direction = getattr(context.emulator, "press_direction", None)
                if callable(press_direction):
                    press_direction(action.direction.button_name, run=bool(action.run))
                else:
                    context.emulator.press_button(action.direction.button_name)
            else:
                # This method is called once per emulator frame.  A repeated
                # press_direction() is an input edge and is intentionally
                # suppressed by LibmgbaEmulator when the same direction was
                # present on the previous frame.  Cached running routes must
                # hold their inputs instead, otherwise the first step works
                # and every following frame is neutral input.
                hold_button = getattr(context.emulator, "hold_button", None)
                if callable(hold_button):
                    hold_button(action.direction.button_name)
                    if action.run:
                        hold_button("B")
                else:
                    press_direction = getattr(context.emulator, "press_direction", None)
                    if callable(press_direction):
                        press_direction(action.direction.button_name, run=bool(action.run))
                    else:
                        context.emulator.press_button(action.direction.button_name)
            fast_path_completed = True
            count("cached_route_fast_path_frames")
            # These counters intentionally remain zero on the fast path.  They
            # make the absence of expensive work visible in debug snapshots.
            count("cached_route_full_overworld_perceptions", 0)
            count("cached_route_goal_evaluations", 0)
            count("cached_route_pathfinding_calls", 0)
            return True
        except (AttributeError, RuntimeError, TypeError, ValueError):
            count("cached_route_fast_path_fallback_frames")
            self._cancel_movement_batch("state_read_failed")
            return False
        finally:
            if trace is not None:
                trace.mark("fast_path_used", fast_path_completed)
                trace.duration("fast_path_duration_ms", trace_fast_path_start)
            if fast_path_completed:
                timing("cached_route_fast_path_execution", fast_path_start)

    def _force_periodic_reobserve(self) -> bool:
        """Invalidate runtime caches at a frame-based observation boundary.

        Most control-loop iterations already call ``self._observe``.  Cached
        movement batches are the exception, so this boundary lives here and
        is also checked before each normal observation.  Using the emulator
        frame counter keeps the behavior stable across debug and headless
        runs, whose Python-loop timing can differ substantially.
        """
        emulator = getattr(context, "emulator", None)
        get_frame_count = getattr(emulator, "get_frame_count", None)
        if not callable(get_frame_count):
            return False
        try:
            frame = get_frame_count()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        if not isinstance(frame, int):
            return False
        if (
            self._last_observation_frame is not None
            and frame - self._last_observation_frame >= self._periodic_reobserve_after
        ):
            state_cache.invalidate_runtime_observations()
            self._report(f"WATCHDOG: forcing full observation after {self._periodic_reobserve_after} emulator frames")
            return True
        return False

    def _report(self, message: str) -> None:
        self._logger(f"AGENT_{message}")

    @staticmethod
    def _release_transition_input(pending: _PendingTransition) -> None:
        """Release only the direction sustained by a terminal transition."""
        release_button = getattr(context.emulator, "release_button", None)
        if callable(release_button):
            release_button(pending.direction.button_name)

    def dispose(self) -> None:
        """Release controller-owned transition input at a lifecycle boundary."""
        pending = self._pending_transition
        if pending is None:
            return
        # Clear ownership before calling the adapter so disposal is
        # idempotent even if adapter cleanup re-enters controller teardown.
        self._pending_transition = None
        self._expected_world_transition = None
        self._release_transition_input(pending)

    def _diagnostics_enabled(self) -> bool:
        return self._custom_logger or (context.debug and getattr(context, "debug_trace", False))

    def _diagnose_route103_objects(self, observation: AgentObservation) -> None:
        """Log Route 103's complete runtime object table at stable points.

        This is deliberately logging-only.  It is called after perception and
        does not alter the observation, binding resolution, or selected action.
        Multiple snapshots show whether object loading settles after a warp.
        """
        world = observation.overworld
        if world is None or world.map_id != MapRSE.ROUTE103.value:
            self._route103_stable_observations = 0
            self._route103_object_dumped_at.clear()
            return

        try:
            width, height = get_map_data(world.map_id, (0, 0)).map_size
        except (RuntimeError, ValueError, IndexError):
            self._route103_stable_observations = 0
            return
        in_bounds = 0 <= world.player_coordinates[0] < width and 0 <= world.player_coordinates[1] < height
        if world.movement_state is not MovementState.STANDING or not in_bounds:
            self._route103_stable_observations = 0
            return

        self._route103_stable_observations += 1
        # Capture immediately, then again after a short and a longer wait.
        if self._route103_stable_observations not in (1, 3, 10, 30):
            return
        stable_count = self._route103_stable_observations
        if stable_count in self._route103_object_dumped_at:
            return
        self._route103_object_dumped_at.add(stable_count)
        frame = None
        if context.emulator is not None:
            frame = context.emulator.get_frame_count()
        runtime_table = get_runtime_object_table(world.map_id)
        player_object = next((row for row in runtime_table if row.slot == 0), None)
        active_rows = tuple(row for row in runtime_table if row.active)
        local_id_2_rows = tuple(row for row in active_rows if row.local_id == 2)
        rival_rows = tuple(row for row in active_rows if row.script == "Route103_EventScript_Rival")
        source = self._last_world_transition_source
        saveblock_map = tuple(get_save_block(1, offset=4, size=2))
        self._report(
            f"ROUTE103_OBJECT_TABLE: stable_observation={stable_count} frame={frame!r} "
            f"player={world.player_coordinates!r} map={world.map_id!r} "
            f"map_name={get_map_enum(world.map_id).name!r} "
            f"hide_flag_HIDE_ROUTE_103_RIVAL={get_event_flag('HIDE_ROUTE_103_RIVAL')!r}"
        )
        self._report(
            "ROUTE103_RUNTIME_COMPARISON: "
            f"global_map={world.map_id!r} global_map_name={get_map_enum(world.map_id).name!r} "
            f"saveblock_map={saveblock_map!r} "
            f"player_coordinates={world.player_coordinates!r} "
            f"object_slot0_map={getattr(player_object, 'map_id', None)!r} "
            f"object_slot0_map_name="
            f"{get_map_enum(player_object.map_id).name if player_object and player_object.map_id else None!r} "
            f"object_slot0_coordinates={getattr(player_object, 'coordinates', None)!r} "
            f"object_slot0_local_id={getattr(player_object, 'local_id', None)!r} "
            f"active_object_count={len(active_rows)} local_id_2={tuple(row.slot for row in local_id_2_rows)!r} "
            f"rival_script={tuple(row.slot for row in rival_rows)!r} "
            f"last_warp_source={source!r}"
        )
        static_rival = next(
            (template for template in get_map_data(world.map_id, (0, 0)).objects if template.local_id == 2),
            None,
        )
        self._report(
            "ROUTE103_STATIC_OBJECT: "
            f"local_id=2 coordinates={getattr(static_rival, 'local_coordinates', None)!r} "
            f"script={getattr(static_rival, 'script_symbol', '')!r} "
            f"flag_id={getattr(static_rival, 'flag_id', None)!r} "
            f"flag_name={get_event_flag_name(getattr(static_rival, 'flag_id', 0))!r}"
        )
        for row in runtime_table:
            self._report(
                "ROUTE103_OBJECT: "
                f"slot={row.slot} active={row.active} local_id={row.local_id!r} "
                f"map={row.map_id!r} initial={row.initial_coordinates!r} "
                f"coordinates={row.coordinates!r} previous={row.previous_coordinates!r} "
                f"flags={row.flags!r} script={row.script!r} "
                f"template_coordinates={row.template_coordinates!r} "
                f"template_flag_id={row.template_flag_id!r} "
                f"template_flag_name={row.template_flag_name!r} "
                f"hide_flag_set={row.hide_flag_set!r}"
            )

    @staticmethod
    @profiled("navigation_world_signature", "world_signature_calculations")
    def _world_signature(observation: AgentObservation) -> tuple | None:
        world = observation.overworld
        if world is None:
            return None
        return (
            world.map_id,
            tuple(
                (
                    warp.entry,
                    warp.destination,
                    warp.required_facing,
                    warp.activation,
                    warp.activation_locations,
                    warp.activation_direction,
                )
                for warp in world.warps
            ),
            tuple(
                (
                    trigger.trigger_id,
                    trigger.activation_locations,
                    trigger.navigation_locations,
                    trigger.activation_requirements,
                    trigger.elevation,
                    trigger.target_map,
                    trigger.requires_input,
                )
                for trigger in world.triggers
            ),
            world.dynamic_blocked_coordinates,
            tuple((obj.local_id, obj.location, obj.script) for obj in world.objects),
            tuple(
                (
                    binding.binding.trigger_id,
                    binding.static_location,
                    binding.runtime_match,
                    binding.interaction_positions,
                )
                for binding in world.bindings
            ),
        )

    @staticmethod
    def _signature_change_affects_remaining_route(
        previous: tuple,
        current: tuple,
        remaining_actions: tuple[NavigationAction, ...],
    ) -> tuple[bool, str]:
        """Determine whether a live-world change can affect cached movement.

        Runtime object observations can change while a player is traversing a
        route.  Only changes on the remaining route can invalidate movement;
        static topology and bindings remain conservatively invalidating.
        """
        # A cached evaluation may be injected by a caller (or restored from
        # an older checkpoint) without the corresponding world signature.
        # Treat that as an unknown baseline and force the normal replan path;
        # attempting to diff it would mask the useful recovery with a TypeError.
        if previous is None or current is None:
            return True, "missing_world_signature baseline"
        changed = {index for index, (old, new) in enumerate(zip(previous, current)) if old != new}
        route_locations = {location for action in remaining_actions for location in (action.source, action.destination)}
        names = {0: "map", 1: "warps", 2: "triggers", 3: "dynamic_blocked", 4: "objects", 5: "bindings"}
        detail = f"changed={tuple(names[index] for index in sorted(changed))!r} route_locations={len(route_locations)}"
        if not changed:
            return False, detail
        if changed - {2, 3, 4}:
            return True, detail + " relevant=static_or_topology"

        if 3 in changed:
            changed_blocked = set(previous[3]) ^ set(current[3])
            if changed_blocked & {location[1] for location in route_locations}:
                return True, detail + " relevant=blocked_route_tile"
        if 4 in changed:
            old_objects = {item[1] for item in previous[4]}
            new_objects = {item[1] for item in current[4]}
            if (old_objects ^ new_objects) & route_locations:
                return True, detail + " relevant=route_object"
        if 2 in changed:
            old_triggers = {item[0]: item for item in previous[2]}
            new_triggers = {item[0]: item for item in current[2]}
            for trigger_id in old_triggers.keys() | new_triggers.keys():
                if old_triggers.get(trigger_id) == new_triggers.get(trigger_id):
                    continue
                for trigger in (old_triggers.get(trigger_id), new_triggers.get(trigger_id)):
                    if trigger is not None and (set(trigger[1]) | set(trigger[2])) & route_locations:
                        return True, detail + " relevant=route_trigger"
        return False, detail + " relevant=none"

    def _invalidate_plan(self, reason: str = "unspecified") -> None:
        trace = getattr(context, "stutter_trace", None)
        if trace is not None:
            trace.mark("route_invalidated", True)
            trace.mark("invalidation_reason", reason)
            trace.mark("state_transition", f"cached_plan -> replanning: {reason}")
            if "warp" in reason or "map" in reason:
                trace.mark("map_warp_transition", True)
        invalidation(reason)
        self._cached_evaluation = None
        if self._route_plan is not None:
            self._report(f"ROUTE_PLAN_INVALIDATED route_id={self._route_plan.route_id} invalidation_reason={reason!r}")
        self._route_plan = None
        self._cached_actions = ()
        self._cached_action_index = 0
        self._in_flight_move = None
        self._in_flight_move_initial_facing = None
        self._cached_goal = None
        self._cached_world_signature = None
        self._movement_batch = None
        self._movement_blocked_retries = 0
        reset_held_buttons = getattr(context.emulator, "reset_held_buttons", None)
        if callable(reset_held_buttons):
            reset_held_buttons()

    def _cached_decision(self, observation: AgentObservation) -> ActionDecision | None:
        trace = getattr(context, "stutter_trace", None)
        span = trace.span("cached_route_validation") if trace is not None else nullcontext()
        with span:
            return self._cached_decision_instrumented(observation)

    def _cached_decision_instrumented(self, observation: AgentObservation) -> ActionDecision | None:
        if self._cached_evaluation is None or self._cached_goal != self._goal:
            return None

        goal_target = _goal_target(self._goal)
        if isinstance(goal_target, ActivateTrigger) and _trigger_condition_completed(
            observation, goal_target.trigger_id
        ):
            # The ROM has completed an automatic scene while the old tactical
            # route is still mounted.  Drop that route so the next selection
            # observes the completed goal and lets campaign policy choose the
            # next objective.
            self._invalidate_plan("automatic trigger condition completed")
            return None

        count("cached_plan_validations")
        count("cached_plan_hits")

        location = (observation.overworld.map_id, observation.overworld.player_coordinates)
        cached_action = (
            self._cached_actions[self._cached_action_index]
            if self._cached_action_index < len(self._cached_actions)
            else None
        )
        current_world_signature = self._world_signature(observation)
        world_signature_changed = self._cached_world_signature != current_world_signature
        if world_signature_changed and not (
            cached_action is not None
            and cached_action.action_type is NavigationActionType.WARP
            and location == cached_action.source
        ):
            affects_route, signature_detail = self._signature_change_affects_remaining_route(
                self._cached_world_signature,
                current_world_signature,
                self._cached_actions[self._cached_action_index :],
            )
            trace = getattr(context, "stutter_trace", None)
            if trace is not None:
                trace.mark(
                    "route_invalidation_detail",
                    f"world_signature_changed current={location!r} "
                    f"cached_source={cached_action.source if cached_action else None!r} "
                    f"cached_destination={cached_action.destination if cached_action else None!r} "
                    f"{signature_detail} affects_route={affects_route}",
                )
            if cached_action is not None and not affects_route:
                self._cached_world_signature = current_world_signature
                return ActionDecision(
                    AgentAction(
                        AgentActionType.NAVIGATE_TOWARD_GOAL,
                        direction=cached_action.direction,
                        navigation=cached_action,
                        reason="continue cached route; irrelevant world change",
                    ),
                    self._cached_evaluation,
                )
            invalidation("world_signature_changed")
            return None

        if self._cached_action_index >= len(self._cached_actions):
            if self._cached_evaluation.plan is not None and location == self._cached_evaluation.plan.destination:
                if isinstance(_goal_target(self._goal), (ActivateTrigger, ReachInteractionPosition)):
                    world = NavigationWorld.from_overworld(observation.overworld)
                    if not GoalAwareNavigator(world).satisfies(
                        location, observation.overworld.facing, _goal_target(self._goal)
                    ):
                        # A plan may have reached the source tile without the
                        # final turn being accepted by the emulator. Never
                        # issue A on geometric adjacency alone.
                        self._invalidate_plan("interaction_precondition_not_ready")
                        return None
                    if isinstance(_goal_target(self._goal), ReachInteractionPosition) and _trigger_requires_input(
                        observation, _goal_target(self._goal).trigger_id
                    ):
                        return ActionDecision(
                            AgentAction(
                                AgentActionType.INTERACT,
                                option=_goal_target(self._goal).trigger_id,
                                reason="cached interaction source state reached",
                            ),
                            self._cached_evaluation,
                        )
                    if isinstance(_goal_target(self._goal), ReachInteractionPosition):
                        return ActionDecision(
                            AgentAction(
                                AgentActionType.WAIT_REOBSERVE,
                                reason="automatic coordinate trigger fires on tile entry",
                            ),
                            self._cached_evaluation,
                        )
                    if not _trigger_requires_input(observation, _goal_target(self._goal).trigger_id):
                        return ActionDecision(
                            AgentAction(
                                AgentActionType.WAIT_REOBSERVE,
                                reason="automatic coordinate trigger fires on tile entry",
                            ),
                            self._cached_evaluation,
                        )
                    return ActionDecision(
                        AgentAction(
                            AgentActionType.INTERACT,
                            option=_goal_target(self._goal).trigger_id,
                            reason="cached goal activation position reached",
                        ),
                        self._cached_evaluation,
                    )
                # ReachLocation completion belongs to the semantic goal
                # evaluator.  Drop the exhausted fast-path plan so the next
                # decision can produce GOAL_COMPLETE authoritatively.
                self._invalidate_plan("cached goal position reached")
                return None
            if self._implicit_transition_boundary(observation):
                trace = getattr(context, "stutter_trace", None)
                if trace is not None:
                    trace.mark("transition_state", "implicit_boundary_observed")
                    trace.mark("route_invalidation_detail", "cached route ended at known transition destination")
                self._warp_settling = True
                return ActionDecision(
                    AgentAction(
                        AgentActionType.WAIT_REOBSERVE,
                        reason="waiting for implicit transition destination to settle",
                    ),
                    self._cached_evaluation,
                )
            self._invalidate_plan("cached_plan_exhausted")
            return None
        navigation = cached_action
        if location != navigation.source:
            self._report(f"REPLAN: reason='plan divergence' expected={navigation.source!r} observed={location!r}")
            self._invalidate_plan("plan_divergence")
            return None
        return ActionDecision(
            AgentAction(
                AgentActionType.NAVIGATE_TOWARD_GOAL,
                direction=navigation.direction,
                navigation=navigation,
                reason="continue cached goal plan",
            ),
            self._cached_evaluation,
        )

    def step(self) -> tuple[AgentObservation, ActionDecision, ActionResult]:
        trace = getattr(context, "stutter_trace", None)
        span = trace.span("post_transition_replan_or_decision") if trace is not None else nullcontext()
        with span:
            return self._step_instrumented()

    def _step_instrumented(self) -> tuple[AgentObservation, ActionDecision, ActionResult]:
        profiling = getattr(context, "debug_profile", False)
        profile_start = perf_counter_ns() if profiling else 0
        trace = getattr(context, "stutter_trace", None)
        if trace is not None and self._cached_action_index < len(self._cached_actions):
            trace.mark("cached_route_active", True)
            trace.mark("cached_route_actions_remaining", len(self._cached_actions) - self._cached_action_index)
        previous_location = self._last_observed_location
        self._force_periodic_reobserve()
        observation = self._observe()
        emulator = getattr(context, "emulator", None)
        get_frame_count = getattr(emulator, "get_frame_count", None)
        if callable(get_frame_count):
            try:
                frame = get_frame_count()
                if isinstance(frame, int):
                    self._last_observation_frame = frame
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._previous_observed_location = previous_location
        if observation.overworld is not None:
            self._last_observed_location = (
                observation.overworld.map_id,
                observation.overworld.player_coordinates,
            )
        observe_elapsed = perf_counter_ns() - profile_start if profiling else 0
        if observation.goal is not None:
            self._goal = observation.goal
        elif self._goal is not None:
            observation = replace(observation, goal=self._goal)

        interaction_type = observation.interaction_type
        tactical_trace = isinstance(self._goal, (ReachLocation, ReachWarp))
        if tactical_trace:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_STEP: "
                    f"phase=observed frame={getattr(context, 'frame', None)!r} "
                    f"loop_id={id(self)!r} "
                    f"game_state={getattr(observation.interaction.game_state, 'name', repr(observation.interaction.game_state))!r} "
                    f"map={getattr(observation.overworld, 'map_id', None)!r} "
                    f"coordinates={getattr(observation.overworld, 'player_coordinates', None)!r} "
                    f"controllable={getattr(observation.overworld, 'controllable', None)!r} "
                    f"interaction_type={interaction_type.name!r} goal={self._goal!r} "
                    f"cached_action_index={self._cached_action_index!r} "
                    f"cached_actions_remaining={len(self._cached_actions) - self._cached_action_index!r} "
                    f"in_flight_move={self._in_flight_move!r} pending_transition={self._pending_transition!r} "
                    f"warp_settling={self._warp_settling!r} "
                    f"dynamic_blocked_here={getattr(observation.overworld, 'player_coordinates', None) in getattr(observation.overworld, 'dynamic_blocked_coordinates', ()) if observation.overworld is not None else None!r}"
                ),
                trace=True,
            )
        if observation.overworld is not None and not observation.overworld.controllable:
            # ROM/script movement owns the avatar now.  Do not let a bot-held
            # direction bleed into the first frames after control returns.
            release_button = getattr(context.emulator, "release_button", None)
            if callable(release_button):
                for direction in Direction:
                    release_button(direction.button_name)
        if self._dialogue_input_in_flight:
            if not observation.interaction.field_message_lifecycle_active:
                self._dialogue_input_in_flight = False
            elif observation.interaction.dialogue_waiting:
                self._dialogue_input_in_flight = False
            else:
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="waiting for field dialogue transition after input",
                )
                wait_decision = ActionDecision(wait_action)
                wait_result = self._executor.execute(wait_action, observation)
                return observation, wait_decision, wait_result
        if interaction_type is not InteractionType.OVERWORLD:
            self._invalidate_plan("non_overworld_interaction")
        if interaction_type is not self._last_interaction_type:
            self._report(f"STATE: {interaction_type.name}")
            if self._battle_was_active and interaction_type is InteractionType.OVERWORLD:
                self._report("BATTLE_RETURN: restoring previous tactical goal")
            self._last_interaction_type = interaction_type
        self._battle_was_active = interaction_type is InteractionType.BATTLE

        # Once this loop has emitted an interaction input, an active script is
        # the generic handoff from positioning to the interaction owner.  Do
        # not run the reached-position decision again and emit another input.
        if (
            self._started_interaction_id is not None
            and isinstance(_goal_target(self._goal), (ActivateTrigger, ReachInteractionPosition))
            and _goal_target(self._goal).trigger_id == self._started_interaction_id
            and observation.interaction.interaction_phase is InteractionPhase.FIELD_MESSAGE_INPUT_WAIT
        ):
            # A newly-started interaction may expose its first field message
            # before the global script context reports active. A ready field
            # message is the specific ownership signal, so let the dialogue
            # handler consume it instead of waiting on the generic handoff.
            self._started_interaction_id = None
            self._interaction_start_waits = 0
        if (
            self._started_interaction_id is not None
            and isinstance(_goal_target(self._goal), (ActivateTrigger, ReachInteractionPosition))
            and _goal_target(self._goal).trigger_id == self._started_interaction_id
            and observation.interaction.script_active
        ):
            wait_action = AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="started interaction script is executing",
            )
            return (
                observation,
                ActionDecision(wait_action),
                ActionResult(
                    ActionResultType.GOAL_COMPLETE,
                    wait_action,
                    "started interaction script owns the interaction",
                ),
            )
        if (
            self._started_interaction_id is not None
            and isinstance(_goal_target(self._goal), (ActivateTrigger, ReachInteractionPosition))
            and _goal_target(self._goal).trigger_id == self._started_interaction_id
            and not observation.interaction.script_active
        ):
            # Interaction dispatch is asynchronous. Re-emitting A every frame
            # while the ROM is still installing the object script can race the
            # first script/native transition (especially across a warp). Wait
            # for the script to claim ownership before considering another A.
            self._interaction_start_waits += 1
            if self._interaction_start_waits <= 32:
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="waiting for interaction script activation",
                )
                return (
                    observation,
                    ActionDecision(wait_action),
                    ActionResult(
                        ActionResultType.WAITING,
                        wait_action,
                        wait_action.reason,
                    ),
                )
            # The input was genuinely ignored. Permit one fresh evaluation so
            # the existing navigation/interaction machinery can retry it.
            self._started_interaction_id = None
            self._interaction_start_waits = 0

        pending = self._pending_transition
        if observation.overworld is not None and pending is not None:
            observed = (observation.overworld.map_id, observation.overworld.player_coordinates)
            destination_map = pending.destination[0] if pending.destination is not None else None
            if destination_map is not None and observation.overworld.map_id == destination_map:
                if pending.destination is not None and observed != pending.destination:
                    # A map connection can briefly expose the destination map
                    # while the avatar still has a boundary coordinate (for
                    # example, y == -1).  That coordinate is not a failed
                    # prediction: the destination map has changed, but its
                    # static tile model is not coherent yet.  Keep the
                    # transition authoritative and wait for a real tile
                    # before recording runtime transition evidence or
                    # invalidating the route.
                    if observation.overworld.tile_at(observed[1]) is None:
                        pending.settling_observations += 1
                        if pending.settling_observations <= self._pending_transition_watchdog_limit:
                            if pending.settling_observations == 1:
                                self._release_transition_input(pending)
                            self._report(
                                f"TRANSITION: destination_coordinate_settling "
                                f"observation={pending.settling_observations} "
                                f"observed={observed!r} expected={pending.destination!r}"
                            )
                            wait_action = AgentAction(
                                AgentActionType.WAIT_REOBSERVE,
                                reason="transition destination coordinate settling",
                            )
                            wait_decision = ActionDecision(wait_action)
                            wait_result = self._executor.execute(wait_action, observation)
                            return observation, wait_decision, wait_result
                    if observation.overworld.movement_state is MovementState.MOVING:
                        pending.settling_observations += 1
                        if pending.settling_observations <= 8:
                            self._report(
                                f"TRANSITION: settling observation={pending.settling_observations} "
                                f"observed={observed!r} expected={pending.destination!r}"
                            )
                            wait_action = AgentAction(
                                AgentActionType.WAIT_REOBSERVE,
                                reason="transition destination settling",
                            )
                            wait_decision = ActionDecision(wait_action)
                            wait_result = self._executor.execute(wait_action, observation)
                            return observation, wait_decision, wait_result
                    record_runtime_transition_observation(
                        pending.source, pending.transition_kind, pending.destination, observed
                    )
                    self._report(
                        f"TRANSITION_PREDICTION_MISMATCH source={pending.source!r} "
                        f"kind={pending.transition_kind!r} predicted={pending.destination!r} observed={observed!r}"
                    )
                    self._report(
                        f"TRANSITION: destination_coordinate_mismatch expected={pending.destination!r}"
                        f" observed={observed!r}"
                    )
                    self._release_transition_input(pending)
                    self._expected_world_transition = None
                    self._pending_transition = None
                    self._invalidate_plan("warp_destination_coordinate_mismatch")
                    self._warp_settling = True
                    return (
                        observation,
                        ActionDecision(
                            AgentAction(AgentActionType.WAIT_REOBSERVE, reason="transition destination mismatch")
                        ),
                        ActionResult(ActionResultType.UNREACHABLE, None, "transition destination mismatch"),
                    )
                self._report("TRANSITION: destination_observed")
                self._release_transition_input(pending)
                self._expected_world_transition = None
                self._pending_transition = None
                self._last_world_transition_source = pending.source
                trace = getattr(context, "stutter_trace", None)
                if trace is not None:
                    trace.mark("map_changed_event", True)
                    trace.mark("transition_state", "destination_observed")
                self._warp_wait_observations = 0
                # A confirmed destination is a successful hard checkpoint;
                # retain the complete route and continue with its suffix.
                if self._route_plan is not None:
                    self._report(
                        f"ROUTE_CHECKPOINT_REACHED route_id={self._route_plan.route_id} "
                        "checkpoint_kind='transition_postcondition'"
                    )
                    self._report(
                        f"ROUTE_CHECKPOINT_VALID route_id={self._route_plan.route_id} "
                        f"actual_map={observed[0]!r} actual_coordinate={observed[1]!r}"
                    )
                    self._route_plan.current_checkpoint += 1
                self._cached_world_signature = self._world_signature(observation)
                self._warp_settling = True
            elif observed[0] != pending.source[0] and destination_map is None:
                self._report("TRANSITION: destination_observed destination_unknown=True")
                self._release_transition_input(pending)
                self._pending_transition = None
                self._expected_world_transition = None
                self._invalidate_plan("transition_destination_observed")
                self._warp_settling = True
            elif observed[0] != pending.source[0]:
                record_runtime_transition_observation(
                    pending.source, pending.transition_kind, pending.destination, observed
                )
                self._report(
                    f"TRANSITION_PREDICTION_MISMATCH source={pending.source!r} "
                    f"kind={pending.transition_kind!r} predicted={pending.destination!r} observed={observed!r}"
                )
                self._report(
                    f"TRANSITION: blocked destination_mismatch expected={destination_map!r}"
                    f" observed={observed[0]!r}"
                )
                self._release_transition_input(pending)
                self._pending_transition = None
                self._expected_world_transition = None
                self._invalidate_plan("warp_destination_mismatch")
            elif interaction_type is InteractionType.OVERWORLD:
                if pending.last_position is not None and observed != pending.last_position:
                    pending.moved = True
                    pending.observations = 0
                    self._report("TRANSITION: player_moved_while_pending")
                else:
                    pending.observations += 1
                pending.last_position = observed
                if pending.observations >= self._pending_transition_watchdog_limit:
                    reason = "transition_watchdog_reobserve"
                    self._report(
                        f"WATCHDOG: {reason} observations={pending.observations} "
                        f"source={pending.source!r} destination={pending.destination!r}"
                    )
                    state_cache.invalidate_runtime_observations()
                    self._release_transition_input(pending)
                    self._pending_transition = None
                    self._expected_world_transition = None
                    self._warp_wait_observations = 0
                    self._invalidate_plan(reason)
                    wait_action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason=reason)
                    return (
                        observation,
                        ActionDecision(wait_action),
                        ActionResult(ActionResultType.WAITING, wait_action, reason),
                    )
                trace = getattr(context, "stutter_trace", None)
                if trace is not None:
                    trace.mark("transition_state", "player_moved" if pending.moved else "pending")
                # A fresh pulse is used only for dispatch.  Holding/ordinary
                # presses here lets directional activations remain sustained.
                hold_button = getattr(context.emulator, "hold_button", None)
                if callable(hold_button):
                    hold_button(pending.direction.button_name)
                else:
                    # Adapters without a held-button API can still provide a
                    # non-fresh directional event; importantly, never emit a
                    # fresh pulse for this unchanged pending transition.
                    context.emulator.press_button(pending.direction.button_name)
                self._report("TRANSITION: input_continuing")
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="transition pending; destination not yet observed",
                )
                wait_decision = ActionDecision(wait_action)
                return observation, wait_decision, self._executor.execute(wait_action, observation)

        if self._warp_settling and interaction_type is InteractionType.OVERWORLD and observation.overworld is not None:
            observed_location = (observation.overworld.map_id, observation.overworld.player_coordinates)
            # A map connection can expose the avatar on the destination map
            # before its coordinates belong to that map's static navigation
            # model.  Do not evaluate a local goal or replan from that
            # transient location; wait for a coherent destination observation.
            if observation.overworld.tile_at(observation.overworld.player_coordinates) is None:
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="waiting for a valid post-transition location",
                )
                wait_decision = ActionDecision(wait_action)
                wait_result = self._executor.execute(wait_action, observation)
                return observation, wait_decision, wait_result
            if not observation.overworld.controllable:
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="waiting for post-warp exit transition to settle",
                )
                wait_decision = ActionDecision(wait_action)
                wait_result = self._executor.execute(wait_action, observation)
                return observation, wait_decision, wait_result
            self._warp_settling = False

        if self._blocked_warp is not None:
            still_observed = observation.overworld is not None and any(
                warp == self._blocked_warp for warp in observation.overworld.warps
            )
            if not still_observed:
                self._blocked_warp = None
            elif isinstance(observation.goal, ReachWarp) and observation.goal.warp == self._blocked_warp:
                blocked_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="selected warp previously failed; awaiting recovery or a new observation",
                )
                blocked_evaluation = GoalEvaluation(
                    GoalStatus.UNREACHABLE,
                    reason=blocked_action.reason,
                )
                return (
                    observation,
                    ActionDecision(blocked_action, blocked_evaluation),
                    ActionResult(ActionResultType.UNREACHABLE, blocked_action, blocked_action.reason),
                )

        # Diagnostic-only snapshotting happens after transition handling and
        # requires a standing, in-bounds Route 103 observation.  It does not
        # participate in the decision below.
        if interaction_type is InteractionType.OVERWORLD and self._diagnostics_enabled():
            self._diagnose_route103_objects(observation)

        if self._in_flight_move is not None:
            location = (observation.overworld.map_id, observation.overworld.player_coordinates)
            if location == self._in_flight_move.source:
                movement_state = observation.overworld.movement_state
                if (
                    movement_state is MovementState.STANDING
                    and observation.overworld.facing is self._in_flight_move.direction
                    and self._in_flight_move_initial_facing is not self._in_flight_move.direction
                ):
                    # Emerald consumes the first opposite-direction press as a
                    # turn.  That is a completed input event, not a movement
                    # in flight, so send the cached step once more.
                    self._report("MOVE: turn_only_input_confirmed")
                    self._in_flight_move = None
                    self._in_flight_move_initial_facing = None
                elif movement_state is MovementState.STANDING:
                    trace = getattr(context, "stutter_trace", None)
                    if trace is not None:
                        trace.mark(
                            "route_invalidation_detail",
                            f"movement_blocked current={location!r} "
                            f"expected_source={self._in_flight_move.source!r} "
                            f"expected_destination={self._in_flight_move.destination!r} "
                            f"direction={self._in_flight_move.direction.name!r} "
                            f"destination_blocked={self._in_flight_move.destination[1] in observation.overworld.dynamic_blocked_coordinates!r}",
                        )
                    destination_blocked = (
                        self._in_flight_move.destination[1] in observation.overworld.dynamic_blocked_coordinates
                    )
                    if not destination_blocked and self._movement_blocked_retries == 0:
                        # A standing frame can be a transient script/input
                        # boundary. Retry a legal, unoccupied destination once
                        # before paying for synchronous replanning.
                        self._movement_blocked_retries = 1
                        self._in_flight_move = None
                        self._in_flight_move_initial_facing = None
                    else:
                        self._movement_failure_count += 1
                        self._report(
                            "REPLAN: reason='movement blocked'" f" requested={self._in_flight_move.direction.name!r}"
                        )
                        self._invalidate_plan("movement_blocked")
                        if self._movement_failure_count >= self._movement_failure_limit:
                            raise NavigationError(
                                "movement made no progress after " f"{self._movement_failure_count} attempts"
                            )
                else:
                    wait_action = AgentAction(
                        AgentActionType.WAIT_REOBSERVE,
                        reason="waiting for movement destination",
                    )
                    wait_decision = ActionDecision(wait_action)
                    wait_result = self._executor.execute(wait_action, observation)
                    return observation, wait_decision, wait_result
            if self._in_flight_move is not None and location == self._in_flight_move.destination:
                self._cached_action_index += 1
                self._movement_blocked_retries = 0
                self._movement_failure_count = 0
                self._in_flight_move = None
                self._in_flight_move_initial_facing = None
            elif self._in_flight_move is not None:
                trace = getattr(context, "stutter_trace", None)
                if trace is not None:
                    trace.mark(
                        "route_invalidation_detail",
                        f"movement_divergence current={location!r} "
                        f"expected_source={self._in_flight_move.source!r} "
                        f"expected_destination={self._in_flight_move.destination!r} "
                        f"direction={self._in_flight_move.direction.name!r}",
                    )
                self._report(
                    f"REPLAN: reason='movement divergence'"
                    f" expected={self._in_flight_move.destination!r} observed={location!r}"
                )
                self._invalidate_plan("movement_divergence")

        if observation.overworld is not None and self._diagnostics_enabled():
            self._report(
                f"OBSERVE: map={observation.overworld.map_id!r}"
                f" coordinates={observation.overworld.player_coordinates!r}"
                f" facing={getattr(observation.overworld.facing, 'name', None)!r}"
            )

        navigation_start = now()
        self._seed_navigation_plan(observation)
        decision = self._cached_decision(observation)
        if decision is None:
            decision = select_action(observation)
            if (
                decision.goal_evaluation is not None
                and decision.goal_evaluation.status is GoalStatus.REACHABLE
                and decision.goal_evaluation.plan is not None
            ):
                self._cached_evaluation = decision.goal_evaluation
                self._cached_actions = decision.goal_evaluation.plan.actions
                self._cached_action_index = 0
                self._route_plan = RoutePlan.from_navigation_plan(decision.goal_evaluation.plan, target=self._goal)
                self._report(
                    f"ROUTE_PLAN_CREATED route_id={self._route_plan.route_id} "
                    f"actions={len(self._route_plan.actions)} checkpoints={len(self._route_plan.checkpoints)}"
                )
                if self._diagnostics_enabled():
                    self._report(
                        "ROUTE_PLAN_ACTIONS "
                        + repr(
                            tuple(
                                (
                                    index,
                                    action.action_type.name,
                                    action.source,
                                    action.destination,
                                    action.direction.name,
                                    (
                                        observation.overworld.tile_at(action.destination[1]).has_encounters
                                        if observation.overworld is not None
                                        and action.destination is not None
                                        and action.destination[0] == observation.overworld.map_id
                                        and observation.overworld.tile_at(action.destination[1]) is not None
                                        else None
                                    ),
                                )
                                for index, action in enumerate(self._route_plan.actions)
                            )
                        )
                    )
                self._cached_goal = self._goal
                self._cached_world_signature = self._world_signature(observation)
                count("navigation_replans")
                trace = getattr(context, "stutter_trace", None)
                if trace is not None:
                    trace.mark("route_replanned", True)
                if self._diagnostics_enabled():
                    self._report("PLAN: replanned")
        elif self._diagnostics_enabled():
            if self._route_plan is not None:
                self._report(
                    f"ROUTE_PLAN_REUSED route_id={self._route_plan.route_id} "
                    f"action_index={self._cached_action_index}"
                )
            self._report("PLAN: continuing_cached")
        if tactical_trace:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_STEP: "
                    f"phase=selected frame={getattr(context, 'frame', None)!r} "
                    f"loop_id={id(self)!r} "
                    f"action={decision.action.action_type.name!r} "
                    f"direction={getattr(decision.action.direction, 'name', None)!r} "
                    f"navigation={decision.action.navigation!r} "
                    f"evaluation_status={getattr(getattr(decision, 'goal_evaluation', None), 'status', None)!r}"
                ),
                trace=True,
            )
        timing("navigation_total_decision", navigation_start)
        decision_elapsed = perf_counter_ns() - profile_start - observe_elapsed if profiling else 0
        if decision.goal_evaluation is not None and self._diagnostics_enabled():
            diagnostic_print(
                lambda: "CAMPAIGN_INTERACTION_DECISION: "
                + json.dumps(
                    {
                        "map": observation.overworld.map_id if observation.overworld else None,
                        "coordinates": observation.overworld.player_coordinates if observation.overworld else None,
                        "interaction_type": observation.interaction_type.name,
                        "goal": repr(observation.goal),
                        "goal_status": decision.goal_evaluation.status.name,
                        "action": decision.action.action_type.name,
                        "action_reason": decision.action.reason,
                        "option": decision.action.option,
                    },
                    sort_keys=True,
                    default=str,
                ),
                trace=True,
            )
            self._report(f"GOAL: {observation.goal!r} status={decision.goal_evaluation.status.name}")
            diagnostics = decision.goal_evaluation.diagnostics
            if diagnostics and diagnostics != self._last_navigation_diagnostics:
                for diagnostic in diagnostics:
                    self._report(f"NAV: {diagnostic}")
                self._last_navigation_diagnostics = diagnostics
            binding_diagnostics = decision.goal_evaluation.binding_diagnostics
            if binding_diagnostics and binding_diagnostics != self._last_binding_diagnostics:
                for diagnostic in binding_diagnostics:
                    self._report(f"BIND: {diagnostic}")
                self._last_binding_diagnostics = binding_diagnostics
            world_diagnostics = decision.goal_evaluation.world_diagnostics
            if world_diagnostics and world_diagnostics != self._last_world_diagnostics:
                for diagnostic in world_diagnostics:
                    self._report(f"WORLD: {diagnostic}")
                self._last_world_diagnostics = world_diagnostics
        if (
            decision.action.navigation is not None
            and decision.action.navigation.action_type is NavigationActionType.WARP
        ):
            warp = next(
                (
                    warp
                    for warp in (observation.overworld.warps if observation.overworld else ())
                    if warp.destination == decision.action.navigation.destination
                ),
                None,
            )
            self._report(f"WORLD: approaching_warp={warp.entry if warp else decision.action.navigation.destination!r}")
            self._report(f"WORLD: source_position={decision.action.navigation.source!r}")
            self._report(
                f"WORLD: warp_ready={decision.action.navigation.source == (observation.overworld.map_id, observation.overworld.player_coordinates) if observation.overworld else False}"
            )
        if self._diagnostics_enabled():
            self._report(
                f"ACTION: {decision.action.action_type.name}"
                f" direction={getattr(decision.action.direction, 'name', None)!r}"
                f" option={decision.action.option!r} reason={decision.action.reason!r}"
            )
        if decision.goal_evaluation is not None:
            if decision.goal_evaluation.status is GoalStatus.COMPLETE:
                return (
                    observation,
                    decision,
                    ActionResult(ActionResultType.GOAL_COMPLETE, decision.action, decision.goal_evaluation.reason),
                )
            if decision.goal_evaluation.status is GoalStatus.UNREACHABLE:
                return (
                    observation,
                    decision,
                    ActionResult(ActionResultType.UNREACHABLE, decision.action, decision.goal_evaluation.reason),
                )
        if (
            decision.action.navigation is not None
            and decision.action.navigation.action_type is NavigationActionType.WARP
            and observation.overworld is not None
            and observation.overworld.movement_state is not None
            and observation.overworld.movement_state is not MovementState.STANDING
        ):
            self._report(
                "WARP_WAIT: activation_deferred" f" movement_state={observation.overworld.movement_state.name!r}"
            )
            wait_action = AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="waiting for warp tile movement to settle",
            )
            wait_decision = ActionDecision(wait_action, decision.goal_evaluation)
            wait_result = self._executor.execute(wait_action, observation)
            return observation, wait_decision, wait_result
        if decision.action.navigation is not None:
            prewarm_warp_destination(observation, decision.action.navigation)
        result = self._executor.execute(decision.action, observation)
        if result.result_type is ActionResultType.EXECUTED and decision.action.action_type is AgentActionType.INTERACT:
            self._started_interaction_id = decision.action.option
            self._interaction_start_waits = 0
        if tactical_trace:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_STEP: "
                    f"phase=executed frame={getattr(context, 'frame', None)!r} "
                    f"loop_id={id(self)!r} "
                    f"action={decision.action.action_type.name!r} "
                    f"result={result.result_type.name!r} message={result.message!r} yielded=True"
                ),
                trace=True,
            )
        if (
            result.result_type is ActionResultType.EXECUTED
            and decision.action.action_type is AgentActionType.ADVANCE_DIALOGUE
            and not observation.interaction.controllable
            and observation.interaction.field_message_advance_ready
        ):
            self._dialogue_input_in_flight = True
        if profiling:
            total_elapsed = perf_counter_ns() - profile_start
            profile_print(
                lambda: f"AGENT_PROFILE: {format_snapshot()} "
                f"step_observe_ms={observe_elapsed / 1_000_000:.3f} "
                f"step_decision_ms={decision_elapsed / 1_000_000:.3f} "
                f"step_execute_ms={(total_elapsed - observe_elapsed - decision_elapsed) / 1_000_000:.3f} "
                f"step_total_ms={total_elapsed / 1_000_000:.3f}",
                every=60,
            )
        if decision.action.navigation is not None:
            if result.result_type is ActionResultType.EXECUTED:
                if decision.action.navigation.action_type is NavigationActionType.MOVE:
                    self._in_flight_move = decision.action.navigation
                    self._in_flight_move_initial_facing = observation.overworld.facing
                    safe_segment = self._safe_movement_batch(observation)
                    if len(safe_segment) > 1:
                        self._movement_batch = _MovementBatch(safe_segment)
                        count("cached_route_batch_starts")
                        count("cached_route_batch_actions", len(safe_segment))
                        self._report(
                            f"MOVE_BATCH: started steps={len(safe_segment)}"
                            f" from={safe_segment[0].source!r} to={safe_segment[-1].destination!r}"
                        )
                else:
                    self._cached_action_index += 1
            else:
                self._invalidate_plan("action_execution_failed")
        if (
            result.result_type is ActionResultType.EXECUTED
            and decision.action.navigation is not None
            and decision.action.navigation.action_type is NavigationActionType.WARP
        ):
            transition_kind = decision.action.navigation.transition_kind or "warp"
            transition_label = transition_kind.upper()
            self._report(
                f"WORLD: transition_ready={transition_label} " f"source_position={decision.action.navigation.source!r}"
            )
            self._report(f"WORLD: required_facing={decision.action.direction.name!r}")
            self._report(
                f"ACTION: transition_activation kind={transition_label} "
                f"direction={decision.action.direction.name!r}"
            )
            self._expected_world_transition = (
                decision.action.navigation.source,
                decision.action.navigation.destination,
            )
            self._pending_transition = _PendingTransition(
                action=decision.action.navigation,
                source=decision.action.navigation.source,
                activation_position=decision.action.navigation.source,
                direction=decision.action.direction,
                destination=decision.action.navigation.destination,
                transition_kind=transition_kind,
                input_issued=True,
                last_position=(observation.overworld.map_id, observation.overworld.player_coordinates),
            )
            self._warp_wait_observations = 0
            trace = getattr(context, "stutter_trace", None)
            if trace is not None:
                trace.mark("transition_state", "input_issued")
            self._report(
                f"TRANSITION_DISPATCH: kind={transition_label} direction={decision.action.direction.name!r}"
                f" source={decision.action.navigation.source!r}"
                f" expected_destination={decision.action.navigation.destination!r}"
                f" movement_state={getattr(observation.overworld.movement_state, 'name', None)!r}"
            )
            self._report("WORLD: waiting_for_warp_destination")
        if result.result_type is ActionResultType.DELEGATED:
            self._report("DELEGATE: existing BattleListener/battle system")
        elif result.result_type is ActionResultType.EXECUTED:
            self._report("EXECUTED")
        elif result.result_type is ActionResultType.WAITING:
            self._report(f"WAIT: {result.message or decision.action.reason}")
        return observation, decision, result

    def run(self) -> Generator:
        try:
            while True:
                if self._movement_batch is not None:
                    # The batch owns only the current emulator frame.  If it ends
                    # or is interrupted, immediately return to the normal loop on
                    # the next iteration so existing handlers process the cause.
                    if self._advance_movement_batch():
                        yield
                        continue
                _, _, result = self.step()
                if result.result_type is ActionResultType.GOAL_COMPLETE:
                    return
                if result.result_type in (ActionResultType.UNREACHABLE, ActionResultType.UNSUPPORTED):
                    # A generator returning here is indistinguishable from a
                    # successful recovery to CampaignController.  Propagate
                    # terminal navigation failure explicitly so recovery is
                    # marked failed and the interrupted objective remains
                    # available for safe policy-level handling.
                    raise NavigationError(result.message or result.result_type.name.lower())
                yield
        finally:
            self.dispose()
