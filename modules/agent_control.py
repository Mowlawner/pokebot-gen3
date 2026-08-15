"""State-specific affordances, goal evaluation, and action execution.

The control loop is intentionally independent from ``main_loop`` and bot
modes.  A future mode or strategic planner can supply observations and goals,
then mount ``AgentControlLoop.run()`` on the existing controller stack when it
is ready to do so.
"""

from dataclasses import dataclass, replace
from enum import Enum, auto
from time import perf_counter_ns
from typing import Callable, Generator

from modules.context import context
from modules.console import diagnostic_print, profile_print
from modules.goals import (
    ActivateTrigger,
    Goal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
)
from modules.interaction_state import (
    InteractionObservation,
    InteractionType,
    classify_interaction,
    observe_interaction,
)
from modules.map_path import Direction
from modules.map import get_event_flag, get_map_data, get_runtime_object_table
from modules.map_data import MapRSE, get_map_enum
from modules.game import get_event_flag_name
from modules.memory import get_save_block
from modules.navigation import (
    GoalAwareNavigator,
    NavigationAction,
    NavigationActionType,
    NavigationError,
    NavigationPlan,
    NavigationWorld,
    WorldNavigationError,
    goal_target_map,
    plan_with_world_navigation,
    navigation_diagnostics,
)
from modules.overworld import MovementState, OverworldObservation, perceive_overworld
from modules.profiler import count, invalidation, now, profiled, timing, format_snapshot


class AgentActionType(Enum):
    ADVANCE_DIALOGUE = auto()
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


@profiled("agent_observation", "agent_observations")
def observe_agent(
    *,
    goal: Goal | None = None,
    choice_options: tuple[str, ...] = (),
    menu_options: tuple[str, ...] = (),
    special_interaction: str | None = None,
) -> AgentObservation:
    """Create one live observation, reading overworld data only when relevant."""

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
    perception_start = now()
    overworld = perceive_overworld() if interaction_type is InteractionType.OVERWORLD else None
    timing("agent_overworld_perception", perception_start)
    count("agent_overworld_perceptions")
    finalize_start = now()
    observation = AgentObservation(interaction=interaction, overworld=overworld, goal=goal)
    timing("agent_observation_finalize", finalize_start)
    count("agent_observation_finalizations")
    return observation


@dataclass(frozen=True)
class ActionDecision:
    action: AgentAction
    goal_evaluation: GoalEvaluation | None = None


def _dialogue_actions(observation: AgentObservation) -> tuple[AgentAction, ...]:
    if observation.interaction.dialogue_waiting:
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
            AgentAction(AgentActionType.NAVIGATE_MENU, option=option)
            for option in observation.interaction.menu_options
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

    if observation.goal is None:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="no goal supplied")
    if observation.overworld is None or observation.interaction_type is not InteractionType.OVERWORLD:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="goal requires an overworld observation")

    world = NavigationWorld.from_overworld(observation.overworld)
    start = (observation.overworld.map_id, observation.overworld.player_coordinates)
    binding_diagnostics = ()
    world_diagnostics = ()
    if isinstance(observation.goal, (ActivateTrigger, ReachInteractionPosition)):
        resolution = next(
            (binding for binding in observation.overworld.bindings
             if binding.binding.trigger_id == observation.goal.trigger_id),
            None,
        )
        if resolution is not None:
            binding_diagnostics = (
                f"trigger={resolution.binding.trigger_id!r}"
                f" map={resolution.binding.map_id!r}"
                f" script={resolution.binding.script_symbol!r}",
                f"runtime_match={resolution.runtime_match}"
                f" object_id={resolution.object_ids!r}",
                f"static_match={resolution.static_match}"
                f" static_available={resolution.static_available}"
                f" static_ambiguous={resolution.static_ambiguous}"
                f" static_location={resolution.static_location!r}",
                f"target_positions={resolution.interaction_positions!r}",
            )
    try:
        plan, world_route = plan_with_world_navigation(world, start, observation.goal)
        target_map = goal_target_map(world, observation.goal)
        if world_route is not None and target_map is not None:
            world_diagnostics = (
                f"current_map={start[0]!r} target_map={target_map!r}",
                f"maps={world_route.maps!r}",
                f"edges={tuple((edge.kind, edge.source_map, edge.destination_map,
                                edge.source_coordinates) for edge in world_route.edges)!r}",
            )
    except (NavigationError, WorldNavigationError) as error:
        return GoalEvaluation(
            GoalStatus.UNREACHABLE,
            reason=str(error),
            diagnostics=navigation_diagnostics(world, start, observation.goal),
            binding_diagnostics=binding_diagnostics,
            world_diagnostics=world_diagnostics,
        )

    if isinstance(observation.goal, (ReachLocation, ReachWarp, ReachInteractionPosition)) and not plan.actions:
        return GoalEvaluation(
            GoalStatus.COMPLETE, plan=plan, reason="goal position reached",
            binding_diagnostics=binding_diagnostics,
            world_diagnostics=world_diagnostics,
        )
    if isinstance(observation.goal, ActivateTrigger):
        resolution = next(
            (binding for binding in observation.overworld.bindings
             if binding.binding.trigger_id == observation.goal.trigger_id),
            None,
        )
        activated = observation.interaction.metadata.get("activated_trigger_ids", ())
        if observation.goal.trigger_id in activated:
            return GoalEvaluation(
                GoalStatus.COMPLETE, plan=plan, reason="trigger is already activated",
                binding_diagnostics=binding_diagnostics,
                world_diagnostics=world_diagnostics,
            )
        if resolution is not None and not resolution.runtime_match:
            return GoalEvaluation(
                GoalStatus.REACHABLE, plan=plan,
                reason="static target reached; waiting for runtime object",
                binding_diagnostics=binding_diagnostics,
                world_diagnostics=world_diagnostics,
            )
    return GoalEvaluation(
        GoalStatus.REACHABLE, plan=plan,
        binding_diagnostics=binding_diagnostics,
        world_diagnostics=world_diagnostics,
    )


@profiled("agent_action_selection", "actions_selected")
def select_action(observation: AgentObservation) -> ActionDecision:
    """Select one deterministic action after state-specific affordance generation."""

    interaction_type = observation.interaction_type
    if interaction_type is not InteractionType.OVERWORLD:
        return ActionDecision(available_actions(observation)[0])
    evaluation = evaluate_goal(observation)
    if evaluation.status is GoalStatus.COMPLETE:
        return ActionDecision(
            AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation
        )
    if evaluation.status is GoalStatus.UNREACHABLE:
        return ActionDecision(
            AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation
        )
    if evaluation.status is GoalStatus.NOT_APPLICABLE:
        return ActionDecision(
            AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation
        )
    if (
        isinstance(observation.goal, ActivateTrigger)
        and evaluation.status is GoalStatus.REACHABLE
        and evaluation.plan is not None
        and not evaluation.plan.actions
    ):
        resolution = next(
            (binding for binding in observation.overworld.bindings
             if binding.binding.trigger_id == observation.goal.trigger_id),
            None,
        )
        runtime_available = (
            resolution.runtime_match if resolution is not None else any(
                trigger.trigger_id == observation.goal.trigger_id
                and trigger.activation_locations
                for trigger in observation.overworld.triggers
            )
        )
        if not runtime_available:
            return ActionDecision(
                AgentAction(AgentActionType.WAIT_REOBSERVE,
                            reason="static target reached; waiting for runtime object"),
                evaluation,
            )
        return ActionDecision(
            AgentAction(
                AgentActionType.INTERACT,
                option=observation.goal.trigger_id,
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


class AgentActionExecutor:
    """Translate semantic actions to input only after selection is complete."""

    def __init__(self, *, choose_option: Callable[[str], None] | None = None,
                 navigate_menu: Callable[[str], None] | None = None):
        self._choose_option = choose_option
        self._navigate_menu = navigate_menu

    @profiled("agent_action_execution", "actions_executed")
    def execute(self, action: AgentAction, observation: AgentObservation) -> ActionResult:
        if action.action_type is AgentActionType.WAIT_REOBSERVE:
            return ActionResult(ActionResultType.WAITING, action, action.reason)
        if action.action_type is AgentActionType.DELEGATE_BATTLE:
            return ActionResult(ActionResultType.DELEGATED, action, action.reason)
        if observation.interaction_type in (InteractionType.UNKNOWN, InteractionType.SPECIAL_INTERACTION):
            return ActionResult(ActionResultType.WAITING, AgentAction(AgentActionType.WAIT_REOBSERVE,
                                                                       reason="unsafe interaction state"),
                                "unsafe interaction state")
        if context.emulator is None and action.action_type not in (
                AgentActionType.CHOOSE_DIALOGUE_OPTION, AgentActionType.NAVIGATE_MENU):
            return ActionResult(ActionResultType.UNSUPPORTED, action, "no emulator is attached")

        if action.action_type is AgentActionType.ADVANCE_DIALOGUE:
            if not observation.interaction.dialogue_waiting:
                return ActionResult(ActionResultType.WAITING, action, "dialogue is not ready")
            context.emulator.press_button("A")
        elif action.action_type is AgentActionType.CHOOSE_DIALOGUE_OPTION:
            if action.option not in observation.interaction.choice_options:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "choice is not available")
            if self._choose_option is None:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "no choice controller is attached")
            self._choose_option(action.option)
        elif action.action_type is AgentActionType.NAVIGATE_MENU:
            if action.option not in observation.interaction.menu_options:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "menu option is not available")
            if self._navigate_menu is None:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "no menu controller is attached")
            self._navigate_menu(action.option)
        elif action.action_type in (AgentActionType.MOVE, AgentActionType.NAVIGATE_TOWARD_GOAL):
            if action.direction is None:
                return ActionResult(ActionResultType.UNSUPPORTED, action, "movement direction is missing")
            press_button_fresh = getattr(type(context.emulator), "press_button_fresh", None)
            if (action.navigation is not None
                    and action.navigation.action_type is NavigationActionType.WARP
                    and callable(press_button_fresh)):
                context.emulator.press_button_fresh(action.direction.button_name)
            else:
                context.emulator.press_button(action.direction.button_name)
        elif action.action_type is AgentActionType.INTERACT:
            context.emulator.press_button("A")
        return ActionResult(ActionResultType.EXECUTED, action)


class AgentControlLoop:
    """A generator-compatible observe/select/execute loop for future mounting."""

    def __init__(
        self,
        observe: Callable[[], AgentObservation],
        executor: AgentActionExecutor | None = None,
        *,
        goal: Goal | None = None,
        logger: Callable[[str], None] | None = None,
    ):
        self._observe = observe
        self._executor = executor or AgentActionExecutor()
        self._goal = goal
        self._custom_logger = logger is not None
        self._logger = logger or (lambda message: diagnostic_print(message, trace=True))
        self._last_interaction_type: InteractionType | None = None
        self._battle_was_active = False
        self._last_navigation_diagnostics: tuple[str, ...] | None = None
        self._last_binding_diagnostics: tuple[str, ...] | None = None
        self._last_world_diagnostics: tuple[str, ...] | None = None
        self._expected_world_transition: tuple[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[tuple[int, int], tuple[int, int]],
        ] | None = None
        self._warp_wait_observations = 0
        self._cached_evaluation: GoalEvaluation | None = None
        self._cached_actions: tuple[NavigationAction, ...] = ()
        self._cached_action_index = 0
        self._in_flight_move: NavigationAction | None = None
        self._in_flight_move_initial_facing: Direction | None = None
        self._cached_goal: Goal | None = None
        self._cached_world_signature: tuple | None = None
        self._route103_stable_observations = 0
        self._route103_object_dumped_at: set[int] = set()
        self._last_world_transition_source: tuple[tuple[int, int], tuple[int, int]] | None = None

    def _report(self, message: str) -> None:
        self._logger(f"AGENT_{message}")

    def _diagnostics_enabled(self) -> bool:
        return self._custom_logger or (
            context.debug and getattr(context, "debug_trace", False)
        )

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
            tuple((warp.entry, warp.destination, warp.required_facing) for warp in world.warps),
            tuple((trigger.trigger_id, trigger.activation_locations,
                   trigger.navigation_locations, trigger.target_map)
                  for trigger in world.triggers),
            world.dynamic_blocked_coordinates,
            tuple((obj.local_id, obj.location, obj.script) for obj in world.objects),
            tuple((binding.binding.trigger_id, binding.static_location,
                   binding.runtime_match, binding.interaction_positions)
                  for binding in world.bindings),
        )

    def _invalidate_plan(self, reason: str = "unspecified") -> None:
        invalidation(reason)
        self._cached_evaluation = None
        self._cached_actions = ()
        self._cached_action_index = 0
        self._in_flight_move = None
        self._in_flight_move_initial_facing = None
        self._cached_goal = None
        self._cached_world_signature = None

    def _cached_decision(self, observation: AgentObservation) -> ActionDecision | None:
        if self._cached_evaluation is None or self._cached_goal != self._goal:
            return None

        count("cached_plan_validations")

        location = (observation.overworld.map_id, observation.overworld.player_coordinates)
        cached_action = (
            self._cached_actions[self._cached_action_index]
            if self._cached_action_index < len(self._cached_actions)
            else None
        )
        world_signature_changed = self._cached_world_signature != self._world_signature(observation)
        if world_signature_changed and not (
            cached_action is not None
            and cached_action.action_type is NavigationActionType.WARP
            and location == cached_action.source
        ):
            invalidation("world_signature_changed")
            return None

        if self._cached_action_index >= len(self._cached_actions):
            if (self._cached_evaluation.plan is not None
                    and location == self._cached_evaluation.plan.destination):
                if isinstance(self._goal, ActivateTrigger):
                    resolution = next(
                        (binding for binding in observation.overworld.bindings
                         if binding.binding.trigger_id == self._goal.trigger_id),
                        None,
                    )
                    runtime_available = (
                        resolution.runtime_match if resolution is not None else any(
                            trigger.trigger_id == self._goal.trigger_id
                            and trigger.activation_locations
                            for trigger in observation.overworld.triggers
                        )
                    )
                    if not runtime_available:
                        return ActionDecision(AgentAction(
                            AgentActionType.WAIT_REOBSERVE,
                            reason="static target reached; waiting for runtime object",
                        ), self._cached_evaluation)
                    return ActionDecision(AgentAction(
                        AgentActionType.INTERACT, option=self._goal.trigger_id,
                        reason="cached goal activation position reached",
                    ), self._cached_evaluation)
                return ActionDecision(AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="cached goal position reached",
                ), self._cached_evaluation)
            self._invalidate_plan("cached_plan_exhausted")
            return None
        navigation = cached_action
        if location != navigation.source:
            self._report(f"REPLAN: reason='plan divergence' expected={navigation.source!r} observed={location!r}")
            self._invalidate_plan("plan_divergence")
            return None
        return ActionDecision(AgentAction(
            AgentActionType.NAVIGATE_TOWARD_GOAL,
            direction=navigation.direction, navigation=navigation,
            reason="continue cached goal plan",
        ), self._cached_evaluation)

    def step(self) -> tuple[AgentObservation, ActionDecision, ActionResult]:
        profiling = getattr(context, "debug_profile", False)
        profile_start = perf_counter_ns() if profiling else 0
        observation = self._observe()
        observe_elapsed = perf_counter_ns() - profile_start if profiling else 0
        if observation.goal is not None:
            self._goal = observation.goal
        elif self._goal is not None:
            observation = replace(observation, goal=self._goal)

        interaction_type = observation.interaction_type
        if interaction_type is not InteractionType.OVERWORLD:
            self._invalidate_plan("non_overworld_interaction")
        if interaction_type is not self._last_interaction_type:
            self._report(f"STATE: {interaction_type.name}")
            if self._battle_was_active and interaction_type is InteractionType.OVERWORLD:
                self._report("BATTLE_RETURN: restoring previous tactical goal")
            self._last_interaction_type = interaction_type
        self._battle_was_active = interaction_type is InteractionType.BATTLE

        if observation.overworld is not None and self._expected_world_transition is not None:
            expected_source, expected_destination = self._expected_world_transition
            observed_map = observation.overworld.map_id
            if observed_map == expected_destination[0]:
                self._report(
                    f"WARP_TRANSITION: map_changed=True map={observed_map!r}"
                    f" position={observation.overworld.player_coordinates!r}"
                )
                self._report(f"WORLD: warp_destination_confirmed={observed_map!r}")
                self._expected_world_transition = None
                self._last_world_transition_source = expected_source
                self._warp_wait_observations = 0
                self._invalidate_plan("warp_destination_confirmed")
            elif observed_map != expected_source[0]:
                # A map transition happened, but not the one predicted by the
                # static route.  Clear the expectation so the next decision
                # recomputes from the observed map.
                self._report(
                    f"WORLD: destination_mismatch=True"
                    f" expected={expected_destination[0]!r} observed={observed_map!r}"
                )
                self._expected_world_transition = None
                self._invalidate_plan("warp_destination_mismatch")
            elif interaction_type is InteractionType.OVERWORLD:
                self._warp_wait_observations += 1
                if self._warp_wait_observations <= 3 or self._warp_wait_observations % 10 == 0:
                    try:
                        emulator_frame = context.emulator.get_frame_count()
                    except (AttributeError, RuntimeError, TypeError):
                        emulator_frame = None
                    self._report(
                        f"WARP_WAIT: observation={self._warp_wait_observations}"
                        f" frame={emulator_frame!r} map={observation.overworld.map_id!r}"
                        f" position={observation.overworld.player_coordinates!r}"
                        f" facing={getattr(observation.overworld.facing, 'name', None)!r}"
                        f" movement_state={getattr(observation.overworld.movement_state, 'name', None)!r}"
                        f" controllable={observation.overworld.controllable!r}"
                        f" expected_map={expected_destination[0]!r}"
                    )
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason=f"awaiting warp destination {expected_destination[0]!r}",
                )
                wait_decision = ActionDecision(wait_action)
                wait_result = self._executor.execute(wait_action, observation)
                return observation, wait_decision, wait_result

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
                    self._report(
                        "REPLAN: reason='movement blocked'"
                        f" requested={self._in_flight_move.direction.name!r}"
                    )
                    self._invalidate_plan("movement_blocked")
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
                self._in_flight_move = None
                self._in_flight_move_initial_facing = None
            elif self._in_flight_move is not None:
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
        decision = self._cached_decision(observation)
        if decision is None:
            decision = select_action(observation)
            if (decision.goal_evaluation is not None
                    and decision.goal_evaluation.status is GoalStatus.REACHABLE
                    and decision.goal_evaluation.plan is not None):
                self._cached_evaluation = decision.goal_evaluation
                self._cached_actions = decision.goal_evaluation.plan.actions
                self._cached_action_index = 0
                self._cached_goal = self._goal
                self._cached_world_signature = self._world_signature(observation)
                if self._diagnostics_enabled():
                    self._report("PLAN: replanned")
        elif self._diagnostics_enabled():
            self._report("PLAN: continuing_cached")
        timing("navigation_total_decision", navigation_start)
        decision_elapsed = perf_counter_ns() - profile_start - observe_elapsed if profiling else 0
        if decision.goal_evaluation is not None and self._diagnostics_enabled():
            self._report(
                f"GOAL: {observation.goal!r} status={decision.goal_evaluation.status.name}"
            )
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
        if (decision.action.navigation is not None
                and decision.action.navigation.action_type is NavigationActionType.WARP):
            warp = next((warp for warp in (observation.overworld.warps if observation.overworld else ())
                          if warp.destination == decision.action.navigation.destination), None)
            self._report(f"WORLD: approaching_warp={warp.entry if warp else decision.action.navigation.destination!r}")
            self._report(f"WORLD: source_position={decision.action.navigation.source!r}")
            self._report(f"WORLD: warp_ready={decision.action.navigation.source == (observation.overworld.map_id, observation.overworld.player_coordinates) if observation.overworld else False}")
        if self._diagnostics_enabled():
            self._report(
                f"ACTION: {decision.action.action_type.name}"
                f" direction={getattr(decision.action.direction, 'name', None)!r}"
                f" option={decision.action.option!r} reason={decision.action.reason!r}"
            )
        if decision.goal_evaluation is not None:
            if decision.goal_evaluation.status is GoalStatus.COMPLETE:
                return observation, decision, ActionResult(
                    ActionResultType.GOAL_COMPLETE, decision.action, decision.goal_evaluation.reason
                )
            if decision.goal_evaluation.status is GoalStatus.UNREACHABLE:
                return observation, decision, ActionResult(
                    ActionResultType.UNREACHABLE, decision.action, decision.goal_evaluation.reason
                )
        if (
            decision.action.navigation is not None
            and decision.action.navigation.action_type is NavigationActionType.WARP
            and observation.overworld is not None
            and observation.overworld.movement_state is not None
            and observation.overworld.movement_state is not MovementState.STANDING
        ):
            self._report(
                "WARP_WAIT: activation_deferred"
                f" movement_state={observation.overworld.movement_state.name!r}"
            )
            wait_action = AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="waiting for warp tile movement to settle",
            )
            wait_decision = ActionDecision(wait_action, decision.goal_evaluation)
            wait_result = self._executor.execute(wait_action, observation)
            return observation, wait_decision, wait_result
        result = self._executor.execute(decision.action, observation)
        if profiling:
            total_elapsed = perf_counter_ns() - profile_start
            profile_print(lambda: f"AGENT_PROFILE: {format_snapshot()} "
                                  f"step_observe_ms={observe_elapsed / 1_000_000:.3f} "
                                  f"step_decision_ms={decision_elapsed / 1_000_000:.3f} "
                                  f"step_execute_ms={(total_elapsed - observe_elapsed - decision_elapsed) / 1_000_000:.3f} "
                                  f"step_total_ms={total_elapsed / 1_000_000:.3f}")
        if decision.action.navigation is not None:
            if result.result_type is ActionResultType.EXECUTED:
                if decision.action.navigation.action_type is NavigationActionType.MOVE:
                    self._in_flight_move = decision.action.navigation
                    self._in_flight_move_initial_facing = observation.overworld.facing
                else:
                    self._cached_action_index += 1
            else:
                    self._invalidate_plan("action_execution_failed")
        if (
            result.result_type is ActionResultType.EXECUTED
            and decision.action.navigation is not None
            and decision.action.navigation.action_type is NavigationActionType.WARP
        ):
            self._report(f"WORLD: warp_ready=True source_position={decision.action.navigation.source!r}")
            self._report(f"WORLD: required_facing={decision.action.direction.name!r}")
            self._report(f"ACTION: warp_activation direction={decision.action.direction.name!r}")
            self._expected_world_transition = (
                decision.action.navigation.source,
                decision.action.navigation.destination,
            )
            self._warp_wait_observations = 0
            self._report(
                f"WARP_DISPATCH: direction={decision.action.direction.name!r}"
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
        while True:
            self.step()
            yield
