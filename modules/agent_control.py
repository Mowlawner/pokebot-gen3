"""State-specific affordances, goal evaluation, and action execution.

The control loop is intentionally independent from ``main_loop`` and bot
modes.  A future mode or strategic planner can supply observations and goals,
then mount ``AgentControlLoop.run()`` on the existing controller stack when it
is ready to do so.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Generator

from modules.context import context
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
from modules.navigation import (
    GoalAwareNavigator,
    NavigationAction,
    NavigationActionType,
    NavigationError,
    NavigationPlan,
    NavigationWorld,
)
from modules.overworld import OverworldObservation, perceive_overworld


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


@dataclass(frozen=True)
class AgentObservation:
    interaction: InteractionObservation
    overworld: OverworldObservation | None = None
    goal: Goal | None = None

    @property
    def interaction_type(self) -> InteractionType:
        return classify_interaction(self.interaction)


def observe_agent(
    *,
    goal: Goal | None = None,
    choice_options: tuple[str, ...] = (),
    menu_options: tuple[str, ...] = (),
    special_interaction: str | None = None,
) -> AgentObservation:
    """Create one live observation, reading overworld data only when relevant."""

    interaction = observe_interaction(
        choice_options=choice_options,
        menu_options=menu_options,
        special_interaction=special_interaction,
    )
    interaction_type = classify_interaction(interaction)
    overworld = perceive_overworld() if interaction_type is InteractionType.OVERWORLD else None
    return AgentObservation(interaction=interaction, overworld=overworld, goal=goal)


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


def evaluate_goal(observation: AgentObservation) -> GoalEvaluation:
    """Evaluate a tactical goal against the current overworld observation."""

    if observation.goal is None:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="no goal supplied")
    if observation.overworld is None or observation.interaction_type is not InteractionType.OVERWORLD:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="goal requires an overworld observation")

    world = NavigationWorld.from_overworld(observation.overworld)
    start = (observation.overworld.map_id, observation.overworld.player_coordinates)
    try:
        plan = GoalAwareNavigator(world).plan(start, observation.goal)
    except NavigationError as error:
        return GoalEvaluation(GoalStatus.UNREACHABLE, reason=str(error))

    if isinstance(observation.goal, (ReachLocation, ReachWarp, ReachInteractionPosition)) and not plan.actions:
        return GoalEvaluation(GoalStatus.COMPLETE, plan=plan, reason="goal position reached")
    if isinstance(observation.goal, ActivateTrigger):
        activated = observation.interaction.metadata.get("activated_trigger_ids", ())
        if observation.goal.trigger_id in activated:
            return GoalEvaluation(GoalStatus.COMPLETE, plan=plan, reason="trigger is already activated")
    return GoalEvaluation(GoalStatus.REACHABLE, plan=plan)


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
            context.emulator.press_button(action.direction.button_name)
        elif action.action_type is AgentActionType.INTERACT:
            context.emulator.press_button("A")
        return ActionResult(ActionResultType.EXECUTED, action)


class AgentControlLoop:
    """A generator-compatible observe/select/execute loop for future mounting."""

    def __init__(self, observe: Callable[[], AgentObservation], executor: AgentActionExecutor | None = None):
        self._observe = observe
        self._executor = executor or AgentActionExecutor()

    def step(self) -> tuple[AgentObservation, ActionDecision, ActionResult]:
        observation = self._observe()
        decision = select_action(observation)
        if decision.goal_evaluation is not None:
            if decision.goal_evaluation.status is GoalStatus.COMPLETE:
                return observation, decision, ActionResult(
                    ActionResultType.GOAL_COMPLETE, decision.action, decision.goal_evaluation.reason
                )
            if decision.goal_evaluation.status is GoalStatus.UNREACHABLE:
                return observation, decision, ActionResult(
                    ActionResultType.UNREACHABLE, decision.action, decision.goal_evaluation.reason
                )
        result = self._executor.execute(decision.action, observation)
        return observation, decision, result

    def run(self) -> Generator:
        while True:
            self.step()
            yield
