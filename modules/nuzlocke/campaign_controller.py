"""Small orchestration layer for the ordered campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Generator, Iterator, Any
import json
import traceback
from time import monotonic
from functools import lru_cache

from modules.agent_control import AgentControlLoop, observe_agent
from modules.goals import (
    ActivateTrigger,
    EncounterMode,
    EngageTrainer,
    Goal,
    GoalConstraints,
    NavigationGoal,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    TrainerEngagementMode,
    TrainerMode,
)
from modules.context import context
from modules.interaction_state import InteractionPhase, InteractionType, classify_interaction, observe_interaction
from modules.memory import GameState, get_game_state
from modules.tasks import get_tasks
from modules.navigation import (
    GoalAwareNavigator,
    NavigationError,
    NavigationSearchLimitExceeded,
    NavigationWorld,
    resolve_trainer_engagement_plan,
)
from modules.trainer_checkpoints import trainer_checkpoint_for_id
from modules.modes.util.lead_rotation import CampaignTrainerLeadFailure, ensure_campaign_trainer_lead

from .campaign_execution import CampaignExecutionResult, CampaignExecutionStatus, adapt_campaign_execution
from .campaign_objectives import (
    ObjectiveSelection,
    ObjectiveStatus,
    campaign_restock_item_summary,
    campaign_frontier_summary,
    campaign_task_diagnostics,
    select_campaign_objective,
    select_available_campaign_task,
    heal_party_objective,
)
from modules.console import (
    diagnostic_print,
    print_always,
    format_campaign_objective_id,
    print_campaign_frontier,
    print_campaign_snapshot,
    print_recovery_handoff,
    print_recovery_plan,
    print_recovery_failure,
)
from .campaign_status import CampaignStatus, recovery_status
from .campaign_state import CampaignState, Fact, FactStatus
from .snapshots import CampaignObservationLifecycle
from .readiness_diagnostics import (
    CampaignReadinessPolicy,
    ProgressionReadinessDiagnostic,
    ReadinessDecision,
    ReadinessReason,
    evaluate_progression_readiness,
    _readiness_calculation_pending,
)
from .campaign_planner import CampaignPlan, build_campaign_plan, RecoveryStop
from .rule_config import CampaignRuleId


class CampaignControllerStatus(Enum):
    """Lifecycle status exposed by the campaign controller."""

    READY = "ready"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    FAILED = "failed"
    RUN_LOST = "run_lost"


@dataclass(frozen=True, slots=True)
class CampaignControllerState:
    """Immutable controller state snapshot for GUI and integration consumers."""

    status: CampaignControllerStatus
    objective_id: str | None
    tactical_goal: Goal | None
    selection: ObjectiveSelection | None
    execution: CampaignExecutionResult | None
    reason: str
    execution_phase: str = "CAMPAIGN"
    recovery_status: str | None = None
    plan: CampaignPlan | None = None


TacticalLoopFactory = Callable[[Goal], Iterator[object]]
CampaignBoundaryHandler = Callable[[str], bool]
ReadinessProvider = Callable[[Any, Goal | None], ProgressionReadinessDiagnostic]
RecoveryFactory = Callable[[RecoveryStop], Iterator[object]]


# These are completed ROM milestones, not mutable resources. Keeping their
# high-water values prevents a transient battle/script read from rewinding the
# dependency planner to an earlier story objective.
_MONOTONIC_CAMPAIGN_FACTS = (
    "text_speed_fast",
    "new_game_setup_complete",
    "wall_clock_set",
    "rival_met",
    "birch_rescued",
    "starter_obtained",
    "intro_rival_battle_complete",
    "pokedex_received",
    "pokeballs_received",
    "visited_petalburg",
    "petalburg_wally_scene_complete",
    "petalburg_woods_scene_complete",
    "devon_goods_stolen",
    "devon_goods_reported",
    "devon_goods_recovered",
    "devon_goods_returned",
    "devon_goods_delivered",
    "devon_corp_3f_scene_complete",
    "visited_rustboro",
    "first_badge_obtained",
)

# A ROM observation can briefly expose a later map variable or flag while a
# script is being torn down.  A later milestone is not coherent evidence on
# its own: these are the causal boundaries that must already be observed (or
# have been durably confirmed) before a high-water value can be promoted.
_CAMPAIGN_FACT_PREREQUISITES = {
    "wall_clock_set": ("new_game_setup_complete",),
    "rival_met": ("new_game_setup_complete",),
    "birch_rescued": ("rival_met",),
    "starter_obtained": ("birch_rescued",),
    "intro_rival_battle_complete": ("starter_obtained",),
    "pokedex_received": ("intro_rival_battle_complete",),
    "pokeballs_received": ("pokedex_received",),
    "visited_petalburg": ("pokedex_received",),
    "petalburg_wally_scene_complete": ("visited_petalburg",),
    "petalburg_woods_scene_complete": ("petalburg_wally_scene_complete",),
    "visited_rustboro": ("petalburg_woods_scene_complete",),
    "devon_goods_stolen": ("visited_rustboro",),
    "devon_goods_reported": ("devon_goods_stolen",),
    "devon_goods_recovered": ("devon_goods_stolen",),
    "devon_goods_returned": ("devon_goods_recovered",),
    "devon_goods_delivered": ("devon_goods_returned",),
    "devon_corp_3f_scene_complete": ("devon_goods_returned",),
    "first_badge_obtained": ("visited_rustboro",),
}

# Require two controller observations. This is intentionally small: the
# controller is refreshed at event boundaries, not as a second emulator
# simulation, and a genuine save-backed milestone remains cheap to confirm.
_CAMPAIGN_FACT_CONFIRMATION_OBSERVATIONS = 2
_CAMPAIGN_FACTS_REQUIRING_CONFIRMATION = frozenset(
    {
        "birch_rescued",
        "intro_rival_battle_complete",
        "pokedex_received",
        "pokeballs_received",
        "visited_petalburg",
        "petalburg_wally_scene_complete",
        "petalburg_woods_scene_complete",
        "visited_rustboro",
        "devon_goods_stolen",
        "devon_goods_reported",
        "devon_goods_recovered",
        "devon_goods_returned",
        "devon_goods_delivered",
        "devon_corp_3f_scene_complete",
        "first_badge_obtained",
    }
)

_RESTOCK_OBJECTIVE_IDS = frozenset(
    {
        "restock_pokeballs",
        "restock_recovery_items",
    }
)

_RESTOCK_TRANSACTION_PHASES = frozenset(
    {
        "planned",
        "traveling",
        "shop_interaction",
        "purchase_complete",
        "mart_exit_complete",
        "recovery_pending",
        "recovery_active",
    }
)

# Preparation is a bounded resource decision.  Once the party is one level
# above the strongest known member of an unavoidable trainer, another pass
# through wild grass is no longer a justified default response to an
# unproven horizon.  The battle planner still owns the actual fight and may
# report the remaining risk at the engagement boundary.
_TRAINER_PREPARATION_OVERLEVEL_MARGIN = 1

# Encounter-stage route discovery is allowed to fail closed, but a blocked
# stage must still get a chance to observe a changed map/trainer boundary.
# Bound the retry cadence so the same expensive AVOID/IGNORE probes do not
# monopolize the frame loop while the route remains unchanged.
_ENCOUNTER_STAGE_RETRY_DELAY_MAX = 64

# Emerald installs these tasks before the battle callback changes the coarse
# game-state value.  During that window the avatar is already owned by the
# battle listener, but an overworld readiness read only reports that the
# avatar is not controllable.  Treat the ROM transition as authoritative so
# campaign planning does not tear down its loop and rebuild the frontier once
# per emulator frame.
_BATTLE_TRANSITION_TASKS = frozenset(
    {
        "Task_BattleStart",
        "Task_BattleTransition",
        "Task_BattleTransition_Intro",
        "Task_BattleTransitionMain",
    }
)


def _campaign_battle_transition_active() -> tuple[bool, str | None]:
    """Return whether battle ROM state currently owns the emulator frame.

    ``GameState`` is the normal signal.  The transition tasks are also needed
    because Emerald can leave the coarse state as ``OVERWORLD`` while its
    battle-start callback and transition animation are already active.  This
    probe is intentionally limited to cheap cached memory/task reads; a full
    overworld or interaction observation here would recreate the stutter it
    is meant to prevent.
    """

    try:
        game_state = get_game_state()
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        game_state = None
    if game_state in {
        GameState.BATTLE,
        GameState.BATTLE_STARTING,
        GameState.BATTLE_ENDING,
    }:
        return True, f"game_state:{getattr(game_state, 'name', game_state)}"

    try:
        active_tasks = {
            getattr(task, "symbol", task)
            for task in (get_tasks() or ())
        }
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        active_tasks = set()
    transition_tasks = active_tasks.intersection(_BATTLE_TRANSITION_TASKS)
    if transition_tasks:
        return True, f"tasks:{','.join(sorted(transition_tasks))}"
    return False, None


def _is_restock_interior_map(map_id: Any) -> bool:
    """Return whether ``map_id`` is a cataloged ordinary purchase interior."""

    try:
        from .emerald_pokeball_catalog import emerald_pokeball_sources

        normalized_map_id = getattr(map_id, "value", map_id)
        return any(
            source.source_type == "pokemart"
            and source.interior_map.value == normalized_map_id
            for source in emerald_pokeball_sources()
        )
    except (AttributeError, ImportError, TypeError, ValueError):
        return False


def _restock_owns_mart_transaction(
    execution: CampaignExecutionResult,
    readiness: ProgressionReadinessDiagnostic,
) -> bool:
    """Return whether the mounted restock capability has reached a Mart."""

    return (
        execution.objective is not None
        and execution.objective.objective_id in _RESTOCK_OBJECTIVE_IDS
        and execution.capability is not None
        and _is_restock_interior_map(getattr(readiness, "current_map", None))
    )


def _restock_transaction_is_active() -> bool:
    """Return whether the restock capability still owns its transaction."""

    return getattr(context, "campaign_restock_phase", None) in _RESTOCK_TRANSACTION_PHASES


@lru_cache(maxsize=1)
def _canonical_campaign_objective_ranks() -> dict[str, int]:
    """Return the declarative order used to arbitrate transient ownership.

    The planner is allowed to report an earlier objective briefly while a
    battle or field script is being torn down.  Ownership may move forward
    across that boundary, but an earlier objective must never take ownership
    away from a later one.  Keep this lookup local to the controller so
    custom objective lists used by embedders/tests retain the historical
    fallback behavior below.
    """
    from .campaign_objectives import initial_emerald_campaign

    return {objective.objective_id: index for index, objective in enumerate(initial_emerald_campaign())}


def _retention_order_allows(active_id: str, selected_id: str) -> bool:
    """Return whether the active objective may remain during a handoff."""

    # These two resource objectives share one Emerald Mart baseline
    # transaction. A medicine purchase can make the Poké Ball objective
    # visible (and vice versa) before the live multi-item shop generator has
    # finished. Keep ownership with that generator until it returns; replacing
    # it leaves the ROM in its purchase-confirmation task and strands the next
    # item in the plan.
    if {active_id, selected_id} == _RESTOCK_OBJECTIVE_IDS:
        return True
    # A resource shortfall is a genuine interruption, not a transient story
    # regression. Do not retain an already-mounted route objective over the
    # dynamically-created restock task.
    if selected_id in _RESTOCK_OBJECTIVE_IDS:
        return False
    # Encounter objectives are spatial and their executor owns a concrete
    # destination. Once the planner selects a different encounter after a
    # map transition, retaining the old one can leave its nested stage route
    # active while the campaign frontier names the new map. Let the new
    # encounter mount with a fresh ROM-backed route instead.
    if active_id.startswith("obtain_encounter:") and selected_id.startswith("obtain_encounter:"):
        return False
    ranks = _canonical_campaign_objective_ranks()
    active_rank = ranks.get(active_id)
    selected_rank = ranks.get(selected_id)
    if active_rank is None or selected_rank is None:
        # Preserve support for custom objective sequences, which have no
        # canonical campaign order to compare.
        return True
    return active_rank > selected_rank


def _spatial_observation_is_coherent(state: CampaignState) -> bool:
    """Return whether the current save exposes a usable field position.

    Battle start/return and scripted map transitions temporarily remove the
    player projection even though save-backed campaign facts are still being
    read.  Those frames are not evidence that a later active objective has
    regressed; they are an observation boundary at which its generator must
    remain mounted.  A coherent map and coordinate pair, on the other hand,
    is required before allowing this exception to retain ownership across a
    real save regression.
    """
    return state.raw_map.is_known and state.coordinates.is_known


_TRANSIENT_READINESS_REASONS = frozenset(
    {
        ReadinessReason.OVERWORLD_UNAVAILABLE,
        ReadinessReason.RESOURCE_INFORMATION_UNKNOWN,
        ReadinessReason.PARTY_INFORMATION_UNKNOWN,
        ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN,
    }
)

# Recovery reads several ROM-owned structures immediately after a battle or
# map transition.  A partially replaced table can raise KeyError for a frame,
# but blindly retrying forever would recreate the original stuck safety loop.
_RECOVERY_KEYERROR_RETRY_LIMIT = 3
_RECOVERY_FAILURE_ESCALATION_LIMIT = 3

# ``None`` means that the level-cap rule is disabled or that no leader cap is
# currently active.  A distinct sentinel is needed for the live case where
# the rule is enabled but the ROM facts needed to resolve its cap are not
# available yet; preparation must defer in that case rather than guessing.
_PREPARATION_LEVEL_CAP_UNAVAILABLE = object()
_PREPARATION_LEVEL_CAP_UNSET = object()


def _readiness_requires_recheck(readiness: ProgressionReadinessDiagnostic) -> bool:
    """Return whether an UNKNOWN result is expected to change next frame."""

    return getattr(readiness, "readiness_reason", None) in _TRANSIENT_READINESS_REASONS


def _campaign_trainer_lead_rotation_active() -> bool:
    """Return whether a campaign trainer lead owns the party menu."""

    return getattr(context, "campaign_trainer_lead_rotation", None) is not None


def _actionable_scripted_interaction_is_active() -> bool:
    """Return whether a ROM-owned field interaction currently owns the frame.

    The controller may observe the first native wait of a trainer scene before
    the field-message printer has reached its A/B boundary.  That wait is not
    itself actionable, but it still owns the emulator frame.  Treating only
    actionable dialogue as an ownership boundary lets readiness invalidate the
    tactical loop one frame too early, so the later dialogue input never gets
    delegated to the agent controller.
    """

    # The campaign controller can remain resident while another mode or a
    # test harness owns the shared emulator.  In Manual mode a stale emulator
    # script must not be allowed to make a campaign tactical loop look like a
    # script-owned capability boundary.
    if getattr(context, "bot_mode", "Manual") == "Manual":
        return False
    try:
        # This probe is used while readiness is already handling an incomplete
        # overworld observation.  Calling observe_agent() here performs a
        # second full overworld read and can block on a transient map-boundary
        # coordinate (for example Route 103's (9, -1)).  The decision only
        # needs the interaction classifier, so keep it on the cheap ROM task
        # read and leave map perception to the tactical owner on the next
        # frame.
        interaction = observe_interaction()
        if classify_interaction(interaction) in {
            InteractionType.DIALOGUE,
            InteractionType.CHOICE,
            InteractionType.MENU,
            InteractionType.SPECIAL_INTERACTION,
        }:
            return True
        # Preserve the mounted campaign loop through the non-actionable part
        # of a ROM script handoff as well.  Battle ownership is handled by the
        # dedicated battle transition guard and must never be claimed here.
        return (
            bool(getattr(interaction, "script_active", False))
            and getattr(interaction, "game_state", None)
            not in {GameState.BATTLE, GameState.BATTLE_STARTING, GameState.BATTLE_ENDING}
            and getattr(interaction, "interaction_phase", InteractionPhase.NONE) is not InteractionPhase.NONE
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return False


def _navigation_transition_probe(loop: Any) -> bool:
    """Safely query transition ownership from a mounted loop or test double."""

    try:
        probe = getattr(loop, "navigation_transition_active", None)
        if callable(probe):
            return bool(probe())
        if probe is not None:
            return bool(probe)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return False


def _campaign_target_for_execution(objective, tactical_goal: Goal | None) -> SemanticTarget | None:
    """Resolve the GUI target for a newly mounted campaign execution.

    Tactical executors are allowed to publish more specific status as they
    observe the ROM.  This initial projection is nevertheless important at a
    handoff: encounter executors do not otherwise publish a status, and a
    completed recovery status must not remain attached to their next route.
    """

    if isinstance(tactical_goal, NavigationGoal) and isinstance(tactical_goal.target, SemanticTarget):
        return tactical_goal.target
    if isinstance(tactical_goal, NavigationGoal) and isinstance(tactical_goal.target, EngageTrainer):
        destination = getattr(objective, "destination", None)
        if isinstance(destination, tuple) and len(destination) == 2:
            return SemanticTarget.interaction(destination[0], tactical_goal.target.trainer_id)
    if isinstance(tactical_goal, ReachLocation):
        return SemanticTarget.at(tactical_goal.location)
    if isinstance(tactical_goal, ReachWarp):
        target_map = tactical_goal.destination_map
        if target_map is None and tactical_goal.destination is not None:
            target_map = tactical_goal.destination[0]
        return SemanticTarget.map(target_map) if target_map is not None else None
    if isinstance(tactical_goal, ActivateTrigger):
        target_map = None
        destination = getattr(objective, "destination", None)
        if isinstance(destination, tuple) and len(destination) == 2:
            target_map = destination[0]
        return SemanticTarget.interaction(target_map, tactical_goal.trigger_id)
    if isinstance(tactical_goal, EngageTrainer):
        destination = getattr(objective, "destination", None)
        if isinstance(destination, tuple) and len(destination) == 2:
            return SemanticTarget.interaction(destination[0], tactical_goal.trainer_id)
    destination = getattr(objective, "destination", None)
    if isinstance(destination, tuple) and len(destination) == 2:
        return SemanticTarget.map(destination)
    return None


def _is_trainer_goal(goal: Goal | None) -> bool:
    """Return whether a tactical goal deliberately owns a trainer checkpoint."""

    return isinstance(goal, EngageTrainer) or (
        isinstance(goal, NavigationGoal)
        and isinstance(getattr(goal, "target", None), EngageTrainer)
    )


def _objective_requires_trainer_battle(objective) -> bool:
    """Return whether an objective owns a mandatory trainer encounter.

    Some story encounters are represented by a generic ``ActivateTrigger``
    until route-aware readiness promotes them to ``EngageTrainer``. The
    recovery gate must still protect those objectives during that earlier
    representation.
    """

    return bool(
        getattr(getattr(objective, "resource_policy", None), "mandatory_battle", False)
        or getattr(objective, "objective_id", None) == "complete_intro_rival"
    )


def _story_progression_owns_restock_boundary(
    objective,
    *,
    refresh_reason: str | None,
) -> bool:
    """Return whether an active required scene must retain controller ownership."""

    objective_id = getattr(objective, "objective_id", None)
    if (
        objective is None
        or getattr(objective, "task_kind", None) != "required"
        or objective_id in _RESTOCK_OBJECTIVE_IDS
        or (isinstance(objective_id, str) and objective_id.startswith("obtain_encounter:"))
    ):
        return False
    # Battle completion is an explicit boundary from the battle listener. The
    # story capability remains the owner until its ROM completion fact is
    # visible, even if the field script is quiet for that one frame.
    if refresh_reason == "battle_ended":
        return True
    battle_active, _ = _campaign_battle_transition_active()
    return battle_active or _actionable_scripted_interaction_is_active()


def _trainer_preparation_ceiling_reached(readiness) -> tuple[bool, int | None, int | None]:
    """Return whether current preparation has reached the trainer-level ceiling."""

    checkpoint = getattr(readiness, "trainer_checkpoint", None)
    trainer_levels = tuple(
        int(getattr(member, "level"))
        for member in getattr(checkpoint, "party", ())
        if getattr(member, "level", None) is not None
    )
    if not trainer_levels:
        return False, None, None
    try:
        from modules.pokemon_party import get_party

        party_levels = tuple(
            int(getattr(pokemon, "level"))
            for pokemon in get_party()
            if not getattr(pokemon, "is_egg", False)
            and getattr(pokemon, "current_hp", 0) > 0
            and getattr(pokemon, "level", None) is not None
        )
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return False, None, max(trainer_levels) + _TRAINER_PREPARATION_OVERLEVEL_MARGIN
    if not party_levels:
        return False, None, max(trainer_levels) + _TRAINER_PREPARATION_OVERLEVEL_MARGIN
    ceiling = max(trainer_levels) + _TRAINER_PREPARATION_OVERLEVEL_MARGIN
    current_level = max(party_levels)
    return current_level >= ceiling, current_level, ceiling


def _readiness_requires_hard_recovery(readiness, policy: CampaignReadinessPolicy) -> bool:
    """Return whether unfinished recovery work is a safety boundary."""

    reason = getattr(getattr(readiness, "readiness_reason", None), "value", None)
    if reason in {
        ReadinessReason.CRITICAL_PARTY_HP.value,
        ReadinessReason.NO_USABLE_POKEMON.value,
        ReadinessReason.FAINTED_PARTY_MEMBER.value,
        ReadinessReason.POISONED_PARTY.value,
        ReadinessReason.STATUS_CONDITION.value,
    }:
        return True
    if getattr(readiness, "usable_count", None) == 0 or getattr(readiness, "fainted_count", None):
        return True
    for member in getattr(readiness, "party", ()) or ():
        if getattr(member, "fainted", False):
            return True
        status = str(getattr(getattr(member, "status", None), "value", getattr(member, "status", "none")))
        if status.casefold().replace(" ", "_") in {"poison", "poisoned", "bad_poison", "badly_poisoned"}:
            return True
    lowest_hp_ratio = getattr(readiness, "lowest_hp_ratio", None)
    if isinstance(lowest_hp_ratio, (int, float)) and not isinstance(lowest_hp_ratio, bool):
        if lowest_hp_ratio < policy.critical_hp_ratio_for(
            getattr(readiness, "objective_id", None), getattr(readiness, "current_map", None)
        ):
            return True
        if (
            getattr(readiness, "needs_status_recovery", False)
            and lowest_hp_ratio < policy.status_hp_ratio_for(getattr(readiness, "objective_id", None))
        ):
            return True
    return False


def _normalized_campaign_location(location):
    """Normalize map enums and coordinates for recovery-origin checks."""

    if not isinstance(location, tuple) or len(location) != 2:
        return location
    map_id = getattr(location[0], "value", location[0])
    coordinates = location[1]
    if isinstance(coordinates, list):
        coordinates = tuple(coordinates)
    return map_id, coordinates


def _campaign_locations_match(left, right) -> bool:
    """Return whether two map/coordinate pairs identify the same tile."""

    return _normalized_campaign_location(left) == _normalized_campaign_location(right)


def _post_pokedex_campaign_boundary(state: CampaignState) -> bool:
    """Return whether the current save is past the overworld safety boundary.

    Opening capabilities such as title-screen setup and starter selection are
    intentionally targetless and cannot supply meaningful overworld readiness
    input.  Once the Pokédex has been received, however, targetless Emerald
    capabilities include ordinary route traversal (for example
    ``complete_petalburg_wally``).  Those capabilities
    still need the event-driven post-battle/map readiness check.
    """

    facts = getattr(state, "campaign_facts", None)
    pokedex = getattr(facts, "pokedex_received", None)
    return getattr(pokedex, "status", None) is FactStatus.KNOWN and getattr(pokedex, "value", None) is True


class _ControllerExceptionReporter:
    """Report swallowed controller exceptions in the normal log, rate limited.

    The controller deliberately turns capability exceptions into a retry.  The
    decisive error used to be written only with ``--debug-trace``, so an
    ordinary run showed "failed; retrying" with no cause.  Identical errors are
    reported on their 1st, 2nd, 3rd, 10th, 100th, and every 500th occurrence.
    """

    _REPORT_AT = frozenset({1, 2, 3, 10, 100})

    def __init__(self) -> None:
        self._counts: dict[tuple, int] = {}

    def should_report(self, kind: str, error: BaseException) -> tuple[bool, int]:
        key = (kind, type(error).__name__, str(error)[:200])
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        return (count in self._REPORT_AT or count % 500 == 0), count


class _ControllerIdleWatchdog:
    """Detect a controller that sits in one state without a mounted capability."""

    def __init__(self, *, first_report_after: int = 600, repeat_every: int = 3600) -> None:
        self._first = first_report_after
        self._repeat = repeat_every
        self._signature: object = None
        self._count = 0

    def observe(self, signature: object, *, idle: bool) -> int | None:
        """Return the idle frame count when it should be reported, else None."""

        if not idle:
            self._signature = None
            self._count = 0
            return None
        if signature != self._signature:
            self._signature = signature
            self._count = 0
        self._count += 1
        if self._count == self._first or (
            self._count > self._first and (self._count - self._first) % self._repeat == 0
        ):
            return self._count
        return None


# A mounted capability that issues no input leaves the idle watchdog silent (it
# only counts steps with nothing mounted).  This one watches real elapsed time
# while a capability is mounted but the phase, objective, goal, status, and the
# player's map position are all unchanged.  It is wall-clock time, not frames: an
# unthrottled emulator runs thousands of frames per second, and a long intro,
# menu, or cutscene legitimately leaves the position unchanged for a while.
_STALL_FIRST_REPORT_AFTER_SECONDS = 180.0
_STALL_REPEAT_EVERY_SECONDS = 600.0
_PLAYER_SAMPLE_EVERY = 60


class _ProgressWatchdog:
    """Report when a signature has not changed for a long stretch of real time."""

    def __init__(self, *, first_report_after: float, repeat_every: float, clock=monotonic) -> None:
        self._first = first_report_after
        self._repeat = repeat_every
        self._clock = clock
        self._signature: object = None
        self._since: float | None = None
        self._next_report = 0.0

    def observe(self, signature: object, *, active: bool) -> float | None:
        """Return the seconds without change when it should be reported, else None."""

        if not active:
            self._signature = None
            self._since = None
            return None
        now = self._clock()
        if self._since is None or signature != self._signature:
            self._signature = signature
            self._since = now
            self._next_report = now + self._first
            return None
        if now < self._next_report:
            return None
        self._next_report = now + self._repeat
        return now - self._since


def _sample_player_location():
    """Return ``(map, coordinates)`` or ``None`` when the avatar is not readable."""

    try:
        from modules.player import get_player_location

        location = get_player_location()
        return (str(location[0]), tuple(location[1]))
    except Exception:  # an unreadable avatar is "unknown", never a controller failure
        return None


_CONTROLLER_EXCEPTIONS = _ControllerExceptionReporter()


def _report_controller_exception(kind: str, error: BaseException, **fields) -> None:
    """Write a compact, rate-limited exception record to the normal log."""

    should, count = _CONTROLLER_EXCEPTIONS.should_report(kind, error)
    if not should:
        return
    frames = traceback.extract_tb(error.__traceback__)[-3:] if error.__traceback__ else []
    where = " <- ".join(f"{frame.name}:{frame.lineno}" for frame in reversed(frames)) or "unknown"
    extra = " ".join(f"{name}={value!r}" for name, value in fields.items())
    print_always(
        lambda: (
            f"CAMPAIGN_CONTROLLER_EXCEPTION: kind={kind!r} occurrence={count} "
            f"error_type={type(error).__name__!r} error={str(error)[:300]!r} where={where!r} {extra}"
        ),
    )


class CampaignController:
    """Select and mount one tactical goal, reevaluating at event boundaries."""

    def __init__(
        self,
        state_provider: Callable[[], CampaignState],
        *,
        tactical_loop_factory: TacticalLoopFactory | None = None,
        # Campaign Progression must use the dependency frontier: optional
        # encounter objectives can be strategic prerequisites for required
        # story progression. The legacy ordered selector cannot discover them.
        selector: Callable[[CampaignState], ObjectiveSelection] = select_available_campaign_task,
        adapter: Callable[[ObjectiveSelection], CampaignExecutionResult] = adapt_campaign_execution,
        campaign_boundary_handler: CampaignBoundaryHandler | None = None,
        readiness_provider: ReadinessProvider | None = None,
        recovery_factory: RecoveryFactory | None = None,
        readiness_policy: CampaignReadinessPolicy | None = None,
    ) -> None:
        """Create a controller around state, selection, and execution adapters."""

        self._state_provider = state_provider
        self._selector = selector
        self._adapter = adapter
        self._campaign_boundary_handler = campaign_boundary_handler
        self._tactical_loop_factory = tactical_loop_factory or self._default_tactical_loop
        self.current_objective_id: str | None = None
        self.current_tactical_goal: Goal | None = None
        self.last_selection: ObjectiveSelection | None = None
        self.last_execution: CampaignExecutionResult | None = None
        self.status = CampaignControllerStatus.UNKNOWN
        self.transition_reason = "not started"
        self._tactical_loop: Iterator[object] | None = None
        self._tactical_loop_objective_id: str | None = None
        self._readiness_provider = readiness_provider
        self._recovery_factory = recovery_factory
        self._readiness_policy = readiness_policy or CampaignReadinessPolicy()
        self._execution_phase = "CAMPAIGN"
        self._recovery_status: str | None = None
        # Once readiness accepts recovery for an objective, keep that
        # decision authoritative across the mount boundary. A planner refresh
        # must not replace it with the parent trainer/encounter route while
        # the recovery worker or generator is still being installed.
        self._accepted_recovery_objective_id: str | None = None
        self._recovery_keyerror_retries = 0
        self._recovery_failure_signature = None
        self._recovery_failure_repeats = 0
        self._trainer_preparation_checkpoint_key = None
        self._trainer_preparation_last_target = None
        # A capability may have been selected before route analysis resolved a
        # mandatory trainer. Retain the promoted trainer goal across that
        # calculation boundary so the capability cannot remount and walk into
        # the checkpoint while its roster is still unknown.
        self._pending_trainer_handoff: tuple[str, Goal] | None = None
        self.interrupted_objective_id: str | None = None
        # A readiness check can land on the same frame as battle/script
        # teardown.  Keep the recheck obligation separate from the scheduler
        # cache so a deferred recovery decision cannot be lost at that
        # boundary.
        self._readiness_recheck_pending = False
        self._pending_readiness_objective_id: str | None = None
        self._deferred_handoff_requires_refresh = False
        # Advisory route work may be polled while the current tactical loop
        # continues to own the emulator. Hard recovery and trainer gates use
        # the existing recovery-wait phase instead.
        self._advisory_route_polling = False
        # A wild battle can prove that the mounted preparation route is no
        # longer executable: the battle policy may choose Run Away because
        # the party is below its training margin.  Retain that recovery
        # handoff across the battle-return boundary so pending route analysis
        # cannot remount the same encounter search.
        self._wild_battle_recovery_pending = False
        # A blocked encounter stage has no executable tactical loop. Keep its
        # refresh obligation explicit: pending recovery work must be polled,
        # while an unchanged blocked route is retried with bounded backoff.
        self._encounter_stage_retry_signature = None
        self._encounter_stage_retry_failures = 0
        self._encounter_stage_retry_delay = 0
        # A capability can observe its completion from the live rules
        # projection just before the next controller refresh can see that
        # projection. Keep the completed objective as a one-boundary fence so
        # the old tactical capability cannot be remounted from a stale
        # selection cache during that handoff.
        self._completed_capability_objective = None
        # Campaign selection and adaptation are boundary work.  Keep the
        # mounted tactical generator on the hot path and only rebuild the
        # campaign envelope when an event invalidates it.
        self._refresh_required = True
        self._refresh_required_reason = "initial"
        self._last_readiness_boundary_key = None
        self._campaign_plan: CampaignPlan | None = None
        self._last_published_campaign_status: CampaignStatus | None = None
        self._campaign_snapshot_diagnostics_key = None
        # Task diagnostics contain route/encounter discovery and are only a
        # trace aid.  Keep them on the same semantic signature as campaign
        # planning so enabling tracing cannot turn every frame into a second
        # planner pass.
        self._campaign_task_diagnostics_key = None
        self._confirmed_campaign_facts: set[str] = set()
        self._pending_campaign_fact_confirmations: dict[str, int] = {}
        # A save-state load starts a new observed runtime session. The
        # high-water guard must not carry later story milestones backwards
        # into an earlier save state.
        self._confirmed_campaign_session_id: str | None = None
        # Suppress repeated transition diagnostics while the battle listener
        # owns a multi-frame ROM animation. The ownership probe itself remains
        # cheap and runs before any readiness or frontier work.
        self._battle_transition_signature: str | None = None

    @staticmethod
    def _trainer_id_for_goal(goal: Goal) -> str | None:
        target = goal.target if isinstance(goal, NavigationGoal) else goal
        return target.trainer_id if isinstance(target, EngageTrainer) else None

    def _trainer_map_for_goal(self, goal: Goal, trainer_id: str) -> Any | None:
        checkpoint = trainer_checkpoint_for_id(trainer_id)
        if checkpoint is not None:
            return checkpoint.map_id
        objective = getattr(self.last_selection, "objective", None)
        destination = getattr(objective, "destination", None)
        if isinstance(destination, tuple) and len(destination) == 2:
            if isinstance(destination[1], tuple):
                return destination[0]
            return destination
        return None

    @staticmethod
    def _trainer_staging_location(
        world: NavigationWorld,
        trainer_id: str,
        engagement_plan,
        *,
        start=None,
        constraints=None,
    ):
        """Choose the best reachable safe tile before trainer exposure.

        Exposure geometry produces several valid adjacent tiles.  Sorting
        those coordinates is not a route decision: a live trainer can block
        one candidate or divide the map so that only another candidate is
        reachable under ``TrainerMode.AVOID``.  Resolve all candidates from
        the current world in one shared search and rank only the routes that
        actually exist.
        """

        matching = tuple(
            trigger
            for trigger in getattr(world, "triggers", ())
            if getattr(trigger, "affordance_id", None) == trainer_id
        )
        hazards = frozenset(
            location
            for trigger in matching
            for location in getattr(trigger, "hazard_locations", ()) or ()
        )
        if engagement_plan.mode is TrainerEngagementMode.INTERACTION:
            locations = tuple(getattr(engagement_plan, "interaction_locations", ()) or ())
            return sorted(locations, key=repr)[0] if locations else None
        if engagement_plan.mode is not TrainerEngagementMode.SIGHTLINE:
            return None

        engagement_locations = frozenset(getattr(engagement_plan, "engagement_locations", ()) or ())
        if not engagement_locations:
            return None
        candidates = []
        for target in sorted(engagement_locations, key=repr):
            if target in hazards:
                # Standing on an activation tile is acceptable as the final
                # boundary, but it is not a safe staging tile for a sightline
                # that triggers on entry.
                pass
            x, y = target[1]
            for candidate in (
                (target[0], (x - 1, y)),
                (target[0], (x + 1, y)),
                (target[0], (x, y - 1)),
                (target[0], (x, y + 1)),
            ):
                if candidate in hazards or candidate not in world.tiles:
                    continue
                tile = world.tiles.get(candidate)
                if tile is None or tile.blocked:
                    continue
                if any(destination == target and not is_warp for _direction, destination, is_warp in world.neighbors(candidate)):
                    candidates.append(candidate)
        candidates = tuple(sorted(set(candidates), key=repr))
        if not candidates:
            return None
        if not isinstance(world, NavigationWorld):
            # Compatibility fixtures and non-live callers may expose only
            # geometry.  The live campaign always supplies NavigationWorld;
            # retain the old geometry-only seam when route metrics cannot be
            # computed from the caller's lightweight object.
            return candidates[0]
        if start is None:
            # Preserve the pure geometry seam for callers that do not have a
            # live origin.  The campaign exposure loop always supplies start
            # and therefore takes the route-aware branch below.
            return candidates[0]

        try:
            diagnostic_print(
                lambda: (
                    "NAVIGATION_REQUEST: caller='campaign_controller.trainer_staging' "
                    f"start={start!r} trainer={trainer_id!r} targets={candidates!r}"
                ),
                trace=True,
                prefix="NAVIGATION_REQUEST",
            )
            plans = GoalAwareNavigator(world).plan_many_locations(
                start,
                candidates,
                encounter_mode=EncounterMode.AVOID,
                constraints=constraints or GoalConstraints(trainer_mode=TrainerMode.AVOID),
            )
        except (NavigationError, NavigationSearchLimitExceeded, RuntimeError, TypeError, ValueError) as error:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_STAGING: "
                    f"trainer={trainer_id!r} state='route_search_failed' "
                    f"start={start!r} candidates={candidates!r} "
                    f"exception_type={type(error).__name__!r} error={error!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_STAGING",
            )
            return None

        ranked = tuple(
            sorted(
                plans.items(),
                key=lambda item: (
                    getattr(getattr(item[1], "metrics", None), "encounter_opportunities", float("inf")),
                    getattr(getattr(item[1], "metrics", None), "total_route_cost", float("inf")),
                    getattr(getattr(item[1], "metrics", None), "movement_actions", float("inf")),
                    repr(item[0]),
                ),
            )
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_TRAINER_STAGING: "
                f"trainer={trainer_id!r} state='route_candidates' start={start!r} "
                f"candidates={candidates!r} reachable={tuple(location for location, _plan in ranked)!r} "
                f"selected={ranked[0][0] if ranked else None!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_TRAINER_STAGING",
        )
        return ranked[0][0] if ranked else None

    def _trainer_exposure_tactical_loop(self, goal: Goal) -> Iterator[object]:
        """Own the final approach, lead rotation, and trainer engagement."""

        trainer_id = self._trainer_id_for_goal(goal)
        if trainer_id is None:
            return

        target_map = self._trainer_map_for_goal(goal, trainer_id)
        target_map = getattr(target_map, "value", target_map)
        constraints = getattr(goal, "constraints", GoalConstraints())
        staging_constraints = replace(
            constraints,
            trainer_mode=TrainerMode.AVOID,
            avoid_trainer_ids=frozenset({trainer_id}),
            allow_forced_trainer_exposure=False,
        )

        if target_map is None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_EXPOSURE: "
                    f"trainer={trainer_id!r} state='fallback_direct' "
                    "reason='trainer map unavailable from checkpoint or objective'"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_EXPOSURE",
            )
            yield from AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()
            return

        encounter_mode = getattr(goal, "encounter_mode", EncounterMode.AVOID)

        # Reach only the trainer's map first.  The final EngageTrainer route
        # is deliberately withheld until the local exposure geometry and the
        # readiness-selected lead have both been established.
        staging_search_signature = None
        staging_location = None
        while True:
            observation = observe_agent()
            world = getattr(observation, "overworld", None)
            if world is None or not getattr(world, "controllable", False):
                yield
                continue
            current = (getattr(world.map_id, "value", world.map_id), world.player_coordinates)
            if current[0] != target_map:
                map_goal = NavigationGoal(
                    SemanticTarget.map(target_map),
                    constraints=staging_constraints,
                    encounter_mode=encounter_mode,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_EXPOSURE: "
                        f"trainer={trainer_id!r} state='approach_map' current={current!r} target_map={target_map!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_EXPOSURE",
                )
                yield from AgentControlLoop(
                    lambda: observe_agent(goal=map_goal),
                    goal=map_goal,
                ).run()
                continue

            navigation_world = NavigationWorld.from_overworld(world)
            engagement_plan = resolve_trainer_engagement_plan(navigation_world, trainer_id)
            revision_getter = getattr(navigation_world, "revision_key", None)
            world_revision = revision_getter() if callable(revision_getter) else None
            staging_signature = (
                current,
                world_revision,
                repr(engagement_plan),
            )
            # The selected tile is a handoff boundary.  Once the staging
            # route has delivered the avatar there, do not rerun candidate
            # selection from that new origin before the lead owner takes
            # control.  Re-ranking at this point can choose a second tile,
            # launch another route search, and leave the trainer exposure
            # owner waiting indefinitely at a valid staging position.
            staging_reached = staging_location is not None and current == staging_location
            if staging_reached:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_EXPOSURE: "
                        f"trainer={trainer_id!r} state='staging_reached' "
                        f"location={staging_location!r} action='handoff_to_lead_owner'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_EXPOSURE",
                )
            elif staging_signature != staging_search_signature:
                staging_search_signature = staging_signature
                staging_location = self._trainer_staging_location(
                    navigation_world,
                    trainer_id,
                    engagement_plan,
                    start=current,
                    constraints=staging_constraints,
                )
            if staging_location is None:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_EXPOSURE: "
                        f"trainer={trainer_id!r} state='deferred' current={current!r} "
                        f"mode={getattr(engagement_plan.mode, 'value', None)!r} "
                        f"reason={getattr(engagement_plan, 'reason', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_EXPOSURE",
                )
                yield
                continue

            staging_goal = NavigationGoal(
                ReachLocation(staging_location),
                constraints=staging_constraints,
                encounter_mode=encounter_mode,
            )
            if current != staging_location:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_EXPOSURE: "
                        f"trainer={trainer_id!r} state='approach_staging' current={current!r} "
                        f"staging={staging_location!r} mode={engagement_plan.mode.value!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_EXPOSURE",
                )
                yield from AgentControlLoop(
                    lambda: observe_agent(goal=staging_goal),
                    goal=staging_goal,
                ).run()
                continue

            readiness = getattr(context, "campaign_trainer_readiness", None)
            lead_index = getattr(readiness, "lead_party_index", None)
            self._publish_campaign_status(f"Rotating lead for {trainer_id}")
            rotated = yield from ensure_campaign_trainer_lead(
                lead_index,
                trainer_id=trainer_id,
                reason="final trainer exposure staging",
            )
            if not rotated:
                yield
                continue

            self._publish_campaign_status(f"Trainer lead ready; exposing {trainer_id}")
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_EXPOSURE: "
                    f"trainer={trainer_id!r} state='engage' staging={staging_location!r} "
                    f"lead_party_index={lead_index!r} engagement_mode={engagement_plan.mode.value!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_EXPOSURE",
            )
            yield from AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()
            return

    def _default_tactical_loop(self, goal: Goal) -> Iterator[object]:
        """Build the normal agent-control generator for a tactical goal.

        Recovery can finish at a Center before the parent route resumes. Keep
        the route composed before that stop available when the controller
        promotes the parent objective to a generic tactical goal, such as a
        prepared trainer encounter. Capability-backed objectives consume the
        same handoff in ``emerald_capabilities``; this is the corresponding
        path for the generic ``AgentControlLoop``.
        """

        navigation_plan = None
        handoff = getattr(context, "campaign_continuation_route", None)
        if isinstance(handoff, tuple) and len(handoff) == 2:
            handoff_objective_id, candidate = handoff
            trainer_goal_handoff = isinstance(goal, EngageTrainer) or (
                isinstance(goal, NavigationGoal)
                and isinstance(getattr(goal, "target", None), EngageTrainer)
            )
            if handoff_objective_id == self.current_objective_id and not trainer_goal_handoff:
                navigation_plan = candidate
                context.campaign_continuation_route = None
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_HANDOFF: generic_loop_adopted=True "
                        f"objective={self.current_objective_id!r} "
                        f"destination={getattr(candidate, 'destination', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_HANDOFF",
                )
            else:
                # A continuation route belongs to the objective that was
                # suspended. Never let a later objective consume it after a
                # transient frontier change.
                context.campaign_continuation_route = None
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_HANDOFF: generic_loop_discarded=True "
                        f"reason={'trainer_goal_requires_replan' if trainer_goal_handoff else 'objective_mismatch'!r} "
                        f"handoff_objective={handoff_objective_id!r} "
                        f"current_objective={self.current_objective_id!r} "
                        f"goal={goal!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_HANDOFF",
                )
        if _is_trainer_goal(goal):
            return self._trainer_exposure_tactical_loop(goal)
        return AgentControlLoop(
            lambda: observe_agent(goal=goal),
            goal=goal,
            navigation_plan=navigation_plan,
        ).run()

    @property
    def state(self) -> CampaignControllerState:
        """Return the controller's immutable externally visible state."""

        return CampaignControllerState(
            self.status,
            self.current_objective_id,
            self.current_tactical_goal,
            self.last_selection,
            self.last_execution,
            self.transition_reason,
            self._execution_phase,
            self._recovery_status,
            self._campaign_plan,
        )

    def _publish_campaign_status(self, intent: str) -> None:
        """Publish a fresh status at a campaign/recovery ownership boundary."""

        objective = self.last_selection.objective if self.last_selection is not None else None
        self._publish_status_for_execution(objective, self.current_tactical_goal, intent)

    def _publish_status_for_execution(self, objective, tactical_goal: Goal | None, intent: str) -> None:
        """Publish status for an execution before its tactical loop is mounted.

        Readiness is evaluated between objective selection and tactical-loop
        mounting. When that observation is deferred, the controller has no
        loop from which to publish status, while the previous capability may
        have left an obsolete GUI projection behind. Keep the projection tied
        to the selected execution so a deferred handoff is truthful without
        granting the execution permission to run.
        """

        context.campaign_preparation_target = None

        if objective is None:
            status = CampaignStatus(None, None, intent)
            if self._last_published_campaign_status == status:
                return
            self._last_published_campaign_status = status
            context.campaign_status = status
            self._emit_campaign_snapshot(status)
            diagnostic_print(
                lambda: f"CAMPAIGN_STATUS_BOUNDARY: objective=None target=None intent={intent!r}",
                trace=True,
                prefix="CAMPAIGN_STATUS_BOUNDARY",
            )
            return
        status = CampaignStatus(
            getattr(objective, "description", None) or objective.objective_id,
            _campaign_target_for_execution(objective, tactical_goal),
            intent,
            tactical_goal,
        )
        if self._last_published_campaign_status == status:
            return
        self._last_published_campaign_status = status
        context.campaign_status = status
        self._emit_campaign_snapshot(status)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_STATUS_BOUNDARY: "
                f"objective={status.objective!r} target={status.target!r} intent={status.intent!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_STATUS_BOUNDARY",
        )

    def _emit_campaign_snapshot(self, status: CampaignStatus) -> None:
        """Publish a deduplicated Rich campaign report at a status boundary."""

        runtime = getattr(context, "nuzlocke_runtime", None)
        snapshot = getattr(runtime, "latest_snapshot", None) if runtime is not None else None
        if snapshot is None:
            return
        level_cap = None
        boss = None
        try:
            from modules.nuzlocke.level_cap import assess_level_cap

            assessment = assess_level_cap(runtime_campaign_state().campaign_facts, snapshot.party)
            level_cap = assessment.level_cap
            boss = assessment.active_boss
        except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
            pass
        checkpoint = getattr(context, "campaign_trainer_checkpoint", None)
        trainer_readiness = getattr(context, "campaign_trainer_readiness", None)
        readiness = getattr(context, "campaign_readiness_diagnostic", None)
        trainer_facts = tuple(getattr(context, "campaign_trainer_facts", ()) or ())
        budget = getattr(context, "campaign_preparation_exp_budget", None)
        if budget is None and level_cap is not None:
            try:
                from modules.nuzlocke.trainer_preparation import build_exp_budget
                from modules.pokemon_party import get_party

                # Preparation stores the richer ledger, but ordinary campaign
                # boundaries should still show current cap headroom when the
                # party exposes the required EXP facts. Include a detected
                # unavoidable trainer so the boundary shows the reward that
                # must fit under the active cap even before preparation mounts.
                budget = build_exp_budget(
                    get_party(),
                    target_level=None,
                    level_cap=level_cap,
                    unavoidable_checkpoints=(checkpoint,) if checkpoint is not None else (),
                )
            except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
                budget = None
        key = (
            status,
            level_cap,
            getattr(boss, "boss_id", None),
            repr(budget),
            getattr(checkpoint, "checkpoint_id", None),
            repr(trainer_readiness),
            repr(readiness),
            tuple(getattr(fact, "checkpoint_id", None) for fact in trainer_facts),
            (
                getattr(getattr(snapshot, "player", None), "map_group", None),
                getattr(getattr(snapshot, "player", None), "map_number", None),
                getattr(getattr(snapshot, "player", None), "map_name", None),
            ),
            tuple(
                (
                    getattr(pokemon, "species", None),
                    getattr(pokemon, "level", None),
                    getattr(pokemon, "current_hp", None),
                    getattr(pokemon, "max_hp", None),
                    getattr(pokemon, "status", None),
                )
                for pokemon in getattr(snapshot, "party", ())
            ),
        )
        if key == self._campaign_snapshot_diagnostics_key:
            return
        self._campaign_snapshot_diagnostics_key = key
        print_campaign_snapshot(
            snapshot=snapshot,
            status=status,
            level_cap=level_cap,
            boss=boss,
            exp_budget=budget,
            trainer_checkpoint=checkpoint,
            trainer_facts=trainer_facts,
            trainer_readiness=trainer_readiness,
            readiness=readiness,
        )

    def _active_preparation_level_cap(self):
        """Resolve the hard cap that bounds trainer preparation targets.

        Preparation runs in the campaign controller, so it cannot safely
        infer a cap from the trainer checkpoint or from the party's current
        levels.  The runtime's materialized campaign facts are the authority
        for the active Emerald leader cap.  If that authority is incomplete,
        defer preparation until the next readiness boundary.
        """

        runtime = getattr(context, "nuzlocke_runtime", None)
        if runtime is None:
            return _PREPARATION_LEVEL_CAP_UNAVAILABLE
        try:
            from modules.nuzlocke.level_cap import assess_level_cap
            from modules.pokemon_party import get_party

            rule_config = getattr(runtime, "rule_config", None)
            if rule_config is None:
                return _PREPARATION_LEVEL_CAP_UNAVAILABLE
            if not rule_config.is_enabled(CampaignRuleId.LEVEL_CAP):
                return None

            assessment = assess_level_cap(runtime_campaign_state().campaign_facts, get_party())
            if assessment.status is not FactStatus.KNOWN:
                return _PREPARATION_LEVEL_CAP_UNAVAILABLE
            return assessment.level_cap
        except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError):
            return _PREPARATION_LEVEL_CAP_UNAVAILABLE

    def _trainer_preparation_parameters(
        self,
        readiness,
        *,
        level_cap=_PREPARATION_LEVEL_CAP_UNSET,
    ) -> tuple[Any, int] | None:
        """Return the ROM map and bounded target for a known trainer.

        A preparation request is only valid when the checkpoint is explicit.
        Unknown trainers must continue through the existing conservative
        battle policy instead of inventing an opponent roster or a training
        destination.
        """

        checkpoint = getattr(readiness, "trainer_checkpoint", None)
        party = tuple(getattr(checkpoint, "party", ()) or ()) if checkpoint is not None else ()
        training_map = getattr(checkpoint, "map_id", None) if checkpoint is not None else None
        levels = tuple(
            int(getattr(member, "level"))
            for member in party
            if getattr(member, "level", None) is not None
        )
        if training_map is None or not levels:
            return None
        if level_cap is _PREPARATION_LEVEL_CAP_UNSET:
            level_cap = self._active_preparation_level_cap()
        if level_cap is _PREPARATION_LEVEL_CAP_UNAVAILABLE:
            return None
        target_level = max(levels)
        if level_cap is not None:
            target_level = min(target_level, level_cap)
        return training_map, target_level

    def _terminal(self, execution: CampaignExecutionResult, status: CampaignControllerStatus) -> None:
        """Clear active execution and publish a terminal controller status."""

        diagnostic_print(
            lambda: (
                "CAMPAIGN_TACTICAL_LOOP_INVALIDATED: "
                f"controller_id={id(self)!r} loop_id={id(self._tactical_loop) if self._tactical_loop else None!r} "
                f"old_objective={self._tactical_loop_objective_id!r} new_objective=None reason='terminal'"
            ),
            trace=True,
        )
        self.current_objective_id = None
        self.current_tactical_goal = None
        self._tactical_loop = None
        self._execution_phase = "TERMINAL"
        if status is CampaignControllerStatus.RUN_LOST:
            self._recovery_status = "RUN_LOST"
        self._pending_readiness_objective_id = None
        self._readiness_recheck_pending = False
        self.last_execution = execution
        self.status = status
        self.transition_reason = execution.reason
        diagnostic_print(
            lambda: (
                "CAMPAIGN_STATUS_TRANSITION: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"status={status.value!r} reason={execution.reason!r} terminal=True"
            ),
            trace=True,
        )

    def _emit_campaign_task_diagnostics(self, observed_state: CampaignState, selection: ObjectiveSelection) -> None:
        """Emit one panel for each semantic frontier, not each state change."""

        try:
            completed, available = campaign_frontier_summary(
                observed_state,
                executable_objective_ids=getattr(selection, "frontier_objective_ids", None) or None,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Custom controller embeddings may intentionally expose only the
            # facts needed by their selector. Diagnostics must not turn that
            # compatibility seam into a campaign refresh failure.
            return
        key = (
            completed,
            available,
            getattr(selection.objective, "objective_id", None),
            selection.status.value,
            selection.reason,
            repr(getattr(selection, "selection_diagnostics", ())),
            repr(getattr(selection, "encounter_portfolio", ())),
        )
        selected_id = selection.objective.objective_id if selection.objective else None
        restock_detail = campaign_restock_item_summary(observed_state, selected_id)
        key += (restock_detail,)
        if key == self._campaign_task_diagnostics_key:
            return
        # A tactical controller can be remounted during a readiness handoff.
        # Keep the human-facing frontier panel session-scoped so remounting
        # the same campaign does not redraw an identical frontier. When no
        # runtime is installed (as in lightweight embeddings), retain the
        # controller-local behavior so separate test/custom sessions remain
        # independent.
        runtime = getattr(context, "nuzlocke_runtime", None)
        session_key = id(runtime) if runtime is not None else id(self)
        session_frontier_key = (session_key, key)
        if session_frontier_key == getattr(context, "campaign_frontier_diagnostics_key", None):
            self._campaign_task_diagnostics_key = key
            return
        self._campaign_task_diagnostics_key = key
        context.campaign_frontier_diagnostics_key = session_frontier_key
        print_campaign_frontier(
            completed,
            available,
            selected=selected_id,
            status=selection.status.value,
            reason=selection.reason,
            selected_detail=restock_detail,
            decision_details=getattr(selection, "selection_diagnostics", ()),
            encounter_portfolio=getattr(selection, "encounter_portfolio", ()),
        )

        # Keep the machine-readable diagnostic stream opt-in. The Rich
        # frontier above is the normal human-facing campaign report; detailed
        # task discovery remains debug-only because it can perform route and
        # encounter analysis.
        if not context.debug:
            return
        diagnostic_print(
            lambda: (
                "CAMPAIGN_OBJECTIVE_FRONTIER: "
                f"completed={tuple(format_campaign_objective_id(item) for item in completed)!r} "
                f"available={tuple(format_campaign_objective_id(item) for item in available)!r} "
                f"selected={format_campaign_objective_id(selection.objective.objective_id) if selection.objective else None!r} "
                f"status={selection.status.value!r} reason={selection.reason!r}"
            ),
            trace=False,
        )
        # Detailed rows perform route and encounter analysis. Keep those
        # expensive diagnostics opt-in while leaving the concise frontier
        # visible in ordinary debug console output.
        if not getattr(context, "debug_trace", False):
            return
        diagnostic_print(
            lambda: "CAMPAIGN_TASK_DISCOVERY: "
            + json.dumps(
                {
                    "tasks": campaign_task_diagnostics(observed_state),
                    "selected": selection.objective.objective_id if selection.objective else None,
                    "selection_status": selection.status.value,
                    "selection_reason": selection.reason,
                    "decision_details": getattr(selection, "selection_diagnostics", ()),
                },
                sort_keys=True,
                default=str,
            ),
            trace=True,
            prefix="CAMPAIGN_TASK_DISCOVERY",
        )

    def _stabilize_campaign_facts(self, state: CampaignState) -> CampaignState:
        """Retain confirmed story milestones across transient ROM reads."""

        if not hasattr(state, "campaign_lifecycle") or not hasattr(state, "campaign_facts"):
            # Lightweight controller embeddings may provide only the facts
            # needed by a custom selector. Preserve that compatibility seam.
            return state
        if state.campaign_lifecycle is CampaignObservationLifecycle.FRESH_START:
            self._confirmed_campaign_facts.clear()
            self._pending_campaign_fact_confirmations.clear()
            self._confirmed_campaign_session_id = None

        session_id = getattr(state, "session_id", None)
        if session_id is not None:
            if (
                self._confirmed_campaign_session_id is not None
                and session_id != self._confirmed_campaign_session_id
            ):
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_FACT_HIGH_WATER_RESET: "
                        f"old_session={self._confirmed_campaign_session_id!r} "
                        f"new_session={session_id!r} reason='observed session changed'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_FACT_HIGH_WATER_RESET",
                )
                self._confirmed_campaign_facts.clear()
                self._pending_campaign_fact_confirmations.clear()
            self._confirmed_campaign_session_id = session_id

        facts = state.campaign_facts
        newly_confirmed = set()
        for name in _MONOTONIC_CAMPAIGN_FACTS:
            fact = getattr(facts, name, Fact.unavailable())
            if fact.status is not FactStatus.KNOWN or fact.value is not True:
                self._pending_campaign_fact_confirmations.pop(name, None)
                continue
            if not self._campaign_fact_observation_is_coherent(name, facts):
                self._pending_campaign_fact_confirmations.pop(name, None)
                diagnostic_print(
                    lambda name=name: (
                        "CAMPAIGN_FACT_CONFIRMATION_DEFERRED: "
                        f"fact={name!r} reason='causal prerequisite not confirmed'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_FACT_CONFIRMATION_DEFERRED",
                )
                continue
            if name in self._confirmed_campaign_facts:
                continue
            observations = self._pending_campaign_fact_confirmations.get(name, 0) + 1
            required_observations = (
                _CAMPAIGN_FACT_CONFIRMATION_OBSERVATIONS
                if name in _CAMPAIGN_FACTS_REQUIRING_CONFIRMATION
                else 1
            )
            if observations >= required_observations:
                newly_confirmed.add(name)
                self._pending_campaign_fact_confirmations.pop(name, None)
            else:
                self._pending_campaign_fact_confirmations[name] = observations
                diagnostic_print(
                    lambda name=name, observations=observations, required_observations=required_observations: (
                        "CAMPAIGN_FACT_CONFIRMATION_PENDING: "
                        f"fact={name!r} observations={observations!r}/{required_observations!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_FACT_CONFIRMATION_PENDING",
                )
        self._confirmed_campaign_facts.update(newly_confirmed)
        retained = {
            name
            for name in self._confirmed_campaign_facts
            if getattr(facts, name, Fact.unavailable()).value is not True
        }
        suppressed = {
            name
            for name in _MONOTONIC_CAMPAIGN_FACTS
            if (
                (fact := getattr(facts, name, Fact.unavailable())).status is FactStatus.KNOWN
                and fact.value is True
                and name not in self._confirmed_campaign_facts
                and name in _CAMPAIGN_FACTS_REQUIRING_CONFIRMATION
            )
        }
        if not retained and not suppressed:
            return state

        diagnostic_print(
            lambda: (
                "CAMPAIGN_FACT_STABILIZED: "
                f"retained={tuple(sorted(retained))!r} "
                f"suppressed={tuple(sorted(suppressed))!r} "
                f"lifecycle={getattr(state.campaign_lifecycle, 'value', state.campaign_lifecycle)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_FACT_STABILIZED",
        )
        return replace(
            state,
            campaign_facts=replace(
                facts,
                **{
                    **{name: Fact.known(True) for name in retained},
                    **{name: Fact.unknown() for name in suppressed},
                },
            ),
        )

    def _campaign_fact_observation_is_coherent(self, name: str, facts) -> bool:
        """Reject a later true fact until its causal story is also present."""

        return all(
            (
                prerequisite in self._confirmed_campaign_facts
                or (
                    (fact := getattr(facts, prerequisite, Fact.unavailable())).status is FactStatus.KNOWN
                    and fact.value is True
                )
            )
            for prerequisite in _CAMPAIGN_FACT_PREREQUISITES.get(name, ())
        )

    def request_readiness_recheck(self, reason: str) -> None:
        """Require the next campaign refresh to observe readiness afresh.

        Battle completion can change HP and encounter legality without
        changing the cheap scheduler context. Keep that explicit event
        boundary separate from ordinary movement so a stale readiness result
        cannot authorize the next tactical step.
        """

        self._refresh_required = True
        self._refresh_required_reason = reason
        self._reset_encounter_stage_retry(reason)
        scheduler = getattr(self._readiness_provider, "__self__", None)
        invalidate = getattr(scheduler, "invalidate", None)
        if callable(invalidate):
            invalidate(reason)
        if (
            self._execution_phase == "CAMPAIGN"
            and self.current_objective_id is not None
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id == self.current_objective_id
        ):
            self._readiness_recheck_pending = True
            self._pending_readiness_objective_id = self.current_objective_id
        diagnostic_print(
            lambda: (
                "CAMPAIGN_READINESS_RECHECK: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"reason={reason!r} objective={self.current_objective_id!r} "
                f"pending={self._readiness_recheck_pending!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_READINESS_RECHECK:",
        )

    def request_recovery_after_wild_battle(self, reason: str = "wild_battle_resource_boundary") -> None:
        """Retain recovery ownership after a low-health wild battle.

        The battle planner owns the active battle, but it cannot mount an
        overworld recovery route until the battle has returned control to the
        campaign controller.  This marker bridges that boundary and turns a
        pending recovery calculation into a hard handoff for the current
        preparation/encounter route.
        """

        self._wild_battle_recovery_pending = True
        self.request_readiness_recheck(reason)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_WILD_BATTLE_RECOVERY_HANDOFF: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"reason={reason!r} objective={self.current_objective_id!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_WILD_BATTLE_RECOVERY_HANDOFF",
        )

    def request_refresh(self, reason: str) -> None:
        """Require the campaign envelope to be rebuilt at the next step.

        This is for cheap lifecycle boundaries that do not necessarily alter
        readiness data (for example, a map transition or a capability
        finishing).  Readiness-changing events should use
        :meth:`request_readiness_recheck` so the scheduler is invalidated as
        well.
        """

        self._refresh_required = True
        self._refresh_required_reason = reason
        self._reset_encounter_stage_retry(reason)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_REFRESH_REQUEST: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"reason={reason!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_REFRESH_REQUEST",
        )

    def _reset_encounter_stage_retry(self, reason: str | None = None) -> None:
        """Clear blocked-stage backoff after an explicit lifecycle boundary."""

        if self._encounter_stage_retry_delay or self._encounter_stage_retry_failures:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_ENCOUNTER_STAGE_RETRY_RESET: "
                    f"controller_id={id(self)!r} reason={reason!r} "
                    f"previous_failures={self._encounter_stage_retry_failures!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_ENCOUNTER_STAGE_RETRY_RESET",
            )
        self._encounter_stage_retry_signature = None
        self._encounter_stage_retry_failures = 0
        self._encounter_stage_retry_delay = 0

    def _close_tactical_loop(self, reason: str) -> None:
        """Close a discarded tactical owner before clearing its reference.

        AgentControlLoop owns a single-flight navigation request. Dropping the
        generator is usually enough under CPython, but it does not make the
        ownership boundary explicit and can leave a background search running
        while Campaign Progression is waiting at a blocked stage. Close the
        loop before publishing the idle boundary so its ``finally`` block can
        release held input and cancel that worker deterministically.
        """

        loop = self._tactical_loop
        loop_id = id(loop) if loop is not None else None
        self._tactical_loop = None
        self._tactical_loop_objective_id = None
        if loop is None:
            return
        close = getattr(loop, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TACTICAL_LOOP_CLOSE_FAILURE: "
                        f"controller_id={id(self)!r} loop_id={loop_id!r} reason={reason!r} "
                        f"exception_type={type(error).__name__!r} exception={error!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TACTICAL_LOOP_CLOSE_FAILURE",
                )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_TACTICAL_LOOP_CLOSED: "
                f"controller_id={id(self)!r} loop_id={loop_id!r} reason={reason!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_TACTICAL_LOOP_CLOSED",
        )

    def _encounter_stage_retry_waiting(self) -> bool:
        """Consume one blocked-stage backoff tick without refreshing."""

        if (
            self._encounter_stage_retry_delay <= 0
            or self._execution_phase == "RECOVERY_WAIT"
            or self._refresh_required_reason != "encounter_stage_blocked_retry"
        ):
            return False
        self._encounter_stage_retry_delay -= 1
        diagnostic_print(
            lambda: (
                "CAMPAIGN_ENCOUNTER_STAGE_RETRY_WAIT: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"remaining={self._encounter_stage_retry_delay!r} "
                f"failures={self._encounter_stage_retry_failures!r} input_emitted=False"
            ),
            trace=True,
            prefix="CAMPAIGN_ENCOUNTER_STAGE_RETRY_WAIT",
        )
        return True

    def _navigation_transition_owned(self) -> bool:
        """Return whether the mounted tactical owner still owns a warp.

        Most campaign capabilities expose an outer generator, while the
        actual navigation loop is suspended inside that generator. Check a
        directly mounted test/adapter loop first, then use the active loop
        marker maintained by ``AgentControlLoop.run`` for the nested case.
        Missing probes are treated as false so custom tactical iterators keep
        their existing behavior.
        """

        if self._tactical_loop is None:
            return False
        if _navigation_transition_probe(self._tactical_loop):
            return True
        active_loop = getattr(context, "active_agent_control_loop", None)
        return _navigation_transition_probe(active_loop) if active_loop is not None else False

    def refresh(self) -> CampaignControllerState:
        """Rebuild state, select, adapt, and mount work if needed."""
        if self._navigation_transition_owned():
            # A map/door transition is already owned by the mounted tactical
            # loop. Do not ask campaign readiness to inspect the transient
            # map-loading frame: it can report OVERWORLD_UNAVAILABLE and
            # otherwise tear down the very loop that must settle the warp.
            self._refresh_required = True
            self._refresh_required_reason = "navigation_transition_settling"
            self.status = CampaignControllerStatus.READY
            self.transition_reason = "navigation transition settling; preserving tactical loop"
            self._publish_campaign_status("Settling navigation transition")
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_NAVIGATION_TRANSITION_OWNERSHIP: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"objective={self.current_objective_id!r} "
                    f"execution_phase={self._execution_phase!r} "
                    "action='preserve_tactical_loop'"
                ),
                trace=True,
                prefix="CAMPAIGN_NAVIGATION_TRANSITION_OWNERSHIP",
            )
            return self.state
        observed_state = self._stabilize_campaign_facts(self._state_provider())
        pending_completed_objective = self._completed_capability_objective
        if pending_completed_objective is not None:
            completion = pending_completed_objective.completion.evaluate(observed_state)
            if completion.status is not FactStatus.KNOWN or completion.value is not True:
                # The capability already returned on an authoritative
                # completion observation. Wait for the campaign state facade
                # to publish the same fact before asking the selector to mount
                # another generator; otherwise a one-frame stale projection
                # replays the completed encounter route indefinitely.
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_COMPLETION_WAIT: "
                        f"controller_id={id(self)!r} "
                        f"objective={pending_completed_objective.objective_id!r} "
                        f"completion_status={completion.status.value!r} "
                        f"completion_value={completion.value!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_COMPLETION_WAIT",
                )
                self._close_tactical_loop("completion_projection_pending")
                self._refresh_required = True
                self._refresh_required_reason = "completion_projection_pending"
                self.status = CampaignControllerStatus.READY
                self.transition_reason = "capability completed; awaiting authoritative campaign projection"
                return self.state
            if getattr(context, "campaign_capability_completion_latch", None) == pending_completed_objective.objective_id:
                context.campaign_capability_completion_latch = None
            self._completed_capability_objective = None
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_COMPLETED_OWNERSHIP_RELEASE: "
                    f"controller_id={id(self)!r} "
                    f"objective={pending_completed_objective.objective_id!r} "
                    f"completion_status={completion.status.value!r} "
                    f"completion_value={completion.value!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_COMPLETED_OWNERSHIP_RELEASE",
            )
            # Release directly from the completion fence. The ordinary
            # ownership probe below assumes the last execution still has the
            # exact objective identity that was mounted; after a battle/script
            # boundary that assumption can be false even though the fence's
            # own completion predicate is authoritative.
            self.last_selection = None
            self.last_execution = None
            self.current_objective_id = None
            self.current_tactical_goal = None
            self._readiness_recheck_pending = False
            self._pending_readiness_objective_id = None
        completed_capability = (
            self.last_selection
            if self._tactical_loop is None
            and self._execution_phase == "CAMPAIGN"
            and self.last_selection is not None
            and self.last_selection.objective is not None
            and self.last_selection.objective.task_kind != "safety"
            and self.last_execution is not None
            and self.last_execution.capability is not None
            and self.current_objective_id == self.last_selection.objective.objective_id
            else None
        )
        if completed_capability is not None:
            completion = completed_capability.objective.completion.evaluate(observed_state)
            if completion.status is FactStatus.KNOWN and completion.value is True:
                # A capability can finish between controller calls. Release
                # the completed selection before asking the planner for the
                # next frontier; otherwise readiness ownership and the mode's
                # planning cache can both hand the completed objective back to
                # the execution adapter for one more remount.
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_COMPLETED_OWNERSHIP_RELEASE: "
                        f"controller_id={id(self)!r} "
                        f"objective={completed_capability.objective.objective_id!r} "
                        f"completion_status={completion.status.value!r} "
                        f"completion_value={completion.value!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_COMPLETED_OWNERSHIP_RELEASE",
                )
                self.last_selection = None
                self.last_execution = None
                self.current_objective_id = None
                self.current_tactical_goal = None
                self._readiness_recheck_pending = False
                self._pending_readiness_objective_id = None
        active_safety = (
            self.last_selection
            if self.last_selection is not None
            and self.last_selection.objective is not None
            and self.last_selection.objective.task_kind == "safety"
            and self.current_objective_id == self.last_selection.objective.objective_id
            else None
        )
        if active_safety is not None:
            completion = active_safety.objective.completion.evaluate(observed_state)
            # A safety capability can restore its resource before the ROM has
            # released the interaction that performed that work.  For
            # example, Emerald restores the party during the nurse's healing
            # script and displays the closing messages afterward.  Keep the
            # mounted generator authoritative until it returns; otherwise a
            # parent campaign objective can replace it while its final input
            # prompt is still active.
            safety_loop_active = self._tactical_loop is not None
            if completion.is_known and completion.value is True and not safety_loop_active:
                # A terminal safety objective releases ownership.  The saved
                # parent is deliberately left for the normal selector so it
                # is resumed using the current campaign facts.
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_SAFETY_TERMINAL: "
                        f"controller_id={id(self)!r} objective={active_safety.objective.objective_id!r} "
                        f"status='complete' interrupted_objective={self.interrupted_objective_id!r}"
                    ),
                    trace=True,
                )
                self.current_objective_id = None
                self.current_tactical_goal = None
                self._tactical_loop = None
                self._tactical_loop_objective_id = None
                self.last_selection = None
                self.interrupted_objective_id = None
                selection = self._selector(observed_state)
            else:
                # Safety ownership is an execution invariant, not a planner
                # preference: an incomplete interrupting objective, or a
                # completed objective with a still-running capability, cannot
                # be replaced merely because its parent remains incomplete.
                selection = active_safety
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_SAFETY_OWNERSHIP: "
                        f"controller_id={id(self)!r} objective={selection.objective.objective_id!r} "
                        f"completion_status={completion.status!r} "
                        f"completion_value={completion.value!r} "
                        f"loop_active={safety_loop_active!r} "
                        f"interrupted_objective={self.interrupted_objective_id!r}"
                    ),
                    trace=True,
                )
        else:
            selection = self._selector(observed_state)

        # A recovery route calculation owns the campaign boundary while its
        # worker is in flight. Keep polling the same objective instead of
        # allowing encounter selection to mount a new navigation loop while
        # the party may still require healing.
        if (
            self._execution_phase == "RECOVERY_WAIT"
            and self.last_selection is not None
            and self.last_selection.objective is not None
            and self.current_objective_id == self.last_selection.objective.objective_id
            and self.last_execution is not None
        ):
            if selection.objective is None or selection.objective.objective_id != self.current_objective_id:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_WAIT_OWNERSHIP: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"retained={self.current_objective_id!r} "
                        f"planner_selected={getattr(getattr(selection, 'objective', None), 'objective_id', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_WAIT_OWNERSHIP",
                )
                selection = self.last_selection

        # A live restock capability owns the entire Mart transaction, including
        # purchase confirmations and the clerk's exit dialogue.  Inventory
        # facts can become sufficient as soon as the first items are written,
        # so the planner may select an encounter while the generator still has
        # to close the menu.  Retaining the selection here protects the
        # generator before any readiness or tactical handoff can replace it.
        mounted_restock = (
            self.last_selection
            if self._execution_phase == "CAMPAIGN"
            and self._tactical_loop is not None
            and self._tactical_loop_objective_id in _RESTOCK_OBJECTIVE_IDS
            and self.current_objective_id == self._tactical_loop_objective_id
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id == self._tactical_loop_objective_id
            and self.last_execution is not None
            and self.last_execution.capability is not None
            else None
        )
        transactional_restock = (
            self.last_selection
            if _restock_transaction_is_active()
            and getattr(context, "campaign_restock_phase", None) not in {"recovery_active"}
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id in _RESTOCK_OBJECTIVE_IDS
            and self.last_execution is not None
            and self.last_execution.capability is not None
            else None
        )
        if transactional_restock is not None and (
            selection.objective is None
            or selection.objective.objective_id != transactional_restock.objective.objective_id
        ):
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RESTOCK_TRANSACTION_OWNERSHIP: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"phase={getattr(context, 'campaign_restock_phase', None)!r} "
                    f"retained={transactional_restock.objective.objective_id!r} "
                    f"planner_selected={getattr(getattr(selection, 'objective', None), 'objective_id', None)!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RESTOCK_TRANSACTION_OWNERSHIP",
            )
            selection = transactional_restock
        selected_objective = getattr(selection, "objective", None)
        if (
            mounted_restock is not None
            and (
                selected_objective is None
                or selected_objective.objective_id != mounted_restock.objective.objective_id
            )
            and (selected_objective is None or selected_objective.task_kind != "safety")
        ):
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RESTOCK_OWNERSHIP: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"retained={mounted_restock.objective.objective_id!r} "
                    f"planner_selected={getattr(selected_objective, 'objective_id', None)!r} "
                    "reason='mounted_restock_transaction_incomplete'"
                ),
                trace=True,
                prefix="CAMPAIGN_RESTOCK_OWNERSHIP",
            )
            selection = mounted_restock

        # A deferred readiness observation has not mounted a tactical loop,
        # so the normal active-loop ownership guard below cannot protect its
        # objective. A battle/script transition can nevertheless make the
        # planner report an earlier task for one frame. Keep the objective
        # that owns the pending recheck while it is still incomplete and
        # safe; otherwise a cached RECOVER result could be applied to the
        # transiently selected task instead of to the task that produced the
        # observation.
        pending_readiness_objective = (
            self.last_selection.objective
            if self._readiness_recheck_pending
            and self._pending_readiness_objective_id is not None
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id == self._pending_readiness_objective_id
            and self.last_execution is not None
            and self.last_execution.objective is not None
            and self.last_execution.objective.objective_id == self._pending_readiness_objective_id
            else None
        )
        if pending_readiness_objective is not None:
            pending_completion = pending_readiness_objective.completion.evaluate(observed_state)
            pending_failure = (
                pending_readiness_objective.failure.evaluate(observed_state)
                if pending_readiness_objective.failure is not None
                else None
            )
            pending_prerequisites_safe = all(
                prerequisite.evaluate(observed_state).value is not False
                for prerequisite in pending_readiness_objective.prerequisites
            )
            pending_completion_value = getattr(pending_completion, "value", None)
            pending_safe = (
                pending_completion_value is not True
                and (pending_failure is None or pending_failure.value is not True)
                and pending_prerequisites_safe
            )
            if (
                selection.objective is None
                or selection.objective.objective_id != pending_readiness_objective.objective_id
            ):
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_OWNERSHIP_CHECK: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"pending={pending_readiness_objective.objective_id!r} "
                        f"planner_selected={getattr(selection.objective, 'objective_id', None)!r} "
                        f"completion_status={getattr(getattr(pending_completion, 'status', None), 'value', None)!r} "
                        f"completion_value={pending_completion_value!r} "
                        f"prerequisites_safe={pending_prerequisites_safe!r} "
                        f"failure_safe={pending_failure is None or pending_failure.value is not True!r} "
                        f"pending_safe={pending_safe!r}"
                    ),
                    trace=True,
                )
                if pending_safe:
                    selection = self.last_selection
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_READINESS_OWNERSHIP: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"retained={pending_readiness_objective.objective_id!r} "
                            f"planner_selected={getattr(selection.objective, 'objective_id', None)!r} "
                            "reason='deferred_readiness_recheck'"
                        ),
                        trace=True,
                    )
                else:
                    self._readiness_recheck_pending = False
                    self._pending_readiness_objective_id = None

        # Keep ownership of an already-mounted objective across transient
        # battle/script observations.  The planner is intentionally rebuilt
        # from ROM-backed facts on every frame, but a battle-start/return
        # boundary can briefly expose an older or incomplete campaign view.
        # Replacing a live capability at that point discards its generator
        # position; preparation then starts over at its map-level handoff.
        #
        # This is deliberately narrower than making the planner sticky: an
        # active objective may be retained only while it is still incomplete,
        # has no known failure, and none of its prerequisites is known false.
        # A genuine save reset or story regression therefore still releases
        # the old objective for normal ROM-fact selection.
        active_objective = (
            self.last_selection.objective
            if (
                self._tactical_loop is not None
                or (
                    # Observation-driven capabilities can return at a ROM
                    # script boundary before their authoritative completion
                    # fact is visible to the controller.  Keep that
                    # capability's objective as the owner for the remount;
                    # otherwise optional encounter selection can replace a
                    # live story scene while its dialogue or tutorial battle
                    # is still queued (the Petalburg Norman/Wally boundary).
                    self._refresh_required_reason in {"capability_boundary", "battle_ended"}
                    and self.last_execution is not None
                    and self.last_execution.capability is not None
                )
            )
            and self.current_objective_id is not None
            and self.last_selection is not None
            and self.last_selection.status is ObjectiveStatus.READY
            and self.last_selection.objective is not None
            and self.last_selection.objective.objective_id == self.current_objective_id
            else None
        )
        if active_objective is not None and selection.objective is not None:
            active_completion = active_objective.completion.evaluate(observed_state)
            active_failure = (
                active_objective.failure.evaluate(observed_state) if active_objective.failure is not None else None
            )
            prerequisites_safe = all(
                prerequisite.evaluate(observed_state).value is not False
                for prerequisite in active_objective.prerequisites
            )
            failure_safe = active_failure is None or active_failure.value is not True
            spatial_observation_coherent = _spatial_observation_is_coherent(observed_state)
            if selection.objective.objective_id != active_objective.objective_id:
                retention_order_allows = _retention_order_allows(
                    active_objective.objective_id,
                    selection.objective.objective_id,
                )
                if (
                    selection.objective.objective_id in _RESTOCK_OBJECTIVE_IDS
                    and _story_progression_owns_restock_boundary(
                        active_objective,
                        refresh_reason=self._refresh_required_reason,
                    )
                ):
                    retention_order_allows = True
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_STORY_OWNERSHIP: "
                            f"active={active_objective.objective_id!r} "
                            f"planner_selected={selection.objective.objective_id!r} "
                            f"refresh_reason={self._refresh_required_reason!r} "
                            "reason='active story boundary retains ownership over restock'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_STORY_OWNERSHIP",
                    )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_OBJECTIVE_OWNERSHIP_CHECK: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"active={active_objective.objective_id!r} "
                        f"planner_selected={selection.objective.objective_id!r} "
                        f"completion_status={active_completion.status.value!r} "
                        f"completion_value={active_completion.value!r} "
                        f"prerequisites_safe={prerequisites_safe!r} "
                        f"failure_safe={failure_safe!r} "
                        f"spatial_observation_coherent={spatial_observation_coherent!r} "
                        f"retention_order_allows={retention_order_allows!r}"
                    ),
                    trace=True,
                )
            else:
                retention_order_allows = False
            if (
                selection.objective.objective_id != active_objective.objective_id
                and retention_order_allows
                and (
                    active_completion.status is not FactStatus.KNOWN
                    or (
                        active_completion.status is FactStatus.KNOWN
                        and active_completion.value is False
                        and (prerequisites_safe or not spatial_observation_coherent)
                        and failure_safe
                    )
                    or (
                        # Emerald can publish an objective's completion fact
                        # while its capability is still consuming the final
                        # script/dialogue sequence. Keep that capability's
                        # ownership until its generator returns; otherwise a
                        # downstream objective can be selected and readiness
                        # can defer before the old loop is replaced, leaving
                        # the controller with mismatched selected/mounted
                        # objectives. Explicit restocking remains an
                        # interrupt and may pre-empt a completed capability.
                        active_completion.status is FactStatus.KNOWN
                        and active_completion.value is True
                        and self.last_execution is not None
                        and self.last_execution.capability is not None
                        and (
                            selection.objective.objective_id != "restock_pokeballs"
                            or active_objective.objective_id == "restock_recovery_items"
                        )
                    )
                )
            ):
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_OBJECTIVE_OWNERSHIP: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"retained={active_objective.objective_id!r} "
                        f"planner_selected={selection.objective.objective_id!r} "
                        f"completion_status={active_completion.status.value!r} "
                        f"completion_value={active_completion.value!r} "
                        f"prerequisites_safe={prerequisites_safe!r} "
                        f"spatial_observation_coherent={spatial_observation_coherent!r} "
                        f"failure_safe={failure_safe!r} reason='transient_observation_boundary'"
                    ),
                    trace=True,
                )
                selection = self.last_selection
        previous_id = (
            self.last_selection.objective.objective_id
            if self.last_selection and self.last_selection.objective
            else None
        )
        selected_id = selection.objective.objective_id if selection.objective else None
        if selected_id in _RESTOCK_OBJECTIVE_IDS and previous_id not in _RESTOCK_OBJECTIVE_IDS:
            context.campaign_restock_parent_objective_id = previous_id
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RESTOCK_TRANSACTION: "
                    f"phase='planned' owner={selected_id!r} parent_objective={previous_id!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RESTOCK_TRANSACTION",
            )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_SELECTION_HANDOFF: "
                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                f"selected_objective={selected_id!r} status={selection.status.value!r} "
                f"previous_objective={previous_id!r} changed={selected_id != previous_id!r} "
                f"map={getattr(getattr(observed_state, 'raw_map', None), 'value', None)!r} "
                f"facts={getattr(observed_state, 'campaign_facts', None)!r}"
            ),
            trace=True,
        )
        self._emit_campaign_task_diagnostics(observed_state, selection)
        trace = getattr(context, "stutter_trace", None)
        if trace is not None and callable(getattr(trace, "mark", None)):
            intro_completion = getattr(observed_state.campaign_facts, "intro_rival_battle_complete", None)
            trace.mark("campaign_selected_objective", selected_id)
            trace.mark("campaign_mounted_objective", self._tactical_loop_objective_id)
            trace.mark(
                "campaign_intro_rival_completion",
                (
                    None
                    if intro_completion is None
                    else {
                        "status": getattr(getattr(intro_completion, "status", None), "value", None),
                        "value": getattr(intro_completion, "value", None),
                    }
                ),
            )
            trace.mark("campaign_execution_phase", self._execution_phase)
        self.last_selection = selection
        execution = self._adapter(selection)
        pending_trainer_handoff = self._pending_trainer_handoff
        if (
            pending_trainer_handoff is None
            and self._execution_phase == "RECOVERY_WAIT"
            and self.last_execution is not None
            and getattr(self.last_execution.objective, "objective_id", None)
            == getattr(execution.objective, "objective_id", None)
            and _is_trainer_goal(getattr(self.last_execution, "tactical_goal", None))
        ):
            pending_trainer_handoff = (
                getattr(execution.objective, "objective_id", None),
                self.last_execution.tactical_goal,
            )
            self._pending_trainer_handoff = pending_trainer_handoff
        if (
            pending_trainer_handoff is not None
            and getattr(execution.objective, "objective_id", None) == pending_trainer_handoff[0]
        ):
            execution = replace(
                execution,
                tactical_goal=pending_trainer_handoff[1],
                capability=None,
                reason="waiting for unresolved mandatory trainer checkpoint",
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TRAINER_HANDOFF_RETAINED: "
                    f"controller_id={id(self)!r} objective={pending_trainer_handoff[0]!r} "
                    f"goal={pending_trainer_handoff[1]!r} reason='trainer_roster_pending'"
                ),
                trace=True,
                prefix="CAMPAIGN_TRAINER_HANDOFF_RETAINED",
            )
        elif pending_trainer_handoff is not None:
            self._pending_trainer_handoff = None
        diagnostic_print(
            lambda: (
                "CAMPAIGN_EXECUTION_HANDOFF: "
                f"controller_id={id(self)!r} objective_passed={getattr(execution.objective, 'objective_id', None)!r} "
                f"existing_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r} "
                f"existing_loop_objective={self._tactical_loop_objective_id!r} "
                f"action={'REPLACE_OR_CREATE' if self._tactical_loop is None or self.current_objective_id != getattr(execution.objective, 'objective_id', None) or self.current_tactical_goal != execution.tactical_goal else 'REUSE'} "
                f"tactical_goal={execution.tactical_goal!r} capability={getattr(execution.capability, '__name__', None)!r}"
            ),
            trace=True,
        )
        self.last_execution = execution
        terminal_status = {
            CampaignExecutionStatus.BLOCKED: CampaignControllerStatus.BLOCKED,
            CampaignExecutionStatus.UNKNOWN: CampaignControllerStatus.UNKNOWN,
            CampaignExecutionStatus.COMPLETE: CampaignControllerStatus.COMPLETE,
            CampaignExecutionStatus.FAILED: CampaignControllerStatus.FAILED,
            CampaignExecutionStatus.UNSUPPORTED: CampaignControllerStatus.FAILED,
        }
        if execution.status in terminal_status:
            # Objective selection can legitimately be unavailable for one
            # observation immediately after a battle or script boundary.  In
            # that case the selection's objective is still useful diagnostic
            # context, but it must not strand the controller in TERMINAL:
            # ``CampaignProgressionMode`` remains mounted so the next frame
            # can observe the ROM again and resolve the frontier.
            if selection.status in {ObjectiveStatus.UNKNOWN, ObjectiveStatus.BLOCKED}:
                self.current_objective_id = (
                    selection.objective.objective_id if selection.objective is not None else None
                )
                self.current_tactical_goal = None
                self._tactical_loop = None
                self._tactical_loop_objective_id = None
                self._execution_phase = "CAMPAIGN"
                self.status = terminal_status[execution.status]
                self.transition_reason = execution.reason
                self._refresh_required = True
                self._refresh_required_reason = "campaign_selection_retry"
                self._deferred_handoff_requires_refresh = False
                self._publish_status_for_execution(
                    selection.objective,
                    None,
                    "Campaign observation unavailable; retrying autonomously"
                    if selection.status is ObjectiveStatus.UNKNOWN
                    else "Campaign prerequisite unavailable; retrying autonomously",
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_SELECTION_RETRY: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={getattr(selection.objective, 'objective_id', None)!r} "
                        f"status={selection.status.value!r} reason={selection.reason!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_SELECTION_RETRY",
                )
                return self.state
            self._terminal(execution, terminal_status[execution.status])
            diagnostic_print(
                lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='terminal_execution' status={self.status.value!r} reason={self.transition_reason!r}",
                trace=True,
            )
            return self.state
        if execution.status is not CampaignExecutionStatus.READY or execution.objective is None:
            self._terminal(
                CampaignExecutionResult(
                    execution.objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "malformed ready execution result",
                    execution.execution_id,
                ),
                CampaignControllerStatus.FAILED,
            )
            diagnostic_print(
                lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='malformed_ready' status={self.status.value!r} reason={self.transition_reason!r}",
                trace=True,
            )
            return self.state
        if execution.tactical_goal is None and execution.capability is None:
            self._terminal(
                CampaignExecutionResult(
                    execution.objective,
                    CampaignExecutionStatus.UNSUPPORTED,
                    "ready execution result has no tactical goal",
                    execution.execution_id,
                ),
                CampaignControllerStatus.FAILED,
            )
            diagnostic_print(
                lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='missing_execution_target' status={self.status.value!r} reason={self.transition_reason!r}",
                trace=True,
            )
            return self.state
        objective_id = execution.objective.objective_id
        # A readiness report belongs to the current execution boundary. Clear
        # the previous one before asking the provider for a fresh observation
        # so an opening or non-overworld capability cannot inherit a stale
        # trainer decision in the next Rich snapshot.
        context.campaign_readiness_diagnostic = None
        if self._campaign_plan is None or self._campaign_plan.parent_objective_id != objective_id:
            self._campaign_plan = CampaignPlan(objective_id)
        readiness_map = None
        readiness_coordinates = None
        # Readiness is tactical/resource gating. Startup Emerald capabilities
        # intentionally have no tactical Goal: they must be allowed to observe
        # title/menu/naming/clock state before overworld readiness exists.
        # Most capability-backed opening objectives must be allowed to observe
        # title/menu/script state before an overworld exists. A capability
        # with a declared campaign destination, or any targetless capability
        # after the Pokédex boundary, can own ordinary overworld traversal and
        # therefore must pass through the same recovery readiness gate. This
        # is deliberately derived from execution shape and campaign state,
        # rather than from one objective ID.
        readiness_capability_boundary = (
            execution.capability is not None
            and (
                getattr(execution.objective, "destination", None) is not None
                or _post_pokedex_campaign_boundary(observed_state)
            )
        )
        readiness_applies = (
            execution.tactical_goal is not None
            or readiness_capability_boundary
        )
        if (
            self._readiness_provider is not None
            and self._execution_phase in {"CAMPAIGN", "RECOVERY_WAIT"}
            and readiness_applies
        ):
            # RECOVERY_WAIT is the ownership state used while the same
            # readiness worker is still calculating. Polling the provider is
            # required here; skipping it lets the original capability fall
            # through to the generic mount path on the next refresh.
            if self._execution_phase == "RECOVERY_WAIT":
                self._execution_phase = "CAMPAIGN"
            forced_readiness_recheck = False
            if self._readiness_recheck_pending:
                scheduler = getattr(self._readiness_provider, "__self__", None)
                invalidate = getattr(scheduler, "invalidate", None)
                if callable(invalidate):
                    invalidate("deferred_readiness_recheck")
                forced_readiness_recheck = True
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_RECHECK_FORCED: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        "reason='deferred_readiness_recheck'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_READINESS_RECHECK_FORCED:",
                )
                # A recheck request is a one-observation invalidation.  A
                # transient result below may schedule another one; stable
                # policy results must not create a per-frame invalidation loop.
                self._readiness_recheck_pending = False
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_PROVIDER: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"objective={objective_id!r} tactical_goal={execution.tactical_goal!r} "
                    f"provider_id={id(self._readiness_provider)!r} "
                    f"provider_type={type(self._readiness_provider).__name__!r}"
                ),
                trace=True,
            )
            readiness_input = self._readiness_provider(execution.objective, execution.tactical_goal)
            # The provider supplies observations. Evaluate those observations
            # before applying controller-owned handoff overrides so the final
            # campaign plan can be composed exactly once from the effective
            # decision. Previously the plan was built here, then a trainer
            # precondition could promote ``evaluated`` to RECOVER below while
            # the old plan still had no RecoveryStop.
            if isinstance(readiness_input, ProgressionReadinessDiagnostic):
                evaluated = readiness_input
                if evaluated.readiness_decision is None or evaluated.readiness_reason is None:
                    evaluated = evaluate_progression_readiness(
                        readiness_input,
                        policy=self._readiness_policy,
                    )
            else:
                evaluated = readiness_input
            if evaluated.readiness_decision is None:
                evaluated = replace(
                    evaluated,
                    readiness_decision=getattr(readiness_input, "readiness_decision", None),
                    readiness_reason=getattr(readiness_input, "readiness_reason", None),
                )
            # Keep the final policy evidence available to the next campaign
            # status boundary. The Rich snapshot is the ordinary operator
            # view; the structured readiness object remains useful to GUI and
            # integration consumers as well.
            wild_battle_recovery_active = False
            if self._wild_battle_recovery_pending and isinstance(evaluated, ProgressionReadinessDiagnostic):
                lowest_hp_ratio = getattr(evaluated, "lowest_hp_ratio", None)
                if (
                    isinstance(lowest_hp_ratio, (int, float))
                    and not isinstance(lowest_hp_ratio, bool)
                    and lowest_hp_ratio <= self._readiness_policy.opportunistic_hp_ratio
                ):
                    wild_battle_recovery_active = True
                    recovery_pending = _readiness_calculation_pending(evaluated)
                    recovery_available = getattr(evaluated.recovery_availability, "value", None) == "known"
                    stable_overworld = getattr(evaluated, "game_state", None) == "OVERWORLD"
                    if recovery_pending and evaluated.readiness_decision is not ReadinessDecision.RECOVER:
                        # Do not let an advisory route worker keep the
                        # encounter capability moving after the battle policy
                        # has already demonstrated that training is unsafe.
                        # The existing UNKNOWN boundary will stop the tactical
                        # loop and poll the same recovery observation.
                        evaluated = replace(
                            evaluated,
                            readiness_decision=ReadinessDecision.UNKNOWN,
                            readiness_reason=ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN,
                        )
                    elif (
                        stable_overworld
                        and recovery_available
                        and evaluated.readiness_decision is not ReadinessDecision.RECOVER
                    ):
                        # Once the recovery observation is complete, healing
                        # owns the next action even if the normal-vs-recovery
                        # comparison would otherwise keep this optional
                        # encounter route alive.
                        evaluated = replace(
                            evaluated,
                            readiness_decision=ReadinessDecision.RECOVER,
                            readiness_reason=ReadinessReason.OPPORTUNISTIC_RECOVERY,
                        )
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_WILD_BATTLE_RECOVERY_DECISION: "
                            f"objective={objective_id!r} hp_ratio={lowest_hp_ratio!r} "
                            f"calculation_pending={recovery_pending!r} "
                            f"recovery_available={recovery_available!r} "
                            f"decision={evaluated.readiness_decision.value!r} "
                            f"reason={evaluated.readiness_reason.value!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_WILD_BATTLE_RECOVERY_DECISION",
                    )
                elif (
                    isinstance(lowest_hp_ratio, (int, float))
                    and not isinstance(lowest_hp_ratio, bool)
                    and lowest_hp_ratio > self._readiness_policy.opportunistic_hp_ratio
                ):
                    # A later authoritative observation may show that the
                    # party has already recovered by another route. Do not
                    # carry the old battle marker into a healthy encounter.
                    self._wild_battle_recovery_pending = False
            context.campaign_readiness_diagnostic = evaluated
            # A route-risk promotion turns a map/capability objective into a
            # deliberate trainer checkpoint.  Use the enriched readiness goal
            # as the tactical owner for this boundary; otherwise an encounter
            # capability would immediately remount its original grass route
            # and walk into the same sight line without the preparation gate.
            promoted_goal = getattr(evaluated, "navigation_goal", None)
            promoted_trainer_handoff = False
            if isinstance(promoted_goal, EngageTrainer) or (
                isinstance(promoted_goal, NavigationGoal)
                and isinstance(getattr(promoted_goal, "target", None), EngageTrainer)
            ):
                if execution.tactical_goal != promoted_goal or execution.capability is not None:
                    promoted_trainer_handoff = execution.capability is not None
                    execution = replace(
                        execution,
                        tactical_goal=promoted_goal,
                        capability=None,
                        reason="route trainer risk promoted to a prepared checkpoint",
                    )
                    # The adapter result was recorded before readiness could
                    # enrich an encounter route with its known trainer
                    # checkpoint. Keep the authoritative execution aligned
                    # with the goal that will actually be mounted. In
                    # particular, a later StopIteration must not mistake the
                    # generic trainer route for the old encounter capability.
                    self.last_execution = execution
            encounter_stage = getattr(context, "campaign_encounter_stage", None)
            recovery = getattr(evaluated, "recovery", None)
            completed_recovery_route = (
                bool(getattr(recovery, "center_available", False))
                and bool(getattr(recovery, "safe_to_reach_center", False))
                and getattr(recovery, "route", None) is not None
            )
            encounter_stage_blocks_owner = (
                execution.capability is not None
                and isinstance(objective_id, str)
                and objective_id.startswith("obtain_encounter:")
                and getattr(encounter_stage, "blocked", False)
                and getattr(encounter_stage, "current_map", None) == getattr(evaluated, "current_map", None)
                and not isinstance(promoted_goal, EngageTrainer)
                and not (
                    isinstance(promoted_goal, NavigationGoal)
                    and isinstance(getattr(promoted_goal, "target", None), EngageTrainer)
                )
                and evaluated.readiness_decision not in {
                    ReadinessDecision.RECOVER,
                    ReadinessDecision.PREPARE,
                }
            )
            if encounter_stage_blocks_owner and completed_recovery_route:
                # A blocked optional encounter has no safe executable owner,
                # but a completed Center route is an executable safety handoff.
                # Promote it before the blocked-stage return below so the
                # generic recovery mount can consume the route. Without this
                # promotion the controller closes the encounter loop, leaves
                # readiness at CONTINUE/UNKNOWN, and spins with no owner.
                evaluated = replace(
                    evaluated,
                    readiness_decision=ReadinessDecision.RECOVER,
                    readiness_reason=ReadinessReason.OPPORTUNISTIC_RECOVERY,
                )
                context.campaign_readiness_diagnostic = evaluated
                encounter_stage_blocks_owner = False
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_DECISION: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={objective_id!r} decision='RECOVER' "
                        "reason='blocked_encounter_stage' action='mount_safe_center_route'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_DECISION",
                )
            if encounter_stage_blocks_owner:
                # The encounter stage already proved that the current
                # capability cannot safely cross its next boundary. Preserve
                # that handoff as an explicit unknown/block rather than
                # remounting the same generator and replaying the same
                # AVOID/IGNORE searches. A recovery calculation may still be
                # in flight, however. In that case the controller must keep
                # the refresh obligation and poll the worker; clearing it
                # strands the campaign with no loop and no future observer.
                self._close_tactical_loop("encounter_stage_blocked")
                self.current_objective_id = objective_id
                self.current_tactical_goal = execution.tactical_goal
                calculation_pending = any(
                    _readiness_calculation_pending(value)
                    for value in (readiness_input, evaluated)
                )
                recovery_pending = any(
                    getattr(getattr(value, "recovery", None), "calculation_pending", False)
                    for value in (readiness_input, evaluated)
                )
                if calculation_pending:
                    self._encounter_stage_retry_signature = None
                    self._encounter_stage_retry_failures = 0
                    self._encounter_stage_retry_delay = 0
                    self._execution_phase = "RECOVERY_WAIT"
                    self._recovery_status = "CALCULATING"
                    self._refresh_required = True
                    self._refresh_required_reason = "encounter_stage_recovery_pending"
                    self.status = CampaignControllerStatus.READY
                    self.transition_reason = (
                        "encounter stage blocked; waiting for recovery route analysis"
                    )
                    status_message = "Calculating recovery route before encounter-stage retry"
                else:
                    retry_signature = (
                        objective_id,
                        getattr(encounter_stage, "current_map", None),
                        getattr(encounter_stage, "next_map", None),
                        getattr(encounter_stage, "world_signature", None),
                        tuple(getattr(encounter_stage, "known_hazard_ids", ()) or ()),
                        tuple(getattr(encounter_stage, "unresolved_hazard_ids", ()) or ()),
                    )
                    if retry_signature != self._encounter_stage_retry_signature:
                        self._encounter_stage_retry_signature = retry_signature
                        self._encounter_stage_retry_failures = 0
                        self._encounter_stage_retry_delay = 0
                    self._encounter_stage_retry_failures += 1
                    self._encounter_stage_retry_delay = min(
                        _ENCOUNTER_STAGE_RETRY_DELAY_MAX,
                        2 ** min(self._encounter_stage_retry_failures, 6),
                    )
                    self._execution_phase = "CAMPAIGN"
                    self._refresh_required = True
                    self._refresh_required_reason = "encounter_stage_blocked_retry"
                    self.status = CampaignControllerStatus.UNKNOWN
                    self.transition_reason = (
                        "encounter stage blocked; awaiting trainer promotion or a changed world boundary"
                    )
                    status_message = "Encounter stage blocked; awaiting a changed route or trainer evidence"
                self._publish_status_for_execution(
                    execution.objective,
                    execution.tactical_goal,
                    status_message,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_ENCOUNTER_STAGE_BLOCKED: "
                        f"objective={objective_id!r} current_map={getattr(encounter_stage, 'current_map', None)!r} "
                        f"next_map={getattr(encounter_stage, 'next_map', None)!r} "
                        f"world_signature={getattr(encounter_stage, 'world_signature', None)!r} "
                        f"known_hazards={getattr(encounter_stage, 'known_hazard_ids', ())!r} "
                        f"unresolved_hazards={getattr(encounter_stage, 'unresolved_hazard_ids', ())!r} "
                        f"reason={getattr(encounter_stage, 'reason', None)!r} "
                        f"calculation_pending={calculation_pending!r} "
                        f"recovery_pending={recovery_pending!r} "
                        f"retry_failures={self._encounter_stage_retry_failures!r} "
                        f"retry_delay={self._encounter_stage_retry_delay!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_ENCOUNTER_STAGE_BLOCKED",
                )
                return self.state
            recovery_route_pending = any(
                getattr(getattr(value, "recovery", None), "calculation_pending", False)
                for value in (readiness_input, evaluated)
            )
            trainer_battle_owner = (
                _objective_requires_trainer_battle(execution.objective)
                or _is_trainer_goal(execution.tactical_goal)
                or getattr(evaluated, "trainer_engagement_required", False)
            )
            if recovery_route_pending and trainer_battle_owner and evaluated.readiness_decision is not ReadinessDecision.RECOVER:
                # A story trainer is sometimes represented by ActivateTrigger
                # until route-aware readiness resolves its checkpoint. The
                # recovery worker can therefore become pending before the
                # tactical goal is rewritten to EngageTrainer. Keep the
                # trainer objective blocked across that representation change;
                # otherwise the old trigger loop can dispatch the battle
                # before the selected Center route has returned.
                evaluated = replace(
                    evaluated,
                    readiness_decision=ReadinessDecision.UNKNOWN,
                    readiness_reason=ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_RECOVERY_GATE: "
                        f"objective={objective_id!r} tactical_goal={execution.tactical_goal!r} "
                        "recovery_route_pending=True action='block_trainer_handoff'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_RECOVERY_GATE",
                )
            if (
                self._accepted_recovery_objective_id == objective_id
                and self._execution_phase in {"RECOVERY_WAIT", "RECOVERY"}
                and evaluated.readiness_decision is not ReadinessDecision.RECOVER
            ):
                # A completed readiness calculation may briefly be replaced by
                # a stale campaign snapshot while the selected recovery stop
                # is crossing into the controller. Preserve ownership until
                # the recovery generator reports completion or failure.
                evaluated = replace(
                    evaluated,
                    readiness_decision=ReadinessDecision.RECOVER,
                    readiness_reason=ReadinessReason.OPPORTUNISTIC_RECOVERY,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_DECISION: "
                        f"objective={objective_id!r} decision='RECOVER' "
                        "reason='accepted_recovery_owner' action='retain_until_mounted_or_completed'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_DECISION",
                )
            # Trainer preparation is evaluated before the generic recovery
            # mount below. If both policies are true, healing owns the next
            # action: entering the trainer route while a safe recovery stop is
            # already available is exactly the ordering failure this boundary
            # is meant to prevent.
            recovery = getattr(evaluated, "recovery", None)
            lowest_hp_ratio = getattr(evaluated, "lowest_hp_ratio", None)
            trainer_readiness = getattr(evaluated, "trainer_readiness", None)
            trainer_floor = getattr(trainer_readiness, "required_hp_ratio", None)
            below_trainer_floor = (
                isinstance(lowest_hp_ratio, (int, float))
                and isinstance(trainer_floor, (int, float))
                and lowest_hp_ratio < trainer_floor
            )
            recovery_route_ready = (
                getattr(getattr(evaluated, "recovery_availability", None), "value", None) == "known"
                and bool(getattr(recovery, "center_available", False))
                and bool(getattr(recovery, "safe_to_reach_center", False))
                and getattr(recovery, "route", None) is not None
            )
            recovery_distance = getattr(recovery, "distance_to_center", None)
            short_recovery = (
                isinstance(recovery_distance, (int, float))
                and recovery_distance <= self._readiness_policy.opportunistic_detour_threshold
            )
            trainer_recovery_required = (
                trainer_battle_owner
                and recovery_route_ready
                and (
                    below_trainer_floor
                    or (
                        isinstance(lowest_hp_ratio, (int, float))
                        and lowest_hp_ratio <= self._readiness_policy.opportunistic_hp_ratio
                        and (
                            short_recovery
                            # A mandatory trainer is a deliberate safety
                            # boundary. If its readiness assessment still
                            # says PREPARE, spend the known safe Center route
                            # even when the route is just beyond the generic
                            # opportunistic detour budget. Otherwise the
                            # preparation ceiling can convert a damaged,
                            # single-member party into direct exposure.
                            or getattr(getattr(trainer_readiness, "status", None), "value", None)
                            == "prepare"
                        )
                    )
                )
            )
            if trainer_recovery_required and evaluated.readiness_decision is not ReadinessDecision.RECOVER:
                previous_decision = evaluated.readiness_decision
                evaluated = replace(
                    evaluated,
                    readiness_decision=ReadinessDecision.RECOVER,
                    readiness_reason=ReadinessReason.OPPORTUNISTIC_RECOVERY,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_DECISION: "
                        f"objective={objective_id!r} decision='RECOVER' "
                        f"reason='trainer_precondition' previous_decision={getattr(previous_decision, 'value', None)!r} "
                        f"hp_ratio={lowest_hp_ratio!r} trainer_floor={trainer_floor!r} "
                        f"distance={recovery_distance!r} action='mount_before_trainer'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_DECISION",
                )
            # Compose the executable plan only after all readiness ownership
            # decisions above are final. In particular, a trainer
            # precondition may promote a healthy-looking parent route to
            # opportunistic recovery once a verified Center route is known.
            # The plan must be rebuilt from that promoted diagnostic so its
            # RecoveryStop is available to the handoff below.
            plan = build_campaign_plan(
                execution.objective,
                evaluated,
                policy=self._readiness_policy,
            )
            self._emit_campaign_task_diagnostics(observed_state, selection)
            # The selected-objective snapshot is emitted before readiness has
            # inspected the normal route. If that inspection discovers an
            # unavoidable trainer, publish the enriched facts at this same
            # campaign boundary so the operator sees the checkpoint and EXP
            # budget before the tactical loop can enter the battle.
            if (
                getattr(context, "campaign_trainer_checkpoint", None) is not None
                and getattr(context, "campaign_status", None) is not None
            ):
                self._emit_campaign_snapshot(context.campaign_status)
            readiness_map = evaluated.current_map
            readiness_coordinates = evaluated.current_coordinates
            encounter_stage = getattr(context, "campaign_encounter_stage", None)
            encounter_stage_fingerprint = (
                getattr(encounter_stage, "current_map", None),
                getattr(encounter_stage, "next_map", None),
                getattr(encounter_stage, "world_signature", None),
                getattr(encounter_stage, "readiness", None),
                bool(getattr(encounter_stage, "blocked", False)),
                tuple(getattr(encounter_stage, "known_hazard_ids", ()) or ()),
                tuple(getattr(encounter_stage, "unresolved_hazard_ids", ()) or ()),
                tuple(
                    getattr(checkpoint, "checkpoint_id", None)
                    for checkpoint in tuple(getattr(encounter_stage, "known_hazard_checkpoints", ()) or ())
                ),
            ) if encounter_stage is not None else None
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_RESULT: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"evaluation_id={id(evaluated)!r} objective={objective_id!r} "
                    f"tactical_goal={execution.tactical_goal!r} "
                    f"decision={evaluated.readiness_decision.value!r} "
                    f"reason={evaluated.readiness_reason.value!r} "
                    f"overworld_availability={evaluated.overworld_availability.value!r} "
                    f"overworld_reason={getattr(evaluated, 'overworld_reason', None)!r} "
                    f"navigation_availability={evaluated.navigation_availability.value!r} "
                    f"provider_result_type={type(readiness_input).__name__!r} "
                    f"provider_state_type={type(getattr(self._readiness_provider, '__self__', None)).__name__!r} "
                    f"scheduler_cache_hit={getattr(getattr(self._readiness_provider, '__self__', None), 'last_observation_was_cache_hit', None)!r} "
                    f"encounter_stage_fingerprint={encounter_stage_fingerprint!r} "
                    f"map={evaluated.current_map!r} coordinates={evaluated.current_coordinates!r}"
                ),
                trace=True,
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_DECISION: "
                    f"frame={getattr(getattr(evaluated, 'snapshot', None), 'frame', None)!r} "
                    f"evaluation_id={id(evaluated)} objective={objective_id!r} "
                    f"decision={evaluated.readiness_decision.value if evaluated.readiness_decision else None} "
                    f"reason={evaluated.readiness_reason.value if evaluated.readiness_reason else None} "
                        f"recover_accepted={evaluated.readiness_decision is ReadinessDecision.RECOVER} "
                        f"recovery_factory_available={self._recovery_factory is not None} "
                        f"recovery_calculation_pending={getattr(getattr(evaluated, 'recovery', None), 'calculation_pending', False)!r} "
                        f"recovery_route_available={getattr(getattr(evaluated, 'recovery', None), 'route', None) is not None!r} "
                        f"recovery_center={getattr(getattr(evaluated, 'recovery', None), 'center_location', None)!r} "
                        f"recovery_owner={self._accepted_recovery_objective_id!r} "
                        f"recovery_not_invoked_reason={None if evaluated.readiness_decision is ReadinessDecision.RECOVER else 'decision_not_recover'} "
                    f"controller_phase_before={self._execution_phase}"
                ),
                trace=True,
            )
            schedule = getattr(getattr(self._readiness_provider, "__self__", None), "state", None)
            readiness_boundary_key = (
                objective_id,
                getattr(schedule, "refresh_count", None),
                getattr(evaluated, "current_map", None),
                getattr(getattr(evaluated, "readiness_decision", None), "value", None),
                getattr(getattr(evaluated, "readiness_reason", None), "value", None),
                getattr(getattr(evaluated, "overworld_availability", None), "value", None),
                getattr(getattr(evaluated, "resource_availability", None), "value", None),
            )
            if readiness_boundary_key != self._last_readiness_boundary_key:
                self._last_readiness_boundary_key = readiness_boundary_key
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_BOUNDARY: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={objective_id!r} refresh_count={getattr(schedule, 'refresh_count', None)!r} "
                        f"tick_count={getattr(schedule, 'tick_count', None)!r} "
                        f"current_map={evaluated.current_map!r} "
                        f"decision={getattr(getattr(evaluated, 'readiness_decision', None), 'value', None)!r} "
                        f"reason={getattr(getattr(evaluated, 'readiness_reason', None), 'value', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_READINESS_BOUNDARY",
                )
            diagnostic_print(
                lambda: "CAMPAIGN_READINESS_LIFECYCLE: "
                + json.dumps(
                    {
                        "controller_tick": schedule.tick_count if schedule is not None else None,
                        "objective_id": objective_id,
                        "current_map": evaluated.current_map,
                        "coordinates": evaluated.current_coordinates,
                        "lowest_hp_ratio": evaluated.lowest_hp_ratio,
                        "usable_pokemon_count": evaluated.usable_count,
                        "trainer_hazard_state": evaluated.trainers,
                        "overworld_reason": getattr(evaluated, "overworld_reason", None),
                        "navigation_availability": evaluated.navigation_availability.value,
                        "recovery_availability": evaluated.recovery_availability.value,
                        "readiness_decision": evaluated.readiness_decision.value,
                        "readiness_reason": evaluated.readiness_reason.value,
                        "execution_phase": self._execution_phase,
                        "recovery_status": self._recovery_status,
                        "schedule": (
                            {
                                "status": schedule.status,
                                "age_ticks": schedule.age_ticks,
                                "refresh_count": schedule.refresh_count,
                                "tick_count": schedule.tick_count,
                                "invalidation_reason": schedule.invalidation_reason,
                            }
                            if schedule is not None
                            else None
                        ),
                    },
                    sort_keys=True,
                    default=str,
                ),
                trace=True,
            )
            from modules.route102_diagnostics import emit as route102_emit

            route102_emit(
                "readiness",
                controller_tick=schedule.tick_count if schedule is not None else None,
                execution_phase=self._execution_phase,
                objective_id=objective_id,
                objective_status=getattr(execution.objective, "status", None),
                tactical_goal=repr(execution.tactical_goal),
                readiness_decision=evaluated.readiness_decision.value,
                readiness_reason=evaluated.readiness_reason.value,
                scheduler=(
                    schedule.__dict__ if schedule is not None and hasattr(schedule, "__dict__") else repr(schedule)
                ),
                current_map=evaluated.current_map,
                coordinates=evaluated.current_coordinates,
                lowest_hp_ratio=evaluated.lowest_hp_ratio,
                usable_pokemon_count=evaluated.usable_count,
                trainer_availability=evaluated.trainer_availability.value,
                trainers=evaluated.trainers,
                recovery_availability=evaluated.recovery_availability.value,
                recovery_status=self._recovery_status,
            )
            pending_evidence = any(
                _readiness_calculation_pending(value)
                for value in (readiness_input, evaluated)
            )
            timed_out_evidence = any(
                getattr(value, "route_analysis_timed_out", False)
                or getattr(getattr(value, "recovery", None), "calculation_timed_out", False)
                for value in (readiness_input, evaluated)
            )
            hard_evidence = any(
                _readiness_requires_hard_recovery(value, self._readiness_policy)
                for value in (readiness_input, evaluated)
            )
            hard_evidence = hard_evidence or (
                trainer_battle_owner and pending_evidence
            )
            advisory_evidence = (
                (pending_evidence or timed_out_evidence)
                and not hard_evidence
                and not trainer_battle_owner
            )
            if advisory_evidence and evaluated.readiness_decision is not ReadinessDecision.UNKNOWN:
                self._advisory_route_polling = True
                self._refresh_required = True
                self._refresh_required_reason = "advisory_route_polling"
            elif not advisory_evidence:
                self._advisory_route_polling = False
            trainer_battle_unknown = (
                evaluated.readiness_reason is ReadinessReason.TRAINER_BATTLE_UNKNOWN
                and _is_trainer_goal(execution.tactical_goal)
            )
            if trainer_battle_unknown:
                self._pending_trainer_handoff = (objective_id, execution.tactical_goal)
            if evaluated.readiness_decision is ReadinessDecision.UNKNOWN:
                # An incomplete observation is a defer/reobserve boundary, not
                # permission to transfer ownership to recovery.  In
                # particular, battle and script transitions can temporarily
                # hide the overworld while the campaign context remains the
                # rightful owner. A targetless observation capability still
                # needs to be mounted at this boundary, however: it is the
                # capability that can observe the script handoff and wait for
                # the overworld to become actionable. Leaving it unmounted
                # strands the controller with the completed capability's GUI
                # status and no loop to advance on the next frame.
                if self._execution_phase == "RECOVERY_WAIT":
                    # The worker has completed with an unavailable/invalid
                    # result rather than remaining pending. Release the
                    # polling ownership so the normal bounded retry policy
                    # can decide whether to replan or resume campaign work.
                    self._execution_phase = "CAMPAIGN"
                    self._recovery_status = "RETRYABLE_UNAVAILABLE"
                diagnostic_print(
                    lambda: f"CAMPAIGN_READINESS_DEFERRED: controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} objective={objective_id!r} reason={evaluated.readiness_reason.value!r}",
                    trace=True,
                )
                input_pending = _readiness_calculation_pending(readiness_input)
                evaluated_pending = _readiness_calculation_pending(evaluated)
                recovery_pending = any(
                    getattr(getattr(value, "recovery", None), "calculation_pending", False)
                    for value in (readiness_input, evaluated)
                )
                route_pending = any(
                    getattr(value, "route_analysis_pending", False)
                    for value in (readiness_input, evaluated)
                )
                calculation_timed_out = any(
                    getattr(value, "route_analysis_timed_out", False)
                    or getattr(getattr(value, "recovery", None), "calculation_timed_out", False)
                    for value in (readiness_input, evaluated)
                )
                # Healthy ordinary navigation may continue while the
                # campaign-level route envelope is being computed.  Agent
                # Control still validates the executable movement plan from
                # the live world each frame. Recovery and explicit trainer
                # preparation keep their existing hard wait until their
                # evidence is complete.
                # A low-health snapshot is safety evidence, but it is not by
                # itself evidence that a recovery calculation is in flight.
                # In particular, an OVERWORLD_UNAVAILABLE observation can
                # still carry the last party HP values from before a script
                # or map transition. Treating those stale values as a
                # pending recovery route strands the controller in
                # RECOVERY_WAIT and bypasses the bounded UNKNOWN recheck.
                hard_recovery_pending = any(
                    _readiness_requires_hard_recovery(value, self._readiness_policy)
                    and _readiness_calculation_pending(value)
                    for value in (readiness_input, evaluated)
                )
                hard_recovery_pending = hard_recovery_pending or (
                    wild_battle_recovery_active
                    and pending_evidence
                )
                hard_trainer_pending = trainer_battle_owner and (recovery_pending or route_pending)
                calculation_pending = (
                    hard_recovery_pending
                    or hard_trainer_pending
                )
                advisory_pending = (
                    (recovery_pending or route_pending or calculation_timed_out)
                    and not calculation_pending
                    and not trainer_battle_owner
                )
                if advisory_pending:
                    # Keep the scheduler polling the same semantic request,
                    # but leave tactical ownership intact. The step loop will
                    # still advance the mounted capability after refresh().
                    self._advisory_route_polling = True
                    self._refresh_required = True
                    self._refresh_required_reason = "advisory_route_polling"
                else:
                    self._advisory_route_polling = False
                scripted_interaction_boundary = (
                    execution.tactical_goal is not None
                    and evaluated.game_state not in {"BATTLE", "BATTLE_STARTING", "BATTLE_ENDING"}
                    and _actionable_scripted_interaction_is_active()
                )
                targetless_capability_boundary = execution.capability is not None and execution.tactical_goal is None
                targetless_story_capability = (
                    targetless_capability_boundary
                    and getattr(execution.objective, "task_kind", None) == "required"
                    and getattr(execution.objective, "objective_id", None) not in _RESTOCK_OBJECTIVE_IDS
                    and not (
                        isinstance(getattr(execution.objective, "objective_id", None), str)
                        and getattr(execution.objective, "objective_id", "").startswith("obtain_encounter:")
                    )
                )
                targetless_recovery_uncertain = (
                    targetless_capability_boundary
                    and not targetless_story_capability
                    and evaluated.lowest_hp_ratio is not None
                    and evaluated.lowest_hp_ratio <= self._readiness_policy.opportunistic_hp_ratio
                    and evaluated.readiness_reason
                    in {
                        ReadinessReason.OPPORTUNISTIC_ROUTE_UNAVAILABLE,
                        ReadinessReason.RECOVERY_CAPABILITY_UNKNOWN,
                        ReadinessReason.RESOURCE_INFORMATION_UNKNOWN,
                    }
                    and (
                        evaluated.route_analysis is None
                        or getattr(evaluated.recovery_availability, "value", None) != "known"
                        or getattr(getattr(evaluated, "recovery", None), "calculation_pending", False)
                    )
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_UNKNOWN_DIAGNOSTIC: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={objective_id!r} reason={evaluated.readiness_reason.value!r} "
                        f"input_pending={input_pending!r} evaluated_pending={evaluated_pending!r} "
                        f"input_route_analysis_pending={getattr(readiness_input, 'route_analysis_pending', None)!r} "
                        f"evaluated_route_analysis_pending={getattr(evaluated, 'route_analysis_pending', None)!r} "
                        f"input_recovery_pending={getattr(getattr(readiness_input, 'recovery', None), 'calculation_pending', None)!r} "
                        f"evaluated_recovery_pending={getattr(getattr(evaluated, 'recovery', None), 'calculation_pending', None)!r} "
                        f"recovery_timed_out={getattr(getattr(evaluated, 'recovery', None), 'calculation_timed_out', None)!r} "
                        f"recovery_error={getattr(getattr(evaluated, 'recovery', None), 'observation_error', None)!r} "
                        f"route_analysis_present={getattr(evaluated, 'route_analysis', None) is not None!r} "
                        f"scripted_interaction_boundary={scripted_interaction_boundary!r} "
                        f"scheduler_cache_hit={getattr(getattr(self._readiness_provider, '__self__', None), 'last_observation_was_cache_hit', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_READINESS_UNKNOWN_DIAGNOSTIC",
                )
                lead_rotation_marker = getattr(context, "campaign_trainer_lead_rotation", None)
                lead_rotation_owned = self._tactical_loop is not None and _campaign_trainer_lead_rotation_active()
                if lead_rotation_owned:
                    trainer_id = (
                        lead_rotation_marker.get("trainer_id")
                        if isinstance(lead_rotation_marker, dict)
                        else getattr(lead_rotation_marker, "trainer_id", None)
                    )
                    # Opening the party menu intentionally makes the
                    # overworld unavailable. The suspended exposure loop is
                    # still the owner of the emulator and must resume so it
                    # can finish the swap and close both menus.
                    self._readiness_recheck_pending = False
                    self._pending_readiness_objective_id = None
                    self._deferred_handoff_requires_refresh = False
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "CAMPAIGN"
                    self.transition_reason = "trainer lead rotation owns party menu"
                    self._publish_status_for_execution(
                        execution.objective,
                        execution.tactical_goal,
                        f"Rotating lead for {trainer_id}" if trainer_id is not None else "Rotating trainer lead",
                    )
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_LEAD_ROTATION_OWNERSHIP: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"objective={objective_id!r} trainer={trainer_id!r} "
                            "readiness_reason='OVERWORLD_UNAVAILABLE' action='preserve_tactical_loop'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_LEAD_ROTATION_OWNERSHIP",
                    )
                    return self.state
                if (
                    advisory_pending
                    and not scripted_interaction_boundary
                    and self._tactical_loop is not None
                    and self.current_objective_id == objective_id
                    and self._tactical_loop_objective_id == objective_id
                    and self.current_tactical_goal == execution.tactical_goal
                ):
                    self._readiness_recheck_pending = False
                    self._pending_readiness_objective_id = None
                    self._deferred_handoff_requires_refresh = False
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "CAMPAIGN"
                    self.transition_reason = "advisory readiness pending; preserving tactical loop"
                    self._publish_status_for_execution(
                        execution.objective,
                        execution.tactical_goal,
                        "Continuing route while advisory readiness analysis runs",
                    )
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_READINESS_ADVISORY_POLL: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"objective={objective_id!r} recovery_pending={recovery_pending!r} "
                            f"route_pending={route_pending!r} timed_out={calculation_timed_out!r} "
                            "action='preserve_tactical_loop'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_READINESS_ADVISORY_POLL",
                    )
                    return self.state
                if (calculation_pending or targetless_recovery_uncertain) and not scripted_interaction_boundary:
                    # A worker owns a valid snapshot, but its route result is
                    # not ready. Do not let the old tactical generator keep
                    # advancing while the party may require recovery; the
                    # next controller frame must poll the same request.
                    self._readiness_recheck_pending = False
                    self._pending_readiness_objective_id = None
                    readiness_status = (
                        "Calculating recovery route"
                        if calculation_pending
                        else "Waiting / recovery readiness temporarily unavailable"
                    )
                    self._publish_status_for_execution(
                        execution.objective,
                        execution.tactical_goal,
                        readiness_status,
                    )
                    old_loop_id = id(self._tactical_loop) if self._tactical_loop is not None else None
                    self._tactical_loop = None
                    self._tactical_loop_objective_id = None
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    if trainer_battle_unknown:
                        self.last_execution = replace(execution, capability=None)
                    self._deferred_handoff_requires_refresh = True
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "RECOVERY_WAIT" if calculation_pending else "CAMPAIGN"
                    self._recovery_status = "CALCULATING" if calculation_pending else "RETRYABLE_UNAVAILABLE"
                    self._advisory_route_polling = False
                    self._refresh_required = True
                    self._refresh_required_reason = "recovery_calculation_pending" if calculation_pending else "recovery_readiness_unresolved"
                    self.transition_reason = (
                        "readiness deferred: recovery route calculation pending"
                        if calculation_pending
                        else "readiness deferred: targetless recovery route is unresolved"
                    )
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_READINESS_CALCULATION_PENDING: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"objective={objective_id!r} old_loop_id={old_loop_id!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_READINESS_CALCULATION_PENDING",
                    )
                    return self.state
                if trainer_battle_unknown:
                    # A mandatory trainer with no resolved roster is a hard
                    # safety boundary. Keep the promoted trainer goal as the
                    # owner, but do not mount either the original capability
                    # or a generic trainer route. Retry on a lifecycle event
                    # (or the one scheduler recheck) after the ROM facts can
                    # be observed again.
                    self._tactical_loop = None
                    self._tactical_loop_objective_id = None
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    self._deferred_handoff_requires_refresh = False
                    self._refresh_required = False
                    self._refresh_required_reason = None
                    self.status = CampaignControllerStatus.UNKNOWN
                    self._publish_status_for_execution(
                        execution.objective,
                        execution.tactical_goal,
                        "Waiting for mandatory trainer roster",
                    )
                    self.transition_reason = "mandatory trainer roster unresolved; waiting for re-observation"
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TRAINER_HANDOFF_BLOCKED: "
                            f"controller_id={id(self)!r} objective={objective_id!r} "
                            f"goal={execution.tactical_goal!r} reason='TRAINER_BATTLE_UNKNOWN'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_TRAINER_HANDOFF_BLOCKED",
                    )
                    return self.state
                # A targetful campaign objective with no executable loop has
                # no safe work to perform while ordinary readiness is
                # UNKNOWN. Keep the campaign owner, but do not fall through
                # to the generic mount path: that path would either create a
                # route from incomplete map facts or leave the refresh flag
                # active and call readiness again on every frame.
                #
                # Preserve the one compatibility retry for a transient
                # overworld observation. Stable policy results such as
                # RECOVERY_UNAVAILABLE are deliberately left cached until an
                # explicit lifecycle event invalidates them.
                if (
                    not scripted_interaction_boundary
                    and not advisory_pending
                    and not targetless_capability_boundary
                ):
                    scheduler = getattr(self._readiness_provider, "__self__", None)
                    cached_observation = bool(getattr(scheduler, "last_observation_was_cache_hit", False))
                    self._readiness_recheck_pending = (
                        not forced_readiness_recheck
                        and not cached_observation
                        and _readiness_requires_recheck(evaluated)
                    )
                    self._pending_readiness_objective_id = (
                        objective_id if self._readiness_recheck_pending else None
                    )
                    stale_loop = self._tactical_loop is not None and (
                        self.current_objective_id != objective_id
                        or self._tactical_loop_objective_id != objective_id
                    )
                    if stale_loop:
                        # The planner has already selected a new objective,
                        # so the old capability cannot remain executable while
                        # readiness for the new one is unresolved.
                        self._tactical_loop = None
                        self._tactical_loop_objective_id = None
                    capability_handoff_pending = execution.capability is not None or promoted_trainer_handoff
                    self._deferred_handoff_requires_refresh = capability_handoff_pending
                    self._refresh_required = capability_handoff_pending
                    self._refresh_required_reason = (
                        "readiness_deferred_handoff" if capability_handoff_pending else None
                    )
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "CAMPAIGN"
                    self.transition_reason = f"readiness deferred: {evaluated.readiness_reason.value}"
                    self._publish_status_for_execution(
                        execution.objective,
                        execution.tactical_goal,
                        "Waiting / readiness temporarily unavailable",
                    )
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_READINESS_DEFERRED_IDLE: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"objective={objective_id!r} "
                            f"recheck_pending={self._readiness_recheck_pending!r} "
                            f"forced={forced_readiness_recheck!r} "
                            f"reason={evaluated.readiness_reason.value!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_READINESS_DEFERRED_IDLE",
                    )
                    return self.state
                # The scheduler retains this bounded UNKNOWN observation, so
                # do not turn it into a per-frame invalidation loop.  An
                # explicit battle-end request has already forced the next
                # observation; if that observation is still transient, let
                # its bounded cache/age policy handle the retry.  Retain the
                # one retry for custom providers that do not expose a cache,
                # preserving the controller's historical startup behavior.
                scheduler = getattr(self._readiness_provider, "__self__", None)
                cached_observation = bool(getattr(scheduler, "last_observation_was_cache_hit", False))
                self._readiness_recheck_pending = (
                    not forced_readiness_recheck and not cached_observation and _readiness_requires_recheck(evaluated)
                )
                self._pending_readiness_objective_id = objective_id if self._readiness_recheck_pending else None
                self._publish_status_for_execution(
                    execution.objective,
                    execution.tactical_goal,
                    "Waiting / readiness temporarily unavailable",
                )
                targetless_capability_boundary = execution.capability is not None and execution.tactical_goal is None
                capability_remount_pending = (
                    execution.tactical_goal is not None
                    and self._tactical_loop is None
                    and (execution.capability is not None or promoted_trainer_handoff)
                )
                # A targetful navigation loop must not keep issuing movement
                # while readiness is UNKNOWN.  The route may still be the
                # pre-promotion map goal even though the next observation is
                # about to promote a known trainer checkpoint (Calvin is the
                # early example).  Advancing that stale loop makes the bot
                # walk around the current map while the GUI says it is
                # waiting, and can carry it into an avoidable oscillation at
                # a Center or town boundary.  Script-owned tactical loops
                # are the exception: they need to advance in order to release
                # the ROM's dialogue/native-script owner.
                targetful_loop_boundary = (
                    self._tactical_loop is not None
                    and not targetless_capability_boundary
                    and not scripted_interaction_boundary
                )
                stale_interaction_handoff = (
                    scripted_interaction_boundary
                    and self._tactical_loop is not None
                    and (
                        self.current_objective_id != objective_id
                        or self._tactical_loop_objective_id != objective_id
                    )
                )
                # A capability can finish one local route exactly as the ROM
                # enters a battle/script transition.  The next readiness
                # observation is then UNKNOWN and there is no loop left to
                # advance.  Keep the refresh request alive until readiness is
                # actionable so the capability can be mounted again; without
                # this, step() clears the request and returns READY/loop_absent
                # forever.
                self._deferred_handoff_requires_refresh = capability_remount_pending
                if targetful_loop_boundary:
                    old_loop_id = id(self._tactical_loop)
                    old_loop_objective = self._tactical_loop_objective_id
                    self._tactical_loop = None
                    self._tactical_loop_objective_id = None
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    self._deferred_handoff_requires_refresh = True
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TACTICAL_LOOP_INVALIDATED: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"loop_id={old_loop_id!r} old_objective={old_loop_objective!r} "
                            f"new_objective={objective_id!r} "
                            "reason='readiness_deferred_targetful_navigation'"
                        ),
                        trace=True,
                    )
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "CAMPAIGN"
                    self.transition_reason = f"readiness deferred: {evaluated.readiness_reason.value}"
                    diagnostic_print(
                        lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='readiness_deferred' status={self.status.value!r} reason={self.transition_reason!r}",
                        trace=True,
                    )
                    return self.state
                elif stale_interaction_handoff:
                    old_loop_id = id(self._tactical_loop)
                    old_loop_objective = self._tactical_loop_objective_id
                    self._tactical_loop = None
                    self._tactical_loop_objective_id = None
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    self._deferred_handoff_requires_refresh = True
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TACTICAL_LOOP_INVALIDATED: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"loop_id={old_loop_id!r} old_objective={old_loop_objective!r} "
                            f"new_objective={objective_id!r} "
                            "reason='readiness_deferred_interaction_handoff'"
                        ),
                        trace=True,
                    )
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "CAMPAIGN"
                    self.transition_reason = f"readiness deferred: {evaluated.readiness_reason.value}"
                    diagnostic_print(
                        lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='readiness_deferred' status={self.status.value!r} reason={self.transition_reason!r}",
                        trace=True,
                    )
                    return self.state
                elif not targetless_capability_boundary and not scripted_interaction_boundary and not advisory_pending:
                    # The planner may advance to a new tactical objective at
                    # the same frame that the ROM still hides the overworld
                    # behind a script transition.  Do not leave the prior
                    # generator executable in that state: ``step()`` would
                    # otherwise advance the old capability while
                    # ``last_selection`` and the GUI already describe the
                    # new one.  This is especially visible at Birch's lab,
                    # where the Pokédex capability can still be mounted while
                    # the Poké Ball objective becomes authoritative.
                    stale_loop = self._tactical_loop is not None and (
                        self.current_objective_id != objective_id or self._tactical_loop_objective_id != objective_id
                    )
                    if stale_loop:
                        self._deferred_handoff_requires_refresh = True
                        old_loop_id = id(self._tactical_loop)
                        old_loop_objective = self._tactical_loop_objective_id
                        self._tactical_loop = None
                        self._tactical_loop_objective_id = None
                        self.current_objective_id = objective_id
                        self.current_tactical_goal = execution.tactical_goal
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_TACTICAL_LOOP_INVALIDATED: "
                                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                                f"loop_id={old_loop_id!r} old_objective={old_loop_objective!r} "
                                f"new_objective={objective_id!r} reason='readiness_deferred_handoff'"
                            ),
                            trace=True,
                        )
                    elif capability_remount_pending:
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_TACTICAL_LOOP_REMOUNT_DEFERRED: "
                                f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                                f"objective={objective_id!r} reason='readiness_deferred_without_loop'"
                            ),
                            trace=True,
                        )
                    self.status = CampaignControllerStatus.READY
                    self._execution_phase = "CAMPAIGN"
                    self.transition_reason = f"readiness deferred: {evaluated.readiness_reason.value}"
                    diagnostic_print(
                        lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='readiness_deferred' status={self.status.value!r} reason={self.transition_reason!r}",
                        trace=True,
                    )
                    return self.state
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_READINESS_CAPABILITY_MOUNT: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={objective_id!r} reason={evaluated.readiness_reason.value!r} "
                        f"action={'preserve_scripted_interaction' if scripted_interaction_boundary else 'mount_targetless_capability'}"
                    ),
                    trace=True,
                )
            self._readiness_recheck_pending = False
            self._pending_readiness_objective_id = None
            if not trainer_battle_unknown:
                self._pending_trainer_handoff = None
            trainer_preparation_owner = (
                _is_trainer_goal(execution.tactical_goal)
                or getattr(evaluated, "trainer_engagement_required", False)
            )
            if evaluated.readiness_decision is ReadinessDecision.PREPARE and trainer_preparation_owner:
                ceiling_reached, current_level, ceiling = _trainer_preparation_ceiling_reached(evaluated)
                trainer_readiness = getattr(evaluated, "trainer_readiness", None)
                trainer_floor = getattr(trainer_readiness, "required_hp_ratio", None)
                lowest_hp_ratio = getattr(evaluated, "lowest_hp_ratio", None)
                below_trainer_floor = (
                    isinstance(trainer_floor, (int, float))
                    and not isinstance(trainer_floor, bool)
                    and isinstance(lowest_hp_ratio, (int, float))
                    and not isinstance(lowest_hp_ratio, bool)
                    and lowest_hp_ratio < float(trainer_floor)
                )
                if ceiling_reached and not below_trainer_floor:
                    # The party has reached the small, general overlevel
                    # allowance for this known trainer.  Do not mount another
                    # grass-training loop merely because a bounded projection
                    # did not produce a terminal proof; the engagement
                    # planner now owns the remaining modeled risk.
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TRAINER_PREPARATION_CEILING: "
                            f"objective={objective_id!r} current_level={current_level!r} "
                            f"ceiling={ceiling!r} preparation='complete' action='stop_training' "
                            "handoff='battle_plan' reason='continue_with_battle_plan'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_TRAINER_PREPARATION_CEILING",
                    )
                    evaluated = replace(
                        evaluated,
                        readiness_decision=ReadinessDecision.CONTINUE,
                        readiness_reason=ReadinessReason.TRAINER_PREPARATION_CEILING,
                        preparation_current_level=current_level,
                        preparation_ceiling_level=ceiling,
                    )
                    # The readiness object is also the source rendered in the
                    # campaign snapshot. Republish it after the handoff so
                    # the UI/log cannot retain PREPARE while the controller
                    # has already transferred ownership to battle exposure.
                    context.campaign_readiness_diagnostic = evaluated
                    self._publish_status_for_execution(
                        execution.objective,
                        execution.tactical_goal,
                        f"Trainer preparation ceiling reached ({current_level}/{ceiling}); exposing trainer",
                    )
            if evaluated.readiness_decision is not ReadinessDecision.PREPARE:
                self._trainer_preparation_checkpoint_key = None
                self._trainer_preparation_last_target = None
            if evaluated.readiness_decision is ReadinessDecision.PREPARE:
                if evaluated.readiness_reason is ReadinessReason.TRAINER_ROSTER_CHANGE:
                    pc_index = getattr(getattr(evaluated, "trainer_readiness", None), "lead_party_index", None)
                    if not isinstance(pc_index, int) or pc_index >= 0:
                        self.status = CampaignControllerStatus.UNKNOWN
                        self._execution_phase = "CAMPAIGN"
                        self._refresh_required = True
                        self._refresh_required_reason = "trainer_roster_plan_unavailable"
                        self.transition_reason = "trainer roster plan is incomplete; retrying"
                        return self.state
                    from .resource_runtime import execute_trainer_roster_preparation

                    self._tactical_loop = execute_trainer_roster_preparation(pc_index)
                    self._execution_phase = "PREPARATION"
                    self._recovery_status = None
                    self.current_objective_id = objective_id
                    self.current_tactical_goal = execution.tactical_goal
                    self._tactical_loop_objective_id = objective_id
                    self.status = CampaignControllerStatus.READY
                    self._publish_status_for_execution(
                        execution.objective,
                        execution.tactical_goal,
                        f"Changing roster for known trainer lead (PC slot {pc_index})",
                    )
                    self.transition_reason = f"withdraw trainer lead before {objective_id}"
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TRAINER_ROSTER_PREPARATION_MOUNTED: "
                            f"objective={objective_id!r} pc_index={pc_index!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_TRAINER_ROSTER_PREPARATION_MOUNTED",
                    )
                    return self.state
                level_cap = self._active_preparation_level_cap()
                preparation = self._trainer_preparation_parameters(evaluated, level_cap=level_cap)
                if preparation is None:
                    # This should only be reachable if a custom readiness
                    # provider reports PREPARE without the checkpoint facts
                    # that make preparation safe. Keep campaign ownership and
                    # wait for a complete observation rather than engaging an
                    # unknown trainer or mounting a blind training route.
                    self.status = CampaignControllerStatus.UNKNOWN
                    self._execution_phase = "CAMPAIGN"
                    self._refresh_required = True
                    self._refresh_required_reason = "trainer_preparation_facts_unavailable"
                    context.campaign_status = CampaignStatus(
                        getattr(execution.objective, "description", None) or objective_id,
                        _campaign_target_for_execution(execution.objective, execution.tactical_goal),
                        "Trainer preparation facts unavailable; retrying autonomously",
                        execution.tactical_goal,
                    )
                    self.transition_reason = "known trainer or level-cap facts unavailable; retrying"
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TRAINER_PREPARATION_DEFERRED: "
                            f"objective={objective_id!r} reason='checkpoint or active level-cap facts unavailable'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_TRAINER_PREPARATION_DEFERRED",
                    )
                    return self.state
                training_map, target_level = preparation
                checkpoint_key = (
                    objective_id,
                    getattr(getattr(evaluated, "trainer_checkpoint", None), "checkpoint_id", None),
                    training_map,
                )
                escalating = checkpoint_key == self._trainer_preparation_checkpoint_key
                if escalating:
                    target_level = self._trainer_preparation_last_target + 1
                if level_cap is not None and target_level > level_cap:
                    self._tactical_loop = None
                    self._tactical_loop_objective_id = None
                    self._execution_phase = "BLOCKED"
                    self.status = CampaignControllerStatus.BLOCKED
                    self._refresh_required = False
                    self.transition_reason = "known trainer remains unsafe at active level cap; manual review required"
                    context.campaign_status = CampaignStatus(
                        getattr(execution.objective, "description", None) or objective_id,
                        _campaign_target_for_execution(execution.objective, execution.tactical_goal),
                        "Known trainer remains unsafe at active level cap; manual review required",
                        execution.tactical_goal,
                    )
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_TRAINER_PREPARATION_BLOCKED: "
                            f"objective={objective_id!r} checkpoint={checkpoint_key[1]!r} "
                            f"last_target={self._trainer_preparation_last_target!r} cap={level_cap!r} "
                            "reason='known trainer remained unsafe at the active level cap'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_TRAINER_PREPARATION_BLOCKED",
                    )
                    return self.state
                self._trainer_preparation_checkpoint_key = checkpoint_key
                self._trainer_preparation_last_target = target_level
                from .resource_runtime import (
                    execute_campaign_preparation,
                    trainer_checkpoint_ready_check,
                )

                self._campaign_plan = plan
                self.interrupted_objective_id = objective_id
                trainer_checkpoint = getattr(evaluated, "trainer_checkpoint", None)
                self._tactical_loop = execute_campaign_preparation(
                    training_map,
                    target_level=target_level,
                    ready_check=(
                        trainer_checkpoint_ready_check(trainer_checkpoint)
                        if trainer_checkpoint is not None
                        else None
                    ),
                )
                self._execution_phase = "PREPARATION"
                self._recovery_status = None
                self.current_objective_id = objective_id
                self.current_tactical_goal = execution.tactical_goal
                self._tactical_loop_objective_id = objective_id
                self.status = CampaignControllerStatus.READY
                self._publish_status_for_execution(
                    execution.objective,
                    execution.tactical_goal,
                    f"Preparing for known trainer battle (target level {target_level})",
                )
                self.transition_reason = f"prepare before {objective_id}"
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_PREPARATION_MOUNTED: "
                        f"objective={objective_id!r} checkpoint={getattr(getattr(evaluated, 'trainer_checkpoint', None), 'checkpoint_id', None)!r} "
                        f"training_map={training_map!r} target_level={target_level!r} cap={level_cap!r} "
                        f"escalated={escalating!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_PREPARATION_MOUNTED",
                )
                return self.state
            if (
                evaluated.readiness_decision is ReadinessDecision.RECOVER
                and evaluated.readiness_reason is ReadinessReason.OPPORTUNISTIC_RECOVERY
                and _restock_owns_mart_transaction(execution, evaluated)
            ):
                # Once a restock capability has entered a Mart, its purchase
                # and exit dialogue are one transaction.  Opportunistic
                # recovery may wait for that transaction to finish; otherwise
                # the controller can send the capability back out of the Mart
                # and immediately route back in for the same purchase.
                self._campaign_plan = CampaignPlan(
                    objective_id,
                    readiness_decision=ReadinessDecision.CONTINUE,
                    readiness_reason=ReadinessReason.OPPORTUNISTIC_ROUTE_UNAVAILABLE,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RESTOCK_RECOVERY_DEFERRED: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"objective={objective_id!r} map={evaluated.current_map!r} "
                        "reason='complete_mart_transaction_before_opportunistic_recovery'"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RESTOCK_RECOVERY_DEFERRED",
                )
            elif evaluated.readiness_decision is ReadinessDecision.RECOVER:
                stop = plan.recovery_stop
                route_analysis = getattr(readiness_input, "route_analysis", None)
                analysis_candidates = tuple(getattr(route_analysis, "candidates", ()) or ())
                critical_recovery = getattr(getattr(evaluated, "readiness_reason", None), "value", None) in {
                    ReadinessReason.CRITICAL_PARTY_HP.value,
                    ReadinessReason.NO_USABLE_POKEMON.value,
                    ReadinessReason.FAINTED_PARTY_MEMBER.value,
                    ReadinessReason.POISONED_PARTY.value,
                    ReadinessReason.STATUS_CONDITION.value,
                }
                eligible_candidates = tuple(
                    candidate
                    for candidate in analysis_candidates
                    if getattr(candidate, "reachable", False)
                    and getattr(candidate, "first_route", None) is not None
                    and (
                        critical_recovery
                        or (
                            getattr(candidate, "total_cost", None) is not None
                            and getattr(candidate, "detour", None) is not None
                        )
                    )
                )
                observed_recovery = getattr(readiness_input, "recovery", None)
                observed_route = getattr(observed_recovery, "route", None)
                selected_route = getattr(stop, "route", None) if stop is not None else None
                selected_metrics = getattr(selected_route, "metrics", None)
                fallback_selected = stop is not None and observed_route is not None and selected_route is observed_route
                print_recovery_plan(
                    objective_id=objective_id,
                    decision=getattr(evaluated.readiness_decision, "value", str(evaluated.readiness_decision)),
                    reason=getattr(getattr(evaluated, "readiness_reason", None), "value", None),
                    route_analysis_candidates=len(analysis_candidates),
                    route_analysis_eligible=len(eligible_candidates),
                    observed_route_available=observed_route is not None,
                    observed_center_available=getattr(observed_recovery, "center_available", None),
                    observed_healing_source_available=getattr(observed_recovery, "healing_source_available", None),
                    observed_safe_to_reach=getattr(observed_recovery, "safe_to_reach_center", None),
                    observed_destination=getattr(observed_recovery, "center_location", None),
                    stop_present=stop is not None,
                    stop_destination=getattr(stop, "destination", None) if stop is not None else None,
                    stop_route_actions=(len(getattr(selected_route, "actions", ())) if selected_route is not None else None),
                    stop_route_cost=getattr(selected_metrics, "total_route_cost", None),
                    stop_execution_kind=(getattr(stop, "execution_kind", None) if stop is not None else None),
                    detail=(
                        "using completed recovery route fallback; parent route will be replanned after healing"
                        if fallback_selected
                        else "using route-analysis recovery candidate"
                        if stop is not None and getattr(stop, "execution_kind", "center_route") == "center_route"
                        else "using explicitly selected field recovery item"
                        if stop is not None
                        else "no executable RecoveryStop was composed"
                    ),
                )
                self._campaign_plan = plan
                self.interrupted_objective_id = objective_id
                self._accepted_recovery_objective_id = objective_id
                if _restock_transaction_is_active():
                    context.campaign_restock_phase = "recovery_active"
                if stop is None:
                    print_recovery_handoff(
                        phase="mount_blocked",
                        objective_id=objective_id,
                        execution_phase=self._execution_phase,
                        detail="readiness accepted recovery but supplied no executable RecoveryStop",
                    )
                    if (
                        evaluated.game_state != "OVERWORLD"
                        or evaluated.overworld_availability is not None
                        and evaluated.overworld_availability.value != "known"
                    ):
                        # A contradictory recovery result can still be
                        # supplied by a stale/custom readiness provider at a
                        # battle or script boundary.  Preserve campaign
                        # ownership and ask for a stable re-observation; do
                        # not turn the absence of a route into a fallback
                        # Center selection.
                        # A forced retry has already consumed the one
                        # compatibility re-observation. If the second
                        # observation is still contradictory, wait for a new
                        # lifecycle event instead of refreshing every frame.
                        self._readiness_recheck_pending = not forced_readiness_recheck
                        self._pending_readiness_objective_id = objective_id if self._readiness_recheck_pending else None
                        self.status = CampaignControllerStatus.READY
                        self._execution_phase = "CAMPAIGN"
                        self._publish_campaign_status("Recovery deferred; re-evaluating campaign readiness")
                        self.transition_reason = (
                            "recovery deferred: "
                            f"{evaluated.readiness_reason.value if evaluated.readiness_reason else 'recovery requested'}; "
                            "OVERWORLD_UNAVAILABLE (overworld observation unavailable)"
                        )
                        return self.state
                    # A missing stop is a retryable autonomous safety
                    # boundary. The controller must keep ownership while a
                    # fresh resource/route observation is obtained.
                    self.status = CampaignControllerStatus.UNKNOWN
                    self._execution_phase = "CAMPAIGN"
                    self._recovery_status = "RETRYABLE_UNAVAILABLE"
                    self._refresh_required = True
                    self._refresh_required_reason = "recovery_stop_unavailable"
                    context.campaign_status = recovery_status(None, "Recovery route unavailable; retrying autonomously")
                    self.transition_reason = "campaign planner produced no executable recovery stop; retrying"
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_RECOVERY_PLAN_UNAVAILABLE: "
                            f"controller_id={id(self)!r} objective={objective_id!r} "
                            "fallback='disabled'"
                        ),
                        trace=True,
                    )
                    return self.state
                diagnostic_print(
                    lambda: "recovery_start: "
                    + json.dumps(
                        {
                            "objective_id": objective_id,
                            "execution_phase": "RECOVERY",
                            "readiness_decision": evaluated.readiness_decision.value,
                            "readiness_reason": evaluated.readiness_reason.value,
                        },
                        sort_keys=True,
                        default=str,
                    ),
                    trace=True,
                )
                try:
                    route_origin = getattr(stop, "route_origin", None)
                    current_location = (evaluated.current_map, evaluated.current_coordinates)
                    if (
                        route_origin is not None
                        and not _campaign_locations_match(route_origin, current_location)
                    ):
                        # The worker planned this route from a different tile.
                        # Reusing it would make the first movement action point
                        # at stale local geometry, so invalidate readiness and
                        # obtain a fresh route from the live origin.
                        self._tactical_loop = None
                        self._tactical_loop_objective_id = None
                        self._campaign_plan = CampaignPlan(objective_id)
                        self._accepted_recovery_objective_id = None
                        self.interrupted_objective_id = None
                        self._execution_phase = "CAMPAIGN"
                        self._recovery_status = "RETRYABLE_UNAVAILABLE"
                        self._advisory_route_polling = False
                        self._refresh_required = True
                        self._refresh_required_reason = "stale_recovery_route_origin"
                        scheduler = getattr(self._readiness_provider, "__self__", None)
                        invalidate = getattr(scheduler, "invalidate", None)
                        if callable(invalidate):
                            invalidate("stale_recovery_route_origin")
                        self.status = CampaignControllerStatus.UNKNOWN
                        context.campaign_status = recovery_status(
                            None,
                            "Recovery route became stale; recalculating from current location",
                        )
                        self.transition_reason = (
                            "discarded stale recovery route: "
                            f"origin={route_origin!r} current={current_location!r}"
                        )
                        diagnostic_print(
                            lambda: (
                                "CAMPAIGN_RECOVERY_ROUTE_STALE: "
                                f"objective={objective_id!r} origin={route_origin!r} "
                                f"current={current_location!r} action='recalculate'"
                            ),
                            trace=True,
                            prefix="CAMPAIGN_RECOVERY_ROUTE_STALE",
                        )
                        return self.state
                    selected_route = getattr(stop, "route", None)
                    selected_metrics = getattr(selected_route, "metrics", None)
                    print_recovery_handoff(
                        phase="mount_requested",
                        objective_id=objective_id,
                        execution_phase=self._execution_phase,
                        destination=getattr(stop, "destination", None),
                        source_id=getattr(getattr(stop, "healing_source", None), "source_id", None),
                        route_actions=(
                            len(getattr(selected_route, "actions", ()))
                            if selected_route is not None
                            else None
                        ),
                        route_cost=getattr(selected_metrics, "total_route_cost", None),
                        detail="controller accepted completed recovery route",
                    )
                    if self._recovery_factory is None:
                        raise RuntimeError("recovery factory is unavailable")
                    self._tactical_loop = self._recovery_factory(stop)
                    diagnostic_print(
                        lambda: f"CAMPAIGN_RECOVERY_GENERATOR: event=created controller_id={id(self)!r} loop_id={id(self._tactical_loop)!r} frame={getattr(context, 'frame', None)!r}",
                        trace=True,
                    )
                    diagnostic_print(
                        lambda: f"CAMPAIGN_RECOVERY_FACTORY: evaluation_id={id(evaluated)} invoked=True method={getattr(self._recovery_factory, '__name__', type(self._recovery_factory).__name__)}",
                        trace=True,
                    )
                except Exception as error:
                    self._tactical_loop = None
                    print_recovery_handoff(
                        phase="mount_failed",
                        objective_id=objective_id,
                        execution_phase=self._execution_phase,
                        destination=getattr(stop, "destination", None),
                        detail=f"recovery factory failed: {type(error).__name__}: {error}",
                    )
                    if getattr(error, "run_lost", False):
                        self._terminal(
                            CampaignExecutionResult(
                                execution.objective,
                                CampaignExecutionStatus.FAILED,
                                f"run lost during recovery: {error}",
                                execution.execution_id,
                            ),
                            CampaignControllerStatus.RUN_LOST,
                        )
                        return self.state
                    self.status = CampaignControllerStatus.UNKNOWN
                    self._execution_phase = "CAMPAIGN"
                    self._recovery_status = "RETRYABLE_UNAVAILABLE"
                    self._refresh_required = True
                    self._refresh_required_reason = "recovery_factory_exception"
                    context.campaign_status = recovery_status(None, "Recovery unavailable; retrying autonomously")
                    self.transition_reason = f"recovery unavailable; retrying: {error}"
                    diagnostic_print(
                        lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='recovery_unavailable' status={self.status.value!r} reason={self.transition_reason!r}",
                        trace=True,
                    )
                    return self.state
                self._execution_phase = "RECOVERY"
                self._recovery_status = "ACTIVE"
                self._wild_battle_recovery_pending = False
                self._advisory_route_polling = False
                self._recovery_keyerror_retries = 0
                self.current_objective_id = objective_id
                self.current_tactical_goal = execution.tactical_goal
                self._tactical_loop_objective_id = objective_id
                self.status = CampaignControllerStatus.READY
                context.campaign_status = recovery_status(None, "Recovery planned")
                self.transition_reason = f"recover before {objective_id}"
                print_recovery_handoff(
                    phase="route_mounted",
                    objective_id=objective_id,
                    execution_phase=self._execution_phase,
                    destination=getattr(stop, "destination", None),
                    source_id=getattr(getattr(stop, "healing_source", None), "source_id", None),
                    loop_id=id(self._tactical_loop),
                    detail="recovery generator mounted; awaiting first executable action",
                )
                diagnostic_print(
                    lambda: f"CAMPAIGN_REFRESH_RETURN: frame={getattr(context, 'frame', None)!r} branch='recovery_mounted' status={self.status.value!r} reason={self.transition_reason!r}",
                    trace=True,
                )
                return self.state
        elif self._readiness_recheck_pending:
            # A lifecycle event can arrive while an opening targetless
            # capability owns the controller.  That capability has no
            # meaningful overworld readiness input, so retaining the pending
            # flag would force ``step()`` to refresh every frame forever.
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_READINESS_RECHECK_DISCARDED: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"objective={objective_id!r} reason='readiness_not_applicable'"
                ),
                trace=True,
                prefix="CAMPAIGN_READINESS_RECHECK_DISCARDED",
            )
            self._readiness_recheck_pending = False
            self._pending_readiness_objective_id = None
        if (
            self.current_objective_id != objective_id
            or self.current_tactical_goal != execution.tactical_goal
            or self._tactical_loop is None
        ):
            self.current_objective_id = objective_id
            self.current_tactical_goal = execution.tactical_goal
            old_loop_id = id(self._tactical_loop) if self._tactical_loop else None
            old_loop_objective = self._tactical_loop_objective_id
            self._tactical_loop = (
                execution.capability()
                if execution.capability is not None
                else self._tactical_loop_factory(execution.tactical_goal)
            )
            self._tactical_loop_objective_id = objective_id
            self._deferred_handoff_requires_refresh = False
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_LOOP_BOUNDARY: "
                    f"controller_id={id(self)!r} old_loop_id={old_loop_id!r} old_objective={old_loop_objective!r} "
                    f"new_loop_id={id(self._tactical_loop)!r} new_objective={objective_id!r} reason='mount_or_replace'"
                ),
                trace=True,
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_LOOP_CREATED: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"loop_id={id(self._tactical_loop)!r} objective={objective_id!r} "
                    f"tactical_goal={execution.tactical_goal!r} phase={self._execution_phase!r} "
                    f"status={self.status.value!r} map={readiness_map!r} coordinates={readiness_coordinates!r}"
                ),
                trace=True,
            )
            self._publish_campaign_status("Campaign objective selected")
            self.transition_reason = f"selected {objective_id}"
        self.status = CampaignControllerStatus.READY
        return self.state

    def _retry_recovery_after_keyerror(self, error: KeyError) -> bool:
        """Remount the planner-owned recovery stop after a transient read.

        Recovery generators are deliberately discarded after an exception:
        resuming one could replay an input after the ROM has moved to a new
        interaction state.  The immutable CampaignPlan supplies the exact
        destination again, so this retry does not reopen recovery selection or
        choose a different healing source.
        """
        if (
            self._recovery_keyerror_retries >= _RECOVERY_KEYERROR_RETRY_LIMIT
            or self._campaign_plan is None
            or self._campaign_plan.recovery_stop is None
            or self._recovery_factory is None
        ):
            return False

        failed_loop = self._tactical_loop
        failed_loop_id = id(failed_loop) if failed_loop is not None else None
        stop = self._campaign_plan.recovery_stop
        self._recovery_keyerror_retries += 1
        if failed_loop is not None:
            close = getattr(failed_loop, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as close_error:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_RECOVERY_RETRY_CLOSE_FAILURE: "
                            f"controller_id={id(self)!r} loop_id={failed_loop_id!r} "
                            f"exception_type={type(close_error).__name__!r} exception={close_error!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_RECOVERY_RETRY_CLOSE_FAILURE",
                    )
        try:
            self._tactical_loop = self._recovery_factory(stop)
        except Exception as remount_error:
            self._tactical_loop = None
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_RETRY_MOUNT_FAILURE: "
                    f"controller_id={id(self)!r} attempt={self._recovery_keyerror_retries!r} "
                    f"exception_type={type(remount_error).__name__!r} exception={remount_error!r} "
                    f"traceback={traceback.format_exc()!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_RETRY_MOUNT_FAILURE",
            )
            return False

        self._execution_phase = "RECOVERY"
        self._recovery_status = "ACTIVE"
        self.status = CampaignControllerStatus.READY
        self._tactical_loop_objective_id = self.current_objective_id
        retry_intent = (
            "Recovery retrying after transient KeyError "
            f"({self._recovery_keyerror_retries}/{_RECOVERY_KEYERROR_RETRY_LIMIT})"
        )
        context.campaign_status = recovery_status(None, retry_intent)
        self.transition_reason = (
            "recovery observation raised transient KeyError; "
            f"remounted attempt {self._recovery_keyerror_retries}/{_RECOVERY_KEYERROR_RETRY_LIMIT}"
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_RETRY: "
                f"frame={getattr(context, 'frame', None)!r} "
                f"controller_id={id(self)!r} failed_loop_id={failed_loop_id!r} "
                f"new_loop_id={id(self._tactical_loop)!r} "
                f"attempt={self._recovery_keyerror_retries!r} "
                f"limit={_RECOVERY_KEYERROR_RETRY_LIMIT!r} "
                f"key={error.args!r} traceback={traceback.format_exc()!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_RETRY",
        )
        return True

    def _record_recovery_failure(self, error: Exception) -> bool:
        """Suppress an identical recovery loop after bounded retries.

        A transient ROM boundary should get a fresh observation and retry. If
        the same objective, exception type, and normalized message recur,
        continuing to remount the same recovery generator only repeats the
        same inputs. Escalate after a small bound so the run remains visible
        and actionable instead of appearing stuck forever.
        """

        if getattr(error, "run_lost", False):
            return False
        normalized = " ".join(str(error).split())
        signature = (
            self.current_objective_id,
            type(error).__name__,
            normalized,
            getattr(self._campaign_plan.recovery_stop, "destination", None)
            if self._campaign_plan is not None and self._campaign_plan.recovery_stop is not None
            else None,
        )
        if signature == self._recovery_failure_signature:
            self._recovery_failure_repeats += 1
        else:
            self._recovery_failure_signature = signature
            self._recovery_failure_repeats = 1
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_FAILURE: "
                f"objective={self.current_objective_id!r} exception_type={type(error).__name__!r} "
                f"exception={normalized!r} repeated={self._recovery_failure_repeats!r} "
                f"limit={_RECOVERY_FAILURE_ESCALATION_LIMIT!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_FAILURE:",
        )
        if self._recovery_failure_repeats < _RECOVERY_FAILURE_ESCALATION_LIMIT:
            return False
        self._tactical_loop = None
        self._execution_phase = "BLOCKED"
        self._recovery_status = "ESCALATED"
        self._accepted_recovery_objective_id = None
        self.status = CampaignControllerStatus.BLOCKED
        self._refresh_required = False
        self.transition_reason = (
            "recovery failed repeatedly with the same signature; autonomous retries paused: "
            f"{type(error).__name__}: {normalized}"
        )
        context.campaign_status = recovery_status(None, "Recovery failed repeatedly; autonomous retry paused")
        print_recovery_failure(
            method="campaign recovery",
            error=f"{type(error).__name__}: {normalized}",
            repeated=self._recovery_failure_repeats,
            escalated=True,
        )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_ESCALATED: "
                f"objective={self.current_objective_id!r} repeated={self._recovery_failure_repeats!r} "
                f"signature={signature!r} reason={self.transition_reason!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_ESCALATED:",
        )
        return True

    def _observe_mounted_stall(self, result) -> None:
        """Report a mounted capability that makes no visible progress for a long time.

        The idle watchdog only fires with nothing mounted.  A mounted capability
        can also wait forever, for example on a goal whose search failed, and
        until now that was completely silent.  This is a log record only; it
        changes no behavior.
        """

        watchdog = getattr(self, "_stall_watchdog", None)
        if watchdog is None:
            watchdog = self._stall_watchdog = _ProgressWatchdog(
                first_report_after=_STALL_FIRST_REPORT_AFTER_SECONDS,
                repeat_every=_STALL_REPEAT_EVERY_SECONDS,
            )
            self._player_sample = None
            self._player_sample_age = _PLAYER_SAMPLE_EVERY
        mounted = self._tactical_loop is not None
        if mounted:
            self._player_sample_age += 1
            if self._player_sample_age >= _PLAYER_SAMPLE_EVERY:
                self._player_sample_age = 0
                self._player_sample = _sample_player_location()
        stalled_seconds = watchdog.observe(
            (
                self._execution_phase,
                result.status.value,
                self.transition_reason,
                self.current_objective_id,
                repr(self.current_tactical_goal),
                self._player_sample,
            ),
            active=mounted,
        )
        if stalled_seconds is None:
            return
        print_always(
            lambda: (
                "CAMPAIGN_CONTROLLER_STALLED: a capability is mounted but nothing has changed "
                f"for {stalled_seconds:.0f} seconds phase={self._execution_phase!r} "
                f"status={result.status.value!r} objective={self.current_objective_id!r} "
                f"goal={self.current_tactical_goal!r} player={self._player_sample!r} "
                f"reason={self.transition_reason!r}"
            ),
        )

    def step(self) -> CampaignControllerState:
        """Advance one mounted capability and return its current status."""

        diagnostic_print(
            lambda: (
                "CAMPAIGN_CONTROLLER_STEP_ENTRY: "
                f"controller_id={id(self)!r} phase={self._execution_phase!r} "
                f"status={self.status.value!r} tactical_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r}"
            ),
            trace=True,
        )
        result = self._step_impl()
        watchdog = getattr(self, "_idle_watchdog", None)
        if watchdog is None:
            watchdog = self._idle_watchdog = _ControllerIdleWatchdog()
        idle_frames = watchdog.observe(
            (
                self._execution_phase,
                result.status.value,
                self.transition_reason,
                self.current_objective_id,
            ),
            idle=self._tactical_loop is None,
        )
        self._observe_mounted_stall(result)
        if idle_frames is not None:
            print_always(
                lambda: (
                    "CAMPAIGN_CONTROLLER_IDLE: no capability mounted and state unchanged "
                    f"for {idle_frames} controller steps phase={self._execution_phase!r} "
                    f"status={result.status.value!r} objective={self.current_objective_id!r} "
                    f"refresh_required={self._refresh_required!r} "
                    f"refresh_reason={self._refresh_required_reason!r} "
                    f"reason={self.transition_reason!r}"
                ),
            )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_CONTROLLER_STEP_EXIT: "
                f"controller_id={id(self)!r} phase={self._execution_phase!r} "
                f"status={result.status.value!r} tactical_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r}"
            ),
            trace=True,
        )
        return result

    def _step_impl(self) -> CampaignControllerState:
        """Perform one controller step across campaign or recovery phases."""

        if self._execution_phase == "BLOCKED" or self.status in {
            CampaignControllerStatus.COMPLETE,
            CampaignControllerStatus.FAILED,
            CampaignControllerStatus.RUN_LOST,
        }:
            return self.state
        battle_transition_active, transition_signature = _campaign_battle_transition_active()
        if battle_transition_active:
            if transition_signature != self._battle_transition_signature:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_BATTLE_TRANSITION_WAIT: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"phase={self._execution_phase!r} status={self.status.value!r} "
                        f"objective={self.current_objective_id!r} reason={transition_signature!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_BATTLE_TRANSITION_WAIT",
                )
                self._battle_transition_signature = transition_signature
            # Keep refresh/recheck requests pending. BattleListener owns the
            # ROM transition and will return control at the first stable field
            # boundary; only then may campaign planning inspect party, map, or
            # route state again. In particular, do not advance recovery or a
            # navigation generator while the battle transition tasks are live.
            return self.state
        self._battle_transition_signature = None
        if self._encounter_stage_retry_waiting():
            return self.state
        if self._execution_phase == "PREPARATION" and self._wild_battle_recovery_pending:
            # Preparation owns a nested encounter loop. A low-health wild
            # battle can return control while that loop is still mounted, so
            # let its next step become the recovery handoff instead of
            # issuing another grass search first.
            old_loop_id = id(self._tactical_loop) if self._tactical_loop is not None else None
            self._tactical_loop = None
            self._tactical_loop_objective_id = None
            self._execution_phase = "CAMPAIGN"
            self._refresh_required = True
            self._refresh_required_reason = "wild_battle_resource_boundary"
            self.status = CampaignControllerStatus.READY
            self.transition_reason = "low-health wild battle returned; reevaluating recovery"
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PREPARATION_RECOVERY_HANDOFF: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"old_loop_id={old_loop_id!r} action='refresh_for_recovery'"
                ),
                trace=True,
                prefix="CAMPAIGN_PREPARATION_RECOVERY_HANDOFF",
            )
        if self._execution_phase == "PREPARATION":
            try:
                if self._tactical_loop is None:
                    raise RuntimeError("trainer preparation has no executable capability")
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_PREPARATION_GENERATOR: event=advance "
                        f"controller_id={id(self)!r} loop_id={id(self._tactical_loop)!r} "
                        f"frame={getattr(context, 'frame', None)!r}"
                    ),
                    trace=True,
                )
                next(self._tactical_loop)
            except StopIteration:
                completed_loop_id = id(self._tactical_loop) if self._tactical_loop is not None else None
                self._tactical_loop = None
                self._tactical_loop_objective_id = None
                self._execution_phase = "CAMPAIGN"
                self._refresh_required = True
                self._refresh_required_reason = "trainer_preparation_completed"
                self.status = CampaignControllerStatus.READY
                scheduler = getattr(self._readiness_provider, "__self__", None)
                invalidate = getattr(scheduler, "invalidate", None)
                if callable(invalidate):
                    invalidate("trainer_preparation_completed")
                self._publish_campaign_status("Trainer preparation complete; re-evaluating readiness")
                self.transition_reason = "trainer preparation complete; reevaluating campaign readiness"
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_PREPARATION_COMPLETE: "
                        f"controller_id={id(self)!r} loop_id={completed_loop_id!r} "
                        f"objective={self.current_objective_id!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_PREPARATION_COMPLETE",
                )
            except Exception as error:
                self._tactical_loop = None
                self._tactical_loop_objective_id = None
                # A preparation setup failure (for example a stale route or
                # an unavailable ROM observation) says nothing about the
                # trainer's battle safety. Do not turn repeated setup errors
                # into artificial level escalation on the next refresh.
                self._trainer_preparation_checkpoint_key = None
                self._trainer_preparation_last_target = None
                self._execution_phase = "CAMPAIGN"
                self._refresh_required = True
                self._refresh_required_reason = "trainer_preparation_exception"
                self.status = CampaignControllerStatus.UNKNOWN
                context.campaign_status = CampaignStatus(
                    self.current_objective_id,
                    _campaign_target_for_execution(
                        self.last_selection.objective if self.last_selection is not None else None,
                        self.current_tactical_goal,
                    ),
                    "Trainer preparation failed; retrying autonomously",
                    self.current_tactical_goal,
                )
                self.transition_reason = f"trainer preparation failed; retrying: {error}"
                _report_controller_exception(
                    "trainer_preparation",
                    error,
                    objective=self.current_objective_id,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_PREPARATION_EXCEPTION: "
                        f"objective={self.current_objective_id!r} error_type={type(error).__name__!r} "
                        f"error={error!r} traceback={traceback.format_exc()!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_PREPARATION_EXCEPTION",
                )
            return self.state
        if self._execution_phase == "RECOVERY":
            try:
                if self._tactical_loop is None:
                    raise RuntimeError("recovery has no executable capability")
                diagnostic_print(
                    lambda: f"CAMPAIGN_RECOVERY_GENERATOR: event=advance controller_id={id(self)!r} loop_id={id(self._tactical_loop)!r} frame={getattr(context, 'frame', None)!r}",
                    trace=True,
                )
                next(self._tactical_loop)
            except StopIteration:
                completed_recovery_loop_id = id(self._tactical_loop) if self._tactical_loop is not None else None
                completed_stop = self._campaign_plan.recovery_stop if self._campaign_plan is not None else None
                continuation_route = getattr(completed_stop, "continuation_route", None)
                if continuation_route is not None:
                    # The recovery stop is temporary, but its parent route is
                    # still authoritative until a fresh observation either
                    # adopts its first action or proves that the world changed.
                    context.campaign_continuation_route = (
                        self.current_objective_id,
                        continuation_route,
                    )
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_RECOVERY_HANDOFF: "
                            f"objective={self.current_objective_id!r} "
                            f"continuation_destination={getattr(continuation_route, 'destination', None)!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_RECOVERY_HANDOFF",
                    )
                self._tactical_loop = None
                self._refresh_required = True
                self._refresh_required_reason = "recovery_completed"
                self._execution_phase = "CAMPAIGN"
                self._recovery_status = "COMPLETED"
                self._wild_battle_recovery_pending = False
                self._accepted_recovery_objective_id = None
                self._recovery_keyerror_retries = 0
                self._recovery_failure_signature = None
                self._recovery_failure_repeats = 0
                if _restock_transaction_is_active():
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_RESTOCK_TRANSACTION: "
                            f"phase='campaign_resume' parent_objective="
                            f"{getattr(context, 'campaign_restock_parent_objective_id', None)!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_RESTOCK_TRANSACTION",
                    )
                    context.campaign_restock_phase = "campaign_resume"
                    context.campaign_restock_owner = None
                    context.campaign_restock_parent_objective_id = None
                if self._campaign_plan is not None:
                    self._campaign_plan = self._campaign_plan.complete_active_stop()
                scheduler = getattr(self._readiness_provider, "__self__", None)
                invalidate = getattr(scheduler, "invalidate", None)
                if callable(invalidate):
                    invalidate("recovery_completed")
                self.status = CampaignControllerStatus.READY
                self._publish_campaign_status("Recovery complete; resuming campaign")
                self.transition_reason = "recovery verified; reevaluating campaign readiness"
                diagnostic_print(
                    lambda: f"CAMPAIGN_RECOVERY_COMPLETE: frame={getattr(context, 'frame', None)!r} controller_id={id(self)!r} recovery_loop_id={completed_recovery_loop_id!r} objective={self.current_objective_id!r} invalidation='recovery_completed' resumed_phase={self._execution_phase}",
                    trace=True,
                )
            except Exception as error:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_EXCEPTION: "
                        f"frame={getattr(context, 'frame', None)!r} "
                        f"emulator_frame={context.emulator.get_frame_count()!r} "
                        f"controller_id={id(self)!r} "
                        f"recovery_loop_id={id(self._tactical_loop) if self._tactical_loop else None!r} "
                        f"exception_type={type(error).__name__!r} exception={str(error)!r} "
                        f"exception_repr={error!r} phase={self._execution_phase!r} "
                        f"status={self.status.value!r} recovery_status={self._recovery_status!r} "
                        f"objective={self.current_objective_id!r} transition_reason={self.transition_reason!r}"
                        f" traceback={traceback.format_exc()!r}"
                    ),
                    trace=True,
                )
                if self._record_recovery_failure(error):
                    return self.state
                if isinstance(error, KeyError) and self._retry_recovery_after_keyerror(error):
                    return self.state
                self._tactical_loop = None
                if getattr(error, "run_lost", False):
                    self._terminal(
                        CampaignExecutionResult(
                            self.last_selection.objective if self.last_selection else None,
                            CampaignExecutionStatus.FAILED,
                            f"run lost during recovery: {error}",
                            self.last_execution.execution_id if self.last_execution else None,
                        ),
                        CampaignControllerStatus.RUN_LOST,
                    )
                else:
                    self._execution_phase = "CAMPAIGN"
                    self._recovery_status = "RETRYABLE_FAILED"
                    self._refresh_required = True
                    self._refresh_required_reason = "recovery_exception"
                    self.status = CampaignControllerStatus.UNKNOWN
                    context.campaign_status = recovery_status(None, "Recovery failed; retrying autonomously")
                    self.transition_reason = f"recovery failed; retrying: {error}"
            return self.state
        try:
            if self._refresh_required or self._readiness_recheck_pending:
                refresh_reason = self._refresh_required_reason
                current = self.refresh()
                # A refresh may deliberately leave no tactical loop mounted
                # while readiness is deferred (for example, while a battle
                # listener owns the emulator).  That is still a handled
                # boundary; at most the one compatibility retry represented
                # by _readiness_recheck_pending may follow it. Further
                # refreshes must come from an explicit event rather than from
                # the absence of a loop alone.
                if current.status in {
                    CampaignControllerStatus.READY,
                    CampaignControllerStatus.UNKNOWN,
                    CampaignControllerStatus.BLOCKED,
                    CampaignControllerStatus.COMPLETE,
                    CampaignControllerStatus.FAILED,
                }:
                    # A readiness-deferred handoff can intentionally leave no
                    # loop mounted after invalidating the previous objective.
                    # Keep refreshing until the new objective can be mounted;
                    # otherwise the controller would settle with a READY
                    # state and no executable loop after one transient ROM
                    # transition.
                    deferred_handoff_without_loop = (
                        current.status is CampaignControllerStatus.READY
                        and self._tactical_loop is None
                        and self.transition_reason.startswith("readiness deferred:")
                        and self._deferred_handoff_requires_refresh
                    )
                    selection_retry_pending = (
                        self.last_selection is not None
                        and self.last_selection.status in {ObjectiveStatus.UNKNOWN, ObjectiveStatus.BLOCKED}
                    )
                    completion_projection_pending = self._refresh_required_reason == "completion_projection_pending"
                    navigation_transition_pending = self._navigation_transition_owned()
                    encounter_stage_refresh_pending = self._refresh_required_reason in {
                        "encounter_stage_recovery_pending",
                        "encounter_stage_blocked_retry",
                    }
                    if (
                        not self._advisory_route_polling
                        and
                        not deferred_handoff_without_loop
                        and not selection_retry_pending
                        and not completion_projection_pending
                        and not navigation_transition_pending
                        and not encounter_stage_refresh_pending
                    ):
                        self._refresh_required = False
                        self._refresh_required_reason = None
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_REFRESH_BOUNDARY: "
                        f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                        f"reason={refresh_reason!r} status={current.status.value!r} "
                        f"loop_id={id(self._tactical_loop) if self._tactical_loop else None!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_REFRESH_BOUNDARY",
                )
            else:
                current = self.state
        except Exception as error:
            # Readiness and planner inputs are assembled from several ROM
            # structures that do not become visible on the same frame after a
            # battle. A transient refresh failure must not unwind the mode
            # generator and silently return the bot to manual mode; discard
            # only the frame-local tactical loop and retry from fresh facts.
            failed_loop_id = id(self._tactical_loop) if self._tactical_loop is not None else None
            self._tactical_loop = None
            self._tactical_loop_objective_id = None
            self._refresh_required = True
            self._refresh_required_reason = "refresh_exception"
            self.status = CampaignControllerStatus.UNKNOWN
            self.transition_reason = f"campaign refresh failed; will re-evaluate: {type(error).__name__}: {error!r}"
            _report_controller_exception("refresh", error, objective=self.current_objective_id)
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_REFRESH_EXCEPTION: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"failed_loop_id={failed_loop_id!r} status={self.status.value!r} "
                    f"reason={self.transition_reason!r} traceback={traceback.format_exc()!r}"
                ),
                trace=True,
            )
            return self.state
        # Preparation is a separate ownership phase. Do not advance its
        # generator in the same controller call that mounted it; this keeps a
        # zero-frame test/capability boundary from being misclassified as a
        # completed campaign objective by the generic campaign StopIteration
        # handler below.
        if self._execution_phase == "PREPARATION":
            return current
        if current.status is not CampaignControllerStatus.READY or self._tactical_loop is None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_STEP_RETURN: "
                    f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                    f"branch={'loop_absent' if self._tactical_loop is None else 'not_ready'} "
                    f"status={current.status.value!r} reason={current.reason!r} input_emitted=False"
                ),
                trace=True,
            )
            return current
        try:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_TACTICAL_STEP_HANDOFF: "
                    f"controller_id={id(self)!r} loop_id={id(self._tactical_loop)!r} "
                    f"mounted_objective={self._tactical_loop_objective_id!r} "
                    f"planner_objective={getattr(getattr(self.last_selection, 'objective', None), 'objective_id', None)!r} "
                    f"tactical_goal={self.current_tactical_goal!r}"
                ),
                trace=True,
            )
            next(self._tactical_loop)
        except StopIteration:
            if self.last_execution is not None and self.last_execution.capability is not None:
                completed_objective = False
                completion = None
                completed_objective_id = self.current_objective_id
                completion_latched = (
                    getattr(context, "campaign_capability_completion_latch", None)
                    == completed_objective_id
                )
                if self.last_execution.objective is not None:
                    try:
                        completion = self.last_execution.objective.completion.evaluate(self._state_provider())
                        completed_objective = (
                            completion.status is FactStatus.KNOWN and completion.value is True
                        )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        # A capability can return while the ROM is still
                        # between battle/script snapshots. Preserve the
                        # existing remount boundary for an incomplete or
                        # unavailable completion fact.
                        completion = None
                if completion_latched:
                    completed_objective = True
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_CAPABILITY_COMPLETION_LATCH: "
                            f"objective={completed_objective_id!r} "
                            "reason='live_capability_observed_authoritative_fact'"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_CAPABILITY_COMPLETION_LATCH",
                    )
                if completed_objective:
                    completed_loop_id = id(self._tactical_loop)
                    self._tactical_loop = None
                    self._tactical_loop_objective_id = None
                    self._completed_capability_objective = self.last_execution.objective
                    self._refresh_required = True
                    self._refresh_required_reason = "capability_completed"
                    self.status = CampaignControllerStatus.READY
                    self.transition_reason = "capability completed; reevaluating campaign frontier"
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_CAPABILITY_COMPLETE: "
                            f"controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} "
                            f"loop_id={completed_loop_id!r} objective={completed_objective_id!r} "
                            f"completion_status={getattr(completion, 'status', None)!r} "
                            f"completion_value={getattr(completion, 'value', None)!r}"
                        ),
                        trace=True,
                        prefix="CAMPAIGN_CAPABILITY_COMPLETE",
                    )
                    return self.state
                # A capability may yield control at a ROM boundary while its
                # authoritative completion fact is still false. Re-observe
                # and remount it on the next frame instead of declaring a
                # tactical failure.
                self._tactical_loop = None
                self._refresh_required = True
                self._refresh_required_reason = "capability_boundary"
                self.transition_reason = "capability boundary; awaiting authoritative completion"
                diagnostic_print(
                    lambda: f"CAMPAIGN_STEP_RETURN: controller_id={id(self)!r} frame={getattr(context, 'frame', None)!r} branch='capability_boundary' status={self.status.value!r} reason={self.transition_reason!r} input_emitted=unknown",
                    trace=True,
                )
                return self.state
            self._terminal(
                CampaignExecutionResult(
                    self.last_selection.objective if self.last_selection else None,
                    CampaignExecutionStatus.FAILED,
                    "tactical execution stopped before objective completion",
                    self.last_execution.execution_id if self.last_execution else None,
                ),
                CampaignControllerStatus.FAILED,
            )
        except Exception as error:
            if self._execution_phase == "RECOVERY":
                if self._record_recovery_failure(error):
                    return self.state
                if isinstance(error, KeyError) and self._retry_recovery_after_keyerror(error):
                    return self.state
                self._tactical_loop = None
                if getattr(error, "run_lost", False):
                    self._terminal(
                        CampaignExecutionResult(
                            self.last_selection.objective if self.last_selection else None,
                            CampaignExecutionStatus.FAILED,
                            f"run lost during recovery: {error}",
                            self.last_execution.execution_id if self.last_execution else None,
                        ),
                        CampaignControllerStatus.RUN_LOST,
                    )
                else:
                    self._execution_phase = "CAMPAIGN"
                    self._recovery_status = "RETRYABLE_FAILED"
                    self._refresh_required = True
                    self._refresh_required_reason = "recovery_exception"
                    self.status = CampaignControllerStatus.UNKNOWN
                    context.campaign_status = recovery_status(None, "Recovery failed; retrying autonomously")
                    self.transition_reason = f"recovery failed; retrying: {error}"
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_EXCEPTION: "
                        f"frame={getattr(context, 'frame', None)!r} "
                        f"exception_type={type(error).__name__!r} exception={error!r} "
                        f"traceback={traceback.format_exc()!r} phase={self._execution_phase!r}"
                    ),
                    trace=True,
                )
                return self.state
            if isinstance(error, CampaignTrainerLeadFailure):
                self._tactical_loop = None
                self._tactical_loop_objective_id = None
                self._execution_phase = "BLOCKED"
                self.status = CampaignControllerStatus.BLOCKED
                self._refresh_required = False
                self.transition_reason = str(error)
                context.campaign_status = CampaignStatus(
                    self.current_objective_id,
                    _campaign_target_for_execution(
                        self.last_selection.objective if self.last_selection is not None else None,
                        self.current_tactical_goal,
                    ),
                    "Trainer lead rotation failed; manual review required",
                    self.current_tactical_goal,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_LEAD_FAILURE: "
                        f"controller_id={id(self)!r} objective={self.current_objective_id!r} "
                        f"trainer={error.trainer_id!r} phase={error.phase!r} reason={error.reason!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_LEAD_FAILURE",
                )
                return self.state
            if getattr(error, "navigation_loop", False):
                # A repeated transition cycle is a real ownership boundary,
                # not a transient tactical exception. Stop the mounted
                # campaign capability so it cannot remount the same route
                # and replay the loop forever.
                self._tactical_loop = None
                self._execution_phase = "BLOCKED"
                self.status = CampaignControllerStatus.BLOCKED
                self._refresh_required = False
                self.transition_reason = f"navigation loop detected; autonomous route stopped: {error}"
                context.campaign_status = CampaignStatus(
                    self.current_objective_id,
                    _campaign_target_for_execution(
                        self.last_selection.objective if self.last_selection is not None else None,
                        self.current_tactical_goal,
                    ),
                    "Navigation loop detected; manual review required",
                    self.current_tactical_goal,
                )
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_NAVIGATION_LOOP_STOPPED: "
                        f"controller_id={id(self)!r} objective={self.current_objective_id!r} "
                        f"exception={error!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_NAVIGATION_LOOP_STOPPED",
                )
                return self.state
            # A failed generator cannot be resumed safely. Clear it and leave
            # the controller recoverable so a later Campaign-mode entry can
            # select the current objective and rebuild fresh execution state.
            self._tactical_loop = None
            self._refresh_required = True
            self._refresh_required_reason = "tactical_exception"
            self.status = CampaignControllerStatus.UNKNOWN
            self.transition_reason = f"tactical execution failed; will re-evaluate: {type(error).__name__}: {error!r}"
            diagnostic_print(
                lambda: (
                    f"CAMPAIGN_STEP_RETURN: controller_id={id(self)!r} "
                    f"frame={getattr(context, 'frame', None)!r} branch='tactical_exception' "
                    f"status={self.status.value!r} reason={self.transition_reason!r} "
                    f"input_emitted=False traceback={traceback.format_exc()!r}"
                ),
                trace=True,
            )
        return self.state

    def run(self) -> Generator:
        """Yield frame boundaries until the campaign reaches a terminal status."""

        while True:
            state = self.step()
            if state.status is CampaignControllerStatus.UNKNOWN:
                # UNKNOWN is a transient observation boundary.  In
                # particular, UI capabilities can be mounted while the ROM
                # has not yet exposed overworld readiness; do not terminate
                # the controller before the mounted capability gets a chance
                # to execute on the next frame.
                yield
                continue
            if state.status is not CampaignControllerStatus.READY:
                return
            yield


def runtime_campaign_state() -> CampaignState:
    """Build a facade from the already-materialized runtime state."""
    from modules.context import context
    from modules.map_data import get_map_enum
    from .snapshots import get_nuzlocke_snapshot

    runtime = context.nuzlocke_runtime
    # ``NuzlockeRuntime.update`` already materializes the authoritative
    # snapshot for this application frame.  Reusing it avoids a second set of
    # save-block reads between observation and campaign selection, which can
    # otherwise stall the frame loop on loaded mGBA states.
    snapshot = getattr(runtime, "latest_snapshot", None) if runtime is not None else None
    if snapshot is None:
        snapshot = get_nuzlocke_snapshot()
    canonical_area = None
    if snapshot.player_available and snapshot.player.map_group is not None and snapshot.player.map_number is not None:
        try:
            canonical_area = get_map_enum((snapshot.player.map_group, snapshot.player.map_number)).name
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
    history_ready = runtime is None or runtime.campaign_history_ready
    return CampaignState.from_runtime_state(
        snapshot=snapshot,
        observed_projection=runtime.observed_projection if history_ready and runtime is not None else None,
        rules_projection=runtime.rules_projection if history_ready and runtime is not None else None,
        canonical_area=canonical_area,
        pokeball_policy=(runtime.rule_config.pokeball_policy if runtime is not None else None),
        require_set_battle_style=(
            runtime is not None and runtime.rule_config.is_enabled(CampaignRuleId.SET_BATTLE_STYLE)
        ),
    )


def runtime_campaign_boundary(objective_id: str) -> bool:
    """Execute a Nuzbot-owned campaign boundary through the live runtime."""
    if objective_id != "start_nuzlocke":
        return False
    from modules.context import context

    runtime = context.nuzlocke_runtime
    if runtime is None:
        return False
    runtime.mark_nuzlocke_started()
    return True
