"""State-specific affordances, goal evaluation, and action execution.

The control loop is intentionally independent from ``main_loop`` and bot
modes.  A future mode or strategic planner can supply observations and goals,
then mount ``AgentControlLoop.run()`` on the existing controller stack when it
is ready to do so.
"""

from dataclasses import dataclass, replace
from enum import Enum, auto
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from threading import Event
from time import monotonic, perf_counter_ns
import os
import time as _time
import traceback
from typing import Callable, Generator

from modules.context import context
from modules.console import (
    diagnostic_print,
    print_always,
    print_navigation_cancellation,
    print_navigation_invalidation,
    print_recovery_handoff,
    profile_print,
)
from modules.goals import (
    ActivateTrigger,
    EngageTrainer,
    Goal,
    NavigationGoal,
    ReachInteractionPosition,
    ReachLocation,
    ReachWarp,
    SemanticTarget,
    TrainerEngagementMode,
    TrainerEngagementPlan,
    TrainerMode,
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
from modules.map import get_event_flag, get_map_data, get_map_objects, get_runtime_object_table
from modules.map_data import MapRSE, get_map_enum
from modules.game import get_event_flag_name
from modules.memory import GameState, get_game_state, get_save_block
from modules.navigation import (
    GoalAwareNavigator,
    NavigationAction,
    NavigationActionType,
    NavigationError,
    NavigationSearchCancelled,
    NavigationPlan,
    RoutePlan,
    RouteCheckpoint,
    record_runtime_transition_observation,
    NavigationWorld,
    canonical_location,
    WorldNavigationError,
    goal_target_map,
    plan_with_world_navigation,
    plan_observed_warp_locally,
    navigation_diagnostics,
    prewarm_navigation_tiles,
    resolve_trainer_engagement_plan,
)
from modules.overworld import (
    MovementState,
    OverworldObservation,
    perceive_overworld,
    prewarm_static_map_observation,
)
from modules.player import get_player_avatar
from modules.profiler import count, invalidation, now, profiled, timing, format_snapshot
from modules.tasks import get_global_script_context, is_field_message_waiting_for_input
from modules.state_cache import state_cache
from modules.stutter_trace import suppress_background_instrumentation
from modules.nuzlocke.readiness_diagnostics import (
    build_progression_readiness_diagnostic,
    evaluate_progression_readiness,
)
import json

UNIVERSAL_REOBSERVE_FRAMES = 90
_last_universal_observation_frame: int | None = None
_battle_end_generation = 0
_last_battle_end_emulator_frame: int | None = None
_recent_battle_return_window = 120


_NAVIGATION_PLAN_UNSET = object()
_NAVIGATION_PLAN_PENDING = object()
_DEFAULT_NAVIGATION_OWNER = object()


def _gate_observation_frame() -> int | None:
    """Return the emulator frame at a live gate evidence boundary."""

    emulator = getattr(context, "emulator", None)
    get_frame_count = getattr(emulator, "get_frame_count", None)
    if callable(get_frame_count):
        try:
            frame = get_frame_count()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            frame = None
        if isinstance(frame, int) and not isinstance(frame, bool):
            return frame
    frame = getattr(context, "frame", None)
    return frame if isinstance(frame, int) and not isinstance(frame, bool) else None


def _gate_observer_call(method_name: str, *args, **kwargs) -> None:
    """Call optional live evidence instrumentation without affecting control."""

    observer = getattr(context, "gate_live_observer", None)
    method = getattr(observer, method_name, None)
    if not callable(method):
        return
    if getattr(context, "debug_trace", False):
        diagnostic_print(
            lambda: f"GATE_LIVE_HOOK: method={method_name!r} "
            f"navigation={getattr(args[0], 'action_type', None)!r} "
            f"source={getattr(args[0], 'source', None)!r} "
            f"destination={getattr(args[0], 'destination', None)!r} "
            f"outcome={args[1] if method_name == 'record_navigation_outcome' and len(args) > 1 else None!r}",
            trace=True,
        )
    try:
        method(*args, **kwargs)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        # Evidence collection is diagnostic. A malformed observer must never
        # change the action selected by the controller.
        diagnostics = getattr(observer, "_diagnostics", None)
        if isinstance(diagnostics, set):
            diagnostics.add(f"observer hook {method_name!r} failed: {type(error).__name__}: {error}")
        return


def _navigation_world_summary(world, start, goal) -> str:
    """Summarize the facts that decide whether a trainer-safe search can succeed.

    A search that is cancelled by its time budget and a search that has no
    safe route look identical in the pathfinding panel.  This compact line is
    logged when a background search fails so the two can be told apart.
    """

    try:
        constraints = getattr(goal, "constraints", None)
        mode = getattr(getattr(constraints, "trainer_mode", None), "name", None)
        hazards = []
        start_in = []
        total = 0
        for trigger in getattr(world, "triggers", ()) or ():
            locations = frozenset(getattr(trigger, "hazard_locations", ()) or ())
            if not locations:
                continue
            total += len(locations)
            label = getattr(trigger, "affordance_id", None) or getattr(trigger, "trigger_id", None)
            hazards.append(f"{label}:{len(locations)}")
            if start in locations:
                start_in.append(str(label))
        tiles = len(getattr(world, "tiles", ()) or ())
        return (
            f"trainer_mode={mode} forced_exposure_allowed="
            f"{getattr(constraints, 'allow_forced_trainer_exposure', None)} tiles={tiles} "
            f"hazard_tiles={total} hazards={sorted(hazards)!r} start={start!r} "
            f"start_inside_hazard_of={sorted(start_in)!r}"
        )
    except Exception as error:  # diagnostics must never break navigation
        return f"world_summary_unavailable error={type(error).__name__}: {error}"


def _dump_slow_navigation_search(world, start, goal, elapsed_ms: float) -> None:
    """Optionally persist a slow failed search for offline profiling.

    Set ``POKEBOT_DUMP_SLOW_SEARCH`` to a directory to capture the exact
    immutable world, start and goal of a search that ran for more than three
    seconds without a result.  Disabled by default.
    """

    directory = os.environ.get("POKEBOT_DUMP_SLOW_SEARCH")
    if not directory or elapsed_ms < 3000.0:
        return
    try:
        import pickle

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"slow_search_{int(_time.time())}_{int(elapsed_ms)}ms.pkl")
        with open(path, "wb") as handle:
            pickle.dump({"world": world, "start": start, "goal": goal, "elapsed_ms": elapsed_ms}, handle)
        print_always(
            lambda: f"NAVIGATION_SEARCH_DUMP: path={path!r}",
        )
    except Exception as error:  # diagnostics must never break navigation
        print_always(
            lambda: f"NAVIGATION_SEARCH_DUMP_FAILED: {type(error).__name__}: {error}",
        )


class _SingleFlightNavigationPlanner:
    """Run one tactical world search away from the emulator frame thread."""

    DEFAULT_SEARCH_TIMEOUT_SECONDS = 10.0

    def __init__(self, *, search_timeout_seconds: float | None = None):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pokebot-navigation")
        self._search_timeout_seconds = (
            self.DEFAULT_SEARCH_TIMEOUT_SECONDS
            if search_timeout_seconds is None
            else float(search_timeout_seconds)
        )
        if self._search_timeout_seconds <= 0:
            raise ValueError("search_timeout_seconds must be positive")
        self._key = None
        self._failure_key = None
        self._future = None
        self._owner = None
        self._owner_label = None
        self._cancel_event: Event | None = None
        self._started_at: float | None = None
        # A failed search is often observed on many consecutive controller
        # frames. Remember it for the exact immutable request so an impossible
        # route does not consume the worker again every frame. The request
        # key includes the start, goal, and stable transition topology at the
        # call site. Transient runtime occupancy is deliberately excluded;
        # the selected action is validated against the current observation
        # before an input reaches the emulator.
        self._failed_results: dict[object, tuple] = {}
        self._last_discard_reason: str | None = None
        self._last_event: tuple | None = None

    def _event(self, event: str, **details) -> None:
        """Retain one concise lifecycle event for the frame-thread logger."""

        self._last_event = (event, tuple(sorted(details.items(), key=lambda item: item[0])))

    def _future_result(self, future, key, failure_key=None):
        try:
            result = future.result()
        except (NavigationError, WorldNavigationError) as error:
            result = (None, None, error)
        except Exception as error:
            result = (None, None, error)
        failure_key = key if failure_key is None else failure_key
        # Cooperative cancellation is a scheduling outcome, not evidence that
        # this route is impossible.  Caching it under the immutable request
        # key turns one superseded worker into a permanent failure: every new
        # controller sees the cached NavigationSearchCancelled and tears down
        # before it can submit a fresh search.
        if result[0] is None and not isinstance(result[2], NavigationSearchCancelled):
            self._failed_results[failure_key] = result
            if len(self._failed_results) > 64:
                self._failed_results.pop(next(iter(self._failed_results)))
        else:
            self._failed_results.pop(failure_key, None)
        return result

    def poll(self, key, operation, *, failure_key=None, owner=None, owner_label=None):
        # ThreadPoolExecutor cannot interrupt a search that is already
        # running. Replacing that future for every new player tile therefore
        # queues searches behind one another while the control loop keeps
        # changing its start key. Keep one in-flight search, discard its
        # result when the requested key has moved on, and submit only the
        # newest request once the worker is actually free.
        self._last_discard_reason = None
        self._last_event = None
        owner = _DEFAULT_NAVIGATION_OWNER if owner is None else owner
        owner_label = owner_label or ("default" if owner is _DEFAULT_NAVIGATION_OWNER else repr(owner))
        failure_key = key if failure_key is None else failure_key
        cached_failure = self._failed_results.get(failure_key)
        if cached_failure is not None:
            self._event("failure_cached", owner=owner_label, key=key)
            return cached_failure

        # The process-wide worker is shared by campaign loops, recovery loops,
        # and tests.  A loop boundary therefore has to be an ownership
        # boundary as well; otherwise a finished search from an old loop can
        # be handed to a newly mounted controller with an equivalent goal.
        if self._future is not None and not self._future.done() and self._owner != owner:
            self._cancel_active(
                reason="navigation request owner changed",
                replacement_owner=owner_label,
            )

        if self._future is not None and self._future.done():
            future = self._future
            completed_key = self._key
            completed_failure_key = self._failure_key
            completed_owner = self._owner
            completed_owner_label = self._owner_label
            self._future = None
            self._key = None
            self._failure_key = None
            self._owner = None
            self._owner_label = None
            self._cancel_event = None
            self._started_at = None
            if completed_owner == owner and _navigation_request_keys_equivalent(completed_key, key):
                self._event("result_adopted", owner=owner_label, key=key)
                return self._future_result(future, completed_key, completed_failure_key)
            # The search was composed from an older start tile, goal, or
            # transition topology. Its route can be structurally valid while
            # still being unusable from the current observation, so never
            # expose it to action selection.
            self._last_discard_reason = (
                "request_key_changed "
                f"changed={_navigation_request_key_changes(completed_key, key)!r}"
            )
            if completed_owner != owner:
                self._last_discard_reason = (
                    "request_owner_changed "
                    f"owner={completed_owner_label!r} replacement={owner_label!r}"
                )
            self._event(
                "result_discarded",
                owner=completed_owner_label,
                replacement_owner=owner_label,
                changed=_navigation_request_key_changes(completed_key, key),
            )
            self._future_result(future, completed_key, completed_failure_key)

        # A goal or topology change invalidates the worker immediately. A
        # start-only change is ordinary frame progress and is allowed to
        # finish below; the exact key still prevents its result from being
        # exposed from the wrong origin.
        if self._future is not None and not self._future.done() and (
            self._key != key or self._owner != owner
        ):
            changed = _navigation_request_key_changes(self._key, key)
            # The exact start remains in the key so a completed route can
            # never be accepted from the wrong origin. While a search is
            # running, however, a start-only change is normal frame progress;
            # let the current worker finish and submit the newest exact-start
            # request only after that result is harvested. This preserves one
            # worker without repeatedly canceling the same search as the
            # avatar advances.
            if (
                (changed != ("start",) or self._owner != owner)
                and self._cancel_event is not None
                and not self._cancel_event.is_set()
            ):
                previous_key = self._key
                elapsed_ms = (
                    (monotonic() - self._started_at) * 1000.0
                    if self._started_at is not None
                    else None
                )
                self._cancel_event.set()
                self._last_discard_reason = (
                    (
                        "request_owner_changed "
                        f"owner={self._owner_label!r} replacement={owner_label!r}"
                    )
                    if self._owner != owner
                    else (
                        "request_superseded "
                        f"changed={_navigation_request_key_changes(previous_key, key)!r}"
                    )
                )
                self._event(
                    "request_cancelled",
                    owner=self._owner_label,
                    replacement_owner=owner_label,
                    changed=changed,
                )
                print_navigation_cancellation(
                    phase="requested",
                    reason=(
                        "navigation request owner changed"
                        if self._owner != owner
                        else "newer navigation request superseded the active request"
                    ),
                    request_key=previous_key,
                    replacement_key=key,
                    changed=_navigation_request_key_changes(previous_key, key),
                    duration_ms=elapsed_ms,
                    request_label=self._owner_label,
                    replacement_label=owner_label,
                    worker="agent_control.navigation",
                )

        if self._future is None:
            self._key = key
            self._failure_key = failure_key
            self._owner = owner
            self._owner_label = owner_label
            self._cancel_event = Event()
            self._started_at = monotonic()
            self._future = self._executor.submit(self._run, operation, self._cancel_event, key)
            self._event("search_started", owner=owner_label, key=key)
        if not self._future.done():
            return _NAVIGATION_PLAN_PENDING
        future = self._future
        completed_key = self._key
        completed_failure_key = self._failure_key
        self._future = None
        self._key = None
        self._failure_key = None
        self._owner = None
        self._owner_label = None
        self._cancel_event = None
        self._started_at = None
        self._event("result_ready", owner=owner_label, key=completed_key)
        return self._future_result(future, completed_key, completed_failure_key)

    def take_discard_reason(self) -> str | None:
        """Return and clear the most recent stale-result diagnostic."""

        reason = self._last_discard_reason
        self._last_discard_reason = None
        return reason

    def take_event(self) -> tuple | None:
        """Return and clear the most recent planner lifecycle event."""

        event = self._last_event
        self._last_event = None
        return event

    def _cancel_active(self, *, reason: str, replacement_owner: str | None = None) -> bool:
        """Request cancellation without replacing the executor future."""

        if self._future is None or self._future.done() or self._cancel_event is None:
            return False
        if self._cancel_event.is_set():
            return True
        self._cancel_event.set()
        elapsed_ms = (
            (monotonic() - self._started_at) * 1000.0
            if self._started_at is not None
            else None
        )
        self._last_discard_reason = (
            f"request_cancelled reason={reason!r} owner={self._owner_label!r}"
        )
        self._event(
            "request_cancelled",
            owner=self._owner_label,
            replacement_owner=replacement_owner,
            reason=reason,
        )
        print_navigation_cancellation(
            phase="requested",
            reason=reason,
            request_key=self._key,
            replacement_label=replacement_owner,
            duration_ms=elapsed_ms,
            request_label=self._owner_label,
            worker="agent_control.navigation",
        )
        return True

    def cancel(self, owner, *, reason="navigation owner disposed", owner_label=None) -> bool:
        """Cancel and detach a request owned by one control loop."""

        owner = _DEFAULT_NAVIGATION_OWNER if owner is None else owner
        if self._owner != owner:
            return False
        if self._future is None:
            return False
        if self._future.done():
            # No caller remains to adopt a completed result.  Consume it and
            # detach the logical request while leaving the executor alive for
            # the next owner.
            future = self._future
            try:
                future.result()
            except Exception:
                pass
            self._future = None
            self._key = None
            self._failure_key = None
            self._owner = None
            self._owner_label = None
            self._cancel_event = None
            self._started_at = None
            return True
        return self._cancel_active(
            reason=reason,
            replacement_owner=owner_label or "none",
        )

    def active_key(self):
        """Return the request currently being searched, if any."""

        if self._future is None or self._future.done():
            return None
        return self._key

    def _run(self, operation, cancel_event, key):
        # Profiling and rolling stutter traces belong to the frame thread. The
        # world and goal passed to this worker are immutable snapshots.
        started_at = monotonic()
        with suppress_background_instrumentation():
            deadline = monotonic() + self._search_timeout_seconds

            def cancel_check():
                if cancel_event.is_set():
                    raise NavigationSearchCancelled("navigation request superseded")
                if monotonic() >= deadline:
                    raise NavigationSearchCancelled(
                        f"navigation search exceeded {self._search_timeout_seconds:g} second budget"
                    )
                # Preserve the simple boolean callback contract used by
                # lower-level callers that poll for cancellation themselves.
                return False

            try:
                plan, route = operation(cancel_check)
            except NavigationSearchCancelled as error:
                print_navigation_cancellation(
                    phase="terminal",
                    reason=str(error),
                    request_key=key,
                    duration_ms=(monotonic() - started_at) * 1000.0,
                )
                return (None, None, error)
            except (NavigationError, WorldNavigationError) as error:
                return (None, None, error)
            except Exception as error:
                return (None, None, error)
        return (plan, route, None)


_background_navigation_planner = _SingleFlightNavigationPlanner()


def _navigation_request_key_changes(previous, current) -> tuple[str, ...]:
    """Describe the stable fields that changed between tactical requests."""

    if not isinstance(previous, tuple) or not isinstance(current, tuple):
        return ("request_key",)
    names = (
        ("start", "goal", "transition_topology")
        if len(previous) <= 3 and len(current) <= 3
        else ("start", "goal", "selected_transition", "navigation_policy")
    )
    changed = []
    for index, name in enumerate(names):
        if index >= len(previous) or index >= len(current) or previous[index] != current[index]:
            changed.append(name)
    return tuple(changed) or ("request_key",)


def _navigation_request_keys_equivalent(previous, current) -> bool:
    """Return whether two canonical requests can share a completed plan."""

    if previous == current:
        return True
    # Keep this boundary explicit so future diagnostic-only fields can be
    # added to a request without making a completed equivalent plan stale.
    if (
        isinstance(previous, tuple)
        and isinstance(current, tuple)
        and len(previous) >= 4
        and len(current) >= 4
    ):
        return tuple(previous[:4]) == tuple(current[:4])
    return False


def _navigation_key_value(value):
    """Convert goal/transition values into deterministic request-key data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "name") and hasattr(value, "value"):
        return (type(value).__name__, value.name)
    if isinstance(value, tuple):
        return tuple(_navigation_key_value(item) for item in value)
    if isinstance(value, (frozenset, set)):
        return tuple(sorted((_navigation_key_value(item) for item in value), key=repr))
    return repr(value)


def _navigation_transition_identity(transition):
    if transition is None:
        return None
    return (
        getattr(transition, "kind", None),
        _navigation_key_value(canonical_location(getattr(transition, "entry", None))),
        _navigation_key_value(canonical_location(getattr(transition, "destination", None))),
        _navigation_key_value(getattr(transition, "required_facing", None)),
        _navigation_key_value(getattr(transition, "activation", None)),
        _navigation_key_value(getattr(transition, "activation_locations", ())),
        _navigation_key_value(getattr(transition, "activation_direction", None)),
    )


def _navigation_goal_identity(goal):
    """Return semantic goal identity without reprs of live observations."""

    if isinstance(goal, NavigationGoal):
        constraints = goal.constraints
        return (
            "NavigationGoal",
            _navigation_goal_identity(goal.target),
            (
                tuple(sorted(constraints.avoid_trigger_ids)),
                tuple(sorted((_navigation_key_value(item) for item in constraints.avoid_locations), key=repr)),
                getattr(getattr(constraints, "trainer_mode", None), "name", None),
                bool(getattr(constraints, "allow_forced_trainer_exposure", False)),
                tuple(sorted(getattr(constraints, "avoid_trainer_ids", ()))),
            ),
            getattr(goal.encounter_mode, "name", goal.encounter_mode),
            goal.encounter_penalty,
            _navigation_key_value(getattr(goal, "preferred_map_route", None)),
        )
    if isinstance(goal, SemanticTarget):
        return (
            "SemanticTarget",
            getattr(goal.kind, "name", goal.kind),
            _navigation_key_value(goal.target_map),
            _navigation_key_value(goal.location),
            goal.interaction_id,
        )
    if isinstance(goal, ReachLocation):
        return ("ReachLocation", _navigation_key_value(canonical_location(goal.location)))
    if isinstance(goal, ReachWarp):
        transitions = tuple(
            sorted(
                (_navigation_transition_identity(transition) for transition in goal.warps),
                key=repr,
            )
        )
        return (
            "ReachWarp",
            _navigation_key_value(goal.destination_map),
            _navigation_key_value(canonical_location(goal.destination)),
            _navigation_transition_identity(goal.warp),
            transitions,
        )
    if isinstance(goal, EngageTrainer):
        return ("EngageTrainer", goal.trainer_id)
    if isinstance(goal, ActivateTrigger):
        return ("ActivateTrigger", goal.trigger_id)
    if isinstance(goal, ReachInteractionPosition):
        return ("ReachInteractionPosition", goal.trigger_id)
    return (type(goal).__name__, _navigation_key_value(goal))


def _navigation_policy_identity(goal) -> tuple:
    constraints = getattr(goal, "constraints", None)
    return (
        getattr(getattr(constraints, "trainer_mode", None), "name", None),
        bool(getattr(constraints, "allow_forced_trainer_exposure", False)),
        tuple(sorted(getattr(constraints, "avoid_trigger_ids", ()))),
        tuple(sorted(getattr(constraints, "avoid_trainer_ids", ()))),
        getattr(getattr(goal, "encounter_mode", None), "name", None),
        getattr(goal, "encounter_penalty", None),
    )


def _navigation_request_key(start, planning_goal, world) -> tuple:
    """Build tactical identity without transient runtime occupancy.

    The worker receives a complete immutable world snapshot, but the request
    identity contains only facts that define ownership of that plan.  In
    particular, live goal reprs and the full moving object table must not
    turn an equivalent plan into a new request every frame.
    """

    _dynamic_blocked, transition_revision = world.revision_key()
    target = _goal_target(planning_goal)
    selected_transition = None
    if isinstance(target, ReachWarp):
        selected_transition = (
            _navigation_transition_identity(target.warp)
            if target.warp is not None
            else tuple(
                sorted(
                    (_navigation_transition_identity(transition) for transition in target.warps),
                    key=repr,
                )
            )
        )
    if selected_transition is None:
        # A semantic route has no selected live transition until the global
        # planner resolves its map edge.  Keep the executable transition
        # catalog as a stable topology identity for those requests; dynamic
        # occupancy remains an execution-time concern.
        selected_transition = tuple(
            sorted((_navigation_key_value(item) for item in transition_revision), key=repr)
        )
    return (
        canonical_location(start),
        _navigation_goal_identity(planning_goal),
        selected_transition,
        _navigation_policy_identity(planning_goal),
    )


def _navigation_failure_revision(world) -> tuple:
    """Return world facts that can reopen a previously failed route.

    The active search key intentionally ignores wandering occupancy so a
    long-running route is not cancelled every frame. Failed results need a
    slightly stronger boundary: a trainer hazard, trigger condition, or
    interaction binding changing can turn an unreachable AVOID route into a
    valid one.
    """

    def trigger_signature(trigger):
        return (
            getattr(trigger, "trigger_id", None),
            getattr(trigger, "affordance_id", None),
            getattr(trigger, "condition_active", None),
            getattr(trigger, "trainer_defeated", None),
            tuple(sorted(getattr(trigger, "hazard_locations", ()) or (), key=repr)),
            tuple(sorted(getattr(trigger, "activation_locations", ()) or (), key=repr)),
            tuple(sorted(getattr(trigger, "engagement_locations", ()) or (), key=repr)),
        )

    return (
        tuple(sorted((trigger_signature(trigger) for trigger in getattr(world, "triggers", ())), key=repr)),
        tuple(sorted(repr(binding) for binding in getattr(world, "bindings", ()))),
    )


def navigation_background_search_active() -> bool:
    """Return whether tactical navigation currently owns a route search."""

    return _background_navigation_planner.active_key() is not None


def notify_battle_ended(*, invalidate_navigation: bool = True) -> None:
    """Publish a battle-end boundary to suspended tactical controllers.

    The normal campaign controller is not advanced while ``BattleListener``
    owns a battle. Consequently an ``AgentControlLoop`` may resume with a
    cached movement route that was composed before a trainer approached or
    moved. A generation counter keeps this notification independent of the
    generator stack and lets the loop invalidate that route on its first
    post-battle frame. Ordinary wild battles do not change overworld
    occupancy, so callers may retain the route and avoid an expensive search
    after every encounter while still refreshing runtime observations.
    """

    global _battle_end_generation, _last_battle_end_emulator_frame
    if invalidate_navigation:
        _battle_end_generation += 1
    get_frame_count = getattr(getattr(context, "emulator", None), "get_frame_count", None)
    try:
        frame = get_frame_count() if callable(get_frame_count) else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        frame = None
    _last_battle_end_emulator_frame = frame if isinstance(frame, int) else None
    # Battle cleanup can move a trainer and update its defeated state without
    # advancing the emulator frame that owns the shared overworld snapshot.
    # Force the first post-battle recovery observation to reread runtime
    # objects rather than reusing the pre-battle occupancy table.
    state_cache.invalidate_runtime_observations()
    # The campaign loop may be replaced while BattleListener owns the battle.
    # In that case the old loop never gets a chance to cancel its movement
    # batch, and a held direction can carry into the newly mounted recovery
    # loop.  Release it at the ownership boundary so recovery starts from a
    # standing avatar and chooses its first input from the post-battle map.
    reset_held_buttons = getattr(getattr(context, "emulator", None), "reset_held_buttons", None)
    if callable(reset_held_buttons):
        reset_held_buttons()


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
    engagement_plan: TrainerEngagementPlan | None = None


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
    try:
        interaction = observe_interaction(
            choice_options=choice_options,
            menu_options=menu_options,
            special_interaction=special_interaction,
        )
    except KeyError as error:
        # A ROM task/map table can be read while its owner is being replaced
        # by an interaction script. Treat that single incomplete read as a
        # retryable observation boundary instead of failing a safety loop.
        diagnostic_print(
            lambda: (
                "AGENT_OBSERVATION_RETRY: stage='interaction' "
                f"exception_type='KeyError' key={error.args!r} "
                f"traceback={traceback.format_exc()!r}"
            ),
            trace=True,
            prefix="AGENT_OBSERVATION_RETRY",
        )
        interaction = InteractionObservation(GameState.UNKNOWN)
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
        try:
            overworld = perceive_overworld()
        except KeyError as error:
            # Keep transient map/trigger binding gaps inside the observation
            # boundary. The next frame will rebuild the live world model.
            diagnostic_print(
                lambda: (
                    "AGENT_OBSERVATION_RETRY: stage='overworld' "
                    f"exception_type='KeyError' key={error.args!r} "
                    f"traceback={traceback.format_exc()!r}"
                ),
                trace=True,
                prefix="AGENT_OBSERVATION_RETRY",
            )
            overworld = None
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


def _campaign_trainer_goal_is_defeated(goal: Goal | None) -> bool:
    """Return whether campaign state has already cleared a trainer goal.

    The live object table can be stale or omit the object immediately after a
    trainer battle.  Campaign battle teardown records the stable affordance
    identity in ``campaign_defeated_trainer_ids``; use that fact at the
    tactical ownership boundary so a suspended trainer loop can return to
    campaign planning without requiring the object to remain visible.
    """

    target = _goal_target(goal)
    if not isinstance(target, EngageTrainer):
        return False
    defeated = set(getattr(context, "campaign_defeated_trainer_ids", ()) or ())
    trainer_id = getattr(target, "trainer_id", None)
    if trainer_id is None:
        return False
    aliases = {("trainer_affordance_id", trainer_id)}
    # A few test and compatibility embeddings publish the trainer goal as a
    # checkpoint ID.  Accept that alias only when campaign teardown explicitly
    # recorded it; never infer defeat from a missing live object.
    aliases.add(("checkpoint_id", trainer_id))
    return bool(aliases.intersection(defeated))


def _map_goal_completion_reason(observation: AgentObservation) -> str | None:
    """Return a completion reason when a local navigation goal is already visible.

    Background route planning may still be pending on the first frame after a
    map transition.  Map goals can be settled locally from the observed map
    header, so they must not wait for a stale or unnecessary worker result.
    Exact ``ReachLocation`` goals can be settled from the observed coordinate
    for the same reason. Interaction targets still require their normal
    navigation evaluation.
    """
    if observation.overworld is None:
        return None
    target = _goal_target(observation.goal)
    current_location = (observation.overworld.map_id, observation.overworld.player_coordinates)
    if isinstance(target, ReachLocation) and current_location == target.location:
        return "goal position reached"
    destination_map = None
    if isinstance(target, SemanticTarget):
        if target.kind.name != "MAP":
            return None
        destination_map = target.target_map
    elif isinstance(target, ReachWarp):
        destination_map = target.destination_map
        if destination_map is None and target.destination is not None:
            destination_map = target.destination[0]
    if destination_map is not None and observation.overworld.map_id == destination_map:
        return "destination map reached"
    return None


def _effective_navigation_goal(observation: AgentObservation, world: NavigationWorld):
    """Project a trainer goal onto the executable interaction geometry."""

    goal_target = _goal_target(observation.goal)
    if not isinstance(goal_target, EngageTrainer):
        return observation.goal, None
    engagement_plan = resolve_trainer_engagement_plan(world, goal_target.trainer_id)
    interaction_target = engagement_plan.interaction_target
    if interaction_target is None:
        return observation.goal, engagement_plan
    planning_goal = (
        replace(observation.goal, target=interaction_target)
        if isinstance(observation.goal, NavigationGoal)
        else interaction_target
    )
    return planning_goal, engagement_plan


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


def _interaction_trigger_id(goal: Goal | None, evaluation: GoalEvaluation | None) -> str | None:
    """Return the trigger currently owning an adjacent interaction goal."""

    target = _goal_target(goal)
    if isinstance(target, (ActivateTrigger, ReachInteractionPosition)):
        return target.trigger_id
    if isinstance(target, EngageTrainer) and evaluation is not None:
        engagement_plan = evaluation.engagement_plan
        if engagement_plan is not None and engagement_plan.mode is TrainerEngagementMode.INTERACTION:
            return engagement_plan.trigger_id
    return None


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
def evaluate_goal(observation: AgentObservation, *, navigation_result=_NAVIGATION_PLAN_UNSET) -> GoalEvaluation:
    """Evaluate a tactical goal against the current overworld observation."""

    trace = getattr(context, "stutter_trace", None)
    span = trace.span("goal_evaluation") if trace is not None else nullcontext()
    with span:
        return _evaluate_goal_instrumented(observation, navigation_result=navigation_result)


def _evaluate_goal_instrumented(observation: AgentObservation, *, navigation_result=_NAVIGATION_PLAN_UNSET) -> GoalEvaluation:
    if observation.goal is None:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="no goal supplied")
    if observation.overworld is None or observation.interaction_type is not InteractionType.OVERWORLD:
        return GoalEvaluation(GoalStatus.NOT_APPLICABLE, reason="goal requires an overworld observation")

    goal_target = _goal_target(observation.goal)
    if _campaign_trainer_goal_is_defeated(observation.goal):
        return GoalEvaluation(
            GoalStatus.COMPLETE,
            reason="campaign trainer clearance recorded",
            world_diagnostics=(
                f"trainer_id={getattr(goal_target, 'trainer_id', None)!r}",
                "campaign_defeated_trainer_ids=True",
            ),
        )
    if isinstance(goal_target, ActivateTrigger) and _trigger_condition_completed(observation, goal_target.trigger_id):
        return GoalEvaluation(
            GoalStatus.COMPLETE,
            reason="trigger condition is inactive; ROM scene already completed",
        )

    # Diagnostics only: this deliberately does not gate, score, or alter the
    # navigation decision.  Keep the snapshot at the same overworld decision
    # boundary as the active goal evaluation so later telemetry can explain
    # exactly what was available before a trainer approach.  Readiness
    # diagnostics are expensive state reads, so do not collect them on every
    # frame unless the trace stream has explicitly been requested.
    if context.debug and getattr(context, "debug_trace", False):
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
                lambda: "CAMPAIGN_READINESS: " + json.dumps(diagnostic.as_dict(), sort_keys=True, default=str),
                trace=True,
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
    planning_goal, engagement_plan = _effective_navigation_goal(observation, world)
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
                f" static_availability={resolution.static_availability}"
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
        if navigation_result is _NAVIGATION_PLAN_UNSET:
            plan, world_route = plan_with_world_navigation(
                world,
                start,
                planning_goal,
                caller="agent_control.goal_evaluation",
            )
        else:
            plan, world_route, planning_error = navigation_result
            if plan is None and planning_error is None:
                planning_error = NavigationError("navigation planner returned no plan or error")
            if planning_error is not None:
                raise planning_error
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PLAN_TRACE: phase=world_plan_end "
                f"actions={len(plan.actions)} destination={plan.destination!r} "
                f"first={plan.actions[0] if plan.actions else None!r} "
                f"last={plan.actions[-1] if plan.actions else None!r}"
            ),
            trace=True,
        )
        constraints = getattr(observation.goal, "constraints", None)
        trainer_avoidance_enabled = getattr(constraints, "trainer_mode", None) is TrainerMode.AVOID
        forced_trainer_exposure_allowed = bool(
            getattr(constraints, "allow_forced_trainer_exposure", False)
        )
        if (
            getattr(plan, "forced_trainer_exposure", False)
            and trainer_avoidance_enabled
            and not forced_trainer_exposure_allowed
        ):
            # GoalAwareNavigator retains a general-purpose fallback for
            # callers that explicitly prefer avoiding trainers but can still
            # accept an exposed route. Recovery is different: it is mounted
            # to protect a damaged party, so dispatching that fallback would
            # turn a dynamic trainer hazard into an unplanned battle.
            return GoalEvaluation(
                GoalStatus.UNREACHABLE,
                reason="no trainer-free route is currently available",
                diagnostics=navigation_diagnostics(world, start, observation.goal),
                binding_diagnostics=binding_diagnostics,
                world_diagnostics=world_diagnostics + ("route rejected: forced trainer exposure",),
                engagement_plan=engagement_plan,
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
                local_plan = plan_observed_warp_locally(
                    world,
                    start,
                    observation.goal,
                    caller="agent_control.goal_evaluation.local_observed_warp",
                )
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
                    engagement_plan=engagement_plan,
                )
        timing("goal_planning", planning_start)
        count("goal_planning_attempts")
        return GoalEvaluation(
            GoalStatus.UNREACHABLE,
            reason=str(error),
            diagnostics=navigation_diagnostics(world, start, observation.goal),
            binding_diagnostics=binding_diagnostics,
            world_diagnostics=world_diagnostics,
            engagement_plan=engagement_plan,
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
                local_plan = plan_observed_warp_locally(
                    world,
                    start,
                    observation.goal,
                    caller="agent_control.goal_evaluation.local_observed_warp",
                )
            except NavigationError:
                local_plan = None
            if local_plan is not None and local_plan.actions:
                return GoalEvaluation(
                    GoalStatus.REACHABLE,
                    plan=local_plan,
                    reason="warp entry observed; activation still required",
                    binding_diagnostics=binding_diagnostics,
                    world_diagnostics=world_diagnostics,
                    engagement_plan=engagement_plan,
                )
        return GoalEvaluation(
            GoalStatus.COMPLETE,
            plan=plan,
            reason="goal position reached",
            binding_diagnostics=binding_diagnostics,
            world_diagnostics=world_diagnostics,
            engagement_plan=engagement_plan,
        )
    if isinstance(goal_target, EngageTrainer):
        trainer = next((obj for obj in observation.overworld.objects if obj.trainer_id == goal_target.trainer_id), None)
        if trainer is not None and trainer.trainer_defeated is True:
            return GoalEvaluation(
                GoalStatus.COMPLETE,
                plan=plan,
                reason="engaged trainer is defeated",
                world_diagnostics=(f"trainer_id={goal_target.trainer_id!r}", "trainer_defeated=True"),
                engagement_plan=engagement_plan,
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
                engagement_plan=engagement_plan,
            )
    return GoalEvaluation(
        GoalStatus.REACHABLE,
        plan=plan,
        binding_diagnostics=binding_diagnostics,
        world_diagnostics=world_diagnostics,
        engagement_plan=engagement_plan,
    )


@profiled("agent_action_selection", "actions_selected")
def select_action(observation: AgentObservation, *, navigation_result=_NAVIGATION_PLAN_UNSET) -> ActionDecision:
    """Select one deterministic action after state-specific affordance generation."""

    trace = getattr(context, "stutter_trace", None)
    span = trace.span("action_selection") if trace is not None else nullcontext()
    with span:
        return _select_action_instrumented(observation, navigation_result=navigation_result)


def _select_action_instrumented(observation: AgentObservation, *, navigation_result=_NAVIGATION_PLAN_UNSET) -> ActionDecision:
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
        if (
            observation.interaction.interaction_phase is InteractionPhase.FIELD_MESSAGE_RENDER_WAIT
            and observation.interaction.field_message_render_rescue_available
        ):
            return ActionDecision(
                AgentAction(
                    AgentActionType.ACCELERATE_DIALOGUE_RENDER,
                    reason="fresh-B render rescue while scripted field message is rendering",
                )
            )
        if observation.interaction.interaction_phase is InteractionPhase.FIELD_MESSAGE_INPUT_WAIT:
            return ActionDecision(
                AgentAction(
                    AgentActionType.ADVANCE_DIALOGUE,
                    reason="field dialogue transition requires input",
                )
            )
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason="player avatar is not controllable"))
    if navigation_result is _NAVIGATION_PLAN_PENDING:
        map_completion_reason = _map_goal_completion_reason(observation)
        if map_completion_reason is not None:
            return ActionDecision(
                AgentAction(AgentActionType.WAIT_REOBSERVE, reason=map_completion_reason),
                GoalEvaluation(GoalStatus.COMPLETE, reason=map_completion_reason),
            )
        return ActionDecision(
            AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="navigation plan is being calculated off the frame thread",
            )
        )
    evaluation = evaluate_goal(observation, navigation_result=navigation_result)
    if evaluation.status is GoalStatus.COMPLETE:
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation)
    if evaluation.status is GoalStatus.UNREACHABLE:
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation)
    if evaluation.status is GoalStatus.NOT_APPLICABLE:
        return ActionDecision(AgentAction(AgentActionType.WAIT_REOBSERVE, reason=evaluation.reason), evaluation)
    if (
        isinstance(_goal_target(observation.goal), EngageTrainer)
        and evaluation.status is GoalStatus.REACHABLE
        and evaluation.plan is not None
        and not evaluation.plan.actions
    ):
        if evaluation.engagement_plan is not None:
            if evaluation.engagement_plan.mode is TrainerEngagementMode.INTERACTION:
                trigger_id = evaluation.engagement_plan.trigger_id
                if trigger_id is not None:
                    world = NavigationWorld.from_overworld(observation.overworld)
                    interaction_target = evaluation.engagement_plan.interaction_target
                    if interaction_target is not None and GoalAwareNavigator(world).satisfies(
                        (observation.overworld.map_id, observation.overworld.player_coordinates),
                        observation.overworld.facing,
                        interaction_target,
                    ):
                        return ActionDecision(
                            AgentAction(
                                AgentActionType.INTERACT,
                                option=trigger_id,
                                reason="trainer interaction position reached",
                            ),
                            evaluation,
                        )
                return ActionDecision(
                    AgentAction(
                        AgentActionType.WAIT_REOBSERVE,
                        reason="waiting for trainer interaction position",
                    ),
                    evaluation,
                )
            if evaluation.engagement_plan.mode is TrainerEngagementMode.UNKNOWN:
                return ActionDecision(
                    AgentAction(
                        AgentActionType.WAIT_REOBSERVE,
                        reason="waiting for trainer engagement geometry",
                    ),
                    evaluation,
                )
        return ActionDecision(
            AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="waiting for trainer sight-line activation",
            ),
            evaluation,
        )
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


def _is_intermediate_move_location(location, action: NavigationAction) -> bool:
    """Return whether ``location`` is progress along a skipped move.

    The legacy Emerald pathfinder can represent forced movement (ledges,
    slopes, and similar tiles) as one cardinal action whose destination is
    more than one tile away. During that movement the avatar can be observed
    on a coordinate strictly between the action endpoints. That is a valid
    in-flight state, not route divergence.
    """

    if action.action_type is not NavigationActionType.MOVE or action.destination is None:
        return False
    source_map, source_coordinates = action.source
    destination_map, destination_coordinates = action.destination
    if location[0] != source_map or source_map != destination_map:
        return False
    source_x, source_y = source_coordinates
    destination_x, destination_y = destination_coordinates
    current_x, current_y = location[1]
    if action.direction is Direction.North:
        return current_x == source_x == destination_x and destination_y < current_y < source_y
    if action.direction is Direction.East:
        return current_y == source_y == destination_y and source_x < current_x < destination_x
    if action.direction is Direction.South:
        return current_x == source_x == destination_x and source_y < current_y < destination_y
    if action.direction is Direction.West:
        return current_y == source_y == destination_y and destination_x < current_x < source_x
    return False


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

    def reset_navigation_held_inputs(self) -> None:
        """Clear movement inputs without interrupting an owned text render.

        Plan invalidation normally has to release every held input: a cached
        movement direction must not leak into a new interaction or route.
        ``ACCELERATE_DIALOGUE_RENDER`` is different.  Its held B is an
        executor-owned part of the field-message transaction, and plan
        invalidation happens on every non-overworld observation while that
        transaction is active.  Preserve that one input until the executor
        observes the message's next phase and releases it normally.
        """

        preserve_render_b = self._field_message_render_b_held
        reset_held_buttons = getattr(context.emulator, "reset_held_buttons", None)
        if callable(reset_held_buttons):
            reset_held_buttons()
        if preserve_render_b:
            hold_button = getattr(context.emulator, "hold_button", None)
            if callable(hold_button):
                hold_button("B")

    @profiled("agent_action_execution", "actions_executed")
    def execute(self, action: AgentAction, observation: AgentObservation) -> ActionResult:
        trace = getattr(context, "stutter_trace", None)
        span = trace.span("action_construction_and_execution") if trace is not None else nullcontext()
        with span:
            return self._execute_instrumented(action, observation)

    def _execute_instrumented(self, action: AgentAction, observation: AgentObservation) -> ActionResult:
        if getattr(context, "debug", False) and action.action_type is AgentActionType.ADVANCE_DIALOGUE:
            get_frame_count = getattr(context.emulator, "get_frame_count", None)
            try:
                input_frame = get_frame_count() if callable(get_frame_count) else None
            except (AttributeError, RuntimeError, TypeError, ValueError):
                input_frame = None
            is_button_held = getattr(context.emulator, "is_button_held", None)
            try:
                b_held = is_button_held("B") if callable(is_button_held) else None
            except (AttributeError, RuntimeError, TypeError, ValueError):
                b_held = None
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_LIVE_DIALOGUE_INPUT: "
                    f"frame={input_frame!r} operation='A' b_held_before={b_held!r} "
                    f"phase={observation.interaction.interaction_phase.name!r} "
                    f"native={observation.interaction.native_function!r} "
                    f"script={observation.interaction.script_function!r}"
                )
            )
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
                    context.emulator.press_button_fresh(Direction.North.button_name)
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
    # Battle return can leave the avatar's movement flags stale even though
    # control has returned and its logical tile no longer changes. Do not
    # wait forever on that flag, but give a real movement transition enough
    # frames to advance before treating it as stale.
    _post_battle_stale_movement_limit = 8
    # Emerald can leave the avatar's movement flag set after the logical
    # coordinate has stopped changing. Keep the normal settling window long
    # enough for a genuine step, but do not let a stale flag suppress an
    # interaction route forever.
    _stale_interaction_movement_limit = 24
    # A single delayed re-observation absorbs a stale transition/object table
    # without allowing an unreachable route to spin forever.
    _terminal_navigation_fallback_wait_frames = 20
    # A trainer battle or script should claim the field shortly after the
    # engagement tile is reached. Bound an unchanged automatic wait so a stale
    # trainer trigger cannot keep the campaign generator alive forever.
    _automatic_trainer_wait_limit = 32

    def __init__(
        self,
        observe: Callable[[], AgentObservation],
        executor: AgentActionExecutor | None = None,
        *,
        goal: Goal | None = None,
        navigation_plan: NavigationPlan | None = None,
        logger: Callable[[str], None] | None = None,
        use_movement_batch: bool = True,
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
        # Recovery is safety-critical and may begin immediately after a
        # trainer battle. Keep its movement on the full observation boundary
        # so a live object moving onto a route tile is seen before the next
        # input. Ordinary campaign navigation retains the bounded fast path.
        self._use_movement_batch = use_movement_batch
        # Campaign Progression is the live owner that can spend multiple
        # milliseconds composing cross-map routes. Keep the pure control-loop
        # and validation modes synchronous so their deterministic test seams
        # remain unchanged.
        self._background_navigation_enabled = context.bot_mode == "Campaign Progression"
        # The global background worker is shared across mounted controllers;
        # this token prevents a result from one loop/goal boundary from being
        # adopted by the next loop merely because their requests look alike.
        self._navigation_owner = object()
        self._navigation_owner_label = f"agent_control_loop:{id(self)}"
        self._last_interaction_type: InteractionType | None = None
        self._battle_was_active = False
        self._last_navigation_diagnostics: tuple[str, ...] | None = None
        self._last_binding_diagnostics: tuple[str, ...] | None = None
        self._last_world_diagnostics: tuple[str, ...] | None = None
        self._last_navigation_poll_report: tuple | None = None
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
        self._terminal_navigation_fallback_used = False
        self._terminal_navigation_fallback_attempts = 0
        self._rejected_first_action_signature: tuple | None = None
        self._rejected_first_action_count = 0
        # A failed movement is evidence of an obstacle even when the ROM's
        # object table has not exposed the object yet. Keep that evidence
        # across the immediate replan so a stale/partial observation cannot
        # produce the same input forever.
        self._locally_blocked_destinations: set[tuple] = set()
        self._dialogue_input_in_flight = False
        self._started_interaction_id: str | None = None
        self._interaction_start_waits = 0
        # Some Emerald object interactions start automatically when the
        # avatar enters their activation position. Keep that boundary
        # separate from `_started_interaction_id`, which is only populated
        # after this loop emits an explicit A input.
        self._interaction_position_ready = False
        self._last_observed_location: Location | None = None
        self._previous_observed_location: Location | None = None
        self._battle_end_generation_seen = _battle_end_generation
        self._post_battle_movement_location: Location | None = None
        self._post_battle_movement_observations = 0
        self._post_battle_movement_pending = self._recent_battle_return()
        self._stale_interaction_movement_location: Location | None = None
        self._stale_interaction_movement_observations = 0
        self._automatic_trainer_wait_signature: tuple | None = None
        self._automatic_trainer_wait_observations = 0
        self._automatic_trainer_wait_reobserved = False
        self._recovery_first_action_reported = False

    @staticmethod
    def _recent_battle_return() -> bool:
        """Return whether this loop was mounted immediately after a battle.

        Campaign recovery generators are created after ``notify_battle_ended``
        has already advanced the controller, so a newly constructed loop has
        no generation edge to observe. The frame marker bridges that handoff
        without making every later loop permanently battle-sensitive.
        """

        if _last_battle_end_emulator_frame is None:
            return False
        get_frame_count = getattr(getattr(context, "emulator", None), "get_frame_count", None)
        if not callable(get_frame_count):
            return False
        try:
            frame = get_frame_count()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return (
            isinstance(frame, int)
            and frame >= _last_battle_end_emulator_frame
            and frame - _last_battle_end_emulator_frame <= _recent_battle_return_window
        )

    @property
    def navigation_transition_active(self) -> bool:
        """Return whether this loop still owns a map or door transition.

        A transition remains owned through both the destination-map
        observation and the short post-warp settling window. Campaign
        planning must not replace the suspended loop during either phase,
        because doing so prevents the loop from consuming its own transition
        checkpoint and leaves the controller without a valid route suffix.
        """

        return (
            self._pending_transition is not None
            or self._expected_world_transition is not None
            or self._warp_settling
        )

    def _resolve_stale_post_battle_movement(self, observation: AgentObservation) -> AgentObservation:
        """Bound a stale ``MOVING`` flag after battle control returns.

        The field engine can retain ``MOVING`` in the avatar state after a
        trainer battle has ended. In that state the normal control loop waits
        before evaluating the recovery goal, so an avatar adjacent to the
        defeated trainer can remain motionless forever. This is deliberately
        limited to the battle-return boundary and only applies after the
        avatar is controllable, no transition is active, and its logical
        location is unchanged for a bounded number of observations.

        The held-input reset is defensive: ``notify_battle_ended`` and plan
        invalidation already release controller-owned input, but a recovery
        loop may be mounted after either of those boundaries.
        """

        world = observation.overworld
        if not self._post_battle_movement_pending:
            return observation
        if world is None or observation.interaction_type is not InteractionType.OVERWORLD:
            self._post_battle_movement_location = None
            self._post_battle_movement_observations = 0
            return observation
        movement_state = world.movement_state
        if movement_state not in (MovementState.MOVING, MovementState.TURNING):
            # A real settled observation proves that the movement flag is no
            # longer suppressing planning. Clear the boundary state so later
            # ordinary movement is governed by the normal in-flight logic.
            self._post_battle_movement_pending = False
            self._post_battle_movement_location = None
            self._post_battle_movement_observations = 0
            return observation
        if (
            not world.controllable
            or not observation.interaction.controllable
            or world.transition_in_progress
            or self._pending_transition is not None
            or self._in_flight_move is not None
            or self._movement_batch is not None
        ):
            self._post_battle_movement_location = None
            self._post_battle_movement_observations = 0
            return observation

        location = (world.map_id, world.player_coordinates)
        if location == self._post_battle_movement_location:
            self._post_battle_movement_observations += 1
        else:
            self._post_battle_movement_location = location
            self._post_battle_movement_observations = 1
        if self._post_battle_movement_observations < self._post_battle_stale_movement_limit:
            return observation

        reset_held_buttons = getattr(context.emulator, "reset_held_buttons", None)
        if callable(reset_held_buttons):
            reset_held_buttons()
        self._report(
            "BATTLE_RETURN: treating stationary MOVING state as settled "
            f"location={location!r} observations={self._post_battle_movement_observations}"
        )
        self._post_battle_movement_pending = False
        self._post_battle_movement_location = None
        self._post_battle_movement_observations = 0
        return replace(observation, overworld=replace(world, movement_state=MovementState.STANDING))

    def _resolve_stale_interaction_movement(self, observation: AgentObservation) -> AgentObservation:
        """Bound a stale movement flag while pursuing an interaction goal.

        The field engine may publish a coordinate before it clears the
        avatar's movement state. That is normally a short, legitimate
        settling interval. If the same coordinate remains stationary while
        an interaction route is still active, however, waiting on ``MOVING``
        forever prevents the route from advancing. Restrict this recovery to
        an active interaction goal so ordinary movement retains its normal
        settling checks, including the initial observation before a route has
        been cached.
        """

        world = observation.overworld
        target = _goal_target(self._goal)
        trainer_interaction = (
            isinstance(target, EngageTrainer)
            and self._cached_evaluation is not None
            and self._cached_evaluation.engagement_plan is not None
            and self._cached_evaluation.engagement_plan.mode is TrainerEngagementMode.INTERACTION
        )

        def reset() -> AgentObservation:
            self._stale_interaction_movement_location = None
            self._stale_interaction_movement_observations = 0
            return observation

        if (
            world is None
            or observation.interaction_type is not InteractionType.OVERWORLD
            or (not isinstance(target, (ActivateTrigger, ReachInteractionPosition)) and not trainer_interaction)
        ):
            return reset()
        if world.movement_state not in (MovementState.MOVING, MovementState.TURNING):
            return reset()
        if (
            not world.controllable
            or not observation.interaction.controllable
            or world.transition_in_progress
            or self._pending_transition is not None
            or (self._cached_goal is not None and self._cached_goal != self._goal)
        ):
            return reset()

        location = (world.map_id, world.player_coordinates)

        if location == self._stale_interaction_movement_location:
            self._stale_interaction_movement_observations += 1
        else:
            self._stale_interaction_movement_location = location
            self._stale_interaction_movement_observations = 1
        if self._stale_interaction_movement_observations < self._stale_interaction_movement_limit:
            return observation

        if self._movement_batch is not None:
            self._cancel_movement_batch("stale_interaction_movement")
        else:
            reset_held_buttons = getattr(context.emulator, "reset_held_buttons", None)
            if callable(reset_held_buttons):
                reset_held_buttons()
        self._report(
            "INTERACTION_MOVEMENT_WATCHDOG: treating stationary MOVING state as settled "
            f"location={location!r} "
            f"observations={self._stale_interaction_movement_observations} "
            f"route_actions_remaining={len(self._cached_actions) - self._cached_action_index} "
            f"in_flight_move={self._in_flight_move!r} "
            f"target={target!r}"
        )
        self._stale_interaction_movement_location = None
        self._stale_interaction_movement_observations = 0
        return replace(observation, overworld=replace(world, movement_state=MovementState.STANDING))

    def _seed_navigation_plan(self, observation: AgentObservation) -> None:
        """Install a planner-supplied route once its origin is observable."""

        plan = self._initial_navigation_plan
        if plan is None or self._cached_evaluation is not None or observation.overworld is None:
            return
        actions = tuple(getattr(plan, "actions", ()) or ())
        origin = actions[0].source if actions else getattr(plan, "destination", None)
        current_location = canonical_location((observation.overworld.map_id, observation.overworld.player_coordinates))
        if canonical_location(origin) != current_location:
            self._report(
                "ROUTE_PLAN_REJECTED "
                f"reason='origin mismatch' origin={origin!r} current={current_location!r}"
            )
            # A recovery continuation is valid only from the exact Center
            # exit where the planner composed it. If the first observation
            # is still inside the Center (or the world changed), clear it so
            # the normal goal planner can build a route from the real tile.
            self._initial_navigation_plan = None
            return
        constraints = getattr(self._goal, "constraints", None)
        trainer_mode = getattr(constraints, "trainer_mode", None)
        trainer_avoidance_enabled = getattr(trainer_mode, "name", None) == "AVOID"
        forced_trainer_exposure_allowed = bool(
            getattr(constraints, "allow_forced_trainer_exposure", False)
        )
        trainer_hazards = frozenset(
            location
            for trigger in getattr(observation.overworld, "triggers", ())
            for location in getattr(trigger, "hazard_locations", ())
        )
        trainer_approaches = frozenset(
            location
            for trigger in getattr(observation.overworld, "triggers", ())
            if getattr(trigger, "hazard_kind", None) == "trainer"
            or getattr(trigger, "kind", None) in {"trainer", "trainer_hazard"}
            for location in getattr(trigger, "activation_locations", ())
        )
        trainer_hazards |= trainer_approaches
        hazard_conflict = next(
            (
                (index, location, action)
                for index, action in enumerate(plan.actions)
                if trainer_avoidance_enabled
                and action.action_type in (NavigationActionType.MOVE, NavigationActionType.TURN)
                for location in (action.source, action.destination)
                if location is not None and location in trainer_hazards
            ),
            None,
        )
        if trainer_avoidance_enabled and not forced_trainer_exposure_allowed and (
            hazard_conflict is not None or getattr(plan, "forced_trainer_exposure", False)
        ):
            if hazard_conflict is None:
                detail = "planner marked route as forced trainer exposure"
            else:
                index, location, action = hazard_conflict
                detail = (
                    f"action_index={index} hazard={location!r} "
                    f"source={action.source!r} destination={action.destination!r}"
                )
            self._report("ROUTE_PLAN_REJECTED " f"reason='trainer hazard on supplied route' {detail}")
            # The route was composed from a prior readiness observation. Leave
            # the goal available to the ordinary planner, which will rebuild
            # it with the current live trainer constraints.
            self._initial_navigation_plan = None
            return
        blocked_action = self._blocked_action(observation, plan.actions[0] if plan.actions else None)
        if blocked_action is not None:
            self._report(
                "ROUTE_PLAN_REJECTED "
                f"reason='destination currently blocked' source={blocked_action.source!r} "
                f"destination={blocked_action.destination!r}"
            )
            self._reject_first_action(blocked_action, "supplied route destination blocked", observation)
            # The route was composed from an earlier readiness observation.
            # Leave the goal available to the ordinary planner, which will
            # rebuild it against the current dynamic occupancy.
            self._initial_navigation_plan = None
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
        self._report(
            "ROUTE_PLAN_INPUT "
            f"map={observation.overworld.map_id!r} "
            f"position={observation.overworld.player_coordinates!r} "
            f"dynamic_blocked={tuple(sorted(observation.overworld.dynamic_blocked_coordinates))!r} "
            f"static_blocked={tuple(sorted(getattr(observation.overworld, 'static_blocked_coordinates', ())))!r} "
            f"objects={tuple((obj.local_id, obj.location, obj.trainer_defeated) for obj in observation.overworld.objects)!r} "
            f"actions={tuple((action.action_type.name, action.source, action.destination, action.direction.name) for action in plan.actions[:8])!r}"
        )

    @staticmethod
    def _blocked_coordinates(world) -> set[tuple[int, int]]:
        """Return all live same-map coordinates that must not receive input."""

        object_blocked_coordinates = {
            object_observation.location[1]
            for object_observation in getattr(world, "objects", ())
            if object_observation.location[0] == world.map_id
        }
        return (
            set(getattr(world, "dynamic_blocked_coordinates", ()))
            | set(getattr(world, "static_blocked_coordinates", ()))
            | object_blocked_coordinates
        )

    @staticmethod
    def _trainer_hazard_coordinates(world) -> set[tuple[int, int]]:
        """Return live trainer sight and approach positions on this map.

        Trainer hazards are deliberately separate from collision.  The player
        can legally walk onto a trainer's sight-line or interaction position,
        but doing so may hand control to the ROM and start a battle before the
        next planner observation.  Keep the final action gate conservative so
        a route cannot cross a newly observed trainer merely because its tile
        is not occupied.
        """

        map_id = getattr(world, "map_id", None)
        hazards: set[tuple[int, int]] = set()
        for trigger in getattr(world, "triggers", ()):
            hazard_kind = getattr(trigger, "hazard_kind", None)
            trigger_kind = getattr(trigger, "kind", None)
            if hazard_kind != "trainer" and trigger_kind not in {"trainer", "trainer_hazard"}:
                continue
            sight_line = frozenset(
                location for location in getattr(trigger, "hazard_locations", ()) if location[0] == map_id
            )
            hazards.update(sight_line)
            # Some older/injected observations expose trainer activation
            # positions without a computed sight line.  Those are still unsafe
            # under AVOID and must be fail-closed at the input boundary.  Once
            # a positive sight-line envelope exists, an adjacent interaction
            # tile is only reached by pressing A, so it is not a movement
            # hazard.  This must match the planner's own model
            # (NavigationWorld hazard test): validating more strictly than the
            # planner rejects the planner's only trainer-free corridor and
            # leaves the control loop replanning the same route forever.
            if not sight_line:
                hazards.update(
                    location for location in getattr(trigger, "activation_locations", ()) if location[0] == map_id
                )
        return hazards

    @staticmethod
    def _blocked_cached_action(
        observation: AgentObservation,
        action: NavigationAction | None,
    ) -> NavigationAction | None:
        """Return a movement action whose live destination is occupied.

        A planner-supplied route can outlive the observation used to compose
        it.  This is particularly common after a trainer battle, because the
        battle listener may return control after the trainer has moved.  The
        route signature is not sufficient at adoption time: its baseline is
        initialized from the current observation.  Check the action itself at
        the final input boundary instead.
        """

        world = observation.overworld
        if (
            action is None
            or world is None
            or action.action_type is not NavigationActionType.MOVE
            or action.destination is None
            or action.destination[0] != world.map_id
        ):
            return None
        blocked_coordinates = AgentControlLoop._blocked_coordinates(world)
        if action.destination[1] not in blocked_coordinates:
            return None
        return action

    def _blocked_action(
        self,
        observation: AgentObservation,
        action: NavigationAction | None,
    ) -> NavigationAction | None:
        """Apply observed and locally inferred occupancy to one action."""

        blocked_action = self._blocked_cached_action(observation, action)
        if blocked_action is not None:
            return blocked_action
        if (
            action is not None
            and action.action_type is NavigationActionType.MOVE
            and action.destination is not None
            and self._trainer_avoidance_enabled()
            and action.destination in self._trainer_hazard_coordinates(observation.overworld)
        ):
            self._report("TRAINER_HAZARD_BLOCKED " f"source={action.source!r} destination={action.destination!r}")
            return action
        if (
            action is not None
            and action.action_type is NavigationActionType.MOVE
            and action.destination is not None
            and action.destination in self._locally_blocked_destinations
        ):
            return action
        return None

    def _trainer_avoidance_enabled(self) -> bool:
        constraints = getattr(self._goal, "constraints", None)
        trainer_mode = getattr(constraints, "trainer_mode", None)
        return getattr(trainer_mode, "name", None) == "AVOID"

    def _remember_blocked_destination(self, action: NavigationAction | None, reason: str) -> None:
        """Remember a movement destination that the emulator refused."""

        if action is None or action.action_type is not NavigationActionType.MOVE or action.destination is None:
            return
        destination = action.destination
        self._locally_blocked_destinations.add(destination)
        self._report(
            "AGENT_REPLAN: local movement obstacle "
            f"source={action.source!r} destination={destination!r} reason={reason!r}"
        )

    def _reject_first_action(self, action: NavigationAction | None, reason: str, observation) -> None:
        """Feed a rejected route edge back into planning and bound retries."""

        if action is None:
            return
        current = (
            (observation.overworld.map_id, observation.overworld.player_coordinates)
            if observation is not None and observation.overworld is not None
            else None
        )
        signature = (
            _navigation_goal_identity(self._goal),
            current,
            action.action_type.name,
            action.source,
            action.destination,
            getattr(action.direction, "name", action.direction),
        )
        if signature == self._rejected_first_action_signature:
            self._rejected_first_action_count += 1
        else:
            self._rejected_first_action_signature = signature
            self._rejected_first_action_count = 1
        self._remember_blocked_destination(action, reason)
        self._report(
            "ROUTE_ACTION_REJECTED "
            f"count={self._rejected_first_action_count} reason={reason!r} "
            f"source={action.source!r} destination={action.destination!r}"
        )
        if self._rejected_first_action_count >= 2:
            # A dynamic occupancy read can be stale for one frame.  After the
            # same edge is rejected again, force the next search to consume a
            # fresh object table while the locally learned block prevents an
            # identical route from being selected immediately again.
            state_cache.invalidate_runtime_observations()
            self._report(
                "ROUTE_ACTION_REJECTION_BOUND "
                f"count={self._rejected_first_action_count} action={action.destination!r}"
            )

    def _clear_rejected_first_action(self) -> None:
        self._rejected_first_action_signature = None
        self._rejected_first_action_count = 0

    def _local_detour_around_blocked_action(
        self,
        observation: AgentObservation,
        blocked_action: NavigationAction,
    ) -> tuple[NavigationAction, ...] | None:
        """Reuse a cached route after a safe, local occupancy change.

        A wandering object can occupy the next tile after a route has already
        been approved.  Re-running the full cross-map planner for that one
        edge is safe but can leave the emulator waiting for several seconds.
        Search a small same-map neighborhood for the first later source tile
        in the cached suffix, retaining the full planner as the fallback when
        no bounded detour exists.

        Trainer hazard coordinates remain forbidden throughout this search.
        This optimization therefore cannot turn ``TrainerMode.AVOID`` into a
        permissive route, and it never crosses a warp while repairing a local
        movement edge.
        """

        world_observation = observation.overworld
        if (
            world_observation is None
            or blocked_action.action_type is not NavigationActionType.MOVE
            or blocked_action.destination is None
            or blocked_action.source[0] != world_observation.map_id
            or blocked_action.destination[0] != world_observation.map_id
            or self._cached_action_index >= len(self._cached_actions)
        ):
            return None

        remaining = self._cached_actions[self._cached_action_index :]
        if not remaining or remaining[0] != blocked_action:
            return None

        # The suffix must stay on the current map until the first warp.  A
        # local repair ending at a later source tile can then append the
        # already-approved suffix without invoking global pathfinding.
        candidates: dict[tuple, tuple[int, int]] = {}
        for index, action in enumerate(remaining[1:], start=1):
            if action.action_type is not NavigationActionType.MOVE:
                break
            if action.source[0] != world_observation.map_id or action.destination[0] != world_observation.map_id:
                break
            # Prefer the next action's source, but also allow its destination
            # as an anchor. The source can remain occupied by the NPC that
            # blocked the original edge; reaching the open destination lets
            # the repair skip that stale edge entirely.
            candidates.setdefault(action.source, (index, index))
            candidates.setdefault(action.destination, (index, index + 1))
        if not candidates:
            return None

        world = NavigationWorld.from_overworld(world_observation)
        trainer_hazards = self._trainer_hazard_coordinates(world_observation) if self._trainer_avoidance_enabled() else set()
        start = (world_observation.map_id, world_observation.player_coordinates)
        blocked_coordinates = self._blocked_coordinates(world_observation)
        if start in trainer_hazards:
            return None

        queue = deque([start])
        came_from: dict[tuple, tuple[tuple, Direction]] = {}
        visited = {start}
        anchor: tuple | None = None
        append_index: int | None = None
        # The local repair is deliberately bounded.  If a real route choice
        # is required, the normal planner must re-evaluate the complete world.
        while queue and len(visited) <= 128:
            current = queue.popleft()
            if current in candidates and current != start:
                anchor = current
                _anchor_index, append_index = candidates[current]
                break
            for direction, destination, is_warp in world.neighbors(current):
                if is_warp or destination in visited:
                    continue
                if destination[0] != world_observation.map_id:
                    continue
                if destination[1] in blocked_coordinates or destination in trainer_hazards:
                    continue
                visited.add(destination)
                came_from[destination] = (current, direction)
                queue.append(destination)

        if anchor is None or append_index is None:
            return None

        detour: list[NavigationAction] = []
        current = anchor
        while current != start:
            previous, direction = came_from[current]
            detour.append(
                NavigationAction(
                    NavigationActionType.MOVE,
                    direction,
                    previous,
                    current,
                    run=blocked_action.run,
                )
            )
            current = previous
        detour.reverse()
        repaired = tuple(detour) + remaining[append_index:]
        if not repaired or repaired[0].source != start:
            return None
        self._report(
            "LOCAL_DETOUR: reused cached suffix "
            f"blocked={blocked_action.destination!r} anchor={anchor!r} "
            f"detour_actions={len(detour)} suffix_index={append_index}"
        )
        return repaired

    def _retain_local_blocks_for_map(self, map_id) -> None:
        """Discard collision inferences from maps no longer being traversed."""

        self._locally_blocked_destinations = {
            location for location in self._locally_blocked_destinations if location[0] == map_id
        }

    def _augment_local_blocked_destinations(self, observation: AgentObservation) -> AgentObservation:
        """Expose collision evidence to the next route search."""

        world = observation.overworld
        if world is None:
            return observation
        self._retain_local_blocks_for_map(world.map_id)
        local_coordinates = {
            location[1] for location in self._locally_blocked_destinations if location[0] == world.map_id
        }
        if not local_coordinates:
            return observation
        dynamic_blocked = frozenset(world.dynamic_blocked_coordinates) | local_coordinates
        if dynamic_blocked == world.dynamic_blocked_coordinates:
            return observation
        return replace(observation, overworld=replace(world, dynamic_blocked_coordinates=dynamic_blocked))

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
        # Treat occupied object tiles as hard collision boundaries. Nearby
        # tiles only need a checkpoint when the object is a live trainer: a
        # generic wandering NPC is already checked immediately before each
        # cached input, and making every adjacent object a checkpoint turns a
        # long route into repeated one- or two-step replans.
        occupied_object_locations = {obj.location for obj in world.objects if obj.location[0] == world.map_id}
        for obj in world.objects:
            if obj.trainer_type in (None, "None") or obj.trainer_defeated:
                continue
            x, y = obj.location[1]
            checkpoints.update(
                {
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
            # A directional input while facing another way is consumed by
            # Emerald as a turn.  The full control-loop path tracks that
            # turn-only transaction and retries the movement after the avatar
            # is standing again.  Do not put such an action into a cached
            # movement batch: the batch has no observation boundary between
            # the turn and the subsequent move and would eventually infer a
            # false obstacle, causing the next replan to choose an unnecessary
            # detour (for example, stepping south from Route 101's (10,14)).
            if not result and world.facing is not None and action.direction is not world.facing:
                break
            # Keep cached segments straight.  Direction changes are handled by
            # the normal movement transaction on the next observation, where
            # the game can acknowledge the turn before the next step.
            if result and action.direction is not result[-1].direction:
                break
            # A checkpoint is normally included in the batch so the ordinary
            # path can process the interaction boundary on the next frame.
            # An occupied object tile is different: sending the final input
            # toward it is itself unsafe, especially when a trainer has just
            # moved or returned from battle.  Stop before that action and let
            # the normal observation/action gate replan around the object.
            if action.destination in occupied_object_locations:
                break
            result.append(action)
            if action.destination in checkpoints:
                break
        return tuple(result)

    @staticmethod
    def _live_object_coordinates(map_id) -> frozenset[tuple[int, int]]:
        """Read current same-map object occupancy for the movement fast path.

        Cached movement deliberately skips full overworld perception, but a
        runtime NPC can move onto a future route tile between observations.
        ``get_map_objects`` is the small raw ``gObjectEvents`` read; unlike
        ``get_runtime_object_table`` it does not resolve templates or event
        flags, so it is suitable for this final collision check.
        """

        return frozenset(
            object_event.current_coords
            for object_event in get_map_objects()
            if object_event.map_group_and_number == map_id and "isPlayer" not in object_event.flags
        )

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
        if self._in_flight_move is not None:
            _gate_observer_call(
                "discard_navigation",
                self._in_flight_move,
                source_kind="movement",
                reason=reason,
            )
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
        self._invalidate_after_battle_return()
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
            if action.destination in self._locally_blocked_destinations:
                count("cached_route_fast_path_fallback_frames")
                self._cancel_movement_batch("local_destination_blocked")
                return False
            if location[0] != action.source[0]:
                count("cached_route_fast_path_fallback_frames")
                self._cancel_movement_batch("map_transition")
                return False
            if location == action.destination:
                _gate_observer_call(
                    "record_navigation_outcome",
                    action,
                    "open",
                    observation_frame=_gate_observation_frame(),
                    source_kind="movement",
                )
                self._locally_blocked_destinations.discard(action.destination)
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
                if _is_intermediate_move_location(location, action):
                    count("cached_route_intermediate_progress")
                    if trace is not None:
                        trace.mark(
                            "route_invalidation_detail",
                            f"movement_intermediate_progress current={location!r} "
                            f"expected_destination={action.destination!r} "
                            f"direction={action.direction.name!r}",
                        )
                else:
                    count("cached_route_fast_path_fallback_frames")
                    self._cancel_movement_batch("position_divergence")
                    return False

            # The cached batch has no full observation boundary between its
            # actions.  Re-read the compact runtime object table immediately
            # before dispatching each held direction so a trainer/NPC that
            # moved after the route was composed cannot receive movement
            # input.  Invalidate the shared observation before falling back so
            # the next normal step plans against the newly observed blocker.
            live_object_coordinates = self._live_object_coordinates(location[0])
            if action.destination[1] in live_object_coordinates:
                count("cached_route_fast_path_fallback_frames")
                self._report(
                    "MOVE_BATCH_BLOCKED_LIVE_OBJECT "
                    f"map={location[0]!r} source={action.source!r} "
                    f"destination={action.destination!r} "
                    f"occupied={tuple(sorted(live_object_coordinates))!r}"
                )
                state_cache.invalidate_runtime_observations()
                self._cancel_movement_batch("live_object_blocked")
                return False

            batch.frames_waiting += 1
            if batch.frames_waiting > 24:
                count("cached_route_fast_path_fallback_frames")
                self._movement_failure_count += 1
                _gate_observer_call(
                    "record_navigation_outcome",
                    action,
                    "blocked",
                    observation_frame=_gate_observation_frame(),
                    source_kind="movement",
                )
                self._remember_blocked_destination(action, "cached movement made no progress")
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
                    _gate_observer_call(
                        "record_navigation_issued",
                        action,
                        action_frame=_gate_observation_frame(),
                        source_kind="movement",
                    )
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
                    _gate_observer_call(
                        "record_navigation_issued",
                        action,
                        action_frame=_gate_observation_frame(),
                        source_kind="movement",
                    )
                    hold_button(action.direction.button_name)
                    if action.run:
                        hold_button("B")
                else:
                    press_direction = getattr(context.emulator, "press_direction", None)
                    if callable(press_direction):
                        _gate_observer_call(
                            "record_navigation_issued",
                            action,
                            action_frame=_gate_observation_frame(),
                            source_kind="movement",
                        )
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
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
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

    def _poll_background_navigation(self, observation: AgentObservation):
        """Return a completed world plan without blocking the control loop."""

        # A campaign loop can be constructed one frame before the mode switch
        # is committed to BotContext. Recheck at use time so that boundary
        # does not silently fall back to a synchronous multi-map search.
        if not self._background_navigation_enabled and context.bot_mode == "Campaign Progression":
            self._background_navigation_enabled = True
        if not self._background_navigation_enabled:
            return _NAVIGATION_PLAN_UNSET
        if observation.overworld is None or observation.interaction_type is not InteractionType.OVERWORLD:
            return _NAVIGATION_PLAN_UNSET
        if self._goal is None:
            return _NAVIGATION_PLAN_UNSET
        world = NavigationWorld.from_overworld(observation.overworld)
        start = (observation.overworld.map_id, observation.overworld.player_coordinates)
        planning_goal, _ = _effective_navigation_goal(observation, world)
        # Do not include runtime NPC coordinates in the single-flight key.
        # Emerald updates wandering object positions every frame; using the
        # full world signature here cancelled the worker on every such update,
        # so a long encounter or trainer-safe route could remain permanently
        # pending. The worker still receives the complete immutable snapshot
        # for the search, and the resulting plan is validated against the
        # current world before an input is emitted. Static topology and
        # dynamic occupancy are therefore handled at the two appropriate
        # boundaries: plan identity here, safety validation at execution.
        # Only stable request identity belongs in the worker key. Dynamic
        # blocked coordinates are an execution-time safety input: a wandering
        # object can change them while a multi-second search is running, but
        # that does not make the route's first action invalid. The cached
        # route validator and the final destination check handle that live
        # occupancy boundary. Transition topology remains in the key because
        # a changed warp/connection can change the global route itself.
        key = _navigation_request_key(start, planning_goal, world)
        dynamic_blocked_revision, _transition_revision = world.revision_key()
        failure_key = (key, dynamic_blocked_revision, _navigation_failure_revision(world))
        def operation(cancel_check):
            began = monotonic()
            try:
                return plan_with_world_navigation(
                    world,
                    start,
                    planning_goal,
                    caller=f"agent_control.background_navigation[{self._navigation_owner_label}]",
                    cancel_check=cancel_check,
                )
            except (NavigationError, WorldNavigationError, NavigationSearchCancelled) as error:
                elapsed_ms = (monotonic() - began) * 1000.0
                # Unconditional like the pathfinding panels (and written to the
                # debug log, not only stdout): this explains a visible
                # navigation pause without requiring --debug-trace.
                print_always(
                    lambda: (
                        "NAVIGATION_SEARCH_FAILURE: "
                        f"error_type={type(error).__name__!r} elapsed_ms={elapsed_ms:.0f} "
                        f"{_navigation_world_summary(world, start, planning_goal)}"
                    ),
                )
                _dump_slow_navigation_search(world, start, planning_goal, elapsed_ms)
                raise

        result = _background_navigation_planner.poll(
            key,
            operation,
            failure_key=failure_key,
            owner=self._navigation_owner,
            owner_label=self._navigation_owner_label,
        )
        planner_event = _background_navigation_planner.take_event()
        if planner_event is not None and self._diagnostics_enabled():
            self._report(f"NAVIGATION_WORKER event={planner_event!r}")
        discarded_reason = _background_navigation_planner.take_discard_reason()
        expected_start_discard = discarded_reason == "request_key_changed changed=('start',)"
        if discarded_reason is not None and not expected_start_discard:
            self._report(f"NAVIGATION_PLAN_DISCARDED reason={discarded_reason!r}")
        if result is _NAVIGATION_PLAN_PENDING:
            report_key = ("pending", repr(key), None if expected_start_discard else discarded_reason)
            if report_key != self._last_navigation_poll_report:
                self._last_navigation_poll_report = report_key
                self._report("NAVIGATION_PLAN_PENDING worker='single_flight'")
        else:
            plan = result[0]
            error = result[2]
            if plan is None and error is None:
                error = NavigationError("navigation planner returned no plan or error")
                result = (None, None, error)
            report_key = (
                "ready",
                repr(key),
                len(plan.actions) if plan is not None else None,
                type(error).__name__ if error is not None else None,
                None if expected_start_discard else discarded_reason,
            )
            if report_key != self._last_navigation_poll_report:
                self._last_navigation_poll_report = report_key
                self._report(
                    "NAVIGATION_PLAN_READY "
                    f"actions={len(plan.actions) if plan is not None else None} "
                    f"error={type(error).__name__ if error is not None else None!r}"
                )
        return result

    def _automatic_trainer_wait_key(self, observation: AgentObservation, decision: ActionDecision) -> tuple | None:
        """Return a stable key for an unchanged automatic trainer wait."""

        evaluation = decision.goal_evaluation
        target = _goal_target(self._goal)
        engagement = getattr(evaluation, "engagement_plan", None)
        if not isinstance(target, EngageTrainer) or engagement is None:
            return None
        if engagement.mode not in (TrainerEngagementMode.SIGHTLINE, TrainerEngagementMode.UNKNOWN):
            return None
        if decision.action.action_type is not AgentActionType.WAIT_REOBSERVE:
            return None
        plan = getattr(evaluation, "plan", None)
        if plan is None or getattr(plan, "actions", ()):
            return None
        world = observation.overworld
        if world is None:
            return None
        return (
            target.trainer_id,
            (world.map_id, world.player_coordinates),
            engagement.mode,
            engagement.trigger_id,
        )

    def _apply_automatic_trainer_wait_watchdog(
        self,
        observation: AgentObservation,
        decision: ActionDecision,
    ) -> tuple[ActionDecision, ActionResult] | None:
        """Force one fresh trainer observation, then fail boundedly.

        A valid sight-line wait is normally short-lived: the ROM starts the
        battle or changes the interaction state. If the same empty route is
        returned for a full bounded window, invalidate runtime perception once
        and retry. A second unchanged window raises a normal navigation error;
        CampaignController handles that as a retryable autonomous boundary.
        """

        key = self._automatic_trainer_wait_key(observation, decision)
        if key is None:
            self._automatic_trainer_wait_signature = None
            self._automatic_trainer_wait_observations = 0
            self._automatic_trainer_wait_reobserved = False
            return None
        if key == self._automatic_trainer_wait_signature:
            self._automatic_trainer_wait_observations += 1
        else:
            self._automatic_trainer_wait_signature = key
            self._automatic_trainer_wait_observations = 1
            self._automatic_trainer_wait_reobserved = False
        if self._automatic_trainer_wait_observations < self._automatic_trainer_wait_limit:
            return None

        reason = "automatic_trainer_wait_watchdog"
        if self._automatic_trainer_wait_reobserved:
            raise NavigationError(
                f"trainer engagement remained unchanged after {self._automatic_trainer_wait_limit} re-observations"
            )
        self._automatic_trainer_wait_reobserved = True
        self._automatic_trainer_wait_observations = 0
        self._invalidate_plan(reason, observation=observation)
        state_cache.invalidate_runtime_observations()
        self._report(
            f"WATCHDOG: {reason} trainer={key[0]!r} location={key[1]!r} "
            f"action='invalidate_and_reobserve'"
        )
        wait_action = AgentAction(AgentActionType.WAIT_REOBSERVE, reason=reason)
        wait_decision = ActionDecision(wait_action, decision.goal_evaluation)
        return wait_decision, self._executor.execute(wait_action, observation)

    @staticmethod
    def _release_transition_input(pending: _PendingTransition) -> None:
        """Release only the direction sustained by a terminal transition."""
        release_button = getattr(context.emulator, "release_button", None)
        if callable(release_button):
            release_button(pending.direction.button_name)

    def dispose(self) -> None:
        """Release controller-owned input and cancel its background search."""
        _background_navigation_planner.cancel(
            self._navigation_owner,
            reason="navigation owner disposed",
            owner_label=self._navigation_owner_label,
        )
        pending = self._pending_transition
        if pending is None:
            return
        # Clear ownership before calling the adapter so disposal is
        # idempotent even if adapter cleanup re-enters controller teardown.
        self._pending_transition = None
        self._expected_world_transition = None
        _gate_observer_call(
            "discard_navigation",
            pending.action,
            source_kind="transition",
            reason="controller disposed before transition outcome",
        )
        self._release_transition_input(pending)

    def _terminal_navigation_fallback_limit(self) -> int:
        """Return the bounded retry count for this controller's goal.

        A preferred campaign corridor carries enough structure to retry the
        same controller after a stale first action: its locally learned
        blocker set remains available to the next search. Other goals retain
        the original single delayed retry so an actually unreachable goal
        cannot spin indefinitely.
        """

        target = _goal_target(self._goal)
        if (
            isinstance(self._goal, NavigationGoal)
            and isinstance(target, SemanticTarget)
            and target.kind.name == "MAP"
            and self._goal.preferred_map_route
        ):
            return 2
        return 1

    def _wait_for_terminal_navigation_fallback(self, reason: str):
        """Wait, then allow a bounded route observation to be retried."""

        if self._terminal_navigation_fallback_attempts >= self._terminal_navigation_fallback_limit():
            return False
        self._terminal_navigation_fallback_attempts += 1
        self._terminal_navigation_fallback_used = True
        self._report(
            "NAVIGATION_FALLBACK: "
            f"attempt={self._terminal_navigation_fallback_attempts} "
            f"limit={self._terminal_navigation_fallback_limit()} "
            f"waiting_frames={self._terminal_navigation_fallback_wait_frames} reason={reason!r}"
        )
        # Release all controller-owned input before yielding the delay. This
        # prevents a held direction from continuing to press into the stale
        # state while the ROM's object/script tables settle.
        reset_navigation_held_inputs = getattr(self._executor, "reset_navigation_held_inputs", None)
        if callable(reset_navigation_held_inputs):
            reset_navigation_held_inputs()
        else:
            reset_held_buttons = getattr(getattr(context, "emulator", None), "reset_held_buttons", None)
            if callable(reset_held_buttons):
                reset_held_buttons()
        self._invalidate_plan("terminal_navigation_fallback")
        state_cache.invalidate_runtime_observations()
        for _ in range(self._terminal_navigation_fallback_wait_frames):
            yield
        self._report("NAVIGATION_FALLBACK: retrying after delayed re-observation")
        return True

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
                    trigger.hazard_locations,
                    trigger.hazard_kind,
                    getattr(trigger, "engagement_locations", ()),
                )
                for trigger in world.triggers
            ),
            world.dynamic_blocked_coordinates,
            getattr(world, "static_blocked_coordinates", ()),
            tuple((obj.local_id, obj.location, obj.script) for obj in world.objects),
            tuple(
                (
                    binding.binding.trigger_id,
                    binding.static_location,
                    binding.static_availability,
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
        missing = object()
        changed = {
            index
            for index in range(max(len(previous), len(current)))
            if (previous[index] if index < len(previous) else missing)
            != (current[index] if index < len(current) else missing)
        }
        route_locations = {location for action in remaining_actions for location in (action.source, action.destination)}
        next_action_locations = (
            {
                location
                for location in (remaining_actions[0].source, remaining_actions[0].destination)
                if location is not None
            }
            if remaining_actions
            else set()
        )
        route_coordinates = {
            location[1]
            for location in route_locations
            if isinstance(location, tuple)
            and len(location) == 2
            and location[0] == current[0]
        }
        names = {
            0: "map",
            1: "warps",
            2: "triggers",
            3: "dynamic_blocked",
            4: "static_blocked",
            5: "objects",
            6: "bindings",
        }
        changed_names = tuple(names.get(index, f"field_{index}") for index in sorted(changed))
        detail = f"changed={changed_names!r} route_locations={len(route_locations)}"
        if not changed:
            return False, detail
        # Unknown fields, map topology, static blockers, and bindings can all
        # alter the route without a location-level comparison. Invalidate
        # conservatively when any of them changes. The runtime fields below
        # are the only fields eligible for a remaining-route comparison.
        if changed - {2, 3, 5}:
            return True, detail + " relevant=static_or_topology"

        if 3 in changed:
            # A moving NPC can leave the next destination while entering a
            # later tile.  Only newly occupied coordinates can make the next
            # input unsafe; a vacated coordinate is positive evidence that
            # the cached edge is available again.
            changed_blocked = set(current[3]) - set(previous[3])
            next_action_coordinates = {
                location[1]
                for location in next_action_locations
                if isinstance(location, tuple)
                and len(location) == 2
                and location[0] == current[0]
            }
            if changed_blocked & next_action_coordinates:
                return True, detail + " relevant=immediate_blocked_route_tile"
        if 5 in changed:
            old_objects = {item[1] for item in previous[5]}
            new_objects = {item[1] for item in current[5]}
            if (new_objects - old_objects) & next_action_locations:
                return True, detail + " relevant=immediate_route_object"
        if 2 in changed:
            old_triggers = {item[0]: item for item in previous[2]}
            new_triggers = {item[0]: item for item in current[2]}

            def route_relevant_locations(trigger) -> set:
                if trigger is None:
                    return set()
                # Hazard locations are part of the trigger signature because
                # a trainer can turn or become defeated without changing its
                # activation geometry. Keep the lenient fallback for older
                # injected signatures used by lightweight callers.
                return (
                    set(trigger[1])
                    | set(trigger[2])
                    | (set(trigger[7]) if len(trigger) > 7 else set())
                    | (set(trigger[9]) if len(trigger) > 9 else set())
                )

            def is_nontrainer_runtime_object(trigger) -> bool:
                """Return whether a moved generic NPC is the changed trigger.

                Runtime objects publish interaction triggers as well as
                occupancy.  A wandering non-trainer therefore changes both
                the object and trigger portions of the signature while it
                walks.  Occupancy is already checked immediately before each
                cached input; treating its future interaction position as a
                route-wide hazard recreates the background-search stall.
                Trainer geometry remains conservative below.
                """

                if trigger is None or not str(trigger[0]).startswith("object:"):
                    return False
                hazard_locations = set(trigger[7]) if len(trigger) > 7 else set()
                engagement_locations = set(trigger[9]) if len(trigger) > 9 else set()
                hazard_kind = trigger[8] if len(trigger) > 8 else None
                return not hazard_locations and not engagement_locations and hazard_kind != "trainer"

            for trigger_id in old_triggers.keys() | new_triggers.keys():
                old_trigger = old_triggers.get(trigger_id)
                new_trigger = new_triggers.get(trigger_id)
                if old_trigger == new_trigger:
                    continue
                if all(
                    trigger is None or is_nontrainer_runtime_object(trigger)
                    for trigger in (old_trigger, new_trigger)
                ):
                    continue
                for trigger in (old_triggers.get(trigger_id), new_triggers.get(trigger_id)):
                    if route_relevant_locations(trigger) & route_locations:
                        return True, detail + " relevant=route_trigger"
        return False, detail + " relevant=none"

    def _invalidate_plan(
        self,
        reason: str = "unspecified",
        *,
        observation: AgentObservation | None = None,
        detail: str | None = None,
    ) -> None:
        route_plan = self._route_plan
        cached_action = (
            self._cached_actions[self._cached_action_index]
            if self._cached_action_index < len(self._cached_actions)
            else None
        )
        current = self._last_observed_location
        if observation is not None and observation.overworld is not None:
            current = (observation.overworld.map_id, observation.overworld.player_coordinates)
        if route_plan is not None or self._cached_actions:
            print_navigation_invalidation(
                reason=reason,
                route_id=route_plan.route_id if route_plan is not None else None,
                current=current,
                goal=self._cached_goal if self._cached_goal is not None else self._goal,
                action_index=self._cached_action_index,
                remaining_actions=max(0, len(self._cached_actions) - self._cached_action_index),
                total_actions=len(self._cached_actions),
                cached_action=cached_action,
                detail=detail,
            )
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
        reset_navigation_held_inputs = getattr(self._executor, "reset_navigation_held_inputs", None)
        if callable(reset_navigation_held_inputs):
            reset_navigation_held_inputs()

    def _invalidate_after_battle_return(self) -> bool:
        """Invalidate a route mounted before the battle listener took over."""

        if self._battle_end_generation_seen == _battle_end_generation:
            return False
        self._battle_end_generation_seen = _battle_end_generation
        self._post_battle_movement_pending = True
        self._post_battle_movement_location = None
        self._post_battle_movement_observations = 0
        self._report("BATTLE_RETURN: invalidating suspended tactical route")
        self._invalidate_plan("battle_return")
        return True

    def _cached_decision(self, observation: AgentObservation) -> ActionDecision | None:
        trace = getattr(context, "stutter_trace", None)
        span = trace.span("cached_route_validation") if trace is not None else nullcontext()
        with span:
            return self._cached_decision_instrumented(observation)

    def _cached_decision_instrumented(self, observation: AgentObservation) -> ActionDecision | None:
        if self._cached_evaluation is None or self._cached_goal != self._goal:
            return None
        if self._cached_evaluation.status is GoalStatus.COMPLETE and not self.navigation_transition_active:
            # Preserve the authoritative terminal evaluation across one more
            # controller observation.  This keeps a completed route from
            # being mistaken for a fresh navigation request when a caller
            # polls the generator again before unmounting the loop.
            return ActionDecision(
                AgentAction(AgentActionType.WAIT_REOBSERVE, reason="goal already complete"),
                self._cached_evaluation,
            )

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

        # A map-load callback can temporarily provide no overworld model at
        # all.  Navigation validation has no location or world signature to
        # inspect in that frame; let the interaction-state selector emit its
        # bounded wait while preserving the cached route for the next coherent
        # destination observation.
        if observation.overworld is None:
            return None

        count("cached_plan_validations")
        count("cached_plan_hits")

        location = (observation.overworld.map_id, observation.overworld.player_coordinates)
        cached_action = (
            self._cached_actions[self._cached_action_index]
            if self._cached_action_index < len(self._cached_actions)
            else None
        )
        blocked_action = self._blocked_action(observation, cached_action)
        if blocked_action is not None:
            self._report(
                "REPLAN: reason='cached route destination blocked' "
                f"source={blocked_action.source!r} destination={blocked_action.destination!r}"
            )
            self._reject_first_action(blocked_action, "cached route destination blocked", observation)
            local_detour = self._local_detour_around_blocked_action(observation, blocked_action)
            if local_detour is not None:
                previous_plan = self._cached_evaluation.plan
                repaired_plan = NavigationPlan(
                    local_detour,
                    previous_plan.destination if previous_plan is not None else local_detour[-1].destination,
                    previous_plan.metrics if previous_plan is not None else None,
                    previous_plan.candidate_metrics if previous_plan is not None else (),
                    previous_plan.forced_trainer_exposure if previous_plan is not None else False,
                )
                self._cached_evaluation = replace(self._cached_evaluation, plan=repaired_plan)
                self._cached_actions = local_detour
                self._cached_action_index = 0
                self._route_plan = RoutePlan.from_navigation_plan(repaired_plan, target=self._goal)
                self._cached_world_signature = self._world_signature(observation)
                self._movement_batch = None
                self._in_flight_move = None
                self._in_flight_move_initial_facing = None
                count("navigation_local_detours")
                return ActionDecision(
                    AgentAction(
                        AgentActionType.NAVIGATE_TOWARD_GOAL,
                        direction=local_detour[0].direction,
                        navigation=local_detour[0],
                        reason="local detour around current route obstacle",
                    ),
                    self._cached_evaluation,
                )
            self._invalidate_plan("cached_route_destination_blocked")
            return None
        current_world_signature = self._world_signature(observation)
        world_signature_changed = self._cached_world_signature != current_world_signature
        route_complete_at_destination = (
            self._cached_action_index >= len(self._cached_actions)
            and self._cached_evaluation.plan is not None
            and location == self._cached_evaluation.plan.destination
        )
        if world_signature_changed and self._cached_world_signature is not None and route_complete_at_destination:
            # The route has already reached its terminal location. Runtime
            # objects, triggers, and dynamic blockers can legitimately change
            # while the ROM starts the destination interaction. There is no
            # remaining movement edge for those changes to invalidate; keep
            # the terminal evaluation alive and let the goal-specific
            # completion/interaction branch below inspect the fresh world.
            self._cached_world_signature = current_world_signature
            self._report(
                "ROUTE_SIGNATURE_UPDATED "
                f"reason='completed route at destination' location={location!r}"
            )
            world_signature_changed = False
        if world_signature_changed and cached_action is not None:
            # A newly loaded trainer can make a later edge unsafe before it
            # becomes the current action.  Compare changed trigger/object
            # geometry with the remaining route and discard the cached plan
            # when the change can affect that route.  The explicit
            # ``_blocked_action`` check above remains the final occupancy and
            # trainer-hazard gate before every input.
            affects_route, detail = self._signature_change_affects_remaining_route(
                self._cached_world_signature,
                current_world_signature,
                self._cached_actions[self._cached_action_index :],
            )
            if affects_route:
                self._invalidate_plan("cached_route_world_changed", observation=observation, detail=detail)
                return None
            self._cached_world_signature = current_world_signature
            self._report(
                "ROUTE_SIGNATURE_UPDATED "
                "reason='next-action-only validation' "
                f"location={location!r} next_destination={cached_action.destination!r}"
            )

        if self._cached_action_index >= len(self._cached_actions):
            if self._cached_evaluation.plan is not None and location == self._cached_evaluation.plan.destination:
                cached_target = _goal_target(self._goal)
                cached_engagement = self._cached_evaluation.engagement_plan
                if isinstance(cached_target, EngageTrainer) and cached_engagement is not None:
                    if cached_engagement.mode is TrainerEngagementMode.UNKNOWN:
                        # A cross-map trainer route may be planned before the
                        # destination map's object table or hazard geometry is
                        # visible.  Its destination is then only a map-entry
                        # checkpoint, not an executable engagement boundary.
                        # Reusing the empty geometry forever strands the loop
                        # at that boundary, so let fresh perception resolve
                        # the trainer before deciding whether to wait or move.
                        self._invalidate_plan("trainer_engagement_geometry_requires_reobserve")
                        return None
                    if (
                        cached_engagement.mode is TrainerEngagementMode.SIGHTLINE
                        and location not in cached_engagement.engagement_locations
                    ):
                        # The cached route can reach a map boundary while the
                        # sight line was resolved from an earlier map scope.
                        # Do not treat that boundary as an engagement point.
                        self._invalidate_plan("cached_trainer_sightline_not_reached")
                        return None
                    if cached_engagement.mode is TrainerEngagementMode.SIGHTLINE:
                        return ActionDecision(
                            AgentAction(
                                AgentActionType.WAIT_REOBSERVE,
                                reason="waiting for trainer sight-line activation",
                            ),
                            self._cached_evaluation,
                        )
                    cached_target = cached_engagement.interaction_target
                if isinstance(cached_target, (ActivateTrigger, ReachInteractionPosition)):
                    world = NavigationWorld.from_overworld(observation.overworld)
                    if not GoalAwareNavigator(world).satisfies(
                        location, observation.overworld.facing, cached_target
                    ):
                        # A plan may have reached the source tile without the
                        # final turn being accepted by the emulator. Never
                        # issue A on geometric adjacency alone.
                        self._invalidate_plan("interaction_precondition_not_ready")
                        return None
                    if isinstance(cached_target, ReachInteractionPosition) and _trigger_requires_input(
                        observation, cached_target.trigger_id
                    ):
                        return ActionDecision(
                            AgentAction(
                                AgentActionType.INTERACT,
                                option=cached_target.trigger_id,
                                reason="cached interaction source state reached",
                            ),
                            self._cached_evaluation,
                        )
                    if isinstance(cached_target, ReachInteractionPosition):
                        if not _trigger_requires_input(observation, cached_target.trigger_id):
                            self._interaction_position_ready = True
                        return ActionDecision(
                            AgentAction(
                                AgentActionType.WAIT_REOBSERVE,
                                reason="automatic coordinate trigger fires on tile entry",
                            ),
                            self._cached_evaluation,
                        )
                    if not _trigger_requires_input(observation, cached_target.trigger_id):
                        self._interaction_position_ready = True
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
                            option=cached_target.trigger_id,
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
        self._invalidate_after_battle_return()
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
        observation = self._augment_local_blocked_destinations(observation)

        interaction_type = observation.interaction_type
        observation = self._resolve_stale_post_battle_movement(observation)
        observation = self._resolve_stale_interaction_movement(observation)
        emulator_frame = (
            getattr(emulator, "get_frame_count", None) if (emulator := getattr(context, "emulator", None)) else None
        )
        try:
            live_frame = emulator_frame() if callable(emulator_frame) else None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            live_frame = None
        if (
            getattr(context, "debug", False)
            and isinstance(live_frame, int)
            and live_frame % 60 == 0
            and interaction_type is not InteractionType.OVERWORLD
        ):
            script_context = get_global_script_context()
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_LIVE_INTERACTION: "
                    f"frame={live_frame!r} type={interaction_type.name!r} "
                    f"map={getattr(observation.overworld, 'map_id', None)!r} "
                    f"controllable={observation.interaction.controllable!r} "
                    f"phase={observation.interaction.interaction_phase.name!r} "
                    f"script={observation.interaction.script_function!r} "
                    f"native={observation.interaction.native_function!r} "
                    f"stack={getattr(script_context, 'stack', None)!r} "
                    f"dialogue_waiting={observation.interaction.dialogue_waiting!r} "
                    f"render_rescue={observation.interaction.field_message_render_rescue_available!r} "
                    f"task_state={observation.interaction.metadata.get('field_message_task_state')!r} "
                    f"printer={observation.interaction.metadata.get('text_printer_active')!r}/"
                    f"{observation.interaction.metadata.get('text_printer_state')!r} "
                    f"input_in_flight={self._dialogue_input_in_flight!r}"
                )
            )
        if (
            self._interaction_position_ready
            and interaction_type is not InteractionType.OVERWORLD
            and observation.interaction.script_active
        ):
            # A coordinate/object script has claimed the interaction after
            # the positioning goal was reached. Return ownership to the
            # caller exactly as we do for an interaction started by our own
            # A input; otherwise the positioning loop remains mounted while
            # the nurse/clerk script owns the field.
            self._interaction_position_ready = False
            self._invalidate_plan("automatic interaction script started")
            wait_action = AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="automatic interaction script owns the interaction",
            )
            wait_decision = ActionDecision(wait_action)
            return (
                observation,
                wait_decision,
                ActionResult(
                    ActionResultType.GOAL_COMPLETE,
                    wait_action,
                    "automatic interaction script owns the interaction",
                ),
            )
        if (
            interaction_type is not InteractionType.OVERWORLD
            and observation.interaction.script_active
            and self._cached_evaluation is not None
            and self._cached_evaluation.plan is not None
            and self._cached_action_index >= len(self._cached_actions)
            and _interaction_trigger_id(self._goal, self._cached_evaluation) is not None
        ):
            # A cached movement batch can arrive on an automatic trigger tile
            # on the same emulator frame that the ROM installs its script.
            # There is no intervening overworld observation in which
            # ``_cached_decision`` can set ``_interaction_position_ready``.
            # The exhausted route plus the active script is nevertheless an
            # authoritative handoff: return ownership to the capability that
            # will consume the script's dialogue.
            self._invalidate_plan("automatic interaction script started after route completion")
            wait_action = AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="automatic interaction script owns the interaction after route completion",
            )
            self._report(
                "INTERACTION_HANDOFF: automatic script started after cached route completion " f"goal={self._goal!r}"
            )
            return (
                observation,
                ActionDecision(wait_action),
                ActionResult(
                    ActionResultType.GOAL_COMPLETE,
                    wait_action,
                    "automatic interaction script owns the interaction after route completion",
                ),
            )
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
            elif observation.interaction.field_message_render_rescue_available:
                # A scripted scene can start another message while the
                # previous A handoff is still marked in flight.  The render
                # boundary is actionable in its own right: allow the
                # dialogue selector to issue its bounded fresh-B rescue
                # rather than waiting for the old input transition forever.
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
            if self.navigation_transition_active:
                # A warp can hand control to the ROM on the source tile before
                # the destination map is observable.  This is a transition
                # boundary, not evidence that the exhausted route failed.
                # Keep the pending transition/route owner alive until its
                # destination checkpoint has been observed.
                self._report(
                    "TRANSITION: preserving route during non-overworld boundary "
                    f"interaction={interaction_type.name!r}"
                )
            else:
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
            and _interaction_trigger_id(self._goal, self._cached_evaluation) == self._started_interaction_id
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
            and _interaction_trigger_id(self._goal, self._cached_evaluation) == self._started_interaction_id
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
            and _interaction_trigger_id(self._goal, self._cached_evaluation) == self._started_interaction_id
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
                    # Map connections can land at a valid destination tile
                    # chosen by the ROM rather than the first predicted
                    # boundary pair. The map change is still explicit proof
                    # that the candidate edge opened; retain the old route
                    # mismatch diagnostic for controller recovery.
                    if pending.transition_kind == "map_connection":
                        _gate_observer_call(
                            "record_navigation_outcome",
                            pending.action,
                            "open",
                            observation_frame=_gate_observation_frame(),
                            source_kind="transition",
                        )
                    else:
                        _gate_observer_call(
                            "discard_navigation",
                            pending.action,
                            source_kind="transition",
                            reason="transition destination coordinate mismatch",
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
                _gate_observer_call(
                    "record_navigation_outcome",
                    pending.action,
                    "open",
                    observation_frame=_gate_observation_frame(),
                    source_kind="transition",
                )
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
                _gate_observer_call(
                    "discard_navigation",
                    pending.action,
                    source_kind="transition",
                    reason="transition destination mismatch",
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
            elif interaction_type is not InteractionType.OVERWORLD:
                # The ROM owns the field between warp activation and the
                # destination-map observation.  Returning a waiting action
                # keeps AgentControlLoop mounted, which lets CampaignController
                # preserve ownership and prevents the campaign capability from
                # treating this boundary as a completed or failed route.
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="transition script owns field; waiting for destination",
                )
                wait_decision = ActionDecision(wait_action)
                wait_result = self._executor.execute(wait_action, observation)
                return observation, wait_decision, wait_result
            elif interaction_type is InteractionType.OVERWORLD:
                if not observation.overworld.controllable:
                    # The ROM owns the avatar while a door/warp task is
                    # finishing. Releasing the transition direction here is
                    # essential: the next observation may still expose the
                    # source map, and holding the old input can drive the
                    # player back and forth at the Center exit.
                    pending.observations = 0
                    self._release_transition_input(pending)
                    self._report(
                        "TRANSITION: input_released_for_rom_control "
                        f"source={pending.source!r} destination={pending.destination!r}"
                    )
                    wait_action = AgentAction(
                        AgentActionType.WAIT_REOBSERVE,
                        reason="transition script owns avatar; waiting for control",
                    )
                    wait_decision = ActionDecision(wait_action)
                    wait_result = self._executor.execute(wait_action, observation)
                    return observation, wait_decision, wait_result
                if pending.last_position is not None and observed != pending.last_position:
                    pending.moved = True
                    pending.observations = 0
                    self._report("TRANSITION: player_moved_while_pending")
                elif observation.overworld.transition_in_progress:
                    # A directional/door warp can remain on its source map
                    # while the ROM is running its map-load task.  That is
                    # progress, not a failed input.  Count the watchdog only
                    # during stable observations with no transition signal;
                    # otherwise a legitimate long map load expires just
                    # before the destination becomes observable. Once the
                    # ROM has installed that task, it owns the transition;
                    # continuing to hold the activation direction can race
                    # the map-load boundary and leaves the bot appearing to
                    # issue no useful commands while the task is pending.
                    pending.observations = 0
                    self._release_transition_input(pending)
                    self._report(
                        "TRANSITION: input_released_for_rom_transition_task "
                        f"signals={observation.overworld.transition_signals!r}"
                    )
                    wait_action = AgentAction(
                        AgentActionType.WAIT_REOBSERVE,
                        reason="ROM transition task owns field; waiting for destination",
                    )
                    wait_decision = ActionDecision(wait_action)
                    wait_result = self._executor.execute(wait_action, observation)
                    return observation, wait_decision, wait_result
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
                    _gate_observer_call(
                        "discard_navigation",
                        pending.action,
                        source_kind="transition",
                        reason=reason,
                    )
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

        if self._warp_settling and observation.overworld is not None:
            if interaction_type is not InteractionType.OVERWORLD:
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="waiting for post-warp script to release field",
                )
                wait_decision = ActionDecision(wait_action)
                wait_result = self._executor.execute(wait_action, observation)
                return observation, wait_decision, wait_result
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
                            f"destination_blocked={self._in_flight_move.destination[1] in self._blocked_coordinates(observation.overworld)!r}",
                        )
                    destination_blocked = self._in_flight_move.destination[1] in self._blocked_coordinates(
                        observation.overworld
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
                        _gate_observer_call(
                            "record_navigation_outcome",
                            self._in_flight_move,
                            "blocked",
                            observation_frame=_gate_observation_frame(),
                            source_kind="movement",
                        )
                        self._remember_blocked_destination(self._in_flight_move, "movement made no progress")
                        # The observation was augmented before this in-flight
                        # movement was judged blocked.  Add the newly learned
                        # obstacle to the same observation before replanning;
                        # otherwise the planner can select the failed edge
                        # again and the final safety gate can only wait.
                        observation = self._augment_local_blocked_destinations(observation)
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
                # Emerald updates the logical tile coordinate before the
                # avatar finishes centering on that tile.  A direction change
                # issued during that MOVING frame can be consumed or ignored
                # by the field engine.  Treat the move as complete only after
                # a standing observation; this preserves the normal turn-only
                # retry path for the next action.
                if (
                    observation.overworld.movement_state is not None
                    and observation.overworld.movement_state is not MovementState.STANDING
                ):
                    wait_action = AgentAction(
                        AgentActionType.WAIT_REOBSERVE,
                        reason="waiting for movement destination to settle",
                    )
                    wait_decision = ActionDecision(wait_action)
                    wait_result = self._executor.execute(wait_action, observation)
                    return observation, wait_decision, wait_result
                self._locally_blocked_destinations.discard(self._in_flight_move.destination)
                _gate_observer_call(
                    "record_navigation_outcome",
                    self._in_flight_move,
                    "open",
                    observation_frame=_gate_observation_frame(),
                    source_kind="movement",
                )
                self._cached_action_index += 1
                self._movement_blocked_retries = 0
                self._movement_failure_count = 0
                self._clear_rejected_first_action()
                self._in_flight_move = None
                self._in_flight_move_initial_facing = None
            elif self._in_flight_move is not None and _is_intermediate_move_location(location, self._in_flight_move):
                trace = getattr(context, "stutter_trace", None)
                if trace is not None:
                    trace.mark(
                        "route_invalidation_detail",
                        f"movement_intermediate_progress current={location!r} "
                        f"expected_destination={self._in_flight_move.destination!r} "
                        f"direction={self._in_flight_move.direction.name!r}",
                    )
                wait_action = AgentAction(
                    AgentActionType.WAIT_REOBSERVE,
                    reason="waiting for forced movement destination",
                )
                wait_decision = ActionDecision(wait_action)
                wait_result = self._executor.execute(wait_action, observation)
                return observation, wait_decision, wait_result
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

        # The emulator can publish the new logical tile before the avatar has
        # finished moving across it.  This is especially easy to hit after a
        # cached straight segment, which clears ``_in_flight_move`` as soon as
        # the destination coordinate appears.  Do not select or emit the next
        # navigation input until the field engine reports a settled avatar;
        # otherwise a direction change can be consumed by the still-running
        # movement and the failed input is later misclassified as a blocked
        # tile.
        if observation.overworld is not None and observation.overworld.movement_state in (
            MovementState.MOVING,
            MovementState.TURNING,
        ):
            wait_action = AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="waiting for avatar movement to settle",
            )
            wait_decision = ActionDecision(wait_action)
            wait_result = self._executor.execute(wait_action, observation)
            return observation, wait_decision, wait_result

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
            # A completed staging route is an ownership boundary.  Once the
            # avatar is on the exact staging tile, the trainer exposure loop
            # must immediately continue with lead handoff; starting another
            # background search for the already-satisfied ReachLocation can
            # occupy the single-flight worker and delay that handoff.
            local_completion_reason = (
                _map_goal_completion_reason(observation)
                if interaction_type is InteractionType.OVERWORLD
                else None
            )
            if local_completion_reason is not None:
                self._report(
                    "NAVIGATION_PLAN_SUPPRESSED "
                    f"reason='goal already reached' target={_goal_target(self._goal)!r}"
                )
                # Keep the normal goal evaluator responsible for the terminal
                # result and supply it an already-complete empty plan.  This
                # preserves its diagnostics and interaction semantics while
                # avoiding a redundant background search.
                current_location = (
                    observation.overworld.map_id,
                    observation.overworld.player_coordinates,
                )
                decision = select_action(
                    observation,
                    navigation_result=(NavigationPlan((), current_location), None, None),
                )
            else:
                navigation_result = self._poll_background_navigation(observation)
                decision = select_action(observation, navigation_result=navigation_result)
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
        trainer_wait_result = self._apply_automatic_trainer_wait_watchdog(observation, decision)
        if trainer_wait_result is not None:
            wait_decision, wait_result = trainer_wait_result
            return observation, wait_decision, wait_result
        blocked_action = self._blocked_action(observation, decision.action.navigation)
        if blocked_action is not None:
            # This is the last boundary before an input reaches the emulator.
            # A freshly recomputed plan can still be stale if an object moved
            # during the planning call, so fail closed even when the route was
            # not supplied by CampaignPlan.
            self._report(
                "REPLAN: reason='selected route destination blocked' "
                f"source={blocked_action.source!r} destination={blocked_action.destination!r}"
            )
            self._reject_first_action(blocked_action, "selected route destination blocked", observation)
            self._invalidate_plan("selected_route_destination_blocked")
            wait_action = AgentAction(
                AgentActionType.WAIT_REOBSERVE,
                reason="selected movement destination is currently occupied",
            )
            wait_decision = ActionDecision(wait_action, decision.goal_evaluation)
            wait_result = self._executor.execute(wait_action, observation)
            return observation, wait_decision, wait_result
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
                self._cached_evaluation = decision.goal_evaluation
                self._cached_goal = self._goal
                self._cached_actions = ()
                self._cached_action_index = 0
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
        if result.result_type is ActionResultType.EXECUTED and not self._recovery_first_action_reported:
            controller = getattr(getattr(context, "bot_mode_instance", None), "controller", None)
            if getattr(controller, "_execution_phase", None) == "RECOVERY":
                navigation = decision.action.navigation
                print_recovery_handoff(
                    phase="first_action_emitted",
                    objective_id=getattr(controller, "current_objective_id", None),
                    execution_phase=getattr(controller, "_execution_phase", None),
                    action=decision.action.action_type.name,
                    action_source=getattr(navigation, "source", None),
                    action_destination=getattr(navigation, "destination", None),
                    detail=result.message or decision.action.reason or "first recovery action executed",
                )
                self._recovery_first_action_reported = True
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
                _gate_observer_call(
                    "record_navigation_issued",
                    decision.action.navigation,
                    action_frame=_gate_observation_frame(),
                    source_kind=(
                        "movement"
                        if decision.action.navigation.action_type is NavigationActionType.MOVE
                        else "transition"
                    ),
                )
                if decision.action.navigation.action_type is NavigationActionType.MOVE:
                    self._in_flight_move = decision.action.navigation
                    self._in_flight_move_initial_facing = observation.overworld.facing
                    safe_segment = self._safe_movement_batch(observation)
                    if self._use_movement_batch and len(safe_segment) > 1:
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
            # Keep the transition recorder tied to the controller's
            # transition ownership boundary as well as the generic action
            # dispatch boundary. The duplicate is idempotent in the recorder
            # and covers adapters that hand off the directional input through
            # a transition-specific path.
            _gate_observer_call(
                "record_navigation_issued",
                decision.action.navigation,
                action_frame=_gate_observation_frame(),
                source_kind="transition",
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
        previous_active_loop = getattr(context, "active_agent_control_loop", None)
        context.active_agent_control_loop = self
        try:
            while True:
                try:
                    if self._movement_batch is not None:
                        # The batch owns only the current emulator frame. If it ends
                        # or is interrupted, immediately return to the normal loop on
                        # the next iteration so existing handlers process the cause.
                        if self._advance_movement_batch():
                            yield
                            continue
                    _, _, result = self.step()
                except NavigationError as error:
                    retried = yield from self._wait_for_terminal_navigation_fallback(str(error))
                    if retried:
                        continue
                    raise
                if result.result_type is ActionResultType.GOAL_COMPLETE:
                    return
                if result.result_type in (ActionResultType.UNREACHABLE, ActionResultType.UNSUPPORTED):
                    # A generator returning here is indistinguishable from a
                    # successful recovery to CampaignController.  Propagate
                    # terminal navigation failure explicitly so recovery is
                    # marked failed and the interrupted objective remains
                    # available for safe policy-level handling.
                    reason = result.message or result.result_type.name.lower()
                    retried = yield from self._wait_for_terminal_navigation_fallback(reason)
                    if retried:
                        continue
                    raise NavigationError(reason)
                yield
        finally:
            if getattr(context, "active_agent_control_loop", None) is self:
                context.active_agent_control_loop = previous_active_loop
            self.dispose()
