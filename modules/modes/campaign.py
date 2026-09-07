"""Autonomous execution of the small ordered Emerald campaign."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import time
from typing import Generator
import traceback

from modules.context import context
from modules.agent_control import notify_battle_ended
from modules.console import diagnostic_print
from modules.nuzlocke.campaign_controller import CampaignController, runtime_campaign_state
from modules.nuzlocke.campaign_objectives import campaign_planning_signature, plan_campaign
from modules.nuzlocke.campaign_status import CampaignStatus

from ._interface import BattleAction, BotMode
from modules.nuzlocke.resource_policy import (
    PartyResource,
    ResourceObservationStatus,
    ResourceDecision,
    ResourceSnapshot,
    RouteRecovery,
    assess_wild_encounter,
)
from modules.nuzlocke.resource_runtime import (
    execute_planned_recovery,
    observe_resource_snapshot,
    observe_route_recovery,
    plan_route_recovery,
    prepare_route_recovery,
)
from modules.nuzlocke.emerald_healing_catalog import emerald_healing_sources_for_map
from modules.nuzlocke.emerald_healing_catalog import emerald_healing_source_for_destination
from modules.nuzlocke.level_cap import evaluate_battle_entry
from modules.modes._interface import BotModeError
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    CampaignReadinessPolicy,
    ReadinessObservationScheduler,
    build_progression_readiness_diagnostic,
)
from modules.nuzlocke.snapshots import get_nuzlocke_snapshot
from modules.overworld import (
    perceive_overworld,
    OverworldObservationResult,
    publish_shared_overworld_observation,
)
from modules.interaction_state import InteractionPhase, observe_interaction
from modules.player import get_player_avatar, player_avatar_is_controllable
from modules.memory import GameState, get_game_state
from modules.pokemon_party import get_party
from modules.goals import EncounterMode, GoalConstraints, NavigationGoal, ReachLocation, SemanticTarget, TrainerMode
from modules.navigation import NavigationWorld, RouteCostAnalyzer
from modules.stutter_trace import suppress_background_instrumentation
from modules.world_navigation import get_world_map_graph
from modules.modes.util.map import pokemon_center_candidates
from modules.battle_strategies.nuzlocke_level_balancing import (
    EmeraldIntroRivalBattleStrategy,
    NuzlockeCaptureStrategy,
    NuzlockeLevelBalancingBattleStrategy,
    RoxanneBattleStrategy,
)
from modules.modes.util.lead_rotation import ensure_campaign_field_lead
from modules.nuzlocke.field_lead import build_campaign_field_lead_context


_RECOVERY_ROUTE_ANALYSIS_MAX_EXPANSIONS = 4000


class _RoutePlanningBudget:
    """Cooperative wall-clock budget shared by a background route job."""

    def __init__(self, timeout_seconds: float):
        self.deadline = time.monotonic() + timeout_seconds
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def check(self) -> None:
        if self.cancelled:
            raise TimeoutError("background route search was superseded")
        if time.monotonic() >= self.deadline:
            self.cancelled = True
            raise TimeoutError("background route search exceeded 10 second budget")


class _SingleFlightBackgroundPlanner:
    """Run prioritized, cancellable route jobs without blocking frames.

    The name is retained for compatibility with existing tests/callers. The
    planner now permits one critical and one speculative job to overlap, which
    means a stale analysis cannot prevent urgent recovery from starting.
    """

    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nuzbot-route")
        self._jobs = {}
        self._completed = {}

    @staticmethod
    def _run(operation, budget):
        # Navigation's trace/profiler hooks refer to the mutable frame trace
        # and process-wide timing state. Suppress those hooks only on this
        # worker; the route calculation itself remains unchanged.
        with suppress_background_instrumentation():
            budget.check()
            try:
                accepts_budget = bool(inspect.signature(operation).parameters)
            except (TypeError, ValueError):
                accepts_budget = False
            return operation(budget) if accepts_budget else operation()

    @staticmethod
    def _key_digest(key) -> str:
        """Return a compact stable identifier for route-job diagnostics."""

        return hashlib.sha1(repr(key).encode("utf-8", errors="replace")).hexdigest()[:12]

    @classmethod
    def _key_domain(cls, key) -> str:
        domain = cls._job_domain(key)
        return domain or "unknown"

    def poll(self, key, operation, *, priority: int = 1, timeout_seconds: float | None = None):
        """Return ``pending``/``complete``/``error`` for the requested job."""

        cached = self._completed.get(key)
        if cached is not None:
            value, error = cached
            return "error" if error is not None else "complete", value, error

        job = self._jobs.get(key)
        if job is not None:
            if job["future"].done():
                return self._harvest(key, job)
            if time.monotonic() >= job["budget"].deadline:
                job["budget"].cancel()
                error = TimeoutError("background route search exceeded 10 second budget")
                # A running thread cannot be force-killed safely. Retain it
                # until it observes cancellation, but make the timeout a
                # terminal result for this request immediately.
                self._completed[key] = (None, error)
                if job["future"].cancel():
                    self._jobs.pop(key, None)
                diagnostic_print(
                    lambda: (
                        "BACKGROUND_ROUTE_JOB: "
                        "event='timeout' "
                        f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                        f"elapsed_ms={(time.monotonic() - job['started_at']) * 1000.0:.1f} "
                        f"active_jobs={len(self._jobs)!r} completed_jobs={len(self._completed)!r}"
                    ),
                    trace=True,
                    prefix="BACKGROUND_ROUTE_JOB",
                )
                return "error", None, error
            return "pending", None, None

        # Urgent recovery supersedes queued/speculative route work. Search
        # functions receive this cancellation token at every expansion, so a
        # lower-priority worker yields promptly instead of monopolizing CPU.
        for other_key, other in tuple(self._jobs.items()):
            if other["priority"] > priority:
                other["budget"].cancel()
                if other["future"].cancel():
                    self._jobs.pop(other_key, None)

        budget = _RoutePlanningBudget(timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS)
        future = self._executor.submit(self._run, operation, budget)
        self._jobs[key] = {
            "future": future,
            "budget": budget,
            "priority": priority,
            "started_at": time.monotonic(),
        }
        diagnostic_print(
            lambda: (
                "BACKGROUND_ROUTE_JOB: "
                "event='submitted' "
                f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                f"priority={priority!r} timeout_seconds={timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS!r} "
                f"active_jobs={len(self._jobs)!r} completed_jobs={len(self._completed)!r}"
            ),
            trace=True,
            prefix="BACKGROUND_ROUTE_JOB",
        )
        return "pending", None, None

    def _harvest(self, key, job):
        elapsed_ms = (time.monotonic() - job["started_at"]) * 1000.0
        try:
            value = job["future"].result()
            error = None
        except Exception as caught:
            value = None
            error = caught
        self._jobs.pop(key, None)
        self._completed[key] = (value, error)
        diagnostic_print(
            lambda: (
                "BACKGROUND_ROUTE_JOB: "
                f"event={'error' if error is not None else 'complete'!r} "
                f"domain={self._key_domain(key)!r} key_digest={self._key_digest(key)!r} "
                f"elapsed_ms={elapsed_ms:.1f} exception_type={type(error).__name__ if error else None!r} "
                f"active_jobs={len(self._jobs)!r} completed_jobs={len(self._completed)!r}"
            ),
            trace=True,
            prefix="BACKGROUND_ROUTE_JOB",
        )
        return "error" if error is not None else "complete", value, error

    def discard(self, key) -> None:
        """Forget a completed/retired request before an explicit retry."""

        self._completed.pop(key, None)
        job = self._jobs.get(key)
        if job is not None:
            job["budget"].cancel()
            job["future"].cancel()
            # Remove the logical slot immediately. The old worker is already
            # cancellation-marked and will exit cooperatively; a retry must
            # not inherit its expired deadline.
            self._jobs.pop(key, None)

    @staticmethod
    def _job_domain(key):
        if isinstance(key, tuple) and key and key[0] in {"recovery", "analysis"}:
            return key[0]
        return None

    def close(self) -> None:
        """Stop accepting route jobs when an embedding discards the mode."""

        self._executor.shutdown(wait=False, cancel_futures=True)


def _overworld_position_is_coherent(overworld) -> bool:
    """Return whether the observed avatar position belongs to the map model.

    During a map connection Emerald can publish the destination map header
    before it publishes a valid destination coordinate (for example ``y ==
    -1``).  That observation is useful to the transition executor, but it is
    not safe input for campaign readiness or recovery route selection.
    """
    tile_at = getattr(overworld, "tile_at", None)
    if not callable(tile_at):
        # Lightweight test/dry-run observations may not expose topology. Keep
        # their historical behavior and let the normal readiness checks apply.
        return True
    coordinates = getattr(overworld, "player_coordinates", None)
    if not isinstance(coordinates, tuple) or len(coordinates) != 2:
        return False
    try:
        return tile_at(coordinates) is not None
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return False


class CampaignProgressionMode(BotMode):
    """Bot mode that delegates each frame to observation-driven campaign planning."""

    @staticmethod
    def name() -> str:
        """Return the user-facing mode name."""

        return "Campaign Progression"

    @staticmethod
    def is_selectable() -> bool:
        """Return whether the active ROM supports Emerald campaign execution."""

        return context.rom is not None and context.rom.is_emerald

    def __init__(self):
        """Create the campaign controller and readiness scheduler."""

        self._readiness_scheduler = ReadinessObservationScheduler(
            self._readiness_input,
            self._cheap_readiness_context,
            max_age_ticks=15,
        )
        self.controller = CampaignController(
            runtime_campaign_state,
            selector=self._select_campaign,
            readiness_provider=self._readiness_scheduler.observe,
            # CampaignPlan owns recovery selection.  There is deliberately no
            # fallback here that can reopen the healing catalog or select a
            # different Center after the plan has been composed.
            recovery_factory=lambda stop: execute_planned_recovery(
                stop.destination,
                emerald_healing_source_for_destination(stop.destination),
                planned_route=stop.route,
            ),
        )
        self._readiness_evaluated = False
        self._campaign_selection_key = None
        self._campaign_selection = None
        self._campaign_boundary_context_seen = False
        self._last_campaign_boundary_context = None
        self._active_battle_wild: bool | None = None
        self._lead_handoff_pending = False
        # Recovery and opportunistic route analysis share one worker so a
        # stale search cannot queue behind a newer one indefinitely.
        self._background_route_planner = _SingleFlightBackgroundPlanner()
        self._prepared_recovery_request = None
        self._prepared_recovery_anchor = None

    def _clear_background_route_state(self) -> None:
        """Discard frame-bound recovery inputs after a semantic boundary."""

        self._prepared_recovery_request = None
        self._prepared_recovery_anchor = None

    @staticmethod
    def _recovery_anchor():
        runtime = getattr(context, "nuzlocke_runtime", None)
        snapshot = getattr(runtime, "latest_snapshot", None)
        player = getattr(snapshot, "player", None)
        if player is None:
            return None
        return (
            getattr(player, "map_group", None),
            getattr(player, "map_number", None),
            getattr(player, "coordinates", None),
        )

    def _select_campaign(self, state):
        """Reuse map-level planning until a relevant campaign fact changes."""

        key = campaign_planning_signature(state)
        if key == self._campaign_selection_key and self._campaign_selection is not None:
            return self._campaign_selection
        selection = plan_campaign(state)
        self._campaign_selection_key = key
        self._campaign_selection = selection
        return selection

    def _observe_route_recovery_async(
        self, *, candidate_limit: int | None = None, critical: bool = False
    ) -> RouteRecovery:
        """Prepare recovery on the frame thread and search it in one worker."""

        # Most unit-level callers construct the mode with ``__new__`` to
        # exercise readiness in isolation. Preserve their synchronous seam;
        # the live mode always initializes the background planner.
        planner = getattr(self, "_background_route_planner", None)
        if planner is None:
            return observe_route_recovery(candidate_limit=candidate_limit)
        anchor = self._recovery_anchor()
        prepared = self._prepared_recovery_request
        if (
            prepared is None
            or self._prepared_recovery_anchor != anchor
            or getattr(prepared, "critical", False) is not critical
        ):
            try:
                prepared = prepare_route_recovery(candidate_limit=candidate_limit, critical=critical)
            except Exception as error:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RECOVERY_PREPARATION_FAILURE: "
                        f"exception_type={type(error).__name__!r} exception={error!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_RECOVERY_PREPARATION_FAILURE",
                )
                return RouteRecovery(observation_available=False, observation_error=str(error))
            self._prepared_recovery_request = prepared if hasattr(prepared, "key") else None
            self._prepared_recovery_anchor = anchor
        if isinstance(prepared, RouteRecovery):
            return prepared

        status, result, error = planner.poll(
            ("recovery", prepared.key),
            lambda budget, prepared=prepared: plan_route_recovery(prepared, cancel_check=budget.check),
            priority=0 if critical else 1,
        )
        if status == "pending":
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_CALCULATION: "
                    f"state='pending' request_key={prepared.key!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_CALCULATION",
            )
            return RouteRecovery(
                observation_available=False,
                observation_error="route_calculation_pending",
                calculation_pending=True,
            )
        if error is not None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_CALCULATION: "
                    f"state='error' exception_type={type(error).__name__!r} exception={error!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_CALCULATION",
            )
            timed_out = isinstance(error, TimeoutError)
            if timed_out:
                planner.discard(("recovery", prepared.key))
                self._clear_background_route_state()
            return RouteRecovery(
                # A non-empty catalog is not evidence that a route was
                # found. Reporting it as available makes readiness return
                # RECOVER even though CampaignPlan has no executable
                # RecoveryStop, producing a false campaign block.
                # Treat worker failures as unavailable observations so the
                # same request can be retried after a transient boundary.
                observation_available=timed_out,
                observation_error=str(error),
                calculation_timed_out=timed_out,
            )
        diagnostic_print(
            lambda: (
                "CAMPAIGN_RECOVERY_CALCULATION: "
                f"state='complete' center={getattr(result, 'center_location', None)!r} "
                f"distance={getattr(result, 'distance_to_center', None)!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_RECOVERY_CALCULATION",
        )
        return result

    @staticmethod
    def _route_analysis_key(world, start, goal, candidates) -> str:
        """Build a stable key from analysis inputs without loading new maps."""
        blocked, transitions = world.revision_key()
        return repr(
            (
                start,
                repr(goal),
                tuple(repr(candidate) for candidate in candidates),
                getattr(world, "facing", None),
                getattr(world, "running_shoes", None),
                getattr(world, "player_elevation", None),
                getattr(world, "surfing", None),
                transitions,
                blocked,
            )
        )

    def _analyze_route_async(self, world, graph, start, goal, candidates):
        """Run opportunistic route composition without blocking a frame."""

        planner = getattr(self, "_background_route_planner", None)
        if planner is None:
            return RouteCostAnalyzer(
                world,
                graph=graph,
                max_expansions=_RECOVERY_ROUTE_ANALYSIS_MAX_EXPANSIONS,
            ).analyze(start, goal, candidates), False, False

        key = ("analysis", self._route_analysis_key(world, start, goal, candidates))
        status, result, error = planner.poll(
            key,
            lambda budget: RouteCostAnalyzer(
                world,
                graph=graph,
                max_expansions=_RECOVERY_ROUTE_ANALYSIS_MAX_EXPANSIONS,
            ).analyze(start, goal, candidates, cancel_check=budget.check),
            priority=2,
        )
        if status == "pending":
            diagnostic_print(
                lambda: f"CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: state='pending' request_key={key[1]!r}",
                trace=True,
                prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
            )
            return None, True, False
        if error is not None:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: "
                    f"state='error' exception_type={type(error).__name__!r} exception={error!r}"
                ),
                trace=True,
                prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
            )
            timed_out = isinstance(error, TimeoutError)
            if timed_out:
                # The request key is stable, so discard the terminal worker
                # result and allow the readiness scheduler's cooldown to
                # submit a fresh bounded request later.
                planner.discard(key)
            return None, False, timed_out
        diagnostic_print(
            lambda: f"CAMPAIGN_RECOVERY_ROUTE_ANALYSIS: state='complete' request_key={key[1]!r}",
            trace=True,
            prefix="CAMPAIGN_RECOVERY_ROUTE_ANALYSIS",
        )
        return result, False, False

    def _campaign_controller(self):
        """Return the controller owned by this mode instance."""

        return self.controller

    @staticmethod
    def _cheap_readiness_context():
        """Read inexpensive boundary inputs used to age readiness diagnostics.

        Exact coordinates are intentionally excluded.  Walking normally changes
        them every frame, but does not change the party's readiness facts or the
        set of map-level recovery affordances.  Battle completion explicitly
        invalidates the scheduler, while the map identity below naturally
        invalidates it across a map transition.
        """

        runtime = getattr(context, "nuzlocke_runtime", None)
        snapshot = getattr(runtime, "latest_snapshot", None)
        if snapshot is not None:
            player = getattr(snapshot, "player", None)
            map_id = (
                getattr(player, "map_group", None),
                getattr(player, "map_number", None),
            )
            state = getattr(getattr(snapshot, "game_state", None), "name", None)
            controllable = getattr(player, "controllable", None)
            return (
                state,
                map_id,
                controllable,
                getattr(snapshot, "readiness_revision", None),
            )
        try:
            avatar = get_player_avatar()
            state = getattr(get_game_state(), "name", None)
            map_id = avatar.map_group_and_number
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        try:
            # A script can leave the game state and map unchanged while
            # releasing control on a later frame. Include this cheap
            # boundary so an unavailable readiness result is refreshed
            # immediately when tactical ownership returns. Keep a failed
            # control read distinct from a failed avatar/map read so test
            # doubles and transient emulator boundaries still have a useful
            # map-level cache key.
            controllable = player_avatar_is_controllable()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            controllable = None
        # Runtime already materializes the inventory and party snapshot once
        # per emulator frame. Reuse those immutable observations so HP/status,
        # item, healing, and party-composition changes invalidate readiness
        # without rereading save blocks on the hot path.
        ball_count = None
        if snapshot is not None and getattr(snapshot, "inventory_available", False):
            try:
                ball_count = sum(
                    item.quantity for item in snapshot.inventory.poke_balls if item.name.casefold() != "master ball"
                )
            except (AttributeError, TypeError, ValueError):
                ball_count = None
        if snapshot is None:
            # Lightweight embedders may not mount the Nuzlocke runtime. Keep
            # their historical coarse key shape and behavior.
            return (state, map_id, controllable, ball_count)

        try:
            party_signature = tuple(
                (
                    pokemon.party_index,
                    pokemon.species,
                    pokemon.level,
                    pokemon.current_hp,
                    pokemon.max_hp,
                    pokemon.status,
                    pokemon.fainted,
                    pokemon.egg,
                    pokemon.held_item,
                    tuple((move.name, move.pp, move.total_pp, move.pp_ups) for move in pokemon.moves),
                )
                for pokemon in snapshot.party
            )
        except (AttributeError, TypeError, ValueError):
            party_signature = None
        try:
            inventory_signature = tuple(
                (item.name, item.quantity)
                for item in (
                    tuple(snapshot.inventory.items)
                    + tuple(snapshot.inventory.poke_balls)
                    + tuple(snapshot.inventory.key_items)
                )
            )
        except (AttributeError, TypeError, ValueError):
            inventory_signature = None
        try:
            campaign_observation = snapshot.campaign_observation
            campaign_signature = (
                tuple((flag.name, flag.value) for flag in campaign_observation.flags),
                tuple((variable.name, variable.value) for variable in campaign_observation.variables),
                campaign_observation.text_speed,
                getattr(campaign_observation.lifecycle, "value", campaign_observation.lifecycle),
            )
        except (AttributeError, TypeError, ValueError):
            campaign_signature = None
        return (state, map_id, controllable, ball_count, party_signature, inventory_signature, campaign_signature)

    @staticmethod
    def _recovery_candidate_goals(current_location, recovery, *, is_rse):
        """Return a bounded set of recovery destinations for readiness analysis.

        ``observe_route_recovery`` has already selected the nearest safe
        source.  Readiness only needs to compare that route with the active
        objective, so enumerating every healing source in the region is both
        redundant and capable of blocking the tactical loop for a long time.
        Keep same-map sources when present (there can be more than one in a
        future ROM), otherwise compare the selected destination only.
        """
        selected_goal = getattr(recovery, "navigation_goal", None)
        if selected_goal is not None:
            # Recovery runtime already selected and validated one catalog
            # source. Reuse its executable interior-warp goal so the
            # readiness composition measures the same route that recovery
            # will execute.
            return (selected_goal,)
        if is_rse:
            local_sources = emerald_healing_sources_for_map(current_location[0])
            if local_sources:
                return tuple(ReachLocation(source.outdoor_location) for source in local_sources)
            selected = getattr(recovery, "center_location", None)
            if selected is not None:
                return (ReachLocation(selected),)

        centers = pokemon_center_candidates(current_location)
        return tuple(ReachLocation(center.value) for center in centers[:1])

    def _readiness_input(self, objective, goal):
        """Build the current readiness diagnostic from shared ROM observations."""

        self._readiness_evaluated = True
        # Perceive the overworld first.  The callback-based game-state reader
        # can briefly report UNKNOWN at a battle/script boundary even when
        # the authoritative overworld observation has already become stable.
        # Several Emerald capabilities intentionally have no tactical goal:
        # their observation-driven delegate owns the script interaction. They
        # still have a declared map destination, which is sufficient for
        # readiness to compare continuing with returning to a healing source.
        # Without this projection a damaged party is classified as
        # opportunistically unknown and the campaign waits in place.
        readiness_goal = goal
        if readiness_goal is None:
            destination = getattr(objective, "destination", None)
            if isinstance(destination, tuple) and len(destination) == 2:
                target_map = (
                    destination[0]
                    if isinstance(destination[0], tuple)
                    and len(destination[0]) == 2
                    and isinstance(destination[1], tuple)
                    else destination
                )
                readiness_goal = NavigationGoal(
                    SemanticTarget.map(target_map),
                    constraints=GoalConstraints(trainer_mode=TrainerMode.AVOID),
                    encounter_mode=EncounterMode.AVOID,
                )

        overworld = perceive_overworld()
        publish_shared_overworld_observation(overworld)
        runtime = getattr(context, "nuzlocke_runtime", None)
        snapshot = getattr(runtime, "latest_snapshot", None) if runtime is not None else None
        if snapshot is None:
            snapshot = get_nuzlocke_snapshot()
        diagnostic_print(
            lambda: (
                "READINESS_INPUT_LIFECYCLE: "
                f"controller_id={id(self.controller)!r} evaluation_id={id(snapshot)!r} "
                f"perceive_type={type(overworld).__name__!r} "
                f"perceive_status={getattr(getattr(overworld, 'status', None), 'value', None)!r}"
            ),
            trace=True,
        )
        overworld_availability = Availability.KNOWN
        overworld_reason = None
        stable_overworld = False
        if isinstance(overworld, OverworldObservationResult):
            diagnostic_print(
                lambda: (
                    "READINESS_OVERWORLD_OBSERVATION_FAILURE: "
                    f"status={getattr(overworld.status, 'value', overworld.status)!r} "
                    f"reason={getattr(overworld, 'reason', None)!r} "
                    f"frame={getattr(snapshot, 'frame', None)!r} "
                    f"game_state={getattr(get_game_state(), 'name', get_game_state())!r}"
                ),
                trace=True,
            )
            overworld_availability = Availability.UNKNOWN
            overworld_reason = overworld.reason
            overworld = None
        elif not getattr(overworld, "controllable", False):
            # Coordinates and party state can remain readable while a battle
            # return or scripted post-battle sequence still owns the field.
            # Do not let readiness mount recovery during that interval; doing
            # so replaces the campaign loop before its dialogue can advance.
            overworld_availability = Availability.UNKNOWN
            overworld_reason = "player avatar is not controllable"
        elif getattr(overworld, "transition_in_progress", False):
            # A controllable bit can lead the rest of a warp/script transition.
            # Do not let recovery take ownership until the transition signals
            # have cleared.
            overworld_availability = Availability.UNKNOWN
            overworld_reason = "overworld transition is in progress"
        elif not _overworld_position_is_coherent(overworld):
            # The destination header and avatar coordinates are read from
            # different ROM structures.  A map connection can therefore
            # briefly expose an out-of-bounds coordinate while the map is
            # settling.  Do not mount recovery from that half-observation;
            # doing so would replace the campaign loop before the transition
            # executor can observe its postcondition.
            overworld_availability = Availability.UNKNOWN
            overworld_reason = "avatar coordinate is not present in current overworld topology"
        else:
            # The avatar can become controllable before a post-battle native
            # script has released the field. Use the shared interaction
            # classifier so recovery cannot mount during dialogue or a native
            # script wait, while allowing it as soon as the interaction phase
            # is genuinely clear.
            try:
                interaction = observe_interaction()
                interaction_phase = getattr(interaction, "interaction_phase", InteractionPhase.NONE)
                dialogue_owned = interaction_phase in {
                    InteractionPhase.FIELD_MESSAGE_RENDER_WAIT,
                    InteractionPhase.FIELD_MESSAGE_INPUT_WAIT,
                    InteractionPhase.CHOICE_MENU_INPUT_WAIT,
                } or (
                    interaction_phase is InteractionPhase.SCRIPT_NATIVE_WAIT
                    and (
                        # A native script can own the field before it
                        # exposes a dialogue box (for example, the
                        # post-battle rival exit movement). Readiness
                        # must not mount a recovery plan across that
                        # ownership boundary.
                        getattr(interaction, "script_active", False)
                        or getattr(interaction, "dialogue_waiting", False)
                        or getattr(interaction, "field_message_lifecycle_active", False)
                        or getattr(interaction, "native_function", None)
                        in {"WaitForAorBPress", "IsFieldMessageBoxHidden"}
                    )
                )
                if dialogue_owned:
                    overworld_availability = Availability.UNKNOWN
                    overworld_reason = "interaction phase is active: " f"{getattr(interaction_phase, 'name', None)}"
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                pass
            stable_overworld = overworld_availability is Availability.KNOWN

        # ``get_nuzlocke_snapshot`` and ``perceive_overworld`` read different
        # emulator signals.  When only the callback classifier is unknown,
        # use the validated stable-overworld boundary instead of deferring
        # indefinitely.  Never normalize a known non-overworld state, and
        # never do this while dialogue or a transition owns the field.
        if stable_overworld and getattr(snapshot, "game_state", None) is GameState.UNKNOWN:
            snapshot = replace(snapshot, game_state=GameState.OVERWORLD)
            diagnostic_print(
                lambda: (
                    "READINESS_GAME_STATE_NORMALIZED: "
                    "snapshot_state='UNKNOWN' observation_state='STABLE_OVERWORLD' "
                    "normalized_state='OVERWORLD'"
                ),
                trace=True,
            )
        resources = observe_resource_snapshot()
        # The party reader can briefly return a valid empty tuple while a
        # battle-return script is handing control back. That is not evidence
        # of a wiped party: treat it as unavailable unless the campaign
        # snapshot independently confirms that an empty party is real.
        resource_party_available = bool(resources.party) or bool(getattr(snapshot, "party_available", False))
        resource_observation_valid = (
            resources.observation_status is ResourceObservationStatus.VALID and resource_party_available
        )
        resource_availability = Availability.KNOWN if resource_observation_valid else Availability.UNKNOWN
        trace = getattr(context, "stutter_trace", None)
        if trace is not None and callable(getattr(trace, "mark", None)):
            trace.mark("campaign_snapshot_party_available", getattr(snapshot, "party_available", None))
            trace.mark("campaign_snapshot_party_count", len(getattr(snapshot, "party", ()) or ()))
            trace.mark("campaign_resource_party_count", len(resources.party))
            trace.mark("campaign_resource_usable_party_count", len(resources.usable_party))
            trace.mark("campaign_resource_worst_hp_ratio", resources.worst_hp_ratio)
            trace.mark("campaign_resource_party_available", resource_party_available)
        # Route recovery is only a policy input when the party may actually
        # need recovery.  Computing it here on every readiness refresh walks
        # the map synchronously, which stalls the active tactical controller
        # even when the party is healthy and already navigating normally.
        objective_id = getattr(objective, "objective_id", None)
        critical_recovery_needed = resource_observation_valid and (
            not resources.usable_party
            or any(member.fainted for member in resources.party)
            or resources.has_poisoned_party
            or resources.worst_hp_ratio < CampaignReadinessPolicy().critical_hp_ratio
        )
        # The main readiness policy owns opportunistic recovery for every
        # stable-overworld objective. Objective resource metadata still
        # controls battle/encounter posture, but never whether this safety
        # check is eligible.
        opportunistic_recovery_needed = (
            resource_observation_valid
            and resources.worst_hp_ratio <= CampaignReadinessPolicy().opportunistic_hp_ratio
        )
        recovery_needed = critical_recovery_needed or opportunistic_recovery_needed
        # Critical recovery is a safety decision, not a detour optimization.
        # ``observe_route_recovery`` already selected and retained the
        # executable route to the nearest safe healing source. Running the
        # full normal-vs-recovery RouteCostAnalyzer here launches additional
        # world searches immediately after every damaging battle and can make
        # the emulator appear frozen for many seconds.
        if trace is not None and callable(getattr(trace, "mark", None)):
            trace.mark("campaign_recovery_needed", recovery_needed)
            trace.mark("campaign_critical_recovery", critical_recovery_needed)
        if overworld_availability is not Availability.KNOWN:
            recovery = RouteRecovery(
                observation_available=False,
                observation_error=overworld_reason or "overworld_observation_unavailable",
            )
        elif not resource_observation_valid:
            recovery = RouteRecovery(
                observation_available=False,
                observation_error=resources.observation_error or "resource_observation_unavailable",
            )
        else:
            recovery = (
                # Critical recovery already selected the nearest safe source
                # in the recovery observer. A second candidate only repeats
                # the expensive synchronous route search at the worst
                # possible time: immediately after a battle.
                self._observe_route_recovery_async(
                    candidate_limit=None,
                    critical=critical_recovery_needed,
                )
                if recovery_needed
                else RouteRecovery()
            )
        recovery_available = (
            recovery.center_available
            or getattr(recovery, "healing_source_available", False)
            or bool(resources.bag_healing_items)
        )
        recovery_availability = (
            Availability.UNKNOWN
            if not recovery.observation_available
            else (
                Availability.UNKNOWN
                if resources.observation_status is not ResourceObservationStatus.VALID
                else Availability.KNOWN if recovery_available else Availability.UNAVAILABLE
            )
        )
        # Diagnostics must remain strictly non-failing when a partial snapshot
        # is observed during startup or a map/battle transition.
        try:
            party_rows = []
            for pokemon in getattr(snapshot, "party", None) or ():
                current_hp = getattr(pokemon, "current_hp", None)
                max_hp = getattr(pokemon, "max_hp", None)
                fainted = getattr(pokemon, "fainted", None)
                if fainted is True or current_hp is None or current_hp <= 0:
                    continue
                ratio = current_hp / max_hp if isinstance(max_hp, (int, float)) and max_hp > 0 else None
                party_rows.append({"current_hp": current_hp, "max_hp": max_hp, "hp_ratio": ratio})
        except Exception:
            party_rows = None
        resource_status = getattr(getattr(resources, "observation_status", None), "value", None)
        recovery_center_available = getattr(recovery, "center_available", None)
        recovery_safe = getattr(recovery, "safe_to_reach_center", None)
        recovery_distance = getattr(recovery, "distance_to_center", None)
        diagnostic_print(
            lambda: (
                "READINESS_INPUT: "
                f"frame={getattr(snapshot, 'frame', None)!r} map={getattr(overworld, 'map_id', None)!r} "
                f"location={getattr(overworld, 'player_coordinates', None)!r} "
                f"party={party_rows!r} party_status={getattr(snapshot, 'party_available', None)!r} resource_status={resource_status!r} "
                f"resource_party_count={len(resources.party)!r} resource_usable_party_count={len(resources.usable_party)!r} "
                f"resource_party_available={resource_party_available!r} recovery_needed={recovery_needed!r} "
                f"recovery_availability={getattr(recovery_availability, 'value', None)!r} center_available={recovery_center_available!r} "
                f"center_safe={recovery_safe!r} distance_to_center={recovery_distance!r} "
                "distance_metric='world navigation route cost' "
                f"nearest_center={getattr(recovery, 'center_location', None)!r}"
            ),
            trace=True,
        )
        route_analysis = None
        route_analysis_pending = False
        route_analysis_timed_out = False
        # A targetless capability has no tactical route to compare against.
        # Its readiness policy uses the already-selected recovery route and
        # its bounded distance; composing a normal route plus recovery
        # outbound/continuation routes here launches redundant synchronous
        # searches in one application frame.
        direct_targetless_recovery = (
            objective is not None and goal is None
        )
        if direct_targetless_recovery and recovery_needed and not critical_recovery_needed:
            diagnostic_print(
                lambda: (
                    "READINESS_ROUTE_ANALYSIS_SKIPPED: "
                    "reason='targetless capability uses selected recovery route directly' "
                    f"distance_to_center={getattr(recovery, 'distance_to_center', None)!r}"
                ),
                trace=True,
                prefix="READINESS_ROUTE_ANALYSIS_SKIPPED",
            )
        if overworld is None or readiness_goal is None:
            diagnostic_print(
                lambda: (
                    "ROUTE_ANALYSIS_CONSTRUCTION_FAILURE\n"
                    "exception_type=UnavailableContext\n"
                    "exception_message=route-analysis precondition unavailable\n"
                    "failing_operation=precondition\n"
                    f"active_tactical_goal={readiness_goal!r}\n"
                    f"goal_type={type(readiness_goal).__name__ if readiness_goal is not None else None}\n"
                    f"navigation_world_available={overworld is not None}\n"
                    f"world_graph_available=False\n"
                    "candidate_count=0\n"
                    "candidate_destinations=[]"
                ),
                trace=True,
            )
        if (
            recovery_needed
            and not critical_recovery_needed
            and not direct_targetless_recovery
            and not getattr(recovery, "calculation_pending", False)
            and overworld is not None
            and overworld_availability is Availability.KNOWN
            and readiness_goal is not None
        ):
            operation = "initialization"
            candidate_goals = ()
            current_location = None
            world = None
            graph = None
            try:
                operation = "current location"
                current_location = (overworld.map_id, overworld.player_coordinates)
                operation = "NavigationWorld.from_overworld"
                world = NavigationWorld.from_overworld(overworld)
                operation = "candidate enumeration"
                if context.rom.is_rse:
                    candidates = self._recovery_candidate_goals(
                        current_location,
                        recovery,
                        is_rse=True,
                    )
                else:
                    candidates = self._recovery_candidate_goals(
                        current_location,
                        recovery,
                        is_rse=False,
                    )
                candidate_goals = candidates
                operation = "WorldMapGraph acquisition"
                graph = get_world_map_graph()
                operation = "RouteCostAnalyzer.analyze"
                route_analysis, route_analysis_pending, route_analysis_timed_out = self._analyze_route_async(
                    world,
                    graph,
                    current_location,
                    readiness_goal,
                    candidates,
                )
                if route_analysis_pending:
                    diagnostic_print(
                        lambda: (
                            "READINESS_ROUTE_ANALYSIS: "
                            f"state='pending' start={current_location!r} goal={readiness_goal!r}"
                        ),
                        trace=True,
                        prefix="READINESS_ROUTE_ANALYSIS",
                    )
                else:
                    diagnostic_print(
                        lambda: (
                            "READINESS_ROUTE_ANALYSIS: "
                            f"state='complete' start={current_location!r} goal={readiness_goal!r} "
                            f"normal_cost={route_analysis.normal_cost!r} "
                            f"normal_destination={getattr(route_analysis.normal_route, 'destination', None)!r} "
                            "candidates="
                            f"{[
                                {
                                    'destination': repr(candidate.destination),
                                    'reachable': candidate.reachable,
                                    'total_cost': candidate.total_cost,
                                    'detour': candidate.detour,
                                    'reason': candidate.reason,
                                    'first_destination': getattr(candidate.first_route, 'destination', None),
                                    'continuation_destination': getattr(candidate.continuation_route, 'destination', None),
                                }
                                for candidate in route_analysis.candidates
                            ]!r}"
                        ),
                        trace=True,
                        prefix="READINESS_ROUTE_ANALYSIS",
                    )
            except (RuntimeError, TypeError, ValueError) as error:
                diagnostic_print(
                    lambda: (
                        "ROUTE_ANALYSIS_CONSTRUCTION_FAILURE\n"
                        f"exception_type={type(error).__name__}\n"
                        f"exception_message={error}\n"
                        f"failing_operation={operation}\n"
                        f"active_tactical_goal={readiness_goal!r}\n"
                        f"goal_type={type(readiness_goal).__name__ if readiness_goal is not None else None}\n"
                        f"current_map_location={current_location!r}\n"
                        f"navigation_world_available={world is not None}\n"
                        f"world_graph_available={graph is not None}\n"
                        f"candidate_count={len(candidate_goals)}\n"
                        f"candidate_destinations={[repr(candidate) for candidate in candidate_goals]!r}\n"
                        f"traceback={traceback.format_exc()}"
                    ),
                    trace=True,
                )
                route_analysis = None
        elif recovery_needed and critical_recovery_needed:
            diagnostic_print(
                lambda: (
                    "READINESS_ROUTE_ANALYSIS_SKIPPED: "
                    "reason='critical recovery uses selected recovery route' "
                    f"recovery_route_available={getattr(recovery, 'route', None) is not None}"
                ),
                trace=True,
                prefix="READINESS_ROUTE_ANALYSIS_SKIPPED",
            )
        return build_progression_readiness_diagnostic(
            snapshot,
            objective_id=objective.objective_id if objective is not None else None,
            objective_status="ready",
            destination=getattr(objective, "destination", None),
            navigation_goal=readiness_goal,
            campaign_mode="Campaign Progression",
            overworld=overworld,
            resource_snapshot=resources,
            recovery=recovery,
            recovery_availability=recovery_availability,
            overworld_availability=overworld_availability,
            overworld_reason=overworld_reason,
            resource_availability=resource_availability,
            resource_reason=resources.observation_error,
            route_analysis=route_analysis,
            targetless=goal is None,
            route_analysis_pending=route_analysis_pending,
            route_analysis_timed_out=route_analysis_timed_out,
        )

    def on_battle_started(self, encounter) -> BattleAction:
        """Choose campaign battle policy, leaving execution to the listener."""
        self._active_battle_wild = encounter is not None
        scheduler = getattr(self, "_readiness_scheduler", None)
        if scheduler is not None:
            scheduler.invalidate("battle_started")

        controller = getattr(self, "controller", None)
        objective = (
            controller.last_selection.objective
            if controller is not None and controller.last_selection is not None
            else None
        )
        if objective is not None:
            legality = evaluate_battle_entry(
                runtime_campaign_state().campaign_facts,
                get_party(),
                objective_id=objective.objective_id,
                rule_config=getattr(getattr(context, "nuzlocke_runtime", None), "rule_config", None),
            )
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_LEVEL_CAP_ENTRY: "
                    f"objective={objective.objective_id!r} allowed={legality.allowed!r} "
                    f"reason={legality.reason!r} boss={getattr(legality.assessment.active_boss, 'boss_id', None)!r} "
                    f"cap={legality.assessment.level_cap!r} illegal_party_indices={legality.assessment.illegal_party_indices!r}"
                ),
                trace=True,
            )
            if not legality.allowed:
                raise BotModeError(f"Campaign battle entry rejected: {legality.reason}")

            # A preparation area still owns the first legal wild encounter.
            # Resolve that ownership before selecting the training strategy;
            # otherwise the strategy would defeat a catchable encounter.
            if encounter is not None and controller is not None and controller.last_selection is not None:
                runtime = getattr(context, "nuzlocke_runtime", None)
                try:
                    location = get_player_avatar().map_group_and_number
                    legal_capture_target = bool(
                        runtime is not None and runtime.capture_target_for(location, is_wild=True, is_trainer=False)
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                    legal_capture_target = False
                if legal_capture_target:
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_WILD_CAPTURE_HANDOFF: "
                            f"location={location!r} action='Catch' objective="
                            f"{controller.last_selection.objective.objective_id!r}"
                        ),
                        trace=False,
                    )
                    return BattleAction.Catch
            if objective.objective_id == "prepare_roxanne":
                return NuzlockeLevelBalancingBattleStrategy()
            if objective.objective_id == "defeat_roxanne":
                return RoxanneBattleStrategy()
            if objective.objective_id == "complete_intro_rival":
                return EmeraldIntroRivalBattleStrategy()
        if encounter is not None and controller is not None and controller.last_selection is not None:
            # The Nuzlocke rules projection is authoritative about whether
            # this wild battle owns the area's first legal encounter.  Make
            # that decision at the listener boundary so Campaign Progression
            # can use the existing CatchStrategy; relying only on the later
            # battle-state property lets the opening turn fall through to a
            # defeating move when the projection has just crossed the battle
            # start boundary.
            runtime = getattr(context, "nuzlocke_runtime", None)
            try:
                location = get_player_avatar().map_group_and_number
                legal_capture_target = bool(
                    runtime is not None and runtime.capture_target_for(location, is_wild=True, is_trainer=False)
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                legal_capture_target = False
            if legal_capture_target:
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_WILD_CAPTURE_HANDOFF: "
                        f"location={location!r} action='Catch' objective="
                        f"{controller.last_selection.objective.objective_id!r}"
                    ),
                    trace=False,
                )
                return BattleAction.Catch
            objective = controller.last_selection.objective
            resource_policy = getattr(objective, "resource_policy", None)
            if resource_policy is not None:
                party = get_party()
                snapshot = ResourceSnapshot(
                    tuple(
                        PartyResource(
                            pokemon.current_hp,
                            pokemon.total_hp,
                            getattr(pokemon.status_condition, "value", None),
                            pokemon.current_hp <= 0,
                        )
                        for pokemon in party
                    )
                )
                decision = assess_wild_encounter(resource_policy, snapshot, RouteRecovery())
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_RESOURCE: "
                        f"objective={resource_policy.objective_id} encounter=wild "
                        f"hp_ratio={snapshot.worst_hp_ratio:.3f} "
                        f"missing_hp={snapshot.total_missing_hp} "
                        f"encounter_policy={resource_policy.encounters.value} "
                        f"decision={decision.value}"
                    ),
                    trace=True,
                )
                if decision is ResourceDecision.PREFER_RUN:
                    return BattleAction.RunAway
        # Ordinary campaign-owned battles should rotate the lowest-level
        # living party member into the lead.  The explicit handlers above
        # retain their specialized policies, and legal wild captures were
        # handed to CatchStrategy before reaching this point.  This applies
        # to trainer battles too, where ``encounter`` is None.
        if objective is not None:
            return NuzlockeLevelBalancingBattleStrategy()
        return BattleAction.Fight

    def capture_battle_strategy(self):
        """Return the campaign capture policy for the battle listener."""

        return NuzlockeCaptureStrategy()

    def on_spotted_by_trainer(self) -> None:
        """Recheck campaign readiness when a trainer battle becomes imminent."""

        controller = getattr(self, "controller", None)
        request_recheck = getattr(controller, "request_readiness_recheck", None)
        if callable(request_recheck):
            request_recheck("trainer_spotted")

    def on_battle_ended(self, outcome) -> None:
        """Invalidate readiness after a battle changes party or route state."""

        # Campaign tactical loops are suspended while BattleListener owns the
        # battle. Trainer battles can move/defeat an overworld object, so the
        # cached route must be discarded. Wild battles do not alter route
        # geometry; retaining their route avoids a synchronous world search
        # after every encounter. Readiness is still refreshed below in both
        # cases so HP, party, and Poké Ball changes are observed.
        wild_battle = getattr(self, "_active_battle_wild", None)
        self._active_battle_wild = None
        self._clear_background_route_state()
        controller = getattr(self, "controller", None)
        selection = getattr(controller, "last_selection", None)
        objective = getattr(selection, "objective", None)
        notify_battle_ended(invalidate_navigation=wild_battle is not True)
        diagnostic_print(
            lambda: (
                "CAMPAIGN_BATTLE_ENDED: "
                f"frame={getattr(context, 'frame', None)!r} outcome={outcome!r} "
                f"objective={getattr(getattr(self.controller, 'last_selection', None), 'objective', None)!r} "
                f"facts={runtime_campaign_state().campaign_facts!r}"
            ),
            trace=True,
        )
        scheduler = getattr(self, "_readiness_scheduler", None)
        request_recheck = getattr(controller, "request_readiness_recheck", None)
        if callable(request_recheck):
            request_recheck("battle_ended")
        elif scheduler is not None:
            # Keep lightweight/test embeddings safe when they provide only a
            # scheduler and not the full campaign controller. The normal mode
            # path invalidates through the controller so the pending
            # ownership boundary and scheduler cache stay synchronized.
            scheduler.invalidate("battle_ended")
        # The battle listener owns the immediate return-to-field boundary.
        # Keep a second, mode-level event so special battle strategies and a
        # capture/recovery handoff cannot skip the overworld lead policy.
        self._lead_handoff_pending = True

    def _observe_campaign_boundary(self, controller) -> None:
        """Invalidate campaign planning when a coarse ROM boundary changes."""

        current = self._cheap_readiness_context()
        if not getattr(self, "_campaign_boundary_context_seen", False):
            self._campaign_boundary_context_seen = True
            self._last_campaign_boundary_context = current
            return

        previous = self._last_campaign_boundary_context
        self._last_campaign_boundary_context = current
        if current == previous:
            return

        reason = "campaign_boundary_changed"
        if isinstance(previous, tuple) and isinstance(current, tuple):
            if len(previous) > 1 and len(current) > 1 and previous[1] != current[1]:
                reason = "map_changed"
            elif len(previous) > 0 and len(current) > 0 and previous[0] != current[0]:
                reason = "game_state_changed"
            elif len(previous) > 2 and len(current) > 2 and previous[2] != current[2]:
                reason = "control_boundary_changed"
            elif len(previous) > 3 and len(current) > 3 and previous[3] != current[3]:
                reason = "pokeball_inventory_changed"
            elif len(previous) > 4 and len(current) > 4 and previous[4] != current[4]:
                reason = "party_state_changed"
            elif len(previous) > 5 and len(current) > 5 and previous[5] != current[5]:
                reason = "inventory_changed"
            elif len(previous) > 6 and len(current) > 6 and previous[6] != current[6]:
                reason = "campaign_fact_changed"

        request_recheck = getattr(controller, "request_readiness_recheck", None)
        request_refresh = getattr(controller, "request_refresh", None)
        if callable(request_recheck):
            request_recheck(reason)
        elif callable(request_refresh):
            request_refresh(reason)

    def run(self) -> Generator:
        """Yield frame boundaries while the campaign controller remains active."""

        # CampaignProgression owns campaign intent from the first observation.
        # The selected objective's executor owns frame-local emulator details.
        previous = None
        previous_phase = None
        while True:
            controller = self._campaign_controller()
            self._observe_campaign_boundary(controller)
            state = controller.step()
            marker = (state.status, state.objective_id, state.reason)
            lead_handoff_pending = getattr(self, "_lead_handoff_pending", False)
            if (
                getattr(state, "execution_phase", "CAMPAIGN") == "CAMPAIGN"
                and state.status.value == "ready"
                and (
                    lead_handoff_pending
                    or previous_phase == "RECOVERY"
                    or (
                        previous is not None
                        and getattr(previous[0], "value", previous[0]) == "ready"
                        and previous[1] is not None
                        and state.objective_id is not None
                        and previous[1] != state.objective_id
                    )
                )
            ):
                reason = (
                    "battle_or_capture_completed"
                    if lead_handoff_pending
                    else "recovery_completed"
                    if previous_phase == "RECOVERY"
                    else "campaign_objective_handoff"
                )
                self._lead_handoff_pending = False
                try:
                    leveling_strategy = NuzlockeLevelBalancingBattleStrategy()
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
                    # Opening campaign objectives can hand off before the
                    # starter has been inserted into the party. The lead
                    # policy has nothing to enforce at that boundary; defer
                    # it until the next battle/recovery event.
                    diagnostic_print(
                        lambda: (
                            "CAMPAIGN_FIELD_LEAD_HANDOFF: "
                            f"rotated=False reason={reason!r} "
                            f"strategy_unavailable={type(error).__name__!r}"
                        ),
                        trace=True,
                    )
                else:
                    objective = getattr(getattr(state, "selection", None), "objective", None)
                    field_context = build_campaign_field_lead_context(
                        objective,
                        campaign_state=runtime_campaign_state(),
                    )
                    yield from ensure_campaign_field_lead(
                        leveling_strategy,
                        field_context,
                        reason=reason,
                    )
            if marker != previous:
                diagnostic_print(
                    f"CAMPAIGN: status={state.status.value} objective={state.objective_id!r} "
                    f"reason={state.reason!r}",
                    trace=True,
                )
                previous = marker
            previous_phase = getattr(state, "execution_phase", "CAMPAIGN")
            if state.status.value == "complete":
                context.campaign_status = CampaignStatus.complete_status()
            elif state.status.value != "ready" and state.objective_id is None:
                context.campaign_status = CampaignStatus(None, None, f"Campaign {state.status.value}")
            # Startup observations can be transiently unavailable while the
            # ROM is booting. Keep the campaign controller mounted so the
            # next frame can re-observe and plan; UNKNOWN/BLOCKED are not
            # emulator phases and are not ownership handoffs.
            if state.status.value in {"unknown", "blocked"}:
                yield
                continue
            if state.status.value != "ready":
                return
            yield
