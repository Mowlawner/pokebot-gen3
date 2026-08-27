"""Autonomous execution of the small ordered Emerald campaign."""

from typing import Generator
import traceback

from modules.context import context
from modules.console import diagnostic_print
from modules.nuzlocke.campaign_controller import CampaignController, runtime_campaign_boundary, runtime_campaign_state
from modules.nuzlocke.campaign_objectives import plan_campaign
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
    execute_campaign_recovery,
    observe_resource_snapshot,
    observe_route_recovery,
)
from modules.nuzlocke.level_cap import evaluate_battle_entry
from modules.modes._interface import BotModeError
from modules.nuzlocke.readiness_diagnostics import (
    Availability,
    ReadinessObservationScheduler,
    build_progression_readiness_diagnostic,
)
from modules.nuzlocke.snapshots import get_nuzlocke_snapshot
from modules.overworld import perceive_overworld, OverworldObservationResult
from modules.player import get_player_avatar
from modules.memory import GameState, get_game_state
from modules.pokemon_party import get_party
from modules.goals import ReachLocation
from modules.navigation import NavigationWorld, RouteCostAnalyzer
from modules.world_navigation import get_world_map_graph
from modules.modes.util.map import pokemon_center_candidates


class CampaignProgressionMode(BotMode):
    @staticmethod
    def name() -> str:
        return "Campaign Progression"

    @staticmethod
    def is_selectable() -> bool:
        return context.rom is not None and context.rom.is_emerald

    def __init__(self):
        self._readiness_scheduler = ReadinessObservationScheduler(
            self._readiness_input,
            self._cheap_readiness_context,
            max_age_ticks=15,
        )
        self.controller = CampaignController(
            runtime_campaign_state,
            selector=plan_campaign,
            campaign_boundary_handler=runtime_campaign_boundary,
            readiness_provider=self._readiness_scheduler.observe,
            recovery_factory=lambda _readiness: execute_campaign_recovery(),
        )
        self._readiness_evaluated = False

    def _campaign_controller(self):
        return self.controller

    @staticmethod
    def _cheap_readiness_context():
        try:
            avatar = get_player_avatar()
            return (
                getattr(get_game_state(), "name", None),
                avatar.map_group_and_number,
                avatar.local_coordinates,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

    def _readiness_input(self, objective, goal):
        self._readiness_evaluated = True
        snapshot = get_nuzlocke_snapshot()
        overworld = perceive_overworld()
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
        resources = observe_resource_snapshot()
        resource_availability = (
            Availability.KNOWN
            if resources.observation_status is ResourceObservationStatus.VALID
            else Availability.UNKNOWN
        )
        # Route recovery is only a policy input when the party may actually
        # need recovery.  Computing it here on every readiness refresh walks
        # the map synchronously, which stalls the active tactical controller
        # even when the party is healthy and already navigating normally.
        minimum_hp_ratio = getattr(getattr(objective, "resource_policy", None), "minimum_hp_ratio", 0.5)
        recovery_needed = (
            resources.observation_status is not ResourceObservationStatus.VALID
            or not resources.usable_party
            or resources.worst_hp_ratio < minimum_hp_ratio
        )
        recovery = observe_route_recovery() if recovery_needed else RouteRecovery()
        recovery_available = recovery.center_available or bool(resources.bag_healing_items)
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
                f"recovery_availability={getattr(recovery_availability, 'value', None)!r} center_available={recovery_center_available!r} "
                f"center_safe={recovery_safe!r} distance_to_center={recovery_distance!r} "
                "distance_metric='calculate_path step count' "
                f"nearest_center={getattr(recovery, 'center_location', None)!r}"
            ),
            trace=True,
        )
        route_analysis = None
        if overworld is None or goal is None or not snapshot.player_available:
            diagnostic_print(
                lambda: (
                    "ROUTE_ANALYSIS_CONSTRUCTION_FAILURE\n"
                    "exception_type=UnavailableContext\n"
                    "exception_message=route-analysis precondition unavailable\n"
                    "failing_operation=precondition\n"
                    f"active_tactical_goal={goal!r}\n"
                    f"goal_type={type(goal).__name__ if goal is not None else None}\n"
                    f"navigation_world_available={overworld is not None}\n"
                    f"world_graph_available=False\n"
                    "candidate_count=0\n"
                    "candidate_destinations=[]"
                ),
                trace=True,
            )
        if recovery_needed and overworld is not None and goal is not None and snapshot.player_available:
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
                candidates = tuple(
                    ReachLocation(center.value) for center in pokemon_center_candidates(current_location)
                )
                candidate_goals = candidates
                operation = "WorldMapGraph acquisition"
                graph = get_world_map_graph()
                operation = "RouteCostAnalyzer.analyze"
                route_analysis = RouteCostAnalyzer(world, graph=graph).analyze(current_location, goal, candidates)
            except (RuntimeError, TypeError, ValueError) as error:
                diagnostic_print(
                    lambda: (
                        "ROUTE_ANALYSIS_CONSTRUCTION_FAILURE\n"
                        f"exception_type={type(error).__name__}\n"
                        f"exception_message={error}\n"
                        f"failing_operation={operation}\n"
                        f"active_tactical_goal={goal!r}\n"
                        f"goal_type={type(goal).__name__ if goal is not None else None}\n"
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
        return build_progression_readiness_diagnostic(
            snapshot,
            objective_id=objective.objective_id if objective is not None else None,
            objective_status="ready",
            destination=getattr(objective, "destination", None),
            navigation_goal=goal,
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
        )

    def on_battle_started(self, encounter) -> BattleAction:
        """Choose campaign battle policy, leaving execution to the listener."""
        scheduler = getattr(self, "_readiness_scheduler", None)
        if scheduler is not None:
            scheduler.invalidate("battle_started")

        controller = getattr(self, "controller", None)
        objective = controller.last_selection.objective if controller is not None and controller.last_selection is not None else None
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
        if encounter is not None and controller is not None and controller.last_selection is not None:
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
        return BattleAction.Fight

    def on_battle_ended(self, outcome) -> None:
        scheduler = getattr(self, "_readiness_scheduler", None)
        if scheduler is not None:
            scheduler.invalidate("battle_ended")

    def run(self) -> Generator:
        # CampaignProgression owns campaign intent from the first observation.
        # The selected objective's executor owns frame-local emulator details.
        previous = None
        while True:
            controller = self._campaign_controller()
            state = controller.step()
            marker = (state.status, state.objective_id, state.reason)
            if marker != previous:
                diagnostic_print(
                    f"CAMPAIGN: status={state.status.value} objective={state.objective_id!r} "
                    f"reason={state.reason!r}",
                    trace=True,
                )
                previous = marker
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
