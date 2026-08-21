"""Autonomous execution of the small ordered Emerald campaign."""

from typing import Generator

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
        overworld_availability = Availability.KNOWN
        overworld_reason = None
        if isinstance(overworld, OverworldObservationResult):
            overworld_availability = Availability.UNKNOWN
            overworld_reason = overworld.reason
            overworld = None
        resources = observe_resource_snapshot()
        resource_availability = (
            Availability.KNOWN
            if resources.observation_status is ResourceObservationStatus.VALID
            else Availability.UNKNOWN
        )
        recovery = observe_route_recovery()
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
        )

    def on_battle_started(self, encounter) -> BattleAction:
        """Hand campaign battles to the existing default battle controller."""
        scheduler = getattr(self, "_readiness_scheduler", None)
        if scheduler is not None:
            scheduler.invalidate("battle_started")
        controller = getattr(self, "controller", None)
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
