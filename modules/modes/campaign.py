"""Autonomous execution of the small ordered Emerald campaign."""

from typing import Generator

from modules.context import context
from modules.console import diagnostic_print
from modules.nuzlocke.campaign_controller import CampaignController, runtime_campaign_boundary, runtime_campaign_state
from modules.nuzlocke.campaign_status import CampaignStatus

from ._interface import BattleAction, BotMode
from modules.nuzlocke.resource_policy import (
    PartyResource,
    ResourceDecision,
    ResourceSnapshot,
    RouteRecovery,
    assess_wild_encounter,
)
from modules.pokemon_party import get_party


class CampaignProgressionMode(BotMode):
    @staticmethod
    def name() -> str:
        return "Campaign Progression"

    @staticmethod
    def is_selectable() -> bool:
        return context.rom is not None and context.rom.is_emerald

    def __init__(self):
        self.controller = CampaignController(
            runtime_campaign_state,
            campaign_boundary_handler=runtime_campaign_boundary,
        )

    def on_battle_started(self, encounter) -> BattleAction:
        """Hand campaign battles to the existing default battle controller."""
        if encounter is not None and self.controller.last_selection is not None:
            objective = self.controller.last_selection.objective
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

    def run(self) -> Generator:
        # CampaignProgression remains the owner from a fresh save onward.
        # Emerald-specific work is mounted by campaign_execution capabilities;
        # no other mode is entered at an objective boundary.
        previous = None
        while True:
            state = self.controller.step()
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
            if state.status.value != "ready":
                return
            yield
