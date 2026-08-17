"""Autonomous execution of the small ordered Emerald campaign."""

from typing import Generator

from modules.context import context
from modules.console import diagnostic_print
from modules.nuzlocke.campaign_controller import CampaignController, runtime_campaign_boundary, runtime_campaign_state

from ._interface import BotMode


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
            if state.status.value != "ready":
                return
            yield
