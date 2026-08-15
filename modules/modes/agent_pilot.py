"""Selectable composition of the generic agent loop for the Emerald pilot."""

from typing import Generator

from modules.agent_control import AgentControlLoop, observe_agent
from modules.console import diagnostic_print
from modules.context import context
from modules.goals import Goal, introductory_rival_goal

from ._interface import BotMode


class AgentRivalBattlePilot(BotMode):
    """Run the existing agent-control loop toward the introductory rival trigger."""

    @staticmethod
    def name() -> str:
        return "Agent Rival Battle Pilot"

    @staticmethod
    def is_selectable() -> bool:
        return context.rom is not None and context.rom.is_emerald

    def __init__(self):
        self.goal: Goal = introductory_rival_goal()
        self.control_loop = AgentControlLoop(
            lambda: observe_agent(goal=self.goal),
            goal=self.goal,
        )

    def run(self) -> Generator:
        diagnostic_print("AGENT_PILOT: Agent Rival Battle Pilot started", trace=True)
        diagnostic_print(f"AGENT_GOAL: {self.goal!r}", trace=True)
        yield from self.control_loop.run()
