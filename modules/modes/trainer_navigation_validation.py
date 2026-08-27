"""Validation-only trainer engagement controller."""

from typing import Generator

from modules.agent_control import AgentControlLoop, observe_agent
from modules.console import diagnostic_print
from modules.goals import EngageTrainer
from modules.memory import GameState, get_game_state
from modules.overworld import perceive_overworld

from ._interface import BattleAction, BotMode


class TrainerNavigationValidationMode(BotMode):
    """Mount an observed undefeated trainer as a one-shot tactical goal."""

    @staticmethod
    def name() -> str:
        return "Trainer Navigation Validation"

    @staticmethod
    def is_selectable() -> bool:
        from modules.context import context

        return context.rom is not None and context.rom.is_emerald

    def run(self) -> Generator:
        from modules.context import context

        while True:
            if get_game_state() is not GameState.OVERWORLD:
                yield
                continue
            observation = perceive_overworld()
            if not observation.controllable:
                yield
                continue
            trainer = next(
                (obj for obj in observation.objects if obj.trainer_id is not None and obj.trainer_defeated is False),
                None,
            )
            if trainer is None:
                diagnostic_print("TRAINER_VALIDATION_COMPLETE: no undefeated trainer remains", trace=True)
                context.set_manual_mode(enable_video_and_slow_down=False)
                return
            goal = EngageTrainer(trainer.trainer_id)
            diagnostic_print(
                lambda: (
                    "TRAINER_VALIDATION_START: "
                    f"trainer_id={trainer.trainer_id!r} map={trainer.location[0]!r} "
                    f"coordinates={trainer.location[1]!r} range={trainer.trainer_range!r}"
                ),
                trace=True,
            )
            loop = AgentControlLoop(lambda: observe_agent(goal=goal), goal=goal).run()
            yield from loop
            diagnostic_print("TRAINER_VALIDATION_COMPLETE: engagement loop returned", trace=True)
            context.set_manual_mode(enable_video_and_slow_down=False)
            return
