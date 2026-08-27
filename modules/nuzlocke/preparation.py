"""Pure, reproducible preparation decisions for campaign progression."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random

from .campaign_state import FactStatus
from .emerald_campaign_registry import EmeraldBossDefinition
from modules.goals import EngageTrainer, EncounterMode, GoalConstraints, NavigationGoal, ReachLocation, TrainerMode


def discover_trainer_ids(observation) -> tuple[str, ...]:
    """Return stable, actionable trainer affordance IDs from one overworld view."""
    triggers = getattr(observation, "triggers", ()) if observation is not None else ()
    ids = {
        trigger.affordance_id
        for trigger in triggers
        if getattr(trigger, "hazard_locations", ())
        and getattr(trigger, "affordance_id", None)
        and getattr(trigger, "currently_actionable", True) is not False
    }
    return tuple(sorted(ids))


class PreparationStrategy(str, Enum):
    WILD_FIRST = "wild_first"
    TRAINERS_FIRST = "trainers_first"
    BALANCED = "balanced"


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True, slots=True)
class PreparationInput:
    boss: EmeraldBossDefinition | None
    party_levels: tuple[int, ...]
    party_hp_ratios: tuple[float, ...] = ()
    wild_areas: tuple[tuple[int, int], ...] = ()
    active_cap: int | None = None
    risk_tolerance: RiskTolerance = RiskTolerance.NORMAL
    seed: int = 0
    trainer_ids: tuple[str, ...] = ()
    preferred_strategy: PreparationStrategy | None = None


@dataclass(frozen=True, slots=True)
class PreparationDecision:
    status: FactStatus
    strategy: PreparationStrategy | None
    boss_id: str | None
    target_level: int | None
    wild_area: tuple[int, int] | None
    risk_tolerance: RiskTolerance
    seed: int
    reason: str
    trainer_id: str | None = None
    encounter_mode: EncounterMode = EncounterMode.IGNORE
    trainer_mode: TrainerMode = TrainerMode.IGNORE

    @property
    def required(self) -> bool:
        return self.status is FactStatus.KNOWN and self.strategy is not None and self.target_level is not None

    def navigation_goal(self) -> NavigationGoal | None:
        if not self.required:
            return None
        if self.strategy is PreparationStrategy.TRAINERS_FIRST and self.trainer_id is not None:
            return NavigationGoal(
                EngageTrainer(self.trainer_id),
                constraints=GoalConstraints(trainer_mode=TrainerMode.ENGAGE),
                encounter_mode=self.encounter_mode,
            )
        if self.wild_area is not None:
            return NavigationGoal(
                ReachLocation(self.wild_area),
                constraints=GoalConstraints(trainer_mode=self.trainer_mode),
                encounter_mode=self.encounter_mode,
            )
        return None


def decide_preparation(request: PreparationInput) -> PreparationDecision:
    """Choose a bounded wild-training target without reading or mutating runtime state."""
    if request.boss is None:
        return PreparationDecision(
            FactStatus.KNOWN, None, None, None, None, request.risk_tolerance, request.seed, "no active boss"
        )
    if not request.party_levels:
        return PreparationDecision(
            FactStatus.UNKNOWN,
            None,
            request.boss.boss_id,
            None,
            None,
            request.risk_tolerance,
            request.seed,
            "party levels are unavailable",
        )
    cap = request.active_cap if request.active_cap is not None else request.boss.level_cap
    upper = min(request.boss.level_cap, cap)
    lower = min(upper, 7)
    target = random.Random(request.seed).randint(lower, min(upper, 8))
    current = max(request.party_levels)
    if current >= target:
        return PreparationDecision(
            FactStatus.KNOWN,
            None,
            request.boss.boss_id,
            target,
            None,
            request.risk_tolerance,
            request.seed,
            f"party already meets bounded target {target}",
        )
    strategy = request.preferred_strategy
    if strategy is None:
        strategy = PreparationStrategy.TRAINERS_FIRST if request.trainer_ids else PreparationStrategy.WILD_FIRST
    if strategy is PreparationStrategy.TRAINERS_FIRST and request.trainer_ids:
        trainer = request.trainer_ids[random.Random(request.seed).randrange(len(request.trainer_ids))]
        return PreparationDecision(
            FactStatus.KNOWN, strategy, request.boss.boss_id, target, None, request.risk_tolerance, request.seed,
            f"train selected trainer {trainer!r} to bounded target {target} before {request.boss.display_name}",
            trainer, EncounterMode.IGNORE, TrainerMode.ENGAGE,
        )
    if not request.wild_areas:
        return PreparationDecision(
            FactStatus.KNOWN,
            PreparationStrategy.WILD_FIRST,
            request.boss.boss_id,
            target,
            None,
            request.risk_tolerance,
            request.seed,
            "party needs training but no safe wild area is available",
            None, EncounterMode.SEEK, TrainerMode.IGNORE,
        )
    area = request.wild_areas[random.Random(request.seed).randrange(len(request.wild_areas))]
    return PreparationDecision(
        FactStatus.KNOWN,
        PreparationStrategy.WILD_FIRST,
        request.boss.boss_id,
        target,
        area,
        request.risk_tolerance,
        request.seed,
        f"train wild encounters to bounded target {target} before {request.boss.display_name}",
        None, EncounterMode.SEEK, TrainerMode.IGNORE,
    )
