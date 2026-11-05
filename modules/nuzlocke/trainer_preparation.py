"""Pure trainer-source selection for campaign preparation.

This layer answers whether a trainer is useful preparation.  It does not send
inputs or predict a particular miss, critical hit, or secondary effect; those
remain the battle-readiness engine's stochastic branches.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.nuzlocke.experience import gen3_battle_experience_reward
from modules.nuzlocke.level_cap import experience_remaining_in_cap
from modules.trainer_readiness import assess_live_trainer_checkpoint
from modules.pokemon import get_species_by_name


@dataclass(frozen=True, slots=True)
class TrainerSource:
    trainer_id: str
    checkpoint: object
    distance: int = 0
    route_simplification: int = 0
    gym_trainer: bool = False
    mandatory: bool = False


@dataclass(frozen=True, slots=True)
class PreparationExpBudget:
    """EXP ledger used before a boss or an unavoidable trainer sequence."""

    target_level: int | None
    level_cap: int | None
    required_exp: int | None
    unavoidable_exp: int | None
    cap_headroom: int | None

    @property
    def remaining_deficit(self) -> int | None:
        if self.required_exp is None:
            return None
        return max(0, self.required_exp - self.unavoidable_exp)

    @property
    def mandatory_exp_is_legal(self) -> bool | None:
        if self.cap_headroom is None or self.unavoidable_exp is None:
            return None
        return self.cap_headroom >= self.unavoidable_exp


@dataclass(frozen=True, slots=True)
class TrainerSourceEvaluation:
    source: TrainerSource
    readiness: object | None
    projected_exp: int | None
    best_recipient: int | None
    recipient_headroom: int | None
    cap_safe: bool | None
    useful_exp: int
    selected: bool = False
    rejection_reason: str | None = None

    @property
    def win_probability(self) -> float:
        return float(getattr(self.readiness, "win_probability", 0.0) or 0.0)

    @property
    def catastrophic_probability(self) -> float:
        return float(getattr(self.readiness, "catastrophic_probability", 1.0) or 0.0)

    @property
    def safe(self) -> bool:
        return (
            getattr(getattr(self.readiness, "status", None), "value", None) == "ready"
            and self.cap_safe is not False
        )

    @property
    def rank(self) -> tuple:
        """Rank survivability before utility, then utility before travel cost."""

        return (
            self.safe,
            self.win_probability,
            -self.catastrophic_probability,
            self.source.mandatory,
            self.source.route_simplification,
            self.source.gym_trainer,
            self.useful_exp,
            -self.source.distance,
            self.source.trainer_id,
        )


def projected_trainer_experience(checkpoint) -> int | None:
    """Return the total Gen III trainer reward for a verified roster."""

    total = 0
    try:
        for member in getattr(checkpoint, "party", ()):
            species = get_species_by_name(member.species)
            total += gen3_battle_experience_reward(
                species.base_experience_yield,
                int(member.level),
                trainer_battle=True,
            )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return total


def _cap_assignment(party, projected_exp: int, level_cap: int | None):
    if level_cap is None:
        return None, None, None
    candidates = []
    for index, pokemon in enumerate(party or ()):
        if getattr(pokemon, "is_egg", False) or getattr(pokemon, "current_hp", 0) <= 0:
            continue
        headroom = experience_remaining_in_cap(pokemon, level_cap)
        if headroom is None:
            return None, None, None
        candidates.append((headroom, index))
    if not candidates:
        return None, None, False
    candidates.sort(reverse=True)
    headroom, index = candidates[0]
    # EXP recipients can be deliberately rotated by the battle strategy.  A
    # trainer reward is therefore absorbable when the party's combined known
    # headroom is sufficient, while the largest single headroom identifies the
    # best initial recipient.
    return index, headroom, sum(item[0] for item in candidates) >= projected_exp


def build_exp_budget(
    party,
    *,
    target_level: int | None,
    level_cap: int | None,
    unavoidable_checkpoints=(),
) -> PreparationExpBudget:
    """Build the EXP ledger without selecting a battle or mutating state."""

    required = 0
    headroom = 0
    for pokemon in party or ():
        if getattr(pokemon, "is_egg", False) or getattr(pokemon, "current_hp", 0) <= 0:
            continue
        try:
            if target_level is not None and pokemon.level < target_level:
                required += max(0, pokemon.species.level_up_type.get_experience_needed_for_level(target_level) - pokemon.total_exp)
            if level_cap is not None:
                available = experience_remaining_in_cap(pokemon, level_cap)
                if available is None:
                    headroom = None
                elif headroom is not None:
                    headroom += available
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            required = None
            headroom = None
    unavoidable = 0
    for checkpoint in unavoidable_checkpoints:
        reward = projected_trainer_experience(checkpoint)
        if reward is None:
            unavoidable = None
            break
        unavoidable += reward
    return PreparationExpBudget(target_level, level_cap, required, unavoidable, headroom)


def evaluate_trainer_source(
    source: TrainerSource,
    party,
    *,
    target_level: int | None = None,
    level_cap: int | None = None,
) -> TrainerSourceEvaluation:
    projected_exp = projected_trainer_experience(source.checkpoint)
    if projected_exp is None:
        return TrainerSourceEvaluation(source, None, None, None, None, False, 0, rejection_reason="trainer EXP is unknown")
    try:
        readiness = assess_live_trainer_checkpoint(source.checkpoint, party, expected_exp=projected_exp)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        readiness = None
    recipient, headroom, cap_safe = _cap_assignment(party, projected_exp, level_cap)
    if target_level is None:
        useful_exp = projected_exp
    else:
        deficits = []
        for pokemon in party or ():
            if getattr(pokemon, "is_egg", False) or getattr(pokemon, "current_hp", 0) <= 0:
                continue
            try:
                if pokemon.level < target_level:
                    deficits.append(
                        max(
                            0,
                            pokemon.species.level_up_type.get_experience_needed_for_level(target_level)
                            - pokemon.total_exp,
                        )
                    )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                deficits = []
                break
        useful_exp = min(projected_exp, max(deficits, default=0))
    rejection = None
    if readiness is None or getattr(getattr(readiness, "status", None), "value", None) != "ready":
        rejection = "battle line is not proven safe"
    elif cap_safe is False:
        rejection = "no living party member can absorb the projected EXP below the cap"
    return TrainerSourceEvaluation(
        source,
        readiness,
        projected_exp,
        recipient,
        headroom,
        cap_safe,
        useful_exp,
        rejection_reason=rejection,
    )


def select_trainer_source(
    sources,
    party,
    *,
    target_level: int | None = None,
    level_cap: int | None = None,
) -> tuple[TrainerSourceEvaluation | None, tuple[TrainerSourceEvaluation, ...]]:
    """Evaluate and select the safest useful trainer source.

    Unknown or unsafe candidates remain in the returned diagnostics but cannot
    be selected.  If cap facts are unavailable, a source is retained for
    diagnostics and rejected by the hard cap policy.
    """

    evaluations = tuple(
        evaluate_trainer_source(source, party, target_level=target_level, level_cap=level_cap)
        for source in sources
    )
    eligible = tuple(item for item in evaluations if item.safe and item.cap_safe is not None)
    selected = max(eligible, key=lambda item: item.rank, default=None)
    if selected is None:
        return None, evaluations
    return (
        TrainerSourceEvaluation(
            selected.source,
            selected.readiness,
            selected.projected_exp,
            selected.best_recipient,
            selected.recipient_headroom,
            selected.cap_safe,
            selected.useful_exp,
            selected=True,
        ),
        evaluations,
    )


__all__ = [
    "PreparationExpBudget",
    "TrainerSource",
    "TrainerSourceEvaluation",
    "evaluate_trainer_source",
    "projected_trainer_experience",
    "build_exp_budget",
    "select_trainer_source",
]
