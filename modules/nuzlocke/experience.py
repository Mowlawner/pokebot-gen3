"""Pure Generation III battle experience calculations.

The battle strategy uses this module for projections before an opponent is
defeated.  The emulator's post-battle party values remain authoritative when
the battle has finished, since that also covers ROM-specific modifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ExperienceDistribution:
    """Projected EXP award for one defeated opposing Pokémon."""

    base_reward: int
    participant_pool: int
    exp_share_pool: int
    participant_count: int
    exp_share_count: int
    awards: tuple[tuple[int, int], ...]

    @property
    def total_awarded(self) -> int:
        """Return the total integer EXP distributed to the party."""

        return sum(amount for _, amount in self.awards)

    def award_for(self, party_index: int) -> int:
        """Return the projected award for one party slot."""

        return dict(self.awards).get(party_index, 0)


@dataclass(frozen=True, slots=True)
class TrainingCandidate:
    """One party member's direct-training opportunity."""

    party_index: int
    level: int
    target_level: int | None = None
    hp_ratio: float = 1.0
    matchup_score: float = 0.0
    safe: bool = True

    @property
    def training_deficit(self) -> int:
        """Return levels still needed for the assigned checkpoint role."""

        if self.target_level is None:
            return 0
        return max(0, int(self.target_level) - int(self.level))


@dataclass(frozen=True, slots=True)
class TrainingLeadDecision:
    """Direct-lead choice for one training boundary."""

    party_index: int | None
    reason: str


def choose_training_lead(candidates: Iterable[TrainingCandidate]) -> TrainingLeadDecision:
    """Choose one direct trainee without routine EXP tagging.

    The largest checkpoint deficit wins while the matchup remains safe. Once
    all members meet their assigned target, matchup and health determine the
    field lead. This keeps experience concentrated on one useful recipient.
    """

    usable = tuple(candidate for candidate in candidates if candidate.safe and candidate.hp_ratio > 0)
    if not usable:
        return TrainingLeadDecision(None, "no safe living training candidate")
    deficient = tuple(candidate for candidate in usable if candidate.training_deficit > 0)
    if deficient:
        selected = max(
            deficient,
            key=lambda candidate: (
                candidate.training_deficit,
                candidate.matchup_score,
                candidate.hp_ratio,
                -candidate.party_index,
            ),
        )
        return TrainingLeadDecision(
            selected.party_index,
            f"directly train party slot {selected.party_index} with deficit {selected.training_deficit}",
        )
    selected = max(
        usable,
        key=lambda candidate: (
            candidate.matchup_score,
            candidate.hp_ratio,
            -candidate.party_index,
        ),
    )
    return TrainingLeadDecision(
        selected.party_index,
        f"all assigned training deficits are satisfied; use matchup lead {selected.party_index}",
    )


def gen3_battle_experience_reward(
    base_experience_yield: int,
    opponent_level: int,
    *,
    trainer_battle: bool,
) -> int:
    """Return the Generation III reward before participant splitting.

    Generation III uses ``base yield * defeated level / 7`` and applies the
    trainer-battle 3/2 multiplier. Integer truncation is intentional.
    Lucky Egg, traded-Pokémon, and other recipient-specific modifiers are
    applied by :func:`distribute_gen3_experience`.
    """

    if base_experience_yield <= 0 or opponent_level <= 0:
        return 0
    reward = (base_experience_yield * opponent_level) // 7
    if trainer_battle:
        reward = (reward * 3) // 2
    return max(0, reward)


def distribute_gen3_experience(
    base_reward: int,
    participants: Iterable[int],
    exp_share_holders: Iterable[int] = (),
    *,
    lucky_egg_holders: Iterable[int] = (),
    recipient_multipliers: Mapping[int, tuple[int, int]] | None = None,
) -> ExperienceDistribution:
    """Project one Generation III EXP award across party recipients.

    Without Exp. Share, the reward is divided equally among Pokémon that
    participated. With Exp. Share, half is divided among participants and
    half among Exp. Share holders. A holder that also participated receives
    both portions, matching the Generation III split model.

    ``recipient_multipliers`` is an optional numerator/denominator mapping
    for recipient-specific ROM modifiers. It is deliberately separate from
    the base formula so callers can keep unknown modifiers explicit.
    """

    participant_indices = tuple(dict.fromkeys(int(index) for index in participants))
    share_indices = tuple(dict.fromkeys(int(index) for index in exp_share_holders))
    lucky_indices = set(int(index) for index in lucky_egg_holders)
    base_reward = max(0, int(base_reward))

    if not participant_indices and not share_indices:
        return ExperienceDistribution(base_reward, 0, 0, 0, 0, ())

    if share_indices:
        # The ROM performs integer division for each half of the award.
        participant_pool = base_reward // 2
        exp_share_pool = base_reward // 2
    else:
        participant_pool = base_reward
        exp_share_pool = 0

    participant_each = (
        participant_pool // len(participant_indices) if participant_indices else 0
    )
    share_each = exp_share_pool // len(share_indices) if share_indices else 0
    recipients = set(participant_indices) | set(share_indices)
    awards: list[tuple[int, int]] = []
    for index in sorted(recipients):
        amount = (participant_each if index in participant_indices else 0) + (
            share_each if index in share_indices else 0
        )
        if index in lucky_indices:
            amount = (amount * 3) // 2
        if recipient_multipliers and index in recipient_multipliers:
            numerator, denominator = recipient_multipliers[index]
            if denominator > 0:
                amount = (amount * max(0, numerator)) // denominator
        awards.append((index, amount))

    return ExperienceDistribution(
        base_reward,
        participant_pool,
        exp_share_pool,
        len(participant_indices),
        len(share_indices),
        tuple(awards),
    )
