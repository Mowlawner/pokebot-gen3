"""Small pure battle-model primitives shared by decision procedures.

The richer Gen III mechanics (stat stages, status, Bide, switching) still live
inside ``battle_expectiminimax``.  This module is the seed of the shared model:
only the pieces a solver needs that do not depend on a search object.  New
solvers should build on these, and the larger model can move here over time
without changing their contract.
"""

from __future__ import annotations

from dataclasses import dataclass

# Gen III critical hits deal double damage.
CRITICAL_MULTIPLIER = 2
DEFAULT_CRITICAL_CHANCE = 1 / 16


@dataclass(frozen=True, slots=True)
class DamageOutcome:
    """One branch of a single attack: its probability and HP damage (0 = miss)."""

    probability: float
    damage: int
    critical: bool = False


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def move_outcomes(
    damage_min: int,
    damage_max: int,
    accuracy: float = 1.0,
    *,
    critical_chance: float = DEFAULT_CRITICAL_CHANCE,
    critical_damage: int | None = None,
    points: int = 3,
) -> tuple[DamageOutcome, ...]:
    """Return the discrete outcome distribution of one attack.

    Damage rolls are approximated by ``points`` evenly spaced values between
    the minimum and maximum, each equally likely.  A miss deals no damage.  A
    critical hit deals ``critical_damage`` (default: double the maximum).  A
    move that deals no damage is a single zero-damage outcome.  Probabilities
    always sum to one.
    """

    if damage_max <= 0:
        return (DamageOutcome(1.0, 0),)
    damage_min = max(0, min(int(damage_min), int(damage_max)))
    damage_max = int(damage_max)
    accuracy = _clamp(accuracy)
    critical_chance = _clamp(critical_chance)
    if critical_damage is None:
        critical_damage = damage_max * CRITICAL_MULTIPLIER
    critical_damage = max(int(critical_damage), damage_max)

    if damage_min == damage_max or points <= 1:
        rolls = (damage_max,)
    else:
        rolls = tuple(
            sorted(
                {
                    damage_min + round(index * (damage_max - damage_min) / (points - 1))
                    for index in range(points)
                }
            )
        )

    outcomes: list[DamageOutcome] = []
    miss = 1.0 - accuracy
    if miss > 0:
        outcomes.append(DamageOutcome(miss, 0))
    hit = accuracy * (1.0 - critical_chance)
    for roll in rolls:
        outcomes.append(DamageOutcome(hit / len(rolls), roll))
    critical = accuracy * critical_chance
    if critical > 0:
        outcomes.append(DamageOutcome(critical, critical_damage, True))
    return tuple(outcome for outcome in outcomes if outcome.probability > 0)


__all__ = [
    "CRITICAL_MULTIPLIER",
    "DEFAULT_CRITICAL_CHANCE",
    "DamageOutcome",
    "move_outcomes",
]
