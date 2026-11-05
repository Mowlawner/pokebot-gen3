"""Name the kind of battle before choosing how to decide it.

Different battles need different decision procedures: a trainer fight is an
adversarial search, a wild capture is a single-agent problem with chance, and a
forced flee needs no search.  This module only classifies.  It reads facts that
the campaign and the Nuzlocke rules have already decided (is this a legal first
encounter, is a ball available, what is the campaign's intent) and never
decides them itself, so a solver never has to reason about whether capturing is
legal.

The result is derived from the same objective string the evaluator already
uses, so classifying changes no existing behavior; it makes the choice named,
loggable, and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BattleKind(str, Enum):
    SCRIPTED_DEFEAT = "scripted_defeat"
    TRAINER = "trainer"
    WILD_CAPTURE = "wild_capture"
    WILD_TRAINING = "wild_training"
    WILD_FLEE = "wild_flee"
    WILD_DEFEAT = "wild_defeat"


@dataclass(frozen=True, slots=True)
class BattleClassification:
    kind: BattleKind
    reason: str


def classify_battle(
    *,
    objective: str | None,
    is_trainer: bool,
    first_battle: bool = False,
    capture_target: bool = False,
    capture_available: bool = False,
    intent: str | None = None,
) -> BattleClassification:
    """Classify one battle boundary.

    ``objective`` is the normalized objective (``capture``, ``defeat``,
    ``training``, ``avoid``, ``transit``).  The other arguments explain it.
    """

    if first_battle:
        return BattleClassification(
            BattleKind.SCRIPTED_DEFEAT,
            "FirstBattle flag: a mandatory defeat boundary before capture or escape is legal",
        )
    if is_trainer:
        return BattleClassification(BattleKind.TRAINER, "trainer battle")
    if objective == "capture":
        return BattleClassification(
            BattleKind.WILD_CAPTURE,
            "legal first encounter with at least one usable ball",
        )
    if capture_target and not capture_available:
        return BattleClassification(
            BattleKind.WILD_DEFEAT,
            "legal capture target but no usable ball, so capture is not an available objective",
        )
    if objective in ("avoid", "transit"):
        return BattleClassification(BattleKind.WILD_FLEE, f"campaign intent is {objective!r}")
    if objective == "training":
        return BattleClassification(BattleKind.WILD_TRAINING, "campaign requested preparation (training)")
    return BattleClassification(
        BattleKind.WILD_DEFEAT,
        f"no capture or flee objective (intent={intent or objective or 'default'!r})",
    )


__all__ = ["BattleClassification", "BattleKind", "classify_battle"]
