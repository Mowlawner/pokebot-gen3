"""Pure trainer exposure policy for campaign navigation."""

from dataclasses import dataclass
from enum import Enum

from modules.goals import TrainerMode
from modules.battle_readiness import BattleReadiness


class TrainerExposureKind(str, Enum):
    """Route relationship between a trainer and the current campaign task."""

    MANDATORY = "mandatory"
    COSTLY_TO_AVOID = "costly_to_avoid"
    OPTIONAL = "optional"
    UNKNOWN = "unknown"


class TrainerCheckpointKind(str, Enum):
    """Campaign ownership of a trainer battle."""

    STORY = "story"
    MANDATORY_BATTLE = "mandatory_battle"
    STRATEGIC = "strategic"
    HAZARD = "hazard"


@dataclass(frozen=True, slots=True)
class TrainerPolicyInput:
    """Observed considerations for one navigation objective."""

    seeking_encounter: bool = False
    trainer_required: bool = False
    trainer_blocks_route: bool = False
    party_can_survive: bool | None = None
    experience_needed: bool = False
    battle_readiness: BattleReadiness | None = None
    exposure: TrainerExposureKind = TrainerExposureKind.OPTIONAL
    checkpoint_kind: TrainerCheckpointKind = TrainerCheckpointKind.HAZARD
    strategic_promotion: bool = False


def choose_trainer_mode(value: TrainerPolicyInput) -> TrainerMode:
    """Choose whether navigation avoids or engages trainers."""

    battle_ready = value.battle_readiness.ready if value.battle_readiness is not None else value.party_can_survive

    mandatory = value.trainer_required or value.exposure is TrainerExposureKind.MANDATORY or value.checkpoint_kind in {
        TrainerCheckpointKind.STORY,
        TrainerCheckpointKind.MANDATORY_BATTLE,
    }
    if mandatory:
        if battle_ready is False:
            # The checkpoint remains part of the route, but the tactical
            # owner must prepare before navigation is allowed to engage it.
            return TrainerMode.AVOID
        return TrainerMode.ENGAGE
    if value.seeking_encounter and not value.strategic_promotion:
        # Encounter acquisition has priority; trainers can be scheduled as a
        # separate preparation/engagement objective after the first legal
        # encounter.  If avoidance cannot execute, navigation must report a
        # blocked route so campaign readiness can identify the actual trainer.
        return TrainerMode.AVOID
    promoted = value.strategic_promotion or (
        value.exposure is TrainerExposureKind.COSTLY_TO_AVOID and value.experience_needed
    )
    if promoted:
        if battle_ready is True:
            return TrainerMode.ENGAGE
        return TrainerMode.AVOID
    if value.experience_needed and battle_ready is True:
        return TrainerMode.ENGAGE
    if battle_ready is False:
        return TrainerMode.AVOID
    return TrainerMode.IGNORE
