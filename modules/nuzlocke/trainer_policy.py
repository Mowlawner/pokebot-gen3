"""Pure trainer-mode policy for campaign navigation."""

from dataclasses import dataclass

from modules.goals import TrainerMode


@dataclass(frozen=True, slots=True)
class TrainerPolicyInput:
    """Observed considerations for one navigation objective."""

    seeking_encounter: bool = False
    trainer_required: bool = False
    trainer_blocks_route: bool = False
    party_can_survive: bool | None = None
    experience_needed: bool = False


def choose_trainer_mode(value: TrainerPolicyInput) -> TrainerMode:
    """Choose whether navigation avoids or engages trainers."""

    if value.trainer_required:
        return TrainerMode.ENGAGE
    if value.seeking_encounter:
        # Encounter acquisition has priority; trainers can be scheduled as a
        # separate experience objective after the first legal encounter.
        return TrainerMode.AVOID if not value.trainer_blocks_route else TrainerMode.IGNORE
    if value.experience_needed and value.party_can_survive is True:
        return TrainerMode.ENGAGE
    if value.party_can_survive is False:
        return TrainerMode.AVOID
    return TrainerMode.IGNORE
