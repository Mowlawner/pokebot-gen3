from types import SimpleNamespace

from modules.nuzlocke.experience import (
    distribute_gen3_experience,
    gen3_battle_experience_reward,
)
from modules.nuzlocke.level_cap import experience_remaining_in_cap


def test_gen3_trainer_reward_applies_the_trainer_multiplier():
    assert gen3_battle_experience_reward(50, 5, trainer_battle=False) == 35
    assert gen3_battle_experience_reward(50, 5, trainer_battle=True) == 52


def test_gen3_experience_is_split_between_participants():
    distribution = distribute_gen3_experience(100, participants=(0, 1))

    assert distribution.awards == ((0, 50), (1, 50))
    assert distribution.total_awarded == 100


def test_switching_in_another_participant_reduces_the_first_award():
    one_participant = distribute_gen3_experience(100, participants=(0,))
    two_participants = distribute_gen3_experience(100, participants=(0, 1))

    assert one_participant.award_for(0) == 100
    assert two_participants.award_for(0) == 50
    assert two_participants.award_for(1) == 50


def test_gen3_exp_share_splits_a_second_pool_and_participating_holder_gets_both():
    non_participating_holder = distribute_gen3_experience(
        100,
        participants=(0,),
        exp_share_holders=(1,),
    )
    participating_holder = distribute_gen3_experience(
        100,
        participants=(0, 1),
        exp_share_holders=(1,),
    )

    assert non_participating_holder.awards == ((0, 50), (1, 50))
    assert participating_holder.awards == ((0, 25), (1, 75))


def test_lucky_egg_modifier_applies_to_recipient_award():
    distribution = distribute_gen3_experience(100, participants=(0,), lucky_egg_holders=(0,))

    assert distribution.award_for(0) == 150


def test_experience_remaining_in_cap_is_species_growth_aware():
    growth = SimpleNamespace(get_experience_needed_for_level=lambda level: {15: 3000, 16: 4000}[level])
    pokemon = SimpleNamespace(total_exp=3900, species=SimpleNamespace(level_up_type=growth))

    assert experience_remaining_in_cap(pokemon, 15) == 99
