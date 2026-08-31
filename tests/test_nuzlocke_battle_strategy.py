from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_strategies import TurnAction
from modules.battle_strategies.nuzlocke_level_balancing import (
    EmeraldIntroRivalBattleStrategy,
    NuzlockeLevelBalancingBattleStrategy,
)


def test_nuzlocke_replacement_excludes_the_fainted_active_slot():
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(own_side=SimpleNamespace(active_battlers=[active]))
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    with patch(
        "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
        return_value=(0,),
    ):
        assert not strategy.has_replacement_after_faint(battle_state)


def test_nuzlocke_replacement_accepts_an_eligible_non_fainted_slot():
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(own_side=SimpleNamespace(active_battlers=[active]))
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    with patch(
        "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
        return_value=(1,),
    ):
        assert strategy.has_replacement_after_faint(battle_state)


def test_emerald_intro_rival_opens_with_two_growls_before_attacking():
    party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
    growl = SimpleNamespace(name="Growl")
    active = SimpleNamespace(
        party_index=0,
        moves=(SimpleNamespace(move=growl, pp=40),),
        disabled_move=None,
    )
    battle_state = SimpleNamespace(
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active),
    )

    with patch("modules.battle_strategies.default.get_party", return_value=party):
        strategy = EmeraldIntroRivalBattleStrategy()

    with patch.object(EmeraldIntroRivalBattleStrategy.__mro__[1], "decide_turn") as fallback:
        assert strategy.decide_turn(battle_state) == (TurnAction.UseMove, 0)
        assert strategy.decide_turn(battle_state) == (TurnAction.UseMove, 0)
        fallback.assert_not_called()


def test_emerald_intro_rival_does_not_open_wild_encounters_with_growl():
    party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
    growl = SimpleNamespace(name="Growl")
    active = SimpleNamespace(
        party_index=0,
        moves=(SimpleNamespace(move=growl, pp=40),),
        disabled_move=None,
    )
    battle_state = SimpleNamespace(
        is_trainer_battle=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    expected = (TurnAction.UseMove, 0)

    with patch("modules.battle_strategies.default.get_party", return_value=party):
        strategy = EmeraldIntroRivalBattleStrategy()
    with patch.object(
        EmeraldIntroRivalBattleStrategy.__mro__[1],
        "decide_turn",
        return_value=expected,
    ) as fallback:
        assert strategy.decide_turn(battle_state) == expected
        fallback.assert_called_once_with(battle_state)
