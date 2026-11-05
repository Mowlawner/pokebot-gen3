from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_strategies import TurnAction
from modules.battle_strategies.default import DefaultBattleStrategy


def _move(name, power=10):
    return SimpleNamespace(name=name, base_power=power, accuracy=1.0)


def _learned(name, power=10):
    return SimpleNamespace(move=_move(name, power), pp=20)


def test_double_planner_assigns_guaranteed_kos_to_different_opponents():
    left = SimpleNamespace(
        party_index=0,
        current_hp=20,
        moves=(_learned("Scratch"),),
        disabled_move=None,
        disabled_turns_remaining=0,
    )
    right = SimpleNamespace(
        party_index=1,
        current_hp=20,
        moves=(_learned("Tackle"),),
        disabled_move=None,
        disabled_turns_remaining=0,
    )
    left_foe = SimpleNamespace(current_hp=5, species=SimpleNamespace(name="Zigzagoon"))
    right_foe = SimpleNamespace(current_hp=5, species=SimpleNamespace(name="Poochyena"))
    battle_state = SimpleNamespace(
        is_double_battle=True,
        own_side=SimpleNamespace(left_battler=left, right_battler=right),
        opponent=SimpleNamespace(left_battler=left_foe, right_battler=right_foe),
    )
    party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
    damage = SimpleNamespace(min=5, max=10)

    with (
        patch("modules.battle_strategies.default.get_party", return_value=party),
        patch("modules.battle_strategies._util.BattleStrategyUtil.calculate_move_damage_range", return_value=damage),
    ):
        strategy = DefaultBattleStrategy()
        plan = strategy._double_battle_plan(battle_state)

    assert plan[0][2] != plan[1][2]
    assert plan[0][0] in (TurnAction.UseMove, TurnAction.UseMoveAgainstRightSideOpponent)
    assert plan[1][0] in (TurnAction.UseMove, TurnAction.UseMoveAgainstRightSideOpponent)


def test_strict_strategy_uses_double_planner_instead_of_manual_stop():
    left = SimpleNamespace(
        party_index=0,
        current_hp=20,
        moves=(_learned("Scratch"),),
        disabled_move=None,
        disabled_turns_remaining=0,
    )
    right = SimpleNamespace(
        party_index=1,
        current_hp=20,
        moves=(_learned("Tackle"),),
        disabled_move=None,
        disabled_turns_remaining=0,
    )
    foe = SimpleNamespace(current_hp=5, species=SimpleNamespace(name="Zigzagoon"))
    battle_state = SimpleNamespace(
        is_double_battle=True,
        own_side=SimpleNamespace(left_battler=left, right_battler=right),
        opponent=SimpleNamespace(left_battler=foe, right_battler=None),
    )
    party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))

    with (
        patch("modules.battle_strategies.default.get_party", return_value=party),
        patch(
            "modules.battle_strategies._util.BattleStrategyUtil.calculate_move_damage_range",
            return_value=SimpleNamespace(min=5, max=10),
        ),
    ):
        strategy = DefaultBattleStrategy()
        action, move_index = strategy.decide_turn_in_double_battle(battle_state, 0)

    assert action in (TurnAction.UseMove, TurnAction.UseMoveAgainstRightSideOpponent)
    assert move_index == 0


def test_double_planner_preserves_a_threatened_partner_before_target_preferences():
    left = SimpleNamespace(
        party_index=0,
        current_hp=5,
        moves=(_learned("Scratch"),),
        stats=SimpleNamespace(speed=20),
        disabled_move=None,
        disabled_turns_remaining=0,
    )
    right = SimpleNamespace(
        party_index=1,
        current_hp=20,
        moves=(_learned("Tackle"),),
        stats=SimpleNamespace(speed=15),
        disabled_move=None,
        disabled_turns_remaining=0,
    )
    left_foe = SimpleNamespace(
        current_hp=5,
        species=SimpleNamespace(name="Zigzagoon"),
        moves=(_learned("Tackle"),),
        stats=SimpleNamespace(speed=10),
    )
    right_foe = SimpleNamespace(
        current_hp=5,
        species=SimpleNamespace(name="Poochyena"),
        moves=(_learned("Tackle"),),
        stats=SimpleNamespace(speed=10),
    )
    battle_state = SimpleNamespace(
        is_double_battle=True,
        own_side=SimpleNamespace(left_battler=left, right_battler=right),
        opponent=SimpleNamespace(left_battler=left_foe, right_battler=right_foe),
    )
    party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))

    def damage(_move, attacker, defender, **_kwargs):
        if attacker is left:
            return SimpleNamespace(min=5, max=5)
        if attacker is right:
            return SimpleNamespace(min=1, max=1)
        if attacker is left_foe and defender is left:
            return SimpleNamespace(min=10, max=10)
        return SimpleNamespace(min=0, max=0)

    with (
        patch("modules.battle_strategies.default.get_party", return_value=party),
        patch(
            "modules.battle_strategies._util.BattleStrategyUtil.calculate_move_damage_range",
            side_effect=damage,
        ),
    ):
        plan = DefaultBattleStrategy()._double_battle_plan(battle_state)

    assert plan[0][2] == "left opponent"
    assert plan[1][2] == "right opponent"
