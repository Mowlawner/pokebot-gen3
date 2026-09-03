from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_strategies import TurnAction
from modules.context import context
from modules.battle_strategies.nuzlocke_level_balancing import (
    EmeraldIntroRivalBattleStrategy,
    NuzlockeCaptureStrategy,
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

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(1,),
        ),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(None, object())),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
    ):
        assert strategy.has_replacement_after_faint(battle_state)


def test_nuzlocke_party_can_battle_ignores_unusable_lowest_member():
    low = SimpleNamespace(name="low", current_hp=0)
    high = SimpleNamespace(name="high", current_hp=10)
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(low, high)),
        patch.object(strategy, "pokemon_can_battle", side_effect=(False, True)),
    ):
        assert strategy.party_can_battle()


def test_nuzlocke_level_policy_selects_the_lowest_living_member_after_battle():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    party = (
        SimpleNamespace(level=10, current_hp=20, is_egg=False),
        SimpleNamespace(level=3, current_hp=12, is_egg=False),
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
    ):
        assert strategy.choose_new_lead_after_battle() == 1


def test_nuzlocke_level_policy_keeps_unreadable_identity_members_eligible():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    party = (SimpleNamespace(level=10, current_hp=20, is_egg=False),)

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.PokemonIdentity.from_pokemon",
            return_value=None,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._nuzlocke_dead_identities",
            return_value=frozenset(),
        ),
    ):
        assert strategy.choose_new_lead_after_battle() is None


def test_nuzlocke_level_policy_switches_to_lowest_capable_member_at_battle_start():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    party = (
        SimpleNamespace(level=10, current_hp=20),
        SimpleNamespace(level=3, current_hp=12),
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.BattleStrategyUtil.can_switch",
            return_value=True,
        ),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            side_effect=AssertionError("the battle should switch before choosing a move"),
        ),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.RotateLead, 1)


def test_nuzlocke_level_policy_keeps_support_member_in_after_emergency_switch():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_support_index = 0
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    party = (
        SimpleNamespace(level=12, current_hp=20),
        SimpleNamespace(level=3, current_hp=12),
    )
    expected = (TurnAction.UseMove, 0)

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=expected,
        ) as fallback,
    ):
        assert strategy.decide_turn(battle_state) == expected

    fallback.assert_called_once_with(battle_state)


def test_nuzlocke_level_policy_records_inherited_support_switch_and_keeps_it_in():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_support_index = None
    active = SimpleNamespace(party_index=0)
    first_battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
    )
    support = SimpleNamespace(party_index=1)
    second_battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=support, active_battlers=[support]),
    )
    party = (
        SimpleNamespace(level=3, current_hp=12),
        SimpleNamespace(level=12, current_hp=20),
    )
    inherited_actions = iter(((TurnAction.RotateLead, 1), (TurnAction.UseMove, 0)))

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            side_effect=inherited_actions,
        ) as inherited,
        patch.object(context, "battle_decision_source", "PLANNER SAFE"),
    ):
        assert strategy.decide_turn(first_battle_state) == (TurnAction.RotateLead, 1)
        assert strategy._battle_support_index == 1
        assert strategy.decide_turn(second_battle_state) == (TurnAction.UseMove, 0)

    assert inherited.call_count == 2


def test_nuzlocke_level_policy_switches_to_support_when_weak_lead_line_is_not_safe():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_support_index = None
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    party = (
        SimpleNamespace(level=3, current_hp=12),
        SimpleNamespace(level=12, current_hp=20),
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=(TurnAction.UseMove, 0),
        ),
        patch.object(context, "battle_decision_source", "PLANNER BEST_AVAILABLE"),
        patch.object(strategy, "_stronger_battle_support_target", return_value=1) as support,
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.RotateLead, 1)

    support.assert_called_once_with(battle_state, 0)
    assert strategy._battle_support_index == 1


def test_nuzlocke_capture_policy_switches_to_stronger_member_before_catching():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    party = (
        SimpleNamespace(level=10, current_hp=20, is_egg=False),
        SimpleNamespace(level=3, current_hp=12, is_egg=False),
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch.object(strategy, "_stronger_capture_switch_target", return_value=1),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.RotateLead, 1)


def test_nuzlocke_capture_policy_prefers_safe_weakeners_over_lethal_stronger_members():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    active = SimpleNamespace(party_index=0)
    opponent = SimpleNamespace(current_hp=10)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    party = (
        SimpleNamespace(level=3, current_hp=12, is_egg=False, moves=(SimpleNamespace(move=object(), pp=1),)),
        SimpleNamespace(level=12, current_hp=30, is_egg=False, moves=(SimpleNamespace(move=object(), pp=1),)),
        SimpleNamespace(level=8, current_hp=20, is_egg=False, moves=(SimpleNamespace(move=object(), pp=1),)),
    )

    def damage_for(_move, pokemon, _opponent):
        return SimpleNamespace(max=20 if pokemon is party[1] else 2)

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1, 2),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch("modules.battle_strategies.nuzlocke_level_balancing.BattleStrategyUtil.can_switch", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.BattleStrategyUtil.get_strongest_move_against",
            return_value=0,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.BattleStrategyUtil.calculate_move_damage_range",
            side_effect=damage_for,
        ),
    ):
        assert strategy._stronger_capture_switch_target(battle_state, 0) == 2


def test_nuzlocke_capture_policy_keeps_no_ball_fallback_automated():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    expected = (TurnAction.UseMove, 0)

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_party",
            return_value=(SimpleNamespace(level=10, current_hp=20, is_egg=False),),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0,),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.BattleStrategyUtil.can_switch",
            return_value=False,
        ),
        patch.object(strategy, "_get_best_poke_ball", return_value=None),
        patch(
            "modules.battle_strategies.default.DefaultBattleStrategy.decide_turn",
            return_value=expected,
        ) as fallback,
    ):
        assert strategy.decide_turn(battle_state) == expected

    fallback.assert_called_once_with(battle_state)


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
