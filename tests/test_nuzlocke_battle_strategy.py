from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_strategies import TurnAction
from modules.context import context
from modules.battle_strategies._util import BattleStrategyUtil
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


def test_nuzlocke_level_policy_rejects_zero_damage_lowest_member_at_battle_start():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0)
    opponent = SimpleNamespace(current_hp=10)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    lotad = SimpleNamespace(level=3, current_hp=12)
    starter = SimpleNamespace(level=10, current_hp=20)
    party = (lotad, starter)

    def strongest_move(pokemon, _opponent):
        return None if pokemon is lotad else 0

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", side_effect=strongest_move),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.BattleStrategyUtil.can_switch",
            return_value=True,
        ),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            side_effect=AssertionError("the battle should switch to the effective matchup"),
        ),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.RotateLead, 1)


def _trainer_replacement_fixture(*, enabled=True, double=False):
    from modules.nuzlocke.rule_config import CampaignRuleId, CampaignRulesConfig

    active = SimpleNamespace(party_index=0)
    current_opponent = SimpleNamespace(party_index=0)
    next_opponent = SimpleNamespace(
        index=1,
        is_empty=False,
        is_valid=True,
        current_hp=10,
        moves=(SimpleNamespace(move=SimpleNamespace(name="Tackle"), pp=1),),
        species=SimpleNamespace(name="Zigzagoon"),
    )
    battle_state = SimpleNamespace(
        is_double_battle=double,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=current_opponent),
    )
    runtime = SimpleNamespace(
        rule_config=CampaignRulesConfig(
                frozenset({CampaignRuleId.SET_BATTLE_STYLE}) if enabled else frozenset()
        )
    )
    current = SimpleNamespace(level=10, current_hp=20, moves=(SimpleNamespace(move=object(), pp=1),))
    candidate = SimpleNamespace(level=3, current_hp=12, moves=(SimpleNamespace(move=object(), pp=1),))
    return battle_state, runtime, (current, candidate), next_opponent


def test_nuzlocke_trainer_replacement_is_disabled_by_default():
    battle_state, runtime, party, next_opponent = _trainer_replacement_fixture(enabled=False)
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    with (
        patch.object(context, "nuzlocke_runtime", runtime),
        patch("modules.pokemon_party.get_opponent_party", return_value=(next_opponent,)),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
    ):
        assert strategy.choose_trainer_replacement(battle_state) is None


def test_nuzlocke_trainer_replacement_selects_safe_lower_level_counter():
    battle_state, runtime, party, next_opponent = _trainer_replacement_fixture()
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    def strongest_move(pokemon, _opponent):
        return 0

    def damage_range(_move, attacker, _defender, **_kwargs):
        return SimpleNamespace(min=2 if attacker is party[0] else 10, max=2 if attacker is party[0] else 10)

    with (
        patch.object(context, "nuzlocke_runtime", runtime),
        patch("modules.pokemon_party.get_opponent_party", return_value=(next_opponent,)),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch("modules.battle_strategies.nuzlocke_level_balancing._eligible_indices", return_value=(0, 1)),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", side_effect=strongest_move),
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch.object(strategy, "_survives_visible_response", return_value=True),
        patch("modules.battle_strategies.nuzlocke_level_balancing._participants_remain_cap_legal", return_value=True),
    ):
        assert strategy.choose_trainer_replacement(battle_state) == 1


def test_nuzlocke_trainer_replacement_answers_no_when_current_is_best():
    battle_state, runtime, party, next_opponent = _trainer_replacement_fixture()
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    def damage_range(_move, attacker, _defender, **_kwargs):
        damage = 10 if attacker is party[0] else 2
        return SimpleNamespace(min=damage, max=damage)

    with (
        patch.object(context, "nuzlocke_runtime", runtime),
        patch("modules.pokemon_party.get_opponent_party", return_value=(next_opponent,)),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch("modules.battle_strategies.nuzlocke_level_balancing._eligible_indices", return_value=(0, 1)),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", return_value=0),
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch.object(strategy, "_survives_visible_response", return_value=True),
        patch("modules.battle_strategies.nuzlocke_level_balancing._participants_remain_cap_legal", return_value=True),
    ):
        assert strategy.choose_trainer_replacement(battle_state) is None


def test_nuzlocke_trainer_replacement_does_not_change_double_battle_path():
    battle_state, runtime, party, next_opponent = _trainer_replacement_fixture(double=True)
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    with (
        patch.object(context, "nuzlocke_runtime", runtime),
        patch("modules.pokemon_party.get_opponent_party", return_value=(next_opponent,)),
    ):
        assert strategy.choose_trainer_replacement(battle_state) is None


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


def test_nuzlocke_level_policy_tags_safe_low_member_then_returns_to_finisher():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0)
    opponent = SimpleNamespace(
        current_hp=10,
        stats=SimpleNamespace(speed=10),
        species=SimpleNamespace(base_experience_yield=50),
        level=5,
    )
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    finisher = SimpleNamespace(
        level=10,
        total_exp=1000,
        current_hp=20,
        current_hp_percentage=100,
        is_egg=False,
        stats=SimpleNamespace(speed=30),
        moves=(SimpleNamespace(move=object(), pp=1),),
    )
    tag = SimpleNamespace(
        level=3,
        total_exp=100,
        current_hp=12,
        current_hp_percentage=100,
        is_egg=False,
        stats=SimpleNamespace(speed=5),
        moves=(SimpleNamespace(move=object(), pp=1),),
    )
    party = (finisher, tag)
    response_move = object()

    def strongest_move(pokemon, _opponent):
        return None if pokemon is tag else 0

    def damage_range(_move, attacker, _defender, **_kwargs):
        return SimpleNamespace(min=2, max=2) if attacker is opponent else SimpleNamespace(min=4, max=4)

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", side_effect=strongest_move),
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_move_by_name", return_value=response_move),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_move_names",
            return_value=("Tackle",),
        ),
        patch.object(context, "nuzlocke_runtime", None),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.RotateLead, 1)
        assert strategy._pending_exp_tag_index == 1
        assert strategy._pending_exp_finisher_index == 0

        continuation_state = SimpleNamespace(
            is_double_battle=False,
            is_trainer_battle=False,
            own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=1)),
            opponent=SimpleNamespace(active_battler=opponent),
        )
        assert strategy.decide_turn(continuation_state) == (TurnAction.RotateLead, 0)
        assert strategy._pending_exp_tag_index is None


def test_nuzlocke_level_policy_rejects_exp_tag_without_a_valid_finisher():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=0)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=10)),
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
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._battle_capable_indices",
            return_value=(0,),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_move_names",
            return_value=("Tackle",),
        ),
        patch.object(strategy, "_battle_finisher_target", return_value=None),
        patch.object(strategy, "_survives_visible_response", return_value=True),
    ):
        assert strategy._choose_exp_tag_plan(battle_state, 0, 0) is None


def test_nuzlocke_level_policy_rejects_exp_tag_when_projected_exp_breaks_cap():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=0)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=10)),
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
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._battle_capable_indices",
            return_value=(0,),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_move_names",
            return_value=("Tackle",),
        ),
        patch.object(strategy, "_battle_finisher_target", return_value=0),
        patch.object(strategy, "_survives_visible_response", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._participants_remain_cap_legal",
            return_value=False,
        ),
    ):
        assert strategy._choose_exp_tag_plan(battle_state, 0, 0) is None


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

    fallback.assert_called_once_with(strategy, battle_state)


def test_nuzlocke_capture_policy_hard_vetoes_ball_for_ineligible_encounter():
    from modules.nuzlocke.runtime import CaptureEligibility

    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    battle_state = SimpleNamespace(
        is_trainer_battle=False,
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=0)),
    )
    runtime = SimpleNamespace(
        capture_eligibility_for=lambda *_args, **_kwargs: CaptureEligibility(
            False,
            "previous encounter was already obtained species/evolution line: Wurmple",
        )
    )
    expected = (TurnAction.UseMove, 0)
    with (
        patch.object(context, "nuzlocke_runtime", runtime),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_player_avatar", return_value=SimpleNamespace(map_group_and_number=(1, 2))),
        patch.object(strategy, "_stronger_capture_switch_target", return_value=None),
        patch.object(strategy, "_get_best_poke_ball", side_effect=AssertionError("ineligible encounter used a ball")),
        patch("modules.battle_strategies.default.DefaultBattleStrategy.decide_turn", return_value=expected),
    ):
        assert strategy.decide_turn(battle_state) == expected
    assert "previous encounter was already obtained" in context.message


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
