from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_strategies import TurnAction
from modules.context import context
from modules.battle_strategies._util import BattleStrategyUtil
from modules.battle_strategies.nuzlocke_level_balancing import (
    EmeraldIntroRivalBattleStrategy,
    NuzlockeCaptureStrategy,
    NuzlockeLevelBalancingBattleStrategy,
    _SwitchRisk,
    _active_has_guaranteed_ko_before_response,
    _response_risk_for_line,
)
from modules.nuzlocke.encounter_catalog import EncounterMethod
from modules.nuzlocke.field_lead import CampaignFieldLeadContext, FieldEncounterThreat


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


def test_nuzlocke_level_policy_tracks_participation_per_opponent():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    party = (
        SimpleNamespace(total_exp=100),
        SimpleNamespace(total_exp=100),
    )
    opponent_one = SimpleNamespace(party_index=0, species=SimpleNamespace(name="Poochyena"), level=5)
    opponent_two = SimpleNamespace(party_index=1, species=SimpleNamespace(name="Zigzagoon"), level=5)
    active_zero = SimpleNamespace(party_index=0)
    active_one = SimpleNamespace(party_index=1)
    first_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active_zero, active_battlers=[active_zero]),
        opponent=SimpleNamespace(active_battler=opponent_one),
    )
    second_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active_one, active_battlers=[active_one]),
        opponent=SimpleNamespace(active_battler=opponent_one),
    )
    next_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active_one, active_battlers=[active_one]),
        opponent=SimpleNamespace(active_battler=opponent_two),
    )

    with patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party):
        strategy._sync_opponent_participation(first_state)
        strategy._sync_opponent_participation(second_state)
        assert strategy._participated_against_current_opponent(0)
        assert strategy._participated_against_current_opponent(1)

        strategy._sync_opponent_participation(next_state)
        assert not strategy._participated_against_current_opponent(0)
        assert strategy._participated_against_current_opponent(1)


def test_capture_strategy_shared_switch_evaluator_has_participation_default():
    """Capture emergency switching must not call a level-only method missing on capture strategy."""

    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    assert strategy._participated_against_current_opponent(0) is False


def test_nuzlocke_field_lead_prefers_area_coverage_before_level_priority():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    class FakeType:
        def __init__(self, effectiveness):
            self.effectiveness = effectiveness

        def get_effectiveness_against(self, _defender_type):
            return self.effectiveness

    threat_species = SimpleNamespace(types=(object(),))
    neutral_move = SimpleNamespace(
        type=FakeType(1),
        base_power=40,
    )
    super_effective_move = SimpleNamespace(
        type=FakeType(2),
        base_power=40,
    )
    low_level_neutral = SimpleNamespace(
        level=3,
        total_exp=100,
        current_hp=20,
        total_hp=20,
        current_hp_percentage=100,
        is_egg=False,
        moves=(SimpleNamespace(move=neutral_move, pp=10),),
    )
    higher_level_counter = SimpleNamespace(
        level=6,
        total_exp=500,
        current_hp=20,
        total_hp=20,
        current_hp_percentage=100,
        is_egg=False,
        moves=(SimpleNamespace(move=super_effective_move, pp=10),),
    )

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_party",
            return_value=(low_level_neutral, higher_level_counter),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._field_threats_for_current_map",
            return_value=((threat_species, 20),),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._field_encounter_level_ceiling",
            return_value=None,
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
    ):
        assert strategy._field_lead_candidate() == 1


def test_nuzlocke_field_lead_prioritizes_underleveled_member_before_coverage():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    threat_species = SimpleNamespace(types=(object(),))
    strong_move = SimpleNamespace(type=SimpleNamespace(get_effectiveness_against=lambda _type: 2), base_power=40)
    underleveled = SimpleNamespace(
        level=3,
        total_exp=100,
        current_hp=10,
        total_hp=10,
        is_egg=False,
        moves=(),
    )
    well_levelled_counter = SimpleNamespace(
        level=5,
        total_exp=500,
        current_hp=20,
        total_hp=20,
        is_egg=False,
        moves=(SimpleNamespace(move=strong_move, pp=10),),
    )

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_party",
            return_value=(underleveled, well_levelled_counter),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._field_threats_for_current_map",
            return_value=((threat_species, 20),),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._field_encounter_level_ceiling",
            return_value=5,
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
    ):
        assert strategy._field_lead_candidate() == 0


def test_campaign_field_lead_uses_route_context_instead_of_current_map_lookup():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    threat_species = SimpleNamespace(types=(object(),))
    neutral_move = SimpleNamespace(type=SimpleNamespace(get_effectiveness_against=lambda _type: 1), base_power=40)
    party = (
        SimpleNamespace(
            level=6,
            total_exp=500,
            current_hp=20,
            total_hp=20,
            is_egg=False,
            moves=(SimpleNamespace(move=neutral_move, pp=10),),
        ),
        SimpleNamespace(
            level=2,
            total_exp=50,
            current_hp=10,
            total_hp=10,
            is_egg=False,
            moves=(SimpleNamespace(move=neutral_move, pp=10),),
        ),
    )
    field_context = CampaignFieldLeadContext(
        objective_id="obtain_encounter:0:17",
        current_map=(0, 10),
        objective_map=(0, 17),
        route_maps=((0, 10), (0, 17)),
        threats=(FieldEncounterThreat((0, 17), threat_species, 20, 2, 4, EncounterMethod.LAND),),
        encounter_level_ceiling=4,
        route_cost=10,
        source="objective_route_corridor",
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
    ):
        decision = strategy.choose_field_lead(field_context)

    assert decision.selected_index == 1
    assert decision.objective_id == "obtain_encounter:0:17"
    assert decision.envelope_source == "objective_route_corridor"


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
        assert strategy._battle_support_index == 1


def test_nuzlocke_exp_rotation_rejects_used_candidate_for_fresh_active():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_opponent_participants = {1}
    strategy._battle_opponent_exp_recipients = set()
    active = SimpleNamespace(party_index=0)
    opponent = SimpleNamespace(current_hp=10)
    party = (
        SimpleNamespace(level=5, total_exp=500, current_hp=20, status_condition=None),
        SimpleNamespace(level=3, total_exp=100, current_hp=20, status_condition=None),
    )
    safe_profile = _SwitchRisk(2, 3, 20, 1, 2, 0.0, 1.0, "safe")
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_party",
            return_value=party,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison",
            return_value=False,
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(strategy, "_switch_candidate_risk", return_value=safe_profile),
    ):
        assert (
            strategy._select_switch_target(
                battle_state,
                0,
                util=SimpleNamespace(can_switch=lambda: True),
                require_strict_safe=True,
                prefer_lowest_level=True,
                prefer_fresh_for_opponent=True,
            )
            is None
        )


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


def test_nuzlocke_level_policy_does_not_switch_to_a_candidate_a_response_can_ko():
    """A damaging low-level candidate is not automatically a safe switch."""

    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=26, total_hp=28)
    tackle = SimpleNamespace(name="Tackle")
    howl = SimpleNamespace(name="Howl")
    opponent = SimpleNamespace(
        current_hp=10,
        moves=(SimpleNamespace(move=tackle, pp=1), SimpleNamespace(move=howl, pp=1)),
        stats=SimpleNamespace(speed=10),
    )
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    starter = SimpleNamespace(level=10, current_hp=26, total_hp=28, moves=(SimpleNamespace(move=object(), pp=1),))
    low_poochyena = SimpleNamespace(
        level=3,
        current_hp=4,
        total_hp=4,
        moves=(SimpleNamespace(move=object(), pp=1),),
        stats=SimpleNamespace(speed=5),
    )

    def damage_range(_move, attacker, _defender, **kwargs):
        if attacker is opponent:
            return SimpleNamespace(min=2, max=3 if not kwargs.get("is_critical_hit") else 6)
        return SimpleNamespace(min=2, max=3)

    def strongest_move(pokemon, _opponent):
        return 0 if pokemon in (starter, low_poochyena) else None

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(starter, low_poochyena)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", side_effect=strongest_move),
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch.object(strategy, "_choose_exp_tag_plan", return_value=None),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=(TurnAction.UseMove, 0),
        ),
        patch.object(context, "battle_decision_source", "PLANNER SAFE"),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.UseMove, 0)


def test_nuzlocke_emergency_switch_is_remembered_as_battle_support():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_support_index = None
    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_party",
            return_value=(SimpleNamespace(current_hp=1, total_hp=20), SimpleNamespace(current_hp=20, total_hp=20)),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_has_guaranteed_ko_before_response",
            return_value=False,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(5, 5),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison",
            return_value=False,
        ),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch.object(strategy, "_risk_aware_switch_target", return_value=1),
    ):
        result = strategy._risk_aware_emergency_action(
            SimpleNamespace(
                own_side=SimpleNamespace(
                    active_battler=SimpleNamespace(party_index=0, current_hp=1, total_hp=20)
                ),
                opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=10)),
                is_trainer_battle=True,
            ),
            0,
        )
    assert result == (TurnAction.RotateLead, 1)
    assert strategy._battle_support_index == 1


def test_nuzlocke_risk_policy_switches_critical_lead_before_uncertain_attack():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=2, total_hp=20, status_permanent=None)
    opponent = SimpleNamespace(current_hp=10, species=SimpleNamespace(name="Zigzagoon"))
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    lead = SimpleNamespace(level=3, current_hp=2, total_hp=20, moves=(SimpleNamespace(move=object(), pp=1),))
    replacement = SimpleNamespace(level=10, current_hp=20, total_hp=20, moves=(SimpleNamespace(move=object(), pp=1),))

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(lead, replacement)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", return_value=0),
        patch.object(
            BattleStrategyUtil,
            "calculate_move_damage_range",
            return_value=SimpleNamespace(min=5, max=8),
        ),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.RotateLead, 1)


def test_nuzlocke_trainer_risk_defers_to_best_available_action():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=2, total_hp=20, status_permanent=None)
    opponent = SimpleNamespace(current_hp=10, species=SimpleNamespace(name="Poochyena"))
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    best_available = SimpleNamespace(
        label="Tackle",
        reason="the opponent can KO before the attack line completes",
    )
    planner_report = SimpleNamespace(best_safe_candidate=None, best_available_candidate=best_available)

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(active,)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_has_guaranteed_ko_before_response",
            return_value=False,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(5, 5),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison",
            return_value=False,
        ),
        patch.object(BattleStrategyUtil, "can_switch", return_value=False),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._safe_battle_healing_action",
            return_value=None,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=planner_report,
        ),
    ):
        result = strategy._risk_aware_emergency_action(battle_state, 0)

    assert result is None
    assert context.battle_decision_source == "NUZLOCKE RISK DEFER"
    assert "best_available='Tackle'" in context.battle_decision_detail


def test_nuzlocke_risk_policy_does_not_treat_a_slower_guaranteed_ko_as_safe():
    move = SimpleNamespace(
        name="Tackle",
        base_power=40,
        accuracy=1.0,
        priority=0,
        effect="HIT",
    )
    active = SimpleNamespace(
        current_hp=2,
        total_hp=20,
        stats=SimpleNamespace(speed=5),
        moves=(SimpleNamespace(move=move, pp=1),),
    )
    opponent = SimpleNamespace(
        current_hp=10,
        stats=SimpleNamespace(speed=10),
        moves=(),
    )
    battle_state = SimpleNamespace(
        opponent=SimpleNamespace(active_battler=opponent),
    )
    util = SimpleNamespace(
        calculate_move_damage_range=lambda *_args, **_kwargs: SimpleNamespace(min=10, max=10),
    )

    with patch("modules.battle_strategies.nuzlocke_level_balancing.get_move_by_name"):
        assert not _active_has_guaranteed_ko_before_response(util, active, opponent, battle_state)


def test_nuzlocke_risk_policy_switches_poisoned_lead_before_field_damage():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(
        party_index=0,
        current_hp=20,
        total_hp=20,
        status_permanent=SimpleNamespace(value="poisoned"),
    )
    opponent = SimpleNamespace(
        current_hp=10,
        species=SimpleNamespace(name="Zigzagoon"),
        ability=SimpleNamespace(name="None"),
    )
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    poisoned = SimpleNamespace(
        level=3,
        current_hp=20,
        total_hp=20,
        status_condition=SimpleNamespace(value="poisoned"),
        moves=(SimpleNamespace(move=object(), pp=1),),
    )
    healthy = SimpleNamespace(
        level=10,
        current_hp=20,
        total_hp=20,
        status_condition=SimpleNamespace(value="none"),
        moves=(SimpleNamespace(move=object(), pp=1),),
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(poisoned, healthy)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", return_value=0),
        patch.object(
            BattleStrategyUtil,
            "calculate_move_damage_range",
            return_value=SimpleNamespace(min=5, max=8),
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


def test_nuzlocke_trainer_replacement_is_evaluated_in_shift_mode():
    battle_state, runtime, party, next_opponent = _trainer_replacement_fixture(enabled=False)
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)

    with (
        patch.object(context, "nuzlocke_runtime", runtime),
        patch("modules.pokemon_party.get_opponent_party", return_value=(next_opponent,)),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch("modules.battle_strategies.nuzlocke_level_balancing._eligible_indices", return_value=(0, 1)),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", return_value=0),
        patch.object(
            BattleStrategyUtil,
            "calculate_move_damage_range",
            return_value=SimpleNamespace(min=10, max=10),
        ),
        patch.object(strategy, "_survives_visible_response", return_value=True),
        patch("modules.battle_strategies.nuzlocke_level_balancing._participants_remain_cap_legal", return_value=True),
    ):
        assert strategy.choose_trainer_replacement(battle_state) == 1


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


def test_nuzlocke_capture_policy_keeps_selected_support_member_in():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    active = SimpleNamespace(party_index=0)
    support = SimpleNamespace(party_index=1)
    party = (
        SimpleNamespace(level=10, current_hp=20, is_egg=False),
        SimpleNamespace(level=6, current_hp=12, is_egg=False),
    )
    first_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    support_state = SimpleNamespace(
        is_double_battle=False,
        own_side=SimpleNamespace(active_battler=support),
    )
    expected = (TurnAction.UseMove, 0)

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_party",
            return_value=party,
        ),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(strategy, "_stronger_capture_switch_target", return_value=1),
    ):
        assert strategy.decide_turn(first_state) == (TurnAction.RotateLead, 1)

    with (
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(
            strategy,
            "_stronger_capture_switch_target",
            side_effect=AssertionError("selected capture support must not be reconsidered"),
        ),
        patch("modules.battle_strategies.catch.CatchStrategy.decide_turn", return_value=expected),
    ):
        assert strategy.decide_turn(support_state) == expected


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


def test_probability_model_accepts_critical_only_ko_risk_above_threshold():
    move = SimpleNamespace(name="Tackle", accuracy=1.0)
    opponent = SimpleNamespace(current_hp=10, stats=SimpleNamespace(speed=10))
    candidate = SimpleNamespace(current_hp=20, stats=SimpleNamespace(speed=5))

    def damage_range(_move, _attacker, _defender, **kwargs):
        return SimpleNamespace(
            min=5,
            max=5 if not kwargs.get("is_critical_hit") else 25,
        )

    util = SimpleNamespace(calculate_move_damage_range=damage_range)
    with patch(
        "modules.battle_strategies.nuzlocke_level_balancing.get_move_by_name",
        return_value=move,
    ):
        result = _response_risk_for_line(
            util,
            opponent,
            candidate,
            ("Tackle",),
            damage_min=10,
            damage_max=10,
        )

    assert result is not None
    assert result[2] == 1 / 16
    assert result[3] > 0.85


def test_mid_turn_switch_models_one_response_before_a_faster_replacement_attacks():
    move = SimpleNamespace(name="Tackle", accuracy=1.0)
    opponent = SimpleNamespace(current_hp=10, stats=SimpleNamespace(speed=10))
    candidate = SimpleNamespace(current_hp=20, stats=SimpleNamespace(speed=20))

    def damage_range(_move, _attacker, _defender, **kwargs):
        return SimpleNamespace(min=5, max=5 if not kwargs.get("is_critical_hit") else 5)

    util = SimpleNamespace(calculate_move_damage_range=damage_range)
    with patch(
        "modules.battle_strategies.nuzlocke_level_balancing.get_move_by_name",
        return_value=move,
    ):
        active_result = _response_risk_for_line(
            util,
            opponent,
            candidate,
            ("Tackle",),
            damage_min=10,
            damage_max=10,
        )
        switch_result = _response_risk_for_line(
            util,
            opponent,
            candidate,
            ("Tackle",),
            damage_min=10,
            damage_max=10,
            response_timing="mid_turn_switch",
        )

    assert active_result is not None
    assert switch_result is not None
    assert active_result[0] == 0
    assert switch_result[0] == 5
    assert active_result[3] == 1.0
    assert switch_result[3] == 1.0


def test_capture_weakening_rejects_a_party_faint_risk_even_when_critical_only():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    absorb = SimpleNamespace(name="Absorb", accuracy=1.0)
    tackle = SimpleNamespace(name="Tackle", accuracy=1.0)
    active = SimpleNamespace(
        party_index=0,
        current_hp=20,
        moves=(SimpleNamespace(move=absorb, pp=10),),
        disabled_move=None,
        stats=SimpleNamespace(speed=5),
    )
    opponent = SimpleNamespace(
        current_hp=10,
        stats=SimpleNamespace(speed=10),
        moves=(SimpleNamespace(move=tackle, pp=10),),
    )
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )

    def damage_range(move, attacker, _defender, **kwargs):
        if attacker is opponent:
            return SimpleNamespace(min=5, max=25 if kwargs.get("is_critical_hit") else 5)
        return SimpleNamespace(min=9, max=9)

    with (
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_move_by_name",
            side_effect=lambda name: tackle if name == "Tackle" else absorb,
        ),
    ):
        assert strategy._get_safe_weakening_move(battle_state) is None
    assert "target KO risk" in context.battle_decision_detail


def test_capture_weakening_rejects_critical_ko_when_ball_has_better_expected_value():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    mud_slap = SimpleNamespace(name="Mud-Slap", accuracy=1.0)
    active = SimpleNamespace(
        party_index=0,
        current_hp=20,
        moves=(SimpleNamespace(move=mud_slap, pp=10),),
        disabled_move=None,
    )
    opponent = SimpleNamespace(
        current_hp=10,
        total_hp=20,
        species=SimpleNamespace(catch_rate=45),
    )
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )

    def damage_range(_move, attacker, _defender, **kwargs):
        if attacker is active:
            return SimpleNamespace(min=3, max=12 if kwargs.get("is_critical_hit") else 3)
        return SimpleNamespace(min=0, max=0)

    with patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range):
        assert strategy._get_safe_weakening_move(battle_state) is None
    assert "target KO risk" in context.battle_decision_detail


def test_capture_weakening_uses_expected_catch_gain_when_no_ko_risk_exists():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    absorb = SimpleNamespace(name="Absorb", accuracy=1.0)
    active = SimpleNamespace(
        party_index=0,
        current_hp=20,
        moves=(SimpleNamespace(move=absorb, pp=10),),
        disabled_move=None,
    )
    opponent = SimpleNamespace(
        current_hp=10,
        total_hp=20,
        species=SimpleNamespace(catch_rate=45),
    )
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )

    def damage_range(_move, attacker, _defender, **kwargs):
        if attacker is active:
            return SimpleNamespace(min=3, max=3)
        return SimpleNamespace(min=0, max=0)

    with patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range):
        assert strategy._get_safe_weakening_move(battle_state) == 0
    assert "expected capture=" in context.battle_decision_detail


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
