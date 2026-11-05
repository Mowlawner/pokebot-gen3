from types import SimpleNamespace
from unittest.mock import patch

from modules.battle_strategies import TurnAction
from modules.battle_planner import PlannerAction, PlannerDecisionClass
from modules.battle_expectiminimax import SearchResultCategory
from modules.context import context
from modules.battle_strategies._util import BattleStrategyUtil
from modules.pokemon import get_move_by_name, get_species_by_name
from modules.battle_strategies.nuzlocke_level_balancing import (
    EmeraldIntroRivalBattleStrategy,
    NuzlockeCaptureStrategy,
    NuzlockeLevelBalancingBattleStrategy,
    _SwitchRisk,
    _active_has_guaranteed_ko_before_response,
    _battle_training_priority,
    _response_risk_for_line,
    _safe_battle_healing_action,
    _planner_move_is_zero_damage,
    _FREE_TRAINER_SWITCH_RESPONSE_TURNS,
    _EMERGENCY_SWITCH_RESPONSE_TURNS,
)
from modules.items import ItemBattleUse
from modules.nuzlocke.encounter_catalog import EncounterMethod
from modules.nuzlocke.field_lead import CampaignFieldLeadContext, FieldEncounterThreat


def test_rotation_target_builds_switch_facts_with_active_party_indices():
    active = SimpleNamespace(party_index=0, current_hp=10, total_hp=10)
    replacement = SimpleNamespace(party_index=1, current_hp=10, total_hp=10, level=5, status_condition=None)
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(left_battler=active, right_battler=None),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=10)),
    )
    util = BattleStrategyUtil(battle_state)

    with (
        patch("modules.battle_strategies._util.get_party", return_value=(active, replacement)),
        patch.object(util, "get_potential_rotation_targets", return_value=[1]),
        patch.object(util, "get_strongest_move_against", return_value=None),
        patch("modules.battle_strategies._util.rank_switch_candidates", side_effect=lambda candidates, _policy: candidates),
    ):
        selected = util.select_rotation_target(battle_state)

    assert selected == 1


def test_damage_range_preserves_ground_immunity_against_flying_type():
    move = get_move_by_name("Mud-slap")
    attacker = SimpleNamespace(
        species=get_species_by_name("Mudkip"),
        types=get_species_by_name("Mudkip").types,
        ability=SimpleNamespace(name="None"),
    )
    defender = SimpleNamespace(
        species=get_species_by_name("Taillow"),
        types=get_species_by_name("Taillow").types,
        ability=SimpleNamespace(name="None"),
    )
    util = BattleStrategyUtil(SimpleNamespace())

    with patch.object(util, "_calculate_base_move_damage", return_value=10):
        damage = util.calculate_move_damage_range(move, attacker, defender)

    assert (damage.min, damage.max) == (0, 0)


def test_capture_evaluator_rejects_ground_immune_weakening_move():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    move = get_move_by_name("Mud-slap")
    active = SimpleNamespace(
        species=get_species_by_name("Mudkip"),
        types=get_species_by_name("Mudkip").types,
        ability=SimpleNamespace(name="None"),
        moves=(SimpleNamespace(move=move, pp=10),),
        disabled_move=None,
    )
    opponent = SimpleNamespace(
        species=get_species_by_name("Taillow"),
        types=get_species_by_name("Taillow").types,
        ability=SimpleNamespace(name="None"),
        current_hp=20,
        total_hp=20,
    )
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    ball = SimpleNamespace(index=4, name="Poké Ball")

    with (
        patch.object(BattleStrategyUtil, "_calculate_base_move_damage", return_value=10),
        patch.object(strategy, "_capture_response_profile", return_value=(1.0, None, None, "test")),
    ):
        assert (
            strategy._evaluate_capture_move(
                battle_state,
                0,
                active.moves[0],
                ball,
                BattleStrategyUtil(battle_state),
            )
            is None
        )


def test_committed_support_uses_healing_before_another_imperfect_switch():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_support_index = 1
    strategy._trainer_high_risk_signature = ((None, None, None), 1)
    strategy._trainer_high_risk_streak = 1
    strategy._trainer_high_risk_last_hp = None
    active = SimpleNamespace(party_index=1, current_hp=6, total_hp=29)
    opponent = SimpleNamespace(current_hp=20)
    battle_state = SimpleNamespace(
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    action = (TurnAction.UseMove, 0)
    report = SimpleNamespace(
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        risk=SimpleNamespace(value="high"),
    )
    healing = (TurnAction.UseItem, (SimpleNamespace(name="Potion"), 1))

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(4, 6),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._safe_battle_healing_action",
            return_value=healing,
        ),
        patch.object(strategy, "_least_bad_trainer_switch_target") as imperfect_switch,
        patch.object(strategy, "_remember_battle_support") as remember_support,
    ):
        result = strategy._bounded_trainer_high_risk_fallback(battle_state, action, report)

    assert result == healing
    imperfect_switch.assert_not_called()
    remember_support.assert_called_once()


def test_known_poison_threat_prefers_safe_healing_before_switching_matchup():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=10, total_hp=20, status_permanent=None)
    replacement = SimpleNamespace(party_index=1, current_hp=20, total_hp=20)
    opponent = SimpleNamespace(current_hp=20, species=SimpleNamespace(name="Shroomish"))
    battle_state = SimpleNamespace(
        is_trainer_battle=True,
        is_wild=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    healing = (TurnAction.UseItem, (SimpleNamespace(name="Potion"), 0))

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(active, replacement)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_has_guaranteed_ko_before_response",
            return_value=False,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(3, 3),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_response_risk_for_line",
            return_value=None,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison",
            return_value=True,
        ),
        patch("modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state", return_value=None),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._safe_battle_healing_action",
            return_value=healing,
        ),
        patch.object(strategy, "_risk_aware_switch_target", return_value=1) as switch_target,
        patch.object(strategy, "_remember_battle_support"),
    ):
        result = strategy._risk_aware_emergency_action(battle_state, 0)

    assert result == healing
    switch_target.assert_not_called()
    assert context.battle_decision_source == "NUZLOCKE POISON PREVENTION HEAL"


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


def test_battle_training_priority_prefers_exp_remaining_in_cap_over_level_order():
    growth_with_less_headroom = SimpleNamespace(
        get_experience_needed_for_level=lambda level: {16: 1600}[level]
    )
    growth_with_more_headroom = SimpleNamespace(
        get_experience_needed_for_level=lambda level: {16: 3000}[level]
    )
    party = (
        SimpleNamespace(
            level=10,
            total_exp=1000,
            species=SimpleNamespace(level_up_type=growth_with_less_headroom),
        ),
        SimpleNamespace(
            level=11,
            total_exp=1500,
            species=SimpleNamespace(level_up_type=growth_with_more_headroom),
        ),
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch("modules.battle_strategies.nuzlocke_level_balancing._active_level_cap", return_value=15),
    ):
        selected = min((0, 1), key=_battle_training_priority)

    assert selected == 1


def test_forced_replacement_ranks_survival_before_training_priority():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=0, total_hp=15, is_egg=False)
    torchic = SimpleNamespace(
        name="Torchic",
        party_index=1,
        current_hp=21,
        total_hp=24,
        is_egg=False,
        level=7,
        status_condition=None,
    )
    zigzagoon = SimpleNamespace(
        name="Zigzagoon",
        party_index=2,
        current_hp=12,
        total_hp=17,
        is_egg=False,
        level=4,
        status_condition=None,
    )
    opponent = SimpleNamespace(party_index=0, current_hp=15, species=SimpleNamespace(name="Poochyena"))
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=opponent),
    )

    profiles = {
        1: _SwitchRisk(4, 6, 21, 6, 8, 0.0, 1.0, "Torchic survives the response"),
        2: _SwitchRisk(1, 2, 12, 6, 8, 0.0, 1.0, "Zigzagoon survives the response"),
    }

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(active, torchic, zigzagoon)),
        patch("modules.battle_strategies.nuzlocke_level_balancing._eligible_indices", return_value=(1, 2)),
        patch("modules.battle_strategies.nuzlocke_level_balancing._active_party_indices", return_value=frozenset({0})),
        patch("modules.battle_strategies.nuzlocke_level_balancing._opponent_move_names_for_safety", return_value=("Tackle",)),
        patch("modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison", return_value=False),
        patch.object(strategy, "_switch_candidate_risk", side_effect=lambda _state, index, **_kwargs: profiles[index]),
        patch("modules.battle_strategies.nuzlocke_level_balancing._battle_training_priority", side_effect=lambda index, *args: (0, index)),
        patch("modules.battle_strategies.nuzlocke_level_balancing._active_level_cap", return_value=None),
    ):
        assert strategy.choose_new_lead_after_faint(battle_state) == 1


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
        patch.object(context, "campaign_preparation_target", "Normal trainer on Route 116 at (12, 17)"),
    ):
        decision = strategy.choose_field_lead(field_context)

    assert decision.selected_index == 1
    assert decision.objective_id == "obtain_encounter:0:17"
    assert decision.envelope_source == "objective_route_corridor"
    assert "target Normal trainer on Route 116 at (12, 17)" in context.message


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


def test_committed_battle_support_switches_when_multi_turn_exposure_becomes_unsafe():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_support_index = 1
    active = SimpleNamespace(party_index=1, current_hp=10, total_hp=20)
    opponent = SimpleNamespace(current_hp=20)
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
        is_trainer_battle=True,
        is_wild=False,
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(object(), active)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_has_guaranteed_ko_before_response",
            return_value=False,
        ),
        patch.object(
            strategy,
            "_risk_aware_switch_target",
            return_value=0,
        ) as switch_target,
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(1, 2),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_response_risk_for_line",
            return_value=(1, 2, 0.5, 0.5),
        ),
            patch(
                "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison",
                return_value=False,
            ),
            patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        ):
        result = strategy._risk_aware_emergency_action(battle_state, 1)

    assert result == (TurnAction.RotateLead, 0)
    switch_target.assert_called_once()


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


def test_capture_policy_gets_a_safe_turn_before_wild_risk_escape():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    active = SimpleNamespace(party_index=0, current_hp=14, total_hp=16, status_permanent=None)
    battle_state = SimpleNamespace(
        is_trainer_battle=False,
        is_wild=True,
        nuzlocke_capture_target=True,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=15)),
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(active,)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_has_guaranteed_ko_before_response",
            return_value=False,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(2, 4),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_response_risk_for_line",
            return_value=(2, 4, 0.9, 0.1),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison",
            return_value=False,
        ),
        patch("modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state", return_value=None),
        patch.object(BattleStrategyUtil, "can_switch", return_value=False),
        patch.object(BattleStrategyUtil, "get_best_escape_method", return_value=(TurnAction.RunAway, None)),
        patch.object(strategy, "_get_best_poke_ball", return_value=SimpleNamespace(name="Poké Ball")),
    ):
        result = strategy._risk_aware_emergency_action(battle_state, 0)

    assert result is None
    assert context.battle_decision_source == "CAPTURE POLICY DEFER"


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


def test_complete_trainer_expectiminimax_owns_action_before_risk_safety():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=20, total_hp=20)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=10)),
    )
    planner_report = SimpleNamespace(
        stochastic_result=object(),
        evaluation_method="expectiminimax",
    )
    planner_action = (TurnAction.UseMove, 0)

    def planner(_strategy, _battle_state):
        strategy._last_planner_decision = planner_report
        return planner_action

    with (
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action") as emergency,
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.DefaultBattleStrategy.decide_turn",
            side_effect=planner,
        ),
    ):
        assert strategy.decide_turn(battle_state) == planner_action

    emergency.assert_not_called()
    assert context.battle_decision_source != "NUZLOCKE RISK SAFETY"


def test_nuzlocke_risk_does_not_override_a_planner_safe_attack_with_healing():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=2, total_hp=20, status_permanent=None)
    opponent = SimpleNamespace(current_hp=10, species=SimpleNamespace(name="Poochyena"))
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    planner_report = SimpleNamespace(
        action=PlannerAction.UseMove,
        classification=PlannerDecisionClass.SAFE,
        best_safe_candidate=SimpleNamespace(label="Tackle"),
        best_available_candidate=SimpleNamespace(label="Tackle"),
    )

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
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=planner_report,
        ),
        patch.object(BattleStrategyUtil, "can_switch", return_value=False),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._safe_battle_healing_action",
            side_effect=AssertionError("safe attack should be selected before emergency healing"),
        ),
    ):
        result = strategy._risk_aware_emergency_action(battle_state, 0)

    assert result is None
    assert context.battle_decision_source == "NUZLOCKE RISK DEFER"
    assert "safe attack line" in context.battle_decision_detail


def test_nuzlocke_emergency_switch_uses_a_bounded_response_window():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    battle_state = SimpleNamespace()
    util = SimpleNamespace()

    with patch.object(strategy, "_select_switch_target", return_value=1) as select_target:
        result = strategy._risk_aware_switch_target(battle_state, 0, util)

    assert result == 1
    select_target.assert_called_once_with(
        battle_state,
        0,
        util=util,
        require_strict_safe=True,
        prefer_lowest_level=False,
        prefer_fresh_for_opponent=False,
        response_turns=_EMERGENCY_SWITCH_RESPONSE_TURNS,
    )


def test_nuzlocke_emergency_switch_reuses_bounded_horizon_at_executor_validation():
    """The switch menu guard must validate the same line that selected support."""

    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._battle_support_index = 1
    strategy._battle_support_response_turns = _EMERGENCY_SWITCH_RESPONSE_TURNS
    strategy._battle_support_opponent_key = (0, "Taillow", 5)
    active = SimpleNamespace(party_index=0)
    opponent = SimpleNamespace(party_index=0, species=SimpleNamespace(name="Taillow"), level=5)
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    party = (SimpleNamespace(name="Mudkip"), SimpleNamespace(name="Wurmple"))
    safe_profile = _SwitchRisk(2, 3, 20, 5, 6, 0.0, 1.0, "safe over emergency horizon")

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._participants_remain_cap_legal",
            return_value=True,
        ),
        patch.object(strategy, "_switch_candidate_risk", return_value=safe_profile) as evaluate,
    ):
        assert strategy.is_switch_target_valid(battle_state, 1)

    assert evaluate.call_args.kwargs["response_turns"] == _EMERGENCY_SWITCH_RESPONSE_TURNS


def test_nuzlocke_risk_policy_switches_when_active_multi_turn_line_is_unsafe():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    tackle = SimpleNamespace(name="Tackle", accuracy=1.0, priority=0, base_power=40, effect="HIT")
    active = SimpleNamespace(
        party_index=0,
        current_hp=25,
        total_hp=25,
        status_permanent=None,
        stats=SimpleNamespace(speed=5),
        moves=(SimpleNamespace(move=tackle, pp=10),),
    )
    replacement = SimpleNamespace(
        party_index=1,
        current_hp=30,
        total_hp=30,
        stats=SimpleNamespace(speed=10),
        moves=(SimpleNamespace(move=tackle, pp=10),),
    )
    opponent = SimpleNamespace(
        current_hp=25,
        stats=SimpleNamespace(speed=10),
        moves=(SimpleNamespace(move=tackle, pp=10),),
    )
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=opponent),
    )

    def damage_range(move, attacker, _defender, **kwargs):
        if attacker is opponent:
            return SimpleNamespace(min=3, max=4 if not kwargs.get("is_critical_hit") else 8)
        return SimpleNamespace(min=3, max=4)

    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_party",
            return_value=(active, replacement),
        ),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", return_value=0),
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch.object(BattleStrategyUtil, "critical_hit_chance", return_value=1 / 16),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.get_move_by_name",
            return_value=tackle,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._eligible_indices",
            return_value=(0, 1),
        ),
        patch.object(strategy, "_risk_aware_switch_target", return_value=1),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=SimpleNamespace(best_safe_candidate=None, best_available_candidate=None),
        ),
    ):
        result = strategy._risk_aware_emergency_action(battle_state, 0)

    assert result == (TurnAction.RotateLead, 1)
    assert context.battle_decision_source == "NUZLOCKE RISK SAFETY"
    assert "multi-turn survival" in context.battle_decision_detail


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
        patch("modules.battle_strategies.nuzlocke_level_balancing._participants_remain_cap_legal", return_value=True),
    ):
        assert strategy.choose_trainer_replacement(battle_state) == 1


def test_nuzlocke_trainer_replacement_uses_free_safe_lower_level_rotation():
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
        assert strategy.choose_trainer_replacement(battle_state) == 1


def test_nuzlocke_trainer_replacement_uses_free_switch_for_safe_exp_tag():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0)
    current_opponent = SimpleNamespace(party_index=0)
    next_opponent = SimpleNamespace(
        index=1,
        is_empty=False,
        is_valid=True,
        current_hp=10,
        level=5,
        moves=(SimpleNamespace(move=SimpleNamespace(name="Tackle"), pp=1),),
        species=SimpleNamespace(name="Zigzagoon"),
    )
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active, active_battlers=[active]),
        opponent=SimpleNamespace(active_battler=current_opponent),
    )
    current = SimpleNamespace(level=10, current_hp=20, total_hp=20, moves=(SimpleNamespace(move=object(), pp=1),))
    tag = SimpleNamespace(level=3, total_exp=100, current_hp=12, total_hp=12, moves=(SimpleNamespace(move=object(), pp=1),))
    finisher = SimpleNamespace(level=8, total_exp=500, current_hp=20, total_hp=20, moves=(SimpleNamespace(move=object(), pp=1),))
    party = (current, tag, finisher)

    def strongest_move(pokemon, _opponent):
        return None if pokemon is tag else 0

    def damage_range(_move, attacker, _defender, **_kwargs):
        return SimpleNamespace(min=10 if attacker is current else 8, max=10 if attacker is current else 8)

    with (
        patch.object(context, "nuzlocke_runtime", None),
        patch("modules.pokemon_party.get_opponent_party", return_value=(next_opponent,)),
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=party),
        patch("modules.battle_strategies.nuzlocke_level_balancing._eligible_indices", return_value=(0, 1, 2)),
        patch.object(strategy, "pokemon_can_battle", return_value=True),
        patch.object(BattleStrategyUtil, "can_switch", return_value=True),
        patch.object(BattleStrategyUtil, "get_strongest_move_against", side_effect=strongest_move),
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch.object(strategy, "_survives_visible_response", return_value=True),
        patch("modules.battle_strategies.nuzlocke_level_balancing._participants_remain_cap_legal", return_value=True),
    ):
        assert strategy.choose_trainer_replacement(battle_state) == 1

    assert strategy._pending_exp_tag_index == 1
    assert strategy._pending_exp_finisher_index in (0, 2)
    assert len(strategy._free_switch_history) == 1


def test_nuzlocke_free_trainer_switch_is_tracked_separately_from_voluntary_switches():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    opponent = SimpleNamespace(party_index=1, level=5, species=SimpleNamespace(name="Nincada"))
    battle_state = SimpleNamespace(opponent=SimpleNamespace(active_battler=opponent))

    strategy._remember_free_trainer_switch(2, opponent, "free trainer replacement")
    strategy._remember_voluntary_switch(0, battle_state, "validated safety switch")

    assert strategy._free_switch_origin == (2, (1, "Nincada", 5), "free trainer replacement")
    assert strategy._free_switch_history == [(2, (1, "Nincada", 5))]
    assert strategy._voluntary_switch_history == [(0, (1, "Nincada", 5), "validated safety switch")]


def test_ordinary_switch_guard_rejects_immediate_reversal_but_keeps_other_targets():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._voluntary_switch_commitment = ((1, "Nincada", 5), 0, 1, "validated safety switch")
    strategy._switch_validation_bypass_target = None
    active = SimpleNamespace(party_index=1)
    opponent = SimpleNamespace(party_index=1, level=5, species=SimpleNamespace(name="Nincada"))
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    with patch.object(strategy, "_commitment_reversal_is_necessary", return_value=False):
        assert strategy._ordinary_switch_guard_blocks(battle_state, 0)
        assert not strategy._ordinary_switch_guard_blocks(battle_state, 2)


def test_bounded_trainer_high_risk_fallback_switches_after_repeated_risky_attacks():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._trainer_high_risk_signature = None
    strategy._trainer_high_risk_streak = 0
    strategy._trainer_high_risk_last_hp = None
    strategy._voluntary_switch_commitment = None
    strategy._switch_validation_bypass_target = None
    active = SimpleNamespace(party_index=0, current_hp=10, total_hp=20)
    opponent = SimpleNamespace(party_index=1, level=5, current_hp=20, species=SimpleNamespace(name="Nincada"))
    battle_state = SimpleNamespace(
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    report = SimpleNamespace(
        classification=PlannerDecisionClass.BEST_AVAILABLE,
        risk=SimpleNamespace(value="high"),
    )
    profile = _SwitchRisk(3, 5, 20, 8, 10, 0.05, 0.40, "imperfect")
    with (
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(8, 12),
        ),
        patch.object(strategy, "_least_bad_trainer_switch_target", return_value=(2, profile)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._safe_battle_healing_action",
            return_value=None,
        ),
        patch.object(strategy, "_remember_battle_support"),
    ):
        action = (TurnAction.UseMove, 0)
        assert strategy._bounded_trainer_high_risk_fallback(battle_state, action, report) is None
        fallback = strategy._bounded_trainer_high_risk_fallback(battle_state, action, report)

    assert fallback == TurnAction.rotate_lead(2)


def test_nuzlocke_rechecks_an_unsafe_free_switch_before_spending_a_turn():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=1, current_hp=4, moves=())
    opponent = SimpleNamespace(party_index=1, level=5, current_hp=20, species=SimpleNamespace(name="Nincada"))
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    strategy._remember_free_trainer_switch(1, opponent, "free trainer replacement")

    util = SimpleNamespace(get_strongest_move_against=lambda _active, _opponent: None)
    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.BattleStrategyUtil", return_value=util),
        patch.object(strategy, "_select_switch_target", return_value=2) as select_target,
    ):
        action = strategy._recheck_after_free_switch(battle_state, 1)

    assert action == (TurnAction.RotateLead, 2)
    assert strategy._free_switch_origin is None
    assert strategy._battle_support_index == 2
    assert select_target.call_args.kwargs["response_turns"] == _EMERGENCY_SWITCH_RESPONSE_TURNS


def test_nuzlocke_trainer_replacement_rejects_exp_tag_against_known_pursuit():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=0)),
    )
    opponent = SimpleNamespace(
        party_index=1,
        level=5,
        species=SimpleNamespace(name="Mightyena"),
    )

    assert (
        strategy._choose_exp_tag_plan_for_opponent(
            battle_state,
            active_index=0,
            opponent=opponent,
            response_moves=("Pursuit",),
        )
        is None
    )


def test_nuzlocke_battle_healing_does_not_use_item_at_full_hp():
    active = SimpleNamespace(current_hp=20, total_hp=20)
    potion = SimpleNamespace(battle_use=ItemBattleUse.Healing, parameter=20, name="Potion")
    slot = SimpleNamespace(quantity=1, item=potion)

    with patch(
        "modules.battle_strategies.nuzlocke_level_balancing.get_item_bag",
        return_value=SimpleNamespace(items=(slot,)),
    ):
        assert _safe_battle_healing_action(
            active,
            response_damage=5,
            critical_response_damage=5,
            active_index=0,
        ) is None


def test_nuzlocke_battle_healing_does_not_spend_turn_when_active_clears_response():
    active = SimpleNamespace(current_hp=17, total_hp=18)
    potion = SimpleNamespace(battle_use=ItemBattleUse.Healing, parameter=20, name="Potion")
    slot = SimpleNamespace(quantity=1, item=potion)

    with patch(
        "modules.battle_strategies.nuzlocke_level_balancing.get_item_bag",
        return_value=SimpleNamespace(items=(slot,)),
    ):
        assert _safe_battle_healing_action(
            active,
            response_damage=5,
            critical_response_damage=5,
            active_index=0,
        ) is None


def test_nuzlocke_battle_healing_uses_potion_before_a_critical_response_can_ko():
    active = SimpleNamespace(current_hp=3, total_hp=20)
    potion = SimpleNamespace(battle_use=ItemBattleUse.Healing, parameter=20, name="Potion")
    slot = SimpleNamespace(quantity=1, item=potion)

    with patch(
        "modules.battle_strategies.nuzlocke_level_balancing.get_item_bag",
        return_value=SimpleNamespace(items=(slot,)),
    ):
        assert _safe_battle_healing_action(
            active,
            response_damage=4,
            critical_response_damage=10,
            active_index=0,
        ) == TurnAction.use_item_on(potion, 0)


def test_nuzlocke_emergency_heal_commits_active_for_current_opponent():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(
        party_index=0,
        current_hp=5,
        total_hp=20,
        status_permanent=None,
    )
    opponent = SimpleNamespace(
        party_index=0,
        current_hp=20,
        species=SimpleNamespace(name="Taillow"),
        level=5,
    )
    battle_state = SimpleNamespace(
        is_trainer_battle=True,
        is_wild=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    healing_action = TurnAction.use_item_on(SimpleNamespace(name="Potion"), 0)

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(active,)),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_has_guaranteed_ko_before_response",
            return_value=False,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._opponent_response_damage",
            return_value=(10, 10),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._active_response_risk_for_line",
            return_value=None,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._known_opponent_can_poison",
            return_value=False,
        ),
        patch("modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state", return_value=None),
        patch.object(BattleStrategyUtil, "can_switch", return_value=False),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._safe_battle_healing_action",
            return_value=healing_action,
        ),
    ):
        assert strategy._risk_aware_emergency_action(battle_state, 0) == healing_action

    assert strategy._battle_support_index == 0
    assert strategy._battle_support_opponent_key == (0, "Taillow", 5)


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


def test_critical_zero_damage_wild_planner_action_is_blocked_when_escape_is_unavailable():
    active_move = SimpleNamespace(move=SimpleNamespace(base_power=0), pp=1)
    active = SimpleNamespace(
        party_index=0,
        current_hp=2,
        total_hp=20,
        moves=(active_move,),
    )
    opponent = SimpleNamespace(current_hp=10)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        is_wild=True,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )
    planner = SimpleNamespace(
        action=PlannerAction.AbortForSafety,
        target=None,
        risk=SimpleNamespace(value="critical"),
        best_available_candidate=SimpleNamespace(action=PlannerAction.UseMove, target=0, label="Leer"),
        classification=PlannerDecisionClass.ABORT,
    )
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    strategy._last_planner_decision = planner

    with (
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(strategy, "_choose_exp_tag_plan", return_value=None),
        patch.object(strategy, "_stronger_battle_support_target", return_value=None),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._lowest_battle_capable_index",
            return_value=0,
        ),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=(TurnAction.UseMove, 0),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=planner,
        ),
        patch.object(BattleStrategyUtil, "can_switch", return_value=False),
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", return_value=SimpleNamespace(max=0)),
        patch.object(BattleStrategyUtil, "get_best_escape_method", return_value=None),
        patch.object(context, "battle_decision_source", None),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.SwitchToManual, None)

    assert _planner_move_is_zero_damage(battle_state, planner)


def test_trainer_projection_executes_legal_action_when_search_budget_is_incomplete():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=1)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active),
    )
    planner = SimpleNamespace(
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.INCOMPLETE,
            complete=False,
            is_safe=False,
            catastrophic_probability=0.0,
        ),
        evaluation_method="expectiminimax",
    )
    strategy._last_planner_decision = planner
    projected_action = (TurnAction.UseMove, 0)

    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.DefaultBattleStrategy.decide_turn",
            return_value=projected_action,
        ),
    ):
        assert strategy.decide_turn(battle_state) == projected_action
        assert context.battle_decision_source == "NUZLOCKE EXPECTIMINIMAX PROJECTION"


def test_wild_projection_executes_legal_action_when_escape_is_unavailable():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=5, total_hp=20)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        is_wild=True,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=20)),
    )
    planner = SimpleNamespace(
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.INCOMPLETE,
            complete=False,
            is_safe=False,
            catastrophic_probability=0.0,
        ),
        evaluation_method="expectiminimax",
        best_available_candidate=SimpleNamespace(label="Pound"),
    )
    projected_action = (TurnAction.UseMove, 0)

    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(strategy, "_choose_exp_tag_plan", return_value=None),
        patch.object(strategy, "_stronger_battle_support_target", return_value=None),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=projected_action,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=planner,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._lowest_battle_capable_index",
            return_value=0,
        ),
        patch.object(BattleStrategyUtil, "get_best_escape_method", return_value=None),
    ):
        assert strategy.decide_turn(battle_state) == projected_action
    assert context.battle_decision_source == "NUZLOCKE STOCHASTIC SEARCH PROJECTION"


def test_preparation_wild_projection_keeps_exp_action_when_escape_is_available():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=20, total_hp=20)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        is_wild=True,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=20)),
    )
    planner = SimpleNamespace(
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.INCOMPLETE,
            complete=False,
            is_safe=False,
            catastrophic_probability=0.0,
        ),
        evaluation_method="expectiminimax",
        best_available_candidate=SimpleNamespace(label="Pound"),
    )
    projected_action = (TurnAction.UseMove, 0)
    strategy._last_planner_decision = planner

    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(strategy, "_choose_exp_tag_plan", return_value=None),
        patch.object(strategy, "_stronger_battle_support_target", return_value=None),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=projected_action,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=planner,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._lowest_battle_capable_index",
            return_value=0,
        ),
        patch.object(BattleStrategyUtil, "get_best_escape_method", return_value=TurnAction.run_away()) as escape,
        patch.object(context, "campaign_preparation_target", "reach preparation level 5"),
    ):
        assert strategy.decide_turn(battle_state) == projected_action

    escape.assert_not_called()
    assert context.battle_decision_source == "NUZLOCKE PREPARATION EXPECTIMINIMAX"


def test_optional_wild_projection_still_escapes_when_escape_is_available():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=20, total_hp=20)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        is_wild=True,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=20)),
    )
    planner = SimpleNamespace(
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.INCOMPLETE,
            complete=False,
            is_safe=False,
            catastrophic_probability=0.0,
        ),
        evaluation_method="expectiminimax",
        best_available_candidate=SimpleNamespace(label="Pound"),
    )
    strategy._last_planner_decision = planner
    escape_action = TurnAction.run_away()

    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(strategy, "_choose_exp_tag_plan", return_value=None),
        patch.object(strategy, "_stronger_battle_support_target", return_value=None),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=(TurnAction.UseMove, 0),
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=planner,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._lowest_battle_capable_index",
            return_value=0,
        ),
        patch.object(BattleStrategyUtil, "get_best_escape_method", return_value=escape_action),
        patch.object(context, "campaign_preparation_target", None),
    ):
        assert strategy.decide_turn(battle_state) == escape_action


def test_preparation_complete_nonfatal_wild_projection_keeps_planner_item_or_attack():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0, current_hp=8, total_hp=20)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=False,
        is_wild=True,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=20)),
    )
    planner = SimpleNamespace(
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.TERMINAL_COMPLETE,
            complete=True,
            is_safe=False,
            catastrophic_probability=0.25,
        ),
        evaluation_method="expectiminimax",
        best_available_candidate=SimpleNamespace(label="Potion"),
    )
    projected_action = (TurnAction.UseItem, (SimpleNamespace(name="Potion"), 0))
    strategy._last_planner_decision = planner

    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(strategy, "_choose_exp_tag_plan", return_value=None),
        patch.object(strategy, "_stronger_battle_support_target", return_value=None),
        patch.object(
            NuzlockeLevelBalancingBattleStrategy.__mro__[1],
            "decide_turn",
            return_value=projected_action,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.plan_battle_state",
            return_value=planner,
        ),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing._lowest_battle_capable_index",
            return_value=0,
        ),
        patch.object(BattleStrategyUtil, "get_best_escape_method", return_value=TurnAction.run_away()) as escape,
        patch.object(context, "campaign_preparation_target", "reach preparation level 5"),
    ):
        assert strategy.decide_turn(battle_state) == projected_action

    escape.assert_not_called()
    assert context.battle_decision_source == "NUZLOCKE PREPARATION EXPECTIMINIMAX"


def test_guaranteed_catastrophic_trainer_projection_reaches_whiteout_lifecycle():
    strategy = NuzlockeLevelBalancingBattleStrategy.__new__(NuzlockeLevelBalancingBattleStrategy)
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_double_battle=False,
        is_trainer_battle=True,
        own_side=SimpleNamespace(active_battler=active),
    )
    strategy._last_planner_decision = SimpleNamespace(
        stochastic_result=SimpleNamespace(
            category=SearchResultCategory.TERMINAL_COMPLETE,
            complete=True,
            is_safe=False,
            catastrophic_probability=1.0,
        ),
        evaluation_method="expectiminimax",
    )
    projected_action = (TurnAction.UseMove, 0)

    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch(
            "modules.battle_strategies.nuzlocke_level_balancing.DefaultBattleStrategy.decide_turn",
            return_value=projected_action,
        ),
    ):
        assert strategy.decide_turn(battle_state) == projected_action
        assert context.battle_decision_source == "NUZLOCKE EXPECTIMINIMAX CATASTROPHIC PROJECTION"


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


def test_capture_weakening_can_accept_critical_ko_when_sequence_failure_is_lower():
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
        assert strategy._get_safe_weakening_move(battle_state) == 0
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

    with (
        patch.object(BattleStrategyUtil, "calculate_move_damage_range", side_effect=damage_range),
        patch.object(strategy, "_capture_response_profile", return_value=(1.0, None, None, "test")),
    ):
        assert strategy._get_safe_weakening_move(battle_state) == 0
    assert "expected capture=" in context.battle_decision_detail


def test_capture_exp_projection_is_zero_without_level_strategy_helpers():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)

    detail = strategy._experience_projection_detail(SimpleNamespace(), {0})

    assert detail == "objective=capture; expected_exp_gain=0"


def test_capture_emergency_switch_uses_zero_exp_diagnostic():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    active = SimpleNamespace(party_index=0, current_hp=1, total_hp=20)
    replacement = SimpleNamespace(current_hp=20, total_hp=20)
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=10)),
        is_trainer_battle=True,
    )

    with (
        patch("modules.battle_strategies.nuzlocke_level_balancing.get_party", return_value=(active, replacement)),
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
        assert strategy._risk_aware_emergency_action(battle_state, 0) == (TurnAction.RotateLead, 1)
    assert "expected_exp_gain=0" in context.battle_decision_detail


def test_capture_treats_damaging_paralysis_move_as_damage_candidate():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    lick = SimpleNamespace(
        name="Lick",
        effect="PARALYZE_HIT",
        base_power=20,
        accuracy=1.0,
    )
    active = SimpleNamespace(
        party_index=0,
        current_hp=20,
        moves=(SimpleNamespace(move=lick, pp=10),),
        disabled_move=None,
    )
    opponent = SimpleNamespace(current_hp=10)
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=opponent),
    )

    with (
        patch.object(
            BattleStrategyUtil,
            "calculate_move_damage_range",
            return_value=SimpleNamespace(min=3, max=3),
        ),
        patch.object(
            BattleStrategyUtil,
            "calculate_catch_success_chance",
            side_effect=(0.2, 0.4),
        ),
        patch.object(strategy, "_capture_response_profile", return_value=(1.0, None, None, "test")),
    ):
        assert strategy._get_safe_weakening_move(battle_state) == 0


def test_capture_rejects_damage_over_time_status_moves():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    poison_fang = SimpleNamespace(
        name="Poison Fang",
        effect="POISON_FANG",
        base_power=50,
        accuracy=1.0,
    )
    active = SimpleNamespace(
        party_index=0,
        current_hp=20,
        moves=(SimpleNamespace(move=poison_fang, pp=10),),
        disabled_move=None,
    )
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=active),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(current_hp=10)),
    )

    with (
        patch.object(BattleStrategyUtil, "calculate_move_damage_range") as damage_range,
        patch.object(strategy, "_capture_response_profile", return_value=(1.0, None, None, "test")),
    ):
        assert strategy._get_safe_weakening_move(battle_state) is None

    damage_range.assert_not_called()


def test_capture_tries_damage_before_standalone_status():
    strategy = NuzlockeCaptureStrategy.__new__(NuzlockeCaptureStrategy)
    battle_state = SimpleNamespace(
        own_side=SimpleNamespace(active_battler=SimpleNamespace(party_index=0)),
        opponent=SimpleNamespace(active_battler=SimpleNamespace(status_permanent=None)),
    )

    with (
        patch.object(strategy, "_risk_aware_emergency_action", return_value=None),
        patch.object(strategy, "_stronger_capture_switch_target", return_value=None),
        patch.object(strategy, "capture_target_is_authorized", return_value=True),
        patch.object(strategy, "_get_best_poke_ball", return_value=SimpleNamespace(name="Poké Ball")),
        patch.object(strategy, "_get_safe_weakening_move", return_value=0),
        patch.object(
            strategy,
            "_get_best_status_changing_move",
            side_effect=AssertionError("status should be considered after damage"),
        ),
    ):
        assert strategy.decide_turn(battle_state) == (TurnAction.UseMove, 0)


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


def test_emerald_intro_rival_delegates_the_opening_action_to_the_planner():
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

    expected = (TurnAction.UseMove, 0)
    with patch.object(EmeraldIntroRivalBattleStrategy.__mro__[1], "decide_turn", return_value=expected) as fallback:
        assert strategy.decide_turn(battle_state) == expected
        fallback.assert_called_once_with(battle_state)


def test_emerald_intro_rival_uses_the_nuzlocke_emergency_action_path():
    party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_trainer_battle=True,
        is_double_battle=False,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    expected = TurnAction.use_item_on(SimpleNamespace(name="Potion"), 0)

    with patch("modules.battle_strategies.default.get_party", return_value=party):
        strategy = EmeraldIntroRivalBattleStrategy()
    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch("modules.battle_strategies.default.DefaultBattleStrategy.decide_turn", return_value=(TurnAction.UseMove, 0)),
        patch.object(strategy, "_risk_aware_emergency_action", return_value=expected) as emergency,
    ):
        assert strategy.decide_turn(battle_state) == expected

    emergency.assert_called_once_with(battle_state, 0)


def test_complete_expectiminimax_item_action_is_not_replaced_by_nuzlocke_emergency_policy():
    party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
    active = SimpleNamespace(party_index=0)
    battle_state = SimpleNamespace(
        is_trainer_battle=True,
        is_double_battle=False,
        nuzlocke_capture_target=False,
        own_side=SimpleNamespace(active_battler=active),
    )
    expected = TurnAction.use_item_on(SimpleNamespace(name="Potion"), 0)
    planner_result = SimpleNamespace(complete=True, is_safe=False)
    planner = SimpleNamespace(stochastic_result=planner_result)

    with patch("modules.battle_strategies.default.get_party", return_value=party):
        strategy = EmeraldIntroRivalBattleStrategy()
    strategy._last_planner_decision = planner
    with (
        patch.object(strategy, "_publish_expectiminimax_experience_facts"),
        patch.object(strategy, "_sync_opponent_participation"),
        patch.object(strategy, "_trace_experience_projection"),
        patch.object(strategy, "_recheck_after_free_switch", return_value=None),
        patch(
            "modules.battle_strategies.default.DefaultBattleStrategy.decide_turn",
            return_value=expected,
        ),
        patch.object(strategy, "_risk_aware_emergency_action") as emergency,
    ):
        assert strategy.decide_turn(battle_state) == expected

    emergency.assert_not_called()


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
