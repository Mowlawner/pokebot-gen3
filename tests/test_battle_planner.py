import unittest
from types import SimpleNamespace
from unittest.mock import patch
from modules.battle_observation import BattleKnowledge, KnowledgePolicy
from modules.battle_planner import (
    BattlePlanner,
    PlannerAction,
    PlannerConfidence,
    PlannerContext,
    PlannerDecisionClass,
    PlannerEvidenceKind,
    PlannerSafety,
    PlannerMove,
    PlannerItem,
    PlannerOpponentMove,
    PlannerRisk,
    PlannerSwitch,
    knowledge_for_planner,
    plan_battle_state,
    format_planner_decision_diagnostic,
)
from modules.battle_strategies.default import DefaultBattleStrategy
from modules.battle_strategies import TurnAction


def context(**changes):
    values = dict(
        active_name="Treecko",
        active_hp=20,
        active_max_hp=20,
        opponent_name="Wurmple",
        opponent_hp=12,
        moves=(PlannerMove(0, "Scratch", 12, 15),),
    )
    values.update(changes)
    return PlannerContext(**values)


class BattlePlannerTests(unittest.TestCase):
    def test_capture_target_prefers_nonlethal_move_over_guaranteed_ko(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=10,
                moves=(PlannerMove(0, "Strong", 10, 14), PlannerMove(1, "Weak", 3, 5)),
                capture_target=True,
                capture_available=True,
                capture_appropriate_hp=2,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 1)

    def test_capture_target_prefers_ball_at_capture_hp(self):
        ball = SimpleNamespace(name="Poké Ball")
        decision = BattlePlanner().plan(
            context(
                opponent_hp=2,
                capture_target=True,
                capture_available=True,
                capture_appropriate_hp=3,
                items=(PlannerItem(ball, 5, 0, True),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseItem)
        self.assertEqual(decision.target, (ball, 0))

    def test_capture_target_with_only_lethal_attacks_uses_ball(self):
        ball = SimpleNamespace(name="Poké Ball")
        decision = BattlePlanner().plan(
            context(
                opponent_hp=10,
                moves=(PlannerMove(0, "Strong", 10, 14),),
                capture_target=True,
                capture_available=True,
                capture_appropriate_hp=1,
                items=(PlannerItem(ball, 1, 0, True),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseItem)

    def test_best_available_diagnostic_explains_unknown_response_and_rejected_move(self):
        battle_context = context(
            opponent_hp=30,
            moves=(PlannerMove(0, "Pound", 8, 10), PlannerMove(1, "Tail Whip", 0, 0, effect="DEFENSE_DOWN")),
        )
        decision = BattlePlanner().plan(battle_context)
        diagnostic = format_planner_decision_diagnostic(decision, battle_context)
        self.assertIn("BEST_AVAILABLE", diagnostic)
        self.assertIn("No fully safe line established", diagnostic)
        self.assertIn("Pound", diagnostic)
        self.assertIn("opponent_response_unknown=True", diagnostic)
        self.assertIn("setup_line=False", diagnostic)

    def test_default_strategy_consults_planner_above_and_below_hp_threshold(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        active = SimpleNamespace(current_hp_percentage=80)
        battle_state = SimpleNamespace(own_side=SimpleNamespace(active_battler=active))
        safe_attack = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=2,
            confidence=PlannerConfidence.HIGH,
            rationale="safe",
            is_safe_to_execute=True,
        )
        safe_switch = SimpleNamespace(
            action=PlannerAction.SwitchPokemon,
            target=1,
            confidence=PlannerConfidence.MEDIUM,
            rationale="switch",
            is_safe_to_execute=True,
        )
        util = SimpleNamespace()
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil", return_value=util
        ), patch(
            "modules.battle_strategies.default.plan_battle_state", side_effect=[safe_attack, safe_switch]
        ) as planner:
            strategy = DefaultBattleStrategy()
            self.assertEqual(strategy.decide_turn(battle_state), (TurnAction.UseMove, 2))

            active.current_hp_percentage = 5
            self.assertEqual(strategy.decide_turn(battle_state), (TurnAction.RotateLead, 1))

        self.assertEqual(planner.call_count, 2)

    def test_ready_legal_nuzlocke_battle_delegates_to_existing_capture_strategy(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=(SimpleNamespace(quantity=1),))
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True,
            own_side=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        expected = (TurnAction.UseItem, SimpleNamespace(name="Poké Ball"))
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn", return_value=expected
        ) as capture_turn, patch(
            "modules.battle_strategies.default.plan_battle_state",
            return_value=SimpleNamespace(
                action=PlannerAction.UseItem, classification=PlannerDecisionClass.SAFE, is_safe_to_execute=True
            ),
        ):
            result = DefaultBattleStrategy().decide_turn(battle_state)
        self.assertIs(result[0], TurnAction.UseItem)
        self.assertEqual(result[1].name, "Poké Ball")
        capture_turn.assert_called_once_with(battle_state)

    def test_legal_target_uses_safe_planner_move_before_capture(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=(SimpleNamespace(quantity=1),))
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True, own_side=SimpleNamespace(active_battler=SimpleNamespace())
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove, target=1, classification=PlannerDecisionClass.SAFE, is_safe_to_execute=True
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=decision), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn"
        ) as catch:
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 1))
        catch.assert_not_called()

    def test_legal_target_falls_back_to_capture_when_planner_is_not_safe(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=(SimpleNamespace(quantity=1),))
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True, own_side=SimpleNamespace(active_battler=SimpleNamespace())
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=1,
            classification=PlannerDecisionClass.BEST_AVAILABLE,
            is_safe_to_execute=False,
        )
        expected = (TurnAction.UseItem, SimpleNamespace(name="Poké Ball"))
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=decision), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn", return_value=expected
        ):
            result = DefaultBattleStrategy().decide_turn(battle_state)
        self.assertIs(result[0], TurnAction.UseItem)

    def test_legal_target_without_balls_returns_to_normal_battle_policy(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        bag = SimpleNamespace(poke_balls=())
        active = SimpleNamespace(current_hp_percentage=80)
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=True,
            own_side=SimpleNamespace(active_battler=active),
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=1,
            confidence=PlannerConfidence.HIGH,
            classification=PlannerDecisionClass.SAFE,
            is_safe_to_execute=True,
            rationale="ordinary battle action",
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.get_item_bag", return_value=bag
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=decision), patch(
            "modules.battle_strategies.catch.CatchStrategy.decide_turn"
        ) as catch:
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 1))
        catch.assert_not_called()

    def test_ready_non_target_does_not_delegate_to_capture_strategy(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        active = SimpleNamespace(current_hp_percentage=80)
        battle_state = SimpleNamespace(
            nuzlocke_capture_target=False,
            own_side=SimpleNamespace(active_battler=active),
        )
        decision = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=0,
            confidence=PlannerConfidence.HIGH,
            rationale="safe",
            is_safe_to_execute=True,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.plan_battle_state", return_value=decision
        ), patch("modules.battle_strategies.catch.CatchStrategy.decide_turn") as capture_turn:
            result = DefaultBattleStrategy().decide_turn(battle_state)
        self.assertEqual(result, (TurnAction.UseMove, 0))
        capture_turn.assert_not_called()

    def test_safe_planner_actions_keep_existing_turn_action_shapes(self):
        item = SimpleNamespace(name="Potion")
        cases = (
            (PlannerAction.UseMove, 0, (TurnAction.UseMove, 0)),
            (PlannerAction.SwitchPokemon, 2, (TurnAction.RotateLead, 2)),
            (PlannerAction.UseItem, (item, 0), (TurnAction.UseItem, (item, 0))),
            (PlannerAction.RunAway, None, (TurnAction.RunAway, None)),
        )
        for action, target, expected in cases:
            decision = SimpleNamespace(action=action, target=target)
            self.assertEqual(DefaultBattleStrategy._turn_action_from_planner_decision(decision), expected)

    def test_best_available_planner_action_is_accepted_without_safety_flag(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp_percentage=80)),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        best = SimpleNamespace(
            action=PlannerAction.UseMove,
            target=1,
            confidence=PlannerConfidence.MEDIUM,
            rationale="unknown response",
            is_safe_to_execute=False,
            classification=PlannerDecisionClass.BEST_AVAILABLE,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil"
        ) as util, patch("modules.battle_strategies.default.plan_battle_state", return_value=best):
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 1))
            util.return_value.pokemon_has_enough_hp.assert_not_called()

    def test_unsafe_planner_result_keeps_legacy_attack_fallback_above_threshold(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=SimpleNamespace(current_hp_percentage=80)),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        util = SimpleNamespace(
            pokemon_has_enough_hp=lambda pokemon: True,
            get_strongest_move_against=lambda active, opponent: 3,
        )
        unsafe = SimpleNamespace(
            action=PlannerAction.AbortForSafety,
            target=None,
            confidence=PlannerConfidence.INSUFFICIENT,
            rationale="unknown response",
            is_safe_to_execute=False,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil", return_value=util
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=unsafe):
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 3))

    def test_live_party_switch_candidates_use_ordered_slot_not_party_index(self):
        class PartyPokemonWithoutPartyIndex:
            def __init__(self, name, hp):
                self.name = name
                self.current_hp = hp
                self.total_hp = 20
                self.is_egg = False
                self.moves = ()

        move = SimpleNamespace(name="Scratch", base_power=40)
        active_move = SimpleNamespace(move=move, pp=1)
        active = SimpleNamespace(
            party_index=0,
            species=SimpleNamespace(name="Treecko"),
            current_hp=2,
            total_hp=20,
            moves=(active_move,),
            disabled_move=None,
        )
        opponent = SimpleNamespace(
            species=SimpleNamespace(name="Wurmple"),
            current_hp=12,
            moves=(),
        )
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=active),
            opponent=SimpleNamespace(active_battler=opponent),
            is_trainer_battle=False,
        )
        alternate = PartyPokemonWithoutPartyIndex("Torchic", 18)
        party = [PartyPokemonWithoutPartyIndex("Treecko", 2), alternate]
        damage_range = SimpleNamespace(min=3, max=4)
        util = SimpleNamespace(
            calculate_move_damage_range=lambda *args: damage_range,
            can_switch=lambda: True,
            get_escape_chance=lambda: 0,
        )
        knowledge = BattleKnowledge(
            "battle-1",
            KnowledgePolicy.HUMAN_KNOWLEDGE,
            revealed_opponent_moves=((("opponent", None, None), "Tackle"),),
        )

        with patch("modules.pokemon_party.get_party", return_value=party), patch(
            "modules.battle_strategies._util.BattleStrategyUtil", return_value=util
        ), patch("modules.pokemon.get_move_by_name", return_value=move):
            decision = plan_battle_state(battle_state, knowledge)

        self.assertEqual(decision.action, PlannerAction.SwitchPokemon)
        self.assertEqual(decision.target, 1)

    def test_safe_attack_is_high_confidence(self):
        decision = BattlePlanner().plan(context())
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.confidence, PlannerConfidence.HIGH)

    def test_immediate_ko_rationale_contains_known_and_derived_evidence(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                active_max_hp=20,
                opponent_name="Torchic",
                opponent_hp=4,
                moves=(PlannerMove(0, "Pound", 4, 5),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.safety, PlannerSafety.SAFE)
        self.assertIn("Pound", decision.rationale)
        self.assertIn("4-5 damage", decision.rationale)
        self.assertIn("8/20", decision.rationale)
        self.assertIn("guaranteed KO", decision.rationale)
        self.assertIn(PlannerEvidenceKind.KNOWN, {item.kind for item in decision.evidence})
        self.assertIn(PlannerEvidenceKind.DERIVED, {item.kind for item in decision.evidence})

    def test_guaranteed_immediate_ko_is_low_risk_safe(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=4,
                moves=(PlannerMove(0, "Pound", 4, 5),),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.SAFE)
        self.assertEqual(decision.risk, PlannerRisk.LOW)

    def test_risky_attack_is_ranked_against_a_worse_legal_attack(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=5,
                opponent_hp=12,
                moves=(
                    PlannerMove(0, "Pound", 4, 5),
                    PlannerMove(1, "Scratch", 2, 3),
                ),
                opponent_moves=(PlannerOpponentMove("Tackle", 3, 3),),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.target, 0)

    def test_critical_hit_ceiling_makes_winning_line_risky(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Tackle", 2, 2, critical_damage_max=10),),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.risk, PlannerRisk.CRITICAL)
        self.assertIn("critical-hit ceiling", decision.rationale)

    def test_safe_multi_turn_attack_is_established_from_ranges(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5, accuracy=1.0),),
                opponent_moves=(PlannerOpponentMove("Ember", 3, 4),),
                active_speed=30,
                opponent_speed=20,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertIn("estimated 3 hits", decision.rationale)
        self.assertIn("safe sequence established", decision.rationale)

    def test_multi_turn_attack_is_rejected_when_exchange_can_faint_player(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Ember", 4, 5),),
                active_speed=20,
                opponent_speed=30,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.safety, PlannerSafety.UNSAFE_OR_UNCERTAIN)
        self.assertIn("8/20 HP", decision.rationale)
        self.assertIn("projected incoming damage 10", decision.rationale)
        self.assertTrue(decision.rejected_candidates)

    def test_multi_turn_damage_range_uses_minimum_for_hit_count(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=13,
                moves=(PlannerMove(0, "Pound", 4, 9),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertIn("estimated 4 hits", decision.rationale)

    def test_accuracy_can_make_a_multi_turn_sequence_unsafe(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=12,
                moves=(PlannerMove(0, "Low Accuracy", 4, 5, accuracy=0.9),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertIn("accuracy", decision.rationale)

    def test_unknown_opponent_response_is_not_collapsed_into_generic_failure(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertFalse(decision.is_safe_to_execute)
        self.assertIn("opponent response is unknown", decision.rationale)
        self.assertIn(PlannerEvidenceKind.UNKNOWN, {item.kind for item in decision.evidence})

    def test_priority_and_speed_change_whether_final_hit_gets_a_response(self):
        first = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=8,
                moves=(PlannerMove(0, "Quick Attack", 4, 4, priority=1),),
                opponent_moves=(PlannerOpponentMove("Tackle", 4, 4, priority=0),),
                active_speed=1,
                opponent_speed=99,
            )
        )
        second = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=8,
                moves=(PlannerMove(0, "Pound", 4, 4),),
                opponent_moves=(PlannerOpponentMove("Tackle", 4, 4),),
                active_speed=1,
                opponent_speed=99,
            )
        )
        self.assertEqual(first.action, PlannerAction.UseMove)
        self.assertEqual(second.action, PlannerAction.UseMove)
        self.assertEqual(second.classification, PlannerDecisionClass.BEST_AVAILABLE)

    def test_multiple_known_opponent_moves_use_the_most_dangerous_one(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=8,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(
                    PlannerOpponentMove("Growl", 0, 0),
                    PlannerOpponentMove("Ember", 4, 6),
                ),
                active_speed=30,
                opponent_speed=20,
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)

    def test_unsupported_mechanics_do_not_look_like_safe_damage(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=12,
                moves=(PlannerMove(0, "Counter", 4, 8, mechanics_supported=False),),
                opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.AbortForSafety)
        self.assertIn("unsupported", decision.rationale)

    def test_low_hp_prefers_a_clearly_safer_switch(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=2,
                active_max_hp=20,
                opponent_damage_max=8,
                moves=(PlannerMove(0, "Scratch", 3, 5),),
                can_switch=True,
                switches=(PlannerSwitch(1, "Torchic", 18, 20, 5, 10),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.SwitchPokemon)
        self.assertEqual(decision.target, 1)

    def test_low_hp_still_attacks_when_ko_is_guaranteed(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=4,
                active_max_hp=20,
                opponent_hp=8,
                moves=(PlannerMove(0, "Scratch", 8, 11),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)

    def test_wild_escape_is_used_when_other_actions_are_not_safe(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=1,
                active_max_hp=20,
                moves=(),
                is_wild=True,
                can_run=True,
                escape_confidence=PlannerConfidence.HIGH,
            )
        )
        self.assertEqual(decision.action, PlannerAction.RunAway)

    def test_trainer_battle_never_runs(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=1,
                active_max_hp=20,
                moves=(),
                is_trainer=True,
                can_run=True,
                escape_confidence=PlannerConfidence.HIGH,
            )
        )
        self.assertEqual(decision.action, PlannerAction.AbortForSafety)

    def test_no_safe_action_is_explicit(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=1,
                active_max_hp=20,
                moves=(),
            )
        )
        self.assertEqual(decision.action, PlannerAction.AbortForSafety)
        self.assertEqual(decision.confidence, PlannerConfidence.INSUFFICIENT)

    def test_best_available_damaging_line_is_explicit_when_worst_case_is_unsafe(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=5,
                opponent_hp=12,
                moves=(PlannerMove(0, "Pound", 4, 5),),
                opponent_moves=(PlannerOpponentMove("Ember", 4, 5),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertIn("No guaranteed-safe sequence established", decision.rationale)

    def test_supported_setup_move_can_beat_repeated_damage(self):
        decision = BattlePlanner().plan(
            context(
                active_hp=20,
                opponent_hp=10,
                moves=(
                    PlannerMove(0, "Pound", 2, 2),
                    PlannerMove(1, "Tail Whip", 0, 0, effect="DEFENSE_DOWN_2"),
                ),
                opponent_moves=(PlannerOpponentMove("Ember", 2, 2),),
            )
        )
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 1)
        self.assertIn("stat/effect changes", decision.rationale)

    def test_full_emulator_facts_are_not_forwarded_as_planner_knowledge(self):
        knowledge = BattleKnowledge("b", KnowledgePolicy.FULL_EMULATOR, known_opponent_moves=(("id", "Thunder"),))
        safe = knowledge_for_planner(knowledge)
        self.assertEqual(safe.policy, KnowledgePolicy.HUMAN_KNOWLEDGE)
        self.assertEqual(safe.known_opponent_moves, ())

    def test_fallback_rationale_identifies_legacy_path_and_does_not_claim_ko(self):
        party = SimpleNamespace(first_non_fainted=SimpleNamespace(index=0))
        active = SimpleNamespace(current_hp_percentage=80, moves=())
        battle_state = SimpleNamespace(
            own_side=SimpleNamespace(active_battler=active),
            opponent=SimpleNamespace(active_battler=SimpleNamespace()),
        )
        util = SimpleNamespace(
            pokemon_has_enough_hp=lambda pokemon: True,
            get_strongest_move_against=lambda active, opponent: 3,
        )
        unsafe = SimpleNamespace(
            action=PlannerAction.AbortForSafety,
            target=None,
            confidence=PlannerConfidence.INSUFFICIENT,
            rationale="safe sequence cannot be established",
            is_safe_to_execute=False,
        )
        with patch("modules.battle_strategies.default.get_party", return_value=party), patch(
            "modules.battle_strategies.default.BattleStrategyUtil", return_value=util
        ), patch("modules.battle_strategies.default.plan_battle_state", return_value=unsafe):
            self.assertEqual(DefaultBattleStrategy().decide_turn(battle_state), (TurnAction.UseMove, 3))
        self.assertIn("LEGACY FALLBACK", __import__("modules.context", fromlist=["context"]).context.message)
        self.assertIn("legacy selector chose", __import__("modules.context", fromlist=["context"]).context.message)

    def test_unknown_response_gets_best_available_attack(self):
        decision = BattlePlanner().plan(
            context(
                opponent_hp=30,
                moves=(PlannerMove(0, "Pound", 8, 10), PlannerMove(1, "Scratch", 5, 12)),
            )
        )
        self.assertEqual(decision.classification, PlannerDecisionClass.BEST_AVAILABLE)
        self.assertEqual(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 0)
        self.assertFalse(decision.is_safe_to_execute)


if __name__ == "__main__":
    unittest.main()
