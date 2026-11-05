import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from modules.battle_capture_planner import (
    CaptureConfig,
    choose_capture_plan,
    evaluate_capture_plans,
)
from modules.battle_planner import (
    BattlePlanner,
    PlannerAction,
    PlannerContext,
    PlannerDecisionClass,
    PlannerMove,
    PlannerOpponentMove,
    PlannerSafety,
    RecoveryItem,
)


def catch_probability(hp: int, max_hp: int, rate: int = 255, ball: float = 1.0) -> float:
    """The Gen III catch formula, so these tests use realistic odds."""

    a = (3 * max_hp - 2 * hp) * rate * ball / (3 * max_hp)
    if a >= 255:
        return 1.0
    b = 1048560 / math.sqrt(math.sqrt(16711680 / a))
    return (b / 65536) ** 4


def ball(quantity: int):
    return RecoveryItem(SimpleNamespace(name="Poke Ball"), quantity, 0, True)


def capture_context(**changes) -> PlannerContext:
    max_hp = changes.pop("target_max_hp", 19)
    values = dict(
        active_name="Lead",
        active_hp=40,
        active_max_hp=40,
        opponent_name="Wurmple",
        opponent_hp=19,
        opponent_max_hp=max_hp,
        moves=(PlannerMove(0, "Tackle", 3, 4),),
        items=(ball(5),),
        opponent_moves=(PlannerOpponentMove("Tackle", 3, 5, accuracy=0.95, critical_damage_max=10),),
        active_speed=20,
        opponent_speed=10,
        is_wild=True,
        capture_target=True,
        capture_available=True,
        capture_ball_count=5,
        battle_objective="capture",
        campaign_owned=True,
        opponent_response_facts_complete=True,
        capture_probability_for_state=lambda state, item: catch_probability(state.opponent_hp, max_hp),
    )
    values.update(changes)
    return PlannerContext(**values)


class CapturePlanChoiceTests(unittest.TestCase):
    def test_a_sturdy_lead_weakens_the_target_before_throwing(self):
        choice = choose_capture_plan(capture_context())

        self.assertIsNotNone(choice)
        self.assertGreater(choice.plan.weaken_turns, 0)
        self.assertEqual(choice.plan.move_name, "Tackle")
        self.assertGreater(choice.plan.capture_probability, 0.95)
        self.assertEqual(choice.plan.target_ko_probability, 0.0)

    def test_weakening_beats_throwing_at_full_hp(self):
        plans, _ = evaluate_capture_plans(capture_context())
        throw_now = next(plan for plan in plans if plan.weaken_turns == 0)
        best = max(plans, key=lambda plan: plan.capture_probability)

        self.assertGreater(best.capture_probability, throw_now.capture_probability + 0.1)

    def test_a_target_already_low_is_thrown_at_not_attacked(self):
        choice = choose_capture_plan(capture_context(opponent_hp=3))

        self.assertEqual(choice.plan.weaken_turns, 0)
        self.assertEqual(choice.plan.first_action, "throw")

    def test_a_move_that_would_knock_out_the_target_is_not_used(self):
        # 8-12 damage against 10 HP knocks the target out most of the time.
        choice = choose_capture_plan(
            capture_context(opponent_hp=10, moves=(PlannerMove(0, "Slam", 8, 12),))
        )

        self.assertEqual(choice.plan.weaken_turns, 0)

    def test_false_swipe_is_used_because_it_cannot_knock_out(self):
        choice = choose_capture_plan(
            capture_context(
                opponent_hp=10,
                moves=(PlannerMove(0, "False Swipe", 8, 12, effect="FALSE_SWIPE"),),
            )
        )

        self.assertGreater(choice.plan.weaken_turns, 0)
        self.assertEqual(choice.plan.target_ko_probability, 0.0)

    def test_fewer_balls_make_weakening_more_valuable(self):
        many = choose_capture_plan(capture_context(items=(ball(10),), capture_ball_count=10)).plan
        one_ball = choose_capture_plan(capture_context(items=(ball(1),), capture_ball_count=1)).plan

        self.assertGreaterEqual(one_ball.weaken_turns, many.weaken_turns)
        self.assertGreater(one_ball.weaken_turns, 0)

    def test_it_defers_when_no_plan_keeps_the_lead_safe(self):
        # A 5 HP lead against 6-7 damage cannot be kept safe by any plan.
        context = capture_context(
            active_hp=5,
            active_max_hp=5,
            opponent_moves=(PlannerOpponentMove("Tackle", 6, 7),),
        )

        self.assertIsNone(choose_capture_plan(context))

    def test_a_fragile_lead_is_deferred_rather_than_gambled_with(self):
        # The live Woods case: a 13 HP lead against a Tackle that can crit for 14.
        context = capture_context(
            active_hp=13,
            active_max_hp=13,
            opponent_moves=(PlannerOpponentMove("Tackle", 5, 7, accuracy=0.95, critical_damage_max=14),),
        )
        plans, _ = evaluate_capture_plans(context)

        self.assertTrue(all(plan.active_faint_probability > 0.03 for plan in plans))
        self.assertIsNone(choose_capture_plan(context))

    def test_probabilities_are_consistent(self):
        plans, _ = evaluate_capture_plans(capture_context(active_hp=14, active_max_hp=14))

        for plan in plans:
            with self.subTest(plan=plan.summary()):
                total = plan.capture_probability + plan.active_faint_probability + plan.target_ko_probability
                self.assertLessEqual(total, 1.0 + 1e-9)
                for value in (plan.capture_probability, plan.active_faint_probability, plan.target_ko_probability):
                    self.assertGreaterEqual(value, 0.0)

    def test_the_tradeoff_between_ko_risk_and_running_out_of_balls_is_weighed(self):
        # With one ball, taking some KO risk to land the throw at low HP wins.
        plans, _ = evaluate_capture_plans(capture_context(items=(ball(1),), capture_ball_count=1))
        throw_now = next(plan for plan in plans if plan.weaken_turns == 0)
        risky = max(plans, key=lambda plan: plan.capture_probability)

        self.assertGreater(risky.capture_probability, throw_now.capture_probability)
        self.assertGreaterEqual(risky.target_ko_probability, 0.0)


class CapturePreconditionTests(unittest.TestCase):
    def assertDefers(self, **changes):
        self.assertIsNone(choose_capture_plan(capture_context(**changes)))

    def test_not_a_capture_objective(self):
        self.assertDefers(battle_objective="defeat")

    def test_trainer_battle(self):
        self.assertDefers(is_trainer=True)

    def test_not_a_legal_capture_target(self):
        self.assertDefers(capture_target=False)

    def test_no_usable_ball(self):
        self.assertDefers(capture_available=False)
        self.assertDefers(items=(ball(0),))

    def test_no_capture_probability_model(self):
        self.assertDefers(capture_probability_for_state=None)

    def test_incomplete_opponent_facts_cannot_support_a_safety_claim(self):
        self.assertDefers(opponent_response_facts_complete=False)

    def test_unsupported_opponent_moves(self):
        self.assertDefers(unsupported_opponent_moves=("Counter",))

    def test_unknown_opponent_damage(self):
        self.assertDefers(opponent_moves=(), opponent_damage_max=None)

    def test_a_bare_damage_ceiling_is_enough_to_model_the_response(self):
        self.assertIsNotNone(
            choose_capture_plan(capture_context(opponent_moves=(), opponent_damage_max=4))
        )

    def test_status_and_unsupported_moves_are_never_used_to_weaken(self):
        context = capture_context(
            opponent_hp=19,
            moves=(
                PlannerMove(0, "Growl", 0, 0, effect="ATTACK_DOWN"),
                PlannerMove(1, "Counter", 5, 6, mechanics_supported=False),
                PlannerMove(2, "Explosion", 100, 120, effect="EXPLOSION"),
                PlannerMove(3, "Bide", 0, 0, effect="BIDE"),
            ),
        )
        plans, _ = evaluate_capture_plans(context)

        self.assertEqual([plan.weaken_turns for plan in plans], [0])

    def test_an_exhausted_move_is_not_used(self):
        plans, _ = evaluate_capture_plans(capture_context(moves=(PlannerMove(0, "Tackle", 3, 4, pp=0),)))

        self.assertEqual([plan.weaken_turns for plan in plans], [0])

    def test_planning_is_fast_even_with_many_balls_and_moves(self):
        import time

        context = capture_context(
            items=(ball(99),),
            capture_ball_count=99,
            moves=tuple(PlannerMove(index, f"Move{index}", 2 + index, 5 + index) for index in range(4)),
        )
        started = time.perf_counter()
        choose_capture_plan(context)

        self.assertLess(time.perf_counter() - started, 1.0)


class CapturePlannerIntegrationTests(unittest.TestCase):
    def test_the_planner_uses_the_capture_solver_for_a_wild_capture(self):
        decision = BattlePlanner().plan(capture_context())

        self.assertEqual(decision.evaluation_method, "capture_planner")
        self.assertIs(decision.action, PlannerAction.UseMove)
        self.assertEqual(decision.target, 0)
        self.assertIs(decision.safety, PlannerSafety.SAFE)
        self.assertIs(decision.classification, PlannerDecisionClass.SAFE)
        self.assertTrue(decision.is_safe_to_execute)
        self.assertIn("capture=", decision.rationale)

    def test_a_ball_decision_targets_the_ball_and_the_active_slot(self):
        context = capture_context(opponent_hp=3, active_party_index=2)
        decision = BattlePlanner().plan(context)

        self.assertEqual(decision.evaluation_method, "capture_planner")
        self.assertIs(decision.action, PlannerAction.UseItem)
        item, party_index = decision.target
        self.assertEqual(item.name, "Poke Ball")
        self.assertEqual(party_index, 2)

    def test_the_generic_search_still_owns_trainer_and_defeat_battles(self):
        for changes in (dict(is_trainer=True), dict(battle_objective="defeat")):
            with self.subTest(changes=changes):
                decision = BattlePlanner().plan(capture_context(**changes))
                self.assertNotEqual(decision.evaluation_method, "capture_planner")

    def test_a_deferred_capture_falls_back_to_the_generic_search(self):
        context = capture_context(
            active_hp=5,
            active_max_hp=5,
            opponent_moves=(PlannerOpponentMove("Tackle", 6, 7),),
        )
        decision = BattlePlanner().plan(context)

        self.assertNotEqual(decision.evaluation_method, "capture_planner")


if __name__ == "__main__":
    unittest.main()
