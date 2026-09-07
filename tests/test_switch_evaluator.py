from dataclasses import replace

from modules.battle_strategies.switch_evaluator import (
    SwitchCandidateFacts,
    SwitchEvaluationContext,
    evaluate_switch_candidate,
    rank_switch_candidates,
)


def _candidate(index, *, survival, training, damage, hp=20, incoming=2, critical=None, known=True):
    return SwitchCandidateFacts(
        party_index=index,
        hp=hp,
        max_hp=hp,
        opponent_hp=20,
        damage_min=max(1, damage // 2),
        damage_max=damage,
        incoming_damage_max=incoming,
        incoming_critical_damage_max=critical,
        critical_ko_probability=None if survival is None else max(0.0, 1.0 - survival),
        survival_probability=survival,
        response_facts_known=known,
        training_priority=training,
    )


def test_policy_context_changes_ranking_without_changing_candidate_facts():
    candidates = (
        _candidate(0, survival=0.90, training=1.0, damage=2, incoming=5, critical=25),
        _candidate(1, survival=1.0, training=0.0, damage=20, incoming=2, critical=4),
    )

    leveling = rank_switch_candidates(candidates, SwitchEvaluationContext.LEVELING)
    emergency = rank_switch_candidates(candidates, SwitchEvaluationContext.EMERGENCY)

    assert leveling[0].party_index == 0
    assert emergency[0].party_index == 1
    assert leveling[0].component_map["training_priority"] > leveling[1].component_map["training_priority"]


def test_safety_gates_cannot_be_overridden_by_training_reward():
    no_damage = evaluate_switch_candidate(
        _candidate(0, survival=1.0, training=1.0, damage=0),
        SwitchEvaluationContext.LEVELING,
    )
    unsafe = evaluate_switch_candidate(
        _candidate(1, survival=0.20, training=1.0, damage=20, incoming=20, critical=40),
        SwitchEvaluationContext.LEVELING,
    )

    assert not no_damage.eligible
    assert "no effective damaging move" in no_damage.reason
    assert not unsafe.eligible
    assert "survival probability" in unsafe.reason


def test_active_party_slots_are_a_hard_switch_exclusion():
    result = evaluate_switch_candidate(
        replace(_candidate(1, survival=1.0, training=1.0, damage=20), active_party_indices=frozenset({1})),
        SwitchEvaluationContext.LEVELING,
    )

    assert not result.eligible
    assert result.reason == "replacement is already active in battle"


def test_trainer_replacement_can_compare_unknown_response_with_explicit_uncertainty_cost():
    known = evaluate_switch_candidate(
        _candidate(0, survival=1.0, training=0.0, damage=8, known=True),
        SwitchEvaluationContext.TRAINER_REPLACEMENT,
    )
    unknown = evaluate_switch_candidate(
        _candidate(1, survival=None, training=0.0, damage=8, known=False),
        SwitchEvaluationContext.TRAINER_REPLACEMENT,
    )

    assert known.eligible
    assert unknown.eligible
    assert unknown.score < known.score
    assert unknown.component_map["unknown_response"] == 1.0
