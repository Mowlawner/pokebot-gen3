from types import SimpleNamespace

from modules.battle_expectiminimax import (
    Expectiminimax,
    OpponentPolicy,
    SearchConfig,
    SearchResultCategory,
    SearchMove,
    SearchItem,
    SearchSwitch,
    SearchState,
    _Value,
    _acts_first,
    _apply_move_damage,
    _apply_move_effect,
    _move_outcomes,
    _move_outcomes_for_state,
    _escape_probability,
    _failed_escape_state,
    search_planner_context,
)
from modules.battle_planner import PlannerContext, PlannerMove, PlannerOpponentMove, PlannerSwitch, RecoveryItem


def test_expectiminimax_keeps_misses_and_critical_hits_as_uncertainty():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10, accuracy=0.5),),
        opponent_moves=(SearchMove(0, "Tackle", 20, 20),),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    assert result.action_name == "Tackle"
    assert 0 < result.win_probability < 1
    assert result.loss_probability > 0
    assert not result.is_safe
    assert result.catastrophic_probability == result.loss_probability


def test_incomplete_search_keeps_immediate_hit_miss_critical_and_response_branches():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=3,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 4, 8, accuracy=0.5),),
        opponent_moves=(
            SearchMove(0, "Ember", 3, 5, secondary_chance=0.5, secondary_effect="BURN_HIT"),
        ),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=8, max_nodes=1)).evaluate(state)

    projection = result.immediate_projection
    assert projection is not None
    assert {branch.result for branch in projection.player_outcomes} >= {"MISS", "HIT", "CRITICAL"}
    assert any(branch.result == "HIT+STATUS" for branch in projection.opponent_responses)
    assert any(branch.damage > 0 for branch in projection.player_outcomes)
    assert sum(branch.probability for branch in projection.player_outcomes) == 1
    assert projection.turn_order == ("Ember:player_first",)


def test_conservative_opponent_policy_uses_the_most_dangerous_known_response():
    state = SearchState(
        player_hp=10,
        player_max_hp=10,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(
            SearchMove(0, "Growl", 0, 0),
            SearchMove(1, "Tackle", 10, 10),
        ),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=2, opponent_policy=OpponentPolicy.CONSERVATIVE)).evaluate(state)

    assert result.loss_probability == 0
    assert result.win_probability == 1


def test_planner_context_requires_complete_response_facts():
    incomplete = SimpleNamespace(
        opponent_response_facts_complete=False,
        moves=(),
        opponent_moves=(),
    )
    assert search_planner_context(incomplete) is None


def test_unavailable_escape_is_not_a_search_root_action():
    state = SearchState(
        player_hp=10,
        player_max_hp=10,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        can_run=False,
        escape_probability=1.0,
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    assert all(candidate.action_kind != "run" for candidate in result.root_evaluations)
    assert "Run Away" not in result.principal_variation


def test_zero_probability_escape_is_not_a_search_root_action():
    state = SearchState(
        player_hp=10,
        player_max_hp=10,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        can_run=True,
        escape_probability=0.0,
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    assert all(candidate.action_kind != "run" for candidate in result.root_evaluations)
    assert result.action_kind == "move"


def test_failed_escape_attempts_raise_the_next_gen_iii_flee_chance():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Growl", 0, 0),),
        opponent_moves=(SearchMove(0, "Growl", 0, 0),),
        player_speed=5,
        opponent_speed=10,
        can_run=True,
        escape_attempts=0,
    )

    failed = _failed_escape_state(state)

    assert _escape_probability(state) == 64 / 255
    assert failed.escape_attempts == 1
    assert _escape_probability(failed) == 94 / 255


def test_avoidance_can_switch_to_a_faster_pokemon_before_running():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Growl", 0, 0),),
        opponent_moves=(SearchMove(0, "Growl", 0, 0),),
        player_speed=5,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(
                1,
                "fast reserve",
                20,
                20,
                (SearchMove(0, "Growl", 0, 0),),
                20,
                escape_guaranteed=True,
            ),
        ),
        battle_objective="avoid",
        can_run=True,
        escape_attempts=0,
        response_facts_complete=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=3, max_duration_seconds=None)).evaluate(state)

    assert result.action_kind == "switch"
    assert result.action_index == 1
    assert any(
        candidate.action_kind == "run"
        and candidate.win_probability > 0
        for candidate in result.root_evaluations
    )


def test_avoidance_does_not_chain_potions_after_the_first_heal():
    state = SearchState(
        player_hp=5,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Growl", 0, 0),),
        opponent_moves=(SearchMove(0, "Tackle", 4, 4),),
        player_speed=20,
        opponent_speed=10,
        player_items=(SearchItem(0, "Potion", 2, heal_amount=20),),
        battle_objective="avoid",
        response_facts_complete=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=3, max_duration_seconds=None)).evaluate(state)

    assert not any(
        result.principal_variation[index : index + 2] == ("Potion", "Potion")
        for index in range(len(result.principal_variation) - 1)
    )


def test_explicit_opponent_escape_move_is_not_a_generic_run_node():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 4, 4),),
        opponent_moves=(
            SearchMove(0, "Teleport", causes_opponent_escape=True),
            SearchMove(1, "Tackle", 1, 1),
        ),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    assert all(candidate.action_kind != "run" for candidate in result.root_evaluations)
    assert "Teleport" in result.principal_variation
    assert "OPPONENT_FLED" in result.principal_variation
    assert result.loss_probability == 0
    assert result.party_wipe_probability == 0


def test_opponent_escape_counts_as_encounter_loss_for_capture():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 4, 4),),
        opponent_moves=(SearchMove(0, "Teleport", causes_opponent_escape=True),),
        player_speed=20,
        opponent_speed=10,
        battle_objective="capture",
        capture_target=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    assert result.encounter_loss_probability == 1.0
    assert result.party_wipe_probability == 0.0


def test_planner_adapter_preserves_explicit_opponent_escape_fact():
    context = PlannerContext(
        "Mudkip",
        20,
        20,
        "Abra",
        10,
        moves=(PlannerMove(0, "Tackle", 4, 4),),
        opponent_moves=(
            PlannerOpponentMove(
                "Teleport",
                0,
                0,
                effect="TELEPORT",
                causes_opponent_escape=True,
            ),
        ),
        is_wild=True,
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=1))

    assert result is not None
    assert "OPPONENT_FLED" in result.principal_variation


def test_dominated_player_move_is_pruned_but_distinct_setup_remains():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(
            SearchMove(0, "Weak Tackle", 2, 3),
            SearchMove(1, "Strong Tackle", 4, 5),
            SearchMove(2, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    names = {candidate.action_name for candidate in result.root_evaluations}
    assert "Strong Tackle" in names
    assert "Weak Tackle" not in names
    assert "Growl" in names


def test_campaign_defeat_does_not_offer_stat_only_setup_when_damage_exists():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(
            SearchMove(0, "Tackle", 4, 5),
            SearchMove(1, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        battle_objective="defeat",
        campaign_owned=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert "Growl" not in {candidate.action_name for candidate in result.root_evaluations}


def test_secondary_status_effects_remain_branches_instead_of_being_predicted():
    state = SearchState(
        player_hp=10,
        player_max_hp=10,
        opponent_hp=24,
        opponent_max_hp=24,
        player_moves=(SearchMove(0, "Tackle", 8, 8),),
        opponent_moves=(
            SearchMove(
                0,
                "Ember",
                4,
                4,
                secondary_chance=1.0,
                secondary_effect="BURN_HIT",
            ),
        ),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=4)).evaluate(state)

    assert result.action_name == "Tackle"
    assert result.principal_variation
    assert not result.is_safe


def test_configured_safety_thresholds_are_used_by_the_result():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10, accuracy=0.5),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(
        SearchConfig(
            max_turns=2,
            minimum_safe_win_probability=0.4,
            maximum_safe_loss_probability=0.1,
            maximum_safe_catastrophic_probability=0.1,
        )
    ).evaluate(state)

    assert result.category is SearchResultCategory.HORIZON_PROJECTION
    assert not result.is_safe
    assert "depth horizon" in result.reason


def test_safe_horizon_projection_can_be_executed_when_it_makes_progress():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=20,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 5, 5),),
        opponent_moves=(SearchMove(0, "Tackle", 3, 3),),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert result.category is SearchResultCategory.HORIZON_PROJECTION
    assert result.complete is True
    assert result.projection_safe is True
    assert result.is_safe is True


def test_horizon_projection_with_no_progress_remains_uncertain():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 0, 0),),
        opponent_moves=(SearchMove(0, "Tackle", 3, 3),),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert result.category is SearchResultCategory.HORIZON_PROJECTION
    assert result.projection_safe is False
    assert result.is_safe is False


def test_node_budget_returns_an_explicitly_incomplete_result():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 1, 3), SearchMove(1, "Growl", 0, 0)),
        opponent_moves=(SearchMove(0, "Tackle", 1, 3), SearchMove(1, "Growl", 0, 0)),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=8, max_nodes=10)).evaluate(state)

    assert result.complete is False
    assert result.is_safe is False
    assert result.expanded_nodes == 10
    assert "node budget" in result.reason


def test_incomplete_defeat_search_falls_back_from_setup_to_damage(monkeypatch):
    state = SearchState(
        player_hp=12,
        player_max_hp=21,
        opponent_hp=15,
        opponent_max_hp=19,
        player_moves=(
            SearchMove(0, "Pound", 4, 5),
            SearchMove(1, "Leer", 0, 0, secondary_effect="DEFENSE_DOWN"),
        ),
        opponent_moves=(SearchMove(0, "Scratch", 3, 3),),
        player_speed=20,
        opponent_speed=10,
        battle_objective="defeat",
    )
    engine = Expectiminimax(SearchConfig(max_turns=8))

    def incomplete_value(current, move, depth):
        engine._budget_exhausted = True
        engine._budget_exhaustion = "time"
        return _Value(100.0 if move.name == "Leer" else 0.0, 0.0, 1.0, (move.name, "SEARCH_LIMIT"))

    monkeypatch.setattr(engine, "_player_move_value", incomplete_value)

    result = engine.evaluate(state)

    assert result.complete is False
    assert result.action_name == "Pound"


def test_proven_terminal_success_outranks_an_incomplete_root():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(
            SearchMove(0, "Tackle", 10, 10),
            SearchMove(1, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 5, 5),),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=4, max_nodes=2)).evaluate(state)

    assert result.action_name == "Tackle"
    assert result.category is SearchResultCategory.TERMINAL_COMPLETE
    assert result.complete is True
    growl = next(candidate for candidate in result.root_evaluations if candidate.action_name == "Growl")
    assert growl.category is SearchResultCategory.INCOMPLETE


def test_time_budget_returns_an_explicitly_incomplete_result():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 1, 3),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 3),),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=8, max_duration_seconds=0.0)).evaluate(state)

    assert result.complete is False
    assert result.is_safe is False
    assert result.budget_exhaustion == "time"
    assert "time budget" in result.reason


def test_root_evaluations_include_switches_even_when_attack_is_selected():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "Torchic", 20, 20, (SearchMove(0, "Scratch", 4, 4),), 10),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=3)).evaluate(state)

    assert any(candidate.action_kind == "switch" for candidate in result.root_evaluations)


def test_switch_root_evaluation_includes_immediate_replacement_response():
    state = SearchState(
        player_hp=8,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 4, 4),),
        opponent_moves=(SearchMove(0, "Tackle", 3, 5),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "Torchic", 20, 20, (SearchMove(0, "Scratch", 4, 4),), 10),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)
    switch = next(candidate for candidate in result.root_evaluations if candidate.action_kind == "switch")

    assert switch.immediate_projection is not None
    assert [branch.result for branch in switch.immediate_projection.player_outcomes] == ["SWITCH"]
    assert switch.immediate_projection.opponent_responses
    assert all(
        branch.phase == "opponent_after_switch"
        for branch in switch.immediate_projection.opponent_responses
    )
    assert switch.immediate_projection.turn_order == ("opponent_after_switch:Tackle",)


def test_generic_hit_effect_never_creates_a_fake_status_branch():
    outcomes = _move_outcomes(
        SearchMove(
            0,
            "Scratch",
            4,
            4,
            secondary_chance=1.0,
            secondary_effect="HIT",
        ),
        attacker_is_player=True,
    )

    assert all(outcome.secondary_effect is None for outcome in outcomes)
    assert all("STATUS" not in outcome.label for outcome in outcomes)


def test_root_risk_keeps_active_faint_separate_from_party_wipe():
    state = SearchState(
        player_hp=7,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 10, 10),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "Torchic", 20, 20, (SearchMove(0, "Tackle", 10, 10),), 20),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    attack = next(candidate for candidate in result.root_evaluations if candidate.action_name == "Tackle")
    assert attack.active_faint_probability == 1.0
    assert attack.party_wipe_probability == 0.0
    assert attack.run_loss_probability == 0.0


def test_zero_damage_stat_moves_change_future_search_state():
    state = SearchState(
        player_hp=10,
        player_max_hp=10,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 4, 4),),
        opponent_moves=(SearchMove(0, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),),
        player_speed=10,
        opponent_speed=20,
    )
    growl = state.opponent_moves[0]
    outcome = _move_outcomes(growl, attacker_is_player=False)[0]

    changed = _apply_move_effect(
        state,
        growl,
        outcome,
        attacker_is_player=False,
        target_is_player=True,
    )

    assert dict(changed.player_stat_stages)["attack"] == -1
    assert outcome.damage == 0


def test_string_shot_changes_future_turn_order():
    state = SearchState(
        player_hp=10,
        player_max_hp=10,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "String Shot", 0, 0, secondary_effect="SPEED_DOWN"),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=10,
        opponent_speed=11,
    )
    string_shot = state.player_moves[0]
    changed = _apply_move_effect(
        state,
        string_shot,
        _move_outcomes(string_shot, attacker_is_player=True)[0],
        attacker_is_player=True,
        target_is_player=False,
    )

    assert dict(changed.opponent_stat_stages)["speed"] == -1
    assert _acts_first(True, string_shot, changed)


def test_transposition_cache_completes_repeated_damage_roll_states():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=13,
        opponent_max_hp=13,
        player_moves=(
            SearchMove(0, "Scratch", 5, 7),
            SearchMove(1, "Growl", 0, 0),
        ),
        opponent_moves=(
            SearchMove(0, "Growl", 0, 0),
            SearchMove(1, "Tackle", 3, 4),
        ),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(
        SearchConfig(
            max_turns=8,
            max_nodes=2_000,
            opponent_policy=OpponentPolicy.RULE_BASED,
        )
    ).evaluate(state)

    assert result.complete is True
    assert result.expanded_nodes < 2_000


def test_expectiminimax_can_choose_a_switch_as_the_first_action():
    state = SearchState(
        player_hp=5,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 10, 10),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(
                1,
                "Zigzagoon",
                20,
                20,
                (SearchMove(0, "Tackle", 10, 10),),
                20,
            ),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=4)).evaluate(state)

    assert result.action_kind == "switch"
    assert result.action_index == 1
    assert result.action_name == "Zigzagoon"
    assert result.principal_variation[0] == "Switch(Zigzagoon)"
    assert result.win_probability >= 0.9


def test_speed_ties_are_exposed_as_random_ordering():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 20, 20),),
        player_speed=10,
        opponent_speed=10,
    )

    assert _acts_first(True, state.player_moves[0], state, state.opponent_moves[0], tie_probability=True) is None
    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)
    assert result.win_probability == 0.5
    assert result.active_faint_probability == 0.5


def test_mixed_opponent_priorities_are_evaluated_per_response():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10, priority=1),),
        opponent_moves=(
            SearchMove(0, "Quick Attack", 20, 20, priority=2),
            SearchMove(1, "Growl", 0, 0, priority=0),
        ),
        player_speed=100,
        opponent_speed=1,
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)
    assert result.win_probability == 0.0
    assert result.active_faint_probability == 1.0
    assert result.party_wipe_probability == 1.0


def test_opponent_status_skip_still_allows_the_selected_player_action():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 20, 20),),
        player_speed=10,
        opponent_speed=20,
        opponent_status="SLEEP",
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert result.action_name == "Tackle"
    assert round(result.win_probability, 2) == 0.66
    assert round(result.active_faint_probability, 2) == 0.34
    assert "STATUS_SKIP" in result.principal_variation


def test_expectiminimax_can_choose_a_healing_item_as_the_first_action():
    state = SearchState(
        player_hp=5,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 5, 5),),
        player_speed=10,
        opponent_speed=20,
        player_items=(SearchItem(0, "Potion", 1, heal_amount=15),),
    )

    result = Expectiminimax(SearchConfig(max_turns=4)).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_index == 0
    assert result.action_name == "Potion"
    assert result.principal_variation[0] == "Potion"
    assert result.win_probability >= 0.9


def test_healthy_status_does_not_make_a_targeted_cure_a_capture_root():
    state = SearchState(
        player_hp=15,
        player_max_hp=15,
        opponent_hp=15,
        opponent_max_hp=15,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_status="none",
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "reserve", 20, 20, (SearchMove(0, "Tackle", 1, 1),), 10, "none"),
        ),
        capture_target=True,
        battle_objective="capture",
        player_items=(
            SearchItem(0, "Antidote", 4, cures_status=True, status_healed=frozenset({"poison"})),
            SearchItem(1, "Poké Ball", 9, is_capture_ball=True, capture_probability=0.5),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert not any(
        candidate.action_kind == "item" and candidate.action_index == 0
        for candidate in result.root_evaluations
    )
    assert result.action_name == "Poké Ball"


def test_planner_context_preserves_healthy_status_when_building_search_items():
    context = PlannerContext(
        "active",
        15,
        15,
        "opponent",
        15,
        moves=(PlannerMove(0, "Tackle", 1, 1),),
        switches=(
            PlannerSwitch(1, "reserve", 20, 20, 1, status="none", moves=(PlannerMove(0, "Tackle", 1, 1),)),
        ),
        items=(
            RecoveryItem(
                SimpleNamespace(name="Antidote"),
                4,
                status_healed=frozenset({"poison"}),
            ),
            RecoveryItem(SimpleNamespace(name="Poké Ball"), 9, is_capture_ball=True),
        ),
        opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
        active_status="none",
        is_wild=True,
        can_run=False,
        capture_target=True,
        capture_available=True,
        battle_objective="capture",
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=1))

    assert not any(
        candidate.action_kind == "item" and candidate.action_index == 0
        for candidate in result.root_evaluations
    )


def test_targeted_cure_is_not_a_root_for_a_different_active_status():
    state = SearchState(
        player_hp=15,
        player_max_hp=15,
        opponent_hp=15,
        opponent_max_hp=15,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_status="burned",
        player_items=(
            SearchItem(0, "Antidote", 1, cures_status=True, status_healed=frozenset({"poison"})),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert all(
        not (candidate.action_kind == "item" and candidate.action_index == 0)
        for candidate in result.root_evaluations
    )


def test_targeted_cure_can_target_a_statused_full_hp_reserve():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=15,
        opponent_max_hp=15,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "reserve", 20, 20, (SearchMove(0, "Tackle", 1, 1),), 10, "poisoned"),
        ),
        player_items=(
            SearchItem(0, "Antidote", 1, cures_status=True, status_healed=frozenset({"poison"})),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    antidote = next(
        candidate
        for candidate in result.root_evaluations
        if candidate.action_kind == "item" and candidate.action_index == 0
    )
    assert antidote.action_kind == "item"
    assert antidote.target_party_index == 1


def test_item_resource_cost_prefers_a_guaranteed_attack_when_healing_is_unnecessary():
    state = SearchState(
        player_hp=19,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        player_items=(SearchItem(0, "Potion", 1, heal_amount=20),),
    )

    result = Expectiminimax(SearchConfig(max_turns=3)).evaluate(state)

    assert result.action_kind == "move"
    assert result.action_name == "Tackle"
    potion = next(candidate for candidate in result.root_evaluations if candidate.action_name == "Potion")
    attack = next(candidate for candidate in result.root_evaluations if candidate.action_name == "Tackle")
    assert attack.expected_score > potion.expected_score


def test_item_waste_cost_prefers_the_smallest_item_that_survives():
    state = SearchState(
        player_hp=10,
        player_max_hp=30,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 20, 20),),
        player_speed=10,
        opponent_speed=20,
        player_items=(
            SearchItem(0, "Potion", 1, heal_amount=20),
            SearchItem(1, "Super Potion", 1, heal_amount=50),
        ),
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Potion"
    assert result.expected_score > next(
        candidate.expected_score
        for candidate in result.root_evaluations
        if candidate.action_name == "Super Potion"
    )


def test_capture_objective_prefers_a_ball_to_ko_when_the_target_is_legal():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 1, is_capture_ball=True, capture_probability=0.6),),
    )

    result = Expectiminimax(SearchConfig(max_turns=3)).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Poké Ball"
    assert result.capture_probability == 0.6
    assert result.encounter_loss_probability == 0.4
    assert result.is_safe is False
    ko_root = next(root for root in result.root_evaluations if root.action_kind == "move")
    assert ko_root.objective_failed is True
    assert result.objective_failed is False


def test_capture_objective_rejects_a_guaranteed_ko_when_no_ball_remains():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
    )

    result = Expectiminimax(SearchConfig(max_turns=2)).evaluate(state)

    assert result.action_kind == "move"
    assert result.objective_failed is True
    assert result.is_safe is False
    assert result.is_autonomous_root is False


def test_planner_adapter_preserves_special_damage_stat_after_attack_drop():
    planner_context = PlannerContext(
        "Treecko",
        20,
        20,
        "Wingull",
        5,
        moves=(PlannerMove(0, "Absorb", 5, 5, damage_stat="special_attack"),),
        opponent_response_facts_complete=True,
        active_stat_stages=(("attack", -1),),
    )

    result = search_planner_context(
        planner_context,
        config=SearchConfig(max_turns=1, max_duration_seconds=None),
    )

    assert result is not None
    assert result.immediate_projection is not None
    assert result.immediate_projection.player_outcomes[0].damage == 5


def test_capture_ready_ball_dominates_a_projected_switch_line():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=5,
        opponent_max_hp=15,
        player_moves=(SearchMove(0, "Growl", 0, 0),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(
                1,
                "Wurmple",
                20,
                20,
                (SearchMove(0, "Tackle", 1, 1),),
                10,
            ),
        ),
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 3, is_capture_ball=True, capture_probability=0.5),),
        campaign_owned=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=5)).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Poké Ball"


def test_capture_ready_ball_dominates_unnecessary_attack_drop():
    """Do not spend a response turn on Growl when the target is already ready."""

    state = SearchState(
        player_hp=18,
        player_max_hp=20,
        opponent_hp=1,
        opponent_max_hp=15,
        player_moves=(
            SearchMove(0, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 2, 2),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 3, is_capture_ball=True, capture_probability=0.5),),
    )

    result = Expectiminimax(SearchConfig(max_turns=5)).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Poké Ball"


def test_capture_turn_cost_prefers_ball_when_setup_does_not_change_capture_odds():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 2, is_capture_ball=True, capture_probability=0.4),),
    )

    result = Expectiminimax(
        SearchConfig(max_turns=4, max_duration_seconds=None, capture_turn_penalty=25.0)
    ).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Poké Ball"
    growl = next(candidate for candidate in result.root_evaluations if candidate.action_name == "Growl")
    ball = next(candidate for candidate in result.root_evaluations if candidate.action_name == "Poké Ball")
    assert growl.expected_score < ball.expected_score


def test_capture_utility_prefers_viable_ball_over_terminal_run_failure_when_incomplete():
    """A bounded root still compares a viable capture line with Run Away utility."""

    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 3, is_capture_ball=True, capture_probability=0.5),),
        can_run=True,
        escape_probability=0.647,
        response_facts_complete=True,
    )

    result = Expectiminimax(
        SearchConfig(max_turns=8, max_nodes=10, max_duration_seconds=None)
    ).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Poké Ball"
    assert result.category is SearchResultCategory.INCOMPLETE
    run = next(candidate for candidate in result.root_evaluations if candidate.action_kind == "run")
    ball = next(candidate for candidate in result.root_evaluations if candidate.action_name == "Poké Ball")
    assert run.expected_score < ball.expected_score


def test_capture_utility_can_choose_run_when_continuing_is_run_loss_risk():
    state = SearchState(
        player_hp=1,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),),
        opponent_moves=(SearchMove(0, "Tackle", 5, 5),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 3, is_capture_ball=True, capture_probability=0.05),),
        can_run=True,
        escape_probability=0.647,
        response_facts_complete=True,
    )

    result = Expectiminimax(
        SearchConfig(max_turns=8, max_nodes=10, max_duration_seconds=None)
    ).evaluate(state)

    assert result.action_kind == "run"
    assert result.action_name == "Run Away"
    assert result.party_wipe_probability == 0.353


def test_capture_search_blocks_immediate_switch_reversal():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_party_index=1,
        player_switches=(
            SearchSwitch(0, "Treecko", 20, 20, (SearchMove(0, "Tackle", 1, 1),), 10),
            SearchSwitch(2, "Lotad", 20, 20, (SearchMove(0, "Tackle", 1, 1),), 10),
        ),
        switch_history=(0, 1),
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 1, is_capture_ball=True, capture_probability=0.2),),
        campaign_owned=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert not any(
        candidate.action_kind == "switch" and candidate.action_index == 0
        for candidate in result.root_evaluations
    )


def test_non_campaign_search_preserves_switch_candidates_for_legacy_callers():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_party_index=1,
        player_switches=(
            SearchSwitch(0, "Treecko", 20, 20, (SearchMove(0, "Tackle", 1, 1),), 10),
            SearchSwitch(2, "Lotad", 20, 20, (SearchMove(0, "Tackle", 1, 1),), 10),
        ),
        switch_history=(0, 1),
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 1, is_capture_ball=True, capture_probability=0.2),),
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert any(
        candidate.action_kind == "switch" and candidate.action_index == 0
        for candidate in result.root_evaluations
    )


def test_experience_utility_can_prefer_a_safe_participant_with_more_value():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(
                1,
                "Treecko",
                20,
                20,
                (SearchMove(0, "Tackle", 10, 10),),
                10,
            ),
        ),
        experience_reward=20,
        experience_values=((0, 1.0), (1, 10.0)),
    )

    result = Expectiminimax(SearchConfig(max_turns=3, experience_weight=1.0)).evaluate(state)

    assert result.action_kind == "switch"
    assert result.action_index == 1
    assert result.expected_experience == 20
    assert result.experience_utility == 11.0


def test_campaign_ordinary_switch_cost_is_paid_but_free_replacement_is_not():
    common = dict(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(
                1,
                "Treecko",
                20,
                20,
                (SearchMove(0, "Tackle", 10, 10),),
                10,
            ),
        ),
        experience_reward=20,
        experience_values=((0, 1.0), (1, 10.0)),
        campaign_owned=True,
    )
    config = SearchConfig(
        max_turns=3,
        experience_weight=1.0,
        ordinary_switch_penalty=35.0,
    )

    ordinary_state = SearchState(**common)
    ordinary = Expectiminimax(config).evaluate(ordinary_state)
    free = Expectiminimax(config).evaluate(SearchState(**common, free_switch=True))
    uncharged = Expectiminimax(
        SearchConfig(
            max_turns=3,
            experience_weight=1.0,
            ordinary_switch_penalty=0.0,
        )
    ).evaluate(ordinary_state)

    assert ordinary.action_kind == "move"
    assert ordinary.action_name == "Tackle"
    assert free.action_kind == "switch"
    assert free.action_index == 1
    ordinary_switch = next(
        candidate for candidate in ordinary.root_evaluations if candidate.action_kind == "switch"
    )
    uncharged_switch = next(
        candidate for candidate in uncharged.root_evaluations if candidate.action_kind == "switch"
    )
    assert uncharged_switch.expected_score - ordinary_switch.expected_score == 35.0


def test_bounded_switch_projection_prefers_the_best_followup_matchup():
    """A free replacement is ranked by the move it enables after exposure."""

    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=22,
        opponent_max_hp=22,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(
            SearchMove(
                0,
                "Growth",
                0,
                0,
                stat_target="special_attack",
                stat_delta=1,
            ),
        ),
        player_speed=1,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(
                1,
                "Freddie",
                25,
                25,
                (SearchMove(0, "Scratch", 5, 5),),
                10,
            ),
            SearchSwitch(
                2,
                "Robert",
                16,
                16,
                (SearchMove(0, "Astonish", 2, 2),),
                10,
            ),
        ),
        campaign_owned=True,
        free_switch=True,
        battle_objective="defeat",
        response_facts_complete=True,
    )

    result = Expectiminimax(
        SearchConfig(max_turns=2, max_duration_seconds=None, max_nodes=100_000)
    ).evaluate(state)

    assert result.action_kind == "switch"
    assert result.action_index == 1
    freddie = next(root for root in result.root_evaluations if root.action_index == 1)
    robert = next(root for root in result.root_evaluations if root.action_index == 2)
    assert freddie.expected_score > robert.expected_score
    assert freddie.principal_variation[0] == "Switch(Freddie)"
    assert "Scratch" in freddie.principal_variation


def test_switch_cycle_roots_remain_legal_but_no_progress_is_heavily_devalued():
    stat_move = SearchMove(
        0,
        "Growl",
        0,
        0,
        stat_target="attack",
        stat_delta=-1,
    )
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(stat_move,),
        opponent_moves=(
            SearchMove(
                0,
                "Growth",
                0,
                0,
                stat_target="special_attack",
                stat_delta=1,
            ),
        ),
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "B", 20, 20, (stat_move,), 10),
            SearchSwitch(2, "C", 20, 20, (stat_move,), 10),
        ),
        campaign_owned=True,
        battle_objective="defeat",
        response_facts_complete=True,
    )

    result = Expectiminimax(
        SearchConfig(max_turns=4, max_duration_seconds=None, max_nodes=100_000)
    ).evaluate(state)
    switch_roots = tuple(root for root in result.root_evaluations if root.action_kind == "switch")

    assert len(switch_roots) == 2
    assert all(root.expected_score < 0 for root in switch_roots)
    assert result.action_kind == "move"


def test_capture_probability_callback_values_post_weakening_state():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Scratch", 5, 5),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 1, is_capture_ball=True),),
    )

    result = Expectiminimax(
        SearchConfig(
            max_turns=4,
            capture_probability=lambda current, _item: 0.8 if current.opponent_hp <= 5 else 0.1,
        )
    ).evaluate(state)

    assert result.action_name == "Scratch"
    assert result.capture_probability == 0.8


def test_capture_cutoff_compares_safe_weakening_with_remaining_ball_value():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Scratch", 5, 5),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 4, is_capture_ball=True),),
    )

    result = Expectiminimax(
        SearchConfig(
            max_turns=4,
            max_duration_seconds=0.0,
            capture_probability=lambda current, _item: 0.8 if current.opponent_hp <= 5 else 0.1,
        )
    ).evaluate(state)

    assert result.action_kind == "move"
    assert result.action_name == "Scratch"
    assert round(result.capture_probability, 4) == 0.9984


def test_capture_prices_ko_risk_against_the_remaining_ball_sequence():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(
            SearchMove(
                0,
                "Scratch",
                5,
                5,
                critical_damage_max=10,
                critical_hit_chance=0.0625,
            ),
        ),
        opponent_moves=(),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        player_items=(SearchItem(0, "Poké Ball", 1, is_capture_ball=True),),
    )

    result = Expectiminimax(
        SearchConfig(
            max_turns=8,
            capture_probability=lambda current, _item: 0.45 if current.opponent_hp <= 5 else 0.337,
        )
    ).evaluate(state)

    assert result.action_kind == "move"
    assert result.action_name == "Scratch"


def test_final_failed_ball_transitions_to_defeat_and_can_preserve_exp_value():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=5,
        opponent_max_hp=5,
        player_moves=(SearchMove(0, "Scratch", 5, 5),),
        opponent_moves=(),
        player_speed=20,
        opponent_speed=10,
        capture_target=True,
        battle_objective="capture",
        experience_reward=25,
        player_items=(SearchItem(0, "Poké Ball", 1, is_capture_ball=True, capture_probability=0.0),),
    )

    result = Expectiminimax(SearchConfig(max_turns=4, experience_weight=1.0)).evaluate(state)

    assert result.action_kind == "item"
    assert result.action_name == "Poké Ball"
    assert result.expected_experience == 25
    assert "ENCOUNTER_LOST" in result.principal_variation


def test_expectiminimax_can_flee_when_a_zero_progress_line_is_worse():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=20,
        opponent_max_hp=20,
        player_moves=(SearchMove(0, "Growl", 0, 0),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        can_run=True,
        escape_probability=1.0,
    )

    result = Expectiminimax(SearchConfig(max_turns=3)).evaluate(state)

    assert result.action_kind == "run"
    assert result.action_name == "Run Away"


def test_avoid_intent_prefers_immediate_escape_over_horizon_damage_progress():
    state = SearchState(
        player_hp=14,
        player_max_hp=20,
        opponent_hp=3,
        opponent_max_hp=15,
        player_moves=(
            SearchMove(0, "Tackle", 4, 8),
            SearchMove(1, "Growl", 0, 0, stat_target="opponent_attack", stat_delta=-1),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        can_run=True,
        escape_probability=1.0,
        battle_objective="avoid",
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert result.action_kind == "run"
    assert result.action_name == "Run Away"
    candidates = {candidate.action_name: candidate.expected_score for candidate in result.root_evaluations}
    assert candidates["Run Away"] > candidates["Growl"]


def test_avoid_intent_prefers_guaranteed_escape_over_nonprogress_damage():
    state = SearchState(
        player_hp=16,
        player_max_hp=16,
        opponent_hp=16,
        opponent_max_hp=16,
        player_moves=(
            SearchMove(0, "Tackle", 3, 4),
            SearchMove(1, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),
        ),
        opponent_moves=(
            SearchMove(0, "Astonish", 0, 0),
            SearchMove(1, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),
        ),
        player_speed=20,
        opponent_speed=10,
        can_run=True,
        escape_probability=1.0,
        battle_objective="avoid",
    )

    result = Expectiminimax(SearchConfig(max_turns=8)).evaluate(state)

    assert result.action_kind == "run"
    assert result.action_name == "Run Away"


def test_avoid_intent_prioritizes_survivable_run_before_terminal_attack_or_switch():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(
            SearchMove(0, "Tackle", 10, 10),
            SearchMove(1, "Growl", 0, 0),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(
                1,
                "reserve",
                20,
                20,
                (SearchMove(0, "Tackle", 10, 10),),
                20,
            ),
        ),
        can_run=True,
        escape_probability=0.50,
        battle_objective="avoid",
    )

    result = Expectiminimax(SearchConfig(max_turns=3, max_duration_seconds=None)).evaluate(state)

    assert result.action_kind == "run"
    assert result.action_name == "Run Away"
    assert {candidate.action_name for candidate in result.root_evaluations} >= {
        "Tackle",
        "reserve",
        "Run Away",
    }


def test_avoid_intent_keeps_run_priority_when_search_is_cut_off_at_root():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        can_run=True,
        escape_probability=1.0,
        battle_objective="avoid",
    )

    result = Expectiminimax(SearchConfig(max_turns=8, max_duration_seconds=0.0)).evaluate(state)

    assert result.action_kind == "run"
    assert result.action_name == "Run Away"


def test_avoid_intent_does_not_prioritize_run_when_failed_escape_can_faint_active():
    state = SearchState(
        player_hp=1,
        player_max_hp=20,
        opponent_hp=3,
        opponent_max_hp=30,
        player_moves=(
            SearchMove(0, "Tackle", 10, 10),
            SearchMove(1, "Growl", 0, 0),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 5, 5),),
        player_speed=20,
        opponent_speed=10,
        can_run=True,
        escape_probability=0.50,
        battle_objective="avoid",
    )

    result = Expectiminimax(SearchConfig(max_turns=3, max_duration_seconds=None)).evaluate(state)

    assert result.action_kind == "move"
    assert result.action_name == "Tackle"
    run = next(candidate for candidate in result.root_evaluations if candidate.action_kind == "run")
    assert run.active_faint_probability == 0.50


def test_avoid_intent_does_not_prove_run_safe_from_incomplete_response_facts():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=3,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 10, 10),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        can_run=True,
        escape_probability=0.50,
        battle_objective="avoid",
        response_facts_complete=False,
    )

    result = Expectiminimax(SearchConfig(max_turns=3, max_duration_seconds=None)).evaluate(state)

    run = next(candidate for candidate in result.root_evaluations if candidate.action_kind == "run")
    assert run.category is SearchResultCategory.TERMINAL_COMPLETE
    assert result.action_name == "Run Away"
    assert result.category is SearchResultCategory.HORIZON_PROJECTION
    assert result.is_safe is False


def test_stat_move_at_stage_floor_is_not_a_legal_root():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(
            SearchMove(0, "Tackle", 1, 1),
            SearchMove(1, "Growl", 0, 0, secondary_effect="ATTACK_DOWN"),
        ),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        opponent_stat_stages=(("attack", -6),),
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert {candidate.action_name for candidate in result.root_evaluations} == {"Tackle"}


def test_known_pp_is_consumed_by_simulated_player_turns():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=10,
        player_moves=(SearchMove(0, "Tackle", 1, 1, pp=1),),
        opponent_moves=(),
        player_speed=20,
        opponent_speed=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=3)).evaluate(state)

    assert result.principal_variation.count("Tackle") == 1


def test_ordinary_battle_search_ignores_capture_balls():
    context = PlannerContext(
        "Torchic",
        20,
        20,
        "Wurmple",
        10,
        moves=(PlannerMove(0, "Scratch", 4, 4),),
        opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
        is_wild=True,
        items=(RecoveryItem(SimpleNamespace(name="Poké Ball"), 5, 0, True),),
        battle_objective="defeat",
        capture_target=False,
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context)

    assert result is not None
    assert not any(candidate.action_kind == "item" for candidate in result.root_evaluations)


def test_campaign_incomplete_defeat_projection_prefers_safe_damage_over_switch():
    state = SearchState(
        player_hp=10,
        player_max_hp=20,
        opponent_hp=20,
        opponent_max_hp=20,
        player_moves=(SearchMove(0, "Tackle", 5, 5),),
        opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "reserve", 20, 20, (SearchMove(0, "Growl", 0, 0),), 10),
        ),
        battle_objective="defeat",
        campaign_owned=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=8, max_nodes=1)).evaluate(state)

    assert result.action_kind == "move"
    assert result.action_name == "Tackle"
    assert result.category is SearchResultCategory.INCOMPLETE
    assert result.projection_safe is True


def test_planner_result_carries_a_reserve_potion_target():
    context = PlannerContext(
        "active",
        20,
        20,
        "opponent",
        30,
        switches=(PlannerSwitch(1, "reserve", 5, 20, 1, moves=()),),
        items=(RecoveryItem(SimpleNamespace(name="Potion"), 1, heal_amount=20),),
        opponent_moves=(PlannerOpponentMove("Tackle", 1, 1),),
        is_trainer=True,
        battle_objective="defeat",
        campaign_owned=True,
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=1))

    assert result is not None
    assert result.action_kind == "item"
    assert result.target_party_index == 1
    reserve = next(root for root in result.root_evaluations if root.action_kind == "item")
    assert reserve.target_party_index == 1


def test_bounded_capture_search_preserves_selected_reserve_potion_target():
    context = PlannerContext(
        "Wurmple",
        18,
        18,
        "Silcoon",
        21,
        moves=(),
        switches=(PlannerSwitch(1, "reserve", 5, 20, None, moves=()),),
        items=(
            RecoveryItem(SimpleNamespace(name="Potion"), 6, heal_amount=20),
            RecoveryItem(SimpleNamespace(name="Poké Ball"), 0, is_capture_ball=True),
        ),
        opponent_moves=(),
        is_wild=True,
        can_switch=True,
        can_run=False,
        capture_target=True,
        capture_available=True,
        battle_objective="capture",
        campaign_owned=True,
        opponent_response_facts_complete=False,
    )

    result = search_planner_context(
        context,
        config=SearchConfig(max_turns=8, max_nodes=1, max_duration_seconds=0.01),
    )

    assert result is not None
    assert result.action_kind == "item"
    assert result.action_index == 0
    assert result.target_party_index == 1
    selected_root = next(
        root
        for root in result.root_evaluations
        if root.action_kind == "item" and root.action_index == 0
    )
    assert selected_root.target_party_index == 1


def test_campaign_attack_beats_unnecessary_active_potion():
    context = PlannerContext(
        "active",
        18,
        22,
        "opponent",
        5,
        moves=(PlannerMove(0, "Tackle", 5, 5),),
        items=(RecoveryItem(SimpleNamespace(name="Potion"), 1, heal_amount=5),),
        opponent_moves=(PlannerOpponentMove("Tackle", 3, 3),),
        is_trainer=True,
        battle_objective="defeat",
        campaign_owned=True,
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=2))

    assert result is not None
    assert result.action_kind == "move"
    assert result.action_name == "Tackle"


def test_planner_adapter_preserves_campaign_switch_history_across_turns():
    context = PlannerContext(
        "Treecko",
        20,
        20,
        "Lotad",
        5,
        moves=(PlannerMove(0, "Tackle", 1, 1),),
        opponent_moves=(PlannerOpponentMove("Astonish", 1, 1),),
        switches=(
            # The active party member is slot 1. Slot 0 is the member used on
            # the preceding turn and must not be selected as an immediate
            # reversal by the next campaign planner call.
            PlannerSwitch(0, "Wurmple", 14, 14, 4, 1),
            PlannerSwitch(2, "Mudkip", 20, 20, 2, 4),
        ),
        active_party_index=1,
        capture_target=True,
        capture_available=True,
        items=(RecoveryItem(SimpleNamespace(name="Poké Ball"), 2, 0, True),),
        battle_objective="capture",
        campaign_owned=True,
        switch_history=(0, 1),
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=2))

    assert result is not None
    assert not any(
        candidate.action_kind == "switch" and candidate.action_index == 0
        for candidate in result.root_evaluations
    )


def test_campaign_search_allows_emergency_reversal_when_active_is_inside_response_ceiling():
    context = PlannerContext(
        "Wurmple",
        2,
        18,
        "Taillow",
        10,
        moves=(PlannerMove(0, "Tackle", 4, 5),),
        opponent_moves=(PlannerOpponentMove("Peck", 10, 12, critical_damage_max=24),),
        switches=(
            PlannerSwitch(
                0,
                "Thomas",
                15,
                27,
                12,
                4,
                incoming_critical_damage_max=24,
                moves=(PlannerMove(0, "Mud-Slap", 3, 4),),
            ),
        ),
        active_party_index=1,
        campaign_owned=True,
        switch_history=(0, 1),
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=1))

    assert result is not None
    assert any(candidate.action_kind == "switch" and candidate.action_index == 0 for candidate in result.root_evaluations)
    assert result.action_kind == "switch"
    assert result.action_index == 0


def test_campaign_defeat_root_remains_autonomous_when_the_run_is_doomed():
    state = SearchState(
        player_hp=1,
        player_max_hp=18,
        opponent_hp=30,
        opponent_max_hp=30,
        player_moves=(SearchMove(0, "Tackle", 1, 1),),
        opponent_moves=(SearchMove(0, "Peck", 10, 10),),
        player_party_index=0,
        campaign_owned=True,
        battle_objective="defeat",
    )

    result = Expectiminimax(SearchConfig(max_turns=1)).evaluate(state)

    assert result.action_kind == "move"
    assert result.party_wipe_probability == 1.0
    assert result.is_autonomous_root


def test_previous_opponent_damage_does_not_make_a_fresh_switch_projection_safe():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=10,
        opponent_max_hp=20,
        player_moves=(SearchMove(0, "Growl", 0, 0, stat_target="attack", stat_delta=-1),),
        opponent_moves=(SearchMove(0, "Growl", 0, 0, stat_target="special_attack", stat_delta=1),),
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "reserve", 20, 20, (SearchMove(0, "Growl", 0, 0),), 10),
        ),
        campaign_owned=True,
        root_opponent_hp=10,
    )

    result = Expectiminimax(SearchConfig(max_turns=1, max_duration_seconds=None)).evaluate(state)

    switch = next(root for root in result.root_evaluations if root.action_kind == "switch")
    assert switch.objective_progress == 0.0
    assert switch.projection_safe is False


def test_unsupported_opponent_mechanics_are_an_incomplete_response_threat():
    context = PlannerContext(
        "active",
        20,
        20,
        "Seedot",
        20,
        moves=(PlannerMove(0, "Scratch", 4, 4),),
        items=(RecoveryItem(SimpleNamespace(name="Potion"), 1, 20),),
        opponent_moves=(
            PlannerOpponentMove(
                "Bide",
                0,
                0,
                mechanics_supported=False,
                response_facts_complete=True,
            ),
        ),
        is_trainer=True,
        campaign_owned=True,
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=1, max_duration_seconds=None))

    assert result is not None
    assert result.projection_safe is False
    assert result.category is SearchResultCategory.HORIZON_PROJECTION
    assert all(root.projection_safe is False for root in result.root_evaluations)


def test_persistent_switch_cycle_pressure_devalues_repeated_slot_without_removing_it():
    common = dict(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=20,
        opponent_max_hp=20,
        player_moves=(SearchMove(0, "Growl", 0, 0, stat_target="attack", stat_delta=-1),),
        opponent_moves=(SearchMove(0, "Growth", 0, 0, stat_target="special_attack", stat_delta=1),),
        player_party_index=2,
        player_switches=(
            SearchSwitch(0, "A", 20, 20, (SearchMove(0, "Growl", 0, 0),), 10),
        ),
        campaign_owned=True,
        switch_history=(0, 1, 1, 2),
    )
    pressured = Expectiminimax(
        SearchConfig(max_turns=1, max_duration_seconds=None)
    ).evaluate(SearchState(**common, switch_streak=2, switch_cycle_pressure=1))
    unpressured = Expectiminimax(
        SearchConfig(max_turns=1, max_duration_seconds=None)
    ).evaluate(SearchState(**common))

    pressured_root = next(
        root for root in pressured.root_evaluations if root.action_kind == "switch" and root.action_index == 0
    )
    unpressured_root = next(
        root for root in unpressured.root_evaluations if root.action_kind == "switch" and root.action_index == 0
    )
    assert pressured_root.expected_score < unpressured_root.expected_score
    assert any(root.action_kind == "switch" and root.action_index == 0 for root in pressured.root_evaluations)


def test_free_switch_compares_staying_and_attacking_against_weak_replacement():
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=8,
        opponent_max_hp=8,
        player_moves=(SearchMove(0, "Tackle", 5, 5),),
        opponent_moves=(SearchMove(0, "Growth", 0, 0, stat_target="special_attack", stat_delta=1),),
        player_speed=20,
        opponent_speed=10,
        player_party_index=0,
        player_switches=(
            SearchSwitch(1, "weak reserve", 20, 20, (SearchMove(0, "Scratch", 1, 1),), 10),
        ),
        campaign_owned=True,
        free_switch=True,
        response_facts_complete=True,
    )

    result = Expectiminimax(SearchConfig(max_turns=2, max_duration_seconds=None)).evaluate(state)

    assert result.action_kind == "move"
    assert result.action_name == "Tackle"
    assert any(root.action_kind == "switch" for root in result.root_evaluations)
    assert any(root.action_kind == "move" and root.objective_progress > 0 for root in result.root_evaluations)


def test_campaign_capture_search_does_not_switch_healthy_lead_without_progress_reason():
    context = PlannerContext(
        "Treecko",
        20,
        20,
        "Lotad",
        10,
        moves=(PlannerMove(0, "Tackle", 1, 1),),
        opponent_moves=(PlannerOpponentMove("Astonish", 1, 1),),
        switches=(PlannerSwitch(1, "Wurmple", 14, 14, 2, 1),),
        active_party_index=0,
        capture_target=True,
        capture_available=False,
        battle_objective="capture",
        campaign_owned=True,
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=2))

    assert result is not None
    assert result.action_kind == "move"
    assert not any(candidate.action_kind == "switch" for candidate in result.root_evaluations)


def test_bide_stores_actual_hp_damage_then_releases_double_without_stat_scaling():
    bide = SearchMove(0, "Bide", 1, 1, secondary_effect="BIDE")
    tackle = SearchMove(0, "Tackle", 3, 3)
    state = SearchState(
        player_hp=20,
        player_max_hp=20,
        opponent_hp=20,
        opponent_max_hp=20,
        player_moves=(bide,),
        opponent_moves=(tackle,),
        player_stat_stages=(("attack", 6),),
        opponent_stat_stages=(("defense", -6),),
    )

    start = _move_outcomes_for_state(state, bide, attacker_is_player=True)[0]
    state = _apply_move_effect(
        _apply_move_damage(state, bide, start, attacker_is_player=True),
        bide,
        start,
        attacker_is_player=True,
        target_is_player=False,
    )
    assert start.label == "BIDE_CHARGE"
    assert start.damage == 0
    assert state.player_bide_turns == 2

    hit = _move_outcomes_for_state(state, tackle, attacker_is_player=False)[0]
    state = _apply_move_effect(
        _apply_move_damage(state, tackle, hit, attacker_is_player=False),
        tackle,
        hit,
        attacker_is_player=False,
        target_is_player=True,
    )
    assert state.player_hp == 17
    assert state.player_bide_damage == 3

    storing = _move_outcomes_for_state(state, bide, attacker_is_player=True)[0]
    state = _apply_move_effect(
        _apply_move_damage(state, bide, storing, attacker_is_player=True),
        bide,
        storing,
        attacker_is_player=True,
        target_is_player=False,
    )
    assert storing.label == "BIDE_STORING"
    assert state.player_bide_turns == 1

    release = _move_outcomes_for_state(state, bide, attacker_is_player=True)[0]
    assert release.label == "BIDE_RELEASE"
    assert release.damage == 6
    state = _apply_move_effect(
        _apply_move_damage(state, bide, release, attacker_is_player=True),
        bide,
        release,
        attacker_is_player=True,
        target_is_player=False,
    )
    assert state.opponent_hp == 14
    assert state.player_bide_turns == 0
    assert state.player_bide_damage == 0


def test_bide_without_direct_damage_fails_on_release():
    bide = SearchMove(0, "Bide", 1, 1, secondary_effect="BIDE")
    state = SearchState(20, 20, 20, 20, (bide,), ())
    for _ in range(3):
        outcome = _move_outcomes_for_state(state, bide, attacker_is_player=True)[0]
        state = _apply_move_effect(
            _apply_move_damage(state, bide, outcome, attacker_is_player=True),
            bide,
            outcome,
            attacker_is_player=True,
            target_is_player=False,
        )
    assert outcome.label == "BIDE_NO_ENERGY"
    assert outcome.damage == 0
    assert state.player_bide_turns == 0


def test_switching_clears_active_bide_state():
    bide = SearchMove(0, "Bide", 1, 1, secondary_effect="BIDE")
    replacement = SearchSwitch(1, "reserve", 10, 10, (bide,), 10)
    state = SearchState(
        10,
        10,
        10,
        10,
        (bide,),
        (),
        player_party_index=0,
        player_switches=(replacement,),
        player_bide_turns=1,
        player_bide_damage=4,
        player_bide_target="opponent",
    )

    switched = Expectiminimax._switch_state(state, replacement)
    assert switched.player_bide_turns == 0
    assert switched.player_bide_damage == 0
    assert switched.player_bide_target is None


def test_bide_is_a_complete_known_response_and_does_not_create_full_hp_ceiling():
    context = PlannerContext(
        "active",
        20,
        26,
        "Seedot",
        20,
        moves=(PlannerMove(0, "Scratch", 4, 4),),
        items=(RecoveryItem(SimpleNamespace(name="Potion"), 1, 20),),
        opponent_moves=(
            PlannerOpponentMove("Bide", 1, 1, effect="BIDE", response_facts_complete=True),
        ),
        is_trainer=True,
        campaign_owned=True,
        opponent_response_facts_complete=True,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=1, max_duration_seconds=None))

    assert result is not None
    assert result.action_kind == "move"
    assert result.category is SearchResultCategory.HORIZON_PROJECTION
    assert result.projection_safe is True
    potion = next(root for root in result.root_evaluations if root.action_name == "Potion")
    assert potion.expected_score < result.expected_score


def test_search_root_preserves_bide_state_observed_before_planner_call():
    context = PlannerContext(
        "active",
        20,
        20,
        "Seedot",
        20,
        moves=(PlannerMove(0, "Scratch", 1, 1),),
        opponent_moves=(
            PlannerOpponentMove("Bide", 0, 0, effect="BIDE", response_facts_complete=True),
        ),
        active_speed=20,
        opponent_speed=10,
        is_trainer=True,
        campaign_owned=True,
        opponent_response_facts_complete=True,
        opponent_bide_turns=1,
        opponent_bide_damage=4,
    )

    result = search_planner_context(context, config=SearchConfig(max_turns=1, max_duration_seconds=None))

    assert result is not None
    assert result.immediate_projection is not None
    assert any(
        branch.action_name == "Bide"
        and branch.result == "BIDE_RELEASE"
        and branch.damage == 8
        for branch in result.immediate_projection.opponent_responses
    )
