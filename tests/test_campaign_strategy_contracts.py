from types import SimpleNamespace

from modules.battle_expectiminimax import Expectiminimax, SearchConfig, SearchMove, SearchState
from modules.nuzlocke.capture_policy import (
    CapturePolicyContext,
    EncounterCandidate,
    EncounterMethod,
    EncounterPortfolioContext,
    choose_encounter_portfolio,
)
from modules.nuzlocke.experience import TrainingCandidate, choose_training_lead
from modules.nuzlocke.resource_policy import (
    DuplicateEncounterPolicy,
    WildBattleIntent,
    WildBattlePolicyInput,
    choose_wild_battle_intent,
)
from modules.nuzlocke.trainer_policy import (
    TrainerExposureKind,
    TrainerPolicyInput,
    choose_trainer_mode,
)
from modules.trainer_readiness import (
    TrainerLineAssessment,
    TrainerReadinessMember,
    _roster_battle_plan,
)
from modules.goals import TrainerMode


def test_portfolio_removes_areas_with_no_legal_species_after_species_clause():
    candidates = (
        EncounterCandidate(
            (0, 101),
            EncounterMethod.LAND,
            species_pool=("Zigzagoon",),
            species_rates=(("Zigzagoon", 1),),
        ),
        EncounterCandidate(
            (0, 116),
            EncounterMethod.LAND,
            species_pool=("Whismur",),
            species_rates=(("Whismur", 1),),
        ),
        EncounterCandidate(
            (0, 102),
            EncounterMethod.LAND,
            species_pool=("Ralts",),
            species_rates=(("Ralts", 1),),
        ),
    )
    plan = choose_encounter_portfolio(
        candidates,
        EncounterPortfolioContext(
            capture=CapturePolicyContext(
                species_clause_enabled=True,
                captured_species=frozenset({"Zigzagoon", "Whismur"}),
            )
        ),
    )

    assert [candidate.area for candidate in plan.candidates] == [(0, 102)]
    assert plan.selected.area == (0, 102)
    assert "legal conditioned outcome" not in plan.reason


def test_portfolio_prefers_level_appropriate_area_when_capture_value_is_close():
    candidates = (
        EncounterCandidate(
            (0, 101),
            EncounterMethod.LAND,
            species_pool=("Zigzagoon",),
            min_level=2,
            max_level=3,
        ),
        EncounterCandidate(
            (0, 116),
            EncounterMethod.LAND,
            species_pool=("Whismur",),
            min_level=8,
            max_level=10,
        ),
    )
    plan = choose_encounter_portfolio(
        candidates,
        EncounterPortfolioContext(
            capture=CapturePolicyContext(),
            current_level=9,
            max_encounters=1,
        ),
    )

    assert plan.selected.area == (0, 116)


def test_duplicate_training_is_policy_driven():
    assert (
        choose_wild_battle_intent(
            WildBattlePolicyInput(
                duplicate_encounter=True,
                duplicate_policy=DuplicateEncounterPolicy.TRAIN,
            )
        )
        is WildBattleIntent.TRAIN
    )
    assert (
        choose_wild_battle_intent(
            WildBattlePolicyInput(
                duplicate_encounter=True,
                duplicate_policy=DuplicateEncounterPolicy.RUN,
            )
        )
        is WildBattleIntent.AVOID
    )


def test_capture_intent_overrides_duplicate_and_training_preferences():
    assert (
        choose_wild_battle_intent(
            WildBattlePolicyInput(
                legal_capture_target=True,
                duplicate_encounter=True,
                preparation_requested=True,
                duplicate_policy=DuplicateEncounterPolicy.TRAIN,
            )
        )
        is WildBattleIntent.CAPTURE
    )


def test_costly_trainer_can_be_promoted_only_when_ready():
    assert choose_trainer_mode(
        TrainerPolicyInput(
            exposure=TrainerExposureKind.COSTLY_TO_AVOID,
            experience_needed=True,
            party_can_survive=True,
        )
    ) is TrainerMode.ENGAGE
    assert choose_trainer_mode(
        TrainerPolicyInput(
            exposure=TrainerExposureKind.COSTLY_TO_AVOID,
            experience_needed=True,
            party_can_survive=False,
        )
    ) is TrainerMode.AVOID


def test_training_scheduler_concentrates_exp_on_one_direct_trainee():
    decision = choose_training_lead(
        (
            TrainingCandidate(0, level=7, target_level=10, matchup_score=3),
            TrainingCandidate(1, level=8, target_level=10, matchup_score=9),
        )
    )

    assert decision.party_index == 0
    assert "directly train" in decision.reason


def test_roster_readiness_can_assign_different_party_members_to_different_opponents():
    checkpoint = SimpleNamespace(
        party=(
            SimpleNamespace(species="Poochyena", level=5),
            SimpleNamespace(species="Zigzagoon", level=5),
        )
    )
    members = (
        TrainerReadinessMember(0, "Poochyena", "Poochyena", 5, 20, 20, 10, ()),
        TrainerReadinessMember(1, "Zigzagoon", "Zigzagoon", 5, 20, 20, 10, ()),
    )
    safe = lambda: SimpleNamespace(
        is_safe=True,
        complete=True,
        win_probability=1.0,
        active_faint_probability=0.0,
        catastrophic_probability=0.0,
    )
    per_party = {
        0: [TrainerLineAssessment(0, "Poochyena", 5, safe())],
        1: [TrainerLineAssessment(1, "Zigzagoon", 5, safe())],
    }

    plan, covered, count, opening = _roster_battle_plan(checkpoint, members, per_party, 0.5)

    assert plan == ((0, 0), (1, 1))
    assert (covered, count, opening) == (2, 2, 0)


def test_avoid_intent_makes_a_safe_escape_compete_as_the_successful_result():
    result = Expectiminimax(SearchConfig(max_turns=3, max_nodes=1000)).evaluate(
        SearchState(
            player_hp=20,
            player_max_hp=20,
            opponent_hp=5,
            opponent_max_hp=5,
            player_moves=(SearchMove(0, "Tackle", 5, 5),),
            opponent_moves=(SearchMove(0, "Tackle", 1, 1),),
            player_speed=10,
            opponent_speed=5,
            battle_objective="avoid",
            can_run=True,
            escape_probability=1.0,
        )
    )

    assert result.action_kind == "run"
    assert result.win_probability == 1.0
