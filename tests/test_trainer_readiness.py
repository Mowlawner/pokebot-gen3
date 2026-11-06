from types import SimpleNamespace

from modules.trainer_checkpoints import TrainerCheckpoint, TrainerPokemon
from modules.trainer_readiness import (
    TrainerReadinessMember,
    _damage_range,
    _neutral_stats,
    _preparation_line_is_sufficient,
    assess_trainer_checkpoint,
)
from modules.battle_expectiminimax import SearchResultCategory
from modules.battle_readiness import BattleReadinessStatus
from modules.modes.campaign import _known_trainer_checkpoint_on_route
from modules.pokemon import get_move_by_name, get_species_by_name


class FakeType:
    def __init__(self, name, *, effectiveness=None, physical=True):
        self.name = name
        self.is_physical = physical
        self._effectiveness = effectiveness or {}

    def get_effectiveness_against(self, other):
        return self._effectiveness.get(other.name, 1.0)


def _species(name, type_, *, hp=10, attack=5, defence=5, speed=5):
    return SimpleNamespace(
        name=name,
        types=[type_],
        abilities=[],
        base_stats=SimpleNamespace(
            hp=hp,
            attack=attack,
            defence=defence,
            speed=speed,
            special_attack=attack,
            special_defence=defence,
        ),
    )


def _move(name, type_, power=40):
    return SimpleNamespace(name=name, type=type_, base_power=power, accuracy=1.0, priority=0)


def test_trainer_readiness_keeps_immunities_as_zero_damage(monkeypatch):
    flying = FakeType("Flying")
    ground = FakeType("Ground", effectiveness={"Flying": 0.0})
    move = _move("Mud-Slap", ground)
    attacker = SimpleNamespace(
        level=10,
        stats=SimpleNamespace(attack=30, special_attack=30),
        types=[ground],
    )
    defender = SimpleNamespace(
        level=5,
        stats=SimpleNamespace(defence=20, special_defence=20),
        types=[flying],
    )

    assert _damage_range(move, attacker, defender) == (0, 0, 0)


def test_known_trainer_with_a_one_turn_line_is_ready(monkeypatch):
    normal = FakeType("Normal")
    player_species = _species("Starter", normal, hp=40, attack=100, defence=40, speed=100)
    trainer_species = _species("Poochyena", normal, hp=10, attack=5, defence=5, speed=5)
    tackle = _move("Tackle", normal)

    monkeypatch.setattr(
        "modules.trainer_readiness.get_species_by_name",
        lambda name: player_species if name == "Starter" else trainer_species,
    )
    monkeypatch.setattr("modules.trainer_readiness.get_move_by_name", lambda name: tackle)

    party_member = TrainerReadinessMember(
        0,
        "Starter",
        "Starter",
        10,
        40,
        40,
        100,
        (SimpleNamespace(move=tackle, pp=20),),
        SimpleNamespace(
            hp=40,
            attack=100,
            defence=40,
            speed=100,
            special_attack=100,
            special_defence=40,
        ),
    )
    checkpoint = TrainerCheckpoint(
        "test:trainer",
        (0, 17),
        "trainer:test",
        "Test trainer",
        (TrainerPokemon("Poochyena", 5, ("Tackle",)),),
    )

    readiness = assess_trainer_checkpoint(checkpoint, (party_member,))

    assert readiness.status is BattleReadinessStatus.READY
    assert readiness.ready is True
    assert readiness.search is not None
    assert readiness.lead_party_index == 0
    assert readiness.lead_source == "party"


def test_known_trainer_with_low_hp_is_prepare_even_when_the_line_wins(monkeypatch):
    normal = FakeType("Normal")
    player_species = _species("Starter", normal, hp=40, attack=100, defence=40, speed=100)
    trainer_species = _species("Poochyena", normal, hp=10, attack=5, defence=5, speed=5)
    tackle = _move("Tackle", normal)

    monkeypatch.setattr(
        "modules.trainer_readiness.get_species_by_name",
        lambda name: player_species if name == "Starter" else trainer_species,
    )
    monkeypatch.setattr("modules.trainer_readiness.get_move_by_name", lambda name: tackle)

    party_member = TrainerReadinessMember(
        0,
        "Starter",
        "Starter",
        10,
        9,
        40,
        100,
        (SimpleNamespace(move=tackle, pp=20),),
        SimpleNamespace(
            hp=40,
            attack=100,
            defence=40,
            speed=100,
            special_attack=100,
            special_defence=40,
        ),
    )
    checkpoint = TrainerCheckpoint(
        "test:trainer:low-hp",
        (0, 17),
        "trainer:test:low-hp",
        "Test trainer",
        (TrainerPokemon("Poochyena", 5, ("Tackle",)),),
    )

    readiness = assess_trainer_checkpoint(checkpoint, (party_member,))

    assert readiness.status is BattleReadinessStatus.PREPARE
    assert readiness.required_hp_ratio == 0.5


def test_intro_starter_projection_does_not_claim_readiness_for_a_non_progress_horizon_line():
    species = get_species_by_name("Treecko")
    stats = _neutral_stats(species, 5)
    member = TrainerReadinessMember(
        0,
        "Aron",
        "Treecko",
        5,
        stats.hp,
        stats.hp,
        stats.speed,
        tuple(SimpleNamespace(move=get_move_by_name(name), pp=20) for name in ("Pound", "Growl")),
        stats,
    )
    checkpoint = TrainerCheckpoint(
        "rom:BPEE0:532",
        (0, 18),
        "trainer:(0, 18):2",
        "May",
        (TrainerPokemon("Torchic", 5, ("Scratch", "Growl")),),
    )

    readiness = assess_trainer_checkpoint(checkpoint, (member,))

    assert readiness.status is BattleReadinessStatus.PREPARE
    assert readiness.search is not None
    assert readiness.search.complete is True
    # A stat-only root is not evidence that the trainer checkpoint is ready.
    # Its terminal counters are intentionally unavailable at the horizon.
    assert readiness.search.projection_safe is False
    assert readiness.win_probability is None
    assert readiness.run_loss_probability is None


def test_horizon_projection_never_claims_mandatory_trainer_readiness():
    projection = SimpleNamespace(
        is_safe=True,
        category=SearchResultCategory.HORIZON_PROJECTION,
    )

    assert _preparation_line_is_sufficient(projection) is False


def test_preparation_accepts_a_strong_nonterminal_line_before_terminal_proof():
    species = get_species_by_name("Mudkip")
    checkpoint = TrainerCheckpoint(
        "rom:BPEE0:520",
        (0, 18),
        "trainer:(0, 18):2",
        "Brendan",
        (TrainerPokemon("Treecko", 5, ("Pound", "Leer")),),
    )

    def member_at(level):
        stats = _neutral_stats(species, level)
        return TrainerReadinessMember(
            0,
            "Leslie",
            "Mudkip",
            level,
            stats.hp,
            stats.hp,
            stats.speed,
            tuple(SimpleNamespace(move=get_move_by_name(name), pp=20) for name in ("Tackle", "Growl")),
            stats,
        )

    underprepared = assess_trainer_checkpoint(checkpoint, (member_at(5),))
    prepared = assess_trainer_checkpoint(checkpoint, (member_at(6),))

    assert underprepared.status is BattleReadinessStatus.PREPARE
    # Readiness uses the campaign-owned action set, which removes Growl when a
    # damaging move exists. The remaining line is not yet safe enough for the
    # live campaign planner, so preparation must continue.
    assert prepared.status is BattleReadinessStatus.PREPARE
    assert prepared.search is not None
    assert [root.action_name for root in prepared.search.root_evaluations] == ["Tackle"]
    assert prepared.search.run_loss_probability > 0.05


def test_trainer_readiness_can_select_a_safe_pc_lead_after_party_candidates_fail(monkeypatch):
    normal = FakeType("Normal")
    weak_species = _species("Weak", normal, hp=10, attack=5, defence=5, speed=5)
    strong_species = _species("Strong", normal, hp=40, attack=100, defence=40, speed=100)
    trainer_species = _species("Poochyena", normal, hp=10, attack=5, defence=5, speed=5)
    tackle = _move("Tackle", normal)

    monkeypatch.setattr(
        "modules.trainer_readiness.get_species_by_name",
        lambda name: {"Weak": weak_species, "Strong": strong_species, "Poochyena": trainer_species}[name],
    )
    monkeypatch.setattr("modules.trainer_readiness.get_move_by_name", lambda name: tackle)

    weak = TrainerReadinessMember(
        0,
        "Weak",
        "Weak",
        5,
        5,
        10,
        5,
        (SimpleNamespace(move=tackle, pp=20),),
        SimpleNamespace(hp=10, attack=5, defence=5, speed=5, special_attack=5, special_defence=5),
    )
    pc = TrainerReadinessMember(
        -1,
        "Strong",
        "Strong",
        10,
        40,
        40,
        100,
        (SimpleNamespace(move=tackle, pp=20),),
        SimpleNamespace(hp=40, attack=100, defence=40, speed=100, special_attack=100, special_defence=40),
    )
    checkpoint = TrainerCheckpoint(
        "test:trainer",
        (0, 17),
        "trainer:test",
        "Test trainer",
        (TrainerPokemon("Poochyena", 5, ("Tackle",)),),
    )

    readiness = assess_trainer_checkpoint(checkpoint, (weak,), pc_candidates=(pc,))

    assert readiness.status is BattleReadinessStatus.PREPARE
    assert readiness.lead_party_index == -1
    assert readiness.lead_source == "pc"
    assert readiness.roster_action == "withdraw_pc_candidate"


def test_forced_route_exposure_resolves_only_the_trainer_on_the_selected_path():
    trainer_id = "trainer:(0, 17):2"
    route_location = ((0, 17), (8, 8))
    overworld = SimpleNamespace(
        map_id=(0, 17),
        objects=(SimpleNamespace(trainer_id=trainer_id, trainer_defeated=False),),
        triggers=(
            SimpleNamespace(
                affordance_id=trainer_id,
                hazard_locations=frozenset({route_location}),
                activation_locations=frozenset(),
                engagement_locations=frozenset(),
                locations=frozenset({((0, 17), (8, 9))}),
            ),
        ),
    )
    route = SimpleNamespace(
        forced_trainer_exposure=True,
        actions=(SimpleNamespace(source=route_location, destination=route_location),),
    )
    analysis = SimpleNamespace(normal_route=route)
    goal = SimpleNamespace(constraints=SimpleNamespace(trainer_mode=SimpleNamespace(name="AVOID")))

    assert _known_trainer_checkpoint_on_route(overworld, goal, analysis) == trainer_id


def _engine_line(lead_hp, damage, accuracy, opponent_hp, opponent_damage, opponent_accuracy):
    """A real bounded readiness line, built exactly like the pre-battle adapter's."""

    from modules.battle_expectiminimax import (
        Expectiminimax,
        OpponentPolicy,
        SearchConfig,
        SearchMove,
        SearchState,
    )
    from modules.trainer_readiness import _PRE_BATTLE_SEARCH_NODES, _PRE_BATTLE_SEARCH_TURNS

    player = (SearchMove(0, "Scratch", damage[0], damage[1], accuracy),)
    opponent = (
        SearchMove(
            0, "Tackle", opponent_damage[0], opponent_damage[1], opponent_accuracy,
            critical_damage_max=opponent_damage[1] * 2,
        ),
    )
    return Expectiminimax(
        SearchConfig(
            opponent_policy=OpponentPolicy.CONSERVATIVE,
            max_turns=_PRE_BATTLE_SEARCH_TURNS,
            max_nodes=_PRE_BATTLE_SEARCH_NODES,
            max_duration_seconds=None,
        )
    ).evaluate(
        SearchState(
            lead_hp, lead_hp, opponent_hp, opponent_hp, player, opponent, 20, 18,
            battle_objective="defeat", campaign_owned=True,
        )
    )


def test_a_strong_lead_with_a_sub_100_accuracy_move_is_ready_despite_the_horizon_label():
    # A move that can miss can never close every branch, so the engine labels
    # even a lead that wins 99.6% of the time a horizon projection.  Readiness
    # must decide on the measured numbers instead.
    line = _engine_line(32, (8, 10), 0.95, 28, (4, 6), 0.95)

    assert line.win_probability > 0.95
    assert _preparation_line_is_sufficient(line) is True


def test_a_lv9_lead_is_ready_and_a_lv8_lead_is_not_against_the_same_trainer():
    ready = _engine_line(29, (7, 9), 0.95, 28, (4, 6), 0.95)
    not_ready = _engine_line(26, (5, 7), 0.95, 28, (4, 6), 0.95)

    assert _preparation_line_is_sufficient(ready) is True
    assert _preparation_line_is_sufficient(not_ready) is False  # about 70% win, 30% faint


def test_a_line_whose_battle_has_not_resolved_by_the_horizon_is_not_ready():
    # Needs more than five turns to finish: almost nothing has resolved.
    unresolved = _engine_line(26, (5, 7), 1.0, 28, (4, 6), 0.95)

    assert _preparation_line_is_sufficient(unresolved) is False


def test_horizon_acceptance_requires_small_unresolved_mass_and_unexhausted_budget():
    def line(**changes):
        values = dict(
            is_safe=False,
            category=SearchResultCategory.HORIZON_PROJECTION,
            complete=True,
            budget_exhaustion=None,
            win_probability=0.97,
            loss_probability=0.02,
            catastrophic_probability=0.02,
            minimum_safe_win_probability=0.90,
            maximum_safe_loss_probability=0.05,
            maximum_safe_catastrophic_probability=0.05,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    assert _preparation_line_is_sufficient(line()) is True  # 1% unresolved
    assert _preparation_line_is_sufficient(line(win_probability=0.90, loss_probability=0.02)) is False  # 8%
    assert _preparation_line_is_sufficient(line(budget_exhaustion="nodes")) is False
    assert _preparation_line_is_sufficient(line(complete=False)) is False
    assert _preparation_line_is_sufficient(line(win_probability=0.88, loss_probability=0.11)) is False
    assert _preparation_line_is_sufficient(line(loss_probability=0.06, win_probability=0.94)) is False
    assert _preparation_line_is_sufficient(line(catastrophic_probability=0.06)) is False


def test_other_search_categories_are_never_sufficient():
    for category in (SearchResultCategory.INCOMPLETE,):
        result = SimpleNamespace(is_safe=True, category=category, win_probability=1.0, loss_probability=0.0)
        assert _preparation_line_is_sufficient(result) is False
