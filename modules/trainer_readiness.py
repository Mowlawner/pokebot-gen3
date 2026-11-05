"""Conservative pre-battle readiness evaluation for known trainer checkpoints.

The live battle planner gets exact battler stats from the ROM.  Before a
trainer battle starts, only the player's party and the published trainer
roster are available, so this adapter builds a deliberately conservative
single-battle model from those facts.  It never predicts the next RNG result:
accuracy, ordinary damage variation, and critical hits remain branches in the
expectiminimax search.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

from modules.battle_expectiminimax import (
    Expectiminimax,
    OpponentPolicy,
    SearchConfig,
    SearchMove,
    SearchResult,
    SearchResultCategory,
    SearchState,
)
from modules.battle_readiness import (
    BattleReadiness,
    BattleReadinessStatus,
    established_search_probability,
)
from modules.pokemon import get_move_by_name, get_species_by_name


# Pre-battle readiness is a gate that runs during campaign navigation. It
# needs to establish safe progress into the known trainer, rather than solve
# the complete battle at live-action depth. Keeping this projection shallow
# prevents a stochastic tree from exhausting its budget and being mistaken
# for evidence that another level of grinding is required.
# A two-turn projection ends before the opening starter matchup has enough
# information to establish a safe line, which makes the controller interpret
# ordinary horizon uncertainty as a request to grind another level. Five turns
# is still a bounded pre-battle calculation, but is long enough for the
# low-level rival exchange to show measurable progress and survive the next
# modeled response without using the live battle deadline.
_PRE_BATTLE_SEARCH_TURNS = 5
# Root-action slicing gives each candidate its own share of the node budget.
# Two common starter moves therefore need roughly twice the single-line
# budget to finish the five-turn readiness projection.
_PRE_BATTLE_SEARCH_NODES = 16_000
# A pre-battle gate has to decide whether preparation is useful, not demand a
# terminal proof of a complete battle before allowing an unavoidable trainer.
# A bounded line must nevertheless include explicit immediate progress and
# survival evidence; unresolved terminal counters are not treated as zero.
_PRE_BATTLE_MAX_ACTIVE_FAINT_PROBABILITY = 0.05


@dataclass(frozen=True, slots=True)
class TrainerReadinessMember:
    """Facts needed to evaluate one living party member."""

    party_index: int
    name: str
    species: str
    level: int
    current_hp: int
    max_hp: int
    speed: int
    moves: tuple[object, ...]
    stats: object | None = None


@dataclass(frozen=True, slots=True)
class TrainerLineAssessment:
    """One party member's result against one member of a checkpoint."""

    party_index: int
    opponent_species: str
    opponent_level: int
    result: SearchResult


@dataclass(frozen=True, slots=True)
class TrainerLeadAssessment:
    """Aggregate expectiminimax evidence for one possible initial lead."""

    party_index: int
    name: str
    source: str
    lines: tuple[TrainerLineAssessment, ...]
    complete: bool
    safe: bool
    minimum_win_probability: float
    maximum_faint_probability: float
    recommended_action: str | None

    @property
    def rank(self) -> tuple:
        """Prefer robust full-roster lines before tactical utility."""

        return (
            self.safe,
            self.complete,
            self.minimum_win_probability,
            -self.maximum_faint_probability,
            -self.party_index,
        )


def _lead_assessment(
    member: TrainerReadinessMember,
    lines: tuple[TrainerLineAssessment, ...],
    trainer_count: int,
    required_hp_ratio: float,
) -> TrainerLeadAssessment:
    results = tuple(line.result for line in lines)
    complete = trainer_count > 0 and len(results) == trainer_count
    minimum_win = min((result.win_probability for result in results), default=0.0)
    maximum_faint = max(
        (getattr(result, "active_faint_probability", result.catastrophic_probability) for result in results),
        default=1.0,
    )
    hp_ratio = member.current_hp / max(1, member.max_hp)
    safe = (
        complete
        and hp_ratio >= required_hp_ratio
        and all(
            _preparation_line_is_sufficient(result)
            and getattr(result, "active_faint_probability", 1.0) <= _PRE_BATTLE_MAX_ACTIVE_FAINT_PROBABILITY
            for result in results
        )
    )
    first_action = lines[0].result.action_name if lines else None
    return TrainerLeadAssessment(
        member.party_index,
        member.name,
        "party",
        lines,
        complete,
        safe,
        minimum_win,
        maximum_faint,
        first_action,
    )


def _preparation_line_is_sufficient(result: SearchResult) -> bool:
    """Return whether a bounded trainer line is sufficient to stop grinding.

    Mandatory trainer readiness requires the battle engine's terminal safety
    contract. A horizon projection may be executable once the live battle
    planner owns the encounter, but its unresolved terminal counters are not
    evidence that preparation is complete.
    """

    category = getattr(result, "category", None)
    # A few compatibility adapters expose the older SearchResult shape.  The
    # live evaluator always supplies ``category``; preserve the older test and
    # integration boundary without weakening the live terminal-complete gate.
    return bool(
        result.is_safe
        and (category is None or category is SearchResultCategory.TERMINAL_COMPLETE)
    )


def _roster_battle_plan(
    checkpoint,
    members: tuple[TrainerReadinessMember, ...],
    per_party: dict[int, list[TrainerLineAssessment]],
    required_hp_ratio: float,
) -> tuple[tuple[tuple[int, int], ...], int, int, int | None]:
    """Assemble a role-based roster plan from independent bounded lines.

    The pre-battle adapter does not yet model the entire live party inside one
    search state. It can still establish the important campaign fact that
    each known opponent has at least one viable living answer. The returned
    plan preserves that evidence and leaves sequencing to the live battle
    planner.
    """

    trainer_party = tuple(getattr(checkpoint, "party", ()) or ())
    member_by_index = {member.party_index: member for member in members}
    plan: list[tuple[int, int]] = []
    for opponent_index, trainer_member in enumerate(trainer_party):
        options = []
        for party_index, lines in per_party.items():
            member = member_by_index.get(party_index)
            if member is None or member.current_hp / max(1, member.max_hp) < required_hp_ratio:
                continue
            matching = tuple(
                line
                for line in lines
                if line.opponent_species == trainer_member.species
                and line.opponent_level == trainer_member.level
                and _preparation_line_is_sufficient(line.result)
                and getattr(line.result, "active_faint_probability", 1.0)
                <= _PRE_BATTLE_MAX_ACTIVE_FAINT_PROBABILITY
            )
            if matching:
                best = max(
                    matching,
                    key=lambda line: (
                        line.result.win_probability,
                        -getattr(line.result, "active_faint_probability", 1.0),
                        -party_index,
                    ),
                )
                options.append((party_index, best))
        if options:
            party_index, _ = max(
                options,
                key=lambda item: (
                    item[1].result.win_probability,
                    -getattr(item[1].result, "active_faint_probability", 1.0),
                    -item[0],
                ),
            )
            plan.append((opponent_index, party_index))
    opening_party_index = next(
        (party_index for opponent_index, party_index in plan if opponent_index == 0),
        None,
    )
    return tuple(plan), len(plan), len(trainer_party), opening_party_index


def _stat_value(stats, name: str, default: int = 1) -> int:
    try:
        return max(1, int(getattr(stats, name)))
    except (AttributeError, TypeError, ValueError):
        return default


def _neutral_stats(species, level: int):
    """Return the low-end trainer stats used for a known level-only roster."""

    level = max(1, int(level))
    base = species.base_stats
    return SimpleNamespace(
        hp=max(1, ((2 * base.hp) * level) // 100 + 10 + level),
        attack=max(1, ((2 * base.attack) * level) // 100 + 5),
        defence=max(1, ((2 * base.defence) * level) // 100 + 5),
        speed=max(1, ((2 * base.speed) * level) // 100 + 5),
        special_attack=max(1, ((2 * base.special_attack) * level) // 100 + 5),
        special_defence=max(1, ((2 * base.special_defence) * level) // 100 + 5),
    )


def _damage_range(move, attacker, defender) -> tuple[int, int, int]:
    """Approximate Gen III damage while preserving type immunities."""

    power = int(getattr(move, "base_power", 0) or 0)
    if power <= 0:
        return 0, 0, 0
    move_type = move.type
    ability_name = getattr(getattr(defender, "ability", None), "name", None)
    if (
        (ability_name == "Levitate" and getattr(move_type, "name", None) == "Ground")
        or (ability_name == "Volt Absorb" and getattr(move_type, "name", None) == "Electric")
        or (ability_name == "Water Absorb" and getattr(move_type, "name", None) == "Water")
        or (ability_name == "Flash Fire" and getattr(move_type, "name", None) == "Fire")
        or (ability_name == "Soundproof" and getattr(move, "is_sound_move", False))
    ):
        return 0, 0, 0
    effectiveness = 1.0
    for defender_type in defender.types:
        effectiveness *= move_type.get_effectiveness_against(defender_type)
    if effectiveness <= 0:
        return 0, 0, 0
    physical = bool(getattr(move_type, "is_physical", False))
    attack_stat = _stat_value(attacker.stats, "attack" if physical else "special_attack")
    defense_stat = _stat_value(defender.stats, "defence" if physical else "special_defence")
    level = max(1, int(getattr(attacker, "level", 1)))
    base = (((2 * level // 5 + 2) * power * attack_stat // defense_stat) // 50) + 2
    if move_type in attacker.types:
        base = (base * 3) // 2
    base = int(base * effectiveness)
    if base <= 0:
        return 0, 0, 0
    minimum = max(1, (base * 85) // 100)
    return minimum, max(minimum, base), max(minimum, base * 2)


def _search_move(move, attacker, defender, index: int) -> SearchMove:
    minimum, maximum, critical = _damage_range(move, attacker, defender)
    accuracy = max(0.0, min(1.0, float(getattr(move, "accuracy", 1.0))))
    return SearchMove(
        index=index,
        name=move.name,
        damage_min=minimum,
        damage_max=maximum,
        accuracy=accuracy,
        priority=int(getattr(move, "priority", 0) or 0),
        critical_damage_max=critical,
        critical_hit_chance=1 / 16,
        secondary_chance=max(0.0, min(1.0, float(getattr(move, "secondary_accuracy", 0.0) or 0.0))),
        secondary_effect=getattr(move, "effect", None),
        damage_stat=(
            "special_attack"
            if getattr(getattr(move, "type", None), "kind", None) == "Special"
            else "attack"
        ),
    )


def _trainer_battler(species_name: str, level: int):
    species = get_species_by_name(species_name)
    stats = _neutral_stats(species, level)
    return SimpleNamespace(
        species=species,
        level=max(1, int(level)),
        stats=stats,
        types=species.types,
        current_hp=stats.hp,
        total_hp=stats.hp,
        ability=species.abilities[0] if species.abilities else None,
    )


def _party_member_from_live(pokemon, party_index: int) -> TrainerReadinessMember:
    return TrainerReadinessMember(
        party_index=party_index,
        name=str(getattr(pokemon, "name", getattr(pokemon, "species", "party member"))),
        species=getattr(getattr(pokemon, "species", None), "name", "Unknown"),
        level=max(1, int(getattr(pokemon, "level", 1))),
        current_hp=max(0, int(getattr(pokemon, "current_hp", 0))),
        max_hp=max(1, int(getattr(pokemon, "total_hp", 1))),
        speed=_stat_value(getattr(pokemon, "stats", None), "speed"),
        moves=tuple(move for move in getattr(pokemon, "moves", ()) if move is not None and getattr(move, "pp", 0) > 0),
        stats=getattr(pokemon, "stats", None),
    )


def live_party_readiness_members(party: Iterable[object]) -> tuple[TrainerReadinessMember, ...]:
    """Normalize live party objects without exposing emulator objects to search."""

    result = []
    for index, pokemon in enumerate(party or ()):
        if getattr(pokemon, "is_egg", False) or getattr(pokemon, "current_hp", 0) <= 0:
            continue
        result.append(_party_member_from_live(pokemon, index))
    return tuple(result)


def live_storage_readiness_members(storage) -> tuple[TrainerReadinessMember, ...]:
    """Normalize readable PC Pokémon for hypothetical roster evaluation.

    Negative indexes keep PC candidates distinct from current party slots and
    make the resulting readiness record directly actionable by a PC planner.
    Storage reads are supplied by the caller so this evaluator never performs
    emulator I/O on its own.
    """

    result = []
    for box in getattr(storage, "boxes", ()) or ():
        for slot in getattr(box, "slots", ()) or ():
            pokemon = getattr(slot, "pokemon", None)
            if pokemon is None or getattr(pokemon, "is_egg", False):
                continue
            try:
                member = _party_member_from_live(
                    pokemon,
                    -(int(box.number) * 30 + int(slot.slot_index) + 1),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
            if member.current_hp > 0 and member.moves:
                result.append(member)
    return tuple(result)


def assess_trainer_checkpoint(
    checkpoint,
    party: Iterable[object] | Iterable[TrainerReadinessMember],
    *,
    expected_exp: int | None = None,
    pc_candidates: Iterable[TrainerReadinessMember] = (),
    required_hp_ratio: float = 0.5,
) -> BattleReadiness:
    """Assess whether the living roster has a safe checkpoint plan.

    A single robust carry remains a valid plan, but it is no longer required.
    When different living party members cover different opposing slots, the
    result records that roster plan and lets the live expectiminimax battle
    planner resolve the actual switch sequence.
    """

    checkpoint_id = getattr(checkpoint, "checkpoint_id", None)
    if expected_exp is None:
        try:
            expected_exp = sum(
                int(get_species_by_name(member.species).base_experience_yield) * int(member.level) // 7
                for member in getattr(checkpoint, "party", ())
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            expected_exp = None
    members = tuple(party or ())
    if members and not isinstance(members[0], TrainerReadinessMember):
        members = live_party_readiness_members(members)
    pc_members = tuple(pc_candidates or ())
    if not members and not pc_members:
        return BattleReadiness(
            checkpoint_id,
            BattleReadinessStatus.UNKNOWN,
            expected_exp=expected_exp,
            reason="no living party member is available for the trainer checkpoint",
        )

    lines: list[TrainerLineAssessment] = []
    per_party: dict[int, list[TrainerLineAssessment]] = {member.party_index: [] for member in members}
    for trainer_member in getattr(checkpoint, "party", ()):
        try:
            defender = _trainer_battler(trainer_member.species, trainer_member.level)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            continue
        for member in members:
            try:
                attacker_species = get_species_by_name(member.species)
                # The live object is not retained here, so calculate stats from
                # its public values and conservative species fallbacks. The
                # adapter below supplies exact stats through ``stats`` when it
                # constructs this normalized member.
                attacker = SimpleNamespace(
                    species=attacker_species,
                    level=member.level,
                    stats=member.stats or _neutral_stats(attacker_species, member.level),
                    types=attacker_species.types,
                    current_hp=member.current_hp,
                    total_hp=member.max_hp,
                )
                player_moves = tuple(
                    _search_move(learned.move, attacker, defender, index)
                    for index, learned in enumerate(member.moves)
                    if getattr(learned, "move", None) is not None
                )
                opponent_moves = tuple(
                    _search_move(get_move_by_name(move_name), defender, attacker, index)
                    for index, move_name in enumerate(trainer_member.moves)
                )
                if not player_moves or not opponent_moves:
                    continue
                result = Expectiminimax(
                    # Readiness is a pre-battle gate, not the full live
                    # battle solve. Use the projection contract over a short
                    # horizon. The node cap remains a structural bound for
                    # this adapter; unlike the live action planner, this
                    # search must finish quickly enough that an incomplete
                    # result cannot repeatedly escalate preparation.
                    SearchConfig(
                        opponent_policy=OpponentPolicy.CONSERVATIVE,
                        max_turns=_PRE_BATTLE_SEARCH_TURNS,
                        max_nodes=_PRE_BATTLE_SEARCH_NODES,
                        max_duration_seconds=None,
                    )
                ).evaluate(
                    SearchState(
                        member.current_hp,
                        member.max_hp,
                        defender.current_hp,
                        defender.total_hp,
                        player_moves,
                        opponent_moves,
                        member.speed,
                        defender.stats.speed,
                        player_party_index=member.party_index,
                        experience_reward=int(expected_exp or 0),
                        battle_objective="defeat",
                        campaign_owned=True,
                    )
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            assessment = TrainerLineAssessment(member.party_index, trainer_member.species, trainer_member.level, result)
            per_party[member.party_index].append(assessment)
            lines.append(assessment)

    trainer_count = len(getattr(checkpoint, "party", ()))
    lead_assessments = tuple(
        _lead_assessment(
            next(member for member in members if member.party_index == party_index),
            tuple(member_lines),
            trainer_count,
            required_hp_ratio,
        )
        for party_index, member_lines in per_party.items()
    )
    safe_leads = tuple(assessment for assessment in lead_assessments if assessment.safe)
    best_lead = max(lead_assessments, key=lambda assessment: assessment.rank, default=None)
    roster_plan, roster_covered, roster_opponent_count, roster_opening = _roster_battle_plan(
        checkpoint,
        members,
        per_party,
        required_hp_ratio,
    )
    roster_ready = (
        roster_opponent_count > 0
        and roster_covered == roster_opponent_count
        and roster_opening is not None
    )

    # When no party member is safe, evaluate explicitly supplied PC candidates
    # as a second roster tier.  The caller decides whether a PC transaction is
    # executable; this function only identifies the best candidate and keeps
    # that recommendation separate from current-party readiness.
    pc_leads: tuple[TrainerLeadAssessment, ...] = ()
    if not safe_leads and pc_members:
        pc_per_party: dict[int, list[TrainerLineAssessment]] = {member.party_index: [] for member in pc_members}
        for trainer_member in getattr(checkpoint, "party", ()):
            try:
                defender = _trainer_battler(trainer_member.species, trainer_member.level)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            for member in pc_members:
                try:
                    attacker_species = get_species_by_name(member.species)
                    attacker = SimpleNamespace(
                        species=attacker_species,
                        level=member.level,
                        stats=member.stats or _neutral_stats(attacker_species, member.level),
                        types=attacker_species.types,
                        current_hp=member.current_hp,
                        total_hp=member.max_hp,
                    )
                    player_moves = tuple(
                        _search_move(learned.move, attacker, defender, index)
                        for index, learned in enumerate(member.moves)
                        if getattr(learned, "move", None) is not None
                    )
                    opponent_moves = tuple(
                        _search_move(get_move_by_name(move_name), defender, attacker, index)
                        for index, move_name in enumerate(trainer_member.moves)
                    )
                    if not player_moves or not opponent_moves:
                        continue
                    result = Expectiminimax(
                        SearchConfig(
                            opponent_policy=OpponentPolicy.CONSERVATIVE,
                            max_turns=_PRE_BATTLE_SEARCH_TURNS,
                            max_nodes=_PRE_BATTLE_SEARCH_NODES,
                            max_duration_seconds=None,
                        )
                    ).evaluate(
                        SearchState(
                            member.current_hp,
                            member.max_hp,
                            defender.current_hp,
                            defender.total_hp,
                            player_moves,
                            opponent_moves,
                            member.speed,
                            defender.stats.speed,
                            player_party_index=member.party_index,
                            experience_reward=int(expected_exp or 0),
                        )
                    )
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                    continue
                pc_per_party[member.party_index].append(
                    TrainerLineAssessment(member.party_index, trainer_member.species, trainer_member.level, result)
                )
        pc_leads = tuple(
            TrainerLeadAssessment(
                member.party_index,
                member.name,
                "pc",
                tuple(pc_per_party[member.party_index]),
                trainer_count > 0 and len(pc_per_party[member.party_index]) == trainer_count,
                trainer_count > 0
                and len(pc_per_party[member.party_index]) == trainer_count
                and member.current_hp / max(1, member.max_hp) >= required_hp_ratio
                and all(
                    _preparation_line_is_sufficient(line.result)
                    and getattr(line.result, "active_faint_probability", 1.0)
                    <= _PRE_BATTLE_MAX_ACTIVE_FAINT_PROBABILITY
                    for line in pc_per_party[member.party_index]
                ),
                min((line.result.win_probability for line in pc_per_party[member.party_index]), default=0.0),
                max(
                    (
                        getattr(line.result, "active_faint_probability", line.result.catastrophic_probability)
                        for line in pc_per_party[member.party_index]
                    ),
                    default=1.0,
                ),
                pc_per_party[member.party_index][0].result.action_name
                if pc_per_party[member.party_index]
                else None,
            )
            for member in pc_members
        )
        safe_pc = tuple(assessment for assessment in pc_leads if assessment.safe)
        if safe_pc:
            best_lead = max(safe_pc, key=lambda assessment: assessment.rank)

    if roster_ready:
        opening_member = next(
            (member for member in members if member.party_index == roster_opening),
            None,
        )
        opening_lines = tuple(per_party.get(roster_opening, ()))
        opening_result = next(
            (
                line.result
                for line in opening_lines
                if line.opponent_species == getattr(getattr(checkpoint, "party", ())[0], "species", None)
                and line.opponent_level == getattr(getattr(checkpoint, "party", ())[0], "level", None)
            ),
            None,
        )
        planned_results = tuple(
            line.result
            for opponent_index, party_index in roster_plan
            for line in per_party.get(party_index, ())
            if opponent_index < roster_opponent_count
            and line.opponent_species == getattr(getattr(checkpoint, "party", ())[opponent_index], "species", None)
            and line.opponent_level == getattr(getattr(checkpoint, "party", ())[opponent_index], "level", None)
        )
        status = BattleReadinessStatus.READY
        lead_source = "party"
        roster_action = "use_party_roster_plan"
        reason = (
            f"roster plan covers {roster_covered}/{roster_opponent_count} known opposing slots "
            "without requiring one Pokémon to solo the checkpoint"
        )
        best_result = opening_result or (planned_results[0] if planned_results else None)
        selected_party_index = roster_opening
        selected_name = opening_member.name if opening_member is not None else None
        selected_win = established_search_probability(best_result, "win_probability")
        selected_faint = established_search_probability(best_result, "active_faint_probability")
        selected_action = getattr(best_result, "action_name", None)
    elif best_lead is None:
        status = BattleReadinessStatus.UNKNOWN
        reason = "trainer or party move facts were incomplete"
        best_result = None
        lead_source = "party"
        roster_action = None
        selected_party_index = None
        selected_name = None
        selected_win = None
        selected_faint = None
        selected_action = None
    else:
        # A PC lead may be tactically safe, but it is not available until the
        # roster change is executed. Keep the campaign out of the trainer
        # sight line until that transaction is complete.
        status = (
            BattleReadinessStatus.PREPARE
            if best_lead.source == "pc" or not best_lead.safe
            else BattleReadinessStatus.READY
        )
        lead_source = best_lead.source
        roster_action = "use_party_lead" if lead_source == "party" else "withdraw_pc_candidate"
        if best_lead.safe and best_lead.source == "party":
            reason = f"{best_lead.source} lead {best_lead.name} has a robust line through the known trainer roster"
        elif best_lead.safe and best_lead.source == "pc":
            reason = f"PC lead {best_lead.name} is safe but requires a roster change before engagement"
        else:
            reason = "no available lead has a robust line through the known trainer roster"
        best_result = best_lead.lines[0].result if best_lead.lines else None
        selected_party_index = best_lead.party_index
        selected_name = best_lead.name
        selected_win = established_search_probability(best_result, "win_probability")
        selected_faint = established_search_probability(best_result, "active_faint_probability")
        selected_action = best_lead.recommended_action
    return BattleReadiness(
        checkpoint_id,
        status,
        selected_win,
        selected_faint,
        selected_action,
        required_hp_ratio,
        expected_exp,
        reason,
        best_result,
        selected_party_index,
        selected_name,
        lead_source,
        lead_assessments + pc_leads,
        roster_action,
        active_faint_probability=(
            established_search_probability(best_result, "active_faint_probability")
            if best_result is not None
            else None
        ),
        party_wipe_probability=(
            established_search_probability(best_result, "party_wipe_probability")
            if best_result is not None
            else None
        ),
        run_loss_probability=(
            established_search_probability(best_result, "run_loss_probability")
            if best_result is not None
            else None
        ),
        roster_plan=roster_plan,
        roster_covered_opponents=roster_covered,
        roster_opponent_count=roster_opponent_count,
    )


def assess_live_trainer_checkpoint(
    checkpoint,
    party,
    *,
    expected_exp: int | None = None,
    pc_candidates: Iterable[TrainerReadinessMember] = (),
    required_hp_ratio: float = 0.5,
) -> BattleReadiness:
    """Build readiness facts from live party objects with their exact stats."""

    members = []
    for index, pokemon in enumerate(party or ()):
        if getattr(pokemon, "is_egg", False) or getattr(pokemon, "current_hp", 0) <= 0:
            continue
        member = _party_member_from_live(pokemon, index)
        member = TrainerReadinessMember(
            member.party_index,
            member.name,
            member.species,
            member.level,
            member.current_hp,
            member.max_hp,
            member.speed,
            member.moves,
            pokemon.stats,
        )
        members.append(member)
    return assess_trainer_checkpoint(
        checkpoint,
        tuple(members),
        expected_exp=expected_exp,
        pc_candidates=pc_candidates,
        required_hp_ratio=required_hp_ratio,
    )
