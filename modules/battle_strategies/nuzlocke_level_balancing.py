"""Level-balancing battle strategies that respect Nuzlocke deaths."""

from __future__ import annotations

from dataclasses import dataclass

from modules.battle_state import BattleState
from modules.battle_observation import default_battle_recorder
from modules.battle_planner import (
    PlannerAction,
    PlannerDecisionClass,
    _recovery_item,
    _immediate_projection_text,
    _switch_options_text,
    format_search_principal_variation,
    opponent_response_move_names,
    plan_battle_state,
)
from modules.battle_expectiminimax import SearchResultCategory
from modules.battle_strategies import DefaultBattleStrategy, TurnAction
from modules.battle_strategies.campaign import CampaignExpectiminimaxStrategy
from modules.battle_strategies.catch import CatchStrategy
from modules.battle_strategies._util import BattleStrategyUtil
from modules.battle_strategies.switch_evaluator import (
    SwitchCandidateFacts,
    SwitchEvaluationContext,
    rank_switch_candidates,
    relative_training_score,
)
from modules.context import context
from modules.console import diagnostic_print
from modules.items import ItemBattleUse, get_item_bag
from modules.nuzlocke.identity import PokemonIdentity
from modules.nuzlocke.field_lead import CampaignFieldLeadContext, CampaignFieldLeadDecision
from modules.nuzlocke.experience import (
    ExperienceDistribution,
    TrainingCandidate,
    distribute_gen3_experience,
    gen3_battle_experience_reward,
    choose_training_lead,
)
from modules.nuzlocke.level_cap import experience_remaining_in_cap
from modules.pokemon import StatusCondition, get_move_by_name
from modules.pokemon_party import get_party
from modules.player import get_player_avatar


# Utility penalties for capture decisions.  The encounter penalty is large
# enough that a small catch-rate improvement cannot justify a meaningful KO
# tail; party loss remains the dominant failure mode by design.
# Losing the one legal encounter is a hard policy failure. Capture weakening
# may improve catch odds only when every modeled outcome remains nonlethal.
_CAPTURE_ENCOUNTER_LOSS_PENALTY = float("inf")
_CAPTURE_PARTY_FAINT_PENALTY = 100.0
# A trainer replacement prompt happens before the incoming Pokémon gets its
# next action. The replacement therefore needs to survive the first response,
# while a mid-turn emergency switch needs the more conservative two-response
# horizon below.
_FREE_TRAINER_SWITCH_RESPONSE_TURNS = 1
_EMERGENCY_SWITCH_RESPONSE_TURNS = 2
_RESPONSE_TIMING_ACTIVE_TURN = "active_turn"
_RESPONSE_TIMING_MID_TURN_SWITCH = "mid_turn_switch"
_RESPONSE_TIMING_PRE_TURN_REPLACEMENT = "pre_turn_replacement"
# A line that survives at least 85% of the modeled outcomes is preferable to
# stopping the bot when no deterministic line exists.  This is intentionally
# a high bar for a Nuzlocke, while still permitting a single critical-only
# exposure (93.75% survival) and similar common cases.
_MIN_ACCEPTABLE_SURVIVAL_PROBABILITY = 0.85
_CAP_UNSET = object()


def _campaign_progression_is_active() -> bool:
    """Return whether this battle belongs to the strict campaign mode."""

    return getattr(context, "bot_mode", None) == "Campaign Progression"


class _NuzlockeBattleDispatch:
    """Bridge campaign ownership and the pre-campaign battle modes.

    Campaign Progression calls the dedicated expectiminimax implementation.
    Other modes retain the established default strategy behavior.  This is a
    dispatch helper rather than a superclass relationship so campaign
    strategies cannot reach ``DefaultBattleStrategy`` through their MRO.
    """

    def decide_turn(self, battle_state: BattleState):
        if _campaign_progression_is_active():
            return CampaignExpectiminimaxStrategy.decide_turn(self, battle_state)
        return DefaultBattleStrategy.decide_turn(self, battle_state)


def _expectiminimax_reason(result, *, suffix: str) -> str:
    """Keep search confidence and shallow RNG branches visible in diagnostics."""

    action_name = getattr(result, "action_name", None) or "best available action"
    category = getattr(getattr(result, "category", None), "value", "unknown")
    line = format_search_principal_variation(result)
    immediate = _immediate_projection_text(result)
    complete = category == SearchResultCategory.TERMINAL_COMPLETE.value and bool(
        getattr(result, "complete", False)
    )
    win_probability = getattr(result, "win_probability", None)
    catastrophic_probability = getattr(result, "catastrophic_probability", None)
    win_text = f"{win_probability:.1%}" if complete and isinstance(win_probability, (int, float)) else "unknown"
    catastrophe_text = (
        f"{catastrophic_probability:.1%}"
        if complete and isinstance(catastrophic_probability, (int, float))
        else "unknown"
    )
    return (
        f"expectiminimax selected {action_name}; win={win_text}; party-wipe risk={catastrophe_text}; "
        f"category={category!r}; "
        f"line={line or 'unavailable'}; immediate={immediate}; "
        f"switches={_switch_options_text(result)}; {suffix}"
    )


def _preparation_target_suffix() -> str:
    target = getattr(context, "campaign_preparation_target", None)
    return f"; target {target}" if target else ""


def _campaign_preparation_is_active() -> bool:
    """Return whether campaign preparation currently owns wild-battle EXP."""

    return bool(getattr(context, "campaign_preparation_target", None))


def _planner_move_is_zero_damage(battle_state: BattleState, planner_report) -> bool:
    """Return whether the selected planner move cannot lower opponent HP."""

    planner_action = getattr(planner_report, "action", None)
    if planner_action is PlannerAction.UseMove:
        target = getattr(planner_report, "target", None)
    else:
        candidate = getattr(planner_report, "best_available_candidate", None)
        if getattr(candidate, "action", None) is not PlannerAction.UseMove:
            return False
        target = getattr(candidate, "target", None)
    active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
    opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
    try:
        learned = active.moves[target]
        if learned is None:
            return True
        move = learned.move
        try:
            damage = BattleStrategyUtil(battle_state).calculate_move_damage_range(move, active, opponent)
            return int(getattr(damage, "max", 0)) <= 0
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return int(getattr(move, "base_power", 0) or 0) <= 0
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return False


@dataclass(frozen=True)
class _SwitchRisk:
    """One normalized evaluation of a prospective switch line.

    A switch is not safe merely because the replacement can damage the
    opponent.  This object keeps the offensive line and the response line
    together so every caller makes the same decision.  ``critical`` is kept
    separate from ordinary survival: a critical-only KO is possible, but it
    is materially less likely than an ordinary lethal response and should be
    ranked below a line with no modeled lethal outcome.
    """

    damage_min: int
    damage_max: int
    target_hp: int | None
    normal_incoming: int | None
    critical_incoming: int | None
    critical_ko_probability: float | None
    survival_probability: float | None
    reason: str

    @property
    def normal_survives(self) -> bool:
        return (
            self.target_hp is not None
            and self.normal_incoming is not None
            and self.normal_incoming < self.target_hp
        )

    @property
    def strict_safe(self) -> bool:
        return (
            self.normal_incoming is not None
            and self.critical_incoming is not None
            and self.target_hp is not None
            and self.normal_incoming < self.target_hp
            and self.critical_incoming < self.target_hp
        )

    @property
    def probabilistically_safe(self) -> bool:
        return (
            self.survival_probability is not None
            and self.survival_probability >= _MIN_ACCEPTABLE_SURVIVAL_PROBABILITY
        )


def _nuzlocke_dead_identities() -> frozenset[PokemonIdentity]:
    runtime = getattr(context, "nuzlocke_runtime", None)
    projection = getattr(runtime, "rules_projection", None)
    state = getattr(projection, "state", None)
    return frozenset(getattr(state, "dead_pokemon", ()) or ())


def _is_eligible_pokemon(pokemon) -> bool:
    if getattr(pokemon, "is_egg", False):
        return False
    # Ordinary Level Grind has no Nuzlocke runtime and retains its historical
    # behavior. During campaign preparation, an unreadable identity is still
    # a valid live party observation; only an identity explicitly present in
    # the death projection should exclude a Pokémon from the policy. The
    # party reader already filters malformed/empty records before they reach
    # this strategy.
    if getattr(context, "nuzlocke_runtime", None) is None:
        return True
    identity = PokemonIdentity.from_pokemon(pokemon)
    return identity is None or identity not in _nuzlocke_dead_identities()


def _eligible_indices(*, only_non_fainted: bool = False) -> tuple[int, ...]:
    return tuple(
        index
        for index, pokemon in enumerate(get_party())
        if _is_eligible_pokemon(pokemon) and (not only_non_fainted or pokemon.current_hp > 0)
    )


def _lowest_level_index(*, only_non_fainted: bool = False) -> int | None:
    indices = _eligible_indices(only_non_fainted=only_non_fainted)
    return min(indices, key=lambda index: _training_priority(index)) if indices else None


def _training_priority(index: int) -> tuple[int, int, int]:
    """Order candidates by level, then by progress within that level."""

    pokemon = get_party()[index]
    try:
        total_exp = pokemon.total_exp
        if not isinstance(total_exp, int):
            total_exp = 0
    except (AttributeError, RuntimeError, TypeError, ValueError):
        total_exp = 0
    level = getattr(pokemon, "level", 0)
    if not isinstance(level, int):
        level = 0
    return level, total_exp, index


def _experience_headroom(index: int, level_cap: int | None = None) -> int | None:
    """Return a party member's remaining legal EXP before the active cap."""

    try:
        pokemon = get_party()[index]
        return experience_remaining_in_cap(pokemon, level_cap) if level_cap is not None else None
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return None


def _battle_training_priority(index: int, level_cap=_CAP_UNSET) -> tuple:
    """Order battle candidates by cap headroom, then legacy EXP priority."""

    if level_cap is _CAP_UNSET:
        level_cap = _active_level_cap()
    headroom = _experience_headroom(index, level_cap)
    if headroom is not None:
        # ``relative_training_score`` rewards the lexicographically smallest
        # priority. Negating headroom therefore prefers the member that can
        # absorb the largest legal EXP award before the current cap.
        return 0, -headroom, *_training_priority(index)
    return 1, *_training_priority(index)


def _battle_pokemon_key(pokemon) -> tuple:
    """Return a stable key for one opposing party member during a battle."""

    return (
        getattr(pokemon, "party_index", getattr(pokemon, "index", None)),
        getattr(getattr(pokemon, "species", None), "name", getattr(pokemon, "name", None)),
        getattr(pokemon, "level", None),
    )


def _active_party_indices(battle_state: BattleState) -> frozenset[int]:
    """Return every currently deployed party slot from a battle snapshot."""

    own_side = getattr(battle_state, "own_side", None)
    battlers = list(getattr(own_side, "active_battlers", ()) or ())
    battlers.extend(
        battler
        for battler in (
            getattr(own_side, "active_battler", None),
            getattr(own_side, "left_battler", None),
            getattr(own_side, "right_battler", None),
        )
        if battler is not None
    )
    return frozenset(
        getattr(battler, "party_index", None)
        for battler in battlers
        if getattr(battler, "party_index", None) is not None
    )


def _party_exp_snapshot() -> dict[int, int]:
    """Read party EXP values without making EXP bookkeeping a battle dependency."""

    try:
        return {
            index: int(getattr(pokemon, "total_exp"))
            for index, pokemon in enumerate(get_party())
            if isinstance(getattr(pokemon, "total_exp", None), int)
        }
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return {}


def _field_threats_for_current_map() -> tuple[tuple[object, int], ...]:
    """Return weighted wild threats for the current map, when ROM data permits."""

    try:
        from modules.battle_state import EncounterType, get_encounter_type
        from modules.map import get_wild_encounters_for_map

        map_group, map_number = get_player_avatar().map_group_and_number
        encounter_list = get_wild_encounters_for_map(map_group, map_number)
        if encounter_list is None:
            return ()
        encounter_type = get_encounter_type()
        encounter_attribute = {
            EncounterType.Land: "land_encounters",
            EncounterType.Surfing: "surf_encounters",
            EncounterType.RockSmash: "rock_smash_encounters",
            EncounterType.FishingWithOldRod: "old_rod_encounters",
            EncounterType.FishingWithGoodRod: "good_rod_encounters",
            EncounterType.FishingWithSuperRod: "super_rod_encounters",
        }.get(encounter_type, "land_encounters")
        encounters = getattr(encounter_list, encounter_attribute, ())
        return tuple(
            (encounter.species, max(1, int(getattr(encounter, "encounter_rate", 1))))
            for encounter in encounters
            if getattr(encounter, "species", None) is not None
        )
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError, IndexError):
        return ()


def _field_encounter_level_ceiling() -> int | None:
    """Return the highest ordinary wild level available on the current map."""

    try:
        from modules.battle_state import EncounterType, get_encounter_type
        from modules.map import get_wild_encounters_for_map

        map_group, map_number = get_player_avatar().map_group_and_number
        encounter_list = get_wild_encounters_for_map(map_group, map_number)
        if encounter_list is None:
            return None
        encounter_attribute = {
            EncounterType.Land: "land_encounters",
            EncounterType.Surfing: "surf_encounters",
            EncounterType.RockSmash: "rock_smash_encounters",
            EncounterType.FishingWithOldRod: "old_rod_encounters",
            EncounterType.FishingWithGoodRod: "good_rod_encounters",
            EncounterType.FishingWithSuperRod: "super_rod_encounters",
        }.get(get_encounter_type(), "land_encounters")
        encounters = getattr(encounter_list, encounter_attribute, ())
        levels = tuple(int(getattr(encounter, "max_level")) for encounter in encounters)
        return max(levels) if levels else None
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError, IndexError):
        return None


def _field_coverage_profile(pokemon, threats: tuple[tuple[object, int], ...]) -> tuple[float, int, int] | None:
    """Estimate a field lead's coverage against the local encounter envelope."""

    if not threats:
        return None
    try:
        damaging_moves = tuple(
            learned_move
            for learned_move in pokemon.moves
            if learned_move is not None
            and learned_move.pp > 0
            and learned_move.move.base_power > 0
        )
        if not damaging_moves:
            return None
        weighted_coverage = 0.0
        covered_weight = 0
        super_effective_weight = 0
        for species, weight in threats:
            best_effectiveness = 0.0
            for learned_move in damaging_moves:
                effectiveness = 1.0
                for defender_type in species.types:
                    effectiveness *= learned_move.move.type.get_effectiveness_against(defender_type)
                best_effectiveness = max(best_effectiveness, effectiveness)
            if best_effectiveness > 0:
                covered_weight += weight
            if best_effectiveness > 1:
                super_effective_weight += weight
            weighted_coverage += weight * best_effectiveness
        return weighted_coverage, covered_weight, super_effective_weight
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return None


def _trainer_replacement_pokemon_trace(pokemon, index: int | None = None) -> tuple:
    try:
        return (
            index,
            getattr(pokemon, "index", None),
            getattr(getattr(pokemon, "species", None), "name", getattr(pokemon, "name", None)),
            getattr(pokemon, "level", None),
            getattr(pokemon, "current_hp", None),
            getattr(pokemon, "total_hp", None),
            getattr(pokemon, "is_empty", None),
            getattr(pokemon, "is_valid", None),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return (index, "<unreadable>")


def _trace_trainer_replacement_policy(message: str) -> None:
    try:
        import traceback

        caller = traceback.extract_stack(limit=3)[-2]
        caller_summary = f"{caller.name}:{caller.lineno}"
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        caller_summary = "<unavailable>"
    diagnostic_print(
        lambda: (
            f"TRAINER_REPLACEMENT_POLICY: context_frame={getattr(context, 'frame', None)!r} "
            f"emulator_frame={context.emulator.get_frame_count() if getattr(context, 'emulator', None) else None!r} "
            f"caller={caller_summary!r} strategy='NuzlockeLevelBalancingBattleStrategy' {message}"
        ),
        trace=True,
        prefix="TRAINER_REPLACEMENT_POLICY:",
    )


def _trace_switch_candidate(index: int, pokemon, profile: _SwitchRisk | None, result: str) -> None:
    """Emit one compact, opt-in explanation for a switch candidate."""

    diagnostic_print(
        lambda: (
            "BATTLE_SWITCH_CANDIDATE: "
            f"index={index!r} species={getattr(getattr(pokemon, 'species', None), 'name', getattr(pokemon, 'name', None))!r} "
            f"level={getattr(pokemon, 'level', None)!r} hp={getattr(pokemon, 'current_hp', None)!r}/"
            f"{getattr(pokemon, 'total_hp', None)!r} result={result!r} "
            f"damage={getattr(profile, 'damage_min', None)!r}-{getattr(profile, 'damage_max', None)!r} "
            f"normal_incoming={getattr(profile, 'normal_incoming', None)!r} "
            f"critical_incoming={getattr(profile, 'critical_incoming', None)!r} "
            f"survival_probability={getattr(profile, 'survival_probability', None)!r} "
            f"reason={getattr(profile, 'reason', None)!r}"
        ),
        trace=True,
        prefix="BATTLE_SWITCH_CANDIDATE:",
    )


def _trace_capture_candidate(index: int, move, profile: _SwitchRisk | None, result: str) -> None:
    """Emit one compact, opt-in explanation for a capture action candidate."""

    move_name = getattr(move, "name", None)
    if move_name is None:
        move_name = getattr(getattr(move, "move", None), "name", None)
    diagnostic_print(
        lambda: (
            "BATTLE_CAPTURE_CANDIDATE: "
            f"move_index={index!r} move={move_name!r} result={result!r} "
            f"damage={getattr(profile, 'damage_min', None)!r}-{getattr(profile, 'damage_max', None)!r} "
            f"normal_incoming={getattr(profile, 'normal_incoming', None)!r} "
            f"critical_incoming={getattr(profile, 'critical_incoming', None)!r} "
            f"survival_probability={getattr(profile, 'survival_probability', None)!r} "
            f"reason={getattr(profile, 'reason', None)!r}"
        ),
        trace=True,
        prefix="BATTLE_CAPTURE_CANDIDATE:",
    )


def _active_level_cap() -> int | None:
    """Return the currently active cap when campaign facts can prove one."""

    runtime = getattr(context, "nuzlocke_runtime", None)
    if runtime is None:
        return None
    try:
        from modules.nuzlocke.campaign_controller import runtime_campaign_state
        from modules.nuzlocke.level_cap import assess_level_cap
        from modules.nuzlocke.rule_config import CampaignRuleId

        rule_config = getattr(runtime, "rule_config", None)
        if rule_config is not None and not rule_config.is_enabled(CampaignRuleId.LEVEL_CAP):
            return None
        assessment = assess_level_cap(runtime_campaign_state().campaign_facts, get_party())
        return assessment.level_cap if getattr(assessment.status, "value", None) == "known" else None
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None


def _encounter_experience_reward(battle_state: BattleState, opponent=None) -> int | None:
    """Return the Gen III reward before participant splitting."""

    try:
        opponent = opponent or battle_state.opponent.active_battler
        base_yield = opponent.species.base_experience_yield
        level = opponent.level
        if not isinstance(base_yield, int) or not isinstance(level, int):
            return None
        return gen3_battle_experience_reward(
            base_yield,
            level,
            trainer_battle=bool(getattr(battle_state, "is_trainer_battle", False)),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _held_item_effect(pokemon) -> str | None:
    """Return a normalized held-item effect without requiring a live Item."""

    try:
        item = getattr(pokemon, "held_item", None)
        effect = getattr(item, "hold_effect", None)
        value = getattr(effect, "value", effect)
        return str(value).casefold() if value is not None else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _exp_share_indices() -> tuple[int, ...]:
    try:
        return tuple(
            index
            for index, pokemon in enumerate(get_party())
            if _held_item_effect(pokemon) == "exp_share"
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return ()


def _lucky_egg_indices() -> tuple[int, ...]:
    try:
        return tuple(
            index
            for index, pokemon in enumerate(get_party())
            if _held_item_effect(pokemon) == "lucky_egg"
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return ()


def _experience_distribution_for_participants(
    battle_state: BattleState,
    participant_indices: set[int] | frozenset[int],
    *,
    opponent=None,
) -> ExperienceDistribution | None:
    """Project the EXP awarded for one foe and one participant set."""

    reward = _encounter_experience_reward(battle_state, opponent)
    if reward is None:
        return None
    try:
        return distribute_gen3_experience(
            reward,
            sorted(participant_indices),
            _exp_share_indices(),
            lucky_egg_holders=_lucky_egg_indices(),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return None


def _participants_remain_cap_legal(
    battle_state: BattleState, participant_indices: set[int], *, opponent=None
) -> bool:
    """Check projected split EXP for every recipient of the win."""

    cap = _active_level_cap()
    if cap is None:
        return True
    distribution = _experience_distribution_for_participants(
        battle_state,
        participant_indices,
        opponent=opponent,
    )
    if distribution is None:
        # A cap is active but the encounter's EXP yield is not observable;
        # optional training tags must not be authorized on an unknown budget.
        return False
    from modules.nuzlocke.level_cap import can_receive_experience_without_exceeding_cap

    return all(
        can_receive_experience_without_exceeding_cap(get_party()[index], amount, cap)
        for index, amount in distribution.awards
        if amount > 0
    )


def _battle_capable_indices(
    strategy: "NuzlockeLevelBalancingBattleStrategy", battle_state: BattleState | None = None
) -> tuple[int, ...]:
    """Return living members that can damage the current opponent, if known."""

    candidates = tuple(
        index for index in _eligible_indices(only_non_fainted=True) if strategy.pokemon_can_battle(get_party()[index])
    )
    opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
    if opponent is None:
        return candidates

    try:
        util = BattleStrategyUtil(battle_state)
        return tuple(
            index for index in candidates if util.get_strongest_move_against(get_party()[index], opponent) is not None
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        # A live battle with unavailable matchup facts is not proof that a
        # generic damaging move is effective.  The caller can still fall back
        # to the ordinary planner/escape policy.
        return ()


def _lowest_battle_capable_index(
    strategy: "NuzlockeLevelBalancingBattleStrategy", battle_state: BattleState | None = None
) -> int | None:
    """Return the preferred living member that can damage this opponent."""

    candidates = _battle_capable_indices(strategy, battle_state)
    level_cap = _active_level_cap()
    return min(candidates, key=lambda index: _battle_training_priority(index, level_cap)) if candidates else None


class NuzlockeRunAwayStrategy(_NuzlockeBattleDispatch, CampaignExpectiminimaxStrategy):
    """Campaign escape policy whose root action still comes from the planner."""

    def allows_legacy_fallback(self) -> bool:
        return False

    def should_flee_after_faint(self, battle_state: BattleState) -> bool:
        return True

    def choose_new_lead_after_battle(self) -> int | None:
        return None

    def decide_turn(self, battle_state: BattleState):
        if _campaign_progression_is_active():
            return CampaignExpectiminimaxStrategy.decide_turn(self, battle_state)
        escape = BattleStrategyUtil(battle_state).get_best_escape_method()
        if escape is not None:
            return escape
        return _NuzlockeBattleDispatch.decide_turn(self, battle_state)


class NuzlockeLevelBalancingBattleStrategy(_NuzlockeBattleDispatch, CampaignExpectiminimaxStrategy):
    """Balance living campaign Pokémon without ever sending out a dead one."""

    def allows_legacy_fallback(self) -> bool:
        """Require Campaign Progression to resolve turns through planning."""

        return False

    @staticmethod
    def _campaign_exp_tag_enabled() -> bool:
        """Allow legacy isolated strategy tests while campaign opts out.

        Campaign Progression sets an explicit false policy and trains by
        rotating the intended lead directly. Older strategy embeddings do
        not mount that campaign contract, so retain their focused EXP-tag
        behavior until those callers migrate.
        """

        return bool(
            getattr(context, "campaign_allow_exp_tag", False)
            or getattr(context, "bot_mode", None) != "Campaign Progression"
        )

    def __init__(self):
        super().__init__()
        # If the weak lead cannot safely continue, the inherited strategy may
        # switch to a stronger party member.  Keep that support Pokémon in
        # until the battle ends; otherwise the next turn's lowest-level
        # policy immediately switches back and the battle can ping-pong
        # between the two members without resolving.
        self._battle_support_index: int | None = None
        # Emergency switches are selected against a bounded response horizon.
        # Keep that horizon with the selected support member so the executor's
        # menu-boundary validation checks the same line that selected it.
        self._battle_support_response_turns: int | None = None
        # A temporary EXP tag is a two-step policy action: send a safe but
        # matchup-incapable low-level member in, then immediately return to a
        # validated finisher.  The executor only sees ordinary switch actions;
        # this state carries the continuation across the next turn boundary.
        self._pending_exp_tag_index: int | None = None
        self._pending_exp_finisher_index: int | None = None
        # Battle-local EXP bookkeeping. Participation is scoped to the
        # currently active opposing Pokémon, not to the whole trainer battle:
        # every new opposing Pokémon is a fresh EXP opportunity.
        self._battle_opponent_key: tuple | None = None
        self._battle_opponent_participants: set[int] = set()
        self._battle_opponent_exp_recipients: set[int] = set()
        self._last_party_exp_snapshot: dict[int, int] = {}
        self._battle_support_opponent_key: tuple | None = None
        # Trainer replacement prompts are free actions. Keep their origin
        # separate from ordinary turn-consuming switches so the next turn can
        # re-evaluate the selected Pokémon without tripping any voluntary
        # switch/EXP-rotation guard.
        self._free_switch_origin: tuple[int, tuple | None, str] | None = None
        self._free_switch_history: list[tuple[int, tuple | None]] = []
        self._voluntary_switch_history: list[tuple[int, tuple | None, str]] = []
        # Ordinary voluntary switches are one-way commitments for the current
        # opponent.  This prevents the level-balancing policy from immediately
        # switching back to the Pokémon it just replaced.  Free trainer
        # replacements and emergency exceptions use separate paths below.
        self._voluntary_switch_commitment: tuple[tuple | None, int, int, str] | None = None
        self._switch_validation_bypass_target: int | None = None
        self._trainer_high_risk_signature: tuple | None = None
        self._trainer_high_risk_streak = 0
        self._trainer_high_risk_last_hp: int | None = None

    def _sync_opponent_participation(self, battle_state: BattleState, opponent=None) -> None:
        """Update current-opponent participation at a decision boundary.

        Active-party history is the reliable fallback because a Pokémon that
        switched out before the foe fainted still earns EXP. When the party's
        EXP values have already updated, those deltas are retained as the
        authoritative recipient set as well (covering EXP Share and similar
        cases).
        """

        try:
            opponent = opponent or battle_state.opponent.active_battler
            opponent_key = _battle_pokemon_key(opponent)
            if opponent_key[0] is None and opponent_key[1] is None:
                return
            current_exp = _party_exp_snapshot()
            previous_key = getattr(self, "_battle_opponent_key", None)
            previous_exp = getattr(self, "_last_party_exp_snapshot", {})
            if previous_key is not None:
                for index, total_exp in current_exp.items():
                    if total_exp > previous_exp.get(index, total_exp):
                        self._battle_opponent_exp_recipients.add(index)
            if previous_key != opponent_key:
                self._battle_opponent_key = opponent_key
                self._battle_opponent_participants = set()
                self._battle_opponent_exp_recipients = set()
                self._voluntary_switch_commitment = None
                self._switch_validation_bypass_target = None
                self._trainer_high_risk_signature = None
                self._trainer_high_risk_streak = 0
                self._trainer_high_risk_last_hp = None
                if getattr(self, "_battle_support_opponent_key", None) != opponent_key:
                    self._battle_support_index = None
                    self._battle_support_response_turns = None
                self._battle_support_opponent_key = opponent_key
            active_indices = _active_party_indices(battle_state)
            self._battle_opponent_participants.update(active_indices)
            self._last_party_exp_snapshot = current_exp
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return

    def _advance_to_opponent(self, opponent) -> None:
        """Start a fresh participation window for a known incoming foe."""

        opponent_key = _battle_pokemon_key(opponent)
        if opponent_key[0] is None and opponent_key[1] is None:
            return
        self._battle_opponent_key = opponent_key
        self._battle_opponent_participants = set()
        self._battle_opponent_exp_recipients = set()
        self._last_party_exp_snapshot = _party_exp_snapshot()
        self._battle_support_index = None
        self._battle_support_response_turns = None
        self._battle_support_opponent_key = opponent_key
        self._voluntary_switch_commitment = None
        self._switch_validation_bypass_target = None
        self._trainer_high_risk_signature = None
        self._trainer_high_risk_streak = 0
        self._trainer_high_risk_last_hp = None

    def _remember_battle_support(
        self,
        party_index: int,
        *,
        battle_state: BattleState | None = None,
        response_turns: int | None = None,
    ) -> None:
        """Remember the switch contract that produced a support action."""

        self._battle_support_index = party_index
        self._battle_support_response_turns = response_turns
        opponent_key = getattr(self, "_battle_opponent_key", None)
        if opponent_key is None and battle_state is not None:
            try:
                opponent_key = _battle_pokemon_key(battle_state.opponent.active_battler)
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                opponent_key = None
        self._battle_support_opponent_key = opponent_key
        if battle_state is not None:
            try:
                active_index = battle_state.own_side.active_battler.party_index
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                active_index = party_index
            if active_index != party_index and callable(getattr(self, "_remember_voluntary_switch", None)):
                self._remember_voluntary_switch(party_index, battle_state, "validated support/safety switch")

    def _response_turns_for_switch_validation(
        self, battle_state: BattleState, party_index: int
    ) -> int | None:
        """Return the response horizon used by the pending support switch."""

        if party_index != getattr(self, "_battle_support_index", None):
            return None
        response_turns = getattr(self, "_battle_support_response_turns", None)
        if response_turns is None:
            return None
        support_opponent_key = getattr(self, "_battle_support_opponent_key", None)
        if support_opponent_key is None:
            return response_turns
        try:
            current_opponent_key = _battle_pokemon_key(battle_state.opponent.active_battler)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        return response_turns if current_opponent_key == support_opponent_key else None

    def _participated_against_current_opponent(self, party_index: int) -> bool:
        return party_index in getattr(self, "_battle_opponent_participants", set()) or party_index in getattr(
            self, "_battle_opponent_exp_recipients", set()
        )

    def _current_opponent_participants(self, battle_state: BattleState) -> set[int]:
        """Return observed participants plus the currently active battler."""

        self._sync_opponent_participation(battle_state)
        participants = set(getattr(self, "_battle_opponent_participants", set()))
        try:
            active_index = battle_state.own_side.active_battler.party_index
            if active_index is not None:
                participants.add(active_index)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            pass
        return participants

    @staticmethod
    def _experience_party_label(index: int, pokemon) -> str:
        species = getattr(getattr(pokemon, "species", None), "name", None) or getattr(
            getattr(pokemon, "species", None), "__str__", lambda: "slot"
        )()
        nickname = getattr(pokemon, "nickname", None)
        return f"{species}({nickname})" if nickname and str(nickname).casefold() != str(species).casefold() else f"{species}[{index}]"

    def _experience_projection_detail(
        self,
        battle_state: BattleState,
        participant_indices: set[int] | frozenset[int],
        *,
        opponent=None,
    ) -> str:
        """Return a compact EXP split and cap-headroom diagnostic."""

        distribution = _experience_distribution_for_participants(
            battle_state,
            participant_indices,
            opponent=opponent,
        )
        cap = _active_level_cap()
        if distribution is None:
            return "reward=unknown; split=unknown"
        party = get_party()
        entries = []
        for index, amount in distribution.awards:
            if index < 0 or index >= len(party):
                continue
            headroom = _experience_headroom(index, cap)
            headroom_text = "unknown" if headroom is None else str(headroom)
            entries.append(
                f"{self._experience_party_label(index, party[index])}:gain={amount};remaining_in_cap={headroom_text}"
            )
        cap_text = "unknown" if cap is None else str(cap)
        return (
            f"reward={distribution.base_reward}; participants={distribution.participant_count}; "
            f"exp_share={distribution.exp_share_count}; cap={cap_text}; "
            f"awards=[{', '.join(entries) or 'none'}]"
        )

    def _trace_experience_projection(self, battle_state: BattleState, active_index: int | None) -> None:
        """Log current and one-switch EXP outcomes at a battle decision."""

        try:
            participants = self._current_opponent_participants(battle_state)
            current = self._experience_projection_detail(battle_state, participants)
            alternatives = []
            for index in _eligible_indices(only_non_fainted=True):
                if index == active_index:
                    continue
                hypothetical = participants | {index}
                alternatives.append(
                    f"slot {index}: {self._experience_projection_detail(battle_state, hypothetical)}"
                )
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_EXP_PROJECTION: "
                    f"active={active_index!r} current={current} "
                    f"alternatives=[{' | '.join(alternatives) or 'none'}]"
                ),
                trace=True,
                prefix="NUZLOCKE_EXP_PROJECTION:",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            return

    def _remember_free_trainer_switch(self, party_index: int, opponent, reason: str) -> None:
        """Record a free trainer switch independently of voluntary switches."""

        opponent_key = _battle_pokemon_key(opponent)
        self._free_switch_origin = (party_index, opponent_key, reason)
        history = getattr(self, "_free_switch_history", None)
        if not isinstance(history, list):
            history = []
            self._free_switch_history = history
        history.append((party_index, opponent_key))
        diagnostic_print(
            lambda: (
                "NUZLOCKE_SWITCH_TRACKING: "
                f"kind='free_trainer_switch' party_index={party_index!r} opponent={opponent_key!r} "
                f"reason={reason!r} voluntary_switch_count={len(getattr(self, '_voluntary_switch_history', ()))!r}"
            ),
            trace=True,
            prefix="NUZLOCKE_SWITCH_TRACKING:",
        )

    def _remember_voluntary_switch(self, party_index: int, battle_state: BattleState, reason: str) -> None:
        try:
            opponent_key = _battle_pokemon_key(battle_state.opponent.active_battler)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            opponent_key = None
        try:
            previous_index = battle_state.own_side.active_battler.party_index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            previous_index = None
        history = getattr(self, "_voluntary_switch_history", None)
        if not isinstance(history, list):
            history = []
            self._voluntary_switch_history = history
        history.append((party_index, opponent_key, reason))
        if previous_index is not None and previous_index != party_index:
            self._voluntary_switch_commitment = (opponent_key, previous_index, party_index, reason)
            self._switch_validation_bypass_target = party_index
        diagnostic_print(
            lambda: (
                "NUZLOCKE_SWITCH_TRACKING: "
                f"kind='voluntary_switch' party_index={party_index!r} opponent={opponent_key!r} "
                f"reason={reason!r} free_switch_count={len(getattr(self, '_free_switch_history', ()))!r}"
            ),
            trace=True,
            prefix="NUZLOCKE_SWITCH_TRACKING:",
        )

    def _committed_prior_index(self, battle_state: BattleState) -> int | None:
        """Return the ordinary switch target that is temporarily committed."""

        commitment = getattr(self, "_voluntary_switch_commitment", None)
        if commitment is None:
            return None
        try:
            opponent_key = _battle_pokemon_key(battle_state.opponent.active_battler)
            active_index = battle_state.own_side.active_battler.party_index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        if commitment[0] != opponent_key or commitment[2] != active_index:
            return None
        return commitment[1]

    def _commitment_reversal_is_necessary(self, battle_state: BattleState, active_index: int | None) -> bool:
        """Allow a reversal only for the explicitly approved emergencies."""

        if active_index is None:
            return False
        try:
            active = battle_state.own_side.active_battler
            if active.party_index != active_index:
                return False
            if BattleStrategyUtil.is_poisoned(active):
                return True
            total_hp = getattr(active, "total_hp", 0)
            if total_hp and active.current_hp / total_hp <= 0.25:
                return True
            response_damage, critical_response_damage = _opponent_response_damage(
                BattleStrategyUtil(battle_state),
                battle_state.opponent.active_battler,
                active,
                battle_state,
            )
            return (
                response_damage is not None and response_damage >= active.current_hp
            ) or (
                critical_response_damage is not None and critical_response_damage >= active.current_hp
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError, ZeroDivisionError):
            return False

    def _ordinary_switch_guard_blocks(self, battle_state: BattleState, party_index: int) -> bool:
        """Return whether a normal switch would immediately reverse itself."""

        if party_index != self._committed_prior_index(battle_state):
            return False
        if party_index == getattr(self, "_switch_validation_bypass_target", None):
            return False
        return not self._commitment_reversal_is_necessary(
            battle_state,
            getattr(battle_state.own_side.active_battler, "party_index", None),
        )

    def notify_pokemon_fainted(self) -> None:
        """Clear a voluntary commitment when a battler leaves the field."""

        self._voluntary_switch_commitment = None
        self._switch_validation_bypass_target = None
        self._battle_support_index = None
        self._battle_support_response_turns = None
        self._trainer_high_risk_signature = None
        self._trainer_high_risk_streak = 0
        self._trainer_high_risk_last_hp = None

    def notify_battle_ended(self, _outcome=None) -> None:
        """Clear all battle-local switch and risk state at battle teardown."""

        self.notify_pokemon_fainted()
        self._free_switch_origin = None
        self._pending_exp_tag_index = None
        self._pending_exp_finisher_index = None

    def _recheck_after_free_switch(self, battle_state: BattleState, active_index: int | None):
        """Re-evaluate a free replacement before spending its first turn.

        The optional trainer prompt is a zero-cost switch.  A candidate can
        pass that prompt's one-response screen and still be a poor active
        lead once the ROM has installed the real battler.  Consume the origin
        once, then permit one ordinary safety switch if the newly active
        Pokémon cannot safely resolve the incoming foe.
        """

        origin = getattr(self, "_free_switch_origin", None)
        if origin is None or active_index != origin[0]:
            return None
        self._free_switch_origin = None
        try:
            current_opponent_key = _battle_pokemon_key(battle_state.opponent.active_battler)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            current_opponent_key = None
        if origin[1] is not None and current_opponent_key != origin[1]:
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_FREE_SWITCH_RECHECK: "
                    f"party_index={active_index!r} result='skip_opponent_changed' "
                    f"origin_opponent={origin[1]!r} current_opponent={current_opponent_key!r}"
                ),
                trace=True,
                prefix="NUZLOCKE_FREE_SWITCH_RECHECK:",
            )
            return None
        try:
            util = BattleStrategyUtil(battle_state)
            active = battle_state.own_side.active_battler
            opponent = battle_state.opponent.active_battler
            move_index = util.get_strongest_move_against(active, opponent)
            damage = (
                util.calculate_move_damage_range(active.moves[move_index].move, active, opponent)
                if move_index is not None
                else None
            )
            response_moves = _opponent_move_names_for_safety(battle_state)
            if damage is not None and getattr(damage, "min", 0) >= opponent.current_hp:
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_FREE_SWITCH_RECHECK: "
                        f"party_index={active_index!r} result='keep' reason='guaranteed_ko'"
                    ),
                    trace=True,
                    prefix="NUZLOCKE_FREE_SWITCH_RECHECK:",
                )
                return None
            response = (
                _response_risk_for_line(
                    util,
                    opponent,
                    active,
                    response_moves,
                    damage_min=getattr(damage, "min", 1) if damage is not None else 1,
                    damage_max=getattr(damage, "max", 1) if damage is not None else 1,
                    response_turns=_FREE_TRAINER_SWITCH_RESPONSE_TURNS,
                    response_timing=_RESPONSE_TIMING_MID_TURN_SWITCH,
                )
                if response_moves and damage is not None
                else None
            )
            if response is not None and response[3] >= _MIN_ACCEPTABLE_SURVIVAL_PROBABILITY:
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_FREE_SWITCH_RECHECK: "
                        f"party_index={active_index!r} result='keep' survival_probability={response[3]!r}"
                    ),
                    trace=True,
                    prefix="NUZLOCKE_FREE_SWITCH_RECHECK:",
                )
                return None
            target = self._select_switch_target(
                battle_state,
                active_index,
                util=util,
                require_strict_safe=True,
                prefer_lowest_level=False,
                prefer_fresh_for_opponent=False,
                response_turns=_EMERGENCY_SWITCH_RESPONSE_TURNS,
                ignore_voluntary_commitment=True,
            )
            if target is None:
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_FREE_SWITCH_RECHECK: "
                        f"party_index={active_index!r} result='keep_no_safe_target' "
                        f"response_moves={response_moves!r} survival={response[3] if response else None!r}"
                    ),
                    trace=True,
                    prefix="NUZLOCKE_FREE_SWITCH_RECHECK:",
                )
                return None
            context.battle_decision_source = "NUZLOCKE FREE SWITCH RECHECK"
            context.battle_decision_detail = (
                f"free trainer switch origin={origin[2]!r}; re-evaluated before first turn; "
                f"active slot {active_index} failed the response safety check; switching to validated slot {target}"
            )
            context.message = f"NUZLOCKE SAFETY: replacing unsafe free switch with party slot {target}"
            self._remember_battle_support(target, battle_state=battle_state, response_turns=_EMERGENCY_SWITCH_RESPONSE_TURNS)
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_FREE_SWITCH_RECHECK: "
                    f"party_index={active_index!r} result='switch' target={target!r} "
                    f"survival={response[3] if response else None!r}"
                ),
                trace=True,
                prefix="NUZLOCKE_FREE_SWITCH_RECHECK:",
            )
            return TurnAction.rotate_lead(target)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError) as error:
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_FREE_SWITCH_RECHECK: "
                    f"party_index={active_index!r} result='error' error={error!r}"
                ),
                trace=True,
                prefix="NUZLOCKE_FREE_SWITCH_RECHECK:",
            )
            return None

    def _field_lead_candidate(self, field_context: CampaignFieldLeadContext | None = None) -> int | None:
        """Choose a training lead for a local or campaign-provided envelope.

        ``None`` retains the historical local-map behavior for legacy callers
        and focused strategy tests. Campaign Progression supplies an explicit
        :class:`CampaignFieldLeadContext`, which may describe the route toward
        the mounted objective rather than only the current map.
        """

        if field_context is None:
            threats = _field_threats_for_current_map()
            level_ceiling = _field_encounter_level_ceiling()
        else:
            threats = tuple((threat.species, threat.weight) for threat in field_context.threats)
            level_ceiling = field_context.encounter_level_ceiling
        if not threats:
            return None
        candidates = []
        for index in _eligible_indices(only_non_fainted=True):
            try:
                pokemon = get_party()[index]
                if not self.pokemon_can_battle(pokemon):
                    continue
                profile = _field_coverage_profile(pokemon, threats)
                current_hp = getattr(pokemon, "current_hp", 0)
                total_hp = getattr(pokemon, "total_hp", 0)
                hp_ratio = current_hp / total_hp if total_hp else 0.0
                coverage, covered_weight, super_effective_weight = profile or (0.0, 0, 0)
                level = getattr(pokemon, "level", 0)
                if not isinstance(level, int):
                    level = 0
                # Coverage and health are safety/field-fit criteria. Level and
                # EXP become the primary criteria only while a party member is
                # below the local wild-level ceiling. A weak lead can safely
                # tag an encounter and switch on turn one; it does not need a
                # damaging move merely to be eligible for field rotation.
                candidates.append(
                    (
                        index,
                        level,
                        coverage,
                        covered_weight,
                        super_effective_weight,
                        hp_ratio,
                    )
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                continue
        if not candidates:
            return None
        training_decision = choose_training_lead(
            TrainingCandidate(
                party_index=candidate[0],
                level=candidate[1],
                target_level=level_ceiling,
                hp_ratio=candidate[5],
                matchup_score=(candidate[2] + candidate[3] + candidate[4]),
            )
            for candidate in candidates
        )
        if training_decision.party_index is not None and level_ceiling is not None:
            return training_decision.party_index
        underleveled = (
            tuple(candidate for candidate in candidates if candidate[1] < level_ceiling)
            if level_ceiling is not None
            else ()
        )
        if underleveled:
            return min(
                underleveled,
                key=lambda candidate: (_training_priority(candidate[0]), -candidate[5], candidate[0]),
            )[0]
        return max(
            candidates,
            key=lambda candidate: (
                candidate[2],
                candidate[3],
                candidate[4],
                candidate[5],
                -_training_priority(candidate[0])[0],
                -_training_priority(candidate[0])[1],
                -candidate[0],
            ),
        )[0]

    def choose_field_lead(self, field_context: CampaignFieldLeadContext) -> CampaignFieldLeadDecision:
        """Choose the campaign overworld lead from explicit route context.

        This is the campaign-only replacement for using
        ``choose_new_lead_after_battle`` as an implicit field policy. The
        battle strategy still owns party safety and EXP priorities, while the
        campaign adapter owns objective and route interpretation.
        """

        field_candidate = self._field_lead_candidate(field_context)
        index = field_candidate
        if index is None:
            index = _lowest_level_index(only_non_fainted=True)
        candidate_level = getattr(get_party()[index], "level", None) if index is not None else None
        if field_candidate is not None and field_context.encounter_level_ceiling is not None and (
            candidate_level is not None and candidate_level < field_context.encounter_level_ceiling
        ):
            reason = "lowest-level eligible member below route encounter-level ceiling"
        elif field_candidate is not None:
            reason = "best route-envelope coverage after party reached route encounter-level ceiling"
        elif field_context.has_encounter_envelope:
            reason = "no route-envelope candidate; lowest living eligible level"
        else:
            reason = "no usable encounter envelope toward mounted objective; lowest living eligible level"

        try:
            current_lead = get_party().first_non_fainted.index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            current_lead = 0
        selected = index if index is not None and index != current_lead else None
        diagnostic_print(
            lambda: (
                "NUZLOCKE_CAMPAIGN_FIELD_LEAD: "
                f"objective={field_context.objective_id!r} current_map={field_context.current_map!r} "
                f"objective_map={field_context.objective_map!r} route_maps={field_context.route_maps!r} "
                f"envelope_source={field_context.source!r} candidate={index!r} "
                f"current_lead={current_lead!r} selected={selected!r} reason={reason!r}"
            ),
            trace=True,
        )
        if selected is not None:
            context.battle_decision_source = "NUZLOCKE CAMPAIGN FIELD LEAD"
            context.battle_decision_detail = (
                f"selected party slot {selected} using {reason}; "
                f"objective={field_context.objective_id!r}; source={field_context.source!r}"
            )
            context.message = (
                f"LEVELING: campaign field lead slot {selected} ({reason})"
                f"{_preparation_target_suffix()}"
            )
        return CampaignFieldLeadDecision(
            selected_index=selected,
            reason=reason,
            objective_id=field_context.objective_id,
            envelope_source=field_context.source,
            route_maps=field_context.route_maps,
        )

    def pokemon_can_battle(self, pokemon) -> bool:
        return _is_eligible_pokemon(pokemon) and super().pokemon_can_battle(pokemon)

    def party_can_battle(self) -> bool:
        # A fainted/dead low-level member must not make a higher-level living
        # member appear unusable.  The lowest living eligible member is the
        # preferred lead, but any legal battle-capable member keeps the party
        # in the battle loop.
        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

    def is_switch_target_valid(self, battle_state: BattleState, party_index: int) -> bool:
        """Allow only ordinary matchup switches or a live EXP-tag plan."""

        if party_index in _active_party_indices(battle_state):
            return False
        if self._ordinary_switch_guard_blocks(battle_state, party_index):
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_SWITCH_GUARD: "
                    f"party_index={party_index!r} result='reject_immediate_reversal'"
                ),
                trace=True,
                prefix="NUZLOCKE_SWITCH_GUARD:",
            )
            return False
        if party_index != getattr(self, "_pending_exp_tag_index", None):
            if party_index not in _eligible_indices(only_non_fainted=True):
                return False
            active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
            if active_index is None:
                return False
            response_turns = self._response_turns_for_switch_validation(battle_state, party_index)
            profile = self._switch_candidate_risk(
                battle_state,
                party_index,
                active_index=active_index,
                response_turns=response_turns,
            )
            return (
                profile is not None
                and (profile.strict_safe or profile.probabilistically_safe)
                and _participants_remain_cap_legal(battle_state, {active_index, party_index})
            )
        active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
        if active_index == party_index:
            return False
        if party_index not in _eligible_indices(only_non_fainted=True):
            return False
        try:
            util = BattleStrategyUtil(battle_state)
            opponent = battle_state.opponent.active_battler
            known_response_moves = _known_opponent_move_names()
            if not util.can_switch() or not known_response_moves:
                return False
            tag = get_party()[party_index]
            if not self.pokemon_can_battle(tag) or not self._survives_visible_response(
                util, opponent, tag, known_response_moves
            ):
                return False
            finisher = self._battle_finisher_target(battle_state, excluded={party_index})
            return finisher == getattr(self, "_pending_exp_finisher_index", None) and _participants_remain_cap_legal(
                battle_state, {active_index, party_index, finisher}
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return False

    def switch_target_validation_reason(self, battle_state: BattleState, party_index: int) -> str:
        """Explain the Nuzlocke safety predicate used for a switch target."""

        try:
            if party_index in _active_party_indices(battle_state):
                return "target is already active"
            if self._ordinary_switch_guard_blocks(battle_state, party_index):
                return "ordinary switch guard rejected an immediate reversal"
            eligible = set(_eligible_indices(only_non_fainted=True))
            if party_index not in eligible:
                party = get_party()
                target = party[party_index] if 0 <= party_index < len(party) else None
                if target is None:
                    return "party slot is outside the observed party"
                if getattr(target, "is_egg", False):
                    return "target is an egg"
                if getattr(target, "current_hp", 0) <= 0:
                    return "target has 0 HP"
                return "target is excluded by the Nuzlocke eligibility projection"

            active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
            if active_index is None:
                return "active battler is unavailable"
            if party_index != getattr(self, "_pending_exp_tag_index", None):
                response_turns = self._response_turns_for_switch_validation(battle_state, party_index)
                profile = self._switch_candidate_risk(
                    battle_state,
                    party_index,
                    active_index=active_index,
                    response_turns=response_turns,
                )
                if profile is None:
                    return "no switch risk profile could be modeled"
                if not profile.strict_safe and not profile.probabilistically_safe:
                    return getattr(profile, "reason", None) or "switch survival or damage envelope was unsafe"
                if not _participants_remain_cap_legal(battle_state, {active_index, party_index}):
                    return "switch would violate the remaining cap-legal participant set"
                return "validated"

            util = BattleStrategyUtil(battle_state)
            opponent = battle_state.opponent.active_battler
            known_response_moves = _known_opponent_move_names()
            if not util.can_switch():
                return "the battle state does not currently permit switching"
            if not known_response_moves:
                return "the opponent response is unknown for the pending EXP-tag safety check"
            tag = get_party()[party_index]
            if not self.pokemon_can_battle(tag):
                return "pending EXP-tag target cannot battle under the current party policy"
            if not self._survives_visible_response(util, opponent, tag, known_response_moves):
                return "pending EXP-tag target does not survive the visible response"
            finisher = self._battle_finisher_target(battle_state, excluded={party_index})
            if finisher != getattr(self, "_pending_exp_finisher_index", None):
                return "pending EXP-tag finisher no longer matches the modeled battle"
            if not _participants_remain_cap_legal(battle_state, {active_index, party_index, finisher}):
                return "pending EXP-tag switch would violate the remaining cap-legal participant set"
            return "validated"
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            return "Nuzlocke switch validation could not read the current battle facts"

    def get_valid_switch_targets(self, battle_state: BattleState) -> tuple[int, ...]:
        """Filter executor fallbacks through the Nuzlocke death projection."""

        eligible = set(_eligible_indices(only_non_fainted=True))
        active_indices = _active_party_indices(battle_state)
        active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
        if active_index is None:
            return ()
        valid = []
        for index in super().get_valid_switch_targets(battle_state):
            if (
                index in active_indices
                or index not in eligible
                or not _participants_remain_cap_legal(battle_state, {active_index, index})
                or self._ordinary_switch_guard_blocks(battle_state, index)
            ):
                continue
            response_turns = self._response_turns_for_switch_validation(battle_state, index)
            profile = self._switch_candidate_risk(
                battle_state,
                index,
                active_index=active_index,
                response_turns=response_turns,
            )
            if profile is not None and (profile.strict_safe or profile.probabilistically_safe):
                valid.append(index)
        return tuple(valid)

    def _risk_aware_emergency_action(
        self, battle_state: BattleState, active_index: int | None
    ) -> tuple[TurnAction, int | None] | None:
        """Return a switch/escape when continuing would expose an avoidable loss.

        The ordinary planner is deliberately allowed to choose a best-
        available attack when response facts are incomplete. That fallback is
        not sufficient for a Nuzlocke lead at critical HP or one already
        poisoned: an uncertain attack can spend the last safe turn before the
        field-poison tick or an opposing response. A replacement must still
        have a damaging matchup and meet either the deterministic or the
        probability-weighted survival threshold; this is therefore a risk
        gate, not a duplicate-encounter rule.
        """

        try:
            party = get_party()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        if active_index is None or active_index < 0 or active_index >= len(party):
            return None
        active = battle_state.own_side.active_battler
        try:
            active_ratio = active.current_hp / active.total_hp if active.total_hp > 0 else 0.0
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None
        poisoned = BattleStrategyUtil.is_poisoned(active)
        util = BattleStrategyUtil(battle_state)
        active_can_resolve_before_response = _active_has_guaranteed_ko_before_response(
            util,
            active,
            battle_state.opponent.active_battler,
            battle_state,
        )
        response_damage, critical_response_damage = _opponent_response_damage(
            util,
            battle_state.opponent.active_battler,
            active,
            battle_state,
        )
        active_response_line = _active_response_risk_for_line(
            util,
            active,
            battle_state.opponent.active_battler,
            battle_state,
        )
        poison_threat = _known_opponent_can_poison(battle_state)
        # A known poisoning move is relevant before the active member is at
        # the critical threshold, but not enough by itself to force a switch
        # from a healthy, useful matchup.
        critical = active_ratio <= 0.25
        vulnerable_to_poison = poison_threat and active_ratio <= 0.5
        # A normal maximum-damage KO is an immediate safety trigger.  A
        # critical-only KO is retained as a reason to leave a critically weak
        # lead, but is not allowed to make a healthy Pokémon panic-switch: the
        # modeled critical chance is only 1/16 per damaging response.
        response_can_ko = response_damage is not None and response_damage >= active.current_hp
        critical_only_ko = (
            response_damage is not None
            and response_damage < active.current_hp
            and critical_response_damage is not None
            and critical_response_damage >= active.current_hp
            and active_ratio <= 0.25
        )
        multi_turn_exposure = (
            active_response_line is not None
            and active_response_line[3] < _MIN_ACCEPTABLE_SURVIVAL_PROBABILITY
        )
        if not (
            critical
            or poisoned
            or vulnerable_to_poison
            or response_can_ko
            or critical_only_ko
            or multi_turn_exposure
        ) or active_can_resolve_before_response:
            return None

        # Keep the strict Nuzlocke safety gate separate from the comparative
        # battle recommendation.  The planner's report tells us what could be
        # selected and what would be the best available action even when no
        # autonomous action clears the safety proof.
        planner_decision = None
        try:
            planner_decision = plan_battle_state(battle_state, default_battle_recorder.knowledge)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            planner_decision = None

        best_available = getattr(planner_decision, "best_available_candidate", None)
        best_safe = getattr(planner_decision, "best_safe_candidate", None)
        evaluation_summary = (
            f"best_safe={getattr(best_safe, 'label', 'none')!r}; "
            f"best_available={getattr(best_available, 'label', 'none')!r}"
        )

        stochastic_result = getattr(planner_decision, "stochastic_result", None)
        if stochastic_result is not None:
            # The caller will execute the exact root action selected by the
            # bounded search. In particular, do not replace an incomplete
            # search's item root with the legacy strategy-local healer.
            return None

        # If normalized response facts are unavailable, BattlePlanner may
        # still establish a conservative one-turn item action. Keep that
        # fallback in the shared planner contract rather than selecting the
        # item here from raw inventory facts.
        if getattr(planner_decision, "action", None) is PlannerAction.UseItem:
            target = getattr(planner_decision, "target", None)
            if isinstance(target, tuple) and len(target) == 2:
                context.battle_decision_source = "NUZLOCKE PLANNER ITEM FALLBACK"
                context.battle_decision_detail = (
                    f"expectiminimax response facts unavailable; using shared planner item action; {evaluation_summary}"
                )
                return TurnAction.use_item_on(*target)
            if target is not None:
                context.battle_decision_source = "NUZLOCKE PLANNER ITEM FALLBACK"
                context.battle_decision_detail = (
                    f"expectiminimax response facts unavailable; using shared planner item action; {evaluation_summary}"
                )
                return TurnAction.use_item(target)

        # The emergency gate runs before the ordinary planner, but it must
        # not override a planner-proven safe attack line.  This is especially
        # important for a poisoned or low-HP lead: if the attack is already
        # safe, spending a turn on an item only gives the opponent another
        # response opportunity.
        if (
            not poisoned
            and getattr(planner_decision, "action", None) is PlannerAction.UseMove
            and getattr(planner_decision, "classification", None) is PlannerDecisionClass.SAFE
        ):
            context.battle_decision_source = "NUZLOCKE RISK DEFER"
            context.battle_decision_detail = (
                f"planner found a safe attack line; emergency heal deferred; {evaluation_summary}"
            )
            return None

        try:
            if not util.can_switch():
                switch_allowed = False
            else:
                switch_allowed = True
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            util = None
            switch_allowed = False

        if switch_allowed and util is not None:
            # A support marker prevents harmless immediate reversals, but it
            # must not turn an unsafe multi-turn line into a forced faint
            # when a validated replacement exists.
            # A known poison move does not automatically make a healthy,
            # usable lead expend a switch. If the current member can absorb
            # the complete modeled response and a battle item improves its
            # position, preserve the matchup and heal first. The switch
            # remains preferred when the active attack line is already below
            # the normal survival threshold or when the response can KO this
            # turn.
            poison_healing_action = None
            if not getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False)) and vulnerable_to_poison:
                poison_healing_action = _safe_battle_healing_action(
                    active,
                    response_damage=response_damage,
                    critical_response_damage=critical_response_damage,
                    active_index=active_index,
                    require_safe=False,
                )
                required_response = max(response_damage or 0, critical_response_damage or response_damage or 0)
                active_can_take_response = response_damage is not None and active.current_hp > required_response
                attack_line_is_acceptable = (
                    active_response_line is None
                    or active_response_line[3] >= _MIN_ACCEPTABLE_SURVIVAL_PROBABILITY
                )
                if (
                    poison_healing_action is not None
                    and active_can_take_response
                    and attack_line_is_acceptable
                ):
                    self._remember_battle_support(active_index, battle_state=battle_state)
                    context.battle_decision_source = "NUZLOCKE POISON PREVENTION HEAL"
                    context.battle_decision_detail = (
                        "known poison threat; active survives the modeled response; healing before switching "
                        f"preserves the current matchup; {evaluation_summary}"
                    )
                    context.message = "NUZLOCKE SAFETY: healing before a poison-risk switch"
                    return poison_healing_action

            target = self._risk_aware_switch_target(battle_state, active_index, util)

            # A support switch is a commitment for the current opponent.  If
            # that committed Pokémon is now critically weak, spend the
            # available battle item before replacing it again.  Without this
            # boundary the safety selector and the high-risk fallback can
            # alternate between two imperfect replacements every turn.
            if (
                getattr(self, "_battle_support_index", None) == active_index
                and active_ratio <= 0.35
                and target is None
            ):
                support_healing_action = _safe_battle_healing_action(
                    active,
                    response_damage=response_damage,
                    critical_response_damage=critical_response_damage,
                    active_index=active_index,
                    require_safe=False,
                )
                if support_healing_action is not None:
                    self._remember_battle_support(active_index, battle_state=battle_state)
                    context.battle_decision_source = "NUZLOCKE SUPPORT HEAL"
                    context.battle_decision_detail = (
                        "support switch is committed for this opponent; healing the committed active "
                        "before considering another voluntary switch"
                    )
                    context.message = "NUZLOCKE SAFETY: healing the committed support Pokémon"
                    return support_healing_action
            if target is not None:
                reason = (
                    "poisoned active"
                    if poisoned
                    else "critical active HP"
                    if critical
                    else "known poison threat"
                    if vulnerable_to_poison
                    else "opponent response can KO active"
                    if response_can_ko or critical_only_ko
                    else "active attack line has low multi-turn survival"
                )
                context.battle_decision_source = "NUZLOCKE RISK SAFETY"
                context.battle_decision_detail = (
                    f"{reason}; switched to validated damaging party slot {target}; {evaluation_summary}; "
                    f"exp={self._experience_projection_detail(battle_state, self._current_opponent_participants(battle_state) | {target})}"
                )
                context.message = f"NUZLOCKE SAFETY: switching to party slot {target} ({reason})"
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_RISK_SAFETY: "
                        f"active={active_index!r} poisoned={poisoned!r} poison_threat={poison_threat!r} "
                        f"hp_ratio={active_ratio!r} multi_turn_survival="
                        f"{active_response_line[3] if active_response_line is not None else None!r} "
                        f"action='switch' target={target!r}"
                    ),
                    trace=True,
                )
                # Preserve the safety replacement across the next turn.  The
                # ordinary EXP-rotation policy must not immediately put the
                # threatened Pokémon back in front just because it is lower
                # level.
                self._remember_battle_support(
                    target,
                    battle_state=battle_state,
                    response_turns=_EMERGENCY_SWITCH_RESPONSE_TURNS,
                )
                return TurnAction.rotate_lead(target)

        # A trainer battle cannot be escaped, but an available battle item can
        # still turn a known lethal response into a survivable line. Use only
        # an item whose post-heal HP clears the critical-hit ceiling; an item
        # is not a safety action when the response facts are unavailable.
        if not getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False)):
            healing_action = _safe_battle_healing_action(
                active,
                response_damage=response_damage,
                critical_response_damage=critical_response_damage,
                active_index=active_index,
            )
            if healing_action is not None:
                # Healing consumes this turn. Keep the healed active Pokémon
                # in battle for the current opponent so the ordinary level
                # balancing policy cannot immediately undo the recovery with
                # a lower-level rotation on the next decision.
                self._remember_battle_support(active_index, battle_state=battle_state)
                context.battle_decision_source = "NUZLOCKE RISK HEAL"
                context.battle_decision_detail = (
                    "battle item clears the known response damage ceiling; "
                    "keeping healed active in for this opponent"
                )
                context.message = "NUZLOCKE SAFETY: healing before an avoidable faint"
                return healing_action

        if getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False)):
            try:
                escape = BattleStrategyUtil(battle_state).get_best_escape_method()
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                escape = None
            if escape is not None:
                # A legal capture encounter gets one explicit capture-policy
                # turn before the generic wild-risk escape is allowed. This
                # matters when the lead cannot support a multi-turn attack
                # line but can safely throw a ball or make one modeled
                # weakening attempt. The capture evaluator already prices
                # target KO and party-faint risk; fleeing here would discard
                # that decision before it ever runs.
                capture_target = bool(getattr(battle_state, "nuzlocke_capture_target", False))
                ball_available = False
                get_best_ball = getattr(self, "_get_best_poke_ball", None)
                if capture_target and callable(get_best_ball):
                    try:
                        ball_available = get_best_ball(battle_state) is not None
                    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                        ball_available = False
                capture_one_response_safe = (
                    capture_target
                    and ball_available
                    and response_damage is not None
                    and critical_response_damage is not None
                    and critical_response_damage < active.current_hp
                )
                if capture_one_response_safe:
                    context.battle_decision_source = "CAPTURE POLICY DEFER"
                    context.battle_decision_detail = (
                        "capture-owned encounter has a legal ball and the active Pokémon survives the modeled "
                        "normal and critical one-response ceiling; defer to capture evaluation before fleeing"
                    )
                    context.message = "CAPTURE POLICY: evaluating a safe ball or weakening action"
                    return None
                context.battle_decision_source = "NUZLOCKE RISK ESCAPE"
                context.battle_decision_detail = (
                    "active Pokémon is critically exposed and no validated safe switch was available"
                )
                context.message = "NUZLOCKE SAFETY: escaping to avoid an avoidable faint"
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_RISK_SAFETY: "
                        f"active={active_index!r} poisoned={poisoned!r} poison_threat={poison_threat!r} "
                        f"hp_ratio={active_ratio!r} action='escape'"
                    ),
                    trace=True,
                )
                return escape
        # Wild encounters retain the strict stop/flee behavior below. A
        # trainer battle cannot be escaped, though, and the ordinary planner
        # already has a probability-ranked BEST_AVAILABLE fallback. Yield to
        # that path instead of forcing Manual whenever no replacement clears
        # the strict safety gate.
        if not getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False)):
            context.battle_decision_source = "NUZLOCKE RISK DEFER"
            context.battle_decision_detail = (
                f"no validated trainer switch or heal; defer to probabilistic planner; {evaluation_summary}"
            )
            context.message = "NUZLOCKE RISK: no strictly safe switch; evaluating the best available trainer action"
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_ACTION_EVALUATION: defer trainer battle to planner; "
                    f"{evaluation_summary}"
                ),
                trace=True,
            )
            return None

        # There is no safe autonomous action left in a wild encounter. Pausing
        # is safer than allowing the inherited BEST_AVAILABLE attack to
        # knowingly spend the active Pokémon's last survivable turn.
        context.battle_decision_source = "NUZLOCKE RISK STOP"
        best_reason = getattr(best_available, "reason", None)
        if best_available is None:
            best_detail = "no autonomous action was available"
        else:
            best_detail = f"best available is {best_available.label} but it is not proven safe: {best_reason}"
        context.battle_decision_detail = (
            f"no validated Nuzlocke-safe action was available; {best_detail}; {evaluation_summary}"
        )
        context.message = (
            f"NUZLOCKE SAFETY: no safe autonomous action; {best_detail}; switching to manual mode"
        )
        diagnostic_print(
            lambda: (
                "NUZLOCKE_ACTION_EVALUATION: no safe autonomous action; "
                f"{best_detail}; {evaluation_summary}"
            ),
            trace=True,
        )
        return TurnAction.switch_to_manual()

    def _risk_aware_switch_target(
        self, battle_state: BattleState, active_index: int, util: BattleStrategyUtil
    ) -> int | None:
        """Choose one validated emergency replacement.

        Emergency switches and ordinary leveling switches intentionally use
        the same evaluator. The emergency caller uses a bounded two-response
        horizon: a replacement must be able to absorb the switch-in response
        and one follow-up response while dealing useful damage. A member that
        cannot damage the opponent remains invalid.
        """

        return self._select_switch_target(
            battle_state,
            active_index,
            util=util,
            require_strict_safe=True,
            prefer_lowest_level=False,
            prefer_fresh_for_opponent=False,
            response_turns=_EMERGENCY_SWITCH_RESPONSE_TURNS,
        )

    def _least_bad_trainer_switch_target(
        self, battle_state: BattleState, active_index: int | None
    ) -> tuple[int, _SwitchRisk] | None:
        """Return the best imperfect damaging switch for a bounded fallback."""

        if active_index is None:
            return None
        target = self._select_switch_target(
            battle_state,
            active_index,
            require_strict_safe=False,
            prefer_lowest_level=False,
            response_turns=_FREE_TRAINER_SWITCH_RESPONSE_TURNS,
            allow_imperfect=True,
        )
        if target is None:
            return None
        profile = self._switch_candidate_risk(
            battle_state,
            target,
            active_index=active_index,
            response_turns=_FREE_TRAINER_SWITCH_RESPONSE_TURNS,
        )
        return (target, profile) if profile is not None else None

    def _bounded_trainer_high_risk_fallback(
        self,
        battle_state: BattleState,
        action: tuple[TurnAction, object],
        planner_report,
    ) -> tuple[TurnAction, object] | None:
        """Compare a bounded trainer fallback after repeated risky attacks.

        The ordinary planner remains the first choice.  This path only runs
        after repeated high-risk attack decisions for the same active and
        opponent, or after the active HP worsens.  It compares a one-turn
        healing action with the least-bad damaging switch and then clears its
        streak so it cannot create a switch loop of its own.
        """

        if not getattr(battle_state, "is_trainer_battle", False):
            return None
        if action[0] is not TurnAction.UseMove:
            return None
        classification = getattr(planner_report, "classification", None)
        if classification is PlannerDecisionClass.SAFE:
            self._trainer_high_risk_streak = 0
            return None
        risk = getattr(getattr(planner_report, "risk", None), "value", None)
        if risk not in {"high", "critical"}:
            self._trainer_high_risk_streak = 0
            return None
        active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
        opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
        active_index = getattr(active, "party_index", None)
        opponent_key = _battle_pokemon_key(opponent)
        signature = (opponent_key, active_index)
        current_hp = getattr(active, "current_hp", None)
        previous_hp = self._trainer_high_risk_last_hp
        if signature != self._trainer_high_risk_signature:
            self._trainer_high_risk_signature = signature
            self._trainer_high_risk_streak = 0
            previous_hp = None
        self._trainer_high_risk_streak += 1
        self._trainer_high_risk_last_hp = current_hp
        worsening_hp = (
            previous_hp is not None
            and current_hp is not None
            and current_hp < previous_hp
        )
        if self._trainer_high_risk_streak < 2 and not worsening_hp:
            return None

        util = BattleStrategyUtil(battle_state)
        response_damage, critical_response_damage = _opponent_response_damage(
            util, opponent, active, battle_state
        )
        healing_action = _safe_battle_healing_action(
            active,
            response_damage=response_damage,
            critical_response_damage=critical_response_damage,
            active_index=active_index,
            require_safe=False,
        )
        support_active = getattr(self, "_battle_support_index", None) == active_index
        if support_active:
            try:
                active_ratio = active.current_hp / active.total_hp if active.total_hp > 0 else 0.0
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                active_ratio = 0.0
            if active_ratio <= 0.35 and healing_action is not None:
                self._remember_battle_support(active_index, battle_state=battle_state)
                context.battle_decision_source = "NUZLOCKE SUPPORT HEAL"
                context.battle_decision_detail = (
                    "support switch is committed for this opponent; bounded fallback selected healing "
                    "before another imperfect switch"
                )
                context.message = "NUZLOCKE SAFETY: healing the committed support Pokémon"
                return healing_action
            if active_ratio > 0.25:
                # Let the committed support Pokémon continue its attack line;
                # the normal switch fallback is not allowed to undo the
                # validated safety decision merely because the planner remains
                # uncertain about the opponent's hidden move set.
                return None
        switch_option = self._least_bad_trainer_switch_target(battle_state, active_index)
        healing_score = -1.0
        if healing_action is not None:
            item = healing_action[1][0]
            recovery = _recovery_item(item, getattr(item, "quantity", 1))
            healed_hp = (
                active.total_hp
                if recovery.full_heal
                else min(active.total_hp, active.current_hp + recovery.amount_healed)
            )
            if critical_response_damage is None:
                healing_score = 0.35 + healed_hp / max(1, active.total_hp) * 0.25
            elif healed_hp > critical_response_damage:
                healing_score = 1.0
            elif response_damage is not None and healed_hp > response_damage:
                healing_score = 0.90
            else:
                healing_score = 0.10
        switch_score = -1.0
        if switch_option is not None:
            target, profile = switch_option
            switch_score = (
                profile.survival_probability
                if profile.survival_probability is not None
                else 0.20
            )
            switch_score += min(0.25, profile.damage_max / max(1, opponent.current_hp) * 0.25)
        self._trainer_high_risk_streak = 0
        if healing_action is None and switch_option is None:
            return None
        if healing_score >= switch_score and healing_action is not None:
            self._remember_battle_support(active_index, battle_state=battle_state)
            context.battle_decision_source = "NUZLOCKE TRAINER HIGH-RISK FALLBACK"
            context.battle_decision_detail = (
                f"repeated {risk} attack risk; bounded comparison selected healing over an imperfect switch"
            )
            context.message = "NUZLOCKE SAFETY: healing after repeated high-risk trainer turns"
            return healing_action
        target, profile = switch_option
        self._remember_battle_support(target, battle_state=battle_state)
        context.battle_decision_source = "NUZLOCKE TRAINER HIGH-RISK FALLBACK"
        context.battle_decision_detail = (
            f"repeated {risk} attack risk; selected least-bad damaging switch to slot {target}; "
            f"estimated survival={profile.survival_probability!r}"
        )
        context.message = "NUZLOCKE SAFETY: switching after repeated high-risk trainer turns"
        return TurnAction.rotate_lead(target)

    def _switch_candidate_risk(
        self,
        battle_state: BattleState,
        party_index: int,
        *,
        active_index: int | None = None,
        util: BattleStrategyUtil | None = None,
        response_turns: int | None = None,
        response_timing: str = _RESPONSE_TIMING_MID_TURN_SWITCH,
    ) -> _SwitchRisk | None:
        """Evaluate one switch and its response line.

        ``response_turns`` bounds the line for emergency decisions, while
        callers that omit it retain the full projected exchange.
        """

        try:
            if active_index is None:
                active_index = battle_state.own_side.active_battler.party_index
            if party_index == active_index:
                return None
            party = get_party()
            pokemon = party[party_index]
            opponent = battle_state.opponent.active_battler
            util = util or BattleStrategyUtil(battle_state)
            move_index = util.get_strongest_move_against(pokemon, opponent)
            if move_index is None:
                return None
            move = pokemon.moves[move_index].move
            damage = util.calculate_move_damage_range(move, pokemon, opponent)
            damage_min = int(getattr(damage, "min", getattr(damage, "max", 0)))
            damage_max = int(getattr(damage, "max", 0))
            if damage_max <= 0 or damage_min <= 0:
                return None
            move_names = _opponent_move_names_for_safety(battle_state)
            # A few pre-battle/test snapshots have no opponent move field at
            # all. That is different from a live battle with an observed
            # empty move set: preserve the compatibility fallback for the
            # former while treating live unknown response facts as unsafe.
            if not move_names and not hasattr(opponent, "moves"):
                return _SwitchRisk(
                    damage_min,
                    damage_max,
                    getattr(pokemon, "current_hp", None),
                    0,
                    0,
                    0.0,
                    1.0,
                    "opponent move snapshot is unavailable; no modeled response damage",
                )
            response = _response_risk_for_line(
                util,
                opponent,
                pokemon,
                move_names,
                damage_min=damage_min,
                damage_max=damage_max,
                response_turns=response_turns,
                response_timing=response_timing,
            )
            if response is None:
                return _SwitchRisk(
                    damage_min,
                    damage_max,
                    getattr(pokemon, "current_hp", None),
                    None,
                    None,
                    None,
                    None,
                    "opponent response is unknown; switch survival cannot be proven",
                )
            (
                normal_incoming,
                critical_incoming,
                critical_probability,
                survival_probability,
            ) = response
            if normal_incoming >= pokemon.current_hp:
                reason = (
                    f"known response can deal up to {normal_incoming} damage over "
                    f"{response_turns or 'the projected'} response turns to {pokemon.current_hp} HP"
                )
            elif critical_incoming >= pokemon.current_hp:
                reason = (
                    f"only a critical response can be lethal over {response_turns or 'the projected'} "
                    f"response turns ({critical_incoming} damage); "
                    f"estimated lethal critical probability is {critical_probability:.1%}"
                )
            else:
                reason = (
                    f"survives normal and critical response ceilings over {response_turns or 'the projected'} "
                    f"response turns "
                    f"({normal_incoming}/{critical_incoming} incoming vs {pokemon.current_hp} HP)"
                )
            return _SwitchRisk(
                damage_min,
                damage_max,
                pokemon.current_hp,
                normal_incoming,
                critical_incoming,
                critical_probability,
                survival_probability,
                reason,
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            return None

    def _select_switch_target(
        self,
        battle_state: BattleState,
        active_index: int | None,
        *,
        util: BattleStrategyUtil | None = None,
        require_strict_safe: bool,
        prefer_lowest_level: bool,
        require_stronger_than_active: bool = False,
        prefer_fresh_for_opponent: bool = False,
        response_turns: int | None = None,
        allow_imperfect: bool = False,
        ignore_voluntary_commitment: bool = False,
    ) -> int | None:
        """Select a target through the one switch-action safety path.

        ``require_strict_safe`` retains the call-site vocabulary used by the
        earlier deterministic policy. It now means that unknown response
        facts are not accepted; a complete line may still qualify through the
        probability threshold when its only lethal outcome is sufficiently
        unlikely. ``response_turns`` bounds the response horizon when the
        caller needs a near-term switch decision.
        """

        if active_index is None:
            return None
        try:
            util = util or BattleStrategyUtil(battle_state)
            if not util.can_switch():
                return None
            active_level = getattr(get_party()[active_index], "level", 0)
            opponent = battle_state.opponent.active_battler
            poison_threat = _known_opponent_can_poison(battle_state)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        eligible_indices = _eligible_indices(only_non_fainted=True)
        active_party_indices = _active_party_indices(battle_state)
        level_cap = _active_level_cap()
        priorities = tuple(_battle_training_priority(index, level_cap) for index in eligible_indices)
        policy_context = (
            SwitchEvaluationContext.SUPPORT
            if require_stronger_than_active
            else SwitchEvaluationContext.LEVELING
            if prefer_lowest_level
            else SwitchEvaluationContext.FALLBACK
            if allow_imperfect
            else SwitchEvaluationContext.EMERGENCY
        )
        committed_prior = (
            None
            if ignore_voluntary_commitment
            else self._committed_prior_index(battle_state)
        )
        allow_commitment_override = self._commitment_reversal_is_necessary(battle_state, active_index)
        candidates: list[SwitchCandidateFacts] = []
        profiles: dict[int, _SwitchRisk] = {}
        party = get_party()
        for index in eligible_indices:
            if index == active_index:
                continue
            pokemon = party[index]
            if (
                committed_prior is not None
                and index == committed_prior
                and not allow_commitment_override
            ):
                _trace_switch_candidate(index, pokemon, None, "ordinary switch would immediately reverse the prior commitment")
                continue
            # A fresh active Pokémon has already earned its share for this
            # opposing Pokémon. Replacing it with someone who participated
            # earlier cannot improve EXP distribution, so ordinary leveling
            # selection must leave the fresh active in place. Emergency paths
            # use a separate selector and may override this for safety.
            if (
                prefer_fresh_for_opponent
                and not self._participated_against_current_opponent(active_index)
                and self._participated_against_current_opponent(index)
            ):
                _trace_switch_candidate(index, pokemon, None, "already participated; no EXP benefit over fresh active")
                continue
            if require_stronger_than_active and getattr(pokemon, "level", 0) <= active_level:
                _trace_switch_candidate(index, pokemon, None, "not stronger than active policy lead")
                continue
            if not self.pokemon_can_battle(pokemon):
                _trace_switch_candidate(index, pokemon, None, "pokemon cannot battle")
                continue
            profile = self._switch_candidate_risk(
                battle_state,
                index,
                active_index=active_index,
                util=util,
                response_turns=response_turns,
            )
            if profile is None and not hasattr(pokemon, "moves"):
                # Incomplete synthetic snapshots do not expose the party
                # move list. The caller has already established that this
                # slot is battle-capable; treat its missing response line as
                # zero modeled damage for compatibility only.
                profile = _SwitchRisk(
                    1,
                    1,
                    getattr(pokemon, "current_hp", None),
                    0,
                    0,
                    0.0,
                    1.0,
                    "party move snapshot is unavailable; no modeled response damage",
                )
            if profile is None:
                _trace_switch_candidate(index, pokemon, None, "no usable damaging move or response profile")
                continue
            status_value = getattr(getattr(pokemon, "status_condition", None), "value", None)
            healthy = status_value in (None, StatusCondition.Healthy.value)
            fresh_for_opponent = int(not self._participated_against_current_opponent(index))
            training_score = relative_training_score(_battle_training_priority(index, level_cap), priorities)
            candidates.append(
                SwitchCandidateFacts(
                    party_index=index,
                    hp=getattr(pokemon, "current_hp", 0),
                    max_hp=getattr(pokemon, "total_hp", getattr(pokemon, "current_hp", 0)),
                    opponent_hp=getattr(opponent, "current_hp", 0),
                    damage_min=profile.damage_min,
                    damage_max=profile.damage_max,
                    incoming_damage_max=profile.normal_incoming,
                    incoming_critical_damage_max=profile.critical_incoming,
                    critical_ko_probability=profile.critical_ko_probability,
                    survival_probability=profile.survival_probability,
                    strict_safe=profile.strict_safe,
                    probabilistically_safe=profile.probabilistically_safe,
                    response_facts_known=profile.normal_incoming is not None,
                    fresh_participant=bool(fresh_for_opponent),
                    training_priority=training_score,
                    healthy=healthy,
                    poison_threat=poison_threat,
                    active_party_indices=active_party_indices,
                )
            )
            profiles[index] = profile

        if not candidates:
            return None
        all_evaluations = rank_switch_candidates(candidates, policy_context)
        evaluations = tuple(evaluation for evaluation in all_evaluations if evaluation.eligible)
        for evaluation in all_evaluations:
            if not evaluation.eligible:
                _trace_switch_candidate(
                    evaluation.party_index,
                    party[evaluation.party_index],
                    profiles[evaluation.party_index],
                    evaluation.reason,
                )
        if not evaluations:
            return None
        selected_evaluation = evaluations[0]
        selected_index = selected_evaluation.party_index
        profile = profiles[selected_index]
        for evaluation in evaluations:
            _trace_switch_candidate(
                evaluation.party_index,
                party[evaluation.party_index],
                profiles[evaluation.party_index],
                f"accepted score={evaluation.score:.2f} {evaluation.reason}",
            )
        diagnostic_print(
            lambda: (
                "NUZLOCKE_SWITCH_EVALUATION: "
                f"active={active_index!r} target={selected_index!r} "
                f"policy={policy_context.value!r} score={selected_evaluation.score:.2f} "
                f"strict_safe={profile.strict_safe!r} normal_incoming={profile.normal_incoming!r} "
                f"critical_incoming={profile.critical_incoming!r} "
                f"critical_ko_probability={profile.critical_ko_probability!r} "
                f"survival_probability={profile.survival_probability!r} "
                f"response_turns={response_turns!r} "
                f"damage={profile.damage_min}-{profile.damage_max} "
                f"exp_remaining_in_cap={_experience_headroom(selected_index, _active_level_cap())!r} "
                f"reason={selected_evaluation.reason!r}"
            ),
            trace=True,
        )
        return selected_index

    def _choose_trainer_replacement_with_expectiminimax(
        self,
        battle_state: BattleState,
    ) -> tuple[bool, int | None]:
        """Let the shared planner own the optional trainer switch boundary.

        ``True`` means the normalized planner was reached and its answer is
        authoritative, including ``None`` for answering No. ``False`` is
        reserved for incomplete legacy test doubles that cannot be adapted to
        planner facts; live campaign battles must never fall back to the old
        ranking policy after a planner result exists.
        """

        try:
            current_opponent = battle_state.opponent.active_battler
            current_opponent_index = current_opponent.party_index
            from modules.pokemon_party import get_opponent_party

            opponent_party = get_opponent_party()
            next_opponent = next(
                (
                    pokemon
                    for pokemon in (opponent_party or ())
                    if getattr(pokemon, "index", -1) > current_opponent_index
                    and not pokemon.is_empty
                    and pokemon.is_valid
                    and pokemon.current_hp > 0
                ),
                None,
            )
            if next_opponent is None:
                context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX TRAINER FREE SWITCH"
                context.battle_decision_detail = "no living incoming trainer Pokémon remains; answering No"
                return True, None

            self._advance_to_opponent(next_opponent)
            self._publish_expectiminimax_experience_facts(battle_state, opponent=next_opponent)
            decision = plan_battle_state(
                battle_state,
                incoming_opponent=next_opponent,
                free_switch=True,
            )
            stochastic = getattr(decision, "stochastic_result", None)
            action = getattr(decision, "action", None)
            target = getattr(decision, "target", None)
            context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX TRAINER FREE SWITCH"
            context.battle_decision_detail = (
                f"incoming_opponent={getattr(getattr(next_opponent, 'species', None), 'name', next_opponent)!r}; "
                f"action={getattr(action, 'value', action)!r}; target={target!r}; "
                f"classification={getattr(getattr(decision, 'classification', None), 'value', None)!r}; "
                f"category={getattr(getattr(stochastic, 'category', None), 'value', None)!r}; "
                f"rationale={getattr(decision, 'rationale', '')}"
            )
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_TRAINER_FREE_SWITCH_PLANNER: "
                    f"incoming={_trainer_replacement_pokemon_trace(next_opponent)!r} "
                    f"action={getattr(action, 'value', action)!r} target={target!r} "
                    f"category={getattr(getattr(stochastic, 'category', None), 'value', None)!r} "
                    f"classification={getattr(getattr(decision, 'classification', None), 'value', None)!r}"
                ),
                trace=True,
                prefix="NUZLOCKE_TRAINER_FREE_SWITCH_PLANNER:",
            )
            if action is PlannerAction.SwitchPokemon and isinstance(target, int):
                if target == getattr(battle_state.own_side.active_battler, "party_index", None):
                    return True, None
                self._remember_free_trainer_switch(target, next_opponent, "expectiminimax trainer replacement root")
                return True, target
            # A move root means the planner chose No: the current active
            # Pokémon gets the next ordinary turn against the incoming foe.
            return True, None
        except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError, IndexError) as error:
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_TRAINER_FREE_SWITCH_PLANNER: "
                    f"result='adapter_unavailable' exception_type={type(error).__name__!r} error={error!r}"
                ),
                trace=True,
                prefix="NUZLOCKE_TRAINER_FREE_SWITCH_PLANNER:",
            )
            if _campaign_progression_is_active():
                context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX TRAINER FREE SWITCH"
                context.battle_decision_detail = (
                    "planner could not normalize the incoming trainer replacement boundary; answering No"
                )
                return True, None
            return False, None

    def choose_trainer_replacement(self, battle_state: BattleState) -> int | None:
        """Choose a safe EXP-aware replacement for a trainer's next Pokémon."""

        planner_handled, planner_choice = self._choose_trainer_replacement_with_expectiminimax(battle_state)
        if planner_handled:
            return planner_choice

        self._sync_opponent_participation(battle_state)

        runtime = getattr(context, "nuzlocke_runtime", None)
        rule_config = getattr(runtime, "rule_config", None)
        from modules.nuzlocke.rule_config import CampaignRuleId

        set_battle_style_enabled = bool(
            rule_config is not None and rule_config.is_enabled(CampaignRuleId.SET_BATTLE_STYLE)
        )
        _trace_trainer_replacement_policy(
            f"entry set_battle_style={set_battle_style_enabled!r} "
            f"runtime_present={runtime is not None!r} rule_config_present={rule_config is not None!r}"
        )
        # ``set_battle_style`` controls the game's startup configuration. In
        # Set mode this prompt normally never appears; if it does appear, the
        # strategic policy must still be allowed to evaluate it. In Shift
        # mode this is the normal path and the same evaluator chooses Yes/No.
        context.battle_decision_source = "NUZLOCKE TRAINER SWITCH"
        context.battle_decision_detail = "inspecting the incoming trainer Pokémon"
        _trace_trainer_replacement_policy(
            f"prompt_visible style={'set' if set_battle_style_enabled else 'shift'} action='evaluate'"
        )
        if getattr(battle_state, "is_double_battle", False):
            # Double battles deliberately retain their existing action path.
            _trace_trainer_replacement_policy("reject reason=double_battle")
            context.battle_decision_detail = "optional replacement is not evaluated for double battles"
            return None

        try:
            current_opponent = battle_state.opponent.active_battler
            current_opponent_index = current_opponent.party_index
            from modules.pokemon_party import get_opponent_party

            opponent_party = get_opponent_party()
            _trace_trainer_replacement_policy(
                f"battle_snapshot current_opponent={_trainer_replacement_pokemon_trace(current_opponent)} "
                f"opponent_party={tuple(_trainer_replacement_pokemon_trace(pokemon) for pokemon in (opponent_party or ()))!r}"
            )
            next_opponent = next(
                (
                    pokemon
                    for pokemon in (opponent_party or ())
                    if getattr(pokemon, "index", -1) > current_opponent_index
                    and not pokemon.is_empty
                    and pokemon.is_valid
                    and pokemon.current_hp > 0
                ),
                None,
            )
            if next_opponent is None:
                _trace_trainer_replacement_policy(
                    f"reject reason=no_next_opponent current_opponent_index={current_opponent_index!r}"
                )
                context.battle_decision_detail = "no living incoming trainer Pokémon remains; staying active"
                return None
            # The incoming trainer Pokémon represents a new EXP opportunity.
            # Do this before ranking candidates so a member used against the
            # previous opponent is not incorrectly treated as already trained
            # for the next one.
            self._advance_to_opponent(next_opponent)
            util = BattleStrategyUtil(battle_state)
            active_indices = _active_party_indices(battle_state)
            self._trace_experience_projection(
                battle_state,
                getattr(battle_state.own_side.active_battler, "party_index", None),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
            _trace_trainer_replacement_policy(f"reject reason=snapshot_exception exception={error!r}")
            context.battle_decision_detail = "could not inspect the incoming trainer Pokémon; staying active"
            return None

        response_moves = tuple(
            learned_move.move.name
            for learned_move in getattr(next_opponent, "moves", ())
            if learned_move is not None
            and getattr(learned_move, "move", None) is not None
            and getattr(learned_move, "pp", 1) > 0
        )
        response_facts_known = bool(response_moves)
        next_poison_threat = any(
            _move_name_may_inflict_poison(name)
            for name in response_moves
        )

        eligible_indices = _eligible_indices(only_non_fainted=True)
        level_cap = _active_level_cap()
        priorities = tuple(_battle_training_priority(index, level_cap) for index in eligible_indices)
        candidates: list[SwitchCandidateFacts] = []
        current_index = getattr(battle_state.own_side.active_battler, "party_index", None)
        force_switch_from_current = False
        if current_index is not None and 0 <= current_index < len(get_party()):
            current = get_party()[current_index]
            try:
                force_switch_from_current = (
                    current.current_hp / current.total_hp <= 0.25
                    or BattleStrategyUtil.is_poisoned(current)
                )
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                force_switch_from_current = False
        _trace_trainer_replacement_policy(
            f"next_opponent={_trainer_replacement_pokemon_trace(next_opponent)} "
            f"response_moves={response_moves!r} response_facts_known={response_facts_known!r} "
            f"poison_threat={next_poison_threat!r} current_index={current_index!r} "
            f"active_indices={tuple(sorted(active_indices))!r} force_switch_from_current={force_switch_from_current!r} "
            f"party={tuple(_trainer_replacement_pokemon_trace(pokemon, index) for index, pokemon in enumerate(get_party()))!r}"
        )
        for index in eligible_indices:
            # In a single battle the current Pokémon is a legitimate
            # "stay in" candidate.  Returning None is how the handler
            # answers the ROM prompt with No. Other active slots remain
            # excluded for safety and for future multi-battle callers.
            if index in active_indices and index != current_index:
                _trace_trainer_replacement_policy(f"candidate index={index} reject=already_active_other_slot")
                continue
            if force_switch_from_current and index == current_index:
                _trace_trainer_replacement_policy(f"candidate index={index} reject=current_is_unsafe")
                continue
            pokemon = get_party()[index]
            if not self.pokemon_can_battle(pokemon):
                _trace_trainer_replacement_policy(f"candidate index={index} reject=pokemon_can_battle_false")
                continue
            try:
                move_index = util.get_strongest_move_against(pokemon, next_opponent)
                if move_index is None:
                    _trace_trainer_replacement_policy(f"candidate index={index} reject=no_damaging_move")
                    continue
                move = pokemon.moves[move_index].move
                damage = util.calculate_move_damage_range(move, pokemon, next_opponent)
                if damage.max <= 0:
                    _trace_trainer_replacement_policy(f"candidate index={index} reject=zero_damage")
                    continue
                response_profile = (
                    _response_risk_for_line(
                        util,
                        next_opponent,
                        pokemon,
                        response_moves,
                        damage_min=damage.min,
                        damage_max=damage.max,
                        response_turns=_FREE_TRAINER_SWITCH_RESPONSE_TURNS,
                        response_timing=_RESPONSE_TIMING_PRE_TURN_REPLACEMENT,
                    )
                    if response_facts_known
                    else None
                )
                if response_facts_known and not self._survives_visible_response(
                    util,
                    next_opponent,
                    pokemon,
                    response_moves,
                    damage_min=damage.min,
                    damage_max=damage.max,
                    response_turns=_FREE_TRAINER_SWITCH_RESPONSE_TURNS,
                ):
                    _trace_trainer_replacement_policy(
                        f"candidate index={index} reject=does_not_survive_visible_response "
                        f"survival_probability={response_profile[3] if response_profile is not None else None!r}"
                    )
                    continue
                if not _participants_remain_cap_legal(
                    battle_state,
                    {index},
                    opponent=next_opponent,
                ):
                    _trace_trainer_replacement_policy(f"candidate index={index} reject=level_cap")
                    continue
                guaranteed_ko = damage.min >= next_opponent.current_hp
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError) as error:
                _trace_trainer_replacement_policy(f"candidate index={index} reject=evaluation_exception exception={error!r}")
                continue
            candidate_profile = _SwitchRisk(
                damage.min,
                damage.max,
                getattr(pokemon, "current_hp", None),
                response_profile[0] if response_profile is not None else None,
                response_profile[1] if response_profile is not None else None,
                response_profile[2] if response_profile is not None else None,
                response_profile[3] if response_profile is not None else None,
                "trainer replacement response line",
            )
            status_value = getattr(getattr(pokemon, "status_condition", None), "value", None)
            candidates.append(
                SwitchCandidateFacts(
                    party_index=index,
                    hp=getattr(pokemon, "current_hp", 0),
                    max_hp=getattr(pokemon, "total_hp", getattr(pokemon, "current_hp", 0)),
                    opponent_hp=getattr(next_opponent, "current_hp", 0),
                    damage_min=damage.min,
                    damage_max=damage.max,
                    incoming_damage_max=candidate_profile.normal_incoming,
                    incoming_critical_damage_max=candidate_profile.critical_incoming,
                    critical_ko_probability=candidate_profile.critical_ko_probability,
                    survival_probability=candidate_profile.survival_probability,
                    strict_safe=candidate_profile.strict_safe,
                    probabilistically_safe=candidate_profile.probabilistically_safe,
                    response_facts_known=response_facts_known,
                    fresh_participant=True,
                    training_priority=relative_training_score(_battle_training_priority(index, level_cap), priorities),
                    healthy=status_value in (None, StatusCondition.Healthy.value),
                    poison_threat=next_poison_threat,
                    guaranteed_ko=guaranteed_ko,
                    is_current=index == current_index,
                    # The current battler is retained as a legitimate
                    # "stay in" comparison candidate. Other active slots
                    # cannot be selected by the optional replacement path.
                    active_party_indices=frozenset(active_indices - {current_index}),
                )
            )
            _trace_trainer_replacement_policy(
                f"candidate index={index} accepted damage={damage.min}-{damage.max} "
                f"response_survival_probability={response_profile[3] if response_profile is not None else None!r} "
                f"guaranteed_ko={guaranteed_ko!r} fresh_for_next_opponent=True"
            )

        evaluations = rank_switch_candidates(candidates, SwitchEvaluationContext.TRAINER_REPLACEMENT)
        evaluations = tuple(evaluation for evaluation in evaluations if evaluation.eligible)
        if not evaluations:
            _trace_trainer_replacement_policy("result selected_index=None candidate_count=0")
            tag_index = self._authorize_trainer_exp_tag(
                battle_state,
                active_index=current_index,
                opponent=next_opponent,
                response_moves=response_moves,
                util=util,
            )
            if tag_index is not None:
                return tag_index
            context.battle_decision_source = "NUZLOCKE TRAINER SWITCH"
            context.battle_decision_detail = (
                "no replacement candidate survived the one-response safety check "
                "with a damaging matchup"
            )
            return None
        selected_evaluation = evaluations[0]
        # This is a free replacement decision. If the current member is the
        # weighted winner, still use the prompt for a lower-level member when
        # that member has a safe damaging line. This switch costs no turn, so
        # requiring equal damage would throw away a free EXP opportunity and
        # can force the ordinary policy to spend a later turn doing the same
        # rotation under a stricter emergency horizon.
        if selected_evaluation.party_index == current_index:
            # Preserve the dedicated no-damage EXP-tag plan when it is
            # available. It is more valuable than choosing a merely
            # damaging lower-level finisher because it trains both members
            # during the same free replacement opportunity.
            tag_index = self._authorize_trainer_exp_tag(
                battle_state,
                active_index=current_index,
                opponent=next_opponent,
                response_moves=response_moves,
                util=util,
            )
            if tag_index is not None:
                return tag_index
            free_training_candidates = tuple(
                evaluation
                for evaluation in evaluations
                if evaluation.party_index != current_index
                and getattr(get_party()[evaluation.party_index], "level", 0)
                < getattr(get_party()[current_index], "level", 0)
            )
            if free_training_candidates:
                selected_evaluation = min(
                    free_training_candidates,
                    key=lambda evaluation: _battle_training_priority(evaluation.party_index, level_cap),
                )
                context.battle_decision_source = "NUZLOCKE TRAINER FREE SWITCH"
                context.battle_decision_detail = (
                    f"free replacement to lower-level party slot {selected_evaluation.party_index}; "
                    "its validated one-response survival and damaging line make the free EXP rotation worthwhile"
                )
        selected = selected_evaluation.party_index
        if selected == current_index:
            _trace_trainer_replacement_policy(
                f"result selected_index=None candidate_count={len(candidates)} "
                f"reason=current_is_safest score={selected_evaluation.score:.2f} "
                f"candidate_reason={selected_evaluation.reason!r}"
            )
            context.battle_decision_source = "NUZLOCKE TRAINER SWITCH"
            context.battle_decision_detail = "current Pokémon is the safest eligible response"
            return None
        _trace_trainer_replacement_policy(
            f"result selected_index={selected!r} candidate_count={len(candidates)} "
            f"score={selected_evaluation.score:.2f} candidate_reason={selected_evaluation.reason!r}"
        )
        context.battle_decision_source = (
            getattr(context, "battle_decision_source", None) or "NUZLOCKE TRAINER SWITCH"
        )
        context.battle_decision_detail = (
            f"next opponent={getattr(next_opponent.species, 'name', next_opponent.species)!r}; "
            f"selected lowest safe cap-legal party slot {selected}; "
            f"response_facts_known={response_facts_known!r}; "
            f"score={selected_evaluation.score:.2f}; "
            f"exp={self._experience_projection_detail(battle_state, {selected}, opponent=next_opponent)}"
        )
        context.message = f"TRAINER SWITCH: sending party slot {selected} into the next opponent"
        self._remember_free_trainer_switch(selected, next_opponent, "trainer replacement prompt selection")
        return selected

    def _choose_forced_replacement(self, battle_state: BattleState) -> int | None:
        """Choose the safest legal replacement after a faint.

        The forced replacement prompt is already a free battle transition, so
        ``can_switch()`` is intentionally not consulted here. A fainted lead
        must be replaced using the same one-response survival facts as an
        emergency switch, with offense and training serving only as secondary
        preferences.
        """

        opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
        if opponent is None:
            return None
        try:
            active_indices = _active_party_indices(battle_state)
            eligible = tuple(
                index
                for index in _eligible_indices(only_non_fainted=True)
                if index not in active_indices
            )
            util = BattleStrategyUtil(battle_state)
            party = get_party()
            profiles: dict[int, _SwitchRisk] = {}
            priorities = tuple(_battle_training_priority(index) for index in eligible)
            candidates: list[SwitchCandidateFacts] = []
            response_moves = _opponent_move_names_for_safety(battle_state)
            for index in eligible:
                profile = self._switch_candidate_risk(
                    battle_state,
                    index,
                    active_index=None,
                    util=util,
                    response_turns=_FREE_TRAINER_SWITCH_RESPONSE_TURNS,
                    response_timing=_RESPONSE_TIMING_PRE_TURN_REPLACEMENT,
                )
                if profile is None:
                    _trace_switch_candidate(index, party[index], None, "forced replacement rejected: no response profile")
                    continue
                profiles[index] = profile
                facts = SwitchCandidateFacts(
                    party_index=index,
                    hp=getattr(party[index], "current_hp", 0),
                    max_hp=getattr(party[index], "total_hp", getattr(party[index], "current_hp", 0)),
                    opponent_hp=getattr(opponent, "current_hp", 0),
                    damage_min=profile.damage_min,
                    damage_max=profile.damage_max,
                    incoming_damage_max=profile.normal_incoming,
                    incoming_critical_damage_max=profile.critical_incoming,
                    critical_ko_probability=profile.critical_ko_probability,
                    survival_probability=profile.survival_probability,
                    strict_safe=profile.strict_safe,
                    probabilistically_safe=profile.probabilistically_safe,
                    response_facts_known=bool(response_moves and profile.normal_incoming is not None),
                    training_priority=relative_training_score(
                        _battle_training_priority(index), priorities
                    ),
                    healthy=getattr(getattr(party[index], "status_condition", None), "value", None)
                    in (None, StatusCondition.Healthy.value),
                    poison_threat=_known_opponent_can_poison(battle_state),
                    active_party_indices=active_indices,
                )
                candidates.append(facts)
                _trace_switch_candidate(index, party[index], profile, "forced replacement candidate")
            if not candidates:
                return None
            evaluations = rank_switch_candidates(candidates, SwitchEvaluationContext.EMERGENCY)
            evaluations = tuple(evaluation for evaluation in evaluations if evaluation.eligible)
            if not evaluations:
                return None
            selected = evaluations[0]
            for evaluation in evaluations:
                _trace_switch_candidate(
                    evaluation.party_index,
                    party[evaluation.party_index],
                    profiles.get(evaluation.party_index),
                    f"forced replacement score={evaluation.score:.2f} {evaluation.reason}",
                )
            profile = profiles[selected.party_index]
            context.battle_decision_source = "NUZLOCKE FORCED REPLACEMENT"
            context.battle_decision_detail = (
                f"selected slot {selected.party_index} by one-response survival evaluation; "
                f"survival={profile.survival_probability!r}; damage={profile.damage_min}-{profile.damage_max}; "
                f"reason={selected.reason}"
            )
            context.message = (
                f"NUZLOCKE SAFETY: forced replacement slot {selected.party_index}; "
                f"survival={profile.survival_probability!r}"
            )
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_FORCED_REPLACEMENT_EVALUATION: "
                    f"selected={selected.party_index!r} survival={profile.survival_probability!r} "
                    f"normal_incoming={profile.normal_incoming!r} critical_incoming={profile.critical_incoming!r} "
                    f"score={selected.score:.2f}"
                ),
                trace=True,
                prefix="NUZLOCKE_FORCED_REPLACEMENT_EVALUATION:",
            )
            return selected.party_index
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            return None

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
        index = self._choose_forced_replacement(battle_state)
        if index is None:
            index = _lowest_battle_capable_index(self, battle_state)
        if index is None and getattr(getattr(battle_state, "opponent", None), "active_battler", None) is None:
            # Preserve the old selection behavior when the replacement menu is
            # reached without an opponent snapshot.  In a live battle with an
            # opponent, ``has_replacement_after_faint`` prevents this branch
            # from selecting a known non-damaging replacement.
            index = _lowest_level_index(only_non_fainted=True)
        if index is None:
            raise RuntimeError("no living Pokémon is available for campaign preparation")
        return index

    def has_replacement_after_faint(self, battle_state: BattleState) -> bool:
        active_indices = _active_party_indices(battle_state)
        eligible_non_active = tuple(
            index for index in _eligible_indices(only_non_fainted=True) if index not in active_indices
        )
        if not eligible_non_active:
            return False
        return any(index in _battle_capable_indices(self, battle_state) for index in eligible_non_active)

    def choose_new_lead_after_battle(self) -> int | None:
        field_candidate = self._field_lead_candidate()
        index = field_candidate
        if index is None:
            index = _lowest_level_index(only_non_fainted=True)
        level_ceiling = _field_encounter_level_ceiling()
        candidate_level = getattr(get_party()[index], "level", None) if index is not None else None
        lead_policy_reason = (
            "lowest-level member below local wild-level ceiling"
            if (
                field_candidate is not None
                and level_ceiling is not None
                and candidate_level is not None
                and candidate_level < level_ceiling
            )
            else "local encounter coverage after party reached local wild-level ceiling"
            if field_candidate is not None
            else "no usable local encounter envelope; lowest living level"
        )
        try:
            current_lead = get_party().first_non_fainted.index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            current_lead = 0
        selected = index if index is not None and index != current_lead else None
        diagnostic_print(
            lambda: (
                "NUZLOCKE_LEVEL_BALANCE_AFTER_BATTLE: "
                f"candidate={index!r} current_lead={current_lead!r} selected={selected!r} "
                f"lead_policy={lead_policy_reason!r} "
                f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
            ),
            trace=True,
        )
        if selected is not None:
            context.battle_decision_source = "NUZLOCKE FIELD LEAD"
            context.battle_decision_detail = (
                f"selected party slot {selected} using {lead_policy_reason}"
            )
            context.message = (
                f"LEVELING: field lead selected party slot {selected} ({lead_policy_reason})"
                f"{_preparation_target_suffix()}"
            )
        return selected

    def decide_turn(self, battle_state: BattleState):
        # Battle tactics are owned by expectiminimax. The scheduler and
        # Nuzlocke policy publish objective facts into the normalized planner
        # context, while this strategy remains responsible for lifecycle
        # bookkeeping and lead selection between battles.
        self._publish_expectiminimax_experience_facts(battle_state)
        if _campaign_progression_is_active() or not hasattr(battle_state, "own_side"):
            return CampaignExpectiminimaxStrategy.decide_turn(self, battle_state)

        # Legacy strategy-local switching, capture delegation, and risk
        # fallbacks remain below as reference only. They must not select a
        # battle action after the unified planner boundary above.
        self._publish_expectiminimax_experience_facts(battle_state)
        # The battle listener normally hands a legal encounter directly to
        # NuzlockeCaptureStrategy.  There is a narrow ready-turn race where
        # the listener has already selected this ordinary campaign strategy,
        # but the battle-state projection now says that the encounter is
        # capture-owned.  Keep that fallback on the same capture policy so a
        # weak lead is not asked to fight or throw a ball without the capture
        # strategy's lead switch and no-ball behavior.
        if getattr(battle_state, "nuzlocke_capture_target", False):
            capture_strategy = getattr(self, "_nuzlocke_capture_strategy", None)
            if capture_strategy is None:
                capture_strategy = NuzlockeCaptureStrategy()
                self._nuzlocke_capture_strategy = capture_strategy
            capture_action = capture_strategy.decide_turn(battle_state)
            if capture_action[0] is not TurnAction.SwitchToManual:
                return capture_action

        active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
        active_index = getattr(active, "party_index", None)
        self._sync_opponent_participation(battle_state)
        self._trace_experience_projection(battle_state, active_index)

        # A free trainer replacement has already consumed the prompt's
        # switch opportunity. Re-evaluate that exact active Pokémon before
        # allowing the planner to spend a turn attacking or tagging again.
        free_switch_recheck = self._recheck_after_free_switch(battle_state, active_index)
        if free_switch_recheck is not None:
            return free_switch_recheck

        # A complete trainer envelope gives expectiminimax the authoritative
        # action-selection role. The level-balancing superclass can switch a
        # weak lead before DefaultBattleStrategy ever invokes the evaluator,
        # which made ordinary trainer turns appear as NUZLOCKE RISK SAFETY.
        # Run the planner-owned path first for recognized trainer battles and
        # return its result whenever the stochastic evaluator is available.
        # Unknown or incomplete trainer facts retain the conservative safety
        # fallback below.
        planner_owned_trainer_action = None
        blocked_stochastic_result = None
        if getattr(battle_state, "is_trainer_battle", False) and not getattr(battle_state, "is_double_battle", False):
            try:
                planner_owned_trainer_action = _NuzlockeBattleDispatch.decide_turn(self, battle_state)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                # Lightweight tests and ROM transition snapshots can omit the
                # active move projection. Treat that as incomplete planner
                # facts and retain the emergency path below.
                planner_owned_trainer_action = None
            planner_report = getattr(self, "_last_planner_decision", None)
            stochastic_result = getattr(planner_report, "stochastic_result", None)
            if stochastic_result is not None:
                # A projection or node-budget result is useful evidence for
                # ranking, but it has not established a line that a Nuzlocke
                # may execute. Keep the result visible to the emergency policy
                # and stop if that policy cannot produce a validated action.
                result_complete = getattr(stochastic_result, "complete", None)
                result_safe = getattr(stochastic_result, "is_safe", None)
                result_complete = True if result_complete is None else bool(result_complete)
                result_safe = True if result_safe is None else bool(result_safe)
                diagnostic_print(
                    lambda: (
                        "CAMPAIGN_TRAINER_PLANNER_OWNERSHIP: "
                        f"action={planner_owned_trainer_action!r} "
                        f"evaluation_method={getattr(planner_report, 'evaluation_method', None)!r} "
                        f"complete={result_complete!r} safe={result_safe!r} "
                        f"category={getattr(getattr(stochastic_result, 'category', None), 'value', None)!r}"
                    ),
                    trace=True,
                    prefix="CAMPAIGN_TRAINER_PLANNER_OWNERSHIP",
                )
                if (
                    result_complete
                    and planner_owned_trainer_action is not None
                    and planner_owned_trainer_action[0] is not TurnAction.SwitchToManual
                    and getattr(stochastic_result, "catastrophic_probability", 0.0) < 1.0
                ):
                    # A complete expectiminimax result owns the comparative
                    # single-battle action, including healing items. The
                    # Nuzlocke layer still owns legality and campaign policy,
                    # but it must not replace the search's resource/risk
                    # comparison with a separate emergency item heuristic.
                    return planner_owned_trainer_action
                blocked_stochastic_result = stochastic_result

        emergency = self._risk_aware_emergency_action(battle_state, active_index)
        if emergency is not None:
            return emergency
        if blocked_stochastic_result is not None and planner_owned_trainer_action is not None:
            category = getattr(getattr(blocked_stochastic_result, "category", None), "value", None)
            catastrophic_probability = getattr(blocked_stochastic_result, "catastrophic_probability", None)
            projection_category = category in {
                SearchResultCategory.INCOMPLETE.value,
                SearchResultCategory.HORIZON_PROJECTION.value,
            }
            guaranteed_catastrophe = (
                isinstance(catastrophic_probability, (int, float))
                and catastrophic_probability >= 1.0
            )
            # A bounded search is still the best available legal policy when
            # it ran out of horizon or node budget. A guaranteed catastrophic
            # projection also has to execute when a concrete legal action was
            # selected: the battle lifecycle owns the resulting faint/whiteout
            # boundary, where campaign policy can distinguish a failed run
            # from a recoverable PC roster. Keep Manual for an unsafe result
            # that is neither a bounded projection nor a guaranteed terminal
            # line, or when no legal action exists.
            if (
                (projection_category or guaranteed_catastrophe)
                and planner_owned_trainer_action[0] is not TurnAction.SwitchToManual
            ):
                context.battle_decision_source = (
                    "NUZLOCKE EXPECTIMINIMAX CATASTROPHIC PROJECTION"
                    if guaranteed_catastrophe
                    else "NUZLOCKE EXPECTIMINIMAX PROJECTION"
                )
                context.battle_decision_detail = _expectiminimax_reason(
                    blocked_stochastic_result,
                    suffix=(
                        "executing its legal best-available trainer action so the battle lifecycle can own the "
                        "outcome"
                    ),
                )
                context.message = (
                    "NUZLOCKE EXPECTIMINIMAX: executing projected trainer action; "
                    "battle outcome remains under campaign handling"
                )
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_ACTION_EVALUATION: executing projected trainer action; "
                        f"category={category!r} action={planner_owned_trainer_action!r} "
                        f"catastrophic_probability={catastrophic_probability!r}"
                    ),
                    trace=True,
                    prefix="NUZLOCKE_ACTION_EVALUATION",
                )
                return planner_owned_trainer_action
        if blocked_stochastic_result is not None:
            category = getattr(getattr(blocked_stochastic_result, "category", None), "value", "unsafe_or_uncertain")
            context.battle_decision_source = "NUZLOCKE STOCHASTIC SEARCH BLOCK"
            context.battle_decision_detail = (
                f"expectiminimax result category={category!r} did not establish a safe terminal line; "
                "no validated switch or healing action was available"
            )
            context.message = "NUZLOCKE SAFETY: incomplete or unsafe battle search; manual decision required"
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_ACTION_EVALUATION: blocked incomplete_or_unsafe_stochastic_result; "
                    f"category={category!r}"
                ),
                trace=True,
                prefix="NUZLOCKE_ACTION_EVALUATION",
            )
            return TurnAction.switch_to_manual()

        # Complete a previously authorized EXP-tag line before asking the
        # generic planner to choose an action for a Pokémon that may have no
        # effective move.  Revalidate the finisher because HP, status, and
        # battle restrictions can change during the switch response.
        pending_tag = getattr(self, "_pending_exp_tag_index", None)
        pending_finisher = getattr(self, "_pending_exp_finisher_index", None)
        if pending_tag is not None:
            if active_index == pending_tag:
                finisher = self._battle_finisher_target(battle_state, excluded={pending_tag})
                if finisher is not None and finisher == pending_finisher:
                    try:
                        can_switch = BattleStrategyUtil(battle_state).can_switch()
                    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                        can_switch = False
                    if can_switch:
                        self._pending_exp_tag_index = None
                        self._pending_exp_finisher_index = None
                        context.battle_decision_source = "NUZLOCKE EXP TAG CONTINUATION"
                        context.battle_decision_detail = (
                            f"return from temporary EXP tag {pending_tag} to finisher {pending_finisher}"
                        )
                        return TurnAction.rotate_lead(pending_finisher)
            # A battle transition, unexpected active slot, or invalidated
            # finisher ends the pending line. The ordinary policy below can
            # still choose an effective replacement or an escape.
            self._pending_exp_tag_index = None
            self._pending_exp_finisher_index = None

        participation_hold_reason = None
        # Lead rotation after the previous battle is a useful fallback, but
        # it cannot cover a newly captured member or a battle that begins
        # before the post-battle menu handoff is available.  Make the policy
        # explicit at every single-battle turn boundary: use the lowest-level
        # living, battle-capable member only after the same response-survival
        # evaluator has approved the complete switch line.
        if not getattr(battle_state, "is_double_battle", False):
            candidate = _lowest_battle_capable_index(self, battle_state)
            tag_plan = self._choose_exp_tag_plan(battle_state, active_index, candidate)
            if tag_plan is not None:
                tag_index, finisher_index = tag_plan
                self._pending_exp_tag_index = tag_index
                self._pending_exp_finisher_index = finisher_index
                context.battle_decision_source = "NUZLOCKE EXP TAG"
                context.battle_decision_detail = (
                    f"temporary tag {tag_index} is safe; validated finisher is {finisher_index}"
                )
                context.message = (
                    f"NUZLOCKE EXP TAG: switching to party slot {tag_index} for EXP, "
                    f"then returning to slot {finisher_index}"
                )
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_EXP_TAG: " f"tag={tag_index!r} finisher={finisher_index!r} active={active_index!r}"
                    ),
                    trace=True,
                )
                return TurnAction.rotate_lead(tag_index)
            if getattr(self, "_battle_support_index", None) == active_index:
                emergency = self._risk_aware_emergency_action(battle_state, active_index)
                if emergency is not None:
                    return emergency
                action = super().decide_turn(battle_state)
                if action[0] is TurnAction.RotateLead:
                    self._remember_battle_support(action[1], battle_state=battle_state)
                planner_report = getattr(self, "_last_planner_decision", None)
                fallback = self._bounded_trainer_high_risk_fallback(
                    battle_state, action, planner_report
                ) if planner_report is not None else None
                if fallback is not None:
                    return fallback
                return action
            try:
                can_switch = BattleStrategyUtil(battle_state).can_switch()
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                can_switch = False
            safe_candidate = None
            if can_switch and active_index is not None:
                safe_candidate = self._select_switch_target(
                    battle_state,
                    active_index,
                    require_strict_safe=True,
                    prefer_lowest_level=True,
                    prefer_fresh_for_opponent=True,
                )
                # A fresh active Pokémon should not be replaced by a member
                # that has already participated against this same opponent
                # merely because the latter is lower level. If the active
                # Pokémon is not battle-capable, the emergency/support paths
                # above may still authorize that switch.
                if (
                    safe_candidate is not None
                    and candidate == active_index
                    and not self._participated_against_current_opponent(active_index)
                    and self._participated_against_current_opponent(safe_candidate)
                ):
                    safe_candidate = None
                    participation_hold_reason = (
                        f"kept fresh active party slot {active_index}; replacement slot was already "
                        "used against this opponent"
                    )
                    diagnostic_print(
                        lambda: (
                            "NUZLOCKE_EXP_PARTICIPATION_HOLD: "
                            f"active={active_index!r} reason={participation_hold_reason!r}"
                        ),
                        trace=True,
                    )
            # Keep incomplete synthetic battle snapshots usable for campaign
            # preparation and tests. A live battle always has an opponent;
            # when it does not, there is no response line to evaluate and the
            # historical lowest-capable fallback is the only meaningful one.
            if (
                safe_candidate is None
                and getattr(getattr(battle_state, "opponent", None), "active_battler", None) is None
            ):
                safe_candidate = candidate
            if (
                safe_candidate is not None
                and safe_candidate != active_index
                and can_switch
                and active_index is not None
                and _participants_remain_cap_legal(battle_state, {active_index, safe_candidate})
            ):
                selected_profile = self._switch_candidate_risk(
                    battle_state,
                    safe_candidate,
                    active_index=active_index,
                )
                survival_detail = (
                    f"; estimated survival={selected_profile.survival_probability:.1%}"
                    if selected_profile is not None and selected_profile.survival_probability is not None
                    else ""
                )
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_LEVEL_BALANCE_SWITCH: "
                        f"active={active_index!r} candidate={safe_candidate!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
                context.battle_decision_source = "NUZLOCKE LEVEL BALANCE"
                context.battle_decision_detail = (
                    f"selected {'fresh ' if not self._participated_against_current_opponent(safe_candidate) else ''}"
                    f"lowest-level battle-capable party slot {safe_candidate} "
                    f"after switch survival evaluation{survival_detail}"
                )
                context.message = f"LEVELING: switching safely to party slot {safe_candidate}"
                self._remember_battle_support(safe_candidate, battle_state=battle_state)
                return TurnAction.rotate_lead(safe_candidate)

            # Do not let an otherwise legal attack consume the encounter's
            # EXP on a member whose projected total would cross the active
            # cap. Prefer another cap-legal finisher, or escape a wild battle
            # when no such line exists.
            if active_index is not None and not _participants_remain_cap_legal(battle_state, {active_index}):
                cap_legal_finisher = self._battle_finisher_target(battle_state, excluded={active_index})
                if cap_legal_finisher is not None and can_switch:
                    return TurnAction.rotate_lead(cap_legal_finisher)
                if getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False)):
                    try:
                        escape = BattleStrategyUtil(battle_state).get_best_escape_method()
                    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                        escape = None
                    if escape is not None:
                        context.battle_decision_source = "NUZLOCKE LEVEL CAP ESCAPE"
                        context.battle_decision_detail = "no cap-legal participant can safely resolve the encounter"
                        return escape
                context.battle_decision_source = "NUZLOCKE LEVEL CAP BLOCK"
                context.battle_decision_detail = "projected encounter EXP would cross the active level cap"
                return TurnAction.switch_to_manual()
        self._last_planner_decision = None
        action = super().decide_turn(battle_state)
        if participation_hold_reason is not None:
            context.battle_decision_source = "NUZLOCKE EXP BALANCE"
            context.battle_decision_detail = participation_hold_reason
            context.message = f"LEVELING: {participation_hold_reason}{_preparation_target_suffix()}"
        if action[0] is TurnAction.RotateLead:
            self._remember_battle_support(action[1], battle_state=battle_state)

        # The generic strategy accepts a ``BEST_AVAILABLE`` planner action
        # when it cannot prove a safe line.  That is useful as a last resort,
        # but it defeats the Nuzlocke training policy when the active Pokémon
        # is the weakest member: an uncertain attack spends the weak member's
        # turn without giving it a safe path to useful experience.  Give a
        # stronger, healthy member the battle instead when one can actually
        # damage the opponent and survive the visible response.
        if (
            not getattr(battle_state, "is_double_battle", False)
            and candidate is not None
            and candidate == active_index
            and action[0] not in (TurnAction.RotateLead, TurnAction.SwitchToManual)
            and getattr(context, "battle_decision_source", None) != "PLANNER SAFE"
        ):
            support_index = self._stronger_battle_support_target(battle_state, active_index)
            if support_index is not None:
                self._remember_battle_support(support_index, battle_state=battle_state)
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_LEVEL_BALANCE_SUPPORT_SWITCH: "
                        f"active={active_index!r} candidate={support_index!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
                return TurnAction.rotate_lead(support_index)

        # A best-available attack is a comparative result, not a safety
        # result. Nuzlocke battles must not silently execute it when a wild
        # encounter has no capture value or when a trainer line has no
        # survivable continuation. Fleeing is allowed to compete with EXP;
        # preserving the party wins when the modeled risk is high.
        planner_report = getattr(self, "_last_planner_decision", None)
        if planner_report is None:
            try:
                planner_report = plan_battle_state(battle_state, default_battle_recorder.knowledge)
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                pass
        stochastic_result = getattr(planner_report, "stochastic_result", None)
        if stochastic_result is not None:
            result_complete = getattr(stochastic_result, "complete", None)
            result_safe = getattr(stochastic_result, "is_safe", None)
            result_complete = True if result_complete is None else bool(result_complete)
            result_safe = True if result_safe is None else bool(result_safe)
            if not (result_complete and result_safe):
                category = getattr(getattr(stochastic_result, "category", None), "value", "unsafe_or_uncertain")
                wild = getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False))
                if wild and not getattr(battle_state, "nuzlocke_capture_target", False):
                    catastrophic_probability = getattr(stochastic_result, "catastrophic_probability", None)
                    guaranteed_catastrophe = (
                        isinstance(catastrophic_probability, (int, float))
                        and catastrophic_probability >= 1.0
                    )
                    if (
                        _campaign_preparation_is_active()
                        and action[0] not in (TurnAction.RunAway, TurnAction.SwitchToManual)
                        and not guaranteed_catastrophe
                    ):
                        best = getattr(
                            getattr(planner_report, "best_available_candidate", None),
                            "label",
                            "best available legal action",
                        )
                        context.battle_decision_source = "NUZLOCKE PREPARATION EXPECTIMINIMAX"
                        context.battle_decision_detail = _expectiminimax_reason(
                            stochastic_result,
                            suffix=(
                                f"executing {best} because campaign preparation owns this wild battle for EXP"
                            ),
                        )
                        context.message = (
                            "NUZLOCKE PREPARATION: executing the planner's best legal wild action for EXP"
                        )
                        diagnostic_print(
                            lambda: (
                                "NUZLOCKE_ACTION_EVALUATION: preparation owns incomplete wild result; "
                                f"category={category!r} action={action!r} best_available={best!r} "
                                f"catastrophic_probability={catastrophic_probability!r}"
                            ),
                            trace=True,
                            prefix="NUZLOCKE_ACTION_EVALUATION",
                        )
                        return action
                    try:
                        escape = BattleStrategyUtil(battle_state).get_best_escape_method()
                    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                        escape = None
                    if escape is not None:
                        context.battle_decision_source = "NUZLOCKE STOCHASTIC SEARCH ESCAPE"
                        context.battle_decision_detail = (
                            f"expectiminimax category={category!r} did not establish a safe line; escaping"
                        )
                        return escape
                    # A bounded search is still an autonomous policy result.
                    # If escape is unavailable, execute its concrete legal
                    # root action and let the battle lifecycle observe the
                    # outcome. Manual mode is reserved for a result that has
                    # no executable action at all; uncertainty alone must not
                    # make a live campaign wait for player input.
                    if action[0] is not TurnAction.SwitchToManual:
                        best = getattr(
                            getattr(planner_report, "best_available_candidate", None),
                            "label",
                            "best available legal action",
                        )
                        context.battle_decision_source = "NUZLOCKE STOCHASTIC SEARCH PROJECTION"
                        context.battle_decision_detail = _expectiminimax_reason(
                            stochastic_result,
                            suffix=f"executing {best} because no escape action is available",
                        )
                        context.message = (
                            "NUZLOCKE EXPECTIMINIMAX: executing projected wild action; "
                            "battle outcome remains under campaign handling"
                        )
                        diagnostic_print(
                            lambda: (
                                "NUZLOCKE_ACTION_EVALUATION: executing projected wild action; "
                                f"category={category!r} action={action!r} escape=False"
                            ),
                            trace=True,
                            prefix="NUZLOCKE_ACTION_EVALUATION",
                        )
                        return action
                context.battle_decision_source = "NUZLOCKE STOCHASTIC SEARCH BLOCK"
                context.battle_decision_detail = (
                    f"expectiminimax category={category!r} did not establish a safe line after planner fallback"
                )
                context.message = "NUZLOCKE SAFETY: stochastic battle result was not safe to execute"
                return TurnAction.switch_to_manual()
        high_risk_fallback = self._bounded_trainer_high_risk_fallback(
            battle_state, action, planner_report
        )
        if high_risk_fallback is not None:
            return high_risk_fallback
        planner_safe = getattr(planner_report, "classification", None) is PlannerDecisionClass.SAFE
        if (
            not getattr(battle_state, "is_double_battle", False)
            and action[0] is TurnAction.UseMove
            and not planner_safe
            and getattr(context, "battle_decision_source", None) != "PLANNER SAFE"
        ):
            risk = getattr(getattr(planner_report, "risk", None), "value", "high")
            wild = getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False))
            capture_target = bool(getattr(battle_state, "nuzlocke_capture_target", False))
            zero_damage_action = _planner_move_is_zero_damage(battle_state, planner_report)
            if wild and capture_target and risk == "critical" and zero_damage_action:
                best = getattr(getattr(planner_report, "best_available_candidate", None), "label", "attack")
                context.battle_decision_source = "NUZLOCKE CAPTURE RISK"
                context.battle_decision_detail = (
                    f"capture-owned encounter permits explicit high-risk continuation for {best}; "
                    "the capture policy must weigh target value against active faint risk"
                )
                context.message = (
                    "NUZLOCKE CAPTURE RISK: continuing only under the explicit capture-value policy"
                )
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_ACTION_EVALUATION: capture-owned critical zero-damage action allowed; "
                        f"best_available={best!r} capture_target={capture_target!r}"
                    ),
                    trace=True,
                )
            if wild and not capture_target:
                try:
                    escape = BattleStrategyUtil(battle_state).get_best_escape_method()
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                    escape = None
                try:
                    hp_ratio = active.current_hp / active.total_hp if active.total_hp > 0 else 0.0
                except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                    hp_ratio = 0.0
                if escape is not None and (risk == "critical" or hp_ratio <= 0.5 and risk == "high"):
                    best = getattr(getattr(planner_report, "best_available_candidate", None), "label", "attack")
                    safest = getattr(getattr(planner_report, "best_safe_candidate", None), "label", "none")
                    context.battle_decision_source = "NUZLOCKE WILD RISK ESCAPE"
                    context.battle_decision_detail = (
                        f"capture value is absent; safest action is escape; planner safest was {safest}; "
                        f"best available was {best} "
                        f"with {risk} risk"
                    )
                    context.message = (
                        f"NUZLOCKE RISK: flee; safest={safest}; best available={best} ({risk} attack risk)"
                    )
                    diagnostic_print(
                        lambda: (
                            "NUZLOCKE_ACTION_EVALUATION: "
                            f"wild_duplicate_or_no_capture_value=True escape=True best_available={best!r} risk={risk!r}"
                        ),
                        trace=True,
                    )
                    return escape
                if risk == "critical" and zero_damage_action:
                    best = getattr(getattr(planner_report, "best_available_candidate", None), "label", "attack")
                    context.battle_decision_source = "NUZLOCKE WILD RISK STOP"
                    context.battle_decision_detail = (
                        f"selected action {best} deals no damage and has CRITICAL risk; "
                        "no executable escape was available, so stopping is safer than spending another turn"
                    )
                    context.message = (
                        "NUZLOCKE SAFETY: stopping; critical zero-damage wild action had no escape fallback"
                    )
                    diagnostic_print(
                        lambda: (
                            "NUZLOCKE_ACTION_EVALUATION: critical zero-damage wild action blocked; "
                            f"best_available={best!r} escape=False capture_target={capture_target!r}"
                        ),
                        trace=True,
                    )
                    return TurnAction.switch_to_manual()
            if not wild:
                best = getattr(getattr(planner_report, "best_available_candidate", None), "label", "attack")
                safest = getattr(getattr(planner_report, "best_safe_candidate", None), "label", "none")
                # A planner classification of BEST_AVAILABLE means that the
                # action is legal and comparatively preferable, not that it
                # is impossible. Once the switch evaluator has found no
                # acceptable replacement, continue with that action and make
                # the residual risk explicit instead of freezing the bot in
                # manual mode merely because a deterministic proof is absent.
                if action[0] is TurnAction.UseMove:
                    context.battle_decision_source = "NUZLOCKE PROBABILISTIC FALLBACK"
                    context.battle_decision_detail = (
                        f"safest planner action was {safest}; best available was {best} with {risk} risk; "
                        "no replacement met the survival threshold"
                    )
                    context.message = (
                        f"NUZLOCKE RISK: executing {best}; safest={safest}; "
                        f"best available has {risk} risk; no switch met the survival threshold"
                    )
                    diagnostic_print(
                        lambda: (
                            "NUZLOCKE_ACTION_EVALUATION: probabilistic fallback; "
                            f"action={best!r} safest={safest!r} risk={risk!r}"
                        ),
                        trace=True,
                    )
                    return action
                context.battle_decision_source = "NUZLOCKE RISK STOP"
                context.battle_decision_detail = (
                    f"safest planner action was {safest}; best available was {best} with {risk} risk; "
                    "no safe switch or battle line was available"
                )
                context.message = (
                    f"NUZLOCKE SAFETY: no autonomous non-attack line; safest={safest}; "
                    f"best available={best} ({risk} risk); manual mode"
                )
                return TurnAction.switch_to_manual()

        # The generic battle planner does not know the Nuzlocke death
        # projection. Never honor a planner switch to a healed dead Pokémon.
        if action[0] is TurnAction.RotateLead:
            replacement = get_party()[action[1]]
            active_indices = _active_party_indices(battle_state)
            replacement_is_usable = (
                _is_eligible_pokemon(replacement) and replacement.current_hp > 0 and action[1] not in active_indices
            )
            opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
            if replacement_is_usable and opponent is not None:
                try:
                    replacement_is_usable = (
                        BattleStrategyUtil(battle_state).get_strongest_move_against(replacement, opponent) is not None
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                    replacement_is_usable = False
            if not replacement_is_usable:
                util = BattleStrategyUtil(battle_state)
                fallback_index = _lowest_battle_capable_index(self, battle_state)
                if (
                    fallback_index is not None
                    and fallback_index != getattr(battle_state.own_side.active_battler, "party_index", None)
                    and util.can_switch()
                ):
                    return TurnAction.rotate_lead(fallback_index)
                move = util.get_strongest_move_against(
                    battle_state.own_side.active_battler,
                    battle_state.opponent.active_battler,
                )
                if move is not None:
                    return TurnAction.use_move(move)
        return action

    @staticmethod
    def _publish_expectiminimax_experience_facts(battle_state: BattleState, opponent=None) -> None:
        """Expose cap-aware recipient values to the normalized battle search."""

        try:
            reward = _encounter_experience_reward(battle_state, opponent=opponent)
            cap = _active_level_cap()
            values = []
            for index, pokemon in enumerate(get_party()):
                if getattr(pokemon, "is_egg", False) or getattr(pokemon, "current_hp", 0) <= 0:
                    continue
                headroom = _experience_headroom(index, cap)
                if headroom is not None:
                    values.append((index, float(max(0, headroom))))
            context.battle_experience_reward = int(reward or 0)
            context.battle_experience_values = tuple(values)
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_EXPECTIMINIMAX_EXP_FACTS: "
                    f"reward={context.battle_experience_reward!r} values={context.battle_experience_values!r}"
                ),
                trace=True,
                prefix="NUZLOCKE_EXPECTIMINIMAX_EXP_FACTS",
            )
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError, IndexError):
            context.battle_experience_reward = 0
            context.battle_experience_values = ()

    def _authorize_trainer_exp_tag(
        self,
        battle_state: BattleState,
        *,
        active_index: int | None,
        opponent,
        response_moves: tuple[str, ...],
        util: BattleStrategyUtil,
    ) -> int | None:
        """Authorize a safe EXP tag at the trainer's free-switch prompt."""

        if not self._campaign_exp_tag_enabled():
            diagnostic_print(
                lambda: "NUZLOCKE_EXP_TAG: disabled; training scheduler requires direct lead rotation",
                trace=True,
                prefix="NUZLOCKE_EXP_TAG",
            )
            return None

        plan = self._choose_exp_tag_plan_for_opponent(
            battle_state,
            active_index=active_index,
            opponent=opponent,
            response_moves=response_moves,
            util=util,
        )
        if plan is None:
            return None
        tag_index, finisher_index = plan
        self._pending_exp_tag_index = tag_index
        self._pending_exp_finisher_index = finisher_index
        context.battle_decision_source = "NUZLOCKE EXP TAG"
        context.battle_decision_detail = (
            f"free trainer replacement switch to safe tag {tag_index}; validated finisher is {finisher_index}; "
            f"exp={self._experience_projection_detail(battle_state, {tag_index, finisher_index}, opponent=opponent)}"
        )
        context.message = (
            f"TRAINER SWITCH: free EXP tag to party slot {tag_index}, "
            f"then return to slot {finisher_index}"
        )
        diagnostic_print(
            lambda: (
                "NUZLOCKE_EXP_TAG_PROMPT: "
                f"tag={tag_index!r} finisher={finisher_index!r} active={active_index!r} action='answer_yes'"
            ),
            trace=True,
        )
        self._remember_free_trainer_switch(tag_index, opponent, "free EXP tag authorization")
        return tag_index

    def _choose_exp_tag_plan(
        self,
        battle_state: BattleState,
        active_index: int | None,
        lowest_capable_index: int | None,
    ) -> tuple[int, int] | None:
        """Return a safe low-level tag and its immediate battle finisher.

        A tag is considered only when a lower-level eligible member cannot
        damage the current opponent but another legal member can.  This keeps
        the useful EXP from the weak member without turning the next turn into
        the old ``no damaging moves`` error path.
        """

        if not self._campaign_exp_tag_enabled():
            return None

        if active_index is None or lowest_capable_index is None:
            return None
        try:
            util = BattleStrategyUtil(battle_state)
            opponent = battle_state.opponent.active_battler
            if not util.can_switch():
                return None
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        known_response_moves = _known_opponent_move_names()
        # A temporary no-damage switch is allowed only when the response is
        # actually known; otherwise there is no basis for a survival claim.
        if not known_response_moves:
            return None

        return self._choose_exp_tag_plan_for_opponent(
            battle_state,
            active_index=active_index,
            opponent=opponent,
            response_moves=known_response_moves,
            util=util,
            capable_indices=set(_battle_capable_indices(self, battle_state)),
        )

    def _choose_exp_tag_plan_for_opponent(
        self,
        battle_state: BattleState,
        *,
        active_index: int | None,
        opponent,
        response_moves: tuple[str, ...],
        util: BattleStrategyUtil | None = None,
        capable_indices: set[int] | None = None,
    ) -> tuple[int, int] | None:
        """Find a safe tag and finisher for a specified opposing Pokémon."""

        if active_index is None or not response_moves:
            return None
        if any(str(name).casefold() == "pursuit" for name in response_moves):
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_EXP_TAG_REJECTED: "
                    f"opponent={_battle_pokemon_key(opponent)!r} reason='known Pursuit punishes tag switch-out'"
                ),
                trace=True,
            )
            return None
        try:
            util = util or BattleStrategyUtil(battle_state)
            if not util.can_switch():
                return None
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        if capable_indices is None:
            capable_indices = {
                index
                for index in _eligible_indices(only_non_fainted=True)
                if self.pokemon_can_battle(get_party()[index])
                and util.get_strongest_move_against(get_party()[index], opponent) is not None
            }
        level_cap = _active_level_cap()
        tag_candidates = sorted(
            (
                index
                for index in _eligible_indices(only_non_fainted=True)
                if index != active_index and index not in capable_indices
                and not self._participated_against_current_opponent(index)
                and not self._ordinary_switch_guard_blocks(battle_state, index)
            ),
            key=lambda index: _battle_training_priority(index, level_cap),
        )
        for tag_index in tag_candidates:
            tag = get_party()[tag_index]
            if not self.pokemon_can_battle(tag):
                continue
            if not self._survives_visible_response(util, opponent, tag, response_moves):
                continue
            finisher = self._battle_finisher_target(
                battle_state,
                excluded={tag_index},
                opponent=opponent,
                response_moves=response_moves,
                util=util,
            )
            if finisher is None:
                continue
            if not _participants_remain_cap_legal(
                battle_state,
                {active_index, tag_index, finisher},
                opponent=opponent,
            ):
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_EXP_TAG_REJECTED: "
                        f"tag={tag_index!r} finisher={finisher!r} reason='projected EXP exceeds active cap'"
                    ),
                    trace=True,
                )
                continue
            return tag_index, finisher
        return None

    def _battle_finisher_target(
        self,
        battle_state: BattleState,
        *,
        excluded: set[int],
        opponent=None,
        response_moves: tuple[str, ...] | None = None,
        util: BattleStrategyUtil | None = None,
    ) -> int | None:
        """Choose the preferred eligible member with a winning damage line."""

        try:
            util = util or BattleStrategyUtil(battle_state)
            opponent = opponent or battle_state.opponent.active_battler
            can_switch = util.can_switch()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
        known_response_moves = (
            tuple(response_moves)
            if response_moves is not None
            else _opponent_move_names_for_safety(battle_state)
        )
        candidates = []
        level_cap = _active_level_cap()
        for index in _eligible_indices(only_non_fainted=True):
            if index in excluded:
                continue
            pokemon = get_party()[index]
            if not self.pokemon_can_battle(pokemon):
                continue
            try:
                move_index = util.get_strongest_move_against(pokemon, opponent)
                if move_index is None:
                    continue
                damage = util.calculate_move_damage_range(pokemon.moves[move_index].move, pokemon, opponent)
                damage_min = getattr(damage, "min", getattr(damage, "max", 0))
                damage_max = getattr(damage, "max", 0)
                if damage_max <= 0 or damage_min <= 0:
                    continue
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                continue

            if known_response_moves and not self._survives_visible_response(
                util,
                opponent,
                pokemon,
                known_response_moves,
                damage_min=damage_min,
                damage_max=damage_max,
            ):
                continue
            if active_index is not None and not _participants_remain_cap_legal(
                battle_state,
                {active_index, index},
                opponent=opponent,
            ):
                continue
            if index != active_index and not can_switch:
                continue
            # EXP remaining in cap is the training priority when known; legacy
            # level/EXP ordering and damage remain tie-breakers after safety.
            candidates.append(
                (
                    int(self._participated_against_current_opponent(index)),
                    _battle_training_priority(index, level_cap),
                    damage_min >= getattr(opponent, "current_hp", 0),
                    damage_min,
                    damage_max,
                    getattr(pokemon, "current_hp", 0),
                    -index,
                    index,
                )
            )
        return (
            min(candidates, key=lambda item: (item[0], item[1], -int(item[2]), -item[3], -item[4], -item[5], item[6]))[-1]
            if candidates
            else None
        )

    @staticmethod
    def _survives_visible_response(
        util: BattleStrategyUtil,
        opponent,
        pokemon,
        move_names: tuple[str, ...],
        *,
        damage_min: int | None = None,
        damage_max: int | None = None,
        response_turns: int | None = None,
        response_timing: str = "active",
    ) -> bool:
        """Require a visible response with an acceptable survival probability."""

        response = _response_risk_for_line(
            util,
            opponent,
            pokemon,
            move_names,
            damage_min=damage_min,
            damage_max=damage_max,
            response_turns=response_turns,
            response_timing=response_timing,
        )
        return response is not None and response[3] >= _MIN_ACCEPTABLE_SURVIVAL_PROBABILITY

    def _stronger_battle_support_target(self, battle_state: BattleState, active_index: int | None) -> int | None:
        """Choose a stronger healthy member for an unsafe weak-lead battle.

        This is deliberately a battle-support policy, not a replacement for
        the planner.  The planner has already failed to establish a safe line
        for the weak lead.  A support candidate therefore needs a usable
        damaging move, and—when an opponent response has been observed—must
        survive that response (including its critical-hit ceiling).
        """

        if active_index is None or active_index < 0 or active_index >= len(get_party()):
            return None
        selected_index = self._select_switch_target(
            battle_state,
            active_index,
            require_strict_safe=True,
            prefer_lowest_level=False,
            require_stronger_than_active=True,
        )
        if selected_index is not None and not _participants_remain_cap_legal(
            battle_state, {active_index, selected_index}
        ):
            return None
        return selected_index


class NuzlockeCaptureStrategy(_NuzlockeBattleDispatch, CampaignExpectiminimaxStrategy):
    """Capture wild encounters while keeping the weakest member in front.

    Campaign capture battles historically used the legacy capture strategy,
    which meant the party's first member stayed active for every encounter.
    The campaign lead policy leaves the weakest living member at the front
    between battles; the campaign planner now owns capture, weakening,
    switching, healing, and no-ball outcomes from that boundary.
    """

    def __init__(self):
        super().__init__()
        # A capture support switch is a commitment for this wild opponent.
        # Without it, the stronger-candidate search can immediately reverse
        # the switch on the next turn and oscillate between two safe members.
        self._battle_support_index: int | None = None
        self._battle_support_response_turns: int | None = None
        self._battle_support_opponent_key: tuple | None = None
        self._voluntary_switch_commitment: tuple[tuple | None, int, int, str] | None = None
        self._switch_validation_bypass_target: int | None = None

    def allows_legacy_fallback(self) -> bool:
        """Do not fall through to the generic legacy move selector."""

        return False

    def pokemon_can_battle(self, pokemon) -> bool:
        """Return whether ``pokemon`` is a legal living capture battler."""

        return _is_eligible_pokemon(pokemon) and not pokemon.is_egg and pokemon.current_hp > 0

    def party_can_battle(self) -> bool:
        """Return whether any living eligible party member remains usable."""

        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

    # Preserve the legacy capture evaluator surface for non-campaign modes.
    # Assigning the implementations keeps them available to callers without
    # putting CatchStrategy (and therefore DefaultBattleStrategy) in the
    # campaign class MRO.
    _get_safe_weakening_move = CatchStrategy._get_safe_weakening_move
    _evaluate_capture_move = CatchStrategy._evaluate_capture_move
    _get_best_poke_ball = CatchStrategy._get_best_poke_ball
    _get_best_status_changing_move = CatchStrategy._get_best_status_changing_move
    _get_poke_ball_catch_rate_multiplier = CatchStrategy._get_poke_ball_catch_rate_multiplier
    _capture_sequence_failure_probability = staticmethod(CatchStrategy._capture_sequence_failure_probability)
    _capture_damage_outcomes = staticmethod(CatchStrategy._capture_damage_outcomes)
    _capture_penalty_cost = staticmethod(CatchStrategy._capture_penalty_cost)
    _trace_capture_evaluation = staticmethod(CatchStrategy._trace_capture_evaluation)
    _capture_ball_count = staticmethod(CatchStrategy._capture_ball_count)

    # Capture battles share the same Nuzlocke safety facts as ordinary
    # campaign battles. These aliases keep the capture-specific safety
    # bookkeeping available without delegating action selection to a legacy
    # capture strategy.
    def _risk_aware_emergency_action(self, battle_state: BattleState, active_index: int | None):
        return NuzlockeLevelBalancingBattleStrategy._risk_aware_emergency_action(self, battle_state, active_index)

    def _risk_aware_switch_target(
        self, battle_state: BattleState, active_index: int, util: BattleStrategyUtil
    ) -> int | None:
        return NuzlockeLevelBalancingBattleStrategy._risk_aware_switch_target(self, battle_state, active_index, util)

    _switch_candidate_risk = NuzlockeLevelBalancingBattleStrategy._switch_candidate_risk
    _select_switch_target = NuzlockeLevelBalancingBattleStrategy._select_switch_target
    _sync_opponent_participation = NuzlockeLevelBalancingBattleStrategy._sync_opponent_participation
    _current_opponent_participants = NuzlockeLevelBalancingBattleStrategy._current_opponent_participants

    def _experience_projection_detail(
        self,
        battle_state: BattleState,
        participant_indices: set[int] | frozenset[int],
        *,
        opponent=None,
    ) -> str:
        """Describe capture EXP without invoking the defeat EXP model."""

        return "objective=capture; expected_exp_gain=0"

    def _trace_experience_projection(self, battle_state: BattleState, active_index: int | None) -> None:
        """Record the zero-EXP outcome for a successful capture objective."""

        diagnostic_print(
            lambda: (
                "NUZLOCKE_EXP_PROJECTION: "
                f"active={active_index!r} objective='capture' expected_exp_gain=0"
            ),
            trace=True,
            prefix="NUZLOCKE_EXP_PROJECTION:",
        )

    _remember_battle_support = NuzlockeLevelBalancingBattleStrategy._remember_battle_support
    _remember_voluntary_switch = NuzlockeLevelBalancingBattleStrategy._remember_voluntary_switch
    _committed_prior_index = NuzlockeLevelBalancingBattleStrategy._committed_prior_index
    _commitment_reversal_is_necessary = NuzlockeLevelBalancingBattleStrategy._commitment_reversal_is_necessary
    _ordinary_switch_guard_blocks = NuzlockeLevelBalancingBattleStrategy._ordinary_switch_guard_blocks
    notify_pokemon_fainted = NuzlockeLevelBalancingBattleStrategy.notify_pokemon_fainted
    notify_battle_ended = NuzlockeLevelBalancingBattleStrategy.notify_battle_ended
    # The shared switch evaluator scores EXP freshness even for emergency
    # capture switches. Capture battles do not maintain the level-balancing
    # participation state, but the evaluator still needs the same safe
    # default (no recorded participation) instead of attempting to resolve a
    # method that only exists on the level-balancing strategy.
    _participated_against_current_opponent = (
        NuzlockeLevelBalancingBattleStrategy._participated_against_current_opponent
    )
    _survives_visible_response = staticmethod(NuzlockeLevelBalancingBattleStrategy._survives_visible_response)

    def _capture_encounter_loss_penalty(self, battle_state: BattleState) -> float:
        """Make a lethal weakening outcome ineligible for a legal encounter."""

        return _CAPTURE_ENCOUNTER_LOSS_PENALTY

    def _capture_party_faint_penalty(self, battle_state: BattleState) -> float:
        """Keep party preservation dominant over capture-rate improvement."""

        return _CAPTURE_PARTY_FAINT_PENALTY

    def _capture_response_profile(
        self, battle_state: BattleState
    ) -> tuple[float, int | None, int | None, str] | None:
        """Model the one opponent response before the next capture decision."""

        active = battle_state.own_side.active_battler
        opponent = battle_state.opponent.active_battler
        move_names = _opponent_move_names_for_safety(battle_state)
        if not move_names and not hasattr(opponent, "moves"):
            return 1.0, None, None, "opponent response facts unavailable; no damage modeled"
        if not move_names:
            return None
        response = _response_risk_for_line(
            BattleStrategyUtil(battle_state),
            opponent,
            active,
            move_names,
            # The capture action is evaluated one turn at a time. The next
            # ball decision occurs after at most one opponent response,
            # regardless of how many turns a full KO exchange would take.
            damage_min=1,
            damage_max=1,
            response_turns=1,
            response_timing=_RESPONSE_TIMING_ACTIVE_TURN,
        )
        if response is None:
            return None
        normal_incoming, critical_incoming, _critical_probability, survival_probability = response
        return (
            survival_probability,
            normal_incoming,
            critical_incoming,
            "modeled one response turn with normal/critical damage and accuracy",
        )

    def capture_target_is_authorized(self, battle_state: BattleState) -> bool:
        """Apply the runtime's hard no-ball eligibility decision."""

        runtime = getattr(context, "nuzlocke_runtime", None)
        if runtime is None:
            self._capture_veto_message = "Nuzlocke capture eligibility is unavailable; no Poké Ball will be thrown."
            return False
        try:
            location = get_player_avatar().map_group_and_number
            opponent_side = getattr(battle_state, "opponent", None)
            eligibility = runtime.capture_eligibility_for(
                location,
                is_wild=not battle_state.is_trainer_battle,
                is_trainer=battle_state.is_trainer_battle,
                species=tuple(
                    getattr(getattr(pokemon, "species", None), "name", "")
                    for pokemon in getattr(opponent_side, "active_battlers", ())
                    if getattr(getattr(pokemon, "species", None), "name", None)
                ) or None,
            )
            self._capture_veto_message = (
                f"Capture {'authorized' if eligibility.eligible else 'rejected'} at {location}: "
                f"{eligibility.reason}."
            )
            diagnostic_print(
                lambda: (
                    "NUZLOCKE_CAPTURE_DECISION: "
                    f"eligible={eligibility.eligible!r} location={location!r} "
                    f"species={eligibility.species!r} reason={eligibility.reason!r}"
                ),
                trace=True,
            )
            return eligibility.eligible
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
            self._capture_veto_message = (
                "Nuzlocke capture eligibility could not be resolved "
                f"({type(error).__name__}); no Poké Ball will be thrown."
            )
            diagnostic_print(lambda: f"NUZLOCKE_CAPTURE_DECISION: error={error!r} eligible=False", trace=True)
            return False

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
        """Choose the lowest-level eligible replacement, excluding actives."""

        active_indices = _active_party_indices(battle_state)
        candidates = tuple(index for index in _eligible_indices(only_non_fainted=True) if index not in active_indices)
        if not candidates:
            raise RuntimeError("no living Pokémon is available for capture battle replacement")
        return min(candidates, key=lambda index: (get_party()[index].level, index))

    def has_replacement_after_faint(self, battle_state: BattleState) -> bool:
        """Report whether a legal non-active replacement exists."""

        active_indices = _active_party_indices(battle_state)
        return any(
            index not in active_indices and self.pokemon_can_battle(get_party()[index])
            for index in _eligible_indices(only_non_fainted=True)
        )

    def choose_new_lead_after_battle(self) -> int | None:
        """Choose the weakest living member for the next encounter."""

        index = _lowest_level_index(only_non_fainted=True)
        try:
            current_lead = get_party().first_non_fainted.index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            current_lead = 0
        return index if index is not None and index != current_lead else None

    def decide_turn(self, battle_state: BattleState):
        # Capture legality, ball choice, weakening, switching, healing, and
        # escape are all planner inputs/actions now. This compatibility class
        # must not reintroduce a capture-specific selector.
        if _campaign_progression_is_active():
            return CampaignExpectiminimaxStrategy.decide_turn(self, battle_state)

        # Legacy capture lead balancing remains below as reference only.
        """Switch from the weak lead to a stronger capture-capable battler."""

        if not getattr(battle_state, "is_double_battle", False):
            active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
            active_index = getattr(active, "party_index", None)
            emergency = self._risk_aware_emergency_action(battle_state, active_index)
            if emergency is not None:
                return emergency
            if getattr(self, "_battle_support_index", None) == active_index:
                try:
                    return CatchStrategy.decide_turn(self, battle_state)
                except TypeError:
                    # CatchStrategy's ``super()`` points at DefaultBattleStrategy
                    # only when it is the actual superclass. A campaign class
                    # reuses the method for legacy compatibility, so retain the
                    # same no-ball fallback explicitly.
                    return DefaultBattleStrategy.decide_turn(self, battle_state)
            candidate = self._stronger_capture_switch_target(battle_state, active_index)
            if candidate is not None:
                profile = self._switch_candidate_risk(
                    battle_state,
                    candidate,
                    active_index=active_index,
                )
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_CAPTURE_LEVEL_BALANCE_SWITCH: "
                        f"active={active_index!r} candidate={candidate!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
                survival_detail = (
                    f"; estimated survival={profile.survival_probability:.1%}"
                    if profile is not None and profile.survival_probability is not None
                    else ""
                )
                context.battle_decision_source = "NUZLOCKE CAPTURE LEVEL BALANCE"
                context.battle_decision_detail = (
                    f"selected stronger capture battler {candidate} with a nonlethal weakening move"
                    f"{survival_detail}"
                )
                self._remember_battle_support(candidate, battle_state=battle_state)
                return TurnAction.rotate_lead(candidate)

        # Preserve CatchStrategy's authorization and no-ball behavior in
        # non-campaign modes. Campaign Progression returned above before
        # reaching this compatibility path.
        try:
            return CatchStrategy.decide_turn(self, battle_state)
        except TypeError:
            # CatchStrategy's ``super()`` assumes it is actually inherited.
            # Here it is reused as a compatibility method by a campaign-owned
            # class, so make the same fallback explicit.
            return DefaultBattleStrategy.decide_turn(self, battle_state)

    def _stronger_capture_switch_target(self, battle_state: BattleState, active_index: int | None) -> int | None:
        """Find a stronger living member that can damage the capture target.

        A capture lead is intentionally the weakest living member so the
        party order remains useful for ordinary training.  Once a battle is
        active, prefer a stronger member with a nonlethal move; if none can
        weaken safely, accept a stronger member that can at least damage the
        target and therefore resolve the encounter. Members without a usable
        move are not selected because they cannot satisfy that battle role.
        """

        if active_index is None or active_index < 0 or active_index >= len(get_party()):
            return None
        active_level = getattr(get_party()[active_index], "level", 0)
        try:
            opponent = battle_state.opponent.active_battler
            util = BattleStrategyUtil(battle_state)
            can_switch = util.can_switch()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        if not can_switch:
            return None

        known_response_moves = _opponent_move_names_for_safety(battle_state)
        candidate_facts: list[SwitchCandidateFacts] = []
        candidate_profiles: dict[int, _SwitchRisk] = {}
        for index in _eligible_indices(only_non_fainted=True):
            if index == active_index:
                continue
            pokemon = get_party()[index]
            if getattr(pokemon, "level", 0) <= active_level or not self.pokemon_can_battle(pokemon):
                _trace_switch_candidate(index, pokemon, None, "not a stronger living capture battler")
                continue
            profile = self._switch_candidate_risk(
                battle_state,
                index,
                active_index=active_index,
                util=util,
            )
            if profile is None:
                continue
            # A capture switch must be able to stay alive long enough to take
            # its weakening turn. A critical-only KO is a probability-weighted
            # risk, not an automatic rejection, but an unacceptable survival
            # probability is still a hard veto.
            if not profile.probabilistically_safe:
                _trace_switch_candidate(index, pokemon, profile, "survival probability below capture risk limit")
                continue
            safe_to_weaken = profile.damage_max < opponent.current_hp
            if not safe_to_weaken:
                _trace_switch_candidate(index, pokemon, profile, "strongest move could KO capture target")
                continue
            status_value = getattr(getattr(pokemon, "status_condition", None), "value", None)
            candidate_facts.append(
                SwitchCandidateFacts(
                    party_index=index,
                    hp=getattr(pokemon, "current_hp", 0),
                    max_hp=getattr(pokemon, "total_hp", getattr(pokemon, "current_hp", 0)),
                    opponent_hp=getattr(opponent, "current_hp", 0),
                    damage_min=profile.damage_min,
                    damage_max=profile.damage_max,
                    incoming_damage_max=profile.normal_incoming,
                    incoming_critical_damage_max=profile.critical_incoming,
                    critical_ko_probability=profile.critical_ko_probability,
                    survival_probability=profile.survival_probability,
                    strict_safe=profile.strict_safe,
                    probabilistically_safe=profile.probabilistically_safe,
                    response_facts_known=bool(known_response_moves) or profile.normal_incoming is not None,
                    fresh_participant=True,
                    training_priority=0.0,
                    healthy=status_value in (None, StatusCondition.Healthy.value),
                    poison_threat=_known_opponent_can_poison(battle_state),
                    active_party_indices=_active_party_indices(battle_state),
                )
            )
            candidate_profiles[index] = profile
        evaluations = rank_switch_candidates(candidate_facts, SwitchEvaluationContext.CAPTURE)
        evaluations = tuple(evaluation for evaluation in evaluations if evaluation.eligible)
        if not evaluations:
            return None
        selected = evaluations[0]
        profile = candidate_profiles[selected.party_index]
        _trace_switch_candidate(
            selected.party_index,
            get_party()[selected.party_index],
            profile,
            f"capture policy score={selected.score:.2f} {selected.reason}",
        )
        return selected.party_index


def _known_opponent_move_names() -> tuple[str, ...]:
    """Return only moves already visible to the human-facing battle model."""

    knowledge = getattr(default_battle_recorder, "knowledge", None)
    if knowledge is None:
        return ()
    names = {
        name
        for _, name in (
            tuple(getattr(knowledge, "observed_opponent_moves", ()))
            + tuple(getattr(knowledge, "known_opponent_moves", ()))
            + tuple(getattr(knowledge, "revealed_opponent_moves", ()))
        )
        if name
    }
    return tuple(sorted(names))


def _opponent_move_names_for_safety(battle_state: BattleState) -> tuple[str, ...]:
    """Return response moves usable by the Nuzlocke safety gate.

    The general battle planner intentionally limits itself to human-visible
    move knowledge. A Nuzlocke loss-prevention gate has a different job: the
    active battler's ROM projection is authoritative for whether an observed
    opponent can actually KO or poison it. Keep this stronger fact source
    local to the safety gate so ordinary planner confidence semantics do not
    change.
    """

    return opponent_response_move_names(
        battle_state,
        getattr(default_battle_recorder, "knowledge", None),
        include_hidden_live_moves=True,
    )


def _active_has_guaranteed_ko_before_response(
    util: BattleStrategyUtil,
    active,
    opponent,
    battle_state: BattleState,
) -> bool:
    """Return whether a guaranteed KO is ordered before every response.

    A minimum-damage KO is not automatically a safe action: a slower lead can
    still faint before its attack resolves.  Only exempt the risk gate when a
    specific guaranteed-KO move also wins the known priority/speed ordering;
    ties and unavailable ordering facts remain conservative and go through
    the switch/escape safety policy.
    """

    response_priorities = []
    for name in _opponent_move_names_for_safety(battle_state):
        try:
            response_priorities.append(getattr(get_move_by_name(name), "priority", 0))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            continue
    opponent_priority = max(response_priorities, default=0)
    opponent_speed = getattr(getattr(opponent, "stats", None), "speed", None)
    active_speed = getattr(getattr(active, "stats", None), "speed", None)
    opponent_hp = getattr(opponent, "current_hp", 0)
    if opponent_hp <= 0:
        return True

    for learned_move in getattr(active, "moves", ()):
        try:
            if (
                learned_move is None
                or learned_move.pp <= 0
                or learned_move.move.base_power <= 0
                or learned_move.move.name in context.config.battle.banned_moves
                or getattr(learned_move.move, "effect", None) == "SEMI_INVULNERABLE"
                or getattr(learned_move.move, "accuracy", 1.0) < 1
            ):
                continue
            damage = util.calculate_move_damage_range(learned_move.move, active, opponent)
            if getattr(damage, "min", 0) < opponent_hp:
                continue
            move_priority = getattr(learned_move.move, "priority", 0)
            if move_priority > opponent_priority:
                return True
            if move_priority == opponent_priority and active_speed is not None and opponent_speed is not None:
                # A speed tie is intentionally not treated as safe because
                # Gen III resolves ties randomly.
                if active_speed > opponent_speed:
                    return True
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            continue
    return False


def _opponent_response_damage(
    util: BattleStrategyUtil,
    opponent,
    active,
    battle_state: BattleState,
) -> tuple[int | None, int | None]:
    """Return normal and critical maximum response damage from ROM facts."""

    response_moves = _opponent_move_names_for_safety(battle_state)
    if not response_moves:
        return None, None
    normal_values = []
    critical_values = []
    for name in response_moves:
        try:
            move = get_move_by_name(name)
            normal_values.append(util.calculate_move_damage_range(move, opponent, active).max)
            try:
                critical_values.append(
                    util.calculate_move_damage_range(move, opponent, active, is_critical_hit=True).max
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                critical_values.append(normal_values[-1])
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            continue
    if not normal_values:
        return None, None
    return max(normal_values), max(critical_values, default=max(normal_values))


def _active_response_risk_for_line(
    util: BattleStrategyUtil,
    active,
    opponent,
    battle_state: BattleState,
) -> tuple[int, int, float, float] | None:
    """Model the incoming damage over the active battler's winning line.

    The one-turn emergency check is sufficient for a critically weakened
    Pokémon, but a healthy low-damage member can still be lost over several
    turns.  Use the same outgoing move and response model as switch
    evaluation so an active member is held to the same survival standard.
    """

    move_names = _opponent_move_names_for_safety(battle_state)
    if not move_names:
        return None
    try:
        move_index = util.get_strongest_move_against(active, opponent)
        if move_index is None:
            return None
        learned_move = active.moves[move_index]
        if learned_move is None or learned_move.pp <= 0:
            return None
        damage = util.calculate_move_damage_range(learned_move.move, active, opponent)
        damage_min = int(getattr(damage, "min", getattr(damage, "max", 0)))
        damage_max = int(getattr(damage, "max", 0))
        if damage_min <= 0 or damage_max <= 0:
            return None
        return _response_risk_for_line(
            util,
            opponent,
            active,
            move_names,
            damage_min=damage_min,
            damage_max=damage_max,
            response_timing=_RESPONSE_TIMING_ACTIVE_TURN,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return None


def _safe_battle_healing_action(
    active,
    *,
    response_damage: int | None,
    critical_response_damage: int | None,
    active_index: int,
    require_safe: bool = True,
) -> tuple[TurnAction, tuple[object, int]] | None:
    """Choose a targeted recovery item only when its turn is survivable.

    Status cures are considered independently from HP restoration. A targeted
    cure therefore beats a broad HP/status item when HP restoration is not
    needed, while a larger HP item is retained as an emergency option when a
    smaller item cannot clear the response ceiling.
    """

    if response_damage is None or active_index is None:
        return None
    required_hp = max(response_damage, critical_response_damage or response_damage)
    # Healing is resolved before the opponent's response. The active member
    # only needs to be below the modeled response ceiling before the item is
    # used; the post-heal HP must clear that ceiling. A previous guard
    # required the pre-heal HP to survive the response, which rejected the
    # exact Potion action needed when a critical hit could otherwise KO the
    # active member.
    active_can_survive = getattr(active, "current_hp", 0) > required_hp
    candidates = []
    active_status = getattr(getattr(active, "status_permanent", None), "value", None)
    try:
        for slot in get_item_bag().items:
            item = slot.item
            if (
                slot.quantity <= 0
                or getattr(item, "battle_use", None) is not ItemBattleUse.Healing
            ):
                continue
            recovery_item = _recovery_item(item, slot.quantity)
            status_progress = bool(
                active_status
                and active_status != "none"
                and (
                    recovery_item.cures_all_status
                    or active_status in recovery_item.status_healed
                )
            )
            healed_hp = (
                active.total_hp
                if recovery_item.full_heal
                else min(active.total_hp, active.current_hp + recovery_item.amount_healed)
            )
            hp_progress = healed_hp > active.current_hp
            safe_hp = hp_progress and healed_hp > required_hp
            # A status cure still spends a turn. It is safe only if the active
            # member can survive that response; a cure does not substitute
            # for HP when a response can immediately KO.
            if require_safe and status_progress and not active_can_survive and not safe_hp:
                continue
            safe_hp_action = (
                (safe_hp if require_safe else hp_progress)
                and active.current_hp <= required_hp
                and (healed_hp > required_hp if require_safe else healed_hp > active.current_hp)
            )
            if safe_hp_action or status_progress:
                missing_hp = max(0, active.total_hp - active.current_hp)
                waste = (
                    missing_hp
                    if recovery_item.full_heal
                    else max(0, recovery_item.amount_healed - missing_hp)
                )
                if require_safe:
                    # The item is selected before the opponent gets to act.
                    # For HP recovery, post-heal survival is the relevant
                    # proof. Status-only cures still require the current
                    # member to survive the response because they do not add
                    # HP.
                    if not safe_hp_action and not (status_progress and active_can_survive):
                        continue
                    candidates.append(
                        (
                            int(status_progress and not recovery_item.cures_all_status),
                            int(status_progress),
                            int(safe_hp_action),
                            -waste,
                            -recovery_item.amount_healed,
                            -slot.quantity,
                            item,
                        )
                    )
                else:
                    candidates.append(
                        (
                            int(status_progress),
                            healed_hp,
                            int(safe_hp_action),
                            -waste,
                            -recovery_item.amount_healed,
                            -slot.quantity,
                            item,
                        )
                    )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return None
    if not candidates:
        return None
    item = max(candidates)[-1]
    return TurnAction.use_item_on(item, active_index)


def _known_opponent_can_poison(battle_state: BattleState) -> bool:
    """Return whether the shared safety envelope includes a poison threat."""

    for name in _opponent_move_names_for_safety(battle_state):
        if not name:
            continue
        try:
            if BattleStrategyUtil.move_may_inflict_poison(get_move_by_name(name)):
                return True
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            continue
    try:
        ability = battle_state.opponent.active_battler.ability.name
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return False
    return ability in {"Poison Point", "Effect Spore"}


def _move_name_may_inflict_poison(move_name: str) -> bool:
    """Return poison metadata for a move name, tolerating partial battle data."""

    try:
        return BattleStrategyUtil.move_may_inflict_poison(get_move_by_name(move_name))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
        return False


def _capture_response_damage_ceiling(
    util: BattleStrategyUtil,
    opponent,
    candidate,
    move_names: tuple[str, ...],
) -> int | None:
    """Calculate the worst visible response, including a critical hit.

    ``None`` means the response could not be established from visible facts;
    it is intentionally different from zero damage for a known status move.
    """

    if not move_names:
        return None
    damage_values = []
    for move_name in move_names:
        try:
            move = get_move_by_name(move_name)
            regular = util.calculate_move_damage_range(move, opponent, candidate).max
            critical = util.calculate_move_damage_range(
                move,
                opponent,
                candidate,
                is_critical_hit=True,
            ).max
            damage_values.append(max(regular, critical))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            return None
    return max(damage_values, default=0)


def _response_risk_for_line(
    util: BattleStrategyUtil,
    opponent,
    candidate,
    move_names: tuple[str, ...],
    *,
    damage_min: int | None = None,
    damage_max: int | None = None,
    response_turns: int | None = None,
    response_timing: str = "active",
) -> tuple[int, int, float, float] | None:
    """Return response ceilings and an approximate survival probability.

    Damage ranges in the battle utility represent the Gen III random damage
    roll (roughly 85--100% of the calculated value), while critical damage is
    evaluated separately.  This function keeps the deterministic ceilings for
    explanations, but ranks lines using a small discrete model of:

    * the normal damage roll;
    * a 1/16 critical-hit chance;
    * move accuracy, including a zero-damage miss; and
    * the number of response turns expected before the candidate defeats the
      opponent.

    The opponent's most damaging visible move is used for each response. That
    is still conservative about move selection, but it no longer treats every
    critical-only ceiling as a certain KO or estimates exchanges from the
    minimum possible outgoing roll.
    """

    if not move_names:
        return None
    profiles = []
    for move_name in move_names:
        try:
            move = get_move_by_name(move_name)
            normal_range = util.calculate_move_damage_range(move, opponent, candidate)
            critical_range = util.calculate_move_damage_range(
                move,
                opponent,
                candidate,
                is_critical_hit=True,
            )
            if normal_range.max <= 0 and critical_range.max <= 0:
                continue
            accuracy = max(0.0, min(1.0, getattr(move, "accuracy", 1.0)))
            normal_min = max(0, int(getattr(normal_range, "min", normal_range.max)))
            normal_max = max(normal_min, int(normal_range.max))
            critical_min = max(0, int(getattr(critical_range, "min", critical_range.max)))
            critical_max = max(critical_min, int(critical_range.max))
            critical_probability = BattleStrategyUtil.critical_hit_chance(move)
            normal_mean = (normal_min + normal_max) / 2
            critical_mean = (critical_min + critical_max) / 2
            expected_damage = accuracy * (
                (1 - critical_probability) * normal_mean + critical_probability * critical_mean
            )
            profiles.append(
                (
                    normal_min,
                    normal_max,
                    critical_min,
                    critical_max,
                    accuracy,
                    critical_probability,
                    expected_damage,
                )
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            return None
    if not profiles:
        return 0, 0, 0.0, 1.0

    # Choose the response move by expected damage. This avoids making a low-
    # accuracy move look certain merely because its maximum damage is high.
    response_profile = max(profiles, key=lambda profile: profile[6])
    normal_min, normal_max, critical_min, critical_max, accuracy, critical_probability, _ = response_profile
    try:
        if response_turns is not None:
            exchanges = max(1, int(response_turns))
        elif damage_min is None:
            # A temporary EXP tag has no winning move by design. It only
            # needs to survive the single response turn before the finisher
            # returns, rather than pretending it will remain in battle for a
            # full multi-turn exchange.
            exchanges = 1
        else:
            opponent_hp = getattr(opponent, "current_hp", 0)
            outgoing_min = max(1, int(damage_min))
            outgoing_max = max(outgoing_min, int(damage_max or outgoing_min))
            outgoing_expected = (outgoing_min + outgoing_max) / 2
            hits = max(1, int((opponent_hp + outgoing_expected - 1) // outgoing_expected))
            candidate_speed = getattr(candidate.stats, "speed", None)
            opponent_speed = getattr(opponent.stats, "speed", None)
            if response_timing == _RESPONSE_TIMING_MID_TURN_SWITCH:
                # A switch consumes this turn's action before the incoming
                # Pokémon can act. Even a faster replacement must therefore
                # survive one response, and it receives no attack until the
                # following turn. Once that turn begins, speed can prevent
                # later responses, but the first response is unconditional.
                exchanges = hits
            else:
                # The active Pokémon can attack immediately. A faster
                # attacker therefore needs one fewer incoming response. This
                # is also the timing used by the post-KO trainer replacement
                # evaluator, whose prompt appears before the next turn.
                exchanges = (
                    max(0, hits - 1)
                    if candidate_speed is not None and opponent_speed is not None and candidate_speed > opponent_speed
                    else hits
                )
    except (AttributeError, TypeError, ValueError):
        exchanges = 1

    normal_total = normal_max * exchanges
    critical_total = critical_max * exchanges
    candidate_hp = max(0, int(getattr(candidate, "current_hp", 0)))

    # Build the response distribution.  The utility returns an inclusive
    # integer range, so each displayed damage value gets equal probability.
    # Critical and normal rolls are kept distinct even when their ranges
    # happen to overlap.
    outcomes: dict[int, float] = {}
    if accuracy < 1:
        outcomes[0] = 1 - accuracy
    normal_probability = accuracy * (1 - critical_probability)
    critical_probability = accuracy * critical_probability
    normal_count = normal_max - normal_min + 1
    critical_count = critical_max - critical_min + 1
    for amount in range(normal_min, normal_max + 1):
        outcomes[amount] = outcomes.get(amount, 0.0) + normal_probability / normal_count
    for amount in range(critical_min, critical_max + 1):
        outcomes[amount] = outcomes.get(amount, 0.0) + critical_probability / critical_count

    distribution = {0: 1.0}
    for _ in range(max(0, exchanges)):
        next_distribution: dict[int, float] = {}
        for accumulated, accumulated_probability in distribution.items():
            for damage, damage_probability in outcomes.items():
                total = accumulated + damage
                # Capping at the candidate's HP keeps the state space small
                # and makes the survival sum below explicit.
                total = min(candidate_hp, total)
                next_distribution[total] = next_distribution.get(total, 0.0) + (
                    accumulated_probability * damage_probability
                )
        distribution = next_distribution

    survival_probability = sum(
        probability for accumulated, probability in distribution.items() if accumulated < candidate_hp
    )
    if candidate_hp <= 0:
        survival_probability = 0.0

    if critical_total < candidate_hp:
        critical_ko_probability = 0.0
    elif normal_total < candidate_hp:
        critical_ko_probability = 1 - (1 - critical_probability) ** max(1, exchanges)
    else:
        critical_ko_probability = 1 - survival_probability

    return normal_total, critical_total, critical_ko_probability, survival_probability


class EmeraldIntroRivalBattleStrategy(NuzlockeLevelBalancingBattleStrategy):
    """Use the strict planner for Emerald's scripted Route 103 battle.

    This remains a distinct strategy so Campaign Progression can identify the
    fixed objective and keep legacy fallback disabled. The opening action is
    deliberately planner-owned: a hard-coded Growl sequence spends turns
    while the rival gets free actions, and critical hits ignore the attack
    reduction. The planner may still choose Growl when the current battle
    facts justify it.
    """

    def allows_legacy_fallback(self) -> bool:
        """Keep the scripted rival battle on the planner-owned path."""

        return False


class RoxanneBattleStrategy(NuzlockeLevelBalancingBattleStrategy):
    """Open a Roxanne battle with the strongest observed party matchup."""

    def __init__(self):
        super().__init__()
        self._opening_matchup_considered = False

    def decide_turn(self, battle_state: BattleState):
        # Matchup selection is part of the expectiminimax root action set.
        return super().decide_turn(battle_state)

        # Legacy opening-matchup selector retained below as reference only.
        if not self._opening_matchup_considered:
            self._opening_matchup_considered = True
            active = battle_state.own_side.active_battler
            opponent = battle_state.opponent.active_battler
            util = BattleStrategyUtil(battle_state)
            active_damage = util.get_strongest_move_against(active, opponent)
            active_max = (
                util.calculate_move_damage_range(active.moves[active_damage].move, active, opponent).max
                if active_damage is not None
                else 0
            )
            candidates = []
            for index in _eligible_indices(only_non_fainted=True):
                if index == active.party_index:
                    continue
                pokemon = get_party()[index]
                move = util.get_strongest_move_against(pokemon, opponent)
                if move is None:
                    continue
                damage = util.calculate_move_damage_range(pokemon.moves[move].move, pokemon, opponent).max
                candidates.append((damage, pokemon.current_hp, -index, index))
            if candidates and max(candidates)[0] > active_max and util.can_switch():
                return TurnAction.rotate_lead(max(candidates)[3])
        return super().decide_turn(battle_state)
