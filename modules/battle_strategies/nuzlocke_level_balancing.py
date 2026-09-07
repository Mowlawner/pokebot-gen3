"""Level-balancing battle strategies that respect Nuzlocke deaths."""

from __future__ import annotations

from dataclasses import dataclass

from modules.battle_state import BattleState
from modules.battle_observation import default_battle_recorder
from modules.battle_planner import PlannerDecisionClass, plan_battle_state
from modules.battle_strategies import DefaultBattleStrategy, TurnAction
from modules.battle_strategies._util import BattleStrategyUtil
from modules.battle_strategies.catch import CatchStrategy
from modules.battle_strategies.level_balancing import LevelBalancingBattleStrategy
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
from modules.pokemon import StatusCondition, get_move_by_name
from modules.pokemon_party import get_party
from modules.player import get_player_avatar


# Utility penalties for capture decisions.  The encounter penalty is large
# enough that a small catch-rate improvement cannot justify a meaningful KO
# tail; party loss remains the dominant failure mode by design.
_CAPTURE_ENCOUNTER_LOSS_PENALTY = 8.0
_CAPTURE_PARTY_FAINT_PENALTY = 100.0
# A line that survives at least 85% of the modeled outcomes is preferable to
# stopping the bot when no deterministic line exists.  This is intentionally
# a high bar for a Nuzlocke, while still permitting a single critical-only
# exposure (93.75% survival) and similar common cases.
_MIN_ACCEPTABLE_SURVIVAL_PROBABILITY = 0.85


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
    """Estimate a conservative EXP budget for the active encounter.

    This intentionally omits participant splitting and assumes the full
    encounter reward for every participant. That is conservative for cap
    legality and covers temporary EXP-tag switches as well as the finisher.
    """

    try:
        opponent = opponent or battle_state.opponent.active_battler
        base_yield = opponent.species.base_experience_yield
        level = opponent.level
        if not isinstance(base_yield, int) or not isinstance(level, int):
            return None
        reward = max(1, (base_yield * level) // 7)
        if battle_state.is_trainer_battle:
            reward = (reward * 3) // 2
        # A participant can also receive Gen III trade/Lucky Egg modifiers.
        # Use a deliberately conservative fourfold budget rather than risk
        # authorizing a line whose actual reward crosses the next cap level.
        return max(1, reward * 4)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _participants_remain_cap_legal(
    battle_state: BattleState, participant_indices: set[int], *, opponent=None
) -> bool:
    """Check projected EXP for every member that can participate in the win."""

    cap = _active_level_cap()
    if cap is None:
        return True
    reward = _encounter_experience_reward(battle_state, opponent)
    if reward is None:
        # A cap is active but the encounter's EXP yield is not observable;
        # optional training tags must not be authorized on an unknown budget.
        return False
    from modules.nuzlocke.level_cap import can_receive_experience_without_exceeding_cap

    return all(
        can_receive_experience_without_exceeding_cap(get_party()[index], reward, cap) for index in participant_indices
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
    """Return the lowest-level living member that can damage this opponent."""

    candidates = _battle_capable_indices(strategy, battle_state)
    return min(candidates, key=_training_priority) if candidates else None


class NuzlockeLevelBalancingBattleStrategy(LevelBalancingBattleStrategy):
    """Balance living campaign Pokémon without ever sending out a dead one."""

    def __init__(self):
        super().__init__()
        # If the weak lead cannot safely continue, the inherited strategy may
        # switch to a stronger party member.  Keep that support Pokémon in
        # until the battle ends; otherwise the next turn's lowest-level
        # policy immediately switches back and the battle can ping-pong
        # between the two members without resolving.
        self._battle_support_index: int | None = None
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
                if getattr(self, "_battle_support_opponent_key", None) != opponent_key:
                    self._battle_support_index = None
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
        self._battle_support_opponent_key = opponent_key

    def _participated_against_current_opponent(self, party_index: int) -> bool:
        return party_index in getattr(self, "_battle_opponent_participants", set()) or party_index in getattr(
            self, "_battle_opponent_exp_recipients", set()
        )

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
            context.message = f"LEVELING: campaign field lead slot {selected} ({reason})"
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
        if party_index != getattr(self, "_pending_exp_tag_index", None):
            if party_index not in _eligible_indices(only_non_fainted=True):
                return False
            active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
            if active_index is None:
                return False
            profile = self._switch_candidate_risk(
                battle_state,
                party_index,
                active_index=active_index,
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
            ):
                continue
            profile = self._switch_candidate_risk(battle_state, index, active_index=active_index)
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
        if not (critical or poisoned or vulnerable_to_poison or response_can_ko or critical_only_ko) or active_can_resolve_before_response:
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

        try:
            if not util.can_switch():
                switch_allowed = False
            else:
                switch_allowed = True
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            util = None
            switch_allowed = False

        if switch_allowed and util is not None:
            target = self._risk_aware_switch_target(battle_state, active_index, util)
            if target is not None:
                reason = (
                    "poisoned active"
                    if poisoned
                    else "critical active HP"
                    if critical
                    else "known poison threat"
                    if vulnerable_to_poison
                    else "opponent response can KO active"
                )
                context.battle_decision_source = "NUZLOCKE RISK SAFETY"
                context.battle_decision_detail = (
                    f"{reason}; switched to validated damaging party slot {target}; {evaluation_summary}"
                )
                context.message = f"NUZLOCKE SAFETY: switching to party slot {target} ({reason})"
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_RISK_SAFETY: "
                        f"active={active_index!r} poisoned={poisoned!r} poison_threat={poison_threat!r} "
                        f"hp_ratio={active_ratio!r} action='switch' target={target!r}"
                    ),
                    trace=True,
                )
                # Preserve the safety replacement across the next turn.  The
                # ordinary EXP-rotation policy must not immediately put the
                # threatened Pokémon back in front just because it is lower
                # level.
                self._battle_support_index = target
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
                context.battle_decision_source = "NUZLOCKE RISK HEAL"
                context.battle_decision_detail = "battle item clears the known response damage ceiling"
                context.message = "NUZLOCKE SAFETY: healing before an avoidable faint"
                return healing_action

        if getattr(battle_state, "is_wild", not getattr(battle_state, "is_trainer_battle", False)):
            try:
                escape = BattleStrategyUtil(battle_state).get_best_escape_method()
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                escape = None
            if escape is not None:
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
        the same evaluator. The emergency caller prefers a strict-safe line,
        but can use a complete probability-weighted line when no strict-safe
        replacement exists. A member that cannot damage the opponent remains
        invalid.
        """

        return self._select_switch_target(
            battle_state,
            active_index,
            util=util,
            require_strict_safe=True,
            prefer_lowest_level=False,
            prefer_fresh_for_opponent=False,
        )

    def _switch_candidate_risk(
        self,
        battle_state: BattleState,
        party_index: int,
        *,
        active_index: int | None = None,
        util: BattleStrategyUtil | None = None,
    ) -> _SwitchRisk | None:
        """Evaluate one switch and its complete response line."""

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
                response_timing="mid_turn_switch",
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
                    f"known response can deal up to {normal_incoming} damage to "
                    f"{pokemon.current_hp} HP before its winning line completes"
                )
            elif critical_incoming >= pokemon.current_hp:
                reason = (
                    f"only a critical response can be lethal ({critical_incoming} damage); "
                    f"estimated lethal critical probability is {critical_probability:.1%}"
                )
            else:
                reason = (
                    f"survives normal and critical response ceilings "
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
    ) -> int | None:
        """Select a target through the one switch-action safety path.

        ``require_strict_safe`` retains the call-site vocabulary used by the
        earlier deterministic policy. It now means that unknown response
        facts are not accepted; a complete line may still qualify through the
        probability threshold when its only lethal outcome is sufficiently
        unlikely.
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
        priorities = tuple(_training_priority(index) for index in eligible_indices)
        policy_context = (
            SwitchEvaluationContext.SUPPORT
            if require_stronger_than_active
            else SwitchEvaluationContext.LEVELING
            if prefer_lowest_level
            else SwitchEvaluationContext.EMERGENCY
        )
        candidates: list[SwitchCandidateFacts] = []
        profiles: dict[int, _SwitchRisk] = {}
        party = get_party()
        for index in eligible_indices:
            if index == active_index:
                continue
            pokemon = party[index]
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
            training_score = relative_training_score(_training_priority(index), priorities)
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
                f"damage={profile.damage_min}-{profile.damage_max} reason={selected_evaluation.reason!r}"
            ),
            trace=True,
        )
        return selected_index

    def choose_trainer_replacement(self, battle_state: BattleState) -> int | None:
        """Choose a safe EXP-aware replacement for a trainer's next Pokémon."""

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
        context.battle_decision_detail = (
            "set battle style enabled; evaluating unexpected replacement prompt"
            if set_battle_style_enabled
            else "set battle style disabled; evaluating optional replacement prompt"
        )
        _trace_trainer_replacement_policy(
            f"prompt_visible style={'set' if set_battle_style_enabled else 'shift'} action='evaluate'"
        )
        if getattr(battle_state, "is_double_battle", False):
            # Double battles deliberately retain their existing action path.
            _trace_trainer_replacement_policy("reject reason=double_battle")
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
                return None
            # The incoming trainer Pokémon represents a new EXP opportunity.
            # Do this before ranking candidates so a member used against the
            # previous opponent is not incorrectly treated as already trained
            # for the next one.
            self._advance_to_opponent(next_opponent)
            util = BattleStrategyUtil(battle_state)
            active_indices = _active_party_indices(battle_state)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError) as error:
            _trace_trainer_replacement_policy(f"reject reason=snapshot_exception exception={error!r}")
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
        priorities = tuple(_training_priority(index) for index in eligible_indices)
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
                    training_priority=relative_training_score(_training_priority(index), priorities),
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
            return None
        selected_evaluation = evaluations[0]
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
        context.battle_decision_source = "NUZLOCKE TRAINER SWITCH"
        context.battle_decision_detail = (
            f"next opponent={getattr(next_opponent.species, 'name', next_opponent.species)!r}; "
            f"selected lowest safe cap-legal party slot {selected}; "
            f"response_facts_known={response_facts_known!r}; "
            f"score={selected_evaluation.score:.2f}"
        )
        context.message = f"TRAINER SWITCH: sending party slot {selected} into the next opponent"
        self._battle_support_index = selected
        return selected

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
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
            context.message = f"LEVELING: field lead selected party slot {selected} ({lead_policy_reason})"
        return selected

    def decide_turn(self, battle_state: BattleState):
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

        emergency = self._risk_aware_emergency_action(battle_state, active_index)
        if emergency is not None:
            return emergency

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
                action = super().decide_turn(battle_state)
                if action[0] is TurnAction.RotateLead:
                    self._battle_support_index = action[1]
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
                self._battle_support_index = safe_candidate
                self._battle_support_opponent_key = getattr(self, "_battle_opponent_key", None)
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
            context.message = f"LEVELING: {participation_hold_reason}"
        if action[0] is TurnAction.RotateLead:
            self._battle_support_index = action[1]

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
                self._battle_support_index = support_index
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

        capable_indices = set(_battle_capable_indices(self, battle_state))
        tag_candidates = sorted(
            (
                index
                for index in _eligible_indices(only_non_fainted=True)
                if index != active_index and index not in capable_indices
                and not self._participated_against_current_opponent(index)
            ),
            key=_training_priority,
        )
        for tag_index in tag_candidates:
            tag = get_party()[tag_index]
            if not self.pokemon_can_battle(tag):
                continue
            if not self._survives_visible_response(util, opponent, tag, known_response_moves):
                continue
            finisher = self._battle_finisher_target(battle_state, excluded={tag_index})
            if finisher is None:
                continue
            if not _participants_remain_cap_legal(battle_state, {active_index, tag_index, finisher}):
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

    def _battle_finisher_target(self, battle_state: BattleState, *, excluded: set[int]) -> int | None:
        """Choose the lowest-level eligible member with a winning damage line."""

        try:
            util = BattleStrategyUtil(battle_state)
            opponent = battle_state.opponent.active_battler
            can_switch = util.can_switch()
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
        known_response_moves = _opponent_move_names_for_safety(battle_state)
        candidates = []
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
            if active_index is not None and not _participants_remain_cap_legal(battle_state, {active_index, index}):
                continue
            if index != active_index and not can_switch:
                continue
            # Lower level and lower EXP progress are preferred; damage is only
            # a tie-breaker after the training objective and survival checks.
            candidates.append(
                (
                    int(self._participated_against_current_opponent(index)),
                    _training_priority(index),
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
    ) -> bool:
        """Require a visible response with an acceptable survival probability."""

        response = _response_risk_for_line(
            util,
            opponent,
            pokemon,
            move_names,
            damage_min=damage_min,
            damage_max=damage_max,
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


class NuzlockeCaptureStrategy(CatchStrategy):
    """Capture wild encounters while keeping the weakest member in front.

    Campaign capture battles historically used :class:`CatchStrategy`
    directly, which meant the party's first member stayed active for every
    encounter.  The campaign lead policy leaves the weakest living member at
    the front between battles; capture turns then switch to a stronger member
    with a usable move, preferring one that can weaken without a KO. Keep the
    capture policy (including its no-ball fallback) otherwise unchanged.
    """

    def __init__(self):
        super().__init__()
        # A capture support switch is a commitment for this wild opponent.
        # Without it, the stronger-candidate search can immediately reverse
        # the switch on the next turn and oscillate between two safe members.
        self._battle_support_index: int | None = None

    def pokemon_can_battle(self, pokemon) -> bool:
        """Return whether ``pokemon`` is a legal living capture battler."""

        return _is_eligible_pokemon(pokemon) and not pokemon.is_egg and pokemon.current_hp > 0

    def party_can_battle(self) -> bool:
        """Return whether any living eligible party member remains usable."""

        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

    # Capture battles share the same Nuzlocke safety gate as ordinary
    # campaign battles, but intentionally retain CatchStrategy's capture
    # policy and inheritance. Delegate only the shared safety helpers rather
    # than making capture strategy a subclass of the level-balancing strategy.
    def _risk_aware_emergency_action(self, battle_state: BattleState, active_index: int | None):
        return NuzlockeLevelBalancingBattleStrategy._risk_aware_emergency_action(self, battle_state, active_index)

    def _risk_aware_switch_target(
        self, battle_state: BattleState, active_index: int, util: BattleStrategyUtil
    ) -> int | None:
        return NuzlockeLevelBalancingBattleStrategy._risk_aware_switch_target(self, battle_state, active_index, util)

    _switch_candidate_risk = NuzlockeLevelBalancingBattleStrategy._switch_candidate_risk
    _select_switch_target = NuzlockeLevelBalancingBattleStrategy._select_switch_target
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
        """Make losing an eligible first encounter materially costly."""

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
        """Switch from the weak lead to a stronger capture-capable battler."""

        if not getattr(battle_state, "is_double_battle", False):
            active = getattr(getattr(battle_state, "own_side", None), "active_battler", None)
            active_index = getattr(active, "party_index", None)
            emergency = self._risk_aware_emergency_action(battle_state, active_index)
            if emergency is not None:
                return emergency
            if getattr(self, "_battle_support_index", None) == active_index:
                return super().decide_turn(battle_state)
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
                self._battle_support_index = candidate
                return TurnAction.rotate_lead(candidate)

        # CatchStrategy delegates to the ordinary battle policy when the bag
        # has no usable ball, so an empty reserve still resolves as a normal
        # defeat/escape outcome instead of freezing the controller.
        return super().decide_turn(battle_state)

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
        eligible_indices = _eligible_indices(only_non_fainted=True)
        priorities = tuple(_training_priority(index) for index in eligible_indices)
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
                _trace_switch_candidate(index, pokemon, profile, "survival probability below capture threshold")
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
                    fresh_participant=not self._participated_against_current_opponent(index),
                    training_priority=relative_training_score(_training_priority(index), priorities),
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

    names = set(_known_opponent_move_names())
    try:
        opponent = battle_state.opponent.active_battler
        names.update(
            learned_move.move.name
            for learned_move in opponent.moves
            if learned_move is not None
            and getattr(learned_move, "move", None) is not None
            and getattr(learned_move, "pp", 0) > 0
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        pass
    return tuple(sorted(name for name in names if name))


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


def _safe_battle_healing_action(
    active,
    *,
    response_damage: int | None,
    critical_response_damage: int | None,
    active_index: int,
) -> tuple[TurnAction, tuple[object, int]] | None:
    """Choose a battle healing item only when it clears known damage."""

    if response_damage is None or active_index is None:
        return None
    required_hp = max(response_damage, critical_response_damage or response_damage)
    candidates = []
    try:
        for slot in get_item_bag().items:
            item = slot.item
            if (
                slot.quantity <= 0
                or getattr(item, "battle_use", None) is not ItemBattleUse.Healing
                or getattr(item, "parameter", 0) <= 0
            ):
                continue
            healed_hp = min(getattr(active, "total_hp", 0), active.current_hp + item.parameter)
            if healed_hp > required_hp:
                candidates.append((item.parameter, -slot.quantity, item))
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return None
    if not candidates:
        return None
    item = max(candidates)[-1]
    return TurnAction.use_item_on(item, active_index)


def _known_opponent_can_poison(battle_state: BattleState) -> bool:
    """Return whether visible opponent facts include a poison threat."""

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
            if response_timing == "mid_turn_switch":
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


class EmeraldIntroRivalBattleStrategy(DefaultBattleStrategy):
    """Use Emerald's safe opening line for the scripted Route 103 battle.

    The introductory rival has a single low-level Pokémon and the player's
    starter has Growl. The generic planner cannot prove a multi-turn attack
    line when the rival's move is not human-visible, so its best-available
    fallback attacks immediately. Two opening Growls are a ROM-backed,
    trainer-battle-local safety policy: they reduce the only damaging move
    available in this fixed opening without affecting optional wild encounters.
    """

    _OPENING_GROWLS = 2

    def __init__(self):
        super().__init__()
        self._opening_growls_used = 0

    def decide_turn(self, battle_state: BattleState):
        if battle_state.is_trainer_battle and self._opening_growls_used < self._OPENING_GROWLS:
            active = battle_state.own_side.active_battler
            for index, learned_move in enumerate(active.moves):
                if (
                    learned_move is not None
                    and learned_move.move.name == "Growl"
                    and learned_move.pp > 0
                    and active.disabled_move is not learned_move.move
                ):
                    self._opening_growls_used += 1
                    context.battle_decision_source = "EMERALD INTRO RIVAL SETUP"
                    context.battle_decision_detail = f"opening Growl {self._opening_growls_used}/{self._OPENING_GROWLS}"
                    context.message = (
                        f"EMERALD INTRO RIVAL SETUP: Growl " f"{self._opening_growls_used}/{self._OPENING_GROWLS}"
                    )
                    return TurnAction.use_move(index)
        return super().decide_turn(battle_state)


class RoxanneBattleStrategy(NuzlockeLevelBalancingBattleStrategy):
    """Open a Roxanne battle with the strongest observed party matchup."""

    def __init__(self):
        super().__init__()
        self._opening_matchup_considered = False

    def decide_turn(self, battle_state: BattleState):
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
