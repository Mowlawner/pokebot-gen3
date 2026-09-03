"""Level-balancing battle strategies that respect Nuzlocke deaths."""

from __future__ import annotations

from modules.battle_state import BattleState
from modules.battle_observation import default_battle_recorder
from modules.battle_strategies import DefaultBattleStrategy, TurnAction
from modules.battle_strategies._util import BattleStrategyUtil
from modules.battle_strategies.catch import CatchStrategy
from modules.battle_strategies.level_balancing import LevelBalancingBattleStrategy
from modules.context import context
from modules.console import diagnostic_print
from modules.nuzlocke.identity import PokemonIdentity
from modules.pokemon import get_move_by_name
from modules.pokemon_party import get_party
from modules.player import get_player_avatar


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

        if party_index != getattr(self, "_pending_exp_tag_index", None):
            if party_index not in _eligible_indices(only_non_fainted=True) or not super().is_switch_target_valid(
                battle_state, party_index
            ):
                return False
            active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
            if active_index is None:
                return False
            return _participants_remain_cap_legal(battle_state, {active_index, party_index})
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
        active_index = getattr(battle_state.own_side.active_battler, "party_index", None)
        if active_index is None:
            return ()
        return tuple(
            index
            for index in super().get_valid_switch_targets(battle_state)
            if index in eligible and _participants_remain_cap_legal(battle_state, {active_index, index})
        )

    def choose_trainer_replacement(self, battle_state: BattleState) -> int | None:
        """Choose a safe EXP-aware replacement for a trainer's next Pokémon."""

        runtime = getattr(context, "nuzlocke_runtime", None)
        rule_config = getattr(runtime, "rule_config", None)
        from modules.nuzlocke.rule_config import CampaignRuleId

        if rule_config is None or not rule_config.is_enabled(CampaignRuleId.SET_BATTLE_STYLE):
            return None
        if getattr(battle_state, "is_double_battle", False):
            # Double battles deliberately retain their existing action path.
            return None

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
                return None
            util = BattleStrategyUtil(battle_state)
            active_indices = {
                battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
            }
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        response_moves = tuple(
            learned_move.move.name
            for learned_move in getattr(next_opponent, "moves", ())
            if learned_move is not None
            and getattr(learned_move, "move", None) is not None
            and getattr(learned_move, "pp", 1) > 0
        )
        if not response_moves:
            return None

        candidates = []
        current_index = getattr(battle_state.own_side.active_battler, "party_index", None)
        for index in _eligible_indices(only_non_fainted=True):
            # In a single battle the current Pokémon is a legitimate
            # "stay in" candidate.  Returning None is how the handler
            # answers the ROM prompt with No. Other active slots remain
            # excluded for safety and for future multi-battle callers.
            if index in active_indices and index != current_index:
                continue
            pokemon = get_party()[index]
            if not self.pokemon_can_battle(pokemon):
                continue
            try:
                move_index = util.get_strongest_move_against(pokemon, next_opponent)
                if move_index is None:
                    continue
                move = pokemon.moves[move_index].move
                damage = util.calculate_move_damage_range(move, pokemon, next_opponent)
                if damage.max <= 0:
                    continue
                if not self._survives_visible_response(
                    util,
                    next_opponent,
                    pokemon,
                    response_moves,
                    damage_min=damage.min,
                ):
                    continue
                if not _participants_remain_cap_legal(
                    battle_state,
                    {index},
                    opponent=next_opponent,
                ):
                    continue
                guaranteed_ko = damage.min >= next_opponent.current_hp
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                continue
            candidates.append(
                (
                    not guaranteed_ko,
                    _training_priority(index),
                    -damage.min,
                    -damage.max,
                    -getattr(pokemon, "current_hp", 0),
                    index,
                )
            )

        if not candidates:
            return None
        selected = min(candidates)[-1]
        if selected == current_index:
            context.battle_decision_source = "NUZLOCKE TRAINER SWITCH"
            context.battle_decision_detail = "current Pokémon is the safest eligible response"
            return None
        context.battle_decision_source = "NUZLOCKE TRAINER SWITCH"
        context.battle_decision_detail = (
            f"next opponent={getattr(next_opponent.species, 'name', next_opponent.species)!r}; "
            f"selected lowest safe cap-legal party slot {selected}"
        )
        context.message = f"TRAINER SWITCH: sending party slot {selected} into the next opponent"
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
        active_indices = {
            battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
        }
        eligible_non_active = tuple(
            index for index in _eligible_indices(only_non_fainted=True) if index not in active_indices
        )
        if not eligible_non_active:
            return False
        return any(index in _battle_capable_indices(self, battle_state) for index in eligible_non_active)

    def choose_new_lead_after_battle(self) -> int | None:
        index = _lowest_level_index(only_non_fainted=True)
        try:
            current_lead = get_party().first_non_fainted.index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            current_lead = 0
        selected = index if index is not None and index != current_lead else None
        diagnostic_print(
            lambda: (
                "NUZLOCKE_LEVEL_BALANCE_AFTER_BATTLE: "
                f"candidate={index!r} current_lead={current_lead!r} selected={selected!r} "
                f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
            ),
            trace=True,
        )
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

        # Lead rotation after the previous battle is a useful fallback, but
        # it cannot cover a newly captured member or a battle that begins
        # before the post-battle menu handoff is available.  Make the policy
        # explicit at every single-battle turn boundary: use the lowest-level
        # living, battle-capable member whenever switching is legal.
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
            if (
                candidate is not None
                and candidate != active_index
                and can_switch
                and active_index is not None
                and _participants_remain_cap_legal(battle_state, {active_index, candidate})
            ):
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_LEVEL_BALANCE_SWITCH: "
                        f"active={active_index!r} candidate={candidate!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
                return TurnAction.rotate_lead(candidate)

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
        action = super().decide_turn(battle_state)
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

        # The generic battle planner does not know the Nuzlocke death
        # projection. Never honor a planner switch to a healed dead Pokémon.
        if action[0] is TurnAction.RotateLead:
            replacement = get_party()[action[1]]
            active_indices = {
                battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
            }
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
        known_response_moves = _known_opponent_move_names()
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
                util, opponent, pokemon, known_response_moves, damage_min=damage_min
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
            min(candidates, key=lambda item: (item[0], -int(item[1]), -item[2], -item[3], -item[4], item[5]))[-1]
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
    ) -> bool:
        """Require a visible response to leave the candidate alive."""

        incoming = _capture_response_damage_ceiling(util, opponent, pokemon, move_names)
        if incoming is None or incoming >= getattr(pokemon, "current_hp", 0):
            return False
        if damage_min is None:
            return True
        opponent_hp = getattr(opponent, "current_hp", 0)
        hits = (opponent_hp + damage_min - 1) // damage_min if damage_min > 0 else 0
        if hits <= 0:
            return False
        # If the candidate is slower, the opponent can respond before each
        # strike. If it is faster, the final KO prevents the last response.
        try:
            player_speed = pokemon.stats.speed
            opponent_speed = opponent.stats.speed
            exchanges = max(0, hits - 1) if player_speed > opponent_speed else hits
        except (AttributeError, TypeError, ValueError):
            exchanges = hits
        return incoming * exchanges < getattr(pokemon, "current_hp", 0)

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
        try:
            opponent = battle_state.opponent.active_battler
            util = BattleStrategyUtil(battle_state)
            if not util.can_switch():
                return None
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None

        active_level = getattr(get_party()[active_index], "level", 0)
        known_response_moves = _known_opponent_move_names()
        candidates = []
        for index in _eligible_indices(only_non_fainted=True):
            if index == active_index:
                continue
            pokemon = get_party()[index]
            if getattr(pokemon, "level", 0) <= active_level or not self.pokemon_can_battle(pokemon):
                continue
            try:
                move_index = util.get_strongest_move_against(pokemon, opponent)
                if move_index is None:
                    continue
                damage = util.calculate_move_damage_range(pokemon.moves[move_index].move, pokemon, opponent)
                if damage.max <= 0:
                    continue
                incoming = _capture_response_damage_ceiling(
                    util,
                    opponent,
                    pokemon,
                    known_response_moves,
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
                continue

            response_is_known = bool(known_response_moves) and incoming is not None
            response_is_survivable = response_is_known and incoming < pokemon.current_hp
            if known_response_moves and not response_is_survivable:
                continue
            if not _participants_remain_cap_legal(battle_state, {active_index, index}):
                continue
            candidates.append(
                (
                    damage.min >= opponent.current_hp,
                    response_is_survivable,
                    damage.min,
                    damage.max,
                    getattr(pokemon, "level", 0),
                    pokemon.current_hp,
                    -index,
                    index,
                )
            )
        return max(candidates)[-1] if candidates else None


class NuzlockeCaptureStrategy(CatchStrategy):
    """Capture wild encounters while keeping the weakest member in front.

    Campaign capture battles historically used :class:`CatchStrategy`
    directly, which meant the party's first member stayed active for every
    encounter.  The campaign lead policy leaves the weakest living member at
    the front between battles; capture turns then switch to a stronger member
    with a usable move, preferring one that can weaken without a KO. Keep the
    capture policy (including its no-ball fallback) otherwise unchanged.
    """

    def pokemon_can_battle(self, pokemon) -> bool:
        """Return whether ``pokemon`` is a legal living capture battler."""

        return _is_eligible_pokemon(pokemon) and not pokemon.is_egg and pokemon.current_hp > 0

    def party_can_battle(self) -> bool:
        """Return whether any living eligible party member remains usable."""

        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

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

        active_indices = {
            battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
        }
        candidates = tuple(index for index in _eligible_indices(only_non_fainted=True) if index not in active_indices)
        if not candidates:
            raise RuntimeError("no living Pokémon is available for capture battle replacement")
        return min(candidates, key=lambda index: (get_party()[index].level, index))

    def has_replacement_after_faint(self, battle_state: BattleState) -> bool:
        """Report whether a legal non-active replacement exists."""

        active_indices = {
            battler.party_index for battler in battle_state.own_side.active_battlers if battler is not None
        }
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
            candidate = self._stronger_capture_switch_target(battle_state, active_index)
            if candidate is not None:
                diagnostic_print(
                    lambda: (
                        "NUZLOCKE_CAPTURE_LEVEL_BALANCE_SWITCH: "
                        f"active={active_index!r} candidate={candidate!r} "
                        f"party_levels={[getattr(pokemon, 'level', None) for pokemon in get_party()]!r}"
                    ),
                    trace=True,
                )
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

        known_response_moves = _known_opponent_move_names()
        candidates = []
        for index in _eligible_indices(only_non_fainted=True):
            if index == active_index:
                continue
            pokemon = get_party()[index]
            if getattr(pokemon, "level", 0) <= active_level or not self.pokemon_can_battle(pokemon):
                continue
            try:
                move_index = util.get_strongest_move_against(pokemon, opponent)
                if move_index is None:
                    continue
                move = pokemon.moves[move_index].move
                damage = util.calculate_move_damage_range(move, pokemon, opponent)
                if damage.max <= 0:
                    continue
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                continue
            safe_to_weaken = damage.max < opponent.current_hp
            incoming_damage = _capture_response_damage_ceiling(
                util,
                opponent,
                pokemon,
                known_response_moves,
            )
            response_is_known = bool(known_response_moves) and incoming_damage is not None
            response_is_survivable = response_is_known and incoming_damage < pokemon.current_hp
            candidates.append(
                (
                    safe_to_weaken,
                    response_is_survivable,
                    response_is_known,
                    pokemon.level,
                    pokemon.current_hp,
                    -damage.max,
                    -index,
                    index,
                )
            )
        # If an opponent move has already been observed, do not knowingly
        # switch into a replacement that the observed response can KO.  When
        # no response is visible yet, retain the conservative fallback of
        # choosing the strongest usable member with the best weakening line;
        # the battle planner cannot manufacture move knowledge either.
        if known_response_moves:
            candidates = [candidate for candidate in candidates if candidate[1]]
        return max(candidates)[-1] if candidates else None


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
