from typing import Optional

from modules.battle_state import BattleState, BattlePokemon, get_battle_state
from modules.context import context
from modules.modes._interface import BotModeError
from modules.pokemon import Pokemon, Move, LearnedMove
from modules.pokemon_party import get_party
from modules.items import get_item_bag
from modules.battle_observation import default_battle_recorder
from modules.battle_planner import (
    PlannerAction,
    PlannerConfidence,
    PlannerDecision,
    PlannerDecisionClass,
    PlannerSafety,
    plan_battle_state,
)
from modules.battle_move_policy import choose_move_replacement
from modules.battle_readiness import assess_battle_readiness, established_search_probability
from modules.trainer_checkpoints import trainer_checkpoint_for_battle
from modules.console import diagnostic_print
from ._interface import BattleStrategy, TurnAction, SafariTurnAction
from ._util import BattleStrategyUtil


def _double_battle_response_profile(util: BattleStrategyUtil, opponent, defender) -> tuple[int, int] | None:
    """Return the strongest known normal and critical response for a battler.

    Opponent move data is not present in a number of menu-bound test and
    pre-battle snapshots.  ``None`` deliberately means unknown so the joint
    planner retains its older damage/accuracy behavior for those snapshots.
    """

    moves = getattr(opponent, "moves", None)
    if not moves:
        return None
    normal = []
    critical = []
    for learned_move in moves:
        if learned_move is None or getattr(learned_move, "pp", 0) <= 0:
            continue
        move = getattr(learned_move, "move", None)
        if move is None or getattr(move, "base_power", 0) <= 0:
            continue
        try:
            damage = util.calculate_move_damage_range(move, opponent, defender)
            normal.append(max(0, int(getattr(damage, "max", 0))))
            critical_damage = util.calculate_move_damage_range(
                move, opponent, defender, is_critical_hit=True
            )
            critical.append(max(0, int(getattr(critical_damage, "max", normal[-1]))))
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            continue
    if not normal:
        return None
    return max(normal), max(critical, default=max(normal))


def _double_attack_precedes_response(attacker, move, opponent) -> bool:
    """Return whether a known guaranteed attack can finish before a response."""

    move_priority = getattr(move, "priority", 0)
    opponent_moves = getattr(opponent, "moves", ()) or ()
    opponent_priority = max((getattr(getattr(item, "move", None), "priority", 0) for item in opponent_moves), default=0)
    if move_priority != opponent_priority:
        return move_priority > opponent_priority
    attacker_speed = getattr(getattr(attacker, "stats", None), "speed", None)
    opponent_speed = getattr(getattr(opponent, "stats", None), "speed", None)
    return attacker_speed is not None and opponent_speed is not None and attacker_speed > opponent_speed


class DefaultBattleStrategy(BattleStrategy):
    def __init__(self):
        self._first_non_fainted_party_index_before_battle = get_party().first_non_fainted.index
        # Legacy capture modes retain their battle-local capture policy. The
        # campaign strategy hierarchy does not inherit this class.
        self._capture_strategy = None
        # Expose the normalized decision for lifecycle diagnostics without
        # letting another strategy rebuild or replace the selected action.
        self._last_planner_decision = None

    def party_can_battle(self) -> bool:
        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

    def is_switch_target_valid(self, battle_state: BattleState, party_index: int) -> bool:
        """Validate a switch against the current opponent before execution.

        Strategies may override this for an explicitly modeled continuation
        (for example a temporary EXP tag). The default contract keeps the
        executor from blindly sending out a party member that cannot damage
        the opponent.
        """

        return party_index in self.get_valid_switch_targets(battle_state)

    def switch_target_validation_reason(self, battle_state: BattleState, party_index: int) -> str:
        """Return the concrete default-strategy switch rejection reason."""

        try:
            active_indices = {
                battler.party_index
                for battler in battle_state.own_side.active_battlers
                if battler is not None
            }
            if party_index in active_indices:
                return "target is already active"
            party = get_party()
            if party_index < 0 or party_index >= len(party):
                return "party slot is outside the observed party"
            target = party[party_index]
            if target.is_egg:
                return "target is an egg"
            if target.current_hp <= 0:
                return "target has 0 HP"
            targets = self.get_valid_switch_targets(battle_state)
            if party_index not in targets:
                return "target has no validated matchup against the current opponent"
            return "validated"
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            return "switch validation could not read the target or matchup"

    def get_valid_switch_targets(self, battle_state: BattleState) -> tuple[int, ...]:
        """Return current-opponent-compatible switch targets."""

        try:
            return tuple(BattleStrategyUtil(battle_state).get_potential_rotation_targets(battle_state))
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return ()

    def select_valid_switch_target(self, battle_state: BattleState) -> int | None:
        """Select one target from the strategy's validated switch set."""

        targets = self.get_valid_switch_targets(battle_state)
        if not targets:
            return None
        try:
            selected = BattleStrategyUtil(battle_state).select_rotation_target(battle_state)
            return selected if selected in targets else targets[0]
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return targets[0]

    def pokemon_can_battle(self, pokemon: Pokemon) -> bool:
        if pokemon.is_egg or not self._pokemon_has_enough_hp(pokemon):
            return False

        return any(self._move_is_usable(move) for move in pokemon.moves)

    def which_move_should_be_replaced(self, pokemon: Pokemon, new_move: Move) -> int:
        if context.config.battle.new_move == "stop":
            context.message = f"{pokemon.name} is attempting to learn {new_move.name}. Switching to manual mode."
            context.set_manual_mode()
            return 4

        if context.config.battle.new_move == "cancel":
            return 4

        # Do not learn move if new move is banned.
        if new_move.name in context.config.battle.banned_moves:
            context.message = f"{new_move.name} is on the block list, so {pokemon.name} will skip learning it."
            return 4
        decision = choose_move_replacement(
            pokemon,
            new_move,
            banned_moves=context.config.battle.banned_moves,
        )
        context.message = decision.reason
        diagnostic_print(
            lambda: (
                "BATTLE_MOVE_POLICY: "
                f"pokemon={pokemon.name!r} new_move={new_move.name!r} "
                f"replacement_index={decision.replacement_index!r} "
                f"current_score={decision.current_score!r} candidate_score={decision.candidate_score!r} "
                f"reason={decision.reason!r}"
            ),
            trace=True,
        )
        return decision.replacement_index

    def should_allow_evolution(self, pokemon: Pokemon, party_index: int) -> bool:
        if context.bot_mode_instance is None:
            return True
        mode = context.bot_mode_instance
        mode_decision = mode.on_pokemon_evolving_after_battle(pokemon, party_index)
        owns_policy = getattr(mode, "owns_evolution_policy", None)
        if callable(owns_policy) and owns_policy():
            return mode_decision
        return mode_decision and not context.config.battle.stop_evolution

    def should_flee_after_faint(self, battle_state: BattleState) -> bool:
        if context.config.battle.faint_action == "stop":
            context.message = "Active Pokémon fainted. Switching to manual mode."
            context.set_manual_mode()
        elif context.config.battle.faint_action == "flee":
            if battle_state.is_trainer_battle:
                context.message = "Active Pokémon fainted. `faint_action` is set to `flee`, but this is a trainer battle. Switching to manual mode."
                context.set_manual_mode()
            return True
        elif not self.party_can_battle():
            return True
        else:
            return False

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
        new_lead: int | None = None
        new_lead_current_hp: int = 0
        party = get_party()
        for index in range(len(party)):
            pokemon = party[index]
            if battle_state.is_double_battle and (
                (
                    battle_state.own_side.left_battler is not None
                    and battle_state.own_side.left_battler.party_index == index
                )
                or (
                    battle_state.own_side.right_battler is not None
                    and battle_state.own_side.right_battler.party_index == index
                )
            ):
                continue
            if not pokemon.is_egg and pokemon.current_hp > 0 and pokemon.current_hp > new_lead_current_hp:
                new_lead = index
                new_lead_current_hp = pokemon.current_hp
        return new_lead

    def choose_new_lead_after_battle(self) -> int | None:
        party = get_party()
        if not self.pokemon_can_battle(party[self._first_non_fainted_party_index_before_battle]):
            util = BattleStrategyUtil(get_battle_state())
            return util.select_rotation_target()

        return None

    def decide_turn(self, battle_state: BattleState) -> tuple["TurnAction", Optional[any]]:
        """
        Decides the action to take for the current turn based on the battle state.
        """
        # Legacy modes retain their capture and heuristic fallbacks. Campaign
        # mode uses CampaignExpectiminimaxStrategy, which does not inherit
        # this method and therefore remains strict planner-only.
        nuzlocke_capture_target = getattr(battle_state, "nuzlocke_capture_target", False)
        diagnostic_print(
            lambda: (
                "READY_TURN_CAPTURE_CHECK: ready_turn_reached=True "
                f"nuzlocke_capture_target={nuzlocke_capture_target}"
            ),
            trace=True,
        )
        try:
            balls_available = any(ball.quantity > 0 for ball in get_item_bag().poke_balls)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            balls_available = False

        if nuzlocke_capture_target and balls_available:
            from modules.battle_strategies.catch import CatchStrategy

            if self._capture_strategy is None:
                self._capture_strategy = CatchStrategy()
            try:
                usable_balls = tuple(ball for ball in get_item_bag().poke_balls if ball.quantity > 0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                usable_balls = ()
            diagnostic_print(
                lambda: (
                    "READY_TURN_CAPTURE: CatchStrategy invoked "
                    f"usable_pokeball_found={bool(usable_balls)} "
                    f"available_balls={[getattr(ball.item, 'name', None) for ball in usable_balls]!r}"
                ),
                trace=True,
            )
            capture_action = self._capture_strategy.decide_turn(battle_state)
            diagnostic_print(
                lambda: (
                    "READY_TURN_CAPTURE_RESULT: "
                    f"action={getattr(capture_action[0], 'name', capture_action[0])!r} "
                    f"item={getattr(capture_action[1], 'name', capture_action[1])!r}"
                ),
                trace=True,
            )
            if capture_action[0] is not TurnAction.SwitchToManual:
                return capture_action

        util = BattleStrategyUtil(battle_state)
        try:
            planner_decision = plan_battle_state(battle_state, default_battle_recorder.knowledge)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError) as error:
            planner_decision = PlannerDecision(
                PlannerAction.AbortForSafety,
                confidence=PlannerConfidence.INSUFFICIENT,
                rationale=(
                    "Expectiminimax could not normalize the current battle boundary: "
                    f"{type(error).__name__}: {error}"
                ),
                safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
                evaluation_method="expectiminimax",
            )
        self._last_planner_decision = planner_decision
        stochastic_result = getattr(planner_decision, "stochastic_result", None)
        context.battle_evaluation_method = getattr(
            planner_decision,
            "evaluation_method",
            "expectiminimax",
        )
        if stochastic_result is not None and battle_state.is_trainer_battle:
            checkpoint = trainer_checkpoint_for_battle(
                battle_state,
                getattr(getattr(battle_state, "opponent", None), "active_battler", None),
            )
            context.battle_readiness = assess_battle_readiness(checkpoint, stochastic_result)
            diagnostic_print(
                lambda: (
                    "BATTLE_READINESS: "
                    f"checkpoint={getattr(checkpoint, 'checkpoint_id', None)!r} "
                    f"status={context.battle_readiness.status.value!r} "
                    f"win_probability={established_search_probability(stochastic_result, 'win_probability')!r} "
                    f"catastrophic_probability={established_search_probability(stochastic_result, 'catastrophic_probability')!r} "
                    f"reason={context.battle_readiness.reason!r}"
                ),
                trace=True,
            )
        classification = getattr(planner_decision, "classification", None)
        safe = bool(getattr(planner_decision, "is_safe_to_execute", False))
        if classification is None and safe:
            # Compatibility with decision-like test doubles and older callers
            # that predate the explicit classification field.
            classification = PlannerDecisionClass.SAFE
        planner_action = self._turn_action_from_planner_decision(planner_decision)
        confidence = getattr(planner_decision, "confidence", PlannerConfidence.INSUFFICIENT)
        best_available = classification is PlannerDecisionClass.BEST_AVAILABLE and confidence in (
            PlannerConfidence.HIGH,
            PlannerConfidence.MEDIUM,
        )
        accepted = (safe or best_available) and planner_action is not None
        diagnostic_print(
            lambda: (
                "BATTLE_PLANNER: "
                f"classification={getattr(classification, 'name', classification)} "
                f"action={planner_decision.action.value} target={planner_decision.target!r} "
                f"confidence={getattr(confidence, 'value', confidence)} is_safe_to_execute={safe} "
                f"accepted={accepted} rationale={getattr(planner_decision, 'rationale', None)!r}"
            ),
            trace=True,
        )
        if accepted:
            source = f"PLANNER {classification.name}"
            context.battle_decision_source = source
            rationale = getattr(planner_decision, "rationale", "planner-selected action")
            context.battle_decision_detail = rationale
            context.message = f"{source}: {self._planner_action_name(planner_decision)} — {rationale}"
            diagnostic_print(lambda: f"BATTLE_DECISION: source={source} returned={planner_action!r}", trace=True)
            return planner_action

        # BEST_AVAILABLE is intentionally not gated by is_safe_to_execute.
        # The legacy mode may still send its legal heuristic action when the
        # planner cannot establish a complete safety proof.
        fallback_reason = (
            "planner returned ABORT"
            if classification is PlannerDecisionClass.ABORT
            else "planner decision was not executable"
        )
        if not self.allows_legacy_fallback():
            context.battle_decision_source = "EXPECTIMINIMAX STOP"
            context.battle_decision_detail = f"{fallback_reason}; legacy fallback disabled"
            context.message = "EXPECTIMINIMAX STOP: no executable battle action was established"
            diagnostic_print(
                lambda: (
                    "BATTLE_DECISION: source=EXPECTIMINIMAX STOP "
                    f"reason={fallback_reason!r} strategy={type(self).__name__!r}"
                ),
                trace=True,
            )
            return TurnAction.switch_to_manual()
        if not util.pokemon_has_enough_hp(battle_state.own_side.active_battler):
            context.battle_decision_source = "LEGACY FALLBACK"
            context.battle_decision_detail = fallback_reason
            return self._handle_lead_cannot_battle(
                battle_state, util, reason="planner could not establish a safe action"
            )

        strongest_move = util.get_strongest_move_against(
            battle_state.own_side.active_battler, battle_state.opponent.active_battler
        )
        if strongest_move is not None:
            active = battle_state.own_side.active_battler
            move_name = "move"
            try:
                move_name = active.moves[strongest_move].move.name
            except (AttributeError, IndexError, TypeError):
                pass
            context.battle_decision_source = "LEGACY FALLBACK"
            context.battle_decision_detail = fallback_reason
            context.message = (
                f"LEGACY FALLBACK: {move_name} — {planner_decision.rationale} "
                f"(legacy selector chose move index {strongest_move})"
            )
            diagnostic_print(
                lambda: (
                    "BATTLE_DECISION: source=LEGACY FALLBACK "
                    f"returned={TurnAction.use_move(strongest_move)!r} reason={fallback_reason}"
                ),
                trace=True,
            )
            return TurnAction.use_move(strongest_move)

        context.battle_decision_source = "LEGACY FALLBACK"
        context.battle_decision_detail = fallback_reason
        context.message = f"LEGACY FALLBACK: no damaging move — {planner_decision.rationale}"
        return self._handle_lead_cannot_battle(battle_state, util, reason="No damaging moves available")

    @staticmethod
    def _planner_action_name(planner_decision) -> str:
        if planner_decision.action is PlannerAction.UseMove:
            rationale = getattr(planner_decision, "rationale", "")
            return rationale.split(":", 1)[0] or str(planner_decision.target)
        return planner_decision.action.value

    @staticmethod
    def _turn_action_from_planner_decision(planner_decision):
        """Adapt a safe planner result to the existing battle executor API."""
        match planner_decision.action:
            case PlannerAction.UseMove:
                return TurnAction.use_move(planner_decision.target)
            case PlannerAction.SwitchPokemon:
                return TurnAction.rotate_lead(planner_decision.target)
            case PlannerAction.UseItem:
                target = getattr(planner_decision, "target", None)
                if isinstance(target, tuple):
                    item, party_index = target
                    if getattr(getattr(item, "battle_use", None), "value", None) == "catch":
                        return TurnAction.use_item(item)
                    return TurnAction.use_item_on(*target)
                return TurnAction.use_item(target)
            case PlannerAction.RunAway:
                return TurnAction.run_away()
        return None

    def decide_turn_in_double_battle(self, battle_state: BattleState, battler_index: int) -> tuple["TurnAction", any]:
        """Choose one side of a coordinated, conservative double-battle plan.

        The action-selection loop asks once for the left battler and once for
        the right battler.  Rebuilding the joint assignment for each request
        keeps the policy stateless across ROM menu boundaries while still
        avoiding the common failure where both Pokémon spend the turn on the
        same already-guaranteed target.
        """

        try:
            plan = self._double_battle_plan(battle_state)
            selected = plan.get(battler_index)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError):
            selected = None

        if selected is not None:
            action, index, target_name, reason = selected
            context.battle_decision_source = "PLANNER DOUBLE"
            context.battle_decision_detail = reason
            context.message = f"PLANNER DOUBLE: {target_name} — {reason}"
            return action, index

        context.battle_decision_source = "BATTLE SAFETY STOP"
        context.battle_decision_detail = "double battle has no executable move or validated replacement"
        context.message = "BATTLE SAFETY STOP: no executable double-battle action was established"
        return TurnAction.switch_to_manual()

    def _double_battle_plan(self, battle_state: BattleState) -> dict[int, tuple]:
        """Return actions for both active player battlers.

        Candidate scoring is intentionally bounded to facts available at the
        current menu boundary.  When live opponent move facts are available,
        preserving both active battlers takes priority over damage and target
        preferences. A guaranteed KO is treated as suppressing that
        opponent's response only when speed and priority prove the KO lands
        first. Snapshots without opponent move data retain the previous
        damage/accuracy behavior.
        """

        util = BattleStrategyUtil(battle_state)
        own = (
            (0, battle_state.own_side.left_battler),
            (1, battle_state.own_side.right_battler),
        )
        opponents = tuple(
            (name, opponent)
            for name, opponent in (
                ("left opponent", battle_state.opponent.left_battler),
                ("right opponent", battle_state.opponent.right_battler),
            )
            if opponent is not None and getattr(opponent, "current_hp", 0) > 0
        )
        if not opponents:
            return {}

        candidate_sets: dict[int, list[dict]] = {}
        for logical_index, battler in own:
            if battler is None or getattr(battler, "current_hp", 0) <= 0:
                candidate_sets[logical_index] = []
                continue
            candidates = []
            for opponent_name, opponent in opponents:
                for move_index, learned_move in enumerate(getattr(battler, "moves", ())):
                    if learned_move is None or getattr(learned_move, "pp", 0) <= 0:
                        continue
                    move = getattr(learned_move, "move", None)
                    if move is None or getattr(move, "base_power", 0) <= 0:
                        continue
                    if getattr(move, "name", None) in getattr(context.config.battle, "banned_moves", ()):
                        continue
                    if (
                        getattr(battler, "disabled_move", None) is move
                        and getattr(battler, "disabled_turns_remaining", 1) > 0
                    ):
                        continue
                    try:
                        damage_range = util.calculate_move_damage_range(move, battler, opponent)
                        damage_min = max(0, int(damage_range.min))
                        damage_max = max(damage_min, int(damage_range.max))
                    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                        continue
                    accuracy = float(getattr(move, "accuracy", 1.0) or 1.0)
                    guaranteed_ko = accuracy >= 1.0 and damage_min >= getattr(opponent, "current_hp", 0)
                    candidates.append(
                        {
                            "opponent": opponent_name,
                            "move_index": move_index,
                            "move_name": getattr(move, "name", "move"),
                            "damage_min": damage_min,
                            "damage_max": damage_max,
                            "accuracy": accuracy,
                            "guaranteed_ko": guaranteed_ko,
                            "ko_before_response": guaranteed_ko
                            and _double_attack_precedes_response(battler, move, opponent),
                        }
                    )
            candidates.sort(
                key=lambda candidate: (
                    candidate["guaranteed_ko"],
                    candidate["damage_min"],
                    candidate["damage_max"],
                    candidate["accuracy"],
                    -candidate["move_index"],
                ),
                reverse=True,
            )
            candidate_sets[logical_index] = candidates

        left_candidates = candidate_sets.get(0, [])
        right_candidates = candidate_sets.get(1, [])
        response_profiles = {
            (opponent_name, logical_index): _double_battle_response_profile(util, opponent, battler)
            for opponent_name, opponent in opponents
            for logical_index, battler in own
            if battler is not None and getattr(battler, "current_hp", 0) > 0
        }

        def survival_score(left, right):
            selected = (left, right)
            known = False
            normal_incoming = {logical_index: 0 for logical_index, battler in own if battler is not None}
            critical_incoming = {logical_index: 0 for logical_index, battler in own if battler is not None}
            for opponent_name, _opponent in opponents:
                defeated_before_response = any(
                    candidate is not None
                    and candidate["opponent"] == opponent_name
                    and candidate["ko_before_response"]
                    for candidate in selected
                )
                for logical_index, battler in own:
                    if battler is None or getattr(battler, "current_hp", 0) <= 0:
                        continue
                    profile = response_profiles.get((opponent_name, logical_index))
                    if profile is None:
                        continue
                    known = True
                    if not defeated_before_response:
                        normal_incoming[logical_index] += profile[0]
                        critical_incoming[logical_index] += profile[1]
            if not known:
                return None
            normal_lethal = 0
            critical_lethal = 0
            margins = []
            for logical_index, battler in own:
                if battler is None or logical_index not in normal_incoming:
                    continue
                hp = getattr(battler, "current_hp", 0)
                margins.append(hp - normal_incoming[logical_index])
                normal_lethal += normal_incoming[logical_index] >= hp
                critical_lethal += critical_incoming[logical_index] >= hp
            return (
                -normal_lethal,
                -critical_lethal,
                min(margins, default=0),
                -sum(normal_incoming.values()),
            )

        assignments = []
        if left_candidates and right_candidates:
            for left in left_candidates:
                for right in right_candidates:
                    distinct_targets = left["opponent"] != right["opponent"]
                    survival = survival_score(left, right)
                    score = (
                        survival or (0, 0, 0, 0),
                        int(left["guaranteed_ko"]) + int(right["guaranteed_ko"]),
                        int(distinct_targets),
                        left["damage_min"] + right["damage_min"],
                        left["damage_max"] + right["damage_max"],
                        left["accuracy"] + right["accuracy"],
                    )
                    assignments.append((score, left, right))
        elif left_candidates:
            assignments.append(((int(left_candidates[0]["guaranteed_ko"]),), left_candidates[0], None))
        elif right_candidates:
            assignments.append(((int(right_candidates[0]["guaranteed_ko"]),), None, right_candidates[0]))

        chosen = max(assignments, key=lambda item: item[0]) if assignments else ((), None, None)
        selected_candidates = {0: chosen[1], 1: chosen[2]}
        plan: dict[int, tuple] = {}
        for logical_index, candidate in selected_candidates.items():
            if candidate is None:
                replacement = self._double_battle_switch_target(battle_state, logical_index, opponents)
                if replacement is not None:
                    plan[logical_index] = (
                        TurnAction.RotateLead,
                        replacement,
                        f"party slot {replacement}",
                        "no damaging move was available; selected a validated healthy replacement",
                    )
                continue
            action = (
                TurnAction.use_move_against_right_side_opponent(candidate["move_index"])
                if candidate["opponent"] == "right opponent"
                else TurnAction.use_move(candidate["move_index"])
            )
            ko_text = "guaranteed KO" if candidate["guaranteed_ko"] else "highest bounded damage line"
            survival = survival_score(selected_candidates.get(0), selected_candidates.get(1))
            survival_text = ""
            if survival is not None:
                survival_text = f"; modeled joint response survival margin {survival[2]} HP"
            plan[logical_index] = (
                action[0],
                action[1],
                candidate["opponent"],
                f"{candidate['move_name']} is the {ko_text}; "
                f"damage {candidate['damage_min']}-{candidate['damage_max']} at "
                f"{candidate['accuracy']:.0%} accuracy{survival_text}",
            )
        return plan

    @staticmethod
    def _double_battle_switch_target(battle_state, logical_index: int, opponents) -> int | None:
        active_indices = {
            getattr(battler, "party_index", None)
            for battler in (
                getattr(battle_state.own_side, "left_battler", None),
                getattr(battle_state.own_side, "right_battler", None),
            )
            if battler is not None
        }
        for index, pokemon in enumerate(get_party()):
            if index in active_indices or getattr(pokemon, "is_egg", False) or getattr(pokemon, "current_hp", 0) <= 0:
                continue
            if any(
                move is not None
                and getattr(move, "pp", 0) > 0
                and getattr(getattr(move, "move", None), "base_power", 0) > 0
                for move in getattr(pokemon, "moves", ())
            ):
                return index
        return None

    def decide_turn_in_safari_zone(self, battle_state: BattleState) -> tuple["SafariTurnAction", any]:
        return SafariTurnAction.switch_to_manual()

    def _move_is_usable(self, move: LearnedMove):
        return (
            move is not None
            and move.move.base_power > 0
            and move.pp > 0
            and move.move.name not in context.config.battle.banned_moves
        )

    def _pokemon_has_enough_hp(self, pokemon: Pokemon | BattlePokemon):
        return pokemon.current_hp_percentage > context.config.battle.hp_threshold

    def _handle_lead_cannot_battle(
        self, battle_state: BattleState, util: BattleStrategyUtil, reason: str
    ) -> tuple["TurnAction", Optional[any]]:
        """
        Handles situations where the lead Pokémon cannot battle due to low HP or lack of damaging moves.
        """
        lead_action = context.config.battle.lead_cannot_battle_action

        if lead_action == "flee":
            return self._handle_flee(battle_state, util, reason)

        if lead_action == "rotate":
            return self._handle_rotate(battle_state, util)

        raise BotModeError(f"Your Pokémon cannot battle due to {reason}, switching to manual mode...")

    def _handle_flee(
        self, battle_state: BattleState, util: BattleStrategyUtil, reason: str
    ) -> tuple["TurnAction", Optional[any]]:
        """
        Handles the flee action when the lead Pokémon cannot battle.
        """
        if battle_state.is_trainer_battle:
            raise BotModeError(
                f"Your Pokémon cannot battle due to {reason}, and 'flee' is not allowed in a trainer battle."
            )

        best_escape_method = util.get_best_escape_method()
        if best_escape_method is not None:
            return best_escape_method

        raise BotModeError(f"Your Pokémon cannot battle due to {reason}, and escape is not possible.")

    def _handle_rotate(self, battle_state: BattleState, util: BattleStrategyUtil) -> tuple["TurnAction", Optional[any]]:
        """
        Handles the rotate action when the lead Pokémon cannot battle.
        """
        if len(util.get_potential_rotation_targets(battle_state)) > 0 and util.can_switch():
            return TurnAction.rotate_lead(util.select_rotation_target(battle_state))

        raise BotModeError(
            "Your Pokémon cannot battle, 'lead_cannot_battle_action' is set to 'rotate', but no eligible Pokémon are available for switching."
        )
