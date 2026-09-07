from dataclasses import dataclass

from modules.battle_state import BattleState, TemporaryStatus
from modules.battle_strategies import (
    SafariTurnAction,
    DefaultBattleStrategy,
    BattleStrategyUtil,
)
from modules.battle_strategies import TurnAction
from modules.context import context
from modules.console import diagnostic_print
from modules.items import Item, get_item_bag, PokeblockType
from modules.map import get_map_data_for_current_position
from modules.pokedex import get_pokedex
from modules.pokemon import Pokemon, get_type_by_name, StatusCondition, get_opponent
from modules.pokeblock_feeder import get_active_pokeblock_feeder_for_location
from modules.safari_strategy import (
    get_safari_strategy_action,
    is_watching_carefully,
    get_safari_balls_left,
    get_lowest_feel_any_pokeblock,
    get_lowest_feel_excluding_type,
    get_lowest_feel_pokeblock_by_type,
    get_baiting_state,
    PokeblockState,
    RSESafariStrategy,
)


@dataclass(frozen=True)
class CaptureMoveEvaluation:
    """Expected result of spending one turn on a capture weakening move."""

    move_index: int
    move_name: str
    damage_min: int
    damage_max: int
    critical_damage_min: int
    critical_damage_max: int
    accuracy: float
    critical_hit_probability: float
    expected_capture_probability: float
    target_ko_probability: float
    active_survival_probability: float
    active_faint_probability: float
    utility: float
    response_reason: str


class CatchStrategy(DefaultBattleStrategy):

    def __init__(self):
        super().__init__()
        self._current_catching_strategy_index = 0
        self._is_current_catching_strategy_baited = False
        self._number_of_balls_strategy = 0
        self._has_been_baited = False
        self._has_been_rocked = False
        self._pokeblock_state = None
        self._given_pokeblock = None

    def pokemon_can_battle(self, pokemon: Pokemon) -> bool:
        return not pokemon.is_egg and pokemon.current_hp > 0

    def should_flee_after_faint(self, battle_state: BattleState) -> bool:
        return False

    def capture_target_is_authorized(self, battle_state: BattleState) -> bool:
        """Return whether this strategy may use a ball on the battle."""

        return True

    def decide_turn(self, battle_state: BattleState) -> tuple["TurnAction", any]:
        if not self.capture_target_is_authorized(battle_state):
            # This is a hard veto: a caller may have mounted CatchStrategy
            # during a battle-start race, but it must never turn an ineligible
            # encounter into a capture merely because a ball is available.
            context.message = getattr(self, "_capture_veto_message", None) or (
                "Capture strategy rejected this encounter; no Poké Ball will be thrown."
            )
            diagnostic_print(
                lambda: f"CAPTURE_BALL_VETO: reason={context.message!r}",
                trace=True,
            )
            return DefaultBattleStrategy.decide_turn(self, battle_state)
        ball_to_throw = self._get_best_poke_ball(battle_state)
        if ball_to_throw is None:
            context.message = "Player does not have any Poké Balls, cannot catch."
            # An empty capture inventory is a normal encounter outcome.  Do
            # not hand control to a human here: continue with the ordinary
            # battle policy so the encounter can be defeated or escaped.
            return super().decide_turn(battle_state)

        # The chance of a Pokémon being caught increases if it has a status condition (sleeping,
        # paralysed, poisoned, burned, frozen.) If possible, we will try to inflict a status
        # condition to increase catch odds.
        if battle_state.opponent.active_battler.status_permanent == StatusCondition.Healthy:
            catch_success_chance = BattleStrategyUtil(battle_state).calculate_catch_success_chance(
                battle_state,
                self._get_poke_ball_catch_rate_multiplier(battle_state, ball_to_throw),
            )

            # Only bother inflicting a status condition if the chance of the opponent being caught
            # in one turn is less than 50%, otherwise just throw balls and hope for the best.
            if catch_success_chance < 0.5:
                status_move = self._get_best_status_changing_move(battle_state)
                if status_move is not None:
                    status_name = getattr(
                        battle_state.own_side.active_battler.moves[status_move].move,
                        "name",
                        f"move {status_move}",
                    )
                    context.battle_decision_source = "CAPTURE POLICY"
                    context.battle_decision_detail = (
                        f"capture objective: use {status_name} before throwing; "
                        f"one-ball catch chance={catch_success_chance:.1%}"
                    )
                    return TurnAction.use_move(status_move)

        # Preserve a legal encounter while balls remain.  The capture action
        # evaluator compares weakening outcomes against throwing a ball now,
        # including critical-hit and response risk.
        weakening_move = self._get_safe_weakening_move(battle_state, ball_to_throw)
        if weakening_move is not None:
            return TurnAction.use_move(weakening_move)

        if context.battle_decision_detail is None:
            context.battle_decision_source = "CAPTURE POLICY"
            context.battle_decision_detail = "capture objective: throw a Poké Ball; no weakening action improved expected value"
        return TurnAction.use_item(ball_to_throw)

    def _get_safe_weakening_move(
        self, battle_state: BattleState, ball_to_throw: Item | None = None
    ) -> int | None:
        """Choose a weakening move only when its expected value beats a ball.

        The default capture policy uses an effectively infinite encounter-loss
        penalty and therefore remains strictly nonlethal. Nuzlocke capture
        overrides the penalty and response-risk hooks to permit a very small,
        explicitly modeled risk when the expected capture benefit justifies it.
        """

        active = battle_state.own_side.active_battler
        util = BattleStrategyUtil(battle_state)
        ball_multiplier = (
            self._get_poke_ball_catch_rate_multiplier(battle_state, ball_to_throw)
            if ball_to_throw is not None
            else 1.0
        )

        ball_probability = util.calculate_catch_success_chance(
            battle_state,
            ball_multiplier,
        )
        ball_response = self._capture_response_profile(battle_state)
        if ball_response is None:
            ball_survival, _, _, ball_response_reason = 1.0, None, None, "response facts unavailable"
        else:
            ball_survival, _, _, ball_response_reason = ball_response
        ball_active_faint_probability = max(0.0, 1.0 - ball_survival) * max(0.0, 1.0 - ball_probability)
        ball_utility = ball_probability - self._capture_penalty_cost(
            self._capture_party_faint_penalty(battle_state), ball_active_faint_probability
        )

        candidates: list[CaptureMoveEvaluation] = []
        for index, learned_move in enumerate(getattr(active, "moves", ())):
            evaluation = self._evaluate_capture_move(
                battle_state,
                index,
                learned_move,
                ball_to_throw,
                util,
            )
            if evaluation is None:
                self._trace_capture_evaluation(index, learned_move, None, "unavailable or not modeled")
                continue
            candidates.append(evaluation)
            self._trace_capture_evaluation(index, learned_move, evaluation, "evaluated")

        if not candidates:
            context.battle_decision_source = "CAPTURE POLICY"
            context.battle_decision_detail = (
                f"capture objective: throw {getattr(ball_to_throw, 'name', ball_to_throw)}; "
                "no modeled weakening move was available"
            )
            return None

        best = max(candidates, key=lambda candidate: (candidate.utility, -candidate.target_ko_probability, -candidate.move_index))
        if best.utility <= ball_utility:
            context.battle_decision_source = "CAPTURE POLICY"
            context.battle_decision_detail = (
                f"capture objective: throw {getattr(ball_to_throw, 'name', ball_to_throw)}; "
                f"ball value={ball_utility:.1%}, best {best.move_name} value={best.utility:.1%}; "
                f"target KO risk={best.target_ko_probability:.1%}; "
                f"active survival={best.active_survival_probability:.1%}"
            )
            diagnostic_print(
                lambda: (
                    "BATTLE_CAPTURE_DECISION: action='ball' "
                    f"ball={getattr(ball_to_throw, 'name', ball_to_throw)!r} "
                    f"ball_probability={ball_probability:.4f} ball_utility={ball_utility:.4f} "
                    f"best_move={best.move_name!r} best_utility={best.utility:.4f} "
                    f"target_ko_probability={best.target_ko_probability:.4f} "
                    f"response={ball_response_reason!r}"
                ),
                trace=True,
                prefix="BATTLE_CAPTURE_DECISION:",
            )
            return None

        context.battle_decision_source = "CAPTURE POLICY"
        context.battle_decision_detail = (
            f"capture objective: weaken with {best.move_name}; "
            f"expected capture={best.expected_capture_probability:.1%} vs ball={ball_probability:.1%}; "
            f"target KO risk={best.target_ko_probability:.1%}; "
            f"active survival={best.active_survival_probability:.1%}"
        )
        diagnostic_print(
            lambda: (
                "BATTLE_CAPTURE_DECISION: action='weaken' "
                f"move={best.move_name!r} expected_capture={best.expected_capture_probability:.4f} "
                f"ball_probability={ball_probability:.4f} utility={best.utility:.4f} "
                f"target_ko_probability={best.target_ko_probability:.4f} "
                f"active_survival_probability={best.active_survival_probability:.4f}"
            ),
            trace=True,
            prefix="BATTLE_CAPTURE_DECISION:",
        )
        return best.move_index

    def _capture_encounter_loss_penalty(self, battle_state: BattleState) -> float:
        """Penalty for KO'ing a capture target before it can be caught."""

        return float("inf")

    def _capture_party_faint_penalty(self, battle_state: BattleState) -> float:
        """Penalty for losing the active Pokémon during capture preparation."""

        return float("inf")

    def _capture_response_profile(
        self, battle_state: BattleState
    ) -> tuple[float, int | None, int | None, str] | None:
        """Return survival probability and response ceilings for one response turn."""

        return 1.0, None, None, "opponent response not modeled by generic capture strategy"

    def _evaluate_capture_move(
        self,
        battle_state: BattleState,
        index: int,
        learned_move,
        ball_to_throw: Item,
        util: BattleStrategyUtil,
    ) -> CaptureMoveEvaluation | None:
        if learned_move is None or learned_move.pp == 0:
            return None
        active = battle_state.own_side.active_battler
        opponent = battle_state.opponent.active_battler
        if getattr(active, "disabled_move", None) is learned_move.move:
            return None
        try:
            normal = util.calculate_move_damage_range(learned_move.move, active, opponent)
            critical = util.calculate_move_damage_range(
                learned_move.move,
                active,
                opponent,
                is_critical_hit=True,
            )
            damage_min = max(0, int(getattr(normal, "min", normal.max)))
            damage_max = max(damage_min, int(normal.max))
            critical_damage_min = max(0, int(getattr(critical, "min", critical.max)))
            critical_damage_max = max(critical_damage_min, int(critical.max))
            accuracy = max(0.0, min(1.0, float(getattr(learned_move.move, "accuracy", 1.0))))
            critical_hit_probability = BattleStrategyUtil.critical_hit_chance(learned_move.move)
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return None
        if max(damage_max, critical_damage_max) <= 0 or accuracy <= 0:
            return None

        response = self._capture_response_profile(battle_state)
        if response is None:
            return None
        active_survival, normal_incoming, critical_incoming, response_reason = response
        active_survival = max(0.0, min(1.0, active_survival))
        opponent_hp = max(0, int(opponent.current_hp))
        ball_multiplier = (
            self._get_poke_ball_catch_rate_multiplier(battle_state, ball_to_throw)
            if ball_to_throw is not None
            else 1.0
        )
        outcomes = self._capture_damage_outcomes(
            damage_min,
            damage_max,
            critical_damage_min,
            critical_damage_max,
            accuracy,
            critical_hit_probability,
        )
        expected_capture = 0.0
        target_ko_probability = 0.0
        active_faint_probability = 0.0
        for damage, probability in outcomes:
            hp_after = max(0, opponent_hp - damage)
            if hp_after <= 0:
                target_ko_probability += probability
                continue
            expected_capture += probability * active_survival * util.calculate_catch_success_chance(
                battle_state,
                ball_multiplier,
                current_hp=hp_after,
                status=getattr(opponent, "status_permanent", StatusCondition.Healthy),
            )
            active_faint_probability += probability * (1.0 - active_survival)

        utility = (
            expected_capture
            - self._capture_penalty_cost(
                self._capture_encounter_loss_penalty(battle_state), target_ko_probability
            )
            - self._capture_penalty_cost(
                self._capture_party_faint_penalty(battle_state), active_faint_probability
            )
        )
        return CaptureMoveEvaluation(
            index,
            getattr(learned_move.move, "name", f"move {index}"),
            damage_min,
            damage_max,
            critical_damage_min,
            critical_damage_max,
            accuracy,
            critical_hit_probability,
            expected_capture,
            target_ko_probability,
            active_survival,
            active_faint_probability,
            utility,
            (
                f"response={response_reason}; incoming={normal_incoming}/{critical_incoming}"
            ),
        )

    @staticmethod
    def _capture_penalty_cost(penalty: float, probability: float) -> float:
        """Avoid ``inf * 0`` while retaining hard penalties when risk exists."""

        if probability <= 0:
            return 0.0
        return penalty * probability

    @staticmethod
    def _capture_damage_outcomes(
        damage_min: int,
        damage_max: int,
        critical_damage_min: int,
        critical_damage_max: int,
        accuracy: float,
        critical_hit_probability: float,
    ) -> tuple[tuple[int, float], ...]:
        """Approximate Gen III miss, normal-roll, and critical-roll outcomes."""

        outcomes: dict[int, float] = {}
        if accuracy < 1.0:
            outcomes[0] = 1.0 - accuracy
        normal_probability = accuracy * (1.0 - critical_hit_probability)
        critical_probability = accuracy * critical_hit_probability
        for damage in range(damage_min, damage_max + 1):
            outcomes[damage] = outcomes.get(damage, 0.0) + normal_probability / (damage_max - damage_min + 1)
        for damage in range(critical_damage_min, critical_damage_max + 1):
            outcomes[damage] = outcomes.get(damage, 0.0) + critical_probability / (
                critical_damage_max - critical_damage_min + 1
            )
        return tuple(outcomes.items())

    @staticmethod
    def _trace_capture_evaluation(index: int, learned_move, evaluation: CaptureMoveEvaluation | None, result: str) -> None:
        diagnostic_print(
            lambda: (
                "BATTLE_CAPTURE_CANDIDATE: "
                f"move_index={index!r} move={getattr(getattr(learned_move, 'move', learned_move), 'name', None)!r} "
                f"result={result!r} damage={getattr(evaluation, 'damage_min', None)!r}-"
                f"{getattr(evaluation, 'damage_max', None)!r} critical_damage="
                f"{getattr(evaluation, 'critical_damage_min', None)!r}-"
                f"{getattr(evaluation, 'critical_damage_max', None)!r} "
                f"critical_hit_probability={getattr(evaluation, 'critical_hit_probability', None)!r} "
                f"expected_capture={getattr(evaluation, 'expected_capture_probability', None)!r} "
                f"target_ko_probability={getattr(evaluation, 'target_ko_probability', None)!r} "
                f"active_survival={getattr(evaluation, 'active_survival_probability', None)!r} "
                f"utility={getattr(evaluation, 'utility', None)!r} "
                f"reason={getattr(evaluation, 'response_reason', None)!r}"
            ),
            trace=True,
            prefix="BATTLE_CAPTURE_CANDIDATE:",
        )

    def decide_turn_in_double_battle(self, battle_state: BattleState, battler_index: int) -> tuple["TurnAction", any]:
        return self.decide_turn(battle_state)

    def decide_turn_in_safari_zone(self, battle_state: BattleState) -> tuple["SafariTurnAction", any]:
        """
        Determines the next action in the Safari Zone based on the game.

        Parameters:
            battle_state (BattleState): The current battle state in Safari Zone.

        Returns:
            tuple[SafariTurnAction, any]: The next Safari action to perform and any additional context.
        """
        if context.rom.is_rse:
            return self._decide_turn_safari_rse(battle_state)
        elif context.rom.is_frlg:
            return self._decide_turn_safari_frlg(battle_state)
        return SafariTurnAction.switch_to_manual()

    def _decide_turn_safari_rse(self, battle_state: BattleState) -> tuple["SafariTurnAction", any]:
        """
        Handles the turn decision for RSE games.
        """
        if RSESafariStrategy.should_start_pokeblock_strategy(get_opponent()):
            if battle_state.current_turn == 0:
                feeder = get_active_pokeblock_feeder_for_location()
                if feeder:
                    flavor_str = feeder.pokeblock.type.value
                    pokeblock_index, pokeblock = get_lowest_feel_pokeblock_by_type(flavor_str)
                else:
                    pokeblock_index, pokeblock = get_lowest_feel_any_pokeblock()
                if pokeblock_index is None:
                    return SafariTurnAction.ThrowBall, None

                self._pokeblock_state = get_baiting_state(pokeblock)
                self._given_pokeblock = pokeblock.type.value

                return SafariTurnAction.Pokeblock, pokeblock_index

            if battle_state.current_turn == 1:
                if self._pokeblock_state == PokeblockState.IGNORED:
                    excluded_type = PokeblockType(self._given_pokeblock)
                    pokeblock_index, pokeblock = get_lowest_feel_excluding_type(excluded_type)
                    if pokeblock_index is None:
                        return SafariTurnAction.ThrowBall, None
                    return SafariTurnAction.Pokeblock, pokeblock_index

        return SafariTurnAction.ThrowBall, None

    def _decide_turn_safari_frlg(self, battle_state: BattleState) -> tuple["SafariTurnAction", any]:
        """
        Handles the turn decision for FRLG games based on the watching/rocked/baited status.
        """
        if self._is_new_strategy_required():
            if not self._has_been_baited:
                return self._start_new_baited_strategy()
            else:
                return self._start_new_continuing_baited_strategy()
        return self._continue_current_strategy()

    def _is_new_strategy_required(self) -> bool:
        """
        Checks if a new catching strategy should be initiated.
        """
        return is_watching_carefully() and not self._has_been_rocked

    def _start_new_baited_strategy(self) -> tuple["SafariTurnAction", any]:
        """
        Initiates a new baited catch strategy and returns the next action.
        """
        self._number_of_balls_strategy = get_safari_balls_left()
        self._has_been_baited = True

        action, rocked = self._execute_strategy_action()

        if rocked:
            self._has_been_rocked = rocked

        self._current_catching_strategy_index += 1

        return action, None

    def _start_new_continuing_baited_strategy(self) -> tuple["SafariTurnAction", any]:
        """
        Initiates a new baited catch strategy after first baited one and returns the next action.
        """
        self._current_catching_strategy_index = 0
        self._number_of_balls_strategy = get_safari_balls_left()
        self._is_current_catching_strategy_baited == True

        action, rocked = self._execute_strategy_action()

        if rocked:
            self._has_been_rocked = rocked
        self._current_catching_strategy_index += 1

        return action, None

    def _continue_current_strategy(self) -> tuple["SafariTurnAction", any]:
        """
        Continues the current catching strategy and returns the next action.
        """
        action, rocked = self._execute_strategy_action()

        if rocked:
            self._has_been_rocked = rocked
        self._current_catching_strategy_index += 1

        return action, None

    def _execute_strategy_action(self) -> tuple["SafariTurnAction", bool]:
        """
        Executes the catch strategy action based on the current strategy index, baited state, and remaining balls.
        """
        action, rocked = get_safari_strategy_action(
            get_opponent(),
            self._number_of_balls_strategy,
            self._current_catching_strategy_index,
            self._is_current_catching_strategy_baited,
        )
        return action, rocked

    def _get_best_status_changing_move(self, battle_state: BattleState) -> int | None:
        # A Pokémon under the influence of Taunt can only use damaging moves.
        if battle_state.own_side.active_battler.taunt_turns_remaining > 0:
            return None

        status_move_index: int | None = None
        status_move_value: float = 0

        for index in range(len(battle_state.own_side.active_battler.moves)):
            learned_move = battle_state.own_side.active_battler.moves[index]
            if learned_move is None or learned_move.pp == 0:
                continue

            opponent_ability = battle_state.opponent.active_battler.ability.name

            value = 0
            if learned_move.move.effect == "SLEEP" and opponent_ability not in (
                "Insomnia",
                "Vital Spirit",
            ):
                value = 2 * learned_move.move.accuracy
            if learned_move.move.effect == "PARALYZE" and opponent_ability != "Limber":
                value = 1.5 * learned_move.move.accuracy
            if status_move_value < value:
                status_move_index = index
                status_move_value = value

        return status_move_index

    def _get_best_poke_ball(self, battle_state: BattleState) -> Item | None:
        best_poke_ball: Item | None = None
        best_catch_rate_multiplier: float = 0
        for ball in get_item_bag().poke_balls:
            if ball.quantity <= 0:
                continue
            catch_rate_multiplier = self._get_poke_ball_catch_rate_multiplier(battle_state, ball.item)

            if best_catch_rate_multiplier < catch_rate_multiplier:
                best_poke_ball = ball.item
                best_catch_rate_multiplier = catch_rate_multiplier

        return best_poke_ball

    def _get_poke_ball_catch_rate_multiplier(self, battle_state: BattleState, ball: Item) -> float:
        opponent = battle_state.opponent.active_battler
        catch_rate_multiplier = 1
        match ball.index:
            # Master Ball -- we never choose to throw this one, should be the player's choice
            case 1:
                catch_rate_multiplier = -1

            # Ultra Ball
            case 2:
                catch_rate_multiplier = 2

            # Great Ball, Safari Ball:
            case 3 | 5:
                catch_rate_multiplier = 1.5

            # Net Ball
            case 6:
                water = get_type_by_name("Water")
                bug = get_type_by_name("Bug")
                if opponent.species.has_type(water) or opponent.species.has_type(bug):
                    catch_rate_multiplier = 3

            # Dive Ball
            case 7:
                if get_map_data_for_current_position().map_type == "Underwater":
                    catch_rate_multiplier = 3.5

            # Nest Ball
            case 8:
                if opponent.level < 40:
                    catch_rate_multiplier = max(1.0, (40 - opponent.level) / 10)

            # Repeat Ball
            case 9:
                if opponent.species in get_pokedex().owned_species:
                    catch_rate_multiplier = 3

            # Timer Ball
            case 10:
                catch_rate_multiplier = min(4.0, (10 + battle_state.current_turn) / 10)

        return catch_rate_multiplier
