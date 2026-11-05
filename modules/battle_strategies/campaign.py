"""Battle strategy owned by Campaign Progression.

Campaign battles have a different ownership contract from the older battle
modes.  In particular, a campaign turn must come from the expectiminimax
planner, even when the result is an item, switch, capture, or escape action.
Keeping this class separate from ``DefaultBattleStrategy`` makes that
contract visible in the type hierarchy and prevents a legacy selector from
being reached through ``super()``.
"""

from __future__ import annotations

from typing import Optional

from modules.battle_observation import default_battle_recorder, observe_current_battle
from modules.battle_planner import (
    PlannerAction,
    PlannerConfidence,
    PlannerDecision,
    PlannerDecisionClass,
    PlannerSafety,
    plan_battle_state,
)
from modules.battle_readiness import assess_battle_readiness, established_search_probability
from modules.battle_state import BattlePokemon, BattleState, get_battle_state
from modules.battle_strategies._interface import BattleStrategy, SafariTurnAction, TurnAction
from modules.battle_strategies._util import BattleStrategyUtil
from modules.battle_move_policy import choose_move_replacement
from modules.console import diagnostic_print
from modules.context import context
from modules.pokemon import LearnedMove, Move, Pokemon
from modules.pokemon_party import get_party
from modules.state_cache import state_cache
from modules.trainer_checkpoints import trainer_checkpoint_for_battle


_CAMPAIGN_PLANNER_RETRY_LIMIT = 60


class CampaignExpectiminimaxStrategy(BattleStrategy):
    """Strict campaign strategy whose single-battle root is the planner.

    The support methods are intentionally implemented here instead of being
    inherited from a legacy strategy.  They provide the lifecycle hooks used
    by the campaign Nuzlocke policies while leaving action ownership with
    ``plan_battle_state``.
    """

    def __init__(self):
        try:
            first_non_fainted = get_party().first_non_fainted
            self._first_non_fainted_party_index_before_battle = first_non_fainted.index
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            # A restored battle can be handed to the listener before the party
            # cache is readable. The active battle planner will re-read the
            # party at its own boundary; keep lifecycle construction alive.
            self._first_non_fainted_party_index_before_battle = 0
        self._last_planner_decision = None

    def allows_legacy_fallback(self) -> bool:
        return False

    def requires_planner_owned_actions(self) -> bool:
        """Require the executor to preserve the planner's selected root.

        Campaign policy and expectiminimax own the tactical choice.  The
        executor may still reject a mechanically impossible stale target, but
        it must replan at the same battle boundary instead of substituting a
        move, item, or escape action from a legacy strategy.
        """

        return True

    def party_can_battle(self) -> bool:
        return any(self.pokemon_can_battle(pokemon) for pokemon in get_party())

    def is_switch_target_valid(self, battle_state: BattleState, party_index: int) -> bool:
        return party_index in self.get_valid_switch_targets(battle_state)

    def switch_target_validation_reason(self, battle_state: BattleState, party_index: int) -> str:
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
            if party_index not in self.get_valid_switch_targets(battle_state):
                return "target has no validated matchup against the current opponent"
            return "validated"
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError):
            return "switch validation could not read the target or matchup"

    def get_valid_switch_targets(self, battle_state: BattleState) -> tuple[int, ...]:
        try:
            return tuple(BattleStrategyUtil(battle_state).get_potential_rotation_targets(battle_state))
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
            return ()

    def select_valid_switch_target(self, battle_state: BattleState) -> int | None:
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
                context.message = (
                    "Active Pokémon fainted. `faint_action` is set to `flee`, "
                    "but this is a trainer battle. Switching to manual mode."
                )
                context.set_manual_mode()
            return True
        return not self.party_can_battle()

    def choose_new_lead_after_faint(self, battle_state: BattleState) -> int:
        new_lead: int | None = None
        new_lead_current_hp = 0
        active_indices = {
            battler.party_index
            for battler in getattr(battle_state.own_side, "active_battlers", ())
            if battler is not None
        }
        for index, pokemon in enumerate(get_party()):
            if index in active_indices:
                continue
            if not pokemon.is_egg and pokemon.current_hp > 0 and pokemon.current_hp > new_lead_current_hp:
                new_lead = index
                new_lead_current_hp = pokemon.current_hp
        if new_lead is None:
            raise RuntimeError("no living campaign replacement is available")
        return new_lead

    def choose_new_lead_after_battle(self) -> int | None:
        party = get_party()
        if not self.pokemon_can_battle(party[self._first_non_fainted_party_index_before_battle]):
            return BattleStrategyUtil(get_battle_state()).select_rotation_target()
        return None

    def decide_turn(self, battle_state: BattleState) -> tuple[TurnAction, Optional[any]]:
        """Select the only autonomous campaign action at this boundary."""

        if getattr(context, "battle_switch_replan_required", False):
            # A switch mismatch means the planner's previous active-slot
            # projection is stale. Take one fresh observation/replan turn
            # before the normal safety boundary can request manual input.
            context.battle_switch_replan_required = False
            context.battle_switch_reconciliation_status = "replanning"
            diagnostic_print(
                lambda: (
                    "BATTLE_SWITCH_RECONCILIATION: "
                    "status='fresh_observation' action='replan' "
                    f"pending={tuple(getattr(context, 'battle_pending_switches', ()))!r}"
                ),
                trace=False,
                prefix="BATTLE_SWITCH_RECONCILIATION",
            )
            try:
                default_battle_recorder.record(observe_current_battle())
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError) as error:
                diagnostic_print(
                    lambda: (
                        "BATTLE_SWITCH_RECONCILIATION: "
                        f"status='fresh_observation_failed' exception_type={type(error).__name__!r}"
                    ),
                    trace=False,
                    prefix="BATTLE_SWITCH_RECONCILIATION",
                )
        return self._decide_turn_after_reconciliation(battle_state)

    def _planner_boundary_diagnostic(
        self,
        battle_state: BattleState,
        decision: PlannerDecision | None,
        *,
        branch: str,
        error: Exception | None = None,
    ) -> None:
        try:
            active = battle_state.own_side.active_battler
            opponent = getattr(getattr(battle_state, "opponent", None), "active_battler", None)
            active_identity = (
                getattr(getattr(active, "species", None), "name", None),
                getattr(active, "party_index", None),
            )
            roots = tuple(
                f"{getattr(root, 'action_kind', None)}:{getattr(root, 'action_index', None)}:{getattr(root, 'action_name', None)!r}"
                for root in getattr(getattr(decision, "stochastic_result", None), "root_evaluations", ())
            )
            if not roots:
                roots = tuple(
                    f"{getattr(candidate, 'label', None)!r}:{getattr(candidate, 'target', None)!r}"
                    for candidate in getattr(decision, "candidate_evaluations", ())
                )
            payload = (
                f"branch={branch!r} active_slot={getattr(active, 'party_index', None)!r} "
                f"active_identity={active_identity!r} legal_roots={roots!r} "
                f"active_hp={getattr(active, 'current_hp', None)!r}/{getattr(active, 'total_hp', None)!r} "
                f"opponent_hp={getattr(opponent, 'current_hp', None)!r}/{getattr(opponent, 'total_hp', None)!r} "
                f"returned_action={getattr(getattr(decision, 'action', None), 'value', None)!r} "
                f"returned_target={getattr(decision, 'target', None)!r} "
                f"classification={getattr(getattr(decision, 'classification', None), 'value', None)!r} "
                f"category={getattr(getattr(decision, 'stochastic_result', None), 'category', None)!r} "
                f"loss_probability={getattr(getattr(decision, 'stochastic_result', None), 'loss_probability', None)!r} "
                f"party_wipe_probability={getattr(getattr(decision, 'stochastic_result', None), 'party_wipe_probability', None)!r} "
                f"run_loss_probability={getattr(getattr(decision, 'stochastic_result', None), 'run_loss_probability', None)!r} "
                f"objective_failed={getattr(getattr(decision, 'stochastic_result', None), 'objective_failed', None)!r} "
                f"pending_switches={tuple(getattr(context, 'battle_pending_switches', ()))!r} "
                f"reconciliation={getattr(context, 'battle_switch_reconciliation_status', None)!r} "
                f"exception_type={type(error).__name__!r} exception={error!r}"
                if error is not None
                else (
                    f"branch={branch!r} active_slot={getattr(active, 'party_index', None)!r} "
                    f"active_identity={active_identity!r} legal_roots={roots!r} "
                    f"active_hp={getattr(active, 'current_hp', None)!r}/{getattr(active, 'total_hp', None)!r} "
                    f"opponent_hp={getattr(opponent, 'current_hp', None)!r}/{getattr(opponent, 'total_hp', None)!r} "
                    f"returned_action={getattr(getattr(decision, 'action', None), 'value', None)!r} "
                    f"returned_target={getattr(decision, 'target', None)!r} "
                    f"classification={getattr(getattr(decision, 'classification', None), 'value', None)!r} "
                    f"category={getattr(getattr(decision, 'stochastic_result', None), 'category', None)!r} "
                    f"loss_probability={getattr(getattr(decision, 'stochastic_result', None), 'loss_probability', None)!r} "
                    f"party_wipe_probability={getattr(getattr(decision, 'stochastic_result', None), 'party_wipe_probability', None)!r} "
                    f"run_loss_probability={getattr(getattr(decision, 'stochastic_result', None), 'run_loss_probability', None)!r} "
                    f"objective_failed={getattr(getattr(decision, 'stochastic_result', None), 'objective_failed', None)!r} "
                    f"pending_switches={tuple(getattr(context, 'battle_pending_switches', ()))!r} "
                    f"reconciliation={getattr(context, 'battle_switch_reconciliation_status', None)!r}"
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError) as diagnostic_error:
            payload = f"branch={branch!r} diagnostic_exception={type(diagnostic_error).__name__!r}"
        diagnostic_print(
            lambda: f"CAMPAIGN_PLANNER_ABORT: {payload}",
            trace=False,
            prefix="CAMPAIGN_PLANNER_ABORT",
        )

    @staticmethod
    def _planner_probability_text(result, attribute: str) -> str:
        value = established_search_probability(result, attribute)
        return f"{value:.1%}" if isinstance(value, (int, float)) else "unknown"

    @staticmethod
    def _planner_action_label(decision: PlannerDecision | None) -> str:
        if decision is None:
            return "none"
        action = getattr(getattr(decision, "action", None), "value", None)
        target = getattr(decision, "target", None)
        return f"{action or 'unknown'}({target!r})"

    @staticmethod
    def _is_empty_normalized_root(decision: PlannerDecision | None) -> bool:
        """Return whether the planner could not translate any legal root."""

        return (
            decision is not None
            and getattr(decision, "stochastic_result", None) is None
            and CampaignExpectiminimaxStrategy._turn_action_from_planner_decision(decision) is None
        )

    def _refresh_empty_root_boundary(self, battle_state: BattleState) -> BattleState | None:
        """Refresh ROM-backed battle facts once before stopping on an empty root.

        The battle handler normally observes immediately before calling the
        strategy. At battle start, however, the action callback can become
        ready in the same frame that the battler buffers and inventory cache
        finish settling. A campaign-owned refresh preserves planner ownership
        while allowing that transient boundary to become coherent.
        """

        if getattr(context, "battle_empty_root_replan_used", False):
            return None
        context.battle_empty_root_replan_used = True
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PLANNER_REFRESH: "
                "status='empty_root' action='fresh_observation_replan' attempt=1"
            ),
            trace=False,
            prefix="CAMPAIGN_PLANNER_REFRESH",
        )
        try:
            # The outer handler's observation may have populated a same-frame
            # cache before the ROM finished writing the battle structures.
            # Invalidate the live adapters before reading the retry snapshot.
            state_cache.invalidate_runtime_observations()
            observe_current_battle()
            refreshed_state = get_battle_state()
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PLANNER_REFRESH: "
                    "status='fresh_observation_complete' "
                    f"active_slot={getattr(getattr(refreshed_state.own_side, 'active_battler', None), 'party_index', None)!r} "
                    f"opponent_hp={getattr(getattr(refreshed_state.opponent, 'active_battler', None), 'current_hp', None)!r}"
                ),
                trace=False,
                prefix="CAMPAIGN_PLANNER_REFRESH",
            )
            return refreshed_state
        except (AttributeError, RuntimeError, TypeError, ValueError, IndexError, KeyError) as error:
            diagnostic_print(
                lambda: (
                    "CAMPAIGN_PLANNER_REFRESH: "
                    f"status='fresh_observation_failed' exception_type={type(error).__name__!r}"
                ),
                trace=False,
                prefix="CAMPAIGN_PLANNER_REFRESH",
            )
            return None

    @staticmethod
    def _request_planner_retry(reason: str) -> bool:
        """Keep the battle callback alive while normalized facts settle."""

        retry_count = int(getattr(context, "battle_planner_retry_count", 0) or 0)
        if retry_count >= _CAMPAIGN_PLANNER_RETRY_LIMIT:
            return False
        retry_count += 1
        context.battle_planner_retry_count = retry_count
        context.battle_planner_retry_requested = True
        context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX RETRY"
        context.battle_decision_detail = (
            f"planner boundary unavailable ({reason}); "
            f"waiting for a fresh battle snapshot ({retry_count}/{_CAMPAIGN_PLANNER_RETRY_LIMIT})"
        )
        context.message = f"CAMPAIGN EXPECTIMINIMAX RETRY: {context.battle_decision_detail}"
        diagnostic_print(
            lambda: (
                "CAMPAIGN_PLANNER_RETRY: "
                f"reason={reason!r} retry={retry_count!r} limit={_CAMPAIGN_PLANNER_RETRY_LIMIT!r}"
            ),
            trace=True,
            prefix="CAMPAIGN_PLANNER_RETRY:",
        )
        return True

    def _decide_turn_after_reconciliation(self, battle_state: BattleState):

        planner_error = None
        try:
            planner_decision = plan_battle_state(battle_state, default_battle_recorder.knowledge)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, IndexError) as error:
            planner_error = error
            self._planner_boundary_diagnostic(
                battle_state,
                None,
                branch="planner_exception",
                error=error,
            )
            planner_decision = PlannerDecision(
                PlannerAction.AbortForSafety,
                confidence=PlannerConfidence.INSUFFICIENT,
                rationale=(
                    "Expectiminimax could not normalize the current campaign battle boundary: "
                    f"{type(error).__name__}: {error}"
                ),
                safety=PlannerSafety.UNSAFE_OR_UNCERTAIN,
                evaluation_method="expectiminimax",
            )

        if (planner_error is not None or self._is_empty_normalized_root(planner_decision)) and not getattr(
            context, "battle_planner_retry_requested", False
        ):
            refreshed_state = self._refresh_empty_root_boundary(battle_state)
            if refreshed_state is not None:
                return self._decide_turn_after_reconciliation(refreshed_state)

        self._last_planner_decision = planner_decision
        context.battle_evaluation_method = getattr(planner_decision, "evaluation_method", "expectiminimax")
        stochastic_result = getattr(planner_decision, "stochastic_result", None)
        if stochastic_result is not None and getattr(battle_state, "is_trainer_battle", False):
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
                    f"win_probability={self._planner_probability_text(stochastic_result, 'win_probability')!r} "
                    f"catastrophic_probability={self._planner_probability_text(stochastic_result, 'catastrophic_probability')!r} "
                    f"category={getattr(getattr(stochastic_result, 'category', None), 'value', None)!r}"
                ),
                trace=True,
            )

        classification = getattr(planner_decision, "classification", None)
        safe = bool(getattr(planner_decision, "is_safe_to_execute", False))
        if classification is None and safe:
            classification = PlannerDecisionClass.SAFE
        planner_action = self._turn_action_from_planner_decision(planner_decision)
        confidence = getattr(planner_decision, "confidence", PlannerConfidence.INSUFFICIENT)
        stochastic_category = getattr(getattr(stochastic_result, "category", None), "value", None)
        # ``is_autonomous_root`` is the search result's explicit contract for
        # a concrete legal root that may own the next live action.  Do not add
        # the planner confidence threshold here: defeat/training results are
        # intentionally executable even when their only line ends in an
        # active faint or party wipe, so Emerald can resolve the loss and the
        # campaign recovery path can restart the run.  Confidence remains
        # relevant to classification and diagnostics, while missing roots,
        # untranslatable roots, and known objective failure still stop below.
        autonomous_root_allowed = bool(
            stochastic_result is not None
            and getattr(stochastic_result, "is_autonomous_root", False)
        )
        accepted = (safe or autonomous_root_allowed) and planner_action is not None
        diagnostic_print(
            lambda: (
                "CAMPAIGN_BATTLE_PLANNER: "
                f"classification={getattr(classification, 'name', classification)} "
                f"action={planner_decision.action.value} target={planner_decision.target!r} "
                f"confidence={getattr(confidence, 'value', confidence)} "
                f"projection_safe={getattr(stochastic_result, 'projection_safe', None)!r} "
                f"autonomous_root={autonomous_root_allowed!r} "
                f"accepted={accepted} "
                f"rationale={getattr(planner_decision, 'rationale', None)!r}"
            ),
            trace=True,
        )
        if accepted:
            context.battle_decision_source = (
                "CAMPAIGN EXPECTIMINIMAX"
                if safe
                else "CAMPAIGN EXPECTIMINIMAX BEST_AVAILABLE"
            )
            rationale = getattr(planner_decision, "rationale", "planner-selected action")
            context.battle_decision_detail = (
                f"classification={classification.name}; "
                f"category={stochastic_category!r}; "
                f"projection_safe={getattr(stochastic_result, 'projection_safe', None)!r}; {rationale}"
            )
            context.message = f"CAMPAIGN EXPECTIMINIMAX: {self._planner_action_name(planner_decision)} — {rationale}"
            return planner_action

        if planner_error is not None:
            abort_branch = "planner_exception"
        elif getattr(planner_decision, "stochastic_result", None) is None:
            abort_branch = "no_legal_normalized_root"
        elif planner_action is None:
            abort_branch = "returned_root_not_executable"
        else:
            abort_branch = "safety_rejection"
        if abort_branch in {"planner_exception", "no_legal_normalized_root"}:
            retry_reason = "planner exception" if abort_branch == "planner_exception" else "no normalized legal root"
            if self._request_planner_retry(retry_reason):
                return TurnAction.switch_to_manual()
        self._planner_boundary_diagnostic(
            battle_state,
            planner_decision,
            branch=abort_branch,
        )
        context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX STOP"
        action_label = self._planner_action_label(planner_decision)
        if abort_branch == "planner_exception":
            detail = (
                f"planner exception {type(planner_error).__name__}: {planner_error}; "
                "no normalized legal planner root was returned"
            )
        elif abort_branch == "no_legal_normalized_root":
            facts = tuple(
                f"{candidate.label}={candidate.available!r}"
                for candidate in getattr(planner_decision, "candidate_evaluations", ())
            )
            detail = (
                "no normalized legal planner root was returned; "
                f"planner_rationale={getattr(planner_decision, 'rationale', '')!r}; "
                f"candidate_inventory={facts!r}"
            )
        elif abort_branch == "returned_root_not_executable":
            detail = f"planner returned {action_label}, but it could not be translated into a battle action"
        else:
            detail = (
                f"safety rejection of legal root {action_label}; "
                f"classification={getattr(classification, 'value', classification)!r}; "
                f"category={stochastic_category!r}; "
                f"projection_safe={getattr(stochastic_result, 'projection_safe', None)!r}; "
                f"loss_probability={getattr(stochastic_result, 'loss_probability', None)!r}; "
                f"party_wipe_probability={getattr(stochastic_result, 'party_wipe_probability', None)!r}; "
                f"objective_failed={getattr(stochastic_result, 'objective_failed', None)!r}"
            )
        context.battle_decision_detail = detail
        context.message = f"CAMPAIGN EXPECTIMINIMAX STOP: {detail}"
        return TurnAction.switch_to_manual()

    @staticmethod
    def _planner_action_name(planner_decision) -> str:
        if planner_decision.action is PlannerAction.UseMove:
            return getattr(planner_decision, "rationale", "").split(":", 1)[0] or str(planner_decision.target)
        return planner_decision.action.value

    @staticmethod
    def _turn_action_from_planner_decision(planner_decision):
        match planner_decision.action:
            case PlannerAction.UseMove:
                return TurnAction.use_move(planner_decision.target)
            case PlannerAction.SwitchPokemon:
                return TurnAction.rotate_lead(planner_decision.target)
            case PlannerAction.UseItem:
                target = getattr(planner_decision, "target", None)
                if isinstance(target, tuple):
                    item, party_index = target
                    if (
                        getattr(getattr(item, "battle_use", None), "value", None) == "catch"
                        or "ball" in str(getattr(item, "name", "")).casefold()
                    ):
                        return TurnAction.use_item(item)
                    return TurnAction.use_item_on(item, party_index)
                return TurnAction.use_item(target)
            case PlannerAction.RunAway:
                return TurnAction.run_away()
        return None

    def decide_turn_in_double_battle(self, battle_state: BattleState, battler_index: int):
        context.battle_decision_source = "CAMPAIGN EXPECTIMINIMAX STOP"
        context.battle_decision_detail = "double-battle root ownership is not represented by the single-battle planner"
        context.message = "CAMPAIGN EXPECTIMINIMAX STOP: no campaign double-battle root action was established"
        return TurnAction.switch_to_manual()

    def decide_turn_in_safari_zone(self, battle_state: BattleState):
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
