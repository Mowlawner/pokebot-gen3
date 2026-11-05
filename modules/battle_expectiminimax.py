"""Small, deterministic expectiminimax search for single battles.

The live battle controller remains responsible for reading the ROM and sending
inputs.  This module only evaluates normalized facts.  It deliberately models
random outcomes as distributions without attempting to predict the emulator's
next random value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from time import monotonic
from typing import Callable


class OpponentPolicy(Enum):
    """How the search chooses among known legal opponent actions."""

    CONSERVATIVE = "conservative"
    RULE_BASED = "rule_based"


class SearchResultCategory(Enum):
    """How much of the selected line the search actually established."""

    TERMINAL_COMPLETE = "terminal_complete"
    HORIZON_PROJECTION = "horizon_projection"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class SearchImmediateOutcome:
    """One first-turn stochastic branch for a selected battle action."""

    phase: str
    action_name: str
    probability: float
    result: str
    damage: int = 0
    target: str | None = None
    critical: bool = False
    secondary_effect: str | None = None


@dataclass(frozen=True, slots=True)
class SearchImmediateProjection:
    """A shallow projection kept valid when the deep search is cut off."""

    action_kind: str
    action_index: int | None
    action_name: str | None
    player_outcomes: tuple[SearchImmediateOutcome, ...] = ()
    opponent_responses: tuple[SearchImmediateOutcome, ...] = ()
    turn_order: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchMove:
    """Planner-facing move facts used by the stochastic search."""

    index: int
    name: str
    damage_min: int = 0
    damage_max: int = 0
    accuracy: float = 1.0
    priority: int = 0
    critical_damage_max: int | None = None
    critical_hit_chance: float = 1 / 16
    # Secondary effects are reported as uncertainty.  The search does not
    # rely on a favorable status activation to prove a safe line.
    secondary_chance: float = 0.0
    secondary_effect: str | None = None
    # Stat effects are kept after the original fields for positional
    # compatibility with callers that construct opponent moves directly.
    stat_target: str | None = None
    stat_delta: int = 0
    damage_stat: str | None = None
    # A small set of moves, such as wild Abra's Teleport, ends the encounter
    # instead of creating a normal damage exchange.  This is deliberately an
    # explicit fact: the evaluator must not invent an opponent ``run`` node
    # for ordinary wild battles.
    causes_opponent_escape: bool = False
    # ``None`` preserves the legacy meaning of an unbounded synthetic move.
    # Live player moves carry their remaining PP so the search cannot plan an
    # unlimited sequence of setup turns.
    pp: int | None = None


@dataclass(frozen=True, slots=True)
class SearchSwitch:
    """Facts for a party member that can replace the active battler."""

    party_index: int
    name: str
    hp: int
    max_hp: int
    moves: tuple[SearchMove, ...]
    speed: int | None = None
    status: str | None = None
    # Optional objective value for assigning the next KO's EXP to this slot.
    # Ordinary battle callers leave this at zero.
    experience_value: float = 0.0
    # Escape facts for this replacement.  ``None`` preserves compatibility
    # with synthetic callers that only provide a scalar root probability;
    # live campaign callers provide speed and the evaluator derives the
    # action-specific Gen III flee chance after the switch.
    escape_probability: float | None = None
    escape_guaranteed: bool = False


@dataclass(frozen=True, slots=True)
class SearchItem:
    """A battle item usable by the active Pokémon in the search model."""

    item_index: int
    name: str
    quantity: int = 1
    heal_amount: int = 0
    full_heal: bool = False
    cures_status: bool = False
    is_capture_ball: bool = False
    # This is the current normalized probability for a successful throw.  A
    # live adapter may refresh it for each observed HP/status state.
    capture_probability: float = 0.0
    # Keep targeted status coverage when the live adapter has it.  An empty
    # set retains the legacy meaning of ``cures_status=True``: the item is a
    # generic status cure in a synthetic caller.
    status_healed: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class SearchState:
    """Immutable single-battle state for one search node."""

    player_hp: int
    player_max_hp: int
    opponent_hp: int
    opponent_max_hp: int
    player_moves: tuple[SearchMove, ...]
    opponent_moves: tuple[SearchMove, ...]
    player_speed: int | None = None
    opponent_speed: int | None = None
    player_status: str | None = None
    opponent_status: str | None = None
    player_party_index: int | None = None
    player_switches: tuple[SearchSwitch, ...] = ()
    player_items: tuple[SearchItem, ...] = ()
    # Objective facts are deliberately optional so the core search remains
    # usable for ordinary battles.  Campaign adapters can opt into capture
    # and EXP-aware utility without coupling this module to Nuzlocke state.
    battle_objective: str = "defeat"
    capture_target: bool = False
    experience_reward: int = 0
    experience_values: tuple[tuple[int, float], ...] = ()
    experience_participants: frozenset[int] = frozenset()
    player_stat_stages: tuple[tuple[str, int], ...] = ()
    opponent_stat_stages: tuple[tuple[str, int], ...] = ()
    # Wild-battle escape is a legal search action.  Keep the probability on
    # the normalized state so the evaluator can compare a failed escape
    # attempt with continuing the battle.
    can_run: bool = False
    escape_probability: float = 0.0
    # The capture objective may transition to ordinary defeat after the final
    # ball fails.  Preserve that the original capture attempt has already
    # failed so a later KO can earn EXP without being mistaken for a capture
    # success in the parent branch.
    capture_failed: bool = False
    capture_attempted: bool = False
    # A projected response set can still rank legal roots, but cannot prove
    # that a line is safe against undiscovered opponent actions.
    response_facts_complete: bool = True
    # Recent active-slot transitions are part of the search state.  They keep
    # a capture line from spending successive turns reversing the same switch
    # without changing HP, status, or capture progress.
    switch_history: tuple[int, ...] = ()
    # Campaign Progression explicitly opts into the stricter action-shaping
    # rules below.  Direct/offline and legacy battle callers retain their
    # previous search semantics unless they identify themselves as campaign
    # planning.
    campaign_owned: bool = False
    # Trainer replacement prompts are zero-cost boundaries. They may consider
    # every living reserve, including the member just switched out.
    free_switch: bool = False
    # Consecutive switches without a player move are legal, but they are a
    # progressively worse way to pursue a defeat/training objective.  Keeping
    # this in the normalized state lets utility rank a real lead rotation
    # against a three-member stall without removing any legal root action.
    switch_streak: int = 0
    # The opponent HP at the root of the current planner call.  A bounded
    # switch projection must make progress from this value; damage that was
    # already dealt before the root does not make a fresh switch safe.
    root_opponent_hp: int | None = None
    # Known opponent mechanics that the bounded evaluator does not model are
    # retained as a conservative response ceiling instead of being discarded.
    unknown_response_ceiling: int = 0
    # Repeating a previously visited slot is a stronger signal of a rotation
    # than a single consecutive switch.  This pressure persists at the live
    # planner boundary and remains a utility cost rather than a legality gate.
    switch_cycle_pressure: int = 0
    # Gen III Bide state.  These fields are appended so existing positional
    # SearchState constructors keep their meaning.  A Bide user is locked to
    # Bide for two subsequent turns; damage is accumulated only when it loses
    # actual HP and is released doubled on the final turn.
    player_bide_turns: int = 0
    player_bide_damage: int = 0
    player_bide_target: str | None = None
    opponent_bide_turns: int = 0
    opponent_bide_damage: int = 0
    opponent_bide_target: str | None = None
    # ``None`` retains the explicit scalar probability used by legacy and
    # fixture callers.  Live battle planning supplies the ROM flee-attempt
    # counter so switching and failed attempts can change the next chance.
    escape_attempts: int | None = None
    escape_guaranteed: bool = False
    # Prevent consecutive healing actions from becoming a fixed point in an
    # avoid encounter.  A move, switch, or escape attempt breaks the streak.
    healing_streak: int = 0


def _normalized_status(status: str | None) -> str:
    """Normalize battle status names used by live and synthetic callers."""

    value = getattr(status, "value", status)
    normalized = str(value or "").strip().casefold().replace("_", " ")
    return {
        "healthy": "none",
        "normal": "none",
        "poisoned": "poison",
        "paralyzed": "paralysed",
    }.get(normalized, normalized)


def _status_is_healthy(status: str | None) -> bool:
    return _normalized_status(status) in {"", "none"}


def _item_cures_status(item: SearchItem, status: str | None) -> bool:
    """Return whether an item has an effect on the supplied condition."""

    if not item.cures_status or _status_is_healthy(status):
        return False
    coverage = {_normalized_status(value) for value in item.status_healed}
    return not coverage or "all" in coverage or _normalized_status(status) in coverage


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Bound the tree and define how conservative the result must be."""

    max_turns: int = 8
    minimum_safe_win_probability: float = 0.90
    maximum_safe_loss_probability: float = 0.05
    maximum_safe_catastrophic_probability: float = 0.05
    opponent_policy: OpponentPolicy = OpponentPolicy.CONSERVATIVE
    # Depth alone does not bound a branching stochastic tree. Keep live
    # battle planning responsive even when a ROM move envelope contains
    # several damage/critical/status branches.
    max_nodes: int = 25_000
    # A node cap alone can still produce long pauses when individual nodes
    # have wide stochastic branching. The live planner uses this cooperative
    # deadline to return the best line established so far without blocking
    # the emulator frame loop indefinitely. ``None`` is useful to callers
    # that explicitly want an unbounded offline evaluation.
    max_duration_seconds: float | None = 0.75
    # Useful EXP is a secondary objective among otherwise comparable safe
    # lines.  Safety and objective success retain dominant terminal values.
    experience_weight: float = 0.0
    # Ordinary switching consumes the player's turn and exposes the incoming
    # Pokémon to the opponent. The response tree models the latter; this
    # utility cost models the lost opportunity to make battle progress.
    # Trainer replacement prompts set ``SearchState.free_switch`` and do not
    # pay this cost.
    ordinary_switch_penalty: float = 0.0
    # A switch that follows another switch is increasingly likely to be a
    # no-progress rotation.  This is a utility term, not a legality gate.
    switch_stall_penalty: float = 150.0
    # Re-entering a slot already present in the observed no-progress sequence
    # is especially likely to be a three-member cycle. Keep it legal while
    # making an attacking continuation win whenever it is materially better.
    switch_cycle_penalty: float = 250.0
    # A bounded switch projection must establish a useful continuation.  If
    # it only changes the active slot and leaves the opponent untouched, make
    # that root materially less attractive while retaining it for reporting.
    switch_no_progress_penalty: float = 175.0
    maximum_safe_encounter_loss_probability: float = 0.0
    # Items are finite battle resources.  A successful item action therefore
    # carries a small utility cost, while an item that heals beyond the
    # missing HP carries an additional waste cost.  These costs are kept far
    # below the terminal win/faint values so preserving a living Pokémon
    # always dominates conserving a resource.
    item_use_penalty: float = 25.0
    item_scarcity_penalty: float = 50.0
    item_waste_penalty: float = 5.0
    # Optional ROM-specific catch evaluator. It receives the post-action
    # normalized state and ball, so weakening and status branches can improve
    # capture odds during the same search.
    capture_probability: Callable | None = None
    # Capture is an encounter objective, so a failed encounter must carry a
    # real negative utility through chance nodes.  These weights are kept
    # objective-specific: ordinary defeat and avoidance retain their existing
    # utility scale, while capture roots compare success, encounter loss, and
    # party risk in one expected-utility calculation.
    capture_success_reward: float = 1_000_000.0
    capture_encounter_loss_penalty: float = 1_000_000.0
    capture_run_away_penalty: float = 1_500_000.0
    capture_active_faint_penalty: float = 2_000_000.0
    capture_party_wipe_penalty: float = 3_000_000.0
    capture_run_loss_penalty: float = 3_000_000.0
    # Every capture setup move consumes a turn in which the target can attack
    # and the ball is not thrown. Keep this finite and small relative to the
    # encounter outcome values so meaningful weakening or status can still
    # win when it materially improves capture odds or survival.
    capture_turn_penalty: float = 25.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Comparison of the best first actions from one search root."""

    action_index: int | None
    action_name: str | None
    expected_score: float
    win_probability: float
    loss_probability: float
    catastrophic_probability: float
    principal_variation: tuple[str, ...] = ()
    expanded_nodes: int = 0
    complete: bool = True
    reason: str = ""
    # ``time`` and ``nodes`` distinguish an autonomous projection caused by
    # the frame deadline from one caused by the structural search cap.
    budget_exhaustion: str | None = None
    minimum_safe_win_probability: float = 0.90
    maximum_safe_loss_probability: float = 0.05
    maximum_safe_catastrophic_probability: float = 0.05
    action_kind: str = "move"
    capture_probability: float = 0.0
    encounter_loss_probability: float = 0.0
    expected_experience: float = 0.0
    experience_utility: float = 0.0
    objective: str = "defeat"
    maximum_safe_encounter_loss_probability: float = 0.0
    category: SearchResultCategory = SearchResultCategory.TERMINAL_COMPLETE
    root_evaluations: tuple["SearchRootEvaluation", ...] = ()
    # Risk dimensions are kept separate so campaign policy can distinguish an
    # ordinary active-Pokémon faint from a party wipe or run-ending outcome.
    active_faint_probability: float = 0.0
    party_wipe_probability: float = 0.0
    run_loss_probability: float = 0.0
    # A bounded nonterminal line may be executable only when the projected
    # state makes progress and survives the worst known immediate response.
    # This is separate from ``complete`` so a budget cutoff cannot acquire
    # permission to execute.
    projection_safe: bool = False
    immediate_projection: SearchImmediateProjection | None = None
    # Healing roots may target a living reserve while the active Pokémon
    # absorbs the opponent's response.  Carry that decision through the
    # search result instead of reconstructing it from the active slot.
    target_party_index: int | None = None
    # A terminal capture/encounter objective can be established as failed
    # even when the root action remains mechanically legal.  Preserve that
    # distinction so the autonomous uncertainty policy does not execute a
    # known objective failure.
    objective_failed: bool = False

    @property
    def is_safe(self) -> bool:
        if self.objective_failed:
            return False
        win_or_projection_safe = (
            self.win_probability >= self.minimum_safe_win_probability
            if self.category is SearchResultCategory.TERMINAL_COMPLETE
            else self.projection_safe
        )
        return (
            self.complete
            and win_or_projection_safe
            and self.loss_probability <= self.maximum_safe_loss_probability
            and self.catastrophic_probability <= self.maximum_safe_catastrophic_probability
            and self.encounter_loss_probability <= self.maximum_safe_encounter_loss_probability
            and (
                self.category is SearchResultCategory.TERMINAL_COMPLETE
                or (self.category is SearchResultCategory.HORIZON_PROJECTION and self.projection_safe)
            )
        )

    @property
    def is_executable_projection(self) -> bool:
        """Whether a bounded result has enough first-turn evidence to run.

        ``is_safe`` intentionally remains a proof-level result.  Campaign
        execution may use a bounded line only when the selected root itself
        advances the objective and survives the known immediate response.
        Keeping that permission separate prevents a partial deep search from
        being mistaken for a full safety proof.
        """

        return bool(
            self.category in {
                SearchResultCategory.INCOMPLETE,
                SearchResultCategory.HORIZON_PROJECTION,
            }
            and self.projection_safe
            and self.action_kind in {"move", "switch", "item", "run"}
        )

    @property
    def is_autonomous_projection(self) -> bool:
        """Whether a legal unresolved root may continue autonomously.

        Horizon and budget-limited results are policy uncertainty, rather
        than coverage failures.  They may therefore own the next action when
        the search returned a concrete root.  A proven objective failure or
        a proven guaranteed catastrophe remains outside this contract and is
        handled by the caller's safety boundary.
        """

        if self.category not in {
            SearchResultCategory.INCOMPLETE,
            SearchResultCategory.HORIZON_PROJECTION,
        }:
            return False
        if self.action_kind not in {"move", "switch", "item", "run"} or self.action_name is None:
            return False
        if self.objective_failed:
            return False
        # Defeat and training battles remain executable until the ROM resolves
        # the outcome. An active faint can be replaced by the battle handler,
        # while a party wipe must reach Emerald's native whiteout flow so the
        # run can be restarted autonomously.
        if self.objective in {"defeat", "training"}:
            return True
        return not any(
            isinstance(probability, (int, float)) and probability >= 1.0
            for probability in (
                self.loss_probability,
                self.party_wipe_probability,
                self.run_loss_probability,
                self.encounter_loss_probability,
            )
        )

    @property
    def is_autonomous_root(self) -> bool:
        """Whether the selected legal root may own the next live action.

        A complete search can still return ``BEST_AVAILABLE`` when no root
        meets the proof-level risk thresholds. That result is different from
        a proven objective failure or guaranteed loss: the selected action is
        still a concrete legal root, and campaign policy may continue one
        turn so the live battle can produce the next authoritative state.
        """

        if self.category is SearchResultCategory.TERMINAL_COMPLETE:
            if self.action_kind not in {"move", "switch", "item", "run"} or self.action_name is None:
                return False
            if self.objective_failed:
                return False
            if self.objective in {"defeat", "training"}:
                return True
            return not any(
                isinstance(probability, (int, float)) and probability >= 1.0
                for probability in (
                    self.loss_probability,
                    self.party_wipe_probability,
                    self.run_loss_probability,
                    self.encounter_loss_probability,
                )
            )
        return self.is_autonomous_projection


def _run_action_available(state: SearchState) -> bool:
    """Return whether Run Away is a legal, nonzero-probability action."""

    return bool(state.can_run and _escape_probability(state) > 0.0)


def _escape_probability(state: SearchState) -> float:
    """Return the flee chance for the currently active Pokémon.

    The live adapter supplies the current ROM attempt counter.  Gen III uses
    the active Pokémon's speed and the opponent's speed for each attempt, so
    a switch must recompute this value instead of inheriting the lead's
    probability.  Older callers that do not provide the counter retain their
    explicit normalized probability.
    """

    if not state.can_run:
        return 0.0
    if state.escape_guaranteed:
        return 1.0
    if state.escape_attempts is None:
        return _clamp_probability(state.escape_probability)
    if state.player_speed is not None and state.opponent_speed is not None:
        if state.player_speed > state.opponent_speed:
            return 1.0
        if state.opponent_speed > 0:
            chance = (
                (state.player_speed * 128) // state.opponent_speed
                + max(0, state.escape_attempts) * 30
            ) % 256
            return _clamp_probability(chance / 255.0)
    return _clamp_probability(state.escape_probability)


def _failed_escape_state(state: SearchState) -> SearchState:
    """Advance the flee-attempt boundary after a failed Run Away action."""

    return replace(
        state,
        escape_attempts=(
            None if state.escape_attempts is None else max(0, state.escape_attempts) + 1
        ),
        healing_streak=0,
    )


@dataclass(frozen=True, slots=True)
class SearchRootEvaluation:
    """The normalized result for one legal action at the search root."""

    action_kind: str
    action_index: int | None
    action_name: str
    expected_score: float
    win_probability: float
    loss_probability: float
    catastrophic_probability: float
    category: SearchResultCategory
    principal_variation: tuple[str, ...] = ()
    active_faint_probability: float = 0.0
    party_wipe_probability: float = 0.0
    run_loss_probability: float = 0.0
    # Keep the same shallow first-turn evidence for every legal root action,
    # including switches that were considered but not selected.
    immediate_projection: SearchImmediateProjection | None = None
    target_party_index: int | None = None
    objective_progress: float = 0.0
    projection_safe: bool = False
    budget_exhaustion: str | None = None
    objective_failed: bool = False


@dataclass(frozen=True, slots=True)
class _Value:
    score: float
    win: float
    loss: float
    line: tuple[str, ...]
    capture: float = 0.0
    encounter_loss: float = 0.0
    expected_experience: float = 0.0
    experience_utility: float = 0.0
    category: SearchResultCategory = SearchResultCategory.TERMINAL_COMPLETE
    active_faint: float = 0.0
    party_wipe: float = 0.0
    run_loss: float = 0.0
    projection_safe: bool = False
    target_party_index: int | None = None
    objective_failed: bool = False
    # Expected normalized opponent progress established by the player line.
    # This is kept separate from the terminal score so switch projections can
    # tell a damaging continuation from a pure lead rotation.
    objective_progress: float = 0.0


def _success_state_utility(state: SearchState) -> float:
    """Value the surviving state after the battle objective succeeds."""

    active_ratio = state.player_hp / max(1, state.player_max_hp)
    reserve_ratio = sum(
        switch.hp / max(1, switch.max_hp)
        for switch in state.player_switches
        if switch.hp > 0 and switch.max_hp > 0
    )
    remaining_balls = sum(
        max(0, int(item.quantity))
        for item in state.player_items
        if item.is_capture_ball
    )
    # Keep state quality large enough to distinguish two successful lines,
    # while leaving the objective-success value dominant over ordinary HP and
    # finite-resource tradeoffs.
    return 100.0 * active_ratio + 20.0 * reserve_ratio + float(remaining_balls)


def _value_selection_key(
    value: _Value,
    *,
    category: SearchResultCategory | None = None,
    prefer_unresolved: bool = False,
    objective: str | None = None,
) -> tuple:
    """Compare objective state quality, risk, and search completeness.

    Completeness is evidence about how much of a line was searched; it is not
    a quality score for the resulting state. In particular, a projection that
    spends a Potion for two HP must compete on its resulting state against a
    weakening move or ball throw. Proven objective success remains best, while
    an unresolved line remains preferable to a proven objective failure when
    it still has a viable continuation.
    """

    effective_category = category or value.category
    if objective == "capture":
        # Capture roots are an expected-utility decision. Search completeness
        # remains part of the result and safety gate, but it must not make a
        # terminal encounter failure outrank a materially better incomplete
        # capture continuation (or vice versa) by category alone.
        return (
            value.score,
            value.capture,
            -value.encounter_loss,
            -value.active_faint,
            -value.party_wipe,
            -value.run_loss,
            int(effective_category is SearchResultCategory.TERMINAL_COMPLETE),
            int(value.projection_safe),
        )
    terminal_success = (
        effective_category is SearchResultCategory.TERMINAL_COMPLETE
        and value.win >= 1.0 - 1e-9
        and value.loss <= 1e-9
        and value.party_wipe <= 1e-9
        and value.run_loss <= 1e-9
        and value.encounter_loss <= 1e-9
    )
    unresolved_line = effective_category in (
        SearchResultCategory.INCOMPLETE,
        SearchResultCategory.HORIZON_PROJECTION,
    ) and not value.objective_failed and max(value.party_wipe, value.run_loss, value.loss, value.encounter_loss) < 1.0
    unresolved_nonterminal = (
        prefer_unresolved
        and unresolved_line
        and value.party_wipe < 1.0
        and value.run_loss < 1.0
        and value.loss < 1.0
        and value.encounter_loss < 1.0
    )
    if terminal_success:
        outcome_rank = 4
    elif effective_category is SearchResultCategory.TERMINAL_COMPLETE:
        # A completed stochastic result with a real objective-success branch
        # is more informative than an unresolved projection. A completed
        # result that only proves failure remains below a viable incomplete
        # line.
        outcome_rank = 3 if value.win > 0.0 else 1
    elif unresolved_line:
        # ``projection_safe`` remains a confidence/reporting field. It does
        # not elevate one incomplete successor state above another.
        outcome_rank = 2
    elif unresolved_nonterminal:
        # At the root, an unresolved line with no established terminal loss
        # is still preferable to a proven failed root. It remains below a
        # safe projection and is still rejected by ``SearchResult.is_safe``
        # when its projection evidence is insufficient.
        outcome_rank = 2
    elif effective_category is SearchResultCategory.TERMINAL_COMPLETE:
        outcome_rank = 1
    else:
        # An incomplete line with a proven catastrophic outcome remains below
        # a terminal failure. An unresolved line with no established terminal
        # loss was handled above, even when it is not safe to execute. The
        # latter distinction lets the planner explain the best unresolved
        # root without granting it execution permission.
        outcome_rank = 0
    # Carry risk and objective failure back into the scalar tie-break with the
    # same order of magnitude as a terminal battle value. This keeps a small,
    # acceptable risk from becoming a hard veto against objective progress,
    # while a near-certain loss dominates a no-progress setup line.
    unresolved_risk = max(
        value.loss,
        value.party_wipe,
        value.run_loss,
        value.encounter_loss,
    )
    risk_adjusted_score = value.score - 100000.0 * unresolved_risk
    # A bounded root with immediate survival and objective progress outranks
    # a root whose apparent value came from an unmodified pre-action state.
    # This is the key distinction between a useful partial line and a
    # speculative switch or item action.
    projection_rank = int(value.projection_safe) if effective_category is not SearchResultCategory.TERMINAL_COMPLETE else 0
    return (
        outcome_rank,
        projection_rank,
        risk_adjusted_score,
        -value.party_wipe,
        -value.run_loss,
        -value.loss,
        -value.encounter_loss,
    )


@dataclass(frozen=True, slots=True)
class _MoveOutcome:
    probability: float
    damage: int
    secondary_effect: str | None = None
    critical: bool = False
    hit: bool = True
    label: str = "HIT"


def _move_effect_key(move: SearchMove) -> tuple:
    """Return the state-changing portion of a normalized move."""

    return (
        move.secondary_chance,
        move.secondary_effect,
        move.stat_target,
        move.stat_delta,
        move.damage_stat,
        move.critical_damage_max,
        move.critical_hit_chance,
        move.causes_opponent_escape,
        move.pp,
    )


def _deduplicate_moves(moves: tuple[SearchMove, ...]) -> tuple[SearchMove, ...]:
    """Collapse actions with identical normalized outcomes."""

    seen = set()
    result = []
    for move in moves:
        key = (
            move.damage_min,
            move.damage_max,
            move.accuracy,
            move.priority,
            _move_effect_key(move),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(move)
    return tuple(result)


def _prune_dominated_player_moves(moves: tuple[SearchMove, ...]) -> tuple[SearchMove, ...]:
    """Remove strictly weaker copies of the same state-changing action.

    Distinct setup/status effects remain candidates. A move is dominated only
    when another move has the same effect envelope and is at least as good on
    every direct outcome dimension, including priority and accuracy.
    """

    unique = _deduplicate_moves(moves)
    result = []
    for candidate in unique:
        dominated = any(
            other is not candidate
            and _move_effect_key(other) == _move_effect_key(candidate)
            and other.priority >= candidate.priority
            and other.accuracy >= candidate.accuracy
            and other.damage_min >= candidate.damage_min
            and other.damage_max >= candidate.damage_max
            and (
                other.accuracy > candidate.accuracy
                or other.damage_min > candidate.damage_min
                or other.damage_max > candidate.damage_max
                or other.priority > candidate.priority
            )
            for other in unique
        )
        if not dominated:
            result.append(candidate)
    return tuple(result)


def _objective_move_candidates(state: SearchState) -> tuple[SearchMove, ...]:
    """Keep defeat/training lines focused on actions that advance the KO.

    Stat-only setup can be useful for a specialized objective, but it is not
    an acceptable default answer to a defeat or training objective when a
    damaging move is available. Capture remains free to compare setup,
    weakening, balls, and survival through its own state valuation.
    """

    if state.player_bide_turns > 0:
        # Emerald's multi-turn lock overrides ordinary move selection. The
        # move remains in the normalized moveset because the initial Bide
        # action started the lock.
        return tuple(move for move in state.player_moves if _is_bide_move(move))
    moves = tuple(
        move
        for move in state.player_moves
        if (move.pp is None or move.pp > 0)
        and _move_has_effective_player_action(state, move)
    )
    if state.campaign_owned and state.battle_objective in {"defeat", "training"}:
        damaging = tuple(move for move in moves if move.damage_max > 0)
        if damaging:
            moves = damaging
    return moves


def _move_has_effective_player_action(state: SearchState, move: SearchMove) -> bool:
    """Return whether a normalized player move can change the battle state.

    A zero-damage stat move at its Gen III stage floor consumes a turn and PP
    without changing anything. Keeping it in the action set lets a bounded
    search repeatedly select an apparent setup action even though the game
    will report that the stat cannot fall any further.
    """

    if move.damage_max > 0:
        return True
    effect = _stat_effect_info(move, attacker_is_player=True)
    if effect is None:
        return True
    target_is_player, stat, delta = effect
    stages = _stage_map(
        state.player_stat_stages if target_is_player else state.opponent_stat_stages
    )
    current = stages.get(stat, 0)
    projected = max(-6, min(6, current + delta))
    return projected != current


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _move_outcomes(move: SearchMove, *, attacker_is_player: bool) -> tuple[_MoveOutcome, ...]:
    """Return damage and secondary-effect branches without choosing one.

    ``attacker_is_player`` is retained in the adapter signature for callers
    that build role-specific move facts. The outcome distribution itself is
    symmetric: the live battle, rather than this search, owns RNG state.
    """

    accuracy = _clamp_probability(move.accuracy)
    stat_effect = _stat_effect_info(move, attacker_is_player=attacker_is_player)
    if move.damage_max <= 0:
        # Growl, String Shot, and similar moves are actions with a state
        # transition. They must not collapse into a free no-op in the tree.
        return (_MoveOutcome(1.0, 0, move.secondary_effect or "__STAT_EFFECT__", label="EFFECT"),) if stat_effect else (
            _MoveOutcome(1.0, 0, label="NO_DAMAGE"),
        )
    low = max(0, int(move.damage_min))
    high = max(low, int(move.damage_max))
    critical = max(low, int(move.critical_damage_max if move.critical_damage_max is not None else high))
    critical_chance = _clamp_probability(move.critical_hit_chance)
    # A critical branch is not exposed as a predicted event. It is simply one
    # possible branch in the distribution used to assess risk.
    labeled_branches: list[tuple[float, int, bool, bool, str]] = []
    if accuracy < 1:
        labeled_branches.append((1 - accuracy, 0, False, False, "MISS"))
    ordinary_probability = accuracy * (1 - critical_chance)
    critical_probability = accuracy * critical_chance
    if ordinary_probability:
        if high == low:
            labeled_branches.append((ordinary_probability, low, False, True, "HIT"))
        else:
            labeled_branches.extend(
                (
                    (ordinary_probability * 0.5, low, False, True, "HIT"),
                    (ordinary_probability * 0.5, high, False, True, "HIT"),
                )
            )
    if critical_probability:
        labeled_branches.append((critical_probability, critical, True, True, "CRITICAL"))
    total = sum(probability for probability, *_ in labeled_branches)
    if total <= 0:
        return (_MoveOutcome(1.0, 0, label="NO_DAMAGE"),)
    secondary_chance = _clamp_probability(move.secondary_chance)
    secondary_effect = move.secondary_effect
    # The extractor uses a generic HIT effect for ordinary damaging moves and
    # historically converted a ROM zero secondary byte to 100%. Neither fact
    # describes a real status branch. Keep actual *_HIT and stat effects, but
    # suppress generic effect labels defensively at the normalized boundary.
    if secondary_effect in {"HIT", "ALWAYS_HIT", "MULTI_HIT", "DOUBLE_HIT"}:
        secondary_chance = 0.0
        secondary_effect = None
    outcomes = []
    for probability, damage, critical_hit, hit, label in labeled_branches:
        normalized_probability = probability / total
        if secondary_effect and secondary_chance > 0 and damage > 0:
            outcomes.append(
                _MoveOutcome(
                    normalized_probability * (1 - secondary_chance),
                    damage,
                    critical=critical_hit,
                    hit=hit,
                    label=label,
                )
            )
            outcomes.append(
                _MoveOutcome(
                    normalized_probability * secondary_chance,
                    damage,
                    secondary_effect,
                    critical_hit,
                    hit,
                    f"{label}+STATUS",
                )
            )
        else:
            outcomes.append(
                _MoveOutcome(
                    normalized_probability,
                    damage,
                    critical=critical_hit,
                    hit=hit,
                    label=label,
                )
            )
    return tuple(outcomes)


def _is_bide_move(move: SearchMove) -> bool:
    return str(move.name or "").strip().casefold() == "bide" or str(
        move.secondary_effect or ""
    ).strip().upper() == "BIDE"


def _bide_state(state: SearchState, *, attacker_is_player: bool) -> tuple[int, int]:
    if attacker_is_player:
        return state.player_bide_turns, state.player_bide_damage
    return state.opponent_bide_turns, state.opponent_bide_damage


def _move_outcomes_for_state(
    state: SearchState,
    move: SearchMove,
    *,
    attacker_is_player: bool,
) -> tuple[_MoveOutcome, ...]:
    """Return move outcomes after applying stateful Gen III Bide rules."""

    if not _is_bide_move(move):
        return _move_outcomes(move, attacker_is_player=attacker_is_player)

    turns, stored_damage = _bide_state(state, attacker_is_player=attacker_is_player)
    if turns <= 0:
        # The initial Bide command starts the lock and never deals the base
        # power-one damage shown in the ROM move table.
        return (_MoveOutcome(1.0, 0, label="BIDE_CHARGE"),)
    if turns > 1:
        return (_MoveOutcome(1.0, 0, label="BIDE_STORING"),)
    if stored_damage <= 0:
        return (_MoveOutcome(1.0, 0, label="BIDE_NO_ENERGY"),)

    # Emerald still performs Bide's accuracy check on release.  Ordinary Bide
    # has 100% accuracy, but preserving this branch keeps synthetic callers
    # honest when they provide a different accuracy fact.
    accuracy = _clamp_probability(move.accuracy)
    release = _MoveOutcome(accuracy, stored_damage * 2, label="BIDE_RELEASE")
    if accuracy >= 1.0:
        return (release,)
    return (
        _MoveOutcome(1.0 - accuracy, 0, label="MISS"),
        release,
    )


def _resolved_move_damage(
    state: SearchState,
    move: SearchMove,
    outcome: _MoveOutcome,
    *,
    attacker_is_player: bool,
) -> int:
    """Resolve damage after stateful move behavior and stat-stage modifiers."""

    if _is_bide_move(move):
        # Bide's release damage is already the stored direct HP damage × 2;
        # Emerald clears type-effectiveness flags and does not apply ordinary
        # attack/defense stage scaling to this copied damage.
        return max(0, int(outcome.damage))
    return _relative_damage(state, move, outcome.damage, attacker_is_player=attacker_is_player)


_STAT_EFFECTS = {
    "ATTACK_UP": ("self", "attack", 1),
    "ATTACK_UP_2": ("self", "attack", 2),
    "ATTACK_DOWN": ("opponent", "attack", -1),
    "ATTACK_DOWN_2": ("opponent", "attack", -2),
    "DEFENSE_UP": ("self", "defense", 1),
    "DEFENSE_UP_2": ("self", "defense", 2),
    "DEFENSE_DOWN": ("opponent", "defense", -1),
    "DEFENSE_DOWN_2": ("opponent", "defense", -2),
    "SPEED_UP": ("self", "speed", 1),
    "SPEED_UP_2": ("self", "speed", 2),
    "SPEED_DOWN": ("opponent", "speed", -1),
    "SPEED_DOWN_2": ("opponent", "speed", -2),
    "SPECIAL_ATTACK_UP": ("self", "special_attack", 1),
    "SPECIAL_ATTACK_UP_2": ("self", "special_attack", 2),
    "SPECIAL_ATTACK_DOWN": ("opponent", "special_attack", -1),
    "SPECIAL_ATTACK_DOWN_2": ("opponent", "special_attack", -2),
    "SPECIAL_DEFENSE_UP": ("self", "special_defense", 1),
    "SPECIAL_DEFENSE_UP_2": ("self", "special_defense", 2),
    "SPECIAL_DEFENSE_DOWN": ("opponent", "special_defense", -1),
    "SPECIAL_DEFENSE_DOWN_2": ("opponent", "special_defense", -2),
}


def _stat_effect_info(move: SearchMove, *, attacker_is_player: bool) -> tuple[bool, str, int] | None:
    effect = str(move.secondary_effect or "").upper().replace(" ", "_")
    mapped = _STAT_EFFECTS.get(effect) or _STAT_EFFECTS.get(effect.removesuffix("_HIT"))
    if mapped is None and move.stat_delta:
        mapped = (move.stat_target or "self", move.damage_stat or "attack", move.stat_delta)
    if mapped is None:
        return None
    target, stat, delta = mapped
    target_is_player = attacker_is_player if target == "self" else not attacker_is_player
    return target_is_player, stat, delta


def _stage_multiplier(stage: int) -> float:
    stage = max(-6, min(6, int(stage)))
    return (2.0 + stage) / 2.0 if stage >= 0 else 2.0 / (2.0 - stage)


def _stage_map(stages: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return {str(stat): max(-6, min(6, int(value))) for stat, value in stages}


def _stage_tuple(stages: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((stat, value) for stat, value in stages.items() if value))


def _effective_speed(state: SearchState, *, player: bool) -> float | None:
    speed = state.player_speed if player else state.opponent_speed
    if speed is None:
        return None
    stages = _stage_map(state.player_stat_stages if player else state.opponent_stat_stages)
    return float(speed) * _stage_multiplier(stages.get("speed", 0))


def _relative_damage(state: SearchState, move: SearchMove, damage: int, *, attacker_is_player: bool) -> int:
    """Apply forecast stat-stage changes to the already-live damage range."""

    stat = move.damage_stat or "attack"
    defense_stat = {
        "attack": "defense",
        "special_attack": "special_defense",
    }.get(stat, "defense")
    attacker_stages = _stage_map(state.player_stat_stages if attacker_is_player else state.opponent_stat_stages)
    defender_stages = _stage_map(state.opponent_stat_stages if attacker_is_player else state.player_stat_stages)
    multiplier = _stage_multiplier(attacker_stages.get(stat, 0)) / _stage_multiplier(defender_stages.get(defense_stat, 0))
    return max(0, int(round(max(0, damage) * multiplier)))


def _apply_move_effect(
    state: SearchState,
    move: SearchMove,
    outcome: _MoveOutcome,
    *,
    attacker_is_player: bool,
    target_is_player: bool,
) -> SearchState:
    state = _apply_secondary(state, outcome.secondary_effect, target_is_player=target_is_player)
    if _is_bide_move(move):
        if attacker_is_player:
            if state.player_bide_turns > 0:
                if state.player_bide_turns == 1:
                    state = replace(
                        state,
                        player_bide_turns=0,
                        player_bide_damage=0,
                        player_bide_target=None,
                    )
                else:
                    state = replace(state, player_bide_turns=state.player_bide_turns - 1)
            else:
                state = replace(
                    state,
                    player_bide_turns=2,
                    player_bide_damage=0,
                    player_bide_target=None,
                )
        elif state.opponent_bide_turns > 0:
            if state.opponent_bide_turns == 1:
                state = replace(
                    state,
                    opponent_bide_turns=0,
                    opponent_bide_damage=0,
                    opponent_bide_target=None,
                )
            else:
                state = replace(state, opponent_bide_turns=state.opponent_bide_turns - 1)
        else:
            state = replace(
                state,
                opponent_bide_turns=2,
                opponent_bide_damage=0,
                opponent_bide_target=None,
            )
    if outcome.secondary_effect is None:
        return state
    effect = _stat_effect_info(move, attacker_is_player=attacker_is_player)
    if effect is None:
        return state
    effect_target_is_player, stat, delta = effect
    if effect_target_is_player != target_is_player:
        return state
    stages = _stage_map(state.player_stat_stages if target_is_player else state.opponent_stat_stages)
    stages[stat] = max(-6, min(6, stages.get(stat, 0) + delta))
    return replace(
        state,
        player_stat_stages=_stage_tuple(stages) if target_is_player else state.player_stat_stages,
        opponent_stat_stages=_stage_tuple(stages) if not target_is_player else state.opponent_stat_stages,
    )


def _apply_move_damage(
    state: SearchState,
    move: SearchMove,
    outcome: _MoveOutcome,
    *,
    attacker_is_player: bool,
) -> SearchState:
    """Apply direct HP damage and charge an active Bide with actual damage."""

    damage = _resolved_move_damage(
        state,
        move,
        outcome,
        attacker_is_player=attacker_is_player,
    )
    if attacker_is_player:
        before_hp = state.opponent_hp
        target_hp = max(0, before_hp - damage)
        state = replace(state, opponent_hp=target_hp)
        actual_damage = max(0, before_hp - target_hp)
        if actual_damage and state.opponent_bide_turns > 0:
            state = replace(
                state,
                opponent_bide_damage=state.opponent_bide_damage + actual_damage,
                opponent_bide_target="player",
            )
        return state

    before_hp = state.player_hp
    target_hp = max(0, before_hp - damage)
    state = replace(state, player_hp=target_hp)
    actual_damage = max(0, before_hp - target_hp)
    if actual_damage and state.player_bide_turns > 0:
        state = replace(
            state,
            player_bide_damage=state.player_bide_damage + actual_damage,
            player_bide_target="opponent",
        )
    return state


def _cancel_bide(state: SearchState, *, player: bool) -> SearchState:
    """Apply Emerald's multi-turn cancellation boundary."""

    if player:
        return replace(
            state,
            player_bide_turns=0,
            player_bide_damage=0,
            player_bide_target=None,
        )
    return replace(
        state,
        opponent_bide_turns=0,
        opponent_bide_damage=0,
        opponent_bide_target=None,
    )


def _consume_player_move_pp(state: SearchState, move: SearchMove) -> SearchState:
    """Consume one PP for a simulated player move when PP is known."""

    if move.pp is None or (_is_bide_move(move) and state.player_bide_turns > 0):
        return state
    remaining_pp = max(0, int(move.pp) - 1)
    moves = tuple(
        replace(candidate, pp=remaining_pp)
        if candidate.index == move.index and candidate.name == move.name
        else candidate
        for candidate in state.player_moves
    )
    return replace(state, player_moves=moves)


def _secondary_status(effect: str | None) -> str | None:
    normalized = str(effect or "").upper()
    for status in ("PARALYZE", "SLEEP", "BURN", "POISON", "FREEZE", "CONFUSE"):
        if status in normalized:
            return status
    return None


def _apply_secondary(state: SearchState, effect: str | None, *, target_is_player: bool) -> SearchState:
    status = _secondary_status(effect)
    if status is None:
        return state
    if target_is_player:
        if not _status_is_healthy(state.player_status):
            return state
        return replace(state, player_status=status)
    if not _status_is_healthy(state.opponent_status):
        return state
    return replace(state, opponent_status=status)


def _status_action_probability(status: str | None) -> float:
    """Use a distribution for status interference, never a predicted turn."""

    normalized = str(status or "").upper()
    return {
        "PARALYZE": 0.75,
        "PARALYZED": 0.75,
        "PARALYSED": 0.75,
        "SLEEP": 0.34,
        "FREEZE": 0.20,
        "CONFUSE": 0.50,
    }.get(normalized, 1.0)


def _apply_residual_damage(state: SearchState) -> SearchState:
    """Apply modeled end-of-turn residual damage as another known branch fact."""

    def residual(status: str | None, maximum: int) -> int:
        return max(1, maximum // 8) if str(status or "").upper() in {"BURN", "POISON"} else 0

    player_damage = residual(state.player_status, state.player_max_hp)
    opponent_damage = residual(state.opponent_status, state.opponent_max_hp)
    if not player_damage and not opponent_damage:
        return state
    return replace(
        state,
        player_hp=max(0, state.player_hp - player_damage),
        opponent_hp=max(0, state.opponent_hp - opponent_damage),
    )


def _acts_first(
    player: bool,
    move: SearchMove,
    state: SearchState,
    other_move: SearchMove | None = None,
    *,
    tie_probability: bool = False,
) -> bool | None:
    """Return turn order against one specific opposing action.

    Gen III resolves priority per selected move, then uses a random speed tie.
    Returning ``None`` for a tie lets the caller evaluate both orderings rather
    than silently giving the player the first turn.
    """

    if other_move is None:
        other_moves = state.opponent_moves if player else state.player_moves
        priorities = tuple(candidate.priority for candidate in other_moves)
        if not priorities:
            other_priority = 0
        elif len(set(priorities)) == 1:
            other_priority = priorities[0]
        else:
            # No single ordering exists until the opponent action is selected.
            # Preserve the old boolean contract for callers that only need a
            # conservative pre-check, while action-specific callers pass the
            # selected opposing move.
            other_priority = max(priorities)
    else:
        other_priority = other_move.priority
    if move.priority != other_priority:
        return move.priority > other_priority if player else move.priority >= other_priority
    player_speed = _effective_speed(state, player=True)
    opponent_speed = _effective_speed(state, player=False)
    if player_speed is None or opponent_speed is None:
        return None if tie_probability else player
    if player_speed == opponent_speed:
        return None if tie_probability else player
    return player_speed > opponent_speed if player else opponent_speed > player_speed


def _faint_value(state: SearchState, line: tuple[str, ...]) -> _Value:
    """Represent an active faint and independently classify a party wipe."""

    viable_switch = any(
        switch.hp > 0 and switch.max_hp > 0 and switch.moves
        for switch in state.player_switches
    )
    party_wipe = 0.0 if viable_switch else 1.0
    return _Value(
        -100000.0,
        0.0,
        1.0,
        line,
        active_faint=1.0,
        party_wipe=party_wipe,
        run_loss=party_wipe,
    )


class Expectiminimax:
    """Evaluate a small single battle with bounded stochastic search."""

    def __init__(self, config: SearchConfig | None = None, *, opponent_selector: Callable | None = None):
        self.config = config or SearchConfig()
        self.opponent_selector = opponent_selector
        self.expanded_nodes = 0
        self._budget_exhausted = False
        self._budget_exhaustion: str | None = None
        self._deadline: float | None = None
        self._root_node_limit: int | None = None
        self._root_nodes_start = 0
        self._turn_limit = self.config.max_turns
        # Search states are immutable and hashable.  Cache the value of each
        # transposed state so damage-roll branches that reach the same HP and
        # status position do not recursively rebuild the remaining battle.
        # Lines are stored relative to the cached node, so a transposition does
        # not make the principal variation depend on the first path that found
        # it.
        self._memo: dict[tuple[str, SearchState, int, SearchMove | None], _Value] = {}

    @staticmethod
    def _switch_candidates(state: SearchState) -> tuple[SearchSwitch, ...]:
        """Return switches that do not immediately reverse recent movement."""

        if not state.campaign_owned:
            return tuple(
                switch
                for switch in state.player_switches
                if switch.party_index != state.player_party_index
            )
        # The optional trainer replacement is free and is evaluated before the
        # incoming Pokémon gets its turn. The ordinary anti-oscillation guard
        # must not hide a training choice or the previous active member here.
        if state.free_switch:
            return tuple(
                switch
                for switch in state.player_switches
                if switch.party_index != state.player_party_index
            )
        response_ceiling = Expectiminimax._response_ceiling(state)
        emergency_reversal = response_ceiling > 0 and state.player_hp <= response_ceiling
        blocked = frozenset() if emergency_reversal else frozenset(state.switch_history[-2:])
        return tuple(
            switch
            for switch in state.player_switches
            if switch.party_index not in blocked and switch.party_index != state.player_party_index
        )

    @classmethod
    def _capture_switch_allowed(cls, state: SearchState) -> bool:
        """Allow a capture switch only when it changes immediate survivability."""

        if not (state.campaign_owned and state.capture_target and state.battle_objective == "capture"):
            return True
        response_ceiling = cls._response_ceiling(state)
        if response_ceiling > 0 and state.player_hp <= response_ceiling:
            return True
        has_nonlethal_progress = any(
            move.damage_max > 0 and move.damage_max < state.opponent_hp
            for move in state.player_moves
        )
        # When the active lead can survive the known response and has a legal
        # weakening action, changing leads spends a turn without moving the
        # encounter toward a ball throw.  This prevents a capture plan from
        # oscillating between healthy party members while the horizon keeps
        # projecting a future capture after every switch.
        return not has_nonlethal_progress

    def evaluate(self, state: SearchState) -> SearchResult:
        if state.root_opponent_hp is None:
            state = replace(state, root_opponent_hp=state.opponent_hp)
        self.expanded_nodes = 0
        self._budget_exhausted = False
        self._budget_exhaustion = None
        duration = self.config.max_duration_seconds
        self._deadline = monotonic() + duration if duration is not None else None
        self._turn_limit = self.config.max_turns
        self._memo.clear()
        if state.player_hp <= 0:
            party_wipe = 0.0 if any(
                switch.hp > 0 and switch.max_hp > 0 and switch.moves
                for switch in state.player_switches
            ) else 1.0
            return SearchResult(
                None,
                None,
                -100000.0,
                0.0,
                1.0,
                party_wipe,
                reason="player is already fainted",
                objective=state.battle_objective,
                category=SearchResultCategory.TERMINAL_COMPLETE,
                active_faint_probability=1.0,
                party_wipe_probability=party_wipe,
                run_loss_probability=party_wipe,
            )
        if state.opponent_hp <= 0:
            value = self._opponent_defeated_value(state, ("KO",))
            return SearchResult(
                None,
                None,
                value.score,
                value.win,
                value.loss,
                value.party_wipe,
                principal_variation=value.line,
                capture_probability=value.capture,
                encounter_loss_probability=value.encounter_loss,
                expected_experience=value.expected_experience,
                experience_utility=value.experience_utility,
                reason="opponent is already defeated",
                objective=state.battle_objective,
                category=SearchResultCategory.TERMINAL_COMPLETE,
                active_faint_probability=value.active_faint,
                party_wipe_probability=value.party_wipe,
                run_loss_probability=value.run_loss,
            )
        # Lethal weakening moves remain in the tree. Their KO branches are
        # objective failures, but a fixed HP threshold must not decide the
        # capture policy before odds, remaining balls, crit risk, and the
        # resulting state have been compared.
        capture_eligible_moves = _objective_move_candidates(state)
        move_candidates = (
            _deduplicate_moves(capture_eligible_moves)
            if state.capture_target and state.battle_objective == "capture"
            else _prune_dominated_player_moves(capture_eligible_moves)
        )
        root_actions = [
            ("move", move, move.index, move.name)
            for move in move_candidates
        ]
        if state.player_bide_turns <= 0:
            root_actions.extend(
                ("switch", switch, switch.party_index, switch.name)
                for switch in self._switch_candidates(state)
                if switch.hp > 0
                and switch.max_hp > 0
                and switch.moves
                and self._capture_switch_allowed(state)
            )
            for item in state.player_items:
                if item.quantity <= 0:
                    continue
                if item.is_capture_ball:
                    if state.capture_target and state.battle_objective == "capture":
                        root_actions.append(("item", item, item.item_index, item.name))
                    continue
                # Keep a merely injured target in the root report for
                # diagnostics, but let _player_item_value reject a no-op or
                # ineffective heal.
                healed_hp = state.player_max_hp if item.full_heal else min(
                    state.player_max_hp,
                    state.player_hp + max(0, item.heal_amount),
                )
                if (
                    healed_hp > state.player_hp
                    or self._healing_item_is_materially_needed(state, item)
                ):
                    root_actions.append(("item", item, item.item_index, item.name))
            if _run_action_available(state):
                root_actions.append(("run", None, None, "Run Away"))
        # Keep unnecessary items in the root report when another legal action
        # exists, but do not turn a stale healing action into the only action
        # available to the executor.
        if not any(kind != "item" for kind, *_ in root_actions):
            root_actions = [
                candidate
                for candidate in root_actions
                if candidate[1].is_capture_ball
                or self._healing_item_is_materially_needed(state, candidate[1])
            ]
        if not root_actions:
            return SearchResult(None, None, -100000.0, 0.0, 1.0, 1.0, reason="no legal player moves")

        if state.capture_target and state.battle_objective == "capture":
            # Give an immediately legal throw the first root slot. Iterative
            # deepening still compares every root, so this is only a search
            # scheduling/tie-break preference; capture correctness comes from
            # the finite sequence valuation below.
            root_actions.sort(
                key=lambda candidate: 0
                if candidate[0] == "item" and candidate[1].is_capture_ball
                else 1
            )

        # Use iterative deepening at the root. The first pass gives every
        # action a shallow post-action comparison; later passes deepen the
        # same roots with the remaining budget. Recursive search remains
        # depth-first inside each root, but an early move line cannot starve a
        # switch or item before its immediate state has been scored.
        root_count = len(root_actions)
        global_deadline = self._deadline
        root_node_budget = max(1, self.config.max_nodes // root_count)
        root_nodes_used: dict[tuple[str, int | None], int] = {}
        best_by_root: dict[tuple[str, int | None], tuple] = {}
        any_root_budget_exhausted = False
        first_budget_exhaustion = None
        depth_rounds = tuple(dict.fromkeys((1, 2, max(1, self.config.max_turns))))
        shallow_phase_end = (
            global_deadline
            if global_deadline is None or len(depth_rounds) == 1
            else global_deadline - (self.config.max_duration_seconds * 0.35)
        )
        for round_number, turn_limit in enumerate(depth_rounds):
            phase_end = shallow_phase_end if round_number == 0 else global_deadline
            remaining_roots = list(root_actions)
            while remaining_roots:
                action_kind, action, action_index, action_name = remaining_roots.pop(0)
                root_key = (action_kind, action_index)
                remaining_nodes = root_node_budget - root_nodes_used.get(root_key, 0)
                if remaining_nodes <= 0:
                    continue
                self._memo.clear()
                self._budget_exhausted = False
                self._budget_exhaustion = None
                self._root_nodes_start = self.expanded_nodes
                self._root_node_limit = remaining_nodes
                self._turn_limit = turn_limit
                if phase_end is not None:
                    remaining_time = max(0.0, phase_end - monotonic())
                    root_duration = remaining_time / max(1, len(remaining_roots) + 1)
                    self._deadline = min(
                        global_deadline,
                        monotonic() + root_duration,
                    ) if global_deadline is not None else monotonic() + root_duration
                else:
                    self._deadline = global_deadline
                if action_kind == "move":
                    value = self._player_move_value(state, action, 0)
                elif action_kind == "switch":
                    value = self._player_switch_value(state, action, 0)
                elif action_kind == "item":
                    value = self._player_item_value(state, action, 0)
                else:
                    value = self._player_run_value(state, 0)
                root_nodes_used[root_key] = root_nodes_used.get(root_key, 0) + (
                    self.expanded_nodes - self._root_nodes_start
                )
                root_limit_reached = root_nodes_used[root_key] >= root_node_budget
                if self._budget_exhausted:
                    # The recursive branch may have stopped before applying
                    # the root action. Replace that stale pre-action score with
                    # a bounded one-turn projection of the actual successor.
                    value = self._root_projection_value(state, action_kind, action, action_name)
                root_category = (
                    SearchResultCategory.INCOMPLETE
                    if self._budget_exhausted or (
                        root_limit_reached
                        and value.category is not SearchResultCategory.TERMINAL_COMPLETE
                    )
                    else value.category
                )
                candidate_budget_exhaustion = self._budget_exhaustion or (
                    "nodes" if root_limit_reached and root_category is SearchResultCategory.INCOMPLETE else None
                )
                candidate = (
                    value,
                    action_kind,
                    action_index,
                    action_name,
                    root_category,
                    candidate_budget_exhaustion,
                )
                previous = best_by_root.get(root_key)
                if previous is None:
                    best_by_root[root_key] = candidate
                else:
                    capture_root = state.capture_target and state.battle_objective == "capture"
                    if capture_root:
                        # Iterative deepening refines one root over several
                        # passes. Keep the most complete evidence for that
                        # same action, then compare its expected utility with
                        # the other roots. This prevents an optimistic shallow
                        # projection from hiding a later terminal encounter
                        # failure while still keeping completeness out of the
                        # cross-root objective ranking.
                        category_rank = {
                            SearchResultCategory.INCOMPLETE: 0,
                            SearchResultCategory.HORIZON_PROJECTION: 1,
                            SearchResultCategory.TERMINAL_COMPLETE: 2,
                        }
                        replace_root = category_rank[root_category] > category_rank[previous[4]]
                        if category_rank[root_category] == category_rank[previous[4]]:
                            replace_root = _value_selection_key(
                                value,
                                category=root_category,
                                prefer_unresolved=True,
                                objective="capture",
                            ) > _value_selection_key(
                                previous[0],
                                category=previous[4],
                                prefer_unresolved=True,
                                objective="capture",
                            )
                    else:
                        candidate_key = _value_selection_key(
                            value,
                            category=root_category,
                            prefer_unresolved=True,
                        )
                        previous_key = _value_selection_key(
                            previous[0],
                            category=previous[4],
                            prefer_unresolved=True,
                        )
                        replace_root = candidate_key > previous_key or (
                            candidate_key == previous_key
                            and candidate[5] is None
                            and previous[5] is not None
                        )
                    if replace_root:
                        best_by_root[root_key] = candidate
                if self._budget_exhausted or root_limit_reached:
                    any_root_budget_exhausted = True
                    first_budget_exhaustion = first_budget_exhaustion or candidate_budget_exhaustion
            if phase_end is not None and monotonic() >= phase_end:
                continue
        self._root_node_limit = None
        self._deadline = global_deadline
        self._turn_limit = self.config.max_turns
        candidates = [
            best_by_root[(action_kind, action_index)]
            for action_kind, _action, action_index, _action_name in root_actions
            if (action_kind, action_index) in best_by_root
        ]
        self._budget_exhausted = any_root_budget_exhausted
        self._budget_exhaustion = first_budget_exhaustion

        def avoidance_run_is_survivable(candidate) -> bool:
            """Return whether a failed escape attempt leaves an acceptable line.

            Avoidance is a root objective, so a safe Run Away action must be
            compared before ordinary outcome quality.  The exception is an
            escape whose failure can cost the active Pokémon or the run.  A
            bounded or horizon result cannot use its default zero risk as
            proof; it needs a complete response envelope and enough HP for
            the worst known response.
            """

            value, candidate_kind, _candidate_index, _name, category, _budget = candidate
            if (
                state.battle_objective not in {"avoid", "transit"}
                or candidate_kind != "run"
            ):
                return False

            probability = _escape_probability(state)
            if probability >= 1.0:
                return True

            # A terminal category can still be an artifact of searching only
            # the observed response set.  A non-guaranteed escape therefore
            # cannot receive objective priority while that set is incomplete.
            if not state.response_facts_complete:
                return False

            risk_is_acceptable = (
                value.active_faint <= self.config.maximum_safe_catastrophic_probability
                and value.party_wipe <= self.config.maximum_safe_catastrophic_probability
                and value.run_loss <= self.config.maximum_safe_loss_probability
                and value.loss <= self.config.maximum_safe_loss_probability
            )
            if not risk_is_acceptable:
                return False

            if category is SearchResultCategory.TERMINAL_COMPLETE:
                return True

            # A known response ceiling can make a bounded line usable when
            # the response envelope itself is complete.
            return state.player_hp > self._response_ceiling(state)

        def avoidance_escape_probability(candidate) -> float | None:
            """Return the modeled escape chance for an avoid root.

            A switch is a useful avoid root when its successor line reaches a
            survivable Run Away action. Comparing that line's success
            probability with the active Pokémon's direct Run Away chance lets
            a faster replacement win when it materially improves the escape
            outcome.
            """

            value, candidate_kind, _candidate_index, _name, _category, _budget = candidate
            if state.battle_objective not in {"avoid", "transit"}:
                return None
            if candidate_kind == "run":
                if not avoidance_run_is_survivable(candidate):
                    return None
            elif candidate_kind == "switch":
                if "Run Away" not in value.line:
                    return None
                if not state.response_facts_complete:
                    return None
                if (
                    value.active_faint > self.config.maximum_safe_catastrophic_probability
                    or value.party_wipe > self.config.maximum_safe_catastrophic_probability
                    or value.run_loss > self.config.maximum_safe_loss_probability
                    or value.loss > self.config.maximum_safe_loss_probability
                ):
                    return None
            else:
                return None
            return _clamp_probability(value.win)

        def root_immediate_survival(candidate) -> int:
            """Prefer a root that survives the next known response.

            A full line may still be incomplete, but using a healing item that
            raises the active member above the known critical response is
            materially better than taking an action whose first response
            immediately faints it. This is a root-local survival fact; it does
            not replace the deeper expectiminimax result.
            """

            _value, candidate_kind, candidate_index, _name, _category, _budget = candidate
            if _value_selection_key(_value, category=_category)[0] >= 4:
                return 0
            if candidate_kind == "item":
                item = next(
                    (item for item in state.player_items if item.item_index == candidate_index),
                    None,
                )
                if item is not None and not item.is_capture_ball:
                    healed_hp = (
                        state.player_max_hp
                        if item.full_heal
                        else min(state.player_max_hp, state.player_hp + max(0, item.heal_amount))
                    )
                    response_ceiling = max(
                        (
                            move.critical_damage_max
                            if move.critical_damage_max is not None
                            else move.damage_max
                            for move in state.opponent_moves
                        ),
                        default=0,
                    )
                    return int(healed_hp > response_ceiling)
            return 0

        def root_selection_key(candidate) -> tuple:
            value_key = _value_selection_key(
                candidate[0],
                category=candidate[4],
                prefer_unresolved=True,
                objective=state.battle_objective if state.capture_target else None,
            )
            # A survivable escape is the avoidance objective itself. Compare
            # direct Run Away and switch-then-Run lines by their modeled
            # success probability before ordinary battle utility. An unsafe
            # escape remains in the ordinary comparison and may lose to an
            # attack or switch that protects the party.
            escape_probability = avoidance_escape_probability(candidate)
            avoidance_run_priority = int(escape_probability is not None)
            avoidance_escape_rank = escape_probability if escape_probability is not None else 0.0
            if not state.campaign_owned:
                # Preserve the established offline/legacy tie behavior. The
                # campaign path uses the state-first comparison below.
                return (avoidance_run_priority, avoidance_escape_rank) + value_key[:1] + (
                    root_immediate_survival(candidate),
                ) + value_key[1:]
            # Immediate survival breaks close utility ties after the avoidance
            # objective and successor-state ranking have been considered.
            return (avoidance_run_priority, avoidance_escape_rank) + value_key[:2] + (
                root_immediate_survival(candidate),
            ) + value_key[2:]

        selected_candidate = max(
            candidates,
            key=root_selection_key,
        )
        value, action_kind, action_index, action_name, selected_category, selected_budget_exhaustion = selected_candidate
        selected_move = next(
            (move for move in state.player_moves if move.index == action_index),
            None,
        ) if action_kind == "move" else None
        immediate_ko_without_response = bool(
            selected_move is not None
            and getattr(selected_move, "turns_required", 1) == 1
            and _status_action_probability(state.player_status) >= 1.0
            and value.line[:2] == (selected_move.name, "KO")
        )
        if selected_budget_exhaustion is not None and action_kind == "move" and state.battle_objective in {"defeat", "training"}:
            selected_move = next(
                (move for move in state.player_moves if move.index == action_index),
                None,
            )
            damaging_moves = tuple(move for move in state.player_moves if move.damage_max > 0)
            if selected_move is not None and selected_move.damage_max <= 0 and damaging_moves:
                # A cutoff can leave a setup move with a better partial
                # heuristic than the damaging roots that were not fully
                # expanded. For defeat/training, preserve immediate battle
                # progress at this boundary. Capture has its own utility
                # model and deliberately keeps setup/weakening candidates.
                fallback_move = max(
                    damaging_moves,
                    key=lambda move: (
                        move.damage_min * _clamp_probability(move.accuracy),
                        move.damage_max,
                        move.priority,
                        -move.index,
                    ),
                )
                fallback = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate[1] == "move" and candidate[2] == fallback_move.index
                    ),
                    None,
                )
                if fallback is not None:
                    value, action_kind, action_index, action_name, selected_category, selected_budget_exhaustion = fallback
        selected_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate[1] == action_kind and candidate[2] == action_index
            ),
            None,
        )
        if selected_candidate is not None:
            selected_category = selected_candidate[4]
            selected_budget_exhaustion = selected_candidate[5]
        if (
            selected_budget_exhaustion is None
            and not state.response_facts_complete
            and selected_category is SearchResultCategory.TERMINAL_COMPLETE
            and not immediate_ko_without_response
            and "CAPTURED" not in value.line
        ):
            selected_category = SearchResultCategory.HORIZON_PROJECTION
        category = selected_category
        if (
            any_root_budget_exhausted
            and category is SearchResultCategory.HORIZON_PROJECTION
        ):
            # A horizon retained from an earlier root pass is still incomplete
            # once the shared node budget is consumed elsewhere. Preserve a
            # proven terminal success, but do not report a full line when the
            # final comparison could not finish.
            category = SearchResultCategory.INCOMPLETE
            selected_budget_exhaustion = selected_budget_exhaustion or first_budget_exhaustion or "nodes"
        reason = (
            "bounded expectiminimax found a robust line"
            if value.win >= self.config.minimum_safe_win_probability
            and value.loss <= self.config.maximum_safe_loss_probability
            and value.party_wipe <= self.config.maximum_safe_catastrophic_probability
            and category is SearchResultCategory.TERMINAL_COMPLETE
            else "best available line remains exposed to modeled uncertainty"
        )
        # The bounded search completed even when it reached its depth
        # horizon. A node/time cutoff remains incomplete and can never be
        # treated as a safe projection.
        complete = category is not SearchResultCategory.INCOMPLETE
        if category is SearchResultCategory.INCOMPLETE:
            if selected_budget_exhaustion == "time":
                reason = (
                    f"search time budget ({self.config.max_duration_seconds:g} seconds) expired; "
                    "the returned line is incomplete and cannot establish safety"
                )
            else:
                reason = (
                    f"search node budget ({self.config.max_nodes}) was reached; "
                    "the returned line is incomplete and cannot establish safety"
                )
        elif category is SearchResultCategory.HORIZON_PROJECTION:
            reason = (
                f"search reached the configured depth horizon ({self._turn_limit} turns); "
                if state.response_facts_complete
                else "response envelope is incomplete; "
            ) + (
                "the returned projection made progress and survives the worst known immediate response"
                if state.response_facts_complete and value.projection_safe
                else "the returned line is a projection and cannot establish safety"
            )
        root_evaluations = tuple(
            SearchRootEvaluation(
                action_kind=candidate_kind,
                action_index=candidate_index,
                action_name=candidate_name,
                expected_score=candidate_value.score,
                win_probability=candidate_value.win,
                loss_probability=candidate_value.loss,
                catastrophic_probability=candidate_value.party_wipe,
                category=candidate_category,
                principal_variation=candidate_value.line,
                active_faint_probability=candidate_value.active_faint,
                party_wipe_probability=candidate_value.party_wipe,
                run_loss_probability=candidate_value.run_loss,
                immediate_projection=self._immediate_projection(
                    state,
                    candidate_kind,
                    candidate_index,
                    candidate_name,
                ),
                target_party_index=candidate_value.target_party_index,
                objective_progress=candidate_value.objective_progress,
                projection_safe=candidate_value.projection_safe,
                budget_exhaustion=_candidate_budget_exhaustion,
                objective_failed=candidate_value.objective_failed,
            )
            for (
                candidate_value,
                candidate_kind,
                candidate_index,
                candidate_name,
                candidate_category,
                _candidate_budget_exhaustion,
            ) in candidates
        )
        # The selected aggregate and its root evaluation are built through
        # separate bounded-search paths. A time cutoff can leave the
        # root-local healing target intact while the aggregate _Value loses
        # that field during selection. Reconcile from the selected root so a
        # reserve heal cannot be translated as an active-slot no-op.
        selected_root_evaluation = next(
            (
                candidate
                for candidate in root_evaluations
                if candidate.action_kind == action_kind
                and candidate.action_index == action_index
                and candidate.target_party_index is not None
            ),
            None,
        )
        selected_target_party_index = (
            selected_root_evaluation.target_party_index
            if selected_root_evaluation is not None
            else value.target_party_index
        )
        immediate_projection = self._immediate_projection(
            state,
            action_kind,
            action_index,
            action_name,
        )
        root_is_objective_progress = True
        if action_kind == "move" and state.battle_objective in {"defeat", "training"}:
            selected_root_move = next(
                (move for move in state.player_moves if move.index == action_index),
                None,
            )
            root_is_objective_progress = bool(
                selected_root_move is not None and selected_root_move.damage_max > 0
            )
        return SearchResult(
            action_index=action_index,
            action_name=action_name,
            expected_score=value.score,
            win_probability=value.win,
            loss_probability=value.loss,
            catastrophic_probability=value.party_wipe,
            principal_variation=value.line,
            expanded_nodes=self.expanded_nodes,
            complete=complete,
            reason=reason,
            budget_exhaustion=selected_budget_exhaustion,
            minimum_safe_win_probability=self.config.minimum_safe_win_probability,
            maximum_safe_loss_probability=self.config.maximum_safe_loss_probability,
            maximum_safe_catastrophic_probability=self.config.maximum_safe_catastrophic_probability,
            action_kind=action_kind,
            capture_probability=value.capture,
            encounter_loss_probability=value.encounter_loss,
            expected_experience=value.expected_experience,
            experience_utility=value.experience_utility,
            objective=state.battle_objective,
            maximum_safe_encounter_loss_probability=self.config.maximum_safe_encounter_loss_probability,
            category=category,
            root_evaluations=root_evaluations,
            active_faint_probability=value.active_faint,
            party_wipe_probability=value.party_wipe,
            run_loss_probability=value.run_loss,
            projection_safe=(
                value.projection_safe
                and state.response_facts_complete
                and value.active_faint <= self.config.maximum_safe_catastrophic_probability
                and value.party_wipe <= self.config.maximum_safe_catastrophic_probability
                and value.run_loss <= self.config.maximum_safe_loss_probability
                and value.encounter_loss <= self.config.maximum_safe_encounter_loss_probability
            ),
            immediate_projection=immediate_projection,
            target_party_index=selected_target_party_index,
            objective_failed=value.objective_failed,
        )

    def _immediate_projection(
        self,
        state: SearchState,
        action_kind: str,
        action_index: int | None,
        action_name: str | None,
    ) -> SearchImmediateProjection:
        """Expose first-turn RNG branches independently of deep-search status."""

        def outcome(
            phase: str,
            move: SearchMove,
            branch: _MoveOutcome,
            *,
            target: str,
            action_probability: float = 1.0,
            source_state: SearchState | None = None,
        ) -> SearchImmediateOutcome:
            source_state = source_state or state
            return SearchImmediateOutcome(
                phase,
                move.name,
                action_probability * branch.probability,
                branch.label,
                _resolved_move_damage(
                    source_state,
                    move,
                    branch,
                    attacker_is_player=phase == "player",
                ),
                target,
                branch.critical,
                branch.secondary_effect,
            )

        player_outcomes: list[SearchImmediateOutcome] = []
        opponent_responses: list[SearchImmediateOutcome] = []
        turn_order: list[str] = []

        selected_move = next(
            (move for move in state.player_moves if move.index == action_index),
            None,
        ) if action_kind == "move" else None
        if selected_move is not None:
            action_probability = _status_action_probability(state.player_status)
            if action_probability < 1.0:
                player_outcomes.append(
                    SearchImmediateOutcome(
                        "player",
                        selected_move.name,
                        1.0 - action_probability,
                        f"STATUS_SKIP:{str(state.player_status).upper()}",
                        target="opponent",
                    )
                )
            player_outcomes.extend(
                outcome(
                    "player",
                    selected_move,
                    branch,
                    target="opponent",
                    action_probability=action_probability,
                )
                for branch in _move_outcomes_for_state(state, selected_move, attacker_is_player=True)
            )
            response_moves = self._opponent_choices(state, selected_move)
            for response in response_moves:
                if response.causes_opponent_escape:
                    opponent_responses.append(
                        SearchImmediateOutcome(
                            "opponent",
                            response.name,
                            1.0,
                            "OPPONENT_FLED",
                            target="encounter",
                        )
                    )
                else:
                    response_probability = _status_action_probability(state.opponent_status)
                    if response_probability < 1.0:
                        opponent_responses.append(
                            SearchImmediateOutcome(
                                "opponent",
                                response.name,
                                1.0 - response_probability,
                                f"STATUS_SKIP:{str(state.opponent_status).upper()}",
                                target="player",
                            )
                        )
                    opponent_responses.extend(
                        outcome(
                            "opponent",
                            response,
                            branch,
                            target="player",
                            action_probability=response_probability,
                        )
                        for branch in _move_outcomes_for_state(state, response, attacker_is_player=False)
                    )
                order = _acts_first(True, selected_move, state, response, tie_probability=True)
                turn_order.append(
                    f"{response.name}:"
                    + ("player_first" if order is True else "opponent_first" if order is False else "speed_tie")
                )
        elif action_kind == "switch":
            replacement = next(
                (
                    candidate
                    for candidate in state.player_switches
                    if candidate.party_index == action_index
                    and candidate.hp > 0
                    and candidate.max_hp > 0
                    and candidate.moves
                ),
                None,
            )
            if replacement is not None:
                switched = self._switch_state(state, replacement)
                player_outcomes.append(
                    SearchImmediateOutcome(
                        "player",
                        f"Switch({replacement.name})",
                        1.0,
                        "SWITCH",
                        target="opponent",
                    )
                )
                for response in self._opponent_choices(switched, None):
                    if response.causes_opponent_escape:
                        opponent_responses.append(
                            SearchImmediateOutcome(
                                "opponent_after_switch",
                                response.name,
                                1.0,
                                "OPPONENT_FLED",
                                target="encounter",
                            )
                        )
                    else:
                        response_probability = _status_action_probability(switched.opponent_status)
                        if response_probability < 1.0:
                            opponent_responses.append(
                                SearchImmediateOutcome(
                                    "opponent_after_switch",
                                    response.name,
                                    1.0 - response_probability,
                                    f"STATUS_SKIP:{str(switched.opponent_status).upper()}",
                                    target="replacement",
                                )
                            )
                        opponent_responses.extend(
                            outcome(
                                "opponent_after_switch",
                                response,
                                branch,
                                target="replacement",
                                action_probability=response_probability,
                                source_state=switched,
                            )
                            for branch in _move_outcomes_for_state(switched, response, attacker_is_player=False)
                        )
                    turn_order.append(f"opponent_after_switch:{response.name}")
        elif action_kind == "run":
            escape_probability = _escape_probability(state)
            player_outcomes.extend(
                (
                    SearchImmediateOutcome("player", action_name or "Run Away", escape_probability, "ESCAPED", target="encounter"),
                    SearchImmediateOutcome("player", action_name or "Run Away", 1.0 - escape_probability, "FAILED", target="encounter"),
                )
            )
        elif action_name is not None:
            player_outcomes.append(
                SearchImmediateOutcome("player", action_name, 1.0, action_kind.upper(), target="opponent")
            )

        return SearchImmediateProjection(
            action_kind,
            action_index,
            action_name,
            tuple(player_outcomes),
            tuple(opponent_responses),
            tuple(turn_order),
        )

    def _within_budget(self) -> bool:
        """Reserve one search expansion while honoring both live budgets."""

        if self.expanded_nodes >= self.config.max_nodes:
            self._budget_exhausted = True
            self._budget_exhaustion = self._budget_exhaustion or "nodes"
            return False
        if self._root_node_limit is not None and self.expanded_nodes - self._root_nodes_start >= self._root_node_limit:
            self._budget_exhausted = True
            self._budget_exhaustion = self._budget_exhaustion or "nodes"
            return False
        if self._deadline is not None and monotonic() >= self._deadline:
            self._budget_exhausted = True
            self._budget_exhaustion = self._budget_exhaustion or "time"
            return False
        return True

    def _experience_utility(self, state: SearchState) -> float:
        values = dict(state.experience_values)
        participants = state.experience_participants
        if not participants and state.player_party_index is not None:
            participants = frozenset({state.player_party_index})
        if values and participants:
            return sum(float(values.get(index, 0.0)) for index in participants)
        return float(state.experience_reward)

    def _ordinary_switch_penalty(self, state: SearchState) -> float:
        """Return the cost of spending an ordinary battle turn switching."""

        if state.free_switch:
            return 0.0
        return max(0.0, float(self.config.ordinary_switch_penalty))

    def _charge_capture_move(self, value: _Value, state: SearchState, move: SearchMove) -> _Value:
        """Price one capture setup turn without vetoing the move.

        A move root remains eligible when it improves the modeled capture
        sequence, lowers incoming risk, or creates another material state
        advantage. The finite charge only prevents a no-benefit setup turn
        from beating an immediately legal ball because it was free in the
        horizon heuristic.
        """

        if not (state.capture_target and state.battle_objective == "capture"):
            return value
        penalty = max(0.0, float(self.config.capture_turn_penalty))
        if penalty <= 0.0:
            return value
        return replace(value, score=value.score - penalty)

    def _capture_probability(self, state: SearchState, item: SearchItem) -> float:
        evaluator = self.config.capture_probability
        if callable(evaluator):
            try:
                return _clamp_probability(evaluator(state, item))
            except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
                pass
        return _clamp_probability(item.capture_probability)

    def _faint_value(self, state: SearchState, line: tuple[str, ...]) -> _Value:
        """Value a faint with capture-specific party and run-loss costs."""

        value = _faint_value(state, line)
        if not (state.capture_target and state.battle_objective == "capture"):
            return value
        if value.party_wipe > 0.0 or value.run_loss > 0.0:
            penalty = max(
                self.config.capture_party_wipe_penalty,
                self.config.capture_run_loss_penalty,
            )
        else:
            penalty = self.config.capture_active_faint_penalty
        return replace(value, score=-penalty)

    def _capture_sequence_probability(self, state: SearchState) -> float:
        """Estimate capture probability from the current state.

        Horizon projections need a bounded capture value even when the full
        tree cannot reach the next ball throw. The old geometric estimate
        treated every remaining ball as a free attempt and only capped the
        count with a single response ceiling. That made repeated stat setup
        look better than throwing immediately: lowering Attack could increase
        the estimated number of future throws without charging the response
        turn that produced the improvement.

        This projection models the finite sequence directly. Each branch
        consumes one ball, then applies the opponent's selected response after
        a failed throw. Damage, criticals, stat effects, escape, active faint,
        and depleted balls therefore all affect the result. It is deliberately
        limited to ball sequences; ordinary move search remains responsible
        for comparing the setup action itself with the ball root.
        """

        if not (state.capture_target and state.battle_objective == "capture"):
            return 0.0
        capture_progress = (
            state.capture_attempted
            or state.opponent_hp < state.opponent_max_hp
            or not _status_is_healthy(state.opponent_status)
        )
        if not capture_progress:
            return 0.0
        if self._capture_balls_remaining(state) <= 0:
            return 0.0

        memo: dict[SearchState, float] = {}

        def weighted(values: tuple[tuple[float, float], ...]) -> float:
            total = sum(probability for probability, _ in values if probability > 0)
            if total <= 0:
                return 0.0
            return _clamp_probability(
                sum(probability * value for probability, value in values if probability > 0) / total
            )

        def after_failed_throw(used: SearchState) -> float:
            responses = self._opponent_choices(used, None)
            if not responses:
                return sequence(used)

            response_values = []
            for response in responses:
                if response.causes_opponent_escape:
                    response_values.append(0.0)
                    continue
                action_probability = _status_action_probability(used.opponent_status)
                branches = []
                if action_probability < 1.0:
                    branches.append(
                        (1.0 - action_probability, sequence(_cancel_bide(used, player=False)))
                    )
                for outcome in _move_outcomes_for_state(used, response, attacker_is_player=False):
                    after_damage = _apply_move_damage(
                        used, response, outcome, attacker_is_player=False
                    )
                    if after_damage.player_hp <= 0:
                        branches.append((action_probability * outcome.probability, 0.0))
                        continue
                    next_state = _apply_move_effect(
                        after_damage,
                        response,
                        outcome,
                        attacker_is_player=False,
                        target_is_player=True,
                    )
                    branches.append((action_probability * outcome.probability, sequence(next_state)))
                response_values.append(weighted(tuple(branches)))

            if self.config.opponent_policy is OpponentPolicy.CONSERVATIVE:
                return min(response_values, default=0.0)
            return response_values[0] if response_values else 0.0

        def sequence(current: SearchState) -> float:
            cached = memo.get(current)
            if cached is not None:
                return cached
            balls = tuple(
                item
                for item in current.player_items
                if item.is_capture_ball and item.quantity > 0
            )
            if not balls or current.player_hp <= 0:
                memo[current] = 0.0
                return 0.0

            choices = []
            for item in balls:
                remaining = self._remaining_items(current, item)
                used = replace(current, player_items=remaining, capture_attempted=True)
                probability = self._capture_probability(current, item)
                failure_value = after_failed_throw(used) if probability < 1.0 else 0.0
                choices.append(
                    probability + (1.0 - probability) * failure_value
                )
            value = _clamp_probability(max(choices, default=0.0))
            memo[current] = value
            return value

        return sequence(state)

    @staticmethod
    def _remaining_items(state: SearchState, item: SearchItem) -> tuple[SearchItem, ...]:
        """Consume one copy while preserving every normalized item fact."""

        return tuple(
            replace(candidate, quantity=candidate.quantity - 1)
            if candidate.item_index == item.item_index
            else candidate
            for candidate in state.player_items
            if candidate.item_index != item.item_index or candidate.quantity > 1
        )

    def _item_cost(self, state: SearchState, item: SearchItem) -> float:
        """Return the finite-resource and overheal cost of one item action."""

        quantity = max(1, int(item.quantity))
        scarcity = self.config.item_scarcity_penalty / quantity
        if item.is_capture_ball:
            # Capture utility already prices the success/failure probability
            # and the remaining-ball sequence. Healing-resource penalties must
            # not distort the separate encounter objective.
            return 0.0

        missing_hp = max(0, state.player_max_hp - state.player_hp)
        nominal_healing = state.player_max_hp if item.full_heal else max(0, item.heal_amount)
        effective_healing = min(nominal_healing, missing_hp)
        wasted_healing = max(0, nominal_healing - effective_healing)
        return (
            self.config.item_use_penalty
            + scarcity
            + self.config.item_waste_penalty * wasted_healing
        )

    @staticmethod
    def _response_ceiling(state: SearchState) -> int:
        """Return the largest known one-turn response, including crits."""

        return max(
            max(
                (
                    state.opponent_bide_damage * 2
                    if _is_bide_move(move) and state.opponent_bide_turns == 1
                    else 0
                    if _is_bide_move(move)
                    else move.critical_damage_max
                    if move.critical_damage_max is not None
                    else move.damage_max
                    for move in state.opponent_moves
                ),
                default=0,
            ),
            state.unknown_response_ceiling,
        )

    @classmethod
    def _healing_item_is_materially_needed(cls, state: SearchState, item: SearchItem) -> bool:
        """Reject battle healing that cannot change the immediate risk state."""

        if item.is_capture_ball:
            return False
        if _item_cures_status(item, state.player_status):
            return True
        heals_hp = item.full_heal or item.heal_amount > 0
        response_ceiling = cls._response_ceiling(state)
        # A healing turn is justified when the active Pokémon is within the
        # known one-turn faint envelope.  Spending a limited item while the
        # active lead already survives that response only buys overheal.
        healed_hp = state.player_max_hp if item.full_heal else min(
            state.player_max_hp,
            state.player_hp + max(0, item.heal_amount),
        )
        if (
            heals_hp
            and not (
                state.battle_objective in {"avoid", "transit"}
                and state.healing_streak > 0
            )
            and response_ceiling > 0
            and state.player_hp <= response_ceiling
            and healed_hp > response_ceiling
        ):
            return True
        # The executor may target an injured benched member while the active
        # Pokémon safely absorbs the response. Keep that valid item boundary
        # available; the adapter reconciles the concrete target before input.
        return (
            any(
                switch.hp > 0
                and switch.max_hp > 0
                and (
                    (heals_hp and switch.hp < switch.max_hp)
                    or _item_cures_status(item, switch.status)
                )
                for switch in state.player_switches
            )
        )

    def _charge_item(self, value: _Value, state: SearchState, item: SearchItem) -> _Value:
        """Apply item cost without changing terminal risk priorities."""

        return replace(value, score=value.score - self._item_cost(state, item))

    def _project_opponent_response(
        self,
        state: SearchState,
        *,
        line: tuple[str, ...],
        continue_player: bool = False,
    ) -> _Value:
        """Project exactly one known opponent response without deep search.

        This is used only after a root budget cutoff.  It gives an incomplete
        root a real post-action state to compare, instead of scoring the
        untouched pre-action state and accidentally making every speculative
        switch or item look equivalent.
        """

        responses = self._opponent_choices(state, None)
        if not responses:
            return (
                self._project_best_followup(state, line=line)
                if continue_player
                else self._horizon_value(state, line=line)
            )

        def continuation(next_state: SearchState, next_line: tuple[str, ...]) -> _Value:
            return (
                self._project_best_followup(next_state, line=next_line)
                if continue_player
                else self._horizon_value(next_state, line=next_line)
            )

        choices = []
        for response in responses:
            if response.causes_opponent_escape:
                choices.append(self._opponent_escape_value(state, response))
                continue
            action_probability = _status_action_probability(state.opponent_status)
            branches = []
            if action_probability < 1.0:
                branches.append(
                    (
                        1.0 - action_probability,
                        continuation(
                            _cancel_bide(state, player=False),
                            line + (response.name, "STATUS_SKIP"),
                        ),
                    )
                )
            for outcome in _move_outcomes_for_state(state, response, attacker_is_player=False):
                after_damage = _apply_move_damage(
                    state, response, outcome, attacker_is_player=False
                )
                if after_damage.player_hp <= 0:
                    value = self._faint_value(state, line + (response.name, "FAINT"))
                else:
                    next_state = _apply_move_effect(
                        after_damage,
                        response,
                        outcome,
                        attacker_is_player=False,
                        target_is_player=True,
                    )
                    value = continuation(next_state, line + (response.name,))
                branches.append((action_probability * outcome.probability, value))
            choices.append(self._weighted(branches))
        return self._choose_opponent_value(choices)

    def _project_best_followup(self, state: SearchState, *, line: tuple[str, ...]) -> _Value:
        """Project the best player action after a bounded switch response.

        Root timeouts used to score a switch after only the opponent response,
        then compare the replacement's raw damage ceiling with the current
        lead.  That made a safe but ineffective replacement look equivalent to
        a lead that had an established KO continuation.  Give the bounded
        projection one concrete player action so its score reflects the state
        the switch actually enables.
        """

        moves = tuple(
            move
            for move in _objective_move_candidates(state)
            if move.damage_max > 0 or move.stat_delta != 0 or move.secondary_effect
        )
        if not moves:
            return self._horizon_value(state, line=line)

        candidates = []
        for move in moves:
            value = self._project_move_root(state, move)
            expected_progress = 0.0
            if state.opponent_max_hp > 0 and move.damage_max > 0:
                progress_baseline = state.root_opponent_hp or state.opponent_hp
                expected_progress = min(
                    1.0,
                    max(0.0, move.damage_max)
                    * _clamp_probability(move.accuracy)
                    / max(1, progress_baseline),
                )
            value = replace(
                value,
                line=line + value.line,
                objective_progress=max(value.objective_progress, expected_progress),
            )
            candidates.append(value)

        return max(
            candidates,
            key=lambda value: _value_selection_key(
                value,
                category=value.category,
                prefer_unresolved=True,
                objective=state.battle_objective if state.capture_target else None,
            ),
        )

    def _project_player_outcome(
        self,
        state: SearchState,
        move: SearchMove,
        outcome: _MoveOutcome,
        *,
        line: tuple[str, ...],
    ) -> _Value:
        next_state = _apply_move_effect(
            _apply_move_damage(state, move, outcome, attacker_is_player=True),
            move,
            outcome,
            attacker_is_player=True,
            target_is_player=False,
        )
        if next_state.opponent_hp <= 0:
            return self._opponent_defeated_value(next_state, line + (move.name, "KO"))
        return self._horizon_value(next_state, line=line + (move.name,))

    def _project_move_root(self, state: SearchState, move: SearchMove) -> _Value:
        """Evaluate one player action and one known response at the root."""

        state = _consume_player_move_pp(state, move)
        responses = self._opponent_choices(state, move)
        if not responses:
            value = self._weighted(
                [
                    (
                        outcome.probability,
                        self._project_player_outcome(state, move, outcome, line=()),
                    )
                    for outcome in _move_outcomes_for_state(state, move, attacker_is_player=True)
                ]
            )
            return self._charge_capture_move(value, state, move)

        choices = []
        for response in responses:
            if response.causes_opponent_escape:
                choices.append(self._opponent_escape_value(state, response))
                continue
            order = _acts_first(True, move, state, response, tie_probability=True)

            def player_first() -> _Value:
                branches = []
                for player_outcome in _move_outcomes_for_state(state, move, attacker_is_player=True):
                    after_player = _apply_move_effect(
                        _apply_move_damage(
                            state, move, player_outcome, attacker_is_player=True
                        ),
                        move,
                        player_outcome,
                        attacker_is_player=True,
                        target_is_player=False,
                    )
                    if after_player.opponent_hp <= 0:
                        value = self._opponent_defeated_value(
                            after_player,
                            (move.name, "KO"),
                        )
                    else:
                        value = self._project_opponent_response(
                            after_player,
                            line=(move.name, response.name),
                        )
                    branches.append((player_outcome.probability, value))
                return self._weighted(branches)

            def opponent_first() -> _Value:
                branches = []
                action_probability = _status_action_probability(state.opponent_status)
                if action_probability < 1.0:
                    cancelled_response_state = _cancel_bide(state, player=False)
                    branches.append(
                        (
                            1.0 - action_probability,
                            self._weighted(
                                [
                                    (
                                        outcome.probability,
                                        self._project_player_outcome(
                                            cancelled_response_state,
                                            move,
                                            outcome,
                                            line=(response.name, "STATUS_SKIP"),
                                        ),
                                    )
                                    for outcome in _move_outcomes_for_state(
                                        cancelled_response_state,
                                        move,
                                        attacker_is_player=True,
                                    )
                                ]
                            ),
                        )
                    )
                for response_outcome in _move_outcomes_for_state(state, response, attacker_is_player=False):
                    after_damage = _apply_move_damage(
                        state, response, response_outcome, attacker_is_player=False
                    )
                    if after_damage.player_hp <= 0:
                        branches.append(
                            (
                                action_probability * response_outcome.probability,
                                self._faint_value(state, (response.name, "FAINT")),
                            )
                        )
                        continue
                    after_response = _apply_move_effect(
                        after_damage,
                        response,
                        response_outcome,
                        attacker_is_player=False,
                        target_is_player=True,
                    )
                    branches.append(
                        (
                            action_probability * response_outcome.probability,
                            self._weighted(
                                [
                                    (
                                        outcome.probability,
                                        self._project_player_outcome(
                                            after_response,
                                            move,
                                            outcome,
                                            line=(response.name,),
                                        ),
                                    )
                                    for outcome in _move_outcomes_for_state(after_response, move, attacker_is_player=True)
                                ]
                            ),
                        )
                    )
                return self._weighted(branches)

            if order is True:
                choices.append(player_first())
            elif order is False:
                choices.append(opponent_first())
            else:
                choices.append(
                    self._weighted(
                        [(0.5, player_first()), (0.5, opponent_first())]
                    )
                )
        return self._charge_capture_move(self._choose_opponent_value(choices), state, move)

    def _project_switch_root(self, state: SearchState, replacement: SearchSwitch) -> _Value:
        switched = self._switch_state(state, replacement)
        value = self._project_opponent_response(
            switched,
            line=(f"Switch({replacement.name})",),
            continue_player=True,
        )
        value = replace(
            value,
            projection_safe=(
                value.objective_progress > 0.0
                and value.loss <= 0.0
                and value.party_wipe <= 0.0
                and value.run_loss <= 0.0
            ),
        )
        return self._charge_switch(value, state)

    def _healing_item_state(
        self,
        state: SearchState,
        item: SearchItem,
    ) -> tuple[SearchState, int | None, bool] | None:
        """Return the post-item state, target slot, and active necessity."""

        cured_status = _item_cures_status(item, state.player_status)
        heals_hp = item.full_heal or item.heal_amount > 0
        response_ceiling = self._response_ceiling(state)
        healed_hp = state.player_max_hp if item.full_heal else min(
            state.player_max_hp,
            state.player_hp + max(0, item.heal_amount),
        )
        active_needs_item = (
            heals_hp
            and not (
                state.battle_objective in {"avoid", "transit"}
                and state.healing_streak > 0
            )
            and response_ceiling > 0
            and state.player_hp <= response_ceiling
            and healed_hp > response_ceiling
        )
        reserve_target = None
        if not active_needs_item and not cured_status:
            reserve_candidates = []
            for switch in state.player_switches:
                if switch.hp <= 0 or switch.max_hp <= 0:
                    continue
                if switch.hp >= switch.max_hp and not _item_cures_status(item, switch.status):
                    continue
                status_progress = _item_cures_status(item, switch.status)
                healed_reserve_hp = (
                    switch.max_hp
                    if item.full_heal
                    else min(switch.max_hp, switch.hp + max(0, item.heal_amount))
                )
                hp_progress = heals_hp and healed_reserve_hp > switch.hp
                if not hp_progress and not status_progress:
                    continue
                missing_hp = max(0, switch.max_hp - switch.hp)
                waste = missing_hp if item.full_heal else max(0, item.heal_amount - missing_hp)
                reserve_candidates.append(
                    (
                        int(status_progress),
                        int(hp_progress),
                        -waste,
                        -item.heal_amount,
                        -switch.party_index,
                        switch,
                        healed_reserve_hp,
                        status_progress,
                    )
                )
            if reserve_candidates:
                reserve_target = max(reserve_candidates, key=lambda candidate: candidate[:5])

        if not active_needs_item and not cured_status and reserve_target is None:
            return None
        remaining = self._remaining_items(state, item)
        if reserve_target is not None:
            target_switch = reserve_target[5]
            used_switches = tuple(
                replace(
                    switch,
                    hp=reserve_target[6],
                    status=None if reserve_target[7] else switch.status,
                )
                if switch.party_index == target_switch.party_index
                else switch
                for switch in state.player_switches
            )
            return (
                replace(
                    state,
                    player_switches=used_switches,
                    player_items=remaining,
                    healing_streak=state.healing_streak + 1,
                ),
                target_switch.party_index,
                False,
            )
        return (
            replace(
                state,
                player_hp=healed_hp,
                player_status=None if cured_status else state.player_status,
                player_items=remaining,
                healing_streak=state.healing_streak + 1,
            ),
            state.player_party_index,
            True,
        )

    def _project_item_root(self, state: SearchState, item: SearchItem) -> _Value:
        if item.is_capture_ball:
            remaining = self._remaining_items(state, item)
            used = self._objective_after_capture_resources(
                replace(state, player_items=remaining, capture_attempted=True)
            )
            probability = self._capture_probability(state, item)
            branches = [(probability, self._capture_success_value(used, (item.name,)))]
            if probability < 1.0:
                branches.append(
                    (
                        1.0 - probability,
                        self._project_opponent_response(used, line=(item.name,)),
                    )
                )
            return self._charge_item(self._weighted(branches), state, item)
        projected = self._healing_item_state(state, item)
        if projected is None:
            return _Value(-100000.0, 0.0, 1.0, (item.name, "NO_EFFECT"))
        used, target_party_index, active_needs_item = projected
        value = self._project_opponent_response(used, line=(item.name,))
        value = replace(
            value,
            projection_safe=(
                active_needs_item
                and value.loss <= 0.0
                and value.party_wipe <= 0.0
                and value.run_loss <= 0.0
            ),
            target_party_index=target_party_index,
        )
        return self._charge_item(value, state, item)

    def _root_projection_value(
        self,
        state: SearchState,
        action_kind: str,
        action,
        action_name: str,
    ) -> _Value:
        if action_kind == "move":
            return self._project_move_root(state, action)
        if action_kind == "switch":
            return self._project_switch_root(state, action)
        if action_kind == "item":
            return self._project_item_root(state, action)
        if action_kind == "run":
            probability = _escape_probability(state)
            if probability >= 1.0:
                return self._run_success_value(state)
            failed_state = _failed_escape_state(state)
            return self._weighted(
                [
                    (probability, self._run_success_value(state)),
                    (
                        1.0 - probability,
                        self._project_opponent_response(
                            failed_state,
                            line=(action_name, "FAILED"),
                        ),
                    ),
                ]
            )
        return self._horizon_value(state, line=(action_name,))

    def _opponent_defeated_value(self, state: SearchState, line: tuple[str, ...]) -> _Value:
        """Return the correct terminal value for a defeated opponent."""

        experience_utility = self._experience_utility(state)
        if state.battle_objective == "capture" and state.capture_target:
            return _Value(
                -self.config.capture_encounter_loss_penalty,
                0.0,
                0.0,
                line + ("ENCOUNTER_LOST",),
                encounter_loss=1.0,
                expected_experience=0.0,
                experience_utility=0.0,
                objective_failed=True,
            )
        if state.capture_failed and state.capture_target:
            return _Value(
                -self.config.capture_encounter_loss_penalty
                + self.config.experience_weight * experience_utility,
                0.0,
                0.0,
                line + ("ENCOUNTER_LOST",),
                encounter_loss=1.0,
                expected_experience=float(state.experience_reward),
                experience_utility=experience_utility,
                objective_failed=True,
            )
        if state.battle_objective in {"avoid", "transit"}:
            # Avoidance is a successful campaign outcome.  Defeating an
            # incidental wild Pokémon remains legal, but it spends HP, PP,
            # and turns without advancing the mounted objective, so a safe
            # escape must outrank an unnecessary KO.
            player_ratio = state.player_hp / max(1, state.player_max_hp)
            return _Value(
                2.0 + 2.0 * player_ratio,
                1.0,
                0.0,
                line + ("WILD_BATTLE_AVOIDED",),
                expected_experience=0.0,
                experience_utility=0.0,
            )
        return _Value(
            100000.0
            + _success_state_utility(state)
            + self.config.experience_weight * experience_utility,
            1.0,
            0.0,
            line,
            expected_experience=float(state.experience_reward),
            experience_utility=experience_utility,
            objective_progress=1.0,
        )

    @staticmethod
    def _capture_balls_remaining(state: SearchState) -> int:
        return sum(
            max(0, int(item.quantity))
            for item in state.player_items
            if item.is_capture_ball
        )

    @classmethod
    def _objective_after_capture_resources(cls, state: SearchState) -> SearchState:
        """Move a depleted capture encounter into ordinary battle policy."""

        if (
            state.capture_target
            and state.battle_objective == "capture"
            and cls._capture_balls_remaining(state) <= 0
        ):
            return replace(state, battle_objective="defeat", capture_failed=True)
        return state

    def _run_success_value(self, state: SearchState) -> _Value:
        encounter_loss = (
            1.0
            if state.capture_target and (state.battle_objective == "capture" or state.capture_failed)
            else 0.0
        )
        if encounter_loss:
            return _Value(
                -self.config.capture_run_away_penalty,
                0.0,
                0.0,
                ("Run Away", "ESCAPED"),
                encounter_loss=encounter_loss,
                category=SearchResultCategory.TERMINAL_COMPLETE,
                objective_failed=True,
            )

        # A successful escape preserves the active party member.  Avoidance
        # has no EXP objective, so do not let a prepared-but-unrelated battle
        # make a successful escape look worse through the secondary EXP term.
        player_ratio = state.player_hp / max(1, state.player_max_hp)
        experience_utility = self._experience_utility(state)
        score = 1.0 + 10.0 * player_ratio
        if state.battle_objective not in {"avoid", "transit"}:
            score -= self.config.experience_weight * experience_utility
        if state.battle_objective in {"avoid", "transit"}:
            score += 5.0
        return _Value(
            score,
            1.0,
            0.0,
            ("Run Away", "ESCAPED"),
            expected_experience=0.0,
            experience_utility=0.0,
            category=SearchResultCategory.TERMINAL_COMPLETE,
        )

    def _player_run_value(self, state: SearchState, depth: int) -> _Value:
        key = ("run", state, depth, None)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if not self._within_budget():
            return self._horizon_value(state, line=("Run Away", "SEARCH_LIMIT"))
        self.expanded_nodes += 1
        probability = _escape_probability(state)
        if not _run_action_available(state):
            # This should be unreachable from a legal root, but keeping the
            # guard here makes recursive callers robust when an adapter
            # changes the battle state between observations.
            value = _Value(-100000.0, 0.0, 1.0, ("Run Away", "UNAVAILABLE"))
        elif probability >= 1:
            value = self._run_success_value(state)
        else:
            failure = self._prepend(
                ("Run Away", "FAILED"),
                self._opponent_response_value(_failed_escape_state(state), depth),
            )
            value = self._weighted(
                [
                    (probability, self._run_success_value(state)),
                    (1.0 - probability, failure),
                ]
            )
        self._memo[key] = value
        return value

    def _capture_success_value(self, state: SearchState, line: tuple[str, ...]) -> _Value:
        return _Value(
            self.config.capture_success_reward + _success_state_utility(state),
            1.0,
            0.0,
            line + ("CAPTURED",),
            capture=1.0,
            objective_progress=1.0,
        )

    @staticmethod
    def _switch_state(state: SearchState, replacement: SearchSwitch) -> SearchState:
        """Install a replacement and retain the old active member as a switch."""

        available = tuple(
            candidate for candidate in state.player_switches if candidate.party_index != replacement.party_index
        )
        if state.player_party_index is not None:
            available += (
                SearchSwitch(
                    state.player_party_index,
                    "previous active",
                    state.player_hp,
                    state.player_max_hp,
                    state.player_moves,
                    state.player_speed,
                    state.player_status,
                    escape_probability=state.escape_probability,
                    escape_guaranteed=state.escape_guaranteed,
                ),
            )
        return replace(
            state,
            player_hp=replacement.hp,
            player_max_hp=replacement.max_hp,
            player_moves=replacement.moves,
            player_speed=replacement.speed,
            player_status=replacement.status,
            escape_probability=(
                1.0
                if replacement.escape_guaranteed
                else replacement.escape_probability
                if replacement.escape_probability is not None
                else 0.0
            ),
            escape_guaranteed=replacement.escape_guaranteed,
            player_party_index=replacement.party_index,
            player_switches=available,
            player_stat_stages=(),
            player_bide_turns=0,
            player_bide_damage=0,
            player_bide_target=None,
            experience_participants=state.experience_participants
            | frozenset(
                index
                for index in (state.player_party_index, replacement.party_index)
                if index is not None
            ),
            switch_history=(
                state.switch_history
                + tuple(
                    index
                    for index in (state.player_party_index, replacement.party_index)
                    if index is not None
                )
            )[-4:],
            campaign_owned=state.campaign_owned,
            free_switch=False,
            switch_streak=state.switch_streak + 1,
            switch_cycle_pressure=state.switch_cycle_pressure + int(
                replacement.party_index in state.switch_history
            ),
            healing_streak=0,
        )

    def _player_switch_value(self, state: SearchState, replacement: SearchSwitch, depth: int) -> _Value:
        key = ("switch", state, depth, replacement)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if not self._within_budget():
            return self._horizon_value(state, line=(f"Switch({replacement.name})", "SEARCH_LIMIT"))
        self.expanded_nodes += 1
        switched = self._switch_state(state, replacement)
        value = self._prepend(
            (f"Switch({replacement.name})",),
            self._opponent_response_value(switched, depth),
        )
        value = self._charge_switch(value, state)
        self._memo[key] = value
        return value

    def _charge_switch(self, value: _Value, state: SearchState) -> _Value:
        """Apply the ordinary switch cost while preserving free replacements."""

        penalty = self._ordinary_switch_penalty(state)
        # The first switch can be the correct tactical response. Consecutive
        # switches are still legal, but their utility falls as the line starts
        # looking like a rotation that has not advanced the KO objective.
        penalty += self.config.switch_stall_penalty * max(0, state.switch_streak)
        penalty += self.config.switch_cycle_penalty * max(0, state.switch_cycle_pressure)
        if value.objective_progress <= 0.0 and not (
            state.battle_objective in {"avoid", "transit"}
            and value.win > 0.0
        ):
            penalty += self.config.switch_no_progress_penalty
        if penalty <= 0:
            return value
        return replace(value, score=value.score - penalty)

    def _player_item_value(self, state: SearchState, item: SearchItem, depth: int) -> _Value:
        # An item action is objective progress and therefore ends a
        # no-progress switch streak.
        state = replace(
            state,
            switch_history=(),
            free_switch=False,
            switch_streak=0,
            switch_cycle_pressure=0,
        )
        key = ("item", state, depth, item)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if item.is_capture_ball:
            remaining = self._remaining_items(state, item)
            used = replace(state, player_items=remaining, capture_attempted=True)
            used = self._objective_after_capture_resources(used)
            probability = self._capture_probability(state, item)
            branches = [(probability, self._capture_success_value(used, (item.name,)))]
            if probability < 1.0:
                if self._within_budget():
                    self.expanded_nodes += 1
                    failure = self._prepend((item.name,), self._opponent_response_value(used, depth))
                else:
                    failure = self._horizon_value(used, line=(item.name, "SEARCH_LIMIT"))
                branches.append((1.0 - probability, failure))
            value = self._charge_item(self._weighted(branches), state, item)
            self._memo[key] = value
            return value
        projected = self._healing_item_state(state, item)
        if projected is None:
            value = _Value(-100000.0, 0.0, 1.0, (item.name, "NO_EFFECT"))
            self._memo[key] = value
            return value
        used, target_party_index, _active_needs_item = projected
        if self._within_budget():
            self.expanded_nodes += 1
            value = self._prepend((item.name,), self._opponent_response_value(used, depth))
        else:
            # A live deadline must still compare the item's post-heal state
            # with already-evaluated roots. Scoring the pre-heal state made a
            # Potion look useless exactly when the search ran out of time
            # before reaching the item branch.
            value = self._horizon_value(used, line=(item.name, "SEARCH_LIMIT"))
        value = self._charge_item(value, state, item)
        value = replace(value, target_party_index=target_party_index)
        self._memo[key] = value
        return value

    def _player_move_value(self, state: SearchState, move: SearchMove, depth: int) -> _Value:
        # A move changes the opponent state (or at least consumes a turn), so
        # a later switch is no longer an immediate reversal of the same setup
        # action.
        state = replace(
            state,
            switch_history=(),
            free_switch=False,
            switch_streak=0,
            switch_cycle_pressure=0,
            healing_streak=0,
        )
        key = ("player", state, depth, move)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if not self._within_budget():
            value = self._horizon_value(state, line=(move.name, "SEARCH_LIMIT"))
        else:
            self.expanded_nodes += 1
            if depth >= self._turn_limit:
                value = self._horizon_value(state, line=(move.name,))
            else:
                action_probability = _status_action_probability(state.player_status)
                if action_probability < 1.0:
                    # A status outcome changes the future action distribution. Keep
                    # both branches in the tree so the planner never assumes that a
                    # secondary effect will or will not take hold on a specific turn.
                    value = self._weighted(
                        [
                            (
                                1.0 - action_probability,
                                self._prepend(
                                    (move.name, "STATUS_SKIP"),
                                    self._opponent_response_value(
                                        _cancel_bide(state, player=True), depth
                                    ),
                                ),
                            ),
                            (
                                action_probability,
                                self._player_move_value_unblocked(state, move, depth),
                            ),
                        ]
                    )
                else:
                    value = self._player_move_value_unblocked(state, move, depth)
        value = self._charge_capture_move(value, state, move)
        expected_progress = 0.0
        if state.opponent_max_hp > 0 and move.damage_max > 0:
            expected_progress = min(
                1.0,
                max(0.0, move.damage_max)
                * _clamp_probability(move.accuracy)
                / state.opponent_max_hp,
            )
        value = replace(
            value,
            objective_progress=max(value.objective_progress, expected_progress),
        )
        self._memo[key] = value
        return value

    def _player_move_value_unblocked(self, state: SearchState, move: SearchMove, depth: int) -> _Value:
        state = _consume_player_move_pp(state, move)
        responses = self._opponent_choices(state, move)
        if not responses:
            return self._player_first_without_response(state, move, depth)
        choices = [
            self._ordered_exchange_value(state, move, response, depth)
            for response in responses
        ]
        return self._choose_opponent_value(choices)

    def _ordered_exchange_value(
        self,
        state: SearchState,
        player_move: SearchMove,
        response: SearchMove,
        depth: int,
    ) -> _Value:
        """Evaluate one selected opponent response with Gen III ordering."""

        order = _acts_first(True, player_move, state, response, tie_probability=True)
        if order is True:
            return self._player_then_opponent_response(state, player_move, response, depth)
        if order is False:
            return self._opponent_then_player_response(state, player_move, response, depth)
        # Equal effective speed is a random Gen III tie, not a player win.
        return self._weighted(
            [
                (0.5, self._player_then_opponent_response(state, player_move, response, depth)),
                (0.5, self._opponent_then_player_response(state, player_move, response, depth)),
            ]
        )

    def _player_first_without_response(self, state: SearchState, move: SearchMove, depth: int) -> _Value:
        branches = []
        for outcome in _move_outcomes_for_state(state, move, attacker_is_player=True):
            next_state = _apply_move_effect(
                _apply_move_damage(state, move, outcome, attacker_is_player=True),
                move,
                outcome,
                attacker_is_player=True,
                target_is_player=False,
            )
            if next_state.opponent_hp <= 0:
                value = self._opponent_defeated_value(next_state, (move.name, "KO"))
            else:
                value = self._prepend((move.name,), self._next_turn(next_state, depth + 1))
            branches.append((outcome.probability, value))
        return self._weighted(branches)

    def _player_then_opponent_response(
        self,
        state: SearchState,
        player_move: SearchMove,
        response: SearchMove,
        depth: int,
    ) -> _Value:
        branches = []
        for outcome in _move_outcomes_for_state(state, player_move, attacker_is_player=True):
            next_state = _apply_move_effect(
                _apply_move_damage(state, player_move, outcome, attacker_is_player=True),
                player_move,
                outcome,
                attacker_is_player=True,
                target_is_player=False,
            )
            if next_state.opponent_hp <= 0:
                value = self._opponent_defeated_value(next_state, (player_move.name, "KO"))
            else:
                value = self._prepend(
                    (player_move.name,),
                    self._opponent_response_move_value(next_state, response, depth),
                )
            branches.append((outcome.probability, value))
        return self._weighted(branches)

    def _opponent_then_player(self, state: SearchState, player_move: SearchMove, depth: int) -> _Value:
        key = ("opponent_first", state, depth, player_move)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        choices = [
            self._opponent_then_player_response(state, player_move, response, depth)
            for response in self._opponent_choices(state, player_move)
        ]
        value = self._choose_opponent_value(choices)
        self._memo[key] = value
        return value

    def _opponent_then_player_response(
        self,
        state: SearchState,
        player_move: SearchMove,
        response_move: SearchMove,
        depth: int,
    ) -> _Value:
        if response_move.causes_opponent_escape:
            return self._opponent_escape_value(state, response_move)
        branches = []
        action_probability = _status_action_probability(state.opponent_status)
        if action_probability < 1.0:
            branches.append(
                (
                    1.0 - action_probability,
                    # The opponent's failed action does not consume the
                    # turn.  The already selected player action still gets
                    # to execute, just as it would when the opponent has no
                    # usable move.
                    self._prepend(
                        ("STATUS_SKIP",),
                        self._player_first_without_response(
                            _cancel_bide(state, player=False), player_move, depth
                        ),
                    ),
                )
            )
        for outcome in _move_outcomes_for_state(state, response_move, attacker_is_player=False):
            after_damage = _apply_move_damage(
                state, response_move, outcome, attacker_is_player=False
            )
            if after_damage.player_hp <= 0:
                branches.append(
                    (
                        action_probability * outcome.probability,
                        self._faint_value(state, (response_move.name, "FAINT")),
                    )
                )
                continue
            next_state = _apply_move_effect(
                after_damage,
                response_move,
                outcome,
                attacker_is_player=False,
                target_is_player=True,
            )
            for player_outcome in _move_outcomes_for_state(next_state, player_move, attacker_is_player=True):
                after_player = _apply_move_damage(
                    next_state, player_move, player_outcome, attacker_is_player=True
                )
                if after_player.opponent_hp <= 0:
                    value = self._opponent_defeated_value(
                        after_player,
                        (response_move.name, player_move.name, "KO"),
                    )
                else:
                    value = self._prepend(
                        (response_move.name, player_move.name),
                        self._next_turn(
                            _apply_move_effect(
                                after_player,
                                player_move,
                                player_outcome,
                                attacker_is_player=True,
                                target_is_player=False,
                            ),
                            depth + 1,
                        ),
                    )
                branches.append((action_probability * outcome.probability * player_outcome.probability, value))
        return self._weighted(branches)

    def _opponent_escape_value(self, state: SearchState, response: SearchMove) -> _Value:
        """Represent a modeled opponent escape without inventing a run node.

        Escape is an objective failure for capture and defeat objectives, but
        it is not a player faint or a run-ending loss.  Keeping it in the
        encounter-loss dimension lets campaign policy decide whether the
        opportunity was acceptable without corrupting party-risk metrics.
        """

        encounter_loss = 1.0 if state.capture_target else 0.0
        return _Value(
            (
                -self.config.capture_encounter_loss_penalty
                if encounter_loss
                else -1000.0
            ),
            0.0,
            0.0,
            (response.name, "OPPONENT_FLED"),
            encounter_loss=encounter_loss,
            category=SearchResultCategory.TERMINAL_COMPLETE,
            objective_failed=bool(encounter_loss),
        )

    def _opponent_response_move_value(
        self,
        state: SearchState,
        response: SearchMove,
        depth: int,
    ) -> _Value:
        if response.causes_opponent_escape:
            return self._opponent_escape_value(state, response)
        branches = []
        action_probability = _status_action_probability(state.opponent_status)
        if action_probability < 1.0:
            branches.append(
                (
                    1.0 - action_probability,
                    self._prepend(
                        ("STATUS_SKIP",),
                        self._next_turn(_cancel_bide(state, player=False), depth + 1),
                    ),
                )
            )
        for outcome in _move_outcomes_for_state(state, response, attacker_is_player=False):
            after_damage = _apply_move_damage(
                state, response, outcome, attacker_is_player=False
            )
            if after_damage.player_hp <= 0:
                value = self._faint_value(state, (response.name, "FAINT"))
            else:
                next_state = _apply_move_effect(
                    after_damage,
                    response,
                    outcome,
                    attacker_is_player=False,
                    target_is_player=True,
                )
                value = self._prepend((response.name,), self._next_turn(next_state, depth + 1))
            branches.append((action_probability * outcome.probability, value))
        return self._weighted(branches)

    def _opponent_response_value(self, state: SearchState, depth: int) -> _Value:
        key = ("opponent_response", state, depth, None)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if not state.opponent_moves:
            value = self._next_turn(state, depth + 1)
            self._memo[key] = value
            return value
        choices = []
        for response in self._opponent_choices(state, None):
            if response.causes_opponent_escape:
                choices.append(self._opponent_escape_value(state, response))
                continue
            branches = []
            action_probability = _status_action_probability(state.opponent_status)
            if action_probability < 1.0:
                branches.append(
                    (
                        1.0 - action_probability,
                        self._prepend(
                            ("STATUS_SKIP",),
                            self._next_turn(_cancel_bide(state, player=False), depth + 1),
                        ),
                    )
                )
            for outcome in _move_outcomes_for_state(state, response, attacker_is_player=False):
                after_damage = _apply_move_damage(
                    state, response, outcome, attacker_is_player=False
                )
                if after_damage.player_hp <= 0:
                    branches.append(
                        (
                            action_probability * outcome.probability,
                            self._faint_value(state, (response.name, "FAINT")),
                        )
                    )
                else:
                    next_state = _apply_move_effect(
                        after_damage,
                        response,
                        outcome,
                        attacker_is_player=False,
                        target_is_player=True,
                    )
                    branches.append(
                        (
                            action_probability * outcome.probability,
                            self._prepend(
                                (response.name,),
                                self._next_turn(
                                    next_state,
                                    depth + 1,
                                ),
                            ),
                        )
                    )
            choices.append(self._weighted(branches))
        value = self._choose_opponent_value(choices)
        self._memo[key] = value
        return value

    def _next_turn(self, state: SearchState, depth: int) -> _Value:
        state = _apply_residual_damage(state)
        key = ("next_turn", state, depth, None)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if state.opponent_hp <= 0:
            value = self._opponent_defeated_value(state, ("KO",))
            self._memo[key] = value
            return value
        if state.player_hp <= 0:
            value = self._faint_value(state, ("FAINT",))
            self._memo[key] = value
            return value
        if depth >= self._turn_limit:
            value = self._horizon_value(state, line=())
            self._memo[key] = value
            return value
        capture_eligible_moves = _objective_move_candidates(state)
        move_candidates = (
            _deduplicate_moves(capture_eligible_moves)
            if state.capture_target and state.battle_objective == "capture"
            else _prune_dominated_player_moves(capture_eligible_moves)
        )
        choices = [self._player_move_value(state, move, depth) for move in move_candidates]
        if state.player_bide_turns <= 0:
            choices.extend(
                self._player_switch_value(state, switch, depth)
                for switch in self._switch_candidates(state)
                if self._capture_switch_allowed(state)
            )
            choices.extend(
                self._player_item_value(state, item, depth)
                for item in state.player_items
                if item.quantity > 0
                and (
                    item.is_capture_ball
                    and state.capture_target
                    and state.battle_objective == "capture"
                    or self._healing_item_is_materially_needed(state, item)
                )
            )
            if _run_action_available(state):
                choices.append(self._player_run_value(state, depth))
        if not choices:
            value = _Value(-100000.0, 0.0, 1.0, ("NO_ACTION",))
        else:
            value = max(
                choices,
                key=lambda candidate: _value_selection_key(
                    candidate,
                    objective=state.battle_objective if state.capture_target else None,
                ),
            )
        self._memo[key] = value
        return value

    @staticmethod
    def _prepend(prefix: tuple[str, ...], value: _Value) -> _Value:
        return _Value(
            value.score,
            value.win,
            value.loss,
            prefix + value.line,
            value.capture,
            value.encounter_loss,
            value.expected_experience,
            value.experience_utility,
            value.category,
            value.active_faint,
            value.party_wipe,
            value.run_loss,
            value.projection_safe,
            value.target_party_index,
            value.objective_failed,
            value.objective_progress,
        )

    def _opponent_choices(self, state: SearchState, player_move: SearchMove | None) -> tuple[SearchMove, ...]:
        if not state.opponent_moves:
            return ()
        opponent_moves = _deduplicate_moves(state.opponent_moves)
        if state.opponent_bide_turns > 0:
            # Gen III locks the opponent to Bide until its storing/release
            # sequence is complete. This also prevents the conservative
            # response policy from treating the lock as a fresh free choice.
            return tuple(move for move in opponent_moves if _is_bide_move(move))
        if self.config.opponent_policy is OpponentPolicy.CONSERVATIVE:
            return opponent_moves
        if self.opponent_selector is not None:
            selected = self.opponent_selector(state, opponent_moves, player_move)
            if selected is not None:
                return (selected,)
        # This is a simple approximation of the known Gen III preference for
        # effective damaging moves. A later ROM-specific adapter can replace
        # it without changing the search state contract.
        damaging = tuple(move for move in opponent_moves if move.damage_max > 0)
        return (max(damaging or opponent_moves, key=lambda move: (move.damage_max, move.accuracy, -move.index)),)

    def _choose_opponent_value(self, choices: list[_Value]) -> _Value:
        if not choices:
            return _Value(0.0, 0.0, 0.0, ())
        if self.config.opponent_policy is OpponentPolicy.CONSERVATIVE:
            return min(choices, key=lambda value: value.score)
        return choices[0]

    @staticmethod
    def _weighted(branches: list[tuple[float, _Value]], *, minimize: bool = False) -> _Value:
        if not branches:
            return _Value(-100000.0, 0.0, 1.0, ("NO_BRANCH",))
        if minimize:
            # Opponent choice is a policy decision, not a random draw. The
            # conservative policy therefore chooses the least favorable
            # expected branch rather than averaging all legal moves.
            return min((value for _, value in branches), key=lambda value: value.score)
        total = sum(probability for probability, _ in branches)
        if total <= 0:
            return branches[0][1]
        score = sum(probability * value.score for probability, value in branches) / total
        win = sum(probability * value.win for probability, value in branches) / total
        loss = sum(probability * value.loss for probability, value in branches) / total
        capture = sum(probability * value.capture for probability, value in branches) / total
        encounter_loss = sum(probability * value.encounter_loss for probability, value in branches) / total
        expected_experience = sum(probability * value.expected_experience for probability, value in branches) / total
        experience_utility = sum(probability * value.experience_utility for probability, value in branches) / total
        active_faint = sum(probability * value.active_faint for probability, value in branches) / total
        party_wipe = sum(probability * value.party_wipe for probability, value in branches) / total
        run_loss = sum(probability * value.run_loss for probability, value in branches) / total
        objective_progress = sum(
            probability * value.objective_progress
            for probability, value in branches
        ) / total
        live_branches = tuple((probability, value) for probability, value in branches if probability > 0)
        line = max((value.line for _, value in live_branches), key=len, default=())
        categories = tuple(value.category for _, value in live_branches)
        category = (
            SearchResultCategory.INCOMPLETE
            if SearchResultCategory.INCOMPLETE in categories
            else SearchResultCategory.HORIZON_PROJECTION
            if SearchResultCategory.HORIZON_PROJECTION in categories
            else SearchResultCategory.TERMINAL_COMPLETE
        )
        projection_safe = all(
            value.category is SearchResultCategory.TERMINAL_COMPLETE or value.projection_safe
            for _, value in live_branches
        )
        objective_failed = bool(live_branches) and all(
            value.objective_failed for _, value in live_branches
        )
        return _Value(
            score,
            win,
            loss,
            line,
            capture,
            encounter_loss,
            expected_experience,
            experience_utility,
            category,
            active_faint,
            party_wipe,
            run_loss,
            projection_safe,
            None,
            objective_failed,
            objective_progress,
        )

    def _horizon_value(self, state: SearchState, *, line: tuple[str, ...]) -> _Value:
        player_ratio = state.player_hp / max(1, state.player_max_hp)
        opponent_ratio = state.opponent_hp / max(1, state.opponent_max_hp)
        progress_baseline = state.root_opponent_hp
        if progress_baseline is None:
            progress_baseline = state.opponent_hp
        objective_progress = min(
            1.0,
            max(0.0, progress_baseline - state.opponent_hp)
            / max(1, progress_baseline),
        )
        experience_utility = self._experience_utility(state)
        if state.battle_objective in {"avoid", "transit"}:
            # The objective is to end the encounter by escaping.  Damage dealt
            # before the horizon is irrelevant because it neither advances the
            # campaign nor grants EXP until the opponent is defeated.  In
            # particular, do not turn an already damaged opponent into a large
            # score for a nonterminal setup move such as Growl.
            score = 10 * player_ratio
            experience_utility = 0.0
        elif state.battle_objective == "capture" and state.capture_target:
            # Capture setup has no intrinsic value.  Only the modeled chance
            # of eventually securing the encounter earns the large objective
            # reward; the small survival term keeps otherwise equal safe
            # projections ordered without making a no-op stat move attractive.
            score = 10 * player_ratio
        else:
            # Progress matters at the horizon, but survival dominates it by design.
            progress_weight = 60 if state.battle_objective == "capture" else 100
            score = progress_weight * objective_progress + 10 * player_ratio
            score += self.config.experience_weight * experience_utility * objective_progress
        capture_projection = 0.0
        # A remaining-ball projection is valid after a ball has been thrown
        # or after the current line has made measurable weakening/status
        # progress. Before either boundary, preserving balls is not capture
        # progress and must not make a setup move such as Growl look
        # equivalent to throwing a Poké Ball at a safe target.
        capture_projection = self._capture_sequence_probability(state)
        if capture_projection:
            score += self.config.capture_success_reward * capture_projection
        response_ceiling = self._response_ceiling(state)
        # A projection is actionable only when it made measurable objective
        # progress and leaves enough HP for the next ordinary known response.
        # Critical-hit risk is retained separately in ``active_faint`` and
        # ``party_wipe`` so readiness policy can apply its probability limit
        # without turning every low-probability critical into a hard horizon
        # failure.
        projection_safe = (
            state.player_hp > response_ceiling
            and state.battle_objective not in {"avoid", "transit"}
            and state.opponent_hp < progress_baseline
            and str(state.player_status or "").upper() not in {"BURN", "POISON"}
        )
        return _Value(
            score,
            0.0,
            0.0,
            line + ("HORIZON",),
            capture=capture_projection,
            experience_utility=experience_utility,
            category=SearchResultCategory.HORIZON_PROJECTION,
            projection_safe=projection_safe,
            objective_failed=bool(state.capture_failed and state.capture_target),
            objective_progress=objective_progress,
        )


def search_planner_context(planner_context, *, config: SearchConfig | None = None) -> SearchResult | None:
    """Adapt normalized planner facts to expectiminimax.

    The search can evaluate an incomplete response envelope as a projection.
    It must still own the root action in that case; callers are not allowed to
    replace an incomplete search with a strategy-local move, capture, or item
    selector. Unknown opponent actions remain absent from the state and are
    reflected in the result category and safety classification.
    """
    required_fields = ("active_hp", "active_max_hp", "opponent_hp")
    if any(not hasattr(planner_context, field) for field in required_fields):
        return None
    player_moves = tuple(
        SearchMove(
            move.index,
            move.name,
            move.damage_min,
            move.damage_max,
            move.accuracy,
            move.priority,
            secondary_chance=getattr(move, "secondary_chance", 0.0),
            secondary_effect=getattr(move, "effect", None),
            stat_target=getattr(move, "stat_target", None),
            stat_delta=getattr(move, "stat_delta", 0),
            damage_stat=getattr(move, "damage_stat", None),
            causes_opponent_escape=False,
            pp=getattr(move, "pp", None),
        )
        for move in getattr(planner_context, "moves", ())
        if getattr(move, "mechanics_supported", True)
    )
    raw_opponent_moves = tuple(getattr(planner_context, "opponent_moves", ()))
    unsupported_opponent_moves = tuple(
        dict.fromkeys(
            (
                *getattr(planner_context, "unsupported_opponent_moves", ()),
                *(
                    getattr(move, "name", "unknown")
                    for move in raw_opponent_moves
                    if not getattr(move, "mechanics_supported", True)
                ),
            )
        )
    )
    opponent_moves = tuple(
        SearchMove(
            index,
            move.name,
            move.damage_min,
            move.damage_max,
            move.accuracy,
            move.priority,
            move.critical_damage_max,
            move.critical_hit_chance,
            getattr(move, "secondary_chance", 0.0),
            getattr(move, "effect", None),
            getattr(move, "stat_target", None),
            getattr(move, "stat_delta", 0),
            causes_opponent_escape=getattr(move, "causes_opponent_escape", False),
            damage_stat=getattr(move, "damage_stat", None),
        )
        for index, move in enumerate(raw_opponent_moves)
        if getattr(move, "mechanics_supported", True)
    )
    if int(getattr(planner_context, "opponent_bide_turns", 0) or 0) > 0 and not any(
        _is_bide_move(move) for move in opponent_moves
    ):
        # A Bide-locked battler can only continue Bide, whether or not the
        # move was ever revealed or enumerated.  Without it the search offers
        # the opponent no response and models the release as harmless.
        opponent_moves = (
            *opponent_moves,
            SearchMove(
                max((move.index for move in opponent_moves), default=-1) + 1,
                "Bide",
                0,
                0,
                secondary_effect="BIDE",
            ),
        )
    if not opponent_moves and getattr(planner_context, "opponent_damage_max", None) is not None:
        # Some offline and readiness callers provide only a bounded response
        # ceiling. Preserve that fact as a conservative synthetic response so
        # the unified search can evaluate healing and switch survival instead
        # of silently treating the opponent as unable to act.
        response_ceiling = max(0, int(getattr(planner_context, "opponent_damage_max", 0) or 0))
        critical_ceiling = getattr(planner_context, "opponent_critical_damage_max", None)
        opponent_moves = (
            SearchMove(
                0,
                "Known response ceiling",
                response_ceiling,
                response_ceiling,
                critical_damage_max=critical_ceiling,
            ),
        )
    player_switches = []
    for switch in getattr(planner_context, "switches", ()):
        switch_moves = tuple(
            SearchMove(
                move.index,
                move.name,
                move.damage_min,
                move.damage_max,
                move.accuracy,
                move.priority,
                secondary_chance=getattr(move, "secondary_chance", 0.0),
            secondary_effect=getattr(move, "effect", None),
            stat_target=getattr(move, "stat_target", None),
            stat_delta=getattr(move, "stat_delta", 0),
            damage_stat=getattr(move, "damage_stat", None),
            causes_opponent_escape=False,
            pp=getattr(move, "pp", None),
            )
            for move in getattr(switch, "moves", ())
            if getattr(move, "mechanics_supported", True)
        )
        if not switch_moves and getattr(switch, "offensive_damage_max", 0) > 0:
            # Readiness and fixture callers may provide only a matchup damage
            # ceiling. Keep that candidate visible to the unified search with
            # a conservative synthetic attack rather than silently dropping
            # the party member from switch evaluation.
            switch_moves = (
                SearchMove(
                    0,
                    "Known matchup attack",
                    int(getattr(switch, "offensive_damage_max", 0)),
                    int(getattr(switch, "offensive_damage_max", 0)),
                ),
            )
        if switch_moves or (
            getattr(switch, "hp", 0) > 0
            and getattr(switch, "max_hp", 0) > 0
            and (
                getattr(switch, "hp", 0) < getattr(switch, "max_hp", 0)
                or not _status_is_healthy(getattr(switch, "status", None))
            )
        ):
            player_switches.append(
                SearchSwitch(
                    switch.party_index,
                    switch.name,
                    switch.hp,
                    switch.max_hp,
                    switch_moves,
                    getattr(switch, "speed", None),
                    getattr(switch, "status", None),
                    escape_probability=getattr(switch, "escape_probability", None),
                    escape_guaranteed=bool(getattr(switch, "escape_guaranteed", False)),
                )
            )
    player_items = []
    capture_fallback_probability = (
        0.5
        if getattr(planner_context, "capture_target", False)
        and (
            getattr(planner_context, "battle_objective", "defeat") == "capture"
            or getattr(planner_context, "capture_available", False)
        )
        else 0.0
    )
    for item_index, item in enumerate(getattr(planner_context, "items", ())):
        if getattr(item, "quantity", 0) <= 0:
            continue
        is_capture_ball = bool(getattr(item, "is_capture_ball", False))
        if is_capture_ball and not (
            getattr(planner_context, "capture_target", False)
            and (
                getattr(planner_context, "battle_objective", "defeat") == "capture"
                or getattr(planner_context, "capture_available", False)
            )
        ):
            continue
        player_items.append(
            SearchItem(
                item_index,
                getattr(getattr(item, "item", None), "name", "item"),
                item.quantity,
                getattr(item, "heal_amount", 0),
                getattr(item, "full_heal", False),
                bool(getattr(item, "cures_all_status", False) or getattr(item, "status_healed", ())),
                is_capture_ball,
                float(
                    getattr(item, "capture_probability", None)
                    if getattr(item, "capture_probability", None) is not None
                    else getattr(planner_context, "capture_success_probability", None)
                    if getattr(planner_context, "capture_success_probability", None) is not None
                    else capture_fallback_probability
                ),
                frozenset(getattr(item, "status_healed", ()) or ()),
            )
        )
    escape_probability = float(getattr(planner_context, "escape_probability", 0.0) or 0.0)
    if escape_probability <= 0 and getattr(planner_context, "can_run", False):
        confidence = getattr(getattr(planner_context, "escape_confidence", None), "value", None)
        escape_probability = {"high": 1.0, "medium": 0.75}.get(confidence, 0.0)

    battle_objective = getattr(planner_context, "battle_objective", "defeat")
    if getattr(planner_context, "capture_target", False) and getattr(planner_context, "capture_available", False):
        battle_objective = "capture"

    state = SearchState(
        planner_context.active_hp,
        planner_context.active_max_hp,
        planner_context.opponent_hp,
        planner_context.opponent_max_hp or planner_context.opponent_hp,
        player_moves,
        opponent_moves,
        planner_context.active_speed,
        planner_context.opponent_speed,
        planner_context.active_status,
        planner_context.opponent_status,
        planner_context.active_party_index,
        tuple(player_switches),
        tuple(player_items),
        battle_objective,
        bool(getattr(planner_context, "capture_target", False)),
        int(getattr(planner_context, "experience_reward", 0) or 0),
        tuple(getattr(planner_context, "experience_values", ()) or ()),
        frozenset({planner_context.active_party_index}),
        tuple(getattr(planner_context, "active_stat_stages", ()) or ()),
        tuple(getattr(planner_context, "opponent_stat_stages", ()) or ()),
        bool(
            getattr(planner_context, "can_run", False)
            and getattr(planner_context, "is_wild", False)
            and not getattr(planner_context, "is_trainer", False)
        ),
        escape_probability,
        False,
        capture_attempted=bool(getattr(planner_context, "capture_attempted", False)),
        response_facts_complete=bool(
            getattr(planner_context, "opponent_response_facts_complete", True)
            and (opponent_moves or getattr(planner_context, "opponent_damage_max", None) is not None)
            and all(
                getattr(move, "response_facts_complete", True)
                for move in raw_opponent_moves
            )
            and not unsupported_opponent_moves
        ),
        campaign_owned=bool(getattr(planner_context, "campaign_owned", False)),
        switch_history=tuple(getattr(planner_context, "switch_history", ()) or ()),
        free_switch=bool(getattr(planner_context, "free_switch", False)),
        switch_streak=max(0, int(getattr(planner_context, "switch_streak", 0) or 0)),
        root_opponent_hp=planner_context.opponent_hp,
        unknown_response_ceiling=max(
            0,
            int(
                getattr(planner_context, "unsupported_response_ceiling", 0)
                or (planner_context.active_max_hp if unsupported_opponent_moves else 0)
            ),
        ),
        switch_cycle_pressure=max(
            0,
            int(getattr(planner_context, "switch_cycle_pressure", 0) or 0),
        ),
        escape_attempts=getattr(planner_context, "escape_attempts", None),
        escape_guaranteed=bool(getattr(planner_context, "escape_guaranteed", False)),
        player_bide_turns=max(0, int(getattr(planner_context, "player_bide_turns", 0) or 0)),
        player_bide_damage=max(0, int(getattr(planner_context, "player_bide_damage", 0) or 0)),
        opponent_bide_turns=max(0, int(getattr(planner_context, "opponent_bide_turns", 0) or 0)),
        opponent_bide_damage=max(0, int(getattr(planner_context, "opponent_bide_damage", 0) or 0)),
    )
    if config is None:
        config = SearchConfig()
    probability_evaluator = getattr(planner_context, "capture_probability_for_state", None)
    if (
        battle_objective == "capture"
        and getattr(planner_context, "capture_target", False)
        and callable(probability_evaluator)
        and config.capture_probability is None
    ):
        config = replace(config, capture_probability=probability_evaluator)
    if (state.experience_values or state.experience_reward) and config.experience_weight <= 0:
        # The presence of normalized recipient values opts the planner into
        # useful-EXP comparison. A reward without recipient headroom still
        # matters when deciding whether a wild battle is worth preserving.
        # The weight remains secondary to the large terminal objective value.
        config = replace(config, experience_weight=1.0)
    return Expectiminimax(config).evaluate(state)
